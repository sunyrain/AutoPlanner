"""Deterministically bind exact literature reactions to source PDF pages.

Codex/vision output is only a proposal.  This module can approve a proposed
edge only when an independent name-to-structure parser reconstructs the
product heading from the source text and every proposed reactant is found in
that product's experimental procedure as either a parsed compound label or a
database-resolved chemical name.  The resulting registry remains out of band
from model payloads and is replayed by the normal trusted-precedent verifier.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Mapping

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from cascade_planner.harness.reaction_step_verifier import (
    canonical_reaction_digest,
)
from cascade_planner.harness.deterministic_resolver_cache import (
    DeterministicResolverCache,
)
from cascade_planner.harness.stitched_route import (
    _materialized_source_evidence_valid,
)
from cascade_planner.harness.source_text_companion import (
    materialize_source_text_companion_pages,
    validate_source_text_companion_binding,
)
from cascade_planner.runtime.run_metrics import (
    current_run_metrics,
    run_metric_stage,
)


REGISTRY_SCHEMA = "trusted_literature_step_registry.v1"
AUDIT_SCHEMA = "deterministic_literature_registry_audit.v1"
PARSER_AUTHORITY_ID = "autoplanner.opsin_pubchem_source_text.v8"
DEFAULT_OPSIN_BASE_URL = "https://opsin.ch.cam.ac.uk/opsin"
DEFAULT_PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_MAX_HEADING_PARSE_ATTEMPTS_PER_EDGE = 16

StructureResolver = Callable[[str], str]
CandidateNameResolver = Callable[[str], list[str]]
PdfTextLoader = Callable[[Path], list[dict[str, Any]]]


def build_deterministic_literature_resolvers(
    *,
    opsin_base_url: str = DEFAULT_OPSIN_BASE_URL,
    pubchem_base_url: str = DEFAULT_PUBCHEM_BASE_URL,
    timeout_s: float = 30.0,
    persistent_cache: DeterministicResolverCache | None = None,
) -> tuple[StructureResolver, CandidateNameResolver]:
    """Build one run-scoped resolver pair with shared in-memory caches."""

    return (
        _opsin_resolver(
            base_url=opsin_base_url,
            pubchem_base_url=pubchem_base_url,
            timeout_s=timeout_s,
            persistent_cache=persistent_cache,
        ),
        _pubchem_name_resolver(
            base_url=pubchem_base_url,
            timeout_s=timeout_s,
            persistent_cache=persistent_cache,
        ),
    )


def compile_deterministic_literature_step_registry(
    steps: Iterable[dict[str, Any]],
    *,
    registry_path: str | Path,
    audit_path: str | Path | None = None,
    opsin_base_url: str = DEFAULT_OPSIN_BASE_URL,
    pubchem_base_url: str = DEFAULT_PUBCHEM_BASE_URL,
    timeout_s: float = 30.0,
    structure_resolver: StructureResolver | None = None,
    candidate_name_resolver: CandidateNameResolver | None = None,
    pdf_text_loader: PdfTextLoader | None = None,
) -> dict[str, Any]:
    """Compile approved bindings or fail closed with per-edge diagnostics."""

    registry_file = Path(registry_path).expanduser().resolve()
    audit_file = (
        Path(audit_path).expanduser().resolve()
        if audit_path
        else registry_file.with_name(
            f"{registry_file.stem}.audit.json"
        )
    )
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    resolve_structure = structure_resolver or _opsin_resolver(
        base_url=opsin_base_url,
        pubchem_base_url=pubchem_base_url,
        timeout_s=timeout_s,
    )
    resolve_candidate_names = candidate_name_resolver or _pubchem_name_resolver(
        base_url=pubchem_base_url,
        timeout_s=timeout_s,
    )
    load_pdf_text = pdf_text_loader or _load_pdf_page_text

    document_cache: dict[str, dict[str, Any]] = {}
    bindings_by_id: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for step_index, raw in enumerate(steps, start=1):
        step = dict(raw) if isinstance(raw, dict) else {}
        record = _compile_step_binding(
            step,
            step_index=step_index,
            document_cache=document_cache,
            resolve_structure=resolve_structure,
            resolve_candidate_names=resolve_candidate_names,
            load_pdf_text=load_pdf_text,
        )
        records.append(record)
        binding = record.get("binding")
        if isinstance(binding, dict) and record.get("accepted") is True:
            bindings_by_id[str(binding.get("binding_id") or "")] = binding

    newly_approved_bindings = list(bindings_by_id.values())
    prior_bindings = _prior_approved_bindings(registry_file)
    for binding in prior_bindings:
        bindings_by_id.setdefault(str(binding.get("binding_id") or ""), binding)
    bindings = sorted(
        bindings_by_id.values(),
        key=lambda row: (
            str(row.get("reaction_digest") or ""),
            str(row.get("source_ref") or ""),
            str(row.get("binding_id") or ""),
        ),
    )
    registry: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": "autoplanner-deterministic-" + _digest(bindings)[:16],
        "bindings": bindings,
        "policy": {
            "model_authored_visual_or_text_translation_is_advisory": True,
            "accepted_authorities": ["deterministic_structure_parser"],
            "source_text_product_reconstruction_required": True,
            "all_candidate_reactants_must_be_source_resolved": True,
            "current_host_source_artifact_replay_required": True,
            "hash_bound_text_companion_allowed_for_image_only_pdf": True,
        },
    }
    registry["content_sha256"] = _digest(registry)
    registry_file.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "accepted": bool(bindings),
        "authority": {
            "type": "deterministic_structure_parser",
            "id": PARSER_AUTHORITY_ID,
        },
        "registry_path": str(registry_file),
        "registry_sha256": _file_sha256(registry_file),
        "input_step_count": len(records),
        "approved_binding_count": len(newly_approved_bindings),
        "prior_binding_count": len(prior_bindings),
        "registry_binding_count": len(bindings),
        "rejected_step_count": sum(
            1 for row in records if row.get("accepted") is not True
        ),
        "records": records,
        "source_procedure_inventory": _source_procedure_inventory(
            document_cache
        ),
        "semantics": {
            "codex_candidate_cannot_self_sign": True,
            "opsin_reconstructs_source_heading": True,
            "pubchem_names_are_candidate_lookup_only": True,
            "source_pdf_page_and_image_are_digest_bound": True,
            "source_text_companion_is_replayed_when_used": True,
        },
    }
    audit["content_sha256"] = _digest(audit)
    audit_file.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _prior_approved_bindings(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or payload.get("schema_version") != REGISTRY_SCHEMA:
        return []
    out: list[dict[str, Any]] = []
    for raw in payload.get("bindings") or []:
        if not isinstance(raw, dict):
            continue
        binding = dict(raw)
        authority = dict(binding.get("authority") or {})
        if (
            binding.get("status") != "approved"
            or authority.get("type")
            not in {"human_curator", "deterministic_structure_parser"}
            or not str(authority.get("id") or "").strip()
            or not str(binding.get("binding_id") or "").strip()
            or not _is_sha256(binding.get("reaction_digest"))
            or not _is_sha256(binding.get("source_pdf_sha256"))
            or not _is_sha256(binding.get("image_sha256"))
        ):
            continue
        if (
            authority.get("type") == "deterministic_structure_parser"
            and authority.get("id") != PARSER_AUTHORITY_ID
        ):
            continue
        companion = binding.get("source_text_companion")
        if companion and not validate_source_text_companion_binding(
            companion,
            expected_source_ref=str(binding.get("source_ref") or ""),
        ):
            continue
        out.append(binding)
    return out


def _compile_step_binding(
    step: dict[str, Any],
    *,
    step_index: int,
    document_cache: dict[str, dict[str, Any]],
    resolve_structure: StructureResolver,
    resolve_candidate_names: CandidateNameResolver,
    load_pdf_text: PdfTextLoader,
) -> dict[str, Any]:
    product = _canonical_smiles(step.get("product_smiles"))
    reactants = _canonical_reactants(step.get("reactant_smiles"))
    source_candidate_reaction_digest = canonical_reaction_digest(
        product,
        reactants,
    )
    source_ref = str(step.get("source_ref") or "").strip().lower()
    evidence = [
        dict(item)
        for item in step.get("source_evidence") or []
        if isinstance(item, dict)
    ]
    reasons: list[str] = []
    if not product:
        reasons.append("product_smiles_invalid")
    if not reactants:
        reasons.append("reactant_smiles_invalid")
    if not source_ref:
        reasons.append("source_ref_missing")
    valid_evidence = [
        row for row in evidence if _materialized_source_evidence_valid(row)
    ]
    if not valid_evidence:
        reasons.append("materialized_source_evidence_not_replayable")
    if reasons:
        return _rejected_record(step, step_index=step_index, reasons=reasons)

    documents: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    source_text_companions = [
        dict(item)
        for item in step.get("source_text_companions") or []
        if isinstance(item, dict)
    ]
    for row in valid_evidence:
        pdf_path = Path(str(row.get("source_pdf_path") or "")).resolve()
        key = (
            f"{pdf_path}|{row.get('source_pdf_sha256')}|{source_ref}|"
            f"{_digest(source_text_companions)}"
        )
        if key in seen_documents:
            continue
        seen_documents.add(key)
        if key not in document_cache:
            document_cache[key] = _build_document_index(
                pdf_path,
                source_ref=source_ref,
                source_pdf_sha256=str(row.get("source_pdf_sha256") or ""),
                load_pdf_text=load_pdf_text,
                source_text_companions=source_text_companions,
            )
        documents.append(document_cache[key])

    product_parent = _parent_identity(product)
    if "." not in product and any(
        product_parent in _fragment_parent_identities(reactant)
        for reactant in reactants
    ):
        return _rejected_record(
            step,
            step_index=step_index,
            reasons=[
                "noncovalent_formulation_or_salt_state_transition_not_synthesis_edge"
            ],
        )

    product_matches: list[
        tuple[dict[str, Any], dict[str, Any], str]
    ] = []
    declared_product_label = _compound_label(step.get("product_name"))
    for document in documents:
        evidence_pages = {
            int(row.get("page_number") or 0)
            for row in valid_evidence
            if str(Path(str(row.get("source_pdf_path") or "")).resolve())
            == str(document.get("pdf_path") or "")
        }
        procedures = [
            item
            for item in document.get("procedures") or []
            if isinstance(item, dict) and item.get("declaration_only") is not True
        ]
        procedures.sort(
            key=lambda row: (
                -int(int(row.get("page_number") or 0) in evidence_pages),
                -_procedure_product_name_match_score(
                    row,
                    product_name=str(step.get("product_name") or ""),
                ),
                int(row.get("page_number") or 0),
            )
        )
        for procedure in procedures[:_MAX_HEADING_PARSE_ATTEMPTS_PER_EDGE]:
            _resolve_procedure_structure(
                procedure,
                resolve_structure=resolve_structure,
            )
            procedure_smiles = str(procedure.get("canonical_smiles") or "")
            if procedure_smiles == product:
                product_matches.append(
                    (document, procedure, "source_heading_exact_structure")
                )
                continue
            if (
                declared_product_label
                and _compound_label(procedure.get("label"))
                == declared_product_label
                and procedure_smiles
            ):
                # The model selects a source procedure by label, but the
                # independently parsed source heading supplies the chemistry.
                # This safely repairs missing/wrong visual stereochemistry
                # without granting authority to the proposed SMILES.
                product_matches.append(
                    (
                        document,
                        procedure,
                        "source_label_authoritative_structure_reconstruction",
                    )
                )
                continue
            if (
                procedure_smiles
                and _parent_identity(procedure_smiles)
                in _fragment_parent_identities(product)
                and _counterions_confirmed(
                    product,
                    " ".join(
                        [
                            str(procedure.get("name") or ""),
                            str(procedure.get("procedure") or ""),
                        ]
                    ),
                    resolve_candidate_names=resolve_candidate_names,
                    parent_smiles=procedure_smiles,
                )
            ):
                product_matches.append(
                    (
                        document,
                        procedure,
                        "source_heading_parent_plus_confirmed_formulation",
                    )
                )
    if not product_matches:
        return _rejected_record(
            step,
            step_index=step_index,
            reasons=["product_not_reconstructed_from_source_heading"],
        )

    if any(
        _parent_identity(str(procedure.get("canonical_smiles") or ""))
        in _fragment_parent_identities(reactant)
        for _, procedure, _ in product_matches
        for reactant in reactants
    ):
        return _rejected_record(
            step,
            step_index=step_index,
            reasons=[
                "noncovalent_formulation_or_salt_state_transition_not_synthesis_edge"
            ],
        )

    candidate_diagnostics: list[dict[str, Any]] = []
    # Visual extraction calls these ``reactant_labels`` while the exact-row
    # compiler intentionally renames them to ``reactant_names``.  Treat both
    # fields only as parser candidates: the name must still occur verbatim in
    # the product procedure and an independent OPSIN/PubChem parse must match
    # the proposed structure before it can authorize an edge.
    source_declared_reactant_labels = list(
        dict.fromkeys(
            str(item).strip()
            for field in ("reactant_labels", "reactant_names")
            for item in step.get(field) or []
            if str(item or "").strip()
        )
    )
    for document, procedure, product_match_mode in product_matches:
        document_procedures = [
            item
            for item in document.get("procedures") or []
            if isinstance(item, dict)
        ]
        procedure_text = str(procedure.get("procedure") or "")
        for candidate in document_procedures:
            label = str(candidate.get("label") or "")
            if label and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])",
                procedure_text,
                flags=re.IGNORECASE,
            ):
                _resolve_procedure_structure(
                    candidate,
                    resolve_structure=resolve_structure,
                )
        label_materialization = _materialize_source_label_reactants(
            source_declared_reactant_labels,
            procedure=procedure,
            procedures=document_procedures,
            expected_count=len(reactants),
            resolve_structure=resolve_structure,
        )
        if label_materialization:
            source_reactants = [
                str(row.get("source_formulation_smiles") or "")
                for row in label_materialization
            ]
            reactant_matches = label_materialization
            all_reactants_matched = True
        else:
            source_reactants = list(reactants)
            reactant_matches = []
            all_reactants_matched = True
            for reactant in reactants:
                match = _match_reactant_in_procedure(
                    reactant,
                    procedure=procedure,
                    procedures=document_procedures,
                    source_declared_labels=source_declared_reactant_labels,
                    resolve_structure=resolve_structure,
                    resolve_candidate_names=resolve_candidate_names,
                )
                reactant_matches.append(match)
                if match.get("accepted") is not True:
                    all_reactants_matched = False
        page_number = int(procedure.get("page_number") or 0)
        page_evidence = next(
            (
                row
                for row in valid_evidence
                if str(Path(str(row.get("source_pdf_path") or "")).resolve())
                == str(document.get("pdf_path") or "")
                and int(row.get("page_number") or 0) == page_number
            ),
            {},
        )
        diagnostic = {
            "product_label": str(procedure.get("label") or ""),
            "product_name": str(procedure.get("name") or ""),
            "page_number": page_number,
            "reactant_matches": reactant_matches,
            "all_reactants_matched": all_reactants_matched,
            "page_evidence_matched": bool(page_evidence),
            "product_match_mode": product_match_mode,
            "source_label_reactants_materialized": bool(label_materialization),
        }
        candidate_diagnostics.append(diagnostic)
        if not all_reactants_matched or not page_evidence:
            continue
        source_product = (
            _canonical_smiles(procedure.get("canonical_smiles") or "")
            if product_match_mode
            == "source_label_authoritative_structure_reconstruction"
            else product
        ) or product
        inventory_product = (
            _canonical_smiles(procedure.get("canonical_smiles") or "")
            or synthesis_projection_smiles(source_product)
        )
        atom_donor = _condition_atom_donor(
            step,
            product=inventory_product,
            reactants=source_reactants,
            source_text=" ".join(
                [
                    str(procedure.get("name") or ""),
                    str(procedure.get("procedure") or ""),
                ]
            ),
            resolve_structure=resolve_structure,
        )
        if atom_donor:
            donor_smiles = str(atom_donor["smiles"])
            source_reactants.append(donor_smiles)
            reactant_matches.append(
                {
                    "accepted": True,
                    "reactant_smiles": donor_smiles,
                    "synthesis_smiles": donor_smiles,
                    "match_mode": (
                        "source_condition_atom_donor_opsin_exact_structure"
                    ),
                    "matched_name_sha256": str(
                        atom_donor["matched_name_sha256"]
                    ),
                }
            )
        element_deficit = _element_inventory_deficit(
            inventory_product,
            source_reactants,
        )
        diagnostic["condition_atom_donor_added"] = bool(atom_donor)
        diagnostic["element_inventory_deficit"] = element_deficit
        if element_deficit:
            continue
        source_formulation_reaction_digest = canonical_reaction_digest(
            source_product,
            source_reactants,
        )
        procedure_formulation_product = _canonical_smiles(
            procedure.get("canonical_smiles") or ""
        )
        synthesis_product = synthesis_projection_smiles(
            procedure_formulation_product or source_product
        )
        synthesis_reactants = sorted(
            str(match.get("synthesis_smiles") or "")
            or synthesis_projection_smiles(reactant)
            for reactant, match in zip(
                source_reactants,
                reactant_matches,
                strict=True,
            )
        )
        reaction_digest = canonical_reaction_digest(
            synthesis_product,
            synthesis_reactants,
        )
        binding_core = {
            "reaction_digest": reaction_digest,
            "source_candidate_reaction_digest": (
                source_candidate_reaction_digest
            ),
            "source_formulation_reaction_digest": (
                source_formulation_reaction_digest
            ),
            "source_ref": source_ref,
            "document_id": str(page_evidence.get("document_id") or ""),
            "source_pdf_sha256": str(
                page_evidence.get("source_pdf_sha256") or ""
            ).lower(),
            "page_number": page_number,
            "image_sha256": str(page_evidence.get("image_sha256") or "").lower(),
            "synthesis_projection": {
                "schema_version": "literature_synthesis_projection.v1",
                "normalization_policy": (
                    "largest_covalent_fragment_and_counterion_neutralization"
                ),
                "normalization_applied": bool(
                    synthesis_product != product
                    or synthesis_reactants != sorted(reactants)
                ),
                "product_smiles": synthesis_product,
                "reactant_smiles": synthesis_reactants,
                "reaction_digest": reaction_digest,
            },
            "source_formulation": {
                "schema_version": "literature_source_formulation.v1",
                "product_smiles": source_product,
                "reactant_smiles": sorted(source_reactants),
                "reaction_digest": source_formulation_reaction_digest,
            },
        }
        companion_binding = dict(
            procedure.get("source_text_companion_binding") or {}
        )
        if companion_binding:
            binding_core["source_text_companion"] = companion_binding
        binding = {
            "binding_id": "det-parser:" + _digest(binding_core)[:24],
            **binding_core,
            "status": "approved",
            "authority": {
                "type": "deterministic_structure_parser",
                "id": PARSER_AUTHORITY_ID,
            },
            "parser_audit": {
                "product_label": str(procedure.get("label") or ""),
                "source_name_sha256": hashlib.sha256(
                    str(procedure.get("name") or "").encode("utf-8")
                ).hexdigest(),
                "procedure_text_sha256": hashlib.sha256(
                    str(procedure.get("procedure") or "").encode("utf-8")
                ).hexdigest(),
                "reactant_match_modes": [
                    str(row.get("match_mode") or "") for row in reactant_matches
                ],
                "product_match_mode": product_match_mode,
                "source_text_authority": (
                    "hash_bound_source_text_companion"
                    if companion_binding
                    else "embedded_pdf_text"
                ),
            },
        }
        return {
            "schema_version": "deterministic_literature_binding_record.v1",
            "accepted": True,
            "step_index": step_index,
            "step_id": str(step.get("step_id") or ""),
            "reaction_digest": reaction_digest,
            "source_candidate_reaction_digest": (
                source_candidate_reaction_digest
            ),
            "source_formulation_reaction_digest": (
                source_formulation_reaction_digest
            ),
            "source_ref": source_ref,
            "binding": binding,
            "candidate_diagnostics": candidate_diagnostics,
            "reasons": [],
        }
    rejection_reasons = [
        "candidate_reactants_not_all_source_resolved",
        "exact_product_page_evidence_not_matched",
    ]
    if any(
        row.get("element_inventory_deficit")
        for row in candidate_diagnostics
    ):
        rejection_reasons.append(
            "source_atom_donor_not_resolved_for_element_inventory"
        )
    return _rejected_record(
        step,
        step_index=step_index,
        reasons=rejection_reasons,
        candidate_diagnostics=candidate_diagnostics,
    )


def _build_document_index(
    pdf_path: Path,
    *,
    source_ref: str,
    source_pdf_sha256: str,
    load_pdf_text: PdfTextLoader,
    source_text_companions: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        not pdf_path.is_file()
        or _file_sha256(pdf_path) != source_pdf_sha256.lower()
    ):
        return {
            "accepted": False,
            "pdf_path": str(pdf_path),
            "source_ref": source_ref,
            "procedures": [],
            "reasons": ["source_pdf_digest_mismatch"],
        }
    reasons: list[str] = []
    try:
        pages = load_pdf_text(pdf_path)
    except (OSError, RuntimeError, ValueError):
        pages = []
        reasons.append("source_pdf_text_unavailable")
    companion_bindings: list[dict[str, Any]] = []
    for companion in source_text_companions:
        companion_pages, companion_binding, companion_reasons = (
            materialize_source_text_companion_pages(
                companion,
                source_ref=source_ref,
            )
        )
        if companion_reasons:
            reasons.extend(companion_reasons)
            continue
        pages.extend(companion_pages)
        companion_bindings.append(companion_binding)
    if source_text_companions and len(companion_bindings) != len(
        source_text_companions
    ):
        return {
            "accepted": False,
            "pdf_path": str(pdf_path),
            "source_ref": source_ref,
            "procedures": [],
            "reasons": sorted(set(reasons)),
        }
    procedures = _extract_labeled_procedures(pages)
    return {
        "accepted": bool(procedures),
        "pdf_path": str(pdf_path),
        "source_ref": source_ref,
        "source_pdf_sha256": source_pdf_sha256.lower(),
        "source_text_companion_bindings": companion_bindings,
        "procedure_count": len(procedures),
        "procedures": procedures,
        "reasons": (
            []
            if procedures
            else sorted(set([*reasons, "no_source_headings_extracted"]))
        ),
    }


def _resolve_procedure_structure(
    procedure: dict[str, Any],
    *,
    resolve_structure: StructureResolver,
) -> None:
    if procedure.get("structure_parse_attempted") is True:
        return
    procedure["structure_parse_attempted"] = True
    name = str(procedure.get("name") or "")
    try:
        smiles = resolve_structure(name)
    except (OSError, RuntimeError, ValueError):
        smiles = ""
    canonical = _canonical_smiles(smiles)
    if not canonical:
        procedure["structure_parse_accepted"] = False
        return
    procedure["structure_parse_accepted"] = True
    procedure["canonical_smiles"] = canonical
    procedure["structure_parser"] = "opsin_name_to_structure"
    procedure["parser_output_sha256"] = hashlib.sha256(
        str(smiles).encode("utf-8")
    ).hexdigest()


def _procedure_product_name_match_score(
    procedure: dict[str, Any],
    *,
    product_name: str,
) -> int:
    key = _name_key(product_name).strip()
    if not key:
        return 0
    label = _name_key(procedure.get("label") or "").strip()
    name = _name_key(procedure.get("name") or "").strip()
    if key == label or key == name:
        return 2
    return int(key in name or label in key)


def _extract_labeled_procedures(
    page_texts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks: list[str] = []
    ranges: list[tuple[int, int, int, dict[str, Any]]] = []
    cursor = 0
    for row in page_texts:
        if not isinstance(row, dict):
            continue
        page_number = int(row.get("page_number") or 0)
        text = str(row.get("text") or "")
        marker = f"\n\nAUTOPLANNER_PAGE_{page_number}\n\n"
        chunk = marker + text
        chunks.append(chunk)
        ranges.append(
            (
                cursor,
                cursor + len(chunk),
                page_number,
                dict(row.get("source_text_companion_binding") or {}),
            )
        )
        cursor += len(chunk)
    full_text = "".join(chunks)
    declaration = re.compile(
        r"\((?P<label>[TC]?\d+)(?:\s*,[^)]*)?\s*\)"
        r"[ \t]*(?:[.,:]|(?=\r?\n))",
        flags=re.IGNORECASE,
    )
    heading_candidates: list[dict[str, Any]] = []
    for match in declaration.finditer(full_text):
        separators = list(re.finditer(r"\n\s*\n", full_text[: match.start()]))
        heading_start = separators[-1].end() if separators else 0
        raw_name = full_text[heading_start : match.start()]
        name = _clean_source_name(raw_name)
        if (
            len(name) < 3
            or len(name) > 1000
            or not re.search(r"[A-Za-z]", name)
            or _document_metadata_heading(name)
        ):
            continue
        heading_candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "heading_start": heading_start,
                "label": str(match.group("label") or "").upper(),
                "name": name,
            }
        )
    # Patent PDFs rarely use SI-style ``name (T12)`` declarations.  Admit the
    # equally explicit, source-authored ``Example 4 Preparation of name``
    # heading, including common line wrapping before the example number and
    # within long chemical names.  The following numbered paragraph or the
    # first procedural sentence remains part of the procedure, not the name.
    patent_declaration = re.compile(
        r"(?im)^\s*Example\s*:?[ \t]*(?:\r?\n[ \t]*)?"
        r"(?P<label>\d+[A-Za-z]?)[ \t]*(?:\r?\n[ \t]*)?"
        r"(?:Preparation|Synthesis)[ \t]+of[ \t]+"
        r"(?P<name>.{3,1000}?)"
        r"(?=\r?\n[ \t]*(?:\d{3,5}\b|To\b|A[ \t]+total\b|"
        r"\d+(?:\.\d+)?[ \t]*(?:mL|ml|g)\b))",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    for match in patent_declaration.finditer(full_text):
        if any(
            int(row["start"]) <= match.start() < int(row["end"])
            for row in heading_candidates
        ):
            continue
        name = _clean_source_name(str(match.group("name") or ""))
        if (
            len(name) < 3
            or len(name) > 1000
            or not re.search(r"[A-Za-z]", name)
            or _document_metadata_heading(name)
        ):
            continue
        heading_candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "heading_start": match.start(),
                "label": str(match.group("label") or "").upper(),
                "name": name,
            }
        )
    # Process patents also commonly use a standalone, source-authored heading
    # such as ``Synthesis of Vismodegib (5)`` without an Example prefix.  The
    # generic ``name (T12)`` parser cannot safely recover this when patent page
    # headers remove blank-line boundaries, so bind the complete line directly.
    standalone_patent_declaration = re.compile(
        r"(?im)^\s*(?:Preparation|Synthesis)[ \t]+of[ \t]+"
        r"(?P<name>[^\r\n]{3,1000}?)\s*"
        r"\((?P<label>[TC]?\d+)\)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for match in standalone_patent_declaration.finditer(full_text):
        name = _clean_source_name(str(match.group("name") or ""))
        if (
            len(name) < 3
            or len(name) > 1000
            or not re.search(r"[A-Za-z]", name)
            or _document_metadata_heading(name)
        ):
            continue
        heading_candidates = [
            row
            for row in heading_candidates
            if not match.start() <= int(row["start"]) < match.end()
        ]
        heading_candidates.append(
            {
                "start": match.start(),
                "end": match.end(),
                "heading_start": match.start(),
                "label": str(match.group("label") or "").upper(),
                "name": name,
            }
        )
    heading_candidates.sort(
        key=lambda row: (int(row["start"]), int(row["end"]))
    )
    out: list[dict[str, Any]] = []
    for index, heading in enumerate(heading_candidates):
        procedure_end = (
            int(heading_candidates[index + 1]["heading_start"])
            if index + 1 < len(heading_candidates)
            else len(full_text)
        )
        procedure = _compact_source_text(
            full_text[int(heading["end"]) : procedure_end]
        )
        if not _procedure_like(procedure):
            continue
        page_range = next(
            (
                (page, companion_binding)
                for start, end, page, companion_binding in ranges
                if start <= int(heading["start"]) < end
            ),
            (0, {}),
        )
        page_number, companion_binding = page_range
        out.append(
            {
                "schema_version": "deterministic_source_procedure.v1",
                "label": str(heading["label"]),
                "name": str(heading["name"]),
                "page_number": page_number,
                "procedure": procedure,
                **(
                    {"source_text_companion_binding": companion_binding}
                    if companion_binding
                    else {}
                ),
            }
        )
    # Patent prose often defines numbered structures independently of the
    # experimental heading: ``compound (3) (chemical name)``.  These rows are
    # not themselves experimental procedures and can never authorize a product
    # edge, but they let a later hash-bound procedure resolve references such as
    # ``intermediate 3`` to an independently parsed structure.
    out.extend(_compound_label_definition_rows(full_text, ranges=ranges))
    out.sort(
        key=lambda row: (
            int(row.get("page_number") or 0),
            int(row.get("declaration_only") is True),
            str(row.get("label") or ""),
            str(row.get("name") or ""),
        )
    )
    return out


def _compound_label_definition_rows(
    full_text: str,
    *,
    ranges: list[tuple[int, int, int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    starts = re.compile(
        r"(?i)\bcompound\s*\((?P<label>[TC]?\d+)\)\s*\("
    )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for match in starts.finditer(full_text):
        opening = match.end() - 1
        depth = 0
        closing = -1
        for index in range(opening, min(len(full_text), opening + 1200)):
            character = full_text[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing < 0:
            continue
        name = _clean_source_name(full_text[opening + 1 : closing])
        label = str(match.group("label") or "").upper()
        if (
            len(name) < 3
            or len(name) > 1000
            or not re.search(r"[A-Za-z]", name)
            or _document_metadata_heading(name)
        ):
            continue
        page, companion_binding = next(
            (
                (page_number, binding)
                for start, end, page_number, binding in ranges
                if start <= match.start() < end
            ),
            (0, {}),
        )
        key = (label, _name_key(name).strip())
        rows[key] = {
            "schema_version": "deterministic_source_procedure.v1",
            "label": label,
            "name": name,
            "page_number": page,
            "procedure": "",
            "declaration_only": True,
            **(
                {"source_text_companion_binding": companion_binding}
                if companion_binding
                else {}
            ),
        }
    return [rows[key] for key in sorted(rows)]


def _source_procedure_inventory(
    document_cache: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose bounded, non-authoritative source observations for replanning."""

    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document in document_cache.values():
        source_ref = str(document.get("source_ref") or "")
        pdf_sha256 = str(document.get("source_pdf_sha256") or "")
        identity = (source_ref, pdf_sha256)
        if identity in seen:
            continue
        seen.add(identity)
        procedures = [
            {
                "label": str(row.get("label") or "")[:80],
                "name": " ".join(str(row.get("name") or "").split())[:1000],
                "page_number": int(row.get("page_number") or 0),
                "procedure_excerpt": " ".join(
                    str(row.get("procedure") or "").split()
                )[:800],
            }
            for row in document.get("procedures") or []
            if isinstance(row, Mapping)
        ][:64]
        documents.append(
            {
                "source_ref": source_ref,
                "source_pdf_sha256": pdf_sha256,
                "procedure_count": len(procedures),
                "procedures": procedures,
                "semantics": {
                    "discovery_only": True,
                    "grants_no_exact_reaction_evidence": True,
                },
            }
        )
    return documents[:8]


def _document_metadata_heading(value: str) -> bool:
    key = _name_key(value)
    return any(
        term in key
        for term in (
            " patent application publication ",
            " publication classification ",
            " foreign application priority data ",
            " united states patent ",
        )
    )


def _materialize_source_label_reactants(
    raw_labels: list[str],
    *,
    procedure: dict[str, Any],
    procedures: list[dict[str, Any]],
    expected_count: int,
    resolve_structure: StructureResolver,
) -> list[dict[str, Any]]:
    labels = [_compound_label(item) for item in raw_labels]
    if (
        expected_count <= 0
        or len(labels) != expected_count
        or any(not item for item in labels)
        or len(labels) != len(set(labels))
    ):
        return []
    procedure_text = str(procedure.get("procedure") or "")
    out: list[dict[str, Any]] = []
    for label in labels:
        if not re.search(
            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])",
            procedure_text,
            flags=re.IGNORECASE,
        ):
            return []
        candidates = [
            row
            for row in procedures
            if _compound_label(row.get("label")) == label
        ]
        structures: dict[str, str] = {}
        for candidate in candidates:
            _resolve_procedure_structure(
                candidate,
                resolve_structure=resolve_structure,
            )
            source_formulation = str(candidate.get("canonical_smiles") or "")
            synthesis_smiles = synthesis_projection_smiles(source_formulation)
            if source_formulation and synthesis_smiles:
                structures[source_formulation] = synthesis_smiles
        if len(structures) != 1:
            return []
        source_formulation, synthesis_smiles = next(iter(structures.items()))
        out.append(
            {
                "accepted": True,
                "reactant_smiles": synthesis_smiles,
                "source_formulation_smiles": source_formulation,
                "synthesis_smiles": synthesis_smiles,
                "match_mode": "source_label_authoritative_structure_reconstruction",
                "matched_label": label,
            }
        )
    return out


def _match_reactant_in_procedure(
    reactant: str,
    *,
    procedure: dict[str, Any],
    procedures: list[dict[str, Any]],
    source_declared_labels: list[str],
    resolve_structure: StructureResolver,
    resolve_candidate_names: CandidateNameResolver,
) -> dict[str, Any]:
    procedure_text = str(procedure.get("procedure") or "")
    normalized_procedure = _name_key(procedure_text)
    for candidate in procedures:
        candidate_smiles = str(candidate.get("canonical_smiles") or "")
        if not candidate_smiles:
            continue
        exact = candidate_smiles == reactant
        parent = _parent_identity(candidate_smiles) == _parent_identity(reactant)
        if not exact and not parent:
            continue
        label = str(candidate.get("label") or "")
        if not re.search(
            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])",
            procedure_text,
            flags=re.IGNORECASE,
        ):
            continue
        if parent and not exact and not _counterions_confirmed(
            reactant,
            procedure_text,
            resolve_candidate_names=resolve_candidate_names,
            parent_smiles=candidate_smiles,
        ):
            continue
        return {
            "accepted": True,
            "reactant_smiles": reactant,
            "match_mode": (
                "source_label_exact_structure"
                if exact
                else "source_label_parent_plus_counterion"
            ),
            "matched_label": label,
            "synthesis_smiles": (
                synthesis_projection_smiles(candidate_smiles)
            ),
        }
    declared_name_match = _match_source_declared_reactant_name(
        reactant,
        source_text=procedure_text,
        source_declared_labels=source_declared_labels,
        resolve_structure=resolve_structure,
        resolve_candidate_names=resolve_candidate_names,
    )
    if declared_name_match:
        return declared_name_match
    try:
        names = resolve_candidate_names(reactant)
    except (OSError, RuntimeError, ValueError):
        names = []
    matched_name = next(
        (
            name
            for name in names
            if len(_name_key(name)) >= 4
            and _name_key(name) in normalized_procedure
        ),
        "",
    )
    if matched_name:
        return {
            "accepted": True,
            "reactant_smiles": reactant,
            "match_mode": "pubchem_structure_name_in_source_procedure",
            "synthesis_smiles": synthesis_projection_smiles(reactant),
            "matched_name_sha256": hashlib.sha256(
                matched_name.encode("utf-8")
            ).hexdigest(),
        }
    return {
        "accepted": False,
        "reactant_smiles": reactant,
        "match_mode": "unresolved",
        "reasons": ["reactant_not_resolved_in_product_procedure"],
    }


def _counterions_confirmed(
    smiles: str,
    source_text: str,
    *,
    resolve_candidate_names: CandidateNameResolver,
    parent_smiles: str = "",
) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if len(fragments) <= 1:
        return True
    fragments = sorted(
        fragments,
        key=lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1),
        reverse=True,
    )
    parent_index = 0
    expected_parent = _parent_identity(parent_smiles)
    if expected_parent:
        for index, fragment in enumerate(fragments):
            fragment_smiles = Chem.MolToSmiles(
                fragment,
                canonical=True,
                isomericSmiles=True,
            )
            if _parent_identity(fragment_smiles) == expected_parent:
                parent_index = index
                break
    normalized = _name_key(source_text)
    aliases = {
        "[Cl-]": ("hcl", "hydrochloride", "hydrogen chloride"),
        "Cl": ("hcl", "hydrochloride", "hydrogen chloride"),
        "[Na+]": ("sodium", "sodium salt"),
        "[K+]": ("potassium", "potassium salt"),
        "COC(C)(C)C": (
            "mtbe",
            "tert-butyl methyl ether",
            "methyl tert-butyl ether",
        ),
        "CS(=O)(=O)O": (
            "methanesulfonic acid",
            "methanesulfonate",
            "mesylate",
        ),
    }
    for index, fragment in enumerate(fragments):
        if index == parent_index:
            continue
        canonical = Chem.MolToSmiles(
            fragment,
            canonical=True,
            isomericSmiles=True,
        )
        names = list(aliases.get(canonical) or ())
        try:
            names.extend(resolve_candidate_names(canonical))
        except (OSError, RuntimeError, ValueError):
            pass
        if not any(
            len(_name_key(name)) >= 3 and _name_key(name) in normalized
            for name in names
        ):
            return False
    return True


def _condition_atom_donor(
    step: dict[str, Any],
    *,
    product: str,
    reactants: list[str],
    source_text: str,
    resolve_structure: StructureResolver,
) -> dict[str, str]:
    """Resolve one source-declared reagent needed to conserve product atoms."""

    initial_deficit = _element_inventory_deficit(product, reactants)
    if not initial_deficit:
        return {}
    condition = dict(step.get("condition_candidate") or {})
    raw_names: list[str] = []
    for field in (
        "atom_contributing_reagent",
        "reagent",
        "acylating_agent",
        "coupling_partner",
    ):
        value = condition.get(field)
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            compact = " ".join(str(item or "").strip().split())
            if not compact:
                continue
            raw_names.extend(
                part
                for part in re.split(r"\s*;\s*|\s+\+\s+", compact)
                if part
            )
    normalized_source = _name_key(source_text)
    for name in dict.fromkeys(raw_names):
        name_key = _name_key(name)
        if len(name_key.strip()) < 4 or name_key not in normalized_source:
            continue
        try:
            donor = _canonical_smiles(resolve_structure(name))
        except (OSError, RuntimeError, ValueError):
            donor = ""
        if not donor or donor in reactants:
            continue
        if _element_inventory_deficit(product, [*reactants, donor]):
            continue
        return {
            "smiles": donor,
            "matched_name_sha256": hashlib.sha256(
                name.encode("utf-8")
            ).hexdigest(),
        }
    return {}


def _element_inventory_deficit(
    product: str,
    reactants: Iterable[str],
) -> dict[str, int]:
    product_counts = _element_inventory([product])
    reactant_counts = _element_inventory(reactants)
    periodic_table = Chem.GetPeriodicTable()
    return {
        periodic_table.GetElementSymbol(atomic_number): count
        - reactant_counts.get(atomic_number, 0)
        for atomic_number, count in sorted(product_counts.items())
        if count > reactant_counts.get(atomic_number, 0)
    }


def _element_inventory(values: Iterable[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for value in values:
        molecule = Chem.MolFromSmiles(str(value or ""))
        if molecule is None:
            continue
        counts.update(
            int(atom.GetAtomicNum())
            for atom in molecule.GetAtoms()
            if int(atom.GetAtomicNum()) > 1
        )
    return counts


def _match_source_declared_reactant_name(
    reactant: str,
    *,
    source_text: str,
    source_declared_labels: list[str],
    resolve_structure: StructureResolver,
    resolve_candidate_names: CandidateNameResolver,
) -> dict[str, Any]:
    normalized_source = _name_key(source_text)
    for raw_label in source_declared_labels:
        label = " ".join(str(raw_label or "").strip().split())
        label_key = _name_key(label).strip()
        if len(label_key) < 8 or f" {label_key} " not in normalized_source:
            continue
        parse_name = re.sub(
            r"\s*,?\s*(?:(?:hcl|hydrochloride)\s+salt|hydrochloride)\s*$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()
        try:
            parsed = _canonical_smiles(resolve_structure(parse_name))
        except (OSError, RuntimeError, ValueError):
            parsed = ""
        if not parsed or _parent_identity(parsed) != _parent_identity(reactant):
            continue
        exact = parsed == reactant
        if not exact and not _counterions_confirmed(
            reactant,
            f"{label} {source_text}",
            resolve_candidate_names=resolve_candidate_names,
            parent_smiles=parsed,
        ):
            continue
        return {
            "accepted": True,
            "reactant_smiles": reactant,
            "match_mode": (
                "source_declared_name_opsin_exact_structure"
                if exact
                else "source_declared_name_opsin_parent_plus_counterion"
            ),
            "matched_name_sha256": hashlib.sha256(
                parse_name.encode("utf-8")
            ).hexdigest(),
            "synthesis_smiles": synthesis_projection_smiles(parsed),
        }
    return {}


def _opsin_resolver(
    *,
    base_url: str,
    pubchem_base_url: str = DEFAULT_PUBCHEM_BASE_URL,
    timeout_s: float,
    persistent_cache: DeterministicResolverCache | None = None,
) -> StructureResolver:
    cache: dict[str, str] = {}

    def resolve(name: str) -> str:
        key = _compact_source_text(name)
        if key in cache:
            _increment_metric("resolver.structure.cache_hit")
            return cache[key]
        if persistent_cache is not None:
            persistent_hit, persistent_value = persistent_cache.get(
                "structure",
                key,
            )
            if persistent_hit:
                _increment_metric("resolver.structure.persistent_cache_hit")
                value = str(persistent_value or "")
                if not _canonical_smiles(value):
                    raise RuntimeError("source_name_structure_resolution_failed")
                cache[key] = value
                return value
            _increment_metric("resolver.structure.persistent_cache_miss")
        _increment_metric("resolver.structure.cache_miss")
        url = f"{base_url.rstrip('/')}/{urllib.parse.quote(key, safe='')}.smi"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AutoPlanner deterministic literature parser/1"},
        )
        value = ""
        try:
            with run_metric_stage("resolver.opsin.request", category="network"):
                with urllib.request.urlopen(
                    request,
                    timeout=max(1.0, timeout_s),
                ) as response:
                    value = response.read(1_000_000).decode("utf-8").strip()
        except (urllib.error.URLError, UnicodeDecodeError, TimeoutError):
            _increment_metric("resolver.opsin.failure")
            value = ""
        else:
            _increment_metric("resolver.opsin.success")
        if not _canonical_smiles(value):
            _increment_metric("resolver.opsin.fallback_to_pubchem")
            value = _pubchem_structure_from_name(
                key,
                base_url=pubchem_base_url,
                timeout_s=timeout_s,
            )
        if not _canonical_smiles(value):
            if persistent_cache is not None:
                persistent_cache.put("structure", key, "", success=False)
            raise RuntimeError("source_name_structure_resolution_failed")
        cache[key] = value
        if persistent_cache is not None:
            persistent_cache.put("structure", key, value, success=True)
        return value

    return resolve


def _pubchem_structure_from_name(
    name: str,
    *,
    base_url: str,
    timeout_s: float,
) -> str:
    encoded = urllib.parse.quote(str(name or "").strip(), safe="")
    url = (
        f"{base_url.rstrip('/')}/compound/name/{encoded}/property/SMILES/JSON"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AutoPlanner deterministic literature parser/1"},
    )
    try:
        with run_metric_stage(
            "resolver.pubchem_structure.request",
            category="network",
        ):
            with urllib.request.urlopen(
                request,
                timeout=max(1.0, timeout_s),
            ) as response:
                payload = json.loads(response.read(2_000_000).decode("utf-8"))
    except (
        urllib.error.URLError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TimeoutError,
    ):
        _increment_metric("resolver.pubchem_structure.failure")
        return ""
    _increment_metric("resolver.pubchem_structure.success")
    for row in ((payload.get("PropertyTable") or {}).get("Properties") or []):
        if isinstance(row, dict) and str(row.get("SMILES") or "").strip():
            return str(row.get("SMILES") or "").strip()
    return ""


def _pubchem_name_resolver(
    *,
    base_url: str,
    timeout_s: float,
    persistent_cache: DeterministicResolverCache | None = None,
) -> CandidateNameResolver:
    cache: dict[str, list[str]] = {}

    def resolve(smiles: str) -> list[str]:
        canonical = _canonical_smiles(smiles)
        if canonical in cache:
            _increment_metric("resolver.candidate_names.cache_hit")
            return list(cache[canonical])
        if persistent_cache is not None:
            persistent_hit, persistent_value = persistent_cache.get(
                "candidate_names",
                canonical,
            )
            if persistent_hit:
                _increment_metric("resolver.candidate_names.persistent_cache_hit")
                names = [str(item) for item in persistent_value or []]
                cache[canonical] = names
                return list(names)
            _increment_metric("resolver.candidate_names.persistent_cache_miss")
        _increment_metric("resolver.candidate_names.cache_miss")
        encoded = urllib.parse.quote(canonical, safe="")
        url = (
            f"{base_url.rstrip('/')}/compound/smiles/{encoded}/"
            "property/Title,IUPACName/JSON"
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AutoPlanner deterministic literature parser/1"},
        )
        try:
            with run_metric_stage(
                "resolver.pubchem_names.request",
                category="network",
            ):
                with urllib.request.urlopen(
                    request,
                    timeout=max(1.0, timeout_s),
                ) as response:
                    payload = json.loads(
                        response.read(2_000_000).decode("utf-8")
                    )
        except (
            urllib.error.URLError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TimeoutError,
        ):
            _increment_metric("resolver.pubchem_names.failure")
            cache[canonical] = []
            if persistent_cache is not None:
                persistent_cache.put(
                    "candidate_names",
                    canonical,
                    [],
                    success=False,
                )
            return []
        _increment_metric("resolver.pubchem_names.success")
        properties = ((payload.get("PropertyTable") or {}).get("Properties") or [])
        names: list[str] = []
        for row in properties:
            if not isinstance(row, dict):
                continue
            for field in ("Title", "IUPACName"):
                value = str(row.get(field) or "").strip()
                if value and value not in names:
                    names.append(value)
        cache[canonical] = names
        if persistent_cache is not None:
            persistent_cache.put(
                "candidate_names",
                canonical,
                names,
                success=True,
            )
        return list(names)

    return resolve


def _increment_metric(name: str, value: int = 1) -> None:
    metrics = current_run_metrics()
    if metrics is not None:
        metrics.increment(name, value)


def _load_pdf_page_text(path: Path) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pymupdf_unavailable") from exc
    rows: list[dict[str, Any]] = []
    document = fitz.open(str(path))
    try:
        for index in range(len(document)):
            rows.append(
                {
                    "page_number": index + 1,
                    "text": document[index].get_text("text") or "",
                }
            )
    finally:
        document.close()
    return rows


def _clean_source_name(value: str) -> str:
    text = re.sub(r"AUTOPLANNER_PAGE_\d+", " ", str(value or ""))
    text = re.sub(r"-\s*\n\s*", "-", text)
    text = re.sub(r"(?<=\d)\s*\n\s*(?=(?:yl|amine|amide)\b)", "-", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"^\s*\d+\s+", "", text)
    text = re.sub(
        r"^\s*(?:step\s+\d+\s*[.:]\s*)?"
        r"(?:procedure\s+for\s+(?:the\s+)?)?"
        r"(?:(?:streamlined|multicomponent|general)\s+)*"
        r"synthesis\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*\([^)]*(?:solvate|crystal\s+form|salt)[^)]*\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" .,:;\t\r\n")


def _compact_source_text(value: str) -> str:
    text = re.sub(r"AUTOPLANNER_PAGE_\d+", " ", str(value or ""))
    text = re.sub(r"-\s*\n\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _procedure_like(value: str) -> bool:
    key = _name_key(value)
    return any(
        term in key
        for term in (
            " was added ",
            " was stirred ",
            " reaction mixture ",
            " afforded ",
            " provided ",
            " was treated ",
        )
    )


def _name_key(value: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip() + " "


def _compound_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if re.fullmatch(r"[TC]?\d+", text) else ""


def _canonical_reactants(value: Any) -> list[str]:
    if isinstance(value, str):
        rows = [value]
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        return []
    out: list[str] = []
    for row in rows:
        canonical = _canonical_smiles(row)
        if not canonical:
            return []
        out.append(canonical)
    return sorted(out)


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _largest_fragment(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        return ""
    largest = max(
        fragments,
        key=lambda mol: (
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1),
            mol.GetNumAtoms(),
        ),
    )
    return Chem.MolToSmiles(largest, canonical=True, isomericSmiles=True)


def _parent_identity(smiles: str) -> str:
    parent = _largest_fragment(smiles)
    molecule = Chem.MolFromSmiles(parent)
    if molecule is None:
        return ""
    try:
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    except (RuntimeError, ValueError):
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def synthesis_projection_smiles(smiles: str) -> str:
    """Project a salt/solvate record onto its covalent synthesis parent."""

    canonical = _canonical_smiles(smiles)
    if not canonical:
        return ""
    if "." not in canonical:
        return canonical
    return _parent_identity(canonical)


def _fragment_parent_identities(smiles: str) -> set[str]:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return set()
    identities: set[str] = set()
    for fragment in Chem.GetMolFrags(
        molecule,
        asMols=True,
        sanitizeFrags=True,
    ):
        try:
            neutral = rdMolStandardize.Uncharger().uncharge(fragment)
        except (RuntimeError, ValueError):
            continue
        identity = Chem.MolToSmiles(
            neutral,
            canonical=True,
            isomericSmiles=True,
        )
        if identity:
            identities.add(identity)
    return identities


def _rejected_record(
    step: dict[str, Any],
    *,
    step_index: int,
    reasons: list[str],
    candidate_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "deterministic_literature_binding_record.v1",
        "accepted": False,
        "step_index": step_index,
        "step_id": str(step.get("step_id") or ""),
        "source_ref": str(step.get("source_ref") or "").strip().lower(),
        "candidate_diagnostics": list(candidate_diagnostics or []),
        "reasons": sorted(set(reasons)),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()
