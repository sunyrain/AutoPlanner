"""Compile hash-bound source procedures into bounded V4 route proposals.

The output is intentionally L0: source text can suggest a complete DAG, but
only the canonical materialization and reaction workers may admit and validate
its hyperedges.  Exact-source authority is compiled separately by the trusted
literature registry after the same structures are replayed.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import itertools
import json
import re
from typing import Any

from rdkit import Chem

from cascade_planner.harness.deterministic_literature_registry import (
    StructureResolver,
    source_amount_reagent_names,
    synthesis_projection_smiles,
)
from cascade_planner.harness.source_condition_extraction import (
    extract_source_conditions,
)
from cascade_planner.harness.source_form_bridge import (
    build_lactone_form_bridge_proposals,
)
from cascade_planner.harness.source_narrative import (
    narrative_product_name_candidates,
    narrative_reactant_name_candidates,
    source_name_resolution_candidates,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


SOURCE_ROUTE_OBSERVATION_SCHEMA = "deterministic_source_route_observation.v1"
SOURCE_ROUTE_PROPOSAL_SCHEMA = "deterministic_source_route_proposal.v1"

_SOLVENT_OR_WORKUP_TERMS = {
    "acetone",
    "brine",
    "dichloromethane",
    "diethyl ether",
    "dimethylformamide",
    "ethyl acetate",
    "ethanol",
    "hexane",
    "hexanes",
    "isopropanol",
    "methanol",
    "tetrahydrofuran",
    "toluene",
    "water",
}
_CATALYST_OR_BASE_TERMS = {
    "butyllithium",
    "celite",
    "hcl in dioxane",
    "hydrochloric acid",
    "iron powder",
    "lithium hydroxide",
    "magnesium sulfate",
    "n,n-diisopropylamine",
    "palladium on carbon",
    "palladium",
    "potassium acetate",
    "rhodium",
    "sodium bicarbonate",
    "sodium hydride",
    "sodium sulfate",
    "tetrabutylammonium chloride",
    "triethylamine",
}
_NON_ATOM_DONOR_TERMS = {
    "escherichia coli",
    "e. coli",
    "hepes",
    "hplc",
    "iptg",
    "lovd",
}


def compile_deterministic_source_route_observation(
    document: Mapping[str, Any],
    *,
    structure_resolver: StructureResolver,
    source_evidence: Iterable[Mapping[str, Any]] = (),
    anchor_smiles: Iterable[str] = (),
    max_procedures: int = 64,
    max_ingredient_resolutions: int = 64,
    max_precursors_per_step: int = 4,
) -> dict[str, Any]:
    """Translate one parsed source document into a target-connected DAG."""

    if not 1 <= max_procedures <= 128:
        raise ValueError("source_route_procedure_limit_invalid")
    if not 1 <= max_ingredient_resolutions <= 512:
        raise ValueError("source_route_resolver_limit_invalid")
    if not 1 <= max_precursors_per_step <= 6:
        raise ValueError("source_route_precursor_limit_invalid")

    source_ref = str(document.get("source_ref") or "")
    artifact_sha256 = str(
        document.get("source_artifact_sha256")
        or document.get("source_pdf_sha256")
        or ""
    )
    evidence_by_page = {
        int(row.get("page_number") or 0): dict(row)
        for row in source_evidence
        if isinstance(row, Mapping) and int(row.get("page_number") or 0) > 0
    }
    raw_procedures = [
        dict(row)
        for row in document.get("procedures") or []
        if isinstance(row, Mapping)
        and row.get("declaration_only") is not True
    ][:max_procedures]
    resolver_attempts = 0
    resolver_cache: dict[str, str] = {}
    source_aliases = {
        str(key).casefold(): str(value)
        for key, value in dict(document.get("source_name_aliases") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    procedures: list[dict[str, Any]] = []
    for row in raw_procedures:
        if row.get("structure_parse_accepted") is True and _canonical(
            row.get("canonical_smiles")
        ):
            procedures.append(row)
            continue
        recovered = _recover_product_from_narrative_heading(
            row,
            structure_resolver=structure_resolver,
        ) or _recover_product_from_downstream_source_name(
            row,
            procedures=raw_procedures,
            structure_resolver=structure_resolver,
        )
        if recovered:
            resolver_attempts += 1
            procedures.append(recovered)
    product_rows: list[dict[str, Any]] = []
    for row in procedures:
        source_product = _canonical(row.get("canonical_smiles"))
        product = synthesis_projection_smiles(source_product)
        if not product:
            continue
        product_rows.append(
            {
                **row,
                "source_product_smiles": source_product,
                "product_smiles": product,
                "product_parent": _parent_identity(product),
                "name_key": _name_key(row.get("name")),
            }
        )

    diagnostics: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for procedure in product_rows:
        procedure_text = str(procedure.get("procedure") or "")
        procedure_key = _name_key(procedure_text)
        concentration_names = _source_concentration_reagent_names(procedure_text)
        concentration_keys = {
            _name_key(_clean_ingredient_name(value)).strip()
            for value in concentration_names
        }
        heading_reactant_names = _narrative_heading_reactant_names(
            procedure.get("narrative_context") or procedure.get("name")
        )
        heading_reactant_keys = {
            _name_key(_clean_ingredient_name(value)).strip()
            for value in heading_reactant_names
        }
        ingredient_names = [
            *source_amount_reagent_names(procedure_text),
            *concentration_names,
            *heading_reactant_names,
        ]
        ingredients: list[dict[str, Any]] = []
        for raw_name in ingredient_names:
            name = _clean_ingredient_name(raw_name)
            if not name:
                continue
            key = _name_key(name).strip()
            if any(term == key or term in key for term in _NON_ATOM_DONOR_TERMS):
                continue
            known = _known_product_ingredient(
                name,
                product_rows=product_rows,
                current=procedure,
            )
            if known:
                ingredients.append(known)
                continue
            name_role = _ingredient_role_from_name(name)
            if (
                name_role != "candidate_reactant"
                and key not in concentration_keys
                and key not in heading_reactant_keys
            ):
                # Conditions retain every source-authored name.  Nonstructural
                # roles do not need a network name-to-structure request merely
                # to be displayed as solvent/catalyst metadata.
                continue
            if key not in resolver_cache and resolver_attempts < max_ingredient_resolutions:
                resolver_attempts += 1
                resolver_cache[key] = _resolve_name(
                    name,
                    structure_resolver=structure_resolver,
                    source_aliases=source_aliases,
                )
            smiles = resolver_cache.get(key, "")
            if not smiles:
                continue
            synthesis_smiles = synthesis_projection_smiles(smiles)
            if not synthesis_smiles or synthesis_smiles == procedure["product_smiles"]:
                continue
            ingredients.append(
                {
                    "name": name,
                    "smiles": synthesis_smiles,
                    "source_formulation_smiles": smiles,
                    "parent": _parent_identity(synthesis_smiles),
                    "name_key": _name_key(name),
                    "role": (
                        "candidate_reactant"
                        if key in concentration_keys or key in heading_reactant_keys
                        else _ingredient_role(name, synthesis_smiles)
                    ),
                }
            )

        # A previous source product is stronger than fuzzy role inference.  It
        # is included whenever its source-authored name occurs in this exact
        # procedure, even if a line-wrap typo prevented the ingredient parser
        # from resolving the same phrase independently.
        for other in product_rows:
            if other is procedure or not other["name_key"].strip():
                continue
            if other["name_key"].strip() not in procedure_key:
                continue
            ingredients.append(
                {
                    "name": str(other.get("name") or ""),
                    "smiles": other["product_smiles"],
                    "source_formulation_smiles": other["source_product_smiles"],
                    "parent": other["product_parent"],
                    "name_key": other["name_key"],
                    "role": "source_route_intermediate",
                }
            )
        ingredients = _dedupe_ingredients(ingredients)
        known_parents = {row["product_parent"] for row in product_rows}
        for ingredient in ingredients:
            if ingredient["parent"] in known_parents:
                ingredient["role"] = "source_route_intermediate"

        selection = _select_precursors(
            procedure["product_smiles"],
            ingredients,
            max_precursors=max_precursors_per_step,
        )
        if not selection["accepted"]:
            diagnostics.append(
                {
                    "label": str(procedure.get("label") or ""),
                    "product_name": str(procedure.get("name") or ""),
                    "status": "rejected",
                    "reasons": selection["reasons"],
                    "resolved_ingredient_count": len(ingredients),
                }
            )
            continue
        selected = list(selection["selected"])
        page_number = int(procedure.get("page_number") or 0)
        page_evidence = evidence_by_page.get(page_number, {})
        evidence_refs = [
            value
            for value in (
                f"pdf_sha256:{artifact_sha256}" if artifact_sha256 else "",
                (
                    f"image_sha256:{str(page_evidence.get('image_sha256') or '')}"
                    if page_evidence.get("image_sha256")
                    else ""
                ),
            )
            if value
        ]
        identity = {
            "source_ref": source_ref,
            "artifact_sha256": artifact_sha256,
            "label": str(procedure.get("label") or ""),
            "product_smiles": procedure["product_smiles"],
            "precursor_smiles": sorted(row["smiles"] for row in selected),
        }
        proposal_id = "source-route:" + _digest(identity)[:24]
        conditions = extract_source_conditions(
            procedure_text,
            source_amount_names=ingredient_names,
        )
        proposals.append(
            {
                "schema_version": SOURCE_ROUTE_PROPOSAL_SCHEMA,
                "proposal_id": proposal_id,
                "step_id": proposal_id,
                "source_ref": source_ref,
                "source_artifact_sha256": artifact_sha256,
                "source_location": {
                    "kind": str(document.get("source_location_kind") or "pdf_page"),
                    "page_number": page_number,
                    "label": str(procedure.get("label") or ""),
                },
                "evidence_refs": evidence_refs,
                "product_name": str(procedure.get("name") or ""),
                "product_smiles": procedure["product_smiles"],
                "source_product_smiles": procedure["source_product_smiles"],
                "product_structure_recovery_mode": str(
                    procedure.get("structure_recovery_mode") or "source_heading_opsin"
                ),
                "precursor_smiles": sorted(row["smiles"] for row in selected),
                "reactant_smiles": sorted(row["smiles"] for row in selected),
                "reactant_names": [str(row["name"]) for row in selected],
                "reagent_smiles": sorted(
                    {
                        row["smiles"]
                        for row in ingredients
                        if row not in selected and row["role"] != "solvent_or_workup"
                    }
                ),
                "condition_candidate": conditions,
                "origin_kind": "literature_source_route",
                "origin_ref": source_ref,
                "transformation_hypothesis": (
                    "hash-bound source procedure translated into a host-validated hyperedge"
                ),
                "admission_audit": selection["audit"],
                "semantics": {
                    "proposal_only": True,
                    "source_text_grants_no_reaction_validation": True,
                    "deterministic_registry_replay_required_for_exact_proof": True,
                },
            }
        )
        diagnostics.append(
            {
                "label": str(procedure.get("label") or ""),
                "product_name": str(procedure.get("name") or ""),
                "status": "proposed",
                "resolved_ingredient_count": len(ingredients),
                "selected_precursor_count": len(selected),
                "selected_roles": [str(row["role"]) for row in selected],
                "element_deficit": selection["element_deficit"],
            }
        )

    anchor_values = tuple(anchor_smiles)
    anchors = {
        parent
        for value in anchor_values
        if (parent := _parent_identity(_canonical(value)))
    }
    proposals.extend(
        build_lactone_form_bridge_proposals(
            proposals,
            anchor_smiles=anchor_values,
        )
    )
    connected = _target_connected_proposals(proposals, anchors=anchors)
    route_key = "source-route-family:" + _digest(
        {
            "source_ref": source_ref,
            "artifact_sha256": artifact_sha256,
            "anchors": sorted(anchors),
        }
    )[:24]
    for proposal in connected:
        proposal["route_family_id"] = route_key
    observation: dict[str, Any] = {
        "schema_version": SOURCE_ROUTE_OBSERVATION_SCHEMA,
        "source_ref": source_ref,
        "source_artifact_sha256": artifact_sha256,
        "route_family": {
            "route_family_id": route_key,
            "family_key": route_key,
            "strategy": "target-connected route DAG extracted from a hash-bound source",
            "selected": True,
        },
        "proposal_count": len(connected),
        "proposals": connected,
        "diagnostics": diagnostics[:max_procedures],
        "resolver_attempt_count": resolver_attempts,
        "resolved_procedure_count": len(product_rows),
        "unconnected_proposal_count": max(0, len(proposals) - len(connected)),
        "semantics": {
            "source_route_is_dag_not_linear_chain": True,
            "target_anchor_required": True,
            "proposals_grant_no_exact_or_reaction_proof": True,
            "host_materialization_and_mapping_required": True,
        },
    }
    observation["content_sha256"] = _digest(observation)
    return observation


def _select_precursors(
    product: str,
    ingredients: list[dict[str, Any]],
    *,
    max_precursors: int,
) -> dict[str, Any]:
    required = [
        row for row in ingredients if row["role"] == "source_route_intermediate"
    ]
    optional = [
        row
        for row in ingredients
        if row not in required and row["role"] != "solvent_or_workup"
    ]
    required = _dedupe_ingredients(required)
    optional = _dedupe_ingredients(optional)
    if len(required) > max_precursors:
        return {
            "accepted": False,
            "reasons": ["source_route_required_precursor_limit_exceeded"],
        }
    candidates: list[tuple[tuple[int, int, int, int], list[dict[str, Any]], dict[str, Any]]] = []
    min_optional = 0 if required else 1
    max_optional = min(len(optional), max_precursors - len(required))
    for count in range(min_optional, max_optional + 1):
        for combination in itertools.combinations(optional, count):
            selected = _dedupe_ingredients([*required, *combination])
            if not selected or len(selected) > max_precursors:
                continue
            audit = audit_retrosynthetic_candidate(
                product,
                [row["smiles"] for row in selected],
            )
            if audit.get("accepted") is not True:
                continue
            deficit = _element_deficit(product, [row["smiles"] for row in selected])
            surplus = _element_surplus(product, [row["smiles"] for row in selected])
            catalyst_count = sum(
                row["role"] == "catalyst_or_base" for row in selected
            )
            score = (
                sum(deficit.values()),
                catalyst_count,
                sum(surplus.values()),
                len(selected),
            )
            candidates.append((score, selected, audit))
    if not candidates:
        return {
            "accepted": False,
            "reasons": ["no_source_ingredient_subset_passed_admission"],
        }
    candidates.sort(key=lambda row: (row[0], _digest(row[2])))
    _, selected, audit = candidates[0]
    return {
        "accepted": True,
        "selected": selected,
        "audit": audit,
        "element_deficit": _element_deficit(
            product, [row["smiles"] for row in selected]
        ),
        "reasons": [],
    }


def _target_connected_proposals(
    proposals: list[dict[str, Any]],
    *,
    anchors: set[str],
) -> list[dict[str, Any]]:
    if not anchors:
        return []
    by_product: dict[str, list[dict[str, Any]]] = {}
    for proposal in proposals:
        parent = _parent_identity(str(proposal.get("product_smiles") or ""))
        if parent:
            by_product.setdefault(parent, []).append(proposal)
    pending = list(sorted(anchors))
    seen_products: set[str] = set()
    selected: dict[str, dict[str, Any]] = {}
    while pending:
        product = pending.pop(0)
        if product in seen_products:
            continue
        seen_products.add(product)
        for proposal in by_product.get(product, []):
            selected[str(proposal["proposal_id"])] = proposal
            for precursor in proposal.get("precursor_smiles") or []:
                parent = _parent_identity(str(precursor))
                if parent and parent in by_product and parent not in seen_products:
                    pending.append(parent)
    return sorted(
        selected.values(),
        key=lambda row: (
            int(dict(row.get("source_location") or {}).get("page_number") or 0),
            str(row.get("proposal_id") or ""),
        ),
    )


def _resolve_name(
    name: str,
    *,
    structure_resolver: StructureResolver,
    source_aliases: Mapping[str, str] | None = None,
) -> str:
    attempts = source_name_resolution_candidates(name, source_aliases)
    neutral = re.sub(
        r"\s+(?:hydrochloride\s+salt|hydrochloride)\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    if neutral and neutral != name:
        attempts.append(neutral)
    sodium_salt = re.sub(
        r"\s+sodium\s+salt\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    if sodium_salt and sodium_salt != name:
        attempts.extend(
            [
                (
                    sodium_salt
                    if sodium_salt.casefold().endswith(" acid")
                    else f"{sodium_salt} acid"
                ),
                sodium_salt,
            ]
        )
    for candidate in attempts:
        try:
            resolved = _canonical(structure_resolver(candidate))
        except (OSError, RuntimeError, ValueError):
            resolved = ""
        if resolved:
            return resolved
    return ""


def _recover_product_from_narrative_heading(
    procedure: Mapping[str, Any],
    *,
    structure_resolver: StructureResolver,
) -> dict[str, Any]:
    """Recover the named product from a source-authored narrative heading."""

    candidates = narrative_product_name_candidates(
        procedure.get("narrative_context") or procedure.get("name")
    )
    for name in candidates:
        if not 3 <= len(name) <= 180 or not re.search(r"[A-Za-z]", name):
            continue
        smiles = _resolve_name(name, structure_resolver=structure_resolver)
        if not smiles:
            continue
        return {
            **dict(procedure),
            "structure_parse_accepted": True,
            "canonical_smiles": smiles,
            "structure_recovery_mode": "narrative_product_name_advisory",
            "structure_recovery_name_sha256": hashlib.sha256(
                name.encode("utf-8")
            ).hexdigest(),
        }
    return {}


def _narrative_heading_reactant_names(value: Any) -> list[str]:
    """Recover only reactants explicitly named by a narrative source heading.

    This is intentionally narrower than general NLP: the names must occur in
    a source-authored ``product from A and B`` or ``conversion of A to B``
    construction.  They remain L0 candidates and still require deterministic
    structure resolution plus normal host admission.
    """

    names: list[str] = []
    for raw_name in narrative_reactant_name_candidates(value):
        name = _clean_ingredient_name(raw_name)
        if (
            2 <= len(name) <= 180
            and re.search(r"[A-Za-z]", name)
            and name.casefold() not in {item.casefold() for item in names}
        ):
            names.append(name)
    return names[:6]


def _source_concentration_reagent_names(source_text: str) -> list[str]:
    """Extract source names attached to explicit molar concentrations."""

    text = " ".join(str(source_text or "").split())
    patterns = (
        re.compile(
            r"(?i)(?P<name>[A-Za-z][A-Za-z0-9'’\- ]{2,120}?)"
            r"\s*\(\s*(?:final\s+concentration\s*)?[~∼]?[0-9.]+\s*"
            r"(?:nM|μM|uM|mM|M)\s*\)"
        ),
        re.compile(
            r"(?i)(?P<name>[A-Za-z][A-Za-z0-9'’\- ]{2,120}?)"
            r"\s+was\s+added\s+to\s+(?:a\s+)?final\s+concentrat(?:ion|e)\b"
        ),
    )
    names: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = str(match.group("name") or "").strip(" ,;.")
            name = re.split(r"(?i)[.;]|\b(?:and|then)\b", name)[-1]
            name = re.sub(r"(?i)^(?:the\s+)?(?:neat|pure)\s+", "", name)
            salt = re.match(r"(?i)^(?:the\s+)?sodium\s+salt\s+form\s+of\s+(.+)$", name)
            if salt:
                name = f"{salt.group(1).strip()} sodium salt"
            name = " ".join(name.split()).strip(" ,;.")
            if (
                3 <= len(name) <= 120
                and re.search(r"[A-Za-z]", name)
                and name.casefold() not in {value.casefold() for value in names}
            ):
                names.append(name)
    return names[:24]


def _recover_product_from_downstream_source_name(
    procedure: Mapping[str, Any],
    *,
    procedures: Iterable[Mapping[str, Any]],
    structure_resolver: StructureResolver,
) -> dict[str, Any]:
    """Recover an L0 product candidate from a near-identical later mention.

    The source heading remains untrusted for exact proof.  This only prevents a
    PDF text-layer omission such as a missing ``-one`` from disconnecting the
    search graph when a later experimental paragraph spells out the material.
    """

    heading = str(procedure.get("name") or "")
    heading_tokens = _significant_name_tokens(heading)
    if len(heading_tokens) < 4:
        return {}
    candidates: list[tuple[float, str]] = []
    for other in procedures:
        if other is procedure:
            continue
        for raw_name in source_amount_reagent_names(str(other.get("procedure") or "")):
            name = _clean_ingredient_name(raw_name)
            tokens = _significant_name_tokens(name)
            if len(tokens) < 4:
                continue
            overlap = len(heading_tokens & tokens)
            score = overlap / max(len(heading_tokens), len(tokens))
            if score >= 0.60:
                candidates.append((score, name))
    for _, name in sorted(candidates, key=lambda row: (-row[0], row[1])):
        smiles = _resolve_name(name, structure_resolver=structure_resolver)
        if not smiles:
            continue
        return {
            **dict(procedure),
            "structure_parse_accepted": True,
            "canonical_smiles": smiles,
            "structure_recovery_mode": (
                "downstream_hash_bound_source_name_advisory"
            ),
            "structure_recovery_name_sha256": hashlib.sha256(
                name.encode("utf-8")
            ).hexdigest(),
        }
    return {}


def _significant_name_tokens(value: Any) -> set[str]:
    ignored = {"acid", "hydrochloride", "methyl", "salt", "tert", "the"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) > 1 and token not in ignored
    }


def _clean_ingredient_name(value: Any) -> str:
    name = " ".join(str(value or "").replace("_", " ").split()).strip(" ,;.")
    name = re.sub(r"\bpropanoa(?=\s+hydrochloride\b)", "propanoate", name, flags=re.I)
    name = re.sub(
        r"N,N[^A-Za-z0-9\s]*-?disuccinimidyl",
        "N,N'-disuccinimidyl",
        name,
        flags=re.I,
    )
    name = re.sub(r"^(?:more|warm|hot|neat|pure)\s+", "", name, flags=re.I)
    name = re.sub(
        r"^(?:a\s+)?\d+(?:\.\d+)?\s*[mMkK]?[lL]\s+"
        r"(?:three[- ]neck\s+)?(?:round[- ]bottom\s+)?flask\s+was\s+"
        r"charged(?:\s+with)?\s+",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"^a\s+mechanical\s+stirrer\s*,\s*", "", name, flags=re.I)
    return name.strip(" ,;.")


def _ingredient_role(name: str, smiles: str) -> str:
    named_role = _ingredient_role_from_name(name)
    if named_role != "candidate_reactant":
        return named_role
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None and any(atom.GetAtomicNum() > 20 for atom in molecule.GetAtoms()):
        return "catalyst_or_base"
    return "candidate_reactant"


def _ingredient_role_from_name(name: str) -> str:
    key = _name_key(name).strip()
    if any(term == key or term in key for term in _SOLVENT_OR_WORKUP_TERMS):
        return "solvent_or_workup"
    if any(term == key or term in key for term in _CATALYST_OR_BASE_TERMS):
        return "catalyst_or_base"
    return "candidate_reactant"


def _known_product_ingredient(
    name: str,
    *,
    product_rows: Iterable[Mapping[str, Any]],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    key = _name_key(
        re.sub(
            r"\s+(?:hydrochloride\s+salt|hydrochloride)\s*$",
            "",
            name,
            flags=re.IGNORECASE,
        )
    ).strip()
    for raw in product_rows:
        row = dict(raw)
        if row.get("label") == current.get("label"):
            continue
        product_key = _name_key(
            re.sub(
                r"\s+(?:hydrochloride\s+salt|hydrochloride)\s*$",
                "",
                str(row.get("name") or ""),
                flags=re.IGNORECASE,
            )
        ).strip()
        if not key or not product_key or key != product_key:
            continue
        return {
            "name": name,
            "smiles": str(row.get("product_smiles") or ""),
            "source_formulation_smiles": str(
                row.get("source_product_smiles") or ""
            ),
            "parent": str(row.get("product_parent") or ""),
            "name_key": key,
            "role": "source_route_intermediate",
        }
    return {}


def _dedupe_ingredients(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    roles = {
        "source_route_intermediate": 3,
        "candidate_reactant": 2,
        "catalyst_or_base": 1,
        "solvent_or_workup": 0,
    }
    rows: dict[str, dict[str, Any]] = {}
    for raw in values:
        row = dict(raw)
        smiles = str(row.get("smiles") or "")
        if not smiles:
            continue
        current = rows.get(smiles)
        if current is None or roles.get(str(row.get("role") or ""), -1) > roles.get(
            str(current.get("role") or ""), -1
        ):
            rows[smiles] = row
    return [rows[key] for key in sorted(rows)]


def _parent_identity(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return ""
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        return ""
    parent = max(
        fragments,
        key=lambda mol: (
            sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()),
            mol.GetNumAtoms(),
        ),
    )
    return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)


def _element_counts(smiles: str) -> Counter[str]:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return Counter()
    return Counter(
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )


def _element_deficit(product: str, precursors: Iterable[str]) -> dict[str, int]:
    product_counts = _element_counts(product)
    precursor_counts: Counter[str] = Counter()
    for precursor in precursors:
        precursor_counts.update(_element_counts(precursor))
    return {
        element: count - precursor_counts[element]
        for element, count in sorted(product_counts.items())
        if count > precursor_counts[element]
    }


def _element_surplus(product: str, precursors: Iterable[str]) -> dict[str, int]:
    product_counts = _element_counts(product)
    precursor_counts: Counter[str] = Counter()
    for precursor in precursors:
        precursor_counts.update(_element_counts(precursor))
    return {
        element: count - product_counts[element]
        for element, count in sorted(precursor_counts.items())
        if count > product_counts[element]
    }


def _canonical(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _name_key(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SOURCE_ROUTE_OBSERVATION_SCHEMA",
    "SOURCE_ROUTE_PROPOSAL_SCHEMA",
    "compile_deterministic_source_route_observation",
]
