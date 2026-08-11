"""Host-side normalization of advisory visual literature observations."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from cascade_planner.harness.reaction_step_verifier import canonical_reaction_digest
from cascade_planner.interfaces.visual_observation_chemistry import (
    atom_contributing_reactant_partition as _atom_contributing_reactant_partition,
    canonical_reactants as _canonical_reactants,
    canonical_smiles as _canonical_smiles,
    connectivity_smiles as _connectivity_smiles,
    partition_reactant_labels as _partition_reactant_labels,
    raw_reactants as _raw_reactants,
)


VISUAL_EVIDENCE_OBSERVATION_SCHEMA = "visual_source_candidate_observation.v1"


def normalize_visual_observation(
    request: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    if str(result.get("request_sha256") or "") != str(
        request.get("content_sha256") or ""
    ):
        raise ValueError("visual_provider_request_digest_mismatch")
    source = dict(request.get("source") or {})
    target_smiles = _canonical_smiles(request.get("target_smiles"))
    current_edges = {
        canonical_reaction_digest(
            _canonical_smiles(row.get("product_smiles")),
            _canonical_reactants(row.get("precursor_smiles")),
        ): str(row.get("edge_id") or "")
        for row in request.get("edges") or []
        if isinstance(row, Mapping)
        and _canonical_smiles(row.get("product_smiles"))
        and _canonical_reactants(row.get("precursor_smiles"))
    }
    current_nodes = {
        canonical
        for row in request.get("edges") or []
        if isinstance(row, Mapping)
        for value in [row.get("product_smiles"), *(row.get("precursor_smiles") or [])]
        if (canonical := _canonical_smiles(value))
    }
    current_nodes.add(target_smiles)
    node_by_connectivity = {
        _connectivity_smiles(value): value
        for value in current_nodes
        if _connectivity_smiles(value)
    }
    chain = dict(result.get("candidate_chain") or {})
    steps = []
    prior_precursors: list[str] = []
    chain_prefix_eligible = bool(target_smiles)
    chain_reasons: list[str] = []
    for index, raw in enumerate(chain.get("steps") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        product = _canonical_smiles(row.get("product_smiles"))
        raw_reactants = _raw_reactants(row.get("reactant_smiles"))
        reactants = _canonical_reactants(raw_reactants)
        if not product or not reactants:
            continue
        partition = _atom_contributing_reactant_partition(product, reactants)
        reactants = list(partition["precursor_smiles"])
        spectator_reactants = list(partition["spectator_smiles"])
        reactant_labels, spectator_labels = _partition_reactant_labels(
            raw_reactants,
            row.get("reactant_labels"),
            precursor_smiles=reactants,
        )
        root_anchor = ""
        step_reasons: list[str] = []
        if index == 1:
            if product == target_smiles:
                root_anchor = "exact_target_identity"
            elif _connectivity_smiles(product) == _connectivity_smiles(target_smiles):
                product = target_smiles
                root_anchor = "target_connectivity_stereo_anchor"
            elif _connectivity_smiles(product) in node_by_connectivity:
                product = node_by_connectivity[_connectivity_smiles(product)]
                root_anchor = "canonical_frontier_identity"
            else:
                chain_prefix_eligible = False
                step_reasons.append("visual_chain_root_not_target_connected")
        elif not any(
            _connectivity_smiles(product) == _connectivity_smiles(value)
            for value in prior_precursors
        ):
            chain_prefix_eligible = False
            step_reasons.append("visual_chain_step_not_connected_to_prior_precursor")
        if partition["accepted"] is not True:
            chain_prefix_eligible = False
            step_reasons.append("visual_precursor_partition_not_host_admitted")
        chain_reasons.extend(step_reasons)
        reaction_digest = canonical_reaction_digest(product, reactants)
        steps.append(
            {
                "candidate_id": (
                    "visual:"
                    + _digest(
                        {
                            "source": source.get("source_ref"),
                            "reaction": reaction_digest,
                        }
                    )[:24]
                ),
                "product_smiles": product,
                "precursor_smiles": reactants,
                "product_label": str(row.get("product_label") or "")[:300],
                "reactant_labels": reactant_labels,
                "spectator_reactant_labels": spectator_labels,
                "spectator_reactant_smiles": spectator_reactants,
                "source_locator": str(row.get("source_locator") or "")[:500],
                "condition_candidate": _condition_candidate(row),
                "reaction_digest": reaction_digest,
                "matched_current_edge_id": current_edges.get(reaction_digest, ""),
                "relation_type": "visual_candidate",
                "allowed_use": "global_replan_hypothesis_only",
                "host_smiles_parse_accepted": True,
                "host_precursor_partition": partition,
                "grants_exact_evidence": False,
                "admission_eligible": chain_prefix_eligible,
                "root_anchor": root_anchor,
                "structure_derivation": dict(row.get("structure_derivation") or {}),
                "stereochemistry_status": str(
                    row.get("stereochemistry_status") or ""
                )[:100],
                "not_exact_literature_segment": bool(
                    row.get("not_exact_literature_segment")
                ),
                "risk_flags": [
                    str(value)[:200]
                    for value in row.get("risk_flags") or []
                    if str(value).strip()
                ][:16],
                "exact_structure_binding_candidate": bool(
                    root_anchor == "exact_target_identity"
                    and partition["accepted"] is True
                    and str(row.get("source_locator") or "").strip()
                ),
                "chain_rejection_reasons": step_reasons,
            }
        )
        prior_precursors = reactants
        if len(steps) >= max_steps:
            break
    observation = {
        "schema_version": VISUAL_EVIDENCE_OBSERVATION_SCHEMA,
        "request_sha256": str(request.get("content_sha256") or ""),
        "source_ref": str(source.get("source_ref") or ""),
        "source_pdf_sha256": str(source.get("source_pdf_sha256") or ""),
        "source_fulltext_sha256": str(source.get("source_fulltext_sha256") or ""),
        "source_artifact_sha256": str(source.get("source_artifact_sha256") or ""),
        "source_artifact_kind": str(source.get("source_artifact_kind") or ""),
        "page_bindings": [dict(row) for row in source.get("pages") or []],
        "provider_receipt": dict(result.get("provider_receipt") or {}),
        "provider_status": str(result.get("provider_status") or ""),
        "candidate_steps": steps,
        "candidate_step_count": len(steps),
        "admission_eligible_step_count": sum(
            row["admission_eligible"] is True for row in steps
        ),
        "chain_admission_accepted": bool(steps)
        and all(row["admission_eligible"] is True for row in steps)
        and any(
            row["matched_current_edge_id"]
            or row["root_anchor"]
            in {
                "exact_target_identity",
                "target_connectivity_stereo_anchor",
                "canonical_frontier_identity",
            }
            for row in steps
        ),
        "chain_admission_reasons": sorted(set(chain_reasons)),
        "matched_current_edge_count": sum(
            bool(row["matched_current_edge_id"]) for row in steps
        ),
        "frontier_anchored_step_count": sum(
            row["root_anchor"] == "canonical_frontier_identity" for row in steps
        ),
        "target_anchored_step_count": sum(
            row["root_anchor"]
            in {"exact_target_identity", "target_connectivity_stereo_anchor"}
            for row in steps
        ),
        "exact_structure_binding_candidate_count": sum(
            row["exact_structure_binding_candidate"] is True for row in steps
        ),
        "semantics": {
            "model_output_is_advisory": True,
            "host_canonicalization_is_not_source_verification": True,
            "deterministic_source_parser_must_independently_reconstruct_exact_rows": True,
            "observation_cannot_grant_L2_L3_or_stock": True,
            "canonical_frontier_anchor_enables_module_replacement": True,
        },
    }
    observation["content_sha256"] = _digest(observation)
    return observation


def _condition_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("condition_candidate") or row.get("conditions")
    if isinstance(value, Mapping):
        sanitized = {
            str(key)[:100]: item
            for key, item in value.items()
            if item not in (None, "", [])
            and str(key).casefold()
            not in {
                "accepted",
                "condition_status",
                "evidence_backed",
                "exact",
                "exact_ready",
                "grants_exact_evidence",
                "proof_level",
                "source_type",
                "validated",
            }
        }
        text_values = list(
            dict.fromkeys(
                " ".join(str(sanitized.get(key) or "").split())
                for key in (
                    "condition_text_transcribed",
                    "procedure_text",
                    "source_excerpt",
                )
                if str(sanitized.get(key) or "").strip()
            )
        )
        text = "; ".join(text_values)
        reference_only = bool(
            text_values
            and all(
                re.fullmatch(
                    r"(?:ref(?:erence)?s?\.?\s*)?[\d,;\-–—\s]+",
                    item,
                    flags=re.IGNORECASE,
                )
                for item in text_values
            )
        )
        if reference_only:
            return {
                "schema_version": "visual_condition_candidate.v1",
                "source_reference_annotation": text[:500],
                "condition_status": "reference_citation_only",
                "source_type": "visual_hypothesis",
                "grants_exact_evidence": False,
            }
        return {
            **sanitized,
            "schema_version": "visual_condition_candidate.v1",
            "condition_status": "unverified_visual_transcription",
            "source_type": "visual_hypothesis",
            "grants_exact_evidence": False,
        }
    text = " ".join(str(value or "").split())
    return (
        {
            "schema_version": "visual_condition_candidate.v1",
            "procedure_text": text[:2000],
            "condition_status": "unverified_visual_transcription",
            "source_type": "visual_hypothesis",
            "grants_exact_evidence": False,
        }
        if text
        else {}
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["VISUAL_EVIDENCE_OBSERVATION_SCHEMA", "normalize_visual_observation"]
