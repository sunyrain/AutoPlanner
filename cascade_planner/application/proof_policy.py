"""The single V4 reaction and stock proof policy.

Proof axes are stitched only through canonical IDs.  Exact literature cannot
replace reaction validation, model consensus cannot replace a source, and a
stock label cannot replace a current trusted observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


PROOF_POLICY_SCHEMA = "retrosynthesis_proof_policy.v1"
EDGE_PROOF_STITCH_SCHEMA = "edge_proof_stitch.v1"
LEAF_PROOF_STITCH_SCHEMA = "leaf_stock_proof_stitch.v1"
PROOF_POLICY_VERSION = "autoplanner.proof_policy.v1"

PROOF_LEVEL_NAMES = {
    0: "L0_hypothesis",
    1: "L1_structural_materialized",
    2: "L2_reaction_validated",
    3: "L3_exact_source",
    4: "L4_procurement_ready",
}


@dataclass(frozen=True, slots=True)
class ProofPolicy:
    minimum_edge_proof_level: int
    minimum_independent_source_groups: int
    require_stock_for_every_selected_leaf: bool
    stock_boundary: str
    version: str = PROOF_POLICY_VERSION
    schema_version: str = PROOF_POLICY_SCHEMA

    @classmethod
    def from_acceptance(
        cls,
        acceptance: RetrosynthesisAcceptanceSpec,
    ) -> "ProofPolicy":
        return cls(
            minimum_edge_proof_level=acceptance.minimum_edge_proof_level,
            minimum_independent_source_groups=(
                acceptance.minimum_independent_source_groups
            ),
            require_stock_for_every_selected_leaf=(
                acceptance.require_all_selected_leaves_stock_closed
            ),
            stock_boundary=acceptance.stock_boundary,
        )

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": self.schema_version,
            "version": self.version,
            "minimum_edge_proof_level": self.minimum_edge_proof_level,
            "minimum_independent_source_groups": (
                self.minimum_independent_source_groups
            ),
            "require_stock_for_every_selected_leaf": (
                self.require_stock_for_every_selected_leaf
            ),
            "stock_boundary": self.stock_boundary,
            "level_names": PROOF_LEVEL_NAMES,
            "semantics": {
                "reaction_validation_and_exact_source_are_distinct_axes": True,
                "independent_support_is_host_grouped": True,
                "route_completion_uses_weakest_link": True,
                "aggregate_counts_never_grant_completion": True,
            },
        }
        row["content_sha256"] = _digest(row)
        return row


def stitch_edge_proof(
    graph: Mapping[str, Any],
    edge_id: str,
    *,
    policy: ProofPolicy,
) -> dict[str, Any]:
    edge = dict(dict(graph.get("edges") or {}).get(edge_id) or {})
    reasons: list[str] = []
    if not edge or not _valid_content_digest(edge):
        reasons.append("canonical_edge_missing_or_digest_invalid")
    reaction_proofs = [
        dict(value)
        for value in edge.get("reaction_proofs") or []
        if isinstance(value, Mapping) and _valid_reaction_proof(value)
    ]
    reaction_level = max(
        (_reaction_proof_level(value) for value in reaction_proofs),
        default=0,
    )
    reaction_validated = reaction_level >= 2
    if not reaction_validated:
        reasons.append("reaction_validation_missing")

    exact_records: list[dict[str, Any]] = []
    source_binding_ids: set[str] = set()
    source_groups: set[str] = set()
    for record_id in edge.get("exact_record_ids") or []:
        record = dict(dict(graph.get("exact_records") or {}).get(record_id) or {})
        if not record or not _valid_content_digest(record):
            reasons.append(f"exact_record_invalid:{record_id}")
            continue
        if str(record.get("edge_digest") or "") != str(edge.get("edge_digest") or ""):
            reasons.append(f"exact_record_edge_mismatch:{record_id}")
            continue
        external_source_id = str(record.get("source_binding_id") or "")
        canonical_source_id = str(
            dict(graph.get("source_aliases") or {}).get(external_source_id) or ""
        )
        source = dict(
            dict(graph.get("source_bindings") or {}).get(canonical_source_id) or {}
        )
        if not source or not _valid_content_digest(source):
            reasons.append(f"exact_record_source_binding_invalid:{record_id}")
            continue
        exact_records.append(record)
        source_binding_ids.add(canonical_source_id)
        group = str(source.get("independence_group") or record.get("independence_group") or "")
        if group and group != "codex_model":
            source_groups.add(group)

    conflicts = _edge_conflicts(graph, edge=edge, exact_records=exact_records)
    if conflicts:
        reasons.append("unresolved_edge_conflict")
    exact_bound = bool(exact_records)
    if policy.minimum_edge_proof_level >= 3 and not exact_bound:
        reasons.append("exact_source_binding_missing")

    if reaction_validated and exact_bound:
        achieved = 3
    elif reaction_validated:
        achieved = 2
    elif edge:
        achieved = 1
    else:
        achieved = 0
    if reaction_level >= 4 and exact_bound:
        achieved = 4
    accepted = achieved >= policy.minimum_edge_proof_level and not conflicts
    independently_supported = (
        len(source_groups) >= policy.minimum_independent_source_groups
    )
    row = {
        "schema_version": EDGE_PROOF_STITCH_SCHEMA,
        "policy_version": policy.version,
        "edge_id": edge_id,
        "edge_digest": str(edge.get("edge_digest") or ""),
        "product_molecule_id": str(edge.get("product_molecule_id") or ""),
        "precursor_molecule_ids": list(edge.get("precursor_molecule_ids") or []),
        "structural_materialized": bool(edge),
        "reaction_validated": reaction_validated,
        "reaction_proof_digests": sorted(
            str(value.get("proof_digest") or "") for value in reaction_proofs
        ),
        "exact_source_bound": exact_bound,
        "exact_record_ids": sorted(
            str(value.get("record_id") or "") for value in exact_records
        ),
        "source_binding_ids": sorted(source_binding_ids),
        "independent_source_groups": sorted(source_groups),
        "independently_supported": independently_supported,
        "conflict_ids": sorted(
            str(value.get("conflict_id") or "") for value in conflicts
        ),
        "achieved_level": achieved,
        "achieved_level_name": PROOF_LEVEL_NAMES[achieved],
        "required_level": policy.minimum_edge_proof_level,
        "accepted": accepted,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "weakest_axis_controls_level": True,
            "exact_source_does_not_replace_reaction_validation": True,
            "conflict_blocks_edge_acceptance": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def stitch_leaf_stock_proof(
    graph: Mapping[str, Any],
    molecule_id: str,
    *,
    policy: ProofPolicy,
) -> dict[str, Any]:
    molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
    reasons: list[str] = []
    if not molecule or not _valid_content_digest(molecule):
        reasons.append("canonical_leaf_missing_or_digest_invalid")
    observation_id = str(molecule.get("active_stock_observation_id") or "")
    observation = dict(
        dict(graph.get("stock_observations") or {}).get(observation_id) or {}
    )
    if not observation or not _valid_content_digest(observation):
        reasons.append("trusted_active_stock_observation_missing")
    elif str(observation.get("molecule_id") or "") != molecule_id:
        reasons.append("stock_observation_molecule_mismatch")
    elif observation.get("accepted") is not True:
        reasons.append("active_stock_observation_not_accepted")
    accepted = not reasons and observation.get("accepted") is True
    row = {
        "schema_version": LEAF_PROOF_STITCH_SCHEMA,
        "policy_version": policy.version,
        "molecule_id": molecule_id,
        "canonical_smiles": str(molecule.get("canonical_smiles") or ""),
        "stock_observation_id": observation_id,
        "inventory_snapshot_set_id": str(
            observation.get("inventory_snapshot_set_id") or ""
        ),
        "audited_as_of": str(observation.get("audited_as_of") or ""),
        "required_boundary": policy.stock_boundary,
        "accepted": accepted,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "commonness_is_not_stock_authority": True,
            "active_observation_required": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def validate_canonical_graph_entities(graph: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if graph.get("schema_version") != "canonical_retrosynthesis_hypergraph.v1":
        reasons.append("canonical_graph_schema_invalid")
    for section in (
        "molecules",
        "edges",
        "source_bindings",
        "exact_records",
        "stock_observations",
        "route_families",
        "hypotheses",
        "conflicts",
    ):
        for entity_id, value in dict(graph.get(section) or {}).items():
            if not isinstance(value, Mapping) or not _valid_content_digest(value):
                reasons.append(f"canonical_entity_digest_invalid:{section}:{entity_id}")
                continue
            row = dict(value)
            identity_field = {
                "molecules": "molecule_id",
                "edges": "edge_id",
                "source_bindings": "source_binding_id",
                "exact_records": "record_id",
                "stock_observations": "stock_observation_id",
                "route_families": "route_family_id",
                "hypotheses": "hypothesis_id",
                "conflicts": "conflict_id",
            }[section]
            if str(row.get(identity_field) or "") != str(entity_id):
                reasons.append(
                    f"canonical_entity_identity_mismatch:{section}:{entity_id}"
                )
    molecules = dict(graph.get("molecules") or {})
    target_id = str(graph.get("target_molecule_id") or "")
    if not target_id or target_id not in molecules:
        reasons.append("canonical_target_molecule_missing")
    for edge_id, value in dict(graph.get("edges") or {}).items():
        if not isinstance(value, Mapping):
            continue
        edge = dict(value)
        if str(edge_id) != f"edge:{edge.get('edge_digest') or ''}":
            reasons.append(f"canonical_edge_digest_identity_mismatch:{edge_id}")
        molecule_refs = {
            str(edge.get("product_molecule_id") or ""),
            *(str(item) for item in edge.get("precursor_molecule_ids") or []),
        }
        if "" in molecule_refs or not molecule_refs <= set(molecules):
            reasons.append(f"canonical_edge_molecule_reference_invalid:{edge_id}")
    for observation_id, value in dict(graph.get("stock_observations") or {}).items():
        if isinstance(value, Mapping) and str(value.get("molecule_id") or "") not in molecules:
            reasons.append(
                f"canonical_stock_molecule_reference_invalid:{observation_id}"
            )
    return sorted(set(reasons))


def _edge_conflicts(
    graph: Mapping[str, Any],
    *,
    edge: Mapping[str, Any],
    exact_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edge_digest = str(edge.get("edge_digest") or "")
    record_ids = {str(value.get("record_id") or "") for value in exact_records}
    out: list[dict[str, Any]] = []
    for conflict in dict(graph.get("conflicts") or {}).values():
        if not isinstance(conflict, Mapping) or conflict.get("status") == "resolved":
            continue
        row = dict(conflict)
        subject = str(row.get("subject_id") or "")
        conflict_records = {str(value) for value in row.get("record_ids") or []}
        if edge_digest in subject or record_ids & conflict_records:
            out.append(row)
    return out


def _reaction_proof_level(value: Mapping[str, Any]) -> int:
    name = str(value.get("proof_level") or "")
    if name == "L4_procurement_ready":
        return 4
    if name == "L3_precedent_supported":
        return 3
    if value.get("accepted") is True or name == "L2_reaction_validated":
        return 2
    return 0


def _valid_reaction_proof(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("proof_digest", ""))
    return bool(
        supplied
        and supplied == _digest(row)
        and row.get("schema_version") == "reaction_step_proof.v1"
    )


def _valid_content_digest(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return bool(supplied and supplied == _digest(row))


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
