"""Build one proof-stitched route candidate from a canonical subroute."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.proof_policy import ProofPolicy, stitch_leaf_stock_proof
from cascade_planner.application.route_innovations import (
    BIOCATALYTIC_KINDS,
    route_innovation_summary,
)
from cascade_planner.application.route_strategy_value import (
    compile_evidence_maturity_vector,
    compile_strategic_value_vector,
)


PROOF_ROUTE_SCHEMA = "proof_stitched_route.v1"


def build_route_candidate(
    graph: Mapping[str, Any],
    *,
    family_id: str,
    family: Mapping[str, Any],
    variant: Any,
    edge_proofs: Mapping[str, Mapping[str, Any]],
    leaf_proof_cache: dict[str, dict[str, Any]],
    policy: ProofPolicy,
) -> dict[str, Any]:
    edge_ids = sorted(variant.edge_ids)
    leaf_ids = sorted(variant.leaf_ids)
    proofs = [dict(edge_proofs[edge_id]) for edge_id in edge_ids]
    innovation_summary = route_innovation_summary(
        graph,
        edge_ids,
        route_family_id=family_id,
    )
    unvalidated_biocatalytic_edge_ids: list[str] = []
    for option in innovation_summary["selected_options"]:
        if option.get("kind") not in BIOCATALYTIC_KINDS:
            continue
        edge_id = str(option.get("edge_id") or "")
        gate = dict(edge_proofs.get(edge_id, {}).get("innovation_proof_gate") or {})
        option_id = str(option.get("innovation_id") or "")
        validated_ids = {
            str(value) for value in gate.get("validated_innovation_ids") or []
        }
        if gate.get("generic_validation") is not True and option_id not in validated_ids:
            unvalidated_biocatalytic_edge_ids.append(edge_id)
    leaves = [
        leaf_proof_cache.setdefault(
            molecule_id,
            stitch_leaf_stock_proof(graph, molecule_id, policy=policy),
        )
        for molecule_id in leaf_ids
    ]
    source_groups = sorted(
        {
            str(group)
            for proof in proofs
            for group in proof.get("independent_source_groups") or []
            if str(group)
        }
    )
    conflicts = sorted(
        {
            str(conflict_id)
            for proof in proofs
            for conflict_id in proof.get("conflict_ids") or []
            if str(conflict_id)
        }
    )
    min_proof = min((int(value["achieved_level"]) for value in proofs), default=0)
    if unvalidated_biocatalytic_edge_ids:
        min_proof = min(min_proof, 1)
    unproven_edge_ids = sorted(
        {
            *(
                str(value["edge_id"])
                for value in proofs
                if value.get("accepted") is not True
            ),
            *unvalidated_biocatalytic_edge_ids,
        }
    )
    stock_rate = sum(value["accepted"] is True for value in leaves) / max(1, len(leaves))
    reaction_feasibility_rate = sum(
        value.get("reaction_validated") is True for value in proofs
    ) / max(1, len(proofs))
    exact_evidence_rate = sum(
        value.get("exact_source_bound") is True for value in proofs
    ) / max(1, len(proofs))
    condition_completeness_rate = sum(
        any(
            dict(
                dict(graph.get("procedure_records") or {}).get(str(record_id)) or {}
            ).get("condition_completeness", {}).get("complete")
            is True
            for record_id in value.get("procedure_record_ids") or []
        )
        for value in proofs
    ) / max(1, len(proofs))
    open_leaf_molecule_ids = sorted(
        str(value["molecule_id"])
        for value in leaves
        if value.get("accepted") is not True
    )
    source_required = policy.minimum_edge_proof_level >= 3
    source_met = (
        not source_required
        or len(source_groups) >= policy.minimum_independent_source_groups
    )
    complete = (
        bool(edge_ids)
        and all(value["accepted"] is True for value in proofs)
        and not unvalidated_biocatalytic_edge_ids
    )
    if policy.require_stock_for_every_selected_leaf:
        complete = complete and bool(leaves) and stock_rate == 1.0
    complete = complete and source_met and not conflicts
    root_edges = sorted(
        edge_id
        for edge_id in edge_ids
        if str(graph["edges"][edge_id]["product_molecule_id"])
        == str(graph.get("target_molecule_id") or "")
    )
    precursor_frequency: dict[str, int] = {}
    for edge_id in edge_ids:
        for molecule_id in graph["edges"][edge_id]["precursor_molecule_ids"]:
            key = str(molecule_id)
            precursor_frequency[key] = precursor_frequency.get(key, 0) + 1
    convergence = sum(value > 1 for value in precursor_frequency.values()) / max(
        1, len(precursor_frequency)
    )
    strategy_card = dict(family.get("strategy_card") or {})
    strategic_value = compile_strategic_value_vector(
        graph,
        edge_ids=edge_ids,
        root_edge_ids=root_edges,
        strategy_card=strategy_card,
        convergence_score=convergence,
    )
    evidence_maturity = compile_evidence_maturity_vector(
        reaction_feasibility_rate=reaction_feasibility_rate,
        exact_evidence_rate=exact_evidence_rate,
        condition_completeness_rate=condition_completeness_rate,
        source_independence_met=source_met,
    )
    critic_uncertainties = sorted(
        {
            str(reason)
            for edge_id in edge_ids
            for reason in dict(
                dict(graph["edges"].get(edge_id) or {}).get("chemical_strategy_critic")
                or {}
            ).get("uncertainties")
            or []
            if str(reason)
        }
    )
    codex_critic = dict(family.get("chemical_critic") or {})
    critic_uncertainties = sorted(
        {
            *critic_uncertainties,
            *(
                str(value)
                for value in codex_critic.get("route_level_risks") or []
                if str(value)
            ),
        }
    )
    codex_critic_status = str(
        codex_critic.get("status")
        or codex_critic.get("overall_assessment")
        or "unavailable"
    )
    risk = (
        0.42 * bool(conflicts)
        + 0.24 * min(1.0, len(critic_uncertainties) / 4.0)
        + 0.18 * (codex_critic_status == "uncertain")
        + 0.80 * (codex_critic_status == "reject")
        + 0.12 * min(1.0, len(edge_ids) / 12.0)
        + 0.14 * min(1.0, len(unvalidated_biocatalytic_edge_ids))
        + 0.08 * min(1.0, int(innovation_summary["mechanism_extrapolation_count"]))
    )
    identity = {
        "route_family_id": family_id,
        "edge_ids": edge_ids,
        "leaf_molecule_ids": leaf_ids,
    }
    return _with_content_digest(
        {
            "schema_version": PROOF_ROUTE_SCHEMA,
            "route_id": f"route:{_digest(identity)}",
            "route_family_id": family_id,
            "strategy": str(family.get("strategy") or ""),
            "strategy_card": strategy_card,
            "strategy_id": str(
                family.get("strategy_id") or strategy_card.get("strategy_id") or ""
            ),
            "strategy_digest": str(
                family.get("strategy_digest")
                or strategy_card.get("strategy_digest")
                or ""
            ),
            "execution_domain": str(
                family.get("execution_domain")
                or strategy_card.get("execution_domain")
                or "chemical"
            ),
            "edge_ids": edge_ids,
            "leaf_molecule_ids": leaf_ids,
            "root_edge_ids": root_edges,
            "module_selections": dict(variant.module_selections),
            "minimum_edge_proof_level": min_proof,
            "all_edges_proven": bool(proofs)
            and all(value["accepted"] for value in proofs)
            and not unvalidated_biocatalytic_edge_ids,
            "unproven_edge_ids": unproven_edge_ids,
            "stock_closure_rate": round(stock_rate, 6),
            "reaction_feasibility_rate": round(reaction_feasibility_rate, 6),
            "exact_evidence_rate": round(exact_evidence_rate, 6),
            "condition_completeness_rate": round(condition_completeness_rate, 6),
            "all_leaves_stock_closed": bool(leaves) and stock_rate == 1.0,
            "open_leaf_molecule_ids": open_leaf_molecule_ids,
            "independent_source_groups": source_groups,
            "source_independence_met": source_met,
            "source_independence_required": source_required,
            "conflict_ids": conflicts,
            "length": len(edge_ids),
            "physical_step_count": len(edge_ids),
            "chemical_step_equivalent_count": int(
                innovation_summary["chemical_step_equivalent_count"]
            ),
            "net_step_savings": int(innovation_summary["net_step_savings"]),
            "biocatalytic_superstep_count": int(
                innovation_summary["biocatalytic_superstep_count"]
            ),
            "biocatalytic_step_count": int(
                innovation_summary["biocatalytic_step_count"]
            ),
            "mechanism_extrapolation_count": int(
                innovation_summary["mechanism_extrapolation_count"]
            ),
            "unvalidated_biocatalytic_edge_ids": sorted(
                set(unvalidated_biocatalytic_edge_ids)
            ),
            "route_innovation_summary": innovation_summary,
            "convergence_score": round(convergence, 6),
            "strategic_value": strategic_value,
            "strategic_value_score": strategic_value["score"],
            "evidence_maturity": evidence_maturity,
            "evidence_maturity_score": evidence_maturity["score"],
            "chemical_critic_uncertainties": critic_uncertainties,
            "codex_chemical_critic": codex_critic,
            "codex_chemical_critic_status": codex_critic_status,
            "risk_score": round(float(risk), 6),
            "evidence_risk_score": round(1.0 - evidence_maturity["score"], 6),
            "complete": complete,
            "selected": False,
            "reported_in_source": family.get("reported_in_source") is True,
            "reported_source_refs": sorted(
                {
                    str(value)
                    for value in family.get("reported_source_refs") or []
                    if str(value)
                }
            ),
            "semantics": {
                "weakest_edge_controls_route": True,
                "every_leaf_requires_stock_observation": True,
                "counts_do_not_override_boolean_proofs": True,
                "reported_route_survives_unresolved_edge_for_display": (
                    family.get("reported_in_source") is True
                ),
                "biocatalytic_superstep_requires_specific_validation": True,
                "mechanism_extrapolation_never_claims_anchor_source_reported_it": True,
                "strategic_value_is_independent_of_evidence": True,
                "evidence_maturity_is_independent_of_strategy_wording": True,
                "unrecognized_reaction_class_is_not_an_exploration_rejection": True,
            },
        }
    )


def _with_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row["content_sha256"] = _digest(row)
    return row


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


__all__ = ["PROOF_ROUTE_SCHEMA", "build_route_candidate"]
