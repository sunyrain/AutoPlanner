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
from cascade_planner.application.route_edge_scope import (
    route_family_bound_origin_records,
)
from cascade_planner.application.route_strategy_value import (
    compile_evidence_maturity_vector,
    compile_strategic_value_vector,
)


PROOF_ROUTE_SCHEMA = "proof_stitched_route.v1"
CHEMICAL_CRITIC_RISK_VECTOR_SCHEMA = "chemical_critic_risk_vector.v1"

_CRITIC_VERDICTS = ("pass", "uncertain", "reject")
_CRITIC_VERDICT_RANK = {
    "pass": 0,
    "uncertain": 1,
    "reject": 2,
}


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
    for edge_id in edge_ids:
        gate = dict(
            edge_proofs.get(edge_id, {}).get("biocatalytic_step_proof_gate") or {}
        )
        if gate.get("required") is True and gate.get("accepted") is not True:
            unvalidated_biocatalytic_edge_ids.append(str(edge_id))
    unvalidated_biocatalytic_edge_ids = sorted(
        set(unvalidated_biocatalytic_edge_ids)
    )
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
    critic_step_id_to_edge_id = _route_critic_step_id_to_edge_id(
        graph=graph,
        edge_ids=edge_ids,
        route_family_id=family_id,
    )
    chemical_critic_risk = _compile_chemical_critic_risk_vector(
        edge_ids=edge_ids,
        family=family,
        chemical_critic=codex_critic,
        critic_step_id_to_edge_id=critic_step_id_to_edge_id,
    )
    risk = (
        0.42 * bool(conflicts)
        + float(chemical_critic_risk["score"])
        + 0.12 * min(1.0, len(edge_ids) / 12.0)
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
            "chemical_critic_risk_vector": chemical_critic_risk,
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
                "chemical_critic_risk_is_ranking_metadata_only": True,
            },
        }
    )


def _route_critic_step_id_to_edge_id(
    *,
    graph: Mapping[str, Any],
    edge_ids: list[str],
    route_family_id: str,
) -> dict[str, str]:
    """Bind Critic step identities to canonical route edges without guessing.

    Final-Critic input uses the proposal identity from the origin record bound
    to this route family, while older Critic records may already use the
    canonical edge identity.  Build both exact bindings from provenance.  Any
    token that names more than one edge is intentionally left unbound.
    """

    edges = dict(graph.get("edges") or {})
    bindings: dict[str, set[str]] = {}
    for raw_edge_id in edge_ids:
        edge_id = str(raw_edge_id or "")
        if not edge_id:
            continue
        bindings.setdefault(edge_id, set()).add(edge_id)
        edge = dict(edges.get(edge_id) or {})
        for origin in route_family_bound_origin_records(
            edge,
            route_family_id=route_family_id,
        ):
            proposal_id = str(origin.get("proposal_id") or "")
            if proposal_id:
                bindings.setdefault(proposal_id, set()).add(edge_id)
    return {
        step_id: next(iter(bound_edge_ids))
        for step_id, bound_edge_ids in bindings.items()
        if len(bound_edge_ids) == 1
    }


def _compile_chemical_critic_risk_vector(
    *,
    edge_ids: list[str],
    family: Mapping[str, Any],
    chemical_critic: Mapping[str, Any],
    critic_step_id_to_edge_id: Mapping[str, str],
) -> dict[str, Any]:
    """Derive route-ranking risk from already bound final-Critic assessments."""

    route_step_ids = {str(value) for value in edge_ids if str(value)}
    assessments_by_step_id: dict[str, dict[str, Any]] = {}
    unmatched_assessment_count = 0
    for raw in chemical_critic.get("step_assessments") or []:
        if not isinstance(raw, Mapping):
            unmatched_assessment_count += 1
            continue
        assessment = dict(raw)
        step_id = str(assessment.get("step_id") or "")
        verdict = str(assessment.get("verdict") or "")
        edge_id = str(critic_step_id_to_edge_id.get(step_id) or "")
        if edge_id not in route_step_ids or verdict not in _CRITIC_VERDICT_RANK:
            unmatched_assessment_count += 1
            continue
        previous = assessments_by_step_id.get(edge_id)
        if previous is None or _CRITIC_VERDICT_RANK[verdict] > _CRITIC_VERDICT_RANK[
            str(previous.get("verdict") or "")
        ]:
            assessments_by_step_id[edge_id] = assessment

    assessments = list(assessments_by_step_id.values())
    reviewed_step_count = len(assessments)
    route_step_count = len(route_step_ids)
    verdict_counts = {
        verdict: sum(row.get("verdict") == verdict for row in assessments)
        for verdict in _CRITIC_VERDICTS
    }
    verdict_rates = {
        verdict: round(count / max(1, reviewed_step_count), 6)
        for verdict, count in verdict_counts.items()
    }

    identifiable_key_event_step_ids: set[str] = set()
    for raw in family.get("key_event_critic_history") or []:
        if not isinstance(raw, Mapping) or raw.get("checkpoint_match") is not True:
            continue
        raw_assessment = raw.get("assessment")
        assessment = dict(raw_assessment) if isinstance(raw_assessment, Mapping) else {}
        step_id = str(
            raw.get("focus_step_id")
            or assessment.get("step_id")
            or ""
        )
        edge_id = str(critic_step_id_to_edge_id.get(step_id) or "")
        if edge_id in route_step_ids:
            identifiable_key_event_step_ids.add(edge_id)
    reviewed_key_event_step_ids = (
        identifiable_key_event_step_ids & assessments_by_step_id.keys()
    )
    key_event_verdict_counts = {
        verdict: sum(
            assessments_by_step_id[step_id].get("verdict") == verdict
            for step_id in reviewed_key_event_step_ids
        )
        for verdict in _CRITIC_VERDICTS
    }
    key_event_reviewed_step_count = len(reviewed_key_event_step_ids)
    key_event_uncertain_rate = (
        key_event_verdict_counts["uncertain"]
        / max(1, key_event_reviewed_step_count)
    )

    blocking_type_counts: dict[str, int] = {}
    for assessment in assessments:
        blocking_type = str(assessment.get("blocking_type") or "none")
        if blocking_type == "none":
            continue
        blocking_type_counts[blocking_type] = (
            blocking_type_counts.get(blocking_type, 0) + 1
        )
    typed_blocking_step_count = sum(blocking_type_counts.values())
    typed_blocking_rate = typed_blocking_step_count / max(1, reviewed_step_count)

    # A concrete reject is a route-level concern, while uncertain assessments
    # scale with their prevalence.  Key-event uncertainty receives a separate,
    # smaller term because it concentrates risk at the route's strategic hinge.
    # Coverage is deliberately excluded: it describes review completeness, not
    # chemistry, and must not turn an old partial review into a chemical defect.
    score_components = {
        "reject_presence": 0.52 * float(verdict_counts["reject"] > 0),
        "reject_prevalence": 0.13 * verdict_rates["reject"],
        "uncertainty_prevalence": 0.30 * verdict_rates["uncertain"],
        "key_event_uncertainty": 0.12 * key_event_uncertain_rate,
        "typed_blocking_prevalence": 0.03 * typed_blocking_rate,
    }
    score = min(1.0, sum(score_components.values()))
    return {
        "schema_version": CHEMICAL_CRITIC_RISK_VECTOR_SCHEMA,
        "route_step_count": route_step_count,
        "reviewed_step_count": reviewed_step_count,
        "unreviewed_step_count": max(0, route_step_count - reviewed_step_count),
        "unmatched_assessment_count": unmatched_assessment_count,
        "review_coverage_rate": round(
            reviewed_step_count / max(1, route_step_count),
            6,
        ),
        "verdict_counts": verdict_counts,
        "verdict_rates": verdict_rates,
        "identifiable_key_event_step_count": len(identifiable_key_event_step_ids),
        "key_event_reviewed_step_count": key_event_reviewed_step_count,
        "key_event_verdict_counts": key_event_verdict_counts,
        "key_event_uncertain_rate": round(key_event_uncertain_rate, 6),
        "blocking_type_counts": dict(sorted(blocking_type_counts.items())),
        "typed_blocking_step_count": typed_blocking_step_count,
        "typed_blocking_rate": round(typed_blocking_rate, 6),
        "score_components": {
            key: round(value, 6) for key, value in score_components.items()
        },
        "score": round(score, 6),
        "semantics": {
            "derived_from_existing_host_bound_final_critic_assessments": True,
            "sorting_and_display_metadata_only": True,
            "grants_no_route_admission": True,
            "review_coverage_is_not_chemical_risk": True,
            "route_level_prose_risks_are_not_double_counted": True,
        },
    }


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


__all__ = [
    "CHEMICAL_CRITIC_RISK_VECTOR_SCHEMA",
    "PROOF_ROUTE_SCHEMA",
    "build_route_candidate",
]
