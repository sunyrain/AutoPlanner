"""Read-only scientific acceptance projection for replay execution."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.proof_portfolio import compile_proof_portfolio
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)

from .replay_contract import REPLAY_RESULT_SCHEMA


def build_replay_report(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
    stages: list[dict[str, Any]],
    *,
    interrupted: bool,
) -> dict[str, Any]:
    graph = service.graph_store.load()
    workbench = service.workbench()["snapshot"]
    portfolio = compile_proof_portfolio(
        graph, acceptance_spec=service.kernel.spec.acceptance
    )
    observed = {
        "accepted": portfolio["accepted"],
        "complete_route_count": portfolio["closeout"]["complete_route_count"],
        "selected_route_count": len(portfolio["selected_routes"]),
        "hyperedge_count": len(graph["edges"]),
        "validated_edge_count": sum(
            any(dict(proof).get("accepted") is True for proof in edge.get("reaction_proofs") or [])
            for edge in graph["edges"].values()
        ),
        "exact_record_count": len(graph["exact_records"]),
        "stock_terminal_count": sum(
            dict(proof).get("accepted") is True
            for proof in portfolio["leaf_proofs"].values()
        ),
        "independent_source_groups": sorted(
            {
                str(group)
                for edge in graph["edges"].values()
                for group in edge.get("independent_source_groups") or []
                if str(group)
            }
        ),
        "accepted_expansion_count": service.kernel.state.accepted_expansion_count,
        "attempt_count": service.kernel.state.attempt_count,
        "settled_task_count": service.kernel.state.settled_task_count,
        "model_invocations": int(
            service.kernel.state.model_totals.get("model_invocations") or 0
        ),
        "visual_invocations": int(
            service.kernel.state.model_totals.get("visual_invocations") or 0
        ),
    }
    expected = dict(pack.get("expected") or {})
    checks = {
        key: observed.get(key) == expected_value
        for key, expected_value in expected.items()
        if key in observed
    }
    if not interrupted:
        checks["all_expected_metrics_present"] = all(
            key in observed for key in expected
        )
        checks["zero_model_and_visual_calls"] = (
            observed["model_invocations"] == observed["visual_invocations"] == 0
        )
        checks["pack_reactions_all_validated"] = observed[
            "validated_edge_count"
        ] == len(pack["reactions"])
    accepted = bool(not interrupted and checks and all(checks.values()))
    return {
        "schema_version": REPLAY_RESULT_SCHEMA,
        "case_id": pack["case_id"],
        "run_id": service.kernel.spec.run_id,
        "run_dir": str(service.kernel.run_dir),
        "pack_sha256": pack["content_sha256"],
        "status": service.kernel.state.status,
        "interrupted": interrupted,
        "accepted": accepted,
        "stages": stages,
        "observed": observed,
        "expected": expected,
        "checks": checks,
        "graph_scientific_sha256": graph["scientific_sha256"],
        "portfolio_sha256": portfolio["content_sha256"],
        "workbench_sha256": workbench["content_sha256"],
        "stop_decision": service.kernel.decide_stop().to_dict(),
    }


__all__ = ["build_replay_report"]
