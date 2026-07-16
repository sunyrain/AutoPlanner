"""Validation and replay read models extracted from the bounded gateway."""

from __future__ import annotations

from typing import Any

from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
)
from cascade_planner.interfaces.campaign_recovery_stores import (
    recovery_program_stores,
)


def validate_campaign(service: Any) -> dict[str, Any]:
    recovery = service.kernel.recover()
    graph = service.graph_store.load()
    oracle = service.graph_store.full_recompute_oracle()
    workbench = service.workbench()["snapshot"]
    stores = recovery_program_stores(service)
    program_store = stores["baseline"]
    biocatalytic_store = stores["biocatalytic"]["replay"]
    experimental_claim_store = stores["experimental_claims"]["replay"]
    program_status = program_store["status"]
    checks = {
        "event_replay_matches_snapshot": (
            recovery["replayed_state_sha256"] == service.kernel.state.to_dict()["content_sha256"]
        ),
        "graph_scientific_oracle_equal": (
            graph["scientific_sha256"] == oracle["scientific_sha256"]
        ),
        "graph_topology_oracle_equal": (graph["topology_sha256"] == oracle["topology_sha256"]),
        "workbench_binds_graph": (
            workbench["revision"]["graph_scientific_sha256"] == graph["scientific_sha256"]
        ),
        "program_store_replay_valid": (
            program_store["replay"]["event_count"] == program_status["event_count"]
        ),
        "program_store_current_projection_equal": (
            program_status["initialized"] is False or program_status["oracle"]["accepted"] is True
        ),
        "biocatalytic_program_store_replay_valid": (
            biocatalytic_store["event_count"] == len(biocatalytic_store["events"])
        ),
        "experimental_claim_store_replay_valid": (
            experimental_claim_store["event_count"]
            == len(experimental_claim_store["events"])
        ),
    }
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "validate",
        "run_id": service.kernel.spec.run_id,
        "accepted": all(checks.values()),
        "checks": checks,
        "recovery": recovery,
        "graph_revision": graph["revision"],
        "graph_scientific_sha256": graph["scientific_sha256"],
        "workbench_sha256": workbench["content_sha256"],
        "program_store": program_status,
        "biocatalytic_program_store": biocatalytic_store,
        "experimental_claim_store": experimental_claim_store,
    }


def replay_campaign(service: Any) -> dict[str, Any]:
    before = service.kernel.state.to_dict()["content_sha256"]
    recovery = service.kernel.recover()
    after = service.kernel.state.to_dict()["content_sha256"]
    oracle = service.graph_store.full_recompute_oracle()
    graph = service.graph_store.load()
    stores = recovery_program_stores(service)
    program_store = stores["baseline"]
    biocatalytic_store = stores["biocatalytic"]["replay"]
    experimental_claim_store = stores["experimental_claims"]["replay"]
    program_status = program_store["status"]
    checks = {
        "snapshot_reproduced": before == after,
        "event_replay_digest_equal": recovery["replayed_state_sha256"] == after,
        "graph_oracle_equal": (graph["scientific_sha256"] == oracle["scientific_sha256"]),
        "program_store_replay_valid": (
            program_store["replay"]["event_count"] == program_status["event_count"]
        ),
        "program_store_current_projection_equal": (
            program_status["initialized"] is False or program_status["oracle"]["accepted"] is True
        ),
        "biocatalytic_program_store_replay_valid": (
            biocatalytic_store["event_count"] == len(biocatalytic_store["events"])
        ),
        "experimental_claim_store_replay_valid": (
            experimental_claim_store["event_count"]
            == len(experimental_claim_store["events"])
        ),
    }
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "replay",
        "run_id": service.kernel.spec.run_id,
        "accepted": all(checks.values()),
        "checks": checks,
        "recovery": recovery,
        "program_store": program_status,
        "biocatalytic_program_store": biocatalytic_store,
        "experimental_claim_store": experimental_claim_store,
    }


__all__ = ["replay_campaign", "validate_campaign"]
