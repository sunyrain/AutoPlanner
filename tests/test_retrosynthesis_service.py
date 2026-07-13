from __future__ import annotations

from pathlib import Path

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.harness.agentic_blackboard_controller import (
    run_agentic_blackboard_controller,
)


def _spec() -> RunSpec:
    return RunSpec(
        run_id="v4-service",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        created_at="2026-07-13T00:00:00Z",
        limits=RunLimits(
            model=RetrosynthesisRunBudget(
                max_model_invocations=0,
                max_accepted_expansions=8,
                max_attempt_runs=12,
            ),
            max_total_tasks=32,
        ),
    )


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:acyl",
                "strategic_disconnection": "late acyl substitution",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:acyl",
                "route_family_id": "family:acyl",
                "steps": [
                    {
                        "step_id": "step:ester",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "transformation_hypothesis": "acyl substitution",
                    }
                ],
            }
        ],
    }


def test_v4_service_owns_one_kernel_graph_frontier_and_closeout(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_dir = tmp_path / "run"
    service = RetrosynthesisCampaignService.create(
        runtime_root,
        run_dir,
        spec=_spec(),
    )

    planned = service.apply_global_plan(_plan(), idempotency_key="plan")
    assert planned["changed"] is True
    assert service.kernel.state.graph_revision == 1
    assert {row["kind"] for row in service.kernel.state.deficits} >= {
        "materialization",
        "route_closure",
        "diversity",
    }

    materialized = service.execute_frontier_materialization(
        idempotency_key="materialize"
    )
    status = service.status()

    assert materialized["executed_command_count"] == 1
    assert service.kernel.state.graph_revision == 2
    assert service.kernel.state.accepted_expansion_count == 1
    assert service.kernel.state.attempt_count == 1
    assert service.kernel.state.model_totals["model_invocations"] == 0
    assert {row["kind"] for row in status["frontier"]} >= {
        "validation",
        "evidence",
        "stock",
        "route_closure",
    }
    assert status["semantics"] == {
        "single_kernel": True,
        "single_graph": True,
        "single_frontier": True,
        "blackboard_is_not_authority": True,
    }
    assert not (run_dir / "campaign_state.json").exists()
    assert not (run_dir / "frontier_queue").exists()

    closeout = service.closeout(idempotency_key="unresolved")
    assert closeout["portfolio"]["closeout"]["decision"] == "unresolved"
    assert closeout["acceptance_report"]["accepted"] is False
    assert service.kernel.state.acceptance_report["accepted"] is False


def test_v4_service_reopens_from_kernel_without_private_campaign_state(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_dir = tmp_path / "run"
    first = RetrosynthesisCampaignService.create(
        runtime_root,
        run_dir,
        spec=_spec(),
    )
    first.apply_global_plan(_plan(), idempotency_key="plan")
    first.execute_frontier_materialization(idempotency_key="materialize")
    before = first.status()

    reopened = RetrosynthesisCampaignService.open(runtime_root, run_dir)
    after = reopened.status()

    assert after["graph_revision"] == before["graph_revision"]
    assert after["attempt_count"] == before["attempt_count"]
    assert after["accepted_expansion_count"] == before["accepted_expansion_count"]
    assert (
        after["portfolio"]["graph_scientific_sha256"]
        == before["portfolio"]["graph_scientific_sha256"]
    )
    assert after["portfolio"]["content_sha256"] == before["portfolio"]["content_sha256"]
    assert reopened.kernel.recover()["event_count"] == reopened.kernel.state.event_count


def test_public_controller_surface_can_dispatch_to_single_kernel_v4(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "adapter-run"

    result = run_agentic_blackboard_controller(
        engine="v4",
        target_name="ethyl acetate",
        target_smiles="CCOC(C)=O",
        output_dir=run_dir,
        retrosynthesis_run_budget=RetrosynthesisRunBudget(
            max_model_invocations=0,
        ),
    )

    assert result["engine"] == "v4"
    assert result["semantics"]["thin_adapter"] is True
    assert result["status"]["semantics"]["single_kernel"] is True
    assert result["status"]["model_totals"]["model_invocations"] == 0
    assert (run_dir / ".autoplanner" / "kernel" / "events.jsonl").is_file()
    assert not (run_dir / "campaign_state.json").exists()
