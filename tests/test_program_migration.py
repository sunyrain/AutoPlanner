from __future__ import annotations

from pathlib import Path

import pytest

from cascade_planner.application.canonical_hypergraph import CanonicalHypergraphError
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.program_migration import audit_program_migration
from cascade_planner.runtime.paths import RuntimePaths


def _gateway(tmp_path: Path) -> CampaignGateway:
    repository = tmp_path / "repository"
    repository.mkdir()
    return CampaignGateway(
        RuntimePaths.discover(
            repository_root=repository,
            environ={
                "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
                "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
                "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
                "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "run-index.sqlite3"),
            },
        )
    )


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:alcohol",
                "strategic_disconnection": "single precursor",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:alcohol",
                "route_family_id": "family:alcohol",
                "steps": [
                    {
                        "step_id": "step:alcohol",
                        "product_smiles": "CCO",
                        "precursor_smiles": ["CC=O"],
                        "transformation_hypothesis": "reduction",
                    }
                ],
            }
        ],
    }


def test_cross_run_program_audit_is_read_only_and_has_no_target_rules(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    first = gateway.create_run(run_id="migration-empty", target_name="first", target_smiles="CCO")
    second = gateway.create_run(
        run_id="migration-edge",
        target_name="second",
        target_smiles="CCO",
        global_plan=_plan(),
        materialize=True,
    )

    report = gateway.audit_programs(limit=10)
    selected = gateway.audit_programs(run_ids=("migration-edge",), limit=10)

    assert report["run_count"] == 2
    assert report["accepted_run_count"] == 2
    assert report["target_count"] == 2
    assert report["migration_state_counts"] == {
        "projection_ready": 1,
        "empty_graph": 1,
        "canonical_replay_required": 0,
        "error": 0,
    }
    assert report["semantics"]["read_only"] is True
    assert report["semantics"]["target_names_are_labels_not_rules"] is True
    assert selected["run_count"] == 1
    assert selected["runs"][0]["program_counts"]["programs"] == 1
    assert not (Path(first["run_dir"]) / ".autoplanner" / "program_store").exists()
    assert not (Path(second["run_dir"]) / ".autoplanner" / "program_store").exists()


def test_cross_run_program_audit_reports_stale_store_and_missing_run(
    tmp_path: Path,
) -> None:
    gateway = _gateway(tmp_path)
    gateway.create_run(run_id="migration-stale", target_name="stale", target_smiles="CCO")
    gateway.admit_programs("migration-stale", enable_program_admission=True)
    gateway.apply_plan("migration-stale", _plan(), materialize=True)

    report = gateway.audit_programs(run_ids=("migration-stale",))

    assert report["accepted_run_count"] == 0
    assert report["runs"][0]["checks"]["store_current_or_uninitialized"] is False
    with pytest.raises(ValueError, match="program_migration_runs_not_found:missing"):
        gateway.audit_programs(run_ids=("missing",))


def test_cross_run_program_audit_classifies_old_canonical_graphs_for_replay() -> None:
    class OldRunGateway:
        def list_runs(self, *, limit):
            assert limit == 10
            return {
                "runs": [
                    {
                        "run_id": "old-run",
                        "target_name": "historical target",
                    }
                ]
            }

        def program_projection(self, _run_id):
            raise CanonicalHypergraphError("canonical_graph_validation_failed")

    report = audit_program_migration(OldRunGateway(), limit=10)

    assert report["accepted_run_count"] == 0
    assert report["migration_state_counts"]["canonical_replay_required"] == 1
    assert report["runs"][0]["migration_state"] == "canonical_replay_required"
