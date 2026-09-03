from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import (
    _commit_route_closeout_revision,
    run_agentic_blackboard_controller,
)
from cascade_planner.legacy.harness_runtime.schemas import FinalVerdict
from cascade_planner.legacy.harness_runtime.tools import ToolExecutionState
from cascade_planner.legacy.runtime.artifact_revision import (
    ArtifactRevisionError,
    load_latest_closeout_manifest,
    load_latest_closeout_decision,
    publish_closeout_revision,
    sha256_file,
    validate_closeout_manifest,
    validate_latest_closeout_revision,
)
from scripts.legacy.evaluate_agentic_run import evaluate_run
from scripts.legacy.refresh_agentic_closeout_artifacts import refresh_agentic_closeout_artifacts


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_route_artifacts(root: Path, *, marker: str) -> dict[str, Path]:
    paths = {
        "route_consensus": root / "route_consensus_fused.json",
        "route_consensus_graph": root / "route_consensus_graph_fused.json",
        "explored_route_forest": root / "explored_route_forest.json",
        "route_forest_html": root / "route_forest.html",
    }
    _write_json(
        paths["route_consensus"],
        {"schema_version": "route_consensus.v1", "marker": marker},
    )
    _write_json(
        paths["route_consensus_graph"],
        {"schema_version": "route_consensus_graph.v1", "marker": marker},
    )
    _write_json(
        paths["explored_route_forest"],
        {"schema_version": "explored_route_forest.v1", "marker": marker},
    )
    paths["route_forest_html"].write_text(
        f"<!doctype html><title>{marker}</title>",
        encoding="utf-8",
    )
    return paths


def _publish(root: Path, paths: dict[str, Path], **kwargs: object) -> dict:
    return publish_closeout_revision(
        root,
        artifacts=paths,
        dependencies={
            "route_consensus_graph": ("route_consensus",),
            "explored_route_forest": ("route_consensus", "route_consensus_graph"),
            "route_forest_html": ("explored_route_forest",),
        },
        producer="test_closeout",
        case_id="case-1",
        **kwargs,
    )


def test_closeout_revision_is_content_addressed_and_dependency_bound(tmp_path: Path) -> None:
    paths = _write_route_artifacts(tmp_path, marker="r1")

    published = _publish(tmp_path, paths)
    validation = validate_latest_closeout_revision(tmp_path)
    loaded = load_latest_closeout_manifest(tmp_path)

    assert published["accepted"] is True
    assert validation["accepted"] is True
    assert published["revision_id"].startswith("sha256:")
    assert loaded["revision_id"] == published["revision_id"]
    rows = {row["artifact_id"]: row for row in loaded["artifacts"]}
    assert rows["route_consensus"]["artifact_schema_version"] == "route_consensus.v1"
    assert rows["explored_route_forest"]["dependencies"] == [
        {
            "artifact_id": "route_consensus",
            "sha256": rows["route_consensus"]["sha256"],
        },
        {
            "artifact_id": "route_consensus_graph",
            "sha256": rows["route_consensus_graph"]["sha256"],
        },
    ]
    for row in rows.values():
        content_path = tmp_path / row["content_path"]
        assert content_path.is_file()
        assert sha256_file(content_path) == row["sha256"]
    assert Path(published["staging_manifest_path"]).is_file()
    assert Path(published["manifest_path"]).is_file()
    assert Path(published["latest_pointer_path"]).is_file()


def test_latest_revision_keeps_cas_authority_when_compatibility_view_drifts(
    tmp_path: Path,
) -> None:
    old_paths = _write_route_artifacts(tmp_path, marker="old")
    old_consensus = old_paths["route_consensus"].read_bytes()
    first = _publish(tmp_path, old_paths)

    new_paths = _write_route_artifacts(tmp_path, marker="new")
    second = _publish(tmp_path, new_paths)
    assert first["revision_id"] != second["revision_id"]

    # Simulate a stale fixed-name reference: the newest forest remains, but a
    # previous consensus is copied back into its compatibility filename.
    new_paths["route_consensus"].write_bytes(old_consensus)

    validation = validate_latest_closeout_revision(tmp_path)
    assert validation["accepted"] is True
    assert validation["compatibility_projection_drift"] is True
    assert (
        "closeout_artifact_content_drift:route_consensus"
        in validation["compatibility_projection_validation"]["reasons"]
    )
    assert load_latest_closeout_manifest(tmp_path)["revision_id"] == second["revision_id"]


def test_evaluator_uses_cas_route_projection_when_fixed_view_drifts(tmp_path: Path) -> None:
    old_paths = _write_route_artifacts(tmp_path, marker="old")
    old_consensus = old_paths["route_consensus"].read_bytes()
    _publish(tmp_path, old_paths)

    new_paths = _write_route_artifacts(tmp_path, marker="new")
    _write_json(
        new_paths["explored_route_forest"],
        {
            "schema_version": "explored_route_forest.v1",
            "marker": "new",
            "primary_branch_id": "candidate",
            "branches": [
                {
                    "branch_id": "candidate",
                    "kind": "route_consensus",
                    "step_ids": ["s1"],
                    "advisory_only": True,
                }
            ],
            "steps": [{"step_id": "s1"}],
        },
    )
    _publish(tmp_path, new_paths)
    new_paths["route_consensus"].write_bytes(old_consensus)

    report = evaluate_run(tmp_path)

    assert report["closeout_revision"]["accepted"] is True
    assert report["closeout_revision"]["route_projection_trusted"] is True
    assert report["closeout_revision"]["compatibility_projection_drift"] is True
    assert report["route_forest"]["branch_count"] == 1
    assert "closeout_compatibility_projection_drift_using_cas_authority" in report["warnings"]


def test_manifest_rejects_stale_declared_dependency_hash(tmp_path: Path) -> None:
    paths = _write_route_artifacts(tmp_path, marker="dependency-drift")
    _publish(tmp_path, paths)
    manifest = load_latest_closeout_manifest(tmp_path)
    forest = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "explored_route_forest"
    )
    consensus_dependency = next(
        row
        for row in forest["dependencies"]
        if row["artifact_id"] == "route_consensus"
    )
    consensus_dependency["sha256"] = "0" * 64

    validation = validate_closeout_manifest(tmp_path, manifest)

    assert validation["accepted"] is False
    assert (
        "closeout_dependency_stale:explored_route_forest:route_consensus"
        in validation["reasons"]
    )


def test_failed_staging_validation_does_not_switch_latest_pointer(tmp_path: Path) -> None:
    paths = _write_route_artifacts(tmp_path, marker="stable")
    first = _publish(tmp_path, paths)
    pointer_path = Path(first["latest_pointer_path"])
    prior_pointer = pointer_path.read_bytes()
    captured_forest_digest = sha256_file(paths["explored_route_forest"])

    _write_json(
        paths["explored_route_forest"],
        {"schema_version": "explored_route_forest.v1", "marker": "changed"},
    )
    with pytest.raises(
        ArtifactRevisionError,
        match="closeout_artifact_changed_before_commit:explored_route_forest",
    ):
        _publish(
            tmp_path,
            paths,
            expected_digests={"explored_route_forest": captured_forest_digest},
        )

    assert pointer_path.read_bytes() == prior_pointer
    latest = validate_latest_closeout_revision(tmp_path)
    assert latest["accepted"] is True
    assert latest["revision_id"] == first["revision_id"]
    assert latest["compatibility_projection_drift"] is True


def test_required_route_dependency_cannot_be_omitted(tmp_path: Path) -> None:
    paths = _write_route_artifacts(tmp_path, marker="missing-dependency")
    first = _publish(tmp_path, paths)
    pointer_path = Path(first["latest_pointer_path"])
    prior_pointer = pointer_path.read_bytes()
    with pytest.raises(ArtifactRevisionError, match="closeout_staging_validation_failed"):
        publish_closeout_revision(
            tmp_path,
            artifacts=paths,
            dependencies={},
            producer="test_closeout",
            case_id="case-1",
        )

    assert pointer_path.read_bytes() == prior_pointer


def test_controller_closeout_exposes_digest_bound_compatibility_refs(tmp_path: Path) -> None:
    result = run_agentic_blackboard_controller(
        target_name="invalid-case",
        target_smiles="not-a-smiles",
        output_dir=tmp_path,
        max_rounds=0,
    )

    revision = result["agent_blackboard"]["closeout_revision"]
    digest_refs = result["final_verdict"]["artifact_digest_refs"]
    assert revision["accepted"] is True
    assert revision["authority"] == "content_addressed_closeout_manifest"
    assert digest_refs["parent_route_proof_snapshot"]["artifact_schema_version"] == (
        "parent_route_proof_snapshot.v1"
    )
    proof_dependencies = digest_refs["parent_route_proof_snapshot"]["dependencies"]
    if "route_consensus_graph" in digest_refs:
        assert proof_dependencies == [
            {
                "artifact_id": "route_consensus_graph",
                "sha256": digest_refs["route_consensus_graph"]["sha256"],
            }
        ]
    else:
        assert proof_dependencies == []
    assert digest_refs["final_verdict_core"]["artifact_schema_version"] == (
        "final_verdict_core.v1"
    )
    assert digest_refs["final_verdict_core"]["dependencies"] == [
        {
            "artifact_id": "parent_route_proof_snapshot",
            "sha256": digest_refs["parent_route_proof_snapshot"]["sha256"],
        }
    ]
    assert digest_refs["explored_route_forest"]["revision_id"] == revision["revision_id"]
    forest_dependency_ids = {
        row["artifact_id"] for row in digest_refs["explored_route_forest"]["dependencies"]
    }
    assert {"parent_route_proof_snapshot", "final_verdict_core"} <= forest_dependency_ids
    assert {
        artifact_id
        for artifact_id in ("route_consensus", "route_consensus_graph")
        if artifact_id in digest_refs
    } <= forest_dependency_ids
    assert digest_refs["route_forest_html"]["dependencies"] == [
        {
            "artifact_id": "explored_route_forest",
            "sha256": digest_refs["explored_route_forest"]["sha256"],
        }
    ]
    assert Path(result["artifacts"]["closeout_revision_manifest"]).is_file()
    assert Path(result["artifacts"]["closeout_latest_pointer"]).is_file()
    assert validate_latest_closeout_revision(tmp_path)["accepted"] is True


def test_saved_run_refresh_republishes_proof_and_verdict_bound_closeout(
    tmp_path: Path,
) -> None:
    run_agentic_blackboard_controller(
        target_name="invalid-case",
        target_smiles="not-a-smiles",
        output_dir=tmp_path,
        max_rounds=0,
    )

    summary = refresh_agentic_closeout_artifacts(tmp_path)
    manifest = load_latest_closeout_manifest(tmp_path)
    artifact_ids = {row["artifact_id"] for row in manifest["artifacts"]}
    refreshed_board = json.loads((tmp_path / "agent_blackboard.json").read_text(encoding="utf-8"))

    assert summary["accepted"] is True
    assert {"parent_route_proof_snapshot", "final_verdict_core"} <= artifact_ids
    assert refreshed_board["closeout_revision"]["accepted"] is True
    assert validate_latest_closeout_revision(tmp_path)["accepted"] is True


def test_saved_run_refresh_rebuilds_snapshot_capability_and_run_audits(
    tmp_path: Path,
) -> None:
    run_agentic_blackboard_controller(
        target_name="invalid-case",
        target_smiles="not-a-smiles",
        output_dir=tmp_path,
        max_rounds=0,
    )
    board_path = tmp_path / "agent_blackboard.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))
    board["agent_team_history"] = [
        {
            "coordinator": {
                "observed_child_agents": [
                    {
                        "prompt": (
                            "This trusted coordinator prompt forbids reaction strings "
                            "containing '>>'."
                        )
                    }
                ]
            }
        }
    ]
    _write_json(board_path, board)
    for filename in (
        "agent_blackboard_snapshot.json",
        "agentic_capability_audit.json",
        "agentic_run_audit.json",
    ):
        _write_json(tmp_path / filename, {"stale": True})
    bundle_path = tmp_path / "artifact_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle.setdefault("safety_flags", []).append(
        "typed_artifact_validation_failed:agent_blackboard_snapshot"
    )
    bundle.setdefault("validations", []).append(
        {
            "schema_version": "agentic_typed_artifact_validation_record.v1",
            "artifact_key": "agent_blackboard_snapshot",
            "accepted": False,
            "reasons": ["raw_reaction_injection"],
        }
    )
    _write_json(bundle_path, bundle)

    summary = refresh_agentic_closeout_artifacts(tmp_path)

    snapshot = json.loads(
        (tmp_path / "agent_blackboard_snapshot.json").read_text(encoding="utf-8")
    )
    capability = json.loads(
        (tmp_path / "agentic_capability_audit.json").read_text(encoding="utf-8")
    )
    run_audit = json.loads(
        (tmp_path / "agentic_run_audit.json").read_text(encoding="utf-8")
    )
    refreshed_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    snapshot_rows = [
        row
        for row in refreshed_bundle["validations"]
        if row.get("artifact_key") == "agent_blackboard_snapshot"
    ]

    assert snapshot["schema_version"] == "agent_blackboard_snapshot_artifact.v1"
    assert capability["schema_version"] == "agentic_capability_audit_artifact.v1"
    assert run_audit["schema_version"] == "agentic_run_audit_artifact.v1"
    assert summary["derived_diagnostics"][
        "agent_blackboard_snapshot_validation_accepted"
    ] is True
    assert summary["derived_diagnostics"][
        "agentic_run_audit_validation_accepted"
    ] is True
    assert "typed_artifacts_self_validated" not in capability["payload"][
        "failed_requirements"
    ]
    assert "artifact_refs_and_typed_validation_integrity" not in capability[
        "payload"
    ]["failed_requirements"]
    assert "agent_blackboard_snapshot" not in run_audit["payload"][
        "typed_artifact_validation_summary"
    ]["failed_artifact_keys"]
    assert len(snapshot_rows) == 1
    assert snapshot_rows[0]["accepted"] is True
    assert all(
        flag != "typed_artifact_validation_failed:agent_blackboard_snapshot"
        for flag in refreshed_bundle["safety_flags"]
    )


def test_evaluator_ignores_mutated_fixed_proof_and_verdict_after_closeout(
    tmp_path: Path,
) -> None:
    run_agentic_blackboard_controller(
        target_name="invalid-case",
        target_smiles="not-a-smiles",
        output_dir=tmp_path,
        max_rounds=0,
    )
    board_path = tmp_path / "agent_blackboard.json"
    verdict_path = tmp_path / "final_verdict.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))
    board["parent_route_proof"] = {
        "schema_version": "stitched_parent_route_proof.v1",
        "accepted": True,
        "solved": True,
    }
    _write_json(board_path, board)
    _write_json(
        verdict_path,
        {
            "schema_version": "codex_entry_final_verdict.v1",
            "case_id": "invalid-case",
            "verdict": "solved",
            "route_status": "solved",
            "solved": True,
        },
    )

    report = evaluate_run(tmp_path)

    assert validate_latest_closeout_revision(tmp_path)["accepted"] is True
    assert report["parent_route_proof"]["strict_solved"] is False
    assert report["final_verdict"]["claimed_solved"] is False
    assert (
        "closeout_compatibility:agent_blackboard_parent_proof_drift"
        in report["warnings"]
    )
    assert "closeout_compatibility:final_verdict_compatibility_drift" in report["warnings"]
def test_failed_new_controller_closeout_preserves_prior_cas_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = run_agentic_blackboard_controller(
        target_name="invalid-case",
        target_smiles="not-a-smiles",
        output_dir=tmp_path,
        max_rounds=0,
    )
    prior_validation = validate_latest_closeout_revision(tmp_path)
    prior_decision = load_latest_closeout_decision(tmp_path)
    board = dict(first["agent_blackboard"])
    board["parent_route_proof"] = {
        "schema_version": "parent_route_proof_attempt.v1",
        "accepted": False,
        "solved": False,
        "reasons": ["new_failed_attempt"],
    }
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input=dict(first["target_input"]),
        preflight=dict(first["preflight"]),
    )
    final = FinalVerdict(
        case_id=str(first["final_verdict"].get("case_id") or "invalid-case"),
        verdict="unresolved",
        route_status="unresolved",
        solved=False,
    )

    def fail_publish(*args, **kwargs):
        raise ArtifactRevisionError("injected_staging_failure")

    monkeypatch.setattr(
        "cascade_planner.legacy.harness_runtime.agentic_blackboard_controller.publish_closeout_revision",
        fail_publish,
    )
    failed = _commit_route_closeout_revision(
        state=state,
        blackboard=board,
        final_verdict=final,
        final_validation={
            "schema_version": "agentic_final_verdict_validation.v1",
            "accepted": True,
            "reasons": [],
        },
    )

    latest = validate_latest_closeout_revision(tmp_path)
    assert failed["accepted"] is False
    assert latest["accepted"] is True
    assert latest["revision_id"] == prior_validation["revision_id"]
    assert latest["compatibility_projection_drift"] is True
    assert load_latest_closeout_decision(tmp_path)["parent_route_proof"] == prior_decision[
        "parent_route_proof"
    ]


def test_controller_closeout_binds_ledger_to_canonical_not_caller_graph(
    tmp_path: Path,
) -> None:
    paths = {
        "route_consensus": tmp_path / "route_consensus_fused.json",
        "route_consensus_graph": tmp_path / "route_consensus_graph_fused.json",
        "canonical_route_consensus_graph": (
            tmp_path / "route_consensus_graph_canonical.json"
        ),
        "codex_campaign_proof_reconciliation": (
            tmp_path / "codex_campaign_proof_reconciliation.json"
        ),
        "frontier_ledger": tmp_path / "frontier_ledger.json",
        "explored_route_forest": tmp_path / "explored_route_forest.json",
    }
    for artifact_id, path in paths.items():
        schema = {
            "route_consensus": "route_consensus.v1",
            "route_consensus_graph": "route_consensus_graph.v1",
            "canonical_route_consensus_graph": "route_consensus_graph.v1",
            "codex_campaign_proof_reconciliation": (
                "codex_campaign_proof_reconciliation.v1"
            ),
            "frontier_ledger": "frontier_ledger.v1",
            "explored_route_forest": "explored_route_forest.v1",
        }[artifact_id]
        _write_json(
            path,
            {
                "schema_version": schema,
                "projection": (
                    "caller"
                    if artifact_id == "route_consensus_graph"
                    else "canonical"
                    if artifact_id == "canonical_route_consensus_graph"
                    else artifact_id
                ),
            },
        )
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_smiles": "CCO"},
        preflight={"case_id": "canonical-closeout"},
    )
    blackboard = {
        "case_id": "canonical-closeout",
        "target_profile": {"target_smiles": "CCO"},
        "artifact_refs": {key: str(path) for key, path in paths.items()},
    }

    result = _commit_route_closeout_revision(
        state=state,
        blackboard=blackboard,
        final_verdict=FinalVerdict(
            case_id="canonical-closeout",
            verdict="unresolved",
            route_status="unresolved",
            solved=False,
        ),
        final_validation={
            "schema_version": "agentic_final_verdict_validation.v1",
            "accepted": True,
            "reasons": [],
        },
    )

    assert result["accepted"] is True
    manifest = load_latest_closeout_manifest(tmp_path)
    rows = {row["artifact_id"]: row for row in manifest["artifacts"]}
    ledger_dependencies = {
        row["artifact_id"] for row in rows["frontier_ledger"]["dependencies"]
    }
    assert ledger_dependencies == {
        "canonical_route_consensus_graph",
        "codex_campaign_proof_reconciliation",
    }
    assert "route_consensus_graph" not in ledger_dependencies
    assert {
        row["artifact_id"]
        for row in rows["parent_route_proof_snapshot"]["dependencies"]
    } == {"canonical_route_consensus_graph"}
