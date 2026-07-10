"""Refresh derived closeout artifacts for a saved agentic blackboard run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.artifact_schemas import ARTIFACT_CLASSES  # noqa: E402
from cascade_planner.harness.agentic_blackboard_controller import (  # noqa: E402
    _commit_route_closeout_revision,
    _downgrade_invalid_agentic_final_verdict,
    _record_agent_blackboard_snapshot_artifact,
    _record_agentic_capability_audit_artifact,
    _record_agentic_run_audit_artifact,
    _refresh_multisource_route_consensus,
    _validate_and_record_typed_artifact,
    _validate_agentic_final_verdict,
    emit_agentic_final_verdict,
)
from cascade_planner.harness.hypothesis_execution_report import (  # noqa: E402
    compile_hypothesis_execution_report,
)
from cascade_planner.harness.hypothetical_retrosynthesis_report import (  # noqa: E402
    compile_hypothesis_only_retrosynthesis_report,
)
from cascade_planner.harness.route_forest import write_route_forest_artifacts  # noqa: E402
from cascade_planner.harness.tools import ToolExecutionState  # noqa: E402


def refresh_agentic_closeout_artifacts(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    blackboard_path = root / "agent_blackboard.json"
    if not blackboard_path.exists():
        raise FileNotFoundError(f"agent_blackboard.json not found: {blackboard_path}")
    blackboard = _read_json(blackboard_path)
    case_id = str(blackboard.get("case_id") or root.name)
    refs = blackboard.setdefault("artifact_refs", {})
    refs.update(
        {
            "agentic_final_verdict_validation": str(
                root / "agentic_final_verdict_validation.json"
            ),
            "agent_blackboard_snapshot": str(root / "agent_blackboard_snapshot.json"),
            "agentic_capability_audit": str(root / "agentic_capability_audit.json"),
            "agentic_run_audit": str(root / "agentic_run_audit.json"),
        }
    )
    action_batches = _load_numbered_json(root, "action_batch_round_")
    action_batch_validations = _load_numbered_json(
        root,
        "action_batch_validation_round_",
    )
    tool_calls = _read_jsonl(root / "tool_calls.jsonl")
    target_input_path = root / "target_input.json"
    preflight_path = root / "preflight.json"
    state = ToolExecutionState(
        run_dir=root,
        target_input=_read_json(target_input_path) if target_input_path.is_file() else {},
        preflight=_read_json(preflight_path) if preflight_path.is_file() else {},
    )
    # Make saved verifier reports available before rebuilding the portfolio.
    # This lets the current code replay every route-proof-bank entry instead of
    # falling back to the single parent route embedded in an old proof object.
    state.artifacts.update(_load_existing_artifacts(root, blackboard))
    blackboard = _refresh_multisource_route_consensus(
        state=state,
        blackboard=blackboard,
    )
    artifacts = _load_existing_artifacts(root, blackboard)
    artifacts.update(state.artifacts)

    hypothesis_payload = compile_hypothesis_only_retrosynthesis_report(
        blackboard=blackboard,
        artifacts=artifacts,
    )
    hypothesis_artifact = _artifact(
        schema_version="hypothesis_only_retrosynthesis_report_artifact.v1",
        artifact_type="HypothesisOnlyRetrosynthesisReport",
        artifact_id=f"{case_id}:hypothesis_only_retrosynthesis_report",
        case_id=case_id,
        source="refresh_agentic_closeout_artifacts",
        input_refs=[str(blackboard_path)],
        evidence_refs=_evidence_refs(blackboard),
        payload=hypothesis_payload,
        artifact_ref=str(root / "hypothesis_only_retrosynthesis_report.json"),
    )
    _write_json(root / "hypothesis_only_retrosynthesis_report.json", hypothesis_artifact)
    artifacts["hypothesis_only_retrosynthesis_report"] = hypothesis_artifact

    execution_payload = compile_hypothesis_execution_report(
        blackboard=blackboard,
        hypothesis_report=hypothesis_artifact,
        artifacts=artifacts,
    )
    execution_artifact = _artifact(
        schema_version="hypothesis_execution_report_artifact.v1",
        artifact_type="HypothesisExecutionReport",
        artifact_id=f"{case_id}:hypothesis_execution_report",
        case_id=case_id,
        source="refresh_agentic_closeout_artifacts",
        input_refs=[str(blackboard_path), str(root / "hypothesis_only_retrosynthesis_report.json")],
        evidence_refs=_execution_evidence_refs(root, artifacts),
        payload=execution_payload,
        artifact_ref=str(root / "hypothesis_execution_report.json"),
    )
    _write_json(root / "hypothesis_execution_report.json", execution_artifact)
    artifacts["hypothesis_execution_report"] = execution_artifact

    final = emit_agentic_final_verdict(
        blackboard=blackboard,
        artifacts=artifacts,
        bundle={"case_id": case_id, "artifacts": artifacts, "preflight": {"accepted": True}},
    )
    final.artifact_refs = dict(blackboard.get("artifact_refs") or {})
    final_validation = _validate_agentic_final_verdict(final.to_dict(), blackboard=blackboard, validations=[])
    if not final_validation.get("accepted"):
        final = _downgrade_invalid_agentic_final_verdict(final, final_validation)
        final.artifact_refs = dict(blackboard.get("artifact_refs") or {})
        final_validation["corrected_final_verdict"] = final.to_dict()
        final_validation["corrected_validation"] = _validate_agentic_final_verdict(
            final.to_dict(),
            blackboard=blackboard,
            validations=[],
        )

    blackboard["final_verdict"] = final.to_dict()
    forest_result = write_route_forest_artifacts(
        blackboard,
        run_dir=root,
    )
    refs = blackboard.setdefault("artifact_refs", {})
    refs["explored_route_forest"] = str(forest_result["forest_path"])
    refs["route_forest_html"] = str(forest_result["html_path"])
    state.artifacts["explored_route_forest"] = dict(forest_result.get("forest") or {})
    _commit_route_closeout_revision(
        state=state,
        blackboard=blackboard,
        final_verdict=final,
        final_validation=final_validation,
    )
    final.artifact_refs = dict(blackboard.get("artifact_refs") or {})
    blackboard["final_verdict"] = final.to_dict()
    _write_json(blackboard_path, blackboard)
    _write_json(root / "final_verdict.json", final.to_dict())
    artifacts.update(state.artifacts)
    final_validation_artifact = _artifact(
        schema_version="agentic_final_verdict_validation_artifact.v1",
        artifact_type="AgenticFinalVerdictValidation",
        artifact_id=f"{case_id}:agentic_final_verdict_validation",
        case_id=case_id,
        source="refresh_agentic_closeout_artifacts",
        input_refs=[str(blackboard_path), str(root / "final_verdict.json")],
        evidence_refs=[],
        payload=final_validation,
        artifact_ref=str(root / "agentic_final_verdict_validation.json"),
    )
    _write_json(root / "agentic_final_verdict_validation.json", final_validation_artifact)
    artifacts["agentic_final_verdict_validation"] = final_validation_artifact
    state.artifacts.update(
        {
            "hypothesis_only_retrosynthesis_report": hypothesis_artifact,
            "hypothesis_execution_report": execution_artifact,
            "agentic_final_verdict_validation": final_validation_artifact,
        }
    )

    rebuilt_keys = {
        "hypothesis_only_retrosynthesis_report",
        "hypothesis_execution_report",
        "agentic_final_verdict_validation",
        "agent_blackboard_snapshot",
        "agentic_capability_audit",
        "agentic_run_audit",
    }
    _revalidate_existing_typed_artifacts(
        state,
        artifacts,
        excluded_keys=rebuilt_keys,
    )
    validation_rows = [
        _validate_and_record_typed_artifact(
            state,
            "hypothesis_only_retrosynthesis_report",
            hypothesis_artifact,
        ),
        _validate_and_record_typed_artifact(
            state,
            "hypothesis_execution_report",
            execution_artifact,
        ),
        _validate_and_record_typed_artifact(
            state,
            "agentic_final_verdict_validation",
            final_validation_artifact,
        ),
    ]

    snapshot_artifact = _record_agent_blackboard_snapshot_artifact(
        state=state,
        blackboard=blackboard,
    )
    state.artifacts["agent_blackboard_snapshot"] = snapshot_artifact
    snapshot_validation = _validate_and_record_typed_artifact(
        state,
        "agent_blackboard_snapshot",
        snapshot_artifact,
    )
    validation_rows.append(snapshot_validation)

    capability_artifact = _record_agentic_capability_audit_artifact(
        state=state,
        blackboard=blackboard,
        action_batches=action_batches,
        action_batch_validations=action_batch_validations,
        typed_validations=list(state.validations),
        tool_calls=tool_calls,
        final_verdict=final.to_dict(),
        final_validation=final_validation,
    )
    state.artifacts["agentic_capability_audit"] = capability_artifact
    capability_validation = _validate_and_record_typed_artifact(
        state,
        "agentic_capability_audit",
        capability_artifact,
    )
    validation_rows.append(capability_validation)

    run_audit_artifact = _record_agentic_run_audit_artifact(
        state=state,
        blackboard=blackboard,
        action_batches=action_batches,
        validations=action_batch_validations,
        typed_validations=list(state.validations),
        tool_calls=tool_calls,
        final_verdict=final.to_dict(),
    )
    state.artifacts["agentic_run_audit"] = run_audit_artifact
    run_audit_validation = _validate_and_record_typed_artifact(
        state,
        "agentic_run_audit",
        run_audit_artifact,
    )
    validation_rows.append(run_audit_validation)

    artifacts.update(state.artifacts)
    bundle_validations = [
        *action_batch_validations,
        final_validation,
        *state.validations,
    ]
    _refresh_artifact_bundle(
        root,
        artifacts,
        validations=bundle_validations,
        safety_flags=state.safety_flags,
    )

    summary = {
        "schema_version": "agentic_closeout_refresh_summary.v1",
        "accepted": bool(final_validation.get("accepted"))
        and all(row.get("accepted") for row in validation_rows),
        "run_dir": str(root),
        "case_id": case_id,
        "hypothesis_candidate_count": int(hypothesis_payload.get("candidate_precursor_count") or 0),
        "hypothesis_execution_status": str(execution_payload.get("route_status") or ""),
        "pending_candidate_count": int(execution_payload.get("pending_candidate_count") or 0),
        "pending_recursive_followup_count": int(execution_payload.get("pending_recursive_followup_count") or 0),
        "final_verdict": final.to_dict(),
        "typed_validations": validation_rows,
        "derived_diagnostics": {
            "agent_blackboard_snapshot_validation_accepted": snapshot_validation.get(
                "accepted"
            )
            is True,
            "agentic_capability_audit_accepted": (
                capability_artifact.get("payload") or {}
            ).get("accepted")
            is True,
            "agentic_run_audit_validation_accepted": run_audit_validation.get(
                "accepted"
            )
            is True,
        },
    }
    _write_json(root / "agentic_closeout_refresh_summary.json", summary)
    return summary


def _load_existing_artifacts(root: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    bundle_path = root / "artifact_bundle.json"
    if bundle_path.exists():
        bundle = _read_json(bundle_path)
        if isinstance(bundle.get("artifacts"), dict):
            artifacts.update(dict(bundle.get("artifacts") or {}))
    refs = dict(blackboard.get("artifact_refs") or {})
    for key, ref in refs.items():
        path = Path(str(ref))
        if not path.exists():
            continue
        if key in artifacts:
            continue
        try:
            artifacts[key] = _read_json(path)
        except Exception:
            continue
    for key, filename in {
        "route_expansion_subgoal_search": "route_expansion_subgoal_search_result.json",
        "route_proof_bundle": "r9_compile_objective_route_proof_route_proof_bundle_v1.json",
    }.items():
        path = root / filename
        if path.exists() and key not in artifacts:
            artifacts[key] = _read_json(path)
    return artifacts


def _revalidate_existing_typed_artifacts(
    state: ToolExecutionState,
    artifacts: dict[str, Any],
    *,
    excluded_keys: set[str],
) -> None:
    """Rebuild typed validation records instead of retaining stale closeout rows."""

    for key in sorted(artifacts):
        if key in excluded_keys:
            continue
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_type") or "") not in ARTIFACT_CLASSES:
            continue
        _validate_and_record_typed_artifact(state, key, artifact)


def _load_numbered_json(root: Path, prefix: str) -> list[dict[str, Any]]:
    paths = sorted(
        root.glob(f"{prefix}*.json"),
        key=lambda path: (_numeric_suffix(path, prefix), path.name),
    )
    return [_read_json(path) for path in paths]


def _numeric_suffix(path: Path, prefix: str) -> int:
    suffix = path.stem.removeprefix(prefix)
    try:
        return int(suffix)
    except ValueError:
        return 2**31 - 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _artifact(
    *,
    schema_version: str,
    artifact_type: str,
    artifact_id: str,
    case_id: str,
    source: str,
    input_refs: list[str],
    evidence_refs: list[str],
    payload: dict[str, Any],
    artifact_ref: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "case_id": case_id,
        "source": source,
        "input_refs": input_refs,
        "evidence_refs": evidence_refs,
        "validation_status": "accepted",
        "payload": payload,
        "artifact_ref": artifact_ref,
        "no_solved_claim": bool(payload.get("no_solved_claim") or payload.get("no_parent_solved_claim")),
    }


def _evidence_refs(blackboard: dict[str, Any]) -> list[str]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    refs = [str(item) for item in evidence.get("source_refs") or [] if str(item or "").strip()]
    refs.extend(str(row.get("source_ref") or "") for row in evidence.get("source_candidates") or [] if isinstance(row, dict))
    refs.extend(str(row.get("proposal_id") or "") for row in blackboard.get("retrosynthetic_proposals") or [] if isinstance(row, dict))
    refs.extend(str(row.get("task_id") or "") for row in blackboard.get("recursive_hypothesis_tasks") or [] if isinstance(row, dict))
    return _dedupe(refs)


def _execution_evidence_refs(root: Path, artifacts: dict[str, Any]) -> list[str]:
    refs = []
    route_expansion = artifacts.get("route_expansion_subgoal_search")
    if isinstance(route_expansion, dict):
        refs.append(str(route_expansion.get("artifact_ref") or root / "route_expansion_subgoal_search_result.json"))
    return _dedupe(refs)


def _refresh_artifact_bundle(
    root: Path,
    artifacts: dict[str, Any],
    *,
    validations: list[dict[str, Any]],
    safety_flags: list[str],
) -> None:
    bundle_path = root / "artifact_bundle.json"
    if not bundle_path.exists():
        return
    bundle = _read_json(bundle_path)
    bundle["artifacts"] = artifacts
    bundle["validations"] = validations
    retained_flags = [
        str(value)
        for value in bundle.get("safety_flags") or []
        if not str(value).startswith("typed_artifact_validation_failed:")
    ]
    bundle["safety_flags"] = _dedupe([*retained_flags, *safety_flags])
    _write_json(bundle_path, bundle)


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Saved agentic run directory containing agent_blackboard.json")
    args = parser.parse_args()
    print(json.dumps(refresh_agentic_closeout_artifacts(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
