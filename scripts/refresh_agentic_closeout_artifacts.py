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

from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.harness.agentic_blackboard_controller import (
    _validate_agentic_final_verdict,
    emit_agentic_final_verdict,
)
from cascade_planner.harness.hypothesis_execution_report import compile_hypothesis_execution_report
from cascade_planner.harness.hypothetical_retrosynthesis_report import (
    compile_hypothesis_only_retrosynthesis_report,
)


def refresh_agentic_closeout_artifacts(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    blackboard_path = root / "agent_blackboard.json"
    if not blackboard_path.exists():
        raise FileNotFoundError(f"agent_blackboard.json not found: {blackboard_path}")
    blackboard = _read_json(blackboard_path)
    case_id = str(blackboard.get("case_id") or root.name)
    artifacts = _load_existing_artifacts(root, blackboard)

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
    _write_json(root / "final_verdict.json", final.to_dict())
    _write_json(root / "agentic_final_verdict_validation.json", final_validation_artifact)
    artifacts["agentic_final_verdict_validation"] = final_validation_artifact
    _refresh_artifact_bundle(root, artifacts)

    validation_rows = [
        validate_typed_artifact(hypothesis_artifact),
        validate_typed_artifact(execution_artifact),
        validate_typed_artifact(final_validation_artifact),
    ]
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


def _refresh_artifact_bundle(root: Path, artifacts: dict[str, Any]) -> None:
    bundle_path = root / "artifact_bundle.json"
    if not bundle_path.exists():
        return
    bundle = _read_json(bundle_path)
    bundle["artifacts"] = artifacts
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
