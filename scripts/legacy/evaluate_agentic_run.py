#!/usr/bin/env python3
"""Evaluate one or compare two saved agentic retrosynthesis runs.

The evaluator intentionally depends only on the Python standard library.  It
reads saved JSON artifacts without initializing RDKit, Torch, ChemEnzy, or the
web runtime.  A solved claim is never inferred from an advisory branch: only
the repository's deterministic parent-proof predicate can make
``strict_solved`` true.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cascade_planner.source_locators import canonical_traceable_source_ref  # noqa: E402


SCHEMA_VERSION = "agentic_run_evaluation.v1"
COMPARISON_SCHEMA_VERSION = "agentic_run_comparison.v1"
_BRANCH_SEMANTIC_FIELDS = ("solved", "executable", "advisory_only", "not_parent_route_proof")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_true(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


class ArtifactReader:
    """Small tolerant JSON reader that records missing and invalid artifacts."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.status: dict[str, str] = {}
        self.errors: list[str] = []

    def load(self, relative_path: str, *, optional: bool = True) -> Any:
        path = self.run_dir / relative_path
        if not path.is_file():
            self.status[relative_path] = "missing"
            if not optional:
                self.errors.append(f"missing:{relative_path}")
            return None
        try:
            value = _loads_json_bytes(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.status[relative_path] = "invalid"
            self.errors.append(f"invalid_json:{relative_path}:{type(exc).__name__}:{exc}")
            return None
        self.status[relative_path] = "loaded"
        return value

    def load_path(self, path: Path) -> Any:
        label = _relative_label(path, self.run_dir)
        if not path.is_file():
            return None
        try:
            return _loads_json_bytes(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.errors.append(f"invalid_json:{label}:{type(exc).__name__}:{exc}")
            return None


def _loads_json_bytes(raw: bytes) -> Any:
    """Decode normal UTF-8 JSON plus BOM/UTF-16 and legacy GB text safely."""
    decode_errors: list[Exception] = []
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeError as exc:
            decode_errors.append(exc)
        except json.JSONDecodeError:
            # A successful decode with invalid JSON should not be reinterpreted
            # with an unrelated encoding, except for the UTF-8 -> UTF-16 BOM
            # case handled by the next iteration.
            if encoding != "utf-8-sig" or raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            raise
    if decode_errors:
        raise decode_errors[-1]
    return json.loads(raw.decode("utf-8", errors="replace"))


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _load_strict_proof_predicate() -> tuple[Callable[..., bool] | None, str, str]:
    """Load the pure deterministic predicate without importing the controller."""
    repo_root = Path(__file__).resolve().parents[2]
    inserted = False
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        inserted = True
    try:
        from cascade_planner.legacy.harness_runtime.parent_route_proof import (
            is_solved_parent_route_proof,
        )

        return (
            is_solved_parent_route_proof,
            "cascade_planner.legacy.harness_runtime.parent_route_proof."
            "is_solved_parent_route_proof",
            "",
        )
    except Exception as exc:  # pragma: no cover - exercised only in partial exports
        return None, "unavailable", f"{type(exc).__name__}:{exc}"
    finally:
        if inserted and sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)


def _evaluate_closeout_revision(run_dir: Path) -> dict[str, Any]:
    """Validate an immutable closeout when present, preserving old-run reads."""
    pointer = run_dir / ".autoplanner" / "closeout" / "latest.json"
    if not pointer.is_file():
        return {
            "schema_version": "closeout_revision_evaluation.v1",
            "present": False,
            "accepted": None,
            "compatibility_mode": True,
            "route_projection_trusted": True,
            "reasons": ["closeout_latest_pointer_missing_legacy_run"],
        }
    repo_root = Path(__file__).resolve().parents[2]
    inserted = False
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        inserted = True
    manifest: dict[str, Any] = {}
    try:
        from cascade_planner.legacy.runtime.artifact_revision import (
            load_latest_closeout_manifest,
            validate_latest_closeout_revision,
        )

        validation = dict(validate_latest_closeout_revision(run_dir))
        manifest = (
            load_latest_closeout_manifest(run_dir)
            if validation.get("accepted") is True
            else {}
        )
    except Exception as exc:  # pragma: no cover - partial source exports
        validation = {
            "schema_version": "closeout_revision_validation.v1",
            "present": True,
            "accepted": False,
            "reasons": [f"closeout_validator_unavailable:{type(exc).__name__}:{exc}"],
        }
    finally:
        if inserted and sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)
    accepted = validation.get("accepted") is True
    content_paths = {
        str(row.get("artifact_id") or ""): str(row.get("content_path") or "")
        for row in manifest.get("artifacts") or []
        if isinstance(row, dict) and str(row.get("artifact_id") or "")
    }
    return {
        **validation,
        "schema_version": "closeout_revision_evaluation.v1",
        "present": True,
        "compatibility_mode": False,
        "route_projection_trusted": accepted,
        "authoritative_artifact_content_paths": content_paths,
    }


def evaluate_run(run_dir: str | Path) -> dict[str, Any]:
    """Return a schema-stable, fail-closed evaluation of ``run_dir``."""
    root = Path(run_dir).expanduser().resolve()
    reader = ArtifactReader(root)

    target_input = _as_dict(reader.load("target_input.json"))
    final_verdict = _as_dict(reader.load("final_verdict.json"))
    blackboard = _as_dict(reader.load("agent_blackboard.json"))
    compatibility_final_verdict = dict(final_verdict)
    compatibility_parent_proof = _as_dict(blackboard.get("parent_route_proof"))
    run_audit_artifact = _as_dict(reader.load("agentic_run_audit.json"))
    run_audit = _as_dict(run_audit_artifact.get("payload") or run_audit_artifact)
    team_report = _as_dict(reader.load("codex_retrosynthesis_team/team_report.json"))
    coordinator = _as_dict(reader.load("codex_retrosynthesis_team/coordinator_run_record.json"))
    coordinator_task = _as_dict(reader.load("codex_retrosynthesis_team/coordinator_task.json"))
    runtime_summary = _as_dict(reader.load("codex_retrosynthesis_team/runtime_summary.json"))
    guided = _as_dict(reader.load("guided_chemenzy_result.json"))
    guided_verifier = _as_dict(reader.load("guided_route_verifier_report.json"))
    capability_artifact = _as_dict(reader.load("agentic_capability_audit.json"))
    route_forest = _as_dict(reader.load("explored_route_forest.json"))
    closeout_revision = _evaluate_closeout_revision(root)
    authoritative_proof: dict[str, Any] = {}
    compatibility_semantic_drift: list[str] = []
    if closeout_revision.get("accepted") is True:
        content_paths = _as_dict(
            closeout_revision.get("authoritative_artifact_content_paths")
        )

        def load_cas(artifact_id: str) -> dict[str, Any]:
            raw = str(content_paths.get(artifact_id) or "")
            if not raw:
                return {}
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            return _as_dict(reader.load_path(path))

        proof_snapshot = load_cas("parent_route_proof_snapshot")
        verdict_core = load_cas("final_verdict_core")
        cas_forest = load_cas("explored_route_forest")
        authoritative_proof = _as_dict(proof_snapshot.get("proof"))
        final_verdict = _as_dict(verdict_core.get("verdict"))
        if cas_forest:
            route_forest = cas_forest
        if compatibility_parent_proof != authoritative_proof:
            compatibility_semantic_drift.append("agent_blackboard_parent_proof_drift")
        compatibility_core = dict(compatibility_final_verdict)
        compatibility_core.pop("artifact_refs", None)
        compatibility_core.pop("artifact_digest_refs", None)
        if compatibility_core != final_verdict:
            compatibility_semantic_drift.append("final_verdict_compatibility_drift")
        closeout_revision["decision_authority"] = "content_addressed_closeout_objects"
    elif closeout_revision.get("present"):
        # A fixed-name forest that no longer matches its active consensus/graph
        # revision is quarantined instead of being evaluated as current truth.
        route_forest = {}
    closeout_revision["compatibility_semantic_drift"] = compatibility_semantic_drift

    if not guided_verifier:
        guided_verifier = _as_dict(
            guided.get("raw_route_verifier") or guided.get("backend_raw_route_verifier")
        )

    target = _evaluate_target(target_input, blackboard, route_forest, final_verdict)
    if authoritative_proof:
        parent_proof = authoritative_proof
        parent_proof_source = "CAS:parent_route_proof_snapshot"
    else:
        parent_proof, parent_proof_source = _select_parent_proof(blackboard, run_audit)
    strict_predicate, strict_evaluator, strict_error = _load_strict_proof_predicate()
    proof_report = _evaluate_parent_proof(
        parent_proof,
        predicate=strict_predicate,
        evaluator_name=strict_evaluator,
        evaluator_error=strict_error,
        source_artifact=parent_proof_source,
        expected_target_smiles=str(target.get("smiles") or ""),
    )
    verdict_report = _evaluate_final_verdict(final_verdict, proof_report)
    team = _evaluate_team(team_report, coordinator, coordinator_task, runtime_summary)
    planner = _evaluate_planner(blackboard, root)
    evidence, visual, process = _evaluate_evidence(
        blackboard,
        route_forest,
        reader,
        target_input=target_input,
    )
    guided_report = _evaluate_guided(guided, guided_verifier)
    capability = _evaluate_capability(capability_artifact)
    forest = _evaluate_route_forest(route_forest, team, proof_report)

    warnings = list(reader.errors)
    if verdict_report["claimed_solved"] and proof_report.get("strict_solved") is not True:
        warnings.append("final_solved_claim_not_supported_by_strict_parent_proof")
    semantics = _as_dict(forest.get("branch_semantics"))
    if semantics.get("advisory_claimed_solved_count"):
        warnings.append("advisory_branch_claimed_solved")
    quarantine = _as_dict(forest.get("rejected_team_consensus_quarantine"))
    if quarantine.get("required") and quarantine.get("passed") is not True:
        warnings.append("rejected_team_consensus_not_quarantined")
    if route_forest and not forest.get("primary_branch_id"):
        warnings.append("route_forest_has_no_primary_branch")
    primary_semantics = _as_dict(forest.get("primary_branch_semantics"))
    primary_selection = _as_dict(forest.get("primary_selection"))
    if (
        forest.get("primary_branch_id")
        and proof_report.get("strict_solved") is not True
        and (
            primary_semantics.get("advisory_only") is True
            or str(primary_selection.get("status") or "") == "advisory"
        )
    ):
        warnings.append("route_forest_primary_is_advisory")
    if evidence.get("resolved_structures_invalid_target_shortcuts"):
        warnings.append("invalid_target_identity_shortcut_excluded")
    if closeout_revision.get("present") and closeout_revision.get("accepted") is not True:
        warnings.append("closeout_revision_invalid_route_projection_quarantined")
        warnings.extend(
            f"closeout_revision:{reason}"
            for reason in _string_list(closeout_revision.get("reasons"))
        )
    if closeout_revision.get("compatibility_projection_drift") is True:
        warnings.append("closeout_compatibility_projection_drift_using_cas_authority")
    warnings.extend(
        f"closeout_compatibility:{reason}"
        for reason in compatibility_semantic_drift
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(root),
        "run_exists": root.is_dir(),
        "target": target,
        "final_verdict": verdict_report,
        "parent_route_proof": proof_report,
        "codex_team": team,
        "planner": planner,
        "evidence_counts": evidence,
        "visual_evidence": visual,
        "process_evidence": process,
        "guided_verifier": guided_report,
        "capability_audit": capability,
        "closeout_revision": closeout_revision,
        "route_forest": forest,
        "artifact_status": dict(sorted(reader.status.items())),
        "warnings": sorted(set(warnings)),
    }


def _evaluate_target(
    target_input: Mapping[str, Any],
    blackboard: Mapping[str, Any],
    route_forest: Mapping[str, Any],
    final_verdict: Mapping[str, Any],
) -> dict[str, Any]:
    profile = _as_dict(blackboard.get("target_profile"))
    forest_target = _as_dict(route_forest.get("target"))
    return {
        "case_id": str(
            target_input.get("case_id")
            or blackboard.get("case_id")
            or route_forest.get("case_id")
            or final_verdict.get("case_id")
            or ""
        ),
        "name": str(
            target_input.get("target_name")
            or profile.get("target_name")
            or forest_target.get("name")
            or ""
        ),
        "smiles": str(
            target_input.get("target_smiles")
            or profile.get("target_smiles")
            or profile.get("isomeric_smiles")
            or profile.get("canonical_smiles")
            or forest_target.get("smiles")
            or ""
        ),
        "family_hint": str(
            target_input.get("family_hint")
            or profile.get("family_hint")
            or forest_target.get("family_hint")
            or ""
        ),
        "inchi_key": str(profile.get("inchi_key") or ""),
    }


def _select_parent_proof(
    blackboard: Mapping[str, Any], run_audit: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    direct = blackboard.get("parent_route_proof")
    if isinstance(direct, dict) and direct:
        return direct, "agent_blackboard.json:parent_route_proof"
    audited = run_audit.get("parent_route_proof")
    if isinstance(audited, dict) and audited:
        return audited, "agentic_run_audit.json:payload.parent_route_proof"
    return {}, ""


def _evaluate_parent_proof(
    proof: Mapping[str, Any],
    *,
    predicate: Callable[..., bool] | None,
    evaluator_name: str,
    evaluator_error: str,
    source_artifact: str,
    expected_target_smiles: str,
) -> dict[str, Any]:
    present = bool(proof)
    strict_solved: bool | None = None
    evaluation_status = "unavailable"
    if predicate is not None:
        strict_solved = bool(
            predicate(
                dict(proof),
                expected_target_smiles=expected_target_smiles,
            )
        )
        evaluation_status = "evaluated"
    return {
        "present": present,
        "source_artifact": source_artifact,
        "schema_version": str(proof.get("schema_version") or ""),
        "claimed_accepted": proof.get("accepted") is True,
        "claimed_solved": proof.get("solved") is True,
        "route_status": str(proof.get("route_status") or ""),
        "proof_mode": str(proof.get("proof_mode") or ""),
        "strict_solved": strict_solved,
        "strict_evaluation_status": evaluation_status,
        "strict_evaluator": evaluator_name,
        "strict_evaluator_error": evaluator_error,
        "proof_clauses": _as_dict(proof.get("proof_clauses")),
        "source_policy": _as_dict(proof.get("source_policy")),
        "reasons": _string_list(proof.get("reasons")),
        "note": (
            "strict_solved comes only from the deterministic repository predicate"
            if predicate is not None
            else "strict proof predicate unavailable; claimed fields are reported but not trusted"
        ),
    }


def _evaluate_final_verdict(
    verdict: Mapping[str, Any], proof_report: Mapping[str, Any]
) -> dict[str, Any]:
    claimed_solved = verdict.get("solved") is True
    strict = proof_report.get("strict_solved")
    consistent = None if strict is None else claimed_solved is strict
    return {
        "present": bool(verdict),
        "verdict": str(verdict.get("verdict") or ""),
        "route_status": str(verdict.get("route_status") or ""),
        "claimed_solved": claimed_solved,
        "stock_audit_passed": verdict.get("stock_audit_passed") is True,
        "reasons": _string_list(verdict.get("reasons")),
        "consistent_with_strict_parent_proof": consistent,
        "strictly_supported_solved": claimed_solved and strict is True,
    }


def _evaluate_team(
    team_report: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    coordinator_task: Mapping[str, Any],
    runtime_summary: Mapping[str, Any],
) -> dict[str, Any]:
    coordinator_view = _as_dict(team_report.get("coordinator"))
    event_summary = _as_dict(coordinator_view.get("event_summary"))
    tool_calls = _dict_rows(coordinator.get("tool_calls"))
    tool_counts = Counter(str(row.get("tool") or "unknown") for row in tool_calls)
    allowed_tools = set(_string_list(coordinator_task.get("allowed_tools")))
    unauthorized = sorted(
        tool
        for tool in tool_counts
        if tool != "unknown" and allowed_tools and tool not in allowed_tools
    )
    output_validation = _as_dict(coordinator.get("output_validation"))
    validation_reasons = _string_list(output_validation.get("reasons"))
    explicit_call_violations = sorted(
        {
            str(row.get("tool") or row.get("call_id") or "unknown")
            for row in tool_calls
            if row.get("allowed") is False or row.get("policy_violation") is True
        }
    )

    observed_children = _dict_rows(coordinator_view.get("observed_child_agents"))
    runtime_children = _dict_rows(runtime_summary.get("children"))
    required_roles = _string_list(coordinator_view.get("required_child_roles"))
    if not required_roles:
        required_roles = _string_list(coordinator_task.get("child_roles"))
    observed_roles = sorted(
        {
            str(row.get("role") or "")
            for row in [*observed_children, *runtime_children]
            if str(row.get("role") or "").strip()
        }
    )
    runtime_states = Counter(str(row.get("state") or "unknown") for row in runtime_children)
    spawn_count = _safe_int(event_summary.get("child_agent_spawn_count"), -1)
    if spawn_count < 0:
        spawn_count = len(observed_children) or len(runtime_children)
    completed_count = _safe_int(event_summary.get("child_agent_completed_count"), -1)
    if completed_count < 0:
        completed_count = sum(1 for row in runtime_children if row.get("state") == "succeeded")

    return {
        "present": bool(team_report or coordinator or runtime_summary),
        "accepted": team_report.get("accepted") is True,
        "reasons": _string_list(team_report.get("reasons")),
        "coordinator_status": str(
            coordinator.get("status") or coordinator_view.get("status") or ""
        ),
        "coordinator_output_validation_accepted": output_validation.get("accepted") is True,
        "coordinator_output_validation_reasons": validation_reasons,
        "child_spawn_count": spawn_count,
        "child_completion_count": completed_count,
        "observed_child_count": len(observed_children),
        "runtime_child_count": len(runtime_children),
        "runtime_child_state_counts": dict(sorted(runtime_states.items())),
        "runtime_consistent": runtime_summary.get("consistent"),
        "required_child_roles": required_roles,
        "observed_child_roles": observed_roles,
        "missing_child_roles": sorted(set(required_roles) - set(observed_roles)),
        "tool_calls": {
            "count": len(tool_calls),
            "by_tool": dict(sorted(tool_counts.items())),
            "allowed_tools": sorted(allowed_tools),
            "unauthorized_tools": sorted(set(unauthorized + explicit_call_violations)),
            "validation_violation_reasons": [
                reason for reason in validation_reasons if "tool" in reason.lower()
            ],
            "has_violation": bool(
                unauthorized
                or explicit_call_violations
                or any("tool" in reason.lower() for reason in validation_reasons)
            ),
        },
    }


def _evaluate_planner(blackboard: Mapping[str, Any], root: Path) -> dict[str, Any]:
    history = _dict_rows(blackboard.get("planner_history"))
    actions = _dict_rows(blackboard.get("action_history"))
    if not history and root.is_dir():
        # Old runs may predate planner_history but still have round artifacts.
        round_numbers = sorted(
            {
                _round_from_name(path.stem)
                for path in root.glob("action_batch_round_*.json")
                if _round_from_name(path.stem) is not None
            }
        )
        history = [{"round_index": number} for number in round_numbers]

    rounds: list[dict[str, Any]] = []
    fallback_rounds: list[int] = []
    rejected_rounds: list[int] = []
    attempted_rounds = 0
    for index, row in enumerate(history, start=1):
        round_index = _safe_int(row.get("round_index"), index)
        codex = _as_dict(row.get("codex_action_planner"))
        fallback = codex.get("fallback_used") is True or row.get("fallback_used") is True
        attempted = codex.get("attempted") is True or row.get("codex_attempted") is True
        validation_accepted = row.get("validation_accepted")
        if fallback:
            fallback_rounds.append(round_index)
        if attempted:
            attempted_rounds += 1
        if validation_accepted is False:
            rejected_rounds.append(round_index)
        rounds.append(
            {
                "round_index": round_index,
                "mode": str(row.get("mode") or ""),
                "action_count": _safe_int(row.get("action_count")),
                "action_types": _string_list(row.get("action_types")),
                "codex_attempted": attempted,
                "fallback_used": fallback,
                "fallback_reason": str(codex.get("fallback_reason") or ""),
                "validation_accepted": validation_accepted,
                "validation_reasons": _string_list(row.get("validation_reasons")),
            }
        )

    action_types = Counter(str(row.get("action_type") or "unknown") for row in actions)
    statuses = Counter(str(row.get("status") or "unknown") for row in actions)
    useful = _count_true(actions, "useful_artifact")
    stale = _count_true(actions, "stale")
    changed = sum(1 for row in actions if _string_list(row.get("changed_blackboard_fields")))
    planned_action_count = sum(row["action_count"] for row in rounds)
    if not planned_action_count and actions:
        planned_action_count = len(actions)
    return {
        "round_count": len(history),
        "codex_attempted_round_count": attempted_rounds,
        "fallback_round_count": len(fallback_rounds),
        "fallback_rounds": fallback_rounds,
        "validation_rejected_rounds": rejected_rounds,
        "planned_action_count": planned_action_count,
        "rounds": rounds,
        "transitions": {
            "count": len(actions),
            "useful_count": useful,
            "stale_count": stale,
            "useful_non_stale_count": sum(
                1 for row in actions if row.get("useful_artifact") is True and row.get("stale") is not True
            ),
            "changed_blackboard_count": changed,
            "status_counts": dict(sorted(statuses.items())),
            "action_type_counts": dict(sorted(action_types.items())),
        },
    }


def _round_from_name(stem: str) -> int | None:
    suffix = stem.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _evaluate_evidence(
    blackboard: Mapping[str, Any],
    route_forest: Mapping[str, Any],
    reader: ArtifactReader,
    *,
    target_input: Mapping[str, Any] | None = None,
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    literature = _as_dict(blackboard.get("literature_evidence"))
    forest_index = _as_dict(route_forest.get("evidence_index"))
    list_keys = (
        "source_candidates",
        "source_refs",
        "source_lifecycle",
        "scout_attempts",
        "pdf_structure_evidence",
        "exact_rows",
        "resolved_structures",
        "visual_chains",
        "process_evidence_rows",
        "structure_resolution_attempts",
        "structure_resolution_tasks",
        "terminal_candidates",
    )
    counts: dict[str, int] = {}
    for key in list_keys:
        value = literature.get(key)
        if not isinstance(value, list):
            value = forest_index.get(key)
        counts[key] = len(value) if isinstance(value, list) else 0
    resolved_rows = _dict_rows(literature.get("resolved_structures"))
    invalid_shortcuts = [
        row
        for row in resolved_rows
        if row.get("target_identity_shortcut") is True
        and not _resolved_shortcut_matches_target(
            row,
            target_input=target_input or {},
            blackboard=blackboard,
        )
    ]
    counts["resolved_structures_raw"] = len(resolved_rows)
    counts["resolved_structures_invalid_target_shortcuts"] = len(invalid_shortcuts)
    counts["resolved_structures"] = len(resolved_rows) - len(invalid_shortcuts)
    forest_counts = _as_dict(route_forest.get("counts"))
    for key in ("exact_rows", "visual_chains", "process_evidence_rows"):
        if counts[key] == 0:
            counts[key] = _safe_int(forest_counts.get(key))

    pdf_rows = _dict_rows(literature.get("pdf_structure_evidence"))
    counts["pdf_structure_evidence_accepted"] = _count_true(pdf_rows, "accepted")
    counts["pdf_rendered_pages"] = sum(
        _safe_int(_as_dict(row.get("summary")).get("rendered_page_count")) for row in pdf_rows
    )
    source_rows = _dict_rows(literature.get("source_candidates"))
    if source_rows:
        real_source_rows = [row for row in source_rows if _source_candidate_has_real_source(row)]
        placeholder_source_rows = [
            row for row in source_rows if not _source_candidate_has_real_source(row)
        ]
    else:
        source_rows = _dict_rows(forest_index.get("source_candidates"))
        real_source_rows = _dict_rows(forest_index.get("real_source_candidates"))
        if not real_source_rows:
            real_source_rows = [
                row for row in source_rows if _source_candidate_has_real_source(row)
            ]
        placeholder_source_rows = _dict_rows(forest_index.get("placeholder_candidates"))
        if not placeholder_source_rows:
            placeholder_source_rows = [
                row for row in source_rows if not _source_candidate_has_real_source(row)
            ]
    counts["source_candidate_records"] = len(source_rows)
    counts["source_candidates"] = len(source_rows)
    counts["real_source_candidates"] = len(real_source_rows)
    counts["placeholder_candidates"] = len(placeholder_source_rows)
    counts["source_documents"] = sum(
        _source_document_count(row) for row in real_source_rows
    )
    source_refs = _string_list(literature.get("source_refs"))
    real_source_refs = [ref for ref in source_refs if not _placeholder_source_ref(ref)]
    counts["source_ref_records"] = len(source_refs)
    counts["source_refs"] = len(source_refs)
    counts["real_source_refs"] = len(real_source_refs)
    counts["placeholder_source_refs"] = len(source_refs) - len(real_source_refs)

    visual_rows = _dict_rows(literature.get("visual_chains"))
    if not visual_rows:
        visual_rows = _dict_rows(forest_index.get("visual_chains"))
    if not visual_rows:
        visual_rows = _visual_fallback_rows(reader)
    visual_reasons = Counter(
        reason for row in visual_rows for reason in _string_list(row.get("reasons"))
    )
    accepted_visual = _count_true(visual_rows, "accepted")
    rejected_visual = sum(1 for row in visual_rows if row.get("accepted") is False)
    visual_report = {
        "total": len(visual_rows),
        "accepted": accepted_visual,
        "rejected": rejected_visual,
        "unknown": len(visual_rows) - accepted_visual - rejected_visual,
        "exact_ready": _count_true(visual_rows, "exact_ready"),
        "exploratory_accepted": _count_true(visual_rows, "exploratory_accepted"),
        "step_count": sum(
            _safe_int(row.get("step_count"), _safe_int(row.get("candidate_step_count")))
            for row in visual_rows
        ),
        "rejection_reasons": dict(sorted(visual_reasons.items())),
    }

    process_rows = _dict_rows(literature.get("process_evidence_rows"))
    if not process_rows:
        process_rows = _dict_rows(forest_index.get("process_evidence_rows"))
    process_types = Counter(str(row.get("process_type") or "unspecified") for row in process_rows)
    process_report = {
        "total": len(process_rows),
        "types": dict(sorted(process_types.items())),
        "explicitly_not_parent_route_proof": _count_true(process_rows, "not_parent_route_proof"),
    }
    return counts, visual_report, process_report


def _source_document_count(source: Mapping[str, Any]) -> int:
    documents = source.get("documents")
    if isinstance(documents, list):
        return len(documents)
    return 1 if _source_candidate_has_real_source(source) else 0


def _source_candidate_has_real_source(source: Mapping[str, Any]) -> bool:
    if bool(source.get("placeholder_only")):
        return False
    if str(source.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    if str(source.get("source_type") or "").strip().lower() == "placeholder_query":
        return False
    if str(source.get("source_discovery_mode") or "").strip().lower() == "placeholder":
        return False
    locators = [
        source.get("doi"),
        source.get("pii"),
        source.get("url"),
        source.get("source_ref"),
    ]
    local_path = str(source.get("local_pdf") or source.get("pdf_path") or "").strip()
    if local_path:
        locators.append(
            local_path if local_path.lower().startswith("local_pdf:") else f"local_pdf:{local_path}"
        )
    return any(canonical_traceable_source_ref(value) for value in locators)


def _placeholder_source_ref(value: Any) -> bool:
    return not bool(canonical_traceable_source_ref(value))


def _resolved_shortcut_matches_target(
    row: Mapping[str, Any],
    *,
    target_input: Mapping[str, Any],
    blackboard: Mapping[str, Any],
) -> bool:
    profile = _as_dict(blackboard.get("target_profile"))
    aliases = target_input.get("target_aliases") or target_input.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    target_labels = [
        target_input.get("target_name"),
        target_input.get("name"),
        target_input.get("case_id"),
        profile.get("target_name"),
        profile.get("name"),
        blackboard.get("case_id"),
        *aliases,
    ]
    target_keys = {_identity_label(value) for value in target_labels if _identity_label(value)}
    observed = _identity_label(row.get("label"))
    if observed in target_keys:
        return True
    without_number = re.sub(r"\s+(?:compound\s+)?\d+[a-z]?$", "", observed).strip()
    return bool(without_number and without_number in target_keys)


def _identity_label(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _visual_fallback_rows(reader: ArtifactReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(reader.run_dir.glob("*_visual_literature_chain_extraction_result_v1.json"))
    if not paths:
        path = reader.run_dir / "visual_literature_chain_extraction_result.json"
        paths = [path] if path.is_file() else []
    for path in paths:
        value = reader.load_path(path)
        payload = _as_dict(_as_dict(value).get("payload") or value)
        if payload:
            rows.append(payload)
    return rows


def _evaluate_guided(
    guided: Mapping[str, Any], verifier: Mapping[str, Any]
) -> dict[str, Any]:
    failure_reasons = Counter(
        str(row.get("reason") or "unspecified")
        for row in _dict_rows(verifier.get("failure_events"))
    )
    return {
        "present": bool(guided or verifier),
        "guided_claimed_accepted": guided.get("accepted") is True,
        "guided_claimed_solved": guided.get("solved") is True,
        "guided_route_status": str(guided.get("route_status") or ""),
        "verifier_accepted": verifier.get("accepted") is True,
        "verifier_route_status": str(verifier.get("route_status") or ""),
        "target_match": (
            verifier.get("target_match") is True
            or _as_dict(verifier.get("target_equivalence_audit")).get("target_match") is True
        ),
        "route_count": _safe_int(verifier.get("route_count")),
        "accepted_route_count": _safe_int(verifier.get("accepted_route_count")),
        "rejected_route_count": _safe_int(verifier.get("rejected_route_count")),
        "best_route_rank": verifier.get("best_route_rank"),
        "route_proof_blocked": verifier.get("route_proof_blocked") is True,
        "reasons": _string_list(verifier.get("reasons") or guided.get("reasons")),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
    }


def _evaluate_capability(artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(artifact.get("payload") or artifact)
    checks = _dict_rows(payload.get("requirement_checks"))
    failed_checks = [
        str(row.get("requirement_id") or "unknown")
        for row in checks
        if row.get("accepted") is False
    ]
    failed_requirements = _string_list(payload.get("failed_requirements")) or failed_checks
    return {
        "present": bool(artifact),
        "accepted": payload.get("accepted") is True,
        "artifact_validation_status": str(artifact.get("validation_status") or ""),
        "audit_authority": str(payload.get("audit_authority") or ""),
        "failed_requirements": failed_requirements,
        "warning_requirements": _string_list(payload.get("warning_requirements")),
        "requirement_count": len(checks),
        "accepted_requirement_count": _count_true(checks, "accepted"),
        "failed_requirement_count": max(
            len(failed_requirements),
            sum(1 for row in checks if row.get("accepted") is False),
        ),
        "observed_action_types": _string_list(payload.get("observed_action_types")),
        "tool_call_count": _safe_int(payload.get("tool_call_count")),
        "no_solved_claim": payload.get("no_solved_claim") is True,
    }


def _evaluate_route_forest(
    forest: Mapping[str, Any],
    team: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    branches = _dict_rows(forest.get("branches"))
    kinds = Counter(str(row.get("kind") or "unspecified") for row in branches)
    synthesis_classes = Counter(
        str(row.get("synthesis_class") or "unspecified") for row in branches
    )
    missing_fields = {
        key: sum(1 for row in branches if not isinstance(row.get(key), bool))
        for key in _BRANCH_SEMANTIC_FIELDS
    }
    explicit_rows = [
        row
        for row in branches
        if all(isinstance(row.get(key), bool) for key in _BRANCH_SEMANTIC_FIELDS)
    ]
    claimed_solved = [row for row in branches if row.get("solved") is True]
    advisory_solved = [
        row
        for row in claimed_solved
        if row.get("advisory_only") is True or row.get("not_parent_route_proof") is True
    ]
    strictly_usable = [
        row
        for row in claimed_solved
        if proof.get("strict_solved") is True
        and row.get("executable") is True
        and row.get("advisory_only") is False
        and row.get("not_parent_route_proof") is False
    ]

    primary_selection = _as_dict(forest.get("primary_selection"))
    primary_id = str(
        forest.get("primary_branch_id") or primary_selection.get("primary_branch_id") or ""
    )
    primary = next((row for row in branches if str(row.get("branch_id") or "") == primary_id), {})

    consensus_view = _as_dict(forest.get("route_consensus"))
    consensus_branches = [row for row in branches if row.get("kind") == "route_consensus"]
    rejected_team = team.get("present") is True and team.get("accepted") is not True
    consensus_source_present = bool(
        consensus_view
        and (
            consensus_view.get("source_schema_version") == "route_consensus.v1"
            or consensus_view.get("quarantined") is True
            or consensus_view.get("available") is True
            or consensus_view.get("proposals")
        )
    )
    quarantine_required = rejected_team and consensus_source_present
    quarantined = consensus_view.get("quarantined") is True
    quarantine_passed: bool | None
    if quarantine_required:
        quarantine_passed = quarantined and not consensus_branches
    else:
        quarantine_passed = None

    counts = _as_dict(forest.get("counts"))
    return {
        "present": bool(forest),
        "branch_count": len(branches),
        "node_count": len(forest.get("nodes")) if isinstance(forest.get("nodes"), list) else _safe_int(counts.get("nodes")),
        "step_count": len(forest.get("steps")) if isinstance(forest.get("steps"), list) else _safe_int(counts.get("steps")),
        "branch_kinds": dict(sorted(kinds.items())),
        "synthesis_classes": dict(sorted(synthesis_classes.items())),
        "primary_branch_id": primary_id,
        "primary_exists": bool(primary),
        "primary_selection": primary_selection,
        "primary_branch_semantics": {
            key: primary.get(key) for key in _BRANCH_SEMANTIC_FIELDS
        }
        if primary
        else {},
        "branch_semantics": {
            "explicit_all_fields_count": len(explicit_rows),
            "missing_field_counts": missing_fields,
            "claimed_solved_count": len(claimed_solved),
            "claimed_executable_count": _count_true(branches, "executable"),
            "advisory_count": _count_true(branches, "advisory_only"),
            "advisory_claimed_solved_count": len(advisory_solved),
            "strictly_usable_solved_count": len(strictly_usable),
            "strictly_usable_solved_branch_ids": [
                str(row.get("branch_id") or "") for row in strictly_usable
            ],
            "note": "advisory branches never contribute to strictly_usable_solved_count",
        },
        "rejected_team_consensus_quarantine": {
            "required": quarantine_required,
            "team_rejected": rejected_team,
            "consensus_source_present": consensus_source_present,
            "quarantined": quarantined,
            "leaked_consensus_branch_count": len(consensus_branches) if rejected_team else 0,
            "passed": quarantine_passed,
            "reasons": _string_list(consensus_view.get("reasons")),
        },
    }


def compare_reports(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Return two full reports plus a concise, direction-neutral metric delta."""
    metric_paths: dict[str, Sequence[str]] = {
        "final_claimed_solved": ("final_verdict", "claimed_solved"),
        "strict_parent_proof_solved": ("parent_route_proof", "strict_solved"),
        "team_accepted": ("codex_team", "accepted"),
        "child_spawn_count": ("codex_team", "child_spawn_count"),
        "child_completion_count": ("codex_team", "child_completion_count"),
        "team_tool_violation": ("codex_team", "tool_calls", "has_violation"),
        "planner_round_count": ("planner", "round_count"),
        "planner_fallback_round_count": ("planner", "fallback_round_count"),
        "useful_transition_count": ("planner", "transitions", "useful_count"),
        "stale_transition_count": ("planner", "transitions", "stale_count"),
        "visual_accepted": ("visual_evidence", "accepted"),
        "visual_rejected": ("visual_evidence", "rejected"),
        "guided_accepted_routes": ("guided_verifier", "accepted_route_count"),
        "capability_accepted": ("capability_audit", "accepted"),
        "capability_failed_requirements": ("capability_audit", "failed_requirement_count"),
        "route_forest_branches": ("route_forest", "branch_count"),
        "route_forest_explicit_semantics": (
            "route_forest",
            "branch_semantics",
            "explicit_all_fields_count",
        ),
        "rejected_team_consensus_quarantine_passed": (
            "route_forest",
            "rejected_team_consensus_quarantine",
            "passed",
        ),
    }
    delta: dict[str, Any] = {}
    for label, path in metric_paths.items():
        before = _nested_get(baseline, path)
        after = _nested_get(candidate, path)
        row: dict[str, Any] = {"baseline": before, "candidate": after, "changed": before != after}
        if (
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
        ):
            row["delta"] = after - before
        delta[label] = row

    baseline_evidence = _as_dict(baseline.get("evidence_counts"))
    candidate_evidence = _as_dict(candidate.get("evidence_counts"))
    delta["evidence_counts"] = {
        key: {
            "baseline": _safe_int(baseline_evidence.get(key)),
            "candidate": _safe_int(candidate_evidence.get(key)),
            "delta": _safe_int(candidate_evidence.get(key)) - _safe_int(baseline_evidence.get(key)),
        }
        for key in sorted(set(baseline_evidence) | set(candidate_evidence))
    }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "delta": delta,
    }


def _nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def format_human(report: Mapping[str, Any]) -> str:
    """Render a compact companion summary; JSON remains the canonical output."""
    if report.get("schema_version") == COMPARISON_SCHEMA_VERSION:
        baseline = _as_dict(report.get("baseline"))
        candidate = _as_dict(report.get("candidate"))
        return "\n".join(
            [
                f"Agentic run comparison: {_nested_get(baseline, ('target', 'name')) or 'unknown'}",
                f"  baseline:  {_nested_get(baseline, ('run_dir',))}",
                f"  candidate: {_nested_get(candidate, ('run_dir',))}",
                f"  strict solved: {_nested_get(baseline, ('parent_route_proof', 'strict_solved'))} -> {_nested_get(candidate, ('parent_route_proof', 'strict_solved'))}",
                f"  team accepted: {_nested_get(baseline, ('codex_team', 'accepted'))} -> {_nested_get(candidate, ('codex_team', 'accepted'))}",
                f"  visual accepted: {_nested_get(baseline, ('visual_evidence', 'accepted'))} -> {_nested_get(candidate, ('visual_evidence', 'accepted'))}",
            ]
        )
    target = _as_dict(report.get("target"))
    verdict = _as_dict(report.get("final_verdict"))
    proof = _as_dict(report.get("parent_route_proof"))
    team = _as_dict(report.get("codex_team"))
    planner = _as_dict(report.get("planner"))
    visual = _as_dict(report.get("visual_evidence"))
    forest = _as_dict(report.get("route_forest"))
    return "\n".join(
        [
            f"Agentic run: {target.get('name') or target.get('case_id') or 'unknown'}",
            f"  verdict: {verdict.get('verdict') or 'missing'} ({verdict.get('route_status') or 'unknown'})",
            f"  strict parent proof solved: {proof.get('strict_solved')}",
            f"  Codex team: accepted={team.get('accepted')} children={team.get('child_completion_count')}/{team.get('child_spawn_count')}",
            f"  planner: rounds={planner.get('round_count')} fallback={planner.get('fallback_round_count')}",
            f"  visual: accepted={visual.get('accepted')} rejected={visual.get('rejected')}",
            f"  route forest: branches={forest.get('branch_count')} primary={forest.get('primary_branch_id') or 'none'}",
            f"  warnings: {len(report.get('warnings') or [])}",
        ]
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Saved agentic run directory")
    parser.add_argument(
        "--compare-to",
        metavar="RUN_DIR",
        help="Evaluate a candidate/final run and emit a baseline-to-candidate comparison",
    )
    parser.add_argument("--output", type=Path, help="Write canonical JSON to this path")
    parser.add_argument(
        "--human",
        action="store_true",
        help="Also print a short human summary to stderr",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline = evaluate_run(args.run_dir)
    report: dict[str, Any]
    if args.compare_to:
        report = compare_reports(baseline, evaluate_run(args.compare_to))
    else:
        report = baseline
    text = json.dumps(
        report,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.human:
        print(format_human(report), file=sys.stderr)
    return 0 if baseline.get("run_exists") else 2


if __name__ == "__main__":
    raise SystemExit(main())
