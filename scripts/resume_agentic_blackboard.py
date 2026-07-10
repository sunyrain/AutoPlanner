"""Resume a saved agentic blackboard run from its latest blackboard snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.agent_action_planner import validate_action_batch  # noqa: E402
from cascade_planner.harness.agentic_blackboard import (  # noqa: E402
    complete_round,
    refresh_target_derived_blackboard_priors,
    update_blackboard_from_action,
    update_blackboard_from_action_batch,
    update_budget_for_action,
)
from cascade_planner.harness.agentic_blackboard_controller import (  # noqa: E402
    _auto_update_critic,
    _emit_blackboard_step,
    _execute_agent_action,
    _finalize_agentic_run,
    _obtain_action_batch,
    _parent_proof_accepted,
    _record_action_batch_artifacts,
    _record_stop_on_problem,
    _refresh_blackboard_from_local_pdf_proxy_downloads,
    _result,
    _stop_on_problem_action_batch_reason,
    _stop_on_problem_action_result_reason,
)
from cascade_planner.harness.schemas import append_jsonl, write_json  # noqa: E402
from cascade_planner.harness.tools import HarnessBudget, ToolExecutionState  # noqa: E402


def resume_agentic_blackboard_run(
    run_dir: str | Path,
    *,
    max_new_rounds: int = 1,
    exhaust_round_budget: bool = False,
    extend_exploration_budget: bool = False,
    extra_guided_runs: int = 2,
    extra_child_target_runs: int = 4,
    extra_codex_research_runs: int = 2,
    extra_scout_calls: int = 2,
    extra_visual_calls: int = 2,
    extra_template_actions: int = 1,
    use_codex_action_planner: bool = False,
    stop_on_problem: bool = False,
    emit_blackboard_steps: bool = False,
    plan_only: bool = False,
    key_path: str | Path = "",
    base_url: str = "https://api.wellau.com/v1",
    model: str = "gpt-5.5",
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    blackboard_path = root / "agent_blackboard.json"
    if not blackboard_path.exists():
        raise FileNotFoundError(f"agent_blackboard.json not found: {blackboard_path}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "tool_calls.jsonl").touch()
    (root / "decision_trace.jsonl").touch()

    blackboard = _read_json(blackboard_path)
    target_input = _load_target_input(root, blackboard)
    preflight = _load_preflight(root, blackboard)
    blackboard = refresh_target_derived_blackboard_priors(
        blackboard,
        target_input=target_input,
        preflight=preflight,
    )
    _extend_round_budget(blackboard, max_new_rounds=max_new_rounds)
    if extend_exploration_budget:
        _extend_exploration_budget(
            blackboard,
            extra_guided_runs=extra_guided_runs,
            extra_child_target_runs=extra_child_target_runs,
            extra_codex_research_runs=extra_codex_research_runs,
            extra_scout_calls=extra_scout_calls,
            extra_visual_calls=extra_visual_calls,
            extra_template_actions=extra_template_actions,
        )
    budget = _load_budget(root, blackboard)

    state = ToolExecutionState(
        run_dir=root,
        target_input=target_input,
        preflight=preflight,
        budget=budget,
        key_path=key_path or ROOT / "key.txt",
        base_url=base_url,
        model=model,
    )
    _hydrate_state_from_blackboard(state, blackboard)
    state.artifacts.update(_load_existing_artifacts(root, blackboard))

    action_batches = _load_round_jsons(root, prefix="action_batch_round_")
    validations = _load_round_jsons(root, prefix="action_batch_validation_round_")
    tool_calls: list[dict[str, Any]] = []
    start_round = _next_round_index(blackboard, action_batches)
    step_index = _next_blackboard_step_index(root)
    append_jsonl(
        root / "decision_trace.jsonl",
        {
            "stage": "resume_start",
            "start_round": start_round,
            "max_new_rounds": int(max_new_rounds or 1),
            "extend_exploration_budget": bool(extend_exploration_budget),
            "plan_only": bool(plan_only),
        },
    )

    if plan_only:
        blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=root)
        batch = _obtain_action_batch(
            blackboard=blackboard,
            round_index=start_round,
            run_dir=root,
            state=state,
            action_planner=None,
            exhaust_round_budget=exhaust_round_budget,
            use_codex_action_planner=use_codex_action_planner,
        )
        validation = validate_action_batch(batch, blackboard=blackboard)
        preview = {
            "schema_version": "agentic_blackboard_resume_plan_preview.v1",
            "run_dir": str(root),
            "round_index": start_round,
            "accepted": bool(validation.get("accepted")),
            "validation": validation,
            "action_batch": batch,
            "action_types": [str(row.get("action_type") or "") for row in batch.get("actions") or []],
        }
        write_json(root / f"resume_plan_round_{start_round}.json", preview)
        return preview

    stop_requested = False
    executed_rounds = 0
    for round_index in range(start_round, start_round + max(1, int(max_new_rounds or 1))):
        blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=root)
        action_batch = _obtain_action_batch(
            blackboard=blackboard,
            round_index=round_index,
            run_dir=root,
            state=state,
            action_planner=None,
            exhaust_round_budget=exhaust_round_budget,
            use_codex_action_planner=use_codex_action_planner,
        )
        validation = validate_action_batch(action_batch, blackboard=blackboard)
        blackboard = update_blackboard_from_action_batch(
            blackboard,
            action_batch=action_batch,
            validation=validation,
            round_index=round_index,
        )
        if emit_blackboard_steps:
            step_index = _emit_blackboard_step(
                blackboard,
                run_dir=root,
                step_index=step_index,
                stage="resume_action_batch",
                round_index=round_index,
                detail={
                    "action_count": len(action_batch.get("actions") or []),
                    "validation_accepted": bool(validation.get("accepted")),
                    "validation_reasons": [str(item) for item in validation.get("reasons") or []],
                },
            )
        validations.append(validation)
        action_batches.append(action_batch)
        batch_path = root / f"action_batch_round_{round_index}.json"
        validation_path = root / f"action_batch_validation_round_{round_index}.json"
        write_json(batch_path, action_batch)
        write_json(validation_path, validation)
        _record_action_batch_artifacts(
            state=state,
            blackboard=blackboard,
            action_batch=action_batch,
            validation=validation,
            round_index=round_index,
            batch_path=batch_path,
            validation_path=validation_path,
        )
        append_jsonl(root / "decision_trace.jsonl", {"stage": "resume_action_batch", "round_index": round_index, "validation": validation})
        problem_reason = _stop_on_problem_action_batch_reason(action_batch, validation)
        if stop_on_problem and problem_reason:
            blackboard = _record_stop_on_problem(
                blackboard,
                run_dir=root,
                round_index=round_index,
                reason=problem_reason,
                action_type="action_batch",
            )
            stop_requested = True
            break
        if not validation.get("accepted"):
            state.validations.append(validation)
            break

        round_useful = False
        for action in action_batch.get("actions") or []:
            action_type = str(action.get("action_type") or "")
            action_result, records = _execute_agent_action(
                action=action,
                state=state,
                blackboard=blackboard,
            )
            tool_calls.extend(records)
            blackboard = update_budget_for_action(blackboard, action_type, payload=dict(action.get("payload") or {}))
            blackboard = update_blackboard_from_action(
                blackboard,
                action=action,
                action_result=action_result,
                round_index=round_index,
                run_dir=root,
            )
            if emit_blackboard_steps:
                step_index = _emit_blackboard_step(
                    blackboard,
                    run_dir=root,
                    step_index=step_index,
                    stage="resume_agent_action",
                    round_index=round_index,
                    action_id=str(action.get("action_id") or ""),
                    action_type=action_type,
                    detail={
                        "accepted": bool(action_result.get("accepted", True)),
                        "useful_artifact": bool(blackboard["action_history"][-1].get("useful_artifact")),
                    },
                )
            round_useful = round_useful or bool(blackboard["action_history"][-1].get("useful_artifact"))
            append_jsonl(
                root / "decision_trace.jsonl",
                {
                    "stage": "resume_agent_action",
                    "round_index": round_index,
                    "action_type": action_type,
                    "accepted": bool(action_result.get("accepted", True)),
                    "useful_artifact": bool(blackboard["action_history"][-1].get("useful_artifact")),
                },
            )
            problem_reason = _stop_on_problem_action_result_reason(
                action=action,
                action_result=action_result,
                history_record=dict(blackboard["action_history"][-1]),
            )
            if stop_on_problem and problem_reason:
                blackboard = _record_stop_on_problem(
                    blackboard,
                    run_dir=root,
                    round_index=round_index,
                    reason=problem_reason,
                    action_type=action_type,
                )
                stop_requested = True
                break
            if action_type == "stop_unresolved":
                stop_requested = True
                break
        blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=root)
        blackboard = _auto_update_critic(blackboard, state=state, run_dir=root, round_index=round_index)
        blackboard = complete_round(blackboard, round_index)
        write_json(root / "agent_blackboard.json", blackboard)
        executed_rounds += 1
        if stop_requested or _parent_proof_accepted(blackboard):
            break
        if not round_useful and executed_rounds >= int(max_new_rounds or 1):
            break

    blackboard = _refresh_blackboard_from_local_pdf_proxy_downloads(blackboard, run_dir=root)
    blackboard, bundle, final = _finalize_agentic_run(
        state=state,
        blackboard=blackboard,
        action_batches=action_batches,
        validations=validations,
        tool_calls=tool_calls,
    )
    result = _result(root, target_input, preflight, blackboard, action_batches, validations, bundle, final, tool_calls)
    result["resume_summary"] = {
        "schema_version": "agentic_blackboard_resume_summary.v1",
        "start_round": start_round,
        "executed_rounds": executed_rounds,
        "new_tool_call_count": len(tool_calls),
        "stop_requested": bool(stop_requested),
    }
    write_json(root / "resume_summary.json", result["resume_summary"])
    return result


def _load_target_input(root: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    path = root / "target_input.json"
    if path.exists():
        return _read_json(path)
    profile = dict(blackboard.get("target_profile") or {})
    return {
        "schema_version": "target_input.v1",
        "case_id": str(blackboard.get("case_id") or profile.get("case_id") or root.name),
        "target_name": str(profile.get("target_name") or root.name),
        "target_smiles": str(profile.get("target_smiles") or profile.get("smiles") or ""),
        "family_hint": str(profile.get("family_hint") or ""),
    }


def _load_preflight(root: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    path = root / "preflight.json"
    if path.exists():
        return _read_json(path)
    return {
        "schema_version": "preflight_result.v1",
        "accepted": bool((blackboard.get("target_profile") or {}).get("valid", True)),
        "case_id": str(blackboard.get("case_id") or root.name),
        "target_profile": dict(blackboard.get("target_profile") or {}),
        "reasons": [],
    }


def _load_budget(root: Path, blackboard: dict[str, Any]) -> HarnessBudget:
    data = _read_json(root / "budget.json") if (root / "budget.json").exists() else {}
    board_budget = dict(blackboard.get("budget_state") or {})
    budget = HarnessBudget()
    for field in fields(HarnessBudget):
        if field.name == "schema_version":
            continue
        board_value = _board_budget_alias(board_budget, field.name)
        if field.name.startswith("max_") and board_value is not None:
            value = board_value
        else:
            value = data.get(field.name)
            if value is None:
                value = board_value
        if value is None:
            continue
        current = getattr(budget, field.name)
        try:
            if isinstance(current, float):
                setattr(budget, field.name, float(value))
            elif isinstance(current, int):
                setattr(budget, field.name, int(value))
            else:
                setattr(budget, field.name, value)
        except (TypeError, ValueError):
            continue
    return budget


def _board_budget_alias(board_budget: dict[str, Any], field_name: str) -> Any:
    aliases = {
        "max_chem_enzy_runs": "max_chemenzy_runs",
        "max_guided_chemenzy_runs": "max_chemenzy_runs",
        "max_route_expansion_subgoal_runs": "max_child_target_runs",
    }
    return board_budget.get(field_name) if field_name in board_budget else board_budget.get(aliases.get(field_name, ""))


def _extend_round_budget(blackboard: dict[str, Any], *, max_new_rounds: int) -> None:
    budget = dict(blackboard.get("budget_state") or {})
    start_round = _next_round_index(blackboard, _load_round_jsons(Path("."), prefix="__never__"))
    current_max = int(budget.get("max_rounds") or 0)
    budget["max_rounds"] = max(current_max, start_round + max(1, int(max_new_rounds or 1)) - 1)
    blackboard["budget_state"] = budget


def _extend_exploration_budget(
    blackboard: dict[str, Any],
    *,
    extra_guided_runs: int = 2,
    extra_child_target_runs: int = 4,
    extra_codex_research_runs: int = 2,
    extra_scout_calls: int = 2,
    extra_visual_calls: int = 2,
    extra_template_actions: int = 1,
) -> None:
    budget = dict(blackboard.get("budget_state") or {})
    _ensure_budget_headroom(budget, used_key="chemenzy_runs", limit_key="max_chemenzy_runs", extra=extra_guided_runs)
    _ensure_budget_headroom(
        budget,
        used_key="child_target_runs",
        limit_key="max_child_target_runs",
        extra=extra_child_target_runs,
    )
    _ensure_budget_headroom(
        budget,
        used_key="codex_research_runs",
        limit_key="max_codex_research_runs",
        extra=extra_codex_research_runs,
    )
    _ensure_budget_headroom(budget, used_key="scout_calls", limit_key="max_scout_calls", extra=extra_scout_calls)
    _ensure_budget_headroom(budget, used_key="visual_calls", limit_key="max_visual_calls", extra=extra_visual_calls)
    _ensure_budget_headroom(
        budget,
        used_key="template_application_actions",
        limit_key="max_template_application_actions",
        extra=extra_template_actions,
    )
    blackboard["budget_state"] = budget


def _ensure_budget_headroom(budget: dict[str, Any], *, used_key: str, limit_key: str, extra: int) -> None:
    try:
        extra_int = max(0, int(extra or 0))
    except (TypeError, ValueError):
        extra_int = 0
    if extra_int <= 0:
        return
    try:
        used = int(budget.get(used_key) or 0)
    except (TypeError, ValueError):
        used = 0
    try:
        current_limit = int(budget.get(limit_key) or 0)
    except (TypeError, ValueError):
        current_limit = 0
    budget[limit_key] = max(current_limit, used + extra_int)


def _hydrate_state_from_blackboard(state: ToolExecutionState, blackboard: dict[str, Any]) -> None:
    budget = dict(blackboard.get("budget_state") or {})
    state.chem_enzy_runs = int(budget.get("chemenzy_runs") or 0)
    state.guided_chemenzy_runs = int(budget.get("chemenzy_runs") or 0)
    state.route_expansion_subgoal_runs = int(budget.get("child_target_runs") or 0)
    state.codex_research_runs = int(budget.get("codex_research_runs") or 0)


def _load_existing_artifacts(root: Path, blackboard: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    bundle_path = root / "artifact_bundle.json"
    if bundle_path.exists():
        bundle = _read_json(bundle_path)
        if isinstance(bundle.get("artifacts"), dict):
            artifacts.update(dict(bundle.get("artifacts") or {}))
    for key, ref in (blackboard.get("artifact_refs") or {}).items():
        path = Path(str(ref))
        if not path.exists() or key in artifacts:
            continue
        try:
            artifacts[str(key)] = _read_json(path)
        except Exception:
            continue
    for key, filename in {
        "agent_blackboard": "agent_blackboard.json",
        "route_expansion_subgoal_search": "route_expansion_subgoal_search_result.json",
        "guided_chemenzy": "guided_chemenzy_result.json",
        "route_proof_bundle": "route_proof_bundle.json",
    }.items():
        path = root / filename
        if path.exists() and key not in artifacts:
            artifacts[key] = _read_json(path)
    _index_pdf_evidence_artifacts(root, artifacts)
    return artifacts


def _index_pdf_evidence_artifacts(root: Path, artifacts: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for value in list(artifacts.values()):
        row = _extract_pdf_evidence_payload(value)
        if row:
            rows.append(row)
    for path in root.glob("*literature_pdf_structure_evidence*.json"):
        try:
            row = _extract_pdf_evidence_payload(_read_json(path))
        except Exception:
            continue
        if row:
            rows.append(row)
    history: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        key = _pdf_evidence_primary_key(row) or str(id(row))
        if key in seen:
            continue
        seen.add(key)
        history.append(row)
        for source_key in _pdf_evidence_keys(row):
            by_source[source_key] = row
    if history:
        artifacts["literature_pdf_structure_evidence_history"] = history
        artifacts["literature_pdf_structure_evidence_by_source"] = by_source
        artifacts["literature_pdf_structure_evidence"] = history[-1]


def _extract_pdf_evidence_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    candidates = [
        value,
        value.get("payload") if isinstance(value.get("payload"), dict) else {},
        value.get("result") if isinstance(value.get("result"), dict) else {},
        ((value.get("output") or {}).get("result") if isinstance(value.get("output"), dict) else {}),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("schema_version") or "") == "literature_pdf_structure_evidence.v1":
            return dict(candidate)
        if candidate.get("rendered_pages") and (
            candidate.get("source_ref")
            or candidate.get("source_pdf_path")
            or candidate.get("pdf_path")
        ):
            return dict(candidate)
    return {}


def _pdf_evidence_primary_key(evidence: dict[str, Any]) -> str:
    keys = _pdf_evidence_keys(evidence)
    return keys[0] if keys else ""


def _pdf_evidence_keys(evidence: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    source_ref = str(evidence.get("source_ref") or "").strip().lower()
    if source_ref:
        keys.append(f"ref:{source_ref}")
    pdf_path = str(evidence.get("source_pdf_path") or evidence.get("pdf_path") or "").strip()
    if pdf_path:
        keys.append(f"pdf:{str(Path(pdf_path).expanduser().resolve()).lower()}")
    title = " ".join(str(evidence.get("source_title") or "").strip().lower().split())
    if title:
        keys.append(f"title:{title}")
    return keys


def _load_round_jsons(root: Path, *, prefix: str) -> list[dict[str, Any]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for path in root.glob(f"{prefix}*.json"):
        try:
            round_index = int(path.stem.replace(prefix, ""))
        except ValueError:
            continue
        try:
            rows.append((round_index, _read_json(path)))
        except Exception:
            continue
    return [row for _, row in sorted(rows, key=lambda item: item[0])]


def _next_round_index(blackboard: dict[str, Any], action_batches: list[dict[str, Any]]) -> int:
    candidates = [int((blackboard.get("budget_state") or {}).get("rounds_completed") or 0)]
    candidates.extend(int(row.get("round_index") or 0) for row in action_batches if isinstance(row, dict))
    for row in blackboard.get("planner_history") or []:
        if isinstance(row, dict):
            candidates.append(int(row.get("round_index") or 0))
    for row in blackboard.get("action_history") or []:
        if isinstance(row, dict):
            candidates.append(int(row.get("round_index") or 0))
    return max(candidates or [0]) + 1


def _next_blackboard_step_index(root: Path) -> int:
    step_dir = root / "blackboard_steps"
    if not step_dir.exists():
        return 0
    highest = 0
    for path in step_dir.glob("*.json"):
        token = path.stem.split("_", 1)[0]
        try:
            highest = max(highest, int(token))
        except ValueError:
            continue
    return highest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Saved run directory containing agent_blackboard.json")
    parser.add_argument("--max-new-rounds", type=int, default=1)
    parser.add_argument("--exhaust-round-budget", action="store_true")
    parser.add_argument("--extend-exploration-budget", action="store_true")
    parser.add_argument("--extra-guided-runs", type=int, default=2)
    parser.add_argument("--extra-child-target-runs", type=int, default=4)
    parser.add_argument("--extra-codex-research-runs", type=int, default=2)
    parser.add_argument("--extra-scout-calls", type=int, default=2)
    parser.add_argument("--extra-visual-calls", type=int, default=2)
    parser.add_argument("--extra-template-actions", type=int, default=1)
    parser.add_argument("--codex-action-planner", action="store_true")
    parser.add_argument("--stop-on-problem", action="store_true")
    parser.add_argument("--emit-blackboard-steps", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--key-path", default="")
    parser.add_argument("--base-url", default="https://api.wellau.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--print-full-result", action="store_true")
    args = parser.parse_args()
    result = resume_agentic_blackboard_run(
        args.run_dir,
        max_new_rounds=int(args.max_new_rounds or 1),
        exhaust_round_budget=bool(args.exhaust_round_budget),
        extend_exploration_budget=bool(args.extend_exploration_budget),
        extra_guided_runs=int(args.extra_guided_runs or 0),
        extra_child_target_runs=int(args.extra_child_target_runs or 0),
        extra_codex_research_runs=int(args.extra_codex_research_runs or 0),
        extra_scout_calls=int(args.extra_scout_calls or 0),
        extra_visual_calls=int(args.extra_visual_calls or 0),
        extra_template_actions=int(args.extra_template_actions or 0),
        use_codex_action_planner=bool(args.codex_action_planner),
        stop_on_problem=bool(args.stop_on_problem),
        emit_blackboard_steps=bool(args.emit_blackboard_steps),
        plan_only=bool(args.plan_only),
        key_path=args.key_path,
        base_url=args.base_url,
        model=args.model,
    )
    output = result if args.print_full_result else _compact_cli_result(result)
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


def _compact_cli_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema_version") == "agentic_blackboard_resume_plan_preview.v1":
        return {
            "schema_version": "agentic_blackboard_resume_plan_preview_summary.v1",
            "run_dir": result.get("run_dir"),
            "round_index": result.get("round_index"),
            "accepted": result.get("accepted"),
            "action_types": result.get("action_types") or [],
            "validation_reasons": (result.get("validation") or {}).get("reasons") or [],
            "preview_ref": str(Path(str(result.get("run_dir") or ".")) / f"resume_plan_round_{int(result.get('round_index') or 0)}.json"),
        }
    final = dict(result.get("final_verdict") or {})
    route_display = _route_display_summary(result)
    return {
        "schema_version": "agentic_blackboard_resume_cli_summary.v1",
        "run_dir": result.get("run_dir"),
        "resume_summary": result.get("resume_summary") or {},
        "final_verdict": {
            "verdict": final.get("verdict"),
            "route_status": final.get("route_status"),
            "solved": final.get("solved"),
            "reasons": final.get("reasons") or [],
        },
        "new_action_batch_count": len(result.get("action_batches") or []),
        "validation_reasons": [
            reason
            for row in result.get("validations") or []
                for reason in row.get("reasons") or []
        ],
        "route_display": route_display,
        "artifacts": result.get("artifacts") or {},
    }


def _route_display_summary(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(result.get("artifacts") or {})
    bundle_artifacts = dict((result.get("artifact_bundle") or {}).get("artifacts") or {})
    display = dict(bundle_artifacts.get("route_forest_display") or {})
    payload = dict(display.get("payload") or {})
    counts = dict(payload.get("counts") or {})
    primary = dict(payload.get("primary_branch") or {})
    final = dict(result.get("final_verdict") or {})
    has_display_route = bool(primary.get("branch_id")) or int(counts.get("branches") or 0) > 0
    if bool(final.get("solved")):
        outcome = "verified_solved_route"
    elif has_display_route:
        outcome = "advisory_route_available_not_solved"
    else:
        outcome = "no_display_route_compiled"
    return {
        "schema_version": "route_display_cli_summary.v1",
        "outcome": outcome,
        "accepted": bool(payload.get("accepted")),
        "html_path": artifacts.get("route_forest_html") or payload.get("html_path"),
        "forest_path": artifacts.get("explored_route_forest") or payload.get("forest_path"),
        "branch_count": int(counts.get("branches") or 0),
        "step_count": int(counts.get("steps") or 0),
        "node_count": int(counts.get("nodes") or 0),
        "primary_branch": {
            "branch_id": primary.get("branch_id"),
            "title": primary.get("title"),
            "kind": primary.get("kind"),
            "step_count": primary.get("step_count"),
        },
    }


if __name__ == "__main__":
    main()
