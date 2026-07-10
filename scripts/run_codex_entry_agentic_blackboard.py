#!/usr/bin/env python3
"""Run the agentic blackboard Codex-entry controller."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.target_profile import build_target_profile  # noqa: E402
from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller  # noqa: E402
from cascade_planner.harness.tools import HarnessBudget  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "results" / "shared"
DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="*", help="One or more target SMILES.")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--target-smiles", default="")
    parser.add_argument("--family-hint", default="")
    parser.add_argument("--literature-pdf-path", default="")
    parser.add_argument("--literature-pdf-source-ref", default="")
    parser.add_argument(
        "--literature-source",
        "--local-literature-cache",
        action="append",
        default=[],
        help="Repeatable local PDF cache entry as PATH or PATH::DOI/SOURCE_REF or JSON object; used only after agent-discovered DOI/title matches it.",
    )
    parser.add_argument(
        "--local-pdf-search-dir",
        action="append",
        default=[],
        help="Repeatable directory or PDF path to index as auto local PDF cache; matched only after agent-discovered DOI/title/PII.",
    )
    parser.add_argument(
        "--auto-local-pdf-discovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically index local PDFs as a metadata-matched cache. Disabled by default; online source acquisition is primary.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-prefix", default="agentic_blackboard")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--exhaust-round-budget", action="store_true", help="Continue with non-stale alternative actions until max rounds are consumed.")
    parser.add_argument(
        "--stop-on-problem",
        action="store_true",
        help="Stop immediately on fallback planning, invalid action batch, rejected action, or stale/no-useful-artifact action.",
    )
    parser.add_argument(
        "--emit-blackboard-steps",
        action="store_true",
        help="Write full blackboard snapshots and compact summaries after each planner/action/round step.",
    )
    parser.add_argument(
        "--codex-action-planner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Codex as the blackboard action planner. When --codex-agent-team is enabled, "
            "Codex failures stop unresolved instead of invoking a deterministic scientific planner."
        ),
    )
    parser.add_argument(
        "--codex-agent-team",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a Codex coordinator that must directly spawn specialist retrosynthesis child agents.",
    )
    parser.add_argument("--codex-agent-team-max-depth", type=int, default=2)
    parser.add_argument("--codex-agent-team-max-expansions", type=int, default=4)
    parser.add_argument("--codex-agent-team-frontier-batch-size", type=int, default=2)
    parser.add_argument(
        "--codex-agent-team-model",
        default="",
        help="Coordinator/child model; empty inherits --model.",
    )
    parser.add_argument(
        "--codex-agent-team-auth-mode",
        choices=("ambient_codex_cli", "auto", "api_key"),
        default=None,
        help="Coordinator/child auth; omitted inherits --codex-worker-auth.",
    )
    parser.add_argument(
        "--codex-action-planner-tools",
        default=None,
        help=(
            "Comma-separated audited tools for Codex action planning "
            "(default: web_search,browser,literature_search; use 'none' to disable planner tools)."
        ),
    )
    parser.add_argument(
        "--codex-action-planner-max-tool-calls",
        type=int,
        default=None,
        help="Maximum tool calls for each Codex action-planner worker run.",
    )
    parser.add_argument(
        "--codex-action-planner-timeout-s",
        type=float,
        default=None,
        help="Timeout for each Codex action-planner worker run; defaults to --timeout-s.",
    )
    parser.add_argument(
        "--codex-scout-timeout-s",
        type=float,
        default=None,
        help="Timeout for each Codex literature scout worker run.",
    )
    parser.add_argument(
        "--codex-scout-reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort for Codex literature scout workers.",
    )
    parser.add_argument(
        "--codex-worker-auth",
        choices=["auto", "ambient", "key"],
        default="auto",
        help="Auth source for Codex CLI workers. Use ambient to reuse the current Codex login.",
    )
    parser.add_argument(
        "--codex-worker-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access", "bypassed"],
        default=None,
        help="Sandbox mode passed to Codex CLI workers; use bypassed on hosts without user namespace support.",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--guided-chemenzy-timeout-s", type=float, default=None)
    parser.add_argument("--max-chem-enzy-runs", type=int, default=None)
    parser.add_argument("--max-guided-chemenzy-runs", type=int, default=None)
    parser.add_argument("--max-route-expansion-subgoal-runs", type=int, default=None)
    parser.add_argument("--max-codex-research-runs", type=int, default=None)
    parser.add_argument("--max-scout-calls", type=int, default=None)
    parser.add_argument("--max-visual-calls", type=int, default=None)
    parser.add_argument("--enable-analogical-templates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-template-applications-per-round", type=int, default=5)
    parser.add_argument("--template-radius-policy", choices=["auto", "local", "broad"], default="auto")
    parser.add_argument("--analog-template-confidence-threshold", choices=["low", "medium", "medium_high", "high"], default="medium")
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    targets = _resolve_targets(args)
    results = [_run_one(target, args) for target in targets]
    print(json.dumps(_cli_result(results), indent=2, ensure_ascii=False, sort_keys=True))


def _resolve_targets(args: argparse.Namespace) -> list[dict[str, str | Path]]:
    smiles_values = [str(item).strip() for item in args.smiles if str(item).strip()]
    if str(args.target_smiles or "").strip():
        smiles_values.append(str(args.target_smiles).strip())
    if not smiles_values:
        raise SystemExit("Provide at least one SMILES, either positional or with --target-smiles.")
    if str(args.target_name or "").strip() and len(smiles_values) != 1:
        raise SystemExit("--target-name can only be used with exactly one target SMILES.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() if str(args.output_dir or "").strip() else None
    output_root = Path(args.output_root).expanduser()
    rows: list[dict[str, str | Path]] = []
    for idx, smiles in enumerate(smiles_values, start=1):
        name = str(args.target_name or "").strip() or _case_slug(smiles, idx=idx, total=len(smiles_values))
        slug = _safe_path_part(name)
        run_dir = output_dir if output_dir is not None and len(smiles_values) == 1 else (
            (output_dir or output_root) / f"{_safe_path_part(args.run_prefix) or 'agentic_blackboard'}_{slug}_{timestamp}"
        )
        rows.append({"target_name": name, "target_smiles": smiles, "output_dir": run_dir})
    return rows


def _run_one(target: dict[str, str | Path], args: argparse.Namespace) -> dict:
    overrides = _codex_action_planner_env_overrides(args)
    team_model, team_auth_mode = _codex_agent_team_runtime_args(args)
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        return run_agentic_blackboard_controller(
            target_name=str(target["target_name"]),
            target_smiles=str(target["target_smiles"]),
            family_hint=str(args.family_hint or ""),
            output_dir=Path(target["output_dir"]),
            literature_pdf_path=args.literature_pdf_path,
            literature_pdf_source_ref=args.literature_pdf_source_ref,
            literature_sources=_literature_sources_from_args(args),
            auto_discover_local_pdfs=bool(args.auto_local_pdf_discovery),
            local_pdf_search_dirs=[Path(item).expanduser() for item in args.local_pdf_search_dir or []],
            timeout_s=float(args.timeout_s),
            key_path=args.key_path,
            base_url=args.base_url,
            model=args.model,
            max_rounds=int(args.max_rounds or 3),
            exhaust_round_budget=bool(args.exhaust_round_budget),
            enable_analogical_templates=bool(args.enable_analogical_templates),
            max_template_applications_per_round=int(args.max_template_applications_per_round or 5),
            template_radius_policy=str(args.template_radius_policy or "auto"),
            analog_template_confidence_threshold=str(args.analog_template_confidence_threshold or "medium"),
            use_codex_action_planner=bool(args.codex_action_planner),
            use_codex_agent_team=bool(args.codex_agent_team),
            codex_agent_team_max_depth=max(1, int(args.codex_agent_team_max_depth or 1)),
            codex_agent_team_max_expansions=max(1, int(args.codex_agent_team_max_expansions or 1)),
            codex_agent_team_frontier_batch_size=max(1, int(args.codex_agent_team_frontier_batch_size or 1)),
            codex_agent_team_model=team_model,
            codex_agent_team_auth_mode=team_auth_mode,
            stop_on_problem=bool(args.stop_on_problem),
            budget=_budget_from_args(args),
            emit_blackboard_steps=bool(args.emit_blackboard_steps),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _codex_action_planner_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {}
    tools = getattr(args, "codex_action_planner_tools", None)
    if tools is not None:
        overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_ALLOWED_TOOLS"] = str(tools)
    max_calls = getattr(args, "codex_action_planner_max_tool_calls", None)
    if max_calls is not None:
        overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_TOOL_CALLS"] = str(int(max_calls))
    planner_timeout = getattr(args, "codex_action_planner_timeout_s", None)
    if planner_timeout is None:
        planner_timeout = getattr(args, "timeout_s", None)
    if planner_timeout is not None:
        overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_TIMEOUT_S"] = str(float(planner_timeout))
    scout_timeout = getattr(args, "codex_scout_timeout_s", None)
    if scout_timeout is not None:
        overrides["AUTOPLANNER_CODEX_SCOUT_TIMEOUT_S"] = str(float(scout_timeout))
    scout_effort = str(getattr(args, "codex_scout_reasoning_effort", "") or "").strip()
    if scout_effort:
        overrides["AUTOPLANNER_CODEX_SCOUT_REASONING_EFFORT"] = scout_effort
    worker_auth = str(getattr(args, "codex_worker_auth", "") or "").strip().lower()
    if worker_auth and worker_auth != "auto":
        overrides["AUTOPLANNER_CODEX_WORKER_AUTH"] = worker_auth
    worker_sandbox = getattr(args, "codex_worker_sandbox", None)
    if worker_sandbox:
        overrides["AUTOPLANNER_CODEX_WORKER_SANDBOX"] = str(worker_sandbox)
    local_pdf_seeded = bool(
        getattr(args, "literature_pdf_path", None)
        or getattr(args, "literature_source", None)
        or (getattr(args, "auto_local_pdf_discovery", False) and getattr(args, "local_pdf_search_dir", None))
    )
    overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_LOCAL_PDF_FALLBACK_ALLOWED"] = "1" if local_pdf_seeded else "0"
    return overrides


def _codex_agent_team_runtime_args(args: argparse.Namespace) -> tuple[str, str]:
    """Keep coordinator workers on the same model/auth policy as other Codex workers."""
    model = str(
        getattr(args, "codex_agent_team_model", "")
        or getattr(args, "model", "")
        or ""
    ).strip()
    explicit_auth = str(getattr(args, "codex_agent_team_auth_mode", "") or "").strip()
    if explicit_auth:
        return model, explicit_auth
    worker_auth = str(getattr(args, "codex_worker_auth", "auto") or "auto").strip().lower()
    inherited = {
        "ambient": "ambient_codex_cli",
        "key": "api_key",
        "auto": "auto",
    }.get(worker_auth, "auto")
    return model, inherited


def _budget_from_args(args: argparse.Namespace) -> HarnessBudget:
    budget = HarnessBudget(timeout_s=float(args.timeout_s))
    if args.max_chem_enzy_runs is not None:
        budget.max_chem_enzy_runs = int(args.max_chem_enzy_runs)
    if args.max_guided_chemenzy_runs is not None:
        budget.max_guided_chemenzy_runs = int(args.max_guided_chemenzy_runs)
    elif args.max_chem_enzy_runs is not None:
        budget.max_guided_chemenzy_runs = int(args.max_chem_enzy_runs)
    if args.guided_chemenzy_timeout_s is not None:
        budget.guided_chemenzy_timeout_s = float(args.guided_chemenzy_timeout_s)
    if args.max_route_expansion_subgoal_runs is not None:
        budget.max_route_expansion_subgoal_runs = int(args.max_route_expansion_subgoal_runs)
    if args.max_codex_research_runs is not None:
        budget.max_codex_research_runs = int(args.max_codex_research_runs)
    if args.max_scout_calls is not None:
        budget.max_scout_calls = int(args.max_scout_calls)
    if args.max_visual_calls is not None:
        budget.max_visual_calls = int(args.max_visual_calls)
    budget.max_template_applications_per_round = int(args.max_template_applications_per_round or 5)
    return budget


def _literature_sources_from_args(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in args.literature_source or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid --literature-source JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise SystemExit("--literature-source JSON must be an object.")
            rows.append({str(k): str(v) for k, v in data.items() if v is not None})
            continue
        if "::" in text:
            path, source_ref = text.split("::", 1)
        else:
            path, source_ref = text, ""
        rows.append({"local_pdf": path.strip(), "source_ref": source_ref.strip()})
    return rows


def _case_slug(smiles: str, *, idx: int, total: int) -> str:
    profile = build_target_profile(smiles)
    slug = profile.case_id or f"target_{idx}"
    return slug if total == 1 else f"{slug}_{idx:02d}"


def _safe_path_part(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    return "_".join(part for part in safe.split("_") if part)


def _cli_result(results: list[dict]) -> dict:
    if len(results) == 1:
        result = results[0]
        return {
            "schema_version": "agentic_blackboard_controller_cli_result.v1",
            "run_dir": result["run_dir"],
            "final_verdict": result["final_verdict"],
            "artifacts": result["artifacts"],
        }
    return {
        "schema_version": "agentic_blackboard_controller_cli_batch_result.v1",
        "run_count": len(results),
        "runs": [
            {
                "run_dir": result["run_dir"],
                "target_name": result["target_input"]["target_name"],
                "final_verdict": result["final_verdict"],
                "artifacts": result["artifacts"],
            }
            for result in results
        ],
    }


if __name__ == "__main__":
    main()
