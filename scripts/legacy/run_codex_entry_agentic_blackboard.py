#!/usr/bin/env python3
"""Run the agentic blackboard Codex-entry controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.agent.target_profile import build_target_profile  # noqa: E402
from cascade_planner.application.retrosynthesis_run_contract import (  # noqa: E402
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import run_agentic_blackboard_controller  # noqa: E402
from cascade_planner.legacy.harness_runtime.tools import HarnessBudget  # noqa: E402
from cascade_planner.providers.stock import (  # noqa: E402
    canonicalize_stock_snapshot,
    stock_snapshot_sha256,
)


DEFAULT_OUTPUT_ROOT = ROOT / "results" / "shared"
DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TRUSTED_STOCK_CATALOGS_CONFIG = ROOT / "config" / "trusted_stock_catalogs.json"


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
        "--literature-sources-file",
        action="append",
        default=[],
        help=(
            "Repeatable UTF-8 JSON manifest containing a source object, a list "
            "of source objects, or {'literature_sources': [...]}. This avoids "
            "lossy native-shell quoting of structured source metadata."
        ),
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
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Deterministic evidence/action rounds (bounded default: 4).",
    )
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
        default=False,
        help=(
            "Use a separate Codex call as the blackboard action planner. Disabled by default; "
            "the deterministic route-deficit scheduler is the normal control plane."
        ),
    )
    parser.add_argument(
        "--codex-agent-team",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a Codex coordinator that must directly spawn specialist retrosynthesis child agents.",
    )
    parser.add_argument(
        "--codex-agent-team-max-depth",
        type=int,
        default=2,
        help="Maximum recursive molecule-frontier depth per bounded campaign (default: 2).",
    )
    parser.add_argument(
        "--codex-agent-team-max-expansions",
        type=int,
        default=8,
        help="Hard cumulative accepted-expansion cap shared with the run contract (default: 8).",
    )
    parser.add_argument(
        "--codex-agent-team-max-attempt-runs",
        type=int,
        default=12,
        help=(
            "Hard cumulative Agent attempt cap for the campaign (default: 12). "
            "This is independent of accepted frontier expansions."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-bootstrap-expansions",
        type=int,
        default=1,
        help="Accepted expansions allowed before the first evidence/action round (standard profile: 1).",
    )
    parser.add_argument(
        "--codex-agent-team-max-expansions-per-invocation",
        type=int,
        default=1,
        help="Accepted expansions allowed in one model-backed invocation (default: 1).",
    )
    parser.add_argument(
        "--codex-agent-team-max-attempt-runs-per-invocation",
        type=int,
        default=1,
        help="Total child execution attempts allowed in one invocation (default: 1).",
    )
    parser.add_argument("--codex-agent-team-frontier-batch-size", type=int, default=1)
    parser.add_argument(
        "--codex-agent-team-auto-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Permit later model-backed campaign invocations after deterministic work is drained. "
            "Disabled by default; enabling it never bypasses the run-wide cost ledger."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-child-roles",
        default="target_structure_strategist,route_evidence_critic",
        help=(
            "Comma-separated directly spawned specialist roles. The bounded default uses one "
            "proposal role and one independent critic; add literature/chemoenzymatic roles only "
            "when their capabilities are required."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-closure-objective",
        choices=("benchmark_search", "procurement", "in_house"),
        default="benchmark_search",
        help=(
            "Scientific closure boundary for this immutable campaign. Changing it "
            "requires a new campaign directory."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-exploration-mode",
        choices=("first_solved", "exhaustive"),
        default="exhaustive",
        help=(
            "Stop after the first objective-closed route or exhaustively close every "
            "accepted alternative (default)."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-child-acceptance-mode",
        choices=("strict_all", "valid_subset_l0"),
        default="strict_all",
        help=(
            "Immutable child-report acceptance policy. valid_subset_l0 still "
            "requires every role to be explicitly spawned and caps retained "
            "partial proposals at L0/model-only/low confidence."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-authority-lock-timeout-s",
        type=float,
        default=3600.0,
        help=(
            "Maximum wait for the non-stealable run-directory campaign "
            "authority lock."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-benchmark-stock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load the pinned benchmark catalog configured below. Membership closes only the "
            "benchmark boundary; it is not a commercial availability or procurement claim."
        ),
    )
    parser.add_argument(
        "--trusted-stock-catalogs-config",
        default=str(DEFAULT_TRUSTED_STOCK_CATALOGS_CONFIG),
        help="Path to trusted_stock_catalogs.v1 JSON; relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--benchmark-stock-catalog",
        default="",
        help="Catalog key in the trusted stock config; empty uses default_catalog.",
    )
    parser.add_argument(
        "--trusted-stock-snapshot",
        action="append",
        default=[],
        help=(
            "Repeatable operator-trusted JSON artifact containing one "
            "stock_offer_snapshot.v1, a list of snapshots, or a "
            "trusted_stock_snapshots.v1 object. Every snapshot must carry its "
            "canonical snapshot_sha256. Required for procurement closure."
        ),
    )
    parser.add_argument(
        "--codex-agent-team-model",
        default="",
        help="Coordinator/child model; empty inherits --model.",
    )
    parser.add_argument(
        "--codex-agent-team-reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default="low",
        help="Coordinator/child reasoning effort (bounded default: low).",
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
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--chem-enzy-env-prefix",
        default=None,
        help=(
            "Explicit host-compatible ChemEnzy environment prefix. This has "
            "higher priority than CHEMENZY_ENV_PREFIX and is capability-probed "
            "before a ChemEnzy action can launch."
        ),
    )
    parser.add_argument("--guided-chemenzy-timeout-s", type=float, default=None)
    parser.add_argument("--max-chem-enzy-runs", type=int, default=None)
    parser.add_argument("--max-guided-chemenzy-runs", type=int, default=None)
    parser.add_argument("--max-route-expansion-subgoal-runs", type=int, default=None)
    parser.add_argument("--max-codex-research-runs", type=int, default=None)
    parser.add_argument("--max-scout-calls", type=int, default=None)
    parser.add_argument("--max-visual-calls", type=int, default=1)
    parser.add_argument("--minimum-complete-routes", type=int, default=2)
    parser.add_argument("--minimum-edge-proof-level", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--minimum-independent-source-groups", type=int, default=2)
    parser.add_argument("--max-model-invocations", type=int, default=3)
    parser.add_argument("--max-total-input-tokens", type=int, default=60_000)
    parser.add_argument("--max-total-output-tokens", type=int, default=12_000)
    parser.add_argument("--max-model-wall-time-s", type=float, default=1_800.0)
    parser.add_argument("--max-prompt-context-bytes", type=int, default=96_000)
    parser.add_argument("--enable-analogical-templates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-template-applications-per-round", type=int, default=5)
    parser.add_argument("--template-radius-policy", choices=["auto", "local", "broad"], default="auto")
    parser.add_argument("--analog-template-confidence-threshold", choices=["low", "medium", "medium_high", "high"], default="medium")
    parser.add_argument(
        "--deterministic-literature-parser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reconstruct source headings with OPSIN and resolve every proposed "
            "reactant in the exact experimental paragraph before emitting an "
            "out-of-band trusted precedent binding."
        ),
    )
    parser.add_argument(
        "--opsin-base-url",
        default="https://opsin.ch.cam.ac.uk/opsin",
        help="OPSIN name-to-structure endpoint used by the deterministic parser.",
    )
    parser.add_argument(
        "--deterministic-literature-parser-timeout-s",
        type=float,
        default=30.0,
    )
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
    if bool(args.deterministic_literature_parser):
        overrides["AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY"] = str(
            (
                Path(target["output_dir"])
                / "trusted_literature_step_registry.generated.json"
            ).resolve()
        )
    team_model, team_auth_mode = _codex_agent_team_runtime_args(args)
    benchmark_stock = (
        _benchmark_stock_catalog_from_args(args)
        if bool(args.codex_agent_team)
        else {}
    )
    trusted_stock_snapshots = (
        _trusted_stock_snapshots_from_args(args)
        if bool(args.codex_agent_team)
        else {}
    )
    if (
        bool(args.codex_agent_team)
        and str(args.codex_agent_team_closure_objective or "") == "procurement"
        and not trusted_stock_snapshots
    ):
        raise SystemExit(
            "Procurement closure requires at least one operator-trusted "
            "--trusted-stock-snapshot artifact; a benchmark catalog is not "
            "commercial availability evidence."
        )
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
            max_rounds=int(args.max_rounds or 6),
            exhaust_round_budget=bool(args.exhaust_round_budget),
            enable_analogical_templates=bool(args.enable_analogical_templates),
            max_template_applications_per_round=int(args.max_template_applications_per_round or 5),
            template_radius_policy=str(args.template_radius_policy or "auto"),
            analog_template_confidence_threshold=str(args.analog_template_confidence_threshold or "medium"),
            enable_deterministic_literature_parser=bool(
                args.deterministic_literature_parser
            ),
            deterministic_literature_parser_opsin_base_url=str(
                args.opsin_base_url
                or "https://opsin.ch.cam.ac.uk/opsin"
            ),
            deterministic_literature_parser_timeout_s=max(
                1.0,
                float(args.deterministic_literature_parser_timeout_s or 30.0),
            ),
            use_codex_action_planner=bool(args.codex_action_planner),
            use_codex_agent_team=bool(args.codex_agent_team),
            codex_agent_team_max_depth=max(1, int(args.codex_agent_team_max_depth or 1)),
            codex_agent_team_max_expansions=max(1, int(args.codex_agent_team_max_expansions or 1)),
            codex_agent_team_max_attempt_runs=max(
                1,
                int(args.codex_agent_team_max_attempt_runs or 1),
            ),
            codex_agent_team_bootstrap_expansions=max(
                1,
                int(args.codex_agent_team_bootstrap_expansions or 1),
            ),
            codex_agent_team_max_expansions_per_invocation=max(
                1,
                int(args.codex_agent_team_max_expansions_per_invocation or 1),
            ),
            codex_agent_team_max_attempt_runs_per_invocation=max(
                1,
                int(args.codex_agent_team_max_attempt_runs_per_invocation or 1),
            ),
            codex_agent_team_frontier_batch_size=max(1, int(args.codex_agent_team_frontier_batch_size or 1)),
            codex_agent_team_closure_objective=str(
                args.codex_agent_team_closure_objective or "benchmark_search"
            ),
            codex_agent_team_exploration_mode=str(
                args.codex_agent_team_exploration_mode or "exhaustive"
            ),
            codex_agent_team_child_acceptance_mode=str(
                args.codex_agent_team_child_acceptance_mode or "strict_all"
            ),
            codex_agent_team_authority_lock_timeout_s=max(
                0.1,
                float(args.codex_agent_team_authority_lock_timeout_s or 3600.0),
            ),
            codex_agent_team_model=team_model,
            codex_agent_team_reasoning_effort=str(
                args.codex_agent_team_reasoning_effort or "low"
            ),
            codex_agent_team_auth_mode=team_auth_mode,
            codex_agent_team_stock_snapshots=trusted_stock_snapshots,
            codex_agent_team_benchmark_stock_catalog_artifact=benchmark_stock.get("artifact", ""),
            codex_agent_team_benchmark_stock_catalog_sha256=benchmark_stock.get("sha256", ""),
            codex_agent_team_benchmark_stock_catalog_name=benchmark_stock.get("name", ""),
            codex_agent_team_auto_resume=bool(args.codex_agent_team_auto_resume),
            codex_agent_team_child_roles=_child_roles_from_args(args),
            retrosynthesis_acceptance_spec=_acceptance_spec_from_args(args),
            retrosynthesis_run_budget=_run_budget_from_args(args),
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
    chem_enzy_prefix = getattr(args, "chem_enzy_env_prefix", None)
    if chem_enzy_prefix is not None and str(chem_enzy_prefix).strip():
        overrides["CHEMENZY_ENV_PREFIX"] = str(
            Path(str(chem_enzy_prefix)).expanduser().resolve()
        )
        overrides["AUTOPLANNER_CHEMENZY_ENV_PREFIX_SOURCE"] = "cli"
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
        or getattr(args, "literature_sources_file", None)
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


def _benchmark_stock_catalog_from_args(args: argparse.Namespace) -> dict[str, str | Path | bool]:
    """Load and verify a pinned benchmark boundary without implying procurement."""
    if not bool(getattr(args, "codex_agent_team_benchmark_stock", True)):
        return {}

    raw_config_path = str(
        getattr(args, "trusted_stock_catalogs_config", "")
        or DEFAULT_TRUSTED_STOCK_CATALOGS_CONFIG
    ).strip()
    config_path = Path(raw_config_path).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Trusted stock catalog config does not exist: {config_path}")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read trusted stock catalog config {config_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "trusted_stock_catalogs.v1":
        raise SystemExit(
            f"Trusted stock catalog config must use schema trusted_stock_catalogs.v1: {config_path}"
        )

    catalogs = payload.get("catalogs")
    if not isinstance(catalogs, dict):
        raise SystemExit(f"Trusted stock catalog config has no catalogs object: {config_path}")
    catalog_key = str(
        getattr(args, "benchmark_stock_catalog", "")
        or payload.get("default_catalog")
        or ""
    ).strip()
    catalog = catalogs.get(catalog_key)
    if not catalog_key or not isinstance(catalog, dict):
        raise SystemExit(f"Unknown benchmark stock catalog {catalog_key!r} in {config_path}")
    if str(catalog.get("boundary_type") or "") != "benchmark_stock":
        raise SystemExit(f"Catalog {catalog_key!r} is not a benchmark_stock boundary")
    if catalog.get("commercial_orderability_claimed") is not False:
        raise SystemExit(
            f"Catalog {catalog_key!r} must explicitly set commercial_orderability_claimed=false"
        )

    name = str(catalog.get("name") or catalog_key).strip()
    artifact_value = str(catalog.get("artifact") or "").strip()
    expected_sha256 = str(catalog.get("sha256") or "").strip().lower()
    if not artifact_value:
        raise SystemExit(f"Catalog {catalog_key!r} has no artifact path")
    if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
        raise SystemExit(f"Catalog {catalog_key!r} has an invalid SHA-256 binding")

    artifact = Path(artifact_value).expanduser()
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise SystemExit(f"Pinned benchmark stock artifact does not exist: {artifact}")
    observed_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit(
            f"Pinned benchmark stock artifact SHA-256 mismatch for {catalog_key!r}: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    return {
        "artifact": artifact,
        "sha256": expected_sha256,
        "name": name,
        "catalog_key": catalog_key,
        "boundary_type": "benchmark_stock",
        "commercial_orderability_claimed": False,
    }


def _trusted_stock_snapshots_from_args(
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    """Load operator-selected supplier observations with exact content binding.

    Selecting an artifact is the operator trust decision.  The per-snapshot
    digest prevents silent mutation between export, campaign-policy binding,
    and current-host replay; it is integrity metadata, not a supplier
    signature or a live-availability claim.
    """

    snapshots: dict[str, dict[str, object]] = {}
    for raw_path in getattr(args, "trusted_stock_snapshot", None) or []:
        path = Path(str(raw_path or "")).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"Trusted stock snapshot artifact does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Cannot read trusted stock snapshot artifact {path}: {exc}"
            ) from exc

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and payload.get("schema_version") == (
            "trusted_stock_snapshots.v1"
        ):
            rows = payload.get("snapshots")
            if not isinstance(rows, list):
                raise SystemExit(
                    f"Trusted stock snapshot bundle has no snapshots list: {path}"
                )
        elif isinstance(payload, dict) and payload.get("schema_version") == (
            "stock_offer_snapshot.v1"
        ):
            rows = [payload]
        else:
            raise SystemExit(
                "Trusted stock snapshot artifact must contain "
                f"stock_offer_snapshot.v1, trusted_stock_snapshots.v1, or a list: {path}"
            )

        for index, raw in enumerate(rows):
            if not isinstance(raw, dict):
                raise SystemExit(
                    f"Trusted stock snapshot {path} row {index} is not an object"
                )
            supplied_sha256 = str(raw.get("snapshot_sha256") or "").strip().lower()
            try:
                canonical = canonicalize_stock_snapshot(raw)
                observed_sha256 = stock_snapshot_sha256(canonical)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Trusted stock snapshot {path} row {index} is invalid: {exc}"
                ) from exc
            if supplied_sha256 != observed_sha256:
                raise SystemExit(
                    f"Trusted stock snapshot {path} row {index} SHA-256 mismatch: "
                    f"expected {supplied_sha256 or '<missing>'}, observed {observed_sha256}"
                )
            canonical["snapshot_sha256"] = observed_sha256
            prior = snapshots.get(observed_sha256)
            if prior is not None and prior != canonical:
                raise SystemExit(
                    f"Trusted stock snapshot digest collision in {path} row {index}"
                )
            snapshots[observed_sha256] = canonical
    return snapshots


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


def _child_roles_from_args(args: argparse.Namespace) -> list[str]:
    roles = [
        item.strip()
        for item in str(args.codex_agent_team_child_roles or "").split(",")
        if item.strip()
    ]
    if len(set(roles)) < 2:
        raise SystemExit("--codex-agent-team-child-roles requires at least two distinct roles")
    return list(dict.fromkeys(roles))


def _acceptance_spec_from_args(
    args: argparse.Namespace,
) -> RetrosynthesisAcceptanceSpec:
    return RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=max(1, int(args.minimum_complete_routes)),
        minimum_edge_proof_level=int(args.minimum_edge_proof_level),
        require_all_selected_leaves_stock_closed=True,
        stock_boundary=str(args.codex_agent_team_closure_objective),
        minimum_independent_source_groups=max(
            1, int(args.minimum_independent_source_groups)
        ),
        require_distinct_edge_sets=True,
    )


def _run_budget_from_args(args: argparse.Namespace) -> RetrosynthesisRunBudget:
    return RetrosynthesisRunBudget(
        max_model_invocations=max(0, int(args.max_model_invocations)),
        max_total_input_tokens=max(0, int(args.max_total_input_tokens)),
        max_total_output_tokens=max(0, int(args.max_total_output_tokens)),
        max_total_wall_time_s=max(0.0, float(args.max_model_wall_time_s)),
        max_visual_invocations=max(0, int(args.max_visual_calls)),
        max_accepted_expansions=max(
            0, int(args.codex_agent_team_max_expansions)
        ),
        max_attempt_runs=max(0, int(args.codex_agent_team_max_attempt_runs)),
        max_prompt_context_bytes=max(0, int(args.max_prompt_context_bytes)),
        automatic_budget_extension=False,
    )


def _literature_sources_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in getattr(args, "literature_sources_file", None) or []:
        path = Path(str(raw_path or "")).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SystemExit(
                f"Unable to read --literature-sources-file {path}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid --literature-sources-file JSON {path}: {exc}"
            ) from exc
        if isinstance(data, dict) and "literature_sources" in data:
            data = data["literature_sources"]
        manifest_rows = data if isinstance(data, list) else [data]
        if not manifest_rows or any(not isinstance(item, dict) for item in manifest_rows):
            raise SystemExit(
                "--literature-sources-file must contain a source object or a "
                "non-empty list of source objects."
            )
        rows.extend(
            {str(key): value for key, value in item.items() if value is not None}
            for item in manifest_rows
        )
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
            rows.append({str(k): v for k, v in data.items() if v is not None})
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
