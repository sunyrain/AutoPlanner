"""Codex workflow planner wrapper and JSON plan validation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from cascade_planner.harness.schemas import (
    WORKFLOW_PLAN_SCHEMA,
    WorkflowPlan,
    append_jsonl,
    validate_workflow_plan,
    workflow_plan_from_dict,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_WIRE_API = "responses"


def plan_workflow_with_codex(
    *,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    run_dir: str | Path,
    timeout_s: float = 1800.0,
    key_path: str | Path = DEFAULT_KEY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run live Codex and return a validated workflow plan record."""
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    prompt = build_planner_prompt(target_input=target_input, preflight=preflight)
    (run_path / "codex_planner_prompt.txt").write_text(prompt, encoding="utf-8")

    executable = shutil.which("codex")
    if not executable:
        return _planner_error("codex_executable_not_found", run_path=run_path)
    api_key = _read_key(Path(key_path))
    if not api_key:
        return _planner_error("api_key_missing", run_path=run_path)

    event_log = run_path / "codex_events.jsonl"
    stderr_log = run_path / "codex_planner_stderr.log"
    last_message = run_path / "codex_planner_last_message.txt"
    command = [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "--cd",
        str(run_path),
        "--dangerously-bypass-approvals-and-sandbox",
        "--color",
        "never",
        "--output-last-message",
        str(last_message),
        "-",
    ]

    (run_path / "codex_planner_provider_config.toml").write_text(
        _codex_config_toml(base_url=str(base_url).rstrip("/"), model=str(model), run_dir=run_path),
        encoding="utf-8",
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="autoplanner_codex_planner_") as tmp:
        codex_home = Path(tmp) / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        _write_codex_home(
            codex_home=codex_home,
            api_key=api_key,
            base_url=str(base_url).rstrip("/"),
            model=str(model),
            run_dir=run_path,
        )
        env = os.environ.copy()
        env["CODEX_HOME"] = str(codex_home)
        env["OPENAI_API_KEY"] = api_key
        env.pop("OPENAI_BASE_URL", None)
        try:
            with event_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    command,
                    cwd=str(run_path),
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=err,
                    text=True,
                    env=env,
                )
                try:
                    proc.communicate(input=prompt, timeout=float(timeout_s))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    return _planner_error(
                        "codex_planner_timeout",
                        run_path=run_path,
                        command=command,
                        elapsed_s=time.monotonic() - started,
                    )
        except OSError as exc:
            return _planner_error(f"codex_planner_os_error:{type(exc).__name__}", run_path=run_path, command=command)

    text = last_message.read_text(encoding="utf-8", errors="replace") if last_message.exists() else ""
    try:
        payload = parse_workflow_plan_json(text)
    except ValueError as exc:
        return _planner_error(
            f"planner_json_parse_error:{exc}",
            run_path=run_path,
            command=command,
            elapsed_s=time.monotonic() - started,
            exit_code=proc.returncode,
        )

    plan = workflow_plan_from_dict(payload)
    validation = validate_workflow_plan(plan, case_id=str(preflight.get("case_id") or ""))
    record = {
        "schema_version": "codex_entry_planner_run.v1",
        "accepted": bool(validation.get("accepted")),
        "workflow_plan": plan.to_dict(),
        "validation": validation,
        "command": command,
        "exit_code": int(proc.returncode),
        "event_log_path": str(event_log),
        "stderr_log_path": str(stderr_log),
        "last_message_path": str(last_message),
        "elapsed_s": round(time.monotonic() - started, 3),
        "transport_contract": {
            "provider_wire_api": DEFAULT_WIRE_API,
            "codex_cli_event_stream": True,
            "required_cli_flag": "--json",
            "usage_location": "codex_events.jsonl turn.completed.usage",
        },
    }
    return record


def deterministic_workflow_plan(*, target_input: dict[str, Any], preflight: dict[str, Any]) -> WorkflowPlan:
    """Local fallback used by tests or explicit offline mode."""
    case_id = str(preflight.get("case_id") or target_input.get("case_id") or target_input.get("target_name") or "target")
    if not preflight.get("accepted"):
        return WorkflowPlan(
            case_id=case_id,
            recommended_strategy="reject_invalid_input",
            planned_tools=[],
            rationale="preflight rejected target input before live research",
            risk_flags=[str(item) for item in preflight.get("initial_risk_flags") or ["invalid_smiles"]],
            expected_verdict_floor="invalid_input",
        )
    flags = {str(item) for item in preflight.get("initial_risk_flags") or []}
    family = str(target_input.get("family_hint") or "").lower()
    if "glycoside" in family or "glycoside_or_o_glycoside_like" in flags:
        strategy = "literature_first"
        tools = [
            {"tool_name": "run_smiles_first_literature_workflow", "payload": {"frontier_smiles": target_input.get("target_smiles")}},
            {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
            {"tool_name": "validate_artifact_bundle", "payload": {}},
            {"tool_name": "emit_final_verdict", "payload": {}},
        ]
        floor = "partial_only"
        reason = "glycoside_or_o_glycoside_like"
    else:
        strategy = "hybrid" if flags else "chem_enzy_first"
        tools = [
            {"tool_name": "run_chemenzy", "payload": _default_chemenzy_payload(preflight=preflight)},
            {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
            {"tool_name": "run_smiles_first_literature_workflow", "payload": {"frontier_smiles": target_input.get("target_smiles")}},
            {"tool_name": "run_open_structure_research_agent", "payload": {}},
            {"tool_name": "run_guided_chemenzy_rerun", "payload": {}},
            {"tool_name": "run_route_expansion_subgoal_search", "payload": {}},
            {"tool_name": "run_self_evo_replay_gate", "payload": {}},
            {"tool_name": "validate_artifact_bundle", "payload": {}},
            {"tool_name": "emit_final_verdict", "payload": {}},
        ]
        floor = "needs_audit"
        reason = ""
    return WorkflowPlan(
        case_id=case_id,
        recommended_strategy=strategy,
        planned_tools=tools,
        rationale="deterministic local route selection for Codex-entry harness",
        risk_flags=sorted(flags),
        expected_verdict_floor=floor,
        planner_decision_reason=reason,
    )


def _default_chemenzy_payload(*, preflight: dict[str, Any]) -> dict[str, Any]:
    profile = dict(preflight.get("target_profile") or {})
    heavy_atoms = int(profile.get("heavy_atoms") or 0)
    if heavy_atoms >= 25:
        return {
            "search_preset": "thorough",
            "max_steps": 20,
            "chem_enzy_iterations": 50,
            "chem_enzy_expansion_topk": 100,
            "stock_mode": "building-block",
        }
    return {
        "search_preset": "quick",
        "max_steps": 6,
        "chem_enzy_iterations": 10,
        "chem_enzy_expansion_topk": 50,
        "stock_mode": "building-block",
    }


def build_planner_prompt(*, target_input: dict[str, Any], preflight: dict[str, Any]) -> str:
    allowed_tools = [
        "run_chemenzy",
        "audit_route_and_extract_frontier",
        "run_smiles_first_literature_workflow",
        "run_open_structure_research_agent",
        "extract_pdf_literature_structures",
        "validate_literature_intermediate_chain",
        "build_source_detail_curator_records",
        "compile_source_detail_chain_route",
        "compile_hybrid_route_set",
        "run_guided_chemenzy_rerun",
        "run_route_expansion_subgoal_search",
        "run_self_evo_replay_gate",
        "validate_artifact_bundle",
        "emit_final_verdict",
    ]
    return (
        "You are the AutoPlanner Codex-entry route controller. Return exactly one JSON object.\n"
        "Your job is only workflow planning. You may choose a route strategy and an ordered list of local tools.\n"
        "Do not solve the chemistry in this response. Do not emit raw reaction SMILES, raw reaction candidates, "
        "route tree mutations, production KB writes, or a solved verdict.\n\n"
        f"Required schema_version: {WORKFLOW_PLAN_SCHEMA}\n"
        "recommended_strategy must be one of: chem_enzy_first, literature_first, hybrid, reject_invalid_input.\n"
        f"Allowed local tool names: {', '.join(allowed_tools)}.\n"
        "Each planned_tools row must be an object with tool_name and optional payload.\n"
        "If recommended_strategy is literature_first, include planner_decision_reason as one of: "
        "user_requested_literature, glycoside_or_o_glycoside_like, natural_product_like, "
        "macrocycle_or_steroid_like, steroid_or_polycyclic_core, known_backend_unsuitable.\n"
        "For chem_enzy_first, the first executable tool must be run_chemenzy. For hybrid, "
        "run_smiles_first_literature_workflow and run_open_structure_research_agent must come after "
        "run_chemenzy and audit_route_and_extract_frontier unless literature_first is selected with an accepted reason.\n"
        "Codex may reason about literature later only through run_open_structure_research_agent; deterministic validators make the final verdict.\n\n"
        "Target input JSON:\n"
        f"{json.dumps(target_input, indent=2, ensure_ascii=False, sort_keys=True)}\n\n"
        "Preflight JSON:\n"
        f"{json.dumps(preflight, indent=2, ensure_ascii=False, sort_keys=True)}\n\n"
        "Return JSON with keys: schema_version, case_id, recommended_strategy, planned_tools, rationale, "
        "risk_flags, expected_verdict_floor, planner_decision_reason, run_semantics."
    )


def parse_workflow_plan_json(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if not value:
        raise ValueError("empty_planner_output")
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        data = _extract_first_json_object(value)
    if not isinstance(data, dict):
        raise ValueError("planner_output_not_object")
    return data


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no_json_object_found")


def _planner_error(
    reason: str,
    *,
    run_path: Path,
    command: list[str] | None = None,
    elapsed_s: float = 0.0,
    exit_code: int | None = None,
) -> dict[str, Any]:
    plan = WorkflowPlan(
        case_id="",
        recommended_strategy="reject_invalid_input",
        planned_tools=[],
        rationale=reason,
        risk_flags=["planner_error"],
        expected_verdict_floor="needs_followup",
    )
    validation = validate_workflow_plan(plan)
    record = {
        "schema_version": "codex_entry_planner_run.v1",
        "accepted": False,
        "workflow_plan": plan.to_dict(),
        "validation": validation,
        "reasons": [reason],
        "command": list(command or []),
        "exit_code": exit_code,
        "elapsed_s": round(float(elapsed_s), 3),
    }
    append_jsonl(run_path / "decision_trace.jsonl", {"stage": "codex_plan", "record": record})
    return record


def _write_codex_home(*, codex_home: Path, api_key: str, base_url: str, model: str, run_dir: Path) -> None:
    (codex_home / "auth.json").write_text(
        json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": api_key}, ensure_ascii=False),
        encoding="utf-8",
    )
    (codex_home / "config.toml").write_text(
        _codex_config_toml(base_url=base_url, model=model, run_dir=run_dir),
        encoding="utf-8",
    )


def _codex_config_toml(*, base_url: str, model: str, run_dir: Path) -> str:
    return "\n".join([
        f"model = {_toml_string(model)}",
        'model_provider = "wellau"',
        'model_reasoning_effort = "xhigh"',
        "",
        "[model_providers.wellau]",
        'name = "WellAU"',
        f"base_url = {_toml_string(base_url)}",
        'env_key = "OPENAI_API_KEY"',
        f"wire_api = {_toml_string(DEFAULT_WIRE_API)}",
        "",
        f"[projects.{_toml_string(str(run_dir))}]",
        'trust_level = "trusted"',
        "",
        "[features]",
        "goals = true",
        "",
    ])


def _read_key(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    for quote in ('"', "'"):
        if value.startswith(quote):
            value = value[1:]
        if value.endswith(quote):
            value = value[:-1]
    return value.strip()


def _toml_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
