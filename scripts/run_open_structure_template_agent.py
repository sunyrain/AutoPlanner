#!/usr/bin/env python3
"""Launch an open Codex structure/template exploration agent.

This intentionally mirrors the Bufotalin open-agent run contract:
ephemeral CODEX_HOME, repository key.txt auth, WellAU-compatible base URL,
and an unconstrained Codex CLI process that writes audited files in a run dir.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_WIRE_API = "responses"
BUFOTALIN_SCHEMA = "open_codex_structure_template_run.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--frontier-smiles", default="")
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=os.environ.get("AUTOPLANNER_WORKER_API_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--model", default=os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--sandbox", choices=["workspace-write", "bypassed"], default="bypassed")
    parser.add_argument("--prompt-path", default="")
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream-jsonl",
        dest="stream_jsonl",
        action="store_true",
        default=True,
        help=(
            "Run `codex exec --json` and write the streaming JSONL event feed "
            "under the run directory. This is the default and preserves "
            "`turn.completed.usage`."
        ),
    )
    stream_group.add_argument(
        "--no-stream-jsonl",
        dest="stream_jsonl",
        action="store_false",
        help="Disable Codex CLI JSONL event streaming. Not recommended for audited runs.",
    )
    args = parser.parse_args()

    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence").mkdir(exist_ok=True)
    (run_dir / "scripts").mkdir(exist_ok=True)

    prompt = _read_or_build_prompt(args=args, run_dir=run_dir)
    prompt_path = run_dir / "open_agent_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    config_snapshot_path = run_dir / "codex_provider_config.toml"
    config_snapshot_path.write_text(
        _codex_config_toml(
            base_url=str(args.base_url).rstrip("/"),
            model=str(args.model),
            run_dir=run_dir,
        ),
        encoding="utf-8",
    )

    executable = shutil.which("codex")
    if not executable:
        raise SystemExit("codex executable not found on PATH")

    key_path = Path(args.key_path).resolve()
    api_key = _read_key(key_path)
    if not api_key:
        raise SystemExit(f"API key missing or empty: {key_path}")

    last_message = run_dir / "last_message.txt"
    event_log = run_dir / "codex_events.jsonl"
    stderr_log = run_dir / "codex_stderr.log"
    command = [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        *(["--json"] if args.stream_jsonl else []),
        "--cd",
        str(run_dir),
    ]
    if args.sandbox == "bypassed":
        command.append("--dangerously-bypass-approvals-and-sandbox")
        sandbox_metadata = "bypassed"
    else:
        command.extend(["--sandbox", "workspace-write"])
        sandbox_metadata = "workspace-write"
    command.extend([
        "--color",
        "never",
        "--output-last-message",
        str(last_message),
        "-",
    ])

    record_path = run_dir / "open_agent_run_record.json"
    record: dict[str, Any] = {
        "schema_version": BUFOTALIN_SCHEMA,
        "run_dir": str(run_dir),
        "last_message_path": str(last_message),
        "command": command,
        "metadata": {
            "auth_source": str(key_path),
            "provider": "wellau",
            "base_url": str(args.base_url).rstrip("/"),
            "model": str(args.model),
            "wire_api": DEFAULT_WIRE_API,
            "sandbox": sandbox_metadata,
            "codex_home": "ephemeral",
            "config_snapshot_path": str(config_snapshot_path),
            "launcher": "scripts/run_open_structure_template_agent.py",
            "prompt_path": str(prompt_path),
            "context_root": str(Path(args.context_root).resolve()),
            "target_name": str(args.target_name),
            "stream_jsonl": bool(args.stream_jsonl),
            "event_log_path": str(event_log) if args.stream_jsonl else "",
            "stderr_log_path": str(stderr_log) if args.stream_jsonl else "",
            "transport_contract": _transport_contract(stream_jsonl=bool(args.stream_jsonl)),
            "parent_openai_base_url_present": "OPENAI_BASE_URL" in os.environ,
        },
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="autoplanner_codex_open_agent_") as tmp:
        codex_home = Path(tmp) / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        _write_codex_home(
            codex_home=codex_home,
            api_key=api_key,
            base_url=str(args.base_url).rstrip("/"),
            model=str(args.model),
            run_dir=run_dir,
        )
        env = os.environ.copy()
        env["CODEX_HOME"] = str(codex_home)
        # Custom Codex providers read their key from env_key, not auth.json.
        env["OPENAI_API_KEY"] = api_key
        # Do not let ambient OpenAI-compatible settings bypass the explicit
        # WellAU provider config below.
        env.pop("OPENAI_BASE_URL", None)

        try:
            if args.stream_jsonl:
                record.update(
                    _run_streaming_codex(
                        command=command,
                        cwd=run_dir,
                        prompt=prompt,
                        timeout_s=float(args.timeout_s),
                        env=env,
                        event_log=event_log,
                        stderr_log=stderr_log,
                    )
                )
            else:
                proc = subprocess.run(
                    command,
                    cwd=str(run_dir),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=float(args.timeout_s),
                    check=False,
                    env=env,
                )
                record.update({
                    "exit_code": int(proc.returncode),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                })
        except subprocess.TimeoutExpired as exc:
            record.update({
                "exit_code": None,
                "timeout_s": float(args.timeout_s),
                "stdout": _decode_timeout_stream(exc.stdout),
                "stderr": _decode_timeout_stream(exc.stderr),
                "error": "timeout",
            })
        finally:
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(str(run_dir))


def _run_streaming_codex(
    *,
    command: list[str],
    cwd: Path,
    prompt: str,
    timeout_s: float,
    env: dict[str, str],
    event_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    with event_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            text=True,
            env=env,
        )
        try:
            proc.communicate(input=prompt, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
    return {
        "exit_code": int(proc.returncode),
        "stdout": "",
        "stderr": "",
        "streaming": {
            "mode": "codex_exec_jsonl",
            "event_log_path": str(event_log),
            "stderr_log_path": str(stderr_log),
            "elapsed_s": round(time.monotonic() - started, 3),
            "event_summary": _summarize_codex_events(event_log),
        },
    }


def _read_or_build_prompt(*, args: argparse.Namespace, run_dir: Path) -> str:
    if args.prompt_path:
        return Path(args.prompt_path).read_text(encoding="utf-8")
    context_root = Path(args.context_root).resolve()
    workflow_dir = context_root / "smiles_first_after_chemenzy_stuck"
    if not workflow_dir.exists():
        workflow_dir = context_root / "smiles_first_after_native"
    target = str(args.target_name)
    frontier = str(args.frontier_smiles or "")
    return f"""You are a high-autonomy Codex chemistry extraction agent running inside AutoPlanner.

Goal: for {target} retrosynthesis after native ChemEnzy audit failure, use your own tools to turn database/literature evidence into validated structure/template candidates. This should reproduce the open Bufotalin structure/template agent style from 2026-06-05 around 00:20, not a bounded one-JSON worker.

You may use shell commands, Python, curl, PubChem/PUG-REST, CrossRef, PubMed/NCBI E-utilities, RDKit, local files, live web search, and downloads. Write all intermediate and final files under this workdir only:
{run_dir}

Current AutoPlanner context to read first:
- Target input: {context_root / "target_input.json"}
- Native ChemEnzy raw result: {context_root / "chemenzy_native_raw_result.json"}
- Audited ChemEnzy baseline routes: {context_root / "chemenzy_baseline_routes.json"}
- SMILES-first workflow dir: {workflow_dir}
- Literature trigger report: {workflow_dir / "literature_trigger_report.json"}
- Existing bounded-worker evidence card, if any: {workflow_dir / "evidence_cards.jsonl"}
- Existing case bundle, if any: {workflow_dir / "case_bundle.json"}

Target: {target}.
Target SMILES:
{args.target_smiles}
Stuck frontier from audit:
{frontier}

Operational requirements:
1. Inspect the local AutoPlanner context and summarize why native ChemEnzy was rejected or downgraded.
2. Query literature/databases yourself. At minimum attempt PubChem/PUG-REST for the target and plausible named intermediates, CrossRef/DOI metadata for route-relevant synthesis/manufacturing papers, PubMed/NCBI where applicable, patents/metadata pages, and live web/search.
3. Explicitly investigate DOI 10.1021/acs.orglett.0c03251, DOI 10.1021/acs.joc.6c00124, and related bufadienolide synthesis sources such as DOI 10.1021/acs.orglett.4c00625. Search for additional exact-target or exact-intermediate sources. Record whether each source is exact-target, exact-intermediate, close analog, family-only, or unusable.
4. Use RDKit to validate every non-null SMILES you introduce. Canonicalize and compute formula/InChIKey where possible. Invalid SMILES must go to rejected_items.
5. Combine validated named structures with literature-described transformations/disconnections to propose route/template candidates. Keep weak evidence as route_anchor/template draft.
6. Do not claim AutoPlanner solved the route unless all reactants/products parse and source evidence is structurally adequate. Do not promote anything to production KB.
7. Avoid raw reaction SMILES unless fully justified from source and RDKit-valid structures.

Create these files:
- structure_template_report.md
- structure_template_candidates.json
- evidence/literature_sources.json
- evidence/pubchem_validated_compounds.json, if PubChem queries produce records
- validated_compounds.smi, if any named compounds have validated SMILES
- open_agent_audit.json

The JSON files must parse. Final response should include file paths and a short verdict. Do not mark AutoPlanner solved and do not promote to production KB.
"""


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
    return (
        "\n".join([
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
    )


def _transport_contract(*, stream_jsonl: bool) -> dict[str, Any]:
    return {
        "provider": "wellau",
        "provider_wire_api": DEFAULT_WIRE_API,
        "codex_cli_event_stream": bool(stream_jsonl),
        "required_cli_flag": "--json" if stream_jsonl else "",
        "usage_location": "codex_events.jsonl turn.completed.usage",
        "known_bad_modes": [
            "missing `codex exec --json`: no structured event stream or usage event",
            "ambient OPENAI_BASE_URL overriding explicit provider config",
            "workspace-write sandbox on hosts where bwrap cannot create namespaces",
            "confusing provider wire_api with Codex CLI JSONL streaming",
        ],
    }


def _summarize_codex_events(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "event_count": 0,
        "turn_completed": False,
        "usage": None,
        "last_event_type": "",
    }
    if not path.exists():
        return summary
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary["event_count"] = int(summary["event_count"]) + 1
        summary["last_event_type"] = str(event.get("type") or "")
        if event.get("type") == "turn.completed":
            summary["turn_completed"] = True
            summary["usage"] = event.get("usage")
    return summary


def _read_key(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    for quote in ('"', "'"):
        if value.startswith(quote):
            value = value[1:]
        if value.endswith(quote):
            value = value[:-1]
    return value.strip()


def _toml_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


if __name__ == "__main__":
    main()
