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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.open_research_contract import (
    OPEN_RESEARCH_JSON_SCHEMAS,
    REQUIRED_OPEN_RESEARCH_ARTIFACTS,
    REQUIRED_OPEN_RESEARCH_JSON_ARTIFACTS,
    validate_open_research_json_payload,
)
from cascade_planner.harness.open_research_experience import (
    audit_local_pdf_proxy_fallback,
    audit_open_research_boundary,
    extract_open_research_experience,
    write_open_research_manifest,
)
from cascade_planner.harness.open_research_retrieval import (
    prefetch_open_research_evidence,
    retrieval_prefetch_manifest_entry,
    validate_retrieval_prefetch_consumption,
    write_prefetch_checkpoint_seed,
    write_retrieval_prefetch_error,
)
from cascade_planner.harness.open_research_seed_consumables import (
    build_local_downstream_seed,
    write_local_downstream_seed_artifacts,
)
from cascade_planner.harness.source_detail_resolution import (
    resolve_source_detail_extraction_pack,
    source_detail_curator_records_path,
    source_detail_resolution_manifest_entry,
    write_source_detail_resolution_error,
)
from cascade_planner.harness.source_material_locator import (
    locate_source_materials,
    source_material_locator_manifest_entry,
    write_source_material_locator_error,
)

DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_WIRE_API = "responses"
BUFOTALIN_SCHEMA = "open_codex_structure_template_run.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--search-name", default="")
    parser.add_argument("--target-smiles", required=True)
    parser.add_argument("--frontier-smiles", default="")
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=os.environ.get("AUTOPLANNER_WORKER_API_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--model", default=os.environ.get("AUTOPLANNER_CODEX_WORKER_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--sandbox", choices=["workspace-write", "bypassed"], default="bypassed")
    parser.add_argument("--prompt-path", default="")
    parser.add_argument("--retrieval-timeout-s", type=float, default=6.0)
    parser.add_argument("--retrieval-max-results", type=int, default=5)
    parser.add_argument("--source-detail-timeout-s", type=float, default=10.0)
    parser.add_argument("--source-detail-max-items", type=int, default=5)
    parser.add_argument("--skip-retrieval-prefetch", action="store_true")
    parser.add_argument("--skip-source-detail-resolution", action="store_true")
    parser.add_argument(
        "--experience-path",
        default="",
        help="Optional prior open_research_experience.json used to bound and seed the next search.",
    )
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

    manifest = write_open_research_manifest(
        run_dir=run_dir,
        context_root=args.context_root,
        target_name=args.target_name,
        target_smiles=args.target_smiles,
        frontier_smiles=args.frontier_smiles,
        search_name=args.search_name,
        experience_path=args.experience_path or None,
    )
    retrieval_prefetch: dict[str, Any] = {}
    source_detail_resolution: dict[str, Any] = {}
    source_material_locator: dict[str, Any] = {}
    checkpoint_seed: dict[str, Any] = {}
    local_downstream_seed: dict[str, Any] = {}
    if not args.skip_retrieval_prefetch:
        try:
            retrieval_prefetch = prefetch_open_research_evidence(
                manifest,
                output_dir=run_dir,
                timeout_s=float(args.retrieval_timeout_s),
                max_results=int(args.retrieval_max_results),
            )
        except Exception as exc:
            retrieval_prefetch = write_retrieval_prefetch_error(
                output_dir=run_dir,
                manifest=manifest,
                error=f"{type(exc).__name__}: {exc}",
            )
        manifest["retrieval_prefetch"] = retrieval_prefetch_manifest_entry(
            retrieval_prefetch,
            output_dir=run_dir,
        )
        if not args.skip_source_detail_resolution:
            pack_path = Path(str(manifest["retrieval_prefetch"].get("source_detail_extraction_pack_path") or ""))
            try:
                source_detail_resolution = resolve_source_detail_extraction_pack(
                    pack_path,
                    output_dir=run_dir,
                    timeout_s=float(args.source_detail_timeout_s),
                    max_items=int(args.source_detail_max_items),
                )
            except Exception as exc:
                pack_payload = {}
                if pack_path.exists():
                    try:
                        pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pack_payload = {}
                source_detail_resolution = write_source_detail_resolution_error(
                    output_dir=run_dir,
                    pack=pack_payload,
                    error=f"{type(exc).__name__}: {exc}",
                )
            manifest["source_detail_resolution"] = source_detail_resolution_manifest_entry(
                source_detail_resolution,
                output_dir=run_dir,
            )
            try:
                source_material_locator = locate_source_materials(
                    pack_path,
                    output_dir=run_dir,
                    timeout_s=float(args.source_detail_timeout_s),
                    max_items=int(args.source_detail_max_items),
                )
            except Exception as exc:
                pack_payload = {}
                if pack_path.exists():
                    try:
                        pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pack_payload = {}
                source_material_locator = write_source_material_locator_error(
                    output_dir=run_dir,
                    extraction_pack=pack_payload,
                    error=f"{type(exc).__name__}: {exc}",
                )
            manifest["source_material_locator"] = source_material_locator_manifest_entry(
                source_material_locator,
                output_dir=run_dir,
            )
        (run_dir / "open_research_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_seed = write_prefetch_checkpoint_seed(
            output_dir=run_dir,
            manifest=manifest,
            prefetch=retrieval_prefetch,
            overwrite=False,
        )
        local_downstream_seed = write_local_downstream_seed_artifacts(
            output_dir=run_dir,
            seed=build_local_downstream_seed(manifest=manifest, output_dir=run_dir),
            overwrite=False,
        )
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
            "manifest_path": str(run_dir / "open_research_manifest.json"),
            "manifest_schema": manifest.get("schema_version"),
            "retrieval_prefetch_path": str(run_dir / "evidence" / "harness_retrieval_prefetch.json"),
            "retrieval_prefetch_enabled": not bool(args.skip_retrieval_prefetch),
            "retrieval_prefetch_record_counts": dict(retrieval_prefetch.get("record_counts") or {}),
            "source_detail_resolution_path": str(run_dir / "evidence" / "source_detail_resolution_pack.json"),
            "source_detail_resolution_enabled": not bool(args.skip_source_detail_resolution),
            "source_detail_resolution_summary": dict(source_detail_resolution.get("summary") or {}),
            "source_material_locator_path": str(run_dir / "evidence" / "source_material_locator_pack.json"),
            "source_material_locator_summary": dict(source_material_locator.get("summary") or {}),
            "prefetch_checkpoint_seed": checkpoint_seed,
            "local_downstream_seed": local_downstream_seed,
            "experience_path": str(Path(args.experience_path).resolve()) if args.experience_path else "",
            "context_root": str(Path(args.context_root).resolve()),
            "target_name": str(args.target_name),
            "search_name": str(args.search_name or ""),
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

    output_validation = _validate_open_agent_outputs(run_dir=run_dir, record=record)
    if source_detail_curator_records_path(run_dir).exists():
        manifest, source_detail_resolution, local_downstream_seed = _refresh_source_detail_resolution_from_curator(
            run_dir=run_dir,
            manifest=manifest,
            source_detail_timeout_s=float(args.source_detail_timeout_s),
            source_detail_max_items=int(args.source_detail_max_items),
        )
        metadata = dict(record.get("metadata") or {})
        metadata["source_detail_resolution_summary"] = dict(source_detail_resolution.get("summary") or {})
        metadata["source_material_locator_summary"] = dict((manifest.get("source_material_locator") or {}).get("summary") or {})
        metadata["local_downstream_seed"] = local_downstream_seed
        record["metadata"] = metadata
        output_validation = _validate_open_agent_outputs(run_dir=run_dir, record=record)
    record["output_validation"] = output_validation
    experience = extract_open_research_experience(run_dir=run_dir, run_record=record)
    record["generated_experience_path"] = str(run_dir / "open_research_experience.json")
    record["generated_experience_summary"] = {
        "observed_inefficiency_count": len(experience.get("observed_inefficiencies") or []),
        "suggested_policy_update_count": len(experience.get("suggested_policy_updates") or []),
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if not output_validation["accepted"]:
        print(str(run_dir))
        raise SystemExit(1)

    print(str(run_dir))


def _refresh_source_detail_resolution_from_curator(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    source_detail_timeout_s: float,
    source_detail_max_items: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefetch_entry = dict(manifest.get("retrieval_prefetch") or {})
    pack_path = Path(str(prefetch_entry.get("source_detail_extraction_pack_path") or ""))
    if not pack_path.exists():
        return manifest, {}, {}
    try:
        resolution = resolve_source_detail_extraction_pack(
            pack_path,
            output_dir=run_dir,
            timeout_s=float(source_detail_timeout_s),
            max_items=int(source_detail_max_items),
        )
    except Exception as exc:
        pack_payload = {}
        try:
            pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pack_payload = {}
        resolution = write_source_detail_resolution_error(
            output_dir=run_dir,
            pack=pack_payload,
            error=f"{type(exc).__name__}: {exc}",
        )
    manifest = dict(manifest)
    manifest["source_detail_resolution"] = source_detail_resolution_manifest_entry(
        resolution,
        output_dir=run_dir,
    )
    try:
        material_pack = locate_source_materials(
            pack_path,
            output_dir=run_dir,
            timeout_s=float(source_detail_timeout_s),
            max_items=int(source_detail_max_items),
        )
    except Exception as exc:
        pack_payload = {}
        try:
            pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pack_payload = {}
        material_pack = write_source_material_locator_error(
            output_dir=run_dir,
            extraction_pack=pack_payload,
            error=f"{type(exc).__name__}: {exc}",
        )
    manifest["source_material_locator"] = source_material_locator_manifest_entry(
        material_pack,
        output_dir=run_dir,
    )
    (run_dir / "open_research_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seed = write_local_downstream_seed_artifacts(
        output_dir=run_dir,
        seed=build_local_downstream_seed(manifest=manifest, output_dir=run_dir),
        overwrite=True,
    )
    return manifest, resolution, seed


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
    workflow_dir = context_root / "smiles_first_literature_workflow"
    if not workflow_dir.exists():
        workflow_dir = context_root / "smiles_first_after_chemenzy_stuck"
    if not workflow_dir.exists():
        workflow_dir = context_root / "smiles_first_after_native"
    target = str(args.target_name)
    frontier = str(args.frontier_smiles or "")
    schema_block = json.dumps(OPEN_RESEARCH_JSON_SCHEMAS, indent=2, ensure_ascii=False, sort_keys=True)
    return f"""You are a high-autonomy Codex chemistry extraction agent running inside AutoPlanner.

Goal: for {target} retrosynthesis after native ChemEnzy audit downgrade or failure, use literature evidence to produce downstream-consumable planning assets: structure/template candidates, guided ChemEnzy rerun requests, template drafts, route-segment drafts, and self-evolution candidates.

Operating mode:
- This is an open multi-step research run, not a bounded one-JSON worker.
- Use iterative searches, local file inspection, small helper scripts, and explicit audit trails.
- Keep weak evidence, failed lookups, and excluded sources, but label them honestly.
- Literature research is not the terminal step. After finding stuck-node or analog routes, emit draft consumables that deterministic validators can pass to guided ChemEnzy, route-segment unroll, template plugin replay, or selfEVO candidate layers.
- Harness may already have written a conservative prefetch checkpoint seed in the required artifact files. Preserve or enrich that checkpoint immediately; do not delete it. Write any improved checkpoint files before optional searches.
- Harness may also have written `harness_local_downstream_seed.json` and a conservative `downstream_consumables.json` from validated local evidence and route-failure feedback. Preserve its guided_rerun_requests, literature_template_cards, route_expansion_tasks, and evolution_candidates unless you replace them with stronger source-grounded versions.
- If the seed contains `executable_template_extraction_tasks`, treat them as the checklist for turning advisory literature into executable assets. Keep them when product/reactant SMILES are not yet source-grounded; replace or supplement them with `literature_route_segments` or `executable_template_candidates` only after exact RDKit-valid product/reactant structures are available.
- If a queue/source gives exact product and reactant structures for one step, you may emit `source_detail_route_steps` in downstream_consumables using schema_version `source_detail_route_step.v1`; harness will compile two or more same-segment steps into LiteratureRouteSegmentCard and one-step rows after validation.
- Do not wait for slow optional sources before writing the minimum required artifact set.
- Do not rely on an external prior run or undocumented "style"; this prompt is the full contract.

Use the repo-controlled manifest for runtime capabilities, case paths, search budgets, and forbidden operations. The harness-owned PubChem/CrossRef/PubMed, patent, DOI, and web retrieval artifacts are the default retrieval boundary. Shell/Python are allowed only for local deterministic transformation, RDKit validation, and JSON/schema checks over already available or manifest-listed data. Do not use shell for environment discovery, file discovery, process inspection, or external HTTP retrieval. Do not run RDKit availability/version probes; use `runtime_capabilities.rdkit_available` and `runtime_capabilities.rdkit_version` from the manifest, and only import RDKit when validating actual SMILES. Do not use curl/wget/urllib/requests/httpx for PubChem, CrossRef, PubMed, patent, DOI, or web retrieval; record lookup requests and provenance stubs instead when a typed connector is not exposed. Write all intermediate and final files under this workdir only:
{run_dir}

Current AutoPlanner context to read first:
- Target input: {context_root / "target_input.json"}
- Route verifier report, if present: {context_root / "route_verifier_report.json"}
- Route failure feedback, if present: {context_root / "route_failure_feedback.json"}
- Audited ChemEnzy baseline routes: {context_root / "chemenzy_baseline_routes.json"}
- SMILES-first workflow dir: {workflow_dir}
- Literature trigger report: {workflow_dir / "literature_trigger_report.json"}
- Existing bounded-worker evidence card, if any: {workflow_dir / "evidence_cards.jsonl"}
- Existing case bundle, if any: {workflow_dir / "case_bundle.json"}

Do not read the large native ChemEnzy raw route dump by default. Use `local_context.native_chemenzy_summary`, `route_verifier_report.json`, `route_failure_feedback.json`, and `route_audit.json` for route status and stuck-node evidence; raw route dumps are audit artifacts, not the open-research working context.

Target: {target}.
Target SMILES:
{args.target_smiles}
Stuck frontier from audit:
{frontier}

Repo-controlled search manifest:
- Read first: {run_dir / "open_research_manifest.json"}
- Treat the manifest as the binding search budget and source-order contract.
- Do not rediscover information already summarized in `local_context`.
- Read `retrieval_prefetch.source_detail_extraction_pack_path` before the larger `retrieval_prefetch.path` when it exists; it is the small harness-owned source-detail extraction queue.
- Do not read `retrieval_prefetch.path` / `evidence/harness_retrieval_prefetch.json` directly. It is a large raw retrieval audit. Use the manifest `retrieval_prefetch` summary, `source_detail_extraction_pack`, `source_detail_resolution`, and `source_material_locator` instead.
- Then read `source_detail_resolution.path` when present; it is the harness-owned DOI/PMC/source-detail resolution pack. If it contains `source_detail_route_steps`, merge those exact rows into downstream_consumables. If it contains only `extraction_gaps`, preserve those gaps and keep or create executable_template_extraction_tasks instead of fabricating structures.
- If `source_material_locator.path` is present, read it after source-detail resolution. It contains metadata-only publisher landing/SI/material URLs for unresolved DOI sources; use it only to plan structured extraction into `source_detail_curator_records.v1`, never as route evidence by itself.
- Agent access comes first. Dedupe candidate leads by DOI/URL/title, then inspect at most 5 exact-target or exact-intermediate leads per run and at most 3 native web/source checks per lead. Stop checking a lead after one of these statuses is clear: `agent_accessible_full_text`, `agent_accessible_metadata_only`, `agent_access_blocked_login_or_paywall`, or `agent_access_unavailable`.
- Before writing any local PDF fallback request, record a matching row in `evidence/literature_sources.json` `sources`, `excluded_sources`, or `search_log` with the same DOI/URL/source_ref, `agent_access_status`, and `content_scope` (`article`, `si`, `pdf`, `landing_page`, or `unknown`). Use separate rows when the article page is readable but SI/PDF is blocked.
- Only after the matching same-scope agent-access row says `agent_accessible_metadata_only`, `agent_access_blocked_login_or_paywall`, or `agent_access_unavailable`, write a fallback request to `local_pdf_proxy.request_queue_path` from the manifest. Do not queue local PDF fallback for the same DOI/URL/source_ref and same `content_scope` when it is marked `agent_accessible_full_text`. You may use `python {ROOT / "scripts" / "local_pdf_proxy.py"} --output-dir {run_dir} request --doi DOI --reason agent_access_failed_pdf_needed --content-scope article` for local deterministic queue writing; use `--content-scope si` for supporting information. Do not run the PDF proxy `fetch` command on the remote server. The user's local machine will fetch authorized PDFs through school/library access and sync back `local_pdf_proxy.download_manifest_path`; do not ask for or store institutional credentials, cookies, or browser profiles.
- You may translate source text, source tables, compound numbers, IUPAC/name descriptions, or structure diagrams into SMILES/template drafts when the source is exact-target or exact-intermediate relevant. Put these as structured records in `evidence/source_detail_curator_records.json` with `provenance: "codex_source_text_translation"` or as `source_detail_route_steps` in downstream_consumables. Include `structure_derivation` with basis, source_locator, confidence, and tool_checks; include a short `source_excerpt` only, not full text or procedures.
- If you manually extract exact structured product/reactant SMILES from a source table, SI, patent, or curator-owned record, write only structured fields to `evidence/source_detail_curator_records.json` using schema_version `source_detail_curator_records.v1`. Do not store source full text, copied procedures, raw reaction strings, or production KB writes in this file; include source_ref, evidence_refs, product_smiles, reactant_smiles, source-grounded condition_candidate, and structure_derivation when Codex inferred or translated the structure.
- If the extraction pack or `retrieval_prefetch.structured_extraction_queue` exists, process those queue items before broad searches. Each item names the source, relation hint, evidence/task refs, and required structured fields for LiteratureRouteSegmentCard or extraction-task update.
- If required artifacts already contain a harness prefetch checkpoint seed, upgrade them in place only after you have better source-grounded content. If you cannot complete enrichment, leave the seed files valid and summarize limitations.
- If `harness_local_downstream_seed.json` exists, read it before editing `downstream_consumables.json`; it is the minimum guided Chemenzy/selfEVO handoff when Codex cannot finish richer extraction.
- Consume `retrieval_prefetch.compound_seed_rows` into `evidence/pubchem_validated_compounds.json` when structurally relevant, preserving provenance with source `harness_retrieval_prefetch`. Do not bulk-copy all `source_seed_rows` into `sources`; keep only deduped exact-target/exact-intermediate/close-analog seeds as `sources`, put unrelated or query-only metadata into `excluded_sources` or compact `search_log` summaries.
- Use `research_policy.source_budgets`, `network_timeouts_s`, `skip_or_defer`, and `stop_rules` to decide what not to do.
- If prior_experience is present, apply its suggested policy updates before starting new searches.
- If prior_experience.self_evo_memory is present, use it only as reusable query/template seed material. Re-check current-target relation before using any item as evidence_refs; never treat memory as a solved route or production KB.
- If prior_experience.route_failure_feedback is present, prioritize its frontier_research_targets/query_hints and keep next_guided_policy_patch.terminal_blacklist out of closure claims.

Named intermediate source priority:
1. Names/structures already present in the local ChemEnzy and SMILES-first artifacts.
2. Names/synonyms returned by PubChem for the exact target, salts, esters, lactones, or validated target-family intermediates.
3. Intermediates named in exact-target or exact-intermediate synthesis/manufacturing literature.
4. Intermediates implied by validated strategic-disconnection cards, only if marked as hypothesis/template draft.
5. General chemistry intuition is allowed only for search query generation; do not promote it as evidence.

Source relation policy:
- Classify every source as exact_target, exact_intermediate, close_analog, family_only, method_reference, unrelated, or unusable.
- For DOI or sources unrelated to {target}'s chemical family, record them only as method_reference/unrelated/unusable.
- Do not let unrelated-family sources support {target} route candidates.
- If this prompt or local context mentions bufadienolide/bufotalin sources, treat them as method/style references only unless you prove a direct structural or target-family relationship. They must not enter {target} evidence_refs as chemical route support.

Operational requirements:
1. Inspect the local AutoPlanner context and summarize why native ChemEnzy was rejected or downgraded.
2. Plan literature/database lookups using the manifest query_plan and record lookup_request/search_log rows. Use native web/source access for search and for manifest-listed DOI/publisher/source access checks. Do not perform raw HTTP retrieval yourself. If a full-text/PDF/SI source is not readable to the agent, write the same-scope access outcome to `evidence/literature_sources.json` before queueing any `local_pdf_proxy` request; output validation rejects PDF fallback without that prior access record.
3. Search for exact-target and exact-intermediate sources first. Use harness source_triage/structured_extraction_queue rows as the source-order hint. If unrelated-family DOI/source references appear in the local context, classify them under source_relation_policy but exclude them from chemical evidence support.
4. Use RDKit only for structure validation/canonicalization of local or already retrieved SMILES. Canonicalize and compute formula/InChIKey where possible. Invalid SMILES must go to rejected_items.
5. Combine validated named structures with literature-described transformations/disconnections to propose route/template candidates. Keep weak evidence as route_anchor/template draft.
6. When literature gives one or more route steps with source-grounded structures, expand them into structured LiteratureRouteSegmentCard or SegmentStepCandidate drafts; do not stop at a prose summary. If the source gives enough structure/name/compound-number information for you to derive a SMILES with tools, emit a Codex source-text translation draft with provenance, structure_derivation, and a short source_excerpt. If any product/reactant SMILES remains unsupported, keep or create an `executable_template_extraction_tasks` row instead of fabricating structures.
   For incremental exact steps, `source_detail_route_steps` rows are acceptable if they include step_id, segment_id, product_smiles, reactant_smiles, source_ref, evidence_refs, relation_type=exact, applicability, and condition_candidate.
7. When literature gives a reusable transformation, emit LiteratureTemplateCard drafts. Emit ExecutableTemplateCandidate drafts only when every reactant/product SMILES is RDKit-valid, source-grounded, relation_type is exact, and the candidate is marked as requiring audit.
8. Emit guided_rerun_requests / RouteAnchorExpansionTask drafts when Chemenzy should continue from the discovered stuck node, intermediate, template, or route segment.
9. Emit EvolutionCandidate drafts for reusable templates/segments so selfEVO can retain them in candidate/shadow/staging layers. Never target production from this run.
10. Do not claim AutoPlanner solved the route unless all reactants/products parse and source evidence is structurally adequate. Do not promote anything to production KB.
11. Avoid raw reaction SMILES unless fully justified from source and RDKit-valid structures; non-executable template/segment drafts must stay structured.

Create these files, using exactly the schema_version and top-level fields specified below:
- structure_template_report.md
- structure_template_candidates.json
- downstream_consumables.json
- evidence/literature_sources.json
- evidence/pubchem_validated_compounds.json
- validated_compounds.smi
- open_agent_audit.json

JSON schema contract:
{schema_block}

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


def _validate_open_agent_outputs(*, run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    transport_reasons: list[str] = []
    warnings: list[str] = []
    if record.get("error"):
        transport_reasons.append(f"open_agent_{record['error']}")
    if record.get("exit_code") != 0:
        transport_reasons.append("open_agent_nonzero_or_missing_exit")

    metadata = dict(record.get("metadata") or {})
    if metadata.get("stream_jsonl"):
        event_log = Path(str(metadata.get("event_log_path") or run_dir / "codex_events.jsonl"))
        event_summary = _summarize_codex_events(event_log)
        if not event_summary.get("turn_completed"):
            transport_reasons.append("codex_events_missing_turn_completed")
        if not event_summary.get("usage"):
            transport_reasons.append("codex_events_missing_usage")
    else:
        event_summary = {}

    boundary_audit = audit_open_research_boundary(run_dir=run_dir)
    local_pdf_fallback_audit = audit_local_pdf_proxy_fallback(run_dir=run_dir)

    missing = [name for name in REQUIRED_OPEN_RESEARCH_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        reasons.extend(f"missing_open_agent_artifact:{name}" for name in missing)

    invalid_json: list[str] = []
    schema_reasons: list[str] = []
    for name in REQUIRED_OPEN_RESEARCH_JSON_ARTIFACTS:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            invalid_json.append(name)
            continue
        schema_reasons.extend(validate_open_research_json_payload(name=name, payload=payload))
    reasons.extend(f"invalid_open_agent_json:{name}" for name in invalid_json)
    reasons.extend(schema_reasons)
    reasons.extend(str(item) for item in boundary_audit.get("reasons") or [])
    reasons.extend(str(item) for item in local_pdf_fallback_audit.get("reasons") or [])
    retrieval_consumption = validate_retrieval_prefetch_consumption(run_dir=run_dir)
    reasons.extend(str(item) for item in retrieval_consumption.get("reasons") or [])

    checkpoint_valid = (
        not missing
        and not invalid_json
        and not schema_reasons
        and bool(boundary_audit.get("accepted", True))
        and bool(local_pdf_fallback_audit.get("accepted", True))
        and bool(retrieval_consumption.get("accepted", True))
    )
    checkpoint_after_timeout = bool(record.get("error") == "timeout" and checkpoint_valid)
    if checkpoint_after_timeout:
        warnings.extend(sorted(set(transport_reasons + ["checkpoint_valid_but_turn_timeout"])))
    else:
        reasons.extend(transport_reasons)

    return {
        "schema_version": "open_codex_structure_template_output_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "checkpoint_valid": checkpoint_valid,
        "checkpoint_after_timeout": checkpoint_after_timeout,
        "required_artifacts": list(REQUIRED_OPEN_RESEARCH_ARTIFACTS),
        "missing_artifacts": missing,
        "invalid_json_artifacts": invalid_json,
        "schema_reasons": schema_reasons,
        "event_summary": event_summary,
        "boundary_audit": boundary_audit,
        "local_pdf_fallback_audit": local_pdf_fallback_audit,
        "retrieval_prefetch_consumption": retrieval_consumption,
    }


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
