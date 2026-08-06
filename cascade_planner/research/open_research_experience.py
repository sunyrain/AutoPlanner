"""Bounded search manifests and experience extraction for open research runs."""
from __future__ import annotations

import json
import re
import shutil
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cascade_planner.harness.local_pdf_proxy import (
    load_pdf_requests,
    local_pdf_proxy_manifest_entry,
    local_pdf_proxy_request_queue_path,
    normalize_doi,
)
from cascade_planner.research.open_research_retrieval import retrieval_prefetch_manifest_entry
from cascade_planner.research.source_material_locator import source_material_locator_manifest_entry
from cascade_planner.research.route_failure_feedback import ROUTE_FAILURE_FEEDBACK_SCHEMA


OPEN_RESEARCH_MANIFEST_SCHEMA = "open_structure_research_manifest.v1"
OPEN_RESEARCH_EXPERIENCE_SCHEMA = "open_structure_research_experience.v1"
OPEN_RESEARCH_BOUNDARY_AUDIT_SCHEMA = "open_structure_research_boundary_audit.v1"
OPEN_RESEARCH_LOCAL_PDF_FALLBACK_AUDIT_SCHEMA = "open_structure_research_local_pdf_fallback_audit.v1"

AGENT_ACCESS_FULL_TEXT_STATUS = "agent_accessible_full_text"
AGENT_ACCESS_FALLBACK_STATUSES = {
    "agent_accessible_metadata_only",
    "agent_access_blocked_login_or_paywall",
    "agent_access_unavailable",
}


def write_open_research_manifest(
    *,
    run_dir: str | Path,
    context_root: str | Path,
    target_name: str,
    target_smiles: str,
    frontier_smiles: str = "",
    search_name: str = "",
    experience_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a repo-controlled manifest that bounds open-agent discovery."""
    run_path = Path(run_dir)
    manifest = build_open_research_manifest(
        run_dir=run_path,
        context_root=context_root,
        target_name=target_name,
        target_smiles=target_smiles,
        frontier_smiles=frontier_smiles,
        search_name=search_name,
        experience_path=experience_path,
    )
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "open_research_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_open_research_manifest(
    *,
    run_dir: str | Path,
    context_root: str | Path,
    target_name: str,
    target_smiles: str,
    frontier_smiles: str = "",
    search_name: str = "",
    experience_path: str | Path | None = None,
) -> dict[str, Any]:
    context = Path(context_root).resolve()
    workflow_dir = _workflow_dir(context)
    target = str(target_name or "target")
    family_hint = _target_family_hint(context)
    query_target = _search_name(
        target_name=target,
        family_hint=family_hint,
        explicit=search_name,
    )
    local_summary = _local_context_summary(context=context, workflow_dir=workflow_dir)
    experience = _load_prior_experience(experience_path)
    self_evo_memory = _load_self_evo_memory(experience_path)
    route_failure_feedback = _load_route_failure_feedback(
        experience_path,
        context_root=context,
    )
    return {
        "schema_version": OPEN_RESEARCH_MANIFEST_SCHEMA,
        "target": {
            "name": target,
            "search_name": query_target,
            "smiles": str(target_smiles or ""),
            "frontier_smiles": str(frontier_smiles or ""),
            "family_hint": family_hint,
        },
        "run_dir": str(Path(run_dir).resolve()),
        "context_root": str(context),
        "runtime_capabilities": _runtime_capabilities(run_dir=Path(run_dir)),
        "case_manifest": _case_manifest(context=context, workflow_dir=workflow_dir),
        "local_context": local_summary,
        "research_policy": _research_policy(experience),
        "operation_boundary": _operation_boundary(run_dir=Path(run_dir)),
        "retrieval_prefetch": retrieval_prefetch_manifest_entry(None, output_dir=run_dir),
        "source_material_locator": source_material_locator_manifest_entry(None, output_dir=run_dir),
        "local_pdf_proxy": local_pdf_proxy_manifest_entry(None, output_dir=run_dir),
        "query_plan": _query_plan(
            target=query_target,
            family_hint=family_hint,
            experience=experience,
            self_evo_memory=self_evo_memory,
            route_failure_feedback=route_failure_feedback,
        ),
        "prior_experience": {
            **experience,
            **({"self_evo_memory": self_evo_memory} if self_evo_memory else {}),
            **({"route_failure_feedback": route_failure_feedback} if route_failure_feedback else {}),
        },
    }


def extract_open_research_experience(
    *,
    run_dir: str | Path,
    run_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize useful process lessons from a completed or timed-out open run."""
    run_path = Path(run_dir)
    event_log = run_path / "codex_events.jsonl"
    run_record = dict(run_record or _load_json(run_path / "open_agent_run_record.json", {}))
    counters: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    web_queries: list[str] = []
    completed_commands: list[dict[str, Any]] = []
    inefficiencies: set[str] = set()

    for event in _iter_jsonl(event_log):
        counters[str(event.get("type") or "unknown")] += 1
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type") or "")
        if item_type:
            counters[f"item:{item_type}"] += 1
        if event.get("type") != "item.completed":
            continue
        if item_type == "command_execution":
            command = str(item.get("command") or "")
            output = str(item.get("aggregated_output") or "")
            exit_code = item.get("exit_code")
            completed_commands.append({
                "command": _compact(command, 240),
                "exit_code": exit_code,
                "status": item.get("status"),
            })
            for domain in _domains_from_text(command + "\n" + output):
                domains[domain] += 1
            _classify_command_inefficiencies(command, output, exit_code, inefficiencies)
        elif item_type == "web_search":
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            queries = action.get("queries") or ([action.get("query")] if action.get("query") else [])
            top_query = str(item.get("query") or action.get("query") or "")
            for domain in _domains_from_text("\n".join([top_query, *[str(query) for query in queries if query]])):
                domains[domain] += 1
            if _is_direct_url_query(top_query) or any(_is_direct_url_query(str(query)) for query in queries):
                inefficiencies.add("direct_url_web_search_without_connector")
            for query in queries:
                if query:
                    web_queries.append(str(query))

    validation = dict(run_record.get("output_validation") or {})
    reasons = [str(item) for item in validation.get("reasons") or []]
    missing_artifacts = [str(item) for item in validation.get("missing_artifacts") or []]
    if run_record.get("error") == "timeout" or "open_agent_timeout" in reasons:
        if missing_artifacts:
            inefficiencies.add("open_agent_timeout_before_required_artifacts")
        else:
            inefficiencies.add("open_agent_timeout_after_required_artifacts")
    if any(reason.startswith("missing_open_agent_artifact:") for reason in reasons):
        inefficiencies.add("minimum_artifacts_not_checkpointed_before_optional_work")
    if not validation.get("event_summary", {}).get("turn_completed") and event_log.exists():
        inefficiencies.add("missing_turn_completed_usage_trace")

    policy_updates = _policy_updates_from_inefficiencies(inefficiencies)
    experience = {
        "schema_version": OPEN_RESEARCH_EXPERIENCE_SCHEMA,
        "run_dir": str(run_path.resolve()),
        "event_counts": dict(sorted(counters.items())),
        "domain_counts": dict(sorted(domains.items())),
        "web_queries": web_queries[:40],
        "completed_command_count": len(completed_commands),
        "failed_command_count": sum(1 for row in completed_commands if row.get("exit_code") not in (0, None)),
        "observed_inefficiencies": sorted(inefficiencies),
        "suggested_policy_updates": policy_updates,
        "reusable_search_hints": _reusable_search_hints(web_queries=web_queries, domains=domains),
    }
    (run_path / "open_research_experience.json").write_text(
        json.dumps(experience, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return experience


def audit_open_research_boundary(*, run_dir: str | Path) -> dict[str, Any]:
    """Audit Codex event logs for operations that should be harness-owned."""
    run_path = Path(run_dir)
    violations: list[dict[str, Any]] = []
    for event in _iter_jsonl(run_path / "codex_events.jsonl"):
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type") or "")
        if event.get("type") != "item.completed":
            continue
        if item_type == "command_execution":
            command = str(item.get("command") or "")
            output = str(item.get("aggregated_output") or "")
            for violation in _boundary_command_violations(command, output=output):
                violations.append(violation)
        elif item_type == "web_search":
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            action_type = str(action.get("type") or "").strip()
            query = str(item.get("query") or action.get("query") or "").strip()
            queries = [str(row).strip() for row in action.get("queries") or [] if str(row).strip()]
            if action_type and action_type not in {"search", "query"} and not query and not queries:
                continue
            if len(query) < 3 and not queries:
                violations.append({
                    "category": "query_policy",
                    "reason": "empty_or_too_short_web_search_query",
                    "detail": "web_search was called without a meaningful query.",
                })
            for row in queries:
                if len(row) < 3:
                    violations.append({
                        "category": "query_policy",
                    "reason": "empty_or_too_short_web_search_query",
                    "detail": row,
                })
    reasons = [
        f"open_agent_boundary_violation:{item['category']}:{item['reason']}"
        for item in violations
    ]
    return {
        "schema_version": OPEN_RESEARCH_BOUNDARY_AUDIT_SCHEMA,
        "accepted": not violations,
        "reasons": sorted(set(reasons)),
        "violations": violations,
    }


def audit_local_pdf_proxy_fallback(*, run_dir: str | Path) -> dict[str, Any]:
    """Require agent access failure records before local PDF proxy fallback requests."""
    run_path = Path(run_dir)
    queue_path = local_pdf_proxy_request_queue_path(run_path)
    violations: list[dict[str, str]] = []
    try:
        requests = load_pdf_requests(queue_path) if queue_path.exists() else []
    except Exception as exc:
        requests = []
        violations.append({
            "category": "local_pdf_proxy",
            "reason": "invalid_pdf_request_queue",
            "detail": f"{queue_path}: {exc}",
        })
    if not requests:
        return {
            "schema_version": OPEN_RESEARCH_LOCAL_PDF_FALLBACK_AUDIT_SCHEMA,
            "accepted": not violations,
            "reasons": [
                f"open_agent_boundary_violation:{item['category']}:{item['reason']}"
                for item in violations
            ],
            "request_count": 0,
            "agent_access_record_count": 0,
            "violations": violations,
        }

    access_records = _agent_access_records(run_path / "evidence" / "literature_sources.json")
    for request in requests:
        request_scope = _request_content_scope(request)
        if not request_scope:
            violations.append({
                "category": "local_pdf_proxy",
                "reason": "missing_pdf_request_content_scope",
                "detail": _pdf_request_detail(request),
            })
            continue
        matches = [
            record
            for record in access_records
            if _pdf_request_matches_access_record(request, record)
        ]
        scoped_matches = [
            record
            for record in matches
            if _content_scope_matches(request_scope, _record_content_scope(record))
        ]
        if not scoped_matches:
            if matches:
                violations.append({
                    "category": "local_pdf_proxy",
                    "reason": "missing_or_mismatched_agent_access_content_scope",
                    "detail": _pdf_request_detail(request),
                })
                continue
            violations.append({
                "category": "local_pdf_proxy",
                "reason": "missing_agent_access_failure_record",
                "detail": _pdf_request_detail(request),
            })
            continue
        statuses = sorted(set(_agent_access_status(record) for record in scoped_matches if _agent_access_status(record)))
        if AGENT_ACCESS_FULL_TEXT_STATUS in statuses:
            violations.append({
                "category": "local_pdf_proxy",
                "reason": "fallback_requested_despite_agent_full_text",
                "detail": _pdf_request_detail(request),
            })
            continue
        if not any(status in AGENT_ACCESS_FALLBACK_STATUSES for status in statuses):
            violations.append({
                "category": "local_pdf_proxy",
                "reason": "fallback_request_without_failed_agent_access",
                "detail": f"{_pdf_request_detail(request)} statuses={statuses}",
            })
    reasons = [
        f"open_agent_boundary_violation:{item['category']}:{item['reason']}"
        for item in violations
    ]
    return {
        "schema_version": OPEN_RESEARCH_LOCAL_PDF_FALLBACK_AUDIT_SCHEMA,
        "accepted": not violations,
        "reasons": sorted(set(reasons)),
        "request_count": len(requests),
        "agent_access_record_count": len(access_records),
        "violations": violations,
    }


def _agent_access_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    records: list[dict[str, Any]] = []
    for key in ("sources", "excluded_sources", "search_log"):
        for row in payload.get(key) or []:
            if not isinstance(row, dict):
                continue
            status = _agent_access_status(row)
            if status:
                record = dict(row)
                record["_agent_access_status"] = status
                records.append(record)
    return records


def _agent_access_status(record: dict[str, Any]) -> str:
    for key in (
        "_agent_access_status",
        "agent_access_status",
        "remote_agent_access_status",
        "source_access_status",
        "full_text_access_status",
        "access_status",
        "status",
    ):
        value = str(record.get(key) or "").strip()
        if value.startswith("agent_access"):
            return value
    outcome = str(record.get("agent_access_outcome") or record.get("access_outcome") or "").strip()
    if outcome.startswith("agent_access"):
        return outcome
    return ""


def _pdf_request_matches_access_record(request: dict[str, Any], record: dict[str, Any]) -> bool:
    request_doi = normalize_doi(str(request.get("doi") or request.get("url") or request.get("source_ref") or ""))
    record_doi = normalize_doi(" ".join(_record_text_values(record)))
    if request_doi and record_doi and request_doi == record_doi:
        return True

    request_url = _normalize_access_url(str(request.get("url") or ""))
    record_urls = {_normalize_access_url(value) for value in _record_text_values(record) if value.startswith("http")}
    if request_url and request_url in record_urls:
        return True

    request_ref = str(request.get("source_ref") or "").strip().lower()
    if request_ref and request_ref in {value.strip().lower() for value in _record_text_values(record)}:
        return True
    return False


def _request_content_scope(request: dict[str, Any]) -> str:
    return _normalize_content_scope(" ".join([
        str(request.get("content_scope") or ""),
        str(request.get("requested_content_scope") or ""),
        str(request.get("material_type") or ""),
        str(request.get("source_ref") or ""),
        str(request.get("url") or ""),
    ]))


def _record_content_scope(record: dict[str, Any]) -> str:
    return _normalize_content_scope(" ".join([
        str(record.get("content_scope") or ""),
        str(record.get("access_content_scope") or ""),
        str(record.get("requested_content_scope") or ""),
        str(record.get("material_type") or ""),
        str(record.get("source_ref") or ""),
        str(record.get("url") or ""),
        str(record.get("query") or ""),
    ]))


def _normalize_content_scope(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if (
        any(marker in text for marker in ("supporting", "supplement", "_si", "-si", "s.i."))
        or re.search(r"(^|[\s_:/.-])si($|[\s_:/.-])", text)
    ):
        return "si"
    if "pdf" in text:
        return "pdf"
    if "landing" in text or "abstract" in text:
        return "landing_page"
    if "article" in text or "full_text" in text or "full text" in text:
        return "article"
    if text in {"si", "pdf", "article", "landing_page", "unknown"}:
        return text
    return ""


def _content_scope_matches(request_scope: str, record_scope: str) -> bool:
    if not request_scope or not record_scope:
        return False
    if request_scope == record_scope:
        return True
    if request_scope == "pdf" and record_scope == "article":
        return True
    return False


def _record_text_values(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "doi",
        "DOI",
        "url",
        "source_url",
        "source_ref",
        "source_locator",
        "source",
        "query",
        "title",
        "citation",
        "detail",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _normalize_access_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return parsed._replace(fragment="", query="").geturl().rstrip("/").lower()


def _pdf_request_detail(request: dict[str, Any]) -> str:
    bits = [
        str(request.get("request_id") or ""),
        str(request.get("doi") or ""),
        str(request.get("url") or ""),
        str(request.get("source_ref") or ""),
    ]
    return _compact(" ".join(bit for bit in bits if bit), 320)


def _workflow_dir(context: Path) -> Path:
    for name in (
        "smiles_first_literature_workflow",
        "smiles_first_after_chemenzy_stuck",
        "smiles_first_after_native",
    ):
        path = context / name
        if path.exists():
            return path
    return context / "smiles_first_literature_workflow"


def _target_family_hint(context: Path) -> str:
    target_input = _load_json(context / "target_input.json", {})
    return str(target_input.get("family_hint") or "")


def _search_name(*, target_name: str, family_hint: str, explicit: str = "") -> str:
    explicit = str(explicit or "").strip()
    if explicit:
        return explicit
    target = str(target_name or "").strip()
    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", f"{target} {family_hint}")
        if token
    ]
    for token in tokens:
        lower = token.lower()
        if len(lower) > 6 and lower.endswith("statin"):
            return lower
    if "_" in target:
        prefix = target.split("_", 1)[0].strip()
        if prefix and prefix.isalpha() and len(prefix) > 2:
            return prefix
    return target or "target"


def _local_context_summary(*, context: Path, workflow_dir: Path) -> dict[str, Any]:
    target_input = _load_json(context / "target_input.json", {})
    native = _load_json(context / "chemenzy_native_raw_result.json", {})
    baseline = _load_json(context / "chemenzy_baseline_routes.json", {})
    route_audit = _load_json(context / "route_audit.json", {})
    route_failure_feedback = _load_json(context / "route_failure_feedback.json", {})
    trigger = _load_json(workflow_dir / "literature_trigger_report.json", {})
    evidence_cards = _load_jsonl(workflow_dir / "evidence_cards.jsonl")
    candidates = _load_jsonl(workflow_dir / "fluvastatin_literature_rxn_candidates.jsonl")
    return {
        "recommended_read_order": [
            str(context / "target_input.json"),
            str(context / "route_verifier_report.json"),
            str(context / "route_failure_feedback.json"),
            str(context / "route_audit.json"),
            str(workflow_dir / "literature_trigger_report.json"),
            str(workflow_dir / "evidence_cards.jsonl"),
        ],
        "skip_local_rediscovery": [
            "Avoid broad recursive grep unless a required manifest file is missing.",
            "Use route_verifier_report/route_failure_feedback/local_context before reopening large raw route payloads.",
            "Do not read chemenzy_native_raw_result.json by default; it is raw audit data, not open-research context.",
        ],
        "target_input": {
            "case_id": target_input.get("case_id"),
            "target_name": target_input.get("target_name"),
            "family_hint": target_input.get("family_hint"),
            "target_smiles_present": bool(target_input.get("target_smiles")),
        },
        "native_chemenzy_summary": {
            "ok": native.get("ok"),
            "n_results": native.get("n_results"),
            "reported_solved": dict(native.get("search_status") or {}).get("solved"),
            "search_status": dict(native.get("search_status") or {}),
            "route_count": len(native.get("routes") or []),
        },
        "baseline_summary": {
            "status": baseline.get("status"),
            "solved": baseline.get("solved"),
            "route_count": len(baseline.get("routes") or []),
        },
        "route_audit_summary": {
            "route_status": route_audit.get("route_status"),
            "reasons": [str(item) for item in route_audit.get("reasons") or []],
            "condition_status": route_audit.get("condition_status"),
            "evidence_status": route_audit.get("evidence_status"),
            "fake_closure_rejected": bool(route_audit.get("fake_closure_rejected")),
            "stock_audit_passed": bool(route_audit.get("stock_audit_passed")),
        },
        "route_failure_feedback_summary": {
            "present": route_failure_feedback.get("schema_version") == ROUTE_FAILURE_FEEDBACK_SCHEMA,
            "accepted": bool(route_failure_feedback.get("accepted")),
            "source_route_status": route_failure_feedback.get("source_route_status"),
            "source_reasons": [str(item) for item in route_failure_feedback.get("source_reasons") or []],
            "terminal_blacklist_count": len(route_failure_feedback.get("terminal_blacklist") or []),
            "frontier_research_target_count": len(
                route_failure_feedback.get("frontier_research_targets") or []
            ),
            "query_hint_count": len(route_failure_feedback.get("query_hints") or []),
        },
        "literature_trigger_summary": {
            "should_trigger": trigger.get("should_trigger"),
            "decision": trigger.get("decision"),
            "frontier_reasons": (
                dict(trigger.get("audit_summary") or {}).get("frontier_reasons") or []
            ),
        },
        "existing_evidence_cards": [
            {
                "evidence_id": row.get("evidence_id"),
                "claim_type": row.get("claim_type"),
                "route_role": row.get("route_role"),
                "confidence": row.get("confidence"),
                "doi": row.get("doi"),
                "source_title": row.get("source_title"),
                "local_ref": row.get("local_ref"),
            }
            for row in evidence_cards[:20]
            if isinstance(row, dict)
        ],
        "existing_candidate_count": len(candidates),
    }


def _runtime_capabilities(*, run_dir: Path) -> dict[str, Any]:
    rdkit_available, rdkit_version = _rdkit_capability()
    return {
        "rdkit_available": rdkit_available,
        "rdkit_version": rdkit_version,
        "shell_available": True,
        "ripgrep_available": shutil.which("rg") is not None,
        "network_allowed": True,
        "allowed_write_root": str(run_dir.resolve()),
        "capability_probe_policy": (
            "Do not run package/environment discovery commands; use this object."
        ),
    }


def _rdkit_capability() -> tuple[bool, str]:
    try:
        import rdkit  # type: ignore[import-not-found]
    except Exception:
        return False, ""
    return True, str(getattr(rdkit, "__version__", ""))


def _case_manifest(*, context: Path, workflow_dir: Path) -> dict[str, Any]:
    paths = {
        "base_dir": context,
        "target_input": context / "target_input.json",
        "preflight": context / "preflight.json",
        "chemenzy_baseline_routes": context / "chemenzy_baseline_routes.json",
        "route_verifier_report": context / "route_verifier_report.json",
        "route_audit": context / "route_audit.json",
        "route_audit_tool_result": context / "route_audit_tool_result.json",
        "route_failure_feedback": context / "route_failure_feedback.json",
        "frontier_report": workflow_dir / "frontier_report.json",
        "literature_search_report": workflow_dir / "literature_search_report.json",
        "literature_escalation_decision": workflow_dir / "literature_escalation_decision.json",
        "literature_trigger_report": workflow_dir / "literature_trigger_report.json",
        "target_profile": workflow_dir / "target_profile.json",
        "evidence_cards": workflow_dir / "evidence_cards.jsonl",
        "rxn_candidates": workflow_dir / "fluvastatin_literature_rxn_candidates.jsonl",
        "strategic_disconnection_cards": workflow_dir / "fluvastatin_strategic_disconnection_cards.jsonl",
        "hybrid_route": workflow_dir / "fluvastatin_hybrid_retrosynthesis_route.json",
    }
    return {
        key: str(path)
        for key, path in paths.items()
        if key == "base_dir" or Path(path).exists()
    }


def _operation_boundary(*, run_dir: Path) -> dict[str, Any]:
    return {
        "shell_policy": {
            "allowed": [
                "Read manifest-listed files.",
                "Run local deterministic transformation or RDKit validation scripts under allowed_write_root.",
                "Parse/check JSON artifacts under allowed_write_root.",
            ],
            "forbidden": [
                "environment probing: pwd, which, pip show, conda list, uname",
                "file discovery: rg --files, find over the case tree, ls -R for status recovery",
                "recursive content search over raw logs/routes: grep -R, rg over the case tree",
                "external HTTP retrieval: curl, wget, urllib/httpx/requests in helper scripts",
                "process inspection or control: pgrep, pkill, ps for helper recovery, kill",
                "patching long network/retrieval orchestrator scripts after timeout",
            ],
        },
        "retrieval_policy": {
            "current_interim_mode": (
                "Codex may propose lookup requests and use native web_search with non-empty intentful queries. "
                "Native direct DOI/publisher/source URL checks are allowed for agent-access outcomes. "
                "Shell HTTP retrieval is harness-owned and is audited as a boundary violation."
            ),
            "lookup_request_schema": {
                "source": "pubchem|crossref|pubmed|patent_metadata|web_search|doi",
                "query": "non-empty exact target/intermediate/source query",
                "intent": "exact_target|exact_intermediate|close_analog|family_only|method_reference",
                "expected_relation": "exact_target|exact_intermediate|close_analog|family_only",
            },
        },
        "artifact_policy": {
            "required_checkpoint_files": [
                "structure_template_report.md",
                "structure_template_candidates.json",
                "downstream_consumables.json",
                "evidence/literature_sources.json",
                "evidence/pubchem_validated_compounds.json",
                "validated_compounds.smi",
                "open_agent_audit.json",
            ],
            "downstream_consumables": {
                "purpose": (
                    "Codex literature research should hand off draft assets that downstream validators can consume, "
                    "not only narrative reports."
                ),
                "allowed_draft_assets": [
                    "guided_rerun_requests for Chemenzy policy compilation",
                    "LiteratureTemplateCard drafts",
                    "LiteratureRouteSegmentCard drafts for multi-step literature routes",
                    "ExecutableTemplateCandidate drafts only when all structures are RDKit-valid and source-grounded",
                    "RouteAnchorExpansionTask drafts for recursive subgoal planning",
                    "EvolutionCandidate drafts targeting candidate/shadow/staging layers, never production",
                ],
            },
            "write_root": str(run_dir.resolve()),
            "final_claim_constraints": [
                "solved must remain false unless deterministic validators override.",
                "production_kb_promotion must remain false.",
                "unrelated-family sources cannot support route evidence_refs.",
            ],
        },
    }


def _research_policy(experience: dict[str, Any]) -> dict[str, Any]:
    observed = set(str(item) for item in experience.get("observed_inefficiencies") or [])
    skip = [
        "Do not fetch Google Patents or patent HTML pages; record metadata/search URLs only.",
        "Do not run unbounded recursive grep over the full result tree when manifest summaries are present.",
        "Do not leave helper scripts running in the background; write checkpoints before optional network calls.",
        "Do not use PubMed broad synthesis queries as route evidence unless the result title has route/manufacturing terms.",
    ]
    if "rg_unavailable" in observed:
        skip.append("Prefer Python/find over rg in this container unless availability has been rechecked.")
    return {
        "mode": "bounded_manifest_first",
        "source_budgets": {
            "pubchem_name_queries_max": 8,
            "crossref_queries_max": 6,
            "pubmed_queries_max": 2,
            "patent_metadata_queries_max": 3,
            "live_web_search_batches_max": 3,
            "helper_script_runs_max": 1,
        },
        "network_timeouts_s": {
            "default_api_request": 6,
            "metadata_request": 8,
            "optional_source": 3,
        },
        "checkpoint_rule": (
            "Write the six required artifacts as soon as local audit, target PubChem identity, "
            "and at least one source-search pass are available; append optional evidence later."
        ),
        "stop_rules": [
            "Stop searching when exact-target and exact-intermediate queries produce only metadata/noise and the candidate set is already labeled partial.",
            "Stop before optional full-text/patent HTML fetches; emit skipped_with_reason search_log rows instead.",
            "Do not rerun the helper script after a timeout unless the patch removes the blocking source class.",
        ],
        "skip_or_defer": skip,
    }


def _query_plan(
    *,
    target: str,
    family_hint: str,
    experience: dict[str, Any],
    self_evo_memory: dict[str, Any] | None = None,
    route_failure_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = target.strip()
    lower_hint = family_hint.lower()
    pubchem_names = [normalized]
    if "statin" in lower_hint or normalized.lower().endswith("statin"):
        pubchem_names.extend([
            f"{normalized} sodium",
            f"{normalized} lactone",
            f"methyl {normalized}",
        ])
    pubchem_names = _dedupe([item for item in pubchem_names if item])
    feedback = dict(route_failure_feedback or {})
    route_failure_queries = _dedupe([
        str(row.get("query") or "")
        for row in feedback.get("query_hints") or []
        if isinstance(row, dict) and row.get("query")
    ])
    route_failure_frontiers = _dedupe([
        str(row.get("canonical_smiles") or row.get("smiles") or "")
        for row in feedback.get("frontier_research_targets") or []
        if isinstance(row, dict) and (row.get("canonical_smiles") or row.get("smiles"))
    ])
    extraction_tasks = _self_evo_extraction_tasks(dict(self_evo_memory or {}))
    lookup_requests = _lookup_requests_from_extraction_tasks(
        target=normalized,
        extraction_tasks=extraction_tasks,
    )
    lookup_budget = {
        "crossref": 3,
        "patent_metadata": 3,
        "web_search_metadata": 3,
        "max_tasks": 3,
        "fragments_per_task": 1,
    }
    lookup_requests = _apply_lookup_request_budget(lookup_requests, lookup_budget)
    return {
        "source_order": [
            "local_manifest_summary",
            *(
                ["Route failure feedback frontier targets before fresh broad search"]
                if route_failure_queries or route_failure_frontiers
                else []
            ),
            *(
                ["SelfEVO executable-template extraction tasks before fresh broad search"]
                if lookup_requests
                else []
            ),
            "PubChem exact target/salt/close analog names",
            *(
                ["Typed lookup requests from reusable extraction tasks"]
                if lookup_requests
                else []
            ),
            "CrossRef exact-title synthesis/manufacturing metadata",
            "PubMed route-title gap check only",
            "Patent metadata URL recording only",
            "Live web search only for unresolved exact-intermediate names",
        ],
        "budget_mode": "self_evo_targeted" if lookup_requests else "standard",
        "prioritize_self_evo_lookup_requests": bool(lookup_requests),
        "self_evo_lookup_request_budget": lookup_budget,
        "pubchem_name_queries": pubchem_names[:8],
        "crossref_queries": _dedupe([
            f"{normalized} synthesis",
            f"Synthesis of {normalized}",
            f"{normalized} manufacturing process",
            f"{normalized} intermediate",
            f"{normalized} process chemistry",
        ])[:6],
        "pubmed_terms": _dedupe([
            f'"{normalized}" synthesis',
            f'"{normalized}" intermediate',
        ])[:2],
        "patent_metadata_queries": _dedupe([
            f'"{normalized}" process',
            f'"{normalized}" intermediate',
            f'"{normalized}" lactone',
        ])[:3],
        "live_web_search_gap_queries": _dedupe([
            f"{normalized} synthesis intermediate DOI",
            f"{normalized} process chemistry intermediate patent",
        ])[:3],
        "route_failure_feedback_queries": route_failure_queries[:6],
        "route_failure_frontier_smiles": route_failure_frontiers[:8],
        "route_failure_terminal_blacklist": [
            str(item)
            for item in dict(feedback.get("next_guided_policy_patch") or {}).get("terminal_blacklist") or []
            if str(item)
        ][:20],
        "self_evo_extraction_task_count": len(extraction_tasks),
        "self_evo_extraction_tasks": [
            _condensed_extraction_task(task)
            for task in extraction_tasks[:8]
        ],
        "lookup_requests": lookup_requests,
        "experience_adjustments": [str(item) for item in experience.get("suggested_policy_updates") or []],
    }


def _self_evo_extraction_tasks(memory: dict[str, Any]) -> list[dict[str, Any]]:
    if memory.get("schema_version") != "self_evo_reusable_memory.v1":
        return []
    tasks = [
        dict(item)
        for item in memory.get("reusable_executable_template_extraction_tasks") or []
        if isinstance(item, dict)
    ]
    return [
        task
        for task in tasks
        if task.get("task_id") and task.get("evidence_refs") and not _contains_raw_reaction(task)
    ][:12]


def _lookup_requests_from_extraction_tasks(
    *,
    target: str,
    extraction_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for task in extraction_tasks[:6]:
        task_id = str(task.get("task_id") or "")
        evidence_refs = [str(item) for item in task.get("evidence_refs") or []]
        required_fields = [str(item) for item in task.get("required_structured_fields") or []]
        source_title = str(task.get("source_title") or "").strip()
        reaction_class = str(task.get("reaction_class") or "").strip()
        fragments = _task_query_fragments(task)
        if source_title and target.lower() in source_title.lower():
            requests.append(_lookup_request(
                source="crossref",
                query=source_title,
                target=target,
                task_id=task_id,
                evidence_refs=evidence_refs,
                required_fields=required_fields,
                intent="exact_intermediate",
                relation="exact_target_or_exact_intermediate",
                reason="self_evo_source_title_recheck",
            ))
        for fragment in fragments[:1]:
            requests.append(_lookup_request(
                source="crossref",
                query=f"{target} {fragment} synthesis",
                target=target,
                task_id=task_id,
                evidence_refs=evidence_refs,
                required_fields=required_fields,
                intent="exact_intermediate",
                relation="exact_target_or_exact_intermediate",
                reason="self_evo_precursor_role_to_crossref",
            ))
            requests.append(_lookup_request(
                source="patent_metadata",
                query=f'"{target}" "{fragment}" intermediate',
                target=target,
                task_id=task_id,
                evidence_refs=evidence_refs,
                required_fields=required_fields,
                intent="exact_intermediate",
                relation="exact_target_or_exact_intermediate",
                reason="self_evo_precursor_role_to_patent_metadata",
            ))
            requests.append(_lookup_request(
                source="web_search_metadata",
                query=f"{target} {fragment} product reactant SMILES",
                target=target,
                task_id=task_id,
                evidence_refs=evidence_refs,
                required_fields=required_fields,
                intent="exact_intermediate",
                relation="exact_target_or_exact_intermediate",
                reason="self_evo_missing_structured_smiles",
            ))
        if reaction_class and not fragments:
            requests.append(_lookup_request(
                source="crossref",
                query=f"{target} {reaction_class} process chemistry",
                target=target,
                task_id=task_id,
                evidence_refs=evidence_refs,
                required_fields=required_fields,
                intent="exact_intermediate",
                relation="exact_target_or_exact_intermediate",
                reason="self_evo_reaction_class_fallback",
            ))
    return _dedupe_lookup_requests(requests)


def _lookup_request(
    *,
    source: str,
    query: str,
    target: str,
    task_id: str,
    evidence_refs: list[str],
    required_fields: list[str],
    intent: str,
    relation: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "typed_lookup_request.v1",
        "request_id": _lookup_request_id(task_id=task_id, source=source, query=query),
        "source": source,
        "query": _compact(query, 180),
        "intent": intent,
        "expected_relation": relation,
        "origin": "self_evo_executable_template_extraction_task",
        "target": target,
        "task_id": task_id,
        "extraction_task_ids": [task_id] if task_id else [],
        "evidence_refs": evidence_refs,
        "required_structured_fields": required_fields,
        "reason": reason,
    }


def _task_query_fragments(task: dict[str, Any]) -> list[str]:
    values = [str(item) for item in task.get("precursor_roles") or [] if str(item)]
    if not values and task.get("source_title"):
        values = [str(task.get("source_title") or "")]
    fragments: list[str] = []
    for value in values:
        fragment = _role_query_fragment(value)
        if fragment:
            fragments.append(fragment)
    return _dedupe(fragments)[:4]


def _role_query_fragment(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9 -]+", " ", str(value))
    stop = {
        "or",
        "and",
        "the",
        "a",
        "an",
        "equivalent",
        "partner",
        "core",
        "protected",
        "chiral",
        "related",
        "validated",
    }
    words = [
        word
        for word in text.split()
        if len(word) > 2 and word.lower() not in stop
    ]
    return " ".join(words[:6])


def _apply_lookup_request_budget(
    requests: list[dict[str, Any]],
    budget: dict[str, int],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for request in requests:
        source = str(request.get("source") or "")
        limit = int(budget.get(source) or 0)
        if limit <= 0:
            continue
        if counts[source] >= limit:
            continue
        out.append(request)
        counts[source] += 1
    return out


def _lookup_request_id(*, task_id: str, source: str, query: str) -> str:
    stem = _safe_id(f"{task_id}_{source}")[:48]
    digest = hashlib.sha256(f"{task_id}\n{source}\n{query}".encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}".strip("_")


def _condensed_extraction_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "source_title": str(task.get("source_title") or ""),
        "reaction_class": str(task.get("reaction_class") or ""),
        "evidence_refs": [str(item) for item in task.get("evidence_refs") or []],
        "precursor_roles": [str(item) for item in task.get("precursor_roles") or []][:4],
        "required_structured_fields": [str(item) for item in task.get("required_structured_fields") or []],
    }


def _dedupe_lookup_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for request in requests:
        source = str(request.get("source") or "")
        query = str(request.get("query") or "").strip()
        if len(query) < 3:
            continue
        key = (source, query.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(request)
    return out


def _contains_raw_reaction(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"rxn", "rxn_smiles", "reaction_smiles", "raw_reaction", "raw_reactions"}:
                return True
            if _contains_raw_reaction(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_reaction(item) for item in value)
    if isinstance(value, str):
        return ">>" in value
    return False


def _load_prior_experience(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _load_json(Path(path), {})
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") == OPEN_RESEARCH_EXPERIENCE_SCHEMA:
        return payload
    if isinstance(payload.get("latest"), dict):
        return dict(payload["latest"])
    return payload


def _load_self_evo_memory(path: str | Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if path:
        p = Path(path)
        candidates.append(p)
        candidates.append(p.parent / "self_evo_memory.json")
    for candidate in candidates:
        payload = _load_json(candidate, {})
        if isinstance(payload, dict) and payload.get("schema_version") == "self_evo_reusable_memory.v1":
            return payload
        nested = payload.get("self_evo_memory") if isinstance(payload, dict) else None
        if isinstance(nested, dict) and nested.get("schema_version") == "self_evo_reusable_memory.v1":
            return nested
    return {}


def _load_route_failure_feedback(
    path: str | Path | None,
    *,
    context_root: str | Path | None = None,
) -> dict[str, Any]:
    candidates: list[Path] = []
    if context_root:
        candidates.append(Path(context_root) / "route_failure_feedback.json")
    if path:
        p = Path(path)
        candidates.append(p)
        if p.is_dir():
            candidates.append(p / "route_failure_feedback.json")
        else:
            candidates.append(p.parent / "route_failure_feedback.json")
    for candidate in candidates:
        payload = _load_json(candidate, {})
        if isinstance(payload, dict) and payload.get("schema_version") == ROUTE_FAILURE_FEEDBACK_SCHEMA:
            return payload
        nested = payload.get("route_failure_feedback") if isinstance(payload, dict) else None
        if isinstance(nested, dict) and nested.get("schema_version") == ROUTE_FAILURE_FEEDBACK_SCHEMA:
            return nested
    return {}


def _classify_command_inefficiencies(
    command: str,
    output: str,
    exit_code: Any,
    inefficiencies: set[str],
) -> None:
    text = f"{command}\n{output}".lower()
    if "rg --files" in text and "rg: command not found" in text:
        inefficiencies.add("rg_unavailable")
    command_text = command.lower()
    if "pgrep" in command_text or "pkill" in command_text or re.search(r"\bkill\s+\d+", command_text):
        inefficiencies.add("helper_process_management_overhead")
    if "python research_fluvastatin.py" in text and exit_code not in (0, None):
        inefficiencies.add("helper_script_run_failed_or_interrupted")
    if "patents.google.com" in text:
        inefficiencies.add("patent_html_or_search_page_is_optional")
    if "eutils.ncbi.nlm.nih.gov" in text and "metabolism" in text:
        inefficiencies.add("broad_pubmed_synthesis_query_noisy")
    if _looks_like_large_artifact_dump(command, output):
        inefficiencies.add("large_raw_artifact_overread")


def _boundary_command_violations(command: str, *, output: str = "") -> list[dict[str, str]]:
    text = _unwrap_shell_command(command).strip()
    lowered = text.lower()
    checks = [
        ("environment_probe", "pwd_command", r"(^|[;&|]\s*)pwd(\s|$)"),
        ("environment_probe", "which_command", r"(^|[;&|]\s*)which\s+"),
        ("environment_probe", "pip_show", r"python\s+-m\s+pip\s+show|\bpip\s+show\b"),
        ("environment_probe", "conda_list", r"\bconda\s+list\b"),
        ("environment_probe", "uname_command", r"(^|[;&|]\s*)uname(\s|$)"),
        ("file_discovery", "ripgrep_file_discovery", r"\brg\s+--files\b"),
        ("file_discovery", "find_case_tree", r"(^|[;&|]\s*)find\s+"),
        ("file_discovery", "recursive_ls_recovery", r"\bls\s+-r\b|\bls\s+-R\b"),
        ("artifact_search", "recursive_grep", r"\bgrep\s+-R|\brg\s+.*\s/|\brg\s+.*\s\."),
        ("external_http", "curl_http_retrieval", r"\bcurl\s+"),
        ("external_http", "wget_http_retrieval", r"\bwget\s+"),
        ("external_http", "python_http_client", r"\burllib\.request\b|\brequests\.|\bhttpx\."),
        ("process_management", "process_probe", r"\bpgrep\b|\bps\s+-"),
        ("process_management", "process_kill", r"\bpkill\b|(^|[;&|]\s*)kill\s+\d+"),
    ]
    violations: list[dict[str, str]] = []
    for category, reason, pattern in checks:
        if re.search(pattern, lowered):
            violations.append({
                "category": category,
                "reason": reason,
                "detail": _compact(text, 240),
            })
    if _looks_like_rdkit_capability_probe(lowered):
        violations.append({
            "category": "environment_probe",
            "reason": "python_rdkit_capability_probe",
            "detail": _compact(text, 240),
        })
    if _looks_like_large_artifact_dump(text, output):
        violations.append({
            "category": "context_boundary",
            "reason": "large_raw_artifact_dump",
            "detail": _compact(text, 240),
        })
    return violations


def _looks_like_large_artifact_dump(command: str, output: str) -> bool:
    text = command.lower()
    if len(str(output or "")) < 8000:
        return False
    if "sed -n" not in text and "cat " not in text:
        return False
    bulky_artifacts = (
        "chemenzy_native_raw_result.json",
        "case_bundle.json",
        "harness_retrieval_prefetch.json",
        "hybrid_retrosynthesis_route.json",
        "literature_rxn_candidates.jsonl",
        "strategic_disconnection_cards.jsonl",
        "codex_events.jsonl",
        "tool_calls.jsonl",
        "decision_trace.jsonl",
    )
    return any(name in text for name in bulky_artifacts)


def _is_direct_url_query(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_like_rdkit_capability_probe(text: str) -> bool:
    if "import rdkit" not in text:
        return False
    if "molfromsmiles" in text or "moltosmiles" in text or "molinchikey" in text:
        return False
    capability_markers = (
        "rdkit.__version__",
        "__version__",
        "rdkit available",
        "rdkit unavailable",
        "runtime capability",
    )
    return any(marker in text for marker in capability_markers)


def _unwrap_shell_command(command: str) -> str:
    value = str(command or "")
    match = re.match(r"^/bin/bash\s+-lc\s+['\"](?P<body>.*)['\"]$", value, flags=re.DOTALL)
    if match:
        return match.group("body")
    return value


def _policy_updates_from_inefficiencies(inefficiencies: set[str]) -> list[dict[str, str]]:
    updates: list[dict[str, str]] = []
    if "rg_unavailable" in inefficiencies:
        updates.append({
            "policy_id": "prefer_find_or_python_for_local_discovery",
            "reason": "rg was unavailable in the open-agent container.",
        })
    if "helper_process_management_overhead" in inefficiencies or "helper_script_run_failed_or_interrupted" in inefficiencies:
        updates.append({
            "policy_id": "single_helper_run_with_checkpoint_first",
            "reason": "Repeated helper runs and process cleanup consumed time without producing required artifacts.",
        })
    if "broad_pubmed_synthesis_query_noisy" in inefficiencies:
        updates.append({
            "policy_id": "pubmed_only_after_exact_route_terms",
            "reason": "Broad PubMed synthesis queries can expand into metabolism/biomedical noise.",
        })
    if "patent_html_or_search_page_is_optional" in inefficiencies:
        updates.append({
            "policy_id": "record_patent_metadata_url_without_html_fetch",
            "reason": "Patent HTML/search pages are slow optional sources and should not block artifacts.",
        })
    if "direct_url_web_search_without_connector" in inefficiencies:
        updates.append({
            "policy_id": "route_url_lookups_through_typed_connectors",
            "reason": "Direct URL web_search calls bypass source-specific retrieval policy and provenance handling.",
        })
    if "large_raw_artifact_overread" in inefficiencies:
        updates.append({
            "policy_id": "use_structured_artifact_reader_not_raw_sed",
            "reason": "Large raw artifact dumps consume time and context before required downstream files are enriched.",
        })
    if "minimum_artifacts_not_checkpointed_before_optional_work" in inefficiencies:
        updates.append({
            "policy_id": "minimum_artifacts_before_optional_sources",
            "reason": "Required artifacts were missing when the open run ended.",
        })
    if "open_agent_timeout_before_required_artifacts" in inefficiencies:
        updates.append({
            "policy_id": "source_budget_deadline_enforced_by_manifest",
            "reason": "Open-agent timeout occurred before turn completion.",
        })
    if "open_agent_timeout_after_required_artifacts" in inefficiencies:
        updates.append({
            "policy_id": "accept_schema_valid_checkpoint_before_turn_timeout",
            "reason": "Required artifacts were written and schema-valid, but the Codex turn timed out during post-write validation.",
        })
    return updates


def _reusable_search_hints(*, web_queries: list[str], domains: Counter[str]) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    if any("doi" in query.lower() for query in web_queries):
        hints.append({
            "hint_id": "doi_exact_title_search_is_high_value",
            "hint": "Exact-title DOI searches are better live-web candidates than broad synthesis searches.",
        })
    if domains.get("pubchem.ncbi.nlm.nih.gov"):
        hints.append({
            "hint_id": "pubchem_structure_identity_first",
            "hint": "Use PubChem for identity/synonym/SMILES validation before assigning route roles.",
        })
    if domains.get("api.crossref.org"):
        hints.append({
            "hint_id": "crossref_metadata_for_source_triage",
            "hint": "Use CrossRef metadata to classify exact-target versus unrelated sources before deeper fetching.",
        })
    return hints


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            rows.append(event)
    return rows


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iter_jsonl(path):
        rows.append(item)
    return rows


def _domains_from_text(text: str) -> list[str]:
    domains: list[str] = []
    for url in re.findall(r"https?://[^\s'\"<>]+", text):
        parsed = urlparse(url)
        if parsed.netloc:
            domains.append(parsed.netloc)
    return domains


def _compact(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "request"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
