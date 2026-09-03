"""Read-only catalog joining generated showcases with the canonical Web surface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote


def compile_showcase_catalog(
    *,
    root: Path,
    shared_root: Path,
    manifest_path: Path | None = None,
    artifact_endpoint: str = "/api/v4/result-file",
) -> dict[str, Any]:
    root = root.resolve()
    shared_root = shared_root.resolve()
    legacy = _read_json(manifest_path) if manifest_path else {}
    fresh_cases, reruns, generated_at = _fresh_cases(
        root=root,
        shared_root=shared_root,
        artifact_endpoint=artifact_endpoint,
    )
    fresh_audits = _fresh_benchmark_audits(
        root=root,
        shared_root=shared_root,
        artifact_endpoint=artifact_endpoint,
    )
    legacy_cases = [
        _published_row(
            row,
            root=root,
            shared_root=shared_root,
            artifact_endpoint=artifact_endpoint,
        )
        for row in legacy.get("cases") or []
        if isinstance(row, Mapping)
    ]
    cases = _deduplicate_cases([*fresh_cases, *legacy_cases])
    legacy_audits = [
        _published_row(
            row,
            root=root,
            shared_root=shared_root,
            artifact_endpoint=artifact_endpoint,
        )
        for row in legacy.get("audits") or []
        if isinstance(row, Mapping)
    ]
    audits = [*fresh_audits, *legacy_audits]
    standard_case = next(
        (
            row
            for row in cases
            if row.get("available") and "bufotalin" in str(row.get("target_name") or "").casefold()
        ),
        next((row for row in cases if row.get("available")), {}),
    )
    standard = _closure_standard(standard_case) or copy.deepcopy(legacy.get("standard") or {})
    return {
        "schema_version": "autoplanner.presentation_showcase.v2",
        "ok": any(row.get("available") for row in cases),
        "generated_at": generated_at or str(legacy.get("generated_at") or ""),
        "standard_case_id": str(
            standard_case.get("case_id") or legacy.get("standard_case_id") or ""
        ),
        "standard": standard,
        "cases": cases,
        "statin_catalog": _statin_catalog(root, legacy),
        "audits": audits,
        "reruns": [*reruns, *copy.deepcopy(legacy.get("reruns") or [])],
        "excluded": copy.deepcopy(legacy.get("excluded") or []),
        "message": (
            "已汇总 fresh blind showcase、当前工作台与后端运行索引；路线长度只作描述。"
            if cases
            else "尚未发现可展示工作台；可从运行控制台启动新的 target-only campaign。"
        ),
        "semantics": {
            "route_length_is_descriptive_not_an_objective": True,
            "graph_reaction_evidence_and_procurement_closure_are_independent": True,
            "unavailable_artifacts_are_never_published": True,
        },
    }


def _fresh_cases(
    *, root: Path, shared_root: Path, artifact_endpoint: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    cases: list[dict[str, Any]] = []
    reruns: list[dict[str, Any]] = []
    generated_at = ""
    summaries = sorted(
        shared_root.glob("*/showcase/summary.json"),
        key=lambda path: path.stat().st_mtime if path.is_file() else 0,
        reverse=True,
    )[:64]
    for summary_path in summaries:
        summary = _read_json(summary_path)
        if summary.get("schema_version") != "v4_blind_expert_showcase.v1":
            continue
        generated_at = max(generated_at, str(summary.get("generated_at") or ""))
        completed = 0
        targets = summary.get("targets") or []
        for target in targets:
            if not isinstance(target, Mapping):
                continue
            case = _fresh_case(target, summary_path, root, shared_root, artifact_endpoint)
            cases.append(case)
            completed += int(str(target.get("status") or "") == "completed")
        reruns.append(
            {
                "panel_id": summary_path.parents[1].name,
                "label": summary_path.parents[1].name,
                "status": "completed" if completed == len(targets) and targets else "partial",
                "complete_target_count": completed,
                "target_count": len(targets),
            }
        )
    return cases, reruns, generated_at


def _fresh_benchmark_audits(
    *, root: Path, shared_root: Path, artifact_endpoint: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    release_summaries = sorted(
        shared_root.glob("*/blind-release-gate/summary.json"),
        key=lambda path: path.stat().st_mtime if path.is_file() else 0,
        reverse=True,
    )[:16]
    for summary_path in release_summaries:
        summary = _read_json(summary_path)
        if summary.get("schema_version") != "v4_blind_release_gate.v1":
            continue
        gates = dict(summary.get("gates") or {})
        coverage = dict(summary.get("coverage") or {})
        cost = dict(summary.get("cost") or {})
        rows.append(
            _published_row(
                {
                    "audit_id": "blind-release:" + summary_path.parents[1].name,
                    "label": "V4 fresh blind 发布门",
                    "category": "v4 fresh blind release gate",
                    "artifact_path": str(summary_path.parent / "index.html"),
                    "accepted": summary.get("accepted") is True,
                    "target_count": int(summary.get("target_count") or 0),
                    "gate_count": len(gates),
                    "passed_gate_count": sum(
                        dict(value).get("passed") is True for value in gates.values()
                    ),
                    "validated_case_count": int(
                        coverage.get("two_distinct_validated_case_count") or 0
                    ),
                    "condition_complete_case_count": int(
                        coverage.get("condition_complete_case_count") or 0
                    ),
                    "procurement_closed_case_count": int(
                        coverage.get("procurement_closed_case_count") or 0
                    ),
                    "median_model_invocations": float(cost.get("median_model_invocations") or 0.0),
                    "note": (
                        "发布门通过；打开查看逐目标、消融、成本与失败分布。"
                        if summary.get("accepted") is True
                        else "发布门尚未通过；失败项与低可信/未闭合路线均保留。"
                    ),
                },
                root=root,
                shared_root=shared_root,
                artifact_endpoint=artifact_endpoint,
            )
        )
    summaries = sorted(
        shared_root.glob("*/benchmark/summary.json"),
        key=lambda path: path.stat().st_mtime if path.is_file() else 0,
        reverse=True,
    )[:16]
    for summary_path in summaries:
        summary = _read_json(summary_path)
        if summary.get("schema_version") != "classic_multistep_benchmark_summary.v1":
            continue
        overall = dict(dict(summary.get("aggregates") or {}).get("overall") or {})
        rows.append(
            _published_row(
                {
                    "audit_id": summary_path.parents[1].name,
                    "label": "PaRoutes 经典多步盲测 · 20 targets",
                    "category": "classic multistep benchmark",
                    "artifact_path": str(summary_path.parent / "index.html"),
                    "target_count": int(overall.get("target_count") or 0),
                    "route_found_rate": float(overall.get("chem_enzy_route_found_rate") or 0.0),
                    "route_retained_rate": float(overall.get("cascade_route_retained_rate") or 0.0),
                    "accepted_rate": float(overall.get("cascade_accepted_rate") or 0.0),
                    "stock_closed_rate": float(overall.get("benchmark_stock_closed_rate") or 0.0),
                },
                root=root,
                shared_root=shared_root,
                artifact_endpoint=artifact_endpoint,
            )
        )
    patent_summaries = sorted(
        shared_root.glob("*/patent-procedure-gate/summary.json"),
        key=lambda path: path.stat().st_mtime if path.is_file() else 0,
        reverse=True,
    )[:16]
    for summary_path in patent_summaries:
        summary = _read_json(summary_path)
        if summary.get("schema_version") != "real_patent_procedure_gate_suite.v1":
            continue
        rows.append(
            _published_row(
                {
                    "audit_id": str(summary.get("suite_id") or summary_path.parents[1].name),
                    "label": "官方结构化专利 procedure · 3-case gate",
                    "category": "official patent procedure gate",
                    "artifact_path": str(summary_path.parent / "index.html"),
                    "accepted": summary.get("accepted") is True,
                    "case_count": int(summary.get("case_count") or 0),
                    "accepted_count": int(summary.get("accepted_count") or 0),
                    "unique_publication_count": int(summary.get("unique_publication_count") or 0),
                    "unique_reaction_class_count": int(
                        summary.get("unique_reaction_class_count") or 0
                    ),
                    "structured_source_closed_count": int(
                        dict(summary.get("acquisition_cascade") or {}).get(
                            "structured_source_closed_count"
                        )
                        or 0
                    ),
                    "pdf_fallback_count": int(
                        dict(summary.get("acquisition_cascade") or {}).get("pdf_fallback_count")
                        or 0
                    ),
                    "vision_fallback_count": int(
                        dict(summary.get("acquisition_cascade") or {}).get("vision_fallback_count")
                        or 0
                    ),
                },
                root=root,
                shared_root=shared_root,
                artifact_endpoint=artifact_endpoint,
            )
        )
    return rows


def _fresh_case(
    target: Mapping[str, Any],
    summary_path: Path,
    root: Path,
    shared_root: Path,
    artifact_endpoint: str,
) -> dict[str, Any]:
    workbench = dict(target.get("workbench") or {})
    evidence = dict(target.get("evidence") or {})
    maturity = dict(target.get("maturity") or {})
    closed = int(workbench.get("graph_closed_program_count") or 0)
    declared = int(workbench.get("declared_program_count") or 0)
    row = {
        "case_id": str(
            target.get("run_id") or target.get("target_name") or summary_path.parents[1].name
        ),
        "run_id": str(target.get("run_id") or ""),
        "target_name": str(target.get("target_name") or "target"),
        "display_name": str(target.get("target_name") or "target"),
        "category": "fresh blind run",
        "artifact_path": str(summary_path.parent / str(target.get("workbench_file") or "")),
        "route_closed": (closed > 0) if declared > 0 else None,
        "all_declared_routes_closed": declared > 0 and closed == declared,
        "graph_closed_program_count": closed,
        "declared_program_count": declared,
        "graph_open_program_count": int(workbench.get("graph_open_program_count") or 0),
        "max_step_count": int(
            workbench.get("longest_graph_closed_step_count")
            or workbench.get("max_route_steps")
            or workbench.get("max_selected_route_steps")
            or 0
        ),
        "process_ready": int(workbench.get("process_ready_route_count") or 0) > 0,
        "claim_status": str(target.get("claim") or "unresolved"),
        "presentation_note": str(maturity.get("label") or ""),
        "source_reference_count": int(evidence.get("sources") or 0),
        "source_group_count": int(evidence.get("bindings") or 0),
        "proof_distribution": copy.deepcopy(target.get("gates") or {}),
    }
    return _published_row(
        row,
        root=root,
        shared_root=shared_root,
        artifact_endpoint=artifact_endpoint,
    )


def _published_row(
    raw: Mapping[str, Any], *, root: Path, shared_root: Path, artifact_endpoint: str
) -> dict[str, Any]:
    row = copy.deepcopy(dict(raw))
    raw_path = str(row.get("artifact_path") or "").strip()
    candidate = (
        Path(raw_path).resolve() if Path(raw_path).is_absolute() else (root / raw_path).resolve()
    )
    available = (
        bool(raw_path)
        and candidate.is_relative_to(shared_root)
        and candidate.is_file()
        and candidate.suffix.casefold() in {".html", ".htm"}
    )
    relative = candidate.relative_to(root).as_posix() if available else ""
    row["available"] = available
    row["artifact_path"] = relative
    row["artifact_url"] = (
        f"{artifact_endpoint}?path={quote(relative, safe='/')}" if available else ""
    )
    return row


def _deduplicate_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        target_name = str(row.get("target_name") or "").strip().casefold()
        key = (
            f"target:{target_name}"
            if target_name
            else str(row.get("run_id") or row.get("case_id") or row.get("artifact_path") or "")
        )
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _closure_standard(case: Mapping[str, Any]) -> dict[str, str]:
    if not case:
        return {}
    closed = int(case.get("graph_closed_program_count") or int(case.get("route_closed") is True))
    declared = int(case.get("declared_program_count") or closed)
    steps = int(case.get("max_step_count") or 0)
    return {
        "summary": f"闭合优先：{closed}/{declared} 条声明路线结构闭合，最长 {steps} 步；长度不作为优化目标。"
    }


def _statin_catalog(root: Path, legacy: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(legacy.get("statin_catalog"), Mapping) and legacy["statin_catalog"]:
        return copy.deepcopy(dict(legacy["statin_catalog"]))
    catalog = _read_json(root / "benchmarks" / "statin_target_catalog.v1.json")
    readiness = _read_json(root / "benchmarks" / "statin_target_route_readiness.v1.json")
    by_name = {
        str(row.get("target_name") or ""): row
        for row in readiness.get("targets") or []
        if isinstance(row, Mapping)
    }
    targets = []
    for source in catalog.get("targets") or []:
        if not isinstance(source, Mapping):
            continue
        row = copy.deepcopy(dict(source))
        row["route_readiness"] = copy.deepcopy(by_name.get(str(row.get("target_name") or ""), {}))
        targets.append(row)
    return {
        "entity_count": len(targets),
        "normalization_note": str(catalog.get("normalization_note") or ""),
        "readiness_summary": copy.deepcopy(readiness.get("summary") or {}),
        "targets": targets,
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["compile_showcase_catalog"]
