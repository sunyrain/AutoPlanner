#!/usr/bin/env python3
"""Build a standalone expert-facing HTML from one V4 blind panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.v4_route_workbench import (  # noqa: E402
    render_v4_route_workbench_html,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway  # noqa: E402
from cascade_planner.runtime.paths import RuntimePaths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-status", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--console-url", default="http://127.0.0.1:8878/v4")
    parser.add_argument(
        "--target-override-panel",
        action="append",
        default=[],
        help="newer independent blind panel whose matching target replaces the baseline card",
    )
    parser.add_argument(
        "--supplemental-showcase",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--target-validation-report",
        action="append",
        default=[],
        help=(
            "newer evidence/validation-fork report whose proof/workbench "
            "projection is merged into the matching blind target card"
        ),
    )
    args = parser.parse_args(argv)

    status_path = Path(args.panel_status).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = json.loads(status_path.read_text(encoding="utf-8"))
    rows = _compile_rows(panel, output_dir=output_dir, artifact_role="baseline")
    override_paths = [
        Path(value).expanduser().resolve() for value in args.target_override_panel
    ]
    for override_path in override_paths:
        override_panel = json.loads(override_path.read_text(encoding="utf-8"))
        rows = _merge_target_rows(
            rows,
            _compile_rows(
                override_panel,
                output_dir=output_dir,
                artifact_role="latest_independent_blind_rerun",
            ),
        )
    validation_paths = [
        Path(value).expanduser().resolve()
        for value in args.target_validation_report
    ]
    for validation_path in validation_paths:
        validation_report = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        target_name = str(
            dict(validation_report.get("target") or {}).get("name") or ""
        )
        if not target_name:
            raise ValueError("validation_fork_target_name_missing")
        validation_rows = _compile_rows(
            {
                "output_root": str(validation_path.parents[2]),
                "targets": {
                    target_name: {
                        "status": "completed",
                        "run_id": str(validation_report.get("run_id") or ""),
                        "report_path": str(validation_path),
                    }
                },
            },
            output_dir=output_dir,
            artifact_role="latest_evidence_validation_fork",
        )
        rows = _merge_validation_rows(rows, validation_rows)
    summary = _summary(panel, rows)
    payload = {
        "schema_version": "v4_blind_expert_showcase.v1",
        "generated_at": _utc_now(),
        "panel_status_path": str(status_path),
        "target_override_panel_paths": [str(path) for path in override_paths],
        "target_validation_report_paths": [
            str(path) for path in validation_paths
        ],
        "summary": summary,
        "targets": rows,
        "demo_links": {
            "console_url": str(args.console_url),
            "supplemental": [
                _supplemental_link(value, output_dir=output_dir)
                for value in args.supplemental_showcase
            ],
        },
        "runtime_revision": {
            "benchmark_semantics": (
                "目标卡片来自不可变的 blind-run artifact；路线闭合、条件完整和采购闭合分别计数。"
            ),
            "implemented": [
                "Codex 首轮总揽路线族和战略断键，证据到达后最多再做 1 次全局重规划。",
                "ChemEnzy 只处理 Codex/host 接纳的非平凡前沿，并保留库存缺口恢复额度。",
                "Europe PMC/专利采用 HTML/XML 优先、PDF 回退；视觉只在确有图像页时稀疏触发。",
                "文献路线先进入 canonical hypergraph，再经原子映射和反应验证晋升。",
                "PMC 缓存重放保留 PMCID、官方 URL 和内容哈希，避免 exact 条件降级。",
                "工作台区分路线骨架、反应验证、benchmark 边界、采购闭合和工艺就绪。",
            ],
            "next_acceptance": (
                "在同一版本上完成 9 个 statin 的独立 blind panel；逐目标审计路线深度、"
                "来源反应、exact 条件、ChemEnzy 调用、替换路线和边界类型。"
            ),
            "measured_after_revision": True,
            "validation_measurement": {
                "target_count": summary["target_count"],
                "report_count": summary["report_count"],
                "codex_calls": summary["codex_calls"],
                "chemenzy_calls": summary["chemenzy_calls"],
                "visual_calls": summary["visual_calls"],
                "condition_complete_edge_count": summary[
                    "condition_complete_edge_count"
                ],
            },
        },
        "semantics": {
            "unresolved_is_not_failure": True,
            "provider_configured_is_not_provider_invoked": True,
            "gates_are_shown_independently": True,
            "no_legacy_dossier_or_local_pdf_seed": True,
        },
    }
    _write_json(output_dir / "summary.json", payload)
    (output_dir / "index.html").write_text(
        _render(payload),
        encoding="utf-8",
    )
    print(json.dumps({
        "index_html": str(output_dir / "index.html"),
        "summary_json": str(output_dir / "summary.json"),
        "target_count": len(rows),
        "workbench_count": sum(bool(row.get("workbench_file")) for row in rows),
    }, ensure_ascii=False, indent=2))
    return 0


def _compile_rows(
    panel: Mapping[str, Any],
    *,
    output_dir: Path,
    artifact_role: str = "baseline",
) -> list[dict[str, Any]]:
    panel_root = Path(str(panel.get("output_root") or "")).resolve()
    paths = RuntimePaths.discover(
        repository_root=ROOT,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(panel_root / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(panel_root / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(panel_root / "artifacts"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(panel_root / "runtime" / "run_index.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(panel_root / "external"),
            "AUTOPLANNER_VENDOR_ROOT": str(ROOT / "vendor"),
        },
    )
    gateway = CampaignGateway(paths)
    rows: list[dict[str, Any]] = []
    for target_name, raw in dict(panel.get("targets") or {}).items():
        state = dict(raw or {})
        report_path = Path(str(state.get("report_path") or ""))
        if not report_path.is_file():
            candidate = panel_root / "runs" / str(target_name) / "target-only-solve-report.json"
            report_path = candidate if candidate.is_file() else report_path
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        run_id = str(report.get("run_id") or state.get("run_id") or "")
        gates = dict(dict(report.get("gates") or {}).get("gates") or state.get("gate_summary") or {})
        normalized_gates = {
            key: bool(
                gates.get(key)
                if key in gates
                else next(
                    (
                        value
                        for gate_name, value in gates.items()
                        if str(gate_name).startswith(key + "_")
                    ),
                    False,
                )
            )
            for key in ("B0", "B1", "B2", "B3", "B4", "B5")
        }
        stages = [dict(value) for value in report.get("stages") or [] if isinstance(value, Mapping)]
        chemenzy_details = [
            dict(row.get("detail") or {})
            for row in stages
            if row.get("stage") in {"chemenzy_guided_frontier", "chemenzy_stock_recovery"}
        ]
        evidence_passes = [
            dict(row.get("detail") or {})
            for row in stages
            if row.get("stage")
            in {"evidence_acquisition", "replan_evidence_acquisition"}
        ]
        evidence = evidence_passes[-1] if evidence_passes else {}
        source_route_passes = [
            dict(value.get("source_route") or {}) for value in evidence_passes
        ]
        delegation = next(
            (dict(row.get("detail") or {}) for row in stages if row.get("stage") == "chemenzy_delegation"),
            {},
        )
        model_cost = dict(report.get("model_cost") or state.get("model_cost") or {})
        counts = dict(dict(report.get("gates") or {}).get("counts") or state.get("route_counts") or {})
        timing = _stage_timing_summary(stages)
        event_timing = _event_runtime_summary(
            report_path.parent / ".autoplanner" / "kernel" / "events.jsonl",
            fallback=timing,
        )
        workbench_file = ""
        workbench_metrics: dict[str, Any] = {}
        if run_id:
            try:
                snapshot = gateway.workbench(run_id)["snapshot"]
                replacement = dict(snapshot.get("replacement_validation") or {})
                origins = {
                    str(kind)
                    for edge in dict(snapshot.get("edges") or {}).values()
                    if isinstance(edge, Mapping)
                    for kind in edge.get("origin_kinds") or []
                    if str(kind)
                }
                route_values = [
                    dict(route)
                    for route in dict(snapshot.get("routes") or {}).values()
                    if isinstance(route, Mapping)
                ]
                workbench_metrics = {
                    "route_count": len(route_values),
                    "molecule_count": len(snapshot.get("molecules") or {}),
                    "edge_count": len(snapshot.get("edges") or {}),
                    "module_count": len(snapshot.get("modules") or {}),
                    "replacement_candidate_count": int(
                        replacement.get("candidate_count") or 0
                    ),
                    "validated_replacement_count": int(
                        replacement.get("validated_count") or 0
                    ),
                    "replacement_route_count": len(
                        snapshot.get("replacement_routes") or {}
                    ),
                    "origin_kinds": sorted(origins),
                    "max_route_steps": max(
                        (
                            len(dict(route).get("steps") or [])
                            or len(dict(route).get("edge_ids") or [])
                            for route in route_values
                        ),
                        default=0,
                    ),
                    "exploration_closed_route_count": sum(
                        route.get("closure_profile") == "exploration_closed"
                        for route in route_values
                    ),
                    "procurement_closed_route_count": sum(
                        route.get("closure_profile") == "procurement_closed"
                        for route in route_values
                    ),
                    "process_ready_route_count": sum(
                        route.get("process_ready") is True
                        for route in route_values
                    ),
                    "condition_complete_edge_count": sum(
                        dict(dict(edge).get("proof_vector") or {}).get(
                            "condition_completeness"
                        )
                        == "complete"
                        for edge in dict(snapshot.get("edges") or {}).values()
                        if isinstance(edge, Mapping)
                    ),
                }
                workbench_file = f"{_slug(str(target_name))}-workbench.html"
                (output_dir / workbench_file).write_text(
                    render_v4_route_workbench_html(snapshot),
                    encoding="utf-8",
                )
            except Exception:
                workbench_file = ""
        rows.append(
            {
                "target_name": str(target_name),
                "artifact_role": artifact_role,
                "status": str(state.get("status") or ("completed" if report else "queued")),
                "run_id": run_id,
                "claim": str(dict(report.get("claim") or {}).get("achieved_profile") or state.get("claim") or "unresolved"),
                "time_to_first_route_s": event_timing["time_to_first_route_s"],
                "full_pass_s": event_timing["full_pass_s"],
                "resume_elapsed_s": float(state.get("elapsed_s") or 0.0),
                "stage_timings": timing["stages"],
                "runtime_timing_semantics": event_timing["semantics"],
                "model_cost": model_cost,
                "gates": normalized_gates,
                "highest_gate": str(dict(report.get("gates") or {}).get("highest_contiguous_gate") or _highest_gate(normalized_gates)),
                "route_counts": counts,
                "chemenzy": {
                    "delegation_status": str(delegation.get("status") or "not_recorded"),
                    "delegated_requests": int(delegation.get("request_count") or 0),
                    "delegated_queued": int(delegation.get("queued_count") or 0),
                    "provider_calls": sum(int(value.get("provider_invocation_count") or value.get("executed_frontier_count") or 0) for value in chemenzy_details),
                    "proposals": sum(int(value.get("proposal_count") or 0) for value in chemenzy_details),
                    "stock_recovery": len(chemenzy_details) > 1 and chemenzy_details[-1].get("status") == "completed",
                },
                "evidence": {
                    "status": str(evidence.get("status") or "not_recorded"),
                    "passes": len(evidence_passes),
                    "sources": sum(
                        max(
                            int(value.get("source_count") or 0),
                            len(
                                dict(value.get("discovery") or {}).get("sources")
                                or []
                            ),
                        )
                        for value in evidence_passes
                    ),
                    "bindings": sum(
                        int(value.get("source_binding_count") or 0)
                        for value in evidence_passes
                    ),
                    "exact_rows": sum(
                        int(value.get("exact_record_count") or 0)
                        for value in evidence_passes
                    ),
                    "visual_calls": sum(
                        int(value.get("visual_invocations") or 0)
                        for value in evidence_passes
                    ),
                    "source_route_proposals": sum(
                        int(value.get("proposal_count") or 0)
                        for value in source_route_passes
                    ),
                    "source_route_validated": sum(
                        int(
                            dict(value.get("validation") or {}).get(
                                "accepted_validation_count"
                            )
                            or 0
                        )
                        for value in source_route_passes
                    ),
                },
                "attempt_count": int(report.get("attempt_count") or state.get("attempt_count") or 0),
                "accepted_expansions": int(report.get("accepted_expansion_count") or state.get("accepted_expansion_count") or 0),
                "workbench": workbench_metrics,
                "workbench_file": workbench_file,
                "report_path": str(report_path) if report_path.is_file() else "",
                "error": str(state.get("error") or "")[:500],
            }
        )
    return sorted(rows, key=lambda row: row["target_name"])


def _merge_target_rows(
    baseline: list[dict[str, Any]],
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {str(row.get("target_name") or "").casefold(): row for row in baseline}
    for row in overrides:
        name = str(row.get("target_name") or "").casefold()
        if name:
            merged[name] = row
    return sorted(merged.values(), key=lambda row: str(row.get("target_name") or ""))


def _merge_validation_rows(
    baseline: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay current host proof without erasing original blind-run cost."""

    merged = {
        str(row.get("target_name") or "").casefold(): dict(row)
        for row in baseline
    }
    for validation in validation_rows:
        name = str(validation.get("target_name") or "").casefold()
        if not name:
            continue
        original = dict(merged.get(name) or {})
        if not original:
            merged[name] = dict(validation)
            continue
        current = {**original, **validation}
        for field in (
            "model_cost",
            "time_to_first_route_s",
            "full_pass_s",
            "resume_elapsed_s",
            "runtime_timing_semantics",
            "chemenzy",
        ):
            current[field] = original.get(field)
        current["source_blind_run_id"] = str(original.get("run_id") or "")
        current["validation_model_cost"] = dict(
            validation.get("model_cost") or {}
        )
        current["artifact_role"] = "latest_evidence_validation_fork"
        merged[name] = current
    return sorted(
        merged.values(),
        key=lambda row: str(row.get("target_name") or ""),
    )


def _summary(panel: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("report_path")]
    first_route = [
        float(row.get("time_to_first_route_s") or 0.0)
        for row in completed
        if float(row.get("time_to_first_route_s") or 0.0) > 0
    ]
    full_pass = [
        float(row.get("full_pass_s") or 0.0)
        for row in completed
        if float(row.get("full_pass_s") or 0.0) > 0
    ]
    return {
        "target_count": len(rows),
        "report_count": len(completed),
        "model": str(panel.get("model") or ""),
        "codex_calls": sum(int(dict(row.get("model_cost") or {}).get("model_invocations") or 0) for row in completed),
        "visual_calls": sum(int(dict(row.get("model_cost") or {}).get("visual_invocations") or 0) for row in completed),
        "chemenzy_calls": sum(int(dict(row.get("chemenzy") or {}).get("provider_calls") or 0) for row in completed),
        "input_tokens": sum(int(dict(row.get("model_cost") or {}).get("input_tokens") or 0) for row in completed),
        "median_time_to_first_route_s": (
            round(statistics.median(first_route), 3) if first_route else 0.0
        ),
        "median_full_pass_s": (
            round(statistics.median(full_pass), 3) if full_pass else 0.0
        ),
        "gate_counts": {
            gate: sum(dict(row.get("gates") or {}).get(gate) is True for row in completed)
            for gate in ("B0", "B1", "B2", "B3", "B4", "B5")
        },
        "route_count": sum(
            int(dict(row.get("workbench") or {}).get("route_count") or 0)
            for row in completed
        ),
        "validated_replacement_count": sum(
            int(
                dict(row.get("workbench") or {}).get(
                    "validated_replacement_count"
                )
                or 0
            )
            for row in completed
        ),
        "condition_complete_edge_count": sum(
            int(
                dict(row.get("workbench") or {}).get(
                    "condition_complete_edge_count"
                )
                or 0
            )
            for row in completed
        ),
        "source_route_validated_count": sum(
            int(
                dict(row.get("evidence") or {}).get(
                    "source_route_validated"
                )
                or 0
            )
            for row in completed
        ),
    }


def _render(payload: Mapping[str, Any]) -> str:
    summary = dict(payload["summary"])
    revision = dict(payload.get("runtime_revision") or {})
    cards = "".join(_target_card(row) for row in payload["targets"])
    gate_counts = dict(summary.get("gate_counts") or {})
    implemented = "".join(
        f"<li>{_h(item)}</li>" for item in revision.get("implemented") or []
    )
    demo_links = dict(payload.get("demo_links") or {})
    supplemental_links = "".join(
        f'<a class="button" href="{_h(row.get("href", ""))}">{_h(row.get("label", "补充案例"))}</a>'
        for row in demo_links.get("supplemental") or []
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoPlanner V4 · Blind Retrosynthesis Review</title>
<style>
:root{{--ink:#172033;--muted:#6d778c;--line:#dfe4ee;--panel:#fff;--bg:#f4f6fa;--blue:#3e5eea;--green:#18856f;--amber:#a76913;--red:#bb4554}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,"PingFang SC","Microsoft YaHei",sans-serif}}header{{padding:38px clamp(20px,5vw,72px) 28px;background:linear-gradient(135deg,#17213d,#273c87 68%,#386fc5);color:#fff}}header small{{letter-spacing:.16em;text-transform:uppercase;opacity:.68}}h1{{font-size:clamp(28px,4vw,48px);line-height:1.08;margin:10px 0}}header p{{max-width:850px;color:#dce4ff}}main{{max-width:1500px;margin:auto;padding:24px}}.summary{{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:10px;margin-top:-46px}}.metric,.card,.architecture,.revision{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 30px #15254b0c}}.metric{{padding:16px}}.metric b{{display:block;font-size:23px}}.metric span,.muted{{color:var(--muted);font-size:12px}}.demo-links{{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 0}}.architecture{{margin:14px 0;padding:18px;display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}.architecture div{{border-left:3px solid #6f84e8;padding:5px 12px}}.architecture b{{display:block}}.revision{{margin:0 0 22px;padding:18px;display:grid;grid-template-columns:minmax(210px,.7fr) minmax(360px,1.8fr);gap:22px;border-color:#e6d5a9;background:linear-gradient(135deg,#fffdf7,#fff)}}.revision h2{{margin:0 0 6px}}.revision ul{{margin:0;padding-left:20px}}.revision li+li{{margin-top:4px}}.baseline{{color:var(--amber);font-weight:700}}.slo{{margin-top:10px;padding:9px 11px;border-radius:8px;background:#f4f7ff;color:#334886}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:14px}}.card{{padding:17px;min-width:0}}.top{{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}}h2{{font-size:20px;margin:0}}code{{font-size:10px;color:#7a8499;word-break:break-all}}.pill{{font-size:11px;border-radius:99px;padding:4px 8px;background:#eef1f6;color:#58657a}}.gates{{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin:14px 0}}.gate{{text-align:center;padding:7px 2px;border-radius:7px;background:#f1f2f5;color:#8a93a4;font-weight:800}}.gate.on{{background:#e7f6f1;color:var(--green)}}.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}}.fact{{background:#f7f8fb;border-radius:8px;padding:8px}}.fact b{{display:block}}.provider{{margin:12px 0;padding:10px;border:1px solid var(--line);border-radius:9px}}.provider strong{{color:var(--blue)}}.links{{display:flex;gap:8px;flex-wrap:wrap}}a{{color:var(--blue);text-decoration:none;font-weight:700}}a.button{{background:#edf1ff;border-radius:8px;padding:7px 10px}}.warn{{color:var(--amber)}}.error{{color:var(--red);font-size:11px;word-break:break-all}}footer{{padding:28px;color:var(--muted);text-align:center}}@media(max-width:1000px){{.summary{{grid-template-columns:repeat(3,1fr)}}.architecture{{grid-template-columns:1fr 1fr}}.revision{{grid-template-columns:1fr}}.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:650px){{.summary,.grid{{grid-template-columns:1fr}}.architecture{{grid-template-columns:1fr}}}}
</style></head><body>
<header><small>Target-only · isolated · auditable</small><h1>V4 全新分子盲测工作台</h1><p>输入仅含目标名称与唯一 SMILES；无本地案卷、DOI、专利号、PDF 或既有路线种子。这里分开呈现全局规划、局部候选、确定性验证、证据和库存，不用“分支数”冒充完成度。</p></header>
<main><section class="summary">{_metric('已生成报告',f"{summary['report_count']} / {summary['target_count']}")}{_metric('当前主路线',summary['route_count'])}{_metric('条件完整边',summary['condition_complete_edge_count'])}{_metric('文献验证边',summary['source_route_validated_count'])}{_metric('Codex / ChemEnzy',f"{summary['codex_calls']} / {summary['chemenzy_calls']}")}{_metric('B1 / B3 / B5',f"{gate_counts.get('B1',0)} / {gate_counts.get('B3',0)} / {gate_counts.get('B5',0)}")}</section>
<nav class="demo-links"><a class="button" href="{_h(demo_links.get('console_url',''))}">打开实时控制台</a>{supplemental_links}</nav>
<section class="architecture"><div><b>Codex</b><span class="muted">一次总揽路线族、战略断键与局部前沿</span></div><div><b>ChemEnzy</b><span class="muted">只展开 host 接纳的子目标或库存缺口</span></div><div><b>Evidence</b><span class="muted">HTML-first，PDF 回退，视觉稀疏触发</span></div><div><b>Host verifier</b><span class="muted">守恒、映射、循环、原子突变与条件</span></div><div><b>Inventory</b><span class="muted">逐叶审计，闭合与可采购分离</span></div></section>
<section class="revision"><div><h2>冷启动性能说明</h2><p class="baseline">上方首路线秒数仍是不可变的旧冷跑基线；证据验证优化已另行实测。</p><p class="muted">{_h(revision.get('benchmark_semantics',''))}</p></div><div><ul>{implemented}</ul><div class="slo"><b>下一验收门：</b> {_h(revision.get('next_acceptance',''))}</div></div></section>
<section class="grid">{cards}</section></main><footer>Generated {_h(payload['generated_at'])} · 所有 unresolved 均保留真实缺口，不自动升级为 solved。</footer></body></html>"""


def _target_card(row: Mapping[str, Any]) -> str:
    gates = dict(row.get("gates") or {})
    counts = dict(row.get("route_counts") or {})
    chem = dict(row.get("chemenzy") or {})
    evidence = dict(row.get("evidence") or {})
    cost = dict(row.get("model_cost") or {})
    workbench = dict(row.get("workbench") or {})
    origins = ", ".join(
        _origin_label(value) for value in workbench.get("origin_kinds") or []
    ) or "未记录"
    artifact_note = (
        {
            "latest_independent_blind_rerun": " · 最新独立盲跑",
            "latest_evidence_validation_fork": " · 最新证据/验证叉",
            "latest_zero_model_validation_fork": " · 最新零模型验证叉",
        }.get(str(row.get("artifact_role") or ""), "")
    )
    gate_html = "".join(
        f'<span class="gate {"on" if gates.get(key) else ""}" title="{_h(_gate_label(key))}">{key}</span>'
        for key in ("B0", "B1", "B2", "B3", "B4", "B5")
    )
    link = (
        f'<a class="button" href="{_h(row["workbench_file"])}">打开路线工作台</a>'
        if row.get("workbench_file")
        else '<span class="warn">工作台待生成</span>'
    )
    validation_cost = dict(row.get("validation_model_cost") or {})
    validation_note = ""
    if validation_cost:
        validation_note = (
            '<br><span class="muted">增量证据验证：'
            f'{int(validation_cost.get("model_invocations") or 0)} 次模型 · '
            f'{int(validation_cost.get("visual_invocations") or 0)} 次视觉 · '
            f'{int(validation_cost.get("input_tokens") or 0):,} 输入 token'
            "</span>"
        )
    return f"""<article class="card"><div class="top"><div><h2>{_h(row['target_name'])}</h2><code>{_h(row.get('run_id',''))}{_h(artifact_note)}</code></div><span class="pill">{_h(row.get('claim','unresolved'))}</span></div><div class="gates">{gate_html}</div><div class="facts">{_fact('首个路线',_duration(row.get('time_to_first_route_s',0)))}{_fact('最长路线',f"{int(workbench.get('max_route_steps') or 0)} 步")}{_fact('Codex',f"{int(cost.get('model_invocations') or 0)} 次")}{_fact('当前主路线',workbench.get('route_count',0))}{_fact('完整条件边',workbench.get('condition_complete_edge_count',0))}{_fact('已验证替换',workbench.get('validated_replacement_count',0))}</div><div class="provider"><strong>ChemEnzy</strong> · 实调 {int(chem.get('provider_calls') or 0)} · 候选 {int(chem.get('proposals') or 0)} · 委派 { _h(chem.get('delegation_status','')) }<br><span class="muted">边界状态：采购闭合 {int(workbench.get('procurement_closed_route_count') or 0)} · benchmark/探索闭合 {int(workbench.get('exploration_closed_route_count') or 0)} · 工艺就绪 {int(workbench.get('process_ready_route_count') or 0)}</span><br><span class="muted">来源引擎：{_h(origins)} · 分子 {int(workbench.get('molecule_count') or 0)} · 反应边 {int(workbench.get('edge_count') or 0)} · 库存闭合 {counts.get('stock_closed_skeletons',0)}</span><br><span class="muted">证据轮次 {int(evidence.get('passes') or 0)} · 来源 {int(evidence.get('sources') or 0)} · 文献路线 {int(evidence.get('source_route_validated') or 0)}/{int(evidence.get('source_route_proposals') or 0)} 验证 · exact rows {int(evidence.get('exact_rows') or 0)} · visual {int(evidence.get('visual_calls') or 0)}</span>{validation_note}</div><div class="links">{link}</div>{f'<p class="error">{_h(row["error"])} </p>' if row.get('error') else ''}</article>"""


def _origin_label(value: Any) -> str:
    return {
        "codex_global_director": "Codex 全局规划",
        "chemenzy": "ChemEnzy 局部展开",
        "literature_visual_extraction": "文献视觉提取",
        "patent_template_reuse": "专利 self-evo",
        "source_route_observation": "文献确定性提取",
    }.get(str(value), str(value))


def _supplemental_link(value: str, *, output_dir: Path) -> dict[str, str]:
    label, separator, raw_path = str(value).partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("supplemental_showcase_must_be_LABEL_equals_PATH")
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        path = path / "index.html"
    if not path.is_file():
        raise ValueError(f"supplemental_showcase_missing:{path}")
    return {
        "label": label.strip(),
        "href": Path(os.path.relpath(path, output_dir)).as_posix(),
    }


def _metric(label: str, value: Any) -> str:
    return f'<article class="metric"><b>{_h(value)}</b><span>{_h(label)}</span></article>'


def _fact(label: str, value: Any) -> str:
    return f'<div class="fact"><b>{_h(value)}</b><span class="muted">{_h(label)}</span></div>'


def _gate_label(key: str) -> str:
    return {
        "B0": "盲输入合规", "B1": "全局多路线", "B2": "host 反应验证",
        "B3": "exact 多信源", "B4": "库存边界", "B5": "配置验收",
    }[key]


def _stage_timing_summary(stages: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate the original bounded pass from later cache/resume latency."""

    maxima: dict[str, float] = {}
    for row in stages:
        stage = str(row.get("stage") or "")
        if not stage:
            continue
        detail = dict(row.get("detail") or {})
        elapsed = max(
            0.0,
            float(row.get("elapsed_s") or detail.get("elapsed_s") or 0.0),
        )
        maxima[stage] = max(maxima.get(stage, 0.0), elapsed)
    first_route_stages = (
        "target_identity",
        "patent_template_retrieval",
        "chemenzy_baseline",
        "global_campaign",
        "materialization",
        "patent_template_reuse",
        "reaction_validation",
        "precursor_repair",
        "initial_workbench",
    )
    time_to_first_route = sum(maxima.get(stage, 0.0) for stage in first_route_stages)
    return {
        "time_to_first_route_s": round(time_to_first_route, 3),
        "full_pass_s": round(sum(maxima.values()), 3),
        "stages": dict(sorted(maxima.items())),
        "semantics": {
            "repeat_resume_stages_use_max_not_sum": True,
            "panel_resume_elapsed_not_presented_as_initial_runtime": True,
        },
    }


def _event_runtime_summary(
    path: Path,
    *,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the first run cluster without counting later resume sessions."""

    if not path.is_file():
        return {
            "time_to_first_route_s": float(
                fallback.get("time_to_first_route_s") or 0.0
            ),
            "full_pass_s": float(fallback.get("full_pass_s") or 0.0),
            "semantics": {"source": "stage_elapsed_fallback"},
        }
    events: list[tuple[datetime, dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            created = datetime.fromisoformat(
                str(row.get("created_at") or "").replace("Z", "+00:00")
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        events.append((created, row))
    if not events:
        return {
            "time_to_first_route_s": float(
                fallback.get("time_to_first_route_s") or 0.0
            ),
            "full_pass_s": float(fallback.get("full_pass_s") or 0.0),
            "semantics": {"source": "stage_elapsed_fallback"},
        }
    events.sort(key=lambda item: item[0])
    started = events[0][0]
    cluster_end = events[-1][0]
    first_route_at: datetime | None = None
    in_flight: set[str] = set()
    previous_at = started
    for created, row in events:
        if (
            (created - previous_at).total_seconds() > 120.0
            and not in_flight
        ):
            cluster_end = previous_at
            break
        event_type = str(row.get("event_type") or "")
        payload = dict(row.get("payload") or {})
        task_id = str(payload.get("task_id") or "")
        if event_type == "task_reserved" and task_id:
            in_flight.add(task_id)
        elif event_type == "task_settled":
            in_flight.discard(task_id)
            model_usage = dict(payload.get("model_usage") or {})
            if (
                first_route_at is None
                and int(model_usage.get("model_invocations") or 0) > 0
                and payload.get("status") == "completed"
            ):
                first_route_at = created
        previous_at = created
    return {
        "time_to_first_route_s": round(
            max(
                0.0,
                (
                    (first_route_at - started).total_seconds()
                    if first_route_at is not None
                    else float(fallback.get("time_to_first_route_s") or 0.0)
                ),
            ),
            3,
        ),
        "full_pass_s": round(max(0.0, (cluster_end - started).total_seconds()), 3),
        "semantics": {
            "source": "kernel_event_timestamps",
            "first_idle_gap_over_120s_starts_resume_session": True,
            "in_flight_model_or_worker_gap_remains_in_first_pass": True,
        },
    }


def _highest_gate(gates: Mapping[str, bool]) -> str:
    highest = "none"
    for key in ("B0", "B1", "B2", "B3", "B4", "B5"):
        if gates.get(key) is not True:
            break
        highest = key
    return highest


def _duration(value: Any) -> str:
    seconds = max(0.0, float(value or 0.0))
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {int(round(seconds % 60))}s"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-") or "target"


def _h(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
