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
        "--supplemental-showcase",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    args = parser.parse_args(argv)

    status_path = Path(args.panel_status).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = json.loads(status_path.read_text(encoding="utf-8"))
    rows = _compile_rows(panel, output_dir=output_dir)
    summary = _summary(panel, rows)
    payload = {
        "schema_version": "v4_blind_expert_showcase.v1",
        "generated_at": _utc_now(),
        "panel_status_path": str(status_path),
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
                "目标卡片来自不可变的真实冷跑 artifact；其中墙钟时间早于下列延迟优化。"
            ),
            "implemented": [
                "快速首轮只生成 2 条完整路线族，每条最多 5 步，输出上限 3,800 tokens。",
                "初始 Codex 默认不携带联网工具；HTML/XML 证据预取与全局规划并行。",
                "延迟隐藏阶段执行 HTML-first，不盲目下载 PDF。",
                "证据只允许触发 1 次增量重规划，且只传入排名最高的 4 个来源。",
                "Simvastatin 样例的证据上下文由 21,250 降至 9,130 bytes（减少 57.0%）。",
            ],
            "next_acceptance": (
                "使用 1 个全新复杂目标验证：90 秒内发布 host-validated 首骨架，"
                "Codex 最多 2 次，默认视觉调用 0 次。"
            ),
            "measured_after_revision": False,
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


def _compile_rows(panel: Mapping[str, Any], *, output_dir: Path) -> list[dict[str, Any]]:
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
        evidence = next(
            (dict(row.get("detail") or {}) for row in stages if row.get("stage") == "evidence_acquisition"),
            {},
        )
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
        if run_id:
            try:
                snapshot = gateway.workbench(run_id)["snapshot"]
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
                    "sources": int(evidence.get("source_count") or 0),
                    "exact_rows": int(evidence.get("exact_record_count") or 0),
                    "visual_calls": int(evidence.get("visual_invocations") or 0),
                },
                "attempt_count": int(report.get("attempt_count") or state.get("attempt_count") or 0),
                "accepted_expansions": int(report.get("accepted_expansion_count") or state.get("accepted_expansion_count") or 0),
                "workbench_file": workbench_file,
                "report_path": str(report_path) if report_path.is_file() else "",
                "error": str(state.get("error") or "")[:500],
            }
        )
    return sorted(rows, key=lambda row: row["target_name"])


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
<main><section class="summary">{_metric('已生成报告',f"{summary['report_count']} / {summary['target_count']}")}{_metric('Codex 实调',summary['codex_calls'])}{_metric('ChemEnzy 实调',summary['chemenzy_calls'])}{_metric('中位首路线',_duration(summary['median_time_to_first_route_s']))}{_metric('中位完整首轮',_duration(summary['median_full_pass_s']))}{_metric('B1 / B3 / B5',f"{gate_counts.get('B1',0)} / {gate_counts.get('B3',0)} / {gate_counts.get('B5',0)}")}</section>
<nav class="demo-links"><a class="button" href="{_h(demo_links.get('console_url',''))}">打开实时控制台</a>{supplemental_links}</nav>
<section class="architecture"><div><b>Codex</b><span class="muted">一次总揽路线族、战略断键与局部前沿</span></div><div><b>ChemEnzy</b><span class="muted">只展开 host 接纳的子目标或库存缺口</span></div><div><b>Evidence</b><span class="muted">HTML-first，PDF 回退，视觉稀疏触发</span></div><div><b>Host verifier</b><span class="muted">守恒、映射、循环、原子突变与条件</span></div><div><b>Inventory</b><span class="muted">逐叶审计，闭合与可采购分离</span></div></section>
<section class="revision"><div><h2>冷启动性能说明</h2><p class="baseline">上方秒数是不可变的旧冷跑基线，不是优化后的实测成绩。</p><p class="muted">{_h(revision.get('benchmark_semantics',''))}</p></div><div><ul>{implemented}</ul><div class="slo"><b>下一验收门：</b> {_h(revision.get('next_acceptance',''))}</div></div></section>
<section class="grid">{cards}</section></main><footer>Generated {_h(payload['generated_at'])} · 所有 unresolved 均保留真实缺口，不自动升级为 solved。</footer></body></html>"""


def _target_card(row: Mapping[str, Any]) -> str:
    gates = dict(row.get("gates") or {})
    counts = dict(row.get("route_counts") or {})
    chem = dict(row.get("chemenzy") or {})
    evidence = dict(row.get("evidence") or {})
    cost = dict(row.get("model_cost") or {})
    gate_html = "".join(
        f'<span class="gate {"on" if gates.get(key) else ""}" title="{_h(_gate_label(key))}">{key}</span>'
        for key in ("B0", "B1", "B2", "B3", "B4", "B5")
    )
    link = (
        f'<a class="button" href="{_h(row["workbench_file"])}">打开路线工作台</a>'
        if row.get("workbench_file")
        else '<span class="warn">工作台待生成</span>'
    )
    return f"""<article class="card"><div class="top"><div><h2>{_h(row['target_name'])}</h2><code>{_h(row.get('run_id',''))}</code></div><span class="pill">{_h(row.get('claim','unresolved'))}</span></div><div class="gates">{gate_html}</div><div class="facts">{_fact('首个路线',_duration(row.get('time_to_first_route_s',0)))}{_fact('完整首轮',_duration(row.get('full_pass_s',0)))}{_fact('Codex',f"{int(cost.get('model_invocations') or 0)} 次")}{_fact('路线骨架',counts.get('target_rooted_distinct_skeletons',0))}{_fact('反应验证',counts.get('reaction_validated_skeletons',0))}{_fact('库存闭合',counts.get('stock_closed_skeletons',0))}</div><div class="provider"><strong>ChemEnzy</strong> · 实调 {int(chem.get('provider_calls') or 0)} · 候选 {int(chem.get('proposals') or 0)} · 委派 { _h(chem.get('delegation_status','')) }<br><span class="muted">Codex 输入 {int(cost.get('input_tokens') or 0):,} token · Evidence 来源 {int(evidence.get('sources') or 0)} · exact rows {int(evidence.get('exact_rows') or 0)} · visual {int(evidence.get('visual_calls') or 0)}</span></div><div class="links">{link}</div>{f'<p class="error">{_h(row["error"])}</p>' if row.get('error') else ''}</article>"""


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
