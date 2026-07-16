"""Summarize the frozen PaRoutes multistep panel without leaking references.

The report keeps runtime availability, route retention, host acceptance, stock
closure, and reference recovery as separate axes.  In particular, a retained
low-confidence route is not rewritten as either an accepted route or a runtime
failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "classic_multistep_benchmark_summary.v1"


def summarize_classic_multistep_benchmark(
    *,
    reference_pack: Path,
    runs: Mapping[str, Path],
    proxies: Mapping[str, Path],
    output_json: Path,
    output_md: Path | None = None,
    output_html: Path | None = None,
    v4_panel_status: Path | None = None,
) -> dict[str, Any]:
    reference = _read_json(reference_pack)
    cases = [dict(row) for row in reference.get("cases") or []]
    run_payloads = {split: _read_json(path) for split, path in runs.items()}
    proxy_payloads = {split: _read_json(path) for split, path in proxies.items()}
    v4_panel = _read_json(v4_panel_status) if v4_panel_status else {}
    v4_targets = dict(v4_panel.get("targets") or {})
    rows: list[dict[str, Any]] = []
    for case in cases:
        split = str(case.get("split") or "")
        target_smiles = str(case.get("target_smiles") or "")
        target = _target_by_smiles(run_payloads.get(split, {}), target_smiles)
        proxy = _proxy_by_id(
            proxy_payloads.get(split, {}), str(case.get("case_id") or "")
        )
        rows.append(
            _case_row(
                case,
                target,
                proxy,
                dict(v4_targets.get(str(case.get("target_name") or "")) or {}),
            )
        )

    aggregates = {
        "overall": _aggregate(rows),
        "by_split": {
            split: _aggregate([row for row in rows if row["split"] == split])
            for split in sorted({row["split"] for row in rows})
        },
        "by_depth_stratum": {
            stratum: _aggregate(
                [row for row in rows if row["depth_stratum"] == stratum]
            )
            for stratum in sorted({row["depth_stratum"] for row in rows})
        },
        "v4_panel": _aggregate_v4(rows, v4_panel),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "benchmark": {
            "name": "PaRoutes classic multistep 20",
            "manifest_sha256": str(
                dict(reference.get("metadata") or {}).get("manifest_sha256")
                or ""
            ),
            "target_count": len(rows),
            "splits": sorted({row["split"] for row in rows}),
            "reference_is_evaluator_only": True,
        },
        "aggregates": aggregates,
        "targets": rows,
        "semantics": {
            "route_length_is_descriptive_not_an_acceptance_gate": True,
            "shorter_valid_closed_route_is_allowed": True,
            "low_confidence_routes_are_retained_with_warnings": True,
            "runtime_failure_is_separate_from_route_rejection": True,
            "stock_closure_is_the_split_specific_paroutes_boundary": True,
            "reference_recovery_is_an_atom_map_invariant_proxy": True,
            "reference_recovery_is_not_the_official_paroutes_tree_metric": True,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown(report), encoding="utf-8")
    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(_html(report), encoding="utf-8")
    return report


def _case_row(
    case: Mapping[str, Any],
    target: Mapping[str, Any],
    proxy: Mapping[str, Any],
    v4_state: Mapping[str, Any],
) -> dict[str, Any]:
    chem_enzy = dict(target.get("chem_enzy") or {})
    cascade = dict(target.get("cascade_search") or {})
    programs = [
        row
        for row in cascade.get("result_programs") or []
        if isinstance(row, Mapping)
    ]
    top10 = dict(dict(proxy.get("topk") or {}).get("10") or {})
    reference_metrics = dict(case.get("reference_metrics") or {})
    failures = Counter(
        _failure_category(value)
        for value in cascade.get("failure_categories") or []
    )
    failures.update(
        _failure_category(value) for value in chem_enzy.get("failures") or []
    )
    return {
        "case_id": str(case.get("case_id") or ""),
        "split": str(case.get("split") or ""),
        "depth_stratum": str(case.get("depth_stratum") or ""),
        "reference_reaction_count": int(
            reference_metrics.get("reaction_count") or 0
        ),
        "reference_longest_linear_depth": int(
            reference_metrics.get("longest_linear_depth") or 0
        ),
        "runtime_completed": bool(target),
        "chem_enzy_route_found": chem_enzy.get("solved") is True,
        "chem_enzy_route_count": int(chem_enzy.get("route_count") or 0),
        "cascade_route_retained": bool(programs),
        "cascade_retained_route_count": len(programs),
        "cascade_accepted": cascade.get("solved") is True,
        "benchmark_stock_closed": cascade.get("stock_closed") is True,
        "condition_conflict_free": cascade.get("condition_conflict_free") is True,
        "top10_exact_reference_route": (
            top10.get("exact_reaction_sequence_hit") is True
        ),
        "top10_shorter_or_equal_route": top10.get("shorter_or_equal_hit") is True,
        "top10_best_reference_leaf_overlap": round(
            float(top10.get("best_leaf_overlap") or 0.0), 6
        ),
        "cascade_search_elapsed_s": round(float(cascade.get("elapsed_s") or 0.0), 3),
        "warning_counts": dict(sorted(failures.items())),
        "v4": _v4_case_row(v4_state),
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    warnings: Counter[str] = Counter()
    for row in rows:
        warnings.update(dict(row.get("warning_counts") or {}))
    return {
        "target_count": n,
        "runtime_completion_rate": _rate(rows, "runtime_completed"),
        "chem_enzy_route_found_rate": _rate(rows, "chem_enzy_route_found"),
        "cascade_route_retained_rate": _rate(rows, "cascade_route_retained"),
        "cascade_accepted_rate": _rate(rows, "cascade_accepted"),
        "benchmark_stock_closed_rate": _rate(rows, "benchmark_stock_closed"),
        "condition_conflict_free_rate": _rate(rows, "condition_conflict_free"),
        "top10_exact_reference_route_rate": _rate(
            rows, "top10_exact_reference_route"
        ),
        "top10_shorter_or_equal_route_rate": _rate(
            rows, "top10_shorter_or_equal_route"
        ),
        "avg_top10_best_reference_leaf_overlap": round(
            sum(float(row.get("top10_best_reference_leaf_overlap") or 0.0) for row in rows)
            / max(n, 1),
            6,
        ),
        "avg_cascade_search_elapsed_s": round(
            sum(float(row.get("cascade_search_elapsed_s") or 0.0) for row in rows)
            / max(n, 1),
            3,
        ),
        "warning_counts": dict(sorted(warnings.items())),
    }


def _v4_case_row(state: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(state.get("gate_summary") or {})
    counts = dict(state.get("route_counts") or {})
    chemenzy = dict(state.get("chemenzy") or {})
    return {
        "status": str(state.get("status") or "not_run"),
        "claim": str(state.get("claim") or ""),
        "accepted_under_configured_policy": (
            state.get("accepted_under_configured_policy") is True
        ),
        "elapsed_s": round(float(state.get("elapsed_s") or 0.0), 3),
        "gates": {key: gates.get(key) is True for key in ("B0", "B1", "B2", "B3", "B4", "B5")},
        "target_rooted_distinct_skeletons": int(
            counts.get("target_rooted_distinct_skeletons") or 0
        ),
        "reaction_validated_skeletons": int(
            counts.get("reaction_validated_skeletons") or 0
        ),
        "stock_closed_skeletons": int(counts.get("stock_closed_skeletons") or 0),
        "chem_enzy_provider_calls": int(
            chemenzy.get("provider_invocation_count") or 0
        ),
        "chem_enzy_proposals": int(chemenzy.get("proposal_count") or 0),
    }


def _aggregate_v4(
    rows: list[Mapping[str, Any]], panel: Mapping[str, Any]
) -> dict[str, Any]:
    states = [dict(row.get("v4") or {}) for row in rows]
    completed = [row for row in states if row.get("status") == "completed"]
    target_count = int(panel.get("target_count") or len(rows)) if panel else 0
    if not panel:
        return {
            "available": False,
            "target_count": 0,
            "completed_count": 0,
            "completion_rate": 0.0,
        }
    gate_rates = {
        key: round(
            sum(dict(row.get("gates") or {}).get(key) is True for row in completed)
            / max(len(completed), 1),
            6,
        )
        for key in ("B0", "B1", "B2", "B3", "B4", "B5")
    }
    return {
        "available": True,
        "target_count": target_count,
        "completed_count": len(completed),
        "failed_count": sum(row.get("status") == "failed" for row in states),
        "completion_rate": round(len(completed) / max(target_count, 1), 6),
        "accepted_rate_over_completed": round(
            sum(row.get("accepted_under_configured_policy") is True for row in completed)
            / max(len(completed), 1),
            6,
        ),
        "gate_pass_rates_over_completed": gate_rates,
        "total_target_rooted_distinct_skeletons": sum(
            int(row.get("target_rooted_distinct_skeletons") or 0)
            for row in completed
        ),
        "total_reaction_validated_skeletons": sum(
            int(row.get("reaction_validated_skeletons") or 0)
            for row in completed
        ),
        "total_stock_closed_skeletons": sum(
            int(row.get("stock_closed_skeletons") or 0) for row in completed
        ),
        "total_chem_enzy_provider_calls": sum(
            int(row.get("chem_enzy_provider_calls") or 0) for row in completed
        ),
    }


def _rate(rows: list[Mapping[str, Any]], key: str) -> float:
    return round(sum(row.get(key) is True for row in rows) / max(len(rows), 1), 6)


def _failure_category(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("category") or value.get("code") or "unknown_failure")
    return str(value)


def _target_by_smiles(
    run: Mapping[str, Any], target_smiles: str
) -> dict[str, Any]:
    return next(
        (
            dict(row)
            for row in run.get("targets") or []
            if isinstance(row, Mapping)
            and str(row.get("target_smiles") or "") == target_smiles
        ),
        {},
    )


def _proxy_by_id(proxy: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    return next(
        (
            dict(row)
            for row in proxy.get("targets") or []
            if isinstance(row, Mapping)
            and str(row.get("target_id") or "") == case_id
        ),
        {},
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _markdown(report: Mapping[str, Any]) -> str:
    overall = dict(
        dict(report.get("aggregates") or {}).get("overall") or {}
    )
    lines = [
        "# PaRoutes classic multistep 20",
        "",
        (
            "Runtime, retained routes, host acceptance, benchmark-stock closure, "
            "and reference recovery are reported independently."
        ),
        "",
        "| Axis | Result |",
        "| --- | ---: |",
        f"| Targets completed | {overall.get('runtime_completion_rate', 0):.1%} |",
        f"| ChemEnzy route found | {overall.get('chem_enzy_route_found_rate', 0):.1%} |",
        f"| Cascade route retained | {overall.get('cascade_route_retained_rate', 0):.1%} |",
        f"| Cascade accepted | {overall.get('cascade_accepted_rate', 0):.1%} |",
        f"| PaRoutes stock closed | {overall.get('benchmark_stock_closed_rate', 0):.1%} |",
        f"| Top-10 exact reference proxy | {overall.get('top10_exact_reference_route_rate', 0):.1%} |",
        f"| Top-10 shorter/equal | {overall.get('top10_shorter_or_equal_route_rate', 0):.1%} |",
        f"| Top-10 mean reference-leaf overlap | {overall.get('avg_top10_best_reference_leaf_overlap', 0):.1%} |",
        "",
        "Reference matching ignores atom-map annotations and is a flattened proxy, not the official PaRoutes tree metric.",
        "",
        "## By split",
        "",
        "| Split | n | route found | retained | accepted | stock closed | exact ref | leaf overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in dict(
        dict(report.get("aggregates") or {}).get("by_split") or {}
    ).items():
        lines.append(
            f"| {split} | {row['target_count']} | {row['chem_enzy_route_found_rate']:.1%} | "
            f"{row['cascade_route_retained_rate']:.1%} | {row['cascade_accepted_rate']:.1%} | "
            f"{row['benchmark_stock_closed_rate']:.1%} | {row['top10_exact_reference_route_rate']:.1%} | "
            f"{row['avg_top10_best_reference_leaf_overlap']:.1%} |"
        )
    v4 = dict(dict(report.get("aggregates") or {}).get("v4_panel") or {})
    if v4.get("available"):
        gate_rates = dict(v4.get("gate_pass_rates_over_completed") or {})
        lines.extend(
            [
                "",
                "## Full V4 blind panel",
                "",
                f"- Completed: `{v4.get('completed_count', 0)}/{v4.get('target_count', 0)}`",
                f"- B1 structural route: `{gate_rates.get('B1', 0):.1%}`",
                f"- B2 reaction validation: `{gate_rates.get('B2', 0):.1%}`",
                f"- B3 exact evidence: `{gate_rates.get('B3', 0):.1%}`",
                f"- B4 configured stock boundary: `{gate_rates.get('B4', 0):.1%}`",
                f"- Retained skeletons: `{v4.get('total_target_rooted_distinct_skeletons', 0)}`",
                f"- Reaction-validated skeletons: `{v4.get('total_reaction_validated_skeletons', 0)}`",
                f"- Stock-closed skeletons: `{v4.get('total_stock_closed_skeletons', 0)}`",
            ]
        )
    lines.extend(["", "## Warnings", ""])
    for warning, count in dict(overall.get("warning_counts") or {}).items():
        lines.append(f"- `{warning}`: {count}")
    return "\n".join(lines) + "\n"


def _html(report: Mapping[str, Any]) -> str:
    aggregates = dict(report.get("aggregates") or {})
    overall = dict(aggregates.get("overall") or {})
    split_rows = "".join(
        "<tr>"
        f"<td>{escape(split)}</td>"
        f"<td>{int(row.get('target_count') or 0)}</td>"
        f"<td>{float(row.get('chem_enzy_route_found_rate') or 0):.0%}</td>"
        f"<td>{float(row.get('cascade_route_retained_rate') or 0):.0%}</td>"
        f"<td>{float(row.get('cascade_accepted_rate') or 0):.0%}</td>"
        f"<td>{float(row.get('benchmark_stock_closed_rate') or 0):.0%}</td>"
        f"<td>{float(row.get('top10_exact_reference_route_rate') or 0):.0%}</td>"
        f"<td>{float(row.get('avg_top10_best_reference_leaf_overlap') or 0):.1%}</td>"
        "</tr>"
        for split, row in dict(aggregates.get("by_split") or {}).items()
    )
    target_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('case_id') or ''))}</td>"
        f"<td>{escape(str(row.get('split') or ''))}</td>"
        f"<td>{int(row.get('reference_longest_linear_depth') or 0)}</td>"
        f"<td>{_mark(row.get('chem_enzy_route_found'))}</td>"
        f"<td>{_mark(row.get('cascade_route_retained'))}</td>"
        f"<td>{_mark(row.get('cascade_accepted'))}</td>"
        f"<td>{_mark(row.get('benchmark_stock_closed'))}</td>"
        f"<td>{float(row.get('top10_best_reference_leaf_overlap') or 0):.0%}</td>"
        f"<td>{escape(', '.join(dict(row.get('warning_counts') or {})) or '—')}</td>"
        "</tr>"
        for row in report.get("targets") or []
    )
    v4 = dict(aggregates.get("v4_panel") or {})
    v4_gate_rates = dict(v4.get("gate_pass_rates_over_completed") or {})
    v4_target_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('case_id') or ''))}</td>"
        f"<td>{escape(str(dict(row.get('v4') or {{}}).get('status') or 'not_run'))}</td>"
        f"<td>{_mark_gate(row, 'B1')}</td>"
        f"<td>{_mark_gate(row, 'B2')}</td>"
        f"<td>{_mark_gate(row, 'B3')}</td>"
        f"<td>{_mark_gate(row, 'B4')}</td>"
        f"<td>{int(dict(row.get('v4') or {{}}).get('target_rooted_distinct_skeletons') or 0)}</td>"
        f"<td>{int(dict(row.get('v4') or {{}}).get('reaction_validated_skeletons') or 0)}</td>"
        f"<td>{int(dict(row.get('v4') or {{}}).get('stock_closed_skeletons') or 0)}</td>"
        f"<td>{float(dict(row.get('v4') or {{}}).get('elapsed_s') or 0):.1f}s</td>"
        "</tr>"
        for row in report.get("targets") or []
    )
    cards = (
        ("目标完成", "runtime_completion_rate"),
        ("ChemEnzy 找到路线", "chem_enzy_route_found_rate"),
        ("低可信路线保留", "cascade_route_retained_rate"),
        ("Host 接纳", "cascade_accepted_rate"),
        ("PaRoutes 库闭合", "benchmark_stock_closed_rate"),
        ("Top-10 精确参考路线", "top10_exact_reference_route_rate"),
    )
    card_html = "".join(
        f"<article><b>{float(overall.get(key) or 0):.0%}</b><span>{escape(label)}</span></article>"
        for label, key in cards
    )
    v4_html = ""
    if v4.get("available"):
        v4_cards = (
            ("V4 完成", float(v4.get("completion_rate") or 0.0)),
            ("B1 结构路线", float(v4_gate_rates.get("B1") or 0.0)),
            ("B2 反应验证", float(v4_gate_rates.get("B2") or 0.0)),
            ("B3 精确证据", float(v4_gate_rates.get("B3") or 0.0)),
            ("B4 库存边界", float(v4_gate_rates.get("B4") or 0.0)),
            (
                "V4 策略接纳",
                float(v4.get("accepted_rate_over_completed") or 0.0),
            ),
        )
        v4_card_html = "".join(
            f"<article><b>{value:.0%}</b><span>{escape(label)}</span></article>"
            for label, value in v4_cards
        )
        v4_html = f"""
<section><h2>完整 V4 target-only 盲测</h2><div class="cards">{v4_card_html}</div>
<p>已完成 {int(v4.get('completed_count') or 0)}/{int(v4.get('target_count') or 0)}；结构骨架、反应验证、精确证据和库存边界独立计数。</p>
<table><thead><tr><th>Case</th><th>状态</th><th>B1</th><th>B2</th><th>B3</th><th>B4</th><th>骨架</th><th>验证骨架</th><th>库存闭合</th><th>耗时</th></tr></thead><tbody>{v4_target_rows}</tbody></table></section>"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaRoutes 多步盲测 20</title><style>
:root{{--ink:#14213d;--muted:#667085;--line:#dbe3f0;--blue:#315efb;--bg:#f5f7fb;--warn:#b54708}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:32px 24px 56px}}h1{{margin:0 0 6px;font-size:28px}}p{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:10px;margin:22px 0}}article{{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}}article b{{display:block;font-size:24px;color:var(--blue)}}article span{{color:var(--muted)}}
section{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;margin-top:16px;overflow:auto}}h2{{font-size:17px;margin:0 0 14px}}
table{{border-collapse:collapse;width:100%;min-width:760px}}th,td{{padding:10px 9px;border-bottom:1px solid #edf0f5;text-align:left}}th{{font-size:12px;color:var(--muted);background:#f9fafb;position:sticky;top:0}}
.yes{{color:#087443;font-weight:700}}.no{{color:var(--warn);font-weight:700}}.note{{border-left:3px solid #f79009;padding-left:12px}}
@media(max-width:900px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>PaRoutes 经典多步盲测 · 20 targets</h1>
<p class="note">运行时、路线保留、Host 接纳、基准库存闭合与参考路线重合分别报告。低可信路线不会因为条件或中间证据缺口而消失。</p>
<div class="cards">{card_html}</div>
<section><h2>按数据集分层</h2><table><thead><tr><th>Split</th><th>n</th><th>找到路线</th><th>保留</th><th>接纳</th><th>库闭合</th><th>精确参考</th><th>叶节点重合</th></tr></thead><tbody>{split_rows}</tbody></table></section>
{v4_html}
<section><h2>逐目标结果（不展示参考答案）</h2><table><thead><tr><th>Case</th><th>Split</th><th>参考 LLR</th><th>ChemEnzy</th><th>保留</th><th>接纳</th><th>库闭合</th><th>叶节点重合</th><th>警示</th></tr></thead><tbody>{target_rows}</tbody></table></section>
<p>参考比较忽略 atom-map 标签；当前为展平代理指标，不冒充 PaRoutes 官方树级评价。</p>
</main></body></html>"""


def _mark(value: Any) -> str:
    return '<span class="yes">是</span>' if value is True else '<span class="no">否</span>'


def _mark_gate(row: Mapping[str, Any], gate: str) -> str:
    v4 = dict(row.get("v4") or {})
    if v4.get("status") != "completed":
        return "—"
    return _mark(dict(v4.get("gates") or {}).get(gate) is True)


def _parse_bindings(values: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected SPLIT=PATH, got {value!r}")
        split, path = value.split("=", 1)
        output[split.strip()] = Path(path).expanduser()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pack", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="SPLIT=PATH")
    parser.add_argument("--proxy", action="append", required=True, metavar="SPLIT=PATH")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--output-html")
    parser.add_argument("--v4-panel-status")
    args = parser.parse_args()
    report = summarize_classic_multistep_benchmark(
        reference_pack=Path(args.reference_pack),
        runs=_parse_bindings(args.run),
        proxies=_parse_bindings(args.proxy),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        output_html=Path(args.output_html) if args.output_html else None,
        v4_panel_status=(
            Path(args.v4_panel_status) if args.v4_panel_status else None
        ),
    )
    print(json.dumps(report["aggregates"]["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
