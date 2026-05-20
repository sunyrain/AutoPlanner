"""Build compact case studies for native/rule/learned route reranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit


def build_v4_cascade_case_studies(
    *,
    native_pool: Path,
    learned_pool: Path,
    output_md: Path,
    output_json: Path | None = None,
    benchmark: Path | None = None,
    label: str = "benchmark",
    limit: int = 10,
) -> dict[str, Any]:
    native = json.loads(Path(native_pool).read_text(encoding="utf-8"))
    learned = json.loads(Path(learned_pool).read_text(encoding="utf-8"))
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    native_audit = build_product_route_feasibility_audit(native, benchmark_rows=benchmark_rows)
    learned_audit = build_product_route_feasibility_audit(learned, benchmark_rows=benchmark_rows)
    cases = []
    for idx, (native_target, learned_target) in enumerate(zip(native_audit.get("targets") or [], learned_audit.get("targets") or [])):
        native_ranked = sorted(native_target.get("routes") or [], key=lambda row: int(row.get("rank") or 10**9))
        learned_ranked = sorted(learned_target.get("routes") or [], key=lambda row: int(row.get("rank") or 10**9))
        if not native_ranked or not learned_ranked:
            continue
        native_top = native_ranked[0]
        learned_top = learned_ranked[0]
        learned_route = ((learned.get("targets") or [])[idx].get("routes") or [{}])[0]
        native_rank = learned_route.get("native_rank")
        movement = int(native_rank) if isinstance(native_rank, int) else None
        case_type = _case_type(native_top, learned_top, movement)
        if case_type is None:
            continue
        cases.append(
            {
                "case_type": case_type,
                "benchmark": label,
                "target_index": idx,
                "target_smiles": native_target.get("target_smiles"),
                "native_top_class": native_top.get("route_class"),
                "learned_top_class": learned_top.get("route_class"),
                "learned_top_native_rank": native_rank,
                "learned_v4_value": learned_route.get("v4_cascade_product_value"),
                "native_top_issues": native_top.get("issues") or [],
                "learned_top_issues": learned_top.get("issues") or [],
                "learned_top_first_rxn": _first_rxn(learned_route),
            }
        )
    cases = sorted(cases, key=lambda row: (_case_priority(row), row.get("target_index") or 0))[:limit]
    result = {
        "schema_version": "v4_cascade_case_studies.v1",
        "metadata": {
            "native_pool": str(native_pool),
            "learned_pool": str(learned_pool),
            "benchmark": str(benchmark) if benchmark else None,
            "label": label,
            "limit": limit,
        },
        "cases": cases,
    }
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _case_type(native_top: dict[str, Any], learned_top: dict[str, Any], movement: int | None) -> str | None:
    native_class = native_top.get("route_class")
    learned_class = learned_top.get("route_class")
    if native_class == "reject_artifact" and learned_class != "reject_artifact":
        return "native_top_artifact_replaced"
    if learned_class in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"} and movement and movement >= 3:
        return "learned_promoted_lower_native_rank_triage"
    if native_class != learned_class:
        return "top_route_class_changed"
    if movement and movement >= 10:
        return "large_rank_movement_same_class"
    return None


def _case_priority(row: dict[str, Any]) -> int:
    order = {
        "native_top_artifact_replaced": 0,
        "learned_promoted_lower_native_rank_triage": 1,
        "top_route_class_changed": 2,
        "large_rank_movement_same_class": 3,
    }
    return order.get(str(row.get("case_type")), 99)


def _first_rxn(route: dict[str, Any]) -> str:
    steps = route.get("steps") or []
    if not steps:
        return ""
    return str(steps[0].get("rxn_smiles") or "")[:240]


def _read_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    return None


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# v4 Cascade Rerank Case Studies",
        "",
        f"- Label: `{result['metadata']['label']}`",
        f"- Cases: `{len(result.get('cases') or [])}`",
        "",
        "| type | target | native class | learned class | learned native rank | v4 value | native issues | learned issues | first learned rxn |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for case in result.get("cases") or []:
        lines.append(
            "| {case_type} | {target} | `{native}` | `{learned}` | {rank} | {value} | {native_issues} | {learned_issues} | `{rxn}` |".format(
                case_type=case.get("case_type"),
                target=case.get("target_index"),
                native=case.get("native_top_class"),
                learned=case.get("learned_top_class"),
                rank=case.get("learned_top_native_rank"),
                value=_fmt(case.get("learned_v4_value")),
                native_issues=", ".join(case.get("native_top_issues") or []),
                learned_issues=", ".join(case.get("learned_top_issues") or []),
                rxn=(case.get("learned_top_first_rxn") or "").replace("|", " "),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v4 cascade rerank case studies")
    ap.add_argument("--native-pool", required=True)
    ap.add_argument("--learned-pool", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json")
    ap.add_argument("--benchmark")
    ap.add_argument("--label", default="benchmark")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    result = build_v4_cascade_case_studies(
        native_pool=Path(args.native_pool),
        learned_pool=Path(args.learned_pool),
        output_md=Path(args.output_md),
        output_json=Path(args.output_json) if args.output_json else None,
        benchmark=Path(args.benchmark) if args.benchmark else None,
        label=args.label,
        limit=args.limit,
    )
    print(json.dumps({"cases": len(result["cases"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
