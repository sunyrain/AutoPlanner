"""Build a publication-readiness and completion-audit report for v4 reranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_v4_cascade_publication_readiness(
    *,
    root: Path,
    output_md: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    split_manifest = _read_json(root / "splits" / "v4_trace_split_manifest.json")
    inventory = _read_json(root / "data_inventory.json")
    model_report = _read_json(root / "reports" / "v4_cascade_product_value_report.json")
    comparison = _read_json(root / "comparison.json")
    case_studies = _read_json(root / "case_studies.json")
    checklist = [
        _item("v4 inventory", bool(inventory), root / "data_inventory.json"),
        _item("v4 split manifest", bool(split_manifest), root / "splits" / "v4_trace_split_manifest.json"),
        _item("no split leakage", _no_leakage(split_manifest), root / "splits" / "v4_trace_split_manifest.json"),
        _item("feature packs", (root / "packs" / "cascade_v4_route_feature_pack_train.jsonl").exists(), root / "packs" / "cascade_v4_route_feature_pack_train.jsonl"),
        _item("preference pack", (root / "packs" / "cascade_v4_preference_train.jsonl").exists(), root / "packs" / "cascade_v4_preference_train.jsonl"),
        _item("trained checkpoint", (root / "models" / "v4_cascade_product_value.pt").exists(), root / "models" / "v4_cascade_product_value.pt"),
        _item("model validation report", bool(model_report), root / "reports" / "v4_cascade_product_value_report.json"),
        _item("native vs learned comparison", bool(comparison), root / "comparison.json"),
        _item("case studies", len(case_studies.get("cases") or []) >= 10, root / "case_studies.json"),
        _item("rule-post baseline", any((root / "rerank").glob("*rule_post_report.json")), root / "rerank"),
        _item("top-k ablation", all((root / "rerank" / f"full100_v4_rerank_top{k}_report.json").exists() for k in (5, 10, 20, 50)), root / "rerank"),
        _item("external smoke reports", all((root / "rerank" / name).exists() for name in [
            "paroutes_n1_v4_rerank_report.json",
            "paroutes_n5_v4_rerank_report.json",
            "uspto190_cached150_v4_rerank_report.json",
        ]), root / "rerank"),
    ]
    promotion = comparison.get("promotion_readiness") or {}
    final_verdict = {
        "implementation_complete": all(row["ok"] for row in checklist),
        "ready_for_publication_claim": bool(promotion.get("ready_for_promotion")),
        "recommended_claim": (
            "engineering pipeline complete; model not ready for publication-performance claim"
            if not promotion.get("ready_for_promotion")
            else "candidate for publication-performance claim pending chemist review"
        ),
        "blocking_reason": promotion.get("interpretation"),
    }
    result = {
        "schema_version": "v4_cascade_publication_readiness.v1",
        "root": str(root),
        "checklist": checklist,
        "model_metrics": (model_report.get("final_metrics") or {}),
        "comparison_promotion_readiness": promotion,
        "final_verdict": final_verdict,
    }
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(result), encoding="utf-8")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _item(name: str, ok: bool, evidence: Path) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "evidence": str(evidence)}


def _no_leakage(manifest: dict[str, Any]) -> bool:
    checks = manifest.get("leakage_checks") or {}
    return all(int((checks.get(key) or {}).get("count") or 0) == 0 for key in ["doi_cross_split", "target_cross_split", "scaffold_cross_split"])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _markdown(result: dict[str, Any]) -> str:
    verdict = result.get("final_verdict") or {}
    lines = [
        "# v4 Cascade Reranker Publication Readiness",
        "",
        "## Verdict",
        "",
        f"- Implementation complete: `{verdict.get('implementation_complete')}`",
        f"- Ready for publication-performance claim: `{verdict.get('ready_for_publication_claim')}`",
        f"- Recommended claim: {verdict.get('recommended_claim')}",
        f"- Blocking reason: {verdict.get('blocking_reason')}",
        "",
        "## Checklist",
        "",
        "| item | ok | evidence |",
        "| --- | ---: | --- |",
    ]
    for row in result.get("checklist") or []:
        lines.append(f"| {row.get('name')} | `{row.get('ok')}` | `{row.get('evidence')}` |")
    lines.extend(
        [
            "",
            "## Model Metrics",
            "",
            "```json",
            json.dumps(result.get("model_metrics") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Promotion Readiness",
            "",
            "```json",
            json.dumps(result.get("comparison_promotion_readiness") or {}, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v4 cascade reranker publication readiness report")
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json")
    args = ap.parse_args()
    result = build_v4_cascade_publication_readiness(
        root=Path(args.root),
        output_md=Path(args.output_md),
        output_json=Path(args.output_json) if args.output_json else None,
    )
    print(json.dumps(result["final_verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
