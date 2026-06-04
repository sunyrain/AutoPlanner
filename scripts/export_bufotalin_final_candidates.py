"""Export conservative final candidates from a bufotalin long-run directory."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.route_plausibility import (
    has_unsupported_biosynthetic_prenyl_terminal,
    route_condition_risk_warnings,
)
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from scripts.run_bufotalin_12h_iteration import BUFOTALIN_TARGET
from scripts.summarize_bufotalin_iteration import summarize_iteration_root


HIGH_RISK_WARNINGS = {
    "non_mild_predicted_temperature",
    "strong_hydride_reagent_predicted",
    "unsupported_biosynthetic_prenyl_terminal",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export conservative bufotalin final route candidates.")
    parser.add_argument("root", help="12h iteration root directory")
    parser.add_argument("--output-dir", default="", help="Default: <root>/final_candidates")
    parser.add_argument("--top-native", type=int, default=5)
    parser.add_argument("--top-stitched", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    report = export_final_candidates(
        Path(args.root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        top_native=args.top_native,
        top_stitched=args.top_stitched,
        render=not args.no_render,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def export_final_candidates(
    root: Path,
    *,
    output_dir: Path | None = None,
    top_native: int = 5,
    top_stitched: int = 5,
    render: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    output_dir = output_dir or root / "final_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_records = _load_payload_records(root)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for payload_path, payload in payload_records:
        target = str(payload.get("target") or payload.get("target_smiles") or BUFOTALIN_TARGET)
        for index, route in enumerate(payload.get("routes") or [], start=1):
            if not isinstance(route, dict):
                continue
            classification = classify_route(route, target_smiles=target)
            row = {
                "source_payload": str(payload_path),
                "source_cycle": payload_path.parent.name,
                "source_route_index": index,
                "classification": classification,
                "route": route,
            }
            if classification["reportable"]:
                candidates.append(row)
            else:
                excluded.append(
                    {
                        "source_payload": str(payload_path),
                        "source_cycle": payload_path.parent.name,
                        "source_route_index": index,
                        "confidence_tier": classification["confidence_tier"],
                        "exclusion_reasons": classification["exclusion_reasons"],
                    }
                )

    deduped = _dedupe_candidates(candidates)
    high_confidence = [
        row for row in deduped if row["classification"]["confidence_tier"] == "high_confidence_source_supported"
    ]
    stitched_review = sorted(
        [
            row
            for row in deduped
            if row["classification"]["confidence_tier"] == "stitched_semisynthesis_upstream_review_only"
        ],
        key=_stitched_candidate_key,
        reverse=True,
    )
    stitched_review = _select_diverse_candidates(stitched_review, limit=max(0, int(top_stitched)))
    native_review = _select_diverse_candidates(
        [
            row
            for row in deduped
            if row["classification"]["confidence_tier"] == "native_model_candidate_review_only"
        ],
        limit=max(0, int(top_native)),
    )
    selected = [*high_confidence, *stitched_review, *native_review]
    selected_routes = [_route_with_final_metadata(row) for row in selected]
    final_payload = {
        "schema_version": "bufotalin_final_candidates.v1",
        "target_smiles": BUFOTALIN_TARGET,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "summary": summarize_iteration_root(root),
        "selection_policy": {
            "high_confidence": "source-supported semisynthesis with complete conditions",
            "stitched_semisynthesis_upstream_review_only": (
                "source-supported late-stage semisynthesis anchor plus native ChemEnzy upstream search; "
                "use only as a review target, not as a validated total synthesis"
            ),
            "native_review_only": "native ChemEnzy route with verifier pass and complete RCR conditions; not treated as literature-supported",
            "excluded": "circular/target-terminal, incomplete conditions, verifier failure, or unsupported route class",
        },
        "n_results": len(selected_routes),
        "routes": selected_routes,
        "excluded_route_count": len(excluded),
    }
    final_payload_path = output_dir / "final_candidates_payload.json"
    final_json_path = output_dir / "final_candidates.json"
    final_payload_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    final_json_path.write_text(
        json.dumps(
            {
                "source_root": str(root),
                "generated_at": final_payload["generated_at"],
                "high_confidence_count": len(high_confidence),
                "stitched_review_only_count": len(stitched_review),
                "native_review_only_count": len(native_review),
                "selected_count": len(selected),
                "excluded_route_count": len(excluded),
                "selected": [_candidate_summary(row) for row in selected],
                "excluded_sample": excluded[:50],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path = output_dir / "final_candidates.md"
    md_path.write_text(
        _render_markdown(root, high_confidence, stitched_review, native_review, excluded),
        encoding="utf-8",
    )
    figures: dict[str, Any] = {"enabled": False}
    if render and selected_routes:
        figures = _render_figures(final_payload_path, output_dir / "figures", top_k=len(selected_routes))
    return {
        "output_dir": str(output_dir),
        "final_candidates_json": str(final_json_path),
        "final_candidates_md": str(md_path),
        "final_candidates_payload": str(final_payload_path),
        "high_confidence_count": len(high_confidence),
        "stitched_review_only_count": len(stitched_review),
        "native_review_only_count": len(native_review),
        "selected_count": len(selected),
        "excluded_route_count": len(excluded),
        "figures": figures,
    }


def classify_route(route: dict[str, Any], *, target_smiles: str = BUFOTALIN_TARGET) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    raw = route.get("raw_backend_metadata") or {}
    source_supported = bool(metrics.get("source_supported_semisynthesis"))
    stitched_semisynthesis = bool(metrics.get("stitched_semisynthesis")) or (
        str(raw.get("route_class_hint") or "") == "stitched_semisynthesis_upstream"
    )
    native = bool(metrics.get("native_returned_route"))
    verifier = metrics.get("cascade_verifier") or {}
    verifier_feasible = bool(verifier.get("feasible"))
    condition_coverage = _condition_coverage(route)
    target_terminal = _terminal_contains_target(metrics, target_smiles)
    source_record = raw.get("advanced_precursor_record") or _step_source_record(route)
    warnings = _route_warnings(route, target_terminal=target_terminal)
    exclusion_reasons: list[str] = []
    confidence_tier = "excluded"
    reportable = False
    presentation_ready = False

    if target_terminal:
        exclusion_reasons.append("terminal_reactants_include_target")
    if condition_coverage < 1.0:
        exclusion_reasons.append("incomplete_condition_coverage")
    if not source_supported and _route_n_steps(route) < 3:
        exclusion_reasons.append("review_route_too_short")
    if not source_supported:
        for warning in sorted(set(warnings) & HIGH_RISK_WARNINGS):
            if warning not in exclusion_reasons:
                exclusion_reasons.append(warning)

    if source_supported:
        if not source_record:
            exclusion_reasons.append("missing_advanced_precursor_source_record")
        if not exclusion_reasons:
            confidence_tier = "high_confidence_source_supported"
            reportable = True
            presentation_ready = True
    elif stitched_semisynthesis:
        if not verifier_feasible:
            exclusion_reasons.append("cascade_verifier_not_feasible")
        if not exclusion_reasons:
            confidence_tier = "stitched_semisynthesis_upstream_review_only"
            reportable = True
    elif native:
        if not verifier_feasible:
            exclusion_reasons.append("cascade_verifier_not_feasible")
        if not exclusion_reasons:
            confidence_tier = "native_model_candidate_review_only"
            reportable = True
    else:
        exclusion_reasons.append("unsupported_route_class")

    return {
        "confidence_tier": confidence_tier,
        "reportable": reportable,
        "presentation_ready": presentation_ready,
        "source_supported_semisynthesis": source_supported,
        "stitched_semisynthesis_upstream": stitched_semisynthesis,
        "native_model_candidate": native,
        "cascade_verifier_feasible": verifier_feasible,
        "condition_coverage": condition_coverage,
        "target_terminal": target_terminal,
        "source_record": source_record,
        "warnings": warnings,
        "exclusion_reasons": exclusion_reasons,
    }


def _load_payload_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*/web_payload.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append((path, payload))
    return records


def _condition_coverage(route: dict[str, Any]) -> float:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    if not steps:
        return 0.0
    covered = sum(1 for step in steps if step.get("condition_predictions"))
    return covered / max(1, len(steps))


def _terminal_contains_target(metrics: dict[str, Any], target_smiles: str) -> bool:
    target = canonical_smiles(target_smiles)
    if not target:
        return False
    for reactant in metrics.get("terminal_reactants") or []:
        if canonical_smiles(str(reactant or "")) == target:
            return True
    return False


def _step_source_record(route: dict[str, Any]) -> dict[str, Any]:
    for step in route.get("steps") or []:
        metadata = ((step or {}).get("raw_backend_metadata") or {}).get("semisynthesis_rescue") or {}
        record = metadata.get("precursor_source_record") or {}
        if record:
            return dict(record)
    return {}


def _route_warnings(route: dict[str, Any], *, target_terminal: bool) -> list[str]:
    warnings: list[str] = []
    if target_terminal:
        warnings.append("circular_or_target_as_terminal_stock")
    if has_unsupported_biosynthetic_prenyl_terminal(route):
        warnings.append("unsupported_biosynthetic_prenyl_terminal")
    warnings.extend(route_condition_risk_warnings(route))
    for step in route.get("steps") or []:
        for condition in (step or {}).get("condition_predictions") or []:
            label = str(condition.get("condition_label") or "")
            if "RCR model prediction" in label and "rcr_condition_prediction_only" not in warnings:
                warnings.append("rcr_condition_prediction_only")
            score = condition.get("Score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = None
            if score_value is not None and score_value < 0.1:
                warnings.append("low_condition_prediction_score")
    return sorted(set(warnings))


def _route_n_steps(route: dict[str, Any]) -> int:
    return int(route.get("n_steps") or len(route.get("steps") or []))


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_signature: dict[str, dict[str, Any]] = {}
    for row in candidates:
        signature = _route_signature(row["route"])
        old = best_by_signature.get(signature)
        if old is None or _candidate_key(row) > _candidate_key(old):
            best_by_signature[signature] = row
    return sorted(best_by_signature.values(), key=_candidate_key, reverse=True)


def _route_signature(route: dict[str, Any]) -> str:
    return "|".join(str(step.get("reaction_smiles") or "") for step in route.get("steps") or [])


def _candidate_key(row: dict[str, Any]) -> tuple[int, int, int, float, int]:
    classification = row["classification"]
    route = row["route"]
    tier_score = {
        "high_confidence_source_supported": 3,
        "stitched_semisynthesis_upstream_review_only": 2,
        "native_model_candidate_review_only": 1,
    }.get(str(classification["confidence_tier"]), 0)
    return (
        tier_score,
        1 if classification.get("cascade_verifier_feasible") else 0,
        1 if classification.get("condition_coverage") >= 1.0 else 0,
        *_warning_quality(classification),
        float(route.get("score") or 0.0),
    )


def _stitched_candidate_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
    classification = row["classification"]
    route = row["route"]
    n_steps = int(route.get("n_steps") or len(route.get("steps") or []))
    return (
        1 if classification.get("cascade_verifier_feasible") else 0,
        1 if classification.get("condition_coverage") >= 1.0 else 0,
        *_warning_quality(classification),
        n_steps,
        float(route.get("score") or 0.0),
    )


def _warning_quality(classification: dict[str, Any]) -> tuple[int, int]:
    warnings = {str(item) for item in classification.get("warnings") or []}
    high_risk = len(warnings & HIGH_RISK_WARNINGS)
    weighted = high_risk * 5 + len(warnings - HIGH_RISK_WARNINGS)
    return (-weighted, -high_risk)


def _select_diverse_candidates(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    seen_signatures: set[str] = set()
    preferred = [row for row in rows if _route_step_count(row) >= 3]
    fallback = [row for row in rows if _route_step_count(row) < 3]

    def take(candidates: list[dict[str, Any]], *, allow_duplicate_signature: bool = False) -> bool:
        for row in candidates:
            if id(row) in selected_ids:
                continue
            signature = _review_diversity_signature(row)
            if not allow_duplicate_signature and signature in seen_signatures:
                continue
            selected.append(row)
            selected_ids.add(id(row))
            seen_signatures.add(signature)
            if len(selected) >= limit:
                return True
        return False

    if take(preferred):
        return selected
    if take(fallback):
        return selected
    take(rows, allow_duplicate_signature=True)
    return selected


def _route_step_count(row: dict[str, Any]) -> int:
    route = row.get("route") or {}
    return _route_n_steps(route)


def _review_diversity_signature(row: dict[str, Any]) -> str:
    route = row.get("route") or {}
    classification = row.get("classification") or {}
    try:
        score = f"{float(route.get('score') or 0.0):.6g}"
    except (TypeError, ValueError):
        score = "0"
    return "|".join(
        [
            str(classification.get("confidence_tier") or ""),
            str(route.get("n_steps") or len(route.get("steps") or [])),
            score,
            str(row.get("source_route_index")),
        ]
    )


def _route_with_final_metadata(row: dict[str, Any]) -> dict[str, Any]:
    route = copy.deepcopy(row["route"])
    route["final_candidate"] = {
        "source_payload": row["source_payload"],
        "source_cycle": row["source_cycle"],
        "source_route_index": row["source_route_index"],
        **row["classification"],
    }
    return route


def _candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    route = row["route"]
    classification = row["classification"]
    return {
        "source_cycle": row["source_cycle"],
        "source_route_index": row["source_route_index"],
        "confidence_tier": classification["confidence_tier"],
        "presentation_ready": classification["presentation_ready"],
        "n_steps": route.get("n_steps"),
        "score": route.get("score"),
        "warnings": classification["warnings"],
        "source_record": classification["source_record"],
        "route_class_hint": (route.get("raw_backend_metadata") or {}).get("route_class_hint"),
    }


def _render_markdown(
    root: Path,
    high_confidence: list[dict[str, Any]],
    stitched_review: list[dict[str, Any]],
    native_review: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> str:
    lines = [
        "# Bufotalin 逆合成最终候选导出",
        "",
        f"- 来源目录：`{root}`",
        f"- 高置信可汇报路线：{len(high_confidence)} 条",
        f"- 半合成锚点 + 上游拆解候选，仅供复核：{len(stitched_review)} 条",
        f"- native 模型候选，仅供复核：{len(native_review)} 条",
        f"- 被排除路线：{len(excluded)} 条",
        "",
        "## 结论",
        "",
        "当前可作为专家汇报主结论的是 source-supported semisynthesis：以 Deacetylbufotalin / Bufogenin B 为高级前体，用乙酸酐/DMAP 在二氯甲烷中进行末端 O-乙酰化得到目标分子。半合成锚点 + 上游拆解路线和 native ChemEnzy 路线都只作为模型候选，不按文献支撑路线表述。",
        "",
        "## 高置信路线",
        "",
    ]
    if not high_confidence:
        lines.append("暂无高置信路线。")
    for idx, row in enumerate(high_confidence, start=1):
        route = row["route"]
        classification = row["classification"]
        record = classification.get("source_record") or {}
        lines.extend(
            [
                f"### Route H{idx}",
                "",
                f"- cycle：`{row['source_cycle']}`，原 route index：{row['source_route_index']}",
                f"- 步数：{route.get('n_steps')}",
                f"- 高级前体：{record.get('name', 'unknown')}；CAS：{record.get('cas', 'unknown')}",
                f"- 条件覆盖：{classification.get('condition_coverage'):.2f}",
                f"- 说明：{record.get('source_note', '')}",
            ]
        )
        for step_index, step in enumerate(route.get("steps") or [], start=1):
            cond = ((step or {}).get("condition_predictions") or [{}])[0]
            lines.append(
                "- Step "
                f"{step_index}: `{step.get('main_reactant', '')}` + "
                f"{', '.join(step.get('aux_reactants') or [])} -> product；"
                f"condition={cond.get('condition_label') or cond}"
            )
        refs = record.get("references") or []
        if refs:
            lines.append("- 来源链接：" + "；".join(str(ref) for ref in refs))
        lines.append("")
    lines.extend(["## 半合成锚点 + 上游拆解候选", ""])
    if not stitched_review:
        lines.append("暂无建议展示的上游拆解候选。")
    for idx, row in enumerate(stitched_review, start=1):
        route = row["route"]
        classification = row["classification"]
        lines.extend(
            [
                f"### Route S{idx}",
                "",
                f"- cycle：`{row['source_cycle']}`，原 route index：{row['source_route_index']}",
                f"- 步数：{route.get('n_steps')}；score：{route.get('score')}",
                f"- 条件覆盖：{classification.get('condition_coverage'):.2f}",
                f"- 警告：{', '.join(classification.get('warnings') or ['none'])}",
                "- 定位：已知半合成末端步骤加模型上游拆解，适合作为专家复核入口，不作为当前高置信全合成结论。",
                "",
            ]
        )
    lines.extend(["## native 模型候选", ""])
    if not native_review:
        lines.append("暂无建议展示的 native 模型候选。")
    for idx, row in enumerate(native_review, start=1):
        route = row["route"]
        classification = row["classification"]
        lines.extend(
            [
                f"### Route N{idx}",
                "",
                f"- cycle：`{row['source_cycle']}`，原 route index：{row['source_route_index']}",
                f"- 步数：{route.get('n_steps')}；score：{route.get('score')}",
                f"- 警告：{', '.join(classification.get('warnings') or ['none'])}",
                "- 定位：模型候选，仅可作为化学家复核入口，不作为当前高置信逆合成结论。",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_figures(payload_path: Path, output_dir: Path, *, top_k: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("scheme_route_*.svg", "scheme_route_*.pdf"):
        for stale in output_dir.glob(pattern):
            stale.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "scripts/render_linear_route_schemes.py",
        "--input",
        str(payload_path),
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(max(1, int(top_k))),
        "--formats",
        "svg,pdf",
        "--steps-per-row",
        "3",
        "--aux-mode",
        "mini",
        "--only-feasible",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    manifest = output_dir / "manifest.json"
    figures = []
    if manifest.exists():
        try:
            figures = json.loads(manifest.read_text(encoding="utf-8")).get("figures") or []
        except Exception:
            figures = []
    return {
        "enabled": True,
        "returncode": completed.returncode,
        "output_dir": str(output_dir),
        "figure_count": len(figures),
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


if __name__ == "__main__":
    main()
