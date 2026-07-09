"""Validate Stage-2 fragment scorer as an offline route reranker."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cascade_planner.eval.train_cascade_fragment_scorer import (
    FragmentScorerNetwork,
    fragment_feature_vector,
)
from cascade_planner.cascade_search.pair_scorer import pair_rule_features
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


def validate_fragment_rerank(
    *,
    result_path: Path,
    model_path: Path,
    output_path: Path,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    weights = weights or [0.0, 0.1, 0.25, 0.5, 1.0]
    scorer = _LoadedFragmentScorer(model_path)
    target_reports = []
    route_rows = []
    fragment_counts = Counter()

    for target in payload.get("targets") or []:
        programs = list(((target.get("cascade_search") or {}).get("result_programs") or []))
        scored_programs = []
        for program in programs:
            fragment_score, details = scorer.score_route(
                program.get("route_rxns") or [],
                route_domain=target.get("route_domain") or "unknown",
            )
            fragment_counts["programs"] += 1
            fragment_counts["programs_with_fragments"] += int(bool(details.get("fragment_scores")))
            fragment_counts["two_step_fragments"] += int(details.get("two_step_fragments") or 0)
            fragment_counts["three_step_fragments"] += int(details.get("three_step_fragments") or 0)
            row = dict(program)
            row["fragment_score"] = fragment_score
            row["fragment_details"] = details
            scored_programs.append(row)
            route_rows.append(
                {
                    "target_smiles": target.get("target_smiles"),
                    "route_domain": target.get("route_domain"),
                    "base_rank": program.get("rank"),
                    "base_score": program.get("score"),
                    "fragment_score": fragment_score,
                    "route_outcome_value": program.get("route_outcome_value"),
                    "partial_gt_step_overlap": bool(program.get("partial_gt_step_overlap")),
                    "gt_reactant_in_route": bool(program.get("gt_reactant_in_route")),
                    "exact_reaction_hit_count": int(program.get("exact_reaction_hit_count") or 0),
                    "gt_reactant_hit_count": int(program.get("gt_reactant_hit_count") or 0),
                    "solved": bool(program.get("solved")),
                }
            )
        by_weight = {}
        for weight in weights:
            ranked = sorted(
                scored_programs,
                key=lambda program: float(program.get("score") or 0.0) + float(weight) * float(program.get("fragment_score") or 0.0),
                reverse=True,
            )
            best = ranked[0] if ranked else {}
            by_weight[str(weight)] = _program_summary(best)
        target_reports.append(
            {
                "target_smiles": target.get("target_smiles"),
                "route_domain": target.get("route_domain"),
                "n_programs": len(scored_programs),
                "by_weight": by_weight,
                "programs": scored_programs,
            }
        )

    sweep = {}
    for weight in weights:
        key = str(weight)
        top_rows = [row["by_weight"][key] for row in target_reports if row.get("by_weight", {}).get(key)]
        sweep[key] = _aggregate(top_rows)
    report = {
        "metadata": {
            "result_path": str(result_path),
            "model_path": str(model_path),
            "output_path": str(output_path),
            "weights": weights,
            "validation_contract": "offline_result_program_fragment_rerank.v1",
            "caution": (
                "This is an offline rerank of existing result_programs. It does not prove search-time improvement, "
                "and route_rxns usually lack conditions/catalysts, so fragment scores rely mostly on structure/order."
            ),
        },
        "input_summary": payload.get("summary") or {},
        "fragment_coverage": dict(fragment_counts),
        "route_score_correlations": _correlations(route_rows),
        "sweep": sweep,
        "targets": target_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


class _LoadedFragmentScorer:
    def __init__(self, model_path: Path):
        checkpoint = torch.load(str(model_path), map_location="cpu")
        self.schema = dict(checkpoint["feature_schema"])
        self.label_names = list(checkpoint.get("label_names") or [])
        hidden = int(checkpoint.get("hidden") or 160)
        self.model = FragmentScorerNetwork(
            int(self.schema["feature_dim"]),
            hidden=hidden,
            output_dim=len(self.label_names),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def score_route(self, route_rxns: list[str], *, route_domain: str) -> tuple[float, dict[str, Any]]:
        steps = [_step_from_rxn(rxn, idx) for idx, rxn in enumerate(route_rxns)]
        two = _adjacent_fragments(steps, route_domain=route_domain, window_size=2)
        three = _adjacent_fragments(steps, route_domain=route_domain, window_size=3)
        fragments = two + three
        scores = [self.score_fragment(fragment) for fragment in fragments]
        if not scores:
            return 0.0, {
                "fragment_scores": [],
                "two_step_fragments": 0,
                "three_step_fragments": 0,
            }
        score = 0.65 * max(scores) + 0.35 * float(np.mean(scores))
        return float(score), {
            "fragment_scores": [round(float(value), 6) for value in scores],
            "two_step_fragments": len(two),
            "three_step_fragments": len(three),
            "max_fragment_score": round(float(max(scores)), 6),
            "mean_fragment_score": round(float(np.mean(scores)), 6),
        }

    def score_fragment(self, row: dict[str, Any]) -> float:
        x = fragment_feature_vector(row, self.schema)
        with torch.no_grad():
            tensor = torch.tensor(x[None, :], dtype=torch.float32)
            probs = torch.sigmoid(self.model(tensor))[0].detach().cpu().numpy()
        if "fragment_preference" in self.label_names:
            return float(probs[self.label_names.index("fragment_preference")])
        return float(probs[0])


def _adjacent_fragments(steps: list[dict[str, Any]], *, route_domain: str, window_size: int) -> list[dict[str, Any]]:
    if window_size == 2:
        pairs = []
        for left in steps:
            for right in steps:
                if left is right:
                    continue
                if _shared_intermediate(left["rxn_smiles"], right["rxn_smiles"]):
                    pairs.append(_fragment_row([left, right], route_domain=route_domain))
        return pairs
    if window_size == 3:
        rows = []
        two = _adjacent_fragments(steps, route_domain=route_domain, window_size=2)
        for first in two:
            first_steps = first["steps"]
            for step in steps:
                if step in first_steps:
                    continue
                if _shared_intermediate(first_steps[-1]["rxn_smiles"], step["rxn_smiles"]):
                    rows.append(_fragment_row([*first_steps, step], route_domain=route_domain))
        return rows
    return []


def _fragment_row(steps: list[dict[str, Any]], *, route_domain: str) -> dict[str, Any]:
    pair_rows = []
    for idx, (left, right) in enumerate(zip(steps, steps[1:])):
        pair = {
            "route_domain": route_domain or "unknown",
            "left_step": left,
            "right_step": right,
            "shared_intermediate": _shared_intermediate(left.get("rxn_smiles"), right.get("rxn_smiles")),
            "left_pairwise_mode": "unknown",
            "right_pairwise_mode": "unknown",
            "pair_index": idx,
        }
        pair["rule_features"] = pair_rule_features(pair)
        pair_rows.append(pair)
    return {
        "route_domain": route_domain or "unknown",
        "window_size": len(steps),
        "steps": steps,
        "pair_rows": pair_rows,
    }


def _step_from_rxn(rxn: str, idx: int) -> dict[str, Any]:
    return {
        "step_id": f"route_step_{idx}",
        "step_index": idx,
        "rxn_smiles": rxn,
        "pairwise_mode": "unknown",
        "step_mode": "unknown",
        "transformation_name": "unknown",
        "transformation_superclass": "unknown",
        "intermediate_isolated": None,
        "step_conditions": {},
        "catalyst_components": [],
    }


def _shared_intermediate(left_rxn: Any, right_rxn: Any) -> str:
    if not left_rxn or not right_rxn or ">>" not in str(left_rxn) or ">>" not in str(right_rxn):
        return ""
    left_products = str(left_rxn).split(">>", 1)[1].split(".")
    right_reactants = str(right_rxn).split(">>", 1)[0].split(".")
    left_keys = {canonical_smiles(value) or value: value for value in left_products if value}
    for value in right_reactants:
        key = canonical_smiles(value) or value
        if key in left_keys:
            return key
    return ""


def _program_summary(program: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_rank": program.get("rank"),
        "base_score": program.get("score"),
        "fragment_score": program.get("fragment_score"),
        "solved": bool(program.get("solved")),
        "partial_gt_step_overlap": bool(program.get("partial_gt_step_overlap")),
        "gt_reactant_in_route": bool(program.get("gt_reactant_in_route")),
        "exact_gt_route_recovered": bool(program.get("exact_gt_route_recovered")),
        "route_outcome_value": float(program.get("route_outcome_value") or 0.0),
        "exact_reaction_hit_count": int(program.get("exact_reaction_hit_count") or 0),
        "gt_reactant_hit_count": int(program.get("gt_reactant_hit_count") or 0),
        "failure_count": len(program.get("failure_categories") or []),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n_targets": n,
        "solved_rate": _rate(rows, "solved"),
        "partial_gt_step_overlap_rate": _rate(rows, "partial_gt_step_overlap"),
        "gt_reactant_in_route_rate": _rate(rows, "gt_reactant_in_route"),
        "exact_gt_route_recovered_rate": _rate(rows, "exact_gt_route_recovered"),
        "avg_route_outcome_value": _mean([row.get("route_outcome_value") for row in rows]),
        "avg_exact_reaction_hit_count": _mean([row.get("exact_reaction_hit_count") for row in rows]),
        "avg_gt_reactant_hit_count": _mean([row.get("gt_reactant_hit_count") for row in rows]),
        "avg_fragment_score": _mean([row.get("fragment_score") for row in rows]),
        "avg_base_rank": _mean([row.get("base_rank") for row in rows]),
    }


def _correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    fragment = np.asarray([float(row.get("fragment_score") or 0.0) for row in rows], dtype=np.float32)
    out = {}
    for field in ("route_outcome_value", "gt_reactant_hit_count", "exact_reaction_hit_count"):
        y = np.asarray([float(row.get(field) or 0.0) for row in rows], dtype=np.float32)
        out[f"pearson_fragment_vs_{field}"] = _pearson(fragment, y)
    out["mean_fragment_score_gt_reactant_route"] = _mean(
        [row.get("fragment_score") for row in rows if row.get("gt_reactant_in_route")]
    )
    out["mean_fragment_score_non_gt_reactant_route"] = _mean(
        [row.get("fragment_score") for row in rows if not row.get("gt_reactant_in_route")]
    )
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 6)


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for row in rows if row.get(field)) / len(rows), 6) if rows else 0.0


def _mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fragment Rerank Validation",
        "",
        f"- result: `{report['metadata']['result_path']}`",
        f"- model: `{report['metadata']['model_path']}`",
        f"- caution: {report['metadata']['caution']}",
        "",
        "## Fragment Coverage",
        "",
        _table(["metric", "value"], [[k, v] for k, v in sorted((report.get("fragment_coverage") or {}).items())]),
        "",
        "## Sweep",
        "",
        _table(
            [
                "weight",
                "solved",
                "partial_gt",
                "gt_reactant",
                "route_value",
                "avg_rank",
            ],
            [
                [
                    weight,
                    metrics.get("solved_rate"),
                    metrics.get("partial_gt_step_overlap_rate"),
                    metrics.get("gt_reactant_in_route_rate"),
                    metrics.get("avg_route_outcome_value"),
                    metrics.get("avg_base_rank"),
                ]
                for weight, metrics in (report.get("sweep") or {}).items()
            ],
        ),
        "",
        "## Correlations",
        "",
        _table(["metric", "value"], [[k, v] for k, v in sorted((report.get("route_score_correlations") or {}).items())]),
    ]
    return "\n".join(lines) + "\n"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate fragment scorer as offline route reranker")
    ap.add_argument("--result", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weight", action="append", type=float)
    args = ap.parse_args()
    report = validate_fragment_rerank(
        result_path=Path(args.result),
        model_path=Path(args.model),
        output_path=Path(args.output),
        weights=args.weight,
    )
    print(json.dumps(report["sweep"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
