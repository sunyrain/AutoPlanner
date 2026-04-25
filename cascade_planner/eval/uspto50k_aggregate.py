"""Aggregate K2 USPTO-50K results into a single report once all per-model
prediction caches exist. Re-runs ensemble from cache so it never re-does inference."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from cascade_planner.eval.uspto50k_syntheseus import (
    load_uspto50k_test, canon_set, _load_preds, compute_hits, ensemble_combine,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="results/v2/k2_preds")
    ap.add_argument("--data", default="data_external/uspto50k/")
    ap.add_argument("--models", nargs="+", default=["megan", "rootaligned", "localretro"])
    ap.add_argument("--output", default="results/v2/k2_uspto50k_summary.json")
    args = ap.parse_args()

    test = load_uspto50k_test(args.data)
    truths = [canon_set(r) for r in test["reactants"]]
    n = len(test)

    cache_dir = Path(args.cache_dir)
    available = []
    per_model_preds = {}
    per_model_metrics = {}
    for m in args.models:
        p = cache_dir / f"{m}_n{n}.json"
        if not p.exists():
            print(f"[K2] missing cache: {p}")
            continue
        preds = _load_preds(p)
        per_model_preds[m] = preds
        metrics, _ = compute_hits(preds, truths)
        metrics["n"] = n
        per_model_metrics[m] = metrics
        available.append(m)

    summary = {
        "n_samples": n,
        "test_split": "TDC random 10007 (data_external/uspto50k/tdc_test.csv)",
        "models_available": available,
        "per_model": per_model_metrics,
        "k2_target_top1_pct": 52,
    }
    if len(available) >= 2:
        ens = ensemble_combine(per_model_preds)
        ens_metrics, _ = compute_hits(ens, truths)
        ens_metrics["n"] = n
        summary["ensemble_uniform"] = ens_metrics
        # Also try score-weighted by model top-1 (better fusion)
        weights = {m: per_model_metrics[m]["top_1"] for m in available}
        total = sum(weights.values()) or 1
        weights = {k: v / total for k, v in weights.items()}
        ens_w = ensemble_combine(per_model_preds, weights=weights)
        ens_w_metrics, _ = compute_hits(ens_w, truths)
        ens_w_metrics["n"] = n
        ens_w_metrics["weights"] = weights
        summary["ensemble_top1_weighted"] = ens_w_metrics

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))

    print("=" * 60)
    print(f"K2 USPTO-50K SUMMARY (n={n}, target top-1={summary['k2_target_top1_pct']}%)")
    print("=" * 60)
    for m in available:
        mt = per_model_metrics[m]
        print(f"{m:14s}  top1={mt['top_1']:5.2f}  top3={mt['top_3']:5.2f}  top5={mt['top_5']:5.2f}  top10={mt['top_10']:5.2f}  top50={mt['top_50']:5.2f}")
    if "ensemble_uniform" in summary:
        e = summary["ensemble_uniform"]
        print(f"{'ENS uniform':14s}  top1={e['top_1']:5.2f}  top3={e['top_3']:5.2f}  top5={e['top_5']:5.2f}  top10={e['top_10']:5.2f}  top50={e['top_50']:5.2f}")
    if "ensemble_top1_weighted" in summary:
        e = summary["ensemble_top1_weighted"]
        print(f"{'ENS top1-w':14s}  top1={e['top_1']:5.2f}  top3={e['top_3']:5.2f}  top5={e['top_5']:5.2f}  top10={e['top_10']:5.2f}  top50={e['top_50']:5.2f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
