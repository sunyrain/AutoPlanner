"""Annotate real candidate caches with dual-tower enzyme retrieval."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.expand.dual_tower import DualTowerRetriever


def annotate_candidate_cache(
    *,
    input_cache: str,
    output_cache: str,
    checkpoint: str,
    esm_cache: str,
    topk: int = 5,
    min_score: float = -1.0,
    device: str | None = None,
) -> dict[str, Any]:
    """Add enzyme_uid/ec suggestions to each reaction candidate in a cache."""
    kwargs = {"device": device} if device else {}
    retriever = DualTowerRetriever(checkpoint, esm_cache, **kwargs)
    cache = json.loads(Path(input_cache).read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}

    n_candidates = 0
    n_annotated = 0
    n_with_ec = 0
    top_ec = Counter()
    t0 = time.time()

    for product, rows in cache.items():
        new_rows = []
        for row in rows or []:
            n_candidates += 1
            copied = dict(row)
            rxn = copied.get("reaction_smiles") or copied.get("rxn_smiles") or ""
            suggestions = retriever.rank(rxn, topk=topk)
            suggestions = [s for s in suggestions if float(s.get("score", -1.0)) >= min_score]
            if suggestions:
                best = suggestions[0]
                n_annotated += 1
                copied["enzyme_suggestions"] = suggestions
                copied["dual_tower_score"] = float(best["score"])
                copied["e_enzyme"] = max(0.0, min(1.0, (float(best["score"]) + 1.0) / 2.0))
                copied["enzyme_source"] = "dual_tower"
                if best.get("uniprot_id") and not copied.get("enzyme_uid"):
                    copied["enzyme_uid"] = best["uniprot_id"]
                if best.get("ec_number") and not copied.get("ec"):
                    copied["ec"] = best["ec_number"]
                if copied.get("ec"):
                    n_with_ec += 1
                    top_ec[str(copied["ec"]).split(".")[0]] += 1
            new_rows.append(copied)
        out[product] = new_rows

    output = Path(output_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "input_cache": input_cache,
        "output_cache": output_cache,
        "checkpoint": checkpoint,
        "esm_cache": esm_cache,
        "topk": topk,
        "min_score": min_score,
        "n_products": len(out),
        "n_candidates": n_candidates,
        "n_annotated": n_annotated,
        "n_with_ec": n_with_ec,
        "top_ec1_counts": dict(top_ec),
        "elapsed_s": round(time.time() - t0, 3),
    }
    Path(str(output) + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-cache", default="results/shared/enzexpand_candidates_100.json")
    ap.add_argument("--output-cache", default="results/shared/enzexpand_dualtower_candidates_100.json")
    ap.add_argument("--checkpoint", default="results/shared/dual_tower_v2.pt")
    ap.add_argument("--esm-cache", default="results/shared/esm_cache")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=-1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    summary = annotate_candidate_cache(
        input_cache=args.input_cache,
        output_cache=args.output_cache,
        checkpoint=args.checkpoint,
        esm_cache=args.esm_cache,
        topk=args.topk,
        min_score=args.min_score,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
