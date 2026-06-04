#!/usr/bin/env python3
"""Build one-step GraphFP candidate packs for a lightweight reranker."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger  # noqa: E402

from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider  # noqa: E402
from cascade_planner.cascadeboard.route_recovery import canonical_side  # noqa: E402
from cascade_planner.cascadeboard.value_function import candidate_value_features, heavy_atoms  # noqa: E402


SCHEMA_VERSION = "graphfp_reranker_pack.v1"
GRAPHFP_MODEL = "graphfp_models.USPTO-full_remapped"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = _parse_args()
    products = _read_lines(args.src, limit=args.limit, offset=args.offset)
    targets = _read_lines(args.tgt, limit=args.limit, offset=args.offset)
    if len(products) != len(targets):
        raise ValueError(f"src/tgt length mismatch: {len(products)} != {len(targets)}")

    provider = ChemEnzyOneStepProposalProvider(
        vendor_root=args.vendor_root,
        models=(GRAPHFP_MODEL,),
        expansion_topk=int(args.topk),
        gpu=int(args.gpu),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    target_rows = []
    label_counts: Counter[str] = Counter()
    candidate_count_hist: Counter[int] = Counter()
    exact_hit = 0
    any_hit = 0
    errors = []
    n_candidates = 0
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, (product, target) in enumerate(zip(products, targets)):
            absolute_idx = int(args.offset) + idx
            target_key = canonical_side(target)
            error = ""
            candidates: list[dict[str, Any]] = []
            try:
                candidates = provider.predict(product, top_k=int(args.topk))
            except Exception as exc:  # pragma: no cover - depends on vendor runtime
                error = f"{type(exc).__name__}:{exc}"
                errors.append(error)

            row_exact_rank = None
            row_any_rank = None
            for rank, candidate in enumerate(candidates, start=1):
                reactants_text = _reactants_text(candidate)
                pred_key = canonical_side(reactants_text)
                exact = bool(pred_key and pred_key == target_key)
                any_reactant = bool(set(pred_key) & set(target_key))
                if exact and row_exact_rank is None:
                    row_exact_rank = rank
                if any_reactant and row_any_rank is None:
                    row_any_rank = rank
                out = _candidate_row(
                    split=args.split,
                    idx=absolute_idx,
                    product=product,
                    target=target,
                    target_key=target_key,
                    candidate=candidate,
                    rank=rank,
                    candidate_count=len(candidates),
                    exact=exact,
                    any_reactant=any_reactant,
                )
                label_counts["exact_positive" if exact else "negative"] += 1
                n_candidates += 1
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if row_exact_rank is not None:
                exact_hit += 1
            if row_any_rank is not None:
                any_hit += 1
            candidate_count_hist[len(candidates)] += 1
            target_rows.append(
                {
                    "idx": absolute_idx,
                    "candidate_count": len(candidates),
                    "exact_rank": row_exact_rank,
                    "any_reactant_rank": row_any_rank,
                    "error": error,
                }
            )
            if args.progress_every and (idx + 1) % int(args.progress_every) == 0:
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "processed": idx + 1,
                            "candidates": n_candidates,
                            "elapsed_s": round(elapsed, 3),
                            "examples_per_s": round((idx + 1) / max(elapsed, 1e-9), 4),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    elapsed = time.monotonic() - started
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "src": str(args.src),
            "tgt": str(args.tgt),
            "split": args.split,
            "limit": args.limit,
            "offset": args.offset,
            "topk": args.topk,
            "gpu": args.gpu,
            "vendor_root": str(args.vendor_root),
            "model": GRAPHFP_MODEL,
        },
        "summary": {
            "n_examples": len(products),
            "n_candidates": n_candidates,
            "exact_hit": exact_hit,
            "exact_hit_rate": round(exact_hit / max(len(products), 1), 6),
            "any_reactant_hit": any_hit,
            "any_reactant_hit_rate": round(any_hit / max(len(products), 1), 6),
            "zero_candidate": sum(1 for row in target_rows if row["candidate_count"] == 0),
            "avg_candidate_count": round(n_candidates / max(len(products), 1), 6),
            "elapsed_s": round(elapsed, 3),
            "avg_elapsed_s_per_example": round(elapsed / max(len(products), 1), 6),
            "label_counts": dict(label_counts),
            "candidate_count_histogram": dict(candidate_count_hist.most_common()),
            "errors": len(errors),
            "error_preview": errors[:5],
        },
        "target_rows_preview": target_rows[:20],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_json.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output_jsonl": str(args.output_jsonl)}, indent=2, ensure_ascii=False))


def _candidate_row(
    *,
    split: str,
    idx: int,
    product: str,
    target: str,
    target_key: tuple[str, ...],
    candidate: dict[str, Any],
    rank: int,
    candidate_count: int,
    exact: bool,
    any_reactant: bool,
) -> dict[str, Any]:
    reactants = [str(item) for item in candidate.get("reactant_smiles") or [] if str(item or "")]
    if not reactants:
        reactants = [item for item in _reactants_text(candidate).split(".") if item]
    features = candidate_value_features(product, candidate)
    product_atoms = heavy_atoms(product)
    reactant_atoms = [heavy_atoms(smi) for smi in reactants]
    template = str(candidate.get("template") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "idx": idx,
        "group_id": f"{split}:{idx}",
        "product": product,
        "target_reactants": target,
        "target_key": list(target_key),
        "candidate": {
            "rank": rank,
            "candidate_count": candidate_count,
            "score": _safe_float(candidate.get("score")),
            "reactants": reactants,
            "reactants_text": ".".join(reactants),
            "reaction_smiles": candidate.get("reaction_smiles") or candidate.get("rxn_smiles"),
            "main_reactant": candidate.get("main_reactant"),
            "aux_reactants": candidate.get("aux_reactants") or [],
            "template": template,
            "model_full_name": candidate.get("model_full_name"),
            "source": candidate.get("source"),
            "type": candidate.get("type"),
        },
        "labels": {
            "exact": exact,
            "any_reactant": any_reactant,
        },
        "features": {
            **features,
            "rank": float(rank),
            "inverse_rank": 1.0 / max(rank, 1),
            "candidate_count": float(candidate_count),
            "num_reactants": float(len(reactants)),
            "product_heavy_atoms": float(product_atoms),
            "largest_reactant_heavy_atoms": float(max(reactant_atoms, default=0)),
            "total_reactant_heavy_atoms": float(sum(reactant_atoms)),
            "template_length": float(len(template)),
            "template_has_chiral": float("@" in template),
            "template_has_ring": float(any(ch.isdigit() for ch in template)),
        },
    }


def _reactants_text(candidate: dict[str, Any]) -> str:
    reaction = str(candidate.get("reaction_smiles") or candidate.get("rxn_smiles") or "")
    if ">>" in reaction:
        return reaction.split(">>", 1)[0]
    values = candidate.get("reactant_smiles") or []
    return ".".join(str(item) for item in values if str(item or ""))


def _read_lines(path: Path, *, limit: int | None, offset: int) -> list[str]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < int(offset):
                continue
            text = line.strip()
            if text:
                rows.append(text.replace(" ", ""))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# GraphFP Reranker Pack",
        "",
        f"generated_at: `{report['generated_at']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "n_examples",
        "n_candidates",
        "exact_hit",
        "exact_hit_rate",
        "any_reactant_hit",
        "any_reactant_hit_rate",
        "zero_candidate",
        "avg_candidate_count",
        "elapsed_s",
        "avg_elapsed_s_per_example",
        "errors",
    ]:
        lines.append(f"| `{key}` | `{summary.get(key)}` |")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--tgt", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/ChemEnzyRetroPlanner"))
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
