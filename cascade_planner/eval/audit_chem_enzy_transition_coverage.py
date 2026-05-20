"""Audit ChemEnzy one-step candidate coverage for v4 cascade transitions."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles, canonical_side


AUDIT_SCHEMA_VERSION = "chem_enzy_v4_transition_coverage.v1"


def audit_chem_enzy_transition_coverage(
    *,
    v4_jsonl: Path,
    output: Path,
    report: Path,
    cache: Path | None = None,
    top_ks: list[int] | None = None,
    limit_transitions: int | None = None,
    limit_products: int | None = None,
    offset: int = 0,
    gpu: int = -1,
    one_step_models: list[str] | None = None,
    onmt_model_path: Path | str | None = None,
    expansion_topk: int = 100,
    product_smiles_filter: set[str] | None = None,
    split_manifest: Path | None = None,
    split: str | None = None,
    include_multi_product_steps: bool = False,
    similarity_threshold: float = 0.7,
) -> dict[str, Any]:
    top_ks = sorted({int(k) for k in (top_ks or [20, 50, 100]) if int(k) > 0})
    max_topk = max(top_ks) if top_ks else int(expansion_topk)
    transitions_all = collect_v4_transitions(
        v4_jsonl=v4_jsonl,
        split_manifest=split_manifest,
        split=split,
        include_multi_product_steps=include_multi_product_steps,
    )
    if product_smiles_filter:
        transitions_all = [row for row in transitions_all if row.get("product_smiles") in product_smiles_filter]
    offset = max(0, int(offset or 0))
    if offset:
        transitions_all = transitions_all[offset:]
    if limit_transitions is not None and limit_transitions > 0:
        transitions_all = transitions_all[: int(limit_transitions)]
    product_order = []
    seen_products = set()
    for row in transitions_all:
        product = str(row.get("product_smiles") or "")
        if product and product not in seen_products:
            seen_products.add(product)
            product_order.append(product)
    if limit_products is not None and limit_products > 0:
        keep = set(product_order[: int(limit_products)])
        transitions = [row for row in transitions_all if row.get("product_smiles") in keep]
    else:
        keep = set(product_order)
        transitions = transitions_all

    cached = _read_cache(cache) if cache else {}
    provider = None
    cache_updates = 0
    product_rows: dict[str, list[dict[str, Any]]] = {}
    started = time.monotonic()
    for product in product_order:
        if product not in keep:
            continue
        cache_key = _cache_key(product, max_topk, one_step_models, onmt_model_path=onmt_model_path)
        rows = cached.get(cache_key)
        if rows is None:
            if provider is None:
                provider = ChemEnzyOneStepProposalProvider(
                    models=tuple(one_step_models or ()),
                    expansion_topk=max(int(expansion_topk), max_topk),
                    gpu=int(gpu),
                    onmt_model_path=onmt_model_path,
                )
            rows = provider.predict(product, top_k=max_topk)
            cached[cache_key] = rows
            cache_updates += 1
            if cache and cache_updates % 10 == 0:
                _write_cache(cache, cached)
        product_rows[product] = rows
    if cache:
        _write_cache(cache, cached)

    transition_results = [
        _score_transition(row, product_rows.get(str(row.get("product_smiles") or ""), []), top_ks=top_ks, similarity_threshold=similarity_threshold)
        for row in transitions
    ]
    summary = _summarize(transition_results, top_ks=top_ks)
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "output": str(output),
            "report": str(report),
            "cache": str(cache) if cache else None,
            "top_ks": top_ks,
            "max_topk": max_topk,
            "limit_transitions": limit_transitions,
            "limit_products": limit_products,
            "offset": offset,
            "gpu": gpu,
            "one_step_models": one_step_models or [],
            "onmt_model_path": str(onmt_model_path) if onmt_model_path else None,
            "expansion_topk": expansion_topk,
            "split_manifest": str(split_manifest) if split_manifest else None,
            "split": split,
            "include_multi_product_steps": include_multi_product_steps,
            "similarity_threshold": similarity_threshold,
            "elapsed_s": round(time.monotonic() - started, 3),
            "cache_updates": cache_updates,
        },
        "summary": summary,
        "transitions": transition_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"metadata": payload["metadata"], "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def collect_v4_transitions(
    *,
    v4_jsonl: Path,
    split_manifest: Path | None = None,
    split: str | None = None,
    include_multi_product_steps: bool = False,
) -> list[dict[str, Any]]:
    split_keys = _split_keys(split_manifest, split)
    rows = []
    skipped = Counter()
    for raw in _iter_jsonl(v4_jsonl):
        key = (_norm(raw.get("doi")), _norm(raw.get("cascade_id")))
        if split_keys is not None and key not in split_keys:
            skipped["not_in_split"] += 1
            continue
        if ";" in str(raw.get("target_product_smiles") or ""):
            skipped["multi_target_row"] += 1
            continue
        steps = [step for step in raw.get("steps") or [] if isinstance(step, dict)]
        previous_superclass = ""
        for step_pos, step in enumerate(sorted(steps, key=lambda item: int(item.get("step_index") or 0))):
            rxn = str(step.get("rxn_smiles") or "").strip()
            if ">>" not in rxn:
                skipped["missing_rxn_smiles"] += 1
                continue
            reactants, products = _rxn_sides(rxn)
            if not reactants or not products:
                skipped["empty_side"] += 1
                continue
            if len(products) != 1 and not include_multi_product_steps:
                skipped["multi_product_step"] += 1
                continue
            product = canonical_smiles(products[0]) or products[0]
            canon_reactants = sorted(canonical_smiles(smi) or smi for smi in reactants if smi)
            if not product or not canon_reactants:
                skipped["invalid_canonical_side"] += 1
                continue
            superclass = str(step.get("transformation_superclass") or "unknown")
            rows.append(
                {
                    "transition_id": _stable_id(raw.get("doi"), raw.get("cascade_id"), step.get("step_id"), rxn),
                    "doi": raw.get("doi"),
                    "cascade_id": raw.get("cascade_id"),
                    "quality_tier": raw.get("quality_tier"),
                    "route_domain": raw.get("cascade_type") or raw.get("route_domain") or "unknown",
                    "target_smiles": canonical_smiles(str(raw.get("target_product_smiles") or "")) or str(raw.get("target_product_smiles") or ""),
                    "step_index": step.get("step_index"),
                    "step_pos": step_pos,
                    "remaining_steps": max(0, len(steps) - step_pos - 1),
                    "rxn_smiles": canonical_reaction(rxn) or rxn,
                    "product_smiles": product,
                    "reactants": canon_reactants,
                    "main_reactant": _largest_smiles(canon_reactants),
                    "transformation_superclass": superclass,
                    "previous_transformation_superclass": previous_superclass,
                    "step_mode": step.get("step_mode") or "unknown",
                    "pairwise_mode": step.get("pairwise_mode") or "unknown",
                    "intermediate_isolated": step.get("intermediate_isolated"),
                    "catalyst_classes": sorted({
                        str(cat.get("catalyst_class") or "")
                        for cat in step.get("catalyst_components") or []
                        if isinstance(cat, dict) and cat.get("catalyst_class")
                    }),
                    "ec1_values": sorted({
                        str(cat.get("ec_number") or "").split(".", 1)[0]
                        for cat in step.get("catalyst_components") or []
                        if isinstance(cat, dict) and cat.get("ec_number")
                    }),
                }
            )
            previous_superclass = superclass
    rows.sort(key=lambda row: (str(row.get("doi") or ""), str(row.get("cascade_id") or ""), int(row.get("step_index") or 0), row["transition_id"]))
    return rows


def _score_transition(
    transition: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    top_ks: list[int],
    similarity_threshold: float,
) -> dict[str, Any]:
    gt_rxn = canonical_reaction(str(transition.get("rxn_smiles") or ""))
    gt_reactants = set(str(smi) for smi in transition.get("reactants") or [])
    gt_main = str(transition.get("main_reactant") or "")
    scored_candidates = []
    exact_rank = None
    reactant_set_rank = None
    main_reactant_rank = None
    any_reactant_rank = None
    similarity_rank = None
    best_reactant_similarity = 0.0
    for idx, cand in enumerate(candidates, 1):
        rxn = canonical_reaction(cand.get("reaction_smiles") or cand.get("rxn_smiles"))
        cand_reactants = set(canonical_side((rxn.split(">>", 1)[0] if ">>" in rxn else "")))
        cand_main = canonical_smiles(cand.get("main_reactant")) or str(cand.get("main_reactant") or "")
        reactant_sim = _best_set_similarity(gt_reactants, cand_reactants)
        best_reactant_similarity = max(best_reactant_similarity, reactant_sim)
        is_exact = bool(gt_rxn and rxn == gt_rxn)
        is_reactant_set = bool(gt_reactants and cand_reactants == gt_reactants)
        is_main = bool(gt_main and cand_main == gt_main)
        is_any_reactant = bool(gt_reactants & cand_reactants)
        is_similar = reactant_sim >= similarity_threshold
        if is_exact and exact_rank is None:
            exact_rank = idx
        if is_reactant_set and reactant_set_rank is None:
            reactant_set_rank = idx
        if is_main and main_reactant_rank is None:
            main_reactant_rank = idx
        if is_any_reactant and any_reactant_rank is None:
            any_reactant_rank = idx
        if is_similar and similarity_rank is None:
            similarity_rank = idx
        scored_candidates.append(
            {
                "rank": idx,
                "reaction_smiles": rxn,
                "reactants": sorted(cand_reactants),
                "main_reactant": cand_main,
                "source": cand.get("source"),
                "model_full_name": cand.get("model_full_name"),
                "score": cand.get("score"),
                "exact_reaction_hit": is_exact,
                "reactant_set_hit": is_reactant_set,
                "main_reactant_hit": is_main,
                "any_reactant_hit": is_any_reactant,
                "reactant_similarity": round(reactant_sim, 6),
            }
        )
    out = {
        **transition,
        "candidate_count": len(candidates),
        "exact_reaction_rank": exact_rank,
        "reactant_set_rank": reactant_set_rank,
        "main_reactant_rank": main_reactant_rank,
        "any_reactant_rank": any_reactant_rank,
        "similar_reactant_rank": similarity_rank,
        "best_reactant_similarity": round(best_reactant_similarity, 6),
        "coverage": {},
        "candidates_preview": scored_candidates[: min(max(top_ks or [10]), 20)],
    }
    for k in top_ks:
        out["coverage"][f"exact@{k}"] = bool(exact_rank is not None and exact_rank <= k)
        out["coverage"][f"reactant_set@{k}"] = bool(reactant_set_rank is not None and reactant_set_rank <= k)
        out["coverage"][f"main_reactant@{k}"] = bool(main_reactant_rank is not None and main_reactant_rank <= k)
        out["coverage"][f"any_reactant@{k}"] = bool(any_reactant_rank is not None and any_reactant_rank <= k)
        out["coverage"][f"similar_reactant@{k}"] = bool(similarity_rank is not None and similarity_rank <= k)
    return out


def _summarize(rows: list[dict[str, Any]], *, top_ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_transitions": len(rows),
        "unique_products": len({row.get("product_smiles") for row in rows if row.get("product_smiles")}),
        "avg_candidate_count": round(sum(int(row.get("candidate_count") or 0) for row in rows) / max(len(rows), 1), 6),
        "route_domain_counts": dict(Counter(row.get("route_domain") for row in rows)),
        "transformation_superclass_counts": dict(Counter(row.get("transformation_superclass") for row in rows)),
        "candidate_count_zero": sum(1 for row in rows if int(row.get("candidate_count") or 0) == 0),
    }
    for metric in ("exact", "reactant_set", "main_reactant", "any_reactant", "similar_reactant"):
        for k in top_ks:
            key = f"{metric}@{k}"
            out[key] = round(sum(int(bool((row.get("coverage") or {}).get(key))) for row in rows) / max(len(rows), 1), 6)
    for group_field in ("route_domain", "transformation_superclass"):
        grouped = {}
        values = sorted({str(row.get(group_field) or "unknown") for row in rows})
        for value in values:
            subset = [row for row in rows if str(row.get(group_field) or "unknown") == value]
            grouped[value] = {
                "n": len(subset),
                **{
                    f"exact@{k}": round(sum(int(bool((row.get("coverage") or {}).get(f"exact@{k}"))) for row in subset) / max(len(subset), 1), 6)
                    for k in top_ks
                },
                **{
                    f"reactant_set@{k}": round(sum(int(bool((row.get("coverage") or {}).get(f"reactant_set@{k}"))) for row in subset) / max(len(subset), 1), 6)
                    for k in top_ks
                },
                **{
                    f"similar_reactant@{k}": round(sum(int(bool((row.get("coverage") or {}).get(f"similar_reactant@{k}"))) for row in subset) / max(len(subset), 1), 6)
                    for k in top_ks
                },
            }
        out[f"by_{group_field}"] = grouped
    return out


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    meta = payload.get("metadata") or {}
    top_ks = meta.get("top_ks") or []
    lines = [
        "# ChemEnzy v4 Transition Coverage Audit",
        "",
        f"- Transitions: `{summary.get('n_transitions')}`",
        f"- Unique products: `{summary.get('unique_products')}`",
        f"- Avg candidates: `{summary.get('avg_candidate_count')}`",
        f"- Zero-candidate transitions: `{summary.get('candidate_count_zero')}`",
        "",
        "## Coverage",
        "",
        "| Metric | " + " | ".join(f"@{k}" for k in top_ks) + " |",
        "|---|" + "|".join("---:" for _ in top_ks) + "|",
    ]
    for metric in ("exact", "reactant_set", "main_reactant", "any_reactant", "similar_reactant"):
        lines.append("| " + metric + " | " + " | ".join(str(summary.get(f"{metric}@{k}")) for k in top_ks) + " |")
    lines.extend(["", "## Domains", "", "```json", json.dumps(summary.get("by_route_domain") or {}, indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def _rxn_sides(rxn: str) -> tuple[list[str], list[str]]:
    lhs, rhs = rxn.split(">>", 1)
    return list(canonical_side(lhs)), list(canonical_side(rhs))


def _largest_smiles(values: list[str]) -> str:
    if not values:
        return ""
    heavy_counts = []
    for smi in values:
        mol = Chem.MolFromSmiles(smi)
        heavy_counts.append((mol.GetNumHeavyAtoms() if mol is not None else len(smi), smi))
    return max(heavy_counts)[1]


def _best_set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return max((_tanimoto(a, b) for a in left for b in right), default=0.0)


def _tanimoto(a: str, b: str) -> float:
    mol_a = Chem.MolFromSmiles(str(a or ""))
    mol_b = Chem.MolFromSmiles(str(b or ""))
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _cache_key(product: str, topk: int, models: list[str] | None, *, onmt_model_path: Path | str | None = None) -> str:
    return json.dumps(
        {
            "product": product,
            "topk": int(topk),
            "models": list(models or []),
            "onmt_model_path": str(onmt_model_path) if onmt_model_path else None,
        },
        sort_keys=True,
    )


def _read_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _split_keys(path: Path | None, split: str | None) -> set[tuple[str, str]] | None:
    if path is None or not split:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split_path = (((manifest.get("outputs") or {}).get(split)) if isinstance(manifest, dict) else None)
    if not split_path:
        split_path = str(Path(path).parent / f"v4_trace_{split}.json")
    rows = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return {
        (_norm(row.get("doi")), _norm(row.get("cascade_id")))
        for row in rows
        if isinstance(row, dict)
    }


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _stable_id(*parts: Any) -> str:
    import hashlib

    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_topks(raw: str) -> list[int]:
    return [int(token.strip()) for token in str(raw or "").split(",") if token.strip()]


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit ChemEnzy one-step coverage of v4 cascade transitions")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--top-ks", default="20,50,100")
    ap.add_argument("--limit-transitions", type=int)
    ap.add_argument("--limit-products", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--one-step-model", action="append", default=[])
    ap.add_argument("--onmt-model-path")
    ap.add_argument("--expansion-topk", type=int, default=100)
    ap.add_argument("--split-manifest")
    ap.add_argument("--split")
    ap.add_argument("--include-multi-product-steps", action="store_true")
    ap.add_argument("--similarity-threshold", type=float, default=0.7)
    args = ap.parse_args()
    payload = audit_chem_enzy_transition_coverage(
        v4_jsonl=Path(args.v4_jsonl),
        output=Path(args.output),
        report=Path(args.report),
        cache=Path(args.cache) if args.cache else None,
        top_ks=_parse_topks(args.top_ks),
        limit_transitions=args.limit_transitions,
        limit_products=args.limit_products,
        offset=args.offset,
        gpu=args.gpu,
        one_step_models=args.one_step_model,
        onmt_model_path=Path(args.onmt_model_path) if args.onmt_model_path else None,
        expansion_topk=args.expansion_topk,
        split_manifest=Path(args.split_manifest) if args.split_manifest else None,
        split=args.split,
        include_multi_product_steps=args.include_multi_product_steps,
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
