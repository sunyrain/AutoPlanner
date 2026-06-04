#!/usr/bin/env python
"""Audit ChemEnzy one-step proposal coverage on benchmarks with gt_route."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side


SCHEMA_VERSION = "chem_enzy_onestep_benchmark_audit.v1"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, help="NAME=model1,model2")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument("--limit-targets", type=int)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--step-scope", choices=["all", "target"], default="all")
    args = ap.parse_args()

    rows = _load_benchmark(args.benchmark)
    if args.limit_targets is not None:
        rows = rows[: max(0, int(args.limit_targets))]
    transitions = _collect_transitions(rows, step_scope=args.step_scope)
    cache = _read_cache(args.cache)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark": str(args.benchmark),
        "settings": {
            "limit_targets": args.limit_targets,
            "topk": int(args.topk),
            "gpu": int(args.gpu),
            "step_scope": args.step_scope,
        },
        "transition_count": len(transitions),
        "runs": {},
        "pairwise": {},
    }

    runs = [_parse_run(raw) for raw in args.run]
    for name, models in runs:
        started = time.monotonic()
        provider = ChemEnzyOneStepProposalProvider(
            vendor_root=Path(args.vendor_root),
            models=tuple(models),
            expansion_topk=int(args.topk),
            gpu=int(args.gpu),
        )
        scored = []
        cache_updates = 0
        for transition in transitions:
            product = str(transition.get("product_smiles") or "")
            key = _cache_key(product, models, int(args.topk), int(args.gpu))
            candidates = cache.get(key)
            if candidates is None:
                candidates = provider.predict(product, top_k=int(args.topk))
                cache[key] = candidates
                cache_updates += 1
                if args.cache and cache_updates % 10 == 0:
                    _write_cache(args.cache, cache)
            scored.append(_score_transition(transition, candidates, topk=int(args.topk)))
        if args.cache:
            _write_cache(args.cache, cache)
        payload["runs"][name] = {
            "models": models,
            "summary": _summarize(scored),
            "transitions": scored,
            "elapsed_s": round(time.monotonic() - started, 3),
            "cache_updates": cache_updates,
            "load_error": provider.load_error,
        }

    if len(runs) >= 2:
        names = [name for name, _models in runs]
        for left_idx, left in enumerate(names):
            for right in names[left_idx + 1 :]:
                payload["pairwise"][f"{left}_vs_{right}"] = _pairwise(
                    payload["runs"][left]["transitions"],
                    payload["runs"][right]["transitions"],
                    left,
                    right,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md = _markdown(payload)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(md, encoding="utf-8")
    print(md)


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in data if isinstance(row, dict) and row.get("target_smiles")]


def _parse_run(raw: str) -> tuple[str, list[str]]:
    if "=" not in raw:
        raise ValueError("--run must be NAME=model1,model2")
    name, model_text = raw.split("=", 1)
    models = [item.strip() for item in model_text.split(",") if item.strip()]
    if not name.strip() or not models:
        raise ValueError("--run must include a nonempty name and at least one model")
    return name.strip(), models


def _collect_transitions(rows: list[dict[str, Any]], *, step_scope: str) -> list[dict[str, Any]]:
    transitions = []
    for target_idx, row in enumerate(rows):
        target_side = canonical_side(str(row.get("target_smiles") or ""))
        for step_idx, step in enumerate(row.get("gt_route") or []):
            rxn = canonical_reaction(step.get("rxn_smiles"))
            if not rxn or ">>" not in rxn:
                continue
            lhs, rhs = rxn.split(">>", 1)
            if step_scope == "target" and canonical_side(rhs) != target_side:
                continue
            transitions.append(
                {
                    "target_index": target_idx,
                    "step_index": step_idx,
                    "cascade_id": row.get("cascade_id"),
                    "route_domain": row.get("route_domain"),
                    "target_smiles": row.get("target_smiles"),
                    "product_smiles": ".".join(canonical_side(rhs)),
                    "rxn_smiles": rxn,
                    "reactants": sorted(canonical_side(lhs)),
                    "transformation": step.get("transformation"),
                    "step_role": step.get("step_role"),
                }
            )
    return transitions


def _score_transition(transition: dict[str, Any], candidates: list[dict[str, Any]], *, topk: int) -> dict[str, Any]:
    gt_rxn = str(transition.get("rxn_smiles") or "")
    gt_reactants = set(str(item) for item in transition.get("reactants") or [])
    exact_rank = None
    reactant_set_rank = None
    any_reactant_rank = None
    candidate_rows = []
    for rank, candidate in enumerate(candidates[: max(0, int(topk))], 1):
        rxn = canonical_reaction(candidate.get("reaction_smiles") or candidate.get("rxn_smiles"))
        lhs = rxn.split(">>", 1)[0] if ">>" in rxn else ""
        reactants = set(canonical_side(lhs))
        exact = bool(rxn and rxn == gt_rxn)
        reactant_set = bool(gt_reactants and reactants == gt_reactants)
        any_reactant = bool(gt_reactants & reactants)
        if exact and exact_rank is None:
            exact_rank = rank
        if reactant_set and reactant_set_rank is None:
            reactant_set_rank = rank
        if any_reactant and any_reactant_rank is None:
            any_reactant_rank = rank
        candidate_rows.append(
            {
                "rank": rank,
                "reaction_smiles": rxn,
                "reactants": sorted(reactants),
                "source": candidate.get("source"),
                "model_full_name": candidate.get("model_full_name"),
                "score": candidate.get("score"),
                "exact_reaction_hit": exact,
                "reactant_set_hit": reactant_set,
                "any_reactant_hit": any_reactant,
            }
        )
    return {
        **transition,
        "candidate_count": len(candidates),
        "exact_reaction_rank": exact_rank,
        "reactant_set_rank": reactant_set_rank,
        "any_reactant_rank": any_reactant_rank,
        "exact_reaction_hit": exact_rank is not None,
        "reactant_set_hit": reactant_set_rank is not None,
        "any_reactant_hit": any_reactant_rank is not None,
        "source_counts": dict(Counter(str(row.get("source") or "") for row in candidate_rows)),
        "model_counts": dict(Counter(str(row.get("model_full_name") or "") for row in candidate_rows)),
        "candidates_preview": candidate_rows[:10],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    candidate_counts = [int(row.get("candidate_count") or 0) for row in rows]
    source_counts = Counter()
    model_counts = Counter()
    domain_counts = Counter(str(row.get("route_domain") or "unknown") for row in rows)
    for row in rows:
        source_counts.update(row.get("source_counts") or {})
        model_counts.update(row.get("model_counts") or {})
    return {
        "n_transitions": n,
        "unique_products": len({row.get("product_smiles") for row in rows if row.get("product_smiles")}),
        "route_domain_counts": dict(domain_counts),
        "avg_candidate_count": sum(candidate_counts) / max(n, 1),
        "zero_candidate_transitions": sum(1 for value in candidate_counts if value == 0),
        "exact_reaction_hit": sum(1 for row in rows if row.get("exact_reaction_hit")),
        "exact_reaction_hit_rate": _rate(sum(1 for row in rows if row.get("exact_reaction_hit")), n),
        "reactant_set_hit": sum(1 for row in rows if row.get("reactant_set_hit")),
        "reactant_set_hit_rate": _rate(sum(1 for row in rows if row.get("reactant_set_hit")), n),
        "any_reactant_hit": sum(1 for row in rows if row.get("any_reactant_hit")),
        "any_reactant_hit_rate": _rate(sum(1 for row in rows if row.get("any_reactant_hit")), n),
        "source_counts": dict(source_counts.most_common()),
        "model_counts": dict(model_counts.most_common()),
    }


def _pairwise(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    out: dict[str, Any] = {"paired_transitions": min(len(left_rows), len(right_rows))}
    for key in ("exact_reaction_hit", "reactant_set_hit", "any_reactant_hit"):
        left_only = right_only = both = neither = 0
        for lrow, rrow in zip(left_rows, right_rows):
            lhit = bool(lrow.get(key))
            rhit = bool(rrow.get(key))
            if lhit and rhit:
                both += 1
            elif lhit:
                left_only += 1
            elif rhit:
                right_only += 1
            else:
                neither += 1
        out[key] = {
            f"{left}_only": left_only,
            f"{right}_only": right_only,
            "both": both,
            "neither": neither,
        }
    return out


def _rate(value: int, n: int) -> float:
    return round(value / max(n, 1), 6)


def _read_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _cache_key(product: str, models: list[str], topk: int, gpu: int) -> str:
    return json.dumps(
        {"product": product, "models": models, "topk": int(topk), "gpu": int(gpu)},
        sort_keys=True,
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ChemEnzy One-step Benchmark Audit",
        "",
        f"Benchmark: `{payload.get('benchmark')}`",
        f"Transitions: `{payload.get('transition_count')}`",
        "",
        "| run | transitions | avg candidates | zero candidates | exact rxn | reactant set | any reactant | elapsed s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, run in payload.get("runs", {}).items():
        s = run.get("summary") or {}
        lines.append(
            f"| {name} | {s.get('n_transitions')} | {s.get('avg_candidate_count')} | "
            f"{s.get('zero_candidate_transitions')} | "
            f"{s.get('exact_reaction_hit')} ({s.get('exact_reaction_hit_rate')}) | "
            f"{s.get('reactant_set_hit')} ({s.get('reactant_set_hit_rate')}) | "
            f"{s.get('any_reactant_hit')} ({s.get('any_reactant_hit_rate')}) | "
            f"{run.get('elapsed_s')} |"
        )
    if payload.get("pairwise"):
        lines.extend(["", "## Pairwise", ""])
        for name, pair in payload["pairwise"].items():
            lines.append(f"### {name}")
            lines.append(f"- paired transitions: {pair.get('paired_transitions')}")
            for key in ("exact_reaction_hit", "reactant_set_hit", "any_reactant_hit"):
                lines.append(f"- {key}: {pair.get(key)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
