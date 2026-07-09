"""Audit train-split cascade evidence inside a fixed ChemEnzy route pool.

This is intentionally not a route-quality model and not a held-out GT recovery
metric.  It answers a narrower question after the Phase II CCTS no-go:

    Given a fixed native route pool, how many route steps / connected blocks are
    supported by cascade evidence from the v4 train split?

The output is meant to guide product review and later supervision design.  It
does not claim that an evidence-supported route is chemically feasible.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import DataStructs, RDLogger

from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.eval.audit_v4_heldout_block_recovery import (
    _load_routes,
    _native_rank,
    _norm_transform,
    _route_blocks,
    _transition_fp,
)
from cascade_planner.eval.replay_block_coherence_on_route_pool import _main_reactant


SCHEMA_VERSION = "route_pool_cascade_evidence_audit.v1"


def audit_route_pool_cascade_evidence(
    *,
    route_pool: Path,
    program_manifest: Path,
    output_json: Path,
    output_md: Path | None = None,
    evidence_split: str = "train",
    analog_similarity: float = 0.55,
    top_ks: tuple[int, ...] = (1, 3, 5, 10, 50),
    max_examples: int = 40,
) -> dict[str, Any]:
    started = time.monotonic()
    routes = _load_routes(route_pool)
    bank = _evidence_bank(program_manifest, split=evidence_split)
    audited = [
        _audit_route(route, bank=bank, analog_similarity=analog_similarity)
        for route in routes
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "program_manifest": str(program_manifest),
            "evidence_split": evidence_split,
            "analog_similarity": analog_similarity,
            "top_ks": list(top_ks),
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "fixed route-pool evidence audit; v4 evidence is used only from "
                f"{evidence_split!r} split; no route preference labels and no learned scoring"
            ),
        },
        "evidence_bank": bank["summary"],
        "summary": _summary(audited, top_ks=top_ks),
        "target_summary": _target_summary(audited, top_ks=top_ks),
        "examples": _examples(audited, max_examples=max_examples),
        "routes": [_compact_route(row) for row in audited],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _audit_route(route: dict[str, Any], *, bank: dict[str, Any], analog_similarity: float) -> dict[str, Any]:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    step_rows = [_audit_step(step, bank=bank) for step in steps]
    block_rows = [_audit_block(block, bank=bank, analog_similarity=analog_similarity) for block in _route_blocks(route)]
    best_any = max((float(row.get("best_any_block_min_sim") or 0.0) for row in block_rows), default=0.0)
    best_same_pair = max((float(row.get("best_same_pair_block_min_sim") or 0.0) for row in block_rows), default=0.0)
    pair_supported = [row for row in block_rows if row.get("pair_observed_in_evidence")]
    same_pair_analog = [row for row in block_rows if row.get("same_pair_analog_supported")]
    any_analog = [row for row in block_rows if row.get("any_analog_supported")]
    known_transform_steps = [row for row in step_rows if row.get("transform_observed_in_evidence")]
    route_evidence = {
        "n_steps": len(steps),
        "n_blocks": len(block_rows),
        "known_transform_step_count": len(known_transform_steps),
        "known_transform_step_fraction": _rate(len(known_transform_steps), len(step_rows)),
        "observed_pair_block_count": len(pair_supported),
        "observed_pair_block_fraction": _rate(len(pair_supported), len(block_rows)),
        "any_analog_block_count": len(any_analog),
        "same_pair_analog_block_count": len(same_pair_analog),
        "has_observed_pair_block": bool(pair_supported),
        "has_any_analog_block": bool(any_analog),
        "has_same_pair_analog_block": bool(same_pair_analog),
        "best_any_block_min_sim": round(best_any, 6),
        "best_same_pair_block_min_sim": round(best_same_pair, 6),
        "top_supported_block": _top_supported_block(block_rows),
        "transform_pairs": [row.get("transform_pair") for row in block_rows],
    }
    out = dict(route)
    out["cascade_evidence_audit"] = route_evidence
    out["cascade_evidence_steps"] = step_rows
    out["cascade_evidence_blocks"] = block_rows
    return out


def _audit_step(step: dict[str, Any], *, bank: dict[str, Any]) -> dict[str, Any]:
    product = canonical_smiles(str(step.get("product_smiles") or ((step.get("products") or [""])[0] if isinstance(step.get("products"), list) else "")))
    main = canonical_smiles(_main_reactant(step))
    transform = _norm_transform(step.get("transformation_superclass"))
    fp = _transition_fp(product, main)
    best_any = _best_transition(fp, bank["transitions"])
    best_same_transform = _best_transition(fp, bank["transitions_by_transform"].get(transform) or [])
    return {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "product_smiles": product,
        "main_reactant": main,
        "transform": transform,
        "transform_count_in_evidence": int((bank["transform_counts"]).get(transform, 0)),
        "transform_observed_in_evidence": bool((bank["transform_counts"]).get(transform, 0)),
        "best_any_transition_sim": best_any.get("similarity", 0.0),
        "best_any_transition_support": best_any.get("support"),
        "best_same_transform_transition_sim": best_same_transform.get("similarity", 0.0),
        "best_same_transform_transition_support": best_same_transform.get("support"),
    }


def _audit_block(block: dict[str, Any], *, bank: dict[str, Any], analog_similarity: float) -> dict[str, Any]:
    pair = str(block.get("transform_pair") or "").lower()
    best_any = _best_block(block, bank["adjacencies"])
    best_same_pair = _best_block(block, bank["adjacencies_by_pair"].get(pair) or [])
    pair_count = int(bank["pair_counts"].get(pair, 0))
    return {
        "route_block_index": block.get("route_block_index"),
        "upstream_rxn": block.get("upstream_rxn"),
        "downstream_rxn": block.get("downstream_rxn"),
        "upstream_product": block.get("upstream_product"),
        "downstream_product": block.get("downstream_product"),
        "upstream_main_reactant": block.get("upstream_main_reactant"),
        "downstream_main_reactant": block.get("downstream_main_reactant"),
        "upstream_transform": block.get("upstream_transform"),
        "downstream_transform": block.get("downstream_transform"),
        "transform_pair": pair,
        "pair_count_in_evidence": pair_count,
        "pair_observed_in_evidence": bool(pair_count),
        "best_any_block_min_sim": best_any.get("min_sim", 0.0),
        "best_any_block_mean_sim": best_any.get("mean_sim", 0.0),
        "best_any_support": best_any.get("support"),
        "best_same_pair_block_min_sim": best_same_pair.get("min_sim", 0.0),
        "best_same_pair_block_mean_sim": best_same_pair.get("mean_sim", 0.0),
        "best_same_pair_support": best_same_pair.get("support"),
        "any_analog_supported": bool(float(best_any.get("min_sim") or 0.0) >= analog_similarity),
        "same_pair_analog_supported": bool(pair_count and float(best_same_pair.get("min_sim") or 0.0) >= analog_similarity),
    }


def _evidence_bank(program_manifest: Path, *, split: str) -> dict[str, Any]:
    manifest = json.loads(Path(program_manifest).read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    if split not in outputs:
        raise ValueError(f"split {split!r} not found in {program_manifest}; available={sorted(outputs)}")
    programs = _read_jsonl(Path(outputs[split]))
    transitions = []
    adjacencies = []
    for program in programs:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        step_items = [_evidence_transition(program, step) for step in steps]
        transitions.extend([item for item in step_items if item.get("transition_fp") is not None])
        for upstream, downstream in zip(step_items, step_items[1:]):
            if upstream.get("transition_fp") is None or downstream.get("transition_fp") is None:
                continue
            pair = f"{upstream['transform']}->{downstream['transform']}".lower()
            adjacencies.append(
                {
                    "program_id": program.get("program_id"),
                    "doi": program.get("doi"),
                    "cascade_id": program.get("cascade_id"),
                    "quality_tier": program.get("quality_tier"),
                    "cascade_type": program.get("cascade_type"),
                    "transform_pair": pair,
                    "upstream_transform": upstream["transform"],
                    "downstream_transform": downstream["transform"],
                    "upstream_fp": upstream["transition_fp"],
                    "downstream_fp": downstream["transition_fp"],
                    "upstream_product": upstream.get("product_smiles"),
                    "downstream_product": downstream.get("product_smiles"),
                }
            )
    transitions_by_transform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in transitions:
        transitions_by_transform[str(item.get("transform") or "unknown").lower()].append(item)
    adjacencies_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in adjacencies:
        adjacencies_by_pair[str(item.get("transform_pair") or "").lower()].append(item)
    transform_counts = Counter(str(item.get("transform") or "unknown").lower() for item in transitions)
    pair_counts = Counter(str(item.get("transform_pair") or "").lower() for item in adjacencies)
    return {
        "transitions": transitions,
        "transitions_by_transform": dict(transitions_by_transform),
        "adjacencies": adjacencies,
        "adjacencies_by_pair": dict(adjacencies_by_pair),
        "transform_counts": dict(transform_counts),
        "pair_counts": dict(pair_counts),
        "summary": {
            "split": split,
            "programs": len(programs),
            "transitions": len(transitions),
            "adjacencies": len(adjacencies),
            "unique_transforms": len(transform_counts),
            "unique_transform_pairs": len(pair_counts),
            "top_transforms": dict(transform_counts.most_common(20)),
            "top_transform_pairs": dict(pair_counts.most_common(20)),
            "source_split_path": outputs[split],
        },
    }


def _evidence_transition(program: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    product = canonical_smiles(str(step.get("product_smiles") or ""))
    main = canonical_smiles(str(step.get("main_reactant") or ""))
    transform = _norm_transform(step.get("transformation_superclass"))
    return {
        "transition_id": step.get("transition_id"),
        "program_id": program.get("program_id"),
        "doi": program.get("doi"),
        "cascade_id": program.get("cascade_id"),
        "quality_tier": program.get("quality_tier"),
        "cascade_type": program.get("cascade_type"),
        "transform": transform,
        "product_smiles": product,
        "main_reactant": main,
        "transition_fp": _transition_fp(product, main),
    }


def _best_transition(fp: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    if fp is None or not items:
        return {"similarity": 0.0, "support": None}
    fps = [item["transition_fp"] for item in items]
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    if not sims:
        return {"similarity": 0.0, "support": None}
    best_idx = int(np.argmax(sims))
    return {
        "similarity": round(float(sims[best_idx]), 6),
        "support": _compact_transition_support(items[best_idx]),
    }


def _best_block(block: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items or block.get("upstream_fp") is None or block.get("downstream_fp") is None:
        return {"min_sim": 0.0, "mean_sim": 0.0, "support": None}
    upstream_sims = DataStructs.BulkTanimotoSimilarity(block["upstream_fp"], [item["upstream_fp"] for item in items])
    downstream_sims = DataStructs.BulkTanimotoSimilarity(block["downstream_fp"], [item["downstream_fp"] for item in items])
    best_idx = -1
    best_min = -1.0
    best_mean = -1.0
    for idx, (up_sim, down_sim) in enumerate(zip(upstream_sims, downstream_sims)):
        min_sim = min(float(up_sim), float(down_sim))
        mean_sim = 0.5 * (float(up_sim) + float(down_sim))
        if (min_sim, mean_sim) > (best_min, best_mean):
            best_idx = idx
            best_min = min_sim
            best_mean = mean_sim
    if best_idx < 0:
        return {"min_sim": 0.0, "mean_sim": 0.0, "support": None}
    return {
        "min_sim": round(best_min, 6),
        "mean_sim": round(best_mean, 6),
        "support": _compact_adjacency_support(items[best_idx]),
    }


def _summary(routes: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, Any]:
    evidence_rows = [route.get("cascade_evidence_audit") or {} for route in routes]
    multistep = [row for row in routes if len(row.get("steps") or []) >= 2]
    out: dict[str, Any] = {
        "routes": len(routes),
        "targets": len(_target_groups(routes)),
        "steps": sum(len(route.get("steps") or []) for route in routes),
        "blocks": sum(int(row.get("n_blocks") or 0) for row in evidence_rows),
        "multistep_routes": len(multistep),
        "routes_with_observed_pair_block": sum(1 for row in evidence_rows if row.get("has_observed_pair_block")),
        "routes_with_any_analog_block": sum(1 for row in evidence_rows if row.get("has_any_analog_block")),
        "routes_with_same_pair_analog_block": sum(1 for row in evidence_rows if row.get("has_same_pair_analog_block")),
        "mean_known_transform_step_fraction": round(float(np.mean([row.get("known_transform_step_fraction") or 0.0 for row in evidence_rows])) if evidence_rows else 0.0, 6),
        "mean_observed_pair_block_fraction": round(float(np.mean([row.get("observed_pair_block_fraction") or 0.0 for row in evidence_rows])) if evidence_rows else 0.0, 6),
        "best_any_block_min_sim_max": round(max((float(row.get("best_any_block_min_sim") or 0.0) for row in evidence_rows), default=0.0), 6),
        "best_same_pair_block_min_sim_max": round(max((float(row.get("best_same_pair_block_min_sim") or 0.0) for row in evidence_rows), default=0.0), 6),
        "top_transform_pairs_in_routes": dict(
            Counter(pair for row in evidence_rows for pair in row.get("transform_pairs") or []).most_common(20)
        ),
    }
    target_summary = _target_summary(routes, top_ks=top_ks)
    out["target_rates"] = target_summary["target_rates"]
    return out


def _target_summary(routes: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, Any]:
    groups = _target_groups(routes)
    target_rows = []
    for target, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: (_native_rank(row), str(row.get("route_id") or "")))
        target_rows.append(_target_row(target, group, top_ks=top_ks))
    rates: dict[str, Any] = {
        "targets": len(target_rows),
        "targets_with_multistep_route": sum(1 for row in target_rows if row.get("multistep_route_count")),
        "targets_with_observed_pair_block_anywhere": sum(1 for row in target_rows if row.get("best_observed_pair_rank") is not None),
        "targets_with_any_analog_block_anywhere": sum(1 for row in target_rows if row.get("best_any_analog_rank") is not None),
        "targets_with_same_pair_analog_block_anywhere": sum(1 for row in target_rows if row.get("best_same_pair_analog_rank") is not None),
    }
    denom = max(len(target_rows), 1)
    for k in top_ks:
        rates[f"observed_pair_block_at_{k}"] = round(
            sum(1 for row in target_rows if row.get("best_observed_pair_rank") is not None and int(row["best_observed_pair_rank"]) < k) / denom,
            6,
        )
        rates[f"any_analog_block_at_{k}"] = round(
            sum(1 for row in target_rows if row.get("best_any_analog_rank") is not None and int(row["best_any_analog_rank"]) < k) / denom,
            6,
        )
        rates[f"same_pair_analog_block_at_{k}"] = round(
            sum(1 for row in target_rows if row.get("best_same_pair_analog_rank") is not None and int(row["best_same_pair_analog_rank"]) < k) / denom,
            6,
        )
    return {"target_rates": rates, "targets": target_rows}


def _target_row(target: str, group: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, Any]:
    observed_pair_ranks = []
    any_analog_ranks = []
    same_pair_analog_ranks = []
    for route in group:
        evidence = route.get("cascade_evidence_audit") or {}
        rank = _native_rank(route)
        if evidence.get("has_observed_pair_block"):
            observed_pair_ranks.append(rank)
        if evidence.get("has_any_analog_block"):
            any_analog_ranks.append(rank)
        if evidence.get("has_same_pair_analog_block"):
            same_pair_analog_ranks.append(rank)
    return {
        "target_smiles": target,
        "route_count": len(group),
        "multistep_route_count": sum(1 for route in group if len(route.get("steps") or []) >= 2),
        "best_observed_pair_rank": min(observed_pair_ranks) if observed_pair_ranks else None,
        "best_any_analog_rank": min(any_analog_ranks) if any_analog_ranks else None,
        "best_same_pair_analog_rank": min(same_pair_analog_ranks) if same_pair_analog_ranks else None,
        "topk": {
            str(k): {
                "observed_pair_block": any(rank < k for rank in observed_pair_ranks),
                "any_analog_block": any(rank < k for rank in any_analog_ranks),
                "same_pair_analog_block": any(rank < k for rank in same_pair_analog_ranks),
            }
            for k in top_ks
        },
        "top_routes": [_compact_route(route) for route in group[:5]],
    }


def _target_groups(routes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        target = canonical_smiles(str(route.get("target_smiles") or "")) or str(route.get("target_smiles") or "")
        groups[target].append(route)
    return dict(groups)


def _examples(routes: list[dict[str, Any]], *, max_examples: int) -> dict[str, list[dict[str, Any]]]:
    same_pair = [route for route in routes if (route.get("cascade_evidence_audit") or {}).get("has_same_pair_analog_block")]
    any_analog = [route for route in routes if (route.get("cascade_evidence_audit") or {}).get("has_any_analog_block")]
    no_block = [route for route in routes if len(route.get("steps") or []) >= 2 and not (route.get("cascade_evidence_audit") or {}).get("has_observed_pair_block")]
    same_pair.sort(key=lambda row: (-float((row.get("cascade_evidence_audit") or {}).get("best_same_pair_block_min_sim") or 0.0), _native_rank(row)))
    any_analog.sort(key=lambda row: (-float((row.get("cascade_evidence_audit") or {}).get("best_any_block_min_sim") or 0.0), _native_rank(row)))
    no_block.sort(key=lambda row: (_native_rank(row), str(row.get("route_id") or "")))
    return {
        "same_pair_analog_supported_routes": [_compact_route(route, include_blocks=True) for route in same_pair[:max_examples]],
        "any_analog_supported_routes": [_compact_route(route, include_blocks=True) for route in any_analog[:max_examples]],
        "multistep_without_observed_pair_examples": [_compact_route(route, include_blocks=True) for route in no_block[:max_examples]],
    }


def _compact_route(route: dict[str, Any], *, include_blocks: bool = False) -> dict[str, Any]:
    evidence = route.get("cascade_evidence_audit") or {}
    out = {
        "route_id": route.get("route_id"),
        "target_id": route.get("target_id"),
        "target_smiles": route.get("target_smiles"),
        "native_rank": _native_rank(route),
        "native_score": route.get("native_score"),
        "stock_closed": route.get("stock_closed"),
        "n_steps": evidence.get("n_steps", len(route.get("steps") or [])),
        "n_blocks": evidence.get("n_blocks"),
        "known_transform_step_fraction": evidence.get("known_transform_step_fraction"),
        "observed_pair_block_count": evidence.get("observed_pair_block_count"),
        "any_analog_block_count": evidence.get("any_analog_block_count"),
        "same_pair_analog_block_count": evidence.get("same_pair_analog_block_count"),
        "best_any_block_min_sim": evidence.get("best_any_block_min_sim"),
        "best_same_pair_block_min_sim": evidence.get("best_same_pair_block_min_sim"),
        "has_observed_pair_block": evidence.get("has_observed_pair_block"),
        "has_any_analog_block": evidence.get("has_any_analog_block"),
        "has_same_pair_analog_block": evidence.get("has_same_pair_analog_block"),
        "transform_pairs": evidence.get("transform_pairs") or [],
    }
    if include_blocks:
        out["top_supported_block"] = evidence.get("top_supported_block")
        out["blocks"] = route.get("cascade_evidence_blocks") or []
    return out


def _top_supported_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not blocks:
        return None
    return max(
        blocks,
        key=lambda row: (
            int(bool(row.get("same_pair_analog_supported"))),
            float(row.get("best_same_pair_block_min_sim") or 0.0),
            int(bool(row.get("any_analog_supported"))),
            float(row.get("best_any_block_min_sim") or 0.0),
            int(row.get("pair_count_in_evidence") or 0),
        ),
    )


def _compact_transition_support(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "transition_id": item.get("transition_id"),
        "program_id": item.get("program_id"),
        "doi": item.get("doi"),
        "cascade_id": item.get("cascade_id"),
        "quality_tier": item.get("quality_tier"),
        "cascade_type": item.get("cascade_type"),
        "transform": item.get("transform"),
        "product_smiles": item.get("product_smiles"),
        "main_reactant": item.get("main_reactant"),
    }


def _compact_adjacency_support(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "program_id": item.get("program_id"),
        "doi": item.get("doi"),
        "cascade_id": item.get("cascade_id"),
        "quality_tier": item.get("quality_tier"),
        "cascade_type": item.get("cascade_type"),
        "transform_pair": item.get("transform_pair"),
        "upstream_product": item.get("upstream_product"),
        "downstream_product": item.get("downstream_product"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    target_rates = (summary.get("target_rates") or {})
    meta = result.get("metadata") or {}
    bank = result.get("evidence_bank") or {}
    lines = [
        "# Route Pool Cascade Evidence Audit",
        "",
        f"- route pool: `{meta.get('route_pool')}`",
        f"- evidence split: `{meta.get('evidence_split')}`",
        f"- analog threshold: `{meta.get('analog_similarity')}`",
        "",
        "## Evidence Bank",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("programs", "transitions", "adjacencies", "unique_transforms", "unique_transform_pairs"):
        lines.append(f"| `{key}` | `{bank.get(key)}` |")
    lines.extend(["", "## Route Pool Summary", "", "| Metric | Value |", "|---|---:|"])
    for key, value in summary.items():
        if key == "target_rates":
            continue
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Target-Level Evidence Rates", "", "| Metric | Value |", "|---|---:|"])
    for key, value in target_rates.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "- `observed_pair_block` means a route has a connected adjacent step pair whose transform pair exists in the evidence split.",
            "- `any_analog_block` means both transitions in a connected route block have structural analogy to some evidence block above the threshold.",
            "- `same_pair_analog_block` additionally requires the route block transform pair to be observed in the evidence split.",
            "- These are evidence/support diagnostics, not expert feasibility labels and not proof that a route is synthetically valid.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_top_ks(value: str) -> tuple[int, ...]:
    out = tuple(sorted({int(item.strip()) for item in str(value or "").split(",") if item.strip()}))
    return out or (1, 3, 5, 10, 50)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit train-split cascade evidence inside a fixed route pool")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--evidence-split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--top-ks", default="1,3,5,10,50")
    ap.add_argument("--max-examples", type=int, default=40)
    args = ap.parse_args()
    result = audit_route_pool_cascade_evidence(
        route_pool=Path(args.route_pool),
        program_manifest=Path(args.program_manifest),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        evidence_split=args.evidence_split,
        analog_similarity=float(args.analog_similarity),
        top_ks=_parse_top_ks(args.top_ks),
        max_examples=int(args.max_examples),
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "outputs": {
                    "json": str(args.output_json),
                    "md": str(args.output_md or Path(args.output_json).with_suffix(".md")),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
