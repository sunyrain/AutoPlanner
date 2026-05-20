"""Build a transform-pair selector pack from ChemEnzy connectors and v4 labels."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.eval.audit_nonoracle_provider_bridge import (
    _cache_index,
    _downstream_candidates,
    _fp_similarity,
    _label_downstream_candidates,
    _read_json,
    _reference_connection_similarity,
    _target_list,
)
from cascade_planner.eval.audit_provider_routepool_oracle import _load_reference_blocks, _transition_fp


SCHEMA_VERSION = "v4_transform_pair_selector_pack.v1"


def build_v4_transform_pair_selector_pack(
    *,
    program_manifest: Path,
    chem_enzy_cache: Path,
    output_jsonl: Path,
    report_json: Path,
    split: str = "test",
    route_recovery_json: Path | None = None,
    only_routed: bool = False,
    max_targets: int | None = None,
    max_chem_candidates: int = 100,
    main_reactant_only: bool = True,
    min_connector_heavy_atoms: int = 6,
    connected_ref_similarity: float = 0.55,
    connector_label_similarity: float = 0.55,
    top_pairs: int = 80,
    candidate_pair_split: str = "train",
) -> dict[str, Any]:
    started = time.monotonic()
    cache = _read_json(chem_enzy_cache)
    cache_index = _cache_index(cache)
    refs = _load_reference_blocks(program_manifest, split=split)
    refs_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        if connected_ref_similarity > 0.0 and _reference_connection_similarity(ref) < float(connected_ref_similarity):
            continue
        target = canonical_smiles(str(ref.get("target_smiles") or "")) or str(ref.get("target_smiles") or "")
        if not target:
            continue
        refs_by_target[target].append(ref)

    candidate_pair_counts = _candidate_pair_counts(
        program_manifest=program_manifest,
        split=candidate_pair_split,
        connected_ref_similarity=connected_ref_similarity,
    )
    label_pair_counts = Counter()
    for target_refs in refs_by_target.values():
        for ref in target_refs:
            label_pair_counts[str(ref.get("transform_pair") or "")] += 1
    candidate_pairs = [pair for pair, _ in candidate_pair_counts.most_common(max(1, int(top_pairs)))]
    targets = _target_list(
        refs_by_target=refs_by_target,
        route_recovery_json=route_recovery_json,
        only_routed=only_routed,
        cache_index=cache_index,
    )
    if max_targets is not None:
        targets = targets[: max(0, int(max_targets))]

    rows = []
    target_summaries = []
    for target in targets:
        refs_for_target = refs_by_target.get(target) or []
        true_pairs = {str(ref.get("transform_pair") or "") for ref in refs_for_target if ref.get("transform_pair")}
        downstream_candidates = _downstream_candidates(
            cache_index,
            target,
            max_chem_candidates=max_chem_candidates,
            main_reactant_only=main_reactant_only,
            min_connector_heavy_atoms=min_connector_heavy_atoms,
        )
        downstream_label = _label_downstream_candidates(downstream_candidates, refs_for_target, analog_similarity=connected_ref_similarity)
        target_rows = []
        for connector_rank, connector in enumerate(downstream_candidates, start=1):
            connector_pair_labels = _connector_pair_labels(
                target=target,
                connector=connector,
                refs_for_target=refs_for_target,
                connector_label_similarity=connector_label_similarity,
            )
            connector_true_pairs = set(connector_pair_labels)
            best_ref_similarity = max((float(row.get("similarity") or 0.0) for row in connector_pair_labels.values()), default=0.0)
            for pair_rank, pair in enumerate(candidate_pairs, start=1):
                upstream, downstream = _split_pair(pair)
                matched_ref = connector_pair_labels.get(pair)
                row = {
                    "target_smiles": target,
                    "split": split,
                    "connector": connector.get("connector"),
                    "connector_rank": connector_rank,
                    "downstream_rank": connector.get("downstream_rank"),
                    "connector_heavy_atoms": connector.get("connector_heavy_atoms"),
                    "connector_is_main_reactant": connector.get("is_main_reactant"),
                    "candidate_score": connector.get("candidate_score"),
                    "candidate_source": connector.get("candidate_source"),
                    "candidate_model": connector.get("candidate_model"),
                    "transform_pair": pair,
                    "transform_pair_rank_by_train_frequency": pair_rank,
                    "upstream_transform": upstream,
                    "downstream_transform": downstream,
                    "label": pair in connector_true_pairs,
                    "target_level_label": pair in true_pairs,
                    "connector_downstream_best_ref_similarity": round(float(best_ref_similarity), 6),
                    "connector_matched_reference_block_id": (matched_ref or {}).get("block_id"),
                    "connector_matched_reference_similarity": (matched_ref or {}).get("similarity"),
                    "target_true_pairs": sorted(true_pairs),
                    "connector_true_pairs": sorted(connector_true_pairs),
                    "downstream_analog_any_for_target": bool(downstream_label.get("analog_any")),
                }
                rows.append(row)
                target_rows.append(row)
        target_summaries.append(
            {
                "target_smiles": target,
                "reference_blocks": len(refs_for_target),
                "true_pairs": sorted(true_pairs),
                "connectors": len(downstream_candidates),
                "connectors_with_pair_labels": sum(
                    1
                    for connector in downstream_candidates
                    if _connector_pair_labels(
                        target=target,
                        connector=connector,
                        refs_for_target=refs_for_target,
                        connector_label_similarity=connector_label_similarity,
                    )
                ),
                "rows": len(target_rows),
                "positive_rows": sum(1 for row in target_rows if row["label"]),
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "chem_enzy_cache": str(chem_enzy_cache),
            "output_jsonl": str(output_jsonl),
            "split": split,
            "route_recovery_json": str(route_recovery_json) if route_recovery_json else None,
            "only_routed": bool(only_routed),
            "max_targets": max_targets,
            "max_chem_candidates": max_chem_candidates,
            "main_reactant_only": bool(main_reactant_only),
            "min_connector_heavy_atoms": min_connector_heavy_atoms,
            "connected_ref_similarity": connected_ref_similarity,
            "connector_label_similarity": connector_label_similarity,
            "top_pairs": top_pairs,
            "candidate_pair_split": candidate_pair_split,
            "candidate_pair_contract": (
                "Candidate transform-pair vocabulary and frequency ranks are "
                "computed only from candidate_pair_split, default train. The "
                "label split references are used only to assign labels."
            ),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "summary": {
            "targets": len(targets),
            "rows": len(rows),
            "positive_rows": sum(1 for row in rows if row["label"]),
            "target_level_positive_rows": sum(1 for row in rows if row.get("target_level_label")),
            "candidate_pairs": len(candidate_pairs),
            "targets_with_positive_rows": sum(1 for row in target_summaries if row["positive_rows"]),
            "connectors_with_pair_labels": sum(row["connectors_with_pair_labels"] for row in target_summaries),
            "connectors": sum(row["connectors"] for row in target_summaries),
        },
        "candidate_pairs": candidate_pairs,
        "top_candidate_pair_source_pairs": dict(candidate_pair_counts.most_common(20)),
        "top_label_split_pairs": dict(label_pair_counts.most_common(20)),
        "targets": target_summaries,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _candidate_pair_counts(*, program_manifest: Path, split: str, connected_ref_similarity: float) -> Counter:
    counts = Counter()
    for ref in _load_reference_blocks(program_manifest, split=split):
        if connected_ref_similarity > 0.0 and _reference_connection_similarity(ref) < float(connected_ref_similarity):
            continue
        pair = str(ref.get("transform_pair") or "")
        if pair:
            counts[pair] += 1
    return counts


def _split_pair(pair: str) -> tuple[str, str]:
    left, sep, right = str(pair or "").partition("->")
    return (left.strip().lower() or "unknown", right.strip().lower() if sep else "")


def _connector_pair_labels(
    *,
    target: str,
    connector: dict[str, Any],
    refs_for_target: list[dict[str, Any]],
    connector_label_similarity: float,
) -> dict[str, dict[str, Any]]:
    connector_fp = _transition_fp(target, connector.get("connector"))
    out: dict[str, dict[str, Any]] = {}
    for ref in refs_for_target:
        ref_fp = _transition_fp(ref.get("downstream_product"), ref.get("downstream_main_reactant"))
        sim = _fp_similarity(connector_fp, ref_fp)
        if sim < float(connector_label_similarity):
            continue
        pair = str(ref.get("transform_pair") or "")
        previous = out.get(pair)
        if previous is None or sim > float(previous.get("similarity") or 0.0):
            out[pair] = {
                "block_id": ref.get("block_id"),
                "similarity": round(float(sim), 6),
            }
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v4 Transform-Pair Selector Pack",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Top Candidate-Pair Source Pairs",
            "",
            "```json",
            json.dumps(report.get("top_candidate_pair_source_pairs") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Top Label-Split Pairs",
            "",
            "```json",
            json.dumps(report.get("top_label_split_pairs") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 transform-pair selector pack")
    parser.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    parser.add_argument("--chem-enzy-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--route-recovery-json")
    parser.add_argument("--only-routed", action="store_true")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-chem-candidates", type=int, default=100)
    parser.add_argument("--include-all-reactants", action="store_true")
    parser.add_argument("--min-connector-heavy-atoms", type=int, default=6)
    parser.add_argument("--connected-ref-similarity", type=float, default=0.55)
    parser.add_argument("--connector-label-similarity", type=float, default=0.55)
    parser.add_argument("--top-pairs", type=int, default=80)
    parser.add_argument("--candidate-pair-split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    report = build_v4_transform_pair_selector_pack(
        program_manifest=Path(args.program_manifest),
        chem_enzy_cache=Path(args.chem_enzy_cache),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        split=args.split,
        route_recovery_json=Path(args.route_recovery_json) if args.route_recovery_json else None,
        only_routed=args.only_routed,
        max_targets=args.max_targets,
        max_chem_candidates=args.max_chem_candidates,
        main_reactant_only=not args.include_all_reactants,
        min_connector_heavy_atoms=args.min_connector_heavy_atoms,
        connected_ref_similarity=args.connected_ref_similarity,
        connector_label_similarity=args.connector_label_similarity,
        top_pairs=args.top_pairs,
        candidate_pair_split=args.candidate_pair_split,
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl, "report_json": args.report_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
