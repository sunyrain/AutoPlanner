"""Build Stage-1 adjacent-step cascade compatibility training packs.

The pack is built from raw v4 cascade records, not from ChemEnzy traces.  Each
row represents two adjacent process steps plus multi-task compatibility labels.
Held-out full100 benchmarks are excluded by DOI/cascade id and target overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.pair_scorer import PAIR_LABEL_NAMES, pair_rule_features
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


PACK_SCHEMA_VERSION = "cascade_pair_pack.v1"


def build_cascade_pair_pack(
    *,
    v4_jsonl: Path,
    benchmark_path: Path | None,
    output_dir: Path,
    hard_negative_per_positive: int = 1,
    split_salt: str = "cascade_pair_stage1_2026-05-10",
    include_weak_unclear: bool = True,
    max_rows: int | None = None,
    hard_negative_candidate_cap: int = 256,
) -> dict[str, Any]:
    excluded = _benchmark_exclusions(benchmark_path) if benchmark_path else {"keys": set(), "targets": set()}
    records = []
    skipped = Counter()
    for raw in _read_jsonl(v4_jsonl):
        payload, reason = _record_payload(raw, excluded)
        if payload is None:
            skipped[reason or "unknown"] += 1
            continue
        records.append(payload)
    positives = []
    for record in records:
        for idx, (left, right) in enumerate(zip(record["steps"], record["steps"][1:])):
            row = _pair_row(record, left, right, pair_index=idx, example_type="positive")
            if row is None:
                skipped["positive_pair_unusable"] += 1
                continue
            labels = _positive_labels(row, include_weak_unclear=include_weak_unclear)
            if labels is None:
                skipped["positive_label_unclear_excluded"] += 1
                continue
            row["labels"] = labels
            positives.append(row)
    if max_rows is not None and max_rows > 0:
        max_positive_rows = max(1, int(max_rows) // max(1, 1 + max(0, hard_negative_per_positive)))
        positives_for_negatives = positives[:max_positive_rows]
    else:
        positives_for_negatives = positives
    negative_positive_source = positives_for_negatives
    negatives = _hard_negatives(
        records,
        negative_positive_source,
        per_positive=hard_negative_per_positive,
        candidate_cap=hard_negative_candidate_cap,
    )
    rows = negative_positive_source + negatives if max_rows is not None and max_rows > 0 else positives + negatives
    rows.sort(key=lambda row: (row.get("doi") or "", row.get("cascade_id") or "", row.get("example_type") or "", row.get("pair_id") or ""))
    if max_rows is not None and max_rows > 0:
        rows = _balanced_limit(rows, int(max_rows))
    output_positive_rows = [row for row in rows if row.get("example_type") == "positive"]
    output_negative_rows = [row for row in rows if row.get("example_type") == "hard_negative"]
    splits = _assign_splits(rows, split_salt=split_salt)
    for row in rows:
        row["split"] = splits[row["split_group_id"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "pair_value.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    _write_jsonl(pack_path, rows)
    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "benchmark_path": str(benchmark_path) if benchmark_path else None,
            "output_dir": str(output_dir),
            "hard_negative_per_positive": hard_negative_per_positive,
            "include_weak_unclear": include_weak_unclear,
            "max_rows": max_rows,
            "hard_negative_candidate_cap": hard_negative_candidate_cap,
            "label_names": list(PAIR_LABEL_NAMES),
            "split_policy": "grouped stable split by DOI/cascade/target/scaffold; full100 excluded before splitting",
            "training_contract": "adjacent_step_pair_cascade_preference_not_route_gold.v1",
        },
        "counts": {
            "records": len(records),
            "rows": len(rows),
            "positive_rows": len(output_positive_rows),
            "hard_negative_rows": len(output_negative_rows),
            "candidate_positive_rows_before_limit": len(positives),
            "generated_hard_negative_rows_before_limit": len(negatives),
            "skipped": dict(skipped),
        },
        "splits": _split_summary(rows),
        "label_rates": _label_rates(rows),
        "example_type_counts": dict(Counter(row.get("example_type") for row in rows)),
        "route_domain_counts": dict(Counter(row.get("route_domain") for row in rows)),
        "outputs": {
            "pair_value": str(pack_path),
            "summary": str(summary_path),
            "readme": str(readme_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(_readme(report), encoding="utf-8")
    return report


def _record_payload(raw: dict[str, Any], excluded: dict[str, set[str]]) -> tuple[dict[str, Any] | None, str | None]:
    doi = _norm(raw.get("doi"))
    cascade_id = _norm(raw.get("cascade_id"))
    if (doi, cascade_id) in excluded["keys"]:
        return None, "benchmark_key_overlap"
    if not _truthy(raw.get("trainable_recommended")):
        return None, "not_trainable_recommended"
    target = canonical_smiles(str(raw.get("target_product_smiles") or "")) or str(raw.get("target_product_smiles") or "").strip()
    if target and target in excluded["targets"]:
        return None, "benchmark_target_overlap"
    steps = [dict(step) for step in raw.get("steps") or [] if isinstance(step, dict)]
    if len(steps) < 2:
        return None, "fewer_than_two_steps"
    return {
        "doi": raw.get("doi"),
        "cascade_id": raw.get("cascade_id"),
        "target_smiles": target,
        "route_domain": raw.get("cascade_type") or raw.get("route_domain") or "unknown",
        "compatibility": raw.get("compatibility") or {},
        "quality_tier": raw.get("quality_tier"),
        "steps": steps,
    }, None


def _pair_row(
    record: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    pair_index: int,
    example_type: str,
    negative_reason: str = "",
) -> dict[str, Any] | None:
    if not left.get("rxn_smiles") or not right.get("rxn_smiles"):
        return None
    group_id = _stable_id(record.get("doi"), record.get("cascade_id"), record.get("target_smiles"))
    pair_id = _stable_id(group_id, pair_index, example_type, negative_reason, left.get("rxn_smiles"), right.get("rxn_smiles"))
    row = {
        "schema_version": PACK_SCHEMA_VERSION,
        "pair_id": pair_id,
        "split_group_id": group_id,
        "example_type": example_type,
        "negative_reason": negative_reason,
        "doi": record.get("doi"),
        "cascade_id": record.get("cascade_id"),
        "target_smiles": record.get("target_smiles"),
        "route_domain": record.get("route_domain") or "unknown",
        "quality_tier": record.get("quality_tier"),
        "compatibility_label": (record.get("compatibility") or {}).get("compatibility_label"),
        "compatibility_evidence_strength": (record.get("compatibility") or {}).get("evidence_strength"),
        "issue_types": list((record.get("compatibility") or {}).get("issue_types") or []),
        "mitigation_strategies": list((record.get("compatibility") or {}).get("mitigation_strategies") or []),
        "left_pairwise_mode": left.get("pairwise_mode") or "unknown",
        "right_pairwise_mode": right.get("pairwise_mode") or "unknown",
        "left_step": _compact_step(left),
        "right_step": _compact_step(right),
        "shared_intermediate": _shared_intermediate(left.get("rxn_smiles"), right.get("rxn_smiles")),
        "labels": {},
    }
    row["rule_features"] = pair_rule_features(row)
    return row


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "rxn_smiles": step.get("rxn_smiles"),
        "pairwise_mode": step.get("pairwise_mode"),
        "step_mode": step.get("step_mode"),
        "transformation_name": step.get("transformation_name"),
        "transformation_superclass": step.get("transformation_superclass"),
        "intermediate_isolated": step.get("intermediate_isolated"),
        "step_conditions": step.get("step_conditions") or {},
        "catalyst_components": [
            {
                "catalyst_class": item.get("catalyst_class"),
                "component_name": item.get("component_name"),
                "ec_number": item.get("ec_number"),
                "cofactor_required": item.get("cofactor_required"),
                "cofactor_regeneration_mode": item.get("cofactor_regeneration_mode"),
                "metal_identity": item.get("metal_identity"),
                "organism": item.get("organism"),
            }
            for item in step.get("catalyst_components") or []
            if isinstance(item, dict)
        ],
    }


def _positive_labels(row: dict[str, Any], *, include_weak_unclear: bool) -> dict[str, float] | None:
    label = _norm(row.get("compatibility_label"))
    left_mode = _norm(row.get("left_pairwise_mode"))
    right_mode = _norm(row.get("right_pairwise_mode"))
    features = row.get("rule_features") or {}
    explicit_bad = label in {"empirically_incompatible", "sequential_preferred"}
    unclear = label in {"", "unclear"}
    if unclear and not include_weak_unclear:
        return None
    one_pot = float(left_mode == "simultaneous" and not explicit_bad)
    telescoped = float(left_mode in {"telescoped", "sequential_addition", "compartmentalized"} and not explicit_bad)
    isolation = float(left_mode == "isolated_transfer" or explicit_bad or features.get("intermediate_isolated", 0.0) > 0.0)
    compatibility = 0.80
    if label == "empirically_compatible":
        compatibility = 1.0
    elif label in {"compatible_with_mitigation", "compatible_with_compromise"}:
        compatibility = 0.75
    elif label == "sequential_preferred":
        compatibility = 0.35
    elif label == "empirically_incompatible":
        compatibility = 0.05
    elif unclear:
        compatibility = 0.55
    if explicit_bad:
        one_pot = 0.0
        telescoped = max(telescoped, 0.35)
    return {
        "compatibility": compatibility,
        "one_pot": one_pot,
        "telescoped": telescoped,
        "condition_compatible": float(max(0.0, min(1.0, features.get("temp_overlap", 0.55) * 0.45 + features.get("ph_overlap", 0.55) * 0.35 + features.get("solvent_match", 0.55) * 0.20))),
        "cofactor_compatible": float(0.0 if features.get("cofactor_conflict") else 1.0),
        "isolation_required": isolation,
        "biocascade": float(features.get("both_enzymatic", 0.0)),
        "label_weight": 0.45 if unclear else 1.0,
    }


def _hard_negatives(
    records: list[dict[str, Any]],
    positives: list[dict[str, Any]],
    *,
    per_positive: int,
    candidate_cap: int,
) -> list[dict[str, Any]]:
    if per_positive <= 0 or not positives:
        return []
    steps_by_domain: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        for step in record["steps"]:
            if step.get("rxn_smiles"):
                steps_by_domain[str(record.get("route_domain") or "unknown")].append((record, step))
    negatives = []
    for pos in positives:
        domain = str(pos.get("route_domain") or "unknown")
        candidates = steps_by_domain.get(domain) or sum(steps_by_domain.values(), [])
        candidates = _stable_candidate_sample(candidates, seed=pos.get("pair_id"), cap=candidate_cap)
        chosen = _choose_negative_steps(pos, candidates, per_positive=per_positive)
        for idx, (record, right_step, reason) in enumerate(chosen):
            left = pos["left_step"]
            row = _pair_row(
                {
                    "doi": pos.get("doi"),
                    "cascade_id": pos.get("cascade_id"),
                    "target_smiles": pos.get("target_smiles"),
                    "route_domain": pos.get("route_domain"),
                    "quality_tier": pos.get("quality_tier"),
                    "compatibility": {"compatibility_label": "hard_negative"},
                },
                left,
                right_step,
                pair_index=idx,
                example_type="hard_negative",
                negative_reason=reason,
            )
            if row is None:
                continue
            row["labels"] = _negative_labels(row, reason=reason)
            negatives.append(row)
    return negatives


def _stable_candidate_sample(
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    seed: Any,
    cap: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cap = int(cap or 0)
    if cap <= 0 or len(candidates) <= cap:
        return candidates
    return sorted(
        candidates,
        key=lambda item: _stable_id(seed, item[0].get("doi"), item[0].get("cascade_id"), item[1].get("rxn_smiles")),
    )[:cap]


def _choose_negative_steps(
    pos: dict[str, Any],
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    per_positive: int,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    left = pos["left_step"]
    left_key = _stable_id(left.get("rxn_smiles"))
    scored = []
    for record, step in candidates:
        if _stable_id(step.get("rxn_smiles")) == left_key:
            continue
        if record.get("doi") == pos.get("doi") and record.get("cascade_id") == pos.get("cascade_id"):
            continue
        probe = {
            "route_domain": pos.get("route_domain"),
            "left_step": left,
            "right_step": _compact_step(step),
            "left_pairwise_mode": "hard_negative",
            "right_pairwise_mode": step.get("pairwise_mode") or "unknown",
        }
        features = pair_rule_features(probe)
        reason = _negative_reason(features, left, step)
        hardness = (
            2.0 * float(features.get("mixed_chemo_enzymatic", 0.0))
            + 1.5 * float(features.get("redox_conflict", 0.0))
            + 1.5 * float(features.get("cofactor_conflict", 0.0))
            + 1.0 * (1.0 - float(features.get("solvent_match", 0.5)))
            + 0.5 * (1.0 - float(features.get("temp_overlap", 0.55)))
        )
        scored.append((hardness, record, step, reason))
    scored.sort(key=lambda item: (-item[0], _stable_id(pos.get("pair_id"), item[2].get("rxn_smiles"))))
    return [(record, step, reason) for _, record, step, reason in scored[:per_positive]]


def _negative_reason(features: dict[str, float], left: dict[str, Any], right: dict[str, Any]) -> str:
    if features.get("cofactor_conflict"):
        return "cofactor_conflict"
    if features.get("redox_conflict"):
        return "redox_conflict"
    if features.get("mixed_chemo_enzymatic") and features.get("solvent_match", 0.5) <= 0.0:
        return "enzyme_chemical_condition_conflict"
    if features.get("temp_overlap", 0.55) <= 0.0 or features.get("ph_overlap", 0.55) <= 0.0:
        return "condition_range_conflict"
    return "mismatched_adjacent_process"


def _negative_labels(row: dict[str, Any], *, reason: str) -> dict[str, float]:
    features = row.get("rule_features") or {}
    condition_ok = float(features.get("temp_overlap", 0.55) > 0.0 and features.get("ph_overlap", 0.55) > 0.0 and features.get("solvent_match", 0.55) > 0.0)
    return {
        "compatibility": 0.0,
        "one_pot": 0.0,
        "telescoped": 0.05 if condition_ok else 0.0,
        "condition_compatible": condition_ok * 0.35,
        "cofactor_compatible": float(0.0 if features.get("cofactor_conflict") else 0.50),
        "isolation_required": 1.0 if reason != "mismatched_adjacent_process" else 0.75,
        "biocascade": 0.0,
        "label_weight": 0.75,
    }


def _assign_splits(rows: list[dict[str, Any]], *, split_salt: str) -> dict[str, str]:
    groups = sorted({row["split_group_id"] for row in rows})
    out = {}
    for group in groups:
        value = int(hashlib.sha1(f"{split_salt}\t{group}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if value < 0.10:
            split = "test"
        elif value < 0.20:
            split = "val"
        else:
            split = "train"
        out[group] = split
    if rows and "val" not in out.values():
        out[groups[0]] = "val"
    if len(groups) > 2 and "test" not in out.values():
        out[groups[-1]] = "test"
    return out


def _balanced_limit(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    positives = [row for row in rows if row.get("example_type") == "positive"]
    negatives = [row for row in rows if row.get("example_type") == "hard_negative"]
    if not positives or not negatives:
        return rows[:max_rows]
    n_neg = min(len(negatives), max(1, max_rows // 2))
    n_pos = max_rows - n_neg
    selected = positives[:n_pos] + negatives[:n_neg]
    selected.sort(key=lambda row: (row.get("doi") or "", row.get("cascade_id") or "", row.get("example_type") or "", row.get("pair_id") or ""))
    return selected


def _benchmark_exclusions(path: Path | None) -> dict[str, set[Any]]:
    keys = set()
    targets = set()
    if path is None or not path.exists():
        return {"keys": keys, "targets": targets}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    if not isinstance(rows, list):
        return {"keys": keys, "targets": targets}
    for row in rows:
        if not isinstance(row, dict):
            continue
        keys.add((_norm(row.get("doi")), _norm(row.get("cascade_id"))))
        target = canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "").strip()
        if target:
            targets.add(target)
    return {"keys": keys, "targets": targets}


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        out[split] = {
            "rows": len(split_rows),
            "groups": len({row.get("split_group_id") for row in split_rows}),
            "example_type_counts": dict(Counter(row.get("example_type") for row in split_rows)),
            "route_domain_counts": dict(Counter(row.get("route_domain") for row in split_rows)),
        }
    return out


def _label_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for name in PAIR_LABEL_NAMES:
        values = [float((row.get("labels") or {}).get(name) or 0.0) for row in rows]
        out[name] = round(sum(values) / len(values), 6) if values else 0.0
    return out


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


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _readme(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Cascade Pair Compatibility Pack",
        "",
        "This pack trains the Stage-1 adjacent-step cascade compatibility scorer.",
        "It is not a route-level gold classifier and it is not built from full100.",
        "",
        "## Counts",
        "",
        f"- rows: {counts['rows']}",
        f"- positives: {counts['positive_rows']}",
        f"- hard negatives: {counts['hard_negative_rows']}",
        "",
        "## Contract",
        "",
        report["metadata"]["training_contract"],
    ]
    return "\n".join(lines) + "\n"


def _stable_id(*values: Any) -> str:
    return hashlib.sha1("\t".join(str(value or "") for value in values).encode("utf-8")).hexdigest()[:16]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _norm(value) in {"1", "true", "yes", "y"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build adjacent-step cascade compatibility pack")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hard-negative-per-positive", type=int, default=1)
    ap.add_argument("--hard-negative-candidate-cap", type=int, default=256)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--exclude-weak-unclear", action="store_true")
    args = ap.parse_args()
    report = build_cascade_pair_pack(
        v4_jsonl=Path(args.v4_jsonl),
        benchmark_path=Path(args.benchmark) if args.benchmark else None,
        output_dir=Path(args.output_dir),
        hard_negative_per_positive=args.hard_negative_per_positive,
        hard_negative_candidate_cap=args.hard_negative_candidate_cap,
        include_weak_unclear=not args.exclude_weak_unclear,
        max_rows=args.max_rows,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
