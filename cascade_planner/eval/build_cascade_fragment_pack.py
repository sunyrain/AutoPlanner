"""Build fast Stage-2 cascade fragment preference packs.

This pack is intentionally small and deterministic.  It turns v4 cascade
routes into 2-step and 3-step process fragments, then creates hard negatives by
swapping one step with a same-domain step from another record.  The labels are
process-preference labels, not route-gold labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.pair_scorer import pair_rule_features
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


PACK_SCHEMA_VERSION = "cascade_fragment_pack.v1"
FRAGMENT_LABEL_NAMES = [
    "fragment_preference",
    "cascade_compatible",
    "one_pot",
    "telescoped",
    "condition_compatible",
    "cofactor_compatible",
    "isolation_required",
    "biocascade",
]


def build_cascade_fragment_pack(
    *,
    v4_jsonl: Path,
    benchmark_path: Path | None,
    output_dir: Path,
    max_window_size: int = 3,
    hard_negative_per_positive: int = 1,
    split_salt: str = "cascade_fragment_stage2_2026-05-10",
    max_rows: int | None = None,
    hard_negative_candidate_cap: int = 256,
) -> dict[str, Any]:
    excluded = _benchmark_exclusions(benchmark_path) if benchmark_path else {"keys": set(), "targets": set()}
    records: list[dict[str, Any]] = []
    skipped = Counter()
    for raw in _read_jsonl(v4_jsonl):
        record, reason = _record_payload(raw, excluded)
        if record is None:
            skipped[reason or "unknown"] += 1
            continue
        records.append(record)

    positives: list[dict[str, Any]] = []
    for record in records:
        steps = record["steps"]
        for window_size in range(2, max(2, int(max_window_size)) + 1):
            if len(steps) < window_size:
                continue
            for start in range(0, len(steps) - window_size + 1):
                row = _fragment_row(
                    record,
                    steps[start : start + window_size],
                    window_start=start,
                    window_size=window_size,
                    example_type="positive",
                )
                if row is None:
                    skipped["positive_fragment_unusable"] += 1
                    continue
                row["labels"] = _positive_labels(row)
                positives.append(row)

    positive_source = positives
    if max_rows is not None and max_rows > 0:
        max_pos = max(1, int(max_rows) // max(1, 1 + max(0, hard_negative_per_positive)))
        positive_source = positives[:max_pos]

    negatives = _hard_negatives(
        records,
        positive_source,
        per_positive=hard_negative_per_positive,
        candidate_cap=hard_negative_candidate_cap,
    )
    rows = positive_source + negatives if max_rows is not None and max_rows > 0 else positives + negatives
    rows.sort(key=lambda row: (row.get("split_group_id") or "", row.get("example_type") or "", row.get("fragment_id") or ""))
    if max_rows is not None and max_rows > 0:
        rows = _balanced_limit(rows, int(max_rows))
    splits = _assign_splits(rows, split_salt=split_salt)
    for row in rows:
        row["split"] = splits[row["split_group_id"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "fragment_value.jsonl"
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
            "max_window_size": int(max_window_size),
            "hard_negative_per_positive": int(hard_negative_per_positive),
            "hard_negative_candidate_cap": int(hard_negative_candidate_cap),
            "max_rows": max_rows,
            "label_names": list(FRAGMENT_LABEL_NAMES),
            "split_policy": "stable grouped DOI/cascade/target split; full100 excluded before splitting",
            "training_contract": "fragment_level_cascade_preference_not_route_gold.v1",
        },
        "counts": {
            "records": len(records),
            "rows": len(rows),
            "positive_rows": sum(1 for row in rows if row.get("example_type") == "positive"),
            "hard_negative_rows": sum(1 for row in rows if row.get("example_type") == "hard_negative"),
            "candidate_positive_rows_before_limit": len(positives),
            "generated_hard_negative_rows_before_limit": len(negatives),
            "skipped": dict(skipped),
        },
        "splits": _split_summary(rows),
        "label_rates": _label_rates(rows),
        "example_type_counts": dict(Counter(row.get("example_type") for row in rows)),
        "window_size_counts": dict(Counter(str(row.get("window_size")) for row in rows)),
        "route_domain_counts": dict(Counter(row.get("route_domain") for row in rows)),
        "outputs": {
            "fragment_value": str(pack_path),
            "summary": str(summary_path),
            "readme": str(readme_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(_readme(report), encoding="utf-8")
    return report


def _record_payload(raw: dict[str, Any], excluded: dict[str, set[Any]]) -> tuple[dict[str, Any] | None, str | None]:
    doi = _norm(raw.get("doi"))
    cascade_id = _norm(raw.get("cascade_id"))
    if (doi, cascade_id) in excluded["keys"]:
        return None, "benchmark_key_overlap"
    if not _truthy(raw.get("trainable_recommended")):
        return None, "not_trainable_recommended"
    target = canonical_smiles(str(raw.get("target_product_smiles") or "")) or str(raw.get("target_product_smiles") or "").strip()
    if target and target in excluded["targets"]:
        return None, "benchmark_target_overlap"
    steps = [_compact_step(step) for step in raw.get("steps") or [] if isinstance(step, dict) and step.get("rxn_smiles")]
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


def _fragment_row(
    record: dict[str, Any],
    steps: list[dict[str, Any]],
    *,
    window_start: int,
    window_size: int,
    example_type: str,
    negative_reason: str = "",
) -> dict[str, Any] | None:
    if len(steps) < 2 or any(not step.get("rxn_smiles") for step in steps):
        return None
    group_id = _stable_id(record.get("doi"), record.get("cascade_id"), record.get("target_smiles"))
    fragment_id = _stable_id(
        group_id,
        window_start,
        window_size,
        example_type,
        negative_reason,
        *[step.get("rxn_smiles") for step in steps],
    )
    pair_rows = []
    for idx, (left, right) in enumerate(zip(steps, steps[1:])):
        pair = {
            "route_domain": record.get("route_domain") or "unknown",
            "left_step": left,
            "right_step": right,
            "shared_intermediate": _shared_intermediate(left.get("rxn_smiles"), right.get("rxn_smiles")),
            "left_pairwise_mode": left.get("pairwise_mode") or "unknown",
            "right_pairwise_mode": right.get("pairwise_mode") or "unknown",
            "pair_index": idx,
        }
        pair["rule_features"] = pair_rule_features(pair)
        pair_rows.append(pair)
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "fragment_id": fragment_id,
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
        "window_start": int(window_start),
        "window_size": int(window_size),
        "steps": steps,
        "pair_rows": pair_rows,
        "labels": {},
    }


def _positive_labels(row: dict[str, Any]) -> dict[str, float]:
    label = _norm(row.get("compatibility_label"))
    explicit_bad = label in {"empirically_incompatible", "sequential_preferred"}
    unclear = label in {"", "unclear"}
    base = 0.80
    if label == "empirically_compatible":
        base = 1.0
    elif label in {"compatible_with_mitigation", "compatible_with_compromise"}:
        base = 0.75
    elif label == "sequential_preferred":
        base = 0.35
    elif label == "empirically_incompatible":
        base = 0.05
    elif unclear:
        base = 0.55
    pair_features = [pair.get("rule_features") or {} for pair in row.get("pair_rows") or []]
    isolation = max([float(feat.get("rule_isolation_need") or 0.0) for feat in pair_features] or [0.0])
    condition = min([float(feat.get("temp_overlap", 0.55)) * 0.45 + float(feat.get("ph_overlap", 0.55)) * 0.35 + float(feat.get("solvent_match", 0.55)) * 0.20 for feat in pair_features] or [0.55])
    cofactor_ok = 0.0 if any(float(feat.get("cofactor_conflict") or 0.0) > 0.0 for feat in pair_features) else 1.0
    modes = [_norm(pair.get("left_pairwise_mode")) for pair in row.get("pair_rows") or []]
    one_pot = float(modes and all(mode == "simultaneous" for mode in modes) and not explicit_bad)
    telescoped = float(any(mode in {"telescoped", "sequential_addition", "compartmentalized"} for mode in modes) and not explicit_bad)
    biocascade = float(pair_features and all(float(feat.get("both_enzymatic") or 0.0) > 0.0 for feat in pair_features))
    preference = max(0.0, min(1.0, base - 0.25 * isolation + 0.10 * min(condition, 1.0) + 0.05 * cofactor_ok))
    if explicit_bad:
        one_pot = 0.0
        preference = min(preference, 0.35)
    return {
        "fragment_preference": float(preference),
        "cascade_compatible": float(preference >= 0.65),
        "one_pot": one_pot,
        "telescoped": telescoped,
        "condition_compatible": float(max(0.0, min(1.0, condition))),
        "cofactor_compatible": cofactor_ok,
        "isolation_required": float(max(isolation, 1.0 if explicit_bad else 0.0)),
        "biocascade": biocascade,
        "label_weight": 0.45 if unclear else 1.0,
    }


def _negative_labels(row: dict[str, Any], *, reason: str) -> dict[str, float]:
    pair_features = [pair.get("rule_features") or {} for pair in row.get("pair_rows") or []]
    condition = min([float(feat.get("rule_compatibility") or 0.0) for feat in pair_features] or [0.0])
    cofactor_ok = 0.0 if any(float(feat.get("cofactor_conflict") or 0.0) > 0.0 for feat in pair_features) else 0.45
    severe = reason in {"cofactor_conflict", "redox_conflict", "enzyme_chemical_condition_conflict", "route_order_mismatch"}
    return {
        "fragment_preference": 0.05 if severe else 0.20,
        "cascade_compatible": 0.0,
        "one_pot": 0.0,
        "telescoped": 0.05 if condition > 0.4 else 0.0,
        "condition_compatible": max(0.0, min(0.35, condition)),
        "cofactor_compatible": cofactor_ok,
        "isolation_required": 1.0 if severe else 0.75,
        "biocascade": 0.0,
        "label_weight": 0.75,
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
            steps_by_domain[str(record.get("route_domain") or "unknown")].append((record, step))
    negatives = []
    for pos in positives:
        candidates = steps_by_domain.get(str(pos.get("route_domain") or "unknown")) or sum(steps_by_domain.values(), [])
        candidates = _stable_candidate_sample(candidates, seed=pos.get("fragment_id"), cap=candidate_cap)
        chosen = _choose_negative_steps(pos, candidates, per_positive=per_positive)
        for idx, (candidate_record, replacement, reason) in enumerate(chosen):
            steps = [dict(step) for step in pos.get("steps") or []]
            if not steps:
                continue
            replace_idx = min(len(steps) - 1, max(1, len(steps) // 2))
            steps[replace_idx] = _compact_step(replacement)
            record = {
                "doi": pos.get("doi"),
                "cascade_id": pos.get("cascade_id"),
                "target_smiles": pos.get("target_smiles"),
                "route_domain": pos.get("route_domain"),
                "quality_tier": pos.get("quality_tier"),
                "compatibility": {"compatibility_label": "hard_negative"},
            }
            row = _fragment_row(
                record,
                steps,
                window_start=int(pos.get("window_start") or 0),
                window_size=int(pos.get("window_size") or len(steps)),
                example_type="hard_negative",
                negative_reason=f"{reason}:{candidate_record.get('cascade_id') or idx}",
            )
            if row is None:
                continue
            row["labels"] = _negative_labels(row, reason=reason)
            negatives.append(row)
    return negatives


def _choose_negative_steps(
    pos: dict[str, Any],
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    per_positive: int,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    steps = pos.get("steps") or []
    left = steps[0] if steps else {}
    scored = []
    for record, step in candidates:
        if record.get("doi") == pos.get("doi") and record.get("cascade_id") == pos.get("cascade_id"):
            continue
        if _stable_id(step.get("rxn_smiles")) in {_stable_id(item.get("rxn_smiles")) for item in steps}:
            continue
        probe = {
            "route_domain": pos.get("route_domain"),
            "left_step": left,
            "right_step": _compact_step(step),
            "left_pairwise_mode": "hard_negative",
            "right_pairwise_mode": step.get("pairwise_mode") or "unknown",
        }
        features = pair_rule_features(probe)
        reason = _negative_reason(features)
        hardness = (
            2.0 * float(features.get("mixed_chemo_enzymatic", 0.0))
            + 1.5 * float(features.get("redox_conflict", 0.0))
            + 1.5 * float(features.get("cofactor_conflict", 0.0))
            + 1.0 * (1.0 - float(features.get("solvent_match", 0.5)))
            + 0.5 * (1.0 - float(features.get("temp_overlap", 0.55)))
        )
        scored.append((hardness, record, step, reason))
    scored.sort(key=lambda item: (-item[0], _stable_id(pos.get("fragment_id"), item[2].get("rxn_smiles"))))
    return [(record, step, reason) for _, record, step, reason in scored[:per_positive]]


def _negative_reason(features: dict[str, float]) -> str:
    if features.get("cofactor_conflict"):
        return "cofactor_conflict"
    if features.get("redox_conflict"):
        return "redox_conflict"
    if features.get("mixed_chemo_enzymatic") and features.get("solvent_match", 0.5) <= 0.0:
        return "enzyme_chemical_condition_conflict"
    if features.get("temp_overlap", 0.55) <= 0.0 or features.get("ph_overlap", 0.55) <= 0.0:
        return "condition_range_conflict"
    return "route_order_mismatch"


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
    selected.sort(key=lambda row: (row.get("split_group_id") or "", row.get("example_type") or "", row.get("fragment_id") or ""))
    return selected


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        out[split] = {
            "rows": len(split_rows),
            "groups": len({row.get("split_group_id") for row in split_rows}),
            "example_type_counts": dict(Counter(row.get("example_type") for row in split_rows)),
            "window_size_counts": dict(Counter(str(row.get("window_size")) for row in split_rows)),
            "route_domain_counts": dict(Counter(row.get("route_domain") for row in split_rows)),
        }
    return out


def _label_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for name in FRAGMENT_LABEL_NAMES:
        values = [float((row.get("labels") or {}).get(name) or 0.0) for row in rows]
        out[name] = round(sum(values) / len(values), 6) if values else 0.0
    return out


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
        "# Cascade Fragment Preference Pack",
        "",
        "This pack trains the Stage-2 2/3-step fragment preference scorer.",
        "It is not a route-level gold classifier and does not use full100 as training data.",
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
    ap = argparse.ArgumentParser(description="Build 2/3-step cascade fragment preference pack")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-window-size", type=int, default=3)
    ap.add_argument("--hard-negative-per-positive", type=int, default=1)
    ap.add_argument("--hard-negative-candidate-cap", type=int, default=256)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--split-salt", default="cascade_fragment_stage2_2026-05-10")
    args = ap.parse_args()
    report = build_cascade_fragment_pack(
        v4_jsonl=Path(args.v4_jsonl),
        benchmark_path=Path(args.benchmark) if args.benchmark else None,
        output_dir=Path(args.output_dir),
        max_window_size=args.max_window_size,
        hard_negative_per_positive=args.hard_negative_per_positive,
        split_salt=args.split_salt,
        max_rows=args.max_rows,
        hard_negative_candidate_cap=args.hard_negative_candidate_cap,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
