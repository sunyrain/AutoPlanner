"""Build pairwise preferences for the v4 cascade product-value model."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PREFERENCE_SCHEMA_VERSION = "v4_cascade_product_preference_pack.v1"


def build_v4_cascade_preference_pack(
    *,
    feature_pack: Path,
    output: Path,
    listwise_output: Path | None = None,
    negative_feature_output: Path | None = None,
    preference_mode: str = "observed_corruptions",
    max_pairs_per_route: int = 4,
    max_pairs: int | None = None,
    corruptions_per_route: int = 3,
    max_listwise_routes_per_group: int = 32,
) -> dict[str, Any]:
    rows = _read_jsonl(feature_pack)
    by_id = {str(row.get("route_id")): row for row in rows if row.get("route_id")}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("route_family") or row.get("route_domain") or "unknown")].append(row)

    negative_rows: list[dict[str, Any]] = []
    if preference_mode == "observed_corruptions":
        negative_rows, preferences, listwise_groups = _build_corruption_preferences(
            rows,
            corruptions_per_route=corruptions_per_route,
            max_pairs=max_pairs,
            max_listwise_routes_per_group=max_listwise_routes_per_group,
        )
    elif preference_mode == "observed_quality_support":
        preferences = _build_quality_support_preferences(
            groups,
            max_pairs_per_route=max_pairs_per_route,
            max_pairs=max_pairs,
        )
        listwise_groups = _build_quality_support_listwise_groups(
            groups,
            max_routes_per_group=max_listwise_routes_per_group,
        )
    else:
        raise ValueError(f"unknown preference_mode: {preference_mode}")

    if negative_rows and negative_feature_output is None:
        negative_feature_output = output.with_name(f"{output.stem}_negative_features.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, preferences)
    if negative_feature_output is not None:
        negative_feature_output.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(negative_feature_output, negative_rows)
    if listwise_output is not None:
        listwise_output.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(listwise_output, listwise_groups)
    report = {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "feature_pack": str(feature_pack),
            "output": str(output),
            "listwise_output": str(listwise_output) if listwise_output else None,
            "negative_feature_output": str(negative_feature_output) if negative_feature_output else None,
            "preference_mode": preference_mode,
            "max_pairs_per_route": max_pairs_per_route,
            "max_pairs": max_pairs,
            "corruptions_per_route": corruptions_per_route,
            "max_listwise_routes_per_group": max_listwise_routes_per_group,
            "training_contract": (
                "observed_v4_route_over_structurally_corrupted_same_target_route.v1"
                if preference_mode == "observed_corruptions"
                else "legacy_pairwise_preferences_from_observed_v4_quality_and_support.v1"
            ),
            "important_note": (
                "gold/silver quality_tier is record completeness metadata and is not used as a route preference "
                "in observed_corruptions mode."
            ),
        },
        "counts": {
            "feature_rows": len(rows),
            "unique_routes": len(by_id),
            "negative_feature_rows": len(negative_rows),
            "preferences": len(preferences),
            "listwise_groups": len(listwise_groups),
        },
        "preference_source_counts": dict(Counter(row.get("preference_source") for row in preferences)),
        "confidence_tier_counts": dict(Counter(row.get("confidence_tier") for row in preferences)),
        "split_counts": dict(Counter(row.get("split") for row in preferences)),
        "output": str(output),
        "listwise_output": str(listwise_output) if listwise_output else None,
        "negative_feature_output": str(negative_feature_output) if negative_feature_output else None,
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _build_corruption_preferences(
    rows: list[dict[str, Any]],
    *,
    corruptions_per_route: int,
    max_pairs: int | None,
    max_listwise_routes_per_group: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    negative_rows: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    listwise_groups: list[dict[str, Any]] = []
    cap = max(1, int(corruptions_per_route or 1))
    listwise_cap = max(2, int(max_listwise_routes_per_group or 32))
    donor_rows = [row for row in rows if row.get("steps") and row.get("target_smiles")]
    for idx, row in enumerate(rows):
        if not row.get("route_id") or not row.get("steps") or not row.get("target_smiles"):
            continue
        corruptions = _corrupted_rows_for_route(row, donor_rows=donor_rows, row_index=idx, cap=cap)
        if not corruptions:
            continue
        negative_rows.extend(corruptions)
        for neg in corruptions:
            preferences.append(
                {
                    "schema_version": PREFERENCE_SCHEMA_VERSION,
                    "preference_id": _pref_id(row, neg, str((neg.get("metadata") or {}).get("corruption_kind") or "corruption")),
                    "route_family": row.get("route_family") or row.get("route_domain") or "unknown",
                    "better_route_id": row.get("route_id"),
                    "worse_route_id": neg.get("route_id"),
                    "split": row.get("split") or "",
                    "preference_source": f"observed_route_over_{(neg.get('metadata') or {}).get('corruption_kind')}",
                    "confidence_tier": "T1_structural_corruption_negative",
                    "evidence_summary": {
                        "observed_route_id": row.get("route_id"),
                        "corrupted_route_id": neg.get("route_id"),
                        "corruption_kind": (neg.get("metadata") or {}).get("corruption_kind"),
                        "donor_route_id": (neg.get("metadata") or {}).get("donor_route_id"),
                        "note": "Preference is observed experimental route over an intentionally disconnected or target-mismatched synthetic negative.",
                    },
                }
            )
            if max_pairs is not None and max_pairs > 0 and len(preferences) >= int(max_pairs):
                break
        group_ids = [row.get("route_id"), *[neg.get("route_id") for neg in corruptions]]
        group_ids = [route_id for route_id in group_ids[:listwise_cap] if route_id]
        if len(group_ids) >= 2:
            listwise_groups.append(
                {
                    "schema_version": "v4_cascade_product_listwise_pack.v1",
                    "listwise_id": _pref_id(row, {"route_id": group_ids[-1]}, "observed_route_over_corruptions_listwise"),
                    "route_family": row.get("route_family") or row.get("route_domain") or "unknown",
                    "split": row.get("split") or "",
                    "ordered_route_ids": group_ids,
                    "preference_source": "observed_route_over_corruptions_listwise",
                    "confidence_tier": "T1_structural_corruption_negative",
                    "evidence_summary": {
                        "observed_route_id": row.get("route_id"),
                        "negative_count": len(group_ids) - 1,
                    },
                }
            )
        if max_pairs is not None and max_pairs > 0 and len(preferences) >= int(max_pairs):
            break
    return negative_rows, preferences, listwise_groups


def _corrupted_rows_for_route(
    row: dict[str, Any],
    *,
    donor_rows: list[dict[str, Any]],
    row_index: int,
    cap: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    donor = _donor_for_row(row, donor_rows, row_index=row_index)
    if donor is not None:
        out.append(_foreign_route_body_same_target(row, donor))
    dropped = _drop_target_product_step(row)
    if dropped is not None:
        out.append(dropped)
    reversed_row = _reverse_target_product_step(row)
    if reversed_row is not None:
        out.append(reversed_row)
    unique = []
    seen = set()
    for item in out:
        key = item.get("route_id")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[: max(1, cap)]


def _foreign_route_body_same_target(row: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    neg = _base_corruption(row, kind="foreign_route_body_same_target", donor=donor)
    neg["steps"] = _deepcopy(donor.get("steps") or [])
    neg["starting_material_smiles"] = donor.get("starting_material_smiles") or ""
    neg["starting_material_name"] = donor.get("starting_material_name")
    neg["terminal_reactants"] = list(donor.get("terminal_reactants") or [])
    neg["route_domain"] = donor.get("route_domain") or neg.get("route_domain")
    neg["route_family"] = row.get("route_family") or row.get("route_domain") or "unknown"
    return neg


def _drop_target_product_step(row: dict[str, Any]) -> dict[str, Any] | None:
    steps = _deepcopy(row.get("steps") or [])
    if len(steps) < 2:
        return None
    target = str(row.get("target_smiles") or "")
    keep = [
        step for step in steps
        if target not in {str(product or "") for product in step.get("products") or []}
    ]
    if len(keep) == len(steps):
        keep = steps[:-1]
    if not keep or len(keep) == len(steps):
        return None
    neg = _base_corruption(row, kind="drop_target_product_step")
    neg["steps"] = keep
    terminal = _terminal_reactants_from_steps(keep)
    if terminal:
        neg["terminal_reactants"] = terminal
        neg["starting_material_smiles"] = "; ".join(terminal)
    return neg


def _reverse_target_product_step(row: dict[str, Any]) -> dict[str, Any] | None:
    steps = _deepcopy(row.get("steps") or [])
    if not steps:
        return None
    target = str(row.get("target_smiles") or "")
    target_idx = None
    for idx, step in enumerate(steps):
        products = {str(product or "") for product in step.get("products") or []}
        if target in products:
            target_idx = idx
    if target_idx is None:
        target_idx = len(steps) - 1
    step = dict(steps[target_idx])
    reactants = [str(value) for value in step.get("reactants") or [] if value]
    products = [str(value) for value in step.get("products") or [] if value]
    if not reactants or not products:
        return None
    step["reactants"] = products
    step["products"] = reactants
    step["rxn_smiles"] = f"{'.'.join(products)}>>{'.'.join(reactants)}"
    steps[target_idx] = step
    neg = _base_corruption(row, kind="reverse_target_product_step")
    neg["steps"] = steps
    terminal = _terminal_reactants_from_steps(steps)
    if terminal:
        neg["terminal_reactants"] = terminal
        neg["starting_material_smiles"] = "; ".join(terminal)
    return neg


def _base_corruption(row: dict[str, Any], *, kind: str, donor: dict[str, Any] | None = None) -> dict[str, Any]:
    neg = _deepcopy(row)
    neg["route_id"] = _stable_id(row.get("route_id"), kind, donor.get("route_id") if donor else None)
    neg["route_source"] = "dataset_v4_release_structural_corruption"
    neg["quality_tier"] = "corrupted_negative"
    neg["is_high_quality"] = False
    neg["trainable_recommended"] = False
    neg["is_demonstrated_success"] = False
    neg["labels"] = {str(key): 0.0 for key in (row.get("labels") or {})}
    neg["value_target"] = 0.0
    metadata = dict(neg.get("metadata") or {})
    metadata.update({
        "corruption_kind": kind,
        "source_observed_route_id": row.get("route_id"),
        "donor_route_id": donor.get("route_id") if donor else None,
        "negative_contract": "structurally corrupted route; not a gold/silver quality comparison",
    })
    neg["metadata"] = metadata
    return neg


def _donor_for_row(
    row: dict[str, Any],
    donor_rows: list[dict[str, Any]],
    *,
    row_index: int,
) -> dict[str, Any] | None:
    if len(donor_rows) < 2:
        return None
    target = str(row.get("target_smiles") or "")
    route_id = str(row.get("route_id") or "")
    n = len(donor_rows)
    for offset in range(1, n):
        donor = donor_rows[(row_index + offset) % n]
        if str(donor.get("route_id") or "") == route_id:
            continue
        if str(donor.get("target_smiles") or "") == target:
            continue
        if not donor.get("steps"):
            continue
        return donor
    return None


def _terminal_reactants_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    products = {
        str(product or "")
        for step in steps
        for product in step.get("products") or []
        if product
    }
    out: list[str] = []
    for step in steps:
        for reactant in step.get("reactants") or []:
            text = str(reactant or "")
            if text and text not in products and text not in out:
                out.append(text)
    return out


def _deepcopy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _build_quality_support_preferences(
    groups: dict[str, list[dict[str, Any]]],
    *,
    max_pairs_per_route: int,
    max_pairs: int | None,
) -> list[dict[str, Any]]:
    preferences: list[dict[str, Any]] = []
    per_route_counts: Counter[str] = Counter()
    for family, group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=lambda row: (str(row.get("doi") or ""), str(row.get("cascade_id") or "")))
        for better in ordered:
            for worse in ordered:
                if better.get("route_id") == worse.get("route_id"):
                    continue
                decision = _dominance_reason(better, worse)
                if decision is None:
                    continue
                if per_route_counts[str(better.get("route_id"))] >= max_pairs_per_route:
                    continue
                preferences.append(
                    {
                        "schema_version": PREFERENCE_SCHEMA_VERSION,
                        "preference_id": _pref_id(better, worse, decision["reason"]),
                        "route_family": family,
                        "better_route_id": better.get("route_id"),
                        "worse_route_id": worse.get("route_id"),
                        "split": better.get("split") if better.get("split") == worse.get("split") else "mixed",
                        "preference_source": decision["reason"],
                        "confidence_tier": decision["confidence_tier"],
                        "evidence_summary": decision["evidence_summary"],
                    }
                )
                per_route_counts[str(better.get("route_id"))] += 1
                if max_pairs is not None and max_pairs > 0 and len(preferences) >= int(max_pairs):
                    break
            if max_pairs is not None and max_pairs > 0 and len(preferences) >= int(max_pairs):
                break
        if max_pairs is not None and max_pairs > 0 and len(preferences) >= int(max_pairs):
            break
    return preferences


def _build_quality_support_listwise_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    max_routes_per_group: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cap = max(2, int(max_routes_per_group or 32))
    for family, group_rows in sorted(groups.items()):
        ordered = sorted(
            group_rows,
            key=lambda row: (
                *_negative_quality_rank(row),
                str(row.get("doi") or ""),
                str(row.get("cascade_id") or ""),
                str(row.get("route_id") or ""),
            ),
        )
        if len(ordered) < 2:
            continue
        distinct_keys = {_listwise_quality_rank(row) for row in ordered}
        if len(distinct_keys) < 2:
            continue
        for chunk_index, start in enumerate(range(0, len(ordered), cap)):
            chunk = ordered[start:start + cap]
            if len(chunk) < 2:
                continue
            if len({_listwise_quality_rank(row) for row in chunk}) < 2:
                continue
            split_values = {str(row.get("split") or "") for row in chunk}
            out.append(
                {
                    "schema_version": "v4_cascade_product_listwise_pack.v1",
                    "listwise_id": _pref_id({"route_id": family}, {"route_id": chunk_index}, "observed_quality_support_listwise"),
                    "route_family": family,
                    "split": split_values.pop() if len(split_values) == 1 else "mixed",
                    "ordered_route_ids": [row.get("route_id") for row in chunk],
                    "preference_source": "observed_quality_support_listwise",
                    "confidence_tier": "T2_observed_quality_and_support",
                    "evidence_summary": [
                        {
                            "route_id": row.get("route_id"),
                            "quality_tier": row.get("quality_tier"),
                            "demonstrated_success": _label(row.get("labels") or {}, "demonstrated_success"),
                            "support_count": _support_count(row.get("labels") or {}),
                        }
                        for row in chunk
                    ],
                }
            )
    return out


def _listwise_quality_rank(row: dict[str, Any]) -> tuple[int, float, float]:
    labels = row.get("labels") or {}
    return (
        0,
        _label(labels, "demonstrated_success"),
        _support_count(labels),
    )


def _negative_quality_rank(row: dict[str, Any]) -> tuple[float, float, float]:
    tier, success, support = _listwise_quality_rank(row)
    return (-float(tier), -float(success), -float(support))


def _support_count(labels: dict[str, Any]) -> float:
    support_labels = [
        "outcome_supported",
        "condition_supported",
        "substrate_scope_supported",
        "rxn_step_supported",
        "catalyst_supported",
        "species_supported",
    ]
    return float(sum(_label(labels, name) for name in support_labels))


def _dominance_reason(better: dict[str, Any], worse: dict[str, Any]) -> dict[str, Any] | None:
    better_labels = better.get("labels") or {}
    worse_labels = worse.get("labels") or {}
    if _label(better_labels, "demonstrated_success") > _label(worse_labels, "demonstrated_success"):
        return {
            "reason": "demonstrated_success_over_unmarked_same_family",
            "confidence_tier": "T2_observed_success",
            "evidence_summary": {
                "better_demonstrated_success": _label(better_labels, "demonstrated_success"),
                "worse_demonstrated_success": _label(worse_labels, "demonstrated_success"),
            },
        }
    better_support = _support_count(better_labels)
    worse_support = _support_count(worse_labels)
    if better_support >= worse_support + 3.0:
        return {
            "reason": "evidence_complete_over_sparse_same_family",
            "confidence_tier": "T2_evidence_completeness",
            "evidence_summary": {
                "better_support_count": better_support,
                "worse_support_count": worse_support,
            },
        }
    return None


def _label(labels: dict[str, Any], name: str) -> float:
    try:
        return float(labels.get(name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pref_id(better: dict[str, Any], worse: dict[str, Any], reason: str) -> str:
    import hashlib

    text = json.dumps([better.get("route_id"), worse.get("route_id"), reason], sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _stable_id(*parts: Any) -> str:
    import hashlib

    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v4 cascade product-value pairwise preferences")
    ap.add_argument("--feature-pack", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--listwise-output")
    ap.add_argument("--negative-feature-output")
    ap.add_argument(
        "--preference-mode",
        default="observed_corruptions",
        choices=["observed_corruptions", "observed_quality_support"],
    )
    ap.add_argument("--max-pairs-per-route", type=int, default=4)
    ap.add_argument("--max-pairs", type=int)
    ap.add_argument("--corruptions-per-route", type=int, default=3)
    ap.add_argument("--max-listwise-routes-per-group", type=int, default=32)
    args = ap.parse_args()
    report = build_v4_cascade_preference_pack(
        feature_pack=Path(args.feature_pack),
        output=Path(args.output),
        listwise_output=Path(args.listwise_output) if args.listwise_output else None,
        negative_feature_output=Path(args.negative_feature_output) if args.negative_feature_output else None,
        preference_mode=args.preference_mode,
        max_pairs_per_route=args.max_pairs_per_route,
        max_pairs=args.max_pairs,
        corruptions_per_route=args.corruptions_per_route,
        max_listwise_routes_per_group=args.max_listwise_routes_per_group,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
