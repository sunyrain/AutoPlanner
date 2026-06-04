"""Build enzyme substrate-product-EC verifier v1 data.

This pack is intentionally different from bridge_verifier_v0.  v0 asks whether a
chemical product is close to the enzyme substrate/product space.  This v1 pack
asks whether a concrete enzymatic transformation tuple is plausible:

    substrate side + product side + EC evidence -> label

Positive rows come from curated enzyme reaction pools.  Negative rows are
constructed as hard counterexamples around the same EC bucket, random mismatches,
and common/cofactor artifact mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def split_name(key: str) -> str:
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "valid"
    return "test"


def parse_ecs(value: Any) -> list[str]:
    ecs = json_loads(value, [])
    out = []
    for item in ecs:
        ec = str(item or "").strip()
        if ec:
            out.append(ec)
    return out


def ec1_from_ecs(ecs: list[str]) -> str:
    for ec in ecs:
        head = ec.split(".", 1)[0]
        if head in {"1", "2", "3", "4", "5", "6", "7"}:
            return head
    return "unknown"


def ec2_from_ecs(ecs: list[str]) -> str:
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) >= 2 and parts[0] in {"1", "2", "3", "4", "5", "6", "7"} and parts[1] != "-":
            return ".".join(parts[:2])
    ec1 = ec1_from_ecs(ecs)
    return ec1 if ec1 != "unknown" else "unknown"


class SideCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}
        self.invalid = Counter()

    def get(self, smiles: str) -> dict[str, Any]:
        smiles = str(smiles or "")
        if smiles in self.cache:
            return self.cache[smiles]

        components: list[dict[str, Any]] = []
        for part in [item.strip() for item in smiles.split(".") if item.strip()]:
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                self.invalid["mol_from_smiles_failed"] += 1
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            components.append(
                {
                    "smiles": canonical,
                    "heavy": int(mol.GetNumHeavyAtoms()),
                    "rings": int(mol.GetRingInfo().NumRings()),
                    "hetero": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
                }
            )

        if not components:
            info = {
                "valid": False,
                "canonical": "",
                "largest_smiles": "",
                "total_heavy": 0,
                "largest_heavy": 0,
                "component_count": 0,
                "ring_count": 0,
                "hetero_count": 0,
            }
            self.cache[smiles] = info
            return info

        components.sort(key=lambda item: (item["heavy"], item["smiles"]), reverse=True)
        canonical = ".".join(sorted(item["smiles"] for item in components))
        info = {
            "valid": True,
            "canonical": canonical,
            "largest_smiles": components[0]["smiles"],
            "total_heavy": int(sum(item["heavy"] for item in components)),
            "largest_heavy": int(components[0]["heavy"]),
            "component_count": int(len(components)),
            "ring_count": int(sum(item["rings"] for item in components)),
            "hetero_count": int(sum(item["hetero"] for item in components)),
        }
        self.cache[smiles] = info
        return info


def side_is_usable(info: dict[str, Any], *, min_largest_heavy: int, max_total_heavy: int, max_components: int) -> bool:
    return bool(
        info.get("valid")
        and int(info.get("largest_heavy") or 0) >= min_largest_heavy
        and int(info.get("total_heavy") or 0) <= max_total_heavy
        and int(info.get("component_count") or 0) <= max_components
    )


def weight_for_positive(row: dict[str, Any], ecs: list[str]) -> float:
    ec_unique = int(row.get("ec_unique") or len(set(ecs)))
    occurrences = int(row.get("occurrences") or 1)
    if ec_unique >= 30:
        return 0.35
    if ec_unique >= 15:
        return 0.55
    if occurrences >= 1000 and ec_unique >= 8:
        return 0.65
    return 1.0


def make_row(
    *,
    row_id: str,
    reaction_id: str,
    substrate_smiles: str,
    product_smiles: str,
    substrate_key: str,
    product_key: str,
    reaction_smiles: str,
    ecs: list[str],
    substrate_info: dict[str, Any],
    product_info: dict[str, Any],
    label: int,
    label_type: str,
    label_weight: float,
    source: str,
    source_reaction_id: str,
    source_counts_json: str,
    occurrences: int,
    rhea_unique: int,
) -> dict[str, Any]:
    ec1 = ec1_from_ecs(ecs)
    ec2 = ec2_from_ecs(ecs)
    heavy_delta = int(product_info["total_heavy"]) - int(substrate_info["total_heavy"])
    max_heavy = max(int(product_info["total_heavy"]), int(substrate_info["total_heavy"]), 1)
    min_heavy = min(int(product_info["total_heavy"]), int(substrate_info["total_heavy"]))
    split_key = f"{substrate_info['canonical']}>>{product_info['canonical']}|{ec1}"
    row = {
        "row_id": row_id,
        "reaction_id": reaction_id,
        "source_reaction_id": source_reaction_id,
        "substrate_key": substrate_key,
        "product_key": product_key,
        "substrate_smiles": substrate_smiles,
        "product_smiles": product_smiles,
        "substrate_canonical": substrate_info["canonical"],
        "product_canonical": product_info["canonical"],
        "substrate_largest_smiles": substrate_info["largest_smiles"],
        "product_largest_smiles": product_info["largest_smiles"],
        "reaction_smiles": reaction_smiles,
        "ec_numbers_json": json.dumps(sorted(set(ecs)), ensure_ascii=False),
        "ec1": ec1,
        "ec2": ec2,
        "ec_count": len(set(ecs)),
        "ec_known": ec1 != "unknown",
        "label": int(label),
        "label_type": label_type,
        "label_weight": float(label_weight),
        "source": source,
        "source_counts_json": source_counts_json or "{}",
        "occurrences": int(occurrences or 0),
        "rhea_unique": int(rhea_unique or 0),
        "substrate_total_heavy": int(substrate_info["total_heavy"]),
        "product_total_heavy": int(product_info["total_heavy"]),
        "substrate_largest_heavy": int(substrate_info["largest_heavy"]),
        "product_largest_heavy": int(product_info["largest_heavy"]),
        "heavy_signed_delta": heavy_delta,
        "heavy_abs_delta": abs(heavy_delta),
        "heavy_ratio_min_over_max": round(float(min_heavy / max_heavy), 6),
        "substrate_component_count": int(substrate_info["component_count"]),
        "product_component_count": int(product_info["component_count"]),
        "component_count_delta": int(product_info["component_count"]) - int(substrate_info["component_count"]),
        "substrate_ring_count": int(substrate_info["ring_count"]),
        "product_ring_count": int(product_info["ring_count"]),
        "substrate_hetero_count": int(substrate_info["hetero_count"]),
        "product_hetero_count": int(product_info["hetero_count"]),
        "split_key": split_key,
        "split": split_name(split_key),
    }
    return row


def build_positive_rows(
    reaction_rows: list[dict[str, Any]],
    *,
    cache: SideCache,
    max_positives: int | None,
    min_largest_heavy: int,
    max_total_heavy: int,
    max_components: int,
) -> tuple[list[dict[str, Any]], Counter]:
    positives: list[dict[str, Any]] = []
    counts = Counter()
    seen = set()
    for row in reaction_rows:
        ecs = parse_ecs(row.get("ec_numbers_json"))
        if not ecs:
            counts["skip_no_ec"] += 1
            continue
        substrate_info = cache.get(row.get("substrate_smiles") or "")
        product_info = cache.get(row.get("product_smiles") or "")
        if not side_is_usable(
            substrate_info,
            min_largest_heavy=min_largest_heavy,
            max_total_heavy=max_total_heavy,
            max_components=max_components,
        ):
            counts["skip_unusable_substrate_side"] += 1
            continue
        if not side_is_usable(
            product_info,
            min_largest_heavy=min_largest_heavy,
            max_total_heavy=max_total_heavy,
            max_components=max_components,
        ):
            counts["skip_unusable_product_side"] += 1
            continue
        ec1 = ec1_from_ecs(ecs)
        pair_key = (substrate_info["canonical"], product_info["canonical"], ec1)
        if pair_key in seen:
            counts["skip_duplicate_positive_pair_ec1"] += 1
            continue
        seen.add(pair_key)
        reaction_id = str(row.get("reaction_id") or f"enzyme_positive_{len(positives):08d}")
        positives.append(
            make_row(
                row_id=f"pos:{reaction_id}",
                reaction_id=reaction_id,
                source_reaction_id=reaction_id,
                substrate_smiles=row.get("substrate_smiles") or "",
                product_smiles=row.get("product_smiles") or "",
                substrate_key=row.get("substrate_key") or "",
                product_key=row.get("product_key") or "",
                reaction_smiles=row.get("reaction_smiles") or "",
                ecs=ecs,
                substrate_info=substrate_info,
                product_info=product_info,
                label=1,
                label_type="enzyme_reaction_positive",
                label_weight=weight_for_positive(row, ecs),
                source="enzyme_reaction_pool",
                source_counts_json=row.get("source_counts_json") or "{}",
                occurrences=int(row.get("occurrences") or 0),
                rhea_unique=int(row.get("rhea_unique") or 0),
            )
        )
        counts["positive_rows"] += 1
        counts[f"positive_ec1_{ec1}"] += 1
        if max_positives is not None and len(positives) >= max_positives:
            counts["stopped_at_max_positives"] = max_positives
            break
    return positives, counts


def build_common_artifact_sides(root: Path, cache: SideCache, *, max_rows: int = 500) -> list[dict[str, Any]]:
    path = root / "cofactor_common_metabolite_blacklist.parquet"
    if not path.exists():
        return []
    sides = []
    for row in read_rows(path):
        info = cache.get(row.get("canonical_smiles") or "")
        if not info.get("valid"):
            continue
        sides.append(
            {
                "smiles": row.get("canonical_smiles") or "",
                "key": row.get("inchikey") or "",
                "info": info,
                "source_counts_json": json.dumps({"cofactor_common_metabolite_blacklist": 1}),
            }
        )
        if len(sides) >= max_rows:
            break
    return sides


def positive_key_set(rows: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    pair_any = set()
    pair_ec1 = set()
    for row in rows:
        pair = (row["substrate_canonical"], row["product_canonical"])
        pair_any.add(pair)
        pair_ec1.add((pair[0], pair[1], row["ec1"]))
    return pair_any, pair_ec1


def build_negative_rows(
    positives: list[dict[str, Any]],
    *,
    root: Path,
    cache: SideCache,
    negatives_per_positive: int,
    max_negatives: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(seed)
    counts = Counter()
    pair_any, pair_ec1 = positive_key_set(positives)
    by_ec2: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ec1: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        by_ec2[row["ec2"]].append(row)
        by_ec1[row["ec1"]].append(row)
    all_rows = list(positives)
    common_sides = build_common_artifact_sides(root, cache)

    negatives: list[dict[str, Any]] = []
    seen = set()

    def choose_bucket(pos: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = by_ec2.get(pos["ec2"]) or []
        if len(candidates) >= 2:
            return candidates
        return by_ec1.get(pos["ec1"]) or []

    def add_neg(
        pos: dict[str, Any],
        *,
        substrate_smiles: str,
        product_smiles: str,
        substrate_key: str,
        product_key: str,
        substrate_info: dict[str, Any],
        product_info: dict[str, Any],
        neg_type: str,
        source_reaction_id: str,
        label_weight: float = 1.0,
        source: str = "enzyme_sp_hard_negative",
        source_counts_json: str = "{}",
    ) -> bool:
        if (substrate_info["canonical"], product_info["canonical"]) in pair_any:
            counts[f"skip_{neg_type}_known_positive_pair"] += 1
            return False
        if (substrate_info["canonical"], product_info["canonical"], pos["ec1"]) in pair_ec1:
            counts[f"skip_{neg_type}_known_positive_pair_ec1"] += 1
            return False
        key = (substrate_info["canonical"], product_info["canonical"], pos["ec1"], neg_type)
        if key in seen:
            counts[f"skip_{neg_type}_duplicate_negative"] += 1
            return False
        seen.add(key)
        reaction_id = f"neg_{len(negatives):08d}"
        negatives.append(
            make_row(
                row_id=f"neg:{reaction_id}",
                reaction_id=reaction_id,
                source_reaction_id=source_reaction_id,
                substrate_smiles=substrate_smiles,
                product_smiles=product_smiles,
                substrate_key=substrate_key,
                product_key=product_key,
                reaction_smiles=f"{substrate_smiles}>>{product_smiles}",
                ecs=parse_ecs(pos["ec_numbers_json"]),
                substrate_info=substrate_info,
                product_info=product_info,
                label=0,
                label_type=neg_type,
                label_weight=label_weight,
                source=source,
                source_counts_json=source_counts_json,
                occurrences=0,
                rhea_unique=0,
            )
        )
        counts[f"negative_{neg_type}"] += 1
        return True

    shuffled = list(positives)
    rng.shuffle(shuffled)
    for index, pos in enumerate(shuffled):
        if len(negatives) >= max_negatives:
            counts["stopped_at_max_negatives"] = max_negatives
            break
        target_added = 0
        attempts = 0
        while target_added < negatives_per_positive and attempts < 20 and len(negatives) < max_negatives:
            attempts += 1
            neg_kind = (index + attempts) % 4
            if neg_kind == 0:
                bucket = choose_bucket(pos)
                candidates = [row for row in bucket if row["product_canonical"] != pos["product_canonical"]]
                if not candidates:
                    continue
                other = rng.choice(candidates)
                ok = add_neg(
                    pos,
                    substrate_smiles=pos["substrate_smiles"],
                    product_smiles=other["product_smiles"],
                    substrate_key=pos["substrate_key"],
                    product_key=other["product_key"],
                    substrate_info={
                        "canonical": pos["substrate_canonical"],
                        "largest_smiles": pos["substrate_largest_smiles"],
                        "total_heavy": pos["substrate_total_heavy"],
                        "largest_heavy": pos["substrate_largest_heavy"],
                        "component_count": pos["substrate_component_count"],
                        "ring_count": pos["substrate_ring_count"],
                        "hetero_count": pos["substrate_hetero_count"],
                        "valid": True,
                    },
                    product_info={
                        "canonical": other["product_canonical"],
                        "largest_smiles": other["product_largest_smiles"],
                        "total_heavy": other["product_total_heavy"],
                        "largest_heavy": other["product_largest_heavy"],
                        "component_count": other["product_component_count"],
                        "ring_count": other["product_ring_count"],
                        "hetero_count": other["product_hetero_count"],
                        "valid": True,
                    },
                    neg_type="same_ec_wrong_product",
                    source_reaction_id=other["reaction_id"],
                    label_weight=1.0,
                )
                target_added += int(ok)
            elif neg_kind == 1:
                bucket = choose_bucket(pos)
                candidates = [row for row in bucket if row["substrate_canonical"] != pos["substrate_canonical"]]
                if not candidates:
                    continue
                other = rng.choice(candidates)
                ok = add_neg(
                    pos,
                    substrate_smiles=other["substrate_smiles"],
                    product_smiles=pos["product_smiles"],
                    substrate_key=other["substrate_key"],
                    product_key=pos["product_key"],
                    substrate_info={
                        "canonical": other["substrate_canonical"],
                        "largest_smiles": other["substrate_largest_smiles"],
                        "total_heavy": other["substrate_total_heavy"],
                        "largest_heavy": other["substrate_largest_heavy"],
                        "component_count": other["substrate_component_count"],
                        "ring_count": other["substrate_ring_count"],
                        "hetero_count": other["substrate_hetero_count"],
                        "valid": True,
                    },
                    product_info={
                        "canonical": pos["product_canonical"],
                        "largest_smiles": pos["product_largest_smiles"],
                        "total_heavy": pos["product_total_heavy"],
                        "largest_heavy": pos["product_largest_heavy"],
                        "component_count": pos["product_component_count"],
                        "ring_count": pos["product_ring_count"],
                        "hetero_count": pos["product_hetero_count"],
                        "valid": True,
                    },
                    neg_type="same_ec_wrong_substrate",
                    source_reaction_id=other["reaction_id"],
                    label_weight=1.0,
                )
                target_added += int(ok)
            elif neg_kind == 2:
                other = rng.choice(all_rows)
                if other["ec1"] == pos["ec1"] and len(set(parse_ecs(other["ec_numbers_json"])) & set(parse_ecs(pos["ec_numbers_json"]))) > 0:
                    continue
                ok = add_neg(
                    pos,
                    substrate_smiles=pos["substrate_smiles"],
                    product_smiles=other["product_smiles"],
                    substrate_key=pos["substrate_key"],
                    product_key=other["product_key"],
                    substrate_info={
                        "canonical": pos["substrate_canonical"],
                        "largest_smiles": pos["substrate_largest_smiles"],
                        "total_heavy": pos["substrate_total_heavy"],
                        "largest_heavy": pos["substrate_largest_heavy"],
                        "component_count": pos["substrate_component_count"],
                        "ring_count": pos["substrate_ring_count"],
                        "hetero_count": pos["substrate_hetero_count"],
                        "valid": True,
                    },
                    product_info={
                        "canonical": other["product_canonical"],
                        "largest_smiles": other["product_largest_smiles"],
                        "total_heavy": other["product_total_heavy"],
                        "largest_heavy": other["product_largest_heavy"],
                        "component_count": other["product_component_count"],
                        "ring_count": other["product_ring_count"],
                        "hetero_count": other["product_hetero_count"],
                        "valid": True,
                    },
                    neg_type="random_wrong_product",
                    source_reaction_id=other["reaction_id"],
                    label_weight=1.0,
                )
                target_added += int(ok)
            else:
                if not common_sides:
                    continue
                common = rng.choice(common_sides)
                ok = add_neg(
                    pos,
                    substrate_smiles=pos["substrate_smiles"],
                    product_smiles=common["smiles"],
                    substrate_key=pos["substrate_key"],
                    product_key=common["key"],
                    substrate_info={
                        "canonical": pos["substrate_canonical"],
                        "largest_smiles": pos["substrate_largest_smiles"],
                        "total_heavy": pos["substrate_total_heavy"],
                        "largest_heavy": pos["substrate_largest_heavy"],
                        "component_count": pos["substrate_component_count"],
                        "ring_count": pos["substrate_ring_count"],
                        "hetero_count": pos["substrate_hetero_count"],
                        "valid": True,
                    },
                    product_info=common["info"],
                    neg_type="common_or_cofactor_wrong_product",
                    source_reaction_id="cofactor_common_metabolite_blacklist",
                    label_weight=1.0,
                    source="cofactor_common_metabolite_artifact_negative",
                    source_counts_json=common["source_counts_json"],
                )
                target_added += int(ok)
        counts["negative_attempted_positive_rows"] += 1
    counts["negative_rows"] = len(negatives)
    counts["negative_unique_keys"] = len(seen)
    return negatives, counts


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_label_type = Counter()
    by_ec1 = Counter()
    for row in rows:
        split = row.get("split") or "unknown"
        label = int(row.get("label") or 0)
        by_split[split]["rows"] += 1
        by_split[split]["positives"] += int(label == 1)
        by_split[split]["negatives"] += int(label == 0)
        by_label_type[str(row.get("label_type") or "unknown")] += 1
        by_ec1[str(row.get("ec1") or "unknown")] += 1
    return {
        "total_rows": len(rows),
        "by_split": {split: dict(counter) for split, counter in sorted(by_split.items())},
        "by_label_type": dict(by_label_type),
        "by_ec1": dict(by_ec1),
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Enzyme Substrate-Product Verifier v1 Data Report",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- elapsed_seconds: `{manifest['elapsed_seconds']}`",
        f"- input_pack_dir: `{manifest['input_pack_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## Summary",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| total rows | {manifest['summary']['total_rows']} |",
        f"| positives | {manifest['counts'].get('positive_rows', 0)} |",
        f"| negatives | {manifest['counts'].get('negative_rows', 0)} |",
        f"| molecule parse cache | {manifest['counts'].get('side_cache_size', 0)} |",
    ]
    lines.extend(["", "## Splits", "", "| Split | Rows | Positives | Negatives |", "|---|---:|---:|---:|"])
    for split, row in manifest["summary"]["by_split"].items():
        lines.append(f"| `{split}` | {row.get('rows', 0)} | {row.get('positives', 0)} | {row.get('negatives', 0)} |")
    lines.extend(["", "## Label Types", "", "| Label Type | Rows |", "|---|---:|"])
    for key, value in sorted(manifest["summary"]["by_label_type"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## EC1 Distribution", "", "| EC1 | Rows |", "|---|---:|"])
    for key, value in sorted(manifest["summary"]["by_ec1"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Builder Counters", "", "| Counter | Value |", "|---|---:|"])
    for key, value in sorted(manifest["counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Files", "", "| Artifact | Path |", "|---|---|"])
    for key, value in manifest["files"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enzyme substrate-product verifier v1 pack")
    parser.add_argument("--input-pack-dir", default="data/bridge_pack_v0")
    parser.add_argument("--output-dir", default="data/enzyme_sp_verifier_v1")
    parser.add_argument("--max-positives", type=int, default=None)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=330000)
    parser.add_argument("--min-largest-heavy", type=int, default=5)
    parser.add_argument("--max-total-heavy", type=int, default=240)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    started = time.time()
    input_dir = Path(args.input_pack_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reaction_rows = read_rows(input_dir / "enzyme_reaction_pool.parquet")
    cache = SideCache()
    positives, pos_counts = build_positive_rows(
        reaction_rows,
        cache=cache,
        max_positives=args.max_positives,
        min_largest_heavy=args.min_largest_heavy,
        max_total_heavy=args.max_total_heavy,
        max_components=args.max_components,
    )
    negatives, neg_counts = build_negative_rows(
        positives,
        root=input_dir,
        cache=cache,
        negatives_per_positive=args.negatives_per_positive,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )

    rows = positives + negatives
    rows.sort(key=lambda row: (row["split"], row["label"], row["row_id"]))
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        split_rows[row["split"]].append(row)

    files = {
        "train_parquet": str(out_dir / "train.parquet"),
        "valid_parquet": str(out_dir / "valid.parquet"),
        "test_parquet": str(out_dir / "test.parquet"),
        "train_jsonl": str(out_dir / "train.jsonl"),
        "valid_jsonl": str(out_dir / "valid.jsonl"),
        "test_jsonl": str(out_dir / "test.jsonl"),
        "manifest": str(out_dir / "manifest.json"),
        "report": str(out_dir / "dataset_report.md"),
    }
    for split, split_data in split_rows.items():
        write_parquet(out_dir / f"{split}.parquet", split_data)
        write_jsonl(out_dir / f"{split}.jsonl", split_data)

    counts = Counter()
    counts.update(pos_counts)
    counts.update(neg_counts)
    counts["input_reaction_rows"] = len(reaction_rows)
    counts["output_rows"] = len(rows)
    counts["side_cache_size"] = len(cache.cache)
    for key, value in cache.invalid.items():
        counts[f"side_cache_invalid_{key}"] = value

    manifest = {
        "schema_version": "enzyme_sp_verifier_v1.data.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "input_pack_dir": str(input_dir),
        "output_dir": str(out_dir),
        "parameters": vars(args),
        "counts": dict(counts),
        "summary": summarize(rows),
        "files": files,
    }
    Path(files["manifest"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(files["report"]).write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"summary": manifest["summary"], "counts": manifest["counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
