"""Build similarity bridges, hard negatives, and verifier splits for bridge_pack_v0."""
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchiKey


RDLogger.DisableLog("rdApp.*")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def mol_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def row_ecs(row: dict[str, Any]) -> list[str]:
    return [ec for ec in json_loads(row.get("ec_sample_json"), []) if ec]


def is_audit_like(row: dict[str, Any], blacklist: set[str]) -> bool:
    if row.get("inchikey") in blacklist:
        return True
    try:
        if int(row.get("ec_unique") or 0) >= 30:
            return True
        if int(row.get("occurrences") or 0) >= 1000:
            return True
    except Exception:
        pass
    return False


def load_augmented_enzyme_pool(root: Path, *, augment_from_reaction_pool: bool = False) -> list[dict[str, Any]]:
    base = read_rows(root / "enzyme_substrate_product_pool.parquet")
    if not augment_from_reaction_pool:
        return base
    seen = {row["inchikey"]: row for row in base}
    reaction_path = root / "enzyme_reaction_pool.parquet"
    if not reaction_path.exists():
        return base
    for rxn in read_rows(reaction_path):
        ecs = json_loads(rxn.get("ec_numbers_json"), [])
        for role, side in [("substrate", rxn.get("substrate_smiles")), ("product", rxn.get("product_smiles"))]:
            for smiles in str(side or "").split("."):
                smiles = smiles.strip()
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                inchikey = MolToInchiKey(mol)
                if inchikey in seen:
                    continue
                seen[inchikey] = {
                    "pool": "enzyme_substrate_product_augmented",
                    "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True),
                    "inchikey": inchikey,
                    "formula": "",
                    "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                    "occurrences": 1,
                    "source_counts_json": json.dumps({"enzyme:reaction_pool_augmented": 1}),
                    "role_counts_json": json.dumps({role: 1}),
                    "ec_sample_json": json.dumps(ecs[:20]),
                    "ec_unique": len(set(ecs)),
                    "example_ids_json": json.dumps([rxn.get("reaction_id") or ""]),
                }
    return list(seen.values())


def build_similarity_bridges(
    root: Path,
    *,
    threshold: float,
    low_threshold: float,
    max_bucket: int,
    max_pairs: int,
    rare_bits: int,
    min_shared_rare_bits: int,
    augment_enzyme_pool: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    counts = Counter()
    chemicals = read_rows(root / "chemical_product_pool.parquet")
    enzymes = load_augmented_enzyme_pool(root, augment_from_reaction_pool=augment_enzyme_pool)
    blacklist = {row["inchikey"] for row in read_rows(root / "cofactor_common_metabolite_blacklist.parquet")}
    exact_keys = {row["inchikey"] for row in read_rows(root / "exact_bridge_all.parquet")}

    enzyme_items = []
    bit_index: dict[int, list[int]] = defaultdict(list)
    for row in enzymes:
        heavy = int(row.get("heavy_atoms") or 0)
        if heavy < 5 or heavy > 180:
            counts["enzyme_similarity_excluded_heavy"] += 1
            continue
        if is_audit_like(row, blacklist):
            counts["enzyme_similarity_excluded_audit_like"] += 1
            continue
        fp = mol_fp(row.get("canonical_smiles") or "")
        if fp is None:
            counts["enzyme_similarity_invalid_fp"] += 1
            continue
        onbits = list(fp.GetOnBits())
        idx = len(enzyme_items)
        enzyme_items.append({"row": row, "fp": fp, "onbits": onbits})
        for bit in onbits:
            bit_index[bit].append(idx)

    counts["enzyme_similarity_indexed"] = len(enzyme_items)
    counts["bit_index_bits"] = len(bit_index)
    bit_freq = {bit: len(values) for bit, values in bit_index.items()}

    strict_rows: list[dict[str, Any]] = []
    near_negative_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for chem_i, chem in enumerate(chemicals):
        heavy = int(chem.get("heavy_atoms") or 0)
        if heavy < 5 or heavy > 180:
            counts["chemical_similarity_excluded_heavy"] += 1
            continue
        if chem.get("inchikey") in blacklist:
            counts["chemical_similarity_excluded_blacklist"] += 1
            continue
        cfp = mol_fp(chem.get("canonical_smiles") or "")
        if cfp is None:
            counts["chemical_similarity_invalid_fp"] += 1
            continue
        c_onbits = list(cfp.GetOnBits())
        selected_bits = sorted(c_onbits, key=lambda bit: bit_freq.get(bit, 10**9))[:rare_bits]
        candidate_hits: Counter[int] = Counter()
        for bit in selected_bits:
            bucket = bit_index.get(bit, [])
            if len(bucket) > max_bucket:
                continue
            candidate_hits.update(bucket)
        candidate_ids = [
            idx
            for idx, hit_count in candidate_hits.most_common(8000)
            if hit_count >= min_shared_rare_bits
        ]
        if not candidate_ids:
            continue
        scored = []
        for enzyme_idx in candidate_ids:
            enz = enzyme_items[enzyme_idx]["row"]
            if chem["inchikey"] == enz["inchikey"]:
                continue
            pair_key = (chem["inchikey"], enz["inchikey"])
            if pair_key in seen_pairs:
                continue
            h2 = int(enz.get("heavy_atoms") or 0)
            if abs(heavy - h2) > max(8, int(0.30 * max(heavy, h2))):
                continue
            sim = float(DataStructs.TanimotoSimilarity(cfp, enzyme_items[enzyme_idx]["fp"]))
            if sim >= low_threshold:
                scored.append((sim, enz))
        scored.sort(key=lambda item: item[0], reverse=True)
        for sim, enz in scored[:8]:
            seen_pairs.add((chem["inchikey"], enz["inchikey"]))
            role_counts = json_loads(enz.get("role_counts_json"), {})
            directions = []
            if int(role_counts.get("substrate", 0) or 0) > 0:
                directions.append("chemical_product_to_similar_enzyme_substrate")
            if int(role_counts.get("product", 0) or 0) > 0:
                directions.append("chemical_product_to_similar_enzyme_product")
            if not directions:
                directions.append("chemical_product_to_similar_enzyme_molecule")
            row_base = {
                "chemical_inchikey": chem["inchikey"],
                "enzyme_inchikey": enz["inchikey"],
                "chemical_smiles": chem["canonical_smiles"],
                "enzyme_smiles": enz["canonical_smiles"],
                "chemical_heavy_atoms": int(chem.get("heavy_atoms") or 0),
                "enzyme_heavy_atoms": int(enz.get("heavy_atoms") or 0),
                "tanimoto": round(sim, 6),
                "chemical_occurrences": int(chem.get("occurrences") or 0),
                "enzyme_occurrences": int(enz.get("occurrences") or 0),
                "enzyme_ec_sample_json": enz.get("ec_sample_json") or "[]",
                "enzyme_ec_unique": int(enz.get("ec_unique") or 0),
                "chemical_source_counts_json": chem.get("source_counts_json") or "{}",
                "enzyme_source_counts_json": enz.get("source_counts_json") or "{}",
            }
            for direction in directions:
                row = dict(row_base)
                row["bridge_direction"] = direction
                if sim >= threshold:
                    row["confidence_tier"] = "tier3_high_similarity_nonexact_bridge"
                    row["is_similarity_training_positive"] = True
                    strict_rows.append(row)
                else:
                    row["confidence_tier"] = "near_similarity_nonbridge_candidate"
                    row["is_similarity_training_positive"] = False
                    near_negative_rows.append(row)
        if max_pairs and len(strict_rows) >= max_pairs:
            break
        if chem_i and chem_i % 10000 == 0:
            print(f"[similarity] chemicals={chem_i} positives={len(strict_rows)} near={len(near_negative_rows)}")
    counts["similarity_bridge_filtered_rows"] = len(strict_rows)
    counts["similarity_near_negative_candidate_rows"] = len(near_negative_rows)
    counts["similarity_unique_pairs_seen"] = len(seen_pairs)
    return strict_rows[:max_pairs or None], near_negative_rows, counts


def split_name(key: str) -> str:
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "valid"
    return "test"


def make_positive_rows(root: Path, similarity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in read_rows(root / "exact_bridge_strict.parquet"):
        rows.append(
            {
                "chemical_inchikey": row["inchikey"],
                "enzyme_inchikey": row["inchikey"],
                "chemical_smiles": row["canonical_smiles"],
                "enzyme_smiles": row["canonical_smiles"],
                "bridge_direction": row["bridge_direction"],
                "label": 1,
                "label_type": row["confidence_tier"],
                "label_weight": 1.0,
                "tanimoto": 1.0,
                "enzyme_ec_sample_json": row.get("enzyme_ec_sample_json") or "[]",
                "source": "exact_bridge_strict",
            }
        )
    for row in similarity_rows:
        rows.append(
            {
                "chemical_inchikey": row["chemical_inchikey"],
                "enzyme_inchikey": row["enzyme_inchikey"],
                "chemical_smiles": row["chemical_smiles"],
                "enzyme_smiles": row["enzyme_smiles"],
                "bridge_direction": row["bridge_direction"],
                "label": 1,
                "label_type": row["confidence_tier"],
                "label_weight": 0.65,
                "tanimoto": row["tanimoto"],
                "enzyme_ec_sample_json": row.get("enzyme_ec_sample_json") or "[]",
                "source": "similarity_bridge_filtered",
            }
        )
    return rows


def build_hard_negatives(
    root: Path,
    positive_rows: list[dict[str, Any]],
    near_similarity_rows: list[dict[str, Any]],
    *,
    target_negatives: int,
    seed: int,
    augment_enzyme_pool: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(seed)
    counts = Counter()
    enzyme_pool = load_augmented_enzyme_pool(root, augment_from_reaction_pool=augment_enzyme_pool)
    blacklist_rows = read_rows(root / "cofactor_common_metabolite_blacklist.parquet")
    audit_rows = read_rows(root / "exact_bridge_audit_only.parquet")
    exact_positive_keys = {(row["chemical_inchikey"], row["enzyme_inchikey"]) for row in positive_rows}

    fp_cache: dict[str, Any] = {}

    def fp(smiles: str):
        value = fp_cache.get(smiles)
        if value is None and smiles not in fp_cache:
            value = mol_fp(smiles)
            fp_cache[smiles] = value
        return value

    enzymes_by_ec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    enzymes_by_heavy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_enzymes = []
    for row in enzyme_pool:
        heavy = int(row.get("heavy_atoms") or 0)
        if heavy < 2 or heavy > 220:
            continue
        all_enzymes.append(row)
        enzymes_by_heavy[heavy // 5].append(row)
        for ec in row_ecs(row):
            enzymes_by_ec[ec].append(row)

    common_candidates = []
    for row in blacklist_rows:
        common_candidates.append(
            {
                "inchikey": row["inchikey"],
                "canonical_smiles": row["canonical_smiles"],
                "ec_sample_json": "[]",
                "heavy_atoms": row["heavy_atoms"],
            }
        )
    for row in audit_rows:
        common_candidates.append(
            {
                "inchikey": row["inchikey"],
                "canonical_smiles": row["canonical_smiles"],
                "ec_sample_json": row.get("enzyme_ec_sample_json") or "[]",
                "heavy_atoms": row["heavy_atoms"],
            }
        )

    near_by_chem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in near_similarity_rows:
        near_by_chem[row["chemical_inchikey"]].append(row)

    negatives: list[dict[str, Any]] = []
    seen_neg = set()

    def add_neg(pos: dict[str, Any], enz: dict[str, Any], neg_type: str, sim: float | None = None) -> bool:
        key = (pos["chemical_inchikey"], enz["inchikey"], neg_type)
        if key in seen_neg or (pos["chemical_inchikey"], enz["inchikey"]) in exact_positive_keys:
            return False
        seen_neg.add(key)
        if sim is None:
            cfp = fp(pos["chemical_smiles"])
            efp = fp(enz.get("canonical_smiles") or "")
            sim = float(DataStructs.TanimotoSimilarity(cfp, efp)) if cfp is not None and efp is not None else 0.0
        neg = {
            "chemical_inchikey": pos["chemical_inchikey"],
            "enzyme_inchikey": enz["inchikey"],
            "chemical_smiles": pos["chemical_smiles"],
            "enzyme_smiles": enz.get("canonical_smiles") or "",
            "bridge_direction": pos.get("bridge_direction") or "",
            "label": 0,
            "label_type": neg_type,
            "label_weight": 1.0,
            "tanimoto": round(float(sim), 6),
            "enzyme_ec_sample_json": enz.get("ec_sample_json") or "[]",
            "source": "hard_negative_pool",
        }
        negatives.append(neg)
        counts[f"negative_{neg_type}"] += 1
        return True

    pos_cycle = list(positive_rows)
    rng.shuffle(pos_cycle)
    attempts = 0
    while len(negatives) < target_negatives and attempts < target_negatives * 20:
        pos = pos_cycle[attempts % len(pos_cycle)]
        attempts += 1
        neg_type = attempts % 5
        if neg_type == 0:
            ecs = json_loads(pos.get("enzyme_ec_sample_json"), [])
            candidates = []
            for ec in ecs[:4]:
                candidates.extend(enzymes_by_ec.get(ec, [])[:200])
            if candidates:
                add_neg(pos, rng.choice(candidates), "same_ec_wrong_molecule")
        elif neg_type == 1:
            c_heavy = Chem.MolFromSmiles(pos["chemical_smiles"]).GetNumHeavyAtoms()
            candidates = []
            for bucket in range(max(0, c_heavy // 5 - 1), c_heavy // 5 + 2):
                candidates.extend(enzymes_by_heavy.get(bucket, [])[:300])
            if candidates:
                add_neg(pos, rng.choice(candidates), "near_size_wrong_molecule")
        elif neg_type == 2:
            near = near_by_chem.get(pos["chemical_inchikey"]) or []
            if near:
                row = rng.choice(near)
                add_neg(
                    pos,
                    {
                        "inchikey": row["enzyme_inchikey"],
                        "canonical_smiles": row["enzyme_smiles"],
                        "ec_sample_json": row.get("enzyme_ec_sample_json") or "[]",
                        "heavy_atoms": row.get("enzyme_heavy_atoms") or 0,
                    },
                    "near_similarity_below_positive_threshold",
                    sim=float(row["tanimoto"]),
                )
        elif neg_type == 3:
            if common_candidates:
                add_neg(pos, rng.choice(common_candidates), "common_or_cofactor_artifact")
        else:
            add_neg(pos, rng.choice(all_enzymes), "random_easy_negative")

    counts["hard_negative_rows"] = len(negatives)
    counts["hard_negative_attempts"] = attempts
    return negatives, counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_verifier_splits(root: Path, positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for row in positives + negatives:
        split = split_name(row["chemical_inchikey"])
        out = dict(row)
        out["split"] = split
        split_rows[split].append(out)
    files = {}
    for split, rows in split_rows.items():
        jsonl = root / f"verifier_{split}.jsonl"
        parquet = root / f"verifier_{split}.parquet"
        write_jsonl(jsonl, rows)
        write_parquet(parquet, rows)
        files[f"verifier_{split}_jsonl"] = str(jsonl)
        files[f"verifier_{split}_parquet"] = str(parquet)
    summary = {
        split: {
            "rows": len(rows),
            "positives": sum(1 for row in rows if int(row["label"]) == 1),
            "negatives": sum(1 for row in rows if int(row["label"]) == 0),
        }
        for split, rows in split_rows.items()
    }
    return {"files": files, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build similarity bridge and verifier pack")
    parser.add_argument("--pack-dir", default="data/bridge_pack_v0")
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--near-threshold", type=float, default=0.62)
    parser.add_argument("--max-bucket", type=int, default=8000)
    parser.add_argument("--rare-bits", type=int, default=18)
    parser.add_argument("--min-shared-rare-bits", type=int, default=2)
    parser.add_argument("--max-similarity-positives", type=int, default=60000)
    parser.add_argument("--target-negatives", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--augment-enzyme-pool", action="store_true")
    args = parser.parse_args()

    started = time.time()
    root = Path(args.pack_dir)
    similarity, near_similarity, sim_counts = build_similarity_bridges(
        root,
        threshold=args.similarity_threshold,
        low_threshold=args.near_threshold,
        max_bucket=args.max_bucket,
        max_pairs=args.max_similarity_positives,
        rare_bits=args.rare_bits,
        min_shared_rare_bits=args.min_shared_rare_bits,
        augment_enzyme_pool=args.augment_enzyme_pool,
    )
    write_parquet(root / "similarity_bridge_filtered.parquet", similarity)
    write_parquet(root / "similarity_bridge_near_negative_candidates.parquet", near_similarity)

    positives = make_positive_rows(root, similarity)
    negatives, neg_counts = build_hard_negatives(
        root,
        positives,
        near_similarity,
        target_negatives=args.target_negatives,
        seed=args.seed,
        augment_enzyme_pool=args.augment_enzyme_pool,
    )
    write_parquet(root / "hard_negative_pool.parquet", negatives)
    split_info = write_verifier_splits(root, positives, negatives)

    counts = Counter()
    counts.update(sim_counts)
    counts.update(neg_counts)
    counts["positive_rows_total"] = len(positives)
    counts["positive_exact_rows"] = pq.read_table(root / "exact_bridge_strict.parquet").num_rows
    counts["positive_similarity_rows"] = len(similarity)
    counts["verifier_rows_total"] = len(positives) + len(negatives)

    files = {
        "similarity_bridge_filtered": str(root / "similarity_bridge_filtered.parquet"),
        "similarity_bridge_near_negative_candidates": str(root / "similarity_bridge_near_negative_candidates.parquet"),
        "hard_negative_pool": str(root / "hard_negative_pool.parquet"),
        "verifier_manifest": str(root / "verifier_manifest.json"),
        "verifier_report": str(root / "verifier_report.md"),
        **split_info["files"],
    }
    manifest = {
        "schema_version": "bridge_pack_v0.verifier_data.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "pack_dir": str(root),
        "parameters": vars(args),
        "files": files,
        "counts": dict(counts),
        "splits": split_info["summary"],
    }
    (root / "verifier_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "verifier_report.md").write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "splits": manifest["splits"]}, indent=2, ensure_ascii=False))


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Bridge Pack v0 Verifier Data Report",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- elapsed_seconds: `{manifest['elapsed_seconds']}`",
        "",
        "## Counts",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(manifest["counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Splits", "", "| Split | Rows | Positives | Negatives |", "|---|---:|---:|---:|"])
    for split, row in manifest["splits"].items():
        lines.append(f"| `{split}` | {row['rows']} | {row['positives']} | {row['negatives']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
