"""Build the first enzyme-bridge data pack.

This script intentionally starts with cheap, auditable artifacts:

* chemical product pool
* enzyme substrate/product pool
* common/cofactor artifact blacklist
* exact bridge links
* confidence tiers for exact links

Similarity bridge search and hard-negative mining are deliberately left for the
next stage so that this first pack stays deterministic and easy to audit.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchiKey


RDLogger.DisableLog("rdApp.*")


MANUAL_COMMON_SMILES = [
    "O",
    "[H+]",
    "[OH-]",
    "O=O",
    "N",
    "[NH4+]",
    "O=C=O",
    "C(=O)([O-])[O-]",
    "O=P(O)(O)O",
    "O=P([O-])([O-])[O-]",
    "O=S(=O)(O)O",
    "Cl",
    "[Cl-]",
    "[Na+]",
    "[K+]",
    "[Mg+2]",
    "[Ca+2]",
    "CCO",
    "CO",
    "CC(=O)O",
    "CC(=O)[O-]",
    "O=C(O)C(O)C(O)C(O)C(O)CO",
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O",
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O",
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O",
]


@dataclass
class MolRecord:
    canonical_smiles: str
    inchikey: str
    formula: str
    heavy_atoms: int
    occurrences: int = 0
    source_counts: Counter[str] = field(default_factory=Counter)
    role_counts: Counter[str] = field(default_factory=Counter)
    ec_counts: Counter[str] = field(default_factory=Counter)
    example_ids: list[str] = field(default_factory=list)

    def add(
        self,
        *,
        source: str,
        role: str = "",
        ec: str = "",
        example_id: str = "",
    ) -> None:
        self.occurrences += 1
        if source:
            self.source_counts[source] += 1
        if role:
            self.role_counts[role] += 1
        if ec:
            self.ec_counts[ec] += 1
        if example_id and len(self.example_ids) < 5:
            self.example_ids.append(example_id)


class Canonicalizer:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any] | None] = {}
        self.invalid = Counter()
        self.invalid_examples: list[dict[str, str]] = []

    def canonicalize(self, smiles: str, *, context: str = "") -> dict[str, Any] | None:
        raw = clean_smiles_token(smiles)
        if not raw:
            return None
        if raw in self.cache:
            return self.cache[raw]
        try:
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                self._invalid(raw, context, "mol_from_smiles_failed")
                self.cache[raw] = None
                return None
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol2 = Chem.MolFromSmiles(canonical)
            if mol2 is None:
                self._invalid(raw, context, "canonical_mol_failed")
                self.cache[raw] = None
                return None
            info = {
                "canonical_smiles": canonical,
                "inchikey": MolToInchiKey(mol2),
                "formula": rdMolDescriptors.CalcMolFormula(mol2),
                "heavy_atoms": int(mol2.GetNumHeavyAtoms()),
            }
            self.cache[raw] = info
            return info
        except Exception as exc:
            self._invalid(raw, context, type(exc).__name__)
            self.cache[raw] = None
            return None

    def _invalid(self, smiles: str, context: str, reason: str) -> None:
        self.invalid[reason] += 1
        if len(self.invalid_examples) < 50:
            self.invalid_examples.append({"smiles": smiles[:240], "context": context, "reason": reason})


def clean_smiles_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def split_mols(side: str) -> list[str]:
    side = clean_smiles_token(side)
    if not side:
        return []
    return [token for token in side.split(".") if clean_smiles_token(token)]


def split_rxn_smiles(rxn_smiles: str) -> tuple[list[str], list[str]]:
    text = clean_smiles_token(rxn_smiles)
    if ">>" not in text:
        return [], []
    left, right = text.split(">>", 1)
    if "|" in left:
        left = left.split("|", 1)[0]
    if "|" in right:
        right = right.split("|", 1)[0]
    return split_mols(left), split_mols(right)


def add_record(
    pool: dict[str, MolRecord],
    canonicalizer: Canonicalizer,
    smiles: str,
    *,
    source: str,
    role: str = "",
    ec: str = "",
    example_id: str = "",
    context: str = "",
) -> bool:
    info = canonicalizer.canonicalize(smiles, context=context or source)
    if not info or not info.get("inchikey"):
        return False
    key = info["inchikey"]
    rec = pool.get(key)
    if rec is None:
        rec = MolRecord(
            canonical_smiles=info["canonical_smiles"],
            inchikey=key,
            formula=info["formula"],
            heavy_atoms=info["heavy_atoms"],
        )
        pool[key] = rec
    rec.add(source=source, role=role, ec=ec, example_id=example_id)
    return True


def load_external_chemical_products(
    paths: Iterable[Path],
    pool: dict[str, MolRecord],
    canonicalizer: Canonicalizer,
) -> Counter:
    counts = Counter()
    for path in paths:
        if not path.exists():
            counts[f"missing:{path}"] += 1
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                counts["chemical_rows"] += 1
                row = json.loads(line)
                product = row.get("product") or row.get("target_smiles")
                source = f"chemical:{row.get('source') or path.stem}"
                example_id = f"{path.name}:{line_no}"
                if add_record(
                    pool,
                    canonicalizer,
                    product,
                    source=source,
                    role="product",
                    ec=str(row.get("ec") or ""),
                    example_id=example_id,
                    context="chemical_product",
                ):
                    counts["chemical_product_occurrences"] += 1
    counts["chemical_unique_products"] = len(pool)
    return counts


def load_enzymemap(
    path: Path,
    pool: dict[str, MolRecord],
    canonicalizer: Canonicalizer,
    *,
    max_rows: int | None = None,
) -> Counter:
    counts = Counter()
    if not path.exists():
        counts["enzymemap_missing"] += 1
        return counts
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            counts["enzymemap_rows"] += 1
            ec = row.get("ec_num") or ""
            rxn = row.get("unmapped") or row.get("mapped") or ""
            substrates, products = split_rxn_smiles(rxn)
            example_id = f"enzymemap:{row.get('rxn_idx') or idx}"
            for smiles in substrates:
                if add_record(
                    pool,
                    canonicalizer,
                    smiles,
                    source="enzyme:enzymemap",
                    role="substrate",
                    ec=ec,
                    example_id=example_id,
                    context="enzymemap_substrate",
                ):
                    counts["enzymemap_substrate_occurrences"] += 1
            for smiles in products:
                if add_record(
                    pool,
                    canonicalizer,
                    smiles,
                    source="enzyme:enzymemap",
                    role="product",
                    ec=ec,
                    example_id=example_id,
                    context="enzymemap_product",
                ):
                    counts["enzymemap_product_occurrences"] += 1
    return counts


def load_ecreact(
    path: Path,
    pool: dict[str, MolRecord],
    canonicalizer: Canonicalizer,
    *,
    max_rows: int | None = None,
) -> Counter:
    counts = Counter()
    if not path.exists():
        counts["ecreact_missing"] += 1
        return counts
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            counts["ecreact_rows"] += 1
            ec = row.get("ec") or ""
            substrates, products = split_rxn_smiles(row.get("rxn_smiles") or "")
            example_id = f"ecreact:{idx}"
            for smiles in substrates:
                if add_record(
                    pool,
                    canonicalizer,
                    smiles,
                    source=f"enzyme:ecreact:{row.get('source') or 'unknown'}",
                    role="substrate",
                    ec=ec,
                    example_id=example_id,
                    context="ecreact_substrate",
                ):
                    counts["ecreact_substrate_occurrences"] += 1
            for smiles in products:
                if add_record(
                    pool,
                    canonicalizer,
                    smiles,
                    source=f"enzyme:ecreact:{row.get('source') or 'unknown'}",
                    role="product",
                    ec=ec,
                    example_id=example_id,
                    context="ecreact_product",
                ):
                    counts["ecreact_product_occurrences"] += 1
    return counts


def load_enzymatic_retro_json(
    paths: Iterable[Path],
    pool: dict[str, MolRecord],
    canonicalizer: Canonicalizer,
    *,
    max_rows_per_file: int | None = None,
) -> Counter:
    counts = Counter()
    for path in paths:
        if not path.exists():
            counts[f"missing:{path}"] += 1
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows):
            if max_rows_per_file is not None and idx >= max_rows_per_file:
                break
            counts[f"enzymatic_retro_{path.stem}_rows"] += 1
            ec = row.get("ec") or ""
            example_id = f"enzymatic_retro:{path.stem}:{idx}"
            for smiles in split_mols(row.get("reactants") or ""):
                if add_record(
                    pool,
                    canonicalizer,
                    smiles,
                    source=f"enzyme:enzymatic_retro:{path.stem}",
                    role="substrate",
                    ec=ec,
                    example_id=example_id,
                    context="enzymatic_retro_substrate",
                ):
                    counts["enzymatic_retro_substrate_occurrences"] += 1
            for smiles in split_mols(row.get("product") or ""):
                if add_record(
                    pool,
                    canonicalizer,
                    smiles,
                    source=f"enzyme:enzymatic_retro:{path.stem}",
                    role="product",
                    ec=ec,
                    example_id=example_id,
                    context="enzymatic_retro_product",
                ):
                    counts["enzymatic_retro_product_occurrences"] += 1
    return counts


def top_counter(counter: Counter[str], n: int = 12) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(n)]


def pool_rows(pool: dict[str, MolRecord], *, pool_name: str) -> list[dict[str, Any]]:
    rows = []
    for rec in sorted(pool.values(), key=lambda item: (item.inchikey, item.canonical_smiles)):
        rows.append(
            {
                "pool": pool_name,
                "canonical_smiles": rec.canonical_smiles,
                "inchikey": rec.inchikey,
                "formula": rec.formula,
                "heavy_atoms": rec.heavy_atoms,
                "occurrences": rec.occurrences,
                "source_counts_json": json.dumps(dict(rec.source_counts), sort_keys=True),
                "role_counts_json": json.dumps(dict(rec.role_counts), sort_keys=True),
                "ec_sample_json": json.dumps([value for value, _ in rec.ec_counts.most_common(20)]),
                "ec_unique": len(rec.ec_counts),
                "example_ids_json": json.dumps(rec.example_ids),
            }
        )
    return rows


def build_common_blacklist(
    enzyme_pool: dict[str, MolRecord],
    canonicalizer: Canonicalizer,
    *,
    min_occurrences: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    blacklist: dict[str, dict[str, Any]] = {}

    def add(info: dict[str, Any], reason: str, source: str, occurrences: int = 0) -> None:
        inchikey = info.get("inchikey")
        if not inchikey:
            return
        row = blacklist.setdefault(
            inchikey,
            {
                "canonical_smiles": info.get("canonical_smiles") or "",
                "inchikey": inchikey,
                "formula": info.get("formula") or "",
                "heavy_atoms": int(info.get("heavy_atoms") or 0),
                "reasons": set(),
                "sources": set(),
                "enzyme_occurrences": 0,
            },
        )
        row["reasons"].add(reason)
        row["sources"].add(source)
        row["enzyme_occurrences"] = max(int(row["enzyme_occurrences"]), int(occurrences or 0))

    for smiles in MANUAL_COMMON_SMILES:
        info = canonicalizer.canonicalize(smiles, context="manual_common")
        if info:
            add(info, "manual_common_or_cofactor", "manual")

    for rec in enzyme_pool.values():
        info = {
            "canonical_smiles": rec.canonical_smiles,
            "inchikey": rec.inchikey,
            "formula": rec.formula,
            "heavy_atoms": rec.heavy_atoms,
        }
        if rec.heavy_atoms <= 4:
            add(info, "small_molecule_heavy_atoms_le_4", "heuristic", rec.occurrences)
        if rec.occurrences >= min_occurrences:
            add(info, f"high_enzyme_frequency_ge_{min_occurrences}", "heuristic", rec.occurrences)

    rows = []
    for row in blacklist.values():
        rows.append(
            {
                "canonical_smiles": row["canonical_smiles"],
                "inchikey": row["inchikey"],
                "formula": row["formula"],
                "heavy_atoms": row["heavy_atoms"],
                "enzyme_occurrences": row["enzyme_occurrences"],
                "reasons_json": json.dumps(sorted(row["reasons"])),
                "sources_json": json.dumps(sorted(row["sources"])),
            }
        )
    rows.sort(key=lambda item: (-int(item["enzyme_occurrences"]), item["inchikey"]))
    return blacklist, rows


def build_exact_bridges(
    chemical_pool: dict[str, MolRecord],
    enzyme_pool: dict[str, MolRecord],
    blacklist: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    filtered_rows = []
    for inchikey in sorted(set(chemical_pool) & set(enzyme_pool)):
        chem = chemical_pool[inchikey]
        enz = enzyme_pool[inchikey]
        is_common = inchikey in blacklist
        roles = enz.role_counts
        directions = []
        if roles.get("substrate", 0) > 0:
            directions.append("chemical_product_to_enzyme_substrate")
        if roles.get("product", 0) > 0:
            directions.append("chemical_product_to_enzyme_product")
        for direction in directions or ["chemical_product_to_enzyme_molecule"]:
            if is_common:
                tier = "tier5_common_or_cofactor_artifact"
            elif direction.endswith("substrate"):
                tier = "tier1_exact_noncommon_substrate_bridge"
            else:
                tier = "tier2_exact_noncommon_product_bridge"
            row = {
                "inchikey": inchikey,
                "canonical_smiles": chem.canonical_smiles,
                "formula": chem.formula,
                "heavy_atoms": chem.heavy_atoms,
                "bridge_direction": direction,
                "confidence_tier": tier,
                "is_common_or_cofactor_like": is_common,
                "chemical_occurrences": chem.occurrences,
                "enzyme_occurrences": enz.occurrences,
                "enzyme_substrate_occurrences": int(roles.get("substrate", 0)),
                "enzyme_product_occurrences": int(roles.get("product", 0)),
                "chemical_sources_json": json.dumps(dict(chem.source_counts), sort_keys=True),
                "enzyme_sources_json": json.dumps(dict(enz.source_counts), sort_keys=True),
                "enzyme_ec_sample_json": json.dumps([value for value, _ in enz.ec_counts.most_common(20)]),
                "enzyme_ec_unique": len(enz.ec_counts),
                "chemical_example_ids_json": json.dumps(chem.example_ids),
                "enzyme_example_ids_json": json.dumps(enz.example_ids),
            }
            all_rows.append(row)
            if not is_common:
                filtered_rows.append(row)
    return all_rows, filtered_rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    files = manifest["files"]
    lines = [
        "# Bridge Pack v0 Report",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- mode: `{manifest['mode']}`",
        "",
        "## Counts",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key in sorted(counts):
        value = counts[key]
        if isinstance(value, (int, float, str)):
            lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Files", "", "| File | Path |", "|---|---|"])
    for key, value in files.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the first deterministic bridge pack. It only uses exact molecule identity for bridge links. "
            "Similarity bridges, reaction-center filters, stereochemistry filters, and hard negatives still need a follow-up stage.",
            "",
            "Rows marked `tier5_common_or_cofactor_artifact` are retained in `exact_bridge_all.parquet` for audit but excluded from `exact_bridge.parquet`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bridge pack v0")
    parser.add_argument("--output-dir", default="data/bridge_pack_v0")
    parser.add_argument("--max-enzymemap-rows", type=int, default=None)
    parser.add_argument("--max-ecreact-rows", type=int, default=None)
    parser.add_argument("--max-enzymatic-retro-rows-per-file", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonicalizer = Canonicalizer()
    chemical_pool: dict[str, MolRecord] = {}
    enzyme_pool: dict[str, MolRecord] = {}
    counts = Counter()

    chemical_paths = [
        Path("results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_150k/plain.train.meta.jsonl"),
        Path("results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_150k/plain.valid.meta.jsonl"),
        Path("results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_150k/plain.test.meta.jsonl"),
    ]
    counts.update(load_external_chemical_products(chemical_paths, chemical_pool, canonicalizer))
    print(f"[bridge_pack_v0] chemical pool unique={len(chemical_pool)}")

    counts.update(
        load_enzymemap(
            Path("data_external/enzymemap/enzymemap_v2_brenda2023.csv.gz"),
            enzyme_pool,
            canonicalizer,
            max_rows=args.max_enzymemap_rows,
        )
    )
    print(f"[bridge_pack_v0] after EnzymeMap enzyme pool unique={len(enzyme_pool)}")

    counts.update(
        load_ecreact(
            Path("data_external/ecreact/ecreact-1.0.csv"),
            enzyme_pool,
            canonicalizer,
            max_rows=args.max_ecreact_rows,
        )
    )
    print(f"[bridge_pack_v0] after ECREACT enzyme pool unique={len(enzyme_pool)}")

    counts.update(
        load_enzymatic_retro_json(
            [
                Path("data_external/enzymatic_retro_data/train.json"),
                Path("data_external/enzymatic_retro_data/val.json"),
            ],
            enzyme_pool,
            canonicalizer,
            max_rows_per_file=args.max_enzymatic_retro_rows_per_file,
        )
    )
    print(f"[bridge_pack_v0] after enzymatic_retro enzyme pool unique={len(enzyme_pool)}")

    counts["chemical_unique_products"] = len(chemical_pool)
    counts["enzyme_unique_molecules"] = len(enzyme_pool)
    counts["canonicalizer_cache_size"] = len(canonicalizer.cache)
    counts["invalid_smiles_total"] = sum(canonicalizer.invalid.values())

    min_common_occurrences = max(1000, int(sum(rec.occurrences for rec in enzyme_pool.values()) * 0.0025))
    blacklist, blacklist_rows = build_common_blacklist(
        enzyme_pool,
        canonicalizer,
        min_occurrences=min_common_occurrences,
    )
    counts["common_blacklist_rows"] = len(blacklist_rows)
    counts["common_frequency_threshold"] = min_common_occurrences

    exact_all, exact_filtered = build_exact_bridges(chemical_pool, enzyme_pool, blacklist)
    counts["exact_bridge_all_rows"] = len(exact_all)
    counts["exact_bridge_filtered_rows"] = len(exact_filtered)
    counts["exact_bridge_common_artifact_rows"] = len(exact_all) - len(exact_filtered)

    chemical_rows = pool_rows(chemical_pool, pool_name="chemical_product")
    enzyme_rows = pool_rows(enzyme_pool, pool_name="enzyme_substrate_product")

    files = {
        "chemical_product_pool": str(out_dir / "chemical_product_pool.parquet"),
        "enzyme_substrate_product_pool": str(out_dir / "enzyme_substrate_product_pool.parquet"),
        "cofactor_common_metabolite_blacklist": str(out_dir / "cofactor_common_metabolite_blacklist.parquet"),
        "exact_bridge_all": str(out_dir / "exact_bridge_all.parquet"),
        "exact_bridge": str(out_dir / "exact_bridge.parquet"),
        "bridge_confidence_tiers": str(out_dir / "bridge_confidence_tiers.parquet"),
        "manifest": str(out_dir / "manifest.json"),
        "report": str(out_dir / "report.md"),
    }
    write_parquet(Path(files["chemical_product_pool"]), chemical_rows)
    write_parquet(Path(files["enzyme_substrate_product_pool"]), enzyme_rows)
    write_parquet(Path(files["cofactor_common_metabolite_blacklist"]), blacklist_rows)
    write_parquet(Path(files["exact_bridge_all"]), exact_all)
    write_parquet(Path(files["exact_bridge"]), exact_filtered)
    write_parquet(Path(files["bridge_confidence_tiers"]), exact_all)

    mode = "full"
    if any(
        value is not None
        for value in [args.max_enzymemap_rows, args.max_ecreact_rows, args.max_enzymatic_retro_rows_per_file]
    ):
        mode = "limited"
    manifest = {
        "schema_version": "bridge_pack_v0.1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": str(out_dir),
        "mode": mode,
        "input_sources": {
            "chemical_paths": [str(path) for path in chemical_paths],
            "enzymemap": "data_external/enzymemap/enzymemap_v2_brenda2023.csv.gz",
            "ecreact": "data_external/ecreact/ecreact-1.0.csv",
            "enzymatic_retro": [
                "data_external/enzymatic_retro_data/train.json",
                "data_external/enzymatic_retro_data/val.json",
            ],
            "not_yet_integrated": [
                "data_external/rhea/140.tar.bz2",
                "data_external/reactzyme/13635807.zip",
            ],
        },
        "limits": {
            "max_enzymemap_rows": args.max_enzymemap_rows,
            "max_ecreact_rows": args.max_ecreact_rows,
            "max_enzymatic_retro_rows_per_file": args.max_enzymatic_retro_rows_per_file,
        },
        "files": files,
        "counts": dict(counts),
        "invalid_smiles": {
            "counts": dict(canonicalizer.invalid),
            "examples": canonicalizer.invalid_examples,
        },
    }
    write_json(Path(files["manifest"]), manifest)
    Path(files["report"]).write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
