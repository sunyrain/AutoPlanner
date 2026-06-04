"""Supplement bridge_pack_v0 with reaction and sequence level tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchiKey


RDLogger.DisableLog("rdApp.*")


@dataclass
class ReactionRecord:
    substrate_key: str
    product_key: str
    substrate_smiles: str
    product_smiles: str
    occurrences: int = 0
    source_counts: Counter[str] = field(default_factory=Counter)
    ec_counts: Counter[str] = field(default_factory=Counter)
    rhea_counts: Counter[str] = field(default_factory=Counter)
    example_ids: list[str] = field(default_factory=list)

    def add(self, *, source: str, ec: str = "", rhea_id: str = "", example_id: str = "") -> None:
        self.occurrences += 1
        if source:
            self.source_counts[source] += 1
        for value in split_multi(ec):
            self.ec_counts[value] += 1
        for value in split_multi(rhea_id):
            self.rhea_counts[value] += 1
        if example_id and len(self.example_ids) < 8:
            self.example_ids.append(example_id)


class Canonicalizer:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any] | None] = {}
        self.invalid = Counter()
        self.invalid_examples: list[dict[str, str]] = []

    def canonicalize(self, smiles: str, *, context: str = "") -> dict[str, Any] | None:
        raw = str(smiles or "").strip()
        if not raw or raw.lower() in {"none", "nan", "null"}:
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
            out = {
                "canonical_smiles": canonical,
                "inchikey": MolToInchiKey(mol2),
                "formula": rdMolDescriptors.CalcMolFormula(mol2),
                "heavy_atoms": int(mol2.GetNumHeavyAtoms()),
            }
            self.cache[raw] = out
            return out
        except Exception as exc:
            self._invalid(raw, context, type(exc).__name__)
            self.cache[raw] = None
            return None

    def _invalid(self, smiles: str, context: str, reason: str) -> None:
        self.invalid[reason] += 1
        if len(self.invalid_examples) < 50:
            self.invalid_examples.append({"smiles": smiles[:240], "context": context, "reason": reason})


def split_multi(value: str | None) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan"}:
        return []
    out = []
    for sep in [";", ",", "|"]:
        if sep in text:
            for token in text.split(sep):
                token = token.strip()
                if token:
                    out.append(token)
            return out
    return [text]


def split_mols(side: str) -> list[str]:
    text = str(side or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(".") if token.strip()]


def split_rxn_smiles(rxn: str) -> tuple[list[str], list[str]]:
    text = str(rxn or "").strip()
    if ">>" not in text:
        return [], []
    left, right = text.split(">>", 1)
    if "|" in left:
        left = left.split("|", 1)[0]
    if "|" in right:
        right = right.split("|", 1)[0]
    return split_mols(left), split_mols(right)


def canonical_side_key(mols: Iterable[str], canonicalizer: Canonicalizer, *, context: str) -> tuple[str, str] | None:
    canonical = []
    inchikeys = []
    for smiles in mols:
        info = canonicalizer.canonicalize(smiles, context=context)
        if not info:
            continue
        canonical.append(info["canonical_smiles"])
        inchikeys.append(info["inchikey"])
    if not canonical:
        return None
    canonical_sorted = sorted(canonical)
    inchikey_sorted = sorted(inchikeys)
    return ".".join(inchikey_sorted), ".".join(canonical_sorted)


def add_reaction(
    reactions: dict[str, ReactionRecord],
    canonicalizer: Canonicalizer,
    substrates: Iterable[str],
    products: Iterable[str],
    *,
    source: str,
    ec: str = "",
    rhea_id: str = "",
    example_id: str = "",
) -> bool:
    left = canonical_side_key(substrates, canonicalizer, context=f"{source}:substrate")
    right = canonical_side_key(products, canonicalizer, context=f"{source}:product")
    if not left or not right:
        return False
    substrate_key, substrate_smiles = left
    product_key, product_smiles = right
    key = f"{substrate_key}>>{product_key}"
    rec = reactions.get(key)
    if rec is None:
        rec = ReactionRecord(
            substrate_key=substrate_key,
            product_key=product_key,
            substrate_smiles=substrate_smiles,
            product_smiles=product_smiles,
        )
        reactions[key] = rec
    rec.add(source=source, ec=ec, rhea_id=rhea_id, example_id=example_id)
    return True


def load_enzymemap(reactions: dict[str, ReactionRecord], canonicalizer: Canonicalizer) -> Counter:
    counts = Counter()
    path = Path("data_external/enzymemap/enzymemap_v2_brenda2023.csv.gz")
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            counts["enzymemap_rows"] += 1
            substrates, products = split_rxn_smiles(row.get("unmapped") or row.get("mapped") or "")
            if add_reaction(
                reactions,
                canonicalizer,
                substrates,
                products,
                source="enzymemap",
                ec=row.get("ec_num") or "",
                example_id=f"enzymemap:{row.get('rxn_idx') or idx}",
            ):
                counts["enzymemap_reactions_added"] += 1
    return counts


def load_ecreact(reactions: dict[str, ReactionRecord], canonicalizer: Canonicalizer) -> Counter:
    counts = Counter()
    path = Path("data_external/ecreact/ecreact-1.0.csv")
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            counts["ecreact_rows"] += 1
            substrates, products = split_rxn_smiles(row.get("rxn_smiles") or "")
            if add_reaction(
                reactions,
                canonicalizer,
                substrates,
                products,
                source=f"ecreact:{row.get('source') or 'unknown'}",
                ec=row.get("ec") or "",
                example_id=f"ecreact:{idx}",
            ):
                counts["ecreact_reactions_added"] += 1
    return counts


def load_enzymatic_retro(reactions: dict[str, ReactionRecord], canonicalizer: Canonicalizer) -> Counter:
    counts = Counter()
    for path in [Path("data_external/enzymatic_retro_data/train.json"), Path("data_external/enzymatic_retro_data/val.json")]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows):
            counts[f"enzymatic_retro_{path.stem}_rows"] += 1
            if add_reaction(
                reactions,
                canonicalizer,
                split_mols(row.get("reactants") or ""),
                split_mols(row.get("product") or ""),
                source=f"enzymatic_retro:{path.stem}",
                ec=row.get("ec") or "",
                example_id=f"enzymatic_retro:{path.stem}:{idx}",
            ):
                counts[f"enzymatic_retro_{path.stem}_reactions_added"] += 1
    return counts


def load_reactzyme_rhea(reactions: dict[str, ReactionRecord], canonicalizer: Canonicalizer) -> Counter:
    counts = Counter()
    path = Path("data_external/reactzyme/13635807.zip")
    if not path.exists():
        counts["reactzyme_zip_missing"] += 1
        return counts
    with zipfile.ZipFile(path) as zf:
        with zf.open("rhea_molecules.tsv") as raw:
            handle = (line.decode("utf-8", "replace") for line in raw)
            reader = csv.DictReader(handle, delimiter="\t")
            for idx, row in enumerate(reader):
                counts["reactzyme_rhea_rows"] += 1
                if add_reaction(
                    reactions,
                    canonicalizer,
                    split_mols(row.get("substrate") or ""),
                    split_mols(row.get("product") or ""),
                    source="reactzyme:rhea_molecules",
                    rhea_id=row.get("Rhea ID") or "",
                    example_id=f"reactzyme_rhea:{idx}",
                ):
                    counts["reactzyme_rhea_reactions_added"] += 1
    return counts


def build_sequence_pool() -> tuple[list[dict[str, Any]], Counter]:
    counts = Counter()
    rows: dict[str, dict[str, Any]] = {}

    def add(uniprot: str, ecs: str, rhea_ids: str, sequence: str, source: str) -> None:
        uniprot = (uniprot or "").strip()
        sequence = (sequence or "").strip()
        if not uniprot or not sequence:
            return
        row = rows.setdefault(
            uniprot,
            {
                "uniprot_id": uniprot,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "ec_numbers": set(),
                "rhea_ids": set(),
                "sources": set(),
            },
        )
        row["sources"].add(source)
        for ec in split_multi(ecs):
            row["ec_numbers"].add(ec)
        for rhea_id in split_multi(rhea_ids):
            row["rhea_ids"].add(rhea_id)
        counts[f"{source}_sequence_rows_added"] += 1

    local = Path("data_external/enzyme_sequences/autoplanner_enzymes.tsv")
    if local.exists():
        with local.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                counts["local_sequence_rows"] += 1
                add(row.get("uniprot_id") or "", row.get("ec_number") or "", "", row.get("sequence") or "", "local_autoplanner")

    zip_path = Path("data_external/reactzyme/13635807.zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("cleaned_uniprot_rhea.tsv") as raw:
                handle = (line.decode("utf-8", "replace") for line in raw)
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    counts["reactzyme_sequence_rows"] += 1
                    add(
                        row.get("Entry") or "",
                        row.get("EC number") or "",
                        row.get("Rhea ID") or "",
                        row.get("Sequence") or "",
                        "reactzyme_cleaned_uniprot_rhea",
                    )

    out = []
    for row in rows.values():
        out.append(
            {
                "uniprot_id": row["uniprot_id"],
                "sequence": row["sequence"],
                "sequence_length": row["sequence_length"],
                "ec_numbers_json": json.dumps(sorted(row["ec_numbers"])),
                "rhea_ids_json": json.dumps(sorted(row["rhea_ids"])),
                "source_json": json.dumps(sorted(row["sources"])),
                "ec_count": len(row["ec_numbers"]),
                "rhea_count": len(row["rhea_ids"]),
            }
        )
    out.sort(key=lambda item: item["uniprot_id"])
    counts["sequence_unique_uniprot"] = len(out)
    counts["sequence_with_rhea"] = sum(1 for row in out if row["rhea_count"] > 0)
    counts["sequence_with_ec"] = sum(1 for row in out if row["ec_count"] > 0)
    return out, counts


def reaction_rows(reactions: dict[str, ReactionRecord]) -> list[dict[str, Any]]:
    rows = []
    for idx, rec in enumerate(reactions.values()):
        rows.append(
            {
                "reaction_id": f"enzrxn_{idx:08d}",
                "substrate_key": rec.substrate_key,
                "product_key": rec.product_key,
                "substrate_smiles": rec.substrate_smiles,
                "product_smiles": rec.product_smiles,
                "reaction_smiles": f"{rec.substrate_smiles}>>{rec.product_smiles}",
                "occurrences": rec.occurrences,
                "source_counts_json": json.dumps(dict(rec.source_counts), sort_keys=True),
                "ec_numbers_json": json.dumps([value for value, _ in rec.ec_counts.most_common(30)]),
                "ec_unique": len(rec.ec_counts),
                "rhea_ids_json": json.dumps([value for value, _ in rec.rhea_counts.most_common(30)]),
                "rhea_unique": len(rec.rhea_counts),
                "example_ids_json": json.dumps(rec.example_ids),
            }
        )
    rows.sort(key=lambda item: (-int(item["occurrences"]), item["reaction_smiles"]))
    return rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bridge pack supplement tables")
    parser.add_argument("--pack-dir", default="data/bridge_pack_v0")
    args = parser.parse_args()

    started = time.time()
    root = Path(args.pack_dir)
    canonicalizer = Canonicalizer()
    reactions: dict[str, ReactionRecord] = {}
    counts = Counter()

    counts.update(load_enzymemap(reactions, canonicalizer))
    print(f"[supplement] after EnzymeMap reactions={len(reactions)}")
    counts.update(load_ecreact(reactions, canonicalizer))
    print(f"[supplement] after ECREACT reactions={len(reactions)}")
    counts.update(load_enzymatic_retro(reactions, canonicalizer))
    print(f"[supplement] after enzymatic_retro reactions={len(reactions)}")
    counts.update(load_reactzyme_rhea(reactions, canonicalizer))
    print(f"[supplement] after ReactZyme-Rhea reactions={len(reactions)}")

    seq_rows, seq_counts = build_sequence_pool()
    counts.update(seq_counts)

    rxn_rows = reaction_rows(reactions)
    counts["enzyme_reaction_unique"] = len(rxn_rows)
    counts["canonicalizer_cache_size"] = len(canonicalizer.cache)
    counts["invalid_smiles_total"] = sum(canonicalizer.invalid.values())

    files = {
        "enzyme_reaction_pool": str(root / "enzyme_reaction_pool.parquet"),
        "enzyme_sequence_pool": str(root / "enzyme_sequence_pool.parquet"),
        "supplement_manifest": str(root / "supplement_manifest.json"),
        "supplement_report": str(root / "supplement_report.md"),
    }
    write_parquet(Path(files["enzyme_reaction_pool"]), rxn_rows)
    write_parquet(Path(files["enzyme_sequence_pool"]), seq_rows)
    manifest = {
        "schema_version": "bridge_pack_v0.supplement.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "pack_dir": str(root),
        "files": files,
        "counts": dict(counts),
        "invalid_smiles": {
            "counts": dict(canonicalizer.invalid),
            "examples": canonicalizer.invalid_examples,
        },
    }
    Path(files["supplement_manifest"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(files["supplement_report"]).write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Bridge Pack v0 Supplement Report",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- elapsed_seconds: `{manifest['elapsed_seconds']}`",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key in sorted(counts):
        value = counts[key]
        if isinstance(value, (int, float, str)):
            lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Files", "", "| File | Path |", "|---|---|"])
    for key, value in manifest["files"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
