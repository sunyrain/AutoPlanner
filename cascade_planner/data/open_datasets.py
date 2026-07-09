"""Download, cache, and convert external datasets into AutoPlanner StepRowV2 format.

Supported datasets:
  - EnzymeMap v2 (local gzipped CSV)
  - ReactZyme  (GitHub / Zenodo download)
  - USPTO-50K  (standard retrosynthesis benchmark)

Usage:
  python -m cascade_planner.data.open_datasets --verify
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem, RDLogger

from cascade_planner.data.loader_v2 import StepRowV2

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

_DATA_EXT = Path(__file__).resolve().parents[2] / "data_external"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canon_rxn(rxn: str) -> str | None:
    """Canonicalize a reaction SMILES (both sides independently)."""
    if not rxn or ">>" not in rxn:
        return None
    lhs, rhs = rxn.split(">>", 1)

    def _canon_side(side: str) -> str | None:
        frags = []
        for s in side.split("."):
            s = s.strip()
            if not s:
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            frags.append(Chem.MolToSmiles(m))
        if not frags:
            return None
        return ".".join(sorted(frags))

    cl = _canon_side(lhs)
    cr = _canon_side(rhs)
    if cl is None or cr is None:
        return None
    return f"{cl}>>{cr}"


def download_if_missing(url: str, dest: Path, *, desc: str = "") -> bool:
    """Download *url* to *dest* if the file does not already exist.

    Returns True on success, False on failure.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    logger.info("Downloading %s -> %s", label, dest)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoPlanner/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("Download failed for %s: %s", label, exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Wrapper that carries a source tag alongside StepRowV2
# ---------------------------------------------------------------------------

@dataclass
class TaggedStep:
    """Thin wrapper: a StepRowV2 plus a *source* tag for provenance."""
    step: StepRowV2
    source: str


# ---------------------------------------------------------------------------
# EnzymeMap v2
# ---------------------------------------------------------------------------

_ENZYMEMAP_LOCAL = _DATA_EXT / "enzymemap" / "enzymemap_v2_brenda2023.csv.gz"


def load_enzymemap(cache_dir: Path | None = None) -> list[TaggedStep]:
    """Load EnzymeMap v2 from the pre-downloaded gzipped CSV.

    Filters: quality >= 0.95, single-step only, deduplicated by unmapped SMILES.
    """
    path = _ENZYMEMAP_LOCAL
    if cache_dir:
        alt = cache_dir / "enzymemap" / _ENZYMEMAP_LOCAL.name
        if alt.exists():
            path = alt
    if not path.exists():
        logger.warning("EnzymeMap file not found at %s — skipping.", path)
        return []

    seen_unmapped: set[str] = set()
    rows: list[TaggedStep] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, rec in enumerate(reader):
                # quality filter
                try:
                    q = float(rec.get("quality", 0))
                except (TypeError, ValueError):
                    continue
                if q < 0.95:
                    continue
                # single-step only
                if (rec.get("steps") or "").strip().lower() != "single":
                    continue
                unmapped = (rec.get("unmapped") or "").strip()
                if not unmapped or ">>" not in unmapped:
                    continue
                # deduplicate
                if unmapped in seen_unmapped:
                    continue
                seen_unmapped.add(unmapped)

                ec = (rec.get("ec_num") or "").strip() or None
                rows.append(TaggedStep(
                    step=StepRowV2(
                        doi="enzymemap",
                        cascade_id=f"enzymemap_{i}",
                        step_id=f"enzymemap_{i}_0",
                        step_index=0,
                        rxn_smiles=unmapped,
                        pairwise_mode="not_applicable",
                        transformation_superclass=None,
                        ec_number=ec,
                        catalyst_class=None,
                        temperature_c=None,
                        ph=None,
                        solvent_smiles=None,
                    ),
                    source="enzymemap",
                ))
    except Exception as exc:
        logger.warning("Failed to parse EnzymeMap: %s", exc)
        return []

    logger.info("EnzymeMap: loaded %d steps (after quality/dedup filter).", len(rows))
    return rows


# ---------------------------------------------------------------------------
# ReactZyme
# ---------------------------------------------------------------------------

_REACTZYME_URLS = [
    # Primary: Zenodo archive
    "https://zenodo.org/records/13635807/files/reactzyme_data.tsv",
    # Fallback: GitHub raw (may change)
    "https://raw.githubusercontent.com/WillHua127/ReactZyme/main/data/reactzyme_data.tsv",
]


def load_reactzyme(cache_dir: Path | None = None) -> list[TaggedStep]:
    """Download (if needed) and load ReactZyme enzyme-reaction pairs.

    Also writes a ``sequences.tsv`` sidecar for downstream ESM-2 embedding.
    """
    dest_dir = (cache_dir or _DATA_EXT) / "reactzyme"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = dest_dir / "reactzyme_data.tsv"

    # Try each URL until one succeeds
    if not tsv_path.exists():
        for url in _REACTZYME_URLS:
            if download_if_missing(url, tsv_path, desc="ReactZyme"):
                break
    if not tsv_path.exists():
        logger.warning("ReactZyme download failed — skipping.")
        return []

    rows: list[TaggedStep] = []
    seq_map: dict[str, str] = {}  # uniprot_id -> sequence
    try:
        with open(tsv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for i, rec in enumerate(reader):
                rxn = (rec.get("reaction_smiles") or rec.get("rxn_smiles") or "").strip()
                if not rxn or ">>" not in rxn:
                    continue
                ec = (rec.get("ec_number") or rec.get("ec") or "").strip() or None
                uid = (rec.get("uniprot_id") or "").strip()
                seq = (rec.get("sequence") or "").strip()
                if uid and seq:
                    seq_map[uid] = seq

                rows.append(TaggedStep(
                    step=StepRowV2(
                        doi="reactzyme",
                        cascade_id=f"reactzyme_{i}",
                        step_id=f"reactzyme_{i}_0",
                        step_index=0,
                        rxn_smiles=rxn,
                        pairwise_mode="not_applicable",
                        transformation_superclass=None,
                        ec_number=ec,
                        catalyst_class=None,
                        temperature_c=None,
                        ph=None,
                        solvent_smiles=None,
                    ),
                    source="reactzyme",
                ))
    except Exception as exc:
        logger.warning("Failed to parse ReactZyme: %s", exc)
        return []

    # Write sequences sidecar for ESM-2 embedding
    if seq_map:
        seq_path = dest_dir / "sequences.tsv"
        try:
            with open(seq_path, "w", encoding="utf-8", newline="") as fh:
                fh.write("uniprot_id\tsequence\n")
                for uid, seq in sorted(seq_map.items()):
                    fh.write(f"{uid}\t{seq}\n")
            logger.info("ReactZyme: wrote %d sequences to %s", len(seq_map), seq_path)
        except Exception as exc:
            logger.warning("Could not write sequences.tsv: %s", exc)

    logger.info("ReactZyme: loaded %d steps.", len(rows))
    return rows


# ---------------------------------------------------------------------------
# USPTO-50K
# ---------------------------------------------------------------------------

_USPTO50K_URL = (
    "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO50k/raw_test.csv"
)
# We grab all three splits and merge them.
_USPTO50K_SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO50k/raw_train.csv",
    "val": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO50k/raw_val.csv",
    "test": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO50k/raw_test.csv",
}


def load_uspto50k(cache_dir: Path | None = None) -> list[TaggedStep]:
    """Download (if needed) and load USPTO-50K reactions."""
    dest_dir = (cache_dir or _DATA_EXT) / "uspto50k"
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[TaggedStep] = []
    idx = 0
    for split, url in _USPTO50K_SPLIT_URLS.items():
        csv_path = dest_dir / f"raw_{split}.csv"
        if not download_if_missing(url, csv_path, desc=f"USPTO-50K {split}"):
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for rec in reader:
                    # Common column names: "reactants>reagents>production" or "rxn_smiles"
                    rxn = (
                        rec.get("rxn_smiles")
                        or rec.get("reactants>reagents>production")
                        or ""
                    ).strip()
                    # Some USPTO formats use ">" separators; normalise to ">>"
                    if rxn and ">>" not in rxn and ">" in rxn:
                        parts = rxn.split(">")
                        if len(parts) == 3:
                            # reactants > reagents > products
                            reactants = parts[0].strip()
                            products = parts[2].strip()
                            rxn = f"{reactants}>>{products}"
                    if not rxn or ">>" not in rxn:
                        continue
                    rows.append(TaggedStep(
                        step=StepRowV2(
                            doi="uspto50k",
                            cascade_id=f"uspto50k_{idx}",
                            step_id=f"uspto50k_{idx}_0",
                            step_index=0,
                            rxn_smiles=rxn,
                            pairwise_mode="not_applicable",
                            transformation_superclass=None,
                            ec_number=None,
                            catalyst_class=None,
                            temperature_c=None,
                            ph=None,
                            solvent_smiles=None,
                        ),
                        source="uspto50k",
                    ))
                    idx += 1
        except Exception as exc:
            logger.warning("Failed to parse USPTO-50K %s: %s", split, exc)

    logger.info("USPTO-50K: loaded %d steps.", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Merge / deduplicate
# ---------------------------------------------------------------------------

def merge_all(
    internal_steps: Sequence[StepRowV2],
    *external_groups: Sequence[TaggedStep],
) -> list[TaggedStep]:
    """Merge internal AutoPlanner steps with external dataset steps.

    Deduplication is by canonical reaction SMILES.  When duplicates exist the
    internal (richer-annotated) row wins.
    """
    seen: dict[str, TaggedStep] = {}

    # Internal steps first — they take priority
    for s in internal_steps:
        key = _canon_rxn(s.rxn_smiles)
        if key is None:
            key = s.rxn_smiles  # fallback: raw string
        if key not in seen:
            seen[key] = TaggedStep(step=s, source="autoplanner")

    # External groups
    for group in external_groups:
        for ts in group:
            key = _canon_rxn(ts.step.rxn_smiles)
            if key is None:
                key = ts.step.rxn_smiles
            if key not in seen:
                seen[key] = ts

    return list(seen.values())


# ---------------------------------------------------------------------------
# CLI --verify
# ---------------------------------------------------------------------------

def _verify() -> None:
    """Print dataset sizes and overlap statistics."""
    import collections

    print("Loading datasets...\n")

    em = load_enzymemap()
    rz = load_reactzyme()
    us = load_uspto50k()

    print(f"  EnzymeMap : {len(em):>7d} steps")
    print(f"  ReactZyme : {len(rz):>7d} steps")
    print(f"  USPTO-50K : {len(us):>7d} steps")

    # Canonical SMILES sets for overlap
    def _keys(tagged: list[TaggedStep]) -> set[str]:
        out: set[str] = set()
        for ts in tagged:
            k = _canon_rxn(ts.step.rxn_smiles)
            if k:
                out.add(k)
        return out

    em_k, rz_k, us_k = _keys(em), _keys(rz), _keys(us)

    print(f"\n  EnzymeMap canonical unique : {len(em_k)}")
    print(f"  ReactZyme canonical unique : {len(rz_k)}")
    print(f"  USPTO-50K canonical unique : {len(us_k)}")

    print(f"\n  Overlap EnzymeMap & ReactZyme : {len(em_k & rz_k)}")
    print(f"  Overlap EnzymeMap & USPTO-50K : {len(em_k & us_k)}")
    print(f"  Overlap ReactZyme & USPTO-50K : {len(rz_k & us_k)}")
    print(f"  Overlap all three             : {len(em_k & rz_k & us_k)}")

    merged = merge_all([], em, rz, us)
    print(f"\n  Merged total (deduplicated)   : {len(merged)}")

    src_counts = collections.Counter(ts.source for ts in merged)
    print("\n  By source after merge:")
    for src, cnt in src_counts.most_common():
        print(f"    {src:15s} {cnt:>7d}")

    # EC coverage
    with_ec = sum(1 for ts in merged if ts.step.ec_number)
    print(f"\n  Steps with EC number : {with_ec} / {len(merged)}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="AutoPlanner open-dataset loader")
    parser.add_argument("--verify", action="store_true", help="Print dataset sizes and overlap stats")
    args = parser.parse_args()

    if args.verify:
        _verify()
    else:
        parser.print_help()
