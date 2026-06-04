"""ChemEnzy vendor stock lookup helpers.

The native ChemEnzy baseline uses the vendor ``Zinc_Fix-stock`` SMILES file.
This module exposes the same stock as a lightweight SQLite exact-SMILES index
so enhanced route-tree runs can use the same stock boundary without loading a
10M-entry Python set into memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


DEFAULT_VENDOR_ZINC_CSV = Path(
    "vendor/ChemEnzyRetroPlanner/retro_planner/building_block_dataset/"
    "zinc_stock_2021_10_3_canonical_smiles_total_10312151_add_8546.csv"
)
DEFAULT_VENDOR_STOCK_INDEX = Path("results/shared/chemenzy_vendor_stock/zinc_fix_stock_smiles.sqlite")
SCHEMA_VERSION = "chemenzy_vendor_stock_sqlite.v1"


class VendorStockChecker:
    """Exact lookup against a prebuilt ChemEnzy vendor stock SQLite index."""

    def __init__(self, sqlite_path: Path | str = DEFAULT_VENDOR_STOCK_INDEX, *, cache_size: int = 200_000) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.cache_size = max(1, int(cache_size))
        self._conn: sqlite3.Connection | None = None
        self._cache: dict[str, bool] = {}

    @property
    def available(self) -> bool:
        return self.sqlite_path.exists()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __call__(self, smiles: str | None) -> bool:
        if not smiles or not self.available:
            return False
        key = canonical_smiles(str(smiles)) or str(smiles)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = any(self._contains_variant(variant) for variant in stock_query_variants(str(smiles)))
        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[key] = out
        return out

    def _contains_variant(self, smiles: str) -> bool:
        if not smiles:
            return False
        conn = self._connection()
        row = conn.execute("SELECT 1 FROM stock WHERE smiles = ? LIMIT 1", (smiles,)).fetchone()
        return row is not None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            uri = f"file:{self.sqlite_path.resolve()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._conn


def stock_query_variants(smiles: str | None) -> tuple[str, ...]:
    """Return exact/canonical/neutralized variants for vendor stock lookup."""
    raw = str(smiles or "").strip()
    if not raw:
        return ()
    variants = []
    for item in (raw, canonical_smiles(raw), neutralized_smiles(raw)):
        if item and item not in variants:
            variants.append(item)
    return tuple(variants)


def neutralized_smiles(smiles: str | None) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        neutral = rdMolStandardize.Uncharger().uncharge(mol)
        return canonical_smiles(Chem.MolToSmiles(neutral, isomericSmiles=True)) or ""
    except Exception:
        return ""


def wrap_with_vendor_stock(
    stock_checker: Callable[[str], bool] | None,
    *,
    sqlite_path: Path | str = DEFAULT_VENDOR_STOCK_INDEX,
) -> Callable[[str], bool]:
    vendor_checker = VendorStockChecker(sqlite_path)

    def checker(smiles: str) -> bool:
        if stock_checker is not None and bool(stock_checker(smiles)):
            return True
        return bool(vendor_checker(smiles))

    return checker


def build_vendor_stock_index(
    *,
    csv_path: Path | str = DEFAULT_VENDOR_ZINC_CSV,
    sqlite_path: Path | str = DEFAULT_VENDOR_STOCK_INDEX,
    force: bool = False,
    batch_size: int = 100_000,
) -> dict[str, object]:
    csv_path = Path(csv_path)
    sqlite_path = Path(sqlite_path)
    if sqlite_path.exists() and not force:
        return _index_report(sqlite_path, csv_path=csv_path, rebuilt=False)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sqlite_path.with_suffix(sqlite_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    started = time.monotonic()
    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("CREATE TABLE stock (smiles TEXT PRIMARY KEY) WITHOUT ROWID")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        inserted = 0
        batch: list[tuple[str]] = []
        with csv_path.open(encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle):
                text = line.strip()
                if not text:
                    continue
                smiles = text.split(",", 1)[0].strip()
                if line_no == 0 and smiles.lower() == "smiles":
                    continue
                if not smiles:
                    continue
                batch.append((smiles,))
                if len(batch) >= batch_size:
                    conn.executemany("INSERT OR IGNORE INTO stock(smiles) VALUES (?)", batch)
                    inserted += len(batch)
                    batch.clear()
        if batch:
            conn.executemany("INSERT OR IGNORE INTO stock(smiles) VALUES (?)", batch)
            inserted += len(batch)
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema_version", SCHEMA_VERSION))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("csv_path", str(csv_path)))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("input_rows_seen", str(inserted)))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("built_at_unix", str(int(time.time()))))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp_path, sqlite_path)
    report = _index_report(sqlite_path, csv_path=csv_path, rebuilt=True)
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    return report


def _index_report(sqlite_path: Path, *, csv_path: Path, rebuilt: bool) -> dict[str, object]:
    conn = sqlite3.connect(sqlite_path)
    try:
        row_count = int(conn.execute("SELECT COUNT(*) FROM stock").fetchone()[0])
        metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
    finally:
        conn.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "csv_path": str(csv_path),
        "sqlite_path": str(sqlite_path),
        "rebuilt": bool(rebuilt),
        "row_count": row_count,
        "metadata": metadata,
        "size_bytes": sqlite_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/query the ChemEnzy vendor stock SQLite index.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_VENDOR_ZINC_CSV)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_VENDOR_STOCK_INDEX)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()
    report = build_vendor_stock_index(csv_path=args.csv, sqlite_path=args.sqlite, force=args.force)
    if args.query:
        checker = VendorStockChecker(args.sqlite)
        report["queries"] = {smiles: checker(smiles) for smiles in args.query}
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
