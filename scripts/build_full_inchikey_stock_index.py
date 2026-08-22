#!/usr/bin/env python3
"""Build a content-addressed full-InChIKey stock index from licensed inputs.

The script never downloads or redistributes provider data.  It accepts one or
more legally obtained newline/CSV files and unions exact full InChIKeys.

SynthEx reports 39,684,411 *entries* for its combined ZINC + eMolecules input.
The released inputs contain 205,584 redundant or invalid eMolecules rows, so
the exact membership oracle contains 39,478,827 unique full InChIKeys.  A
primary-key membership index must bind the latter; the former remains frozen
as a paper-declared source statistic rather than an impossible set cardinality.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Iterator


INDEX_SCHEMA = "frozen_benchmark_stock_index.v1"
FULL_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
SYNTHEX_DECLARED_ENTRY_COUNT = 39_684_411
SYNTHEX_UNIQUE_MEMBER_COUNT = 39_478_827


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append")
    parser.add_argument(
        "--inchikey-hdf",
        help="Pandas HDF file containing an existing full-InChIKey column",
    )
    parser.add_argument(
        "--smiles-sqlite",
        help="SQLite stock with canonical SMILES to convert and union",
    )
    parser.add_argument(
        "--inchikey-csv",
        help="CSV stock with a full-InChIKey column to union without redrawing SMILES",
    )
    parser.add_argument("--hdf-key", default="table")
    parser.add_argument("--hdf-column", default="inchi_key")
    parser.add_argument("--sqlite-table", default="stock")
    parser.add_argument("--sqlite-column", default="canonical_smiles")
    parser.add_argument("--csv-inchikey-column", default="inchi_key")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--column", default="", help="CSV column name; blank scans fields")
    parser.add_argument("--catalog-name", default="ZINC+eMolecules")
    parser.add_argument(
        "--expected-count", type=int, default=SYNTHEX_UNIQUE_MEMBER_COUNT
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args(argv)
    if args.inchikey_hdf or args.smiles_sqlite or args.inchikey_csv:
        second_sources = int(bool(args.smiles_sqlite)) + int(bool(args.inchikey_csv))
        if not args.inchikey_hdf or second_sources != 1 or args.input:
            parser.error(
                "composite mode requires --inchikey-hdf and exactly one of "
                "--smiles-sqlite or --inchikey-csv, and cannot be combined "
                "with --input"
            )
        common = {
            "inchikey_hdf": Path(args.inchikey_hdf).expanduser().resolve(),
            "output": Path(args.output).expanduser().resolve(),
            "hdf_key": str(args.hdf_key or "table"),
            "hdf_column": str(args.hdf_column or "inchi_key"),
            "catalog_name": str(args.catalog_name or ""),
            "expected_count": int(args.expected_count),
            "batch_size": int(args.batch_size),
            "resume": bool(args.resume),
        }
        if args.inchikey_csv:
            result = build_inchikey_composite_index(
                **common,
                inchikey_csv=Path(args.inchikey_csv).expanduser().resolve(),
                csv_column=str(args.csv_inchikey_column or "inchi_key"),
            )
        else:
            result = build_composite_index(
                **common,
                smiles_sqlite=Path(args.smiles_sqlite).expanduser().resolve(),
                sqlite_table=str(args.sqlite_table or "stock"),
                sqlite_column=str(args.sqlite_column or "canonical_smiles"),
                workers=int(args.workers),
            )
    else:
        if not args.input:
            parser.error("--input is required outside composite mode")
        result = build_index(
            [Path(value).expanduser().resolve() for value in args.input],
            Path(args.output).expanduser().resolve(),
            column=str(args.column or ""),
            catalog_name=str(args.catalog_name or ""),
            expected_count=int(args.expected_count),
            batch_size=int(args.batch_size),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_index(
    inputs: list[Path],
    output: Path,
    *,
    column: str,
    catalog_name: str,
    expected_count: int,
    batch_size: int,
) -> dict:
    if not inputs or any(not path.is_file() for path in inputs):
        raise ValueError("all stock input files must exist")
    if expected_count < 1 or batch_size < 1:
        raise ValueError("expected count and batch size must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {output}")
    source_files = [
        {"path": str(path), "sha256": _file_sha256(path)} for path in inputs
    ]
    source_sha256 = _digest(source_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"partial index requires explicit cleanup: {building}")
    connection = sqlite3.connect(building)
    try:
        connection.execute(
            "CREATE TABLE stock (full_inchikey TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        processed = 0
        valid = 0
        for batch in _batches(_iter_keys(inputs, column=column), batch_size):
            processed += len(batch)
            values = sorted({value for value in batch if FULL_INCHIKEY.fullmatch(value)})
            valid += len(values)
            connection.executemany(
                "INSERT INTO stock(full_inchikey) VALUES (?) "
                "ON CONFLICT(full_inchikey) DO NOTHING",
                ((value,) for value in values),
            )
            connection.commit()
        member_count = int(connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0])
        if member_count != expected_count:
            raise RuntimeError(
                f"stock_member_count_mismatch:expected={expected_count}:actual={member_count}"
            )
        metadata = {
            "schema_version": INDEX_SCHEMA,
            "catalog_name": catalog_name,
            "identity_key": "full_inchikey",
            "source_sha256": source_sha256,
            "source_file_count": str(len(inputs)),
            "processed_rows": str(processed),
            "valid_rows_before_dedup": str(valid),
            "member_count": str(member_count),
            "complete": "true",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.commit()
    except BaseException:
        connection.close()
        if building.exists():
            building.unlink()
        raise
    else:
        connection.close()
    building.replace(output)
    return {
        "index_path": str(output),
        "index_sha256": _file_sha256(output),
        "source_sha256": source_sha256,
        "member_count": member_count,
        "identity_key": "full_inchikey",
        "catalog_name": catalog_name,
        "paper_stock_comparable": member_count == SYNTHEX_UNIQUE_MEMBER_COUNT,
        "paper_declared_entry_count": SYNTHEX_DECLARED_ENTRY_COUNT,
    }


def build_composite_index(
    *,
    inchikey_hdf: Path,
    smiles_sqlite: Path,
    output: Path,
    hdf_key: str,
    hdf_column: str,
    sqlite_table: str,
    sqlite_column: str,
    catalog_name: str,
    expected_count: int,
    batch_size: int,
    workers: int,
    resume: bool,
) -> dict:
    """Union an InChIKey HDF stock with a SMILES SQLite stock.

    The build is checkpointed in ``<output>.building``.  It is safe to resume
    because the destination is a primary-key set and the SMILES reader stores
    its last ordered key after every committed batch.
    """

    if not inchikey_hdf.is_file() or not smiles_sqlite.is_file():
        raise ValueError("composite stock sources must exist")
    if expected_count < 1 or batch_size < 1 or workers < 1:
        raise ValueError("expected count, batch size and workers must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_name(output.name + ".building")
    if building.exists() and not resume:
        raise FileExistsError(f"partial index requires --resume: {building}")
    source_files = [
        {"path": str(inchikey_hdf), "sha256": _file_sha256(inchikey_hdf)},
        {"path": str(smiles_sqlite), "sha256": _file_sha256(smiles_sqlite)},
    ]
    source_sha256 = _digest(source_files)
    connection = sqlite3.connect(building)
    try:
        _configure_build_connection(connection)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS stock "
            "(full_inchikey TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        metadata = _metadata(connection)
        if metadata and metadata.get("source_sha256") != source_sha256:
            raise RuntimeError("partial_index_source_binding_mismatch")
        _set_metadata(
            connection,
            {
                "schema_version": INDEX_SCHEMA,
                "catalog_name": catalog_name,
                "identity_key": "full_inchikey",
                "source_sha256": source_sha256,
                "source_file_count": "2",
                "expected_member_count": str(expected_count),
                "complete": "false",
            },
        )
        connection.commit()

        zinc_offset = int(metadata.get("zinc_offset") or 0)
        if metadata.get("phase") not in {"emolecules", "complete"}:
            zinc_total = _insert_hdf_inchikeys(
                connection,
                inchikey_hdf,
                key=hdf_key,
                column=hdf_column,
                offset=zinc_offset,
                batch_size=batch_size,
            )
            _set_metadata(
                connection,
                {
                    "phase": "emolecules",
                    "zinc_offset": str(zinc_total),
                    "zinc_source_rows": str(zinc_total),
                },
            )
            connection.commit()

        metadata = _metadata(connection)
        last_smiles = str(metadata.get("emolecules_last_smiles") or "")
        processed = int(metadata.get("emolecules_processed_rows") or 0)
        valid = int(metadata.get("emolecules_valid_inchikey_rows") or 0)
        source = sqlite3.connect(f"file:{smiles_sqlite}?mode=ro", uri=True)
        try:
            _validate_sql_identifier(sqlite_table)
            _validate_sql_identifier(sqlite_column)
            query = (
                f"SELECT [{sqlite_column}] FROM [{sqlite_table}] "
                f"WHERE [{sqlite_column}] > ? ORDER BY [{sqlite_column}] LIMIT ?"
            )
            with ProcessPoolExecutor(max_workers=workers) as executor:
                while True:
                    smiles = [
                        str(row[0])
                        for row in source.execute(query, (last_smiles, batch_size))
                    ]
                    if not smiles:
                        break
                    chunks = _balanced_chunks(smiles, workers)
                    keys: list[str] = []
                    for values in executor.map(_smiles_batch_to_inchikeys, chunks):
                        keys.extend(values)
                    connection.executemany(
                        "INSERT INTO stock(full_inchikey) VALUES (?) "
                        "ON CONFLICT(full_inchikey) DO NOTHING",
                        ((value,) for value in sorted(set(keys))),
                    )
                    processed += len(smiles)
                    valid += len(keys)
                    last_smiles = smiles[-1]
                    _set_metadata(
                        connection,
                        {
                            "phase": "emolecules",
                            "emolecules_last_smiles": last_smiles,
                            "emolecules_processed_rows": str(processed),
                            "emolecules_valid_inchikey_rows": str(valid),
                            "current_member_count": str(
                                connection.execute(
                                    "SELECT COUNT(*) FROM stock"
                                ).fetchone()[0]
                            ),
                        },
                    )
                    connection.commit()
        finally:
            source.close()

        member_count = int(
            connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
        )
        _set_metadata(
            connection,
            {
                "phase": "complete" if member_count == expected_count else "mismatch",
                "member_count": str(member_count),
                "complete": "true" if member_count == expected_count else "false",
            },
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if member_count != expected_count:
            raise RuntimeError(
                f"stock_member_count_mismatch:expected={expected_count}:actual={member_count}"
            )
    finally:
        connection.close()
    building.replace(output)
    return {
        "index_path": str(output),
        "index_sha256": _file_sha256(output),
        "source_sha256": source_sha256,
        "member_count": member_count,
        "identity_key": "full_inchikey",
        "catalog_name": catalog_name,
        "paper_stock_comparable": member_count == SYNTHEX_UNIQUE_MEMBER_COUNT,
        "paper_declared_entry_count": SYNTHEX_DECLARED_ENTRY_COUNT,
    }


def build_inchikey_composite_index(
    *,
    inchikey_hdf: Path,
    inchikey_csv: Path,
    output: Path,
    hdf_key: str,
    hdf_column: str,
    csv_column: str,
    catalog_name: str,
    expected_count: int,
    batch_size: int,
    resume: bool,
) -> dict:
    """Union exact full-InChIKeys from an HDF stock and a CSV stock.

    Unlike :func:`build_composite_index`, this path does not regenerate
    InChIKeys from SMILES.  It is the authoritative path when a provider has
    already published full stereochemistry-bearing InChIKeys.  The CSV row
    offset is checkpointed after every committed batch so a large licensed
    source can be resumed without changing its identity contract.
    """

    if not inchikey_hdf.is_file() or not inchikey_csv.is_file():
        raise ValueError("composite stock sources must exist")
    if expected_count < 1 or batch_size < 1:
        raise ValueError("expected count and batch size must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_name(output.name + ".building")
    if building.exists() and not resume:
        raise FileExistsError(f"partial index requires --resume: {building}")
    source_files = [
        {"path": str(inchikey_hdf), "sha256": _file_sha256(inchikey_hdf)},
        {"path": str(inchikey_csv), "sha256": _file_sha256(inchikey_csv)},
    ]
    source_sha256 = _digest(source_files)
    connection = sqlite3.connect(building)
    try:
        _configure_build_connection(connection)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS stock "
            "(full_inchikey TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        metadata = _metadata(connection)
        if metadata and metadata.get("source_sha256") != source_sha256:
            raise RuntimeError("partial_index_source_binding_mismatch")
        _set_metadata(
            connection,
            {
                "schema_version": INDEX_SCHEMA,
                "catalog_name": catalog_name,
                "identity_key": "full_inchikey",
                "source_sha256": source_sha256,
                "source_file_count": "2",
                "expected_member_count": str(expected_count),
                "inchikey_csv_column": csv_column,
                "complete": "false",
            },
        )
        connection.commit()

        zinc_offset = int(metadata.get("zinc_offset") or 0)
        if metadata.get("phase") not in {"inchikey_csv", "complete"}:
            zinc_total = _insert_hdf_inchikeys(
                connection,
                inchikey_hdf,
                key=hdf_key,
                column=hdf_column,
                offset=zinc_offset,
                batch_size=batch_size,
            )
            _set_metadata(
                connection,
                {
                    "phase": "inchikey_csv",
                    "zinc_offset": str(zinc_total),
                    "zinc_source_rows": str(zinc_total),
                },
            )
            connection.commit()

        metadata = _metadata(connection)
        processed = int(metadata.get("inchikey_csv_processed_rows") or 0)
        valid = int(metadata.get("inchikey_csv_valid_rows") or 0)
        with inchikey_csv.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if csv_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"stock column missing:{inchikey_csv}:{csv_column}"
                )
            for _ in itertools.islice(reader, processed):
                pass
            while True:
                rows = list(itertools.islice(reader, batch_size))
                if not rows:
                    break
                keys = [
                    key
                    for row in rows
                    if FULL_INCHIKEY.fullmatch(
                        key := str(row.get(csv_column) or "").strip().upper()
                    )
                ]
                connection.executemany(
                    "INSERT INTO stock(full_inchikey) VALUES (?) "
                    "ON CONFLICT(full_inchikey) DO NOTHING",
                    ((value,) for value in sorted(set(keys))),
                )
                processed += len(rows)
                valid += len(keys)
                _set_metadata(
                    connection,
                    {
                        "phase": "inchikey_csv",
                        "inchikey_csv_processed_rows": str(processed),
                        "inchikey_csv_valid_rows": str(valid),
                        "current_member_count": str(
                            connection.execute(
                                "SELECT COUNT(*) FROM stock"
                            ).fetchone()[0]
                        ),
                    },
                )
                connection.commit()

        member_count = int(
            connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
        )
        _set_metadata(
            connection,
            {
                "phase": "complete" if member_count == expected_count else "mismatch",
                "member_count": str(member_count),
                "complete": "true" if member_count == expected_count else "false",
            },
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if member_count != expected_count:
            raise RuntimeError(
                f"stock_member_count_mismatch:expected={expected_count}:actual={member_count}"
            )
    finally:
        connection.close()
    building.replace(output)
    return {
        "index_path": str(output),
        "index_sha256": _file_sha256(output),
        "source_sha256": source_sha256,
        "member_count": member_count,
        "identity_key": "full_inchikey",
        "catalog_name": catalog_name,
        "paper_stock_comparable": member_count == SYNTHEX_UNIQUE_MEMBER_COUNT,
        "paper_declared_entry_count": SYNTHEX_DECLARED_ENTRY_COUNT,
    }


def _insert_hdf_inchikeys(
    connection: sqlite3.Connection,
    path: Path,
    *,
    key: str,
    column: str,
    offset: int,
    batch_size: int,
) -> int:
    import pandas as pd

    frame = pd.read_hdf(path, key=key)
    if column not in frame.columns:
        raise ValueError(f"HDF InChIKey column missing:{column}")
    values = frame[column]
    total = len(values)
    for start in range(max(0, offset), total, batch_size):
        batch = [
            str(value).strip().upper()
            for value in values.iloc[start : start + batch_size]
            if FULL_INCHIKEY.fullmatch(str(value).strip().upper())
        ]
        connection.executemany(
            "INSERT INTO stock(full_inchikey) VALUES (?) "
            "ON CONFLICT(full_inchikey) DO NOTHING",
            ((value,) for value in sorted(set(batch))),
        )
        _set_metadata(
            connection,
            {
                "phase": "zinc",
                "zinc_offset": str(min(total, start + batch_size)),
            },
        )
        connection.commit()
    return total


def _smiles_batch_to_inchikeys(smiles: list[str]) -> list[str]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    keys: list[str] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        key = str(Chem.MolToInchiKey(molecule) or "").strip().upper()
        if FULL_INCHIKEY.fullmatch(key):
            keys.append(key)
    return keys


def _balanced_chunks(values: list[str], workers: int) -> list[list[str]]:
    size = max(1, (len(values) + workers - 1) // workers)
    return [values[index : index + size] for index in range(0, len(values), size)]


def _configure_build_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(connection.execute("SELECT key,value FROM metadata").fetchall())
    except sqlite3.OperationalError:
        return {}


def _set_metadata(connection: sqlite3.Connection, values: dict[str, str]) -> None:
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        sorted((str(key), str(value)) for key, value in values.items()),
    )


def _validate_sql_identifier(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"unsafe SQLite identifier:{value}")


def _iter_keys(inputs: Iterable[Path], *, column: str) -> Iterator[str]:
    for path in inputs:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            if column:
                reader = csv.DictReader(handle)
                if column not in (reader.fieldnames or []):
                    raise ValueError(f"stock column missing:{path}:{column}")
                for row in reader:
                    yield str(row.get(column) or "").strip().upper()
                continue
            for raw in handle:
                fields = re.split(r"[,;\t ]+", raw.strip())
                yield next(
                    (value.upper() for value in fields if FULL_INCHIKEY.fullmatch(value.upper())),
                    "",
                )


def _batches(values: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
