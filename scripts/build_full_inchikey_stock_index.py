#!/usr/bin/env python3
"""Build a content-addressed full-InChIKey stock index from licensed inputs.

The script never downloads or redistributes provider data.  It accepts one or
more legally obtained newline/CSV files, unions exact full InChIKeys, and
refuses the SynthEx label unless the expected 39,684,411-member count is met.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Iterator


INDEX_SCHEMA = "frozen_benchmark_stock_index.v1"
FULL_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
SYNTHEX_MEMBER_COUNT = 39_684_411


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--column", default="", help="CSV column name; blank scans fields")
    parser.add_argument("--catalog-name", default="ZINC+eMolecules")
    parser.add_argument("--expected-count", type=int, default=SYNTHEX_MEMBER_COUNT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args(argv)
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
        "paper_stock_comparable": member_count == SYNTHEX_MEMBER_COUNT,
    }


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
