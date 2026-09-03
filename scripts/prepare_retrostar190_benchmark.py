#!/usr/bin/env python3
"""Prepare a target-only Retro*-190 manifest and frozen stock index.

Only the 190 target SMILES and eMolecules membership are materialized.  The
Retro* reaction model, value network, templates, and reference routes are not
loaded or exposed to AutoPlanner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterator

from rdkit import Chem, RDLogger, rdBase

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.blind_benchmark_contract import (  # noqa: E402
    BlindCase,
    canonical_smiles,
    load_blind_manifest,
)


RDLogger.DisableLog("rdApp.*")
INDEX_SCHEMA = "frozen_benchmark_stock_index.v1"
DEFAULT_TARGET_MD5 = "aeb06abef693d3e77a881eb4239cbfab"
DEFAULT_INVENTORY_MD5 = "140b886c41c74a9c01ced032edc92fdf"
TARGET_SOURCE = (
    "https://zenodo.org/api/records/14032990/files/retro190.txt/content"
)
INVENTORY_SOURCE = (
    "https://zenodo.org/api/records/14032990/files/origin_dict.csv/content"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--target-md5", default=DEFAULT_TARGET_MD5)
    parser.add_argument(
        "--syntharena-cache",
        help="optional independent 190-page target-set cross-check cache",
    )
    parser.add_argument("--inventory")
    parser.add_argument("--index")
    parser.add_argument("--inventory-md5", default=DEFAULT_INVENTORY_MD5)
    parser.add_argument("--catalog-name", default="Retro*-190 eMolecules ~23M")
    parser.add_argument(
        "--budget-profile",
        choices=("standard", "proof"),
        default="standard",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args(argv)

    targets_path = Path(args.targets).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    target_md5 = _file_digest(targets_path, "md5")
    if target_md5 != str(args.target_md5).strip().lower():
        raise SystemExit("Retro*-190 target file MD5 mismatch")
    targets = _load_targets(targets_path)
    target_crosscheck = (
        _crosscheck_syntharena_targets(
            targets,
            Path(args.syntharena_cache).expanduser().resolve(),
        )
        if args.syntharena_cache
        else {}
    )
    _write_manifest(
        targets,
        path=manifest_path,
        budget_profile=args.budget_profile,
    )

    stock: dict[str, Any] = {
        "catalog_name": args.catalog_name,
        "source_url": INVENTORY_SOURCE,
        "source_md5": "",
        "source_sha256": "",
        "index_path": "",
        "index_sha256": "",
        "member_count": 0,
    }
    if bool(args.inventory) != bool(args.index):
        raise SystemExit("--inventory and --index must be provided together")
    if args.inventory and args.index:
        inventory_path = Path(args.inventory).expanduser().resolve()
        source_md5 = _file_digest(inventory_path, "md5")
        if source_md5 != str(args.inventory_md5).strip().lower():
            raise SystemExit("Retro* inventory MD5 mismatch")
        index_path = Path(args.index).expanduser().resolve()
        index_result = build_stock_index(
            inventory_path,
            index_path,
            catalog_name=args.catalog_name,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        stock.update(
            {
                "source_md5": source_md5,
                "source_sha256": index_result["source_sha256"],
                "index_path": _portable_path(index_path),
                "index_sha256": index_result["index_sha256"],
                "member_count": index_result["member_count"],
            }
        )

    _write_protocol(
        path=protocol_path,
        manifest_path=manifest_path,
        target_path=targets_path,
        target_md5=target_md5,
        target_count=len(targets),
        budget_profile=args.budget_profile,
        stock=stock,
        target_crosscheck=target_crosscheck,
    )
    result = {
        "target_count": len(targets),
        "manifest": str(manifest_path),
        "manifest_sha256": _file_digest(manifest_path, "sha256"),
        "protocol": str(protocol_path),
        "protocol_sha256": _file_digest(protocol_path, "sha256"),
        "stock": stock,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_stock_index(
    source_path: Path,
    output_path: Path,
    *,
    catalog_name: str,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    if workers < 1 or batch_size < 1:
        raise ValueError("stock index workers and batch size must be positive")
    source_sha256 = _file_digest(source_path, "sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        metadata = _index_metadata(output_path)
        if (
            metadata.get("schema_version") == INDEX_SCHEMA
            and metadata.get("source_sha256") == source_sha256
            and int(metadata.get("member_count") or 0) > 0
        ):
            return {
                "source_sha256": source_sha256,
                "index_sha256": _file_digest(output_path, "sha256"),
                "member_count": int(metadata["member_count"]),
                "reused": True,
            }
        raise RuntimeError("existing benchmark stock index does not match source")

    building_path = output_path.with_name(output_path.name + ".building")
    connection = sqlite3.connect(building_path)
    try:
        _initialize_index(
            connection,
            source_sha256=source_sha256,
            catalog_name=catalog_name,
        )
        metadata = _metadata(connection)
        if metadata.get("source_sha256") != source_sha256:
            raise RuntimeError("partial benchmark stock index source mismatch")
        processed_rows = int(metadata.get("processed_rows") or 0)
        inserted_rows = int(metadata.get("inserted_rows") or 0)
        chunks = _inventory_chunks(
            source_path,
            batch_size=batch_size,
            skip_rows=processed_rows,
        )
        if workers == 1:
            canonical_batches = map(_canonicalize_inventory_rows, chunks)
            pool = None
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=workers)
            canonical_batches = pool.imap(
                _canonicalize_inventory_rows,
                chunks,
                chunksize=1,
            )
        try:
            for input_count, values in canonical_batches:
                connection.executemany(
                    "INSERT INTO stock(canonical_smiles) VALUES (?) "
                    "ON CONFLICT(canonical_smiles) DO NOTHING",
                    ((value,) for value in values),
                )
                processed_rows += input_count
                inserted_rows += len(values)
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [
                        ("processed_rows", str(processed_rows)),
                        ("inserted_rows", str(inserted_rows)),
                    ],
                )
                connection.commit()
                if processed_rows % (batch_size * 100) < batch_size:
                    print(
                        json.dumps(
                            {
                                "stage": "stock_index",
                                "processed_rows": processed_rows,
                                "inserted_rows_before_dedup": inserted_rows,
                            }
                        ),
                        flush=True,
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        member_count = int(
            connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [
                ("member_count", str(member_count)),
                ("complete", "true"),
            ],
        )
        connection.commit()
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("VACUUM")
        connection.commit()
    finally:
        connection.close()
    building_path.replace(output_path)
    return {
        "source_sha256": source_sha256,
        "index_sha256": _file_digest(output_path, "sha256"),
        "member_count": member_count,
        "reused": False,
    }


def _initialize_index(
    connection: sqlite3.Connection,
    *,
    source_sha256: str,
    catalog_name: str,
) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -262144")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS stock "
        "(canonical_smiles TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO NOTHING",
        [
            ("schema_version", INDEX_SCHEMA),
            ("catalog_name", catalog_name),
            ("source_sha256", source_sha256),
            ("created_at", _utc_now()),
            ("rdkit_version", rdBase.rdkitVersion),
            ("processed_rows", "0"),
            ("inserted_rows", "0"),
            ("member_count", "0"),
            ("complete", "false"),
        ],
    )
    connection.commit()


def _inventory_chunks(
    source_path: Path,
    *,
    batch_size: int,
    skip_rows: int,
) -> Iterator[list[str]]:
    with source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        skipped = 0
        while skipped < skip_rows:
            if next(reader, None) is None:
                return
            skipped += 1
        chunk: list[str] = []
        for row in reader:
            chunk.append(str(row[1]).strip() if len(row) > 1 else "")
            if len(chunk) >= batch_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _canonicalize_inventory_rows(rows: list[str]) -> tuple[int, list[str]]:
    canonical: list[str] = []
    for value in rows:
        molecule = Chem.MolFromSmiles(value)
        if molecule is not None:
            canonical.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
    return len(rows), canonical


def _load_targets(path: Path) -> list[str]:
    targets = [
        canonical
        for line in path.read_text(encoding="utf-8").splitlines()
        if (canonical := canonical_smiles(line.strip()))
    ]
    if len(targets) != 190 or len(set(targets)) != 190:
        raise ValueError("Retro*-190 target file must contain 190 unique valid SMILES")
    return targets


def _write_manifest(
    targets: list[str],
    *,
    path: Path,
    budget_profile: str,
) -> None:
    budgets = {
        "standard": {
            "max_model_invocations": 3,
            "max_total_input_tokens": 120_000,
            "max_total_output_tokens": 30_000,
            "max_total_wall_time_s": 1_800,
            "max_accepted_expansions": 96,
            "max_attempt_runs": 192,
            "max_prompt_context_bytes": 160_000,
        },
        "proof": {
            "max_model_invocations": 10,
            "max_total_input_tokens": 1_200_000,
            "max_total_output_tokens": 200_000,
            "max_total_wall_time_s": 1_800,
            "max_accepted_expansions": 192,
            "max_attempt_runs": 384,
            "max_prompt_context_bytes": 256_000,
        },
    }
    cases = []
    for index, target in enumerate(targets, start=1):
        short_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()[:10]
        cases.append(
            BlindCase(
                case_id=f"retrostar190-{index:03d}-{short_hash}",
                target_name=f"opaque benchmark target {index:03d}",
                target_smiles=target,
                acceptance={
                    "minimum_complete_routes": 1,
                    "minimum_edge_proof_level": 2,
                    "minimum_independent_source_groups": 1,
                    "minimum_planning_route_steps": 0,
                    "stock_boundary": "benchmark_search",
                },
                budget=budgets[budget_profile],
            ).to_dict()
        )
    payload = {
        "schema_version": "blind_retrosynthesis_manifest.v1",
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    loaded = load_blind_manifest(path)
    if len(loaded) != 190:
        raise RuntimeError("generated Retro*-190 manifest failed validation")


def _write_protocol(
    *,
    path: Path,
    manifest_path: Path,
    target_path: Path,
    target_md5: str,
    target_count: int,
    budget_profile: str,
    stock: dict[str, Any],
    target_crosscheck: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "retrostar190_autoplanner_protocol.v1",
        "benchmark": "Retro*-190 / USPTO-190",
        "target_count": target_count,
        "execution_profile": budget_profile,
        "manifest_sha256": _file_digest(manifest_path, "sha256"),
        "inputs": {
            "target_file": _portable_path(target_path),
            "target_md5": target_md5,
            "target_sha256": _file_digest(target_path, "sha256"),
            "target_source": TARGET_SOURCE,
            "independent_target_crosscheck": target_crosscheck,
            "stock": stock,
        },
        "planner": {
            "system_under_test": "AutoPlanner V4 current mainline",
            "retrostar_reaction_model_loaded": False,
            "retrostar_value_network_loaded": False,
            "retrostar_templates_loaded": False,
            "aizynthfinder_or_retrochimera_sidecar_loaded": False,
            "reference_routes_visible_to_planner": False,
            "target_names_are_opaque": True,
        },
        "metrics": {
            "retrostar_comparable_primary": (
                "at_least_one_host_structural_route_with_all_terminal_leaves_"
                "in_frozen_benchmark_stock"
            ),
            "reported_separately": [
                "structural_route_present",
                "host_reaction_validated",
                "official_stock_closed",
                "configured_proof_policy_accepted",
                "exact_source_grade",
                "condition_completeness",
                "model_calls_and_tokens",
                "wall_time",
            ],
            "proof_or_conditions_do_not_redefine_retrostar_solved": True,
        },
        "provenance": {
            "paper": "https://proceedings.mlr.press/v119/chen20k.html",
            "official_repository": "https://github.com/binghong-ml/retro_star",
            "redistributed_data": "https://doi.org/10.5281/zenodo.14032990",
            "redistributed_data_license": "CC-BY-4.0",
            "independent_wrapper": (
                "https://github.com/AustinT/"
                "syntheseus-retro-star-benchmark"
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in connection.execute(
            "SELECT key, value FROM metadata ORDER BY key"
        )
    }


def _crosscheck_syntharena_targets(
    expected_targets: list[str],
    cache_dir: Path,
) -> dict[str, Any]:
    from cascade_planner.eval.syntharena_uspto190 import (
        SYNTHARENA_USPTO_190,
        parse_target_page,
    )

    observed: list[str] = []
    pages = [
        path
        for path in cache_dir.glob("uspto190_*.html")
        if not path.name.startswith("uspto190_page_")
        and path.name != "uspto190_index.html"
    ]
    for path in pages:
        row = parse_target_page(path)
        if row and (canonical := canonical_smiles(row.get("target_smiles"))):
            observed.append(canonical)
    equal = (
        len(observed) == 190
        and len(set(observed)) == 190
        and set(observed) == set(expected_targets)
    )
    if not equal:
        raise RuntimeError("SynthArena target set does not match Retro*-190")
    return {
        "source": SYNTHARENA_USPTO_190,
        "cached_target_pages": len(pages),
        "parsed_unique_targets": len(set(observed)),
        "canonical_target_sets_equal": True,
        "canonical_set_sha256": _digest(sorted(set(observed))),
    }


def _index_metadata(path: Path) -> dict[str, str]:
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
            return _metadata(connection)
    except sqlite3.Error:
        return {}


def _file_digest(path: Path, algorithm: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
