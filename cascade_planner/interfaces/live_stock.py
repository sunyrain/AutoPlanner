"""Bounded live adapters that freeze non-authoritative stock search inputs.

PubChem's Chemical Vendors category is useful as a generic benchmark-search
boundary.  It is not a real-time supplier inventory, so this adapter deliberately
produces ``benchmark_stock`` material and never a procurement observation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterable, Mapping

import requests
from rdkit import Chem

from cascade_planner.application.blind_benchmark_contract import canonical_smiles
from cascade_planner.application.retrosynthesis_workers import (
    VERSIONED_BENCHMARK_CATALOG_SCHEMA,
)


PUBCHEM_VENDOR_ADAPTER_VERSION = "autoplanner.pubchem_vendor_catalog.v1"
STANDARD_STOCK_CATALOG_NAME = "ZINC+eMolecules"
STANDARD_STOCK_INDEX_RELATIVE_PATH = Path(
    "data_external/synthatlas/zinc_synthelite_20260223_full_inchikey.sqlite3"
)
STANDARD_STOCK_INDEX_SHA256 = (
    "4d2f601ddd5af10b1c179ec583062d3ba3136553e285944d125e7b5ce19b5a65"
)
STANDARD_STOCK_MEMBER_COUNT = 39_478_827
_PUG_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/"
    "CanonicalSMILES,IsomericSMILES/JSON"
)
_PUG_CATEGORIES = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{cid}/JSON"
)
JsonRequester = Callable[..., tuple[int, bytes, Mapping[str, Any]]]


class LiveStockAdapterError(RuntimeError):
    """The generic live catalog could not be frozen within its hard bounds."""


class FrozenBenchmarkStockIndex:
    """Resolve benchmark membership from a content-addressed SQLite index.

    The index is shared read-only across isolated benchmark cases.  It supplies
    only stock-boundary membership and cannot propose reactions or routes.
    """

    schema_version = "frozen_benchmark_stock_index.v1"
    adapter_version = "autoplanner.frozen_benchmark_stock_index.v1"

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_sha256: str,
        catalog_name: str = "",
    ) -> None:
        path = Path(index_path).expanduser().resolve()
        expected = str(expected_sha256 or "").strip().lower()
        if not path.is_file():
            raise LiveStockAdapterError("benchmark_stock_index_missing")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise LiveStockAdapterError("benchmark_stock_index_sha256_required")
        actual = _file_sha256(path)
        if actual != expected:
            raise LiveStockAdapterError("benchmark_stock_index_sha256_mismatch")
        metadata = self._read_metadata(path)
        if metadata.get("schema_version") != self.schema_version:
            raise LiveStockAdapterError("benchmark_stock_index_schema_invalid")
        source_sha256 = str(metadata.get("source_sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise LiveStockAdapterError("benchmark_stock_source_sha256_missing")
        if (
            metadata.get("complete") != "true"
            or int(metadata.get("member_count") or 0) < 1
        ):
            raise LiveStockAdapterError("benchmark_stock_index_incomplete")
        self.index_path = path
        self.index_sha256 = actual
        self.source_sha256 = source_sha256
        self.catalog_name = (
            str(catalog_name or "").strip()
            or str(metadata.get("catalog_name") or "").strip()
            or "frozen-benchmark-stock"
        )
        self.member_count = int(metadata.get("member_count") or 0)
        self.created_at = str(metadata.get("created_at") or "")
        self.rdkit_version = str(metadata.get("rdkit_version") or "")
        self.identity_key = str(
            metadata.get("identity_key") or "canonical_smiles"
        )
        if self.identity_key not in {"canonical_smiles", "full_inchikey"}:
            raise LiveStockAdapterError("benchmark_stock_identity_key_invalid")

    def __call__(
        self,
        smiles_values: Iterable[str],
        *,
        max_molecules: int = 24,
    ) -> dict[str, Any]:
        if max_molecules < 1:
            raise ValueError("benchmark stock lookup limit must be positive")
        canonical_values = sorted(
            {
                canonical
                for value in smiles_values
                if (canonical := canonical_smiles(value))
            }
        )
        truncated = len(canonical_values) > max_molecules
        selected = canonical_values[:max_molecules]
        full_inchikeys = {
            canonical: self._full_inchikey(canonical)
            for canonical in selected
        }
        found = self._lookup(selected)
        connectivity_diagnostics = self._connectivity_diagnostics(
            [
                full_inchikeys[canonical]
                for canonical in selected
                if canonical not in found and full_inchikeys[canonical]
            ]
        )
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        members = [
            {
                "canonical_smiles": canonical,
                "identity_key": self.identity_key,
                "full_inchikey": full_inchikeys[canonical],
                "membership_verified": True,
                "membership_proof_sha256": hashlib.sha256(
                    f"{self.index_sha256}:{canonical}".encode("utf-8")
                ).hexdigest(),
                "catalog_uri": str(self.index_path),
            }
            for canonical in selected
            if canonical in found
        ]
        misses = [
            {
                "canonical_smiles": canonical,
                "identity_key": self.identity_key,
                "full_inchikey": full_inchikeys[canonical],
                "reason": "molecule_not_in_frozen_benchmark_stock_index",
                **(
                    {
                        "connectivity_diagnostic": connectivity_diagnostics[
                            full_inchikeys[canonical]
                        ]
                    }
                    if full_inchikeys[canonical] in connectivity_diagnostics
                    else {}
                ),
            }
            for canonical in selected
            if canonical not in found
        ]
        body = {
            "schema_version": VERSIONED_BENCHMARK_CATALOG_SCHEMA,
            "adapter_version": self.adapter_version,
            "catalog_name": self.catalog_name,
            "catalog_version": self.index_sha256,
            "retrieved_at": timestamp,
            "source": {
                "name": self.catalog_name,
                "boundary": "benchmark_search",
                "index_path": str(self.index_path),
                "index_sha256": self.index_sha256,
                "source_sha256": self.source_sha256,
                "source_member_count": self.member_count,
                "identity_key": self.identity_key,
                "rdkit_version": self.rdkit_version,
                "immutable_content_addressed": True,
                "commercial_orderability_claimed": False,
            },
            "requested_molecule_count": len(canonical_values),
            "queried_molecule_count": len(selected),
            "truncated": truncated,
            "members": members,
            "misses": misses,
            "semantics": {
                "frozen_benchmark_membership_only": True,
                "read_only_shared_index": True,
                "not_a_reaction_or_route_provider": True,
                "not_procurement_authority": True,
                "connectivity_diagnostic_is_non_authoritative": True,
                "only_exact_identity_match_grants_membership": True,
            },
        }
        body["content_sha256"] = _digest(body)
        return body

    @classmethod
    def _read_metadata(cls, path: Path) -> dict[str, str]:
        try:
            with cls._connect(path) as connection:
                rows = connection.execute(
                    "SELECT key, value FROM metadata ORDER BY key"
                ).fetchall()
                stock_table = connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'stock'"
                ).fetchone()
                stock_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(stock)").fetchall()
                }
        except sqlite3.Error as exc:
            raise LiveStockAdapterError("benchmark_stock_index_unreadable") from exc
        if not stock_table:
            raise LiveStockAdapterError("benchmark_stock_index_table_missing")
        metadata = {str(key): str(value) for key, value in rows}
        identity_key = str(metadata.get("identity_key") or "canonical_smiles")
        if identity_key not in stock_columns:
            raise LiveStockAdapterError("benchmark_stock_identity_column_missing")
        return metadata

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
            timeout=30.0,
        )
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _lookup(self, values: list[str]) -> set[str]:
        if not values:
            return set()
        identities = {
            value: self._identity(value)
            for value in values
        }
        query_values = [value for value in identities.values() if value]
        if not query_values:
            return set()
        placeholders = ",".join("?" for _ in query_values)
        column = (
            "full_inchikey"
            if self.identity_key == "full_inchikey"
            else "canonical_smiles"
        )
        try:
            with self._connect(self.index_path) as connection:
                rows = connection.execute(
                    f"SELECT {column} FROM stock "
                    f"WHERE {column} IN ({placeholders})",
                    query_values,
                ).fetchall()
        except sqlite3.Error as exc:
            raise LiveStockAdapterError("benchmark_stock_index_lookup_failed") from exc
        found = {str(row[0]) for row in rows}
        return {
            smiles
            for smiles, identity in identities.items()
            if identity in found
        }

    def _identity(self, canonical: str) -> str:
        if self.identity_key == "canonical_smiles":
            return canonical
        return self._full_inchikey(canonical)

    @staticmethod
    def _full_inchikey(canonical: str) -> str:
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            return ""
        try:
            return str(Chem.MolToInchiKey(molecule) or "")
        except (RuntimeError, ValueError):
            return ""

    def _connectivity_diagnostics(
        self,
        full_inchikeys: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        """Report same-connectivity catalog presence without granting closure."""

        if self.identity_key != "full_inchikey":
            return {}
        values = sorted({str(value) for value in full_inchikeys if str(value)})
        diagnostics: dict[str, dict[str, Any]] = {}
        try:
            with self._connect(self.index_path) as connection:
                for full_inchikey in values:
                    connectivity_block = full_inchikey.split("-", 1)[0]
                    match = connection.execute(
                        "SELECT 1 FROM stock "
                        "WHERE full_inchikey >= ? AND full_inchikey < ? LIMIT 1",
                        (f"{connectivity_block}-", f"{connectivity_block}."),
                    ).fetchone()
                    diagnostics[full_inchikey] = {
                        "connectivity_block": connectivity_block,
                        "catalog_contains_same_connectivity": match is not None,
                        "grants_membership": False,
                        "reason": "full_inchikey_exact_match_required",
                    }
        except sqlite3.Error as exc:
            raise LiveStockAdapterError(
                "benchmark_stock_connectivity_diagnostic_failed"
            ) from exc
        return diagnostics


@lru_cache(maxsize=1)
def standard_stock_catalog_builder() -> FrozenBenchmarkStockIndex:
    """Return the one frozen benchmark-stock authority used by the main pipeline."""

    index_path = (
        Path(__file__).resolve().parents[2] / STANDARD_STOCK_INDEX_RELATIVE_PATH
    )
    builder = FrozenBenchmarkStockIndex(
        index_path,
        expected_sha256=STANDARD_STOCK_INDEX_SHA256,
        catalog_name=STANDARD_STOCK_CATALOG_NAME,
    )
    if builder.identity_key != "full_inchikey":
        raise LiveStockAdapterError("standard_stock_identity_key_mismatch")
    if builder.member_count != STANDARD_STOCK_MEMBER_COUNT:
        raise LiveStockAdapterError("standard_stock_member_count_mismatch")
    return builder


def load_versioned_inventory_snapshot(
    path: str | Path,
    *,
    max_bytes: int = 8_000_000,
) -> dict[str, Any]:
    """Read a bounded supplier snapshot; the stock worker grants authority."""

    resolved = Path(path).expanduser().resolve()
    try:
        size = resolved.stat().st_size
        if size < 1 or size > max_bytes:
            raise LiveStockAdapterError("inventory_snapshot_size_invalid")
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveStockAdapterError("inventory_snapshot_unreadable") from exc
    if not isinstance(value, Mapping):
        raise LiveStockAdapterError("inventory_snapshot_not_object")
    row = dict(value)
    if row.get("schema_version") != "versioned_inventory_snapshot.v1":
        raise LiveStockAdapterError("inventory_snapshot_schema_invalid")
    if not isinstance(row.get("offers"), list):
        raise LiveStockAdapterError("inventory_snapshot_offers_invalid")
    return row


class FrozenInventorySnapshotBuilder:
    """Callable resolver bound to the exact bytes of one inventory snapshot."""

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        self.snapshot = load_versioned_inventory_snapshot(resolved)
        self.snapshot_sha256 = _file_sha256(resolved)
        material = {
            "schema_version": "stock_oracle_binding.v1",
            "kind": "frozen_inventory_snapshot",
            "oracle_id": f"inventory-snapshot:{self.snapshot_sha256[:24]}",
            "snapshot_sha256": self.snapshot_sha256,
            "snapshot_schema_version": str(
                self.snapshot.get("schema_version") or ""
            ),
            "outputs_require_content_addressing": True,
        }
        material["content_sha256"] = _digest(material)
        self.stock_oracle_binding = material

    def __call__(self, _smiles: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(self.snapshot)


def build_pubchem_vendor_catalog(
    smiles_values: Iterable[str],
    *,
    max_molecules: int = 24,
    max_vendors_per_molecule: int = 5,
    max_workers: int = 8,
    timeout_s: float = 20.0,
    requester: JsonRequester | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Resolve a bounded leaf set and freeze PubChem vendor-category records."""

    if max_molecules < 1 or max_vendors_per_molecule < 1 or max_workers < 1:
        raise ValueError("live stock adapter limits must be positive")
    canonical_values = sorted(
        {
            canonical
            for value in smiles_values
            if (canonical := canonical_smiles(value))
        }
    )
    truncated = len(canonical_values) > max_molecules
    selected = canonical_values[:max_molecules]
    request_json = requester or _requests_json
    def lookup(canonical: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            status, property_bytes, property_json = request_json(
                "POST",
                _PUG_PROPERTY_URL,
                timeout_s=timeout_s,
                data={"smiles": canonical},
            )
            properties = list(
                dict(property_json.get("PropertyTable") or {}).get("Properties") or []
            )
            cid = int(dict(properties[0]).get("CID") or 0) if properties else 0
            if status != 200 or cid <= 0:
                raise LiveStockAdapterError(f"pubchem_cid_lookup_failed:{status}")
            category_url = _PUG_CATEGORIES.format(cid=cid)
            status, category_bytes, category_json = request_json(
                "GET",
                category_url,
                timeout_s=timeout_s,
            )
            if status != 200:
                raise LiveStockAdapterError(f"pubchem_category_lookup_failed:{status}")
            sources = _vendor_sources(category_json)
            if not sources:
                return None, {
                    "canonical_smiles": canonical,
                    "cid": cid,
                    "reason": "pubchem_chemical_vendor_category_empty",
                }
            offers = _bounded_vendor_offers(sources, limit=max_vendors_per_molecule)
            return {
                "canonical_smiles": canonical,
                "cid": cid,
                "vendor_count": len(sources),
                "vendors": offers,
                "source_url": category_url,
                "response_sha256": hashlib.sha256(category_bytes).hexdigest(),
                "identity_response_sha256": hashlib.sha256(property_bytes).hexdigest(),
            }, None
        except (
            LiveStockAdapterError,
            OSError,
            TypeError,
            ValueError,
            requests.RequestException,
        ) as exc:
            return None, {
                "canonical_smiles": canonical,
                "cid": 0,
                "reason": f"{type(exc).__name__}:{exc}",
            }

    members: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    worker_count = min(max_workers, len(selected)) if selected else 0
    if worker_count:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="pubchem-stock",
        ) as executor:
            for member, miss in executor.map(lookup, selected):
                if member is not None:
                    members.append(member)
                if miss is not None:
                    misses.append(miss)
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    body = {
        "schema_version": VERSIONED_BENCHMARK_CATALOG_SCHEMA,
        "adapter_version": PUBCHEM_VENDOR_ADAPTER_VERSION,
        "catalog_name": "pubchem-chemical-vendors",
        "catalog_version": timestamp,
        "retrieved_at": timestamp,
        "source": {
            "name": "PubChem Chemical Vendors",
            "base_url": "https://pubchem.ncbi.nlm.nih.gov",
            "boundary": "benchmark_search",
            "commercial_orderability_claimed": False,
        },
        "requested_molecule_count": len(canonical_values),
        "queried_molecule_count": len(selected),
        "truncated": truncated,
        "members": sorted(members, key=lambda row: row["canonical_smiles"]),
        "misses": sorted(misses, key=lambda row: row["canonical_smiles"]),
        "semantics": {
            "generic_live_lookup": True,
            "frozen_before_stock_audit": True,
            "vendor_category_is_benchmark_membership_only": True,
            "not_real_time_inventory": True,
            "not_procurement_authority": True,
            "bounded_parallel_lookup": True,
        },
    }
    body["content_sha256"] = _digest(body)
    return body


def _requests_json(
    method: str,
    url: str,
    *,
    timeout_s: float,
    data: Mapping[str, Any] | None = None,
) -> tuple[int, bytes, Mapping[str, Any]]:
    response = requests.request(
        method,
        url,
        data=dict(data or {}),
        timeout=max(1.0, float(timeout_s)),
        headers={"User-Agent": "AutoPlanner/1.0 blind-retrosynthesis-benchmark"},
    )
    content = bytes(response.content)
    try:
        value = response.json()
    except requests.JSONDecodeError as exc:
        raise LiveStockAdapterError("pubchem_response_not_json") from exc
    if not isinstance(value, Mapping):
        raise LiveStockAdapterError("pubchem_response_not_object")
    return int(response.status_code), content, value


def _vendor_sources(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    categories = list(dict(value.get("SourceCategories") or {}).get("Categories") or [])
    for raw in categories:
        category = dict(raw) if isinstance(raw, Mapping) else {}
        if category.get("Category") != "Chemical Vendors":
            continue
        return [dict(row) for row in category.get("Sources") or [] if isinstance(row, Mapping)]
    return []


def _bounded_vendor_offers(
    sources: Iterable[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in sources:
        row = dict(raw)
        supplier = " ".join(str(row.get("SourceName") or "").split())
        catalog = str(row.get("RegistryID") or row.get("SID") or "").strip()
        if not supplier or not catalog:
            continue
        unique[(supplier, catalog)] = {
            "supplier": supplier,
            "catalog_number": catalog,
            "sid": int(row.get("SID") or 0),
            "source_url": str(row.get("SourceRecordURL") or row.get("SourceURL") or ""),
        }
    return [unique[key] for key in sorted(unique)[:limit]]


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FrozenBenchmarkStockIndex",
    "FrozenInventorySnapshotBuilder",
    "LiveStockAdapterError",
    "PUBCHEM_VENDOR_ADAPTER_VERSION",
    "STANDARD_STOCK_CATALOG_NAME",
    "STANDARD_STOCK_INDEX_RELATIVE_PATH",
    "STANDARD_STOCK_INDEX_SHA256",
    "STANDARD_STOCK_MEMBER_COUNT",
    "build_pubchem_vendor_catalog",
    "load_versioned_inventory_snapshot",
    "standard_stock_catalog_builder",
]
