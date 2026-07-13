"""Bounded live adapters that freeze non-authoritative stock search inputs.

PubChem's Chemical Vendors category is useful as a generic benchmark-search
boundary.  It is not a real-time supplier inventory, so this adapter deliberately
produces ``benchmark_stock`` material and never a procurement observation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

import requests

from cascade_planner.application.blind_benchmark_contract import canonical_smiles
from cascade_planner.application.retrosynthesis_workers import (
    VERSIONED_BENCHMARK_CATALOG_SCHEMA,
)


PUBCHEM_VENDOR_ADAPTER_VERSION = "autoplanner.pubchem_vendor_catalog.v1"
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


def build_pubchem_vendor_catalog(
    smiles_values: Iterable[str],
    *,
    max_molecules: int = 24,
    max_vendors_per_molecule: int = 5,
    timeout_s: float = 20.0,
    requester: JsonRequester | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Resolve a bounded leaf set and freeze PubChem vendor-category records."""

    if max_molecules < 1 or max_vendors_per_molecule < 1:
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
    members: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for canonical in selected:
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
                misses.append(
                    {
                        "canonical_smiles": canonical,
                        "cid": cid,
                        "reason": "pubchem_chemical_vendor_category_empty",
                    }
                )
                continue
            offers = _bounded_vendor_offers(sources, limit=max_vendors_per_molecule)
            members.append(
                {
                    "canonical_smiles": canonical,
                    "cid": cid,
                    "vendor_count": len(sources),
                    "vendors": offers,
                    "source_url": category_url,
                    "response_sha256": hashlib.sha256(category_bytes).hexdigest(),
                    "identity_response_sha256": hashlib.sha256(property_bytes).hexdigest(),
                }
            )
        except (LiveStockAdapterError, OSError, TypeError, ValueError) as exc:
            misses.append(
                {
                    "canonical_smiles": canonical,
                    "cid": 0,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
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


__all__ = [
    "LiveStockAdapterError",
    "PUBCHEM_VENDOR_ADAPTER_VERSION",
    "build_pubchem_vendor_catalog",
]
