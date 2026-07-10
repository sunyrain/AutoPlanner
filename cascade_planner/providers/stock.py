"""Immutable stock boundaries and timestamped supplier-offer snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping

from rdkit import Chem, RDLogger

from cascade_planner.providers.contracts import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
)


RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StockOffer:
    supplier: str
    catalog_number: str
    canonical_smiles: str
    checked_at: str
    snapshot_sha256: str
    available: bool
    purity: str = ""
    pack_size: str = ""
    price: float | None = None
    currency: str = ""
    region: str = ""
    lead_time_days: int | None = None
    source_url: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    snapshot: Mapping[str, Any] = field(default_factory=dict, repr=False)
    snapshot_verified: bool = field(default=False, init=False)
    schema_version: ClassVar[str] = "stock_offer.v1"

    def __post_init__(self) -> None:
        canonical = _canonical_smiles(self.canonical_smiles)
        if not canonical or canonical != self.canonical_smiles:
            raise ValueError("stock offer requires canonical isomeric SMILES")
        if not self.supplier or not self.catalog_number or not self.checked_at:
            raise ValueError("stock offer identity and checked_at are required")
        if not _is_sha256(self.snapshot_sha256):
            raise ValueError("stock offer snapshot_sha256 must be 64 lowercase hex characters")
        _parse_timestamp(self.checked_at)
        if type(self.available) is not bool:
            raise ValueError("stock offer availability must be a boolean snapshot field")
        if self.price is not None and self.price < 0:
            raise ValueError("stock offer price must be nonnegative")
        if self.lead_time_days is not None and self.lead_time_days < 0:
            raise ValueError("stock offer lead_time_days must be nonnegative")
        canonical_snapshot = canonicalize_stock_snapshot(self.snapshot)
        if canonical_snapshot != _snapshot_from_offer(self):
            raise ValueError("stock offer fields do not match canonical snapshot content")
        if stock_snapshot_sha256(canonical_snapshot) != self.snapshot_sha256:
            raise ValueError("stock offer snapshot digest does not match canonical snapshot content")

    @classmethod
    def from_trusted_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        snapshot_sha256: str,
    ) -> "StockOffer":
        canonical = canonicalize_stock_snapshot(snapshot)
        if stock_snapshot_sha256(canonical) != str(snapshot_sha256 or "").lower():
            raise ValueError("trusted stock snapshot digest mismatch")
        offer = cls(
            supplier=canonical["supplier"],
            catalog_number=canonical["catalog_number"],
            canonical_smiles=canonical["canonical_smiles"],
            checked_at=canonical["checked_at"],
            snapshot_sha256=str(snapshot_sha256).lower(),
            available=canonical["available"],
            purity=canonical["purity"],
            pack_size=canonical["pack_size"],
            price=canonical["price"],
            currency=canonical["currency"],
            region=canonical["region"],
            lead_time_days=canonical["lead_time_days"],
            source_url=canonical["source_url"],
            metadata=canonical["metadata"],
            snapshot=canonical,
        )
        object.__setattr__(offer, "snapshot_verified", True)
        return offer

    @property
    def offer_id(self) -> str:
        return f"offer:{_digest(self.to_dict(include_id=False))[:24]}"

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = self.schema_version
        if include_id:
            row["offer_id"] = self.offer_id
        return row


@dataclass(frozen=True)
class StockBoundary:
    canonical_smiles: str
    boundary_type: str
    accepted: bool
    catalog_bindings: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    offers: tuple[StockOffer, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    schema_version: ClassVar[str] = "stock_boundary.v1"

    def __post_init__(self) -> None:
        if self.boundary_type != "unavailable" and (
            not self.canonical_smiles
            or _canonical_smiles(self.canonical_smiles) != self.canonical_smiles
        ):
            raise ValueError("stock boundary requires canonical isomeric SMILES")
        if self.boundary_type not in {
            "benchmark_stock",
            "commercially_orderable",
            "in_house_available",
            "common_commodity",
            "unavailable",
        }:
            raise ValueError("unsupported stock boundary type")
        if self.boundary_type == "commercially_orderable":
            valid_offers = [
                offer for offer in self.offers if offer.available and offer.snapshot_verified
            ]
            if self.accepted is not bool(valid_offers):
                raise ValueError("commercial stock acceptance must match available offers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_smiles": self.canonical_smiles,
            "boundary_type": self.boundary_type,
            "accepted": self.accepted,
            "catalog_bindings": [dict(row) for row in self.catalog_bindings],
            "offers": [offer.to_dict() for offer in self.offers],
            "reasons": list(self.reasons),
        }


class SnapshotStockProvider:
    """Resolve offers only against construction-time trusted snapshots.

    ``invoke`` requests may reference and repeat snapshot fields, but cannot
    create authority by choosing ``available=True`` and hashing that claim.
    Trusted in-memory snapshots or artifact files are loaded when this provider
    is constructed and every request is byte-semantically rebound to their
    canonical content.
    """

    descriptor = ProviderDescriptor(
        provider_id="autoplanner.snapshot_stock",
        kind=ProviderKind.STOCK,
        version="1.1.0",
        input_schemas=("stock_lookup_request.v1",),
        output_schemas=("stock_boundary.v1",),
        correlation_group="stock_snapshot",
        capabilities=("commercial_stock_lookup", "supplier_alternatives"),
        deterministic=True,
    )

    def __init__(
        self,
        *,
        trusted_snapshots: Any = (),
        trusted_snapshot_artifacts: Any = (),
    ) -> None:
        self._trusted_snapshots: dict[str, dict[str, Any]] = {}
        rows: list[Mapping[str, Any]] = []
        if isinstance(trusted_snapshots, Mapping):
            if trusted_snapshots.get("schema_version") == "stock_offer_snapshot.v1":
                rows.append(trusted_snapshots)
            else:
                rows.extend(
                    value
                    for value in trusted_snapshots.values()
                    if isinstance(value, Mapping)
                )
        else:
            rows.extend(
                value
                for value in trusted_snapshots or []
                if isinstance(value, Mapping)
            )
        for artifact in trusted_snapshot_artifacts or []:
            rows.extend(_stock_snapshots_from_artifact(Path(artifact)))
        for raw in rows:
            canonical = canonicalize_stock_snapshot(raw)
            digest = stock_snapshot_sha256(canonical)
            supplied = str(raw.get("snapshot_sha256") or "").lower()
            if supplied and supplied != digest:
                raise ValueError("trusted stock snapshot supplied digest mismatch")
            self._trusted_snapshots[digest] = canonical

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        del context
        target = _canonical_smiles(request.get("smiles"))
        offers: list[StockOffer] = []
        reasons: list[str] = []
        if not target:
            reasons.append("invalid_stock_lookup_smiles")
        for index, raw in enumerate(request.get("offers") or []):
            if not isinstance(raw, Mapping):
                reasons.append(f"offer:{index}:not_object")
                continue
            row = dict(raw)
            supplied_digest = str(row.get("snapshot_sha256") or "").lower()
            if not _is_sha256(supplied_digest):
                reasons.append(f"offer:{index}:invalid_snapshot_sha256")
                continue
            trusted_snapshot = self._trusted_snapshots.get(supplied_digest)
            if trusted_snapshot is None:
                reasons.append(f"offer:{index}:untrusted_snapshot")
                continue
            try:
                requested_snapshot = canonicalize_stock_snapshot(row)
            except (TypeError, ValueError) as exc:
                reasons.append(f"offer:{index}:invalid:{exc}")
                continue
            canonical = str(requested_snapshot.get("canonical_smiles") or "")
            if not canonical or canonical != target:
                reasons.append(f"offer:{index}:molecule_mismatch")
                continue
            if requested_snapshot != trusted_snapshot:
                reasons.append(f"offer:{index}:snapshot_content_mismatch")
                continue
            if stock_snapshot_sha256(requested_snapshot) != supplied_digest:
                reasons.append(f"offer:{index}:snapshot_digest_mismatch")
                continue
            try:
                offers.append(
                    StockOffer.from_trusted_snapshot(
                        trusted_snapshot,
                        snapshot_sha256=supplied_digest,
                    )
                )
            except (TypeError, ValueError) as exc:
                reasons.append(f"offer:{index}:invalid:{exc}")
        offers.sort(
            key=lambda row: (
                not row.available,
                row.price is None,
                row.price if row.price is not None else float("inf"),
                row.lead_time_days is None,
                row.lead_time_days if row.lead_time_days is not None else 10**9,
                row.supplier,
                row.catalog_number,
            )
        )
        boundary = StockBoundary(
            canonical_smiles=target,
            boundary_type="commercially_orderable" if target else "unavailable",
            accepted=bool(
                target
                and any(offer.available and offer.snapshot_verified for offer in offers)
            ),
            offers=tuple(offers),
            reasons=tuple(sorted(set(reasons))),
        )
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema=StockBoundary.schema_version,
            accepted=boundary.accepted,
            payload=boundary.to_dict(),
            reasons=boundary.reasons,
        )


def _canonical_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def canonicalize_stock_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one immutable supplier observation before hashing it."""
    if not isinstance(value, Mapping):
        raise TypeError("stock snapshot must be an object")
    row = dict(value)
    canonical = _canonical_smiles(row.get("canonical_smiles") or row.get("smiles"))
    if not canonical:
        raise ValueError("stock snapshot molecule is invalid")
    supplier = str(row.get("supplier") or "").strip()
    catalog_number = str(row.get("catalog_number") or row.get("sku") or "").strip()
    checked_at = str(row.get("checked_at") or "").strip()
    if not supplier or not catalog_number:
        raise ValueError("stock snapshot supplier and catalog number are required")
    _parse_timestamp(checked_at)
    available = row.get("available")
    if type(available) is not bool:
        raise ValueError("stock snapshot availability must be boolean")
    price = float(row["price"]) if row.get("price") is not None else None
    lead_time = int(row["lead_time_days"]) if row.get("lead_time_days") is not None else None
    if price is not None and price < 0:
        raise ValueError("stock snapshot price must be nonnegative")
    if lead_time is not None and lead_time < 0:
        raise ValueError("stock snapshot lead time must be nonnegative")
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("stock snapshot metadata must be an object")
    return {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": supplier,
        "catalog_number": catalog_number,
        "canonical_smiles": canonical,
        "checked_at": checked_at,
        "available": available,
        "purity": str(row.get("purity") or "").strip(),
        "pack_size": str(row.get("pack_size") or "").strip(),
        "price": price,
        "currency": str(row.get("currency") or "").strip().upper(),
        "region": str(row.get("region") or "").strip(),
        "lead_time_days": lead_time,
        "source_url": str(row.get("source_url") or "").strip(),
        "metadata": dict(metadata),
    }


def stock_snapshot_sha256(value: Mapping[str, Any]) -> str:
    return _digest(canonicalize_stock_snapshot(value))


def _snapshot_from_offer(value: StockOffer) -> dict[str, Any]:
    return canonicalize_stock_snapshot(
        {
            "supplier": value.supplier,
            "catalog_number": value.catalog_number,
            "canonical_smiles": value.canonical_smiles,
            "checked_at": value.checked_at,
            "available": value.available,
            "purity": value.purity,
            "pack_size": value.pack_size,
            "price": value.price,
            "currency": value.currency,
            "region": value.region,
            "lead_time_days": value.lead_time_days,
            "source_url": value.source_url,
            "metadata": dict(value.metadata),
        }
    )


def _parse_timestamp(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("stock snapshot checked_at is required")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError("stock snapshot checked_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stock snapshot checked_at must include a timezone")
    return parsed


def _is_sha256(value: str) -> bool:
    text = str(value or "")
    return bool(
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _stock_snapshots_from_artifact(path: Path) -> list[Mapping[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid trusted stock snapshot artifact: {resolved}") from exc
    if isinstance(payload, Mapping) and payload.get("schema_version") == "stock_offer_snapshot.v1":
        return [payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("offers"), list):
        return [row for row in payload["offers"] if isinstance(row, Mapping)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    raise ValueError(f"trusted stock snapshot artifact has unsupported schema: {resolved}")


def _digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
