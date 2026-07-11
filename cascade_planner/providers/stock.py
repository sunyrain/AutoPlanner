"""Immutable stock boundaries and timestamped supplier-offer snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from rdkit import Chem, RDLogger

from cascade_planner.providers.contracts import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
    validate_provider_result,
)


RDLogger.DisableLog("rdApp.*")


STOCK_PROVIDER_AUTHORITY_BINDING_SCHEMA = "stock_provider_authority_binding.v1"
STOCK_PROVIDER_SET_BINDING_SCHEMA = "stock_provider_set_binding.v1"
STOCK_PROVIDER_OBSERVATION_SCHEMA = "stock_provider_observation.v1"
STOCK_OBSERVATION_STATE_SCHEMA = "stock_observation_state.v1"


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
        version="1.2.0",
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
        raw_offers = request.get("offers") or []
        if not target:
            reasons.append("invalid_stock_lookup_smiles")
        if not self._trusted_snapshots:
            reasons.append("no_trusted_stock_snapshots_configured")
        if not raw_offers:
            reasons.append("no_stock_offers_supplied")
        for index, raw in enumerate(raw_offers):
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


class BenchmarkCatalogStockProvider:
    """Resolve benchmark membership from one construction-time hashed file.

    This boundary deliberately means benchmark membership, never commercial
    orderability. The caller must supply the expected SHA-256; an unhashed
    ChemEnzy/native in-memory stock hit cannot create authority here.
    """

    descriptor = ProviderDescriptor(
        provider_id="autoplanner.benchmark_catalog_stock",
        kind=ProviderKind.STOCK,
        version="1.0.0",
        input_schemas=("stock_lookup_request.v1",),
        output_schemas=("stock_boundary.v1",),
        correlation_group="benchmark_catalog_artifact",
        capabilities=("benchmark_stock_membership",),
        deterministic=True,
    )

    def __init__(
        self,
        *,
        catalog_artifact: str | Path,
        catalog_sha256: str,
        catalog_name: str = "benchmark-stock",
    ) -> None:
        path = Path(catalog_artifact).resolve()
        expected = str(catalog_sha256 or "").strip().lower()
        if not path.is_file():
            raise ValueError("benchmark stock catalog artifact is missing")
        if not _is_sha256(expected):
            raise ValueError("benchmark stock catalog SHA-256 is required")
        actual = _sha256_path(path)
        if actual != expected:
            raise ValueError("benchmark stock catalog SHA-256 mismatch")
        self.catalog_artifact = path
        self.catalog_sha256 = actual
        self.catalog_name = str(catalog_name or "benchmark-stock").strip()
        self._stock = _load_benchmark_catalog_smiles(path)
        if not self._stock:
            raise ValueError("benchmark stock catalog contains no valid SMILES")

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        del context
        target = _canonical_smiles(request.get("smiles"))
        accepted = bool(target and target in self._stock)
        reasons: list[str] = []
        if not target:
            reasons.append("invalid_stock_lookup_smiles")
        elif not accepted:
            reasons.append("molecule_not_in_hashed_benchmark_catalog")
        bindings = (
            {
                "catalog_name": self.catalog_name,
                "catalog_path": str(self.catalog_artifact),
                "catalog_sha256": self.catalog_sha256,
                "canonical_smiles": target,
                "artifact_hash_verified": True,
                "commercial_orderability_claimed": False,
            },
        ) if accepted else ()
        boundary = StockBoundary(
            canonical_smiles=target,
            boundary_type="benchmark_stock" if target else "unavailable",
            accepted=accepted,
            catalog_bindings=bindings,
            reasons=tuple(reasons),
        )
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema=StockBoundary.schema_version,
            accepted=accepted,
            payload=boundary.to_dict(),
            reasons=boundary.reasons,
            source_refs=(str(self.catalog_artifact),),
        )


def stock_provider_authority_binding(provider: Any) -> dict[str, Any]:
    """Describe the host-owned material that makes one stock provider authoritative.

    A provider descriptor alone is insufficient: two benchmark providers can
    share code/version while loading different catalogs, and two snapshot
    providers can trust different immutable observations.  This binding is
    intentionally deterministic so queue refreshes and campaign policy can
    detect either kind of authority change.
    """

    descriptor = getattr(provider, "descriptor", None)
    if descriptor is None or not callable(getattr(descriptor, "to_dict", None)):
        raise TypeError("stock provider descriptor is required")
    descriptor_row = _json_value(descriptor.to_dict())
    if descriptor_row.get("kind") != ProviderKind.STOCK.value:
        raise ValueError("provider set contains a non-stock provider")
    authority_material: dict[str, Any]
    if type(provider) is SnapshotStockProvider:
        authority_material = {
            "kind": "trusted_snapshot_set",
            "trusted_snapshot_sha256": sorted(
                str(item)
                for item in getattr(provider, "_trusted_snapshots", {}).keys()
                if _is_sha256(str(item))
            ),
        }
    elif type(provider) is BenchmarkCatalogStockProvider:
        authority_material = {
            "kind": "hashed_benchmark_catalog",
            "catalog_sha256": str(provider.catalog_sha256),
            "catalog_name": str(provider.catalog_name),
        }
    else:
        # Unknown implementations can be scheduled and recorded, but the
        # ledger's replay allowlist will not grant them positive authority.
        authority_material = {
            "kind": "unrecognized_stock_provider",
            "runtime_type": (
                f"{type(provider).__module__}.{type(provider).__qualname__}"
            ),
        }
    payload = {
        "schema_version": STOCK_PROVIDER_AUTHORITY_BINDING_SCHEMA,
        "provider_descriptor": descriptor_row,
        "authority_material": authority_material,
    }
    payload["content_sha256"] = _digest(payload)
    return _json_value(payload)


def stock_provider_set_authority_binding(
    providers: Sequence[Any] | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a stable policy binding for the complete provider set."""

    values = list(providers.values()) if isinstance(providers, Mapping) else list(providers)
    if not values:
        raise ValueError("at least one stock provider is required")
    bindings = [stock_provider_authority_binding(provider) for provider in values]
    bindings.sort(
        key=lambda row: str(
            dict(row.get("provider_descriptor") or {}).get("provider_id") or ""
        )
    )
    provider_ids = [
        str(dict(row.get("provider_descriptor") or {}).get("provider_id") or "")
        for row in bindings
    ]
    if any(not item for item in provider_ids) or len(provider_ids) != len(
        set(provider_ids)
    ):
        raise ValueError("stock provider ids must be nonempty and unique")
    payload = {
        "schema_version": STOCK_PROVIDER_SET_BINDING_SCHEMA,
        "providers": bindings,
    }
    payload["content_sha256"] = _digest(payload)
    return _json_value(payload)


def build_stock_provider_observation(
    provider: Any,
    *,
    request: Mapping[str, Any],
    observed_at: str,
    provider_result: Mapping[str, Any] | None = None,
    invocation_error: str = "",
) -> dict[str, Any]:
    """Materialize one immutable provider invocation observation."""

    _parse_timestamp(str(observed_at or ""))
    binding = stock_provider_authority_binding(provider)
    descriptor = dict(binding.get("provider_descriptor") or {})
    request_row = _json_value(dict(request))
    result_row = _json_value(dict(provider_result or {}))
    error = str(invocation_error or "")
    if bool(result_row) == bool(error):
        raise ValueError("stock observation requires exactly one result or invocation error")
    payload = {
        "schema_version": STOCK_PROVIDER_OBSERVATION_SCHEMA,
        "observed_at": str(observed_at),
        "provider_id": str(descriptor.get("provider_id") or ""),
        "provider_authority_binding": binding,
        "request": request_row,
        "request_sha256": _digest(request_row),
        "provider_result": result_row,
        "invocation_error": error,
    }
    payload["observation_id"] = "stock-observation:sha256:" + _digest(payload)
    return _json_value(payload)


def build_stock_observation_state(
    *,
    provider_set_binding: Mapping[str, Any],
    current_observations: Sequence[Mapping[str, Any]],
    refreshed_at: str,
    previous_states: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the current stock projection while retaining immutable history."""

    _parse_timestamp(str(refreshed_at or ""))
    provider_set = _json_value(dict(provider_set_binding))
    current = [_json_value(dict(row)) for row in current_observations]
    current.sort(key=lambda row: (str(row.get("provider_id") or ""), str(row.get("observation_id") or "")))
    history_by_id: dict[str, dict[str, Any]] = {}
    for raw_state in previous_states:
        if validate_stock_observation_state(raw_state):
            continue
        for raw in raw_state.get("history") or []:
            if not isinstance(raw, Mapping):
                continue
            row = _json_value(dict(raw))
            observation_id = str(row.get("observation_id") or "")
            if observation_id:
                history_by_id[observation_id] = row
    for row in current:
        observation_id = str(row.get("observation_id") or "")
        if observation_id:
            history_by_id[observation_id] = row
    history = sorted(
        history_by_id.values(),
        key=lambda row: (
            str(row.get("observed_at") or ""),
            str(row.get("provider_id") or ""),
            str(row.get("observation_id") or ""),
        ),
    )
    payload = {
        "schema_version": STOCK_OBSERVATION_STATE_SCHEMA,
        "provider_set_binding": provider_set,
        "refreshed_at": str(refreshed_at),
        "current": current,
        "history": history,
    }
    payload["content_sha256"] = _digest(payload)
    reasons = validate_stock_observation_state(payload)
    if reasons:
        raise ValueError("invalid stock observation state: " + ",".join(reasons))
    return _json_value(payload)


def validate_stock_observation_state(
    value: Any,
    *,
    expected_smiles: str = "",
) -> list[str]:
    """Validate observation history structure without granting stock authority."""

    if not isinstance(value, Mapping):
        return ["stock_observation_state_not_object"]
    state = dict(value)
    reasons: list[str] = []
    if state.get("schema_version") != STOCK_OBSERVATION_STATE_SCHEMA:
        reasons.append("invalid_stock_observation_state_schema")
    state_payload = dict(state)
    supplied_state_digest = str(state_payload.pop("content_sha256", ""))
    if not supplied_state_digest or supplied_state_digest != _digest(state_payload):
        reasons.append("stock_observation_state_digest_invalid")
    try:
        _parse_timestamp(str(state.get("refreshed_at") or ""))
    except ValueError:
        reasons.append("stock_observation_refreshed_at_invalid")
    provider_set = state.get("provider_set_binding")
    provider_bindings: dict[str, dict[str, Any]] = {}
    if not isinstance(provider_set, Mapping):
        reasons.append("stock_provider_set_binding_not_object")
    else:
        set_row = dict(provider_set)
        if set_row.get("schema_version") != STOCK_PROVIDER_SET_BINDING_SCHEMA:
            reasons.append("invalid_stock_provider_set_binding_schema")
        set_payload = dict(set_row)
        supplied_set_digest = str(set_payload.pop("content_sha256", ""))
        if not supplied_set_digest or supplied_set_digest != _digest(set_payload):
            reasons.append("stock_provider_set_binding_digest_invalid")
        raw_bindings = set_row.get("providers")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            reasons.append("stock_provider_set_bindings_missing")
        else:
            for index, raw_binding in enumerate(raw_bindings):
                binding_reasons, provider_id = _stock_provider_binding_reasons(
                    raw_binding
                )
                reasons.extend(
                    f"stock_provider_set:{index}:{reason}"
                    for reason in binding_reasons
                )
                if provider_id:
                    if provider_id in provider_bindings:
                        reasons.append("stock_provider_set_duplicate_provider_id")
                    elif isinstance(raw_binding, Mapping):
                        provider_bindings[provider_id] = dict(raw_binding)
    current = state.get("current")
    history = state.get("history")
    if not isinstance(current, list):
        reasons.append("stock_observation_current_not_list")
        current = []
    if not isinstance(history, list):
        reasons.append("stock_observation_history_not_list")
        history = []
    expected = _canonical_smiles(expected_smiles) if expected_smiles else ""
    if expected_smiles and expected != str(expected_smiles):
        reasons.append("stock_observation_expected_smiles_invalid")
    history_ids: set[str] = set()
    for index, raw in enumerate(history):
        observation_reasons, observation_id, _ = _stock_observation_reasons(
            raw,
            expected_smiles=expected,
        )
        reasons.extend(
            f"stock_observation_history:{index}:{reason}"
            for reason in observation_reasons
        )
        if observation_id:
            if observation_id in history_ids:
                reasons.append("stock_observation_history_duplicate_id")
            history_ids.add(observation_id)
    current_provider_ids: set[str] = set()
    for index, raw in enumerate(current):
        observation_reasons, observation_id, provider_id = _stock_observation_reasons(
            raw,
            expected_smiles=expected,
        )
        reasons.extend(
            f"stock_observation_current:{index}:{reason}"
            for reason in observation_reasons
        )
        if observation_id and observation_id not in history_ids:
            reasons.append("stock_observation_current_missing_from_history")
        if provider_id:
            if provider_id in current_provider_ids:
                reasons.append("stock_observation_current_duplicate_provider_id")
            current_provider_ids.add(provider_id)
            expected_binding = provider_bindings.get(provider_id)
            supplied_binding = (
                dict(raw.get("provider_authority_binding") or {})
                if isinstance(raw, Mapping)
                else {}
            )
            if expected_binding != supplied_binding:
                reasons.append("stock_observation_current_provider_binding_mismatch")
    if current_provider_ids != set(provider_bindings):
        reasons.append("stock_observation_current_provider_set_incomplete")
    return sorted(set(reasons))


def _stock_provider_binding_reasons(value: Any) -> tuple[list[str], str]:
    if not isinstance(value, Mapping):
        return ["binding_not_object"], ""
    row = dict(value)
    reasons: list[str] = []
    if row.get("schema_version") != STOCK_PROVIDER_AUTHORITY_BINDING_SCHEMA:
        reasons.append("binding_schema_invalid")
    payload = dict(row)
    supplied_digest = str(payload.pop("content_sha256", ""))
    if not supplied_digest or supplied_digest != _digest(payload):
        reasons.append("binding_digest_invalid")
    descriptor = row.get("provider_descriptor")
    if not isinstance(descriptor, Mapping):
        reasons.append("binding_descriptor_not_object")
        return reasons, ""
    provider_id = str(descriptor.get("provider_id") or "")
    if not provider_id or descriptor.get("kind") != ProviderKind.STOCK.value:
        reasons.append("binding_descriptor_identity_invalid")
    if not isinstance(row.get("authority_material"), Mapping):
        reasons.append("binding_authority_material_not_object")
    return reasons, provider_id


def _stock_observation_reasons(
    value: Any,
    *,
    expected_smiles: str,
) -> tuple[list[str], str, str]:
    if not isinstance(value, Mapping):
        return ["observation_not_object"], "", ""
    row = dict(value)
    reasons: list[str] = []
    if row.get("schema_version") != STOCK_PROVIDER_OBSERVATION_SCHEMA:
        reasons.append("observation_schema_invalid")
    observation_id = str(row.get("observation_id") or "")
    identity_payload = dict(row)
    identity_payload.pop("observation_id", None)
    expected_id = "stock-observation:sha256:" + _digest(identity_payload)
    if observation_id != expected_id:
        reasons.append("observation_id_invalid")
    try:
        _parse_timestamp(str(row.get("observed_at") or ""))
    except ValueError:
        reasons.append("observation_timestamp_invalid")
    binding_reasons, binding_provider_id = _stock_provider_binding_reasons(
        row.get("provider_authority_binding")
    )
    reasons.extend(f"observation_provider:{reason}" for reason in binding_reasons)
    provider_id = str(row.get("provider_id") or "")
    if not provider_id or provider_id != binding_provider_id:
        reasons.append("observation_provider_id_mismatch")
    request = row.get("request")
    if not isinstance(request, Mapping):
        reasons.append("observation_request_not_object")
        request = {}
    if str(row.get("request_sha256") or "") != _digest(dict(request)):
        reasons.append("observation_request_digest_invalid")
    request_smiles = _canonical_smiles(request.get("smiles"))
    if expected_smiles and request_smiles != expected_smiles:
        reasons.append("observation_request_smiles_mismatch")
    provider_result = row.get("provider_result")
    result_row = dict(provider_result) if isinstance(provider_result, Mapping) else {}
    invocation_error = str(row.get("invocation_error") or "")
    if bool(result_row) == bool(invocation_error):
        reasons.append("observation_result_error_exclusivity_invalid")
    if result_row:
        reasons.extend(
            f"observation_result:{reason}"
            for reason in validate_provider_result(result_row)
        )
        descriptor = (
            dict(dict(row.get("provider_authority_binding") or {}).get("provider_descriptor") or {})
            if isinstance(row.get("provider_authority_binding"), Mapping)
            else {}
        )
        if (
            result_row.get("provider_id") != provider_id
            or result_row.get("provider_version") != descriptor.get("version")
            or result_row.get("provider_kind") != ProviderKind.STOCK.value
            or result_row.get("correlation_group") != descriptor.get("correlation_group")
            or result_row.get("output_schema")
            not in (descriptor.get("output_schemas") or [])
        ):
            reasons.append("observation_result_provider_binding_mismatch")
    return sorted(set(reasons)), observation_id, provider_id


def replay_stock_provider_result(
    value: Mapping[str, Any],
    *,
    expected_smiles: str,
    trusted_provider_instances: Mapping[str, Any] | None,
    context: ProviderContext | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Replay a stock envelope against a host-owned built-in provider.

    A serialized provider envelope and its content hash are not a trust root:
    an attacker can invent both.  Positive stock authority is therefore
    granted only after the exact built-in provider type supplied by the host
    reproduces the complete envelope from materialized request fields.

    The returned binding is JSON-canonical and suitable for embedding in a
    higher-level proof artifact.  It records the replay, but callers that are
    making a fresh authority decision must still call this function with
    their own trusted provider instance.
    """

    expected = _canonical_smiles(expected_smiles)
    if not expected or expected != str(expected_smiles or ""):
        return {}, ("stock_replay_expected_smiles_invalid",)
    if not isinstance(value, Mapping):
        return {}, ("stock_replay_envelope_not_object",)
    result = _json_value(dict(value))
    provider_id = str(result.get("provider_id") or "")
    provider_classes = {
        SnapshotStockProvider.descriptor.provider_id: SnapshotStockProvider,
        BenchmarkCatalogStockProvider.descriptor.provider_id: (
            BenchmarkCatalogStockProvider
        ),
    }
    provider_class = provider_classes.get(provider_id)
    if provider_class is None:
        return {}, ("stock_replay_provider_not_allowlisted",)
    provider = dict(trusted_provider_instances or {}).get(provider_id)
    if type(provider) is not provider_class:
        return {}, ("stock_replay_trusted_provider_missing_or_type_mismatch",)
    descriptor = provider_class.descriptor
    envelope_reasons = validate_provider_result(result, descriptor=descriptor)
    if envelope_reasons:
        return {}, tuple(
            f"stock_replay_envelope:{reason}" for reason in envelope_reasons
        )
    payload = result.get("payload")
    if not isinstance(payload, Mapping):
        return {}, ("stock_replay_payload_not_object",)
    payload_row = dict(payload)
    if (
        result.get("accepted") is not True
        or result.get("output_schema") != StockBoundary.schema_version
        or payload_row.get("schema_version") != StockBoundary.schema_version
        or payload_row.get("accepted") is not True
        or _canonical_smiles(payload_row.get("canonical_smiles")) != expected
    ):
        return {}, ("stock_replay_envelope_not_accepted_boundary",)

    request: dict[str, Any] = {
        "schema_version": "stock_lookup_request.v1",
        "smiles": expected,
    }
    if provider_class is SnapshotStockProvider:
        offers = payload_row.get("offers")
        if not isinstance(offers, list) or not offers:
            return {}, ("stock_replay_snapshot_offers_missing",)
        request_offers: list[dict[str, Any]] = []
        for raw_offer in offers:
            if not isinstance(raw_offer, Mapping):
                return {}, ("stock_replay_snapshot_offer_not_object",)
            offer = dict(raw_offer)
            snapshot = offer.get("snapshot")
            if not isinstance(snapshot, Mapping):
                return {}, ("stock_replay_snapshot_materialization_missing",)
            request_offers.append(
                {
                    **dict(snapshot),
                    "snapshot_sha256": str(offer.get("snapshot_sha256") or ""),
                }
            )
        request["offers"] = request_offers
    elif payload_row.get("boundary_type") != "benchmark_stock":
        return {}, ("stock_replay_benchmark_boundary_type_invalid",)

    replay_context = context or ProviderContext(
        run_id="frontier-ledger-stock-replay",
        case_id="frontier-ledger-stock-replay",
        target_smiles=expected,
    )
    try:
        replayed = _json_value(
            provider.invoke(request, context=replay_context).to_dict()
        )
    except (OSError, TypeError, ValueError) as exc:
        return {}, (f"stock_replay_provider_error:{type(exc).__name__}",)
    if replayed != result:
        return {}, ("stock_replay_result_mismatch",)
    request = _json_value(request)
    descriptor_row = _json_value(descriptor.to_dict())
    return {
        "schema_version": "stock_provider_host_replay_binding.v1",
        "canonical_smiles": expected,
        "provider_id": provider_id,
        "provider_version": descriptor.version,
        "provider_descriptor_sha256": _digest(descriptor_row),
        "replay_request": request,
        "replay_request_sha256": _digest(request),
        "provider_result": replayed,
        "provider_result_content_hash": str(replayed.get("content_hash") or ""),
        "authority": "current_host_stock_provider_replay",
    }, ()


def build_trusted_stock_provider_instances(
    *,
    stock_snapshots: Mapping[str, Any] | None = None,
    benchmark_catalog_artifact: str | Path = "",
    benchmark_catalog_sha256: str = "",
    benchmark_catalog_name: str = "",
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Construct the host trust set used to replay campaign stock results."""

    providers: dict[str, Any] = {}
    reasons: list[str] = []
    artifact = str(benchmark_catalog_artifact or "").strip()
    if artifact:
        try:
            benchmark = BenchmarkCatalogStockProvider(
                catalog_artifact=artifact,
                catalog_sha256=str(benchmark_catalog_sha256 or ""),
                catalog_name=str(benchmark_catalog_name or "benchmark-stock"),
            )
        except (OSError, TypeError, ValueError) as exc:
            reasons.append(
                f"benchmark_stock_provider_construction_error:{type(exc).__name__}"
            )
        else:
            providers[benchmark.descriptor.provider_id] = benchmark

    snapshot_rows: list[dict[str, Any]] = []
    for raw in dict(stock_snapshots or {}).values():
        if not isinstance(raw, Mapping):
            reasons.append("stock_snapshot_config_entry_not_object")
            continue
        row = dict(raw)
        candidates = (
            [row]
            if row.get("schema_version") == "stock_offer_snapshot.v1"
            or {"supplier", "catalog_number", "available"}.issubset(row)
            else [
                dict(candidate)
                for candidate in row.get("offers") or []
                if isinstance(candidate, Mapping)
            ]
        )
        for candidate in candidates:
            try:
                canonicalize_stock_snapshot(candidate)
            except (TypeError, ValueError):
                reasons.append("stock_snapshot_config_entry_invalid")
                continue
            snapshot_rows.append(candidate)
    if snapshot_rows:
        try:
            snapshots = SnapshotStockProvider(trusted_snapshots=snapshot_rows)
        except (OSError, TypeError, ValueError) as exc:
            reasons.append(
                f"snapshot_stock_provider_construction_error:{type(exc).__name__}"
            )
        else:
            providers[snapshots.descriptor.provider_id] = snapshots
    return providers, tuple(sorted(set(reasons)))


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_benchmark_catalog_smiles(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("benchmark stock catalog must be UTF-8 text") from exc
    if not text.strip():
        return set()
    if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        values = [line.strip().split()[0] for line in text.splitlines() if line.strip()]
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(text.splitlines(), dialect=dialect))
        if not rows:
            return set()
        header = [str(value or "").strip().lower() for value in rows[0]]
        aliases = {"smiles", "canonical_smiles", "canonical_isomeric_smiles", "mol"}
        column = next((index for index, name in enumerate(header) if name in aliases), None)
        data = rows[1:] if column is not None else rows
        values = []
        for row in data:
            if column is not None:
                candidates = [row[column]] if column < len(row) else []
            else:
                candidates = row[:1]
            values.extend(str(value or "").strip() for value in candidates)
    return {
        canonical
        for canonical in (_canonical_smiles(value) for value in values)
        if canonical
    }


def _digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
