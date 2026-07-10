"""Capability-aware registry for proposal, evidence, stock and runtime providers."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from cascade_planner.providers.contracts import (
    Provider,
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
    validate_provider_result,
)


class ProviderRegistryError(RuntimeError):
    pass


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._descriptors: dict[str, ProviderDescriptor] = {}
        self._trust_records: dict[str, dict[str, Any]] = {}
        self._by_kind: dict[ProviderKind, list[str]] = defaultdict(list)

    def register(
        self,
        provider: Provider,
        *,
        trusted_descriptor: ProviderDescriptor | None = None,
        authority: str = "",
    ) -> None:
        if not isinstance(provider, Provider):
            raise TypeError("provider does not implement Provider protocol")
        claimed = provider.descriptor
        if claimed.provider_id in self._providers:
            raise ProviderRegistryError(f"duplicate provider_id: {claimed.provider_id}")
        if trusted_descriptor is None:
            if claimed.kind is not ProviderKind.PROPOSAL:
                raise ProviderRegistryError(
                    f"untrusted provider cannot claim privileged kind: {claimed.kind.value}"
                )
            descriptor = replace(
                claimed,
                correlation_group="untrusted_provider",
                capabilities=(),
                deterministic=False,
                network_access=True,
            )
            trust_authority = "untrusted_third_party"
            trusted = False
        else:
            descriptor = trusted_descriptor
            if (
                descriptor.provider_id != claimed.provider_id
                or descriptor.version != claimed.version
                or descriptor.input_schemas != claimed.input_schemas
                or descriptor.output_schemas != claimed.output_schemas
            ):
                raise ProviderRegistryError("trusted descriptor identity/schema mismatch")
            trust_authority = str(authority or "").strip()
            if not trust_authority:
                raise ProviderRegistryError("trusted descriptor requires host authority")
            trusted = True
        self._providers[claimed.provider_id] = provider
        self._descriptors[claimed.provider_id] = descriptor
        self._trust_records[claimed.provider_id] = {
            "schema_version": "provider_host_trust_record.v1",
            "provider_id": claimed.provider_id,
            "trusted": trusted,
            "authority": trust_authority,
            "effective_kind": descriptor.kind.value,
            "effective_correlation_group": descriptor.correlation_group,
            "effective_deterministic": descriptor.deterministic,
            "claimed_kind": claimed.kind.value,
            "claimed_correlation_group": claimed.correlation_group,
            "claimed_deterministic": claimed.deterministic,
        }
        self._by_kind[descriptor.kind].append(claimed.provider_id)
        self._by_kind[descriptor.kind].sort()

    def unregister(self, provider_id: str) -> None:
        provider = self._providers.pop(str(provider_id), None)
        if provider is None:
            return
        descriptor = self._descriptors.pop(str(provider_id))
        self._trust_records.pop(str(provider_id), None)
        ids = self._by_kind.get(descriptor.kind) or []
        self._by_kind[descriptor.kind] = [value for value in ids if value != provider_id]

    def get(self, provider_id: str) -> Provider:
        try:
            return self._providers[str(provider_id)]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown provider_id: {provider_id}") from exc

    def descriptor(self, provider_id: str) -> ProviderDescriptor:
        try:
            return self._descriptors[str(provider_id)]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown provider_id: {provider_id}") from exc

    def trust_record(self, provider_id: str) -> dict[str, Any]:
        if str(provider_id) not in self._trust_records:
            raise ProviderRegistryError(f"unknown provider_id: {provider_id}")
        return dict(self._trust_records[str(provider_id)])

    def descriptors(
        self,
        *,
        kind: ProviderKind | None = None,
        capability: str = "",
    ) -> list[ProviderDescriptor]:
        ids: Iterable[str]
        if kind is None:
            ids = sorted(self._providers)
        else:
            ids = list(self._by_kind.get(kind) or [])
        rows = [self._descriptors[provider_id] for provider_id in ids]
        if capability:
            rows = [row for row in rows if capability in row.capabilities]
        return rows

    def invoke(
        self,
        provider_id: str,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        provider = self.get(provider_id)
        descriptor = self.descriptor(provider_id)
        request_schema = str(request.get("schema_version") or "")
        if request_schema not in descriptor.input_schemas:
            raise ProviderRegistryError(
                f"provider {provider_id} does not accept schema {request_schema!r}"
            )
        result = provider.invoke(dict(request), context=context)
        claimed = provider.descriptor
        raw = result.to_dict()
        identity_reasons = []
        if raw.get("provider_id") != claimed.provider_id:
            identity_reasons.append("provider_result_id_mismatch")
        if raw.get("provider_version") != claimed.version:
            identity_reasons.append("provider_result_version_mismatch")
        if raw.get("output_schema") not in descriptor.output_schemas:
            identity_reasons.append("provider_result_output_schema_not_declared")
        if identity_reasons:
            raise ProviderRegistryError(
                f"provider {provider_id} returned invalid result: {','.join(identity_reasons)}"
            )
        result = ProviderResultEnvelope(
            provider_id=descriptor.provider_id,
            provider_version=descriptor.version,
            provider_kind=descriptor.kind,
            correlation_group=descriptor.correlation_group,
            output_schema=result.output_schema,
            accepted=result.accepted,
            payload=dict(result.payload),
            reasons=tuple(result.reasons),
            source_refs=tuple(result.source_refs),
            evidence_refs=tuple(result.evidence_refs),
            no_solved_claim=result.no_solved_claim,
        )
        reasons = validate_provider_result(result.to_dict(), descriptor=descriptor)
        if reasons:
            raise ProviderRegistryError(
                f"provider {provider_id} returned invalid result: {','.join(reasons)}"
            )
        return result

    def correlation_groups(self, provider_ids: Iterable[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for provider_id in provider_ids:
            descriptor = self.descriptor(provider_id)
            groups[descriptor.correlation_group].append(descriptor.provider_id)
        return {key: sorted(values) for key, values in sorted(groups.items())}


def descriptor_from_dict(value: Mapping[str, Any]) -> ProviderDescriptor:
    row = dict(value)
    return ProviderDescriptor(
        provider_id=str(row.get("provider_id") or ""),
        kind=ProviderKind(str(row.get("kind") or "")),
        version=str(row.get("version") or ""),
        input_schemas=tuple(str(item) for item in row.get("input_schemas") or []),
        output_schemas=tuple(str(item) for item in row.get("output_schemas") or []),
        correlation_group=str(row.get("correlation_group") or ""),
        capabilities=tuple(str(item) for item in row.get("capabilities") or []),
        deterministic=bool(row.get("deterministic")),
        network_access=bool(row.get("network_access")),
        estimated_cost_units=float(row.get("estimated_cost_units") or 0.0),
    )
