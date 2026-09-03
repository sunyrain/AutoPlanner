"""Compile the default provider-set stock oracle for CampaignGateway."""

from __future__ import annotations

from cascade_planner.application.unified_campaign_spec import StockOracleReference
from cascade_planner.providers.contracts import ProviderKind
from cascade_planner.providers.registry import ProviderRegistry
from cascade_planner.providers.stock import stock_provider_set_authority_binding


def default_stock_oracle_reference(
    providers: ProviderRegistry,
    *,
    boundary: str,
) -> StockOracleReference:
    stock_providers = [
        providers.get(descriptor.provider_id)
        for descriptor in providers.descriptors(kind=ProviderKind.STOCK)
    ]
    binding = stock_provider_set_authority_binding(stock_providers)
    return StockOracleReference.from_binding(
        oracle_id=f"provider-set:{binding['content_sha256'][:24]}",
        boundary=boundary,
        binding=binding,
    )


__all__ = ["default_stock_oracle_reference"]
