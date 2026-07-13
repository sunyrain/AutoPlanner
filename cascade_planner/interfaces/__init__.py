"""Operator-facing adapters over the canonical V4 application service."""

from .campaign_gateway import CampaignGateway, CampaignGatewayError

__all__ = ["CampaignGateway", "CampaignGatewayError"]
