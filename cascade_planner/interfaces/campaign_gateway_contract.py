"""Shared result and error contract for CampaignGateway helper modules."""

CAMPAIGN_GATEWAY_RESULT_SCHEMA = "autoplanner_campaign_gateway_result.v1"


class CampaignGatewayError(RuntimeError):
    """An operator request cannot be mapped to one canonical V4 run."""


__all__ = ["CAMPAIGN_GATEWAY_RESULT_SCHEMA", "CampaignGatewayError"]
