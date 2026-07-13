"""Operator-facing adapters over the canonical V4 application service."""

from .case_dossier import (
    CASE_DOSSIER_SCHEMA,
    CaseDossierError,
    compile_case_dossier,
    load_case_dossier,
    with_case_dossier_digest,
    write_compiled_replay_pack,
)
from .case_runner import run_case_dossier
from .campaign_gateway import CampaignGateway, CampaignGatewayError
from .replay_pack import (
    ReplayPackError,
    load_replay_pack,
    run_replay_pack,
    with_replay_pack_digest,
)

__all__ = [
    "CASE_DOSSIER_SCHEMA",
    "CaseDossierError",
    "CampaignGateway",
    "CampaignGatewayError",
    "ReplayPackError",
    "compile_case_dossier",
    "load_case_dossier",
    "load_replay_pack",
    "run_replay_pack",
    "with_case_dossier_digest",
    "with_replay_pack_digest",
    "write_compiled_replay_pack",
    "run_case_dossier",
]
