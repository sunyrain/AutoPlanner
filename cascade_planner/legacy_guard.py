"""Guards for frozen research lines.

These helpers keep old experiments importable for reproducibility while making
direct execution opt-in. Runtime code should not call them.
"""
from __future__ import annotations

import os


LEGACY_RESEARCH_ENV = "AUTOPLANNER_ALLOW_LEGACY_RESEARCH"


def legacy_research_enabled() -> bool:
    value = str(os.environ.get(LEGACY_RESEARCH_ENV) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def require_legacy_research_enabled(component: str) -> None:
    if legacy_research_enabled():
        return
    raise SystemExit(
        f"{component} is archived/frozen research code and is not part of the "
        f"current mainline. Set {LEGACY_RESEARCH_ENV}=1 to reproduce old reports."
    )
