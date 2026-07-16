"""Run identity helpers kept outside the bounded CampaignGateway façade."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re


def new_run_id(target_name: str, target_smiles: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "-", target_name.lower()).strip("-")[:36]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(
        f"{target_name}\0{target_smiles}\0{stamp}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{label or 'target'}-{stamp}-{digest}"


def run_segment(run_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip(".-")[:64] or "run"
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"{label}--{digest}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["new_run_id", "run_segment", "utc_now"]
