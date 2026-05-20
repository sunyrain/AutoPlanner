"""Centralized results-path helper.

Use ``results_dir(version="v2")`` instead of hard-coding ``results/`` so we
can keep v1 and v2 artifacts in separate folders without filename collisions.

Convention (post 2026-04-23):
  results/v1/      -- original cascade_dataset.normalized.uniprot.json artifacts (archived)
  results/v2/      -- cascade_dataset_v2.normalized.json artifacts (current)
  results/shared/  -- cross-version: enzymemap templates, demo route json, atom-map cache, etc.

Default version is taken from env ``CASCADE_VERSION`` (default "v2").
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = ROOT / "results"

DEFAULT_VERSION = os.environ.get("CASCADE_VERSION", "v2")


def results_dir(version: str | None = None) -> Path:
    v = version or DEFAULT_VERSION
    p = RESULTS_BASE / v
    p.mkdir(parents=True, exist_ok=True)
    return p


def shared_dir() -> Path:
    p = RESULTS_BASE / "shared"
    p.mkdir(parents=True, exist_ok=True)
    return p


def aizdata_dir() -> Path:
    p = ROOT / "workspace" / "aizdata"
    if p.exists():
        return p
    return ROOT / "aizdata"
