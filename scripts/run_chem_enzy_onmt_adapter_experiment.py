#!/usr/bin/env python3
"""Run a ChemEnzy OpenNMT adapter experiment from the active repository root.

The implementation lives in the 2026-06-05 harness archive. This thin wrapper
keeps the historical runner usable after repository cleanup while making all
relative paths resolve against the current project root.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_RUNNER = REPO_ROOT / "archive" / "harness_prep_20260605" / "scripts" / "run_chem_enzy_onmt_adapter_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_archived_chem_enzy_onmt_runner", ARCHIVED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load archived runner: {ARCHIVED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO_ROOT = REPO_ROOT
    return module


def main() -> None:
    _load_runner().main()


if __name__ == "__main__":
    main()
