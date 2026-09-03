#!/usr/bin/env python3
"""Discover and audit Candidate Program Workbench snapshots without mutation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.interfaces.candidate_migration import (  # noqa: E402
    audit_candidate_workbench_snapshots,
)
from cascade_planner.application.target_route_readiness import (  # noqa: E402
    compile_target_route_readiness,
    current_replay_attestation_from_receipt,
)
from cascade_planner.runtime.canonical_json import (  # noqa: E402
    strict_canonical_json_sha256,
)


_SIGNATURE = b"retrosynthesis_route_workbench.v1"
_SKIP_DIRECTORIES = {
    ".browser_pdf_profile",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Files or directories to inspect.")
    parser.add_argument("--output", help="Optional full audit JSON output path.")
    parser.add_argument("--catalog", help="Optional target_route_catalog.v1 JSON.")
    parser.add_argument(
        "--current-replay-receipt",
        action="append",
        default=[],
        help="Verified current canonical replay receipt; repeatable.",
    )
    parser.add_argument("--readiness-output", help="Optional readiness JSON output path.")
    parser.add_argument("--minimum-long-route-steps", type=int, default=10)
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    args = parser.parse_args(argv)

    paths = _discover([Path(value).expanduser().resolve() for value in args.roots])
    snapshots: list[tuple[str, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            if path.stat().st_size > args.max_bytes:
                raise ValueError("candidate_workbench_file_too_large")
            payload = path.read_bytes()
            named_workbench = path.name == "route_workbench.json"
            if _SIGNATURE not in payload and not named_workbench:
                continue
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("candidate_workbench_json_object_required")
            if value.get("schema_version") != "retrosynthesis_route_workbench.v1":
                if named_workbench:
                    raise ValueError("candidate_workbench_schema_invalid")
                continue
            snapshots.append((_display_path(path), value))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "source_ref": _display_path(path),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )

    audit = audit_candidate_workbench_snapshots(snapshots)
    _write(args.output, audit)
    readiness = None
    if args.catalog:
        catalog_path = Path(args.catalog).expanduser().resolve()
        catalog = _load_catalog(catalog_path)
        attestations = [
            current_replay_attestation_from_receipt(
                _load_json(Path(value).expanduser().resolve()),
                source_ref=_display_path(Path(value).expanduser().resolve()),
            )
            for value in args.current_replay_receipt
        ]
        readiness = compile_target_route_readiness(
            catalog["targets"],
            audit,
            authority_attestations=attestations,
            minimum_long_route_steps=args.minimum_long_route_steps,
        )
        _write(args.readiness_output, readiness)
    elif args.current_replay_receipt or args.readiness_output:
        parser.error("--catalog is required for readiness output or attestations")
    summary = {
        "schema_version": "candidate_program_migration_scan_result.v1",
        "discovered_candidate_file_count": len(paths),
        "parsed_workbench_count": len(snapshots),
        "discovery_error_count": len(errors),
        "discovery_errors": errors,
        "audit_sha256": audit["content_sha256"],
        "snapshot_count": audit["snapshot_count"],
        "unique_workbench_count": audit["unique_workbench_count"],
        "duplicate_snapshot_count": audit["duplicate_snapshot_count"],
        "target_count": audit["target_count"],
        "migration_state_counts": audit["migration_state_counts"],
        "semantics": audit["semantics"],
    }
    if readiness is not None:
        summary["readiness_sha256"] = readiness["content_sha256"]
        summary["readiness_summary"] = readiness["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


def _discover(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"candidate_workbench_root_not_found:{root}")
        if root.is_file():
            found.add(root)
            continue
        for directory, names, files in os.walk(root):
            names[:] = sorted(name for name in names if name not in _SKIP_DIRECTORIES)
            parent = Path(directory)
            object_store = "objects" in parent.parts and "sha256" in parent.parts
            for name in sorted(files):
                if name == "route_workbench.json" or (object_store and "." not in name):
                    found.add(parent / name)
    return sorted(found, key=lambda path: str(path).casefold())


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write(value: str | None, payload: dict[str, Any]) -> None:
    if not value:
        return
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _load_catalog(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    digest = value.pop("content_sha256", None)
    if type(digest) is not str or digest != strict_canonical_json_sha256(value):
        raise ValueError("target_route_catalog_digest_invalid")
    if value.get("schema_version") != "target_route_catalog.v1":
        raise ValueError("target_route_catalog_schema_invalid")
    if not isinstance(value.get("targets"), list):
        raise ValueError("target_route_catalog_targets_invalid")
    value["content_sha256"] = digest
    return value


if __name__ == "__main__":
    raise SystemExit(main())
