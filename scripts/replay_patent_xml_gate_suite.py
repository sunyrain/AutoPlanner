#!/usr/bin/env python3
"""Run the three-case official EPO structured-procedure release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
    build_deterministic_literature_resolvers,
)
from cascade_planner.research.real_patent_procedure_gate import (
    SNAPSHOT_SCHEMA,
    canonical_smiles,
    compile_patent_procedure_gate_suite,
    content_digest,
    load_patent_procedure_gate_config,
    replay_patent_procedure_case,
    validate_resolver_snapshot,
)
from cascade_planner.interfaces.patent_source_discovery import fetch_bounded_bytes


DEFAULT_CONFIG = ROOT / "benchmarks" / "real_patent_procedure_gate_cases.v1.json"
DEFAULT_OUTPUT = (
    ROOT / "results" / "shared" / "patent_procedure_gate_20260717" / "patent-procedure-gate"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch each official EPO ST.36 XML once, snapshot independent name "
            "resolution, then replay every exact procedure twice offline."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require already cached official XML and resolver snapshots",
    )
    args = parser.parse_args()

    config = load_patent_procedure_gate_config(args.config.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    acceptances: list[dict[str, Any]] = []
    for raw_case in config.get("cases") or []:
        case = dict(raw_case)
        case_dir = output_dir / str(case["case_id"])
        content = _source_bytes(case, case_dir, offline=args.offline)
        snapshot = _resolver_snapshot(case, case_dir, offline=args.offline)
        acceptance = replay_patent_procedure_case(
            case,
            source_content=content,
            resolver_snapshot=snapshot,
            output_dir=case_dir,
        )
        acceptances.append(acceptance)
        print(f"{case['case_id']}: {'PASS' if acceptance.get('accepted') else 'FAIL'}")
    summary = compile_patent_procedure_gate_suite(
        config,
        acceptances,
        output_dir=output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("accepted") is True else 2


def _source_bytes(case: Mapping[str, Any], case_dir: Path, *, offline: bool) -> bytes:
    manifest_path = case_dir / "source" / "primary-patent-xml-materialization.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        path = Path(str(manifest.get("artifact_path") or ""))
        expected = str(manifest.get("artifact_sha256") or "")
        if path.is_file():
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() == expected:
                return content
    if offline:
        raise RuntimeError(f"offline_epo_xml_cache_missing_or_invalid:{case.get('case_id')}")
    return fetch_bounded_bytes(str(case.get("source_url") or ""), 30.0, 20_000_000)


def _resolver_snapshot(case: Mapping[str, Any], case_dir: Path, *, offline: bool) -> dict[str, Any]:
    path = case_dir / "resolver-snapshot.json"
    if path.is_file():
        try:
            return validate_resolver_snapshot(case, _read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            if offline:
                raise RuntimeError(f"offline_resolver_snapshot_invalid:{case.get('case_id')}")
    if offline:
        raise RuntimeError(f"offline_resolver_snapshot_missing:{case.get('case_id')}")
    resolve_structure, _resolve_names = build_deterministic_literature_resolvers(timeout_s=30.0)
    structures = {
        str(name): canonical_smiles(resolve_structure(str(name)))
        for name in case.get("source_structure_names") or []
    }
    candidate_names: dict[str, list[str]] = {}
    for name, smiles in structures.items():
        if smiles:
            candidate_names.setdefault(smiles, []).append(name)
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA,
        "authority_id": PARSER_AUTHORITY_ID,
        "structures": structures,
        "candidate_names": {
            smiles: sorted(set(names)) for smiles, names in sorted(candidate_names.items())
        },
        "semantics": {
            "snapshot_is_content_addressed": True,
            "snapshot_replay_uses_no_network": True,
            "snapshot_does_not_replace_source_text": True,
            "source_names_were_resolved_by_independent_parser": True,
        },
    }
    snapshot["content_sha256"] = content_digest(snapshot)
    validate_resolver_snapshot(case, snapshot)
    _write_json_atomic(path, snapshot)
    return snapshot


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"json_object_required:{path}")
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
