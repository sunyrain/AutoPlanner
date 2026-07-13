"""Revalidate advisory literature candidates with the current deterministic parser.

This migration utility never trusts an older exact-row or visual acceptance
flag.  Those artifacts only nominate candidate structures and labels; the
current parser must reconstruct every product heading and reactant from the
hash-bound source document before it emits a new registry binding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.deterministic_literature_registry import (  # noqa: E402
    PARSER_AUTHORITY_ID,
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.source_detail_chain_builder import (  # noqa: E402
    materialize_source_detail_step_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blackboard", type=Path, nargs="?")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    if bool(args.blackboard) == bool(args.candidate_manifest):
        raise SystemExit(
            "Provide exactly one blackboard positional path or --candidate-manifest"
        )
    if args.candidate_manifest:
        steps = candidate_steps_from_manifest(
            _read_object(args.candidate_manifest),
            source_ref=str(args.source_ref),
        )
    else:
        board = _read_object(args.blackboard)
        steps = candidate_steps_from_blackboard(
            board,
            source_ref=str(args.source_ref),
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = compile_deterministic_literature_step_registry(
        steps,
        registry_path=output_dir / "trusted_literature_step_registry.generated.json",
        audit_path=output_dir / "deterministic_literature_registry_audit.json",
        timeout_s=max(1.0, float(args.timeout_s)),
    )
    print(
        json.dumps(
            {
                "schema_version": "deterministic_literature_registry_replay.v1",
                "parser_authority_id": PARSER_AUTHORITY_ID,
                "candidate_step_count": len(steps),
                "approved_binding_count": int(
                    audit.get("approved_binding_count") or 0
                ),
                "rejected_step_count": int(audit.get("rejected_step_count") or 0),
                "registry_path": str(
                    output_dir
                    / "trusted_literature_step_registry.generated.json"
                ),
                "audit_path": str(
                    output_dir / "deterministic_literature_registry_audit.json"
                ),
                "semantics": {
                    "prior_candidate_acceptance_is_ignored": True,
                    "current_source_reconstruction_required": True,
                    "model_invocations": 0,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def candidate_steps_from_blackboard(
    blackboard: dict[str, Any],
    *,
    source_ref: str,
) -> list[dict[str, Any]]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    requested = str(source_ref or "").strip().casefold()
    visual_by_step: dict[str, dict[str, Any]] = {}
    for chain in evidence.get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        if str(chain.get("source_ref") or "").strip().casefold() != requested:
            continue
        for raw in chain.get("steps") or []:
            if not isinstance(raw, dict):
                continue
            step = dict(raw)
            step_id = str(step.get("step_id") or "")
            if step_id:
                visual_by_step[step_id] = step

    steps: list[dict[str, Any]] = []
    for raw in evidence.get("exact_rows") or []:
        if not isinstance(raw, dict):
            continue
        exact = dict(raw)
        if str(exact.get("source_ref") or "").strip().casefold() != requested:
            continue
        step_id = str(
            exact.get("step_id")
            or str(exact.get("row_id") or "").removeprefix(
                "source_detail_exact_step:"
            )
        )
        visual = visual_by_step.get(step_id, {})
        steps.append(
            {
                "schema_version": "deterministic_literature_candidate_step.v1",
                "step_id": step_id,
                "source_ref": str(exact.get("source_ref") or source_ref),
                "product_name": str(visual.get("product_label") or ""),
                "product_smiles": str(exact.get("product_smiles") or ""),
                "reactant_labels": [
                    str(item)
                    for item in visual.get("reactant_labels") or []
                    if str(item or "").strip()
                ],
                "reactant_smiles": [
                    str(item)
                    for item in exact.get("reactant_smiles") or []
                    if str(item or "").strip()
                ],
                "condition_candidate": dict(
                    exact.get("condition_candidate") or {}
                ),
                "source_evidence": [
                    dict(item)
                    for item in exact.get("source_evidence") or []
                    if isinstance(item, dict)
                ],
                "source_text_companions": [
                    dict(item)
                    for item in exact.get("source_text_companions") or []
                    if isinstance(item, dict)
                ],
                "candidate_provenance": {
                    "blackboard_schema_version": str(
                        blackboard.get("schema_version") or ""
                    ),
                    "prior_acceptance_ignored": True,
                },
            }
        )
    return steps


def candidate_steps_from_manifest(
    manifest: dict[str, Any],
    *,
    source_ref: str,
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "source_route_candidate_manifest.v1":
        raise SystemExit(
            "candidate manifest must use source_route_candidate_manifest.v1"
        )
    requested = str(source_ref or "").strip().casefold()
    source = next(
        (
            dict(row)
            for row in manifest.get("sources") or []
            if isinstance(row, dict)
            and str(row.get("source_ref") or "").strip().casefold()
            == requested
        ),
        {},
    )
    if not source:
        raise SystemExit(f"candidate manifest has no source {source_ref!r}")
    evidence_manifest = _resolve_repo_path(
        source.get("source_evidence_manifest")
    )
    companion = dict(source.get("source_text_companion") or {})
    if companion:
        companion["artifact_path"] = str(
            _resolve_repo_path(companion.get("artifact_path"))
        )
    out: list[dict[str, Any]] = []
    for raw in manifest.get("steps") or []:
        if not isinstance(raw, dict):
            continue
        step = dict(raw)
        if str(step.get("source_ref") or "").strip().casefold() != requested:
            continue
        page_number = int(step.pop("source_page_number", 0) or 0)
        evidence_refs = (
            [f"{evidence_manifest}#page={page_number}"]
            if page_number > 0
            else [str(evidence_manifest)]
        )
        step["source_evidence"] = materialize_source_detail_step_evidence(
            {
                "source_ref": str(source.get("source_ref") or source_ref),
                "evidence_refs": evidence_refs,
            }
        )
        step["source_text_companions"] = [companion] if companion else []
        step["candidate_provenance"] = {
            "manifest_schema_version": manifest["schema_version"],
            "prior_acceptance_ignored": True,
        }
        out.append(step)
    return out


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read blackboard {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("blackboard JSON must be an object")
    return payload


if __name__ == "__main__":
    main()
