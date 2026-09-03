from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from rdkit import Chem

from cascade_planner.application.target_route_readiness import (
    TARGET_ROUTE_READINESS_SCHEMA,
    TargetRouteReadinessError,
    compile_target_route_readiness,
    current_replay_attestation_from_receipt,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
FLUVASTATIN_RECEIPT = ROOT / "benchmarks" / "fluvastatin_current_canonical_replay_receipt.v1.json"
FLUVASTATIN_WORKBENCH = ROOT / "benchmarks" / "fluvastatin_current_route_workbench.v1.json"
STATIN_CATALOG = ROOT / "benchmarks" / "statin_target_catalog.v1.json"
STATIN_FIRST_WAVE = ROOT / "benchmarks" / "statins_v4_blind.json"
STATIN_EXTENDED_WAVE = ROOT / "benchmarks" / "statins_v4_extended.json"


def _digest(value: dict) -> dict:
    row = copy.deepcopy(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _migration_audit() -> dict:
    rows = [
        _workbench("bufotalin", steps=20, conditions=15, sources=1, proofs={"0": 5, "1": 15}),
        _workbench("atorvastatin", steps=11, conditions=0, sources=0, proofs={"0": 11}),
        _workbench("fluvastatin", steps=5, conditions=0, sources=0, proofs={"1": 7, "2": 4}),
        {
            "target_name": "glenvastatin",
            "migration_state": "empty_graph",
            "source_diagnostics": {},
        },
    ]
    return _digest(
        {
            "schema_version": "candidate_program_migration_audit.v1",
            "workbenches": rows,
        }
    )


def _workbench(
    target_name: str,
    *,
    steps: int,
    conditions: int,
    sources: int,
    proofs: dict[str, int],
) -> dict:
    return {
        "target_name": target_name,
        "migration_state": "projection_ready",
        "source_diagnostics": {
            "max_route_steps": steps,
            "complete_route_count": 1,
            "condition_observation_edge_count": conditions,
            "reported_source_ref_count": sources,
            "proof_level_counts": proofs,
            "portfolio_accepted_claim": False,
        },
    }


def _catalog() -> list[dict]:
    return [
        {"target_name": name, "display_name": name.title(), "aliases": []}
        for name in ("bufotalin", "atorvastatin", "fluvastatin", "glenvastatin", "missing")
    ]


def test_readiness_keeps_route_evidence_conditions_and_authority_separate() -> None:
    receipt = json.loads(FLUVASTATIN_RECEIPT.read_text(encoding="utf-8"))
    attestation = current_replay_attestation_from_receipt(
        receipt,
        source_ref="benchmarks/fluvastatin_current_canonical_replay_receipt.v1.json",
    )

    report = compile_target_route_readiness(
        _catalog(),
        _migration_audit(),
        authority_attestations=[attestation],
    )

    digest = report.pop("content_sha256")
    assert digest == strict_canonical_json_sha256(report)
    assert report["schema_version"] == TARGET_ROUTE_READINESS_SCHEMA
    rows = {row["target_name"]: row for row in report["targets"]}
    assert rows["bufotalin"]["readiness"] == "candidate_long_route"
    assert rows["bufotalin"]["confidence"] == "low"
    assert rows["bufotalin"]["observations"]["condition_observation_edge_count"] == 15
    assert "CANDIDATE_ONLY_NO_CURRENT_REPLAY" in rows["bufotalin"]["warning_codes"]
    assert rows["atorvastatin"]["observations"]["long_route_observed"] is True
    assert "NO_CONDITION_OBSERVATIONS" in rows["atorvastatin"]["warning_codes"]
    assert rows["fluvastatin"]["readiness"] == "current_canonical_unaccepted"
    assert rows["fluvastatin"]["authority"]["current_canonical_replay_attested"] is True
    assert "CURRENT_REPLAY_NOT_ROUTE_ACCEPTED" in rows["fluvastatin"]["warning_codes"]
    assert rows["glenvastatin"]["readiness"] == "empty_workbench_only"
    assert rows["missing"]["readiness"] == "not_observed"
    assert report["summary"] == {
        "readiness_counts": {
            "candidate_long_route": 2,
            "current_canonical_unaccepted": 1,
            "empty_workbench_only": 1,
            "not_observed": 1,
        },
        "route_observed": 3,
        "long_route_observed": 2,
        "conditioned_route_observed": 1,
        "reported_source_observed": 1,
        "current_canonical_replay_attested": 1,
        "route_acceptance_attested": 0,
    }


def test_unverified_payload_cannot_elevate_route_authority() -> None:
    receipt = json.loads(FLUVASTATIN_RECEIPT.read_text(encoding="utf-8"))
    receipt["workbench"]["accepted"] = True

    with pytest.raises(TargetRouteReadinessError, match="receipt_digest_invalid"):
        current_replay_attestation_from_receipt(receipt, source_ref="tampered.json")

    fake = _digest(
        {
            "schema_version": "target_route_authority_attestation.v1",
            "target_name": "fluvastatin",
            "source_ref": "client.json",
            "authority_level": "client_declared",
            "route_accepted": True,
            "condition_complete": True,
            "literature_grounded": True,
        }
    )
    with pytest.raises(TargetRouteReadinessError, match="authority_level_invalid"):
        compile_target_route_readiness(
            _catalog(), _migration_audit(), authority_attestations=[fake]
        )


def test_readiness_policy_uses_exact_integer_type() -> None:
    with pytest.raises(TargetRouteReadinessError, match="minimum_long_route_steps_invalid"):
        compile_target_route_readiness(  # type: ignore[arg-type]
            _catalog(), _migration_audit(), minimum_long_route_steps=True
        )


def test_candidate_audit_cli_emits_catalog_readiness(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_candidate_workbenches.py"),
            str(FLUVASTATIN_WORKBENCH),
            "--catalog",
            str(STATIN_CATALOG),
            "--current-replay-receipt",
            str(FLUVASTATIN_RECEIPT),
            "--readiness-output",
            str(readiness_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    fluvastatin = next(
        row for row in readiness["targets"] if row["target_name"] == "fluvastatin"
    )
    assert summary["readiness_sha256"] == readiness["content_sha256"]
    assert readiness["target_count"] == 12
    assert fluvastatin["readiness"] == "current_canonical_unaccepted"
    assert fluvastatin["observations"]["max_route_steps"] == 5
    assert fluvastatin["observations"]["workbench_source_refs"] == [
        "benchmarks/fluvastatin_current_route_workbench.v1.json"
    ]


def test_statin_catalog_is_digest_bound_and_matches_both_blind_manifests() -> None:
    catalog = json.loads(STATIN_CATALOG.read_text(encoding="utf-8"))
    digest = catalog.pop("content_sha256")
    first = json.loads(STATIN_FIRST_WAVE.read_text(encoding="utf-8"))["cases"]
    extended = json.loads(STATIN_EXTENDED_WAVE.read_text(encoding="utf-8"))["cases"]
    cases = {row["target_name"]: row["target_smiles"] for row in [*first, *extended]}

    assert digest == strict_canonical_json_sha256(catalog)
    assert catalog["schema_version"] == "target_route_catalog.v1"
    assert len(catalog["targets"]) == 12
    assert len({row["target_name"] for row in catalog["targets"]}) == 12
    assert {row["target_name"]: row["target_smiles"] for row in catalog["targets"]} == cases
    assert all(Chem.MolFromSmiles(row["target_smiles"]) for row in catalog["targets"])
