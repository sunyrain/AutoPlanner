from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.eval.v4_blind_release_gate import (
    compile_v4_blind_release_gate,
)


SMILES = (
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "C1CCCC1",
    "c1ccoc1",
    "c1ccsc1",
    "C1CCNCC1",
    "C1COCCN1",
)


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _panel(root: Path, repository: Path, *, arm: str) -> Path:
    targets = {}
    artifacts = root / "artifacts"
    base_environment = "environment:fixed"
    snapshot = {
        "schema_version": "v4_blind_benchmark_snapshot.v1",
        "base_environment_sha256": base_environment,
        "manifest_sha256": "manifest:fixed",
        "selected_case_ids": [f"blind-{index:02d}" for index in range(1, 9)],
        "ablation": arm,
        "provider_snapshot": {
            "model": "model:fixed",
            "reasoning_effort": "low",
            "execution_profile": "standard",
            "worker_count": 1,
            "codex_cli": {"sha256": "codex:fixed"},
            "host_python": {"sha256": "python:fixed"},
            "chemenzy_python": {"sha256": "chemenzy:fixed"},
        },
        "knowledge": {
            "self_evo_library_sha256": "template:fixed",
            "inventory_snapshot_sha256": "inventory:fixed",
        },
    }
    snapshot["content_sha256"] = _digest(snapshot)
    snapshot_path = root / "snapshots" / "benchmark-snapshot.json"
    _write_json(snapshot_path, snapshot)
    for index, smiles in enumerate(SMILES, start=1):
        case_id = f"blind-{index:02d}"
        target_name = f"opaque {index:02d}"
        workbench = {
            "schema_version": "retrosynthesis_route_workbench.v1",
            "routes": {
                "route:one": {
                    "route_id": "route:one",
                    "route_family_id": "family:one",
                    "edge_ids": ["edge:one"],
                    "acceptance_profiles": {
                        "exploration_closed": True,
                        "reaction_validated": True,
                        "condition_complete": True,
                        "procurement_closed": True,
                        "process_ready": True,
                    },
                    "proof_vector": {"stock": "offer_verified"},
                },
                "route:two": {
                    "route_id": "route:two",
                    "route_family_id": "family:two",
                    "edge_ids": ["edge:two"],
                    "acceptance_profiles": {
                        "exploration_closed": True,
                        "reaction_validated": arm != "no-chemenzy",
                        "condition_complete": False,
                        "procurement_closed": False,
                        "process_ready": False,
                    },
                    "proof_vector": {"stock": "offer_verified"},
                },
            },
            "edges": {
                "edge:one": {
                    "edge_id": "edge:one",
                    "proof_vector": {
                        "conditions": "source_exact",
                        "exact_procedure_record_count": 1,
                    },
                },
                "edge:two": {
                    "edge_id": "edge:two",
                    "proof_vector": {
                        "conditions": "source_exact",
                        "exact_procedure_record_count": 1,
                    },
                },
            },
            "inspectors": {
                "edges": {
                    "edge:one": {"condition_status": "source_exact"},
                    "edge:two": {"condition_status": "source_exact"},
                }
            },
        }
        workbench["content_sha256"] = _digest(workbench)
        raw = (json.dumps(workbench, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        object_sha = hashlib.sha256(raw).hexdigest()
        object_path = f"objects/sha256/{object_sha[:2]}/{object_sha}"
        object_file = artifacts / object_path
        object_file.parent.mkdir(parents=True, exist_ok=True)
        object_file.write_bytes(raw)
        report = {
            "schema_version": "target_only_solve_report.v1",
            "run_id": case_id,
            "preflight": {
                "accepted": True,
                "repository_root": str(repository.resolve()),
                "case": {"case_id": case_id, "target_smiles": smiles},
                "semantics": {"target_name_smiles_and_inchikey_checked": True},
            },
            "workbench_ref": {
                "object_path": object_path,
                "sha256": object_sha,
            },
            "claim": {
                "accepted_under_configured_policy": True,
                "achieved_profile": "process_ready",
                "no_unqualified_complete_claim": True,
            },
            "current_disposition": {"reasons": []},
            "resource_envelope": {"within_budget": True},
            "gates": {"false_closure_claim_count": 0},
            "model_cost": {
                "model_invocations": 2,
                "input_tokens": 50_000,
                "output_tokens": 10_000,
                "wall_time_s": 100.0,
            },
            "stages": [{"stage": "chemenzy_guided_frontier", "elapsed_s": 10.0}],
        }
        report["content_sha256"] = _digest(report)
        report_path = root / "runs" / target_name / "target-only-solve-report.json"
        _write_json(report_path, report)
        supervisor_preflight = {
            "accepted": True,
            "semantics": {
                "target_synonym_needles_checked": True,
                "key_intermediate_needles_checked": True,
            },
        }
        supervisor_preflight["content_sha256"] = _digest(supervisor_preflight)
        receipt = {
            "case_id": case_id,
            "panel_snapshot_sha256": snapshot["content_sha256"],
            "base_environment_sha256": base_environment,
            "supervisor_preflight": supervisor_preflight,
        }
        receipt["content_sha256"] = _digest(receipt)
        targets[target_name] = {
            "status": "completed",
            "case_id": case_id,
            "report_path": str(report_path),
            "snapshot_receipt": receipt,
        }
    status = {
        "schema_version": "v4_blind_panel_status.v1",
        "ablation": arm,
        "model": "model:fixed",
        "reasoning_effort": "low",
        "execution_profile": "standard",
        "worker_count": 1,
        "target_count": len(targets),
        "complete": True,
        "completed_count": len(targets),
        "frozen_snapshot": {
            "path": str(snapshot_path),
            "content_sha256": snapshot["content_sha256"],
            "base_environment_sha256": base_environment,
            "manifest_sha256": "manifest:fixed",
            "self_evo_library_sha256": "template:fixed",
            "inventory_snapshot_sha256": "inventory:fixed",
        },
        "targets": targets,
    }
    path = root / "panel-status.json"
    _write_json(path, status)
    return path


def test_release_gate_passes_only_with_all_profiles_snapshots_and_ablations(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    baseline = _panel(tmp_path / "baseline", repository, arm="baseline")
    ablations = [
        _panel(tmp_path / arm, repository, arm=arm)
        for arm in ("no-chemenzy", "no-self-evo", "no-replan")
    ]

    report = compile_v4_blind_release_gate(
        baseline,
        ablation_status_paths=ablations,
        repository_root=repository,
        output_dir=tmp_path / "release",
    )

    assert report["accepted"] is True
    assert report["target_count"] == 8
    assert report["coverage"]["two_distinct_validated_case_count"] == 8
    assert report["gates"]["no_false_closure_or_authority_laundering"]["passed"] is True
    ablation_gate = report["gates"]["three_required_ablations"]["actual"]
    assert ablation_gate["reaction_validated_route_delta_vs_disabled"] == {
        "no-chemenzy": 8,
        "no-replan": 0,
        "no-self-evo": 0,
    }
    assert (tmp_path / "release" / "summary.json").is_file()
    assert "发布门通过" in (tmp_path / "release" / "index.html").read_text(encoding="utf-8")


def test_release_gate_keeps_baseline_but_fails_without_ablations(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    baseline = _panel(tmp_path / "baseline", repository, arm="baseline")

    report = compile_v4_blind_release_gate(
        baseline,
        repository_root=repository,
    )

    assert report["accepted"] is False
    assert report["gates"]["three_required_ablations"]["passed"] is False
    assert len(report["cases"]) == 8


def test_release_gate_rejects_a_tampered_frozen_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    baseline = _panel(tmp_path / "baseline", repository, arm="baseline")
    status = json.loads(baseline.read_text(encoding="utf-8"))
    snapshot_path = Path(status["frozen_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["provider_snapshot"]["codex_cli"]["sha256"] = "tampered"
    _write_json(snapshot_path, snapshot)

    report = compile_v4_blind_release_gate(baseline, repository_root=repository)

    assert report["accepted"] is False
    frozen_gate = report["gates"]["frozen_provider_template_inventory_snapshot"]
    assert frozen_gate["passed"] is False
    assert frozen_gate["actual"]["panel_snapshot_present_and_digest_valid"] is False
    assert all(row["snapshot_receipt_valid"] is False for row in report["cases"])


def test_release_gate_recovers_case_id_from_digest_bound_receipt(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    baseline = _panel(tmp_path / "baseline", repository, arm="baseline")
    status = json.loads(baseline.read_text(encoding="utf-8"))
    for row in status["targets"].values():
        row.pop("case_id")
    _write_json(baseline, status)

    report = compile_v4_blind_release_gate(baseline, repository_root=repository)

    frozen_gate = report["gates"]["frozen_provider_template_inventory_snapshot"]
    assert frozen_gate["actual"]["panel_snapshot_present_and_digest_valid"] is True
    assert all(row["snapshot_receipt_valid"] is True for row in report["cases"])
