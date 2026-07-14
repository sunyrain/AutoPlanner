from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import build_source_channel_showcase as source_showcase
from scripts import build_v4_blind_showcase as blind_showcase


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_validation_fork_showcase_binds_frozen_html_and_zero_model_receipt(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "panel"
    run_dir = panel / "runs" / "simvastatin-fork"
    paper = run_dir / "paper.html"
    paper.parent.mkdir(parents=True)
    paper.write_text("<html>hash-bound source</html>", encoding="utf-8")
    paper_sha = hashlib.sha256(paper.read_bytes()).hexdigest()
    receipt = panel / "artifacts" / "objects" / "sha256" / "aa" / "receipt"
    _write_json(
        receipt,
        {"child_elapsed_s": {"connector_1": 8.8, "connector_2": 9.8}},
    )
    report = run_dir / "target-validation-fork-report.json"
    _write_json(
        report,
        {
            "run_id": "simvastatin-fork",
            "model_cost": {"model_invocations": 0, "visual_invocations": 0},
            "gates": {"B3_exact_multi_source": False},
            "stages": [
                {
                    "stage": "evidence_acquisition",
                    "detail": {
                        "receipt_ref": {
                            "object_path": "objects/sha256/aa/receipt"
                        },
                        "discovery": {
                            "sources": [
                                {
                                    "source_kind": "patent",
                                    "procedure_inventory": [{"name": "patent"}],
                                },
                                {
                                    "source_kind": "paper_si",
                                    "pmcid": "PMC1855665",
                                    "acquisition_method": (
                                        "pmc_repository_fulltext_html"
                                    ),
                                    "acquisition_receipt": {
                                        "access_class": "free_repository_fulltext"
                                    },
                                    "fulltext_html_path": str(paper),
                                    "source_fulltext_sha256": paper_sha,
                                    "procedure_inventory": [
                                        {
                                            "name": "Whole-cell biocatalysis",
                                            "procedure_excerpt": "15 mM substrate",
                                        }
                                    ],
                                },
                            ]
                        },
                    },
                }
            ],
        },
    )

    payload, asset = source_showcase._validation_fork_payload(
        report,
        output=tmp_path / "showcase",
        artifact_store_root=panel / "artifacts",
    )

    assert asset.read_bytes() == paper.read_bytes()
    assert payload["source_count"] == 2
    assert payload["paper_procedure_count"] == 1
    assert payload["patent_procedure_count"] == 1
    assert payload["model_invocations"] == 0
    assert payload["connector_elapsed_s"]["connector_2"] == 9.8
    assert payload["B3_exact_multi_source"] is False


def test_blind_showcase_uses_current_routes_and_validated_replacements() -> None:
    rows = [
        {
            "status": "completed",
            "report_path": "report.json",
            "model_cost": {"model_invocations": 1},
            "chemenzy": {"provider_calls": 2},
            "time_to_first_route_s": 60,
            "full_pass_s": 90,
            "gates": {"B1": True, "B3": False, "B5": True},
            "workbench": {
                "route_count": 5,
                "validated_replacement_count": 3,
            },
        }
    ]

    summary = blind_showcase._summary({"target_count": 1}, rows)

    assert summary["route_count"] == 5
    assert summary["validated_replacement_count"] == 3
    assert summary["gate_counts"]["B1"] == 1
    assert blind_showcase._origin_label("chemenzy") == "ChemEnzy 局部展开"
