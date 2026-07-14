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
        {
            "child_elapsed_s": {"connector_1": 8.8, "connector_2": 9.8},
            "child_receipts": [
                {
                    "provider_id": "autoplanner.builtin_patent_evidence",
                    "audits": [
                        {
                            "source_route_exact_row_count": 1,
                            "source_byte_cache": {
                                "pdf": {"cache_hit": True}
                            },
                            "source_route_observation": {
                                "proposal_count": 1,
                                "proposals": [
                                    {
                                        "origin_kind": "literature_source_route",
                                        "product_name": (
                                            "synthesis of ethyl acetate from "
                                            "ethanol and acetic acid"
                                        ),
                                        "reactant_names": [
                                            "ethanol",
                                            "acetic acid",
                                        ],
                                        "condition_candidate": {
                                            "temperature": "25 °C"
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        },
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
                        "validation": {"accepted_validation_count": 1},
                        "visual_evidence": {
                            "observation": {"candidate_step_count": 2},
                            "materialization": {"proposal_count": 2},
                            "validation": {
                                "accepted_validation_count": 1,
                                "rejected_validation_count": 1,
                            },
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
            "self_evolution": {
                "learning_stages": [
                    {"template_count": 1, "generation": 2}
                ]
            },
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
    assert payload["source_route_proposal_count"] == 1
    assert payload["source_route_host_accepted_count"] == 1
    assert payload["source_route_exact_row_count"] == 1
    assert payload["visual_candidate_count"] == 2
    assert payload["visual_materialized_count"] == 2
    assert payload["visual_host_accepted_count"] == 1
    assert payload["visual_host_rejected_count"] == 1
    assert payload["patent_source_cache_hit"] is True
    assert payload["self_evo_template_count"] == 1
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


def test_blind_showcase_newer_target_panel_replaces_only_matching_target() -> None:
    baseline = [
        {"target_name": "lovastatin", "workbench": {"route_count": 0}},
        {"target_name": "simvastatin", "workbench": {"route_count": 5}},
    ]
    override = [
        {
            "target_name": "lovastatin",
            "artifact_role": "latest_independent_blind_rerun",
            "workbench": {"route_count": 4},
        }
    ]

    merged = blind_showcase._merge_target_rows(baseline, override)

    assert [row["target_name"] for row in merged] == ["lovastatin", "simvastatin"]
    assert merged[0]["workbench"]["route_count"] == 4
    assert merged[0]["artifact_role"] == "latest_independent_blind_rerun"
    assert merged[1]["workbench"]["route_count"] == 5


def test_blind_showcase_validation_overlay_keeps_original_codex_cost() -> None:
    baseline = [
        {
            "target_name": "simvastatin",
            "run_id": "blind-source",
            "model_cost": {"model_invocations": 1, "input_tokens": 20_000},
            "time_to_first_route_s": 12.0,
            "full_pass_s": 80.0,
            "chemenzy": {"provider_calls": 1},
            "gates": {"B2": False},
            "evidence": {"exact_rows": 0},
        }
    ]
    validation = [
        {
            "target_name": "simvastatin",
            "run_id": "validation-fork",
            "model_cost": {"model_invocations": 0},
            "time_to_first_route_s": 0.0,
            "full_pass_s": 5.0,
            "chemenzy": {"provider_calls": 0},
            "gates": {"B2": True},
            "evidence": {"exact_rows": 1},
        }
    ]

    merged = blind_showcase._merge_validation_rows(baseline, validation)[0]

    assert merged["run_id"] == "validation-fork"
    assert merged["source_blind_run_id"] == "blind-source"
    assert merged["model_cost"]["model_invocations"] == 1
    assert merged["validation_model_cost"]["model_invocations"] == 0
    assert merged["time_to_first_route_s"] == 12.0
    assert merged["gates"]["B2"] is True
    assert merged["evidence"]["exact_rows"] == 1
    assert merged["artifact_role"] == "latest_evidence_validation_fork"
