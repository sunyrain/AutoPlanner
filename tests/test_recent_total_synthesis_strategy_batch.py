from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRATEGY_BATCH = _load_script(
    "recent_total_synthesis_strategy_batch",
    "scripts/run_recent_total_synthesis_strategy_batch.py",
)
STRATEGY_EVALUATION = _load_script(
    "recent_total_synthesis_strategy_evaluation",
    "scripts/run_recent_total_synthesis_strategy_evaluation.py",
)


def test_candidate_coverage_exposes_only_blind_planner_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "target_slot_id": "target-1",
            "target_name": "must not leak",
            "doi": "10.1/must-not-leak",
            "review_flags": [],
            "candidates": [
                {
                    "rdkit_validation": {
                        "status": "roundtrip_valid",
                        "canonical_isomeric_smiles": "CCO",
                    }
                }
            ],
        },
        {
            "target_slot_id": "target-2",
            "review_flags": ["identity_conflict"],
            "candidates": [
                {
                    "rdkit_validation": {
                        "status": "roundtrip_valid",
                        "canonical_isomeric_smiles": "CCN",
                    }
                }
            ],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    targets = STRATEGY_BATCH._candidate_coverage_targets(path)

    assert targets == [
        {
            "target_slot_id": "target-1",
            "target_smiles": "CCO",
            "input_status": "unverified_unique_structure_candidate",
            "formal_benchmark_eligible": False,
        }
    ]


def test_strategy_result_digest_binds_both_prompt_contracts() -> None:
    target = {"target_slot_id": "target-1", "target_smiles": "CCO"}
    common = {
        "target": target,
        "model": "gpt-5.6-sol",
        "effort": "high",
        "critic_template_prompt": "critic-v1",
    }

    first = STRATEGY_BATCH._target_digest(
        **common,
        generator_prompt="generator-v1",
    )
    generator_changed = STRATEGY_BATCH._target_digest(
        **common,
        generator_prompt="generator-v2",
    )
    critic_changed = STRATEGY_BATCH._target_digest(
        **{**common, "critic_template_prompt": "critic-v2"},
        generator_prompt="generator-v1",
    )

    assert len({first, generator_changed, critic_changed}) == 3


def test_evaluator_derives_match_and_rejects_invented_evidence_refs() -> None:
    payload = {
        "schema_version": "literature_strategy_match_report.v1",
        "target_slot_id": "target-1",
        "comparability": "comparable",
        "card_assessments": [
            {"card_index": 1, "match_level": "none"},
            {"card_index": 2, "match_level": "partial"},
            {"card_index": 3, "match_level": "exact"},
        ],
        "evidence_locator_refs": ["passage-allowed"],
    }

    assert STRATEGY_EVALUATION._valid_evaluation_payload(
        payload,
        target_slot_id="target-1",
        allowed_evidence_refs={"passage-allowed"},
    )
    assert STRATEGY_EVALUATION._derived_match(payload) == ("exact", 3)

    payload["evidence_locator_refs"] = ["passage-invented"]
    assert not STRATEGY_EVALUATION._valid_evaluation_payload(
        payload,
        target_slot_id="target-1",
        allowed_evidence_refs={"passage-allowed"},
    )


def test_evaluator_rates_exclude_non_comparable_targets() -> None:
    summary = STRATEGY_EVALUATION._aggregate(
        [
            {
                "status": "evaluated",
                "overall_match": "exact",
                "paper_strategy_classes": ["cycloaddition"],
                "paper_id": "paper-1",
                "best_card_index": 1,
                "evaluation": {
                    "card_assessments": [
                        {
                            "card_index": 1,
                            "missing_or_conflicting_elements": [],
                        }
                    ]
                },
            },
            {
                "status": "evaluated",
                "overall_match": "none",
                "paper_strategy_classes": ["cycloaddition"],
                "paper_id": "paper-1",
            },
            {
                "status": "non_comparable_no_passages",
                "overall_match": "non_comparable",
                "paper_strategy_classes": ["cycloaddition"],
                "paper_id": "paper-2",
            },
        ]
    )

    assert summary["target_count"] == 3
    assert summary["comparable_target_count"] == 2
    assert summary["exact_match_rate_among_comparable"] == 0.5
    assert summary["gap_free_exact_match_count"] == 1
    assert summary["at_least_partial_match_rate_among_comparable"] == 0.5
    cycloaddition = summary["strategy_class_strata"]["cycloaddition"]
    assert cycloaddition["target_count"] == 3
    assert cycloaddition["comparable_target_count"] == 2
    assert cycloaddition["non_comparable_target_count"] == 1
    assert cycloaddition["at_least_partial_rate"] == 0.5
    paper_sensitivity = summary["paper_cluster_sensitivity"]
    assert paper_sensitivity["paper_count"] == 2
    assert paper_sensitivity["comparable_paper_count"] == 1
    assert paper_sensitivity["match_counts"] == {
        "exact": 1,
        "non_comparable": 1,
    }
    assert summary["claim_boundary"]["formal_benchmark_denominator"] == 0
