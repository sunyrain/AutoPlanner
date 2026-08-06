from __future__ import annotations

from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import (
    _merge_codex_team_source_hints,
)


def test_codex_team_citations_enter_metadata_source_lifecycle_without_authority() -> None:
    board = {
        "case_id": "case",
        "literature_evidence": {
            "source_candidates": [],
            "source_refs": [],
            "exact_rows": [],
        },
    }
    report = {
        "route_consensus": {
            "schema_version": "route_consensus.v1",
            "proposals": [
                {
                    "consensus_id": "consensus:late-step",
                    "reaction_family": "amide dehydration",
                    "source_refs": [
                        "patent_publication:WO2021250648A1;url:https://patents.google.com/patent/WO2021250648A1/en"
                    ],
                    "evidence_refs": [
                        "patent_publication:WO2021250648A1;url:https://patents.google.com/patent/WO2021250648A1/en;lines:2164-2171"
                    ],
                },
                {
                    "consensus_id": "consensus:paper",
                    "reaction_family": "late-stage coupling",
                    "source_refs": ["doi:10.1126/science.abl4784"],
                    "evidence_refs": [],
                },
            ],
        }
    }

    merged = _merge_codex_team_source_hints(board, report=report)
    evidence = merged["literature_evidence"]

    assert evidence["team_source_hint_count"] == 2
    assert evidence["team_source_hints_are_metadata_only"] is True
    assert len(evidence["source_candidates"]) == 2
    assert not evidence["exact_rows"]
    assert {
        row["source_ref"] for row in evidence["source_candidates"]
    } == {
        "patent:WO2021250648A1",
        "doi:10.1126/science.abl4784",
    }
    assert all(
        row["not_exact_literature_evidence"] is True
        and row["access_status"] == "metadata_pointer_only"
        and row["no_solved_claim"] is True
        for row in evidence["source_candidates"]
    )
    assert evidence["source_identity_summary"][
        "independent_source_group_count"
    ] == 2
    assert {
        row["independent_source_group"]
        for row in evidence["source_lifecycle"]
    } == {
        "patent:WO2021250648A1",
        "doi:10.1126/science.abl4784",
    }
