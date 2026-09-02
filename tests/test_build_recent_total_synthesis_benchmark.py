from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_recent_total_synthesis_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("recent_total_synthesis_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _row(**overrides):
    row = {
        "doi": "10.1021/acs.orglett.6c00001",
        "title": "Total Synthesis of Example A",
        "journal": "Organic Letters",
        "abstract": "",
    }
    row.update(overrides)
    return row


def test_repository_formal_synthesis_is_not_a_chemistry_candidate() -> None:
    status, _ = builder.automated_screen(
        _row(
            doi="10.5281/zenodo.123456",
            title="A Formal Synthesis of Algorithmic Geometry",
            journal="Zenodo",
        )
    )
    assert status == "exclude_repository_or_supplement"


def test_total_synthesis_in_primary_chemistry_journal_is_high_priority() -> None:
    status, _ = builder.automated_screen(_row())
    assert status == "high_priority_primary_candidate"


def test_plural_total_syntheses_is_high_priority() -> None:
    status, _ = builder.automated_screen(
        _row(title="Divergent Total Syntheses of Example Alkaloids")
    )
    assert status == "high_priority_primary_candidate"


def test_nonstandard_completed_natural_product_route_is_omission_candidate() -> None:
    status, _ = builder.automated_screen(
        _row(title="Synthesis and Stereochemical Revision of Natural Product Example B")
    )
    assert status == "high_priority_omission_candidate"


def test_formal_total_synthesis_is_conditional_not_primary() -> None:
    status, _ = builder.automated_screen(
        _row(title="A Concise Formal Total Synthesis of Example A")
    )
    assert status == "scope_review_formal_synthesis"


def test_plural_peptides_are_noncore_scope_review() -> None:
    status, _ = builder.automated_screen(
        _row(title="Total Synthesis of Two Antimicrobial Peptides")
    )
    assert status == "scope_review_noncore"


def test_route_improvement_is_not_a_new_primary_candidate() -> None:
    status, _ = builder.automated_screen(_row(title="An Improved Total Synthesis of Example A"))
    assert status == "scope_review_route_improvement"


def test_drug_or_metabolite_title_is_a_control_scope_review() -> None:
    status, _ = builder.automated_screen(_row(title="Total Synthesis of a Major Human Metabolite"))
    assert status == "scope_review_control_target"


def test_review_disclosed_in_abstract_is_excluded() -> None:
    status, _ = builder.automated_screen(
        _row(
            title="The Disorazole Family: Structures and Total Synthesis",
            abstract="This review provides a comprehensive analysis.",
        )
    )
    assert status == "exclude_review"


def test_peptide_disclosed_in_abstract_is_noncore() -> None:
    status, _ = builder.automated_screen(
        _row(
            title="Total Synthesis of Example A",
            abstract="Example A is a cyclic tetrapeptide.",
        )
    )
    assert status == "scope_review_noncore"


def test_method_platform_with_total_synthesis_application_is_boundary_scope() -> None:
    status, _ = builder.automated_screen(
        _row(title=("Catalytic Rearrangement: Application to the Total Synthesis of Example A"))
    )
    assert status == "scope_review_method_application"


def test_general_divergent_method_is_not_promoted_without_np_context() -> None:
    status, _ = builder.automated_screen(
        _row(
            title="Divergent Synthesis of Functionalized Aminonaphthalenes",
            abstract="A general substrate-scope study.",
        )
    )
    assert status == "manual_title_review"


def test_english_angewandte_record_is_preferred_within_title_family() -> None:
    raw = [
        {
            "provider": "crossref",
            "provider_query": "total_synthesis",
            "doi": "10.1002/ange.1234",
            "title": "Total Synthesis of Example A",
            "journal": "Angewandte Chemie",
            "publication_date": "2026-06-01",
        },
        {
            "provider": "crossref",
            "provider_query": "total_synthesis",
            "doi": "10.1002/anie.1234",
            "title": "Total Synthesis of Example A",
            "journal": "Angewandte Chemie International Edition",
            "publication_date": "2026-06-01",
        },
    ]
    rows = builder.merge_records(raw)
    preferred = [row for row in rows if row["preferred_family_record"]]
    assert [row["doi"] for row in preferred] == ["10.1002/anie.1234"]


def test_target_phrase_extraction_stops_before_route_description() -> None:
    assert (
        builder.title_target_phrase(
            "Asymmetric Total Synthesis of Example A via Radical Cyclization"
        )
        == "Example A"
    )


def test_openalex_inverted_abstract_is_reconstructed_in_word_order() -> None:
    assert (
        builder.openalex_abstract({"total": [2], "We": [0], "report": [1], "synthesis.": [3]})
        == "We report total synthesis."
    )


def _queue_row(**overrides):
    row = {
        "paper_id": "paper-example",
        "article_family_id": "family-example",
        "doi": "10.1021/example",
        "title": "Total Synthesis of Example A",
        "journal": "Organic Letters",
        "publication_date": "2025-07-01",
        "first_author": "Example A.",
        "source_url": "https://doi.org/10.1021/example",
        "providers": ["crossref"],
        "open_access": False,
        "repository_fulltext": False,
        "fulltext_link_count": 1,
        "abstract": "",
        "title_target_phrase": "Example A",
        "preferred_family_record": True,
        "after_strict_model_cutoff": False,
        "curation_status": "admitted_metadata_primary",
        "automated_status": "high_priority_primary_candidate",
    }
    row.update(overrides)
    return row


def test_admitted_primary_before_strict_cutoff_still_gets_source_task() -> None:
    queue = builder.build_paper_review_queue([_queue_row()])
    assert len(queue) == 1
    assert queue[0]["review_tier"] == "P0_source_extraction"


def test_explicit_abstract_completion_cue_prioritizes_but_does_not_admit() -> None:
    queue = builder.build_paper_review_queue(
        [
            _queue_row(
                after_strict_model_cutoff=True,
                curation_status="unreviewed",
                abstract="We report the first total synthesis of Example A.",
            )
        ]
    )
    assert queue[0]["first_pass_scope_status"] == ("likely_completed_route_needs_dual_review")
    assert queue[0]["automated_screening_is_admission"] is False


def test_unreviewed_candidate_before_strict_cutoff_is_not_in_novelty_queue() -> None:
    queue = builder.build_paper_review_queue([_queue_row(curation_status="unreviewed")])
    assert queue == []


def test_abstract_only_boundary_signal_does_not_promote_fuzzy_discovery_hit() -> None:
    queue = builder.build_paper_review_queue(
        [
            _queue_row(
                title="Metabolomics of Marine Peptides",
                abstract="A synthesis workflow is discussed.",
                after_strict_model_cutoff=True,
                curation_status="unreviewed",
                automated_status="scope_review_noncore",
            )
        ]
    )
    assert queue == []


def test_source_receipts_distinguish_supplemental_and_complete_queries(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "crossref-total_synthesis.json").write_text(
        json.dumps({"message": {"items": [{"DOI": "10.1/a"}], "total-results": 8}}),
        encoding="utf-8",
    )
    for page, count in ((0, 2), (1, 1)):
        (cache / f"openalex-total_synthesis-{page:03d}.json").write_text(
            json.dumps({"results": [{}] * count, "meta": {"count": 3}}),
            encoding="utf-8",
        )
    receipts = builder.build_cache_receipts(cache, tmp_path)
    crossref = next(row for row in receipts if row["provider"] == "crossref")
    openalex = next(row for row in receipts if row["provider"] == "openalex")
    assert crossref["retrieval_mode"] == "top_k_relevance_sample"
    assert crossref["retrieval_complete"] is False
    assert openalex["retrieval_mode"] == "cursor_paginated_enumeration"
    assert openalex["retrieval_complete"] is True
    assert openalex["query_returned_record_count"] == 3


def _human_review_target() -> dict[str, object]:
    return {
        "target_slot_id": "target-slot-reviewed",
        "paper_id": "paper-reviewed",
        "doi": "10.1021/reviewed",
        "slot_class": "primary",
        "target_smiles": "",
        "structure_status": "pending_source_concordant_structure",
        "route_evidence_status": "pending_fulltext_and_si_extraction",
        "runnable": False,
    }


def _source_binding(tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source-bound review fixture")
    return {
        "source_artifact_path": source.name,
        "source_artifact_sha256": builder.sha256(source),
        "source_locator": "PDF page 3, Scheme 1",
    }


def _structure_record(binding: dict[str, str], smiles: str = "CCO") -> dict[str, object]:
    return {
        "isomeric_smiles": smiles,
        "source_doi": "10.1021/reviewed",
        **binding,
        "identity_confirmed": True,
        "relative_stereochemistry_confirmed": True,
        "absolute_stereochemistry_status": "not_applicable",
    }


def _route_record(binding: dict[str, str]) -> dict[str, object]:
    return {
        "source_doi": "10.1021/reviewed",
        "reference_scope": "ordered_route",
        "source_artifacts": [binding],
        "steps": [
            {
                "step_id": "lit-step-1",
                "product_label": "1",
                "precursor_labels": ["2"],
                "transformation_class": "intramolecular cyclization",
                "strategic_role": "route-defining scaffold construction",
                "source_locator": "PDF page 3, Scheme 1",
            }
        ],
        "strategic_events": [
            {
                "event_id": "event-1",
                "description": "Construct the core by intramolecular cyclization.",
                "transformation_class": "intramolecular cyclization",
                "source_locator": "PDF page 3, Scheme 1",
            }
        ],
    }


def _accepted_review(
    reviewer_id: str,
    record: dict[str, object],
) -> dict[str, object]:
    return {
        "target_slot_id": "target-slot-reviewed",
        "reviewer_id": reviewer_id,
        "reviewed_at": "2026-09-02T00:00:00Z",
        "reviewer_attestation": True,
        "decision": "accept",
        "record": record,
    }


def _paper_reviews() -> list[dict[str, object]]:
    return [
        {
            "paper_id": "paper-reviewed",
            "reviewer_id": reviewer_id,
            "reviewed_at": "2026-09-02T00:00:00Z",
            "reviewer_attestation": True,
            "decision": "primary",
            "target_slot_ids": ["target-slot-reviewed"],
            "evidence_locators": ["PDF page 1, title and abstract"],
        }
        for reviewer_id in ("chemist-a", "chemist-b")
    ]


def test_two_matching_human_reviews_materialize_one_runnable_target(tmp_path) -> None:
    target = _human_review_target()
    binding = _source_binding(tmp_path)
    structure = _structure_record(binding)
    route = _route_record(binding)
    decisions = {
        "paper_reviews": _paper_reviews(),
        "structure_reviews": [
            _accepted_review("chemist-a", structure),
            _accepted_review("chemist-b", structure),
        ],
        "route_reviews": [
            _accepted_review("chemist-a", route),
            _accepted_review("chemist-b", route),
        ],
        "adjudications": [],
    }
    paper_states = builder.materialize_paper_review_states(
        [{"paper_id": "paper-reviewed"}],
        [target],
        decisions,
    )

    structures, routes = builder.materialize_human_admissions(
        [target],
        decisions,
        repo_root=tmp_path,
        paper_review_states=paper_states,
    )

    assert len(structures) == 1
    assert len(routes) == 1
    assert structures[0]["admission_basis"] == "two_matching_accepts"
    assert routes[0]["reference_scope"] == "ordered_route"
    assert target["target_smiles"] == "CCO"
    assert target["runnable"] is True


def test_conflicting_structure_transcriptions_wait_for_adjudication(tmp_path) -> None:
    target = _human_review_target()
    binding = _source_binding(tmp_path)
    decisions = {
        "structure_reviews": [
            _accepted_review("chemist-a", _structure_record(binding, "CCO")),
            _accepted_review("chemist-b", _structure_record(binding, "CCN")),
        ],
        "route_reviews": [],
        "adjudications": [],
    }

    structures, routes = builder.materialize_human_admissions(
        [target],
        decisions,
        repo_root=tmp_path,
    )

    assert structures == []
    assert routes == []
    assert target["structure_status"] == "human_review_in_progress"
    assert target["runnable"] is False


def test_structure_and_route_consensus_cannot_bypass_paper_admission(tmp_path) -> None:
    target = _human_review_target()
    binding = _source_binding(tmp_path)
    structure = _structure_record(binding)
    route = _route_record(binding)
    decisions = {
        "structure_reviews": [
            _accepted_review("chemist-a", structure),
            _accepted_review("chemist-b", structure),
        ],
        "route_reviews": [
            _accepted_review("chemist-a", route),
            _accepted_review("chemist-b", route),
        ],
        "adjudications": [],
    }

    structures, routes = builder.materialize_human_admissions(
        [target],
        decisions,
        repo_root=tmp_path,
        paper_review_states={"paper-reviewed": "not_started"},
    )

    assert len(structures) == 1
    assert len(routes) == 1
    assert target["runnable"] is False


def test_accepted_review_with_wrong_source_hash_fails_closed(tmp_path) -> None:
    target = _human_review_target()
    binding = _source_binding(tmp_path)
    binding["source_artifact_sha256"] = "0" * 64
    record = _structure_record(binding)
    decisions = {
        "structure_reviews": [
            _accepted_review("chemist-a", record),
            _accepted_review("chemist-b", record),
        ],
        "route_reviews": [],
        "adjudications": [],
    }

    with pytest.raises(RuntimeError, match="human_review_source_hash_invalid"):
        builder.materialize_human_admissions(
            [target],
            decisions,
            repo_root=tmp_path,
        )
