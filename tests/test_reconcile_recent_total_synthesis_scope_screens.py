from __future__ import annotations

from scripts.reconcile_recent_total_synthesis_scope_screens import (
    normalize_name,
    reconcile,
)


def _row(paper_id: str, *, names: list[str], disposition: str = "likely_primary"):
    return {
        "paper_id": paper_id,
        "disposition": disposition,
        "completed_synthesis": True,
        "target_identity": "exact_complete",
        "target_names": names,
        "evidence_basis": "title_and_abstract",
        "reason": "example",
    }


def test_reconciliation_projects_only_full_exact_primary_agreement() -> None:
    queue = [{"paper_id": "paper-a", "doi": "10.1/a", "title": "Example"}]

    consensus, disagreements, slots = reconcile(
        [_row("paper-a", names=["(−)-Example A"])],
        [_row("paper-a", names=["(-)-Example A"])],
        queue,
    )

    assert consensus[0]["fully_agreed"] is True
    assert disagreements == []
    assert [row["target_name"] for row in slots] == ["(−)-Example A"]
    assert slots[0]["formal_benchmark_eligible"] is False


def test_reconciliation_routes_target_disagreement_to_review() -> None:
    queue = [{"paper_id": "paper-a", "doi": "10.1/a", "title": "Example"}]

    _, disagreements, slots = reconcile(
        [_row("paper-a", names=["Example A"])],
        [_row("paper-a", names=["Example B"])],
        queue,
    )

    assert disagreements[0]["target_names_agree"] is False
    assert slots == []


def test_normalize_name_preserves_stereochemical_tokens() -> None:
    assert normalize_name("(−)-Example  A") == normalize_name("(-)-Example A")
    assert normalize_name("(+)-Example A") != normalize_name("(-)-Example A")
