from __future__ import annotations

from scripts.screen_recent_total_synthesis_scope import (
    SCHEMA_VERSION,
    response_schema,
    screening_prompt,
    valid_payload,
)


def test_scope_screen_schema_avoids_unsupported_unique_items() -> None:
    schema = response_schema()

    assert schema["additionalProperties"] is False
    assert "uniqueItems" not in repr(schema)


def test_scope_screen_requires_exact_input_order() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [{"paper_id": "paper-b"}, {"paper_id": "paper-a"}],
    }

    assert valid_payload(payload, ["paper-b", "paper-a"])
    assert not valid_payload(payload, ["paper-a", "paper-b"])


def test_scope_screen_prompt_declares_nonadmission_and_no_browsing() -> None:
    prompt = screening_prompt(
        reviewer_id="reviewer-test",
        records=[
            {
                "paper_id": "paper-a",
                "doi": "10.1/example",
                "title": "Total Synthesis of Example A",
                "abstract": "We report the total synthesis of Example A.",
                "journal": "Example Journal",
                "publication_date": "2026-01-01",
            }
        ],
    )

    assert "preliminary and have no admission authority" in prompt
    assert "Do not browse" in prompt
    assert "paper-a" in prompt
