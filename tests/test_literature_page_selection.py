from cascade_planner.harness.literature_page_selection import (
    select_pdf_page_numbers,
)


def test_page_selection_uses_focus_and_document_coverage_not_first_prefix() -> None:
    selected = select_pdf_page_numbers(
        {
            "focus_page_numbers": [42, 43, 44, 45],
            "page_relevance": [
                {"page_number": 76, "score": 3, "route_context_score": 2}
            ],
        },
        page_count=100,
        max_pages=6,
    )

    assert selected[:3] == [42, 43, 44]
    assert 76 in selected
    assert any(page not in {42, 43, 44, 45, 76} for page in selected)
    assert selected != [1, 2, 3, 4, 5, 6]


def test_page_selection_spreads_when_native_text_has_no_focus() -> None:
    selected = select_pdf_page_numbers({}, page_count=80, max_pages=4)

    assert len(selected) == 4
    assert selected[0] > 1
    assert selected[-1] < 80
    assert len(set(selected)) == 4
