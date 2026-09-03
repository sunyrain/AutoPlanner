from pathlib import Path

import fitz
from PIL import Image

from cascade_planner.harness.visual_literature_chain_agent import (
    _codex_visual_infrastructure_failure,
)
from scripts.extract_recent_total_synthesis_structure_images import (
    _render_pdf_page,
    _select_pdf_pages,
)
from scripts.project_recent_total_synthesis_visual_candidates import project_rows


def _artifact(path: Path, kind: str) -> dict[str, str]:
    return {
        "absolute_path": str(path),
        "cache_path": str(path),
        "artifact_kind": kind,
    }


def test_main_pdf_is_preferred_over_supporting_information(tmp_path: Path) -> None:
    main = tmp_path / "main.pdf"
    si = tmp_path / "si.pdf"
    for path, text in ((main, "Target A synthesis"), (si, "Target A references")):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((40, 40), text)
        document.save(path)
        document.close()

    selected = _select_pdf_pages(
        [_artifact(si, "supporting_information"), _artifact(main, "repository_main_pdf")],
        ["Target A"],
        max_images=8,
    )

    assert [(Path(row["absolute_path"]).name, page) for row, page in selected] == [
        ("main.pdf", 1)
    ]


def test_text_only_reference_pages_are_not_selected(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    document = fitz.open()
    body = document.new_page()
    body.insert_text((40, 40), "Target A synthesis")
    for index in range(25):
        y = 80 + index * 5
        body.draw_line((40, y), (90, y))
    references = document.new_page()
    references.insert_text((40, 40), "REFERENCES\nTarget A total synthesis")
    document.save(source)
    document.close()

    selected = _select_pdf_pages(
        [_artifact(source, "repository_main_pdf")], ["Target A"], max_images=8
    )

    assert [page for _row, page in selected] == [1]


def test_pdf_render_keeps_the_complete_page(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    output = tmp_path / "page.png"
    document = fitz.open()
    page = document.new_page(width=100, height=200)
    page.insert_text((10, 190), "Target A")
    document.save(source)
    document.close()

    locator = _render_pdf_page(source, 1, output)

    with Image.open(output) as image:
        assert image.size == (250, 500)
    assert locator == "full page"


def test_visual_projection_keeps_pubchem_separate_and_non_admitting() -> None:
    slots = [
        {
            "target_slot_id": "slot-1",
            "paper_id": "paper-1",
            "doi": "10.1/example",
            "target_name": "Target A",
            "slot_class": "primary",
        }
    ]
    pubchem = [
        {
            "target_slot_id": "slot-1",
            "lookup_status": "candidate_found_unverified",
            "candidates": [
                {
                    "reported_smiles": "C[C@H](O)F",
                    "rdkit_validation": {
                        "status": "roundtrip_valid",
                        "canonical_isomeric_smiles": "C[C@H](O)F",
                    },
                }
            ],
        }
    ]
    visual = {
        "paper-1": {
            "paper_id": "paper-1",
            "status": "completed",
            "input_sha256": "abc",
            "_result_path": "result.json",
            "source_images": [
                {
                    "image_path": "page-01.png",
                    "source_locator": "PDF page 1; full page",
                    "source_artifact_path": "paper.pdf",
                    "source_artifact_sha256": "source-hash",
                    "image_sha256": "image-hash",
                }
            ],
            "targets": [
                {
                    "target_name": "Target A",
                    "status": "exact_source_structure_candidate",
                    "reported_isomeric_smiles": "C[C@H](O)F",
                    "source_image_name": "page-01.png",
                    "source_locator": "PDF page 1; Figure 1",
                    "rdkit_validation": {
                        "status": "roundtrip_valid",
                        "canonical_isomeric_smiles": "C[C@H](O)F",
                    },
                }
            ],
        }
    }

    [row] = project_rows(slots, pubchem, visual)

    assert row["visual_pubchem_relation"] == "exact_isomeric_match"
    assert row["source_locator_complete"] is True
    assert row["visual_canonical_isomeric_smiles"] == "C[C@H](O)F"
    assert row["pubchem_canonical_isomeric_smiles"] == ["C[C@H](O)F"]
    assert row["admission_authority"] is False


def test_visual_projection_marks_stereo_difference_for_review() -> None:
    slots = [
        {
            "target_slot_id": "slot-1",
            "paper_id": "paper-1",
            "target_name": "Target A",
            "slot_class": "primary_candidate",
        }
    ]
    pubchem = [
        {
            "target_slot_id": "slot-1",
            "candidates": [
                {
                    "rdkit_validation": {
                        "canonical_isomeric_smiles": "C[C@H](O)F"
                    }
                }
            ],
        }
    ]
    visual = {
        "paper-1": {
            "paper_id": "paper-1",
            "status": "completed",
            "targets": [
                {
                    "target_name": "Target A",
                    "status": "exact_source_structure_candidate",
                    "rdkit_validation": {
                        "status": "roundtrip_valid",
                        "canonical_isomeric_smiles": "C[C@@H](O)F",
                    },
                }
            ],
        }
    }

    [row] = project_rows(slots, pubchem, visual)

    assert row["visual_pubchem_relation"] == "connectivity_match_stereo_difference"
    assert row["review_priority"] == "stereochemistry_review"


def test_visual_model_capacity_is_retryable_infrastructure_failure() -> None:
    failure = _codex_visual_infrastructure_failure(
        'Selected model is at capacity. Please try a different model.'
    )

    assert failure == {
        "reason": "codex_visual_model_capacity",
        "retry_after_hint": "",
    }
