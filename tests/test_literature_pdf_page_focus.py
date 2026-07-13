from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from cascade_planner.harness.agentic_blackboard import _pdf_structure_summary
from cascade_planner.harness.agentic_blackboard_controller import (
    _restore_materialized_literature_artifacts,
)
from cascade_planner.harness.codex_action_planner import (
    _bounded_planner_prompt_payload,
    _planner_blackboard_handoff,
)
from cascade_planner.harness.literature_pdf_extraction import (
    PAGE_FOCUS_ALGORITHM_VERSION,
    _build_page_focus,
)
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    _refresh_stale_pdf_focus_for_visual,
    _visual_chain_image_paths,
    execute_local_tool,
)


def _page_manifest(tmp_path: Path, focus: dict[str, object], *, page_count: int = 14) -> dict[str, object]:
    rendered_pages = []
    for page_number in range(1, page_count + 1):
        image = tmp_path / f"page_{page_number:03d}.png"
        image.write_bytes(f"page {page_number}".encode())
        rendered_pages.append({"page_number": page_number, "image_path": str(image)})
    return {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "rendered_pages": rendered_pages,
        "scheme_crops": [],
        "compound_text_snippets": [],
        **focus,
    }


def _visual_state(tmp_path: Path) -> ToolExecutionState:
    return ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "example target", "target_smiles": "CCO"},
        preflight={"case_id": "example_case"},
        budget=HarnessBudget(timeout_s=30),
    )


def test_restart_reindexes_digest_bound_pdf_render_manifest(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"current host source PDF")
    evidence_dir = tmp_path / "literature_pdf_structure_extraction_action"
    pages_dir = evidence_dir / "pages"
    pages_dir.mkdir(parents=True)
    image = pages_dir / "page_001.png"
    image.write_bytes(b"rendered page")
    manifest = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "accepted": True,
        "source_ref": "doi:10.example/recovery",
        "source_pdf_path": str(source_pdf),
        "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "rendered_pages": [
            {
                "page_number": 1,
                "image_path": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        ],
        "scheme_crops": [],
        "focus_audit": {
            "algorithm_version": PAGE_FOCUS_ALGORITHM_VERSION,
        },
    }
    (evidence_dir / "literature_pdf_structure_evidence.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    state = _visual_state(tmp_path)

    board = _restore_materialized_literature_artifacts(
        {"artifact_refs": {}},
        state=state,
    )

    recovered = state.artifacts["literature_pdf_structure_evidence_by_source"][
        "ref:doi:10.example/recovery"
    ]
    assert recovered["current_host_source_pdf_replay"] is True
    assert recovered["rendered_pages"][0]["image_path"] == str(image)
    assert state.artifacts["literature_pdf_artifact_recovery"][
        "selected_source_count"
    ] == 1
    assert Path(board["artifact_refs"]["literature_pdf_artifact_recovery"]).is_file()


def test_pmc10069651_like_label_after_first_six_is_selected_for_visual(tmp_path: Path) -> None:
    page_texts = [
        {"page_number": page, "text": f"General experimental information section {page}."}
        for page in range(1, 15)
    ]
    page_texts[10]["text"] = "Scheme 7. Conversion of labeled intermediate C42 into the advanced fragment."
    focus = _build_page_focus(
        page_texts,
        target_name="example target",
        target_aliases=["example alias"],
        expected_labels=["C42"],
        route_sequence_hint="Inspect the sequence from C42 to C16.",
        explicit_page_numbers=[],
    )
    evidence = _page_manifest(tmp_path, focus)

    selected = _visual_chain_image_paths(
        _visual_state(tmp_path),
        {"max_images": 6, "compress_images": False},
        evidence,
    )

    assert focus["focus_page_numbers"][0] == 11
    assert selected[0].name == "page_011.png"
    assert len(selected) == 6
    assert [path.name for path in selected] != [f"page_{page:03d}.png" for page in range(1, 7)]


def test_visual_selection_reserves_coverage_for_later_focus_pages(tmp_path: Path) -> None:
    focus = {
        "focus_page_numbers": [15, 4, 12, 1, 3, 14, 6, 7, 10, 20],
        "page_relevance": [],
    }
    evidence = _page_manifest(tmp_path, focus, page_count=20)

    selected = _visual_chain_image_paths(
        _visual_state(tmp_path),
        {"max_images": 6, "compress_images": False},
        evidence,
    )

    assert [path.name for path in selected[:4]] == [
        "page_015.png",
        "page_004.png",
        "page_012.png",
        "page_001.png",
    ]
    assert any(path.name in {"page_006.png", "page_007.png", "page_010.png", "page_014.png", "page_020.png"} for path in selected[4:])


def test_explicit_page_numbers_override_pmc10476182_like_text_focus(tmp_path: Path) -> None:
    focus = _build_page_focus(
        [
            {"page_number": page, "text": "C33 final coupling" if page == 12 else "Supporting information"}
            for page in range(1, 15)
        ],
        target_name="another target",
        target_aliases=[],
        expected_labels=["C33"],
        route_sequence_hint="",
        explicit_page_numbers=[],
    )
    evidence = _page_manifest(tmp_path, focus)

    selected = _visual_chain_image_paths(
        _visual_state(tmp_path),
        {"page_numbers": [4], "max_images": 6, "compress_images": False},
        evidence,
    )

    assert focus["focus_page_numbers"] == [12]
    assert [path.name for path in selected] == ["page_004.png"]


def test_page_focus_is_deterministic_and_strictly_bounded() -> None:
    page_texts = [
        {
            "page_number": page,
            "text": f"Route label X{page % 29} appears in this experimental page.",
        }
        for page in range(1, 701)
    ]
    labels = [f"X{index}" for index in range(80)]
    kwargs = {
        "target_name": "bounded example",
        "target_aliases": [f"alias {index}" for index in range(20)],
        "expected_labels": labels,
        "route_sequence_hint": " ".join(f"hint{index}" for index in range(100)),
        "explicit_page_numbers": [],
    }

    first = _build_page_focus(page_texts, **kwargs)
    second = _build_page_focus(page_texts, **kwargs)

    assert first == second
    assert len(first["focus_terms"]) <= 48
    assert len(first["focus_page_numbers"]) <= 16
    assert len(first["page_relevance"]) <= 160
    assert first["focus_audit"]["scanned_page_count"] == 512
    assert first["focus_audit"]["scan_truncated"] is True


def test_scanned_pdf_without_text_does_not_fabricate_relevance() -> None:
    focus = _build_page_focus(
        [{"page_number": page, "text": ""} for page in range(1, 20)],
        target_name="image only target",
        target_aliases=["image alias"],
        expected_labels=["C17"],
        route_sequence_hint="C17 to C18",
        explicit_page_numbers=[],
    )

    assert focus["focus_page_numbers"] == []
    assert focus["page_relevance"] == []
    assert focus["focus_audit"]["selection_strategy"] == "text_unavailable_fail_soft"
    assert focus["focus_audit"]["relevance_available"] is False
    assert focus["focus_audit"]["no_ocr_or_relevance_fabrication"] is True


def test_focus_normalizes_pdf_letter_number_spacing_and_ignores_bare_route_numbers() -> None:
    focus = _build_page_focus(
        [
            {"page_number": 1, "text": "general information"},
            {"page_number": 12, "text": "PF 07321332 was prepared from C 43."},
        ],
        target_name="nirmatrelvir",
        target_aliases=["PF-07321332"],
        expected_labels=["C43"],
        route_sequence_hint="compound 6 acid 10 then 11 to 4 and 1",
        explicit_page_numbers=[],
    )

    assert focus["focus_page_numbers"][0] == 12
    assert "6" not in focus["focus_terms"]
    assert "10" not in focus["focus_terms"]
    matched = {
        item["term"]
        for item in focus["page_relevance"][0]["matched_terms"]
    }
    assert {"PF-07321332", "C43"}.issubset(matched)


def test_focus_orders_distinct_expected_label_coverage_before_repeated_target_hits() -> None:
    focus = _build_page_focus(
        [
            {"page_number": 1, "text": "nirmatrelvir compound 6"},
            {"page_number": 2, "text": "nirmatrelvir acid 10"},
            {
                "page_number": 3,
                "text": "nirmatrelvir nirmatrelvir nirmatrelvir overview",
            },
        ],
        target_name="nirmatrelvir",
        target_aliases=[],
        expected_labels=["compound 6", "acid 10"],
        route_sequence_hint="",
        explicit_page_numbers=[],
    )

    assert set(focus["focus_page_numbers"][:2]) == {1, 2}


def test_synthesis_context_outranks_repeated_target_mentions_in_assay_pages(
    tmp_path: Path,
) -> None:
    page_texts = [
        {"page_number": page, "text": "supporting information"}
        for page in range(1, 71)
    ]
    for page, label in ((4, "2"), (6, "3"), (8, "4"), (10, "5")):
        page_texts[page - 1]["text"] = f"Synthesis of Compound {label}. Yield 71%."
    page_texts[11]["text"] = (
        "Synthesis of PF-07321332. Compound 5 was prepared and isolated."
    )
    page_texts[58]["text"] = (
        "Enzyme kinetics assay. Final reaction conditions used PF-07321332."
    )
    page_texts[64]["text"] = "PF-07321332 " * 20 + "plasma protein binding assay"
    focus = _build_page_focus(
        page_texts,
        target_name="nirmatrelvir",
        target_aliases=["PF-07321332"],
        expected_labels=["PF-07321332"],
        route_sequence_hint="complete manufacturing route and final nitrile formation",
        explicit_page_numbers=[],
    )
    evidence = _page_manifest(tmp_path, focus, page_count=70)

    selected = _visual_chain_image_paths(
        _visual_state(tmp_path),
        {"max_images": 6, "compress_images": False},
        evidence,
    )

    assert set(focus["focus_page_numbers"][:5]) == {4, 6, 8, 10, 12}
    assert 65 not in focus["focus_page_numbers"][:5]
    assert {path.name for path in selected[:5]} == {
        "page_004.png",
        "page_006.png",
        "page_008.png",
        "page_010.png",
        "page_012.png",
    }
    assert focus["focus_audit"]["algorithm_version"] == (
        PAGE_FOCUS_ALGORITHM_VERSION
    )


def test_target_route_heading_keeps_contiguous_procedure_pages_together(
    tmp_path: Path,
) -> None:
    page_texts = [
        {"page_number": page, "text": "supporting information"}
        for page in range(1, 21)
    ]
    for page in (4, 6, 8, 10):
        page_texts[page - 1]["text"] = (
            f"Synthesis of unrelated compound {page}. Yield 70%."
        )
    page_texts[11]["text"] = (
        "Synthesis of PF-07321332 (Compound 6): manufacturing route."
    )
    page_texts[12]["text"] = (
        "Intermediate T12 was prepared and isolated in 98% yield."
    )
    page_texts[13]["text"] = (
        "Intermediate T15 was prepared and used in the following step. Yield 100%."
    )
    page_texts[14]["text"] = (
        "Burgess reagent was added to T18 to afford compound 6. Yield 75%."
    )
    page_texts[15]["text"] = (
        "Recrystallization of PF-07321332 provided anhydrous material. Yield 97%."
    )
    focus = _build_page_focus(
        page_texts,
        target_name="nirmatrelvir",
        target_aliases=["PF-07321332"],
        expected_labels=["PF-07321332", "T18"],
        route_sequence_hint="complete manufacturing route and final nitrile formation",
        explicit_page_numbers=[],
    )
    evidence = _page_manifest(tmp_path, focus, page_count=20)

    selected = _visual_chain_image_paths(
        _visual_state(tmp_path),
        {"max_images": 6, "compress_images": False},
        evidence,
    )

    assert focus["focus_page_numbers"][:5] == [12, 13, 14, 15, 16]
    assert focus["focus_audit"]["route_anchor_window_page_numbers"] == [
        12,
        13,
        14,
        15,
        16,
    ]
    assert [path.name for path in selected[:4]] == [
        "page_012.png",
        "page_013.png",
        "page_014.png",
        "page_015.png",
    ]


def test_visual_execution_reindexes_stale_advisory_page_numbers(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- stale focus fixture")
    state = _visual_state(tmp_path)
    evidence = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "source_ref": "doi:example",
        "source_pdf_path": str(pdf),
        "source_pdf_sha256": "a" * 64,
        "focus_page_numbers": [65, 64, 57],
        "focus_audit": {
            "schema_version": "literature_pdf_page_focus_audit.v1",
            "selection_strategy": "deterministic_text_relevance",
        },
    }
    refreshed = {
        "focus_terms": ["PF-07321332"],
        "focus_page_numbers": [12, 13, 14, 15, 16],
        "page_relevance": [],
        "focus_hit_audit": [],
        "focus_audit": {
            "schema_version": "literature_pdf_page_focus_audit.v1",
            "algorithm_version": PAGE_FOCUS_ALGORITHM_VERSION,
            "selection_strategy": "deterministic_text_relevance",
        },
    }

    with patch(
        "cascade_planner.harness.tools.rebuild_literature_pdf_page_focus",
        return_value=refreshed,
    ):
        updated_evidence, updated_payload, audit = (
            _refresh_stale_pdf_focus_for_visual(
                state,
                payload={
                    "source_ref": "doi:example",
                    "pdf_path": str(pdf),
                    "page_numbers": [65, 64, 57],
                },
                pdf_evidence=evidence,
                matched_sources=[],
            )
        )

    assert updated_payload["page_numbers"] == [12, 13, 14, 15, 16]
    assert updated_evidence["focus_page_numbers"] == [12, 13, 14, 15, 16]
    assert audit["stale_advisory_page_numbers_replaced"] is True
    assert audit["prior_page_numbers"] == [65, 64, 57]


def test_pdf_tool_builds_focus_from_host_target_and_source_labels(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"synthetic pdf placeholder")
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={
            "target_name": "host target",
            "target_aliases": ["host alias"],
            "target_smiles": "CCO",
            "literature_sources": [
                {
                    "source_ref": "pmc:10069651",
                    "local_pdf": str(pdf),
                    "expected_scheme_or_compound_labels": ["C42", "C16"],
                    "route_sequence_hint": "C42 then C16",
                }
            ],
        },
        preflight={"case_id": "host_case"},
    )
    result = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "accepted": True,
        "status": "completed",
        "rendered_pages": [],
        "indexed_images": [],
        "scheme_crops": [],
        "compound_text_snippets": [],
        "focus_terms": [],
        "focus_page_numbers": [],
        "page_relevance": [],
        "focus_hit_audit": [],
        "focus_audit": {},
        "summary": {},
        "reasons": [],
    }

    with patch(
        "cascade_planner.harness.tools.extract_literature_pdf_assets",
        return_value=result,
    ) as extractor:
        record = execute_local_tool(
            "extract_pdf_literature_structures",
            {"source_ref": "pmc:10069651", "pdf_path": str(pdf)},
            state,
        )

    assert record.status == "accepted"
    kwargs = extractor.call_args.kwargs
    assert kwargs["target_name"] == "host target"
    assert kwargs["target_aliases"] == ["host alias"]
    assert kwargs["expected_labels"] == ["C42", "C16"]
    assert kwargs["route_sequence_hint"] == "C42 then C16"


def test_pdf_focus_survives_bounded_planner_handoff(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPLANNER_CODEX_ACTION_PLANNER_PROMPT_SNAPSHOT_MAX_BYTES", "12000")
    pdf_summary = _pdf_structure_summary(
        {
            "evidence_id": "pdf:pmc-like",
            "source_ref": "pmc:example",
            "focus_terms": ["C42", "C16"],
            "focus_page_numbers": [11, 14],
            "page_relevance": [],
            "focus_audit": {
                "selection_strategy": "deterministic_text_relevance",
                "relevance_available": True,
                "no_ocr_or_relevance_fabrication": True,
            },
            "summary": {},
        },
        artifact_ref="evidence.json",
    )
    blackboard = {
        "case_id": "bounded-focus",
        "target_profile": {"target_name": "example", "target_smiles": "CCO"},
        "literature_evidence": {"pdf_structure_evidence": [pdf_summary]},
        "current_belief": {},
    }
    handoff = _planner_blackboard_handoff(
        blackboard,
        planner_context={
            "literature_processing": {},
            "source_acquisition": {},
            "budget_remaining": {},
        },
    )
    handoff["evidence_board"]["source_candidates"] = [
        {"source_ref": f"doi:example/{index}", "title": "x" * 1_500}
        for index in range(40)
    ]

    payload, bounds = _bounded_planner_prompt_payload(handoff, round_index=2)

    retained = payload["blackboard_handoff"]["evidence_board"]["pdf_focus"][0]
    assert retained["focus"]["focus_page_numbers"] == [11, 14]
    assert retained["focus"]["focus_terms"] == ["C42", "C16"]
    assert bounds["within_bound"] is True
