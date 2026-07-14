from __future__ import annotations

from pathlib import Path

from cascade_planner.harness.deterministic_literature_registry import (
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.source_html import (
    PatentHtmlConfig,
    materialize_primary_patent_html,
)
from cascade_planner.harness.source_text_companion import (
    materialize_source_text_companion_pages,
    primary_html_companion,
    validate_source_text_companion_binding,
)


PUBLICATION = "US1234567A1"
SOURCE_REF = f"patent:{PUBLICATION}"


def _html() -> bytes:
    return f"""
    <html><head><meta name="DC.relation" content="{PUBLICATION}"></head>
    <body>
      <div id="p0001" class="description-paragraph">
        Background material unrelated to the experimental examples.
      </div>
      <div id="p0002" class="description-paragraph">Example 1</div>
      <div id="p0003" class="description-paragraph">
        Ethyl acetate (T1). Ethanol and acetic acid were added. The reaction
        mixture was stirred to afford T1 in 85 percent yield.
      </div>
      <div id="p0004" class="description-paragraph">
        The product was isolated and characterized.
      </div>
      <div id="p0005" class="description-paragraph">
        General claims and formulation text.
      </div>
    </body></html>
    """.encode()


def test_primary_patent_html_is_bounded_hash_bound_and_replayable(
    tmp_path: Path,
) -> None:
    audit = materialize_primary_patent_html(
        content=_html(),
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=f"https://patents.google.com/patent/{PUBLICATION}/en",
        output_dir=tmp_path,
        target_terms=["ethyl acetate", "ethanol", "acetic acid"],
    )

    assert audit["status"] == "completed"
    assert audit["model_invocations"] == 0
    assert audit["visual_invocations"] == 0
    assert audit["section_count"] >= 1
    assert primary_html_companion(audit["companion"])

    registry = compile_deterministic_literature_step_registry(
        [
            {
                "step_id": "html-ester",
                "product_smiles": "CCOC(C)=O",
                "reactant_smiles": ["CCO", "CC(=O)O"],
                "source_ref": SOURCE_REF,
                "source_text_companions": [audit["companion"]],
            }
        ],
        registry_path=tmp_path / "registry.json",
        structure_resolver=lambda name: {"Ethyl acetate": "CCOC(C)=O"}[name],
        candidate_name_resolver=lambda smiles: {
            "CCO": ["ethanol"],
            "CC(=O)O": ["acetic acid"],
        }.get(smiles, []),
    )

    assert registry["approved_binding_count"] == 1
    binding = registry["records"][0]["binding"]
    assert binding["source_artifact_kind"] == "html"
    assert binding["source_pdf_sha256"] == ""
    assert binding["image_sha256"] == ""
    assert binding["source_location"]["kind"] == "html_paragraph_range"
    companion = binding["source_text_companion"]
    assert validate_source_text_companion_binding(
        companion,
        expected_source_ref=SOURCE_REF,
    )

    Path(companion["artifact_path"]).write_text("tampered", encoding="utf-8")
    assert not validate_source_text_companion_binding(
        companion,
        expected_source_ref=SOURCE_REF,
    )


def test_search_snippet_or_wrong_publication_cannot_be_primary_html(
    tmp_path: Path,
) -> None:
    wrong = materialize_primary_patent_html(
        content=_html().replace(PUBLICATION.encode(), b"US9999999A1"),
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=f"https://patents.google.com/patent/{PUBLICATION}/en",
        output_dir=tmp_path,
        target_terms=["ethyl acetate"],
    )
    snippet = materialize_primary_patent_html(
        content=b"ethyl acetate synthesis patent snippet",
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=f"https://patents.google.com/patent/{PUBLICATION}/en",
        output_dir=tmp_path,
        target_terms=["ethyl acetate"],
    )

    assert wrong["status"] == "failed"
    assert wrong["reasons"] == ["patent_html_identity_not_found"]
    assert snippet["status"] == "failed"
    assert snippet["reasons"] == ["patent_html_byte_limit_invalid"]


def test_primary_html_window_budget_retains_late_high_relevance_procedure(
    tmp_path: Path,
) -> None:
    paragraphs = []
    for number in range(1, 121):
        if number in {10, 30, 50, 70, 90}:
            text = "General procedure for an unrelated formulation"
        elif number == 110:
            text = "Synthesis of rare-target-needle from precursor alpha"
        else:
            text = "Background description"
        paragraphs.append(
            f'<div id="p{number:04d}" class="description-paragraph">'
            f"{text}</div>"
        )
    content = (
        f'<html><meta name="DC.relation" content="{PUBLICATION}"><body>'
        + "".join(paragraphs)
        + "</body></html>"
    ).encode()

    audit = materialize_primary_patent_html(
        content=content,
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=f"https://patents.google.com/patent/{PUBLICATION}/en",
        output_dir=tmp_path,
        target_terms=["rare-target-needle"],
        config=PatentHtmlConfig(
            max_sections=2,
            max_selected_paragraphs=28,
        ),
    )

    assert audit["status"] == "completed"
    assert audit["section_count"] <= 2
    assert any(
        int(row["start_element_id"][1:])
        <= 110
        <= int(row["end_element_id"][1:])
        for row in audit["sections"]
    )


def test_primary_html_requires_official_publication_url_and_source_ref(
    tmp_path: Path,
) -> None:
    mirror = materialize_primary_patent_html(
        content=_html(),
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=f"https://mirror.invalid/patent/{PUBLICATION}/en",
        output_dir=tmp_path / "mirror",
        target_terms=["ethyl acetate"],
    )
    wrong_ref = materialize_primary_patent_html(
        content=_html(),
        publication=PUBLICATION,
        source_ref="patent:US9999999A1",
        source_url=f"https://patents.google.com/patent/{PUBLICATION}/en",
        output_dir=tmp_path / "wrong-ref",
        target_terms=["ethyl acetate"],
    )

    assert mirror["status"] == "failed"
    assert mirror["reasons"] == ["patent_html_source_binding_invalid"]
    assert wrong_ref["status"] == "failed"
    assert wrong_ref["reasons"] == ["patent_html_source_binding_invalid"]

    valid = materialize_primary_patent_html(
        content=_html(),
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=f"https://patents.google.com/patent/{PUBLICATION}/en",
        output_dir=tmp_path / "valid",
        target_terms=["ethyl acetate"],
    )
    bypass = dict(valid["companion"])
    bypass["source_url"] = f"https://mirror.invalid/patent/{PUBLICATION}/en"
    pages, binding, reasons = materialize_source_text_companion_pages(
        bypass,
        source_ref=SOURCE_REF,
    )
    assert pages == []
    assert binding == {}
    assert reasons == ("source_text_companion_primary_html_origin_invalid",)
