from __future__ import annotations

from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    independent_source_group,
    source_document_identity,
    source_record_representations,
    source_record_support_group,
)


def test_traceable_locator_validation_accepts_supported_identifiers() -> None:
    assert canonical_traceable_source_ref("DOI:10.1021/JA00083A066") == (
        "doi:10.1021/ja00083a066"
    )
    assert canonical_traceable_source_ref("PMID:012345678") == "pmid:12345678"
    assert canonical_traceable_source_ref("PMC:PMC987654") == "pmc:987654"
    assert canonical_traceable_source_ref("https://example.org/paper/1") == (
        "url:https://example.org/paper/1"
    )
    assert canonical_traceable_source_ref("local_pdf:Evidence/Paper.PDF#page=3") == (
        "local_pdf:evidence/paper.pdf"
    )
    assert canonical_traceable_source_ref("PII:S0140673610611059") == (
        "pii:S0140673610611059"
    )
    assert canonical_traceable_source_ref("patent:WO-2021-250648-A1") == (
        "patent:WO2021250648A1"
    )
    assert canonical_traceable_source_ref(
        "https://patents.google.com/patent/WO2021250648A1/en"
    ) == "patent:WO2021250648A1"
    assert canonical_traceable_source_ref(
        "https://[2606:4700:4700::1111]:443/paper"
    ) == "url:https://[2606:4700:4700::1111]:443/paper"


def test_traceable_locator_validation_rejects_free_text_and_private_urls() -> None:
    invalid = [
        "doi:not-a-doi",
        "banana",
        "source:free-form-name",
        "C:/papers/article.pdf",
        "local_pdf:not-a-document",
        "http://example.org/paper",
        "https://banana/paper",
        "https://localhost/paper",
        "https://127.0.0.1/paper",
        "https://10.0.0.8/paper",
        "https://999.999/paper",
        "https://example.com:bad/path",
        "https://example.com:99999/path",
        "pmid:not-a-pmid",
        "pmc:PMCbanana",
        "pii:not-a-pii",
    ]
    assert all(not canonical_traceable_source_ref(value) for value in invalid)


def test_compound_locator_only_parses_explicit_whitelisted_fields() -> None:
    compound = (
        "patent_publication:WO2021250648A1;"
        "url:https://patents.google.com/patent/WO2021250648A1/en;lines:10-30"
    )
    assert canonical_traceable_source_ref(compound) == "patent:WO2021250648A1"
    assert not canonical_traceable_source_ref(
        "description:see https://example.org/paper;lines:1-2"
    )


def test_only_trusted_source_channels_form_external_independence_groups() -> None:
    doi = ["doi:10.1000/example"]
    assert source_record_support_group("literature_exact", "literature_exact", doi) == (
        "literature:doi:10.1000/example"
    )
    assert source_record_support_group(
        "literature_exact",
        "literature_exact",
        ["pii:S0140673610611059"],
    ) == "literature:pii:S0140673610611059"
    assert source_record_support_group(
        "literature_exact",
        "literature_exact",
        ["patent_publication:WO2021250648A1"],
    ) == "literature:patent:WO2021250648A1"
    assert source_record_support_group("other", "model_only", doi) == "codex_model"
    assert source_record_support_group("invented_channel", "literature_exact", doi) == (
        "codex_model"
    )
    assert source_record_support_group("literature_analogy", "analogy", doi) == (
        "codex_model"
    )


def test_document_group_and_representation_identities_are_distinct() -> None:
    article = {
        "doi": "10.1000/shared",
        "url": "https://doi.org/10.1000/shared",
        "local_pdf": "cache/article.pdf",
        "content_scope": "article",
    }
    si = {
        "doi": "10.1000/shared",
        "local_pdf": "cache/shared_si.pdf",
        "content_scope": "article",
        "document_id": "pdf:0123456789abcdef",
    }
    assert independent_source_group(article) == "doi:10.1000/shared"
    assert independent_source_group(si) == "doi:10.1000/shared"
    assert source_document_identity(article) == (
        "document:doi:10.1000/shared:article"
    )
    assert source_document_identity(si) == (
        "document:doi:10.1000/shared:supplementary_information"
    )
    assert source_record_representations(article) == [
        "doi:10.1000/shared",
        "local_pdf:cache/article.pdf",
    ]
