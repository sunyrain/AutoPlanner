from __future__ import annotations

from scripts import browser_pdf_fetch


def test_candidate_urls_cover_publisher_specific_authorized_endpoints() -> None:
    assert (
        "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/anie.202400001"
        in browser_pdf_fetch._candidate_pdf_urls("", "10.1002/anie.202400001")
    )
    assert (
        "https://link.springer.com/content/pdf/10.1007/s00000-024-00001-1.pdf"
        in browser_pdf_fetch._candidate_pdf_urls("", "10.1007/s00000-024-00001-1")
    )
    assert (
        "https://www.rsc.org/suppdata/ob/c5/c5ob01148e/c5ob01148e1.pdf"
        in browser_pdf_fetch._candidate_pdf_urls("", "10.1039/c5ob01148e")
    )


def test_article_html_requires_doi_and_procedure_signals() -> None:
    useful = (
        "<html><head><meta name='citation_doi' content='10.1000/example'></head>"
        "<body><h2>Experimental</h2><p>The reaction mixture was stirred and "
        "purified by chromatography.</p></body></html>"
        + " " * 2_000
    ).encode()
    abstract_only = (
        "<html><body>10.1000/example abstract" + " " * 2_000 + "</body></html>"
    ).encode()

    assert browser_pdf_fetch._is_usable_article_html(
        useful,
        doi="10.1000/example",
    )
    assert not browser_pdf_fetch._is_usable_article_html(
        abstract_only,
        doi="10.1000/example",
    )
    assert not browser_pdf_fetch._is_usable_article_html(
        useful,
        doi="10.1000/different",
    )
