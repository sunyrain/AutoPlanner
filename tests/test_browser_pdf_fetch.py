from __future__ import annotations

import base64
from pathlib import Path

from scripts import browser_pdf_fetch
from scripts import authorized_literature_fetch


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


def test_fetch_prefers_authenticated_main_pdf_over_usable_landing_html(
    tmp_path: Path,
) -> None:
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "html").mkdir()
    doi = "10.1021/example"
    landing = (
        "<html><body>10.1021/example Experimental reaction mixture "
        "was stirred and purified by chromatography.</body></html>"
        + " " * 2_000
    ).encode()
    pdf = b"%PDF-1.7\nmain article"

    class Response:
        def __init__(self, url: str, body: bytes, content_type: str) -> None:
            self.url = url
            self._body = body
            self.headers = {"content-type": content_type}
            self.status = 200

        def body(self) -> bytes:
            return self._body

        def dispose(self) -> None:
            return None

    class Request:
        def get(self, url: str, **_kwargs: object) -> Response:
            return Response(url, pdf, "application/pdf")

    class Locator:
        def inner_text(self, **_kwargs: object) -> str:
            return "Experimental reaction mixture was stirred"

    class Page:
        def __init__(self) -> None:
            self.url = f"https://pubs.acs.org/doi/{doi}"
            self.context = type("Context", (), {"request": Request()})()

        def goto(self, url: str, **_kwargs: object) -> Response:
            self.url = url
            return Response(url, landing, "text/html")

        def wait_for_load_state(self, *_args: object, **_kwargs: object) -> None:
            return None

        def wait_for_timeout(self, *_args: object, **_kwargs: object) -> None:
            return None

        def locator(self, _selector: str) -> Locator:
            return Locator()

        def content(self) -> str:
            return landing.decode()

        def evaluate(self, _script: str) -> list[dict[str, str]]:
            return [
                {
                    "href": f"/doi/pdf/{doi}?ref=article_openPDF",
                    "label": "Open PDF",
                }
            ]

    result = browser_pdf_fetch._fetch_one(
        Page(),
        {
            "request_id": "acs-main-pdf",
            "source_ref": f"doi:{doi}",
            "doi": doi,
            "url": f"https://doi.org/{doi}",
        },
        pdf_dir=tmp_path / "pdfs",
        html_dir=tmp_path / "html",
        timeout_ms=30_000,
    )

    assert result["status"] == "downloaded"
    assert result["artifact_kind"] == "publisher_or_supplementary_pdf"
    assert Path(result["pdf_path"]).read_bytes() == pdf
    assert result["html_path"] == ""


def test_publisher_provider_downloads_authenticated_main_pdf(tmp_path: Path) -> None:
    payload = b"%PDF-1.7\nauthorized main article"

    class Driver:
        def execute_async_script(self, _script: str, url: str) -> dict[str, object]:
            return {
                "ok": True,
                "status": 200,
                "contentType": "application/pdf",
                "bodyBase64": base64.b64encode(payload).decode(),
                "url": url,
            }

    receipt = authorized_literature_fetch._download_authenticated_main_pdf(
        Driver(),
        doi="10.1021/example",
        publisher="ACS",
        output_dir=tmp_path,
    )

    assert receipt["status"] == "downloaded"
    assert Path(receipt["pdf_path"]).read_bytes() == payload
