from __future__ import annotations

from pathlib import Path
import base64

import pytest

from scripts.authorized_literature_fetch import (
    _download_authenticated_main_pdf,
    _install_tolerant_navigation,
    _download_authenticated_supplements,
    _looks_like_main_article_link,
    _publisher,
    _verified_publisher_landing,
    _prepare_isolated_chromedriver,
    _usable_fulltext_content,
    _usable_fulltext_html,
)


class _Executor:
    def __init__(self) -> None:
        self.timeout = 0

    def set_timeout(self, value: int) -> None:
        self.timeout = value


class _Driver:
    def __init__(self, page_source: str) -> None:
        self.command_executor = _Executor()
        self.page_source = page_source

    def get(self, _url: str) -> None:
        raise TimeoutError("load event never settled")


def test_tolerant_navigation_accepts_loaded_doi_page_after_timeout() -> None:
    driver = _Driver("10.1039/c5ob01148e" + " full text" * 300)

    _install_tolerant_navigation(driver, doi="10.1039/c5ob01148e")

    assert driver.get("https://doi.org/10.1039/c5ob01148e") is None
    assert driver.command_executor.timeout == 180


def test_tolerant_navigation_rejects_unrelated_timeout_page() -> None:
    driver = _Driver("unrelated page" * 300)
    _install_tolerant_navigation(driver, doi="10.1039/c5ob01148e")

    with pytest.raises(TimeoutError):
        driver.get("https://doi.org/10.1039/c5ob01148e")


def test_usable_fulltext_html_does_not_depend_on_legacy_login_banner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug_unknown_status.html"
    content = (
        "<meta name='citation_doi' content='10.1039/c5ob01148e'>"
        "<h2>Experimental</h2><p>The reaction mixture was stirred and purified "
        "by chromatography.</p>" + " " * 2_000
    )
    path.write_text(content, encoding="utf-8")

    assert _usable_fulltext_html(
        [{"suffix": ".html", "path": str(path)}],
        doi="10.1039/c5ob01148e",
    )


def test_cloudflare_challenge_is_not_usable_fulltext() -> None:
    content = (
        "10.1021/example experimental supporting information "
        "https://challenges.cloudflare.com/challenge-platform" + " " * 2_000
    )
    assert not _usable_fulltext_content(content, doi="10.1021/example")


def test_current_acs_supplement_link_is_downloaded(tmp_path: Path) -> None:
    class FetchDriver:
        def execute_async_script(self, _script: str, _url: str):
            return {
                "ok": True,
                "status": 200,
                "contentType": "application/pdf",
                "bodyBase64": base64.b64encode(b"%PDF-1.7\nexample").decode("ascii"),
            }

    receipts = _download_authenticated_supplements(
        FetchDriver(),
        publisher="ACS",
        page_source=('<a href="/joceah/article-supplement/1/pdf/example_si_001/">SI</a>'),
        current_url="https://pubs.acs.org/joceah/article/1/example",
        output_dir=tmp_path,
    )

    assert [row["status"] for row in receipts] == ["downloaded"]
    assert Path(receipts[0]["path"]).read_bytes().startswith(b"%PDF-")


def test_ccs_chemistry_reuses_wiley_browser_adapter() -> None:
    assert _publisher("10.31635/ccschem.025.202506037", "") == "Wiley"
    assert _publisher("10.1055/a-2927-4044", "") == "Thieme"
    assert _publisher("10.1093/bbb/zbag055", "") == "OUP"
    assert _publisher("10.2174/example", "") == "Bentham"
    assert _publisher("10.5059/example", "") == "JStage"


def test_generic_citation_pdf_link_is_downloaded(tmp_path: Path) -> None:
    class FetchDriver:
        def execute_async_script(self, _script: str, url: str):
            assert url.endswith("/article.pdf")
            return {
                "ok": True,
                "status": 200,
                "contentType": "application/pdf",
                "bodyBase64": base64.b64encode(b"%PDF-1.7\narticle").decode("ascii"),
            }

    receipt = _download_authenticated_main_pdf(
        FetchDriver(),
        doi="10.1093/example",
        publisher="OUP",
        output_dir=tmp_path,
        page_source=(
            '<meta name="citation_pdf_url" content="https://academic.oup.com/example/article.pdf">'
        ),
        current_url="https://academic.oup.com/example/article",
    )

    assert receipt["status"] == "downloaded"
    assert Path(receipt["pdf_path"]).read_bytes().startswith(b"%PDF-")


def test_generic_publisher_media_pack_is_not_a_main_article() -> None:
    assert not _looks_like_main_article_link(
        "https://www.eurekaselect.com/images/pdf/Media-Pack-2024.pdf",
        doi="10.2174/example",
    )
    assert _looks_like_main_article_link(
        "https://www.eurekaselect.com/article/download?doi=10.2174/example",
        doi="10.2174/example",
    )
    assert _looks_like_main_article_link(
        "https://www.jstage.jst.go.jp/article/example/1/1/1/_pdf/-char/en",
        doi="10.5059/example",
    )


def test_bentham_uses_the_crossref_declared_article_download_endpoint(
    tmp_path: Path,
) -> None:
    class FetchDriver:
        def execute_async_script(self, _script: str, url: str):
            assert url == (
                "https://www.eurekaselect.com/article/download?"
                "doi=10.2174/example"
            )
            return {
                "ok": True,
                "status": 200,
                "contentType": "application/pdf",
                "bodyBase64": base64.b64encode(b"%PDF-1.7\narticle").decode("ascii"),
            }

    receipt = _download_authenticated_main_pdf(
        FetchDriver(),
        doi="10.2174/example",
        publisher="Bentham",
        output_dir=tmp_path,
    )

    assert receipt["status"] == "downloaded"


def test_supplement_landing_is_expanded_before_download(tmp_path: Path) -> None:
    class FetchDriver:
        def execute_async_script(self, _script: str, url: str):
            if url.endswith("/doi/suppl/10.31635/example"):
                payload = b'<a href="/files/example_supporting_information.pdf">SI</a>'
                content_type = "text/html"
            else:
                assert url.endswith("example_supporting_information.pdf")
                payload = b"%PDF-1.7\nsupplement"
                content_type = "application/pdf"
            return {
                "ok": True,
                "status": 200,
                "contentType": content_type,
                "bodyBase64": base64.b64encode(payload).decode("ascii"),
            }

    receipts = _download_authenticated_supplements(
        FetchDriver(),
        publisher="Wiley",
        page_source='<a href="/doi/suppl/10.31635/example">SI</a>',
        current_url="https://www.chinesechemsoc.org/doi/full/10.31635/example",
        output_dir=tmp_path,
    )

    assert [row["status"] for row in receipts] == [
        "expanded_landing",
        "downloaded",
    ]


def test_supplement_pdf_is_not_misclassified_as_main_article(
    tmp_path: Path,
) -> None:
    class FetchDriver:
        def execute_async_script(self, _script: str, _url: str):
            return {
                "ok": True,
                "status": 200,
                "contentType": "text/html",
                "bodyBase64": base64.b64encode(b"not a pdf").decode("ascii"),
            }

    receipt = _download_authenticated_main_pdf(
        FetchDriver(),
        doi="10.1055/example",
        publisher="Thieme",
        output_dir=tmp_path,
        page_source=(
            '<a href="https://www.thieme-connect.de/media/journal/supmat/example.pdf">'
            "Supplementary Material</a>"
        ),
        current_url="https://www.thieme-connect.com/article/10.1055/example",
    )

    assert receipt["status"] == "failed"
    assert not list(tmp_path.glob("main-article-*.pdf"))


def test_verified_landing_requires_doi_and_rejects_cloudflare() -> None:
    content = "<meta name='citation_doi' content='10.1055/example'>" + " " * 2_000
    assert _verified_publisher_landing(content, doi="10.1055/example")
    assert not _verified_publisher_landing(
        content + "challenges.cloudflare.com", doi="10.1055/example"
    )


def test_prepare_isolated_chromedriver_ignores_global_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePatcher:
        data_path = "global-stale-cache"

        def __init__(self, *, version_main: int, user_multi_procs: bool) -> None:
            assert version_main == 152
            assert user_multi_procs is False
            self.executable_path = str(Path(self.data_path) / "undetected_chromedriver.exe")

        def auto(self) -> None:
            Path(self.executable_path).write_bytes(b"driver-152")

    fake_uc = type("FakeUc", (), {"Patcher": FakePatcher})
    monkeypatch.setitem(__import__("sys").modules, "undetected_chromedriver", fake_uc)

    driver = _prepare_isolated_chromedriver(tmp_path, 152)

    assert driver == (tmp_path / "uc-driver-cache" / "undetected_chromedriver.exe").resolve()
    assert driver.read_bytes() == b"driver-152"
    assert FakePatcher.data_path == str(tmp_path / "uc-driver-cache")
