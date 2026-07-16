from __future__ import annotations

from pathlib import Path

import pytest

from scripts.authorized_literature_fetch import (
    _install_tolerant_navigation,
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
        "by chromatography.</p>"
        + " " * 2_000
    )
    path.write_text(content, encoding="utf-8")

    assert _usable_fulltext_html(
        [{"suffix": ".html", "path": str(path)}],
        doi="10.1039/c5ob01148e",
    )
