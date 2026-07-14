"""Bounded browser fallback for repository HTML blocked to HTTP clients."""
from __future__ import annotations

from threading import Lock
from urllib.parse import urlsplit


_BROWSER_FETCH_LOCK = Lock()
_ALLOWED_REPOSITORY_HOSTS = {"pmc.ncbi.nlm.nih.gov"}


def fetch_repository_html_with_browser(
    url: str,
    timeout_s: float,
    max_bytes: int,
) -> bytes:
    """Fetch one public repository page in an isolated, credential-free tab."""

    parsed = urlsplit(str(url or ""))
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_REPOSITORY_HOSTS:
        raise ValueError("literature_browser_url_not_allowed")
    if timeout_s <= 0 or max_bytes < 10_000:
        raise ValueError("literature_browser_limit_invalid")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ValueError("literature_browser_playwright_unavailable") from exc

    with _BROWSER_FETCH_LOCK, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                accept_downloads=False,
                service_workers="block",
            )
            page = context.new_page()
            page.set_default_navigation_timeout(max(1_000, int(timeout_s * 1_000)))
            response = page.goto(url, wait_until="domcontentloaded")
            if response is None or not 200 <= int(response.status) < 300:
                raise ValueError("literature_browser_repository_http_failure")
            final = urlsplit(page.url)
            if final.scheme != "https" or final.hostname not in _ALLOWED_REPOSITORY_HOSTS:
                raise ValueError("literature_browser_repository_redirect_rejected")
            content = page.content().encode("utf-8")
            if len(content) > max_bytes:
                raise ValueError("literature_browser_repository_response_too_large")
            return content
        finally:
            browser.close()
__all__ = [
    "fetch_repository_html_with_browser",
]
