"""Shared display assets for live routes and self-contained exports."""

from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parent / "static"


def enhance_route_html(body: str) -> str:
    """Inline the same reading tools in live, static, and collection viewers."""
    styles = (STATIC_DIR / "route_quality.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "route_quality.js").read_text(encoding="utf-8")
    return body.replace(
        "</head>", f'<style id="routeQualityStyles">{styles}</style></head>', 1
    ).replace("</body>", f"<script>{script}</script></body>", 1)
