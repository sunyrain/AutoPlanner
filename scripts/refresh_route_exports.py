"""Refresh native offline viewers and depictions while preserving their saved data.

Usage: python scripts/refresh_route_exports.py results/discussion/*.html
"""

from __future__ import annotations

import argparse
from glob import glob
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade_planner.web.route_display import enhance_route_html
from cascade_planner.web.v4_live_synthesis import render_molecule_svg
from cascade_planner.web.v4_showcase_export import (
    STATIC_DIR,
    render_run_export_bundle_html,
)


class SavedDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.identifier = ""
        self.active = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("id") in {"showcaseData", "collectionData"}:
            self.identifier = str(attributes["id"])
            self.active = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.active = False

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append(data)


def refresh_export(path: Path) -> dict[str, object]:
    parser = SavedDataParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if not parser.identifier:
        return {"path": str(path), "status": "skipped: not a native route export"}
    payload = json.loads("".join(parser.parts))
    bundles = (
        [entry["bundle"] for entry in payload["entries"]]
        if parser.identifier == "collectionData" else [payload]
    )
    for bundle in bundles:
        # Only the depiction library changes. Projections, selections, reviews,
        # replay frames, labels, timestamps, and route counts stay as saved.
        for smiles in bundle.get("molecules", {}):
            svg, valid = render_molecule_svg(smiles)
            if valid:
                bundle["molecules"][smiles] = svg
    if parser.identifier == "showcaseData":
        body = render_run_export_bundle_html(payload)
    else:
        viewer = (STATIC_DIR / "run_showcase.html").read_text(encoding="utf-8")
        payload["viewer_template"] = enhance_route_html(viewer.replace(
            "__AUTOPLANNER_EXPORT_STYLES__",
            (STATIC_DIR / "run_export.css").read_text(encoding="utf-8"),
        ))
        body = (STATIC_DIR / "run_replay_collection.html").read_text(encoding="utf-8")
        body = body.replace(
            "__AUTOPLANNER_COLLECTION_DATA__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
        )
    path.write_text(body, encoding="utf-8")
    return {"path": str(path), "status": "refreshed", "runs": len(bundles)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    paths = dict.fromkeys(Path(match).resolve() for pattern in args.paths for match in glob(pattern))
    if not paths:
        parser.error("no matching exports")
    for path in paths:
        print(json.dumps(refresh_export(path), ensure_ascii=False))


if __name__ == "__main__":
    main()
