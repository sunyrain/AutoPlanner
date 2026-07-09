#!/usr/bin/env python3
"""Inspect curated strategic disconnection records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DB_GLOB = "data/strategic_disconnections/strategic_disconnections*.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        action="append",
        default=None,
        help=(
            "Strategic DB JSON file. May be repeated. "
            f"Default: merge files matching {DEFAULT_DB_GLOB!r}."
        ),
    )
    parser.add_argument("--query", default="", help="Case-insensitive text query over families, anchors, and disconnections.")
    parser.add_argument("--family", default="", help="Filter by family_id.")
    parser.add_argument("--json", action="store_true", help="Emit matching records as JSON.")
    args = parser.parse_args()

    data = load_databases(args.db)
    matches = query_records(data, query=args.query, family=args.family)
    if args.json:
        print(json.dumps(matches, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(matches))


def load_databases(paths: list[Path] | None = None) -> dict[str, Any]:
    db_paths = list(paths or sorted(Path().glob(DEFAULT_DB_GLOB)))
    if not db_paths:
        raise FileNotFoundError(DEFAULT_DB_GLOB)
    merged: dict[str, Any] = {
        "schema_version": "strategic_disconnections.merged",
        "sources": [str(path) for path in db_paths],
        "families": [],
        "anchors": [],
        "disconnections": [],
        "pptx_sources": [],
        "source_notes": [],
    }
    seen: dict[str, set[str]] = {
        "families": set(),
        "anchors": set(),
        "disconnections": set(),
    }
    key_by_section = {
        "families": "family_id",
        "anchors": "anchor_id",
        "disconnections": "id",
    }
    for path in db_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for section, key_name in key_by_section.items():
            for item in data.get(section, []) or []:
                key = str(item.get(key_name) or "")
                if key and key in seen[section]:
                    continue
                if key:
                    seen[section].add(key)
                merged[section].append(item)
        for section in ("pptx_sources", "source_notes"):
            merged[section].extend(data.get(section, []) or [])
    return merged


def query_records(data: dict[str, Any], *, query: str = "", family: str = "") -> dict[str, Any]:
    query_lc = query.strip().lower()
    family = family.strip()

    families = [
        item for item in data.get("families", [])
        if _record_matches(item, query_lc) and _family_matches(item, family)
    ]
    anchors = [
        item for item in data.get("anchors", [])
        if _record_matches(item, query_lc) and _family_matches(item, family)
    ]
    disconnections = [
        item for item in data.get("disconnections", [])
        if _record_matches(item, query_lc) and _family_matches(item, family)
    ]
    return {
        "schema_version": data.get("schema_version"),
        "query": query,
        "family": family,
        "counts": {
            "families": len(families),
            "anchors": len(anchors),
            "disconnections": len(disconnections),
        },
        "families": families,
        "anchors": anchors,
        "disconnections": disconnections,
    }


def render_markdown(matches: dict[str, Any]) -> str:
    lines = [
        "# Strategic Disconnection Query",
        "",
        f"- Query: `{matches.get('query') or '*'}`",
        f"- Family: `{matches.get('family') or '*'}`",
        f"- Counts: `{json.dumps(matches.get('counts') or {}, ensure_ascii=False)}`",
    ]

    if matches.get("families"):
        lines.extend(["", "## Families"])
        for item in matches["families"]:
            lines.append(f"- `{item.get('family_id')}`: {item.get('name')}")
            if item.get("strategic_principle"):
                lines.append(f"  - Principle: {item.get('strategic_principle')}")

    if matches.get("anchors"):
        lines.extend(["", "## Anchors"])
        for item in matches["anchors"]:
            lines.append(f"- `{item.get('anchor_id')}` ({item.get('family_id')}): {item.get('name')}")
            if item.get("role"):
                lines.append(f"  - Role: {item.get('role')}")
            if item.get("acceptance_policy"):
                lines.append(f"  - Policy: {item.get('acceptance_policy')}")

    if matches.get("disconnections"):
        lines.extend(["", "## Disconnections"])
        for item in matches["disconnections"]:
            lines.append(f"- `{item.get('id')}` ({item.get('family_id')}): {item.get('name')}")
            move = item.get("retrosynthetic_move") or {}
            break_bonds = move.get("break_bonds") or []
            if break_bonds:
                lines.append(f"  - Break bonds: {', '.join(str(x) for x in break_bonds)}")
            if move.get("planner_hint"):
                lines.append(f"  - Planner hint: {move.get('planner_hint')}")
            risks = item.get("risks") or []
            if risks:
                lines.append(f"  - Risks: {', '.join(str(x) for x in risks[:4])}")

    return "\n".join(lines)


def _family_matches(item: dict[str, Any], family: str) -> bool:
    return not family or item.get("family_id") == family


def _record_matches(item: Any, query_lc: str) -> bool:
    if not query_lc:
        return True
    return query_lc in json.dumps(item, ensure_ascii=False).lower()


if __name__ == "__main__":
    main()
