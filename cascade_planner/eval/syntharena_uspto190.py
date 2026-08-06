"""SynthArena USPTO-190 target discovery and cache parsing utilities."""
from __future__ import annotations

import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


SYNTHARENA_USPTO_190 = (
    "https://syntharena.ischemist.com/benchmarks/cmisbzsr30000xvdd613ymmbx"
)
SYNTHARENA_TARGET = SYNTHARENA_USPTO_190 + "/{target_path}"
FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
MAX_PLANNER_DEPTH = 8


def download_url(url: str, output: Path, *, timeout: int) -> None:
    """Download one cache artifact using the benchmark fetch identity."""
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": FETCH_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        output.write_bytes(response.read())


def target_paths(text: str) -> list[str]:
    """Return stable unique SynthArena target paths in document order."""
    out = []
    seen = set()
    for path in re.findall(r"targets/[A-Za-z0-9_-]+", text):
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def pagination_pages(text: str) -> list[int]:
    """Return the complete page range advertised by a benchmark index."""
    pages = {
        int(value)
        for value in re.findall(r"page=([0-9]+)", html.unescape(text))
        if int(value) > 1
    }
    if not pages:
        return []
    return list(range(2, max(pages) + 1))


def parse_target_page(path: Path) -> dict[str, Any] | None:
    """Parse one cached SynthArena target page into the benchmark row contract."""
    text = html.unescape(path.read_text(encoding="utf-8", errors="ignore"))
    obj = _extract_route_json(text)
    if not obj:
        return None
    target = obj.get("target") or {}
    molecule = target.get("molecule") or {}
    target_smiles = molecule.get("smiles")
    if not target_smiles:
        return None
    steps = []
    _walk_route(obj.get("rootNode") or {}, steps)
    reference_depth = int(target.get("routeLength") or len(steps) or 3)
    return {
        "doi": "SynthArena USPTO-190",
        "cascade_id": str(
            target.get("targetId") or target.get("id") or path.stem
        ),
        "target_smiles": target_smiles,
        "route_domain": "all_chemical",
        "operation_mode": "external_smoke",
        "depth": _planner_depth(reference_depth),
        "reference_depth": reference_depth,
        "gt_route": [
            {
                "rxn_smiles": reaction,
                "transformation": "other",
                "step_role": "external_acceptable_route",
            }
            for reaction in steps
        ],
    }


def _extract_route_json(text: str) -> dict[str, Any] | None:
    start = text.find('{\n  "route"')
    if start < 0:
        start = text.find('{"route"')
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        else:
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : index + 1])
    return None


def _walk_route(node: dict[str, Any], steps: list[str]) -> None:
    parent = ((node.get("molecule") or {}).get("smiles") or "").strip()
    children = node.get("children") or []
    reactants = [
        ((child.get("molecule") or {}).get("smiles") or "").strip()
        for child in children
    ]
    reactants = [smiles for smiles in reactants if smiles]
    if node.get("reactionStep") and parent and reactants:
        steps.append(".".join(reactants) + ">>" + parent)
    for child in children:
        _walk_route(child, steps)


def _planner_depth(depth: Any) -> int:
    try:
        value = int(depth)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(MAX_PLANNER_DEPTH, value))


__all__ = [
    "SYNTHARENA_TARGET",
    "SYNTHARENA_USPTO_190",
    "download_url",
    "pagination_pages",
    "parse_target_page",
    "target_paths",
]
