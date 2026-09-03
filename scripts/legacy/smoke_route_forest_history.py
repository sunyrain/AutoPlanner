"""Smoke-test route forest rendering across saved blackboard runs.

This is intentionally read-only for run directories: it loads each
agent_blackboard.json, compiles the in-memory route forest, renders HTML in
memory, and writes only an optional aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.legacy.harness_runtime.route_forest import (  # noqa: E402
    SCHEMA_VERSION,
    compile_explored_route_forest,
    render_route_forest_html,
)
from cascade_planner.harness.route_forest_delivery import (  # noqa: E402
    route_forest_delivery_integrity_reasons,
)


_REQUIRED_ROUTE_FOREST_IDS = frozenset({"forest-data", "mainRoute", "detail"})


class _RouteForestHTMLParser(HTMLParser):
    """Collect the small semantic contract shared by route-forest renderers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.has_html_root = False
        self.element_ids: set[str] = set()
        self.body_attributes: dict[str, str] = {}
        self._in_forest_data = False
        self._forest_data_parts: list[str] = []
        self.forest_data_type = ""

    @property
    def forest_data(self) -> str:
        return "".join(self._forest_data_parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        if tag == "html":
            self.has_html_root = True
        if tag == "body":
            self.body_attributes = attributes
        element_id = attributes.get("id", "")
        if element_id:
            self.element_ids.add(element_id)
        if tag == "script" and element_id == "forest-data":
            self._in_forest_data = True
            self.forest_data_type = attributes.get("type", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_forest_data:
            self._in_forest_data = False

    def handle_data(self, data: str) -> None:
        if self._in_forest_data:
            self._forest_data_parts.append(data)


def route_forest_html_contract_reasons(
    html: str,
    *,
    expected_forest: dict[str, Any] | None = None,
) -> list[str]:
    """Validate renderer semantics without depending on labels or layout IDs.

    Display copy and navigation widgets can change freely. The durable contract
    is a document with a route host, an inspector, and a self-consistent
    ``route_forest_delivery.v1`` projection bound to the authoritative
    ``explored_route_forest.v1`` source digest.
    """
    if not isinstance(html, str):
        return ["route_forest_html_not_text"]

    parser = _RouteForestHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - defensive parser boundary.
        return [f"route_forest_html_parse_failed:{type(exc).__name__}"]

    reasons: list[str] = []
    if not parser.has_html_root:
        reasons.append("route_forest_html_missing_html_root")
    for element_id in sorted(_REQUIRED_ROUTE_FOREST_IDS - parser.element_ids):
        reasons.append(f"route_forest_html_missing_semantic_node:{element_id}")
    if parser.forest_data_type != "application/json":
        reasons.append("route_forest_html_invalid_forest_data_type")

    embedded_delivery: Any = None
    if parser.forest_data:
        try:
            embedded_delivery = json.loads(parser.forest_data)
        except json.JSONDecodeError:
            reasons.append("route_forest_html_invalid_forest_data_json")
    elif "forest-data" in parser.element_ids:
        reasons.append("route_forest_html_empty_forest_data")

    if isinstance(embedded_delivery, dict):
        reasons.extend(
            route_forest_delivery_integrity_reasons(
                embedded_delivery,
                source_forest=expected_forest,
            )
        )
    elif embedded_delivery is not None:
        reasons.append("route_forest_html_embedded_delivery_not_object")
    return reasons


@dataclass(frozen=True)
class HistorySmokeConfig:
    root: Path
    max_runs: int = 0
    require_nonempty_branches: bool = True
    require_nonempty_steps: bool = True


def discover_blackboard_runs(root: Path, *, max_runs: int = 0) -> list[Path]:
    root = root.resolve()
    runs = sorted(path.parent for path in root.rglob("agent_blackboard.json"))
    if max_runs > 0:
        return runs[:max_runs]
    return runs


def smoke_route_forest_history(config: HistorySmokeConfig) -> dict[str, Any]:
    runs = discover_blackboard_runs(config.root, max_runs=config.max_runs)
    rows: list[dict[str, Any]] = []
    for run_dir in runs:
        rows.append(_smoke_one_run(run_dir, config=config))

    accepted = all(row.get("accepted") for row in rows)
    summary = {
        "schema_version": "route_forest_history_smoke_summary.v1",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "root": str(config.root.resolve()),
        "accepted": accepted,
        "checked": len(rows),
        "compiled": sum(1 for row in rows if row.get("compiled")),
        "failed": sum(1 for row in rows if not row.get("accepted")),
        "zero_branch": sum(
            1
            for row in rows
            if int((row.get("counts") or {}).get("branches") or 0) == 0
        ),
        "zero_step": sum(
            1 for row in rows if int((row.get("counts") or {}).get("steps") or 0) == 0
        ),
        "html_bad": sum(
            1 for row in rows if "route_forest_html_invalid" in row.get("reasons", [])
        ),
        "rows": rows,
    }
    return summary


def _smoke_one_run(run_dir: Path, *, config: HistorySmokeConfig) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "route_forest_history_smoke_row.v1",
        "run_dir": str(run_dir),
        "accepted": False,
        "compiled": False,
        "reasons": [],
    }
    try:
        blackboard = _load_json(run_dir / "agent_blackboard.json")
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        row["reasons"] = [f"agent_blackboard_unreadable:{type(exc).__name__}:{exc}"]
        return row

    try:
        forest = compile_explored_route_forest(blackboard, run_dir=run_dir)
        html = render_route_forest_html(forest)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        row["reasons"] = [f"route_forest_compile_failed:{type(exc).__name__}:{exc}"]
        return row

    reasons = _forest_smoke_reasons(forest, html, config=config)
    row.update(
        {
            "accepted": not reasons,
            "compiled": True,
            "reasons": reasons,
            "case_id": str(forest.get("case_id") or blackboard.get("case_id") or ""),
            "target": dict(forest.get("target") or {}),
            "counts": dict(forest.get("counts") or {}),
            "branch_kinds": sorted(
                {
                    str(branch.get("kind") or "")
                    for branch in forest.get("branches") or []
                    if isinstance(branch, dict)
                }
            ),
        }
    )
    return row


def _forest_smoke_reasons(
    forest: dict[str, Any],
    html: str,
    *,
    config: HistorySmokeConfig,
) -> list[str]:
    reasons: list[str] = []
    if forest.get("schema_version") != SCHEMA_VERSION:
        reasons.append("invalid_or_missing_forest_schema")
    counts = dict(forest.get("counts") or {})
    if config.require_nonempty_branches and int(counts.get("branches") or 0) <= 0:
        reasons.append("route_forest_has_zero_branches")
    if config.require_nonempty_steps and int(counts.get("steps") or 0) <= 0:
        reasons.append("route_forest_has_zero_steps")
    if not isinstance(html, str) or len(html) < 1200:
        reasons.append("route_forest_html_too_small")
    if route_forest_html_contract_reasons(html, expected_forest=forest):
        reasons.append("route_forest_html_invalid")
    if ">undefined<" in html or ">NaN<" in html:
        reasons.append("route_forest_html_contains_obvious_js_placeholder")
    return reasons


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "shared")
    parser.add_argument(
        "--max-runs", type=int, default=0, help="Optional cap for quick local probes."
    )
    parser.add_argument("--allow-empty-branches", action="store_true")
    parser.add_argument("--allow-empty-steps", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = smoke_route_forest_history(
        HistorySmokeConfig(
            root=args.root,
            max_runs=max(0, int(args.max_runs or 0)),
            require_nonempty_branches=not bool(args.allow_empty_branches),
            require_nonempty_steps=not bool(args.allow_empty_steps),
        )
    )
    if args.summary_output:
        out = args.summary_output.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
