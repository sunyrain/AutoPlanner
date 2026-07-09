"""Resumable cache builder for SynthArena USPTO-190 target pages."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.eval.build_external_reservoir_smokes import (
    SYNTHARENA_TARGET,
    SYNTHARENA_USPTO_190,
    _download,
    _target_paths,
    _uspto_pagination_pages,
)


def cache_uspto190_targets(
    *,
    cache_dir: Path,
    offset: int = 0,
    limit: int = 190,
    max_fetches: int | None = None,
    fetch: bool = True,
    timeout: int = 30,
    sleep_s: float = 0.0,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    offset = max(0, int(offset))
    limit = max(0, int(limit))
    requested = offset + limit
    index_html = cache_dir / "uspto190_index.html"
    errors: list[dict[str, Any]] = []

    if fetch and not index_html.exists():
        try:
            _download(SYNTHARENA_USPTO_190, index_html, timeout=timeout)
        except Exception as exc:
            errors.append({"stage": "index", "error": f"{type(exc).__name__}:{exc}"})

    target_paths: list[str] = []
    if index_html.exists():
        index_text = index_html.read_text(encoding="utf-8", errors="ignore")
        target_paths.extend(_target_paths(index_text))
        for page_no in _uspto_pagination_pages(index_text):
            if len(dict.fromkeys(target_paths)) >= requested:
                break
            page_html = cache_dir / f"uspto190_page_{page_no}.html"
            if fetch and not page_html.exists():
                try:
                    _download(f"{SYNTHARENA_USPTO_190}?page={page_no}", page_html, timeout=timeout)
                except Exception as exc:
                    errors.append({"stage": "page", "page": page_no, "error": f"{type(exc).__name__}:{exc}"})
            if page_html.exists():
                target_paths.extend(_target_paths(page_html.read_text(encoding="utf-8", errors="ignore")))

    target_paths = list(dict.fromkeys(target_paths))
    selected = target_paths[offset:requested]
    fetched = []
    skipped = []
    missing = []
    fetch_budget = None if max_fetches is None else max(0, int(max_fetches))
    for target_path in selected:
        slug = target_path.rsplit("/", 1)[-1]
        html_path = cache_dir / f"uspto190_{slug}.html"
        if html_path.exists():
            skipped.append(str(html_path))
            continue
        if not fetch or fetch_budget == 0:
            missing.append(target_path)
            continue
        try:
            _download(SYNTHARENA_TARGET.format(target_path=target_path), html_path, timeout=timeout)
            fetched.append(str(html_path))
            if fetch_budget is not None:
                fetch_budget -= 1
            if sleep_s > 0:
                time.sleep(float(sleep_s))
        except Exception as exc:
            errors.append({"stage": "target", "target_path": target_path, "error": f"{type(exc).__name__}:{exc}"})
            missing.append(target_path)
            if fetch_budget is not None:
                fetch_budget -= 1

    cached_target_pages = [
        path
        for path in cache_dir.glob("uspto190_*.html")
        if not path.name.startswith("uspto190_page_") and path.name != "uspto190_index.html"
    ]
    report = {
        "schema_version": "uspto190_target_cache.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_dir": str(cache_dir),
        "offset": offset,
        "limit": limit,
        "requested": requested,
        "target_paths_discovered": len(target_paths),
        "selected_targets": len(selected),
        "cached_target_pages": len(cached_target_pages),
        "fetched_this_run": len(fetched),
        "skipped_existing": len(skipped),
        "missing_selected": len(missing),
        "errors": errors,
        "ready_for_selected_window": len(selected) > 0 and len(missing) == 0,
    }
    (cache_dir / "uspto190_cache_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (cache_dir / "uspto190_cache_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    rows = [
        ("target paths discovered", report.get("target_paths_discovered")),
        ("selected targets", report.get("selected_targets")),
        ("cached target pages", report.get("cached_target_pages")),
        ("fetched this run", report.get("fetched_this_run")),
        ("skipped existing", report.get("skipped_existing")),
        ("missing selected", report.get("missing_selected")),
        ("errors", len(report.get("errors") or [])),
    ]
    lines = [
        "# USPTO-190 Target Cache Report",
        "",
        f"Cache dir: `{report.get('cache_dir')}`",
        "",
        "| Item | Value |",
        "| --- | ---: |",
    ]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append(f"Ready for selected window: `{bool(report.get('ready_for_selected_window'))}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/resume a local cache of USPTO-190 target pages")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=190)
    ap.add_argument("--max-fetches", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    report = cache_uspto190_targets(
        cache_dir=Path(args.cache_dir),
        offset=args.offset,
        limit=args.limit,
        max_fetches=args.max_fetches,
        fetch=not args.no_fetch,
        timeout=args.timeout,
        sleep_s=args.sleep_s,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
