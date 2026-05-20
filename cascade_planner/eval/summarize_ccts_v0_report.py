"""Summarize CCTS-v0 report key metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(report: Path, output: Path | None = None) -> str:
    data = json.loads(report.read_text(encoding="utf-8"))
    lines = [
        "# CCTS-v0 Summary",
        "",
        f"- Report: `{report}`",
        "",
        "## Test Positive-Label Ranking",
        "",
        "| Model | MRR covered | R@1 all | R@3 all | R@5 all | R@10 all | Exact MRR | Exact R@1 | Exact R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("chem_rank", (data.get("baseline_chem_rank") or {}).get("test") or {})]
    rows.extend((name, (rep.get("test") or {})) for name, rep in (data.get("models") or {}).items())
    rows.extend((name, (rep.get("test") or {})) for name, rep in (data.get("blends") or {}).items())
    for name, rep in rows:
        pos = rep.get("positive_label") or {}
        exact = rep.get("exact_label") or {}
        pos_k = pos.get("recall_at_k_all") or {}
        exact_k = exact.get("recall_at_k_all") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(pos.get("mrr_covered")),
                    _fmt(pos_k.get("1")),
                    _fmt(pos_k.get("3")),
                    _fmt(pos_k.get("5")),
                    _fmt(pos_k.get("10")),
                    _fmt(exact.get("mrr_covered")),
                    _fmt(exact_k.get("1")),
                    _fmt(exact_k.get("5")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Counts", "", "```json", json.dumps(data.get("counts") or {}, indent=2, ensure_ascii=False), "```"])
    text = "\n".join(lines) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize CCTS-v0 report")
    ap.add_argument("--report", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    print(summarize(Path(args.report), Path(args.output) if args.output else None))


if __name__ == "__main__":
    main()
