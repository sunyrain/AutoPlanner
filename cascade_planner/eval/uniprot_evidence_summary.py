"""Summarize UniProt/evidence coverage used by enzymatic retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.enz_retrieval import _load_db


def summarize_cache(cache_path: str = "data/uniprot_cache.json") -> dict[str, int]:
    path = Path(cache_path)
    if not path.exists():
        return {
            "cache_entries": 0,
            "detailed_entries": 0,
            "with_sequence": 0,
            "with_cofactor": 0,
            "with_rhea_ids": 0,
        }
    try:
        cache = json.loads(path.read_text())
    except Exception:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    rows = [v for v in cache.values() if isinstance(v, dict)]
    return {
        "cache_entries": len(cache),
        "detailed_entries": len(rows),
        "with_sequence": sum(1 for row in rows if row.get("sequence")),
        "with_cofactor": sum(1 for row in rows if row.get("cofactor") or row.get("cofactors")),
        "with_rhea_ids": sum(1 for row in rows if row.get("rhea_ids")),
    }


def summarize_retrieval_db(data_path: str = "cascade_dataset_v3.json") -> dict[str, int]:
    db = _load_db(data_path)
    keys = [
        "uniprot_accession",
        "organism",
        "sequence_length",
        "sequence",
        "cofactor",
        "rhea_ids",
        "doi",
    ]
    out = {"enzymatic_reactions": len(db)}
    for key in keys:
        out[key] = sum(1 for row in db if (row.evidence or {}).get(key))
    return out


def write_markdown(summary: dict[str, Any], output_path: str, cache_path: str) -> None:
    cache = summary["cache"]
    retrieval = summary["retrieval"]
    lines = [
        "# UniProt Enrichment Summary",
        "",
        f"Source cache: `{cache_path}`",
        "",
        "## Cache Coverage",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Cache entries | {cache['cache_entries']} |",
        f"| Detailed cached entries | {cache['detailed_entries']} |",
        f"| Cached entries with sequence | {cache['with_sequence']} |",
        f"| Cached entries with cofactor | {cache['with_cofactor']} |",
        f"| Cached entries with Rhea IDs | {cache['with_rhea_ids']} |",
        "",
        "## v3 Enzymatic Retrieval Coverage",
        "",
        "| Field | Count |",
        "|---|---:|",
        f"| Enzymatic reactions | {retrieval['enzymatic_reactions']} |",
        f"| `uniprot_accession` | {retrieval['uniprot_accession']} |",
        f"| `organism` | {retrieval['organism']} |",
        f"| `sequence_length` | {retrieval['sequence_length']} |",
        f"| `sequence` | {retrieval['sequence']} |",
        f"| `cofactor` | {retrieval['cofactor']} |",
        f"| `rhea_ids` | {retrieval['rhea_ids']} |",
        f"| `doi` | {retrieval['doi']} |",
        "",
        "This is the local evidence layer that supports cascade planning. It is not a",
        "claim of universal UniProt coverage. The remaining gap is external enrichment",
        "for reactions and enzymes whose local cascade records do not already carry",
        "sequence, organism, cofactor, or Rhea links.",
        "",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize UniProt evidence coverage")
    ap.add_argument("--cache", default="data/uniprot_cache.json")
    ap.add_argument("--data", default="cascade_dataset_v3.json")
    ap.add_argument("--output", default="results/v2/uniprot_enrichment_summary.md")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    summary = {
        "cache": summarize_cache(args.cache),
        "retrieval": summarize_retrieval_db(args.data),
    }
    write_markdown(summary, args.output, args.cache)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
