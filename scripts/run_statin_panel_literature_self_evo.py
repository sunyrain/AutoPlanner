#!/usr/bin/env python3
"""Run the nine-statin literature workflow and self-evolution replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.agent.statin_panel import run_statin_panel_literature_self_evo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="docs/statins/summary.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets", default="", help="Comma-separated statin names/safe ids. Empty means all nine.")
    parser.add_argument("--query-budget", type=int, default=6)
    parser.add_argument(
        "--literature-backend",
        default="local",
        choices=["local", "manual", "pubmed", "local_pubmed", "codex", "api_json"],
    )
    parser.add_argument(
        "--execute-closure-followups",
        action="store_true",
        help="Execute PubMed lead searches for route-closure blocker follow-up queries when using a PubMed backend.",
    )
    parser.add_argument("--closure-followup-limit", type=int, default=1)
    parser.add_argument(
        "--execute-all-closure-followups",
        action="store_true",
        help="Execute PubMed lead searches for every queued route-closure blocker. Overrides --closure-followup-limit.",
    )
    parser.add_argument(
        "--execute-open-gap-searches",
        action="store_true",
        help="Execute field-level PubMed lead searches for open closure curation gaps.",
    )
    parser.add_argument("--open-gap-search-limit", type=int, default=0)
    parser.add_argument(
        "--execute-all-open-gap-searches",
        action="store_true",
        help="Execute field-level PubMed lead searches for every open gap. Overrides --open-gap-search-limit.",
    )
    parser.add_argument(
        "--execute-full-text-access-probes",
        action="store_true",
        help="Probe selected open-gap PubMed leads for PMID/DOI/PMC access metadata without storing full text.",
    )
    parser.add_argument("--full-text-access-probe-limit", type=int, default=0)
    parser.add_argument(
        "--execute-all-full-text-access-probes",
        action="store_true",
        help="Probe every selected open-gap lead for access metadata. Overrides --full-text-access-probe-limit.",
    )
    parser.add_argument(
        "--execute-full-text-signal-extractions",
        action="store_true",
        help=(
            "Fetch PMC XML for open-access open-gap leads and store field/route signal counts only; "
            "does not store full text or procedure text."
        ),
    )
    parser.add_argument("--full-text-signal-extraction-limit", type=int, default=0)
    parser.add_argument(
        "--execute-all-full-text-signal-extractions",
        action="store_true",
        help=(
            "Execute signal-only PMC extraction for every open-access open-gap lead. "
            "Overrides --full-text-signal-extraction-limit."
        ),
    )
    args = parser.parse_args()
    selected = [item.strip() for item in args.targets.split(",") if item.strip()]
    execute_closure_followups = args.execute_closure_followups or args.execute_all_closure_followups
    execute_open_gap_searches = args.execute_open_gap_searches or args.execute_all_open_gap_searches
    execute_full_text_access_probes = (
        args.execute_full_text_access_probes or args.execute_all_full_text_access_probes
    )
    execute_full_text_signal_extractions = (
        args.execute_full_text_signal_extractions or args.execute_all_full_text_signal_extractions
    )
    closure_followup_limit = -1 if args.execute_all_closure_followups else args.closure_followup_limit
    open_gap_search_limit = -1 if args.execute_all_open_gap_searches else args.open_gap_search_limit
    full_text_access_probe_limit = (
        -1 if args.execute_all_full_text_access_probes else args.full_text_access_probe_limit
    )
    full_text_signal_extraction_limit = (
        -1 if args.execute_all_full_text_signal_extractions else args.full_text_signal_extraction_limit
    )
    report = run_statin_panel_literature_self_evo(
        output_root=args.output_root,
        summary_path=args.summary,
        targets=selected or None,
        query_budget=args.query_budget,
        literature_backend=args.literature_backend,
        execute_closure_followups=execute_closure_followups,
        closure_followup_limit=closure_followup_limit,
        execute_open_gap_searches=execute_open_gap_searches,
        open_gap_search_limit=open_gap_search_limit,
        execute_full_text_access_probes=execute_full_text_access_probes,
        full_text_access_probe_limit=full_text_access_probe_limit,
        execute_full_text_signal_extractions=execute_full_text_signal_extractions,
        full_text_signal_extraction_limit=full_text_signal_extraction_limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
