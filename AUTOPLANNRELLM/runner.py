"""Run AutoPlanner benchmarks with AUTOPLANNRELLM hooks enabled.

Usage:
    PYTHONPATH=. python -m AUTOPLANNRELLM.runner -- --bench data/benchmark_v2_100.json ...
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Enable AUTOPLANNRELLM and forward args to run_live_benchmark_parallel")
    ap.add_argument("--disable-llm-selection", action="store_true")
    ap.add_argument("--disable-llm-candidate", action="store_true")
    ap.add_argument("--cache", default=None, help="Optional JSONL cache for DeepSeek responses")
    ap.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    forwarded = list(args.benchmark_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise SystemExit("pass run_live_benchmark_parallel args after --")
    os.environ["AUTOPLANNRELLM_ENABLE"] = "1"
    os.environ["AUTOPLANNRELLM_LLM_SELECTION"] = "0" if args.disable_llm_selection else "1"
    os.environ["AUTOPLANNRELLM_ADD_LLM_CANDIDATE"] = "0" if args.disable_llm_candidate else "1"
    if args.cache:
        os.environ["AUTOPLANNRELLM_CACHE"] = args.cache
    from cascade_planner.eval.run_live_benchmark_parallel import main as benchmark_main

    sys.argv = ["python -m cascade_planner.eval.run_live_benchmark_parallel", *forwarded]
    benchmark_main()


if __name__ == "__main__":
    main()
