"""CLI for agent prior and critique utilities."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from AUTOPLANNRELLM.deepseek_client import is_placeholder_deepseek_key, normalize_deepseek_key_value
from cascade_planner.agent.failure_policy import predict_failure_risk
from cascade_planner.agent.prior_generator import generate_strategic_prior
from cascade_planner.agent.route_critic import critique_route_payload


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoPlanner agent utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prior = sub.add_parser("prior")
    p_prior.add_argument("--target", required=True)
    p_prior.add_argument("--provider", default="deterministic", choices=["deterministic", "deepseek"])

    p_crit = sub.add_parser("critique")
    p_crit.add_argument("--input", required=True, help="Route JSON payload from CLI or live benchmark target artifact")

    p_fail = sub.add_parser("failure-risk")
    p_fail.add_argument("--input", required=True, help="Route JSON payload or live benchmark target artifact")
    p_fail.add_argument("--model", default="results/shared/failure_classifier/pack_failure_classifier_20260507.pt")
    p_fail.add_argument("--threshold", type=float, default=0.5)

    p_check = sub.add_parser("check")
    p_check.add_argument("--provider", default="deepseek", choices=["deterministic", "deepseek"])
    p_check.add_argument("--target", default="CCO")
    p_check.add_argument("--strict", action="store_true", help="Exit non-zero if requested provider falls back")

    args = ap.parse_args()
    if args.cmd == "prior":
        print(json.dumps(generate_strategic_prior(args.target, provider=args.provider), indent=2))
    elif args.cmd == "critique":
        data = json.loads(Path(args.input).read_text())
        if "planner_output" in data:
            data = data["planner_output"]
        print(json.dumps(critique_route_payload(data), indent=2))
    elif args.cmd == "failure-risk":
        data = json.loads(Path(args.input).read_text())
        print(json.dumps(
            predict_failure_risk(data, model_path=args.model, threshold=args.threshold),
            indent=2,
        ))
    elif args.cmd == "check":
        prior = generate_strategic_prior(args.target, provider=args.provider)
        key = normalize_deepseek_key_value(os.environ.get("DEEPSEEK_API_KEY"))
        result = {
            "requested_provider": args.provider,
            "resolved_source": prior.get("source"),
            "key_present": bool(key) and not is_placeholder_deepseek_key(key) if args.provider == "deepseek" else None,
            "fallback": args.provider == "deepseek" and prior.get("source") != "deepseek",
            "unsupported_claims_count": len(prior.get("unsupported_claims") or []),
        }
        print(json.dumps(result, indent=2))
        if args.strict and result["fallback"]:
            sys.exit(2)


if __name__ == "__main__":
    main()
