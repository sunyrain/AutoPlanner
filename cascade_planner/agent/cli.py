"""CLI for agent prior and critique utilities."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cascade_planner.agent.deepseek_credentials import (
    is_placeholder_deepseek_key,
    normalize_deepseek_key_value,
)
from cascade_planner.agent.case_blackboard import load_blackboard
from cascade_planner.agent.case_trace import load_case_bundle
from cascade_planner.agent.chem_enzy_policy import (
    apply_chem_enzy_search_policy,
    compile_chem_enzy_search_policy,
    compile_strategic_operator_from_case_bundle,
)
from cascade_planner.agent.codex_worker import run_codex_worker, worker_task_from_dict
from cascade_planner.agent.failure_policy import predict_failure_risk
from cascade_planner.agent.prior_generator import generate_strategic_prior
from cascade_planner.agent.route_critic import critique_route_payload
from cascade_planner.agent.route_auditor import audit_route_package
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.baselines.route_contract import RouteSearchConfig


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
    p_check.add_argument("--provider", default="deterministic", choices=["deterministic", "deepseek"])
    p_check.add_argument("--target", default="CCO")
    p_check.add_argument("--strict", action="store_true", help="Exit non-zero if requested provider falls back")

    p_run_case = sub.add_parser("run-case", help="Run the SMILES-first literature case workflow")
    p_run_case.add_argument("--target-smiles", required=True)
    p_run_case.add_argument("--target-name", default="")
    p_run_case.add_argument("--family-hint", default="")
    p_run_case.add_argument("--objective", default="route")
    p_run_case.add_argument("--output-dir", required=True)
    p_run_case.add_argument("--frontier-smiles", default="")
    p_run_case.add_argument("--baseline-json", default=None)
    p_run_case.add_argument("--evidence-jsonl", default=None)
    p_run_case.add_argument("--db", action="append", default=None)
    p_run_case.add_argument("--query-budget", type=int, default=12)
    p_run_case.add_argument(
        "--literature-backend",
        default="api_json",
        choices=["local", "manual", "pubmed", "local_pubmed", "codex", "api_json"],
        help=(
            "Literature evidence backend. Defaults to api_json, whose retrosynthesis "
            "worker key is read from the repository key.txt file; local/manual are deterministic; "
            "pubmed/local_pubmed use NCBI E-utilities."
        ),
    )
    p_run_case.add_argument("--worker-timeout-s", type=float, default=60.0)
    p_run_case.add_argument("--worker-max-output-bytes", type=int, default=200_000)
    p_run_case.add_argument("--worker-max-tool-calls", type=int, default=8)

    p_audit = sub.add_parser("audit-route", help="Audit a route package and emit RouteStatus")
    p_audit.add_argument("--package", required=True)
    p_audit.add_argument("--validation", default=None)
    p_audit.add_argument("--conditions-json", default=None)
    p_audit.add_argument("--enzyme-actions-json", default=None)
    p_audit.add_argument("--stock-audit-passed", action="store_true")
    p_audit.add_argument("--target-match", action="store_true", default=True)
    p_audit.add_argument("--target-mismatch", action="store_false", dest="target_match")

    p_inspect = sub.add_parser("inspect-blackboard", help="Inspect a case bundle or blackboard JSON")
    p_inspect.add_argument("--case-bundle", default=None)
    p_inspect.add_argument("--blackboard", default=None)

    p_worker = sub.add_parser("worker-trace", help="Run a controlled worker task with backend trace")
    p_worker.add_argument("--task-json", required=True)
    p_worker.add_argument("--mock-output-json", default=None)
    p_worker.add_argument(
        "--backend",
        default=os.environ.get("AUTOPLANNER_CODEX_WORKER_BACKEND") or "codex",
        choices=["codex", "api_json", "mock"],
        help="Worker backend for non-dry-run tasks. Defaults to Codex CLI.",
    )

    p_rerun = sub.add_parser("rerun-with-policy", help="Compile a guided ChemEnzy policy from a case bundle")
    p_rerun.add_argument("--case-bundle", required=True)
    p_rerun.add_argument("--target-smiles", default="")
    p_rerun.add_argument("--max-iterations", type=int, default=16)
    p_rerun.add_argument("--max-depth", type=int, default=6)
    p_rerun.add_argument("--expansion-topk", type=int, default=50)

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
    elif args.cmd == "run-case":
        result = run_smiles_first_workflow(
            SmilesFirstWorkflowConfig(
                target_smiles=args.target_smiles,
                target_name=args.target_name,
                family_hint=args.family_hint,
                objective=args.objective,
                output_dir=Path(args.output_dir),
                frontier_smiles=args.frontier_smiles,
                baseline_json=args.baseline_json,
                evidence_jsonl=args.evidence_jsonl,
                db_paths=args.db,
                query_budget=args.query_budget,
                literature_backend=args.literature_backend,
                worker_timeout_s=args.worker_timeout_s,
                worker_max_output_bytes=args.worker_max_output_bytes,
                worker_max_tool_calls=args.worker_max_tool_calls,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    elif args.cmd == "audit-route":
        package = _read_json(args.package)
        validation = _read_json(args.validation) if args.validation else None
        conditions = _read_json_list(args.conditions_json)
        enzyme_actions = _read_json_list(args.enzyme_actions_json)
        report = audit_route_package(
            package,
            validation=validation,
            stock_audit_passed=bool(args.stock_audit_passed),
            target_match=bool(args.target_match),
            condition_candidates=conditions,
            enzyme_actions=enzyme_actions,
        )
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    elif args.cmd == "inspect-blackboard":
        if bool(args.case_bundle) == bool(args.blackboard):
            raise SystemExit("provide exactly one of --case-bundle or --blackboard")
        if args.case_bundle:
            bundle = load_case_bundle(args.case_bundle)
            payload = {
                "schema_version": "agent_cli_case_bundle_inspection.v1",
                "case_id": bundle.case_id,
                "route_status": bundle.route_status.value,
                "artifact_count": len(bundle.artifacts),
                "failure_event_count": len(bundle.failure_events),
                "artifact_types": sorted({artifact.artifact_type for artifact in bundle.artifacts}),
                "failure_reasons": [event.reason for event in bundle.failure_events],
                "case_bundle": bundle.to_dict(),
            }
        else:
            board = load_blackboard(args.blackboard)
            payload = {
                "schema_version": "agent_cli_blackboard_inspection.v1",
                "summary": board.current_summary(),
                "blackboard": board.to_dict(),
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    elif args.cmd == "worker-trace":
        task = worker_task_from_dict(_read_json(args.task_json))
        mock_output = _read_json(args.mock_output_json) if args.mock_output_json else None
        use_codex_cli = args.backend == "codex" and mock_output is None and not task.dry_run
        use_api_json = args.backend == "api_json" and mock_output is None and not task.dry_run
        record = run_codex_worker(
            task,
            mock_output=mock_output,
            use_codex_cli=use_codex_cli,
            use_api_json=use_api_json,
        )
        print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    elif args.cmd == "rerun-with-policy":
        bundle = load_case_bundle(args.case_bundle)
        operator = compile_strategic_operator_from_case_bundle(
            bundle,
            max_iterations=args.max_iterations,
            max_depth=args.max_depth,
            expansion_topk=args.expansion_topk,
        )
        policy = compile_chem_enzy_search_policy(operator)
        target = args.target_smiles or _target_smiles_from_case_bundle(bundle) or "CCO"
        guided_config = apply_chem_enzy_search_policy(RouteSearchConfig(target_smiles=target), policy)
        payload = {
            "schema_version": "agent_cli_guided_policy_compile.v1",
            "case_id": bundle.case_id,
            "operator": operator.to_dict(),
            "policy": policy.to_dict(),
            "guided_config": {
                "target_smiles": guided_config.target_smiles,
                "max_iterations": guided_config.max_iterations,
                "max_depth": guided_config.max_depth,
                "expansion_topk": guided_config.expansion_topk,
                "search_flags": guided_config.search_flags,
            },
            "rerun_history": {
                "policy_id": policy.policy_id,
                "operator_id": operator.operator_id,
                "evidence_refs": policy.evidence_refs,
                "budget": policy.budget.to_dict(),
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _read_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_list(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        return [dict(data)]
    raise ValueError(f"expected JSON object or array: {path}")


def _target_smiles_from_case_bundle(bundle) -> str:
    for artifact in bundle.accepted_artifacts("HybridRoutePackage"):
        payload = artifact.payload or {}
        return str(((payload.get("target") or {}).get("smiles")) or "")
    return ""


if __name__ == "__main__":
    main()
