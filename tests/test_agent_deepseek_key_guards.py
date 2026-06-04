import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cascade_planner.agent.cli import main as agent_cli_main
from cascade_planner.agent.prior_generator import deepseek_prior, generate_strategic_prior
from cascade_planner.cascadeboard.prior_benchmark import run_prior_comparison


class AgentDeepSeekKeyGuardsTest(unittest.TestCase):
    def test_deepseek_prior_rejects_placeholder_key(self):
        with self.assertRaisesRegex(RuntimeError, "placeholder"):
            deepseek_prior("CCO", api_key="'  replace_with_your_deepseek_key  '")

    def test_generate_prior_falls_back_when_deepseek_key_is_placeholder(self):
        prior = generate_strategic_prior(
            "CCO",
            provider="deepseek",
            api_key='"  replace_with_your_deepseek_key  "',
        )

        self.assertEqual(prior["source"], "deterministic")
        self.assertIn("deepseek_fallback: RuntimeError", prior["unsupported_claims"])

    def test_prior_benchmark_skips_placeholder_key_without_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": " replace_with_your_deepseek_key "}, clear=False):
                result = run_prior_comparison(
                    providers=["deepseek"],
                    bench_path=str(root / "bench.json"),
                    output_prefix=str(root / "prior_compare"),
                    report_path=str(root / "prior_compare.md"),
                    model_path=str(root / "model.pt"),
                    limit=1,
                    n_results=1,
                    n_candidates_per_skeleton=1,
                    skeleton_samples=1,
                    device="cpu",
                    check_stock=False,
                    prior_weight=1.0,
                    search_mode="rerank",
                    search_budget=None,
                    allow_deepseek_fallback=False,
                )

            self.assertEqual(result["rows"][0]["status"], "skipped: DEEPSEEK_API_KEY not configured")
            self.assertIsNone(result["rows"][0]["output"])
            report = (root / "prior_compare.md").read_text(encoding="utf-8")
            self.assertIn("not configured", report)

    def test_agent_cli_check_reports_placeholder_key_as_not_present(self):
        out = StringIO()
        argv = ["agent", "check", "--provider", "deepseek", "--target", "CCO"]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": " replace_with_your_deepseek_key "}, clear=False):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(out):
                    agent_cli_main()

        payload = json.loads(out.getvalue())
        self.assertFalse(payload["key_present"])
        self.assertTrue(payload["fallback"])
        self.assertEqual(payload["resolved_source"], "deterministic")

    def test_agent_cli_case_audit_worker_and_guided_policy_smoke(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case"
            run_case = _run_agent_cli([
                "agent",
                "run-case",
                "--target-smiles",
                target,
                "--target-name",
                "bufotalin_cli_case",
                "--family-hint",
                "bufotalin, bufadienolide, steroid, pyrone",
                "--frontier-smiles",
                target,
                "--output-dir",
                str(case_dir),
                "--query-budget",
                "4",
            ])
            artifacts = run_case["artifacts"]

            inspect = _run_agent_cli(["agent", "inspect-blackboard", "--case-bundle", artifacts["case_bundle"]])
            audit = _run_agent_cli([
                "agent",
                "audit-route",
                "--package",
                artifacts["hybrid_route_package"],
                "--validation",
                artifacts["validation"],
            ])
            task_path = root / "worker_task.json"
            task_path.write_text(json.dumps({
                "schema_version": "worker_task.v1",
                "task_id": "cli_worker",
                "case_id": run_case["case_id"],
                "task_type": "stuck_node_research",
                "required_artifact_type": "ResearchReport",
                "input_refs": ["frontier_report"],
                "allowed_tools": ["local_search"],
                "budget": {"timeout_s": 5, "max_output_bytes": 20000, "max_tool_calls": 2, "max_worker_runs": 1},
                "dry_run": True,
            }), encoding="utf-8")
            worker = _run_agent_cli(["agent", "worker-trace", "--task-json", str(task_path)])
            policy = _run_agent_cli([
                "agent",
                "rerun-with-policy",
                "--case-bundle",
                artifacts["case_bundle"],
                "--target-smiles",
                target,
            ])

        self.assertEqual(inspect["route_status"], "partial_anchor")
        self.assertIn("EvidenceCardList", inspect["artifact_types"])
        self.assertEqual(audit["route_status"], "partial_anchor")
        self.assertEqual(worker["status"], "accepted_draft")
        self.assertIn("chem_enzy_search_policy", policy["guided_config"]["search_flags"])
        self.assertEqual(policy["rerun_history"]["policy_id"], policy["policy"]["policy_id"])


if __name__ == "__main__":
    unittest.main()


def _run_agent_cli(argv: list[str]) -> dict:
    out = StringIO()
    with patch.object(sys, "argv", argv):
        with redirect_stdout(out):
            agent_cli_main()
    return json.loads(out.getvalue())
