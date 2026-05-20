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


if __name__ == "__main__":
    unittest.main()
