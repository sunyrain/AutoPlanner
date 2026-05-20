import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class StrictReviewWrapperExitCodesTest(unittest.TestCase):
    def test_after_key_wrapper_exits_nonzero_when_primary_merge_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_readiness_stub(root)
            _write_executable(
                root / "scripts/run_strict_model_review_real.sh",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                out="results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline"
                mkdir -p "$out"
                printf '%s\n' '{"decision":{"ready_for_expert_training":false}}' \
                  > "$out/strict_model_real_merged_route_block_value_pack_report.json"
                """,
            )

            result = _run_wrapper(root, "scripts/run_strict_review_full_after_key.sh")

            self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
            self.assertIn("Primary merged pack is not ready", result.stdout)

            allowed = _run_wrapper(
                root,
                "scripts/run_strict_review_full_after_key.sh",
                extra_env={"ALLOW_NOT_READY_EXIT_ZERO": "1"},
            )

            self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)

    def test_real_review_runner_rejects_placeholder_env_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            result = _run_wrapper(
                root,
                "scripts/run_route_block_review_expansion_real.sh",
                extra_env={"DEEPSEEK_API_KEY": "replace_with_your_deepseek_key"},
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("dotenv placeholder", result.stderr)

    def test_real_review_runner_rejects_whitespace_placeholder_env_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            result = _run_wrapper(
                root,
                "scripts/run_route_block_review_expansion_real.sh",
                extra_env={"DEEPSEEK_API_KEY": "  replace_with_your_deepseek_key  "},
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("dotenv placeholder", result.stderr)

    def test_real_review_runner_rejects_copied_placeholder_template(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".env.local").write_text(
                "# Copy this file to .env.local and replace the placeholder.\n"
                "# Do not commit .env.local or any real API key.\n"
                "DEEPSEEK_API_KEY=replace_with_your_deepseek_key\n",
                encoding="utf-8",
            )

            result = _run_wrapper(root, "scripts/run_route_block_review_expansion_real.sh")

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("dotenv placeholder", result.stderr)

    def test_filled_csv_wrapper_exits_nonzero_when_human_merge_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_readiness_stub(root)
            _write_csv_pipeline_stubs(root)
            csv_path = root / (
                "results/shared/model_strengthening_20260519_strict_model_review_packet/"
                "route_pool_evidence_review_calibration_subset_TO_FILL.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("review_id,route_id,expert_route_plausible\nreview-1,route-1,yes\n", encoding="utf-8")
            value_pack = root / (
                "results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/"
                "route_block_value_pack.jsonl"
            )
            value_pack.parent.mkdir(parents=True, exist_ok=True)
            value_pack.write_text("{}\n", encoding="utf-8")

            result = _run_wrapper(root, "scripts/run_strict_review_from_filled_csv.sh")

            self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
            self.assertIn("Human CSV merged pack is not ready", result.stdout)

            allowed = _run_wrapper(
                root,
                "scripts/run_strict_review_from_filled_csv.sh",
                extra_env={"ALLOW_NOT_READY_EXIT_ZERO": "1"},
            )

            self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)

    def test_filled_csv_wrapper_rejects_filled_rows_without_route_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / (
                "results/shared/model_strengthening_20260519_strict_model_review_packet/"
                "route_pool_evidence_review_calibration_subset_TO_FILL.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(
                "review_id,route_id,expert_route_plausible,expert_comments\n"
                "review-1,,yes,valid decision but missing route id\n",
                encoding="utf-8",
            )
            value_pack = root / (
                "results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/"
                "route_block_value_pack.jsonl"
            )
            value_pack.parent.mkdir(parents=True, exist_ok=True)
            value_pack.write_text("{}\n", encoding="utf-8")

            result = _run_wrapper(root, "scripts/run_strict_review_from_filled_csv.sh")

            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("Filled expert decision rows missing route_id", result.stdout + result.stderr)

    def test_filled_csv_wrapper_rejects_comment_only_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / (
                "results/shared/model_strengthening_20260519_strict_model_review_packet/"
                "route_pool_evidence_review_calibration_subset_TO_FILL.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(
                "review_id,expert_route_plausible,expert_comments\n"
                "review-1,,comment-only row\n",
                encoding="utf-8",
            )
            value_pack = root / (
                "results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/"
                "route_block_value_pack.jsonl"
            )
            value_pack.parent.mkdir(parents=True, exist_ok=True)
            value_pack.write_text("{}\n", encoding="utf-8")

            result = _run_wrapper(root, "scripts/run_strict_review_from_filled_csv.sh")

            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("No filled expert decision rows found", result.stdout + result.stderr)


def _run_wrapper(root: Path, script: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["ROOT"] = str(root)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(REPO_ROOT / script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_readiness_stub(root: Path) -> None:
    module = root / "cascade_planner/eval/check_strict_review_pipeline_readiness.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    (root / "cascade_planner/__init__.py").write_text("", encoding="utf-8")
    (root / "cascade_planner/eval/__init__.py").write_text("", encoding="utf-8")
    module.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--root")
                parser.add_argument("--output-json", type=Path)
                args = parser.parse_args()
                payload = {"decision": {"ready_for_expert_value_training": False}}
                if args.output_json:
                    args.output_json.parent.mkdir(parents=True, exist_ok=True)
                    args.output_json.write_text(json.dumps(payload), encoding="utf-8")
                print(json.dumps(payload["decision"]))


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )


def _write_csv_pipeline_stubs(root: Path) -> None:
    _write_python_module(
        root / "cascade_planner/eval/run_route_pool_evidence_review_csv_pipeline.py",
        """
        import argparse
        import json
        from pathlib import Path


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-dir", required=True)
            parser.add_argument("--prefix", required=True)
            args, _ = parser.parse_known_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{args.prefix}_labels.jsonl").write_text("{}\\n", encoding="utf-8")
            (out / f"{args.prefix}_csv_pipeline_manifest.json").write_text("{}", encoding="utf-8")


        if __name__ == "__main__":
            main()
        """,
    )
    _write_python_module(
        root / "cascade_planner/eval/build_route_block_review_label_pack.py",
        """
        import argparse
        from pathlib import Path


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-jsonl", required=True)
            parser.add_argument("--report", required=True)
            args, _ = parser.parse_known_args()
            Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_jsonl).write_text("{}\\n", encoding="utf-8")
            Path(args.report).write_text("{}", encoding="utf-8")


        if __name__ == "__main__":
            main()
        """,
    )
    _write_python_module(
        root / "cascade_planner/eval/merge_route_block_review_labels.py",
        """
        import argparse
        import json
        from pathlib import Path


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-jsonl", required=True)
            parser.add_argument("--report", required=True)
            args, _ = parser.parse_known_args()
            Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_jsonl).write_text("{}\\n", encoding="utf-8")
            payload = {"decision": {"ready_for_expert_training": False}}
            Path(args.report).write_text(json.dumps(payload), encoding="utf-8")


        if __name__ == "__main__":
            main()
        """,
    )


def _write_python_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
