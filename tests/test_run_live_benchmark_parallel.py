import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.run_live_benchmark_parallel import (
    _write_limited_benchmark,
    parse_log_progress,
    simulate_dynamic_worker_claims,
    shard_log_path,
    shard_output_path,
)


class RunLiveBenchmarkParallelTest(unittest.TestCase):
    def test_parse_log_progress_counts_done_and_slowest_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "shard.log"
            log.write_text(
                "\n".join(
                    [
                        "[benchmark] start 1/2 idx=4 domain=all_enzymatic depth=3 target=CCO",
                        "[benchmark] done 1/2 idx=4 domain=all_enzymatic depth=3 routes=5 "
                        "plan=1 type@1=1 type@5=1 stock=True cond=1 compat=1 "
                        "exact_pool=0 gt_pool=1 cand_exact=0 cand_gt=1 elapsed=12.5s error=",
                        "[benchmark] start 2/2 idx=10 domain=all_chemical depth=2 target=CCC",
                        "[benchmark] done 2/2 idx=10 domain=all_chemical depth=2 routes=0 "
                        "plan=0 type@1=0 type@5=0 stock=None cond=0 compat=0 "
                        "exact_pool=0 gt_pool=0 cand_exact=0 cand_gt=0 elapsed=1.2s error=",
                    ]
                )
            )

            progress = parse_log_progress(log)

        self.assertEqual(progress.starts, 2)
        self.assertEqual(progress.dones, 2)
        self.assertEqual(progress.total, 2)
        self.assertEqual(progress.slowest, (12.5, 4, "all_enzymatic"))
        self.assertIn("idx=10", progress.last)

    def test_shard_paths_keep_output_stem(self):
        output = Path("results/run.json")

        self.assertEqual(
            shard_output_path(output, 2, 6),
            Path("results/run_shard2of6.json"),
        )
        self.assertEqual(
            shard_log_path(Path("logs"), output, 2, 6),
            Path("logs/run_shard2of6.log"),
        )

    def test_dynamic_benchmark_queue_completes_all_targets_once(self):
        result = simulate_dynamic_worker_claims(list(range(17)), workers=4)

        self.assertEqual(result.claimed_indices, list(range(17)))
        self.assertEqual(result.duplicate_indices, [])
        self.assertEqual(result.missing_indices, [])

    def test_write_limited_benchmark_applies_global_limit_before_sharding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench.json"
            output = root / "run.json"
            bench.write_text(
                '[{"target_smiles":"A"},{"target_smiles":"B"},{"target_smiles":"C"},{"target_smiles":"D"}]',
                encoding="utf-8",
            )

            limited = _write_limited_benchmark(bench, output=output, limit=3)
            limited_name = limited.name
            limited_rows = json.loads(limited.read_text(encoding="utf-8"))

        self.assertEqual(limited_name, "run_limit3_benchmark.json")
        self.assertEqual([row["target_smiles"] for row in limited_rows], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
