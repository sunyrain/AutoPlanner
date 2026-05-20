import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor_autoplanner_web.py"
SPEC = importlib.util.spec_from_file_location("monitor_autoplanner_web", MODULE_PATH)
monitor = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(monitor)


class MonitorAutoPlannerWebTest(unittest.TestCase):
    def test_snapshot_summarizes_status_and_jobs(self):
        def fake_fetch(url):
            if url.endswith("/api/status"):
                return {
                    "cuda": {
                        "available": True,
                        "devices": [
                            {"index": 0, "memory_used_mb": 10, "memory_total_mb": 100},
                        ],
                    }
                }
            if url.endswith("/api/jobs"):
                return {
                    "jobs": [
                        {
                            "job_id": "plan_a",
                            "status": "complete",
                            "elapsed_s": 12.3,
                            "output_json": "results/v2/a.json",
                            "rejected_output_json": "results/v2/a_rejected.json",
                            "summary": {"routes": 4},
                        },
                        {
                            "job_id": "plan_b",
                            "status": "queued",
                            "summary": {},
                        },
                    ]
                }
            raise AssertionError(url)

        with patch.object(monitor, "_fetch_json", side_effect=fake_fetch):
            text = monitor.snapshot("http://127.0.0.1:7991", n_jobs=2)

        self.assertIn("OK http://127.0.0.1:7991", text)
        self.assertIn('"complete": 1', text)
        self.assertIn('"queued": 1', text)
        self.assertIn("gpu0=10/100MB", text)
        self.assertIn("plan_a status=complete routes=4", text)
        self.assertIn("rejected=results/v2/a_rejected.json", text)

    def test_snapshot_reports_down_state(self):
        with patch.object(monitor, "_fetch_json", side_effect=RuntimeError("connection refused")):
            text = monitor.snapshot("http://127.0.0.1:7991")

        self.assertIn("DOWN http://127.0.0.1:7991", text)
        self.assertIn("connection refused", text)


if __name__ == "__main__":
    unittest.main()

