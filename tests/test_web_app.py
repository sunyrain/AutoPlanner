import json
import tempfile
import unittest
from pathlib import Path

import cascade_planner.web.app as web_app
from cascade_planner.web.app import (
    _annotate_route_statuses,
    _normalize_planner_mode,
    _plan_depths,
    _plan_failure_diagnosis,
    _plan_search_status,
    _payload_has_solved_route,
    create_app,
)


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()

    def test_status_endpoint(self):
        response = self.app.get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("cuda", payload)

    def test_molecule_svg_endpoint(self):
        response = self.app.get("/api/mol.svg?smiles=CCO")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/svg+xml", response.content_type)
        self.assertIn(b"<svg", response.data)

    def test_artifacts_endpoint(self):
        response = self.app.get("/api/artifacts")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("artifacts", payload)

    def test_artifact_path_is_restricted(self):
        response = self.app.get("/api/artifact?path=/etc/passwd")
        self.assertEqual(response.status_code, 400)

    def test_save_native_raw_output_writes_independent_sidecar(self):
        with tempfile.TemporaryDirectory(dir=web_app.ROOT) as td:
            raw_path = Path(td) / "plan_raw.json"
            output = {
                "ui_metadata": {"saved_at": "results/v2/plan.json"},
                "routes": [{"score": 1.0, "steps": []}],
            }

            web_app._save_native_raw_output(output, raw_path)
            output["routes"][0]["score"] = 2.0

            saved = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["routes"][0]["score"], 1.0)
            self.assertTrue(saved["ui_metadata"]["saved_at"].endswith("plan_raw.json"))
            self.assertEqual(saved["ui_metadata"]["filtered_saved_at"], "results/v2/plan.json")

    def test_adaptive_depth_range_defaults_to_sweep(self):
        mode, depths = _plan_depths({"min_steps": 3, "max_steps": 5})

        self.assertEqual(mode, "adaptive")
        self.assertEqual(depths, [3, 4, 5])

    def test_fixed_depth_mode_uses_single_n_steps(self):
        mode, depths = _plan_depths({"search_mode": "fixed", "n_steps": 6, "min_steps": 3, "max_steps": 8})

        self.assertEqual(mode, "fixed")
        self.assertEqual(depths, [6])

    def test_depth_range_rejects_more_steps_than_model_slots(self):
        with self.assertRaises(Exception) as ctx:
            _plan_depths({"search_mode": "adaptive", "min_steps": 3, "max_steps": 10})

        self.assertIn("skeleton model supports at most 8 slots", str(ctx.exception))

    def test_planner_mode_aliases_are_integrated_into_advanced(self):
        for value in ("advanced", "hybrid", "and_or", "stock_and_or", "cascade"):
            normalized, requested = _normalize_planner_mode(value)
            self.assertEqual(normalized, "advanced")
            self.assertTrue(requested)

        with self.assertRaises(Exception) as ctx:
            _normalize_planner_mode("legacy_debug_mode")

        self.assertIn("planner_mode must be advanced", str(ctx.exception))

    def test_failed_plan_diagnosis_separates_filled_from_solved(self):
        route = {
            "metrics": {
                "filled_route": True,
                "progressive_route": False,
                "route_solved": False,
                "strict_stock_solve": False,
                "retrosynthesis_progress": {
                    "main_chain_reduction": 0.0,
                    "progressive_step_fraction": 0.0,
                    "terminal_simplified": False,
                    "leaf_simplified": False,
                },
                "route_naturalness": {"naturalness_score": 1.0},
                "cascade_compatibility": {"issues": []},
            }
        }
        attempts = [{"depth": 3, "n_routes": 1, "best": {"filled_route": True, "progressive_route": False}}]

        diagnosis = _plan_failure_diagnosis([route], attempts)

        self.assertIn("insufficient_retrosynthesis_progress", diagnosis)
        self.assertIn("main_chain_not_reduced", diagnosis)
        self.assertIn("insufficient_stepwise_disconnection", diagnosis)
        self.assertIn("largest_leaf_reactant_still_complex", diagnosis)
        self.assertIn("terminal_reactants_not_all_in_stock", diagnosis)
        self.assertIn("no_solved_route_within_depth_range", diagnosis)

    def test_plan_search_status_reports_partial_before_solved(self):
        payload = {
            "routes": [{
                "n_steps": 5,
                "score": 1.0,
                "metrics": {
                    "filled_route": True,
                    "progressive_route": True,
                    "route_solved": False,
                    "strict_stock_solve": False,
                    "retrosynthesis_progress": {"main_chain_reduction": 0.5},
                    "route_naturalness": {"naturalness_score": 1.0},
                    "cascade_compatibility": {"cascade_compatibility_success": True, "issues": []},
                },
            }]
        }

        status = _plan_search_status(payload, [{"depth": 5}], mode="adaptive", stopped_on_solved=False)

        self.assertEqual(status["status"], "partial")
        self.assertFalse(status["solved"])
        self.assertTrue(status["progressive"])
        self.assertEqual(status["best_depth"], 5)

    def test_stock_closed_non_progressive_route_is_diagnostic(self):
        payload = {
            "routes": [{
                "n_steps": 3,
                "score": 0.8,
                "metrics": {
                    "filled_route": True,
                    "progressive_route": False,
                    "route_solved": True,
                    "strict_stock_solve": True,
                    "retrosynthesis_progress": {
                        "main_chain_reduction": 0.0,
                        "progressive_step_fraction": 0.0,
                        "terminal_simplified": False,
                        "leaf_simplified": False,
                    },
                    "route_naturalness": {"naturalness_score": 1.0},
                    "cascade_compatibility": {"cascade_compatibility_success": False, "issues": []},
                },
            }]
        }
        _annotate_route_statuses(payload["routes"])
        attempts = [{
            "depth": 3,
            "n_routes": 1,
            "best": {
                "filled_route": True,
                "progressive_route": False,
                "route_solved": True,
                "professional_solved": False,
                "diagnostic_solved": True,
            },
        }]

        status = _plan_search_status(payload, attempts, mode="adaptive", stopped_on_solved=False)
        diagnosis = _plan_failure_diagnosis(payload["routes"], attempts)

        self.assertFalse(_payload_has_solved_route(payload))
        self.assertEqual(status["status"], "diagnostic")
        self.assertFalse(status["solved"])
        self.assertTrue(status["stock_closed"])
        self.assertTrue(payload["routes"][0]["metrics"]["diagnostic_solved"])
        self.assertIn("diagnostic_stock_closed_but_not_progressive", diagnosis)
        self.assertIn("insufficient_retrosynthesis_progress", diagnosis)
        self.assertIn("no_solved_route_within_depth_range", diagnosis)

    def test_cancel_queued_plan_job_removes_it_from_queue(self):
        with web_app._LOCK:
            web_app._JOBS.clear()
            web_app._PLAN_JOB_QUEUE.clear()
            web_app._PLAN_PROCESS_BY_JOB.clear()
            web_app._PLAN_CURRENT_JOB_ID = None
            web_app._JOBS["plan_cancel_a"] = {
                "job_id": "plan_cancel_a",
                "kind": "plan",
                "status": "queued",
                "log_path": "results/v2/ui_jobs/plan_cancel_a.log",
                "created_at": "2026-05-18T00:00:00Z",
            }
            web_app._JOBS["plan_cancel_b"] = {
                "job_id": "plan_cancel_b",
                "kind": "plan",
                "status": "queued",
                "log_path": "results/v2/ui_jobs/plan_cancel_b.log",
                "created_at": "2026-05-18T00:00:01Z",
            }
            web_app._PLAN_JOB_QUEUE.extend(["plan_cancel_a", "plan_cancel_b"])
            web_app._refresh_plan_queue_positions_locked()

        cancelled = web_app._cancel_job("plan_cancel_a")

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["summary"]["status"], "cancelled")
        with web_app._LOCK:
            self.assertEqual(web_app._JOBS["plan_cancel_b"]["queue_position"], 1)
            self.assertNotIn("plan_cancel_a", list(web_app._PLAN_JOB_QUEUE))


if __name__ == "__main__":
    unittest.main()
