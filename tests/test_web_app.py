import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cascade_planner.web.app as web_app
from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.web.app import (
    _annotate_route_statuses,
    _normalize_planner_mode,
    _plan_depths,
    _plan_failure_diagnosis,
    _plan_output_summary,
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
        self.assertEqual(
            payload["chem_enzy_runtime"]["probe_scope"],
            "filesystem_only_no_process_or_model_execution",
        )
        self.assertIn("template_relevance", payload)
        self.assertIn("available_model_names", payload["template_relevance"])

    def test_production_runtime_status_binds_web_request_selection(self):
        ready = {
            "accepted": True,
            "production_ready": True,
            "python_executable": "python",
            "issues": [],
        }
        with patch.object(
            web_app,
            "diagnose_chem_enzy_runtime",
            return_value=ready,
        ) as diagnose:
            report = web_app._chem_enzy_runtime_status(
                production=True,
                request_payload={
                    "one_step_models": ["template_relevance.reaxys"],
                    "stock_names": ["PaRotes_n5-stock"],
                    "chem_enzy_onmt_tokenizer": "token",
                },
            )

        self.assertIs(report, ready)
        kwargs = diagnose.call_args.kwargs
        self.assertEqual(kwargs["one_step_models"], ["template_relevance.reaxys"])
        self.assertEqual(kwargs["stock_names"], ["PaRotes_n5-stock"])
        self.assertEqual(
            kwargs["model_overrides"],
            {"chem_enzy_onmt_tokenizer": "token"},
        )

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

    def test_route_examples_endpoint_only_exposes_available_local_files(self):
        repository_results = web_app.ROOT / "results"
        repository_results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=repository_results) as td:
            isolated_results = Path(td)
            shared = isolated_results / "shared"
            v2 = isolated_results / "v2"
            available = shared / "known" / "route_forest.html"
            available.parent.mkdir(parents=True)
            available.write_text("<!doctype html><title>route</title>", encoding="utf-8")
            discovered_shared = shared / "novel_target" / "route_forest.html"
            discovered_shared.parent.mkdir(parents=True)
            discovered_shared.write_text("<!doctype html><title>shared</title>", encoding="utf-8")
            discovered_v2 = v2 / "job_002" / "nested" / "route_forest.html"
            discovered_v2.parent.mkdir(parents=True)
            discovered_v2.write_text("<!doctype html><title>v2</title>", encoding="utf-8")
            ignored_html = shared / "not_a_route.html"
            ignored_html.write_text("<!doctype html><title>ignore</title>", encoding="utf-8")
            missing = shared / "missing" / "route_forest.html"
            specs = (
                {"key": "available", "label": "Available", "path": str(available.relative_to(web_app.ROOT))},
                {"key": "missing", "label": "Missing", "path": str(missing.relative_to(web_app.ROOT))},
                {"key": "outside", "label": "Outside", "path": "README.md"},
            )
            with (
                patch.object(web_app, "SHARED_RESULTS_DIR", shared),
                patch.object(web_app, "RESULTS_DIR", v2),
                patch.object(web_app, "ROUTE_EXAMPLE_SPECS", specs),
            ):
                response = self.app.get("/api/route-examples")

        self.assertEqual(response.status_code, 200, response.data)
        payload = response.get_json()
        self.assertEqual(payload["schema_version"], "route_example_availability.v1")
        expected_discovered = sorted(
            (discovered_shared, discovered_v2),
            key=lambda path: str(path.relative_to(web_app.ROOT)).replace("\\", "/").casefold(),
        )
        self.assertEqual(
            [row["path"] for row in payload["examples"]],
            [
                str(available.relative_to(web_app.ROOT)),
                *(str(path.relative_to(web_app.ROOT)) for path in expected_discovered),
            ],
        )
        self.assertEqual(payload["examples"][0]["key"], "available")
        self.assertTrue(all("本地" in row["label"] for row in payload["examples"][1:]))
        self.assertEqual(payload["examples"][0]["path"], str(available.relative_to(web_app.ROOT)))
        self.assertEqual(payload["available_count"], 3)
        self.assertEqual(
            [(row["key"], row["reason"]) for row in payload["unavailable_examples"]],
            [("missing", "artifact_missing"), ("outside", "artifact_outside_results")],
        )
        self.assertTrue(all("path" not in row for row in payload["unavailable_examples"]))
        self.assertFalse(payload["scan"]["truncated"])
        self.assertEqual(payload["scan"]["roots"], [str(shared.relative_to(web_app.ROOT)), str(v2.relative_to(web_app.ROOT))])

    def test_route_examples_endpoint_has_actionable_clean_checkout_fallback(self):
        repository_results = web_app.ROOT / "results"
        repository_results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=repository_results) as td:
            isolated_results = Path(td)
            shared = isolated_results / "shared"
            v2 = isolated_results / "v2"
            shared.mkdir()
            v2.mkdir()
            specs = ({
                "key": "not_cloned",
                "label": "Not cloned",
                "path": str((shared / "not_cloned" / "route_forest.html").relative_to(web_app.ROOT)),
            },)
            with (
                patch.object(web_app, "SHARED_RESULTS_DIR", shared),
                patch.object(web_app, "RESULTS_DIR", v2),
                patch.object(web_app, "ROUTE_EXAMPLE_SPECS", specs),
            ):
                response = self.app.get("/api/route-examples")

        self.assertEqual(response.status_code, 200, response.data)
        payload = response.get_json()
        self.assertEqual(payload["examples"], [])
        self.assertEqual(payload["available_count"], 0)
        self.assertIn("启动 Agent", payload["message"])
        self.assertNotIn("path", payload["unavailable_examples"][0])

    def test_route_examples_discovery_is_bounded(self):
        repository_results = web_app.ROOT / "results"
        repository_results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=repository_results) as td:
            isolated_results = Path(td)
            shared = isolated_results / "shared"
            v2 = isolated_results / "v2"
            shared.mkdir()
            v2.mkdir()
            for index in range(4):
                child = shared / f"run_{index}"
                child.mkdir()
                (child / "route_forest.html").write_text("route", encoding="utf-8")
            with (
                patch.object(web_app, "SHARED_RESULTS_DIR", shared),
                patch.object(web_app, "RESULTS_DIR", v2),
                patch.object(web_app, "ROUTE_EXAMPLE_SPECS", ()),
                patch.object(web_app, "ROUTE_EXAMPLE_MAX_ENTRIES_PER_ROOT", 2),
            ):
                response = self.app.get("/api/route-examples")

        self.assertEqual(response.status_code, 200, response.data)
        payload = response.get_json()
        self.assertTrue(payload["scan"]["truncated"])
        self.assertLessEqual(payload["scan"]["scanned_entries"], 3)

    def test_artifacts_endpoint_filters_worker_traces_and_rejected_artifacts(self):
        with tempfile.TemporaryDirectory(dir=web_app.RESULTS_DIR) as td:
            root = Path(td)
            worker_path = root / "literature_worker_run_record.json"
            worker_path.write_text(json.dumps({
                "schema_version": "worker_run_record.v1",
                "run_id": "worker_run",
                "task_id": "worker_task",
                "case_id": "case",
                "status": "worker_error",
                "backend": "api_json",
                "output_validation": {"accepted": False, "reasons": ["worker_error"]},
            }), encoding="utf-8")
            accepted_path = root / "accepted_artifact.json"
            accepted_path.write_text(json.dumps({
                "schema_version": "case_bundle.v1",
                "artifacts": [{"artifact_type": "EvidenceCard", "validation_status": "accepted"}],
            }), encoding="utf-8")

            worker_response = self.app.get("/api/artifacts", query_string={"filter": "worker_traces"})
            rejected_response = self.app.get("/api/artifacts", query_string={"filter": "rejected"})

        self.assertEqual(worker_response.status_code, 200, worker_response.data)
        worker_rows = worker_response.get_json()["artifacts"]
        self.assertTrue(any(row["name"] == "literature_worker_run_record.json" for row in worker_rows))
        self.assertTrue(all("worker_traces" in row["tags"] for row in worker_rows))
        rejected_rows = rejected_response.get_json()["artifacts"]
        self.assertTrue(any(row["name"] == "literature_worker_run_record.json" for row in rejected_rows))
        self.assertTrue(all("rejected" in row["tags"] for row in rejected_rows))

    def test_artifact_path_is_restricted(self):
        response = self.app.get("/api/artifact?path=/etc/passwd")
        self.assertEqual(response.status_code, 400)

    def test_result_file_head_checks_availability_without_sending_artifact(self):
        web_app.SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=web_app.SHARED_RESULTS_DIR) as td:
            route = Path(td) / "route_forest.html"
            route.write_text("<!doctype html><title>route</title>", encoding="utf-8")
            route_size = route.stat().st_size
            relative = str(route.relative_to(web_app.ROOT))
            response = self.app.head("/api/result-file", query_string={"path": relative})
            missing_response = self.app.head(
                "/api/result-file",
                query_string={"path": str((Path(td) / "missing.html").relative_to(web_app.ROOT))},
            )

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data, b"")
        self.assertIn("text/html", response.content_type)
        self.assertEqual(response.content_length, route_size)
        self.assertEqual(missing_response.status_code, 404)

    def test_presentation_showcase_exposes_only_available_shared_html(self):
        repository_results = web_app.ROOT / "results"
        repository_results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=repository_results) as td:
            shared = Path(td) / "shared"
            shared.mkdir(parents=True)
            available = shared / "case" / "route_forest.html"
            available.parent.mkdir(parents=True)
            available.write_text("<!doctype html><title>case</title>", encoding="utf-8")
            audit = shared / "audit" / "index.html"
            audit.parent.mkdir(parents=True)
            audit.write_text("<!doctype html><title>audit</title>", encoding="utf-8")
            manifest = shared / "presentation" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "autoplanner.presentation_showcase.v1",
                        "standard_case_id": "closed",
                        "standard": {"summary": "closed-route standard"},
                        "statin_catalog": {
                            "entity_count": 12,
                            "readiness_summary": {"long_route_observed": 1},
                            "targets": [
                                {
                                    "target_name": "atorvastatin",
                                    "route_readiness": {
                                        "readiness": "candidate_long_route",
                                        "confidence": "low",
                                        "warning_codes": ["NO_CONDITION_OBSERVATIONS"],
                                        "observations": {
                                            "max_route_steps": 11,
                                            "best_proof_level": 0,
                                            "condition_observation_edge_count": 0,
                                            "reported_source_ref_count": 0,
                                        },
                                    },
                                }
                            ],
                        },
                        "audits": [
                            {
                                "audit_id": "statin-v4-live-rerun",
                                "artifact_path": str(audit.relative_to(web_app.ROOT)),
                            }
                        ],
                        "cases": [
                            {
                                "case_id": "closed",
                                "artifact_path": str(available.relative_to(web_app.ROOT)),
                                "route_closed": True,
                            },
                            {
                                "case_id": "missing",
                                "artifact_path": str(
                                    (shared / "missing" / "route_forest.html").relative_to(web_app.ROOT)
                                ),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(web_app, "SHARED_RESULTS_DIR", shared),
                patch.object(web_app, "PRESENTATION_SHOWCASE_PATH", manifest),
            ):
                page = self.app.get("/showcase")
                response = self.app.get("/api/showcase")

        self.assertEqual(page.status_code, 200, page.data)
        self.assertIn(b"/api/v4/showcase", page.data)
        self.assertIn(b".empty[hidden]{display:none}", page.data)
        self.assertIn(b"openStatins", page.data)
        self.assertIn("他汀路线就绪度".encode(), page.data)
        self.assertIn(b"NO_CONDITION_OBSERVATIONS", page.data)
        self.assertEqual(response.status_code, 200, response.data)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["cases"][0]["available"])
        self.assertEqual(payload["statin_catalog"]["entity_count"], 12)
        self.assertEqual(
            payload["statin_catalog"]["targets"][0]["target_name"],
            "atorvastatin",
        )
        self.assertEqual(
            payload["statin_catalog"]["targets"][0]["route_readiness"]["confidence"],
            "low",
        )
        self.assertTrue(payload["audits"][0]["available"])
        self.assertIn("/api/result-file?path=", payload["audits"][0]["artifact_url"])
        self.assertIn("/api/result-file?path=", payload["cases"][0]["artifact_url"])
        self.assertFalse(payload["cases"][1]["available"])
        self.assertEqual(payload["cases"][1]["artifact_path"], "")

    def test_agent_case_audit_worker_policy_and_final_report_api_smoke(self):
        web_app.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        with tempfile.TemporaryDirectory(dir=web_app.RESULTS_DIR) as td:
            case_response = self.app.post(
                "/api/cases",
                json={
                    "target_smiles": target,
                    "target_name": "bufotalin_api_case",
                    "family_hint": "bufotalin, bufadienolide, steroid, pyrone",
                    "frontier_smiles": target,
                    "output_dir": td,
                    "query_budget": 4,
                    "literature_backend": "local",
                },
            )
            self.assertEqual(case_response.status_code, 200, case_response.data)
            case_payload = case_response.get_json()
            self.assertTrue(case_payload["ok"], case_payload)
            artifacts = case_payload["artifacts"]

            inspect_response = self.app.get("/api/blackboard", query_string={"case_bundle": artifacts["case_bundle"]})
            self.assertEqual(inspect_response.status_code, 200, inspect_response.data)
            inspect_payload = inspect_response.get_json()
            self.assertEqual(inspect_payload["route_status"], "partial_anchor")
            self.assertIn("EvidenceCardList", inspect_payload["artifact_types"])

            audit_response = self.app.post(
                "/api/route-audit",
                json={
                    "package_path": artifacts["hybrid_route_package"],
                    "validation_path": artifacts["validation"],
                },
            )
            self.assertEqual(audit_response.status_code, 200, audit_response.data)
            audit_payload = audit_response.get_json()
            self.assertEqual(audit_payload["route_status"], "partial_anchor")
            self.assertIn("audit", audit_payload["final_report"])
            self.assertIn("failure_events", audit_payload["final_report"])

            worker_response = self.app.post(
                "/api/worker-trace",
                json={
                    "task": {
                        "schema_version": "worker_task.v1",
                        "task_id": "api_worker",
                        "case_id": case_payload["case_id"],
                        "task_type": "stuck_node_research",
                        "required_artifact_type": "ResearchReport",
                        "input_refs": ["frontier_report"],
                        "allowed_tools": ["local_search"],
                        "budget": {"timeout_s": 5, "max_output_bytes": 20000, "max_tool_calls": 2, "max_worker_runs": 1},
                        "dry_run": True,
                    }
                },
            )
            self.assertEqual(worker_response.status_code, 200, worker_response.data)
            worker_payload = worker_response.get_json()
            self.assertTrue(worker_payload["ok"], worker_payload)
            self.assertEqual(worker_payload["worker_trace"]["status"], "accepted_draft")

            policy_response = self.app.post(
                "/api/guided-policy",
                json={"case_bundle": artifacts["case_bundle"], "target_smiles": target},
            )
            self.assertEqual(policy_response.status_code, 200, policy_response.data)
            policy_payload = policy_response.get_json()
            self.assertTrue(policy_payload["ok"], policy_payload)
            self.assertIn("chem_enzy_search_policy", policy_payload["guided_request_payload"])
            self.assertEqual(policy_payload["rerun_history"]["policy_id"], policy_payload["policy"]["policy_id"])

            report_response = self.app.get("/api/final-report", query_string={"case_bundle": artifacts["case_bundle"]})
            self.assertEqual(report_response.status_code, 200, report_response.data)
            report_payload = report_response.get_json()
            self.assertEqual(report_payload["route_status"], "partial_anchor")
            self.assertIn("audit", report_payload)
            self.assertIn("evidence_refs", report_payload)
            self.assertIn("condition", report_payload)
            self.assertIn("rerun_history", report_payload)

    def test_worker_trace_api_defaults_to_codex_backend(self):
        record = WorkerRunRecord(
            run_id="api_codex_worker:run",
            task_id="api_codex_worker",
            case_id="case",
            status="accepted_draft",
            backend="codex_cli",
            command=["codex", "exec", "-"],
            output_validation={"accepted": True, "reasons": []},
            output_artifact={
                "schema_version": "researchreport.draft.v1",
                "artifact_id": "api_codex_worker:ResearchReport",
                "artifact_type": "ResearchReport",
                "case_id": "case",
                "source": "codex_cli",
                "input_refs": ["target_profile"],
                "evidence_refs": [],
                "validation_status": "draft",
            },
        )
        with patch.dict(web_app.os.environ, {"AUTOPLANNER_WEB_ENABLE_REAL_WORKER_TRACE": "1"}):
            with patch("cascade_planner.web.app.run_codex_worker", return_value=record) as run_worker:
                response = self.app.post(
                    "/api/worker-trace",
                    json={
                        "task": {
                            "schema_version": "worker_task.v1",
                            "task_id": "api_codex_worker",
                            "case_id": "case",
                            "task_type": "target_research",
                            "required_artifact_type": "ResearchReport",
                            "input_refs": ["target_profile"],
                            "budget": {"timeout_s": 5, "max_output_bytes": 20000, "max_tool_calls": 2, "max_worker_runs": 1},
                            "dry_run": False,
                        }
                    },
                )

        self.assertEqual(response.status_code, 200, response.data)
        payload = response.get_json()
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["worker_trace"]["backend"], "codex_cli")
        self.assertEqual(payload["worker_trace"]["command"][:2], ["codex", "exec"])
        self.assertTrue(run_worker.call_args.kwargs["use_codex_cli"])

    def test_codex_plan_rejects_http_runtime_controls_before_queueing(self):
        before = set(web_app._JOBS)
        response = self.app.post(
            "/api/plan-jobs",
            json={
                "planner_backend": "codex_fullflow",
                "target_smiles": "CCO",
                "key_path": "key.txt",
                "base_url": "https://attacker.invalid/v1",
                "codex_worker_sandbox": "bypassed",
            },
        )
        self.assertEqual(response.status_code, 400, response.data)
        self.assertEqual(set(web_app._JOBS), before)

    def test_mutating_api_rejects_non_json_cross_site_requests(self):
        non_json = self.app.post(
            "/api/plan-jobs",
            data='{"planner_backend":"codex_fullflow","target_smiles":"CCO"}',
            content_type="text/plain",
        )
        self.assertEqual(non_json.status_code, 415, non_json.data)
        cross_site = self.app.post(
            "/api/plan-jobs",
            json={"planner_backend": "codex_fullflow", "target_smiles": "CCO"},
            headers={"Origin": "https://attacker.invalid", "Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(cross_site.status_code, 403, cross_site.data)

    def test_real_worker_trace_is_disabled_by_default(self):
        with patch.dict(web_app.os.environ, {}, clear=False):
            web_app.os.environ.pop("AUTOPLANNER_WEB_ENABLE_REAL_WORKER_TRACE", None)
            response = self.app.post(
                "/api/worker-trace",
                json={
                    "task": {
                        "schema_version": "worker_task.v1",
                        "task_id": "blocked_worker",
                        "case_id": "case",
                        "task_type": "target_research",
                        "required_artifact_type": "ResearchReport",
                        "input_refs": ["target_profile"],
                        "budget": {"timeout_s": 5, "max_output_bytes": 20000, "max_tool_calls": 2, "max_worker_runs": 1},
                    }
                },
            )
        self.assertEqual(response.status_code, 403, response.data)

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

    def test_web_static_payload_includes_rule_verifier_gate_toggle(self):
        app_js = (web_app.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        index_html = (web_app.STATIC_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="enable-rule-verifier-gate"', index_html)
        self.assertIn("enable_rule_verifier_gate", app_js)
        self.assertIn("cascadeVerifierGateSummary", app_js)
        self.assertIn('id="enable-learned-verifier-annotation"', index_html)
        self.assertIn("enable_learned_verifier_annotation", app_js)
        self.assertIn("learnedVerifierAnnotationSummary", app_js)
        self.assertIn('id="product-audit-filter-mode"', index_html)
        self.assertIn('value="risk_guarded" selected', index_html)
        self.assertIn("product_audit_filter_mode: auditMode", app_js)
        self.assertIn('auditMode !== "off"', app_js)
        self.assertIn('id="proposal-gate-mode"', index_html)
        self.assertIn('value="hard_reject" selected', index_html)
        self.assertIn("proposal_gate_mode: proposalGateMode", app_js)
        self.assertIn('proposalGateMode !== "off"', app_js)
        self.assertIn('id="one-step-model-mode"', index_html)
        self.assertIn("selectedOneStepModels", app_js)
        self.assertIn("one_step_models: oneStepModels", app_js)
        self.assertIn("renderTemplateRelevanceStatus", app_js)
        self.assertIn('value="template_available" selected', index_html)

    def test_agent_workbench_describes_complete_trust_dag_and_validated_replacements(self):
        agent_html = (web_app.STATIC_DIR / "agent.html").read_text(encoding="utf-8")
        agent_js = (web_app.STATIC_DIR / "agent.js").read_text(encoding="utf-8")

        self.assertIn("全路径依赖图、可信度与安全备选", agent_html)
        self.assertIn("颜色表示 proof tier", agent_html)
        self.assertIn("线宽表示独立支持", agent_html)
        self.assertIn("已通过后端 AND/OR", agent_html)
        self.assertIn("整路线重验的替换分支可预览", agent_html)
        self.assertIn("接口比较仅作诊断", agent_html)
        self.assertIn("预览不建立父路线证明", agent_html)
        self.assertIn('value="standard" selected', agent_html)
        self.assertIn("最多 8 次有效扩展", agent_html)
        self.assertIn("不代表商业可采购", agent_html)
        self.assertIn("const RUN_PROFILES", agent_js)
        self.assertIn('run_profile: $("run-profile").value || "standard"', agent_js)

        response = self.app.get("/agent")
        self.assertEqual(response.status_code, 200)
        self.assertIn("20260711-science-workbench-v3", response.get_data(as_text=True))

    def test_codex_profiles_and_server_owned_benchmark_stock_are_explicit(self):
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["rounds"], 4)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["depth"], 2)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["accepted_expansions"], 8)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["attempt_runs"], 12)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["chem_enzy_runs"], 1)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["child_target_runs"], 2)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["codex_research_runs"], 1)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["scout_calls"], 1)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["standard"]["visual_calls"], 1)
        self.assertEqual(web_app.CODEX_RUN_PROFILES["deep"]["per_invocation"], 1)
        standard_budget = web_app._codex_fullflow_budget(
            {},
            timeout_s=7200.0,
            profile=web_app.CODEX_RUN_PROFILES["standard"],
        )
        self.assertEqual(standard_budget.max_chem_enzy_runs, 1)
        self.assertEqual(standard_budget.max_guided_chemenzy_runs, 1)
        self.assertEqual(standard_budget.max_route_expansion_subgoal_runs, 2)
        self.assertEqual(standard_budget.max_codex_research_runs, 1)
        self.assertEqual(standard_budget.max_scout_calls, 1)
        self.assertEqual(standard_budget.max_visual_calls, 1)
        agent_html = (web_app.STATIC_DIR / "agent.html").read_text(encoding="utf-8")
        agent_js = (web_app.STATIC_DIR / "agent.js").read_text(encoding="utf-8")
        self.assertIn('id="codex-team-attempts"', agent_html)
        self.assertIn("codex_agent_team_max_attempt_runs", agent_js)
        self.assertIn('id="closure-objective"', agent_html)
        self.assertIn('value="benchmark_search" selected', agent_html)
        self.assertIn('id="exploration-mode"', agent_html)
        self.assertIn('value="exhaustive" selected', agent_html)
        self.assertIn("codex_agent_team_closure_objective", agent_js)
        self.assertIn("codex_agent_team_exploration_mode", agent_js)
        self.assertIn('id="research-runs"', agent_html)
        self.assertIn('max_codex_research_runs: Number($("research-runs").value || 0)', agent_js)
        catalog = web_app._trusted_benchmark_stock_catalog()
        self.assertEqual(catalog["name"], "PaRoutes_n1")
        self.assertEqual(catalog["boundary_type"], "benchmark_stock")
        self.assertIs(catalog["commercial_orderability_claimed"], False)
        self.assertTrue(Path(catalog["artifact"]).is_file())

    def test_agent_workbench_layout_accessibility_and_embed_contracts(self):
        agent_html = (web_app.STATIC_DIR / "agent.html").read_text(encoding="utf-8")
        agent_css = (web_app.STATIC_DIR / "agent.css").read_text(encoding="utf-8")
        agent_js = (web_app.STATIC_DIR / "agent.js").read_text(encoding="utf-8")

        for control_id in (
            "toggle-controls",
            "toggle-activity",
            "focus-route",
            "mobile-view-route",
            "mobile-view-controls",
            "mobile-view-activity",
        ):
            self.assertIn(f'id="{control_id}"', agent_html)
        self.assertIn('class="skip-link"', agent_html)
        self.assertIn('role="tablist"', agent_html)
        self.assertIn('aria-controls="controls-panel"', agent_html)
        self.assertIn('aria-controls="activity-panel"', agent_html)
        self.assertIn('aria-live="polite"', agent_html)
        self.assertIn("<fieldset", agent_html)
        self.assertIn("<legend>", agent_html)

        self.assertIn('sandbox="allow-scripts"', agent_html)
        self.assertNotIn('sandbox="allow-scripts allow-same-origin"', agent_html)
        self.assertIn("&embed=1", agent_js)
        self.assertIn("parent_token=", agent_js)
        self.assertIn("autoplanner.route_forest.ready.v1", agent_js)
        self.assertIn("handleRouteFrameMessage", agent_js)
        self.assertIn("ROUTE_HANDSHAKE_TIMEOUT_MS", agent_js)
        self.assertIn('message.integrity_status !== "verified"', agent_js)
        self.assertNotIn('message.integrity_status === "invalid"', agent_js)
        self.assertNotIn("contentDocument", agent_js)
        self.assertNotIn("verifyRouteFile", agent_js)
        self.assertIn('api("/api/route-examples")', agent_js)
        self.assertIn("routeExamplesToken", agent_js)
        self.assertIn("inputRevision !== state.routeInputRevision", agent_js)
        self.assertIn("Boolean(state.currentJobId)", agent_js)
        self.assertIn('if (!preserveRouteInput) input.value = "";', agent_js)
        self.assertIn('$("existing-route").value = routePath;', agent_js)
        self.assertIn("loadRouteExamples({ preserveRouteInput: restored })", agent_js)
        self.assertIn('method: "HEAD"', agent_js)
        self.assertIn("此 checkout 暂无本地路线图", agent_js)

        self.assertIn('const LAYOUT_KEY = "autoplanner.agent.layout.v2"', agent_js)
        self.assertIn("localStorage.setItem(LAYOUT_KEY", agent_js)
        self.assertIn("controls-collapsed", agent_css)
        self.assertIn("activity-collapsed", agent_css)
        self.assertIn('@media (max-width: 1240px)', agent_css)
        self.assertIn('@media (max-width: 900px)', agent_css)
        self.assertIn('@media (max-width: 640px)', agent_css)
        self.assertIn('@media (prefers-reduced-motion: reduce)', agent_css)
        self.assertIn(":focus-visible", agent_css)
        self.assertNotIn("min-width: 1180px", agent_css)

        self.assertIn('id="target-name" value="paclitaxel"', agent_html)
        self.assertIn('data-sample="artemisinin"', agent_html)
        self.assertIn('id="demo-route" aria-describedby="existing-route-hint" disabled', agent_html)
        self.assertIn('id="existing-route" value=""', agent_html)
        self.assertNotIn("results/shared/full_rerun_advisory_visual_20260702", agent_html)
        self.assertNotIn("results/shared/paclitaxel_architecture_v2_20260710", agent_html)
        self.assertNotIn("bufotalin solved mixed route", agent_html)
        self.assertNotIn("atorvastatin_online_zero_20260704", agent_html)

    def test_missing_template_relevance_selection_is_rejected_before_search(self):
        missing_model = "template_relevance.autoplanner_missing_for_test"
        self.assertNotIn(
            missing_model,
            web_app._template_relevance_status().get("available_model_names") or [],
        )
        payload = {
            "target_smiles": "CCO",
            "one_step_models": [missing_model],
        }

        with self.app.post("/api/plan", json=payload) as response:
            self.assertEqual(response.status_code, 400)
            self.assertIn(b"missing local template_relevance", response.data)

    def test_plan_output_summary_reports_cascade_verifier_gate(self):
        summary = _plan_output_summary(
            {
                "time_s": 3.2,
                "routes": [{"route_rank": 0}],
                "search_status": {"status": "filtered", "message": "gate", "solved": False},
                "failure_analysis": {"failure_categories": ["cascade_verifier_filtered_all"]},
                "ui_metadata": {
                    "saved_at": "results/v2/plan.json",
                    "raw_saved_at": "results/v2/plan_raw.json",
                },
                "route_set_metrics": {
                    "cascade_verifier_gate": {
                        "enabled": True,
                        "input_routes": 3,
                        "kept_routes": 1,
                        "dropped_routes": 2,
                    },
                    "learned_verifier_annotation": {
                        "enabled": True,
                        "model_loaded": True,
                        "input_routes": 3,
                        "annotated_routes": 1,
                        "policy": "annotation_only",
                    },
                },
            }
        )

        self.assertEqual(summary["status"], "filtered")
        self.assertEqual(summary["routes"], 1)
        self.assertEqual(summary["output_json"], "results/v2/plan.json")
        self.assertTrue(summary["cascade_verifier_gate"]["enabled"])
        self.assertEqual(summary["cascade_verifier_gate"]["input_routes"], 3)
        self.assertEqual(summary["cascade_verifier_gate"]["kept_routes"], 1)
        self.assertEqual(summary["cascade_verifier_gate"]["dropped_routes"], 2)
        self.assertTrue(summary["learned_verifier_annotation"]["enabled"])
        self.assertTrue(summary["learned_verifier_annotation"]["model_loaded"])
        self.assertEqual(summary["learned_verifier_annotation"]["input_routes"], 3)
        self.assertEqual(summary["learned_verifier_annotation"]["annotated_routes"], 1)
        self.assertEqual(summary["learned_verifier_annotation"]["policy"], "annotation_only")

    def test_run_plan_job_logs_failure_analysis_without_marking_failed(self):
        output = {
            "time_s": 3.2,
            "routes": [],
            "search_status": {"status": "filtered", "message": "filtered", "solved": False},
            "failure_analysis": {
                "failure_categories": ["product_audit_filtered_all"],
                "diagnosis": ["ChemEnzy returned candidates, but product-audit removed all."],
                "retry_suggestions": ["inspect rejected diagnostic routes"],
            },
            "ui_metadata": {
                "saved_at": "results/v2/plan.json",
                "request_path": "results/v2/request.json",
                "raw_saved_at": "results/v2/plan_raw.json",
                "rejected_saved_at": "results/v2/plan_rejected.json",
            },
        }
        job_id = "plan_failure_analysis_log_test"
        original_jobs = dict(web_app._JOBS)
        try:
            with tempfile.TemporaryDirectory(dir=web_app.ROOT) as td:
                log_path = Path(td) / "job.log"
                with web_app._LOCK:
                    web_app._JOBS.clear()
                    web_app._JOBS[job_id] = {"kind": "plan", "status": "queued", "payload": {}}

                with patch.object(web_app, "_run_plan", return_value=output):
                    web_app._run_plan_job(job_id, {}, log_path)

                with web_app._LOCK:
                    job = dict(web_app._JOBS[job_id])
                log_text = log_path.read_text(encoding="utf-8")
        finally:
            with web_app._LOCK:
                web_app._JOBS.clear()
                web_app._JOBS.update(original_jobs)

        self.assertEqual(job["status"], "complete")
        self.assertIsNone(job["error"])
        self.assertEqual(job["summary"]["status"], "filtered")
        self.assertIn("failure_analysis=ChemEnzy returned candidates", log_text)
        self.assertIn("retry_suggestions=inspect rejected diagnostic routes", log_text)

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
