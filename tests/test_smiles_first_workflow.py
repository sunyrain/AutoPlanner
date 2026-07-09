import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow
from cascade_planner.agent.target_profile import build_frontier_report, build_target_profile
from cascade_planner.baselines.chem_enzy_adapter import DEFAULT_ONE_STEP_MODELS
from cascade_planner.baselines.route_contract import RouteSearchConfig


class SmilesFirstWorkflowTest(unittest.TestCase):
    def test_target_as_initial_frontier_does_not_claim_complexity_failure(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        profile = build_target_profile(target, target_name="bufadienolide_like", family_hint="steroid")
        report = build_frontier_report(profile, frontier_smiles=target)

        self.assertTrue(profile.valid)
        self.assertGreaterEqual(profile.rings, 4)
        self.assertIn("polycyclic_or_steroid_like", profile.family_hints)
        self.assertFalse(report["advanced_frontier_found"])
        self.assertEqual(report["frontiers"][0]["frontier_role"], "target_as_initial_frontier")
        self.assertIn("target_as_initial_frontier", report["frontiers"][0]["flags"])
        self.assertNotIn("no_complexity_drop", report["frontiers"][0]["flags"])

    def test_full_workflow_writes_required_artifacts_and_guarded_status(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="bufadienolide_like",
                    family_hint="bufadienolide, steroid, pyrone",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=5,
                    literature_backend="local",
                )
            )
            out = Path(tmp)
            required = [
                "target_profile.json",
                "baseline_routes.json",
                "frontier_report.json",
                "literature_search_report.md",
                "evidence_cards.jsonl",
                "validation.json",
                "summary.md",
            ]

            for name in required:
                self.assertTrue((out / name).exists(), name)
            candidate_path = Path(result["artifacts"]["literature_candidates"])
            package_path = Path(result["artifacts"]["hybrid_route_package"])
            route_map_path = Path(result["artifacts"]["route_map"])
            self.assertTrue(candidate_path.exists())
            self.assertTrue(package_path.exists())
            self.assertTrue(route_map_path.exists())

            validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
            package = json.loads(package_path.read_text(encoding="utf-8"))
            kinds = {row["candidate_kind"] for row in package["literature_candidates"]}

        self.assertTrue(validation["accepted"], validation)
        self.assertEqual(validation["route_status"], "partial_anchor")
        self.assertNotEqual(package["route_status"], "solved")
        self.assertIn("exact_fragment_retro", kinds)
        self.assertIn("forward_surrogate", kinds)
        self.assertIn("route_anchor", kinds)
        self.assertTrue(validation["guards"]["route_anchor_not_stock"])
        self.assertTrue(validation["guards"]["forward_surrogate_not_lab_procedure"])

    def test_invalid_target_stops_before_literature_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles="not_a_smiles",
                    target_name="bad",
                    output_dir=tmp,
                    literature_backend="local",
                )
            )
            validation = json.loads((Path(tmp) / "validation.json").read_text(encoding="utf-8"))

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["route_status"], "invalid_package")
        self.assertIn("invalid_target_smiles", validation["reasons"])
        self.assertIn("validation", result)

    def test_baseline_json_is_preserved_and_frontier_is_extracted(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        baseline = {
            "schema_version": "baseline_routes.v1",
            "status": "provided",
            "solved": False,
            "routes": [
                {
                    "route_id": "late_decoration_only",
                    "ordinary_steps": ["O-acetylation"],
                    "unresolved_frontiers": [target],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="bufadienolide_like",
                    family_hint="bufadienolide, steroid, pyrone",
                    baseline_json=baseline_path,
                    output_dir=Path(tmp) / "out",
                    query_budget=3,
                    literature_backend="local",
                )
            )
            saved_baseline = json.loads((Path(tmp) / "out" / "baseline_routes.json").read_text(encoding="utf-8"))
            frontier_report = json.loads((Path(tmp) / "out" / "frontier_report.json").read_text(encoding="utf-8"))

        self.assertEqual(saved_baseline["routes"][0]["route_id"], "late_decoration_only")
        self.assertTrue(frontier_report["advanced_frontier_found"])
        self.assertEqual(frontier_report["frontiers"][0]["frontier_role"], "route_audit_frontier")
        self.assertIn("no_complexity_drop", frontier_report["frontiers"][0]["flags"])
        self.assertIn("ordinary_decoration_only", frontier_report["frontiers"][0]["flags"])
        self.assertIn("unresolved_core", frontier_report["frontiers"][0]["flags"])

    def test_cli_direct_execution_writes_artifacts(self):
        target = "O=C1CCCCCCCCCCCCO1"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_smiles_first_literature_workflow.py",
                    "--target-smiles",
                    target,
                    "--target-name",
                    "macro_lactone_cli",
                    "--family-hint",
                    "macrocycle, macrolactonization, polyketide",
                    "--frontier-smiles",
                    target,
                    "--output-dir",
                    tmp,
                    "--query-budget",
                    "4",
                    "--literature-backend",
                    "local",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            validation = json.loads((Path(tmp) / "validation.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["case_id"], "macro_lactone_cli")
        self.assertTrue(validation["accepted"], validation)
        self.assertEqual(validation["route_status"], "partial_anchor")

    def test_workflow_does_not_mutate_chem_enzy_default_search_config(self):
        before_models = list(DEFAULT_ONE_STEP_MODELS)
        before_config = RouteSearchConfig(target_smiles="CCO").to_dict()
        target = "Oc1ccccc1OC1COC(O)C(O)C1O"

        with tempfile.TemporaryDirectory() as tmp:
            run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="glycoside_default_guard",
                    family_hint="glycoside, sugar, glycosylation",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=3,
                    literature_backend="local",
                )
            )
        after_config = RouteSearchConfig(target_smiles="CCO").to_dict()

        self.assertEqual(DEFAULT_ONE_STEP_MODELS, before_models)
        self.assertEqual(after_config, before_config)
        self.assertEqual(after_config["search_flags"], {})

    def test_macrocycle_case_finds_macrolactonization_template(self):
        target = "O=C1CCCCCCCCCCCCO1"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="macro_lactone_like",
                    family_hint="macrocycle, macrolactonization, polyketide",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=4,
                    literature_backend="local",
                )
            )
            package = json.loads(Path(result["artifacts"]["hybrid_route_package"]).read_text(encoding="utf-8"))
            templates = package["strategy_templates"]

        self.assertTrue(any(t.get("reaction_class") == "macrolactonization" for t in templates))
        self.assertTrue(all(t.get("not_raw_reaction_injection") for t in templates))

    def test_glycoside_case_finds_glycosylation_template(self):
        target = "Oc1ccccc1OC1COC(O)C(O)C1O"
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smiles_first_workflow(
                SmilesFirstWorkflowConfig(
                    target_smiles=target,
                    target_name="phenolic_glycoside_like",
                    family_hint="glycoside, sugar, glycosylation",
                    frontier_smiles=target,
                    output_dir=tmp,
                    query_budget=4,
                    literature_backend="local",
                )
            )
            package = json.loads(Path(result["artifacts"]["hybrid_route_package"]).read_text(encoding="utf-8"))
            templates = package["strategy_templates"]

        self.assertTrue(any(t.get("reaction_class") == "glycosylation" for t in templates))
        self.assertTrue(any(t.get("candidate_kind") == "forward_surrogate" for t in templates))

    def test_run_case_api_json_literature_backend_persists_worker_evidence(self):
        target = "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C"
        evidence_artifact = {
            "schema_version": "evidence_card_artifact.v1",
            "artifact_id": "worker_ev_artifact",
            "artifact_type": "EvidenceCard",
            "case_id": "bufadienolide_worker_case",
            "source": "api_json",
            "input_refs": ["target_profile.json", "frontier_report.json"],
            "evidence_refs": [],
            "validation_status": "draft",
            "payload": {
                "evidence_id": "ev_worker_bufadienolide",
                "case_id": "bufadienolide_worker_case",
                "source_type": "literature",
                "source_title": "Traceable bufadienolide strategic precedent",
                "target_relation": "family_precedent",
                "claim_type": "strategic_disconnection",
                "route_role": "strategic_disconnection",
                "confidence": "medium_high",
                "url": "https://example.org/bufadienolide-route",
                "source_record_id": "worker_record",
                "family_id": "bufadienolide_steroid",
                "validation_status": "draft",
            },
        }
        worker_record = WorkerRunRecord(
            run_id="bufadienolide_worker_case:literature:api_json:run",
            task_id="bufadienolide_worker_case:literature:api_json",
            case_id="bufadienolide_worker_case",
            status="accepted_draft",
            backend="api_json",
            command=["api_json", "POST", "/responses"],
            output_artifact=evidence_artifact,
            output_validation={"schema_version": "worker_output_validation.v1", "accepted": True, "reasons": []},
            metadata={"provider": "test-provider", "base_url_fingerprint": "abc123", "model": "test-model"},
            usage={"input_tokens": 12},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("cascade_planner.agent.smiles_first.run_codex_worker", return_value=worker_record) as run_worker:
                result = run_smiles_first_workflow(
                    SmilesFirstWorkflowConfig(
                        target_smiles=target,
                        target_name="bufadienolide_worker_case",
                        family_hint="bufadienolide, steroid, pyrone",
                        frontier_smiles=target,
                        output_dir=tmp,
                        query_budget=3,
                        literature_backend="api_json",
                    )
                )
            out = Path(tmp)
            worker_run = json.loads((out / "literature_worker_run_record.json").read_text(encoding="utf-8"))
            worker_validation = json.loads((out / "literature_worker_artifact_validation.json").read_text(encoding="utf-8"))
            literature_report = json.loads((out / "literature_search_report.json").read_text(encoding="utf-8"))
            case_bundle = json.loads(Path(result["artifacts"]["case_bundle"]).read_text(encoding="utf-8"))

        self.assertTrue(run_worker.call_args.kwargs["use_api_json"])
        self.assertFalse(run_worker.call_args.kwargs["use_codex_cli"])
        self.assertEqual(worker_run["backend"], "api_json")
        self.assertEqual(worker_run["command"], ["api_json", "POST", "/responses"])
        self.assertEqual(worker_run["metadata"]["provider"], "test-provider")
        self.assertEqual(literature_report["backend"], "api_json")
        self.assertEqual(literature_report["worker_status"], "accepted_draft")
        self.assertTrue(worker_validation["validations"][0]["accepted"], worker_validation)
        artifact_types = [item["artifact_type"] for item in case_bundle["artifacts"]]
        self.assertIn("WorkerRunRecord", artifact_types)
        self.assertIn("EvidenceCard", artifact_types)

    def test_run_case_local_pubmed_backend_adds_traceable_external_evidence(self):
        target = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
        esearch = {"esearchresult": {"idlist": ["12345"]}}
        esummary = {
            "result": {
                "uids": ["12345"],
                "12345": {
                    "uid": "12345",
                    "title": "Synthesis of atorvastatin side-chain intermediates",
                    "fulljournalname": "Journal of Synthetic Evidence",
                    "pubdate": "2001",
                    "articleids": [
                        {"idtype": "pubmed", "value": "12345"},
                        {"idtype": "doi", "value": "10.1000/test-atorvastatin"},
                    ],
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "cascade_planner.agent.literature_research._fetch_pubmed_json",
                side_effect=[esearch, esummary],
            ) as fetch_pubmed:
                result = run_smiles_first_workflow(
                    SmilesFirstWorkflowConfig(
                        target_smiles=target,
                        target_name="atorvastatin_pubmed_case",
                        family_hint="atorvastatin, synthetic statin, Paal-Knorr, Wittig, HWE",
                        frontier_smiles=target,
                        output_dir=tmp,
                        query_budget=3,
                        literature_backend="local_pubmed",
                    )
                )
            out = Path(tmp)
            literature_report = json.loads((out / "literature_search_report.json").read_text(encoding="utf-8"))
            evidence_rows = [
                json.loads(line)
                for line in (out / "evidence_cards.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            package = json.loads(Path(result["artifacts"]["hybrid_route_package"]).read_text(encoding="utf-8"))

        self.assertEqual(fetch_pubmed.call_count, 2)
        self.assertEqual(literature_report["backend"], "local_pubmed")
        self.assertIn("local_curated", literature_report["component_backends"])
        self.assertIn("pubmed", literature_report["component_backends"])
        self.assertTrue(any(row["evidence_id"] == "ev_pubmed_12345" for row in evidence_rows))
        pubmed_card = next(row for row in evidence_rows if row["evidence_id"] == "ev_pubmed_12345")
        self.assertEqual(pubmed_card["source_type"], "literature")
        self.assertEqual(pubmed_card["url"], "https://pubmed.ncbi.nlm.nih.gov/12345/")
        self.assertEqual(pubmed_card["doi"], "10.1000/test-atorvastatin")
        self.assertIn("pubmed_summary_only", pubmed_card["limitations"])
        self.assertEqual(package["route_status"], "partial_anchor")

    def test_run_case_pubmed_backend_keeps_summary_hits_as_leads_not_templates(self):
        target = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
        esearch = {"esearchresult": {"idlist": ["54321"]}}
        esummary = {
            "result": {
                "uids": ["54321"],
                "54321": {
                    "uid": "54321",
                    "title": "Synthesis of atorvastatin intermediates",
                    "fulljournalname": "Mock PubMed Journal",
                    "pubdate": "2003",
                    "articleids": [{"idtype": "doi", "value": "10.1000/pubmed-lead"}],
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "cascade_planner.agent.literature_research._fetch_pubmed_json",
                side_effect=[esearch, esummary],
            ):
                result = run_smiles_first_workflow(
                    SmilesFirstWorkflowConfig(
                        target_smiles=target,
                        target_name="atorvastatin_pubmed_only_case",
                        family_hint="atorvastatin, synthetic statin, synthesis",
                        frontier_smiles=target,
                        output_dir=tmp,
                        query_budget=2,
                        literature_backend="pubmed",
                    )
                )
            out = Path(tmp)
            evidence_rows = [
                json.loads(line)
                for line in (out / "evidence_cards.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            candidates = [
                json.loads(line)
                for line in Path(result["artifacts"]["literature_candidates"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            package = json.loads(Path(result["artifacts"]["hybrid_route_package"]).read_text(encoding="utf-8"))
            validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))

        self.assertEqual([row["evidence_id"] for row in evidence_rows], ["ev_pubmed_54321"])
        self.assertEqual(candidates, [])
        self.assertEqual(package["route_status"], "literature_gap")
        self.assertEqual(validation["route_status"], "literature_gap")


if __name__ == "__main__":
    unittest.main()
