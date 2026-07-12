import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from cascade_planner.harness.codex_plan import _workflow_plan_normalization_audit, deterministic_workflow_plan
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.runner import (
    _normalize_controller_plan_for_execution,
    emit_final_verdict,
    run_codex_entry_controller,
)
from cascade_planner.harness.schemas import (
    CANONICAL_RUN_SEMANTICS,
    TargetInput,
    WORKFLOW_PLAN_SCHEMA,
    validate_workflow_plan,
    workflow_plan_from_dict,
)
from cascade_planner.harness.route_failure_feedback import compile_route_failure_feedback
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    _compile_open_research_downstream,
    _target_search_name,
    execute_local_tool,
    run_open_structure_research_agent,
)
from cascade_planner.harness.open_research_retrieval import prefetch_open_research_evidence
from cascade_planner.harness.source_detail_chain_builder import resolve_curator_records_to_source_detail_steps
from cascade_planner.harness.source_detail_resolution import source_detail_curator_records_path
from cascade_planner.providers.stock import (
    canonicalize_stock_snapshot,
    stock_snapshot_sha256,
)
from scripts.run_codex_entry_agentic_blackboard import (
    _codex_action_planner_env_overrides,
    _codex_agent_team_runtime_args,
    _trusted_stock_snapshots_from_args,
)
from scripts.run_codex_entry_controller import _resolve_cli_targets
from scripts.run_chem_enzy_plan_for_web import _stock_names_from_payload
from scripts.run_open_structure_template_agent import _read_or_build_prompt, _validate_open_agent_outputs


BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4"
    "([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
O_GLYCOSIDE_SMILES = "Oc1ccccc1OC1COC(O)C(O)C1O"
ATORVASTATIN_SMILES = (
    "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
    "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
)
ATORVASTATIN_REGIOISOMER_SMILES = (
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)"
    "n(CC[C@H](O)C[C@H](O)CC(=O)O)c1-c1ccccc1"
)
_SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "source_evidence_stub.pdf"
_SOURCE_PAGE_FIXTURE = Path(__file__).parent / "fixtures" / "source_page.ppm"
_SOURCE_MANIFEST_FIXTURE = Path(__file__).parent / "fixtures" / "source_evidence_manifest.json"
_TRUSTED_REGISTRY_FIXTURE = Path(__file__).parent / "fixtures" / "trusted_literature_step_registry.json"
SOURCE_EVIDENCE_MANIFEST_FIXTURE = (
    Path(__file__).parent / "fixtures" / "source_evidence_manifest.json"
).resolve()
TRUSTED_LITERATURE_REGISTRY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "trusted_literature_step_registry.json"
).resolve()


class CodexEntryHarnessContractTest(unittest.TestCase):
    def test_preflight_rejects_known_target_name_smiles_mismatch(self):
        correct = run_preflight(TargetInput(target_name="atorvastatin", target_smiles=ATORVASTATIN_SMILES))
        wrong = run_preflight(TargetInput(target_name="atorvastatin", target_smiles=ATORVASTATIN_REGIOISOMER_SMILES))
        wrong_renamed = run_preflight(
            TargetInput(target_name="atorvastatin_latest_small_stock_depth20_real", target_smiles=ATORVASTATIN_REGIOISOMER_SMILES)
        )
        analog_probe = run_preflight(TargetInput(target_name="atorvastatin_like", target_smiles="CCO"))

        self.assertTrue(correct["accepted"], correct["reasons"])
        self.assertEqual(correct["known_target_identity_audit"]["observed_inchi_key"], "XUKUURHRXDUEBC-KAYWLYCHSA-N")
        self.assertFalse(wrong["accepted"])
        self.assertIn("known_target_identity_mismatch:atorvastatin", wrong["reasons"])
        self.assertEqual(wrong["known_target_identity_audit"]["observed_inchi_key"], "OYBJITKZFDHGHP-SVBPBHIXSA-N")
        self.assertFalse(wrong_renamed["accepted"])
        self.assertIn("known_target_identity_mismatch:atorvastatin", wrong_renamed["reasons"])
        self.assertTrue(analog_probe["accepted"], analog_probe["reasons"])
        self.assertEqual(analog_probe["known_target_identity_audit"], {})

    def test_cli_accepts_positional_smiles_and_creates_batch_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_codex_entry_controller.py",
                    "not_a_smiles",
                    "still_not_a_smiles",
                    "--output-dir",
                    tmp,
                    "--offline-planner",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(proc.stdout)
            run_dirs = [Path(item["run_dir"]) for item in payload["runs"]]

            self.assertEqual(payload["schema_version"], "codex_entry_controller_cli_batch_result.v1")
            self.assertEqual(payload["run_count"], 2)
            self.assertEqual(len(run_dirs), 2)
            self.assertNotEqual(run_dirs[0], run_dirs[1])
            for run_dir in run_dirs:
                self.assertEqual(run_dir.parent, Path(tmp))
                self.assertTrue((run_dir / "target_input.json").exists())
                self.assertTrue((run_dir / "final_verdict.json").exists())

    def test_cli_legacy_single_target_keeps_output_dir_as_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                smiles=[],
                target_smiles="not_a_smiles",
                target_name="legacy_case",
                output_dir=tmp,
                output_root=str(Path(tmp) / "unused"),
                run_prefix="codex_entry",
            )
            targets = _resolve_cli_targets(args)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].target_name, "legacy_case")
        self.assertEqual(targets[0].target_smiles, "not_a_smiles")
        self.assertEqual(targets[0].output_dir, Path(tmp))

    def test_agentic_blackboard_cli_maps_codex_planner_tool_budget_env(self):
        args = Namespace(
            codex_action_planner_tools="web_search,browser,literature_search",
            codex_action_planner_max_tool_calls=6,
            codex_action_planner_timeout_s=420.0,
            codex_scout_timeout_s=240.0,
            codex_scout_reasoning_effort="medium",
            timeout_s=300.0,
            codex_worker_auth="ambient",
            codex_worker_sandbox="bypassed",
        )

        overrides = _codex_action_planner_env_overrides(args)

        self.assertEqual(
            overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_ALLOWED_TOOLS"],
            "web_search,browser,literature_search",
        )
        self.assertEqual(overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_TOOL_CALLS"], "6")
        self.assertEqual(overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_TIMEOUT_S"], "420.0")
        self.assertEqual(overrides["AUTOPLANNER_CODEX_SCOUT_TIMEOUT_S"], "240.0")
        self.assertEqual(overrides["AUTOPLANNER_CODEX_SCOUT_REASONING_EFFORT"], "medium")
        self.assertEqual(overrides["AUTOPLANNER_CODEX_WORKER_AUTH"], "ambient")
        self.assertEqual(overrides["AUTOPLANNER_CODEX_WORKER_SANDBOX"], "bypassed")

    def test_agentic_blackboard_cli_defaults_codex_planner_timeout_to_total_timeout(self):
        args = Namespace(
            codex_action_planner_tools=None,
            codex_action_planner_max_tool_calls=None,
            codex_action_planner_timeout_s=None,
            codex_scout_timeout_s=None,
            codex_scout_reasoning_effort=None,
            timeout_s=300.0,
            codex_worker_auth="auto",
            codex_worker_sandbox=None,
        )

        overrides = _codex_action_planner_env_overrides(args)

        self.assertEqual(overrides["AUTOPLANNER_CODEX_ACTION_PLANNER_TIMEOUT_S"], "300.0")

    def test_agentic_blackboard_cli_explicit_chemenzy_prefix_overrides_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                chem_enzy_env_prefix=tmp,
                codex_action_planner_tools=None,
                codex_action_planner_max_tool_calls=None,
                codex_action_planner_timeout_s=None,
                codex_scout_timeout_s=None,
                codex_scout_reasoning_effort=None,
                timeout_s=300.0,
                codex_worker_auth="auto",
                codex_worker_sandbox=None,
            )

            overrides = _codex_action_planner_env_overrides(args)

        self.assertEqual(
            overrides["CHEMENZY_ENV_PREFIX"],
            str(Path(tmp).resolve()),
        )
        self.assertEqual(
            overrides["AUTOPLANNER_CHEMENZY_ENV_PREFIX_SOURCE"],
            "cli",
        )

    def test_agent_team_inherits_global_model_and_worker_auth(self):
        model, auth = _codex_agent_team_runtime_args(
            Namespace(
                codex_agent_team_model="",
                codex_agent_team_auth_mode=None,
                model="gpt-5.5",
                codex_worker_auth="key",
            )
        )

        self.assertEqual(model, "gpt-5.5")
        self.assertEqual(auth, "api_key")

    def test_explicit_agent_team_runtime_settings_override_global_values(self):
        model, auth = _codex_agent_team_runtime_args(
            Namespace(
                codex_agent_team_model="team-model",
                codex_agent_team_auth_mode="ambient_codex_cli",
                model="global-model",
                codex_worker_auth="key",
            )
        )

        self.assertEqual(model, "team-model")
        self.assertEqual(auth, "ambient_codex_cli")

    def test_agentic_cli_loads_digest_bound_commercial_stock_snapshot(self):
        snapshot = canonicalize_stock_snapshot(
            {
                "schema_version": "stock_offer_snapshot.v1",
                "supplier": "Example Supplier",
                "catalog_number": "EX-001",
                "canonical_smiles": "CCO",
                "checked_at": "2026-07-12T00:00:00+00:00",
                "available": True,
                "purity": "99%",
                "pack_size": "1 g",
                "price": 12.5,
                "currency": "USD",
                "region": "US",
                "lead_time_days": 3,
                "source_url": "https://supplier.invalid/EX-001",
                "metadata": {"export_id": "snapshot-001"},
            }
        )
        digest = stock_snapshot_sha256(snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "trusted-stock.json"
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": "trusted_stock_snapshots.v1",
                        "snapshots": [{**snapshot, "snapshot_sha256": digest}],
                    }
                ),
                encoding="utf-8",
            )
            loaded = _trusted_stock_snapshots_from_args(
                Namespace(trusted_stock_snapshot=[str(artifact)])
            )

        self.assertEqual(list(loaded), [digest])
        self.assertEqual(loaded[digest]["canonical_smiles"], "CCO")
        self.assertEqual(loaded[digest]["snapshot_sha256"], digest)

    def test_agentic_cli_rejects_mutated_commercial_stock_snapshot(self):
        snapshot = canonicalize_stock_snapshot(
            {
                "supplier": "Example Supplier",
                "catalog_number": "EX-002",
                "canonical_smiles": "CCO",
                "checked_at": "2026-07-12T00:00:00+00:00",
                "available": True,
                "metadata": {},
            }
        )
        digest = stock_snapshot_sha256(snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "tampered-stock.json"
            artifact.write_text(
                json.dumps(
                    {
                        **snapshot,
                        "available": False,
                        "snapshot_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                _trusted_stock_snapshots_from_args(
                    Namespace(trusted_stock_snapshot=[str(artifact)])
                )

    def test_invalid_smiles_stops_before_codex_research_and_emits_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="bad_input",
                target_smiles="not_a_smiles",
                output_dir=tmp,
                use_live_planner=False,
            )
            out = Path(tmp)
            verdict = json.loads((out / "final_verdict.json").read_text(encoding="utf-8"))
            tool_calls = (out / "tool_calls.jsonl").read_text(encoding="utf-8")
            required_exists = {
                name: (out / name).exists()
                for name in [
                    "target_input.json",
                    "preflight.json",
                    "codex_workflow_plan.json",
                    "decision_trace.jsonl",
                    "tool_calls.jsonl",
                    "artifact_bundle.json",
                    "final_verdict.json",
                    "progress_panel.html",
                ]
            }
            codex_events_exists = (out / "codex_events.jsonl").exists()

        self.assertEqual(result["final_verdict"]["verdict"], "invalid_input")
        self.assertEqual(verdict["verdict"], "invalid_input")
        self.assertEqual(tool_calls, "")
        self.assertFalse(codex_events_exists)
        self.assertTrue(all(required_exists.values()), required_exists)

    def test_forbidden_planner_action_is_rejected(self):
        plan = _plan(
            "ethanol_case",
            strategy="hybrid",
            tools=[{"tool_name": "LLM_RERANK_CANDIDATES", "payload": {}}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="ethanol_case",
                target_smiles="CCO",
                output_dir=tmp,
                planner_plan=plan,
                use_live_planner=False,
            )
            bundle = json.loads((Path(tmp) / "artifact_bundle.json").read_text(encoding="utf-8"))

        self.assertEqual(result["final_verdict"]["verdict"], "needs_followup")
        self.assertIn("forbidden_planner_tool:0:LLM_RERANK_CANDIDATES", result["final_verdict"]["reasons"])
        self.assertFalse(bundle["validations"][0]["accepted"])

    def test_raw_reaction_injection_in_planner_tool_payload_is_rejected(self):
        plan = _plan(
            "raw_payload_case",
            strategy="hybrid",
            tools=[
                {
                    "tool_name": "run_smiles_first_literature_workflow",
                    "payload": {"rxn_smiles": "CCO>>CC=O"},
                }
            ],
        )
        validation = validate_workflow_plan(plan, case_id="raw_payload_case")

        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="raw_payload_case",
                target_smiles="CCO",
                output_dir=tmp,
                planner_plan=plan,
                use_live_planner=False,
            )

        self.assertFalse(validation["accepted"])
        self.assertIn("raw_reaction_injection", validation["reasons"])
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["final_verdict"]["verdict"], "needs_followup")
        self.assertIn("raw_reaction_injection", result["final_verdict"]["reasons"])

    def test_literature_first_requires_accepted_reason(self):
        plan = _plan(
            "literature_without_reason",
            strategy="literature_first",
            tools=[{"tool_name": "run_smiles_first_literature_workflow", "payload": {}}],
        )

        validation = validate_workflow_plan(plan, case_id="literature_without_reason")

        self.assertFalse(validation["accepted"])
        self.assertIn("literature_first_requires_accepted_reason", validation["reasons"])

    def test_literature_first_with_accepted_reason_can_start_with_literature(self):
        plan = _plan(
            "literature_with_reason",
            strategy="literature_first",
            tools=[{"tool_name": "run_smiles_first_literature_workflow", "payload": {}}],
            planner_decision_reason="glycoside_or_o_glycoside_like",
        )

        validation = validate_workflow_plan(plan, case_id="literature_with_reason")

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_live_planner_object_run_semantics_is_normalized_to_canonical(self):
        raw_plan = _plan(
            "literature_with_object_semantics",
            strategy="literature_first",
            tools=[{"tool_name": "run_smiles_first_literature_workflow", "payload": {}}],
            planner_decision_reason="steroid_or_polycyclic_core",
            run_semantics={
                "codex_role": "workflow_planning_only",
                "chemistry_solution_allowed": False,
                "raw_reaction_output_allowed": False,
                "final_verdict_authority": "deterministic_validators",
                "literature_reasoning_channel": "run_open_structure_research_agent_only",
            },
        )

        plan = workflow_plan_from_dict(raw_plan)
        validation = validate_workflow_plan(plan, case_id="literature_with_object_semantics")
        audit = _workflow_plan_normalization_audit(raw_plan, plan)

        self.assertEqual(plan.run_semantics, CANONICAL_RUN_SEMANTICS)
        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(audit["raw_run_semantics_type"], "dict")
        self.assertTrue(audit["run_semantics_changed"])

    def test_chem_enzy_first_cannot_start_with_literature(self):
        plan = _plan(
            "bad_chem_first",
            strategy="chem_enzy_first",
            tools=[{"tool_name": "run_smiles_first_literature_workflow", "payload": {}}],
        )

        validation = validate_workflow_plan(plan, case_id="bad_chem_first")

        self.assertFalse(validation["accepted"])
        self.assertIn("chem_enzy_first_must_start_with_run_chemenzy", validation["reasons"])

    def test_hybrid_open_research_requires_native_audit_first(self):
        plan = _plan(
            "bad_hybrid",
            strategy="hybrid",
            tools=[{"tool_name": "run_open_structure_research_agent", "payload": {}}],
        )

        validation = validate_workflow_plan(plan, case_id="bad_hybrid")

        self.assertFalse(validation["accepted"])
        self.assertIn(
            "run_open_structure_research_agent_requires_native_audit_or_literature_first_reason",
            validation["reasons"],
        )

    def test_steroid_literature_first_plan_is_normalized_to_keep_chemenzy_baseline(self):
        target = {
            "schema_version": "codex_entry_target_input.v1",
            "case_id": "o_c1cc_c_2_c_c_ccc3c2c",
            "target_name": "o_c1cc_c_2_c_c_ccc3c2c",
            "target_smiles": "O=C1CC[C@@]2(C)C(CCC3C2C[C@H](Cl)[C@@]4(C)C3CCC4=O)=C1",
            "family_hint": "",
        }
        preflight = {
            "schema_version": "codex_entry_preflight.v1",
            "accepted": True,
            "case_id": "o_c1cc_c_2_c_c_ccc3c2c",
            "target_profile": {"heavy_atoms": 22, "formula": "C19H25ClO2"},
            "initial_risk_flags": ["polycyclic_or_steroid_like", "unassigned_stereochemistry"],
        }
        plan = _plan(
            "o_c1cc_c_2_c_c_ccc3c2c",
            strategy="literature_first",
            planner_decision_reason="steroid_or_polycyclic_core",
            tools=[
                {"tool_name": "run_smiles_first_literature_workflow", "payload": {}},
                {"tool_name": "run_open_structure_research_agent", "payload": {}},
                {"tool_name": "extract_pdf_literature_structures", "payload": {}},
            ],
        )

        normalized, audit = _normalize_controller_plan_for_execution(plan, target_data=target, preflight=preflight)
        tool_names = [row["tool_name"] for row in normalized["planned_tools"]]
        validation = validate_workflow_plan(normalized, case_id=preflight["case_id"])

        self.assertTrue(audit["changed"], audit)
        self.assertEqual(normalized["recommended_strategy"], "hybrid")
        self.assertEqual(tool_names[:2], ["run_chemenzy", "audit_route_and_extract_frontier"])
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_generated_smiles_slug_does_not_become_single_letter_search_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "o_c1cc_c_2_c_c_ccc3c2c",
                    "target_smiles": "O=C1CC[C@@]2(C)C(CCC3C2C[C@H](Cl)[C@@]4(C)C3CCC4=O)=C1",
                    "family_hint": "",
                },
                preflight={
                    "case_id": "o_c1cc_c_2_c_c_ccc3c2c",
                    "target_profile": {"formula": "C19H25ClO2", "heavy_atoms": 22},
                    "initial_risk_flags": ["polycyclic_or_steroid_like"],
                },
            )

            search_name = _target_search_name(state)

        self.assertNotEqual(search_name, "o")
        self.assertIn("C19H25ClO2", search_name)
        self.assertIn("steroid", search_name)

    def test_pdf_and_visual_tools_reject_missing_inputs_without_directory_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol", "target_profile": {"heavy_atoms": 3}},
            )

            pdf_record = execute_local_tool("extract_pdf_literature_structures", {}, state)
            visual_record = execute_local_tool("extract_visual_literature_chain", {"image_paths": ["."]}, state)

        self.assertEqual(pdf_record.status, "rejected")
        self.assertIn("pdf_or_image_input_missing", pdf_record.reasons)
        self.assertEqual(visual_record.status, "rejected")
        self.assertIn("visual_input_images_missing", visual_record.reasons)

    def test_structure_resolution_uses_target_identity_without_visual_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "paclitaxel_case",
                    "target_smiles": "CCO",
                    "family_hint": "taxane paclitaxel",
                    "target_aliases": ["Taxol"],
                },
                preflight={"case_id": "paclitaxel_case", "target_profile": {"heavy_atoms": 3}},
            )
            with patch("cascade_planner.harness.tools.run_visual_literature_chain_agent") as visual_mock:
                record = execute_local_tool(
                    "resolve_literature_structure_task",
                    {
                        "schema_version": "literature_structure_resolution_payload.v1",
                        "task_id": "resolve_structure:doi_source_taxol",
                        "label": "Taxol",
                        "source_ref": "doi:source",
                        "source_title": "Taxol paper",
                        "source_locator": "Figure 1, compound Taxol",
                        "run_visual": True,
                        "no_solved_claim": True,
                    },
                    state,
                )

        visual_mock.assert_not_called()
        result = record.output
        self.assertEqual(record.status, "accepted")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["visual_attempt"], {})
        self.assertEqual(result["resolved_structures"][0]["smiles"], "CCO")
        self.assertEqual(result["resolved_structures"][0]["derivation_mode"], "target_input_identity")
        self.assertTrue(result["resolved_structures"][0]["target_identity_shortcut"])

    def test_visual_chain_drafts_connect_as_advisory_without_materialized_evidence(self):
        target = {
            "schema_version": "codex_entry_target_input.v1",
            "case_id": "visual_chain_case",
            "target_name": "acetaldehyde",
            "target_smiles": "CC=O",
            "family_hint": "small molecule",
        }
        preflight = {
            "schema_version": "codex_entry_preflight.v1",
            "accepted": True,
            "case_id": "visual_chain_case",
            "target_profile": {"heavy_atoms": 3},
            "initial_risk_flags": [],
        }
        plan = _plan(
            "visual_chain_case",
            strategy="hybrid",
            tools=[
                {"tool_name": "extract_visual_literature_chain", "payload": {}},
                {"tool_name": "validate_literature_intermediate_chain", "payload": {}},
                {"tool_name": "build_source_detail_curator_records", "payload": {}},
                {"tool_name": "compile_source_detail_chain_route", "payload": {}},
                {"tool_name": "compile_hybrid_route_set", "payload": {}},
            ],
        )
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "visual_chain_case",
            "target_name": "acetaldehyde",
            "target_smiles": "CC=O",
            "source_ref": "doi:10.0000/visual-chain",
            "source_title": "Visual chain source",
            "evidence_refs": ["scheme:1"],
            "route_order": "forward_start_to_target",
            "source_excerpt": "Scheme 1 reports conversion of compound 1 to compound 3.",
            "default_condition_candidate": {
                "solvent": "water",
                "temperature": "25 C",
            },
            "chain": [
                {"label": "1", "smiles": "CC", "source_locator": "Scheme 1, compound 1"},
                {"label": "2", "smiles": "CCO", "source_locator": "Scheme 1, compound 2"},
                {"label": "3", "smiles": "CC=O", "source_locator": "Scheme 1, compound 3"},
            ],
        }

        validation = validate_workflow_plan(plan, case_id="visual_chain_case")
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target,
                preflight=preflight,
                mock_tool_results={
                    "extract_visual_literature_chain": {
                        "schema_version": "visual_literature_chain_extraction_result.v1",
                        "accepted": True,
                        "status": "completed",
                        "candidate_chain": candidate_chain,
                        "candidate_step_count": 2,
                        "extraction_policy": {
                            "pdf_reuse_allowed": True,
                            "prior_candidate_chain_reuse_allowed": False,
                            "prior_source_detail_records_reuse_allowed": False,
                            "must_derive_from_current_images": True,
                        },
                    }
                },
            )
            extraction_record = execute_local_tool("extract_visual_literature_chain", {}, state)
            chain_record = execute_local_tool(
                "validate_literature_intermediate_chain",
                {},
                state,
            )
            curator_record = execute_local_tool("build_source_detail_curator_records", {}, state)
            route_record = execute_local_tool(
                "compile_source_detail_chain_route",
                {"terminal_smiles": "CC", "terminal_name": "compound 1"},
                state,
            )
            hybrid_record = execute_local_tool("compile_hybrid_route_set", {}, state)

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(extraction_record.status, "accepted")
        self.assertIn("visual_structure_candidate_chain", state.artifacts)
        self.assertEqual(chain_record.status, "accepted")
        self.assertEqual(curator_record.status, "accepted")
        self.assertEqual(route_record.status, "rejected")
        self.assertIn("source_detail_step_not_trusted_curated", route_record.reasons)
        self.assertEqual(hybrid_record.status, "accepted")
        self.assertEqual(curator_record.output["summary"]["one_step_row_count"], 0)
        plugin = route_record.output["result"]["compiled_downstream"]["literature_template_plugin"]
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 2)
        self.assertFalse(route_record.output["result"]["chain_audit"]["terminal_reached"])
        self.assertEqual(hybrid_record.output["result"]["summary"]["literature_route_count"], 0)
        literature_route = hybrid_record.output["result"]["routes"][0]
        self.assertEqual(literature_route["status"], "needs_exact_curation")
        self.assertEqual(literature_route["step_count"], 0)

    def test_source_text_condition_repair_preserves_visual_smiles_for_partial_chain(self):
        target = {
            "schema_version": "codex_entry_target_input.v1",
            "case_id": "partial_visual_case",
            "target_name": "acetaldehyde",
            "target_smiles": "CC=O",
            "family_hint": "small molecule",
        }
        preflight = {
            "schema_version": "codex_entry_preflight.v1",
            "accepted": True,
            "case_id": "partial_visual_case",
            "target_profile": {"heavy_atoms": 3},
            "initial_risk_flags": [],
        }
        plan = _plan(
            "partial_visual_case",
            strategy="hybrid",
            tools=[
                {"tool_name": "apply_source_text_condition_repairs", "payload": {}},
                {"tool_name": "validate_literature_intermediate_chain", "payload": {}},
            ],
        )
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "partial_visual_case",
            "target_name": "acetaldehyde",
            "target_smiles": "",
            "source_ref": "doi:10.0000/partial-visual",
            "source_title": "Partial visual source",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "step_2_from_1",
                    "segment_id": "visual_literature_chain",
                    "product_label": "2",
                    "product_smiles": "CCO",
                    "reactant_labels": ["1"],
                    "reactant_smiles": ["CC"],
                    "main_reactant_smiles": "CC",
                    "source_ref": "doi:10.0000/partial-visual",
                    "source_title": "Partial visual source",
                    "source_locator": "Scheme 1",
                    "structure_derivation": {
                        "basis": "current_pdf_image_to_smiles",
                        "source_locator": "Scheme 1",
                        "confidence": "medium",
                        "tool_checks": ["visual extraction performed in this run"],
                    },
                }
            ],
        }

        validation = validate_workflow_plan(plan, case_id="partial_visual_case")
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target,
                preflight=preflight,
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            repair_record = execute_local_tool(
                "apply_source_text_condition_repairs",
                {
                    "condition_repairs": [
                        {
                            "step_id": "step_2_from_1",
                            "condition_candidate": {
                                "schema_version": "condition_candidate.v1",
                                "source_type": "exact",
                                "condition_status": "evidence_backed",
                                "reagent": "NaBH4",
                                "solvent": "MeOH",
                                "source_grounding": "Scheme 1 caption",
                            },
                            "source_excerpt": "Scheme 1 reports reduction with NaBH4 in MeOH.",
                            "source_locator": "Scheme 1, arrow 1 to 2",
                        }
                    ],
                },
                state,
            )
            chain_record = execute_local_tool(
                "validate_literature_intermediate_chain",
                {
                    "allow_partial_chain_without_target_match": True,
                    "require_contiguous": False,
                },
                state,
            )

        repaired = repair_record.output["candidate_chain"]
        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(repair_record.status, "accepted")
        self.assertTrue(repaired["condition_repair_audit"]["structure_smiles_unchanged"])
        self.assertEqual(repaired["steps"][0]["product_smiles"], "CCO")
        self.assertEqual(repaired["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain_record.status, "accepted")
        self.assertEqual(chain_record.output["result"]["summary"]["accepted_step_count"], 1)

    def test_mock_bufotalin_fake_closure_cannot_become_solved(self):
        plan = _plan(
            "bufotalin_mock",
            strategy="hybrid",
            tools=[
                {"tool_name": "run_chemenzy", "payload": {}},
                {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
                {"tool_name": "validate_artifact_bundle", "payload": {}},
                {"tool_name": "emit_final_verdict", "payload": {}},
            ],
        )
        fake_audit = {
            "schema_version": "route_audit_report.v1",
            "case_id": "bufotalin_mock",
            "route_status": "fake_closed_rejected",
            "target_match": True,
            "step_structural_audit": "failed",
            "stock_audit_passed": False,
            "route_mode": "unresolved_core",
            "enzyme_step_status": "unknown",
            "evidence_status": "unknown",
            "condition_status": "unknown",
            "fake_closure_rejected": True,
            "unresolved_core": True,
            "top_route_summary": {"case_id": "bufotalin_mock", "route_status": "solved"},
            "rejected_terminal_list": [
                {
                    "smiles": BUFOTALIN_SMILES,
                    "reason": "advanced_same_scaffold_no_complexity_drop",
                }
            ],
            "failure_events": [{"case_id": "bufotalin_mock", "reason": "fake_closure"}],
            "reasons": ["advanced_same_scaffold", "no_complexity_drop"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="bufotalin_mock",
                target_smiles=BUFOTALIN_SMILES,
                family_hint="bufotalin, bufadienolide, steroid",
                output_dir=tmp,
                planner_plan=plan,
                use_live_planner=False,
                mock_tool_results={
                    "run_chemenzy": {
                        "schema_version": "chemenzy_mock_result.v1",
                        "accepted": True,
                        "status": "mock_solved_claim",
                        "route_status": "solved",
                        "stock_audit_passed": False,
                    },
                    "audit_route_and_extract_frontier": fake_audit,
                },
            )

        self.assertEqual(result["final_verdict"]["verdict"], "fake_closed_rejected")
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertIn("fake_closure_evidence_present", result["final_verdict"]["reasons"])

    def test_raw_route_verifier_rejects_hidden_nonstock_fake_solved_route(self):
        target = ATORVASTATIN_SMILES
        advanced = (
            "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)"
            "n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C"
        )
        plan = _plan(
            "atorvastatin",
            strategy="chem_enzy_first",
            tools=[
                {"tool_name": "run_chemenzy", "payload": {}},
                {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
                {"tool_name": "validate_artifact_bundle", "payload": {}},
                {"tool_name": "emit_final_verdict", "payload": {}},
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="atorvastatin",
                target_smiles=target,
                output_dir=tmp,
                planner_plan=plan,
                use_live_planner=False,
                mock_tool_results={
                    "run_chemenzy": {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "search_status": {"solved": True},
                        "target": target,
                        "routes": [
                            {
                                "route_rank": 0,
                                "score": 0.1,
                                "n_steps": 2,
                                "metrics": {
                                    "route_solved": True,
                                    "strict_stock_solve": True,
                                    "terminal_reactants": ["Nc1ccccc1"],
                                    "terminal_stock_status": {"Nc1ccccc1": True},
                                },
                                "steps": [
                                    {
                                        "index": 0,
                                        "main_reactant": "Nc1ccccc1",
                                        "aux_reactants": [advanced],
                                        "product": target,
                                        "stock_status": {"Nc1ccccc1": True, advanced: False},
                                    },
                                ],
                            }
                        ],
                    }
                },
            )
            verifier = json.loads((Path(tmp) / "route_verifier_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result["final_verdict"]["verdict"], "fake_closed_rejected")
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertFalse(verifier["accepted"])
        self.assertIn("hidden_nonstock_reactants", verifier["reasons"])
        self.assertNotIn("large_atom_jump", verifier["reasons"])
        self.assertIn("route_verifier_rejected_raw_routes", result["final_verdict"]["reasons"])

    def test_raw_route_verifier_does_not_treat_generated_intermediate_as_hidden_nonstock(self):
        target = ATORVASTATIN_SMILES
        advanced = (
            "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)"
            "n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C"
        )
        verifier = verify_chemenzy_raw_routes(
            {
                "target": target,
                "routes": [
                    {
                        "route_rank": 0,
                        "score": 0.1,
                        "n_steps": 2,
                        "metrics": {
                            "route_solved": True,
                            "strict_stock_solve": True,
                            "terminal_reactants": ["CO", "Nc1ccccc1"],
                            "terminal_stock_status": {"CO": True, "Nc1ccccc1": True},
                        },
                        "steps": [
                            {
                                "index": 0,
                                "main_reactant": "CO",
                                "aux_reactants": [],
                                "product": advanced,
                                "stock_status": {"CO": True},
                            },
                            {
                                "index": 1,
                                "main_reactant": "Nc1ccccc1",
                                "aux_reactants": [advanced],
                                "product": target,
                                "stock_status": {"Nc1ccccc1": True, advanced: False},
                            },
                        ],
                    }
                ],
            },
            target_smiles=target,
            case_id="atorvastatin",
        )

        self.assertFalse(verifier["accepted"])
        self.assertIn("large_atom_jump", verifier["reasons"])
        self.assertNotIn("hidden_nonstock_reactants", verifier["reasons"])
        self.assertEqual(verifier["rejected_route_summary"][0]["hidden_nonstock_count"], 0)

    def test_raw_route_verifier_rejects_backend_target_stereo_mismatch(self):
        request_target = "C[C@H](O)C(=O)O"
        backend_target = "C[C@@H](O)C(=O)O"
        verifier = verify_chemenzy_raw_routes(
            {
                "target": backend_target,
                "routes": [
                    {
                        "route_rank": 0,
                        "score": 0.9,
                        "n_steps": 1,
                        "metrics": {
                            "route_solved": True,
                            "strict_stock_solve": True,
                            "terminal_reactants": ["CCO", "C", "O"],
                            "terminal_stock_status": {"CCO": True, "C": True, "O": True},
                        },
                        "steps": [
                            {
                                "index": 0,
                                "main_reactant": "CCO",
                                "aux_reactants": ["C", "O", "O"],
                                "product": request_target,
                                "stock_status": {"CCO": True, "C": True, "O": True},
                            }
                        ],
                    }
                ],
            },
            target_smiles=request_target,
            case_id="lactic_acid_exact",
        )

        self.assertFalse(verifier["accepted"])
        self.assertEqual(verifier["accepted_route_count"], 0)
        self.assertFalse(verifier["target_match"])
        self.assertIn("target_equivalence_mismatch", verifier["reasons"])
        audit = verifier["target_equivalence_audit"]
        self.assertEqual(audit["request_target_smiles"], request_target)
        self.assertEqual(audit["backend_target_smiles"], backend_target)
        self.assertNotEqual(audit["request_canonical_isomeric_smiles"], audit["backend_canonical_isomeric_smiles"])
        self.assertEqual(audit["route_candidate_accepted_count_before_target_match"], 1)

    def test_deterministic_plan_uses_deep_strict_stock_for_complex_targets(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_SMILES,
            family_hint="statin synthetic atorvastatin",
        )
        preflight = run_preflight(target)
        plan = deterministic_workflow_plan(target_input=target.to_dict(), preflight=preflight)

        payload = plan.planned_tools[0]["payload"]
        self.assertEqual(plan.planned_tools[0]["tool_name"], "run_chemenzy")
        self.assertIn("run_guided_chemenzy_rerun", [tool["tool_name"] for tool in plan.planned_tools])
        self.assertIn("run_route_expansion_subgoal_search", [tool["tool_name"] for tool in plan.planned_tools])
        self.assertIn("run_self_evo_replay_gate", [tool["tool_name"] for tool in plan.planned_tools])
        self.assertEqual(payload["stock_mode"], "building-block")
        self.assertEqual(payload["search_preset"], "thorough")
        self.assertEqual(payload["max_steps"], 20)
        self.assertEqual(payload["chem_enzy_iterations"], 50)
        self.assertEqual(payload["chem_enzy_expansion_topk"], 100)

    def test_deterministic_plan_keeps_quick_strict_stock_for_small_targets(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        plan = deterministic_workflow_plan(target_input=target.to_dict(), preflight=preflight)

        payload = plan.planned_tools[0]["payload"]
        self.assertEqual(payload["stock_mode"], "building-block")
        self.assertEqual(payload["search_preset"], "quick")
        self.assertEqual(payload["max_steps"], 6)

    def test_noncanonical_artifact_bundle_cannot_emit_solved_verdict(self):
        verdict = emit_final_verdict(
            {
                "case_id": "replay_success",
                "run_semantics": "replay",
                "preflight": {"accepted": True},
                "artifacts": {
                    "route_audit": {
                        "route_status": "solved",
                        "stock_audit_passed": True,
                    }
                },
            }
        )

        self.assertEqual(verdict.verdict, "unresolved")
        self.assertFalse(verdict.solved)
        self.assertIn("solved_requires_deterministic_parent_route_proof", verdict.reasons)

    def test_guided_chemenzy_verified_route_can_drive_final_solved_verdict(self):
        raw = _accepted_ethanol_chemenzy_result_for_target("CCO")
        verifier = verify_chemenzy_raw_routes(raw, target_smiles="CCO")
        self.assertTrue(verifier["accepted"], verifier["reasons"])
        verdict = emit_final_verdict(
            {
                "case_id": "guided_success",
                "target_input": {"target_name": "ethanol", "target_smiles": "CCO"},
                "preflight": {"accepted": True},
                "artifacts": {
                    "guided_chemenzy": {
                        "schema_version": "guided_chemenzy_rerun_result.v1",
                        "accepted": True,
                        "raw_route_verifier": verifier,
                    }
                },
                "tool_calls": [],
                "validations": [],
            }
        )

        self.assertEqual(verdict.verdict, "solved")
        self.assertTrue(verdict.solved)
        self.assertTrue(verdict.stock_audit_passed)

    def test_guided_chemenzy_rejected_route_drives_fake_closed_verdict(self):
        verdict = emit_final_verdict(
            {
                "case_id": "guided_fake",
                "preflight": {"accepted": True},
                "artifacts": {
                    "guided_chemenzy": {
                        "schema_version": "guided_chemenzy_rerun_result.v1",
                        "accepted": False,
                        "raw_route_verifier": {
                            "schema_version": "harness_route_verifier_report.v1",
                            "accepted": False,
                            "route_status": "fake_closed_rejected",
                            "reasons": ["hidden_nonstock_reactants"],
                            "failure_events": [{"reason": "hidden_nonstock_reactants"}],
                        },
                    }
                },
                "tool_calls": [],
                "validations": [],
            }
        )

        self.assertEqual(verdict.verdict, "fake_closed_rejected")
        self.assertFalse(verdict.solved)
        self.assertIn("hidden_nonstock_reactants", verdict.reasons)

    def test_native_chemenzy_rejected_verifier_drives_fake_closed_verdict(self):
        verdict = emit_final_verdict(
            {
                "case_id": "native_fake",
                "preflight": {"accepted": True},
                "artifacts": {
                    "chemenzy": {
                        "schema_version": "chemenzy_web_result.v1",
                        "accepted": True,
                        "search_status": {"solved": True},
                        "raw_route_verifier": {
                            "schema_version": "harness_route_verifier_report.v1",
                            "accepted": False,
                            "route_status": "fake_closed_rejected",
                            "reasons": ["hidden_nonstock_reactants"],
                            "failure_events": [{"reason": "hidden_nonstock_reactants"}],
                        },
                    }
                },
                "tool_calls": [],
                "validations": [],
            }
        )

        self.assertEqual(verdict.verdict, "fake_closed_rejected")
        self.assertFalse(verdict.solved)
        self.assertIn("hidden_nonstock_reactants", verdict.reasons)

    def test_native_chemenzy_tool_writes_raw_route_verifier(self):
        target = ATORVASTATIN_SMILES
        advanced = (
            "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)"
            "n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C"
        )
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "atorvastatin",
                    "target_smiles": target,
                    "case_id": "atorvastatin",
                },
                preflight={"accepted": True, "case_id": "atorvastatin"},
                mock_tool_results={
                    "run_chemenzy": {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "search_status": {"solved": True},
                        "target": target,
                        "routes": [
                            {
                                "route_rank": 0,
                                "score": 0.1,
                                "n_steps": 2,
                                "metrics": {
                                    "route_solved": True,
                                    "strict_stock_solve": True,
                                    "terminal_reactants": ["CO", "Nc1ccccc1"],
                                    "terminal_stock_status": {"CO": True, "Nc1ccccc1": True},
                                },
                                "steps": [
                                    {
                                        "index": 0,
                                        "main_reactant": "CO",
                                        "aux_reactants": [],
                                        "product": advanced,
                                        "stock_status": {"CO": True},
                                    },
                                    {
                                        "index": 1,
                                        "main_reactant": "Nc1ccccc1",
                                        "aux_reactants": [advanced],
                                        "product": target,
                                        "stock_status": {"Nc1ccccc1": True, advanced: False},
                                    },
                                ],
                            }
                        ],
                    }
                },
            )

            record = execute_local_tool("run_chemenzy", {}, state)
            verifier = json.loads((Path(tmp) / "route_verifier_report.json").read_text(encoding="utf-8"))
            feedback = json.loads((Path(tmp) / "route_failure_feedback.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertFalse(verifier["accepted"])
        self.assertIn("large_atom_jump", verifier["reasons"])
        self.assertNotIn("hidden_nonstock_reactants", verifier["reasons"])
        self.assertEqual(state.artifacts["chemenzy"]["raw_route_verifier"], verifier)
        self.assertTrue(feedback["accepted"])
        self.assertEqual(feedback["next_guided_policy_patch"]["terminal_blacklist"], [])
        self.assertEqual(feedback["frontier_research_targets"], [])
        self.assertEqual(feedback["query_hints"][0]["hint_type"], "large_atom_jump")
        self.assertEqual(state.artifacts["route_failure_feedback"], feedback)

    def test_route_failure_feedback_extracts_blacklist_and_frontier_targets(self):
        feedback = compile_route_failure_feedback(
            {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "reasons": ["hidden_nonstock_reactants", "advanced_same_scaffold_terminal"],
                "rejected_terminal_list": [
                    {
                        "smiles": "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
                        "canonical_smiles": "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
                        "heavy_atoms": 25,
                        "target_similarity": 0.18,
                        "reason": "advanced_same_scaffold_terminal",
                    }
                ],
                "failure_events": [
                    {
                        "reason": "hidden_nonstock_reactants",
                        "details": {
                            "sample": {
                                "smiles": "COC(=O)CCO",
                                "canonical_smiles": "COC(=O)CCO",
                                "heavy_atoms": 7,
                                "target_similarity": 0.8,
                            }
                        },
                    }
                ],
            },
            case_id="fluvastatin",
            target_name="fluvastatin",
        )

        self.assertTrue(feedback["accepted"])
        self.assertIn("COC(=O)CCO", feedback["next_guided_policy_patch"]["terminal_blacklist"])
        self.assertEqual(feedback["frontier_research_targets"][0]["required_action"], "find_upstream_synthesis_or_disconnection")
        self.assertTrue(any("fluvastatin synthesis intermediate" in row["query"] for row in feedback["query_hints"]))

    def test_route_failure_feedback_marks_advanced_same_scaffold_terminal_as_frontier(self):
        precursor = "C[C@]12CC[C@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](c3ccc(=O)oc3)[C@@H](O)C[C@]12O"
        feedback = compile_route_failure_feedback(
            {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "reasons": ["advanced_same_scaffold_terminal"],
                "rejected_terminal_list": [
                    {
                        "smiles": precursor,
                        "canonical_smiles": precursor,
                        "heavy_atoms": 29,
                        "target_similarity": 0.76,
                        "reason": "advanced_same_scaffold_terminal",
                    }
                ],
            },
            case_id="bufotalin",
            target_name="bufotalin",
        )

        self.assertTrue(feedback["accepted"])
        self.assertEqual(feedback["frontier_research_targets"][0]["canonical_smiles"], precursor)
        self.assertEqual(feedback["frontier_research_targets"][0]["frontier_role"], "advanced_same_scaffold_terminal")
        self.assertIn(precursor, feedback["next_guided_policy_patch"]["preferred_subgoals"])

    def test_o_glycoside_literature_plan_emits_partial_template_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="phenolic_glycoside_harness",
                target_smiles=O_GLYCOSIDE_SMILES,
                family_hint="O-glycoside, glycoside, sugar, glycosylation",
                output_dir=tmp,
                use_live_planner=False,
            )
            out = Path(tmp)
            workflow_result = json.loads((out / "smiles_first_workflow_result.json").read_text(encoding="utf-8"))
            package_path = Path(workflow_result["artifacts"]["hybrid_route_package"])
            package = json.loads(package_path.read_text(encoding="utf-8"))

        self.assertEqual(result["workflow_plan"]["recommended_strategy"], "literature_first")
        self.assertEqual(result["workflow_plan"]["planner_decision_reason"], "glycoside_or_o_glycoside_like")
        self.assertEqual(result["final_verdict"]["verdict"], "partial_anchor_only_not_solved")
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertTrue(any(t.get("reaction_class") == "glycosylation" for t in package["strategy_templates"]))
        self.assertTrue(any(c.get("candidate_kind") == "forward_surrogate" for c in package["literature_candidates"]))

    def test_open_structure_research_internal_timeout_rejects_launcher_success(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            open_dir = Path(cmd[cmd.index("--output-dir") + 1])
            open_dir.mkdir(parents=True, exist_ok=True)
            event_log = open_dir / "codex_events.jsonl"
            event_log.write_text('{"type":"turn.started"}\n', encoding="utf-8")
            (open_dir / "open_agent_run_record.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_codex_structure_template_run.v1",
                        "run_dir": str(open_dir),
                        "exit_code": None,
                        "timeout_s": 1.0,
                        "error": "timeout",
                        "metadata": {
                            "stream_jsonl": True,
                            "event_log_path": str(event_log),
                        },
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=str(open_dir) + "\n", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
                budget=HarnessBudget(open_research_timeout_s=1.0),
            )
            with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                result = run_open_structure_research_agent(state, {})

            persisted = json.loads((Path(tmp) / "open_structure_research_result.json").read_text(encoding="utf-8"))

        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("open_agent_timeout", result["reasons"])
        self.assertIn("codex_events_missing_turn_completed", result["reasons"])
        self.assertIn("missing_open_agent_artifact:structure_template_report.md", result["reasons"])
        self.assertFalse(persisted["accepted"])
        self.assertFalse(state.artifacts["open_structure_research"]["accepted"])

    def test_tool_failure_takes_precedence_over_partial_anchor_verdict(self):
        verdict = emit_final_verdict(
            {
                "case_id": "open_failure_with_anchor",
                "preflight": {"accepted": True},
                "artifacts": {
                    "smiles_first": {
                        "validation": {
                            "route_status": "partial_anchor",
                        },
                    },
                },
                "tool_calls": [
                    {
                        "tool_name": "run_open_structure_research_agent",
                        "status": "rejected",
                        "reasons": ["open_agent_timeout"],
                    }
                ],
                "validations": [],
            }
        )

        self.assertEqual(verdict.verdict, "needs_followup")
        self.assertFalse(verdict.solved)
        self.assertIn("tool_execution_failed", verdict.reasons)
        self.assertIn("open_agent_timeout", verdict.reasons)

    def test_open_structure_prompt_is_self_contained_target_scoped_and_schema_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "smiles_first_literature_workflow").mkdir()
            prompt = _read_or_build_prompt(
                args=Namespace(
                    prompt_path="",
                    context_root=str(root),
                    target_name="fluvastatin",
                    target_smiles="CCO",
                    frontier_smiles="CCO",
                ),
                run_dir=root / "open",
            )

        self.assertIn("Do not rely on an external prior run or undocumented \"style\"", prompt)
        self.assertIn("open_research_manifest.json", prompt)
        self.assertIn("binding search budget and source-order contract", prompt)
        self.assertIn("Do not use shell for environment discovery", prompt)
        self.assertIn("Do not use curl/wget/urllib/requests/httpx", prompt)
        self.assertIn("Do not run RDKit availability/version probes", prompt)
        self.assertIn("harness prefetch checkpoint seed", prompt)
        self.assertIn("harness_local_downstream_seed.json", prompt)
        self.assertIn("retrieval_prefetch.path", prompt)
        self.assertIn("retrieval_prefetch.source_detail_extraction_pack_path", prompt)
        self.assertIn("source-detail extraction queue", prompt)
        self.assertIn("source_detail_resolution.path", prompt)
        self.assertIn("source-detail resolution pack", prompt)
        self.assertIn("source_material_locator.path", prompt)
        self.assertIn("metadata-only publisher landing/SI/material URLs", prompt)
        self.assertIn("evidence/source_detail_curator_records.json", prompt)
        self.assertIn("source_detail_curator_records.v1", prompt)
        self.assertIn("codex_source_text_translation", prompt)
        self.assertIn("structure_derivation", prompt)
        self.assertIn("source_excerpt", prompt)
        self.assertIn("translate source text", prompt)
        self.assertIn("Do not store source full text", prompt)
        self.assertIn("harness-owned PubChem/CrossRef/PubMed", prompt)
        self.assertIn("Consume `retrieval_prefetch.compound_seed_rows`", prompt)
        self.assertIn("source `harness_retrieval_prefetch`", prompt)
        self.assertIn("smiles_first_literature_workflow", prompt)
        self.assertIn("prior_experience.self_evo_memory", prompt)
        self.assertIn("Re-check current-target relation", prompt)
        self.assertIn("prior_experience.route_failure_feedback", prompt)
        self.assertIn("terminal_blacklist out of closure claims", prompt)
        self.assertIn("Route verifier report, if present", prompt)
        self.assertIn("Route failure feedback, if present", prompt)
        self.assertIn("Do not read the large native ChemEnzy raw route dump by default", prompt)
        self.assertNotIn("Native ChemEnzy raw result", prompt)
        self.assertIn("source_relation_policy", prompt)
        self.assertIn("candidate_generation_policy", prompt)
        self.assertIn("Named intermediate source priority", prompt)
        self.assertNotIn("10.1021/acs.orglett.0c03251", prompt)
        self.assertIn("bufadienolide/bufotalin sources", prompt)
        self.assertIn("method/style references only", prompt)

    def test_open_structure_research_schema_errors_reject_launcher_success(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            open_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (open_dir / "evidence").mkdir(parents=True, exist_ok=True)
            event_log = open_dir / "codex_events.jsonl"
            event_log.write_text(
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}) + "\n",
                encoding="utf-8",
            )
            (open_dir / "open_agent_run_record.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_codex_structure_template_run.v1",
                        "run_dir": str(open_dir),
                        "exit_code": 0,
                        "metadata": {
                            "stream_jsonl": True,
                            "event_log_path": str(event_log),
                        },
                    }
                ),
                encoding="utf-8",
            )
            for name in [
                "structure_template_report.md",
                "validated_compounds.smi",
            ]:
                (open_dir / name).write_text("placeholder\n", encoding="utf-8")
            # Present all JSON artifacts, but make one schema-incomplete.
            (open_dir / "structure_template_candidates.json").write_text(
                json.dumps({"schema_version": "open_structure_template_candidates.v1", "case_id": "ethanol"}),
                encoding="utf-8",
            )
            _write_downstream_consumables(open_dir, case_id="ethanol")
            (open_dir / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "ethanol",
                        "source_relation_policy": {},
                        "sources": [],
                        "excluded_sources": [],
                        "search_log": [],
                    }
                ),
                encoding="utf-8",
            )
            (open_dir / "evidence" / "pubchem_validated_compounds.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_pubchem_validated_compounds.v1",
                        "case_id": "ethanol",
                        "compound_source_policy": {},
                        "compounds": [],
                        "rejected_items": [],
                    }
                ),
                encoding="utf-8",
            )
            (open_dir / "open_agent_audit.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_structure_agent_audit.v1",
                        "case_id": "ethanol",
                        "final_status": "partial_anchor",
                        "solved": False,
                        "production_kb_promotion": False,
                        "checks": [],
                        "limitations": [],
                        "next_actions": [],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=str(open_dir) + "\n", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
                budget=HarnessBudget(open_research_timeout_s=1.0),
            )
            with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                result = run_open_structure_research_agent(state, {})

        self.assertFalse(result["accepted"])
        self.assertIn(
            "open_agent_json_missing_key:structure_template_candidates.json:target",
            result["reasons"],
        )

    def test_open_structure_research_compiles_downstream_consumables(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            open_dir = Path(cmd[cmd.index("--output-dir") + 1])
            _write_valid_open_research_shell(open_dir, case_id="ethanol")
            _write_compilable_downstream_consumables(open_dir, case_id="ethanol")
            return subprocess.CompletedProcess(cmd, 0, stdout=str(open_dir) + "\n", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
                budget=HarnessBudget(open_research_timeout_s=1.0),
            )
            with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                result = run_open_structure_research_agent(state, {})
            open_dir = Path(result["output_dir"])
            self.assertTrue(result["accepted"])
            self.assertTrue(result["compiled_downstream"]["accepted"])
            self.assertEqual(result["compiled_downstream"]["summary"]["guided_policy_count"], 2)
            self.assertEqual(result["compiled_downstream"]["summary"]["route_expansion_task_count"], 1)
            self.assertEqual(result["compiled_downstream"]["summary"]["self_evo_staging_candidate_count"], 1)
            self.assertTrue((open_dir / "compiled_downstream_consumables.json").exists())
            self.assertTrue((open_dir / "compiled_guided_chemenzy_requests.json").exists())
            self.assertTrue((open_dir / "compiled_route_expansion_tasks.json").exists())
            self.assertTrue((open_dir / "compiled_executable_template_maturity.json").exists())
            self.assertTrue((open_dir / "self_evo_staging_kb.json").exists())
            self.assertIn("compiled_downstream", state.artifacts)

    def test_rejected_open_research_with_harness_seed_allows_downstream_continuation(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            open_dir = Path(cmd[cmd.index("--output-dir") + 1])
            _write_valid_open_research_shell(open_dir, case_id="ethanol")
            _write_compilable_downstream_consumables(open_dir, case_id="ethanol")
            (open_dir / "harness_local_downstream_seed.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_research_local_downstream_seed.v1",
                        "accepted": True,
                        "generated_by": "harness_local_downstream_seed",
                        "downstream_consumables": json.loads(
                            (open_dir / "downstream_consumables.json").read_text(encoding="utf-8")
                        ),
                    }
                ),
                encoding="utf-8",
            )
            event_log = open_dir / "codex_events.jsonl"
            event_log.write_text(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "function_call",
                            "name": "shell",
                            "arguments": "sed -n '1,260p' /tmp/chemenzy_native_raw_result.json",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (open_dir / "open_agent_run_record.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_codex_structure_template_run.v1",
                        "run_dir": str(open_dir),
                        "exit_code": None,
                        "error": "timeout",
                        "metadata": {
                            "stream_jsonl": True,
                            "event_log_path": str(event_log),
                        },
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 1, stdout=str(open_dir), stderr="timeout")

        plan = _plan(
            "ethanol",
            strategy="literature_first",
            tools=[
                {"tool_name": "run_open_structure_research_agent", "payload": {}},
                {"tool_name": "run_guided_chemenzy_rerun", "payload": {}},
                {"tool_name": "run_route_expansion_subgoal_search", "payload": {}},
                {"tool_name": "run_self_evo_replay_gate", "payload": {}},
                {"tool_name": "validate_artifact_bundle", "payload": {}},
                {"tool_name": "emit_final_verdict", "payload": {}},
            ],
            planner_decision_reason="user_requested_literature",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                result = run_codex_entry_controller(
                    target_name="ethanol",
                    target_smiles="CCO",
                    output_dir=tmp,
                    planner_plan=plan,
                    use_live_planner=False,
                    budget=HarnessBudget(open_research_timeout_s=1.0),
                    mock_tool_results={
                        "run_guided_chemenzy_rerun": {
                            "schema_version": "guided_chemenzy_rerun_result.v1",
                            "accepted": True,
                            "status": "skipped_by_mock",
                        },
                        "run_route_expansion_subgoal_search": {
                            "schema_version": "route_expansion_subgoal_search_result.v1",
                            "accepted": True,
                            "status": "skipped_by_mock",
                        },
                    },
                )
            calls = [json.loads(line) for line in (Path(tmp) / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(
            [call["tool_name"] for call in calls],
            [
                "run_open_structure_research_agent",
                "run_guided_chemenzy_rerun",
                "run_route_expansion_subgoal_search",
                "run_self_evo_replay_gate",
                "validate_artifact_bundle",
            ],
        )
        open_research = result["artifact_bundle"]["artifacts"]["open_structure_research"]
        self.assertTrue(open_research["accepted"])
        self.assertIn(open_research["compiled_downstream"]["source"], {"downstream_consumables", "harness_local_downstream_seed"})
        self.assertIn("guided_chemenzy", result["artifact_bundle"]["artifacts"])
        self.assertIn("route_expansion_subgoal_search", result["artifact_bundle"]["artifacts"])
        self.assertEqual(result["final_verdict"]["verdict"], "partial_anchor_only_not_solved")

    def test_open_research_downstream_compiler_prefers_harness_local_seed_after_agent_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_downstream_consumables(root, case_id="ethanol")
            seed_source = root / "seed_source"
            seed_source.mkdir()
            _write_compilable_downstream_consumables(seed_source, case_id="ethanol")
            (root / "harness_local_downstream_seed.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_research_local_downstream_seed.v1",
                        "accepted": True,
                        "generated_by": "harness_local_downstream_seed",
                        "downstream_consumables": json.loads(
                            (root / "seed_source" / "downstream_consumables.json").read_text(encoding="utf-8")
                        ),
                    }
                ),
                encoding="utf-8",
            )
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
            )
            result = _compile_open_research_downstream(
                state=state,
                open_dir=root,
                target_smiles="CCO",
                prefer_local_seed=True,
            )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["source"], "harness_local_downstream_seed")
        self.assertEqual(result["summary"]["guided_policy_count"], 2)
        self.assertEqual(result["summary"]["route_expansion_task_count"], 1)

    def test_open_research_manual_curator_draft_is_advisory_without_page_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            evidence_dir.mkdir()
            _write_downstream_consumables(root, case_id="ethanol")
            pack_path = evidence_dir / "source_detail_extraction_pack.json"
            pack_path.write_text(
                json.dumps(
                    {
                        "schema_version": "source_detail_extraction_pack.v1",
                        "target": {"name": "ethanol", "smiles": "CCO"},
                        "queue": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "open_research_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_structure_research_manifest.v1",
                        "target": {"name": "ethanol", "smiles": "CCO"},
                        "retrieval_prefetch": {
                            "source_detail_extraction_pack_path": str(pack_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_detail_curator_records_path(root).write_text(
                json.dumps(
                    {
                        "schema_version": "source_detail_curator_records.v1",
                        "records": [
                            {
                                "schema_version": "source_detail_curator_record.v1",
                                "record_id": "curated_ethanol_step",
                                "source_ref": "doi:10.0000/curated",
                                "evidence_refs": ["ev_curated"],
                                "provenance": "manual_structured_extraction",
                                "steps": [
                                    {
                                        "step_id": "curated_ethanol_step_1",
                                        "segment_id": "curated_ethanol_segment",
                                        "product_smiles": "CCO",
                                        "reactant_smiles": ["CC", "O"],
                                        "condition_candidate": {
                                            "solvent": "water",
                                            "temperature": "25 C",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
            )
            result = _compile_open_research_downstream(
                state=state,
                open_dir=root,
                target_smiles="CCO",
                prefer_local_seed=False,
            )
            plugin = json.loads((root / "compiled_literature_template_plugin.json").read_text(encoding="utf-8"))

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(result["source"], "curator_augmented_downstream")
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        self.assertEqual(result["summary"]["one_step_row_count"], 0)
        self.assertEqual(result["summary"]["template_card_count"], 1)
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 1)
        card = plugin["template_cards"][0]
        self.assertEqual(card["applicability"]["allowed_use"], "mechanistic_template_hint_only")
        self.assertFalse(card["applicability"]["direct_one_step_consumption"])

    def test_open_agent_output_validation_requires_retrieval_prefetch_consumption(self):
        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            if "/cids/" in url:
                return {"IdentifierList": {"CID": [446155]}}
            if "/property/" in url:
                return {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 446155,
                                "ConnectivitySMILES": "CCO",
                                "SMILES": "CCO",
                                "IUPACName": "ethanol",
                                "MolecularFormula": "C2H6O",
                                "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                            }
                        ]
                    }
                }
            if "api.crossref.org" in url:
                return {"message": {"items": [{"DOI": "10.0000/example", "title": ["Synthesis of ethanol"]}]}}
            return {"esearchresult": {"count": "0", "idlist": []}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_valid_open_research_shell(root, case_id="ethanol")
            _write_downstream_consumables(root, case_id="ethanol")
            prefetch_open_research_evidence(
                {
                    "target": {"name": "ethanol", "smiles": "CCO"},
                    "query_plan": {
                        "pubchem_name_queries": ["ethanol"],
                        "crossref_queries": ["ethanol synthesis"],
                    },
                },
                output_dir=root,
                fetch_json=fake_fetch,
            )
            ignored = _validate_open_agent_outputs(
                run_dir=root,
                record={
                    "exit_code": 0,
                    "metadata": {
                        "stream_jsonl": True,
                        "event_log_path": str(root / "codex_events.jsonl"),
                    },
                },
            )

            (root / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "ethanol",
                        "source_relation_policy": {},
                        "sources": [{"doi": "10.0000/example", "source": "harness_retrieval_prefetch"}],
                        "excluded_sources": [],
                        "search_log": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "evidence" / "pubchem_validated_compounds.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_pubchem_validated_compounds.v1",
                        "case_id": "ethanol",
                        "compound_source_policy": {},
                        "compounds": [{"inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "canonical_smiles": "CCO"}],
                        "rejected_items": [],
                    }
                ),
                encoding="utf-8",
            )
            consumed = _validate_open_agent_outputs(
                run_dir=root,
                record={
                    "exit_code": 0,
                    "metadata": {
                        "stream_jsonl": True,
                        "event_log_path": str(root / "codex_events.jsonl"),
                    },
                },
            )

        self.assertFalse(ignored["accepted"])
        self.assertIn("retrieval_prefetch_source_seed_not_consumed_or_explained", ignored["reasons"])
        self.assertIn("retrieval_prefetch_compound_seed_not_consumed_or_explained", ignored["reasons"])
        self.assertTrue(consumed["accepted"], consumed["reasons"])
        self.assertTrue(consumed["retrieval_prefetch_consumption"]["accepted"])

    def test_open_structure_seed_only_checkpoint_does_not_pass_harness_tool(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            open_dir = Path(cmd[cmd.index("--output-dir") + 1])
            _write_valid_open_research_shell(open_dir, case_id="ethanol")
            _write_downstream_consumables(open_dir, case_id="ethanol")
            return subprocess.CompletedProcess(cmd, 0, stdout=str(open_dir) + "\n", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
                budget=HarnessBudget(open_research_timeout_s=1.0),
            )
            with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                result = run_open_structure_research_agent(state, {})

        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("compiled_downstream", result)
        self.assertIn("no_compiled_downstream_assets", result["reasons"])

    def test_open_structure_checkpoint_valid_timeout_is_accepted(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            open_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (open_dir / "evidence").mkdir(parents=True, exist_ok=True)
            event_log = open_dir / "codex_events.jsonl"
            event_log.write_text(
                json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "files written"}})
                + "\n",
                encoding="utf-8",
            )
            (open_dir / "open_agent_run_record.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_codex_structure_template_run.v1",
                        "run_dir": str(open_dir),
                        "exit_code": None,
                        "error": "timeout",
                        "metadata": {
                            "stream_jsonl": True,
                            "event_log_path": str(event_log),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (open_dir / "structure_template_report.md").write_text("checkpoint\n", encoding="utf-8")
            (open_dir / "validated_compounds.smi").write_text("CCO cmp_ethanol\n", encoding="utf-8")
            (open_dir / "structure_template_candidates.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_structure_template_candidates.v1",
                        "case_id": "ethanol",
                        "target": {},
                        "candidate_generation_policy": {},
                        "candidates": [],
                        "rejected_items": [],
                        "source_refs": [],
                        "audit_summary": {"solved": False, "production_kb_promotion": False},
                    }
                ),
                encoding="utf-8",
            )
            _write_compilable_downstream_consumables(open_dir, case_id="ethanol")
            (open_dir / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "ethanol",
                        "source_relation_policy": {},
                        "sources": [],
                        "excluded_sources": [],
                        "search_log": [],
                    }
                ),
                encoding="utf-8",
            )
            (open_dir / "evidence" / "pubchem_validated_compounds.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_pubchem_validated_compounds.v1",
                        "case_id": "ethanol",
                        "compound_source_policy": {},
                        "compounds": [],
                        "rejected_items": [],
                    }
                ),
                encoding="utf-8",
            )
            (open_dir / "open_agent_audit.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_structure_agent_audit.v1",
                        "case_id": "ethanol",
                        "final_status": "partial_anchor",
                        "solved": False,
                        "production_kb_promotion": False,
                        "checks": [],
                        "limitations": [],
                        "next_actions": [],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 1, stdout=str(open_dir) + "\n", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO"},
                preflight={"case_id": "ethanol"},
                budget=HarnessBudget(open_research_timeout_s=1.0),
            )
            with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                result = run_open_structure_research_agent(state, {})

        self.assertTrue(result["accepted"], result.get("reasons"))
        self.assertTrue(result["output_validation"]["checkpoint_after_timeout"])
        self.assertIn("checkpoint_valid_but_turn_timeout", result["output_validation"]["warnings"])
        self.assertEqual(result.get("reasons"), None)

    def test_chemenzy_tool_passes_condition_prediction_flags(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    result = execute_local_tool(
                        "run_chemenzy",
                        {
                            "enable_condition_prediction": True,
                            "enable_enzyme_assignment": True,
                            "enable_easifa": True,
                        },
                        state,
                    )

        request = result.output["result"]["request_echo"]
        self.assertTrue(request["enable_condition_prediction"])
        self.assertTrue(request["enable_enzyme_assignment"])
        self.assertTrue(request["enable_easifa"])

    def test_chemenzy_tool_defaults_to_strict_building_block_stock(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    result = execute_local_tool("run_chemenzy", {}, state)

        request = result.output["result"]["request_echo"]
        self.assertEqual(request["stock_mode"], "building-block")
        self.assertEqual(_stock_names_from_payload(request), ["PaRotes_n1-stock"])

    def test_chemenzy_tool_enforces_deep_small_stock_defaults_for_complex_targets(self):
        captured_requests = []

        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            captured_requests.append(request)
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            target = TargetInput(
                target_name="atorvastatin",
                target_smiles=ATORVASTATIN_SMILES,
                family_hint="statin synthetic atorvastatin",
            )
            preflight = run_preflight(target)
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
            )
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    result = execute_local_tool("run_chemenzy", {"stock_mode": "commercial"}, state)

        request = captured_requests[0]
        self.assertEqual(result.status, "accepted")
        self.assertEqual(request["search_preset"], "thorough")
        self.assertEqual(request["max_steps"], 20)
        self.assertEqual(request["chem_enzy_iterations"], 50)
        self.assertEqual(request["chem_enzy_expansion_topk"], 100)
        self.assertEqual(request["stock_mode"], "building-block")
        self.assertEqual(_stock_names_from_payload(request), ["PaRotes_n1-stock"])
        boundary = request["harness_search_boundary"]
        self.assertTrue(boundary["small_stock_default"])
        self.assertIn("coerced_broad_or_unknown_stock_mode_to_building_block", boundary["stock_policy_actions"])

    def test_guided_chemenzy_rerun_uses_compiled_policy_and_plugin_flags(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {
                    "policy_payloads": [
                        {
                            "schema_version": "chem_enzy_search_policy.v1",
                            "policy_id": "ethanol_policy_1",
                            "operator_id": "ethanol_guided_1",
                            "case_id": "ethanol",
                            "evidence_refs": ["ev1"],
                            "terminal_blacklist": [],
                            "anchor_whitelist": [],
                            "preferred_subgoal": {"preferred_subgoals": ["ethanol precursor"]},
                            "source_budget": {"preferred_reaction_classes": ["literature_guided"]},
                            "rerun_reason": "unit_test",
                            "budget": {
                                "max_reruns": 1,
                                "max_iterations": 12,
                                "max_depth": 9,
                                "expansion_topk": 33,
                            },
                            "mode": "guided",
                            "compiler_metadata": {},
                        }
                    ]
                },
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "top_k": 2,
                        "max_added": 2,
                        "template_cards": [],
                        "one_step_rows": [
                            {
                                "reactants": "CC.O",
                                "scores": 0.7,
                                "costs": None,
                                "template": {
                                    "source": "literature_template_plugin",
                                    "source_model": "literature_template_plugin",
                                    "requires_audit": True,
                                    "not_lab_procedure": True,
                                    "no_solved_claim": True,
                                    "template_validation_report": {
                                        "allowed_for_one_step_source": True,
                                        "accepted": True,
                                        "reasons": [],
                                    },
                                    "template_applicability_report": {
                                        "target_smiles": "CCO",
                                        "frontier_smiles": "CCO",
                                    },
                                },
                                "templates": {},
                                "model_full_name": "autoplanner.literature_template_plugin",
                                "weight": 1.0,
                            }
                        ],
                        "requires_audit": True,
                        "not_raw_reaction_injection": True,
                    }
                },
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    result = execute_local_tool("run_guided_chemenzy_rerun", {}, state)
            request = json.loads((Path(tmp) / "guided_chemenzy_request.json").read_text(encoding="utf-8"))

        self.assertTrue(result.output["result"]["accepted"])
        self.assertEqual(request["max_steps"], 9)
        self.assertEqual(request["chem_enzy_iterations"], 12)
        # A compiled policy without an out-of-band operator capability remains
        # host-profile authority, so the simple-target standard floor is 50.
        self.assertEqual(request["chem_enzy_expansion_topk"], 50)
        self.assertEqual(request["chem_enzy_search_policy"]["policy_id"], "ethanol_policy_1")
        self.assertTrue(request["literature_template_plugin"]["enabled"])
        self.assertEqual(request["literature_template_plugin"]["one_step_rows"][0]["reactants"], "CC.O")

    def test_guided_chemenzy_rerun_unwraps_compiled_downstream_harness_result(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            open_dir = root / "open_structure_research"
            compiled = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {"policy_payloads": [_policy_payload("ethanol_policy_1")]},
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "top_k": 2,
                        "max_added": 2,
                        "template_cards": [],
                        "one_step_rows": [_plugin_row("CC.O")],
                        "requires_audit": True,
                        "not_raw_reaction_injection": True,
                    }
                },
            }
            wrapper = _write_compiled_downstream_wrapper(open_dir, compiled)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = wrapper
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    result = execute_local_tool("run_guided_chemenzy_rerun", {}, state)
            request = json.loads((root / "guided_chemenzy_request.json").read_text(encoding="utf-8"))

        self.assertEqual(result.status, "accepted")
        self.assertEqual(request["chem_enzy_search_policy"]["policy_id"], "ethanol_policy_1")
        self.assertEqual(request["literature_template_plugin"]["one_step_rows"][0]["reactants"], "CC.O")

    def test_guided_chemenzy_rerun_uses_plugin_only_policy_when_rows_exist(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {"policy_payloads": []},
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "top_k": 2,
                        "max_added": 2,
                        "template_cards": [],
                        "one_step_rows": [_plugin_row("CC.O")],
                        "requires_audit": True,
                        "not_raw_reaction_injection": True,
                    }
                },
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    result = execute_local_tool("run_guided_chemenzy_rerun", {}, state)
            request = json.loads((root / "guided_chemenzy_request.json").read_text(encoding="utf-8"))

        self.assertEqual(result.status, "accepted")
        self.assertEqual(state.guided_chemenzy_runs, 1)
        self.assertEqual(
            request["chem_enzy_search_policy"]["rerun_reason"],
            "compiled_literature_template_plugin_available",
        )
        self.assertTrue(request["chem_enzy_search_policy"]["source_budget"]["plugin_only_guided_rerun"])
        self.assertEqual(request["literature_template_plugin"]["one_step_rows"][0]["reactants"], "CC.O")

    def test_analogical_retrosynthesis_hypotheses_are_advisory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={
                    "target_name": "anthranilate_imide_target",
                    "target_smiles": "COC(=O)c1ccccc1N1C(=O)CCC1=O",
                    "family_hint": "alkaloid analogue",
                },
                preflight={"case_id": "anthranilate_imide_target"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "one_step_rows": [
                            _plugin_row_with_trace(
                                product="CC(C)(C)OC(=O)Nc1cc(O)ccc1C=O",
                                reactants=["CC(C)(C)OC(=O)Nc1cc(O[Si](C)(C)C(C)(C)C)ccc1C=O"],
                                template_id="source_detail_exact_step:tbs_deprotection",
                                reagent="TBAF",
                            )
                        ],
                    }
                },
            }
            record = execute_local_tool("build_analogical_retrosynthesis_hypotheses", {}, state)
            persisted = json.loads((root / "analogical_retrosynthesis_hypotheses.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertTrue(persisted["accepted"])
        self.assertEqual(persisted["mode"], "advisory_inspiration_only")
        self.assertTrue(persisted["no_solved_claim"])
        self.assertTrue(persisted["production_write_blocked"])
        self.assertIn("aryl_ester_or_anthranilate_sidechain", persisted["target"]["handles"])
        self.assertGreaterEqual(persisted["hypothesis_count"], 1)
        self.assertTrue(persisted["search_policy_patch"]["require_target_core_retention"])
        self.assertTrue(
            any(item["hypothesis_id"] == "target_side_late_stage_aryl_ester_disconnection" for item in persisted["hypotheses"])
        )

    def test_guided_chemenzy_rerun_merges_analogical_policy_patch(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {"policy_payloads": [_policy_payload("ethanol_policy_1")]},
            }
            state.artifacts["analogical_retrosynthesis_hypotheses"] = {
                "schema_version": "analogical_retrosynthesis_hypotheses.v1",
                "accepted": True,
                "mode": "advisory_inspiration_only",
                "source_row_count": 1,
                "hypothesis_count": 1,
                "search_policy_patch": {
                    "schema_version": "analogical_search_policy_patch.v1",
                    "enabled": True,
                    "preferred_reaction_classes": ["target_side_analogical_disconnection", "n_alkylation"],
                    "active_failure_modes": ["large_atom_jump"],
                    "require_target_core_retention": True,
                    "max_unexplained_heavy_atom_delta": 20,
                },
                "hypotheses": [
                    {
                        "hypothesis_id": "analogy_1",
                        "inspiration_type": "reaction_family_transfer",
                        "reaction_family": "n_alkylation",
                        "target_side_attempt": {"role": "chemist_advisory_disconnection"},
                        "required_verification": ["deterministic_route_verifier_must_accept_before_solved_claim"],
                    }
                ],
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    execute_local_tool("run_guided_chemenzy_rerun", {}, state)
            request = json.loads((root / "guided_chemenzy_request.json").read_text(encoding="utf-8"))

        policy = request["chem_enzy_search_policy"]
        self.assertTrue(policy["source_budget"]["analogical_inspiration_enabled"])
        self.assertTrue(policy["source_budget"]["require_target_core_retention"])
        self.assertIn("n_alkylation", policy["source_budget"]["preferred_reaction_classes"])
        self.assertEqual(policy["compiler_metadata"]["analogical_retrosynthesis"]["hypothesis_count"], 1)
        self.assertEqual(
            policy["preferred_subgoal"]["analogical_retrosynthesis_hypotheses"][0]["hypothesis_id"],
            "analogy_1",
        )

    def test_guided_chemenzy_reports_plugin_enabled_but_not_invoked(self):
        target = "CCCCCCCCCCCCCCCCCCCCCCCCCC"

        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "target": request["target_smiles"],
                        "raw_backend_metadata": {
                            "literature_template_plugin": {
                                "schema_version": "literature_one_step_plugin.stats.v1",
                                "enabled": True,
                                "calls": 0,
                                "candidate_templates": 0,
                                "instantiated_candidates": 0,
                                "added_candidates": 0,
                            }
                        },
                        "routes": [
                            {
                                "route_rank": 0,
                                "score": 1.0,
                                "n_steps": 1,
                                "steps": [
                                    {
                                        "index": 0,
                                        "product": request["target_smiles"],
                                        "main_reactant": "C",
                                        "aux_reactants": [],
                                        "stock_status": {"C": True},
                                    }
                                ],
                                "metrics": {"terminal_reactants": ["C"], "terminal_stock_status": {"C": True}},
                            }
                        ],
                        "search_status": {"solved": True},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "long_alkane", "target_smiles": target, "family_hint": ""},
                preflight={"case_id": "long_alkane"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {"policy_payloads": [_policy_payload("long_alkane_policy_1")]},
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "top_k": 1,
                        "max_added": 1,
                        "template_cards": [],
                        "one_step_rows": [_plugin_row_with_trace(product="CCO", reactants=["CC"], template_id="source_detail_exact_step:ethanol")],
                        "requires_audit": True,
                        "not_raw_reaction_injection": True,
                    }
                },
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    record = execute_local_tool("run_guided_chemenzy_rerun", {}, state)

        result = record.output["result"]
        self.assertFalse(result["accepted"])
        self.assertIn("large_atom_jump", result["reasons"])
        self.assertIn("literature_template_plugin_not_invoked", result["reasons"])
        self.assertEqual(result["literature_template_plugin_runtime"]["calls"], 0)

    def test_untrusted_source_detail_draft_publishes_advisory_downstream_state(self):
        step = {
            "schema_version": "source_detail_route_step.v1",
            "step_id": "ethanol_step",
            "segment_id": "ethanol_segment",
            "source_ref": "doi:test",
            "evidence_refs": ["ev1"],
            "product_smiles": "CCO",
            "reactant_smiles": ["CC"],
            "relation_type": "exact",
            "condition_candidate": {
                "schema_version": "condition_candidate.v1",
                "step_id": "ethanol_step",
                "source_type": "exact",
                "condition_status": "evidence_backed",
                "reagent": "water",
                "evidence_refs": ["ev1"],
            },
            "applicability": {"status": "passed", "product_reconstruction_passed": True},
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            result = execute_local_tool(
                "compile_source_detail_chain_route",
                {
                    "source_detail_steps": [step],
                    "terminal_smiles": "CC",
                    "terminal_name": "ethane",
                },
                state,
            )

        self.assertEqual(result.status, "rejected")
        self.assertIn("source_detail_step_not_trusted_curated", result.reasons)
        compiled = state.artifacts["compiled_downstream"]
        self.assertEqual(compiled["schema_version"], "compiled_downstream_consumables.v1")
        self.assertTrue(compiled["literature_template_plugin"]["plugin_flags"]["enabled"])
        plugin = state.artifacts["compiled_downstream_payload"]["literature_template_plugin"]["plugin_flags"]
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 1)
        self.assertFalse(plugin["template_cards"][0]["applicability"]["direct_one_step_consumption"])
        self.assertEqual(
            plugin["template_cards"][0]["applicability"]["source_policy_decision"],
            "advisory_visual_template_hint",
        )

    def test_human_reviewed_codex_record_with_materialized_evidence_publishes_exact_row(self):
        curator_records = {
            "schema_version": "source_detail_curator_records.v1",
            "records": [
                {
                    "schema_version": "source_detail_curator_record.v1",
                    "record_id": "human_reviewed_codex_curator_record",
                    "source_ref": "doi:10.1000/revalidatable-stitch",
                    "source_title": "Human-reviewed source text route step",
                    "evidence_refs": [f"{SOURCE_EVIDENCE_MANIFEST_FIXTURE}::page:1"],
                    "provenance": "codex_source_text_translation",
                    "curation_status": "human_verified",
                    "validation_status": "deterministically_validated",
                    "authority": {"type": "human_curator", "id": "autoplanner-test-fixture"},
                    "source_excerpt": "Ethanol was converted to ethyl acetate.",
                    "structure_derivation": {
                        "basis": "codex_source_text_translation",
                        "source_locator": {
                            "source_ref": "doi:10.1000/revalidatable-stitch",
                            "source_title": "Human-reviewed source text route step",
                        },
                        "confidence": "medium_high_for_structure_and_route",
                        "tool_checks": {
                            "rdkit_product_parse": True,
                            "rdkit_reactant_parse": True,
                        },
                    },
                    "full_text_content_stored": False,
                    "procedure_text_stored": False,
                    "steps": [
                        {
                            "step_id": "ethyl_acetate",
                            "segment_id": "exact_codex_segment",
                            "product_smiles": "CCOC(C)=O",
                            "reactant_smiles": ["CCO"],
                            "relation_type": "exact",
                            "condition_candidate": {
                                "schema_version": "condition_candidate.v1",
                                "source_type": "exact",
                                "condition_status": "evidence_backed",
                                "reagent": "acetylation conditions",
                                "evidence_refs": [f"{SOURCE_EVIDENCE_MANIFEST_FIXTURE}::page:1"],
                            },
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(TRUSTED_LITERATURE_REGISTRY_FIXTURE)},
        ):
            root = Path(tmp)
            resolution = resolve_curator_records_to_source_detail_steps(
                curator_records,
                output_dir=root / "open_structure_research",
                target_name="ethyl acetate",
                target_smiles="CCOC(C)=O",
                source_ref="doi:10.1000/revalidatable-stitch",
            )
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethyl acetate", "target_smiles": "CCOC(C)=O", "family_hint": ""},
                preflight={"case_id": "ethyl_acetate"},
            )
            result = execute_local_tool(
                "compile_source_detail_chain_route",
                {
                    "source_detail_steps": resolution["source_detail_route_steps"],
                    "terminal_smiles": "CCO",
                    "terminal_name": "ethanol",
                },
                state,
            )

        self.assertEqual(resolution["summary"]["curator_step_count"], 1)
        self.assertEqual(resolution["summary"]["gap_count"], 0)
        self.assertEqual(result.status, "accepted", result.reasons)
        rows = state.artifacts["compiled_downstream"]["literature_template_plugin"]["one_step_rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reactants"], "CCO")
        trace = rows[0]["literature_template_trace"]
        self.assertEqual(trace["source_template_id"], "source_detail_exact_step:ethyl_acetate")
        self.assertEqual(
            trace["structure_derivation"]["source_locator"]["source_ref"],
            "doi:10.1000/revalidatable-stitch",
        )
        chain_step = result.output["result"]["chain_audit"]["chain"][0]
        self.assertTrue(chain_step["source_evidence"])
        self.assertEqual(chain_step["source_evidence"][0]["page_number"], 1)

    def test_guided_chemenzy_rerun_merges_self_evo_memory_plugin_rows(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {"policy_payloads": [_policy_payload("ethanol_policy_1")]},
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "top_k": 2,
                        "max_added": 2,
                        "template_cards": [],
                        "one_step_rows": [_plugin_row("CC.O")],
                        "requires_audit": True,
                        "not_raw_reaction_injection": True,
                    }
                },
            }
            memory = {
                "schema_version": "self_evo_reusable_memory.v1",
                "accepted": True,
                "case_id": "prior_statin_case",
                "future_use_policy": {
                    "not_route_evidence_until_current_target_relation_checked": True,
                    "requires_replay_gate_before_production": True,
                    "no_solved_claim": True,
                },
                "reusable_template_cards": [
                    {
                        "schema_version": "literature_template_card.v1",
                        "template_id": "memory_template",
                        "validation_status": "draft",
                        "template_level": "advisory_strategy",
                        "evidence_refs": ["ev_memory"],
                        "not_raw_reaction_injection": True,
                    }
                ],
                "reusable_one_step_rows": [_plugin_row("CC.N")],
            }
            (root / "self_evo_memory.json").write_text(json.dumps(memory), encoding="utf-8")
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    execute_local_tool(
                        "run_guided_chemenzy_rerun",
                        {"self_evo_memory_path": "self_evo_memory.json"},
                        state,
                    )
            request = json.loads((root / "guided_chemenzy_request.json").read_text(encoding="utf-8"))

        plugin = request["literature_template_plugin"]
        self.assertTrue(plugin["self_evo_memory"]["enabled"])
        self.assertEqual(plugin["self_evo_memory"]["case_id"], "prior_statin_case")
        self.assertEqual([row["reactants"] for row in plugin["one_step_rows"]], ["CC.O", "CC.N"])
        self.assertEqual(plugin["template_cards"][0]["template_id"], "memory_template")

    def test_guided_chemenzy_rerun_applies_route_failure_feedback_blacklist(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "chemenzy_web_result.v1",
                        "ok": True,
                        "accepted": True,
                        "request_echo": request,
                        "routes": [],
                        "search_status": {"solved": False},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        feedback = {
            "schema_version": "route_failure_feedback.v1",
            "accepted": True,
            "source_route_status": "fake_closed_rejected",
            "source_reasons": ["hidden_nonstock_reactants"],
            "frontier_research_targets": [{"canonical_smiles": "CCO"}],
            "next_guided_policy_patch": {
                "terminal_blacklist": ["CCO"],
                "preferred_subgoals": ["CCO"],
                "source_budget": {
                    "active_failure_modes": ["hidden_nonstock_reactants"],
                    "terminal_blacklist_roles": ["hidden_nonstock_advanced_intermediate"],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "guided_chemenzy": {"policy_payloads": [_policy_payload("ethanol_policy_1")]},
            }
            (root / "route_failure_feedback.json").write_text(json.dumps(feedback), encoding="utf-8")
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    execute_local_tool(
                        "run_guided_chemenzy_rerun",
                        {"route_failure_feedback_path": "route_failure_feedback.json"},
                        state,
                    )
            request = json.loads((root / "guided_chemenzy_request.json").read_text(encoding="utf-8"))

        policy = request["chem_enzy_search_policy"]
        self.assertIn("CCO", policy["terminal_blacklist"])
        self.assertIn("CCO", policy["preferred_subgoal"]["preferred_subgoals"])
        self.assertTrue(policy["route_failure_feedback"]["enabled"])
        self.assertIn("hidden_nonstock_reactants", policy["source_budget"]["active_failure_modes"])

    def test_route_expansion_subgoal_search_uses_frontier_as_child_target(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(request["target_smiles"])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "parent", "target_smiles": "CCCC", "family_hint": ""},
                preflight={"case_id": "parent_case"},
                budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "route_expansion": {
                    "tasks": [
                        {
                            "schema_version": "compiled_route_expansion_task.v1",
                            "accepted": True,
                            "task_id": "expand_ethanol_frontier",
                            "frontier_smiles": "CCO",
                            "preferred_subgoals": ["CCO"],
                            "policy_id": "ethanol_policy_1",
                            "max_depth": 11,
                            "max_iterations": 22,
                            "expansion_topk": 44,
                        }
                    ],
                    "policy_payloads": [_policy_payload("ethanol_policy_1")],
                },
                "literature_template_plugin": {"plugin_flags": {"enabled": False}},
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    record = execute_local_tool("run_route_expansion_subgoal_search", {}, state)
            request = json.loads(
                (root / "route_expansion_subgoals" / "01_expand_ethanol_frontier_request.json").read_text(
                    encoding="utf-8"
                )
            )
            persisted = json.loads((root / "route_expansion_subgoal_search_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertEqual(request["target_smiles"], "CCO")
        self.assertNotEqual(request["target_smiles"], "CCCC")
        self.assertEqual(request["max_steps"], 11)
        self.assertEqual(request["chem_enzy_iterations"], 22)
        self.assertEqual(request["chem_enzy_expansion_topk"], 50)
        self.assertEqual(persisted["accepted_subgoal_count"], 1)
        self.assertEqual(persisted["scope"], "route_expansion_subgoals")
        self.assertFalse(persisted["parent_route_solved"])
        self.assertTrue(persisted["no_parent_solved_claim"])
        self.assertTrue(persisted["not_parent_route_proof"])
        self.assertTrue(persisted["subgoals"][0]["verifier"]["accepted"])

    def test_route_expansion_subgoal_search_does_not_accept_small_multicomponent_reagent_as_parent_route(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(request["target_smiles"])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "large_parent", "target_smiles": BUFOTALIN_SMILES, "family_hint": ""},
                preflight={"case_id": "large_parent_case"},
                budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
            }
            child_target = {
                "schema_version": "route_expansion_child_target.v1",
                "name": "acetic_acid_component",
                "smiles": "CC(=O)O",
                "task_scope": "precursor_component",
                "precursor_set_smiles": f"{BUFOTALIN_SMILES}.CC(=O)O",
                "precursor_component_index": 1,
                "precursor_component_count": 2,
                "multi_component_precursor_set": True,
                "requires_precursor_set_stitching": True,
                "sibling_precursor_smiles": [BUFOTALIN_SMILES],
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    with patch(
                        "cascade_planner.harness.tools.verify_chemenzy_raw_routes",
                        return_value={"accepted": True, "route_status": "solved", "reasons": []},
                    ):
                        record = execute_local_tool(
                            "run_route_expansion_subgoal_search",
                            {"child_targets": [child_target]},
                            state,
                        )
            persisted = json.loads((root / "route_expansion_subgoal_search_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertFalse(persisted["accepted"])
        self.assertFalse(persisted["solved"])
        self.assertEqual(persisted["accepted_subgoal_count"], 0)
        row = persisted["subgoals"][0]
        self.assertTrue(row["verifier_accepted_before_parent_relevance_gate"])
        self.assertFalse(row["accepted"])
        self.assertFalse(row["solved"])
        self.assertEqual(row["route_status"], "child_component_not_parent_proximal")
        self.assertIn("child_component_not_parent_proximal", row["reasons"])

    def test_route_expansion_subgoal_search_unwraps_compiled_downstream_harness_result(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(request["target_smiles"])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            open_dir = root / "open_structure_research"
            compiled = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "route_expansion": {
                    "tasks": [
                        {
                            "schema_version": "compiled_route_expansion_task.v1",
                            "accepted": True,
                            "task_id": "expand_ethanol_frontier",
                            "frontier_smiles": "CCO",
                            "preferred_subgoals": ["CCO"],
                            "policy_id": "ethanol_policy_1",
                            "max_depth": 11,
                            "max_iterations": 22,
                            "expansion_topk": 44,
                        }
                    ],
                    "policy_payloads": [_policy_payload("ethanol_policy_1")],
                },
                "literature_template_plugin": {"plugin_flags": {"enabled": False}},
            }
            wrapper = _write_compiled_downstream_wrapper(open_dir, compiled)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "parent", "target_smiles": "CCCC", "family_hint": ""},
                preflight={"case_id": "parent_case"},
                budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
            )
            state.artifacts["compiled_downstream"] = wrapper
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    record = execute_local_tool("run_route_expansion_subgoal_search", {}, state)
            request = json.loads(
                (root / "route_expansion_subgoals" / "01_expand_ethanol_frontier_request.json").read_text(
                    encoding="utf-8"
                )
            )
            persisted = json.loads((root / "route_expansion_subgoal_search_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertEqual(request["target_smiles"], "CCO")
        self.assertEqual(request["chem_enzy_search_policy"]["policy_id"], "ethanol_policy_1")
        self.assertEqual(persisted["accepted_subgoal_count"], 1)

    def test_route_expansion_subgoal_search_uses_source_detail_child_target_and_plugin_flags(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(request["target_smiles"])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "parent", "target_smiles": "CCCC", "family_hint": ""},
                preflight={"case_id": "parent_case"},
                budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "route_expansion": {
                    "child_targets": [
                        {
                            "schema_version": "route_expansion_child_target.v1",
                            "child_target_id": "source_detail_reactant_1",
                            "name": "source_detail_reactant_1",
                            "smiles": "CCO",
                            "source": "source_detail_one_step_reactant",
                            "source_template_id": "source_detail_exact_step:step_1",
                            "source_ref": "pmc:source-detail",
                            "evidence_refs": ["ev_source_detail"],
                            "policy": _policy_payload("source_detail_child_policy"),
                            "max_depth": 7,
                            "max_iterations": 13,
                            "expansion_topk": 21,
                        }
                    ],
                    "tasks": [],
                    "policy_payloads": [],
                },
                "literature_template_plugin": {
                    "plugin_flags": {
                        "enabled": True,
                        "top_k": 2,
                        "max_added": 2,
                        "template_cards": [],
                        "one_step_rows": [_plugin_row("CC.O")],
                        "requires_audit": True,
                        "not_raw_reaction_injection": True,
                    }
                },
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    record = execute_local_tool("run_route_expansion_subgoal_search", {}, state)
            request = json.loads(
                (root / "route_expansion_subgoals" / "01_source_detail_reactant_1_request.json").read_text(
                    encoding="utf-8"
                )
            )
            persisted = json.loads((root / "route_expansion_subgoal_search_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertEqual(request["target_smiles"], "CCO")
        self.assertEqual(request["chem_enzy_search_policy"]["policy_id"], "source_detail_child_policy")
        self.assertEqual(request["max_steps"], 7)
        self.assertEqual(request["chem_enzy_iterations"], 13)
        self.assertEqual(request["chem_enzy_expansion_topk"], 50)
        self.assertTrue(request["literature_template_plugin"]["enabled"])
        self.assertEqual(request["literature_template_plugin"]["one_step_rows"][0]["reactants"], "CC.O")
        self.assertEqual(persisted["subgoals"][0]["subgoal"]["source"], "source_detail_one_step_reactant")
        self.assertEqual(persisted["accepted_subgoal_count"], 1)

    def test_route_expansion_subgoal_search_prioritizes_route_failure_frontier_before_advisory_anchor(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(request["target_smiles"])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "parent", "target_smiles": "CCCC", "family_hint": ""},
                preflight={"case_id": "parent_case"},
                budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "route_expansion": {
                    "child_targets": [
                        {
                            "schema_version": "route_expansion_child_target.v1",
                            "child_target_id": "resolved_anchor_1",
                            "name": "resolved_anchor_1",
                            "smiles": "CCCC",
                            "source": "resolved_advisory_anchor",
                            "policy": _policy_payload("resolved_anchor_policy"),
                        }
                    ],
                    "tasks": [],
                    "policy_payloads": [],
                },
            }
            state.artifacts["route_failure_feedback"] = {
                "schema_version": "route_failure_feedback.v1",
                "accepted": True,
                "frontier_research_targets": [
                    {
                        "canonical_smiles": "CCO",
                        "smiles": "CCO",
                        "frontier_role": "advanced_same_scaffold_terminal",
                        "reason": "advanced_same_scaffold_terminal",
                        "required_action": "find_upstream_synthesis_or_disconnection",
                    }
                ],
                "terminal_blacklist": [],
                "query_hints": [],
                "next_guided_policy_patch": {},
                "source_reasons": ["advanced_same_scaffold_terminal"],
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    record = execute_local_tool("run_route_expansion_subgoal_search", {}, state)
            request = json.loads(
                (root / "route_expansion_subgoals" / "01_advanced_same_scaffold_terminal_request.json").read_text(
                    encoding="utf-8"
                )
            )
            persisted = json.loads((root / "route_expansion_subgoal_search_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertEqual(request["target_smiles"], "CCO")
        self.assertEqual(persisted["subgoals"][0]["subgoal"]["source"], "route_failure_feedback")
        self.assertEqual(persisted["subgoals"][0]["subgoal"]["frontier_role"], "advanced_same_scaffold_terminal")
        self.assertEqual(persisted["subgoal_count"], 1)

    def test_route_expansion_exact_target_override_preempts_advisory_anchor(self):
        def fake_run(cmd, **kwargs):
            del kwargs
            request_path = Path(cmd[cmd.index("--input") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(request["target_smiles"])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "parent", "target_smiles": "CCCC", "family_hint": ""},
                preflight={"case_id": "parent_case"},
                budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "route_expansion": {
                    "child_targets": [
                        {
                            "schema_version": "route_expansion_child_target.v1",
                            "child_target_id": "resolved_anchor_1",
                            "name": "androstenedione",
                            "smiles": "CCCC",
                            "source": "resolved_advisory_anchor",
                            "policy": _policy_payload("resolved_anchor_policy"),
                        }
                    ],
                    "tasks": [
                        {
                            "schema_version": "compiled_route_expansion_task.v1",
                            "accepted": True,
                            "task_id": "compound_11_exact",
                            "frontier_smiles": "CCCC",
                            "exact_target_smiles": "CCO",
                            "exact_target_override": True,
                            "target_equivalence_audit_required": True,
                            "preferred_subgoals": ["androstenedione"],
                            "policy_id": "compound_11_policy",
                            "max_depth": 9,
                            "max_iterations": 12,
                            "expansion_topk": 33,
                        }
                    ],
                    "policy_payloads": [_policy_payload("compound_11_policy")],
                },
            }
            with patch("cascade_planner.harness.tools._chem_enzy_python_bin", return_value=Path("/usr/bin/python3")):
                with patch("cascade_planner.harness.tools.subprocess.run", side_effect=fake_run):
                    record = execute_local_tool("run_route_expansion_subgoal_search", {}, state)
            request = json.loads(
                (root / "route_expansion_subgoals" / "01_compound_11_exact_request.json").read_text(
                    encoding="utf-8"
                )
            )
            persisted = json.loads((root / "route_expansion_subgoal_search_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertEqual(request["target_smiles"], "CCO")
        self.assertNotEqual(request["target_smiles"], "CCCC")
        self.assertTrue(request["exact_target_override"])
        self.assertTrue(request["target_equivalence_audit_required"])
        self.assertEqual(request["requested_exact_target_smiles"], "CCO")
        self.assertEqual(persisted["subgoals"][0]["subgoal"]["source"], "route_expansion_exact_target_override")
        self.assertEqual(persisted["accepted_subgoal_count"], 1)

    def test_route_expansion_subgoal_failure_is_auditable_not_execution_stop(self):
        plan = _plan(
            "subgoal_failure_continues",
            strategy="hybrid",
            tools=[
                {"tool_name": "run_route_expansion_subgoal_search", "payload": {}},
                {"tool_name": "run_self_evo_replay_gate", "payload": {}},
                {"tool_name": "validate_artifact_bundle", "payload": {}},
                {"tool_name": "emit_final_verdict", "payload": {}},
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="subgoal_failure_continues",
                target_smiles="CCO",
                output_dir=tmp,
                planner_plan=plan,
                use_live_planner=False,
                mock_tool_results={
                    "run_route_expansion_subgoal_search": {
                        "schema_version": "route_expansion_subgoal_search_result.v1",
                        "accepted": False,
                        "status": "failed",
                        "solved": False,
                        "reasons": ["no_route_expansion_subgoal_verified_solved"],
                    }
                },
            )
            calls = [json.loads(line) for line in (Path(tmp) / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual([call["tool_name"] for call in calls], [
            "run_route_expansion_subgoal_search",
            "run_self_evo_replay_gate",
            "validate_artifact_bundle",
        ])
        self.assertIn("self_evo_replay", result["artifact_bundle"]["artifacts"])
        self.assertIn("no_route_expansion_subgoal_verified_solved", result["final_verdict"]["reasons"])

    def test_self_evo_replay_gate_tool_blocks_target_run_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "self_evo": {
                    "schema_version": "self_evo_staging_compile_report.v1",
                    "accepted": True,
                    "production_write_blocked": True,
                    "staging_candidate_count": 1,
                    "candidate_validation": [],
                    "kb": {
                        "schema_version": "evolution_layered_kb.v1",
                        "layers": {
                            "candidate": {},
                            "shadow": {},
                            "staging": {
                                "ethanol_template": {
                                    "schema_version": "evolution_candidate.v1",
                                    "candidate_id": "ethanol_template",
                                    "candidate_type": "TemplateCandidate",
                                    "payload": {"template_id": "ethanol_template"},
                                    "evidence_refs": ["ev1"],
                                    "validation_status": "draft",
                                    "source": "open_structure_research",
                                }
                            },
                            "production": {},
                        },
                        "history": [],
                    },
                },
            }
            result = execute_local_tool(
                "run_self_evo_replay_gate",
                {"target_run": True, "allow_production": True},
                state,
            )
            report = json.loads((Path(tmp) / "self_evo_replay_report.json").read_text(encoding="utf-8"))
            memory = json.loads((Path(tmp) / "self_evo_memory.json").read_text(encoding="utf-8"))

        self.assertTrue(result.output["result"]["accepted"])
        self.assertTrue(report["production_write_blocked"])
        self.assertEqual(report["production_promoted_count"], 0)
        self.assertIn("target_run_production_blocked", report["reasons"])
        self.assertIn("self_evo_memory", result.output["result"])
        self.assertTrue(memory["future_use_policy"]["requires_replay_gate_before_production"])

    def test_self_evo_replay_gate_unwraps_compiled_downstream_harness_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            open_dir = root / "open_structure_research"
            compiled = {
                "schema_version": "compiled_downstream_consumables.v1",
                "accepted": True,
                "literature_template_plugin": {
                    "template_cards": [
                        {
                            "schema_version": "literature_template_card.v1",
                            "template_id": "ethanol_template",
                            "template_level": "advisory_strategy",
                            "evidence_refs": ["ev1"],
                            "not_raw_reaction_injection": True,
                        }
                    ],
                    "one_step_rows": [],
                },
                "self_evo": {
                    "schema_version": "self_evo_staging_compile_report.v1",
                    "accepted": True,
                    "production_write_blocked": True,
                    "staging_candidate_count": 1,
                    "candidate_validation": [],
                    "kb": {
                        "schema_version": "evolution_layered_kb.v1",
                        "layers": {
                            "candidate": {},
                            "shadow": {},
                            "staging": {
                                "ethanol_template": {
                                    "schema_version": "evolution_candidate.v1",
                                    "candidate_id": "ethanol_template",
                                    "candidate_type": "TemplateCandidate",
                                    "payload": {"template_id": "ethanol_template"},
                                    "evidence_refs": ["ev1"],
                                    "validation_status": "draft",
                                    "source": "open_structure_research",
                                }
                            },
                            "production": {},
                        },
                        "history": [],
                    },
                },
            }
            wrapper = _write_compiled_downstream_wrapper(open_dir, compiled)
            state = ToolExecutionState(
                run_dir=root,
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["compiled_downstream"] = wrapper
            result = execute_local_tool("run_self_evo_replay_gate", {"target_run": True}, state)
            report = json.loads((root / "self_evo_replay_report.json").read_text(encoding="utf-8"))
            memory = json.loads((root / "self_evo_memory.json").read_text(encoding="utf-8"))

        self.assertEqual(result.status, "accepted")
        self.assertFalse(report.get("skipped", False))
        self.assertEqual(report["production_promoted_count"], 0)
        self.assertTrue(report["production_write_blocked"])
        self.assertEqual(memory["reusable_template_cards"][0]["template_id"], "ethanol_template")

    def test_verified_native_solution_skips_expensive_literature_and_evolution_steps(self):
        plan = _plan(
            "ethanol_verified",
            strategy="chem_enzy_first",
            tools=[
                {"tool_name": "run_chemenzy", "payload": {}},
                {"tool_name": "audit_route_and_extract_frontier", "payload": {}},
                {"tool_name": "run_smiles_first_literature_workflow", "payload": {}},
                {"tool_name": "run_open_structure_research_agent", "payload": {}},
                {"tool_name": "run_guided_chemenzy_rerun", "payload": {}},
                {"tool_name": "run_route_expansion_subgoal_search", "payload": {}},
                {"tool_name": "run_self_evo_replay_gate", "payload": {}},
                {"tool_name": "validate_artifact_bundle", "payload": {}},
                {"tool_name": "emit_final_verdict", "payload": {}},
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="ethanol_verified",
                target_smiles="CCO",
                output_dir=tmp,
                planner_plan=plan,
                use_live_planner=False,
                mock_tool_results={"run_chemenzy": _accepted_ethanol_chemenzy_result()},
            )
            calls = [json.loads(line) for line in (Path(tmp) / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()]
            artifacts = result["artifact_bundle"]["artifacts"]

        self.assertEqual(result["final_verdict"]["verdict"], "solved")
        self.assertTrue(result["final_verdict"]["solved"])
        self.assertTrue(artifacts["route_verifier"]["accepted"])
        self.assertEqual(artifacts["open_structure_research"]["status"], "skipped")
        self.assertEqual(artifacts["guided_chemenzy"]["status"], "skipped")
        self.assertEqual(artifacts["route_expansion_subgoal_search"]["status"], "skipped")
        self.assertEqual(artifacts["self_evo_replay"]["status"], "skipped")
        self.assertTrue(all(call["status"] == "accepted" for call in calls), calls)

    def test_guided_chemenzy_missing_policy_is_auditable_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            record = execute_local_tool("run_guided_chemenzy_rerun", {}, state)
            persisted = json.loads((Path(tmp) / "guided_chemenzy_result.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertEqual(state.guided_chemenzy_runs, 0)
        self.assertTrue(persisted["accepted"])
        self.assertEqual(persisted["status"], "skipped")
        self.assertIn("guided_policy_missing", persisted["reasons"])

    def test_native_solved_gate_rejects_bare_stock_audit_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"accepted": True, "case_id": "ethanol"},
            )
            audit_record = execute_local_tool(
                "audit_route_and_extract_frontier",
                {"stock_audit_passed": True},
                state,
            )
            guided_record = execute_local_tool("run_guided_chemenzy_rerun", {}, state)

        self.assertEqual(audit_record.output["audit"]["route_status"], "solved")
        self.assertEqual(guided_record.status, "accepted")
        self.assertIn("guided_policy_missing", guided_record.output["result"]["reasons"])
        self.assertNotIn("native_route_verified_solved", guided_record.output["result"]["reasons"])

    def test_native_solved_gate_rejects_cross_target_verifier(self):
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {"terminal_reactants": ["CC", "O"]},
                        "steps": [
                            {
                                "main_reactant": "CC",
                                "aux_reactants": ["O"],
                                "product": "CCO",
                                "stock_status": {"CC": True, "O": True},
                            }
                        ],
                    }
                ],
            },
            target_smiles="CCO",
        )
        self.assertTrue(verifier["accepted"], verifier["reasons"])

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethylamine", "target_smiles": "CCN", "family_hint": ""},
                preflight={"accepted": True, "case_id": "ethylamine"},
            )
            state.artifacts["route_verifier"] = verifier
            guided_record = execute_local_tool("run_guided_chemenzy_rerun", {}, state)

        self.assertEqual(guided_record.status, "accepted")
        self.assertIn("guided_policy_missing", guided_record.output["result"]["reasons"])
        self.assertNotIn("native_route_verified_solved", guided_record.output["result"]["reasons"])

    def test_self_evo_missing_staging_is_auditable_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            record = execute_local_tool("run_self_evo_replay_gate", {}, state)
            persisted = json.loads((Path(tmp) / "self_evo_replay_report.json").read_text(encoding="utf-8"))

        self.assertEqual(record.status, "accepted")
        self.assertTrue(persisted["accepted"])
        self.assertEqual(persisted["status"], "skipped")
        self.assertIn("self_evo_staging_missing", persisted["reasons"])
        self.assertTrue(persisted["production_write_blocked"])
        self.assertEqual(persisted["production_promoted_count"], 0)

    def test_bundle_validation_allows_skipped_open_research_without_compiled_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethanol", "target_smiles": "CCO", "family_hint": ""},
                preflight={"case_id": "ethanol"},
            )
            state.artifacts["open_structure_research"] = {
                "schema_version": "open_structure_research_result.v1",
                "accepted": True,
                "status": "skipped",
                "skipped": True,
                "reasons": ["native_route_verified_solved"],
            }
            record = execute_local_tool("validate_artifact_bundle", {}, state)

        self.assertEqual(record.status, "accepted")
        self.assertTrue(record.output["validation"]["accepted"])
        self.assertNotIn("open_research_missing_compiled_downstream", record.output["validation"]["reasons"])

    def test_chemenzy_runner_defaults_to_strict_building_block_stock_names(self):
        self.assertEqual(_stock_names_from_payload({}), ["PaRotes_n1-stock"])

    def test_stitched_literature_chain_with_verified_subgoal_can_claim_solved(self):
        literature_terminal = "CCO"
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "acetaldehyde", "target_smiles": "CC=O"},
                preflight={"accepted": True, "case_id": "stitched_case"},
            )
            literature_chain = {
                "schema_version": "source_detail_route_chain_audit.v1",
                "accepted": True,
                "case_id": "stitched_case",
                "target_smiles": "CC=O",
                "terminal_smiles": literature_terminal,
                "terminal_name": "ethanol",
                "terminal_reached": True,
                "step_count": 1,
                "source_ref": "doi:10.1000/revalidatable-stitch",
                "chain": [
                    _strict_literature_step(
                        step_id="ethanol_oxidation",
                        reactants=[literature_terminal],
                        product="CC=O",
                    )
                ],
            }
            subgoal = verify_chemenzy_raw_routes(
                _accepted_ethanol_chemenzy_result_for_target(literature_terminal),
                target_smiles=literature_terminal,
                case_id="stitched_case:ethanol",
            )
            literature_path = Path(tmp) / "trusted_literature_chain.json"
            verifier_path = Path(tmp) / "subgoal_verifier.json"
            raw_path = Path(tmp) / "subgoal_raw.json"
            literature_path.write_text(json.dumps(literature_chain), encoding="utf-8")
            verifier_path.write_text(json.dumps(subgoal), encoding="utf-8")
            raw_path.write_text(
                json.dumps(_accepted_ethanol_chemenzy_result_for_target(literature_terminal)),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(_TRUSTED_REGISTRY_FIXTURE)},
            ):
                record = execute_local_tool(
                    "stitch_literature_chain_with_subgoal_route",
                    {
                        "literature_chain_audit_path": str(literature_path),
                        "subgoal_verifier_path": str(verifier_path),
                        "subgoal_raw_result_path": str(raw_path),
                    },
                    state,
                )
                verdict = emit_final_verdict(
                    {
                        "case_id": "stitched_case",
                        "target_input": state.target_input,
                        "preflight": state.preflight,
                        "workflow_plan": _plan(
                            "stitched_case",
                            strategy="hybrid",
                            tools=[
                                {"tool_name": "stitch_literature_chain_with_subgoal_route", "payload": {}},
                                {"tool_name": "emit_final_verdict", "payload": {}},
                            ],
                        ),
                        "artifacts": state.artifacts,
                        "validations": [],
                    }
                )

        self.assertEqual(record.status, "accepted")
        self.assertTrue(record.output["result"]["accepted"])
        self.assertEqual(verdict.verdict, "solved")
        self.assertTrue(verdict.solved)
        self.assertTrue(verdict.stock_audit_passed)

    def test_stitched_route_rejects_solved_subgoal_when_terminal_identity_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "ethyl acetate", "target_smiles": "CCOC(C)=O"},
                preflight={"accepted": True, "case_id": "stitched_mismatch_case"},
            )
            literature_chain = {
                "schema_version": "source_detail_route_chain_audit.v1",
                "accepted": True,
                "case_id": "stitched_mismatch_case",
                "target_smiles": "CCOC(C)=O",
                "terminal_smiles": "CCO",
                "terminal_name": "ethanol",
                "terminal_reached": True,
                "step_count": 1,
                "source_ref": "doi:10.1000/revalidatable-stitch",
                "chain": [
                    _strict_literature_step(
                        step_id="ethyl_acetate",
                        reactants=["CCO"],
                        product="CCOC(C)=O",
                    )
                ],
            }
            subgoal = verify_chemenzy_raw_routes(
                _accepted_ethanol_chemenzy_result_for_target("CCN"),
                target_smiles="CCN",
                case_id="stitched_mismatch_case:ethylamine",
            )
            with patch.dict(
                os.environ,
                {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(_TRUSTED_REGISTRY_FIXTURE)},
            ):
                record = execute_local_tool(
                    "stitch_literature_chain_with_subgoal_route",
                    {
                        "literature_chain_audit": literature_chain,
                        "subgoal_verifier": subgoal,
                        "subgoal_raw_result": _accepted_ethanol_chemenzy_result_for_target("CCN"),
                    },
                    state,
                )
                verdict = emit_final_verdict(
                    {
                        "case_id": "stitched_mismatch_case",
                        "target_input": state.target_input,
                        "preflight": state.preflight,
                        "workflow_plan": _plan(
                            "stitched_mismatch_case",
                            strategy="hybrid",
                            tools=[
                                {"tool_name": "stitch_literature_chain_with_subgoal_route", "payload": {}},
                                {"tool_name": "emit_final_verdict", "payload": {}},
                            ],
                        ),
                        "artifacts": state.artifacts,
                        "validations": [],
                    }
                )

        self.assertEqual(record.status, "rejected")
        self.assertIn("subgoal_target_not_verified", record.reasons)
        self.assertIn("subgoal_verifier_not_accepted", record.reasons)
        self.assertNotEqual(verdict.verdict, "solved")


@unittest.skipUnless(
    os.environ.get("AUTOPLANNER_LIVE_CODEX_ENTRY_SMOKE") == "1",
    "set AUTOPLANNER_LIVE_CODEX_ENTRY_SMOKE=1 for live Codex-entry smokes",
)
class CodexEntryHarnessLiveSmokeTest(unittest.TestCase):
    def test_bufotalin_live_harness_not_solved(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="bufotalin_live_smoke",
                target_smiles=BUFOTALIN_SMILES,
                family_hint="bufotalin, bufadienolide, steroid",
                output_dir=tmp,
                timeout_s=1800.0,
            )

        self.assertEqual(result["final_verdict"]["verdict"], "partial_anchor_only_not_solved")
        self.assertFalse(result["final_verdict"]["solved"])

    def test_o_glycoside_live_harness_requires_audit_for_solved(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_codex_entry_controller(
                target_name="o_glycoside_live_smoke",
                target_smiles=O_GLYCOSIDE_SMILES,
                family_hint="O-glycoside, glycoside, sugar, glycosylation",
                output_dir=tmp,
                timeout_s=1800.0,
            )

        if result["final_verdict"]["verdict"] == "solved":
            self.assertTrue(result["final_verdict"]["stock_audit_passed"])
        else:
            self.assertIn(
                result["final_verdict"]["verdict"],
                {"partial_anchor_only_not_solved", "unresolved", "needs_followup", "fake_closed_rejected"},
            )
        self.assertTrue(
            result["artifact_bundle"]["artifacts"].get("smiles_first")
            or result["artifact_bundle"]["artifacts"].get("open_structure_research")
        )


def _plan(
    case_id: str,
    *,
    strategy: str,
    tools: list[dict],
    planner_decision_reason: str = "",
    run_semantics: str = "canonical_agent_controller",
) -> dict:
    return {
        "schema_version": WORKFLOW_PLAN_SCHEMA,
        "case_id": case_id,
        "recommended_strategy": strategy,
        "planned_tools": tools,
        "rationale": "test plan",
        "risk_flags": [],
        "expected_verdict_floor": "needs_audit",
        "planner_decision_reason": planner_decision_reason,
        "run_semantics": run_semantics,
    }


def _accepted_ethanol_chemenzy_result() -> dict:
    return _accepted_ethanol_chemenzy_result_for_target("CCO")


def _common_element_inventory_reactants(target_smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        raise AssertionError(f"invalid test target: {target_smiles}")
    counts: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number != 1:
            counts[atomic_number] = counts.get(atomic_number, 0) + 1
    unsupported = set(counts) - {6, 7, 8, 17, 35}
    if unsupported:
        raise AssertionError(f"unsupported common-inventory elements: {sorted(unsupported)}")
    carbon_count = counts.get(6, 0)
    reactants = ["CC"] * (carbon_count // 2)
    if carbon_count % 2:
        reactants.append("C")
    reactants.extend(["N"] * counts.get(7, 0))
    reactants.extend(["O"] * counts.get(8, 0))
    reactants.extend(["Cl"] * counts.get(17, 0))
    reactants.extend(["Br"] * counts.get(35, 0))
    return reactants


def _accepted_ethanol_chemenzy_result_for_target(target_smiles: str) -> dict:
    reactants = _common_element_inventory_reactants(target_smiles)
    terminal_reactants = list(dict.fromkeys(reactants))
    step = {
        "index": 0,
        "product": target_smiles,
        "reactant_smiles": reactants,
        "stock_status": {item: True for item in terminal_reactants},
        "atom_mapped_reaction_smiles": (
            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
            if target_smiles == "CCO"
            else ""
        ),
    }
    if target_smiles == "CCO" and reactants == ["CC", "O"]:
        step = {
            **_strict_literature_step(
                step_id="ethanol_hydration",
                reactants=reactants,
                product=target_smiles,
            ),
            "index": 0,
            "stock_status": {item: True for item in terminal_reactants},
        }
    return {
        "schema_version": "chemenzy_web_result.v1",
        "ok": True,
        "accepted": True,
        "search_status": {"solved": True},
        "target": target_smiles,
        "routes": [
            {
                "route_rank": 0,
                "score": 0.99,
                "n_steps": 1,
                "stock_closed": True,
                "metrics": {
                    "route_solved": True,
                    "strict_stock_solve": True,
                    "stock_closed": True,
                    "terminal_reactants": terminal_reactants,
                    "terminal_stock_status": {item: True for item in terminal_reactants},
                },
                "steps": [step],
            }
        ],
    }


def _strict_literature_step(*, step_id: str, reactants: list[str], product: str) -> dict:
    pdf_digest = hashlib.sha256(_SOURCE_FIXTURE.read_bytes()).hexdigest()
    image_digest = hashlib.sha256(_SOURCE_PAGE_FIXTURE.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(_SOURCE_MANIFEST_FIXTURE.read_bytes()).hexdigest()
    template_id = f"source_detail_exact_step:{step_id}"
    row = {
        "step_id": step_id,
        "source_template_id": template_id,
        "product_smiles": product,
        "reactant_smiles": reactants,
        "main_reactant_smiles": reactants[0],
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "evidence_refs": [f"{_SOURCE_MANIFEST_FIXTURE}::page:1"],
        "relation_type": "exact",
        "source_detail_exact_step": True,
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": template_id,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "document_id": "fixture:revalidatable-stitch",
                "manifest_path": str(_SOURCE_MANIFEST_FIXTURE.resolve()),
                "manifest_sha256": manifest_digest,
                "source_pdf_path": str(_SOURCE_FIXTURE.resolve()),
                "source_pdf_sha256": pdf_digest,
                "page_number": 1,
                "image_path": str(_SOURCE_PAGE_FIXTURE.resolve()),
                "image_sha256": image_digest,
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }
    if reactants == ["CCO"] and product == "CC=O":
        row["atom_mapped_reaction_smiles"] = (
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        )
    elif reactants == ["CC", "O"] and product == "CCO":
        row["atom_mapped_reaction_smiles"] = (
            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
        )
    return row


def _policy_payload(policy_id: str) -> dict:
    return {
        "schema_version": "chem_enzy_search_policy.v1",
        "policy_id": policy_id,
        "operator_id": f"{policy_id}_operator",
        "case_id": "ethanol",
        "evidence_refs": ["ev1"],
        "terminal_blacklist": [],
        "anchor_whitelist": [],
        "preferred_subgoal": {"preferred_subgoals": ["ethanol precursor"]},
        "source_budget": {"preferred_reaction_classes": ["literature_guided"]},
        "rerun_reason": "unit_test",
        "budget": {
            "max_reruns": 1,
            "max_iterations": 12,
            "max_depth": 9,
            "expansion_topk": 33,
        },
        "mode": "guided",
        "compiler_metadata": {},
    }


def _plugin_row(reactants: str) -> dict:
    return {
        "reactants": reactants,
        "scores": 0.7,
        "costs": None,
        "template": {
            "source": "literature_template_plugin",
            "source_model": "literature_template_plugin",
            "requires_audit": True,
            "not_lab_procedure": True,
            "no_solved_claim": True,
            "template_validation_report": {
                "allowed_for_one_step_source": True,
                "accepted": True,
                "reasons": [],
            },
            "template_applicability_report": {
                "target_smiles": "CCO",
                "frontier_smiles": "CCO",
            },
        },
        "templates": {},
        "model_full_name": "autoplanner.literature_template_plugin",
        "weight": 1.0,
    }


def _plugin_row_with_trace(
    *,
    product: str,
    reactants: list[str],
    template_id: str,
    reagent: str = "",
) -> dict:
    trace = {
        "schema_version": "literature_template_trace.v1",
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "source_ref": "doi:test",
        "evidence_refs": ["ev1"],
        "product_smiles": product,
        "frontier_smiles": product,
        "reactant_smiles": reactants,
        "condition_candidate": {
            "schema_version": "condition_candidate.v1",
            "condition_status": "evidence_backed",
            "reagent": reagent,
        },
        "no_solved_claim": True,
        "requires_audit": True,
    }
    template = {
        "source": "literature_template_plugin",
        "source_model": "literature_template_plugin",
        "requires_audit": True,
        "not_lab_procedure": True,
        "no_solved_claim": True,
        "literature_template_trace": trace,
        "template_validation_report": {
            "allowed_for_one_step_source": True,
            "accepted": True,
            "reasons": [],
        },
        "template_applicability_report": {
            "target_smiles": product,
            "frontier_smiles": product,
        },
    }
    return {
        "reactants": ".".join(reactants),
        "scores": 0.7,
        "costs": None,
        "template": template,
        "templates": template,
        "model_full_name": "autoplanner.literature_template_plugin",
        "weight": 1.0,
        "literature_template_trace": trace,
    }


def _write_compiled_downstream_wrapper(open_dir: Path, compiled: dict) -> dict:
    open_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = open_dir / "compiled_downstream_consumables.json"
    compiled_path.write_text(json.dumps(compiled), encoding="utf-8")
    return {
        "schema_version": "compiled_downstream_harness_result.v1",
        "accepted": bool(compiled.get("accepted", True)),
        "source": "curator_augmented_downstream",
        "reasons": [],
        "artifact_refs": {
            "compiled_downstream_consumables": str(compiled_path),
        },
        "summary": {
            "guided_policy_count": len(((compiled.get("guided_chemenzy") or {}).get("policy_payloads") or [])),
            "route_expansion_task_count": len(((compiled.get("route_expansion") or {}).get("tasks") or [])),
            "template_card_count": len(((compiled.get("literature_template_plugin") or {}).get("template_cards") or [])),
            "one_step_row_count": len(((compiled.get("literature_template_plugin") or {}).get("one_step_rows") or [])),
            "self_evo_staging_candidate_count": int(((compiled.get("self_evo") or {}).get("staging_candidate_count") or 0),
            ),
        },
    }


def _write_downstream_consumables(open_dir: Path, *, case_id: str) -> None:
    (open_dir / "downstream_consumables.json").write_text(
        json.dumps(
            {
                "schema_version": "open_downstream_consumables.v1",
                "case_id": case_id,
                "planner_handoff": {
                    "next_action": "no_consumable_found",
                    "solved": False,
                    "production_kb_promotion": False,
                    "reason": "unit test placeholder",
                },
                "guided_rerun_requests": [],
                "literature_template_cards": [],
                "literature_route_segments": [],
                "executable_template_candidates": [],
                "route_expansion_tasks": [],
                "evolution_candidates": [],
                "rejected_consumables": [],
            }
        ),
        encoding="utf-8",
    )


def _write_compilable_downstream_consumables(open_dir: Path, *, case_id: str) -> None:
    (open_dir / "downstream_consumables.json").write_text(
        json.dumps(
            {
                "schema_version": "open_downstream_consumables.v1",
                "case_id": case_id,
                "planner_handoff": {
                    "next_action": "guided_chemenzy_rerun",
                    "solved": False,
                    "production_kb_promotion": False,
                    "reason": "unit test guided handoff",
                },
                "guided_rerun_requests": [
                    {
                        "request_id": f"{case_id}_guided_1",
                        "request_type": "literature_guided_chemenzy_rerun",
                        "target": "stuck_node",
                        "evidence_refs": ["ev1"],
                        "preferred_subgoals": ["validated intermediate"],
                        "preferred_reaction_classes": ["literature_guided"],
                        "max_depth": 15,
                        "max_iterations": 50,
                        "expansion_topk": 100,
                    }
                ],
                "literature_template_cards": [
                    {
                        "schema_version": "literature_template_card.v1",
                        "template_id": f"{case_id}_advisory_template",
                        "validation_status": "draft",
                        "template_level": "advisory_strategy",
                        "reaction_class": "literature_guided",
                        "product_retron": {"retron_type": "unit_test_retron"},
                        "evidence_refs": ["ev1"],
                        "not_raw_reaction_injection": True,
                    }
                ],
                "literature_route_segments": [],
                "executable_template_candidates": [],
                "route_expansion_tasks": [
                    {
                        "task_id": f"{case_id}_route_expansion_1",
                        "task_type": "stuck_node_rerun",
                        "frontier_smiles": "CCO",
                        "target": "ethanol stuck node",
                        "preferred_subgoals": ["simple ethanol precursor"],
                        "preferred_reaction_classes": ["unit_test_expansion"],
                        "terminal_blacklist": ["CCO"],
                        "anchor_whitelist": ["CC"],
                        "evidence_refs": ["ev1"],
                        "max_depth": 12,
                        "max_iterations": 24,
                        "expansion_topk": 48,
                    }
                ],
                "evolution_candidates": [
                    {
                        "candidate_id": f"{case_id}_template_candidate",
                        "candidate_type": "TemplateCandidate",
                        "validation_status": "draft",
                        "target_layer": "candidate",
                        "evidence_refs": ["ev1"],
                        "payload": {"template_id": f"{case_id}_advisory_template"},
                    }
                ],
                "rejected_consumables": [],
            }
        ),
        encoding="utf-8",
    )


def _write_valid_open_research_shell(open_dir: Path, *, case_id: str) -> None:
    (open_dir / "evidence").mkdir(parents=True, exist_ok=True)
    event_log = open_dir / "codex_events.jsonl"
    event_log.write_text(
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}) + "\n",
        encoding="utf-8",
    )
    (open_dir / "open_agent_run_record.json").write_text(
        json.dumps(
            {
                "schema_version": "open_codex_structure_template_run.v1",
                "run_dir": str(open_dir),
                "exit_code": 0,
                "metadata": {
                    "stream_jsonl": True,
                    "event_log_path": str(event_log),
                },
            }
        ),
        encoding="utf-8",
    )
    (open_dir / "structure_template_report.md").write_text("checkpoint\n", encoding="utf-8")
    (open_dir / "validated_compounds.smi").write_text("CCO cmp_ethanol\n", encoding="utf-8")
    (open_dir / "structure_template_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": "open_structure_template_candidates.v1",
                "case_id": case_id,
                "target": {},
                "candidate_generation_policy": {},
                "candidates": [],
                "rejected_items": [],
                "source_refs": [],
                "audit_summary": {"solved": False, "production_kb_promotion": False},
            }
        ),
        encoding="utf-8",
    )
    (open_dir / "evidence" / "literature_sources.json").write_text(
        json.dumps(
            {
                "schema_version": "open_literature_sources.v1",
                "case_id": case_id,
                "source_relation_policy": {},
                "sources": [],
                "excluded_sources": [],
                "search_log": [],
            }
        ),
        encoding="utf-8",
    )
    (open_dir / "evidence" / "pubchem_validated_compounds.json").write_text(
        json.dumps(
            {
                "schema_version": "open_pubchem_validated_compounds.v1",
                "case_id": case_id,
                "compound_source_policy": {},
                "compounds": [],
                "rejected_items": [],
            }
        ),
        encoding="utf-8",
    )
    (open_dir / "open_agent_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "open_structure_agent_audit.v1",
                "case_id": case_id,
                "final_status": "partial_anchor",
                "solved": False,
                "production_kb_promotion": False,
                "checks": [],
                "limitations": [],
                "next_actions": [],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
