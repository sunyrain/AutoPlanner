import tempfile
import unittest
from pathlib import Path

from cascade_planner.harness.agent_action_planner import plan_action_batch, validate_action_batch
from cascade_planner.harness.agentic_blackboard import (
    initialize_agent_blackboard,
    update_blackboard_from_action,
)
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.process_evidence import (
    _process_type,
    process_evidence_rows_from_pdf_result,
    process_evidence_rows_from_visual_result,
)
from cascade_planner.harness.schemas import TargetInput
from cascade_planner.harness.tools import (
    ToolExecutionState,
    _pdf_evidence_from_payload_or_artifacts,
    _validate_local_pdf_source_binding,
    execute_local_tool,
)
from cascade_planner.harness.visual_literature_chain_agent import _candidate_chain_from_parsed
from cascade_planner.harness.visual_structure_extraction import validate_visual_structure_chain


C22_9OH_4HP_SMILES = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
ATORVASTATIN_SMILES = (
    "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)"
    "C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
)


class VisualRouteStepAliasTest(unittest.TestCase):
    def test_visual_route_steps_alias_is_preserved_as_standard_steps(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "visual_alias_case",
            "source_ref": "doi:10.0000/visual",
            "source_title": "Scheme image source",
            "evidence_refs": ["current_image:page_1"],
            "route_steps": [
                {
                    "product_label": "7",
                    "product_smiles": "CCO",
                    "precursor_labels": ["6"],
                    "visible_conditions": {
                        "reagent": "NCS",
                        "solvent": "MeCN",
                        "source_text": "Scheme text reports NCS in MeCN.",
                    },
                    "stereochemistry_status": "unspecified_or_partial",
                    "not_exact_literature_segment": True,
                    "allowed_use": "exploratory_template_and_guided_hint_only",
                    "risk_flags": ["stereochemistry_unspecified"],
                },
                {
                    "product_label": "6",
                    "product_smiles": "CC",
                    "precursor_labels": ["5"],
                    "precursor_smiles": ["C"],
                    "visible_conditions": {
                        "reagent": "base",
                        "source_text": "Scheme text reports a base-mediated precursor step.",
                    },
                    "stereochemistry_status": "unspecified_or_partial",
                    "not_exact_literature_segment": True,
                    "allowed_use": "exploratory_template_and_guided_hint_only",
                }
            ],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="visual target",
            target_smiles="",
            source_ref="doi:10.0000/visual",
            source_title="Scheme image source",
            image_paths=[],
        )

        self.assertEqual(len(chain["steps"]), 2)
        step = chain["steps"][0]
        self.assertEqual(step["product_label"], "7")
        self.assertEqual(step["reactant_labels"], ["6"])
        self.assertEqual(step["reactant_smiles"], ["CC"])
        self.assertEqual(step["condition_candidate"]["reagent"], "NCS")
        self.assertIn("Scheme text reports", step["source_excerpt"])
        self.assertTrue(step["not_exact_literature_segment"])
        validation = validate_visual_structure_chain(chain, require_contiguous=False)
        self.assertEqual(validation["summary"]["step_count"], 2)
        self.assertEqual(validation["summary"]["accepted_step_count"], 2)


def _process_visual_result():
    return {
        "schema_version": "visual_literature_chain_extraction_result.v1",
        "accepted": False,
        "status": "completed",
        "source_ref": "doi:10.1186/s12934-021-01717-w",
        "source_title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols in Mycobacterium neoaurum",
        "expected_labels": [
            "9-OH-4-HP",
            "phytosterols",
            "Delta kstD Delta hsd4A Delta fadA5",
        ],
        "candidate_chain": {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source_ref": "doi:10.1186/s12934-021-01717-w",
            "source_title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols in Mycobacterium neoaurum",
            "evidence_refs": [
                "current_image:page_4_fig2_peakB_structure_and_caption",
                "current_image:page_5_fig3_table1_text",
                "current_image:page_6_fig4_table2_text",
            ],
            "source_locator": "PDF pages 4-6; Fig. 2d peak B, Fig. 3, Table 1, Table 2",
            "extraction_gaps": [
                {
                    "label": "phytosterols",
                    "gap_type": "structure_gap",
                    "reason": "feedstock is reported as phytosterols and no single defined molecular structure is visible",
                    "source_locator": "Fig. 3 caption and Table 2",
                },
                {
                    "label": "Delta kstD Delta hsd4A Delta fadA5",
                    "gap_type": "structure_gap",
                    "reason": "strain is a biological catalyst/producer, not a molecular reactant",
                    "source_locator": "Fig. 3 legend and Table 2",
                },
                {
                    "label": "9-OH-4-HP from phytosterols",
                    "gap_type": "structure_gap",
                    "reason": "product structure and production conditions are visible, but the substrate is a feedstock mixture",
                    "source_locator": "Fig. 2d peak B, Fig. 3, Table 1, Table 2",
                },
            ],
            "steps": [],
        },
        "candidate_step_count": 0,
        "reasons": [
            "visual_literature_chain_has_no_steps",
            "visual_literature_chain_missing_expected_labels",
        ],
    }


def _process_payload():
    return {
        "source_ref": "doi:10.1186/s12934-021-01717-w",
        "source_title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols in Mycobacterium neoaurum",
        "expected_labels": [
            "9-OH-4-HP",
            "phytosterols",
            "Delta kstD Delta hsd4A Delta fadA5",
        ],
        "route_sequence_hint": "Extract explicit product labels, feedstock, strain, yield/purity rows; do not invent reaction SMILES.",
    }


class ProcessEvidenceTests(unittest.TestCase):
    def test_strained_reagent_language_is_not_misclassified_as_whole_cell(self):
        self.assertEqual(
            _process_type(
                "A total synthesis of paclitaxel uses a strained beta-lactam for side-chain coupling."
            ),
            "small_molecule_process_route",
        )
        self.assertEqual(
            _process_type(
                "The engineered Mycobacterium strain culture performs whole-cell biotransformation."
            ),
            "whole_cell_biotransformation",
        )

    def test_process_gap_visual_result_becomes_process_evidence_not_reaction_row(self):
        rows = process_evidence_rows_from_visual_result(
            _process_visual_result(),
            payload=_process_payload(),
            artifact_ref="/tmp/visual_literature_chain_extraction_result.json",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["schema_version"], "literature_process_evidence_row.v1")
        self.assertEqual(row["process_type"], "whole_cell_biotransformation")
        self.assertIn("9-OH-4-HP", row["endpoint_labels"])
        self.assertIn("phytosterols", row["substrate_or_feedstock_labels"])
        self.assertTrue(row["not_reaction_smiles"])
        self.assertTrue(row["not_parent_route_proof"])
        self.assertTrue(row["no_solved_claim"])
        self.assertNotIn("reaction_smiles", row)

    def test_pdf_fulltext_becomes_process_evidence_when_visual_api_is_unavailable(self):
        text = (
            "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols "
            "in Mycobacterium neoaurum. 9-OH-4-HP production reached 3.58 g L-1 "
            "from 5 g L-1 phytosterols, purity improved to 97%, and the final strain "
            "showed molar yield of 85.5%. Delta kstD Delta hsd4A Delta fadA5-NK was used."
        )
        with tempfile.TemporaryDirectory() as tmp:
            fulltext = Path(tmp) / "fulltext.txt"
            fulltext.write_text(text, encoding="utf-8")
            rows = process_evidence_rows_from_pdf_result(
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": True,
                    "source_ref": "doi:10.1186/s12934-021-01717-w",
                    "source_title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols in Mycobacterium neoaurum",
                    "fulltext_path": str(fulltext),
                    "source_pdf_path": str(Path(tmp) / "source.pdf"),
                },
                payload={"source_ref": "doi:10.1186/s12934-021-01717-w"},
                artifact_ref=str(Path(tmp) / "literature_pdf_structure_evidence.json"),
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["process_type"], "whole_cell_biotransformation")
        self.assertIn("9-OH-4-HP", row["endpoint_labels"])
        self.assertIn("phytosterols", row["substrate_or_feedstock_labels"])
        self.assertIn("Mycobacterium neoaurum", row["biocatalyst_or_process_labels"])
        self.assertEqual(row["quantitative_evidence"]["product_titer_g_per_l"], 3.58)
        self.assertEqual(row["quantitative_evidence"]["phytosterol_loading_g_per_l"], 5.0)
        self.assertEqual(row["quantitative_evidence"]["product_purity_percent"], 97.0)
        self.assertEqual(row["quantitative_evidence"]["molar_yield_percent"], 85.5)
        self.assertTrue(row["not_reaction_smiles"])

    def test_atorvastatin_bmc_process_pdf_becomes_advisory_route_anchor(self):
        text = (
            "An improved kilogram-scale preparation of atorvastatin calcium. "
            "The conversion of an advanced ketal ester intermediate, available in "
            "kilogram quantities via a published Paal-Knorr synthesis, to the "
            "commercial 3-hydroxy-3-methylglutaryl coenzyme A reductase inhibitor "
            "atorvastatin calcium is described. "
            "The prominent route uses 1,4-diketone 2 and protected side-chain amine 3 "
            "to give advanced ketal ester intermediate 4. Intermediate 4 is converted "
            "into diol 5 by ketal deprotection. Ester hydrolysis gives an atorvastatin "
            "sodium solution, followed by sodium-to-calcium counter-ion exchange with "
            "calcium acetate and ethyl acetate extraction to isolate atorvastatin "
            "hemi-calcium on 7 kg scale."
        )
        with tempfile.TemporaryDirectory() as tmp:
            fulltext = Path(tmp) / "fulltext.txt"
            fulltext.write_text(text, encoding="utf-8")
            rows = process_evidence_rows_from_pdf_result(
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": True,
                    "source_ref": "doi:10.1186/s13065-015-0082-7",
                    "source_title": "An improved kilogram-scale preparation of atorvastatin calcium",
                    "fulltext_path": str(fulltext),
                    "source_pdf_path": str(Path(tmp) / "atorvastatin_bmc.pdf"),
                },
                payload={
                    "source_ref": "doi:10.1186/s13065-015-0082-7",
                    "source_title": "An improved kilogram-scale preparation of atorvastatin calcium",
                },
                artifact_ref=str(Path(tmp) / "literature_pdf_structure_evidence.json"),
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["process_type"], "small_molecule_process_route")
        self.assertIn("atorvastatin calcium", row["endpoint_labels"])
        self.assertIn("advanced ketal ester intermediate 4", row["substrate_or_feedstock_labels"])
        self.assertIn("Paal-Knorr pyrrole construction", row["biocatalyst_or_process_labels"])
        self.assertIn("ketal deprotection", row["biocatalyst_or_process_labels"])
        self.assertIn("ester hydrolysis", row["biocatalyst_or_process_labels"])
        self.assertIn("calcium salt formation", row["biocatalyst_or_process_labels"])
        self.assertTrue(row["not_reaction_smiles"])
        self.assertTrue(row["not_parent_route_proof"])
        self.assertTrue(row["no_solved_claim"])

    def test_pdf_quantitative_evidence_prefers_final_optimized_process_metrics(self):
        text = (
            "The wild-type Mycobacterium neoaurum DSM 44074 produces 9-OH-4-HP with a "
            "molar yield of 4.8%. The purity of 9-OH-4-HP obtained from the Hsd4A and "
            "FadA5 deficient strain has reached 94.9%. Ultimately, 9-OH-4-HP production "
            "reached 3.58 g L-1 from 5 g L-1 phytosterols, and the purity of 9-OH-4-HP "
            "improved to 97%. The final 9-OH-4-HP production strain showed the best "
            "molar yield of 85.5%, compared with the previous reported strain."
        )
        with tempfile.TemporaryDirectory() as tmp:
            fulltext = Path(tmp) / "fulltext.txt"
            fulltext.write_text(text, encoding="utf-8")
            rows = process_evidence_rows_from_pdf_result(
                {
                    "accepted": True,
                    "source_ref": "doi:10.1186/s12934-021-01717-w",
                    "source_title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols in Mycobacterium neoaurum",
                    "fulltext_path": str(fulltext),
                },
                payload={"source_ref": "doi:10.1186/s12934-021-01717-w"},
            )

        quantitative = rows[0]["quantitative_evidence"]
        self.assertEqual(quantitative["product_titer_g_per_l"], 3.58)
        self.assertEqual(quantitative["phytosterol_loading_g_per_l"], 5.0)
        self.assertEqual(quantitative["product_purity_percent"], 97.0)
        self.assertEqual(quantitative["molar_yield_percent"], 85.5)
        self.assertIn("candidate_metrics", quantitative)

    def test_visual_tool_attaches_process_evidence_rows_for_mocked_process_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "9-OH-4-HP",
                    "target_smiles": C22_9OH_4HP_SMILES,
                    "family_hint": "steroid biotransformation",
                },
                preflight={"case_id": "target1", "accepted": True, "target_profile": {"heavy_atoms": 25}},
                mock_tool_results={"extract_visual_literature_chain": _process_visual_result()},
            )
            record = execute_local_tool("extract_visual_literature_chain", _process_payload(), state)

        result = record.output["result"]
        self.assertEqual(record.status, "rejected")
        self.assertEqual(result["process_evidence_row_count"], 1)
        self.assertEqual(state.artifacts["literature_process_evidence_rows"][0]["process_type"], "whole_cell_biotransformation")

    def test_pdf_tool_attaches_process_evidence_rows_for_fulltext(self):
        with tempfile.TemporaryDirectory() as tmp:
            fulltext = Path(tmp) / "fulltext.txt"
            fulltext.write_text(
                "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols "
                "in Mycobacterium neoaurum. 9-OH-4-HP production reached 3.58 g L-1 from 5 g L-1 phytosterols.",
                encoding="utf-8",
            )
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "9-OH-4-HP",
                    "target_smiles": C22_9OH_4HP_SMILES,
                    "family_hint": "steroid biotransformation",
                },
                preflight={"case_id": "target1", "accepted": True, "target_profile": {"heavy_atoms": 25}},
                mock_tool_results={
                    "extract_pdf_literature_structures": {
                        "schema_version": "literature_pdf_structure_evidence.v1",
                        "accepted": True,
                        "source_ref": "doi:10.1186/s12934-021-01717-w",
                        "source_title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols in Mycobacterium neoaurum",
                        "fulltext_path": str(fulltext),
                        "rendered_pages": [],
                        "indexed_images": [],
                        "scheme_crops": [],
                        "compound_text_snippets": [],
                        "summary": {},
                    }
                },
            )
            record = execute_local_tool(
                "extract_pdf_literature_structures",
                {"source_ref": "doi:10.1186/s12934-021-01717-w"},
                state,
            )

        result = record.output["result"]
        self.assertEqual(record.status, "accepted")
        self.assertEqual(result["process_evidence_row_count"], 1)
        self.assertEqual(state.artifacts["literature_process_evidence_rows"][0]["process_type"], "whole_cell_biotransformation")

    def test_pdf_tool_rejects_local_pdf_source_ref_mismatch_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "article_pdf.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% fake content should not be parsed\n")
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "9-OH-4-HP",
                    "target_smiles": C22_9OH_4HP_SMILES,
                    "family_hint": "steroid biotransformation",
                    "local_literature_cache": [
                        {
                            "candidate_id": "provided_pdf_1",
                            "source_ref": "doi:10.1186/s12934-021-01717-w",
                            "doi": "10.1186/s12934-021-01717-w",
                            "local_pdf": str(pdf),
                            "source_role": "local_cache",
                        }
                    ],
                },
                preflight={"case_id": "target1", "accepted": True, "target_profile": {"heavy_atoms": 25}},
            )
            record = execute_local_tool(
                "extract_pdf_literature_structures",
                {
                    "pdf_path": str(pdf),
                    "source_ref": "patent:US4397947A",
                    "source_title": "Microbial process for 9alpha-hydroxylation of steroids",
                },
                state,
            )

        result = record.output["result"]
        self.assertEqual(record.status, "rejected")
        self.assertIn("local_pdf_source_ref_mismatch", record.reasons)
        self.assertEqual(result["local_pdf_binding"]["cache_source_refs"], ["doi:10.1186/s12934-021-01717-w"])
        self.assertNotIn("literature_process_evidence_rows", result)
        self.assertNotIn("literature_process_evidence_rows", state.artifacts)

    def test_visual_tool_rejects_local_pdf_source_ref_mismatch_before_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "article_pdf.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% fake content should not be parsed\n")
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "9-OH-4-HP",
                    "target_smiles": C22_9OH_4HP_SMILES,
                    "family_hint": "steroid biotransformation",
                    "local_literature_cache": [
                        {
                            "candidate_id": "provided_pdf_1",
                            "source_ref": "doi:10.1186/s12934-021-01717-w",
                            "doi": "10.1186/s12934-021-01717-w",
                            "local_pdf": str(pdf),
                            "source_role": "local_cache",
                        }
                    ],
                },
                preflight={"case_id": "target1", "accepted": True, "target_profile": {"heavy_atoms": 25}},
            )
            record = execute_local_tool(
                "extract_visual_literature_chain",
                {
                    "pdf_path": str(pdf),
                    "source_ref": "patent:US4397947A",
                    "source_title": "Microbial process for 9alpha-hydroxylation of steroids",
                },
                state,
            )

        result = record.output["result"]
        self.assertEqual(record.status, "rejected")
        self.assertIn("local_pdf_source_ref_mismatch", record.reasons)
        self.assertEqual(result["image_paths"], [])
        self.assertEqual(result["candidate_step_count"], 0)
        self.assertNotIn("literature_process_evidence_rows", state.artifacts)

    def test_pdf_evidence_lookup_does_not_fallback_to_latest_when_source_ref_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "9-OH-4-HP",
                    "target_smiles": C22_9OH_4HP_SMILES,
                    "family_hint": "steroid biotransformation",
                },
                preflight={"case_id": "target1", "accepted": True, "target_profile": {"heavy_atoms": 25}},
                artifacts={
                    "literature_pdf_structure_evidence": {
                        "schema_version": "literature_pdf_structure_evidence.v1",
                        "accepted": True,
                        "source_ref": "doi:10.1186/s12934-021-01717-w",
                        "source_pdf_path": str(Path(tmp) / "article_pdf.pdf"),
                        "rendered_pages": [{"image_path": str(Path(tmp) / "page.png")}],
                    }
                },
            )
            evidence = _pdf_evidence_from_payload_or_artifacts(
                state,
                {"source_ref": "patent:US4397947A", "source_title": "Microbial process for 9alpha-hydroxylation of steroids"},
            )

        self.assertEqual(evidence, {})

    def test_local_pdf_binding_accepts_scout_source_alias_for_same_doi_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "article_pdf.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% fake content should not be parsed\n")
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={
                    "target_name": "9-OH-4-HP",
                    "target_smiles": C22_9OH_4HP_SMILES,
                    "family_hint": "steroid biotransformation",
                    "local_literature_cache": [
                        {
                            "candidate_id": "provided_pdf_1",
                            "source_ref": "doi:10.1186/s12934-021-01717-w",
                            "doi": "10.1186/s12934-021-01717-w",
                            "local_pdf": str(pdf),
                            "source_role": "local_cache",
                        }
                    ],
                },
                preflight={"case_id": "target1", "accepted": True, "target_profile": {"heavy_atoms": 25}},
                artifacts={
                    "literature_scout_report": {
                        "source_candidates": [
                            {
                                "candidate_id": "lit_001",
                                "source_ref": "src_001",
                                "doi": "10.1186/s12934-021-01717-w",
                                "title": "Production of 9,21-dihydroxy-20-methyl-pregna-4-en-3-one from phytosterols",
                                "local_pdf": str(pdf),
                            }
                        ]
                    }
                },
            )
            binding = _validate_local_pdf_source_binding(
                state,
                {"pdf_path": str(pdf), "source_ref": "src_001"},
                pdf_path=pdf,
            )

        self.assertTrue(binding["accepted"], binding)
        self.assertEqual(binding["payload"]["source_ref"], "doi:10.1186/s12934-021-01717-w")
        self.assertIn("Production of 9,21-dihydroxy", binding["payload"]["source_title"])

    def test_blackboard_reroutes_process_evidence_to_objective_path(self):
        target = TargetInput(target_name="9-OH-4-HP", target_smiles=C22_9OH_4HP_SMILES, family_hint="steroid biotransformation")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["route_objective_summary"] = {
            "schema_version": "route_objective_summary.v1",
            "selected_objectives": [
                {
                    "objective_id": "route_objective:biotransformation_endpoint",
                    "objective_type": "biotransformation_endpoint",
                    "proof_type": "biotransformation_proof",
                }
            ],
        }
        board["target_side_disconnection_hypotheses"] = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "hypotheses": [{"hypothesis_id": "same_core_biotransformation", "target_handle": "biotransformation_endpoint"}],
        }
        board["endpoint_candidates"] = [
            {"endpoint_id": "endpoint:biotransformation_endpoint:same_core_biotransformation_substrate"}
        ]
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "visual_process_gap",
            "action_type": "extract_visual_literature_chain",
            "payload": _process_payload(),
        }
        result = {
            "accepted": False,
            "result": {
                **_process_visual_result(),
                "literature_process_evidence_rows": process_evidence_rows_from_visual_result(
                    _process_visual_result(),
                    payload=_process_payload(),
                    artifact_ref="/tmp/visual.json",
                ),
            },
            "reasons": ["visual_literature_chain_has_no_steps"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action=action,
                action_result=result,
                round_index=1,
                run_dir=tmp,
            )

        evidence = board["literature_evidence"]
        self.assertEqual(len(evidence["process_evidence_rows"]), 1)
        self.assertEqual(board["action_history"][-1]["blackboard_delta"]["process_evidence_rows"], 1)
        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn("derive_broad_reaction_template", action_types)
        self.assertIn("compile_objective_route_proof", action_types)
        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertNotIn("resolve_literature_structure_task", action_types)

    def test_process_evidence_bias_promotes_route_progress_over_more_pdf_extraction(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_SMILES,
            family_hint="statin atorvastatin",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["route_objective_summary"] = {
            "schema_version": "route_objective_summary.v1",
            "selected_objectives": [
                {
                    "objective_id": "route_objective:advanced_intermediate_anchor",
                    "objective_type": "advanced_intermediate_anchor",
                    "proof_type": "process_anchor_guided_route",
                }
            ],
        }
        board["target_side_disconnection_hypotheses"] = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "hypotheses": [{"hypothesis_id": "atorvastatin_paal_knorr_anchor"}],
        }
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "src_003",
                "title": "Older atorvastatin analogue paper",
                "local_pdf": "/tmp/older_atorvastatin.pdf",
            }
        ]
        board["literature_evidence"]["process_evidence_rows"] = [
            {
                "schema_version": "literature_process_evidence_row.v1",
                "row_id": "process_evidence:doi_10_1186_s13065_015_0082_7:atorvastatin",
                "process_type": "small_molecule_process_route",
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "endpoint_labels": ["atorvastatin calcium"],
                "substrate_or_feedstock_labels": ["advanced ketal ester intermediate 4"],
                "biocatalyst_or_process_labels": ["Paal-Knorr pyrrole construction", "ester hydrolysis"],
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "not_reaction_smiles": True,
                "no_solved_claim": True,
            }
        ]
        board["current_belief"]["next_action_bias"] = [
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "derive_broad_reaction_template",
            "compile_objective_route_proof",
        ]

        batch = plan_action_batch(board, round_index=2, max_actions=2)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types, ["derive_broad_reaction_template", "compile_objective_route_proof"])

    def test_objective_proof_compile_is_not_repeated_without_new_signal(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_SMILES,
            family_hint="statin atorvastatin",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["route_objective_summary"] = {
            "schema_version": "route_objective_summary.v1",
            "selected_objectives": [{"objective_id": "route_objective:advanced_intermediate_anchor"}],
        }
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "src_003", "title": "Atorvastatin source", "local_pdf": "/tmp/atorvastatin.pdf"}
        ]
        board["literature_evidence"]["process_evidence_rows"] = [
            {
                "schema_version": "literature_process_evidence_row.v1",
                "row_id": "process_evidence:atorvastatin",
                "process_type": "small_molecule_process_route",
                "endpoint_labels": ["atorvastatin calcium"],
                "substrate_or_feedstock_labels": ["advanced ketal ester intermediate 4"],
                "biocatalyst_or_process_labels": ["Paal-Knorr pyrrole construction"],
                "no_solved_claim": True,
            }
        ]
        board["route_proof_bundle"] = {
            "schema_version": "route_proof_bundle.v1",
            "accepted": False,
            "route_status": "plausible_hypothesis_route",
            "reasons": ["deterministic_connected_route_not_proven"],
        }
        board["current_belief"]["next_action_bias"] = [
            "compile_objective_route_proof",
            "extract_pdf_literature_structures",
        ]
        board["action_history"] = [
            {
                "action_type": "compile_objective_route_proof",
                "changed_blackboard_fields": ["artifact_refs"],
                "useful_artifact": True,
            }
        ]

        batch = plan_action_batch(board, round_index=3, max_actions=2)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("compile_objective_route_proof", action_types)
        self.assertIn("extract_pdf_literature_structures", action_types)

    def test_validator_rejects_compile_exact_rows_without_visual_steps(self):
        target = TargetInput(target_name="9-OH-4-HP", target_smiles=C22_9OH_4HP_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1186/s12934-021-01717-w",
                "title": "Exact endpoint process source",
                "expected_scheme_or_compound_labels": ["9-OH-4-HP"],
            }
        ]
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "bad_compile",
                    "action_type": "compile_exact_literature_rows",
                    "rationale": "compile source metadata directly",
                    "expected_artifact": "exact rows",
                    "success_condition": "rows exist",
                    "payload": {"source_ref": "doi:10.1186/s12934-021-01717-w"},
                }
            ],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn("compile_exact_literature_rows_requires_uncompiled_visual_steps:0", validation["reasons"])


if __name__ == "__main__":
    unittest.main()
