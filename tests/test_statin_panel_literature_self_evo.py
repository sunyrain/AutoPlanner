import contextlib
import importlib.util
import io
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.agent.statin_panel import (
    NATURAL_STATINS,
    SYNTHETIC_STATINS,
    STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA,
    STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA,
    STATIN_FULLFLOW_OVERVIEW_SCHEMA,
    STATIN_LITERATURE_QUERY_TRACE_SCHEMA,
    STATIN_MEMBER_ROUTE_TEMPLATE_SCHEMA,
    STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA,
    STATIN_ROUTE_TEMPLATE_STEP_SCHEMA,
    _closure_lead_route_relevance,
    _carry_forward_open_gap_search_traces,
    _closure_open_gap_followup_tasks,
    _closure_followup_execution,
    _execute_single_open_gap_search,
    _execute_open_gap_full_text_access_probes,
    _execute_open_gap_search_packages,
    load_statin_panel_targets,
    run_statin_panel_literature_self_evo,
)


class StatinPanelLiteratureSelfEvoTest(unittest.TestCase):
    def test_loads_all_nine_statin_targets_with_family_expectations(self):
        targets = load_statin_panel_targets()
        by_safe = {target.safe: target for target in targets}

        self.assertEqual(len(targets), 9)
        self.assertEqual(set(by_safe), NATURAL_STATINS | SYNTHETIC_STATINS)
        for safe in NATURAL_STATINS:
            self.assertEqual(by_safe[safe].expected_reaction_class, "statin_semisynthesis")
            self.assertEqual(by_safe[safe].expected_family_id, "natural_statin_semisynthesis")
        for safe in SYNTHETIC_STATINS:
            self.assertEqual(by_safe[safe].expected_reaction_class, "statin_side_chain_convergence")
            self.assertEqual(by_safe[safe].expected_family_id, "synthetic_statin")

    def test_pubmed_lead_relevance_guard_separates_route_sources_from_clinical_context(self):
        target = {item.safe: item for item in load_statin_panel_targets()}["atorvastatin"]

        route_relevance = _closure_lead_route_relevance(
            source_title="Full text synthesis route for atorvastatin intermediates",
            journal="Organic process research",
            query="atorvastatin process synthesis",
            abstract_signal_terms=["synthesis", "intermediate", "process"],
            target=target,
            requirement_id="full_text_route_step_audit",
        )
        clinical_relevance = _closure_lead_route_relevance(
            source_title="Atorvastatin inhibits neuronal apoptosis in hypoxic-ischemic neonatal rats",
            journal="FASEB journal",
            query="atorvastatin process synthesis",
            abstract_signal_terms=["process"],
            target=target,
            requirement_id="full_text_route_step_audit",
        )

        self.assertEqual(route_relevance["lead_relevance_status"], "route_relevant_strong")
        self.assertGreaterEqual(route_relevance["route_relevance_score"], 5)
        self.assertIn("intermediate", route_relevance["route_relevance_strong_signals"])
        self.assertEqual(clinical_relevance["lead_relevance_status"], "non_route_context_suspected")
        self.assertIn("neuronal", clinical_relevance["route_context_guard_signals"])

    def test_pubmed_lead_relevance_guard_rejects_off_target_salt_or_vascular_noise(self):
        fluvastatin = {item.safe: item for item in load_statin_panel_targets()}["fluvastatin"]
        pravastatin = {item.safe: item for item in load_statin_panel_targets()}["pravastatin"]

        off_target_salt = _closure_lead_route_relevance(
            source_title="Material Composition Characteristics of Aspergillus cristatus under High Salt Stress through LC-MS Metabolomics.",
            journal="Food chemistry",
            query="fluvastatin endpoint characterization salt",
            abstract_signal_terms=["synthesis", "salt", "intermediate"],
            target=fluvastatin,
            requirement_id="endpoint_identity_and_salt_state_audit",
        )
        vascular_context = _closure_lead_route_relevance(
            source_title="Pravastatin Corrects Endothelial Dysfunction in Ex Vivo Uterine Radial Arteries in Preeclampsia.",
            journal="Hypertension",
            query="pravastatin intermediate isolation",
            abstract_signal_terms=["process"],
            target=pravastatin,
            requirement_id="condition_and_workup_evidence_audit",
        )
        analytical_context = _closure_lead_route_relevance(
            source_title="[Simultaneous determination of statins in dietary supplements by ultra-performance liquid chromatography].",
            journal="Food hygiene",
            query="cerivastatin intermediate isolation",
            abstract_signal_terms=["intermediate"],
            target={item.safe: item for item in load_statin_panel_targets()}["cerivastatin"],
            requirement_id="condition_and_workup_evidence_audit",
        )
        vascular_smooth_muscle_context = _closure_lead_route_relevance(
            source_title="Statin-exposed vascular smooth muscle cells secrete proteoglycans with decreased binding affinity for LDL.",
            journal="Arteriosclerosis thrombosis and vascular biology",
            query="cerivastatin side chain synthesis",
            abstract_signal_terms=["synthesis", "intermediate", "process"],
            target={item.safe: item for item in load_statin_panel_targets()}["cerivastatin"],
            requirement_id="full_text_route_step_audit",
        )
        non_lipid_context = _closure_lead_route_relevance(
            source_title="Non-lipid-related effects of statins.",
            journal="Annual review of pharmacology",
            query="fluvastatin intermediate synthesis",
            abstract_signal_terms=["synthesis", "intermediate", "process"],
            target=fluvastatin,
            requirement_id="full_text_route_step_audit",
        )
        whole_cell_process = _closure_lead_route_relevance(
            source_title="A highly productive, whole-cell DERA chemoenzymatic process for production of key lactonized side-chain intermediates in statin synthesis.",
            journal="Organic Process Research and Development",
            query="pitavastatin side chain synthesis",
            abstract_signal_terms=["synthesis", "intermediate", "process"],
            target={item.safe: item for item in load_statin_panel_targets()}["pitavastatin"],
            requirement_id="full_text_route_step_audit",
        )

        self.assertNotEqual(off_target_salt["lead_relevance_status"], "route_relevant_strong")
        self.assertIn("off_target_title", off_target_salt["route_context_guard_signals"])
        self.assertIn("metabolomics", off_target_salt["route_context_guard_signals"])
        self.assertEqual(vascular_context["lead_relevance_status"], "non_route_context_suspected")
        self.assertIn("preeclampsia", vascular_context["route_context_guard_signals"])
        self.assertEqual(analytical_context["lead_relevance_status"], "non_route_context_suspected")
        self.assertIn("chromatography", analytical_context["route_context_guard_signals"])
        self.assertEqual(
            vascular_smooth_muscle_context["lead_relevance_status"],
            "non_route_context_suspected",
        )
        self.assertIn("vascular smooth muscle", vascular_smooth_muscle_context["route_context_guard_signals"])
        self.assertNotEqual(non_lipid_context["lead_relevance_status"], "route_relevant_strong")
        self.assertIn("non-lipid", non_lipid_context["route_context_guard_signals"])
        self.assertEqual(whole_cell_process["lead_relevance_status"], "route_relevant_strong")
        self.assertIn("side-chain", whole_cell_process["route_relevance_strong_signals"])

    def test_open_gap_search_execution_carries_forward_existing_traces(self):
        task = {
            "task_id": "atorvastatin:condition_and_workup_evidence_audit",
            "target_safe": "atorvastatin",
            "target_name": "atorvastatin",
            "family_bucket": "synthetic_statin",
            "requirement_id": "condition_and_workup_evidence_audit",
            "priority": "P2",
            "followup_query": "atorvastatin synthesis conditions",
        }
        open_fields = [
            {
                "field": "condition_presence_evidence",
                "status": "audited_gap_local_curator_record",
                "summary": "condition evidence remains open",
                "resolution_required_before_promotion": True,
            },
            {
                "field": "workup_or_isolation_presence",
                "status": "audited_gap_local_curator_record",
                "summary": "workup evidence remains open",
                "resolution_required_before_promotion": True,
            },
        ]
        followups = _closure_open_gap_followup_tasks(
            task,
            open_fields,
            lead_sources=[],
            selected_source_refs=[],
            context_guarded_source_refs=[],
            weak_or_rejected_source_refs=[],
        )
        result = {
            "result_id": "atorvastatin:condition_and_workup_evidence_audit:curation_result",
            "open_gap_followup_tasks": followups,
        }
        previous_trace = {
            "schema_version": "statin_open_gap_search_execution_trace.v1",
            "execution_status": "pubmed_open_gap_search_executed_no_hits",
            "backend_resolved": "pubmed_open_gap_search",
            "target_name": "atorvastatin",
            "requirement_id": "condition_and_workup_evidence_audit",
            "field": "condition_presence_evidence",
            "query": "atorvastatin synthesis conditions",
            "query_variants": ["atorvastatin synthesis conditions"],
            "query_attempt_count": 1,
            "hit_count": 0,
            "evidence_lead_refs": [],
            "lead_sources": [],
            "selected_route_source_refs": [],
            "context_guarded_source_refs": [],
            "weak_or_rejected_source_refs": [],
            "abstract_signal_status": "",
            "abstract_signal_terms": [],
            "search_sources": ["pubmed_open_gap_search"],
            "lead_relevance_gate": "no_leads",
            "not_template_support": True,
            "not_lab_procedure": True,
        }
        previous = {
            "results": [
                {
                    "open_gap_followup_tasks": [
                        {
                            "followup_id": followups[0]["followup_id"],
                            "literature_triage": {
                                "search_execution_package": {
                                    "execution_trace": previous_trace,
                                }
                            },
                        }
                    ]
                }
            ]
        }

        carried = _carry_forward_open_gap_search_traces([result], previous)
        summary = _execute_open_gap_search_packages(
            [result],
            execute_open_gap_searches=False,
            open_gap_search_limit=1,
            carried_forward_search_traces=carried,
        )

        self.assertEqual(carried, 1)
        self.assertEqual(summary["candidate_search_required_count"], 2)
        self.assertEqual(summary["runnable_search_required_count"], 1)
        self.assertEqual(summary["carried_forward_search_trace_count"], 1)

    def test_context_guarded_open_gap_trace_is_not_carried_forward_as_done(self):
        task = {
            "task_id": "atorvastatin:condition_and_workup_evidence_audit",
            "target_safe": "atorvastatin",
            "target_name": "atorvastatin",
            "family_bucket": "synthetic_statin",
            "requirement_id": "condition_and_workup_evidence_audit",
            "priority": "P2",
            "followup_query": "atorvastatin synthesis conditions",
        }
        open_fields = [
            {
                "field": "condition_presence_evidence",
                "status": "audited_gap_local_curator_record",
                "summary": "condition evidence remains open",
                "resolution_required_before_promotion": True,
            },
            {
                "field": "workup_or_isolation_presence",
                "status": "audited_gap_local_curator_record",
                "summary": "workup evidence remains open",
                "resolution_required_before_promotion": True,
            },
        ]
        followups = _closure_open_gap_followup_tasks(
            task,
            open_fields,
            lead_sources=[],
            selected_source_refs=[],
            context_guarded_source_refs=[],
            weak_or_rejected_source_refs=[],
        )
        result = {
            "result_id": "atorvastatin:condition_and_workup_evidence_audit:curation_result",
            "open_gap_followup_tasks": followups,
        }
        previous_trace = {
            "schema_version": "statin_open_gap_search_execution_trace.v1",
            "execution_status": "pubmed_open_gap_search_executed_with_leads",
            "backend_resolved": "pubmed_open_gap_search",
            "target_name": "atorvastatin",
            "requirement_id": "condition_and_workup_evidence_audit",
            "field": "condition_presence_evidence",
            "query": "atorvastatin synthesis conditions",
            "query_attempt_count": 1,
            "hit_count": 1,
            "evidence_lead_refs": ["ev_pubmed_14531725"],
            "lead_sources": [
                {
                    "evidence_ref": "ev_pubmed_14531725",
                    "source_title": "Clinical pharmacokinetics of atorvastatin.",
                    "lead_relevance_status": "non_route_context_suspected",
                    "route_context_guard_signals": ["clinical", "pharmacokinetic"],
                    "not_template_support": True,
                    "not_lab_procedure": True,
                }
            ],
            "selected_route_source_refs": [],
            "context_guarded_source_refs": ["ev_pubmed_14531725"],
            "weak_or_rejected_source_refs": ["ev_pubmed_14531725"],
            "search_sources": ["pubmed_open_gap_search"],
            "lead_relevance_gate": "lead_metadata_only_or_context_guarded",
            "not_template_support": True,
            "not_lab_procedure": True,
        }
        previous = {
            "results": [
                {
                    "open_gap_followup_tasks": [
                        {
                            "followup_id": followups[0]["followup_id"],
                            "literature_triage": {
                                "search_execution_package": {
                                    "execution_trace": previous_trace,
                                }
                            },
                        }
                    ]
                }
            ]
        }

        carried = _carry_forward_open_gap_search_traces([result], previous)
        summary = _execute_open_gap_search_packages(
            [result],
            execute_open_gap_searches=False,
            open_gap_search_limit=1,
            carried_forward_search_traces=carried,
        )

        self.assertEqual(carried, 0)
        self.assertEqual(summary["candidate_search_required_count"], 2)
        self.assertEqual(summary["runnable_search_required_count"], 2)

    def test_open_gap_pubmed_search_retries_after_context_guarded_hits(self):
        task = {
            "task_id": "atorvastatin:condition_and_workup_evidence_audit",
            "target_safe": "atorvastatin",
            "target_name": "atorvastatin",
            "family_bucket": "synthetic_statin",
            "requirement_id": "condition_and_workup_evidence_audit",
            "priority": "P2",
            "followup_query": "atorvastatin synthesis conditions",
        }
        open_fields = [
            {
                "field": "condition_presence_evidence",
                "status": "audited_gap_local_curator_record",
                "summary": "condition evidence remains open",
                "resolution_required_before_promotion": True,
            }
        ]
        followup = _closure_open_gap_followup_tasks(
            task,
            open_fields,
            lead_sources=[],
            selected_source_refs=[],
            context_guarded_source_refs=[],
            weak_or_rejected_source_refs=[],
        )[0]

        clinical_card = SimpleNamespace(
            evidence_id="ev_pubmed_14531725",
            source_title="Clinical pharmacokinetics of atorvastatin.",
            url="https://pubmed.ncbi.nlm.nih.gov/14531725/",
            doi="10.2165/00003088-200342130-00005",
            source_record_id="pubmed:14531725",
            source_metadata={
                "pmid": "14531725",
                "journal": "Clinical pharmacokinetics",
                "pubdate": "2003",
                "query": "atorvastatin synthesis conditions",
                "abstract_signal_audit": {
                    "route_signal_status": "abstract_route_signal_detected",
                    "route_signal_terms": ["route", "lactone"],
                    "abstract_available": True,
                    "abstract_text_char_count": 2736,
                },
            },
        )
        route_card = SimpleNamespace(
            evidence_id="ev_pubmed_999999",
            source_title="Full text synthesis route for atorvastatin intermediates",
            url="https://pubmed.ncbi.nlm.nih.gov/999999/",
            doi="10.1000/atorvastatin-route",
            source_record_id="pubmed:999999",
            source_metadata={
                "pmid": "999999",
                "journal": "Organic Process Research and Development",
                "pubdate": "2004",
                "query": "atorvastatin process chemistry synthesis conditions intermediate",
                "abstract_signal_audit": {
                    "route_signal_status": "abstract_route_signal_detected",
                    "route_signal_terms": ["synthesis", "intermediate", "process"],
                    "abstract_available": True,
                    "abstract_text_char_count": 512,
                },
            },
        )
        clinical_report = {
            "schema_version": "literature_followup_search_report.v1",
            "query": "atorvastatin synthesis conditions",
            "query_variants": ["atorvastatin synthesis conditions"],
            "query_attempt_count": 1,
            "resolved_query": "atorvastatin synthesis conditions",
            "fallback_used": False,
            "searches": [{"source": "pubmed_followup_esearch", "hits": 1}],
            "hit_count": 1,
            "evidence_levels": {"medium": 1},
            "abstract_signal_audit_requested": True,
            "abstract_signal_status": "abstract_route_signal_detected",
            "abstract_signal_record_count": 1,
            "abstract_signal_hit_count": 1,
            "abstract_signal_terms": ["route", "lactone"],
            "abstract_signal_audit": [
                {
                    "pmid": "14531725",
                    "route_signal_terms": ["route", "lactone"],
                }
            ],
        }
        route_report = {
            "schema_version": "literature_followup_search_report.v1",
            "query": "atorvastatin process chemistry synthesis conditions intermediate",
            "query_variants": ["atorvastatin process chemistry synthesis conditions intermediate"],
            "query_attempt_count": 1,
            "resolved_query": "atorvastatin process chemistry synthesis conditions intermediate",
            "fallback_used": False,
            "searches": [{"source": "pubmed_followup_esearch", "hits": 1}],
            "hit_count": 1,
            "evidence_levels": {"medium": 1},
            "abstract_signal_audit_requested": True,
            "abstract_signal_status": "abstract_route_signal_detected",
            "abstract_signal_record_count": 1,
            "abstract_signal_hit_count": 1,
            "abstract_signal_terms": ["synthesis", "intermediate", "process"],
            "abstract_signal_audit": [
                {
                    "pmid": "999999",
                    "route_signal_terms": ["synthesis", "intermediate", "process"],
                }
            ],
        }

        with patch(
            "cascade_planner.agent.statin_panel.retrieve_pubmed_query_evidence",
            side_effect=[([clinical_card], clinical_report), ([route_card], route_report)],
        ) as retrieve:
            trace = _execute_single_open_gap_search(followup)

        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(trace["lead_relevance_gate"], "route_relevant_strong", trace)
        self.assertEqual(trace["query_attempt_count"], 2, trace)
        self.assertTrue(trace["fallback_used"], trace)
        self.assertIn("ev_pubmed_999999", trace["selected_route_source_refs"], trace)
        self.assertIn("ev_pubmed_14531725", trace["context_guarded_source_refs"], trace)
        self.assertIn("ev_pubmed_14531725", trace["weak_or_rejected_source_refs"], trace)
        self.assertEqual(trace["resolved_query"], "atorvastatin process chemistry synthesis conditions intermediate")
        self.assertIn("statin_route_relevance_filter", trace["search_sources"])

    def test_full_text_access_probe_adds_metadata_candidate_without_resolving_open_gap(self):
        task = {
            "task_id": "atorvastatin:endpoint_identity_and_salt_state_audit",
            "target_safe": "atorvastatin",
            "target_name": "atorvastatin",
            "family_bucket": "synthetic_statin",
            "requirement_id": "endpoint_identity_and_salt_state_audit",
            "priority": "P1",
            "followup_query": "atorvastatin endpoint characterization",
        }
        open_fields = [
            {
                "field": "counterion_or_characterization_refs",
                "status": "audited_gap_local_curator_record",
                "summary": "endpoint characterization remains open",
                "resolution_required_before_promotion": True,
            }
        ]
        lead_sources = [
            {
                "schema_version": "statin_closure_pubmed_lead_source.v1",
                "evidence_ref": "ev_pubmed_32296962",
                "source_type": "pubmed",
                "pmid": "32296962",
                "source_record_id": "pubmed:32296962",
                "source_title": "[(18)F]Atorvastatin: synthesis of a potential molecular imaging tool.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/32296962/",
                "doi": "10.1186/s13550-020-00622-4",
                "journal": "EJNMMI research",
                "pubdate": "2020 Apr 15",
                "lead_relevance_status": "route_relevant_strong",
                "route_relevance_score": 10,
                "route_relevance_strong_signals": ["synthesis of"],
                "route_context_guard_signals": [],
                "not_template_support": True,
                "not_lab_procedure": True,
            }
        ]
        followups = _closure_open_gap_followup_tasks(
            task,
            open_fields,
            lead_sources=lead_sources,
            selected_source_refs=["ev_pubmed_32296962"],
            context_guarded_source_refs=[],
            weak_or_rejected_source_refs=[],
        )
        result = {
            "result_id": "atorvastatin:endpoint_identity_and_salt_state_audit:curation_result",
            "open_gap_followup_tasks": followups,
        }

        with patch(
            "cascade_planner.agent.statin_panel._pubmed_pmc_links_for_pmid",
            return_value=["7158976"],
        ) as probe:
            summary = _execute_open_gap_full_text_access_probes(
                [result],
                execute_full_text_access_probes=True,
                full_text_access_probe_limit=-1,
            )

        followup = followups[0]
        review_draft = followup["literature_triage"]["curator_review_draft"]
        access_package = review_draft["full_text_access_package"]
        access_probe = access_package["probes"][0]
        resolution_candidate = review_draft["field_resolution_candidate"]

        self.assertEqual(probe.call_count, 1)
        self.assertEqual(summary["review_ready_followup_count"], 1)
        self.assertEqual(summary["executed_probe_count"], 1)
        self.assertEqual(summary["open_access_candidate_count"], 1)
        self.assertEqual(
            access_package["execution_status"],
            "full_text_access_probe_executed_with_open_access_candidate",
        )
        self.assertEqual(access_probe["full_text_access_status"], "pmc_open_access_link_available")
        self.assertEqual(access_probe["pmcids"], ["7158976"])
        self.assertFalse(access_probe["full_text_content_stored"])
        self.assertNotIn("abstract_text", access_probe)
        self.assertNotIn("raw_reaction", access_probe)
        self.assertEqual(
            resolution_candidate["candidate_status"],
            "full_text_access_candidate_ready_for_curator",
        )
        self.assertEqual(resolution_candidate["candidate_field_resolution"], "still_blocked")
        self.assertEqual(resolution_candidate["resolution_confidence"], "not_resolved_metadata_only")
        self.assertFalse(resolution_candidate["promotion_allowed"])
        self.assertFalse(followup["template_promotion_allowed"])
        self.assertFalse(followup["solved_claim_allowed"])
        self.assertEqual(
            followup["self_evo_inbox_entry"]["field_resolution_candidate_status"],
            "full_text_access_candidate_ready_for_curator",
        )

    def test_all_nine_statins_replay_literature_workflow_and_self_evo_promotes_family_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_statin_panel_literature_self_evo(output_root=tmp, literature_backend="local", query_budget=6)
            manifest_path = Path(tmp) / "typed_artifacts" / "statin_typed_artifact_manifest.json"
            manifest_md = Path(tmp) / "typed_artifacts" / "statin_typed_artifact_manifest.md"
            overview_path = Path(tmp) / "statin_panel_fullflow_overview.json"
            overview_md = Path(tmp) / "statin_panel_fullflow_overview.md"
            closure_matrix_path = Path(tmp) / "statin_route_closure_matrix.json"
            closure_matrix_md = Path(tmp) / "statin_route_closure_matrix.md"
            curation_packet_path = Path(tmp) / "statin_closure_lead_curation_packet.json"
            curation_packet_md = Path(tmp) / "statin_closure_lead_curation_packet.md"
            curation_result_set_path = Path(tmp) / "statin_closure_curation_result_set.json"
            curation_result_set_md = Path(tmp) / "statin_closure_curation_result_set.md"

            self.assertTrue((Path(tmp) / "statin_panel_literature_self_evo_report.json").exists())
            self.assertTrue((Path(tmp) / "statin_panel_literature_self_evo_report.md").exists())
            self.assertTrue(overview_path.exists())
            self.assertTrue(overview_md.exists())
            self.assertTrue(closure_matrix_path.exists())
            self.assertTrue(closure_matrix_md.exists())
            self.assertTrue(curation_packet_path.exists())
            self.assertTrue(curation_packet_md.exists())
            self.assertTrue(curation_result_set_path.exists())
            self.assertTrue(curation_result_set_md.exists())
            overview = json.loads(overview_path.read_text(encoding="utf-8"))
            self.assertEqual(overview["schema_version"], STATIN_FULLFLOW_OVERVIEW_SCHEMA)
            self.assertFalse(overview["skipped"])
            self.assertEqual(overview["target_count"], 9)
            self.assertTrue(overview["validation"]["accepted"], overview)
            self.assertEqual(len(overview["targets"]), 9)
            self.assertIn("九他汀全流程合成模板总览", overview_md.read_text(encoding="utf-8"))
            for item in overview["targets"]:
                self.assertEqual(item["route_status"], "partial_anchor", item)
                self.assertTrue(item["not_lab_procedure"], item)
                self.assertTrue(item["template_sources"], item)
                self.assertGreaterEqual(len(item["synthesis_stages"]), 3, item)
                self.assertGreaterEqual(len(item["route_template_steps"]), 3, item)
                self.assertGreaterEqual(len(item["difficulty_queries"]), 3, item)
                self.assertGreaterEqual(len(item["query_execution_traces"]), 3, item)
                self.assertEqual(
                    item["literature_trace"]["accepted_query_trace_count"],
                    len(item["difficulty_queries"]),
                    item,
                )
                self.assertEqual(
                    item["literature_trace"]["template_supported_query_trace_count"],
                    len(item["difficulty_queries"]),
                    item,
                )
                self.assertTrue(item["key_intermediate_roles"], item)
                self.assertTrue(item["validation"]["accepted"], item)
                closure = item["route_closure_audit"]
                self.assertEqual(closure["readiness_status"], "not_ready_for_solved_status", item)
                self.assertFalse(closure["solved_claim_allowed"], item)
                self.assertTrue(closure["validation"]["accepted"], item)
                self.assertGreaterEqual(closure["passed_requirement_count"], 3, item)
                self.assertGreaterEqual(closure["blocker_count"], 4, item)
                self.assertGreaterEqual(closure["followup_query_count"], closure["blocker_count"], item)
                self.assertEqual(item["self_evolution"]["status"], "staging", item)
                self.assertEqual(item["self_evolution"]["kb_target_layer"], "staging", item)
                self.assertTrue(item["self_evolution"]["production_write_blocked"], item)
            closure_matrix = json.loads(closure_matrix_path.read_text(encoding="utf-8"))
            self.assertEqual(closure_matrix["schema_version"], STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA)
            self.assertFalse(closure_matrix["skipped"])
            self.assertEqual(closure_matrix["target_count"], 9)
            self.assertEqual(closure_matrix["queued_blocker_count"], closure_matrix["blocker_count"])
            self.assertEqual(closure_matrix["unresolved_blocker_count"], closure_matrix["blocker_count"])
            self.assertEqual(closure_matrix["solved_claim_allowed_count"], 0)
            self.assertFalse(closure_matrix["full_trace_coverage"])
            self.assertFalse(closure_matrix["full_execution_coverage"])
            self.assertGreaterEqual(closure_matrix["blocker_count"], 9 * 4)
            self.assertTrue(closure_matrix["validation"]["accepted"], closure_matrix)
            self.assertIn("九他汀 Route Closure Blocker Matrix", closure_matrix_md.read_text(encoding="utf-8"))
            curation_packet = json.loads(curation_packet_path.read_text(encoding="utf-8"))
            self.assertEqual(curation_packet["schema_version"], STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA)
            self.assertFalse(curation_packet["skipped"])
            self.assertEqual(curation_packet["target_count"], 9)
            self.assertEqual(curation_packet["task_count"], closure_matrix["blocker_count"])
            self.assertEqual(curation_packet["ready_for_curator_count"], curation_packet["task_count"])
            self.assertEqual(curation_packet["template_promotion_allowed_count"], 0)
            self.assertFalse(curation_packet["full_execution_coverage"])
            self.assertTrue(curation_packet["validation"]["accepted"], curation_packet)
            self.assertIn("九他汀 Closure Lead Curation Packet", curation_packet_md.read_text(encoding="utf-8"))
            curation_result_set = json.loads(curation_result_set_path.read_text(encoding="utf-8"))
            self.assertEqual(curation_result_set["schema_version"], STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA)
            self.assertFalse(curation_result_set["skipped"])
            self.assertEqual(curation_result_set["target_count"], 9)
            self.assertEqual(curation_result_set["task_count"], curation_packet["task_count"])
            self.assertEqual(curation_result_set["result_count"], curation_packet["task_count"])
            self.assertEqual(curation_result_set["blocked_result_count"], curation_result_set["result_count"])
            self.assertEqual(curation_result_set["template_promotion_allowed_count"], 0)
            self.assertEqual(curation_result_set["solved_claim_allowed_count"], 0)
            self.assertGreater(curation_result_set["curator_record_supported_result_count"], 0)
            self.assertGreater(curation_result_set["validated_route_field_count"], 0)
            self.assertGreater(curation_result_set["audited_gap_route_field_count"], 0)
            self.assertEqual(curation_result_set["missing_route_field_count"], 0)
            self.assertGreater(curation_result_set["open_route_field_count"], 0)
            self.assertEqual(
                curation_result_set["open_gap_followup_count"],
                curation_result_set["open_route_field_count"],
            )
            self.assertEqual(
                curation_result_set["open_gap_review_ready_count"]
                + curation_result_set["open_gap_search_required_count"],
                curation_result_set["open_gap_followup_count"],
            )
            self.assertEqual(
                curation_result_set["open_gap_curator_review_draft_count"],
                curation_result_set["open_gap_followup_count"],
            )
            self.assertEqual(
                curation_result_set["open_gap_search_execution_package_count"],
                curation_result_set["open_gap_followup_count"],
            )
            self.assertEqual(
                curation_result_set["open_gap_search_trace_count"],
                curation_result_set["open_gap_followup_count"],
            )
            self.assertEqual(curation_result_set["open_gap_search_executed_count"], 0)
            self.assertEqual(curation_result_set["open_gap_search_lead_count"], 0)
            self.assertEqual(curation_result_set["open_gap_search_selected_source_count"], 0)
            self.assertEqual(
                curation_result_set["open_gap_self_evo_inbox_count"],
                curation_result_set["open_gap_followup_count"],
            )
            self.assertEqual(
                curation_result_set["candidate_template_gate_status"],
                "blocked_pending_full_text_or_curator_records",
            )
            self.assertTrue(curation_result_set["production_write_blocked"])
            self.assertTrue(curation_result_set["not_lab_procedure"])
            self.assertTrue(curation_result_set["validation"]["accepted"], curation_result_set)
            first_result = curation_result_set["results"][0]
            self.assertEqual(first_result["candidate_template_gate_status"], "blocked_pending_full_text_or_curator_records")
            self.assertGreater(first_result["validated_route_field_count"], 0)
            self.assertTrue(first_result["full_text_or_curator_record_refs"])
            self.assertTrue(
                all(ref.startswith("local_curator:") for ref in first_result["full_text_or_curator_record_refs"])
            )
            self.assertFalse(first_result["template_promotion_allowed"])
            self.assertFalse(first_result["self_evo_template_candidate"]["promotion_allowed"])
            self.assertEqual(first_result["self_evo_template_candidate"]["allowed_layer"], "candidate_only")
            open_results = [
                result for result in curation_result_set["results"]
                if result["open_route_field_count"] > 0
            ]
            self.assertTrue(open_results)
            for result in open_results:
                open_fields = {
                    row["field"]
                    for row in result["route_field_audit"]
                    if row["resolution_required_before_promotion"]
                }
                followups = result["open_gap_followup_tasks"]
                self.assertEqual({item["field"] for item in followups}, open_fields)
                for followup in followups:
                    self.assertEqual(followup["status"], "queued_for_full_text_or_curator_record")
                    self.assertTrue(followup["followup_query"])
                    self.assertTrue(followup["source_requirement"])
                    self.assertTrue(followup["acceptance_signals"])
                    self.assertFalse(followup["template_promotion_allowed"])
                    self.assertFalse(followup["solved_claim_allowed"])
                    self.assertTrue(followup["production_write_blocked"])
                    self.assertTrue(followup["not_template_support"])
                    triage = followup["literature_triage"]
                    self.assertEqual(triage["schema_version"], "statin_closure_open_gap_literature_triage.v1")
                    self.assertTrue(triage["query_variants"])
                    self.assertTrue(triage["full_text_or_curator_record_required"])
                    self.assertIn(
                        triage["triage_status"],
                        {"selected_source_review_ready", "route_specific_search_required"},
                    )
                    self.assertTrue(triage["not_template_support"])
                    self.assertTrue(triage["not_lab_procedure"])
                    self.assertIn(
                        "abstract_text",
                        triage["curator_resolution_schema"]["forbidden_fields"],
                    )
                    review_draft = triage["curator_review_draft"]
                    self.assertEqual(
                        review_draft["schema_version"],
                        "statin_open_gap_curator_review_draft.v1",
                    )
                    self.assertEqual(review_draft["field"], followup["field"])
                    self.assertEqual(review_draft["candidate_field_resolution"], "still_blocked")
                    self.assertEqual(review_draft["resolution_confidence"], "not_resolved_metadata_only")
                    self.assertTrue(review_draft["curator_questions"])
                    self.assertTrue(review_draft["evidence_gap_statement"])
                    self.assertFalse(review_draft["promotion_allowed"])
                    self.assertTrue(review_draft["production_write_blocked"])
                    self.assertTrue(review_draft["not_template_support"])
                    search_package = triage["search_execution_package"]
                    self.assertEqual(
                        search_package["schema_version"],
                        "statin_open_gap_search_execution_package.v1",
                    )
                    self.assertEqual(search_package["field"], followup["field"])
                    self.assertEqual(search_package["execution_status"], "ready_for_pubmed_or_manual_search")
                    self.assertTrue(search_package["query_variants"])
                    self.assertTrue(search_package["source_acceptance_filters"])
                    self.assertIn("abstract_text", search_package["forbidden_fields"])
                    trace = search_package["execution_trace"]
                    self.assertEqual(
                        trace["schema_version"],
                        "statin_open_gap_search_execution_trace.v1",
                    )
                    self.assertEqual(trace["field"], followup["field"])
                    self.assertEqual(trace["execution_status"], "queued_not_executed")
                    self.assertEqual(trace["hit_count"], 0)
                    self.assertFalse(trace["evidence_lead_refs"])
                    self.assertTrue(trace["not_template_support"])
                    inbox = followup["self_evo_inbox_entry"]
                    self.assertEqual(
                        inbox["schema_version"],
                        "statin_closure_open_gap_self_evo_inbox_entry.v1",
                    )
                    self.assertEqual(inbox["field"], followup["field"])
                    self.assertEqual(inbox["allowed_layer"], "candidate_only")
                    self.assertFalse(inbox["promotion_allowed"])
                    self.assertTrue(inbox["production_write_blocked"])
                    self.assertTrue(inbox["not_template_support"])
            self.assertIn("九他汀 Closure Curation Result Set", curation_result_set_md.read_text(encoding="utf-8"))
            self.assertTrue(manifest_path.exists())
            self.assertTrue(manifest_md.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["skipped"])
            self.assertEqual(manifest["artifact_count"], 32)
            self.assertTrue(manifest["validation_summary"]["accepted"], manifest)
            self.assertEqual(manifest["validation_summary"]["accepted_count"], 32)
            self.assertEqual(manifest["validation_summary"]["rejected_count"], 0)
            self.assertTrue(manifest["validation_summary"]["artifact_count_matches_full_panel_contract"])
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinFullflowOverview"),
                1,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinRouteClosureMatrix"),
                1,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinClosureLeadCurationPacket"),
                1,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinClosureCurationResultSet"),
                1,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinFullflowDossier"),
                9,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinRouteTemplate"),
                9,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinRouteClosureAudit"),
                9,
            )
            self.assertEqual(
                [row["artifact_type"] for row in manifest["artifacts"]].count("StatinPanelSelfEvoReport"),
                1,
            )
            for artifact in manifest["artifacts"]:
                self.assertTrue(Path(artifact["json"]).exists(), artifact)
                self.assertTrue(artifact["validation"]["accepted"], artifact)
            for row in report["targets"]:
                dossier = row["fullflow_dossier"]
                self.assertTrue(Path(dossier["json"]).exists())
                self.assertTrue(Path(dossier["markdown"]).exists())
                payload = json.loads(Path(dossier["json"]).read_text(encoding="utf-8"))
                markdown = Path(dossier["markdown"]).read_text(encoding="utf-8")
                self.assertTrue(payload["not_lab_procedure"])
                self.assertNotEqual(payload["route_status"], "solved")
                self.assertGreaterEqual(len(payload["difficulty_escalation"]), 3)
                self.assertIn("## Member-Specific Blueprint", markdown)
                self.assertIn("## Automatic Literature Escalation", markdown)
                self.assertGreaterEqual(
                    len(payload["fullflow_blueprint"]["member_specific_route_outline"]),
                    3,
                    row,
                )
                self.assertGreaterEqual(
                    len(payload["fullflow_blueprint"]["key_intermediate_roles"]),
                    2,
                    row,
                )
                self.assertGreaterEqual(
                    len(payload["automatic_literature_escalation"]["difficulty_queries"]),
                    3,
                    row,
                )
                trace_summary = payload["automatic_literature_escalation"]["query_trace_summary"]
                query_traces = payload["automatic_literature_escalation"]["query_execution_traces"]
                self.assertEqual(len(query_traces), len(payload["automatic_literature_escalation"]["difficulty_queries"]))
                self.assertEqual(trace_summary["accepted_query_trace_count"], len(query_traces))
                self.assertTrue(trace_summary["all_queries_have_validated_evidence"])
                self.assertEqual(trace_summary["template_supported_query_trace_count"], len(query_traces))
                self.assertTrue(trace_summary["all_queries_have_template_support"])
                for trace in query_traces:
                    self.assertEqual(trace["schema_version"], STATIN_LITERATURE_QUERY_TRACE_SCHEMA)
                    self.assertEqual(trace["execution_status"], "covered_by_validated_literature_search")
                    self.assertTrue(Path(trace["task_ref"]).exists(), trace)
                    self.assertTrue(Path(trace["report_ref"]).exists(), trace)
                    self.assertGreaterEqual(trace["hit_count"], 1, trace)
                    self.assertGreaterEqual(trace["validated_evidence_count"], 1, trace)
                    self.assertTrue(trace["supporting_evidence_refs"], trace)
                    self.assertTrue(trace["template_supporting_evidence_refs"], trace)
                    self.assertGreaterEqual(trace["template_supporting_evidence_count"], 1, trace)
                    self.assertIn(
                        trace["quality_gate_status"],
                        {"template_support_only", "template_support_plus_external_leads"},
                        trace,
                    )
                    self.assertTrue(trace["search_queries"], trace)
                closure_audit = payload["route_closure_audit"]
                self.assertEqual(closure_audit["schema_version"], "statin_route_closure_audit.v1")
                self.assertEqual(closure_audit["target_safe"], row["safe"])
                self.assertEqual(closure_audit["readiness_status"], "not_ready_for_solved_status")
                self.assertFalse(closure_audit["solved_claim_allowed"])
                self.assertTrue(closure_audit["not_lab_procedure"])
                self.assertTrue(closure_audit["validation"]["accepted"], closure_audit)
                self.assertGreaterEqual(len(closure_audit["passed_requirements"]), 3)
                self.assertGreaterEqual(len(closure_audit["blocking_requirements"]), 4)
                self.assertGreaterEqual(
                    len(closure_audit["automatic_followup_literature_queue"]),
                    len(closure_audit["blocking_requirements"]),
                )
                self.assertIn("## Route Closure Audit", markdown)
                route_template = payload["route_template"]
                self.assertEqual(route_template["schema_version"], STATIN_MEMBER_ROUTE_TEMPLATE_SCHEMA)
                self.assertEqual(route_template["target_safe"], row["safe"])
                self.assertTrue(route_template["not_lab_procedure"])
                self.assertEqual(route_template["promotion_scope"], "target_run_staging_only; family production requires cross-target replay")
                self.assertGreaterEqual(len(route_template["template_steps"]), 3, row)
                self.assertEqual(
                    len(route_template["template_steps"]),
                    len(payload["synthesis_stages"]),
                    row,
                )
                for step in route_template["template_steps"]:
                    self.assertEqual(step["schema_version"], STATIN_ROUTE_TEMPLATE_STEP_SCHEMA)
                    self.assertTrue(step["step_id"])
                    self.assertTrue(step["template_role"])
                    self.assertTrue(step["evidence_refs"], step)
                    self.assertTrue(step["template_sources"], step)
                    self.assertTrue(step["not_lab_procedure"])
                    self.assertIn(row["expected_reaction_class"], step["self_evo_tags"])
                    if step["requires_literature_evidence"]:
                        self.assertTrue(step["difficulty_queries"], step)
                        self.assertTrue(step["literature_trace_refs"], step)
                        for trace in step["literature_trace_refs"]:
                            self.assertEqual(trace["schema_version"], STATIN_LITERATURE_QUERY_TRACE_SCHEMA)
                            self.assertEqual(trace["execution_status"], "covered_by_validated_literature_search")
                            self.assertTrue(trace["task_ref"], trace)
                            self.assertTrue(trace["report_ref"], trace)
                            self.assertGreaterEqual(trace["validated_evidence_count"], 1, trace)
                            self.assertTrue(trace["template_supporting_evidence_refs"], trace)
                self.assertTrue(payload["primary_template_sources"], row)
                expected_source = _expected_primary_source(row["safe"])
                if expected_source:
                    self.assertIn(expected_source, payload["primary_template_sources"], row)
                if row["safe"] in SYNTHETIC_STATINS:
                    self.assertEqual(
                        set(payload["primary_template_sources"]) & _synthetic_member_specific_sources(),
                        {expected_source},
                        row,
                    )
                dossier_artifact = _typed_artifact(
                    "StatinFullflowDossier",
                    "statin_fullflow_dossier_artifact.v1",
                    f"{row['safe']}_fullflow_dossier",
                    row["safe"],
                    payload,
                )
                route_template_artifact = _typed_artifact(
                    "StatinRouteTemplate",
                    "statin_route_template_artifact.v1",
                    f"{row['safe']}_route_template",
                    row["safe"],
                    payload["route_template"],
                )
                closure_artifact = _typed_artifact(
                    "StatinRouteClosureAudit",
                    "statin_route_closure_audit_artifact.v1",
                    f"{row['safe']}_route_closure_audit",
                    row["safe"],
                    payload["route_closure_audit"],
                )
                self.assertTrue(validate_typed_artifact(dossier_artifact)["accepted"])
                self.assertTrue(validate_typed_artifact(route_template_artifact)["accepted"])
                self.assertTrue(validate_typed_artifact(closure_artifact)["accepted"])

        self.assertEqual(report["target_count"], 9)
        self.assertEqual(report["failed"], 0, report)
        self.assertTrue(all(report["hard_gates"].values()), report["hard_gates"])
        self.assertFalse(report["fullflow_overview"]["skipped"])
        self.assertEqual(report["fullflow_overview"]["target_count"], 9)
        self.assertTrue(report["fullflow_overview"]["validation"]["accepted"])
        self.assertFalse(report["route_closure_matrix"]["skipped"])
        self.assertEqual(report["route_closure_matrix"]["target_count"], 9)
        self.assertTrue(report["route_closure_matrix"]["validation"]["accepted"])
        self.assertGreaterEqual(report["route_closure_matrix"]["blocker_count"], 9 * 4)
        self.assertFalse(report["closure_lead_curation_packet"]["skipped"])
        self.assertEqual(report["closure_lead_curation_packet"]["target_count"], 9)
        self.assertTrue(report["closure_lead_curation_packet"]["validation"]["accepted"])
        self.assertEqual(
            report["closure_lead_curation_packet"]["task_count"],
            report["route_closure_matrix"]["blocker_count"],
        )
        self.assertFalse(report["closure_curation_result_set"]["skipped"])
        self.assertEqual(report["closure_curation_result_set"]["target_count"], 9)
        self.assertTrue(report["closure_curation_result_set"]["validation"]["accepted"])
        self.assertEqual(
            report["closure_curation_result_set"]["result_count"],
            report["closure_lead_curation_packet"]["task_count"],
        )
        self.assertEqual(report["closure_curation_result_set"]["template_promotion_allowed_count"], 0)
        self.assertEqual(report["typed_artifact_manifest"]["artifact_count"], 32)
        self.assertTrue(report["typed_artifact_manifest"]["validation_summary"]["accepted"])
        report_artifact = _typed_artifact(
            "StatinPanelSelfEvoReport",
            "statin_panel_self_evo_report_artifact.v1",
            "statin_panel_report",
            "statin_panel",
            report,
        )
        self.assertTrue(validate_typed_artifact(report_artifact)["accepted"])
        closure_matrix_artifact = _typed_artifact(
            "StatinRouteClosureMatrix",
            "statin_route_closure_matrix_artifact.v1",
            "statin_route_closure_matrix",
            "statin_panel",
            closure_matrix,
        )
        self.assertTrue(validate_typed_artifact(closure_matrix_artifact)["accepted"])
        curation_packet_artifact = _typed_artifact(
            "StatinClosureLeadCurationPacket",
            "statin_closure_lead_curation_packet_artifact.v1",
            "statin_closure_lead_curation_packet",
            "statin_panel",
            curation_packet,
        )
        self.assertTrue(validate_typed_artifact(curation_packet_artifact)["accepted"])
        curation_result_set_artifact = _typed_artifact(
            "StatinClosureCurationResultSet",
            "statin_closure_curation_result_set_artifact.v1",
            "statin_closure_curation_result_set",
            "statin_panel",
            curation_result_set,
        )
        self.assertTrue(validate_typed_artifact(curation_result_set_artifact)["accepted"])
        aggregation = report["self_evolution_aggregation"]
        self.assertTrue(aggregation["accepted"], aggregation)
        self.assertEqual(aggregation["production_promoted_count"], 2)
        self.assertEqual(
            set(report["self_evolution_kb"]["layers"]["production"]),
            {
                "statin_family_natural_statin_statin_semisynthesis_template",
                "statin_family_synthetic_statin_statin_side_chain_convergence_template",
            },
        )
        for row in report["targets"]:
            self.assertEqual(row["route_status"], "partial_anchor")
            self.assertTrue(row["literature_mode_entered"])
            self.assertTrue(row["expected_template_hit"], row)
            self.assertEqual(row["warnings"], [], row)
            self.assertEqual(
                set(row["observed_candidate_kinds"]),
                {"exact_fragment_retro", "forward_surrogate", "route_anchor"},
            )
            self.assertEqual(row["self_evolution"]["kb_target_layer"], "staging")
            self.assertTrue(row["self_evolution"]["production_write_blocked"])
            dossier = row["fullflow_dossier"]
            self.assertTrue(dossier["validation"]["accepted"], dossier)
            self.assertGreaterEqual(dossier["stage_count"], 3)
            self.assertGreaterEqual(dossier["blueprint_outline_count"], 3)
            self.assertGreaterEqual(dossier["difficulty_query_count"], 3)

    def test_statin_typed_artifacts_reject_bad_solved_or_missing_template_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_statin_panel_literature_self_evo(
                output_root=tmp,
                targets=["atorvastatin", "lovastatin"],
                literature_backend="local",
                query_budget=4,
            )
            dossier_path = Path(report["targets"][0]["fullflow_dossier"]["json"])
            dossier = json.loads(dossier_path.read_text(encoding="utf-8"))

        bad_dossier = {**dossier, "route_status": "solved"}
        bad_dossier_without_trace = {
            **dossier,
            "automatic_literature_escalation": {
                **dossier["automatic_literature_escalation"],
                "query_execution_traces": [],
                "query_trace_summary": {
                    **dossier["automatic_literature_escalation"]["query_trace_summary"],
                    "accepted_query_trace_count": 0,
                    "all_queries_have_validated_evidence": False,
                },
            },
        }
        bad_template = {
            **dossier["route_template"],
            "template_steps": [
                {
                    **dossier["route_template"]["template_steps"][0],
                    "difficulty_queries": [],
                    "literature_trace_refs": [],
                }
            ],
        }
        bad_closure = {
            **dossier["route_closure_audit"],
            "solved_claim_allowed": True,
            "readiness_status": "ready_for_solved_status",
        }
        bad_report = {
            **report,
            "target_count": 9,
            "failed": 1,
            "hard_gates": {**report["hard_gates"], "all_targets_have_step_level_route_templates": False},
        }
        bad_matrix = {
            "schema_version": STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA,
            "skipped": False,
            "target_count": 9,
            "blocker_count": 1,
            "queued_blocker_count": 0,
            "unresolved_blocker_count": 1,
            "solved_claim_allowed_count": 1,
            "rows": [
                {
                    "schema_version": "statin_route_closure_matrix_row.v1",
                    "target_safe": "atorvastatin",
                    "requirement_id": "full_text_route_step_audit",
                    "closure_status": "blocked",
                    "blocker": "missing full text",
                    "followup_query": "atorvastatin full text route",
                    "acceptance_signal": "full text route map",
                    "queue_present": False,
                    "trace_present": False,
                    "execution_status": "missing_followup_queue",
                    "solved_claim_allowed": True,
                    "next_action": "audit full text",
                }
            ],
        }
        bad_packet = {
            "schema_version": STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA,
            "skipped": False,
            "target_count": 9,
            "blocker_count": 1,
            "task_count": 1,
            "ready_for_curator_count": 1,
            "lead_backed_task_count": 1,
            "template_promotion_allowed_count": 1,
            "full_execution_coverage": True,
            "tasks": [
                {
                    "schema_version": "statin_closure_lead_curation_task.v1",
                    "task_id": "atorvastatin:full_text_route_step_audit",
                    "target_safe": "atorvastatin",
                    "requirement_id": "full_text_route_step_audit",
                    "blocker": "missing full text",
                    "followup_query": "atorvastatin route",
                    "execution_status": "pubmed_followup_executed_with_leads",
                    "evidence_lead_refs": ["ev_pubmed_bad"],
                    "search_sources": ["pubmed_followup_esearch"],
                    "curation_status": "pending_full_text_or_curator_audit",
                    "template_promotion_allowed": True,
                    "solved_claim_allowed": False,
                    "not_template_support": False,
                    "not_lab_procedure": True,
                    "extraction_schema": {
                        "schema_version": "statin_closure_lead_extraction_schema.v1",
                        "required_source_fields": ["pmid_or_doi"],
                        "required_route_fields": ["route_stage_ids"],
                        "forbidden_fields": [],
                    },
                    "acceptance_criteria": ["source is traceable"],
                    "rejection_rules": ["reject summary-only lead"],
                    "abstract_text": "must not be stored",
                }
            ],
        }
        bad_result_set = {
            "schema_version": STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA,
            "skipped": False,
            "target_count": 9,
            "task_count": 1,
            "result_count": 1,
            "lead_backed_result_count": 1,
            "route_relevant_result_count": 1,
            "context_guarded_result_count": 0,
            "needs_better_lead_count": 0,
            "full_text_extraction_required_count": 1,
            "blocked_result_count": 0,
            "template_promotion_allowed_count": 1,
            "solved_claim_allowed_count": 0,
            "candidate_template_gate_status": "promotion_allowed",
            "production_write_blocked": False,
            "not_lab_procedure": True,
            "results": [
                {
                    "schema_version": "statin_closure_curation_result.v1",
                    "result_id": "atorvastatin:full_text_route_step_audit:curation_result",
                    "task_id": "atorvastatin:full_text_route_step_audit",
                    "target_safe": "atorvastatin",
                    "requirement_id": "full_text_route_step_audit",
                    "curation_result_status": "awaiting_full_text_route_extraction",
                    "candidate_template_gate_status": "promotion_allowed",
                    "template_promotion_allowed": True,
                    "solved_claim_allowed": False,
                    "not_template_support": False,
                    "not_lab_procedure": True,
                    "evidence_lead_refs": ["ev_pubmed_bad"],
                    "source_selection_summary": {
                        "schema_version": "statin_closure_source_selection_summary.v1",
                        "source_count": 1,
                        "selected_route_source_count": 1,
                        "context_guarded_source_count": 0,
                        "weak_or_metadata_only_source_count": 0,
                        "selected_route_source_refs": ["ev_pubmed_bad"],
                        "context_guarded_source_refs": [],
                        "weak_or_rejected_source_refs": [],
                    },
                    "required_route_fields": ["route_stage_ids"],
                    "route_field_audit": [
                        {
                            "field": "route_stage_ids",
                            "status": "verified",
                            "evidence_refs": ["ev_pubmed_bad"],
                            "resolution_required_before_promotion": False,
                        }
                    ],
                    "missing_route_field_count": 0,
                    "full_text_or_curator_record_refs": ["ev_pubmed_bad"],
                    "promotion_blockers": [],
                    "self_evo_template_candidate": {
                        "schema_version": "statin_closure_self_evo_template_candidate.v1",
                        "candidate_id": "bad_candidate",
                        "candidate_status": "promoted",
                        "allowed_layer": "production",
                        "promotion_allowed": True,
                        "production_write_blocked": False,
                        "promotion_blockers": [],
                        "not_template_support": False,
                        "not_lab_procedure": True,
                    },
                    "abstract_text": "must not be stored",
                }
            ],
        }

        dossier_result = validate_typed_artifact(_typed_artifact(
            "StatinFullflowDossier",
            "statin_fullflow_dossier_artifact.v1",
            "bad_dossier",
            "atorvastatin",
            bad_dossier,
        ))
        template_result = validate_typed_artifact(_typed_artifact(
            "StatinRouteTemplate",
            "statin_route_template_artifact.v1",
            "bad_template",
            "atorvastatin",
            bad_template,
        ))
        no_trace_result = validate_typed_artifact(_typed_artifact(
            "StatinFullflowDossier",
            "statin_fullflow_dossier_artifact.v1",
            "bad_dossier_without_trace",
            "atorvastatin",
            bad_dossier_without_trace,
        ))
        closure_result = validate_typed_artifact(_typed_artifact(
            "StatinRouteClosureAudit",
            "statin_route_closure_audit_artifact.v1",
            "bad_closure",
            "atorvastatin",
            bad_closure,
        ))
        report_result = validate_typed_artifact(_typed_artifact(
            "StatinPanelSelfEvoReport",
            "statin_panel_self_evo_report_artifact.v1",
            "bad_report",
            "statin_panel",
            bad_report,
        ))
        matrix_result = validate_typed_artifact(_typed_artifact(
            "StatinRouteClosureMatrix",
            "statin_route_closure_matrix_artifact.v1",
            "bad_matrix",
            "statin_panel",
            bad_matrix,
        ))
        packet_result = validate_typed_artifact(_typed_artifact(
            "StatinClosureLeadCurationPacket",
            "statin_closure_lead_curation_packet_artifact.v1",
            "bad_packet",
            "statin_panel",
            bad_packet,
        ))
        curation_result_set_result = validate_typed_artifact(_typed_artifact(
            "StatinClosureCurationResultSet",
            "statin_closure_curation_result_set_artifact.v1",
            "bad_result_set",
            "statin_panel",
            bad_result_set,
        ))

        self.assertFalse(dossier_result["accepted"])
        self.assertIn("dossier_must_not_claim_solved", dossier_result["reasons"])
        self.assertFalse(no_trace_result["accepted"])
        self.assertIn("missing_automatic_literature_query_traces", no_trace_result["reasons"])
        self.assertFalse(template_result["accepted"])
        self.assertIn("insufficient_route_template_steps", template_result["reasons"])
        self.assertIn("route_template_step_1_missing_difficulty_queries", template_result["reasons"])
        self.assertIn("route_template_step_1_missing_literature_trace_refs", template_result["reasons"])
        self.assertFalse(closure_result["accepted"])
        self.assertIn("route_closure_invalid_readiness_status", closure_result["reasons"])
        self.assertIn("route_closure_unproven_solved_claim_allowed", closure_result["reasons"])
        self.assertFalse(report_result["accepted"])
        self.assertIn("statin_panel_report_has_failed_targets", report_result["reasons"])
        self.assertIn("statin_panel_hard_gate_failed:all_targets_have_step_level_route_templates", report_result["reasons"])
        self.assertFalse(matrix_result["accepted"])
        self.assertIn("route_closure_matrix_not_all_blockers_queued", matrix_result["reasons"])
        self.assertIn("route_closure_matrix_allows_solved_claim", matrix_result["reasons"])
        self.assertFalse(packet_result["accepted"])
        self.assertIn("closure_lead_curation_packet_allows_template_promotion", packet_result["reasons"])
        self.assertIn("closure_lead_curation_task_1_stored_abstract_text", packet_result["reasons"])
        self.assertFalse(curation_result_set_result["accepted"])
        self.assertIn("closure_curation_result_set_allows_template_promotion", curation_result_set_result["reasons"])
        self.assertIn("closure_curation_result_1_stored_abstract_text", curation_result_set_result["reasons"])

    def test_target_subset_can_be_replayed_for_fast_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_statin_panel_literature_self_evo(
                output_root=tmp,
                targets=["atorvastatin", "lovastatin"],
                literature_backend="local",
                query_budget=4,
            )

        self.assertEqual(report["target_count"], 2)
        self.assertEqual({row["safe"] for row in report["targets"]}, {"atorvastatin", "lovastatin"})
        self.assertTrue(report["hard_gates"]["all_targets_have_expected_template"])
        self.assertTrue(report["self_evolution_aggregation"]["skipped"])
        self.assertEqual(set(report["self_evolution_kb"]["layers"]["production"]), set())
        self.assertTrue(report["fullflow_overview"]["skipped"])
        self.assertEqual(report["fullflow_overview"]["artifact_count"] if "artifact_count" in report["fullflow_overview"] else 0, 0)
        self.assertTrue(report["fullflow_overview"]["validation"]["accepted"])
        self.assertTrue(report["route_closure_matrix"]["skipped"])
        self.assertTrue(report["route_closure_matrix"]["validation"]["accepted"])
        self.assertTrue(report["closure_lead_curation_packet"]["skipped"])
        self.assertTrue(report["closure_lead_curation_packet"]["validation"]["accepted"])
        self.assertTrue(report["closure_curation_result_set"]["skipped"])
        self.assertTrue(report["closure_curation_result_set"]["validation"]["accepted"])
        self.assertTrue(report["typed_artifact_manifest"]["skipped"])
        self.assertEqual(report["typed_artifact_manifest"]["artifact_count"], 0)
        self.assertTrue(report["typed_artifact_manifest"]["validation_summary"]["accepted"])

    def test_statin_subset_can_attach_mocked_pubmed_evidence_to_difficulty_traces(self):
        esearch = {"esearchresult": {"idlist": ["98765"]}}
        esummary = {
            "result": {
                "uids": ["98765"],
                "98765": {
                    "uid": "98765",
                    "title": "Synthesis of atorvastatin intermediates",
                    "fulljournalname": "Mock PubMed Journal",
                    "pubdate": "2002",
                    "articleids": [{"idtype": "doi", "value": "10.1000/mock-statin"}],
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "cascade_planner.agent.literature_research._fetch_pubmed_json",
                side_effect=[esearch, esummary],
            ) as fetch_pubmed:
                report = run_statin_panel_literature_self_evo(
                    output_root=tmp,
                    targets=["atorvastatin"],
                    literature_backend="local_pubmed",
                    query_budget=3,
                )
            dossier_path = Path(report["targets"][0]["fullflow_dossier"]["json"])
            dossier = json.loads(dossier_path.read_text(encoding="utf-8"))

        self.assertEqual(fetch_pubmed.call_count, 2)
        escalation = dossier["automatic_literature_escalation"]
        self.assertEqual(escalation["query_trace_summary"]["backend_resolved"], "local_pubmed")
        self.assertEqual(
            escalation["query_trace_summary"]["accepted_query_trace_count"],
            len(escalation["difficulty_queries"]),
        )
        self.assertEqual(
            escalation["query_trace_summary"]["template_supported_query_trace_count"],
            len(escalation["difficulty_queries"]),
        )
        self.assertEqual(
            escalation["query_trace_summary"]["external_lead_query_trace_count"],
            len(escalation["difficulty_queries"]),
        )
        for trace in escalation["query_execution_traces"]:
            self.assertIn("pubmed_esearch", trace["search_sources"], trace)
            self.assertTrue(any(ref == "ev_pubmed_98765" for ref in trace["supporting_evidence_refs"]), trace)
            self.assertTrue(any(ref == "ev_pubmed_98765" for ref in trace["external_literature_lead_refs"]), trace)
            self.assertFalse(any(ref == "ev_pubmed_98765" for ref in trace["template_supporting_evidence_refs"]), trace)
            self.assertEqual(trace["quality_gate_status"], "template_support_plus_external_leads")

    def test_statin_subset_can_execute_route_closure_followup_pubmed_leads(self):
        esearch_main = {"esearchresult": {"idlist": ["11111"]}}
        esummary_main = {
            "result": {
                "uids": ["11111"],
                "11111": {
                    "uid": "11111",
                    "title": "Synthesis of atorvastatin intermediates",
                    "articleids": [{"idtype": "doi", "value": "10.1000/main-statin"}],
                },
            }
        }
        esearch_followup_primary_nohit = {"esearchresult": {"idlist": []}}
        esearch_followup_fallback = {"esearchresult": {"idlist": ["22222"]}}
        esummary_followup = {
            "result": {
                "uids": ["22222"],
                "22222": {
                    "uid": "22222",
                    "title": "Full text synthesis route for atorvastatin intermediates",
                    "articleids": [{"idtype": "doi", "value": "10.1000/followup-statin"}],
                },
            }
        }
        efetch_followup = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222</PMID>
      <Article>
        <Abstract>
          <AbstractText>Process chemistry for atorvastatin intermediate synthesis and salt preparation.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "cascade_planner.agent.literature_research._fetch_pubmed_json",
                side_effect=[
                    esearch_main,
                    esummary_main,
                    esearch_followup_primary_nohit,
                    esearch_followup_fallback,
                    esummary_followup,
                ],
            ) as fetch_pubmed:
                with patch(
                    "cascade_planner.agent.literature_research._fetch_pubmed_text",
                    return_value=efetch_followup,
                ) as fetch_pubmed_text:
                    report = run_statin_panel_literature_self_evo(
                        output_root=tmp,
                        targets=["atorvastatin"],
                        literature_backend="local_pubmed",
                        query_budget=2,
                        execute_closure_followups=True,
                        closure_followup_limit=1,
                    )
            dossier = json.loads(Path(report["targets"][0]["fullflow_dossier"]["json"]).read_text(encoding="utf-8"))

        self.assertEqual(fetch_pubmed.call_count, 5)
        self.assertEqual(fetch_pubmed_text.call_count, 1)
        execution = dossier["route_closure_audit"]["followup_execution"]
        self.assertEqual(execution["policy"], "pubmed_lead_search")
        self.assertTrue(execution["requested"])
        self.assertEqual(execution["executed_trace_count"], 1)
        self.assertEqual(execution["lead_trace_count"], 1)
        self.assertEqual(execution["abstract_signal_trace_count"], 1)
        self.assertFalse(execution["full_queue_requested"])
        self.assertFalse(execution["full_execution_coverage"])
        trace = execution["traces"][0]
        self.assertEqual(trace["execution_status"], "pubmed_followup_executed_with_leads")
        self.assertIn("pubmed_followup_esearch", trace["search_sources"])
        self.assertIn("pubmed_followup_efetch_abstract_signal", trace["search_sources"])
        self.assertIn("ev_pubmed_22222", trace["evidence_lead_refs"])
        self.assertEqual(trace["route_relevant_lead_source_count"], 1)
        self.assertEqual(trace["route_context_guarded_source_count"], 0)
        self.assertEqual(trace["lead_relevance_gate"], "route_relevant_strong")
        self.assertEqual(trace["lead_sources"][0]["evidence_ref"], "ev_pubmed_22222")
        self.assertEqual(trace["lead_sources"][0]["pmid"], "22222")
        self.assertEqual(trace["lead_sources"][0]["source_url"], "https://pubmed.ncbi.nlm.nih.gov/22222/")
        self.assertEqual(trace["lead_sources"][0]["source_title"], "Full text synthesis route for atorvastatin intermediates")
        self.assertEqual(trace["lead_sources"][0]["lead_relevance_status"], "route_relevant_strong")
        self.assertGreaterEqual(trace["lead_sources"][0]["route_relevance_score"], 5)
        self.assertTrue(trace["lead_sources"][0]["not_template_support"])
        self.assertEqual(trace["query_attempt_count"], 2)
        self.assertTrue(trace["fallback_used"])
        self.assertTrue(trace["resolved_query"])
        self.assertTrue(trace["abstract_signal_audit_requested"])
        self.assertEqual(trace["abstract_signal_status"], "abstract_route_signal_detected")
        self.assertGreaterEqual(trace["abstract_signal_hit_count"], 1)
        self.assertIn("synthesis", trace["abstract_signal_terms"])
        self.assertIn("process", trace["abstract_signal_terms"])
        for audit_row in trace["report"]["abstract_signal_audit"]:
            self.assertNotIn("abstract_text", audit_row)
        self.assertTrue(trace["not_template_support"])
        self.assertTrue(dossier["route_closure_audit"]["validation"]["accepted"], dossier["route_closure_audit"])

    def test_statin_subset_can_execute_all_route_closure_followups_when_limit_negative(self):
        esearch_main = {"esearchresult": {"idlist": ["11111"]}}
        esummary_main = {
            "result": {
                "uids": ["11111"],
                "11111": {
                    "uid": "11111",
                    "title": "Synthesis of atorvastatin intermediates",
                    "articleids": [{"idtype": "doi", "value": "10.1000/main-statin"}],
                },
            }
        }

        def fake_followup(**kwargs):
            fake_followup.count += 1
            pmid = str(33000 + fake_followup.count)
            return (
                [
                    SimpleNamespace(
                        evidence_id=f"ev_pubmed_{pmid}",
                        source_title="Full text synthesis route for atorvastatin intermediates",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        doi=f"10.1000/closure-{pmid}",
                        source_record_id=f"pubmed:{pmid}",
                        source_metadata={
                            "pmid": pmid,
                            "journal": "Organic Process Research and Development",
                            "pubdate": "2004",
                            "query": kwargs["query"],
                            "abstract_signal_audit": {
                                "route_signal_status": "abstract_route_signal_detected",
                                "route_signal_terms": ["synthesis", "intermediate", "process"],
                                "abstract_available": True,
                                "abstract_text_char_count": 128,
                            },
                        },
                    )
                ],
                {
                    "schema_version": "literature_followup_search_report.v1",
                    "query_variants": [kwargs["query"]],
                    "resolved_query": kwargs["query"],
                    "query_attempt_count": 1,
                    "fallback_used": False,
                    "hit_count": 1,
                    "searches": [{"source": "pubmed_followup_esearch"}],
                    "abstract_signal_audit_requested": True,
                    "abstract_signal_status": "abstract_route_signal_detected",
                    "abstract_signal_record_count": 1,
                    "abstract_signal_hit_count": 1,
                    "abstract_signal_terms": ["synthesis", "process"],
                    "abstract_signal_audit": [
                        {
                            "schema_version": "pubmed_abstract_route_signal_audit.v1",
                            "pmid": pmid,
                            "abstract_available": True,
                            "abstract_text_char_count": 128,
                            "route_signal_terms": ["synthesis", "intermediate", "process"],
                            "route_signal_count": 3,
                            "route_signal_status": "abstract_route_signal_detected",
                            "limitations": ["abstract_text_not_stored"],
                        }
                    ],
                },
            )

        fake_followup.count = 0
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "cascade_planner.agent.literature_research._fetch_pubmed_json",
                side_effect=[esearch_main, esummary_main],
            ):
                with patch(
                    "cascade_planner.agent.statin_panel.retrieve_pubmed_query_evidence",
                    side_effect=fake_followup,
                ) as followup:
                    report = run_statin_panel_literature_self_evo(
                        output_root=tmp,
                        targets=["atorvastatin"],
                        literature_backend="local_pubmed",
                        query_budget=2,
                        execute_closure_followups=True,
                        closure_followup_limit=-1,
                    )
            dossier = json.loads(Path(report["targets"][0]["fullflow_dossier"]["json"]).read_text(encoding="utf-8"))

        execution = dossier["route_closure_audit"]["followup_execution"]
        blocker_count = len(dossier["route_closure_audit"]["blocking_requirements"])
        self.assertEqual(followup.call_count, blocker_count)
        self.assertTrue(execution["full_queue_requested"])
        self.assertEqual(execution["resolved_limit"], blocker_count)
        self.assertEqual(execution["trace_count"], blocker_count)
        self.assertEqual(execution["executed_trace_count"], blocker_count)
        self.assertEqual(execution["lead_trace_count"], blocker_count)
        self.assertEqual(execution["abstract_signal_trace_count"], blocker_count)
        self.assertTrue(execution["full_trace_coverage"])
        self.assertTrue(execution["full_execution_coverage"])
        self.assertTrue(dossier["route_closure_audit"]["validation"]["accepted"], dossier["route_closure_audit"])

    def test_route_closure_followup_retries_after_context_guarded_hits(self):
        target = {item.safe: item for item in load_statin_panel_targets()}["atorvastatin"]
        row = {"literature_search_summary": {"backend_resolved": "local_pubmed"}}
        followup_queue = [
            {
                "requirement_id": "condition_and_workup_evidence_audit",
                "query": "atorvastatin synthesis conditions",
                "acceptance_signal": "condition/workup/isolation evidence tied to the member route",
            }
        ]
        clinical_card = SimpleNamespace(
            evidence_id="ev_pubmed_14531725",
            source_title="Clinical pharmacokinetics of atorvastatin.",
            url="https://pubmed.ncbi.nlm.nih.gov/14531725/",
            doi="10.2165/00003088-200342130-00005",
            source_record_id="pubmed:14531725",
            source_metadata={
                "pmid": "14531725",
                "journal": "Clinical pharmacokinetics",
                "pubdate": "2003",
                "query": "atorvastatin synthesis conditions",
                "abstract_signal_audit": {
                    "route_signal_status": "abstract_route_signal_detected",
                    "route_signal_terms": ["route", "lactone"],
                    "abstract_available": True,
                    "abstract_text_char_count": 2736,
                },
            },
        )
        route_card = SimpleNamespace(
            evidence_id="ev_pubmed_15069189",
            source_title="Development of an efficient, scalable, aldolase-catalyzed process for enantioselective synthesis of statin intermediates.",
            url="https://pubmed.ncbi.nlm.nih.gov/15069189/",
            doi="10.1021/op0341715",
            source_record_id="pubmed:15069189",
            source_metadata={
                "pmid": "15069189",
                "journal": "Organic Process Research and Development",
                "pubdate": "2004",
                "query": "atorvastatin process route",
                "abstract_signal_audit": {
                    "route_signal_status": "abstract_route_signal_detected",
                    "route_signal_terms": ["synthesis", "intermediate", "process"],
                    "abstract_available": True,
                    "abstract_text_char_count": 512,
                },
            },
        )
        clinical_report = {
            "schema_version": "literature_followup_search_report.v1",
            "query": "atorvastatin synthesis conditions",
            "query_variants": ["atorvastatin synthesis conditions"],
            "query_attempt_count": 1,
            "resolved_query": "atorvastatin synthesis conditions",
            "fallback_used": False,
            "searches": [{"source": "pubmed_followup_esearch", "hits": 1}],
            "hit_count": 1,
            "evidence_levels": {"medium": 1},
            "abstract_signal_audit_requested": True,
            "abstract_signal_status": "abstract_route_signal_detected",
            "abstract_signal_record_count": 1,
            "abstract_signal_hit_count": 1,
            "abstract_signal_terms": ["route", "lactone"],
            "abstract_signal_audit": [
                {"pmid": "14531725", "route_signal_terms": ["route", "lactone"]}
            ],
        }
        route_report = {
            "schema_version": "literature_followup_search_report.v1",
            "query": "atorvastatin process route",
            "query_variants": ["atorvastatin process route"],
            "query_attempt_count": 1,
            "resolved_query": "atorvastatin process route",
            "fallback_used": False,
            "searches": [{"source": "pubmed_followup_esearch", "hits": 1}],
            "hit_count": 1,
            "evidence_levels": {"medium": 1},
            "abstract_signal_audit_requested": True,
            "abstract_signal_status": "abstract_route_signal_detected",
            "abstract_signal_record_count": 1,
            "abstract_signal_hit_count": 1,
            "abstract_signal_terms": ["synthesis", "intermediate", "process"],
            "abstract_signal_audit": [
                {"pmid": "15069189", "route_signal_terms": ["synthesis", "intermediate", "process"]}
            ],
        }

        with patch(
            "cascade_planner.agent.statin_panel.retrieve_pubmed_query_evidence",
            side_effect=[([clinical_card], clinical_report), ([route_card], route_report)],
        ) as retrieve:
            execution = _closure_followup_execution(
                target,
                row,
                followup_queue,
                execute_followups=True,
                followup_limit=1,
            )

        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(execution["executed_trace_count"], 1)
        self.assertEqual(execution["route_relevant_trace_count"], 1)
        trace = execution["traces"][0]
        self.assertEqual(trace["lead_relevance_gate"], "route_relevant_strong", trace)
        self.assertEqual(trace["query_attempt_count"], 2, trace)
        self.assertTrue(trace["fallback_used"], trace)
        self.assertEqual(trace["route_relevant_lead_source_count"], 1, trace)
        self.assertEqual(trace["route_context_guarded_source_count"], 1, trace)
        self.assertEqual(trace["resolved_query"], "atorvastatin process route")
        self.assertIn("ev_pubmed_15069189", trace["evidence_lead_refs"], trace)
        self.assertTrue(any(row.get("source") == "statin_route_relevance_filter" for row in trace["report"]["searches"]))

    def test_cli_forwards_full_text_signal_extraction_flags(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_statin_panel_literature_self_evo.py"
        spec = importlib.util.spec_from_file_location("statin_panel_cli_under_test", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                str(script_path),
                "--output-root",
                tmp,
                "--targets",
                "atorvastatin, lovastatin",
                "--query-budget",
                "2",
                "--literature-backend",
                "local_pubmed",
                "--execute-all-closure-followups",
                "--execute-all-open-gap-searches",
                "--execute-all-full-text-access-probes",
                "--execute-full-text-signal-extractions",
                "--execute-all-full-text-signal-extractions",
            ]
            with patch.object(module.sys, "argv", argv):
                with patch.object(
                    module,
                    "run_statin_panel_literature_self_evo",
                    return_value={"schema_version": "mock_statin_panel_report.v1", "ok": True},
                ) as run_mock:
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        module.main()

        self.assertEqual(json.loads(stdout.getvalue())["ok"], True)
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["targets"], ["atorvastatin", "lovastatin"])
        self.assertEqual(kwargs["query_budget"], 2)
        self.assertEqual(kwargs["literature_backend"], "local_pubmed")
        self.assertTrue(kwargs["execute_closure_followups"])
        self.assertEqual(kwargs["closure_followup_limit"], -1)
        self.assertTrue(kwargs["execute_open_gap_searches"])
        self.assertEqual(kwargs["open_gap_search_limit"], -1)
        self.assertTrue(kwargs["execute_full_text_access_probes"])
        self.assertEqual(kwargs["full_text_access_probe_limit"], -1)
        self.assertTrue(kwargs["execute_full_text_signal_extractions"])
        self.assertEqual(kwargs["full_text_signal_extraction_limit"], -1)


def _expected_primary_source(safe: str) -> str:
    return {
        "atorvastatin": "atorvastatin_paal_knorr_convergent_assembly",
        "fluvastatin": "fluvastatin_aldol_wittig_reduction_process_window",
        "cerivastatin": "cerivastatin_pyridine_wittig_side_chain_convergence",
        "pitavastatin": "pitavastatin_quinoline_side_chain_coupling",
        "rosuvastatin": "rosuvastatin_pyrimidine_wittig_biocatalytic_side_chain",
        "lovastatin": "natural_statin_fermentation_semisynthesis",
        "mevastatin": "natural_statin_fermentation_semisynthesis",
        "pravastatin": "natural_statin_fermentation_semisynthesis",
        "simvastatin": "natural_statin_fermentation_semisynthesis",
    }.get(safe, "")


def _synthetic_member_specific_sources() -> set[str]:
    return {
        "atorvastatin_paal_knorr_convergent_assembly",
        "fluvastatin_aldol_wittig_reduction_process_window",
        "cerivastatin_pyridine_wittig_side_chain_convergence",
        "pitavastatin_quinoline_side_chain_coupling",
        "rosuvastatin_pyrimidine_wittig_biocatalytic_side_chain",
    }


def _typed_artifact(
    artifact_type: str,
    schema_version: str,
    artifact_id: str,
    case_id: str,
    payload: dict,
) -> dict:
    return {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "artifact_id": artifact_id,
        "case_id": case_id,
        "source": "unit_test",
        "input_refs": ["statin_panel_literature_workflow"],
        "evidence_refs": list(payload.get("evidence_refs") or []),
        "validation_status": "validated",
        "payload": payload,
    }


if __name__ == "__main__":
    unittest.main()
