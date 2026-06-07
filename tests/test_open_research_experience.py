import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.harness.downstream_compiler import compile_downstream_consumables
from cascade_planner.agent.chem_enzy_policy import validate_chem_enzy_search_policy
from cascade_planner.harness.open_research_experience import (
    OPEN_RESEARCH_EXPERIENCE_SCHEMA,
    OPEN_RESEARCH_MANIFEST_SCHEMA,
    audit_local_pdf_proxy_fallback,
    audit_open_research_boundary,
    build_open_research_manifest,
    extract_open_research_experience,
)
from cascade_planner.harness.local_pdf_proxy import (
    build_pdf_request,
    local_pdf_proxy_request_queue_path,
    write_pdf_request_queue,
)
from cascade_planner.harness.open_research_retrieval import (
    prefetch_open_research_evidence,
    retrieval_prefetch_manifest_entry,
    validate_retrieval_prefetch_consumption,
    write_prefetch_checkpoint_seed,
    write_retrieval_prefetch_error,
)
from cascade_planner.harness.open_research_seed_consumables import (
    build_local_downstream_seed,
    write_local_downstream_seed_artifacts,
)
from cascade_planner.harness.open_research_contract import (
    normalize_open_research_json_payload,
    validate_open_research_json_payload,
)
from cascade_planner.harness.source_detail_resolution import (
    resolve_source_detail_extraction_pack,
    source_detail_curator_records_path,
    source_detail_resolution_manifest_entry,
    source_detail_resolution_pack_path,
    write_source_detail_resolution_error,
)
from cascade_planner.harness.source_detail_chain_builder import (
    build_source_detail_curator_records_from_chain,
    compile_source_detail_chain_route,
    compile_hybrid_route_set,
    probe_literature_plugin_chain,
    resolve_curator_records_to_source_detail_steps,
)
from cascade_planner.harness.source_material_locator import (
    locate_source_materials,
    source_material_locator_manifest_entry,
    source_material_locator_pack_path,
    write_source_material_locator_error,
)
from cascade_planner.harness.visual_structure_extraction import validate_visual_structure_chain


class OpenResearchExperienceTest(unittest.TestCase):
    def test_manifest_summarizes_local_context_and_bounds_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "smiles_first_literature_workflow"
            workflow.mkdir()
            (root / "target_input.json").write_text(
                json.dumps(
                    {
                        "case_id": "fluvastatin",
                        "target_name": "fluvastatin",
                        "target_smiles": "CCO",
                        "family_hint": "synthetic statin",
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_audit.json").write_text(
                json.dumps(
                    {
                        "route_status": "unresolved",
                        "reasons": ["condition_gap"],
                        "condition_status": "condition_gap",
                    }
                ),
                encoding="utf-8",
            )
            (workflow / "literature_trigger_report.json").write_text(
                json.dumps(
                    {
                        "should_trigger": True,
                        "audit_summary": {"frontier_reasons": ["no_complexity_drop"]},
                    }
                ),
                encoding="utf-8",
            )
            (workflow / "evidence_cards.jsonl").write_text(
                json.dumps({"evidence_id": "ev1", "claim_type": "strategic_disconnection"}) + "\n",
                encoding="utf-8",
            )

            manifest = build_open_research_manifest(
                run_dir=root / "open",
                context_root=root,
                target_name="fluvastatin",
                target_smiles="CCO",
            )

        self.assertEqual(manifest["schema_version"], OPEN_RESEARCH_MANIFEST_SCHEMA)
        self.assertIn("runtime_capabilities", manifest)
        self.assertIn("case_manifest", manifest)
        self.assertIn("operation_boundary", manifest)
        self.assertIn("retrieval_prefetch", manifest)
        self.assertIn("source_material_locator", manifest)
        self.assertIn("local_pdf_proxy", manifest)
        self.assertIn("route_audit", manifest["case_manifest"])
        self.assertNotIn("chemenzy_native_raw_result", manifest["case_manifest"])
        self.assertNotIn(
            "chemenzy_native_raw_result.json",
            "\n".join(manifest["local_context"]["recommended_read_order"]),
        )
        self.assertTrue(
            any(
                "Do not read chemenzy_native_raw_result.json by default" in item
                for item in manifest["local_context"]["skip_local_rediscovery"]
            )
        )
        self.assertEqual(manifest["retrieval_prefetch"]["status"], "planned")
        self.assertEqual(manifest["source_material_locator"]["status"], "planned")
        self.assertEqual(manifest["local_pdf_proxy"]["status"], "planned")
        self.assertFalse(manifest["local_pdf_proxy"]["source_policy"]["credentials_stored"])
        self.assertEqual(manifest["local_context"]["route_audit_summary"]["reasons"], ["condition_gap"])
        self.assertIn("fluvastatin sodium", manifest["query_plan"]["pubchem_name_queries"])
        self.assertLessEqual(
            len(manifest["query_plan"]["crossref_queries"]),
            manifest["research_policy"]["source_budgets"]["crossref_queries_max"],
        )
        self.assertTrue(
            any("Google Patents" in item for item in manifest["research_policy"]["skip_or_defer"])
        )
        forbidden = manifest["operation_boundary"]["shell_policy"]["forbidden"]
        self.assertTrue(any("curl" in item for item in forbidden))
        self.assertTrue(any("pgrep" in item for item in forbidden))

    def test_local_pdf_fallback_requires_prior_agent_access_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = local_pdf_proxy_request_queue_path(root)
            request = build_pdf_request(
                {
                    "doi": "10.0000/example",
                    "title": "Example source",
                    "url": "https://doi.org/10.0000/example",
                    "content_scope": "article",
                },
                case_id="case",
                reason="agent_access_failed_pdf_needed",
                requested_by="open_agent",
            )
            write_pdf_request_queue([request], queue_path, append=False)
            (root / "evidence").mkdir(exist_ok=True)
            (root / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "case",
                        "source_relation_policy": {},
                        "sources": [],
                        "excluded_sources": [],
                        "search_log": [
                            {
                                "source": "native_web_access",
                                "doi": "10.0000/example",
                                "content_scope": "article",
                                "agent_access_status": "agent_accessible_metadata_only",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_local_pdf_proxy_fallback(run_dir=root)

        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["request_count"], 1)

    def test_local_pdf_fallback_without_agent_access_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = local_pdf_proxy_request_queue_path(root)
            request = build_pdf_request(
                {"doi": "10.0000/example", "url": "https://doi.org/10.0000/example", "content_scope": "article"},
                case_id="case",
                reason="agent_access_failed_pdf_needed",
                requested_by="open_agent",
            )
            write_pdf_request_queue([request], queue_path, append=False)
            (root / "evidence").mkdir(exist_ok=True)
            (root / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "case",
                        "source_relation_policy": {},
                        "sources": [],
                        "excluded_sources": [],
                        "search_log": [],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_local_pdf_proxy_fallback(run_dir=root)

        self.assertFalse(audit["accepted"])
        self.assertIn(
            "open_agent_boundary_violation:local_pdf_proxy:missing_agent_access_failure_record",
            audit["reasons"],
        )

    def test_local_pdf_fallback_rejects_missing_request_content_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = local_pdf_proxy_request_queue_path(root)
            request = build_pdf_request(
                {"doi": "10.0000/example", "url": "https://doi.org/10.0000/example"},
                case_id="case",
                reason="agent_access_failed_pdf_needed",
                requested_by="open_agent",
            )
            write_pdf_request_queue([request], queue_path, append=False)
            (root / "evidence").mkdir(exist_ok=True)
            (root / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "case",
                        "source_relation_policy": {},
                        "sources": [],
                        "excluded_sources": [],
                        "search_log": [
                            {
                                "source": "native_web_access",
                                "doi": "10.0000/example",
                                "content_scope": "article",
                                "agent_access_status": "agent_accessible_metadata_only",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_local_pdf_proxy_fallback(run_dir=root)

        self.assertFalse(audit["accepted"])
        self.assertIn(
            "open_agent_boundary_violation:local_pdf_proxy:missing_pdf_request_content_scope",
            audit["reasons"],
        )

    def test_local_pdf_fallback_distinguishes_article_and_si_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_path = local_pdf_proxy_request_queue_path(root)
            request = build_pdf_request(
                {
                    "doi": "10.0000/example",
                    "url": "https://publisher.example/article",
                    "content_scope": "si",
                    "source_ref": "example_si",
                },
                case_id="case",
                reason="agent_access_failed_pdf_needed",
                requested_by="open_agent",
            )
            write_pdf_request_queue([request], queue_path, append=False)
            (root / "evidence").mkdir(exist_ok=True)
            (root / "evidence" / "literature_sources.json").write_text(
                json.dumps(
                    {
                        "schema_version": "open_literature_sources.v1",
                        "case_id": "case",
                        "source_relation_policy": {},
                        "sources": [
                            {
                                "doi": "10.0000/example",
                                "url": "https://publisher.example/article",
                                "content_scope": "article",
                                "agent_access_status": "agent_accessible_full_text",
                            }
                        ],
                        "excluded_sources": [],
                        "search_log": [
                            {
                                "doi": "10.0000/example",
                                "url": "https://publisher.example/article",
                                "content_scope": "si",
                                "agent_access_status": "agent_access_blocked_login_or_paywall",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_local_pdf_proxy_fallback(run_dir=root)

        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["request_count"], 1)

    def test_manifest_uses_clean_search_name_for_run_id_like_target_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "smiles_first_literature_workflow").mkdir()
            (root / "target_input.json").write_text(
                json.dumps(
                    {
                        "case_id": "atorvastatin_latest_small_stock_depth20_real",
                        "target_name": "atorvastatin_latest_small_stock_depth20_real",
                        "target_smiles": "CCO",
                        "family_hint": "statin synthetic atorvastatin",
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_open_research_manifest(
                run_dir=root / "open",
                context_root=root,
                target_name="atorvastatin_latest_small_stock_depth20_real",
                target_smiles="CCO",
            )

        self.assertEqual(manifest["target"]["name"], "atorvastatin_latest_small_stock_depth20_real")
        self.assertEqual(manifest["target"]["search_name"], "atorvastatin")
        self.assertIn("atorvastatin synthesis", manifest["query_plan"]["crossref_queries"])
        self.assertNotIn(
            "atorvastatin_latest_small_stock_depth20_real synthesis",
            manifest["query_plan"]["crossref_queries"],
        )

    def test_manifest_loads_self_evo_memory_next_to_prior_experience(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "smiles_first_literature_workflow"
            workflow.mkdir()
            memory = {
                "schema_version": "self_evo_reusable_memory.v1",
                "accepted": True,
                "reusable_template_cards": [
                    {
                        "template_id": "statin_side_chain_template",
                        "reaction_class": "statin_side_chain_convergence",
                    }
                ],
                "reusable_one_step_rows": [],
                "reusable_route_expansion_tasks": [],
                "future_use_policy": {
                    "not_route_evidence_until_current_target_relation_checked": True,
                    "no_solved_claim": True,
                },
            }
            (root / "open_research_experience.json").write_text(
                json.dumps(
                    {
                        "schema_version": OPEN_RESEARCH_EXPERIENCE_SCHEMA,
                        "suggested_policy_updates": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "self_evo_memory.json").write_text(json.dumps(memory), encoding="utf-8")

            manifest = build_open_research_manifest(
                run_dir=root / "next_open",
                context_root=root,
                target_name="rosuvastatin",
                target_smiles="CCO",
                experience_path=root / "open_research_experience.json",
            )

        loaded = manifest["prior_experience"]["self_evo_memory"]
        self.assertEqual(loaded["schema_version"], "self_evo_reusable_memory.v1")
        self.assertTrue(loaded["future_use_policy"]["not_route_evidence_until_current_target_relation_checked"])
        self.assertEqual(loaded["reusable_template_cards"][0]["template_id"], "statin_side_chain_template")

    def test_manifest_turns_self_evo_extraction_tasks_into_typed_lookup_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "smiles_first_literature_workflow"
            workflow.mkdir()
            (root / "open_research_experience.json").write_text(
                json.dumps(
                    {
                        "schema_version": OPEN_RESEARCH_EXPERIENCE_SCHEMA,
                        "suggested_policy_updates": [],
                    }
                ),
                encoding="utf-8",
            )
            memory = {
                "schema_version": "self_evo_reusable_memory.v1",
                "accepted": True,
                "reusable_template_cards": [],
                "reusable_one_step_rows": [],
                "reusable_route_expansion_tasks": [],
                "reusable_executable_template_extraction_tasks": [
                    {
                        "schema_version": "compiled_executable_template_extraction_task.v1",
                        "task_id": "extract_rosuvastatin_side_chain",
                        "source_title": "Rosuvastatin pyrimidine-core olefination and biocatalytic side-chain strategy",
                        "reaction_class": "synthetic_statin",
                        "evidence_refs": ["ev_rosuvastatin_side_chain"],
                        "required_structured_fields": ["product_smiles", "reactant_smiles"],
                        "precursor_roles": ["pyrimidine aldehyde", "phosphonium ylide side-chain equivalent"],
                        "not_raw_reaction_injection": True,
                    }
                ],
                "future_use_policy": {
                    "not_route_evidence_until_current_target_relation_checked": True,
                    "no_solved_claim": True,
                },
            }
            (root / "self_evo_memory.json").write_text(json.dumps(memory), encoding="utf-8")

            manifest = build_open_research_manifest(
                run_dir=root / "next_open",
                context_root=root,
                target_name="rosuvastatin",
                target_smiles="CCO",
                experience_path=root / "open_research_experience.json",
            )

        query_plan = manifest["query_plan"]
        self.assertEqual(query_plan["self_evo_extraction_task_count"], 1)
        self.assertTrue(query_plan["lookup_requests"])
        self.assertTrue(any(row["source"] == "crossref" for row in query_plan["lookup_requests"]))
        self.assertTrue(any(row["source"] == "patent_metadata" for row in query_plan["lookup_requests"]))
        self.assertTrue(any(row["source"] == "web_search_metadata" for row in query_plan["lookup_requests"]))
        first = query_plan["lookup_requests"][0]
        self.assertEqual(first["origin"], "self_evo_executable_template_extraction_task")
        self.assertEqual(first["task_id"], "extract_rosuvastatin_side_chain")
        self.assertIn("product_smiles", first["required_structured_fields"])
        self.assertTrue(any("pyrimidine aldehyde" in row["query"] for row in query_plan["lookup_requests"]))

    def test_manifest_loads_route_failure_feedback_into_query_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "smiles_first_literature_workflow"
            workflow.mkdir()
            feedback = {
                "schema_version": "route_failure_feedback.v1",
                "accepted": True,
                "source_route_status": "fake_closed_rejected",
                "source_reasons": ["hidden_nonstock_reactants"],
                "terminal_blacklist": [{"canonical_smiles": "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1"}],
                "frontier_research_targets": [
                    {
                        "canonical_smiles": "COC(=O)CCO",
                        "required_action": "find_upstream_synthesis_or_disconnection",
                    }
                ],
                "query_hints": [
                    {
                        "schema_version": "route_failure_query_hint.v1",
                        "query": "rosuvastatin synthesis intermediate COC(=O)CCO",
                        "smiles": "COC(=O)CCO",
                    }
                ],
                "next_guided_policy_patch": {
                    "terminal_blacklist": ["CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1"],
                    "preferred_subgoals": ["COC(=O)CCO"],
                },
            }
            (root / "open_research_experience.json").write_text(
                json.dumps(
                    {
                        "schema_version": OPEN_RESEARCH_EXPERIENCE_SCHEMA,
                        "suggested_policy_updates": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "route_failure_feedback.json").write_text(json.dumps(feedback), encoding="utf-8")

            manifest = build_open_research_manifest(
                run_dir=root / "next_open",
                context_root=root,
                target_name="rosuvastatin",
                target_smiles="CCO",
                experience_path=root / "open_research_experience.json",
            )

        loaded = manifest["prior_experience"]["route_failure_feedback"]
        self.assertEqual(loaded["schema_version"], "route_failure_feedback.v1")
        self.assertEqual(
            manifest["local_context"]["route_failure_feedback_summary"]["frontier_research_target_count"],
            1,
        )
        self.assertIn("COC(=O)CCO", manifest["query_plan"]["route_failure_frontier_smiles"])
        self.assertIn(
            "rosuvastatin synthesis intermediate COC(=O)CCO",
            manifest["query_plan"]["route_failure_feedback_queries"],
        )
        self.assertIn(
            "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
            manifest["query_plan"]["route_failure_terminal_blacklist"],
        )

    def test_retrieval_prefetch_uses_typed_connectors_and_rejects_broad_pubmed(self):
        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            if "pubchem.ncbi.nlm.nih.gov" in url and "/cids/" in url:
                return {"IdentifierList": {"CID": [446155]}}
            if "pubchem.ncbi.nlm.nih.gov" in url and "/property/" in url:
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
                return {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.0000/example",
                                "title": ["Synthesis of ethanol"],
                                "container-title": ["Journal"],
                                "URL": "https://doi.org/10.0000/example",
                                "score": 42,
                            }
                        ]
                    }
                }
            if "esearch.fcgi" in url:
                return {"esearchresult": {"count": "1194", "idlist": ["1"]}}
            raise AssertionError(url)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "target": {"name": "ethanol", "smiles": "CCO"},
                "query_plan": {
                    "pubchem_name_queries": ["ethanol"],
                    "crossref_queries": ["ethanol synthesis"],
                    "pubmed_terms": ['"ethanol" synthesis'],
                    "patent_metadata_queries": ["ethanol process"],
                    "live_web_search_gap_queries": ["ethanol synthesis DOI"],
                },
            }

            prefetch = prefetch_open_research_evidence(
                manifest,
                output_dir=root,
                fetch_json=fake_fetch,
                max_results=2,
            )
            entry = retrieval_prefetch_manifest_entry(prefetch, output_dir=root)
            persisted = json.loads((root / "evidence" / "harness_retrieval_prefetch.json").read_text(encoding="utf-8"))
            extraction_pack = json.loads(
                (root / "evidence" / "source_detail_extraction_pack.json").read_text(encoding="utf-8")
            )

        self.assertEqual(prefetch["schema_version"], "open_research_retrieval_prefetch.v1")
        self.assertEqual(persisted["record_counts"]["pubchem"], 1)
        self.assertEqual(prefetch["compound_seed_rows"][0]["canonical_smiles"], "CCO")
        self.assertEqual(prefetch["compound_seed_rows"][0]["isomeric_smiles"], "CCO")
        self.assertEqual(prefetch["compound_seed_rows"][0]["inchi_key"], "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
        self.assertEqual(prefetch["source_seed_rows"][0]["doi"], "10.0000/example")
        self.assertTrue(any(row["reason"] == "too_broad_pubmed_query" for row in prefetch["query_quality_flags"]))
        self.assertEqual(prefetch["records"]["patent_metadata"][0]["status"], "metadata_url_only")
        self.assertEqual(entry["record_counts"]["crossref"], 1)
        self.assertIn("triage_counts", prefetch)
        self.assertIn("structured_extraction_queue", prefetch)
        self.assertEqual(entry["source_detail_extraction_pack_schema"], "source_detail_extraction_pack.v1")
        self.assertTrue(entry["source_detail_extraction_pack_path"].endswith("source_detail_extraction_pack.json"))
        self.assertEqual(extraction_pack["schema_version"], "source_detail_extraction_pack.v1")
        self.assertTrue(extraction_pack["source_policy"]["do_not_fabricate_smiles"])
        self.assertEqual(
            extraction_pack["required_output_policy"]["preferred_exact_output"],
            "source_detail_route_steps",
        )
        self.assertGreaterEqual(len(extraction_pack["queue"]), 1)

    def test_retrieval_prefetch_executes_self_evo_lookup_requests_with_task_trace(self):
        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            if "api.crossref.org" in url:
                return {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.0000/rosuvastatin",
                                "title": ["Rosuvastatin pyrimidine aldehyde intermediate synthesis"],
                                "URL": "https://doi.org/10.0000/rosuvastatin",
                            }
                        ]
                    }
                }
            raise AssertionError(url)

        request = {
            "schema_version": "typed_lookup_request.v1",
            "request_id": "extract_rosuvastatin_crossref",
            "source": "crossref",
            "query": "rosuvastatin pyrimidine aldehyde synthesis",
            "intent": "exact_intermediate",
            "expected_relation": "exact_target_or_exact_intermediate",
            "origin": "self_evo_executable_template_extraction_task",
            "task_id": "extract_rosuvastatin_side_chain",
            "extraction_task_ids": ["extract_rosuvastatin_side_chain"],
            "evidence_refs": ["ev_rosuvastatin_side_chain"],
            "required_structured_fields": ["product_smiles", "reactant_smiles"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefetch = prefetch_open_research_evidence(
                {
                    "target": {"name": "rosuvastatin", "smiles": "CCO"},
                    "query_plan": {
                        "lookup_requests": [
                            request,
                            {**request, "source": "patent_metadata", "request_id": "extract_rosuvastatin_patent"},
                            {**request, "source": "web_search_metadata", "request_id": "extract_rosuvastatin_web"},
                        ]
                    },
                },
                output_dir=root,
                fetch_json=fake_fetch,
                max_results=2,
            )
            extraction_pack = json.loads(
                (root / "evidence" / "source_detail_extraction_pack.json").read_text(encoding="utf-8")
            )

        crossref = prefetch["records"]["crossref"][0]
        patent = prefetch["records"]["patent_metadata"][0]
        web = prefetch["records"]["web_search_metadata"][0]
        self.assertEqual(crossref["lookup_request_origin"], "self_evo_executable_template_extraction_task")
        self.assertEqual(crossref["extraction_task_ids"], ["extract_rosuvastatin_side_chain"])
        self.assertEqual(crossref["evidence_refs"], ["ev_rosuvastatin_side_chain"])
        self.assertIn("product_smiles", crossref["required_structured_fields"])
        self.assertEqual(patent["lookup_request_origin"], "self_evo_executable_template_extraction_task")
        self.assertEqual(web["lookup_request_origin"], "self_evo_executable_template_extraction_task")
        self.assertGreaterEqual(len(prefetch["structured_extraction_queue"]), 1)
        self.assertFalse(any(
            row["metadata_only"] and row["action"] == "extract_structured_route_step"
            for row in prefetch["structured_extraction_queue"]
        ))
        self.assertTrue(any(
            row["source"] == "crossref" and row["action"] == "extract_structured_route_step"
            for row in prefetch["structured_extraction_queue"]
        ))
        queue_item = prefetch["structured_extraction_queue"][0]
        self.assertEqual(queue_item["extraction_task_ids"], ["extract_rosuvastatin_side_chain"])
        self.assertIn("product_smiles", queue_item["required_structured_fields"])
        self.assertTrue(queue_item["no_solved_claim"])
        pack_item = extraction_pack["queue"][0]
        self.assertEqual(pack_item["source"], "crossref")
        self.assertEqual(pack_item["extraction_task_ids"], ["extract_rosuvastatin_side_chain"])
        self.assertIn("reactant_smiles", pack_item["required_structured_fields"])
        self.assertTrue(extraction_pack["source_policy"]["metadata_only_sources_require_followup"])

    def test_retrieval_prefetch_queues_exact_synthetic_route_doi_with_search_name_alias(self):
        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            if "api.crossref.org" in url:
                return {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1021/jo00798a015",
                                "title": [
                                    "Steroids and related natural products. 78. "
                                    "Bufadienolides. 21. Synthesis of cinobufagin from bufotalin"
                                ],
                                "URL": "https://doi.org/10.1021/jo00798a015",
                            },
                            {
                                "DOI": "10.1021/jo00934a013",
                                "title": [
                                    "Steroids and related natural products. 89. "
                                    "Bufadienolides. 29. Synthetic routes to bufotalin"
                                ],
                                "URL": "https://doi.org/10.1021/jo00934a013",
                            }
                        ]
                    }
                }
            if "pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name" in url:
                return {"IdentifierList": {"CID": [12302120]}}
            if "pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid" in url:
                return {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 12302120,
                                "SMILES": "CCO",
                                "ConnectivitySMILES": "CCO",
                                "IUPACName": "bufotalin",
                                "MolecularFormula": "C26H36O6",
                                "InChIKey": "VOZHMAYHYHEWBW-NVOOAVKYSA-N",
                            }
                        ]
                    }
                }
            if "esearch.fcgi" in url:
                return {"esearchresult": {"count": "1", "idlist": ["42033427"]}}
            if "esummary.fcgi" in url:
                return {
                    "result": {
                        "uids": ["42033427"],
                        "42033427": {
                            "uid": "42033427",
                            "title": "Synthesis of Bufadienolide Natural Products and Analogs",
                            "fulljournalname": "Journal",
                            "pubdate": "2026",
                        },
                    }
                }
            raise AssertionError(url)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefetch = prefetch_open_research_evidence(
                {
                    "target": {
                        "name": "bufotalin_v0_fullflow_20260606",
                        "search_name": "bufotalin",
                        "smiles": "CCO",
                        "family_hint": "bufadienolide steroid",
                    },
                    "query_plan": {
                        "pubchem_name_queries": ["bufotalin"],
                        "crossref_queries": ["bufotalin process chemistry"],
                        "pubmed_terms": ['"bufotalin" synthesis'],
                        "patent_metadata_queries": [],
                        "live_web_search_gap_queries": [],
                    },
                },
                output_dir=root,
                fetch_json=fake_fetch,
                max_results=2,
            )
            extraction_pack = json.loads(
                (root / "evidence" / "source_detail_extraction_pack.json").read_text(encoding="utf-8")
            )

        queued = [
            row for row in prefetch["structured_extraction_queue"]
            if row.get("doi") == "10.1021/jo00934a013"
        ]
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["action"], "extract_structured_route_step")
        self.assertFalse(queued[0]["metadata_only"])
        self.assertIn("target_name_in_source_title", prefetch["source_triage"][0]["triage_reasons"])
        reverse_rows = [
            row for row in prefetch["source_triage"]
            if row.get("doi") == "10.1021/jo00798a015"
        ]
        self.assertEqual(reverse_rows[0]["recommended_action"], "log_only")
        self.assertIn("target_name_as_starting_material_direction", reverse_rows[0]["triage_reasons"])
        self.assertTrue(any(
            row.get("doi") == "10.1021/jo00934a013"
            for row in extraction_pack["queue"]
        ))
        pubmed_rows = [
            row for row in prefetch["structured_extraction_queue"]
            if row.get("source") == "pubmed"
        ]
        self.assertEqual(pubmed_rows[0]["pmid"], "42033427")

    def test_prefetch_checkpoint_seed_surfaces_source_triage_queue(self):
        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            if "api.crossref.org" in url:
                return {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.0000/fluva",
                                "title": ["Fluvastatin indole aldehyde intermediate synthesis"],
                                "URL": "https://doi.org/10.0000/fluva",
                            }
                        ]
                    }
                }
            return {"IdentifierList": {"CID": []}}

        request = {
            "schema_version": "typed_lookup_request.v1",
            "request_id": "extract_fluvastatin_crossref",
            "source": "crossref",
            "query": "fluvastatin indole aldehyde synthesis",
            "intent": "exact_intermediate",
            "expected_relation": "exact_target_or_exact_intermediate",
            "origin": "self_evo_executable_template_extraction_task",
            "task_id": "extract_fluvastatin_side_chain",
            "extraction_task_ids": ["extract_fluvastatin_side_chain"],
            "evidence_refs": ["ev_fluvastatin_side_chain"],
            "required_structured_fields": ["product_smiles", "reactant_smiles"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "target": {"name": "fluvastatin", "smiles": "CCO"},
                "query_plan": {
                    "lookup_requests": [request],
                    "prioritize_self_evo_lookup_requests": True,
                    "self_evo_lookup_request_budget": {"crossref": 1},
                },
            }
            prefetch = prefetch_open_research_evidence(
                manifest,
                output_dir=root,
                fetch_json=fake_fetch,
                max_results=1,
            )
            seed = write_prefetch_checkpoint_seed(output_dir=root, manifest=manifest, prefetch=prefetch)
            literature = json.loads((root / "evidence" / "literature_sources.json").read_text(encoding="utf-8"))
            audit = json.loads((root / "open_agent_audit.json").read_text(encoding="utf-8"))

        self.assertTrue(seed["accepted"])
        self.assertGreaterEqual(seed["structured_extraction_queue_count"], 1)
        self.assertGreaterEqual(seed["source_detail_extraction_pack"]["queue_count"], 1)
        self.assertTrue(any(row.get("status") == "source_triage" for row in literature["search_log"]))
        self.assertGreaterEqual(audit["checks"][0]["structured_extraction_queue_count"], 1)
        self.assertGreaterEqual(audit["checks"][0]["source_detail_extraction_pack"]["queue_count"], 1)
        self.assertIn("Prioritize structured_extraction_queue", " ".join(audit["next_actions"]))

    def test_prefetch_checkpoint_seed_compacts_noisy_source_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"target": {"name": "bufotalin", "smiles": "CCO"}}
            prefetch = {
                "accepted": True,
                "all_record_count": 4,
                "source_seed_rows": [
                    {
                        "source": "crossref",
                        "record_id": "doi:10.0000/bufotalin",
                        "doi": "10.0000/bufotalin",
                        "title": "Construction of bufotalin advanced intermediate",
                        "url": "https://doi.org/10.0000/bufotalin",
                    },
                    {
                        "source": "crossref",
                        "record_id": "doi:10.0000/bufotalin",
                        "doi": "10.0000/bufotalin",
                        "title": "Construction of bufotalin advanced intermediate",
                        "url": "https://doi.org/10.0000/bufotalin",
                    },
                    {
                        "source": "crossref",
                        "record_id": "doi:10.0000/camptothecin",
                        "doi": "10.0000/camptothecin",
                        "title": "Practical total synthesis of camptothecin",
                        "url": "https://doi.org/10.0000/camptothecin",
                    },
                ],
                "source_triage": [
                    {
                        "source": "crossref",
                        "record_id": "doi:10.0000/bufotalin",
                        "doi": "10.0000/bufotalin",
                        "title": "Construction of bufotalin advanced intermediate",
                        "url": "https://doi.org/10.0000/bufotalin",
                        "priority_score": 9,
                        "source_relation_hint": "exact_target_or_exact_intermediate_candidate",
                        "recommended_action": "extract_structured_route_step",
                        "triage_reasons": ["target_name_in_source_title"],
                    },
                    {
                        "source": "crossref",
                        "record_id": "doi:10.0000/camptothecin",
                        "doi": "10.0000/camptothecin",
                        "title": "Practical total synthesis of camptothecin",
                        "url": "https://doi.org/10.0000/camptothecin",
                        "priority_score": 1,
                        "source_relation_hint": "unrelated",
                        "recommended_action": "log_only",
                        "triage_reasons": [],
                    },
                ],
                "compound_seed_rows": [],
                "query_quality_flags": [],
                "rejected_queries": [],
                "structured_extraction_queue": [],
                "source_detail_extraction_pack": {"queue_count": 0, "top_source_count": 0},
            }

            seed = write_prefetch_checkpoint_seed(output_dir=root, manifest=manifest, prefetch=prefetch)
            literature = json.loads((root / "evidence" / "literature_sources.json").read_text(encoding="utf-8"))

        self.assertTrue(seed["accepted"])
        self.assertEqual([row.get("doi") for row in literature["sources"]], ["10.0000/bufotalin"])
        self.assertTrue(any(row.get("doi") == "10.0000/camptothecin" for row in literature["excluded_sources"]))
        self.assertEqual(
            sum(1 for row in literature["search_log"] if row.get("doi") == "10.0000/bufotalin"),
            1,
        )

    def test_retrieval_prefetch_error_artifact_is_nonblocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefetch = write_retrieval_prefetch_error(
                output_dir=root,
                manifest={"target": {"name": "ethanol", "smiles": "CCO"}},
                error="RuntimeError: connector unavailable",
            )
            entry = retrieval_prefetch_manifest_entry(prefetch, output_dir=root)
            persisted = json.loads((root / "evidence" / "harness_retrieval_prefetch.json").read_text(encoding="utf-8"))
            extraction_pack = json.loads(
                (root / "evidence" / "source_detail_extraction_pack.json").read_text(encoding="utf-8")
            )

        self.assertFalse(prefetch["accepted"])
        self.assertEqual(prefetch["status"], "error")
        self.assertIn("retrieval_prefetch_error", prefetch["reasons"])
        self.assertTrue(prefetch["source_policy"]["error_is_nonblocking"])
        self.assertEqual(entry["status"], "error")
        self.assertEqual(persisted["target"]["name"], "ethanol")
        self.assertEqual(extraction_pack["schema_version"], "source_detail_extraction_pack.v1")
        self.assertEqual(extraction_pack["queue"], [])
        self.assertTrue(extraction_pack["source_policy"]["no_solved_claim"])

    def test_source_detail_resolution_extracts_only_explicit_smiles_from_pmc_xml(self):
        def fake_json(url, headers, timeout_s):
            del headers, timeout_s
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": ["12345"]}}
            if "elink.fcgi" in url:
                return {
                    "linksets": [
                        {
                            "linksetdbs": [
                                {"linkname": "pubmed_pmc", "links": ["999999"]}
                            ]
                        }
                    ]
                }
            raise AssertionError(url)

        def fake_text(url, headers, timeout_s):
            del url, headers, timeout_s
            return """
            <article><body><sec><title>Synthesis</title>
            <p>product_smiles: CCO reactant_smiles: CC.O solvent: water temperature: 25 C.</p>
            <p>product_smiles: CCN reactant_smiles: CC.N solvent: ethanol temperature: 25 C.</p>
            </sec></body></article>
            """

        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [
                {
                    "queue_id": "q1",
                    "source": "crossref",
                    "doi": "10.0000/example",
                    "record_id": "doi:10.0000/example",
                    "title": "Synthesis of ethanol",
                    "action": "extract_structured_route_step",
                    "metadata_only": False,
                    "evidence_refs": ["ev_ethanol"],
                    "required_structured_fields": ["product_smiles", "reactant_smiles"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=root,
                fetch_json=fake_json,
                fetch_text=fake_text,
                max_items=1,
            )
            persisted = json.loads(source_detail_resolution_pack_path(root).read_text(encoding="utf-8"))
            entry = source_detail_resolution_manifest_entry(resolution, output_dir=root)
            compiled = compile_downstream_consumables(
                {
                    "schema_version": "open_downstream_consumables.v1",
                    "case_id": "ethanol",
                    "planner_handoff": {
                        "next_action": "route_segment_unroll",
                        "solved": False,
                        "production_kb_promotion": False,
                    },
                    "guided_rerun_requests": [],
                    "literature_template_cards": [],
                    "literature_route_segments": [],
                    "executable_template_candidates": [],
                    "source_detail_route_steps": resolution["downstream_patch"]["source_detail_route_steps"],
                    "route_expansion_tasks": [],
                    "evolution_candidates": [],
                    "rejected_consumables": [],
                },
                target_smiles="CCO",
                case_id="ethanol",
            )

        self.assertEqual(resolution["schema_version"], "source_detail_resolution_pack.v1")
        self.assertTrue(resolution["source_policy"]["do_not_fabricate_smiles"])
        self.assertFalse(resolution["source_policy"]["full_text_content_stored"])
        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 2)
        self.assertEqual(resolution["source_detail_route_steps"][0]["source_ref"], "pmc:999999")
        self.assertEqual(resolution["source_detail_route_steps"][0]["reactant_smiles"], ["CC", "O"])
        self.assertEqual(persisted["summary"]["source_detail_route_step_count"], 2)
        self.assertEqual(entry["schema"], "source_detail_resolution_pack.v1")
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")
        self.assertEqual(len(compiled["literature_template_plugin"]["one_step_rows"]), 2)

    def test_source_detail_resolution_records_gap_for_metadata_only_or_no_pmc(self):
        def fake_json(url, headers, timeout_s):
            del headers, timeout_s
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": []}}
            raise AssertionError(url)

        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "fluvastatin", "smiles": "CCO"},
            "queue": [
                {
                    "queue_id": "metadata_q",
                    "source": "web_search_metadata",
                    "title": "fluvastatin indole aldehyde product reactant SMILES",
                    "query": "fluvastatin indole aldehyde product reactant SMILES",
                    "metadata_only": True,
                    "evidence_refs": ["ev_fluva"],
                },
                {
                    "queue_id": "doi_q",
                    "source": "crossref",
                    "doi": "10.0000/no-pubmed",
                    "record_id": "doi:10.0000/no-pubmed",
                    "title": "Fluvastatin synthesis",
                    "metadata_only": False,
                    "evidence_refs": ["ev_fluva"],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=root,
                fetch_json=fake_json,
                fetch_text=lambda url, headers, timeout_s: "",
                max_items=2,
            )
            error = write_source_detail_resolution_error(
                output_dir=root / "err",
                pack=pack,
                error="network unavailable",
            )

        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 0)
        self.assertEqual(resolution["summary"]["gap_count"], 2)
        reasons = {gap["reason"] for gap in resolution["extraction_gaps"]}
        self.assertIn("metadata_only_source_requires_followup", reasons)
        self.assertIn("doi_not_linked_to_pubmed", reasons)
        self.assertFalse(error["accepted"])
        self.assertEqual(error["source_policy"]["error_is_nonblocking"], True)
        self.assertEqual(error["summary"]["source_detail_route_step_count"], 0)

    def test_source_material_locator_records_metadata_only_si_links(self):
        def fake_json(url, headers, timeout_s):
            del headers, timeout_s
            self.assertIn("api.crossref.org/works/", url)
            return {
                "message": {
                    "DOI": "10.0000/example",
                    "title": ["Synthesis of ethanol"],
                    "URL": "https://doi.org/10.0000/example",
                    "link": [
                        {
                            "URL": "https://publisher.example/ethanol/suppl.pdf",
                            "content-type": "application/pdf",
                            "content-version": "vor",
                            "intended-application": "supplementary-material",
                        }
                    ],
                }
            }

        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [
                {
                    "queue_id": "q1",
                    "source": "crossref",
                    "doi": "10.0000/example",
                    "title": "Synthesis of ethanol",
                    "evidence_refs": ["ev_ethanol"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            located = locate_source_materials(pack, output_dir=root, fetch_json=fake_json)
            persisted = json.loads(source_material_locator_pack_path(root).read_text(encoding="utf-8"))
            entry = source_material_locator_manifest_entry(located, output_dir=root)
            error = write_source_material_locator_error(
                output_dir=root / "err",
                extraction_pack=pack,
                error="network unavailable",
            )

        self.assertEqual(located["schema_version"], "source_material_locator_pack.v1")
        self.assertTrue(located["source_policy"]["metadata_only"])
        self.assertFalse(located["source_policy"]["full_text_content_stored"])
        self.assertFalse(located["source_policy"]["supplementary_file_content_stored"])
        self.assertEqual(located["summary"]["supplementary_candidate_count"], 1)
        self.assertEqual(located["summary"]["publisher_landing_count"], 1)
        self.assertEqual(persisted["material_records"][1]["material_type"], "supplementary")
        self.assertEqual(persisted["material_records"][1]["next_structured_output"], "evidence/source_detail_curator_records.json")
        self.assertTrue(persisted["material_records"][1]["not_route_evidence_until_structured_extraction"])
        self.assertEqual(entry["schema"], "source_material_locator_pack.v1")
        self.assertFalse(error["accepted"])
        self.assertTrue(error["source_policy"]["error_is_nonblocking"])

    def test_source_detail_resolution_consumes_structured_curator_records(self):
        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [],
        }
        curator_records = {
            "schema_version": "source_detail_curator_records.v1",
            "records": [
                {
                    "schema_version": "source_detail_curator_record.v1",
                    "record_id": "curated_ethanol_step",
                    "source_ref": "doi:10.0000/curated",
                    "source_title": "Curated source detail",
                    "evidence_refs": ["ev_curated"],
                    "provenance": "manual_structured_extraction",
                    "full_text_content_stored": False,
                    "procedure_text_stored": False,
                    "steps": [
                        {
                            "step_id": "curated_ethanol_step_1",
                            "segment_id": "curated_ethanol_segment",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CC", "O"],
                            "condition_candidate": {
                                "schema_version": "condition_candidate.v1",
                                "source_type": "exact",
                                "condition_status": "evidence_backed",
                                "solvent": "water",
                                "temperature": "25 C",
                            },
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=root,
                curator_records=curator_records,
                fetch_json=lambda url, headers, timeout_s: {},
                fetch_text=lambda url, headers, timeout_s: "",
            )
            compiled = compile_downstream_consumables(
                {
                    "schema_version": "open_downstream_consumables.v1",
                    "case_id": "ethanol",
                    "planner_handoff": {
                        "next_action": "template_plugin_rerun",
                        "solved": False,
                        "production_kb_promotion": False,
                    },
                    "guided_rerun_requests": [],
                    "literature_template_cards": [],
                    "literature_route_segments": [],
                    "executable_template_candidates": [],
                    "source_detail_route_steps": resolution["downstream_patch"]["source_detail_route_steps"],
                    "route_expansion_tasks": [],
                    "evolution_candidates": [],
                    "rejected_consumables": [],
                },
                target_smiles="CCO",
                case_id="ethanol",
            )

        self.assertEqual(resolution["summary"]["curator_record_count"], 1)
        self.assertEqual(resolution["summary"]["curator_step_count"], 1)
        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 1)
        self.assertFalse(resolution["source_policy"]["full_text_content_stored"])
        self.assertFalse(resolution["source_policy"]["procedure_text_stored"])
        step = resolution["source_detail_route_steps"][0]
        self.assertEqual(step["source_ref"], "doi:10.0000/curated")
        self.assertEqual(step["applicability"]["source_detail_resolution"], "structured_curator_record")
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")
        self.assertEqual(len(compiled["literature_template_plugin"]["one_step_rows"]), 1)

    def test_source_detail_resolution_accepts_step_shaped_curator_records(self):
        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [],
        }
        curator_records = {
            "schema_version": "source_detail_curator_records.v1",
            "records": [
                {
                    "schema_version": "source_detail_route_step.v1",
                    "step_id": "pmc_step_1",
                    "segment_id": "pmc_segment",
                    "source_ref": "pmc:PMC0000001",
                    "evidence_refs": ["doi:10.0000/pmc", "pmc:PMC0000001"],
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC", "O"],
                    "condition_candidate": {
                        "reagent_candidates": ["water"],
                        "solvent_candidates": ["ethanol"],
                        "temperature_C": 25,
                        "reported_yield": "90%",
                    },
                    "full_text_content_stored": False,
                    "procedure_text_stored": False,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=Path(tmp),
                curator_records=curator_records,
                fetch_json=lambda url, headers, timeout_s: {},
                fetch_text=lambda url, headers, timeout_s: "",
            )

        self.assertEqual(resolution["summary"]["curator_step_count"], 1)
        self.assertEqual(resolution["summary"]["gap_count"], 0)
        step = resolution["source_detail_route_steps"][0]
        self.assertEqual(step["step_id"], "pmc_step_1")
        self.assertEqual(step["condition_candidate"]["reagent"], "water")
        self.assertEqual(step["condition_candidate"]["solvent"], "ethanol")

    def test_source_detail_resolution_accepts_codex_source_text_translation(self):
        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [],
        }
        curator_records = {
            "schema_version": "source_detail_curator_records.v1",
            "records": [
                {
                    "schema_version": "source_detail_curator_record.v1",
                    "record_id": "codex_translated_step",
                    "source_ref": "doi:10.0000/source-text",
                    "source_title": "Source text translated route step",
                    "evidence_refs": ["ev_source_text"],
                    "provenance": "codex_source_text_translation",
                    "source_excerpt": "Compound 3 was converted to ethanol 4.",
                    "structure_derivation": {
                        "basis": "source_name_to_smiles",
                        "source_locator": "Scheme 1, compounds 3 and 4",
                        "confidence": "medium_high",
                        "tool_checks": ["RDKit parsed product and reactant SMILES"],
                    },
                    "full_text_content_stored": False,
                    "procedure_text_stored": False,
                    "steps": [
                        {
                            "step_id": "codex_translated_step_1",
                            "segment_id": "codex_translated_segment",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CC", "O"],
                            "condition_candidate": {
                                "schema_version": "condition_candidate.v1",
                                "source_type": "exact",
                                "condition_status": "evidence_backed",
                                "solvent": "water",
                                "temperature": "25 C",
                            },
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=Path(tmp),
                curator_records=curator_records,
                fetch_json=lambda url, headers, timeout_s: {},
                fetch_text=lambda url, headers, timeout_s: "",
            )
            compiled = compile_downstream_consumables(
                {
                    "schema_version": "open_downstream_consumables.v1",
                    "case_id": "ethanol",
                    "planner_handoff": {
                        "next_action": "template_plugin_rerun",
                        "solved": False,
                        "production_kb_promotion": False,
                    },
                    "guided_rerun_requests": [],
                    "literature_template_cards": [],
                    "literature_route_segments": [],
                    "executable_template_candidates": [],
                    "source_detail_route_steps": resolution["downstream_patch"]["source_detail_route_steps"],
                    "route_expansion_tasks": [],
                    "evolution_candidates": [],
                    "rejected_consumables": [],
                },
                target_smiles="CCO",
                case_id="ethanol",
            )

        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 1)
        self.assertEqual(resolution["summary"]["gap_count"], 0)
        step = resolution["source_detail_route_steps"][0]
        self.assertEqual(step["provenance"], "codex_source_text_translation")
        self.assertEqual(step["applicability"]["source_detail_resolution"], "codex_source_text_translation")
        self.assertEqual(step["structure_derivation"]["source_locator"], "Scheme 1, compounds 3 and 4")
        self.assertEqual(step["source_excerpt"], "Compound 3 was converted to ethanol 4.")
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(len(compiled["literature_template_plugin"]["one_step_rows"]), 1)

    def test_visual_literature_chain_builds_source_detail_rows_and_plugin_probe(self):
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "acetaldehyde_chain",
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

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = validate_visual_structure_chain(
                candidate_chain,
                output_dir=root,
                target_smiles="CC=O",
            )
            curator = build_source_detail_curator_records_from_chain(validation, output_dir=root)
            resolution = resolve_curator_records_to_source_detail_steps(
                curator,
                output_dir=root,
                target_name="acetaldehyde",
                target_smiles="CC=O",
                source_ref="doi:10.0000/visual-chain",
            )
            compiled_route = compile_source_detail_chain_route(
                source_detail_steps=resolution["source_detail_route_steps"],
                output_dir=root,
                target_smiles="CC=O",
                case_id="acetaldehyde_chain",
                terminal_smiles="CC",
                terminal_name="compound 1",
            )
            plugin_probe = probe_literature_plugin_chain(
                plugin_payload=compiled_route["compiled_downstream"]["literature_template_plugin"],
                validation=validation,
                output_dir=root,
            )
            hybrid = compile_hybrid_route_set(
                output_dir=root,
                case_id="acetaldehyde_chain",
                target_smiles="CC=O",
                literature_chain_audit=compiled_route["chain_audit"],
                chemenzy_result={"routes": [{"route": ["exploratory"]}]},
                verifier_report={"accepted": False, "reasons": ["fake_closed"]},
            )

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertTrue(validation["continuity_audit"]["target_match"])
        self.assertEqual(validation["summary"]["step_count"], 2)
        self.assertEqual(len(curator["records"]), 2)
        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 2)
        self.assertTrue(compiled_route["accepted"], compiled_route["reasons"])
        self.assertEqual(compiled_route["chain_audit"]["step_count"], 2)
        self.assertTrue(compiled_route["chain_audit"]["terminal_reached"])
        self.assertTrue(plugin_probe["accepted"], plugin_probe["reasons"])
        self.assertEqual(plugin_probe["matched_count"], 2)
        self.assertTrue(hybrid["accepted"])
        self.assertEqual(hybrid["summary"]["literature_route_count"], 1)

    def test_source_detail_resolution_rejects_unsubstantiated_codex_translation(self):
        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [],
        }
        curator_records = {
            "schema_version": "source_detail_curator_records.v1",
            "records": [
                {
                    "schema_version": "source_detail_curator_record.v1",
                    "record_id": "unsupported_codex_translation",
                    "source_ref": "doi:10.0000/source-text",
                    "evidence_refs": ["ev_source_text"],
                    "provenance": "codex_source_text_translation",
                    "steps": [
                        {
                            "step_id": "unsupported_codex_translation_1",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CC", "O"],
                            "condition_candidate": {"solvent": "water"},
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=Path(tmp),
                curator_records=curator_records,
                fetch_json=lambda url, headers, timeout_s: {},
                fetch_text=lambda url, headers, timeout_s: "",
            )

        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 0)
        self.assertEqual(resolution["summary"]["gap_count"], 1)
        reason = resolution["extraction_gaps"][0]["reason"]
        self.assertIn("codex_translation_missing_structure_derivation", reason)
        self.assertIn("codex_translation_missing_source_excerpt", reason)

    def test_source_detail_resolution_rejects_unsafe_curator_records(self):
        pack = {
            "schema_version": "source_detail_extraction_pack.v1",
            "target": {"name": "ethanol", "smiles": "CCO"},
            "queue": [],
        }
        curator_records = {
            "schema_version": "source_detail_curator_records.v1",
            "records": [
                {
                    "schema_version": "source_detail_curator_record.v1",
                    "record_id": "unsafe_record",
                    "source_ref": "doi:10.0000/unsafe",
                    "evidence_refs": ["ev_unsafe"],
                    "provenance": "manual_structured_extraction",
                    "full_text_content_stored": True,
                    "steps": [
                        {
                            "step_id": "unsafe_step_1",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CC", "O"],
                            "condition_candidate": {"solvent": "water"},
                            "raw_reaction": "CC.O>>CCO",
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_detail_curator_records_path(root).parent.mkdir(parents=True)
            source_detail_curator_records_path(root).write_text(
                json.dumps(curator_records),
                encoding="utf-8",
            )
            resolution = resolve_source_detail_extraction_pack(
                pack,
                output_dir=root,
                fetch_json=lambda url, headers, timeout_s: {},
                fetch_text=lambda url, headers, timeout_s: "",
            )
            compiled = compile_downstream_consumables(
                {
                    "schema_version": "open_downstream_consumables.v1",
                    "case_id": "ethanol",
                    "planner_handoff": {
                        "next_action": "chemist_review",
                        "solved": False,
                        "production_kb_promotion": False,
                    },
                    "guided_rerun_requests": [],
                    "literature_template_cards": [],
                    "literature_route_segments": [],
                    "executable_template_candidates": [],
                    "source_detail_route_steps": resolution["downstream_patch"]["source_detail_route_steps"],
                    "route_expansion_tasks": [],
                    "evolution_candidates": [],
                    "rejected_consumables": [],
                },
                target_smiles="CCO",
                case_id="ethanol",
            )

        self.assertEqual(resolution["summary"]["source_detail_route_step_count"], 0)
        self.assertEqual(resolution["summary"]["gap_count"], 1)
        self.assertIn("curator_record_rejected", resolution["extraction_gaps"][0]["reason"])
        self.assertIn("raw_reaction_in_curator_record", resolution["extraction_gaps"][0]["reason"])
        self.assertFalse(compiled["accepted"])
        self.assertIn("no_compiled_downstream_assets", compiled["reasons"])

    def test_local_downstream_seed_consumes_source_detail_resolution_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "context" / "smiles_first_literature_workflow"
            evidence_dir.mkdir(parents=True)
            (evidence_dir / "evidence_cards.jsonl").write_text(
                json.dumps(
                    {
                        "evidence_id": "ev_source_detail",
                        "source_title": "Traceable source detail",
                        "source_metadata": {
                            "record": {
                                "retrosynthetic_move": {
                                    "reaction_class": "source_detail_demo",
                                    "suggested_precursor_roles": ["major fragment", "minor fragment"],
                                }
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            resolution_path = root / "open" / "evidence" / "source_detail_resolution_pack.json"
            resolution_path.parent.mkdir(parents=True)
            resolution_path.write_text(
                json.dumps(
                    {
                        "schema_version": "source_detail_resolution_pack.v1",
                        "accepted": True,
                        "downstream_patch": {
                            "schema_version": "source_detail_resolution_downstream_patch.v1",
                            "source_detail_route_steps": [
                                _source_detail_step("resolved_step_1", segment_id="resolved_segment"),
                                _source_detail_step("resolved_step_2", segment_id="resolved_segment"),
                            ],
                            "rejected_consumables": [
                                {
                                    "reason": "metadata_only_source_requires_followup",
                                    "source": "source_detail_resolution_pack",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "target": {"name": "ethanol", "smiles": "CCO", "frontier_smiles": "CCO"},
                "case_manifest": {"evidence_cards": str(evidence_dir / "evidence_cards.jsonl")},
                "source_detail_resolution": {"path": str(resolution_path)},
                "prior_experience": {},
            }
            seed = build_local_downstream_seed(manifest=manifest, output_dir=root / "open")
            write_local_downstream_seed_artifacts(output_dir=root / "open", seed=seed)
            compiled = compile_downstream_consumables(
                root / "open" / "downstream_consumables.json",
                target_smiles="CCO",
                case_id="ethanol",
            )

        self.assertTrue(seed["accepted"])
        downstream = seed["downstream_consumables"]
        self.assertEqual(len(downstream["source_detail_route_steps"]), 2)
        self.assertEqual(len(downstream["rejected_consumables"]), 1)
        self.assertEqual(seed["source_detail_route_step_count"], 2)
        self.assertEqual(seed["source_detail_resolution_gap_count"], 1)
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")
        self.assertEqual(len(compiled["literature_template_plugin"]["one_step_rows"]), 2)

    def test_local_downstream_seed_treats_missing_evidence_cards_path_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = build_local_downstream_seed(
                manifest={
                    "target": {"name": "atorvastatin", "smiles": "CCO", "frontier_smiles": "CCO"},
                    "case_manifest": {},
                    "prior_experience": {},
                },
                output_dir=root,
            )

        self.assertFalse(seed["accepted"])
        self.assertEqual(seed["status"], "not_applicable")
        self.assertIn("no_local_evidence_or_route_failure_feedback", seed["reasons"])

    def test_retrieval_prefetch_consumption_requires_seed_use_or_explanation(self):
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
            (root / "evidence").mkdir(exist_ok=True)
            (root / "evidence" / "literature_sources.json").write_text(
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
            (root / "evidence" / "pubchem_validated_compounds.json").write_text(
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

            rejected = validate_retrieval_prefetch_consumption(run_dir=root)

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
            accepted = validate_retrieval_prefetch_consumption(run_dir=root)

        self.assertFalse(rejected["accepted"])
        self.assertIn("retrieval_prefetch_source_seed_not_consumed_or_explained", rejected["reasons"])
        self.assertIn("retrieval_prefetch_compound_seed_not_consumed_or_explained", rejected["reasons"])
        self.assertTrue(accepted["accepted"])
        self.assertIn("10.0000/example", accepted["source_matched_tokens"])
        self.assertIn("LFQSCWFLJHTTHZ-UHFFFAOYSA-N", accepted["compound_matched_tokens"])

    def test_prefetch_checkpoint_seed_is_valid_but_not_downstream_consumable(self):
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
            manifest = {
                "target": {"name": "ethanol", "smiles": "CCO"},
                "query_plan": {
                    "pubchem_name_queries": ["ethanol"],
                    "crossref_queries": ["ethanol synthesis"],
                },
            }
            prefetch = prefetch_open_research_evidence(manifest, output_dir=root, fetch_json=fake_fetch)
            seed = write_prefetch_checkpoint_seed(output_dir=root, manifest=manifest, prefetch=prefetch)
            consumption = validate_retrieval_prefetch_consumption(run_dir=root)
            schema_reasons = []
            for name in (
                "structure_template_candidates.json",
                "downstream_consumables.json",
                "evidence/literature_sources.json",
                "evidence/pubchem_validated_compounds.json",
                "open_agent_audit.json",
            ):
                payload = json.loads((root / name).read_text(encoding="utf-8"))
                schema_reasons.extend(validate_open_research_json_payload(name=name, payload=payload))
            compiled = compile_downstream_consumables(root / "downstream_consumables.json", target_smiles="CCO")

        self.assertTrue(seed["accepted"])
        self.assertTrue(consumption["accepted"], consumption["reasons"])
        self.assertEqual(schema_reasons, [])
        self.assertFalse(compiled["accepted"])
        self.assertIn("no_compiled_downstream_assets", compiled["reasons"])

    def test_local_downstream_seed_compiles_guided_template_and_self_evo_assets(self):
        context = Path("results/shared/fluvastatin_open_agent_prefetch_checkpoint_fullrun_20260606/context")
        if not context.exists():
            context = Path("results/shared/fluvastatin_codex_entry_fullflow_prompt_v2_20260605")
        if not (context / "smiles_first_literature_workflow" / "evidence_cards.jsonl").exists():
            self.skipTest("fluvastatin local evidence fixture is unavailable")
        feedback_path = Path("results/shared/fluvastatin_guided_selfevo_memory_probe_20260606/route_failure_feedback.json")
        feedback = json.loads(feedback_path.read_text(encoding="utf-8")) if feedback_path.exists() else {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_open_research_manifest(
                run_dir=root,
                context_root=context,
                target_name="fluvastatin",
                target_smiles="CCO",
                frontier_smiles="COC(=O)C[C@@H](O)C[C@@H](O)/C=C/c1c(-c2ccc(F)cc2)c2ccccc2n1C(C)C",
            )
            manifest["prior_experience"]["route_failure_feedback"] = feedback
            seed = build_local_downstream_seed(manifest=manifest, output_dir=root)
            persisted = write_local_downstream_seed_artifacts(output_dir=root, seed=seed)
            compiled = compile_downstream_consumables(root / "downstream_consumables.json", target_smiles="CCO")

        self.assertTrue(seed["accepted"])
        self.assertTrue(persisted["accepted"])
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertGreaterEqual(len((compiled["guided_chemenzy"] or {}).get("policy_payloads") or []), 1)
        self.assertGreaterEqual(len((compiled["route_expansion"] or {}).get("tasks") or []), 1)
        self.assertGreaterEqual(len((compiled["literature_template_plugin"] or {}).get("template_cards") or []), 1)
        self.assertEqual(len((compiled["literature_template_plugin"] or {}).get("one_step_rows") or []), 0)
        self.assertEqual(compiled["executable_template_maturity"]["status"], "needs_structured_extraction")
        self.assertGreaterEqual(compiled["executable_template_maturity"]["extraction_task_count"], 1)
        self.assertIn(
            "structured_step_extraction_required",
            compiled["executable_template_maturity"]["gap_reasons"],
        )
        self.assertGreaterEqual(seed["executable_template_extraction_task_count"], 1)
        self.assertIn(
            "executable_template_extraction_tasks",
            seed["downstream_consumables"],
        )
        self.assertGreaterEqual((compiled["self_evo"] or {}).get("staging_candidate_count") or 0, 1)

    def test_rosuvastatin_local_seed_keeps_advisory_literature_as_extraction_tasks(self):
        context = Path("results/shared/rosuvastatin_codex_entry_strict_stock_depth20_20260606")
        if not (context / "smiles_first_literature_workflow" / "evidence_cards.jsonl").exists():
            self.skipTest("rosuvastatin local evidence fixture is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_open_research_manifest(
                run_dir=root,
                context_root=context,
                target_name="rosuvastatin",
                target_smiles="COC(=O)C[C@@H](O)C[C@@H](O)/C=C/c1nc(N(C)S(C)(=O)=O)c(C(C)C)cc1-c1ccc(F)cc1",
                frontier_smiles="COC(=O)C[C@@H](O)C[C@@H](O)/C=C/c1nc(N(C)S(C)(=O)=O)c(C(C)C)cc1-c1ccc(F)cc1",
            )
            seed = build_local_downstream_seed(manifest=manifest, output_dir=root)
            write_local_downstream_seed_artifacts(output_dir=root, seed=seed)
            compiled = compile_downstream_consumables(root / "downstream_consumables.json", target_smiles="CCO")

        self.assertTrue(seed["accepted"], seed.get("reasons"))
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertGreaterEqual(seed["executable_template_extraction_task_count"], 1)
        self.assertEqual(compiled["literature_template_plugin"]["one_step_rows"], [])
        self.assertEqual(compiled["executable_template_maturity"]["status"], "needs_structured_extraction")
        first_task = compiled["executable_template_maturity"]["extraction_tasks"][0]
        self.assertEqual(first_task["schema_version"], "compiled_executable_template_extraction_task.v1")
        self.assertIn("product_smiles", first_task["required_structured_fields"])
        self.assertTrue(first_task["extraction_policy"]["do_not_fabricate_smiles"])

    def test_retrieval_prefetch_live_fluvastatin_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefetch = prefetch_open_research_evidence(
                {
                    "target": {"name": "fluvastatin", "smiles": ""},
                    "query_plan": {
                        "pubchem_name_queries": ["fluvastatin"],
                        "crossref_queries": ["fluvastatin synthesis"],
                        "pubmed_terms": ['"fluvastatin" synthesis'],
                        "patent_metadata_queries": ['"fluvastatin" process'],
                        "live_web_search_gap_queries": ["fluvastatin synthesis intermediate DOI"],
                    },
                },
                output_dir=Path(tmp),
                timeout_s=10.0,
                max_results=3,
            )

        self.assertTrue(prefetch["accepted"])
        self.assertGreaterEqual(prefetch["record_counts"]["pubchem"], 1)
        self.assertGreaterEqual(prefetch["record_counts"]["crossref"], 1)
        self.assertEqual(prefetch["record_counts"]["patent_metadata"], 1)
        self.assertEqual(prefetch["record_counts"]["web_search_metadata"], 1)
        self.assertEqual(prefetch["compound_seed_rows"][0]["inchi_key"], "FJLGEFLZQAZZCD-JUFISIKESA-N")
        self.assertTrue(prefetch["compound_seed_rows"][0]["canonical_smiles"])
        self.assertTrue(any("fluvastatin" in row["title"].lower() for row in prefetch["records"]["crossref"]))
        self.assertTrue(any(row["reason"] == "too_broad_pubmed_query" for row in prefetch["query_quality_flags"]))

    def test_experience_extracts_waste_patterns_from_timed_out_event_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = [
                {"type": "turn.started"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "rg --files .",
                        "aggregated_output": "/bin/bash: rg: command not found\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "curl https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                        "aggregated_output": '"querytranslation":"fluvastatin AND metabolism"',
                        "exit_code": 0,
                        "status": "completed",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "pkill -f 'python research_fluvastatin.py'",
                        "aggregated_output": "",
                        "exit_code": 0,
                        "status": "completed",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "sed -n '1,260p' chemenzy_native_raw_result.json",
                        "aggregated_output": "x" * 9000,
                        "exit_code": 0,
                        "status": "completed",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "web_search",
                        "query": "https://pubchem.ncbi.nlm.nih.gov/compound/1548972",
                        "action": {"type": "other"},
                    },
                },
            ]
            (run_dir / "codex_events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            run_record = {
                "error": "timeout",
                "output_validation": {
                    "reasons": [
                        "open_agent_timeout",
                        "missing_open_agent_artifact:structure_template_report.md",
                    ],
                    "event_summary": {"turn_completed": False},
                },
            }

            experience = extract_open_research_experience(run_dir=run_dir, run_record=run_record)
            persisted = json.loads((run_dir / "open_research_experience.json").read_text(encoding="utf-8"))

        self.assertEqual(experience["schema_version"], OPEN_RESEARCH_EXPERIENCE_SCHEMA)
        self.assertEqual(persisted["schema_version"], OPEN_RESEARCH_EXPERIENCE_SCHEMA)
        self.assertIn("rg_unavailable", experience["observed_inefficiencies"])
        self.assertIn("broad_pubmed_synthesis_query_noisy", experience["observed_inefficiencies"])
        self.assertIn("helper_process_management_overhead", experience["observed_inefficiencies"])
        self.assertIn("large_raw_artifact_overread", experience["observed_inefficiencies"])
        self.assertIn("direct_url_web_search_without_connector", experience["observed_inefficiencies"])
        self.assertIn("minimum_artifacts_not_checkpointed_before_optional_work", experience["observed_inefficiencies"])
        policy_ids = {row["policy_id"] for row in experience["suggested_policy_updates"]}
        self.assertIn("prefer_find_or_python_for_local_discovery", policy_ids)
        self.assertIn("minimum_artifacts_before_optional_sources", policy_ids)
        self.assertIn("use_structured_artifact_reader_not_raw_sed", policy_ids)
        self.assertIn("route_url_lookups_through_typed_connectors", policy_ids)

    def test_experience_does_not_treat_manifest_policy_text_as_process_management(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "codex_events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "python3 -m json.tool open_research_manifest.json",
                            "aggregated_output": '"forbidden": "pgrep pkill kill are forbidden"',
                            "exit_code": 0,
                            "status": "completed",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            experience = extract_open_research_experience(
                run_dir=run_dir,
                run_record={
                    "error": "timeout",
                    "output_validation": {
                        "reasons": ["open_agent_timeout"],
                        "missing_artifacts": [],
                        "event_summary": {"turn_completed": False},
                    },
                },
            )

        self.assertNotIn("helper_process_management_overhead", experience["observed_inefficiencies"])
        self.assertIn("open_agent_timeout_after_required_artifacts", experience["observed_inefficiencies"])

    def test_boundary_audit_flags_harness_owned_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": '/bin/bash -lc "pwd && rg --files . && curl https://api.crossref.org/works"',
                        "aggregated_output": "",
                        "exit_code": 0,
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "pgrep -af 'python research_fluvastatin.py'",
                        "aggregated_output": "",
                        "exit_code": 0,
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "/bin/bash -lc \"python - <<'PY'\nimport rdkit\nprint('RDKit available', rdkit.__version__)\nPY\"",
                        "aggregated_output": "RDKit available 2023.09.6\n",
                        "exit_code": 0,
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "web_search",
                        "query": "",
                        "action": {"type": "other"},
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "web_search",
                        "query": "https://pubchem.ncbi.nlm.nih.gov/compound/1548972",
                        "action": {"type": "other"},
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "sed -n '1,260p' /tmp/chemenzy_native_raw_result.json",
                        "aggregated_output": "x" * 9000,
                        "exit_code": 0,
                    },
                },
            ]
            (run_dir / "codex_events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            audit = audit_open_research_boundary(run_dir=run_dir)

        self.assertFalse(audit["accepted"])
        reasons = set(audit["reasons"])
        self.assertIn("open_agent_boundary_violation:environment_probe:pwd_command", reasons)
        self.assertIn("open_agent_boundary_violation:file_discovery:ripgrep_file_discovery", reasons)
        self.assertIn("open_agent_boundary_violation:external_http:curl_http_retrieval", reasons)
        self.assertIn("open_agent_boundary_violation:process_management:process_probe", reasons)
        self.assertIn("open_agent_boundary_violation:environment_probe:python_rdkit_capability_probe", reasons)
        self.assertIn("open_agent_boundary_violation:query_policy:empty_or_too_short_web_search_query", reasons)
        self.assertIn("open_agent_boundary_violation:context_boundary:large_raw_artifact_dump", reasons)

    def test_downstream_consumables_allow_guided_rerun_and_self_evo_drafts(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "fluvastatin",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "fluvastatin_guided_1",
                    "target": "stuck_node",
                    "evidence_refs": ["ev_fluvastatin_aldol_wittig_reduction_process_window"],
                    "preferred_subgoals": ["indole aldehyde/core", "syn-diol side-chain precursor"],
                    "terminal_blacklist_roles": ["advanced same-scaffold ester terminal"],
                }
            ],
            "literature_template_cards": [
                {
                    "template_id": "fluvastatin_indole_aldehyde_hwe",
                    "validation_status": "draft",
                    "template_level": "advisory_strategy",
                    "evidence_refs": ["ev1"],
                }
            ],
            "literature_route_segments": [
                {
                    "schema_version": "literature_route_segment_card.v1",
                    "segment_id": "fluvastatin_exact_segment",
                    "case_id": "fluvastatin",
                    "target_smiles": "CCO",
                    "evidence_refs": ["ev1"],
                    "source_title": "Traceable SI segment",
                    "validation_status": "draft",
                    "steps": [
                        {
                            "schema_version": "segment_step_candidate.v1",
                            "step_id": "seg_1",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CCO"],
                            "evidence_refs": ["ev1"],
                            "source_ref": "doi:10.0000/example-si",
                            "relation_type": "exact",
                            "applicability": {
                                "status": "passed",
                                "product_reconstruction_passed": True,
                                "reconstructed_product_smiles": "CCO",
                            },
                            "condition_candidate": {
                                "step_id": "seg_1",
                                "source_type": "exact",
                                "condition_status": "evidence_backed",
                                "solvent": "MeCN",
                                "temperature": "25 C",
                                "evidence_refs": ["ev1"],
                            },
                        },
                        {
                            "schema_version": "segment_step_candidate.v1",
                            "step_id": "seg_2",
                            "product_smiles": "CCO",
                            "reactant_smiles": ["CCO"],
                            "evidence_refs": ["ev1"],
                            "source_ref": "doi:10.0000/example-si",
                            "relation_type": "exact",
                            "applicability": {
                                "status": "passed",
                                "product_reconstruction_passed": True,
                                "reconstructed_product_smiles": "CCO",
                            },
                            "condition_candidate": {
                                "step_id": "seg_2",
                                "source_type": "exact",
                                "condition_status": "evidence_backed",
                                "solvent": "MeCN",
                                "temperature": "25 C",
                                "evidence_refs": ["ev1"],
                            },
                        },
                    ],
                }
            ],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [
                {
                    "candidate_id": "statin_side_chain_convergence_template",
                    "candidate_type": "TemplateCandidate",
                    "validation_status": "draft",
                    "target_layer": "candidate",
                    "evidence_refs": ["ev1"],
                }
            ],
            "rejected_consumables": [],
        }

        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)

        self.assertEqual(reasons, [])

    def test_downstream_normalizer_accepts_codex_shaped_drafts(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "atorvastatin",
            "planner_handoff": {
                "next_action": "validate then run guided ChemEnzy",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "guided_from_pubchem_anchor",
                    "target": "stuck_node",
                    "source_refs": ["doi:10.0000/example"],
                    "preferred_subgoals": ["CCO"],
                }
            ],
            "literature_template_cards": [
                {
                    "card_id": "atorvastatin_tbutyl_deprotection",
                    "title": "Atorvastatin tert-butyl ester deprotection seed",
                    "source_refs": ["doi:10.0000/example"],
                    "applicability": "advisory only until source-detail extraction passes",
                }
            ],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "executable_template_extraction_tasks": [
                {
                    "schema_version": "executable_template_extraction_task.v1",
                    "task_id": "extract_example",
                    "source_ref": "doi:10.0000/example",
                    "required_structured_fields": ["product_smiles", "reactant_smiles"],
                }
            ],
            "route_expansion_tasks": [
                {
                    "task_id": "expand_anchor",
                    "target": "CCO",
                    "source_refs": ["doi:10.0000/example"],
                }
            ],
            "evolution_candidates": [
                {
                    "candidate_id": "evo_template_seed",
                    "candidate_type": "template_seed",
                    "source_refs": ["doi:10.0000/example"],
                }
            ],
            "rejected_consumables": [],
        }

        normalized = normalize_open_research_json_payload(name="downstream_consumables.json", payload=payload)
        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)
        compiled = compile_downstream_consumables(normalized, target_smiles="CCO", case_id="atorvastatin")

        self.assertEqual(reasons, [])
        self.assertEqual(normalized["planner_handoff"]["next_action"], "guided_chemenzy_rerun")
        self.assertEqual(normalized["literature_template_cards"][0]["template_id"], "atorvastatin_tbutyl_deprotection")
        self.assertEqual(normalized["evolution_candidates"][0]["candidate_type"], "TemplateCandidate")
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertGreaterEqual(len(compiled["guided_chemenzy"]["policy_payloads"]), 1)
        self.assertGreaterEqual(compiled["self_evo"]["staging_candidate_count"], 1)

    def test_downstream_compiler_emits_guided_policy_template_plugin_and_self_evo_staging(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "fluvastatin",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "fluvastatin_guided_1",
                    "request_type": "literature_guided_chemenzy_rerun",
                    "target": "stuck_node",
                    "evidence_refs": ["ev1"],
                    "preferred_subgoals": ["indole aldehyde/core", "syn-diol side-chain precursor"],
                    "preferred_reaction_classes": ["statin_side_chain_convergence"],
                    "terminal_blacklist_roles": ["advanced same-scaffold ester terminal"],
                    "max_depth": 15,
                    "max_iterations": 50,
                    "expansion_topk": 100,
                }
            ],
            "literature_template_cards": [
                {
                    "schema_version": "literature_template_card.v1",
                    "template_id": "fluvastatin_indole_aldehyde_hwe",
                    "validation_status": "draft",
                    "template_level": "advisory_strategy",
                    "reaction_class": "statin_side_chain_convergence",
                    "product_retron": {"retron_type": "statin_heptenoate_side_chain"},
                    "evidence_refs": ["ev1"],
                    "not_raw_reaction_injection": True,
                }
            ],
            "literature_route_segments": [_exact_segment_payload(case_id="fluvastatin")],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [
                {
                    "candidate_id": "statin_side_chain_convergence_template",
                    "candidate_type": "TemplateCandidate",
                    "validation_status": "draft",
                    "target_layer": "candidate",
                    "evidence_refs": ["ev1"],
                    "payload": {"template_ref": "fluvastatin_indole_aldehyde_hwe"},
                }
            ],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(payload, target_smiles="CCO", case_id="fluvastatin")

        self.assertTrue(compiled["accepted"])
        self.assertEqual(len(compiled["guided_chemenzy"]["policy_payloads"]), 1)
        self.assertEqual(compiled["guided_chemenzy"]["policy_payloads"][0]["budget"]["max_depth"], 15)
        self.assertEqual(len(compiled["literature_template_plugin"]["template_cards"]), 1)
        self.assertEqual(len(compiled["literature_template_plugin"]["one_step_rows"]), 2)
        self.assertEqual(len(compiled["literature_template_plugin"]["plugin_flags"]["one_step_rows"]), 2)
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")
        self.assertEqual(compiled["executable_template_maturity"]["one_step_row_count"], 2)
        self.assertEqual(compiled["executable_template_maturity"]["extraction_task_count"], 0)
        self.assertEqual(compiled["self_evo"]["staging_candidate_count"], 1)
        self.assertTrue(compiled["self_evo"]["production_write_blocked"])

    def test_downstream_compiler_promotes_exact_source_detail_steps_to_one_step_rows(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "source_detail_case",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [
                _source_detail_step("detail_step_1", segment_id="detail_segment"),
                _source_detail_step("detail_step_2", segment_id="detail_segment"),
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(payload, target_smiles="CCO", case_id="source_detail_case")

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        rows = compiled["literature_template_plugin"]["one_step_rows"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["template"]["source"] == "literature_template_plugin" for row in rows))
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")
        self.assertEqual(compiled["executable_template_maturity"]["source_detail_route_step_count"], 2)
        self.assertTrue(any(report["kind"] == "source_detail_route_segment_draft" for report in compiled["literature_template_plugin"]["validation_reports"]))

    def test_source_detail_child_targets_prioritize_direct_target_precursor(self):
        early = _source_detail_step("early_step", segment_id="detail_segment")
        early["product_smiles"] = "CCO"
        early["reactant_smiles"] = ["CC"]
        early["applicability"]["reconstructed_product_smiles"] = "CCO"
        late = _source_detail_step("late_step", segment_id="detail_segment")
        late["product_smiles"] = "CCCC"
        late["reactant_smiles"] = ["CCC"]
        late["applicability"]["reconstructed_product_smiles"] = "CCCC"
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "source_detail_case",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [early, late],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(payload, target_smiles="CCCC", case_id="source_detail_case")

        child_targets = compiled["route_expansion"]["child_targets"]
        self.assertEqual(child_targets[0]["source_template_id"], "source_detail_exact_step:late_step")
        self.assertEqual(child_targets[0]["smiles"], "CCC")
        self.assertEqual(child_targets[0]["target_proximal_rank"], 0)

    def test_downstream_compiler_accepts_codex_source_text_translation_step(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "source_text_case",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [
                _codex_source_text_step("codex_text_step_1", segment_id="codex_text_segment"),
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)
        compiled = compile_downstream_consumables(payload, target_smiles="CCO", case_id="source_text_case")

        self.assertEqual(reasons, [])
        self.assertTrue(compiled["accepted"], compiled["reasons"])
        rows = compiled["literature_template_plugin"]["one_step_rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")

    def test_downstream_compiler_accepts_source_detail_salt_step_without_fragment_accounting(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "statin_salt_endgame_case",
            "planner_handoff": {
                "next_action": "template_plugin_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [
                {
                    "schema_version": "source_detail_route_step.v1",
                    "step_id": "source_detail_salt_step",
                    "segment_id": "statin_endgame_segment",
                    "product_smiles": "CC(=O)[O-].[Na+]",
                    "reactant_smiles": ["CC(=O)OC"],
                    "source_ref": "pmc:source-detail-salt-step",
                    "evidence_refs": ["ev_source_detail_salt"],
                    "relation_type": "exact",
                    "validation_status": "draft",
                    "applicability": {
                        "status": "passed",
                        "product_reconstruction_passed": True,
                        "reconstructed_product_smiles": "CC(=O)[O-].[Na+]",
                    },
                    "condition_candidate": {
                        "reagent_candidates": ["sodium hydroxide"],
                        "solvent_candidates": ["methanol", "water"],
                        "temperature_C": 40,
                        "source_grounding": "structured fields paraphrased from source detail",
                    },
                }
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(
            payload,
            target_smiles="CC(=O)[O-].[Na+]",
            case_id="statin_salt_endgame_case",
        )

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(compiled["reasons"], [])
        rows = compiled["literature_template_plugin"]["one_step_rows"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["reactants"], "CC(=O)OC")
        self.assertTrue(row["template"]["template_validation_report"]["allowed_for_one_step_source"])
        self.assertEqual(
            row["literature_template_trace"]["atom_accounting_policy"],
            "source_detail_exact_step_allows_reagent_or_byproduct_atoms_outside_precursor_list",
        )
        condition = row["literature_template_trace"]["condition_candidate"]
        self.assertEqual(condition["source_type"], "exact")
        self.assertEqual(condition["reagent"], "sodium hydroxide")
        self.assertEqual(condition["solvent"], "methanol; water")
        self.assertEqual(compiled["executable_template_maturity"]["status"], "executable_ready")

    def test_downstream_consumables_rejects_unsubstantiated_codex_source_text_step(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "source_text_bad",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [
                {
                    **_source_detail_step("bad_codex_text_step", segment_id="bad_codex_text_segment"),
                    "provenance": "codex_source_text_translation",
                }
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)

        self.assertIn(
            "downstream_consumables_codex_translation_missing_structure_derivation:source_detail_route_steps:0",
            reasons,
        )
        self.assertIn(
            "downstream_consumables_codex_translation_missing_source_excerpt:source_detail_route_steps:0",
            reasons,
        )

    def test_downstream_compiler_rejects_incomplete_source_detail_steps_without_fake_one_step(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "source_detail_bad",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [
                {
                    "schema_version": "source_detail_route_step.v1",
                    "step_id": "missing_structures",
                    "source_ref": "doi:10.0000/source-detail",
                    "evidence_refs": ["ev_source_detail"],
                    "validation_status": "draft",
                }
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(payload, target_smiles="CCO", case_id="source_detail_bad")

        self.assertFalse(compiled["accepted"])
        self.assertIn("missing_source_detail_fields", compiled["reasons"])
        self.assertEqual(compiled["literature_template_plugin"]["one_step_rows"], [])

    def test_downstream_compiler_turns_route_expansion_tasks_into_guided_policy_assets(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "fluvastatin",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [
                {
                    "task_id": "expand_statin_side_chain_node",
                    "task_type": "stuck_node_rerun",
                    "frontier_smiles": "CCO",
                    "target": "validated indole aldehyde subgoal",
                    "preferred_subgoals": ["syn-diol side-chain precursor"],
                    "preferred_reaction_classes": ["statin_side_chain_convergence"],
                    "preferred_disconnection_types": ["aldol_wittig_reduction_sequence"],
                    "terminal_blacklist": ["CCO"],
                    "anchor_whitelist": ["CC"],
                    "evidence_refs": ["ev1"],
                    "max_depth": 20,
                    "max_iterations": 80,
                    "expansion_topk": 120,
                    "reason": "literature segment identified expandable stuck node",
                }
            ],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(payload, target_smiles="CCO", case_id="fluvastatin")

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(len(compiled["route_expansion"]["tasks"]), 1)
        self.assertEqual(len(compiled["route_expansion"]["policy_payloads"]), 1)
        self.assertEqual(len(compiled["guided_chemenzy"]["policy_payloads"]), 1)
        policy = compiled["route_expansion"]["policy_payloads"][0]
        task = compiled["route_expansion"]["tasks"][0]
        self.assertEqual(policy["budget"]["max_depth"], 20)
        self.assertEqual(policy["budget"]["max_iterations"], 80)
        self.assertEqual(policy["budget"]["expansion_topk"], 120)
        self.assertEqual(policy["preferred_subgoal"]["route_expansion_task_id"], "expand_statin_side_chain_node")
        self.assertEqual(task["next_action"], "guided_chemenzy_rerun")
        self.assertTrue(task["production_write_blocked"])
        self.assertTrue(task["not_raw_reaction_injection"])

    def test_downstream_compiler_resolves_steroid_advisory_subgoals_to_anchor_targets(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "bufotalin",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "bufotalin_guided",
                    "target": "stuck_node",
                    "preferred_subgoals": [
                        "DHEA",
                        "pregnenolone",
                        "17-ketosteroid",
                        "C14-hydroxylated steroid",
                    ],
                    "preferred_reaction_classes": ["steroid_semisynthesis"],
                    "terminal_blacklist": ["CCO"],
                    "terminal_blacklist_roles": ["advanced_same_scaffold_terminal"],
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                }
            ],
            "literature_template_cards": [
                {
                    "schema_version": "literature_template_card.v1",
                    "template_id": "steroid_chiral_pool_template",
                    "validation_status": "draft",
                    "reaction_class": "bufadienolide_steroid",
                    "template_level": "advisory_strategy",
                    "precursor_roles": ["androstenedione", "DHEA", "17-ketosteroid"],
                    "product_retron": {
                        "retron_type": "strategic_disconnection",
                        "description": "steroid chiral-pool anchor advisory",
                    },
                    "applicability": {"status": "advisory_only", "direct_one_step_consumption": False},
                    "promotion_status": "advisory_only",
                    "condition_source": "literature",
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                    "safety_flags": ["not_raw_reaction_injection", "requires_current_target_audit"],
                    "scope_limits": ["anchor choice must be audited"],
                    "not_raw_reaction_injection": True,
                }
            ],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [
                {
                    "task_id": "bufotalin_route_expansion",
                    "task_type": "stuck_node_rerun",
                    "frontier_smiles": "CCO",
                    "preferred_subgoals": ["DHEA", "pregnenolone", "17-ketosteroid"],
                    "preferred_reaction_classes": ["steroid_semisynthesis"],
                    "terminal_blacklist": ["CCO"],
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                }
            ],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(
            payload,
            target_smiles="CCO",
            case_id="bufotalin",
            enable_online_anchor_resolution=False,
        )

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        resolution = compiled["advisory_anchor_resolution"]
        self.assertEqual(resolution["resolved_anchor_count"], 3)
        self.assertGreaterEqual(resolution["unresolved_anchor_gap_count"], 2)
        self.assertEqual(
            {row["name"] for row in resolution["resolved_anchor_targets"]},
            {"Androstenedione", "Dehydroepiandrosterone", "Pregnenolone"},
        )
        policy = compiled["guided_chemenzy"]["policy_payloads"][0]
        self.assertEqual(len(policy["anchor_whitelist"]), 3)
        self.assertEqual(policy["terminal_blacklist"], ["CCO"])
        self.assertEqual(
            len(policy["preferred_subgoal"]["resolved_advisory_anchor_targets"]),
            3,
        )
        child_targets = compiled["route_expansion"]["child_targets"]
        self.assertEqual(
            {row["name"] for row in child_targets if row["source"] == "resolved_advisory_anchor"},
            {"Androstenedione", "Dehydroepiandrosterone", "Pregnenolone"},
        )
        self.assertEqual(compiled["literature_template_plugin"]["one_step_rows"], [])

    def test_downstream_compiler_rejects_blacklist_anchor_overlap_after_advisory_resolution(self):
        dhea = "C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)CC=C4[C@@]3(CC[C@@H](C4)O)C"
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "bufotalin_overlap",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "bad_overlap",
                    "target": "stuck_node",
                    "preferred_subgoals": ["DHEA"],
                    "terminal_blacklist": [dhea],
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                }
            ],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(
            payload,
            target_smiles="CCO",
            case_id="bufotalin_overlap",
            enable_online_anchor_resolution=False,
        )

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(len(compiled["guided_chemenzy"]["policy_payloads"]), 1)
        policy = compiled["guided_chemenzy"]["policy_payloads"][0]
        self.assertEqual(policy["anchor_whitelist"], [])
        self.assertEqual(len(policy["preferred_subgoal"]["resolved_advisory_anchor_targets"]), 0)
        self.assertEqual(
            policy["preferred_subgoal"]["blocked_advisory_anchor_targets"][0]["resolution_status"],
            "blocked_by_terminal_blacklist",
        )
        self.assertEqual(compiled["advisory_anchor_resolution"]["resolved_anchor_count"], 1)
        self.assertEqual(compiled["route_expansion"]["child_targets"], [])

    def test_chem_enzy_policy_validation_rejects_terminal_blacklist_anchor_overlap(self):
        policy = {
            "schema_version": "chem_enzy_search_policy.v1",
            "policy_id": "overlap_policy",
            "operator_id": "overlap_operator",
            "case_id": "overlap_case",
            "evidence_refs": ["ev1"],
            "terminal_blacklist": ["CCO"],
            "anchor_whitelist": ["OCC"],
            "preferred_subgoal": {},
            "source_budget": {},
            "rerun_reason": "unit test overlap",
            "budget": {
                "max_reruns": 1,
                "max_iterations": 10,
                "max_depth": 5,
                "expansion_topk": 20,
            },
            "mode": "guided",
        }

        validation = validate_chem_enzy_search_policy(policy)

        self.assertFalse(validation["accepted"])
        self.assertIn("terminal_blacklist_anchor_whitelist_overlap", validation["reasons"])

    def test_downstream_compiler_online_anchor_resolution_accepts_exact_pubchem_name(self):
        calls = []

        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            calls.append(url)
            if "/cids/" in url:
                return {"IdentifierList": {"CID": [5870]}}
            if "/property/" in url:
                return {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 5870,
                                "Title": "Estrone",
                                "IsomericSMILES": (
                                    "C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)"
                                    "CCC4=C3C=CC(=C4)O"
                                ),
                                "MolecularFormula": "C18H22O2",
                                "InChIKey": "DNXHEGUUPJUMQT-CBZIJGRNSA-N",
                            }
                        ]
                    }
                }
            return {}

        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "online_anchor",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "online_anchor_guided",
                    "target": "stuck_node",
                    "preferred_subgoals": ["estrone"],
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                }
            ],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(
            payload,
            target_smiles="CCO",
            case_id="online_anchor",
            enable_online_anchor_resolution=True,
            anchor_resolution_fetch_json=fake_fetch,
        )

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        resolution = compiled["advisory_anchor_resolution"]
        self.assertTrue(resolution["online_resolution_enabled"])
        self.assertEqual(resolution["resolved_anchor_count"], 1)
        anchor = resolution["resolved_anchor_targets"][0]
        self.assertEqual(anchor["name"], "Estrone")
        self.assertEqual(anchor["source"], "live_pubchem_name_lookup")
        self.assertEqual(anchor["source_ref"], "pubchem:5870")
        self.assertEqual(len(compiled["guided_chemenzy"]["policy_payloads"][0]["anchor_whitelist"]), 1)
        self.assertEqual(len(calls), 2)

    def test_downstream_compiler_online_anchor_resolution_is_default(self):
        calls = []

        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            calls.append(url)
            if "/cids/" in url:
                return {"IdentifierList": {"CID": [5870]}}
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 5870,
                            "Title": "Estrone",
                            "IsomericSMILES": (
                                "C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)"
                                "CCC4=C3C=CC(=C4)O"
                            ),
                        }
                    ]
                }
            }

        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "online_anchor_default",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "online_anchor_default_guided",
                    "target": "stuck_node",
                    "preferred_subgoals": ["estrone"],
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                }
            ],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(
            payload,
            target_smiles="CCO",
            case_id="online_anchor_default",
            anchor_resolution_fetch_json=fake_fetch,
        )

        self.assertTrue(compiled["advisory_anchor_resolution"]["online_resolution_enabled"])
        self.assertEqual(compiled["advisory_anchor_resolution"]["resolved_anchor_targets"][0]["name"], "Estrone")
        self.assertEqual(len(calls), 2)

    def test_downstream_compiler_online_anchor_resolution_rejects_generic_terms_before_network(self):
        calls = []

        def fake_fetch(url, headers, timeout_s):
            del headers, timeout_s
            calls.append(url)
            raise AssertionError("generic advisory term should not reach PubChem")

        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "online_anchor_generic",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [
                {
                    "request_id": "online_anchor_generic_guided",
                    "target": "stuck_node",
                    "preferred_subgoals": ["17-ketosteroid", "C14-hydroxylated steroid intermediate"],
                    "evidence_refs": ["ev_steroid_chiral_pool_semisynthesis_core_policy"],
                }
            ],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(
            payload,
            target_smiles="CCO",
            case_id="online_anchor_generic",
            enable_online_anchor_resolution=True,
            anchor_resolution_fetch_json=fake_fetch,
        )

        self.assertTrue(compiled["accepted"], compiled["reasons"])
        self.assertEqual(calls, [])
        self.assertEqual(compiled["advisory_anchor_resolution"]["resolved_anchor_count"], 0)
        self.assertGreaterEqual(compiled["advisory_anchor_resolution"]["unresolved_anchor_gap_count"], 1)
        self.assertEqual(compiled["guided_chemenzy"]["policy_payloads"][0]["anchor_whitelist"], [])
        self.assertEqual(compiled["route_expansion"]["child_targets"], [])

    def test_downstream_compiler_rejects_raw_route_expansion_task_without_accepting_asset(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "bad",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "route_expansion_tasks": [
                {
                    "task_id": "bad_raw_task",
                    "evidence_refs": ["ev1"],
                    "rxn_smiles": "CCO>>CC=O",
                }
            ],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        compiled = compile_downstream_consumables(payload, target_smiles="CCO", case_id="bad")

        self.assertFalse(compiled["accepted"])
        self.assertIn("raw_reaction_injection", compiled["reasons"])
        self.assertEqual(compiled["route_expansion"]["tasks"], [])
        self.assertEqual(compiled["route_expansion"]["policy_payloads"], [])

    def test_downstream_consumables_reject_solved_production_and_raw_route_segment(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "bad",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": True,
                "production_kb_promotion": True,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [
                {
                    "segment_id": "bad_segment",
                    "validation_status": "draft",
                    "rxn_smiles": "CCO>>CC=O",
                }
            ],
            "executable_template_candidates": [],
            "route_expansion_tasks": [],
            "evolution_candidates": [
                {
                    "candidate_id": "bad_evo",
                    "candidate_type": "TemplateCandidate",
                    "validation_status": "draft",
                    "target_layer": "production",
                    "evidence_refs": ["ev1"],
                }
            ],
            "rejected_consumables": [],
        }

        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)

        self.assertIn("downstream_consumables_must_not_claim_solved", reasons)
        self.assertIn("downstream_consumables_must_not_promote_production", reasons)
        self.assertIn("downstream_consumables_raw_reaction_in_non_executable:literature_route_segments:0", reasons)
        self.assertIn("downstream_consumables_evolution_candidate_targets_production:0", reasons)

    def test_downstream_consumables_reject_raw_executable_extraction_task(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "bad",
            "planner_handoff": {
                "next_action": "guided_chemenzy_rerun",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "executable_template_extraction_tasks": [
                {
                    "schema_version": "executable_template_extraction_task.v1",
                    "task_id": "bad_task",
                    "evidence_refs": ["ev1"],
                    "reaction_smiles": "CCO>>CC=O",
                }
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)

        self.assertIn(
            "downstream_consumables_raw_reaction_in_non_executable:executable_template_extraction_tasks:0",
            reasons,
        )

    def test_downstream_consumables_reject_raw_source_detail_route_step(self):
        payload = {
            "schema_version": "open_downstream_consumables.v1",
            "case_id": "bad",
            "planner_handoff": {
                "next_action": "route_segment_unroll",
                "solved": False,
                "production_kb_promotion": False,
            },
            "guided_rerun_requests": [],
            "literature_template_cards": [],
            "literature_route_segments": [],
            "executable_template_candidates": [],
            "source_detail_route_steps": [
                {
                    "schema_version": "source_detail_route_step.v1",
                    "step_id": "bad_raw_step",
                    "source_ref": "doi:10.0000/source-detail",
                    "evidence_refs": ["ev1"],
                    "reaction_smiles": "CCO>>CC=O",
                }
            ],
            "route_expansion_tasks": [],
            "evolution_candidates": [],
            "rejected_consumables": [],
        }

        reasons = validate_open_research_json_payload(name="downstream_consumables.json", payload=payload)

        self.assertIn(
            "downstream_consumables_raw_reaction_in_non_executable:source_detail_route_steps:0",
            reasons,
        )


def _exact_segment_payload(*, case_id: str) -> dict:
    def step(step_id: str) -> dict:
        return {
            "schema_version": "segment_step_candidate.v1",
            "step_id": step_id,
            "product_smiles": "CCO",
            "reactant_smiles": ["CCO"],
            "evidence_refs": ["ev1"],
            "source_ref": "doi:10.0000/example-si",
            "relation_type": "exact",
            "applicability": {
                "status": "passed",
                "product_reconstruction_passed": True,
                "reconstructed_product_smiles": "CCO",
            },
            "condition_candidate": {
                "step_id": step_id,
                "source_type": "exact",
                "condition_status": "evidence_backed",
                "solvent": "MeCN",
                "temperature": "25 C",
                "evidence_refs": ["ev1"],
            },
        }

    return {
        "schema_version": "literature_route_segment_card.v1",
        "segment_id": f"{case_id}_exact_segment",
        "case_id": case_id,
        "target_smiles": "CCO",
        "evidence_refs": ["ev1"],
        "source_title": "Traceable SI segment",
        "validation_status": "draft",
        "steps": [step("seg_1"), step("seg_2")],
    }


def _source_detail_step(step_id: str, *, segment_id: str) -> dict:
    return {
        "schema_version": "source_detail_route_step.v1",
        "step_id": step_id,
        "segment_id": segment_id,
        "product_smiles": "CCO",
        "reactant_smiles": ["CCO"],
        "source_ref": "doi:10.0000/source-detail",
        "source_title": "Traceable source detail",
        "evidence_refs": ["ev_source_detail"],
        "relation_type": "exact",
        "validation_status": "draft",
        "applicability": {
            "status": "passed",
            "product_reconstruction_passed": True,
            "reconstructed_product_smiles": "CCO",
        },
        "condition_candidate": {
            "step_id": step_id,
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "solvent": "MeCN",
            "temperature": "25 C",
            "evidence_refs": ["ev_source_detail"],
        },
    }


def _codex_source_text_step(step_id: str, *, segment_id: str) -> dict:
    row = _source_detail_step(step_id, segment_id=segment_id)
    row.update(
        {
            "provenance": "codex_source_text_translation",
            "source_excerpt": "Compound 3 was converted to ethanol 4.",
            "structure_derivation": {
                "basis": "source_name_to_smiles",
                "source_locator": "Scheme 1, compounds 3 and 4",
                "confidence": "medium_high",
                "tool_checks": ["RDKit parsed product and reactant SMILES"],
            },
        }
    )
    return row


if __name__ == "__main__":
    unittest.main()
