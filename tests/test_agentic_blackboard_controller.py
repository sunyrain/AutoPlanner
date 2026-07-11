import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import rdFMCS

from cascade_planner.agent.codex_worker import _task_allows_cli_search
from cascade_planner.agent.chem_enzy_policy import apply_chem_enzy_search_policy, validate_chem_enzy_search_policy
from cascade_planner.agent.artifact_validators import validate_typed_artifact
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.harness.analogical_reaction_templates import (
    apply_analogical_templates_to_target,
    extract_analogical_reaction_templates_from_blackboard,
    rank_analogical_reaction_templates_from_blackboard,
    validate_analogical_reaction_template,
)
from cascade_planner.harness.agent_action_planner import (
    build_child_expansion_payload_from_blackboard,
    _child_expansion_payload,
    plan_action_batch,
    validate_action_batch,
)
from cascade_planner.harness.agentic_blackboard import (
    build_agentic_guided_payload,
    initialize_agent_blackboard,
    refresh_target_derived_blackboard_priors,
    update_blackboard_from_action_batch,
    update_blackboard_from_action,
    update_budget_for_action,
)
from cascade_planner.harness.agentic_blackboard_controller import (
    _blackboard_step_summary,
    _capability_check_planner_history,
    _capability_check_source_acquisition,
    _codex_campaign_stock_provider_results,
    _codex_scout_timeout_s,
    _codex_literature_scout_task,
    _inject_pdf_defaults,
    _local_pdf_cache_match_report,
    _merge_local_pdf_scout_report,
    _replay_codex_campaign_stock_provider_results,
    _refresh_blackboard_from_local_pdf_proxy_downloads,
    _validate_agentic_final_verdict,
    _visual_action_output_dir,
    emit_agentic_final_verdict,
    run_agentic_blackboard_controller,
)
from cascade_planner.harness.hypothetical_retrosynthesis_report import (
    compile_hypothesis_only_retrosynthesis_report,
)
from cascade_planner.harness.hypothesis_execution_report import (
    compile_hypothesis_execution_report,
)
from cascade_planner.harness.codex_action_planner import (
    _codex_action_planner_task,
    _normalize_codex_batch,
    _planner_model,
    _planner_context_summary,
    _write_codex_blackboard_snapshot,
    plan_action_batch_with_codex,
)
from cascade_planner.harness.local_pdf_proxy import (
    load_pdf_requests,
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
)
from cascade_planner.harness.open_research_experience import audit_local_pdf_proxy_fallback
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.harness.schemas import TargetInput
from cascade_planner.harness.target_side_strategy import build_target_side_disconnection_hypotheses
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    _codex_repaired_or_bounded_timeout_s,
    _guided_chemenzy_runtime_diagnostic,
    _pdf_evidence_from_payload_or_artifacts,
    _route_expansion_child_targets,
    _structure_label_matches,
    _structure_resolution_timeout_s,
    _visual_chain_image_paths,
    _visual_literature_timeout_s,
    execute_local_tool,
    run_guided_chemenzy_rerun,
)
from cascade_planner.providers.contracts import ProviderContext
from cascade_planner.providers.stock import (
    SnapshotStockProvider,
    stock_snapshot_sha256,
)
from cascade_planner.harness.visual_literature_chain_agent import (
    _candidate_chain_from_parsed,
    _candidate_quality,
    _prompt as _visual_literature_prompt,
    _run_codex_visual_prompt,
    _run_direct_visual_prompt,
    run_visual_literature_chain_agent,
)
from scripts.resume_agentic_blackboard import (
    _compact_cli_result,
    _extend_exploration_budget,
    _extend_round_budget,
    _load_budget,
    _load_existing_artifacts,
    resume_agentic_blackboard_run,
)


MLA_LIKE_SMILES = "CN1CC2CCC1CC2OC(=O)c3ccccc3N4C(=O)CCC4=O"
BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H]"
    "(CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
BUFOTALIN_ACHIRAL_SMILES = "CC(=O)OC1CC2(O)C3CCC4CC(O)CCC4(C)C3CCC2(C)C1c1ccc(=O)oc1"
C22_9OH_4HP_SMILES = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
ATORVASTATIN_FREE_ACID_SMILES = (
    "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
    "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
)
_SOURCE_FIXTURES = Path(__file__).parent / "fixtures"
_SOURCE_PDF_FIXTURE = _SOURCE_FIXTURES / "source_evidence_stub.pdf"
_SOURCE_PAGE_FIXTURE = _SOURCE_FIXTURES / "source_page.ppm"
_SOURCE_MANIFEST_FIXTURE = _SOURCE_FIXTURES / "source_evidence_manifest.json"


def _trusted_ethanol_hydration_fields() -> dict:
    template_id = "source_detail_exact_step:ethanol_hydration"
    return {
        "step_id": "ethanol_hydration",
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
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
                "manifest_sha256": hashlib.sha256(
                    _SOURCE_MANIFEST_FIXTURE.read_bytes()
                ).hexdigest(),
                "source_pdf_path": str(_SOURCE_PDF_FIXTURE.resolve()),
                "source_pdf_sha256": hashlib.sha256(
                    _SOURCE_PDF_FIXTURE.read_bytes()
                ).hexdigest(),
                "page_number": 1,
                "image_path": str(_SOURCE_PAGE_FIXTURE.resolve()),
                "image_sha256": hashlib.sha256(
                    _SOURCE_PAGE_FIXTURE.read_bytes()
                ).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


def _strict_parent_route_verifier(
    target_smiles: str,
    *,
    reactants: list[str],
    rejected_sibling: dict | None = None,
    custom_stock_path: Path | None = None,
) -> dict:
    terminals = list(dict.fromkeys(reactants))
    mapped_reaction = _test_atom_mapped_reaction(reactants, target_smiles)
    trusted_fields = (
        _trusted_ethanol_hydration_fields()
        if target_smiles == "CCO" and reactants == ["CC", "O"]
        else {}
    )
    raw = {
        "target": target_smiles,
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": terminals,
                    "terminal_stock_status": {item: True for item in terminals},
                },
                "steps": [
                    {
                        **trusted_fields,
                        "index": 0,
                        "product": target_smiles,
                        "reactant_smiles": reactants,
                        "stock_status": {item: True for item in terminals},
                        "atom_mapped_reaction_smiles": mapped_reaction,
                    }
                ],
            }
        ],
    }
    if rejected_sibling:
        raw["routes"].append(dict(rejected_sibling))
    if custom_stock_path is not None:
        raw["stock_catalog_context"] = {
            "effective_stock_names": ["controller-test-stock"],
            "catalog_bindings": [
                {
                    "name": "controller-test-stock",
                    "path": str(custom_stock_path),
                    "sha256": hashlib.sha256(custom_stock_path.read_bytes()).hexdigest(),
                }
            ],
        }
    return verify_chemenzy_raw_routes(raw, target_smiles=target_smiles)


def _test_atom_mapped_reaction(reactants: list[str], product: str) -> str:
    reactant_molecules = [Chem.MolFromSmiles(value) for value in reactants]
    product_molecule = Chem.MolFromSmiles(product)
    if product_molecule is None or any(molecule is None for molecule in reactant_molecules):
        return ""
    available: dict[int, list[int]] = {}
    product_assigned: set[int] = set()
    map_number = 1
    for molecule in reactant_molecules:
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(map_number)
            available.setdefault(atom.GetAtomicNum(), []).append(map_number)
            map_number += 1
    if len(reactant_molecules) == 1:
        mcs = rdFMCS.FindMCS(
            [reactant_molecules[0], product_molecule],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=2,
        )
        query = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
        if query is not None:
            reactant_match = reactant_molecules[0].GetSubstructMatch(query)
            product_match = product_molecule.GetSubstructMatch(query)
            for reactant_index, product_index in zip(reactant_match, product_match):
                map_value = reactant_molecules[0].GetAtomWithIdx(reactant_index).GetAtomMapNum()
                product_molecule.GetAtomWithIdx(product_index).SetAtomMapNum(map_value)
                product_assigned.add(product_index)
                available[product_molecule.GetAtomWithIdx(product_index).GetAtomicNum()].remove(
                    map_value
                )
    for atom in product_molecule.GetAtoms():
        if atom.GetIdx() in product_assigned:
            continue
        choices = available.get(atom.GetAtomicNum()) or []
        if not choices:
            return ""
        atom.SetAtomMapNum(choices.pop(0))
    left = ".".join(
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        for molecule in reactant_molecules
    )
    right = Chem.MolToSmiles(product_molecule, canonical=True, isomericSmiles=True)
    return f"{left}>>{right}"

_RENDERED_PAGE_FIXTURE = (Path(__file__).parent / "fixtures" / "source_page.ppm").resolve()
_PDF_FIXTURE = (Path(__file__).parent / "fixtures" / "source_evidence_stub.pdf").resolve()


def _rendered_pdf_evidence(
    *,
    source_ref: str,
    pdf_path: str | Path | None = None,
    document_id: str = "",
    **overrides,
) -> dict:
    """Build proof that a real page image was materialized for one document."""
    # Preserve the candidate's spelling: document keys intentionally bind to
    # the exact path string, including POSIX-style fixture paths on Windows.
    bound_pdf = str(pdf_path) if pdf_path else str(_PDF_FIXTURE)
    row = {
        "schema_version": "agent_pdf_structure_evidence_summary.v1",
        "source_ref": source_ref,
        "source_pdf_path": bound_pdf,
        "accepted": True,
        "rendered_page_count": 1,
        "rendered_pages": [
            {
                "page_number": 1,
                "image_path": str(_RENDERED_PAGE_FIXTURE),
            }
        ],
        "reasons": [],
    }
    if document_id:
        row["document_id"] = document_id
    row.update(overrides)
    return row


def _target_handles_from_blackboard(board: dict) -> set[str]:
    target_side = dict(board.get("target_side_disconnection_hypotheses") or {})
    handles = {str(item) for item in (target_side.get("target") or {}).get("handles") or [] if str(item or "").strip()}
    handles.update(
        str(row.get("target_handle") or "")
        for row in target_side.get("hypotheses") or []
        if isinstance(row, dict) and str(row.get("target_handle") or "").strip()
    )
    return handles


def _test_search_payload(query: str = "target proximal synthesis", **overrides):
    payload = {
        "schema_version": "agentic_literature_search_payload.v1",
        "search_intent": "target_proximal_source_discovery",
        "query": query,
        "queries": [query],
        "search_queries": [query],
        "max_sources": 3,
        "source_acquisition_policy": {
            "schema_version": "agentic_source_acquisition_policy.v1",
            "codex_online_first": True,
            "local_pdf_fallback_allowed": True,
            "placeholder_allowed_after_failures": True,
            "auto_local_pdf_requires_agent_discovered_metadata": True,
            "fallback_order": ["codex_online", "local_pdf", "placeholder"],
            "no_solved_claim": True,
        },
        "no_solved_claim": True,
    }
    payload.update(overrides)
    return payload


def _test_analogical_template_payload(action_type: str = "extract_analogical_reaction_templates", **overrides):
    payload = {
        "max_templates": 4,
        "max_applications": 3,
        "template_radius_policy": "auto",
        "analog_template_confidence_threshold": "low",
        "analogical_template_policy": {
            "schema_version": "agentic_analogical_template_action_policy.v1",
            "action_type": action_type,
            "analogy_is_advisory_only": True,
            "no_solved_claim": True,
            "requires_verifier": True,
            "requires_parent_route_proof": True,
            "production_write_blocked": True,
            "raw_reaction_output_allowed": False,
            "final_verdict_authority": "deterministic_parent_route_proof",
            "allowed_use": ["planner_priority", "guided_policy_hint", "template_candidate_validation"],
            "deterministic_template_validation_required": True,
        },
    }
    payload.update(overrides)
    return payload


def _test_search_requirements():
    return {
        "search_literature": {
            "currently_required_when_selected": True,
            "accepted_payload_fields": ["search_intent", "query", "queries", "search_queries", "source_acquisition_policy"],
        }
    }


def _test_source_sensitive_requirements():
    return {
        action_type: {
            "currently_required": False,
            "accepted_payload_fields": ["source_ref", "task_id", "label"],
            "binding_candidates": [],
        }
        for action_type in (
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "resolve_literature_structure_task",
            "compile_exact_literature_rows",
        )
    }


def _test_analogical_template_requirements():
    return {
        action_type: {
            "currently_required_when_selected": True,
            "accepted_payload_fields": [
                "analogical_template_policy",
                "max_templates",
                "max_applications",
                "template_radius_policy",
                "analog_template_confidence_threshold",
            ],
        }
        for action_type in (
            "extract_analogical_reaction_templates",
            "rank_analogical_reaction_templates",
            "apply_analogical_template_to_target",
            "validate_template_application",
        )
    }


def test_campaign_stock_results_include_every_current_provider_observation():
    benchmark = {
        "provider_id": "autoplanner.benchmark_catalog_stock",
        "accepted": True,
        "content_hash": "benchmark-result",
        "payload": {"canonical_smiles": "CC", "boundary_type": "benchmark_stock"},
    }
    commercial = {
        "provider_id": "autoplanner.snapshot_stock",
        "accepted": True,
        "content_hash": "commercial-result",
        "payload": {
            "canonical_smiles": "CC",
            "boundary_type": "commercially_orderable",
        },
    }
    historical = {
        "provider_id": "autoplanner.snapshot_stock",
        "accepted": False,
        "content_hash": "historical-result",
        "payload": {"canonical_smiles": "CC", "boundary_type": "unavailable"},
    }
    board = {
        "codex_agent_team": {
            "campaign": {
                "frontier_queue": {
                    "jobs": [
                        {
                            "metadata": {
                                # Compatibility alias duplicates the first
                                # current observation and must not double count.
                                "stock_audit": benchmark,
                                "stock_observations": {
                                    "current": [
                                        {"provider_result": benchmark},
                                        {"provider_result": commercial},
                                    ],
                                    "history": [{"provider_result": historical}],
                                },
                            }
                        }
                    ]
                }
            }
        }
    }

    rows = _codex_campaign_stock_provider_results(board)

    assert {row["provider_id"] for row in rows} == {
        "autoplanner.benchmark_catalog_stock",
        "autoplanner.snapshot_stock",
    }
    assert {row["content_hash"] for row in rows} == {
        "benchmark-result",
        "commercial-result",
    }


def test_campaign_stock_closure_requires_current_host_provider_replay():
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "smiles": "CCO",
        "supplier": "trusted-supplier",
        "catalog_number": "ETH-1",
        "checked_at": "2026-07-10T00:00:00Z",
        "available": True,
    }
    provider = SnapshotStockProvider(trusted_snapshots=[snapshot])
    result = provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [
                {**snapshot, "snapshot_sha256": stock_snapshot_sha256(snapshot)}
            ],
        },
        context=ProviderContext(
            run_id="stock-replay-test",
            case_id="stock-replay-test",
            target_smiles="CCO",
        ),
    ).to_dict()
    trusted = {provider.descriptor.provider_id: provider}

    accepted = _replay_codex_campaign_stock_provider_results(
        [result],
        trusted_stock_provider_instances=trusted,
    )
    assert accepted["closed_smiles"] == ["CCO"]
    assert accepted["accepted_result_count"] == 1

    forged = json.loads(json.dumps(result))
    forged["payload"]["offers"][0]["snapshot"]["supplier"] = "forged-supplier"
    envelope = dict(forged)
    envelope.pop("content_hash", None)
    forged["content_hash"] = hashlib.sha256(
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    rejected = _replay_codex_campaign_stock_provider_results(
        [forged],
        trusted_stock_provider_instances=trusted,
    )
    assert rejected["closed_smiles"] == []
    assert rejected["accepted_result_count"] == 0
    assert rejected["rejected_result_count"] == 1
    assert rejected["replays"][0]["authority_binding"] == {}

    missing_provider = _replay_codex_campaign_stock_provider_results(
        [result],
        trusted_stock_provider_instances={},
    )
    assert missing_provider["closed_smiles"] == []
    assert "trusted_provider_missing" in " ".join(
        missing_provider["replays"][0]["reasons"]
    )


class AgenticBlackboardControllerTest(unittest.TestCase):
    def test_structure_label_match_keeps_baccatin_and_10dab_distinct(self):
        self.assertTrue(_structure_label_matches("baccatin III", "baccatin III"))
        self.assertFalse(_structure_label_matches("10-deacetyl baccatin III", "10-DAB"))
        self.assertFalse(_structure_label_matches("baccatin III", "10-deacetyl baccatin III"))
        self.assertFalse(_structure_label_matches("10-deacetyl baccatin III", "baccatin III"))
        self.assertTrue(_structure_label_matches("compound 15", "15"))

    def test_blackboard_initialization_writes_target_profile(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)

        self.assertEqual(board["schema_version"], "agent_blackboard.v1")
        self.assertEqual(board["target_profile"]["target_smiles"], "CCO")
        self.assertEqual(board["budget_state"]["max_rounds"], 3)
        self.assertEqual(board["planner_history"], [])
        self.assertEqual(board["budget_state"]["codex_action_planner_runs"], 0)

    def test_blackboard_step_summary_counts_nested_literature_evidence(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["pdf_structure_evidence"] = [{"evidence_id": "pdf:first"}]
        board["literature_evidence"]["visual_chains"] = [{"chain_id": "visual:first"}]
        board["literature_evidence"]["structure_resolution_tasks"] = [{"task_id": "resolve:first"}]
        board["current_belief"]["blocked_directions"] = [{"reason": "duplicate_frontier"}]
        board["current_belief"]["next_action_bias"] = [{"action_type": "stop_unresolved"}]
        board["route_objective_summary"]["objectives"] = [{"objective_id": "objective:first"}]

        summary = _blackboard_step_summary(
            board,
            step_index=1,
            stage="unit_test",
            round_index=1,
            action_id="",
            action_type="",
            detail={},
        )

        self.assertEqual(summary["counts"]["pdf_structure_evidence"], 1)
        self.assertEqual(summary["counts"]["visual_chains"], 1)
        self.assertEqual(summary["counts"]["structure_resolution_tasks"], 1)
        self.assertEqual(summary["counts"]["blocked_directions"], 1)
        self.assertEqual(summary["counts"]["next_action_bias"], 1)
        self.assertEqual(summary["counts"]["route_objectives"], 1)

    def test_hypothesis_only_report_emits_achiral_connectivity_candidates(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["template_applications"] = [
            {
                "application_id": "apply:analog_template:test",
                "template_id": "analog_template:test",
                "evidence_refs": ["local_pdf:test"],
                "hypothetical_route_hypothesis": {
                    "reaction_center_idea": "late same-core alcohol protection or redox adjustment",
                    "template_application": "search same-core protected alcohol precursor",
                    "risk_flags": ["broad_template_scope", "selectivity_not_proven"],
                },
                "hypothetical_precursor_hints": [
                    {
                        "target_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "precursor_smiles": "CC(=O)OC1CC2(O)C3CCC4CC(O)CCC4(C)C3CCC2(C)C1C1=CC(=O)OC=C1",
                        "precursor_role": "same_core_enone_or_protected_alcohol_precursor",
                        "derived_from_retron": "steroid_alcohol_protection_redox_adjustment",
                        "risk_flags": ["hypothesis_only_not_literature_exact"],
                    }
                ],
            }
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "artifact_ref": "/tmp/visual.json",
                "source_ref": "local_pdf:test",
                "exploratory_accepted": True,
                "steps": [
                    {
                        "product_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "main_reactant_smiles": "CC12CCC(=O)C=C1CCC3C2CCC4(C3CCC4(C(=O)CO)O)C",
                        "reactant_labels": ["prednisone"],
                        "confidence": "low",
                        "stereochemistry_status": "unspecified_or_partial",
                        "risk_flags": ["visual_connectivity_approximation"],
                        "source_locator": "page 2 scheme",
                    }
                ],
            }
        ]

        report = compile_hypothesis_only_retrosynthesis_report(blackboard=board)

        self.assertTrue(report["accepted"])
        self.assertFalse(report["solved"])
        self.assertTrue(report["no_solved_claim"])
        self.assertEqual(report["final_verdict_authority"], "none")
        self.assertGreaterEqual(report["candidate_precursor_count"], 2)
        self.assertTrue(report["stereochemistry_policy"]["achiral_connectivity_candidates_allowed"])
        self.assertTrue(
            any(row["source_type"] == "visual_connectivity_candidate" for row in report["candidate_precursors"])
        )
        self.assertTrue(
            all(row["allowed_use"] == "guided_search_seed_only" for row in report["candidate_precursors"])
        )
        artifact = {
            "schema_version": "hypothesis_only_retrosynthesis_report_artifact.v1",
            "artifact_type": "HypothesisOnlyRetrosynthesisReport",
            "artifact_id": "target1_steroid:hypothesis_only_retrosynthesis_report",
            "case_id": "target1_steroid",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["local_pdf:test"],
            "validation_status": "accepted",
            "payload": report,
        }
        validation = validate_typed_artifact(artifact)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_hypothesis_report_uses_proposal_and_recursive_frontier_candidates(self):
        target = TargetInput(target_name="atorvastatin_like", target_smiles=ATORVASTATIN_FREE_ACID_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["retrosynthetic_proposals"] = [
            {
                "proposal_id": "proposal:amide_disconnection",
                "proposal_type": "semi_executable",
                "proposal_granularity": "same_core",
                "recursive_expandable": True,
                "executable": True,
                "precursor_smiles": "CC(C)c1c(C(=O)O)c(-c2ccc(F)cc2)n(CC[C@H](O)C[C@H](O)CC(=O)O)c1-c1ccccc1",
                "confidence": "medium",
                "risk_flags": ["hypothesis_only"],
            },
            {
                "proposal_id": "proposal:trivial_reagent",
                "proposal_type": "semi_executable",
                "recursive_expandable": True,
                "executable": True,
                "precursor_smiles": "[OH]",
            },
        ]
        board["recursive_hypothesis_tasks"] = [
            {
                "task_id": "recursive_hypothesis:aniline",
                "status": "pending",
                "source": "retrosynthetic_proposal",
                "precursor_smiles": "Nc1ccccc1",
                "parent_smiles": ATORVASTATIN_FREE_ACID_SMILES,
                "name": "amide_to_carboxylic_acid_amine_precursors:component:2",
                "recursive_depth": 1,
                "risk_flags": ["recursive_hypothesis_only"],
            }
        ]

        report = compile_hypothesis_only_retrosynthesis_report(blackboard=board)

        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["candidate_precursor_count"], 2)
        source_types = {row["source_type"] for row in report["candidate_precursors"]}
        self.assertIn("retrosynthetic_proposal", source_types)
        self.assertIn("recursive_hypothesis_task", source_types)
        self.assertNotIn("[OH]", {row["precursor_smiles"] for row in report["candidate_precursors"]})
        artifact = {
            "schema_version": "hypothesis_only_retrosynthesis_report_artifact.v1",
            "artifact_type": "HypothesisOnlyRetrosynthesisReport",
            "artifact_id": "atorvastatin_like:hypothesis_only_retrosynthesis_report",
            "case_id": "atorvastatin_like",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["proposal_bus"],
            "validation_status": "accepted",
            "payload": report,
        }
        validation = validate_typed_artifact(artifact)
        self.assertTrue(validation["accepted"], validation["reasons"])

        execution = compile_hypothesis_execution_report(
            blackboard=board,
            hypothesis_report=artifact,
        )
        self.assertEqual(execution["route_status"], "hypothesis_routes_pending_execution")
        self.assertEqual(execution["pending_candidate_count"], 2)

    def test_exhaustive_policy_stops_instead_of_repeating_stale_failure_critic(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=60)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "existing"}]}
        board["route_failures"] = [{"schema_version": "agent_route_failure.v1", "reason": "large_atom_jump"}]
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "known_source", "doi": "10.0000/example", "title": "known source"}
        ]
        board["current_belief"]["next_action_bias"] = []
        board["current_belief"]["template_policy"]["enabled"] = False
        board["analogical_hypothesis_ranking"] = {"ranked_hypotheses": []}
        board["action_history"] = [
            {"round_index": 1, "action_type": "run_guided_chemenzy", "useful_artifact": True, "reasons": ["large_atom_jump"]},
            {"round_index": 2, "action_type": "build_failure_critic_report", "useful_artifact": True, "reasons": []},
            {"round_index": 2, "action_type": "stitch_parent_route", "useful_artifact": True, "reasons": ["parent_route_verifier_not_accepted"]},
        ]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)

        self.assertEqual([row["action_type"] for row in batch["actions"]], ["stop_unresolved"])
        self.assertIn("no non-stale action", batch["actions"][0]["rationale"])

    def test_hypothesis_report_relaxes_final_verdict_without_solved_claim(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["parent_route_proof"] = {
            "accepted": False,
            "solved": False,
            "route_status": "partial_anchor_only_not_solved",
            "reasons": ["parent_route_verifier_not_accepted"],
        }
        report = {
            "schema_version": "hypothesis_only_retrosynthesis_report.v1",
            "accepted": True,
            "candidate_precursor_count": 1,
            "no_solved_claim": True,
        }
        final = emit_agentic_final_verdict(
            blackboard=board,
            artifacts={
                "hypothesis_only_retrosynthesis_report": {
                    "artifact_type": "HypothesisOnlyRetrosynthesisReport",
                    "payload": report,
                }
            },
            bundle={"case_id": "target1_steroid"},
        ).to_dict()

        self.assertEqual(final["verdict"], "hypothesis_route_proposed")
        self.assertEqual(final["route_status"], "hypothesis_route_proposed")
        self.assertFalse(final["solved"])
        self.assertIn("hypothesis_only_retrosynthesis_available", final["reasons"])
        validation = _validate_agentic_final_verdict(final, blackboard=board, validations=[])
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_plausible_route_proof_bundle_surfaces_as_hypothesis_verdict(self):
        target = TargetInput(target_name="atorvastatin_like", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        route_proof_bundle = {
            "accepted": True,
            "result": {
                "schema_version": "route_proof_bundle.v1",
                "accepted": False,
                "solved": False,
                "route_status": "plausible_hypothesis_route",
                "reasons": ["deterministic_connected_route_not_proven"],
                "objective_proofs": [
                    {
                        "schema_version": "objective_specific_route_proof.v1",
                        "accepted": False,
                        "solved": False,
                        "route_status": "plausible_hypothesis_route",
                        "reasons": ["deterministic_connected_route_not_proven"],
                    }
                ],
            },
        }

        final = emit_agentic_final_verdict(
            blackboard=board,
            artifacts={"route_proof_bundle": route_proof_bundle},
            bundle={"case_id": "atorvastatin_like"},
        ).to_dict()

        self.assertEqual(final["verdict"], "hypothesis_route_proposed")
        self.assertEqual(final["route_status"], "plausible_hypothesis_route")
        self.assertFalse(final["solved"])
        self.assertIn("hypothesis_only_retrosynthesis_available", final["reasons"])
        validation = _validate_agentic_final_verdict(final, blackboard=board, validations=[])
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_hypothesis_execution_status_preempts_generic_plausible_proof_status(self):
        target = TargetInput(target_name="atorvastatin_like", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        route_proof_bundle = {
            "accepted": True,
            "result": {
                "schema_version": "route_proof_bundle.v1",
                "accepted": False,
                "solved": False,
                "route_status": "plausible_hypothesis_route",
                "reasons": ["deterministic_connected_route_not_proven"],
                "objective_proofs": [
                    {
                        "schema_version": "objective_specific_route_proof.v1",
                        "accepted": False,
                        "solved": False,
                        "route_status": "plausible_hypothesis_route",
                        "reasons": ["deterministic_connected_route_not_proven"],
                    }
                ],
            },
        }
        hypothesis_execution_report = {
            "schema_version": "hypothesis_execution_report_artifact.v1",
            "artifact_type": "HypothesisExecutionReport",
            "artifact_id": "atorvastatin_like:hypothesis_execution_report",
            "case_id": "atorvastatin_like",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["route_expansion_subgoal_search_result.json"],
            "validation_status": "accepted",
            "payload": {
                "schema_version": "hypothesis_execution_report.v1",
                "accepted": True,
                "route_status": "hypothesis_route_execution_partial",
                "solved": False,
                "no_parent_solved_claim": True,
                "hypotheses_must_be_executed": True,
                "candidate_count": 2,
                "executed_candidate_count": 1,
                "verified_child_route_count": 0,
                "rejected_candidate_count": 1,
                "pending_candidate_count": 1,
                "recursive_followup_task_count": 1,
                "pending_recursive_followup_count": 1,
                "candidate_executions": [
                    {
                        "schema_version": "hypothesis_candidate_execution.v1",
                        "candidate_id": "hypothesis:1",
                        "precursor_smiles": "CCO",
                        "execution_status": "executed_rejected",
                        "verifier_accepted": False,
                        "solved": False,
                        "route_status": "fake_closed_rejected",
                        "no_parent_solved_claim": True,
                    },
                    {
                        "schema_version": "hypothesis_candidate_execution.v1",
                        "candidate_id": "hypothesis:2",
                        "precursor_smiles": "CCN",
                        "execution_status": "not_executed",
                        "verifier_accepted": False,
                        "solved": False,
                        "route_status": "not_executed",
                        "no_parent_solved_claim": True,
                    },
                ],
            },
        }

        final = emit_agentic_final_verdict(
            blackboard=board,
            artifacts={
                "route_proof_bundle": route_proof_bundle,
                "hypothesis_execution_report": hypothesis_execution_report,
            },
            bundle={"case_id": "atorvastatin_like"},
        ).to_dict()

        self.assertEqual(final["verdict"], "hypothesis_route_proposed")
        self.assertEqual(final["route_status"], "hypothesis_route_execution_partial")
        self.assertFalse(final["solved"])

    def test_hypothesis_execution_report_tracks_rejected_route_expansion(self):
        target = TargetInput(target_name="target1_steroid", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        hypothesis_report = {
            "schema_version": "hypothesis_only_retrosynthesis_report_artifact.v1",
            "artifact_type": "HypothesisOnlyRetrosynthesisReport",
            "artifact_id": "target1_steroid:hypothesis_only_retrosynthesis_report",
            "case_id": "target1_steroid",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["local_pdf:test"],
            "validation_status": "accepted",
            "payload": {
                "schema_version": "hypothesis_only_retrosynthesis_report.v1",
                "accepted": True,
                "solved": False,
                "no_solved_claim": True,
                "candidate_precursor_count": 1,
                "candidate_precursors": [
                    {
                        "schema_version": "hypothesis_precursor_candidate.v1",
                        "candidate_id": "hypothesis:protected_alcohol",
                        "precursor_role": "same_core_protected_alcohol",
                        "precursor_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "allowed_use": "guided_search_seed_only",
                    }
                ],
            },
        }
        route_expansion = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "result": {
                "subgoal_count": 1,
                "accepted_subgoal_count": 0,
                "rejected_subgoal_count": 1,
                "subgoals": [
                    {
                        "subgoal": {
                            "name": "same_core_protected_alcohol",
                            "smiles": BUFOTALIN_ACHIRAL_SMILES,
                        },
                        "route_count": 12,
                        "accepted": False,
                        "solved": False,
                        "verifier": {
                            "accepted": False,
                            "route_status": "fake_closed_rejected",
                            "reasons": ["large_atom_jump", "no_verifier_accepted_stock_closed_route"],
                            "accepted_route_count": 0,
                            "rejected_route_count": 12,
                        },
                    }
                ],
            },
        }

        payload = compile_hypothesis_execution_report(
            blackboard=board,
            hypothesis_report=hypothesis_report,
            route_expansion_results=[route_expansion],
        )

        self.assertEqual(payload["route_status"], "hypothesis_routes_executed_rejected")
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["executed_candidate_count"], 1)
        self.assertEqual(payload["rejected_candidate_count"], 1)
        self.assertEqual(payload["pending_candidate_count"], 0)
        row = payload["candidate_executions"][0]
        self.assertEqual(row["execution_status"], "executed_rejected")
        self.assertEqual(row["route_count"], 12)
        self.assertIn("large_atom_jump", row["reasons"])
        artifact = {
            "schema_version": "hypothesis_execution_report_artifact.v1",
            "artifact_type": "HypothesisExecutionReport",
            "artifact_id": "target1_steroid:hypothesis_execution_report",
            "case_id": "target1_steroid",
            "source": "test",
            "input_refs": ["agent_blackboard.json"],
            "evidence_refs": ["route_expansion_subgoal_search_result.json"],
            "validation_status": "accepted",
            "payload": payload,
        }
        validation = validate_typed_artifact(artifact)
        self.assertTrue(validation["accepted"], validation["reasons"])

        final = emit_agentic_final_verdict(
            blackboard=board,
            artifacts={
                "hypothesis_only_retrosynthesis_report": hypothesis_report,
                "hypothesis_execution_report": artifact,
            },
            bundle={"case_id": "target1_steroid"},
        ).to_dict()
        self.assertEqual(final["verdict"], "hypothesis_route_proposed")
        self.assertEqual(final["route_status"], "hypothesis_routes_executed_rejected")
        self.assertFalse(final["solved"])

    def test_rejected_hypothesis_subgoal_creates_recursive_followup_tasks(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=6)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "expand:hypothesis",
            "action_type": "expand_child_target",
            "rationale": "execute first-level hypothesis precursor",
            "expected_artifact": "route_expansion_subgoal_search_result.v1",
            "success_condition": "verifier result is recorded",
            "payload": {
                "subgoal_targets": [
                    {
                        "name": "same_core_alcohol_precursor",
                        "smiles": "CCO",
                        "source": "analogical_hypothesis_precursor_hint",
                        "hypothesis_only_not_solved": True,
                    }
                ]
            },
        }
        action_result = {
            "accepted": True,
            "result": {
                "schema_version": "route_expansion_subgoal_search_result.v1",
                "accepted": False,
                "solved": False,
                "subgoal_count": 1,
                "accepted_subgoal_count": 0,
                "rejected_subgoal_count": 1,
                "reasons": ["no_route_expansion_subgoal_verified_solved"],
                "subgoals": [
                    {
                        "accepted": False,
                        "solved": False,
                        "route_count": 3,
                        "subgoal": {
                            "name": "same_core_alcohol_precursor",
                            "smiles": "CCO",
                            "source": "analogical_hypothesis_precursor_hint",
                            "hypothesis_only_not_solved": True,
                            "policy": {
                                "compiler_metadata": {
                                    "hypothesis_only_not_solved": True,
                                    "no_solved_claim": True,
                                }
                            },
                        },
                        "verifier": {
                            "accepted": False,
                            "route_status": "fake_closed_rejected",
                            "reasons": ["large_atom_jump"],
                        },
                    }
                ],
            },
            "reasons": ["no_route_expansion_subgoal_verified_solved"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action=action,
                action_result=action_result,
                round_index=2,
                run_dir=tmp,
            )

        tasks = board["recursive_hypothesis_tasks"]
        self.assertGreaterEqual(len(tasks), 1)
        self.assertTrue(all(row["schema_version"] == "recursive_hypothesis_task.v1" for row in tasks))
        self.assertTrue(all(row["recursive_depth"] == 1 for row in tasks))
        self.assertTrue(all(row["no_solved_claim"] for row in tasks))
        self.assertTrue(all(row["precursor_smiles"] != "CCO" for row in tasks))
        self.assertIn("expand_child_target", board["current_belief"]["next_action_bias"])
        self.assertFalse(board["current_belief"]["child_route_solved"])

    def test_planner_expands_recursive_hypothesis_tasks(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_route_expansion_subgoal_runs": 4},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "doi:10.0000/example", "doi": "10.0000/example", "title": "Example source"}
        ]
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:test",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "rejected_hypothesis_precursor",
                "parent_smiles": "CCO",
                "precursor_smiles": "CC=O",
                "name": "recursive_primary_alcohol_to_aldehyde_precursor",
                "recursive_depth": 1,
                "operation_idea": "continue through aldehyde-level oxidation state",
                "variant_type": "primary_alcohol_to_aldehyde_precursor",
                "failure_reasons": ["large_atom_jump"],
                "allowed_use": "route_expansion_subgoal_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        expand = [row for row in batch["actions"] if row["action_type"] == "expand_child_target"]

        self.assertTrue(expand)
        target_payload = expand[0]["payload"]["subgoal_targets"][0]
        self.assertEqual(target_payload["smiles"], "CC=O")
        self.assertEqual(target_payload["source"], "recursive_hypothesis_task")
        self.assertEqual(target_payload["recursive_depth"], 1)
        policy = target_payload["chem_enzy_search_policy"]
        self.assertTrue(policy["source_budget"]["recursive_hypothesis_frontier"])
        self.assertIn("recursive_failed_hypothesis_frontier_expansion", policy["source_budget"]["preferred_reaction_classes"])

    def test_planner_continues_with_refined_component_after_child_failure(self):
        target = TargetInput(target_name="phenyl acetate", target_smiles="CC(=O)Oc1ccccc1")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=6,
            budget_limits={
                "max_route_expansion_subgoal_runs": 4,
                "max_scout_calls": 1,
                "max_visual_calls": 1,
            },
        )
        board["budget_state"]["scout_calls"] = board["budget_state"]["max_scout_calls"]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "target_side:ester"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.0000/ester",
                "doi": "10.0000/ester",
                "title": "Ester disconnection precedent",
            }
        ]
        strategy = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "accepted": True,
            "hypotheses": [
                {
                    "hypothesis_id": "target_side:ester",
                    "target_handle": "aryl ester",
                    "proposed_disconnection_region": "acyl-oxygen ester disconnection",
                    "expected_precursor_type": "carboxylic acid or activated acyl donor plus phenol",
                    "must_preserve_substructure": ["aryl fragment"],
                    "confidence": "medium",
                    "required_verification": ["route_expansion_verifier"],
                }
            ],
            "bridge_tasks": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r1:strategy", "action_type": "generate_disconnection_hypotheses"},
                action_result=strategy,
                round_index=1,
                run_dir=tmp,
            )
            first_batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
            first_expand = next(row for row in first_batch["actions"] if row["action_type"] == "expand_child_target")
            acid_target = next(
                row
                for row in first_expand["payload"]["subgoal_targets"]
                if row["smiles"] == "CC(=O)O"
            )
            route_result = {
                "schema_version": "route_expansion_subgoal_search_result.v1",
                "accepted": False,
                "status": "failed",
                "solved": False,
                "subgoals": [
                    {
                        "schema_version": "route_expansion_subgoal_result.v1",
                        "accepted": False,
                        "solved": False,
                        "reasons": ["no_route_expansion_subgoal_verified_solved"],
                        "verifier": {"accepted": False, "reasons": ["target_unresolved"]},
                        "subgoal": {
                            "schema_version": "route_expansion_child_target.v1",
                            "name": acid_target["name"],
                            "smiles": acid_target["smiles"],
                            "source": acid_target["source"],
                            "hypothesis_only_not_solved": True,
                            "recursive_hypothesis_task_id": acid_target["recursive_hypothesis_task_id"],
                            "parent_candidate_id": acid_target["parent_candidate_id"],
                            "parent_smiles": acid_target["parent_smiles"],
                            "task_scope": acid_target["task_scope"],
                            "precursor_set_smiles": acid_target["precursor_set_smiles"],
                            "precursor_component_index": acid_target["precursor_component_index"],
                            "precursor_component_count": acid_target["precursor_component_count"],
                            "multi_component_precursor_set": acid_target["multi_component_precursor_set"],
                            "requires_precursor_set_stitching": acid_target["requires_precursor_set_stitching"],
                            "sibling_precursor_smiles": acid_target["sibling_precursor_smiles"],
                            "policy": acid_target["chem_enzy_search_policy"],
                        },
                    }
                ],
                "reasons": ["no_route_expansion_subgoal_verified_solved"],
            }
            board = update_blackboard_from_action(
                board,
                action={**first_expand, "action_id": "r2:expand"},
                action_result={"accepted": True, "result": route_result},
                round_index=2,
                run_dir=tmp,
            )

        self.assertEqual(len(board["proposal_failure_feedback"]), 1)
        refined = [
            row
            for row in board["retrosynthetic_proposals"]
            if row.get("source_type") == "failure_driven_proposal_refinement"
        ]
        self.assertTrue(any(row["precursor_smiles"] == "COC(C)=O.Oc1ccccc1" for row in refined))

        next_batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        next_action_types = [row["action_type"] for row in next_batch["actions"]]
        self.assertIn("expand_child_target", next_action_types)
        self.assertNotIn("stop_unresolved", next_action_types)
        next_expand = next(row for row in next_batch["actions"] if row["action_type"] == "expand_child_target")
        next_targets = next_expand["payload"]["subgoal_targets"]
        next_smiles = [row["smiles"] for row in next_targets]
        self.assertNotIn("CC(=O)O", next_smiles)
        self.assertTrue({"CC(=O)Cl", "COC(C)=O"} & set(next_smiles))
        refined_target = next(row for row in next_targets if row["smiles"] in {"CC(=O)Cl", "COC(C)=O"})
        self.assertTrue(refined_target["requires_precursor_set_stitching"])
        self.assertIn("Oc1ccccc1", refined_target["sibling_precursor_smiles"])
        validation = validate_action_batch(next_batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_parent_relevance_rejection_does_not_accept_recursive_child_task(self):
        board = initialize_agent_blackboard(
            target_input={"target_name": "parent", "target_smiles": "CCN"},
            preflight={"accepted": True, "case_id": "parent"},
        )
        board["recursive_hypothesis_tasks"] = [
            {
                "task_id": "recursive:child",
                "precursor_smiles": "CCO",
                "status": "pending",
            }
        ]
        route_result = {
            "schema_version": "route_expansion_subgoal_search_result.v1",
            "accepted": False,
            "solved": False,
            "accepted_subgoal_count": 0,
            "subgoals": [
                {
                    "schema_version": "route_expansion_subgoal_result.v1",
                    "accepted": False,
                    "solved": False,
                    "route_status": "child_component_not_parent_proximal",
                    "reasons": ["child_component_not_parent_proximal"],
                    "verifier": {"accepted": True, "route_status": "solved", "reasons": []},
                    "parent_relevance_gate": {
                        "accepted": False,
                        "reasons": ["child_component_not_parent_proximal"],
                    },
                    "subgoal": {
                        "name": "child",
                        "smiles": "CCO",
                        "source": "hypothesis",
                        "recursive_hypothesis_task_id": "recursive:child",
                        "parent_smiles": "CCN",
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r1:expand", "action_type": "expand_child_target"},
                action_result={"accepted": True, "result": route_result},
                round_index=1,
                run_dir=tmp,
            )

        task = board["recursive_hypothesis_tasks"][0]
        self.assertEqual(task["status"], "rejected")
        self.assertFalse(task["last_attempt_accepted"])
        self.assertIn("child_component_not_parent_proximal", task["last_attempt_reasons"])
        self.assertEqual(len(board["proposal_failure_feedback"]), 1)
        self.assertIn(
            "child_component_not_parent_proximal",
            board["proposal_failure_feedback"][0]["failure_reasons"],
        )

    def test_invalid_input_still_emits_agentic_closing_audit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="invalid_case",
                target_smiles="not_a_smiles",
                output_dir=tmp,
                max_rounds=3,
            )
            generated_route_html = Path(result["final_verdict"]["artifact_refs"]["route_forest_html"]).exists()
            generated_route_forest = Path(result["final_verdict"]["artifact_refs"]["explored_route_forest"]).exists()

        self.assertEqual(result["final_verdict"]["verdict"], "invalid_input")
        self.assertFalse(result["preflight"]["accepted"])
        self.assertEqual(result["action_batches"], [])
        artifacts = result["artifact_bundle"]["artifacts"]
        self.assertEqual(artifacts["agent_blackboard_snapshot"]["artifact_type"], "AgentBlackboardSnapshot")
        self.assertEqual(artifacts["agentic_capability_audit"]["artifact_type"], "AgenticCapabilityAudit")
        self.assertEqual(artifacts["agentic_final_verdict_validation"]["artifact_type"], "AgenticFinalVerdictValidation")
        self.assertEqual(artifacts["hypothesis_only_retrosynthesis_report"]["artifact_type"], "HypothesisOnlyRetrosynthesisReport")
        self.assertEqual(artifacts["agentic_run_audit"]["artifact_type"], "AgenticRunAudit")
        self.assertEqual(artifacts["route_forest_display"]["artifact_type"], "ExploredRouteForestDisplay")
        refs = result["final_verdict"]["artifact_refs"]
        self.assertIn("agent_blackboard_snapshot", refs)
        self.assertIn("agentic_capability_audit", refs)
        self.assertIn("agentic_final_verdict_validation", refs)
        self.assertIn("hypothesis_only_retrosynthesis_report", refs)
        self.assertIn("agentic_run_audit", refs)
        self.assertIn("route_forest_html", refs)
        self.assertIn("explored_route_forest", refs)
        self.assertTrue(generated_route_html)
        self.assertTrue(generated_route_forest)
        self.assertEqual(result["artifacts"]["route_forest_html"], refs["route_forest_html"])
        self.assertEqual(result["artifacts"]["explored_route_forest"], refs["explored_route_forest"])
        capability_payload = artifacts["agentic_capability_audit"]["payload"]
        self.assertTrue(capability_payload["accepted"], capability_payload["failed_requirements"])
        capability_checks = {
            row["requirement_id"]: row
            for row in capability_payload["requirement_checks"]
        }
        self.assertTrue(
            capability_checks["artifact_refs_and_typed_validation_integrity"]["accepted"],
            capability_checks["artifact_refs_and_typed_validation_integrity"]["reasons"],
        )
        preflight_statuses = {
            row["requirement_id"]: row["status"]
            for row in capability_payload["requirement_checks"]
            if row["requirement_id"] in {
                "policy_driven_typed_action_batches",
                "deterministic_action_batch_validation_gate",
                "planner_decision_history_audited",
            }
        }
        self.assertEqual(preflight_statuses["policy_driven_typed_action_batches"], "preflight_rejected")
        self.assertEqual(preflight_statuses["deterministic_action_batch_validation_gate"], "preflight_rejected")
        self.assertEqual(preflight_statuses["planner_decision_history_audited"], "preflight_rejected")
        validation_keys = {
            row.get("artifact_key")
            for row in result["artifact_bundle"]["validations"]
            if row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
            and row.get("accepted")
        }
        self.assertIn("agent_blackboard_snapshot", validation_keys)
        self.assertIn("agentic_capability_audit", validation_keys)
        self.assertIn("agentic_final_verdict_validation", validation_keys)
        self.assertIn("agentic_run_audit", validation_keys)

    def test_action_batch_validation_rejects_unknown_solved_and_raw_payloads(self):
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "bad",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "bad:1",
                    "action_type": "unknown",
                    "rationale": "x",
                    "expected_artifact": "x",
                    "success_condition": "x",
                    "payload": {"rxn_smiles": "CCO>>CC=O"},
                    "route_status": "solved",
                }
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("unknown_action:0:unknown", validation["reasons"])
        self.assertIn("raw_reaction_injection", validation["reasons"])
        self.assertIn("planner_direct_solved_claim", validation["reasons"])

    def test_action_batch_validation_rejects_bad_semantics_and_hidden_reaction_string(self):
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "bad_semantics",
            "round_index": 1,
            "route_status": "solved",
            "notes": "Do not allow hidden reaction strings like CCO>>CC=O.",
            "semantics": {
                "planner_can_emit_solved": True,
                "raw_reaction_output_allowed": True,
                "deterministic_validator_required": True,
            },
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "stop",
                    "action_type": "stop_unresolved",
                    "rationale": "stop",
                    "expected_artifact": "stop marker",
                    "success_condition": "stop selected",
                    "payload": {},
                }
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("planner_direct_solved_claim", validation["reasons"])
        self.assertIn("planner_semantics_allow_solved_claim", validation["reasons"])
        self.assertIn("planner_semantics_allow_raw_reaction_output", validation["reasons"])
        self.assertIn("raw_reaction_injection", validation["reasons"])

    def test_action_batch_validation_requires_direction_change_after_two_unproductive_rounds(self):
        repeated_action = {
            "schema_version": "agent_action.v1",
            "action_id": "search:same",
            "action_type": "search_literature",
            "rationale": "repeat same search",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidate",
            "payload": _test_search_payload("same"),
        }
        repeated_signature = json.dumps(
            {"action_type": "search_literature", "payload": _test_search_payload("same")},
            sort_keys=True,
        )
        board = {
            "action_history": [
                {
                    "round_index": 1,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "action_signature": repeated_signature,
                },
                {
                    "round_index": 2,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "action_signature": repeated_signature,
                },
            ]
        }
        repeated_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "stuck",
            "round_index": 3,
            "actions": [repeated_action],
        }
        changed_batch = {
            **repeated_batch,
            "actions": [{**repeated_action, "action_id": "search:new", "payload": _test_search_payload("new")}],
        }
        stop_batch = {
            **repeated_batch,
            "actions": [
                {
                    **repeated_action,
                    "action_id": "stop",
                    "action_type": "stop_unresolved",
                    "payload": {},
                    "rationale": "stop after repeated unproductive rounds",
                    "expected_artifact": "stop marker",
                    "success_condition": "stop selected",
                }
            ],
        }

        repeated_validation = validate_action_batch(repeated_batch, blackboard=board)
        changed_validation = validate_action_batch(changed_batch, blackboard=board)
        stop_validation = validate_action_batch(stop_batch, blackboard=board)

        self.assertFalse(repeated_validation["accepted"])
        self.assertIn(
            "planner_must_stop_or_change_direction_after_two_unproductive_rounds",
            repeated_validation["reasons"],
        )
        self.assertTrue(changed_validation["accepted"], changed_validation["reasons"])
        self.assertTrue(stop_validation["accepted"], stop_validation["reasons"])

    def test_repeated_empty_literature_search_is_not_planned_from_bias(self):
        board = {
            "case_id": "empty_search",
            "target_profile": {"valid": True, "target_name": "steroid", "family_hint": "steroid"},
            "target_side_disconnection_hypotheses": {"hypotheses": [{"hypothesis_id": "h1"}]},
            "analogical_hypothesis_ranking": {"selected_hypotheses": [{"hypothesis_id": "h1"}]},
            "bridge_tasks": [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}],
            "literature_evidence": {
                "source_candidates": [
                    {
                        "source_ref": "doi:10.1000/local",
                        "doi": "10.1000/local",
                        "local_pdf": "/tmp/source.pdf",
                        "access_status": "local_pdf_available",
                    }
                ]
            },
            "current_belief": {"next_action_bias": ["search_literature"]},
            "budget_state": {
                "scout_calls": 2,
                "max_scout_calls": 5,
                "visual_calls": 0,
                "max_visual_calls": 0,
                "chemenzy_runs": 0,
                "max_chemenzy_runs": 1,
                "child_target_runs": 0,
                "max_child_target_runs": 0,
            },
            "action_history": [
                {
                    "round_index": 3,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "reasons": ["no_source_candidates"],
                    "blackboard_delta": {},
                    "action_signature": json.dumps({"action_type": "search_literature", "payload": {"query": "first"}}),
                },
                {
                    "round_index": 4,
                    "action_type": "search_literature",
                    "useful_artifact": False,
                    "stale": True,
                    "reasons": ["no_source_candidates"],
                    "blackboard_delta": {},
                    "action_signature": json.dumps({"action_type": "search_literature", "payload": {"query": "second"}}),
                },
            ],
        }

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("search_literature", action_types)

    def test_repeated_literature_source_is_stale_not_useful(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO", family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "search",
            "action_type": "search_literature",
            "rationale": "search",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidate",
            "payload": _test_search_payload("steroid synthesis"),
        }
        result = {
            "accepted": True,
            "result": {
                "schema_version": "literature_scout_report.v1",
                "source_candidates": [
                    {
                        "source_ref": "doi:10.1000/source",
                        "doi": "10.1000/source",
                        "title": "Source",
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(board, action=action, action_result=result, round_index=1, run_dir=tmp)
            board = update_blackboard_from_action(board, action=action, action_result=result, round_index=2, run_dir=tmp)

        self.assertTrue(board["action_history"][0]["useful_artifact"])
        self.assertFalse(board["action_history"][1]["useful_artifact"])
        self.assertTrue(board["action_history"][1]["stale"])

    def test_action_batch_validation_rejects_round_budget_overrun(self):
        action = {
            "schema_version": "agent_action.v1",
            "rationale": "x",
            "expected_artifact": "x",
            "success_condition": "x",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "budget",
            "round_index": 1,
            "actions": [
                {**action, "action_id": "a", "action_type": "run_guided_chemenzy"},
                {**action, "action_id": "b", "action_type": "run_guided_chemenzy"},
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("guided_chemenzy_round_budget_exceeded", validation["reasons"])

    def test_action_batch_validation_rejects_total_budget_overrun(self):
        action = {
            "schema_version": "agent_action.v1",
            "rationale": "x",
            "expected_artifact": "x",
            "success_condition": "x",
            "payload": {},
        }
        board = {
            "budget_state": {
                "scout_calls": 3,
                "max_scout_calls": 3,
                "visual_calls": 2,
                "max_visual_calls": 2,
                "chemenzy_runs": 1,
                "max_chemenzy_runs": 1,
                "child_target_runs": 2,
                "max_child_target_runs": 2,
                "template_application_actions": 3,
                "max_template_application_actions": 3,
            }
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "budget",
            "round_index": 9,
            "actions": [
                {**action, "action_id": "search", "action_type": "search_literature"},
                {**action, "action_id": "visual", "action_type": "extract_visual_literature_chain"},
                {**action, "action_id": "chemenzy", "action_type": "run_guided_chemenzy"},
            ],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn("scout_total_budget_exceeded", validation["reasons"])
        self.assertIn("visual_total_budget_exceeded", validation["reasons"])
        self.assertIn("guided_chemenzy_total_budget_exceeded", validation["reasons"])

        child_batch = {**batch, "actions": [{**action, "action_id": "child", "action_type": "expand_child_target"}]}
        template_batch = {
            **batch,
            "actions": [{**action, "action_id": "template", "action_type": "apply_analogical_template_to_target"}],
        }

        self.assertIn("child_expansion_total_budget_exceeded", validate_action_batch(child_batch, blackboard=board)["reasons"])
        self.assertIn(
            "template_application_total_budget_exceeded",
            validate_action_batch(template_batch, blackboard=board)["reasons"],
        )

    def test_action_batch_validation_does_not_count_pdf_structure_as_visual_budget(self):
        board = {"budget_state": {"visual_calls": 2, "max_visual_calls": 2}}
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "pdf",
            "action_type": "extract_pdf_literature_structures",
            "rationale": "read local PDF structures",
            "expected_artifact": "literature_pdf_structure_evidence.v1",
            "success_condition": "PDF structure evidence is recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "pdf_budget",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)
        after_pdf = update_budget_for_action(board, "extract_pdf_literature_structures", payload={})
        after_visual = update_budget_for_action(board, "extract_visual_literature_chain", payload={})

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(after_pdf["budget_state"]["visual_calls"], 2)
        self.assertEqual(after_visual["budget_state"]["visual_calls"], 3)

    def test_zero_tool_budgets_disable_optional_agent_actions(self):
        target = TargetInput(target_name="phenyl acetate", target_smiles="CC(=O)Oc1ccccc1")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
            budget_limits={
                "max_scout_calls": 0,
                "max_visual_calls": 0,
                "max_guided_chemenzy_runs": 0,
                "max_route_expansion_subgoal_runs": 0,
                "max_template_application_actions": 0,
            },
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "target_side:ester"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "target_side:ester"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:ester", "task_type": "target_proximal_bridge"}]
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:acid",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "retrosynthetic_proposal",
                "parent_candidate_id": "proposal:acid",
                "parent_smiles": "CC(=O)Oc1ccccc1",
                "precursor_smiles": "CC(=O)O",
                "name": "acid",
                "recursive_depth": 1,
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(board["budget_state"]["max_scout_calls"], 0)
        self.assertEqual(board["budget_state"]["max_visual_calls"], 0)
        self.assertEqual(board["budget_state"]["max_chemenzy_runs"], 0)
        self.assertEqual(board["budget_state"]["max_child_target_runs"], 0)
        self.assertNotIn("search_literature", action_types)
        self.assertNotIn("extract_visual_literature_chain", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertNotIn("expand_child_target", action_types)

    def test_blackboard_accepts_max_chem_enzy_runs_alias_as_guided_budget(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
            budget_limits={"max_chem_enzy_runs": 0},
        )

        self.assertEqual(board["budget_state"]["max_chemenzy_runs"], 0)

    def test_agentic_cli_max_chem_enzy_runs_alias_disables_guided_when_unset(self):
        from scripts.run_codex_entry_agentic_blackboard import _budget_from_args

        args = SimpleNamespace(
            timeout_s=120,
            max_chem_enzy_runs=0,
            max_guided_chemenzy_runs=None,
            guided_chemenzy_timeout_s=None,
            max_route_expansion_subgoal_runs=None,
            max_codex_research_runs=None,
            max_scout_calls=None,
            max_visual_calls=None,
            max_template_applications_per_round=5,
        )

        budget = _budget_from_args(args)

        self.assertEqual(budget.max_chem_enzy_runs, 0)
        self.assertEqual(budget.max_guided_chemenzy_runs, 0)

    def test_controller_preserves_explicit_zero_guided_and_child_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="zero_budget_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                use_codex_action_planner=False,
                budget=HarnessBudget(
                    max_guided_chemenzy_runs=0,
                    max_route_expansion_subgoal_runs=0,
                    max_scout_calls=0,
                    max_visual_calls=0,
                ),
            )

        budget_state = result["agent_blackboard"]["budget_state"]
        action_types = [
            row["action_type"]
            for batch in result["action_batches"]
            for row in batch.get("actions", [])
        ]

        self.assertEqual(budget_state["max_chemenzy_runs"], 0)
        self.assertEqual(budget_state["max_child_target_runs"], 0)
        self.assertEqual(budget_state["max_scout_calls"], 0)
        self.assertEqual(budget_state["max_visual_calls"], 0)
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertNotIn("expand_child_target", action_types)
        self.assertNotIn("search_literature", action_types)

    def test_child_expansion_budget_counts_planned_subgoal_targets(self):
        target = TargetInput(target_name="phenyl acetate", target_smiles="CC(=O)Oc1ccccc1")
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=3,
            budget_limits={"max_route_expansion_subgoal_runs": 2},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "target_side:ester"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "target_side:ester"}]}
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:acid",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "retrosynthetic_proposal",
                "parent_candidate_id": "proposal:acid",
                "parent_smiles": "CC(=O)Oc1ccccc1",
                "precursor_smiles": "CC(=O)O",
                "name": "acid",
                "recursive_depth": 1,
                "no_solved_claim": True,
            },
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:phenol",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "retrosynthetic_proposal",
                "parent_candidate_id": "proposal:phenol",
                "parent_smiles": "CC(=O)Oc1ccccc1",
                "precursor_smiles": "Oc1ccccc1",
                "name": "phenol",
                "recursive_depth": 1,
                "no_solved_claim": True,
            }
        ]
        payload = build_child_expansion_payload_from_blackboard(board)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "child",
            "action_type": "expand_child_target",
            "rationale": "expand two child targets",
            "expected_artifact": "route_expansion_subgoal_search_result.v1",
            "success_condition": "child targets are attempted",
            "payload": payload,
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "child_budget",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)
        after = update_budget_for_action(board, "expand_child_target", payload=payload)

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(after["budget_state"]["child_target_runs"], 2)
        nearly_exhausted = {"budget_state": {"child_target_runs": 1, "max_child_target_runs": 2}}
        over_budget_validation = validate_action_batch(batch, blackboard=nearly_exhausted)
        self.assertIn("child_expansion_total_budget_exceeded", over_budget_validation["reasons"])

        exhausted = {
            **after,
            "recursive_hypothesis_tasks": [
                {
                    "schema_version": "recursive_hypothesis_task.v1",
                    "task_id": "recursive_hypothesis:extra",
                    "task_type": "recursive_hypothesis_frontier_expansion",
                    "status": "pending",
                    "source": "retrosynthetic_proposal",
                    "parent_candidate_id": "proposal:extra",
                    "parent_smiles": "CC(=O)Oc1ccccc1",
                    "precursor_smiles": "CC(=O)Cl",
                    "name": "extra",
                    "recursive_depth": 1,
                    "no_solved_claim": True,
                }
            ],
            "literature_evidence": {"source_candidates": [], "exact_rows": []},
            "current_belief": {"constraints": {}, "template_policy": {"enabled": False}},
        }
        next_batch = plan_action_batch(exhausted, round_index=2, exhaust_round_budget=True)
        self.assertNotIn("expand_child_target", [row["action_type"] for row in next_batch["actions"]])

    def test_resume_exploration_budget_unblocks_accepted_template_hints(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            family_hint="statin",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=2,
            budget_limits={
                "max_guided_chemenzy_runs": 1,
                "max_route_expansion_subgoal_runs": 1,
                "max_visual_calls": 1,
                "max_scout_calls": 1,
                "max_codex_research_runs": 1,
                "max_template_application_actions": 1,
            },
        )
        precursor = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)n(CCC(O)CC(O)CC(=O)OC(C)(C)C)c1-c1ccccc1"
        board["template_applications"] = [
            {
                "application_id": "apply:visual_hydrolysis",
                "template_id": "visual_hydrolysis",
                "accepted": True,
                "executable_candidate_available": True,
                "hypothetical_precursor_hints": [
                    {
                        "precursor_smiles": precursor,
                        "target_smiles": ATORVASTATIN_FREE_ACID_SMILES,
                        "allowed_use": "guided_search_subgoal_hint_only",
                        "not_parent_route_proof": True,
                        "requires_verifier": True,
                    }
                ],
            }
        ]
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "target_side:statin"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "target_side:statin"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "target_side:statin"}]}
        board["analogical_templates"] = [
            {
                "template_id": "visual_hydrolysis",
                "reaction_center": {"product_retron_type": "visual_hydrolysis_salt_bridge"},
                "evidence_refs": ["visual:atorvastatin"],
            }
        ]
        board["analogical_template_ranking"] = {
            "selected_templates": [{"template_id": "visual_hydrolysis", "rank": 1}]
        }
        board["action_history"] = [
            {"round_index": 1, "action_type": "run_guided_chemenzy", "useful_artifact": True, "stale": False},
            {
                "round_index": 2,
                "action_type": "apply_analogical_template_to_target",
                "useful_artifact": True,
                "stale": False,
            },
        ]
        board["budget_state"].update(
            {
                "rounds_completed": 2,
                "max_rounds": 2,
                "chemenzy_runs": 1,
                "max_chemenzy_runs": 1,
                "child_target_runs": 1,
                "max_child_target_runs": 1,
                "codex_research_runs": 1,
                "max_codex_research_runs": 1,
                "scout_calls": 1,
                "max_scout_calls": 1,
                "visual_calls": 1,
                "max_visual_calls": 1,
                "template_application_actions": 1,
                "max_template_application_actions": 1,
            }
        )

        exhausted_batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        self.assertEqual(["stop_unresolved"], [row["action_type"] for row in exhausted_batch["actions"]])

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "budget.json").write_text(
                json.dumps(
                    {
                        "max_guided_chemenzy_runs": 1,
                        "max_route_expansion_subgoal_runs": 1,
                        "max_visual_calls": 1,
                    }
                ),
                encoding="utf-8",
            )
            _extend_round_budget(board, max_new_rounds=1)
            _extend_exploration_budget(board, extra_guided_runs=2, extra_child_target_runs=2)
            loaded_budget = _load_budget(Path(tmp), board)

        self.assertGreaterEqual(board["budget_state"]["max_chemenzy_runs"], 3)
        self.assertGreaterEqual(board["budget_state"]["max_child_target_runs"], 3)
        self.assertGreaterEqual(loaded_budget.max_guided_chemenzy_runs, 3)
        self.assertGreaterEqual(loaded_budget.max_route_expansion_subgoal_runs, 3)
        resumed_batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        resumed_actions = [row["action_type"] for row in resumed_batch["actions"]]

        self.assertNotEqual(["stop_unresolved"], resumed_actions)
        self.assertTrue({"run_guided_chemenzy", "expand_child_target"} & set(resumed_actions))
        child_actions = [row for row in resumed_batch["actions"] if row["action_type"] == "expand_child_target"]
        if child_actions:
            child_payload = child_actions[0]["payload"]
            child_smiles = {
                str(row.get("smiles") or "")
                for row in child_payload.get("subgoal_targets") or []
                if isinstance(row, dict)
            }
            self.assertIn(precursor, child_smiles)
        validation = validate_action_batch(resumed_batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_resume_cli_summary_surfaces_advisory_route_display(self):
        compact = _compact_cli_result(
            {
                "run_dir": "run/atorvastatin",
                "resume_summary": {"executed_rounds": 2},
                "final_verdict": {
                    "verdict": "hypothesis_route_proposed",
                    "route_status": "hypothesis_route_execution_partial",
                    "solved": False,
                    "reasons": ["stock_audit_not_passed"],
                },
                "action_batches": [{}, {}],
                "validations": [],
                "artifacts": {
                    "route_forest_html": "run/atorvastatin/route_forest.html",
                    "explored_route_forest": "run/atorvastatin/explored_route_forest.json",
                },
                "artifact_bundle": {
                    "artifacts": {
                        "route_forest_display": {
                            "payload": {
                                "accepted": True,
                                "counts": {"branches": 4, "steps": 9, "nodes": 12},
                                "primary_branch": {
                                    "branch_id": "branch:recommended_atorvastatin_paal_knorr_process",
                                    "title": "推荐主线: Paal-Knorr 工艺路线到 atorvastatin",
                                    "kind": "recommended_strategy",
                                    "step_count": 5,
                                },
                            }
                        }
                    }
                },
            }
        )

        self.assertEqual(compact["route_display"]["outcome"], "advisory_route_available_not_solved")
        self.assertEqual(compact["route_display"]["branch_count"], 4)
        self.assertEqual(
            compact["route_display"]["primary_branch"]["title"],
            "推荐主线: Paal-Knorr 工艺路线到 atorvastatin",
        )
        self.assertEqual(compact["route_display"]["html_path"], "run/atorvastatin/route_forest.html")

    def test_child_expansion_prioritizes_visual_literature_precursors_over_failure_feedback(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            family_hint="statin",
        )
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=run_preflight(target),
            max_rounds=4,
            budget_limits={"max_route_expansion_subgoal_runs": 2},
        )
        board["recursive_hypothesis_tasks"] = [
            {
                "task_id": "recursive_hypothesis:failed_acetate",
                "status": "pending",
                "name": "failed_acetate_to_alcohol_component",
                "precursor_smiles": "CC(=O)O",
                "parent_candidate_id": "proposal:failed_acetate",
                "proposal_granularity": "same_core",
                "proposal_score": 95,
                "recursive_depth": 1,
                "risk_flags": ["failure_driven_refinement"],
            },
            {
                "task_id": "recursive_hypothesis:src003_32",
                "status": "pending",
                "name": "32",
                "precursor_smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)n(CCC(O)CC(=O)CC(=O)OC(C)(C)C)c1-c1ccc(F)cc1",
                "parent_candidate_id": "proposal:src003_32",
                "proposal_granularity": "fallback",
                "proposal_score": 46,
                "recursive_depth": 1,
                "variant_type": "visual_connectivity_candidate",
                "risk_flags": ["visual_literature_chain_missing_expected_labels", "exploratory_visual_candidate"],
            },
            {
                "task_id": "recursive_hypothesis:bmc_4",
                "status": "pending",
                "name": "4",
                "precursor_smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)n(CCC2CC(CC(=O)OC(C)(C)C)OC(C)(C)O2)c1-c1ccccc1",
                "parent_candidate_id": "proposal:bmc_4",
                "proposal_granularity": "fallback",
                "proposal_score": 55,
                "recursive_depth": 1,
                "variant_type": "visual_connectivity_candidate",
                "risk_flags": ["exploratory_visual_candidate", "visual_connectivity_approximation"],
            },
            {
                "task_id": "recursive_hypothesis:bmc_5",
                "status": "pending",
                "name": "5",
                "precursor_smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)n(CCC(O)CC(O)CC(=O)OC(C)(C)C)c1-c1ccccc1",
                "parent_candidate_id": "proposal:bmc_5",
                "proposal_granularity": "fallback",
                "proposal_score": 52,
                "recursive_depth": 1,
                "variant_type": "visual_connectivity_candidate",
                "risk_flags": ["exploratory_visual_candidate", "visual_connectivity_approximation"],
            },
        ]

        payload = _child_expansion_payload(board)
        selected_names = [row["name"] for row in payload["subgoal_targets"]]

        self.assertEqual(selected_names[0], "4")
        self.assertIn("5", selected_names)
        self.assertNotIn("failed_acetate_to_alcohol_component", selected_names)
        self.assertNotIn("32", selected_names)

    def test_action_batch_validation_does_not_count_template_actions_as_literature_sources(self):
        action = {
            "schema_version": "agent_action.v1",
            "rationale": "x",
            "expected_artifact": "x",
            "success_condition": "x",
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "template_parallel",
            "round_index": 5,
            "actions": [
                {
                    **action,
                    "action_id": "a",
                    "action_type": "search_literature",
                    "payload": _test_search_payload("template parallel literature", max_sources=3),
                },
                {
                    **action,
                    "action_id": "b",
                    "action_type": "extract_analogical_reaction_templates",
                    "payload": _test_analogical_template_payload(
                        "extract_analogical_reaction_templates",
                        max_templates=10,
                    ),
                },
            ],
        }

        validation = validate_action_batch(batch)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_guided_chemenzy_search_policy(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "try guided search",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "guided_missing_policy",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("guided_chemenzy_payload:0:missing_search_policy", validation["reasons"])

    def test_action_batch_validation_accepts_blackboard_guided_chemenzy_policy(self):
        target = TargetInput(target_name="guided_valid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_refs"] = ["doi:10.0000/source"]
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "try guided search with auditable policy",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": build_agentic_guided_payload(board),
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_accepts_runtime_rebuilt_guided_chemenzy_policy(self):
        target = TargetInput(target_name="guided_runtime_rebuild", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "guided",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "try guided search with runtime policy rebuild",
                    "expected_artifact": "guided_chemenzy_result.v1",
                    "success_condition": "verifier feedback is recorded",
                    "payload": {"guided_policy_runtime_rebuild": True},
                }
            ],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_allows_simple_direct_chemenzy_baseline(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "simple target direct baseline",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": build_agentic_guided_payload(board),
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_rejects_complex_guided_without_prior_signal(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "complex target premature guided search",
            "expected_artifact": "guided_chemenzy_result.v1",
            "success_condition": "verifier feedback is recorded",
            "payload": build_agentic_guided_payload(board),
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn(
            "guided_chemenzy_payload:0:guided_chemenzy_missing_prior_signal_for_complex_target",
            validation["reasons"],
        )

    def test_action_batch_validation_allows_bounded_complex_initial_probe(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        payload = build_agentic_guided_payload(board)
        policy = payload["search_policy"]
        policy["mode"] = "guided"
        policy["search_mode"] = "initial_probe"
        policy["source_budget"]["initial_scan_allowed"] = True
        policy["source_budget"]["max_candidates"] = 3
        policy["compiler_metadata"]["initial_scan_probe"] = True
        payload.update(
            {
                "initial_probe": True,
                "max_steps": 4,
                "chem_enzy_iterations": 6,
                "chem_enzy_expansion_topk": 12,
                "timeout_s": 60,
            }
        )
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "guided",
            "action_type": "run_guided_chemenzy",
            "rationale": "complex target cheap initial probe",
            "expected_artifact": "guided_chemenzy_probe_result.v1",
            "success_condition": "bounded verifier feedback is recorded",
            "payload": payload,
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_frontier_after_complex_initial_probe(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
            }
        )
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "extract_pdf",
            "action_type": "extract_pdf_literature_structures",
            "rationale": "continue source extraction",
            "expected_artifact": "literature_pdf_structure_evidence.v1",
            "success_condition": "pdf evidence is recorded",
            "payload": {"pdf_path": "/tmp/source.pdf", "source_ref": "doi:test", "no_solved_claim": True},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 2,
            "actions": [action],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn("complex_target_requires_frontier_bootstrap_after_initial_probe", validation["reasons"])

    def test_guided_retry_syncs_deep_budget_into_policy(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["budget_state"]["max_chemenzy_runs"] = 2
        board["budget_state"]["chemenzy_runs"] = 1
        board["route_failures"] = [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": "no_route_found",
                "route_status": "unresolved",
            }
        ]
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "src_open_process",
                "doi": "10.1186/s13065-015-0082-7",
                "title": "An improved kilogram-scale preparation of atorvastatin calcium",
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 2,
                "action_type": "search_literature",
                "useful_artifact": True,
                "stale": False,
            },
        ]

        batch = plan_action_batch(board, round_index=3, max_actions=3)

        guided = next(row for row in batch["actions"] if row["action_type"] == "run_guided_chemenzy")
        payload = guided["payload"]
        policy_budget = payload["chem_enzy_search_policy"]["budget"]
        self.assertEqual(payload["search_mode"], "guided_retry_after_initial_probe")
        self.assertEqual(payload["search_preset"], "bounded_retry")
        self.assertEqual(payload["max_steps"], 12)
        self.assertEqual(payload["chem_enzy_iterations"], 60)
        self.assertEqual(payload["chem_enzy_expansion_topk"], 120)
        self.assertEqual(payload["timeout_s"], 600)
        self.assertEqual(policy_budget["max_depth"], 12)
        self.assertEqual(policy_budget["max_iterations"], payload["chem_enzy_iterations"])
        self.assertEqual(policy_budget["expansion_topk"], payload["chem_enzy_expansion_topk"])
        self.assertEqual(policy_budget["timeout_s"], payload["timeout_s"])
        self.assertFalse(payload["chem_enzy_search_policy"]["source_budget"]["initial_scan_allowed"])
        self.assertTrue(validate_action_batch(batch, blackboard=board)["accepted"])

    def test_action_batch_validation_requires_explicit_child_subgoal_targets(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "child",
            "action_type": "expand_child_target",
            "rationale": "expand a child target",
            "expected_artifact": "route_expansion_subgoal_search_result.v1",
            "success_condition": "child verifier feedback is recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "child_missing_target",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("child_expansion_payload:0:missing_subgoal_targets", validation["reasons"])

    def test_action_batch_validation_requires_stitch_parent_route_binding(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "stitch",
            "action_type": "stitch_parent_route",
            "rationale": "prove parent route connectivity",
            "expected_artifact": "stitched_parent_route_proof.v1",
            "success_condition": "parent proof clauses are recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "stitch_missing_binding",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("stitch_parent_route_payload:0:missing_proof_binding", validation["reasons"])

    def test_action_batch_validation_requires_literature_search_policy(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "search",
            "action_type": "search_literature",
            "rationale": "search for literature",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidates are recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "search_missing_policy",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("search_literature_payload:0:missing_search_intent_or_queries", validation["reasons"])
        self.assertIn("search_literature_payload:0:missing_source_acquisition_policy", validation["reasons"])

    def test_action_batch_validation_accepts_source_acquisition_policy_schema_alias(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "search",
            "action_type": "search_literature",
            "rationale": "search for source metadata",
            "expected_artifact": "literature_scout_report.v1",
            "success_condition": "source candidates are recorded",
            "payload": {
                "search_intent": "find source metadata",
                "source_acquisition_policy": {
                    "schema_version": "source_acquisition_policy.v1",
                    "codex_online_first": True,
                    "local_pdf_fallback_allowed": True,
                    "placeholder_allowed_after_failures": True,
                    "auto_local_pdf_requires_agent_discovered_metadata": True,
                    "fallback_order": ["codex_online", "local_pdf", "placeholder"],
                    "no_solved_claim": True,
                },
            },
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "search_policy_alias",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_rejects_invalid_planner_source_hints(self):
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "bad_hints",
            "round_index": 1,
            "actions": [],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
            "planner_source_hints": [
                {
                    "schema_version": "planner_source_hint.v1",
                    "hint_id": "bad",
                    "source_ref": "doi:10.1000/bad",
                    "title": "bad",
                    "doi": "10.1000/bad",
                    "evidence_class": "planner_source_hint",
                    "allowed_use": "parent_route_proof",
                    "no_solved_claim": True,
                }
            ],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("planner_source_hint_invalid_allowed_use:0", validation["reasons"])

    def test_action_batch_validation_requires_analogical_template_policy(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "template",
            "action_type": "extract_analogical_reaction_templates",
            "rationale": "extract guarded analogical templates",
            "expected_artifact": "analogical_reaction_template_report.v1",
            "success_condition": "advisory templates are recorded",
            "payload": {},
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "template_missing_policy",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertFalse(validation["accepted"])
        self.assertIn("analogical_template_payload:0:missing_analogical_template_policy", validation["reasons"])

    def test_action_batch_validation_allows_analogical_bridge_task_triage(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "template",
            "action_type": "rank_analogical_reaction_templates",
            "rationale": "rank guarded analogical templates",
            "expected_artifact": "analogical_reaction_template_ranking.v1",
            "success_condition": "advisory templates are ranked",
            "payload": {
                "analogical_template_policy": {
                    "schema_version": "agentic_analogical_template_action_policy.v1",
                    "action_type": "rank_analogical_reaction_templates",
                    "analogy_is_advisory_only": True,
                    "no_solved_claim": True,
                    "requires_verifier": True,
                    "requires_parent_route_proof": True,
                    "production_write_blocked": True,
                    "raw_reaction_output_allowed": False,
                    "final_verdict_authority": "deterministic_parent_route_proof",
                    "allowed_use": ["planner_priority", "bridge_task_triage"],
                    "deterministic_template_validation_required": True,
                }
            },
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "template_bridge_triage",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_search_literature_action_includes_source_acquisition_policy(self):
        target = TargetInput(target_name="policy_search", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]

        batch = plan_action_batch(board, round_index=1, exhaust_round_budget=True)
        search_action = next(row for row in batch["actions"] if row["action_type"] == "search_literature")

        self.assertEqual(search_action["payload"]["source_acquisition_policy"]["fallback_order"], ["codex_online", "local_pdf", "placeholder"])
        self.assertTrue(search_action["payload"]["source_acquisition_policy"]["auto_local_pdf_requires_agent_discovered_metadata"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_search_literature_payload_includes_planner_source_hints(self):
        target = TargetInput(target_name="hinted_search", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["planner_source_hints"] = [
            {
                "schema_version": "planner_source_hint.v1",
                "hint_id": "hint1",
                "hint_key": "10.4242/plannerhint2026",
                "source_ref": "doi:10.4242/plannerhint2026",
                "title": "Planner hinted steroid synthesis",
                "doi": "10.4242/plannerhint2026",
                "pii": "",
                "url": "https://doi.org/10.4242/plannerhint2026",
                "local_pdf": "",
                "local_ref": "",
                "source_type": "planner_discovered_literature_metadata",
                "relevance_rationale": "planner found DOI during search",
                "expected_scheme_or_compound_labels": ["1", "2"],
                "extraction_task_recommendations": [],
                "evidence_class": "planner_source_hint",
                "allowed_use": "source_acquisition_hint_only",
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=1, exhaust_round_budget=True)
        search_action = next(row for row in batch["actions"] if row["action_type"] == "search_literature")

        self.assertEqual(search_action["payload"]["planner_source_hints"][0]["doi"], "10.4242/plannerhint2026")
        self.assertIn("10.4242/plannerhint2026", search_action["payload"]["queries"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_source_binding_for_multi_source_extraction(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "extract",
            "action_type": "extract_visual_literature_chain",
            "rationale": "extract a specific source",
            "expected_artifact": "visual_literature_chain.v1",
            "success_condition": "one source is extracted",
            "payload": {},
        }
        board = {
            "literature_evidence": {
                "source_candidates": [
                    {"source_ref": "doi:first", "doi": "10.1/first", "local_pdf": "/tmp/first.pdf"},
                    {"source_ref": "doi:second", "doi": "10.1/second", "local_pdf": "/tmp/second.pdf"},
                ],
                "pdf_structure_evidence": [
                    _rendered_pdf_evidence(source_ref="doi:first", pdf_path="/tmp/first.pdf")
                ],
            }
        }
        unbound_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "multi_source",
            "round_index": 1,
            "actions": [action],
        }
        bound_batch = {
            **unbound_batch,
            "actions": [{**action, "payload": {"source_ref": "doi:first"}}],
        }
        unrendered_batch = {
            **unbound_batch,
            "actions": [{**action, "payload": {"source_ref": "doi:second"}}],
        }

        unbound_validation = validate_action_batch(unbound_batch, blackboard=board)
        bound_validation = validate_action_batch(bound_batch, blackboard=board)
        unrendered_validation = validate_action_batch(unrendered_batch, blackboard=board)

        self.assertFalse(unbound_validation["accepted"])
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:extract_visual_literature_chain",
            unbound_validation["reasons"],
        )
        self.assertTrue(bound_validation["accepted"], bound_validation["reasons"])
        self.assertFalse(unrendered_validation["accepted"])
        self.assertIn(
            "extract_visual_literature_chain_requires_rendered_pdf_evidence:0",
            unrendered_validation["reasons"],
        )

    def test_action_batch_validation_allows_single_source_extraction_default(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "extract",
            "action_type": "extract_pdf_literature_structures",
            "rationale": "extract the only available source",
            "expected_artifact": "literature_pdf_structure_evidence.v1",
            "success_condition": "single source is rendered",
            "payload": {},
        }
        board = {
            "literature_evidence": {
                "source_candidates": [
                    {"source_ref": "doi:only", "doi": "10.1/only", "local_pdf": "/tmp/only.pdf"},
                ]
            }
        }
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "single_source",
            "round_index": 1,
            "actions": [action],
        }

        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_action_batch_validation_requires_chain_binding_for_multi_chain_compile(self):
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "compile",
            "action_type": "compile_exact_literature_rows",
            "rationale": "compile a specific visual chain",
            "expected_artifact": "literature_exact_rows.v1",
            "success_condition": "one visual chain is compiled",
            "payload": {},
        }
        board = {
            "literature_evidence": {
                "visual_chains": [
                    {"chain_id": "visual:first", "source_ref": "doi:first", "source_pdf_path": "/tmp/first.pdf"},
                    {"chain_id": "visual:second", "source_ref": "doi:second", "source_pdf_path": "/tmp/second.pdf"},
                ]
            }
        }
        unbound_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "multi_chain",
            "round_index": 1,
            "actions": [action],
        }
        bound_batch = {
            **unbound_batch,
            "actions": [{**action, "payload": {"chain_id": "visual:first"}}],
        }

        unbound_validation = validate_action_batch(unbound_batch, blackboard=board)
        bound_validation = validate_action_batch(bound_batch, blackboard=board)

        self.assertFalse(unbound_validation["accepted"])
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:compile_exact_literature_rows",
            unbound_validation["reasons"],
        )
        self.assertTrue(bound_validation["accepted"], bound_validation["reasons"])

    def test_codex_action_planner_batch_is_used_when_enabled(self):
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "mode": "codex_test",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:disconnection",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "Codex chose to inspect target handles before any rerun.",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "advisory hypotheses are recorded",
                    "payload": {},
                }
            ],
            "planner_source_hints": [
                {
                    "schema_version": "planner_source_hint.v1",
                    "hint_id": "planner_hint_1",
                    "source_ref": "doi:10.4242/plannerhint2026",
                    "title": "Planner hinted target-proximal synthesis",
                    "doi": "10.4242/plannerhint2026",
                    "pii": "",
                    "url": "https://doi.org/10.4242/plannerhint2026",
                    "local_pdf": "",
                    "local_ref": "",
                    "source_type": "planner_discovered_literature_metadata",
                    "relevance_rationale": "Codex planner found a traceable source lead while choosing actions.",
                    "expected_scheme_or_compound_labels": ["1", "2"],
                    "extraction_task_recommendations": ["search_literature"],
                    "evidence_class": "planner_source_hint",
                    "allowed_use": "source_acquisition_hint_only",
                    "no_solved_claim": True,
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        expected_snapshot_ref = ""
        snapshot_context_schema = ""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            expected_snapshot_ref = str(run_dir / "codex_action_planner_blackboard_round_1.json")
            result = run_agentic_blackboard_controller(
                target_name="codex_plan",
                target_smiles="CCO",
                output_dir=run_dir,
                max_rounds=1,
                mock_tool_results={"codex_action_planner": codex_batch},
            )
            self.assertTrue((run_dir / "action_batch_round_1.json").exists())
            self.assertTrue((run_dir / "action_batch_validation_round_1.json").exists())
            bundle_artifacts = result["artifact_bundle"]["artifacts"]
            self.assertIn("agent_action_batch_round_1", bundle_artifacts)
            self.assertIn("agent_action_batch_validation_round_1", bundle_artifacts)
            self.assertEqual(bundle_artifacts["agent_action_batch_round_1"]["artifact_type"], "AgentActionBatch")
            self.assertEqual(
                bundle_artifacts["agent_action_batch_round_1"]["validation_ref"],
                str(run_dir / "action_batch_validation_round_1.json"),
            )
            validation_keys = {
                row.get("artifact_key")
                for row in result["artifact_bundle"]["validations"]
                if row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
            }
            self.assertIn("agent_action_batch_round_1", validation_keys)
            self.assertIn("agent_action_batch_validation_round_1", validation_keys)
            snapshot = json.loads(Path(expected_snapshot_ref).read_text(encoding="utf-8"))
            snapshot_context_schema = str((snapshot.get("planner_context") or {}).get("schema_version") or "")

        batch = result["action_batches"][0]
        self.assertEqual(batch["mode"], "codex_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(
            batch["codex_action_planner"]["blackboard_snapshot_ref"],
            expected_snapshot_ref,
        )
        self.assertEqual(snapshot_context_schema, "codex_action_planner_context.v1")
        self.assertEqual(batch["actions"][0]["action_type"], "generate_disconnection_hypotheses")
        self.assertIn("web_search", batch["codex_action_planner"]["tool_policy"]["allowed_tools"])
        self.assertGreater(batch["codex_action_planner"]["tool_policy"]["max_tool_calls"], 0)
        self.assertTrue(batch["codex_action_planner"]["tool_policy"]["cli_search_enabled"])
        self.assertEqual(batch["planner_source_hints"][0]["allowed_use"], "source_acquisition_hint_only")
        self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])
        self.assertEqual(
            result["agent_blackboard"]["literature_evidence"]["planner_source_hints"][0]["doi"],
            "10.4242/plannerhint2026",
        )
        self.assertEqual(result["agent_blackboard"]["literature_evidence"]["source_candidates"], [])
        self.assertEqual(result["agent_blackboard"]["literature_evidence"]["source_lifecycle"][0]["stage"], "planner_hint")
        self.assertEqual(
            result["agent_blackboard"]["literature_evidence"]["source_lifecycle"][0]["next_recommended_stage"],
            "search_literature",
        )
        planner_history = result["agent_blackboard"]["planner_history"]
        self.assertEqual(len(planner_history), 1)
        self.assertEqual(planner_history[0]["mode"], "codex_blackboard_planner")
        self.assertEqual(planner_history[0]["planner_source_hint_count"], 1)
        self.assertTrue(planner_history[0]["codex_action_planner"]["attempted"])
        self.assertFalse(planner_history[0]["codex_action_planner"]["fallback_used"])
        self.assertIn("web_search", planner_history[0]["codex_action_planner"]["tool_policy"]["allowed_tools"])
        self.assertEqual(
            planner_history[0]["codex_action_planner"]["blackboard_snapshot_ref"],
            expected_snapshot_ref,
        )
        self.assertEqual(result["agent_blackboard"]["budget_state"]["codex_action_planner_runs"], 1)
        self.assertIn("codex_action_planner_round_1", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("codex_action_planner_blackboard_snapshot_round_1", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("agent_action_batch_round_1", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("agent_action_batch_validation_round_1", result["agent_blackboard"]["artifact_refs"])
        capability_checks = {
            row["requirement_id"]: row
            for row in result["artifact_bundle"]["artifacts"]["agentic_capability_audit"]["payload"]["requirement_checks"]
        }
        planner_check = capability_checks["planner_decision_history_audited"]
        self.assertTrue(planner_check["accepted"], planner_check["reasons"])
        self.assertIn("codex_snapshot_context_count:1", planner_check["evidence"])
        self.assertIn("codex_snapshot_payload_requirement_count:1", planner_check["evidence"])
        self.assertIn("codex_snapshot_tool_policy_count:1", planner_check["evidence"])
        self.assertIn("codex_history_tool_policy_count:1", planner_check["evidence"])

    def test_codex_planner_snapshot_includes_derived_context_for_pending_sources_and_transitions(self):
        target = TargetInput(target_name="planner_context", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["source_discovery_mode"] = "codex_online+local_pdf_cache"
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:first",
                "doi": "10.1000/first",
                "local_pdf": "/tmp/first.pdf",
                "source_discovery_mode": "codex_online+local_pdf_cache",
                "local_pdf_match": {"match_basis": "doi", "agent_discovered_doi": "10.1000/first"},
                "local_pdf_index": {"match_policy": "agent_discovered_metadata_required"},
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:second",
                "doi": "10.1000/second",
                "local_pdf": "/tmp/second.pdf",
                "source_discovery_mode": "codex_online+local_pdf_cache",
                "local_pdf_match": {"match_basis": "doi", "agent_discovered_doi": "10.1000/second"},
                "local_pdf_index": {"match_policy": "agent_discovered_metadata_required"},
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:third",
                "doi": "10.1000/third",
                "url": "https://doi.org/10.1000/third",
                "local_pdf": "",
                "source_discovery_mode": "codex_online",
                "access_status": "metadata_only",
            },
        ]
        board["literature_evidence"]["local_pdf_proxy_requests"] = [
            {
                "schema_version": "local_pdf_proxy_request.v1",
                "request_id": "req-third",
                "source_ref": "doi:third",
                "doi": "10.1000/third",
                "url": "https://doi.org/10.1000/third",
                "title": "Third",
                "content_scope": "article",
            }
        ]
        board["literature_evidence"]["source_lifecycle"] = [
            {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": "doi:10.1000/first",
                "source_ref": "doi:first",
                "title": "First",
                "doi": "10.1000/first",
                "local_pdf": "/tmp/first.pdf",
                "stage": "pdf_rendered",
                "next_recommended_stage": "extract_visual_literature_chain",
                "stage_flags": {"pdf_rendered": True, "local_pdf_available": True},
                "counts": {"source_candidates": 1, "pdf_structure_evidence": 1},
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": "doi:10.1000/second",
                "source_ref": "doi:second",
                "title": "Second",
                "doi": "10.1000/second",
                "local_pdf": "/tmp/second.pdf",
                "stage": "local_pdf_available",
                "next_recommended_stage": "extract_pdf_literature_structures",
                "stage_flags": {"local_pdf_available": True},
                "counts": {"source_candidates": 1},
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_source_lifecycle.v1",
                "source_key": "doi:10.1000/third",
                "source_ref": "doi:third",
                "title": "Third",
                "doi": "10.1000/third",
                "local_pdf": "",
                "stage": "local_pdf_proxy_requested",
                "next_recommended_stage": "await_local_pdf_proxy_download",
                "stage_flags": {"source_candidate": True, "local_pdf_proxy_requested": True},
                "counts": {"source_candidates": 1, "local_pdf_proxy_requests": 1},
                "no_solved_claim": True,
            },
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(source_ref="doi:first", pdf_path="/tmp/first.pdf")
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "useful_artifact": True,
                "stale": False,
                "blackboard_delta": {"source_candidates": 2},
                "changed_blackboard_fields": ["source_candidates"],
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 2,
                "action_type": "extract_pdf_literature_structures",
                "useful_artifact": True,
                "stale": False,
                "blackboard_delta": {"pdf_structure_evidence": 1},
                "changed_blackboard_fields": ["pdf_structure_evidence"],
            },
        ]

        context = _planner_context_summary(board)

        self.assertEqual(context["source_acquisition"]["auto_local_pdf_cache_match_count"], 2)
        self.assertEqual(context["source_acquisition"]["source_lifecycle_count"], 3)
        self.assertEqual(context["source_acquisition"]["source_lifecycle_stage_counts"]["pdf_rendered"], 1)
        self.assertEqual(context["source_acquisition"]["source_lifecycle_stage_counts"]["local_pdf_proxy_requested"], 1)
        self.assertEqual(context["source_acquisition"]["awaiting_local_pdf_proxy_count"], 1)
        self.assertEqual(context["source_acquisition"]["local_pdf_proxy_request_count"], 1)
        self.assertEqual(context["literature_processing"]["source_lifecycle"][0]["stage"], "pdf_rendered")
        self.assertEqual(
            context["literature_processing"]["pending_local_pdf_proxy_sources"][0]["source_ref"],
            "doi:third",
        )
        self.assertEqual(context["literature_processing"]["pending_pdf_extraction_sources"][0]["source_ref"], "doi:second")
        self.assertEqual(context["literature_processing"]["pending_visual_extraction_sources"][0]["source_ref"], "doi:first")
        search_requirements = context["action_payload_requirements"]["search_actions"]["search_literature"]
        self.assertIn("source_acquisition_policy", search_requirements["accepted_payload_fields"])
        self.assertTrue(search_requirements["blackboard_guidance"]["auto_local_pdf_requires_agent_discovered_metadata"])
        requirements = context["action_payload_requirements"]["source_sensitive_actions"]
        self.assertTrue(requirements["extract_pdf_literature_structures"]["currently_required"])
        self.assertTrue(requirements["extract_visual_literature_chain"]["currently_required"])
        self.assertTrue(requirements["compile_exact_literature_rows"]["currently_required"])
        self.assertIn("source_ref", requirements["extract_visual_literature_chain"]["accepted_payload_fields"])
        self.assertIn("pdf_path", requirements["extract_pdf_literature_structures"]["accepted_payload_fields"])
        self.assertIn("chain_id", requirements["compile_exact_literature_rows"]["accepted_payload_fields"])
        self.assertEqual(
            requirements["extract_pdf_literature_structures"]["binding_candidates"][0]["source_ref"],
            "doi:first",
        )
        guided_requirements = context["action_payload_requirements"]["guided_actions"]["run_guided_chemenzy"]
        self.assertIn("guided_policy_runtime_rebuild", guided_requirements["accepted_payload_fields"])
        self.assertTrue(guided_requirements["runtime_policy_rebuild"])
        self.assertTrue(guided_requirements["do_not_emit_full_policy"])
        self.assertIn("compiler_metadata.requires_verifier", guided_requirements["required_policy_safety_fields"])
        self.assertEqual(guided_requirements["blackboard_guidance"]["exact_row_count"], 0)
        child_requirements = context["action_payload_requirements"]["child_expansion_actions"]["expand_child_target"]
        self.assertIn("subgoal_targets", child_requirements["accepted_payload_fields"])
        self.assertIn("child_route_cannot_promote_parent", child_requirements["required_target_fields"])
        self.assertTrue(child_requirements["blackboard_guidance"]["parent_proof_required_after_child_run"])
        stitch_requirements = context["action_payload_requirements"]["stitch_actions"]["stitch_parent_route"]
        self.assertIn("proof_binding", stitch_requirements["accepted_payload_fields"])
        self.assertIn("exact_literature_row_ids", stitch_requirements["required_binding_fields"])
        self.assertEqual(stitch_requirements["blackboard_guidance"]["final_verdict_authority"], "deterministic_parent_route_proof")
        template_requirements = context["action_payload_requirements"]["analogical_template_actions"][
            "extract_analogical_reaction_templates"
        ]
        self.assertIn("analogical_template_policy", template_requirements["accepted_payload_fields"])
        self.assertIn("requires_parent_route_proof", template_requirements["required_policy_fields"])
        self.assertTrue(template_requirements["blackboard_guidance"]["analogy_is_advisory_only"])
        self.assertEqual(context["recent_blackboard_transitions"][-1]["blackboard_delta"]["pdf_structure_evidence"], 1)
        self.assertFalse(context["safety_boundaries"]["planner_can_emit_solved"])

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = _write_codex_blackboard_snapshot(board, run_dir=Path(tmp), round_index=3)
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            task = _codex_action_planner_task(
                blackboard=board,
                round_index=3,
                run_dir=Path(tmp),
                snapshot_path=snapshot_path,
            )

        self.assertEqual(snapshot["planner_context"]["schema_version"], "codex_action_planner_context.v1")
        self.assertEqual(
            snapshot["planner_context"]["action_payload_requirements"]["schema_version"],
            "codex_action_payload_requirements.v1",
        )
        self.assertEqual(
            snapshot["planner_context"]["planner_tool_policy"]["schema_version"],
            "codex_action_planner_tool_policy.v1",
        )
        self.assertEqual(snapshot["blackboard"]["schema_version"], "codex_action_planner_blackboard_handoff.v1")
        self.assertIn("target_profile", snapshot["blackboard"])
        self.assertIn("evidence_board", snapshot["blackboard"])
        self.assertIn("route_board", snapshot["blackboard"])
        self.assertIn("decision_board", snapshot["blackboard"])
        self.assertNotIn("artifact_refs", snapshot["blackboard"])
        self.assertIn("planner_context", snapshot)
        self.assertIn("action_payload_requirements", task.objective)
        self.assertIn("source_ref", task.objective)
        self.assertIn("chain_id", task.objective)
        self.assertIn("analogical_template_policy", task.objective)
        self.assertEqual(task.budget.reasoning_effort, "medium")
        self.assertIn("Planner tool policy", task.objective)
        self.assertIn("web_search", task.allowed_tools)
        self.assertGreater(task.budget.max_tool_calls, 0)
        self.assertTrue(_task_allows_cli_search(task))

    def test_codex_planner_handoff_exposes_route_anchor_opportunities_for_agent_choice(self):
        target = TargetInput(target_name="atorvastatin", target_smiles=ATORVASTATIN_FREE_ACID_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=6)
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:process",
                "doi": "10.1186/s13065-015-0082-7",
                "title": "An improved kilogram-scale preparation of atorvastatin calcium",
                "local_pdf": "/tmp/atorvastatin_process.pdf",
            }
        ]
        board["literature_evidence"]["process_evidence_rows"] = [
            {
                "schema_version": "literature_process_evidence_row.v1",
                "row_id": "process_evidence:atorvastatin",
                "source_ref": "doi:process",
                "process_type": "small_molecule_process_route",
                "substrate_or_feedstock_labels": ["advanced ketal ester intermediate 4"],
                "endpoint_labels": ["atorvastatin calcium"],
                "biocatalyst_or_process_labels": ["Paal-Knorr pyrrole construction", "ester hydrolysis"],
                "not_exact_literature_segment": True,
                "no_solved_claim": True,
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:intermediate_4",
                "label": "advanced ketal ester intermediate 4",
                "source_ref": "doi:process",
                "status": "open",
                "no_solved_claim": True,
            }
        ]

        context = _planner_context_summary(board)
        opportunities = context["route_anchor_opportunities"]
        process_opportunity = opportunities["opportunities"][0]

        self.assertEqual(opportunities["schema_version"], "route_anchor_opportunities.v1")
        self.assertGreaterEqual(opportunities["opportunity_count"], 2)
        self.assertEqual(process_opportunity["opportunity_type"], "process_or_literature_anchor")
        self.assertIn("advanced ketal ester intermediate 4", process_opportunity["anchor_labels"])
        self.assertIn("resolve_literature_structure_task", process_opportunity["plausible_next_actions"])
        self.assertIn("derive_broad_reaction_template", process_opportunity["plausible_next_actions"])
        self.assertIn("run_guided_chemenzy", process_opportunity["plausible_next_actions"])
        self.assertTrue(process_opportunity["no_solved_claim"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = _write_codex_blackboard_snapshot(board, run_dir=run_dir, round_index=2)
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            task = _codex_action_planner_task(
                blackboard=board,
                round_index=2,
                run_dir=run_dir,
                snapshot_path=snapshot_path,
            )

        self.assertIn("route_anchor_opportunities", snapshot["planner_context"])
        self.assertIn("route_anchor_opportunities", snapshot["blackboard"]["route_board"])
        self.assertIn("route_anchor_opportunities", task.objective)
        self.assertIn("Incomplete process/advisory/name-only anchors", task.objective)
        self.assertIn("Do not choose stop_unresolved while a route_anchor_opportunity", task.objective)

    def test_fallback_planner_stops_when_only_waiting_for_local_pdf_proxy(self):
        target = TargetInput(target_name="proxy_wait", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["target_side_disconnection_hypotheses"] = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "target_handle": "source_bridge",
                    "no_solved_claim": True,
                }
            ],
            "no_solved_claim": True,
        }
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [
            {"mode": "codex_online", "attempted": True, "accepted": True}
        ]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.5555/proxy.wait",
                "doi": "10.5555/proxy.wait",
                "url": "https://doi.org/10.5555/proxy.wait",
                "local_pdf": "",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ]
        board["literature_evidence"]["local_pdf_proxy_requests"] = [
            {
                "schema_version": "local_pdf_proxy_request.v1",
                "request_id": "proxy-wait",
                "source_ref": "doi:10.5555/proxy.wait",
                "doi": "10.5555/proxy.wait",
                "url": "https://doi.org/10.5555/proxy.wait",
                "content_scope": "article",
            }
        ]

        batch = plan_action_batch(board, round_index=2)
        validation = validate_action_batch(batch, blackboard=board)

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual([row["action_type"] for row in batch["actions"]], ["stop_unresolved"])
        self.assertEqual(batch["actions"][0]["payload"]["wait_state"], "local_pdf_proxy_requested")

    def test_codex_action_planner_tool_budget_can_be_disabled(self):
        target = TargetInput(target_name="tool_disabled", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(
                "os.environ",
                {
                    "AUTOPLANNER_CODEX_ACTION_PLANNER_ALLOWED_TOOLS": "none",
                    "AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_TOOL_CALLS": "8",
                },
            ):
                snapshot_path = _write_codex_blackboard_snapshot(board, run_dir=run_dir, round_index=1)
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                task = _codex_action_planner_task(
                    blackboard=board,
                    round_index=1,
                    run_dir=run_dir,
                    snapshot_path=snapshot_path,
                )

        policy = snapshot["planner_context"]["planner_tool_policy"]
        self.assertEqual(policy["allowed_tools"], [])
        self.assertEqual(policy["max_tool_calls"], 0)
        self.assertFalse(policy["cli_search_enabled"])
        self.assertEqual(task.allowed_tools, [])
        self.assertEqual(task.budget.max_tool_calls, 0)
        self.assertFalse(_task_allows_cli_search(task))
        self.assertEqual(task.budget.max_tool_calls, 0)
        self.assertFalse(_task_allows_cli_search(task))
        self.assertEqual(task.budget.reasoning_effort, "medium")

    def test_codex_action_planner_uses_supported_explicit_model_not_cli_default(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("AUTOPLANNER_CODEX_ACTION_PLANNER_MODEL", None)
            os.environ.pop("AUTOPLANNER_CODEX_MODEL", None)
            self.assertEqual(_planner_model(), "gpt-5.5")
        with patch.dict(
            "os.environ",
            {"AUTOPLANNER_CODEX_ACTION_PLANNER_MODEL": "test-planner-model"},
        ):
            self.assertEqual(_planner_model(), "test-planner-model")

    def test_blackboard_records_codex_planner_fallback_as_planner_note(self):
        target = TargetInput(target_name="planner_fallback", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 1,
            "mode": "deterministic_policy_fallback_after_codex_planner",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "fallback:stop",
                    "action_type": "stop_unresolved",
                    "rationale": "fallback",
                    "expected_artifact": "stop",
                    "success_condition": "stop",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
            "codex_action_planner": {
                "schema_version": "codex_action_planner_metadata.v1",
                "fallback_used": True,
                "fallback_reason": "codex_action_planner_batch_invalid",
                "record_status": "accepted_draft",
                "record_backend": "mock_output",
                "record_ref": "/tmp/codex_action_planner_run_record_round_1.json",
                "batch_validation": {"accepted": False, "reasons": ["raw_reaction_injection"]},
            },
        }
        validation = validate_action_batch(batch, blackboard=board)

        updated = update_blackboard_from_action_batch(
            board,
            action_batch=batch,
            validation=validation,
            round_index=1,
        )

        self.assertTrue(updated["planner_history"][0]["codex_action_planner"]["attempted"])
        self.assertTrue(updated["planner_history"][0]["codex_action_planner"]["fallback_used"])
        self.assertEqual(updated["budget_state"]["codex_action_planner_runs"], 1)
        self.assertEqual(updated["current_belief"]["planner_notes"][0]["reason"], "codex_action_planner_batch_invalid")
        self.assertEqual(
            updated["artifact_refs"]["codex_action_planner_round_1"],
            "/tmp/codex_action_planner_run_record_round_1.json",
        )

    def test_capability_audit_rejects_codex_snapshot_without_payload_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_payload_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_tool_policy(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": [
                    "initial_probe",
                    "search_mode",
                    "max_steps",
                    "guided_policy_runtime_rebuild",
                ],
            }
        }
        child_actions = {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
            }
        }
        stitch_actions = {
            "stitch_parent_route": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["proof_binding", "proof_policy", "analogy_refs"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                                "child_expansion_actions": child_actions,
                                "stitch_actions": stitch_actions,
                                "analogical_template_actions": _test_analogical_template_requirements(),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_tool_policy:0", check["reasons"])
        self.assertIn("codex_planner_history_missing_tool_policy:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_search_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_search_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_guided_payload_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_guided_action_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_child_expansion_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": [
                    "initial_probe",
                    "search_mode",
                    "max_steps",
                    "guided_policy_runtime_rebuild",
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_child_expansion_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_stitch_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": [
                    "initial_probe",
                    "search_mode",
                    "max_steps",
                    "guided_policy_runtime_rebuild",
                ],
            }
        }
        child_actions = {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                                "child_expansion_actions": child_actions,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_stitch_requirements:0", check["reasons"])

    def test_capability_audit_rejects_codex_snapshot_without_analogical_template_requirements(self):
        source_sensitive = _test_source_sensitive_requirements()
        guided_actions = {
            "run_guided_chemenzy": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": [
                    "initial_probe",
                    "search_mode",
                    "max_steps",
                    "guided_policy_runtime_rebuild",
                ],
            }
        }
        child_actions = {
            "expand_child_target": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["subgoal_targets", "child_targets"],
            }
        }
        stitch_actions = {
            "stitch_parent_route": {
                "currently_required_when_selected": True,
                "accepted_payload_fields": ["proof_binding", "proof_policy", "analogy_refs"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshot_path = run_dir / "codex_action_planner_blackboard_round_1.json"
            record_path = run_dir / "codex_action_planner_run_record_round_1.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "codex_action_planner_blackboard_snapshot.v1",
                        "planner_context": {
                            "schema_version": "codex_action_planner_context.v1",
                            "no_solved_claim": True,
                            "action_payload_requirements": {
                                "schema_version": "codex_action_payload_requirements.v1",
                                "search_actions": _test_search_requirements(),
                                "source_sensitive_actions": source_sensitive,
                                "guided_actions": guided_actions,
                                "child_expansion_actions": child_actions,
                                "stitch_actions": stitch_actions,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            record_path.write_text("{}", encoding="utf-8")
            board = {
                "target_profile": {"valid": True},
                "planner_history": [
                    {
                        "codex_action_planner": {
                            "attempted": True,
                            "blackboard_snapshot_ref": str(snapshot_path),
                            "record_ref": str(record_path),
                        }
                    }
                ],
            }

            check = _capability_check_planner_history(
                board,
                [{"schema_version": "agent_action_batch.v1", "actions": []}],
                run_dir=run_dir,
            )

        self.assertFalse(check["accepted"])
        self.assertIn("codex_planner_snapshot_missing_analogical_template_requirements:0", check["reasons"])

    def test_codex_action_planner_rejects_solved_or_raw_reaction_and_falls_back(self):
        target = TargetInput(target_name="bad_codex_plan", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        bad_codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "bad:route",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "bad",
                    "expected_artifact": "bad",
                    "success_condition": "bad",
                    "route_status": "solved",
                    "payload": {"rxn_smiles": "CCO>>CC=O"},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        snapshot_exists = False
        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=bad_codex_batch,
            )
            snapshot_exists = Path(batch["codex_action_planner"]["blackboard_snapshot_ref"]).is_file()

        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertTrue(batch["codex_action_planner"]["fallback_used"])
        self.assertTrue(snapshot_exists)
        self.assertIn(
            batch["codex_action_planner"]["fallback_reason"],
            {"codex_action_planner_worker_rejected", "codex_action_planner_batch_invalid"},
        )
        self.assertNotEqual(batch["actions"][0]["action_type"], "run_guided_chemenzy")

    def test_codex_action_planner_can_fail_closed_without_scientific_fallback(self):
        target = TargetInput(target_name="fail_closed_codex_plan", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        unsafe_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "unsafe:route",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "unsafe",
                    "expected_artifact": "unsafe",
                    "success_condition": "unsafe",
                    "route_status": "solved",
                    "payload": {"rxn_smiles": "CCO>>CC=O"},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=unsafe_batch,
                allow_deterministic_fallback=False,
            )

        self.assertEqual(batch["mode"], "codex_planner_fail_closed")
        self.assertEqual([row["action_type"] for row in batch["actions"]], ["stop_unresolved"])
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertTrue(batch["codex_action_planner"]["fail_closed"])
        self.assertFalse(
            batch["actions"][0]["payload"]["deterministic_scientific_fallback_used"]
        )

    def test_codex_action_planner_requires_source_binding_for_multi_source_actions(self):
        target = TargetInput(target_name="codex_multi_source", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "doi:first", "doi": "10.1/first", "local_pdf": "/tmp/first.pdf"},
            {"source_ref": "doi:second", "doi": "10.1/second", "local_pdf": "/tmp/second.pdf"},
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:visual",
                    "action_type": "extract_visual_literature_chain",
                    "rationale": "extract the useful source",
                    "expected_artifact": "visual_literature_chain.v1",
                    "success_condition": "a chain is extracted",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertTrue(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(batch["codex_action_planner"]["fallback_reason"], "codex_action_planner_batch_invalid")
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:extract_visual_literature_chain",
            batch["codex_action_planner"]["batch_validation"]["reasons"],
        )
        self.assertNotEqual(batch["actions"][0]["action_type"], "extract_visual_literature_chain")

    def test_codex_action_planner_repairs_visual_payload_timeout_and_single_source_binding(self):
        target = TargetInput(target_name="codex_visual_repair", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:single",
                "doi": "10.1/single",
                "title": "Single available source",
                "local_pdf": "/tmp/single.pdf",
                "expected_scheme_or_compound_labels": ["target", "precursor"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(source_ref="doi:single", pdf_path="/tmp/single.pdf")
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:visual",
                    "action_type": "extract_visual_literature_chain",
                    "rationale": "extract the available source",
                    "expected_artifact": "visual_literature_chain.v1",
                    "success_condition": "a chain is extracted",
                    "payload": {"timeout_s": 0},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        batch = _normalize_codex_batch(codex_batch, blackboard=board, round_index=1)
        payload = batch["actions"][0]["payload"]

        self.assertEqual(payload["source_ref"], "doi:single")
        self.assertEqual(payload["pdf_path"], "/tmp/single.pdf")
        self.assertGreaterEqual(payload["timeout_s"], 120)
        self.assertIn("target", payload["expected_labels"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_literature_search_policy_from_blackboard(self):
        target = TargetInput(target_name="codex_search_policy", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:search",
                    "action_type": "search_literature",
                    "rationale": "search literature",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "source candidates are recorded",
                    "payload": {
                        "queries": [
                            {"query_id": "q1", "query": "ethanol synthesis DOI"},
                            "{'query_id': 'q2', 'query': 'ethanol retrosynthesis source'}",
                        ]
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["queries"], ["ethanol synthesis DOI", "ethanol retrosynthesis source"])
        self.assertTrue(payload["source_acquisition_policy"]["codex_online_first"])
        self.assertEqual(payload["source_acquisition_policy"]["fallback_order"], ["codex_online", "local_pdf", "placeholder"])
        self.assertTrue(payload["source_acquisition_policy"]["no_solved_claim"])
        self.assertTrue(payload["codex_payload_repair"]["completed_from_blackboard"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_does_not_retry_timeout_worker(self):
        class TimeoutRecord:
            status = "timeout"
            backend = "codex_cli"
            output_artifact = {}
            output_validation = {"accepted": False, "reasons": ["timeout"]}

            def to_dict(self):
                return {
                    "schema_version": "worker_run_record.v1",
                    "status": self.status,
                    "backend": self.backend,
                    "output_artifact": self.output_artifact,
                    "output_validation": self.output_validation,
                    "timed_out": True,
                    "stderr": "worker timeout after 5s",
                }

        target = TargetInput(target_name="codex_timeout", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"AUTOPLANNER_CODEX_ACTION_PLANNER_MAX_WORKER_ATTEMPTS": "3"},
        ), patch(
            "cascade_planner.harness.codex_action_planner.run_codex_worker",
            return_value=TimeoutRecord(),
        ) as run_worker:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
            )

        self.assertEqual(run_worker.call_count, 1)
        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertEqual(batch["codex_action_planner"]["record_status"], "timeout")
        self.assertEqual(batch["codex_action_planner"]["fallback_reason"], "codex_action_planner_worker_rejected")

    def test_codex_action_planner_drops_failure_critic_without_failure_evidence(self):
        target = TargetInput(target_name="fresh_steroid", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=1,
            budget_limits={"max_scout_calls": 0},
        )
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "fresh_steroid",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:critic",
                    "action_type": "build_failure_critic_report",
                    "rationale": "record missing scout budget",
                    "expected_artifact": "failure_critic_report.v1",
                    "success_condition": "blocker recorded",
                    "payload": {"no_solved_claim": True},
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:objectives",
                    "action_type": "classify_route_objectives",
                    "rationale": "classify steroid route objectives",
                    "expected_artifact": "route_objective_summary.v1",
                    "success_condition": "objectives available",
                    "payload": {"no_solved_claim": True},
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:disconnections",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "generate bounded target-side hypotheses",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "hypotheses available",
                    "payload": {"no_solved_claim": True},
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertEqual(batch["mode"], "codex_blackboard_planner_repaired")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertNotIn("build_failure_critic_report", action_types)
        self.assertEqual(action_types, ["classify_route_objectives", "generate_disconnection_hypotheses"])
        self.assertIn(
            "failure_critic_requires_failure_evidence:0",
            batch["codex_action_planner"]["initial_validation"]["reasons"],
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_drops_pdf_extraction_without_local_pdf_binding(self):
        target = TargetInput(target_name="source_no_pdf", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=2)
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "cand:source_no_pdf",
                "source_ref": "doi:10.example/source-no-pdf",
                "doi": "10.example/source-no-pdf",
                "title": "Source without local PDF binding",
                "url": "https://example.org/source-no-pdf",
                "local_pdf": "",
                "no_solved_claim": True,
            }
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "source_no_pdf",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:pdf",
                    "action_type": "extract_pdf_literature_structures",
                    "rationale": "extract the source PDF",
                    "expected_artifact": "literature_pdf_structure_evidence.v1",
                    "success_condition": "pdf rendered",
                    "payload": {
                        "source_ref": "doi:10.example/source-no-pdf",
                        "source_title": "Source without local PDF binding",
                        "no_solved_claim": True,
                    },
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:objectives",
                    "action_type": "classify_route_objectives",
                    "rationale": "classify route objectives while waiting for PDF",
                    "expected_artifact": "route_objective_summary.v1",
                    "success_condition": "objectives available",
                    "payload": {"no_solved_claim": True},
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        initial_validation = validate_action_batch(codex_batch, blackboard=board)
        self.assertFalse(initial_validation["accepted"])
        self.assertIn("extract_pdf_literature_structures_requires_pdf_binding:0", initial_validation["reasons"])

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn(batch["mode"], {"codex_blackboard_planner", "codex_blackboard_planner_repaired"})
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertNotIn("extract_pdf_literature_structures", action_types)
        self.assertEqual(action_types, ["classify_route_objectives"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_drops_premature_complex_guided_chemenzy(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:search",
                    "action_type": "search_literature",
                    "rationale": "search literature",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "source candidates are recorded",
                    "payload": {"query": "steroid target proximal synthesis"},
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:hypotheses",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "generate target-side hypotheses",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "hypotheses are recorded",
                    "payload": {},
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:guided",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "premature complex guided search",
                    "expected_artifact": "guided_chemenzy_result.v1",
                    "success_condition": "verifier feedback is recorded",
                    "payload": build_agentic_guided_payload(board),
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn(batch["mode"], {"codex_blackboard_planner", "codex_blackboard_planner_repaired"})
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertIn("search_literature", action_types)
        self.assertIn("generate_disconnection_hypotheses", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertIn(
            "guided_chemenzy_payload:2:guided_chemenzy_missing_prior_signal_for_complex_target",
            batch["codex_action_planner"]["initial_validation"]["reasons"],
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_bounded_complex_guided_probe_skeleton(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:probe",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "cheap complex-target initial probe",
                    "expected_artifact": "guided_chemenzy_probe_result.v1",
                    "success_condition": "bounded verifier feedback is recorded",
                    "payload": {
                        "initial_probe": True,
                        "search_mode": "initial_probe",
                        "max_steps": 4,
                        "chem_enzy_iterations": 6,
                        "chem_enzy_expansion_topk": 12,
                        "timeout_s": 60,
                        "max_candidates": 3,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        payload = batch["actions"][0]["payload"]
        self.assertEqual(batch["mode"], "codex_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertNotIn("search_policy", payload)
        self.assertTrue(payload["guided_policy_runtime_rebuild"])
        policy = payload["guided_policy_summary"]
        self.assertTrue(policy["source_flags"]["initial_scan_allowed"])
        self.assertEqual(policy["budget"]["max_depth"], 4)
        self.assertEqual(policy["budget"]["max_iterations"], 6)
        self.assertEqual(policy["budget"]["expansion_topk"], 12)
        self.assertEqual(policy["budget"]["timeout_s"], 60.0)
        self.assertTrue(policy["compiler_flags"]["initial_scan_probe"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_inserts_frontier_bootstrap_after_complex_probe(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
            }
        )
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "candidate_id": "cand:test",
                "source_ref": "doi:test",
                "doi": "10.example/test",
                "title": "Complex target source",
                "local_pdf": "/tmp/source.pdf",
                "no_solved_claim": True,
            }
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:pdf",
                    "action_type": "extract_pdf_literature_structures",
                    "rationale": "extract source PDF",
                    "expected_artifact": "literature_pdf_structure_evidence.v1",
                    "success_condition": "pdf evidence is recorded",
                    "payload": {"source_ref": "doi:test", "pdf_path": "/tmp/source.pdf"},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertEqual(batch["mode"], "codex_blackboard_planner_repaired")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(action_types[0], "generate_disconnection_hypotheses")
        self.assertIn("extract_pdf_literature_structures", action_types)
        self.assertIn(
            "complex_target_requires_frontier_bootstrap_after_initial_probe",
            batch["codex_action_planner"]["initial_validation"]["reasons"],
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_guided_chemenzy_policy_from_blackboard(self):
        target = TargetInput(target_name="codex_guided_policy", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:guided",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "try guided search",
                    "expected_artifact": "guided_chemenzy_result.v1",
                    "success_condition": "verifier feedback is recorded",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertNotIn("search_policy", payload)
        self.assertTrue(payload["guided_policy_runtime_rebuild"])
        self.assertTrue(payload["guided_policy_summary"]["compiler_flags"]["requires_verifier"])
        self.assertTrue(payload["guided_policy_summary"]["source_flags"]["require_target_core_retention"])
        self.assertTrue(payload["codex_payload_repair"]["completed_from_blackboard"])
        self.assertTrue(payload["codex_payload_repair"]["runtime_policy_rebuild"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_salvages_guided_probe_when_structure_task_invalid(self):
        target = TargetInput(target_name="codex_guided_salvage", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:target", "task_type": "target_proximal_bridge"}]
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:guided",
                    "action_type": "run_guided_chemenzy",
                    "rationale": "run a bounded bridge probe",
                    "expected_artifact": "guided_chemenzy_result.v1",
                    "success_condition": "verifier feedback is recorded",
                    "payload": {
                        "initial_probe": True,
                        "search_mode": "target_proximal_bridge",
                        "max_steps": 6,
                        "chem_enzy_iterations": 10,
                        "chem_enzy_expansion_topk": 20,
                        "timeout_s": 120,
                        "max_candidates": 5,
                        "no_solved_claim": True,
                    },
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:bad_resolution",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "try resolving a source label",
                    "expected_artifact": "structure resolution draft",
                    "success_condition": "label resolved or rejected",
                    "payload": {"expected_labels": ["17alpha-hydroxyprogesterone"], "no_solved_claim": True},
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_blackboard_planner_repaired")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertEqual(action_types, ["run_guided_chemenzy"])
        payload = batch["actions"][0]["payload"]
        self.assertNotIn("search_policy", payload)
        self.assertTrue(payload["guided_policy_runtime_rebuild"])
        self.assertTrue(payload["guided_policy_summary"]["source_flags"]["initial_scan_allowed"])
        self.assertEqual(payload["guided_policy_summary"]["budget"]["max_depth"], 6)
        self.assertIn(
            "resolve_literature_structure_task_payload:1:missing_task_id",
            batch["codex_action_planner"]["initial_validation"]["reasons"],
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_structure_task_from_expected_labels(self):
        target = TargetInput(target_name="codex_structure_expected_label", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1021/ja952692a",
                "title": "Total Synthesis of Baccatin III and Taxol",
                "local_pdf": "danishefsky.pdf",
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:danishefsky_baccatin_iii_1",
                "task_type": "resolve_literature_structure",
                "label": "baccatin III 1",
                "source_ref": "doi:10.1021/ja952692a",
                "source_title": "Total Synthesis of Baccatin III and Taxol",
                "status": "open",
                "no_solved_claim": True,
            }
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:resolve_baccatin",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "resolve the Danishefsky baccatin III anchor",
                    "expected_artifact": "structure resolution draft",
                    "success_condition": "label resolved or rejected",
                    "payload": {
                        "expected_labels": ["baccatin III 1"],
                        "source_ref": "doi:10.1021/ja952692a",
                        "no_solved_claim": True,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertIn(batch["mode"], {"codex_blackboard_planner", "codex_blackboard_planner_repaired"})
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual([row["action_type"] for row in batch["actions"]], ["resolve_literature_structure_task"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["task_id"], "resolve_structure:danishefsky_baccatin_iii_1")
        self.assertEqual(payload["label"], "baccatin III 1")
        self.assertEqual(payload["source_ref"], "doi:10.1021/ja952692a")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_keeps_explicit_structure_labels_distinct(self):
        target = TargetInput(target_name="codex_structure_10dab_label", target_smiles=BUFOTALIN_SMILES, family_hint="taxane")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1021/np990040k",
                "title": "A New Semisynthesis of Paclitaxel from Baccatin III",
                "local_pdf": "baloglu_kingston.pdf",
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:np990040k_baccatin_iii",
                "task_type": "resolve_literature_structure",
                "label": "baccatin III",
                "source_ref": "doi:10.1021/np990040k",
                "source_title": "A New Semisynthesis of Paclitaxel from Baccatin III",
                "status": "open",
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:np990040k_10_deacetyl_baccatin_iii",
                "task_type": "resolve_literature_structure",
                "label": "10-deacetyl baccatin III",
                "source_ref": "doi:10.1021/np990040k",
                "source_title": "A New Semisynthesis of Paclitaxel from Baccatin III",
                "status": "open",
                "no_solved_claim": True,
            },
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:resolve_10dab",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "resolve 10-DAB anchor before applying the paclitaxel semisynthesis template",
                    "expected_artifact": "structure resolution record",
                    "success_condition": "10-DAB label resolved or rejected",
                    "payload": {
                        "expected_labels": ["10-deacetyl baccatin III"],
                        "source_ref": "doi:10.1021/np990040k",
                        "no_solved_claim": True,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["task_id"], "resolve_structure:np990040k_10_deacetyl_baccatin_iii")
        self.assertEqual(payload["label"], "10-deacetyl baccatin III")
        self.assertEqual(payload["source_ref"], "doi:10.1021/np990040k")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_keeps_structure_resolution_bound_to_requested_source(self):
        target = TargetInput(target_name="codex_structure_source_bound", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1021/ja00083a066",
                "title": "A synthesis of taxol",
                "local_pdf": "holton.pdf",
            },
            {
                "source_ref": "doi:10.1021/ja952692a",
                "title": "Total Synthesis of Baccatin III and Taxol",
                "local_pdf": "danishefsky.pdf",
            },
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_10_1021_ja00083a066_taxol_1_to_lactone_carbonate_15",
                "task_type": "resolve_literature_structure",
                "label": "taxol 1 to lactone carbonate 15 continuation",
                "source_ref": "doi:10.1021/ja00083a066",
                "source_title": "A synthesis of taxol",
                "status": "open",
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_10_1021_ja952692a_1_baccatin_iii",
                "task_type": "resolve_literature_structure",
                "label": "1 baccatin III",
                "source_ref": "doi:10.1021/ja952692a",
                "source_title": "Total Synthesis of Baccatin III and Taxol",
                "status": "open",
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_10_1021_ja952692a_taxol",
                "task_type": "resolve_literature_structure",
                "label": "taxol",
                "source_ref": "doi:10.1021/ja952692a",
                "source_title": "Total Synthesis of Baccatin III and Taxol",
                "status": "open",
                "no_solved_claim": True,
            },
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 4,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "r4_resolve_danishefsky_baccatin_label_1",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "Resolve the Danishefsky baccatin label 1 before retrying exact-row compilation.",
                    "expected_artifact": "structure resolution record",
                    "success_condition": "Label resolved or failure is auditable.",
                    "payload": {
                        "expected_labels": ["taxol", "baccatin III"],
                        "source_ref": "doi:10.1021/ja952692a",
                        "source_title": "Total Synthesis of Baccatin III and Taxol",
                        "no_solved_claim": True,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=4,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["task_id"], "resolve_structure:doi_10_1021_ja952692a_1_baccatin_iii")
        self.assertEqual(payload["label"], "1 baccatin III")
        self.assertEqual(payload["source_ref"], "doi:10.1021/ja952692a")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_recovers_numeric_structure_label_from_action_id(self):
        target = TargetInput(target_name="codex_structure_numeric_label", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1021/ja952692a",
                "title": "Total Synthesis of Baccatin III and Taxol",
                "local_pdf": "danishefsky.pdf",
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_10_1021_ja952692a_49",
                "task_type": "resolve_literature_structure",
                "label": "49",
                "source_ref": "doi:10.1021/ja952692a",
                "source_title": "Total Synthesis of Baccatin III and Taxol",
                "status": "open",
                "no_solved_claim": True,
            }
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 8,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "r8_resolve_danishefsky_label_49",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "Scheme 15 label remains target-proximal and unresolved.",
                    "expected_artifact": "structure resolution record",
                    "success_condition": "Label resolved or failure is auditable.",
                    "payload": {
                        "expected_labels": [],
                        "source_ref": "doi:10.1021/ja952692a",
                        "source_title": "Total Synthesis of Baccatin III and Taxol",
                        "no_solved_claim": True,
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=8,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["task_id"], "resolve_structure:doi_10_1021_ja952692a_49")
        self.assertEqual(payload["label"], "49")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_keeps_anchor_resolution_when_visual_budget_tight(self):
        target = TargetInput(target_name="codex_structure_budget_trim", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=3,
            budget_limits={"max_visual_calls": 6},
        )
        board["budget_state"]["visual_calls"] = 5
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:10.1021/ja952692a",
                "title": "Total Synthesis of Baccatin III and Taxol",
                "local_pdf": "danishefsky.pdf",
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:danishefsky_downstream",
                "task_type": "resolve_literature_structure",
                "label": "compound 5 to downstream baccatin/taxol sequence",
                "source_ref": "doi:10.1021/ja952692a",
                "source_title": "Total Synthesis of Baccatin III and Taxol",
                "status": "open",
                "visual_budget_priority": 8,
                "no_solved_claim": True,
            },
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:danishefsky_baccatin_iii_1",
                "task_type": "resolve_literature_structure",
                "label": "baccatin III 1",
                "source_ref": "doi:10.1021/ja952692a",
                "source_title": "Total Synthesis of Baccatin III and Taxol",
                "status": "open",
                "visual_budget_priority": 0,
                "no_solved_claim": True,
            },
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:resolve_downstream",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "resolve downstream sequence",
                    "expected_artifact": "structure resolution draft",
                    "success_condition": "label resolved or rejected",
                    "payload": {
                        "expected_labels": ["compound 5 to downstream baccatin/taxol sequence"],
                        "source_ref": "doi:10.1021/ja952692a",
                        "no_solved_claim": True,
                    },
                },
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:resolve_baccatin",
                    "action_type": "resolve_literature_structure_task",
                    "rationale": "resolve baccatin III anchor",
                    "expected_artifact": "structure resolution draft",
                    "success_condition": "label resolved or rejected",
                    "payload": {
                        "expected_labels": ["baccatin III 1"],
                        "source_ref": "doi:10.1021/ja952692a",
                        "no_solved_claim": True,
                    },
                },
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_blackboard_planner_repaired")
        payloads = [row["payload"] for row in batch["actions"]]
        self.assertEqual([payload["task_id"] for payload in payloads], ["resolve_structure:danishefsky_baccatin_iii_1"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_repairs_child_target_policy(self):
        target = TargetInput(target_name="codex_child_target", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "literature_terminal_child:target", "task_type": "upstream_terminal_synthesis"}]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:child",
                    "action_type": "expand_child_target",
                    "rationale": "expand the upstream child target",
                    "expected_artifact": "route_expansion_subgoal_search_result.v1",
                    "success_condition": "child verifier feedback is recorded",
                    "payload": {"child_targets": [{"name": "ethanol child", "target_smiles": "CCO"}]},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        child = batch["actions"][0]["payload"]["subgoal_targets"][0]
        self.assertEqual(child["smiles"], "CCO")
        self.assertTrue(child["target_equivalence_audit_required"])
        self.assertTrue(child["child_route_cannot_promote_parent"])
        self.assertTrue(child["policy_runtime_rebuild"])
        self.assertTrue(child["policy_summary"]["compiler_flags"]["requires_verifier"])
        self.assertTrue(child["policy_summary"]["compiler_flags"]["child_route_cannot_promote_parent"])
        self.assertNotIn("chem_enzy_search_policy", child)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_compile_exact_rows_payload_is_whitelisted(self):
        target = TargetInput(target_name="codex_compile_compact", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "doi:10.1/source", "title": "source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "chain_id": "visual:1",
                "source_ref": "doi:10.1/source",
                "source_title": "source",
                "step_count": 1,
                "steps": [{"product_smiles": "CCO", "reactant_smiles": ["CC"]}],
            }
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:compile",
                    "action_type": "compile_exact_literature_rows",
                    "rationale": "compile visual rows",
                    "expected_artifact": "exact rows",
                    "success_condition": "rows",
                    "payload": {
                        "source_ref": "doi:10.1/source",
                        "chem_enzy_expansion_topk": 300,
                        "chem_enzy_iterations": 200,
                        "queries": ["noise"],
                        "route_objectives": [{"objective_id": "noise"}],
                    },
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["source_ref"], "doi:10.1/source")
        self.assertEqual(payload["chain_id"], "visual:1")
        self.assertNotIn("chem_enzy_expansion_topk", payload)
        self.assertNotIn("queries", payload)
        self.assertNotIn("route_objectives", payload)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_action_planner_requires_stitch_parent_route_binding(self):
        target = TargetInput(target_name="codex_stitch", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:ethanol"}]
        board["action_history"] = [
            {"round_index": 1, "action_type": "expand_child_target", "useful_artifact": True, "stale": False}
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 2,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:stitch",
                    "action_type": "stitch_parent_route",
                    "rationale": "prove parent route",
                    "expected_artifact": "stitched_parent_route_proof.v1",
                    "success_condition": "parent proof clauses are recorded",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "deterministic_policy_fallback_after_codex_planner")
        self.assertTrue(batch["codex_action_planner"]["fallback_used"])
        self.assertEqual(batch["codex_action_planner"]["fallback_reason"], "codex_action_planner_batch_invalid")
        self.assertIn(
            "stitch_parent_route_payload:0:missing_proof_binding",
            batch["codex_action_planner"]["batch_validation"]["reasons"],
        )

    def test_simple_target_planner_runs_bounded_direct_guided_probe_first(self):
        target = TargetInput(
            target_name="ibuprofen",
            target_smiles="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)

        batch = plan_action_batch(board, round_index=1)

        self.assertEqual([row["action_type"] for row in batch["actions"]], ["run_guided_chemenzy"])
        payload = batch["actions"][0]["payload"]
        self.assertTrue(payload["initial_probe"])
        self.assertEqual(payload["search_mode"], "direct_parent_initial_probe")
        self.assertEqual(payload["search_policy"]["budget"]["max_depth"], 6)
        self.assertEqual(payload["search_policy"]["budget"]["max_iterations"], 10)
        self.assertEqual(payload["search_policy"]["budget"]["expansion_topk"], 20)
        self.assertTrue(validate_action_batch(batch, blackboard=board)["accepted"])

        with tempfile.TemporaryDirectory() as tmp:
            disabled_batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=False,
            )
        self.assertEqual(disabled_batch["mode"], "deterministic_policy")
        self.assertFalse(disabled_batch["codex_action_planner"]["fallback_used"])
        self.assertTrue(disabled_batch["codex_action_planner"]["planner_disabled"])

    def test_accepted_parent_verifier_warning_does_not_create_route_failure(self):
        target = TargetInput(target_name="ibuprofen", target_smiles="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)

        with tempfile.TemporaryDirectory() as tmp:
            advanced = "COC(=O)[C@@H](C)c1ccc(CC(C)C)cc1"
            stock_path = Path(tmp) / "controller_test_stock.csv"
            stock_path.write_text(
                Chem.MolToSmiles(Chem.MolFromSmiles(advanced), isomericSmiles=True) + "\n",
                encoding="utf-8",
            )
            verifier = _strict_parent_route_verifier(
                target.target_smiles,
                reactants=["CC"] * 6 + ["C", "O", "O"],
                rejected_sibling={
                    "route_rank": 1,
                    "metrics": {
                        "terminal_reactants": [advanced, "O"],
                        "terminal_stock_status": {advanced: True, "O": True},
                    },
                    "steps": [
                        {
                            "index": 0,
                            "product": target.target_smiles,
                            "reactant_smiles": [advanced, "O"],
                            "stock_status": {advanced: True, "O": True},
                        }
                    ],
                },
                custom_stock_path=stock_path,
            )
            self.assertTrue(verifier["accepted"], verifier["reasons"])
            self.assertIn("advanced_same_scaffold_terminal", verifier["warnings"])
            updated = update_blackboard_from_action(
                board,
                action={"action_id": "r1:guided", "action_type": "run_guided_chemenzy", "payload": {}},
                action_result={
                    "accepted": True,
                    "result": {
                        "schema_version": "guided_chemenzy_rerun_result.v1",
                        "accepted": True,
                        "raw_route_verifier": verifier,
                    },
                    "reasons": [],
                },
                round_index=1,
                run_dir=tmp,
            )

        self.assertEqual(updated["route_failures"], [])
        self.assertEqual(updated["current_belief"]["next_action_bias"], [])
        self.assertFalse(updated["current_belief"]["parent_route_verifier"]["accepted"])
        self.assertEqual(
            updated["current_belief"]["parent_route_verifier"]["warnings"],
            ["advanced_same_scaffold_terminal"],
        )

    @patch.dict(
        os.environ,
        {
            "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(
                _SOURCE_FIXTURES / "trusted_literature_step_registry.json"
            )
        },
    )
    def test_direct_parent_verifier_drives_deterministic_stitch_fast_path(self):
        target = TargetInput(
            target_name="ethanol",
            target_smiles="CCO",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        with tempfile.TemporaryDirectory() as tmp:
            verifier = _strict_parent_route_verifier(
                target.target_smiles,
                reactants=["CC", "O"],
            )
            self.assertTrue(verifier["accepted"], verifier["reasons"])
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r1:guided", "action_type": "run_guided_chemenzy", "payload": {}},
                action_result={
                    "accepted": True,
                    "result": {
                        "schema_version": "guided_chemenzy_rerun_result.v1",
                        "accepted": True,
                        "raw_route_verifier": verifier,
                    },
                    "reasons": [],
                },
                round_index=1,
                run_dir=tmp,
            )
            self.assertEqual(len(board["chemenzy_route_proof_banks"]), 1)
            stored_bank = board["chemenzy_route_proof_banks"][0]
            self.assertEqual(
                stored_bank["route_proof_bank"], verifier["route_proof_bank"]
            )
            self.assertTrue(stored_bank["requires_current_host_replay"])
            batch = plan_action_batch(board, round_index=2)
            self.assertEqual([row["action_type"] for row in batch["actions"]], ["stitch_parent_route"])
            payload = batch["actions"][0]["payload"]
            self.assertEqual(payload["proof_binding"]["proof_mode"], "direct_parent_route")
            self.assertFalse(payload["proof_policy"]["child_route_connectivity_required"])
            self.assertTrue(validate_action_batch(batch, blackboard=board)["accepted"])

            codex_batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=2,
                run_dir=Path(tmp),
                enabled=True,
            )

        self.assertEqual(codex_batch["mode"], "deterministic_policy_fast_path_before_codex_planner")
        self.assertFalse(codex_batch["codex_action_planner"]["fallback_used"])
        self.assertTrue(codex_batch["codex_action_planner"]["fast_path_used"])
        self.assertEqual(
            codex_batch["codex_action_planner"]["fast_path_reason"],
            "deterministic_direct_parent_route_proof_ready",
        )

    def test_codex_action_planner_repairs_analogical_template_policy(self):
        target = TargetInput(target_name="codex_template_policy", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["source_refs"] = ["doi:analog"]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:template",
                    "action_type": "extract_analogical_reaction_templates",
                    "rationale": "extract analogical templates",
                    "expected_artifact": "analogical_reaction_template_report.v1",
                    "success_condition": "advisory templates are recorded",
                    "payload": {},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=1,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        self.assertEqual(batch["mode"], "codex_blackboard_planner")
        self.assertFalse(batch["codex_action_planner"]["fallback_used"])
        policy = batch["actions"][0]["payload"]["analogical_template_policy"]
        self.assertTrue(policy["analogy_is_advisory_only"])
        self.assertTrue(policy["no_solved_claim"])
        self.assertIn("bridge_task_triage", policy["allowed_use"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_explicit_action_planner_overrides_codex_action_planner(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "override",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "override:stop",
                        "action_type": "stop_unresolved",
                        "rationale": "explicit planner override",
                        "expected_artifact": "stop marker",
                        "success_condition": "stop selected",
                        "payload": _test_search_payload("online case target proximal literature"),
                    }
                ],
            }

        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 1,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex:disconnection",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "should not run",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "should not run",
                    "payload": {},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="override",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                use_codex_action_planner=True,
                action_planner=planner,
                mock_tool_results={"codex_action_planner": codex_batch},
            )

        self.assertEqual(result["action_batches"][0]["actions"][0]["action_type"], "stop_unresolved")
        self.assertNotIn("codex_action_planner", result["action_batches"][0])

    def test_target_side_strategy_for_mla_like_target_is_advisory(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=MLA_LIKE_SMILES,
            target_name="MLA analog",
            family_hint="MLA alkaloid",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        payload = json.dumps(result, sort_keys=True)

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("aryl_ester_or_anthranilate_sidechain", handles)
        self.assertIn("imide_or_succinimide_fragment", handles)
        self.assertIn("polycyclic_cage_core", handles)
        self.assertIn("tertiary_amine", handles)
        self.assertTrue(result["no_solved_claim"])
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_target_side_strategy_for_bufotalin_recognizes_c17_pyrone_handle(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=BUFOTALIN_SMILES,
            target_name="bufotalin",
            family_hint="bufadienolide steroid C17 pyrone",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        payload = json.dumps(result, sort_keys=True)

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("bufadienolide_c17_pyrone_sidechain", handles)
        self.assertTrue(result["no_solved_claim"])
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_target_side_strategy_for_atorvastatin_does_not_enter_cage_or_steroid_mode(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            target_name="atorvastatin",
            family_hint="statin synthetic atorvastatin Paal-Knorr process route",
            case_id="atorvastatin",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        selected_objectives = {
            row["objective_type"]
            for row in result["route_objective_summary"]["selected_objectives"]
        }

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertNotIn("polycyclic_cage_core", handles)
        self.assertNotIn("steroid_core", handles)
        self.assertIn("protecting_group_level_transformations", handles)
        self.assertIn("advanced_intermediate_anchor", selected_objectives)
        self.assertIn("literature_known_scaffold_anchor", selected_objectives)

    def test_refresh_target_priors_removes_stale_atorvastatin_cage_bias(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            family_hint="statin synthetic atorvastatin Paal-Knorr process route",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["target_side_disconnection_hypotheses"] = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "target": {"handles": ["polycyclic_cage_core", "steroid_core"]},
            "hypotheses": [
                {
                    "hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                    "target_handle": "polycyclic_cage_core",
                    "must_preserve_substructure": ["polycyclic_cage_core"],
                }
            ],
        }
        board["analogical_hypotheses"] = [
            {
                "hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                "target_handle": "polycyclic_cage_core",
            },
            {"hypothesis_id": "external_hypothesis", "target_handle": "external"},
        ]
        board["analogical_hypothesis_ranking"] = {
            "selected_hypotheses": [{"hypothesis_id": "target_side_polycyclic_cage_core_preservation"}]
        }
        board["bridge_tasks"] = [
            {
                "task_id": "bridge:polycyclic_cage_core",
                "source_hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                "target_handle": "polycyclic_cage_core",
            },
            {
                "task_id": "semisynthesis_bridge:resolved_anchor",
                "source_hypothesis_id": "resolved_structure_semisynthesis_anchor",
                "target_handle": "semisynthesis_from_source_resolved_intermediate",
            },
        ]
        board["literature_evidence"]["process_evidence_rows"] = [
            {
                "row_id": "process:atorvastatin",
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "substrate_or_feedstock_labels": ["advanced ketal ester intermediate 4"],
            }
        ]
        board["literature_evidence"]["resolved_structures"] = [
            {
                "accepted": True,
                "label": "2",
                "smiles": "CC(C)C(=O)C(C(=O)Nc1ccccc1)C(c1ccccc1)C(=O)c1ccc(F)cc1",
            }
        ]

        refreshed = refresh_target_derived_blackboard_priors(
            board,
            target_input=target.to_dict(),
            preflight=preflight,
        )
        handles = _target_handles_from_blackboard(refreshed)
        bridge_handles = {str(row.get("target_handle") or "") for row in refreshed.get("bridge_tasks") or []}
        guided_refs = "\n".join(
            str(item)
            for item in (build_agentic_guided_payload(refreshed).get("chem_enzy_search_policy") or {}).get("evidence_refs") or []
        )

        self.assertIn("blackboard_migrations", refreshed)
        self.assertNotIn("polycyclic_cage_core", handles)
        self.assertNotIn("steroid_core", handles)
        self.assertIn("protecting_group_level_transformations", handles)
        self.assertNotIn("polycyclic_cage_core", bridge_handles)
        self.assertIn("semisynthesis_from_source_resolved_intermediate", bridge_handles)
        self.assertEqual(refreshed["analogical_hypothesis_ranking"], {})
        self.assertEqual(len(refreshed["literature_evidence"]["process_evidence_rows"]), 1)
        self.assertEqual(len(refreshed["literature_evidence"]["resolved_structures"]), 1)
        self.assertNotIn("target_side_polycyclic_cage_core_preservation", guided_refs)

    def test_resume_plan_refreshes_stale_target_priors_before_planning(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            family_hint="statin synthetic atorvastatin Paal-Knorr process route",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=2)
        board["target_side_disconnection_hypotheses"] = {
            "schema_version": "target_side_disconnection_hypotheses.v1",
            "target": {"handles": ["polycyclic_cage_core"]},
            "hypotheses": [
                {
                    "hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                    "target_handle": "polycyclic_cage_core",
                    "must_preserve_substructure": ["polycyclic_cage_core"],
                }
            ],
        }
        board["analogical_hypotheses"] = [
            {
                "hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                "target_handle": "polycyclic_cage_core",
            }
        ]
        board["analogical_hypothesis_ranking"] = {
            "selected_hypotheses": [{"hypothesis_id": "target_side_polycyclic_cage_core_preservation"}]
        }
        board["bridge_tasks"] = [
            {
                "task_id": "bridge:polycyclic_cage_core",
                "source_hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                "target_handle": "polycyclic_cage_core",
            }
        ]
        board["retrosynthetic_proposals"] = [
            {
                "proposal_id": "proposal:stale_cage",
                "precursor_smiles": "CCO",
                "recursive_expandable": True,
                "evidence_refs": ["target_side_polycyclic_cage_core_preservation"],
            }
        ]
        board["budget_state"]["chemenzy_runs"] = 1
        board["budget_state"]["max_chemenzy_runs"] = 2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_blackboard.json").write_text(json.dumps(board), encoding="utf-8")
            (root / "target_input.json").write_text(json.dumps(target.to_dict()), encoding="utf-8")
            (root / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")

            preview = resume_agentic_blackboard_run(
                root,
                plan_only=True,
                extend_exploration_budget=True,
                extra_guided_runs=1,
                extra_child_target_runs=1,
                extra_codex_research_runs=0,
                extra_scout_calls=0,
                extra_visual_calls=0,
                extra_template_actions=0,
            )

        action_payload = json.dumps(preview["action_batch"]["actions"], sort_keys=True)
        self.assertTrue(preview["accepted"], preview["validation"]["reasons"])
        self.assertNotIn("polycyclic_cage_core", action_payload)
        self.assertNotIn("target_side_polycyclic_cage_core_preservation", action_payload)

    def test_target_side_strategy_for_9oh4hp_prefers_generic_objective_endpoints(self):
        result = build_target_side_disconnection_hypotheses(
            target_smiles=C22_9OH_4HP_SMILES,
            target_name="9-OH-4-HP",
            family_hint="9,21-dihydroxy-20-methyl-pregna-4-en-3-one steroid",
            case_id="target1_steroid",
        )
        handles = {row["target_handle"] for row in result["hypotheses"]}
        anchors = {row["anchor_id"]: row for row in result["semisynthesis_anchors"]}
        selected_objectives = {
            row["objective_type"]
            for row in result["route_objective_summary"]["selected_objectives"]
        }
        payload = json.dumps(result, sort_keys=True)

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("semisynthesis_or_biotransformation_anchor", handles)
        self.assertTrue(result["route_scope"]["de_novo_core_construction_deprioritized"])
        self.assertTrue(result["route_scope"]["objective_evidence_validation_required"])
        self.assertIn("semisynthesis_from_natural_product", selected_objectives)
        self.assertIn("biotransformation_endpoint", selected_objectives)
        self.assertIn(
            "route_objective_anchor:semisynthesis_from_natural_product:natural_product_or_feedstock_same_scaffold_pool",
            anchors,
        )
        self.assertIn(
            "route_objective_anchor:biotransformation_endpoint:same_core_biotransformation_substrate",
            anchors,
        )
        self.assertEqual(result["source_candidates"], [])
        self.assertNotIn("10.1186/s12934-021-01717-w", payload)
        self.assertTrue(result["no_solved_claim"])
        self.assertNotIn("rxn_smiles", payload)
        self.assertNotIn("reaction_smiles", payload)

    def test_planner_validates_semisynthesis_anchor_before_recursive_small_molecule_expansion(self):
        target = TargetInput(target_name="9-OH-4-HP", target_smiles=C22_9OH_4HP_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_route_expansion_subgoal_runs": 4},
        )
        result = build_target_side_disconnection_hypotheses(
            target_smiles=C22_9OH_4HP_SMILES,
            target_name="9-OH-4-HP",
            family_hint="9,21-dihydroxy-20-methyl-pregna-4-en-3-one steroid",
            case_id="target1_steroid",
        )
        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={
                    "schema_version": "agent_action.v1",
                    "action_id": "generate:semisynthesis",
                    "action_type": "generate_disconnection_hypotheses",
                    "rationale": "classify target route scope",
                    "expected_artifact": "target_side_disconnection_hypotheses.v1",
                    "success_condition": "semisynthesis anchors are recorded",
                    "payload": {},
                },
                action_result={"accepted": True, "result": result, "reasons": []},
                round_index=1,
                run_dir=tmp,
            )
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:should_wait",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "parent_smiles": C22_9OH_4HP_SMILES,
                "precursor_smiles": "CC=O",
                "recursive_depth": 1,
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "search_literature")
        self.assertEqual(batch["actions"][0]["payload"]["search_intent"], "route_objective_endpoint_validation")
        self.assertIn("biotransformation endpoint", " ".join(batch["actions"][0]["payload"]["search_queries"]))
        self.assertNotIn("10.1186/s12934-021-01717-w", " ".join(batch["actions"][0]["payload"]["search_queries"]))
        self.assertNotIn("expand_child_target", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        self.assertTrue(board["current_belief"]["constraints"]["objective_evidence_validation_required"])

    def test_failed_run_replay_generates_bridge_tasks_and_no_solved_verdict(self):
        prior_artifacts = {
            "route_verifier": {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump"}],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="MLA analog",
                target_smiles=MLA_LIKE_SMILES,
                family_hint="MLA alkaloid",
                output_dir=tmp,
                max_rounds=1,
                use_codex_action_planner=False,
                prior_artifacts=prior_artifacts,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "mla_analog",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_no_online_sources"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )
            board = json.loads((Path(tmp) / "agent_blackboard.json").read_text(encoding="utf-8"))
            audit = json.loads((Path(tmp) / "agentic_run_audit.json").read_text(encoding="utf-8"))
            bundle = json.loads((Path(tmp) / "artifact_bundle.json").read_text(encoding="utf-8"))

        action_types = [row["action_type"] for row in result["action_batches"][0]["actions"]]
        task_types = {row["task_type"] for row in board["bridge_tasks"]}
        self.assertIn("generate_disconnection_hypotheses", action_types)
        self.assertIn("build_failure_critic_report", action_types)
        self.assertIn("search_literature", action_types)
        self.assertIn("target_proximal_bridge_required", task_types)
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")
        self.assertEqual(audit["artifact_type"], "AgenticRunAudit")
        self.assertEqual(audit["payload"]["schema_version"], "agentic_blackboard_run_audit.v1")
        self.assertEqual(audit["payload"]["final_verdict"]["verdict"], result["final_verdict"]["verdict"])
        self.assertTrue(audit["payload"]["safety_invariants"]["parent_proof_required_for_solved"])
        self.assertIn("no_deterministic_parent_route_proof", audit["payload"]["unresolved_reasons"])
        self.assertEqual(audit["payload"]["source_acquisition_summary"]["fallback_order"], ["codex_online", "local_pdf", "placeholder"])
        self.assertTrue(audit["payload"]["source_acquisition_summary"]["codex_online_attempted"])
        self.assertTrue(audit["payload"]["source_acquisition_summary"]["placeholder_used"])
        self.assertEqual(audit["payload"]["source_acquisition_summary"]["real_source_count"], 0)
        followup_types = {row["task_type"] for row in audit["payload"]["followup_tasks"]}
        self.assertIn("continue_bridge_task", followup_types)
        transition = audit["payload"]["blackboard_transition_summary"]
        self.assertEqual(transition["schema_version"], "agent_blackboard_transition_summary.v1")
        self.assertEqual(transition["action_transition_count"], len(board["action_history"]))
        self.assertGreaterEqual(transition["changed_transition_count"], 1)
        self.assertTrue(transition["no_solved_claim"])
        self.assertIn("bridge_tasks", transition["changed_blackboard_fields"])
        self.assertTrue(audit["payload"]["round_summaries"][0]["changed_blackboard_fields"])
        self.assertIn(
            "literature_scout_report",
            audit["payload"]["typed_artifact_validation_summary"]["accepted_artifact_keys"],
        )
        self.assertIn("agentic_capability_audit", bundle["artifacts"])
        self.assertEqual(bundle["artifacts"]["agentic_capability_audit"]["artifact_type"], "AgenticCapabilityAudit")
        self.assertTrue(bundle["artifacts"]["agentic_capability_audit"]["payload"]["accepted"])
        capability_checks = {
            row["requirement_id"]: row
            for row in bundle["artifacts"]["agentic_capability_audit"]["payload"]["requirement_checks"]
        }
        self.assertTrue(
            capability_checks["artifact_refs_and_typed_validation_integrity"]["accepted"],
            capability_checks["artifact_refs_and_typed_validation_integrity"]["reasons"],
        )
        self.assertTrue(
            capability_checks["blackboard_transition_history_audited"]["accepted"],
            capability_checks["blackboard_transition_history_audited"]["reasons"],
        )
        self.assertIn("agent_blackboard_snapshot", bundle["artifacts"])
        self.assertEqual(bundle["artifacts"]["agent_blackboard_snapshot"]["artifact_type"], "AgentBlackboardSnapshot")
        self.assertEqual(bundle["artifacts"]["agent_blackboard_snapshot"]["payload"]["schema_version"], "agent_blackboard.v1")
        self.assertIn("agent_blackboard_snapshot", result["final_verdict"]["artifact_refs"])
        self.assertIn("agentic_capability_audit", result["final_verdict"]["artifact_refs"])
        self.assertIn("agentic_run_audit", bundle["artifacts"])
        self.assertEqual(bundle["artifacts"]["agentic_run_audit"]["artifact_type"], "AgenticRunAudit")
        blackboard_validations = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agent_blackboard_snapshot"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(blackboard_validations)
        self.assertTrue(blackboard_validations[0]["accepted"], blackboard_validations[0]["reasons"])
        capability_validations = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agentic_capability_audit"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(capability_validations)
        self.assertTrue(capability_validations[0]["accepted"], capability_validations[0]["reasons"])
        audit_validations = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agentic_run_audit"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(audit_validations)
        self.assertTrue(audit_validations[0]["accepted"], audit_validations[0]["reasons"])
        final_validations = [
            row for row in bundle["validations"] if row.get("schema_version") == "agentic_final_verdict_validation.v1"
        ]
        self.assertTrue(final_validations)
        self.assertTrue(final_validations[-1]["accepted"], final_validations[-1]["reasons"])
        self.assertEqual(
            bundle["artifacts"]["agentic_final_verdict_validation"]["artifact_type"],
            "AgenticFinalVerdictValidation",
        )
        final_validation_artifact_checks = [
            row
            for row in bundle["validations"]
            if row.get("artifact_key") == "agentic_final_verdict_validation"
            and row.get("schema_version") == "agentic_typed_artifact_validation_record.v1"
        ]
        self.assertTrue(final_validation_artifact_checks)
        self.assertTrue(final_validation_artifact_checks[0]["accepted"], final_validation_artifact_checks[0]["reasons"])
        self.assertIn("agentic_run_audit", result["agent_blackboard"]["artifact_refs"])
        self.assertIn("agentic_final_verdict_validation", result["agent_blackboard"]["artifact_refs"])

    def test_prior_analogical_source_pair_seeds_pair_transfer_proposals(self):
        prior_artifacts = {
            "analogical_hypotheses": [
                {
                    "schema_version": "analogical_hypothesis.v1",
                    "hypothesis_id": "analogy:enone_pair",
                    "reaction_family": "enone redox transfer",
                    "source_ref": "doi:10.example/analog-enone",
                    "source_product_smiles": "O=C1C=CCCC1",
                    "source_reactant_smiles": ["O=C1CCCCC1"],
                    "analogy_strength": "medium",
                    "evidence_refs": ["scheme:1"],
                    "risk_flags": ["analog_scope_unknown"],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_agentic_blackboard_controller(
                target_name="enone analog",
                target_smiles="O=C1C=CCCC1",
                family_hint="analogical reaction-center transfer",
                output_dir=tmp,
                max_rounds=1,
                use_codex_action_planner=False,
                budget=HarnessBudget(
                    max_chem_enzy_runs=0,
                    max_guided_chemenzy_runs=0,
                    max_route_expansion_subgoal_runs=0,
                    max_scout_calls=0,
                    max_visual_calls=0,
                ),
                prior_artifacts=prior_artifacts,
            )
            board = json.loads((Path(tmp) / "agent_blackboard.json").read_text(encoding="utf-8"))

        transferred = [
            row
            for row in board["retrosynthetic_proposals"]
            if row["source_type"] == "analogical_reaction_pair_transfer"
        ]
        self.assertTrue(
            any(
                row["proposal_label"] == "enone_to_saturated_ketone_precursor"
                and row["precursor_smiles"] == "O=C1CCCCC1"
                for row in transferred
            )
        )
        self.assertTrue(board["retrosynthetic_proposal_compile_report"]["not_parent_route_proof"])
        task_precursors = {row["precursor_smiles"] for row in board["recursive_hypothesis_tasks"]}
        self.assertIn("O=C1CCCCC1", task_precursors)

    def test_failure_critic_bias_enters_blackboard_and_duplicate_critic_is_stale(self):
        target = TargetInput(target_name="critic_target", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        report = {
            "schema_version": "failure_critic_report.v1",
            "accepted": True,
            "case_id": "critic_target",
            "source_reasons": ["large_atom_jump"],
            "route_failures": [
                {
                    "schema_version": "agent_route_failure.v1",
                    "reason": "large_atom_jump",
                    "summary": "jump",
                }
            ],
            "bridge_tasks": [
                {
                    "schema_version": "agent_bridge_task.v1",
                    "task_id": "target_proximal_bridge_required:critic_target",
                    "task_type": "target_proximal_bridge_required",
                    "status": "open",
                }
            ],
            "terminal_blacklist": [],
            "blocked_directions": [
                {
                    "schema_version": "agent_blocked_direction.v1",
                    "direction": "current_route_family_without_core_bridge",
                    "reason": "large_atom_jump",
                }
            ],
            "next_action_bias": ["generate_disconnection_hypotheses", "search_literature"],
            "constraints": {"target_core_retention_required": True, "max_unexplained_heavy_atom_jump": 12},
            "no_solved_claim": True,
        }
        action = {
            "schema_version": "agent_action.v1",
            "action_id": "critic:1",
            "action_type": "build_failure_critic_report",
            "rationale": "normalize failure",
            "expected_artifact": "failure_critic_report.v1",
            "success_condition": "bridge task",
            "payload": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action=action,
                action_result={"accepted": True, "result": report},
                round_index=1,
                run_dir=tmp,
            )
            board = update_blackboard_from_action(
                board,
                action={**action, "action_id": "critic:2"},
                action_result={"accepted": True, "result": report},
                round_index=2,
                run_dir=tmp,
            )

        belief = board["current_belief"]
        self.assertIn("generate_disconnection_hypotheses", belief["next_action_bias"])
        self.assertIn("search_literature", belief["next_action_bias"])
        self.assertEqual(belief["constraints"]["max_unexplained_heavy_atom_jump"], 12)
        self.assertTrue(board["action_history"][0]["useful_artifact"])
        self.assertFalse(board["action_history"][1]["useful_artifact"])
        self.assertTrue(board["action_history"][1]["stale"])

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn("search_literature", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_literature_scout_default_timeout_respects_open_research_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = TargetInput(target_name="steroid", target_smiles="CCO")
            preflight = run_preflight(target)
            board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(open_research_timeout_s=900.0),
                model="gpt-5.5",
            )

            task = _codex_literature_scout_task(
                blackboard=board,
                state=state,
                payload={},
                max_sources=3,
            )

        self.assertEqual(_codex_scout_timeout_s(state, {}), 900.0)
        self.assertEqual(task.budget.timeout_s, 900.0)
        self.assertEqual(task.budget.reasoning_effort, "high")
        self.assertEqual(task.model, "gpt-5.5")
        self.assertIn("web_search", task.allowed_tools)
        self.assertIn("browser", task.allowed_tools)
        self.assertGreater(task.budget.max_tool_calls, 0)
        self.assertTrue(_task_allows_cli_search(task))

    def test_codex_repaired_guided_timeout_uses_harness_budget(self):
        self.assertEqual(
            _codex_repaired_or_bounded_timeout_s(
                {
                    "timeout_s": 180,
                    "codex_payload_repair": {
                        "schema_version": "codex_action_payload_repair.v1",
                        "action_type": "run_guided_chemenzy",
                    },
                },
                1200,
            ),
            1200.0,
        )
        self.assertEqual(_codex_repaired_or_bounded_timeout_s({"timeout_s": 180}, 1200), 180.0)

    def test_local_pdf_cache_match_prefers_exact_doi_over_same_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wrong_pdf = tmp_path / "Angew Total Synthesis of Ouabagenin and Ouabain.pdf"
            right_pdf = tmp_path / "Asian Journal Total Synthesis of Ouabagenin and Ouabain.pdf"
            wrong_pdf.write_bytes(b"%PDF-1.4\nDOI: 10.1002/anie.200704959\n")
            right_pdf.write_bytes(b"%PDF-1.4\nDOI: 10.1002/asia.200800429\n")
            target = TargetInput(target_name="same_title_doi_binding", target_smiles="CCO")
            target_input = target.to_dict()
            target_input["local_literature_cache"] = [
                {
                    "source_ref": "doi:10.1002/anie.200704959",
                    "doi": "10.1002/anie.200704959",
                    "title": "Total Synthesis of Ouabagenin and Ouabain",
                    "local_pdf": str(wrong_pdf),
                    "source_role": "auto_local_pdf_cache",
                },
                {
                    "source_ref": "doi:10.1002/asia.200800429",
                    "doi": "10.1002/asia.200800429",
                    "title": "Total Synthesis of Ouabagenin and Ouabain",
                    "local_pdf": str(right_pdf),
                    "source_role": "auto_local_pdf_cache",
                },
            ]
            preflight = run_preflight(target)
            state = ToolExecutionState(
                run_dir=tmp_path,
                target_input=target_input,
                preflight=preflight,
            )

            report = _local_pdf_cache_match_report(
                codex_report={
                    "source_candidates": [
                        {
                            "source_ref": "doi:10.1002/asia.200800429",
                            "doi": "10.1002/asia.200800429",
                            "title": "Total Synthesis of Ouabagenin and Ouabain",
                            "url": "https://doi.org/10.1002/asia.200800429",
                        }
                    ]
                },
                state=state,
                payload={},
                max_sources=3,
            )

        self.assertTrue(report["accepted"], report["reasons"])
        self.assertEqual(len(report["source_candidates"]), 1)
        candidate = report["source_candidates"][0]
        self.assertEqual(candidate["local_pdf"], str(right_pdf.resolve()))
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "doi")
        self.assertEqual(candidate["local_pdf_match"]["cache_doi"], "10.1002/asia.200800429")

    def test_dynamic_scout_merge_preserves_article_and_si_sharing_one_doi(self):
        doi = "10.1000/article-with-si"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article = root / "article.pdf"
            si = root / "article_si.pdf"
            article.write_bytes(b"%PDF-1.4\narticle\n%%EOF\n")
            si.write_bytes(b"%PDF-1.4\nsupplementary information\n%%EOF\n")
            codex_report = {
                "schema_version": "literature_scout_report.v1",
                "accepted": True,
                "source_candidates": [
                    {
                        "source_ref": f"doi:{doi}",
                        "doi": doi,
                        "title": "A synthesis article with supporting information",
                        "url": f"https://doi.org/{doi}",
                        "access_status": "metadata_only",
                    }
                ],
                "source_refs": [f"doi:{doi}"],
            }
            local_report = {
                "schema_version": "literature_scout_report.v1",
                "accepted": True,
                "source_candidates": [
                    {
                        "source_ref": f"doi:{doi}",
                        "doi": doi,
                        "local_pdf": str(article),
                        "document_id": "article-document",
                        "content_scope": "article",
                        "access_status": "local_pdf_available",
                    },
                    {
                        "source_ref": f"doi:{doi}",
                        "doi": doi,
                        "local_pdf": str(si),
                        "document_id": "si-document",
                        "content_scope": "supplementary_information",
                        "access_status": "local_pdf_available",
                    },
                ],
                "source_refs": [f"doi:{doi}"],
            }

            merged = _merge_local_pdf_scout_report(
                codex_report,
                local_report,
                max_sources=2,
            )

        candidates = merged["source_candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {row["local_pdf"] for row in candidates},
            {str(article), str(si)},
        )
        self.assertEqual(
            {row["content_scope"] for row in candidates},
            {"article", "supplementary_information"},
        )
        self.assertTrue(all(row["title"] == "A synthesis article with supporting information" for row in candidates))

    def test_search_literature_uses_codex_online_source_before_fallbacks(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "online_case",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "online:search",
                        "action_type": "search_literature",
                        "rationale": "online source scout",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "real source candidate",
                        "payload": _test_search_payload("pdf fallback local literature source"),
                    }
                ],
            }

        codex_scout = {
            "schema_version": "literature_scout_report.v1",
            "accepted": True,
            "case_id": "online_case",
            "source_candidates": [
                {
                    "schema_version": "literature_source_candidate.v1",
                    "candidate_id": "src1",
                    "source_ref": "doi:10.1000/example",
                    "title": "Example target-proximal steroid synthesis",
                    "doi": "10.1000/example",
                    "url": "https://doi.org/10.1000/example",
                    "local_pdf": "",
                    "source_type": "journal_article",
                    "relevance_rationale": "target-proximal source",
                    "expected_scheme_or_compound_labels": ["1", "2"],
                    "extraction_task_recommendations": ["resolve_source_material_or_provide_pdf"],
                    "access_status": "ACS DOI page found by web search; direct article fetch may require licensed access.",
                    "no_solved_claim": True,
                }
            ],
            "source_refs": ["doi:10.1000/example"],
            "search_queries": ["example query"],
            "reasons": [],
            "limitations": [],
            "no_solved_claim": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="online_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"codex_literature_scout": codex_scout},
            )
            evidence = result["agent_blackboard"]["literature_evidence"]
            self.assertEqual(evidence["source_discovery_mode"], "codex_online")
            self.assertEqual(evidence["confidence"], "candidate")
            self.assertEqual(evidence["source_candidates"][0]["doi"], "10.1000/example")
            self.assertEqual(len(evidence["local_pdf_proxy_requests"]), 1)
            self.assertEqual(evidence["local_pdf_proxy_requests"][0]["doi"], "10.1000/example")
            self.assertEqual(evidence["local_pdf_proxy_requests"][0]["content_scope"], "article")
            lifecycle = {
                row["source_key"]: row
                for row in evidence["source_lifecycle"]
            }
            source = lifecycle["document:doi:10.1000/example:article"]
            self.assertEqual(source["stage"], "local_pdf_proxy_requested")
            self.assertEqual(
                source["independent_source_group"], "doi:10.1000/example"
            )
            self.assertEqual(
                source["next_recommended_stage"],
                "await_local_pdf_proxy_download",
            )
            self.assertEqual(source["counts"]["local_pdf_proxy_requests"], 1)
            self.assertEqual(
                evidence["source_identity_summary"]["document_count"], 1
            )
            self.assertEqual(
                evidence["source_identity_summary"][
                    "independent_source_group_count"
                ],
                1,
            )
            queue_path = local_pdf_proxy_request_queue_path(Path(tmp))
            queued = load_pdf_requests(queue_path)
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["doi"], "10.1000/example")
            literature_sources = json.loads((Path(tmp) / "evidence" / "literature_sources.json").read_text(encoding="utf-8"))
            self.assertEqual(
                literature_sources["search_log"][0]["agent_access_status"],
                "agent_accessible_metadata_only",
            )
            proxy_audit = audit_local_pdf_proxy_fallback(run_dir=tmp)
            self.assertTrue(proxy_audit["accepted"], proxy_audit["reasons"])
            summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
            self.assertEqual(summary["local_pdf_proxy_request_count"], 1)
            self.assertEqual(summary["awaiting_local_pdf_proxy_count"], 1)
            self.assertEqual(summary["source_lifecycle_stage_counts"]["local_pdf_proxy_requested"], 1)
            followups = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["followup_tasks"]
            self.assertEqual(followups[0]["task_type"], "await_local_pdf_proxy_download")
            self.assertEqual(followups[0]["doi"], "10.1000/example")
            self.assertEqual(followups[0]["recommended_next_action"], "extract_pdf_literature_structures")
            self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])
            scout_artifact = result["artifact_bundle"]["artifacts"]["literature_scout_report"]
            self.assertEqual(scout_artifact["artifact_type"], "LiteratureScoutReport")
            self.assertEqual(scout_artifact["payload"]["source_discovery_mode"], "codex_online")
            self.assertEqual(scout_artifact["payload"]["local_pdf_proxy_request_summary"]["request_count"], 1)
            scout_validations = [
                row
                for row in result["artifact_bundle"]["validations"]
                if row.get("artifact_key") == "literature_scout_report"
            ]
            self.assertTrue(scout_validations)
            self.assertTrue(scout_validations[-1]["accepted"], scout_validations[-1]["reasons"])

    def test_local_pdf_proxy_download_manifest_is_reused_as_local_source(self):
        doi = "10.1000/proxy.download"

        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "proxy_download_reuse",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "proxy:search",
                        "action_type": "search_literature",
                        "rationale": "online scout fails, downloaded proxy PDF should be reused",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "local proxy PDF candidate",
                        "payload": _test_search_payload("proxy download reuse"),
                    }
                ],
            }

        failed_scout = {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": "proxy_download_reuse",
            "source_candidates": [],
            "source_refs": [],
            "search_queries": ["proxy download reuse"],
            "reasons": ["online_unavailable"],
            "limitations": [],
            "no_solved_claim": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "evidence" / "local_pdf_proxy" / "pdfs"
            pdf_dir.mkdir(parents=True)
            pdf_path = pdf_dir / "proxy_download.pdf"
            pdf_path.write_bytes(f"%PDF-1.4\nDOI {doi}\n%%EOF\n".encode("latin-1"))
            manifest = local_pdf_proxy_download_manifest_path(root)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "local_pdf_proxy_result.v1",
                        "request_id": "proxy-download",
                        "case_id": "proxy_download_reuse",
                        "source_ref": f"doi:{doi}",
                        "doi": doi,
                        "url": f"https://doi.org/{doi}",
                        "title": "Proxy downloaded source",
                        "status": "downloaded",
                        "accepted": True,
                        "pdf_path": str(pdf_path),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_agentic_blackboard_controller(
                target_name="proxy_download_reuse",
                target_smiles="CCO",
                output_dir=root,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"codex_literature_scout": failed_scout},
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_fallback_after_codex_failure")
        self.assertEqual(evidence["source_candidates"][0]["doi"], doi)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf_path.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["access_status"], "local_pdf_available")
        self.assertEqual(evidence["source_lifecycle"][0]["stage"], "local_pdf_available")

    def test_late_local_pdf_proxy_download_refreshes_active_blackboard(self):
        doi = "10.1000/late.browser.download"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "evidence" / "local_pdf_proxy" / "pdfs"
            pdf_dir.mkdir(parents=True)
            pdf_path = pdf_dir / "late_download.pdf"
            pdf_path.write_bytes(f"%PDF-1.4\nDOI {doi}\n%%EOF\n".encode("latin-1"))
            local_pdf_proxy_download_manifest_path(root).write_text(
                json.dumps(
                    {
                        "schema_version": "local_pdf_proxy_result.v1",
                        "request_id": "late-browser-download",
                        "case_id": "late_pdf_refresh",
                        "source_ref": f"doi:{doi}",
                        "doi": doi,
                        "url": f"https://doi.org/{doi}",
                        "title": "Late browser downloaded synthesis source",
                        "status": "downloaded",
                        "accepted": True,
                        "pdf_path": str(pdf_path),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            target = TargetInput(
                target_name="late_pdf_refresh",
                target_smiles="CCO",
                family_hint="",
                case_id="",
            )
            preflight = run_preflight(target)
            board = initialize_agent_blackboard(
                target_input=target.to_dict(),
                preflight=preflight,
                max_rounds=3,
                budget_limits={"max_visual_calls": 3},
            )
            refreshed = _refresh_blackboard_from_local_pdf_proxy_downloads(board, run_dir=root)

        evidence = refreshed["literature_evidence"]
        self.assertEqual(evidence["source_candidates"][0]["doi"], doi)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf_path.resolve()))
        self.assertEqual(evidence["source_lifecycle"][0]["stage"], "local_pdf_available")
        self.assertIn("extract_pdf_literature_structures", refreshed["current_belief"]["next_action_bias"])
        refreshed["action_history"] = [
            {"action_type": "search_literature", "useful_artifact": False},
            {"action_type": "run_guided_chemenzy", "useful_artifact": False},
        ]
        batch = plan_action_batch(refreshed, round_index=2, max_actions=3)
        self.assertIn(
            "extract_pdf_literature_structures",
            [str(action.get("action_type") or "") for action in batch.get("actions") or []],
        )

    def test_planner_source_hint_can_trigger_auto_local_pdf_cache_match_after_scout_failure(self):
        doi = "10.4242/plannerhint2026"
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": "planner_hint_cache",
            "round_index": 1,
            "mode": "codex_test",
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "hint:search",
                    "action_type": "search_literature",
                    "rationale": "confirm planner-discovered source hint",
                    "expected_artifact": "literature_scout_report.v1",
                    "success_condition": "source candidate or explicit unresolved source task",
                    "payload": _test_search_payload("planner hint cache confirmation"),
                }
            ],
            "planner_source_hints": [
                {
                    "schema_version": "planner_source_hint.v1",
                    "hint_id": "hint_doi",
                    "source_ref": f"doi:{doi}",
                    "title": "Planner hinted cache matched source",
                    "doi": doi,
                    "pii": "",
                    "url": f"https://doi.org/{doi}",
                    "local_pdf": "",
                    "local_ref": "",
                    "source_type": "planner_discovered_literature_metadata",
                    "relevance_rationale": "Codex planner found this DOI while selecting actions.",
                    "expected_scheme_or_compound_labels": ["1", "2"],
                    "extraction_task_recommendations": ["extract_pdf_literature_structures"],
                    "evidence_class": "planner_source_hint",
                    "allowed_use": "source_acquisition_hint_only",
                    "no_solved_claim": True,
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }
        failed_scout = {
            "schema_version": "literature_scout_report.v1",
            "accepted": False,
            "case_id": "planner_hint_cache",
            "source_candidates": [],
            "source_refs": [],
            "search_queries": ["planner hint cache confirmation"],
            "reasons": ["mock_online_unavailable"],
            "limitations": [],
            "no_solved_claim": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "pdf_cache"
            cache_dir.mkdir()
            pdf_path = cache_dir / "planner_hint_cache_source.pdf"
            pdf_path.write_bytes(f"%PDF-1.4\nRelated DOI {doi}\n%%EOF\n".encode("latin-1"))
            result = run_agentic_blackboard_controller(
                target_name="planner_hint_cache",
                target_smiles="CCO",
                output_dir=root / "run",
                max_rounds=1,
                local_pdf_search_dirs=[cache_dir],
                mock_tool_results={
                    "codex_action_planner": codex_batch,
                    "codex_literature_scout": failed_scout,
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["planner_source_hints"][0]["doi"], doi)
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_cache_match")
        self.assertEqual(evidence["source_candidates"][0]["doi"], doi)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf_path.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["local_pdf_match"]["agent_discovered_doi"], doi)
        self.assertEqual(
            evidence["source_candidates"][0]["local_pdf_index"]["match_policy"],
            "agent_discovered_metadata_required",
        )
        self.assertEqual(evidence["confidence"], "candidate")
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertEqual(summary["planner_source_hint_count"], 1)
        self.assertTrue(summary["planner_source_hints_are_not_evidence"])
        self.assertEqual(summary["source_lifecycle_stage_counts"]["local_pdf_available"], 1)
        self.assertEqual(summary["auto_local_pdf_cache_match_count"], 1)
        self.assertFalse(result["final_verdict"]["solved"])

    def test_search_literature_falls_back_to_local_pdf_after_codex_failure(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_fallback",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:search",
                        "action_type": "search_literature",
                        "rationale": "online source scout with local fallback",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "local source candidate",
                        "payload": _test_search_payload("pdf merge online metadata local pdf"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_fallback",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_pdf_path=str(pdf),
                literature_pdf_source_ref="doi:10.1000/local",
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "pdf_fallback",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_failed"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_fallback")
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["doi"], "10.1000/local")
        self.assertEqual(evidence["source_candidates"][0]["source_role"], "user_provided_local_pdf_seed")
        self.assertTrue(evidence["source_candidates"][0]["user_provided_source_seed"])
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertTrue(summary["codex_online_attempted"])
        self.assertEqual(summary["user_provided_local_pdf_seed_count"], 1)
        self.assertEqual(summary["direct_local_pdf_after_codex_failure_count"], 1)
        self.assertTrue(result["agent_blackboard"]["action_history"][0]["useful_artifact"])

    def test_search_literature_merges_local_pdf_when_codex_finds_metadata(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_merge",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:merge",
                        "action_type": "search_literature",
                        "rationale": "online source scout with known local PDF",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "online metadata and local PDF source are retained",
                        "payload": _test_search_payload("cache match DOI local PDF"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_merge",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_pdf_path=str(pdf),
                literature_pdf_source_ref="doi:10.1000/local",
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "pdf_merge",
                        "source_candidates": [
                            {
                                "source_ref": "doi:10.1000/local",
                                "doi": "10.1000/local",
                                "title": "Online metadata source",
                                "url": "https://doi.org/10.1000/local",
                            }
                        ],
                        "source_refs": ["doi:10.1000/local"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf")
        self.assertEqual(len(evidence["source_candidates"]), 1)
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["access_status"], "local_pdf_available")
        self.assertIn("extract_visual_literature_chain", evidence["source_candidates"][0]["extraction_task_recommendations"])

    def test_search_literature_matches_agent_discovered_doi_to_local_cache(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_cache_match",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:cache-match",
                        "action_type": "search_literature",
                        "rationale": "agent searches first, then local cache may satisfy access",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "online DOI is retained and local PDF cache is attached",
                        "payload": _test_search_payload("auto cache match DOI local PDF"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_cache_match",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_sources=[
                    {
                        "local_pdf": str(pdf),
                        "source_ref": "doi:10.1000/cache",
                        "title": "Cached local article",
                    }
                ],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "pdf_cache_match",
                        "source_candidates": [
                            {
                                "source_ref": "doi:10.1000/cache",
                                "doi": "10.1000/cache",
                                "title": "Agent discovered article",
                                "url": "https://doi.org/10.1000/cache",
                            }
                        ],
                        "source_refs": ["doi:10.1000/cache"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        candidate = evidence["source_candidates"][0]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf_cache")
        self.assertEqual(candidate["title"], "Agent discovered article")
        self.assertEqual(candidate["local_pdf"], str(pdf.resolve()))
        self.assertEqual(candidate["source_type"], "literature_metadata+local_pdf")
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "doi")
        self.assertIn("local_pdf_cache", [row["mode"] for row in evidence["scout_attempts"]])

    def test_search_literature_auto_discovers_local_pdf_cache_for_agent_doi(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "auto_pdf_cache_match",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:auto-cache-match",
                        "action_type": "search_literature",
                        "rationale": "agent discovers DOI, auto local PDF cache should attach matching file",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "auto-indexed local PDF is attached only after DOI match",
                        "payload": _test_search_payload("ScienceDirect PII local PDF match"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers"
            paper_dir.mkdir()
            pdf = paper_dir / "auto-source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n10.1234/auto.cache\n%%EOF\n")
            result = run_agentic_blackboard_controller(
                target_name="auto_pdf_cache_match",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                local_pdf_search_dirs=[paper_dir],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "auto_pdf_cache_match",
                        "source_candidates": [
                            {
                                "source_ref": "doi:10.1234/auto.cache",
                                "doi": "10.1234/auto.cache",
                                "title": "Agent discovered auto cache paper",
                                "url": "https://doi.org/10.1234/auto.cache",
                            }
                        ],
                        "source_refs": ["doi:10.1234/auto.cache"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        candidate = evidence["source_candidates"][0]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf_cache")
        self.assertEqual(candidate["local_pdf"], str(pdf.resolve()))
        self.assertEqual(candidate["source_type"], "literature_metadata+local_pdf")
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "doi")
        self.assertEqual(candidate["local_pdf_index"]["match_policy"], "agent_discovered_metadata_required")
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertEqual(summary["local_pdf_cache_match_count"], 1)
        self.assertEqual(summary["auto_local_pdf_cache_match_count"], 1)
        self.assertEqual(summary["agent_discovered_local_pdf_match_count"], 1)
        self.assertEqual(summary["local_pdf_match_bases"], ["doi"])
        self.assertFalse(summary["auto_local_pdf_blind_fallback_used"])
        capability_checks = {
            row["requirement_id"]: row
            for row in result["artifact_bundle"]["artifacts"]["agentic_capability_audit"]["payload"]["requirement_checks"]
        }
        self.assertTrue(
            capability_checks["codex_first_source_acquisition_audited"]["accepted"],
            capability_checks["codex_first_source_acquisition_audited"]["reasons"],
        )

    def test_search_literature_auto_pdf_cache_matches_sciencedirect_pii(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "auto_pdf_pii_match",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:auto-pii-match",
                        "action_type": "search_literature",
                        "rationale": "agent finds a ScienceDirect page, local filename PII should match",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "auto-indexed ScienceDirect PDF is attached by PII",
                        "payload": _test_search_payload("auto cache no blind fallback"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers"
            paper_dir.mkdir()
            pdf = paper_dir / "1-s2.0-S0040402025001668-main.pdf"
            pdf.write_bytes(b"%PDF-1.4\nmock science direct pdf\n%%EOF\n")
            result = run_agentic_blackboard_controller(
                target_name="auto_pdf_pii_match",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                local_pdf_search_dirs=[paper_dir],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "case_id": "auto_pdf_pii_match",
                        "source_candidates": [
                            {
                                "source_ref": "sciencedirect:S0040402025001668",
                                "title": "Agent discovered ScienceDirect article",
                                "url": "https://www.sciencedirect.com/science/article/pii/S0040402025001668",
                            }
                        ],
                        "source_refs": ["sciencedirect:S0040402025001668"],
                        "reasons": [],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        candidate = evidence["source_candidates"][0]
        self.assertEqual(evidence["source_discovery_mode"], "codex_online+local_pdf_cache")
        self.assertEqual(candidate["local_pdf"], str(pdf.resolve()))
        self.assertEqual(candidate["pii"], "S0040402025001668")
        self.assertEqual(candidate["local_pdf_match"]["match_basis"], "pii")
        summary = result["artifact_bundle"]["artifacts"]["agentic_run_audit"]["payload"]["source_acquisition_summary"]
        self.assertEqual(summary["auto_local_pdf_cache_match_count"], 1)
        self.assertEqual(summary["local_pdf_match_bases"], ["pii"])
        self.assertEqual(summary["agent_discovered_local_pdf_match_count"], 1)
        self.assertFalse(summary["auto_local_pdf_blind_fallback_used"])

    def test_auto_local_pdf_cache_is_not_blind_fallback_after_online_miss(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "auto_pdf_no_blind_fallback",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:auto-no-blind-fallback",
                        "action_type": "search_literature",
                        "rationale": "online scout failed, auto local cache should not be used blindly",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "placeholder is emitted instead of arbitrary auto cache PDF",
                        "payload": _test_search_payload("local cache fallback after online failure"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "papers"
            paper_dir.mkdir()
            (paper_dir / "unmatched-source.pdf").write_bytes(b"%PDF-1.4\n10.1234/unmatched.cache\n%%EOF\n")
            result = run_agentic_blackboard_controller(
                target_name="auto_pdf_no_blind_fallback",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                local_pdf_search_dirs=[paper_dir],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "auto_pdf_no_blind_fallback",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_no_hit"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "placeholder")
        self.assertTrue(evidence["source_candidates"][0]["placeholder_only"])
        self.assertFalse(str(evidence["source_candidates"][0].get("local_pdf") or "").strip())

    def test_capability_audit_rejects_auto_local_pdf_without_agent_match_provenance(self):
        target = TargetInput(target_name="bad_auto_cache", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [{"mode": "local_pdf_cache", "attempted": True}]
        board["literature_evidence"]["source_discovery_mode"] = "codex_online+local_pdf_cache"
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1234/bad",
                "doi": "10.1234/bad",
                "local_pdf": "/tmp/bad.pdf",
                "source_discovery_mode": "codex_online+local_pdf_cache",
                "local_pdf_index": {
                    "schema_version": "auto_local_pdf_index.v1",
                    "match_policy": "agent_discovered_metadata_required",
                },
                "no_solved_claim": True,
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "status": "accepted",
            }
        ]

        check = _capability_check_source_acquisition(board)

        self.assertFalse(check["accepted"])
        self.assertIn("local_pdf_cache_match_missing_provenance:0", check["reasons"])
        self.assertIn("auto_local_pdf_cache_without_agent_discovered_match:0", check["reasons"])

    def test_capability_audit_rejects_direct_local_pdf_without_codex_attempt_or_user_seed(self):
        target = TargetInput(target_name="bad_direct_pdf", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [{"mode": "local_pdf", "attempted": True, "accepted": True}]
        board["literature_evidence"]["source_discovery_mode"] = "local_pdf_fallback"
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1234/direct",
                "doi": "10.1234/direct",
                "local_pdf": "/tmp/direct.pdf",
                "source_discovery_mode": "local_pdf_fallback",
                "no_solved_claim": True,
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "status": "accepted",
            }
        ]

        check = _capability_check_source_acquisition(board)

        self.assertFalse(check["accepted"])
        self.assertIn("local_pdf_fallback_without_codex_online_attempt:0", check["reasons"])
        self.assertIn("direct_local_pdf_fallback_missing_user_seed_marker:0", check["reasons"])

    def test_capability_audit_rejects_metadata_only_source_without_pdf_proxy_request(self):
        target = TargetInput(target_name="metadata_only_gap", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["fallback_order"] = ["codex_online", "local_pdf", "placeholder"]
        board["literature_evidence"]["scout_attempts"] = [
            {"mode": "codex_online", "attempted": True, "accepted": True}
        ]
        board["literature_evidence"]["source_discovery_mode"] = "codex_online"
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1234/metadata.only",
                "doi": "10.1234/metadata.only",
                "url": "https://doi.org/10.1234/metadata.only",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "search_literature",
                "status": "accepted",
            }
        ]

        check = _capability_check_source_acquisition(board)

        self.assertFalse(check["accepted"])
        self.assertIn("metadata_only_source_without_local_pdf_proxy_request:0", check["reasons"])

    def test_local_pdf_cache_falls_back_after_online_scout_has_no_source(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_cache_no_online_hit",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "pdf:cache-no-hit",
                        "action_type": "search_literature",
                        "rationale": "online source failed, local cache should be tried before placeholder",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "local PDF fallback source before placeholder",
                        "payload": _test_search_payload("placeholder after online and local fail"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock pdf\n")
            result = run_agentic_blackboard_controller(
                target_name="pdf_cache_no_online_hit",
                target_smiles="CCO",
                output_dir=Path(tmp) / "run",
                literature_sources=[{"local_pdf": str(pdf), "source_ref": "doi:10.1000/cache"}],
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "pdf_cache_no_online_hit",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_no_hit"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["source_discovery_mode"], "local_pdf_fallback_after_codex_failure")
        self.assertFalse(evidence["source_candidates"][0].get("placeholder_only", False))
        self.assertEqual(evidence["source_candidates"][0]["local_pdf"], str(pdf.resolve()))
        self.assertEqual(evidence["source_candidates"][0]["source_discovery_mode"], "local_pdf_fallback_after_codex_failure")
        cache_attempts = [row for row in evidence["scout_attempts"] if row.get("mode") == "local_pdf_cache"]
        self.assertTrue(cache_attempts[0]["accepted"])

    def test_search_literature_writes_placeholder_only_after_online_and_local_fail(self):
        def planner(**kwargs):
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "placeholder_case",
                "round_index": kwargs["round_index"],
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "placeholder:search",
                        "action_type": "search_literature",
                        "rationale": "record missing source",
                        "expected_artifact": "literature_scout_report.v1",
                        "success_condition": "placeholder if all source access fails",
                        "payload": _test_search_payload("placeholder after online and local fail"),
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="placeholder_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={
                    "codex_literature_scout": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": False,
                        "case_id": "placeholder_case",
                        "source_candidates": [],
                        "source_refs": [],
                        "reasons": ["mock_online_failed"],
                        "limitations": [],
                        "no_solved_claim": True,
                    }
                },
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["confidence"], "placeholder")
        self.assertTrue(evidence["source_candidates"][0]["placeholder_only"])
        self.assertFalse(result["agent_blackboard"]["action_history"][0]["useful_artifact"])
        scout_artifact = result["artifact_bundle"]["artifacts"]["literature_scout_report"]
        self.assertEqual(scout_artifact["artifact_type"], "LiteratureScoutReport")
        self.assertFalse(scout_artifact["payload"]["accepted"])
        self.assertTrue(scout_artifact["payload"]["placeholder_only"])
        scout_validations = [
            row
            for row in result["artifact_bundle"]["validations"]
            if row.get("artifact_key") == "literature_scout_report"
        ]
        self.assertTrue(scout_validations)
        self.assertTrue(scout_validations[-1]["accepted"], scout_validations[-1]["reasons"])
        self.assertNotIn("typed_artifact_validation_failed:literature_scout_report", result["artifact_bundle"]["safety_flags"])

    @patch.dict(
        os.environ,
        {
            "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(
                _SOURCE_FIXTURES / "trusted_literature_step_registry.json"
            )
        },
    )
    def test_parent_proof_mock_is_required_for_agentic_solved(self):
        verifier = _strict_parent_route_verifier("CCO", reactants=["CC", "O"])
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCO",
            target_name="proof_case",
            case_id="proof_case",
            parent_verifier=verifier,
        )
        self.assertTrue(proof["accepted"], proof["reasons"])
        stitch_payload = {
            "proof_binding": {
                "schema_version": "agentic_parent_stitch_binding.v1",
                "child_route_ref": "mock:child_route",
                "parent_route_ref": "mock:parent_route",
                "exact_literature_segment_ref": "mock:exact_segment",
                "exact_literature_row_ids": ["source_detail_exact_step:mock"],
                "input_refs": ["mock:child_route", "mock:parent_route", "mock:exact_segment"],
                "missing_inputs": [],
            },
            "proof_policy": {
                "schema_version": "agentic_parent_stitch_policy.v1",
                "target_equivalence_required": True,
                "parent_route_verifier_required": True,
                "stock_audit_required": True,
                "no_unexplained_large_atom_jump_required": True,
                "child_route_connectivity_required": True,
                "exact_literature_connectivity_required": True,
                "analogy_is_not_proof": True,
                "child_route_cannot_promote_parent": True,
                "final_verdict_authority": "deterministic_parent_route_proof",
            },
        }

        def planner(**kwargs):
            del kwargs
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "proof_case",
                "round_index": 1,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "proof:stitch",
                        "action_type": "stitch_parent_route",
                        "rationale": "mock accepted parent proof",
                        "expected_artifact": "stitched_parent_route_proof.v1",
                        "success_condition": "parent proof accepted",
                        "payload": stitch_payload,
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="proof_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"stitch_parent_route": {"accepted": True, "result": {"parent_route_proof": proof}}},
            )

        self.assertEqual(result["final_verdict"]["verdict"], "solved")
        self.assertTrue(result["final_verdict"]["solved"])
        final_validations = [
            row
            for row in result["artifact_bundle"]["validations"]
            if row.get("schema_version") == "agentic_final_verdict_validation.v1"
        ]
        self.assertTrue(final_validations)
        self.assertTrue(final_validations[-1]["accepted"], final_validations[-1]["reasons"])
        self.assertEqual(
            result["artifact_bundle"]["artifacts"]["agentic_final_verdict_validation"]["artifact_type"],
            "AgenticFinalVerdictValidation",
        )

    def test_final_verdict_validation_rejects_solved_without_parent_proof(self):
        validation = _validate_agentic_final_verdict(
            {
                "schema_version": "codex_entry_final_verdict.v1",
                "case_id": "bad_final",
                "verdict": "solved",
                "route_status": "solved",
                "solved": True,
                "stock_audit_passed": True,
            },
            blackboard={
                "case_id": "bad_final",
                "parent_route_proof": {"accepted": False, "solved": False},
                "current_belief": {"child_route_solved": True},
            },
            validations=[],
        )

        self.assertFalse(validation["accepted"])
        self.assertIn("final_solved_without_parent_proof", validation["reasons"])
        self.assertIn("child_solved_promoted_without_parent_proof", validation["reasons"])

    def test_pdf_structure_action_updates_blackboard_without_solved_claim(self):
        def planner(**kwargs):
            round_index = kwargs["round_index"]
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "pdf_case",
                "round_index": round_index,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": f"pdf:{round_index}",
                        "action_type": "extract_pdf_literature_structures",
                        "rationale": "local PDF source should be rendered before visual chain extraction",
                        "expected_artifact": "literature_pdf_structure_evidence.v1",
                        "success_condition": "rendered pages are available",
                        "payload": {},
                    }
                ],
            }

        pdf_result = {
            "schema_version": "literature_pdf_structure_evidence.v1",
            "accepted": True,
            "source_pdf_path": "/tmp/source.pdf",
            "rendered_pages": [{"page_number": 1, "image_path": "/tmp/page-1.png"}],
            "indexed_images": [],
            "scheme_crops": [],
            "compound_text_snippets": [],
            "summary": {
                "rendered_page_count": 1,
                "indexed_image_count": 0,
                "scheme_crop_count": 0,
                "compound_text_snippet_count": 0,
            },
            "reasons": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="pdf_case",
                target_smiles="CCO",
                output_dir=tmp,
                max_rounds=1,
                action_planner=planner,
                mock_tool_results={"extract_pdf_literature_structures": pdf_result},
            )

        evidence = result["agent_blackboard"]["literature_evidence"]
        self.assertEqual(evidence["pdf_structure_evidence"][0]["summary"]["rendered_page_count"], 1)
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")

    def test_action_payload_source_context_survives_tool_output_without_source_fields(self):
        target = TargetInput(target_name="multi_pdf_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:first", "local_pdf": "/tmp/first.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:second", "local_pdf": "/tmp/second.pdf"},
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "pdf:first",
                "action_type": "extract_pdf_literature_structures",
                "rationale": "render first PDF",
                "expected_artifact": "literature_pdf_structure_evidence.v1",
                "success_condition": "rendered pages",
                "payload": {"source_ref": "doi:first", "pdf_path": "/tmp/first.pdf"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": True,
                    "rendered_pages": [{"page_number": 1, "image_path": "/tmp/first-1.png"}],
                    "summary": {"rendered_page_count": 1},
                },
            },
            round_index=1,
            run_dir="/tmp",
        )
        pdf_summary = board["literature_evidence"]["pdf_structure_evidence"][0]
        self.assertEqual(pdf_summary["source_ref"], "doi:first")
        self.assertEqual(pdf_summary["source_pdf_path"], "/tmp/first.pdf")
        self.assertEqual(pdf_summary["evidence_id"], "doi:first")
        lifecycle_by_ref = {
            row["source_ref"]: row
            for row in board["literature_evidence"]["source_lifecycle"]
        }
        self.assertEqual(lifecycle_by_ref["doi:first"]["stage"], "pdf_rendered")
        self.assertEqual(lifecycle_by_ref["doi:second"]["stage"], "local_pdf_available")
        pdf_history = board["action_history"][-1]
        self.assertEqual(pdf_history["blackboard_delta"]["pdf_structure_evidence"], 1)
        self.assertEqual(pdf_history["blackboard_delta"]["source_lifecycle"], 2)
        self.assertIn("pdf_structure_evidence", pdf_history["changed_blackboard_fields"])
        self.assertIn("source_lifecycle", pdf_history["changed_blackboard_fields"])
        self.assertEqual(pdf_history["blackboard_counts_before"]["pdf_structure_evidence"], 0)
        self.assertEqual(pdf_history["blackboard_counts_after"]["pdf_structure_evidence"], 1)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:first",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract first visual chain",
                "expected_artifact": "visual_literature_chain.v1",
                "success_condition": "visual chain or explicit gaps",
                "payload": {"source_ref": "doi:first", "pdf_path": "/tmp/first.pdf"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": True,
                    "candidate_chain": {"steps": []},
                    "candidate_quality": {},
                    "reasons": [],
                },
            },
            round_index=2,
            run_dir="/tmp",
        )
        visual_summary = board["literature_evidence"]["visual_chains"][0]
        self.assertEqual(visual_summary["source_ref"], "doi:first")
        self.assertEqual(visual_summary["source_pdf_path"], "/tmp/first.pdf")
        lifecycle_by_ref = {
            row["source_ref"]: row
            for row in board["literature_evidence"]["source_lifecycle"]
        }
        self.assertEqual(lifecycle_by_ref["doi:first"]["stage"], "visual_extracted")
        visual_history = board["action_history"][-1]
        self.assertEqual(visual_history["blackboard_delta"]["visual_chains"], 1)
        self.assertIn("visual_chains", visual_history["changed_blackboard_fields"])

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_pdf_literature_structures")
        self.assertEqual(first["payload"]["source_ref"], "doi:second")
        self.assertEqual(first["payload"]["pdf_path"], "/tmp/second.pdf")

    def test_pdf_extraction_prefers_downloaded_doi_process_anchor_over_weaker_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            acs_pdf = tmp_path / "acs.pdf"
            process_pdf = tmp_path / "process.pdf"
            lecture_pdf = tmp_path / "lecture.pdf"
            for path in (acs_pdf, process_pdf, lecture_pdf):
                path.write_bytes(b"%PDF-1.4\n% test\n")
            board = {
                "case_id": "atorvastatin_case",
                "target_profile": {
                    "valid": True,
                    "target_name": "atorvastatin",
                    "family_hint": "statin atorvastatin free acid",
                    "functional_handles": ["statin", "atorvastatin"],
                },
                "target_side_disconnection_hypotheses": {"hypotheses": [{"hypothesis_id": "h1"}]},
                "current_belief": {"next_action_bias": ["extract_pdf_literature_structures"]},
                "literature_evidence": {
                    "source_candidates": [
                        {
                            "source_ref": "src_003",
                            "doi": "10.1021/jm00105a056",
                            "title": "Inhibitors of Cholesterol Biosynthesis. 3. Pyrrole HMG-CoA reductase inhibitors",
                            "local_pdf": str(acs_pdf),
                            "access_status": "local_pdf_available",
                            "expected_scheme_or_compound_labels": ["atorvastatin precursor pharmacophore"],
                        },
                        {
                            "source_ref": "doi:10.1186/s13065-015-0082-7",
                            "doi": "10.1186/s13065-015-0082-7",
                            "title": "pdfreq_10.1186_s13065-015-0082-7",
                            "local_pdf": str(process_pdf),
                            "access_status": "local_pdf_available",
                            "source_role": "local_pdf_proxy_download",
                        },
                        {
                            "source_ref": "src_web_003",
                            "title": "The Story of LIPITOR - A Peek into the World of Pharmaceutical Process Chemistry",
                            "local_pdf": str(lecture_pdf),
                            "access_status": "local_pdf_available",
                            "source_role": "local_pdf_proxy_download",
                        },
                    ],
                    "pdf_structure_evidence": [],
                },
            }

            batch = plan_action_batch(board, round_index=2, max_actions=1)

        self.assertEqual(batch["actions"][0]["action_type"], "extract_pdf_literature_structures")
        self.assertEqual(batch["actions"][0]["payload"]["source_ref"], "doi:10.1186/s13065-015-0082-7")
        self.assertEqual(batch["actions"][0]["payload"]["pdf_path"], str(process_pdf))

    def test_pdf_default_injection_binds_best_blackboard_downloaded_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            process_pdf = tmp_path / "process.pdf"
            lecture_pdf = tmp_path / "lecture.pdf"
            for path in (process_pdf, lecture_pdf):
                path.write_bytes(b"%PDF-1.4\n% test\n")
            board = {
                "target_profile": {
                    "target_name": "atorvastatin",
                    "family_hint": "statin atorvastatin free acid",
                    "functional_handles": ["statin", "atorvastatin"],
                },
                "literature_evidence": {
                    "source_candidates": [
                        {
                            "source_ref": "src_web_003",
                            "title": "The Story of LIPITOR - A Peek into the World of Pharmaceutical Process Chemistry",
                            "local_pdf": str(lecture_pdf),
                            "access_status": "local_pdf_available",
                            "source_role": "local_pdf_proxy_download",
                        },
                        {
                            "source_ref": "doi:10.1186/s13065-015-0082-7",
                            "doi": "10.1186/s13065-015-0082-7",
                            "title": "pdfreq_10.1186_s13065-015-0082-7",
                            "local_pdf": str(process_pdf),
                            "access_status": "local_pdf_available",
                            "source_role": "local_pdf_proxy_download",
                        },
                    ]
                },
            }
            payload = {}

            _inject_pdf_defaults(payload, {"target_name": "atorvastatin"}, blackboard=board)

        self.assertEqual(payload["source_ref"], "doi:10.1186/s13065-015-0082-7")
        self.assertEqual(payload["pdf_path"], str(process_pdf))

    def test_agentic_guided_payload_is_valid_chemenzy_policy(self):
        target = TargetInput(target_name="bufotalin", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "bridge:polycyclic_core",
                "task_type": "target_proximal_bridge",
                "target_handle": "polycyclic_cage_core",
                "required_bridge": "target-proximal cage intermediate",
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:10.0000/source"]
        board["analogical_hypothesis_ranking"] = {
            "selected_hypotheses": [
                {
                    "hypothesis_id": "target_side_polycyclic_cage_core_preservation",
                    "no_solved_claim": True,
                }
            ]
        }

        payload = build_agentic_guided_payload(board)
        validation = validate_chem_enzy_search_policy(payload["search_policy"])

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(payload["search_policy"]["case_id"], board["case_id"])
        self.assertIn("doi:10.0000/source", payload["search_policy"]["evidence_refs"])
        self.assertEqual(payload["search_policy"]["rerun_reason"], "agentic_blackboard_bridge_tasks_available")

    def test_analogical_template_validation_rejects_raw_and_missing_scope_gap(self):
        template = {
            "schema_version": "analogical_reaction_template.v1",
            "template_id": "bad_tpl",
            "relation_type": "analog",
            "reaction_class": "esterification",
            "mechanistic_class": "acyl_substitution",
            "reaction_center": {"product_retron_type": "aryl_ester_acyl_oxygen"},
            "template_radius": "r1",
            "source_refs": ["doi:analog"],
            "confidence": "medium",
            "no_solved_claim": True,
            "not_raw_reaction_injection": True,
            "rxn_smiles": "CCO>>CC=O",
        }

        validation = validate_analogical_reaction_template(template)

        self.assertFalse(validation["accepted"])
        self.assertIn("analog_template_missing_scope_gap", validation["reasons"])
        self.assertIn("raw_reaction_injection", validation["reasons"])

    def test_analogical_template_extract_rank_and_apply_to_aryl_ester_target(self):
        target = TargetInput(target_name="MLA analog", target_smiles=MLA_LIKE_SMILES, family_hint="MLA alkaloid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [
                {
                    "hypothesis_id": "h_aryl_ester",
                    "target_handle": "aryl_ester_or_anthranilate_sidechain",
                    "proposed_disconnection_region": "aryl ester",
                }
            ]
        }
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["literature_evidence"]["source_refs"] = ["doi:analog"]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=preflight["case_id"],
            target_smiles=MLA_LIKE_SMILES,
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(blackboard=board, target_smiles=MLA_LIKE_SMILES)

        self.assertTrue(extracted["accepted"], extracted["reasons"])
        self.assertTrue(ranking["accepted"], ranking["reasons"])
        self.assertTrue(applied["accepted"], applied["reasons"])
        self.assertEqual(applied["executable_candidate_count"], 1)
        self.assertTrue(applied["no_solved_claim"])
        self.assertNotEqual(applied["applications"][0].get("route_status"), "solved")
        self.assertNotEqual(applied["applications"][0].get("verdict"), "solved")

    def test_analogical_template_extracts_steroid_core_advisory_seed(self):
        target = TargetInput(target_name="steroid target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["analogical_hypotheses"] = [
            {
                "hypothesis_id": "h_core",
                "target_handle": "polycyclic_cage_core",
                "proposed_disconnection_region": "peripheral functionalization while retaining the steroid core",
                "expected_precursor_type": "target-proximal same-core steroid intermediate",
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:steroid"]
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:steroid",
                "title": "Analog steroid route",
                "source_type": "reaction_precedent",
                "relevance_rationale": "steroid family precedent",
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=BUFOTALIN_SMILES,
            max_templates=4,
            radius_policy="auto",
        )

        retrons = {
            (row.get("reaction_center") or {}).get("product_retron_type")
            for row in extracted["templates"]
        }
        self.assertTrue(extracted["accepted"], extracted["reasons"])
        self.assertIn("steroid_core_retention_bridge", retrons)
        self.assertTrue(extracted["no_solved_claim"])

    def test_analogical_template_applies_broad_reaction_center_hypotheses_to_target1(self):
        target1 = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        target = TargetInput(
            target_name="target_molecule_1",
            target_smiles=target1,
            family_hint="steroid ouabagenin analog enone alcohol",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["analogical_hypotheses"] = [
            {
                "hypothesis_id": "h_core",
                "target_handle": "polycyclic_cage_core",
                "proposed_disconnection_region": "peripheral functionalization while retaining the steroid core",
                "expected_precursor_type": "target-proximal same-core steroid intermediate",
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:steroid"]
        board["literature_evidence"]["source_candidates"] = [
            {
                "source_ref": "doi:steroid",
                "title": "Analog steroid route",
                "source_type": "reaction_precedent",
                "relevance_rationale": "same-core steroid redox and protection precedent",
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=target1,
            max_templates=8,
            radius_policy="broad",
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(
            blackboard=board,
            target_smiles=target1,
            confidence_threshold="low",
        )

        retrons = {
            (row.get("reaction_center") or {}).get("product_retron_type")
            for row in extracted["templates"]
        }
        self.assertIn("steroid_core_retention_bridge", retrons)
        self.assertIn("steroid_carbonyl_redox_adjustment", retrons)
        self.assertIn("steroid_alcohol_protection_redox_adjustment", retrons)
        self.assertTrue(applied["accepted"], applied["reasons"])
        self.assertGreaterEqual(applied["accepted_application_count"], 1)
        self.assertEqual(applied["executable_candidate_count"], 0)
        accepted = [row for row in applied["applications"] if row.get("accepted")]
        self.assertTrue(any(row.get("allowed_use") == "hypothesis_only_not_solved" for row in accepted))
        self.assertTrue(all(row.get("no_solved_claim") for row in accepted))
        self.assertTrue(all(row.get("not_parent_route_proof") for row in accepted))
        precursor_hints = [
            hint
            for row in accepted
            for hint in row.get("hypothetical_precursor_hints") or []
            if isinstance(hint, dict)
        ]
        self.assertGreaterEqual(len(precursor_hints), 1)
        self.assertTrue(all(hint.get("allowed_use") == "guided_search_subgoal_hint_only" for hint in precursor_hints))
        self.assertTrue(all(hint.get("not_parent_route_proof") for hint in precursor_hints))
        self.assertTrue(all(Chem.MolFromSmiles(str(hint.get("precursor_smiles") or "")) is not None for hint in precursor_hints))
        self.assertIn(
            "same_core_redox_or_protection_state_precursor",
            {hint.get("candidate_kind") for hint in precursor_hints},
        )

    def test_exploratory_visual_chain_drives_templates_not_exact_compile(self):
        target1 = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        visual_precursor = "O=C1CCC2(C)C(=CCC3(O)C2CCC4(C)C3CCC4C(CO)C)C=C1"
        target = TargetInput(target_name="target_molecule_1", target_smiles=target1, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [{"hypothesis_id": "h_core", "target_handle": "polycyclic_cage_core"}]
        }
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "cortistatin_total_synthesis",
                "local_pdf": "/tmp/cortistatin.pdf",
                "source_type": "literature_metadata+local_pdf",
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="cortistatin_total_synthesis",
                pdf_path="/tmp/cortistatin.pdf",
            )
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "cortistatin_total_synthesis",
                "accepted": True,
                "candidate_step_count": 1,
                "acceptance_level": "exploratory_connectivity_candidate",
                "exact_ready": False,
                "exploratory_accepted": True,
                "steps": [
                    {
                        "step_id": "step_26_to_1",
                        "product_smiles": "O=C1CCC2(C)C(CCC3(O)C2CCC4(C)C3CCC4C(CO)C)=C1",
                        "reactant_smiles": [visual_precursor],
                        "allowed_use": "exploratory_template_and_guided_hint_only",
                        "not_exact_literature_segment": True,
                        "stereochemistry_status": "unspecified_or_partial",
                        "risk_flags": ["stereochemistry_unspecified"],
                    }
                ],
            }
        ]
        board["action_history"] = [
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
        ]
        board["budget_state"]["visual_calls"] = 2

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]
        guided_payload = build_agentic_guided_payload(board)

        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertIn(visual_precursor, guided_payload["search_policy"]["source_budget"]["preferred_precursor_smiles"])
        self.assertIn(visual_precursor, guided_payload["search_policy"]["preferred_subgoal"]["preferred_subgoals"])
        self.assertTrue(guided_payload["search_policy"]["source_budget"]["visual_connectivity_hints_are_not_proof"])
        validation = validate_chem_enzy_search_policy(guided_payload["search_policy"])
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_visual_connectivity_candidate_becomes_low_confidence_template_hint(self):
        target1 = "O=C1CC[C@@]2(C)C(CC[C@]3(O)C2CC[C@@]4(C)C3CCC4[C@@H](CO)C)=C1"
        visual_precursor = "O=C1CCC2(C)C(=CCC3(O)C2CCC4(C)C3CCC4C(CO)C)C=C1"
        visual_precursor_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(visual_precursor), isomericSmiles=True)
        target = TargetInput(target_name="target_molecule_1", target_smiles=target1, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["literature_evidence"]["source_refs"] = ["cortistatin_total_synthesis"]
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "cortistatin_total_synthesis", "source_type": "reaction_precedent"}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "source_ref": "cortistatin_total_synthesis",
                "accepted": True,
                "candidate_step_count": 1,
                "acceptance_level": "exploratory_connectivity_candidate",
                "exact_ready": False,
                "exploratory_accepted": True,
                "steps": [
                    {
                        "product_smiles": "O=C1CCC2(C)C(CCC3(O)C2CCC4(C)C3CCC4C(CO)C)=C1",
                        "reactant_smiles": [visual_precursor],
                        "source_locator": "Scheme 2, compound 26 to 1",
                        "allowed_use": "exploratory_template_and_guided_hint_only",
                        "not_exact_literature_segment": True,
                    }
                ],
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=target1,
            max_templates=8,
            radius_policy="broad",
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(
            blackboard=board,
            target_smiles=target1,
            confidence_threshold="low",
        )

        visual_templates = [
            row
            for row in extracted["templates"]
            if (row.get("reaction_center") or {}).get("product_retron_type") == "steroid_visual_unsaturation_adjustment"
        ]
        self.assertTrue(visual_templates)
        self.assertEqual(visual_templates[0]["visual_connectivity_hint"]["precursor_smiles"], visual_precursor_canonical)
        accepted = [row for row in applied["applications"] if row.get("accepted")]
        precursor_hints = [
            hint
            for row in accepted
            for hint in row.get("hypothetical_precursor_hints") or []
            if isinstance(hint, dict)
        ]
        self.assertTrue(any(hint.get("precursor_smiles") == visual_precursor_canonical for hint in precursor_hints))
        self.assertTrue(all(hint.get("not_exact_literature_segment") for hint in precursor_hints))

    def test_atorvastatin_visual_chain_does_not_generate_steroid_templates(self):
        visual_precursor = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)n(CCC(O)CC(O)CC(=O)OC(C)(C)C)c1-c1ccccc1"
        visual_precursor_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(visual_precursor), isomericSmiles=True)
        target = TargetInput(target_name="atorvastatin", target_smiles=ATORVASTATIN_FREE_ACID_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["literature_evidence"]["source_refs"] = ["doi:10.1186/s13065-015-0082-7"]
        board["literature_evidence"]["source_candidates"] = [
            {"source_ref": "doi:10.1186/s13065-015-0082-7", "source_type": "reaction_precedent"}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "accepted": True,
                "candidate_step_count": 1,
                "acceptance_level": "exploratory_connectivity_candidate",
                "exact_ready": False,
                "exploratory_accepted": True,
                "steps": [
                    {
                        "product_label": "atorvastatin acid",
                        "product_smiles": ATORVASTATIN_FREE_ACID_SMILES,
                        "reactant_smiles": [visual_precursor],
                        "source_locator": "Scheme 2, conversion of 5 to 1",
                        "condition_candidate": {"reagent": "NaOH, then calcium acetate monohydrate"},
                        "allowed_use": "exploratory_template_and_guided_hint_only",
                        "not_exact_literature_segment": True,
                    }
                ],
            }
        ]

        extracted = extract_analogical_reaction_templates_from_blackboard(
            blackboard=board,
            case_id=str(preflight["case_id"]),
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            max_templates=6,
            radius_policy="broad",
        )
        board["analogical_templates"] = extracted["templates"]
        ranking = rank_analogical_reaction_templates_from_blackboard(board)
        board["analogical_template_ranking"] = ranking
        applied = apply_analogical_templates_to_target(
            blackboard=board,
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            confidence_threshold="low",
        )

        retrons = [(row.get("reaction_center") or {}).get("product_retron_type") for row in extracted["templates"]]
        self.assertIn("visual_hydrolysis_salt_bridge", retrons)
        self.assertFalse(any("steroid" in str(retron or "") for retron in retrons))
        accepted = [row for row in applied["applications"] if row.get("accepted")]
        precursor_hints = [
            hint
            for row in accepted
            for hint in row.get("hypothetical_precursor_hints") or []
            if isinstance(hint, dict)
        ]
        self.assertTrue(any(hint.get("precursor_smiles") == visual_precursor_canonical for hint in precursor_hints))

    def test_planner_selects_analogical_template_actions_before_guided(self):
        target = TargetInput(target_name="MLA analog", target_smiles=MLA_LIKE_SMILES, family_hint="MLA alkaloid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [
                {
                    "hypothesis_id": "h_aryl_ester",
                    "target_handle": "aryl_ester_or_anthranilate_sidechain",
                    "proposed_disconnection_region": "aryl ester",
                }
            ]
        }
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_aryl_ester"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:analog",
                "title": "Analog esterification precedent",
                "doi": "10.1000/analog",
                "source_type": "reaction_precedent",
                "relevance_rationale": "analog aryl ester precedent",
                "access_status": "metadata_only",
                "no_solved_claim": True,
            }
        ]
        board["literature_evidence"]["source_refs"] = ["doi:analog"]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        template_action = next(row for row in batch["actions"] if row["action_type"] == "extract_analogical_reaction_templates")
        self.assertTrue(template_action["payload"]["analogical_template_policy"]["analogy_is_advisory_only"])
        self.assertEqual(
            template_action["payload"]["analogical_template_policy"]["final_verdict_authority"],
            "deterministic_parent_route_proof",
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_agentic_controller_compiles_validated_analogical_template_without_final_solved(self):
        def planner(**kwargs):
            round_index = kwargs["round_index"]
            actions_by_round = {
                1: ["generate_disconnection_hypotheses"],
                2: ["extract_analogical_reaction_templates"],
                3: ["rank_analogical_reaction_templates"],
                4: ["apply_analogical_template_to_target"],
                5: ["validate_template_application"],
            }
            return {
                "schema_version": "agent_action_batch.v1",
                "case_id": "mla_analog",
                "round_index": round_index,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": f"r{round_index}:{action_type}",
                        "action_type": action_type,
                        "rationale": "template test",
                        "expected_artifact": "typed template artifact",
                        "success_condition": "typed artifact or explicit rejection",
                        "payload": {
                            **_test_analogical_template_payload(action_type),
                            "max_templates": 4,
                            "max_applications": 3,
                            "template_radius_policy": "auto",
                            "analog_template_confidence_threshold": "low",
                        },
                    }
                    for action_type in actions_by_round.get(round_index, [])
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            result = run_agentic_blackboard_controller(
                target_name="MLA analog",
                target_smiles=MLA_LIKE_SMILES,
                family_hint="MLA alkaloid",
                output_dir=tmp,
                max_rounds=5,
                action_planner=planner,
            )
            compiled = json.loads((Path(tmp) / "compiled_analogical_template_hints.json").read_text(encoding="utf-8"))
            disabled_exact_plugin = json.loads(
                (Path(tmp) / "compiled_literature_template_plugin.json").read_text(encoding="utf-8")
            )
            audit = json.loads((Path(tmp) / "agentic_run_audit.json").read_text(encoding="utf-8"))

        board = result["agent_blackboard"]
        self.assertGreaterEqual(len(board["analogical_templates"]), 1)
        self.assertGreaterEqual(len(board["template_applications"]), 1)
        self.assertEqual(compiled["schema_version"], "analogical_template_guided_hints.v1")
        self.assertTrue(compiled["analogy_is_advisory_only"])
        self.assertTrue(compiled["not_exact_literature_segment"])
        self.assertTrue(compiled["not_parent_route_proof"])
        self.assertFalse(compiled["literature_template_plugin"]["plugin_flags"]["enabled"])
        self.assertEqual(compiled["literature_template_plugin"]["one_step_rows"], [])
        self.assertFalse(disabled_exact_plugin["plugin_flags"]["enabled"])
        self.assertEqual(disabled_exact_plugin["one_step_rows"], [])
        hints = compiled["analogical_template_hints"]
        self.assertFalse(hints["plugin_flags"]["enabled"])
        self.assertTrue(hints["plugin_flags"]["guided_hint_enabled"])
        self.assertEqual(len(hints["one_step_rows"]), 1)
        self.assertEqual(hints["one_step_rows"][0]["allowed_use"], "guided_search_hint_only")
        self.assertEqual(hints["one_step_rows"][0]["source_policy_decision"], "analogical_guided_hint_only")
        self.assertTrue(hints["one_step_rows"][0]["not_exact_literature_segment"])
        self.assertFalse(hints["one_step_rows"][0]["used_as_proof"])
        self.assertNotIn("compiled_downstream", result["artifact_bundle"]["artifacts"])
        self.assertFalse(result["final_verdict"]["solved"])
        self.assertNotEqual(result["final_verdict"]["verdict"], "solved")
        artifacts = result["artifact_bundle"]["artifacts"]
        self.assertEqual(artifacts["analogical_reaction_template_report"]["artifact_type"], "AnalogicalReactionTemplateReport")
        self.assertEqual(artifacts["analogical_reaction_template_ranking_artifact"]["artifact_type"], "AnalogicalReactionTemplateRanking")
        self.assertEqual(artifacts["analogical_template_application_report"]["artifact_type"], "AnalogicalTemplateApplicationReport")
        self.assertEqual(
            artifacts["analogical_template_application_validation_artifact"]["artifact_type"],
            "AnalogicalTemplateApplicationValidation",
        )
        accepted_keys = audit["payload"]["typed_artifact_validation_summary"]["accepted_artifact_keys"]
        self.assertIn("analogical_reaction_template_report", accepted_keys)
        self.assertIn("analogical_reaction_template_ranking", accepted_keys)
        self.assertIn("analogical_template_application_report", accepted_keys)
        self.assertIn("analogical_template_application_validation", accepted_keys)
        self.assertTrue(audit["payload"]["analogical_template_summary"]["analogy_is_advisory_only"])
        self.assertEqual(audit["payload"]["analogical_template_summary"]["final_verdict_authority"], "none")
        self.assertEqual(audit["payload"]["analogical_template_summary"]["validated_one_step_row_count"], 1)

    def test_guided_payload_carries_template_hints_and_forbidden_ids(self):
        target = TargetInput(target_name="MLA analog", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["bridge_tasks"] = [{"task_id": "bridge:ester", "task_type": "target_proximal_bridge"}]
        board["analogical_template_ranking"] = {
            "selected_templates": [{"template_id": "tpl_good", "no_solved_claim": True}]
        }
        board["template_applications"] = [
            {
                "application_id": "app_good",
                "template_id": "tpl_good",
                "accepted": True,
                "allowed_use": "executable_candidate",
                "product_retron_type": "aryl_ester_acyl_oxygen",
                "executable_candidate_available": True,
            },
            {
                "application_id": "app_hypothesis",
                "template_id": "tpl_hypothesis",
                "accepted": True,
                "allowed_use": "hypothesis_only_not_solved",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "executable_candidate_available": False,
                "hypothetical_route_hypothesis": {
                    "schema_version": "analogical_route_hypothesis.v1",
                    "route_status": "hypothesis_only_not_solved",
                    "hypothesis_type": "carbonyl_redox_adjustment",
                    "reaction_center_idea": "transfer analog steroid redox logic",
                    "expected_precursor_type": "same-core hydroxy steroid",
                    "template_application": "prefer late-stage redox variants",
                    "required_verification": ["route_verifier", "parent_route_proof"],
                    "risk_flags": ["selectivity_not_proven"],
                    "no_solved_claim": True,
                },
                "hypothetical_precursor_hints": [
                    {
                        "schema_version": "analogical_hypothesis_precursor_hint.v1",
                        "hint_id": "hyp_precursor_1",
                        "precursor_smiles": "CCO",
                        "precursor_role": "same_core_hydroxy_steroid_carbonyl_precursor",
                        "derived_from_retron": "steroid_carbonyl_redox_adjustment",
                        "hypothesis_type": "carbonyl_redox_adjustment",
                        "candidate_kind": "same_core_redox_or_protection_state_precursor",
                        "allowed_use": "guided_search_subgoal_hint_only",
                        "risk_flags": ["selectivity_not_proven"],
                        "not_exact_literature_segment": True,
                        "not_parent_route_proof": True,
                        "requires_verifier": True,
                        "no_solved_claim": True,
                    }
                ],
            }
        ]
        board["template_failure_memory"] = [
            {"template_id": "tpl_bad", "failure_count": 2, "reasons": ["no_retron_match"]}
        ]

        payload = build_agentic_guided_payload(board)
        policy = payload["search_policy"]

        self.assertIn("tpl_good", policy["selected_analogical_template_ids"])
        self.assertIn("tpl_bad", policy["forbidden_template_ids"])
        self.assertEqual(policy["preferred_subgoal"]["template_application_hints"][0]["template_id"], "tpl_good")
        self.assertTrue(policy["source_budget"]["analogy_is_advisory_only"])
        self.assertIn("steroid_carbonyl_redox_adjustment", policy["source_budget"]["preferred_reaction_classes"])
        hypothesis_hints = policy["preferred_subgoal"]["hypothetical_reaction_center_hints"]
        self.assertEqual(hypothesis_hints[0]["template_id"], "tpl_hypothesis")
        self.assertEqual(hypothesis_hints[0]["hypothesis"]["hypothesis_type"], "carbonyl_redox_adjustment")
        self.assertTrue(hypothesis_hints[0]["not_parent_route_proof"])
        self.assertEqual(policy["anchor_whitelist"], [])
        self.assertIn("CCO", policy["preferred_subgoal"]["preferred_subgoals"])
        self.assertIn("CCO", policy["source_budget"]["preferred_precursor_smiles"])
        precursor_targets = policy["preferred_subgoal"]["hypothetical_precursor_targets"]
        self.assertEqual(precursor_targets[0]["smiles"], "CCO")
        self.assertEqual(precursor_targets[0]["allowed_use"], "guided_search_subgoal_hint_only")
        self.assertTrue(precursor_targets[0]["not_exact_literature_segment"])
        self.assertTrue(precursor_targets[0]["not_parent_route_proof"])
        self.assertTrue(policy["source_budget"]["hypothetical_route_hints_are_not_proof"])
        self.assertTrue(policy["source_budget"]["hypothesis_precursor_hints_are_not_proof"])
        guided_config = apply_chem_enzy_search_policy(RouteSearchConfig(target_smiles=MLA_LIKE_SMILES), policy)
        context = guided_config.search_flags["cascade_search_context"]
        self.assertIn("CCO", context["preferred_subgoal"]["preferred_subgoals"])
        self.assertEqual(context["preferred_subgoal"]["hypothetical_precursor_targets"][0]["smiles"], "CCO")

    def test_planner_expands_hypothetical_precursor_candidates_as_child_targets(self):
        target = TargetInput(target_name="hypothesis precursor target", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=3)
        board["literature_evidence"]["exact_rows"] = [{"row_id": "placeholder_exact_row_blocks_scout"}]
        board["template_applications"] = [
            {
                "application_id": "app_hypothesis",
                "template_id": "tpl_hypothesis",
                "accepted": True,
                "allowed_use": "hypothesis_only_not_solved",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "executable_candidate_available": False,
                "hypothetical_precursor_hints": [
                    {
                        "schema_version": "analogical_hypothesis_precursor_hint.v1",
                        "hint_id": "hyp_precursor_1",
                        "precursor_smiles": "CCO",
                        "precursor_role": "same_core_hydroxy_steroid_carbonyl_precursor",
                        "derived_from_retron": "steroid_carbonyl_redox_adjustment",
                        "hypothesis_type": "carbonyl_redox_adjustment",
                        "allowed_use": "guided_search_subgoal_hint_only",
                        "not_exact_literature_segment": True,
                        "not_parent_route_proof": True,
                        "requires_verifier": True,
                        "no_solved_claim": True,
                    }
                ],
            }
        ]

        batch = plan_action_batch(board, round_index=2, max_actions=3)
        validation = validate_action_batch(batch, blackboard=board)
        child_actions = [action for action in batch["actions"] if action["action_type"] == "expand_child_target"]

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(len(child_actions), 1)
        payload = child_actions[0]["payload"]
        self.assertEqual(payload["subgoal_targets"][0]["smiles"], "CCO")
        self.assertTrue(payload["subgoal_targets"][0]["hypothesis_only_not_solved"])
        policy = payload["subgoal_targets"][0]["chem_enzy_search_policy"]
        self.assertEqual(policy["anchor_whitelist"], [])
        self.assertTrue(policy["source_budget"]["hypothesis_precursor_hint"])
        self.assertTrue(policy["source_budget"]["hypothesis_precursor_hints_are_not_proof"])

    def test_planner_advances_to_unattempted_hypothetical_precursor_child_targets(self):
        target = TargetInput(target_name="hypothesis precursor target", target_smiles=MLA_LIKE_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "placeholder_exact_row_blocks_scout"}]
        hints = []
        for idx, smiles in enumerate(["CCO", "CCN", "CCC"], start=1):
            hints.append(
                {
                    "schema_version": "analogical_hypothesis_precursor_hint.v1",
                    "hint_id": f"hyp_precursor_{idx}",
                    "precursor_smiles": smiles,
                    "precursor_role": f"same_core_precursor_{idx}",
                    "derived_from_retron": "steroid_carbonyl_redox_adjustment",
                    "hypothesis_type": "carbonyl_redox_adjustment",
                    "allowed_use": "guided_search_subgoal_hint_only",
                    "not_exact_literature_segment": True,
                    "not_parent_route_proof": True,
                    "requires_verifier": True,
                    "no_solved_claim": True,
                }
            )
        board["template_applications"] = [
            {
                "application_id": "app_hypothesis",
                "template_id": "tpl_hypothesis",
                "accepted": True,
                "allowed_use": "hypothesis_only_not_solved",
                "product_retron_type": "steroid_carbonyl_redox_adjustment",
                "executable_candidate_available": False,
                "hypothetical_precursor_hints": hints,
            }
        ]
        board["route_failures"] = [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": "large_atom_jump",
                "route_status": "fake_closed_rejected",
            }
        ]
        board["action_history"] = [
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "build_failure_critic_report",
                "status": "accepted",
                "useful_artifact": True,
                "stale": False,
                "reasons": [],
                "action_signature": "{}",
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 2,
                "action_type": "expand_child_target",
                "status": "accepted",
                "useful_artifact": True,
                "stale": False,
                "reasons": ["no_route_expansion_subgoal_verified_solved"],
                "action_signature": json.dumps(
                    {
                        "action_type": "expand_child_target",
                        "payload": {
                            "subgoal_targets": [
                                {"smiles": "CCO"},
                                {"smiles": "CCN"},
                            ],
                        },
                    },
                    sort_keys=True,
                ),
            },
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 3,
                "action_type": "stitch_parent_route",
                "status": "rejected",
                "useful_artifact": True,
                "stale": False,
                "reasons": ["subgoal_verifier_not_accepted"],
                "action_signature": "{}",
            },
        ]

        batch = plan_action_batch(board, round_index=4, max_actions=3, exhaust_round_budget=True)
        validation = validate_action_batch(batch, blackboard=board)
        child_actions = [action for action in batch["actions"] if action["action_type"] == "expand_child_target"]

        self.assertTrue(validation["accepted"], validation["reasons"])
        self.assertEqual(len(child_actions), 1)
        payload = child_actions[0]["payload"]
        self.assertEqual([row["smiles"] for row in payload["subgoal_targets"]], ["CCC"])
        self.assertEqual(payload["max_targets"], 1)

    def test_budget_exhaustive_planner_changes_direction_after_stale_rounds(self):
        target = TargetInput(target_name="stale_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["route_failures"] = [
            {
                "schema_version": "agent_route_failure.v1",
                "reason": "large_atom_jump",
                "route_status": "fake_closed_rejected",
            }
        ]
        board["action_history"] = [
            {
                "round_index": 1,
                "action_type": "compile_exact_literature_rows",
                "action_signature": "{}",
                "useful_artifact": False,
                "stale": True,
            },
            {
                "round_index": 2,
                "action_type": "extract_visual_literature_chain",
                "action_signature": "{}",
                "useful_artifact": False,
                "stale": True,
            },
        ]

        default_batch = plan_action_batch(board, round_index=3)
        exhaustive_batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)

        self.assertEqual(default_batch["actions"][0]["action_type"], "stop_unresolved")
        self.assertNotEqual(exhaustive_batch["actions"][0]["action_type"], "stop_unresolved")
        self.assertEqual(exhaustive_batch["actions"][0]["action_type"], "generate_disconnection_hypotheses")

    def test_planner_defers_guided_until_local_pdf_extraction_branch_finishes(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "bridge:core",
                "task_type": "target_proximal_bridge",
                "target_handle": "core",
            }
        ]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1016/j.tet.2025.134610",
                "local_pdf": "/tmp/bufotalin.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "11"],
            }
        ]

        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_pdf_literature_structures", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)

    def test_failed_zero_step_visual_chain_does_not_block_visual_retry(self):
        target = TargetInput(target_name="single_pdf_retry", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["current_belief"]["next_action_bias"] = ["extract_visual_literature_chain"]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(source_ref="doi:source", pdf_path="/tmp/source.pdf")
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "source_ref": "doi:source",
                "accepted": False,
                "candidate_step_count": 0,
                "step_count": 0,
                "reasons": ["visual_input_images_missing"],
            }
        ]

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_visual_literature_chain", action_types)
        visual = next(row for row in batch["actions"] if row["action_type"] == "extract_visual_literature_chain")
        self.assertEqual(visual["payload"]["source_ref"], "doi:source")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_resume_indexes_legacy_pdf_evidence_for_visual_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "pages"
            image_dir.mkdir()
            image_path = image_dir / "page_001.png"
            image_path.write_bytes(b"not-a-real-png-but-path-exists")
            pdf_result = {
                "accepted": True,
                "result": {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "accepted": True,
                    "source_ref": "doi:source",
                    "source_pdf_path": str(root / "source.pdf"),
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(image_path)}
                    ],
                    "scheme_crops": [],
                    "compound_text_snippets": [],
                },
            }
            (root / "r1_extract_pdf_literature_structures_literature_pdf_structure_evidence_v1.json").write_text(
                json.dumps(pdf_result),
                encoding="utf-8",
            )

            artifacts = _load_existing_artifacts(root, {"artifact_refs": {}})

        self.assertIn("literature_pdf_structure_evidence_history", artifacts)
        self.assertIn("literature_pdf_structure_evidence_by_source", artifacts)
        self.assertEqual(len(artifacts["literature_pdf_structure_evidence_history"]), 1)
        self.assertIn("ref:doi:source", artifacts["literature_pdf_structure_evidence_by_source"])

    def test_planner_expands_frontier_even_when_more_pdf_extraction_is_pending(self):
        target = TargetInput(target_name="atorvastatin", target_smiles=ATORVASTATIN_FREE_ACID_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_route_expansion_subgoal_runs": 4, "max_visual_calls": 4},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["current_belief"]["next_action_bias"] = [
            "extract_pdf_literature_structures",
            "extract_visual_literature_chain",
            "expand_child_target",
        ]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:process",
                "doi": "10.1186/s13065-015-0082-7",
                "title": "process anchor",
                "local_pdf": "/tmp/process.pdf",
                "access_status": "local_pdf_available",
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:followup",
                "doi": "10.1021/jm00105a056",
                "title": "follow-up source",
                "local_pdf": "/tmp/followup.pdf",
                "access_status": "local_pdf_available",
            },
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {
                "schema_version": "agent_pdf_structure_evidence_summary.v1",
                "source_ref": "doi:process",
                "accepted": True,
            }
        ]
        board["literature_evidence"]["process_evidence_rows"] = [
            {
                "schema_version": "process_evidence_row.v1",
                "process_id": "process:atorvastatin_side_chain",
                "source_ref": "doi:process",
                "process_type": "statin_side_chain_installation",
                "accepted": True,
            }
        ]
        board["retrosynthetic_proposals"] = [
            {
                "schema_version": "retrosynthetic_proposal.v1",
                "proposal_id": "proposal:atorvastatin_side_chain",
                "proposal_type": "process_anchor",
                "proposal_granularity": "process",
                "precursor_smiles": "CC(C)c1ccccc1",
                "recursive_expandable": True,
                "executable": True,
                "not_exact_literature_segment": True,
            }
        ]
        board["recursive_hypothesis_tasks"] = [
            {
                "schema_version": "recursive_hypothesis_task.v1",
                "task_id": "recursive_hypothesis:proposal:atorvastatin_side_chain:1",
                "task_type": "recursive_hypothesis_frontier_expansion",
                "status": "pending",
                "source": "retrosynthetic_proposal",
                "parent_candidate_id": "proposal:atorvastatin_side_chain",
                "parent_smiles": ATORVASTATIN_FREE_ACID_SMILES,
                "precursor_smiles": "CC(C)c1ccccc1",
                "name": "atorvastatin side-chain frontier",
                "recursive_depth": 1,
                "operation_idea": "test process-derived side-chain precursor as a child target",
                "proposal_granularity": "process",
                "proposal_score": 80,
                "allowed_use": "route_expansion_subgoal_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "requires_verifier": True,
                "child_route_cannot_promote_parent": True,
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "expand_child_target")
        self.assertIn("extract_pdf_literature_structures", action_types)
        child_action = next(row for row in batch["actions"] if row["action_type"] == "expand_child_target")
        child_target = child_action["payload"]["subgoal_targets"][0]
        self.assertEqual(child_target["smiles"], "CC(C)c1ccccc1")
        self.assertEqual(child_target["source"], "recursive_hypothesis_task")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_process_evidence_does_not_block_open_structure_resolution_task(self):
        target = TargetInput(
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            family_hint="statin synthetic atorvastatin",
        )
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_visual_calls": 3},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "atorvastatin_process_anchor"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "atorvastatin_process_anchor"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "doi": "10.1186/s13065-015-0082-7",
                "title": "An improved kilogram-scale preparation of atorvastatin calcium",
                "local_pdf": "/tmp/atorvastatin_bmc.pdf",
                "access_status": "local_pdf_available",
            }
        ]
        board["literature_evidence"]["process_evidence_rows"] = [
            {
                "schema_version": "literature_process_evidence_row.v1",
                "row_id": "process_evidence:atorvastatin",
                "process_type": "small_molecule_process_route",
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "substrate_or_feedstock_labels": ["advanced ketal ester intermediate 4"],
                "endpoint_labels": ["atorvastatin calcium"],
                "no_solved_claim": True,
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:10.1186/s13065-015-0082-7",
                pdf_path="/tmp/atorvastatin_bmc.pdf",
            )
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "accepted": True,
                "candidate_step_count": 0,
                "steps": [],
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:bmc_intermediate_4",
                "task_type": "resolve_literature_structure",
                "label": "advanced ketal ester intermediate 4",
                "source_ref": "doi:10.1186/s13065-015-0082-7",
                "source_title": "An improved kilogram-scale preparation of atorvastatin calcium",
                "status": "open",
                "no_solved_claim": True,
            }
        ]

        batch = plan_action_batch(board, round_index=4, max_actions=1)

        self.assertEqual([row["action_type"] for row in batch["actions"]], ["resolve_literature_structure_task"])
        payload = batch["actions"][0]["payload"]
        self.assertEqual(payload["task_id"], "resolve_structure:bmc_intermediate_4")
        self.assertEqual(payload["label"], "advanced ketal ester intermediate 4")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_processes_multiple_local_pdfs_in_one_blackboard(self):
        target = TargetInput(target_name="multi_pdf_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:first", "local_pdf": "/tmp/first.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:second", "local_pdf": "/tmp/second.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:first",
                pdf_path="/tmp/first.pdf",
                evidence_id="doi:first",
            )
        ]

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_pdf_literature_structures")
        self.assertEqual(first["payload"]["source_ref"], "doi:second")
        self.assertEqual(first["payload"]["pdf_path"], "/tmp/second.pdf")

    def test_planner_visual_extracts_next_pdf_source_after_first_source_compiled(self):
        target = TargetInput(target_name="multi_pdf_case", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:first", "local_pdf": "/tmp/first.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:second", "local_pdf": "/tmp/second.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(source_ref="doi:first", pdf_path="/tmp/first.pdf"),
            _rendered_pdf_evidence(source_ref="doi:second", pdf_path="/tmp/second.pdf"),
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "first_visual",
                "source_ref": "doi:first",
                "accepted": True,
                "candidate_step_count": 1,
            }
        ]
        board["action_history"] = [
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "useful_artifact": True, "stale": False},
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_visual_literature_chain")
        self.assertEqual(first["payload"]["source_ref"], "doi:second")
        self.assertEqual(first["payload"]["pdf_path"], "/tmp/second.pdf")

    def test_planner_does_not_recompile_visual_chains_without_uncompiled_steps(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h_core"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:zhang", "local_pdf": "/tmp/zhang.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:reddy", "local_pdf": "/tmp/reddy.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:chen", "local_pdf": "/tmp/chen.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:zhang", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:reddy", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:chen", "accepted": True},
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "zhang_visual",
                "source_ref": "doi:zhang",
                "accepted": False,
                "candidate_step_count": 0,
                "gap_labels": ["ouabagenin"],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "reddy_visual",
                "source_ref": "doi:reddy",
                "accepted": False,
                "candidate_step_count": 0,
                "gap_labels": ["18", "19"],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "chen_visual",
                "source_ref": "doi:chen",
                "accepted": False,
                "candidate_step_count": 6,
                "gap_labels": ["21-33 protected tetracyclic intermediates"],
            },
        ]
        board["literature_evidence"]["exact_rows"] = [
            {"schema_version": "agent_exact_literature_row_summary.v1", "row_id": f"source_detail_exact_step:step_{idx}", "source_ref": "doi:chen"}
            for idx in range(6)
        ]
        board["literature_evidence"]["exact_chain_audits"] = [
            {"schema_version": "agent_exact_chain_audit_summary.v1", "audit_id": "chen_audit", "accepted": False, "one_step_row_count": 6}
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve:chen:21-33",
                "label": "21-33 protected tetracyclic intermediates",
                "source_ref": "doi:chen",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "useful_artifact": True, "stale": False},
            {"round_index": 5, "action_type": "compile_exact_literature_rows", "useful_artifact": False, "stale": True},
            {"round_index": 6, "action_type": "compile_exact_literature_rows", "useful_artifact": False, "stale": True},
        ]
        board["budget_state"]["visual_calls"] = 6

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertIn("run_guided_chemenzy", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_visual_tool_selects_pdf_evidence_by_source_ref_not_latest_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first_image = run_dir / "first.png"
            second_image = run_dir / "second.png"
            first_image.write_bytes(b"first")
            second_image.write_bytes(b"second")
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "multi_pdf_case", "target_smiles": "CCO"},
                preflight={"case_id": "multi_pdf_case"},
                budget=HarnessBudget(timeout_s=60),
            )
            state.artifacts["literature_pdf_structure_evidence_history"] = [
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "source_ref": "doi:first",
                    "source_pdf_path": "/tmp/first.pdf",
                    "rendered_pages": [{"image_path": str(first_image)}],
                },
                {
                    "schema_version": "literature_pdf_structure_evidence.v1",
                    "source_ref": "doi:second",
                    "source_pdf_path": "/tmp/second.pdf",
                    "rendered_pages": [{"image_path": str(second_image)}],
                },
            ]
            state.artifacts["literature_pdf_structure_evidence"] = state.artifacts[
                "literature_pdf_structure_evidence_history"
            ][1]

            evidence = _pdf_evidence_from_payload_or_artifacts(state, {"source_ref": "doi:first"})
            image_paths = _visual_chain_image_paths(state, {"source_ref": "doi:first"}, evidence)

        self.assertEqual(evidence["source_ref"], "doi:first")
        self.assertEqual([path.name for path in image_paths], ["first.png"])

    def test_visual_codex_prompt_disables_web_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "visual"
            output_dir.mkdir()
            image_path = root / "page.png"
            image_path.write_bytes(b"fake image")
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json, os, pathlib, sys",
                        "args = sys.argv[1:]",
                        "last_message = ''",
                        "for index, arg in enumerate(args[:-1]):",
                        "    if arg == '--output-last-message':",
                        "        last_message = args[index + 1]",
                        "if last_message:",
                        "    pathlib.Path(last_message).write_text(json.dumps({'schema_version':'visual_structure_candidate_chain.v1','steps':[]}), encoding='utf-8')",
                        "config = pathlib.Path(os.environ['CODEX_HOME']) / 'config.toml'",
                        "pathlib.Path.cwd().joinpath('captured_codex_invocation.json').write_text(",
                        "    json.dumps({'argv': args, 'config': config.read_text(encoding='utf-8')}),",
                        "    encoding='utf-8',",
                        ")",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)

            result = _run_codex_visual_prompt(
                executable=str(fake_codex),
                api_key="sk-test",
                base_url="https://example.test/v1",
                model="fake-model",
                output_dir=output_dir,
                image_paths=[image_path],
                prompt="return json",
                timeout_s=5.0,
                prompt_filename="prompt.txt",
                event_log_filename="events.jsonl",
                stderr_log_filename="stderr.log",
                last_message_filename="last_message.txt",
            )
            captured = json.loads((output_dir / "captured_codex_invocation.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "completed")
        self.assertNotIn("--search", captured["argv"])
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", captured["argv"])
        self.assertIn("--sandbox", captured["argv"])
        self.assertIn("workspace-write", captured["argv"])
        self.assertIn("web_search = false", captured["config"])
        self.assertNotIn("web_search = true", captured["config"])

    def test_visual_prompt_standardizes_conditions_under_condition_candidate(self):
        prompt = _visual_literature_prompt(
            target_name="target",
            target_smiles="CCO",
            source_ref="doi:10.example/visual",
            source_title="Visual source",
            expected_labels=["compound 1"],
            route_sequence_hint="",
            text_snippets=[],
        )

        self.assertIn("condition_candidate only", prompt)
        self.assertIn("Do not emit parallel condition aliases", prompt)
        self.assertIn("condition_text, reaction_conditions, visible_conditions, conditions, condition, or forward_conditions", prompt)
        self.assertIn('"condition_candidate"', prompt)

    def test_direct_visual_prompt_uses_api_payload_without_codex_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "visual"
            output_dir.mkdir()
            image_path = root / "page.png"
            image_path.write_bytes(b"fake image bytes")
            captured = {}

            def fake_post(**kwargs):
                captured.update(kwargs)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "schema_version": "visual_structure_candidate_chain.v1",
                                        "case_id": "direct_visual",
                                        "steps": [],
                                    }
                                )
                            }
                        }
                    ]
                }

            with patch(
                "cascade_planner.harness.visual_literature_chain_agent._post_visual_api_json",
                side_effect=fake_post,
            ):
                result = _run_direct_visual_prompt(
                    api_key="sk-test",
                    base_url="https://api.wellau.com/v1",
                    model="fake-model",
                    output_dir=output_dir,
                    image_paths=[image_path],
                    prompt="return json",
                    timeout_s=5.0,
                    prompt_filename="prompt.txt",
                    event_log_filename="events.jsonl",
                    stderr_log_filename="stderr.log",
                    last_message_filename="last_message.txt",
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["execution_mode"], "direct_visual_api")
        self.assertEqual(captured["endpoint"], "chat/completions")
        user_content = captured["payload"]["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(json.loads(result["raw_last_message"])["schema_version"], "visual_structure_candidate_chain.v1")

    def test_visual_parser_preserves_condition_only_repair_steps(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source": {"doi": "10.1021/ja8023466", "title": "Synthesis of (+)-Cortistatin A"},
            "steps": [
                {
                    "step_id": "step_1",
                    "visible_product_label": "cortistatinone",
                    "product_smiles": "CCO",
                    "source_scheme": "Scheme 2",
                    "source_grounding": "visible arrow and condition block",
                    "reaction_conditions": {
                        "reagent": "Dess-Martin periodinane",
                        "solvent": "CH2Cl2",
                        "temperature": "room temperature",
                        "condition_text_transcribed": "DMP, CH2Cl2, rt",
                    },
                },
                {
                    "step_id": "step_2",
                    "mapped_candidate_label": "cortistatin A",
                    "product_smiles": "CCN",
                    "source_scheme": "Scheme 3",
                    "reaction_conditions": {
                        "reagent": "reducing conditions",
                        "reported_yield": "visible yield",
                    },
                },
            ],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="steroid_target",
            target_smiles=BUFOTALIN_ACHIRAL_SMILES,
            source_ref="doi:10.1021/ja8023466",
            source_title="Synthesis of (+)-Cortistatin A",
            image_paths=[],
        )
        quality = _candidate_quality(chain, expected_labels=["cortistatinone", "cortistatin A"])

        self.assertEqual(len(chain["steps"]), 2)
        self.assertEqual(chain["steps"][0]["product_label"], "cortistatinone")
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "Dess-Martin periodinane")
        self.assertTrue(chain["steps"][0]["structure_derivation"]["visual_structure_anchor_only"])
        self.assertEqual(quality["smiles_precheck"]["invalid_smiles_count"], 0)
        self.assertEqual(quality["structure_gap_count"], 0)
        self.assertEqual(quality["rdkit_structure_anchor_count"], 2)
        self.assertFalse(quality["exact_ready"])
        self.assertTrue(quality["accepted"])

    def test_visual_parser_accepts_achiral_connectivity_candidate_as_exploratory(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source_ref": "doi:10.0000/analog",
            "source_title": "Analog steroid source",
            "route_order": "retro_target_to_start",
            "confidence": "low",
            "steps": [
                {
                    "step_id": "approx_step_1",
                    "product_label": "drawn alcohol",
                    "product_smiles": "CCO",
                    "reactant_labels": ["drawn aldehyde"],
                    "reactant_smiles": ["CC=O"],
                    "main_reactant_smiles": "CC=O",
                    "source_locator": "Scheme 1",
                    "condition_candidate": {"reagent": "NaBH4", "source_grounding": "Scheme 1"},
                    "structure_derivation": {
                        "basis": "current_pdf_image_to_achiral_or_approximate_smiles",
                        "source_locator": "Scheme 1",
                        "confidence": "low",
                        "tool_checks": ["visual extraction performed in this run"],
                    },
                    "stereochemistry_status": "unspecified_or_partial",
                    "not_exact_literature_segment": True,
                    "allowed_use": "exploratory_template_and_guided_hint_only",
                    "risk_flags": ["stereochemistry_unspecified"],
                }
            ],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="steroid_target",
            target_smiles=BUFOTALIN_ACHIRAL_SMILES,
            source_ref="doi:10.0000/analog",
            source_title="Analog steroid source",
            image_paths=[],
        )
        quality = _candidate_quality(chain, expected_labels=["drawn alcohol"])

        self.assertTrue(quality["accepted"])
        self.assertTrue(quality["exploratory_accepted"])
        self.assertFalse(quality["exact_ready"])
        self.assertEqual(quality["acceptance_level"], "exploratory_connectivity_candidate")
        self.assertTrue(chain["steps"][0]["not_exact_literature_segment"])
        self.assertEqual(chain["steps"][0]["allowed_use"], "exploratory_template_and_guided_hint_only")

        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=run_preflight(target), max_rounds=3)
        with tempfile.TemporaryDirectory() as tmp:
            board = update_blackboard_from_action(
                board,
                action={
                    "schema_version": "agent_action.v1",
                    "action_id": "visual:approx",
                    "action_type": "extract_visual_literature_chain",
                    "rationale": "extract approximate visual candidate",
                    "expected_artifact": "visual chain",
                    "success_condition": "exploratory candidate",
                    "payload": {},
                },
                action_result={
                    "accepted": True,
                    "result": {
                        "schema_version": "visual_literature_chain_extraction_result.v1",
                        "accepted": True,
                        "acceptance_level": "exploratory_connectivity_candidate",
                        "exact_ready": False,
                        "exploratory_accepted": True,
                        "source_ref": "doi:10.0000/analog",
                        "candidate_chain": chain,
                        "candidate_quality": quality,
                        "candidate_step_count": 1,
                        "reasons": ["visual_literature_chain_structure_gaps"],
                    },
                },
                round_index=1,
                run_dir=tmp,
            )

        summary = board["literature_evidence"]["visual_chains"][0]
        self.assertTrue(summary["exploratory_accepted"])
        self.assertFalse(summary["exact_ready"])
        self.assertEqual(summary["steps"][0]["allowed_use"], "exploratory_template_and_guided_hint_only")
        self.assertTrue(summary["steps"][0]["not_exact_literature_segment"])

    def test_guided_chemenzy_large_atom_jump_overrides_backend_solved_when_no_route_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "steroid_target", "target_smiles": BUFOTALIN_ACHIRAL_SMILES},
                preflight={"case_id": "steroid_target"},
                budget=HarnessBudget(max_guided_chemenzy_runs=1),
            )
            payload = {
                "chem_enzy_search_policy": {
                    "schema_version": "chem_enzy_search_policy.v1",
                    "policy_id": "test_policy",
                    "operator_id": "test",
                    "case_id": "steroid_target",
                    "preferred_subgoal": {},
                    "source_budget": {},
                    "budget": {"max_depth": 3, "max_iterations": 3, "expansion_topk": 3},
                    "mode": "guided",
                    "compiler_metadata": {"requires_verifier": True},
                }
            }
            backend_result = {"ok": True, "routes": [{"route_id": "r1"}], "search_status": {"solved": True}}
            verifier = {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "accepted_route_count": 0,
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump", "details": {"jumps": [{"delta_heavy_atoms": 24}]}}],
            }

            with patch(
                "cascade_planner.harness.tools._execute_chemenzy_request",
                return_value=backend_result,
            ), patch(
                "cascade_planner.harness.tools.verify_chemenzy_raw_routes",
                return_value=verifier,
            ):
                output = run_guided_chemenzy_rerun(state, payload)

        result = output["result"]
        self.assertTrue(output["accepted"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["solved"])
        self.assertEqual(result["route_status"], "fake_closed_rejected")
        self.assertIn("guided_route_verifier_rejected_large_atom_jump", result["reasons"])
        self.assertTrue(result["route_failure_feedback"]["accepted"])

    def test_guided_chemenzy_preserves_solved_verifier_with_rejected_sibling_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "atorvastatin", "target_smiles": ATORVASTATIN_FREE_ACID_SMILES},
                preflight={"case_id": "atorvastatin"},
                budget=HarnessBudget(max_guided_chemenzy_runs=1),
            )
            payload = {
                "chem_enzy_search_policy": {
                    "schema_version": "chem_enzy_search_policy.v1",
                    "policy_id": "test_policy",
                    "operator_id": "test",
                    "case_id": "atorvastatin",
                    "preferred_subgoal": {},
                    "source_budget": {},
                    "budget": {"max_depth": 3, "max_iterations": 3, "expansion_topk": 3},
                    "mode": "guided",
                    "compiler_metadata": {"requires_verifier": True},
                }
            }
            backend_result = {"ok": True, "routes": [{"route_id": "r1"}, {"route_id": "r2"}], "search_status": {"solved": True}}
            verifier = {
                "schema_version": "harness_route_verifier_report.v1",
                "accepted": True,
                "route_status": "solved",
                "target_match": True,
                "target_equivalence_audit": {"target_match": True},
                "route_count": 3,
                "accepted_route_count": 2,
                "rejected_route_count": 1,
                "best_route_rank": 0,
                "best_route_step_count": 1,
                "reasons": [],
                "warnings": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump", "details": {"jumps": [{"delta_heavy_atoms": 24}]}}],
            }

            with patch(
                "cascade_planner.harness.tools._execute_chemenzy_request",
                return_value=backend_result,
            ), patch(
                "cascade_planner.harness.tools.verify_chemenzy_raw_routes",
                return_value=verifier,
            ):
                output = run_guided_chemenzy_rerun(state, payload)

        result = output["result"]
        self.assertTrue(output["accepted"])
        self.assertTrue(result["accepted"])
        self.assertTrue(result["solved"])
        self.assertEqual(result["route_status"], "solved")
        self.assertNotIn("route_proof_blockers", result)

    def test_guided_chemenzy_timeout_is_blackboard_feedback_not_tool_failure(self):
        target = TargetInput(target_name="timeout_case", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input=target.to_dict(),
                preflight={"case_id": "timeout_case"},
                budget=HarnessBudget(max_guided_chemenzy_runs=1, guided_chemenzy_timeout_s=5),
            )
            payload = {
                "chem_enzy_search_policy": {
                    "schema_version": "chem_enzy_search_policy.v1",
                    "policy_id": "timeout_policy",
                    "operator_id": "test",
                    "case_id": "timeout_case",
                    "terminal_blacklist": [],
                    "active_bridge_tasks": [],
                    "accepted_exact_row_ids": [],
                    "selected_analogical_hypothesis_ids": [],
                    "selected_analogical_template_ids": [],
                    "forbidden_template_ids": [],
                    "preferred_subgoal": {},
                    "source_budget": {"require_target_core_retention": True, "max_unexplained_heavy_atom_jump": 12},
                    "budget": {"max_depth": 3, "max_iterations": 3, "expansion_topk": 3},
                    "mode": "guided",
                    "compiler_metadata": {"requires_verifier": True, "no_solved_claim": True},
                }
            }
            backend_result = {
                "schema_version": "chemenzy_run_result.v1",
                "accepted": False,
                "status": "timeout",
                "exit_code": -15,
                "reasons": ["chem_enzy_timeout"],
                "timeout_s": 5,
            }

            with patch("cascade_planner.harness.tools._execute_chemenzy_request", return_value=backend_result):
                output = run_guided_chemenzy_rerun(state, payload)

            self.assertTrue(output["accepted"])
            result = output["result"]
            self.assertFalse(result["accepted"])
            self.assertEqual(result["chemenzy_runtime_diagnostic"]["reasons"], ["chem_enzy_timeout"])

            board = initialize_agent_blackboard(
                target_input=target.to_dict(),
                preflight=run_preflight(target),
                max_rounds=2,
            )
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r1:guided", "action_type": "run_guided_chemenzy"},
                action_result=result,
                round_index=1,
                run_dir=run_dir,
            )

        self.assertTrue(board["action_history"][-1]["useful_artifact"])
        self.assertIn("plugin_runtime_diagnostics", board["action_history"][-1]["changed_blackboard_fields"])
        self.assertIn("route_failures", board["action_history"][-1]["changed_blackboard_fields"])
        self.assertTrue(any(row.get("reason") == "chem_enzy_timeout" for row in board["route_failures"]))

    def test_guided_chemenzy_detects_onmt_runtime_error_from_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = Path(tmp) / "guided_chemenzy_raw_result.json.stdout.log"
            stdout.write_text(
                "'Vocab' object has no attribute 'stoi'\n"
                "onmt_models.bionav_native_one_step\n",
                encoding="utf-8",
            )
            diagnostic = _guided_chemenzy_runtime_diagnostic(
                {
                    "ok": False,
                    "exit_code": 0,
                    "stdout_path": str(stdout),
                    "search_status": {"status": "failed"},
                }
            )

        self.assertEqual(diagnostic["status"], "one_step_model_runtime_error")
        self.assertIn("onmt_one_step_model_runtime_error", diagnostic["reasons"])
        self.assertIn("torchtext_vocab_legacy_api_mismatch", diagnostic["reasons"])

    def test_guided_chemenzy_no_route_is_blackboard_failure_evidence(self):
        target = TargetInput(target_name="no_route_case", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            board = initialize_agent_blackboard(
                target_input=target.to_dict(),
                preflight=run_preflight(target),
                max_rounds=2,
            )
            board = update_blackboard_from_action(
                board,
                action={"action_id": "r1:guided", "action_type": "run_guided_chemenzy"},
                action_result={
                    "accepted": True,
                    "reasons": [],
                    "result": {
                        "schema_version": "guided_chemenzy_rerun_result.v1",
                        "accepted": True,
                        "solved": False,
                        "route_status": "unresolved",
                        "raw_route_verifier": {},
                        "result": {
                            "ok": False,
                            "n_results": 0,
                            "failure_diagnosis": ["no_route_found"],
                            "search_status": {
                                "status": "failed",
                                "solved": False,
                                "message": "ChemEnzy native core search returned no route",
                            },
                        },
                    },
                },
                round_index=1,
                run_dir=run_dir,
            )

        self.assertTrue(board["action_history"][-1]["useful_artifact"])
        self.assertIn("route_failures", board["action_history"][-1]["changed_blackboard_fields"])
        self.assertTrue(any(row.get("reason") == "no_route_found" for row in board["route_failures"]))
        self.assertIn("build_failure_critic_report", board["current_belief"]["next_action_bias"])

    def test_visual_direct_api_failure_is_auditable_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            image = run_dir / "page.jpg"
            image.write_bytes(b"not-really-an-image")
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "visual_case", "target_smiles": "CCO"},
                preflight={"case_id": "visual_case", "target_profile": {"heavy_atoms": 3}},
            )
            visual_result = {
                "schema_version": "visual_literature_chain_extraction_result.v1",
                "accepted": False,
                "status": "error",
                "reasons": ["visual_direct_api_failed"],
                "attempts": [{"status": "error", "reasons": ["visual_direct_api_failed"]}],
                "image_paths": [str(image)],
                "no_solved_claim": True,
            }

            with patch(
                "cascade_planner.harness.tools._visual_chain_image_paths",
                return_value=[image],
            ), patch(
                "cascade_planner.harness.tools.run_visual_literature_chain_agent",
                return_value=visual_result,
            ):
                record = execute_local_tool("extract_visual_literature_chain", {"source_ref": "doi:test"}, state)

        self.assertEqual(record.status, "accepted")
        self.assertFalse(record.output["result"]["accepted"])
        self.assertIn("visual_direct_api_failed", record.reasons)

    def test_visual_partial_candidate_is_salvaged_when_standard_result_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            image = run_dir / "page.jpg"
            image.write_bytes(b"not-really-an-image")
            out = run_dir / "visual_out"
            out.mkdir()
            candidate = {
                "schema_version": "visual_structure_candidate_chain.v1",
                "steps": [
                    {
                        "step_id": "scheme_step_1",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC=O"],
                        "source_ref": "doi:test",
                        "evidence_refs": ["current_image:page.jpg"],
                    }
                ],
            }
            (out / "visual_structure_candidate_chain.json").write_text(
                json.dumps(candidate, ensure_ascii=False),
                encoding="utf-8",
            )
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "visual_case", "target_smiles": "CCO"},
                preflight={"case_id": "visual_case", "target_profile": {"heavy_atoms": 3}},
            )

            with patch(
                "cascade_planner.harness.tools._visual_chain_image_paths",
                return_value=[image],
            ), patch(
                "cascade_planner.harness.tools.run_visual_literature_chain_agent",
                side_effect=FileNotFoundError(str(out / "visual_literature_chain_extraction_result.json")),
            ):
                record = execute_local_tool(
                    "extract_visual_literature_chain",
                    {"source_ref": "doi:test", "source_title": "Test source", "output_dir": "visual_out"},
                    state,
                )
            self.assertTrue((out / "visual_literature_chain_extraction_result.json").exists())

        result = record.output["result"]
        self.assertEqual(record.status, "accepted")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["status"], "partial_candidate_salvaged")
        self.assertIn("visual_result_file_missing_salvaged_candidate", record.reasons)
        self.assertEqual(result["candidate_step_count"], 1)

    def test_visual_partial_candidate_is_salvaged_when_tool_cleanup_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            image = run_dir / "page.jpg"
            image.write_bytes(b"not-really-an-image")
            out = run_dir / "visual_out"
            out.mkdir()
            candidate = {
                "schema_version": "visual_structure_candidate_chain.v1",
                "steps": [
                    {
                        "step_id": "scheme_step_1",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC=O"],
                        "source_ref": "doi:test",
                        "evidence_refs": ["current_image:page.jpg"],
                    }
                ],
            }
            (out / "visual_structure_candidate_chain.json").write_text(
                json.dumps(candidate, ensure_ascii=False),
                encoding="utf-8",
            )
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "visual_case", "target_smiles": "CCO"},
                preflight={"case_id": "visual_case", "target_profile": {"heavy_atoms": 3}},
            )

            with patch(
                "cascade_planner.harness.tools._visual_chain_image_paths",
                return_value=[image],
            ), patch(
                "cascade_planner.harness.tools.run_visual_literature_chain_agent",
                side_effect=OSError("[WinError 145] directory is not empty"),
            ):
                record = execute_local_tool(
                    "extract_visual_literature_chain",
                    {"source_ref": "doi:test", "source_title": "Test source", "output_dir": "visual_out"},
                    state,
                )

        result = record.output["result"]
        self.assertEqual(record.status, "accepted")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["status"], "partial_candidate_salvaged")
        self.assertIn("visual_tool_error_salvaged_candidate", record.reasons)
        self.assertEqual(result["candidate_step_count"], 1)

    def test_visual_action_output_dir_is_short_enough_for_windows_paths(self):
        output_dir = _visual_action_output_dir(
            {
                "action_id": "r2_extract_danishefsky_visual_chain_after_bootstrap",
                "action_type": "extract_visual_literature_chain",
                "payload": {"source_ref": "doi:10.1021/ja952692a"},
            }
        )

        self.assertLessEqual(len(output_dir), 90)
        self.assertTrue(output_dir.startswith("visual_lit_chain_"))
        self.assertNotIn("visual_literature_chain_extraction_r2_extract", output_dir)

    def test_visual_partial_candidate_is_salvaged_when_packaging_missing_result_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            image = run_dir / "page.jpg"
            image.write_bytes(b"not-really-an-image")
            out = run_dir / "visual_out"
            out.mkdir()
            candidate = {
                "schema_version": "visual_structure_candidate_chain.v1",
                "steps": [
                    {
                        "step_id": "scheme_step_1",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC=O"],
                        "source_ref": "doi:test",
                        "evidence_refs": ["current_image:page.jpg"],
                    }
                ],
            }
            candidate_path = out / "visual_structure_candidate_chain.json"
            candidate_path.write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
            visual_result = {
                "schema_version": "visual_literature_chain_extraction_result.v1",
                "accepted": True,
                "status": "completed",
                "candidate_chain_path": str(candidate_path),
                "candidate_chain": candidate,
                "candidate_step_count": 1,
                "reasons": [],
                "no_solved_claim": True,
            }
            state = ToolExecutionState(
                run_dir=run_dir,
                target_input={"target_name": "visual_case", "target_smiles": "CCO"},
                preflight={"case_id": "visual_case", "target_profile": {"heavy_atoms": 3}},
            )

            with patch(
                "cascade_planner.harness.tools._visual_chain_image_paths",
                return_value=[image],
            ), patch(
                "cascade_planner.harness.tools.run_visual_literature_chain_agent",
                return_value=visual_result,
            ), patch(
                "cascade_planner.harness.tools._attach_process_evidence_rows_to_visual_result",
                side_effect=FileNotFoundError(str(out / "visual_literature_chain_extraction_result.json")),
            ):
                record = execute_local_tool(
                    "extract_visual_literature_chain",
                    {"source_ref": "doi:test", "source_title": "Test source", "output_dir": "visual_out"},
                    state,
                )

        result = record.output["result"]
        self.assertEqual(record.status, "accepted")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["status"], "partial_candidate_salvaged")
        self.assertIn("visual_result_file_missing_salvaged_candidate", record.reasons)
        self.assertEqual(result["candidate_step_count"], 1)

    def test_visual_timeout_zero_inherits_open_research_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input={"target_name": "visual_timeout", "target_smiles": "CCO"},
                preflight={"case_id": "visual_timeout"},
                budget=HarnessBudget(open_research_timeout_s=900.0),
            )

            visual_timeout = _visual_literature_timeout_s(state, {"timeout_s": 0})
            structure_timeout = _structure_resolution_timeout_s(state, {"timeout_s": 0})

        self.assertEqual(visual_timeout, 900.0)
        self.assertEqual(structure_timeout, 900.0)

    def test_planner_does_not_visual_extract_placeholder_or_metadata_only_sources(self):
        target = TargetInput(target_name="metadata_case", target_smiles="CCO")
        preflight = run_preflight(target)
        for candidate in [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "query:metadata_case:bridge",
                "title": "placeholder bridge query",
                "placeholder_only": True,
                "access_status": "placeholder_only",
                "local_pdf": "",
                "url": "",
                "doi": "",
            },
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1000/metadata",
                "title": "metadata-only article",
                "doi": "10.1000/metadata",
                "url": "https://doi.org/10.1000/metadata",
                "local_pdf": "",
                "access_status": "metadata_only",
            },
        ]:
            board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
            board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
            board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
            board["bridge_tasks"] = [
                {
                    "schema_version": "agent_bridge_task.v1",
                    "task_id": "bridge:core",
                    "task_type": "target_proximal_bridge",
                    "target_handle": "core",
                }
            ]
            board["literature_evidence"]["source_candidates"] = [candidate]
            board["literature_evidence"]["source_refs"] = [candidate["source_ref"]]

            batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
            action_types = [row["action_type"] for row in batch["actions"]]

            self.assertNotIn("extract_visual_literature_chain", action_types)

    def test_planner_repairs_visual_gaps_before_compile(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "24", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:source",
                pdf_path="/tmp/source.pdf",
                evidence_id="pdf",
            )
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": True,
                "candidate_step_count": 1,
                "condition_gap_labels": ["33"],
                "steps": [
                    {
                        "product_label": "33",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC=O"],
                        "not_exact_literature_segment": True,
                    }
                ],
                "extraction_gaps": [{"labels": ["24", "25", "11"], "reason": "small structures"}],
            }
        ]
        board["budget_state"]["visual_calls"] = 1

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_visual_literature_chain")
        self.assertTrue(first["payload"]["focused_gap_repair"])
        self.assertIn("24", first["payload"]["expected_labels"])
        self.assertIn("11", first["payload"]["expected_labels"])

    def test_planner_compiles_partial_visual_steps_before_more_gap_repair(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "24", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:source",
                pdf_path="/tmp/source.pdf",
                evidence_id="pdf",
            )
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": False,
                "candidate_step_count": 3,
                "extraction_gaps": [{"labels": ["24", "25", "11"], "gap_type": "structure_gap"}],
            }
        ]
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 3,
                "action_type": "extract_visual_literature_chain",
                "useful_artifact": True,
                "stale": False,
            }
        )
        board["budget_state"]["visual_calls"] = 1

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "compile_exact_literature_rows")

    def test_planner_changes_to_templates_after_one_unresolved_visual_repair(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core", "target_handle": "steroid_core"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "title": "Analog steroid route",
                "source_type": "reaction_precedent",
                "relevance_rationale": "analog steroid family precedent",
                "expected_scheme_or_compound_labels": ["target", "27", "26", "23"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:source",
                pdf_path="/tmp/source.pdf",
                evidence_id="pdf",
            )
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": True,
                "candidate_step_count": 6,
                "extraction_gaps": [{"label": "steroid core 26", "gap_type": "structure_gap"}],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual2",
                "accepted": True,
                "candidate_step_count": 3,
                "extraction_gaps": [{"label": "steroid core 26", "gap_type": "structure_gap"}],
            },
        ]
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:sugar_branch"}]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "schema_version": "agent_exact_chain_audit_summary.v1",
                "audit_id": "audit1",
                "accepted": False,
                "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
                "one_step_row_count": 6,
            }
        ]
        board["action_history"] = [
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "useful_artifact": True, "stale": False},
            {
                "round_index": 5,
                "action_type": "extract_visual_literature_chain",
                "useful_artifact": True,
                "stale": False,
                "action_signature": json.dumps({"payload": {"focused_gap_repair": True}}),
            },
            {"round_index": 6, "action_type": "compile_exact_literature_rows", "useful_artifact": False, "stale": True},
        ]
        board["budget_state"]["visual_calls"] = 3

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertNotIn("extract_visual_literature_chain", action_types)
        self.assertIn("run_guided_chemenzy", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_changes_to_templates_after_visual_tool_failures_without_steps(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=10)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core", "target_handle": "steroid_core"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "src:analog_source",
                "local_pdf": "/tmp/source.pdf",
                "source_type": "literature_metadata+local_pdf",
                "relevance_rationale": "analog steroid family precedent",
                "expected_scheme_or_compound_labels": ["target analogue", "core"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {
                "schema_version": "agent_pdf_structure_evidence_summary.v1",
                "source_ref": "src:analog_source",
                "accepted": True,
            }
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "src:analog_source",
                "accepted": False,
                "candidate_step_count": 0,
                "missing_expected_labels": ["target analogue", "core"],
                "reasons": ["codex_visual_chain_nonzero_exit", "visual_literature_chain_has_no_steps"],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual2",
                "source_ref": "src:analog_source",
                "accepted": False,
                "candidate_step_count": 0,
                "missing_expected_labels": ["target analogue", "core"],
                "reasons": ["codex_visual_chain_nonzero_exit", "visual_literature_chain_has_no_steps"],
            },
        ]
        board["action_history"] = [
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "useful_artifact": True, "stale": False},
            {
                "round_index": 4,
                "action_type": "extract_visual_literature_chain",
                "action_signature": json.dumps({"payload": {"focused_gap_repair": True}}),
                "useful_artifact": True,
                "stale": False,
            },
        ]
        board["budget_state"]["visual_calls"] = 3

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("extract_analogical_reaction_templates", action_types)
        self.assertNotIn("compile_exact_literature_rows", action_types)
        self.assertNotIn("stop_unresolved", action_types)

    def test_planner_runs_guided_after_template_extraction_failure(self):
        target = TargetInput(target_name="steroid target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core", "target_handle": "polycyclic_cage_core"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["literature_evidence"]["source_candidates"] = [{"source_ref": "doi:source", "doi": "10.1000/source"}]
        board["literature_evidence"]["source_refs"] = ["doi:source"]
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:anchor"}]
        board["literature_evidence"]["exact_chain_audits"] = [
            {"audit_id": "audit1", "accepted": False, "reasons": ["no_chain_unrolled"]}
        ]
        board["action_history"] = [
            {
                "round_index": 7,
                "action_type": "extract_analogical_reaction_templates",
                "status": "rejected",
                "useful_artifact": False,
                "stale": True,
            }
        ]

        batch = plan_action_batch(board, round_index=8, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("extract_analogical_reaction_templates", action_types)
        self.assertIn("run_guided_chemenzy", action_types)
        guided_action = next(row for row in batch["actions"] if row["action_type"] == "run_guided_chemenzy")
        self.assertIn("search_policy", guided_action["payload"])
        self.assertTrue(
            guided_action["payload"]["search_policy"]["compiler_metadata"]["requires_verifier"]
        )
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_compiles_complete_visual_chain_with_stereo_warnings(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:source",
                "local_pdf": "/tmp/source.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "32", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:source",
                pdf_path="/tmp/source.pdf",
                evidence_id="pdf",
            )
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "accepted": False,
                "candidate_step_count": 3,
                "missing_expected_labels": [],
                "extraction_gaps": [
                    {
                        "labels": ["30", "11"],
                        "gap_type": "stereochemical_ambiguity",
                        "detail": "valid connectivity, stereo warning only",
                    }
                ],
            }
        ]
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 3,
                "action_type": "extract_visual_literature_chain",
                "useful_artifact": True,
                "stale": False,
            }
        )

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "compile_exact_literature_rows")

    def test_planner_prefers_guided_before_exact_literature_terminal_child(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:24_from_11"}]
        board["literature_evidence"]["terminal_candidates"] = [
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "name": "Androstenedione",
                "smiles": "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
                "canonical_smiles": "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",
            }
        ]
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "literature_terminal_child:androstenedione",
                "task_type": "upstream_terminal_synthesis",
            }
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "run_guided_chemenzy")
        self.assertIn("expand_child_target", action_types)
        child_action = next(row for row in batch["actions"] if row["action_type"] == "expand_child_target")
        self.assertEqual(child_action["payload"]["subgoal_targets"][0]["name"], "Androstenedione")
        child_target = child_action["payload"]["subgoal_targets"][0]
        self.assertTrue(child_target["target_equivalence_audit_required"])
        self.assertTrue(child_target["exact_target_override"])
        self.assertTrue(child_target["no_solved_claim"])
        self.assertTrue(child_target["child_route_cannot_promote_parent"])
        self.assertTrue(child_target["chem_enzy_search_policy"]["compiler_metadata"]["requires_verifier"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_stops_repeating_same_failed_child_terminal(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_ACHIRAL_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        terminal_smiles = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:terminal"}]
        board["literature_evidence"]["terminal_candidates"] = [
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "name": "source detail literature terminal",
                "smiles": terminal_smiles,
                "canonical_smiles": terminal_smiles,
            }
        ]
        board["bridge_tasks"] = [
            {
                "schema_version": "agent_bridge_task.v1",
                "task_id": "literature_terminal_child:terminal",
                "task_type": "upstream_terminal_synthesis",
            }
        ]
        for attempt in (1, 2):
            board["action_history"].append(
                {
                    "schema_version": "agent_action_history_record.v1",
                    "round_index": attempt,
                    "action_type": "expand_child_target",
                    "status": "accepted",
                    "useful_artifact": True,
                    "stale": False,
                    "reasons": ["no_route_expansion_subgoal_verified_solved"],
                    "action_signature": json.dumps(
                        {
                            "action_type": "expand_child_target",
                            "payload": {
                                "expansion_attempt": attempt,
                                "subgoal_targets": [{"smiles": terminal_smiles}],
                            },
                        },
                        sort_keys=True,
                    ),
                }
            )

        batch = plan_action_batch(board, round_index=3, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("expand_child_target", action_types)

    def test_planner_does_not_stitch_without_parent_and_child_or_exact_refs(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=6)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["route_failures"].append({"reason": "chem_enzy_timeout", "route_status": "unresolved"})
        board["literature_evidence"]["visual_chains"] = [{"chain_id": "visual:1"}]
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 1,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
            }
        )

        batch = plan_action_batch(board, round_index=2, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("stitch_parent_route", action_types)

    def test_planner_stitches_parent_after_exact_terminal_child_solved(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:24_from_11"}]
        terminal_smiles = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
        board["literature_evidence"]["terminal_candidates"] = [
            {
                "schema_version": "agent_literature_terminal_candidate.v1",
                "name": "Androstenedione",
                "smiles": terminal_smiles,
                "canonical_smiles": terminal_smiles,
                "strict_source_proof_eligible": True,
            }
        ]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "audit_id": "strict-chain",
                "artifact_ref": "/tmp/strict_chain.json",
                "strict_source_proof_eligible": True,
                "terminal_frontier": [terminal_smiles],
            }
        ]
        board["route_expansion_subgoals"] = [
            {"canonical_smiles": terminal_smiles, "accepted": True}
        ]
        board["current_belief"]["child_route_solved"] = True
        board["artifact_refs"]["guided_chemenzy"] = "/tmp/parent_route.json"
        board["artifact_refs"]["route_expansion_subgoal_search"] = "/tmp/child_route.json"
        board["artifact_refs"]["compile_source_detail_chain_route"] = "/tmp/strict_chain.json"
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 5,
                "action_type": "expand_child_target",
                "useful_artifact": True,
                "stale": False,
            }
        )

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertEqual(action_types[0], "stitch_parent_route")
        self.assertNotIn("expand_child_target", action_types)
        self.assertNotIn("run_guided_chemenzy", action_types)
        stitch_payload = batch["actions"][0]["payload"]
        self.assertEqual(stitch_payload["proof_policy"]["final_verdict_authority"], "deterministic_parent_route_proof")
        self.assertIn("exact_literature_row_ids", stitch_payload["proof_binding"])
        self.assertTrue(stitch_payload["proof_policy"]["analogy_is_not_proof"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_never_uses_advisory_visual_chain_as_parent_stitch_proof(self):
        target = TargetInput(target_name="paclitaxel analogue", target_smiles="CC1CCC(O)C=C2CCCC(=O)C12")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["visual_chains"] = [
            {
                "accepted": True,
                "chain_id": "visual:danishefsky",
                "source_ref": "doi:10.1021/ja952692a",
                "step_count": 2,
                "steps": [
                    {
                        "step_id": "compound8_from_5",
                        "product_smiles": "CC1CCC(O)C=C2CCCC(=O)C12",
                        "main_reactant_smiles": "CC1CCC(=O)C=C2CCCC(=O)C12",
                    },
                    {
                        "step_id": "compound5_from_7",
                        "product_smiles": "CC1CCC(=O)C=C2CCCC(=O)C12",
                        "main_reactant_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
                    },
                ],
            }
        ]
        board["current_belief"]["child_route_solved"] = True
        board["artifact_refs"]["route_expansion_subgoal_search"] = "/tmp/child_route.json"
        board["action_history"].append(
            {
                "schema_version": "agent_action_history_record.v1",
                "round_index": 5,
                "action_type": "expand_child_target",
                "useful_artifact": True,
                "stale": False,
            }
        )

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("stitch_parent_route", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_parent_stitch_waits_until_every_strict_source_frontier_is_closed(self):
        target = TargetInput(target_name="generic target", target_smiles="CCOO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:s1"}]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "audit_id": "strict-multifrontier",
                "artifact_ref": "/tmp/strict_multifrontier.json",
                "strict_source_proof_eligible": True,
                "terminal_frontier": ["CCO", "O"],
            }
        ]
        board["artifact_refs"].update(
            {
                "compile_source_detail_chain_route": "/tmp/strict_multifrontier.json",
                "route_expansion_subgoal_search": "/tmp/children.json",
            }
        )
        board["route_expansion_subgoals"] = [{"canonical_smiles": "CCO", "accepted": True}]

        incomplete = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        self.assertNotIn("stitch_parent_route", [row["action_type"] for row in incomplete["actions"]])

        board["route_expansion_subgoals"].append({"canonical_smiles": "O", "accepted": True})
        complete = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        self.assertEqual(complete["actions"][0]["action_type"], "stitch_parent_route")
        binding = complete["actions"][0]["payload"]["proof_binding"]
        self.assertTrue(binding["all_terminal_frontiers_closed"])
        self.assertEqual(binding["missing_terminal_frontier"], [])

    def test_source_detail_reaction_promotes_every_terminal_frontier_for_expansion(self):
        target = TargetInput(target_name="generic target", target_smiles="CCOO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:multifrontier",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile a source-detail reaction graph",
                "expected_artifact": "source-detail chain",
                "success_condition": "frontiers are materialized",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "accepted": True,
                    "chain_audit": {
                        "schema_version": "source_detail_route_chain_audit.v1",
                        "accepted": True,
                        "case_id": "generic",
                        "target_smiles": "CCOO",
                        "source_ref": "doi:10.0000/advisory",
                        "step_count": 1,
                        "terminal_reached": True,
                        "chain": [
                            {
                                "step_id": "s1",
                                "product_smiles": "CCOO",
                                "reactant_smiles": ["CCO", "O"],
                                "source_ref": "doi:10.0000/advisory",
                            }
                        ],
                    },
                },
            },
            round_index=2,
            run_dir="/tmp",
        )

        terminals = board["literature_evidence"]["terminal_candidates"]
        self.assertEqual({row["canonical_smiles"] for row in terminals}, {"CCO", "O"})
        self.assertEqual(len({row["terminal_id"] for row in terminals}), 2)
        self.assertTrue(all(row["requires_all_frontiers_closed"] for row in terminals))
        self.assertTrue(all(row["strict_source_proof_eligible"] is False for row in terminals))
        frontier_tasks = [row for row in board["bridge_tasks"] if row.get("task_type") == "upstream_terminal_synthesis"]
        self.assertEqual(len(frontier_tasks), 2)

    def test_planner_does_not_repeat_stitch_without_new_bridge_signal(self):
        target = TargetInput(target_name="paclitaxel analogue", target_smiles="CC1CCC(O)C=C2CCCC(=O)C12")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["visual_chains"] = [
            {
                "accepted": True,
                "chain_id": "visual:danishefsky",
                "source_ref": "doi:10.1021/ja952692a",
                "step_count": 2,
                "steps": [
                    {
                        "step_id": "compound8_from_5",
                        "product_smiles": "CC1CCC(O)C=C2CCCC(=O)C12",
                        "main_reactant_smiles": "CC1CCC(=O)C=C2CCCC(=O)C12",
                    },
                    {
                        "step_id": "compound5_from_7",
                        "product_smiles": "CC1CCC(=O)C=C2CCCC(=O)C12",
                        "main_reactant_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
                    },
                ],
            }
        ]
        board["current_belief"]["child_route_solved"] = True
        board["artifact_refs"]["route_expansion_subgoal_search"] = "/tmp/child_route.json"
        board["action_history"].extend(
            [
                {
                    "schema_version": "agent_action_history_record.v1",
                    "round_index": 5,
                    "action_type": "expand_child_target",
                    "useful_artifact": True,
                    "stale": False,
                },
                {
                    "schema_version": "agent_action_history_record.v1",
                    "round_index": 6,
                    "action_type": "stitch_parent_route",
                    "status": "rejected",
                    "useful_artifact": False,
                    "stale": False,
                    "reasons": ["child_target_route_not_connected_to_parent_bridge"],
                },
            ]
        )

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("stitch_parent_route", action_types)

    def test_blackboard_records_stereo_ambiguity_as_visual_warning(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:1",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract chain",
                "expected_artifact": "visual",
                "success_condition": "chain",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": True,
                    "candidate_step_count": 15,
                    "parsed_output": {
                        "steps": [{"product_label": "bufotalin"}],
                        "extraction_gaps": [
                            {
                                "gap_type": "stereochemical_ambiguity",
                                "labels": ["25", "23"],
                                "reason": "major stereoisomer encoded",
                            }
                        ],
                    },
                    "candidate_quality": {
                        "missing_expected_labels": [],
                        "condition_gap_labels": [],
                    },
                    "reasons": [],
                },
                "reasons": [],
            },
            round_index=3,
            run_dir="/tmp",
        )

        visual = board["literature_evidence"]["visual_chains"][0]
        self.assertEqual(visual["gap_labels"], [])
        self.assertEqual(visual["warning_gap_labels"], ["25", "23"])

    def test_blackboard_structure_gaps_create_resolution_tasks(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:structure-gap",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract chain",
                "expected_artifact": "visual",
                "success_condition": "chain",
                "payload": {"source_ref": "doi:source"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": False,
                    "source_ref": "doi:source",
                    "parsed_output": {
                        "source_ref": "doi:source",
                        "extraction_gaps": [
                            {
                                "label": "compound 15",
                                "gap_type": "structure_gap",
                                "reason": "visible but not safely convertible",
                            }
                        ],
                    },
                    "reasons": ["visual_literature_chain_extraction_gaps"],
                },
                "reasons": ["visual_literature_chain_extraction_gaps"],
            },
            round_index=3,
            run_dir="/tmp",
        )

        tasks = board["literature_evidence"]["structure_resolution_tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["label"], "compound 15")
        self.assertEqual(tasks[0]["source_ref"], "doi:source")

    def test_blackboard_records_resolved_literature_structure_and_marks_task_resolved(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "resolve:15",
                "action_type": "resolve_literature_structure_task",
                "rationale": "resolve one label",
                "expected_artifact": "structure resolution",
                "success_condition": "resolved or unresolved",
                "payload": {"task_id": "resolve_structure:doi_source_compound_15", "label": "compound 15", "source_ref": "doi:source"},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "literature_structure_resolution_result.v1",
                    "accepted": True,
                    "status": "resolved",
                    "task_id": "resolve_structure:doi_source_compound_15",
                    "label": "compound 15",
                    "source_ref": "doi:source",
                    "resolved_structures": [
                        {
                            "schema_version": "literature_resolved_structure_candidate.v1",
                            "structure_id": "resolve_structure_doi_source_compound_15:1",
                            "task_id": "resolve_structure:doi_source_compound_15",
                            "label": "compound 15",
                            "smiles": "CCO",
                            "source_ref": "doi:source",
                            "source_locator": "Scheme 1",
                            "accepted": True,
                            "no_solved_claim": True,
                        }
                    ],
                    "unresolved_tasks": [],
                    "reasons": [],
                    "no_solved_claim": True,
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        evidence = board["literature_evidence"]
        self.assertEqual(len(evidence["resolved_structures"]), 1)
        self.assertEqual(evidence["structure_resolution_tasks"][0]["status"], "resolved")
        self.assertEqual(evidence["structure_resolution_tasks"][0]["last_resolution_status"], "resolved")
        self.assertTrue(board["action_history"][-1]["useful_artifact"])

    def test_explicit_source_resolved_same_scaffold_structure_promotes_semisynthesis_anchor(self):
        target = TargetInput(target_name="generic cyclic target", target_smiles="CC(=O)OC1CCC(CC1)OC(C)=O")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_advanced_intermediate",
                "task_type": "resolve_literature_structure",
                "label": "advanced cyclic diol",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "resolve:advanced_intermediate",
                "action_type": "resolve_literature_structure_task",
                "rationale": "resolve one label",
                "expected_artifact": "structure resolution",
                "success_condition": "resolved or unresolved",
                "payload": {
                    "task_id": "resolve_structure:doi_source_advanced_intermediate",
                    "label": "advanced cyclic diol",
                    "source_ref": "doi:source",
                },
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "literature_structure_resolution_result.v1",
                    "accepted": True,
                    "status": "resolved",
                    "task_id": "resolve_structure:doi_source_advanced_intermediate",
                    "label": "advanced cyclic diol",
                    "source_ref": "doi:source",
                    "resolved_structures": [
                        {
                            "schema_version": "literature_resolved_structure_candidate.v1",
                            "structure_id": "resolve_structure_doi_source_advanced_intermediate:1",
                            "task_id": "resolve_structure:doi_source_advanced_intermediate",
                            "label": "advanced cyclic diol",
                            "smiles": "OC1CCC(CC1)O",
                            "structure_role": "advanced_intermediate",
                            "source_ref": "doi:source",
                            "source_locator": "scheme identifies this as an advanced intermediate",
                            "accepted": True,
                            "rdkit_valid": True,
                            "no_solved_claim": True,
                        }
                    ],
                    "unresolved_tasks": [],
                    "reasons": [],
                    "no_solved_claim": True,
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        anchors = board["semisynthesis_anchors"]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["name"], "advanced cyclic diol")
        self.assertEqual(anchors[0]["source_ref"], "doi:source")
        self.assertTrue(anchors[0]["smiles"])
        bridge_tasks = [row for row in board["bridge_tasks"] if str(row.get("task_id") or "").startswith("semisynthesis_bridge:")]
        self.assertEqual(len(bridge_tasks), 1)
        guided_payload = build_agentic_guided_payload(board)
        self.assertIn(
            anchors[0]["smiles"],
            guided_payload["search_policy"]["source_budget"]["semisynthesis_anchor_smiles"],
        )

    def test_resolved_small_taxane_fragment_is_not_promoted_as_semisynthesis_anchor(self):
        target = TargetInput(target_name="bufotalin_like_complex_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_taxane_intermediates",
                "task_type": "resolve_literature_structure",
                "label": "all taxane intermediates 5b-15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "resolve:taxane_intermediates",
                "action_type": "resolve_literature_structure_task",
                "rationale": "resolve source taxane intermediates",
                "expected_artifact": "structure resolution",
                "success_condition": "resolved or unresolved",
                "payload": {
                    "task_id": "resolve_structure:doi_source_taxane_intermediates",
                    "label": "all taxane intermediates 5b-15",
                    "source_ref": "doi:source",
                },
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "literature_structure_resolution_result.v1",
                    "accepted": True,
                    "status": "resolved",
                    "task_id": "resolve_structure:doi_source_taxane_intermediates",
                    "label": "all taxane intermediates 5b-15",
                    "source_ref": "doi:source",
                    "resolved_structures": [
                        {
                            "schema_version": "literature_resolved_structure_candidate.v1",
                            "structure_id": "resolve_structure_doi_source_taxane_intermediates:small",
                            "task_id": "resolve_structure:doi_source_taxane_intermediates",
                            "label": "all taxane intermediates 5b-15",
                            "smiles": "CC1=CC(=O)CCC1=O",
                            "source_ref": "doi:source",
                            "source_locator": "source reported taxane intermediate panel",
                            "accepted": True,
                            "rdkit_valid": True,
                            "no_solved_claim": True,
                        }
                    ],
                    "unresolved_tasks": [],
                    "reasons": [],
                    "no_solved_claim": True,
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        self.assertEqual(board["semisynthesis_anchors"], [])
        bridge_tasks = [row for row in board["bridge_tasks"] if str(row.get("task_id") or "").startswith("semisynthesis_bridge:")]
        self.assertEqual(bridge_tasks, [])

    def test_blackboard_records_unresolved_structure_attempt_keeps_task_open(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=4)
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "resolve:15",
                "action_type": "resolve_literature_structure_task",
                "rationale": "resolve one label",
                "expected_artifact": "structure resolution",
                "success_condition": "resolved or unresolved",
                "payload": {"task_id": "resolve_structure:doi_source_compound_15", "label": "compound 15", "source_ref": "doi:source"},
            },
            action_result={
                "accepted": False,
                "result": {
                    "schema_version": "literature_structure_resolution_result.v1",
                    "accepted": False,
                    "status": "unresolved",
                    "task_id": "resolve_structure:doi_source_compound_15",
                    "label": "compound 15",
                    "source_ref": "doi:source",
                    "resolved_structures": [],
                    "unresolved_tasks": [
                        {
                            "schema_version": "literature_structure_unresolved_task.v1",
                            "task_id": "resolve_structure:doi_source_compound_15",
                            "label": "compound 15",
                            "source_ref": "doi:source",
                            "status": "unresolved",
                            "reason": "no_rdkit_valid_source_grounded_structure_candidate",
                            "no_solved_claim": True,
                        }
                    ],
                    "reasons": ["no_rdkit_valid_structure_candidate"],
                    "no_solved_claim": True,
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        task = board["literature_evidence"]["structure_resolution_tasks"][0]
        self.assertEqual(task["status"], "open")
        self.assertEqual(task["last_resolution_status"], "unresolved")
        self.assertEqual(task["resolution_attempt_count"], 1)
        self.assertEqual(len(board["literature_evidence"]["structure_resolution_attempts"]), 1)
        self.assertTrue(board["action_history"][-1]["useful_artifact"])

    def test_planner_resolves_structure_task_before_structure_scout(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_visual_calls": 8},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(source_ref="doi:source", pdf_path="/tmp/source.pdf")
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "doi:source",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "compound 15", "gap_type": "structure_gap"}],
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {
                "round_index": 4,
                "action_type": "extract_visual_literature_chain",
                "action_signature": '{"payload":{"focused_gap_repair":true}}',
                "useful_artifact": False,
                "stale": True,
            },
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "resolve_literature_structure_task")
        self.assertEqual(first["payload"]["task_id"], "resolve_structure:doi_source_compound_15")
        self.assertEqual(first["payload"]["label"], "compound 15")
        self.assertTrue(first["payload"]["no_solved_claim"])
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_planner_scouts_structure_resolution_sources_after_visual_gaps(self):
        target = TargetInput(target_name="steroid", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(source_ref="doi:source", pdf_path="/tmp/source.pdf")
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "doi:source",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "compound 15", "gap_type": "structure_gap"}],
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
                "resolution_attempt_count": 1,
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "action_signature": "{}", "useful_artifact": False, "stale": True},
            {
                "round_index": 5,
                "action_type": "extract_visual_literature_chain",
                "action_signature": '{"payload":{"focused_gap_repair":true}}',
                "useful_artifact": False,
                "stale": True,
            },
        ]
        board["budget_state"]["scout_calls"] = 1

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "search_literature")
        self.assertTrue(first["payload"]["focused_structure_resolution"])
        self.assertIn("resolve_structure:doi_source_compound_15", first["payload"]["structure_resolution_task_ids"])

    def test_planner_scouts_structure_resolution_after_stale_compile_even_with_uncompiled_visual_steps(self):
        target = TargetInput(target_name="steroid", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_visual_calls": 2, "max_scout_calls": 4},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h1"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["budget_state"]["visual_calls"] = 2
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:source", "accepted": True}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual1",
                "source_ref": "doi:source",
                "accepted": True,
                "candidate_step_count": 3,
                "step_count": 3,
            }
        ]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "schema_version": "agent_exact_chain_audit_summary.v1",
                "accepted": False,
                "source_ref": "doi:source",
                "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
            }
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_source_compound_15",
                "task_type": "resolve_literature_structure",
                "label": "compound 15",
                "source_ref": "doi:source",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "compile_exact_literature_rows", "action_signature": "{}", "useful_artifact": False, "stale": True},
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "search_literature")
        self.assertTrue(first["payload"]["focused_structure_resolution"])
        self.assertIn("resolve_structure:doi_source_compound_15", first["payload"]["structure_resolution_task_ids"])
        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn("derive_broad_reaction_template", action_types)
        self.assertNotIn("compile_exact_literature_rows", action_types)

    def test_advisory_exact_compile_audit_turns_visual_chain_into_broad_template(self):
        target = TargetInput(target_name="paclitaxel_like", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual:source",
                "source_ref": "doi:source",
                "accepted": True,
                "candidate_step_count": 1,
                "steps": [
                    {
                        "step_id": "scheme_step",
                        "product_label": "paclitaxel",
                        "reactant_labels": ["baccatin III", "side chain"],
                        "reaction_class": "side-chain installation",
                    }
                ],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:source",
                pdf_path="/tmp/source.pdf",
                schema_version="literature_pdf_structure_evidence.v1",
            )
        ]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "schema_version": "agent_exact_chain_audit_summary.v1",
                "audit_id": "audit:advisory",
                "accepted": False,
                "reasons": [
                    "advisory_visual_template_card_available",
                    "source_detail_step_not_exact",
                    "no_chain_unrolled",
                ],
            }
        ]

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=False)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertIn("derive_broad_reaction_template", action_types)
        self.assertNotIn("compile_exact_literature_rows", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_codex_batch_missing_required_advisory_broad_template_is_salvaged_locally(self):
        target = TargetInput(target_name="paclitaxel_like_codex_advisory", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:source", "local_pdf": "/tmp/source.pdf"}
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual:source",
                "source_ref": "doi:source",
                "accepted": True,
                "candidate_step_count": 1,
                "steps": [
                    {
                        "step_id": "scheme_step",
                        "product_label": "paclitaxel",
                        "reactant_labels": ["baccatin III", "side chain"],
                        "reaction_class": "side-chain installation",
                    }
                ],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:source",
                pdf_path="/tmp/source.pdf",
                schema_version="literature_pdf_structure_evidence.v1",
            )
        ]
        board["literature_evidence"]["exact_chain_audits"] = [
            {
                "schema_version": "agent_exact_chain_audit_summary.v1",
                "audit_id": "audit:advisory",
                "accepted": False,
                "reasons": ["advisory_visual_template_card_available", "no_chain_unrolled"],
            }
        ]
        codex_batch = {
            "schema_version": "agent_action_batch.v1",
            "round_index": 6,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "codex_extract_more_visual",
                    "action_type": "extract_visual_literature_chain",
                    "rationale": "inspect the same source for more visual detail",
                    "expected_artifact": "visual chain",
                    "success_condition": "chain or failure",
                    "payload": {"source_ref": "doi:source", "pdf_path": "/tmp/source.pdf", "no_solved_claim": True},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        initial_validation = validate_action_batch(codex_batch, blackboard=board)
        self.assertIn("advisory_visual_template_requires_broad_template", initial_validation["reasons"])

        with tempfile.TemporaryDirectory() as tmp:
            batch = plan_action_batch_with_codex(
                blackboard=board,
                round_index=6,
                run_dir=Path(tmp),
                enabled=True,
                mock_output=codex_batch,
            )

        action_types = [row["action_type"] for row in batch["actions"]]
        self.assertIn("derive_broad_reaction_template", action_types)
        self.assertTrue(batch["codex_action_planner"]["repair_used"])
        self.assertEqual(batch["codex_action_planner"]["repair_source"], "guarded_budget_salvage_of_codex_batch")
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_compile_exact_rejected_after_broad_templates_without_uncompiled_visual_steps(self):
        target = TargetInput(target_name="broad_template_no_more_exact", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "visual:advisory",
                "source_ref": "doi:source",
                "accepted": True,
                "candidate_step_count": 0,
            }
        ]
        board["broad_transform_templates"] = [
            {
                "schema_version": "broad_transform_template.v1",
                "template_id": "broad:side_chain_installation",
                "allowed_use": "planner_priority_and_guided_search_hint_only",
                "no_solved_claim": True,
            }
        ]
        batch = {
            "schema_version": "agent_action_batch.v1",
            "case_id": board["case_id"],
            "round_index": 7,
            "actions": [
                {
                    "schema_version": "agent_action.v1",
                    "action_id": "try_exact_again",
                    "action_type": "compile_exact_literature_rows",
                    "rationale": "retry exact rows",
                    "expected_artifact": "exact rows",
                    "success_condition": "rows",
                    "payload": {"source_ref": "doi:source", "no_solved_claim": True},
                }
            ],
            "semantics": {
                "planner_can_emit_solved": False,
                "raw_reaction_output_allowed": False,
                "deterministic_validator_required": True,
            },
        }

        validation = validate_action_batch(batch, blackboard=board)
        self.assertIn("compile_exact_literature_rows_requires_uncompiled_visual_steps:0", validation["reasons"])

    def test_planner_scouts_structure_resolution_after_all_visual_sources_have_only_gaps(self):
        target = TargetInput(target_name="steroid", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=8)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h_core"}]}
        board["analogical_hypotheses"] = [{"hypothesis_id": "h_core"}]
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h_core"}]}
        board["literature_evidence"]["source_candidates"] = [
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:reddy", "local_pdf": "/tmp/reddy.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:zhang", "local_pdf": "/tmp/zhang.pdf"},
            {"schema_version": "literature_source_candidate.v1", "source_ref": "doi:chen", "local_pdf": "/tmp/chen.pdf"},
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:reddy", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:zhang", "accepted": True},
            {"schema_version": "agent_pdf_structure_evidence_summary.v1", "source_ref": "doi:chen", "accepted": True},
        ]
        board["literature_evidence"]["visual_chains"] = [
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "reddy_visual",
                "source_ref": "doi:reddy",
                "accepted": False,
                "candidate_step_count": 1,
                "condition_gap_labels": ["1b ouabagenin"],
                "extraction_gaps": [{"label": "27", "gap_type": "structure_gap"}],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "zhang_visual",
                "source_ref": "doi:zhang",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "ouabagenin", "gap_type": "structure_gap"}],
            },
            {
                "schema_version": "agent_visual_chain_summary.v1",
                "chain_id": "chen_visual",
                "source_ref": "doi:chen",
                "accepted": False,
                "candidate_step_count": 0,
                "extraction_gaps": [{"label": "ouabagenin precursor", "gap_type": "structure_gap"}],
            },
        ]
        board["literature_evidence"]["structure_resolution_tasks"] = [
            {
                "schema_version": "agent_structure_resolution_task.v1",
                "task_id": "resolve_structure:doi_reddy_27",
                "task_type": "resolve_literature_structure",
                "label": "27",
                "source_ref": "doi:reddy",
                "status": "open",
            }
        ]
        board["action_history"] = [
            {"round_index": 1, "action_type": "search_literature", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 2, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 3, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 4, "action_type": "extract_pdf_literature_structures", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {"round_index": 5, "action_type": "extract_visual_literature_chain", "action_signature": "{}", "useful_artifact": True, "stale": False},
            {
                "round_index": 6,
                "action_type": "extract_visual_literature_chain",
                "action_signature": '{"payload":{"focused_gap_repair":true}}',
                "useful_artifact": False,
                "stale": True,
            },
        ]
        board["budget_state"]["scout_calls"] = 1
        board["budget_state"]["visual_calls"] = 6

        batch = plan_action_batch(board, round_index=7, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "search_literature")
        self.assertTrue(first["payload"]["focused_structure_resolution"])
        self.assertNotEqual(first["action_type"], "compile_exact_literature_rows")

    def test_compile_duplicate_disconnected_exact_rows_are_stale(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["literature_evidence"]["exact_rows"] = [{"row_id": "source_detail_exact_step:ethanol"}]

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:duplicate",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile duplicate rows",
                "expected_artifact": "compiled rows",
                "success_condition": "rows",
                "payload": {},
            },
            action_result={
                "accepted": False,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "compiled_downstream": {
                        "literature_template_plugin": {
                            "one_step_rows": [
                                {
                                    "row_id": "source_detail_exact_step:ethanol",
                                    "product_smiles": "CCO",
                                }
                            ]
                        }
                    },
                    "chain_audit": {
                        "accepted": False,
                        "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
                        "summary": {"one_step_row_count": 1, "chain_step_count": 0},
                    },
                    "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
                },
                "reasons": ["missing_one_step_row_for_product", "no_chain_unrolled"],
            },
            round_index=4,
            run_dir="/tmp",
        )

        self.assertFalse(board["action_history"][-1]["useful_artifact"])
        self.assertTrue(board["action_history"][-1]["stale"])
        self.assertEqual(len(board["literature_evidence"]["exact_rows"]), 1)
        self.assertFalse(board["literature_evidence"]["exact_chain_audits"][0]["accepted"])

    def test_compile_exact_rows_with_only_accepted_audit_is_stale(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:empty",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile empty chain",
                "expected_artifact": "compiled rows",
                "success_condition": "rows",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "compiled_downstream": {"literature_template_plugin": {"one_step_rows": []}},
                    "chain_audit": {
                        "accepted": True,
                        "summary": {"one_step_row_count": 0, "chain_step_count": 0},
                    },
                    "reasons": ["no_chain_unrolled"],
                },
                "reasons": ["no_chain_unrolled"],
            },
            round_index=3,
            run_dir="/tmp",
        )

        self.assertFalse(board["action_history"][-1]["useful_artifact"])
        self.assertTrue(board["action_history"][-1]["stale"])
        self.assertEqual(board["literature_evidence"]["exact_rows"], [])

    def test_compile_exact_rows_marks_sugar_fragment_not_parent_relevant(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:sugar",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile exact rows",
                "expected_artifact": "exact rows",
                "success_condition": "rows are recorded",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "exact_rows": [
                        {
                            "row_id": "source_detail_exact_step:rhamnose_donor",
                            "product_smiles": "OCC1OC(O)C(O)C(O)C1O",
                            "product_label": "rhamnose sugar donor",
                        }
                    ],
                },
            },
            round_index=3,
            run_dir="/tmp",
        )

        row = board["literature_evidence"]["exact_rows"][0]
        self.assertFalse(row["target_relevant_for_parent_bridge"])
        self.assertIn("product_ring_system_too_small_for_target_core", row["target_relevance"]["reasons"])
        self.assertEqual(board["literature_evidence"]["exact_row_target_relevance_summary"]["target_relevant_exact_rows"], 0)
        guided = build_agentic_guided_payload(board)
        self.assertEqual(guided["search_policy"]["accepted_exact_row_ids"], [])
        self.assertEqual(
            guided["search_policy"]["source_budget"]["disconnected_exact_row_ids"],
            ["source_detail_exact_step:rhamnose_donor"],
        )

    def test_planner_does_not_repeat_large_jump_guided_without_new_strong_signal(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_guided_chemenzy_runs": 2},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["route_failures"] = [{"reason": "unexplained_large_atom_jump", "route_status": "rejected"}]
        board["literature_evidence"]["exact_rows"] = [
            {
                "row_id": "source_detail_exact_step:rhamnose_donor",
                "product_smiles": "OCC1OC(O)C(O)C(O)C1O",
                "target_relevant_for_parent_bridge": False,
                "target_relevance": {"target_relevant_for_parent_bridge": False},
            }
        ]
        board["action_history"] = [
            {
                "round_index": 5,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
                "action_signature": json.dumps({"action_type": "run_guided_chemenzy"}),
            }
        ]
        board["budget_state"]["chemenzy_runs"] = 1

        batch = plan_action_batch(board, round_index=6, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("run_guided_chemenzy", action_types)

    def test_planner_does_not_repeat_guided_without_new_signal_after_unresolved_probe(self):
        target = TargetInput(target_name="steroid_target", target_smiles=BUFOTALIN_SMILES, family_hint="steroid")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(
            target_input=target.to_dict(),
            preflight=preflight,
            max_rounds=8,
            budget_limits={"max_guided_chemenzy_runs": 3},
        )
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["analogical_hypotheses"] = list(board["target_side_disconnection_hypotheses"]["hypotheses"])
        board["analogical_hypothesis_ranking"] = {"selected_hypotheses": [{"hypothesis_id": "h1"}]}
        board["bridge_tasks"] = [{"task_id": "bridge:core", "task_type": "target_proximal_bridge"}]
        board["broad_transform_templates"] = [{"template_id": "broad:core", "objective_type": "same_core"}]
        board["action_history"] = [
            {
                "round_index": 4,
                "action_type": "run_guided_chemenzy",
                "useful_artifact": True,
                "stale": False,
                "blackboard_delta": {"artifact_refs": 1},
            }
        ]
        board["budget_state"]["chemenzy_runs"] = 1

        batch = plan_action_batch(board, round_index=5, exhaust_round_budget=True)
        action_types = [row["action_type"] for row in batch["actions"]]

        self.assertNotIn("run_guided_chemenzy", action_types)
        validation = validate_action_batch(batch, blackboard=board)
        self.assertTrue(validation["accepted"], validation["reasons"])

    def test_compile_exact_rows_keeps_untrusted_visual_candidate_steps_advisory(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "ethanol_visual_chain",
            "source_ref": "doi:10.0000/source",
            "evidence_refs": ["current_image:1"],
            "steps": [
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "visual_step_1_ethanol",
                    "segment_id": "visual_chain",
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_labels": ["ethane"],
                    "reactant_smiles": ["CC"],
                    "condition": {
                        "schema_version": "condition_candidate.v1",
                        "source_type": "exact",
                        "condition_status": "evidence_backed",
                        "reagent": "oxidation conditions",
                        "source_grounding": "current PDF image",
                    },
                    "source_locator": "scheme 1",
                },
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "visual_step_2_ethane",
                    "segment_id": "visual_chain",
                    "product_label": "ethane",
                    "product_smiles": "CC",
                    "reactant_labels": ["methane"],
                    "reactant_smiles": ["C"],
                    "condition": {
                        "schema_version": "condition_candidate.v1",
                        "source_type": "exact",
                        "condition_status": "evidence_backed",
                        "reagent": "coupling conditions",
                        "source_grounding": "current PDF image",
                    },
                    "source_locator": "scheme 1",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        plugin = result["compiled_downstream"]["literature_template_plugin"] or {}
        self.assertFalse(result["accepted"])
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 2)

    def test_compile_exact_rows_does_not_promote_untrusted_bufotalin_stereo_repair(self):
        target_smiles = "CC12CCC(=O)C=C1CCC1C2CCC2(C)C1CCC21OCCO1"
        achiral_androstenedione = "CC12CCC3C(C1CCC2=O)CCC4=CC(=O)CCC34C"
        target = TargetInput(target_name="compound_24", target_smiles=target_smiles)
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "bufotalin_visual_chain",
            "source_ref": "doi:10.1016/j.tet.2025.134610",
            "evidence_refs": ["current_image:scheme3.png"],
            "steps": [
                {
                    "schema_version": "visual_structure_candidate_step.v1",
                    "step_id": "24_from_11",
                    "product_label": "24",
                    "product_smiles": target_smiles,
                    "reactant_labels": ["11"],
                    "reactant_smiles": [achiral_androstenedione],
                    "condition": {
                        "schema_version": "condition_candidate.v1",
                        "source_type": "exact",
                        "condition_status": "evidence_backed",
                        "reagent": "ethylene glycol, p-TsOH",
                        "source_grounding": "Scheme 3 ketalization of androstenedione (11) to 24",
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        self.assertFalse(result["accepted"])
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        plugin = result["compiled_downstream"]["literature_template_plugin"] or {}
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 1)

    def test_blackboard_promotes_literature_terminal_to_upstream_child_task(self):
        target = TargetInput(target_name="bufotalin", target_smiles=BUFOTALIN_SMILES)
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        terminal_smiles = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"

        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "compile:1",
                "action_type": "compile_exact_literature_rows",
                "rationale": "compile exact rows",
                "expected_artifact": "rows",
                "success_condition": "rows",
                "payload": {},
            },
            action_result={
                "accepted": True,
                "result": {
                    "schema_version": "compiled_source_detail_chain_route.v1",
                    "accepted": True,
                    "compiled_downstream": {
                        "literature_template_plugin": {
                            "one_step_rows": [
                                {
                                    "template": {
                                        "literature_template_trace": {
                                            "source_template_id": "source_detail_exact_step:24_from_11",
                                            "source_ref": "doi:10.1016/j.tet.2025.134610",
                                            "product_smiles": "CCO",
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "chain_audit": {
                        "accepted": True,
                        "terminal_name": "Androstenedione",
                        "terminal_smiles": terminal_smiles,
                        "terminal_canonical_smiles": terminal_smiles,
                        "terminal_reached": True,
                        "step_count": 15,
                        "chain": [{"source_ref": "doi:10.1016/j.tet.2025.134610"}],
                    },
                },
            },
            round_index=4,
            run_dir="/tmp",
        )

        terminals = board["literature_evidence"]["terminal_candidates"]
        tasks = board["bridge_tasks"]
        self.assertEqual(terminals[0]["name"], "Androstenedione")
        self.assertEqual(tasks[0]["task_type"], "upstream_terminal_synthesis")
        self.assertEqual(tasks[0]["terminal"]["smiles"], terminal_smiles)

    def test_compile_exact_rows_keeps_visual_candidate_chain_shape_advisory(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "ethanol_visual_chain",
            "doi": "10.0000/source",
            "evidence_refs": ["current_image:1"],
            "candidate_chain": [
                {
                    "label": "ethanol",
                    "smiles": "CCO",
                    "precursor_label": "ethane",
                    "precursor_smiles": "CC",
                    "source_locator": "scheme 1",
                    "conditions": {"reagents": "oxidation conditions", "reported_yield": "80%"},
                },
                {
                    "label": "ethane",
                    "smiles": "CC",
                    "precursor_label": "methane",
                    "precursor_smiles": "C",
                    "source_locator": "scheme 1",
                    "conditions": {"reagents": "coupling conditions", "reported_yield": "70%"},
                },
                {
                    "label": "methane",
                    "smiles": "C",
                    "precursor_label": None,
                    "precursor_smiles": None,
                    "source_locator": "scheme 1 starting material",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        plugin = result["compiled_downstream"]["literature_template_plugin"] or {}
        self.assertFalse(result["accepted"])
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 2)

    def test_compile_exact_rows_keeps_visual_reaction_chain_shape_advisory(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        candidate_chain = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "case_id": "ethanol_visual_chain",
            "source_ref": "doi:10.0000/source",
            "evidence_refs": ["current_image:1"],
            "route_order": "retro_target_to_start",
            "chain": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_label": "ethane",
                    "reactant_smiles": "CC",
                    "conditions": "oxidation conditions, 80%",
                    "source_locator": "scheme 1",
                },
                {
                    "product_label": "ethane",
                    "product_smiles": "CC",
                    "reactant_label": "methane",
                    "reactant_smiles": "C",
                    "conditions": "coupling conditions, 70%",
                    "source_locator": "scheme 1",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain"] = candidate_chain
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        plugin = result["compiled_downstream"]["literature_template_plugin"] or {}
        self.assertFalse(result["accepted"])
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(len(plugin["template_cards"]), 2)

    def test_compile_exact_rows_keeps_source_backed_visual_candidate_advisory(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")
        draft_without_source = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_labels": ["ethane"],
                    "reactant_smiles": ["CC"],
                    "condition": {"reagent": "oxidation conditions"},
                    "source_locator": "scheme 1",
                }
            ],
        }
        normalized_candidate = {
            **draft_without_source,
            "source_ref": "doi:10.0000/source",
            "evidence_refs": ["current_image:scheme1.png"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_structure_candidate_chain_history"] = [draft_without_source]
            state.artifacts["visual_structure_candidate_chain"] = normalized_candidate
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        plugin = result["compiled_downstream"]["literature_template_plugin"] or {}
        self.assertFalse(result["accepted"])
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        self.assertEqual(plugin["one_step_rows"], [])
        self.assertEqual(plugin["template_cards"][0]["condition_source"], "doi:10.0000/source")

    def test_compile_exact_rows_enriches_but_does_not_promote_untrusted_visual_output(self):
        target = TargetInput(target_name="ethanol", target_smiles="CCO")
        preflight = run_preflight(target)
        target.case_id = str(preflight.get("case_id") or "")

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            state.artifacts["visual_literature_chain_extraction"] = {
                "schema_version": "visual_literature_chain_extraction_result.v1",
                "accepted": True,
                "source_ref": "doi:10.0000/source",
                "source_title": "Visual source",
                "image_paths": ["/tmp/scheme1.png"],
                "parsed_output": {
                    "schema_version": "visual_structure_candidate_chain.v1",
                    "steps": [
                        {
                            "product_label": "ethanol",
                            "product_smiles": "CCO",
                            "reactant_labels": ["ethane"],
                            "reactant_smiles": "CC",
                            "condition": {"reagent": "oxidation conditions"},
                            "source_locator": "scheme 1",
                        }
                    ],
                },
            }
            record = execute_local_tool("compile_source_detail_chain_route", {}, state)

        result = record.output["result"]
        plugin = result["compiled_downstream"]["literature_template_plugin"] or {}
        self.assertFalse(result["accepted"])
        self.assertIn("source_detail_step_not_trusted_curated", result["reasons"])
        self.assertEqual(plugin["one_step_rows"], [])
        card = plugin["template_cards"][0]
        self.assertEqual(card["condition_source"], "doi:10.0000/source")
        self.assertIn("current_image:/tmp/scheme1.png", card["evidence_refs"])

    def test_route_expansion_prioritizes_literature_starting_material_over_near_target_intermediate(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        compiled = {
            "route_expansion": {
                "child_targets": [
                    {
                        "name": "bufotalin_source_detail_exact_step_bufotalin_from_33_reactant_1",
                        "smiles": "CCO",
                        "source": "source_detail_route_expansion",
                        "source_template_id": "source_detail_exact_step:bufotalin_from_33",
                    },
                    {
                        "name": "bufotalin_source_detail_exact_step_24_from_11_reactant_1",
                        "smiles": "CC",
                        "source": "source_detail_route_expansion",
                        "source_template_id": "source_detail_exact_step:24_from_11",
                    },
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            state = ToolExecutionState(
                run_dir=Path(tmp),
                target_input=target.to_dict(),
                preflight=preflight,
                budget=HarnessBudget(timeout_s=30.0),
            )
            rows = _route_expansion_child_targets(state=state, payload={}, compiled=compiled)

        self.assertEqual(rows[0]["source_template_id"], "source_detail_exact_step:24_from_11")

    def test_visual_repair_steps_accept_single_reactant_smiles_string(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_label": "ethane",
                    "reactant_smiles": "CC",
                    "condition": "oxidation conditions, 80%",
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(len(chain["steps"]), 1)
        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "oxidation conditions, 80%")

    def test_visual_repair_steps_accept_condition_candidate_string(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "chain": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactant_label": "ethane",
                    "reactant_smiles": "CC",
                    "condition": "oxidation conditions, 80%",
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "oxidation conditions, 80%")

    def test_visual_repair_steps_accept_precursor_chain_shape(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "chain": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "precursor_label": "ethane",
                    "precursor_smiles": "CC",
                    "forward_conditions": {"reagent": "oxidation conditions", "reported_yield": "80%"},
                    "source_location": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "oxidation conditions")

    def test_visual_repair_steps_accept_plural_precursor_fields(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "precursor_labels": ["ethane"],
                    "precursor_smiles": ["CC"],
                    "condition_candidate": {"reagent": "oxidation conditions"},
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])

    def test_visual_repair_steps_accept_reactant_object_fields(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "steps": [
                {
                    "product_label": "ethanol",
                    "product_smiles": "CCO",
                    "reactants": [{"label": "ethane", "smiles": "CC"}],
                    "condition_candidate": {"reagent": "oxidation conditions"},
                    "source_locator": "scheme 1",
                }
            ],
            "extraction_gaps": [],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["CC"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethane"])

    def test_pdf_defaults_do_not_infer_focus_from_target_or_doi(self):
        payload = {}
        _inject_pdf_defaults(
            payload,
            {
                "target_name": "bufotalin",
                "family_hint": "bufadienolide",
                "literature_pdf_path": "/tmp/bufotalin.pdf",
                "literature_pdf_source_ref": "doi:10.1016/j.tet.2025.134610",
            },
        )

        self.assertEqual(payload["pdf_path"], "/tmp/bufotalin.pdf")
        self.assertEqual(payload["source_ref"], "doi:10.1016/j.tet.2025.134610")
        self.assertNotIn("page_numbers", payload)
        self.assertNotIn("render_zoom", payload)
        self.assertNotIn("scheme_crops", payload)
        self.assertNotIn("compound_labels", payload)
        self.assertNotIn("expected_labels", payload)
        self.assertNotIn("route_sequence_hint", payload)

    def test_visual_quality_flags_condition_gaps(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "steps": [
                    {
                        "product_label": "ethanol",
                        "product_smiles": "CCO",
                        "reactant_label": "ethane",
                        "reactant_smiles": "CC",
                        "condition_candidate": {
                            "schema_version": "condition_candidate.v1",
                            "source_type": "exact",
                            "condition_status": "evidence_backed",
                            "source_grounding": "current PDF image",
                        },
                        "source_locator": "scheme 1",
                    }
                ],
            },
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        quality = _candidate_quality(chain, expected_labels=["ethanol"])

        self.assertTrue(quality["accepted"])
        self.assertFalse(quality["exact_ready"])
        self.assertTrue(quality["exploratory_accepted"])
        self.assertEqual(quality["condition_gap_labels"], ["ethanol"])
        self.assertEqual(quality["condition_gap_count"], 1)

    def test_visual_structure_anchors_are_exploratory_without_precursor_or_conditions(self):
        parsed = {
            "schema_version": "visual_structure_candidate_chain.v1",
            "route_order": "retro_target_to_start",
            "source_ref": "src_web_003",
            "source_title": "The Story of LIPITOR",
            "steps": [
                {
                    "visible_label": "Lipitor / atorvastatin acid connectivity corresponding to calcium salt",
                    "product_smiles": ATORVASTATIN_FREE_ACID_SMILES,
                    "conditions_from_source": {
                        "other_visible_process_text": ["Chemical Synthesis", "Biocatalysis"],
                    },
                    "not_exact_literature_segment": True,
                    "allowed_use": "exploratory_template_and_guided_hint_only",
                    "visual_evidence": ["The source shows the atorvastatin calcium connectivity."],
                },
                {
                    "visible_label": "pyrrole template / atorvastatin pyrrole core",
                    "product_smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)[nH]c1-c1ccccc1",
                    "not_exact_literature_segment": True,
                    "allowed_use": "exploratory_template_and_guided_hint_only",
                    "visual_evidence": ["The pyrrole core is visible inside atorvastatin."],
                },
            ],
            "extraction_gaps": [
                {
                    "label": "process development",
                    "gap_type": "label_visibility_gap",
                    "reason": "label not visible in the supplied images",
                }
            ],
        }

        chain = _candidate_chain_from_parsed(
            parsed,
            target_name="atorvastatin",
            target_smiles=ATORVASTATIN_FREE_ACID_SMILES,
            source_ref="src_web_003",
            source_title="The Story of LIPITOR",
            image_paths=[],
        )
        quality = _candidate_quality(
            chain,
            expected_labels=["Lipitor", "pyrrole template", "process development"],
        )

        self.assertEqual(chain["steps"][0]["product_label"], "Lipitor / atorvastatin acid connectivity corresponding to calcium salt")
        self.assertTrue(chain["steps"][0]["structure_derivation"]["visual_structure_anchor_only"])
        self.assertEqual(quality["structure_gap_count"], 0)
        self.assertEqual(quality["rdkit_structure_anchor_count"], 2)
        self.assertEqual(quality["acceptance_level"], "exploratory_connectivity_candidate")
        self.assertTrue(quality["accepted"], quality)

    def test_visual_target_label_uses_input_target_smiles_when_visual_target_smiles_is_malformed(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "steps": [
                    {
                        "product_label": "bufotalin",
                        "product_smiles": "CC(",
                        "reactant_label": "33",
                        "reactant_smiles": "CC",
                        "condition": "HF-pyridine, 93%",
                        "source_locator": "scheme 4",
                    }
                ],
            },
            target_name="bufotalin",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["steps"][0]["product_smiles"], "CCO")
        self.assertTrue(chain["steps"][0]["structure_derivation"]["target_product_smiles_fallback"])
        quality = _candidate_quality(chain, expected_labels=["bufotalin"])
        self.assertEqual(quality["smiles_precheck"]["invalid_smiles_count"], 0)

    def test_visual_target_label_preserves_input_target_stereo_when_visual_smiles_is_achiral(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "target": {"name": "bufotalin", "smiles": BUFOTALIN_ACHIRAL_SMILES},
                "steps": [
                    {
                        "product_label": "bufotalin",
                        "product_smiles": BUFOTALIN_ACHIRAL_SMILES,
                        "reactant_label": "33",
                        "reactant_smiles": "CC",
                        "condition": "HF-pyridine, 93%",
                        "source_locator": "scheme 4",
                    }
                ],
            },
            target_name="bufotalin",
            target_smiles=BUFOTALIN_SMILES,
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(chain["target_smiles"], BUFOTALIN_SMILES)
        self.assertEqual(chain["steps"][0]["product_smiles"], BUFOTALIN_SMILES)
        self.assertTrue(chain["steps"][0]["structure_derivation"]["target_product_stereo_repair"])
        quality = _candidate_quality(chain, expected_labels=["bufotalin"])
        self.assertEqual(quality["smiles_precheck"]["invalid_smiles_count"], 0)

    def test_visual_candidate_steps_field_is_normalized(self):
        chain = _candidate_chain_from_parsed(
            {
                "schema_version": "visual_structure_candidate_chain.v1",
                "route_order": "retro_target_to_start",
                "candidate_steps": [
                    {
                        "product_label": "ethanol",
                        "product_smiles": "CCO",
                        "precursor_labels": ["ethene"],
                        "precursor_smiles": ["C=C"],
                        "condition_candidate": {"reagent": "hydration"},
                        "source_locator": "scheme 2",
                    }
                ],
            },
            target_name="ethanol",
            target_smiles="CCO",
            source_ref="doi:10.0000/source",
            source_title="Visual source",
            image_paths=[],
        )

        self.assertEqual(len(chain["steps"]), 1)
        self.assertEqual(chain["steps"][0]["product_smiles"], "CCO")
        self.assertEqual(chain["steps"][0]["reactant_smiles"], ["C=C"])
        self.assertEqual(chain["steps"][0]["reactant_labels"], ["ethene"])
        self.assertEqual(chain["steps"][0]["condition_candidate"]["reagent"], "hydration")

    def test_bufotalin_visual_path_does_not_use_target_specific_hardcoded_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "scheme4_total_synthesis.png"
            image.write_bytes(b"not a real image; path existence is enough for this unit test")
            with patch(
                "cascade_planner.harness.visual_literature_chain_agent._read_key",
                return_value="test-key",
            ), patch(
                "cascade_planner.harness.visual_literature_chain_agent._run_visual_json_prompt",
                return_value={
                    "status": "error",
                    "returncode": 1,
                    "raw_last_message": "",
                    "reasons": ["visual_backend_unavailable"],
                    "event_log_path": "",
                    "stderr_log_path": "",
                },
            ) as backend:
                result = run_visual_literature_chain_agent(
                    image_paths=[image],
                    output_dir=Path(tmp) / "out",
                    target_name="bufotalin",
                    target_smiles=BUFOTALIN_SMILES,
                    source_ref="doi:10.1016/j.tet.2025.134610",
                    source_title="Tetrahedron bufotalin total synthesis",
                    expected_labels=["bufotalin", "33", "32", "31", "30", "24", "11"],
                    route_sequence_hint="bufotalin <= 33 <= 32 <= 31 <= 30 <= 24 <= 11",
                    key_path=Path(tmp) / "key.txt",
                    base_url="https://example.invalid/v1",
                    model="test-model",
                )

        backend.assert_called_once()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "error")
        self.assertFalse((Path(tmp) / "out" / "visual_structure_candidate_chain.json").exists())

    def test_blackboard_condition_gaps_trigger_focused_visual_repair(self):
        target = TargetInput(target_name="bufotalin", target_smiles="CCO")
        preflight = run_preflight(target)
        board = initialize_agent_blackboard(target_input=target.to_dict(), preflight=preflight, max_rounds=5)
        board["target_side_disconnection_hypotheses"] = {"hypotheses": [{"hypothesis_id": "h1"}]}
        board["literature_evidence"]["source_candidates"] = [
            {
                "schema_version": "literature_source_candidate.v1",
                "source_ref": "doi:10.1016/j.tet.2025.134610",
                "local_pdf": "/tmp/bufotalin.pdf",
                "expected_scheme_or_compound_labels": ["bufotalin", "33", "11"],
            }
        ]
        board["literature_evidence"]["pdf_structure_evidence"] = [
            _rendered_pdf_evidence(
                source_ref="doi:10.1016/j.tet.2025.134610",
                pdf_path="/tmp/bufotalin.pdf",
                evidence_id="pdf",
            )
        ]
        board = update_blackboard_from_action(
            board,
            action={
                "schema_version": "agent_action.v1",
                "action_id": "visual:1",
                "action_type": "extract_visual_literature_chain",
                "rationale": "extract chain",
                "expected_artifact": "visual",
                "success_condition": "chain",
                "payload": {},
            },
            action_result={
                "accepted": False,
                "result": {
                    "schema_version": "visual_literature_chain_extraction_result.v1",
                    "accepted": False,
                    "candidate_step_count": 3,
                    "candidate_quality": {
                        "missing_expected_labels": [],
                        "condition_gap_labels": ["bufotalin", "33", "11"],
                    },
                    "reasons": ["visual_literature_chain_condition_gaps"],
                },
                "reasons": ["visual_literature_chain_condition_gaps"],
            },
            round_index=3,
            run_dir="/tmp",
        )
        board["budget_state"]["visual_calls"] = 1

        batch = plan_action_batch(board, round_index=4, exhaust_round_budget=True)
        first = batch["actions"][0]

        self.assertEqual(first["action_type"], "extract_visual_literature_chain")
        self.assertTrue(first["payload"]["focused_gap_repair"])
        self.assertIn("condition_candidate", first["payload"]["route_sequence_hint"])


if __name__ == "__main__":
    unittest.main()
