import unittest
import copy
import hashlib
import os
from pathlib import Path
import tempfile

from cascade_planner.legacy.harness_runtime.agentic_blackboard_controller import emit_agentic_final_verdict
from cascade_planner.legacy.harness_runtime.parent_route_proof import (
    compile_stitched_parent_route_proof,
    is_solved_parent_route_proof,
)
from cascade_planner.harness.route_verifier import (
    is_accepted_route_verifier_report,
    verify_chemenzy_raw_routes,
)
from cascade_planner.legacy.harness_runtime.runner import emit_final_verdict
from cascade_planner.harness.stitched_route import (
    compile_stitched_semisynthesis_route,
    is_reaction_validated_stitched_semisynthesis_route,
    is_solved_stitched_semisynthesis_route,
)


_SOURCE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "source_evidence_stub.pdf"
_SOURCE_PAGE_FIXTURE = Path(__file__).parents[1] / "fixtures" / "source_page.ppm"
_SOURCE_MANIFEST_FIXTURE = Path(__file__).parents[1] / "fixtures" / "source_evidence_manifest.json"
_TRUSTED_REGISTRY_FIXTURE = Path(__file__).parents[1] / "fixtures" / "trusted_literature_step_registry.json"


def _strict_literature_step(*, step_id: str, reactants: list[str], product: str) -> dict:
    pdf_digest = hashlib.sha256(_SOURCE_FIXTURE.read_bytes()).hexdigest()
    image_digest = hashlib.sha256(_SOURCE_PAGE_FIXTURE.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(_SOURCE_MANIFEST_FIXTURE.read_bytes()).hexdigest()
    template_id = f"source_detail_exact_step:{step_id}"
    mapped_reactions = {
        (tuple(["CC", "O"]), "CCO"): "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]",
        (tuple(["C", "C"]), "CC"): "[CH4:1].[CH4:2]>>[CH3:1][CH3:2]",
        (tuple(["O=O"]), "O"): "[O:1]=[O:2]>>[OH2:1]",
        (tuple(["CCO"]), "CC=O"): "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        (
            tuple(["CCO", "O"]),
            "CCOO",
        ): "[CH3:1][CH2:2][OH:3].[OH2:4]>>[CH3:1][CH2:2][O:3][OH:4]",
    }
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
    mapped = mapped_reactions.get((tuple(reactants), product), "")
    if mapped:
        row["atom_mapped_reaction_smiles"] = mapped
    return row


def _accepted_stitch_fixture() -> dict:
    terminal = "CCO"
    raw = _materialized_subgoal_raw(terminal)
    return compile_stitched_semisynthesis_route(
        literature_chain_audit={
            "schema_version": "source_detail_route_chain_audit.v1",
            "accepted": True,
            "target_smiles": "CC=O",
            "terminal_smiles": terminal,
            "terminal_reached": True,
            "source_ref": "doi:10.1000/revalidatable-stitch",
            "chain": [
                _strict_literature_step(
                    step_id="ethanol_oxidation",
                    reactants=[terminal],
                    product="CC=O",
                )
            ],
        },
        route_expansion_result={
            "subgoals": [
                {
                    "accepted": True,
                    "subgoal": {"name": "ethanol", "smiles": terminal},
                    "verifier": _strict_subgoal_verifier(terminal),
                }
            ]
        },
        subgoal_verifier=_strict_subgoal_verifier(terminal),
        subgoal_raw_result=raw,
        target_smiles="CC=O",
    )


def _strict_subgoal_verifier(target_smiles: str) -> dict:
    return verify_chemenzy_raw_routes(_materialized_subgoal_raw(target_smiles), target_smiles=target_smiles)


def _materialized_subgoal_raw(target_smiles: str) -> dict:
    reactants = ["C", "C"] if target_smiles == "CC" else ["CC", "O"]
    terminal_reactants = list(dict.fromkeys(reactants))
    strict_step = _strict_literature_step(
        step_id="methane_coupling" if target_smiles == "CC" else "ethanol_hydration",
        reactants=reactants,
        product=target_smiles,
    )
    strict_step["stock_status"] = {item: True for item in terminal_reactants}
    return {
        "target": target_smiles,
        "search_status": {"solved": True},
        "routes": [
            {
                "route_rank": 0,
                "n_steps": 1,
                "metrics": {
                    "terminal_reactants": terminal_reactants,
                    "terminal_stock_status": {item: True for item in terminal_reactants},
                },
                "steps": [strict_step],
            }
        ],
    }


def _strict_parent_verifier(
    target_smiles: str,
    *,
    precursor_smiles: str = "CC",
    include_rejected_route: bool = False,
) -> dict:
    accepted_reactants = [precursor_smiles, "O"] if target_smiles == "CCO" and precursor_smiles == "CC" else [precursor_smiles]
    accepted_terminals = list(dict.fromkeys(accepted_reactants))
    accepted_step = {
        "product": target_smiles,
        "reactant_smiles": accepted_reactants,
        "stock_status": {item: True for item in accepted_terminals},
        "atom_mapped_reaction_smiles": (
            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
            if target_smiles == "CCO" and accepted_reactants == ["CC", "O"]
            else ""
        ),
    }
    if target_smiles == "CCO" and accepted_reactants == ["CC", "O"]:
        accepted_step = _strict_literature_step(
            step_id="ethanol_hydration",
            reactants=accepted_reactants,
            product=target_smiles,
        )
        accepted_step["stock_status"] = {
            item: True for item in accepted_terminals
        }
    routes = [
        {
            "route_rank": 0,
            "metrics": {
                "terminal_reactants": accepted_terminals,
                "terminal_stock_status": {item: True for item in accepted_terminals},
            },
            "steps": [accepted_step],
        }
    ]
    if include_rejected_route:
        routes.append(
            {
                "route_rank": 1,
                "metrics": {"terminal_reactants": ["NNN"], "terminal_stock_status": {"NNN": True}},
                "steps": [{"product": target_smiles, "main_reactant": "NNN", "stock_status": {"NNN": True}}],
            }
        )
    return verify_chemenzy_raw_routes({"target": target_smiles, "routes": routes}, target_smiles=target_smiles)


class ParentRouteProofTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prior_registry = os.environ.get("AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY")
        os.environ["AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY"] = str(_TRUSTED_REGISTRY_FIXTURE)

    @classmethod
    def tearDownClass(cls):
        if cls._prior_registry is None:
            os.environ.pop("AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY", None)
        else:
            os.environ["AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY"] = cls._prior_registry

    def test_generic_route_audit_booleans_cannot_emit_solved_verdict(self):
        verdict = emit_final_verdict(
            {
                "case_id": "forged-route-audit",
                "preflight": {"accepted": True},
                "artifacts": {
                    "route_audit": {
                        "schema_version": "route_audit_report.v1",
                        "route_status": "solved",
                        "stock_audit_passed": True,
                        "reasons": [],
                    }
                },
            }
        )

        self.assertFalse(verdict.solved)
        self.assertNotEqual(verdict.route_status, "solved")
        self.assertIn("solved_requires_deterministic_parent_route_proof", verdict.reasons)

    def test_three_butanes_cut_and_glue_cannot_be_parent_eligible(self):
        reactant_mapping = ".".join(
            "".join(
                [
                    f"[CH3:{start}]",
                    f"[CH2:{start + 1}]",
                    f"[CH2:{start + 2}]",
                    f"[CH3:{start + 3}]",
                ]
            )
            for start in (1, 5, 9)
        )
        product_mapping = "".join(
            f"[CH3:{index}]" if index in {1, 12} else f"[CH2:{index}]"
            for index in range(1, 13)
        )
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "butane.csv"
            catalog.write_text("CCCC\n", encoding="utf-8")
            catalog_sha = hashlib.sha256(catalog.read_bytes()).hexdigest()
            parent = verify_chemenzy_raw_routes(
                {
                    "target": "CCCCCCCCCCCC",
                    "stock_catalog_context": {
                        "effective_stock_names": ["fixture_butane"],
                        "catalog_bindings": [
                            {
                                "name": "fixture_butane",
                                "path": str(catalog),
                                "sha256": catalog_sha,
                            }
                        ],
                    },
                    "routes": [
                        {
                            "route_rank": 0,
                            "metrics": {
                                "terminal_reactants": ["CCCC"],
                                "terminal_stock_status": {"CCCC": True},
                            },
                            "steps": [
                                {
                                    "product": "CCCCCCCCCCCC",
                                    "reactant_smiles": ["CCCC"] * 3,
                                    "stock_status": {"CCCC": True},
                                    "atom_mapped_reaction_smiles": (
                                        f"{reactant_mapping}>>{product_mapping}"
                                    ),
                                }
                            ],
                        }
                    ],
                },
                target_smiles="CCCCCCCCCCCC",
            )

            self.assertTrue(parent["accepted"])
            self.assertEqual(parent["verification_level"], "L2_mapping_consistent")
            self.assertFalse(parent["reaction_validated"])
            proof = compile_stitched_parent_route_proof(
                target_smiles="CCCCCCCCCCCC",
                parent_verifier=parent,
            )
        self.assertFalse(proof["accepted"])
        self.assertFalse(
            proof["proof_clauses"]["all_reaction_steps_precedent_supported"]
        )
        self.assertIn("parent_route_reaction_steps_not_validated", proof["reasons"])

    def test_bare_stitched_route_booleans_cannot_emit_solved_verdict(self):
        verdict = emit_final_verdict(
            {
                "case_id": "forged-stitch",
                "preflight": {"accepted": True},
                "artifacts": {
                    "stitched_semisynthesis_route": {
                        "accepted": True,
                        "solved": True,
                        "stock_audit_passed": True,
                    }
                },
            }
        )

        self.assertFalse(verdict.solved)
        self.assertNotEqual(verdict.route_status, "solved")

    def test_verified_parent_route_passes_direct_parent_proof(self):
        target = "CCO"
        proof = compile_stitched_parent_route_proof(
            target_smiles=target,
            parent_verifier=_strict_parent_verifier(target),
        )

        self.assertTrue(proof["accepted"], proof["reasons"])
        self.assertEqual(proof["proof_mode"], "direct_parent_route")
        self.assertTrue(proof["proof_clauses"]["direct_parent_route_verifier_accepted"])
        self.assertEqual(proof["route_status"], "solved")
        self.assertTrue(is_solved_parent_route_proof(proof))

    def test_accepted_parent_route_is_not_blocked_by_rejected_sibling_diagnostics(self):
        target = "CCO"
        proof = compile_stitched_parent_route_proof(
            target_smiles=target,
            parent_verifier=_strict_parent_verifier(
                target,
                precursor_smiles="CC",
                include_rejected_route=True,
            ),
        )

        self.assertTrue(proof["accepted"], proof["reasons"])
        self.assertEqual(proof["proof_mode"], "direct_parent_route")
        self.assertNotIn("element_inventory_not_conserved", proof["reasons"])

    def test_graph_and_stock_closed_parent_requires_reaction_step_proof(self):
        parent = _strict_parent_verifier("CCO")
        parent["accepted_route"]["steps"][0].pop("atom_mapped_reaction_smiles")
        parent = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [parent["accepted_route"]],
                "stock_catalog_context": parent["stock_catalog_audit"]["revalidation_context"],
            },
            target_smiles="CCO",
        )

        proof = compile_stitched_parent_route_proof(
            target_smiles="CCO",
            parent_verifier=parent,
        )

        self.assertFalse(proof["accepted"])
        self.assertEqual(proof["proof_mode"], "direct_parent_route")
        self.assertIn("parent_route_reaction_steps_not_validated", proof["reasons"])
        self.assertIn("reaction_step_proof_incomplete", proof["reasons"])
        self.assertFalse(proof["proof_clauses"]["all_reaction_steps_validated"])
        self.assertEqual(
            proof["proof_attempt"]["reaction_validation"]["proof_level"],
            "L1_graph_and_stock_closed",
        )

    def test_child_only_solved_does_not_pass_parent_proof(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            child_route={"accepted": True, "solved": True},
            exact_literature_segment={"accepted": True, "parent_route_connected": False},
        )

        self.assertFalse(proof["accepted"])
        self.assertIn("parent_route_verifier_not_accepted", proof["reasons"])
        self.assertIn("child_target_route_not_connected_to_parent_bridge", proof["reasons"])

    def test_unsolved_child_does_not_report_child_solved_parent_unresolved(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            parent_verifier={
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "target_match": True,
                "target_equivalence_audit": {"target_match": True},
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump"}],
            },
            child_route={"accepted": False, "solved": False, "accepted_subgoal_count": 0},
            exact_literature_segment={"accepted": False, "row_count": 0},
            stock_audit={"stock_audit_passed": False},
        )

        self.assertFalse(proof["accepted"])
        self.assertEqual(proof["route_status"], "fake_closed_rejected")
        self.assertNotEqual(proof["route_status"], "child_solved_parent_unresolved")
        self.assertIn("child_target_route_not_connected_to_parent_bridge", proof["reasons"])

    def test_disconnected_exact_row_does_not_pass(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCOC(C)=O",
            parent_verifier={
                "accepted": True,
                "route_status": "solved",
                "target_match": True,
                "target_equivalence_audit": {"target_match": True},
            },
            child_route={"accepted": True, "parent_bridge_connected": True},
            exact_literature_segment={"accepted": True, "parent_route_connected": False},
            stock_audit={"stock_audit_passed": True},
        )

        self.assertFalse(proof["accepted"])
        self.assertEqual(proof["route_status"], "partial_anchor_only_not_solved")
        self.assertIn("exact_literature_segment_not_connected_to_parent_route", proof["reasons"])

    def test_verifier_and_stitch_connectivity_passes(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CC=O",
            stitched_route=_accepted_stitch_fixture(),
            exact_literature_segment={"accepted": True, "parent_route_connected": True},
        )

        self.assertTrue(proof["accepted"], proof["reasons"])
        self.assertTrue(proof["proof_clauses"]["analogy_used_only_as_rationale"])
        self.assertEqual(proof["route_status"], "solved")

    def test_accepted_stitch_is_not_blocked_by_prior_failed_guided_route(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CC=O",
            parent_verifier={
                "accepted": False,
                "route_status": "fake_closed_rejected",
                "target_match": True,
                "reasons": ["large_atom_jump"],
                "failure_events": [{"reason": "large_atom_jump"}],
            },
            stitched_route=_accepted_stitch_fixture(),
            exact_literature_segment={"accepted": True, "parent_route_connected": True},
        )

        self.assertTrue(proof["accepted"], proof["reasons"])
        self.assertNotIn("unexplained_large_atom_jump", proof["reasons"])

    def test_self_reported_mechanistic_parent_bridge_cannot_close_stitch(self):
        literature_chain = {
            "schema_version": "advisory_visual_literature_chain_audit.v1",
            "accepted": True,
            "target_smiles": "CC1CCC(O)C=C2CCCC(=O)C12",
            "terminal_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
            "terminal_name": "compound 7",
            "source_ref": "doi:10.1000/mechanistic-bridge",
            "terminal_reached": True,
            "step_count": 1,
            "chain": [
                {
                    "reactant_smiles": ["CC(=O)CCC1C(=O)CCCC(=O)1"],
                    "product_smiles": "CC1CCC(O)C=C2CCCC(=O)C12",
                }
            ],
        }
        route_expansion = {
            "accepted": True,
            "solved": True,
            "subgoals": [
                {
                    "accepted": True,
                    "solved": True,
                    "subgoal": {
                        "name": "hydroxy precursor",
                        "smiles": "CC(O)CCC1C(=O)CCCC1=O",
                        "parent_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
                        "operation_idea": "oxidize hydroxy precursor to carbonyl terminal",
                        "variant_type": "carbonyl_to_hydroxy_precursor",
                        "risk_flags": ["same_core_transform_not_literature_exact"],
                        "parent_bridge_validation": {
                            "accepted": True,
                            "method": "forward_reconstruction",
                            "parent_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
                            "child_smiles": "CC(O)CCC1C(=O)CCCC1=O",
                        },
                    },
                    "verifier": {
                        "schema_version": "harness_route_verifier_report.v1",
                        "accepted": True,
                        "route_status": "solved",
                        "target_match": True,
                        "target_equivalence_audit": {
                            "target_match": True,
                            "request_target_smiles": "CC(O)CCC1C(=O)CCCC1=O",
                        },
                        "route_count": 1,
                        "accepted_route_count": 1,
                        "rejected_route_count": 0,
                        "best_route_rank": 1,
                        "best_route_step_count": 1,
                        "stock_audit_passed": True,
                        "reasons": [],
                        "warnings": [],
                    },
                    "raw_result": {
                        "target": "CC(O)CCC1C(=O)CCCC1=O",
                        "search_status": {"solved": True},
                        "routes": [
                            {
                                "route_rank": 1,
                                "stock_closed": True,
                                "metrics": {
                                    "terminal_reactants": ["CC(=O)CCC1C(=O)CCCC1=O"],
                                    "terminal_stock_status": {"CC(=O)CCC1C(=O)CCCC1=O": True},
                                },
                                "steps": [
                                    {
                                        "product_smiles": "CC(O)CCC1C(=O)CCCC1=O",
                                        "reactant_smiles": ["CC(=O)CCC1C(=O)CCCC1=O"],
                                        "stock_status": {"CC(=O)CCC1C(=O)CCCC1=O": True},
                                    }
                                ],
                            }
                        ],
                    },
                }
            ],
        }
        route_expansion["subgoals"][0]["verifier"] = verify_chemenzy_raw_routes(
            route_expansion["subgoals"][0]["raw_result"],
            target_smiles="CC(O)CCC1C(=O)CCCC1=O",
        )

        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit=literature_chain,
            route_expansion_result=route_expansion,
            subgoal_raw_result=route_expansion["subgoals"][0]["raw_result"],
            target_smiles="CC1CCC(O)C=C2CCCC(=O)C12",
        )
        proof = compile_stitched_parent_route_proof(
            target_smiles="CC1CCC(O)C=C2CCCC(=O)C12",
            stitched_route=stitch,
        )

        self.assertFalse(stitch["accepted"])
        self.assertFalse(stitch["terminal_match_audit"]["parent_bridge_accepted"])
        self.assertIn("literature_chain_not_strict_source_detail_schema", stitch["reasons"])
        self.assertEqual(stitch["combined_route"]["parent_bridge_step_count"], 0)
        self.assertFalse(proof["accepted"])

    def test_mechanistic_parent_bridge_does_not_solve_wrong_parent_target(self):
        literature_chain = {
            "schema_version": "advisory_visual_literature_chain_audit.v1",
            "accepted": True,
            "target_smiles": "CC1CCC(O)C=C2CCCC(=O)C12",
            "terminal_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
            "terminal_reached": True,
            "step_count": 1,
            "chain": [
                {
                    "reactant_smiles": ["CC(=O)CCC1C(=O)CCCC(=O)1"],
                    "product_smiles": "CC1CCC(O)C=C2CCCC(=O)C12",
                }
            ],
        }
        route_expansion = {
            "accepted": True,
            "solved": True,
            "subgoals": [
                {
                    "accepted": True,
                    "solved": True,
                    "subgoal": {
                        "smiles": "CC(O)CCC1C(=O)CCCC1=O",
                        "parent_smiles": "CC(=O)CCC1C(=O)CCCC(=O)1",
                        "operation_idea": "oxidize hydroxy precursor to carbonyl terminal",
                        "variant_type": "carbonyl_to_hydroxy_precursor",
                    },
                    "verifier": {
                        "accepted": True,
                        "route_status": "solved",
                        "target_match": True,
                        "target_equivalence_audit": {"request_target_smiles": "CC(O)CCC1C(=O)CCCC1=O"},
                    },
                }
            ],
        }

        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit=literature_chain,
            route_expansion_result=route_expansion,
            target_smiles="CCOC(C)=O",
        )

        self.assertFalse(stitch["accepted"])
        self.assertIn("target_input_literature_chain_mismatch", stitch["reasons"])

    def test_raw_stock_closed_flag_cannot_replace_verifier_stock_audit(self):
        literature_chain = {
            "accepted": True,
            "target_smiles": "CCO",
            "terminal_smiles": "CC",
            "terminal_reached": True,
            "step_count": 1,
            "chain": [{"reactant_smiles": ["CC"], "product_smiles": "CCO"}],
        }
        raw = {
            "search_status": {"solved": True},
            "routes": [
                {
                    "route_rank": 0,
                    "stock_closed": True,
                    "steps": [{"product_smiles": "CC", "reactant_smiles": ["C"]}],
                }
            ],
        }
        verifier = {
            "accepted": False,
            "route_status": "fake_closed_rejected",
            "target_match": True,
            "accepted_route_count": 0,
            "best_route_rank": 0,
        }

        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit=literature_chain,
            subgoal_verifier=verifier,
            subgoal_raw_result=raw,
            target_smiles="CCO",
        )

        self.assertFalse(stitch["accepted"])
        self.assertFalse(stitch["subgoal_closure"]["stock_audit_passed"])
        self.assertIn("subgoal_stock_audit_not_passed", stitch["reasons"])

    def test_stitch_rejects_historical_verifier_bound_to_different_route(self):
        raw = _materialized_subgoal_raw("CCO")
        verifier = copy.deepcopy(_strict_subgoal_verifier("CCO"))
        verifier["accepted_route"]["steps"][0]["product_smiles"] = "CCN"

        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit={
                "schema_version": "source_detail_route_chain_audit.v1",
                "accepted": True,
                "target_smiles": "CC=O",
                "terminal_smiles": "CCO",
                "terminal_reached": True,
                "source_ref": "doi:10.1000/revalidatable-stitch",
                "chain": [
                    _strict_literature_step(
                        step_id="ethanol_oxidation",
                        reactants=["CCO"],
                        product="CC=O",
                    )
                ],
            },
            subgoal_verifier=verifier,
            subgoal_raw_result=raw,
            target_smiles="CC=O",
        )

        self.assertFalse(stitch["accepted"])
        self.assertIn("subgoal_verifier_reverification_mismatch", stitch["reasons"])

    def test_bare_parent_proof_booleans_cannot_emit_solved_verdict(self):
        bare = {"accepted": True, "solved": True, "route_status": "solved"}

        self.assertFalse(is_solved_parent_route_proof(bare))
        verdict = emit_agentic_final_verdict(
            blackboard={"case_id": "bare-proof", "parent_route_proof": bare},
            artifacts={},
        )
        self.assertFalse(verdict.solved)
        self.assertNotEqual(verdict.route_status, "solved")

    def test_bare_parent_verifier_boolean_cannot_compile_solved_proof(self):
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCO",
            parent_verifier={"accepted": True, "target_match": True},
        )

        self.assertFalse(proof["accepted"])
        self.assertIn("parent_route_verifier_not_accepted", proof["reasons"])
        self.assertIn("stock_audit_not_passed", proof["reasons"])

    def test_verifier_solved_claim_with_blocking_reasons_is_not_authoritative(self):
        verifier = _strict_subgoal_verifier("CCO")
        verifier["reasons"] = ["large_atom_jump"]

        self.assertFalse(is_accepted_route_verifier_report(verifier))
        proof = compile_stitched_parent_route_proof(
            target_smiles="CCO",
            parent_verifier=verifier,
        )
        self.assertFalse(proof["accepted"])
        self.assertIn("parent_route_verifier_not_accepted", proof["reasons"])

    def test_route_verifier_rejects_empty_steps_despite_claimed_n_steps(self):
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "n_steps": 9,
                        "metrics": {"terminal_reactants": ["CC", "O"]},
                        "steps": [],
                    }
                ],
            },
            target_smiles="CCO",
        )

        self.assertFalse(verifier["accepted"])
        self.assertFalse(is_accepted_route_verifier_report(verifier))
        self.assertEqual(verifier["accepted_route_count"], 0)
        self.assertIn("missing_route_steps", verifier["reasons"])
        rejected = verifier["rejected_route_summary"][0]
        self.assertEqual(rejected["n_steps"], 0)
        self.assertEqual(rejected["claimed_n_steps"], 9)

        forged_summary = {
            "schema_version": "harness_route_verifier_report.v1",
            "accepted": True,
            "route_status": "solved",
            "target_match": True,
            "target_equivalence_audit": {"target_match": True},
            "route_count": 1,
            "accepted_route_count": 1,
            "rejected_route_count": 0,
            "best_route_rank": 0,
            "best_route_step_count": 0,
            "reasons": [],
            "warnings": [],
        }
        self.assertFalse(is_accepted_route_verifier_report(forged_summary))
        verdict = emit_final_verdict(
            {
                "case_id": "empty-route",
                "preflight": {"accepted": True},
                "artifacts": {"route_verifier": forged_summary},
            }
        )
        self.assertFalse(verdict.solved)
        self.assertNotEqual(verdict.route_status, "solved")

    def test_route_verifier_accepts_materialized_target_route(self):
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "n_steps": 99,
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
        self.assertEqual(verifier["accepted_route_count"], 1)
        self.assertTrue(is_accepted_route_verifier_report(verifier))

    def test_route_verifier_rejects_terminal_without_explicit_stock_evidence(self):
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {"terminal_reactants": ["C"]},
                        "steps": [{"main_reactant": "C", "product": "CCO"}],
                    }
                ],
            },
            target_smiles="CCO",
        )

        self.assertFalse(verifier["accepted"])
        self.assertIn("terminal_stock_status_unproven", verifier["reasons"])

    def test_stitch_rejects_empty_dict_steps_as_non_materialized(self):
        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit={
                "accepted": True,
                "target_smiles": "CCO",
                "terminal_smiles": "CC",
                "terminal_reached": True,
                "chain": [{}],
            },
            subgoal_verifier=_strict_subgoal_verifier("CC"),
            subgoal_raw_result={
                "search_status": {"solved": True},
                "routes": [{"route_rank": 0, "steps": [{}]}],
            },
            target_smiles="CCO",
        )

        self.assertFalse(stitch["accepted"])
        self.assertEqual(stitch["literature_chain"]["step_count"], 0)
        self.assertFalse(stitch["subgoal_closure"]["route_materialization_complete"])

    def test_stitch_rejects_empty_literature_chain_despite_claimed_step_count(self):
        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit={
                "accepted": True,
                "target_smiles": "CCO",
                "terminal_smiles": "CC",
                "terminal_reached": True,
                "step_count": 7,
                "chain": [],
            },
            subgoal_verifier=_strict_subgoal_verifier("CC"),
            subgoal_raw_result=_materialized_subgoal_raw("CC"),
            target_smiles="CCO",
        )

        self.assertFalse(stitch["accepted"])
        self.assertEqual(stitch["literature_chain"]["step_count"], 0)
        self.assertEqual(stitch["literature_chain"]["claimed_step_count"], 7)
        self.assertIn("literature_chain_materialized_steps_missing", stitch["reasons"])

    def test_stitch_rejects_fake_raw_n_steps_without_materialized_steps(self):
        raw = _materialized_subgoal_raw("CC")
        raw["routes"][0]["n_steps"] = 11
        raw["routes"][0]["steps"] = []
        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit={
                "accepted": True,
                "target_smiles": "CCO",
                "terminal_smiles": "CC",
                "terminal_reached": True,
                "step_count": 1,
                "chain": [{"reactant_smiles": ["CC"], "product_smiles": "CCO"}],
            },
            subgoal_verifier=_strict_subgoal_verifier("CC"),
            subgoal_raw_result=raw,
            target_smiles="CCO",
        )

        self.assertFalse(stitch["accepted"])
        self.assertEqual(stitch["subgoal_closure"]["best_route_step_count"], 0)
        self.assertIn("subgoal_materialized_route_missing", stitch["reasons"])

    def test_route_verifier_rejects_equal_size_element_transmutation(self):
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {"terminal_reactants": ["NNN"], "terminal_stock_status": {"NNN": True}},
                        "steps": [{"product": "CCO", "reactant_smiles": ["NNN"], "stock_status": {"NNN": True}}],
                    }
                ],
            },
            target_smiles="CCO",
        )

        self.assertFalse(verifier["accepted"])
        self.assertIn("element_inventory_not_conserved", verifier["reasons"])

    def test_route_verifier_rejects_duplicate_fragment_padding(self):
        target = "C" * 20
        verifier = verify_chemenzy_raw_routes(
            {
                "target": target,
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {"terminal_reactants": ["CC"], "terminal_stock_status": {"CC": True}},
                        "steps": [
                            {
                                "product": target,
                                "reactant_smiles": ["CC"] * 10,
                                "stock_status": {"CC": True},
                            }
                        ],
                    }
                ],
            },
            target_smiles=target,
        )

        self.assertFalse(verifier["accepted"])
        self.assertIn("large_atom_jump", verifier["reasons"])

    def test_route_verifier_rejects_route_cycle_back_to_target(self):
        verifier = verify_chemenzy_raw_routes(
            {
                "target": "CCO",
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {
                            "terminal_reactants": ["C", "O"],
                            "terminal_stock_status": {"C": True, "O": True},
                        },
                        "steps": [
                            {
                                "product": "CCO",
                                "reactant_smiles": ["CC", "C", "O"],
                                "stock_status": {"C": True, "O": True},
                            },
                            {"product": "CC", "reactant_smiles": ["CCO"]},
                        ],
                    }
                ],
            },
            target_smiles="CCO",
        )

        self.assertFalse(verifier["accepted"])
        self.assertIn("disconnected_route_steps", verifier["reasons"])

    def test_backend_stock_true_requires_independent_catalog_hit(self):
        target = "[13CH3]O"
        terminal = "[13CH4]"
        verifier = verify_chemenzy_raw_routes(
            {
                "target": target,
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {
                            "terminal_reactants": [terminal, "O"],
                            "terminal_stock_status": {terminal: True, "O": True},
                        },
                        "steps": [
                            {
                                "product": target,
                                "reactant_smiles": [terminal, "O"],
                                "stock_status": {terminal: True, "O": True},
                            }
                        ],
                    }
                ],
            },
            target_smiles=target,
        )

        self.assertFalse(verifier["accepted"])
        self.assertIn("terminal_stock_status_unproven", verifier["reasons"])

    def test_made_up_or_advisory_literature_schema_cannot_solve_stitch(self):
        for schema in ("made_up_schema.v999", "advisory_visual_literature_chain_audit.v1"):
            stitch = compile_stitched_semisynthesis_route(
                literature_chain_audit={
                    "schema_version": schema,
                    "accepted": True,
                    "target_smiles": "CCO",
                    "terminal_smiles": "CC",
                    "terminal_reached": True,
                    "source_ref": "doi:10.1000/invented",
                    "chain": [{"product_smiles": "CCO", "reactant_smiles": ["CC"]}],
                },
                subgoal_verifier=_strict_subgoal_verifier("CC"),
                subgoal_raw_result=_materialized_subgoal_raw("CC"),
                target_smiles="CCO",
            )

            self.assertFalse(stitch["accepted"])
            self.assertIn("literature_chain_not_strict_source_detail_schema", stitch["reasons"])

    def test_stitch_requires_every_literature_frontier_precursor_closed(self):
        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit={
                "schema_version": "source_detail_route_chain_audit.v1",
                "accepted": True,
                "target_smiles": "CCO",
                "terminal_smiles": "CC",
                "terminal_reached": True,
                "source_ref": "doi:10.1000/revalidatable-stitch",
                "chain": [
                    _strict_literature_step(
                        step_id="unclosed_coreactant",
                        reactants=["CC", "CCCCCCCCCCCCCCC"],
                        product="CCO",
                    )
                ],
            },
            subgoal_verifier=_strict_subgoal_verifier("CC"),
            subgoal_raw_result=_materialized_subgoal_raw("CC"),
            target_smiles="CCO",
        )

        self.assertFalse(stitch["accepted"])
        self.assertIn("literature_chain_has_unclosed_precursors", stitch["reasons"])

    def test_multi_frontier_stitch_requires_and_accepts_every_verified_closure(self):
        oxygen_raw = {
            "target": "O",
            "search_status": {"solved": True},
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": ["O=O"],
                        "terminal_stock_status": {"O=O": True},
                    },
                    "steps": [
                        {
                            **_strict_literature_step(
                                step_id="oxygen_reduction",
                                reactants=["O=O"],
                                product="O",
                            ),
                            "stock_status": {"O=O": True},
                        }
                    ],
                }
            ],
        }
        ethanol_raw = _materialized_subgoal_raw("CCO")
        expansion = {
            "subgoals": [
                {
                    "accepted": True,
                    "subgoal": {"name": "ethanol frontier", "smiles": "CCO"},
                    "verifier": verify_chemenzy_raw_routes(ethanol_raw, target_smiles="CCO"),
                    "raw_result": ethanol_raw,
                },
                {
                    "accepted": True,
                    "subgoal": {"name": "water frontier", "smiles": "O"},
                    "verifier": verify_chemenzy_raw_routes(oxygen_raw, target_smiles="O"),
                    "raw_result": oxygen_raw,
                },
            ]
        }
        literature = {
            "schema_version": "source_detail_route_chain_audit.v1",
            "accepted": True,
            "target_smiles": "CCOO",
            "terminal_smiles": "CCO",
            "terminal_reached": True,
            "source_ref": "doi:10.1000/revalidatable-stitch",
            "chain": [
                _strict_literature_step(
                    step_id="multi_frontier",
                    reactants=["CCO", "O"],
                    product="CCOO",
                )
            ],
        }

        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit=literature,
            route_expansion_result=expansion,
            target_smiles="CCOO",
        )

        self.assertTrue(stitch["accepted"], stitch["reasons"])
        self.assertEqual(stitch["frontier_coverage_audit"]["frontier_count"], 2)
        self.assertEqual(stitch["frontier_coverage_audit"]["closed_frontier_count"], 2)
        self.assertEqual(
            stitch["frontier_coverage_audit"]["precedent_supported_frontier_count"],
            2,
        )
        self.assertTrue(
            stitch["frontier_coverage_audit"]["all_frontiers_precedent_supported"]
        )
        self.assertEqual(len(stitch["subgoal_closures"]), 2)
        self.assertTrue(is_solved_stitched_semisynthesis_route(stitch, expected_target_smiles="CCOO"))
        proof = compile_stitched_parent_route_proof(target_smiles="CCOO", stitched_route=stitch)
        self.assertTrue(proof["proof_clauses"]["all_reaction_steps_validated"])
        self.assertTrue(
            proof["proof_clauses"]["all_reaction_steps_precedent_supported"]
        )
        self.assertTrue(is_solved_parent_route_proof(proof, expected_target_smiles="CCOO"))

        missing_one = compile_stitched_semisynthesis_route(
            literature_chain_audit=literature,
            route_expansion_result={"subgoals": expansion["subgoals"][:1]},
            target_smiles="CCOO",
        )
        self.assertFalse(missing_one["accepted"])
        self.assertIn("literature_chain_has_unclosed_precursors", missing_one["reasons"])

    def test_l2_subgoal_stitch_remains_displayable_but_cannot_solve_parent(self):
        oxygen_raw = {
            "target": "O",
            "search_status": {"solved": True},
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": ["O=O"],
                        "terminal_stock_status": {"O=O": True},
                    },
                    "steps": [
                        {
                            **_strict_literature_step(
                                step_id="oxygen_reduction",
                                reactants=["O=O"],
                                product="O",
                            ),
                            "stock_status": {"O=O": True},
                        }
                    ],
                }
            ],
        }
        literature = {
            "schema_version": "source_detail_route_chain_audit.v1",
            "accepted": True,
            "target_smiles": "CCOO",
            "terminal_smiles": "CCO",
            "terminal_reached": True,
            "source_ref": "doi:10.1000/revalidatable-stitch",
            "chain": [
                _strict_literature_step(
                    step_id="multi_frontier",
                    reactants=["CCO", "O"],
                    product="CCOO",
                )
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "acetaldehyde.csv"
            catalog.write_text("CC=O\n", encoding="utf-8")
            ethanol_l2_raw = {
                "target": "CCO",
                "search_status": {"solved": True},
                "stock_catalog_context": {
                    "effective_stock_names": ["fixture_acetaldehyde"],
                    "catalog_bindings": [
                        {
                            "name": "fixture_acetaldehyde",
                            "path": str(catalog),
                            "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                        }
                    ],
                },
                "routes": [
                    {
                        "route_rank": 0,
                        "metrics": {
                            "terminal_reactants": ["CC=O"],
                            "terminal_stock_status": {"CC=O": True},
                        },
                        "steps": [
                            {
                                "product": "CCO",
                                "reactant_smiles": ["CC=O"],
                                "stock_status": {"CC=O": True},
                                "atom_mapped_reaction_smiles": (
                                    "[CH3:1][CH:2]=[O:3]>>"
                                    "[CH3:1][CH2:2][OH:3]"
                                ),
                            }
                        ],
                    }
                ],
            }
            ethanol_l2_verifier = verify_chemenzy_raw_routes(
                ethanol_l2_raw,
                target_smiles="CCO",
            )
            expansion = {
                "subgoals": [
                    {
                        "accepted": True,
                        "subgoal": {"name": "ethanol frontier", "smiles": "CCO"},
                        "verifier": ethanol_l2_verifier,
                        "raw_result": ethanol_l2_raw,
                    },
                    {
                        "accepted": True,
                        "subgoal": {"name": "water frontier", "smiles": "O"},
                        "verifier": verify_chemenzy_raw_routes(
                            oxygen_raw,
                            target_smiles="O",
                        ),
                        "raw_result": oxygen_raw,
                    },
                ]
            }

            stitch = compile_stitched_semisynthesis_route(
                literature_chain_audit=literature,
                route_expansion_result=expansion,
                target_smiles="CCOO",
            )

            self.assertTrue(stitch["accepted"], stitch["reasons"])
            self.assertFalse(stitch["solved"])
            self.assertEqual(stitch["route_status"], "reaction_validated_l2_candidate")
            self.assertTrue(stitch["stock_audit_passed"])
            self.assertEqual(
                stitch["frontier_coverage_audit"]["closed_frontier_count"],
                2,
            )
            self.assertEqual(
                stitch["frontier_coverage_audit"][
                    "precedent_supported_frontier_count"
                ],
                1,
            )
            self.assertFalse(
                stitch["frontier_coverage_audit"][
                    "all_frontiers_precedent_supported"
                ]
            )
            self.assertTrue(
                is_reaction_validated_stitched_semisynthesis_route(
                    stitch,
                    expected_target_smiles="CCOO",
                )
            )
            self.assertFalse(
                is_solved_stitched_semisynthesis_route(
                    stitch,
                    expected_target_smiles="CCOO",
                )
            )

            proof = compile_stitched_parent_route_proof(
                target_smiles="CCOO",
                stitched_route=stitch,
            )
            self.assertFalse(proof["accepted"])
            self.assertFalse(proof["solved"])
            self.assertTrue(proof["proof_clauses"]["all_reaction_steps_validated"])
            self.assertFalse(
                proof["proof_clauses"]["all_reaction_steps_precedent_supported"]
            )
            self.assertIn("reaction_step_precedent_incomplete", proof["reasons"])
            self.assertIn(
                "reaction_step_precedent_incomplete",
                proof["proof_attempt"]["missing_requirements"],
            )
            self.assertEqual(proof["proof_evidence"]["stitched_route"], {})
            self.assertEqual(proof["proof_evidence"]["stitched_route_attempt"], stitch)
            self.assertFalse(
                is_solved_parent_route_proof(
                    proof,
                    expected_target_smiles="CCOO",
                )
            )

            forged_stitch = copy.deepcopy(stitch)
            forged_stitch["solved"] = True
            forged_stitch["route_status"] = "solved"
            forged_stitch["frontier_coverage_audit"][
                "precedent_supported_frontier_count"
            ] = 2
            forged_stitch["frontier_coverage_audit"][
                "all_frontiers_precedent_supported"
            ] = True
            for closure in forged_stitch["subgoal_closures"]:
                closure["precedent_supported"] = True
            self.assertFalse(
                is_solved_stitched_semisynthesis_route(
                    forged_stitch,
                    expected_target_smiles="CCOO",
                )
            )

            forged_parent = copy.deepcopy(proof)
            forged_parent["accepted"] = True
            forged_parent["solved"] = True
            forged_parent["route_status"] = "solved"
            forged_parent["reasons"] = []
            for clause in forged_parent["proof_clauses"]:
                forged_parent["proof_clauses"][clause] = True
            forged_parent["proof_attempt"]["accepted"] = True
            forged_parent["proof_attempt"]["missing_requirements"] = []
            forged_parent["proof_attempt"][
                "reaction_steps_precedent_supported"
            ] = True
            forged_parent["proof_evidence"]["stitched_route"] = forged_stitch
            self.assertFalse(
                is_solved_parent_route_proof(
                    forged_parent,
                    expected_target_smiles="CCOO",
                )
            )

    def test_pdf_manifest_cannot_validate_unregistered_step_chemistry(self):
        stitch = compile_stitched_semisynthesis_route(
            literature_chain_audit={
                "schema_version": "source_detail_route_chain_audit.v1",
                "accepted": True,
                "target_smiles": "CCN",
                "terminal_smiles": "CC",
                "terminal_reached": True,
                "source_ref": "doi:10.1000/revalidatable-stitch",
                "chain": [
                    _strict_literature_step(
                        step_id="invented_chemistry",
                        reactants=["CC"],
                        product="CCN",
                    )
                ],
            },
            subgoal_verifier=_strict_subgoal_verifier("CC"),
            subgoal_raw_result=_materialized_subgoal_raw("CC"),
            target_smiles="CCN",
        )

        self.assertFalse(stitch["accepted"])
        self.assertIn("literature_chain_step_provenance_not_revalidated", stitch["reasons"])


if __name__ == "__main__":
    unittest.main()
