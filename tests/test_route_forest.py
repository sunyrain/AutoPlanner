from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.harness.route_forest import (
    SCHEMA_VERSION,
    _branch_title_for_display,
    _chain_rows_from_source_detail_payload,
    _display_text_is_corrupt,
    _looks_like_smiles,
    _module_key_for_text,
    _module_label_for_key,
    _route_by_verified_rank,
    compile_explored_route_forest,
    render_route_forest_html,
    write_route_forest_artifacts,
)
from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.harness.stitched_route import compile_stitched_semisynthesis_route


_SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "source_evidence_stub.pdf"
_SOURCE_PAGE_FIXTURE = Path(__file__).parent / "fixtures" / "source_page.ppm"
_SOURCE_MANIFEST_FIXTURE = Path(__file__).parent / "fixtures" / "source_evidence_manifest.json"
_TRUSTED_REGISTRY_FIXTURE = Path(__file__).parent / "fixtures" / "trusted_literature_step_registry.json"


def _sample_paclitaxel_blackboard() -> dict:
    return {
        "case_id": "paclitaxel_route_forest_test",
        "target_profile": {
            "target_name": "paclitaxel",
            "target_smiles": "CC(=O)OC1C(C2(C(C3C(C(C4(C(=C(C(=O)C3(C(OC(=O)C5=CC=CC=C5)C(OC(=O)C)C2O)O)C)C)CO4)OC(=O)C6=CC=CC=C6)O1)C)O",
        },
        "literature_evidence": {
            "source_candidates": [
                {
                    "title": "Baloglu and Kingston paclitaxel analog semisynthesis",
                    "doi": "10.1021/np990040k",
                },
                {
                    "title": "Holton taxol total synthesis",
                    "doi": "10.1021/ja00083a066",
                },
            ],
            "visual_chains": [
                {
                    "source_ref": "doi:10.1021/np990040k",
                    "source_title": "Baloglu/Kingston Scheme 1",
                    "steps": [
                        {
                            "step_id": "scheme_1_step_1",
                            "reaction_class": "side-chain donor preparation",
                            "reactant_labels": ["beta-lactam side-chain precursor"],
                            "product_label": "protected phenylisoserine donor",
                            "confidence": "low",
                            "source_locator": "Scheme 1",
                            "condition_candidate": {
                                "schema_version": "condition_candidate.v1",
                                "condition_status": "evidence_backed",
                                "reagent": "DCC, DMAP",
                                "solvent": "toluene",
                            },
                            "risk_flags": ["visual extraction; stereochemistry partial"],
                        }
                    ],
                }
            ],
        },
        "retrosynthetic_proposals": [
            {
                "proposal_id": "proposal_ester_coupling",
                "proposal_label": "C13 ester coupling disconnection",
                "proposal_type": "side-chain disconnection",
                "precursor_smiles": "CC(=O)O.C1CCOC1",
                "confidence": "medium",
                "score": 0.7,
                "evidence_refs": ["doi:10.1021/np990040k"],
                "risk_flags": ["requires literature validation"],
                "executable": True,
            }
        ],
        "broad_transform_templates": [
            {
                "template_id": "template_c13_sidechain",
                "objective_type": "sidechain installation",
                "transform_logic": "install C13 side chain from protected baccatin core",
                "preserved_scaffold": "protected baccatin core",
                "reaction_center": "C13 alcohol",
                "risk_flags": ["not exact literature row"],
            }
        ],
    }


def _sample_atorvastatin_blackboard() -> dict:
    return {
        "case_id": "atorvastatin_online_zero_test",
        "target_profile": {
            "target_name": "atorvastatin",
            "target_smiles": (
                "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
                "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
            ),
            "family_hint": "statin atorvastatin free acid",
        },
        "literature_evidence": {
            "source_candidates": [
                {
                    "source_ref": "doi:10.1186/s13065-015-0082-7",
                    "title": "An improved kilogram-scale preparation of atorvastatin calcium",
                    "doi": "10.1186/s13065-015-0082-7",
                    "local_pdf": "evidence/local_pdf_proxy/pdfs/atorvastatin_bmc.pdf",
                }
            ],
        },
        "retrosynthetic_proposals": [],
        "broad_transform_templates": [],
    }


def _solved_parent_proof(*, route: dict | None = None, target_smiles: str = "CCO") -> dict:
    del route
    if target_smiles == "CC=O":
        reactants = ["CCO"]
        mapped_reaction = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
        step_id = "ethanol_oxidation"
    else:
        reactants = ["CC", "O"]
        mapped_reaction = "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
        step_id = "common_stock_to_ethanol"
    stock_status = {value: True for value in reactants}
    strict_step = _strict_literature_step(
        step_id=step_id,
        reactants=reactants,
        product=target_smiles,
        atom_mapped_reaction_smiles=mapped_reaction,
    )
    strict_step.update(
        {
            "stock_status": stock_status,
            "reaction_type": "test materialized route",
        }
    )
    raw = {
        "target": target_smiles,
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": reactants,
                    "terminal_stock_status": stock_status,
                },
                "steps": [strict_step],
            }
        ],
    }
    verifier = verify_chemenzy_raw_routes(raw, target_smiles=target_smiles)
    return compile_stitched_parent_route_proof(
        target_smiles=target_smiles,
        target_name="test target",
        parent_verifier=verifier,
    )


def _strict_literature_step(
    *,
    step_id: str,
    reactants: list[str],
    product: str,
    atom_mapped_reaction_smiles: str = "",
) -> dict:
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
    if atom_mapped_reaction_smiles:
        row["atom_mapped_reaction_smiles"] = atom_mapped_reaction_smiles
    return row


def _stitched_parent_proof() -> dict:
    terminal = "CCO"
    target = "CC=O"
    raw = {
        "target": terminal,
        "search_status": {"solved": True},
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                "steps": [
                    {
                        **_strict_literature_step(
                            step_id="common_stock_to_ethanol",
                            reactants=["CC", "O"],
                            product=terminal,
                            atom_mapped_reaction_smiles=(
                                "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                            ),
                        ),
                        "stock_status": {"CC": True, "O": True},
                        "reaction_type": "verified hydration",
                    }
                ],
            }
        ],
    }
    verifier = verify_chemenzy_raw_routes(raw, target_smiles=terminal)
    stitch = compile_stitched_semisynthesis_route(
        literature_chain_audit={
            "schema_version": "source_detail_route_chain_audit.v1",
            "accepted": True,
            "target_smiles": target,
            "terminal_smiles": terminal,
            "terminal_reached": True,
            "source_ref": "doi:10.1000/revalidatable-stitch",
            "chain": [
                _strict_literature_step(
                    step_id="ethanol_oxidation",
                    reactants=[terminal],
                    product=target,
                    atom_mapped_reaction_smiles=(
                        "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
                    ),
                )
            ],
        },
        route_expansion_result={
            "subgoals": [
                {
                    "accepted": True,
                    "solved": True,
                    "subgoal": {"name": "ethanol", "smiles": terminal},
                    "verifier": verifier,
                }
            ]
        },
        subgoal_verifier=verifier,
        subgoal_raw_result=raw,
        target_smiles=target,
        target_name="acetaldehyde",
        case_id="stitched-route-forest-test",
    )
    return compile_stitched_parent_route_proof(
        target_smiles=target,
        target_name="acetaldehyde",
        case_id="stitched-route-forest-test",
        stitched_route=stitch,
    )


def _multi_frontier_stitched_parent_proof() -> dict:
    target = "CCOO"
    ethanol_raw = {
        "target": "CCO",
        "search_status": {"solved": True},
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                "steps": [
                    {
                        **_strict_literature_step(
                            step_id="common_stock_to_ethanol",
                            reactants=["CC", "O"],
                            product="CCO",
                            atom_mapped_reaction_smiles=(
                                "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                            ),
                        ),
                        "stock_status": {"CC": True, "O": True},
                    }
                ],
            }
        ],
    }
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
                            step_id="oxygen_to_water",
                            reactants=["O=O"],
                            product="O",
                            atom_mapped_reaction_smiles="[O:1]=[O:2]>>[OH2:1]",
                        ),
                        "stock_status": {"O=O": True},
                    }
                ],
            }
        ],
    }
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
    stitch = compile_stitched_semisynthesis_route(
        literature_chain_audit={
            "schema_version": "source_detail_route_chain_audit.v1",
            "accepted": True,
            "target_smiles": target,
            "terminal_smiles": "CCO",
            "terminal_reached": True,
            "source_ref": "doi:10.1000/revalidatable-stitch",
            "chain": [
                _strict_literature_step(
                    step_id="multi_frontier",
                    reactants=["CCO", "O"],
                    product=target,
                    atom_mapped_reaction_smiles=(
                        "[CH3:1][CH2:2][OH:3].[OH2:4]>>[CH3:1][CH2:2][O:3][OH:4]"
                    ),
                )
            ],
        },
        route_expansion_result=expansion,
        target_smiles=target,
        target_name="ethyl hydroperoxide",
        case_id="multi-frontier-route-forest-test",
    )
    return compile_stitched_parent_route_proof(
        target_smiles=target,
        target_name="ethyl hydroperoxide",
        case_id="multi-frontier-route-forest-test",
        stitched_route=stitch,
    )


def _sample_route_consensus_blackboard() -> dict:
    consensus = {
        "schema_version": "route_consensus.v1",
        "case_id": "ethanol_consensus",
        "target_smiles": "CCO",
        "accepted": True,
        "source_summary": {
            "candidate_count": 4,
            "rejected_count": 1,
            "proposal_count": 2,
            "channel_counts": {
                "codex_strategy": 1,
                "codex_critic": 1,
                "literature_exact": 1,
                "codex_chemoenzymatic": 1,
            },
        },
        "proposals": [
            {
                "schema_version": "route_consensus_proposal.v1",
                "consensus_id": "consensus:aldehyde-reduction",
                "rank": 1,
                "rank_score": 0.78,
                "product_smiles": "CCO",
                "precursor_smiles": ["CC=O"],
                "reaction_family": "carbonyl reduction",
                "reaction_families": ["carbonyl reduction", "biocatalytic reduction"],
                "source_channels": ["codex_strategy", "codex_critic", "literature_exact"],
                # Deliberately malformed producer groups exercise the display
                # boundary: Codex roles must still collapse to codex_model.
                "independent_support_groups": [
                    "codex_strategy",
                    "codex_critic",
                    "literature:doi:10.1000/example",
                ],
                "support_count": 3,
                "source_records": [
                    {
                        "candidate_id": "strategy",
                        "source_channel": "codex_strategy",
                        "evidence_level": "model_only",
                        "confidence": "medium",
                        "support_group": "codex_strategy",
                        "source_refs": [],
                        "evidence_refs": [],
                    },
                    {
                        "candidate_id": "critic",
                        "source_channel": "codex_critic",
                        "evidence_level": "model_only",
                        "confidence": "medium",
                        "support_group": "codex_critic",
                        "source_refs": [],
                        "evidence_refs": [],
                    },
                    {
                        "candidate_id": "literature",
                        "source_channel": "literature_exact",
                        "evidence_level": "literature_exact",
                        "confidence": "high",
                        "support_group": "literature:doi:10.1000/example",
                        "source_refs": ["doi:10.1000/example"],
                        "evidence_refs": ["source:scheme-2"],
                    },
                ],
                "source_refs": ["doi:10.1000/example"],
                "evidence_refs": ["source:scheme-2"],
                "evidence_level": "literature_exact",
                "confidence": "medium_high",
                "status": "evidence_backed_draft",
                "conditions": ["aqueous alcohol"],
                "catalysts": ["NaBH4", "ADH"],
                "condition_support": [
                    {
                        "candidate_id": "literature",
                        "support_group": "literature:doi:10.1000/example",
                        "conditions": ["MeOH, 0 C"],
                        "catalyst": "NaBH4",
                        "source_refs": ["doi:10.1000/example"],
                    }
                ],
                "condition_conflicts": [
                    {
                        "field": "catalyst",
                        "values": ["ADH", "NaBH4"],
                        "requires_review": True,
                    }
                ],
                "limitations": ["selectivity requires review"],
                "required_validation": ["forward_reconstruction"],
                "not_parent_route_proof": True,
                "no_solved_claim": True,
            },
            {
                "schema_version": "route_consensus_proposal.v1",
                "consensus_id": "consensus:hydration",
                "rank": 2,
                "rank_score": 0.35,
                "product_smiles": "CCO",
                "precursor_smiles": ["C=C"],
                "reaction_family": "alkene hydration",
                "reaction_families": ["alkene hydration"],
                "source_channels": ["codex_chemoenzymatic"],
                "independent_support_groups": ["codex_chemoenzymatic"],
                "support_count": 1,
                "source_records": [
                    {
                        "candidate_id": "enzyme",
                        "source_channel": "codex_chemoenzymatic",
                        "evidence_level": "model_only",
                        "confidence": "low",
                        "support_group": "codex_chemoenzymatic",
                        "source_refs": [],
                        "evidence_refs": [],
                    }
                ],
                "source_refs": [],
                "evidence_refs": [],
                "evidence_level": "model_only",
                "confidence": "low",
                "status": "model_hypothesis",
                "condition_support": [],
                "condition_conflicts": [],
                "limitations": ["model hypothesis"],
                "required_validation": ["reaction feasibility"],
                "not_parent_route_proof": True,
                "no_solved_claim": True,
            },
        ],
        "rejected_candidates": [{"candidate_id": "wrong-target", "reasons": ["target_mismatch"]}],
        "semantics": {
            "advisory_only": True,
            "deterministic_parent_proof_required": True,
            "no_solved_claim": True,
        },
    }
    return {
        "case_id": "ethanol_consensus",
        "target_profile": {"target_name": "ethanol", "target_smiles": "OCC"},
        "codex_agent_team": {
            "schema_version": "codex_retrosynthesis_team.v1",
            "accepted": True,
            "route_consensus_ref": "D:/runs/ethanol/route_consensus.json",
            "route_consensus": consensus,
        },
        # The adapter also places these records on the legacy bus. They should
        # not create duplicate generic proposal branches when consensus exists.
        "retrosynthetic_proposals": [
            {
                "proposal_id": "consensus:consensus:aldehyde-reduction",
                "source_type": "multi_source_consensus",
                "proposal_label": "carbonyl reduction",
                "precursor_smiles": "CC=O",
                "confidence": "medium_high",
                "executable": False,
            }
        ],
        "artifact_refs": {
            "route_consensus": "D:/runs/ethanol/route_consensus.json",
        },
    }


def test_route_forest_projects_canonical_consensus_without_evidence_laundering() -> None:
    forest = compile_explored_route_forest(_sample_route_consensus_blackboard())

    consensus = forest["route_consensus"]
    assert consensus["source_schema_version"] == "route_consensus.v1"
    assert consensus["available"] is True
    assert consensus["accepted_as_route"] is False
    assert consensus["semantics"]["advisory_only"] is True
    assert consensus["semantics"]["solved"] is False
    assert consensus["semantics"]["executable"] is False
    assert forest["counts"]["route_consensus_proposals"] == 2
    assert forest["counts"]["route_consensus_rejected_candidates"] == 1

    consensus_branches = [row for row in forest["branches"] if row["kind"] == "route_consensus"]
    assert len(consensus_branches) == 2
    assert not any(row["kind"] == "retrosynthetic_proposal" for row in forest["branches"])
    first = next(row for row in consensus_branches if row["rank"] == 1)
    assert first["source_channels"] == ["codex_strategy", "codex_critic", "literature_exact"]
    assert first["independent_support_groups"] == [
        "codex_model",
        "literature:doi:10.1000/example",
    ]
    assert first["independent_source_count"] == 2
    assert first["codex_roles_correlated"] is True
    assert first["advisory_only"] is True
    assert first["solved"] is False
    assert first["executable"] is False
    assert first["not_parent_route_proof"] is True
    second = next(row for row in consensus_branches if row["rank"] == 2)
    assert second["source_channels"] == ["codex_chemoenzymatic"]
    assert second["independent_support_groups"] == ["codex_model"]
    assert second["independent_source_count"] == 1


def test_route_forest_does_not_display_consensus_from_rejected_team() -> None:
    blackboard = _sample_route_consensus_blackboard()
    blackboard["codex_agent_team"]["accepted"] = False
    blackboard["route_consensus"] = dict(blackboard["codex_agent_team"]["route_consensus"])

    forest = compile_explored_route_forest(blackboard)

    assert forest["route_consensus"]["available"] is False
    assert forest["route_consensus"]["quarantined"] is True
    assert "codex_agent_team_not_accepted" in forest["route_consensus"]["reasons"]
    assert not any(row["kind"] == "route_consensus" for row in forest["branches"])


def test_route_forest_quarantines_consensus_when_worker_validation_failed() -> None:
    blackboard = _sample_route_consensus_blackboard()
    blackboard["codex_agent_team"]["artifact_validation"] = {
        "accepted": False,
        "reasons": ["invalid_worker_artifact"],
    }
    blackboard["route_consensus"] = dict(blackboard["codex_agent_team"]["route_consensus"])

    forest = compile_explored_route_forest(blackboard)

    assert forest["route_consensus"]["available"] is False
    assert forest["route_consensus"]["quarantined"] is True
    assert "codex_agent_team_artifact_validation_failed" in forest["route_consensus"]["reasons"]
    assert not any(row["kind"] in {"route_consensus", "retrosynthetic_proposal"} for row in forest["branches"])


def test_route_consensus_keeps_step_refs_separate_and_surfaces_conflicts_in_html() -> None:
    forest = compile_explored_route_forest(_sample_route_consensus_blackboard())
    first = next(row for row in forest["branches"] if row.get("rank") == 1)
    step = next(row for row in forest["steps"] if row["step_id"] in first["step_ids"])

    assert step["origin"] == "route_consensus"
    assert step["source_refs"] == ["doi:10.1000/example", "source:scheme-2"]
    assert "D:/runs/ethanol/route_consensus.json" not in step["source_refs"]
    assert first["route_level_source_refs"] == ["D:/runs/ethanol/route_consensus.json"]
    assert step["condition_status"] == "conflicting"
    assert {row["field"] for row in step["conflicts"]} == {"catalyst", "reaction_family"}
    assert len(step["support_records"]) == 3
    assert all("report_ref" not in row for row in step["support_records"])

    html = render_route_forest_html(forest)
    assert "Multi-source consensus audit" in html
    assert "Independent support groups" in html
    assert "Condition conflicts" in html
    assert "Codex roles are correlated" in html
    assert 'data-detail-tab="evidence"' in html
    assert "function renderEvidence(entity, host)" in html
    assert "support_records" in html


def test_route_forest_projects_blackboard_into_final_route_display() -> None:
    forest = compile_explored_route_forest(_sample_paclitaxel_blackboard(), run_dir="run/paclitaxel")

    assert forest["schema_version"] == SCHEMA_VERSION
    assert forest["target"]["name"] == "paclitaxel"
    assert forest["counts"]["branches"] == 3
    assert forest["run_trace"]["literature_counts"]["source_candidates"] == 2
    assert forest["evidence_index"]["visual_chains"][0]["step_count"] == 1
    assert {row["kind"] for row in forest["branches"]} == {
        "visual_chain",
        "retrosynthetic_proposal",
        "broad_template",
    }
    assert forest["counts"]["synthesis_classes"] == {"unspecified": 3}
    assert forest["primary_branch_id"] in {row["branch_id"] for row in forest["branches"]}
    assert all(row["advisory_only"] is True for row in forest["branches"])

    node_labels = [row["label"] for row in forest["nodes"]]
    assert "beta-lactam side-chain precursor" in node_labels
    assert "unbound proposal product" in node_labels
    assert "unbound template product: sidechain installation" in node_labels
    assert "side-chain donor preparation" in [row["label"] for row in forest["steps"]]


def test_advisory_proposals_and_templates_keep_their_own_products() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "unresolved_parent",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "retrosynthetic_proposals": [
                {
                    "proposal_id": "child_intermediate",
                    "proposal_label": "child oxidation",
                    "precursor_smiles": "CC",
                    "target_smiles": "CC=O",
                    "confidence": "medium",
                    "evidence_refs": ["source:child"],
                }
            ],
            "broad_transform_templates": [
                {
                    "template_id": "internal_labels",
                    "objective_type": "internal conversion",
                    "transform_logic": "compound 6 -> compound 7",
                    "source_refs": ["source:scheme"],
                }
            ],
        }
    )
    nodes = {row["node_id"]: row for row in forest["nodes"]}
    proposal = next(
        row for row in forest["branches"] if row["kind"] == "retrosynthetic_proposal"
    )
    proposal_step = next(
        row for row in forest["steps"] if row["step_id"] in proposal["step_ids"]
    )
    assert {nodes[node_id]["canonical_isomeric_smiles"] for node_id in proposal_step["to_node_ids"]} == {
        "CC=O"
    }

    template = next(row for row in forest["branches"] if row["kind"] == "broad_template")
    template_step = next(
        row for row in forest["steps"] if row["step_id"] in template["step_ids"]
    )
    assert {nodes[node_id]["label"] for node_id in template_step["from_node_ids"]} == {
        "compound 6"
    }
    assert {nodes[node_id]["label"] for node_id in template_step["to_node_ids"]} == {
        "compound 7"
    }
    assert all(nodes[node_id]["role"] != "target" for node_id in template_step["to_node_ids"])

    html = render_route_forest_html(forest)
    assert "父路线未闭合" in html
    assert "探索建议" in html
    assert "array_adjacency\":\"never_creates_an_edge" in html


def test_route_forest_does_not_invent_atorvastatin_process_route_from_source_metadata() -> None:
    forest = compile_explored_route_forest(_sample_atorvastatin_blackboard(), run_dir="run/atorvastatin")

    assert forest["target"]["name"] == "atorvastatin"
    assert [row["kind"] for row in forest["branches"]] == ["diagnostic_failure"]
    haystack = json.dumps(forest, ensure_ascii=False)
    assert "Paal-Knorr" not in haystack
    assert "advanced ketal ester intermediate 4" not in haystack


def test_route_forest_does_not_integrate_source_metadata_with_verified_route() -> None:
    blackboard = _sample_atorvastatin_blackboard()
    blackboard["target_profile"]["target_name"] = "test target"
    blackboard["target_profile"]["target_smiles"] = "CCO"
    blackboard["parent_route_proof"] = _solved_parent_proof(target_smiles="CCO")

    forest = compile_explored_route_forest(blackboard, run_dir="run/atorvastatin")

    assert [row["kind"] for row in forest["branches"]] == ["direct_verified_route"]
    assert forest["relationships"] == []


def test_stitched_route_ignores_loose_top_level_route_injection(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    proof = _stitched_parent_proof()
    injected_route = {
        "steps": [
            {
                "product": "CC=O",
                "reactant_smiles": ["c1ccccc1"],
                "reaction_type": "injected loose route",
            }
        ]
    }
    proof["route"] = injected_route
    proof["proof_evidence"]["stitched_route"]["route"] = injected_route
    forest = compile_explored_route_forest(
        {
            "case_id": "stitched-route-injection-test",
            "target_profile": {"target_name": "acetaldehyde", "target_smiles": "CC=O"},
            "parent_route_proof": proof,
        }
    )

    stitched = next(row for row in forest["branches"] if row["kind"] == "stitched_verified_route")
    stitched_steps = [
        row for row in forest["steps"] if row["step_id"] in stitched["step_ids"]
    ]
    assert stitched["solved"] is True
    assert stitched["executable"] is True
    assert "injected loose route" not in {row["label"] for row in stitched_steps}
    assert "c1ccccc1" not in {row.get("input_smiles") for row in forest["nodes"]}


def test_stitched_route_rejects_any_missing_proof_segment(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    for missing_segment in ("subgoal_raw_result", "literature_chain"):
        proof = json.loads(json.dumps(_stitched_parent_proof()))
        inputs = proof["proof_evidence"]["stitched_route"]["proof_inputs"]
        if missing_segment == "subgoal_raw_result":
            inputs["subgoal_raw_result"] = {}
            for row in inputs["route_expansion_result"]["subgoals"]:
                row["raw_result"] = {}
        else:
            inputs["literature_chain_audit"]["chain"] = []
        forest = compile_explored_route_forest(
            {
                "case_id": f"stitched-missing-{missing_segment}",
                "target_profile": {"target_name": "acetaldehyde", "target_smiles": "CC=O"},
                "parent_route_proof": proof,
            }
        )
        assert not any(
            row["kind"] == "stitched_verified_route" for row in forest["branches"]
        )


def test_stitched_route_displays_all_stock_precursors_in_one_forward_dag(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    forest = compile_explored_route_forest(
        {
            "case_id": "stitched-multi-precursor-test",
            "target_profile": {"target_name": "acetaldehyde", "target_smiles": "CC=O"},
            "parent_route_proof": _stitched_parent_proof(),
            "literature_evidence": {
                "visual_chains": [
                    {
                        "source_ref": "doi:10.1000/advisory-only",
                        "steps": [
                            {
                                "reaction_class": "advisory visual alternative",
                                "reactant_smiles": ["C"],
                                "product_smiles": ["CO"],
                            }
                        ],
                    }
                ]
            },
        }
    )

    stitched = next(row for row in forest["branches"] if row["kind"] == "stitched_verified_route")
    steps = {row["step_id"]: row for row in forest["steps"]}
    stock_step = steps[stitched["segments"][0]["step_ids"][0]]
    literature_step = steps[stitched["segments"][1]["step_ids"][0]]
    nodes = {row["node_id"]: row for row in forest["nodes"]}

    assert forest["primary_branch_id"] == stitched["branch_id"]
    assert forest["primary_selection"]["status"] == "deterministically_verified"
    assert stitched["route_direction"] == "stock_to_literature_terminal_to_target"
    assert len(stock_step["from_node_ids"]) == 2
    assert {
        nodes[node_id]["canonical_isomeric_smiles"]
        for node_id in stock_step["from_node_ids"]
    } == {"CC", "O"}
    assert set(stock_step["to_node_ids"]) & set(literature_step["from_node_ids"])
    assert set(literature_step["to_node_ids"]) <= {
        row["node_id"] for row in forest["nodes"] if row["role"] == "target"
    }
    visual = next(row for row in forest["branches"] if row["kind"] == "visual_chain")
    assert visual["advisory_only"] is True
    assert "supporting_branch_ids" not in stitched


def test_stitched_route_closes_and_displays_every_literature_frontier(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    forest = compile_explored_route_forest(
        {
            "case_id": "multi-frontier-route-forest-test",
            "target_profile": {
                "target_name": "ethyl hydroperoxide",
                "target_smiles": "CCOO",
            },
            "parent_route_proof": _multi_frontier_stitched_parent_proof(),
        }
    )

    stitched = next(row for row in forest["branches"] if row["kind"] == "stitched_verified_route")
    steps = {row["step_id"]: row for row in forest["steps"]}
    closure_segments = [
        segment
        for segment in stitched["segments"]
        if segment["segment_id"].startswith("verified_stock_closure_")
    ]
    literature_segment = next(
        segment for segment in stitched["segments"] if segment["segment_id"] == "strict_literature_chain"
    )
    literature_step = steps[literature_segment["step_ids"][0]]
    closure_endpoints = {
        node_id
        for segment in closure_segments
        for node_id in steps[segment["step_ids"][-1]]["to_node_ids"]
    }

    assert len(closure_segments) == 2
    assert len(literature_step["from_node_ids"]) == 2
    assert closure_endpoints == set(literature_step["from_node_ids"])
    assert set(stitched["literature_terminal_node_ids"]) == closure_endpoints
    assert stitched["solved"] is True
    assert stitched["executable"] is True


def test_route_forest_integrates_source_detail_chain_in_synthesis_order(tmp_path) -> None:
    chain_path = tmp_path / "source_detail_chain_route_result.json"
    chain_path.write_text(
        json.dumps(
            {
                "schema_version": "compiled_source_detail_chain_route.v1",
                "accepted": True,
                "chain_audit": {
                    "accepted": True,
                    "chain": [
                        {
                            "step_index": 1,
                            "step_id": "visual_step_1_target",
                            "reactant_smiles": ["CCO"],
                            "product_smiles": "CC=O",
                            "source_ref": "doi:10.test/source-detail",
                            "condition_candidate": {"reagent": "PCC", "solvent": "DCM"},
                        },
                        {
                            "step_index": 2,
                            "step_id": "visual_step_2_precursor",
                            "reactant_smiles": ["CC"],
                            "product_smiles": "CCO",
                            "source_ref": "doi:10.test/source-detail",
                            "condition_candidate": {"reagent": "H2O", "reported_yield": "80%"},
                        },
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    blackboard = {
        "case_id": "mixed_source_detail_route_test",
        "target_profile": {"target_name": "acetaldehyde", "target_smiles": "CC=O"},
        "literature_evidence": {"source_candidates": [{"source_ref": "doi:10.test/source-detail"}]},
        "parent_route_proof": _solved_parent_proof(target_smiles="CC=O"),
    }

    forest = compile_explored_route_forest(blackboard, run_dir=tmp_path)

    core = next(row for row in forest["branches"] if row["kind"] == "literature_candidate")
    steps = {row["step_id"]: row for row in forest["steps"]}
    core_steps = [steps[step_id] for step_id in core["step_ids"]]
    assert [row["label"] for row in core_steps] == ["visual step 2 precursor", "visual step 1 target"]
    assert core["advisory_only"] is True
    assert not any(row["kind"] == "stitched_verified_route" for row in forest["branches"])


def test_route_forest_surfaces_chemenzy_subgoal_closure_as_integrated_evidence(tmp_path) -> None:
    chain_path = tmp_path / "source_detail_chain_route_result.json"
    chain_path.write_text(
        json.dumps(
            {
                "schema_version": "compiled_source_detail_chain_route.v1",
                "accepted": True,
                "chain_audit": {
                    "accepted": True,
                    "chain": [
                        {
                            "step_index": 1,
                            "step_id": "terminal_to_target",
                            "reactant_smiles": ["CCO"],
                            "product_smiles": "CC=O",
                            "source_ref": "doi:10.test/mixed",
                        }
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    subgoal_dir = tmp_path / "route_expansion_subgoals"
    subgoal_dir.mkdir()
    raw_path = subgoal_dir / "raw_result.json"
    raw = {
        "ok": True,
        "target": "CCO",
        "search_status": {"solved": True},
        "routes": [
            {
                "route_rank": 0,
                "metrics": {"terminal_reactants": ["NNN"], "terminal_stock_status": {"NNN": True}},
                "steps": [
                    {
                        "product": "CCO",
                        "main_reactant": "NNN",
                        "stock_status": {"NNN": True},
                        "reaction_type": "fallback",
                    }
                ],
            },
            {
                "route_rank": 2,
                "metrics": {
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                "steps": [
                    {
                        "product": "CCO",
                        "reactant_smiles": ["CC", "O"],
                        "stock_status": {"CC": True, "O": True},
                        "reaction_type": "hydration",
                        "reaction_interpretation": {"forward_summary": "Hydrate ethene to ethanol."},
                        "scores": {"confidence": 0.9},
                    }
                ],
            },
        ],
    }
    raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    verifier = verify_chemenzy_raw_routes(raw, target_smiles="CCO")
    (tmp_path / "route_expansion_subgoal_search_result.json").write_text(
        json.dumps(
            {
                "schema_version": "route_expansion_subgoal_search_result.v1",
                "accepted": True,
                "solved": True,
                "subgoals": [
                    {
                        "accepted": True,
                        "solved": True,
                        "raw_solved": True,
                        "raw_result_path": str(raw_path),
                        "route_count": 2,
                        "subgoal": {
                            "name": "source detail literature terminal",
                            "smiles": "CCO",
                            "policy": {"evidence_refs": ["doi:10.test/mixed"]},
                        },
                        "verifier": verifier,
                        "parent_relevance_gate": {"accepted": True},
                        "route_status": "solved",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    blackboard = {
        "case_id": "mixed_subgoal_closure_test",
        "target_profile": {"target_name": "acetaldehyde", "target_smiles": "CC=O"},
        "literature_evidence": {"source_candidates": [{"source_ref": "doi:10.test/mixed"}]},
        "parent_route_proof": _solved_parent_proof(target_smiles="CC=O"),
    }

    forest = compile_explored_route_forest(blackboard, run_dir=tmp_path)

    assert any(row["kind"] == "subgoal_verified_route" for row in forest["branches"])
    subgoal = next(row for row in forest["branches"] if row["kind"] == "subgoal_verified_route")
    subgoal_steps = {row["step_id"]: row for row in forest["steps"] if row["branch_id"] == subgoal["branch_id"]}
    assert any(row["label"] == "hydration" for row in subgoal_steps.values())
    assert "doi:10.test/mixed" in subgoal["source_refs"]
    assert all("doi:10.test/mixed" not in row["source_refs"] for row in subgoal_steps.values())
    assert not any(row["kind"] == "stitched_verified_route" for row in forest["branches"])
    html = render_route_forest_html(forest)
    assert "subgoal_verified_route" in html
    assert "ChemEnzy 子目标闭合" in html


def test_route_forest_module_classifier_avoids_taxane_labels_for_generic_statin_steps() -> None:
    assert _module_key_for_text("late-stage side-chain installation") == "sidechain_installation"
    assert _module_key_for_text("polycyclic scaffold core construction") == "scaffold_core_construction"
    assert _module_key_for_text("named-author route") == "other_route_module"

    amide_key = _module_key_for_text("amide to acid chloride amine precursors")
    scaffold_key = _module_key_for_text("target-proximal cage intermediate with matched ring system")

    assert amide_key == "amide_or_sidechain_assembly"
    assert scaffold_key == "scaffold_core_construction"
    assert _module_label_for_key(amide_key) == "酰胺 / 侧链连接"
    assert _module_label_for_key(scaffold_key) == "骨架构建 / 母核调整"


def test_route_forest_recovers_common_mojibake_display_text() -> None:
    assert _display_text_is_corrupt("鎺ㄨ崘涓荤嚎锛歅aal-Knorr 宸ヨ壓璺嚎")
    assert not _display_text_is_corrupt("推荐主线: Paal-Knorr 工艺路线")

    title = _branch_title_for_display(
        branch_id="branch:legacy_corrupt_title",
        title="鎺ㄨ崘涓荤嚎锛歅aal-Knorr 宸ヨ壓璺嚎",
        kind="process_evidence",
    )

    assert title == "文献工艺锚点"


def test_route_forest_projects_process_evidence_as_advisory_anchor() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "target1_process_case",
            "target_profile": {
                "target_name": "9-OH-4-HP",
                "target_smiles": "CCO",
                "family_hint": "steroid biotransformation phytosterols",
            },
            "literature_evidence": {
                "process_evidence_rows": [
                    {
                        "row_id": "process_evidence:target1",
                        "process_type": "whole_cell_biotransformation",
                        "source_ref": "doi:10.1186/s12934-021-01717-w",
                        "source_title": "Production of 9-OH-4-HP from phytosterols",
                        "endpoint_labels": ["9-OH-4-HP"],
                        "substrate_or_feedstock_labels": ["phytosterols"],
                        "biocatalyst_or_process_labels": ["Mycobacterium neoaurum"],
                        "confidence": "medium_high",
                        "risk_flags": ["feedstock_mixture_or_class"],
                        "summary": "phytosterols via Mycobacterium neoaurum to 9-OH-4-HP",
                    }
                ]
            },
        },
        run_dir="run/target1",
    )

    assert forest["counts"]["process_evidence_rows"] == 1
    process_branch = next(branch for branch in forest["branches"] if branch["kind"] == "process_evidence")
    assert process_branch["synthesis_class"] == "biosynthesis"
    assert process_branch["advisory_only"] is True
    assert process_branch["solved"] is False
    assert process_branch["executable"] is False
    assert forest["primary_branch_id"] == process_branch["branch_id"]
    assert forest["primary_selection"]["status"] == "advisory"
    haystack = json.dumps(forest, ensure_ascii=False)
    assert "9-OH-4-HP" in haystack
    assert "phytosterols" in haystack
    assert "process evidence is an advisory route anchor" in haystack
    assert "生物合成 / 生物转化" in render_route_forest_html(forest)


def test_route_forest_classifies_only_structured_route_metadata() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "structured_route_classification",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "parent_route_proof": _solved_parent_proof(),
            "literature_evidence": {
                "exact_rows": [
                    {
                        "row_id": "exact:total",
                        "accepted": True,
                        "source_ref": "doi:10.1000/total",
                        "reactant_smiles": ["CC=O"],
                        "product_smiles": "CCO",
                        "synthesis_class": "total_synthesis",
                    }
                ],
                "process_evidence_rows": [
                    {
                        "row_id": "process:bio",
                        "process_type": "whole_cell_biotransformation",
                        "source_ref": "doi:10.1000/bio",
                        "endpoint_labels": ["ethanol"],
                        "substrate_or_feedstock_labels": ["acetaldehyde"],
                        "biocatalyst_or_process_labels": ["whole cell"],
                    }
                ],
            },
            "broad_transform_templates": [
                {
                    "template_id": "template:semi",
                    "objective_type": "semisynthesis_from_natural_product",
                    "transform_logic": "structured semisynthesis objective",
                    "preserved_scaffold": "reported advanced intermediate",
                }
            ],
        }
    )

    by_kind = {branch["kind"]: branch for branch in forest["branches"]}
    assert by_kind["literature_candidate"]["synthesis_class"] == "total_synthesis"
    assert by_kind["process_evidence"]["synthesis_class"] == "biosynthesis"
    assert by_kind["broad_template"]["synthesis_class"] == "semisynthesis"
    assert forest["primary_branch_id"] == by_kind["direct_verified_route"]["branch_id"]
    assert forest["primary_selection"]["status"] == "deterministically_verified"
    assert all(
        {"solved", "executable", "advisory_only", "not_parent_route_proof"} <= set(branch)
        for branch in forest["branches"]
    )


def test_parent_proof_does_not_promote_unbound_process_backbone() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "unbound_display_backbone",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "parent_route_proof": _solved_parent_proof(target_smiles="CCO"),
            "literature_evidence": {
                "process_evidence_rows": [
                    {
                        "row_id": "unrelated-process",
                        "process_type": "whole_cell_biotransformation",
                        "source_ref": "doi:10.1000/unrelated",
                        "substrate_or_feedstock_labels": ["Unrelated A"],
                        "endpoint_labels": ["Unrelated Z"],
                    }
                ]
            },
        }
    )

    assert not any(row["kind"] == "stitched_verified_route" for row in forest["branches"])
    primary = next(row for row in forest["branches"] if row["is_primary"])
    assert primary["kind"] == "direct_verified_route"
    assert primary["solved"] is True
    assert forest["primary_selection"]["status"] == "deterministically_verified"


def test_route_forest_adds_unclosed_exploration_diagnostic_when_blackboard_is_nonempty() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "old_probe_without_route_branches",
            "target_profile": {
                "target_name": "old_probe",
                "target_smiles": "CCO",
            },
            "literature_evidence": {
                "source_candidates": [
                    {
                        "source_ref": "doi:10.1000/example",
                        "title": "A source candidate that was never converted into route evidence",
                    }
                ],
                "pdf_structure_evidence": [],
                "exact_rows": [],
                "visual_chains": [],
            },
            "action_history": [
                {
                    "round_index": 1,
                    "action_type": "search_literature",
                    "useful_artifact": True,
                }
            ],
        },
        run_dir="run/old_probe",
    )

    assert forest["counts"]["branches"] == 1
    assert forest["branches"][0]["kind"] == "diagnostic_failure"
    assert forest["branches"][0]["title"].startswith("Exploration incomplete")
    haystack = json.dumps(forest, ensure_ascii=False)
    assert "no displayable route branch was compiled" in haystack
    assert "source candidates: 1" in haystack


def test_route_forest_projects_direct_verified_chemenzy_route(tmp_path) -> None:
    result_path = tmp_path / "guided_chemenzy_result.json"
    raw = {
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
                        "index": 0,
                        "product": "CCO",
                        "reactant_smiles": ["C", "C", "O"],
                        "stock_status": {"C": True, "O": True},
                        "reaction_type": "unbound artifact B",
                        "scores": {"confidence": 0.91},
                    }
                ],
            }
        ],
    }
    verifier = verify_chemenzy_raw_routes(raw, target_smiles="CCO")
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "guided_chemenzy_rerun_result.v1",
                "solved": True,
                "route_status": "solved",
                "raw_route_verifier": verifier,
                "result": raw,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    blackboard = {
        "case_id": "ethanol_direct_route_test",
        "target_profile": {
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
        "artifact_refs": {"r1_run_guided_chemenzy_guided_chemenzy_rerun_result_v1": str(result_path)},
        "parent_route_proof": _solved_parent_proof(target_smiles="CCO"),
    }

    forest = compile_explored_route_forest(blackboard, run_dir=tmp_path)

    direct = next(row for row in forest["branches"] if row["kind"] == "direct_verified_route")
    assert forest["primary_branch_id"] == direct["branch_id"]
    assert direct["solved"] is True
    assert direct["executable"] is True
    assert direct["advisory_only"] is False
    assert direct["proof_binding"]["accepted"] is True
    assert direct["proof_binding"]["proof_mode"] == "direct_parent_route"
    assert len(direct["proof_binding"]["route_structure_sha256"]) == 64
    assert forest["counts"]["steps"] == 1
    assert any(row["label"] == "ethanol" for row in forest["nodes"])
    assert any(row["label"] == "test materialized route" for row in forest["steps"])
    assert not any(row["label"] == "unbound artifact B" for row in forest["steps"])
    assert not any(row["canonical_isomeric_smiles"] == "C" for row in forest["nodes"])


def test_guided_l1_or_l2_route_without_parent_proof_is_advisory(tmp_path) -> None:
    result_path = tmp_path / "guided_chemenzy_result.json"
    raw = {
        "target": "CCO",
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                "steps": [
                    {
                        "product": "CCO",
                        "reactant_smiles": ["CC", "O"],
                        "stock_status": {"CC": True, "O": True},
                        "reaction_type": "guided candidate",
                        "atom_mapped_reaction_smiles": (
                            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                        ),
                    }
                ],
            }
        ],
    }
    verifier = verify_chemenzy_raw_routes(raw, target_smiles="CCO")
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "guided_chemenzy_rerun_result.v1",
                "raw_route_verifier": verifier,
                "result": raw,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    forest = compile_explored_route_forest(
        {
            "case_id": "unbound-guided-route",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "artifact_refs": {"guided_chemenzy_result": str(result_path)},
        },
        run_dir=tmp_path,
    )

    direct = next(row for row in forest["branches"] if row["kind"] == "direct_verified_route")
    assert direct["solved"] is False
    assert direct["executable"] is False
    assert direct["advisory_only"] is True
    assert direct["proof_binding"]["accepted"] is False
    assert direct["proof_binding"]["reasons"] == [
        "accepted_parent_route_proof_binding_missing"
    ]
    step = next(row for row in forest["steps"] if row["step_id"] in direct["step_ids"])
    assert step["reaction_step_proof"]["accepted"] is False
    assert step["reaction_step_proof"]["proof_level"] == "L2_mapping_consistent"
    assert step["trust_vector"]["proof_tier"] == "L2_mapping_consistent"
    assert forest["primary_selection"]["status"] == "advisory"
    assert forest["primary_selection"]["advisory_only"] is True


def test_explicit_unresolved_final_verdict_cannot_display_solved_branch() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "unresolved-verdict-overrides-display",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "parent_route_proof": _solved_parent_proof(target_smiles="CCO"),
            "final_verdict": {
                "verdict": "unresolved",
                "route_status": "unresolved",
                "solved": False,
            },
        }
    )

    direct = next(row for row in forest["branches"] if row["kind"] == "direct_verified_route")
    assert direct["proof_binding"]["accepted"] is True
    assert direct["solved"] is False
    assert direct["executable"] is False
    assert direct["advisory_only"] is True
    assert not any(
        branch["solved"] or branch["executable"] or not branch["advisory_only"]
        for branch in forest["branches"]
    )


def test_route_forest_rejects_backend_solved_claim_without_deterministic_verifier(tmp_path) -> None:
    result_path = tmp_path / "guided_chemenzy_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "guided_chemenzy_rerun_result.v1",
                "solved": True,
                "route_status": "solved",
                "raw_route_verifier": {"accepted": False, "route_status": "fake_closed_rejected"},
                "result": {
                    "routes": [
                        {
                            "route_rank": 0,
                            "stock_closed": True,
                            "steps": [
                                {"product": "CCO", "main_reactant": "CC", "reaction_type": "hydration"}
                            ],
                        }
                    ]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    blackboard = {
        "case_id": "false_solved_route_test",
        "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
        "artifact_refs": {"guided_chemenzy_result": str(result_path)},
    }

    forest = compile_explored_route_forest(blackboard, run_dir=tmp_path)

    assert not any(row["kind"] == "direct_verified_route" for row in forest["branches"])


def test_route_forest_smiles_detection_uses_chemistry_parser() -> None:
    assert _looks_like_smiles("CC")
    assert _looks_like_smiles("CO")
    assert _looks_like_smiles("CCO")
    assert not _looks_like_smiles("aspirin")
    assert not _looks_like_smiles("ethyl alcohol")


def test_verified_route_rank_selection_is_fail_closed_and_type_tolerant() -> None:
    routes = [
        {"route_rank": 0, "steps": [{"product": "CCO", "main_reactant": "C"}]},
        {"route_rank": 2, "steps": [{"product": "CCO", "main_reactant": "CC"}]},
    ]

    assert _route_by_verified_rank(routes, "2")["route_rank"] == 2
    assert _route_by_verified_rank(routes, 99) == {}
    assert _route_by_verified_rank(routes, None) == {}


def test_route_forest_ignores_rejected_source_detail_chain() -> None:
    payload = {
        "schema_version": "compiled_source_detail_chain_route.v1",
        "accepted": False,
        "chain_audit": {
            "accepted": False,
            "chain": [{"step_index": 1, "reactant_smiles": ["CC"], "product_smiles": "CCO"}],
        },
    }

    assert _chain_rows_from_source_detail_payload(payload) == []


def test_route_forest_embeds_molecule_svgs_and_step_conditions() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "structure_condition_display_test",
            "target_profile": {
                "target_name": "ethanol",
                "target_smiles": "CCO",
            },
            "literature_evidence": {
                "visual_chains": [
                    {
                        "source_ref": "doi:10.example/conditions",
                        "source_title": "Conditioned visual route",
                        "steps": [
                            {
                                "step_id": "conditioned_step",
                                "reaction_class": "reduction",
                                "reactant_smiles": ["CC=O"],
                                "product_smiles": ["CCO"],
                                "condition_candidate": {
                                    "schema_version": "condition_candidate.v1",
                                    "condition_status": "evidence_backed",
                                    "reagent": "NaBH4",
                                    "solvent": "MeOH",
                                    "temperature": "0 °C to rt",
                                    "reported_yield": "82%",
                                },
                                "source_locator": "Scheme 1 arrow conditions",
                            }
                        ],
                    }
                ]
            },
        }
    )

    rendered_nodes = [node for node in forest["nodes"] if node.get("structure_svg")]
    assert rendered_nodes
    assert any(node.get("formula") == "C2H6O" for node in rendered_nodes)
    assert all(node.get("structure_status") == "rendered" for node in rendered_nodes)

    conditioned_steps = [step for step in forest["steps"] if step.get("conditions")]
    assert conditioned_steps
    haystack = json.dumps(conditioned_steps, ensure_ascii=False)
    assert "NaBH4" in haystack
    assert "82%" in haystack

    html = render_route_forest_html(forest)
    assert "mol-structure" in html
    assert "condition-list" in html
    assert "condition-label" in html
    assert "条件" in html


def test_direct_parent_proof_still_projects_without_artifact(tmp_path) -> None:
    blackboard = {
        "case_id": "embedded_parent_route_test",
        "target_profile": {
            "target_name": "ethanol",
            "target_smiles": "CCO",
        },
        "parent_route_proof": _solved_parent_proof(target_smiles="CCO"),
    }

    forest = compile_explored_route_forest(blackboard, run_dir=tmp_path)

    assert forest["counts"]["branches"] == 1
    assert forest["branches"][0]["kind"] == "direct_verified_route"
    assert forest["branches"][0]["solved"] is True
    assert forest["branches"][0]["executable"] is True
    assert forest["counts"]["steps"] == 1
    haystack = json.dumps(forest, ensure_ascii=False)
    assert "ethanol" in haystack
    assert "test materialized route" in haystack


def test_route_forest_preserves_diagnostic_failure_when_no_route_exists() -> None:
    blackboard = {
        "case_id": "aspirin_failed_chemenzy_test",
        "target_profile": {
            "target_name": "aspirin",
            "target_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        },
        "route_failures": [
            {
                "schema_version": "agent_route_failure.v1",
                "failure_class": "chemenzy_runtime_diagnostic",
                "reason": "chemenzy_missing_output",
                "artifact_ref": "guided_chemenzy_result.json",
            }
        ],
    }

    forest = compile_explored_route_forest(blackboard)

    assert forest["counts"]["branches"] == 1
    assert forest["branches"][0]["kind"] == "diagnostic_failure"
    assert "chemenzy_missing_output" in forest["branches"][0]["missing"]
    assert forest["counts"]["steps"] == 1


def test_route_forest_html_is_read_only_and_inspectable() -> None:
    forest = compile_explored_route_forest(_sample_paclitaxel_blackboard())
    html = render_route_forest_html(forest)

    assert "AUTOPLANNER · ROUTE FOREST" in html
    assert "beta-lactam side-chain precursor" in html
    assert "forest-data" in html
    assert "只读视图" in html
    assert "route_forest_delivery.v1" in html
    assert "source_forest_sha256" in html
    assert "delivery_sha256" in html
    assert "relationships" in html
    for branch in forest["branches"]:
        assert branch["branch_id"] in html
        assert branch["kind"] in html
    assert "父路线未闭合" in html
    assert "data-related-branch" not in html
    assert 'data-detail-tab="alternatives"' in html
    assert 'data-graph-mode="clusters"' in html
    assert 'data-graph-mode="shared"' in html
    assert 'data-graph-mode="current"' in html
    assert 'id="graphMinimap"' in html
    assert 'data-graph-action="fit"' in html
    assert 'data-graph-action="zoom-in"' in html
    assert 'id="navResizeHandle"' in html
    assert 'id="inspectorResizeHandle"' in html
    assert "state.zoom < .18 ? 'overview'" in html
    assert html.count("dataset.zoomBand =") == 1
    assert "unique(PROOF_ORDER.map(tierClass))" in html
    assert "primary.proof_level === 'parent_route_proof'" in html
    assert "primaryBranch.not_parent_route_proof === false" in html
    assert "primaryBranch.not_parent_route_proof !== true" not in html
    assert "some(branch => branch.solved" not in html
    assert "routeFlowSvg" not in html


def test_route_forest_initial_selection_uses_compiled_primary_branch() -> None:
    forest = compile_explored_route_forest(_sample_paclitaxel_blackboard())
    html = render_route_forest_html(forest)

    assert forest["primary_branch_id"] in html
    assert "const defaultBranchId = forest.primary_branch_id" in html
    assert "const initialBranchId = persisted.selectedBranchId" in html
    assert "selectedBranchId: initialBranchId" in html
    assert "selectedStepId: laneByBranch.get(initialBranchId)" in html
    assert "lane.is_primary" in html
    assert "主分支" in html


def test_every_compiled_branch_kind_owns_its_step_and_dependency_foreign_keys() -> None:
    blackboard = _sample_paclitaxel_blackboard()
    blackboard["literature_evidence"]["process_evidence_rows"] = [
        {
            "row_id": "shared-process-id",
            "endpoint_labels": ["paclitaxel"],
            "substrate_or_feedstock_labels": ["10-deacetylbaccatin III"],
            "biocatalyst_or_process_labels": ["semisynthesis"],
            "source_ref": "doi:10.1000/process-a",
        },
        {
            "row_id": "shared-process-id",
            "endpoint_labels": ["baccatin III"],
            "substrate_or_feedstock_labels": ["taxane feedstock"],
            "biocatalyst_or_process_labels": ["biotransformation"],
            "source_ref": "doi:10.1000/process-b",
        },
    ]
    blackboard["retrosynthetic_proposals"].append(
        {
            "proposal_id": "proposal_ester_coupling",
            "proposal_label": "distinct duplicate-id proposal",
            "proposal_type": "side-chain disconnection",
            "precursor_smiles": "CC=O.O",
            "confidence": "low",
        }
    )
    blackboard["broad_transform_templates"].append(
        {
            "template_id": "template_c13_sidechain",
            "objective_type": "protection state adjustment",
            "transform_logic": "distinct duplicate-id template",
        }
    )

    forest = compile_explored_route_forest(blackboard)
    assert {
        "visual_chain",
        "process_evidence",
        "retrosynthetic_proposal",
        "broad_template",
    } <= {branch["kind"] for branch in forest["branches"]}

    owner_by_step: dict[str, str] = {}
    for branch in forest["branches"]:
        for step_id in branch["step_ids"]:
            assert step_id not in owner_by_step
            owner_by_step[step_id] = branch["branch_id"]

    assert set(owner_by_step) == {step["step_id"] for step in forest["steps"]}
    assert all(
        step["branch_id"] == owner_by_step[step["step_id"]]
        for step in forest["steps"]
    )

    graph = forest["dependency_graph"]
    for collection in ("reaction_nodes", "hyperedges", "edges"):
        assert all(
            row["branch_id"] == owner_by_step[row["reaction_step_id"]]
            for row in graph[collection]
        )
    assert {
        view["branch_id"]: view["step_ids"] for view in graph["branch_views"]
    } == {
        branch["branch_id"]: branch["step_ids"] for branch in forest["branches"]
    }


def test_write_route_forest_artifacts_writes_json_and_html(tmp_path) -> None:
    result = write_route_forest_artifacts(_sample_paclitaxel_blackboard(), run_dir=tmp_path)

    forest_path = tmp_path / "explored_route_forest.json"
    html_path = tmp_path / "route_forest.html"
    assert result["forest_path"] == str(forest_path.resolve())
    assert result["html_path"] == str(html_path.resolve())
    assert forest_path.exists()
    assert html_path.exists()
    assert "paclitaxel" in html_path.read_text(encoding="utf-8")


def test_route_forest_projects_explicit_bipartite_dependencies_without_array_adjacency() -> None:
    blackboard = {
        "case_id": "explicit_dependency_projection",
        "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
        "literature_evidence": {
            "visual_chains": [
                {
                    "accepted": True,
                    "source_ref": "doi:10.1000/disconnected-figure",
                    "candidate_steps": [
                        {
                            "step_id": "independent-a",
                            "reaction_class": "first independent reaction",
                            "reactant_smiles": ["CC"],
                            "product_smiles": ["CCC"],
                        },
                        {
                            "step_id": "independent-b",
                            "reaction_class": "second independent reaction",
                            "reactant_smiles": ["O"],
                            "product_smiles": ["CO"],
                        },
                    ],
                }
            ]
        },
    }

    forest = compile_explored_route_forest(blackboard)

    graph = forest["dependency_graph"]
    assert graph["schema_version"] == "molecule_reaction_dependency_graph.v1"
    assert graph["graph_kind"] == "molecule_reaction_bipartite_hypergraph"
    assert graph["no_array_adjacency_edges"] is True
    assert {row["node_type"] for row in graph["nodes"]} == {"molecule", "reaction"}
    assert {row["edge_type"] for row in graph["edges"]} == {
        "molecule_to_reaction",
        "reaction_to_molecule",
    }
    branch_view = graph["branch_views"][0]
    assert len(branch_view["step_ids"]) == 2
    assert branch_view["dependencies"] == []
    assert branch_view["dependency_semantics"].startswith("producer/consumer")


def test_name_only_same_label_is_namespaced_per_branch_source_and_evidence_row() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "name-only-collision",
            "target_profile": {"target_name": "unknown target", "target_smiles": ""},
            "literature_evidence": {
                "visual_chains": [
                    {
                        "source_ref": "doi:10.1000/source-a",
                        "candidate_steps": [
                            {
                                "step_id": "source-a-row-1",
                                "reaction_class": "source A reaction",
                                "reactant_labels": ["Starting material A"],
                                "product_label": "Intermediate 3",
                            }
                        ],
                    },
                    {
                        "source_ref": "doi:10.1000/source-b",
                        "candidate_steps": [
                            {
                                "step_id": "source-b-row-1",
                                "reaction_class": "source B reaction",
                                "reactant_labels": ["Intermediate 3"],
                                "product_label": "Product B",
                            }
                        ],
                    },
                ]
            },
        }
    )

    nodes = {row["node_id"]: row for row in forest["nodes"]}
    steps = {row["label"]: row for row in forest["steps"]}
    source_a_intermediate = steps["source A reaction"]["to_node_ids"][0]
    source_b_intermediate = steps["source B reaction"]["from_node_ids"][0]

    assert nodes[source_a_intermediate]["label"] == "Intermediate 3"
    assert nodes[source_b_intermediate]["label"] == "Intermediate 3"
    assert source_a_intermediate != source_b_intermediate
    assert nodes[source_a_intermediate]["canonical_isomeric_smiles"] == ""
    assert nodes[source_b_intermediate]["canonical_isomeric_smiles"] == ""
    assert nodes[source_a_intermediate]["identity_namespace"] != nodes[source_b_intermediate][
        "identity_namespace"
    ]
    assert all(
        view["dependencies"] == []
        for view in forest["dependency_graph"]["branch_views"]
    )
    assert source_a_intermediate in forest["dependency_graph"]["branch_views"][0][
        "terminal_molecule_node_ids"
    ]
    assert source_b_intermediate in forest["dependency_graph"]["branch_views"][1][
        "root_molecule_node_ids"
    ]


def test_route_forest_emits_multidimensional_trust_and_complete_tier_legend() -> None:
    blackboard = {
        "case_id": "trust_vector_projection",
        "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
        "retrosynthetic_proposals": [
            {
                "proposal_id": "proposal:reduction",
                "proposal_label": "carbonyl reduction",
                "precursor_smiles": "CC=O",
                "product_smiles": "CCO",
                "evidence_refs": ["doi:10.1000/example"],
            }
        ],
    }

    forest = compile_explored_route_forest(blackboard)

    step = forest["steps"][0]
    trust = step["trust_vector"]
    assert {
        "identity",
        "connectivity",
        "source_independence",
        "stock",
        "conditions",
        "forward_feasibility",
        "proof_tier",
    } <= set(trust)
    assert trust["proof_tier"] == "L0_materialized"
    assert trust["status"]["forward_feasibility"] == "not_universally_proven"
    assert step["visual_encoding"]["width_semantics"] == "independent_support_group_count"
    assert step["visual_encoding"]["opacity_semantics"] == "mean_trust_dimension"
    assert {row["proof_tier"] for row in forest["dependency_graph"]["proof_tier_legend"]} == {
        "L0_rejected",
        "L0_advisory",
        "L0_materialized",
        "L1_graph_stock_closed",
        "L2_mapping_consistent",
        "L2_reaction_validated",
        "L3_precedent_supported",
        "L4_procurement_ready",
    }


def test_route_forest_legacy_interface_pairs_are_diagnostics_only() -> None:
    blackboard = {
        "case_id": "safe_replacement_projection",
        "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
        "retrosynthetic_proposals": [
            {
                "proposal_id": "proposal:chemical",
                "proposal_label": "chemical carbonyl reduction",
                "precursor_smiles": "CC=O",
                "product_smiles": "CCO",
            },
            {
                "proposal_id": "proposal:enzyme",
                "proposal_label": "enzymatic carbonyl reduction",
                "precursor_smiles": "CC=O",
                "product_smiles": "CCO",
            },
            {
                "proposal_id": "proposal:broken",
                "proposal_label": "broken same-product shortcut",
                "precursor_smiles": "CC",
                "product_smiles": "CCO",
            },
        ],
    }

    forest = compile_explored_route_forest(blackboard)

    validation = forest["replacement_validation"]
    assert validation["schema_version"] == "route_replacement_validation.v1"
    assert validation["candidate_count"] == 0
    assert validation["validated_count"] == 0
    assert validation["rejected_count"] == 0
    assert validation["semantics"]["invalid_candidates_are_not_replaceable"] is True
    assert validation["semantics"]["single_step_splicing_forbidden"] is True
    diagnostics = validation["interface_diagnostics"]
    assert diagnostics["schema_version"] == "route_interface_diagnostics.v1"
    assert diagnostics["candidate_count"] == 6
    assert diagnostics["interface_compatible_count"] == 2
    assert all(row["validated"] is False for row in diagnostics["records"])
    assert all(row["diagnostics_only"] is True for row in diagnostics["records"])

    html = render_route_forest_html(forest)
    assert "No backend AND/OR-revalidated replacement is available." in html
    assert "never enable a single-step splice" in html
    assert "Array adjacency never creates an edge" in html


def _portfolio_projection_blackboard(*, include_alt_in_top_k: bool) -> dict:
    def with_digest(value: dict, *, field: str = "content_sha256") -> dict:
        payload = dict(value)
        payload.pop(field, None)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload[field] = hashlib.sha256(encoded).hexdigest()
        return payload

    molecules = {
        "target": "CCOC",
        "shared": "CCO",
        "side-a": "O",
        "side-b": "N",
        "stock-1": "CC",
        "stock-2": "[Na+]",
        "stock-3": "[Cl-]",
    }

    def hyperedge(
        edge_id: str,
        product_id: str,
        precursor_ids: list[str],
        *,
        support_group: str,
    ) -> dict:
        return {
            "schema_version": "reaction_hyperedge.v2",
            "hyperedge_id": edge_id,
            "product_molecule_id": product_id,
            "precursor_molecule_ids": precursor_ids,
            "candidate_envelope_ids": [],
            "evidence_claim_ids": [],
            "source_channels": ["literature_exact"],
            "independent_support_groups": [support_group],
            "reaction_families": [f"reaction {edge_id}"],
            "rank_score": 0.8,
            "advisory_only": True,
        }

    edges = [
        hyperedge("e-main", "target", ["shared", "side-a"], support_group="paper:main"),
        hyperedge("e-alt", "target", ["shared", "side-b"], support_group="paper:alt"),
        hyperedge("e-shared", "shared", ["stock-1"], support_group="paper:shared"),
        hyperedge("e-side-a", "side-a", ["stock-2"], support_group="paper:side-a"),
        hyperedge("e-side-b", "side-b", ["stock-3"], support_group="paper:side-b"),
    ]

    def route(
        route_id: str,
        target_edge: str,
        side_product: str,
        side_edge: str,
        stock_side: str,
        *,
        diversity: float,
    ) -> dict:
        return with_digest({
            "schema_version": "route_portfolio_item.v1",
            "route_id": route_id,
            "root_molecule_id": "target",
            "selected_hyperedges": [
                {"product_molecule_id": "target", "hyperedge_id": target_edge},
                {"product_molecule_id": "shared", "hyperedge_id": "e-shared"},
                {"product_molecule_id": side_product, "hyperedge_id": side_edge},
            ],
            "hyperedge_ids": [target_edge, "e-shared", side_edge],
            "molecule_ids": ["target", "shared", side_product, "stock-1", stock_side],
            "stock_terminal_ids": ["stock-1", stock_side],
            "source_channels": ["literature_exact"],
            "independent_support_groups": [
                "paper:shared",
                f"paper:{'main' if target_edge == 'e-main' else 'alt'}",
                f"paper:{side_product}",
            ],
            "weakest_proof_level": 2,
            "mean_edge_rank": 0.8,
            "base_score": 0.82,
            "diversity_score": diversity,
            "portfolio_score": 0.8,
            "complete": True,
            "reaction_validated": True,
            "unresolved_frontiers": [],
        })

    main_route = route(
        "portfolio-route:main",
        "e-main",
        "side-a",
        "e-side-a",
        "stock-2",
        diversity=1.0,
    )
    alt_route = route(
        "portfolio-route:alt",
        "e-alt",
        "side-b",
        "e-side-b",
        "stock-3",
        diversity=0.6,
    )
    portfolio_routes = [main_route, alt_route] if include_alt_in_top_k else [main_route]
    edge_by_id = {row["hyperedge_id"]: row for row in edges}
    exact_edge_bindings = {}
    for edge_id, edge_row in edge_by_id.items():
        named_level = (
            "L3_precedent_supported" if edge_id == "e-main" else "L2_reaction_validated"
        )
        exact_edge_bindings[edge_id] = with_digest(
            {
                "schema_version": "exact_edge_proof_binding.v1",
                "hyperedge_id": edge_id,
                "product_molecule_id": edge_row["product_molecule_id"],
                "precursor_molecule_ids": sorted(edge_row["precursor_molecule_ids"]),
                "structure_signature_sha256": hashlib.sha256(edge_id.encode()).hexdigest(),
                "proof_level": named_level,
                "portfolio_proof_level": 3 if edge_id == "e-main" else 2,
                "advisory": False,
                "proof_accepted": True,
                "proof_digest": hashlib.sha256(f"proof:{edge_id}".encode()).hexdigest(),
                "route_proof_digest": hashlib.sha256(
                    f"route-proof:{edge_id}".encode()
                ).hexdigest(),
                "reaction_digest": hashlib.sha256(f"reaction:{edge_id}".encode()).hexdigest(),
                "trusted_precedent_sha256": (
                    hashlib.sha256(f"precedent:{edge_id}".encode()).hexdigest()
                    if named_level == "L3_precedent_supported"
                    else ""
                ),
                "proof_source": "legacy_best_accepted_route",
                "proof_bank_entry_id": "",
                "proof_bank_entry_sha256": "",
            },
            field="binding_sha256",
        )
    stock_bindings = {}
    for molecule_id in ["stock-1", "stock-2", "stock-3"]:
        stock_bindings[molecule_id] = with_digest(
            {
                "schema_version": "exact_stock_binding.v1",
                "molecule_id": molecule_id,
                "canonical_isomeric_smiles": molecules[molecule_id],
                "catalog_id": f"catalog:{molecule_id}",
                "catalog_sha256": hashlib.sha256(f"catalog:{molecule_id}".encode()).hexdigest(),
                "lookup_basis": "canonical_isomeric_smiles",
                "evidence_sha256": hashlib.sha256(f"evidence:{molecule_id}".encode()).hexdigest(),
                "binding_authority": "legacy_best_route_independent_stock_audit",
            },
            field="binding_sha256",
        )
    portfolio = with_digest(
        {
            "schema_version": "route_portfolio.v1",
            "root_molecule_id": "target",
            "routes": portfolio_routes,
            "complete_candidate_count": 2,
            "enumerated_candidate_count": 2,
            "truncated": True,
            "reasons": ["route_enumeration_truncated"],
            "selection_policy": "and_or_closure_then_maximal_marginal_relevance",
            "requires_explicit_stock_and_reaction_proof": True,
        }
    )
    bindings = with_digest(
        {
            "schema_version": "route_portfolio_bindings.v1",
            "stock_molecule_ids": ["stock-1", "stock-2", "stock-3"],
            "edge_proof_levels": {
                "e-main": 3,
                "e-alt": 2,
                "e-shared": 2,
                "e-side-a": 2,
                "e-side-b": 2,
            },
            "exact_edge_proof_bindings": exact_edge_bindings,
            "stock_bindings": stock_bindings,
        }
    )
    replacement_catalog = with_digest(
        {
            "schema_version": "route_replacement_catalog.v1",
            "portfolio_content_sha256": portfolio["content_sha256"],
            "portfolio_integrity_valid": True,
            "candidate_count": 2,
            "accepted_candidate_count": 1,
            "rejected_candidate_count": 1,
            "truncated": False,
            "candidates": [
                {
                    "candidate_id": "replacement:main-to-alt",
                    "base_route_id": "portfolio-route:main",
                    "product_molecule_id": "target",
                    "original_hyperedge_id": "e-main",
                    "replacement_hyperedge_id": "e-alt",
                    "accepted": True,
                    "route": alt_route,
                    "connectivity_revalidated": True,
                    "stock_closure_revalidated": True,
                    "reaction_proof_revalidated": True,
                    "reasons": [],
                },
                {
                    "candidate_id": "replacement:rejected-open-route",
                    "base_route_id": "portfolio-route:main",
                    "product_molecule_id": "target",
                    "original_hyperedge_id": "e-main",
                    "replacement_hyperedge_id": "e-rejected",
                    "accepted": False,
                    "route": {},
                    "connectivity_revalidated": True,
                    "stock_closure_revalidated": True,
                    "reaction_proof_revalidated": True,
                    "reasons": ["no_stock_closed_reaction_validated_route"],
                },
            ],
        }
    )
    graph = {
        "schema_version": "route_consensus_graph.v1",
        "nodes": [],
        "steps": [],
        "route_hypotheses": [],
        "v2_overlay": {
            "schema_version": "route_hypergraph_overlay.v2",
            "root_molecule_id": "target",
            "validation": {"valid": True, "errors": []},
            "molecules": [
                {"molecule_id": molecule_id, "canonical_isomeric_smiles": smiles}
                for molecule_id, smiles in molecules.items()
            ],
            "evidence_claims": [],
            "candidate_envelopes": [],
            "reaction_hyperedges": edges,
        },
        "route_portfolio": portfolio,
        "route_portfolio_bindings": bindings,
        "route_replacement_catalog": replacement_catalog,
    }
    return {
        "case_id": "portfolio-projection",
        "target_profile": {"target_name": "portfolio target", "target_smiles": "CCOC"},
        "route_consensus_graph": graph,
    }


def test_each_top_k_portfolio_route_is_an_independent_closed_branch_dag() -> None:
    forest = compile_explored_route_forest(
        _portfolio_projection_blackboard(include_alt_in_top_k=True)
    )
    portfolio_branches = [
        row
        for row in forest["branches"]
        if row.get("kind") == "proof_eligible_portfolio_route"
    ]
    assert len(portfolio_branches) == 2
    assert {row["portfolio_route_id"] for row in portfolio_branches} == {
        "portfolio-route:main",
        "portfolio-route:alt",
    }
    assert all(row["listed"] is True for row in portfolio_branches)
    assert all(row["proof_eligible"] is True for row in portfolio_branches)
    assert all(row["weakest_proof_tier"] == "L2_reaction_validated" for row in portfolio_branches)
    assert all(row["portfolio_enumeration"]["solver_truncated"] is True for row in portfolio_branches)
    assert {row["diversity_score"] for row in portfolio_branches} == {1.0, 0.6}

    views = {
        row["portfolio_route_id"]: row
        for row in forest["dependency_graph"]["branch_views"]
        if row.get("portfolio_route_id")
        and (next(
            branch
            for branch in forest["branches"]
            if branch["branch_id"] == row["branch_id"]
        )).get("listed") is True
    }
    assert set(views) == {"portfolio-route:main", "portfolio-route:alt"}
    for branch in portfolio_branches:
        view = views[branch["portfolio_route_id"]]
        assert view["acyclic"] is True
        assert view["all_leaves_stock_bound"] is True
        assert set(view["stock_leaf_molecule_node_ids"]) == set(
            branch["stock_terminal_node_ids"]
        )
        assert view["target_molecule_node_ids"] == [branch["root_molecule_node_id"]]
        assert view["weakest_proof_tier"] == "L2_reaction_validated"

    steps = {row["step_id"]: row for row in forest["steps"]}
    main_branch = next(row for row in portfolio_branches if row["portfolio_route_id"].endswith("main"))
    alt_branch = next(row for row in portfolio_branches if row["portfolio_route_id"].endswith("alt"))
    assert set(main_branch["step_ids"]).isdisjoint(alt_branch["step_ids"])
    main_shared = next(
        steps[step_id]
        for step_id in main_branch["step_ids"]
        if steps[step_id].get("portfolio_hyperedge_id") == "e-shared"
    )
    alt_shared = next(
        steps[step_id]
        for step_id in alt_branch["step_ids"]
        if steps[step_id].get("portfolio_hyperedge_id") == "e-shared"
    )
    assert main_shared["to_node_ids"] == alt_shared["to_node_ids"]
    assert main_shared["step_id"] != alt_shared["step_id"]
    assert forest["route_portfolio_projection"]["projected_route_count"] == 2
    assert forest["route_portfolio_projection"]["solver_truncated"] is True


def test_backend_replacement_catalog_previews_complete_hidden_resolved_branch() -> None:
    forest = compile_explored_route_forest(
        _portfolio_projection_blackboard(include_alt_in_top_k=False)
    )
    validation = forest["replacement_validation"]
    assert validation["validation_engine"] == "and_or.validate_route_replacement"
    assert validation["validated_count"] == 1
    assert validation["rejected_count"] == 1
    record = next(row for row in validation["records"] if row["validated"])
    assert record["validated"] is True
    assert record["revalidated_route_branch_id"]
    replacement_branch = next(
        row
        for row in forest["branches"]
        if row["branch_id"] == record["revalidated_route_branch_id"]
    )
    assert replacement_branch["kind"] == "validated_replacement_route"
    assert replacement_branch["listed"] is False
    assert replacement_branch["portfolio_route_id"] == "portfolio-route:alt"
    replacement_steps = {
        row["portfolio_hyperedge_id"]
        for row in forest["steps"]
        if row["step_id"] in replacement_branch["step_ids"]
    }
    assert replacement_steps == {"e-alt", "e-shared", "e-side-b"}
    assert "e-main" not in replacement_steps
    assert "e-side-a" not in replacement_steps
    assert forest["route_portfolio_projection"]["replacement_preview_branch_count"] == 1
    rejected = next(row for row in validation["records"] if not row["validated"])
    assert rejected["candidate_step_id"] == ""
    assert rejected["reasons"] == ["no_stock_closed_reaction_validated_route"]

    html = render_route_forest_html(forest)
    assert "baseRows.slice" not in html
    assert "data-replacement-id" in html
    assert "data-replacement-preview" in html
    assert "function previewReplacement(target)" in html
    assert "function restoreReplacementPreview" in html
    assert "includeReplacementPreview: true" in html
    assert "完整替换路线预览 · 后端已重验" in html
    assert "完整的后端重验替换路线预览" in html
    assert "full AND/OR route re-solved" in html
    assert "no_stock_closed_reaction_validated_route" in html
    assert "function filteredLanes({ includeReplacementPreview = false } = {})" in html
    assert "lane.listed === false && !isReplacementPreview" in html


def test_portfolio_projection_fails_closed_on_tampered_hash_bound_proof_data() -> None:
    def rehash(value: dict, *, field: str = "content_sha256") -> None:
        payload = dict(value)
        payload.pop(field, None)
        value[field] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    route_tampered = _portfolio_projection_blackboard(include_alt_in_top_k=True)
    route_graph = route_tampered["route_consensus_graph"]
    route_graph["route_portfolio"]["routes"][0]["base_score"] = 0.01
    rehash(route_graph["route_portfolio"])
    route_forest = compile_explored_route_forest(route_tampered)
    assert route_forest["route_portfolio_projection"]["projected_route_count"] == 1
    assert "portfolio_route_content_sha256_mismatch" in route_forest[
        "route_portfolio_projection"
    ]["rejected_routes"][0]["reasons"]

    edge_tampered = _portfolio_projection_blackboard(include_alt_in_top_k=True)
    edge_graph = edge_tampered["route_consensus_graph"]
    edge_graph["route_portfolio_bindings"]["exact_edge_proof_bindings"]["e-main"][
        "portfolio_proof_level"
    ] = 4
    rehash(edge_graph["route_portfolio_bindings"])
    edge_forest = compile_explored_route_forest(edge_tampered)
    assert edge_forest["route_portfolio_projection"]["projected_route_count"] == 1
    assert any(
        "binding_sha256_mismatch" in reason
        for reason in edge_forest["route_portfolio_projection"]["rejected_routes"][0][
            "reasons"
        ]
    )

    mapping_promoted = _portfolio_projection_blackboard(include_alt_in_top_k=True)
    mapping_graph = mapping_promoted["route_consensus_graph"]
    mapping_binding = mapping_graph["route_portfolio_bindings"][
        "exact_edge_proof_bindings"
    ]["e-alt"]
    mapping_binding["proof_level"] = "L2_mapping_consistent"
    mapping_binding["portfolio_proof_level"] = 2
    mapping_binding["advisory"] = False
    rehash(mapping_binding, field="binding_sha256")
    rehash(mapping_graph["route_portfolio_bindings"])
    mapping_forest = compile_explored_route_forest(mapping_promoted)
    assert mapping_forest["route_portfolio_projection"]["projected_route_count"] == 1
    assert any(
        "proof_level_portfolio_level_mismatch" in reason
        for reason in mapping_forest["route_portfolio_projection"]["rejected_routes"][0][
            "reasons"
        ]
    )

    stock_tampered = _portfolio_projection_blackboard(include_alt_in_top_k=True)
    stock_graph = stock_tampered["route_consensus_graph"]
    stock_graph["route_portfolio_bindings"]["stock_bindings"]["stock-2"][
        "catalog_id"
    ] = "catalog:tampered"
    rehash(stock_graph["route_portfolio_bindings"])
    stock_forest = compile_explored_route_forest(stock_tampered)
    assert stock_forest["route_portfolio_projection"]["projected_route_count"] == 1
    assert any(
        "binding_sha256_mismatch" in reason
        for reason in stock_forest["route_portfolio_projection"]["rejected_routes"][0][
            "reasons"
        ]
    )

    catalog_tampered = _portfolio_projection_blackboard(include_alt_in_top_k=False)
    catalog_graph = catalog_tampered["route_consensus_graph"]
    catalog_graph["route_replacement_catalog"]["candidates"][0]["accepted"] = False
    catalog_forest = compile_explored_route_forest(catalog_tampered)
    assert catalog_forest["route_portfolio_projection"][
        "replacement_catalog_integrity_valid"
    ] is False
    assert catalog_forest["replacement_validation"]["records"] == []


def test_route_forest_default_is_complete_and_explicit_limits_report_truncation() -> None:
    proposals = [
        {
            "proposal_id": f"proposal:{index}",
            "proposal_label": f"candidate route {index}",
            "precursor_smiles": f"C{'C' * index}",
            "product_smiles": "CCO",
        }
        for index in range(1, 4)
    ]
    blackboard = {
        "case_id": "projection_coverage",
        "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
        "retrosynthetic_proposals": proposals,
    }

    complete = compile_explored_route_forest(blackboard)
    assert complete["counts"]["branches"] == 3
    assert complete["projection_coverage"]["complete"] is True
    assert complete["projection_coverage"]["categories"]["retrosynthetic_proposals"] == {
        "available_count": 3,
        "rendered_count": 3,
        "omitted_count": 0,
        "limit": None,
        "truncated": False,
    }

    limited = compile_explored_route_forest(blackboard, max_proposal_branches=1)
    coverage = limited["projection_coverage"]["categories"]["retrosynthetic_proposals"]
    assert coverage == {
        "available_count": 3,
        "rendered_count": 1,
        "omitted_count": 2,
        "limit": 1,
        "truncated": True,
    }
    assert limited["projection_coverage"]["complete"] is False
    assert limited["counts"]["truncated_projection_rows"] == 2
    assert "Projection truncated" in render_route_forest_html(limited)


def test_route_forest_marks_closeout_revision_as_source_context_only() -> None:
    forest = compile_explored_route_forest(
        {
            "case_id": "revision-aware-display",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "closeout_revision": {
                "schema_version": "closeout_revision.v1",
                "accepted": True,
                "status": "committed",
                "revision_id": "revision-123",
                "manifest_path": "revisions/revision-123/manifest.json",
                "manifest_sha256": "a" * 64,
                "authority": "content_addressed_closeout_manifest",
                "artifact_count": 2,
            },
            "artifact_digest_refs": {
                "route_forest": {
                    "schema_version": "closeout_artifact_digest_ref.v1",
                    "artifact_id": "route_forest",
                    "sha256": "b" * 64,
                    "revision_id": "revision-123",
                }
            },
        }
    )

    assert forest["artifact_revision"] == {
        "schema_version": "route_forest_source_revision_context.v1",
        "status": "source_context_committed",
        "scope": "blackboard_input_closeout_context",
        "committed": True,
        "self_authenticates_current_forest": False,
        "revision_id": "revision-123",
        "manifest_path": "revisions/revision-123/manifest.json",
        "manifest_sha256": "a" * 64,
        "authority": "content_addressed_closeout_manifest",
        "artifact_count": 2,
        "digest_ref_count": 1,
        "semantics": (
            "source closeout context only; an external manifest must bind "
            "this forest and rendered delivery"
        ),
    }
    html = render_route_forest_html(forest)
    assert "Delivery bytes verified" in html
    assert "current closeout requires external manifest" in html
    assert "source_context_only_never_self_authenticates_delivery" in html


def test_route_forest_propagates_only_deterministically_reverified_reaction_proof_tier() -> None:
    raw = {
        "target": "CCO",
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                    "steps": [
                        {
                            **_strict_literature_step(
                                step_id="common_stock_to_ethanol",
                                reactants=["CC", "O"],
                                product="CCO",
                                atom_mapped_reaction_smiles=(
                                    "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                                ),
                            ),
                            "stock_status": {"CC": True, "O": True},
                        }
                    ],
            }
        ],
    }
    verifier = verify_chemenzy_raw_routes(raw, target_smiles="CCO")
    assert verifier["reaction_validation"]["accepted"] is True
    proof = compile_stitched_parent_route_proof(
        target_smiles="CCO",
        target_name="ethanol",
        parent_verifier=verifier,
    )

    forest = compile_explored_route_forest(
        {
            "case_id": "reaction-proof-tier",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "parent_route_proof": proof,
        }
    )

    assert forest["branches"][0]["kind"] == "direct_verified_route"
    step = forest["steps"][0]
    assert step["reaction_step_proof"]["proof_source"] == "deterministic_reverified_route"
    assert step["trust_vector"]["proof_tier"] == "L3_precedent_supported"
    assert step["trust_vector"]["forward_feasibility"] == 0.95
    assert forest["branches"][0]["trust_vector"]["proof_tier"] == "L3_precedent_supported"


def test_exact_precedent_without_l2_reaction_proof_does_not_skip_to_l3(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    row = _strict_literature_step(
        step_id="ethanol_oxidation",
        reactants=["CCO"],
        product="CC=O",
    )
    forest = compile_explored_route_forest(
        {
            "case_id": "precedent-without-l2",
            "target_profile": {"target_name": "acetaldehyde", "target_smiles": "CC=O"},
            "literature_evidence": {"exact_rows": [row]},
        }
    )

    step = forest["steps"][0]
    assert step["exactness"] == "exact_literature_row"
    assert step["trust_vector"]["proof_tier"] == "L0_materialized"
    assert step["trust_vector"]["status"]["forward_feasibility"] == (
        "precedent_without_L2_reaction_validation"
    )
