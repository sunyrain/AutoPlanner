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
    raw = {
        "target": target_smiles,
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                "steps": [
                    {
                        "product": target_smiles,
                        "reactant_smiles": ["CC", "O"],
                        "stock_status": {"CC": True, "O": True},
                        "reaction_type": "test materialized route",
                    }
                ],
            }
        ],
    }
    verifier = verify_chemenzy_raw_routes(raw, target_smiles=target_smiles)
    return compile_stitched_parent_route_proof(
        target_smiles=target_smiles,
        target_name="test target",
        parent_verifier=verifier,
    )


def _strict_literature_step(*, step_id: str, reactants: list[str], product: str) -> dict:
    pdf_digest = hashlib.sha256(_SOURCE_FIXTURE.read_bytes()).hexdigest()
    image_digest = hashlib.sha256(_SOURCE_PAGE_FIXTURE.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(_SOURCE_MANIFEST_FIXTURE.read_bytes()).hexdigest()
    template_id = f"source_detail_exact_step:{step_id}"
    return {
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


def _stitched_parent_proof() -> dict:
    terminal = "CCO"
    target = "CCOC(C)=O"
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
                        "product": terminal,
                        "reactant_smiles": ["CC", "O"],
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
                    step_id="ethyl_acetate",
                    reactants=[terminal],
                    product=target,
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
        target_name="ethyl acetate",
        case_id="stitched-route-forest-test",
    )
    return compile_stitched_parent_route_proof(
        target_smiles=target,
        target_name="ethyl acetate",
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
                        "product": "CCO",
                        "reactant_smiles": ["CC", "O"],
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
                        "product": "O",
                        "reactant_smiles": ["O=O"],
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
    assert "consensusOverview" in html
    assert "selectedBranchId = el.getAttribute('data-view-branch')" in html


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
    assert "父路线：未闭合" in html
    assert "Advisory 分支步骤（不是父路线证明）" in html


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
                "product": "CCOC(C)=O",
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
            "target_profile": {"target_name": "ethyl acetate", "target_smiles": "CCOC(C)=O"},
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
                "target_profile": {"target_name": "ethyl acetate", "target_smiles": "CCOC(C)=O"},
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
            "target_profile": {"target_name": "ethyl acetate", "target_smiles": "CCOC(C)=O"},
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
                    "terminal_reactants": ["CC", "O"],
                    "terminal_stock_status": {"CC": True, "O": True},
                },
                "steps": [
                    {
                        "index": 0,
                        "product": "CCO",
                        "reactant_smiles": ["CC", "O"],
                        "stock_status": {"CC": True, "O": True},
                        "reaction_type": "hydration",
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
    assert forest["counts"]["steps"] == 1
    assert any(row["label"] == "ethanol" for row in forest["nodes"])
    assert any(row["label"] == "hydration" for row in forest["steps"])


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
    assert "conditionBlock" in html
    assert "条件：" in html


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

    assert "Explored Route Forest" in html
    assert "beta-lactam side-chain precursor" in html
    assert "forest-data" in html
    assert "只读结果页" in html
    assert "activeReplacement" in html
    assert "altBranchId" in html
    assert "branchTailSteps" in html
    assert "relationships" in html
    assert "stitched_verified_route" in html
    assert "拼接验证路线" in html
    assert "证据过程" in html
    assert "没有父路线证明时只展示明确标注的 advisory 分支" in html
    assert "父路线：未闭合" in html
    assert "data-related-branch" not in html
    assert "备选预览" in html
    assert "后续步骤按该备选所属分支" in html
    assert "路线关系" in html
    assert "证据过程" in html
    assert "viewPicker" in html
    assert "routeFlowSvg" in html
    assert "核心路线" in html
    assert "data-toggle-panel=\"nav\"" in html
    assert "data-toggle-panel=\"inspector\"" in html


def test_write_route_forest_artifacts_writes_json_and_html(tmp_path) -> None:
    result = write_route_forest_artifacts(_sample_paclitaxel_blackboard(), run_dir=tmp_path)

    forest_path = tmp_path / "explored_route_forest.json"
    html_path = tmp_path / "route_forest.html"
    assert result["forest_path"] == str(forest_path.resolve())
    assert result["html_path"] == str(html_path.resolve())
    assert forest_path.exists()
    assert html_path.exists()
    assert "paclitaxel" in html_path.read_text(encoding="utf-8")
