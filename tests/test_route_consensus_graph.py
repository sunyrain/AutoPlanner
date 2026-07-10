from __future__ import annotations

from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.graph import (
    assemble_route_consensus_graph,
    make_route_consensus_expansion,
    select_route_consensus_frontier,
)
from cascade_planner.harness.route_forest import compile_explored_route_forest, render_route_forest_html


def _candidate(
    candidate_id: str,
    *,
    product: str,
    precursors: list[str],
    source_ref: str,
    channel: str = "codex_strategy",
) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "reaction_family": "test disconnection",
        "transformation_rationale": "bounded graph test",
        "source_channel": channel,
        "source_refs": [source_ref],
        "evidence_refs": [],
        "evidence_level": "literature_exact" if channel == "codex_literature" else "model_only",
        "confidence": "medium",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "required_validation": ["forward_reconstruction"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def _expansion(
    product: str,
    precursor: str,
    *,
    source_ref: str,
    depth: int,
    candidate_id: str,
) -> dict:
    channel = "codex_literature" if source_ref.startswith("doi:") else "codex_strategy"
    consensus = fuse_route_candidates(
        [
            _candidate(
                candidate_id,
                product=product,
                precursors=[precursor],
                source_ref=source_ref,
                channel=channel,
            )
        ],
        target_smiles=product,
    )
    return make_route_consensus_expansion(
        consensus,
        requested_product_smiles=product,
        consensus_ref=f"consensus:{candidate_id}",
        agent_run_ref=f"agent:{candidate_id}",
        depth=depth,
    )


def test_two_one_step_expansions_assemble_into_forward_multistep_route() -> None:
    root = _expansion("OCC", "CC=O", source_ref="doi:10.1000/root", depth=0, candidate_id="root")
    middle = _expansion("CC=O", "C", source_ref="doi:10.1000/middle", depth=1, candidate_id="middle")

    graph = assemble_route_consensus_graph(
        [root, middle],
        case_id="ethanol",
        target_smiles="CCO",
        max_depth=3,
    )

    assert graph["schema_version"] == "route_consensus_graph.v1"
    assert graph["has_hypotheses"] is True
    assert len(graph["nodes"]) == 3
    assert len(graph["steps"]) == 2
    route = graph["route_hypotheses"][0]
    assert len(route["retrosynthetic_step_ids"]) == 2
    assert route["forward_step_ids"] == list(reversed(route["retrosynthetic_step_ids"]))
    assert len(route["forward_dependencies"]) == 1
    assert route["solved"] is False
    assert route["executable"] is False

    steps = {row["step_id"]: row for row in graph["steps"]}
    root_step = next(row for row in steps.values() if row["product_smiles"] == "CCO")
    middle_step = next(row for row in steps.values() if row["product_smiles"] == "CC=O")
    assert root_step["source_refs"] == ["doi:10.1000/root"]
    assert middle_step["source_refs"] == ["doi:10.1000/middle"]
    assert "consensus:root" not in root_step["source_refs"]


def test_unexpanded_precursor_is_selected_as_next_codex_frontier() -> None:
    graph = assemble_route_consensus_graph(
        [_expansion("CCO", "CC=O", source_ref="source:model", depth=0, candidate_id="root")],
        case_id="ethanol",
        target_smiles="CCO",
        max_depth=3,
    )

    frontier = select_route_consensus_frontier(graph)

    assert len(frontier) == 1
    assert frontier[0]["target_smiles"] == "CC=O"
    assert frontier[0]["depth"] == 1
    assert frontier[0]["reason"] == "unexpanded"


def test_cycle_is_audited_and_cut_from_route_enumeration() -> None:
    graph = assemble_route_consensus_graph(
        [
            _expansion("CCO", "CC=O", source_ref="source:root", depth=0, candidate_id="root"),
            _expansion("CC=O", "OCC", source_ref="source:cycle", depth=1, candidate_id="cycle"),
        ],
        case_id="cycle",
        target_smiles="CCO",
        max_depth=4,
    )

    assert graph["cycles"]
    assert len(graph["route_hypotheses"]) == 1
    assert len(graph["route_hypotheses"][0]["retrosynthetic_step_ids"]) == 1
    assert graph["semantics"]["advisory_only"] is True


def test_stereoisomeric_intermediates_do_not_join() -> None:
    root = _expansion("CCO", "C[C@H](O)F", source_ref="source:root", depth=0, candidate_id="root")
    wrong_stereo = _expansion(
        "C[C@@H](O)F",
        "CC(=O)F",
        source_ref="source:wrong-stereo",
        depth=1,
        candidate_id="wrong",
    )

    graph = assemble_route_consensus_graph(
        [root, wrong_stereo],
        case_id="stereo",
        target_smiles="CCO",
        max_depth=3,
    )

    route = graph["route_hypotheses"][0]
    assert len(route["retrosynthetic_step_ids"]) == 1
    assert route["frontier"][0]["reason"] == "unexpanded"


def test_route_forest_projects_multistep_graph_in_forward_order_with_step_scoped_sources() -> None:
    graph = assemble_route_consensus_graph(
        [
            _expansion("CCO", "CC=O", source_ref="doi:10.1000/root", depth=0, candidate_id="root"),
            _expansion("CC=O", "C", source_ref="doi:10.1000/middle", depth=1, candidate_id="middle"),
        ],
        case_id="ethanol",
        target_smiles="CCO",
        max_depth=3,
    )
    forest = compile_explored_route_forest(
        {
            "case_id": "ethanol",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "route_consensus_graph": graph,
            "artifact_refs": {"route_consensus_graph": "D:/runs/ethanol/route_consensus_graph.json"},
        }
    )

    branch = next(row for row in forest["branches"] if row["kind"] == "route_consensus_graph")
    steps = {row["step_id"]: row for row in forest["steps"]}
    rendered = [steps[step_id] for step_id in branch["step_ids"]]
    assert [row["graph_step_id"] for row in rendered] == graph["route_hypotheses"][0]["forward_step_ids"]
    assert rendered[0]["source_refs"] == ["doi:10.1000/middle"]
    assert rendered[1]["source_refs"] == ["doi:10.1000/root"]
    assert "D:/runs/ethanol/route_consensus_graph.json" not in rendered[0]["source_refs"]
    assert branch["route_level_source_refs"] == ["D:/runs/ethanol/route_consensus_graph.json"]
    assert branch["advisory_only"] is True
    assert branch["solved"] is False
    assert branch["executable"] is False
    assert forest["route_consensus_graph"]["route_count"] == 1
    assert "Codex multi-step hypothesis" in render_route_forest_html(forest)
