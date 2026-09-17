from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from cascade_planner.application.reaction_inputs import (
    reaction_input_smiles,
    mapped_reaction_input_smiles,
)
from cascade_planner.application.reaction_proof_versions import (
    active_reaction_proofs,
    CURRENT_REACTION_VALIDATOR_VERSION,
)
from cascade_planner.application.route_review_context import (
    compile_revision_bound_route_critic_context,
)
from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.application.reactionjson_replay import ReactionJsonReplayError
from cascade_planner.interfaces.target_solver_stages import (
    validate_materialized_edges,
    repair_rejected_precursor_typos,
)
from cascade_planner.orchestration.retrosynthesis_service import RetrosynthesisCampaignService
from cascade_planner.orchestration.sequential_strategy_director import (
    compile_frontier_builder_context,
    _critic_step_row,
    _paper_critic_step_row,
    _route_atom_map_namespace,
)
from cascade_planner.application.run_kernel import RunKernel, RunSpec, RunLimits
from cascade_planner.application.retrosynthesis_run_contract import RetrosynthesisRunBudget


@pytest.mark.parametrize("auxiliary_map", [41, 106])
@pytest.mark.parametrize("legacy_audit", [False, True])
def test_builder_fresh_atoms_replay_after_auxiliary_input_leaves_frontier(
    auxiliary_map, legacy_audit
):
    compiler = RouteJSONCompiler()
    target = "[CH4:1]"
    prefix = compiler.compile_route_graph(
        mapped_target_smiles=target,
        steps=[
            {
                "product_smiles": "C",
                "reaction_operations": [
                    {"op": "set_explicit_h", "map_idx": 1, "count": 3},
                    {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Cu:36]"},
                ],
            },
            {
                "product_smiles": "C[Cu]",
                "reaction_operations": [
                    {"op": "break_bond", "map_a": 1, "map_b": 36},
                    {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Li:40]"},
                    {"op": "add_group", "map_idx": 36, "fragment_smiles": f"*[I:{auxiliary_map}]"},
                ],
            },
        ],
    )
    steps = [dict(asdict(row), reactionjson_audit=dict(row.audit)) for row in prefix]
    assert prefix[-1].precursor_smiles == ("[Li]C",)
    assert prefix[-1].auxiliary_reagent_smiles == ("[Cu]I",)
    if legacy_audit:
        for step in steps:
            step.pop("mapped_reaction_input_smiles")
            audit = step["reactionjson_audit"]
            audit["mapped_precursor_smiles"] = audit.pop("mapped_reaction_input_smiles")

    namespace = _route_atom_map_namespace(steps)
    selected = prefix[-1].mapped_precursor_smiles[0]
    operations = [
        {"op": "remove_group", "map_indices": [40]},
        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*O"},
    ]
    next_step = compiler.compile_step(
        mapped_product_smiles=selected,
        operations=operations,
        reserved_atom_maps=namespace,
        target_atom_maps={1},
    )
    replayed = compiler.compile_route_graph(
        mapped_target_smiles=target, steps=steps + [asdict(next_step)]
    )
    assert replayed[-1].precursor_smiles == ("CO",)
    assert replayed[-1].mapped_precursor_smiles == next_step.mapped_precursor_smiles

    # Actual identity reuse still fails at allocation; do not weaken replay.
    with pytest.raises(ReactionJsonReplayError, match="reactionjson_fragment_map_collision"):
        compiler.compile_step(
            mapped_product_smiles=selected,
            operations=[operations[0], {**operations[1], "fragment_smiles": f"*[OH:{auxiliary_map}]"}],
            reserved_atom_maps=namespace,
            target_atom_maps={1},
        )


def _kernel(tmp_path):
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="participant-contract",
            target_name="fixture",
            target_smiles="C[Cu]",
            created_at="2026-09-05T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=8, max_accepted_expansions=16, max_attempt_runs=64
                ),
                max_total_tasks=64,
                max_validation_tasks=16,
            ),
        ),
    )
    kernel.start()
    return kernel


def _graph():
    return {
        "revision": 1,
        "target_molecule_id": "target",
        "molecules": {"target": {"canonical_smiles": "CC=O"}, "leaf": {"canonical_smiles": "CCO"}},
        "route_families": {
            "route": {
                "route_family_id": "route",
                "selected": True,
                "aliases": ["codex:sequential:family:1"],
                "edge_ids": ["edge"],
            }
        },
        "edges": {
            "edge": {
                "edge_id": "edge",
                "product_molecule_id": "target",
                "product_smiles": "CC=O",
                "precursor_molecule_ids": ["leaf"],
                "precursor_smiles": ["CCO"],
                "reaction_input_smiles": ["[Na+]", "CCO"],
                "mapped_product_smiles": "[CH3:1][CH:2]=[O:3]",
                "mapped_reaction_input_smiles": ["[Na+:4]", "[CH3:1][CH2:2][OH:3]"],
            }
        },
    }


@pytest.mark.parametrize("proof_fallback", [False, True])
@pytest.mark.parametrize("reader", ["critic", "frontier"])
@pytest.mark.parametrize("extra", [[], ["[Na+:5]"]])
def test_review_and_frontier_keep_complete_multiset_but_only_expand_route_precursors(
    reader, extra, proof_fallback
):
    graph = _graph()
    edge = graph["edges"]["edge"]
    edge["mapped_reaction_input_smiles"] += extra
    edge["reaction_input_smiles"] += ["[Na+]"] * len(extra)
    expected_mapped = list(edge["mapped_reaction_input_smiles"])
    if proof_fallback:
        edge["reaction_proofs"] = [
            {
                "accepted": True,
                "validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
                "mapped_reaction": ".".join(edge.pop("mapped_reaction_input_smiles"))
                + ">>"
                + edge.pop("mapped_product_smiles"),
                "checks": {
                    key: True
                    for key in (
                        "structures_materialized",
                        "mapped_reaction_present",
                        "mapped_product_matches",
                        "mapped_reactants_match",
                        "atom_maps_complete",
                        "product_atom_maps_complete",
                        "atom_maps_unique",
                        "mapped_elements_preserved",
                        "stereochemical_product_matches",
                    "product_atoms_have_reactant_provenance",
                    )
                },
            }
        ]
    if reader == "critic":
        context, diagnostic = compile_revision_bound_route_critic_context(
            graph, route_family_id="route"
        )
        assert diagnostic == {}
        step = context.steps[0]
    else:
        context, diagnostic = compile_frontier_builder_context(
            graph, frontier_molecule_id="leaf", route_family_ids=("route",)
        )
        assert diagnostic == {}
        step = context.connected_steps[0]
    assert step["precursor_smiles"] == ["CCO"]
    assert step["mapped_precursor_smiles"] == ["[CH3:1][CH2:2][OH:3]"]
    assert step["auxiliary_reagent_smiles"] == ["[Na+]"] * (1 + len(extra))
    for level in (0, 1, 2, 3):
        for project in (_critic_step_row, _paper_critic_step_row):
            prompt = project(step, compact_level=level)
            assert sorted(prompt["mapped_reaction_input_smiles"]) == sorted(expected_mapped)


@pytest.mark.parametrize("reader", ["critic", "frontier"])
@pytest.mark.parametrize(
    "mapped",
    [
        ["[CH3:1][CH2:2][OH:3]"],
        ["[CH3:1][CH2:2][OH:3]", "[K+:4]"],
        ["[CH3:1][CH2:2][OH:3]", "[Na+:4]", "[Na+:5]"],
    ],
)
def test_mismatched_or_unexplained_participants_still_rejected(reader, mapped):
    graph = _graph()
    graph["edges"]["edge"]["mapped_reaction_input_smiles"] = mapped
    if reader == "critic":
        context, diagnostic = compile_revision_bound_route_critic_context(
            graph, route_family_id="route"
        )
    else:
        context, diagnostic = compile_frontier_builder_context(
            graph, frontier_molecule_id="leaf", route_family_ids=("route",)
        )
    assert context is None
    assert diagnostic["reason"].endswith("incomplete")


def test_legacy_audit_and_unsplit_input_fallbacks():
    assert reaction_input_smiles({"precursor_smiles": ["CCO"]}) == ["CCO"]
    row = {
        "precursor_smiles": ["CCO"],
        "reactionjson_audit": {
            "precursor_smiles": ["CCO", "[Na+]"],
            "mapped_precursor_smiles": ["[Na+:4]", "[CH3:1][CH2:2][OH:3]"],
        },
    }
    assert reaction_input_smiles(row) == ["CCO", "[Na+]"]
    assert len(mapped_reaction_input_smiles(row)) == 2
    assert (
        active_reaction_proofs(
            [{"validator_version": "autoplanner.reaction_step_verifier.v11", "accepted": True}]
        )
        == []
    )


def test_compiler_materialization_mapping_worker_and_graph_use_full_inputs(tmp_path):
    compiled = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][Cu:36]",
        operations=[
            {"op": "break_bond", "map_a": 1, "map_b": 36},
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Li:40]"},
            {"op": "add_group", "map_idx": 36, "fragment_smiles": "*[I:41]"},
        ],
        expected_product_smiles="C[Cu]",
        target_atom_maps={1},
    )
    service = RetrosynthesisCampaignService(_kernel(tmp_path))
    proposal = asdict(compiled)
    proposal.update(
        {
            "reactionjson_audit": compiled.audit,
            "origin_kind": "manual",
            "proposal_id": "generic-transmetalation",
            "route_family_id": "route",
        }
    )
    ingestion = service.execute_commands(
        service.graph_store.materialization_commands([proposal]),
        idempotency_key="materialize",
        include_scheduled=False,
    )
    graph = service.graph_store.load()
    assert len(graph["edges"]) == 1, ingestion
    edge = next(iter(graph["edges"].values()))
    assert edge["precursor_smiles"] == ["[Li]C"]
    seen = []

    def mapper(reactions):
        seen.extend(reactions)
        return [
            ".".join(compiled.mapped_reaction_input_smiles) + ">>" + compiled.mapped_product_smiles
            for _ in reactions
        ]

    result = validate_materialized_edges(service, atom_mapper=mapper)
    assert seen == []  # Existing complete Host maps do not need re-inference.
    assert result["mapping"]["backend"] == "host_replay"
    edge = next(iter(service.graph_store.load()["edges"].values()))
    proof = edge["reaction_proofs"][-1]
    assert proof["validator_version"] == CURRENT_REACTION_VALIDATOR_VERSION
    assert proof["checks"]["mapped_reactants_match"] is True
    assert proof["checks"]["product_atom_maps_complete"] is True
    assert edge["precursor_smiles"] == ["[Li]C"]
    assert len(edge["precursor_molecule_ids"]) == 1
    assert result["mapping"]["mapped_count"] == 1
    assert service.kernel.state.model_totals["model_invocations"] == 0


@pytest.mark.parametrize("supply_donor", [False, True])
def test_silylation_proof_survives_worker_json_and_graph_ingestion(tmp_path, supply_donor):
    from cascade_planner.application.canonical_hypergraph import _valid_proof

    operations = (
        [{"op": "break_bond", "map_a": 2, "map_b": 14},
         {"op": "add_group", "map_idx": 14, "fragment_smiles": "[*]Cl"}]
        if supply_donor else [{"op": "remove_group", "map_indices": [14, 6, 7, 8]}]
    )
    compiled = RouteJSONCompiler().compile_step(
        mapped_product_smiles="[CH3:1][O:2][Si:14]([CH3:6])([CH3:7])[CH3:8]",
        operations=operations, target_atom_maps={1, 2},
    )
    service = RetrosynthesisCampaignService(_kernel(tmp_path))
    proposal = {**asdict(compiled), "reactionjson_audit": compiled.audit,
                "origin_kind": "manual", "proposal_id": "silylation", "route_family_id": "route"}
    service.execute_commands(service.graph_store.materialization_commands([proposal]),
                             idempotency_key="materialize", include_scheduled=False)

    def unused_mapper(_):
        raise AssertionError("Host atom identities must survive validation")

    result = validate_materialized_edges(service, atom_mapper=unused_mapper)
    edge = next(iter(service.graph_store.load()["edges"].values()))
    proof = edge["reaction_proofs"][-1]
    assert _valid_proof(json.loads(json.dumps(proof)))
    assert not _valid_proof({**proof, "proof_digest": "0" * 64})
    assert proof["checks"]["product_atoms_have_reactant_provenance"] is supply_donor
    assert edge["precursor_smiles"] == ["CO"]
    if supply_donor:
        assert edge["auxiliary_reagent_smiles"] == ["C[Si](C)(C)Cl"]
    else:
        assert not proof["accepted"]
        assert proof["external_atom_source_audit"]["product_inventory_deficit"] == {"6": 3, "14": 1}
        assert "product_heavy_atom_without_reactant_provenance" in result["rejection_reason_counts"]
        assert "reaction_validation_not_accepted" not in result["rejection_reason_counts"]


def test_stale_host_mapping_uses_fallback_without_hiding_missing_donors():
    from cascade_planner.interfaces.target_solver_stages import _host_mapping_for_validation

    edge = _graph()["edges"]["edge"]
    assert _host_mapping_for_validation(edge)
    edge["mapped_product_smiles"] = "[CH3:1][CH2:2][OH:3]"
    assert _host_mapping_for_validation(edge) == ""
    edge["mapped_product_smiles"] = "[CH3:1][CH:2]=[O:3]"
    edge["mapped_reaction_input_smiles"] = ["[Na+:1]", "[CH3:1][CH2:2][OH:3]"]
    assert _host_mapping_for_validation(edge) == ""


def test_host_ester_cleavage_keeps_product_maps_without_counting_departing_group_bonds():
    from cascade_planner.interfaces.target_solver_stages import _host_mapping_for_validation
    from cascade_planner.harness.reaction_step_verifier import verify_reaction_step

    edge = {
        "product_smiles": "CC(=O)O", "reaction_input_smiles": ["CC(=O)OC(C)(C)C"],
        "mapped_product_smiles": "[CH3:1][C:2](=[O:3])[OH:4]",
        "mapped_reaction_input_smiles": ["[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])([CH3:7])[CH3:8]"],
    }
    mapped = _host_mapping_for_validation(edge)
    assert mapped.endswith(">>" + edge["mapped_product_smiles"])
    proof = verify_reaction_step({"product_smiles": edge["product_smiles"],
                                 "reactant_smiles": edge["reaction_input_smiles"],
                                 "mapped_reaction_smiles": mapped})
    assert proof["accepted"]
    assert proof["bond_change_audit"]["bond_edit_count"] == 1


def test_precursor_repair_preserves_auxiliary_multiset_outside_route_frontier():
    edge = {
        "edge_id": "edge",
        "product_smiles": "CN1CCC[C@@H](N)C1",
        "precursor_smiles": ["CBr", "N[C@@H]1CCCNCC1"],
        "reaction_input_smiles": ["CBr", "N[C@@H]1CCCNCC1", "[Na+]", "[Na+]"],
    }
    mapped = (
        "Br[CH3:1].C1[NH:2][CH2:3][CH2:4][CH2:5][C@@H:6]([NH2:8])[CH2:7]1.[Na+:9].[Na+:10]>>"
        "[CH3:1][N:2]1[CH2:3][CH2:4][CH2:5][C@@H:6]([NH2:8])[CH2:7]1"
    )
    commands_seen = []

    def execute(commands, **kwargs):
        commands_seen.extend(commands)
        return {"executed_command_count": len(commands)}

    service = SimpleNamespace(
        graph_store=SimpleNamespace(load=lambda: {"edges": {"edge": edge}}),
        kernel=SimpleNamespace(
            spec=SimpleNamespace(run_id="repair"),
            state=SimpleNamespace(graph_revision=1, evidence_revision=0),
        ),
        execute_commands=execute,
    )
    reaction = ".".join(edge["reaction_input_smiles"]) + ">>" + edge["product_smiles"]
    result = repair_rejected_precursor_typos(
        service,
        {
            "rejected_edge_ids": ["edge"],
            "mapping": {"mapped_reactions": {reaction: mapped}},
        },
    )
    assert result["accepted_repair_count"] == 1
    assert len(commands_seen) == 1
    candidate = commands_seen[0].payload
    assert candidate["precursor_smiles"] == ["CBr", "N[C@@H]1CCCNC1"]
    assert candidate["reaction_input_smiles"].count("[Na+]") == 2
    assert candidate["auxiliary_reagent_smiles"] == ["[Na+]", "[Na+]"]
    assert edge["precursor_smiles"] == ["CBr", "N[C@@H]1CCCNCC1"]
