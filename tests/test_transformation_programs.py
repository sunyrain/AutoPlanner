from __future__ import annotations

from copy import deepcopy

from cascade_planner.application.transformation_programs import (
    chemical_state_id,
    operation_id,
    program_id,
    program_projection_oracle,
    project_canonical_graph_to_programs,
)


def _graph() -> dict:
    return {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "program-projection",
        "revision": 4,
        "scientific_sha256": "source-scientific-digest",
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:a": {
                "molecule_id": "molecule:a",
                "canonical_smiles": "CCO",
                "stock_observation_ids": ["stock:a"],
            },
            "molecule:b": {
                "molecule_id": "molecule:b",
                "canonical_smiles": "CC(=O)Cl",
            },
            "molecule:target": {
                "molecule_id": "molecule:target",
                "canonical_smiles": "CCOC(C)=O",
            },
        },
        "edges": {
            "edge:ester": {
                "edge_id": "edge:ester",
                "precursor_molecule_ids": [
                    "molecule:a",
                    "molecule:a",
                    "molecule:b",
                ],
                "product_molecule_id": "molecule:target",
                "source_binding_ids": ["source:paper"],
                "exact_record_ids": ["exact:row"],
                "procedure_record_ids": ["procedure:one"],
                "reaction_proofs": [{"proof_digest": "proof:host"}],
            }
        },
        "route_families": {
            "route:ester": {
                "route_family_id": "route:ester",
                "edge_ids": ["edge:ester"],
                "closed": True,
            }
        },
    }


def test_projection_is_deterministic_read_only_and_preserves_stoichiometry() -> None:
    graph = _graph()
    before = deepcopy(graph)

    first = project_canonical_graph_to_programs(graph)
    second = project_canonical_graph_to_programs(graph)

    assert graph == before
    assert first == second
    assert first["target_state_id"] == chemical_state_id("molecule:target")
    program = first["programs"][program_id("edge:ester")]
    assert program["input_state_ids"] == [
        chemical_state_id("molecule:a"),
        chemical_state_id("molecule:a"),
        chemical_state_id("molecule:b"),
    ]
    assert program["operation_node_ids"] == [operation_id("edge:ester")]
    assert program["validation_vector"]["authoritative"] is False
    assert first["routes"]["route:ester"]["program_ids"] == [
        program_id("edge:ester")
    ]
    assert first["semantics"]["edge_ids_remain_production_route_authority"] is True


def test_projection_oracle_rejects_any_program_topology_drift() -> None:
    graph = _graph()
    projection = project_canonical_graph_to_programs(graph)
    accepted = program_projection_oracle(graph, projection)
    corrupted = deepcopy(projection)
    corrupted["programs"][program_id("edge:ester")]["input_state_ids"] = []
    rejected = program_projection_oracle(graph, corrupted)

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["checks"]["programs_equal"] is False
    assert "programs_equal" in rejected["reasons"]


def test_projection_rejects_noncanonical_sources() -> None:
    try:
        project_canonical_graph_to_programs({"schema_version": "legacy.graph"})
    except ValueError as exc:
        assert str(exc) == "program_projection_requires_canonical_v4_graph"
    else:
        raise AssertionError("legacy graph unexpectedly admitted")
