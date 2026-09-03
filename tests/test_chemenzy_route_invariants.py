from cascade_planner.interfaces.chemenzy_probe_routes import (
    _normalized_routes,
    compile_chemenzy_route_fingerprints,
)


def _route(*steps: object) -> dict[str, object]:
    return {"routes": [{"steps": list(steps)}]}


def test_normalization_preserves_direction_duplicates_stereo_and_connectivity() -> None:
    duplicate = _normalized_routes(
        _route(
            {
                "product": "CC",
                "reactant_smiles": ["C", "C"],
                "rxn_smiles": "C.C>>CC",
            }
        ),
        target_smiles="CC",
    )[0]
    connected = _normalized_routes(
        _route(
            {
                "product": "CC(=O)OCC",
                "reactant_smiles": ["CCO", "CC(=O)Cl"],
                "rxn_smiles": "CCO.CC(=O)Cl>>CC(=O)OCC",
            },
            {
                "product": "CCO",
                "reactant_smiles": ["CC=O", "[H][H]"],
                "rxn_smiles": "CC=O.[H][H]>>CCO",
            },
        ),
        target_smiles="CC(=O)OCC",
    )[0]
    stereo = _normalized_routes(
        _route(
            {
                "product": "C[C@H](O)F",
                "reactant_smiles": ["C[C@H](O)Cl", "F"],
                "rxn_smiles": "C[C@H](O)Cl.F>>C[C@H](O)F",
            }
        ),
        target_smiles="C[C@H](O)F",
    )[0]

    assert duplicate["steps"][0]["reactant_smiles"] == ["C", "C"]
    assert duplicate["normalization_audit"][
        "reactant_order_and_multiplicity_preserved"
    ] is True
    assert connected["normalization_audit"][
        "route_step_connectivity_preserved"
    ] is True
    assert stereo["steps"][0]["product_smiles"] == "C[C@H](O)F"
    assert stereo["normalization_audit"][
        "stereochemical_identity_preserved"
    ] is True
    assert all(
        route["proposal_eligible"] is True
        for route in (duplicate, connected, stereo)
    )


def test_reversed_reaction_smiles_fails_closed() -> None:
    route = _normalized_routes(
        _route(
            {
                "product": "CC(=O)OCC",
                "reactant_smiles": ["CCO", "CC(=O)Cl"],
                "rxn_smiles": "CC(=O)OCC>>CCO.CC(=O)Cl",
            }
        ),
        target_smiles="CC(=O)OCC",
    )[0]

    assert route["proposal_eligible"] is False
    assert route["admission_reasons"] == ["reaction_smiles_direction_mismatch"]
    assert route["normalization_audit"]["reaction_direction_consistent"] is False


def test_reaction_smiles_must_retain_precursor_multiplicity() -> None:
    route = _normalized_routes(
        _route(
            {
                "product": "CC",
                "reactant_smiles": ["C", "C"],
                "rxn_smiles": "C>>CC",
            }
        ),
        target_smiles="CC",
    )[0]

    assert route["steps"][0]["reactant_smiles"] == ["C", "C"]
    assert route["proposal_eligible"] is False
    assert "reaction_smiles_reactant_mismatch" in route["admission_reasons"]


def test_reaction_smiles_must_retain_stereochemistry() -> None:
    route = _normalized_routes(
        _route(
            {
                "product": "C[C@H](O)F",
                "reactant_smiles": ["C[C@H](O)Cl", "F"],
                "rxn_smiles": "C[C@H](O)Cl.F>>CC(O)F",
            }
        ),
        target_smiles="C[C@H](O)F",
    )[0]

    assert route["proposal_eligible"] is False
    assert "reaction_smiles_stereochemistry_mismatch" in route["admission_reasons"]
    assert route["normalization_audit"][
        "stereochemical_identity_preserved"
    ] is False


def test_disconnected_or_dropped_steps_fail_closed_with_raw_counts() -> None:
    disconnected = _normalized_routes(
        _route(
            {
                "product": "CC(=O)OCC",
                "reactant_smiles": ["CCO", "CC(=O)Cl"],
            },
            {"product": "CCN", "reactant_smiles": ["CC", "N"]},
        ),
        target_smiles="CC(=O)OCC",
    )[0]
    dropped = _normalized_routes(
        _route(
            {
                "product": "CC(=O)OCC",
                "reactant_smiles": ["CCO", "CC(=O)Cl"],
            },
            "not-a-step",
        ),
        target_smiles="CC(=O)OCC",
    )[0]

    assert "disconnected_route_steps" in disconnected["admission_reasons"]
    assert dropped["raw_step_count"] == 2
    assert len(dropped["steps"]) == 1
    assert {"invalid_step_payload", "raw_normalized_step_count_mismatch"} <= set(
        dropped["admission_reasons"]
    )


def test_fingerprint_binds_normalization_audit_digest() -> None:
    fingerprints = compile_chemenzy_route_fingerprints(
        _route(
            {
                "product": "CC",
                "reactant_smiles": ["C", "C"],
                "rxn_smiles": "C.C>>CC",
            }
        ),
        target_smiles="CC",
    )

    row = fingerprints["routes"][0]
    assert row["raw_step_count"] == 1
    assert row["normalization_invariants_accepted"] is True
    assert len(row["normalization_audit_sha256"]) == 64
