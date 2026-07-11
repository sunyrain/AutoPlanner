from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def test_admission_preserves_precursor_multiplicity_and_allows_balanced_edge() -> None:
    audit = audit_retrosynthetic_candidate("CC", ["C", "C"])

    assert audit["accepted"] is True
    assert audit["precursor_smiles"] == ["C", "C"]
    assert audit["precursor_element_counts"] == {"C": 2}
    assert audit["semantics"]["precursor_multiplicity_preserved"] is True


def test_admission_rejects_large_element_jump_and_ancestor_return() -> None:
    audit = audit_retrosynthetic_candidate(
        "CCCCCCCCCCCCCCCCCCCC",
        ["C", "CCO"],
        forbidden_return_smiles=["CCO"],
    )

    assert audit["accepted"] is False
    assert set(audit["reasons"]) == {
        "ancestor_or_target_cycle",
        "element_inventory_not_conserved",
        "large_atom_jump",
    }


def test_complex_product_keeps_narrow_omitted_transfer_reagent_allowance() -> None:
    audit = audit_retrosynthetic_candidate(
        "CCCCCCCCCCCCCCCO",
        ["CCCCCCCCCCCCCCC"],
    )

    assert audit["accepted"] is True
    assert audit["element_deficits"] == {"O": 1}
    assert audit["missing_product_heavy_atom_count"] == 1
