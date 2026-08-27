from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def test_admission_preserves_precursor_multiplicity_and_allows_balanced_edge() -> None:
    audit = audit_retrosynthetic_candidate("CC", ["C", "C"])
    ordered = audit_retrosynthetic_candidate("CCO", ["CC", "O"])
    reordered = audit_retrosynthetic_candidate("CCO", ["O", "CC"])
    missing_copy = audit_retrosynthetic_candidate("CC", ["C"])

    assert audit["accepted"] is True
    assert audit["precursor_smiles"] == ["C", "C"]
    assert audit["precursor_smiles_multiset"] == ["C", "C"]
    assert audit["precursor_element_counts"] == {"C": 2}
    assert ordered["edge_digest"] == reordered["edge_digest"]
    assert audit["edge_digest"] != missing_copy["edge_digest"]
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


def test_admission_rejects_redundant_advanced_precursor_fragment() -> None:
    audit = audit_retrosynthetic_candidate(
        "CCO",
        ["CC=O", "CCCCCCCCCC"],
    )

    assert audit["accepted"] is False
    assert audit["reasons"] == ["surplus_advanced_precursor_fragment"]
    assert audit["surplus_advanced_precursor_fragments"] == ["CCCCCCCCCC"]


def test_surplus_gate_exempts_small_salts_and_single_precursor_deprotection() -> None:
    salted = audit_retrosynthetic_candidate(
        "CCO",
        ["CC=O", "[Na+]", "[Cl-]"],
    )
    deprotection = audit_retrosynthetic_candidate(
        "N",
        ["CC(C)(C)OC(=O)N"],
    )

    assert salted["accepted"] is True
    assert salted["surplus_advanced_precursor_fragments"] == []
    assert deprotection["accepted"] is True


def test_surplus_gate_keeps_large_precursors_that_contribute_mapped_atoms() -> None:
    wittig = audit_retrosynthetic_candidate(
        "C=C[C@H](C)CCC=C(C)C",
        [
            "CC(C)=CCC[C@@H](C)C=O",
            "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1",
        ],
        mapped_reaction_smiles=(
            r"[CH2:1]=[CH:2][C@H:3]([CH3:4])[CH2:5][CH2:6]/[CH:7]="
            r"[C:8](\[CH3:9])[CH3:10]>>[CH3:1][P+:11]([c:12]1[cH:13]"
            r"[cH:14][cH:15][cH:16][cH:17]1)([c:18]1[cH:19][cH:20]"
            r"[cH:21][cH:22][cH:23]1)[c:24]1[cH:25][cH:26][cH:27]"
            r"[cH:28][cH:29]1.[CH:2]([C@H:3]([CH3:4])[CH2:5][CH2:6]/"
            r"[CH:7]=[C:8](\[CH3:9])[CH3:10])=[O:30]"
        ),
    )
    enol_triflation = audit_retrosynthetic_candidate(
        "CC(=O)[C@H]1CCC=C1OS(=O)(=O)C(F)(F)F",
        [
            "CC(=O)[C@H]1CCCC1=O",
            "O=S(=O)(N(c1ccccc1)S(=O)(=O)C(F)(F)F)C(F)(F)F",
        ],
        mapped_reaction_smiles=(
            "[C:5]1([O:13][S:14]([C:15]([F:16])([F:17])[F:18])"
            "(=[O:19])=[O:20])=[CH:6][CH2:7][CH2:8][C@@H:9]1"
            "[C:10]([CH3:11])=[O:12]>>[C:5]1(=[O:13])[CH2:6][CH2:7]"
            "[CH2:8][C@@H:9]1[C:10]([CH3:11])=[O:12].[S:14]([C:15]"
            "([F:16])([F:17])[F:18])(=[O:19])(=[O:20])[N:21]([S:22]"
            "([C:23]([F:24])([F:25])[F:26])(=[O:27])=[O:28])[c:29]1"
            "[cH:30][cH:31][cH:32][cH:33][cH:34]1"
        ),
    )

    for audit in (wittig, enol_triflation):
        assert audit["accepted"] is True
        assert audit["mapped_reaction_atom_mapping_used"] is True
        assert audit["mapped_atom_contributing_precursor_indices"] == [0, 1]
        assert audit["surplus_advanced_precursor_fragments"] == []


def test_surplus_gate_still_rejects_mapped_noncontributing_fragment() -> None:
    audit = audit_retrosynthetic_candidate(
        "CCO",
        ["CC=O", "CCCCCCCCCC"],
        mapped_reaction_smiles=(
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]."
            "[CH3:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15]"
            "[CH2:16][CH2:17][CH2:18][CH3:19]"
        ),
    )

    assert audit["accepted"] is False
    assert audit["mapped_reaction_atom_mapping_used"] is True
    assert audit["mapped_atom_contributing_precursor_indices"] == [0]
    assert audit["reasons"] == ["surplus_advanced_precursor_fragment"]
    assert audit["surplus_advanced_precursor_fragments"] == ["CCCCCCCCCC"]


def test_surplus_gate_uses_host_replay_mapped_precursors_without_provider_reaction() -> None:
    audit = audit_retrosynthetic_candidate(
        "CCO",
        ["CC=O", "CCCCCCCCCC"],
        reactionjson_audit={
            "accepted": True,
            "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
            "mapped_precursor_smiles": [
                "[CH3:2][CH:4]=[O:3]",
                "[CH3:1][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH3:13]",
            ],
            "semantics": {"deterministic_graph_edit_replay": True},
        },
    )

    assert audit["accepted"] is True
    assert audit["mapped_reaction_atom_mapping_used"] is True
    assert audit["mapped_atom_contributing_precursor_indices"] == [0, 1]
    assert audit["surplus_advanced_precursor_fragments"] == []
