from cascade_planner.application.precursor_repair import (
    propose_precursor_repair,
    propose_ring_size_typo_repair,
)


def test_product_grounded_repair_contracts_one_unmapped_ring_carbon() -> None:
    result = propose_ring_size_typo_repair(
        product_smiles="CN1CCC[C@@H](N)C1",
        precursor_smiles=["CBr", "N[C@@H]1CCCNCC1"],
        mapped_reaction_smiles=(
            "Br[CH3:1].C1[NH:2][CH2:3][CH2:4][CH2:5]"
            "[C@@H:6]([NH2:8])[CH2:7]1>>"
            "[CH3:1][N:2]1[CH2:3][CH2:4][CH2:5]"
            "[C@@H:6]([NH2:8])[CH2:7]1"
        ),
    )

    assert result["accepted"] is True
    assert result["repaired_component_smiles"] == "N[C@@H]1CCCNC1"
    assert result["repaired_precursor_smiles"] == ["CBr", "N[C@@H]1CCCNC1"]
    assert result["semantics"]["normal_host_revalidation_required"] is True


def test_product_grounded_repair_rejects_unmapped_non_ring_carbon() -> None:
    result = propose_ring_size_typo_repair(
        product_smiles="CNC",
        precursor_smiles=["CBr", "CCN"],
        mapped_reaction_smiles="Br[CH3:1].C[NH:2][CH3:3]>>[CH3:1][NH:2][CH3:3]",
    )

    assert result["accepted"] is False
    assert result["reasons"] == ["unmapped_atom_is_not_one_ring_carbon"]


def test_product_grounded_repair_expands_one_missing_ring_carbon() -> None:
    result = propose_ring_size_typo_repair(
        product_smiles="CN1CCC[C@@H](N)C1",
        precursor_smiles=["CBr", "N[C@@H]1CCCN1"],
        mapped_reaction_smiles=(
            "Br[CH3:1].[NH:2]1[CH2:3][CH2:4][CH2:5]"
            "[C@H:6]1[NH2:8]>>"
            "[CH3:1][N:2]1[CH2:3][CH2:4][CH2:5]"
            "[C@@H:6]([NH2:8])C1"
        ),
    )

    assert result["accepted"] is True
    assert result["repair_kind"] == "single_unmapped_ring_carbon_expansion"
    assert result["repaired_component_smiles"] == "N[C@@H]1CCCNC1"
    assert result["carbon_atom_delta"] == 1


def test_product_grounded_repair_swaps_aryl_thioisocyanate_connectivity() -> None:
    result = propose_precursor_repair(
        product_smiles=(
            "CC1(C)C(=O)NC(=S)N1c1ccc(C#N)c(C(F)(F)F)c1"
        ),
        precursor_smiles=[
            "CC(C)(N)C(=O)O",
            "N#Cc1ccc([SH]=C=N)cc1C(F)(F)F",
        ],
        mapped_reaction_smiles=(
            "O[C:4]([C:2]([CH3:1])([CH3:3])[NH2:9])=[O:5]."
            "[NH:6]=[C:7]=[SH:8][c:10]1[cH:11][cH:12][c:13]"
            "([C:14]#[N:15])[c:16]([C:17]([F:18])([F:19])[F:20])"
            "[cH:21]1>>[CH3:1][C:2]1([CH3:3])[C:4](=[O:5])[NH:6]"
            "[C:7](=[S:8])[N:9]1[c:10]1[cH:11][cH:12][c:13]"
            "([C:14]#[N:15])[c:16]([C:17]([F:18])([F:19])[F:20])"
            "[cH:21]1"
        ),
    )

    assert result["accepted"] is True
    assert result["repair_kind"] == "aryl_isothiocyanate_connectivity_swap"
    assert result["repaired_component_smiles"] == (
        "N#Cc1ccc(N=C=S)cc1C(F)(F)F"
    )
    assert result["semantics"]["normal_host_revalidation_required"] is True


def test_isothiocyanate_repair_requires_exact_product_grounding() -> None:
    result = propose_precursor_repair(
        product_smiles="CCOC(C)=O",
        precursor_smiles=["CCO", "N#Cc1ccc([SH]=C=N)cc1"],
        mapped_reaction_smiles="[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]",
    )

    assert result["accepted"] is False
