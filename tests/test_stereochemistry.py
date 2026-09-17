import random

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from cascade_planner.application.chemistry_inspection import inspect_mapped_smiles
from cascade_planner.application.stereochemistry import canonical_stereo_smiles
from cascade_planner.application.reactionjson_replay import (
    ReactionJsonReplayError,
    replay_reactionjson,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    _boundary_stereo_mismatch_atom_maps,
)


def _mapped(smiles, seed):
    mol = Chem.MolFromSmiles(smiles)
    numbers = list(range(101, 101 + mol.GetNumAtoms()))
    random.Random(seed).shuffle(numbers)
    for atom, number in zip(mol.GetAtoms(), numbers):
        atom.SetAtomMapNum(number)
    return mol, numbers


def _independent_labels(mol):
    probe = Chem.Mol(mol)
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    rdCIPLabeler.AssignCIPLabels(probe)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in probe.GetAtoms() if a.HasProp("_CIPCode")}


@pytest.mark.parametrize(
    "smiles",
    [
        "CC[C@H](C)O",
        "C[C@H](O)C(=O)O",
        "[2H][C@](F)(Cl)Br",
        "N[C@@H](CO)[C@H](O)C(=O)O",
        "C[C@@H]1CCCCO1",
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 23, 91])
def test_inspection_cip_is_invariant_to_maps_and_atom_order(smiles, seed):
    mol, maps = _mapped(smiles, seed)
    expected = {maps[i]: label for i, label in _independent_labels(mol).items()}
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed + 100).shuffle(order)
    permuted = Chem.RenumberAtoms(mol, order)
    result = inspect_mapped_smiles(Chem.MolToSmiles(permuted, canonical=False))
    assert {c["map_idx"]: c["cip"] for c in result["centers"]} == expected
    assert [a.GetAtomMapNum() for a in mol.GetAtoms()] == maps


@pytest.mark.parametrize("seed", [0, 1, 23, 91])
@pytest.mark.parametrize("configuration", ["R", "S"])
def test_absolute_assignment_matches_independent_cip(configuration, seed):
    mol, maps = _mapped("CCC(C)O", seed)
    audit = replay_reactionjson(
        mapped_product_smiles=Chem.MolToSmiles(mol, canonical=False),
        operations=[
            {"op": "set_tetrahedral_stereo", "map_idx": maps[2], "configuration": configuration}
        ],
    )
    result = Chem.MolFromSmiles(audit["precursor_smiles"][0])
    assert list(_independent_labels(result).values()) == [configuration]


def test_atom_maps_cannot_create_a_stereocenter_on_identical_substituents():
    mol, maps = _mapped("CC(C)O", 19)
    with pytest.raises(ReactionJsonReplayError, match="tetrahedral_stereo_not_assignable"):
        replay_reactionjson(
            mapped_product_smiles=Chem.MolToSmiles(mol),
            operations=[{"op": "set_tetrahedral_stereo", "map_idx": maps[1], "configuration": "R"}],
        )


@pytest.mark.parametrize("seed", [0, 1, 23, 91])
@pytest.mark.parametrize("configuration", ["E", "Z"])
def test_alkene_assignment_uses_chemical_priorities(configuration, seed):
    mol, maps = _mapped("CCC(C)=C(CC)CCC", seed)
    audit = replay_reactionjson(
        mapped_product_smiles=Chem.MolToSmiles(mol, canonical=False),
        operations=[
            {"op": "set_bond_stereo", "map_a": maps[2], "map_b": maps[4], "stereo": configuration}
        ],
    )
    result = Chem.MolFromSmiles(audit["precursor_smiles"][0])
    rdCIPLabeler.AssignCIPLabels(result)
    double = next(b for b in result.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    assert double.GetProp("_CIPCode") == configuration


def test_repair_boundary_compares_stereo_across_independent_map_namespaces():
    old, old_maps = _mapped("CC[C@H](C)O", 0)
    new, new_maps = _mapped("CC[C@H](C)O", 91)
    translation = dict(zip(old_maps, new_maps))
    assert (
        _boundary_stereo_mismatch_atom_maps(
            Chem.MolToSmiles(old),
            Chem.MolToSmiles(new),
            translation,
        )
        == ()
    )
    new.GetAtomWithIdx(2).InvertChirality()
    assert _boundary_stereo_mismatch_atom_maps(
        Chem.MolToSmiles(old),
        Chem.MolToSmiles(new),
        translation,
    ) == (old_maps[2],)


def test_stereo_replay_cache_requires_current_label_semantics():
    from cascade_planner.orchestration.sequential_strategy_director import (
        _step_has_bound_replay_audit,
    )

    operations = [{"op": "set_tetrahedral_stereo", "map_idx": 3, "configuration": "R"}]
    product = "[CH3:1][CH2:2][CH:3]([CH3:4])[OH:5]"
    audit = replay_reactionjson(mapped_product_smiles=product, operations=operations)
    step = {
        "mapped_product_smiles": product,
        "reaction_operations": operations,
        "precursor_smiles": audit["precursor_smiles"],
        "mapped_precursor_smiles": audit["mapped_precursor_smiles"],
        "reactionjson_audit": audit,
    }
    assert _step_has_bound_replay_audit(step)
    step["reactionjson_audit"] = {
        key: value for key, value in audit.items() if key != "stereochemistry_version"
    }
    assert not _step_has_bound_replay_audit(step)


@pytest.mark.parametrize("mapped,unmapped", [
    ("[CH3:1][C@H:2]([CH3:3])[OH:4]", "CC(C)O"),
    ("[CH2:1]1[C@H:2]([OH:3])[CH2:4][CH2:5][CH2:6]1", "OC1CCCC1"),
    (
        "[CH3:1][C:2]1([CH3:3])[O:4][B:5]([C@@H:6]2[CH2:7][C@:8]3"
        "([c:9]4[cH:10][cH:11][cH:12][cH:13][cH:14]4)[CH2:15][C@@H:16]2"
        "[CH2:17]3)[O:18][C:19]1([CH3:20])[CH3:21]",
        "CC1(C)OB([C@@H]2C[C@]3(c4ccccc4)C[C@@H]2C3)OC1(C)C",
    ),
])
@pytest.mark.parametrize("seed", [0, 23, 91])
def test_chemical_identity_cleans_map_induced_stereo_under_permutation(mapped, unmapped, seed):
    molecule = Chem.MolFromSmiles(mapped)
    order = list(range(molecule.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    permuted = Chem.RenumberAtoms(molecule, order)
    for index, atom in enumerate(permuted.GetAtoms(), 101):
        atom.SetAtomMapNum(index)
    mapped_variant = Chem.MolToSmiles(permuted, canonical=False)
    assert canonical_stereo_smiles(mapped_variant) == canonical_stereo_smiles(unmapped)
    assert all(a.GetAtomMapNum() > 0 for a in permuted.GetAtoms())


@pytest.mark.parametrize("left,right", [
    ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
    ("F/C=C/F", "F/C=C\\F"),
    ("[13CH3]CO", "CCO"),
    ("C[C@H](O)C(=O)O", "CC(O)C(=O)O"),
    ("CCO", "COC"),
    ("CC(=O)[O-]", "CC(=O)O"),
    (
        "CC1(C)OB([C@@H]2C[C@]3(c4ccccc4)C[C@@H]2C3)OC1(C)C",
        "CC1(C)OB([C@H]2C[C@]3(c4ccccc4)C[C@@H]2C3)OC1(C)C",
    ),
])
def test_shared_route_identity_preserves_real_chemical_differences(left, right):
    from cascade_planner.application.route_review_context import _canonical_smiles as review_id
    from cascade_planner.application.routejson_compiler import _canonical_smiles as compiler_id
    from cascade_planner.application.reactionjson_replay import _canonical_smiles as replay_id

    for identify in (review_id, compiler_id, replay_id):
        assert identify(left) != identify(right)


def test_cip_letter_change_does_not_require_an_inversion_operation():
    # Exchange only isotope priorities; the local tetrahedral tag is untouched.
    product = '[CH3:1][C@H:2]([13CH3:3])[OH:4]'
    before = inspect_mapped_smiles(product)['centers'][0]['cip']
    audit = replay_reactionjson(mapped_product_smiles=product, operations=[
        {'op': 'change_atom', 'map_idx': 1, 'isotope': 13},
        {'op': 'change_atom', 'map_idx': 3, 'isotope': 0},
    ])
    after = inspect_mapped_smiles(audit['mapped_precursor_smiles'][0])['centers'][0]['cip']
    assert {before, after} == {'R', 'S'}
    assert audit['primitive_counts']['invert_stereocenter'] == 0


@pytest.mark.parametrize('seed', [0, 23, 91])
def test_absolute_intent_after_ligand_replacement_is_independent_of_atom_order(seed):
    mol, maps = _mapped('C[C@H](O)C(=O)O', seed)
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    mol = Chem.RenumberAtoms(mol, order)
    operations = [
        {'op': 'remove_group', 'map_indices': [maps[2]]},
        {'op': 'add_group', 'map_idx': maps[1], 'fragment_smiles': '[*]Br'},
        {'op': 'set_tetrahedral_stereo', 'map_idx': maps[1], 'configuration': 'R'},
    ]
    audit = replay_reactionjson(mapped_product_smiles=Chem.MolToSmiles(mol, canonical=False), operations=operations)
    assert list(_independent_labels(Chem.MolFromSmiles(audit['precursor_smiles'][0])).values()) == ['R']
