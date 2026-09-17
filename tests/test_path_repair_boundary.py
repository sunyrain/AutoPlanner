from __future__ import annotations

import pytest

from cascade_planner.application.route_review_context import _canonical_smiles
from cascade_planner.orchestration.path_repair_boundary import (
    _path_repair_frontier_reaches_boundaries,
)


@pytest.mark.parametrize("offset", [0, 100, 700])
def test_terminal_repair_can_change_atom_sources_without_changing_the_molecule(offset):
    old = f"[CH3:{1+offset}][C:{2+offset}](=[O:{3+offset}])[O:{4+offset}][CH3:{5+offset}]"
    new = f"[CH3:{1+offset}][C:{2+offset}](=[O:{4+offset}])[O:{3+offset}][CH3:{5+offset}]"
    boundary = dict(product_smiles="COC(C)=O", mapped_product_smiles=old)
    for kind, expected in (
        ("removed_terminal_open_precursor", True),
        ("preserved_suffix_entry", False),
        ("preserved_durable_open_precursor", False),
        ("final_open_precursor", False),
        ("", False),
    ):
        assert _path_repair_frontier_reaches_boundaries(
            product_smiles=["COC(C)=O"],
            mapped_product_smiles=[new],
            reconnect_boundaries=[dict(boundary, boundary_kind=kind)],
        ) is expected


@pytest.mark.parametrize("old,new", [
    ("[CH3:1][CH2:2][OH:3]", "[CH3:1][O:3][CH3:2]"),
    ("[CH3:1][C@H:2]([OH:3])[CH2:4][CH3:5]", "[CH3:1][C@@H:2]([OH:3])[CH2:4][CH3:5]"),
    ("[CH3:1][C@H:2]([OH:3])[CH2:4][CH3:5]", "[CH3:1][CH:2]([OH:3])[CH2:4][CH3:5]"),
    ("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]", "[CH3:1]/[CH:2]=[CH:3]\\[CH3:4]"),
    ("[13CH3:1][CH2:2][OH:3]", "[CH3:1][CH2:2][OH:3]"),
    ("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH2:1][OH:3]"),
    ("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH2:2]O"),
])
def test_terminal_repair_does_not_accept_changed_identity_or_invalid_maps(old, new):
    product = _canonical_smiles(old)
    assert not _path_repair_frontier_reaches_boundaries(
        # A stale or incorrect caller-provided product identity cannot hide
        # different mapped connectivity, stereo, isotopes or invalid maps.
        product_smiles=[product], mapped_product_smiles=[new],
        reconnect_boundaries=[dict(
            product_smiles=product, mapped_product_smiles=old,
            boundary_kind="removed_terminal_open_precursor",
        )],
    )
