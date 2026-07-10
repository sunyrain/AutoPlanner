from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from cascade_planner.harness.preflight import KNOWN_TARGET_IDENTITIES, run_preflight
from cascade_planner.harness.schemas import TargetInput


PACLITAXEL_IDENTITY = KNOWN_TARGET_IDENTITIES["paclitaxel"]
PACLITAXEL_SMILES = PACLITAXEL_IDENTITY["expected_smiles"]
PACLITAXEL_INCHI_KEY = "RCINICONZNJXQF-MZXODVADSA-N"


def test_paclitaxel_reference_smiles_matches_locked_identity() -> None:
    molecule = Chem.MolFromSmiles(PACLITAXEL_SMILES)

    assert molecule is not None
    assert inchi.MolToInchiKey(molecule) == PACLITAXEL_INCHI_KEY
    assert PACLITAXEL_IDENTITY["expected_inchi_key"] == PACLITAXEL_INCHI_KEY


@pytest.mark.parametrize("target_name", ["paclitaxel", "Taxol"])
def test_preflight_accepts_paclitaxel_and_taxol_aliases_with_correct_structure(target_name: str) -> None:
    report = run_preflight(TargetInput(target_name=target_name, target_smiles=PACLITAXEL_SMILES))

    assert report["accepted"] is True
    assert report["inchi_key"] == PACLITAXEL_INCHI_KEY
    assert report["known_target_identity_audit"] == {
        "schema_version": "known_target_identity_audit.v1",
        "target_key": "paclitaxel",
        "accepted": True,
        "observed_inchi_key": PACLITAXEL_INCHI_KEY,
        "expected_inchi_key": PACLITAXEL_INCHI_KEY,
        "expected_smiles": PACLITAXEL_SMILES,
        "description": "paclitaxel (Taxol)",
    }


def test_preflight_rejects_taxol_name_with_wrong_valid_structure() -> None:
    report = run_preflight(TargetInput(target_name="Taxol", target_smiles="CCO"))

    assert report["accepted"] is False
    assert report["reasons"] == ["known_target_identity_mismatch:paclitaxel"]
    assert "known_target_identity_mismatch" in report["initial_risk_flags"]
    assert report["known_target_identity_audit"]["observed_inchi_key"] != PACLITAXEL_INCHI_KEY


@pytest.mark.parametrize("target_name", ["paclitaxel analogue", "Taxol-like", "paclitaxel_analog"])
def test_preflight_does_not_apply_exact_identity_lock_to_analogue_names(target_name: str) -> None:
    report = run_preflight(TargetInput(target_name=target_name, target_smiles="CCO"))

    assert report["accepted"] is True
    assert report["known_target_identity_audit"] == {}
