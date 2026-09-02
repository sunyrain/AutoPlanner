from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "fetch_recent_total_synthesis_structure_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "recent_total_synthesis_structure_candidates", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def test_pubchem_property_response_is_parsed_without_admission_semantics() -> None:
    rows = resolver.pubchem_properties(
        {"PropertyTable": {"Properties": [{"CID": 7, "SMILES": "C[C@H](O)C"}]}}
    )
    candidate = resolver.candidate_record(rows[0])
    assert candidate["pubchem_cid"] == 7
    assert candidate["reported_smiles"] == "C[C@H](O)C"
    assert candidate["rdkit_validation"]["status"] in {
        "roundtrip_valid",
        "rdkit_unavailable",
    }


def test_current_pubchem_smiles_field_takes_precedence_over_connectivity() -> None:
    assert resolver.smiles_value(
        {"SMILES": "C[C@H](O)C", "ConnectivitySMILES": "CC(O)C"}
    ) == "C[C@H](O)C"


def test_pubchem_fault_response_has_no_candidates() -> None:
    assert resolver.pubchem_properties({"Fault": {"Code": "PUGREST.NotFound"}}) == []
