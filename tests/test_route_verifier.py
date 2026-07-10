from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path

import pytest

from cascade_planner.harness import route_verifier as rv


def _rehash_record(value: dict) -> None:
    payload = dict(value)
    payload.pop("content_hash", None)
    value["content_hash"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _route(
    target: str,
    reactants: list[str],
    *,
    step_extra: dict | None = None,
) -> dict:
    terminals = list(dict.fromkeys(reactants))
    return {
        "target": target,
        "routes": [
            {
                "route_rank": 0,
                "metrics": {
                    "terminal_reactants": terminals,
                    "terminal_stock_status": {item: True for item in terminals},
                },
                "steps": [
                    {
                        "index": 0,
                        "product": target,
                        "reactant_smiles": reactants,
                        "stock_status": {item: True for item in terminals},
                        **dict(step_extra or {}),
                    }
                ],
            }
        ],
    }


def _stock_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    vendor_root = tmp_path / "retro_planner"
    config_path = vendor_root / "config" / "config.yaml"
    stock_dir = vendor_root / "building_block_dataset"
    config_path.parent.mkdir(parents=True)
    stock_dir.mkdir(parents=True)
    n1_path = stock_dir / "n1-stock.csv"
    zinc_path = stock_dir / "zinc-stock.csv"
    config_path.write_text(
        "stocks:\n"
        '  PaRotes_n1-stock: "building_block_dataset/n1-stock.csv"\n'
        '  Zinc_Fix-stock: "building_block_dataset/zinc-stock.csv"\n',
        encoding="utf-8",
    )
    n1_path.write_text("smiles\nCN\n", encoding="utf-8")
    zinc_path.write_text("smiles\nCO\n", encoding="utf-8")
    monkeypatch.setattr(rv, "_CHEMENZY_STOCK_CONFIG", config_path)
    return config_path, n1_path, zinc_path


def test_effective_catalog_name_path_and_sha_are_recomputed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, n1_path, _ = _stock_fixture(tmp_path, monkeypatch)
    n1_path.write_text("smiles\nCO\n", encoding="utf-8")
    raw = _route("COC", ["CO", "C"])
    raw["ui_metadata"] = {"stock_names": ["PaRotes_n1-stock"]}

    report = rv.verify_chemenzy_raw_routes(raw, target_smiles="COC")

    assert report["accepted"], report["reasons"]
    audit = report["stock_catalog_audit"]
    assert audit["catalog_binding_valid"] is True
    assert audit["effective_stock_names"] == ["PaRotes_n1-stock"]
    assert audit["effective_catalogs"] == [
        {
            "catalog_name": "PaRotes_n1-stock",
            "catalog_id": f"PaRotes_n1-stock@sha256:{hashlib.sha256(n1_path.read_bytes()).hexdigest()}",
            "path": str(n1_path.resolve()),
            "size_bytes": n1_path.stat().st_size,
            "sha256": hashlib.sha256(n1_path.read_bytes()).hexdigest(),
            "lookup_basis": "exact_canonical_smiles_first_csv_field",
            "binding_source": "chem_enzy_config",
        }
    ]
    assert audit["terminal_evidence"]["CO"]["catalog_name"] == "PaRotes_n1-stock"
    assert rv.is_accepted_route_verifier_report(report, expected_target_smiles="COC")


def test_paroutes_request_cannot_be_rechecked_against_broader_zinc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, zinc_path = _stock_fixture(tmp_path, monkeypatch)
    raw = _route("COC", ["CO", "C"])
    raw["ui_metadata"] = {"stock_names": ["PaRotes_n1-stock"]}
    raw["stock_catalogs"] = [
        {
            "name": "PaRotes_n1-stock",
            "path": str(zinc_path),
            "sha256": hashlib.sha256(zinc_path.read_bytes()).hexdigest(),
        }
    ]

    report = rv.verify_chemenzy_raw_routes(raw, target_smiles="COC")

    assert report["accepted"] is False
    assert "stock_catalog_binding_unverifiable" in report["reasons"]
    audit = report["stock_catalog_audit"]
    assert audit["catalog_binding_valid"] is False
    assert "configured_catalog_path_mismatch:PaRotes_n1-stock" in audit["binding_failures"]
    assert audit["effective_catalogs"] == []


def test_missing_effective_catalog_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path, n1_path, _ = _stock_fixture(tmp_path, monkeypatch)
    n1_path.unlink()
    raw = _route("COC", ["CO", "C"])
    raw["request"] = {"stock_names": ["PaRotes_n1-stock"]}

    report = rv.verify_chemenzy_raw_routes(raw, target_smiles="COC")

    assert report["accepted"] is False
    assert "stock_catalog_binding_unverifiable" in report["reasons"]
    assert report["stock_catalog_audit"]["binding_failures"] == [
        "catalog_file_missing:PaRotes_n1-stock"
    ]
    assert config_path.is_file()


def test_common_commodities_are_an_explicit_separate_catalog():
    report = rv.verify_chemenzy_raw_routes(_route("CCO", ["CC", "O"]), target_smiles="CCO")

    assert report["accepted"], report["reasons"]
    audit = report["stock_catalog_audit"]
    assert audit["catalog_binding_status"] == "common_catalog_only"
    assert audit["effective_catalogs"] == []
    assert audit["common_catalog"]["catalog_name"] == "autoplanner_common_commodity.v1"
    assert audit["common_catalog"]["catalog_role"] == "independent_common_commodity_supplement"
    assert audit["terminal_evidence"]["CC"]["catalog_name"] == "autoplanner_common_commodity.v1"


def test_rejected_sibling_catalog_gap_does_not_contaminate_common_only_route():
    raw = _route("CCO", ["CC", "O"])
    raw["routes"].append(
        {
            "route_rank": 1,
            "metrics": {
                "terminal_reactants": ["NNN"],
                "terminal_stock_status": {"NNN": True},
            },
            "steps": [
                {
                    "product": "CCO",
                    "reactant_smiles": ["NNN"],
                    "stock_status": {"NNN": True},
                }
            ],
        }
    )

    report = rv.verify_chemenzy_raw_routes(raw, target_smiles="CCO")

    assert report["accepted"], report["reasons"]
    assert report["accepted_route_count"] == 1
    assert report["rejected_route_count"] == 1
    assert report["best_route_rank"] == 0
    assert "stock_catalog_binding_unverifiable" in report["warnings"]
    assert rv.is_accepted_route_verifier_report(report, expected_target_smiles="CCO")


def test_duplicate_fragment_padding_does_not_explain_a_large_jump():
    target = "C" * 20
    raw = _route(
        target,
        ["CC"] * 10,
        step_extra={"atom_provenance": {"accepted": True}},
    )

    report = rv.verify_chemenzy_raw_routes(
        raw,
        target_smiles=target,
    )

    assert report["accepted"] is False
    assert "large_atom_jump" in report["reasons"]
    provenance = report["failure_events"][0]["details"]["jumps"][0]["atom_provenance_audit"]
    assert provenance["validated"] is False
    assert provenance["reasons"] == ["complete_atom_mapped_reaction_missing"]


def test_self_reported_atom_provenance_boolean_is_never_trusted():
    raw = _route("CCCC", ["CC", "CC"], step_extra={"atom_provenance": {"accepted": True}})

    report = rv.verify_chemenzy_raw_routes(
        raw,
        target_smiles="CCCC",
        large_atom_jump_heavy_atoms=2,
    )

    assert report["accepted"] is False
    assert "large_atom_jump" in report["reasons"]


def test_duplicate_or_new_atom_maps_do_not_validate_convergence():
    mapped = "[CH3:1][CH3:2].[CH3:1][CH3:4]>>[CH3:1][CH2:2][CH2:3][CH3:4]"
    raw = _route("CCCC", ["CC", "CC"], step_extra={"atom_mapped_reaction_smiles": mapped})

    report = rv.verify_chemenzy_raw_routes(
        raw,
        target_smiles="CCCC",
        large_atom_jump_heavy_atoms=2,
    )

    assert report["accepted"] is False
    jump = report["failure_events"][0]["details"]["jumps"][0]
    reasons = jump["atom_provenance_audit"]["reasons"]
    assert "atom_mapping_not_unique" in reasons
    assert "product_heavy_atom_without_reactant_provenance" in reasons


def test_mapped_element_alchemy_is_rejected():
    mapped = "[CH3:1][CH3:2].[CH3:3][CH3:4]>>[CH3:1][CH2:2][NH:3][NH2:4]"
    raw = _route("CCNN", ["CC", "CC"], step_extra={"atom_mapped_reaction_smiles": mapped})

    report = rv.verify_chemenzy_raw_routes(
        raw,
        target_smiles="CCNN",
        large_atom_jump_heavy_atoms=2,
    )

    assert report["accepted"] is False
    assert "large_atom_jump" in report["reasons"]
    assert "element_inventory_not_conserved" in report["reasons"]
    reasons = report["failure_events"][0]["details"]["jumps"][0]["atom_provenance_audit"]["reasons"]
    assert "mapped_atom_element_changed" in reasons


def test_complete_unique_mapped_convergence_is_recomputed_and_allowed():
    mapped = "[CH3:1][CH3:2].[CH3:3][CH3:4]>>[CH3:1][CH2:2][CH2:3][CH3:4]"
    raw = _route("CCCC", ["CC", "CC"], step_extra={"atom_mapped_reaction_smiles": mapped})

    report = rv.verify_chemenzy_raw_routes(
        raw,
        target_smiles="CCCC",
        large_atom_jump_heavy_atoms=2,
    )

    assert report["accepted"], report["reasons"]
    rejected = report["rejected_route_summary"]
    assert rejected == []
    assert report["failure_events"] == []
    assert report["accepted_route"]["steps"][0]["atom_mapped_reaction_smiles"] == mapped
    convergence = report["accepted_route_audit"]["mapped_convergent_assembly_audit"]
    assert len(convergence) == 1
    assert convergence[0]["atom_provenance_audit"]["validated"] is True
    assert convergence[0]["atom_provenance_audit"]["cross_component_product_bond_count"] == 1
    assert rv.is_accepted_route_verifier_report(report, expected_target_smiles="CCCC")

    forged = dict(report)
    forged["accepted_route"] = dict(report["accepted_route"])
    forged["accepted_route"]["steps"] = [dict(report["accepted_route"]["steps"][0])]
    forged_step = forged["accepted_route"]["steps"][0]
    forged_step.pop("atom_mapped_reaction_smiles")
    forged_step["atom_provenance"] = {"accepted": True}
    assert rv.is_accepted_route_verifier_report(forged, expected_target_smiles="CCCC") is False


def test_route_proof_bank_preserves_and_replays_every_accepted_route():
    raw = _route("CCO", ["CC", "O"])
    raw["routes"].append(
        {
            "route_rank": 1,
            "score": 0.5,
            "metrics": {
                "terminal_reactants": ["C", "O"],
                "terminal_stock_status": {"C": True, "O": True},
            },
            "steps": [
                {
                    "index": 0,
                    "product": "CCO",
                    "reactant_smiles": ["C", "C", "O"],
                    "stock_status": {"C": True, "O": True},
                }
            ],
        }
    )

    report = rv.verify_chemenzy_raw_routes(raw, target_smiles="CCO")

    assert report["accepted"] is True
    assert report["best_route_rank"] == 0
    assert report["accepted_route"]["route_rank"] == 0
    bank = report["route_proof_bank"]
    assert bank["schema_version"] == "route_proof_bank.v1"
    assert bank["entry_count"] == 2
    assert [entry["route_rank"] for entry in bank["entries"]] == [0, 1]
    assert rv.validate_route_proof_bank(bank, expected_target_smiles="CCO") == []
    for entry in bank["entries"]:
        replay = rv.replay_route_proof_bank_entry(
            bank,
            proof_id=entry["proof_id"],
            expected_target_smiles="CCO",
        )
        assert replay["accepted"] is True
        assert replay["reaction_validated"] is False
        assert rv.is_replayable_route_proof_bank_entry(
            bank,
            proof_id=entry["proof_id"],
            expected_target_smiles="CCO",
        )


def test_route_proof_bank_tamper_and_fake_authority_fields_fail_replay():
    report = rv.verify_chemenzy_raw_routes(
        _route("CCO", ["CC", "O"]),
        target_smiles="CCO",
    )
    bank = report["route_proof_bank"]
    proof_id = bank["entries"][0]["proof_id"]

    tampered = copy.deepcopy(bank)
    tampered["entries"][0]["materialized_route"]["steps"][0]["product"] = "CCN"
    _rehash_record(tampered["entries"][0])
    _rehash_record(tampered)
    assert rv.validate_route_proof_bank(tampered, expected_target_smiles="CCO") == []
    assert not rv.is_replayable_route_proof_bank_entry(
        tampered,
        proof_id=proof_id,
        expected_target_smiles="CCO",
    )

    forged = copy.deepcopy(bank)
    forged["entries"][0]["solved"] = True
    _rehash_record(forged["entries"][0])
    _rehash_record(forged)
    reasons = rv.validate_route_proof_bank(forged, expected_target_smiles="CCO")
    assert "entry:0:route_proof_bank_entry_fields_not_strict" in reasons
    assert not rv.is_replayable_route_proof_bank_entry(
        forged,
        proof_id=proof_id,
        expected_target_smiles="CCO",
    )


def test_single_route_bank_and_legacy_report_remain_compatible():
    report = rv.verify_chemenzy_raw_routes(
        _route("CCO", ["CC", "O"]),
        target_smiles="CCO",
    )

    assert report["route_proof_bank"]["entry_count"] == 1
    assert rv.is_replayable_route_proof_bank_entry(
        report["route_proof_bank"],
        expected_target_smiles="CCO",
    )
    legacy = dict(report)
    legacy.pop("route_proof_bank")
    assert rv.is_accepted_route_verifier_report(
        legacy,
        expected_target_smiles="CCO",
    )
