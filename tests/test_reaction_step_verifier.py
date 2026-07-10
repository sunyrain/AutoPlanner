from __future__ import annotations

import hashlib
from pathlib import Path

from cascade_planner.harness.reaction_step_verifier import (
    build_verified_procurement_binding,
    canonical_reaction_digest,
    is_precedent_supported_route,
    is_reaction_validated_route,
    verify_reaction_route,
    verify_reaction_step,
)
from cascade_planner.providers import ProviderContext, SnapshotStockProvider, stock_snapshot_sha256


_FIXTURES = Path(__file__).parent / "fixtures"
_SOURCE_FIXTURE = _FIXTURES / "source_evidence_stub.pdf"
_SOURCE_PAGE_FIXTURE = _FIXTURES / "source_page.ppm"
_SOURCE_MANIFEST_FIXTURE = _FIXTURES / "source_evidence_manifest.json"
_TRUSTED_REGISTRY_FIXTURE = _FIXTURES / "trusted_literature_step_registry.json"


def _mapped_ethanol_step() -> dict:
    return {
        "step_id": "ethanol",
        "product": "CCO",
        "reactant_smiles": ["CC", "O"],
        "atom_mapped_reaction_smiles": (
            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
        ),
    }


def _strict_ethanol_oxidation_step() -> dict:
    template_id = "source_detail_exact_step:ethanol_oxidation"
    return {
        "step_id": "ethanol_oxidation",
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "atom_mapped_reaction_smiles": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": template_id,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "document_id": "fixture:revalidatable-stitch",
                "manifest_path": str(_SOURCE_MANIFEST_FIXTURE.resolve()),
                "manifest_sha256": hashlib.sha256(
                    _SOURCE_MANIFEST_FIXTURE.read_bytes()
                ).hexdigest(),
                "source_pdf_path": str(_SOURCE_FIXTURE.resolve()),
                "source_pdf_sha256": hashlib.sha256(_SOURCE_FIXTURE.read_bytes()).hexdigest(),
                "page_number": 1,
                "image_path": str(_SOURCE_PAGE_FIXTURE.resolve()),
                "image_sha256": hashlib.sha256(_SOURCE_PAGE_FIXTURE.read_bytes()).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


def test_materialized_unmapped_step_is_only_graph_and_stock_closed() -> None:
    proof = verify_reaction_step(
        {"product": "CCO", "reactant_smiles": ["CC", "O"]},
        graph_and_stock_closed=True,
    )

    assert proof["accepted"] is False
    assert proof["proof_level"] == "L1_graph_and_stock_closed"
    assert "complete_atom_mapped_reaction_missing" in proof["reasons"]


def test_complete_mapped_step_is_mapping_consistent_but_not_parent_eligible() -> None:
    proof = verify_reaction_step(_mapped_ethanol_step(), graph_and_stock_closed=True)

    assert proof["accepted"] is False
    assert proof["proof_level"] == "L2_mapping_consistent"
    assert proof["checks"]["mapped_product_matches"] is True
    assert proof["checks"]["mapped_reactants_match"] is True
    assert proof["checks"]["bond_change_present"] is True


def test_self_reported_validation_cannot_replace_mapping() -> None:
    proof = verify_reaction_step(
        {
            "product": "CCO",
            "reactant_smiles": ["CC", "O"],
            "reaction_validated": True,
            "atom_provenance": {"accepted": True},
        },
        graph_and_stock_closed=True,
    )

    assert proof["accepted"] is False
    assert proof["proof_level"] == "L1_graph_and_stock_closed"


def test_atom_balanced_fragment_pile_cannot_fake_reaction_validation() -> None:
    left = ".".join(
        f"[CH3:{index}][CH3:{index + 1}]"
        for index in range(1, 13, 2)
    )
    right = "".join(
        f"[CH3:{index}]" if index in {1, 12} else f"[CH2:{index}]"
        for index in range(1, 13)
    )
    proof = verify_reaction_step(
        {
            "product": "CCCCCCCCCCCC",
            "reactant_smiles": ["CC"] * 6,
            "atom_mapped_reaction_smiles": f"{left}>>{right}",
        },
        graph_and_stock_closed=True,
    )

    assert proof["accepted"] is False
    assert proof["checks"]["scaffold_continuity_plausible"] is False
    assert "reaction_lacks_continuous_precursor_scaffold" in proof["reasons"]


def test_legacy_reactant_aliases_are_not_double_counted() -> None:
    step = _mapped_ethanol_step()
    step["main_reactant"] = "CC"
    step["precursor_smiles"] = ["CC", "O"]

    proof = verify_reaction_step(step, graph_and_stock_closed=True)

    assert proof["accepted"] is False
    assert proof["proof_level"] == "L2_mapping_consistent"
    assert proof["reactant_smiles"] == ("CC", "O") or proof["reactant_smiles"] == ["CC", "O"]


def test_arbitrary_digest_cannot_forge_trusted_precedent() -> None:
    binding = {
        "schema_version": "trusted_precedent_binding.v1",
        "accepted": True,
        "authority": "human_curator",
        "authority_id": "attacker",
        "binding_id": "forged",
        "source_ref": "doi:10.1000/forged",
        "reaction_digest": hashlib.sha256(b"reaction").hexdigest(),
    }

    proof = verify_reaction_step(
        _mapped_ethanol_step(),
        graph_and_stock_closed=True,
        trusted_precedent_binding=binding,
    )

    assert proof["proof_level"] == "L2_mapping_consistent"
    assert proof["checks"]["trusted_precedent_bound"] is False


def test_exact_registry_and_materialized_evidence_reach_l3(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    step = _strict_ethanol_oxidation_step()

    proof = verify_reaction_step(step, graph_and_stock_closed=True)

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L3_precedent_supported"
    assert proof["checks"]["trusted_precedent_bound"] is True
    assert proof["reaction_digest"] == canonical_reaction_digest("CC=O", ["CCO"])


def test_l4_requires_digest_valid_stock_provider_envelopes(monkeypatch) -> None:
    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture-supplier",
        "catalog_number": "ETHANOL-1",
        "smiles": "CCO",
        "checked_at": "2026-07-10T00:00:00Z",
        "available": True,
    }
    offer = {**snapshot, "snapshot_sha256": stock_snapshot_sha256(snapshot)}
    stock_result = SnapshotStockProvider(trusted_snapshots=[snapshot]).invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [offer],
        },
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CC=O"),
    )
    binding = build_verified_procurement_binding(
        [stock_result.to_dict()],
        reactant_smiles=["CCO"],
    )
    step = _strict_ethanol_oxidation_step()
    step["conditions"] = {
        "reagent": "oxidant",
        "solvent": "water",
        "temperature": "25 C",
        "duration": "1 h",
    }

    proof = verify_reaction_step(
        step,
        graph_and_stock_closed=True,
        procurement_binding=binding,
    )
    assert proof["proof_level"] == "L4_procurement_ready"

    forged = dict(binding)
    forged["binding_digest"] = "f" * 64
    downgraded = verify_reaction_step(
        step,
        graph_and_stock_closed=True,
        procurement_binding=forged,
    )
    assert downgraded["proof_level"] == "L3_precedent_supported"


def test_three_butanes_cut_and_glue_is_only_mapping_consistent() -> None:
    reactants = ".".join(
        "".join(
            [
                f"[CH3:{start}]",
                f"[CH2:{start + 1}]",
                f"[CH2:{start + 2}]",
                f"[CH3:{start + 3}]",
            ]
        )
        for start in (1, 5, 9)
    )
    product = "".join(
        f"[CH3:{index}]" if index in {1, 12} else f"[CH2:{index}]"
        for index in range(1, 13)
    )
    proof = verify_reaction_step(
        {
            "product_smiles": "CCCCCCCCCCCC",
            "reactant_smiles": ["CCCC"] * 3,
            "atom_mapped_reaction_smiles": f"{reactants}>>{product}",
        },
        graph_and_stock_closed=True,
    )

    assert proof["checks"]["mapped_reactant_components_contribute"] is True
    assert proof["checks"]["reaction_edit_budget_plausible"] is True
    assert proof["proof_level"] == "L2_mapping_consistent"
    assert proof["accepted"] is False


def test_segmented_complex_target_cannot_gain_reaction_authority() -> None:
    rings = []
    for start in (1, 7, 13, 19):
        atoms = [f"[CH2:{index}]" for index in range(start, start + 6)]
        rings.append(f"{atoms[0]}1{''.join(atoms[1:])}1")
    product = "".join(
        f"[CH3:{index}]" if index in {1, 24} else f"[CH2:{index}]"
        for index in range(1, 25)
    )
    proof = verify_reaction_step(
        {
            "product_smiles": "C" * 24,
            "reactant_smiles": ["C1CCCCC1"] * 4,
            "atom_mapped_reaction_smiles": f"{'.'.join(rings)}>>{product}",
        },
        graph_and_stock_closed=True,
    )

    assert proof["proof_level"] == "L2_mapping_consistent"
    assert proof["accepted"] is False


def test_route_validation_uses_weakest_link_and_digest(monkeypatch) -> None:
    validation = verify_reaction_route(
        [
            _mapped_ethanol_step(),
            {"step_id": "unmapped", "product": "CC", "reactant_smiles": ["C", "C"]},
        ],
        graph_and_stock_closed=True,
    )

    assert validation["accepted"] is False
    assert validation["proof_level"] == "L1_graph_and_stock_closed"
    assert validation["reaction_validated_step_count"] == 0
    assert is_reaction_validated_route(validation) is False

    monkeypatch.setenv(
        "AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY",
        str(_TRUSTED_REGISTRY_FIXTURE),
    )
    accepted = verify_reaction_route(
        [_strict_ethanol_oxidation_step()],
        graph_and_stock_closed=True,
    )
    assert accepted["accepted"] is True
    assert is_reaction_validated_route(accepted) is True
    assert is_precedent_supported_route(accepted) is True

    forged = dict(accepted)
    forged["reaction_validated_step_count"] = 0
    assert is_reaction_validated_route(forged) is False
