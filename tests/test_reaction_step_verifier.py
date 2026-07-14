from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.harness.reaction_step_verifier import (
    build_verified_procurement_binding,
    canonical_reaction_digest,
    is_precedent_supported_route,
    is_reaction_validated_route,
    verify_reaction_route,
    verify_reaction_step,
)
from cascade_planner.providers import (
    BenchmarkCatalogStockProvider,
    ProviderContext,
    SnapshotStockProvider,
    stock_snapshot_sha256,
)


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
    assert proof["checks"]["deterministic_transform_reapplied"] is False
    assert proof["checks"]["trusted_precedent_bound"] is False
    assert proof["deterministic_transform_audit"]["transform_family"] == ""
    assert "reaction_centre_not_in_deterministic_transform_registry" in proof[
        "reasons"
    ]


def test_host_reapplied_carbonyl_redox_reaches_l2_reaction_validated() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "ethanol_oxidation_without_precedent",
            "product_smiles": "CC=O",
            "reactant_smiles": ["CCO"],
            "atom_mapped_reaction_smiles": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        },
        graph_and_stock_closed=True,
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L2_reaction_validated"
    assert proof["checks"]["deterministic_transform_reapplied"] is True
    assert (
        proof["deterministic_transform_audit"]["transform_family"]
        == "carbonyl_alcohol_redox"
    )


def test_host_reapplied_acyl_substitution_reaches_l2_reaction_validated() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "methyl_acetamide",
            "product_smiles": "CC(=O)NC",
            "reactant_smiles": ["CC(=O)O", "CN"],
            "atom_mapped_reaction_smiles": (
                "[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]"
                ">>[CH3:1][C:2](=[O:3])[NH:5][CH3:6]"
            ),
        },
        graph_and_stock_closed=True,
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L2_reaction_validated"
    assert (
        proof["deterministic_transform_audit"]["transform_family"]
        == "acyl_substitution_coupling"
    )


def test_host_reapplied_intramolecular_lactonization_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "gamma_hydroxy_acid_lactonization",
            "product_smiles": "O=C1OCCC1",
            "reactant_smiles": ["O=C(O)CCCO"],
            "atom_mapped_reaction_smiles": (
                "O=[C:5]([CH2:4][CH2:3][CH2:2][OH:1])[OH:6]"
                ">>[O:1]1[CH2:2][CH2:3][CH2:4][C:5](=[O:6])1"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L2_reaction_validated"
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "intramolecular_lactonization"
    )


def test_host_reapplied_amine_to_isothiocyanate_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": "S=C=Nc1ccccc1",
            "reactant_smiles": ["Nc1ccccc1", "S=C(Cl)Cl"],
            "atom_mapped_reaction_smiles": (
                "[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1."
                "Cl[C:8](Cl)=[S:9]>>"
                "[N:1](=[C:8]=[S:9])[c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "amine_to_isocyanate_or_isothiocyanate"
    )


def test_host_reapplied_isothiocyanate_amine_addition_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": "NC(=S)Nc1ccccc1",
            "reactant_smiles": ["N", "S=C=Nc1ccccc1"],
            "atom_mapped_reaction_smiles": (
                "[NH3:1].[C:2](=[S:3])=[N:4][c:5]1[cH:6][cH:7]"
                "[cH:8][cH:9][cH:10]1>>[NH2:1][C:2](=[S:3])[NH:4]"
                "[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "heterocumulene_nucleophile_addition"
    )


def test_symmetry_tolerant_thiohydantoin_n_arylation_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": (
                "CNC(=O)c1ccc(N2C(=S)N(c3ccc(C#N)c(C(F)(F)F)c3)"
                "C(=O)C2(C)C)cc1F"
            ),
            "reactant_smiles": [
                "CC1(C)C(=O)NC(=S)N1c1ccc(C#N)c(C(F)(F)F)c1",
                "CNC(=O)c1ccc(Br)cc1F",
            ],
            "atom_mapped_reaction_smiles": (
                "[NH:9]1[C:10](=[S:11])[N:12]([c:13]2[cH:14][cH:15]"
                "[c:16]([C:17]#[N:18])[c:19]([C:20]([F:21])([F:22])"
                "[F:23])[cH:24]2)[C:27]([CH3:28])([CH3:29])[C:25]1=[O:26]."
                "Br[c:8]1[cH:7][cH:6][c:5]([C:3]([NH:2][CH3:1])=[O:4])"
                "[c:31]([F:32])[cH:30]1>>[CH3:1][NH:2][C:3](=[O:4])"
                "[c:5]1[cH:6][cH:7][c:8]([N:9]2[C:10](=[S:11])[N:12]"
                "([c:13]3[cH:14][cH:15][c:16]([C:17]#[N:18])[c:19]"
                "([C:20]([F:21])([F:22])[F:23])[cH:24]3)[C:25](=[O:26])"
                "[C:27]2([CH3:28])[CH3:29])[cH:30][c:31]1[F:32]"
            ),
        }
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "symmetry_tolerant_heteroatom_nucleophilic_substitution"
    )


def test_amino_acid_isothiocyanate_thiohydantoin_annulation_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": (
                "CC1(C)C(=O)NC(=S)N1c1ccc(C#N)c(C(F)(F)F)c1"
            ),
            "reactant_smiles": [
                "CC(C)(N)C(=O)O",
                "N#Cc1ccc(N=C=S)cc1C(F)(F)F",
            ],
            "atom_mapped_reaction_smiles": (
                "O[C:4]([C:2]([CH3:1])([CH3:3])[NH2:6])=[O:5]."
                "[C:7](=[S:8])=[N:9][c:10]1[cH:11][cH:12][c:13]"
                "([C:14]#[N:15])[c:16]([C:17]([F:18])([F:19])[F:20])"
                "[cH:21]1>>[CH3:1][C:2]1([CH3:3])[C:4](=[O:5])[NH:6]"
                "[C:7](=[S:8])[N:9]1[c:10]1[cH:11][cH:12][c:13]"
                "([C:14]#[N:15])[c:16]([C:17]([F:18])([F:19])[F:20])"
                "[cH:21]1"
            ),
        }
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "amino_acid_isothiocyanate_thiohydantoin_annulation"
    )


def test_aryl_thioisocyanate_connectivity_typo_cannot_gain_annulation_proof() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": (
                "CC1(C)C(=O)NC(=S)N1c1ccc(C#N)c(C(F)(F)F)c1"
            ),
            "reactant_smiles": [
                "CC(C)(N)C(=O)O",
                "N#Cc1ccc([SH]=C=N)cc1C(F)(F)F",
            ],
            "atom_mapped_reaction_smiles": (
                "O[C:4]([C:2]([CH3:1])([CH3:3])[NH2:9])=[O:5]."
                "[NH:6]=[C:7]=[SH:8][c:10]1[cH:11][cH:12][c:13]"
                "([C:14]#[N:15])[c:16]([C:17]([F:18])([F:19])[F:20])"
                "[cH:21]1>>[CH3:1][C:2]1([CH3:3])[C:4](=[O:5])[NH:6]"
                "[C:7](=[S:8])[N:9]1[c:10]1[cH:11][cH:12][c:13]"
                "([C:14]#[N:15])[c:16]([C:17]([F:18])([F:19])[F:20])"
                "[cH:21]1"
            ),
        }
    )

    assert proof["accepted"] is False
    assert proof["deterministic_transform_audit"]["transform_family"] == ""


def test_host_reapplied_aryl_halide_n_coupling_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": "Nc1ccccc1",
            "reactant_smiles": ["N", "Brc1ccccc1"],
            "atom_mapped_reaction_smiles": (
                "[NH3:1].Br[c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1>>"
                "[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "heteroatom_nucleophilic_substitution"
    )


def test_host_reapplied_aryl_halide_boron_cross_coupling_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": "c1ccc(-c2ccccc2)cc1",
            "reactant_smiles": ["Brc1ccccc1", "OB(O)c1ccccc1"],
            "atom_mapped_reaction_smiles": (
                "Br[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1."
                "OB(O)[c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1>>"
                "[c:1]1(-[c:7]2[cH:8][cH:9][cH:10][cH:11][cH:12]2)"
                "[cH:2][cH:3][cH:4][cH:5][cH:6]1"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "carbon_carbon_cross_coupling"
    )


def test_host_reapplied_thiourea_alpha_haloacyl_annulation_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "product_smiles": "CC1(C)C(=O)N(C)C(=S)N1C",
            "reactant_smiles": ["CC(C)(Cl)C(=O)Cl", "CNC(=S)NC"],
            "atom_mapped_reaction_smiles": (
                "Cl[C:1](=[O:2])[C:3](Cl)([CH3:4])[CH3:5]."
                "[CH3:6][NH:7][C:8](=[S:9])[NH:10][CH3:11]>>"
                "[CH3:6][N:7]1[C:8](=[S:9])[N:10]([CH3:11])"
                "[C:1](=[O:2])[C:3]1([CH3:4])[CH3:5]"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "thiourea_alpha_haloacyl_thiohydantoin_annulation"
    )


def test_unmapped_departing_oxygen_in_dehydration_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "acetamide_dehydration",
            "product_smiles": "CC#N",
            "reactant_smiles": ["CC(N)=O"],
            # RXNMapper may intentionally leave the oxygen that departs the
            # recorded major product unmapped.
            "atom_mapped_reaction_smiles": (
                "[CH3:1][C:2]([NH2:3])=[O]>>[CH3:1][C:2]#[N:3]"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L2_reaction_validated"
    assert proof["checks"]["atom_maps_complete"] is False
    assert proof["checks"]["product_atom_maps_complete"] is True
    assert proof["checks"]["reactant_departing_atoms_plausible"] is True
    assert proof["atom_map_audit"]["departing_reactant_heavy_atom_count"] == 1
    assert (
        proof["deterministic_transform_audit"]["transform_family"]
        == "amide_or_oxime_dehydration_to_nitrile"
    )


def test_unmapped_methyl_ester_cleavage_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "methyl_ester_hydrolysis",
            "product_smiles": "CC(=O)O",
            "reactant_smiles": ["CC(=O)OC"],
            "atom_mapped_reaction_smiles": (
                "[CH3:1][C:2](=[O:3])[O:4][CH3]"
                ">>[CH3:1][C:2](=[O:3])[OH:4]"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L2_reaction_validated"
    assert proof["checks"]["bond_change_present"] is True
    assert proof["bond_change_audit"]["departing_unmapped_bonds"] == [
        {
            "retained_atom_map": 4,
            "leaving_atomic_number": 6,
            "bond_type": "SINGLE",
        }
    ]
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "heteroatom_deprotection_or_cleavage"
    )


def test_unmapped_boc_carbamate_cleavage_reaches_l2() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "boc_deprotection",
            "product_smiles": "CN",
            "reactant_smiles": ["CNC(=O)OC(C)(C)C"],
            "atom_mapped_reaction_smiles": (
                "[CH3:1][NH:2][C](=[O])[O][C]([CH3])([CH3])[CH3]"
                ">>[CH3:1][NH2:2]"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L2_reaction_validated"
    assert proof["checks"]["reactant_departing_atoms_plausible"] is True
    assert proof["deterministic_transform_audit"]["transform_family"] == (
        "heteroatom_deprotection_or_cleavage"
    )


def test_small_noncontributing_acid_is_allowed_as_a_bounded_spectator() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "boc_deprotection_with_mesylate_condition",
            "product_smiles": "N#C[C@@H](N)C[C@@H]1CCNC1=O",
            "reactant_smiles": [
                "CC(C)(C)OC(=O)N[C@H](C#N)C[C@@H]1CCNC1=O",
                "CS(=O)(=O)O",
            ],
            "atom_mapped_reaction_smiles": (
                "CC(C)(C)OC(=O)[NH:4][C@H:3]([C:2]#[N:1])"
                "[CH2:5][C@@H:6]1[CH2:7][CH2:8][NH:9][C:10]1=[O:11]."
                "CS(=O)(=O)O>>[N:1]#[C:2][C@@H:3]([NH2:4])"
                "[CH2:5][C@@H:6]1[CH2:7][CH2:8][NH:9][C:10]1=[O:11]"
            ),
        }
    )

    assert proof["accepted"] is True
    assert proof["checks"]["mapped_reactant_components_contribute"] is False
    assert proof["checks"]["reactant_component_participation_plausible"] is True
    assert proof["atom_map_audit"]["spectator_reactant_component_count"] == 1


def test_more_than_two_noncontributing_components_are_rejected() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "oxidation_with_implausible_spectator_bundle",
            "product_smiles": "CC=O",
            "reactant_smiles": ["CCO", "Cl", "Br", "I"],
            "atom_mapped_reaction_smiles": (
                "[CH3:1][CH2:2][OH:3].Cl.Br.I>>[CH3:1][CH:2]=[O:3]"
            ),
        }
    )

    assert proof["accepted"] is False
    assert proof["checks"]["reactant_component_participation_plausible"] is False
    assert proof["atom_map_audit"]["spectator_reactant_component_count"] == 3
    assert "mapped_reactant_component_does_not_contribute_to_product" in proof[
        "reasons"
    ]


def test_excessive_unmapped_reactant_atom_loss_fails_closed() -> None:
    proof = verify_reaction_step(
        {
            "step_id": "implausible_atom_jump",
            "product_smiles": "CC",
            "reactant_smiles": ["C" * 15],
            "atom_mapped_reaction_smiles": (
                "[CH3:1][CH2:2]CCCCCCCCCCCCC>>[CH3:1][CH3:2]"
            ),
        },
        graph_and_stock_closed=False,
    )

    assert proof["accepted"] is False
    assert proof["checks"]["product_atom_maps_complete"] is True
    assert proof["checks"]["reactant_departing_atoms_plausible"] is False
    assert "reactant_departing_atom_budget_exceeded" in proof["reasons"]


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


def _canonical_exact_record(*, product: str = "CCO") -> dict:
    row = {
        "schema_version": "exact_source_reaction_record.v1",
        "record_id": "exact:ethanol-source-row",
        "relation_type": "exact",
        "authority_scope": "source_exact_structure_observation",
        "procedure_authority_scope": "source_exact_reaction_procedure",
        "not_reaction_validation": True,
        "edge_digest": "a" * 64,
        "source_binding_id": "source:ethanol-paper",
        "source_ref": "doi:10.1000/exact-ethanol-row",
        "product_smiles": product,
        "reactant_smiles": ["CC", "O"],
        "extraction_artifact_sha256": "b" * 64,
        "location_refs": ["paper:page:3", f"pdf_sha256:{'c' * 64}"],
        "extractor": {
            "producer_id": "fixture.deterministic_structure_parser.v1",
            "producer_kind": "deterministic_structure_parser",
            "version": "1.0.0",
        },
    }
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row


def test_hash_bound_exact_source_row_and_mapping_reach_l3() -> None:
    proof = verify_reaction_step(
        _mapped_ethanol_step(),
        exact_source_records=[_canonical_exact_record()],
    )

    assert proof["accepted"] is True
    assert proof["proof_level"] == "L3_precedent_supported"
    assert proof["checks"]["trusted_precedent_bound"] is True
    assert proof["trusted_precedent_binding"]["binding_id"] == (
        "exact:ethanol-source-row"
    )


def test_tampered_or_structure_mismatched_exact_row_cannot_grant_l3() -> None:
    tampered = _canonical_exact_record()
    tampered["source_ref"] = "doi:10.1000/tampered-after-hash"
    mismatched = _canonical_exact_record(product="CC=O")

    for row in (tampered, mismatched):
        proof = verify_reaction_step(
            _mapped_ethanol_step(),
            exact_source_records=[row],
        )
        assert proof["accepted"] is False
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
    stock_provider = SnapshotStockProvider(trusted_snapshots=[snapshot])
    stock_result = stock_provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [offer],
        },
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CC=O"),
    ).to_dict()
    # Durable campaign artifacts are JSON, so tuple-valued envelope fields
    # return as lists.  A host replay must compare the canonical JSON value.
    stock_result = json.loads(json.dumps(stock_result))
    binding = build_verified_procurement_binding(
        [stock_result],
        reactant_smiles=["CCO"],
        trusted_stock_providers={stock_provider.descriptor.provider_id: stock_provider},
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
        trusted_stock_providers={stock_provider.descriptor.provider_id: stock_provider},
    )
    assert proof["proof_level"] == "L4_procurement_ready"

    untrusted_binding = build_verified_procurement_binding(
        [stock_result],
        reactant_smiles=["CCO"],
    )
    assert untrusted_binding["accepted"] is False
    untrusted = verify_reaction_step(
        step,
        graph_and_stock_closed=True,
        procurement_binding=untrusted_binding,
    )
    assert untrusted["proof_level"] == "L3_precedent_supported"

    forged = dict(binding)
    forged["binding_digest"] = "f" * 64
    downgraded = verify_reaction_step(
        step,
        graph_and_stock_closed=True,
        procurement_binding=forged,
        trusted_stock_providers={stock_provider.descriptor.provider_id: stock_provider},
    )
    assert downgraded["proof_level"] == "L3_precedent_supported"


def test_procurement_replay_rejects_forged_and_rehashed_tampered_envelopes() -> None:
    trusted_snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "trusted-supplier",
        "catalog_number": "ETHANOL-TRUSTED",
        "smiles": "CCO",
        "checked_at": "2026-07-10T00:00:00Z",
        "available": True,
    }
    trusted_provider = SnapshotStockProvider(trusted_snapshots=[trusted_snapshot])
    context = ProviderContext(run_id="test", case_id="test", target_smiles="CCO")

    invented_snapshot = {
        **trusted_snapshot,
        "supplier": "invented-supplier",
        "catalog_number": "FAKE-1",
    }
    attacker_provider = SnapshotStockProvider(trusted_snapshots=[invented_snapshot])
    forged_result = attacker_provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [
                {
                    **invented_snapshot,
                    "snapshot_sha256": stock_snapshot_sha256(invented_snapshot),
                }
            ],
        },
        context=context,
    ).to_dict()
    forged_binding = build_verified_procurement_binding(
        [forged_result],
        reactant_smiles=["CCO"],
        trusted_stock_providers={
            trusted_provider.descriptor.provider_id: trusted_provider,
        },
    )
    assert forged_binding["accepted"] is False

    legitimate_result = trusted_provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [
                {
                    **trusted_snapshot,
                    "snapshot_sha256": stock_snapshot_sha256(trusted_snapshot),
                }
            ],
        },
        context=context,
    ).to_dict()
    tampered_result = json.loads(json.dumps(legitimate_result))
    offer = tampered_result["payload"]["offers"][0]
    offer["available"] = False
    offer["snapshot"]["available"] = False
    offer["snapshot_sha256"] = stock_snapshot_sha256(offer["snapshot"])
    unsigned = dict(tampered_result)
    unsigned.pop("content_hash", None)
    tampered_result["content_hash"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    tampered_binding = build_verified_procurement_binding(
        [tampered_result],
        reactant_smiles=["CCO"],
        trusted_stock_providers={
            trusted_provider.descriptor.provider_id: trusted_provider,
        },
    )
    assert tampered_binding["accepted"] is False


def test_benchmark_replay_cannot_claim_commercial_procurement(tmp_path: Path) -> None:
    catalog = tmp_path / "benchmark-stock.csv"
    catalog.write_text("smiles\nCCO\n", encoding="utf-8")
    provider = BenchmarkCatalogStockProvider(
        catalog_artifact=catalog,
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        catalog_name="fixture-benchmark",
    )
    result = provider.invoke(
        {"schema_version": "stock_lookup_request.v1", "smiles": "CCO"},
        context=ProviderContext(run_id="test", case_id="test", target_smiles="CCO"),
    ).to_dict()
    binding = build_verified_procurement_binding(
        [json.loads(json.dumps(result))],
        reactant_smiles=["CCO"],
        trusted_stock_providers={provider.descriptor.provider_id: provider},
    )

    assert result["payload"]["boundary_type"] == "benchmark_stock"
    assert binding["accepted"] is False


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

    source_claimed = verify_reaction_step(
        {
            "product_smiles": "C" * 24,
            "reactant_smiles": ["C1CCCCC1"] * 4,
            "atom_mapped_reaction_smiles": f"{'.'.join(rings)}>>{product}",
        },
        graph_and_stock_closed=True,
        source_supported_multicentre=True,
    )
    assert source_claimed["accepted"] is False
    assert source_claimed["deterministic_transform_audit"]["transform_family"] == ""


def test_exact_source_supported_artemisinin_oxidative_cascade_is_validated() -> None:
    artemisinin = (
        "C[C@@H]1CC[C@H]2[C@H](C(=O)O[C@H]3[C@@]24[C@H]1CC[C@](O3)(OO4)C)C"
    )
    dihydroartemisinic_acid = (
        "C[C@@H]1CC[C@H]([C@@H]2[C@H]1CCC(=C2)C)[C@@H](C)C(=O)O"
    )
    mapped = (
        "[CH3:1][C@@H:2]1[CH2:3][CH2:4][C@@H:5]([C@@H:6]([CH3:7])"
        "[C:8](=[O:9])[OH:10])[C@H:18]2[CH:11]=[C:13]([CH3:14])[CH2:15]"
        "[CH2:16][C@@H:17]12.[O:19]=[O:20].O=[O:12]>>[CH3:1][C@@H:2]1"
        "[CH2:3][CH2:4][C@H:5]2[C@@H:6]([CH3:7])[C:8](=[O:9])[O:10]"
        "[C@@H:11]3[O:12][C@@:13]4([CH3:14])[CH2:15][CH2:16][C@@H:17]1"
        "[C@@:18]23[O:19][O:20]4"
    )
    step = {
        "product_smiles": artemisinin,
        "reactant_smiles": [dihydroartemisinic_acid, "O=O", "O=O"],
        "mapped_reaction_smiles": mapped,
    }

    unbound = verify_reaction_step(step)
    supported = verify_reaction_step(step, source_supported_multicentre=True)

    assert unbound["accepted"] is False
    assert "reaction_creates_too_many_rings_in_one_step" in unbound["reasons"]
    assert supported["accepted"] is True
    assert supported["proof_level"] == "L2_reaction_validated"
    assert supported["atom_map_audit"]["net_ring_increase"] == 3
    assert supported["bond_change_audit"]["bond_edit_count"] == 9
    assert supported["deterministic_transform_audit"]["transform_family"] == (
        "source_supported_tandem_oxidative_cyclization"
    )


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
