from cascade_planner.agent.executable_template_validation import (
    basic_chemical_sanity,
    executable_candidate_from_segment_step,
    forward_reconstruction_audit,
    instantiate_literature_template,
    validate_template_candidate,
)
from cascade_planner.agent.literature_segments import SegmentStepCandidate
from cascade_planner.agent.literature_templates import default_literature_template_cards


PHENOLIC_O_GLYCOSIDE = "Oc1ccccc1OC1COC(O)C(O)C1O"
BUFADIENOLIDE = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
TAXANE = "CC(=O)OC1CC(O)C2(C)C(OC(=O)c3ccccc3)C3OC3C(O)C12"
MACROLACTONE = "O=C1CCCCCCCCCCCCO1"


def _card(template_id):
    return next(card for card in default_literature_template_cards() if card.template_id == template_id)


def test_glycoside_instantiates_aglycone_and_sugar_side_candidate():
    candidate = instantiate_literature_template(PHENOLIC_O_GLYCOSIDE, _card("lit_tpl_o_glycoside_split_v1"))
    roles = {row["role"] for row in candidate.precursor_roles}

    assert candidate.validation_report["accepted"], candidate.validation_report
    assert candidate.not_lab_procedure is True
    assert candidate.requires_audit is True
    assert candidate.proposal_source == "literature_template_plugin"
    assert len(candidate.reactant_smiles) == 2
    assert "sugar_donor_or_precursor" in roles
    assert "aglycone_acceptor" in roles
    assert forward_reconstruction_audit(candidate)["passed"]


def test_bufadienolide_instantiates_steroid_and_pyrone_fragments():
    candidate = instantiate_literature_template(BUFADIENOLIDE, _card("lit_tpl_bufadienolide_c17_pyrone_split_v1"))
    roles = {row["role"] for row in candidate.precursor_roles}

    assert candidate.validation_report["accepted"], candidate.validation_report
    assert "steroid_core" in roles
    assert "pyrone_coupling_partner" in roles
    assert any("c1ccc(=O)oc1" in smi for smi in candidate.reactant_smiles)


def test_taxane_instantiates_core_and_side_chain_fragments():
    candidate = instantiate_literature_template(TAXANE, _card("lit_tpl_taxane_c13_side_chain_split_v1"))
    roles = {row["role"] for row in candidate.precursor_roles}

    assert candidate.validation_report["accepted"], candidate.validation_report
    assert "taxane_core" in roles
    assert "side_chain_fragment" in roles
    assert len(candidate.reactant_smiles) == 2


def test_macrolactone_intramolecular_ring_opening_passes_sanity_with_one_fragment():
    candidate = instantiate_literature_template(MACROLACTONE, _card("lit_tpl_macrolactone_split_v1"))
    sanity = basic_chemical_sanity(candidate)

    assert candidate.validation_report["accepted"], candidate.validation_report
    assert len(candidate.reactant_smiles) == 1
    assert sanity["intramolecular_ring_opening"] is True
    assert sanity["passed"] is True


def test_reconstruction_failure_rejects_candidate():
    candidate = instantiate_literature_template(PHENOLIC_O_GLYCOSIDE, _card("lit_tpl_o_glycoside_split_v1"))
    candidate.reactant_smiles = [candidate.reactant_smiles[0]]
    candidate.rxn_smiles = f"{candidate.reactant_smiles[0]}>>{candidate.product_smiles}"
    report = validate_template_candidate(candidate)

    assert report.accepted is False
    assert report.allowed_for_one_step_source is False
    assert "heavy_atom_accounting_mismatch" in report.reasons


def test_structured_literature_segment_step_compiles_to_executable_candidate():
    step = SegmentStepCandidate(
        step_id="seg_1",
        product_smiles="CCO",
        reactant_smiles=["CCO"],
        evidence_refs=["ev_segment"],
        source_ref="doi:10.0000/example-si",
        applicability={
            "status": "passed",
            "product_reconstruction_passed": True,
            "reconstructed_product_smiles": "CCO",
        },
        condition_candidate={
            "step_id": "seg_1",
            "source_type": "exact",
            "condition_status": "evidence_backed",
            "solvent": "MeCN",
            "temperature": "25 C",
            "evidence_refs": ["ev_segment"],
        },
    )

    candidate = executable_candidate_from_segment_step(step, source_template_id="seg:seg_1")
    report = validate_template_candidate(candidate)

    assert report.accepted is True
    assert report.allowed_for_one_step_source is True
    assert candidate.not_lab_procedure is True
    assert candidate.requires_audit is True
    assert candidate.literature_template_trace["structured_segment_step"] is True
    assert forward_reconstruction_audit(candidate)["passed"] is True
