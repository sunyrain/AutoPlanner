from cascade_planner.agent.literature_templates import default_literature_template_cards
from cascade_planner.agent.template_applicability import assess_template_applicability, cut_report_from_applicability


PHENOLIC_O_GLYCOSIDE = "Oc1ccccc1OC1COC(O)C(O)C1O"
BUFADIENOLIDE = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
TAXANE = "CC(=O)OC1CC(O)C2(C)C(OC(=O)c3ccccc3)C3OC3C(O)C12"
MACROLACTONE = "O=C1CCCCCCCCCCCCO1"


def _card(template_id):
    return next(card for card in default_literature_template_cards() if card.template_id == template_id)


def test_phenolic_o_glycoside_matches_o_retron_and_c_glycoside_downgrades():
    o_report = assess_template_applicability(
        target_smiles=PHENOLIC_O_GLYCOSIDE,
        frontier_smiles=PHENOLIC_O_GLYCOSIDE,
        template_card=_card("lit_tpl_o_glycoside_split_v1"),
    )
    c_report = assess_template_applicability(
        target_smiles=PHENOLIC_O_GLYCOSIDE,
        frontier_smiles=PHENOLIC_O_GLYCOSIDE,
        template_card=_card("lit_tpl_c_glycoside_split_v1"),
    )

    assert o_report.allowed_use == "executable_candidate"
    assert o_report.match_confidence == "exact_retron_match"
    assert o_report.selected_bond["retron_type"] == "o_glycoside"
    assert len(o_report.cut_fragments) == 2
    assert c_report.allowed_use == "advisory_or_rerank_only"
    assert c_report.mismatch_reasons == ["same_family_wrong_linkage_o_glycoside"]


def test_taxane_matches_side_chain_boundary_and_not_macrocycle():
    taxane_report = assess_template_applicability(
        target_smiles=TAXANE,
        frontier_smiles=TAXANE,
        template_card=_card("lit_tpl_taxane_c13_side_chain_split_v1"),
    )
    macro_report = assess_template_applicability(
        target_smiles=TAXANE,
        frontier_smiles=TAXANE,
        template_card=_card("lit_tpl_macrolactone_split_v1"),
    )

    assert taxane_report.allowed_use == "executable_candidate"
    assert taxane_report.selected_bond["retron_type"] == "taxane_c13_side_chain"
    assert macro_report.allowed_use == "advisory_or_rerank_only"
    assert macro_report.mismatch_reasons == ["same_family_wrong_linkage_taxane_ester_not_macrocycle"]


def test_bufadienolide_match_locates_steroid_pyrone_c_c_boundary():
    report = assess_template_applicability(
        target_smiles=BUFADIENOLIDE,
        frontier_smiles=BUFADIENOLIDE,
        template_card=_card("lit_tpl_bufadienolide_c17_pyrone_split_v1"),
    )

    assert report.allowed_use == "executable_candidate"
    assert report.selected_bond["retron_type"] == "bufadienolide_c17_pyrone"
    assert report.selected_bond["atom_symbols"] == ["C", "C"]
    assert len(report.cut_fragments) == 2


def test_macrolactone_ring_cut_records_single_seco_acid_fragment_with_two_dummies():
    report = assess_template_applicability(
        target_smiles=MACROLACTONE,
        frontier_smiles=MACROLACTONE,
        template_card=_card("lit_tpl_macrolactone_split_v1"),
    )
    cut = cut_report_from_applicability(report)

    assert report.allowed_use == "executable_candidate"
    assert report.selected_bond["bond_in_ring"] is True
    assert len(report.cut_fragments) == 1
    assert report.cut_fragments[0].count("*") == 2
    assert cut["allowed_use"] == "executable_candidate"


def test_analogy_only_template_does_not_enter_executable_candidate_even_if_retron_matches():
    card = _card("lit_tpl_o_glycoside_split_v1")
    card.scope_limits.append("analogy_only")

    report = assess_template_applicability(
        target_smiles=PHENOLIC_O_GLYCOSIDE,
        frontier_smiles=PHENOLIC_O_GLYCOSIDE,
        template_card=card,
    )

    assert report.allowed_use == "critique_or_retrieve_more"
    assert report.match_confidence == "analogy_only"
    assert report.mismatch_reasons == ["analogy_only_not_executable"]
