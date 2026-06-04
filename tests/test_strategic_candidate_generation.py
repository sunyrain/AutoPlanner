import unittest

from cascade_planner.agent.evidence_cards import EvidenceCard
from cascade_planner.agent.strategic_candidate_generation import (
    generate_literature_candidates,
    validate_literature_candidate,
)


class StrategicCandidateGenerationTest(unittest.TestCase):
    def test_generates_three_candidate_kinds_from_traceable_evidence(self):
        cards = [_strategic_card(), _anchor_card()]

        candidates = generate_literature_candidates(
            case_id="buf_case",
            target_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            frontier_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            evidence_cards=cards,
        )

        kinds = {candidate.candidate_kind for candidate in candidates}
        self.assertIn("exact_fragment_retro", kinds)
        self.assertIn("forward_surrogate", kinds)
        self.assertIn("route_anchor", kinds)

    def test_forward_surrogate_is_parseable_but_marked_not_lab_procedure(self):
        candidates = generate_literature_candidates(
            case_id="buf_case",
            target_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            frontier_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            evidence_cards=[_strategic_card()],
        )
        surrogate = next(candidate for candidate in candidates if candidate.candidate_kind == "forward_surrogate")

        validation = validate_literature_candidate(surrogate)

        self.assertTrue(validation["accepted"], validation)
        self.assertTrue(surrogate.not_lab_procedure)
        self.assertIn("not claimed", surrogate.surrogate_reason)

    def test_route_anchor_with_single_step_rxn_is_rejected(self):
        anchor = _anchor_card()
        candidates = generate_literature_candidates(
            case_id="buf_case",
            target_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            frontier_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            evidence_cards=[anchor],
        )
        route_anchor = next(candidate for candidate in candidates if candidate.candidate_kind == "route_anchor")
        route_anchor.rxn_smiles = "CCO>>CC=O"

        validation = validate_literature_candidate(route_anchor)

        self.assertFalse(validation["accepted"])
        self.assertIn("route_anchor_must_not_have_single_step_rxn", validation["reasons"])

    def test_candidate_without_evidence_refs_is_rejected(self):
        candidates = generate_literature_candidates(
            case_id="buf_case",
            target_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            frontier_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            evidence_cards=[_strategic_card()],
        )
        candidate = candidates[0]
        candidate.evidence_refs = []

        validation = validate_literature_candidate(candidate)

        self.assertFalse(validation["accepted"])
        self.assertIn("missing_evidence_refs", validation["reasons"])


def _strategic_card() -> EvidenceCard:
    return EvidenceCard(
        evidence_id="ev_buf_c17",
        case_id="buf_case",
        source_type="literature",
        source_title="Unified Total Synthesis of Five Bufadienolides",
        url="https://pubs.acs.org/doi/10.1021/acs.orglett.0c03251",
        target_relation="family_precedent",
        claim_type="strategic_disconnection",
        route_role="strategic_disconnection",
        confidence="high",
        source_record_id="bufadienolide_c17_pyrone_installation",
        family_id="bufadienolide_steroid",
        source_metadata={
            "record_type": "disconnection",
            "record": {
                "retrosynthetic_move": {
                    "break_bonds": ["C17 steroid carbon to 2-pyrone substituent"],
                    "forward_logic": ["C-C coupling"],
                    "suggested_precursor_roles": ["steroid partner", "2-pyrone partner"],
                    "planner_hint": "Expose steroid and pyrone fragments.",
                },
                "use_policy": {"proposal_source": "candidate_seed_after_substructure_match"},
            },
        },
        validation_status="validated",
    )


def _anchor_card() -> EvidenceCard:
    return EvidenceCard(
        evidence_id="ev_androstenedione",
        case_id="buf_case",
        source_type="literature",
        source_title="Bufadienolide synthesis from androstenedione",
        url="https://pubmed.ncbi.nlm.nih.gov/39496285/",
        target_relation="family_precedent",
        claim_type="route_anchor",
        route_role="route_anchor",
        confidence="medium_high",
        source_record_id="steroid_chiral_pool_androstenedione",
        family_id="bufadienolide_steroid",
        route_role_detail="steroid_chiral_pool_starting_material",
        source_metadata={
            "record_type": "anchor",
            "record": {
                "anchor_id": "steroid_chiral_pool_androstenedione",
                "name": "Androstenedione",
                "role": "steroid_chiral_pool_starting_material",
                "acceptance_policy": "acceptable_strategic_anchor_not_stock_solve_by_itself",
            },
        },
        validation_status="validated",
    )


if __name__ == "__main__":
    unittest.main()
