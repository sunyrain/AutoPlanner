import unittest

from cascade_planner.agent.case_trace import ArtifactRecord, CaseBundle, RouteStatus
from cascade_planner.agent.chem_enzy_policy import compile_strategic_operator_from_case_bundle
from cascade_planner.agent.evidence_cards import EvidenceCard
from cascade_planner.agent.strategic_disconnection_miner import (
    mine_strategic_disconnection_cards,
    validate_strategic_disconnection_card,
)


class StrategicDisconnectionMinerAgentTest(unittest.TestCase):
    def test_traceable_literature_evidence_generates_search_ready_card(self):
        evidence = _strategic_evidence(target_relation="family_precedent")

        cards = mine_strategic_disconnection_cards(
            case_id="case",
            target_smiles="CCO",
            frontier_smiles="CCO",
            evidence_cards=[evidence],
        )

        self.assertEqual(len(cards), 1)
        card = cards[0]
        validation = validate_strategic_disconnection_card(
            card,
            validated_evidence_refs={evidence.evidence_id},
        )

        self.assertTrue(validation["accepted"], validation)
        self.assertTrue(validation["usable_for_search"])
        self.assertEqual(card.target_relation, "family_precedent")
        self.assertEqual(card.disconnection_type, "fragment_coupling")
        self.assertIn("Expose", card.strategic_subgoal)
        self.assertIn("product-like", card.forbidden_fake_terminal_implication)

    def test_analogy_and_failed_routes_do_not_become_search_ready_disconnections(self):
        analogy = _strategic_evidence(target_relation="analogy_only")
        failed = _strategic_evidence(
            target_relation="family_precedent",
            evidence_id="ev_failed",
            route_role="negative_guidance",
            record={"failed": True, "retrosynthetic_move": {"planner_hint": "avoid this break"}},
        )

        cards = mine_strategic_disconnection_cards(
            case_id="case",
            target_smiles="CCO",
            frontier_smiles="CCO",
            evidence_cards=[analogy, failed],
        )
        by_ref = {card.evidence_refs[0]: card for card in cards}

        self.assertFalse(by_ref["ev_strategy"].usable_for_search)
        self.assertTrue(validate_strategic_disconnection_card(by_ref["ev_strategy"])["accepted"])
        by_ref["ev_strategy"].usable_for_search = True
        self.assertIn(
            "analogy_only_not_search_ready",
            validate_strategic_disconnection_card(by_ref["ev_strategy"])["reasons"],
        )
        self.assertFalse(by_ref["ev_failed"].usable_for_search)
        self.assertEqual(by_ref["ev_failed"].route_claim, "failed")

    def test_invalid_disconnection_card_cannot_influence_policy_compilation(self):
        bundle = _policy_bundle_with_invalid_disconnection_card()

        with self.assertRaisesRegex(ValueError, "no validated search-ready StrategicDisconnectionCard"):
            compile_strategic_operator_from_case_bundle(bundle)


def _strategic_evidence(
    *,
    target_relation: str,
    evidence_id: str = "ev_strategy",
    route_role: str = "strategic_disconnection",
    record: dict | None = None,
) -> EvidenceCard:
    record = record or {
        "retrosynthetic_move": {
            "break_bonds": ["C17 steroid carbon to pyrone carbon"],
            "suggested_precursor_roles": ["steroid core partner", "pyrone partner"],
            "planner_hint": "Expose steroid and pyrone fragments.",
        },
        "use_policy": {
            "hard_reject_counterexamples": [
                "product-like advanced analogue must not close as stock",
            ],
        },
    }
    return EvidenceCard(
        evidence_id=evidence_id,
        case_id="case",
        source_type="literature",
        source_title="Traceable strategic route",
        url="https://example.org/route",
        target_relation=target_relation,
        claim_type="strategic_disconnection",
        route_role=route_role,
        confidence="high",
        source_record_id=f"{evidence_id}_record",
        family_id="bufadienolide_steroid",
        source_metadata={"record": record},
        validation_status="validated",
    )


def _policy_bundle_with_invalid_disconnection_card() -> CaseBundle:
    bundle = CaseBundle(case_id="case", route_status=RouteStatus.PARTIAL_ANCHOR)
    bundle.append_artifact(ArtifactRecord(
        artifact_id="hybrid_route_package",
        case_id="case",
        artifact_type="HybridRoutePackage",
        payload={
            "case_id": "case",
            "route_status": "partial_anchor",
            "frontier": {"frontier_smiles": "CCO", "flags": ["unresolved_core"]},
            "literature_evidence_refs": ["ev_anchor"],
        },
        evidence_refs=["ev_anchor"],
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="route_package_validation",
        case_id="case",
        artifact_type="RoutePackageValidation",
        payload={"case_id": "case", "accepted": True, "route_status": "partial_anchor"},
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="evidence_cards",
        case_id="case",
        artifact_type="EvidenceCardList",
        payload=[
            {
                "evidence_id": "ev_anchor",
                "case_id": "case",
                "source_type": "literature",
                "source_title": "Traceable anchor",
                "target_relation": "family_precedent",
                "claim_type": "route_anchor",
                "route_role": "route_anchor",
                "confidence": "high",
                "url": "https://example.org/anchor",
                "source_record_id": "anchor_record",
                "source_metadata": {"record": {"smiles": "CC"}},
                "validation_status": "validated",
                "schema_version": "evidence_card.v1",
            }
        ],
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="literature_candidates",
        case_id="case",
        artifact_type="LiteratureCandidateList",
        payload=[
            {
                "candidate_id": "candidate_anchor",
                "case_id": "case",
                "candidate_kind": "route_anchor",
                "target_smiles": "CCO",
                "product_smiles": "CCO",
                "precursor_smiles": ["CC"],
                "rxn_smiles": "",
                "reaction_class": "route_anchor",
                "strategic_bond": "multi_step_anchor",
                "literature_basis": "Traceable anchor",
                "use_case": "multi_step_anchor_planning_material",
                "confidence": "high",
                "evidence_refs": ["ev_anchor"],
                "source_record_refs": ["anchor_record"],
                "route_anchor_role": "semisynthesis_anchor",
                "validation_status": "validated",
                "schema_version": "literature_candidate.v1",
            }
        ],
        evidence_refs=["ev_anchor"],
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="strategic_disconnection_cards",
        case_id="case",
        artifact_type="StrategicDisconnectionCardList",
        payload=[
            {
                "card_id": "sd_bad",
                "case_id": "case",
                "target_smiles": "CCO",
                "frontier_smiles": "CCO",
                "target_relation": "analogy_only",
                "route_claim": "unknown",
                "disconnection_type": "fragment_coupling",
                "strategic_subgoal": "bad analog-only hint",
                "forbidden_fake_terminal_implication": "same-scaffold guard",
                "usable_for_search": True,
                "evidence_refs": ["ev_anchor"],
                "validation_status": "validated",
                "schema_version": "strategic_disconnection_card.v1",
            }
        ],
        evidence_refs=["ev_anchor"],
    ))
    return bundle


if __name__ == "__main__":
    unittest.main()
