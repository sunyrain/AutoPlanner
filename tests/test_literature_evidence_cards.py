import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.agent.evidence_cards import EvidenceCard, validate_evidence_card, write_evidence_jsonl, load_evidence_jsonl
from cascade_planner.agent.literature_research import build_literature_task, retrieve_literature_evidence
from cascade_planner.agent.target_profile import build_target_profile


class LiteratureEvidenceCardsTest(unittest.TestCase):
    def test_evidence_card_requires_traceable_source(self):
        card = EvidenceCard(
            evidence_id="ev_missing",
            case_id="case",
            source_type="literature",
            source_title="Untraceable claim",
            target_relation="family_precedent",
            claim_type="strategic_disconnection",
            route_role="strategic_disconnection",
        )

        validation = validate_evidence_card(card)

        self.assertFalse(validation["accepted"])
        self.assertIn("untraceable_source", validation["reasons"])

    def test_analogy_only_disconnection_is_draft_only(self):
        card = EvidenceCard(
            evidence_id="ev_analog",
            case_id="case",
            source_type="literature",
            source_title="Analog-only route",
            url="https://example.org/analog",
            target_relation="analogy_only",
            claim_type="strategic_disconnection",
            route_role="strategic_disconnection",
        )

        validation = validate_evidence_card(card)

        self.assertFalse(validation["accepted"])
        self.assertIn("analogy_only_disconnection_not_search_ready", validation["reasons"])

    def test_jsonl_round_trip_sets_validation_status(self):
        card = EvidenceCard(
            evidence_id="ev_ok",
            case_id="case",
            source_type="literature",
            source_title="Traceable route",
            url="https://example.org/route",
            target_relation="family_precedent",
            claim_type="strategic_disconnection",
            route_role="strategic_disconnection",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            write_evidence_jsonl([card], path)
            loaded = load_evidence_jsonl(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].validation_status, "validated")

    def test_local_strategic_db_retrieval_returns_traceable_bufadienolide_evidence(self):
        profile = build_target_profile(
            "CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            target_name="bufadienolide_like",
            family_hint="bufadienolide, steroid, pyrone",
        )
        task = build_literature_task(profile, profile.isomeric_smiles, query_budget=3)

        cards, report = retrieve_literature_evidence(task)

        self.assertFalse(report["unresolved_literature_gap"])
        self.assertGreaterEqual(len(cards), 1)
        self.assertTrue(any(card.family_id == "bufadienolide_steroid" for card in cards))
        self.assertTrue(all(card.url or card.doi or card.local_ref for card in cards))

    def test_statin_member_specific_records_do_not_cross_scope(self):
        summary = json.loads(Path("docs/statins/summary.json").read_text(encoding="utf-8"))
        targets = {row["safe"]: row["smiles"] for row in summary["targets"]}
        expected = {
            "atorvastatin": "atorvastatin_paal_knorr_convergent_assembly",
            "cerivastatin": "cerivastatin_pyridine_wittig_side_chain_convergence",
            "fluvastatin": "fluvastatin_aldol_wittig_reduction_process_window",
            "pitavastatin": "pitavastatin_quinoline_side_chain_coupling",
            "rosuvastatin": "rosuvastatin_pyrimidine_wittig_biocatalytic_side_chain",
        }
        member_specific = set(expected.values())

        for safe, expected_source in expected.items():
            profile = build_target_profile(
                targets[safe],
                target_name=f"{safe}_retrieval_scope_test",
                family_hint=f"{safe}, synthetic statin, syn-3,5-dihydroxy acid side chain, HWE, Wittig",
            )
            task = build_literature_task(profile, profile.isomeric_smiles, query_budget=6)

            cards, report = retrieve_literature_evidence(task)
            sources = {card.source_record_id for card in cards}

            self.assertFalse(report["unresolved_literature_gap"], safe)
            self.assertIn(expected_source, sources, safe)
            self.assertEqual(sources & member_specific, {expected_source}, safe)


if __name__ == "__main__":
    unittest.main()
