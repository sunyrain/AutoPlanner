import unittest

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from cascade_planner.baselines.route_plausibility import audit_step_plausibility
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteCandidate, RouteStepCandidate
from cascade_planner.baselines.semisynthesis_rescue import (
    ACETIC_ANHYDRIDE,
    N_DEBENZOYLTAXOL,
    PHENYLISOSERINE_ZWITTERION,
    TBS_CHLORIDE,
    TEN_DEACETYLBACCATIN_III,
    known_advanced_precursor_record,
    semisynthesis_open_precursors,
    semisynthesis_rescue_routes,
    semisynthesis_upstream_candidate_precursors,
    stitch_semisynthesis_routes,
)


BUFOTALIN = "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
TBS_DEACETYLBUFOTALIN = "CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1"
DIACETYL_DEACETYLBUFOTALIN = "CC(=O)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](OC(C)=O)C[C@]32O)C1"


class SemisynthesisRescueTest(unittest.TestCase):
    def test_late_stage_o_acetylation_keeps_deacetylated_alcohol_core(self):
        routes = semisynthesis_rescue_routes(BUFOTALIN)

        self.assertEqual(len(routes), 1)
        route = routes[0]
        step = route.steps[0]
        target = Chem.MolFromSmiles(BUFOTALIN)
        precursor = Chem.MolFromSmiles(step.reactant_smiles[0])

        self.assertEqual(route.target_smiles, Chem.MolToSmiles(target, isomericSmiles=True))
        self.assertEqual(step.reactant_smiles[1], ACETIC_ANHYDRIDE)
        self.assertEqual(rdMolDescriptors.CalcMolFormula(target), "C26H36O6")
        self.assertEqual(rdMolDescriptors.CalcMolFormula(precursor), "C24H34O5")
        self.assertEqual(target.GetNumHeavyAtoms() - precursor.GetNumHeavyAtoms(), 3)
        self.assertTrue((route.raw_backend_metadata or {}).get("rescue_type"))
        self.assertTrue((route.raw_backend_metadata or {}).get("advanced_precursor_source_supported"))
        self.assertEqual(route.raw_backend_metadata["advanced_precursor_record"]["cas"], "465-19-0")
        self.assertTrue(step.stock_status[step.reactant_smiles[0]])

    def test_source_supported_precursor_remains_optional_upstream_candidate(self):
        routes = semisynthesis_rescue_routes(BUFOTALIN)
        precursor = routes[0].steps[0].reactant_smiles[0]

        self.assertEqual(semisynthesis_open_precursors(routes), [])
        self.assertEqual(semisynthesis_upstream_candidate_precursors(routes), [precursor])

    def test_known_deacetylbufotalin_precursor_record_is_available(self):
        routes = semisynthesis_rescue_routes(BUFOTALIN)
        precursor = routes[0].steps[0].reactant_smiles[0]
        record = known_advanced_precursor_record(precursor)

        self.assertEqual(record["name"], "Deacetylbufotalin")
        self.assertIn("Bufogenin B", record["synonyms"])

    def test_late_stage_o_acetylation_is_materially_supported_by_acetic_anhydride(self):
        routes = semisynthesis_rescue_routes(BUFOTALIN)
        audit = audit_step_plausibility(routes[0].steps[0])

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["reasons"], [])
        self.assertEqual(audit["raw_element_gains"], {})
        self.assertEqual(audit["unexplained_element_gains"], {})
        self.assertEqual(audit["reactant_counts"].get("C"), 28)
        self.assertEqual(audit["product_counts"].get("C"), 26)

    def test_multi_o_acetylated_known_precursor_gets_source_supported_rescue(self):
        routes = semisynthesis_rescue_routes(DIACETYL_DEACETYLBUFOTALIN)

        self.assertEqual(len(routes), 1)
        route = routes[0]
        step = route.steps[0]
        rescue = step.raw_backend_metadata["semisynthesis_rescue"]

        self.assertEqual(route.raw_backend_metadata["rescue_type"], "late_stage_multi_o_acetylation")
        self.assertEqual(rescue["acetylation_count"], 2)
        self.assertEqual(route.raw_backend_metadata["advanced_precursor_record"]["cas"], "465-19-0")
        self.assertIn("excess Ac2O", step.condition_predictions[0]["condition_label"])
        self.assertTrue(step.stock_status[step.reactant_smiles[0]])

    def test_multi_o_acetylation_is_materially_supported_by_acetic_anhydride(self):
        routes = semisynthesis_rescue_routes(DIACETYL_DEACETYLBUFOTALIN)
        audit = audit_step_plausibility(routes[0].steps[0])

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["reasons"], [])
        self.assertEqual(audit["raw_element_gains"], {})
        self.assertEqual(audit["unexplained_element_gains"], {})
        self.assertEqual(audit["reactant_counts"].get("C"), 28)
        self.assertEqual(audit["product_counts"].get("C"), 28)

    def test_tbs_protected_known_precursor_gets_source_supported_rescue(self):
        routes = semisynthesis_rescue_routes(TBS_DEACETYLBUFOTALIN)

        self.assertEqual(len(routes), 1)
        route = routes[0]
        step = route.steps[0]

        self.assertEqual(step.reactant_smiles[1], TBS_CHLORIDE)
        self.assertEqual(step.source_model, "semisynthesis_rescue.tbs_silylation")
        self.assertTrue((route.raw_backend_metadata or {}).get("advanced_precursor_source_supported"))
        self.assertEqual(route.raw_backend_metadata["advanced_precursor_record"]["cas"], "465-19-0")
        self.assertTrue(step.stock_status[step.reactant_smiles[0]])
        self.assertIn("TBSCl", step.condition_predictions[0]["condition_label"])

    def test_tbs_protection_is_materially_supported_by_tbs_chloride(self):
        routes = semisynthesis_rescue_routes(TBS_DEACETYLBUFOTALIN)
        audit = audit_step_plausibility(routes[0].steps[0])

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["reasons"], [])
        self.assertEqual(audit["raw_element_gains"], {})
        self.assertEqual(audit["unexplained_element_gains"], {})
        self.assertEqual(audit["reactant_counts"].get("Si"), 1)
        self.assertEqual(audit["product_counts"].get("Si"), 1)

    def test_taxane_10dab_semisynthesis_anchor_is_source_supported(self):
        routes = semisynthesis_rescue_routes(N_DEBENZOYLTAXOL)

        self.assertEqual(len(routes), 1)
        route = routes[0]
        step = route.steps[0]

        self.assertEqual(step.reactant_smiles[0], TEN_DEACETYLBACCATIN_III)
        self.assertIn(PHENYLISOSERINE_ZWITTERION, step.reactant_smiles)
        self.assertIn(ACETIC_ANHYDRIDE, step.reactant_smiles)
        self.assertEqual(route.raw_backend_metadata["rescue_type"], "taxane_10dab_side_chain_acetylation")
        self.assertTrue(route.raw_backend_metadata["advanced_precursor_source_supported"])
        self.assertEqual(route.raw_backend_metadata["advanced_precursor_record"]["cas"], "32981-86-5")
        self.assertEqual(step.enzyme_ec_annotations[0]["ec_number"], "2.3.1.167")
        self.assertTrue(step.stock_status[TEN_DEACETYLBACCATIN_III])

    def test_taxane_10dab_semisynthesis_anchor_is_materially_supported(self):
        routes = semisynthesis_rescue_routes(N_DEBENZOYLTAXOL)
        audit = audit_step_plausibility(routes[0].steps[0])

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["reasons"], [])
        self.assertEqual(audit["raw_element_gains"], {})
        self.assertEqual(audit["unexplained_element_gains"], {})

    def test_stitches_matching_upstream_precursor_route(self):
        anchor = semisynthesis_rescue_routes(BUFOTALIN)
        precursor = anchor[0].steps[0].reactant_smiles[0]
        anchor[0].steps[0].stock_status[precursor] = False
        self.assertEqual(semisynthesis_open_precursors(anchor), [precursor])
        upstream_step = RouteStepCandidate(
            product_smiles=precursor,
            reactant_smiles=["C", "CC"],
            rxn_smiles=f"C.CC>>{precursor}",
            source_model="unit_test",
            stock_status={"C": True, "CC": True},
        )
        upstream = BaselineRunResult(
            target_smiles=precursor,
            backend="ChemEnzyRetroPlanner",
            routes=[
                RouteCandidate(
                    target_smiles=precursor,
                    steps=[upstream_step],
                    backend="ChemEnzyRetroPlanner",
                    score=0.5,
                    solved=True,
                    route_rank=7,
                )
            ],
        )

        stitched = stitch_semisynthesis_routes(anchor, upstream)

        self.assertEqual(len(stitched), 1)
        self.assertEqual(len(stitched[0].steps), 2)
        self.assertTrue(stitched[0].solved)
        self.assertEqual(stitched[0].steps[0].source_model, "semisynthesis_rescue.o_acetylation")
        self.assertEqual(stitched[0].steps[1].source_model, "unit_test")
        self.assertEqual(stitched[0].raw_backend_metadata["route_class_hint"], "stitched_semisynthesis_upstream")

    def test_stitches_source_supported_precursor_if_upstream_route_exists(self):
        anchor = semisynthesis_rescue_routes(BUFOTALIN)
        precursor = anchor[0].steps[0].reactant_smiles[0]
        self.assertEqual(semisynthesis_open_precursors(anchor), [])
        upstream_step = RouteStepCandidate(
            product_smiles=precursor,
            reactant_smiles=["C", "CC"],
            rxn_smiles=f"C.CC>>{precursor}",
            source_model="unit_test",
            stock_status={"C": True, "CC": True},
        )
        upstream = BaselineRunResult(
            target_smiles=precursor,
            backend="ChemEnzyRetroPlanner",
            routes=[
                RouteCandidate(
                    target_smiles=precursor,
                    steps=[upstream_step],
                    backend="ChemEnzyRetroPlanner",
                    score=0.5,
                    solved=True,
                    route_rank=7,
                )
            ],
        )

        stitched = stitch_semisynthesis_routes(anchor, upstream)

        self.assertEqual(len(stitched), 1)
        self.assertEqual(len(stitched[0].steps), 2)
        self.assertEqual(stitched[0].raw_backend_metadata["route_class_hint"], "stitched_semisynthesis_upstream")


if __name__ == "__main__":
    unittest.main()
