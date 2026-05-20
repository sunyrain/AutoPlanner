import unittest

from cascade_planner.cascade_search import (
    CascadeActionType,
    CascadeFailureKind,
    CascadeProgramSearch,
    CascadeRepairPolicy,
    CascadeSearchConfig,
    CascadeSearchController,
    CascadeSearchState,
    ChemEnzyProposalProvider,
    ChemicalTemplateProposalProvider,
    CofactorLedger,
    ConditionEnvelope,
    HeuristicCascadeValueModel,
    ProposalRequest,
    RetroChimeraProposalProvider,
    TemplateRelevanceProposalProvider,
    StaticProposalProvider,
    StepAnnotation,
    VerifierAugmentedCascadeValueModel,
    detect_cascade_failures,
    route_step_candidate_to_action,
    score_cascade_state,
)
from cascade_planner.baselines.route_contract import RouteStepCandidate


class CascadeSearchContractTest(unittest.TestCase):
    def test_retrochimera_provider_normalizes_syntheseus_reactions(self):
        class FakeMol:
            def __init__(self, smiles):
                self.smiles = smiles

        class FakeReaction:
            reactants = [FakeMol("CC"), FakeMol("O")]
            metadata = {"probability": 0.7, "individual_ranks": {"smiles_transformer": 0}}

        class FakeRetroChimera:
            def __call__(self, inputs, num_results):
                return [[FakeReaction()]]

        provider = RetroChimeraProposalProvider(model=FakeRetroChimera())

        rows = provider.predict("CCO", top_k=3)
        actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=3))

        self.assertEqual(rows[0]["source"], "retrochimera")
        self.assertEqual(rows[0]["rxn_smiles"], "CC.O>>CCO")
        self.assertEqual(rows[0]["score"], 0.7)
        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(actions[0].source, "retrochimera")

    def test_template_relevance_provider_normalizes_template_rows(self):
        class FakeOneStep:
            def predict(self, product, top_k=10, **_):
                return [
                    {
                        "reactant_smiles": ["CC", "O"],
                        "rxn_smiles": "CC.O>>CCO",
                        "source": "chem_enzy_onmt",
                        "score": 0.6,
                        "template": "fake_template",
                    }
                ]

        provider = TemplateRelevanceProposalProvider(one_step=FakeOneStep(), models=("template_relevance.reaxys",))

        rows = provider.predict("CCO", top_k=1)
        actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))

        self.assertEqual(rows[0]["source"], "template_relevance")
        self.assertEqual(rows[0]["proposal_type"], "template_relevance")
        self.assertEqual(rows[0]["type"], "template_relevance")
        self.assertEqual(rows[0]["template_relevance_model_count"], 1)
        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(actions[0].source, "template_relevance")

    def test_chemical_template_provider_normalizes_local_template_rows(self):
        class FakePreselector:
            available = True

        class FakeTemplateApplicator:
            max_templates = 20000
            max_templates_per_query = 500
            max_outcomes_per_template = 1
            generalize = 0
            template_paths = ["templates_uspto.csv.gz"]
            preselector = FakePreselector()
            pair_ranker = None

            def predict(self, product, top_k=10, ec_token="", skel_type=""):
                return [
                    {
                        "main_reactant": "CC",
                        "aux_reactants": ["O"],
                        "rxn_smiles": "CC.O>>CCO",
                        "source": "uspto_template",
                        "score": 0.7,
                        "template_id": "RR:test",
                        "reaction_type": skel_type or "acylation",
                    }
                ]

        provider = ChemicalTemplateProposalProvider(one_step=FakeTemplateApplicator(), expansion_topk=1)

        rows = provider.predict("CCO", top_k=1, metadata={"reaction_type": "acylation"})
        actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))

        self.assertEqual(rows[0]["source"], "chemtemplates")
        self.assertEqual(rows[0]["proposal_type"], "chemtemplates")
        self.assertEqual(rows[0]["template_ranker_mode"], "preselector")
        self.assertEqual(rows[0]["template_source"], "uspto_template")
        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(actions[0].source, "chemtemplates")

    def test_chemical_template_provider_attaches_weak_condition_predictions(self):
        class FakeTemplateApplicator:
            max_templates = 1
            max_templates_per_query = 1
            max_outcomes_per_template = 1
            generalize = 0
            preselector = None
            pair_ranker = None

            def predict(self, product, top_k=10, ec_token="", skel_type=""):
                return [
                    {
                        "reactant_smiles": ["CC", "O"],
                        "rxn_smiles": "CC.O>>CCO",
                        "source": "uspto_template",
                        "score": 0.7,
                    }
                ]

        class FakeConditionPredictor:
            def predict(self, rxn_smiles, top_k=1):
                return [{"Temperature": 25.0, "Solvent": "water", "Score": 0.42}]

        provider = ChemicalTemplateProposalProvider(
            one_step=FakeTemplateApplicator(),
            condition_predictor=FakeConditionPredictor(),
            predict_conditions=True,
            expansion_topk=1,
        )

        rows = provider.predict("CCO", top_k=1)
        actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))

        self.assertEqual(rows[0]["condition_predictions"][0]["Score"], 0.42)
        self.assertTrue(rows[0]["condition_prediction_reliable"])
        self.assertEqual(actions[0].step.condition.solvents, ["water"])
        self.assertEqual(actions[0].step.condition.confidence, 0.42)

    def test_condition_envelope_does_not_harden_outlier_temperature_predictions(self):
        self.assertIsNone(ConditionEnvelope.from_backend_prediction({"Temperature": 280.0, "Score": 0.7}))

        envelope = ConditionEnvelope.from_backend_prediction({"Temperature": 25.0, "Score": 0.7})

        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.temperature_c_min, 25.0)
        self.assertEqual(envelope.confidence, 0.7)
        self.assertEqual(envelope.raw_evidence[0]["Score"], 0.7)

    def test_chemical_template_provider_keeps_low_score_condition_as_metadata_only(self):
        class FakeTemplateApplicator:
            max_templates = 1
            max_templates_per_query = 1
            max_outcomes_per_template = 1
            generalize = 0
            preselector = None
            pair_ranker = None

            def predict(self, product, top_k=10, ec_token="", skel_type=""):
                return [{"reactant_smiles": ["CC"], "rxn_smiles": "CC>>CCO", "score": 0.7}]

        class FakeConditionPredictor:
            def predict(self, rxn_smiles, top_k=1):
                return [{"Temperature": 25.0, "Solvent": "water", "Score": 0.01}]

        provider = ChemicalTemplateProposalProvider(
            one_step=FakeTemplateApplicator(),
            condition_predictor=FakeConditionPredictor(),
            predict_conditions=True,
            expansion_topk=1,
        )

        action = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))[0]

        self.assertIsNone(action.step.condition)
        self.assertIn("low_condition_prediction_score", action.step.raw_metadata["condition_prediction_issues"])

    def test_state_serializes_stock_and_cofactor_closure(self):
        state = CascadeSearchState(
            target_smiles="CCO",
            open_leaves=["CC", "O"],
            cofactor_ledger=CofactorLedger(required={"NADH": 1.0}, regenerated={"NADH": 0.25}),
        )
        state.append_step(
            StepAnnotation(
                product_smiles="CCO",
                reactant_smiles=["CC", "O"],
                rxn_smiles="CC.O>>CCO",
                score=0.8,
                stock_status={"CC": True, "O": True},
            )
        )

        payload = state.to_dict()

        self.assertTrue(payload["stock_closed"])
        self.assertEqual(payload["cofactor_ledger"]["unclosed_requirements"], {"NADH": 0.75})

    def test_cost_penalizes_condition_and_cofactor_mismatch(self):
        good = CascadeSearchState(
            target_smiles="CCO",
            open_leaves=[],
            stage_partition=["one_pot", "one_pot"],
            cofactor_ledger=CofactorLedger(required={"NADH": 1.0}, regenerated={"NADH": 1.0}),
        )
        bad = CascadeSearchState(
            target_smiles="CCO",
            open_leaves=["missing"],
            stage_partition=["one_pot", "isolated"],
            cofactor_ledger=CofactorLedger(required={"NADH": 1.0}, regenerated={}),
        )
        for state in (good, bad):
            state.steps.extend([
                StepAnnotation(
                    product_smiles="CCO",
                    reactant_smiles=["CC=O"],
                    rxn_smiles="CC=O>>CCO",
                    score=0.9,
                    reaction_type="enzymatic",
                    ec_numbers=["1.1.1.1"] if state is good else [],
                    evidence_confidence=0.9,
                    condition=ConditionEnvelope(
                        temperature_c_min=20.0,
                        temperature_c_max=30.0,
                        ph_min=7.0,
                        ph_max=8.0,
                        solvents=["water"],
                    ),
                ),
                StepAnnotation(
                    product_smiles="CC=O",
                    reactant_smiles=["CC"],
                    rxn_smiles="CC>>CC=O",
                    score=0.9,
                    evidence_confidence=0.9,
                    condition=ConditionEnvelope(
                        temperature_c_min=22.0 if state is good else 80.0,
                        temperature_c_max=28.0 if state is good else 90.0,
                        ph_min=7.2,
                        ph_max=7.8,
                        solvents=["water"] if state is good else ["toluene"],
                    ),
                ),
            ])

        self.assertLess(score_cascade_state(good).total_cost, score_cascade_state(bad).total_cost)

    def test_native_state_exposes_stage_graph_and_typed_failures(self):
        state = CascadeSearchState(target_smiles="CCO", open_leaves=[])
        state.append_step(
            StepAnnotation(
                product_smiles="CCO",
                reactant_smiles=["CC=O"],
                rxn_smiles="CC=O>>CCO",
                reaction_type="enzymatic",
                cofactor_requirements={"NADH": 1.0},
                condition=ConditionEnvelope(
                    temperature_c_min=20,
                    temperature_c_max=30,
                    ph_min=7,
                    ph_max=8,
                    solvents=["water"],
                ),
            )
        )

        failures = detect_cascade_failures(state)
        categories = {failure.category for failure in failures}

        self.assertIn(CascadeFailureKind.COFACTOR_DEBT.value, categories)
        self.assertIn(CascadeFailureKind.ENZYME_EVIDENCE_WEAK.value, categories)
        self.assertEqual(state.to_dict()["stage_graph"]["stages"][0]["stage_id"], "stage_1")

    def test_missing_condition_is_unknown_not_a_conflict(self):
        state = CascadeSearchState(target_smiles="CCO", open_leaves=[])
        state.append_step(
            StepAnnotation(
                product_smiles="CCO",
                reactant_smiles=["CC"],
                rxn_smiles="CC>>CCO",
                score=0.9,
                source_model="retrochimera",
                stock_status={"CC": True},
            )
        )

        failures = detect_cascade_failures(state)

        categories = {failure.category for failure in failures}
        self.assertIn(CascadeFailureKind.CONDITION_MISSING.value, categories)
        self.assertNotIn(CascadeFailureKind.CONDITION_CONFLICT.value, categories)
        self.assertLess(score_cascade_state(state).condition_compatibility, 1.0)

    def test_condition_state_report_summarizes_route_level_risk(self):
        state = CascadeSearchState(target_smiles="CCCCO", open_leaves=[])
        state.append_step(
            StepAnnotation(
                product_smiles="CCCCO",
                reactant_smiles=["CCC"],
                rxn_smiles="CCC>>CCCCO",
                condition=ConditionEnvelope(
                    temperature_c_min=20,
                    temperature_c_max=30,
                    ph_min=7,
                    ph_max=8,
                    solvents=["water"],
                ),
                stage_id="stage_1",
            )
        )
        state.append_step(
            StepAnnotation(
                product_smiles="CCC",
                reactant_smiles=["CC"],
                rxn_smiles="CC>>CCC",
                condition=ConditionEnvelope(
                    temperature_c_min=120,
                    temperature_c_max=130,
                    ph_min=2,
                    ph_max=3,
                    solvents=["toluene"],
                ),
                stage_id="stage_2",
            )
        )
        payload = state.to_dict()
        condition_state = payload["condition_state"]

        self.assertEqual(condition_state["route_risk"], "high")
        self.assertTrue(condition_state["stepwise_required"])
        self.assertGreater(condition_state["temperature_span_c"], 90.0)
        self.assertIn("stage_2", {stage["stage_id"] for stage in condition_state["stage_summaries"]})

    def test_search_result_diagnostics_include_condition_state(self):
        provider = StaticProposalProvider({
            "CCCCO": [
                {
                    "product_smiles": "CCCCO",
                    "reactant_smiles": ["CCC"],
                    "rxn_smiles": "CCC>>CCCCO",
                    "score": 0.8,
                    "stock_status": {"CCC": True},
                    "condition": {"Temperature": 25, "pH": 7, "Solvent": "water"},
                }
            ]
        })
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi == "CCC",
            config=CascadeSearchConfig(max_depth=1, expansion_budget=4),
        )
        result = planner.search("CCCCO", n_results=1)[0]

        self.assertIn("condition_state", result.diagnostics)
        self.assertEqual(result.diagnostics["condition_state"]["route_risk"], "ok")

    def test_repair_policy_maps_failures_to_search_actions(self):
        state = CascadeSearchState(target_smiles="CCO", open_leaves=[])
        state.cofactor_ledger.required["NADH"] = 1.0
        failures = detect_cascade_failures(state)

        repairs = CascadeRepairPolicy.default().propose_repairs(state, failures)

        self.assertEqual(repairs[0].action_type, CascadeActionType.COFACTOR_REPAIR)
        self.assertEqual(repairs[0].module.cofactor_regenerations, {"NADH": 1.0})

    def test_chem_enzy_route_step_normalizes_to_cascade_action(self):
        step = RouteStepCandidate(
            product_smiles="CCO",
            reactant_smiles=["CC", "O"],
            rxn_smiles="CC.O>>CCO",
            source_model="graphfp",
            score=0.8,
            stock_status={"CC": True, "O": True},
            condition_predictions=[{"Temperature": 25, "pH": 7.5, "Solvent": "water"}],
            enzyme_ec_annotations=[{"ec_number": "1.1.1.1", "confidence": 0.9}],
        )

        action = route_step_candidate_to_action(step, provider_name=ChemEnzyProposalProvider.provider_name)

        self.assertEqual(action.action_type, CascadeActionType.RETROSYNTHETIC_STEP)
        self.assertEqual(action.step.ec_numbers, ["1.1.1.1"])
        self.assertEqual(action.step.condition.solvents, ["water"])

    def test_cascade_program_search_repairs_cofactor_debt(self):
        provider = StaticProposalProvider({
            "CCO": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC", "O"],
                    "rxn_smiles": "CC.O>>CCO",
                    "source": "enzyme_source",
                    "score": 0.9,
                    "reaction_type": "enzymatic",
                    "ec": "1.1.1.1",
                    "cofactor_requirements": {"NADH": 1.0},
                    "stock_status": {"CC": True, "O": True},
                    "condition": {"Temperature": 25, "pH": 7, "Solvent": "water"},
                }
            ]
        })
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi in {"CC", "O"},
            config=CascadeSearchConfig(max_depth=2, expansion_budget=10, branch_factor=2),
        )

        results = planner.search("CCO", n_results=1)

        self.assertTrue(results)
        self.assertTrue(results[0].solved)
        self.assertEqual(results[0].state.cofactor_ledger.unclosed_requirements(), {})
        self.assertIn("NADH", results[0].state.cofactor_ledger.regenerated)

    def test_search_accepts_formal_controller_value_model(self):
        class FixedValueModel:
            def predict(self, state):
                return HeuristicCascadeValueModel().predict(state)

        provider = StaticProposalProvider({
            "CCO": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC"],
                    "rxn_smiles": "CC>>CCO",
                    "score": 0.9,
                    "stock_status": {"CC": True},
                    "condition": {"Temperature": 25, "pH": 7, "Solvent": "water"},
                }
            ]
        })
        controller = CascadeSearchController(value_model=FixedValueModel(), metadata={"test": True})
        planner = CascadeProgramSearch(
            [provider],
            stock_checker=lambda smi: smi == "CC",
            config=CascadeSearchConfig(max_depth=1, expansion_budget=4),
            controller=controller,
        )

        result = planner.search("CCO", n_results=1)[0]

        self.assertTrue(result.solved)
        self.assertEqual(result.diagnostics["controller"]["value_model"], "FixedValueModel")
        self.assertEqual(result.diagnostics["controller"]["metadata"], {"test": True})

    def test_verifier_augmented_value_model_exposes_report(self):
        good = CascadeSearchState(target_smiles="CCCCO", open_leaves=[])
        good.append_step(
            StepAnnotation(
                product_smiles="CCCCO",
                reactant_smiles=["CCCC"],
                rxn_smiles="CCCC>>CCCCO",
                score=0.8,
                condition=ConditionEnvelope(
                    temperature_c_min=25,
                    temperature_c_max=30,
                    ph_min=7,
                    ph_max=8,
                    solvents=["water"],
                ),
            )
        )
        bad = CascadeSearchState(target_smiles="CCCCO", open_leaves=[])
        bad.append_step(
            StepAnnotation(
                product_smiles="CCCCO",
                reactant_smiles=["C"],
                rxn_smiles="C>>CCCCO",
                score=0.8,
                condition=ConditionEnvelope(
                    temperature_c_min=25,
                    temperature_c_max=30,
                    ph_min=7,
                    ph_max=8,
                    solvents=["water"],
                ),
            )
        )
        model = VerifierAugmentedCascadeValueModel(verifier_weight=1.0)

        good_pred = model.predict(good)
        bad_pred = model.predict(bad)

        self.assertGreater(good_pred.value, bad_pred.value)
        self.assertEqual(good_pred.metadata["verifier_report"]["reason_counts"], {})
        self.assertIn("atom_balance_violation", bad_pred.metadata["verifier_report"]["reason_counts"])


if __name__ == "__main__":
    unittest.main()
