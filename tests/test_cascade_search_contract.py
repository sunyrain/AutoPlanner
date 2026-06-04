import unittest

import joblib

from cascade_planner.cascade_search import (
    AiZynthFinderONNXProposalProvider,
    CascadeActionType,
    CascadeFailureKind,
    CascadeProgramSearch,
    CascadeRepairPolicy,
    CascadeSearchConfig,
    CascadeSearchController,
    CascadeSearchState,
    ChemEnzyContextONMTProposalProvider,
    ChemEnzyProposalProvider,
    ChemicalTemplateProposalProvider,
    CofactorLedger,
    ConditionEnvelope,
    HeuristicCascadeValueModel,
    LegalCorpusProposalProvider,
    ProposalRequest,
    RetroChimeraProposalProvider,
    RetroKNNProposalProvider,
    TemplateRelevanceProposalProvider,
    StaticProposalProvider,
    StepAnnotation,
    VerifierAugmentedCascadeValueModel,
    detect_cascade_failures,
    route_step_candidate_to_action,
    score_cascade_state,
)
from cascade_planner.cascade_search.proposals import FallbackProposalProvider
from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.cascade_search.proposal_preference import feature_names


class _FakePreferenceModel:
    classes_ = [0, 1]

    def predict_proba(self, x):
        out = []
        for row in x:
            n_reactants = float(row[-30])
            side_equals_product = float(row[-14])
            score = max(0.01, min(0.99, 0.8 if n_reactants > 1.5 and side_equals_product < 0.5 else 0.2))
            out.append([1.0 - score, score])
        return out


class CascadeSearchContractTest(unittest.TestCase):
    def test_legal_corpus_provider_returns_exact_product_then_nearest_legal_candidates(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "meta.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        '{"source":"toy","source_row_id":1,"product":"CCO","reactants":["CC","O"],"reaction_smiles":"CC.O>>CCO"}',
                        '{"source":"toy","source_row_id":2,"product":"CCN","reactants":["CC","N"],"reaction_smiles":"CC.N>>CCN"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            provider = LegalCorpusProposalProvider(corpus, candidate_pool_size=2)

            exact_actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=2))
            nearest_actions = provider.propose(ProposalRequest("CCCl", CascadeSearchState.initial("CCCl"), top_k=1))

        self.assertEqual(exact_actions[0].source, "legal_corpus")
        self.assertEqual(exact_actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(exact_actions[0].step.raw_metadata["match_type"], "exact_product")
        self.assertEqual(nearest_actions[0].step.raw_metadata["match_type"], "nearest_product")
        self.assertIn("corpus_reaction_smiles", nearest_actions[0].step.raw_metadata)
        self.assertEqual(provider.last_diagnostics.provider_name, "legal_corpus")

    def test_legal_corpus_provider_can_reuse_index_cache(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "meta.jsonl"
            cache = root / "legal_index.pkl"
            corpus.write_text(
                '{"source":"toy","source_row_id":1,"product":"CCO","reactants":["CC","O"],"reaction_smiles":"CC.O>>CCO"}\n',
                encoding="utf-8",
            )
            provider = LegalCorpusProposalProvider(corpus, index_cache_path=cache)
            provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))
            cached_provider = LegalCorpusProposalProvider(corpus, index_cache_path=cache)

            actions = cached_provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))

        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertTrue(cached_provider.last_diagnostics.metadata["loaded_from_cache"])
        self.assertEqual(cached_provider.last_diagnostics.metadata["cache_status"], "hit")

    def test_legal_corpus_provider_does_not_overwrite_mismatched_cache(self):
        import pickle
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus_a = root / "a.meta.jsonl"
            corpus_b = root / "b.meta.jsonl"
            cache = root / "legal_index.pkl"
            corpus_a.write_text(
                '{"source":"toy","source_row_id":1,"product":"CCO","reactants":["CC","O"],"reaction_smiles":"CC.O>>CCO"}\n',
                encoding="utf-8",
            )
            corpus_b.write_text(
                '{"source":"toy","source_row_id":2,"product":"CCN","reactants":["CC","N"],"reaction_smiles":"CC.N>>CCN"}\n',
                encoding="utf-8",
            )
            provider_a = LegalCorpusProposalProvider(corpus_a, index_cache_path=cache)
            provider_a.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))
            before = pickle.loads(cache.read_bytes())

            provider_b = LegalCorpusProposalProvider(corpus_b, index_cache_path=cache)
            actions = provider_b.propose(ProposalRequest("CCN", CascadeSearchState.initial("CCN"), top_k=1))
            after = pickle.loads(cache.read_bytes())

        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "N"])
        self.assertEqual(provider_b.last_diagnostics.metadata["cache_status"], "signature_mismatch")
        self.assertEqual(before["source_signature"], after["source_signature"])

    def test_context_onmt_provider_builds_context_source_and_actions(self):
        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self):
                super().__init__(model_path="relative_context.pt")

            def predict_pretokenized_source(self, source, *, product_smiles, top_k):
                self.seen_source = source
                return [
                    {
                        "product_smiles": product_smiles,
                        "reactant_smiles": ["CC", "O"],
                        "rxn_smiles": f"CC.O>>{product_smiles}",
                        "source": self.provider_name,
                        "score": 0.7,
                    }
                ]

        state = CascadeSearchState.initial("CCO")
        provider = FakeProvider()

        actions = provider.propose(ProposalRequest("CCO", state, top_k=1))

        self.assertTrue(provider.model_path.is_absolute())
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(actions[0].source, "chem_enzy_context_onmt")
        self.assertIn("<target> C C O <product> C C O", provider.seen_source)
        self.assertEqual(provider.last_diagnostics.returned, 1)

    def test_context_onmt_provider_converts_log_scores_to_probabilities(self):
        import sys
        import types

        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self):
                super().__init__(model_path="/tmp/context.pt")

            def _ensure_translator(self):
                return types.SimpleNamespace(), object()

        provider = FakeProvider()

        def fake_translate(translator, opt, rows):
            return [[-1.0]], [["C C . O"]]

        names = ["onmt", "onmt.bin", "onmt.bin.translate"]
        original = {name: sys.modules.get(name) for name in names}
        onmt_module = types.ModuleType("onmt")
        onmt_bin_module = types.ModuleType("onmt.bin")
        onmt_translate_module = types.ModuleType("onmt.bin.translate")
        onmt_translate_module.translate = fake_translate
        onmt_module.bin = onmt_bin_module
        onmt_bin_module.translate = onmt_translate_module
        sys.modules["onmt"] = onmt_module
        sys.modules["onmt.bin"] = onmt_bin_module
        sys.modules["onmt.bin.translate"] = onmt_translate_module
        try:
            rows = provider.predict_pretokenized_source("<target> C C O <product> C C O", product_smiles="CCO", top_k=1)
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertAlmostEqual(rows[0]["score"], 0.367879, places=5)
        self.assertEqual(rows[0]["onmt_log_score"], -1.0)

    def test_context_onmt_provider_filters_self_reactions(self):
        import sys
        import types

        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self):
                super().__init__(model_path="/tmp/context.pt")

            def _ensure_translator(self):
                return types.SimpleNamespace(), object()

        provider = FakeProvider()

        def fake_translate(translator, opt, rows):
            return [[-0.1, -1.0]], [["C C O", "C C . O"]]

        names = ["onmt", "onmt.bin", "onmt.bin.translate"]
        original = {name: sys.modules.get(name) for name in names}
        onmt_module = types.ModuleType("onmt")
        onmt_bin_module = types.ModuleType("onmt.bin")
        onmt_translate_module = types.ModuleType("onmt.bin.translate")
        onmt_translate_module.translate = fake_translate
        onmt_module.bin = onmt_bin_module
        onmt_bin_module.translate = onmt_translate_module
        sys.modules["onmt"] = onmt_module
        sys.modules["onmt.bin"] = onmt_bin_module
        sys.modules["onmt.bin.translate"] = onmt_translate_module
        try:
            rows = provider.predict_pretokenized_source("<target> C C O <product> C C O", product_smiles="CCO", top_k=2)
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reactant_smiles"], ["CC", "O"])

    def test_context_onmt_provider_filters_invalid_and_duplicate_predictions(self):
        import sys
        import types

        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self):
                super().__init__(model_path="/tmp/context.pt")

            def _ensure_translator(self):
                return types.SimpleNamespace(), object()

        provider = FakeProvider()

        def fake_translate(translator, opt, rows):
            return [[-0.1, -0.2, -0.3, -0.4]], [["C 1", "O . C C", "C C . O", "C C C"]]

        names = ["onmt", "onmt.bin", "onmt.bin.translate"]
        original = {name: sys.modules.get(name) for name in names}
        onmt_module = types.ModuleType("onmt")
        onmt_bin_module = types.ModuleType("onmt.bin")
        onmt_translate_module = types.ModuleType("onmt.bin.translate")
        onmt_translate_module.translate = fake_translate
        onmt_module.bin = onmt_bin_module
        onmt_bin_module.translate = onmt_translate_module
        sys.modules["onmt"] = onmt_module
        sys.modules["onmt.bin"] = onmt_bin_module
        sys.modules["onmt.bin.translate"] = onmt_translate_module
        try:
            rows = provider.predict_pretokenized_source("<target> C C O <product> C C O", product_smiles="CCO", top_k=4)
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual([row["reactant_smiles"] for row in rows], [["O", "CC"], ["CCC"]])
        self.assertEqual(rows[0]["onmt_log_score"], -0.2)
        self.assertEqual(rows[1]["onmt_log_score"], -0.4)
        self.assertEqual(rows[0]["validity_filter_rejected_count"], 2)
        self.assertEqual(rows[0]["validity_filter_rejected_reasons"], ["duplicate_canonical_side", "invalid_reactant_molecule"])

    def test_context_onmt_provider_oversamples_before_validity_filtering(self):
        import sys
        import types

        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self):
                super().__init__(model_path="/tmp/context.pt", topk=2, beam_size=2, raw_topk_multiplier=3)

            def _ensure_translator(self):
                return types.SimpleNamespace(topk=2, beam_size=2, n_best=2), object()

        provider = FakeProvider()
        seen = {}

        def fake_translate(translator, opt, rows):
            seen["topk"] = opt.topk
            seen["beam_size"] = opt.beam_size
            seen["n_best"] = opt.n_best
            return [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]], [["C 1", "C C O", "C C C", "C C . O", "N", "O"]]

        names = ["onmt", "onmt.bin", "onmt.bin.translate"]
        original = {name: sys.modules.get(name) for name in names}
        onmt_module = types.ModuleType("onmt")
        onmt_bin_module = types.ModuleType("onmt.bin")
        onmt_translate_module = types.ModuleType("onmt.bin.translate")
        onmt_translate_module.translate = fake_translate
        onmt_module.bin = onmt_bin_module
        onmt_bin_module.translate = onmt_translate_module
        sys.modules["onmt"] = onmt_module
        sys.modules["onmt.bin"] = onmt_bin_module
        sys.modules["onmt.bin.translate"] = onmt_translate_module
        try:
            rows = provider.predict_pretokenized_source("<target> C C O <product> C C O", product_smiles="CCO", top_k=2)
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(seen, {"topk": 6, "beam_size": 6, "n_best": 6})
        self.assertEqual([row["reactant_smiles"] for row in rows], [["CCC"], ["CC", "O"]])
        self.assertEqual(rows[0]["raw_top_k"], 6)
        self.assertEqual(rows[0]["requested_top_k"], 2)

    def test_context_onmt_provider_respects_context_step_limit(self):
        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self):
                super().__init__(model_path="/tmp/context.pt", max_context_step=1)

            def predict_pretokenized_source(self, source, *, product_smiles, top_k):
                raise AssertionError("provider should not call ONMT beyond max_context_step")

        state = CascadeSearchState.initial("CCO")
        state.append_step(StepAnnotation(product_smiles="CCO", reactant_smiles=["CC"], rxn_smiles="CC>>CCO"))
        provider = FakeProvider()

        actions = provider.propose(ProposalRequest("CC", state, top_k=2))

        self.assertEqual(actions, [])
        self.assertEqual(provider.last_diagnostics.metadata["skipped_reason"], "context_step_limit")

    def test_context_onmt_provider_can_rerank_with_preference_scorer(self):
        import tempfile

        class FakeProvider(ChemEnzyContextONMTProposalProvider):
            def __init__(self, scorer_path):
                super().__init__(
                    model_path="/tmp/context.pt",
                    preference_scorer_path=scorer_path,
                    preference_rerank=True,
                )

            def predict_pretokenized_source(self, source, *, product_smiles, top_k):
                rows = [
                    {
                        "product_smiles": product_smiles,
                        "reactant_smiles": ["CCC"],
                        "rxn_smiles": f"CCC>>{product_smiles}",
                        "source": self.provider_name,
                        "score": 0.9,
                    },
                    {
                        "product_smiles": product_smiles,
                        "reactant_smiles": ["CC", "O"],
                        "rxn_smiles": f"CC.O>>{product_smiles}",
                        "source": self.provider_name,
                        "score": 0.2,
                    },
                ]
                return self._score_filter_and_rank_preferences(rows, product_smiles=product_smiles)

        with tempfile.TemporaryDirectory() as td:
            scorer_path = f"{td}/scorer.joblib"
            joblib.dump(
                {
                    "model": _FakePreferenceModel(),
                    "n_bits": 8,
                    "feature_names": feature_names(8),
                },
                scorer_path,
            )
            provider = FakeProvider(scorer_path)
            actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=2))

        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertGreater(actions[0].step.raw_metadata["preference_score"], actions[1].step.raw_metadata["preference_score"])
        self.assertEqual(actions[0].step.raw_metadata["preference_rank"], 1)
        self.assertTrue(provider.last_diagnostics.metadata["preference_rerank"])

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

    def test_aizynthfinder_onnx_provider_converts_policy_actions(self):
        class FakeMol:
            def __init__(self, smiles):
                self.smiles = smiles

        class FakeAction:
            metadata = {
                "policy_probability": 0.42,
                "template_hash": "abc",
                "policy_name": "ringbreaker",
            }
            reactants = [(FakeMol("CC"), FakeMol("O"))]

        class FakePolicy:
            def get_actions(self, mols):
                self.last_mols = mols
                return [FakeAction()], [0.42]

        policy = FakePolicy()
        provider = AiZynthFinderONNXProposalProvider(
            policy=policy,
            policy_name="ringbreaker",
            tree_molecule_factory=lambda smiles: {"smiles": smiles},
        )

        rows = provider.predict("CCO", top_k=3)
        actions = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=3))

        self.assertEqual(rows[0]["source"], "aizynth_onnx")
        self.assertEqual(rows[0]["proposal_type"], "aizynthfinder_onnx_policy")
        self.assertEqual(rows[0]["rxn_smiles"], "CC.O>>CCO")
        self.assertEqual(rows[0]["score"], 0.42)
        self.assertEqual(actions[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(actions[0].source, "aizynth_onnx")
        self.assertEqual(provider.last_diagnostics.metadata["policy_name"], "ringbreaker")

    def test_retroknn_provider_marks_known_reaction_retrieval(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "real_reactions.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        '{"source":"real","source_row_id":1,"product":"CCO","reactants":["CC","O"],"reaction_smiles":"CC.O>>CCO"}',
                        '{"source":"real","source_row_id":2,"product":"CCN","reactants":["CC","N"],"reaction_smiles":"CC.N>>CCN"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            provider = RetroKNNProposalProvider(corpus, candidate_pool_size=2)

            exact = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=2))
            nearest = provider.propose(ProposalRequest("CCCl", CascadeSearchState.initial("CCCl"), top_k=1))

        self.assertEqual(exact[0].source, "retroknn")
        self.assertEqual(exact[0].step.raw_metadata["proposal_type"], "retroknn_retrieval")
        self.assertEqual(exact[0].step.raw_metadata["type"], "retroknn_known_reaction_retrieval")
        self.assertEqual(exact[0].step.reactant_smiles, ["CC", "O"])
        self.assertEqual(nearest[0].step.raw_metadata["retrieval_match_type"], "nearest_product")
        self.assertEqual(provider.last_diagnostics.provider_name, "retroknn")

    def test_fallback_provider_only_calls_inner_when_primary_is_weak(self):
        class FakeInner:
            provider_name = "template_relevance"

            def __init__(self):
                self.calls = 0

            def propose(self, request):
                self.calls += 1
                return [
                    {
                        "reactant_smiles": ["C", "O"],
                        "rxn_smiles": "C.O>>CO",
                        "source": "template_relevance",
                        "score": 0.5,
                    }
                ]

        primary = StaticProposalProvider(
            {
                "CCO": [
                    {
                        "reactant_smiles": ["CC", "O"],
                        "rxn_smiles": "CC.O>>CCO",
                        "score": 0.7,
                    }
                ],
                "CO": [],
            }
        )
        inner = FakeInner()
        provider = FallbackProposalProvider(
            inner,
            primary=primary,
            trigger_min_primary=1,
            max_calls=1,
            provider_name="template_relevance",
        )

        skipped = provider.propose(ProposalRequest("CCO", CascadeSearchState.initial("CCO"), top_k=1))
        called = provider.propose(ProposalRequest("CO", CascadeSearchState.initial("CO"), top_k=1))
        capped = provider.propose(ProposalRequest("CN", CascadeSearchState.initial("CN"), top_k=1))

        self.assertEqual(skipped, [])
        self.assertEqual(len(called), 1)
        self.assertEqual(capped, [])
        self.assertEqual(inner.calls, 1)
        self.assertEqual(provider.last_diagnostics.metadata["skipped_reason"], "max_calls_reached")

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

    def test_multi_provider_search_interleaves_sources_before_branch_cutoff(self):
        class NamedProvider(StaticProposalProvider):
            def __init__(self, name, proposals_by_leaf):
                super().__init__(proposals_by_leaf)
                self.provider_name = name

        provider_a = NamedProvider(
            "source_a",
            {
                "CCO": [
                    {
                        "product_smiles": "CCO",
                        "reactant_smiles": ["A1"],
                        "rxn_smiles": "A1>>CCO",
                        "score": 0.9,
                        "stock_status": {"A1": True},
                    },
                    {
                        "product_smiles": "CCO",
                        "reactant_smiles": ["A2"],
                        "rxn_smiles": "A2>>CCO",
                        "score": 0.8,
                        "stock_status": {"A2": True},
                    },
                ]
            },
        )
        provider_b = NamedProvider(
            "source_b",
            {
                "CCO": [
                    {
                        "product_smiles": "CCO",
                        "reactant_smiles": ["B1"],
                        "rxn_smiles": "B1>>CCO",
                        "score": 0.7,
                        "stock_status": {"B1": True},
                    }
                ]
            },
        )
        planner = CascadeProgramSearch(
            [provider_a, provider_b],
            stock_checker=lambda smi: smi in {"A1", "A2", "B1"},
            config=CascadeSearchConfig(max_depth=1, expansion_budget=4, branch_factor=2),
        )

        results = planner.search("CCO", n_results=4)
        rxns = {step.rxn_smiles for result in results for step in result.state.step_annotations}

        self.assertIn("A1>>CCO", rxns)
        self.assertIn("B1>>CCO", rxns)
        self.assertNotIn("A2>>CCO", rxns)

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
