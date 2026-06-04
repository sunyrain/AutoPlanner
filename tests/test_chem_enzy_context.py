import unittest

from cascade_planner.baselines.chem_enzy_context import (
    build_chem_enzy_context_source,
    chem_enzy_smiles_tokenize,
)
from cascade_planner.cascade_search.state import (
    CascadeProgramState,
    ConditionEnvelope,
    StepAnnotation,
)


class ChemEnzyContextSourceTest(unittest.TestCase):
    def test_smiles_tokenizer_matches_chem_enzy_tokens(self):
        self.assertEqual(chem_enzy_smiles_tokenize("O[C@@H](Cl)Br"), "O [C@@H] ( Cl ) Br")

    def test_initial_state_context_source(self):
        state = CascadeProgramState.initial("CCO")

        source = build_chem_enzy_context_source(target_smiles="CCO", product_smiles="CCO", state=state)

        self.assertEqual(
            source,
            "<step_1> <stage_stage_1> <temp_unknown> <ph_unknown> <solv_unknown> "
            "<ec_unknown> <target> C C O <product> C C O",
        )

    def test_uses_stage_condition_and_ec_context(self):
        state = CascadeProgramState.initial("CCO")
        state.append_step(
            StepAnnotation(
                product_smiles="CCO",
                reactant_smiles=["CC"],
                rxn_smiles="CC>>CCO",
                stage_id="stage_2",
                ec_numbers=["1.1.1.1"],
                condition=ConditionEnvelope.from_point(
                    temperature_c=30,
                    ph=7.2,
                    solvent="water",
                ),
            ),
            opened_leaves=["CC"],
        )

        source = build_chem_enzy_context_source(target_smiles="CCO", product_smiles="CC", state=state)

        self.assertIn("<step_2>", source)
        self.assertIn("<stage_stage_2>", source)
        self.assertIn("<temp_ambient>", source)
        self.assertIn("<ph_neutral>", source)
        self.assertIn("<solv_water>", source)
        self.assertIn("<ec_1>", source)
        self.assertTrue(source.endswith("<product> C C"))


if __name__ == "__main__":
    unittest.main()
