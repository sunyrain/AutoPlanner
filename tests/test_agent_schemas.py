import unittest

from cascade_planner.agent.schemas import (
    ConditionRisk,
    EnzymePrior,
    ReactionTypePrior,
    RouteModePrior,
    StrategicPrior,
)


class AgentSchemaTest(unittest.TestCase):
    def test_strategic_prior_normalizes_unsupported_llm_claims(self):
        prior = StrategicPrior(
            target_smiles="CCO",
            route_mode_priors=[RouteModePrior("magic_mode", 2.0)],
            reaction_type_priors=[ReactionTypePrior(None, "teleportation", 1.5)],
            enzyme_priors=[EnzymePrior(ec1=9, weight=2.0)],
            condition_risks=[ConditionRisk("unknown", "extreme")],
            source="llm",
        ).normalize()

        self.assertEqual(prior.route_mode_priors[0].mode, "unknown")
        self.assertEqual(prior.route_mode_priors[0].weight, 1.0)
        self.assertEqual(prior.reaction_type_priors[0].reaction_type, "other")
        self.assertEqual(prior.reaction_type_priors[0].weight, 1.0)
        self.assertIsNone(prior.enzyme_priors[0].ec1)
        self.assertEqual(prior.enzyme_priors[0].weight, 1.0)
        self.assertEqual(prior.condition_risks[0].severity, "medium")
        self.assertIn("unsupported_reaction_type_prior:teleportation", prior.unsupported_claims)
        self.assertIn("unsupported_ec1_prior:9", prior.unsupported_claims)


if __name__ == "__main__":
    unittest.main()
