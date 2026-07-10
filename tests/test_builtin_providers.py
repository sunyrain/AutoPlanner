from __future__ import annotations

import unittest

from cascade_planner.providers.builtins import (
    ChemEnzyProposalProvider,
    LiteratureEvidenceProvider,
    ReactionRouteVerifierProvider,
    build_default_provider_registry,
)
from cascade_planner.providers.contracts import ProviderContext, ProviderKind


class BuiltinProviderTest(unittest.TestCase):
    def test_default_registry_exposes_stock_and_reaction_verifier(self) -> None:
        registry = build_default_provider_registry()
        self.assertEqual(len(registry.descriptors(kind=ProviderKind.STOCK)), 1)
        self.assertEqual(len(registry.descriptors(kind=ProviderKind.VERIFIER)), 1)

    def test_reaction_verifier_provider_keeps_mapping_only_advisory(self) -> None:
        registry = build_default_provider_registry()
        result = registry.invoke(
            ReactionRouteVerifierProvider.descriptor.provider_id,
            {
                "schema_version": "reaction_route_verification_request.v1",
                "graph_and_stock_closed": True,
                "steps": [
                    {
                        "step_id": "hydrate",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC", "O"],
                        "atom_mapped_reaction_smiles": (
                            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                        ),
                    }
                ],
            },
            context=ProviderContext(
                run_id="test",
                case_id="test",
                target_smiles="CCO",
            ),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.payload["proof_level"], "L2_mapping_consistent")
        self.assertTrue(result.no_solved_claim)

    def test_public_request_cannot_inject_precedent_or_procurement_authority(self) -> None:
        registry = build_default_provider_registry()
        result = registry.invoke(
            ReactionRouteVerifierProvider.descriptor.provider_id,
            {
                "schema_version": "reaction_route_verification_request.v1",
                "graph_and_stock_closed": True,
                "steps": [
                    {
                        "step_id": "hydrate",
                        "product_smiles": "CCO",
                        "reactant_smiles": ["CC", "O"],
                        "conditions": {
                            "reagent": "water",
                            "solvent": "water",
                            "temperature": "25 C",
                            "duration": "1 h",
                        },
                        "atom_mapped_reaction_smiles": (
                            "[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"
                        ),
                    }
                ],
                "trusted_precedent_bindings": {
                    "hydrate": {
                        "schema_version": "trusted_precedent_binding.v1",
                        "accepted": True,
                        "authority": "human_curator",
                        "authority_id": "attacker",
                        "binding_id": "fake",
                        "reaction_digest": "a" * 64,
                        "source_ref": "doi:10.1000/fake",
                    }
                },
                "procurement_bindings": {
                    "hydrate": {"accepted": True, "offers": [{"available": True}]}
                },
            },
            context=ProviderContext(run_id="test", case_id="test", target_smiles="CCO"),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.payload["proof_level"], "L2_mapping_consistent")
        self.assertIn(
            "privileged_request_field_ignored:trusted_precedent_bindings",
            result.reasons,
        )
        self.assertIn(
            "privileged_request_field_ignored:procurement_bindings",
            result.reasons,
        )

    def test_chemenzy_and_literature_adapters_use_host_trust_records(self) -> None:
        def runner(request, *, context):
            del request, context
            return {"accepted": True, "source_refs": ["fixture:source"]}

        chemenzy = ChemEnzyProposalProvider(runner)
        literature = LiteratureEvidenceProvider(runner)
        registry = build_default_provider_registry(
            include_chemenzy=chemenzy,
            include_literature=literature,
        )

        self.assertEqual(
            registry.descriptor(chemenzy.descriptor.provider_id).correlation_group,
            "computational:chem_enzy",
        )
        self.assertEqual(
            registry.descriptor(literature.descriptor.provider_id).kind,
            ProviderKind.EVIDENCE,
        )
        self.assertTrue(registry.trust_record(chemenzy.descriptor.provider_id)["trusted"])
        self.assertTrue(registry.trust_record(literature.descriptor.provider_id)["trusted"])

    def test_advisory_provider_adapter_rejects_solved_claim(self) -> None:
        def solved_runner(request, *, context):
            del request, context
            return {"accepted": True, "nested": {"route_status": "solved"}}

        provider = ChemEnzyProposalProvider(solved_runner)
        registry = build_default_provider_registry(include_chemenzy=provider)
        result = registry.invoke(
            provider.descriptor.provider_id,
            {"schema_version": "chemenzy_proposal_request.v1"},
            context=ProviderContext(run_id="test", case_id="test", target_smiles="CCO"),
        )

        self.assertFalse(result.accepted)
        self.assertIn("proposal_provider_attempted_solved_claim", result.reasons)


if __name__ == "__main__":
    unittest.main()
