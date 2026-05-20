import unittest

from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit


class ProductRouteFeasibilityAuditTest(unittest.TestCase):
    def test_rejects_trivial_stock_closure_for_complex_product(self):
        run = {
            "targets": [
                {
                    "index": 0,
                    "cascade_id": "lovastatin",
                    "target_smiles": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
                    "metrics": {"strict_stock_solve_any": True},
                    "planner_output": {
                        "routes": [
                            {
                                "score": -1.0,
                                "steps": [
                                    _step("CCCO>>CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "racemization", "enzexpand"),
                                    _step("CCO>>CCCO", "racemization", "enzyformer"),
                                ],
                                "metrics": {
                                    "strict_stock_solve": True,
                                    "route_solved": True,
                                    "filled_route": True,
                                    "terminal_reactants": ["CCCO"],
                                },
                            }
                        ]
                    },
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        route = audit["targets"][0]["best_route"]
        self.assertEqual(route["route_class"], "reject_artifact")
        self.assertIn("trivial_stock_closure", route["issues"])
        self.assertIn("racemization_artifact", route["issues"])
        self.assertEqual(audit["autonomous_route_candidate_targets"], 0)

    def test_marks_natural_core_acylation_as_semisynthesis_triage(self):
        run = {
            "targets": [
                {
                    "index": 1,
                    "cascade_id": "simvastatin",
                    "target_smiles": "CCC(C)(C)C(=O)OC1CC(C)C2C(C1)C(C)C=CC2CCCC1CC(O)CC(=O)O1",
                    "metrics": {"strict_stock_solve_any": False},
                    "planner_output": {
                        "routes": [
                            {
                                "score": -2.0,
                                "steps": [
                                    _step(
                                        "CCC(C)(C)C(=O)O.CC1CC(C)C2C(C1)C(C)C=CC2CCCC1CC(O)CC(=O)O1>>CCC(C)(C)C(=O)OC1CC(C)C2C(C1)C(C)C=CC2CCCC1CC(O)CC(=O)O1",
                                        "acylation",
                                        "uspto_template",
                                    )
                                ],
                                "metrics": {
                                    "strict_stock_solve": False,
                                    "route_solved": False,
                                    "filled_route": True,
                                    "terminal_reactants": [
                                        "CCC(C)(C)C(=O)O",
                                        "CC1CC(C)C2C(C1)C(C)C=CC2CCCC1CC(O)CC(=O)O1",
                                    ],
                                },
                            }
                        ]
                    },
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        row = audit["targets"][0]
        self.assertEqual(row["target_verdict"], "semisynthesis_triage")
        self.assertEqual(row["best_route"]["route_class"], "triage_semisynthesis")
        self.assertIn("natural_core_terminal", row["best_route"]["tags"])
        self.assertIn("acylating_piece_present", row["best_route"]["tags"])

    def test_accepts_native_chem_enzy_route_contract_with_benchmark_metadata(self):
        target = "CC(C)N1C2=CC=CC=C2C(=C1/C=C/C(O)CC(O)CC(=O)O)c1ccc(F)cc1"
        methyl_ester = "COC(=O)CC(O)CC(O)/C=C/c1c(-c2ccc(F)cc2)c2ccccc2n1C(C)C"
        run = {
            "targets": [
                {
                    "target_smiles": target,
                    "solved": True,
                    "routes": [
                        {
                            "target_smiles": target,
                            "solved": True,
                            "score": 0.8,
                            "steps": [
                                {
                                    "product_smiles": target,
                                    "reactant_smiles": [methyl_ester],
                                    "rxn_smiles": f"{methyl_ester}>>{target}",
                                    "source_model": "([O;D1;H0:3]=[C:2]-[OH;D1;+0:1])>>C-[O;H0;D2;+0:1]-[C:2]=[O;D1;H0:3]",
                                    "score": 0.8,
                                    "stock_status": {methyl_ester: True},
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        audit = build_product_route_feasibility_audit(
            run,
            benchmark_rows=[{"target_smiles": target, "cascade_id": "fluvastatin"}],
        )

        row = audit["targets"][0]
        self.assertEqual(row["target_id"], "fluvastatin")
        self.assertEqual(row["best_route"]["route_class"], "triage_late_stage")
        self.assertIn("late_stage_derivatization", row["best_route"]["tags"])

    def test_native_route_with_large_unexplained_gain_is_rejected(self):
        target = "CC[C@@H](O)C[C@@H](O)CC=O"
        pyruvate = "CC(=O)C(=O)O"
        run = {
            "targets": [
                {
                    "target_smiles": target,
                    "solved": True,
                    "routes": [
                        {
                            "target_smiles": target,
                            "solved": True,
                            "score": 0.62,
                            "steps": [
                                {
                                    "product_smiles": target,
                                    "reactant_smiles": [pyruvate],
                                    "rxn_smiles": f"{pyruvate}>>{target}",
                                    "source_model": "ChemEnzyRetroPlanner",
                                    "score": 0.62,
                                    "stock_status": {pyruvate: True},
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        route = audit["targets"][0]["best_route"]
        self.assertEqual(route["route_class"], "reject_artifact")
        self.assertFalse(route["route_plausibility"]["passed"])
        self.assertIn("large_unexplained_carbon_gain", route["issues"])
        self.assertIn("large_unexplained_heavy_atom_gain", route["issues"])

    def test_benzyl_halide_is_not_misclassified_as_aryl_coupling(self):
        target = "CCCC(=O)OCc1ccccc1"
        run = {
            "targets": [
                {
                    "target_smiles": target,
                    "solved": True,
                    "routes": [
                        {
                            "target_smiles": target,
                            "solved": True,
                            "score": 0.8,
                            "steps": [
                                {
                                    "product_smiles": target,
                                    "reactant_smiles": ["BrCc1ccccc1", "CCCC(=O)O"],
                                    "rxn_smiles": f"BrCc1ccccc1.CCCC(=O)O>>{target}",
                                    "source_model": "ChemEnzyRetroPlanner",
                                    "score": 0.8,
                                    "stock_status": {"BrCc1ccccc1": True, "CCCC(=O)O": True},
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        route = audit["targets"][0]["best_route"]
        self.assertNotIn("aryl_coupling_hint", route["tags"])
        self.assertEqual(route["route_class"], "needs_chemist_review")

    def test_acid_alcohol_ester_route_is_late_stage_derivatization(self):
        target = "CCCC(=O)OCc1ccccc1"
        run = {
            "targets": [
                {
                    "target_smiles": target,
                    "solved": True,
                    "routes": [
                        {
                            "target_smiles": target,
                            "solved": True,
                            "score": 0.8,
                            "steps": [
                                {
                                    "product_smiles": target,
                                    "reactant_smiles": ["CCCC(=O)O", "OCc1ccccc1"],
                                    "rxn_smiles": f"CCCC(=O)O.OCc1ccccc1>>{target}",
                                    "source_model": "ChemEnzyRetroPlanner",
                                    "score": 0.8,
                                    "stock_status": {"CCCC(=O)O": True, "OCc1ccccc1": True},
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        route = audit["targets"][0]["best_route"]
        self.assertIn("late_stage_derivatization", route["tags"])
        self.assertEqual(route["reaction_profile"]["classes"], ["esterification"])

    def test_wittig_phosphorane_terminal_is_carrier_not_product_like(self):
        target = "CCOC(=O)/C=C/c1ccccc1"
        phosphorane = "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1"
        aldehyde = "O=Cc1ccccc1"
        run = {
            "targets": [
                {
                    "target_smiles": target,
                    "solved": True,
                    "routes": [
                        {
                            "target_smiles": target,
                            "solved": True,
                            "score": 0.8,
                            "steps": [
                                {
                                    "product_smiles": target,
                                    "reactant_smiles": [phosphorane, aldehyde],
                                    "rxn_smiles": f"{phosphorane}.{aldehyde}>>{target}",
                                    "source_model": "Template proposal",
                                    "score": 0.8,
                                    "stock_status": {phosphorane: True, aldehyde: True},
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        route = audit["targets"][0]["best_route"]
        terminal_profile = route["terminal_profile"]
        self.assertIn("carrier_reagent_terminal", route["tags"])
        self.assertNotIn("advanced_or_product_like_terminal", route["tags"])
        self.assertEqual(terminal_profile["max_terminal_heavy_atoms"], 25)
        self.assertEqual(terminal_profile["effective_max_terminal_heavy_atoms"], 8)
        self.assertEqual(terminal_profile["carrier_reagents"][0]["role"], "wittig_phosphorane")

    def test_condition_audit_flags_extreme_enzyme_temperature_without_rejecting_route(self):
        target = "CCO"
        run = {
            "targets": [
                {
                    "target_smiles": target,
                    "solved": True,
                    "planner_output": {
                        "routes": [
                            {
                                "target_smiles": target,
                                "solved": True,
                                "score": 0.8,
                                "steps": [
                                    {
                                        "product": target,
                                        "main_reactant": "CC=O",
                                        "aux_reactants": [],
                                        "reaction_smiles": f"CC=O>>{target}",
                                        "reaction_type": "enzymatic",
                                        "source": "enzexpand",
                                        "enzyme_ec_annotations": [{"ec_number": "1.1.1.1", "confidence": 0.8}],
                                        "condition_predictions": [
                                            {"Temperature": -60.0, "Solvent": "Cc1ccccc1", "Score": "0.5"}
                                        ],
                                        "reaction_interpretation": {
                                            "reaction_class": "reduction",
                                            "atom_change": {"heavy_atom_delta": 0},
                                        },
                                    }
                                ],
                                "metrics": {
                                    "strict_stock_solve": True,
                                    "route_solved": True,
                                    "filled_route": True,
                                    "terminal_reactants": ["CC=O"],
                                },
                            }
                        ]
                    },
                }
            ]
        }

        audit = build_product_route_feasibility_audit(run)

        route = audit["targets"][0]["best_route"]
        self.assertEqual(route["condition_audit"]["route_risk"], "high")
        self.assertIn("condition_high_risk", route["issues"])
        self.assertEqual(route["condition_audit"]["steps"][0]["risk"], "high")
        self.assertIn("enzyme_temperature_out_of_window", route["condition_audit"]["steps"][0]["issues"])


def _step(rxn: str, reaction_class: str, source: str) -> dict:
    return {
        "reaction_smiles": rxn,
        "source": source,
        "reaction_interpretation": {
            "reaction_class": reaction_class,
            "atom_change": {"heavy_atom_delta": 0},
        },
    }


if __name__ == "__main__":
    unittest.main()
