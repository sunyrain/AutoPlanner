from __future__ import annotations

from cascade_planner.web.workbench_pdf import compile_workbench_report_html


def _snapshot() -> dict:
    return {
        "run_id": "pdf-designed-example",
        "revision": 7,
        "target": {
            "name": "Designed target",
            "canonical_smiles": "CCO",
        },
        "portfolio": {
            "accepted": False,
            "achieved_profile": "reaction_validated",
            "acceptance_profile_counts": {
                "condition_complete": 0,
                "process_ready": 0,
            },
            "closeout": {
                "complete_route_count": 0,
                "deficit_count": 2,
                "reasons": ["minimum_complete_route_count_not_met"],
            },
        },
        "molecules": {
            "m:target": {"canonical_smiles": "CCO", "label": "ethanol"},
            "m:mid": {"canonical_smiles": "CC=O", "label": "acetaldehyde"},
            "m:start": {"canonical_smiles": "CC(=O)O", "label": "acetic acid"},
        },
        "edges": {
            "e:target": {
                "edge_id": "e:target",
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:mid"],
                "proof_name": "L2_reaction_validated",
                "accepted": True,
                "condition_status": "model_predicted",
                "proof_vector": {"sources": "none", "process": "blocked"},
            },
            "e:mid": {
                "edge_id": "e:mid",
                "product_molecule_id": "m:mid",
                "precursor_molecule_ids": ["m:start"],
                "proof_name": "L1_structural_materialized",
                "accepted": False,
                "condition_status": "missing",
                "proof_vector": {"sources": "none", "process": "blocked"},
            },
        },
        "inspectors": {
            "edges": {
                "e:target": {
                    "condition_status": "model_predicted",
                    "condition_predictions": [
                        {
                            "Catalyst": "Pd/C",
                            "Solvent": "EtOH",
                            "Temperature": 25,
                            "Score": 0.72,
                        }
                    ],
                    "proof": {
                        "achieved_level_name": "L2_reaction_validated",
                        "reasons": [],
                    },
                    "reaction_proofs": [
                        {
                            "accepted": True,
                            "validator_version": "fixture.v1",
                            "mapped_reaction": "CC=O>>CCO",
                            "checks": {"mapped_reaction_present": True},
                        }
                    ],
                    "provenance": [
                        {
                            "origin_kind": "fixture",
                            "origin_ref": "doi:10.1000/example",
                            "transformation_hypothesis": "Hydrogenate the aldehyde.",
                        }
                    ],
                },
                "e:mid": {
                    "condition_status": "missing",
                    "proof": {
                        "achieved_level_name": "L1_structural_materialized",
                        "reasons": ["reaction_validation_missing"],
                    },
                },
            }
        },
        "routes": {
            "route:one": {
                "route_id": "route:one",
                "strategy": "Two-step example route",
                # Target-first input deliberately verifies forward topological order.
                "edge_ids": ["e:target", "e:mid"],
                "proof_name": "L1_structural_materialized",
                "deficit_count": 2,
            }
        },
        "planned_routes": {
            "planned:one": {
                "strategy": "Advisory one-step route",
                "steps": [
                    {
                        "step_id": "planned-step",
                        "precursor_smiles": ["C=C"],
                        "product_smiles": "CCO",
                        "transformation_hypothesis": "Advisory hydration.",
                    }
                ],
            }
        },
    }


def test_designed_pdf_report_contains_cover_all_routes_snake_and_step_details() -> None:
    body = compile_workbench_report_html(_snapshot())

    assert "逆合成路线全景与逐步证据报告" in body
    assert "Designed target" in body
    assert "route:one" in body
    assert "planned:one" in body
    assert body.count('class="route-overview report-page"') == 2
    assert 'class="snake-diagram"' in body
    assert "路线逐步详单" in body
    assert "Hydrogenate the aldehyde." in body
    assert "Pd/C" in body
    assert "模型条件候选（非文献事实）" in body
    assert "条件待取证" in body
    assert "CC=O&gt;&gt;CCO" in body
    assert body.index("e:mid") < body.index("e:target")
    assert "<svg" in body


def test_designed_pdf_report_escapes_project_text() -> None:
    snapshot = _snapshot()
    snapshot["target"]["name"] = "<script>alert(1)</script>"

    body = compile_workbench_report_html(snapshot)

    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


def test_pdf_audit_records_wrap_long_keys_and_values_without_overlapping_columns() -> None:
    snapshot = _snapshot()
    snapshot["inspectors"]["edges"]["e:target"]["exact_records"] = [
        {
            "extraction_artifact_structure_source": "sha256:" + "a" * 128,
            "conditions": {
                "addition_order": "cyclohexanone was added gradually " * 24,
                "workup": "water, extraction, drying, filtration, concentration",
            },
        }
    ]

    body = compile_workbench_report_html(snapshot)

    assert "extraction_<wbr>artifact_<wbr>structure_<wbr>source" in body
    assert "grid-template-columns:42mm minmax(0,1fr)" in body
    assert ".record-card dt,.record-card dd { min-width:0" in body
    assert ".record-card { break-inside:auto; page-break-inside:auto" in body
    assert ".provenance-record > * { min-width:0" in body
    assert ".validation-record > div:first-child > * { min-width:0" in body
