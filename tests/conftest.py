from __future__ import annotations

import pytest


@pytest.fixture
def reported_ethanol_program_pack() -> dict:
    """One digest-bound, low-evidence literature route for adapter tests."""

    from cascade_planner.application.candidate_programs import (
        candidate_route_observation_from_workbench,
        project_candidate_route_to_programs,
    )
    from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256

    workbench = {
        "schema_version": "retrosynthesis_route_workbench.v1",
        "run_id": "reported-ethanol-fixture",
        "revision": {"graph": 1, "evidence": 1},
        "target": {
            "molecule_id": "m:target",
            "canonical_smiles": "CCO",
            "name": "ethanol",
        },
        "molecules": {
            "m:target": {
                "molecule_id": "m:target",
                "canonical_smiles": "CCO",
                "label": "ethanol",
                "role": "target",
                "stock_closed": False,
            },
            "m:leaf": {
                "molecule_id": "m:leaf",
                "canonical_smiles": "CC=O",
                "label": "acetaldehyde",
                "role": "stock_leaf",
                "stock_closed": False,
            },
        },
        "edges": {
            "edge:reported-reduction": {
                "edge_id": "edge:reported-reduction",
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:leaf"],
                "accepted": False,
                "proof_level": 1,
                "proof_vector": {
                    "sources": "single_group",
                    "reaction": "unvalidated",
                    "conditions": "source_recorded_unverified",
                },
            }
        },
        "routes": {
            "route:reported-ethanol": {
                "route_id": "route:reported-ethanol",
                "edge_ids": ["edge:reported-reduction"],
                "root_edge_ids": ["edge:reported-reduction"],
                "leaf_molecule_ids": ["m:leaf"],
                "complete": True,
                "closure_profile": "exploration_closed",
                "reported_source_refs": ["doi:10.1000/reported-ethanol-fixture"],
                "warning_codes": ["reported_reaction_unvalidated"],
            }
        },
        "inspectors": {
            "edges": {
                "edge:reported-reduction": {
                    "condition_status": "source_recorded_unverified",
                    "rejection_reasons": ["reaction_validation_missing"],
                    "provenance": [{"origin_kind": "literature_html"}],
                    "sources": [
                        {"source_ref": "doi:10.1000/reported-ethanol-fixture"}
                    ],
                    "source_observation_records": [],
                }
            }
        },
    }
    workbench["content_sha256"] = strict_canonical_json_sha256(workbench)
    observation = candidate_route_observation_from_workbench(workbench)
    projection = project_candidate_route_to_programs(observation)
    return {
        "schema_version": "reported_program_route_pack.v1",
        "observation": observation,
        "projection": projection,
        "route_ids": [],
    }
