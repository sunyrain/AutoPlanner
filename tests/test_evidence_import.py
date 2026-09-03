from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.evidence_import import import_structured_evidence
from cascade_planner.interfaces.target_solver_stages import (
    validate_materialized_edges,
)
from cascade_planner.runtime.paths import RuntimePaths


TARGET = "CCOC(C)=O"
REACTANTS = ["CCO", "CC(=O)Cl"]


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index" / "runs.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:ester",
                "strategic_disconnection": "late acyl substitution",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:ester",
                "route_family_id": "family:ester",
                "steps": [
                    {
                        "step_id": "step:ester",
                        "product_smiles": TARGET,
                        "precursor_smiles": REACTANTS,
                        "transformation_hypothesis": "acyl substitution",
                    }
                ],
            }
        ],
    }


def _mapper(reactions: list[str]) -> list[str]:
    assert reactions == ["CC(=O)Cl.CCO>>CCOC(C)=O"]
    return [
        "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][CH2:6][OH:7]>>"
        "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
    ]


def _source(source_ref: str, location: str) -> dict:
    return {
        "binding": {
            "source_kind": "patent" if source_ref.startswith("patent:") else "paper_si",
            "source_ref": source_ref,
            "title": f"Exact ester source {location}",
            "provenance": "human_verified_structured_import",
        },
        "extraction": {
            "schema_version": "structured_exact_row_extraction.v1",
            "extractor": {
                "producer_kind": "manual_structured_extraction",
                "producer_id": "tests.independent-curator",
                "version": "1.0.0",
            },
            "rows": [
                {
                    "product_smiles": TARGET,
                    "reactant_smiles": REACTANTS,
                    "location_ref": location,
                    "conditions": {"temperature_c": 20},
                }
            ],
        },
    }


def test_import_structured_evidence_binds_two_sources_and_revalidates(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    created = gateway.create_run(
        run_id="evidence-import",
        target_name="ethyl acetate",
        target_smiles=TARGET,
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=1,
            minimum_edge_proof_level=3,
            minimum_independent_source_groups=2,
            stock_boundary="benchmark_search",
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=0,
            max_accepted_expansions=4,
            max_attempt_runs=16,
        ),
        global_plan=_plan(),
        materialize=True,
    )
    import_path = tmp_path / "evidence.json"
    import_path.write_text(
        json.dumps(
            {
                "schema_version": "structured_evidence_import.v1",
                "sources": [
                    _source("patent:US2020123456A1", "Example 1"),
                    _source("doi:10.1000/independent", "Table 2, entry 4"),
                ],
            }
        ),
        encoding="utf-8",
    )
    service = gateway._open("evidence-import", run_dir=created["run_dir"])
    initial_validation = validate_materialized_edges(
        service,
        atom_mapper=_mapper,
    )
    assert initial_validation["validation_command_count"] == 1

    result = import_structured_evidence(
        gateway,
        run_id="evidence-import",
        run_dir=created["run_dir"],
        import_path=import_path,
        atom_mapper=_mapper,
    )

    graph = gateway._open(
        "evidence-import",
        run_dir=created["run_dir"],
    ).graph_store.load()
    edge = next(iter(graph["edges"].values()))
    assert result["model_invocations"] == 0
    assert result["exact_record_count"] == 2
    assert result["source_binding_count"] == 2
    assert result["evidence_bound_revalidation_edge_count"] == 1
    assert result["validation"]["forced_revalidation_edge_count"] == 1
    assert result["validation"]["validation_command_count"] == 1
    assert len(edge["exact_record_ids"]) == 2
    assert len(edge["independent_source_groups"]) == 2
    assert any(proof.get("accepted") is True for proof in edge["reaction_proofs"])
    stitched = next(iter(result["portfolio"]["edge_proofs"].values()))
    assert stitched["achieved_level"] == 3
