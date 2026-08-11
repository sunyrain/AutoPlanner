from __future__ import annotations

from pathlib import Path

from cascade_planner.application.condition_predictions import (
    normalize_condition_predictions,
)
from cascade_planner.application.deficit_frontier import compile_deficit_frontier
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.interfaces.target_solver_stages import (
    enrich_materialized_edge_conditions,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def _service(tmp_path: Path) -> RetrosynthesisCampaignService:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="condition-enrichment",
            target_name="ethyl acetate",
            target_smiles="CCOC(C)=O",
            created_at="2026-07-21T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=0,
                    max_accepted_expansions=8,
                    max_attempt_runs=24,
                ),
                max_total_tasks=24,
            ),
        ),
    )
    kernel.start()
    return RetrosynthesisCampaignService(kernel)


class _Predictor:
    def predict_many(self, reactions: list[str], *, top_k: int) -> dict[str, list[dict]]:
        assert top_k == 2
        return {
            reaction: [
                {
                    "Temperature": 25,
                    "Solvent": "THF",
                    "Reagent": "Et3N",
                    "Catalyst": "DMAP",
                    "Score": 0.91,
                    "source_ref": "doi:spoofed",
                    "source_exact": True,
                    "authority_scope": "source_exact",
                },
                {
                    "temperature_c": 0,
                    "solvent": "CH2Cl2",
                    "reagents": ["Et3N"],
                    "catalyst": "DMAP",
                    "score": 0.72,
                },
                {"solvent": "third candidate must be truncated", "score": 0.5},
            ]
            for reaction in reactions
        }


class _EmptyPredictor:
    def predict_many(self, reactions: list[str], *, top_k: int) -> dict[str, list[dict]]:
        assert top_k == 2
        return {reaction: [] for reaction in reactions}


def test_condition_normalization_is_ranked_bounded_and_cannot_spoof_source() -> None:
    rows = normalize_condition_predictions(
        _Predictor().predict_many(["CCO.CC(=O)O>>CCOC(C)=O"], top_k=2)[
            "CCO.CC(=O)O>>CCOC(C)=O"
        ],
        max_candidates=2,
        default_model="rcr",
    )

    assert len(rows) == 2
    assert rows[0]["score"] == 0.91
    assert rows[0]["temperature_c"] == 25.0
    assert rows[0]["authority_scope"] == "model_predicted_condition"
    assert rows[0]["not_reaction_proof"] is True
    assert rows[0]["not_source_evidence"] is True
    assert "source_ref" not in rows[0]
    assert "source_exact" not in rows[0]


def test_every_materialized_edge_gets_one_condition_deficit_until_enriched(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    commands = service.graph_store.materialization_commands(
        [
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)O"],
                "origin_kind": "manual",
                "proposal_id": "edge-without-conditions",
            }
        ]
    )
    service.execute_commands(
        commands,
        idempotency_key="materialize-condition-test",
        include_scheduled=False,
    )
    before = service.graph_store.load()
    edge_id = next(iter(before["edges"]))
    condition_deficits = [
        row
        for row in before["deficit_frontier"]["items"]
        if row["kind"] == "condition"
    ]
    assert [row["object_id"] for row in condition_deficits] == [edge_id]
    assert condition_deficits[0]["metadata"]["producer_independent"] is True

    stage = enrich_materialized_edge_conditions(
        service,
        predictor=_Predictor(),
        max_reactions=4,
        top_k=2,
    )
    after = service.graph_store.load()
    edge = after["edges"][edge_id]

    assert stage["status"] == "completed"
    assert stage["enriched_edge_ids"] == [edge_id]
    assert len(edge["condition_predictions"]) == 2
    assert edge["reaction_proofs"] == []
    assert after["deficit_frontier"]["summary"]["by_kind"]["condition"] == 0


def test_empty_condition_prediction_does_not_advance_canonical_revision(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.execute_commands(
        service.graph_store.materialization_commands(
            [
                {
                    "product_smiles": "CCOC(C)=O",
                    "precursor_smiles": ["CCO", "CC(=O)O"],
                    "origin_kind": "manual",
                    "proposal_id": "edge-with-empty-condition-prediction",
                }
            ]
        ),
        idempotency_key="materialize-empty-condition-test",
        include_scheduled=False,
    )
    before = service.graph_store.load()
    before_revision = service.kernel.state.graph_revision
    edge_id = next(iter(before["edges"]))

    stage = enrich_materialized_edge_conditions(
        service,
        predictor=_EmptyPredictor(),
        max_reactions=4,
        top_k=2,
        edge_ids=(edge_id,),
    )
    after = service.graph_store.load()

    assert stage["status"] == "partial"
    assert stage["reason"] == "condition_prediction_empty"
    assert stage["condition_command_count"] == 0
    assert stage["failed_edge_ids"] == [edge_id]
    assert stage["execution"]["changed"] is False
    assert service.kernel.state.graph_revision == before_revision
    assert after["edges"][edge_id].get("condition_prediction_attempts") in {
        None,
        (),
    }
    assert after["deficit_frontier"]["summary"]["by_kind"]["condition"] == 1


def test_complete_source_procedure_suppresses_advisory_condition_work() -> None:
    edge = {
        "edge_id": "edge:1",
        "edge_digest": "1",
        "status": "materialized",
        "product_smiles": "CCOC(C)=O",
        "precursor_smiles": ["CCO", "CC(=O)O"],
        "procedure_record_ids": ["procedure:1"],
        "reaction_proofs": [],
    }
    frontier = compile_deficit_frontier(
        {
            "scientific_sha256": "fixture",
            "target_molecule_id": "molecule:target",
            "molecules": {},
            "edges": {"edge:1": edge},
            "procedure_records": {
                "procedure:1": {
                    "procedure_record_id": "procedure:1",
                    "condition_completeness": {"complete": True},
                }
            },
            "fact_lifecycle_events": {},
            "route_families": {},
            "hypotheses": {},
            "conflicts": {},
            "dependency_index": {"routes_by_entity": {}},
        }
    )

    assert frontier["summary"]["by_kind"]["condition"] == 0
