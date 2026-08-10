from __future__ import annotations

from dataclasses import replace

import pytest

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import (
    CampaignResourceBudget,
    StockOracleReference,
    TargetConstraints,
    UnifiedCampaignSpec,
    stock_oracle_reference_from_builder,
)


def _stock_oracle() -> StockOracleReference:
    return StockOracleReference.compatibility_unbound(boundary="procurement")


def test_unified_campaign_spec_round_trip_excludes_display_and_dataset_controls() -> None:
    spec = UnifiedCampaignSpec(
        target_smiles="CCO",
        stock_oracle=_stock_oracle(),
        constraints=TargetConstraints(
            forbidden_reagents=("benzene", "benzene"),
            max_route_steps=8,
            allowed_execution_domains=("chemical", "biocatalytic"),
            safety_limits={"max_temperature_c": 120.0},
            stock_source_ids=("internal-v2",),
        ),
        resource_budget=CampaignResourceBudget(
            model=RetrosynthesisRunBudget(max_model_invocations=2),
            max_total_tasks=40,
            max_program_tasks=7,
            max_experiment_tasks=3,
        ),
    )

    row = spec.to_dict()
    restored = UnifiedCampaignSpec.from_dict(row)

    assert restored.to_dict() == row
    assert set(row) == {
        "schema_version",
        "target",
        "stock_oracle",
        "constraints",
        "resource_budget",
        "semantics",
        "content_sha256",
    }
    serialized = str(row).casefold()
    assert "target_name" not in serialized
    assert "objective_mode" not in serialized
    assert row["semantics"]["acceptance_is_a_quality_projection"] is True
    assert row["constraints"]["forbidden_reagents"] == ["benzene"]
    assert row["resource_budget"]["max_program_tasks"] == 7
    assert row["resource_budget"]["max_experiment_tasks"] == 3


def test_target_constraints_reject_unknown_dataset_controls() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        TargetConstraints.from_dict(
            {
                "schema_version": "target_constraints.v1",
                "dataset_name": "RetroStar-190",
            }
        )
    with pytest.raises(ValueError, match="dataset controls"):
        TargetConstraints(safety_limits={"benchmark_group": "test"})


def test_stock_oracle_reference_fails_closed_on_nested_binding_tamper() -> None:
    oracle = _stock_oracle()
    row = oracle.to_dict()
    row["binding"]["positive_authority"] = True

    with pytest.raises(ValueError, match="reference digest"):
        StockOracleReference.from_dict(row)

    binding = dict(oracle.binding)
    binding["positive_authority"] = True
    with pytest.raises(ValueError, match="binding digest"):
        replace(oracle, binding=binding)


def test_frozen_index_stock_oracle_binds_exact_index_and_source_hashes() -> None:
    class FrozenIndex:
        index_sha256 = "1" * 64
        source_sha256 = "2" * 64
        catalog_name = "frozen-stock"
        member_count = 123

    oracle = stock_oracle_reference_from_builder(
        FrozenIndex(),
        boundary="benchmark_search",
    )

    assert oracle.oracle_id == "frozen-index:" + "1" * 24
    assert oracle.binding["kind"] == "frozen_benchmark_index"
    assert oracle.binding["index_sha256"] == "1" * 64
    assert oracle.binding["source_sha256"] == "2" * 64


def test_callable_stock_resolver_binds_code_contract_and_requires_snapshots() -> None:
    def resolver(_smiles, **_kwargs):
        return {}

    first = stock_oracle_reference_from_builder(
        resolver,
        boundary="procurement",
    )
    second = stock_oracle_reference_from_builder(
        resolver,
        boundary="procurement",
    )

    assert first.to_dict() == second.to_dict()
    assert first.binding["kind"] == "snapshot_resolver_contract"
    assert first.binding["outputs_require_content_addressing"] is True
