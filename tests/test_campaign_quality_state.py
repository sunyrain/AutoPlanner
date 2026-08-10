from __future__ import annotations

from cascade_planner.application.campaign_quality_state import (
    CAMPAIGN_QUALITY_AXES,
    compile_campaign_quality_state,
)


def test_quality_state_always_emits_every_independent_axis() -> None:
    result = compile_campaign_quality_state(workbench={})

    assert tuple(result["axes"]) == CAMPAIGN_QUALITY_AXES
    assert result["axes"]["topology"]["state"] == "open"
    assert result["axes"]["conditions"]["state"] == "not_assessed"
    assert result["axes"]["program_validation"]["state"] == "not_assessed"
    assert result["configured_acceptance"] is False
    assert result["semantics"]["axes_are_independent"] is True
    assert len(result["content_sha256"]) == 64


def test_quality_state_keeps_open_axes_after_configured_acceptance() -> None:
    workbench = {
        "portfolio": {"accepted": True, "stock_boundary": "in_house"},
        "routes": {
            "route:1": {
                "edge_ids": ["edge:1"],
                "route_family_id": "family:1",
                "reaction_validated": True,
                "literature_grounded": True,
                "configured_boundary_closed": True,
                "condition_complete": True,
                "procurement_closed": False,
            },
            "route:2": {
                "edge_ids": ["edge:2"],
                "route_family_id": "family:2",
                "reaction_validated": True,
                "literature_grounded": False,
                "configured_boundary_closed": True,
                "condition_complete": False,
                "procurement_closed": False,
            },
        },
    }
    gates = {
        "minimum_routes": 2,
        "gates": {
            "B1_global_multi_route": True,
            "B2_host_validated_routes": True,
            "B3_exact_multi_source": False,
            "B4_stock_boundary": True,
            "B5_configured_portfolio_acceptance": True,
        },
        "counts": {
            "target_rooted_distinct_skeletons": 2,
            "reaction_validated_skeletons": 2,
            "evidence_closed_skeletons": 1,
            "stock_closed_skeletons": 2,
        },
    }

    result = compile_campaign_quality_state(workbench=workbench, gates=gates)

    assert result["configured_acceptance"] is True
    assert result["axes"]["topology"]["state"] == "satisfied"
    assert result["axes"]["reaction_validation"]["state"] == "satisfied"
    assert result["axes"]["exact_evidence"]["state"] == "open"
    assert result["axes"]["stock"]["state"] == "satisfied"
    assert result["axes"]["conditions"]["state"] == "open"
    assert result["axes"]["procurement"]["state"] == "open"
    assert result["axes"]["diversity"]["state"] == "satisfied"
    assert result["axes"]["stock"]["metadata"]["boundary"] == "in_house"


def test_program_validation_is_optional_but_auditable_when_supplied() -> None:
    result = compile_campaign_quality_state(
        program_validation={
            "validated_count": 1,
            "required_count": 1,
            "accepted": True,
        }
    )

    program = result["axes"]["program_validation"]
    assert program["state"] == "satisfied"
    assert program["observed_count"] == 1
    assert program["required_count"] == 1


def test_quality_state_digest_is_deterministic() -> None:
    first = compile_campaign_quality_state(
        workbench={"routes": {"route:1": {"edge_ids": ["edge:1"]}}}
    )
    second = compile_campaign_quality_state(
        workbench={"routes": {"route:1": {"edge_ids": ["edge:1"]}}}
    )

    assert first == second
