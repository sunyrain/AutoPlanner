from __future__ import annotations

from copy import deepcopy

import pytest

from cascade_planner.application.action_scheduler import schedule_next_action
from cascade_planner.application.campaign_actions import compile_action_opportunities
from cascade_planner.application.experimental_work_scheduling import (
    compile_experimental_work_scheduling,
    experimental_work_item_rank_key,
    experimental_work_item_scheduling,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _plan(*, priority: float = 0.5, memory: dict | None = None) -> dict:
    return {
        "content_sha256": "a" * 64,
        "priority_score": priority,
        "experience_memory": dict(memory or {}),
    }


def _request(
    plan: dict,
    *,
    check_count: int = 2,
    estimated_cost_units: float = 0.0,
    extra: dict | None = None,
) -> dict:
    return {
        "plan_payload": plan,
        "required_checks": [
            {"check_id": f"check:{index}"} for index in range(check_count)
        ],
        "exact_boundary": {
            "input_states": [{"state_id": "state:input"}],
            "output_states": [{"state_id": "state:output"}],
        },
        "resource_hints": {
            "timeout_s": 3600.0,
            "max_artifact_bytes": 100_000_000,
            "estimated_cost_units": estimated_cost_units,
        },
        **dict(extra or {}),
    }


def _work_item(
    plan: dict,
    request: dict,
    scheduling: dict,
    *,
    domain: str = "execution",
    linked: list[str] | None = None,
    dirty: list[str] | None = None,
) -> dict:
    return {
        "domain": domain,
        "linked_canonical_deficit_ids": list(linked or []),
        "dirty_hint_ids": list(dirty or []),
        "execution_request": request,
        "scheduling": scheduling,
    }


def test_rank_is_deterministic_and_ignores_target_or_dataset_labels() -> None:
    plan = _plan(memory={"disposition": "inconclusive"})
    request = _request(
        plan,
        extra={"target_name": "first label", "dataset_name": "dataset alpha"},
    )
    first = compile_experimental_work_scheduling(
        "execution", plan, ["deficit:b", "deficit:a"], ["dirty:b", "dirty:a"], request
    )
    reordered = compile_experimental_work_scheduling(
        "execution", plan, ["deficit:a", "deficit:b"], ["dirty:a", "dirty:b"], request
    )
    renamed = compile_experimental_work_scheduling(
        "execution",
        plan,
        ["deficit:b", "deficit:a"],
        ["dirty:b", "dirty:a"],
        {**request, "target_name": "unrelated", "dataset_name": "dataset beta"},
    )

    assert first == reordered == renamed
    for domain in ("biocatalytic", "execution", "mechanism"):
        domain_rank = compile_experimental_work_scheduling(
            domain,
            plan,
            ["deficit:a", "deficit:b"],
            ["dirty:a", "dirty:b"],
            request,
        )
        assert domain_rank["information_gain_score"] == first["information_gain_score"]
        assert domain_rank["estimated_cost_units"] == first["estimated_cost_units"]
        assert domain_rank["action_score"] == first["action_score"]


def test_high_information_low_cost_work_ranks_before_low_information_high_cost() -> None:
    high_plan = _plan(
        priority=1.0,
        memory={
            "positive_observation_count": 1,
            "negative_observation_count": 1,
            "strongest_transfer_scope": "structural_analog",
        },
    )
    low_plan = _plan(
        priority=0.1,
        memory={
            "positive_observation_count": 2,
            "strongest_transfer_scope": "exact_boundary",
        },
    )
    high_request = _request(high_plan, check_count=6, estimated_cost_units=0.0)
    low_request = _request(low_plan, check_count=1, estimated_cost_units=30.0)
    high = compile_experimental_work_scheduling(
        "execution", high_plan, ["deficit:1"], ["dirty:1"], high_request
    )
    low = compile_experimental_work_scheduling(
        "execution", low_plan, [], [], low_request
    )
    high_item = _work_item(
        high_plan,
        high_request,
        high,
        linked=["deficit:1"],
        dirty=["dirty:1"],
    )
    low_item = _work_item(low_plan, low_request, low)

    assert high["information_gain_score"] > low["information_gain_score"]
    assert high["estimated_cost_units"] < low["estimated_cost_units"]
    assert high["value_per_cost"] > low["value_per_cost"]
    assert sorted(
        [("low", low_item), ("high", high_item)],
        key=experimental_work_item_rank_key,
    )[0][0] == "high"
    frontier = {
        "content_sha256": "f" * 64,
        "items": [
            {
                "deficit_id": item_id,
                "kind": "program_validation",
                "object_id": item_id,
                "entity_ids": [item_id],
                "route_family_ids": ["route:fixture"],
                "dependency_ids": list(item.get("linked_canonical_deficit_ids") or []),
                "deterministic": True,
                "model_allowed": False,
                "priority": item["scheduling"]["action_priority"],
                "reason": "program_candidate_requires_specialized_validation",
                "score": item["scheduling"]["action_score"],
            }
            for item_id, item in (("low", low_item), ("high", high_item))
        ],
    }
    decision = schedule_next_action(compile_action_opportunities(frontier))
    assert decision["selected_action"]["deficit_id"] == "high"


def test_experience_changes_priority_but_cannot_inject_authority() -> None:
    supported_memory = {
        "positive_observation_count": 1,
        "strongest_transfer_scope": "structural_analog",
    }
    malicious_memory = {**supported_memory, "grants_validation": True}
    supported_plan = _plan(memory=supported_memory)
    malicious_plan = _plan(memory=malicious_memory)
    supported = compile_experimental_work_scheduling(
        "mechanism", supported_plan, [], [], _request(supported_plan)
    )
    malicious = compile_experimental_work_scheduling(
        "mechanism", malicious_plan, [], [], _request(malicious_plan)
    )
    conflicting_plan = _plan(
        memory={"positive_observation_count": 1, "negative_observation_count": 1}
    )
    conflicting = compile_experimental_work_scheduling(
        "mechanism", conflicting_plan, [], [], _request(conflicting_plan)
    )

    assert supported == malicious
    assert conflicting["information_gain_score"] > supported["information_gain_score"]
    assert supported["semantics"][
        "ranking_grants_no_validation_claim_proof_or_completion"
    ] is True
    assert "grants_validation" not in supported["action_score"]


def test_recomputed_scheduling_tamper_fails_closed() -> None:
    plan = _plan()
    request = _request(plan)
    scheduling = compile_experimental_work_scheduling(
        "execution", plan, ["deficit:1"], [], request
    )
    item = _work_item(
        plan, request, scheduling, linked=["deficit:1"]
    )
    assert experimental_work_item_scheduling(item) == scheduling

    tampered = deepcopy(item)
    tampered["scheduling"]["information_gain_score"] = 0.0
    tampered["scheduling"].pop("content_sha256")
    tampered["scheduling"]["content_sha256"] = strict_canonical_json_sha256(
        tampered["scheduling"]
    )

    assert experimental_work_item_scheduling(tampered) == {}
    assert experimental_work_item_rank_key(("tampered", tampered))[0] == 1


def test_invalid_numeric_resource_hint_is_rejected() -> None:
    plan = _plan()
    request = _request(plan)
    request["resource_hints"]["estimated_cost_units"] = float("nan")

    with pytest.raises(ValueError, match="scheduling_number_invalid"):
        compile_experimental_work_scheduling("execution", plan, [], [], request)
