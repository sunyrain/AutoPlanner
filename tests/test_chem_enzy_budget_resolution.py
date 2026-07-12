from __future__ import annotations

import pytest

from cascade_planner.baselines.chem_enzy_budget import (
    budgeted_chemenzy_payload,
    budgeted_chemenzy_policy,
    classify_chemenzy_attempt_outcome,
    finalize_effective_chemenzy_budget,
    resolve_chemenzy_budget,
)
from cascade_planner.harness.agent_action_planner import (
    _can_run_guided_chemenzy,
    _guided_retry_payload,
)
from cascade_planner.harness.agentic_blackboard import (
    initialize_agent_blackboard,
    update_blackboard_from_action,
)
from cascade_planner.harness.failure_critic import compile_failure_critic_report
from cascade_planner.harness.tools import (
    HarnessBudget,
    ToolExecutionState,
    _chemenzy_budget_authority,
    run_chemenzy,
    run_guided_chemenzy_rerun,
    run_route_expansion_subgoal_search,
)


NIRMATRELVIR = (
    "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)"
    "C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C"
)
NIRMATRELVIR_PRIMARY_AMIDE = (
    "CC(C)(C)[C@H](NC(=O)C(F)(F)F)C(=O)N1C[C@H]2[C@@H]([C@H]1C(=O)"
    "N[C@@H](C[C@@H]1CCNC1=O)C(N)=O)C2(C)C"
)


def _policy(*, depth: int = 6, iterations: int = 10, topk: int = 20) -> dict:
    return {
        "schema_version": "chem_enzy_search_policy.v1",
        "policy_id": "budget-test",
        "budget": {
            "max_depth": depth,
            "max_iterations": iterations,
            "expansion_topk": topk,
        },
    }


def _guided_policy(*, depth: int = 6, iterations: int = 10, topk: int = 20) -> dict:
    return {
        **_policy(depth=depth, iterations=iterations, topk=topk),
        "operator_id": "budget-test-operator",
        "case_id": "budget-test",
        "evidence_refs": ["patent:budget-test"],
        "terminal_blacklist": [],
        "anchor_whitelist": [],
        "active_bridge_tasks": [],
        "accepted_exact_row_ids": [],
        "selected_analogical_hypothesis_ids": [],
        "selected_analogical_template_ids": [],
        "forbidden_template_ids": [],
        "preferred_subgoal": {},
        "source_budget": {
            "require_target_core_retention": True,
            "max_unexplained_heavy_atom_jump": 12,
            "analogy_is_advisory_only": True,
        },
        "rerun_reason": "budget contract test",
        "mode": "guided",
        "compiler_metadata": {"requires_verifier": True, "no_solved_claim": True},
    }


def test_complex_standard_planner_request_is_raised_to_deep_floor() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={
            "max_steps": 5,
            "chem_enzy_iterations": 8,
            "chem_enzy_expansion_topk": 20,
        },
        policy=_policy(depth=5, iterations=8, topk=20),
        authority="planner_advisory",
        attempt_index=1,
        timeout_cap_s=3600,
    )

    assert resolution.target_heavy_atoms == 35
    assert resolution.attempt_kind == "standard"
    assert resolution.requested_budget.to_dict()["max_depth"] == 5
    assert resolution.attempt_budget.max_depth == 20
    assert resolution.attempt_budget.max_iterations == 50
    assert resolution.attempt_budget.expansion_topk == 100
    assert "max_depth_raised_to_attempt_floor" in resolution.adjustments


def test_complex_probe_keeps_small_request_and_caps_large_request() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={
            "initial_probe": True,
            "max_steps": 12,
            "chem_enzy_iterations": 60,
            "chem_enzy_expansion_topk": 120,
            "timeout_s": 900,
        },
        policy={},
        authority="planner_advisory",
        attempt_index=1,
        timeout_cap_s=3600,
    )

    assert resolution.attempt_kind == "probe"
    assert resolution.attempt_budget.max_depth == 6
    assert resolution.attempt_budget.max_iterations == 10
    assert resolution.attempt_budget.expansion_topk == 20
    assert resolution.attempt_budget.timeout_s == 180


def test_probe_exhaustion_promotes_next_attempt_to_complex_standard() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={
            "initial_probe": True,
            "max_steps": 5,
            "chem_enzy_iterations": 8,
            "chem_enzy_expansion_topk": 20,
        },
        policy={},
        authority="planner_advisory",
        attempt_index=2,
        prior_attempt={"outcome": "probe_exhausted"},
        timeout_cap_s=3600,
    )

    assert resolution.attempt_kind == "standard"
    assert resolution.attempt_budget.max_depth == 20
    assert resolution.attempt_budget.max_iterations == 50
    assert resolution.attempt_budget.expansion_topk == 100


def test_complex_retry_uses_bounded_retry_floor() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={"max_steps": 6, "chem_enzy_iterations": 10, "chem_enzy_expansion_topk": 20},
        policy={},
        authority="planner_advisory",
        attempt_index=2,
        prior_attempt={"outcome": "search_exhausted"},
        timeout_cap_s=3600,
    )

    assert resolution.attempt_kind == "retry"
    assert resolution.attempt_budget.max_depth == 20
    assert resolution.attempt_budget.max_iterations == 60
    assert resolution.attempt_budget.expansion_topk == 120


def test_operator_explicit_budget_bypasses_floor_but_not_absolute_cap() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={},
        policy=_policy(depth=9, iterations=12, topk=33),
        authority="operator_explicit",
        attempt_index=1,
        timeout_cap_s=3600,
    )

    assert resolution.attempt_budget.max_depth == 9
    assert resolution.attempt_budget.max_iterations == 12
    assert resolution.attempt_budget.expansion_topk == 33

    capped = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="native",
        payload={"max_steps": 99, "chem_enzy_iterations": 900, "chem_enzy_expansion_topk": 700},
        policy={},
        authority="operator_explicit",
        attempt_index=1,
        timeout_cap_s=3600,
    )
    assert capped.attempt_budget.max_depth == 20
    assert capped.attempt_budget.max_iterations == 500
    assert capped.attempt_budget.expansion_topk == 500


def test_child_budget_uses_actual_child_smiles_complexity() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR_PRIMARY_AMIDE,
        action_kind="child",
        payload={"max_steps": 6, "chem_enzy_iterations": 10, "chem_enzy_expansion_topk": 20},
        policy={},
        authority="planner_advisory",
        attempt_index=1,
        timeout_cap_s=3600,
    )

    assert resolution.target_heavy_atoms == 36
    assert resolution.complexity_profile == "complex"
    assert resolution.attempt_budget.max_depth == 20
    assert resolution.attempt_budget.max_iterations == 50
    assert resolution.attempt_budget.expansion_topk == 100


def test_budget_payload_policy_and_effective_audit_remain_consistent() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={"max_steps": 5, "chem_enzy_iterations": 8, "chem_enzy_expansion_topk": 20},
        policy=_policy(depth=5, iterations=8, topk=20),
        authority="planner_advisory",
        attempt_index=1,
        timeout_cap_s=3600,
    )
    payload = budgeted_chemenzy_payload({}, resolution)
    policy = budgeted_chemenzy_policy(_policy(), resolution)
    finalized = finalize_effective_chemenzy_budget(
        resolution,
        max_depth=20,
        max_iterations=50,
        expansion_topk=100,
    )

    assert payload["max_steps"] == 20
    assert payload["chem_enzy_budget_resolution"]["requested_budget"]["max_depth"] == 5
    assert policy["budget"]["max_depth"] == 20
    assert policy["compiler_metadata"]["budget_authority"] == "planner_advisory"
    assert finalized.effective_budget is not None
    assert finalized.effective_budget.max_depth == 20

    with pytest.raises(ValueError, match="exceeds approved"):
        finalize_effective_chemenzy_budget(
            resolution,
            max_depth=21,
            max_iterations=50,
            expansion_topk=100,
        )


def test_probe_no_route_is_not_a_full_search_failure() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={"initial_probe": True},
        policy={},
        authority="planner_advisory",
        attempt_index=1,
        timeout_cap_s=3600,
    )

    outcome = classify_chemenzy_attempt_outcome(
        resolution,
        {"search_status": {"status": "failed", "solved": False}, "failure_diagnosis": ["no_route_found"]},
    )

    assert outcome["outcome"] == "probe_exhausted"
    assert outcome["next_attempt_kind"] == "standard"
    assert outcome["search_exhaustive"] is False
    assert outcome["blocks_same_attempt"] is False


def test_guided_entry_applies_standard_floor_and_records_attempt(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    def fake_execute(*, request, **_kwargs):
        captured.append(dict(request))
        return {
            "schema_version": "chemenzy_web_result.v1",
            "ok": True,
            "routes": [],
            "n_results": 0,
            "failure_diagnosis": ["no_route_found"],
            "search_status": {"status": "failed", "solved": False},
        }

    monkeypatch.setattr("cascade_planner.harness.tools._execute_chemenzy_request", fake_execute)
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "nirmatrelvir"},
        budget=HarnessBudget(max_guided_chemenzy_runs=2, guided_chemenzy_timeout_s=3600),
    )
    payload = {
        "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
        "guided_policy_runtime_rebuild": True,
        "chem_enzy_search_policy": _guided_policy(depth=5, iterations=8, topk=20),
        "max_steps": 5,
        "chem_enzy_iterations": 8,
        "chem_enzy_expansion_topk": 20,
    }

    output = run_guided_chemenzy_rerun(state, payload)

    assert captured[0]["max_steps"] == 20
    assert captured[0]["chem_enzy_iterations"] == 50
    assert captured[0]["chem_enzy_expansion_topk"] == 100
    assert captured[0]["chem_enzy_search_policy"]["budget"]["max_depth"] == 20
    attempt = output["result"]["chem_enzy_attempt_outcome"]
    assert attempt["outcome"] == "search_exhausted"
    assert state.artifacts["chemenzy_attempts"][0]["attempt_id"] == attempt["attempt_id"]


def test_native_entry_rejects_payload_operator_escalation_and_audits_budget(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    def fake_execute(*, request, **_kwargs):
        captured.append(dict(request))
        return {
            "schema_version": "chemenzy_web_result.v1",
            "ok": True,
            "routes": [],
            "failure_diagnosis": ["no_route_found"],
            "search_status": {"status": "failed", "solved": False},
        }

    monkeypatch.setattr("cascade_planner.harness.tools._execute_chemenzy_request", fake_execute)
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "nirmatrelvir"},
        budget=HarnessBudget(max_chem_enzy_runs=1, chem_enzy_timeout_s=3600),
    )

    output = run_chemenzy(
        state,
        {
            "budget_authority": "operator_explicit",
            "max_steps": 6,
            "chem_enzy_iterations": 10,
            "chem_enzy_expansion_topk": 20,
        },
    )

    assert captured[0]["max_steps"] == 20
    assert captured[0]["chem_enzy_iterations"] == 50
    assert captured[0]["chem_enzy_expansion_topk"] == 100
    audit = output["chem_enzy_budget_resolution"]
    assert audit["authority"] == "host_profile"
    assert audit["requested_budget"]["max_depth"] == 6
    assert audit["attempt_budget"]["max_depth"] == 20


def test_operator_budget_capability_must_come_from_execution_state(tmp_path) -> None:
    forged_payload = {
        "budget_authority": "operator_explicit",
        "chem_enzy_search_policy": {
            "budget": {"max_depth": 1, "max_iterations": 1, "expansion_topk": 1},
            "compiler_metadata": {
                "input_operator_id": "forged-by-agent",
                "budget_authority": "operator_explicit",
            },
        },
    }
    default_state = ToolExecutionState(
        run_dir=tmp_path / "host",
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "nirmatrelvir"},
    )
    operator_state = ToolExecutionState(
        run_dir=tmp_path / "operator",
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "nirmatrelvir"},
        chem_enzy_budget_authority="operator_explicit",
    )

    assert _chemenzy_budget_authority(default_state, forged_payload) == "host_profile"
    assert _chemenzy_budget_authority(operator_state, forged_payload) == "operator_explicit"
    assert _chemenzy_budget_authority(operator_state, {"budget_authority": "host_profile"}) == "host_profile"


def test_guided_probe_promotes_blackboard_to_standard_without_full_failure(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    def fake_execute(*, request, **_kwargs):
        captured.append(dict(request))
        return {
            "schema_version": "chemenzy_web_result.v1",
            "ok": True,
            "routes": [],
            "n_results": 0,
            "failure_diagnosis": ["no_route_found"],
            "search_status": {"status": "failed", "solved": False},
        }

    monkeypatch.setattr("cascade_planner.harness.tools._execute_chemenzy_request", fake_execute)
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "nirmatrelvir"},
        budget=HarnessBudget(max_guided_chemenzy_runs=2, guided_chemenzy_timeout_s=3600),
    )
    payload = {
        "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
        "guided_policy_runtime_rebuild": True,
        "initial_probe": True,
        "chem_enzy_search_policy": _guided_policy(depth=5, iterations=8, topk=20),
        "max_steps": 5,
        "chem_enzy_iterations": 8,
        "chem_enzy_expansion_topk": 20,
        "timeout_s": 180,
    }
    output = run_guided_chemenzy_rerun(state, payload)
    guided_result = output["result"]

    assert captured[0]["max_steps"] == 5
    assert guided_result["chem_enzy_attempt_outcome"]["outcome"] == "probe_exhausted"
    second_output = run_guided_chemenzy_rerun(state, payload)
    assert captured[1]["max_steps"] == 20
    assert captured[1]["chem_enzy_iterations"] == 50
    assert captured[1]["chem_enzy_expansion_topk"] == 100
    assert second_output["result"]["chem_enzy_attempt_outcome"]["attempt_kind"] == "standard"

    board = initialize_agent_blackboard(
        target_input=state.target_input,
        preflight={
            "accepted": True,
            "case_id": "nirmatrelvir",
            "canonical_smiles": NIRMATRELVIR,
            "target_profile": {"heavy_atoms": 35, "rings": 3},
        },
        max_rounds=3,
        budget_limits={"max_guided_chemenzy_runs": 2},
    )
    board["literature_evidence"]["source_candidates"] = [
        {
            "source_ref": "doi:10.1000/budget-test",
            "doi": "10.1000/budget-test",
            "title": "budget test patent",
        }
    ]
    board["budget_state"]["chemenzy_runs"] = 1
    board = update_blackboard_from_action(
        board,
        action={"action_id": "probe", "action_type": "run_guided_chemenzy"},
        action_result=guided_result,
        round_index=1,
        run_dir=tmp_path,
    )

    assert board["chemenzy_attempts"][-1]["outcome"] == "probe_exhausted"
    assert any(row["reason"] == "guided_chemenzy_probe_exhausted" for row in board["route_failures"])
    assert not any(row["reason"] == "no_route_found" for row in board["route_failures"])
    assert board["current_belief"]["pending_chemenzy_attempt"]["attempt_kind"] == "standard"
    assert _can_run_guided_chemenzy(board) is True
    standard_payload = _guided_retry_payload(board)
    assert standard_payload["attempt_kind"] == "standard"
    assert standard_payload["max_steps"] == 20
    assert standard_payload["chem_enzy_iterations"] == 50
    assert standard_payload["chem_enzy_expansion_topk"] == 100

    critic = compile_failure_critic_report(blackboard=board)
    assert not any(
        row.get("direction") == "repeat_same_guided_chemenzy_without_new_bridge_signal"
        for row in critic["blocked_directions"]
    )


def test_sequential_explicit_child_targets_start_at_zero_and_use_child_complexity(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    def fake_execute(*, request, **_kwargs):
        captured.append(dict(request))
        return {
            "schema_version": "chemenzy_web_result.v1",
            "ok": True,
            "routes": [],
            "failure_diagnosis": ["no_route_found"],
            "search_status": {"status": "failed", "solved": False},
        }

    monkeypatch.setattr("cascade_planner.harness.tools._execute_chemenzy_request", fake_execute)
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "nirmatrelvir", "target_profile": {"heavy_atoms": 35}},
        budget=HarnessBudget(max_route_expansion_subgoal_runs=3, guided_chemenzy_timeout_s=3600),
    )

    def child_payload(name: str, smiles: str) -> dict:
        return {
            "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
            "child_policy_runtime_rebuild": True,
            "max_targets": 1,
            "max_steps": 6,
            "chem_enzy_iterations": 10,
            "chem_enzy_expansion_topk": 20,
            "subgoal_targets": [
                {
                    "name": name,
                    "smiles": smiles,
                    "exact_target_override": True,
                    "target_equivalence_audit_required": True,
                    "policy_runtime_rebuild": True,
                }
            ],
        }

    first = run_route_expansion_subgoal_search(
        state,
        child_payload("primary amide precursor", NIRMATRELVIR_PRIMARY_AMIDE),
    )
    second = run_route_expansion_subgoal_search(
        state,
        child_payload("amino amide child", "NC(=O)[C@@H](N)C[C@@H]1CCNC1=O"),
    )

    assert first["accepted"] is True
    assert second["accepted"] is True
    assert [row["target_smiles"] for row in captured] == [
        NIRMATRELVIR_PRIMARY_AMIDE,
        "NC(=O)[C@@H](N)C[C@@H]1CCNC1=O",
    ]
    assert captured[0]["harness_search_boundary"]["target_heavy_atoms"] == 36
    assert captured[0]["max_steps"] == 20
    assert captured[0]["chem_enzy_iterations"] == 50
    assert captured[0]["chem_enzy_expansion_topk"] == 100


def test_explicit_child_history_dedupes_canonical_equivalent_smiles(monkeypatch, tmp_path) -> None:
    captured: list[str] = []

    def fake_execute(*, request, **_kwargs):
        captured.append(str(request["target_smiles"]))
        return {
            "schema_version": "chemenzy_web_result.v1",
            "ok": True,
            "routes": [],
            "failure_diagnosis": ["no_route_found"],
            "search_status": {"status": "failed", "solved": False},
        }

    monkeypatch.setattr("cascade_planner.harness.tools._execute_chemenzy_request", fake_execute)
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "parent", "target_smiles": "CCCC"},
        preflight={"case_id": "parent"},
        budget=HarnessBudget(max_route_expansion_subgoal_runs=3),
    )

    def payload(smiles: str) -> dict:
        return {
            "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
            "child_policy_runtime_rebuild": True,
            "max_targets": 1,
            "subgoal_targets": [
                {
                    "name": "ethanol child",
                    "smiles": smiles,
                    "exact_target_override": True,
                    "target_equivalence_audit_required": True,
                    "policy_runtime_rebuild": True,
                }
            ],
        }

    run_route_expansion_subgoal_search(state, payload("CCO"))
    duplicate = run_route_expansion_subgoal_search(state, payload("OCC"))

    assert captured == ["CCO"]
    assert duplicate["result"]["status"] == "exhausted"
    assert duplicate["result"]["reasons"] == ["explicit_child_targets_already_attempted"]


def _raw_stock_solved_but_invalid_result() -> dict:
    return {
        "schema_version": "chemenzy_web_result.v1",
        "ok": True,
        "accepted": True,
        "n_results": 9,
        "routes": [
            {"route_rank": index, "steps": [], "metrics": {"route_solved": True}}
            for index in range(9)
        ],
        "search_status": {"status": "solved", "solved": True},
    }


def _atom_balance_rejected_verifier() -> dict:
    return {
        "schema_version": "harness_route_verifier_report.v1",
        "accepted": False,
        "route_status": "fake_closed_rejected",
        "reasons": ["atom_balance_violation"],
        "warnings": [],
        "route_count": 9,
        "accepted_route_count": 0,
        "rejected_route_count": 9,
        "best_route_step_count": 0,
        "failure_events": [
            {"reason": "atom_balance_violation", "route_rank": index}
            for index in range(9)
        ],
    }


def test_raw_solved_is_not_attempt_authority_when_host_verifier_rejects() -> None:
    standard = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="child",
        payload={},
        policy={},
        authority="host_profile",
        attempt_index=1,
    )
    rejected = classify_chemenzy_attempt_outcome(
        standard,
        _raw_stock_solved_but_invalid_result(),
        verifier=_atom_balance_rejected_verifier(),
        verified_solved=False,
    )

    assert rejected["outcome"] == "verification_rejected"
    assert rejected["raw_solved"] is True
    assert rejected["verified_solved"] is False
    assert rejected["raw_route_count"] == 9
    assert rejected["verified_route_count"] == 0
    assert rejected["next_attempt_kind"] == "retry"
    assert rejected["blocks_same_attempt"] is True
    assert rejected["raw_search_status_is_authority"] is False

    retry = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="child",
        payload={},
        policy={},
        authority="host_profile",
        attempt_index=2,
        prior_attempt=rejected,
    )
    assert retry.attempt_kind == "retry"


def test_rejected_raw_solved_probe_advances_to_standard_without_no_route_blocker() -> None:
    probe = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={"initial_probe": True},
        policy={},
        authority="planner_advisory",
        attempt_index=1,
    )
    rejected = classify_chemenzy_attempt_outcome(
        probe,
        _raw_stock_solved_but_invalid_result(),
        verifier=_atom_balance_rejected_verifier(),
        verified_solved=False,
    )

    assert rejected["outcome"] == "verification_rejected"
    assert rejected["next_attempt_kind"] == "standard"
    assert rejected["blocks_same_attempt"] is False
    assert rejected["search_exhaustive"] is False
    standard = resolve_chemenzy_budget(
        target_smiles=NIRMATRELVIR,
        action_kind="guided",
        payload={"initial_probe": True},
        policy={},
        authority="planner_advisory",
        attempt_index=2,
        prior_attempt=rejected,
    )
    assert standard.attempt_kind == "standard"


def test_native_guided_and_child_outcomes_require_host_verified_route(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_execute(**_kwargs):
        return _raw_stock_solved_but_invalid_result()

    def fake_feedback(*_args, **_kwargs):
        return {
            "schema_version": "route_failure_feedback.v1",
            "accepted": True,
            "terminal_blacklist": [],
            "frontier_research_targets": [],
            "query_hints": [],
        }

    monkeypatch.setattr(
        "cascade_planner.harness.tools._execute_chemenzy_request",
        fake_execute,
    )
    monkeypatch.setattr(
        "cascade_planner.harness.tools.verify_chemenzy_raw_routes",
        lambda *_args, **_kwargs: _atom_balance_rejected_verifier(),
    )
    monkeypatch.setattr(
        "cascade_planner.harness.tools.compile_route_failure_feedback",
        fake_feedback,
    )
    monkeypatch.setattr(
        "cascade_planner.harness.tools.write_route_failure_feedback",
        lambda *_args, **_kwargs: str(tmp_path / "route_failure_feedback.json"),
    )

    native_state = ToolExecutionState(
        run_dir=tmp_path / "native",
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "native"},
        budget=HarnessBudget(max_chem_enzy_runs=1),
    )
    native_state.run_dir.mkdir(parents=True)
    native = run_chemenzy(native_state, {})
    native_attempt = native["chem_enzy_attempt_outcome"]
    assert native_attempt["outcome"] == "verification_rejected"
    assert native_attempt["raw_solved"] is True
    assert native_attempt["verified_solved"] is False
    assert native["result"]["raw_route_verifier"]["accepted"] is False
    host_verification = native["result"]["host_route_verification"]
    assert host_verification["backend_cascade_verifier_gate_enabled"] is False
    assert host_verification["invalid_raw_routes_retained_for_diagnostics"] is True
    assert host_verification["host_verifier_remains_authority_when_backend_gate_disabled"] is True

    guided_state = ToolExecutionState(
        run_dir=tmp_path / "guided",
        target_input={"target_name": "Nirmatrelvir", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "guided", "target_profile": {"heavy_atoms": 35}},
        budget=HarnessBudget(max_guided_chemenzy_runs=2),
    )
    guided_state.run_dir.mkdir(parents=True)
    guided = run_guided_chemenzy_rerun(
        guided_state,
        {
            "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
            "guided_policy_runtime_rebuild": True,
            "initial_probe": True,
            "chem_enzy_search_policy": _guided_policy(),
        },
    )
    guided_result = guided["result"]
    guided_attempt = guided_result["chem_enzy_attempt_outcome"]
    assert guided_result["solved"] is False
    assert guided_attempt["outcome"] == "verification_rejected"
    assert guided_attempt["next_attempt_kind"] == "standard"
    guided_standard = run_guided_chemenzy_rerun(
        guided_state,
        {
            "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
            "guided_policy_runtime_rebuild": True,
            "initial_probe": True,
            "chem_enzy_search_policy": _guided_policy(),
        },
    )["result"]
    standard_attempt = guided_standard["chem_enzy_attempt_outcome"]
    assert standard_attempt["attempt_kind"] == "standard"
    assert standard_attempt["outcome"] == "verification_rejected"
    assert standard_attempt["next_attempt_kind"] == "retry"
    assert standard_attempt["blocks_same_attempt"] is True

    board = initialize_agent_blackboard(
        target_input=guided_state.target_input,
        preflight={
            "accepted": True,
            "case_id": "guided",
            "canonical_smiles": NIRMATRELVIR,
            "target_profile": {"heavy_atoms": 35},
        },
        max_rounds=3,
        budget_limits={"max_guided_chemenzy_runs": 2},
    )
    board["literature_evidence"]["source_candidates"] = [
        {"source_ref": "doi:10.1000/outcome", "doi": "10.1000/outcome"}
    ]
    board["budget_state"]["chemenzy_runs"] = 1
    board = update_blackboard_from_action(
        board,
        action={"action_id": "probe", "action_type": "run_guided_chemenzy"},
        action_result=guided_result,
        round_index=1,
        run_dir=guided_state.run_dir,
    )
    assert board["current_belief"]["pending_chemenzy_attempt"]["attempt_kind"] == "standard"
    assert not any(row.get("reason") == "no_route_found" for row in board["route_failures"])
    board["budget_state"]["chemenzy_runs"] = 2
    board = update_blackboard_from_action(
        board,
        action={"action_id": "standard", "action_type": "run_guided_chemenzy"},
        action_result=guided_standard,
        round_index=2,
        run_dir=guided_state.run_dir,
    )
    assert any(
        row.get("reason") == "guided_chemenzy_verification_rejected"
        for row in board["route_failures"]
    )
    critic = compile_failure_critic_report(blackboard=board)
    assert critic["constraints"]["raw_chemenzy_solved_is_not_proof"] is True
    assert any(
        row.get("direction") == "accept_raw_chemenzy_solved_without_host_verification"
        for row in critic["blocked_directions"]
    )

    child_state = ToolExecutionState(
        run_dir=tmp_path / "child",
        target_input={"target_name": "parent", "target_smiles": NIRMATRELVIR},
        preflight={"case_id": "child", "target_profile": {"heavy_atoms": 35}},
        budget=HarnessBudget(max_route_expansion_subgoal_runs=1),
    )
    child_state.run_dir.mkdir(parents=True)
    child = run_route_expansion_subgoal_search(
        child_state,
        {
            "codex_payload_repair": {"schema_version": "codex_action_payload_repair.v1"},
            "child_policy_runtime_rebuild": True,
            "max_targets": 1,
            "subgoal_targets": [
                {
                    "name": "complex child",
                    "smiles": NIRMATRELVIR_PRIMARY_AMIDE,
                    "exact_target_override": True,
                    "target_equivalence_audit_required": True,
                    "policy_runtime_rebuild": True,
                }
            ],
        },
    )
    child_row = child["result"]["subgoals"][0]
    child_attempt = child_row["chem_enzy_attempt_outcome"]
    assert child_row["raw_solved"] is True
    assert child_row["verified_solved"] is False
    assert child_row["solved"] is False
    assert child_row["route_status"] == "fake_closed_rejected"
    assert child_attempt["outcome"] == "verification_rejected"
    assert child_attempt["next_attempt_kind"] == "retry"
