"""Condition completeness must survive the real wire -> Host -> Critic boundary."""
import json

from cascade_planner.agent.codex_worker import (
    WorkerBudget, WorkerProcessResult, WorkerTask, run_codex_worker,
    _worker_model_output_json_schema,
)
from cascade_planner.orchestration import sequential_strategy_director as d


def test_complete_staged_conditions_survive_wire_replay_and_review_without_admitting_bad_maps():
    # Longer than the old per-string 160-character ceiling: the finishing state
    # matters because acid deprotection alone supplies a salt, not this free base.
    conditions = [
        "Treat the N-Boc precursor with TFA in dichloromethane until deprotection is complete; "
        "remove volatile acid and solvent before adding aqueous NaOH, then extract the neutral "
        "amine. Acid treatment and basic workup are sequential, not simultaneous."
    ]
    mapped = "[CH3:1][CH2:2][NH2:3]"
    task = WorkerTask(task_id="stage-test", case_id="stage-test", task_type="paper_matched_route_step",
        required_artifact_type="RetrosynthesisProposalReport", budget=WorkerBudget(max_tool_calls=0),
        host_context={"target_smiles": "CCN", "selected_product": "CCN"})
    wire = {"checkpoint_relation": "executes_checkpoint", "reaction_intent": "Boc deprotection and neutralization",
            "conditions": conditions, "reaction_operations": [
                {"op": "set_explicit_h", "map_idx": 3, "count": 1, "no_implicit": True},
                {"op": "add_group", "map_idx": 3, "fragment_smiles": "[*]C(=O)OC(C)(C)C"}]}
    # Check the restriction actually supplied to the provider, then exercise
    # the Worker artifact contract, Host compiler and both Critic projections.
    assert "maxLength" not in _worker_model_output_json_schema(task)["properties"]["conditions"]["items"]
    raw = run_codex_worker(task, runner=lambda _: WorkerProcessResult(stdout=json.dumps(wire), exit_code=0, backend="runner"))
    assert raw.status == "accepted_draft"
    candidates, failures = d._reactionjson_candidates_from_record(raw, expected_product="CCN",
        mapped_product_smiles=mapped, require_reaction_operations=True)
    assert not failures and len(candidates) == 1
    step = d._step_row(candidates[0].expansion, step_id="deprotection")
    assert step["precursor_smiles"] == ["CCNC(=O)OC(C)(C)C"]
    prompt = d._critic_prompt(target="CCN", branch_index=0, strategy_card={}, steps=[step], paper_matched=True)
    assert conditions[0] in prompt
    for kind in ("key_event", "final_route"):
        p = d._critic_prompt(target="CCN", branch_index=0, strategy_card={}, steps=[step], paper_matched=True,
                             audit_kind=kind, focus_step_id="deprotection")
        assert "counting convention alone is not chemical failure" in p
        assert conditions[0] in p

    malformed = json.loads(json.dumps(wire))
    malformed["reaction_operations"][0]["map_idx"] = 999
    bad = run_codex_worker(task, runner=lambda _: WorkerProcessResult(stdout=json.dumps(malformed), exit_code=0, backend="runner"))
    candidates, failures = d._reactionjson_candidates_from_record(bad, expected_product="CCN",
        mapped_product_smiles=mapped, require_reaction_operations=True)
    assert not candidates and failures
