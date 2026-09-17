from copy import deepcopy

import pytest

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.orchestration.key_event_review import (
    _selected_path_strategy_checkpoint_state,
    key_event_review_update,
    pending_key_event_runtime_retry,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    _critique_from_record,
    _strategy_milestone_progress,
    _path_repair_reference_rows,
)


CARD = {"strategy_digest": "strategy-a"}


def test_runtime_retry_is_scope_bound_and_once_per_failed_dispatched_review():
    initial = _failed(evidence=None)
    branch = {"key_event_critic_history": [initial]}
    steps = [{"step_id": "focus"}, {"step_id": "upstream"}]
    assert pending_key_event_runtime_retry(branch, steps=steps) == initial
    assert pending_key_event_runtime_retry(branch, steps=[]) == {}
    deferred = {**initial, "task_id": "", "runtime_retry_of_task_id": initial["task_id"], "status": "budget_unavailable"}
    branch["key_event_critic_history"].append(deferred)
    assert pending_key_event_runtime_retry(branch, steps=steps) == initial
    retry = {**initial, "task_id": "retry", "runtime_retry_of_task_id": initial["task_id"]}
    branch["key_event_critic_history"].append(retry)
    assert pending_key_event_runtime_retry(branch, steps=steps) == {}
    # A different newly supplied precursor has its own review scope.
    followup = {**_failed(), "task_id": "new-evidence-failure"}
    branch["key_event_critic_history"].append(followup)
    assert pending_key_event_runtime_retry(branch, steps=steps) == followup
    assert pending_key_event_runtime_retry(branch, steps=steps[:1]) == {}
    branch["key_event_critic_history"].append(_judgment("pass", evidence="upstream"))
    assert pending_key_event_runtime_retry(branch, steps=steps) == {}


def _judgment(verdict="uncertain", *, evidence=None, match=True):
    row = {
        "focus_step_id": "focus",
        "strategy_digest": CARD["strategy_digest"],
        "obligation_id": "obligation",
        "checkpoint_match": match,
        "assessment": {"verdict": verdict, "blocking": verdict == "reject"},
        "required_selected_step_ids": ["focus"],
    }
    if evidence:
        row.update(
            review_of_obligation_id="obligation",
            review_evidence_step_id=evidence,
            required_selected_step_ids=["focus", evidence],
        )
    return row


def _failed(*, evidence="upstream"):
    return {
        **_judgment(evidence=evidence),
        "task_id": "failed-review",
        "status": "not_checkpoint",  # Saved histories used this misleading label.
        "critic_status": "unavailable",
        "checkpoint_match": False,
        "assessment": {},
    }


def _state(history, selected=("focus", "upstream")):
    return _selected_path_strategy_checkpoint_state(
        {"key_event_critic_history": history},
        strategy_card=CARD,
        steps=[{"step_id": sid} for sid in selected],
    )


@pytest.mark.parametrize("verdict", ["uncertain", "pass", "reject"])
def test_failed_followup_retains_scoped_judgment_and_exposes_unreviewed_evidence(verdict):
    previous = _judgment(verdict)
    history = [previous, _failed()]
    original = deepcopy(history)
    state = _state(history)
    assert state["chemical_confidence"] == verdict
    assert state["checkpoint_executed"] == (verdict != "reject")
    assert state["checkpoint_critic_passed"] == (verdict == "pass")
    assert state["source_row"] == previous
    assert state["source_row"]["required_selected_step_ids"] == ["focus"]
    assert state["review_pending"] is True
    assert state["pending_review_rows"][0]["review_evidence_step_id"] == "upstream"
    assert history == original
    projection = _strategy_milestone_progress(
        {"key_event_critic_history": history},
        strategy_card=CARD,
        steps=[{"step_id": "focus"}, {"step_id": "upstream"}],
        use_key_event_critic=True,
    )
    assert projection["review_pending"] is True
    assert projection["pending_reviews"][0]["task_id"] == "failed-review"


def test_first_review_failure_is_pending_and_never_passes():
    state = _state([_failed(evidence=None)])
    assert state["source_row"] == {}
    assert state["chemical_confidence"] == ""
    assert state["checkpoint_executed"] is False
    assert state["checkpoint_critic_passed"] is False
    assert state["review_pending"] is True


def test_valid_noncheckpoint_followup_supersedes_earlier_match():
    state = _state([_judgment(), _judgment("uncertain", evidence="upstream", match=False)])
    assert state["checkpoint_executed"] is False
    assert state["source_row"] == {}
    assert state["review_pending"] is False


def test_real_reject_survives_pruning_of_its_evidence_but_not_its_focus():
    history = [_judgment(), _judgment("reject", evidence="upstream"), _failed()]
    state = _state(history, selected=("focus",))
    assert state["chemical_confidence"] == "reject"
    assert state["checkpoint_executed"] is False
    assert _state(history, selected=("upstream",))["source_row"] == {}


@pytest.mark.parametrize("verdict", ["pass", "uncertain"])
def test_pruned_nonreject_evidence_and_other_strategy_cannot_supply_confidence(verdict):
    previous = _judgment(verdict, evidence="removed")
    foreign = {**_judgment("pass"), "strategy_digest": "other-strategy"}
    state = _state([previous, foreign, _failed()])
    assert state["checkpoint_executed"] is False
    assert state["chemical_confidence"] == ""


def test_later_success_clears_only_the_evidence_it_reviewed():
    history = [_judgment(), _failed(), _judgment("uncertain", evidence="different")]
    assert _state(history, ("focus", "upstream", "different"))["review_pending"] is True
    history.append(_judgment("pass", evidence="upstream"))
    state = _state(history, ("focus", "upstream", "different"))
    assert state["review_pending"] is False
    assert state["chemical_confidence"] == "pass"


@pytest.mark.parametrize("worker_status", ["timeout", "rejected_output", "worker_error"])
def test_invalid_worker_records_are_attempts_not_noncheckpoint_judgments(worker_status):
    record = WorkerRunRecord(
        run_id="failed", task_id="critic", case_id="test", backend="test",
        status=worker_status, output_validation={"accepted": False},
    )
    update = key_event_review_update(
        _critique_from_record(record), focus_step_id="focus",
        task_id=record.task_id, worker_status=record.status,
    )
    assert update["status"] == "review_unavailable"
    assert update["checkpoint_match"] is None
    assert update["assessment"] == {}
    assert update["worker_status"] == worker_status


@pytest.mark.parametrize("assessments", [
    [],
    [{"step_id": "other", "verdict": "pass"}],
    [{"step_id": "focus", "verdict": "pass"}, {"step_id": "focus", "verdict": "reject"}],
    [{"step_id": "focus", "verdict": "unknown"}],
])
def test_unbound_or_ambiguous_assessment_cannot_supersede_a_judgment(assessments):
    result = key_event_review_update(
        {"status": "pass", "checkpoint_match": True, "step_assessments": assessments},
        focus_step_id="focus", task_id="critic", worker_status="accepted_draft",
    )
    assert result["critic_status"] == "unavailable"
    assert result["checkpoint_match"] is None


@pytest.mark.parametrize("match", [True, False])
def test_host_binds_compact_wire_assessment_without_inventing_checkpoint_match(match):
    result = key_event_review_update(
        {"status": "uncertain", "checkpoint_match": match,
         "step_assessments": [{"verdict": "uncertain", "blocking": False}]},
        focus_step_id="focus", task_id="critic", worker_status="accepted_draft",
    )
    assert result["assessment"]["step_id"] == "focus"
    assert result["checkpoint_match"] is match
    assert result["status"] == ("uncertain" if match else "not_checkpoint")


def test_repair_prompt_keeps_valid_critic_summary_after_failed_attempt():
    rows = _path_repair_reference_rows(
        [{"step_id": "focus", "product_smiles": "CCO", "precursor_smiles": ["CC", "O"]}],
        key_event_critic_history=[_judgment(), _failed()],
    )
    assert rows[0]["prior_key_critic"]["verdict"] == "uncertain"
    assert rows[0]["prior_key_critic"]["checkpoint_match"] is True
