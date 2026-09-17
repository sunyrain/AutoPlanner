from dataclasses import replace
from unittest.mock import Mock

import pytest

from cascade_planner.agent.codex_worker import (
    WorkerTask, _worker_model_output_json_schema,
    _materialize_paper_matched_artifact, _paper_matched_route_step_contract_reasons,
)
from cascade_planner.orchestration.repair_recovery import recovery_request, previous_repair_feedback
from cascade_planner.orchestration.sequential_strategy_director import (
    DirectorConfig, SequentialStrategyDirectorRunner,
)


@pytest.mark.parametrize('step', ['retained', 'sibling', '', 'invented'])
def test_backtracking_cannot_modify_retained_or_unselected_work(step):
    request, error = recovery_request(
        {'action': 'backtrack', 'step_id': step, 'reason': 'wrong precursor state'},
        reversible_step_ids=['provisional'],
    )
    assert request is None
    assert error == 'repair_recovery_step_outside_provisional_path'


def test_recovery_actions_do_not_invent_a_chemical_verdict():
    assert recovery_request(None, reversible_step_ids=[]) == (None, '')
    assert recovery_request({'action': 'expand', 'step_id': '', 'reason': ''},
                            reversible_step_ids=[]) == (None, '')
    assert recovery_request({'action': 'expand', 'step_id': 'reference-only', 'reason': ''},
                            reversible_step_ids=[]) == (None, '')
    for action, step in [('backtrack', 'p'), ('expand_scope', '')]:
        request, error = recovery_request(
            {'action': action, 'step_id': step, 'reason': 'required state not reachable here'},
            reversible_step_ids=['p'],
        )
        assert not error and set(request) == {'action', 'step_id', 'reason'}


def test_repair_only_wire_roundtrips_control_without_fake_reactions():
    task = WorkerTask(task_id='t', case_id='c', task_type='paper_matched_route_step',
                      required_artifact_type='RetrosynthesisProposalReport',
                      host_context={'allow_repair_recovery': True})
    wire = {'recovery': {'action': 'backtrack', 'step_id': 'p', 'reason': 'reconsider activation'},
            'checkpoint_relation': 'preparatory', 'reaction_intent': '',
            'reaction_operations': [], 'conditions': []}
    # Host materialization is exercised, not just the model-facing schema.
    assert 'recovery' in _worker_model_output_json_schema(task)['properties']
    assert 'recovery' not in _worker_model_output_json_schema(replace(task, host_context={}))['properties']
    payload = _materialize_paper_matched_artifact(task, wire, backend='test')['payload']
    assert payload['candidates'][0]['recovery'] == wire['recovery']
    assert not _paper_matched_route_step_contract_reasons(payload, allow_recovery=True)
    assert _paper_matched_route_step_contract_reasons(payload) == ['paper_route_step_recovery_forbidden']
    payload['candidates'][0]['reaction_operations'] = [{'op': 'invert_stereocenter', 'map_idx': 2}]
    assert _paper_matched_route_step_contract_reasons(payload, allow_recovery=True) == ['repair_recovery_must_not_include_reaction']


def test_scope_escalation_reuses_transaction_failures_and_stops_at_budget():
    runner = SequentialStrategyDirectorRunner()
    branch = {'editor_attempt_count': 0, 'path_repair_transactions': []}
    config = DirectorConfig(max_route_local_repair_rounds=2)

    def failed_attempt(_spec, **kwargs):
        assert previous_repair_feedback(branch['path_repair_transactions']) == (
            {} if not kwargs['recovery_attempt'] else branch['path_repair_transactions'][-1]
        )
        branch['editor_attempt_count'] += 1
        branch['path_repair_transactions'].append({
            'status': 'retained_uncommitted_prefix',
            'recovery_request': {'action': 'expand_scope', 'reason': 'retained state incompatible'},
        })
        return False

    runner._repair_branch_once = Mock(side_effect=failed_attempt)
    assert not runner._repair_branch_transactionally(None, branch=branch, config=config)
    assert runner._repair_branch_once.call_count == 2
    assert not runner._repair_branch_transactionally(None, branch=branch, config=config)
    assert runner._repair_branch_once.call_count == 2


def test_scope_retry_is_not_triggered_by_an_unclassified_failure():
    runner = SequentialStrategyDirectorRunner()
    runner._repair_branch_once = Mock(return_value=False)
    assert not runner._repair_branch_transactionally(
        None, branch={}, config=DirectorConfig(max_route_local_repair_rounds=6),
    )
    assert runner._repair_branch_once.call_count == 1
