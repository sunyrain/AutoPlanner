"""Current review authority and historical judgments must remain distinct."""
import json
from pathlib import Path
import subprocess

from cascade_planner.web.v4_live_synthesis import (
    _apply_critic_step_review, _bind_critic_slots, _critic_slot_bindings,
    project_live_synthesis,
)


def test_missing_final_review_keeps_history_without_promoting_a_verdict():
    insights = {}
    _apply_critic_step_review(insights, {
        'overall_assessment': 'reject', 'step_assessments': [
            {'step_id': 's1', 'verdict': 'reject', 'reasons': ['Missing oxygen donor'],
             'condition_assessment': 'Oxygen input is unspecified'}]}, ['s1'], task_id='review:old')
    _apply_critic_step_review(insights, {'overall_assessment': 'unavailable'}, ['s1'])
    assert insights['s1']['critic_verdict'] == 'unavailable'
    assert insights['s1']['historical_critic']['critic_verdict'] == 'reject'
    assert insights['s1']['historical_critic']['critic_reasons'] == ['Missing oxygen donor']
    assert insights['s1']['historical_critic']['critic_task_id'] == 'review:old'
    # An overall viable outcome does not invent explicit step-level passes.
    _apply_critic_step_review(insights, {'overall_assessment': 'viable'}, ['s1'])
    assert insights['s1']['critic_verdict'] == 'reviewed'
    assert 'historical_critic' not in insights['s1']


def test_blind_review_slots_are_scoped_to_the_exact_task_and_not_position():
    reviews = [
        {'critic_task_id': 'call:a', 'step_assessments': [
            {'review_slot': 'review-002', 'step_id': 's1'},
            {'review_slot': 'review-001', 'step_id': 's2'}]},
        {'critic_task_id': 'call:b', 'step_assessments': [
            {'review_slot': 'review-001', 'step_id': 'unrelated'}]},
    ]
    bindings = _critic_slot_bindings(reviews)
    rows = [{'review_slot': 'review-001', 'verdict': 'reject'}]
    assert _bind_critic_slots(rows, bindings['call:a'])[0]['step_id'] == 's2'
    assert _bind_critic_slots(rows, bindings['call:b'])[0]['step_id'] == 'unrelated'
    assert not _bind_critic_slots(rows, {})[0]['step_id']
    reviews.append({'critic_task_id': 'call:a', 'step_assessments': [
        {'review_slot': 'review-001', 'step_id': 'conflicting'}]})
    assert 'review-001' not in _critic_slot_bindings(reviews)['call:a']
    assert 'step_id' not in rows[0]


def test_final_history_is_not_leaked_into_earlier_replay_or_current_authority(tmp_path):
    path = tmp_path / '.autoplanner/director-workspace/model-io.jsonl'
    path.parent.mkdir(parents=True)
    step_id = 'codex:branch:1:node:1:candidate:1'
    old_task = 'critic:historical'
    step = {'step_id': step_id, 'product_smiles': 'CCO', 'precursor_smiles': ['CC=O'],
            'conditions': ['Screen KRED'], 'reaction_family': 'Reduction'}
    assessment = {'review_slot': 'review-001', 'verdict': 'uncertain',
                  'reasons': ['Exact substrate activity is unverified']}
    critic = {'overall_assessment': 'uncertain', 'critic_task_id': old_task,
              'route_overall_evaluation': 'An unverified enzyme hypothesis',
              'step_assessments': [{**assessment, 'step_id': step_id}]}
    events = [
        {'event': 'model_input', 'artifact_type': 'ChemicalStrategyCritique', 'task_id': old_task,
         'prompt': 'PaperMatchedRouteCriticInput:\n' + json.dumps({
             'phase': 'independent_chemical_critic', 'branch_id': 1, 'campaign_target': 'CCO',
             'steps': [{**{k: v for k, v in step.items() if k != 'step_id'}, 'review_slot': 'review-001'}]})},
        {'event': 'model_output', 'artifact_type': 'ChemicalStrategyCritique', 'task_id': old_task,
         'status': 'accepted_draft', 'output_artifact': {'payload': {
             'overall_assessment': 'uncertain', 'step_assessments': [assessment]}}},
    ]
    path.write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
    report = {
        'director_outcomes': [{'plan': {'multi_step_skeletons': [{
            'route_family_id': 'codex:sequential:family:1', 'steps': [step], 'chemical_critic': critic}]}}],
        'stages': [{'stage': 'final_route_critic', 'detail': {'results': [{
            'branch_index': 0, 'critic_status': 'unavailable', 'reviewed_step_ids': [step_id]}]}}],
    }
    report_path = tmp_path / 'target-only-solve-report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    saved = (path.read_bytes(), report_path.read_bytes())
    projection = project_live_synthesis(model_io_path=path,
        job={'job_id': 'test', 'status': 'historical', 'target_smiles': 'CCO'}, include_replay=True)
    branch = projection['branches'][0]
    assert branch['chemical_critic_status'] == 'unavailable'
    assert branch['historical_critic']['overall_assessment'] == 'uncertain'
    assert branch['steps'][0]['critic_verdict'] == 'unavailable'
    assert branch['steps'][0]['historical_critic']['critic_reasons'] == assessment['reasons']
    frames = projection['replay']['frames']
    first = frames[0]['branch_updates'][0]
    assert 'historical_critic' not in first
    historical_frames = [b for f in frames[1:-1] for b in f['branch_updates']
                         if any(s.get('critic_verdict') == 'uncertain' for s in b['steps'])]
    assert historical_frames
    assert all(s['step_id'] == step_id for b in historical_frames for s in b['steps'])
    final = frames[-1]['branch_updates'][0]
    assert final['steps'][0]['historical_critic'] == branch['steps'][0]['historical_critic']
    assert (path.read_bytes(), report_path.read_bytes()) == saved


def test_shared_details_render_history_and_missing_binding_as_different_facts():
    static = Path(__file__).resolve().parents[1] / 'cascade_planner/web/static'
    script = r"""
const assert=require('node:assert/strict'),view=require(process.argv[1]);
const step={critic_verdict:'unavailable',historical_critic:{critic_verdict:'reject',
 critic_reasons:['Missing <oxygen> donor'],critic_condition_assessment:'Historical condition review'}};
const saved=JSON.stringify(step),html=view.reactionDetailHtml(step);
assert.ok(html.includes('历史步骤 Critic'));assert.ok(html.includes('建议拒绝 · reject'));
assert.ok(html.includes('当前终审未绑定'));assert.ok(html.includes('历史 Critic 依据'));
assert.ok(html.includes('Missing &lt;oxygen&gt; donor'));assert.ok(!html.includes('<oxygen>'));
assert.ok(html.includes('不代表当前最终路线已通过'));
assert.equal(JSON.stringify(step),saved);
const empty=view.criticPresentation({critic_verdict:'unavailable'});
assert.equal(empty.label,'当前终审未绑定');assert.equal(empty.verdict,'unavailable');
assert.ok(!empty.historical);
const current=view.criticPresentation({...step,critic_verdict:'pass'});
assert.equal(current.verdict,'pass');assert.ok(!current.historical);
assert.ok(view.criticPresentation({critic_verdict:'reviewed'}).label.includes('未单列步骤判定'));
const branch={chemical_critic_status:'unavailable',historical_critic:{overall_assessment:'reject',route_overall_evaluation:'Missing reagent'}};
assert.ok(view.historicalRouteReviewHtml(branch).includes('历史整路 Critic'));
assert.equal(view.historicalRouteReviewHtml({...branch,chemical_critic_status:'uncertain'}),'');
"""
    subprocess.run(['node', '-e', script, str(static / 'route_quality.js')], check=True,
                   capture_output=True, text=True, encoding='utf-8')
