"""Matched frozen-state mechanism pilots through production workers and AiZ.

Prepare once, then execute isolated cells. This is a targeted intervention pilot,
not target-blind discovery or an estimate of population-level synthesis success.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from cascade_planner.agent.codex_worker import WorkerRunRecord  # noqa: E402
from cascade_planner.application.campaign_context import CampaignContext, CampaignContextDelta  # noqa: E402
from cascade_planner.application.run_kernel import RunRevision  # noqa: E402
from cascade_planner.interfaces.live_stock import standard_stock_catalog_builder  # noqa: E402
from cascade_planner.orchestration import sequential_strategy_director as director  # noqa: E402
from cascade_planner.runtime import AgentSpec, Budget  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: list(x) if isinstance(x, deque) else str(x)) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def journal(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def record(raw):
    return WorkerRunRecord(**{f.name: raw[f.name] for f in fields(WorkerRunRecord) if f.name in raw})


def source_case(label):
    audit = read(ROOT / 'docs/evaluation/evidence/current_io_review_20260905/audit.json')
    case = next(c for c in audit['cases'] if c['label'] == label)
    workspace = Path(case['journal_path']).parent
    inputs = {r['task_id']: r for r in journal(workspace / 'model-io.jsonl') if r.get('event') == 'model_input'}
    records = {r['record']['task_id']: r['record'] for r in journal(workspace / 'sequential-director-worker-records.jsonl')}
    report = read(workspace.parent.parent / 'target-only-solve-report.json')
    return inputs, records, report


def reconstruct(step_id, inputs, records):
    raw = next(r for r in records.values() if any(
        str(c.get('candidate_id') or '').split(':branch:', 1)[-1] == step_id.split(':branch:', 1)[-1]
        for c in (r.get('output_artifact') or {}).get('payload', {}).get('candidates', [])))
    context = json.loads(inputs[raw['task_id']]['prompt'].split('PaperMatchedRouteBuilderContext:\n', 1)[1])
    expansions = director._expansions_from_record(
        record(raw), expected_product=director._canonical_smiles(context['selected_leaf_mapped']),
        mapped_product_smiles=context['selected_leaf_mapped'], require_reaction_operations=True,
        single_step_only=True,
    )
    if not expansions or len(expansions) != 1:
        raise ValueError(f'Cannot reconstruct fixed input {step_id}')
    return director._step_row(expansions[0], step_id=step_id), raw


def coherent_prefix(steps):
    compiler = director.RouteJSONCompiler()
    state = compiler.compile_route_graph_state(
        mapped_target_smiles=steps[0]['mapped_product_smiles'], steps=steps,
        minimum_depth=1, rebase_materialized_local_maps=True,
    )
    rows = compiler.assemble_route(state.reactions, metadata=steps)
    validation = director._route_steps_host_replay_validation(rows, mapped_target_smiles=rows[0]['mapped_product_smiles'])
    if not validation.get('complete'):
        raise ValueError(f'Frozen prefix must replay before any paid call: {validation}')
    return rows


def prepare(output):
    if (output / 'inputs.json').exists():
        raise FileExistsError('Prepared inputs already exist; reuse them without overwriting.')
    inputs, records, report = source_case('Homocubane 72')
    source = next(r for r in inputs.values() if 'BlindUpstreamStrategyMilestoneInput:' in r['prompt'])
    milestone = json.loads(source['prompt'].split('BlindUpstreamStrategyMilestoneInput:\n', 1)[1])
    prefix = [reconstruct(r['step_id'], inputs, records)[0] for r in milestone['connected_path_reactions']]
    family = next(f for o in report['director_outcomes'] for f in (o.get('plan') or {}).get('route_families', [])
                  if f['route_family_id'] == 'codex:sequential:family:2')
    ids = {r['step_id'] for r in prefix}
    history = [h for h in family['key_event_critic_history']
               if h.get('assessment') and h.get('focus_step_id') in ids
               and set(h.get('required_selected_step_ids') or []) <= ids]
    # A historical failed re-review is not a new judgment. Freeze the earliest
    # valid checkpoint evidence that made this actual upstream request possible.
    history = history[:1]
    continuous = {'target': report['target']['canonical_smiles'], 'prefix': prefix,
                  'card': family['root_strategy_card'], 'history': history,
                  'source_task_id': source['task_id'], 'original_horizon_input': milestone}
    for step in prefix:
        step.update(strategy_id=continuous['card']['strategy_id'], strategy_digest=continuous['card']['strategy_digest'])
    state = director._selected_path_strategy_checkpoint_state(
        {'key_event_critic_history': history}, strategy_card=continuous['card'], steps=prefix)
    if not state['checkpoint_executed']:
        raise ValueError(f'Frozen continuation lacks a real executed checkpoint: {state}')
    continuous['checkpoint_state'] = state

    route_path = ROOT / 'results/discussion/p0-route-reassessment-20260906/bch/branch-3/context.json'
    route = read(route_path)
    inputs, records, _ = source_case('1,3-BCH boronate 2a')
    prefix = [reconstruct(s['step_id'], inputs, records)[0] for s in route['steps'][:4]]
    fixed_step, fixed_record = reconstruct(route['steps'][4]['step_id'], inputs, records)
    if fixed_step.get('checkpoint_relation') != 'executes_checkpoint':
        raise ValueError('The shared candidate must actually have claimed the checkpoint in original IO.')
    # Old Builder artifacts use local map namespaces. Use the same global
    # rebase + reserialization as production final-route repair before seeding.
    compiled = coherent_prefix([*prefix, fixed_step])
    prefix, fixed_step = compiled[:-1], compiled[-1]
    fixed_record = json.loads(json.dumps(fixed_record))
    fixed_record['output_artifact']['payload']['candidates'][0]['reaction_operations'] = fixed_step['reaction_operations']
    checkpoint = {'target': route['target_smiles'], 'prefix': prefix, 'card': route['strategy_card'],
                  'history': [], 'fixed_candidate_record': fixed_record, 'fixed_step': fixed_step}
    old_review = read(route_path.with_name('result.json'))['critique']
    dependency_review = read(ROOT / 'results/discussion/chemical-intent-canary-20260906/bch3/critic-result.json')['critique']
    dependencies = dependency_review.get('chemical_dependencies') or []
    if not dependencies:
        raise ValueError('The common dependency projection must come from recorded IO.')
    dependency = {'route_context': route, 'critique': {**old_review, 'chemical_dependencies': dependencies},
                  'source': str(route_path), 'dependency_source': 'bch3/critic-result.json',
                  'note': 'Fixed original reject plus separately recorded dependency hypotheses; identical in both arms except link projection.'}
    bundle = {'continuous': continuous, 'checkpoint': checkpoint, 'dependency': dependency}
    write(output / 'inputs.json', bundle)
    write(output / 'protocol.json', {
        'prepared_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': digest(bundle),
        'unit': 'one frozen state per mechanism, two arms, no statistical superiority claim',
        'budget_per_arm': {'model_invocations': 12, 'input_tokens': 800000, 'output_tokens': 100000,
                           'wall_time_s': 1800, 'builder_calls': 8, 'editor_attempts': 2},
        'interventions': {'continuous': 'maximum strategic milestones 1 vs 2; checkpoints enabled in both',
                          'checkpoint': 'Key Critic off vs on; maximum strategic milestones fixed at 1 in both; identical first candidate supplied',
                          'dependency': 'same route and fixed critique, remove vs expose only chemical_dependencies'},
        'evaluation': 'Fresh blinded model review of anonymous route endpoints/conditions; arm labels, planning history, Strategy and original Critic removed. Same backbone does not imply independent knowledge. Human review separate.',
        'cost': 'All planning overhead charged to each arm. Common historical prefixes are replayed without model calls. The supplied checkpoint candidate uses one common policy slot but zero provider tokens; physical provider calls are reported separately. Independent review cost is separate. Actual provider input can exceed the ceiling on the final in-flight call; no further call is then admitted.',
        'implementation': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT / 'cascade_planner/orchestration/sequential_strategy_director.py',
            ROOT / 'cascade_planner/orchestration/repair_recovery.py',
            ROOT / 'cascade_planner/agent/codex_worker.py']},
    })
    print('Prepared three frozen paired inputs', flush=True)


def runner_with_stock():
    stock = standard_stock_catalog_builder()
    def membership(values):
        values = list(values)
        catalog = stock(values, max_molecules=max(1, len(values)))
        members = {r['canonical_smiles'] for r in catalog['members']}
        return {v: director._canonical_smiles(v) in members for v in values}
    return director.SequentialStrategyDirectorRunner(
        stock_membership=membership,
        aizynthfinder_strategy_python_executable=str(ROOT / '.venv_aizynth/Scripts/python.exe'),
        aizynthfinder_strategy_stock_index=str(stock.index_path),
    )


def campaign(spec, target, source):
    return CampaignContext(
        run_id=spec.run_id, target={'canonical_smiles': target},
        revision=RunRevision(run_id=spec.run_id, revision=0, state_sha256=digest(source), graph_revision=0,
                             evidence_revision=0, deficit_sha256=digest([]), acceptance_sha256=digest({}),
                             status='isolated_mechanism_pilot', updated_at=datetime.now(timezone.utc).isoformat()),
        topology={}, route_portfolio={}, evidence={}, stock={}, deficits=(), proposal_history=(),
        failure_history=(), budget_state={}, acceptance_state={}, delta=CampaignContextDelta(),
    )


def execute(output, mechanism, arm):
    source = read(output / 'inputs.json')[mechanism]
    cell = output / f'{mechanism}-{arm}'
    if (cell / 'result.json').exists() or (cell / 'worker/model-io.jsonl').exists():
        raise FileExistsError('Cell already started; inspect durable IO instead of relaunching it.')
    config = director.DirectorConfig(
        planning_mode='sequential_branches', paper_matched_reach_profile=True,
        enable_transactional_path_repair=True, enable_key_event_critic=mechanism != 'dependency' and (mechanism != 'checkpoint' or arm == 'on'),
        strategy_tree_engine='aizynthfinder_mcts', strategy_portfolio_mode='paper_independent',
        strategy_branch_count=1, strategy_branch_workers=1, max_node_expansions_per_branch=8,
        max_strategic_milestones_per_branch=2 if mechanism == 'continuous' and arm == 'on' else 1,
        max_route_local_repair_rounds=2, max_node_prompt_bytes=96000,
        max_node_call_timeout_s=300, critic_call_timeout_s=300, max_wall_time_s=1800,
        max_output_tokens=8000, model='gpt-6-astra', reasoning_effort='medium',
    )
    quota = {'model_invocations': 12, 'input_tokens': 800000, 'output_tokens': 100000, 'wall_time_s': 1800}
    spec = AgentSpec(run_id='mechanism-pilot:' + cell.name, agent_id='planner', role='route_editor',
                     objective='Continue this frozen retrosynthesis state within its shared budget.',
                     idempotency_key=digest(source), context_hash=digest(source), budget=Budget(max_wall_time_s=1800),
                     metadata={'model': config.model, 'reasoning_effort': config.reasoning_effort,
                               'durable_worker_journal': True, 'allowed_workdir': str(cell / 'worker'),
                               'remaining_model_budget': quota})
    write(cell / 'config.json', {'config': asdict(config), 'quota': quota, 'source_sha256': digest(source),
                               'executed_director_sha256': hashlib.sha256(Path(director.__file__).read_bytes()).hexdigest()})
    runner = runner_with_stock()
    if mechanism == 'dependency':
        critique = dict(source['critique'])
        if arm == 'off':
            critique.pop('chemical_dependencies', None)
        route = director.RevisionBoundRouteCriticContext(**source['route_context'])
        result = runner.run_final_route_repair_once(
            spec, campaign_context=campaign(spec, route.target_smiles, source), route_context=route,
            critique=critique, config=config, route_family_alias='isolated:mechanism-pilot',
        )
    else:
        prefix = source['prefix']
        validation = director._route_steps_host_replay_validation(prefix, mapped_target_smiles=prefix[0]['mapped_product_smiles'])
        if not validation.get('complete'):
            raise ValueError(f'Frozen prefix is not replayable: {validation}')
        target = source['target']
        branch = {'branch_index': 0, 'steps': prefix, 'target_mapped_smiles': prefix[0]['mapped_product_smiles'],
                  'lens': 'Continue the supplied target-rooted path', 'strategy_card': source['card'],
                  'root_strategy_card': source['card'], 'strategy_tree_engine': 'aizynthfinder_mcts',
                  'strategy_milestone_cards': [source['card']], 'strategy_milestone_generation_count': 0,
                  'key_event_critic_history': source['history'], 'route_call_count': 0, 'call_count': 0,
                  'generated_step_id_prefix': 'pilot', 'expanded_products': set(), 'open_leaf_states': deque()}
        runner._prepare_worker_record_journal(spec)
        # Replay a common frozen prefix into the real AiZ tree without billing it
        # as a new LLM answer. Every seed still crosses AiZ's Host-graph adapter.
        real_sidecar = director.run_aizynthfinder_strategy_branch_sidecar
        def seeded_sidecar(**kwargs):
            real_handler = kwargs['request_handler']
            def handle(request):
                steps = request.get('route_steps') or []
                if len(steps) < len(prefix):
                    if [s['step_id'] for s in steps] != [s['step_id'] for s in prefix[:len(steps)]]:
                        raise ValueError('Frozen intervention path changed before continuation')
                    s = prefix[len(steps)]
                    return {'candidates': [{'candidate_id': 'fixed:' + s['step_id'],
                        'product_smiles': s['product_smiles'], 'mapped_product_smiles': s['mapped_product_smiles'],
                        'precursor_smiles': s['precursor_smiles'], 'mapped_precursor_smiles': s['mapped_precursor_smiles'],
                        'route_step': s, 'prior': 1.0, 'candidate_key': director._key_event_graph_fingerprint(s)}],
                        'model_call_consumed': False, 'host_replay_seed': True}
                return real_handler(request)
            kwargs['request_handler'] = handle
            return real_sidecar(**kwargs)
        if mechanism == 'checkpoint':
            actual_executor = runner.node_executor
            injected = False
            def first_shared_candidate(task):
                nonlocal injected
                if not injected and task.task_type == 'paper_matched_route_step':
                    injected = True
                    saved = record(source['fixed_candidate_record'])
                    return replace(saved, task_id=task.task_id, run_id=task.task_id + ':fixed', case_id=task.case_id,
                                   usage={}, elapsed_s=0, tool_calls=[], metadata={'fixed_shared_intervention': True})
                return actual_executor(task)
            runner.node_executor = first_shared_candidate
        started = time.monotonic()
        with patch.object(director, 'run_aizynthfinder_strategy_branch_sidecar', seeded_sidecar):
            records = runner._expand_seeded_branches_aizynthfinder(
                spec, target=target, seeded=[branch], existing_records=[], route_quota=director._NodeCallBudget(**quota),
                critic_editor_call_reserve=1, critic_input_reserve=12000, critic_output_reserve=8000,
                config=config, started=started)
        runner._run_codex_critics(spec, campaign(spec, target, source), [branch], records,
                                quota=director._NodeCallBudget(**quota), started=started, config=config)
        result = {'status': 'completed', 'branch': branch, 'usage': director._aggregate_usage(records, elapsed_s=time.monotonic()-started),
                  'records': [asdict(r) for r in records]}
    write(cell / 'result.json', result)
    print(json.dumps({'cell': cell.name, 'status': result['status'], 'usage': result.get('usage')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--mechanism', choices=['continuous', 'checkpoint', 'dependency'])
    parser.add_argument('--arm', choices=['off', 'on'])
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output)
    elif args.mechanism and args.arm:
        execute(output, args.mechanism, args.arm)
    else:
        parser.error('Choose --prepare or an explicit --mechanism and --arm')


if __name__ == '__main__':
    main()
