"""Bounded real AiZ/Builder/checkpoint/final-Critic continuation of a natural fragment.

The alcohol is an experimental observation boundary, never a synthetic stock
claim. Both arms keep real stock lookup and all ordinary chemistry/replay checks.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import ROOT, read, write, digest, d
from cascade_planner.orchestration.reaction_granularity import BUILDER_GRANULARITY_GUIDANCE, CRITIC_GRANULARITY_GUIDANCE
from cascade_planner.runtime import AgentSpec, Budget

helper_path = ROOT / 'docs/evaluation/evidence/chemical_intent_canary_20260906/paired.py'
helper_spec = importlib.util.spec_from_file_location('existing_paired_helpers', helper_path)
helpers = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['strict', 'transformation'], required=True)
    args = parser.parse_args()
    base = ROOT / 'results/discussion/reaction-granularity-20260907'
    source = next(r for r in read(base / 'explicit-no-tools-v2/inputs.json') if r['case'] == 'xanthate' and r['arm'] == 'strict')
    cell = base / ('aiz-canary--' + args.arm)
    if cell.exists():
        raise FileExistsError('Preserve existing run; inspect its durable outputs.')
    target = source['target']
    card = d.normalize_strategy_policy_card(source['card'])
    config = d.DirectorConfig(planning_mode='sequential_branches', paper_matched_reach_profile=True,
        enable_transactional_path_repair=True, enable_key_event_critic=True,
        strategy_tree_engine='aizynthfinder_mcts', strategy_portfolio_mode='paper_independent',
        strategy_branch_count=1, strategy_branch_workers=1, max_node_expansions_per_branch=4,
        max_strategic_milestones_per_branch=1, max_route_local_repair_rounds=0,
        max_node_prompt_bytes=96000, max_node_call_timeout_s=300, critic_call_timeout_s=300,
        max_wall_time_s=1200, max_output_tokens=8000, model='gpt-6-astra', reasoning_effort='medium')
    quota = {'model_invocations': 9, 'input_tokens': 650000, 'output_tokens': 100000, 'wall_time_s': 1200}
    spec = AgentSpec(run_id='granularity-canary:' + args.arm, agent_id='planner', role='route_builder',
        objective='Develop this supplied natural-product route fragment within a fixed horizon.',
        idempotency_key=digest(source), context_hash=digest(source), budget=Budget(max_wall_time_s=1200),
        metadata={'model': config.model, 'reasoning_effort': config.reasoning_effort, 'durable_worker_journal': True,
                  'allowed_workdir': str(cell / 'worker'), 'remaining_model_budget': quota})
    runner = helpers.runner_with_stock()
    runner._prepare_worker_record_journal(spec)
    old_lines = [line for line in source['prompt'].split('\n') if line.startswith((
        'One candidate must represent one executable reaction.',
        'Conditions describe the forward reaction environment',
        'Check functional-group compatibility within the replayed precursor set'))]
    if len(old_lines) != 3:
        raise ValueError('Missing original policy lines')
    old_key = 'Reject a focus edge that telescopes independent reactions into one graph program. A concerted or genuinely inseparable cascade is one event; a separate protection/deprotection, activation, redox, workup transformation, or changed reagent stage with its own covalent change must be an adjacent edge. The smallest suggested revision should request that split rather than hiding both events in one operation.'

    def executor(real):
        def execute(task):
            prompt = task.objective
            if args.arm == 'strict':
                prompt = prompt.replace(BUILDER_GRANULARITY_GUIDANCE, '\n'.join(old_lines))
                prompt = prompt.replace(CRITIC_GRANULARITY_GUIDANCE,
                    old_key if task.task_type == 'paper_matched_key_event_critic' else '')
            prompt = 'Do not browse, call tools, inspect files, or consult external sources. Solve entirely from the supplied chemical input. The tool budget is zero.\n' + prompt
            return real(replace(task, objective=prompt, budget=replace(task.budget, max_tool_calls=0)))
        return execute
    runner.node_executor = executor(runner.node_executor)
    runner.critic_executor = executor(runner.critic_executor)
    branch = {'branch_index': 0, 'steps': [], 'target_mapped_smiles': source['mapped'],
        'lens': 'Continue supplied synthetic horizon', 'strategy_card': card, 'root_strategy_card': card,
        'strategy_tree_engine': 'aizynthfinder_mcts', 'strategy_milestone_cards': [card],
        'strategy_milestone_generation_count': 0, 'key_event_critic_history': [],
        'route_call_count': 0, 'call_count': 0, 'generated_step_id_prefix': 'granularity-canary',
        'expanded_products': set(), 'open_leaf_states': deque()}
    boundary = source['expected_frontier_contains'][0]
    observations = []
    real_sidecar = d.run_aizynthfinder_strategy_branch_sidecar
    def bounded_sidecar(**kwargs):
        real_handler = kwargs['request_handler']
        def handle(request):
            steps = request.get('route_steps') or []
            if any(boundary in s.get('precursor_smiles', []) for s in steps):
                observations.append({'boundary': boundary, 'steps': steps})
                return {'candidates': [], 'model_call_consumed': False, 'stop_search': True,
                        'stop_reason': 'experimental_local_boundary_reached'}
            return real_handler(request)
        kwargs['request_handler'] = handle
        return real_sidecar(**kwargs)
    write(cell / 'config.json', {'config': asdict(config), 'quota': quota, 'source': source,
        'arm': args.arm, 'stop_boundary': boundary, 'boundary_is_stock_claim': False,
        'scope': 'Local natural-product fragment; no full-route success claim. Same real stock and budget.'})
    started = time.monotonic()
    with patch.object(d, 'run_aizynthfinder_strategy_branch_sidecar', bounded_sidecar):
        records = runner._expand_seeded_branches_aizynthfinder(spec, target=target, seeded=[branch], existing_records=[],
            route_quota=d._NodeCallBudget(**quota), critic_editor_call_reserve=1,
            critic_input_reserve=24000, critic_output_reserve=16000, config=config, started=started)
    runner._run_codex_critics(spec, helpers.campaign(spec, target, source), [branch], records,
        quota=d._NodeCallBudget(**quota), started=started, config=config)
    # Existing helpers write deque-bearing branches through their normal serializer.
    helpers.write(cell / 'result.json', {'branch': branch, 'boundary_observations': observations,
        'usage': d._aggregate_usage(records, elapsed_s=time.monotonic()-started), 'records': [asdict(r) for r in records]})
    print({'arm': args.arm, 'steps': len(branch.get('steps', [])), 'boundary_reached': bool(observations),
           'critic': (branch.get('chemical_critic') or {}).get('overall_assessment'),
           'usage': d._aggregate_usage(records, elapsed_s=time.monotonic()-started)}, flush=True)


if __name__ == '__main__':
    main()
