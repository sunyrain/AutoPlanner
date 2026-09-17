"""Frozen-prompt, equal-call granularity pilot through production Worker/Host/Critic.

This tests reaction-boundary behavior, not unbiased route discovery or global
optimality. Original runs and the running service are never mutated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.orchestration import sequential_strategy_director as d
from cascade_planner.orchestration.reaction_granularity import (
    BUILDER_GRANULARITY_GUIDANCE, CRITIC_GRANULARITY_GUIDANCE,
)
from cascade_planner.runtime import AgentSpec

OUTPUT = ROOT / 'results/discussion/reaction-granularity-20260907'
SOURCE = ROOT / 'results/discussion/synthex-mainpaper-comparison-20260906/blind-review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def record(raw):
    return WorkerRunRecord(**{f.name: raw[f.name] for f in fields(WorkerRunRecord) if f.name in raw})


def prepare():
    if (OUTPUT / 'inputs.json').exists():
        raise FileExistsError('Prepared inputs exist; use them unchanged.')
    natural = read(SOURCE / 'route-2849d706a911/packet.json')['steps']
    xanthate = natural[14]['product_smiles']
    alcohol = natural[19]['precursor_smiles'][0]
    cases = [
        ('aldol', 'CC(=O)CC(O)c1ccccc1',
         'Develop a directed aldol synthesis from acetone and benzaldehyde, preserving the beta-hydroxy ketone. Choose the next retro transformation within this horizon.',
         ['CC(C)=O', 'O=Cc1ccccc1'], 'Enolate preparation and protonating workup should not be mandatory frontier nodes.'),
        ('grignard', 'CC(O)c1ccccc1',
         'Develop the carbonyl-addition route using the phenyl nucleophile prepared from bromobenzene and acetaldehyde. Choose the next retro transformation within this horizon.',
         ['Brc1ccccc1', 'CC=O'], 'In situ reagent preparation must preserve carbon source and avoid simultaneous incompatible additions.'),
        ('amide', 'CCNC(C)=O',
         'Develop acyl substitution from acetic acid and ethylamine, with necessary activation. Choose the next retro transformation within this horizon.',
         ['CC(=O)O', 'CCN'], 'Activation/capture can be one amidation; no advanced precursor may silently become stock.'),
        ('deprotection', 'NCCc1ccccc1',
         'Obtain this free amine by removing a Boc group and delivering the neutral amine. Choose the next retro transformation within this horizon.',
         ['CC(C)(C)OC(=O)NCCc1ccccc1'], 'Routine neutralization should not require a standalone salt node.'),
        ('xanthate', xanthate,
         'Prepare the O-alkyl S-methyl xanthate from the corresponding secondary alcohol using carbon disulfide and methylation for a later radical deoxygenation. Choose the next retro transformation within this horizon; retain the ester and pre-existing stereocenters.',
         [alcohol], 'Real Traversiadiene route fragment: xanthate formation was three separate nodes. Added fragment sources must remain explicit.'),
        ('cascade', 'O=C1CCCCC1',
         'Develop a Diels-Alder cycloaddition of butadiene with an oxygenated dienophile followed by its routine hydrolysis to this cyclohexanone. Choose the next retro transformation within this horizon.',
         [], 'Exploratory multibond cascade/hydrolysis: coherence, unsaturation and atom-source accounting still require review.'),
        ('oxidation_olefination', 'C=Cc1ccccc1',
         'Develop a route from benzyl alcohol by oxidation to benzaldehyde followed by Wittig methylenation. These are the proposed operations, not an instruction to combine them. Choose the next retro transformation within this horizon.',
         ['O=Cc1ccccc1'], 'Oxidation and olefination must retain separate synthetic burden even if experimentally telescoped.'),
        ('deoxygenation', natural[12]['product_smiles'],
         'Develop Barton-McCombie deoxygenation through an O-alkyl S-methyl xanthate prepared from the corresponding secondary alcohol. Choose the next retro transformation within this horizon; retain the ester and pre-existing stereocenters.',
         [xanthate], 'Xanthate preparation is distinct from radical deoxygenation; a one-pot label cannot hide it.'),
    ]
    alternatives = {
        'strict': None,
        'procedure': 'One candidate is one executable synthetic procedure. Combine activation, reagent generation, sequential reagent additions, independent transformations and workup into one edge when a chemically compatible one-pot or telescoped procedure is plausible. State its stages and complete net graph edits; avoid fictional mechanistic nodes. Preserve atom sources, stereo and precursor-state compatibility.',
        'transformation': BUILDER_GRANULARITY_GUIDANCE,
    }
    rows = []
    for name, target, query, expected, criterion in cases:
        target = d._canonical_smiles(target)
        mapped = d._mapped_smiles(target)
        card = {'strategy_query': query, 'critical_assumption': 'The indicated synthetic transformation preserves all other required functionality and stereochemistry.', 'critic_checkpoint': query}
        base = d._node_prompt(target=target, branch_index=0, lens='', selected_product=target,
            selected_product_mapped=mapped, steps=[], open_leaves=[target], prior_rejections=[], repair=False,
            strategy_card=card, forbidden_strategy_cards=[], host_failure_feedback={}, paper_matched=True)
        for arm, replacement in alternatives.items():
            prompt = base
            if replacement:
                lines = prompt.split('\n')
                changed = 0
                for i, line in enumerate(lines):
                    if line.startswith('One candidate must represent one executable reaction.'):
                        lines[i] = replacement
                        changed += 1
                    elif line.startswith('Conditions describe the forward reaction environment'):
                        lines[i] = 'Conditions describe the forward reaction for the actual Host-replayed input and product. Encode all net covalent and stereochemical differences through ReactionJSON; describe internal activation, addition order and workup explicitly within a coherent condition hypothesis.'
                        changed += 1
                    elif line.startswith('Check functional-group compatibility within the replayed precursor set'):
                        lines[i] = 'Check functional-group compatibility in the actual precursor set and through every stated internal stage; do not silently assume an independently protected or redox-modified substrate.'
                        changed += 1
                if changed != 3:
                    raise ValueError('Baseline prompt changed; inspect rather than silently patching.')
                prompt = '\n'.join(lines)
            rows.append({'case': name, 'arm': arm, 'target': target, 'mapped': mapped, 'card': card,
                'expected_frontier_contains': [d._canonical_smiles(s) for s in expected], 'criterion': criterion,
                'prompt': prompt, 'prompt_sha256': digest(prompt)})
    write(OUTPUT / 'inputs.json', rows)
    write(OUTPUT / 'protocol.json', {
        'unit': 'Eight frozen local boundary probes, three policies, one Builder sample each; targeted not population representative.',
        'arms': alternatives, 'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
        'budget_per_cell': {'builder_calls': 1, 'timeout_s': 300, 'max_output_bytes': 16000, 'tool_calls': 0},
        'review': 'Fresh same-backbone independent chemical review; no policy label, desired boundary, Strategy or prior verdict. Same review rule and 300 s per valid output. Chemistry and semantic burden judged separately.',
        'success': 'Fewer artificial reactive-state boundaries, no hidden independent transformation, conserved identity/stereo and retained compatibility objections. Replay failures remain outcomes. Boundary match alone is not chemical success.',
        'repeats': 'No automatic retries for schema, chemistry or replay failure. New attempts must be separately named.',
        'selection': 'Prefer smallest policy satisfying both boundary efficiency and preserved chemical scrutiny; no global optimality or measured route-yield claim.',
    })
    print('Prepared 24 immutable local input cells', flush=True)


def builder(row):
    cell = OUTPUT / (row['case'] + '--' + row['arm'])
    if (cell / 'result.json').exists():
        return
    cell.mkdir(parents=True, exist_ok=True)
    spec = AgentSpec(run_id='boundary:' + cell.name, agent_id='builder', role='route_builder',
        objective='Propose the next reaction in the supplied local horizon.', idempotency_key=row['prompt_sha256'],
        context_hash=row['prompt_sha256'], metadata={'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
        'allowed_workdir': str(cell / 'worker'), 'durable_worker_journal': True})
    runner = d.SequentialStrategyDirectorRunner()
    runner._prepare_worker_record_journal(spec)
    task = d._node_task(spec, prompt=row['prompt'], branch_index=0, node_index=0, model='gpt-6-astra',
        reasoning_effort='medium', timeout_s=300, paper_matched=True, target_smiles=row['target'], selected_product=row['target'])
    task = replace(task, budget=replace(task.budget, max_tool_calls=0))
    print('START ' + cell.name, flush=True)
    raw = runner._run_journaled_worker(runner.node_executor, task)
    candidates, failures = d._reactionjson_candidates_from_record(raw, expected_product=row['target'],
        mapped_product_smiles=row['mapped'], require_reaction_operations=True, max_candidates=1)
    expansions = d._expansions_from_record(raw, expected_product=row['target'], mapped_product_smiles=row['mapped'],
        require_reaction_operations=True, single_step_only=True) if candidates else []
    steps = [d._step_row(e, step_id='boundary-step') for e in expansions or []]
    frontier = [s for step in steps for s in step['precursor_smiles']]
    result = {'case': row['case'], 'arm': row['arm'], 'status': raw.status, 'record': asdict(raw),
        'host_failures': failures, 'steps': steps, 'expected_frontier_contains': row['expected_frontier_contains'],
        'expected_boundary_reached': bool(row['expected_frontier_contains']) and all(s in frontier for s in row['expected_frontier_contains'])}
    write(cell / 'result.json', result)
    print(json.dumps({'cell': cell.name, 'status': raw.status, 'replayed': bool(steps), 'boundary': result['expected_boundary_reached'], 'frontier': frontier}), flush=True)


def review(row, *, recover=False):
    cell = OUTPUT / (row['case'] + '--' + row['arm'])
    if not (cell / 'result.json').exists():
        return
    result = read(cell / 'result.json')
    destination = cell / ('review-recovered.json' if recover else 'review.json')
    if not result['steps'] or destination.exists():
        return
    if recover and ((not (cell / 'review.json').exists()) or read(cell / 'review.json')['status'] == 'accepted_draft'):
        return
    steps = [{**s, 'step_id': f'anonymous-step-{i}'} for i, s in enumerate(result['steps'], 1)]
    # Production Critic with the same stage semantics in all arms; no Strategy or desired route.
    prompt = d._critic_prompt(target=row['target'], branch_index=0, strategy_card={}, steps=steps, paper_matched=True)
    prompt = prompt.replace(CRITIC_GRANULARITY_GUIDANCE + '\n', '')
    prompt = 'Do not browse, call tools, inspect files, or consult external sources. Solve entirely from the supplied chemical input. The tool budget is zero.\nNo original policy, Strategy, or prior judgments are available. Review only the supplied chemical transformations. ' + CRITIC_GRANULARITY_GUIDANCE + '\n' + prompt
    label = digest([row['target'], steps, 'runtime-recovery' if recover else 'initial'])[:16]
    spec = AgentSpec(run_id='anonymous:' + label, agent_id='independent-review', role='route_critic',
        objective='Independently assess the supplied transformation.', idempotency_key=digest(prompt),
        context_hash=digest(prompt), metadata={'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
        'allowed_workdir': str(OUTPUT / 'anonymous-review' / label), 'durable_worker_journal': True})
    runner = d.SequentialStrategyDirectorRunner()
    runner._prepare_worker_record_journal(spec)
    task = d._critic_task(spec, prompt=prompt, branch_index=0, iteration=0, timeout_s=300,
        paper_matched=True, target_smiles=row['target'], task_id_override=spec.agent_id, route_steps=steps)
    task = replace(task, budget=replace(task.budget, max_tool_calls=0))
    raw = runner._run_journaled_worker(runner.critic_executor, task)
    critique = d._critique_from_record(raw, route_steps=steps)
    write(destination, {'status': raw.status, 'critique': critique, 'record': asdict(raw), 'prompt': prompt,
        'runtime_recovery': recover, 'prior_chemistry_or_judgments_provided': False})
    print(json.dumps({'cell': cell.name, 'review_status': raw.status, 'verdict': critique.get('overall_assessment')}), flush=True)


def summarize():
    results = []
    for row in read(OUTPUT / 'inputs.json'):
        cell = OUTPUT / (row['case'] + '--' + row['arm'])
        if not (cell / 'result.json').exists():
            continue
        r = read(cell / 'result.json')
        review_result = read(cell / 'review.json') if (cell / 'review.json').exists() else {}
        recovered = read(cell / 'review-recovered.json') if (cell / 'review-recovered.json').exists() else {}
        if recovered.get('status') == 'accepted_draft':
            review_result = recovered
        results.append({k: r[k] for k in ('case', 'arm', 'status', 'host_failures', 'steps', 'expected_boundary_reached')}
            | {'usage': r['record'].get('usage'), 'critique': review_result.get('critique'), 'review_usage': review_result.get('record', {}).get('usage')})
    write(OUTPUT / 'summary.json', results)
    print(json.dumps([{k: r[k] for k in ('case', 'arm', 'status', 'expected_boundary_reached')} | {'review': (r['critique'] or {}).get('overall_assessment')} for r in results], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'build', 'review', 'recover-review', 'summarize'])
    parser.add_argument('--case')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    global OUTPUT
    if args.output:
        OUTPUT = args.output.resolve()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'summarize':
        summarize()
    else:
        rows = [r for r in read(OUTPUT / 'inputs.json')
                if (not args.case or r['case'] == args.case) and (not args.arm or r['arm'] == args.arm)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            if args.action == 'recover-review':
                list(pool.map(lambda row: review(row, recover=True), rows))
            else:
                list(pool.map(builder if args.action == 'build' else review, rows))


if __name__ == '__main__':
    main()
