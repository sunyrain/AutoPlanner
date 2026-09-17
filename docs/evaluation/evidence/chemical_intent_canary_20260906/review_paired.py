"""Build blind packets and independently review completed paired pilot routes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from paired import read, write, digest, director, AgentSpec, journal
from cascade_planner.application.condition_predictions import _OPERATIONAL_FIELDS


def route_packet(result, target):
    branch = result.get('branch') or {}
    pending = branch.get('_pending_path_repair_transaction') or {}
    # A provider interruption may retain a provisional candidate for resume.
    # Its snapshot is still the authoritative route; it must not silently
    # become the scored output merely because branch.steps holds the work view.
    steps = ((pending.get('route_snapshot') or {}).get('steps')
             if pending else branch.get('steps')) or []
    # Whitelist chemical information. Original Critic, Strategy, transaction
    # status, task names and provenance are not independent review inputs.
    chemical_keys = {
        'product_smiles', 'mapped_product_smiles', 'precursor_smiles', 'mapped_precursor_smiles',
        'reaction_input_smiles', 'mapped_reaction_input_smiles', 'auxiliary_reagent_smiles',
        'mapped_auxiliary_reagent_smiles', 'reaction_operations', 'reaction_family',
    }
    rows = []
    for i, step in enumerate(steps, 1):
        row = {k: step[k] for k in chemical_keys if k in step}
        row['step_id'] = f'step-{i}'
        # Condition candidates contain chemistry plus provider metadata. Keep
        # chemistry and hypothesis wording, never source/worker identifiers.
        predictions = []
        for p in step.get('condition_predictions') or []:
            predictions.append({k: v for k, v in p.items()
                                if k in _OPERATIONAL_FIELDS | {'enzyme', 'structural_reagent_smiles'}})
        if predictions:
            row['condition_predictions'] = predictions
        rows.append(row)
    return {'target_smiles': target, 'steps': rows}


def prepare(output):
    source = read(output / 'inputs.json')
    grouped = {}
    for mechanism in ('continuous', 'checkpoint', 'dependency'):
        for arm in ('off', 'on'):
            cell = output / f'{mechanism}-{arm}'
            if not (cell / 'result.json').exists():
                continue
            result = read(cell / 'result.json')
            target = (source[mechanism]['route_context']['target_smiles'] if mechanism == 'dependency'
                      else source[mechanism]['target'])
            packet = route_packet(result, target)
            key = digest(packet)
            grouped.setdefault(key, {'packet': packet, 'cells': []})['cells'].append(cell.name)
    mapping = []
    for key, item in sorted(grouped.items()):
        label = 'route-' + key[:12]
        packet = item['packet']
        write(output / 'blind-review' / label / 'packet.json', packet)
        mapping.append({'packet_id': label, 'packet_sha256': key, 'cells': item['cells'], 'step_count': len(packet['steps'])})
        lines = [f'匿名路线 {label}', '', f'目标 SMILES：`{packet["target_smiles"]}`', '',
                 '请独立检查结构连续性、逐步反应合理性、立体控制、条件相容性和关键底物适用性。缺少文献支持不等同于已证明不可行。', '',
                 '这是计算提出的路线，未提供实验验证；请标注明确错误、重要不确定性及其具体位置。', '']
        for row in packet['steps']:
            lines += [row['step_id'] + ': ' + row.get('reaction_family', ''), '',
                      '产物：`' + row.get('mapped_product_smiles', row.get('product_smiles', '')) + '`', '',
                      '前体：`' + ' + '.join(row.get('mapped_precursor_smiles') or row.get('precursor_smiles') or []) + '`', '',
                      '条件假设：' + json.dumps(row.get('condition_predictions') or [], ensure_ascii=False), '']
        (output / 'blind-review' / label / 'expert-packet.md').write_text('\n'.join(lines), encoding='utf-8')
    write(output / 'review-map.json', mapping)
    print(json.dumps(mapping, ensure_ascii=False), flush=True)


def execute(output, packet_id):
    folder = output / 'blind-review' / packet_id
    packet = read(folder / 'packet.json')
    if (folder / 'result.json').exists() or (folder / 'worker/model-io.jsonl').exists():
        raise FileExistsError('This blind review already started; preserve its IO.')
    if not packet['steps']:
        write(folder / 'result.json', {'status': 'no_route_to_review'})
        return
    context = director.RevisionBoundRouteCriticContext(
        target_smiles=packet['target_smiles'], route_family_id=packet_id,
        route_sha256=digest(packet), graph_revision=0, branch_index=0,
        edge_ids=tuple(row['step_id'] for row in packet['steps']), steps=tuple(packet['steps']),
        strategy_card={}, selected_strategy_lineage=(), strategy_milestone_cards=(),
    )
    config = director.DirectorConfig(max_node_prompt_bytes=96000, critic_call_timeout_s=300,
                                    max_output_tokens=8000, model='gpt-6-astra', reasoning_effort='medium')
    runner = director.SequentialStrategyDirectorRunner()
    prompt = runner.final_route_critic_prompt_for(context, config)
    if not prompt:
        raise ValueError('Blind packet exceeds the shared review prompt bound')
    (folder / 'prompt.txt').write_text(prompt, encoding='utf-8')
    # Fresh task/workspace; only opaque IDs and chemical content enter the model.
    spec = AgentSpec(run_id=packet_id, agent_id='independent-reviewer', role='route_critic',
                     objective='Independently assess the supplied chemical route.', idempotency_key=digest(packet),
                     context_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                     metadata={'model': config.model, 'reasoning_effort': config.reasoning_effort,
                               'durable_worker_journal': True, 'allowed_workdir': str(folder / 'worker')})
    critique, record = runner.run_final_route_critic_once(spec, context=context, config=config, prompt=prompt)
    write(folder / 'result.json', {'status': record.status, 'critique': critique, 'record': asdict(record)})
    print(json.dumps({'packet': packet_id, 'status': record.status, 'verdict': critique.get('overall_assessment')}, ensure_ascii=False), flush=True)


def summarize(output):
    mapping = read(output / 'review-map.json')
    reviews = {}
    for entry in mapping:
        path = output / 'blind-review' / entry['packet_id'] / 'result.json'
        if not path.exists():
            # A single judgment per identical blind chemical input avoids
            # inventing differences by re-reviewing the same retained route.
            # Resolve an existing result directly, without a second cache copy.
            candidates = sorted(output.parent.glob('mechanism-pilot*/blind-review/' + entry['packet_id'] + '/result.json'))
            path = next((p for p in candidates
                         if digest(read(p.with_name('packet.json'))) == entry['packet_sha256']), path)
        if path.exists():
            for cell in entry['cells']:
                reviews[cell] = {**read(path), 'review_source': str(path)}
    rows = []
    for cell in sorted(output.glob('*-*/result.json')):
        if cell.parent.name not in {f'{m}-{a}' for m in ('continuous', 'checkpoint', 'dependency') for a in ('on', 'off')}:
            continue
        result = read(cell)
        branch = result.get('branch') or {}
        review = (reviews.get(cell.parent.name) or {}).get('critique') or {}
        worker_path = cell.parent / 'worker/sequential-director-worker-records.jsonl'
        records = [r['record'] for r in journal(worker_path)] if worker_path.exists() else []
        io_path = cell.parent / 'worker/model-io.jsonl'
        input_roles = {r['task_id']: r.get('task_type') for r in journal(io_path)
                       if r.get('event') == 'model_input'} if io_path.exists() else {}
        by_role = {}
        fixed_slots = 0
        for record in records:
            if (record.get('metadata') or {}).get('fixed_shared_intervention'):
                fixed_slots += 1
                continue
            role = ((record.get('output_artifact') or {}).get('summary')
                    or (record.get('metadata') or {}).get('task_type')
                    or input_roles.get(record['task_id']) or 'unknown')
            totals = by_role.setdefault(role, {'calls': 0, 'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0})
            totals['calls'] += 1
            for key in ('input_tokens', 'cached_input_tokens', 'output_tokens'):
                totals[key] += (record.get('usage') or {}).get(key, 0)
        rows.append({'cell': cell.parent.name, 'status': result['status'], 'reason': result.get('reason'),
                     'steps': len(((branch.get('_pending_path_repair_transaction') or {}).get('route_snapshot') or {}).get('steps')
                                  or branch.get('steps') or []),
                     'pending_provisional_steps': len(branch.get('steps') or []) if branch.get('_pending_path_repair_transaction') else 0,
                     'stock_closed': branch.get('complete_in_bound_stock') if not branch.get('_pending_path_repair_transaction') else None,
                     'usage': result.get('usage'), 'strategy_calls': branch.get('strategy_call_count', 0),
                     'new_worker_attempts': sum(r['calls'] for r in by_role.values()),
                     'fixed_shared_policy_slots': fixed_slots, 'usage_by_role': by_role,
                     'calls_with_unreported_usage': [r['task_id'] for r in records
                         if not r.get('usage') and not (r.get('metadata') or {}).get('fixed_shared_intervention')],
                     'key_critic_calls': branch.get('key_event_critic_call_count', 0),
                     'editor_attempts': branch.get('editor_attempt_count', 0),
                     'transactions': [{k: t.get(k) for k in ('status', 'reason', 'requested_change_step_ids', 'recovery_request', 'builder_calls')}
                                      for t in branch.get('path_repair_transactions') or []],
                     'independent_verdict': review.get('overall_assessment', 'pending'),
                     'independent_review_source': (reviews.get(cell.parent.name) or {}).get('review_source'),
                     'independent_step_verdicts': [{k: s.get(k) for k in ('step_id', 'verdict', 'reasons')}
                                                 for s in review.get('step_assessments') or []],
                     'independent_route_evaluation': review.get('route_overall_evaluation')})
    write(output / 'summary.json', rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--packet')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output.resolve())
    elif args.packet:
        execute(args.output.resolve(), args.packet)
    elif args.summarize:
        summarize(args.output.resolve())
    else:
        parser.error('Choose --prepare, --packet, or --summarize')


if __name__ == '__main__':
    main()
