"""Compare observed worker usage without changing any run or budget ledger.

Usage: python scripts/report_worker_usage.py RUN_DIR ... --output-prefix PATH
Also accepts individual WorkerRunRecord JSON files from a bounded diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascade_planner.agent.worker_usage import search_policy_diagnostics


def rows_from_path(source: Path) -> list[dict]:
    if source.is_dir():
        candidates = [source / '.autoplanner/director-workspace/sequential-director-worker-records.jsonl', source / 'sequential-director-worker-records.jsonl']
        journal = next((p for p in candidates if p.is_file()), None)
        if journal is None:
            raise ValueError(f'Worker journal not found: {source}')
    else:
        journal = source
    if journal.suffix == '.jsonl':
        # JSONL records end at LF. Unicode paragraph/line separators are valid
        # inside JSON strings and must not split a completed worker record.
        raw_rows = [json.loads(line) for line in journal.read_text(encoding='utf-8').split('\n') if line.strip()]
    else:
        raw_rows = [json.loads(journal.read_text(encoding='utf-8'))]
    inputs = {}
    io_path = journal.with_name('model-io.jsonl')
    if io_path.exists():
        for line in io_path.read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('event') == 'model_input':
                inputs[item['task_id']] = item
    single_input = journal.with_name(journal.name.replace('-record.json', '-input.json'))
    if not inputs and single_input != journal and single_input.is_file():
        item = json.loads(single_input.read_text(encoding='utf-8'))
        inputs[item['task_id']] = dict(item, prompt=item.get('objective', ''))
    result = []
    seen = set()
    for raw in raw_rows:
        record = raw.get('record', raw)
        # Seed aliases can refer to exactly the same physical worker record.
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        task_id = record['task_id']
        inp = inputs.get(task_id, {})
        metadata = record.get('metadata') or {}
        diagnostic = metadata.get('usage_diagnostics') or {}
        usage = record.get('usage') or {}
        sizes = diagnostic.get('prompt_sizes') or {}
        requests = diagnostic.get('requests') or {}
        command = record.get('command') or []
        model = command[command.index('--model') + 1] if '--model' in command else metadata.get('model', inp.get('model', 'unknown'))
        tools = Counter(t.get('tool', 'unknown') for t in record.get('tool_calls') or [])
        search_policy = search_policy_diagnostics(
            command, record.get('tool_calls') or [],
            observation_complete=record.get('status') not in {'timeout', 'worker_error', 'provider_error', 'cancelled'},
        )
        def count(key):
            return int(usage[key]) if usage.get(key) is not None else None
        total, cached = count('input_tokens'), count('cached_input_tokens')
        observed = requests.get('telemetry_observed') is True
        result.append({
            'source': str(source), 'model': model, 'task_type': inp.get('task_type', metadata.get('task_type', 'unknown')),
            'reasoning_effort': metadata.get('model_reasoning_effort', inp.get('reasoning_effort', inp.get('budget', {}).get('reasoning_effort'))),
            'cli_versions': ','.join(requests.get('cli_versions') or []),
            'transport': metadata.get('transport'),
            'search_requested_mode': search_policy['requested_mode'],
            'search_policy_status': search_policy['status'],
            'observed_search_events': search_policy['observed_search_events'],
            'task_id': task_id, 'status': record.get('status'), 'elapsed_s': record.get('elapsed_s'),
            'objective_chars': sizes.get('objective', {}).get('characters', len(inp['prompt']) if 'prompt' in inp else None),
            'stdin_chars': sizes.get('stdin_prompt', {}).get('characters'),
            'schema_chars': sizes.get('output_schema', {}).get('characters'),
            'input_tokens': total, 'cached_input_tokens': cached,
            'uncached_input_tokens': total - cached if total is not None and cached is not None else None,
            'output_tokens': count('output_tokens'), 'reasoning_output_tokens': count('reasoning_output_tokens'),
            'tool_calls': sum(tools.values()), 'web_search_calls': tools.get('web_search', 0),
            'structure_checks': tools.get('inspect_mapped_smiles', 0),
            'api_attempts_observed': requests.get('observed_api_attempts') if observed else None,
            'response_usage_records': requests.get('observed_completed_responses') if observed else None,
            'first_response_input_tokens': requests.get('first_response_input_tokens'),
            'followup_input_tokens': requests.get('subsequent_response_input_tokens'),
            'response_usage_matches_turn': diagnostic.get('response_input_sum_matches_turn_usage'),
            'fallback_model_metadata': diagnostic.get('fallback_model_metadata', 'fallback model metadata' in record.get('stderr', '') or 'Defaulting to fallback metadata' in record.get('stderr', '')),
            'request_log_path': requests.get('event_log_path', ''),
        })
    return result


def build_report(sources: list[Path]) -> dict:
    calls = [row for source in sources for row in rows_from_path(source)]
    grouped = defaultdict(list)
    for row in calls:
        grouped[(row['model'], row['task_type'])].append(row)
    groups = []
    for (model, role), values in sorted(grouped.items()):
        known = [r for r in values if r['input_tokens'] is not None]
        group = dict(model=model, task_type=role, calls=len(values), input_usage_known=len(known),
                     median_input=median(r['input_tokens'] for r in known) if known else None,
                     search_policy_mismatch_calls=sum(r['search_policy_status'] == 'requested_disabled_but_observed' for r in values),
                     request_trace_known=sum(r['api_attempts_observed'] is not None for r in values))
        for key in ('input_tokens', 'cached_input_tokens', 'uncached_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'tool_calls', 'web_search_calls', 'structure_checks'):
            group[key] = sum(r[key] or 0 for r in values)
        groups.append(group)
    return dict(schema_version='worker_usage_comparison.v1', semantics={
        'missing_usage_is_null_not_zero': True,
        'group_totals_sum_known_values_only': True,
        'cached_input_is_part_of_input': True,
        'reasoning_is_part_of_output': True,
        'historical_missing_request_trace_is_not_reconstructed': True,
        'no_billing_or_scientific_authority': True,
    }, groups=groups, calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    data = build_report(args.sources)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    with prefix.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data['calls'][0]) if data['calls'] else ['task_id'])
        writer.writeheader()
        writer.writerows(data['calls'])
    lines = ['# Worker token 调用对照', '', '输入含缓存；输出含推理。空值表示未观测，历史缺失的请求级记录不补造。字符数不是 token 数。', '',
             '| 模型 | 任务 | 调用 | 输入已知 | 输入合计 | 缓存 | 输出 | 输入中位数 | 工具 | 请求追踪已知 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in data['groups']:
        lines.append('| ' + ' | '.join(str(g[k]) for k in ('model', 'task_type', 'calls', 'input_usage_known', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'median_input', 'tool_calls', 'request_trace_known')) + ' |')
    mismatch_count = sum(g['search_policy_mismatch_calls'] for g in data['groups'])
    lines += ['', f'请求关闭搜索但观察到搜索工具事件的调用：{mismatch_count}。这是配置与执行记录的差异，不代表检索成功或获得了有效化学证据。',
              '', '逐调用字段见同名 CSV／JSON：提示词、schema、输入／缓存／输出／推理、首个请求与后续输入、工具次数、搜索策略差异、重试、模型元数据回退及原始追踪文件路径。']
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(calls=len(data['calls']), output_prefix=str(prefix)), ensure_ascii=True))


if __name__ == '__main__':
    main()
