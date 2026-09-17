"""Same-target full-route comparison, reusing the existing anonymous review/export path."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
path = ROOT / 'docs/evaluation/evidence/synthex_mainpaper_review_20260906/compare_routes.py'
spec = importlib.util.spec_from_file_location('common_route_review', path)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

OLD = ROOT / 'results/discussion/synthex-mainpaper-comparison-20260906'
NEW = ROOT / 'results/.autoplanner/traversiadiene-granularity-astra-medium25-20260907'
OUTPUT = ROOT / 'results/discussion/traversiadiene-granularity-comparison-20260907'
CASE_ID = 'mainpaper-20260906-02'
c.OUTPUT = OUTPUT
c.RUBRIC = ('Do not browse, call tools, inspect files, or consult external sources. The tool budget is zero.\n' + c.RUBRIC).replace(
    'AnonymousChemicalRouteInput:\n',
    'Assess the complete supplied route, including internal activation, addition order, quench/workup and inter-step compatibility. '
    'A coherent synthetic transformation can contain necessary internal stages; routine activation or neutralization need not be a separate route node. '
    'Independently useful preparative transformations such as oxidation then olefination, or radical-precursor preparation then reduction, retain their separate synthetic burden even when telescoped. '
    'Discuss artificial splitting, bundled independent transformations, protection/redox detours and unresolved precursor complexity in the route evaluation or revision advice. '
    'Counting conventions or inefficiency alone are not a chemical reject. Do not reward a short route for leaving an advanced precursor unresolved. '
    'No atom mapping or edit program is supplied; this unavailable evidence is not a defect.\nAnonymousChemicalRouteInput:\n')


def prepare():
    if (OUTPUT / 'review-map.json').exists():
        raise FileExistsError('Prepared comparison exists; preserve the original packets.')
    target = next(r for r in c.read(OLD / 'target-map.json') if r['case_id'] == CASE_ID)
    c.write(OUTPUT / 'target-map.json', [target])
    entries = []
    for row in c.read(OLD / 'review-map.json'):
        if row['case_id'] != CASE_ID or not row.get('packet_id'):
            continue
        body = c.read(OLD / 'blind-review' / row['packet_id'] / 'packet.json')
        entry = c.save_packet(body, {**row, 'system': 'AutoPlanner_old' if row['system'] == 'AutoPlanner' else row['system'],
            'old_review_not_supplied': True, 'prior_packet_path': str(OLD / 'blind-review' / row['packet_id'] / 'packet.json')})
        entries.append(entry)
    c.update_map(entries)


def collect():
    state = c.read(NEW / 'panel-status.json')
    outcome = state['targets']['opaque target 002']
    if outcome.get('status') in ('running', 'queued'):
        raise RuntimeError('Planning remains active; do not label an interim graph final.')
    with sqlite3.connect((NEW / 'runtime/run_index.sqlite3').as_uri() + '?mode=ro', uri=True) as con:
        row = con.execute("SELECT ref_json FROM artifacts WHERE run_id=? AND artifact_id='canonical_hypergraph' ORDER BY revision DESC LIMIT 1", (CASE_ID,)).fetchone()
    if not row:
        c.write(OUTPUT / 'new-run.json', {'status': 'no_final_graph', 'outcome': outcome})
        return
    ref = json.loads(row[0])
    graph = NEW / 'artifacts' / ref['object_path']
    c.prepare_graph(graph, CASE_ID)
    entries = c.read(OUTPUT / 'review-map.json')
    for entry in entries:
        if entry['system'] == 'AutoPlanner':
            entry['system'] = 'AutoPlanner_new'
    c.write(OUTPUT / 'review-map.json', entries)
    c.write(OUTPUT / 'new-run.json', {'registry': str(NEW), 'canonical_graph': str(graph),
        'canonical_graph_ref': ref, 'outcome': outcome})


def status():
    workspace = NEW / 'runs/opaque target 002/.autoplanner/director-workspace'
    records = [json.loads(line)['record'] for line in
               (workspace / 'sequential-director-worker-records.jsonl').read_text(encoding='utf-8').split('\n')
               if line.strip()]
    events = [json.loads(line) for line in (workspace / 'model-io.jsonl').read_text(encoding='utf-8').split('\n')
              if line.strip()]
    finished = {r['task_id'] for r in events if r['event'] == 'model_output'}
    input_roles = {r['task_id']: r.get('task_type', 'unknown') for r in events if r['event'] == 'model_input'}
    finished.update(r['task_id'] for r in records)
    recent = []
    for record in records[-4:]:
        payload = (record.get('output_artifact') or {}).get('payload') or {}
        candidates = payload.get('candidates') or []
        recent.append({'task': record['task_id'], 'status': record['status'],
                       'verdict': payload.get('overall_assessment'),
                       'intent': candidates[0].get('reaction_family') if candidates else None})
    print(json.dumps({'panel': c.read(NEW / 'panel-status.json')['targets']['opaque target 002'],
        'calls': len(records), 'roles': dict(Counter(input_roles.get(r['task_id'], 'unknown') for r in records)),
        'usage': {k: sum((r.get('usage') or {}).get(k) or 0 for r in records)
                  for k in ('input_tokens', 'cached_input_tokens', 'output_tokens')},
        'last_event': next((r['timestamp'] for r in reversed(events) if r.get('timestamp')), None),
        'pending': [{k: r.get(k) for k in ('task_id', 'task_type', 'timestamp')}
                    for r in events if r['event'] == 'model_input' and r['task_id'] not in finished],
        'recent': recent}, ensure_ascii=False, indent=2), flush=True)


def export():
    from cascade_planner.web.v4_showcase_export import build_run_export_bundle, render_run_export_bundle_html
    state = c.read(NEW / 'panel-status.json')['targets']['opaque target 002']
    if state.get('status') in ('running', 'queued'):
        raise RuntimeError('Wait for final planning state before exporting the comparison.')
    old_registry = ROOT / 'results/.autoplanner/synthex-mainpaper-fig1-astra-medium25-known-target-20260906'
    receipts = []
    for label, registry in [('old', old_registry), ('new', NEW)]:
        outcome = c.read(registry / 'panel-status.json')['targets']['opaque target 002']
        disposition = (outcome.get('final_state') or {}).get('current_disposition') or {}
        bundle = build_run_export_bundle(run_dir=registry / 'runs/opaque target 002',
            job={'run_id': CASE_ID, 'target_name': 'Traversiadiene / ' + label,
                 'status': disposition.get('state') or outcome.get('status', 'completed')},
            export_kind='interaction', branch_indices=[1, 2, 3])
        output = OUTPUT / ('Traversiadiene-' + label + '-replay.html')
        output.write_text(render_run_export_bundle_html(bundle), encoding='utf-8')
        receipts.append({'path': str(output), 'bytes': output.stat().st_size,
                         'metadata': bundle['metadata'], 'summary': bundle['summary'],
                         'branch_summaries': [{k: v for k, v in row.items() if k != 'steps'}
                                              for row in bundle['projection']['branches']]})
    c.write(OUTPUT / 'native-export-receipts.json', receipts)
    print(json.dumps([{'path': r['path'], 'metadata': r['metadata']} for r in receipts], ensure_ascii=False), flush=True)


def diagnostics():
    workspace = NEW / 'runs/opaque target 002/.autoplanner/director-workspace'
    events = [json.loads(line) for line in (workspace / 'model-io.jsonl').read_text(encoding='utf-8').split('\n')
              if line.strip()]
    records = [json.loads(line)['record'] for line in
               (workspace / 'sequential-director-worker-records.jsonl').read_text(encoding='utf-8').split('\n')
               if line.strip()]
    builders = [r for r in events if r['event'] == 'model_input' and r['task_type'] == 'paper_matched_route_step']
    feedback = []
    for row in builders:
        context = json.loads(row['prompt'].rsplit('\n', 1)[-1])
        if context.get('last_rejection_for_this_leaf'):
            feedback.append({'recipient_task_id': row['task_id'],
                             'feedback': context['last_rejection_for_this_leaf']})
    data = {'builder_inputs': len(builders),
            'new_granularity_inputs': sum('Use one chemically meaningful synthetic transformation' in r['prompt'] for r in builders),
            'old_strict_splitting_inputs': sum('an independent protection/deprotection, activation, redox change, workup transformation' in r['prompt'] for r in builders),
            'worker_status_counts': dict(Counter(r['status'] for r in records)),
            'unsuccessful_workers': [{'task_id': r['task_id'], 'status': r['status'],
                                     'elapsed_s': r.get('elapsed_s'), 'usage': r.get('usage')}
                                    for r in records if r['status'] != 'accepted_draft'],
            'builder_feedback': feedback,
            'semantics': 'Feedback is observed in the subsequent Builder input; repeated feedback is not automatically a distinct failed reaction. Missing usage is not zero cost.'}
    c.write(OUTPUT / 'planning-diagnostics.json', data)
    print(json.dumps({k: v for k, v in data.items() if k != 'builder_feedback'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'collect', 'review', 'summarize', 'render', 'status', 'export', 'diagnostics'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'collect':
        collect()
    elif args.action == 'review':
        labels = sorted({r['packet_id'] for r in c.read(OUTPUT / 'review-map.json') if r.get('packet_id')})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(c.review, labels))
        c.summarize()
    elif args.action == 'render':
        c.render_packets()
    elif args.action == 'status':
        status()
    elif args.action == 'export':
        export()
    elif args.action == 'diagnostics':
        diagnostics()
    else:
        c.summarize()


if __name__ == '__main__':
    main()
