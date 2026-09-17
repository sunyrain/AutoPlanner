"""Common, source-hidden chemical review of published and newly planned routes.

This experiment adapter does not edit planning runs or infer public edit programs.
The sole review representation is unmapped isomeric SMILES plus supplied conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from cascade_planner.application.condition_predictions import _OPERATIONAL_FIELDS
from cascade_planner.application.reaction_inputs import reaction_input_smiles
from cascade_planner.application.route_review_context import compile_revision_bound_route_critic_context
from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner, _critic_task, _critique_from_record,
)
from cascade_planner.runtime import AgentSpec

OUTPUT = ROOT / 'results/discussion/synthex-mainpaper-comparison-20260906'
STOCK = ROOT / 'data_external/synthatlas/zinc_synthelite_20260223_full_inchikey.sqlite3'
RUBRIC = """Act as an independent senior synthetic chemist. Review only the supplied chemical route.
Do not browse, use tools, inspect files, identify the source system, or consult prior judgments.
The supplied product and precursor structures are unmapped isomeric SMILES. No atom maps or graph-edit programs are supplied. Their absence is unavailable evidence, never a defect. SMILES @/@@ character order alone cannot establish retention or inversion.
Steps are stored in target-rooted retrosynthetic order. Evaluate the forward chemistry from terminal precursors through their consuming reactions toward the target, preserving the supplied review slots. Repeated identical intermediates can represent separate convergent branches.
Assess mechanism, net structural/redox plausibility, complementary reactive handles, functional-group compatibility, site selectivity, stereochemical control, and dependencies between steps. Product stereochemistry specifies the intended outcome; it does not prove its selective formation. Do not invent a required configuration where the supplied target leaves one unspecified. Routine non-structural reagents and coproducts may be omitted.
Conditions are preserved as supplied descriptions or candidate hypotheses. Multiple descriptions of one step are not necessarily successive operations or jointly applied conditions. If they disagree, explain the actual conflict; do not silently discard a viable stated catalyst or condition. No reaction-family label establishes that the graph transformation is chemically plausible.
For every supplied review_slot exactly once: pass means the structures, mechanism and stated control factors coherently support the product; uncertain means plausible but unresolved scope, selectivity or condition sufficiency without a concrete contradiction; reject requires a specific structural, mechanistic, compatibility, selectivity, condition or sequence contradiction. Missing literature is not proof of failure. Incomplete synthesis of a terminal precursor is scored separately and must not by itself cause a step rejection. A chemical pass is not experimental validation.
Check whether an upstream transformation actually establishes the reactive state and stereochemistry required by its downstream consumer. Distinguish a plausible unverified reaction from a concrete incompatibility. Mention coordinated changes only when needed to resolve a specific coupled defect. Do not improve a route by truncating steps or asserting a complex terminal intermediate is purchasable.
No strategic instructions or original critique are supplied. Set strategy_adherence=false as unavailable observation metadata, not a chemical objection. Mapped-atom fields must be empty because no maps are supplied. Use review slots to locate dependencies and coupled blocker groups where the schema permits them.
Return the compact schema-defined critique. Give at most two concrete reasons per step. In route_overall_evaluation give a concise whole-route judgment, strongest feature, decisive remaining risk or blocker, and experimental maturity. Do not infer a reference route or numerical success probability.
AnonymousChemicalRouteInput:
"""


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid supplied structure: ' + smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def conditions(step):
    values = []
    for key in ('conditions', 'conditions_as_given'):
        if step.get(key):
            value = step[key]
            values.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True))
    for item in step.get('condition_predictions') or []:
        kept = {k: v for k, v in item.items()
                if k in _OPERATIONAL_FIELDS | {'enzyme', 'structural_reagent_smiles'}}
        if kept:
            values.append(json.dumps(kept, ensure_ascii=False, sort_keys=True))
    # Route precursors and auxiliary reaction participants have distinct roles.
    # Preserve explicitly supplied auxiliary structures in the condition context;
    # do not turn their omission from the search frontier into missing chemistry.
    complete = Counter(canonical(x) for x in reaction_input_smiles(step))
    frontier = Counter(canonical(x) for x in step.get('precursor_smiles') or [])
    auxiliary = sorted((complete - frontier).elements())
    if auxiliary:
        values.append('Additional supplied auxiliary reaction inputs (SMILES): ' + ' + '.join(auxiliary))
    return list(dict.fromkeys(values))


def packet(target, steps):
    rows = []
    for i, step in enumerate(steps, 1):
        if 'rxn_smiles' in step:
            lhs, rhs = step['rxn_smiles'].split('>>')
            precursors, product = lhs.split('.'), rhs
        else:
            product = step.get('product_smiles') or step['mapped_product_smiles']
            precursors = step.get('precursor_smiles') or step.get('mapped_precursor_smiles') or []
        rows.append({
            'review_slot': f'review-{i:03d}',
            'product_smiles': canonical(product),
            'precursor_smiles': sorted(canonical(x) for x in precursors),
            'reaction_family': step.get('class') or step.get('reaction_family') or '',
            'condition_hypotheses': conditions(step),
        })
    return {'target_smiles': canonical(target), 'steps': rows}


def structural_summary(body):
    """Chemical identity topology and exact frozen-stock membership, no LLM."""
    rows = body['steps']
    products = {s['product_smiles'] for s in rows}
    precursors = {p for s in rows for p in s['precursor_smiles']}
    terminals = sorted(precursors - products)
    reachable, visiting = set(), set()
    cycle = False

    def walk(product):
        nonlocal cycle
        if product in visiting:
            cycle = True
            return 0
        visiting.add(product)
        depths = []
        for i, row in enumerate(rows):
            if row['product_smiles'] == product:
                reachable.add(i)
                depths.append(1 + max((walk(p) for p in row['precursor_smiles']), default=0))
        visiting.remove(product)
        return max(depths, default=0)

    lls = walk(body['target_smiles'])
    leaf_rows = []
    with sqlite3.connect(STOCK.as_uri() + '?mode=ro', uri=True) as con:
        for smiles in terminals:
            mol = Chem.MolFromSmiles(smiles)
            key = Chem.MolToInchiKey(mol)
            member = con.execute('SELECT 1 FROM stock WHERE full_inchikey=?', (key,)).fetchone() is not None
            leaf_rows.append({'smiles': smiles, 'full_inchikey': key, 'heavy_atoms': mol.GetNumHeavyAtoms(),
                             'in_frozen_stock': member})
    return {
        'step_count': len(rows), 'longest_linear_path_by_exact_identity': lls,
        'distinct_structure_transformations': len({
            (s['product_smiles'], tuple(s['precursor_smiles'])) for s in rows
        }),
        'target_has_producer': body['target_smiles'] in products,
        'all_steps_reachable_from_target': len(reachable) == len(rows),
        'unreachable_review_slots': [s['review_slot'] for i, s in enumerate(rows) if i not in reachable],
        'cycle_detected': cycle,
        'duplicate_product_structures': sorted(p for p in products if sum(s['product_smiles'] == p for s in rows) > 1),
        'terminal_precursors': leaf_rows,
        'stock_closed': bool(rows) and body['target_smiles'] in products and not cycle
                        and len(reachable) == len(rows) and bool(leaf_rows)
                        and all(r['in_frozen_stock'] for r in leaf_rows),
        'stock_note': 'Exact full-InChIKey membership in the shared frozen index; not procurement availability.',
    }


def save_packet(body, source):
    key = digest(body)
    label = 'route-' + key[:12]
    folder = OUTPUT / 'blind-review' / label
    path = folder / 'packet.json'
    if path.exists() and read(path) != body:
        raise ValueError('Packet identity collision')
    write(path, body)
    write(folder / 'structure-summary.json', structural_summary(body))
    lines = [f'# 匿名路线 {label}', '', '目标：`' + body['target_smiles'] + '`', '',
             '独立评估所列反应的合理性、立体控制及跨步相容性。区分明确矛盾与未验证风险；原料闭合另行评分。', '']
    for row in body['steps']:
        lines += [f"## {row['review_slot']} — {row['reaction_family']}", '',
                  '产物：`' + row['product_smiles'] + '`', '',
                  '前体：`' + ' + '.join(row['precursor_smiles']) + '`', '',
                  '所提供条件：' + json.dumps(row['condition_hypotheses'], ensure_ascii=False), '']
    (folder / 'expert-packet.md').write_text('\n'.join(lines), encoding='utf-8')
    return {**source, 'packet_id': label, 'packet_sha256': key, 'step_count': len(body['steps'])}


def update_map(entries):
    path = OUTPUT / 'review-map.json'
    mapping = read(path) if path.exists() else []
    sources = {r['source_key'] for r in entries}
    mapping = [r for r in mapping if r['source_key'] not in sources] + entries
    write(path, mapping)
    print(json.dumps([{k: r.get(k) for k in ('system', 'public_target_name', 'branch_index',
                                            'packet_id', 'step_count', 'status')}
                      for r in entries], ensure_ascii=False), flush=True)


def prepare_public():
    entries = []
    for target in read(OUTPUT / 'target-map.json'):
        source = read(target['source_path'])
        entries.append(save_packet(packet(source['target_smiles'], source['steps']), {
            **target, 'system': 'published_SynthEx', 'source_key': source['id'],
            'source_sha256': hashlib.sha256(Path(target['source_path']).read_bytes()).hexdigest(),
        }))
    update_map(entries)


def prepare_graph(graph_path, case_id):
    graph = read(graph_path)
    if 'graph' in graph and 'route_families' not in graph:
        graph = graph['graph']
    target = next(x for x in read(OUTPUT / 'target-map.json') if x['case_id'] == case_id)
    entries = []
    for family_id, family in graph.get('route_families', {}).items():
        if family.get('selected') is False:
            continue
        context, diagnostic = compile_revision_bound_route_critic_context(graph, route_family_id=family_id)
        if context is None:
            entries.append({**target, 'system': 'AutoPlanner', 'source_key': str(graph_path) + ':' + family_id,
                            'status': 'unreviewable_canonical_context', 'diagnostic': diagnostic})
            continue
        entries.append(save_packet(packet(context.target_smiles, context.steps), {
            **target, 'system': 'AutoPlanner', 'source_key': str(graph_path) + ':' + family_id,
            'source_path': str(graph_path), 'source_sha256': hashlib.sha256(Path(graph_path).read_bytes()).hexdigest(),
            'route_family_id': family_id, 'branch_index': context.branch_index,
            'graph_revision': context.graph_revision,
        }))
    if not entries:
        entries.append({**target, 'system': 'AutoPlanner', 'source_key': str(graph_path), 'status': 'no_selected_route'})
    update_map(entries)


def collect_finished():
    """Resolve completed canonical graphs through their owning run index."""
    run_rows = []
    for target in read(OUTPUT / 'target-map.json'):
        figure = 'fig5' if target['case_id'].endswith('-04') else 'fig1'
        registry = ROOT / f'results/.autoplanner/synthex-mainpaper-{figure}-astra-medium25-known-target-20260906'
        state = read(registry / 'panel-status.json')
        outcome = state['targets'][target['target_name']]
        row = {'case_id': target['case_id'], 'public_target_name': target['public_target_name'],
               'registry_root': str(registry), 'run_dir': str(registry / 'runs' / target['target_name']),
               'outcome': outcome}
        run_rows.append(row)
        if outcome.get('status') in {'running', 'queued'}:
            continue
        index = registry / 'runtime/run_index.sqlite3'
        with sqlite3.connect(index.as_uri() + '?mode=ro', uri=True) as con:
            found = con.execute("SELECT ref_json FROM artifacts WHERE run_id=? AND artifact_id='canonical_hypergraph' ORDER BY revision DESC LIMIT 1",
                                (target['case_id'],)).fetchone()
        if found:
            ref = json.loads(found[0])
            graph_path = registry / 'artifacts' / ref['object_path']
            row['canonical_graph_ref'] = ref
            prepare_graph(graph_path, target['case_id'])
        else:
            update_map([{**target, 'system': 'AutoPlanner', 'source_key': 'missing-final-graph:' + target['case_id'],
                         'status': 'no_final_canonical_graph', 'run_outcome': outcome}])
    write(OUTPUT / 'run-results.json', run_rows)


def review(label):
    folder = OUTPUT / 'blind-review' / label
    if (folder / 'result.json').exists():
        return {'packet': label, 'status': 'existing_result_preserved'}
    body = read(folder / 'packet.json')
    if not body['steps']:
        write(folder / 'result.json', {'status': 'no_route_to_review'})
        return {'packet': label, 'status': 'no_route_to_review'}
    prompt = RUBRIC + json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    prompt_path = folder / 'prompt.txt'
    if prompt_path.exists() and prompt_path.read_text(encoding='utf-8') != prompt:
        raise ValueError('Do not change a previously started review prompt')
    prompt_path.write_text(prompt, encoding='utf-8')
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    workspace = folder / 'worker'
    workspace.mkdir(exist_ok=True)
    spec = AgentSpec(run_id=label, agent_id='independent-chemical-review:' + sha[:16], role='route_critic',
                     objective='Independently review the supplied chemical route.', idempotency_key=sha,
                     context_hash=sha, metadata={'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
                     'allowed_workdir': str(workspace), 'durable_worker_journal': True})
    bindings = [{'step_id': row['review_slot'], 'reaction_operations': []} for row in body['steps']]
    runner = SequentialStrategyDirectorRunner()
    runner._prepare_worker_record_journal(spec)
    task = _critic_task(spec, prompt=prompt, branch_index=0, iteration=0, timeout_s=600,
                        paper_matched=True, target_smiles=body['target_smiles'],
                        task_id_override=spec.agent_id, route_steps=bindings)
    task = replace(task, budget=replace(task.budget, max_tool_calls=0))
    print(json.dumps({'packet': label, 'status': 'review_started'}), flush=True)
    record = runner._run_journaled_worker(runner.critic_executor, task)
    critique = _critique_from_record(record, route_steps=bindings)
    write(folder / 'result.json', {'status': record.status, 'critique': critique, 'record': asdict(record),
          'completed_at': datetime.now(timezone.utc).isoformat(), 'prompt_sha256': sha,
          'independence': 'Fresh context, hidden labels and prior verdicts; same model family, not independent expert validation.'})
    result = {'packet': label, 'status': record.status, 'verdict': critique.get('overall_assessment')}
    print(json.dumps(result), flush=True)
    return result


def summarize():
    entries = []
    for row in read(OUTPUT / 'review-map.json'):
        folder = OUTPUT / 'blind-review' / row.get('packet_id', '_absent')
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else {'status': 'pending'}
        structure = read(folder / 'structure-summary.json') if (folder / 'structure-summary.json').exists() else {}
        critique = result.get('critique') or {}
        entries.append({**row, 'review_status': result['status'], 'structure': structure,
                        'independent_verdict': critique.get('overall_assessment'),
                        'route_evaluation': critique.get('route_overall_evaluation'),
                        'step_assessments': critique.get('step_assessments'),
                        'usage': (result.get('record') or {}).get('usage')})
    write(OUTPUT / 'comparison.json', entries)
    print(json.dumps([{k: r.get(k) for k in ('system', 'public_target_name', 'branch_index', 'step_count',
                                           'review_status', 'independent_verdict')} for r in entries], ensure_ascii=False), flush=True)


def render_packets():
    """Source-neutral drawings for later expert review; no new chemical content."""
    labels = sorted({r['packet_id'] for r in read(OUTPUT / 'review-map.json') if r.get('packet_id')})
    css = ('body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#203344}'
           'h1{font-size:26px}h2{font-size:19px;margin-top:0}.step{border:1px solid #cad5df;border-radius:10px;padding:20px;margin:24px 0}'
           '.reaction{overflow:auto}.reaction svg{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace}'
           'summary{cursor:pointer}code{overflow-wrap:anywhere}.note{background:#f0f5f8;padding:16px;border-radius:8px}')
    for label in labels:
        folder = OUTPUT / 'blind-review' / label
        body = read(folder / 'packet.json')
        write(folder / 'structure-summary.json', structural_summary(body))
        cards = []
        for row in body['steps']:
            reaction = rdChemReactions.ReactionFromSmarts('.'.join(row['precursor_smiles']) + '>>' + row['product_smiles'], useSmiles=True)
            drawer = rdMolDraw2D.MolDraw2DSVG(1160, 300)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawReaction(reaction)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            cards.append('<article class="step"><h2>' + escape(row['review_slot'] + ' · ' + row['reaction_family'])
                         + '</h2><div class="reaction">' + svg + '</div><p><b>所提供条件</b></p><pre>'
                         + escape('\n\n'.join(row['condition_hypotheses']) or '未提供')
                         + '</pre><details><summary>精确结构 SMILES</summary><pre>'
                         + escape('.'.join(row['precursor_smiles']) + '>>' + row['product_smiles']) + '</pre></details></article>')
        html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'
        html += label + '</title><style>' + css + '</style><body><h1>匿名路线 ' + label + '</h1>'
        html += '<p class="note">请独立评价具体化学矛盾、选择性与跨步前提。图示箭头按正向反应绘制；步骤列表按从目标逆推的顺序排列。未验证风险与明确错误分别记录，原料闭合另行评价。</p>'
        html += '<details><summary>目标精确结构</summary><code>' + escape(body['target_smiles']) + '</code></details>'
        html += ''.join(cards) + '</body></html>'
        (folder / 'expert-packet.html').write_text(html, encoding='utf-8')
    print(json.dumps({'rendered_packets': len(labels)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare-public', 'prepare-graph', 'collect', 'review', 'summarize', 'render'])
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--case-id')
    parser.add_argument('--packet')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'prepare-public':
        prepare_public()
    elif args.action == 'prepare-graph':
        if not args.graph or not args.case_id:
            parser.error('prepare-graph requires --graph and --case-id')
        prepare_graph(args.graph.resolve(), args.case_id)
    elif args.action == 'review':
        labels = [args.packet] if args.packet else sorted({r['packet_id'] for r in read(OUTPUT / 'review-map.json') if r.get('packet_id')})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(review, labels))
        summarize()
    elif args.action == 'collect':
        collect_finished()
    elif args.action == 'render':
        render_packets()
    else:
        summarize()


if __name__ == '__main__':
    main()
