"""Read-only structural audit of a literature route table; no chemistry scoring.

All inferred endpoints and graph links are hypotheses from exact molecule identity.
The original file is never changed. Detailed outputs are audit intermediates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from statistics import median

from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def normalize_doi(value):
    value = value.strip().lower()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'http://dx.doi.org/', 'https://dx.doi.org/', 'doi/'):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


@lru_cache(maxsize=None)
def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'valid': False}
    mapped = sum(a.GetAtomMapNum() != 0 for a in mol.GetAtoms())
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return {
        'valid': True, 'iso': Chem.MolToSmiles(mol, isomericSmiles=True),
        'flat': Chem.MolToSmiles(mol, isomericSmiles=False),
        'heavy_atoms': mol.GetNumHeavyAtoms(), 'rings': mol.GetRingInfo().NumRings(),
        'mw': round(Descriptors.MolWt(mol), 3), 'mapped_atoms': mapped,
        'unspecified_tetrahedral_centers': sum(c == '?' for _, c in chiral),
        'specified_tetrahedral_centers': sum(c != '?' for _, c in chiral),
        'potential_unassigned_stereo': sum(str(s.specified) == 'Unspecified' for s in Chem.FindPotentialStereo(mol)),
        'dummy_atoms': sum(a.GetAtomicNum() == 0 for a in mol.GetAtoms()),
        'radical_electrons': sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
    }


def components(side):
    # This table uses dot-separated component SMILES (not CXSMILES records).
    return [molecule(s) for s in side.split('.') if s]


def signature(parts, key):
    return tuple(sorted(x[key] for x in parts))


def graph_analysis(rows, key):
    producers = defaultdict(set)
    for i, row in enumerate(rows):
        for s in row.get('products_' + key, []):
            producers[s].add(i)
    deps = {i: set() for i in range(len(rows))}
    consumed_elsewhere = set()
    for i, row in enumerate(rows):
        for s in row.get('reactants_' + key, []):
            parents = producers.get(s, set()) - {i}
            deps[i].update(parents)
            if parents:
                consumed_elsewhere.add(s)
    sinks = sorted(set(producers) - consumed_elsewhere)
    sink_seen, sink_pending = set(), [i for s in sinks for i in producers[s]]
    while sink_pending:
        i = sink_pending.pop()
        if i not in sink_seen:
            sink_seen.add(i)
            sink_pending.extend(deps[i] - sink_seen)
    seen, pending = set(), [len(rows) - 1]
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(deps[i] - seen)
    colors = {}
    def cyclic(i):
        if colors.get(i) == 1:
            return True
        if colors.get(i) == 2:
            return False
        colors[i] = 1
        if any(cyclic(j) for j in deps[i]):
            return True
        colors[i] = 2
        return False
    return {'sink_products': sinks, 'last_row_ancestor_count': len(seen),
            'sink_ancestor_count': len(sink_seen),
            'unconnected_to_sinks': [r['line'] for i, r in enumerate(rows) if i not in sink_seen],
            'unconnected_to_last_row': [r['line'] for i, r in enumerate(rows) if i not in seen],
            'identity_cycle': any(cyclic(i) for i in deps),
            'dependency_links': sum(map(len, deps.values()))}


def audit(source, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader)
        raw_rows = [(reader.line_num, row) for row in reader]
    fields = [x for x in header if x]
    assert fields == ['PathId', 'TargetName', 'Year', 'DOI', 'Author', 'StepId', 'RxnSMILES', 'RxnName', 'Conditions']
    row_audit, grouped = [], defaultdict(list)
    duplicate_ids, raw_reactions, iso_reactions = defaultdict(list), defaultdict(list), defaultdict(list)
    issues, meta_issues = [], []
    doi_rows = defaultdict(list)
    stats = Counter()
    for n, (line, values) in enumerate(raw_rows, 1):
        if len(values) != len(fields):
            issues.append({'line': line, 'issue': 'wrong_field_count', 'count': len(values)})
            continue
        d = dict(zip(fields, values))
        row = {'line': line, **{k: d[k] for k in ['PathId', 'StepId', 'TargetName', 'Year', 'DOI']}, 'issues': []}
        duplicate_ids[(d['PathId'], d['StepId'])].append(line)
        raw_reactions[d['RxnSMILES']].append(line)
        doi_rows[normalize_doi(d['DOI'])].append(d)
        if not re.fullmatch(r'10\.\d{4,9}/\S+', normalize_doi(d['DOI'])):
            meta_issues.append({'line': line, 'PathId': d['PathId'], 'field': 'DOI', 'value': d['DOI'], 'issue': 'not_normalized_doi'})
        if any(not v.strip() for v in values):
            row['issues'].append('empty_field')
        if d['Conditions'].strip() == '-':
            stats['conditions_dash'] += 1
        for field, pattern, label in [('Conditions', r'\d\s*%', 'conditions_contain_percent'),
                                      ('Conditions', r'°|\b(?:rt|r\.t\.|reflux)\b', 'conditions_temperature_text')]:
            stats[label] += bool(re.search(pattern, d[field], re.I))
        parts = d['RxnSMILES'].split('>')
        if len(parts) != 3 or not parts[0] or not parts[2]:
            row['issues'].append('reaction_syntax_or_empty_side')
        else:
            reactants, agents, products = [components(s) for s in parts]
            all_parts = reactants + agents + products
            bad = [s for side in parts for s in side.split('.') if s and not molecule(s)['valid']]
            if bad:
                row['issues'].append('invalid_smiles')
                row['invalid_components'] = bad
            else:
                stats['valid_reactions'] += 1
                stats['rows_with_agent_field'] += bool(parts[1])
                for key in ('iso', 'flat'):
                    row['reactants_' + key] = signature(reactants, key)
                    row['products_' + key] = signature(products, key)
                iso_reactions[(row['reactants_iso'], row['products_iso'])].append(line)
                if row['reactants_iso'] == row['products_iso']:
                    row['issues'].append('no_net_structure_change')
                elif row['reactants_flat'] == row['products_flat']:
                    row['issues'].append('stereo_only_transformation')
                for attr in ('mapped_atoms', 'dummy_atoms', 'radical_electrons', 'specified_tetrahedral_centers', 'unspecified_tetrahedral_centers', 'potential_unassigned_stereo'):
                    stats['rows_with_' + attr] += any(p[attr] for p in all_parts)
                    if attr in ('dummy_atoms', 'radical_electrons') and any(p[attr] for p in all_parts):
                        row['issues'].append(attr)
                stats['rows_with_multiple_products'] += len(products) > 1
                row['largest_product_heavy_atoms'] = max(p['heavy_atoms'] for p in products)
        for issue in row['issues']:
            stats[issue] += 1
            issues.append({k: row[k] for k in ['line', 'PathId', 'StepId', 'TargetName']} | {'issue': issue})
        grouped[d['PathId']].append((d, row))
        row_audit.append(row)
        if n % 10000 == 0:
            print(json.dumps({'processed': n, 'unique_components_cached': molecule.cache_info().currsize}), flush=True)
    route_audit, route_signatures = [], defaultdict(list)
    for path_id, items in grouped.items():
        ds, rows = zip(*items)
        meta = {k: sorted({d[k] for d in ds}) for k in ['TargetName', 'Year', 'DOI', 'Author']}
        mixed = {k: v for k, v in meta.items() if len(v) > 1}
        if mixed:
            meta_issues.append({'PathId': path_id, 'issue': 'within_path_metadata_conflict', 'fields': mixed})
        record = {'PathId': path_id, 'first_line': rows[0]['line'], 'last_line': rows[-1]['line'],
                  'row_count': len(rows), **meta, 'metadata_conflicts': mixed,
                  'invalid_reaction_lines': [r['line'] for r in rows if 'products_iso' not in r]}
        exact, relaxed = graph_analysis(rows, 'iso'), graph_analysis(rows, 'flat')
        record.update(exact=exact, stereo_ignored_diagnostic=relaxed)
        record['sink_product_hypotheses'] = [dict(smiles=s, **molecule(s)) for s in exact['sink_products']]
        last = rows[-1].get('products_iso', [])
        record['last_row_product_hypotheses'] = [dict(smiles=s, **molecule(s)) for s in last]
        record['route_chemistry_key'] = hashlib.sha256(repr(sorted((r.get('reactants_iso', ()), r.get('products_iso', ())) for r in rows)).encode()).hexdigest()
        if not record['invalid_reaction_lines']:
            route_signatures[record['route_chemistry_key']].append(path_id)
        route_audit.append(record)
    repeated_iso = [v for v in iso_reactions.values() if len(v) > 1]
    line_to_path = {r['line']: r['PathId'] for r in row_audit}
    cross_path = [v for v in repeated_iso if len({line_to_path[line] for line in v}) > 1]
    final_mols = [m for r in route_audit for m in r['last_row_product_hypotheses']]
    sink_mols = [m for r in route_audit for m in r['sink_product_hypotheses']]
    summary = {
        'source': str(source.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
        'bytes': len(raw), 'rdkit_version': rdBase.rdkitVersion, 'delimiter': 'TAB',
        'header_has_trailing_empty_column': header[-1] == '', 'data_row_widths': dict(Counter(len(v) for _, v in raw_rows)),
        'rows': len(raw_rows), 'routes': len(grouped), 'unique_raw_names': len({r['TargetName'] for r in row_audit}),
        'unique_raw_dois': len({r['DOI'] for r in row_audit}), 'unique_normalized_source_values': len(doi_rows),
        'year_range': [min(int(r['Year']) for r in row_audit), max(int(r['Year']) for r in row_audit)],
        'route_step_count': {'min': min(r['row_count'] for r in route_audit), 'median': median(r['row_count'] for r in route_audit), 'max': max(r['row_count'] for r in route_audit)},
        'row_checks': dict(stats),
        'within_path_duplicate_step_ids': [{'PathId': k[0], 'StepId': k[1], 'lines': v} for k, v in duplicate_ids.items() if len(v) > 1],
        'raw_unique_reactions': len(raw_reactions), 'canonical_unique_reactions': len(iso_reactions),
        'canonical_repeated_reaction_groups': len(repeated_iso),
        'canonical_repeated_reaction_excess_rows': sum(len(v) - 1 for v in repeated_iso),
        'cross_path_repeated_reaction_groups': len(cross_path),
        'cross_path_repeated_reaction_rows': sum(map(len, cross_path)),
        'exact_duplicate_route_groups': [v for v in route_signatures.values() if len(v) > 1],
        'routes_with_metadata_conflict': sum(bool(r['metadata_conflicts']) for r in route_audit),
        'routes_with_invalid_reactions': sum(bool(r['invalid_reaction_lines']) for r in route_audit),
        'routes_with_multiple_sink_products': sum(len(r['exact']['sink_products']) > 1 for r in route_audit),
        'routes_with_exact_identity_cycles': sum(r['exact']['identity_cycle'] for r in route_audit),
        'routes_all_rows_connected_to_unique_sink': sum(len(r['exact']['sink_products']) == 1 and r['exact']['sink_ancestor_count'] == r['row_count'] and not r['invalid_reaction_lines'] for r in route_audit),
        'routes_last_row_is_not_sink': sum(set(r['exact']['sink_products']) != {m['smiles'] for m in r['last_row_product_hypotheses']} for r in route_audit),
        'canonical_reparse_non_idempotent_routes': [r['PathId'] for r in route_audit if any(m['smiles'] != m['iso'] for m in r['sink_product_hypotheses'])],
        'unique_sink_isomeric_products': len({m['iso'] for m in sink_mols}),
        'sink_products_with_unspecified_tetrahedral_centers': sum(m['unspecified_tetrahedral_centers'] > 0 for m in sink_mols),
        'sink_products_with_potential_unassigned_stereo': sum(m['potential_unassigned_stereo'] > 0 for m in sink_mols),
        'sink_products_heavy_atoms': {'median': median(m['heavy_atoms'] for m in sink_mols), 'min': min(m['heavy_atoms'] for m in sink_mols), 'max': max(m['heavy_atoms'] for m in sink_mols)},
        'routes_all_rows_connected_to_last_exact': sum(r['exact']['last_row_ancestor_count'] == r['row_count'] and not r['invalid_reaction_lines'] for r in route_audit),
        'routes_all_rows_connected_to_last_stereo_ignored': sum(r['stereo_ignored_diagnostic']['last_row_ancestor_count'] == r['row_count'] and not r['invalid_reaction_lines'] for r in route_audit),
        'routes_with_more_connections_when_stereo_ignored': sum(r['stereo_ignored_diagnostic']['dependency_links'] > r['exact']['dependency_links'] for r in route_audit),
        'last_row_product_hypotheses': len(final_mols),
        'last_row_unique_isomeric_products': len({m['iso'] for m in final_mols}),
        'last_row_products_with_unspecified_tetrahedral_centers': sum(m['unspecified_tetrahedral_centers'] > 0 for m in final_mols),
        'last_row_products_with_potential_unassigned_stereo': sum(m['potential_unassigned_stereo'] > 0 for m in final_mols),
        'last_row_products_heavy_atoms': {'median': median(m['heavy_atoms'] for m in final_mols), 'min': min(m['heavy_atoms'] for m in final_mols), 'max': max(m['heavy_atoms'] for m in final_mols)},
        'semantics': {'route_id_is_not_independent_target': True, 'last_row_product_is_unverified_target_hypothesis': True,
                      'identity_links_are_not_authoritative_route_edges': True, 'stereo_ignored_graph_is_diagnostic_only': True,
                      'missing_reaction_atom_maps_are_not_chemical_error': True, 'stereo_unspecified_can_intentionally_represent_racemate': True,
                      'no_reaction_feasibility_or_literature_ground_truth_claim': True},
    }
    write_json(output / 'summary.json', summary)
    write_json(output / 'routes.json', route_audit)
    write_json(output / 'issues.json', issues)
    write_json(output / 'metadata-issues.json', meta_issues)
    write_json(output / 'repeated-reactions.json', [{'lines': v, 'PathIds': sorted({line_to_path[line] for line in v})} for v in cross_path])
    with (output / 'rows.jsonl').open('w', encoding='utf-8') as handle:
        for row in row_audit:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.source, args.output)
