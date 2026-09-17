"""Compare inferred route endpoints with the explicitly scoped local manifests."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import unicodedata

from audit_absynth_dataset import molecule, normalize_doi, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/discussion/absynth-quality-20260906'


def normalized_name(s):
    return ' '.join(unicodedata.normalize('NFKC', s).strip().casefold().split())


def walk(x):
    if isinstance(x, dict):
        if isinstance(x.get('target_smiles'), str) and x['target_smiles']:
            yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def main():
    routes = json.loads((OUTPUT / 'routes.json').read_text(encoding='utf-8'))
    by_endpoint, by_flat, by_name = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in routes:
        for m in r['sink_product_hypotheses']:
            by_endpoint[m['iso']].append(r)
            by_flat[m['flat']].append(r)
        for n in r['TargetName']:
            by_name[normalized_name(n)].append(r)
    inputs = sorted(set((ROOT / 'benchmarks').glob('synthex*.json')) | set((ROOT / 'benchmarks').glob('baccatin*.json')) | set((ROOT / 'benchmarks').glob('nature_2026*.json')))
    public_map = ROOT / 'results/discussion/synthex-mainpaper-comparison-20260906/target-map.json'
    aliases = {r['case_id']: r['public_target_name'] for r in json.loads(public_map.read_text(encoding='utf-8'))} if public_map.exists() else {}
    matches, unique_checked = [], set()
    for f in inputs:
        for x in walk(json.loads(f.read_text(encoding='utf-8'))):
            m = molecule(x['target_smiles'])
            if not m['valid']:
                continue
            unique_checked.add(m['iso'])
            exact, flat = by_endpoint[m['iso']], by_flat[m['flat']]
            public_name = aliases.get(x.get('case_id'), x.get('target_name', ''))
            names = by_name.get(normalized_name(public_name), [])
            matches.append({'manifest': str(f.relative_to(ROOT)), 'case_id': x.get('case_id'), 'target_name': public_name,
                            'same_name_PathIds': sorted({r['PathId'] for r in names}),
                            'exact_endpoint_PathIds': sorted({r['PathId'] for r in exact}),
                            'connectivity_only_endpoint_PathIds': sorted({r['PathId'] for r in flat} - {r['PathId'] for r in exact}),
                            'matched_names': sorted({n for r in exact + flat for n in r['TargetName']})})
    with (ROOT.parent / 'ABSynth_Dataset.csv').open(encoding='utf-8-sig', newline='') as handle:
        raw = list(csv.DictReader(handle, delimiter='\t'))
    abs_dois = {normalize_doi(r['DOI']) for r in raw}
    recent = []
    for filename in ['papers.jsonl', 'target_slots.jsonl', 'route_coverage.jsonl']:
        path = ROOT / 'benchmarks/recent_total_synthesis' / filename
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        dois = {normalize_doi(r.get('doi', '')) for r in records if r.get('doi')}
        recent.append({'file': str(path.relative_to(ROOT)), 'records': len(records), 'dois': len(dois), 'overlap_dois': sorted(dois & abs_dois)})
    name_groups = [{'names': sorted({n for r in rows for n in r['TargetName']}), 'PathIds': sorted({r['PathId'] for r in rows})}
                   for rows in by_endpoint.values() if len({normalized_name(n) for r in rows for n in r['TargetName']}) > 1]
    multiple_structures = [{'normalized_name': name, 'PathIds': sorted({r['PathId'] for r in rows}),
                            'isomeric_structure_count': len({m['iso'] for r in rows for m in r['sink_product_hypotheses']})}
                           for name, rows in by_name.items() if len({m['iso'] for r in rows for m in r['sink_product_hypotheses']}) > 1]
    sink_mols = [m for r in routes for m in r['sink_product_hypotheses']]
    data = {'scope_manifests': [str(f.relative_to(ROOT)) for f in inputs], 'distinct_target_structures_checked': len(unique_checked),
            'manifest_rows': matches, 'recent_corpus': recent,
            'same_endpoint_different_name_groups': name_groups, 'same_normalized_name_multiple_endpoint_groups': multiple_structures,
            'scope_diagnostics': {'normalized_target_names': len(by_name), 'endpoint_heavy_atoms_at_most_10': sum(m['heavy_atoms'] <= 10 for m in sink_mols),
                                  'endpoint_rings_at_most_1': sum(m['rings'] <= 1 for m in sink_mols),
                                  'endpoint_heavy_atoms_at_least_20_and_rings_at_least_3': sum(m['heavy_atoms'] >= 20 and m['rings'] >= 3 for m in sink_mols),
                                  'routes_year_at_least_2020': sum(int(r['Year'][0]) >= 2020 for r in routes)},
            'semantics': {'exact_match_is_structure_overlap_not_proof_of_model_training_contamination': True,
                          'connectivity_only_match_is_not_identical_stereochemical_target': True,
                          'inferred_sink_may_differ_from_named_natural_product_in_multitarget_sequence': True,
                          'name_groups_are_review_candidates_not_automatic_merges': True}}
    write_json(OUTPUT / 'overlap.json', data)
    print(json.dumps({'targets_checked': len(unique_checked), 'matches': [r for r in matches if r['exact_endpoint_PathIds'] or r['connectivity_only_endpoint_PathIds']],
                      'recent': recent, 'scope': data['scope_diagnostics'], 'same_structure_name_groups': len(name_groups),
                      'same_name_structure_groups': len(multiple_structures)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
