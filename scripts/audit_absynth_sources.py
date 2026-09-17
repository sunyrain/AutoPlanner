"""Deterministic DOI metadata sample; preserves responses and sampling scope."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.parse import quote

import requests

from audit_absynth_dataset import normalize_doi, write_json


def main(source, output):
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    by_doi = defaultdict(list)
    for r in rows:
        by_doi[normalize_doi(r['DOI'])].append(r)
    candidates = sorted(d for d in by_doi if re.fullmatch(r'10\.\d{4,9}/\S+', d))
    sample = random.Random(20260906).sample(candidates, 20)
    by_year = sorted(candidates, key=lambda d: (min(int(r['Year']) for r in by_doi[d]), d))
    selected = sorted(set(sample + by_year[:4] + by_year[-4:] + ['10.1021/acs.joc.1c02638', '10.1038/s41557-020-00603-z']))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'source-sample-design.json', {'seed': 20260906, 'uniform_random_dois': sample,
        'earliest_four': by_year[:4], 'latest_four': by_year[-4:], 'selected_dois': selected,
        'additional_known_examples': ['10.1021/acs.joc.1c02638', '10.1038/s41557-020-00603-z'],
        'scope': 'DOI metadata only, not complete article/SI or reaction validation'})

    def fetch(doi):
        tag = hashlib.sha256(doi.encode()).hexdigest()[:16]
        cache = output / ('crossref-' + tag + '.json')
        if cache.exists():
            return json.loads(cache.read_text(encoding='utf-8'))
        url = 'https://api.crossref.org/works/' + quote(doi, safe='')
        item = {'doi': doi, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'dataset_years': sorted({r['Year'] for r in by_doi[doi]}),
                'dataset_authors': sorted({r['Author'] for r in by_doi[doi]}),
                'dataset_targets': sorted({r['TargetName'] for r in by_doi[doi]}),
                'PathIds': sorted({r['PathId'] for r in by_doi[doi]})}
        try:
            response = requests.get(url, timeout=25, headers={'User-Agent': 'AutoPlanner-dataset-quality-audit/1.0'})
            item['http_status'] = response.status_code
            if response.status_code == 200:
                metadata = response.json()['message']
                item['metadata'] = metadata
                dates = {k: metadata[k]['date-parts'] for k in ['issued', 'published', 'published-online', 'published-print'] if metadata.get(k, {}).get('date-parts')}
                item['date_fields'] = dates
                item['year_matches_one_publication_date'] = bool({int(y) for y in item['dataset_years']} & {d[0] for v in dates.values() for d in v if d})
                authors = {a.get('family', '').lower() for a in metadata.get('author', [])}
                item['author_label_matches_a_family_name'] = all(a.lower() in authors for a in item['dataset_authors'])
            else:
                item['response_excerpt'] = response.text[:400]
        except requests.RequestException as e:
            item['error'] = str(e)
        write_json(cache, item)
        return item
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(fetch, selected))
    write_json(output / 'source-sample-results.json', result)
    print(json.dumps([{k: x.get(k) for k in ['doi', 'http_status', 'dataset_targets', 'dataset_years', 'year_matches_one_publication_date', 'author_label_matches_a_family_name', 'error']} | {'title': x.get('metadata', {}).get('title')} for x in result], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
