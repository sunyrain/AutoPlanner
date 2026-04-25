"""Uniprot enrichment for cascade snapshot.

Walks every catalyst_component with EC (and optionally organism), queries
Uniprot REST for the top reviewed (Swiss-Prot) entry, and writes an
enriched snapshot with `uniprot_id`, `uniprot_entry_name`, and
`uniprot_protein_name` populated per component.

Cache: queries are cached in `data/uniprot_cache.json` so reruns don't
hit the network.

Usage:
    python -m cascade_planner.data.enrich_uniprot \
        --in cascade_dataset_v2.normalized.json \
        --out cascade_dataset_v2.normalized.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data" / "uniprot_cache.json"
URL = "https://rest.uniprot.org/uniprotkb/search"


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=0), encoding="utf-8")


def query_uniprot(ec: str, organism: str | None, cache: dict, sleep: float = 0.0) -> dict | None:
    """Return {accession, entry_name, organism, protein_name} or None.

    Tries reviewed=true with organism first, then reviewed=true without
    organism. Honours cache.
    """
    key = f"{ec}||{organism or ''}"
    if key in cache:
        return cache[key]

    queries = []
    if organism:
        queries.append(f'ec:{ec} AND reviewed:true AND organism_name:"{organism}"')
    queries.append(f"ec:{ec} AND reviewed:true")

    hit = None
    for q in queries:
        try:
            r = requests.get(URL, params={
                "query": q, "format": "tsv",
                "fields": "accession,id,organism_name,protein_name", "size": 1,
            }, timeout=20)
            if r.status_code != 200:
                continue
            lines = r.text.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[1].split("\t")
                if len(parts) >= 4:
                    hit = {
                        "accession": parts[0].strip() or None,
                        "entry_name": parts[1].strip() or None,
                        "organism": parts[2].strip() or None,
                        "protein_name": parts[3].strip() or None,
                        "matched_organism": bool(organism) and (q == queries[0]),
                    }
                    break
        except Exception:
            continue
        finally:
            if sleep > 0:
                time.sleep(sleep)

    cache[key] = hit  # cache misses too, to avoid re-query
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default: <in>.uniprot.json)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit unique queries for testing")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="Sleep between API calls (s) when workers=1")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel HTTP workers (Uniprot REST allows ~10 conc)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".uniprot.json")
    print(f"[enrich] in:  {in_path}")
    print(f"[enrich] out: {out_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    cache = _load_cache()
    print(f"[enrich] cache: {len(cache)} entries")

    pairs = set()
    for r in data.get("records_kept", []):
        for c in r.get("cascades", []):
            for s in c.get("steps", []):
                for cat in s.get("catalyst_components", []) or []:
                    ec = (cat.get("ec_number") or "").strip()
                    org = (cat.get("organism") or "").strip()
                    if ec and not cat.get("uniprot_id"):
                        pairs.add((ec, org or None))

    pairs = sorted(pairs, key=lambda p: (p[0], p[1] or ""))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[enrich] unique (EC, organism) pairs to query: {len(pairs)}")

    # Filter pairs that are already in cache.
    pending = [p for p in pairs if f"{p[0]}||{p[1] or ''}" not in cache]
    cached_n = len(pairs) - len(pending)
    print(f"[enrich] {cached_n} already cached, querying {len(pending)} new pairs")

    n_done = 0
    n_hit = 0
    n_org_match = 0
    t0 = time.time()
    if pending and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        lock_cache = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(query_uniprot, ec, org, lock_cache, sleep=0.0): (ec, org)
                    for ec, org in pending}
            for fut in as_completed(futs):
                ec, org = futs[fut]
                try:
                    res = fut.result()
                except Exception:
                    res = None
                cache[f"{ec}||{org or ''}"] = res
                n_done += 1
                if res:
                    n_hit += 1
                    if res.get("matched_organism"):
                        n_org_match += 1
                if n_done % 25 == 0:
                    _save_cache(cache)
                    elapsed = time.time() - t0
                    rate = n_done / max(1e-6, elapsed)
                    eta = (len(pending) - n_done) / max(1e-6, rate)
                    print(f"  [{n_done}/{len(pending)}] hits={n_hit} ({100*n_hit/n_done:.0f}%) "
                          f"org_match={n_org_match}  rate={rate:.1f}/s eta={eta:.0f}s",
                          flush=True)
    else:
        for ec, org in pending:
            res = query_uniprot(ec, org, cache, sleep=args.sleep)
            n_done += 1
            if res:
                n_hit += 1
                if res.get("matched_organism"):
                    n_org_match += 1
            if n_done % 25 == 0:
                _save_cache(cache)
                elapsed = time.time() - t0
                rate = n_done / max(1e-6, elapsed)
                eta = (len(pending) - n_done) / max(1e-6, rate)
                print(f"  [{n_done}/{len(pending)}] hits={n_hit} ({100*n_hit/n_done:.0f}%) "
                      f"org_match={n_org_match}  rate={rate:.1f}/s eta={eta:.0f}s",
                      flush=True)
    _save_cache(cache)
    print(f"[enrich] queries done: {n_done}, hits={n_hit} ({100*n_hit/max(1,n_done):.1f}%), "
          f"organism-matched={n_org_match} ({100*n_org_match/max(1,n_done):.1f}%)")

    # Apply enrichment back into the snapshot
    n_steps_updated = 0
    n_components_updated = 0
    for r in data.get("records_kept", []):
        for c in r.get("cascades", []):
            for s in c.get("steps", []):
                step_updated = False
                for cat in s.get("catalyst_components", []) or []:
                    ec = (cat.get("ec_number") or "").strip()
                    org = (cat.get("organism") or "").strip()
                    if not ec or cat.get("uniprot_id"):
                        continue
                    res = cache.get(f"{ec}||{org or ''}")
                    if not res:
                        continue
                    cat["uniprot_id"] = res.get("accession")
                    cat["uniprot_entry_name"] = res.get("entry_name")
                    cat["uniprot_protein_name"] = res.get("protein_name")
                    cat["uniprot_organism"] = res.get("organism")
                    cat["uniprot_match_quality"] = (
                        "ec_and_organism" if res.get("matched_organism") else "ec_only"
                    )
                    n_components_updated += 1
                    step_updated = True
                if step_updated:
                    n_steps_updated += 1

    md = data.setdefault("metadata", {})
    md.setdefault("enrichments", []).append({
        "stage": "uniprot_rest_enrichment",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "queries_unique": len(pairs),
        "queries_with_hit": n_hit,
        "components_updated": n_components_updated,
        "steps_updated": n_steps_updated,
    })

    print(f"[enrich] components updated: {n_components_updated}, steps touched: {n_steps_updated}")
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[enrich] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
