"""Uniprot enrichment for cascade snapshot.

Walks every catalyst_component with EC (and optionally organism), queries
Uniprot REST for the top reviewed (Swiss-Prot) entry, and writes an
enriched snapshot with `uniprot_id`, names, reviewed/status fields,
sequence, taxon, Rhea cross-references, and cofactors populated per component.

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
ENTRY_URL = "https://rest.uniprot.org/uniprotkb"


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


def _records(data) -> list[dict]:
    if isinstance(data, dict):
        records = data.get("records_kept", data.get("records", []))
        return records if isinstance(records, list) else []
    return data if isinstance(data, list) else []


def _protein_name(entry: dict) -> str | None:
    protein = entry.get("proteinDescription") or {}
    recommended = protein.get("recommendedName") or {}
    full = recommended.get("fullName") or {}
    if full.get("value"):
        return full["value"]
    submitted = protein.get("submissionNames") or []
    if submitted:
        full = (submitted[0] or {}).get("fullName") or {}
        if full.get("value"):
            return full["value"]
    return None


def _cofactors(entry: dict) -> list[str]:
    out = []
    for comment in entry.get("comments") or []:
        if comment.get("commentType") != "COFACTOR":
            continue
        for cofactor in comment.get("cofactors") or []:
            name = (cofactor.get("name") or "").strip()
            if name:
                out.append(name)
    return sorted(set(out))


def _rhea_ids(entry: dict) -> list[str]:
    out = []
    for xref in entry.get("uniProtKBCrossReferences") or []:
        if xref.get("database") == "Rhea" and xref.get("id"):
            out.append(str(xref["id"]))
    return sorted(set(out))


def _ec_numbers(entry: dict) -> list[str]:
    out = []
    protein = entry.get("proteinDescription") or {}
    names = []
    for key in ("recommendedName", "alternativeNames"):
        value = protein.get(key)
        if isinstance(value, list):
            names.extend(value)
        elif value:
            names.append(value)
    for name in names:
        for ec in (name or {}).get("ecNumbers") or []:
            value = ec.get("value")
            if value:
                out.append(value)
    return sorted(set(out))


def parse_uniprot_entry(entry: dict) -> dict | None:
    """Normalize a UniProtKB JSON entry into compact evidence fields."""
    if not entry:
        return None
    accession = entry.get("primaryAccession")
    if not accession:
        accessions = entry.get("secondaryAccessions") or []
        accession = accessions[0] if accessions else None
    if not accession:
        return None
    organism = entry.get("organism") or {}
    sequence = entry.get("sequence") or {}
    entry_type = entry.get("entryType") or ""
    cofactors = _cofactors(entry)
    return {
        "accession": accession,
        "entry_name": entry.get("uniProtkbId"),
        "organism": organism.get("scientificName"),
        "tax_id": organism.get("taxonId"),
        "protein_name": _protein_name(entry),
        "reviewed": "reviewed" in entry_type.lower() or "swiss-prot" in entry_type.lower(),
        "entry_type": entry_type,
        "protein_existence": entry.get("proteinExistence"),
        "sequence": sequence.get("value"),
        "sequence_length": sequence.get("length"),
        "ec_numbers": _ec_numbers(entry),
        "rhea_ids": _rhea_ids(entry),
        "cofactors": cofactors,
        "cofactor": "; ".join(cofactors) if cofactors else None,
    }


def query_uniprot_entry(accession: str, cache: dict, sleep: float = 0.0) -> dict | None:
    """Fetch one UniProtKB accession as JSON and normalize it."""
    if not accession:
        return None
    key = f"accession::{accession}"
    if key in cache:
        return cache[key]
    hit = None
    try:
        r = requests.get(f"{ENTRY_URL}/{accession}.json", timeout=20)
        if r.status_code == 200:
            hit = parse_uniprot_entry(r.json())
    except Exception:
        hit = None
    finally:
        if sleep > 0:
            time.sleep(sleep)
    cache[key] = hit
    return hit


def query_uniprot(ec: str, organism: str | None, cache: dict, sleep: float = 0.0) -> dict | None:
    """Return normalized UniProt evidence or None.

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
                    hit = query_uniprot_entry(parts[0].strip(), cache, sleep=sleep)
                    if hit is None:
                        hit = {
                            "accession": parts[0].strip() or None,
                            "entry_name": parts[1].strip() or None,
                            "organism": parts[2].strip() or None,
                            "protein_name": parts[3].strip() or None,
                            "reviewed": True,
                            "entry_type": "reviewed",
                        }
                    hit["matched_organism"] = bool(organism) and (q == queries[0])
                    break
        except Exception:
            continue
        finally:
            if sleep > 0:
                time.sleep(sleep)

    cache[key] = hit  # cache misses too, to avoid re-query
    return hit


def _set_if_present(target: dict, key: str, value, *, overwrite: bool = False) -> bool:
    if value in (None, "", [], {}):
        return False
    if not overwrite and target.get(key) not in (None, "", [], {}):
        return False
    if target.get(key) == value:
        return False
    target[key] = value
    return True


def _apply_uniprot_result(
    cat: dict,
    res: dict | None,
    *,
    match_quality: str | None = None,
) -> bool:
    """Apply normalized UniProt evidence to a catalyst component."""
    if not isinstance(res, dict) or not res:
        return False

    changed = False
    changed |= _set_if_present(cat, "uniprot_id", res.get("accession"))
    changed |= _set_if_present(cat, "uniprot_entry_name", res.get("entry_name"))
    changed |= _set_if_present(cat, "uniprot_protein_name", res.get("protein_name"))
    changed |= _set_if_present(cat, "uniprot_organism", res.get("organism"))
    if res.get("reviewed") is not None:
        changed |= _set_if_present(
            cat,
            "uniprot_status",
            "reviewed" if res.get("reviewed") else "unreviewed",
        )
    changed |= _set_if_present(cat, "uniprot_tax_id", res.get("tax_id"))
    changed |= _set_if_present(cat, "uniprot_protein_existence", res.get("protein_existence"))
    changed |= _set_if_present(cat, "uniprot_sequence", res.get("sequence"))
    changed |= _set_if_present(cat, "enzyme_seq_length", res.get("sequence_length"))
    changed |= _set_if_present(cat, "cofactor_required", res.get("cofactor"))

    if res.get("rhea_ids"):
        external = cat.get("enzyme_external_ids")
        if not isinstance(external, dict):
            external = {}
            cat["enzyme_external_ids"] = external
            changed = True
        if external.get("rhea") in (None, "", [], {}):
            external["rhea"] = res["rhea_ids"]
            changed = True

    if match_quality:
        changed |= _set_if_present(
            cat,
            "uniprot_match_quality",
            match_quality,
            overwrite=True,
        )
    return changed


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

    records = _records(data)
    pairs = set()
    accessions = set()
    for r in records:
        for c in r.get("cascades", []):
            for s in c.get("steps", []):
                for cat in s.get("catalyst_components", []) or []:
                    ec = (cat.get("ec_number") or "").strip()
                    org = (cat.get("organism") or "").strip()
                    uid = (cat.get("uniprot_id") or "").strip()
                    if uid:
                        accessions.add(uid)
                    if ec and not cat.get("uniprot_id"):
                        pairs.add((ec, org or None))

    pairs = sorted(pairs, key=lambda p: (p[0], p[1] or ""))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[enrich] unique (EC, organism) pairs to query: {len(pairs)}")
    print(f"[enrich] existing UniProt accessions to detail: {len(accessions)}")

    n_accession_hit = 0
    for i, uid in enumerate(sorted(accessions), 1):
        if query_uniprot_entry(uid, cache, sleep=args.sleep if args.workers <= 1 else 0.0):
            n_accession_hit += 1
        if i % 50 == 0:
            _save_cache(cache)
            print(f"  [accession {i}/{len(accessions)}] detailed={n_accession_hit}", flush=True)
    _save_cache(cache)

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

        def query_pair(ec: str, org: str | None):
            local_cache = {}
            res = query_uniprot(ec, org, local_cache, sleep=0.0)
            return res, local_cache

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(query_pair, ec, org): (ec, org)
                    for ec, org in pending}
            for fut in as_completed(futs):
                ec, org = futs[fut]
                try:
                    res, local_cache = fut.result()
                except Exception:
                    res = None
                    local_cache = {}
                cache.update(local_cache)
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
    n_pair_hit = sum(1 for ec, org in pairs if cache.get(f"{ec}||{org or ''}"))
    print(f"[enrich] total pair hits available: {n_pair_hit}/{len(pairs)}")

    # Apply enrichment back into the snapshot
    n_steps_updated = 0
    n_components_updated = 0
    for r in records:
        for c in r.get("cascades", []):
            for s in c.get("steps", []):
                step_updated = False
                for cat in s.get("catalyst_components", []) or []:
                    ec = (cat.get("ec_number") or "").strip()
                    org = (cat.get("organism") or "").strip()
                    uid = (cat.get("uniprot_id") or "").strip()
                    res = None
                    match_quality = None
                    if uid:
                        res = cache.get(f"accession::{uid}")
                        match_quality = "existing_accession"
                    if not res and ec:
                        res = cache.get(f"{ec}||{org or ''}") or cache.get(f"{ec}||")
                        if res:
                            match_quality = (
                                "ec_and_organism" if res.get("matched_organism") else "ec_only"
                            )
                    if not res:
                        continue
                    if _apply_uniprot_result(cat, res, match_quality=match_quality):
                        n_components_updated += 1
                        step_updated = True
                if step_updated:
                    n_steps_updated += 1

    if isinstance(data, dict):
        md = data.setdefault("metadata", {})
        md.setdefault("enrichments", []).append({
            "stage": "uniprot_rest_enrichment",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "queries_unique": len(pairs),
            "queries_with_hit": n_pair_hit,
            "new_queries_with_hit": n_hit,
            "existing_accessions_detailed": n_accession_hit,
            "components_updated": n_components_updated,
            "steps_updated": n_steps_updated,
        })

    print(f"[enrich] components updated: {n_components_updated}, steps touched: {n_steps_updated}")
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[enrich] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
