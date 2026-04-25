"""One-time atom-mapping of all enzymatic reactions in our snapshot.

Outputs a JSON dict {raw_rxn_smiles: mapped_rxn_smiles_or_null}
to results/enzexpand_atommap_cache.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cascade_planner.data.loader_v2 import load_v2

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--all", action="store_true",
                    help="Map all steps, not just enzymatic")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    print(f"[load] {args.data}", flush=True)
    steps, _, _ = load_v2(args.data)
    if args.all:
        target = steps
    else:
        target = [s for s in steps if s.ec_number]
    print(f"  target: {len(target)} steps", flush=True)

    cache_path = RESULTS_DIR / "enzexpand_atommap_cache.json"
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  loaded cache: {len(cache)} entries", flush=True)

    todo = []
    seen = set()
    for s in target:
        r = s.rxn_smiles
        if not r or r in cache or r in seen:
            continue
        seen.add(r); todo.append(r)

    print(f"  need to map: {len(todo)}", flush=True)
    if not todo:
        print("  nothing to do", flush=True)
        return

    print("  loading rxnmapper ...", flush=True)
    from rxnmapper import RXNMapper
    rm = RXNMapper()
    print("  done", flush=True)

    t0 = time.time()
    for s_ in range(0, len(todo), args.batch):
        chunk = todo[s_:s_ + args.batch]
        try:
            res = rm.get_attention_guided_atom_maps(chunk)
            for r, rr in zip(chunk, res):
                cache[r] = rr.get("mapped_rxn")
        except Exception:
            for r in chunk:
                try:
                    rr = rm.get_attention_guided_atom_maps([r])
                    cache[r] = rr[0].get("mapped_rxn")
                except Exception as e:
                    cache[r] = None
        done = s_ + len(chunk)
        rate = done / (time.time() - t0 + 1e-9)
        eta = (len(todo) - done) / max(rate, 1e-9)
        print(f"    [{done}/{len(todo)}]  {rate:.1f} rxn/s  ETA {eta:.0f}s", flush=True)
        # checkpoint every 200
        if done % 200 == 0 or done == len(todo):
            cache_path.write_text(json.dumps(cache), encoding="utf-8")

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    n_ok = sum(1 for v in cache.values() if v)
    print(f"\n  done. cache: {len(cache)} entries, mapped OK: {n_ok}", flush=True)
    print(f"  saved -> {cache_path}", flush=True)


if __name__ == "__main__":
    main()
