"""Build an atom-mapping cache for v4 cascade program reactions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def build_v4_atommap_cache(
    *,
    program_manifest: Path,
    output_cache: Path,
    seed_cache_paths: list[Path],
    splits: tuple[str, ...] = ("train",),
    limit: int | None = None,
    batch_size: int = 16,
    checkpoint_every: int = 100,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    reactions = _collect_reactions(manifest, splits=splits)
    cache: dict[str, str | None] = {}
    for path in seed_cache_paths:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            cache.update({str(key): value for key, value in payload.items()})
    if output_cache.exists():
        payload = json.loads(output_cache.read_text(encoding="utf-8"))
        cache.update({str(key): value for key, value in payload.items()})

    missing = [rxn for rxn in reactions if rxn not in cache]
    if limit is not None:
        missing = missing[: max(0, int(limit))]
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    mapped_this_run = 0
    failed_this_run = 0
    if missing:
        from rxnmapper import RXNMapper

        mapper = RXNMapper()
        for start in range(0, len(missing), max(1, int(batch_size))):
            chunk = missing[start : start + max(1, int(batch_size))]
            try:
                results = mapper.get_attention_guided_atom_maps(chunk)
                for rxn, result in zip(chunk, results):
                    mapped = result.get("mapped_rxn") if isinstance(result, dict) else None
                    cache[rxn] = mapped
                    if mapped:
                        mapped_this_run += 1
                    else:
                        failed_this_run += 1
            except Exception:
                for rxn in chunk:
                    try:
                        result = mapper.get_attention_guided_atom_maps([rxn])[0]
                        mapped = result.get("mapped_rxn") if isinstance(result, dict) else None
                    except Exception:
                        mapped = None
                    cache[rxn] = mapped
                    if mapped:
                        mapped_this_run += 1
                    else:
                        failed_this_run += 1
            done = start + len(chunk)
            if done % max(1, int(checkpoint_every)) == 0 or done >= len(missing):
                output_cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "mapped_missing": done,
                            "total_missing_this_run": len(missing),
                            "cache_entries": len(cache),
                            "cache_mapped": sum(1 for value in cache.values() if value),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    else:
        output_cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    coverage = _coverage(reactions, cache)
    report = {
        "program_manifest": str(program_manifest),
        "output_cache": str(output_cache),
        "seed_cache_paths": [str(path) for path in seed_cache_paths],
        "splits": list(splits),
        "unique_reactions": len(reactions),
        "limit": limit,
        "missing_selected": len(missing),
        "mapped_this_run": mapped_this_run,
        "failed_this_run": failed_this_run,
        "coverage": coverage,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    output_cache.with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _collect_reactions(manifest: dict[str, Any], *, splits: tuple[str, ...]) -> list[str]:
    seen = set()
    reactions = []
    outputs = manifest.get("outputs") or {}
    for split in splits:
        path = Path(outputs[split])
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                for step in payload.get("steps") or []:
                    if not isinstance(step, dict):
                        continue
                    rxn = str(step.get("rxn_smiles") or "")
                    if rxn and rxn not in seen:
                        seen.add(rxn)
                        reactions.append(rxn)
    return reactions


def _coverage(reactions: list[str], cache: dict[str, str | None]) -> dict[str, Any]:
    return {
        "covered": sum(1 for rxn in reactions if rxn in cache),
        "mapped": sum(1 for rxn in reactions if cache.get(rxn)),
        "total": len(reactions),
        "covered_rate": round(sum(1 for rxn in reactions if rxn in cache) / max(len(reactions), 1), 6),
        "mapped_rate": round(sum(1 for rxn in reactions if cache.get(rxn)) / max(len(reactions), 1), 6),
    }


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    return splits or ("train",)


def _parse_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in str(value or "").split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 atom-map cache")
    parser.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    parser.add_argument("--output-cache", default="results/shared/v4_atommap_cache_merged.json")
    parser.add_argument("--seed-cache", default="results/shared/enzexpand_atommap_cache.json,results/shared/v4_atommap_cache.json")
    parser.add_argument("--splits", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    report = build_v4_atommap_cache(
        program_manifest=Path(args.program_manifest),
        output_cache=Path(args.output_cache),
        seed_cache_paths=_parse_paths(args.seed_cache),
        splits=_parse_splits(args.splits),
        limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
