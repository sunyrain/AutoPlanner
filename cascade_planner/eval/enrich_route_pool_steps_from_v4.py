"""Enrich sparse ChemEnzy route-pool steps with nearest v4 step evidence."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.eval.replay_block_coherence_on_route_pool import _condition_tokens, _main_reactant


ENRICH_SCHEMA_VERSION = "route_pool_step_v4_evidence_enrichment.v1"


def enrich_route_pool_steps_from_v4(
    *,
    route_pool: Path,
    program_manifest: Path,
    output_jsonl: Path,
    report_json: Path,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    started = time.monotonic()
    evidence = _load_train_step_evidence(program_manifest)
    rows = _read_jsonl(route_pool)
    enriched_rows = []
    stats = Counter()
    for row in rows:
        payload = dict(row)
        steps = []
        for idx, step in enumerate(row.get("steps") or []):
            stats["steps"] += 1
            enriched, event = _enrich_step(step, idx=idx, evidence=evidence, min_similarity=min_similarity)
            stats.update(event)
            steps.append(enriched)
        payload["steps"] = steps
        payload["step_enrichment"] = {
            "schema_version": ENRICH_SCHEMA_VERSION,
            "program_manifest": str(program_manifest),
            "min_similarity": min_similarity,
        }
        enriched_rows.append(payload)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in enriched_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": ENRICH_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "program_manifest": str(program_manifest),
            "output_jsonl": str(output_jsonl),
            "report_json": str(report_json),
            "min_similarity": min_similarity,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "routes": len(rows),
            **dict(stats),
        },
        "evidence_summary": evidence["summary"],
        "outputs": {
            "jsonl": str(output_jsonl),
            "report": str(report_json),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _load_train_step_evidence(program_manifest: Path) -> dict[str, Any]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    train_path = Path((manifest.get("outputs") or {})["train"])
    rows = _read_jsonl(train_path)
    items = []
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"fps": [], "items": []})
    for program in rows:
        for step in _program_steps(program):
            item = {
                "program_id": program.get("program_id") or program.get("cascade_id"),
                "doi": program.get("doi"),
                "cascade_id": program.get("cascade_id"),
                "transition_id": step.get("transition_id"),
                "product_smiles": _step_product(step),
                "main_reactant": _step_main_reactant(step),
                "transformation_name": step.get("transformation_name"),
                "transformation_superclass": step.get("transformation_superclass") or "unknown",
                "step_mode": step.get("step_mode") or "unknown",
                "pairwise_mode": step.get("pairwise_mode") or "unknown",
                "intermediate_isolated": step.get("intermediate_isolated"),
                "condition_tokens": step.get("condition_tokens") or _condition_tokens(step.get("step_conditions") or step.get("condition") or {}),
                "catalyst_classes": step.get("catalyst_classes") or [],
                "ec1_values": step.get("ec1_values") or [],
                "enzyme_families": step.get("enzyme_families") or [],
                "cofactors": step.get("cofactors") or [],
                "metal_identities": step.get("metal_identities") or [],
            }
            fp = _transition_fp(item.get("product_smiles"), item.get("main_reactant"))
            if fp is None:
                continue
            items.append(item)
            for key in ("", str(item.get("transformation_superclass") or "")):
                buckets[key]["fps"].append(fp)
                buckets[key]["items"].append(item)
    return {
        "items": items,
        "buckets": dict(buckets),
        "summary": {
            "train_programs": len(rows),
            "train_steps_indexed": len(items),
            "transform_counts": dict(Counter(item.get("transformation_superclass") for item in items)),
        },
    }


def _program_steps(program: dict[str, Any]) -> list[dict[str, Any]]:
    steps = program.get("steps")
    if isinstance(steps, list) and steps:
        return [step for step in steps if isinstance(step, dict)]
    gt_route = program.get("gt_route")
    if isinstance(gt_route, list):
        return [step for step in gt_route if isinstance(step, dict)]
    return []


def _step_product(step: dict[str, Any]) -> str:
    value = step.get("product_smiles")
    if value:
        return _canon(str(value))
    return _largest_component(_reaction_products(step.get("rxn_smiles") or step.get("reaction_smiles")))


def _step_main_reactant(step: dict[str, Any]) -> str:
    value = step.get("main_reactant")
    if value:
        return _canon(str(value))
    reactants = [_canon(str(part)) for part in (step.get("reactants") or []) if part]
    if not reactants:
        reactants = [_canon(str(part)) for part in _reaction_reactants(step.get("rxn_smiles") or step.get("reaction_smiles")) if part]
    return _largest_component(reactants)


def _reaction_reactants(rxn_smiles: Any) -> list[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return []
    left, _ = text.split(">>", 1)
    return [part.strip() for part in left.split(".") if part.strip()]


def _reaction_products(rxn_smiles: Any) -> list[str]:
    text = str(rxn_smiles or "")
    if ">>" not in text:
        return []
    _, right = text.split(">>", 1)
    return [part.strip() for part in right.split(".") if part.strip()]


def _largest_component(parts: list[str]) -> str:
    best = ""
    best_key = (-1, -1)
    for part in parts:
        smi = _canon(str(part))
        mol = Chem.MolFromSmiles(smi)
        heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
        key = (heavy, len(smi))
        if key > best_key:
            best_key = key
            best = smi
    return best


def _canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return str(smiles or "")
    return Chem.MolToSmiles(mol, canonical=True)


def _enrich_step(step: dict[str, Any], *, idx: int, evidence: dict[str, Any], min_similarity: float) -> tuple[dict[str, Any], Counter]:
    out = dict(step)
    event = Counter()
    current_transform = str(step.get("transformation_superclass") or "unknown")
    query_fp = _transition_fp(
        step.get("product_smiles") or ((step.get("products") or [""])[0] if isinstance(step.get("products"), list) else ""),
        _main_reactant(step),
    )
    require_transform = current_transform != "unknown"
    match, similarity = _nearest_evidence(
        query_fp,
        evidence,
        preferred_transform=current_transform if require_transform else "",
        require_preferred_transform=require_transform,
    )
    if match is None:
        event["no_evidence_match"] += 1
        out["v4_step_evidence"] = {"matched": False}
        return out, event
    event["matched"] += 1
    if similarity >= min_similarity:
        event["accepted"] += 1
    else:
        event["below_similarity_threshold"] += 1
    accepted = similarity >= min_similarity
    if accepted:
        _fill_if_missing(out, "transformation_superclass", match.get("transformation_superclass"))
        _fill_if_missing(out, "transformation_name", match.get("transformation_name"))
        _fill_if_missing(out, "step_mode", match.get("step_mode"))
        _fill_if_missing(out, "pairwise_mode", match.get("pairwise_mode"))
        if out.get("intermediate_isolated") is None:
            out["intermediate_isolated"] = match.get("intermediate_isolated")
        if not out.get("catalyst_classes"):
            out["catalyst_classes"] = match.get("catalyst_classes") or []
        if not out.get("condition_tokens"):
            out["condition_tokens"] = match.get("condition_tokens") or []
        if not out.get("ec1_values"):
            out["ec1_values"] = match.get("ec1_values") or []
        if not out.get("enzyme_families"):
            out["enzyme_families"] = match.get("enzyme_families") or []
        if not out.get("cofactors"):
            out["cofactors"] = match.get("cofactors") or []
        if not out.get("metal_identities"):
            out["metal_identities"] = match.get("metal_identities") or []
        if not out.get("step_conditions") and match.get("condition_tokens"):
            out["step_conditions"] = _condition_dict_from_tokens(match.get("condition_tokens") or [])
    out["v4_step_evidence"] = {
        "schema_version": ENRICH_SCHEMA_VERSION,
        "matched": True,
        "accepted": accepted,
        "similarity": float(similarity),
        "program_id": match.get("program_id"),
        "doi": match.get("doi"),
        "cascade_id": match.get("cascade_id"),
        "transition_id": match.get("transition_id"),
        "source_transform": match.get("transformation_superclass"),
        "required_transform_match": require_transform,
        "source_catalyst_classes": match.get("catalyst_classes") or [],
        "source_condition_tokens": match.get("condition_tokens") or [],
        "step_index": idx,
    }
    return out, event


def _nearest_evidence(
    query_fp: Any,
    evidence: dict[str, Any],
    *,
    preferred_transform: str,
    require_preferred_transform: bool,
) -> tuple[dict[str, Any] | None, float]:
    if query_fp is None:
        return None, 0.0
    if preferred_transform and require_preferred_transform:
        bucket_names = [preferred_transform]
    else:
        bucket_names = [preferred_transform, ""] if preferred_transform else [""]
    best_item = None
    best_score = -1.0
    for bucket_name in bucket_names:
        bucket = (evidence.get("buckets") or {}).get(bucket_name) or {}
        fps = bucket.get("fps") or []
        items = bucket.get("items") or []
        if not fps:
            continue
        scores = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score > best_score:
            best_score = score
            best_item = items[idx]
    return best_item, max(best_score, 0.0)


def _transition_fp(product_smiles: Any, main_reactant: Any):
    product_fp = _fp(str(product_smiles or ""))
    main_fp = _fp(str(main_reactant or ""))
    if product_fp is None and main_fp is None:
        return None
    if product_fp is None:
        return main_fp
    if main_fp is None:
        return product_fp
    arr_product = np.zeros((2048,), dtype=np.int8)
    arr_main = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(product_fp, arr_product)
    DataStructs.ConvertToNumpyArray(main_fp, arr_main)
    arr = np.maximum(arr_product, arr_main)
    fp = DataStructs.ExplicitBitVect(2048)
    on_bits = np.nonzero(arr)[0]
    for bit in on_bits:
        fp.SetBit(int(bit))
    return fp


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _fill_if_missing(row: dict[str, Any], key: str, value: Any) -> None:
    if row.get(key) in (None, "", "unknown") and value not in (None, ""):
        row[key] = value


def _condition_dict_from_tokens(tokens: list[str]) -> dict[str, Any]:
    out = {}
    for token in tokens:
        if ":" not in str(token):
            continue
        key, value = str(token).split(":", 1)
        out[key] = value
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"unsupported JSON array format: {path}")
        return [row for row in payload if isinstance(row, dict)]
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Route-Pool Step Enrichment From v4",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Evidence Summary",
        "",
        "```json",
        json.dumps(report.get("evidence_summary") or {}, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Enrich sparse route-pool steps with nearest v4 train evidence")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--min-similarity", type=float, default=0.35)
    args = ap.parse_args()
    report = enrich_route_pool_steps_from_v4(
        route_pool=Path(args.route_pool),
        program_manifest=Path(args.program_manifest),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        min_similarity=args.min_similarity,
    )
    print(json.dumps({"counts": report["counts"], "outputs": report["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
