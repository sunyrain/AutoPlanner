"""Augment CascadeBench split rows with structured steps from dataset_v4_release."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem

from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "cascadebench_structured_split_manifest.v1"


def augment_cascadebench_splits_with_v4_steps(
    *,
    split_manifest: Path,
    source_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    source_rows = _read_json_or_jsonl(source_jsonl)
    source_index = _source_index(source_rows)
    outputs: dict[str, str] = {}
    split_reports: dict[str, Any] = {}
    for split in ("all", "train", "val", "test"):
        input_path = (manifest.get("outputs") or {}).get(split)
        if not input_path:
            continue
        rows = _read_json_or_jsonl(Path(input_path))
        augmented, report = _augment_rows(rows, source_index=source_index, source_jsonl=source_jsonl)
        out_path = output_dir / f"v4_trace_{split}_structured.json"
        out_path.write_text(json.dumps(augmented, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs[split] = str(out_path)
        split_reports[split] = report
    out_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "source_split_manifest": str(split_manifest),
            "source_jsonl": str(source_jsonl),
            "output_dir": str(output_dir),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "same split rows as input manifest, with gt_route/steps augmented from dataset_v4_release structured steps",
        },
        "source_manifest": {
            key: value for key, value in manifest.items() if key not in {"outputs"}
        },
        "splits": split_reports,
        "outputs": {
            **outputs,
            "manifest": str(output_dir / "cascadebench_structured_split_manifest.json"),
            "report": str(output_dir / "cascadebench_structured_split_report.md"),
        },
    }
    (output_dir / "cascadebench_structured_split_manifest.json").write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "cascadebench_structured_split_report.md").write_text(_markdown(out_manifest), encoding="utf-8")
    return out_manifest


def _augment_rows(
    rows: list[dict[str, Any]],
    *,
    source_index: dict[str, Any],
    source_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    stats = Counter()
    for row in rows:
        source, match_method = _match_source(row, source_index)
        if source is None:
            stats["unmatched_rows"] += 1
            payload = dict(row)
            payload["structured_step_match"] = {"matched": False, "source_jsonl": str(source_jsonl)}
            out.append(payload)
            continue
        stats["matched_rows"] += 1
        source_steps = [step for step in source.get("steps") or [] if isinstance(step, dict)]
        split_steps = [step for step in row.get("gt_route") or [] if isinstance(step, dict)]
        augmented_steps = []
        for idx, split_step in enumerate(split_steps):
            source_step = _source_step_for(split_step, source_steps, idx)
            augmented_step = _augment_step(split_step, source_step, idx=idx)
            augmented_steps.append(augmented_step)
            stats["steps"] += 1
            if augmented_step.get("product_smiles"):
                stats["steps_with_product"] += 1
            if augmented_step.get("main_reactant"):
                stats["steps_with_main_reactant"] += 1
            if augmented_step.get("transformation_superclass") and augmented_step.get("transformation_superclass") != "unknown":
                stats["steps_with_transform"] += 1
            if augmented_step.get("step_mode") and augmented_step.get("step_mode") != "unknown":
                stats["steps_with_step_mode"] += 1
            if augmented_step.get("pairwise_mode") and augmented_step.get("pairwise_mode") != "unknown":
                stats["steps_with_pairwise_mode"] += 1
        payload = dict(row)
        payload["gt_route"] = augmented_steps
        payload["steps"] = augmented_steps
        payload["structured_step_match"] = {
            "matched": True,
            "match_method": match_method,
            "source_jsonl": str(source_jsonl),
            "source_id": source.get("id"),
            "source_cascade_id": source.get("cascade_id"),
            "source_doi": source.get("doi"),
        }
        out.append(payload)
    return out, dict(stats)


def _source_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_exact: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_doi_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_cascade_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doi = _norm(row.get("doi"))
        cascade_id = _norm(row.get("cascade_id"))
        for target in _target_smiles_values(row):
            target = canonical_smiles(target) or target
            if not target:
                continue
            by_exact[(doi, cascade_id, target)].append(row)
            by_doi_target[(doi, target)].append(row)
            by_cascade_target[(cascade_id, target)].append(row)
    return {
        "by_exact": dict(by_exact),
        "by_doi_target": dict(by_doi_target),
        "by_cascade_target": dict(by_cascade_target),
    }


def _match_source(row: dict[str, Any], source_index: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    doi = _norm(row.get("doi"))
    cascade_id = _norm(row.get("cascade_id"))
    target = canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
    candidates = (source_index.get("by_exact") or {}).get((doi, cascade_id, target)) or []
    method = "doi+cascade+target"
    if not candidates:
        candidates = (source_index.get("by_doi_target") or {}).get((doi, target)) or []
        method = "doi+target"
    if not candidates:
        candidates = (source_index.get("by_cascade_target") or {}).get((cascade_id, target)) or []
        method = "cascade+target"
    if not candidates:
        return None, "unmatched"
    return _best_source_candidate(row, candidates), method


def _best_source_candidate(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    split_rxns = {canonical_reaction_text(step.get("rxn_smiles")) for step in row.get("gt_route") or [] if isinstance(step, dict)}
    split_rxns.discard("")
    depth = int(row.get("depth") or len(row.get("gt_route") or []))

    def score(candidate: dict[str, Any]) -> tuple[int, int, int]:
        steps = [step for step in candidate.get("steps") or [] if isinstance(step, dict)]
        candidate_rxns = {canonical_reaction_text(step.get("rxn_smiles")) for step in steps}
        candidate_rxns.discard("")
        overlap = len(split_rxns & candidate_rxns)
        transform_count = sum(1 for step in steps if step.get("transformation_superclass"))
        return (overlap, -abs(len(steps) - depth), transform_count)

    return max(candidates, key=score)


def _source_step_for(split_step: dict[str, Any], source_steps: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    split_rxn = canonical_reaction_text(split_step.get("rxn_smiles"))
    if split_rxn:
        for source_step in source_steps:
            if canonical_reaction_text(source_step.get("rxn_smiles")) == split_rxn:
                return source_step
    split_index = split_step.get("step_index")
    for source_step in source_steps:
        if split_index is not None and source_step.get("step_index") == split_index:
            return source_step
    if idx < len(source_steps):
        return source_steps[idx]
    return {}


def _augment_step(split_step: dict[str, Any], source_step: dict[str, Any], *, idx: int) -> dict[str, Any]:
    rxn = split_step.get("rxn_smiles") or source_step.get("rxn_smiles") or ""
    source_conditions = source_step.get("step_conditions") or {}
    source_catalysts = [cat for cat in source_step.get("catalyst_components") or [] if isinstance(cat, dict)]
    out = dict(split_step)
    out["rxn_smiles"] = rxn
    out["step_id"] = source_step.get("step_id") or split_step.get("step_id") or f"step_{idx + 1}"
    out["transition_id"] = source_step.get("step_uuid") or source_step.get("step_id") or split_step.get("transition_id") or f"step_{idx + 1}"
    out["product_smiles"] = _source_product(source_step) or _reaction_product(rxn)
    out["reactants"] = _reaction_reactants(rxn)
    out["main_reactant"] = _source_main_reactant(source_step) or _largest_component(out["reactants"])
    out["transformation_name"] = source_step.get("transformation_name") or split_step.get("transformation_name") or "unknown"
    out["transformation_superclass"] = source_step.get("transformation_superclass") or split_step.get("transformation_superclass") or "unknown"
    out["step_mode"] = source_step.get("step_mode") or split_step.get("step_mode") or "unknown"
    out["pairwise_mode"] = source_step.get("pairwise_mode") or split_step.get("pairwise_mode") or "unknown"
    out["intermediate_isolated"] = source_step.get("intermediate_isolated", split_step.get("intermediate_isolated"))
    out["condition"] = split_step.get("condition") or source_conditions
    out["step_conditions"] = source_conditions or split_step.get("condition") or {}
    out["catalyst_classes"] = split_step.get("catalyst_classes") or sorted({str(cat.get("catalyst_class")) for cat in source_catalysts if cat.get("catalyst_class")})
    out["ec_number"] = split_step.get("ec_number") or _first_value(source_catalysts, "ec_number")
    out["enzyme_families"] = sorted({str(cat.get("enzyme_family")) for cat in source_catalysts if cat.get("enzyme_family")})
    out["cofactors"] = sorted({str(cat.get("cofactor_required")) for cat in source_catalysts if cat.get("cofactor_required")})
    out["metal_identities"] = sorted({str(cat.get("metal_identity")) for cat in source_catalysts if cat.get("metal_identity")})
    out["component_names"] = sorted({str(cat.get("component_name_canonical") or cat.get("component_name")) for cat in source_catalysts if cat.get("component_name") or cat.get("component_name_canonical")})
    out["structured_from_v4"] = bool(source_step)
    return out


def _source_product(step: dict[str, Any]) -> str:
    species = [sp for sp in step.get("output_species") or [] if isinstance(sp, dict)]
    preferred_roles = {"target_product", "intermediate", "product"}
    preferred = [sp for sp in species if str(sp.get("role") or "").lower() in preferred_roles and sp.get("smiles")]
    choices = preferred or [sp for sp in species if sp.get("smiles")]
    return _largest_component([str(sp.get("smiles")) for sp in choices])


def _source_main_reactant(step: dict[str, Any]) -> str:
    species = [sp for sp in step.get("input_species") or [] if isinstance(sp, dict)]
    preferred = [sp for sp in species if str(sp.get("role") or "").lower() == "main_substrate" and sp.get("smiles")]
    choices = preferred or [sp for sp in species if sp.get("smiles")]
    return _largest_component([str(sp.get("smiles")) for sp in choices])


def _target_smiles_values(row: dict[str, Any]) -> list[str]:
    raw = row.get("target_product_smiles") or row.get("target_smiles") or ""
    values = []
    for chunk in str(raw).replace("|", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    return values


def _reaction_reactants(rxn: Any) -> list[str]:
    text = str(rxn or "")
    if ">>" not in text:
        return []
    left, _ = text.split(">>", 1)
    return [_canon(part.strip()) for part in left.split(".") if part.strip()]


def _reaction_product(rxn: Any) -> str:
    text = str(rxn or "")
    if ">>" not in text:
        return ""
    _, right = text.split(">>", 1)
    return _largest_component([part.strip() for part in right.split(".") if part.strip()])


def _largest_component(values: list[str]) -> str:
    best = ""
    best_key = (-1, -1)
    for value in values:
        smi = _canon(value)
        mol = Chem.MolFromSmiles(smi)
        heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
        key = (heavy, len(smi))
        if key > best_key:
            best = smi
            best_key = key
    return best


def _canon(smiles: Any) -> str:
    text = str(smiles or "").strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def canonical_reaction_text(rxn: Any) -> str:
    text = str(rxn or "").strip()
    if ">>" not in text:
        return ""
    left, right = text.split(">>", 1)
    return ".".join(sorted(_reaction_reactants(text))) + ">>" + _reaction_product(text)


def _first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        if row.get(key):
            return row.get(key)
    return None


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        return [row for row in payload if isinstance(row, dict)]
    rows = []
    for line in text.splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# CascadeBench Structured Split Augmentation",
        "",
        f"- Generated: `{manifest.get('generated_at')}`",
        f"- Source split manifest: `{(manifest.get('metadata') or {}).get('source_split_manifest')}`",
        f"- Source v4 JSONL: `{(manifest.get('metadata') or {}).get('source_jsonl')}`",
        "",
        "| split | matched rows | unmatched rows | steps | product | main reactant | transform | step mode | pairwise mode |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, report in (manifest.get("splits") or {}).items():
        lines.append(
            "| {split} | {matched} | {unmatched} | {steps} | {product} | {main} | {transform} | {step_mode} | {pairwise} |".format(
                split=split,
                matched=report.get("matched_rows", 0),
                unmatched=report.get("unmatched_rows", 0),
                steps=report.get("steps", 0),
                product=report.get("steps_with_product", 0),
                main=report.get("steps_with_main_reactant", 0),
                transform=report.get("steps_with_transform", 0),
                step_mode=report.get("steps_with_step_mode", 0),
                pairwise=report.get("steps_with_pairwise_mode", 0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment CascadeBench splits with structured v4 step fields")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--source-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    result = augment_cascadebench_splits_with_v4_steps(
        split_manifest=Path(args.split_manifest),
        source_jsonl=Path(args.source_jsonl),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"splits": result["splits"], "outputs": result["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
