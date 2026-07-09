"""Build CascadeProgramPack from dataset_v4_release.

The pack preserves route, step, catalyst, condition, and adjacency context that
is intentionally absent from the older transition-only coverage audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles


PROGRAM_PACK_SCHEMA_VERSION = "cascade_program_pack.v1"


def build_cascade_program_pack(
    *,
    v4_jsonl: Path,
    split_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        split: _split_keys(split_manifest, split)
        for split in ("train", "val", "test")
    }
    programs_by_split = {split: [] for split in ("train", "val", "test")}
    skipped = Counter()
    for raw in _iter_jsonl(v4_jsonl):
        key = (_norm(raw.get("doi")), _norm(raw.get("cascade_id")))
        split = next((name for name, keys in split_rows.items() if key in keys), None)
        if split is None:
            skipped["not_in_manifest"] += 1
            continue
        program = _program_from_raw(raw)
        if not program.get("steps"):
            skipped["no_usable_steps"] += 1
            continue
        programs_by_split[split].append(program)

    graph = _evidence_graph(programs_by_split["train"])
    outputs = {
        "train": output_dir / "cascade_programs_train.jsonl",
        "val": output_dir / "cascade_programs_val.jsonl",
        "test": output_dir / "cascade_programs_test.jsonl",
        "train_evidence_graph": output_dir / "cascade_evidence_graph_train.json",
        "manifest": output_dir / "cascade_program_pack_manifest.json",
        "report": output_dir / "cascade_program_pack_report.md",
    }
    for split, rows in programs_by_split.items():
        _write_jsonl(outputs[split], rows)
    outputs["train_evidence_graph"].write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": PROGRAM_PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "split_manifest": str(split_manifest),
            "output_dir": str(output_dir),
        },
        "counts": {
            "programs": {split: len(rows) for split, rows in programs_by_split.items()},
            "steps": {split: sum(len(row.get("steps") or []) for row in rows) for split, rows in programs_by_split.items()},
            "adjacencies_train": len(graph.get("adjacency_keys") or {}),
            "skipped": dict(skipped),
        },
        "split_summaries": {split: _program_summary(rows) for split, rows in programs_by_split.items()},
        "evidence_graph_summary": graph.get("summary"),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["report"].write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def _program_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    raw_steps = [step for step in raw.get("steps") or [] if isinstance(step, dict)]
    raw_steps.sort(key=lambda step: int(step.get("step_index") or 0))
    target_raw = _clean_smiles(raw.get("target_product_smiles"))
    target = canonical_smiles(target_raw) or target_raw
    steps = []
    previous = None
    for pos, step in enumerate(raw_steps):
        raw_rxn = str(step.get("rxn_smiles") or "")
        rxn = _canonical_clean_reaction(raw_rxn)
        reactants, products = _reaction_sides(rxn)
        product = products[0] if products else ""
        reactants = sorted(reactants)
        catalysts = [_compact_catalyst(cat) for cat in step.get("catalyst_components") or [] if isinstance(cat, dict)]
        conditions = _compact_conditions(step.get("step_conditions") or {})
        step_row = {
            "transition_id": _stable_id(raw.get("doi"), raw.get("cascade_id"), step.get("step_id"), step.get("rxn_smiles")),
            "step_id": step.get("step_id"),
            "step_index": step.get("step_index"),
            "step_pos": pos,
            "remaining_steps": max(0, len(raw_steps) - pos - 1),
            "rxn_smiles": rxn,
            "product_smiles": product,
            "reactants": reactants,
            "main_reactant": _largest_smiles(reactants),
            "transformation_name": step.get("transformation_name"),
            "transformation_superclass": step.get("transformation_superclass") or "unknown",
            "previous_transformation_superclass": (previous or {}).get("transformation_superclass", ""),
            "next_transformation_superclass": "",
            "step_mode": step.get("step_mode") or "unknown",
            "pairwise_mode": step.get("pairwise_mode") or "unknown",
            "intermediate_isolated": step.get("intermediate_isolated"),
            "step_role": step.get("step_role"),
            "step_notes": step.get("step_notes"),
            "evidence_quote": step.get("evidence_quote"),
            "conditions": conditions,
            "condition_tokens": _condition_tokens(conditions),
            "catalysts": catalysts,
            "catalyst_classes": sorted({cat.get("catalyst_class") for cat in catalysts if cat.get("catalyst_class")}),
            "ec1_values": sorted({str(cat.get("ec_number") or "").split(".", 1)[0] for cat in catalysts if cat.get("ec_number")}),
            "enzyme_families": sorted({cat.get("enzyme_family") for cat in catalysts if cat.get("enzyme_family")}),
            "cofactors": sorted({cat.get("cofactor_required") for cat in catalysts if cat.get("cofactor_required")}),
            "metal_identities": sorted({cat.get("metal_identity") for cat in catalysts if cat.get("metal_identity")}),
        }
        steps.append(step_row)
        previous = step_row
    for idx, step in enumerate(steps[:-1]):
        step["next_transformation_superclass"] = steps[idx + 1].get("transformation_superclass") or ""

    return {
        "program_id": _stable_id(raw.get("doi"), raw.get("cascade_id"), target),
        "doi": raw.get("doi"),
        "cascade_id": raw.get("cascade_id"),
        "target_smiles": target,
        "starting_material_smiles": _canonical_clean_smiles(raw.get("starting_material_smiles")),
        "cascade_type": raw.get("cascade_type") or raw.get("route_domain") or "unknown",
        "quality_tier": raw.get("quality_tier"),
        "quality_score": raw.get("quality_score"),
        "publish_year": raw.get("publish_year"),
        "compatibility": _compact_compatibility(raw.get("compatibility") or {}),
        "route_conditions": _compact_conditions(raw.get("conditions") or {}),
        "route_condition_tokens": _condition_tokens(_compact_conditions(raw.get("conditions") or {})),
        "catalyst_combination_summary": raw.get("catalyst_combination_summary"),
        "overall_yield": raw.get("overall_yield"),
        "overall_ee": raw.get("overall_ee"),
        "overall_dr": raw.get("overall_dr"),
        "total_steps": len(steps),
        "steps": steps,
        "adjacencies": _adjacency_rows(raw, steps),
    }


def _adjacency_rows(raw: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in zip(steps, steps[1:]):
        rows.append(
            {
                "adjacency_id": _stable_id(raw.get("doi"), raw.get("cascade_id"), left.get("transition_id"), right.get("transition_id")),
                "left_transition_id": left.get("transition_id"),
                "right_transition_id": right.get("transition_id"),
                "left_transform": left.get("transformation_superclass") or "",
                "right_transform": right.get("transformation_superclass") or "",
                "transform_pair": f"{left.get('transformation_superclass') or ''}->{right.get('transformation_superclass') or ''}",
                "left_catalyst_classes": left.get("catalyst_classes") or [],
                "right_catalyst_classes": right.get("catalyst_classes") or [],
                "catalyst_class_pair": ".".join(left.get("catalyst_classes") or ["unknown"])
                + "->"
                + ".".join(right.get("catalyst_classes") or ["unknown"]),
                "pairwise_mode": right.get("pairwise_mode") or left.get("pairwise_mode") or "unknown",
                "intermediate_isolated": left.get("intermediate_isolated"),
                "left_condition_tokens": left.get("condition_tokens") or [],
                "right_condition_tokens": right.get("condition_tokens") or [],
                "condition_overlap": len(set(left.get("condition_tokens") or []) & set(right.get("condition_tokens") or [])),
            }
        )
    return rows


def _evidence_graph(programs: list[dict[str, Any]]) -> dict[str, Any]:
    transform_keys = Counter()
    adjacency_keys = Counter()
    catalyst_keys = Counter()
    condition_keys = Counter()
    hidden_keys = Counter()
    route_type_keys = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for program in programs:
        route_type = str(program.get("cascade_type") or "unknown")
        route_type_keys[route_type] += 1
        for step in program.get("steps") or []:
            transform = str(step.get("transformation_superclass") or "unknown")
            transform_keys[transform] += 1
            for cls in step.get("catalyst_classes") or ["unknown"]:
                catalyst_keys[f"{transform}|{cls}"] += 1
            for token in step.get("condition_tokens") or []:
                condition_keys[f"{transform}|{token}"] += 1
            if step.get("intermediate_isolated") is False:
                hidden_keys[transform] += 1
        for adj in program.get("adjacencies") or []:
            key = str(adj.get("transform_pair") or "")
            adjacency_keys[key] += 1
            if len(examples[key]) < 5:
                examples[key].append(
                    {
                        "program_id": program.get("program_id"),
                        "doi": program.get("doi"),
                        "cascade_id": program.get("cascade_id"),
                        "target_smiles": program.get("target_smiles"),
                        "catalyst_class_pair": adj.get("catalyst_class_pair"),
                        "pairwise_mode": adj.get("pairwise_mode"),
                        "intermediate_isolated": adj.get("intermediate_isolated"),
                    }
                )
    return {
        "schema_version": "cascade_evidence_graph.v1",
        "summary": {
            "programs": len(programs),
            "steps": sum(len(program.get("steps") or []) for program in programs),
            "adjacencies": sum(len(program.get("adjacencies") or []) for program in programs),
            "unique_transform_keys": len(transform_keys),
            "unique_adjacency_keys": len(adjacency_keys),
        },
        "transform_keys": dict(transform_keys),
        "adjacency_keys": dict(adjacency_keys),
        "catalyst_transform_keys": dict(catalyst_keys),
        "condition_transform_keys": dict(condition_keys),
        "hidden_intermediate_transform_keys": dict(hidden_keys),
        "route_type_keys": dict(route_type_keys),
        "adjacency_examples": dict(examples),
    }


def _compact_catalyst(cat: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "catalyst_class",
        "component_name",
        "component_name_canonical",
        "ec_number",
        "enzyme_family",
        "biocatalyst_format",
        "cofactor_required",
        "cofactor_regeneration_mode",
        "metal_identity",
        "organism",
        "support_material",
        "engineering_status",
        "catalyst_phase",
        "uniprot_id",
    ]
    return {key: cat.get(key) for key in keep if cat.get(key) not in (None, "")}


def _compact_conditions(cond: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "atmosphere",
        "buffer_concentration_mM",
        "buffer_name",
        "cosolvent",
        "cosolvent_smiles",
        "ph",
        "reaction_time_h",
        "solvent",
        "solvent_smiles",
        "substrate_concentration_mM",
        "temperature_c",
        "additives_text",
        "mixing_method",
        "reactor_type",
    ]
    return {key: cond.get(key) for key in keep if cond.get(key) not in (None, "", "not_specified")}


def _compact_compatibility(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "compatibility_label": row.get("compatibility_label"),
        "evidence_strength": row.get("evidence_strength"),
        "compatibility_basis": row.get("compatibility_basis"),
        "issue_types": row.get("issue_types") or [],
        "failure_modes_discussed": row.get("failure_modes_discussed") or [],
        "mitigation_strategies": row.get("mitigation_strategies") or [],
        "inhibition_profile": row.get("inhibition_profile") or {},
        "stability_profile": row.get("stability_profile") or {},
    }


def _condition_tokens(cond: dict[str, Any]) -> list[str]:
    tokens = []
    for key in ("solvent", "cosolvent", "buffer_name", "atmosphere", "reactor_type", "mixing_method"):
        value = _norm(cond.get(key))
        if value:
            tokens.append(f"{key}:{value}")
    ph = _float(cond.get("ph"))
    if ph is not None:
        tokens.append("ph:acidic" if ph < 6.5 else "ph:basic" if ph > 8.0 else "ph:neutral")
    temp = _float(cond.get("temperature_c"))
    if temp is not None:
        tokens.append("temp:cold" if temp < 15 else "temp:warm" if temp <= 45 else "temp:hot")
    return sorted(set(tokens))


def _program_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "programs": len(rows),
        "steps": sum(len(row.get("steps") or []) for row in rows),
        "adjacencies": sum(len(row.get("adjacencies") or []) for row in rows),
        "cascade_type_counts": dict(Counter(row.get("cascade_type") for row in rows)),
        "compatibility_label_counts": dict(Counter((row.get("compatibility") or {}).get("compatibility_label") for row in rows)),
        "quality_tier_counts": dict(Counter(row.get("quality_tier") for row in rows)),
    }


def _reaction_sides(rxn: str) -> tuple[list[str], list[str]]:
    if not rxn or ">>" not in rxn:
        return [], []
    lhs, rhs = rxn.split(">>", 1)
    return _canonical_side_clean(lhs), _canonical_side_clean(rhs)


def _canonical_clean_reaction(rxn: str) -> str:
    if not rxn or ">>" not in rxn:
        return ""
    lhs, rhs = rxn.split(">>", 1)
    left = _canonical_side_clean(lhs)
    right = _canonical_side_clean(rhs)
    if not left or not right:
        return ""
    return ".".join(sorted(left)) + ">>" + ".".join(sorted(right))


def _canonical_side_clean(side: str) -> list[str]:
    values = []
    for part in str(side or "").replace(";", ".").split("."):
        cleaned = _clean_smiles(part)
        if not cleaned:
            continue
        values.append(canonical_smiles(cleaned) or cleaned)
    return sorted(values)


def _clean_smiles(value: Any) -> str:
    return str(value or "").strip().strip(";").strip()


def _canonical_clean_smiles(value: Any) -> str:
    parts = _canonical_side_clean(str(value or ""))
    return ".".join(parts)


def _largest_smiles(values: list[str]) -> str:
    return max(values, key=len) if values else ""


def _split_keys(path: Path, split: str) -> set[tuple[str, str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split_path = (manifest.get("outputs") or {}).get(split)
    if not split_path:
        raise ValueError(f"split manifest has no output for split {split!r}: {path}")
    rows = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return {
        (_norm(row.get("doi")), _norm(row.get("cascade_id")))
        for row in rows
        if isinstance(row, dict)
    }


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_id(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# CascadeProgramPack",
        "",
        f"- schema: `{manifest.get('schema_version')}`",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(manifest.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
    ]
    for key, path in (manifest.get("outputs") or {}).items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CascadeProgramPack from dataset_v4_release")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    manifest = build_cascade_program_pack(
        v4_jsonl=Path(args.v4_jsonl),
        split_manifest=Path(args.split_manifest),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"counts": manifest["counts"], "evidence_graph_summary": manifest["evidence_graph_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
