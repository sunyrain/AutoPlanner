"""Build a consolidated training-data pack from planner artifacts.

The pack is intentionally model-agnostic. It collects the data needed for the
next training cycle into JSONL files:

* route_value.jsonl: route-level labels and metric-derived features
* candidate_ranking.jsonl: candidate pools with GT/selected weak labels
* skeleton_prior.jsonl: GT and planner skeleton sequences
* failure_diagnosis.jsonl: target-level bottleneck labels

This script does not train a model; it makes the training corpus explicit,
auditable, and reusable.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_side,
    canonical_smiles,
    primary_recovery_bottleneck,
    recovery_bottleneck_labels,
)
from cascade_planner.cascadeboard.value_function import candidate_value_features, metric_value_features


SCHEMA_VERSION = "training_pack.v1"
RDLogger.DisableLog("rdApp.warning")
DEFAULT_INPUTS = [
    "results/v2/live_benchmark_high_recall_full100.json",
    "results/v2/ui_plan_*.json",
]
DEFAULT_BENCHMARKS = ["data/benchmark_v2_100.json"]


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def build_training_pack(
    *,
    input_paths: Iterable[Path],
    benchmark_paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    gt_index = load_gt_index(benchmark_paths)
    rows = {
        "route_value": [],
        "candidate_ranking": [],
        "skeleton_prior": list(gt_index["skeleton_rows"]),
        "failure_diagnosis": [],
    }
    counters = Counter()

    route_seen: set[str] = set()
    candidate_seen: set[str] = set()
    skeleton_seen = {row["skeleton_id"] for row in rows["skeleton_prior"]}
    failure_seen: set[str] = set()

    for path in input_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counters["files_skipped"] += 1
            continue
        counters["files_loaded"] += 1
        for payload in iter_planner_payloads(data, source_path=str(path)):
            target = payload.get("target_smiles") or payload.get("target") or ""
            target_can = canonical_smiles(target)
            target_gt = payload.get("gt_index") or gt_index["by_target"].get(target_can, {})
            route_recovery = payload_recovery(payload)
            failure = failure_row(payload, target_gt=target_gt, source_path=str(path))
            if failure and failure["failure_id"] not in failure_seen:
                rows["failure_diagnosis"].append(failure)
                failure_seen.add(failure["failure_id"])

            for route_index, route in enumerate(payload.get("routes") or []):
                route_row = route_value_row(
                    route,
                    target_smiles=target,
                    route_index=route_index,
                    source_path=str(path),
                    target_gt=target_gt,
                    route_recovery=route_recovery,
                )
                if not route_row or route_row["route_id"] in route_seen:
                    continue
                rows["route_value"].append(route_row)
                route_seen.add(route_row["route_id"])

                skel_row = planner_skeleton_row(route_row, route)
                if skel_row and skel_row["skeleton_id"] not in skeleton_seen:
                    rows["skeleton_prior"].append(skel_row)
                    skeleton_seen.add(skel_row["skeleton_id"])

                for cand_row in candidate_rows_for_route(
                    route,
                    route_row=route_row,
                    target_gt=target_gt,
                    source_path=str(path),
                ):
                    if cand_row["candidate_id"] in candidate_seen:
                        continue
                    rows["candidate_ranking"].append(cand_row)
                    candidate_seen.add(cand_row["candidate_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data_rows in rows.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, data_rows)
        files[name] = str(path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "inputs": [str(p) for p in input_paths],
        "benchmarks": [str(p) for p in benchmark_paths],
        "files": files,
        "counts": {
            "route_value": len(rows["route_value"]),
            "candidate_ranking": len(rows["candidate_ranking"]),
            "skeleton_prior": len(rows["skeleton_prior"]),
            "failure_diagnosis": len(rows["failure_diagnosis"]),
            **dict(counters),
        },
        "quality": quality_summary(rows),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(training_pack_report(manifest), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_gt_index(paths: Iterable[Path]) -> dict[str, Any]:
    by_target: dict[str, dict[str, Any]] = {}
    skeleton_rows: list[dict[str, Any]] = []
    seen_skeletons: set[str] = set()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records = data if isinstance(data, list) else data.get("targets") or data.get("records") or []
        for record in records:
            target = canonical_smiles(record.get("target_smiles") or record.get("target") or "")
            if not target:
                continue
            gt_rows = gt_rows_from_route(record.get("gt_route") or [])
            by_product = {}
            for row in gt_rows:
                by_product.setdefault(row["product"], []).append(row)
            by_target[target] = {
                "target_smiles": record.get("target_smiles") or record.get("target") or "",
                "doi": record.get("doi", ""),
                "cascade_id": record.get("cascade_id", ""),
                "route_domain": record.get("route_domain", ""),
                "operation_mode": record.get("operation_mode", ""),
                "depth": record.get("depth") or len(gt_rows),
                "gt_rows": gt_rows,
                "by_product": by_product,
            }
            skel = gt_skeleton_row(record, gt_rows=gt_rows, source_path=str(path))
            if skel and skel["skeleton_id"] not in seen_skeletons:
                skeleton_rows.append(skel)
                seen_skeletons.add(skel["skeleton_id"])
    return {"by_target": by_target, "skeleton_rows": skeleton_rows}


def gt_rows_from_route(gt_route: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, step in enumerate(gt_route):
        rxn = step.get("rxn_smiles") or step.get("reaction_smiles") or ""
        if ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        product = main_product(rhs)
        reactants = canonical_side(lhs)
        if not product or not reactants:
            continue
        ec = step.get("ec_number") or step.get("ec") or ""
        rows.append({
            "index": idx,
            "product": product,
            "reactants": sorted(reactants),
            "reaction": canonical_reaction(rxn),
            "reaction_type": step.get("transformation") or step.get("reaction_type") or "",
            "ec": ec,
            "ec1": str(ec).split(".", 1)[0] if ec else "",
        })
    return rows


def iter_planner_payloads(data: Any, *, source_path: str) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("planner_output"), dict):
        payload = dict(data["planner_output"])
        payload["target_smiles"] = data.get("target_smiles") or payload.get("target")
        payload["source_target"] = data
        payload["source_path"] = source_path
        target_gt = gt_rows_from_route(data.get("gt_route") or [])
        if target_gt:
            payload["gt_index"] = {
                "target_smiles": payload["target_smiles"],
                "doi": data.get("doi", ""),
                "cascade_id": data.get("cascade_id", ""),
                "route_domain": data.get("route_domain", ""),
                "operation_mode": data.get("operation_mode", ""),
                "depth": data.get("depth") or len(target_gt),
                "gt_rows": target_gt,
                "by_product": rows_by_product(target_gt),
            }
        yield payload
        return

    if isinstance(data, dict) and isinstance(data.get("routes"), list) and (data.get("target") or data.get("target_smiles")):
        payload = dict(data)
        payload["target_smiles"] = data.get("target_smiles") or data.get("target")
        payload["source_path"] = source_path
        yield payload
        return

    if isinstance(data, dict):
        for key in ("targets", "results", "outputs"):
            child = data.get(key)
            if child is not None:
                yield from iter_planner_payloads(child, source_path=source_path)
    elif isinstance(data, list):
        for item in data:
            yield from iter_planner_payloads(item, source_path=source_path)


def route_value_row(
    route: dict[str, Any],
    *,
    target_smiles: str,
    route_index: int,
    source_path: str,
    target_gt: dict[str, Any] | None = None,
    route_recovery: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metrics = route.get("metrics") or {}
    if not metrics:
        return None
    steps = route.get("steps") or []
    route_id = stable_id(source_path, target_smiles, route_index, route_signature(route))
    professional = professional_solved(metrics)
    progressive = bool(metrics.get("progressive_route"))
    filled = bool(metrics.get("filled_route"))
    label = 1.0 if professional else 0.5 if progressive else 0.25 if filled else 0.0
    recovery = route_recovery or {}
    recovery_labels = list(
        recovery.get("recovery_bottleneck_labels")
        or (recovery_bottleneck_labels(recovery) if recovery else [])
    )
    recovery_bottleneck = (
        recovery.get("recovery_bottleneck")
        or (primary_recovery_bottleneck(recovery) if recovery else "")
    )
    return {
        "route_id": route_id,
        "source_path": source_path,
        "target_smiles": target_smiles,
        "doi": (target_gt or {}).get("doi", ""),
        "cascade_id": (target_gt or {}).get("cascade_id", ""),
        "route_domain": (target_gt or {}).get("route_domain", ""),
        "operation_mode": (target_gt or {}).get("operation_mode", ""),
        "route_index": route_index,
        "label": label,
        "label_type": "professional_solved" if professional else "progressive" if progressive else "filled_only" if filled else "unsolved",
        "score": safe_float(route.get("score")),
        "confidence": safe_float(route.get("confidence")),
        "n_steps": route.get("n_steps") or len(steps),
        "type_sequence": [step.get("reaction_type") or "" for step in steps],
        "ec1_sequence": [ec1(step.get("ec")) for step in steps],
        "source_sequence": [step.get("source") or "" for step in steps],
        "recovery_bottleneck": recovery_bottleneck,
        "recovery_bottleneck_labels": recovery_labels,
        "recovery_summary": {
            key: recovery.get(key)
            for key in (
                "exact_route_reaction_match_any",
                "exact_reaction_in_route_pool",
                "candidate_exact_reaction_in_pool",
                "gt_reactant_in_route_pool",
                "candidate_gt_reactant_in_pool",
            )
            if key in recovery
        },
        "metrics_summary": compact_metrics(metrics),
        "features": metric_value_features(metrics),
        "step_reactions": [step.get("reaction_smiles") or "" for step in steps],
    }


def candidate_rows_for_route(
    route: dict[str, Any],
    *,
    route_row: dict[str, Any],
    target_gt: dict[str, Any],
    source_path: str,
) -> Iterable[dict[str, Any]]:
    route_label = float(route_row.get("label") or 0.0)
    by_product = target_gt.get("by_product") or {}
    for step_index, step in enumerate(route.get("steps") or []):
        product = canonical_smiles(step.get("product"))
        if not product:
            continue
        selected_set = reactant_set_from_step(step)
        gt_rows = by_product.get(product) or []
        for rank, cand in enumerate((step.get("candidate_pool") or {}).get("top_candidates") or [], start=1):
            cand_set = reactant_set_from_candidate(cand)
            cand_rxn = canonical_reaction(cand.get("rxn_smiles") or cand.get("reaction_smiles") or "")
            exact_gt_reaction = any(cand_rxn and cand_rxn == gt.get("reaction") for gt in gt_rows)
            exact_gt_reactants = any(cand_set and cand_set == set(gt.get("reactants") or []) for gt in gt_rows)
            selected_exact = bool(cand_set and selected_set and cand_set == selected_set)
            if exact_gt_reaction or exact_gt_reactants:
                label = 1.0
                label_type = "benchmark_exact"
                weight = 2.0
            elif selected_exact and route_label >= 0.5:
                label = 0.75
                label_type = "planner_selected_positive"
                weight = 1.0
            elif selected_exact:
                label = 0.5
                label_type = "planner_selected_weak"
                weight = 0.5
            else:
                label = 0.0
                label_type = "negative"
                weight = 1.0
            candidate = normalize_candidate(cand)
            yield {
                "candidate_id": stable_id(source_path, route_row["route_id"], step_index, rank, cand_rxn, sorted(cand_set)),
                "route_id": route_row["route_id"],
                "source_path": source_path,
                "target_smiles": route_row["target_smiles"],
                "product": step.get("product") or "",
                "step_index": step_index,
                "rank": rank,
                "label": label,
                "label_type": label_type,
                "weight": weight,
                "exact_gt_reaction": bool(exact_gt_reaction),
                "exact_gt_reactants": bool(exact_gt_reactants),
                "selected_exact": selected_exact,
                "gt_available": bool(gt_rows),
                "target_recovery_bottleneck": route_row.get("recovery_bottleneck", ""),
                "target_recovery_bottleneck_labels": route_row.get("recovery_bottleneck_labels") or [],
                "candidate": candidate,
                "features": candidate_value_features(step.get("product") or "", candidate),
            }


def gt_skeleton_row(record: dict[str, Any], *, gt_rows: list[dict[str, Any]], source_path: str) -> dict[str, Any] | None:
    target = record.get("target_smiles") or record.get("target") or ""
    if not target or not gt_rows:
        return None
    type_sequence = [row.get("reaction_type") or "" for row in gt_rows]
    ec1_sequence = [row.get("ec1") or "" for row in gt_rows]
    return {
        "skeleton_id": stable_id("gt", target, type_sequence, ec1_sequence, record.get("doi"), record.get("cascade_id")),
        "source": "benchmark_gt",
        "source_path": source_path,
        "target_smiles": target,
        "doi": record.get("doi", ""),
        "cascade_id": record.get("cascade_id", ""),
        "route_domain": record.get("route_domain", ""),
        "operation_mode": record.get("operation_mode", ""),
        "depth": record.get("depth") or len(gt_rows),
        "type_sequence": type_sequence,
        "ec1_sequence": ec1_sequence,
        "label": 1.0,
    }


def planner_skeleton_row(route_row: dict[str, Any], route: dict[str, Any]) -> dict[str, Any] | None:
    if not route_row.get("type_sequence"):
        return None
    return {
        "skeleton_id": stable_id("planner", route_row["route_id"], route_row.get("type_sequence"), route_row.get("ec1_sequence")),
        "source": "planner_route",
        "source_path": route_row["source_path"],
        "target_smiles": route_row["target_smiles"],
        "doi": route_row.get("doi", ""),
        "cascade_id": route_row.get("cascade_id", ""),
        "route_domain": route_row.get("route_domain", ""),
        "operation_mode": route_row.get("operation_mode", ""),
        "depth": route_row.get("n_steps"),
        "type_sequence": route_row.get("type_sequence") or [],
        "ec1_sequence": route_row.get("ec1_sequence") or [],
        "label": route_row.get("label"),
        "label_type": route_row.get("label_type"),
        "metrics_summary": route_row.get("metrics_summary") or {},
        "route_id": route_row["route_id"],
    }


def failure_row(payload: dict[str, Any], *, target_gt: dict[str, Any], source_path: str) -> dict[str, Any] | None:
    target = payload.get("target_smiles") or payload.get("target") or ""
    if not target:
        return None
    labels = set(payload.get("failure_diagnosis") or [])
    source_target = payload.get("source_target") or {}
    metrics = source_target.get("metrics") or {}
    recovery = payload_recovery(payload)
    if recovery:
        for label in recovery.get("recovery_bottleneck_labels") or recovery_bottleneck_labels(recovery):
            if label != "recovered_exact_route":
                labels.add(label)
    if metrics:
        if metrics.get("strict_stock_solve_any") is False:
            labels.add("stock_dead_end")
        if metrics.get("condition_window_success_any") is False:
            labels.add("condition_failure")
        if metrics.get("cascade_compatibility_success_any") is False:
            labels.add("compatibility_failure")
    routes = payload.get("routes") or []
    if routes and not any(professional_solved((route.get("metrics") or {})) for route in routes):
        labels.add("no_professional_solved_route")
    if not routes:
        labels.add("no_route_returned")
    return {
        "failure_id": stable_id(source_path, target, sorted(labels), len(routes)),
        "source_path": source_path,
        "target_smiles": target,
        "doi": target_gt.get("doi", ""),
        "cascade_id": target_gt.get("cascade_id", ""),
        "route_domain": target_gt.get("route_domain", source_target.get("route_domain", "")),
        "depth": target_gt.get("depth", source_target.get("depth")),
        "n_routes": len(routes),
        "labels": sorted(labels),
        "has_failure_label": bool(labels),
        "route_recovery": recovery,
        "metrics": metrics,
    }


def payload_recovery(payload: dict[str, Any]) -> dict[str, Any]:
    source_target = payload.get("source_target") or {}
    recovery = source_target.get("route_recovery") or payload.get("route_recovery") or {}
    return dict(recovery) if isinstance(recovery, dict) else {}


def quality_summary(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cand = rows["candidate_ranking"]
    route = rows["route_value"]
    failure = rows["failure_diagnosis"]
    return {
        "route_labels": dict(Counter(row.get("label_type") for row in route)),
        "candidate_labels": dict(Counter(row.get("label_type") for row in cand)),
        "candidate_gt_available": sum(1 for row in cand if row.get("gt_available")),
        "candidate_exact_gt": sum(1 for row in cand if row.get("exact_gt_reaction") or row.get("exact_gt_reactants")),
        "failure_labels": dict(Counter(label for row in failure for label in row.get("labels") or [])),
    }


def training_pack_report(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    quality = manifest.get("quality") or {}
    lines = [
        "# Training Data Pack",
        "",
        f"Schema: `{manifest.get('schema_version')}`",
        f"Generated: `{manifest.get('generated_at')}`",
        "",
        "## Files",
        "",
    ]
    for name, path in (manifest.get("files") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend([
        "",
        "## Counts",
        "",
        f"- route value rows: `{counts.get('route_value', 0)}`",
        f"- candidate ranking rows: `{counts.get('candidate_ranking', 0)}`",
        f"- skeleton prior rows: `{counts.get('skeleton_prior', 0)}`",
        f"- failure diagnosis rows: `{counts.get('failure_diagnosis', 0)}`",
        "",
        "## Quality",
        "",
        f"- route labels: `{quality.get('route_labels', {})}`",
        f"- candidate labels: `{quality.get('candidate_labels', {})}`",
        f"- candidate rows with GT available: `{quality.get('candidate_gt_available', 0)}`",
        f"- candidate exact GT rows: `{quality.get('candidate_exact_gt', 0)}`",
        f"- failure labels: `{quality.get('failure_labels', {})}`",
        "",
        "## Caveat",
        "",
        "Planner-selected labels are weak supervision. Benchmark-exact labels should be weighted higher during model training.",
        "",
    ])
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def rows_by_product(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["product"], []).append(row)
    return out


def route_signature(route: dict[str, Any]) -> list[str]:
    return [
        canonical_reaction(step.get("reaction_smiles") or "")
        or f"{canonical_smiles(step.get('product'))}>{canonical_smiles(step.get('main_reactant'))}"
        for step in route.get("steps") or []
    ]


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    progress = metrics.get("retrosynthesis_progress") or {}
    natural = metrics.get("route_naturalness") or {}
    compat = metrics.get("cascade_compatibility") or {}
    return {
        "filled_route": metrics.get("filled_route"),
        "progressive_route": metrics.get("progressive_route"),
        "route_solved": metrics.get("route_solved"),
        "professional_solved": professional_solved(metrics),
        "strict_stock_solve": metrics.get("strict_stock_solve"),
        "main_chain_reduction": progress.get("main_chain_reduction"),
        "largest_leaf_reduction": progress.get("largest_leaf_reduction"),
        "naturalness_score": natural.get("naturalness_score"),
        "compatibility_success": compat.get("cascade_compatibility_success"),
        "issues": list(compat.get("issues") or []),
    }


def professional_solved(metrics: dict[str, Any]) -> bool:
    if "professional_solved" in metrics:
        return bool(metrics.get("professional_solved"))
    return bool(metrics.get("route_solved") and metrics.get("progressive_route"))


def main_product(rhs: str) -> str:
    products = canonical_side(rhs)
    if not products:
        return ""
    return max(products, key=lambda smi: (heavy_atoms(smi), smi))


def reactant_set_from_step(step: dict[str, Any]) -> set[str]:
    parts = []
    if step.get("main_reactant"):
        parts.append(step["main_reactant"])
    parts.extend(step.get("aux_reactants") or [])
    return {canonical_smiles(smi) for smi in parts if canonical_smiles(smi)}


def reactant_set_from_candidate(candidate: dict[str, Any]) -> set[str]:
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or ""
    if ">>" in rxn:
        lhs = rxn.split(">>", 1)[0]
        values = set(canonical_side(lhs))
        if values:
            return values
    parts = []
    if candidate.get("main_reactant"):
        parts.append(candidate["main_reactant"])
    parts.extend(candidate.get("aux_reactants") or [])
    return {canonical_smiles(smi) for smi in parts if canonical_smiles(smi)}


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    if "rxn_smiles" not in out and out.get("reaction_smiles"):
        out["rxn_smiles"] = out["reaction_smiles"]
    if "type" not in out and out.get("reaction_type"):
        out["type"] = out["reaction_type"]
    evidence = out.get("evidence") or {}
    extra_keys = [
        "enzyme_uid",
        "catalyst",
        "T",
        "pH",
        "solvent",
        "condition_match",
        "literature_precedent",
        "uniprot_accession",
        "organism",
        "doi",
        "pmid",
        "cofactor",
        "cofactor_regeneration_mode",
        "substrate_similarity",
        "reaction_center_similarity",
        "biocatalyst_format",
        "engineering_status",
        "source_db",
    ]
    extras = {}
    for key in extra_keys:
        value = out.get(key, evidence.get(key))
        if value not in (None, "", [], {}):
            extras[key] = safe_float(value) if key in {"T", "pH", "substrate_similarity", "reaction_center_similarity"} else value
    return {
        "main_reactant": out.get("main_reactant", ""),
        "aux_reactants": list(out.get("aux_reactants") or []),
        "rxn_smiles": out.get("rxn_smiles", ""),
        "type": out.get("type", ""),
        "ec": out.get("ec", ""),
        "source": out.get("source", ""),
        "score": safe_float(out.get("score")),
        "value_score": safe_float(out.get("value_score")),
        "value_probability": safe_float(out.get("value_probability")),
        "evidence": evidence,
        **extras,
    }


def ec1(ec: Any) -> str:
    value = str(ec or "")
    return value.split(".", 1)[0] if value else ""


def heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def stable_id(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build consolidated AutoPlanner training data pack")
    ap.add_argument("--input", action="append", default=None, help="Planner artifact JSON path or glob")
    ap.add_argument("--benchmark", action="append", default=None, help="Benchmark JSON path with GT routes")
    ap.add_argument("--output-dir", default="results/shared/training_pack/current")
    args = ap.parse_args()

    input_paths = expand_paths(args.input or DEFAULT_INPUTS)
    benchmark_paths = expand_paths(args.benchmark or DEFAULT_BENCHMARKS)
    manifest = build_training_pack(
        input_paths=input_paths,
        benchmark_paths=benchmark_paths,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({
        "output_dir": manifest["output_dir"],
        "counts": manifest["counts"],
        "quality": manifest["quality"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
