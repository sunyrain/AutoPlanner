"""Train a generic cascade subgoal-evidence scorer.

This is the learned counterpart to ``audit_cascade_subgoal_discovery``.  It
does not learn statin-specific rules and it does not judge full route quality.
The task is narrower: given a target/route subgoal and a v4 cascade evidence
record retrieved from the train split, rank evidence that looks like a
cascade-supported precedent above common-fragment or wrong-context matches.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker, early_stopping
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.v4_product_value import canonical_smiles, stable_id
from cascade_planner.eval.audit_cascade_subgoal_discovery import _fragments, _mol_props


RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "cascade_subgoal_scorer.v1"
TOP_KS = (1, 3, 5, 10)
QUERY_ROLES = ("program_target", "target_fragment", "step_product", "step_product_fragment")
EVIDENCE_ROLES = ("program_target", "step_product", "step_reactant")
QUALITY_TIERS = ("gold", "silver")
EVIDENCE_STRENGTHS = ("strong_process_evidence", "process_evidence", "unclear", "")


@dataclass
class SubgoalDataset:
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    group_ids: list[str]
    group_sizes: list[int]
    feature_names: list[str]


def train_cascade_subgoal_scorer(
    *,
    program_manifest: Path,
    output_dir: Path,
    min_query_heavy_atoms: int = 8,
    min_evidence_heavy_atoms: int = 7,
    max_fragments_per_molecule: int = 8,
    candidates_per_query: int = 80,
    positive_similarity: float = 0.55,
    strong_positive_similarity: float = 0.72,
    n_estimators: int = 260,
    learning_rate: float = 0.04,
    seed: int = 42,
    apply_audit_json: Path | None = None,
    apply_output_json: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    programs = _load_program_splits(program_manifest)
    train_evidence = _evidence_items(programs["train"], min_heavy_atoms=min_evidence_heavy_atoms)
    if not train_evidence:
        raise ValueError("no train evidence items available")
    schema = _build_feature_schema(programs["train"], train_evidence)
    split_queries = {
        split: _query_items(
            rows,
            min_heavy_atoms=min_query_heavy_atoms,
            max_fragments_per_molecule=max_fragments_per_molecule,
        )
        for split, rows in programs.items()
    }
    split_rows = {
        split: _candidate_rows(
            queries,
            train_evidence,
            schema=schema,
            candidates_per_query=candidates_per_query,
            positive_similarity=positive_similarity,
            strong_positive_similarity=strong_positive_similarity,
            exclude_same_program=(split == "train"),
            require_positive=(split == "train"),
        )
        for split, queries in split_queries.items()
    }
    train = _dataset(split_rows["train"], schema)
    val = _dataset(split_rows["val"], schema)
    test = _dataset(split_rows["test"], schema)
    if not train.rows or not val.rows or not test.rows:
        raise ValueError("subgoal scorer requires non-empty train/val/test rows")

    model_specs = _model_specs(train.feature_names)
    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for name, indices in model_specs.items():
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=15,
            min_child_samples=18,
            subsample=0.85,
            colsample_bytree=0.90,
            reg_lambda=3.0,
            reg_alpha=0.1,
            random_state=int(seed),
            verbose=-1,
        )
        model.fit(
            train.x[:, indices],
            train.y,
            group=train.group_sizes,
            eval_set=[(val.x[:, indices], val.y)],
            eval_group=[val.group_sizes],
            eval_at=list(TOP_KS),
            callbacks=[early_stopping(30, verbose=False)],
        )
        models[name] = model
        reports[name] = {
            "train": _ranking_metrics(train, model.predict(train.x[:, indices], num_iteration=model.best_iteration_)),
            "val": _ranking_metrics(val, model.predict(val.x[:, indices], num_iteration=model.best_iteration_)),
            "test": _ranking_metrics(test, model.predict(test.x[:, indices], num_iteration=model.best_iteration_)),
            "feature_names": [train.feature_names[idx] for idx in indices],
            "best_iteration": int(model.best_iteration_ or 0),
        }

    baselines = _baseline_reports(train, val, test)
    applied = None
    model_bundle = {
        "schema_version": SCHEMA_VERSION,
        "models": models,
        "feature_schema": schema,
        "feature_names": train.feature_names,
        "model_specs": {name: [train.feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
        "metadata": {
            "program_manifest": str(program_manifest),
            "min_query_heavy_atoms": min_query_heavy_atoms,
            "min_evidence_heavy_atoms": min_evidence_heavy_atoms,
            "max_fragments_per_molecule": max_fragments_per_molecule,
            "candidates_per_query": candidates_per_query,
            "positive_similarity": positive_similarity,
            "strong_positive_similarity": strong_positive_similarity,
            "seed": seed,
        },
    }
    if apply_audit_json and apply_output_json:
        applied = apply_scorer_to_subgoal_audit(
            audit_json=apply_audit_json,
            output_json=apply_output_json,
            model_bundle=model_bundle,
            preferred_model="structure_metadata_no_rank",
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "min_query_heavy_atoms": min_query_heavy_atoms,
            "min_evidence_heavy_atoms": min_evidence_heavy_atoms,
            "max_fragments_per_molecule": max_fragments_per_molecule,
            "candidates_per_query": candidates_per_query,
            "positive_similarity": positive_similarity,
            "strong_positive_similarity": strong_positive_similarity,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "generic subgoal-evidence ranking; train evidence only; "
                "no target-name, drug-family, or statin-specific labels"
            ),
            "label_semantics": (
                "self-supervised analog support: high structural similarity to train evidence "
                "plus compatible v4 transform/route metadata; not a route feasibility label"
            ),
        },
        "counts": {
            "train_programs": len(programs["train"]),
            "val_programs": len(programs["val"]),
            "test_programs": len(programs["test"]),
            "train_evidence": len(train_evidence),
            "train_queries": len(split_queries["train"]),
            "val_queries": len(split_queries["val"]),
            "test_queries": len(split_queries["test"]),
            "train_rows": len(train.rows),
            "val_rows": len(val.rows),
            "test_rows": len(test.rows),
            "train_groups": len(train.group_sizes),
            "val_groups": len(val.group_sizes),
            "test_groups": len(test.group_sizes),
            "train_positive_groups": _positive_group_count(train),
            "val_positive_groups": _positive_group_count(val),
            "test_positive_groups": _positive_group_count(test),
            "train_relevance_rows": int(np.sum(train.y > 0)),
            "val_relevance_rows": int(np.sum(val.y > 0)),
            "test_relevance_rows": int(np.sum(test.y > 0)),
        },
        "baselines": baselines,
        "models": reports,
        "decision": _decision(baselines, reports),
        "feature_schema": {
            "feature_names": train.feature_names,
            "model_specs": {name: [train.feature_names[idx] for idx in indices] for name, indices in model_specs.items()},
            **schema,
        },
        "applied_audit": applied,
    }
    with (output_dir / "cascade_subgoal_scorer.pkl").open("wb") as fh:
        pickle.dump(model_bundle, fh)
    (output_dir / "cascade_subgoal_scorer_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "cascade_subgoal_scorer_report.md").write_text(_markdown(result), encoding="utf-8")
    _write_jsonl(output_dir / "subgoal_scorer_test_rankings.jsonl", _ranked_rows(test, _test_scores(test, models, model_specs), top_n=20))
    return result


def apply_scorer_to_subgoal_audit(
    *,
    audit_json: Path,
    output_json: Path,
    model_bundle: dict[str, Any] | None = None,
    model_path: Path | None = None,
    preferred_model: str = "structure_metadata",
) -> dict[str, Any]:
    if model_bundle is None:
        if model_path is None:
            raise ValueError("model_bundle or model_path is required")
        with Path(model_path).open("rb") as fh:
            model_bundle = pickle.load(fh)
    model = (model_bundle.get("models") or {}).get(preferred_model)
    if model is None:
        raise ValueError(f"model {preferred_model!r} not found in model bundle")
    schema = model_bundle.get("feature_schema") or {}
    feature_names = list(model_bundle.get("feature_names") or _feature_names(schema))
    spec_names = (model_bundle.get("model_specs") or {}).get(preferred_model) or feature_names
    indices = [feature_names.index(name) for name in spec_names if name in feature_names]
    payload = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    changed_targets = 0
    match_rows = 0
    for target in payload.get("targets") or []:
        for subgoal in target.get("top_subgoals") or []:
            scored = []
            for rank, match in enumerate(subgoal.get("matches") or [], start=1):
                row = _row_from_audit_match(target, subgoal, match, rank=rank)
                features = _feature_row(row, schema)
                score = float(model.predict(np.asarray([features], dtype=np.float32)[:, indices])[0])
                match["learned_subgoal_score"] = round(score, 6)
                scored.append(match)
                match_rows += 1
            if scored:
                scored.sort(key=lambda item: (float(item.get("learned_subgoal_score") or 0.0), float(item.get("motif_similarity") or 0.0)), reverse=True)
                subgoal["matches"] = scored
                subgoal["best_learned_subgoal_score"] = scored[0].get("learned_subgoal_score")
        top = target.get("top_subgoals") or []
        if top:
            top.sort(
                key=lambda item: (
                    float(item.get("best_learned_subgoal_score") or -1e9),
                    float(item.get("best_score") or 0.0),
                    int(item.get("heavy_atoms") or 0),
                ),
                reverse=True,
            )
            target["top_subgoals"] = top
            changed_targets += 1
    payload.setdefault("metadata", {})["learned_subgoal_scorer"] = {
        "schema_version": model_bundle.get("schema_version"),
        "preferred_model": preferred_model,
        "audit_json": str(audit_json),
        "contract": "rerank existing audit matches only; does not generate routes",
    }
    payload["learned_summary"] = {"targets_reranked": changed_targets, "match_rows_scored": match_rows}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_audit_markdown(payload), encoding="utf-8")
    return {"output_json": str(output_json), "targets_reranked": changed_targets, "match_rows_scored": match_rows}


def _load_program_splits(program_manifest: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(Path(program_manifest).read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    return {split: _read_jsonl(Path(outputs[split])) for split in ("train", "val", "test")}


def _query_items(
    programs: list[dict[str, Any]],
    *,
    min_heavy_atoms: int,
    max_fragments_per_molecule: int,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for program in programs:
        route_transforms = tuple(_norm_transform(step.get("transformation_superclass")) for step in program.get("steps") or [] if isinstance(step, dict))
        common = {
            "program_id": str(program.get("program_id") or ""),
            "doi": str(program.get("doi") or ""),
            "cascade_id": str(program.get("cascade_id") or ""),
            "cascade_type": _norm(program.get("cascade_type")),
            "quality_tier": _norm(program.get("quality_tier")),
            "route_transforms": route_transforms,
        }
        _add_item(rows, role="program_target", smiles=program.get("target_smiles"), transform="", source_step_id="", common=common, min_heavy_atoms=min_heavy_atoms)
        for frag in _top_fragments(program.get("target_smiles"), max_fragments_per_molecule=max_fragments_per_molecule):
            _add_item(rows, role="target_fragment", smiles=frag, transform="", source_step_id="", common=common, min_heavy_atoms=min_heavy_atoms)
        for step in program.get("steps") or []:
            if not isinstance(step, dict):
                continue
            transform = _norm_transform(step.get("transformation_superclass"))
            step_id = str(step.get("transition_id") or step.get("step_id") or "")
            _add_item(rows, role="step_product", smiles=step.get("product_smiles"), transform=transform, source_step_id=step_id, common=common, min_heavy_atoms=min_heavy_atoms)
            for frag in _top_fragments(step.get("product_smiles"), max_fragments_per_molecule=max_fragments_per_molecule):
                _add_item(rows, role="step_product_fragment", smiles=frag, transform=transform, source_step_id=step_id, common=common, min_heavy_atoms=min_heavy_atoms)
    return list(rows.values())


def _evidence_items(programs: list[dict[str, Any]], *, min_heavy_atoms: int) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for program in programs:
        route_transforms = tuple(_norm_transform(step.get("transformation_superclass")) for step in program.get("steps") or [] if isinstance(step, dict))
        compatibility = program.get("compatibility") or {}
        common = {
            "program_id": str(program.get("program_id") or ""),
            "doi": str(program.get("doi") or ""),
            "cascade_id": str(program.get("cascade_id") or ""),
            "cascade_type": _norm(program.get("cascade_type")),
            "quality_tier": _norm(program.get("quality_tier")),
            "evidence_strength": _norm(compatibility.get("evidence_strength")),
            "compatibility_label": _norm(compatibility.get("compatibility_label")),
            "route_transforms": route_transforms,
        }
        _add_item(rows, role="program_target", smiles=program.get("target_smiles"), transform="", source_step_id="", common=common, min_heavy_atoms=min_heavy_atoms)
        for step in program.get("steps") or []:
            if not isinstance(step, dict):
                continue
            transform = _norm_transform(step.get("transformation_superclass"))
            step_id = str(step.get("transition_id") or step.get("step_id") or "")
            _add_item(rows, role="step_product", smiles=step.get("product_smiles"), transform=transform, source_step_id=step_id, common=common, min_heavy_atoms=min_heavy_atoms)
            for reactant in step.get("reactants") or []:
                _add_item(rows, role="step_reactant", smiles=reactant, transform=transform, source_step_id=step_id, common=common, min_heavy_atoms=min_heavy_atoms)
    out = list(rows.values())
    for row in out:
        row["fingerprint"] = _fp(row["smiles"])
    return [row for row in out if row.get("fingerprint") is not None]


def _add_item(
    rows: dict[str, dict[str, Any]],
    *,
    role: str,
    smiles: Any,
    transform: str,
    source_step_id: str,
    common: dict[str, Any],
    min_heavy_atoms: int,
) -> None:
    smi = canonical_smiles(str(smiles or ""))
    props = _mol_props(smi)
    if not props["valid"] or props["heavy_atoms"] < min_heavy_atoms or props["heavy_atoms"] > 90:
        return
    key = stable_id(common.get("program_id"), role, source_step_id, smi)
    rows[key] = {
        "item_id": key,
        "role": role,
        "smiles": smi,
        "transform": _norm_transform(transform),
        "source_step_id": source_step_id,
        "heavy_atoms": int(props["heavy_atoms"]),
        "ring_count": int(props["ring_count"]),
        "hetero_atoms": int(props["hetero_atoms"]),
        **common,
    }


def _top_fragments(smiles: Any, *, max_fragments_per_molecule: int) -> list[str]:
    rows = []
    for frag in _fragments(str(smiles or "")):
        props = _mol_props(frag)
        if props["valid"]:
            rows.append((int(props["heavy_atoms"]), int(props["hetero_atoms"]), frag))
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [frag for _heavy, _hetero, frag in rows[: int(max_fragments_per_molecule)]]


def _candidate_rows(
    queries: list[dict[str, Any]],
    train_evidence: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    candidates_per_query: int,
    positive_similarity: float,
    strong_positive_similarity: float,
    exclude_same_program: bool,
    require_positive: bool,
) -> list[dict[str, Any]]:
    evidence_fps = [row["fingerprint"] for row in train_evidence]
    rows = []
    for query in queries:
        qfp = _fp(query["smiles"])
        if qfp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qfp, evidence_fps)
        ranked = np.argsort(np.asarray(sims, dtype=float))[-max(1, int(candidates_per_query)) :][::-1]
        group_rows = []
        for candidate_rank, idx in enumerate(ranked, start=1):
            evidence = train_evidence[int(idx)]
            if exclude_same_program and evidence.get("program_id") == query.get("program_id"):
                continue
            sim = float(sims[int(idx)])
            row = _candidate_row(
                query,
                evidence,
                similarity=sim,
                candidate_rank=candidate_rank,
                schema=schema,
                positive_similarity=positive_similarity,
                strong_positive_similarity=strong_positive_similarity,
            )
            group_rows.append(row)
        if require_positive and not any(row["training_relevance"] > 0 for row in group_rows):
            continue
        rows.extend(group_rows)
    return rows


def _candidate_row(
    query: dict[str, Any],
    evidence: dict[str, Any],
    *,
    similarity: float,
    candidate_rank: int,
    schema: dict[str, Any],
    positive_similarity: float,
    strong_positive_similarity: float,
) -> dict[str, Any]:
    query_transform = _norm_transform(query.get("transform"))
    evidence_transform = _norm_transform(evidence.get("transform"))
    query_route_transforms = set(_norm_transform(value) for value in query.get("route_transforms") or [] if value)
    evidence_route_transforms = set(_norm_transform(value) for value in evidence.get("route_transforms") or [] if value)
    transform_match = bool(query_transform and evidence_transform and query_transform == evidence_transform)
    route_transform_overlap = bool(query_route_transforms & evidence_route_transforms)
    same_cascade_type = bool(query.get("cascade_type") and evidence.get("cascade_type") and query.get("cascade_type") == evidence.get("cascade_type"))
    evidence_role = str(evidence.get("role") or "")
    is_product_evidence = evidence_role in {"program_target", "step_product"}
    positive = bool(
        is_product_evidence
        and similarity >= positive_similarity
        and (
            transform_match
            or (not query_transform and route_transform_overlap and same_cascade_type)
            or (query_transform and route_transform_overlap and similarity >= strong_positive_similarity)
        )
    )
    relevance = 0
    if positive:
        relevance = 2 if similarity >= strong_positive_similarity and evidence_role == _preferred_evidence_role(query.get("role")) else 1
    row = {
        "query_id": str(query.get("item_id") or ""),
        "query_program_id": query.get("program_id"),
        "query_role": query.get("role"),
        "query_smiles": query.get("smiles"),
        "query_transform": query_transform,
        "query_cascade_type": query.get("cascade_type"),
        "query_heavy_atoms": query.get("heavy_atoms"),
        "query_ring_count": query.get("ring_count"),
        "query_hetero_atoms": query.get("hetero_atoms"),
        "evidence_id": evidence.get("item_id"),
        "evidence_program_id": evidence.get("program_id"),
        "evidence_role": evidence_role,
        "evidence_smiles": evidence.get("smiles"),
        "evidence_transform": evidence_transform,
        "evidence_cascade_type": evidence.get("cascade_type"),
        "evidence_quality_tier": evidence.get("quality_tier"),
        "evidence_strength": evidence.get("evidence_strength"),
        "candidate_rank": int(candidate_rank),
        "similarity": round(float(similarity), 6),
        "transform_match": transform_match,
        "route_transform_overlap": route_transform_overlap,
        "same_cascade_type": same_cascade_type,
        "training_relevance": relevance,
    }
    row["features"] = _feature_row(row, schema)
    return row


def _feature_row(row: dict[str, Any], schema: dict[str, Any]) -> list[float]:
    qh = _float(row.get("query_heavy_atoms"))
    eh = _float(row.get("evidence_heavy_atoms"))
    if eh == 0.0 and row.get("evidence_smiles"):
        eh = float(_mol_props(str(row.get("evidence_smiles")))["heavy_atoms"])
    qr = _float(row.get("query_ring_count"))
    er = _float(row.get("evidence_ring_count"))
    if er == 0.0 and row.get("evidence_smiles"):
        er = float(_mol_props(str(row.get("evidence_smiles")))["ring_count"])
    qhet = _float(row.get("query_hetero_atoms"))
    ehet = _float(row.get("evidence_hetero_atoms"))
    if ehet == 0.0 and row.get("evidence_smiles"):
        ehet = float(_mol_props(str(row.get("evidence_smiles")))["hetero_atoms"])
    sim = _float(row.get("similarity"))
    rank = max(1.0, _float(row.get("candidate_rank"), 1.0))
    out = [
        sim,
        1.0 / rank,
        1.0 / np.log2(rank + 1.0),
        qh,
        eh,
        abs(qh - eh),
        min(qh, eh) / max(max(qh, eh), 1.0),
        qr,
        er,
        abs(qr - er),
        qhet,
        ehet,
        abs(qhet - ehet),
        float(bool(row.get("same_cascade_type"))),
    ]
    out.extend(_one_hot(row.get("query_role"), QUERY_ROLES))
    out.extend(_one_hot(row.get("evidence_role"), EVIDENCE_ROLES))
    out.extend(_one_hot(row.get("evidence_quality_tier"), QUALITY_TIERS))
    out.extend(_one_hot(row.get("evidence_strength"), EVIDENCE_STRENGTHS))
    for transform in schema.get("evidence_transforms", []):
        out.append(float(_norm_transform(row.get("evidence_transform")) == transform))
    out.extend(
        [
            float(bool(row.get("transform_match"))),
            float(bool(row.get("route_transform_overlap"))),
        ]
    )
    return [float(value) for value in out]


def _feature_names(schema: dict[str, Any]) -> list[str]:
    names = [
        "sim",
        "inv_rank",
        "inv_log_rank",
        "query_heavy",
        "evidence_heavy",
        "heavy_delta_abs",
        "heavy_minmax_ratio",
        "query_rings",
        "evidence_rings",
        "ring_delta_abs",
        "query_hetero",
        "evidence_hetero",
        "hetero_delta_abs",
        "same_cascade_type",
    ]
    names.extend([f"query_role={role}" for role in QUERY_ROLES])
    names.extend([f"evidence_role={role}" for role in EVIDENCE_ROLES])
    names.extend([f"quality={tier}" for tier in QUALITY_TIERS])
    names.extend([f"evidence_strength={value or 'blank'}" for value in EVIDENCE_STRENGTHS])
    names.extend([f"evidence_transform={value}" for value in schema.get("evidence_transforms", [])])
    names.extend(["context_transform_match", "context_route_transform_overlap"])
    return names


def _build_feature_schema(train_programs: list[dict[str, Any]], train_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    transform_counts = Counter(_norm_transform(row.get("transform")) for row in train_evidence if row.get("transform"))
    return {
        "evidence_transforms": [name for name, _count in transform_counts.most_common(24)],
        "train_program_count": len(train_programs),
        "train_evidence_count": len(train_evidence),
    }


def _dataset(rows: list[dict[str, Any]], schema: dict[str, Any]) -> SubgoalDataset:
    feature_names = _feature_names(schema)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("query_id") or "")].append(row)
    ordered = []
    group_ids = []
    group_sizes = []
    for group_id, group_rows in by_group.items():
        group_rows.sort(key=lambda item: int(item.get("candidate_rank") or 10**9))
        ordered.extend(group_rows)
        group_ids.append(group_id)
        group_sizes.append(len(group_rows))
    return SubgoalDataset(
        rows=ordered,
        x=np.asarray([row.get("features") or _feature_row(row, schema) for row in ordered], dtype=np.float32),
        y=np.asarray([int(row.get("training_relevance") or 0) for row in ordered], dtype=np.int32),
        group_ids=group_ids,
        group_sizes=group_sizes,
        feature_names=feature_names,
    )


def _model_specs(feature_names: list[str]) -> dict[str, list[int]]:
    context = {feature_names.index(name) for name in ("context_transform_match", "context_route_transform_overlap") if name in feature_names}
    rank = {idx for idx, name in enumerate(feature_names) if name in {"inv_rank", "inv_log_rank"}}
    all_indices = list(range(len(feature_names)))
    structure_metadata = [idx for idx in all_indices if idx not in context]
    no_rank = [idx for idx in structure_metadata if idx not in rank]
    return {
        "structure_metadata": structure_metadata,
        "structure_metadata_no_rank": no_rank,
        "context_upper_bound": all_indices,
    }


def _baseline_reports(train: SubgoalDataset, val: SubgoalDataset, test: SubgoalDataset) -> dict[str, Any]:
    return {
        name: {
            "train": _ranking_metrics(train, _baseline_scores(train, name)),
            "val": _ranking_metrics(val, _baseline_scores(val, name)),
            "test": _ranking_metrics(test, _baseline_scores(test, name)),
        }
        for name in ("fingerprint_similarity", "retrieval_rank", "metadata_rule_proxy")
    }


def _baseline_scores(dataset: SubgoalDataset, name: str) -> np.ndarray:
    if name == "fingerprint_similarity":
        return np.asarray([_float(row.get("similarity")) for row in dataset.rows], dtype=np.float32)
    if name == "retrieval_rank":
        return np.asarray([1.0 / max(1.0, _float(row.get("candidate_rank"), 1.0)) for row in dataset.rows], dtype=np.float32)
    if name == "metadata_rule_proxy":
        scores = []
        for row in dataset.rows:
            role_bonus = {"program_target": 0.08, "step_product": 0.06, "step_reactant": -0.08}.get(str(row.get("evidence_role") or ""), 0.0)
            quality_bonus = 0.04 if row.get("evidence_quality_tier") == "gold" else 0.0
            strength_bonus = 0.04 if row.get("evidence_strength") == "strong_process_evidence" else 0.0
            context_bonus = 0.04 if row.get("same_cascade_type") else 0.0
            scores.append(_float(row.get("similarity")) + role_bonus + quality_bonus + strength_bonus + context_bonus)
        return np.asarray(scores, dtype=np.float32)
    raise ValueError(name)


def _ranking_metrics(dataset: SubgoalDataset, scores: np.ndarray) -> dict[str, Any]:
    offset = 0
    positive_groups = 0
    reciprocal = []
    hit_counts = {k: 0 for k in TOP_KS}
    all_group_hit_counts = {k: 0 for k in TOP_KS}
    for size in dataset.group_sizes:
        group_scores = np.asarray(scores[offset : offset + size], dtype=float)
        group_y = dataset.y[offset : offset + size]
        order = np.argsort(group_scores)[::-1]
        positives = {idx for idx, value in enumerate(group_y) if int(value) > 0}
        for k in TOP_KS:
            if positives and any(int(idx) in positives for idx in order[:k]):
                all_group_hit_counts[k] += 1
        if positives:
            positive_groups += 1
            best_rank = min(rank for rank, idx in enumerate(order, start=1) if int(idx) in positives)
            reciprocal.append(1.0 / best_rank)
            for k in TOP_KS:
                if best_rank <= k:
                    hit_counts[k] += 1
        offset += size
    return {
        "groups": len(dataset.group_sizes),
        "positive_groups": positive_groups,
        "positive_group_rate": round(positive_groups / max(len(dataset.group_sizes), 1), 6),
        "mrr_positive_groups": round(float(np.mean(reciprocal)) if reciprocal else 0.0, 6),
        "recall_at_k_positive_groups": {str(k): round(hit_counts[k] / max(positive_groups, 1), 6) for k in TOP_KS},
        "recall_at_k_all_groups": {str(k): round(all_group_hit_counts[k] / max(len(dataset.group_sizes), 1), 6) for k in TOP_KS},
    }


def _decision(baselines: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    baseline = float((((baselines.get("fingerprint_similarity") or {}).get("test") or {}).get("mrr_positive_groups")) or 0.0)
    deployable = {
        name: float(((payload.get("test") or {}).get("mrr_positive_groups")) or 0.0)
        for name, payload in models.items()
        if name != "context_upper_bound"
    }
    upper = {
        name: float(((payload.get("test") or {}).get("mrr_positive_groups")) or 0.0)
        for name, payload in models.items()
        if name == "context_upper_bound"
    }
    best_name, best_score = max(deployable.items(), key=lambda item: item[1]) if deployable else ("", 0.0)
    upper_name, upper_score = max(upper.items(), key=lambda item: item[1]) if upper else ("", 0.0)
    return {
        "primary_metric": "test.mrr_positive_groups",
        "fingerprint_similarity_baseline": round(baseline, 6),
        "best_deployable_model": best_name,
        "best_deployable_score": round(best_score, 6),
        "deployable_absolute_delta": round(best_score - baseline, 6),
        "diagnostic_upper_bound_model": upper_name,
        "diagnostic_upper_bound_score": round(upper_score, 6),
        "promote_for_runtime": bool(best_name and best_score > baseline + 0.03),
        "caution": (
            "Deployable models avoid held-out query transform/context labels. "
            "The context_upper_bound model is diagnostic only; route-level improvement still requires provider/search integration."
        ),
    }


def _positive_group_count(dataset: SubgoalDataset) -> int:
    offset = 0
    count = 0
    for size in dataset.group_sizes:
        if np.any(dataset.y[offset : offset + size] > 0):
            count += 1
        offset += size
    return count


def _test_scores(dataset: SubgoalDataset, models: dict[str, Any], model_specs: dict[str, list[int]]) -> dict[str, np.ndarray]:
    scores = {name: _baseline_scores(dataset, name) for name in ("fingerprint_similarity", "retrieval_rank", "metadata_rule_proxy")}
    scores.update({name: model.predict(dataset.x[:, model_specs[name]], num_iteration=model.best_iteration_) for name, model in models.items()})
    return scores


def _ranked_rows(dataset: SubgoalDataset, score_columns: dict[str, np.ndarray], *, top_n: int) -> list[dict[str, Any]]:
    by_group: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(dataset.rows):
        by_group[str(row.get("query_id"))].append((idx, row))
    out = []
    primary = "structure_metadata" if "structure_metadata" in score_columns else next(iter(score_columns))
    for group_id, pairs in by_group.items():
        pairs.sort(key=lambda pair: float(score_columns[primary][pair[0]]), reverse=True)
        for rank, (idx, row) in enumerate(pairs[:top_n], start=1):
            compact = {key: row.get(key) for key in ("query_id", "query_role", "query_smiles", "query_transform", "evidence_id", "evidence_role", "evidence_smiles", "evidence_transform", "similarity", "training_relevance")}
            compact["model_rank"] = rank
            compact["scores"] = {name: round(float(values[idx]), 6) for name, values in score_columns.items()}
            out.append(compact)
    return out


def _row_from_audit_match(target: dict[str, Any], subgoal: dict[str, Any], match: dict[str, Any], *, rank: int) -> dict[str, Any]:
    ev_props = _mol_props(str(match.get("smiles") or ""))
    query_role = _audit_query_role(subgoal.get("sources") or [])
    return {
        "query_id": subgoal.get("subgoal_id") or stable_id(subgoal.get("smiles")),
        "query_role": query_role,
        "query_smiles": subgoal.get("smiles"),
        "query_transform": "",
        "query_cascade_type": "",
        "query_heavy_atoms": subgoal.get("heavy_atoms"),
        "query_ring_count": subgoal.get("ring_count"),
        "query_hetero_atoms": subgoal.get("hetero_atoms"),
        "evidence_id": match.get("evidence_id"),
        "evidence_program_id": match.get("program_id"),
        "evidence_role": match.get("role"),
        "evidence_smiles": match.get("smiles"),
        "evidence_transform": match.get("transformation_superclass"),
        "evidence_cascade_type": match.get("cascade_type"),
        "evidence_quality_tier": match.get("quality_tier"),
        "evidence_strength": match.get("evidence_strength"),
        "evidence_heavy_atoms": ev_props.get("heavy_atoms"),
        "evidence_ring_count": ev_props.get("ring_count"),
        "evidence_hetero_atoms": ev_props.get("hetero_atoms"),
        "candidate_rank": rank,
        "similarity": match.get("motif_similarity") or match.get("similarity") or 0.0,
        "transform_match": False,
        "route_transform_overlap": False,
        "same_cascade_type": False,
        "training_relevance": 0,
    }


def _audit_query_role(sources: list[Any]) -> str:
    source_set = {str(value or "") for value in sources}
    if "target" in source_set:
        return "program_target"
    if "target_fragment" in source_set:
        return "target_fragment"
    if "route_step_product" in source_set or "route_leaf_or_reactant" in source_set:
        return "step_product"
    if "route_leaf_fragment" in source_set:
        return "step_product_fragment"
    return "target_fragment"


def _preferred_evidence_role(query_role: Any) -> str:
    role = str(query_role or "")
    if role == "program_target":
        return "program_target"
    return "step_product"


def _one_hot(value: Any, choices: tuple[str, ...]) -> list[float]:
    norm = _norm(value)
    return [float(norm == _norm(choice)) for choice in choices]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_transform(value: Any) -> str:
    return _norm(value).replace(" ", "_")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _fp(smiles: Any) -> Any:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Cascade Subgoal Scorer",
        "",
        "## Contract",
        "",
        f"- {result.get('metadata', {}).get('contract')}",
        f"- Label semantics: {result.get('metadata', {}).get('label_semantics')}",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (result.get("counts") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Test Metrics", "", "| Method | MRR positive groups | R@1 | R@3 | R@5 | R@10 |", "|---|---:|---:|---:|---:|---:|"])
    methods = {}
    for name, payload in (result.get("baselines") or {}).items():
        methods[f"baseline:{name}"] = payload
    for name, payload in (result.get("models") or {}).items():
        methods[f"model:{name}"] = payload
    for name, payload in methods.items():
        test = payload.get("test") or {}
        recall = test.get("recall_at_k_positive_groups") or {}
        lines.append(
            "| {name} | {mrr:.4f} | {r1:.4f} | {r3:.4f} | {r5:.4f} | {r10:.4f} |".format(
                name=name,
                mrr=float(test.get("mrr_positive_groups") or 0.0),
                r1=float(recall.get("1") or 0.0),
                r3=float(recall.get("3") or 0.0),
                r5=float(recall.get("5") or 0.0),
                r10=float(recall.get("10") or 0.0),
            )
        )
    lines.extend(["", "## Decision", "", "```json", json.dumps(result.get("decision"), indent=2, ensure_ascii=False), "```"])
    if result.get("applied_audit"):
        lines.extend(["", "## Applied Audit", "", "```json", json.dumps(result.get("applied_audit"), indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def _audit_markdown(payload: dict[str, Any]) -> str:
    meta = payload.get("metadata") or {}
    lines = [
        "# Learned Cascade Subgoal Audit",
        "",
        f"- Source audit: `{meta.get('learned_subgoal_scorer', {}).get('audit_json')}`",
        f"- Model: `{meta.get('learned_subgoal_scorer', {}).get('preferred_model')}`",
        "",
        "| target | routes | top subgoal | learned score | top evidence |",
        "|---|---:|---|---:|---|",
    ]
    for target in payload.get("targets") or []:
        subgoal = (target.get("top_subgoals") or [{}])[0]
        match = (subgoal.get("matches") or [{}])[0]
        lines.append(
            "| `{target}` | {routes} | `{subgoal}` | {score:.4f} | {evidence} |".format(
                target=str(target.get("target_smiles") or "")[:32],
                routes=target.get("route_count"),
                subgoal=str(subgoal.get("smiles") or "")[:42],
                score=float(subgoal.get("best_learned_subgoal_score") or 0.0),
                evidence=f"{match.get('doi')} / {match.get('role')} / {match.get('transformation_superclass')}" if match else "",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a generic cascade subgoal-evidence scorer")
    ap.add_argument("--mode", default="train", choices=["train", "apply"])
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-query-heavy-atoms", type=int, default=8)
    ap.add_argument("--min-evidence-heavy-atoms", type=int, default=7)
    ap.add_argument("--max-fragments-per-molecule", type=int, default=8)
    ap.add_argument("--candidates-per-query", type=int, default=80)
    ap.add_argument("--positive-similarity", type=float, default=0.55)
    ap.add_argument("--strong-positive-similarity", type=float, default=0.72)
    ap.add_argument("--n-estimators", type=int, default=260)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-audit-json")
    ap.add_argument("--apply-output-json")
    ap.add_argument("--model-path")
    ap.add_argument("--preferred-model", default="structure_metadata_no_rank")
    args = ap.parse_args()
    if args.mode == "apply":
        if not args.model_path or not args.apply_audit_json or not args.apply_output_json:
            raise SystemExit("--mode apply requires --model-path, --apply-audit-json, and --apply-output-json")
        result = apply_scorer_to_subgoal_audit(
            audit_json=Path(args.apply_audit_json),
            output_json=Path(args.apply_output_json),
            model_path=Path(args.model_path),
            preferred_model=args.preferred_model,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    result = train_cascade_subgoal_scorer(
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        min_query_heavy_atoms=args.min_query_heavy_atoms,
        min_evidence_heavy_atoms=args.min_evidence_heavy_atoms,
        max_fragments_per_molecule=args.max_fragments_per_molecule,
        candidates_per_query=args.candidates_per_query,
        positive_similarity=args.positive_similarity,
        strong_positive_similarity=args.strong_positive_similarity,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        apply_audit_json=Path(args.apply_audit_json) if args.apply_audit_json else None,
        apply_output_json=Path(args.apply_output_json) if args.apply_output_json else None,
    )
    print(json.dumps({"counts": result["counts"], "decision": result["decision"], "applied_audit": result.get("applied_audit")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
