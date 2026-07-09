"""Route-level v4 cascade product-value features and runtime scoring.

The v4 product-value model is deliberately separate from the older
AutoPlanner trace/student path.  It learns route usefulness from
``dataset_v4_release`` records and can be applied as a post-reranker to native
ChemEnzy route pools.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch.nn as nn


ROUTE_LABEL_NAMES = [
    "gold_quality",
    "demonstrated_success",
    "outcome_supported",
    "condition_supported",
    "substrate_scope_supported",
    "rxn_step_supported",
    "catalyst_supported",
    "species_supported",
]

ROUTE_CATEGORICAL_FIELDS = [
    "route_domain",
    "route_source",
    "dominant_transformation",
    "catalyst_class_top",
    "ec1_top",
    "solvent_top",
    "step_count_bucket",
    "audit_route_class",
]

ROUTE_NUMERIC_FIELDS = [
    "step_count_scaled",
    "rxn_step_fraction",
    "condition_step_fraction",
    "catalyst_step_fraction",
    "enzymatic_step_fraction",
    "aqueous_step_fraction",
    "substrate_scope_scaled",
    "overall_yield_scaled",
    "overall_ee_scaled",
    "overall_time_scaled",
    "product_heavy_scaled",
    "starting_material_heavy_scaled",
    "terminal_reactant_count_scaled",
    "max_terminal_similarity",
    "stock_closed",
    "native_rank_scaled",
    "native_score_scaled",
    "audit_is_triage",
    "audit_is_reject",
    "audit_issue_trivial_stock_closure",
    "audit_issue_generic_sequence",
    "audit_issue_racemization",
    "audit_issue_large_atom_gain",
    "audit_issue_large_heavy_atom_gain",
    "audit_issue_large_carbon_gain",
    "audit_issue_large_hetero_atom_gain",
    "audit_issue_invalid_product_smiles",
    "audit_issue_invalid_or_missing_reactants",
    "audit_plausibility_passed",
    "audit_max_heavy_gain_scaled",
    "audit_max_carbon_gain_scaled",
    "audit_max_hetero_gain_scaled",
    "audit_tag_late_stage",
    "audit_tag_semisynthesis_core",
    "audit_tag_fragment_coupling",
]


@dataclass(frozen=True)
class V4RoutePrediction:
    route_value: float
    label_probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_value": round(float(self.route_value), 6),
            "label_probabilities": {
                key: round(float(value), 6)
                for key, value in sorted(self.label_probabilities.items())
            },
            "confidence": round(float(self.confidence), 6),
        }


def route_record_from_v4(raw: dict[str, Any], *, split: str = "", split_group_id: str = "") -> dict[str, Any]:
    steps = [_compact_v4_step(step) for step in raw.get("steps") or [] if isinstance(step, dict)]
    target = canonical_smiles(str(raw.get("target_product_smiles") or "")) or str(raw.get("target_product_smiles") or "").strip()
    starting = str(raw.get("starting_material_smiles") or "").strip()
    route_id = stable_id("v4", raw.get("doi"), raw.get("cascade_id"), target)
    labels = _v4_observable_labels(raw, steps)
    return {
        "schema_version": "v4_cascade_product_route.v1",
        "route_id": route_id,
        "split_group_id": split_group_id or stable_id(raw.get("doi"), raw.get("cascade_id"), target),
        "split": split,
        "dataset": "dataset_v4_release",
        "route_source": "dataset_v4_release",
        "doi": raw.get("doi"),
        "cascade_id": raw.get("cascade_id"),
        "target_smiles": target,
        "target_name": raw.get("target_product_name"),
        "starting_material_smiles": starting,
        "starting_material_name": raw.get("starting_material_name"),
        "route_domain": raw.get("cascade_type") or raw.get("route_domain") or "unknown",
        "quality_tier": raw.get("quality_tier"),
        "is_high_quality": _truthy(raw.get("is_high_quality")),
        "trainable_recommended": _truthy(raw.get("trainable_recommended")),
        "is_demonstrated_success": _truthy(raw.get("is_demonstrated_success")),
        "overall_yield": _safe_float(raw.get("overall_yield")),
        "overall_ee": _safe_float(raw.get("overall_ee")),
        "overall_de": _safe_float(raw.get("overall_de")),
        "overall_dr": _safe_float(raw.get("overall_dr")),
        "total_reaction_time": _safe_float(raw.get("total_reaction_time")),
        "n_substrate_scope_entries": _safe_int(raw.get("n_substrate_scope_entries"), len(raw.get("substrate_scope") or [])),
        "n_input_species": _safe_int(raw.get("n_input_species"), 0),
        "n_output_species": _safe_int(raw.get("n_output_species"), 0),
        "compatibility": raw.get("compatibility") or {},
        "substrate_scope": _compact_scope(raw.get("substrate_scope") or []),
        "steps": steps,
        "terminal_reactants": [s for s in _split_multi_smiles(starting) if s],
        "labels": labels,
        "value_target": observable_value_target(labels),
        "route_family": route_family_from_steps(raw.get("cascade_type") or raw.get("route_domain"), steps),
        "metadata": {
            "article_title": raw.get("article_title"),
            "journal": raw.get("journal"),
            "publish_year": raw.get("publish_year"),
            "quality_flags": raw.get("quality_flags"),
            "llm_max_confidence": raw.get("llm_max_confidence"),
        },
    }


def route_record_from_trace_candidate(row: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for step in row.get("gt_route") or []:
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                "step_index": step.get("step_index"),
                "rxn_smiles": step.get("rxn_smiles"),
                "reactants": _reaction_reactants(step.get("rxn_smiles")),
                "products": _reaction_products(step.get("rxn_smiles")),
                "transformation_superclass": "unknown",
                "transformation_name": "unknown",
                "step_conditions": step.get("condition") or {},
                "catalyst_components": [
                    {
                        "catalyst_class": cls,
                        "ec_number": step.get("ec_number"),
                    }
                    for cls in step.get("catalyst_classes") or []
                ],
                "pairwise_mode": "unknown",
                "intermediate_isolated": None,
                "step_yield_percent": None,
                "step_conversion_percent": None,
                "step_ee_percent": None,
            }
        )
    target = str(row.get("target_smiles") or "")
    labels = {
        name: 0.0
        for name in ROUTE_LABEL_NAMES
    }
    labels["rxn_step_supported"] = float(bool(steps))
    labels["condition_supported"] = float(any(step.get("step_conditions") for step in steps))
    labels["catalyst_supported"] = float(any(step.get("catalyst_components") for step in steps))
    return {
        "schema_version": "v4_cascade_product_route.v1",
        "route_id": stable_id("v4_trace_candidate", row.get("doi"), row.get("cascade_id"), target),
        "split_group_id": row.get("split_group_id") or stable_id(row.get("doi"), row.get("cascade_id"), target),
        "split": row.get("split") or "",
        "dataset": "dataset_v4_release",
        "route_source": "dataset_v4_release_trace_candidate",
        "doi": row.get("doi"),
        "cascade_id": row.get("cascade_id"),
        "target_smiles": target,
        "starting_material_smiles": "",
        "route_domain": row.get("route_domain") or "unknown",
        "quality_tier": row.get("quality_tier"),
        "is_high_quality": True,
        "trainable_recommended": True,
        "is_demonstrated_success": False,
        "overall_yield": None,
        "overall_ee": None,
        "total_reaction_time": None,
        "n_substrate_scope_entries": 0,
        "n_input_species": 0,
        "n_output_species": 0,
        "compatibility": {"compatibility_label": row.get("compatibility_label")},
        "substrate_scope": [],
        "steps": steps,
        "terminal_reactants": [],
        "labels": labels,
        "value_target": observable_value_target(labels),
        "route_family": route_family_from_steps(row.get("route_domain"), steps),
        "metadata": {},
    }


def route_record_from_native_route(
    route: dict[str, Any],
    *,
    target_smiles: str,
    target_id: str | None = None,
    native_rank: int | None = None,
    dataset: str = "native_route_pool",
) -> dict[str, Any]:
    steps = [_compact_native_step(step, idx) for idx, step in enumerate(route.get("steps") or []) if isinstance(step, dict)]
    terminal_reactants = native_terminal_reactants(steps)
    target = canonical_smiles(target_smiles) or str(target_smiles or "").strip()
    rank = native_rank
    if rank is None:
        rank = _safe_int(route.get("route_rank"), 0)
    stock_closed = bool(route.get("stock_closed"))
    if not stock_closed:
        stock_closed = _native_stock_closed(steps, terminal_reactants)
    return {
        "schema_version": "v4_cascade_product_route.v1",
        "route_id": stable_id("native", target_id or target, rank, route.get("score"), [step.get("rxn_smiles") for step in steps]),
        "split_group_id": stable_id("native", target_id or target),
        "split": "eval",
        "dataset": dataset,
        "route_source": route.get("backend") or "ChemEnzyRetroPlanner",
        "target_id": target_id,
        "target_smiles": target,
        "starting_material_smiles": "",
        "route_domain": route.get("route_domain") or "unknown",
        "quality_tier": None,
        "native_rank": rank,
        "native_score": _safe_float(route.get("score")),
        "stock_closed": stock_closed,
        "solved": bool(route.get("solved", stock_closed)),
        "search_time_s": _safe_float(route.get("search_time_s")),
        "overall_yield": None,
        "overall_ee": None,
        "total_reaction_time": None,
        "n_substrate_scope_entries": 0,
        "n_input_species": 0,
        "n_output_species": 0,
        "compatibility": {},
        "substrate_scope": [],
        "steps": steps,
        "terminal_reactants": terminal_reactants,
        "labels": {},
        "value_target": None,
        "route_family": route_family_from_steps(route.get("route_domain"), steps),
        "metadata": {
            "raw_backend_metadata": route.get("raw_backend_metadata") or {},
        },
    }


def route_record_from_planner_route(
    route: dict[str, Any],
    *,
    target_smiles: str,
    target_id: str | None = None,
    native_rank: int | None = None,
    dataset: str = "planner_route_pool",
) -> dict[str, Any]:
    steps = [_compact_planner_step(step, idx) for idx, step in enumerate(route.get("steps") or []) if isinstance(step, dict)]
    terminal_reactants = native_terminal_reactants(steps)
    target = canonical_smiles(target_smiles) or str(target_smiles or "").strip()
    rank = native_rank
    if rank is None:
        rank = _safe_int(route.get("route_rank"), 0)
    metrics = route.get("metrics") or {}
    quality = route.get("quality_vector") or {}
    stock_closed = bool(metrics.get("strict_stock_solve") or metrics.get("strict_stock_solve_any") or quality.get("stock_closed"))
    if not stock_closed:
        stock_closed = _native_stock_closed(steps, terminal_reactants)
    broad = route.get("broad_reservoir") or {}
    source = broad.get("source") or route.get("route_source") or "AutoPlanner"
    return {
        "schema_version": "v4_cascade_product_route.v1",
        "route_id": stable_id("planner", target_id or target, rank, route.get("score"), [step.get("rxn_smiles") for step in steps]),
        "split_group_id": stable_id("planner", target_id or target),
        "split": "eval",
        "dataset": dataset,
        "route_source": source,
        "target_id": target_id,
        "target_smiles": target,
        "starting_material_smiles": "",
        "route_domain": route.get("route_domain") or "unknown",
        "quality_tier": None,
        "native_rank": rank,
        "native_score": _safe_float(route.get("score")),
        "stock_closed": stock_closed,
        "solved": bool(metrics.get("route_solved", stock_closed) or quality.get("route_solved")),
        "search_time_s": _safe_float(route.get("search_time_s")),
        "overall_yield": None,
        "overall_ee": None,
        "total_reaction_time": None,
        "n_substrate_scope_entries": 0,
        "n_input_species": 0,
        "n_output_species": 0,
        "compatibility": {},
        "substrate_scope": [],
        "steps": steps,
        "terminal_reactants": terminal_reactants,
        "labels": {},
        "value_target": None,
        "route_family": route_family_from_steps(route.get("route_domain"), steps),
        "metadata": {
            "broad_reservoir": broad,
        },
    }


def build_route_feature_schema(rows: list[dict[str, Any]], *, n_bits: int = 128) -> dict[str, Any]:
    categories = {}
    for field in ROUTE_CATEGORICAL_FIELDS:
        values = {_feature_field(row, field) for row in rows}
        values.add("__unknown__")
        categories[field] = sorted(values)
    schema = {
        "schema_version": "v4_cascade_product_features.v1",
        "n_bits": int(n_bits),
        "label_names": list(ROUTE_LABEL_NAMES),
        "categorical_fields": list(ROUTE_CATEGORICAL_FIELDS),
        "numeric_fields": list(ROUTE_NUMERIC_FIELDS),
        "categories": categories,
        "fingerprint_blocks": [
            "target_product",
            "starting_materials",
            "route_reactants",
            "route_products",
            "terminal_reactants",
        ],
        "feature_contract": "v4_route_level_cascade_product_value.v1",
    }
    schema["feature_dim"] = len(route_feature_vector(rows[0], schema)) if rows else 0
    return schema


def route_feature_vector(row: dict[str, Any], schema: dict[str, Any]) -> np.ndarray:
    n_bits = int(schema.get("n_bits") or 128)
    out: list[float] = []
    for block in _fingerprint_blocks(row, n_bits=n_bits):
        out.extend(block.tolist())
    for field in schema.get("categorical_fields") or []:
        value = _feature_field(row, field)
        categories = (schema.get("categories") or {}).get(field) or []
        if value not in categories:
            value = "__unknown__"
        out.extend(1.0 if value == category else 0.0 for category in categories)
    numeric = route_numeric_features(row)
    out.extend(float(numeric.get(field, 0.0)) for field in schema.get("numeric_fields") or [])
    return np.asarray(out, dtype=np.float32)


def route_label_vector(row: dict[str, Any]) -> np.ndarray:
    labels = row.get("labels") or {}
    return np.asarray([float(labels.get(name) or 0.0) for name in ROUTE_LABEL_NAMES], dtype=np.float32)


def observable_value_target(labels: dict[str, Any]) -> float:
    values = [float(labels.get(name) or 0.0) for name in ROUTE_LABEL_NAMES]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def route_numeric_features(row: dict[str, Any]) -> dict[str, float]:
    steps = [step for step in row.get("steps") or [] if isinstance(step, dict)]
    n_steps = len(steps)
    rxn_steps = sum(1 for step in steps if step.get("rxn_smiles"))
    condition_steps = sum(1 for step in steps if step.get("step_conditions"))
    catalyst_steps = sum(1 for step in steps if step.get("catalyst_components"))
    enzymatic_steps = sum(1 for step in steps if _is_enzymatic_step(step))
    aqueous_steps = sum(1 for step in steps if _solvent_top([step]) in {"water", "aqueous", "buffer"})
    target_heavy = _heavy_atoms(str(row.get("target_smiles") or ""))
    starting_heavy = max([_heavy_atoms(smi) for smi in _split_multi_smiles(row.get("starting_material_smiles"))] or [0])
    terminal_reactants = [str(smi) for smi in row.get("terminal_reactants") or [] if smi]
    similarities = [_tanimoto(str(row.get("target_smiles") or ""), smi) for smi in terminal_reactants]
    max_similarity = max([sim for sim in similarities if sim is not None] or [0.0])
    return {
        "step_count_scaled": _scale(n_steps, 10.0),
        "rxn_step_fraction": _frac(rxn_steps, n_steps),
        "condition_step_fraction": _frac(condition_steps, n_steps),
        "catalyst_step_fraction": _frac(catalyst_steps, n_steps),
        "enzymatic_step_fraction": _frac(enzymatic_steps, n_steps),
        "aqueous_step_fraction": _frac(aqueous_steps, n_steps),
        "substrate_scope_scaled": _scale(_safe_float(row.get("n_substrate_scope_entries")) or 0.0, 25.0),
        "overall_yield_scaled": _scale(_safe_float(row.get("overall_yield")) or 0.0, 100.0),
        "overall_ee_scaled": _scale(_safe_float(row.get("overall_ee")) or 0.0, 100.0),
        "overall_time_scaled": _log_scale(_safe_float(row.get("total_reaction_time")) or 0.0, 72.0),
        "product_heavy_scaled": _scale(target_heavy, 80.0),
        "starting_material_heavy_scaled": _scale(starting_heavy, 80.0),
        "terminal_reactant_count_scaled": _scale(len(terminal_reactants), 8.0),
        "max_terminal_similarity": float(max_similarity),
        "stock_closed": float(bool(row.get("stock_closed"))),
        "native_rank_scaled": _scale(_safe_float(row.get("native_rank")) or 0.0, 50.0),
        "native_score_scaled": _scale(_safe_float(row.get("native_score")) or 0.0, 1.0),
        **_audit_numeric_features(row),
    }


def route_family_from_steps(route_domain: Any, steps: list[dict[str, Any]]) -> str:
    transformations = [
        _norm(step.get("transformation_superclass"))
        for step in steps
        if _norm(step.get("transformation_superclass")) and _norm(step.get("transformation_superclass")) != "unknown"
    ]
    dominant = Counter(transformations).most_common(1)[0][0] if transformations else "unknown"
    return f"{_norm(route_domain) or 'unknown'}::{dominant}"


def native_terminal_reactants(steps: list[dict[str, Any]]) -> list[str]:
    products = {str(step.get("product_smiles") or "") for step in steps if step.get("product_smiles")}
    out: list[str] = []
    for step in steps:
        for smi in step.get("reactants") or []:
            text = str(smi or "")
            if text and text not in products and text not in out:
                out.append(text)
    if out:
        return out
    for step in steps:
        for smi in (step.get("stock_status") or {}):
            text = str(smi or "")
            if text and text not in out:
                out.append(text)
    return out


class V4CascadeProductValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 192, output_dim: int = len(ROUTE_LABEL_NAMES)):
        super().__init__()
        h2 = max(48, int(hidden) // 2)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, h2),
            nn.GELU(),
        )
        self.label_head = nn.Linear(h2, output_dim)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, x: Any) -> tuple[Any, Any]:
        h = self.shared(x)
        return self.label_head(h), self.value_head(h).squeeze(-1)


class LoadedV4CascadeProductValue:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.feature_schema = dict(checkpoint["feature_schema"])
        self.label_names = list(checkpoint.get("label_names") or ROUTE_LABEL_NAMES)
        hidden = int(checkpoint.get("hidden") or 192)
        input_dim = int(self.feature_schema["feature_dim"])
        self.model = V4CascadeProductValueNetwork(input_dim, hidden=hidden, output_dim=len(self.label_names)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, row: dict[str, Any]) -> V4RoutePrediction:
        x = route_feature_vector(row, self.feature_schema)
        with self.torch.no_grad():
            tensor = self.torch.tensor(x[None, :], dtype=self.torch.float32, device=self.device)
            label_logits, value_logit = self.model(tensor)
            label_probs = self.torch.sigmoid(label_logits)[0].detach().cpu().numpy().tolist()
            route_value = float(self.torch.sigmoid(value_logit)[0].detach().cpu().item())
        label_probabilities = {
            name: float(label_probs[idx])
            for idx, name in enumerate(self.label_names)
        }
        confidence = float(np.mean([abs(value - 0.5) * 2.0 for value in label_probabilities.values()])) if label_probabilities else 0.0
        return V4RoutePrediction(route_value=route_value, label_probabilities=label_probabilities, confidence=confidence)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def stable_id(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles or ""))
        return Chem.MolToSmiles(mol) if mol is not None else ""
    except Exception:
        return str(smiles or "").strip()


def _compact_v4_step(step: dict[str, Any]) -> dict[str, Any]:
    rxn = str(step.get("rxn_smiles") or "").strip()
    return {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "rxn_smiles": rxn,
        "reactants": _reaction_reactants(rxn),
        "products": _reaction_products(rxn),
        "transformation_superclass": step.get("transformation_superclass") or "unknown",
        "transformation_name": step.get("transformation_name") or "unknown",
        "step_mode": step.get("step_mode") or "unknown",
        "pairwise_mode": step.get("pairwise_mode") or "unknown",
        "intermediate_isolated": step.get("intermediate_isolated"),
        "step_conditions": step.get("step_conditions") or {},
        "catalyst_components": [
            {
                "catalyst_class": item.get("catalyst_class"),
                "component_name": item.get("component_name"),
                "ec_number": item.get("ec_number"),
                "organism": item.get("organism"),
                "uniprot_id": item.get("uniprot_id"),
                "cofactor_required": item.get("cofactor_required"),
                "cofactor_regeneration_mode": item.get("cofactor_regeneration_mode"),
            }
            for item in step.get("catalyst_components") or []
            if isinstance(item, dict)
        ],
        "step_yield_percent": _safe_float((step.get("step_outcome") or {}).get("step_yield_percent")),
        "step_conversion_percent": _safe_float((step.get("step_outcome") or {}).get("step_conversion_percent")),
        "step_ee_percent": _safe_float((step.get("step_outcome") or {}).get("step_ee_percent")),
    }


def _compact_native_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    rxn = str(step.get("rxn_smiles") or "").strip()
    reactants = [str(smi) for smi in step.get("reactant_smiles") or _reaction_reactants(rxn)]
    products = [str(step.get("product_smiles") or "")] if step.get("product_smiles") else _reaction_products(rxn)
    ec_annotations = step.get("enzyme_ec_annotations") or []
    catalysts = step.get("catalyst_annotations") or []
    catalyst_components = []
    for value in ec_annotations:
        catalyst_components.append({"catalyst_class": "enzyme", "ec_number": value})
    for value in catalysts:
        if isinstance(value, dict):
            catalyst_components.append(value)
        elif value:
            catalyst_components.append({"catalyst_class": str(value)})
    return {
        "step_id": f"native_step_{index}",
        "step_index": index + 1,
        "rxn_smiles": rxn,
        "product_smiles": step.get("product_smiles") or (products[0] if products else ""),
        "reactants": reactants,
        "products": products,
        "transformation_superclass": _infer_native_transformation(step),
        "transformation_name": _infer_native_transformation(step),
        "step_mode": "unknown",
        "pairwise_mode": "unknown",
        "intermediate_isolated": None,
        "step_conditions": _native_condition(step),
        "catalyst_components": catalyst_components,
        "stock_status": step.get("stock_status") or {},
        "native_step_score": _safe_float(step.get("score")),
        "source_model": step.get("source_model"),
    }


def _compact_planner_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "").strip()
    main = str(step.get("main_reactant") or "")
    aux = [str(smi) for smi in step.get("aux_reactants") or [] if smi]
    reactants = [smi for smi in [main, *aux] if smi] or _reaction_reactants(rxn)
    product = str(step.get("product") or step.get("product_smiles") or "")
    products = [product] if product else _reaction_products(rxn)
    ec_value = step.get("ec") or step.get("ec_number")
    catalyst_components = []
    if ec_value:
        catalyst_components.append({"catalyst_class": "enzyme", "ec_number": ec_value})
    catalyst = step.get("catalyst")
    if catalyst:
        catalyst_components.append({"catalyst_class": str(catalyst)})
    return {
        "step_id": f"planner_step_{index}",
        "step_index": index + 1,
        "rxn_smiles": rxn,
        "product_smiles": product or (products[0] if products else ""),
        "reactants": reactants,
        "products": products,
        "transformation_superclass": step.get("reaction_type") or _infer_native_transformation({"rxn_smiles": rxn, "source_model": step.get("source")}),
        "transformation_name": step.get("reaction_type") or "unknown",
        "step_mode": "unknown",
        "pairwise_mode": "unknown",
        "intermediate_isolated": None,
        "step_conditions": {
            key: value
            for key, value in {
                "temperature_c": step.get("T") or step.get("temperature_c"),
                "ph": step.get("pH") or step.get("ph"),
                "solvent": step.get("solvent"),
            }.items()
            if value not in (None, "")
        },
        "catalyst_components": catalyst_components,
        "stock_status": step.get("stock_status") or {},
        "native_step_score": _safe_float((step.get("scores") or {}).get("retro") or step.get("score")),
        "source_model": step.get("source"),
    }


def _compact_scope(scope: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    out = []
    for item in scope[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "substrate_smiles": item.get("substrate_smiles"),
                "product_smiles": item.get("product_smiles"),
                "yield_percent": item.get("yield_percent"),
                "conversion_percent": item.get("conversion_percent"),
                "ee_percent": item.get("ee_percent"),
            }
        )
    return out


def _v4_observable_labels(raw: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, float]:
    n_steps = len(steps)
    rxn_steps = sum(1 for step in steps if step.get("rxn_smiles"))
    labels = {
        "gold_quality": float(_norm(raw.get("quality_tier")) == "gold"),
        "demonstrated_success": float(_truthy(raw.get("is_demonstrated_success"))),
        "outcome_supported": float(_truthy(raw.get("has_outcome")) or raw.get("overall_yield") is not None or raw.get("overall_ee") is not None),
        "condition_supported": float(_truthy(raw.get("has_conditions")) or any(step.get("step_conditions") for step in steps)),
        "substrate_scope_supported": float(_safe_int(raw.get("n_substrate_scope_entries"), len(raw.get("substrate_scope") or [])) > 0),
        "rxn_step_supported": _frac(rxn_steps, n_steps),
        "catalyst_supported": float(_safe_int(raw.get("n_catalyst_components"), 0) > 0 or any(step.get("catalyst_components") for step in steps)),
        "species_supported": float(_safe_int(raw.get("n_input_species"), 0) > 0 and _safe_int(raw.get("n_output_species"), 0) > 0),
    }
    return {name: float(labels.get(name, 0.0)) for name in ROUTE_LABEL_NAMES}


def _fingerprint_blocks(row: dict[str, Any], *, n_bits: int) -> list[np.ndarray]:
    steps = [step for step in row.get("steps") or [] if isinstance(step, dict)]
    route_reactants: list[str] = []
    route_products: list[str] = []
    for step in steps:
        route_reactants.extend(str(smi) for smi in step.get("reactants") or [])
        route_products.extend(str(smi) for smi in step.get("products") or [])
    return [
        _fp_many([str(row.get("target_smiles") or "")], n_bits=n_bits),
        _fp_many(_split_multi_smiles(row.get("starting_material_smiles")), n_bits=n_bits),
        _fp_many(route_reactants, n_bits=n_bits),
        _fp_many(route_products, n_bits=n_bits),
        _fp_many([str(smi) for smi in row.get("terminal_reactants") or []], n_bits=n_bits),
    ]


def _feature_field(row: dict[str, Any], field: str) -> str:
    steps = [step for step in row.get("steps") or [] if isinstance(step, dict)]
    if field == "route_domain":
        return _norm(row.get("route_domain")) or "unknown"
    if field == "route_source":
        return _norm(row.get("route_source")) or "unknown"
    if field == "dominant_transformation":
        values = [_norm(step.get("transformation_superclass")) for step in steps if _norm(step.get("transformation_superclass"))]
        return Counter(values).most_common(1)[0][0] if values else "unknown"
    if field == "catalyst_class_top":
        values = [
            _norm(cat.get("catalyst_class"))
            for step in steps
            for cat in step.get("catalyst_components") or []
            if isinstance(cat, dict) and _norm(cat.get("catalyst_class"))
        ]
        return Counter(values).most_common(1)[0][0] if values else "none"
    if field == "ec1_top":
        values = [
            str(cat.get("ec_number") or "").split(".", 1)[0].strip().lower()
            for step in steps
            for cat in step.get("catalyst_components") or []
            if isinstance(cat, dict) and cat.get("ec_number")
        ]
        return Counter(values).most_common(1)[0][0] if values else "none"
    if field == "solvent_top":
        return _solvent_top(steps)
    if field == "step_count_bucket":
        n_steps = len(steps)
        if n_steps <= 1:
            return "1"
        if n_steps == 2:
            return "2"
        if n_steps <= 4:
            return "3-4"
        return "5+"
    if field == "audit_route_class":
        return _norm((row.get("product_audit") or {}).get("route_class")) or "none"
    return "unknown"


def _audit_numeric_features(row: dict[str, Any]) -> dict[str, float]:
    audit = row.get("product_audit") or {}
    route_class = str(audit.get("route_class") or "")
    issues = set(str(item) for item in audit.get("issues") or [])
    tags = set(str(item) for item in audit.get("tags") or [])
    plausibility = audit.get("route_plausibility") if isinstance(audit.get("route_plausibility"), dict) else {}
    plausibility_reasons = set(str(item) for item in (plausibility.get("reasons") or []))
    all_issues = issues | plausibility_reasons
    max_heavy_gain, max_carbon_gain, max_hetero_gain = _max_plausibility_gains(plausibility)
    return {
        "audit_is_triage": float(route_class in {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}),
        "audit_is_reject": float(route_class == "reject_artifact"),
        "audit_issue_trivial_stock_closure": float("trivial_stock_closure" in all_issues),
        "audit_issue_generic_sequence": float("generic_reaction_sequence" in all_issues),
        "audit_issue_racemization": float("racemization_artifact" in all_issues),
        "audit_issue_large_atom_gain": float(
            bool(
                {
                    "large_unexplained_atom_gain",
                    "large_unexplained_heavy_atom_gain",
                    "large_unexplained_carbon_gain",
                    "large_unexplained_hetero_atom_gain",
                }
                & all_issues
            )
        ),
        "audit_issue_large_heavy_atom_gain": float("large_unexplained_heavy_atom_gain" in all_issues),
        "audit_issue_large_carbon_gain": float("large_unexplained_carbon_gain" in all_issues),
        "audit_issue_large_hetero_atom_gain": float("large_unexplained_hetero_atom_gain" in all_issues),
        "audit_issue_invalid_product_smiles": float("invalid_product_smiles" in all_issues),
        "audit_issue_invalid_or_missing_reactants": float("invalid_or_missing_reactants" in all_issues),
        "audit_plausibility_passed": float(plausibility.get("passed")) if "passed" in plausibility else 0.0,
        "audit_max_heavy_gain_scaled": _scale(max_heavy_gain, 20.0),
        "audit_max_carbon_gain_scaled": _scale(max_carbon_gain, 20.0),
        "audit_max_hetero_gain_scaled": _scale(max_hetero_gain, 20.0),
        "audit_tag_late_stage": float("late_stage_derivatization" in tags),
        "audit_tag_semisynthesis_core": float("natural_core_terminal" in tags),
        "audit_tag_fragment_coupling": float("aryl_coupling_hint" in tags),
    }


def _max_plausibility_gains(plausibility: dict[str, Any]) -> tuple[float, float, float]:
    max_heavy = 0.0
    max_carbon = 0.0
    max_hetero = 0.0
    for step in plausibility.get("steps") or []:
        if not isinstance(step, dict):
            continue
        max_heavy = max(max_heavy, _safe_float(step.get("heavy_atom_gain")) or 0.0)
        max_carbon = max(max_carbon, _safe_float(step.get("carbon_gain")) or 0.0)
        max_hetero = max(max_hetero, _safe_float(step.get("hetero_atom_gain")) or 0.0)
    return max_heavy, max_carbon, max_hetero


def _solvent_top(steps: list[dict[str, Any]]) -> str:
    values = []
    for step in steps:
        condition = step.get("step_conditions") or {}
        solvent = condition.get("solvent") or condition.get("solvent_name") or condition.get("cosolvent")
        if solvent:
            values.append(_norm(solvent))
    return Counter(values).most_common(1)[0][0] if values else "unknown"


def _is_enzymatic_step(step: dict[str, Any]) -> bool:
    for item in step.get("catalyst_components") or []:
        text = " ".join(str(item.get(key) or "") for key in ("catalyst_class", "ec_number", "component_name")).lower()
        if item.get("ec_number") or "enzyme" in text or "whole_cell" in text:
            return True
    return False


def _native_condition(step: dict[str, Any]) -> dict[str, Any]:
    predictions = step.get("condition_predictions") or []
    if predictions and isinstance(predictions[0], dict):
        condition = predictions[0]
        return {
            "temperature_c": condition.get("temperature_c") or condition.get("Temperature"),
            "ph": condition.get("ph") or condition.get("pH"),
            "solvent": condition.get("solvent") or condition.get("Solvent"),
        }
    return {}


def _native_stock_closed(steps: list[dict[str, Any]], terminal_reactants: list[str]) -> bool:
    if not steps:
        return False
    stock: dict[str, Any] = {}
    for step in steps:
        stock.update(step.get("stock_status") or {})
    if not terminal_reactants:
        return False
    return all(bool(stock.get(smi)) for smi in terminal_reactants)


def _infer_native_transformation(step: dict[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in [
            step.get("source_model"),
            step.get("rxn_smiles"),
            (step.get("raw_backend_metadata") or {}).get("template"),
        ]
    ).lower()
    if "hydrolysis" in text or "c(=o)o" in text:
        return "hydrolysis"
    if "ester" in text or "oc(=o)" in text:
        return "esterification"
    if "boron" in text or "b(o)" in text or "br" in text:
        return "C_C_coupling"
    if "oxid" in text:
        return "oxidation"
    if "reduc" in text:
        return "reduction"
    if "amine" in text or "amination" in text:
        return "amination"
    return "unknown"


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


def _split_multi_smiles(value: Any) -> list[str]:
    out: list[str] = []
    for token in str(value or "").replace(";", ".").split("."):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _fp_many(smiles_values: list[str], *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return arr
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles or ""))
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        tmp = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, tmp)
        arr = np.maximum(arr, tmp)
    return arr


def _heavy_atoms(smiles: str) -> int:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles or ""))
        return int(mol.GetNumHeavyAtoms()) if mol is not None else 0
    except Exception:
        return 0


def _tanimoto(a: str, b: str) -> float | None:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        mol_a = Chem.MolFromSmiles(str(a or ""))
        mol_b = Chem.MolFromSmiles(str(b or ""))
        if mol_a is None or mol_b is None:
            return None
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return None


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scale(value: float | int | None, denom: float) -> float:
    if value is None or denom <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(denom)))


def _log_scale(value: float, denom: float) -> float:
    if value <= 0.0 or denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, math.log1p(value) / math.log1p(denom)))


def _frac(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0
