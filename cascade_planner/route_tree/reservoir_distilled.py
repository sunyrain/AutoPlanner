"""Reservoir-distilled controller runtime.

This module keeps the distilled controller deliberately small: a shared MLP
encoder emits source, budget, leaf, action, stock-risk, route-rerank, and
latency heads. Runtime use is advisory. Any load or inference failure falls
back to the existing heuristic source gate and route-tree policy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.route_tree.source_gate import (
    SOURCE_GROUPS,
    SOURCE_POLICY_BUDGET_LABELS,
    SourceAllocation,
    SourceGate,
    _allocate_budget_by_weight,
    _budget_label_from_multiplier,
    _budget_multiplier_from_label,
    _context_state_id,
    _source_policy_group,
)
from cascade_planner.vnext.features import morgan_fp, stable_bucket


StockChecker = Callable[[str], bool]
RESERVOIR_CONTROLLER_SCHEMA_VERSION = "reservoir_distilled_controller.v1"


@dataclass(frozen=True)
class ReservoirControllerConfig:
    n_bits: int = 256
    input_dim: int = 0
    hidden_dim: int = 256
    dropout: float = 0.10
    source_groups: tuple[str, ...] = tuple(SOURCE_GROUPS)
    budget_labels: tuple[str, ...] = tuple(SOURCE_POLICY_BUDGET_LABELS)
    min_confidence: float = 0.20


class ReservoirDistilledController(nn.Module):
    """Small shared MLP with the heads required by the distillation plan."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        n_source_groups: int = len(SOURCE_GROUPS),
        n_budget_labels: int = len(SOURCE_POLICY_BUDGET_LABELS),
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.source_group_head = nn.Linear(hidden_dim, n_source_groups)
        self.budget_head = nn.Linear(hidden_dim, n_budget_labels)
        self.leaf_value_head = nn.Linear(hidden_dim, 1)
        self.action_value_head = nn.Linear(hidden_dim, 1)
        self.stock_dead_end_head = nn.Linear(hidden_dim, 1)
        self.route_rerank_head = nn.Linear(hidden_dim, 1)
        self.latency_cost_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "source_group_logits": self.source_group_head(h),
            "budget_logits": self.budget_head(h),
            "leaf_value": self.leaf_value_head(h).squeeze(-1),
            "action_value": self.action_value_head(h).squeeze(-1),
            "stock_dead_end_logit": self.stock_dead_end_head(h).squeeze(-1),
            "route_rerank_value": self.route_rerank_head(h).squeeze(-1),
            "latency_cost": self.latency_cost_head(h).squeeze(-1),
        }


class ReservoirDistilledControllerRuntime:
    """Adapter that satisfies both SourceGate and RouteTreeRuntime contracts."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        fallback_source_gate: SourceGate | None = None,
        fallback_runtime: Any | None = None,
    ):
        self.path = Path(checkpoint)
        ckpt = torch.load(str(self.path), map_location="cpu")
        meta = ckpt.get("metadata") or {}
        schema = ckpt.get("feature_schema") or meta.get("feature_schema") or {}
        source_groups = tuple(meta.get("source_groups") or schema.get("source_groups") or SOURCE_GROUPS)
        budget_labels = tuple(meta.get("budget_labels") or schema.get("budget_labels") or SOURCE_POLICY_BUDGET_LABELS)
        input_dim = int(meta.get("input_dim") or schema.get("input_dim") or 0)
        if input_dim <= 0:
            raise ValueError(f"invalid reservoir-distilled controller input_dim: {checkpoint}")
        self.config = ReservoirControllerConfig(
            n_bits=int(meta.get("n_bits") or schema.get("n_bits") or 256),
            input_dim=input_dim,
            hidden_dim=int(meta.get("hidden_dim") or schema.get("hidden_dim") or 256),
            dropout=float(meta.get("dropout") or schema.get("dropout") or 0.10),
            source_groups=source_groups,
            budget_labels=budget_labels,
            min_confidence=_env_float(
                "AUTOPLANNER_RESERVOIR_MIN_CONFIDENCE",
                float(meta.get("min_confidence") or schema.get("min_confidence") or 0.20),
            ),
        )
        self.model = ReservoirDistilledController(
            self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            n_source_groups=len(self.config.source_groups),
            n_budget_labels=len(self.config.budget_labels),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.fallback_source_gate = fallback_source_gate or SourceGate()
        self.fallback_runtime = fallback_runtime

    def allocate(
        self,
        product: str,
        *,
        context: Any | None,
        available_sources: list[str] | tuple[str, ...],
        total_budget: int,
    ) -> SourceAllocation:
        fallback = self.fallback_source_gate.allocate(
            product,
            context=context,
            available_sources=available_sources,
            total_budget=total_budget,
        )
        if not available_sources:
            return fallback
        try:
            out = self._predict(
                product=product,
                leaf=product,
                context=context,
                source="",
                total_budget=total_budget,
            )
            group_probs = torch.softmax(out["source_group_logits"], dim=-1)[0].cpu().numpy().tolist()
            budget_probs = torch.softmax(out["budget_logits"], dim=-1)[0].cpu().numpy().tolist()
            group_map = {
                self.config.source_groups[idx]: float(group_probs[idx])
                for idx in range(min(len(group_probs), len(self.config.source_groups)))
            }
            confidence = max(group_map.values()) if group_map else 0.0
            min_confidence = _min_confidence_for_context(context, default=self.config.min_confidence)
            if confidence < min_confidence:
                return _allocation_with_reason(fallback, f"reservoir_distilled_fallback:low_confidence:{confidence:.3f}")
            selected_group = max(group_map, key=group_map.get) if group_map else ""
            fallback_prob = float(group_map.get("fallback") or 0.0)
            if not _env_truthy("AUTOPLANNER_RESERVOIR_ENABLE_SOURCE_OVERRIDE"):
                if selected_group == "fallback":
                    return _allocation_with_reason(
                        fallback,
                        f"reservoir_distilled_fallback:fallback_group:{fallback_prob:.3f}",
                    )
                return _allocation_with_reason(
                    fallback,
                    f"reservoir_distilled_delegate:{selected_group}:{confidence:.3f}",
                )
            if selected_group == "fallback" and fallback_prob >= _env_float(
                "AUTOPLANNER_RESERVOIR_FALLBACK_GROUP_THRESHOLD",
                0.50,
            ):
                return _allocation_with_reason(
                    fallback,
                    f"reservoir_distilled_fallback:fallback_group:{fallback_prob:.3f}",
                )
            ambiguous_reason = _ambiguous_source_fallback_reason(
                group_map=group_map,
                selected_group=selected_group,
                confidence=confidence,
                context=context,
            )
            if ambiguous_reason:
                return _allocation_with_reason(fallback, ambiguous_reason)
            source_weights = _source_weights_from_group_map(
                group_map=group_map,
                available_sources=available_sources,
            )
            if fallback.safety_guard:
                for source in available_sources:
                    if _source_policy_group(source) in {"chemical", "template"}:
                        source_weights[source] = 0.0
            total_weight = sum(source_weights.values())
            if total_weight <= 0:
                return _allocation_with_reason(fallback, "reservoir_distilled_fallback:no_positive_source_weight")
            source_weights = {source: value / total_weight for source, value in source_weights.items()}
            source_weights = _blend_source_weights(
                controller_weights=source_weights,
                fallback_weights=fallback.source_weights,
                available_sources=available_sources,
            )
            budget_idx = int(np.argmax(budget_probs)) if budget_probs else 1
            budget_label = _safe_choice(list(self.config.budget_labels), budget_idx, default="1x")
            multiplier = _budget_multiplier_from_label(budget_label)
            desired_total = min(16, max(1, int(round(max(1, int(total_budget or 1)) * multiplier))))
            fallback_budget = max(0, desired_total // 4)
            primary_budget = max(1, desired_total - fallback_budget)
            budgets = _allocate_budget_by_weight(
                source_weights,
                primary_budget,
                available_sources=available_sources,
            )
            return SourceAllocation(
                source_weights=source_weights,
                source_budgets=budgets,
                fallback_budget=max(0, desired_total - sum(budgets.values())),
                molecule_flags=dict(fallback.molecule_flags),
                safety_guard=fallback.safety_guard,
                source_group_probs=group_map,
                budget_multiplier=float(multiplier),
                budget_multiplier_label=_budget_label_from_multiplier(multiplier),
                decision="query",
                policy_confidence=float(confidence),
                policy_reason="reservoir_distilled",
                policy_state_id=_context_state_id(context, product),
                selected_source_group=selected_group,
            )
        except Exception as exc:
            return _allocation_with_reason(fallback, f"reservoir_distilled_fallback:{type(exc).__name__}")

    def observe(
        self,
        *,
        product: str,
        context: Any | None,
        allocation: SourceAllocation,
        diagnostics: dict[str, Any],
    ) -> None:
        observer = getattr(self.fallback_source_gate, "observe", None)
        if observer is None:
            return
        try:
            observer(product=product, context=context, allocation=allocation, diagnostics=diagnostics)
        except Exception:
            return

    def evaluate(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        stock_checker: StockChecker | None = None,
    ):
        from cascade_planner.route_tree.runtime import RouteTreeEvaluation, heuristic_action_scores

        if not actions:
            return RouteTreeEvaluation(action_scores=[], model_active=True, reason="reservoir_distilled:no_actions")
        base_eval = _fallback_action_evaluation(
            self.fallback_runtime,
            state,
            leaf,
            actions,
            stock_checker=stock_checker,
        )
        try:
            rows = []
            for action in actions:
                rows.append(
                    reservoir_controller_feature_vector(
                        product=state.target,
                        leaf=leaf,
                        state=state,
                        candidate=action.to_candidate_dict(),
                        source=action.source,
                        stock_checker=stock_checker,
                        n_bits=self.config.n_bits,
                        input_dim=self.config.input_dim,
                        source_groups=self.config.source_groups,
                    )
                )
            x = torch.tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32)
            with torch.no_grad():
                out = self.model(x)
            model_scores = out["action_value"].cpu().numpy().astype(float).tolist()
            baseline = (
                list(base_eval.action_scores)
                if base_eval is not None and len(base_eval.action_scores) == len(actions)
                else heuristic_action_scores(leaf, actions, stock_checker=stock_checker)
            )
            delta_weight = _env_float("AUTOPLANNER_RESERVOIR_ACTION_DELTA_WEIGHT", 0.0)
            scores = [
                float(base + delta_weight * np.tanh(model))
                for base, model in zip(baseline, model_scores)
            ]
            route_values = torch.sigmoid(out["route_rerank_value"]).cpu().numpy().astype(float).tolist()
            stock_risk = torch.sigmoid(out["stock_dead_end_logit"]).cpu().numpy().astype(float).tolist()
            return RouteTreeEvaluation(
                action_scores=scores,
                node_scores=[],
                value_logit=getattr(base_eval, "value_logit", None) if base_eval is not None else None,
                solved_logit=getattr(base_eval, "solved_logit", None) if base_eval is not None else None,
                stock_logit=getattr(base_eval, "stock_logit", None) if base_eval is not None else None,
                progressive_logit=getattr(base_eval, "progressive_logit", None) if base_eval is not None else None,
                compatibility_logit=getattr(base_eval, "compatibility_logit", None) if base_eval is not None else None,
                route_value=_blend_prob(
                    getattr(base_eval, "route_value", 0.0) if base_eval is not None else 0.0,
                    float(max(route_values) if route_values else 0.0),
                ),
                solved_prob=getattr(base_eval, "solved_prob", 0.0) if base_eval is not None else 0.0,
                stock_closed_prob=_blend_prob(
                    getattr(base_eval, "stock_closed_prob", 0.0) if base_eval is not None else 0.0,
                    float(1.0 - min(stock_risk or [1.0])),
                ),
                progressive_prob=getattr(base_eval, "progressive_prob", 0.0) if base_eval is not None else 0.0,
                compatibility_prob=getattr(base_eval, "compatibility_prob", 0.0) if base_eval is not None else 0.0,
                bottleneck_scores=list(getattr(base_eval, "bottleneck_scores", []) or []) if base_eval is not None else [],
                source_budgets=dict(getattr(base_eval, "source_budgets", {}) or {}) if base_eval is not None else {},
                value_calibrated=bool(getattr(base_eval, "value_calibrated", False)) if base_eval is not None else False,
                model_active=True,
                reason=_compose_reason("reservoir_distilled", base_eval),
            )
        except Exception as exc:
            if base_eval is not None:
                return _evaluation_with_reason(base_eval, f"reservoir_distilled_fallback:{type(exc).__name__}")
            return RouteTreeEvaluation(
                action_scores=heuristic_action_scores(leaf, actions, stock_checker=stock_checker),
                model_active=False,
                reason=f"reservoir_distilled_fallback:{type(exc).__name__}",
            )

    def score_open_leaves(
        self,
        state: RouteTreeState,
        leaves: list[str],
        *,
        stock_checker: StockChecker | None = None,
    ):
        from cascade_planner.route_tree.runtime import RouteTreeEvaluation, heuristic_node_scores

        if not leaves:
            return RouteTreeEvaluation(action_scores=[], node_scores=[], model_active=True, reason="reservoir_distilled:no_leaves")
        base_eval = _fallback_leaf_evaluation(
            self.fallback_runtime,
            state,
            leaves,
            stock_checker=stock_checker,
        )
        try:
            rows = [
                reservoir_controller_feature_vector(
                    product=state.target,
                    leaf=leaf,
                    state=state,
                    source="",
                    stock_checker=stock_checker,
                    n_bits=self.config.n_bits,
                    input_dim=self.config.input_dim,
                    source_groups=self.config.source_groups,
                )
                for leaf in leaves
            ]
            x = torch.tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32)
            with torch.no_grad():
                out = self.model(x)
            values = out["leaf_value"].cpu().numpy().astype(float).tolist()
            baseline = (
                list(base_eval.node_scores)
                if base_eval is not None and len(base_eval.node_scores) == len(leaves)
                else heuristic_node_scores(leaves, stock_checker=stock_checker)
            )
            delta_weight = _env_float("AUTOPLANNER_RESERVOIR_LEAF_DELTA_WEIGHT", 0.0)
            scores = [
                float(base + delta_weight * np.tanh(value))
                for base, value in zip(baseline, values)
            ]
            route_values = torch.sigmoid(out["route_rerank_value"]).cpu().numpy().astype(float).tolist()
            stock_risk = torch.sigmoid(out["stock_dead_end_logit"]).cpu().numpy().astype(float).tolist()
            return RouteTreeEvaluation(
                action_scores=[],
                node_scores=scores,
                value_logit=getattr(base_eval, "value_logit", None) if base_eval is not None else None,
                solved_logit=getattr(base_eval, "solved_logit", None) if base_eval is not None else None,
                stock_logit=getattr(base_eval, "stock_logit", None) if base_eval is not None else None,
                progressive_logit=getattr(base_eval, "progressive_logit", None) if base_eval is not None else None,
                compatibility_logit=getattr(base_eval, "compatibility_logit", None) if base_eval is not None else None,
                route_value=_blend_prob(
                    getattr(base_eval, "route_value", 0.0) if base_eval is not None else 0.0,
                    float(max(route_values) if route_values else 0.0),
                ),
                solved_prob=getattr(base_eval, "solved_prob", 0.0) if base_eval is not None else 0.0,
                stock_closed_prob=_blend_prob(
                    getattr(base_eval, "stock_closed_prob", 0.0) if base_eval is not None else 0.0,
                    float(1.0 - min(stock_risk or [1.0])),
                ),
                progressive_prob=getattr(base_eval, "progressive_prob", 0.0) if base_eval is not None else 0.0,
                compatibility_prob=getattr(base_eval, "compatibility_prob", 0.0) if base_eval is not None else 0.0,
                bottleneck_scores=list(getattr(base_eval, "bottleneck_scores", []) or []) if base_eval is not None else [],
                source_budgets=dict(getattr(base_eval, "source_budgets", {}) or {}) if base_eval is not None else {},
                value_calibrated=bool(getattr(base_eval, "value_calibrated", False)) if base_eval is not None else False,
                model_active=True,
                reason=_compose_reason("reservoir_distilled", base_eval),
            )
        except Exception as exc:
            if base_eval is not None:
                return _evaluation_with_reason(base_eval, f"reservoir_distilled_fallback:{type(exc).__name__}")
            return RouteTreeEvaluation(
                action_scores=[],
                node_scores=heuristic_node_scores(leaves, stock_checker=stock_checker),
                model_active=False,
                reason=f"reservoir_distilled_fallback:{type(exc).__name__}",
            )

    def _predict(
        self,
        *,
        product: str,
        leaf: str,
        context: Any | None,
        source: str,
        total_budget: int = 0,
    ) -> dict[str, torch.Tensor]:
        x = reservoir_controller_feature_vector(
            product=product,
            leaf=leaf,
            context=context,
            source=source,
            total_budget=total_budget,
            n_bits=self.config.n_bits,
            input_dim=self.config.input_dim,
            source_groups=self.config.source_groups,
        )
        with torch.no_grad():
            return self.model(torch.tensor(x[None, :], dtype=torch.float32))


class UnavailableReservoirSourceGate(SourceGate):
    """Heuristic source gate that records why the distilled gate was skipped."""

    def __init__(self, reason: str, *, fallback_source_gate: SourceGate | None = None):
        self.reason = str(reason or "unavailable")
        self.fallback_source_gate = fallback_source_gate or SourceGate()

    def allocate(
        self,
        product: str,
        *,
        context: Any | None,
        available_sources: list[str] | tuple[str, ...],
        total_budget: int,
    ) -> SourceAllocation:
        fallback = self.fallback_source_gate.allocate(
            product,
            context=context,
            available_sources=available_sources,
            total_budget=total_budget,
        )
        return _allocation_with_reason(fallback, f"reservoir_distilled_fallback:{self.reason}")

    def observe(
        self,
        *,
        product: str,
        context: Any | None,
        allocation: SourceAllocation,
        diagnostics: dict[str, Any],
    ) -> None:
        observer = getattr(self.fallback_source_gate, "observe", None)
        if observer is None:
            return
        try:
            observer(product=product, context=context, allocation=allocation, diagnostics=diagnostics)
        except Exception:
            return


class UnavailableReservoirRouteTreeRuntime:
    """Heuristic route-tree runtime that exposes fallback diagnostics."""

    def __init__(self, reason: str, *, fallback_runtime: Any | None = None):
        self.reason = str(reason or "unavailable")
        self.fallback_runtime = fallback_runtime

    def evaluate(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        stock_checker: StockChecker | None = None,
    ):
        from cascade_planner.route_tree.runtime import RouteTreeEvaluation, heuristic_action_scores

        base_eval = _fallback_action_evaluation(
            self.fallback_runtime,
            state,
            leaf,
            actions,
            stock_checker=stock_checker,
        )
        if base_eval is not None:
            return _evaluation_with_reason(base_eval, f"reservoir_distilled_fallback:{self.reason}")
        return RouteTreeEvaluation(
            action_scores=heuristic_action_scores(leaf, actions, stock_checker=stock_checker),
            model_active=False,
            reason=f"reservoir_distilled_fallback:{self.reason}",
        )

    def score_open_leaves(
        self,
        state: RouteTreeState,
        leaves: list[str],
        *,
        stock_checker: StockChecker | None = None,
    ):
        from cascade_planner.route_tree.runtime import RouteTreeEvaluation, heuristic_node_scores

        base_eval = _fallback_leaf_evaluation(
            self.fallback_runtime,
            state,
            leaves,
            stock_checker=stock_checker,
        )
        if base_eval is not None:
            return _evaluation_with_reason(base_eval, f"reservoir_distilled_fallback:{self.reason}")
        return RouteTreeEvaluation(
            action_scores=[],
            node_scores=heuristic_node_scores(leaves, stock_checker=stock_checker),
            model_active=False,
            reason=f"reservoir_distilled_fallback:{self.reason}",
        )


def reservoir_controller_feature_vector(
    *,
    product: str,
    leaf: str | None = None,
    state: RouteTreeState | None = None,
    context: Any | None = None,
    candidate: dict[str, Any] | None = None,
    source: str | None = None,
    source_stats: dict[str, Any] | None = None,
    source_diagnostics: dict[str, Any] | None = None,
    route_context_features: dict[str, Any] | None = None,
    reservoir_fields: dict[str, Any] | None = None,
    stock_checker: StockChecker | None = None,
    total_budget: int = 0,
    n_bits: int = 256,
    input_dim: int | None = None,
    source_groups: tuple[str, ...] | list[str] = tuple(SOURCE_GROUPS),
) -> np.ndarray:
    """Build a stable feature vector; missing reservoir fields default to zero."""

    leaf = leaf or product
    candidate = candidate or {}
    route_context_features = route_context_features or {}
    reservoir_fields = reservoir_fields or {}
    source = str(source or candidate.get("source") or candidate.get("enzyme_source") or "")
    product_fp = morgan_fp(product, n_bits=n_bits)
    leaf_fp = morgan_fp(leaf, n_bits=n_bits)
    group = _source_policy_group(source)
    source_group_vec = np.asarray([1.0 if group == item else 0.0 for item in source_groups], dtype=np.float32)
    source_bucket = stable_bucket(source, 8)
    source_name_vec = np.asarray([1.0 if idx == source_bucket else 0.0 for idx in range(8)], dtype=np.float32)
    route_metadata = dict(getattr(context, "route_metadata", {}) or {})
    if state is not None:
        route_metadata.setdefault("state_depth", state.depth)
        route_metadata.setdefault("open_leaf_count", len(state.open_leaves))
        route_metadata.setdefault("target_heavy_atoms", _heavy_atoms(state.target))
        route_metadata.setdefault("leaf_heavy_atoms", _heavy_atoms(leaf))
    route_metadata.update(route_context_features)
    reactants = _candidate_reactants(candidate)
    stock_fraction = _stock_fraction(reactants, stock_checker=stock_checker)
    diagnostics = dict(source_diagnostics or source_stats or {})
    source_row = diagnostics.get(source) if isinstance(diagnostics.get(source), dict) else diagnostics
    source_row = dict(source_row or {})
    product_atoms = _heavy_atoms(product)
    leaf_atoms = _heavy_atoms(leaf)
    main = str(candidate.get("main_reactant") or (reactants[0] if reactants else ""))
    main_atoms = _heavy_atoms(main)
    ec1 = _ec1(getattr(context, "ec1", None) if context is not None else candidate.get("ec"))
    reaction_type = str(getattr(context, "reaction_type", "") if context is not None else candidate.get("reaction_type") or candidate.get("type") or "")
    latency_ms = _safe_float(source_row.get("latency_ms_total"), default=0.0)
    calls = _safe_float(source_row.get("calls"), default=0.0)
    final_returned = _safe_float(source_row.get("final_returned"), default=0.0)
    requested = _safe_float(source_row.get("requested_k_total"), default=0.0)
    scalars = [
        _clip01(float(route_metadata.get("state_depth") or getattr(context, "depth", 0) or 0) / 12.0),
        _clip01(float(route_metadata.get("remaining_depth") or 0.0) / 12.0),
        _clip01(float(route_metadata.get("open_leaf_count") or 1.0) / 8.0),
        _clip01(float(route_metadata.get("nonstock_leaf_count") or 0.0) / 8.0),
        float(bool(route_metadata.get("leaf_stock_hit"))),
        float(bool(route_metadata.get("leaf_parent_adjacent"))),
        float(bool(route_metadata.get("leaf_low_yield"))),
        _clip01(float(route_metadata.get("leaf_heavy_atoms") or leaf_atoms) / 64.0),
        _clip01(float(route_metadata.get("target_heavy_atoms") or product_atoms) / 64.0),
        _clip01(float(total_budget or route_metadata.get("proposal_budget") or 0.0) / 32.0),
        _clip01(float(ec1) / 7.0),
        stable_bucket(reaction_type.lower(), 32) / 31.0,
        _clip01(float(_safe_float(getattr(context, "T", None), default=candidate.get("T") or 0.0)) / 100.0),
        _clip01(float(_safe_float(getattr(context, "pH", None), default=candidate.get("pH") or 0.0)) / 14.0),
        float(bool(route_metadata.get("enzymatic_only_route"))),
        float(bool(route_metadata.get("carbohydrate_like_route"))),
        _clip01(float(product_atoms) / 64.0),
        _clip01(float(leaf_atoms) / max(float(product_atoms), 1.0)),
        _clip01(max(0.0, float(product_atoms - leaf_atoms)) / max(float(product_atoms), 1.0)),
        _clip01(float(len(reactants)) / 6.0),
        _clip01(float(main_atoms) / max(float(max(leaf_atoms, product_atoms)), 1.0)),
        _clip01(float(len(candidate.get("aux_reactants") or [])) / 5.0),
        float(stock_fraction if stock_fraction is not None else 0.0),
        _clip01(float(_safe_float(candidate.get("score"), default=0.0))),
        1.0 / max(float(_safe_float(candidate.get("rank"), default=1.0)), 1.0),
        float("self_loop" in (candidate.get("validity_flags") or [])),
        float(bool(candidate.get("ec"))),
        float(bool(candidate.get("reaction_type") or candidate.get("type"))),
        _clip01(calls / 16.0),
        _clip01(float(bool(source_row.get("queried")))),
        _clip01(requested / 32.0),
        _clip01(_safe_float(source_row.get("raw_returned"), default=0.0) / 32.0),
        _clip01(final_returned / 32.0),
        _clip01(_safe_float(source_row.get("allocated_budget"), default=0.0) / 32.0),
        _clip01(final_returned / max(requested, 1.0)),
        _clip01((latency_ms / max(calls, 1.0)) / 1000.0),
        _clip01(float(reservoir_fields.get("reservoir_rank") or 0.0) / 50.0),
        float(bool(reservoir_fields.get("teacher_stock_closed"))),
        float(bool(reservoir_fields.get("teacher_exact_hit"))),
        float(bool(reservoir_fields.get("teacher_gt_reactant_hit"))),
        _clip01(float(_safe_float(reservoir_fields.get("teacher_route_value"), default=0.0))),
        _clip01(float(_safe_float(reservoir_fields.get("teacher_action_value"), default=0.0))),
    ]
    values = np.concatenate([product_fp, leaf_fp, source_group_vec, source_name_vec, np.asarray(scalars, dtype=np.float32)]).astype(np.float32)
    if input_dim is None:
        return values
    return _resize_vector(values, input_dim)


def reservoir_controller_feature_dim(
    *,
    n_bits: int = 256,
    source_groups: tuple[str, ...] | list[str] = tuple(SOURCE_GROUPS),
) -> int:
    return 2 * int(n_bits) + len(source_groups) + 8 + 42


def load_reservoir_controller_runtime(path: str | Path) -> ReservoirDistilledControllerRuntime:
    return ReservoirDistilledControllerRuntime(path)


def _allocation_with_reason(allocation: SourceAllocation, reason: str) -> SourceAllocation:
    return replace(allocation, policy_reason=reason, fallback_reason=reason)


def _blend_source_weights(
    *,
    controller_weights: dict[str, float],
    fallback_weights: dict[str, float],
    available_sources: list[str] | tuple[str, ...],
) -> dict[str, float]:
    blend = min(1.0, max(0.0, _env_float("AUTOPLANNER_RESERVOIR_SOURCE_BLEND_WEIGHT", 1.0)))
    if blend >= 1.0:
        return dict(controller_weights)
    fallback_total = sum(max(0.0, float(fallback_weights.get(source) or 0.0)) for source in available_sources)
    if fallback_total <= 0:
        return dict(controller_weights)
    mixed = {}
    for source in available_sources:
        controller_value = max(0.0, float(controller_weights.get(source) or 0.0))
        fallback_value = max(0.0, float(fallback_weights.get(source) or 0.0)) / fallback_total
        mixed[source] = blend * controller_value + (1.0 - blend) * fallback_value
    total = sum(mixed.values())
    if total <= 0:
        return dict(controller_weights)
    return {source: value / total for source, value in mixed.items()}


def _min_confidence_for_context(context: Any | None, *, default: float) -> float:
    value = float(default)
    root_value = _env_float("AUTOPLANNER_RESERVOIR_ROOT_MIN_CONFIDENCE", -1.0)
    if root_value >= 0.0 and _context_depth(context) <= 0:
        value = max(value, root_value)
    return value


def _ambiguous_source_fallback_reason(
    *,
    group_map: dict[str, float],
    selected_group: str,
    confidence: float,
    context: Any | None,
) -> str | None:
    if not _env_truthy("AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_FALLBACK"):
        return None
    max_depth = int(_env_float("AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_DEPTH", 0.0))
    if _context_depth(context) > max_depth:
        return None
    if selected_group not in {"chemical", "enzymatic"}:
        return None
    chemical = float(group_map.get("chemical") or 0.0)
    enzymatic = float(group_map.get("enzymatic") or 0.0)
    alternate = enzymatic if selected_group == "chemical" else chemical
    min_alternate = _env_float("AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MIN_ALT_PROB", 0.20)
    max_confidence = _env_float("AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_CONFIDENCE", 0.65)
    max_margin = _env_float("AUTOPLANNER_RESERVOIR_AMBIGUOUS_SOURCE_MAX_MARGIN", 0.20)
    margin = abs(chemical - enzymatic)
    if alternate < min_alternate or confidence > max_confidence or margin > max_margin:
        return None
    return f"reservoir_distilled_fallback:ambiguous_source:{selected_group}:{confidence:.3f}:{margin:.3f}"


def _context_depth(context: Any | None) -> int:
    route_metadata = dict(getattr(context, "route_metadata", {}) or {}) if context is not None else {}
    for value in (
        getattr(context, "depth", None) if context is not None else None,
        route_metadata.get("state_depth"),
        route_metadata.get("depth"),
    ):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            continue
    return 0


def _source_weights_from_group_map(
    *,
    group_map: dict[str, float],
    available_sources: list[str] | tuple[str, ...],
) -> dict[str, float]:
    if not _env_truthy("AUTOPLANNER_RESERVOIR_SPLIT_GROUP_SOURCE_WEIGHT"):
        return {
            source: max(0.0, group_map.get(_source_policy_group(source), 0.0))
            for source in available_sources
        }
    group_counts: dict[str, int] = {}
    for source in available_sources:
        group = _source_policy_group(source)
        group_counts[group] = group_counts.get(group, 0) + 1
    return {
        source: max(0.0, group_map.get(_source_policy_group(source), 0.0)) / max(group_counts.get(_source_policy_group(source), 1), 1)
        for source in available_sources
    }


def _fallback_action_evaluation(
    runtime: Any | None,
    state: RouteTreeState,
    leaf: str,
    actions: list[CandidateAction],
    *,
    stock_checker: StockChecker | None,
) -> Any | None:
    if runtime is None:
        return None
    try:
        return runtime.evaluate(state, leaf, actions, stock_checker=stock_checker)
    except Exception:
        return None


def _fallback_leaf_evaluation(
    runtime: Any | None,
    state: RouteTreeState,
    leaves: list[str],
    *,
    stock_checker: StockChecker | None,
) -> Any | None:
    if runtime is None:
        return None
    try:
        return runtime.score_open_leaves(state, leaves, stock_checker=stock_checker)
    except Exception:
        return None


def _evaluation_with_reason(evaluation: Any, reason: str) -> Any:
    return replace(evaluation, model_active=False, reason=reason)


def _compose_reason(prefix: str, base_eval: Any | None) -> str:
    if base_eval is None or not getattr(base_eval, "reason", ""):
        return prefix
    return f"{prefix}+{base_eval.reason}"


def _blend_prob(base_value: float, controller_value: float) -> float:
    weight = _env_float("AUTOPLANNER_RESERVOIR_VALUE_DELTA_WEIGHT", 0.0)
    return float((1.0 - weight) * float(base_value or 0.0) + weight * float(controller_value or 0.0))


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(__import__("os").environ.get(name, default)))
    except (TypeError, ValueError):
        return float(default)


def _env_truthy(name: str) -> bool:
    return str(__import__("os").environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _resize_vector(values: np.ndarray, size: int) -> np.ndarray:
    if values.shape[-1] == size:
        return values.astype(np.float32)
    if values.shape[-1] > size:
        return values[:size].astype(np.float32)
    out = np.zeros(size, dtype=np.float32)
    out[: values.shape[-1]] = values
    return out


def _candidate_reactants(candidate: dict[str, Any]) -> list[str]:
    reactants: list[str] = []
    main = candidate.get("main_reactant")
    if main:
        reactants.append(str(main))
    reactants.extend(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if rxn and ">>" in str(rxn):
        reactants.extend(part for part in str(rxn).split(">>", 1)[0].split(".") if part)
    out = []
    seen = set()
    for smi in reactants:
        key = canonical_smiles(smi) or smi
        if key in seen:
            continue
        seen.add(key)
        out.append(smi)
    return out


def _stock_fraction(reactants: list[str], *, stock_checker: StockChecker | None) -> float | None:
    if not reactants or stock_checker is None:
        return None
    hits = 0
    for smi in reactants:
        try:
            hits += int(bool(stock_checker(smi)))
        except Exception:
            pass
    return hits / max(len(reactants), 1)


def _safe_choice(labels: list[str], index: int, *, default: str) -> str:
    if 0 <= index < len(labels):
        return str(labels[index])
    return default


def _safe_float(value: Any, *, default: Any = None) -> float:
    if value is None:
        value = default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _ec1(value: Any) -> int:
    try:
        text = str(value or "").split(".", 1)[0]
        return int(text) if text else 0
    except (TypeError, ValueError):
        return 0


def _heavy_atoms(smiles: str | None) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0
