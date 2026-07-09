"""Runtime controller for route-tree policy/value scoring."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from cascade_planner.route_tree.features import action_feature_matrix, node_feature_matrix, state_tensors
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState
from cascade_planner.vnext.schema import SOURCE_BUDGET_GROUPS
from cascade_planner.vnext.models import RouteStateTransformer, SearchPolicyNetwork


DEFAULT_SEARCH_POLICY = Path(
    "results/shared/controller_v2_20260512/fullrun/train_v8/open_leaf_policy/stock_aware_search_policy.pt"
)
_RUNTIME_CACHE: dict[str, "RouteTreeRuntime | Any | None"] = {}


StockChecker = Callable[[str], bool]


@dataclass
class RouteTreeEvaluation:
    action_scores: list[float]
    node_scores: list[float] = field(default_factory=list)
    value_logit: float | None = None
    solved_logit: float | None = None
    stock_logit: float | None = None
    progressive_logit: float | None = None
    compatibility_logit: float | None = None
    route_value: float = 0.0
    solved_prob: float = 0.0
    stock_closed_prob: float = 0.0
    progressive_prob: float = 0.0
    compatibility_prob: float = 0.0
    bottleneck_scores: list[float] = field(default_factory=list)
    source_budgets: dict[str, float] = field(default_factory=dict)
    value_calibrated: bool = False
    model_active: bool = False
    reason: str = ""


class RouteTreeRuntime:
    """Load the trained vNext search policy as a route-tree controller."""

    def __init__(
        self,
        path: str | Path = DEFAULT_SEARCH_POLICY,
        *,
        action_policy_path: str | Path | None = None,
        load_action_override: bool = True,
    ):
        self.path = Path(path)
        ckpt = torch.load(str(self.path), map_location="cpu")
        schema = ckpt.get("feature_schema") or (ckpt.get("metadata") or {}).get("feature_schema") or {}
        meta = ckpt.get("metadata") or {}
        kind = schema.get("model_kind") or meta.get("model_kind")
        if kind != "search_policy":
            raise ValueError(f"unsupported route-tree runtime checkpoint kind: {kind}")
        self.schema = schema
        self.n_bits = int(schema.get("n_bits") or 128)
        self.max_candidates = int(schema.get("max_candidates") or 24)
        self.max_steps = int(schema.get("max_steps") or 8)
        self.max_open_leaves = int(schema.get("max_open_leaves") or self.max_steps)
        self.step_token_dim = int(schema.get("step_token_dim") or 48)
        self.route_feature_dim = int(schema.get("route_feature_dim") or 18)
        self.action_feature_dim = int(schema.get("action_feature_dim") or 290)
        self.node_feature_dim = int(schema.get("node_feature_dim") or 1)
        self.value_supervised = bool(schema.get("value_head_supervised") or meta.get("value_head_supervised"))
        self.value_calibration = schema.get("value_calibration") or meta.get("value_calibration") or {}
        self.value_calibrated = bool(
            schema.get("value_calibrated")
            or meta.get("value_calibrated")
            or self.value_calibration.get("calibrated")
        )
        self.node_policy_supervised = bool(schema.get("node_policy_supervised") or meta.get("node_policy_supervised"))
        self.node_label_target = str(schema.get("node_label_target") or meta.get("node_label_target") or "")
        self.budget_head_supervised = bool(schema.get("budget_head_supervised") or meta.get("budget_head_supervised"))
        self.source_budget_groups = list(
            schema.get("source_budget_groups")
            or meta.get("source_budget_groups")
            or SOURCE_BUDGET_GROUPS
        )
        d_model = int((ckpt.get("model_config") or {}).get("d_model") or meta.get("d_model") or 128)
        route_model = RouteStateTransformer(
            step_token_dim=self.step_token_dim,
            route_feature_dim=self.route_feature_dim,
            max_steps=self.max_steps,
            d_model=d_model,
        )
        self.model = SearchPolicyNetwork(
            route_model=route_model,
            action_feature_dim=self.action_feature_dim,
            node_feature_dim=self.node_feature_dim,
            d_model=d_model,
            n_source_budgets=len(self.source_budget_groups),
        )
        self.load_report = _load_compatible_state(self.model, ckpt["state_dict"])
        self.model.eval()
        self.action_override: RouteTreeRuntime | None = None
        self.action_override_error: str = ""
        if load_action_override:
            override_raw = str(action_policy_path or os.environ.get("AUTOPLANNER_ROUTE_TREE_ACTION_POLICY") or "").strip()
            override_path = Path(override_raw) if override_raw else None
            if override_path is not None and override_path != self.path:
                if override_path.exists():
                    try:
                        self.action_override = RouteTreeRuntime(
                            override_path,
                            load_action_override=False,
                        )
                    except Exception as exc:
                        self.action_override_error = f"{type(exc).__name__}:{exc}"
                else:
                    self.action_override_error = "missing_checkpoint"

    def evaluate(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        stock_checker: StockChecker | None = None,
    ) -> RouteTreeEvaluation:
        if not actions:
            return RouteTreeEvaluation(action_scores=[], model_active=True, reason="no_actions")
        try:
            step_tokens, step_mask, route_features = state_tensors(
                state,
                max_steps=self.max_steps,
                stock_checker=stock_checker,
            )
            action_x, action_mask = action_feature_matrix(
                leaf,
                actions,
                n_bits=self.n_bits,
                max_candidates=self.max_candidates,
                stock_checker=stock_checker,
            )
            action_x = _resize_last_dim_np(action_x, self.action_feature_dim)
            node_x, node_mask = node_feature_matrix(
                state,
                [leaf],
                n_bits=self.n_bits,
                max_open_leaves=max(1, self.max_open_leaves),
                stock_checker=stock_checker,
            )
            node_x = _resize_last_dim_np(node_x, self.node_feature_dim)
            with torch.no_grad():
                out = self.model(
                    torch.tensor(step_tokens[None, :, :], dtype=torch.float32),
                    torch.tensor(step_mask[None, :], dtype=torch.bool),
                    torch.tensor(route_features[None, :], dtype=torch.float32),
                    torch.tensor(action_x[None, :, :], dtype=torch.float32),
                    torch.tensor(action_mask[None, :], dtype=torch.bool),
                    torch.tensor(node_x[None, :, :], dtype=torch.float32),
                    torch.tensor(node_mask[None, :], dtype=torch.bool),
                )
            raw_logits = out["action_logits"][0, : min(len(actions), self.max_candidates)].cpu().tolist()
            if len(actions) > len(raw_logits):
                raw_logits.extend([min(raw_logits) - 1.0 if raw_logits else 0.0] * (len(actions) - len(raw_logits)))
            logits = _blend_policy_and_heuristic_scores(
                raw_logits[: len(actions)],
                heuristic_action_scores(leaf, actions, stock_checker=stock_checker),
                model_weight=_env_float(
                    "AUTOPLANNER_ROUTE_TREE_ACTION_LOGIT_WEIGHT",
                    0.75 if self.value_calibrated else 0.35,
                ),
                heuristic_weight=_env_float("AUTOPLANNER_ROUTE_TREE_HEURISTIC_ACTION_WEIGHT", 1.0),
            )
            reason = "search_policy"
            if self.action_override is not None and _action_override_allowed(state, actions, stock_checker=stock_checker):
                override_eval = self.action_override.evaluate(state, leaf, actions, stock_checker=stock_checker)
                if override_eval.model_active and len(override_eval.action_scores) == len(actions):
                    logits = override_eval.action_scores
                    reason = "search_policy+action_override"
                elif override_eval.reason:
                    reason = f"search_policy+action_override_fallback:{override_eval.reason}"
            elif self.action_override is not None:
                reason = "search_policy+action_override_gated"
            elif self.action_override_error:
                reason = f"search_policy+action_override_unavailable:{self.action_override_error}"
            value_logit = float(out["value_logit"][0].item())
            solved_logit = float(out["solved_logit"][0].item())
            stock_logit = float(out["stock_logit"][0].item())
            progressive_logit = float(out["progressive_logit"][0].item())
            compatibility_logit = float(out["compatibility_logit"][0].item())
            route_value = (
                float(_calibrated_sigmoid(out["value_logit"], self.value_calibration)[0].item())
                if self.value_supervised and self.value_calibrated
                else 0.0
            )
            budget_probs = torch.softmax(out["budget_logits"], dim=-1)[0].cpu().tolist()
            budgets = (
                {
                    name: float(budget_probs[idx])
                    for idx, name in enumerate(self.source_budget_groups[: len(budget_probs)])
                }
                if self.budget_head_supervised
                else {}
            )
            return RouteTreeEvaluation(
                action_scores=[float(x) for x in logits[: len(actions)]],
                node_scores=out["node_policy_logits"][0, :1].cpu().tolist(),
                value_logit=value_logit,
                solved_logit=solved_logit,
                stock_logit=stock_logit,
                progressive_logit=progressive_logit,
                compatibility_logit=compatibility_logit,
                route_value=route_value,
                solved_prob=float(torch.sigmoid(out["solved_logit"])[0].item()),
                stock_closed_prob=float(torch.sigmoid(out["stock_logit"])[0].item()),
                progressive_prob=float(torch.sigmoid(out["progressive_logit"])[0].item()),
                compatibility_prob=float(torch.sigmoid(out["compatibility_logit"])[0].item()),
                bottleneck_scores=torch.sigmoid(out["bottleneck_logits"])[0].cpu().tolist(),
                source_budgets=budgets,
                value_calibrated=self.value_calibrated,
                model_active=True,
                reason=reason,
            )
        except Exception as exc:
            scores = heuristic_action_scores(leaf, actions, stock_checker=stock_checker)
            return RouteTreeEvaluation(action_scores=scores, model_active=False, reason=f"runtime_error:{type(exc).__name__}")

    def score_open_leaves(
        self,
        state: RouteTreeState,
        leaves: list[str],
        *,
        stock_checker: StockChecker | None = None,
    ) -> RouteTreeEvaluation:
        if not leaves:
            return RouteTreeEvaluation(action_scores=[], node_scores=[], model_active=True, reason="no_leaves")
        if not self.node_policy_supervised:
            return RouteTreeEvaluation(
                action_scores=[],
                node_scores=heuristic_node_scores(leaves, stock_checker=stock_checker),
                model_active=False,
                reason="node_policy_not_supervised",
            )
        try:
            step_tokens, step_mask, route_features = state_tensors(
                state,
                max_steps=self.max_steps,
                stock_checker=stock_checker,
            )
            node_x, node_mask = node_feature_matrix(
                state,
                leaves,
                n_bits=self.n_bits,
                max_open_leaves=self.max_open_leaves,
                stock_checker=stock_checker,
            )
            node_x = _resize_last_dim_np(node_x, self.node_feature_dim)
            action_x = np.zeros((1, self.action_feature_dim), dtype=np.float32)
            action_mask = np.ones(1, dtype=np.float32)
            with torch.no_grad():
                out = self.model(
                    torch.tensor(step_tokens[None, :, :], dtype=torch.float32),
                    torch.tensor(step_mask[None, :], dtype=torch.bool),
                    torch.tensor(route_features[None, :], dtype=torch.float32),
                    torch.tensor(action_x[None, :, :], dtype=torch.float32),
                    torch.tensor(action_mask[None, :], dtype=torch.bool),
                    torch.tensor(node_x[None, :, :], dtype=torch.float32),
                    torch.tensor(node_mask[None, :], dtype=torch.bool),
                )
            raw_scores = out["node_policy_logits"][0, : min(len(leaves), self.max_open_leaves)].cpu().tolist()
            if len(leaves) > len(raw_scores):
                raw_scores.extend([min(raw_scores) - 1.0 if raw_scores else 0.0] * (len(leaves) - len(raw_scores)))
            use_node_model = bool(self.node_label_target == "stock_aware_leaf_utility" or self.value_calibrated)
            scores = _blend_policy_and_heuristic_scores(
                raw_scores[: len(leaves)],
                heuristic_node_scores(leaves, stock_checker=stock_checker),
                model_weight=_env_float(
                    "AUTOPLANNER_ROUTE_TREE_NODE_LOGIT_WEIGHT",
                    1.0 if use_node_model else 0.0,
                ),
                heuristic_weight=_env_float("AUTOPLANNER_ROUTE_TREE_HEURISTIC_NODE_WEIGHT", 0.0 if use_node_model else 1.0),
            )
            value_logit = float(out["value_logit"][0].item())
            solved_logit = float(out["solved_logit"][0].item())
            stock_logit = float(out["stock_logit"][0].item())
            progressive_logit = float(out["progressive_logit"][0].item())
            compatibility_logit = float(out["compatibility_logit"][0].item())
            return RouteTreeEvaluation(
                action_scores=[],
                node_scores=[float(x) for x in scores[: len(leaves)]],
                value_logit=value_logit,
                solved_logit=solved_logit,
                stock_logit=stock_logit,
                progressive_logit=progressive_logit,
                compatibility_logit=compatibility_logit,
                route_value=(
                    float(_calibrated_sigmoid(out["value_logit"], self.value_calibration)[0].item())
                    if self.value_supervised and self.value_calibrated
                    else 0.0
                ),
                solved_prob=float(torch.sigmoid(out["solved_logit"])[0].item()),
                stock_closed_prob=float(torch.sigmoid(out["stock_logit"])[0].item()),
                progressive_prob=float(torch.sigmoid(out["progressive_logit"])[0].item()),
                compatibility_prob=float(torch.sigmoid(out["compatibility_logit"])[0].item()),
                bottleneck_scores=torch.sigmoid(out["bottleneck_logits"])[0].cpu().tolist(),
                source_budgets=(
                    {
                        name: float(value)
                        for name, value in zip(
                            self.source_budget_groups,
                            torch.softmax(out["budget_logits"], dim=-1)[0].cpu().tolist(),
                        )
                    }
                    if self.budget_head_supervised
                    else {}
                ),
                value_calibrated=self.value_calibrated,
                model_active=True,
                reason="node_policy",
            )
        except Exception as exc:
            return RouteTreeEvaluation(
                action_scores=[],
                node_scores=heuristic_node_scores(leaves, stock_checker=stock_checker),
                model_active=False,
                reason=f"runtime_error:{type(exc).__name__}",
            )


def default_route_tree_runtime() -> RouteTreeRuntime | Any | None:
    if not _env_truthy("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER"):
        return None
    if _env_truthy("AUTOPLANNRELLM_ENABLE") and _env_truthy_default("AUTOPLANNRELLM_LLM_SELECTION", True):
        base_runtime = _default_route_tree_runtime_without_autoplannrellm()
        cache_id = (
            f"autoplannrellm::base={id(base_runtime)}"
            f"::model={os.environ.get('DEEPSEEK_MODEL') or ''}"
            f"::cache={os.environ.get('AUTOPLANNRELLM_CACHE') or ''}"
        )
        if cache_id in _RUNTIME_CACHE:
            return _RUNTIME_CACHE[cache_id]
        try:
            from AUTOPLANNRELLM.controller import DeepSeekSelectionController

            runtime = DeepSeekSelectionController(fallback_runtime=base_runtime)
        except Exception:
            runtime = base_runtime
        _RUNTIME_CACHE[cache_id] = runtime
        return runtime
    return _default_route_tree_runtime_without_autoplannrellm()


def _default_route_tree_runtime_without_autoplannrellm() -> RouteTreeRuntime | Any | None:
    reservoir_path = os.environ.get("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER")
    if reservoir_path:
        policy_path = os.environ.get("AUTOPLANNER_ROUTE_TREE_POLICY") or str(DEFAULT_SEARCH_POLICY)
        action_path = os.environ.get("AUTOPLANNER_ROUTE_TREE_ACTION_POLICY") or ""
        key = f"reservoir:{reservoir_path}::policy={policy_path}::action={action_path}"
        if key in _RUNTIME_CACHE:
            return _RUNTIME_CACHE[key]
        base_runtime = _default_route_tree_runtime_without_reservoir()
        if not Path(reservoir_path).exists():
            from cascade_planner.route_tree.reservoir_distilled import UnavailableReservoirRouteTreeRuntime

            runtime = UnavailableReservoirRouteTreeRuntime("missing_checkpoint", fallback_runtime=base_runtime)
            _RUNTIME_CACHE[key] = runtime
            return runtime
        try:
            from cascade_planner.route_tree.reservoir_distilled import ReservoirDistilledControllerRuntime

            runtime = ReservoirDistilledControllerRuntime(reservoir_path, fallback_runtime=base_runtime)
        except Exception as exc:
            from cascade_planner.route_tree.reservoir_distilled import UnavailableReservoirRouteTreeRuntime

            runtime = UnavailableReservoirRouteTreeRuntime(
                f"{type(exc).__name__}:load_failed",
                fallback_runtime=base_runtime,
            )
        _RUNTIME_CACHE[key] = runtime
        return runtime

    return _default_route_tree_runtime_without_reservoir()


def _default_route_tree_runtime_without_reservoir() -> RouteTreeRuntime | None:
    path = Path(os.environ.get("AUTOPLANNER_ROUTE_TREE_POLICY") or DEFAULT_SEARCH_POLICY)
    action_path = os.environ.get("AUTOPLANNER_ROUTE_TREE_ACTION_POLICY") or ""
    key = f"{path}::action={action_path}"
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    if not path.exists():
        _RUNTIME_CACHE[key] = None
        return None
    try:
        _RUNTIME_CACHE[key] = RouteTreeRuntime(path)
    except Exception:
        _RUNTIME_CACHE[key] = None
    return _RUNTIME_CACHE[key]


def heuristic_action_scores(
    leaf: str,
    actions: list[CandidateAction],
    *,
    stock_checker: StockChecker | None = None,
) -> list[float]:
    return [float(action.raw_score or 0.0) for action in actions]


def heuristic_node_scores(
    leaves: list[str],
    *,
    stock_checker: StockChecker | None = None,
) -> list[float]:
    scores: list[float] = []
    for leaf in leaves:
        if stock_checker is not None:
            try:
                if bool(stock_checker(leaf)):
                    scores.append(0.0)
                    continue
            except Exception:
                pass
        scores.append(math.log1p(float(max(1, _heavy_atoms(leaf)))))
    return scores


def _load_compatible_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped_shape_mismatch = [
        key
        for key, value in state.items()
        if key in current and tuple(current[key].shape) != tuple(value.shape)
    ]
    result = model.load_state_dict(compatible, strict=False)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def _resize_last_dim_np(array: np.ndarray, size: int) -> np.ndarray:
    if array.shape[-1] == size:
        return array
    if array.shape[-1] > size:
        return array[..., :size].astype(np.float32)
    pad_width = [(0, 0) for _ in range(array.ndim)]
    pad_width[-1] = (0, size - array.shape[-1])
    return np.pad(array, pad_width, mode="constant").astype(np.float32)


def _calibrated_sigmoid(logit: torch.Tensor, calibration: dict[str, object]) -> torch.Tensor:
    try:
        temperature = float(calibration.get("temperature", 1.0)) if calibration else 1.0
    except (TypeError, ValueError):
        temperature = 1.0
    return torch.sigmoid(logit / max(temperature, 1e-6))


def _blend_policy_and_heuristic_scores(
    logits: list[float],
    heuristic_scores: list[float],
    *,
    model_weight: float,
    heuristic_weight: float,
) -> list[float]:
    """Use policy logits as bounded guidance until the value path is frozen."""
    if not logits:
        return []
    if not heuristic_scores or heuristic_weight <= 0:
        return [float(x) for x in logits]
    out = []
    for idx, logit in enumerate(logits):
        heuristic = heuristic_scores[idx] if idx < len(heuristic_scores) else 0.0
        out.append(float(heuristic_weight * heuristic + model_weight * np.tanh(float(logit))))
    return out


def _action_override_allowed(
    state: RouteTreeState,
    actions: list[CandidateAction],
    *,
    stock_checker: StockChecker | None = None,
) -> bool:
    min_depth = _env_int("AUTOPLANNER_ROUTE_TREE_ACTION_OVERRIDE_MIN_DEPTH", 0)
    if int(state.depth or 0) < min_depth:
        return False
    if _env_truthy("AUTOPLANNER_ROUTE_TREE_ACTION_OVERRIDE_STOCK_ONLY"):
        return _any_stock_closing_action(actions, stock_checker=stock_checker)
    return True


def _any_stock_closing_action(
    actions: list[CandidateAction],
    *,
    stock_checker: StockChecker | None = None,
) -> bool:
    if stock_checker is None:
        return False
    for action in actions:
        reactants = [smi for smi in action.reactants if smi]
        if not reactants:
            continue
        hits = 0
        for smi in reactants:
            try:
                hits += int(bool(stock_checker(smi)))
            except Exception:
                pass
        if hits == len(reactants):
            return True
    return False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _heavy_atoms(smiles: str | None) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0
