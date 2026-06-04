"""Native ChemEnzy one-step wrapper for AutoPlanner enzyme proposals.

This module keeps ChemEnzy's native MCTS/search loop in charge.  It only wraps
the per-node one-step expansion function so every molecule expanded by the
vendor planner can receive bridge-gated enzyme precedent candidates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_step_quality import (
    EnzymeStepQualityConfig,
    evaluate_enzyme_step_quality,
)
from cascade_planner.cascadeboard.enzyme_precedent_retrieval import retrieve_enzyme_precedents
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction


PLUGIN_MODEL_FULL_NAME = "autoplanner.enzyme_precedent"


@dataclass(frozen=True)
class NativeEnzymePluginConfig:
    enabled: bool = False
    pack_dir: Path = Path("data/bridge_pack_v0")
    top_k: int = 6
    bridge_top_k: int = 8
    max_ec_contexts: int = 2
    require_bridge: bool = True
    require_verifier_pass: bool = True
    enable_sp_v1: bool = True
    sp_v1_hard_gate: bool = True
    min_similarity: float | None = None
    score_scale: float = 1.0
    sp_v1_score_bonus: float = 0.0
    quality_score_bonus: float = 0.0
    max_added: int = 6
    respect_source_policy: bool = False
    require_material_sanity: bool = True
    min_quality_score: float | None = None
    material_max_heavy_gain: int = 3
    material_max_carbon_gain: int = 2
    material_max_hetero_gain: int = 3

    @classmethod
    def from_raw(cls, raw: Any) -> "NativeEnzymePluginConfig":
        if raw is True:
            return cls(enabled=True)
        if not raw:
            return cls(enabled=False)
        if not isinstance(raw, dict):
            return cls(enabled=bool(raw))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            pack_dir=Path(str(raw.get("pack_dir") or cls.pack_dir)),
            top_k=_int(raw.get("top_k"), cls.top_k, lo=0),
            bridge_top_k=_int(raw.get("bridge_top_k"), cls.bridge_top_k, lo=0),
            max_ec_contexts=_int(raw.get("max_ec_contexts"), cls.max_ec_contexts, lo=0, hi=7),
            require_bridge=bool(raw.get("require_bridge", cls.require_bridge)),
            require_verifier_pass=bool(raw.get("require_verifier_pass", cls.require_verifier_pass)),
            enable_sp_v1=bool(raw.get("enable_sp_v1", cls.enable_sp_v1)),
            sp_v1_hard_gate=bool(raw.get("sp_v1_hard_gate", cls.sp_v1_hard_gate)),
            min_similarity=_float_or_none(raw.get("min_similarity")),
            score_scale=_float(raw.get("score_scale"), cls.score_scale),
            sp_v1_score_bonus=_float(raw.get("sp_v1_score_bonus"), cls.sp_v1_score_bonus),
            quality_score_bonus=_float(raw.get("quality_score_bonus"), cls.quality_score_bonus),
            max_added=_int(raw.get("max_added"), cls.max_added, lo=0),
            respect_source_policy=bool(raw.get("respect_source_policy", cls.respect_source_policy)),
            require_material_sanity=bool(raw.get("require_material_sanity", cls.require_material_sanity)),
            min_quality_score=_float_or_none(raw.get("min_quality_score")),
            material_max_heavy_gain=_int(raw.get("material_max_heavy_gain"), cls.material_max_heavy_gain, lo=0),
            material_max_carbon_gain=_int(raw.get("material_max_carbon_gain"), cls.material_max_carbon_gain, lo=0),
            material_max_hetero_gain=_int(raw.get("material_max_hetero_gain"), cls.material_max_hetero_gain, lo=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pack_dir": str(self.pack_dir),
            "top_k": self.top_k,
            "bridge_top_k": self.bridge_top_k,
            "max_ec_contexts": self.max_ec_contexts,
            "require_bridge": self.require_bridge,
            "require_verifier_pass": self.require_verifier_pass,
            "enable_sp_v1": self.enable_sp_v1,
            "sp_v1_hard_gate": self.sp_v1_hard_gate,
            "min_similarity": self.min_similarity,
            "score_scale": self.score_scale,
            "sp_v1_score_bonus": self.sp_v1_score_bonus,
            "quality_score_bonus": self.quality_score_bonus,
            "max_added": self.max_added,
            "respect_source_policy": self.respect_source_policy,
            "require_material_sanity": self.require_material_sanity,
            "min_quality_score": self.min_quality_score,
            "material_max_heavy_gain": self.material_max_heavy_gain,
            "material_max_carbon_gain": self.material_max_carbon_gain,
            "material_max_hetero_gain": self.material_max_hetero_gain,
        }


@dataclass
class NativeEnzymePluginState:
    config: NativeEnzymePluginConfig
    target_smiles: str = ""
    calls: int = 0
    bridge_hit_calls: int = 0
    skipped_no_bridge: int = 0
    retrieved_candidates: int = 0
    sp_v1_scored: int = 0
    sp_v1_accepted: int = 0
    sp_v1_rejected: int = 0
    added_candidates: int = 0
    duplicate_candidates: int = 0
    invalid_candidates: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    source_policy_skips: int = 0
    quality_scored: int = 0
    quality_passed: int = 0
    quality_warned: int = 0
    quality_rejected: int = 0
    material_rejected: int = 0

    def reset_for_target(self, target_smiles: str) -> None:
        config = self.config
        self.__dict__.clear()
        self.__dict__.update(NativeEnzymePluginState(config=config, target_smiles=target_smiles).__dict__)

    def record_error(self, exc: Exception) -> None:
        self.error_count += 1
        if len(self.errors) < 8:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "native_enzyme_plugin.stats.v1",
            "enabled": bool(self.config.enabled),
            "target_smiles": self.target_smiles,
            "config": self.config.to_dict(),
            "calls": self.calls,
            "bridge_hit_calls": self.bridge_hit_calls,
            "skipped_no_bridge": self.skipped_no_bridge,
            "retrieved_candidates": self.retrieved_candidates,
            "sp_v1_scored": self.sp_v1_scored,
            "sp_v1_accepted": self.sp_v1_accepted,
            "sp_v1_rejected": self.sp_v1_rejected,
            "added_candidates": self.added_candidates,
            "duplicate_candidates": self.duplicate_candidates,
            "invalid_candidates": self.invalid_candidates,
            "source_policy_skips": self.source_policy_skips,
            "quality_scored": self.quality_scored,
            "quality_passed": self.quality_passed,
            "quality_warned": self.quality_warned,
            "quality_rejected": self.quality_rejected,
            "material_rejected": self.material_rejected,
            "error_count": self.error_count,
            "errors": list(self.errors),
        }


class NativeEnzymeOneStepWrapper:
    """Append gated enzyme precedent rows to ChemEnzy one-step predictions."""

    def __init__(self, one_step: Any, *, config: NativeEnzymePluginConfig, state: NativeEnzymePluginState) -> None:
        self.one_step = one_step
        self.config = config
        self.state = state
        self._bridge_retriever: Any | None = None
        self._sp_v1: Any | None = None
        self.one_step_models = dict(getattr(one_step, "one_step_models", {}) or {})
        self.one_step_models.setdefault(PLUGIN_MODEL_FULL_NAME, self)

    def run(self, target: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        base = _run_base(self.one_step, target, *args, **kwargs)
        if not self.config.enabled:
            return base
        if self.config.respect_source_policy:
            selected = kwargs.get("select_models")
            if selected is not None and PLUGIN_MODEL_FULL_NAME not in set(selected):
                self.state.source_policy_skips += 1
                return base
        self.state.calls += 1
        try:
            enzyme_rows = self._enzyme_rows(str(target or ""), base)
        except Exception as exc:  # pragma: no cover - plugin must not break ChemEnzy search
            self.state.record_error(exc)
            return base
        if not enzyme_rows:
            return base
        return _append_rows(base, enzyme_rows)

    def _enzyme_rows(self, target: str, base: dict[str, Any]) -> list[dict[str, Any]]:
        if not target or self.config.top_k <= 0 or self.config.max_added <= 0:
            return []
        bridge_hits = self._bridge_retriever_obj().retrieve(
            target,
            top_k=self.config.bridge_top_k,
            require_verifier_pass=self.config.require_verifier_pass,
        )
        if bridge_hits:
            self.state.bridge_hit_calls += 1
        elif self.config.require_bridge:
            self.state.skipped_no_bridge += 1
            return []

        contexts = _bridge_ec1s(bridge_hits)[: self.config.max_ec_contexts]
        if not contexts:
            contexts = [""]
        candidates: list[dict[str, Any]] = []
        for ec1 in contexts:
            rows = retrieve_enzyme_precedents(
                target,
                ec_class=str(ec1),
                top_k=self.config.top_k,
                min_similarity=self.config.min_similarity,
                pool_path=self.config.pack_dir / "enzyme_reaction_pool.parquet",
            )
            candidates.extend(rows)
        self.state.retrieved_candidates += len(candidates)

        seen = _base_reaction_keys(base, target)
        out: list[dict[str, Any]] = []
        for rank, row in enumerate(candidates, start=1):
            action = CandidateAction.from_candidate(target, row, rank=rank, source="enzyme_precedent")
            if _has_hard_validity_flag(action):
                self.state.invalid_candidates += 1
                continue
            key = action.canonical_key
            if key in seen:
                self.state.duplicate_candidates += 1
                continue
            sp_payload = None
            if self.config.enable_sp_v1:
                score = self._sp_v1_obj().score_action(product=target, action=action)
                sp_payload = score.to_dict()
                self.state.sp_v1_scored += 1
                if score.accepted:
                    self.state.sp_v1_accepted += 1
                else:
                    self.state.sp_v1_rejected += 1
                    if self.config.sp_v1_hard_gate:
                        continue
            quality_payload = _action_quality_payload(
                action,
                source_model=PLUGIN_MODEL_FULL_NAME,
                sp_payload=sp_payload,
                config=self.config,
            )
            self.state.quality_scored += 1
            material_passed = bool((quality_payload.get("material_sanity") or {}).get("passed"))
            if self.config.require_material_sanity and not material_passed:
                self.state.material_rejected += 1
                self.state.quality_rejected += 1
                continue
            min_quality = self.config.min_quality_score
            if min_quality is not None and float(quality_payload.get("quality_score") or 0.0) < float(min_quality):
                self.state.quality_rejected += 1
                continue
            if quality_payload.get("decision") == "reject":
                self.state.quality_rejected += 1
                continue
            if quality_payload.get("decision") == "pass":
                self.state.quality_passed += 1
            else:
                self.state.quality_warned += 1
            out.append(
                _action_to_vendor_row(
                    action,
                    rank=len(out) + 1,
                    sp_payload=sp_payload,
                    quality_payload=quality_payload,
                    config=self.config,
                )
            )
            seen.add(key)
            self.state.added_candidates += 1
            if len(out) >= self.config.max_added:
                break
        return out

    def _bridge_retriever_obj(self) -> Any:
        if self._bridge_retriever is None:
            self._bridge_retriever = _make_bridge_retriever(self.config.pack_dir)
        return self._bridge_retriever

    def _sp_v1_obj(self) -> Any:
        if self._sp_v1 is None:
            self._sp_v1 = _make_sp_v1_scorer()
        return self._sp_v1


def native_enzyme_plugin_config_from_flags(search_flags: dict[str, Any] | None) -> NativeEnzymePluginConfig:
    flags = dict(search_flags or {})
    raw = (
        flags.get("native_enzyme_plugin")
        if "native_enzyme_plugin" in flags
        else flags.get("autoplanner_native_enzyme_plugin")
    )
    return NativeEnzymePluginConfig.from_raw(raw)


def configure_native_enzyme_plugin(api_module: Any, config: NativeEnzymePluginConfig) -> NativeEnzymePluginState | None:
    original = getattr(api_module, "_autoplanner_original_prepare_molstar_planner", None)
    if original is None:
        original = api_module.prepare_molstar_planner
        api_module._autoplanner_original_prepare_molstar_planner = original
    if not config.enabled:
        api_module.prepare_molstar_planner = original
        return None

    state = NativeEnzymePluginState(config=config)

    def patched_prepare_molstar_planner(*args: Any, **kwargs: Any) -> Any:
        if args:
            one_step = args[0]
            args = (NativeEnzymeOneStepWrapper(one_step, config=config, state=state), *args[1:])
        elif "one_step" in kwargs:
            kwargs = dict(kwargs)
            kwargs["one_step"] = NativeEnzymeOneStepWrapper(kwargs["one_step"], config=config, state=state)
        return original(*args, **kwargs)

    api_module.prepare_molstar_planner = patched_prepare_molstar_planner
    return state


def reset_native_enzyme_plugin_state(planner: Any, target_smiles: str) -> None:
    state = getattr(planner, "_autoplanner_native_enzyme_plugin_state", None)
    if isinstance(state, NativeEnzymePluginState):
        state.reset_for_target(str(target_smiles or ""))


def native_enzyme_plugin_stats(planner: Any) -> dict[str, Any] | None:
    state = getattr(planner, "_autoplanner_native_enzyme_plugin_state", None)
    if isinstance(state, NativeEnzymePluginState):
        return state.to_dict()
    return None


def _action_to_vendor_row(
    action: CandidateAction,
    *,
    rank: int,
    sp_payload: dict[str, Any] | None,
    quality_payload: dict[str, Any] | None,
    config: NativeEnzymePluginConfig,
) -> dict[str, Any]:
    score = max(1e-6, min(0.999999, float(action.raw_score or 0.0) * float(config.score_scale or 1.0)))
    if sp_payload and config.sp_v1_score_bonus:
        margin = float(sp_payload.get("score") or 0.0) - float(sp_payload.get("threshold") or 0.0)
        score = max(1e-6, min(0.999999, score + max(0.0, margin) * float(config.sp_v1_score_bonus)))
    if quality_payload and config.quality_score_bonus:
        quality_score = max(0.0, min(1.0, float(quality_payload.get("quality_score") or 0.0)))
        score = max(1e-6, min(0.999999, score + quality_score * float(config.quality_score_bonus)))
    reactants = ".".join(str(smi) for smi in action.reactants if smi)
    template = {
        "model_full_name": PLUGIN_MODEL_FULL_NAME,
        "source": "enzyme_precedent",
        "rank": rank,
        "ec": action.ec,
        "reaction_type": action.reaction_type,
        "autoplanner_native_enzyme_plugin": True,
        "evidence": dict(action.metadata.get("evidence") or {}),
        "enzyme_sp_verifier_v1": sp_payload,
        "autoplanner_enzyme_quality_v1": quality_payload,
    }
    return {
        "reactants": reactants,
        "scores": score,
        "costs": -math.log(max(score, 1e-6)),
        "template": template,
        "model_full_name": PLUGIN_MODEL_FULL_NAME,
        "weight": 1.0,
        "reaction_domains": "enzymatic",
        "enzyme_evidence_confidences": (quality_payload or {}).get("quality_score"),
    }


def _action_quality_payload(
    action: CandidateAction,
    *,
    source_model: str,
    sp_payload: dict[str, Any] | None,
    config: NativeEnzymePluginConfig,
) -> dict[str, Any]:
    return evaluate_enzyme_step_quality(
        product_smiles=action.product,
        reactants=action.reactants,
        source_model=source_model,
        template={
            "model_full_name": source_model,
            "source": action.source,
            "ec": action.ec,
            "evidence": dict(action.metadata.get("evidence") or {}),
            "enzyme_sp_verifier_v1": sp_payload,
        },
        sp_payload=sp_payload,
        ec_numbers=[action.ec] if action.ec else None,
        config=EnzymeStepQualityConfig(
            max_heavy_gain=config.material_max_heavy_gain,
            max_carbon_gain=config.material_max_carbon_gain,
            max_hetero_gain=config.material_max_hetero_gain,
            reject_on_material_failure=config.require_material_sanity,
            reject_below_score=config.min_quality_score,
        ),
    )


def _make_bridge_retriever(pack_dir: Path) -> Any:
    from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0

    return BridgeRetrieverV0(pack_dir, scorer=None)


def _make_sp_v1_scorer() -> Any:
    from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer

    return EnzymeSPVerifierV1Scorer()


def _append_rows(base: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {key: _as_list(value) for key, value in dict(base or {}).items()}
    for key in ("reactants", "scores", "costs", "template", "model_full_name", "weight"):
        out.setdefault(key, [])
    _complete_costs_from_scores(out)
    for row in rows:
        current_len = len(out.get("scores") or out.get("reactants") or [])
        for key in row:
            if key not in out:
                out[key] = [None for _ in range(current_len)]
            elif len(out[key]) < current_len:
                out[key].extend(None for _ in range(current_len - len(out[key])))
        for key, value in row.items():
            out.setdefault(key, [])
            out[key].append(value)
    return out


def _complete_costs_from_scores(out: dict[str, list[Any]]) -> None:
    current_len = len(out.get("scores") or out.get("reactants") or [])
    costs = out.setdefault("costs", [])
    while len(costs) < current_len:
        costs.append(_score_to_cost(_at(out.get("scores") or [], len(costs), 0.0)))


def _score_to_cost(score: Any) -> float:
    value = _float(score, 0.0) or 0.0
    return -math.log(max(1e-6, min(0.999999, float(value))))


def _run_base(one_step: Any, target: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return dict(one_step.run(target, *args, **kwargs) or {})
    except TypeError:
        topk = kwargs.get("topk")
        if topk is None and args:
            topk = args[0]
        return dict(one_step.run(target, topk=topk) or {})


def _base_reaction_keys(base: dict[str, Any], target: str) -> set[str]:
    keys: set[str] = set()
    for reactant_text in list((base or {}).get("reactants") or []):
        lhs = ".".join(sorted(_split_reactants(str(reactant_text or ""))))
        rxn = f"{lhs}>>{target}"
        key = canonical_reaction(rxn) or rxn
        keys.add(key)
    return keys


def _has_hard_validity_flag(action: CandidateAction) -> bool:
    hard = {"no_reactants", "no_main_reactant", "product_mismatch", "self_loop"}
    return any(flag in hard for flag in action.validity_flags)


def _bridge_ec1s(bridge_hits: list[Any]) -> list[str]:
    out: list[str] = []
    for hit in bridge_hits:
        for ec in getattr(hit, "enzyme_ec_sample", ()) or ():
            head = str(ec or "").split(".", 1)[0]
            if head in {"1", "2", "3", "4", "5", "6", "7"} and head not in out:
                out.append(head)
    return out


def _split_reactants(value: str) -> list[str]:
    out = []
    seen = set()
    for part in str(value or "").split("."):
        smi = part.strip()
        key = canonical_smiles(smi) or smi
        if smi and key not in seen:
            seen.add(key)
            out.append(smi)
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _at(values: list[Any], idx: int, default: Any) -> Any:
    return values[idx] if idx < len(values) else default


def _int(value: Any, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
