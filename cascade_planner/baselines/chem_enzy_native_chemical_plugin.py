"""Native ChemEnzy one-step wrapper for AutoPlanner chemical proposals.

The wrapper keeps ChemEnzy's native search loop in charge and only appends
conservative chemical tail candidates at the vendor one-step expansion point.
The first plugin source is the audited GraphFP-first dual-tower supplement:
native GraphFP rows stay in front, dual-tower rows are added only when they are
not already present in ChemEnzy's own one-step output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.baselines.graphfp_dualtower_fusion import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPLATE_VECTOR_CACHE,
    DEFAULT_TEMPLATES_INDEX,
    GraphFPDualTowerFusion,
    GraphFPDualTowerFusionConfig,
    fuse_graphfp_dualtower_rows,
)
from cascade_planner.baselines.proposal_gate import evaluate_step_candidate
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles


PLUGIN_MODEL_FULL_NAME = "autoplanner.chemical_graphfp_dualtower"


@dataclass(frozen=True)
class NativeChemicalPluginConfig:
    enabled: bool = False
    top_k: int = 8
    max_added: int = 8
    dual_top_k: int = 100
    graphfp_top_k: int = 50
    fusion_mode: str = "graphfp_first"
    score_scale: float = 0.75
    require_proposal_gate: bool = True
    model_path: Path = DEFAULT_MODEL_PATH
    template_vector_cache: Path = DEFAULT_TEMPLATE_VECTOR_CACHE
    templates_index: Path = DEFAULT_TEMPLATES_INDEX
    base_model_full_name: str = ""
    device: str | None = None
    template_batch_size: int = 4096

    @classmethod
    def from_raw(cls, raw: Any) -> "NativeChemicalPluginConfig":
        if raw is True:
            return cls(enabled=True)
        if not raw:
            return cls(enabled=False)
        if not isinstance(raw, dict):
            return cls(enabled=bool(raw))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            top_k=_int(raw.get("top_k"), cls.top_k, lo=0),
            max_added=_int(raw.get("max_added"), cls.max_added, lo=0),
            dual_top_k=_int(raw.get("dual_top_k"), cls.dual_top_k, lo=1),
            graphfp_top_k=_int(raw.get("graphfp_top_k"), cls.graphfp_top_k, lo=1),
            fusion_mode=str(raw.get("fusion_mode") or cls.fusion_mode),
            score_scale=_float(raw.get("score_scale"), cls.score_scale),
            require_proposal_gate=bool(raw.get("require_proposal_gate", cls.require_proposal_gate)),
            model_path=Path(str(raw.get("model_path") or cls.model_path)),
            template_vector_cache=Path(str(raw.get("template_vector_cache") or cls.template_vector_cache)),
            templates_index=Path(str(raw.get("templates_index") or cls.templates_index)),
            base_model_full_name=str(raw.get("base_model_full_name") or ""),
            device=str(raw.get("device") or "") or None,
            template_batch_size=_int(raw.get("template_batch_size"), cls.template_batch_size, lo=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "max_added": self.max_added,
            "dual_top_k": self.dual_top_k,
            "graphfp_top_k": self.graphfp_top_k,
            "fusion_mode": self.fusion_mode,
            "score_scale": self.score_scale,
            "require_proposal_gate": self.require_proposal_gate,
            "model_path": str(self.model_path),
            "template_vector_cache": str(self.template_vector_cache),
            "templates_index": str(self.templates_index),
            "base_model_full_name": self.base_model_full_name,
            "device": self.device,
            "template_batch_size": self.template_batch_size,
        }


@dataclass
class NativeChemicalPluginState:
    config: NativeChemicalPluginConfig
    target_smiles: str = ""
    calls: int = 0
    base_candidates: int = 0
    graphfp_base_candidates: int = 0
    dual_candidates: int = 0
    fused_candidates: int = 0
    added_candidates: int = 0
    duplicate_candidates: int = 0
    invalid_candidates: int = 0
    proposal_gate_scored: int = 0
    proposal_gate_kept: int = 0
    proposal_gate_rejected: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)

    def reset_for_target(self, target_smiles: str) -> None:
        config = self.config
        self.__dict__.clear()
        self.__dict__.update(NativeChemicalPluginState(config=config, target_smiles=target_smiles).__dict__)

    def record_error(self, exc: Exception) -> None:
        self.error_count += 1
        if len(self.errors) < 8:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "native_chemical_plugin.stats.v1",
            "enabled": bool(self.config.enabled),
            "target_smiles": self.target_smiles,
            "config": self.config.to_dict(),
            "calls": self.calls,
            "base_candidates": self.base_candidates,
            "graphfp_base_candidates": self.graphfp_base_candidates,
            "dual_candidates": self.dual_candidates,
            "fused_candidates": self.fused_candidates,
            "added_candidates": self.added_candidates,
            "duplicate_candidates": self.duplicate_candidates,
            "invalid_candidates": self.invalid_candidates,
            "proposal_gate_scored": self.proposal_gate_scored,
            "proposal_gate_kept": self.proposal_gate_kept,
            "proposal_gate_rejected": self.proposal_gate_rejected,
            "error_count": self.error_count,
            "errors": list(self.errors),
        }


class NativeChemicalOneStepWrapper:
    """Append GraphFP-first dual-tower chemical tail rows to native output."""

    def __init__(self, one_step: Any, *, config: NativeChemicalPluginConfig, state: NativeChemicalPluginState) -> None:
        self.one_step = one_step
        self.config = config
        self.state = state
        self._fusion: Any | None = None
        self.one_step_models = dict(getattr(one_step, "one_step_models", {}) or {})
        self.one_step_models.setdefault(PLUGIN_MODEL_FULL_NAME, self)

    def run(self, target: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        base = _run_base(self.one_step, target, *args, **kwargs)
        if not self.config.enabled:
            return base
        self.state.calls += 1
        try:
            rows = self._chemical_rows(str(target or ""), base)
        except Exception as exc:  # pragma: no cover - plugin must not break ChemEnzy search
            self.state.record_error(exc)
            return base
        if not rows:
            return base
        return _append_rows(base, rows)

    def _chemical_rows(self, target: str, base: dict[str, Any]) -> list[dict[str, Any]]:
        if not target or self.config.top_k <= 0 or self.config.max_added <= 0:
            return []
        base_rows = _base_rows(
            target,
            base,
            default_model_full_name=_default_model_full_name(
                self.one_step_models,
                fallback=self.config.base_model_full_name,
            ),
        )
        self.state.base_candidates += len(base_rows)
        graphfp_rows = [row for row in base_rows if _is_graphfp_row(row)]
        self.state.graphfp_base_candidates += len(graphfp_rows)
        if not graphfp_rows:
            return []
        dual_rows = self._fusion_obj().dual_rows(target)[: self.config.dual_top_k]
        self.state.dual_candidates += len(dual_rows)
        if not dual_rows:
            return []
        fused = fuse_graphfp_dualtower_rows(
            product=target,
            base_rows=graphfp_rows,
            dual_rows=dual_rows,
            output_k=max(self.config.top_k, len(graphfp_rows) + self.config.max_added),
            mode=self.config.fusion_mode,
        )
        self.state.fused_candidates += len(fused)
        seen = _base_reaction_keys(base, target)
        out: list[dict[str, Any]] = []
        for row in fused:
            if row.get("native_rank") is not None:
                continue
            reactants = _row_reactants(row)
            if not reactants:
                self.state.invalid_candidates += 1
                continue
            rxn = ".".join(reactants) + f">>{target}"
            key = canonical_reaction(rxn) or rxn
            if key in seen:
                self.state.duplicate_candidates += 1
                continue
            gate_payload = _proposal_gate_payload(target, reactants, row)
            self.state.proposal_gate_scored += 1
            if gate_payload.get("decision") == "reject":
                self.state.proposal_gate_rejected += 1
                if self.config.require_proposal_gate:
                    continue
            else:
                self.state.proposal_gate_kept += 1
            out.append(_row_to_vendor_row(row, target=target, reactants=reactants, rank=len(out) + 1, gate_payload=gate_payload, config=self.config))
            seen.add(key)
            self.state.added_candidates += 1
            if len(out) >= self.config.max_added:
                break
        return out

    def _fusion_obj(self) -> Any:
        if self._fusion is None:
            self._fusion = _make_graphfp_dualtower_fusion(self.config)
        return self._fusion


def native_chemical_plugin_config_from_flags(search_flags: dict[str, Any] | None) -> NativeChemicalPluginConfig:
    flags = dict(search_flags or {})
    raw = (
        flags.get("native_chemical_plugin")
        if "native_chemical_plugin" in flags
        else flags.get("autoplanner_native_chemical_plugin")
    )
    return NativeChemicalPluginConfig.from_raw(raw)


def reset_native_chemical_plugin_state(planner: Any, target_smiles: str) -> None:
    state = getattr(planner, "_autoplanner_native_chemical_plugin_state", None)
    if isinstance(state, NativeChemicalPluginState):
        state.reset_for_target(str(target_smiles or ""))


def native_chemical_plugin_stats(planner: Any) -> dict[str, Any] | None:
    state = getattr(planner, "_autoplanner_native_chemical_plugin_state", None)
    if isinstance(state, NativeChemicalPluginState):
        return state.to_dict()
    return None


def _make_graphfp_dualtower_fusion(config: NativeChemicalPluginConfig) -> GraphFPDualTowerFusion:
    return GraphFPDualTowerFusion(
        GraphFPDualTowerFusionConfig(
            model_path=config.model_path,
            template_vector_cache=config.template_vector_cache,
            templates_index=config.templates_index,
            graphfp_topk=config.graphfp_top_k,
            dual_topk=config.dual_top_k,
            mode=config.fusion_mode,
            device=config.device,
            template_batch_size=config.template_batch_size,
        )
    )


def _base_rows(product: str, base: dict[str, Any], *, default_model_full_name: str = "") -> list[dict[str, Any]]:
    reactants = list((base or {}).get("reactants") or [])
    scores = list((base or {}).get("scores") or [])
    templates = list((base or {}).get("template") or [])
    models = list((base or {}).get("model_full_name") or [])
    costs = list((base or {}).get("costs") or [])
    weights = list((base or {}).get("weight") or [])
    out: list[dict[str, Any]] = []
    for idx, reactant_text in enumerate(reactants):
        parts = [part for part in str(reactant_text or "").split(".") if part]
        if not parts:
            continue
        model_full_name = str(_at(models, idx, "") or default_model_full_name or "")
        rxn = ".".join(parts) + f">>{product}"
        out.append(
            {
                "reactant_smiles": parts,
                "rxn_smiles": rxn,
                "reaction_smiles": rxn,
                "source": _source_from_model(model_full_name),
                "score": _float(_at(scores, idx, 0.0), 0.0),
                "rank": len(out) + 1,
                "template": _at(templates, idx, ""),
                "model_full_name": model_full_name,
                "cost": _float(_at(costs, idx, None), None),
                "weight": _float(_at(weights, idx, None), None),
            }
        )
    return out


def _row_to_vendor_row(
    row: dict[str, Any],
    *,
    target: str,
    reactants: list[str],
    rank: int,
    gate_payload: dict[str, Any],
    config: NativeChemicalPluginConfig,
) -> dict[str, Any]:
    score = _row_probability(row, scale=config.score_scale)
    template = {
        "model_full_name": PLUGIN_MODEL_FULL_NAME,
        "source": "autoplanner_dualtower",
        "rank": rank,
        "autoplanner_native_chemical_plugin": True,
        "native_chemical_plugin_type": "graphfp_dualtower_tail",
        "fusion_mode": config.fusion_mode,
        "fusion_score": row.get("fusion_score"),
        "dualtower_rank": row.get("dualtower_rank"),
        "dualtower_score": row.get("dualtower_score", row.get("score")),
        "template_id": row.get("template_id") or row.get("dualtower_template_id"),
        "template_rank": row.get("template_rank") or row.get("dualtower_template_rank"),
        "template": row.get("template"),
        "proposal_gate": gate_payload,
    }
    return {
        "reactants": ".".join(reactants),
        "scores": score,
        "costs": -math.log(max(score, 1e-6)),
        "template": template,
        "model_full_name": PLUGIN_MODEL_FULL_NAME,
        "weight": 1.0,
        "reaction_domains": "organic",
        "autoplanner_chemical_plugin_scores": row.get("dualtower_score", row.get("score")),
    }


def _proposal_gate_payload(target: str, reactants: list[str], row: dict[str, Any]) -> dict[str, Any]:
    rxn = ".".join(reactants) + f">>{target}"
    return evaluate_step_candidate(
        product_smiles=target,
        reactant_smiles=reactants,
        rxn_smiles=rxn,
        source_model=str(row.get("source") or PLUGIN_MODEL_FULL_NAME),
    )


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


def _row_reactants(row: dict[str, Any]) -> list[str]:
    rxn = str(row.get("reaction_smiles") or row.get("rxn_smiles") or "")
    if ">>" in rxn:
        lhs = rxn.split(">>", 1)[0]
        return list(canonical_side(lhs))
    reactants = [str(item) for item in row.get("reactant_smiles") or [] if str(item or "")]
    return list(canonical_side(".".join(reactants)))


def _is_graphfp_row(row: dict[str, Any]) -> bool:
    model = str(row.get("model_full_name") or "")
    source = str(row.get("source") or "")
    return source == "chem_enzy_graphfp" or model.startswith("graphfp_models.")


def _source_from_model(model_full_name: str) -> str:
    if model_full_name.startswith("graphfp_models."):
        return "chem_enzy_graphfp"
    if model_full_name.startswith("onmt_models."):
        return "chem_enzy_onmt"
    if model_full_name.startswith("template_relevance."):
        return "template_relevance"
    if model_full_name == PLUGIN_MODEL_FULL_NAME:
        return "autoplanner_dualtower"
    return "chem_enzy_onestep"


def _default_model_full_name(one_step_models: dict[str, Any], *, fallback: str = "") -> str:
    if fallback:
        return str(fallback)
    names = [str(name) for name in (one_step_models or {}).keys() if str(name or "")]
    graphfp = [name for name in names if name.startswith("graphfp_models.")]
    if len(graphfp) == 1:
        return graphfp[0]
    return names[0] if len(names) == 1 else ""


def _row_probability(row: dict[str, Any], *, scale: float) -> float:
    raw = _float(row.get("dualtower_score", row.get("score")), 0.0) or 0.0
    if 0.0 < raw <= 1.0:
        prob = raw
    else:
        prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw))))
    prob *= max(0.0, float(scale or 1.0))
    return max(1e-6, min(0.999999, prob))


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


def _int(value: Any, default: int, *, lo: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if lo is not None:
        out = max(lo, out)
    return out


def _float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
