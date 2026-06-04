"""ChemEnzy one-step plugin for validated literature executable templates."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cascade_planner.agent.executable_template_validation import (
    candidate_to_one_step_row,
    candidate_to_provider_row,
    instantiate_literature_template,
)
from cascade_planner.agent.literature_templates import (
    LITERATURE_TEMPLATE_PLUGIN_MODEL,
    LITERATURE_TEMPLATE_PLUGIN_SOURCE,
    LiteratureTemplateCard,
    default_literature_template_cards,
    direct_consumption_allowed,
    template_card_from_dict,
)
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles


PLUGIN_MODEL_FULL_NAME = LITERATURE_TEMPLATE_PLUGIN_MODEL


@dataclass(frozen=True)
class LiteratureOneStepPluginConfig:
    enabled: bool = False
    top_k: int = 6
    max_added: int = 6
    score: float = 0.62
    require_validation: bool = True
    respect_source_policy: bool = False
    trigger_reasons: tuple[str, ...] = ()
    template_cards: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_raw(cls, raw: Any) -> "LiteratureOneStepPluginConfig":
        if raw is True:
            return cls(enabled=True)
        if not raw:
            return cls(enabled=False)
        if not isinstance(raw, dict):
            return cls(enabled=bool(raw))
        cards = tuple(dict(item) for item in raw.get("template_cards") or [] if isinstance(item, dict))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            top_k=_int(raw.get("top_k"), cls.top_k, lo=0),
            max_added=_int(raw.get("max_added"), cls.max_added, lo=0),
            score=max(0.0, min(1.0, _float(raw.get("score"), cls.score))),
            require_validation=bool(raw.get("require_validation", cls.require_validation)),
            respect_source_policy=bool(raw.get("respect_source_policy", cls.respect_source_policy)),
            trigger_reasons=tuple(str(item) for item in raw.get("trigger_reasons") or []),
            template_cards=cards,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "max_added": self.max_added,
            "score": self.score,
            "require_validation": self.require_validation,
            "respect_source_policy": self.respect_source_policy,
            "trigger_reasons": list(self.trigger_reasons),
            "template_card_count": len(self.template_cards),
        }


@dataclass
class LiteratureOneStepPluginState:
    config: LiteratureOneStepPluginConfig
    target_smiles: str = ""
    calls: int = 0
    candidate_templates: int = 0
    instantiated_candidates: int = 0
    validation_passed: int = 0
    validation_rejected: int = 0
    added_candidates: int = 0
    duplicate_candidates: int = 0
    source_policy_skips: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)

    def reset_for_target(self, target_smiles: str) -> None:
        config = self.config
        self.__dict__.clear()
        self.__dict__.update(LiteratureOneStepPluginState(config=config, target_smiles=target_smiles).__dict__)

    def record_error(self, exc: Exception) -> None:
        self.error_count += 1
        if len(self.errors) < 8:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "literature_one_step_plugin.stats.v1",
            "enabled": bool(self.config.enabled),
            "target_smiles": self.target_smiles,
            "config": self.config.to_dict(),
            "calls": self.calls,
            "candidate_templates": self.candidate_templates,
            "instantiated_candidates": self.instantiated_candidates,
            "validation_passed": self.validation_passed,
            "validation_rejected": self.validation_rejected,
            "added_candidates": self.added_candidates,
            "duplicate_candidates": self.duplicate_candidates,
            "source_policy_skips": self.source_policy_skips,
            "error_count": self.error_count,
            "errors": list(self.errors),
        }


class LiteratureOneStepPlugin:
    """Standalone one-step source backed by validated template cards."""

    def __init__(
        self,
        *,
        config: LiteratureOneStepPluginConfig | None = None,
        state: LiteratureOneStepPluginState | None = None,
        template_cards: list[LiteratureTemplateCard | dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or LiteratureOneStepPluginConfig(enabled=True)
        self.state = state or LiteratureOneStepPluginState(config=self.config)
        self.template_cards = [
            item if isinstance(item, LiteratureTemplateCard) else template_card_from_dict(item)
            for item in (template_cards if template_cards is not None else self._configured_cards())
        ]
        self.one_step_models = {PLUGIN_MODEL_FULL_NAME: self}

    def run(self, product_smiles: str, topk: int | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if not self.config.enabled:
            return _empty_result()
        self.state.calls += 1
        rows = self.one_step_rows(product_smiles, top_k=topk or self.config.top_k)
        return _rows_to_vendor_result(rows)

    def predict(self, product_smiles: str, top_k: int = 10, **_: Any) -> list[dict[str, Any]]:
        rows = self.one_step_rows(product_smiles, top_k=top_k)
        return [
            candidate_to_provider_row(row["candidate"], rank=idx + 1)
            for idx, row in enumerate(rows)
            if row.get("candidate") is not None
        ]

    def one_step_rows(self, product_smiles: str, *, top_k: int) -> list[dict[str, Any]]:
        if not product_smiles or self.config.max_added <= 0 or int(top_k or 0) <= 0:
            return []
        cards = [card for card in self.template_cards if direct_consumption_allowed(card)]
        self.state.candidate_templates += len(cards)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for card in cards:
            try:
                candidate = instantiate_literature_template(product_smiles, card)
            except Exception as exc:
                self.state.record_error(exc)
                continue
            self.state.instantiated_candidates += 1
            validation = dict(candidate.validation_report or {})
            if validation.get("allowed_for_one_step_source"):
                self.state.validation_passed += 1
            else:
                self.state.validation_rejected += 1
                if self.config.require_validation:
                    continue
            key = canonical_reaction(candidate.rxn_smiles) or candidate.rxn_smiles
            if key in seen:
                self.state.duplicate_candidates += 1
                continue
            seen.add(key)
            vendor_row = candidate_to_one_step_row(candidate, score=self.config.score)
            vendor_row["candidate"] = candidate
            out.append(vendor_row)
            self.state.added_candidates += 1
            if len(out) >= min(self.config.max_added, int(top_k or self.config.top_k or 1)):
                break
        return out

    def _configured_cards(self) -> list[LiteratureTemplateCard]:
        if self.config.template_cards:
            return [template_card_from_dict(item) for item in self.config.template_cards]
        return default_literature_template_cards()


class LiteratureTemplateOneStepWrapper:
    """Append literature-template rows to a native ChemEnzy one-step source."""

    def __init__(self, one_step: Any, *, config: LiteratureOneStepPluginConfig, state: LiteratureOneStepPluginState) -> None:
        self.one_step = one_step
        self.config = config
        self.state = state
        self.plugin = LiteratureOneStepPlugin(config=config, state=state)
        self.one_step_models = dict(getattr(one_step, "one_step_models", {}) or {})
        self.one_step_models.setdefault(PLUGIN_MODEL_FULL_NAME, self.plugin)

    def run(self, target: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        base = _run_base(self.one_step, target, *args, **kwargs)
        if not self.config.enabled:
            return base
        if self.config.respect_source_policy:
            selected = kwargs.get("select_models")
            if selected is not None and PLUGIN_MODEL_FULL_NAME not in set(selected):
                self.state.source_policy_skips += 1
                return base
        try:
            rows = self.plugin.one_step_rows(str(target or ""), top_k=self.config.max_added)
        except Exception as exc:  # pragma: no cover - plugin must not break native search
            self.state.record_error(exc)
            return base
        if not rows:
            return base
        base_keys = _base_reaction_keys(base, str(target or ""))
        append_rows = []
        for row in rows:
            candidate = row.pop("candidate", None)
            del candidate
            key = canonical_reaction(f"{row.get('reactants', '')}>>{target}") or f"{row.get('reactants', '')}>>{target}"
            if key in base_keys:
                self.state.duplicate_candidates += 1
                continue
            append_rows.append(row)
            base_keys.add(key)
        return _append_rows(base, append_rows)


def literature_plugin_config_from_flags(search_flags: dict[str, Any] | None) -> LiteratureOneStepPluginConfig:
    flags = dict(search_flags or {})
    raw = (
        flags.get("literature_template_plugin")
        if "literature_template_plugin" in flags
        else flags.get("autoplanner_literature_template_plugin")
    )
    if raw is None:
        source_policy = dict(flags.get("cascade_source_policy") or {})
        raw = source_policy.get("literature_template_plugin")
    return LiteratureOneStepPluginConfig.from_raw(raw)


def reset_literature_plugin_state(planner: Any, target_smiles: str) -> None:
    state = getattr(planner, "_autoplanner_literature_plugin_state", None)
    if isinstance(state, LiteratureOneStepPluginState):
        state.reset_for_target(str(target_smiles or ""))


def literature_plugin_stats(planner: Any) -> dict[str, Any] | None:
    state = getattr(planner, "_autoplanner_literature_plugin_state", None)
    if isinstance(state, LiteratureOneStepPluginState):
        return state.to_dict()
    return None


def _rows_to_vendor_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = _empty_result()
    for row in rows:
        for key in (
            "reactants",
            "scores",
            "costs",
            "template",
            "templates",
            "model_full_name",
            "weight",
            "reaction_domains",
            "literature_template_trace",
            "source_policy_decision",
        ):
            value = row.get(key)
            if key == "costs" and value is None:
                value = _score_to_cost(row.get("scores"))
            out.setdefault(key, []).append(value)
    return out


def _empty_result() -> dict[str, list[Any]]:
    return {
        "reactants": [],
        "scores": [],
        "costs": [],
        "template": [],
        "templates": [],
        "model_full_name": [],
        "weight": [],
        "reaction_domains": [],
        "literature_template_trace": [],
        "source_policy_decision": [],
    }


def _append_rows(base: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {key: _as_list(value) for key, value in dict(base or {}).items()}
    for key in ("reactants", "scores", "costs", "template", "templates", "model_full_name", "weight"):
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
            if key == "candidate":
                continue
            if key == "costs" and value is None:
                value = _score_to_cost(row.get("scores"))
            out.setdefault(key, [])
            out[key].append(value)
    return out


def _complete_costs_from_scores(out: dict[str, list[Any]]) -> None:
    current_len = len(out.get("scores") or out.get("reactants") or [])
    costs = out.setdefault("costs", [])
    while len(costs) < current_len:
        costs.append(_score_to_cost(_at(out.get("scores") or [], len(costs), 0.0)))


def _score_to_cost(score: Any) -> float:
    value = _float(score, 0.0)
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
        keys.add(canonical_reaction(rxn) or rxn)
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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
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


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        return float(default)
    return out
