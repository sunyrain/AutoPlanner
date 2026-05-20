"""ChemEnzy one-step proposal adapter for AutoPlanner route-tree search."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_adapter import (
    CHEMENZY_ONMT_MODEL_PATH_ENV,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_VENDOR_ROOT,
    ChemEnzyBackendAdapter,
    _patch_dgl_graphbolt_optional_import,
    _patch_numpy_legacy_aliases,
    _patch_optional_easifa_import,
    _patch_optional_graphviz_import,
    _patch_torchdata_legacy_aliases,
    _vendor_pythonpath,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


DEFAULT_CHEMENZY_ONESTEP_SOURCE = "chem_enzy_onestep"


@dataclass
class ChemEnzyOneStepProposalProvider:
    """Expose ChemEnzy graphfp/onmt one-step models as CandidateAction rows."""

    vendor_root: Path | str = DEFAULT_VENDOR_ROOT
    models: tuple[str, ...] = tuple(DEFAULT_ONE_STEP_MODELS)
    expansion_topk: int = 50
    gpu: int = -1
    onmt_model_path: Path | str | None = None
    one_step: Any | None = None
    load_error: str = ""
    _loaded: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_env(cls) -> "ChemEnzyOneStepProposalProvider":
        return cls(
            vendor_root=Path(os.environ.get("AUTOPLANNER_CHEMENZY_ONESTEP_VENDOR_ROOT") or DEFAULT_VENDOR_ROOT),
            models=tuple(_env_list("AUTOPLANNER_CHEMENZY_ONESTEP_MODELS") or DEFAULT_ONE_STEP_MODELS),
            expansion_topk=_env_int("AUTOPLANNER_CHEMENZY_ONESTEP_TOPK", 50),
            gpu=_env_int("AUTOPLANNER_CHEMENZY_ONESTEP_GPU", -1),
            onmt_model_path=os.environ.get(CHEMENZY_ONMT_MODEL_PATH_ENV) or None,
        )

    @property
    def available(self) -> bool:
        if self.one_step is not None:
            return True
        vendor_root = Path(self.vendor_root)
        return vendor_root.exists() and (vendor_root / "retro_planner" / "config" / "config.yaml").exists()

    def predict(self, product: str, top_k: int = 10, **_: Any) -> list[dict[str, Any]]:
        if not product:
            return []
        try:
            one_step = self._ensure_one_step()
            raw = one_step.run(product, topk=max(1, int(top_k or self.expansion_topk or 1)))
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            return []
        return _one_step_rows(product, raw, limit=top_k)

    def _ensure_one_step(self) -> Any:
        if self.one_step is not None:
            return self.one_step
        if self._loaded:
            raise RuntimeError(self.load_error or "ChemEnzy one-step provider failed to load")
        self._loaded = True
        self.one_step = self._load_one_step()
        return self.one_step

    def _load_one_step(self) -> Any:
        adapter = ChemEnzyBackendAdapter(
            vendor_root=Path(self.vendor_root),
            gpu=int(self.gpu),
            onmt_model_path=self.onmt_model_path,
        )
        failures = adapter.preflight()
        if failures:
            message = "; ".join(f"{failure.category}:{failure.message}" for failure in failures)
            raise RuntimeError(message)
        search_config = RouteSearchConfig(
            target_smiles="",
            max_iterations=1,
            max_depth=1,
            expansion_topk=max(1, int(self.expansion_topk or 50)),
            one_step_models=list(self.models or DEFAULT_ONE_STEP_MODELS),
            search_flags={"gpu": int(self.gpu)},
        )
        config = adapter._vendor_config(search_config)
        with _vendor_pythonpath(Path(self.vendor_root)):
            _patch_numpy_legacy_aliases()
            _patch_torchdata_legacy_aliases()
            _patch_dgl_graphbolt_optional_import()
            _patch_optional_easifa_import(False)
            _patch_optional_graphviz_import(False)
            import torch
            from retro_planner.common.prepare_utils import (
                handle_one_step_config,
                handle_one_step_path,
                prepare_multi_single_step,
                prepare_single_step,
            )

            selected_configs, _subnames, selected_types = handle_one_step_config(
                list(self.models or DEFAULT_ONE_STEP_MODELS),
                config["one_step_model_configs"],
            )
            selected_configs = handle_one_step_path(selected_types, selected_configs)
            device = torch.device("cuda:%d" % int(self.gpu) if int(self.gpu) >= 0 else "cpu")
            filter_path = str(Path(self.vendor_root) / "retro_planner" / str(config.get("filter_path") or ""))
            if len(selected_configs) == 1:
                return prepare_single_step(
                    one_step_model_type=selected_types[0],
                    model_configs=selected_configs[0],
                    device=device,
                    use_filter=bool(config.get("use_filter")),
                    filter_path=filter_path,
                    expansion_topk=max(1, int(self.expansion_topk or 50)),
                    keep_score=bool(config.get("keep_score", True)),
                )
            return prepare_multi_single_step(
                one_step_model_types=selected_types,
                model_configs=selected_configs,
                device=device,
                use_filter=bool(config.get("use_filter")),
                filter_path=filter_path,
                expansion_topk=max(1, int(self.expansion_topk or 50)),
                keep_score=bool(config.get("keep_score", True)),
                weights=[float(item.get("weight", 1.0)) for item in selected_configs],
            )


def _one_step_rows(product: str, raw: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    reactants = list((raw or {}).get("reactants") or [])
    scores = list((raw or {}).get("scores") or [])
    templates = list((raw or {}).get("template") or [])
    costs = list((raw or {}).get("costs") or [])
    models = list((raw or {}).get("model_full_name") or [])
    weights = list((raw or {}).get("weight") or [])
    out: list[dict[str, Any]] = []
    for idx, reactant_text in enumerate(reactants):
        parts = _split_reactants(reactant_text)
        if not parts:
            continue
        model_full_name = _at(models, idx, DEFAULT_CHEMENZY_ONESTEP_SOURCE)
        source = _source_from_model(model_full_name)
        main = _largest_smiles(parts)
        aux = [smi for smi in parts if (canonical_smiles(smi) or smi) != (canonical_smiles(main) or main)]
        out.append(
            {
                "main_reactant": main,
                "aux_reactants": aux,
                "rxn_smiles": ".".join(parts) + f">>{product}",
                "reaction_smiles": ".".join(parts) + f">>{product}",
                "source": source,
                "score": _float(_at(scores, idx, 0.0), 0.0),
                "rank": len(out) + 1,
                "type": "template",
                "proposal_type": "chem_enzy_one_step",
                "template": _at(templates, idx, ""),
                "model_full_name": model_full_name,
                "cost": _float(_at(costs, idx, None), None),
                "weight": _float(_at(weights, idx, None), None),
                "teacher_one_step": True,
                "teacher_source": DEFAULT_CHEMENZY_ONESTEP_SOURCE,
            }
        )
        if len(out) >= max(0, int(limit or 0)):
            break
    return out


def _split_reactants(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in text.split(".") if part]


def _source_from_model(model_full_name: Any) -> str:
    text = str(model_full_name or "").strip()
    if text.startswith("graphfp_models."):
        return "chem_enzy_graphfp"
    if text.startswith("onmt_models."):
        return "chem_enzy_onmt"
    return DEFAULT_CHEMENZY_ONESTEP_SOURCE


def _largest_smiles(smiles: list[str]) -> str:
    if not smiles:
        return ""
    return max(smiles, key=lambda smi: (len(canonical_smiles(smi) or smi), canonical_smiles(smi) or smi))


def _at(values: list[Any], idx: int, default: Any) -> Any:
    return values[idx] if idx < len(values) else default


def _float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default
