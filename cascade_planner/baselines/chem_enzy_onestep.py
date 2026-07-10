"""ChemEnzy one-step proposal adapter for AutoPlanner route-tree search."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_adapter import (
    CHEMENZY_ONMT_MODEL_PATH_ENV,
    CHEMENZY_ONMT_TOKENIZER_ENV,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_VENDOR_ROOT,
    ChemEnzyBackendAdapter,
    _patch_dgl_graphbolt_optional_import,
    _patch_numpy_legacy_aliases,
    _patch_optional_easifa_import,
    _patch_optional_graphviz_import,
    _patch_torchdata_legacy_aliases,
    _patch_onmt_tokenizer,
    _vendor_pythonpath,
)
from cascade_planner.baselines.proposal_gate import evaluate_step_candidate
from cascade_planner.baselines.route_contract import RouteSearchConfig
from cascade_planner.baselines.template_relevance_runtime import missing_template_relevance_models
from cascade_planner.agent.literature_templates import (
    LITERATURE_TEMPLATE_PLUGIN_MODEL,
    LITERATURE_TEMPLATE_PLUGIN_SOURCE,
)
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


DEFAULT_CHEMENZY_ONESTEP_SOURCE = "chem_enzy_onestep"
GRAPHFP_RERANKER_ENABLE_ENV = "AUTOPLANNER_ENABLE_GRAPHFP_RERANKER"
GRAPHFP_RERANKER_INTERNAL_TOPK_ENV = "AUTOPLANNER_GRAPHFP_RERANKER_INTERNAL_TOPK"
GRAPHFP_FUSION_PROTECTED_TOPK_ENV = "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_TOPK"
GRAPHFP_FUSION_PROTECTED_FRONT_ENV = "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_FRONT"
GRAPHFP_FUSION_PROTECTED_STRIDE_ENV = "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_STRIDE"


@dataclass
class ChemEnzyOneStepProposalProvider:
    """Expose ChemEnzy graphfp/onmt one-step models as CandidateAction rows."""

    vendor_root: Path | str = DEFAULT_VENDOR_ROOT
    models: tuple[str, ...] = tuple(DEFAULT_ONE_STEP_MODELS)
    expansion_topk: int = 50
    gpu: int = -1
    onmt_model_path: Path | str | None = None
    onmt_tokenizer: str | None = None
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
            onmt_tokenizer=os.environ.get(CHEMENZY_ONMT_TOKENIZER_ENV) or None,
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
        rescue_rows = _rescue_one_step_rows(product)
        reranker = _graphfp_reranker_from_env()
        fusion = _graphfp_dualtower_fusion_from_env()
        request_top_k = max(1, int(top_k or self.expansion_topk or 1))
        internal_top_k = request_top_k
        if reranker is not None:
            internal_top_k = max(request_top_k, _env_int(GRAPHFP_RERANKER_INTERNAL_TOPK_ENV, 50))
        if fusion is not None:
            internal_top_k = max(request_top_k, fusion.graphfp_topk)
        try:
            one_step = self._ensure_one_step()
            raw = one_step.run(product, topk=internal_top_k)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            return rescue_rows[:request_top_k]
        raw_candidate_count = len((raw or {}).get("reactants") or [])
        base_rows = _one_step_rows(product, raw, limit=max(internal_top_k, raw_candidate_count))
        if fusion is not None:
            try:
                graphfp_rows, protected_rows = _split_graphfp_fusion_rows(base_rows)
                if graphfp_rows:
                    fusion_base_rows = graphfp_rows
                else:
                    fusion_base_rows = base_rows
                fused_rows = fusion.fuse(
                    product,
                    fusion_base_rows,
                    fusion.dual_rows(product) if graphfp_rows else [],
                    output_k=internal_top_k,
                )
                if graphfp_rows and protected_rows:
                    fused_rows = _interleave_protected_rows(
                        fused_rows=fused_rows,
                        protected_rows=protected_rows,
                        request_top_k=request_top_k,
                        limit=internal_top_k,
                    )
                rows = _merge_proposal_rows(rescue_rows, fused_rows, limit=internal_top_k)
            except Exception as exc:
                self.load_error = f"graphfp_dualtower_fusion:{type(exc).__name__}:{exc}"
                rows = _merge_proposal_rows(rescue_rows, base_rows, limit=internal_top_k)
        else:
            rows = _merge_proposal_rows(rescue_rows, base_rows, limit=internal_top_k)
        if reranker is not None and fusion is None:
            rows = reranker.rerank(product, rows, output_k=request_top_k)
        return rows[:request_top_k]

    def _ensure_one_step(self) -> Any:
        if self.one_step is not None:
            return self.one_step
        if self._loaded:
            raise RuntimeError(self.load_error or "ChemEnzy one-step provider failed to load")
        self._loaded = True
        self.one_step = self._load_one_step()
        return self.one_step

    def _load_one_step(self) -> Any:
        missing_template_models = missing_template_relevance_models(
            list(self.models or ()), vendor_root=Path(self.vendor_root)
        )
        if missing_template_models:
            raise RuntimeError(
                "missing local template_relevance .mar archive(s): "
                + ", ".join(missing_template_models)
            )
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
            search_flags={
                "gpu": int(self.gpu),
                **({"chem_enzy_onmt_tokenizer": self.onmt_tokenizer} if self.onmt_tokenizer else {}),
            },
        )
        config = adapter._vendor_config(search_config)
        with _vendor_pythonpath(Path(self.vendor_root)):
            _patch_numpy_legacy_aliases()
            _patch_torchdata_legacy_aliases()
            _patch_dgl_graphbolt_optional_import()
            _patch_optional_easifa_import(False)
            _patch_optional_graphviz_import(False)
            import torch
            import retro_planner.api as api
            from retro_planner.common.prepare_utils import (
                handle_one_step_config,
                handle_one_step_path,
                prepare_multi_single_step,
                prepare_single_step,
            )
            onmt_tokenizer = self.onmt_tokenizer or os.environ.get(CHEMENZY_ONMT_TOKENIZER_ENV) or "char"
            _patch_onmt_tokenizer(api, onmt_tokenizer)

            selected_configs, _subnames, selected_types = handle_one_step_config(
                list(self.models or DEFAULT_ONE_STEP_MODELS),
                config["one_step_model_configs"],
            )
            selected_configs = handle_one_step_path(selected_types, selected_configs)
            device = torch.device("cuda:%d" % int(self.gpu) if int(self.gpu) >= 0 else "cpu")
            filter_path = str(Path(self.vendor_root) / "retro_planner" / str(config.get("filter_path") or ""))
            if len(selected_configs) == 1:
                single_step = prepare_single_step(
                    one_step_model_type=selected_types[0],
                    model_configs=selected_configs[0],
                    device=device,
                    use_filter=bool(config.get("use_filter")),
                    filter_path=filter_path,
                    expansion_topk=max(1, int(self.expansion_topk or 50)),
                    keep_score=bool(config.get("keep_score", True)),
                )
                return _SingleModelRunWrapper(single_step, str(selected_configs[0].get("model_full_name") or self.models[0]))
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


class _SingleModelRunWrapper:
    def __init__(self, one_step: Any, model_full_name: str) -> None:
        self.one_step = one_step
        self.model_full_name = model_full_name

    def run(self, product: str, topk: int | None = None) -> dict[str, Any]:
        results = dict(self.one_step.run(product, topk=topk) or {})
        count = len(results.get("reactants") or [])
        results["model_full_name"] = [self.model_full_name for _ in range(count)]
        results.setdefault("weight", [1.0 for _ in range(count)])
        return results


def _one_step_rows(product: str, raw: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    reactants = list((raw or {}).get("reactants") or [])
    scores = list((raw or {}).get("scores") or [])
    templates = list((raw or {}).get("template") or [])
    costs = list((raw or {}).get("costs") or [])
    models = list((raw or {}).get("model_full_name") or [])
    weights = list((raw or {}).get("weight") or [])
    out: list[dict[str, Any]] = []
    raw_candidate_count = len(reactants)
    for idx, reactant_text in enumerate(reactants):
        parts = _split_reactants(reactant_text)
        if not parts:
            continue
        model_full_name = _at(models, idx, DEFAULT_CHEMENZY_ONESTEP_SOURCE)
        source = _source_from_model(model_full_name)
        template_payload = _at(templates, idx, "")
        main = _largest_smiles(parts)
        aux = [smi for smi in parts if (canonical_smiles(smi) or smi) != (canonical_smiles(main) or main)]
        rxn_smiles = ".".join(parts) + f">>{product}"
        proposal_gate = evaluate_step_candidate(
            product_smiles=product,
            reactant_smiles=parts,
            rxn_smiles=rxn_smiles,
            source_model=source,
        )
        out.append(
            {
                "main_reactant": main,
                "aux_reactants": aux,
                "reactant_smiles": parts,
                "rxn_smiles": rxn_smiles,
                "reaction_smiles": rxn_smiles,
                "source": source,
                "score": _float(_at(scores, idx, 0.0), 0.0),
                "rank": len(out) + 1,
                "candidate_count": raw_candidate_count,
                "type": _type_from_model(model_full_name),
                "proposal_type": _proposal_type_from_model(model_full_name),
                "template": template_payload,
                "model_full_name": model_full_name,
                "cost": _float(_at(costs, idx, None), None),
                "weight": _float(_at(weights, idx, None), None),
                "teacher_one_step": True,
                "teacher_source": source if source == LITERATURE_TEMPLATE_PLUGIN_SOURCE else DEFAULT_CHEMENZY_ONESTEP_SOURCE,
                "proposal_gate": proposal_gate,
                **_literature_template_metadata(template_payload),
            }
        )
        if len(out) >= max(0, int(limit or 0)):
            break
    return out


def _rescue_one_step_rows(product: str) -> list[dict[str, Any]]:
    enabled = str(os.environ.get("AUTOPLANNER_ENABLE_SEMISYNTHESIS_RESCUE_PROPOSALS") or "").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return []
    from cascade_planner.baselines.semisynthesis_rescue import semisynthesis_rescue_routes

    rows: list[dict[str, Any]] = []
    for route in semisynthesis_rescue_routes(product):
        if not route.steps:
            continue
        step = route.steps[0]
        parts = [str(item) for item in step.reactant_smiles or [] if str(item or "")]
        if not parts:
            continue
        main = _largest_smiles(parts)
        aux = [smi for smi in parts if (canonical_smiles(smi) or smi) != (canonical_smiles(main) or main)]
        rxn_smiles = step.rxn_smiles or ".".join(parts) + f">>{product}"
        proposal_gate = evaluate_step_candidate(
            product_smiles=product,
            reactant_smiles=parts,
            rxn_smiles=rxn_smiles,
            condition_predictions=list(step.condition_predictions or []),
            source_model=step.source_model,
        )
        rows.append(
            {
                "main_reactant": main,
                "aux_reactants": aux,
                "reactant_smiles": parts,
                "rxn_smiles": rxn_smiles,
                "reaction_smiles": rxn_smiles,
                "source": "autoplanner_semisynthesis_rescue",
                "score": float(step.score or route.score or 0.0),
                "rank": len(rows) + 1,
                "candidate_count": len(rows) + 1,
                "type": "semisynthesis_rescue",
                "proposal_type": "source_supported_derivatization",
                "template": "",
                "model_full_name": str(step.source_model or "semisynthesis_rescue"),
                "cost": None,
                "weight": 1.0,
                "teacher_one_step": True,
                "teacher_source": "autoplanner_semisynthesis_rescue",
                "proposal_gate": proposal_gate,
            }
        )
    return rows


def _merge_proposal_rows(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*primary, *secondary]:
        signature = str(row.get("reaction_smiles") or row.get("rxn_smiles") or "")
        if not signature:
            signature = "|".join(str(item) for item in row.get("reactant_smiles") or []) + f">>{row.get('product') or ''}"
        if signature in seen:
            continue
        seen.add(signature)
        item = dict(row)
        item["rank"] = len(out) + 1
        out.append(item)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def _split_graphfp_fusion_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graphfp_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source") or "")
        model = str(row.get("model_full_name") or "")
        if source == "chem_enzy_graphfp" or model.startswith("graphfp_models."):
            graphfp_rows.append(row)
        else:
            protected_rows.append(row)
    return graphfp_rows, protected_rows


def _interleave_protected_rows(
    *,
    fused_rows: list[dict[str, Any]],
    protected_rows: list[dict[str, Any]],
    request_top_k: int,
    limit: int,
) -> list[dict[str, Any]]:
    if not protected_rows:
        return fused_rows[: max(1, int(limit or 1))]
    default_budget = max(1, min(10, max(1, int(request_top_k or 1)) // 5 or 1))
    protected_budget = max(0, _env_int(GRAPHFP_FUSION_PROTECTED_TOPK_ENV, default_budget))
    if protected_budget <= 0:
        return fused_rows[: max(1, int(limit or 1))]
    protected = protected_rows[:protected_budget]
    front = min(len(protected), max(0, _env_int(GRAPHFP_FUSION_PROTECTED_FRONT_ENV, 2)))
    stride = max(1, _env_int(GRAPHFP_FUSION_PROTECTED_STRIDE_ENV, 4))

    out: list[dict[str, Any]] = []
    protected_index = 0
    fused_index = 0
    while protected_index < front and len(out) < max(1, int(limit or 1)):
        out.append(protected[protected_index])
        protected_index += 1
    while len(out) < max(1, int(limit or 1)) and (
        fused_index < len(fused_rows) or protected_index < len(protected)
    ):
        for _ in range(stride):
            if fused_index >= len(fused_rows) or len(out) >= max(1, int(limit or 1)):
                break
            out.append(fused_rows[fused_index])
            fused_index += 1
        if protected_index < len(protected) and len(out) < max(1, int(limit or 1)):
            out.append(protected[protected_index])
            protected_index += 1
    while fused_index < len(fused_rows) and len(out) < max(1, int(limit or 1)):
        out.append(fused_rows[fused_index])
        fused_index += 1
    return out


def _split_reactants(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in text.split(".") if part]


def _source_from_model(model_full_name: Any) -> str:
    text = str(model_full_name or "").strip()
    if text == LITERATURE_TEMPLATE_PLUGIN_MODEL:
        return LITERATURE_TEMPLATE_PLUGIN_SOURCE
    if text.startswith("graphfp_models."):
        return "chem_enzy_graphfp"
    if text.startswith("onmt_models."):
        return "chem_enzy_onmt"
    if text.startswith("template_relevance."):
        return "template_relevance"
    return DEFAULT_CHEMENZY_ONESTEP_SOURCE


def _type_from_model(model_full_name: Any) -> str:
    text = str(model_full_name or "").strip()
    if text == LITERATURE_TEMPLATE_PLUGIN_MODEL:
        return "literature_executable_template"
    if text.startswith("template_relevance."):
        return "template_relevance"
    return "template"


def _proposal_type_from_model(model_full_name: Any) -> str:
    text = str(model_full_name or "").strip()
    if text == LITERATURE_TEMPLATE_PLUGIN_MODEL:
        return LITERATURE_TEMPLATE_PLUGIN_SOURCE
    if text.startswith("template_relevance."):
        return "template_relevance"
    return "chem_enzy_one_step"


def _literature_template_metadata(template_payload: Any) -> dict[str, Any]:
    if not isinstance(template_payload, dict):
        return {}
    source = str(template_payload.get("source") or template_payload.get("source_model") or "")
    model = str(template_payload.get("model_full_name") or "")
    if source != LITERATURE_TEMPLATE_PLUGIN_SOURCE and model != LITERATURE_TEMPLATE_PLUGIN_MODEL:
        return {}
    return {
        "source_model": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "evidence_refs": list(template_payload.get("evidence_refs") or []),
        "not_lab_procedure": bool(template_payload.get("not_lab_procedure")),
        "requires_audit": bool(template_payload.get("requires_audit", True)),
        "template_validation_report": dict(template_payload.get("template_validation_report") or {}),
        "template_applicability_report": dict(template_payload.get("template_applicability_report") or {}),
        "literature_template_trace": dict(template_payload.get("literature_template_trace") or {}),
        "source_policy_decision": "literature_template_plugin",
        "condition_source": str(template_payload.get("condition_source") or "unknown"),
        "no_solved_claim": bool(template_payload.get("no_solved_claim", True)),
    }


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


def _graphfp_reranker_from_env() -> Any | None:
    raw = str(os.environ.get(GRAPHFP_RERANKER_ENABLE_ENV) or "").lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None
    try:
        from cascade_planner.baselines.graphfp_lightgbm_reranker import GraphFPLightGBMReranker

        return GraphFPLightGBMReranker.from_env()
    except Exception:
        return None


def _graphfp_dualtower_fusion_from_env() -> Any | None:
    try:
        from cascade_planner.baselines.graphfp_dualtower_fusion import fusion_from_env

        return fusion_from_env()
    except Exception:
        return None
