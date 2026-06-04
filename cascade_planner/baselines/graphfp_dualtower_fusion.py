"""Runtime fusion of ChemEnzy one-step rows with the chemical dual-tower retriever."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade_planner.baselines.proposal_gate import evaluate_step_candidate
from cascade_planner.cascadeboard.route_recovery import canonical_side, canonical_smiles


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUE_VALUES = {"1", "true", "yes", "on"}

FUSION_ENABLE_ENV = "AUTOPLANNER_ENABLE_GRAPHFP_DUALTOWER_FUSION"
FUSION_MODEL_PATH_ENV = "AUTOPLANNER_DUALTOWER_TEMPLATE_MODEL"
FUSION_TEMPLATE_CACHE_ENV = "AUTOPLANNER_DUALTOWER_TEMPLATE_VECTOR_CACHE"
FUSION_TEMPLATES_INDEX_ENV = "AUTOPLANNER_DUALTOWER_TEMPLATES_INDEX"
FUSION_DUAL_TOPK_ENV = "AUTOPLANNER_DUALTOWER_TEMPLATE_TOPK"
FUSION_BASE_TOPK_ENV = "AUTOPLANNER_GRAPHFP_FUSION_INTERNAL_TOPK"
FUSION_MODE_ENV = "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_MODE"
FUSION_DEVICE_ENV = "AUTOPLANNER_DUALTOWER_TEMPLATE_DEVICE"
FUSION_TEMPLATE_BATCH_ENV = "AUTOPLANNER_DUALTOWER_TEMPLATE_BATCH_SIZE"

DEFAULT_MODEL_PATH = Path(
    "results/shared/dual_tower_template_retriever_20260530/enhanced_v2_fulltrain_e8_ft/dual_tower_fp_retriever.pt"
)
DEFAULT_TEMPLATE_VECTOR_CACHE = Path(
    "results/shared/dual_tower_template_retriever_20260530/enhanced_v2_fulltrain_e8_ft/template_vectors_329k.pt"
)
DEFAULT_TEMPLATES_INDEX = Path(
    "vendor/ChemEnzyRetroPlanner/retro_planner/packages/graph_retrosyn/graph_retrosyn/data/raw/templates_index.pkl"
)


@dataclass(frozen=True)
class GraphFPDualTowerFusionConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    template_vector_cache: Path = DEFAULT_TEMPLATE_VECTOR_CACHE
    templates_index: Path = DEFAULT_TEMPLATES_INDEX
    graphfp_topk: int = 50
    dual_topk: int = 100
    mode: str = "rrf"
    device: str | None = None
    template_batch_size: int = 4096


class GraphFPDualTowerFusion:
    """Load the enhanced template retriever once and fuse rows at prediction time."""

    def __init__(self, config: GraphFPDualTowerFusionConfig) -> None:
        self.config = config
        self._loaded = False
        self._torch: Any | None = None
        self._rdchiral_run_text: Any | None = None
        self._model: Any | None = None
        self._templates: list[str] = []
        self._template_vectors: Any | None = None
        self._product_features: Any | None = None
        self._n_bits = 512
        self._feature_set = "enhanced"
        self._device: Any | None = None

    @property
    def graphfp_topk(self) -> int:
        return max(1, int(self.config.graphfp_topk or 50))

    @property
    def dual_topk(self) -> int:
        return max(1, int(self.config.dual_topk or 100))

    @property
    def mode(self) -> str:
        return str(self.config.mode or "rrf")

    def dual_rows(self, product: str) -> list[dict[str, Any]]:
        if not product:
            return []
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._template_vectors is not None
        assert self._product_features is not None
        assert self._rdchiral_run_text is not None

        with torch.no_grad():
            import numpy as np

            fps = np.asarray([self._product_features(product, self._n_bits, self._feature_set)], dtype=np.float32)
            product_vec = self._model.product_tower(torch.from_numpy(fps).to(self._device))
            scores = product_vec @ self._template_vectors.T
            topk = min(self.dual_topk, int(self._template_vectors.shape[0]))
            values, indices = torch.topk(scores, k=topk, dim=1)
        template_ids = [int(i) for i in indices.detach().cpu().numpy()[0]]
        template_scores = [float(x) for x in values.detach().cpu().numpy()[0]]
        return self._apply_templates(product, template_ids, template_scores)

    def fuse(
        self,
        product: str,
        base_rows: list[dict[str, Any]],
        dual_rows: list[dict[str, Any]],
        *,
        output_k: int,
    ) -> list[dict[str, Any]]:
        return fuse_graphfp_dualtower_rows(
            product=product,
            base_rows=base_rows,
            dual_rows=dual_rows,
            output_k=output_k,
            mode=self.mode,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from rdchiral.main import rdchiralRunText
        from scripts.evaluate_dual_tower_template_retriever import _idx_ordered_templates, _template_vectors
        from scripts.train_dual_tower_template_retriever import (
            DualTowerRetriever,
            feature_dims,
            product_features,
        )

        ckpt_path = _resolve_path(self.config.model_path)
        templates_index = _resolve_path(self.config.templates_index)
        template_vector_cache = _resolve_path(self.config.template_vector_cache)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"dual-tower model not found: {ckpt_path}")
        if not templates_index.exists():
            raise FileNotFoundError(f"template index not found: {templates_index}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        settings = ckpt.get("settings") or {}
        self._n_bits = int(settings.get("n_bits") or 512)
        hidden = int(settings.get("hidden") or 1024)
        dim = int(settings.get("dim") or 256)
        dropout = float(settings.get("dropout") or 0.0)
        self._feature_set = str(settings.get("feature_set") or "enhanced")
        architecture = str(settings.get("architecture") or "residual")
        product_dim, template_dim = feature_dims(self._n_bits, self._feature_set)
        product_dim = int(settings.get("product_dim") or product_dim)
        template_dim = int(settings.get("template_dim") or template_dim)
        self._device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = DualTowerRetriever(
            n_bits=self._n_bits,
            hidden=hidden,
            dim=dim,
            dropout=dropout,
            product_dim=product_dim,
            template_dim=template_dim,
            architecture=architecture,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(self._device)
        model.eval()

        _template2idx, idx2template = torch.load(templates_index, map_location="cpu")
        templates = _idx_ordered_templates(idx2template)
        template_vectors = _template_vectors(
            model=model,
            templates=templates,
            n_bits=self._n_bits,
            batch_size=max(1, int(self.config.template_batch_size or 4096)),
            device=self._device,
            cache_path=template_vector_cache,
            feature_set=self._feature_set,
            template_dim=template_dim,
            feature_workers=0,
        )
        self._torch = torch
        self._rdchiral_run_text = rdchiralRunText
        self._model = model
        self._templates = templates
        self._template_vectors = template_vectors
        self._product_features = product_features
        self._loaded = True

    def _apply_templates(
        self,
        product: str,
        template_ids: list[int],
        template_scores: list[float],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for template_rank, (template_id, score) in enumerate(zip(template_ids, template_scores), start=1):
            try:
                outcomes = sorted(self._rdchiral_run_text(self._templates[template_id], product))
            except Exception:
                outcomes = []
            for reactants in outcomes:
                key = canonical_side(reactants)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(
                    _dual_candidate_row(
                        product=product,
                        reactants=list(key),
                        template=self._templates[template_id],
                        template_id=template_id,
                        template_rank=template_rank,
                        candidate_rank=len(out) + 1,
                        score=float(score),
                    )
                )
        candidate_count = len(out)
        for row in out:
            row["candidate_count"] = candidate_count
        return out


def fuse_graphfp_dualtower_rows(
    *,
    product: str,
    base_rows: list[dict[str, Any]],
    dual_rows: list[dict[str, Any]],
    output_k: int,
    mode: str = "rrf",
) -> list[dict[str, Any]]:
    """Fuse native rows and dual-tower rows by canonical reactant set."""
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for rank, row in enumerate(base_rows or [], start=1):
        key = _row_reactants_key(row)
        if not key:
            continue
        item = dict(row)
        native_rank = _safe_int(item.get("rank"), rank)
        item["native_rank"] = native_rank
        item["graphfp_rank"] = native_rank
        item["dualtower_rank"] = None
        item["fusion_sources"] = sorted({str(item.get("source") or "native")})
        merged.setdefault(key, item)

    for rank, row in enumerate(dual_rows or [], start=1):
        key = _row_reactants_key(row)
        if not key:
            continue
        dual_rank = _safe_int(row.get("rank"), rank)
        if key in merged:
            item = merged[key]
            item["dualtower_rank"] = dual_rank
            item["dualtower_score"] = _safe_float(row.get("score"), 0.0)
            item["fusion_sources"] = sorted(set(item.get("fusion_sources") or []) | {"autoplanner_dualtower"})
            item.setdefault("dualtower_template_id", row.get("template_id"))
            item.setdefault("dualtower_template_rank", row.get("template_rank"))
        else:
            item = dict(row)
            item["native_rank"] = None
            item["graphfp_rank"] = None
            item["dualtower_rank"] = dual_rank
            item["dualtower_score"] = _safe_float(row.get("score"), 0.0)
            item["fusion_sources"] = ["autoplanner_dualtower"]
            merged[key] = item

    rows = list(merged.values())
    for row in rows:
        row["fusion_score"] = _fusion_score(row, mode=mode)
        row["teacher_source"] = "autoplanner_graphfp_dualtower_fusion"
        row["fusion_mode"] = mode
    rows.sort(key=lambda row: row["fusion_score"], reverse=True)
    out = []
    for rank, row in enumerate(rows[: max(1, int(output_k or 1))], start=1):
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


def fusion_from_env() -> GraphFPDualTowerFusion | None:
    if str(os.environ.get(FUSION_ENABLE_ENV) or "").strip().lower() not in TRUE_VALUES:
        return None
    config = GraphFPDualTowerFusionConfig(
        model_path=Path(os.environ.get(FUSION_MODEL_PATH_ENV) or DEFAULT_MODEL_PATH),
        template_vector_cache=Path(os.environ.get(FUSION_TEMPLATE_CACHE_ENV) or DEFAULT_TEMPLATE_VECTOR_CACHE),
        templates_index=Path(os.environ.get(FUSION_TEMPLATES_INDEX_ENV) or DEFAULT_TEMPLATES_INDEX),
        graphfp_topk=_env_int(FUSION_BASE_TOPK_ENV, 50),
        dual_topk=_env_int(FUSION_DUAL_TOPK_ENV, 100),
        mode=str(os.environ.get(FUSION_MODE_ENV) or "rrf"),
        device=os.environ.get(FUSION_DEVICE_ENV) or None,
        template_batch_size=_env_int(FUSION_TEMPLATE_BATCH_ENV, 4096),
    )
    key = (
        str(_resolve_path(config.model_path)),
        str(_resolve_path(config.template_vector_cache)),
        str(_resolve_path(config.templates_index)),
        config.graphfp_topk,
        config.dual_topk,
        config.mode,
        config.device,
        config.template_batch_size,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = GraphFPDualTowerFusion(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime


def _dual_candidate_row(
    *,
    product: str,
    reactants: list[str],
    template: str,
    template_id: int,
    template_rank: int,
    candidate_rank: int,
    score: float,
) -> dict[str, Any]:
    main = _largest_smiles(reactants)
    aux = [smi for smi in reactants if (canonical_smiles(smi) or smi) != (canonical_smiles(main) or main)]
    rxn_smiles = ".".join(reactants) + f">>{product}"
    proposal_gate = evaluate_step_candidate(
        product_smiles=product,
        reactant_smiles=reactants,
        rxn_smiles=rxn_smiles,
        source_model="autoplanner_dualtower_template",
    )
    return {
        "main_reactant": main,
        "aux_reactants": aux,
        "reactant_smiles": reactants,
        "rxn_smiles": rxn_smiles,
        "reaction_smiles": rxn_smiles,
        "source": "autoplanner_dualtower",
        "score": float(score),
        "rank": int(candidate_rank),
        "candidate_count": 0,
        "type": "template",
        "proposal_type": "chemical_dualtower_template",
        "template": template,
        "template_id": int(template_id),
        "template_rank": int(template_rank),
        "model_full_name": "autoplanner_dualtower.enhanced_v2_e8_ft",
        "cost": None,
        "weight": 1.0,
        "teacher_one_step": True,
        "teacher_source": "autoplanner_dualtower_template",
        "proposal_gate": proposal_gate,
    }


def _row_reactants_key(row: dict[str, Any]) -> tuple[str, ...]:
    rxn = str(row.get("reaction_smiles") or row.get("rxn_smiles") or "")
    if ">>" in rxn:
        return canonical_side(rxn.split(">>", 1)[0])
    parts = [str(item) for item in row.get("reactant_smiles") or [] if str(item or "")]
    return canonical_side(".".join(parts))


def _fusion_score(row: dict[str, Any], *, mode: str) -> float:
    native_rank = row.get("native_rank") or row.get("graphfp_rank")
    dual_rank = row.get("dualtower_rank")
    if mode == "graphfp_first":
        return 1e6 - (native_rank if native_rank is not None else 100000 + int(dual_rank or 100000))
    if mode == "best_rank":
        return 1e6 - min(
            int(native_rank) if native_rank is not None else 100000,
            int(dual_rank) if dual_rank is not None else 100000,
        )
    if mode == "score_sum":
        native_score = math.log(max(_safe_float(row.get("score"), 0.0), 1e-12)) if native_rank is not None else -12.0
        dual_score = _safe_float(row.get("dualtower_score"), _safe_float(row.get("score"), 0.0)) if dual_rank is not None else 0.0
        return native_score + dual_score
    return (1.0 / (60.0 + int(native_rank or 100000))) + (1.0 / (60.0 + int(dual_rank or 100000)))


def _largest_smiles(smiles: list[str]) -> str:
    if not smiles:
        return ""
    return max(smiles, key=lambda smi: (len(canonical_smiles(smi) or smi), canonical_smiles(smi) or smi))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return int(default)


def _resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


_RUNTIME_CACHE: dict[tuple[Any, ...], GraphFPDualTowerFusion] = {}
