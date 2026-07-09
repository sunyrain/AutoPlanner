"""Proposal provider adapters for cascade-native search."""
from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    _patch_numpy_legacy_aliases,
    _vendor_pythonpath,
)
from cascade_planner.baselines.chem_enzy_context import build_chem_enzy_context_source
from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider
from cascade_planner.baselines.route_contract import RouteSearchConfig, RouteStepCandidate
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles
from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.ids import stable_id
from cascade_planner.cascade_search.proposal_preference import score_candidate as score_proposal_preference
from cascade_planner.cascade_search.proposal_validity import (
    ProposalValidityConfig,
    filter_reactant_predictions,
)
from cascade_planner.cascade_search.state import (
    CascadeAction,
    CascadeActionType,
    CascadeModule,
    CascadeProgramState,
    ConditionEnvelope,
    StepAnnotation,
)

RDLogger.DisableLog("rdApp.*")


@dataclass
class ProposalRequest:
    leaf_smiles: str
    state: CascadeProgramState
    depth_remaining: int = 0
    top_k: int = 10
    failure_modes: list[str] = field(default_factory=list)
    source_budgets: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.to_dict()
        return data


@dataclass
class ProposalDiagnostics:
    provider_name: str
    requested: int = 0
    returned: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChemEnzyProposalProvider:
    """Use ChemEnzy as a proposal provider, not as the main planner."""

    provider_name = "chem_enzy"

    def __init__(
        self,
        adapter: ChemEnzyBackendAdapter | None = None,
        *,
        vendor_root: Path | str | None = None,
        stock_names: list[str] | None = None,
        one_step_models: list[str] | None = None,
        max_iterations: int = 1,
        max_depth: int = 1,
        expansion_topk: int = 50,
        dry_run: bool = False,
    ):
        if adapter is not None:
            self.adapter = adapter
        elif vendor_root is not None:
            self.adapter = ChemEnzyBackendAdapter(vendor_root=vendor_root)
        else:
            self.adapter = ChemEnzyBackendAdapter()
        self.stock_names = list(stock_names or [])
        self.one_step_models = list(one_step_models or [])
        self.max_iterations = int(max_iterations)
        self.max_depth = int(max_depth)
        self.expansion_topk = int(expansion_topk)
        self.dry_run = bool(dry_run)
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or self.expansion_topk)
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or self.expansion_topk)
        if not leaf:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={"empty_reason": "missing_leaf"},
            )
            return []
        config = RouteSearchConfig(
            target_smiles=leaf,
            stock_names=self.stock_names,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            expansion_topk=top_k,
            one_step_models=self.one_step_models,
        )
        result = self.adapter.run_target(config, dry_run=self.dry_run)
        actions: list[CascadeAction] = []
        for route in result.routes:
            for step in route.steps:
                if step.product_smiles and step.product_smiles != leaf:
                    continue
                actions.append(route_step_candidate_to_action(step, provider_name=self.provider_name))
                if len(actions) >= top_k:
                    break
            if len(actions) >= top_k:
                break
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            failures=[failure.to_dict() for failure in result.failures],
            metadata={
                "backend": result.backend,
                "solved": result.solved,
                "route_count": result.route_count,
            },
        )
        return actions


class ChemEnzyContextONMTProposalProvider:
    """Experimental context-conditioned ONMT sidecar proposer."""

    provider_name = "chem_enzy_context_onmt"

    def __init__(
        self,
        *,
        model_path: Path | str,
        vendor_root: Path | str = "vendor/ChemEnzyRetroPlanner",
        beam_size: int = 5,
        topk: int = 10,
        batch_size: int = 16,
        device: int = -1,
        min_score: float = 0.0,
        max_context_step: int | None = None,
        preference_scorer_path: Path | str | None = None,
        preference_min_score: float | None = None,
        preference_rerank: bool = False,
        filter_invalid_proposals: bool = True,
        max_reactant_to_product_heavy_ratio: float | None = None,
        raw_topk_multiplier: int = 3,
        translator: Any | None = None,
    ):
        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        self.model_path = model_path
        self.vendor_root = Path(vendor_root)
        self.beam_size = int(beam_size)
        self.topk = int(topk)
        self.batch_size = int(batch_size)
        self.device = int(device)
        self.min_score = max(0.0, float(min_score or 0.0))
        self.max_context_step = int(max_context_step) if max_context_step is not None else None
        self.preference_scorer_path = _resolve_optional_path(preference_scorer_path)
        self.preference_min_score = (
            max(0.0, min(1.0, float(preference_min_score)))
            if preference_min_score is not None
            else None
        )
        self.preference_rerank = bool(preference_rerank)
        self.filter_invalid_proposals = bool(filter_invalid_proposals)
        self.max_reactant_to_product_heavy_ratio = (
            float(max_reactant_to_product_heavy_ratio)
            if max_reactant_to_product_heavy_ratio is not None
            else None
        )
        self.raw_topk_multiplier = max(1, int(raw_topk_multiplier or 1))
        self.preference_scorer: Any | None = None
        self.preference_load_error = ""
        self.translator = translator
        self.opt: Any | None = None
        self._loaded = translator is not None
        self.load_error = ""
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    @property
    def available(self) -> bool:
        return self.translator is not None or self.model_path.exists()

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            state = request.state
            top_k = int(request.top_k or kwargs.get("top_k") or self.topk)
        else:
            leaf = str(request or "")
            state = kwargs.get("state")
            top_k = int(kwargs.get("top_k") or self.topk)
        if not leaf:
            self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name, requested=top_k, returned=0)
            return []
        step_index = len(getattr(state, "step_annotations", []) or []) if state is not None else 0
        context_step = step_index + 1
        if self.max_context_step is not None and context_step > self.max_context_step:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={
                    "model_path": str(self.model_path),
                    "skipped_reason": "context_step_limit",
                    "context_step": context_step,
                    "max_context_step": self.max_context_step,
                },
            )
            return []
        try:
            source = build_chem_enzy_context_source(
                target_smiles=str(getattr(state, "target_smiles", "") or leaf),
                product_smiles=leaf,
                state=state,
                step_index=step_index,
            )
            rows = self.predict_pretokenized_source(source, product_smiles=leaf, top_k=top_k)
            if self.min_score > 0:
                rows = [row for row in rows if _safe_float(row.get("score")) is not None and float(row["score"]) >= self.min_score]
            rows = self._score_filter_and_rank_preferences(rows, product_smiles=leaf)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                failures=[{"category": "context_onmt_failed", "message": self.load_error}],
                metadata={"model_path": str(self.model_path)},
            )
            return []
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows[:top_k]]
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            metadata={
                "model_path": str(self.model_path),
                "context_source": source,
                "min_score": self.min_score,
                "max_context_step": self.max_context_step,
                "preference_scorer_path": str(self.preference_scorer_path) if self.preference_scorer_path else None,
                "preference_min_score": self.preference_min_score,
                "preference_rerank": self.preference_rerank,
                "preference_load_error": self.preference_load_error,
                "filter_invalid_proposals": self.filter_invalid_proposals,
                "max_reactant_to_product_heavy_ratio": self.max_reactant_to_product_heavy_ratio,
                "raw_topk_multiplier": self.raw_topk_multiplier,
                "validity_filter_rejected": sum(
                    int((getattr(action.step, "raw_metadata", {}) or {}).get("validity_filter_rejected_count") or 0)
                    for action in actions
                ),
            },
        )
        return actions

    def predict_pretokenized_source(self, source: str, *, product_smiles: str, top_k: int) -> list[dict[str, Any]]:
        opt, translator = self._ensure_translator()
        from onmt.bin.translate import translate  # type: ignore

        opt.batch_size = max(1, int(self.batch_size))
        requested_top_k = max(1, int(top_k or 1))
        raw_top_k = requested_top_k * self.raw_topk_multiplier if self.filter_invalid_proposals else requested_top_k
        previous_topk = getattr(opt, "topk", None)
        previous_beam_size = getattr(opt, "beam_size", None)
        previous_n_best = getattr(opt, "n_best", None)
        opt.topk = max(int(getattr(opt, "topk", raw_top_k) or raw_top_k), raw_top_k)
        opt.beam_size = max(int(getattr(opt, "beam_size", raw_top_k) or raw_top_k), raw_top_k)
        opt.n_best = max(int(getattr(opt, "n_best", raw_top_k) or raw_top_k), raw_top_k)
        scores, predictions = translate(translator, opt, [source])
        if previous_topk is not None:
            opt.topk = previous_topk
        if previous_beam_size is not None:
            opt.beam_size = previous_beam_size
        if previous_n_best is not None:
            opt.n_best = previous_n_best
        pred_list = [str(item).replace(" ", "") for item in (predictions[0] if predictions else [])]
        score_list = [float(item) for item in (scores[0] if scores else [])]
        if self.filter_invalid_proposals:
            validity = filter_reactant_predictions(
                pred_list,
                product_smiles=product_smiles,
                config=ProposalValidityConfig(
                    max_reactant_to_product_heavy_ratio=self.max_reactant_to_product_heavy_ratio
                ),
            )
            rejected_by_prediction = {str(row["prediction"]): row["reason"] for row in validity.rejected}
            kept_pairs = _align_kept_predictions_with_scores(pred_list, score_list, validity.kept)
        else:
            validity = None
            rejected_by_prediction = {}
            kept_pairs = [
                (prediction, score_list[idx] if idx < len(score_list) else None)
                for idx, prediction in enumerate(pred_list)
            ]
        rows = []
        for rank, (reactant_text, raw_log_score) in enumerate(kept_pairs[:requested_top_k], 1):
            reactants = [part for part in reactant_text.split(".") if part]
            if not reactants:
                continue
            if _is_self_reaction(product_smiles, reactants):
                continue
            rows.append(
                {
                    "product_smiles": product_smiles,
                    "reactant_smiles": reactants,
                    "rxn_smiles": ".".join(reactants) + f">>{product_smiles}",
                    "source": self.provider_name,
                    "source_model": self.provider_name,
                    "score": _probability_from_log_score(raw_log_score),
                    "onmt_log_score": raw_log_score,
                    "rank": rank,
                    "proposal_type": self.provider_name,
                    "main_reactant": _largest_smiles(reactants),
                    "aux_reactants": _aux_reactants(reactants),
                    "raw_context_source": source,
                    "validity_filter_enabled": self.filter_invalid_proposals,
                    "validity_filter_rejected_count": len(validity.rejected) if validity is not None else 0,
                    "validity_filter_rejected_reasons": sorted({str(row["reason"]) for row in validity.rejected}) if validity is not None else [],
                    "validity_filter_rejected_by_prediction": rejected_by_prediction,
                    "requested_top_k": requested_top_k,
                    "raw_top_k": raw_top_k,
                    "raw_topk_multiplier": self.raw_topk_multiplier,
                }
            )
        return rows

    def _score_filter_and_rank_preferences(self, rows: list[dict[str, Any]], *, product_smiles: str) -> list[dict[str, Any]]:
        artifact = self._ensure_preference_scorer()
        if artifact is None:
            return rows
        scored = []
        for row in rows:
            reactants = row.get("reactant_smiles") or []
            reactant_side = ".".join(str(smi) for smi in reactants if smi)
            score = _score_preference_artifact(artifact, product=product_smiles, reactants=reactant_side)
            item = dict(row)
            item["preference_score"] = score
            item["preference_scorer"] = str(self.preference_scorer_path) if self.preference_scorer_path else ""
            if self.preference_min_score is not None and score < self.preference_min_score:
                continue
            scored.append(item)
        if self.preference_rerank:
            scored.sort(
                key=lambda item: (
                    float(item.get("preference_score") or 0.0),
                    float(item.get("score") or 0.0),
                ),
                reverse=True,
            )
            for rank, item in enumerate(scored, 1):
                item["preference_rank"] = rank
        return scored

    def _ensure_preference_scorer(self) -> Any | None:
        if self.preference_scorer is not None:
            return self.preference_scorer
        if self.preference_scorer_path is None:
            return None
        try:
            self.preference_scorer = joblib.load(self.preference_scorer_path)
            return self.preference_scorer
        except Exception as exc:
            self.preference_load_error = f"{type(exc).__name__}:{exc}"
            return None

    def _ensure_translator(self) -> tuple[Any, Any]:
        if self.translator is not None and self.opt is not None:
            return self.opt, self.translator
        if self._loaded and self.translator is None:
            raise RuntimeError(self.load_error or "context ONMT provider failed to load")
        self._loaded = True
        with _vendor_pythonpath(self.vendor_root):
            _patch_numpy_legacy_aliases()
            from onmt.bin.translate import load_model  # type: ignore

            self.opt, self.translator = load_model(
                [str(self.model_path)],
                self.beam_size,
                self.topk,
                self.device,
                "char",
            )
        return self.opt, self.translator


class RetroChimeraProposalProvider:
    """Runtime RetroChimera sidecar provider for cascade-native search."""

    provider_name = "retrochimera"

    def __init__(
        self,
        model_dir: Path | str = "data_external/retrochimera_model",
        *,
        device: str | None = None,
        model: Any | None = None,
        retry_on_connection_reset: int = 1,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model = model
        self._loaded = model is not None
        self._external_model = model is not None
        self.retry_on_connection_reset = max(0, int(retry_on_connection_reset or 0))
        self.load_error = ""
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    @property
    def available(self) -> bool:
        return self.model is not None or (self.model_dir / "models.json").exists()

    def predict(self, product_smiles: str, top_k: int = 10, **_: Any) -> list[dict[str, Any]]:
        if not product_smiles:
            return []
        started = time_monotonic()
        attempts = 0
        try:
            while True:
                attempts += 1
                try:
                    model = self._ensure_model()
                    rows = self._predict_with_model(model, product_smiles, top_k=max(1, int(top_k or 1)))
                    break
                except ConnectionResetError:
                    if attempts > self.retry_on_connection_reset + 1:
                        raise
                    self._reset_model_after_failure()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=int(top_k or 0),
                returned=0,
                failures=[{"category": "retrochimera_failed", "message": self.load_error}],
                metadata={"model_dir": str(self.model_dir), "attempts": attempts},
            )
            return []
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=int(top_k or 0),
            returned=len(rows),
            metadata={"model_dir": str(self.model_dir), "elapsed_s": round(time_monotonic() - started, 3), "attempts": attempts},
        )
        return rows

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or 10)
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or 10)
        rows = self.predict(leaf, top_k=top_k)
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows[:top_k]]
        if self.last_diagnostics.returned != len(actions):
            self.last_diagnostics.returned = len(actions)
        return actions

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        if self._loaded:
            raise RuntimeError(self.load_error or "RetroChimera provider failed to load")
        self._loaded = True
        if not self.available:
            raise RuntimeError(f"RetroChimera model directory is missing models.json: {self.model_dir}")
        try:
            from retrochimera import RetroChimeraModel
        except ImportError as exc:
            raise RuntimeError("retrochimera is not importable") from exc
        kwargs = {"model_dir": self.model_dir}
        if self.device:
            kwargs["device"] = self.device
        self.model = RetroChimeraModel(**kwargs)
        return self.model

    def _reset_model_after_failure(self) -> None:
        self.load_error = ""
        if self._external_model:
            return
        self.model = None
        self._loaded = False

    def _predict_with_model(self, model: Any, product_smiles: str, *, top_k: int) -> list[dict[str, Any]]:
        from syntheseus.interface.molecule import Molecule

        raw = model([Molecule(smiles=product_smiles)], num_results=top_k)
        reactions = []
        for item in raw or []:
            reactions.extend(list(item or []))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, reaction in enumerate(reactions):
            reactants = [
                str(getattr(mol, "smiles", "") or "")
                for mol in list(getattr(reaction, "reactants", []) or [])
                if str(getattr(mol, "smiles", "") or "")
            ]
            if not reactants:
                continue
            rxn_smiles = ".".join(reactants) + f">>{product_smiles}"
            key = canonical_reaction_or_raw(rxn_smiles)
            if key in seen:
                continue
            seen.add(key)
            metadata = dict(getattr(reaction, "metadata", {}) or {})
            score = _retrochimera_score(metadata, rank=idx)
            rows.append(
                {
                    "main_reactant": _largest_smiles(reactants),
                    "aux_reactants": _aux_reactants(reactants),
                    "reactant_smiles": reactants,
                    "rxn_smiles": rxn_smiles,
                    "reaction_smiles": rxn_smiles,
                    "source": self.provider_name,
                    "source_model": self.provider_name,
                    "proposal_type": "retrochimera",
                    "type": "template_free_or_template_ensemble",
                    "score": score,
                    "rank": len(rows) + 1,
                    "retrochimera_metadata": _jsonable_metadata(metadata),
                }
            )
            if len(rows) >= top_k:
                break
        return rows


class AiZynthFinderONNXProposalProvider:
    """AiZynthFinder ONNX expansion-policy adapter for fast one-step proposals."""

    provider_name = "aizynth_onnx"

    def __init__(
        self,
        *,
        config_path: Path | str = "workspace/aizdata/config.yml",
        policy_name: str = "uspto",
        topk: int = 8,
        finder: Any | None = None,
        policy: Any | None = None,
        tree_molecule_factory: Any | None = None,
        validity_config: ProposalValidityConfig | None = None,
    ):
        self.config_path = Path(config_path)
        self.policy_name = str(policy_name or "uspto")
        self.topk = max(1, int(topk or 1))
        self.finder = finder
        self.policy = policy
        self.tree_molecule_factory = tree_molecule_factory
        self.validity_config = validity_config or ProposalValidityConfig()
        self._loaded = finder is not None or policy is not None
        self.load_error = ""
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    @property
    def available(self) -> bool:
        return self.policy is not None or self.finder is not None or self.config_path.exists()

    def predict(self, product_smiles: str, top_k: int = 10, **_: Any) -> list[dict[str, Any]]:
        if not product_smiles:
            return []
        started = time_monotonic()
        requested = max(1, int(top_k or self.topk or 1))
        raw_action_count = 0
        try:
            policy = self._ensure_policy()
            tree_mol = self._tree_molecule(product_smiles)
            actions, priors = policy.get_actions([tree_mol])
            actions = list(actions or [])
            priors = list(priors or [])
            raw_action_count = len(actions)
            rows = self._rows_from_actions(
                product_smiles,
                actions,
                priors,
                top_k=requested,
            )
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=requested,
                returned=0,
                failures=[{"category": "aizynth_onnx_failed", "message": self.load_error}],
                metadata={"config_path": str(self.config_path), "policy_name": self.policy_name},
            )
            return []
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=requested,
            returned=len(rows),
            metadata={
                "config_path": str(self.config_path),
                "policy_name": self.policy_name,
                "raw_action_count": raw_action_count,
                "elapsed_s": round(time_monotonic() - started, 3),
                "validity_filter": asdict(self.validity_config),
            },
        )
        return rows

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or self.topk)
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or self.topk)
        rows = self.predict(leaf, top_k=top_k)
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows[:top_k]]
        if self.last_diagnostics.returned != len(actions):
            self.last_diagnostics.returned = len(actions)
        return actions

    def _ensure_policy(self) -> Any:
        if self.policy is not None:
            return self.policy
        if self.finder is not None:
            self.finder.expansion_policy.select(self.policy_name)
            self.policy = self.finder.expansion_policy
            return self.policy
        if self._loaded:
            raise RuntimeError(self.load_error or "AiZynthFinder ONNX policy failed to load")
        self._loaded = True
        if not self.config_path.exists():
            raise RuntimeError(f"AiZynthFinder config is missing: {self.config_path}")
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder
        except ImportError as exc:
            raise RuntimeError("aizynthfinder is not importable") from exc
        self.finder = AiZynthFinder(configfile=str(self.config_path))
        self.finder.expansion_policy.select(self.policy_name)
        self.policy = self.finder.expansion_policy
        return self.policy

    def _tree_molecule(self, product_smiles: str) -> Any:
        if self.tree_molecule_factory is not None:
            return self.tree_molecule_factory(product_smiles)
        from aizynthfinder.chem import TreeMolecule

        return TreeMolecule(parent=None, smiles=product_smiles)

    def _rows_from_actions(
        self,
        product_smiles: str,
        actions: list[Any],
        priors: list[Any],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_sides: set[tuple[str, ...]] = set()
        for action_rank, action in enumerate(actions, start=1):
            metadata = dict(getattr(action, "metadata", {}) or {})
            score = _safe_float(metadata.get("policy_probability"))
            if score is None and action_rank - 1 < len(priors):
                score = _safe_float(priors[action_rank - 1])
            if score is None:
                score = 1.0 / float(action_rank)
            try:
                reactant_options = list(getattr(action, "reactants", []) or [])
            except Exception:
                reactant_options = []
            for option in reactant_options:
                option_items = list(option) if isinstance(option, (list, tuple)) else [option]
                reactants = [
                    canonical_smiles(str(getattr(mol, "smiles", "") or mol)) or str(getattr(mol, "smiles", "") or mol)
                    for mol in option_items
                    if str(getattr(mol, "smiles", "") or mol)
                ]
                side_text = ".".join(reactants)
                report = filter_reactant_predictions(
                    [side_text],
                    product_smiles=product_smiles,
                    config=self.validity_config,
                )
                if not report.kept:
                    continue
                side_key = canonical_side(report.kept[0])
                if not side_key or side_key in seen_sides:
                    continue
                seen_sides.add(side_key)
                reactants = list(side_key)
                rxn_smiles = ".".join(reactants) + f">>{product_smiles}"
                rows.append(
                    {
                        "product_smiles": product_smiles,
                        "reactant_smiles": reactants,
                        "rxn_smiles": rxn_smiles,
                        "reaction_smiles": rxn_smiles,
                        "source": self.provider_name,
                        "source_model": f"{self.provider_name}.{self.policy_name}",
                        "proposal_type": "aizynthfinder_onnx_policy",
                        "type": "aizynthfinder_onnx_template_policy",
                        "score": float(score),
                        "rank": len(rows) + 1,
                        "main_reactant": _largest_smiles(reactants),
                        "aux_reactants": _aux_reactants(reactants),
                        "aizynth_policy": self.policy_name,
                        "aizynth_action_rank": action_rank,
                        "aizynth_metadata": _jsonable_metadata(metadata),
                        "contract": "AiZynthFinder ONNX expansion-policy proposal; fast sidecar, not a route-quality claim.",
                    }
                )
                if len(rows) >= top_k:
                    return rows
        return rows


class TemplateRelevanceProposalProvider:
    """Template-based lightweight expansion sidecar for cascade-native search."""

    provider_name = "template_relevance"

    def __init__(
        self,
        *,
        vendor_root: Path | str = "vendor/ChemEnzyRetroPlanner",
        models: tuple[str, ...] = ("template_relevance.reaxys_biocatalysis",),
        expansion_topk: int = 8,
        gpu: int = -1,
        one_step: Any | None = None,
    ):
        self.vendor_root = Path(vendor_root)
        self.models = tuple(models or ())
        self.expansion_topk = max(1, int(expansion_topk or 1))
        self.gpu = int(gpu)
        self.one_step = one_step
        self._loaded = one_step is not None
        self.load_error = ""
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    @property
    def available(self) -> bool:
        if self.one_step is not None:
            return True
        return self.vendor_root.exists() and (self.vendor_root / "retro_planner" / "config" / "config.yaml").exists()

    def predict(self, product_smiles: str, top_k: int = 10, **_: Any) -> list[dict[str, Any]]:
        if not product_smiles:
            return []
        started = time_monotonic()
        try:
            one_step = self._ensure_one_step()
            rows = list(one_step.predict(product_smiles, top_k=max(1, int(top_k or self.expansion_topk or 1))) or [])
            rows = self._normalize_rows(rows)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=int(top_k or 0),
                returned=0,
                failures=[{"category": "template_relevance_failed", "message": self.load_error}],
                metadata={
                    "vendor_root": str(self.vendor_root),
                    "models": list(self.models),
                    "gpu": self.gpu,
                },
            )
            return []
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=int(top_k or 0),
            returned=len(rows),
            metadata={
                "vendor_root": str(self.vendor_root),
                "models": list(self.models),
                "gpu": self.gpu,
                "elapsed_s": round(time_monotonic() - started, 3),
            },
        )
        return rows

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or self.expansion_topk)
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or self.expansion_topk)
        rows = self.predict(leaf, top_k=top_k)
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows[:top_k]]
        if self.last_diagnostics.returned != len(actions):
            self.last_diagnostics.returned = len(actions)
        return actions

    def _ensure_one_step(self) -> Any:
        if self.one_step is not None:
            return self.one_step
        if self._loaded:
            raise RuntimeError(self.load_error or "template relevance provider failed to load")
        self._loaded = True
        if not self.models:
            raise RuntimeError("template relevance provider requires at least one model name")
        self.one_step = ChemEnzyOneStepProposalProvider(
            vendor_root=self.vendor_root,
            models=tuple(self.models),
            expansion_topk=self.expansion_topk,
            gpu=self.gpu,
        )
        return self.one_step

    def _normalize_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row or {})
            item["source"] = self.provider_name
            item["source_model"] = self.provider_name
            item["proposal_type"] = self.provider_name
            item["type"] = "template_relevance"
            item["template_relevance_models"] = list(self.models)
            item["template_relevance_model_count"] = len(self.models)
            normalized.append(item)
        return normalized


class FallbackProposalProvider:
    """Call a sidecar proposer only when the primary leaf cache is weak."""

    provider_name = "fallback"

    def __init__(
        self,
        inner: Any,
        *,
        primary: Any,
        trigger_min_primary: int = 1,
        max_calls: int | None = None,
        provider_name: str | None = None,
    ):
        self.inner = inner
        self.primary = primary
        self.trigger_min_primary = max(0, int(trigger_min_primary or 0))
        self.max_calls = max(0, int(max_calls)) if max_calls is not None else None
        self.calls = 0
        self.provider_name = provider_name or str(getattr(inner, "provider_name", type(inner).__name__))
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)
        self._cache: dict[tuple[str, int], list[CascadeAction]] = {}

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        leaf = request.leaf_smiles if isinstance(request, ProposalRequest) else str(request or "")
        top_k = int((request.top_k if isinstance(request, ProposalRequest) else kwargs.get("top_k")) or 10)
        primary_count = self._primary_count(leaf, top_k)
        metadata = {
            "trigger_min_primary": self.trigger_min_primary,
            "primary_provider": str(getattr(self.primary, "provider_name", type(self.primary).__name__)),
            "primary_returned": primary_count,
            "max_calls": self.max_calls,
            "calls": self.calls,
        }
        if primary_count >= self.trigger_min_primary:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={**metadata, "skipped_reason": "primary_sufficient"},
            )
            return []
        if self.max_calls is not None and self.calls >= self.max_calls:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={**metadata, "skipped_reason": "max_calls_reached"},
            )
            return []
        cache_key = (leaf, top_k)
        if cache_key in self._cache:
            cached = list(self._cache[cache_key])
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=len(cached),
                metadata={**metadata, "cache_hit": True},
            )
            return cached
        self.calls += 1
        rows = _call_inner_provider(self.inner, request, *args, top_k=top_k, **kwargs)
        actions = [
            coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name)
            for row in rows[:top_k]
        ]
        self._cache[cache_key] = list(actions)
        inner_diag = getattr(self.inner, "last_diagnostics", None)
        inner_payload = inner_diag.to_dict() if hasattr(inner_diag, "to_dict") else inner_diag
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            metadata={
                **metadata,
                "cache_hit": False,
                "calls": self.calls,
                "inner_provider": str(getattr(self.inner, "provider_name", type(self.inner).__name__)),
                "inner_diagnostics": inner_payload,
            },
        )
        return actions

    def _primary_count(self, leaf: str, top_k: int) -> int:
        proposals = getattr(self.primary, "proposals_by_leaf", None)
        if isinstance(proposals, dict):
            return len(list(proposals.get(leaf, []))[:top_k])
        return 0


def _call_inner_provider(provider: Any, request: ProposalRequest | str, *args: Any, top_k: int, **kwargs: Any) -> list[Any]:
    try:
        if isinstance(request, ProposalRequest):
            return list(provider.propose(request) or [])
        return list(provider.propose(request, *args, **kwargs) or [])
    except TypeError:
        try:
            if isinstance(request, ProposalRequest):
                return list(provider.propose(request.leaf_smiles, request.state, top_k=top_k) or [])
            return list(provider.propose(request, *args, top_k=top_k) or [])
        except TypeError:
            leaf = request.leaf_smiles if isinstance(request, ProposalRequest) else str(request or "")
            return list(provider.propose(leaf, top_k=top_k) or [])


class RouteTreeProposalProvider:
    """Adapt existing route_tree proposal tools into cascade-native actions."""

    provider_name = "route_tree"

    def __init__(self, proposal_tool: Any):
        self.proposal_tool = proposal_tool
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or 10)
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or 10)
        rows = list(self.proposal_tool.propose(leaf, top_k=top_k) or [])
        actions = [_candidate_action_to_cascade_action(row, leaf=leaf) for row in rows[:top_k]]
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            metadata=getattr(self.proposal_tool, "last_diagnostics", {}) or {},
        )
        return actions


class ChemicalTemplateProposalProvider:
    """Local USPTO template expansion sidecar backed by the lightweight selector."""

    provider_name = "chemtemplates"

    def __init__(
        self,
        *,
        vendor_root: Path | str = "vendor/ChemEnzyRetroPlanner",
        template_paths: list[Path | str] | tuple[Path | str, ...] | None = None,
        prefer_preselector: bool = True,
        expansion_topk: int = 12,
        one_step: Any | None = None,
        condition_predictor: Any | None = None,
        predict_conditions: bool = False,
        condition_model: str = "rcr",
        condition_prediction_topk: int = 1,
    ):
        self.vendor_root = Path(vendor_root)
        self.template_paths = tuple(Path(path) for path in (template_paths or self._default_template_paths()))
        self.prefer_preselector = bool(prefer_preselector)
        self.expansion_topk = max(1, int(expansion_topk or 1))
        self.one_step = one_step
        self.condition_predictor = condition_predictor
        self.predict_conditions = bool(predict_conditions or condition_predictor is not None)
        self.condition_model = str(condition_model or "rcr")
        self.condition_prediction_topk = max(1, int(condition_prediction_topk or 1))
        self._loaded = one_step is not None
        self._condition_loaded = condition_predictor is not None
        self._condition_prediction_diagnostics: dict[str, Any] = {}
        self.load_error = ""
        self.condition_load_error = ""
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    @property
    def available(self) -> bool:
        if self.one_step is not None:
            return True
        return any(path.exists() for path in self.template_paths)

    def predict(self, product_smiles: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        if not product_smiles:
            return []
        started = time_monotonic()
        self._condition_prediction_diagnostics = {
            "enabled": self.predict_conditions,
            "model": self.condition_model if self.predict_conditions else None,
            "attempted": 0,
            "returned": 0,
            "reliable": 0,
            "failed": 0,
        }
        try:
            applicator = self._ensure_one_step()
            metadata = dict(kwargs.get("metadata") or {})
            ec_token = str(metadata.get("ec_token") or metadata.get("ec") or kwargs.get("ec_token") or "")
            skel_type = str(
                metadata.get("skel_type")
                or metadata.get("reaction_type")
                or metadata.get("transformation")
                or kwargs.get("skel_type")
                or ""
            )
            rows = list(
                applicator.predict(
                    product_smiles,
                    top_k=max(1, int(top_k or self.expansion_topk or 1)),
                    ec_token=ec_token,
                    skel_type=skel_type,
                )
                or []
            )
            rows = self._normalize_rows(rows, applicator=applicator)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=int(top_k or 0),
                returned=0,
                failures=[{"category": "chemtemplates_failed", "message": self.load_error}],
                metadata={
                    "template_paths": [str(path) for path in self.template_paths],
                    "prefer_preselector": self.prefer_preselector,
                },
            )
            return []
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=int(top_k or 0),
            returned=len(rows),
            metadata={
                "template_paths": [str(path) for path in self.template_paths],
                "ranker_mode": self._ranker_mode(applicator),
                "prefer_preselector": self.prefer_preselector,
                "condition_prediction": dict(self._condition_prediction_diagnostics),
                "template_count": int(getattr(applicator, "max_templates", 0) or 0),
                "template_query_cap": int(getattr(applicator, "max_templates_per_query", 0) or 0),
                "outcomes_per_template": int(getattr(applicator, "max_outcomes_per_template", 0) or 0),
                "generalize": int(getattr(applicator, "generalize", 0) or 0),
                "elapsed_s": round(time_monotonic() - started, 3),
            },
        )
        return rows

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or self.expansion_topk)
            metadata = dict(request.metadata or {})
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or self.expansion_topk)
            metadata = dict(kwargs.get("metadata") or {})
        rows = self.predict(leaf, top_k=top_k, metadata=metadata)
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows[:top_k]]
        if self.last_diagnostics.returned != len(actions):
            self.last_diagnostics.returned = len(actions)
        return actions

    def _ensure_one_step(self) -> Any:
        if self.one_step is not None:
            return self.one_step
        if self._loaded:
            raise RuntimeError(self.load_error or "chemical template provider failed to load")
        self._loaded = True
        cache_key = self._one_step_cache_key()
        if cache_key in _CHEM_TEMPLATE_ONE_STEP_CACHE:
            self.one_step = _CHEM_TEMPLATE_ONE_STEP_CACHE[cache_key]
        else:
            self.one_step = self._load_one_step()
            _CHEM_TEMPLATE_ONE_STEP_CACHE[cache_key] = self.one_step
        return self.one_step

    def _one_step_cache_key(self) -> tuple[Any, ...]:
        return (
            tuple(str(path.resolve() if path.exists() else path) for path in self.template_paths),
            bool(self.prefer_preselector),
            _env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES", 20000),
            _env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_EC1", 20000),
            _env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY", 500),
            _env_int("AUTOPLANNER_CHEM_TEMPLATES_OUTCOMES_PER_TEMPLATE", 1),
            _env_int("AUTOPLANNER_CHEM_TEMPLATES_GENERALIZE", 0),
            bool(_torch_sidecars_available()),
        )

    def _load_one_step(self) -> Any:
        from cascade_planner.cascadeboard.chemical_template_applicator import ChemicalTemplateApplicator

        applicator = ChemicalTemplateApplicator(
            template_paths=self.template_paths,
            max_templates=_env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES", 20000),
            max_per_ec1=_env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_EC1", 20000),
            max_templates_per_query=_env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY", 500),
            max_outcomes_per_template=_env_int("AUTOPLANNER_CHEM_TEMPLATES_OUTCOMES_PER_TEMPLATE", 1),
            generalize=_env_int("AUTOPLANNER_CHEM_TEMPLATES_GENERALIZE", 0),
        )
        preselector = None
        pair_ranker = None
        if _torch_sidecars_available():
            from cascade_planner.cascadeboard.chemical_template_pair_ranker import (
                ChemicalTemplatePairRanker,
                pair_ranker_enabled,
            )
            from cascade_planner.cascadeboard.chemical_template_preselector import (
                ChemicalTemplatePreselector,
                preselector_enabled,
            )

            if self.prefer_preselector and preselector_enabled():
                preselector = ChemicalTemplatePreselector.from_env()
                if not getattr(preselector, "available", False):
                    preselector = None
            if preselector is None and pair_ranker_enabled():
                pair_ranker = ChemicalTemplatePairRanker.from_env()
                if not getattr(pair_ranker, "available", False):
                    pair_ranker = None
        applicator.preselector = preselector
        applicator.pair_ranker = pair_ranker
        return applicator

    def _normalize_rows(self, rows: list[dict[str, Any]], *, applicator: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        ranker_mode = self._ranker_mode(applicator)
        for row in rows:
            item = dict(row or {})
            original_source = str(item.get("source") or "")
            original_reaction_type = str(item.get("reaction_type") or item.get("type") or "")
            reactants = [str(smi) for smi in item.get("reactant_smiles") or item.get("reactants") or [] if smi]
            if not reactants:
                main = str(item.get("main_reactant") or "")
                aux = [str(smi) for smi in item.get("aux_reactants") or [] if smi]
                reactants = [smi for smi in [main, *aux] if smi]
            if reactants:
                item["reactant_smiles"] = reactants
            item["template_source"] = original_source
            if original_reaction_type:
                item["template_reaction_type"] = original_reaction_type
            item["source"] = self.provider_name
            item["source_model"] = self.provider_name
            item["proposal_type"] = self.provider_name
            item["type"] = "chemical_template"
            item["reaction_type"] = "chemical_template"
            item["template_family"] = str(item.get("template_family") or "uspto")
            item["template_provider"] = "chemical_template_applicator"
            item["template_ranker_mode"] = ranker_mode
            item["template_ranker_available"] = bool(ranker_mode != "template_only")
            normalized.append(item)
        self._attach_condition_predictions(normalized)
        return normalized

    def _ranker_mode(self, applicator: Any) -> str:
        preselector = getattr(applicator, "preselector", None)
        if preselector is not None and getattr(preselector, "available", False):
            return "preselector"
        pair_ranker = getattr(applicator, "pair_ranker", None)
        if pair_ranker is not None and getattr(pair_ranker, "available", False):
            return "pair_ranker"
        return "template_only"

    def _default_template_paths(self) -> list[Path]:
        from cascade_planner.cascadeboard.chemical_template_applicator import DEFAULT_USPTO_TEMPLATE_PATHS

        return [Path(path) for path in DEFAULT_USPTO_TEMPLATE_PATHS]

    def _attach_condition_prediction(self, item: dict[str, Any]) -> None:
        self._attach_condition_predictions([item])

    def _attach_condition_predictions(self, items: list[dict[str, Any]]) -> None:
        if not self.predict_conditions:
            return
        rxn_rows: list[tuple[dict[str, Any], str]] = []
        for item in items:
            rxn_smiles = _condition_rxn_smiles_for_item(item)
            if ">>" not in rxn_smiles:
                self._condition_prediction_diagnostics["failed"] += 1
                item["condition_prediction_issues"] = ["missing_reaction_smiles"]
                continue
            rxn_rows.append((item, rxn_smiles))
        if not rxn_rows:
            return
        self._condition_prediction_diagnostics["attempted"] += len(rxn_rows)
        try:
            predictor = self._ensure_condition_predictor()
            if hasattr(predictor, "predict_many"):
                raw_by_rxn = predictor.predict_many(
                    [rxn for _item, rxn in rxn_rows],
                    top_k=self.condition_prediction_topk,
                )
            else:
                raw_by_rxn = {
                    rxn: (
                        predictor.predict(rxn, top_k=self.condition_prediction_topk)
                        if hasattr(predictor, "predict")
                        else _call_condition_predictor(predictor, rxn, self.condition_prediction_topk)
                    )
                    for _item, rxn in rxn_rows
                }
        except Exception as exc:
            self.condition_load_error = f"{type(exc).__name__}:{exc}"
            self._condition_prediction_diagnostics["failed"] += len(rxn_rows)
            for item, _rxn in rxn_rows:
                item["condition_prediction_issues"] = ["condition_prediction_failed"]
                item["condition_prediction_error"] = self.condition_load_error
            return
        for item, rxn_smiles in rxn_rows:
            rows = _normalize_condition_prediction_rows(raw_by_rxn.get(rxn_smiles))
            if not rows:
                self._condition_prediction_diagnostics["failed"] += 1
                item["condition_prediction_issues"] = ["condition_prediction_empty"]
                continue
            item["condition_predictions"] = rows
            item["condition_prediction_model"] = self.condition_model
            item["condition_prediction_source"] = "ChemEnzyRCR" if self.condition_model == "rcr" else self.condition_model
            item["condition_prediction_trust"] = "weak_runtime_prediction"
            self._condition_prediction_diagnostics["returned"] += len(rows)
            reliable_rows = [row for row in rows if _condition_prediction_row_is_supported(row)]
            if reliable_rows:
                item["condition"] = reliable_rows[0]
                item["condition_prediction_reliable"] = True
                self._condition_prediction_diagnostics["reliable"] += 1
            else:
                item["condition_prediction_reliable"] = False
                item["condition_prediction_issues"] = sorted(
                    {issue for row in rows for issue in row.get("condition_prediction_issues", [])}
                ) or ["condition_prediction_out_of_supported_range"]

    def _ensure_condition_predictor(self) -> Any:
        if self.condition_predictor is not None:
            return self.condition_predictor
        if self._condition_loaded:
            raise RuntimeError(self.condition_load_error or "condition predictor failed to load")
        self._condition_loaded = True
        self.condition_predictor = _cached_condition_predictor(self.vendor_root, self.condition_model)
        return self.condition_predictor


class CascadeRetrievalProposalProvider:
    """Expose v4 train cascade evidence as sparse proposal supplements."""

    provider_name = "cascade_retrieval"

    def __init__(
        self,
        program_manifest: str | Path,
        *,
        mode: str = "block_downstream_product",
        min_similarity: float = 0.20,
        require_downstream_transform_context: bool = False,
    ):
        self.retrieval = CascadeRetrievalProvider(program_manifest)
        self.mode = str(mode or "block_downstream_product")
        self.min_similarity = float(min_similarity)
        self.require_downstream_transform_context = bool(require_downstream_transform_context)
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or 10)
            metadata = dict(request.metadata or {})
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or 10)
            metadata = dict(kwargs.get("metadata") or {})
        downstream_transform = str(
            metadata.get("downstream_transform")
            or metadata.get("required_downstream_transform")
            or ""
        )
        if self.require_downstream_transform_context and not downstream_transform:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={
                    "empty_reason": "missing_downstream_transform_context",
                    "mode": self.mode,
                },
            )
            return []
        try:
            hits = self.retrieval.retrieve_for_product(
                leaf,
                mode=self.mode,
                limit=top_k,
                min_similarity=self.min_similarity,
                required_downstream_transform=downstream_transform or None,
            )
        except Exception as exc:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                failures=[{"category": "cascade_retrieval_failed", "message": f"{type(exc).__name__}: {exc}"}],
                metadata={"mode": self.mode},
            )
            return []
        actions = []
        for hit in hits:
            action = hit.to_action(target_leaf=leaf)
            action.source = self.provider_name
            if action.step is not None:
                action.step.source_model = self.provider_name
            actions.append(action)
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            metadata={
                "mode": self.mode,
                "min_similarity": self.min_similarity,
                "downstream_transform": downstream_transform,
                "provider_summary": self.retrieval.summary,
                "top_hits": [hit.to_dict() for hit in hits[:5]],
            },
        )
        return actions


class CascadeSubgoalEvidenceProvider:
    """Attach learned v4 cascade subgoal evidence as search-state hints.

    The provider deliberately emits ``EVIDENCE_RETRIEVAL`` actions rather than
    retrosynthetic steps.  A structurally similar literature precedent is a
    planning hint, not a generated reaction.
    """

    provider_name = "cascade_subgoal_evidence"

    def __init__(
        self,
        program_manifest: str | Path,
        model_path: str | Path,
        *,
        preferred_model: str = "structure_metadata_no_rank",
        min_score: float = -0.25,
        min_similarity: float = 0.35,
        evidence_candidates: int = 80,
        max_subgoals_per_leaf: int = 8,
        min_subgoal_heavy_atoms: int = 8,
        max_hints_per_leaf: int = 3,
    ):
        self.program_manifest = Path(program_manifest)
        self.model_path = Path(model_path)
        self.preferred_model = str(preferred_model)
        self.min_score = float(min_score)
        self.min_similarity = float(min_similarity)
        self.evidence_candidates = int(evidence_candidates)
        self.max_subgoals_per_leaf = int(max_subgoals_per_leaf)
        self.min_subgoal_heavy_atoms = int(min_subgoal_heavy_atoms)
        self.max_hints_per_leaf = int(max_hints_per_leaf)
        self._bundle = _load_subgoal_model_bundle(self.model_path)
        self._model = (self._bundle.get("models") or {}).get(self.preferred_model)
        if self._model is None:
            raise ValueError(f"subgoal model {self.preferred_model!r} not found in {self.model_path}")
        self._schema = self._bundle.get("feature_schema") or {}
        self._feature_names = list(self._bundle.get("feature_names") or [])
        self._model_feature_names = (self._bundle.get("model_specs") or {}).get(self.preferred_model) or self._feature_names
        self._feature_indices = [self._feature_names.index(name) for name in self._model_feature_names if name in self._feature_names]
        self._evidence = _load_subgoal_train_evidence(self.program_manifest)
        self._evidence_fps = [row["fingerprint"] for row in self._evidence]
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        if isinstance(request, ProposalRequest):
            leaf = request.leaf_smiles
            top_k = int(request.top_k or kwargs.get("top_k") or 5)
            state = request.state
        else:
            leaf = str(request or "")
            top_k = int(kwargs.get("top_k") or 5)
            state = kwargs.get("state")
        if not leaf:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={"empty_reason": "missing_leaf"},
            )
            return []
        existing = _existing_subgoal_hint_ids(state, leaf)
        if len(existing) >= max(0, self.max_hints_per_leaf):
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={"empty_reason": "leaf_hint_cap_reached", "target_leaf": leaf, "existing_hint_ids": sorted(existing)},
            )
            return []
        hits = self._score_leaf(leaf, state=state, top_k=max(top_k, self.evidence_candidates))
        actions = []
        for hit in hits:
            hint_id = str(hit.get("subgoal_hint_id") or "")
            if hint_id in existing:
                continue
            if float(hit.get("learned_subgoal_score") or 0.0) < self.min_score:
                continue
            if float(hit.get("similarity") or 0.0) < self.min_similarity:
                continue
            actions.append(
                CascadeAction(
                    CascadeActionType.EVIDENCE_RETRIEVAL,
                    target_leaf=leaf,
                    evidence_payload=hit,
                    source=self.provider_name,
                    metadata={
                        "provider": self.provider_name,
                        "learned_subgoal_score": hit.get("learned_subgoal_score"),
                        "similarity": hit.get("similarity"),
                    },
                )
            )
            if len(actions) >= top_k:
                break
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            metadata={
                "target_leaf": leaf,
                "preferred_model": self.preferred_model,
                "min_score": self.min_score,
                "min_similarity": self.min_similarity,
                "evidence_candidates": self.evidence_candidates,
                "max_subgoals_per_leaf": self.max_subgoals_per_leaf,
                "evidence_count": len(self._evidence),
                "top_hits": hits[:5],
            },
        )
        return actions

    def _score_leaf(self, leaf: str, *, state: Any, top_k: int) -> list[dict[str, Any]]:
        from cascade_planner.eval.train_cascade_subgoal_scorer import _candidate_row, _fp

        subgoals = _runtime_subgoal_candidates(
            leaf,
            state,
            max_subgoals=self.max_subgoals_per_leaf,
            min_heavy_atoms=self.min_subgoal_heavy_atoms,
        )
        if not subgoals or not self._evidence_fps:
            return []
        scored = []
        for subgoal in subgoals:
            qfp = _fp(subgoal["smiles"])
            if qfp is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(qfp, self._evidence_fps)
            ranked = np.argsort(np.asarray(sims, dtype=float))[-max(1, int(top_k)) :][::-1]
            query = _runtime_subgoal_query(subgoal["smiles"], state, role=subgoal["query_role"])
            for candidate_rank, idx in enumerate(ranked, start=1):
                evidence = self._evidence[int(idx)]
                sim = float(sims[int(idx)])
                motif_similarity = max(sim, _runtime_motif_similarity(subgoal["smiles"], str(evidence.get("smiles") or "")))
                row = _candidate_row(
                    query,
                    evidence,
                    similarity=motif_similarity,
                    candidate_rank=candidate_rank,
                    schema=self._schema,
                    positive_similarity=0.55,
                    strong_positive_similarity=0.72,
                )
                features = np.asarray([row.get("features")], dtype=np.float32)
                score = float(self._model.predict(features[:, self._feature_indices])[0])
                scored.append(
                    {
                        "subgoal_hint_id": stable_id("subgoal_hint", leaf, subgoal["smiles"], evidence.get("item_id"), round(score, 6)),
                        "target_leaf": leaf,
                        "subgoal_smiles": subgoal["smiles"],
                        "subgoal_source": subgoal["source"],
                        "subgoal_query_role": subgoal["query_role"],
                        "learned_subgoal_score": round(score, 6),
                        "similarity": round(motif_similarity, 6),
                        "fingerprint_similarity": round(sim, 6),
                        "candidate_rank": int(candidate_rank),
                        "evidence_id": evidence.get("item_id"),
                        "evidence_role": evidence.get("role"),
                        "evidence_smiles": evidence.get("smiles"),
                        "evidence_transform": evidence.get("transform"),
                        "evidence_quality_tier": evidence.get("quality_tier"),
                        "evidence_strength": evidence.get("evidence_strength"),
                        "doi": evidence.get("doi"),
                        "cascade_id": evidence.get("cascade_id"),
                        "program_id": evidence.get("program_id"),
                        "cascade_type": evidence.get("cascade_type"),
                        "source": self.provider_name,
                        "contract": "learned subgoal evidence hint; not a retrosynthetic reaction",
                    }
                )
        scored.sort(key=lambda row: (float(row.get("learned_subgoal_score") or 0.0), float(row.get("similarity") or 0.0)), reverse=True)
        return scored


class StaticProposalProvider:
    """Small deterministic provider useful for smoke tests and examples."""

    provider_name = "static"

    def __init__(self, proposals_by_leaf: dict[str, list[CascadeAction | StepAnnotation | dict[str, Any]]]):
        self.proposals_by_leaf = proposals_by_leaf
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        leaf = request.leaf_smiles if isinstance(request, ProposalRequest) else str(request or "")
        top_k = int((request.top_k if isinstance(request, ProposalRequest) else kwargs.get("top_k")) or 10)
        rows = list(self.proposals_by_leaf.get(leaf, []))[:top_k]
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows]
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
        )
        return actions


class LegalCorpusProposalProvider:
    """Known-legal reaction corpus baseline for proposal-pool coverage audits.

    This is intentionally a constrained proposer: it only emits reactant sides
    that appear in a canonical reaction corpus. It is not a route retriever or a
    learned ranker; it measures whether a legal candidate pool can cover the
    requested product neighborhood.
    """

    provider_name = "legal_corpus"

    def __init__(
        self,
        corpus_paths: list[Path | str] | Path | str,
        *,
        max_index_rows: int | None = None,
        similarity_floor: float = 0.0,
        candidate_pool_size: int = 256,
        validity_config: ProposalValidityConfig | None = None,
        index_cache_path: Path | str | None = None,
    ):
        if isinstance(corpus_paths, (str, Path)):
            corpus_paths = [corpus_paths]
        self.corpus_paths = [Path(path) for path in corpus_paths]
        self.max_index_rows = int(max_index_rows) if max_index_rows is not None else None
        self.similarity_floor = max(0.0, min(1.0, float(similarity_floor)))
        self.candidate_pool_size = max(1, int(candidate_pool_size or 1))
        self.validity_config = validity_config or ProposalValidityConfig()
        self.index_cache_path = _resolve_optional_path(index_cache_path)
        self._loaded = False
        self._entries: list[dict[str, Any]] = []
        self._fps: list[Any] = []
        self._exact_by_product: dict[str, list[int]] = {}
        self._loaded_from_cache = False
        self._cache_status = "disabled" if self.index_cache_path is None else "not_loaded"
        self.load_error = ""
        self.last_diagnostics = ProposalDiagnostics(provider_name=self.provider_name)

    @property
    def available(self) -> bool:
        return any(path.exists() for path in self.corpus_paths)

    def propose(self, request: ProposalRequest | str, *args: Any, **kwargs: Any) -> list[CascadeAction]:
        leaf = request.leaf_smiles if isinstance(request, ProposalRequest) else str(request or "")
        top_k = int((request.top_k if isinstance(request, ProposalRequest) else kwargs.get("top_k")) or 10)
        if not leaf:
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                metadata={"empty_reason": "missing_leaf"},
            )
            return []
        try:
            self._ensure_index()
            rows = self.predict(leaf, top_k=top_k)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            self.last_diagnostics = ProposalDiagnostics(
                provider_name=self.provider_name,
                requested=top_k,
                returned=0,
                failures=[{"category": "legal_corpus_failed", "message": self.load_error}],
                metadata={"corpus_paths": [str(path) for path in self.corpus_paths]},
            )
            return []
        actions = [coerce_to_cascade_action(row, target_leaf=leaf, source=self.provider_name) for row in rows[:top_k]]
        self.last_diagnostics = ProposalDiagnostics(
            provider_name=self.provider_name,
            requested=top_k,
            returned=len(actions),
            metadata={
                "corpus_paths": [str(path) for path in self.corpus_paths],
                "indexed_reactions": len(self._entries),
                "similarity_floor": self.similarity_floor,
                "candidate_pool_size": self.candidate_pool_size,
                "max_index_rows": self.max_index_rows,
                "index_cache_path": str(self.index_cache_path) if self.index_cache_path else None,
                "loaded_from_cache": self._loaded_from_cache,
                "cache_status": self._cache_status,
                "exact_product_candidates": len(self._exact_by_product.get(canonical_smiles(leaf), [])),
                "validity_filter": asdict(self.validity_config),
            },
        )
        return actions

    def predict(self, product_smiles: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_index()
        product_key = canonical_smiles(product_smiles)
        if not product_key:
            return []
        exact_indices = list(self._exact_by_product.get(product_key, []))
        candidate_indices: list[tuple[int, float, str]] = [(idx, 1.0, "exact_product") for idx in exact_indices]
        if len(candidate_indices) < max(1, int(top_k or 1)):
            candidate_indices.extend(self._nearest_indices(product_key, limit=max(self.candidate_pool_size, int(top_k or 1))))

        rows: list[dict[str, Any]] = []
        seen_sides: set[tuple[str, ...]] = set()
        for idx, similarity, match_type in candidate_indices:
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            reactants = [str(smi) for smi in entry.get("reactants") or [] if smi]
            side_text = ".".join(reactants)
            report = filter_reactant_predictions(
                [side_text],
                product_smiles=product_smiles,
                config=self.validity_config,
            )
            if not report.kept:
                continue
            side_key = canonical_side(report.kept[0])
            if not side_key or side_key in seen_sides:
                continue
            seen_sides.add(side_key)
            if similarity < self.similarity_floor:
                continue
            reactants = list(side_key)
            score = float(similarity)
            rows.append(
                {
                    "product_smiles": product_smiles,
                    "reactant_smiles": reactants,
                    "rxn_smiles": ".".join(reactants) + f">>{product_smiles}",
                    "reaction_smiles": ".".join(reactants) + f">>{product_smiles}",
                    "source": self.provider_name,
                    "source_model": self.provider_name,
                    "proposal_type": "known_legal_corpus",
                    "type": "known_legal_corpus_candidate",
                    "score": score,
                    "rank": len(rows) + 1,
                    "main_reactant": _largest_smiles(reactants),
                    "aux_reactants": _aux_reactants(reactants),
                    "corpus_product_smiles": entry.get("product"),
                    "corpus_reaction_smiles": entry.get("reaction_smiles"),
                    "corpus_canonical_reaction": entry.get("canonical_reaction"),
                    "corpus_source": entry.get("source"),
                    "corpus_source_row_id": entry.get("source_row_id"),
                    "corpus_path": entry.get("corpus_path"),
                    "corpus_line_no": entry.get("line_no"),
                    "product_similarity": round(score, 6),
                    "match_type": match_type,
                    "contract": (
                        "Known-legal corpus proposal baseline; no expert label and no route-quality claim."
                    ),
                }
            )
            if len(rows) >= max(1, int(top_k or 1)):
                break
        return rows

    def _ensure_index(self) -> None:
        if self._loaded:
            if self.load_error:
                raise RuntimeError(self.load_error)
            return
        self._loaded = True
        if self._load_index_cache():
            return
        count = 0
        for path in self.corpus_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    if self.max_index_rows is not None and count >= self.max_index_rows:
                        break
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entry = self._entry_from_metadata(raw, path=path, line_no=line_no)
                    if entry is None:
                        continue
                    fp = _morgan_fp(entry["product"])
                    if fp is None:
                        continue
                    idx = len(self._entries)
                    self._entries.append(entry)
                    self._fps.append(fp)
                    self._exact_by_product.setdefault(entry["product"], []).append(idx)
                    count += 1
                if self.max_index_rows is not None and count >= self.max_index_rows:
                    break
        if not self._entries:
            self.load_error = "no valid legal-corpus reactions indexed"
            raise RuntimeError(self.load_error)
        self._write_index_cache()

    def _entry_from_metadata(self, raw: dict[str, Any], *, path: Path, line_no: int) -> dict[str, Any] | None:
        product = canonical_smiles(raw.get("product") or raw.get("product_smiles") or raw.get("target_smiles"))
        reactants = list(canonical_side(".".join(str(smi) for smi in raw.get("reactants") or raw.get("reactant_smiles") or [])))
        if not product or not reactants:
            return None
        reaction = raw.get("canonical_reaction") or raw.get("reaction_smiles") or (".".join(reactants) + f">>{product}")
        key = canonical_reaction(reaction) or (".".join(reactants) + f">>{product}")
        return {
            "product": product,
            "reactants": reactants,
            "reaction_smiles": ".".join(reactants) + f">>{product}",
            "canonical_reaction": key,
            "source": raw.get("source"),
            "source_row_id": raw.get("source_row_id"),
            "corpus_path": str(path),
            "line_no": line_no,
        }

    def _nearest_indices(self, product_key: str, *, limit: int) -> list[tuple[int, float, str]]:
        qfp = _morgan_fp(product_key)
        if qfp is None or not self._fps:
            return []
        sims = DataStructs.BulkTanimotoSimilarity(qfp, self._fps)
        order = np.argsort(np.asarray(sims, dtype=float))[-max(1, int(limit)) :][::-1]
        return [(int(idx), float(sims[int(idx)]), "nearest_product") for idx in order]

    def _load_index_cache(self) -> bool:
        if self.index_cache_path is None or not self.index_cache_path.exists():
            self._cache_status = "miss" if self.index_cache_path is not None else "disabled"
            return False
        try:
            with self.index_cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            if not isinstance(payload, dict):
                self._cache_status = "invalid_payload"
                return False
            if payload.get("schema_version") != "legal_corpus_index.v1":
                self._cache_status = "schema_mismatch"
                return False
            if payload.get("source_signature") != self._source_signature():
                self._cache_status = "signature_mismatch"
                return False
            entries = payload.get("entries")
            fps = payload.get("fps")
            exact = payload.get("exact_by_product")
            if not isinstance(entries, list) or not isinstance(fps, list) or not isinstance(exact, dict):
                self._cache_status = "invalid_payload"
                return False
            self._entries = entries
            self._fps = fps
            self._exact_by_product = {str(k): [int(v) for v in values] for k, values in exact.items()}
            self._loaded_from_cache = True
            self._cache_status = "hit"
            return bool(self._entries and self._fps)
        except Exception:
            self._cache_status = "load_failed"
            return False

    def _write_index_cache(self) -> None:
        if self.index_cache_path is None:
            return
        if self.index_cache_path.exists() and self._cache_status in {
            "signature_mismatch",
            "schema_mismatch",
            "invalid_payload",
            "load_failed",
        }:
            return
        try:
            self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "legal_corpus_index.v1",
                "source_signature": self._source_signature(),
                "entries": self._entries,
                "fps": self._fps,
                "exact_by_product": self._exact_by_product,
            }
            with self.index_cache_path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self._cache_status = "written"
        except Exception:
            self._cache_status = "write_failed"
            return

    def _source_signature(self) -> list[dict[str, Any]]:
        signature = []
        for path in self.corpus_paths:
            try:
                stat = path.stat()
                signature.append(
                    {
                        "path": str(path),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
            except OSError:
                signature.append({"path": str(path), "missing": True})
        signature.append({"max_index_rows": self.max_index_rows})
        return signature


class RetroKNNProposalProvider(LegalCorpusProposalProvider):
    """Fast nearest-neighbor proposal provider over known real reactions.

    The implementation intentionally reuses the legal-corpus index contract:
    every emitted reactant side appears in the supplied reaction corpus. The
    RetroKNN name marks the retrieval role in diagnostics and downstream
    proposal-pool audits.
    """

    provider_name = "retroknn"

    def predict(self, product_smiles: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        rows = super().predict(product_smiles, top_k=top_k)
        for row in rows:
            row["source"] = self.provider_name
            row["source_model"] = self.provider_name
            row["proposal_type"] = "retroknn_retrieval"
            row["type"] = "retroknn_known_reaction_retrieval"
            row["retrieval_similarity"] = row.get("product_similarity")
            row["retrieval_match_type"] = row.get("match_type")
            row["contract"] = (
                "RetroKNN retrieval proposal over known real reactions; "
                "nearest-neighbor support only, not a route-quality claim."
            )
        return rows


def coerce_to_cascade_action(value: CascadeAction | StepAnnotation | dict[str, Any], *, target_leaf: str = "", source: str = "") -> CascadeAction:
    if isinstance(value, CascadeAction):
        return value
    if isinstance(value, StepAnnotation):
        return CascadeAction(
            CascadeActionType.RETROSYNTHETIC_STEP,
            target_leaf=target_leaf or value.product_smiles,
            step=value,
            source=source or value.source_model,
        )
    if isinstance(value, dict):
        step = StepAnnotation(
            product_smiles=str(value.get("product_smiles") or value.get("product") or target_leaf or ""),
            reactant_smiles=[str(smi) for smi in value.get("reactant_smiles") or value.get("reactants") or []],
            rxn_smiles=str(value.get("rxn_smiles") or value.get("reaction_smiles") or ""),
            source_model=str(value.get("source_model") or value.get("source") or source or ""),
            score=_safe_float(value.get("score")),
            reaction_type=str(value.get("reaction_type") or value.get("type") or ""),
            ec_numbers=[str(x) for x in value.get("ec_numbers") or ([value.get("ec")] if value.get("ec") else [])],
            condition=_condition_from_mapping(value),
            cofactor_requirements={str(k): float(v or 0.0) for k, v in (value.get("cofactor_requirements") or {}).items()},
            cofactor_regenerations={str(k): float(v or 0.0) for k, v in (value.get("cofactor_regenerations") or {}).items()},
            stock_status={str(k): v for k, v in (value.get("stock_status") or {}).items()},
            raw_metadata={k: v for k, v in value.items() if k not in {"product_smiles", "reactant_smiles"}},
        )
        return CascadeAction(
            CascadeActionType.RETROSYNTHETIC_STEP,
            target_leaf=target_leaf or step.product_smiles,
            step=step,
            source=source or step.source_model,
        )
    raise TypeError(f"unsupported cascade proposal type: {type(value).__name__}")


def route_step_candidate_to_action(step: RouteStepCandidate, *, provider_name: str) -> CascadeAction:
    condition = ConditionEnvelope.from_backend_prediction((step.condition_predictions or [None])[0])
    ec_numbers = [
        str(row.get("ec_number"))
        for row in step.enzyme_ec_annotations
        if row.get("ec_number")
    ]
    evidence_confidence = _best_confidence(step.enzyme_ec_annotations)
    enzyme_module = None
    if ec_numbers:
        enzyme_module = CascadeModule(
            name="ChemEnzy enzyme assignment",
            module_kind="enzyme",
            ec_numbers=ec_numbers,
            condition_envelope=condition,
            evidence_confidence=evidence_confidence,
            source_model=provider_name,
            raw_metadata={"enzyme_ec_annotations": step.enzyme_ec_annotations},
        )
    annotation = StepAnnotation(
        product_smiles=step.product_smiles,
        reactant_smiles=list(step.reactant_smiles),
        rxn_smiles=step.rxn_smiles,
        source_model=step.source_model or provider_name,
        score=step.score,
        reaction_type="enzymatic" if ec_numbers else "",
        ec_numbers=ec_numbers,
        condition=condition,
        enzyme_module=enzyme_module,
        evidence_confidence=evidence_confidence,
        stock_status=dict(step.stock_status),
        raw_metadata=step.raw_backend_metadata,
    )
    return CascadeAction(
        CascadeActionType.RETROSYNTHETIC_STEP,
        target_leaf=step.product_smiles,
        step=annotation,
        source=provider_name,
    )


def _candidate_action_to_cascade_action(action: Any, *, leaf: str) -> CascadeAction:
    condition = ConditionEnvelope.from_point(
        temperature_c=getattr(action, "T", None),
        ph=getattr(action, "pH", None),
        solvent=getattr(action, "solvent", "") or "",
        catalyst=getattr(action, "catalyst", "") or "",
        confidence=getattr(action, "raw_score", None),
    )
    ec = str(getattr(action, "ec", "") or "")
    enzyme_module = None
    if ec:
        enzyme_module = CascadeModule(
            name="route-tree enzyme module",
            module_kind="enzyme",
            reaction_type=str(getattr(action, "reaction_type", "") or ""),
            ec_numbers=[ec],
            condition_envelope=condition,
            evidence_confidence=getattr(action, "raw_score", None),
            source_model=str(getattr(action, "source", "") or "route_tree"),
        )
    step = StepAnnotation(
        product_smiles=str(getattr(action, "product", "") or leaf),
        reactant_smiles=[str(smi) for smi in getattr(action, "reactants", ())],
        rxn_smiles=str(getattr(action, "rxn_smiles", "") or ""),
        source_model=str(getattr(action, "source", "") or "route_tree"),
        score=getattr(action, "raw_score", None),
        reaction_type=str(getattr(action, "reaction_type", "") or ""),
        ec_numbers=[ec] if ec else [],
        condition=condition,
        enzyme_module=enzyme_module,
        raw_metadata=getattr(action, "metadata", {}) or {},
    )
    return CascadeAction(CascadeActionType.RETROSYNTHETIC_STEP, target_leaf=leaf, step=step, source=step.source_model)


def _condition_from_mapping(value: dict[str, Any]) -> ConditionEnvelope | None:
    if isinstance(value.get("condition"), ConditionEnvelope):
        return value["condition"]
    if isinstance(value.get("condition"), dict):
        return ConditionEnvelope.from_backend_prediction(value["condition"])
    if any(key in value for key in ("temperature_c", "Temperature", "T", "ph", "pH", "solvent", "Solvent")):
        return ConditionEnvelope.from_backend_prediction(value)
    return None


_CHEM_TEMPLATE_ONE_STEP_CACHE: dict[tuple[Any, ...], Any] = {}
_CONDITION_PREDICTOR_CACHE: dict[tuple[str, str], Any] = {}
_TORCH_SIDECARS_AVAILABLE: bool | None = None


def _torch_sidecars_available() -> bool:
    global _TORCH_SIDECARS_AVAILABLE
    if os.environ.get("AUTOPLANNER_DISABLE_TORCH_SIDECARS", "").lower() in {"1", "true", "yes"}:
        return False
    if _TORCH_SIDECARS_AVAILABLE is not None:
        return _TORCH_SIDECARS_AVAILABLE
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        _TORCH_SIDECARS_AVAILABLE = result.returncode == 0
    except Exception:
        _TORCH_SIDECARS_AVAILABLE = False
    return bool(_TORCH_SIDECARS_AVAILABLE)


class _LocalRCRConditionPredictor:
    model_name = "rcr"

    def __init__(self, *, vendor_root: Path | str):
        self.vendor_root = Path(vendor_root)
        self._model: Any | None = None
        self.timeout_s = float(os.environ.get("AUTOPLANNER_RCR_CONDITION_TIMEOUT_S") or 20.0)
        self.inprocess = os.environ.get("AUTOPLANNER_RCR_CONDITION_INPROCESS", "").lower() in {"1", "true", "yes"}

    @property
    def data_dir(self) -> Path:
        return self.vendor_root / "retro_planner" / "packages" / "condition_predictor" / "condition_predictor" / "data"

    def available(self) -> bool:
        required = ["dict_weights.npy", "c1_dict.pickle", "s1_dict.pickle", "s2_dict.pickle", "r1_dict.pickle", "r2_dict.pickle"]
        return all((self.data_dir / name).exists() for name in required)

    def predict(self, rxn_smiles: str, *, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.inprocess:
            return self._predict_subprocess(rxn_smiles, top_k=top_k)
        model = self._ensure_model()
        combos, scores = model.get_n_conditions(rxn_smiles, n=max(1, int(top_k or 1)), return_scores=True)
        return _rcr_rows_from_combos(combos, scores)

    def predict_many(self, rxn_smiles_list: list[str], *, top_k: int = 1) -> dict[str, list[dict[str, Any]]]:
        clean = [str(rxn or "") for rxn in rxn_smiles_list if str(rxn or "")]
        if not clean:
            return {}
        if not self.inprocess:
            return self._predict_many_subprocess(clean, top_k=top_k)
        model = self._ensure_model()
        out: dict[str, list[dict[str, Any]]] = {}
        for rxn_smiles in clean:
            combos, scores = model.get_n_conditions(rxn_smiles, n=max(1, int(top_k or 1)), return_scores=True)
            out[rxn_smiles] = _rcr_rows_from_combos(combos, scores)
        return out

    def _predict_subprocess(self, rxn_smiles: str, *, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.available():
            raise RuntimeError(f"RCR condition model files are missing under {self.data_dir}")
        script = r"""
import json
import sys
from pathlib import Path

payload = json.loads(sys.stdin.read())
vendor_root = Path(payload["vendor_root"])
rxn_smiles = payload["rxn_smiles"]
top_k = max(1, int(payload.get("top_k") or 1))
package_root = vendor_root / "retro_planner" / "packages" / "condition_predictor"
sys.path.insert(0, str(package_root.resolve()))
from condition_predictor.condition_model import NeuralNetContextRecommender

data_dir = package_root / "condition_predictor" / "data"
model = NeuralNetContextRecommender()
model.load_nn_model(info_path=str(data_dir.resolve()), weights_path=str((data_dir / "dict_weights.npy").resolve()))
combos, scores = model.get_n_conditions(rxn_smiles, n=top_k, return_scores=True)
rows = []
for combo, score in zip(combos or [], scores or []):
    if not isinstance(combo, (list, tuple)) or len(combo) < 4:
        continue
    rows.append({
        "Temperature": combo[0],
        "Solvent": combo[1],
        "Reagent": combo[2],
        "Catalyst": combo[3],
        "Score": float(score),
        "condition_model": "rcr",
    })
print(json.dumps(rows))
"""
        payload = {
            "vendor_root": str(self.vendor_root.resolve()),
            "rxn_smiles": rxn_smiles,
            "top_k": max(1, int(top_k or 1)),
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip().splitlines()
            message = stderr[-1] if stderr else f"returncode={result.returncode}"
            raise RuntimeError(f"RCR condition predictor subprocess failed: {message}")
        text = (result.stdout or "").strip()
        if not text:
            return []
        return json.loads(text.splitlines()[-1])

    def _predict_many_subprocess(self, rxn_smiles_list: list[str], *, top_k: int = 1) -> dict[str, list[dict[str, Any]]]:
        if not self.available():
            raise RuntimeError(f"RCR condition model files are missing under {self.data_dir}")
        script = r"""
import json
import sys
from pathlib import Path

payload = json.loads(sys.stdin.read())
vendor_root = Path(payload["vendor_root"])
rxn_smiles_list = list(payload["rxn_smiles_list"])
top_k = max(1, int(payload.get("top_k") or 1))
package_root = vendor_root / "retro_planner" / "packages" / "condition_predictor"
sys.path.insert(0, str(package_root.resolve()))
from condition_predictor.condition_model import NeuralNetContextRecommender

data_dir = package_root / "condition_predictor" / "data"
model = NeuralNetContextRecommender()
model.load_nn_model(info_path=str(data_dir.resolve()), weights_path=str((data_dir / "dict_weights.npy").resolve()))
out = {}
for rxn_smiles in rxn_smiles_list:
    combos, scores = model.get_n_conditions(rxn_smiles, n=top_k, return_scores=True)
    rows = []
    for combo, score in zip(combos or [], scores or []):
        if not isinstance(combo, (list, tuple)) or len(combo) < 4:
            continue
        rows.append({
            "Temperature": combo[0],
            "Solvent": combo[1],
            "Reagent": combo[2],
            "Catalyst": combo[3],
            "Score": float(score),
            "condition_model": "rcr",
        })
    out[rxn_smiles] = rows
print(json.dumps(out))
"""
        payload = {
            "vendor_root": str(self.vendor_root.resolve()),
            "rxn_smiles_list": rxn_smiles_list,
            "top_k": max(1, int(top_k or 1)),
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=max(self.timeout_s, self.timeout_s * len(rxn_smiles_list)),
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip().splitlines()
            message = stderr[-1] if stderr else f"returncode={result.returncode}"
            raise RuntimeError(f"RCR condition predictor batch subprocess failed: {message}")
        text = (result.stdout or "").strip()
        if not text:
            return {}
        return json.loads(text.splitlines()[-1])

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.available():
            raise RuntimeError(f"RCR condition model files are missing under {self.data_dir}")
        package_root = self.vendor_root / "retro_planner" / "packages" / "condition_predictor"
        path_text = str(package_root.resolve())
        inserted = False
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted = True
        try:
            from condition_predictor.condition_model import NeuralNetContextRecommender
        finally:
            if inserted:
                try:
                    sys.path.remove(path_text)
                except ValueError:
                    pass
        model = NeuralNetContextRecommender()
        model.load_nn_model(info_path=str(self.data_dir.resolve()), weights_path=str((self.data_dir / "dict_weights.npy").resolve()))
        self._model = model
        return model


def _cached_condition_predictor(vendor_root: Path | str, model_name: str) -> Any:
    model = str(model_name or "rcr").lower()
    key = (str(Path(vendor_root).resolve()), model)
    if key in _CONDITION_PREDICTOR_CACHE:
        return _CONDITION_PREDICTOR_CACHE[key]
    if model != "rcr":
        raise RuntimeError(f"local template condition prediction currently supports rcr, got {model_name!r}")
    predictor = _LocalRCRConditionPredictor(vendor_root=vendor_root)
    _CONDITION_PREDICTOR_CACHE[key] = predictor
    return predictor


def _call_condition_predictor(predictor: Any, rxn_smiles: str, top_k: int) -> Any:
    if hasattr(predictor, "get_n_conditions"):
        return predictor.get_n_conditions(rxn_smiles, n=max(1, int(top_k or 1)), return_scores=True)
    try:
        return predictor(rxn_smiles, max(1, int(top_k or 1)), return_scores=True)
    except TypeError:
        return predictor(rxn_smiles, top_k=max(1, int(top_k or 1)))


def _normalize_condition_prediction_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if hasattr(raw, "to_dict"):
        try:
            records = raw.to_dict(orient="records")
        except TypeError:
            records = raw.to_dict()
        return _normalize_condition_prediction_rows(records)
    if isinstance(raw, tuple) and len(raw) == 2:
        combos, scores = raw
        rows = []
        for combo, score in zip(combos or [], scores or []):
            if isinstance(combo, dict):
                row = dict(combo)
            elif isinstance(combo, (list, tuple)) and len(combo) >= 4:
                row = {
                    "Temperature": combo[0],
                    "Solvent": combo[1],
                    "Reagent": combo[2],
                    "Catalyst": combo[3],
                }
            else:
                continue
            row.setdefault("Score", score)
            rows.append(_normalize_condition_prediction_row(row))
        return rows
    if isinstance(raw, list):
        rows = []
        for item in raw:
            if isinstance(item, dict):
                rows.append(_normalize_condition_prediction_row(item))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                rows.append(
                    _normalize_condition_prediction_row(
                        {
                            "Temperature": item[0],
                            "Solvent": item[1],
                            "Reagent": item[2],
                            "Catalyst": item[3],
                        }
                    )
                )
        return rows
    if isinstance(raw, dict):
        return [_normalize_condition_prediction_row(raw)]
    return []


def _rcr_rows_from_combos(combos: Any, scores: Any) -> list[dict[str, Any]]:
    rows = []
    for combo, score in zip(combos or [], scores or []):
        if not isinstance(combo, (list, tuple)) or len(combo) < 4:
            continue
        rows.append(
            {
                "Temperature": combo[0],
                "Solvent": combo[1],
                "Reagent": combo[2],
                "Catalyst": combo[3],
                "Score": score,
                "condition_model": "rcr",
            }
        )
    return rows


def _condition_rxn_smiles_for_item(item: dict[str, Any]) -> str:
    rxn_smiles = str(item.get("rxn_smiles") or item.get("reaction_smiles") or "")
    if ">>" in rxn_smiles:
        return rxn_smiles
    reactants = [str(smi) for smi in item.get("reactant_smiles") or [] if smi]
    product = str(item.get("product_smiles") or item.get("product") or "")
    if reactants and product:
        return ".".join(reactants) + f">>{product}"
    return rxn_smiles


def _normalize_condition_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row or {})
    for key in ("Temperature", "temperature", "temperature_c", "T"):
        if key in item:
            value = _safe_float(item.get(key))
            if value is not None:
                item["Temperature"] = value
            break
    for source_key, target_key in (("solvent", "Solvent"), ("catalyst", "Catalyst"), ("reagent", "Reagent")):
        if source_key in item and target_key not in item:
            item[target_key] = item[source_key]
    for key in ("Score", "score", "confidence", "Confidence", "scores"):
        if key in item:
            value = _safe_float(item.get(key))
            if value is not None:
                item["Score"] = value
            break
    issues = list(item.get("condition_prediction_issues") or [])
    temperature = _safe_float(item.get("Temperature"))
    if temperature is not None and (temperature < -100.0 or temperature > 220.0):
        issues.append("temperature_out_of_supported_range")
    score = _safe_float(item.get("Score"))
    min_score = _condition_min_score()
    if score is not None and score < min_score:
        issues.append("low_condition_prediction_score")
    if issues:
        item["condition_prediction_issues"] = sorted(set(str(issue) for issue in issues if issue))
    return item


def _condition_prediction_row_is_supported(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    hard_issues = {"temperature_out_of_supported_range", "low_condition_prediction_score"}
    if hard_issues & set(row.get("condition_prediction_issues") or []):
        return False
    return ConditionEnvelope.from_backend_prediction(row) is not None


def _condition_min_score() -> float:
    value = _safe_float(os.environ.get("AUTOPLANNER_CONDITION_MIN_SCORE"))
    return 0.10 if value is None else max(0.0, float(value))


def _best_confidence(rows: list[dict[str, Any]]) -> float | None:
    values = [_safe_float(row.get("confidence") or row.get("Confidence")) for row in rows]
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_optional_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    out = Path(path).expanduser()
    return out if out.is_absolute() else out.resolve()


def _score_preference_artifact(artifact: Any, *, product: str, reactants: str) -> float:
    if not isinstance(artifact, dict):
        return 0.0
    try:
        return max(0.0, min(1.0, float(score_proposal_preference(artifact, product=product, reactants=reactants))))
    except Exception:
        return 0.0


def _probability_from_log_score(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    try:
        return max(0.0, min(1.0, math.exp(number)))
    except OverflowError:
        return 1.0 if number > 0 else 0.0


def _align_kept_predictions_with_scores(
    predictions: list[str],
    scores: list[float],
    kept: list[str],
) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    used: set[int] = set()
    for item in kept:
        matched_idx = None
        for idx, prediction in enumerate(predictions):
            if idx in used:
                continue
            if prediction == item:
                matched_idx = idx
                break
        if matched_idx is None:
            out.append((item, None))
            continue
        used.add(matched_idx)
        out.append((item, scores[matched_idx] if matched_idx < len(scores) else None))
    return out


def _is_self_reaction(product_smiles: str, reactants: list[str]) -> bool:
    product_key = canonical_smiles(product_smiles)
    if not product_key:
        return False
    return any(canonical_smiles(reactant) == product_key for reactant in reactants if reactant)


def _morgan_fp(smiles: str, *, n_bits: int = 2048) -> Any | None:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(n_bits))


def canonical_reaction_or_raw(rxn_smiles: str) -> str:
    try:
        return canonical_reaction(rxn_smiles) or rxn_smiles
    except Exception:
        return rxn_smiles


def _largest_smiles(smiles: list[str]) -> str:
    if not smiles:
        return ""
    return max(smiles, key=lambda smi: (len(canonical_smiles(smi) or smi), canonical_smiles(smi) or smi))


def _aux_reactants(reactants: list[str]) -> list[str]:
    main = _largest_smiles(reactants)
    main_key = canonical_smiles(main) or main
    used_main = False
    aux = []
    for smi in reactants:
        key = canonical_smiles(smi) or smi
        if key == main_key and not used_main:
            used_main = True
            continue
        aux.append(smi)
    return aux


def _retrochimera_score(metadata: dict[str, Any], *, rank: int) -> float:
    for key in ("probability", "score", "combined_score"):
        value = _safe_float(metadata.get(key))
        if value is not None:
            return float(value)
    return 1.0 / float(rank + 1)


def _jsonable_metadata(value: Any) -> Any:
    try:
        import json

        json.dumps(value)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): _jsonable_metadata(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable_metadata(v) for v in value]
        return str(value)


def time_monotonic() -> float:
    import time

    return time.monotonic()


def _load_subgoal_model_bundle(model_path: Path) -> dict[str, Any]:
    with Path(model_path).open("rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, dict):
        raise ValueError(f"invalid subgoal model bundle: {model_path}")
    return bundle


def _load_subgoal_train_evidence(program_manifest: Path) -> list[dict[str, Any]]:
    from cascade_planner.eval.train_cascade_subgoal_scorer import _evidence_items, _load_program_splits

    programs = _load_program_splits(Path(program_manifest))
    return _evidence_items(programs["train"], min_heavy_atoms=7)


def _runtime_subgoal_candidates(leaf: str, state: Any, *, max_subgoals: int, min_heavy_atoms: int) -> list[dict[str, Any]]:
    from cascade_planner.eval.train_cascade_subgoal_scorer import _fragments, _mol_props

    target = str(getattr(state, "target_smiles", "") or "")
    steps = list(getattr(state, "step_annotations", None) or getattr(state, "steps", None) or [])
    root_role = "program_target" if leaf == target and not steps else "step_product"
    rows: dict[str, dict[str, Any]] = {}

    def add(smiles: str, source: str, role: str) -> None:
        props = _mol_props(smiles)
        if not props.get("valid") or int(props.get("heavy_atoms") or 0) < min_heavy_atoms:
            return
        rows.setdefault(
            str(smiles),
            {
                "smiles": str(smiles),
                "source": source,
                "query_role": role,
                "heavy_atoms": int(props.get("heavy_atoms") or 0),
                "hetero_atoms": int(props.get("hetero_atoms") or 0),
            },
        )

    add(leaf, "leaf", root_role)
    fragment_role = "target_fragment" if root_role == "program_target" else "step_product_fragment"
    for frag in _fragments(leaf):
        add(frag, "leaf_fragment", fragment_role)
    out = sorted(rows.values(), key=lambda row: (row["source"] != "leaf", -row["heavy_atoms"], -row["hetero_atoms"], row["smiles"]))
    return out[: max(1, int(max_subgoals or 1))]


def _runtime_subgoal_query(leaf: str, state: Any, *, role: str | None = None) -> dict[str, Any]:
    from cascade_planner.eval.train_cascade_subgoal_scorer import _mol_props

    props = _mol_props(leaf)
    target = str(getattr(state, "target_smiles", "") or "")
    steps = list(getattr(state, "step_annotations", None) or getattr(state, "steps", None) or [])
    role = role or ("program_target" if leaf == target and not steps else "step_product")
    route_transforms = tuple(str(getattr(step, "reaction_type", "") or "").strip().lower().replace(" ", "_") for step in steps)
    return {
        "item_id": stable_id("runtime_subgoal_query", leaf, role),
        "role": role,
        "smiles": leaf,
        "transform": "",
        "source_step_id": "",
        "heavy_atoms": int(props.get("heavy_atoms") or 0),
        "ring_count": int(props.get("ring_count") or 0),
        "hetero_atoms": int(props.get("hetero_atoms") or 0),
        "program_id": "runtime",
        "doi": "",
        "cascade_id": "",
        "cascade_type": str((getattr(state, "raw_metadata", {}) or {}).get("cascade_type") or ""),
        "quality_tier": "",
        "route_transforms": route_transforms,
    }


def _runtime_motif_similarity(left_smiles: str, right_smiles: str) -> float:
    left = Chem.MolFromSmiles(str(left_smiles or ""))
    right = Chem.MolFromSmiles(str(right_smiles or ""))
    if left is None or right is None:
        return 0.0
    try:
        result = rdFMCS.FindMCS(
            [left, right],
            timeout=1,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        atoms = float(result.numAtoms or 0)
    except Exception:
        atoms = 0.0
    left_cov = atoms / max(float(left.GetNumHeavyAtoms()), 1.0)
    right_cov = atoms / max(float(right.GetNumHeavyAtoms()), 1.0)
    return float(0.55 * left_cov + 0.45 * right_cov)


def _existing_subgoal_hint_ids(state: Any, leaf: str) -> set[str]:
    if state is None:
        return set()
    hints = ((getattr(state, "raw_metadata", {}) or {}).get("cascade_subgoal_hints") or [])
    out = set()
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        if str(hint.get("target_leaf") or "") != str(leaf):
            continue
        hint_id = str(hint.get("subgoal_hint_id") or "")
        if hint_id:
            out.add(hint_id)
    return out
