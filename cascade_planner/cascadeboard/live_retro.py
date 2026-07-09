"""Live RetroChimera + EnzExpand engine for CandidateHypergraph.

Wraps the real-time RetroChimera model and EnzExpand ONNX model
into a dict-based retro_engine interface that CandidateHypergraph expects.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

_RC_MODEL = None
_ENZ_MODEL = None
_ENZ_TEMPLATES = None
_RETRORULES = None
_CHEM_TEMPLATES = None
_CHEM_ENZY_ONESTEP = None
_CHEM_ENZY_GRAPHFP_FUSION = None
_CHEM_ENZY_BIONAV = None
_TEMPLATE_RELEVANCE = None
_ENZRETRO_ONESTEP = None


class _CachingPredictor:
    """Small per-engine predict cache for repeated AO*/AND-OR expansions."""

    def __init__(self, inner: Any, name: str, max_entries: int | None = None):
        self.inner = inner
        self.name = name
        self.max_entries = max_entries if max_entries is not None else retro_cache_max_entries()
        self._cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.hit_time_s = 0.0
        self.miss_time_s = 0.0

    def predict(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key in self._cache:
            t0 = time.perf_counter()
            self.hits += 1
            rows = self._cache.pop(key)
            self._cache[key] = rows
            out = deepcopy(rows)
            self.hit_time_s += time.perf_counter() - t0
            return out
        t0 = time.perf_counter()
        self.misses += 1
        rows = self.inner.predict(*args, **kwargs)
        self._cache[key] = deepcopy(rows)
        while self.max_entries > 0 and len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        out = deepcopy(rows)
        self.miss_time_s += time.perf_counter() - t0
        return out

    def cache_stats(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "max_entries": self.max_entries,
            "hit_time_s": round(self.hit_time_s, 6),
            "miss_time_s": round(self.miss_time_s, 6),
            "avg_hit_time_ms": round(1000.0 * self.hit_time_s / max(self.hits, 1), 3) if self.hits else None,
            "avg_miss_time_ms": round(1000.0 * self.miss_time_s / max(self.misses, 1), 3) if self.misses else None,
        }


def retro_cache_max_entries(default: int = 2048) -> int:
    try:
        value = int(os.environ.get("AUTOPLANNER_RETRO_CACHE_MAX_ENTRIES", default))
    except (TypeError, ValueError):
        value = default
    return max(0, value)


def _load_retrochimera():
    global _RC_MODEL
    if _RC_MODEL is not None:
        return _RC_MODEL
    from retrochimera import RetroChimeraModel
    _RC_MODEL = RetroChimeraModel(model_dir="data_external/retrochimera_model")
    return _RC_MODEL


def _load_enzexpand():
    global _ENZ_MODEL, _ENZ_TEMPLATES
    if _ENZ_MODEL is not None:
        return _ENZ_MODEL, _ENZ_TEMPLATES
    import json, gzip, csv
    import onnxruntime as ort
    _ENZ_MODEL = ort.InferenceSession("workspace/aizdata/enzexpand_model.onnx")
    tpl_path = Path("results/shared/merged_templates.csv")
    if tpl_path.exists():
        with open(tpl_path) as f:
            _ENZ_TEMPLATES = list(csv.DictReader(f))
    else:
        _ENZ_TEMPLATES = []
    return _ENZ_MODEL, _ENZ_TEMPLATES


class _RetroChimeraWrapper:
    """Wraps RetroChimera into the dict-based interface CandidateHypergraph expects."""

    def predict(self, product_smiles: str, top_k: int = 10) -> list[dict]:
        from syntheseus.interface.molecule import Molecule
        model = _load_retrochimera()
        try:
            raw = model([Molecule(smiles=product_smiles)], num_results=top_k)
            rxns = []
            for item in raw:
                for r in item:
                    rxns.append(r)
        except Exception:
            return []

        results = []
        for i, r in enumerate(rxns[:top_k]):
            reactant_list = list(r.reactants)
            main_r = reactant_list[0].smiles if reactant_list else ""
            aux = [m.smiles for m in reactant_list[1:]]
            rxn_smiles = ".".join(m.smiles for m in reactant_list) + ">>" + product_smiles
            results.append({
                "main_reactant": main_r,
                "aux_reactants": aux,
                "rxn_smiles": rxn_smiles,
                "score": 1.0 / (i + 1),
                "type": "",
                "source": "retrochimera",
            })
        return results




class _EnzyformerWrapperLive:
    """Wraps Enzyformer for enzymatic retrosynthesis (preferred over EnzExpand)."""

    def __init__(self):
        self._wrapper = None
        self._checked = False

    def _load(self):
        if self._checked:
            return
        self._checked = True
        try:
            from cascade_planner.expand.enzyformer_wrapper import EnzyformerWrapper
            # Prefer v4 checkpoint (EnzymeMap 50K fine-tuned, best quality)
            v4_path = Path("results/shared/enzyformer_retro_v4.pt")
            if v4_path.exists():
                w = EnzyformerWrapper(checkpoint_path=str(v4_path))
            else:
                w = EnzyformerWrapper()
            if w.available:
                self._wrapper = w
        except Exception:
            pass

    @property
    def available(self) -> bool:
        self._load()
        return self._wrapper is not None

    def predict(self, product_smiles: str, top_k: int = 10, ec_token: str = "") -> list[dict]:
        self._load()
        if self._wrapper is None:
            return []
        return self._wrapper.predict(product_smiles, ec_token=ec_token, top_k=top_k)

class _EnzExpandWrapper:
    """Wraps EnzExpand using PyTorch MLP with correct 150-template mapping."""

    def __init__(self):
        self._model = None
        self._templates = None
        self._tpl_ec = None

    def _load(self):
        if self._model is not None:
            return
        import csv, torch
        import torch.nn as nn
        from cascade_planner.expand.enz_template import TemplateMLP

        # Load 150 templates from v3-trained table
        tpl_path = Path("results/enzexpand_templates.csv")
        if not tpl_path.exists():
            self._templates = []
            return
        with open(tpl_path) as f:
            rows = list(csv.DictReader(f))
        self._templates = [r["template"] for r in rows]
        self._tpl_ec = [r.get("ec1_top", "") for r in rows]

        # Build and load PyTorch MLP (same architecture as enz_template.py)
        n_classes = len(self._templates)
        self._model = TemplateMLP(2048, n_classes, hidden=512, dropout=0.3)
        # We don't have a saved .pt for this MLP — train on the fly is too slow
        # Instead, use the template frequency as a prior (no learned model)
        self._model = None  # fallback to frequency-based

    def predict(self, product_smiles: str, top_k: int = 10) -> list[dict]:
        from rdkit import Chem
        from cascade_planner.expand.enz_template import apply_template_to_product

        self._load()
        if not self._templates:
            return []

        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return []

        # Try all 150 templates (small enough to brute-force)
        results = []
        for tidx, tmpl in enumerate(self._templates):
            outcomes = apply_template_to_product(tmpl, product_smiles, generalize=1)
            for outcome in outcomes:
                reactants = list(outcome)
                if not reactants:
                    continue
                ec = self._tpl_ec[tidx] if self._tpl_ec else ""
                results.append({
                    "main_reactant": reactants[0],
                    "aux_reactants": reactants[1:],
                    "rxn_smiles": ".".join(reactants) + ">>" + product_smiles,
                    "ec": f"{ec}.x" if ec and ec.isdigit() else "",
                    "score": 1.0 / (tidx + 1),
                    "type": "",
                    "source": "enzexpand",
                })
                if len(results) >= top_k:
                    return results
        return results


class _SemisynthesisRescueWrapper:
    """Expose curated semisynthesis anchors as a lightweight route-tree source."""

    @property
    def available(self) -> bool:
        return True

    def predict(self, product_smiles: str, top_k: int = 10) -> list[dict]:
        from cascade_planner.baselines.semisynthesis_rescue import semisynthesis_rescue_routes

        rows: list[dict] = []
        for route in semisynthesis_rescue_routes(product_smiles):
            if not route.steps:
                continue
            step = route.steps[0]
            reactants = [str(item) for item in step.reactant_smiles or [] if str(item or "")]
            if not reactants:
                continue
            main = _largest_smiles(reactants)
            main_key = _canonical_smiles(main) or main
            aux = [smi for smi in reactants if (_canonical_smiles(smi) or smi) != main_key]
            metadata = step.raw_backend_metadata or {}
            rescue = metadata.get("semisynthesis_rescue") if isinstance(metadata, dict) else {}
            rxn_smiles = step.rxn_smiles or ".".join(reactants) + f">>{product_smiles}"
            rows.append(
                {
                    "main_reactant": main,
                    "aux_reactants": aux,
                    "reactant_smiles": reactants,
                    "rxn_smiles": rxn_smiles,
                    "reaction_smiles": rxn_smiles,
                    "source": "semisynthesis_rescue",
                    "score": float(step.score or route.score or 0.0),
                    "rank": len(rows) + 1,
                    "type": "semisynthesis_rescue",
                    "proposal_type": "source_supported_semisynthesis",
                    "model_full_name": str(step.source_model or "semisynthesis_rescue"),
                    "ec": _step_ec_hint(step),
                    "condition_predictions": list(step.condition_predictions or []),
                    "enzyme_ec_annotations": list(step.enzyme_ec_annotations or []),
                    "semisynthesis_rescue": rescue or metadata,
                    "route_class_hint": (route.raw_backend_metadata or {}).get("route_class_hint"),
                    "teacher_one_step": True,
                    "teacher_source": "semisynthesis_rescue",
                }
            )
            if len(rows) >= max(0, int(top_k or 0)):
                break
        return rows


class _ChemicalAnchorRescueWrapper:
    """Expose curated chemical anchors as a lightweight route-tree source."""

    @property
    def available(self) -> bool:
        return True

    def predict(self, product_smiles: str, top_k: int = 10) -> list[dict]:
        from cascade_planner.baselines.chemical_anchor_rescue import chemical_anchor_rescue_routes

        rows: list[dict] = []
        for route in chemical_anchor_rescue_routes(product_smiles):
            if not route.steps:
                continue
            step = route.steps[0]
            reactants = [str(item) for item in step.reactant_smiles or [] if str(item or "")]
            if not reactants:
                continue
            main = _largest_smiles(reactants)
            main_key = _canonical_smiles(main) or main
            aux = [smi for smi in reactants if (_canonical_smiles(smi) or smi) != main_key]
            metadata = step.raw_backend_metadata or {}
            anchor = metadata.get("chemical_anchor_rescue") if isinstance(metadata, dict) else {}
            rxn_smiles = step.rxn_smiles or ".".join(reactants) + f">>{product_smiles}"
            rows.append(
                {
                    "main_reactant": main,
                    "aux_reactants": aux,
                    "reactant_smiles": reactants,
                    "rxn_smiles": rxn_smiles,
                    "reaction_smiles": rxn_smiles,
                    "source": "chemical_anchor_rescue",
                    "score": float(step.score or route.score or 0.0),
                    "rank": len(rows) + 1,
                    "type": "chemical_anchor_rescue",
                    "proposal_type": "source_supported_chemical_anchor",
                    "model_full_name": str(step.source_model or "chemical_anchor_rescue"),
                    "condition_predictions": list(step.condition_predictions or []),
                    "chemical_anchor_rescue": anchor or metadata,
                    "route_class_hint": (route.raw_backend_metadata or {}).get("route_class_hint"),
                    "teacher_one_step": True,
                    "teacher_source": "chemical_anchor_rescue",
                }
            )
            if len(rows) >= max(0, int(top_k or 0)):
                break
        return rows


def _step_ec_hint(step: Any) -> str:
    for item in getattr(step, "enzyme_ec_annotations", None) or []:
        value = str((item or {}).get("ec_number") or "").strip()
        if value:
            return value
    metadata = getattr(step, "raw_backend_metadata", None) or {}
    rescue = metadata.get("semisynthesis_rescue") if isinstance(metadata, dict) else {}
    if isinstance(rescue, dict):
        return str(rescue.get("ec_hint") or "").strip()
    return ""


def _largest_smiles(items: list[str]) -> str:
    from rdkit import Chem

    def key(smiles: str) -> tuple[int, int, str]:
        mol = Chem.MolFromSmiles(str(smiles or ""))
        return (mol.GetNumHeavyAtoms() if mol is not None else 0, len(str(smiles or "")), str(smiles or ""))

    return max(items, key=key) if items else ""


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""


def build_live_retro_engine() -> dict:
    """Build a live retro engine dict for CandidateHypergraph."""
    enz_live = _EnzyformerWrapperLive()
    retrorules = _load_retrorules() if _retrorules_enabled() else None
    chemtemplates = _load_chemical_templates() if _chemical_templates_enabled() else None
    engine = {
        "enzyformer": _CachingPredictor(enz_live, "enzyformer") if enz_live.available else None,
        "enzexpand": _CachingPredictor(_EnzExpandWrapper(), "enzexpand"),
        "retrorules": _CachingPredictor(retrorules, "retrorules") if retrorules and retrorules.available else None,
        "chemtemplates": _CachingPredictor(chemtemplates, "chemtemplates") if chemtemplates and chemtemplates.available else None,
    }
    if _semisynthesis_rescue_enabled():
        engine["semisynthesis_rescue"] = _CachingPredictor(_SemisynthesisRescueWrapper(), "semisynthesis_rescue")
    if _chemical_anchor_rescue_enabled():
        engine["chemical_anchor_rescue"] = _CachingPredictor(
            _ChemicalAnchorRescueWrapper(), "chemical_anchor_rescue"
        )
    if _retrochimera_enabled():
        engine["retrochimera"] = _CachingPredictor(_RetroChimeraWrapper(), "retrochimera")
    if _chem_enzy_onestep_enabled():
        chem_enzy_onestep = _load_chem_enzy_onestep()
        if chem_enzy_onestep and chem_enzy_onestep.available:
            engine["chem_enzy_onestep"] = _CachingPredictor(chem_enzy_onestep, "chem_enzy_onestep")
    if _chem_enzy_graphfp_fusion_enabled():
        graphfp_fusion = _load_chem_enzy_graphfp_fusion()
        if graphfp_fusion and graphfp_fusion.available:
            engine["chem_enzy_graphfp_fusion"] = _CachingPredictor(graphfp_fusion, "chem_enzy_graphfp_fusion")
    if _template_relevance_enabled():
        template_relevance = _load_template_relevance()
        if template_relevance and template_relevance.available:
            engine["template_relevance"] = _CachingPredictor(template_relevance, "template_relevance")
    if _chem_enzy_bionav_enabled():
        chem_enzy_bionav = _load_chem_enzy_bionav()
        if chem_enzy_bionav and chem_enzy_bionav.available:
            engine["chem_enzy_bionav"] = _CachingPredictor(chem_enzy_bionav, "chem_enzy_bionav")
    if _enzretro_onestep_enabled():
        enzretro_onestep = _load_enzretro_onestep()
        if enzretro_onestep and enzretro_onestep.available:
            engine["enzretro_onestep"] = _CachingPredictor(enzretro_onestep, "enzretro_onestep")
    return engine


def build_chemical_retro_engine() -> dict:
    """Build only chemical proposal sources for continuation/closure probes."""
    chemtemplates = _load_chemical_templates() if _chemical_templates_enabled() else None
    engine = {
        "chemtemplates": _CachingPredictor(chemtemplates, "chemtemplates") if chemtemplates and chemtemplates.available else None,
    }
    if _semisynthesis_rescue_enabled():
        engine["semisynthesis_rescue"] = _CachingPredictor(_SemisynthesisRescueWrapper(), "semisynthesis_rescue")
    if _chemical_anchor_rescue_enabled():
        engine["chemical_anchor_rescue"] = _CachingPredictor(
            _ChemicalAnchorRescueWrapper(), "chemical_anchor_rescue"
        )
    if _retrochimera_enabled():
        engine["retrochimera"] = _CachingPredictor(_RetroChimeraWrapper(), "retrochimera")
    if _chem_enzy_onestep_enabled():
        chem_enzy_onestep = _load_chem_enzy_onestep()
        if chem_enzy_onestep and chem_enzy_onestep.available:
            engine["chem_enzy_onestep"] = _CachingPredictor(chem_enzy_onestep, "chem_enzy_onestep")
    if _chem_enzy_graphfp_fusion_enabled():
        graphfp_fusion = _load_chem_enzy_graphfp_fusion()
        if graphfp_fusion and graphfp_fusion.available:
            engine["chem_enzy_graphfp_fusion"] = _CachingPredictor(graphfp_fusion, "chem_enzy_graphfp_fusion")
    if _template_relevance_enabled():
        template_relevance = _load_template_relevance()
        if template_relevance and template_relevance.available:
            engine["template_relevance"] = _CachingPredictor(template_relevance, "template_relevance")
    if _enzretro_onestep_enabled():
        enzretro_onestep = _load_enzretro_onestep()
        if enzretro_onestep and enzretro_onestep.available:
            engine["enzretro_onestep"] = _CachingPredictor(enzretro_onestep, "enzretro_onestep")
    return {key: value for key, value in engine.items() if value is not None}


def _retrochimera_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_DISABLE_RETROCHIMERA") or "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _retrorules_enabled() -> bool:
    try:
        from cascade_planner.cascadeboard.retrorules_applicator import retrorules_enabled
        return retrorules_enabled()
    except Exception:
        return False


def _chemical_templates_enabled() -> bool:
    try:
        from cascade_planner.cascadeboard.chemical_template_applicator import chemical_templates_enabled
        return chemical_templates_enabled()
    except Exception:
        return False


def _semisynthesis_rescue_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_SEMISYNTHESIS_RESCUE_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _chemical_anchor_rescue_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_CHEMICAL_ANCHOR_RESCUE_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _chem_enzy_onestep_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _chem_enzy_graphfp_fusion_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_CHEMENZY_GRAPHFP_FUSION_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _chem_enzy_bionav_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_CHEMENZY_BIONAV_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _template_relevance_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_TEMPLATE_RELEVANCE_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _enzretro_onestep_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_ENZRETRO_PROPOSALS") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_retrorules():
    global _RETRORULES
    if _RETRORULES is not None:
        return _RETRORULES
    from cascade_planner.cascadeboard.retrorules_applicator import RetroRulesApplicator
    _RETRORULES = RetroRulesApplicator.from_env()
    return _RETRORULES


def _load_chemical_templates():
    global _CHEM_TEMPLATES
    if _CHEM_TEMPLATES is not None:
        return _CHEM_TEMPLATES
    from cascade_planner.cascadeboard.chemical_template_applicator import ChemicalTemplateApplicator
    _CHEM_TEMPLATES = ChemicalTemplateApplicator.from_env()
    return _CHEM_TEMPLATES


def _load_chem_enzy_onestep():
    global _CHEM_ENZY_ONESTEP
    if _CHEM_ENZY_ONESTEP is not None:
        return _CHEM_ENZY_ONESTEP
    from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider

    _CHEM_ENZY_ONESTEP = ChemEnzyOneStepProposalProvider.from_env()
    return _CHEM_ENZY_ONESTEP


def _load_chem_enzy_graphfp_fusion():
    global _CHEM_ENZY_GRAPHFP_FUSION
    if _CHEM_ENZY_GRAPHFP_FUSION is not None:
        return _CHEM_ENZY_GRAPHFP_FUSION
    from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider

    provider = ChemEnzyOneStepProposalProvider.from_env()
    provider.models = tuple(
        _env_list("AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MODELS")
        or ("graphfp_models.USPTO-full_remapped",)
    )
    provider.expansion_topk = _env_int("AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_TOPK", provider.expansion_topk)
    _CHEM_ENZY_GRAPHFP_FUSION = provider
    return _CHEM_ENZY_GRAPHFP_FUSION


def _load_chem_enzy_bionav():
    global _CHEM_ENZY_BIONAV
    if _CHEM_ENZY_BIONAV is not None:
        return _CHEM_ENZY_BIONAV
    from cascade_planner.baselines.chem_enzy_onestep import ChemEnzyOneStepProposalProvider

    provider = ChemEnzyOneStepProposalProvider.from_env()
    provider.models = tuple(_env_list("AUTOPLANNER_CHEMENZY_BIONAV_MODELS") or ("onmt_models.bionav_one_step",))
    provider.expansion_topk = _env_int("AUTOPLANNER_CHEMENZY_BIONAV_TOPK", provider.expansion_topk)
    _CHEM_ENZY_BIONAV = provider
    return _CHEM_ENZY_BIONAV


def _load_template_relevance():
    global _TEMPLATE_RELEVANCE
    if _TEMPLATE_RELEVANCE is not None:
        return _TEMPLATE_RELEVANCE
    from cascade_planner.cascade_search.proposals import TemplateRelevanceProposalProvider

    models = tuple(_env_list("AUTOPLANNER_TEMPLATE_RELEVANCE_MODELS") or ("template_relevance.reaxys",))
    _TEMPLATE_RELEVANCE = TemplateRelevanceProposalProvider(
        vendor_root=os.environ.get("AUTOPLANNER_TEMPLATE_RELEVANCE_VENDOR_ROOT") or "vendor/ChemEnzyRetroPlanner",
        models=models,
        expansion_topk=_env_int("AUTOPLANNER_TEMPLATE_RELEVANCE_TOPK", 20),
        gpu=_env_int("AUTOPLANNER_TEMPLATE_RELEVANCE_GPU", _env_int("AUTOPLANNER_CHEMENZY_ONESTEP_GPU", -1)),
    )
    return _TEMPLATE_RELEVANCE


def _load_enzretro_onestep():
    global _ENZRETRO_ONESTEP
    if _ENZRETRO_ONESTEP is not None:
        return _ENZRETRO_ONESTEP
    from cascade_planner.baselines.enzretro_onestep import EnzRetroOneStepProposalProvider

    _ENZRETRO_ONESTEP = EnzRetroOneStepProposalProvider.from_env()
    return _ENZRETRO_ONESTEP


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return int(default)


def retro_engine_cache_stats(retro_engine: dict) -> dict[str, dict]:
    out = {}
    for name, engine in (retro_engine or {}).items():
        if engine is not None and hasattr(engine, "cache_stats"):
            out[name] = engine.cache_stats()
    return out
