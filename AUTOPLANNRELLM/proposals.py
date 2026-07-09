"""DeepSeek one-step candidate proposal for AUTOPLANNRELLM."""
from __future__ import annotations

import os
from typing import Any

from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.route_tree.schema import CandidateAction

from AUTOPLANNRELLM.deepseek_client import DeepSeekJSONClient


def append_llm_candidate(
    *,
    product: str,
    actions: list[CandidateAction],
    context: Any,
    diagnostics: dict[str, Any] | None = None,
    client: DeepSeekJSONClient | None = None,
) -> list[CandidateAction]:
    """Append at most one DeepSeek retrosynthetic candidate to a candidate pool."""
    if not _enabled():
        return actions
    diagnostics = diagnostics if diagnostics is not None else {}
    source_diag = diagnostics.setdefault("sources", {}).setdefault("llm_deepseek", _empty_source_diag())
    source_diag["allocated_budget"] = 1
    source_diag["calls"] = int(source_diag.get("calls") or 0) + 1
    source_diag["requested_k_total"] = int(source_diag.get("requested_k_total") or 0) + 1
    source_diag["kept_k_total"] = int(source_diag.get("kept_k_total") or 0) + 1
    try:
        suggestion = suggest_llm_candidate(product=product, context=context, client=client)
    except Exception as exc:
        source_diag["skip_reason"] = f"llm_error:{type(exc).__name__}"
        return actions
    if suggestion is None:
        source_diag["skip_reason"] = "no_llm_candidate"
        return actions
    source_diag["queried"] = True
    source_diag["raw_returned"] = int(source_diag.get("raw_returned") or 0) + 1
    candidate = CandidateAction.from_candidate(product, suggestion, rank=1, source="llm_deepseek")
    if candidate.validity_flags:
        source_diag["invalid_dropped"] = int(source_diag.get("invalid_dropped") or 0) + 1
        source_diag["skip_reason"] = "invalid_llm_candidate:" + ",".join(candidate.validity_flags)
        return actions
    if candidate.canonical_key in {action.canonical_key for action in actions}:
        source_diag["dedupe_dropped"] = int(source_diag.get("dedupe_dropped") or 0) + 1
        source_diag["skip_reason"] = "duplicate_llm_candidate"
        return actions
    source_diag["ranker_kept"] = int(source_diag.get("ranker_kept") or 0) + 1
    source_diag["kept_returned"] = int(source_diag.get("kept_returned") or 0) + 1
    source_diag["final_returned"] = int(source_diag.get("final_returned") or 0) + 1
    source_diag["skip_reason"] = ""
    return [*actions, candidate]


def suggest_llm_candidate(
    *,
    product: str,
    context: Any,
    client: DeepSeekJSONClient | None = None,
) -> dict[str, Any] | None:
    product_can = canonical_smiles(product)
    if not product_can:
        return None
    response = (client or DeepSeekJSONClient()).request_json(
        task="reaction_suggestion",
        system=_reaction_system_prompt(),
        user_payload=_reaction_payload(product=product, context=context),
        max_tokens=1000,
        temperature=float(os.environ.get("AUTOPLANNRELLM_REACTION_TEMPERATURE") or 0.1),
    )
    reactants = [str(item) for item in response.get("reactants") or [] if item]
    if not reactants:
        return None
    reactants = [canonical_smiles(item) or item for item in reactants]
    if not all(Chem.MolFromSmiles(smi) is not None for smi in reactants):
        return None
    rxn = str(response.get("reaction_smiles") or "")
    if not rxn:
        rxn = ".".join(reactants) + ">>" + product_can
    if not _reaction_product_matches(rxn, product_can):
        rxn = ".".join(reactants) + ">>" + product_can
    confidence = _safe_float(response.get("confidence"), 0.25)
    return {
        "main_reactant": _largest_smiles(reactants),
        "aux_reactants": [smi for smi in reactants if smi != _largest_smiles(reactants)],
        "rxn_smiles": rxn,
        "reaction_smiles": rxn,
        "source": "llm_deepseek",
        "score": confidence,
        "rank": 1,
        "type": str(response.get("reaction_type") or ""),
        "reaction_type": str(response.get("reaction_type") or ""),
        "ec": str(response.get("ec") or ""),
        "llm_rationale": str(response.get("rationale") or ""),
        "llm_unsupported_claims": list(response.get("unsupported_claims") or []),
        "llm_model": os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash",
    }


def _reaction_system_prompt() -> str:
    return (
        "You are a retrosynthesis proposal agent. Return only JSON. Propose "
        "one plausible one-step retrosynthetic disconnection for the product. "
        "Do not claim stock availability, yield, enzyme evidence, pH, or "
        "temperature unless the input provides it. The JSON must include "
        "reactants, reaction_smiles, reaction_type, ec, confidence, rationale, "
        "and unsupported_claims."
    )


def _reaction_payload(*, product: str, context: Any) -> dict[str, Any]:
    return {
        "product_smiles": product,
        "context": {
            "depth": getattr(context, "depth", 0),
            "reaction_type": getattr(context, "reaction_type", ""),
            "ec1": getattr(context, "ec1", 0),
            "objective": getattr(context, "objective", "balanced"),
            "route_metadata": dict(getattr(context, "route_metadata", {}) or {}),
        },
        "required_schema": {
            "reactants": [],
            "reaction_smiles": "reactants>>product",
            "reaction_type": "",
            "ec": "",
            "confidence": 0.0,
            "rationale": "",
            "unsupported_claims": [],
        },
    }


def _enabled() -> bool:
    if not _env_truthy("AUTOPLANNRELLM_ENABLE"):
        return False
    return _env_truthy_default("AUTOPLANNRELLM_ADD_LLM_CANDIDATE", True)


def _reaction_product_matches(rxn: str, product_can: str) -> bool:
    if ">>" not in rxn:
        return False
    rhs = rxn.rsplit(">>", 1)[-1]
    products = [canonical_smiles(item) or item for item in rhs.split(".") if item]
    return product_can in products


def _largest_smiles(values: list[str]) -> str:
    return max(values, key=lambda smi: Chem.MolFromSmiles(smi).GetNumHeavyAtoms() if Chem.MolFromSmiles(smi) else 0)


def _empty_source_diag() -> dict[str, Any]:
    return {
        "calls": 0,
        "queried": False,
        "allocated_budget": 0,
        "requested_k_total": 0,
        "kept_k_total": 0,
        "raw_returned": 0,
        "ranker_kept": 0,
        "ranker_dropped": 0,
        "kept_returned": 0,
        "dedupe_dropped": 0,
        "invalid_dropped": 0,
        "final_returned": 0,
        "latency_ms_total": 0.0,
        "latency_ms_max": 0.0,
        "skip_reason": "",
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).lower() in {"1", "true", "yes", "on"}
