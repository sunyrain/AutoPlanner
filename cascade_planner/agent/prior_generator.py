"""Strategic prior generation for AutoPlanner.

The deterministic fallback is intentionally conservative. Optional DeepSeek
usage is supported through an OpenAI-compatible chat endpoint, but only when the
caller provides an API key via environment or argument.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from AUTOPLANNRELLM.deepseek_client import is_placeholder_deepseek_key, normalize_deepseek_key_value
from rdkit import Chem

from cascade_planner.agent.schemas import (
    ConditionRisk,
    EnzymePrior,
    ReactionTypePrior,
    RouteModePrior,
    StrategicPrior,
)


def _has_substructure(mol, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    return bool(patt is not None and mol is not None and mol.HasSubstructMatch(patt))


def deterministic_prior(target_smiles: str) -> StrategicPrior:
    """Generate a transparent chemistry-prior fallback from simple motifs."""
    mol = Chem.MolFromSmiles(target_smiles)
    prior = StrategicPrior(target_smiles=target_smiles, source="deterministic")
    if mol is None:
        prior.unsupported_claims.append("invalid_smiles")
        prior.route_mode_priors.append(RouteModePrior("unknown", 1.0, "Invalid target SMILES"))
        return prior

    has_alcohol = _has_substructure(mol, "[CX4][OX2H]")
    has_ketone = _has_substructure(mol, "[CX3](=O)[#6]")
    has_amine = _has_substructure(mol, "[NX3;H2,H1;!$(NC=O)]")
    has_ester = _has_substructure(mol, "[CX3](=O)[OX2][#6]")
    has_aromatic = _has_substructure(mol, "a")

    if has_alcohol or has_ketone or has_amine:
        prior.route_mode_priors.append(RouteModePrior(
            "chemoenzymatic_cascade", 0.65,
            "Target contains functional groups commonly addressed by oxidoreductases or transaminases.",
        ))
        prior.route_mode_priors.append(RouteModePrior(
            "enzymatic_late_stage", 0.55,
            "A late enzymatic redox or amination step may control selectivity.",
        ))
    else:
        prior.route_mode_priors.append(RouteModePrior(
            "organic_only", 0.55,
            "No obvious biocatalytic handle found by deterministic motif scan.",
        ))

    if has_alcohol:
        prior.reaction_type_priors.append(ReactionTypePrior(None, "oxidation", 0.65, "Alcohol motif present."))
        prior.enzyme_priors.append(EnzymePrior(ec1=1, cofactor="NAD(P)+", substrate_family="alcohol", weight=0.65))
    if has_ketone:
        prior.reaction_type_priors.append(ReactionTypePrior(None, "reduction", 0.65, "Carbonyl motif present."))
        prior.enzyme_priors.append(EnzymePrior(ec1=1, cofactor="NAD(P)H", substrate_family="ketone", weight=0.65))
    if has_amine:
        prior.reaction_type_priors.append(ReactionTypePrior(None, "amination", 0.55, "Amine motif present."))
        prior.enzyme_priors.append(EnzymePrior(ec1=2, cofactor="PLP", substrate_family="amine", weight=0.55))
    if has_ester:
        prior.reaction_type_priors.append(ReactionTypePrior(None, "hydrolysis", 0.45, "Ester motif present."))
        prior.enzyme_priors.append(EnzymePrior(ec1=3, substrate_family="ester", weight=0.45))
    if has_aromatic:
        prior.condition_risks.append(ConditionRisk("low_aqueous_solubility", "medium", "aromatic motif prior"))

    return prior.normalize()


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start:end + 1])


def deepseek_prior(
    target_smiles: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str = "https://api.deepseek.com/chat/completions",
    timeout_s: int = 60,
) -> StrategicPrior:
    """Call DeepSeek for a structured prior.

    The API key is never stored. Pass it via argument or DEEPSEEK_API_KEY.
    """
    key = normalize_deepseek_key_value(api_key or os.environ.get("DEEPSEEK_API_KEY"))
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    if is_placeholder_deepseek_key(key):
        raise RuntimeError("DEEPSEEK_API_KEY still contains the dotenv placeholder")
    model_name = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"

    system = (
        "You are a chemistry planning prior generator. Return only JSON. "
        "Do not invent reaction SMILES, enzyme availability, stock, yield, pH, or temperature facts. "
        "All claims must be strategic priors."
    )
    user = {
        "target_smiles": target_smiles,
        "required_schema": {
            "route_mode_priors": [{"mode": "chemoenzymatic_cascade", "weight": 0.0, "rationale": ""}],
            "reaction_type_priors": [{"slot": None, "reaction_type": "oxidation", "weight": 0.0, "rationale": ""}],
            "enzyme_priors": [{"ec1": 1, "ec2": "", "cofactor": "", "substrate_family": "", "weight": 0.0, "rationale": ""}],
            "condition_risks": [{"kind": "", "severity": "medium", "evidence": "prior_only"}],
            "unsupported_claims": []
        }
    }
    payload = json.dumps({
        "model": model_name,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        base_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())
    content = data["choices"][0]["message"]["content"]
    obj = _extract_json_object(content)
    prior = StrategicPrior(
        target_smiles=target_smiles,
        route_mode_priors=[RouteModePrior(**x) for x in obj.get("route_mode_priors", [])],
        reaction_type_priors=[ReactionTypePrior(**x) for x in obj.get("reaction_type_priors", [])],
        enzyme_priors=[EnzymePrior(**x) for x in obj.get("enzyme_priors", [])],
        condition_risks=[ConditionRisk(**x) for x in obj.get("condition_risks", [])],
        unsupported_claims=list(obj.get("unsupported_claims", [])),
        source="deepseek",
    )
    return prior.normalize()


def generate_strategic_prior(
    target_smiles: str,
    *,
    provider: str = "deterministic",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Public prior API used by planning and future benchmark code."""
    if provider == "deepseek":
        try:
            return deepseek_prior(target_smiles, api_key=api_key).to_dict()
        except Exception as exc:
            fallback = deterministic_prior(target_smiles)
            fallback.unsupported_claims.append(f"deepseek_fallback: {type(exc).__name__}")
            return fallback.to_dict()
    return deterministic_prior(target_smiles).to_dict()
