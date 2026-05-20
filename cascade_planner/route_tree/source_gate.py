"""Source allocation contract for proposal providers.

The gate is intentionally small at runtime: trained checkpoints can replace the
fallback allocation, but proposal tools still only provide candidates. Route
selection stays in route-tree search.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


CHEMICAL_SOURCES = {
    "retrochimera",
    "chemical",
    "uspto",
    "uspto50k",
    "chem_enzy_onestep",
    "chem_enzy_graphfp",
    "chem_enzy_onmt",
}
ENZYMATIC_SOURCES = {"enzyformer", "enzexpand", "enzymatic", "enzyme"}
RHEA_RETRORULES_SOURCES = {"retrorules", "rhea", "rhea_template", "retrorules_template"}
RETRIEVAL_SOURCES = {"v3_retrieval", "retrieval"}
TEMPLATE_SOURCES = {"chemtemplates", "native_replay", "template", "template_fallback"}
SOURCE_GROUPS = ("chemical", "enzymatic", "rhea_retrorules", "retrieval", "template", "fallback")
LEGACY_SOURCE_BUDGET_GROUPS = ("chemical", "enzymatic", "rhea_retrorules", "fallback")
SOURCE_POLICY_DECISIONS = ("query", "retry_same_leaf", "switch_leaf", "relax_source_gate")
SOURCE_POLICY_BUDGET_LABELS = ("0.5x", "1x", "2x", "3x")


@dataclass(frozen=True)
class SourceAllocation:
    source_weights: dict[str, float]
    source_budgets: dict[str, int]
    fallback_budget: int
    molecule_flags: dict[str, bool] = field(default_factory=dict)
    safety_guard: str = ""
    source_group_probs: dict[str, float] = field(default_factory=dict)
    budget_multiplier: float = 1.0
    budget_multiplier_label: str = "1x"
    decision: str = "query"
    policy_confidence: float = 0.0
    policy_reason: str = ""
    policy_state_id: str = ""
    selected_source_group: str = ""
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_weights": dict(self.source_weights),
            "source_budgets": dict(self.source_budgets),
            "fallback_budget": self.fallback_budget,
            "molecule_flags": dict(self.molecule_flags),
            "safety_guard": self.safety_guard,
            "source_group_probs": dict(self.source_group_probs),
            "budget_multiplier": float(self.budget_multiplier),
            "budget_multiplier_label": self.budget_multiplier_label,
            "decision": self.decision,
            "policy_confidence": float(self.policy_confidence),
            "policy_reason": self.policy_reason,
            "policy_state_id": self.policy_state_id,
            "selected_source_group": self.selected_source_group,
            "fallback_reason": self.fallback_reason,
        }


class SourceGate:
    """Allocate per-source candidate budgets under route context."""

    def observe(
        self,
        *,
        product: str,
        context: Any | None,
        allocation: SourceAllocation,
        diagnostics: dict[str, Any],
    ) -> None:
        del product, context, allocation, diagnostics

    def allocate(
        self,
        product: str,
        *,
        context: Any | None,
        available_sources: list[str] | tuple[str, ...],
        total_budget: int,
    ) -> SourceAllocation:
        total_budget = max(1, int(total_budget or 1))
        flags = molecule_class_flags(product)
        ec1 = int(getattr(context, "ec1", 0) or 0) if context is not None else 0
        reaction_type = str(getattr(context, "reaction_type", "") or "").lower()
        route_metadata = getattr(context, "route_metadata", {}) or {}
        route_carbohydrate = bool(route_metadata.get("carbohydrate_like_route"))
        route_enzymatic = bool(route_metadata.get("enzymatic_only_route"))
        safety_guard = ""

        weights = {
            "chemical": 0.45,
            "enzymatic": 0.35,
            "rhea_retrorules": 0.15,
            "fallback": 0.05,
        }
        if ec1:
            weights.update({"chemical": 0.10, "enzymatic": 0.58, "rhea_retrorules": 0.25, "fallback": 0.07})
        if "uspto" in reaction_type or "chemical" in reaction_type:
            weights.update({"chemical": 0.65, "enzymatic": 0.15, "rhea_retrorules": 0.10, "fallback": 0.10})
        if flags["aromatic_chemical"] and not ec1:
            weights["chemical"] += 0.15
            weights["enzymatic"] = max(0.05, weights["enzymatic"] - 0.10)
        if flags["nucleotide"] or flags["peptide_like"]:
            weights["enzymatic"] += 0.10
            weights["rhea_retrorules"] += 0.05
        if ec1 and (flags["carbohydrate"] or route_carbohydrate):
            weights["chemical"] = 0.0
            weights["enzymatic"] = max(weights["enzymatic"], 0.70)
            weights["rhea_retrorules"] = max(weights["rhea_retrorules"], 0.25)
            safety_guard = "carbohydrate_ec_prefers_enzymatic_sources"
        elif ec1 and flags["large_molecule"]:
            weights["chemical"] = 0.0
            weights["enzymatic"] = max(weights["enzymatic"], 0.70)
            weights["rhea_retrorules"] = max(weights["rhea_retrorules"], 0.20)
            safety_guard = "large_ec_prefers_enzymatic_sources"
        elif ec1 and route_enzymatic:
            weights["chemical"] = min(weights["chemical"], 0.12)

        source_weights = {source: max(0.0, weights[_source_group(source)]) for source in available_sources}
        total_weight = sum(source_weights.values())
        if total_weight <= 0:
            source_weights = {source: 1.0 for source in available_sources}
            total_weight = sum(source_weights.values())
        source_weights = {source: value / total_weight for source, value in source_weights.items()}
        group_probs = _source_group_probs(source_weights)

        budgets = {source: int(round(total_budget * source_weights[source])) for source in available_sources}
        for source in available_sources:
            if source_weights[source] > 0 and budgets[source] <= 0:
                budgets[source] = 1
        while sum(budgets.values()) > total_budget and any(value > 0 for value in budgets.values()):
            source = max(budgets, key=lambda key: budgets[key])
            budgets[source] -= 1
        fallback_budget = max(1, total_budget // 4)
        return SourceAllocation(
            source_weights=source_weights,
            source_budgets=budgets,
            fallback_budget=fallback_budget,
            molecule_flags=flags,
            safety_guard=safety_guard,
            source_group_probs=group_probs,
            budget_multiplier=1.0,
            budget_multiplier_label="1x",
            decision="query",
            policy_confidence=max(source_weights.values()) if source_weights else 0.0,
            policy_reason="heuristic",
            policy_state_id=_context_state_id(context, product),
            selected_source_group=max(group_probs, key=group_probs.get) if group_probs else "",
        )


class LearnedSourceGate(SourceGate):
    """Source gate backed by a trained ``source_gate.pt`` checkpoint."""

    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint)

        ckpt = torch.load(str(self.path), map_location="cpu")
        meta = ckpt.get("metadata") or {}
        self.n_bits = int(meta.get("n_bits") or (ckpt.get("feature_schema") or {}).get("n_bits") or 128)
        self.input_dim = int(meta.get("input_dim") or (ckpt.get("feature_schema") or {}).get("input_dim") or (self.n_bits + 13))
        self.groups = list(meta.get("source_budget_groups") or SOURCE_GROUPS)
        self.model = _SourceGateMLP(self.input_dim, n_classes=len(self.groups))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def allocate(
        self,
        product: str,
        *,
        context: Any | None,
        available_sources: list[str] | tuple[str, ...],
        total_budget: int,
    ) -> SourceAllocation:
        total_budget = max(1, int(total_budget or 1))
        flags = molecule_class_flags(product)
        features = _source_gate_features(product, context=context, n_bits=self.n_bits)
        if features.shape[-1] != self.input_dim:
            features = _resize_vector(features, self.input_dim)
        with torch.no_grad():
            logits = self.model(torch.tensor(features[None, :], dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        group_weights = {self.groups[idx]: float(probs[idx]) for idx in range(min(len(self.groups), len(probs)))}
        fallback = SourceGate().allocate(
            product,
            context=context,
            available_sources=available_sources,
            total_budget=total_budget,
        )
        source_weights = {
            source: max(0.0, group_weights.get(_source_group(source), 0.0))
            for source in available_sources
        }
        if fallback.safety_guard:
            for source in available_sources:
                if _source_group(source) == "chemical":
                    source_weights[source] = 0.0
        total_weight = sum(source_weights.values())
        if total_weight <= 0:
            return fallback
        source_weights = {source: value / total_weight for source, value in source_weights.items()}
        raw_budgets = {source: total_budget * source_weights[source] for source in available_sources}
        budgets = {source: int(np.floor(raw_budgets[source])) for source in available_sources}
        remainder = max(0, total_budget - sum(budgets.values()))
        for source in sorted(available_sources, key=lambda item: raw_budgets[item] - budgets[item], reverse=True):
            if remainder <= 0:
                break
            if source_weights[source] <= 0:
                continue
            budgets[source] += 1
            remainder -= 1
        return SourceAllocation(
            source_weights=source_weights,
            source_budgets=budgets,
            fallback_budget=fallback.fallback_budget,
            molecule_flags=flags,
            safety_guard=fallback.safety_guard,
            source_group_probs=_group_probs_from_scores(source_weights),
            budget_multiplier=1.0,
            budget_multiplier_label="1x",
            decision="query",
            policy_confidence=max(source_weights.values()) if source_weights else 0.0,
            policy_reason="learned_source_gate",
            policy_state_id=_context_state_id(context, product),
            selected_source_group=max(_group_probs_from_scores(source_weights), key=_group_probs_from_scores(source_weights).get)
            if source_weights
            else "",
        )


class _SourceGateMLP(nn.Module):
    """Runtime twin of ``train_proposal_rankers.SourceGateNetwork``."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CascadeSourcePolicyGate(SourceGate):
    """Learned source/budget gate for route-tree proposal scheduling."""

    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint)
        ckpt = torch.load(str(self.path), map_location="cpu")
        meta = ckpt.get("metadata") or {}
        schema = ckpt.get("feature_schema") or meta.get("feature_schema") or {}
        self.n_bits = int(meta.get("n_bits") or schema.get("n_bits") or 64)
        self.input_dim = int(meta.get("input_dim") or schema.get("input_dim") or schema.get("feature_dim") or 0)
        if self.input_dim <= 0:
            raise ValueError(f"invalid cascade source policy checkpoint feature_dim: {checkpoint}")
        self.groups = list(meta.get("source_budget_groups") or schema.get("source_budget_groups") or SOURCE_GROUPS)
        self.budget_labels = list(meta.get("budget_labels") or SOURCE_POLICY_BUDGET_LABELS)
        self.decision_labels = list(meta.get("decision_labels") or SOURCE_POLICY_DECISIONS)
        self.failure_labels = list(meta.get("failure_labels") or (
            "source_not_queried",
            "queried_budget_too_small",
            "provider_missing",
            "stock_dead_end",
        ))
        self.min_confidence = float(meta.get("min_confidence") or 0.45)
        self.model = _CascadeSourcePolicyMLP(
            self.input_dim,
            n_groups=len(self.groups),
            n_budgets=len(self.budget_labels),
            n_decisions=len(self.decision_labels),
            n_failures=len(self.failure_labels),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._history: dict[str, dict[str, float]] = defaultdict(_source_policy_history_template)

    def observe(
        self,
        *,
        product: str,
        context: Any | None,
        allocation: SourceAllocation,
        diagnostics: dict[str, Any],
    ) -> None:
        del product, context, allocation
        for source, row in (diagnostics.get("sources") or {}).items():
            history = self._history[str(source)]
            history["calls"] += float(row.get("calls") or 0)
            history["queried"] += float(bool(row.get("queried")))
            history["requested_k_total"] += float(row.get("requested_k_total") or 0)
            history["raw_returned"] += float(row.get("raw_returned") or 0)
            history["final_returned"] += float(row.get("final_returned") or 0)
            history["latency_ms_total"] += float(row.get("latency_ms_total") or 0.0)
            history["allocated_budget"] += float(row.get("allocated_budget") or 0)
            history["useful_hits"] += float(bool(row.get("final_returned") or 0))

    def allocate(
        self,
        product: str,
        *,
        context: Any | None,
        available_sources: list[str] | tuple[str, ...],
        total_budget: int,
    ) -> SourceAllocation:
        fallback = SourceGate().allocate(
            product,
            context=context,
            available_sources=available_sources,
            total_budget=total_budget,
        )
        total_budget = max(1, int(total_budget or 1))
        if not available_sources:
            return fallback
        try:
            source_rows = []
            for source in available_sources:
                features = _cascade_source_policy_features(
                    product,
                    context=context,
                    source=source,
                    source_stats=self._history.get(source),
                    n_bits=self.n_bits,
                    total_budget=total_budget,
                )
                if features.shape[-1] != self.input_dim:
                    features = _resize_vector(features, self.input_dim)
                with torch.no_grad():
                    out = self.model(torch.tensor(features[None, :], dtype=torch.float32))
                group_probs = torch.softmax(out["group_logits"], dim=-1)[0].cpu().numpy().tolist()
                budget_probs = torch.softmax(out["budget_logits"], dim=-1)[0].cpu().numpy().tolist()
                decision_probs = torch.softmax(out["decision_logits"], dim=-1)[0].cpu().numpy().tolist()
                failure_probs = torch.sigmoid(out["failure_logits"])[0].cpu().numpy().tolist()
                utility = float(torch.sigmoid(out["utility_logit"])[0].item())
                group = _source_policy_group(source)
                score = utility * float(group_probs[self.groups.index(group)] if group in self.groups else max(group_probs or [0.0]))
                source_rows.append(
                    {
                        "source": source,
                        "group": group,
                        "features": features,
                        "group_probs": group_probs,
                        "budget_probs": budget_probs,
                        "decision_probs": decision_probs,
                        "failure_probs": failure_probs,
                        "utility": utility,
                        "score": score,
                    }
                )
        except Exception:
            return fallback

        if not source_rows:
            return fallback

        top_row = max(source_rows, key=lambda row: row["score"])
        confidence = float(top_row["score"])
        if not np.isfinite(confidence) or confidence < self.min_confidence:
            return fallback

        decision_idx = int(np.argmax(top_row["decision_probs"])) if top_row["decision_probs"] else 0
        decision = _safe_choice(self.decision_labels, decision_idx, default="query")
        budget_idx = int(np.argmax(top_row["budget_probs"])) if top_row["budget_probs"] else 1
        budget_multiplier = _budget_multiplier_from_label(
            _safe_choice(self.budget_labels, budget_idx, default="1x")
        )
        if decision == "retry_same_leaf":
            budget_multiplier = max(budget_multiplier, 1.0)
        elif decision == "switch_leaf":
            budget_multiplier = min(budget_multiplier, 0.5)
        elif decision == "relax_source_gate":
            budget_multiplier = max(budget_multiplier, 2.0)

        desired_total = _cap_source_policy_budget(
            int(round(total_budget * budget_multiplier)),
            context=context,
            available_sources=available_sources,
        )
        fallback_budget = _cascade_source_policy_fallback_budget(desired_total, decision)
        if desired_total <= fallback_budget:
            fallback_budget = 0
        primary_budget = max(1, desired_total - fallback_budget)
        source_weights = {
            row["source"]: max(0.0, float(row["score"]))
            for row in source_rows
        }
        if fallback.safety_guard:
            for source in list(source_weights):
                if _source_group(source) == "chemical":
                    source_weights[source] = 0.0
        total_weight = sum(source_weights.values())
        if total_weight <= 0:
            return fallback
        source_weights = {source: value / total_weight for source, value in source_weights.items()}
        budgets = _allocate_budget_by_weight(source_weights, primary_budget, available_sources=available_sources)
        if sum(budgets.values()) > desired_total:
            budgets = _trim_budget_map(budgets, desired_total)
        group_probs = _aggregate_group_probs(source_rows, groups=self.groups)
        return SourceAllocation(
            source_weights=source_weights,
            source_budgets=budgets,
            fallback_budget=max(0, desired_total - sum(budgets.values())),
            molecule_flags=dict(fallback.molecule_flags),
            safety_guard=fallback.safety_guard,
            source_group_probs=group_probs,
            budget_multiplier=float(budget_multiplier),
            budget_multiplier_label=_budget_label_from_multiplier(budget_multiplier),
            decision=decision,
            policy_confidence=confidence,
            policy_reason=f"learned:{top_row['source']}",
            policy_state_id=_context_state_id(context, product),
            selected_source_group=str(top_row.get("group") or ""),
        )


class _CascadeSourcePolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        n_groups: int,
        n_budgets: int,
        n_decisions: int,
        n_failures: int,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.group_head = nn.Linear(128, n_groups)
        self.budget_head = nn.Linear(128, n_budgets)
        self.decision_head = nn.Linear(128, n_decisions)
        self.failure_head = nn.Linear(128, n_failures)
        self.utility_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "group_logits": self.group_head(h),
            "budget_logits": self.budget_head(h),
            "decision_logits": self.decision_head(h),
            "failure_logits": self.failure_head(h),
            "utility_logit": self.utility_head(h).squeeze(-1),
        }


_SOURCE_GATE_CACHE: dict[str, SourceGate] = {}


def default_source_gate() -> SourceGate:
    reservoir_path = os.environ.get("AUTOPLANNER_RESERVOIR_DISTILLED_CONTROLLER")
    if reservoir_path:
        cascade_path = os.environ.get("AUTOPLANNER_CASCADE_SOURCE_POLICY") or ""
        legacy_path = os.environ.get("AUTOPLANNER_SOURCE_GATE") or ""
        key = f"reservoir:{reservoir_path}::cascade={cascade_path}::legacy={legacy_path}"
        cached = _SOURCE_GATE_CACHE.get(key)
        if cached is not None:
            return cached
        delegate = _default_source_gate_without_reservoir()
        try:
            from cascade_planner.route_tree.reservoir_distilled import ReservoirDistilledControllerRuntime

            gate: SourceGate = ReservoirDistilledControllerRuntime(reservoir_path, fallback_source_gate=delegate)
        except Exception as exc:
            from cascade_planner.route_tree.reservoir_distilled import UnavailableReservoirSourceGate

            gate = UnavailableReservoirSourceGate(f"{type(exc).__name__}:load_failed", fallback_source_gate=delegate)
        _SOURCE_GATE_CACHE[key] = gate
        return gate

    return _default_source_gate_without_reservoir()


def _default_source_gate_without_reservoir() -> SourceGate:
    cascade_path = os.environ.get("AUTOPLANNER_CASCADE_SOURCE_POLICY")
    if cascade_path:
        key = f"cascade:{cascade_path}"
        cached = _SOURCE_GATE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            gate: SourceGate = CascadeSourcePolicyGate(cascade_path)
        except Exception:
            gate = SourceGate()
        _SOURCE_GATE_CACHE[key] = gate
        return gate

    path = os.environ.get("AUTOPLANNER_SOURCE_GATE")
    if not path:
        return SourceGate()
    key = f"legacy:{path}"
    cached = _SOURCE_GATE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        gate = LearnedSourceGate(path)
    except Exception:
        gate = SourceGate()
    _SOURCE_GATE_CACHE[key] = gate
    return gate


def molecule_class_flags(smiles: str | None) -> dict[str, bool]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {
            "nucleotide": False,
            "carbohydrate": False,
            "peptide_like": False,
            "small_organic": False,
            "aromatic_chemical": False,
            "large_molecule": False,
        }
    heavy = mol.GetNumHeavyAtoms()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    oxygen = symbols.count("O")
    nitrogen = symbols.count("N")
    phosphorus = symbols.count("P")
    aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    carbohydrate = oxygen >= 4 and oxygen / max(heavy, 1) >= 0.35 and set(symbols).issubset({"C", "O"})
    peptide_like = bool(mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)N")))
    nucleotide = phosphorus > 0 and nitrogen >= 1 and oxygen >= 4
    return {
        "nucleotide": bool(nucleotide),
        "carbohydrate": bool(carbohydrate),
        "peptide_like": bool(peptide_like),
        "small_organic": bool(heavy <= 18),
        "aromatic_chemical": bool(aromatic and not nucleotide and not carbohydrate),
        "large_molecule": bool(heavy >= 30),
    }


def _source_gate_features(product: str, *, context: Any | None, n_bits: int) -> np.ndarray:
    flags = molecule_class_flags(product)
    ec = str(getattr(context, "ec1", "") or "")
    try:
        ec1 = int(ec.split(".", 1)[0])
    except (TypeError, ValueError):
        ec1 = 0
    reaction_type = str(getattr(context, "reaction_type", "") or "")
    t_value = _safe_float(getattr(context, "T", None))
    ph_value = _safe_float(getattr(context, "pH", None))
    values = [
        ec1 / 7.0 if 1 <= ec1 <= 7 else 0.0,
        float(bool(ec1)),
        _stable_bucket(reaction_type, 32) / 31.0,
        float(t_value or 0.0) / 100.0,
        float(ph_value or 0.0) / 14.0,
        float(t_value is not None),
        float(ph_value is not None),
        float(flags.get("nucleotide")),
        float(flags.get("carbohydrate")),
        float(flags.get("peptide_like")),
        float(flags.get("small_organic")),
        float(flags.get("aromatic_chemical")),
        float(flags.get("large_molecule")),
    ]
    return np.concatenate([_morgan_fp(product, n_bits=n_bits), np.asarray(values, dtype=np.float32)]).astype(np.float32)


def _morgan_fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _resize_vector(values: np.ndarray, size: int) -> np.ndarray:
    if values.shape[-1] == size:
        return values.astype(np.float32)
    if values.shape[-1] > size:
        return values[:size].astype(np.float32)
    out = np.zeros(size, dtype=np.float32)
    out[: values.shape[-1]] = values
    return out


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _stable_bucket(value: str, n: int) -> int:
    if not value:
        return 0
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(n, 1)


def _source_group(source: str) -> str:
    source = str(source or "").lower()
    if source in CHEMICAL_SOURCES:
        return "chemical"
    if source in TEMPLATE_SOURCES:
        return "chemical"
    if source in ENZYMATIC_SOURCES:
        return "enzymatic"
    if source in RETRIEVAL_SOURCES:
        return "enzymatic"
    if source in RHEA_RETRORULES_SOURCES:
        return "rhea_retrorules"
    return "fallback"


def source_group(source: str) -> str:
    return _source_group(source)


def source_policy_group(source: str) -> str:
    return _source_policy_group(source)


def _source_policy_group(source: str) -> str:
    source = str(source or "").lower()
    if source in RETRIEVAL_SOURCES:
        return "retrieval"
    if source in TEMPLATE_SOURCES:
        return "template"
    if source in CHEMICAL_SOURCES:
        return "chemical"
    if source in ENZYMATIC_SOURCES:
        return "enzymatic"
    if source in RHEA_RETRORULES_SOURCES:
        return "rhea_retrorules"
    return "fallback"


def _group_probs_from_scores(source_weights: dict[str, float]) -> dict[str, float]:
    group_scores = {group: 0.0 for group in SOURCE_GROUPS}
    for source, weight in source_weights.items():
        group_scores[_source_policy_group(source)] += max(0.0, float(weight))
    total = sum(group_scores.values())
    if total <= 0:
        return {group: 1.0 / len(SOURCE_GROUPS) for group in SOURCE_GROUPS}
    return {group: value / total for group, value in group_scores.items()}


def _source_group_probs(source_weights: dict[str, float]) -> dict[str, float]:
    return _group_probs_from_scores(source_weights)


def _source_policy_history_template() -> dict[str, float]:
    return {
        "calls": 0.0,
        "queried": 0.0,
        "requested_k_total": 0.0,
        "raw_returned": 0.0,
        "final_returned": 0.0,
        "latency_ms_total": 0.0,
        "allocated_budget": 0.0,
        "useful_hits": 0.0,
    }


def _context_state_id(context: Any | None, product: str) -> str:
    route_metadata = getattr(context, "route_metadata", {}) or {}
    state_id = str(route_metadata.get("state_id") or "")
    if state_id:
        return state_id
    return canonical_smiles(product) or str(product or "")


def _safe_choice(labels: list[str], index: int, *, default: str) -> str:
    if not labels:
        return default
    if 0 <= index < len(labels):
        return str(labels[index])
    return default


def _budget_multiplier_from_label(label: str) -> float:
    mapping = {
        "0.5x": 0.5,
        "1x": 1.0,
        "2x": 2.0,
        "3x": 3.0,
    }
    return float(mapping.get(str(label or "").lower(), 1.0))


def _budget_label_from_multiplier(multiplier: float) -> str:
    labels = [0.5, 1.0, 2.0, 3.0]
    value = min(labels, key=lambda item: abs(item - float(multiplier or 1.0)))
    return {0.5: "0.5x", 1.0: "1x", 2.0: "2x", 3.0: "3x"}[value]


def _cap_source_policy_budget(desired: int, *, context: Any | None, available_sources: list[str] | tuple[str, ...]) -> int:
    desired = max(1, int(desired or 1))
    del context, available_sources
    return min(desired, 16)


def _cascade_source_policy_fallback_budget(desired_total: int, decision: str) -> int:
    desired_total = max(1, int(desired_total or 1))
    decision = str(decision or "query")
    if desired_total <= 1:
        return 0
    if decision == "retry_same_leaf":
        return max(1, desired_total // 2)
    if decision == "switch_leaf":
        return 0
    if decision == "relax_source_gate":
        return max(1, desired_total // 3)
    return max(1, desired_total // 4)


def _allocate_budget_by_weight(
    source_weights: dict[str, float],
    budget: int,
    *,
    available_sources: list[str] | tuple[str, ...],
) -> dict[str, int]:
    budget = max(0, int(budget or 0))
    if budget <= 0:
        return {source: 0 for source in available_sources}
    weights = {source: max(0.0, float(source_weights.get(source) or 0.0)) for source in available_sources}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {source: 1.0 for source in available_sources}
        total_weight = float(len(weights))
    raw = {source: budget * (weights[source] / total_weight) for source in available_sources}
    budgets = {source: int(raw[source]) for source in available_sources}
    remainder = budget - sum(budgets.values())
    for source in sorted(available_sources, key=lambda key: raw[key] - budgets[key], reverse=True):
        if remainder <= 0:
            break
        if weights[source] <= 0:
            continue
        budgets[source] += 1
        remainder -= 1
    positive = [source for source, weight in weights.items() if weight > 0]
    if positive and budget >= len(positive):
        for source in positive:
            budgets[source] = max(1, budgets[source])
        while sum(budgets.values()) > budget:
            source = max((src for src in budgets if budgets[src] > 1), key=lambda key: budgets[key], default="")
            if not source:
                break
            budgets[source] -= 1
    return budgets


def _trim_budget_map(budgets: dict[str, int], total: int) -> dict[str, int]:
    total = max(0, int(total or 0))
    out = {source: max(0, int(value or 0)) for source, value in budgets.items()}
    while sum(out.values()) > total and any(value > 0 for value in out.values()):
        source = max(out, key=lambda key: out[key])
        out[source] -= 1
    return out


def _aggregate_group_probs(source_rows: list[dict[str, Any]], *, groups: list[str]) -> dict[str, float]:
    scores = {group: 0.0 for group in groups}
    total = 0.0
    for row in source_rows:
        score = max(0.0, float(row.get("score") or 0.0))
        group_probs = list(row.get("group_probs") or [])
        for idx, group in enumerate(groups):
            if idx < len(group_probs):
                scores[group] += score * float(group_probs[idx])
        total += score
    if total <= 0:
        return {group: 1.0 / max(len(groups), 1) for group in groups}
    norm = sum(scores.values())
    if norm <= 0:
        return {group: 1.0 / max(len(groups), 1) for group in groups}
    return {group: value / norm for group, value in scores.items()}


def _cascade_source_policy_features(
    product: str,
    *,
    context: Any | None,
    source: str,
    source_stats: dict[str, float] | None,
    n_bits: int,
    total_budget: int,
) -> np.ndarray:
    flags = molecule_class_flags(product)
    product_fp = _morgan_fp(product, n_bits=n_bits)
    source_group_vec = np.asarray([1.0 if _source_policy_group(source) == group else 0.0 for group in SOURCE_GROUPS], dtype=np.float32)
    source_bucket = _stable_bucket(source, 8)
    source_name_vec = np.asarray([1.0 if idx == source_bucket else 0.0 for idx in range(8)], dtype=np.float32)
    route_metadata = getattr(context, "route_metadata", {}) or {}
    history = dict(source_stats or {})
    history_calls = float(history.get("calls") or 0.0)
    history_queries = float(history.get("queried") or 0.0)
    history_requested = float(history.get("requested_k_total") or 0.0)
    history_raw = float(history.get("raw_returned") or 0.0)
    history_final = float(history.get("final_returned") or 0.0)
    history_latency = float(history.get("latency_ms_total") or 0.0)
    history_budget = float(history.get("allocated_budget") or 0.0)
    history_useful = float(history.get("useful_hits") or 0.0)
    history_yield = history_final / max(history_calls, 1.0)
    history_query_rate = history_queries / max(history_calls, 1.0)
    history_budget_saturation = history_final / max(history_budget, 1.0)
    ec = str(getattr(context, "ec1", "") or "")
    try:
        ec1 = int(ec.split(".", 1)[0])
    except (TypeError, ValueError):
        ec1 = 0
    reaction_type = str(getattr(context, "reaction_type", "") or "").lower()
    t_value = _safe_float(getattr(context, "T", None))
    ph_value = _safe_float(getattr(context, "pH", None))
    values = [
        float(getattr(context, "depth", 0) or 0) / 12.0,
        float(route_metadata.get("remaining_depth") or 0.0) / 12.0,
        float(route_metadata.get("open_leaf_count") or 1.0) / 8.0,
        float(route_metadata.get("nonstock_leaf_count") or 0.0) / 8.0,
        float(route_metadata.get("leaf_stock_hit") or 0.0),
        float(route_metadata.get("leaf_parent_adjacent") or 0.0),
        float(route_metadata.get("leaf_low_yield") or 0.0),
        float(route_metadata.get("leaf_heavy_atoms") or 0.0) / 64.0,
        float(route_metadata.get("target_heavy_atoms") or 0.0) / 64.0,
        float(total_budget or 0) / 16.0,
        float(ec1) / 7.0,
        float(_stable_bucket(reaction_type, 32)) / 31.0,
        float(t_value or 0.0) / 100.0,
        float(ph_value or 0.0) / 14.0,
        float(route_metadata.get("enzymatic_only_route") or 0.0),
        float(route_metadata.get("carbohydrate_like_route") or 0.0),
        float(flags["nucleotide"]),
        float(flags["carbohydrate"]),
        float(flags["peptide_like"]),
        float(flags["small_organic"]),
        float(flags["aromatic_chemical"]),
        float(flags["large_molecule"]),
        history_calls / 8.0,
        history_query_rate,
        history_requested / max(total_budget, 1),
        history_raw / max(total_budget, 1),
        history_final / max(total_budget, 1),
        history_budget / max(total_budget, 1),
        history_budget_saturation,
        history_yield,
        history_useful / max(history_calls, 1.0),
        min(history_latency / max(history_calls, 1.0), 1000.0) / 1000.0,
        float(route_metadata.get("state_depth") or getattr(context, "depth", 0) or 0) / 12.0,
        float(bool(route_metadata.get("state_id"))),
    ]
    return np.concatenate([product_fp, source_group_vec, source_name_vec, np.asarray(values, dtype=np.float32)]).astype(np.float32)


def source_policy_feature_vector(
    product: str,
    *,
    context: Any | None,
    source: str,
    source_stats: dict[str, float] | None = None,
    n_bits: int = 64,
    total_budget: int = 0,
) -> np.ndarray:
    return _cascade_source_policy_features(
        product,
        context=context,
        source=source,
        source_stats=source_stats,
        n_bits=n_bits,
        total_budget=total_budget,
    )
