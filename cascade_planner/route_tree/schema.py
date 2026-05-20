"""Data contract for route-tree search and training traces."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem

from cascade_planner.cascadeboard import CascadeBoard
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles


@dataclass
class CandidateAction:
    """Normalized retrosynthetic action proposed for one product molecule."""

    product: str
    reactants: tuple[str, ...]
    main_reactant: str
    aux_reactants: tuple[str, ...] = ()
    rxn_smiles: str = ""
    source: str = ""
    raw_score: float = 0.0
    rank: int = 0
    reaction_type: str = ""
    ec: str = ""
    catalyst: str = ""
    T: float | None = None
    pH: float | None = None
    solvent: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    validity_flags: tuple[str, ...] = ()

    @classmethod
    def from_candidate(
        cls,
        product: str,
        candidate: dict[str, Any],
        *,
        rank: int = 0,
        source: str | None = None,
    ) -> "CandidateAction":
        rxn = str(candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or "")
        main = str(candidate.get("main_reactant") or "")
        aux = tuple(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
        reactants = _candidate_reactants(candidate, rxn=rxn, main=main, aux=aux)
        if not main and reactants:
            main = _largest_smiles(list(reactants))
        if main:
            main_can = canonical_smiles(main)
            aux = tuple(smi for smi in reactants if (canonical_smiles(smi) or smi) != (main_can or main))
        flags = _validity_flags(product, rxn, reactants, main)
        score = _safe_float(candidate.get("score"), 0.0)
        evidence = candidate.get("evidence") or {}
        metadata = {
            key: value
            for key, value in candidate.items()
            if key
            not in {
                "main_reactant",
                "aux_reactants",
                "rxn_smiles",
                "reaction_smiles",
                "source",
                "score",
                "rank",
                "type",
                "reaction_type",
                "ec",
                "catalyst",
                "T",
                "pH",
                "solvent",
            }
        }
        if evidence:
            metadata.setdefault("evidence", evidence)
        source_name = str(source or candidate.get("source") or candidate.get("enzyme_source") or "unknown")
        metadata.setdefault(
            "source_provenance",
            {
                "source": source_name,
                "rank": int(candidate.get("rank") or rank or 0),
                "raw_score": score,
                "evidence_present": bool(evidence),
            },
        )
        return cls(
            product=str(product or ""),
            reactants=tuple(reactants),
            main_reactant=main,
            aux_reactants=aux,
            rxn_smiles=rxn or _reaction_from_parts(reactants, product),
            source=source_name,
            raw_score=score,
            rank=int(candidate.get("rank") or rank or 0),
            reaction_type=str(candidate.get("type") or candidate.get("reaction_type") or ""),
            ec=str(candidate.get("ec") or ""),
            catalyst=str(candidate.get("catalyst") or ""),
            T=_safe_float_or_none(candidate.get("T")),
            pH=_safe_float_or_none(candidate.get("pH")),
            solvent=str(candidate.get("solvent") or ""),
            metadata=metadata,
            validity_flags=tuple(flags),
        )

    @property
    def canonical_key(self) -> str:
        rxn = canonical_reaction(self.rxn_smiles)
        if rxn:
            return rxn
        lhs = ".".join(sorted(canonical_smiles(smi) or smi for smi in self.reactants if smi))
        rhs = canonical_smiles(self.product) or self.product
        return f"{lhs}>>{rhs}"

    def to_candidate_dict(self) -> dict[str, Any]:
        out = {
            "main_reactant": self.main_reactant,
            "aux_reactants": list(self.aux_reactants),
            "rxn_smiles": self.rxn_smiles,
            "reaction_smiles": self.rxn_smiles,
            "source": self.source,
            "score": self.raw_score,
            "rank": self.rank,
            "type": self.reaction_type,
            "reaction_type": self.reaction_type,
            "ec": self.ec,
            "catalyst": self.catalyst,
            "T": self.T,
            "pH": self.pH,
            "solvent": self.solvent,
            "validity_flags": list(self.validity_flags),
        }
        out.update(self.metadata)
        return out

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reactants"] = list(self.reactants)
        data["aux_reactants"] = list(self.aux_reactants)
        data["validity_flags"] = list(self.validity_flags)
        return data


@dataclass(frozen=True)
class RouteTreeStep:
    product: str
    action: CandidateAction
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"product": self.product, "depth": self.depth, "action": self.action.to_dict()}


@dataclass
class RouteTreeState:
    """Partial retrosynthesis tree state for neural-guided search."""

    target: str
    steps: tuple[RouteTreeStep, ...] = ()
    open_leaves: tuple[str, ...] = ()
    expanded: frozenset[str] = field(default_factory=frozenset)
    score: float = 0.0
    depth: int = 0
    objective: str = "balanced"
    constraints: dict[str, Any] = field(default_factory=dict)
    candidate_pools: dict[str, list[CandidateAction]] = field(default_factory=dict)
    search_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        target: str,
        *,
        objective: str = "balanced",
        constraints: dict[str, Any] | None = None,
    ) -> "RouteTreeState":
        return cls(
            target=target,
            open_leaves=(target,),
            objective=objective,
            constraints=dict(constraints or {}),
        )

    @property
    def canonical_id(self) -> str:
        payload = {
            "target": canonical_smiles(self.target) or self.target,
            "steps": [
                {
                    "product": canonical_smiles(step.product) or step.product,
                    "action": step.action.canonical_key,
                }
                for step in self.steps
            ],
            "open": sorted(canonical_smiles(smi) or smi for smi in self.open_leaves),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def with_candidate_pool(self, leaf: str, actions: list[CandidateAction]) -> "RouteTreeState":
        pools = {key: list(value) for key, value in self.candidate_pools.items()}
        pools[canonical_smiles(leaf) or leaf] = list(actions)
        return RouteTreeState(
            target=self.target,
            steps=self.steps,
            open_leaves=self.open_leaves,
            expanded=self.expanded,
            score=self.score,
            depth=self.depth,
            objective=self.objective,
            constraints=dict(self.constraints),
            candidate_pools=pools,
            search_metadata=dict(self.search_metadata),
        )

    def advance(
        self,
        *,
        leaf: str,
        action: CandidateAction,
        next_open_leaves: tuple[str, ...],
        score_delta: float,
    ) -> "RouteTreeState":
        expanded = set(self.expanded)
        can = canonical_smiles(leaf)
        if can:
            expanded.add(can)
        return RouteTreeState(
            target=self.target,
            steps=self.steps + (RouteTreeStep(product=leaf, action=action, depth=self.depth),),
            open_leaves=next_open_leaves,
            expanded=frozenset(expanded),
            score=self.score + float(score_delta),
            depth=self.depth + 1,
            objective=self.objective,
            constraints=dict(self.constraints),
            candidate_pools={key: list(value) for key, value in self.candidate_pools.items()},
            search_metadata=dict(self.search_metadata),
        )

    def to_board(self) -> CascadeBoard:
        board = CascadeBoard.from_n_steps(len(self.steps), self.target)
        for idx, step in enumerate(self.steps):
            slot = board.slots[idx]
            action = step.action
            slot.product = step.product
            slot.main_reactant = action.main_reactant
            slot.aux_reactants = list(action.aux_reactants)
            slot.reaction_smiles = action.rxn_smiles
            slot.reaction_type = action.reaction_type
            slot.ec = action.ec or None
            slot.catalyst = action.catalyst or None
            slot.T = action.T
            slot.pH = action.pH
            slot.solvent = action.solvent or None
            slot.e_retro = action.raw_score
            slot.source = action.source
            slot.evidence = dict(action.metadata.get("evidence") or {})
            pool_key = canonical_smiles(step.product) or step.product
            slot.candidates = [cand.to_candidate_dict() for cand in self.candidate_pools.get(pool_key, [])]
        return board

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.canonical_id,
            "target": self.target,
            "depth": self.depth,
            "score": self.score,
            "open_leaves": list(self.open_leaves),
            "expanded": sorted(self.expanded),
            "steps": [step.to_dict() for step in self.steps],
            "candidate_pool_sizes": {key: len(value) for key, value in self.candidate_pools.items()},
            "objective": self.objective,
            "constraints": self.constraints,
            "search_metadata": self.search_metadata,
        }


def _candidate_reactants(
    candidate: dict[str, Any],
    *,
    rxn: str,
    main: str,
    aux: tuple[str, ...],
) -> tuple[str, ...]:
    reactants: list[str] = []
    if main:
        reactants.append(main)
    reactants.extend(aux)
    if rxn and ">>" in rxn:
        lhs = rxn.split(">>", 1)[0]
        reactants.extend(part.strip() for part in lhs.split(".") if part.strip())
    out: list[str] = []
    seen: set[str] = set()
    for smi in reactants:
        key = canonical_smiles(smi) or smi
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(smi)
    return tuple(out)


def _validity_flags(product: str, rxn: str, reactants: tuple[str, ...], main: str) -> list[str]:
    flags: list[str] = []
    if not reactants:
        flags.append("no_reactants")
    if not main:
        flags.append("no_main_reactant")
    if rxn and ">>" in rxn:
        rhs = {canonical_smiles(part.strip()) for part in rxn.split(">>", 1)[1].split(".") if part.strip()}
        expected = canonical_smiles(product)
        if expected and rhs and expected not in rhs:
            flags.append("product_mismatch")
    if canonical_smiles(main) and canonical_smiles(main) == canonical_smiles(product):
        flags.append("self_loop")
    return flags


def _largest_smiles(values: list[str]) -> str:
    return max(values, key=_heavy_atoms, default="")


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reaction_from_parts(reactants: tuple[str, ...], product: str) -> str:
    if not reactants or not product:
        return ""
    return ".".join(reactants) + ">>" + product
