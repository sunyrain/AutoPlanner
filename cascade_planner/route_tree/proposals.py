"""Proposal-tool adapters for route-tree planning."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

from rdkit import Chem

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.proposal_rankers import SourceSpecificProposalRankers, default_proposal_rankers
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import (
    SourceAllocation,
    SourceGate,
    default_source_gate,
    source_group,
    _source_group_probs,
)


DEFAULT_SOURCE_ORDER = (
    "retrochimera",
    "chem_enzy_onestep",
    "enzyformer",
    "enzexpand",
    "v3_retrieval",
    "native_replay",
    "retrorules",
    "chemtemplates",
)
ENZYMATIC_SOURCE_ORDER = (
    "v3_retrieval",
    "enzyformer",
    "enzexpand",
    "retrorules",
)

_NATIVE_REPLAY_CACHE: dict[tuple[str, bool], dict[str, list[dict[str, Any]]]] = {}


@dataclass
class ProposalContext:
    depth: int = 0
    ec1: int = 0
    reaction_type: str = ""
    T: float | None = None
    pH: float | None = None
    objective: str = "balanced"
    constraints: dict[str, Any] = field(default_factory=dict)
    route_metadata: dict[str, Any] = field(default_factory=dict)


class RetroEngineProposalTool:
    """Expose the existing live retro engine dict as route-tree proposals."""

    def __init__(
        self,
        retro_engine: dict[str, Any] | None,
        *,
        source_order: tuple[str, ...] = DEFAULT_SOURCE_ORDER,
        source_gate: SourceGate | None = None,
        proposal_rankers: SourceSpecificProposalRankers | None = None,
    ):
        self.retro_engine = retro_engine or {}
        self.source_order = source_order
        self.source_gate = source_gate or default_source_gate()
        self.proposal_rankers = proposal_rankers if proposal_rankers is not None else default_proposal_rankers()
        self.last_diagnostics: dict[str, Any] = {}

    def propose(
        self,
        product: str,
        context: ProposalContext | None = None,
        *,
        top_k: int = 10,
    ) -> list[CandidateAction]:
        if not product:
            self.last_diagnostics = {
                "product": product,
                "top_k": int(top_k or 0),
                "sources": {},
                "empty_reason": "missing_product",
            }
            return []
        context = context or ProposalContext()
        sources = self._ordered_sources(context if int(context.ec1 or 0) else None)
        if not sources:
            self.last_diagnostics = {
                "product": product,
                "top_k": int(top_k or 0),
                "sources": {},
                "empty_reason": "missing_engine",
            }
            return self._maybe_append_autoplannrellm_candidate(product, [], context)
        self.last_diagnostics = {
            "product": product,
            "top_k": int(top_k or 0),
            "context": _context_diagnostics(context),
            "ordered_sources": list(sources),
            "sources": {},
        }
        allocation = self.source_gate.allocate(
            product,
            context=context,
            available_sources=sources,
            total_budget=top_k,
        )
        allocation = _apply_source_budget_floor(allocation, sources=sources, total_budget=top_k, context=context)
        self.last_diagnostics["allocation"] = allocation.to_dict()
        self._initialize_source_diagnostics(sources, allocation)
        actions = self._propose_with_allocation(product, context, allocation)
        if actions:
            fallback_sources = self._fallback_sources(allocation, allow_safety_override=False)
            if allocation.fallback_budget > 0 and fallback_sources:
                actions.extend(
                    self._propose_from_sources(
                        product,
                        context,
                        fallback_sources,
                        top_k=allocation.fallback_budget,
                        allocation=allocation,
                    )
            )
            actions = self._dedupe_and_record(actions)
            actions = self._maybe_append_autoplannrellm_candidate(product, actions, context)
            self._observe_source_gate(product, context, allocation)
            return actions
        if allocation.fallback_budget <= 0:
            return self._maybe_append_autoplannrellm_candidate(product, [], context)
        fallback_sources = self._fallback_sources(allocation, allow_safety_override=True)
        if not fallback_sources:
            return self._maybe_append_autoplannrellm_candidate(product, [], context)
        actions = self._dedupe_and_record(
            self._propose_from_sources(
                product,
                context,
                fallback_sources,
                top_k=allocation.fallback_budget,
                allocation=allocation,
            )
        )
        actions = self._maybe_append_autoplannrellm_candidate(product, actions, context)
        self._observe_source_gate(product, context, allocation)
        return actions

    def propose_with_diagnostics(
        self,
        product: str,
        context: ProposalContext | None = None,
        *,
        top_k: int = 10,
    ) -> tuple[list[CandidateAction], SourceAllocation]:
        context = context or ProposalContext()
        sources = self._ordered_sources(context if int(context.ec1 or 0) else None)
        self.last_diagnostics = {
            "product": product,
            "top_k": int(top_k or 0),
            "context": _context_diagnostics(context),
            "ordered_sources": list(sources),
            "sources": {},
        }
        allocation = self.source_gate.allocate(
            product,
            context=context,
            available_sources=sources,
            total_budget=top_k,
        )
        allocation = _apply_source_budget_floor(allocation, sources=sources, total_budget=top_k, context=context)
        self.last_diagnostics["allocation"] = allocation.to_dict()
        self._initialize_source_diagnostics(sources, allocation)
        actions = self._propose_with_allocation(product, context, allocation)
        if allocation.fallback_budget > 0:
            fallback_sources = self._fallback_sources(allocation, allow_safety_override=not actions)
            if fallback_sources:
                fallback_actions = self._propose_from_sources(
                    product,
                    context,
                    fallback_sources,
                    top_k=allocation.fallback_budget,
                    allocation=allocation,
                )
                actions = [*actions, *fallback_actions]
        actions = self._dedupe_and_record(actions)
        actions = self._maybe_append_autoplannrellm_candidate(product, actions, context)
        self._observe_source_gate(product, context, allocation)
        return actions, allocation

    def _maybe_append_autoplannrellm_candidate(
        self,
        product: str,
        actions: list[CandidateAction],
        context: ProposalContext,
    ) -> list[CandidateAction]:
        if not _autoplannrellm_candidate_enabled():
            return actions
        try:
            from AUTOPLANNRELLM.proposals import append_llm_candidate

            return append_llm_candidate(
                product=product,
                actions=actions,
                context=context,
                diagnostics=self.last_diagnostics,
            )
        except Exception:
            return actions

    def _fallback_sources(self, allocation: SourceAllocation, *, allow_safety_override: bool) -> list[str]:
        sources = [source for source in self._ordered_sources(None) if allocation.source_budgets.get(source, 0) <= 0]
        if allocation.safety_guard and not allow_safety_override:
            sources = [source for source in sources if source_group(source) != "chemical"]
        return sources

    def _propose_with_allocation(
        self,
        product: str,
        context: ProposalContext,
        allocation: SourceAllocation,
    ) -> list[CandidateAction]:
        actions: list[CandidateAction] = []
        for source in self._ordered_sources(context if int(context.ec1 or 0) else None):
            budget = int(allocation.source_budgets.get(source) or 0)
            if budget <= 0:
                continue
            actions.extend(
                self._propose_from_sources(
                    product,
                    context,
                    [source],
                    top_k=budget,
                    allocation=allocation,
                )
            )
        return actions

    def _propose_from_sources(
        self,
        product: str,
        context: ProposalContext,
        sources: list[str],
        *,
        top_k: int,
        allocation: SourceAllocation | None = None,
    ) -> list[CandidateAction]:
        actions: list[CandidateAction] = []
        for source in sources:
            disabled_reason = _source_disabled_for_context(source, context)
            if disabled_reason:
                self._record_source_skip(source, reason=disabled_reason)
                continue
            engine = self.retro_engine.get(source)
            if engine is None and source not in {"v3_retrieval", "native_replay"}:
                self._record_source_skip(source, reason="missing_engine")
                continue
            request_top_k = top_k
            if self.proposal_rankers is not None:
                request_top_k = self.proposal_rankers.request_k(source, top_k)
            request_top_k = _cap_source_request_k(source, request_top_k)
            t0 = time.monotonic()
            rows = self._predict(source, engine, product, context, top_k=request_top_k)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            raw_count = len(rows)
            if self.proposal_rankers is not None:
                rows = self.proposal_rankers.rerank(product, source, rows, limit=top_k)
            else:
                rows = rows[:top_k]
            self._record_source_diagnostics(
                source,
                requested_k=request_top_k,
                kept_k=top_k,
                raw_count=raw_count,
                kept_count=len(rows),
                elapsed_ms=elapsed_ms,
            )
            for rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                row = dict(row)
                row.setdefault("source", source)
                row.setdefault("rank", rank)
                action = CandidateAction.from_candidate(product, row, rank=rank, source=row.get("source") or source)
                if allocation is not None:
                    metadata = dict(action.metadata)
                    metadata.setdefault("source_gate", allocation.to_dict())
                    action.metadata = metadata
                actions.append(action)
        return actions

    def _record_source_diagnostics(
        self,
        source: str,
        *,
        requested_k: int,
        kept_k: int,
        raw_count: int,
        kept_count: int,
        elapsed_ms: float,
    ) -> None:
        sources = self.last_diagnostics.setdefault("sources", {})
        row = sources.setdefault(
            source,
            {
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
            },
        )
        row["calls"] = int(row.get("calls") or 0) + 1
        row["queried"] = True
        row["skip_reason"] = ""
        row["requested_k_total"] = int(row.get("requested_k_total") or 0) + int(requested_k or 0)
        row["kept_k_total"] = int(row.get("kept_k_total") or 0) + int(kept_k or 0)
        row["raw_returned"] = int(row.get("raw_returned") or 0) + int(raw_count or 0)
        row["ranker_kept"] = int(row.get("ranker_kept") or 0) + int(kept_count or 0)
        row["ranker_dropped"] = int(row.get("ranker_dropped") or 0) + max(0, int(raw_count or 0) - int(kept_count or 0))
        row["kept_returned"] = int(row.get("kept_returned") or 0) + int(kept_count or 0)
        row["latency_ms_total"] = round(float(row.get("latency_ms_total") or 0.0) + float(elapsed_ms or 0.0), 3)
        row["latency_ms_max"] = round(max(float(row.get("latency_ms_max") or 0.0), float(elapsed_ms or 0.0)), 3)

    def _initialize_source_diagnostics(self, sources: list[str], allocation: SourceAllocation) -> None:
        rows = self.last_diagnostics.setdefault("sources", {})
        budgets = allocation.source_budgets or {}
        for source in sources:
            rows.setdefault(
                source,
                {
                    "calls": 0,
                    "queried": False,
                    "allocated_budget": int(budgets.get(source) or 0),
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
                    "skip_reason": "zero_budget" if int(budgets.get(source) or 0) <= 0 else "",
                },
            )

    def _record_source_skip(self, source: str, *, reason: str) -> None:
        row = self.last_diagnostics.setdefault("sources", {}).setdefault(source, {})
        row.setdefault("calls", 0)
        row.setdefault("queried", False)
        row.setdefault("allocated_budget", 0)
        row.setdefault("requested_k_total", 0)
        row.setdefault("kept_k_total", 0)
        row.setdefault("raw_returned", 0)
        row.setdefault("ranker_kept", 0)
        row.setdefault("ranker_dropped", 0)
        row.setdefault("kept_returned", 0)
        row.setdefault("dedupe_dropped", 0)
        row.setdefault("invalid_dropped", 0)
        row.setdefault("final_returned", 0)
        row.setdefault("latency_ms_total", 0.0)
        row.setdefault("latency_ms_max", 0.0)
        row["skip_reason"] = reason

    def _dedupe_and_record(self, actions: list[CandidateAction]) -> list[CandidateAction]:
        deduped, diagnostics = _dedupe_actions_with_diagnostics(actions)
        rows = self.last_diagnostics.setdefault("sources", {})
        for source, dropped in (diagnostics.get("dedupe_dropped") or {}).items():
            row = rows.setdefault(str(source), {})
            row["dedupe_dropped"] = int(row.get("dedupe_dropped") or 0) + int(dropped or 0)
        for source, dropped in (diagnostics.get("invalid_dropped") or {}).items():
            row = rows.setdefault(str(source), {})
            row["invalid_dropped"] = int(row.get("invalid_dropped") or 0) + int(dropped or 0)
        final_counts: dict[str, int] = {}
        for action in deduped:
            source = action.source or "unknown"
            final_counts[source] = final_counts.get(source, 0) + 1
        for source, row in rows.items():
            row["final_returned"] = int(row.get("final_returned") or 0) + int(final_counts.get(source, 0))
        self.last_diagnostics["dedupe"] = diagnostics
        return deduped

    def _observe_source_gate(self, product: str, context: ProposalContext, allocation: SourceAllocation) -> None:
        observer = getattr(self.source_gate, "observe", None)
        if not callable(observer):
            return
        try:
            observer(product=product, context=context, allocation=allocation, diagnostics=self.last_diagnostics)
        except Exception:
            return

    def _ordered_sources(self, context: ProposalContext | None = None) -> list[str]:
        if context is not None and int(context.ec1 or 0):
            ordered = [source for source in ENZYMATIC_SOURCE_ORDER if self._source_available(source)]
            if "v3_retrieval" not in ordered and _retrieval_enabled():
                ordered.insert(0, "v3_retrieval")
            ordered.extend(
                source
                for source in self.retro_engine
                if source not in ordered and source in ENZYMATIC_SOURCE_ORDER
            )
            ordered.extend(source for source in self.source_order if source in self.retro_engine and source not in ordered)
            ordered.extend(source for source in self.retro_engine if source not in ordered)
            if ordered:
                return ordered
        ordered = [source for source in self.source_order if self._source_available(source)]
        if _retrieval_enabled() and "v3_retrieval" not in ordered:
            ordered.insert(0, "v3_retrieval")
        ordered.extend(source for source in self.retro_engine if source not in ordered)
        return ordered

    def _source_available(self, source: str) -> bool:
        return (
            source in self.retro_engine
            or (source == "v3_retrieval" and _retrieval_enabled())
            or (source == "native_replay" and _native_replay_enabled())
        )

    def _non_enzymatic_sources(self) -> list[str]:
        enzymatic = set(ENZYMATIC_SOURCE_ORDER)
        return [source for source in self._ordered_sources(None) if source not in enzymatic]

    def _predict(
        self,
        source: str,
        engine: Any,
        product: str,
        context: ProposalContext,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if source == "native_replay":
            return _native_replay_predict(product, top_k=top_k)
        if source == "v3_retrieval" and engine is None:
            try:
                from cascade_planner.cascadeboard.enz_retrieval import retrieve_enzymatic_reactions

                return list(
                    retrieve_enzymatic_reactions(
                        product,
                        ec_class=str(context.ec1) if context.ec1 else "",
                        top_k=top_k,
                    )
                    or []
                )[:top_k]
            except Exception:
                return []
        kwargs: dict[str, Any] = {"top_k": top_k}
        if source in {"enzyformer", "retrorules", "v3_retrieval"} and context.ec1:
            kwargs["ec_token"] = str(context.ec1)
        if source in {"retrorules", "chemtemplates"} and context.reaction_type:
            kwargs["skel_type"] = context.reaction_type
        attempts = [
            kwargs,
            {k: v for k, v in kwargs.items() if k != "skel_type"},
            {k: v for k, v in kwargs.items() if k != "ec_token"},
            {"top_k": top_k},
            {},
        ]
        for call_kwargs in attempts:
            try:
                rows = engine.predict(product, **call_kwargs)
                return list(rows or [])[:top_k]
            except TypeError:
                continue
            except Exception:
                return []
        return []


def _native_replay_enabled() -> bool:
    return bool(os.environ.get("AUTOPLANNER_NATIVE_REPLAY_PROPOSALS")) and _env_truthy_default(
        "AUTOPLANNER_ENABLE_NATIVE_REPLAY_PROPOSALS",
        True,
    )


def _native_replay_predict(product: str, *, top_k: int) -> list[dict[str, Any]]:
    index = _native_replay_index()
    key = canonical_leaf_key(product)
    rows = list(index.get(key) or [])
    rows.sort(key=lambda row: (-float(row.get("score") or 0.0), int(row.get("rank") or 999999)))
    return [dict(row) for row in rows[: max(0, int(top_k or 0))]]


def _native_replay_index() -> dict[str, list[dict[str, Any]]]:
    path = os.environ.get("AUTOPLANNER_NATIVE_REPLAY_PROPOSALS") or ""
    allow_eval = _env_truthy("AUTOPLANNER_NATIVE_REPLAY_ALLOW_EVAL_ONLY")
    cache_key = (path, allow_eval)
    cached = _NATIVE_REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index: dict[str, list[dict[str, Any]]] = {}
    if not path:
        _NATIVE_REPLAY_CACHE[cache_key] = index
        return index
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        _NATIVE_REPLAY_CACHE[cache_key] = index
        return index
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if bool(row.get("eval_only")) and not allow_eval:
            continue
        candidate = _native_replay_candidate(row)
        if not candidate:
            continue
        leaf_key = canonical_leaf_key(str(row.get("leaf") or row.get("target_smiles") or ""))
        index.setdefault(leaf_key, []).append(candidate)
    _NATIVE_REPLAY_CACHE[cache_key] = index
    return index


def _native_replay_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    reaction = str(row.get("candidate_reaction") or "")
    product = str(row.get("leaf") or row.get("target_smiles") or "")
    reactants = [str(item) for item in row.get("reactants") or [] if item]
    if not reaction or not product or not reactants:
        return None
    score = row.get("teacher_action_value", row.get("teacher_route_value", 0.0))
    return {
        "main_reactant": reactants[0],
        "aux_reactants": reactants[1:],
        "rxn_smiles": reaction,
        "reaction_smiles": reaction,
        "source": "native_replay",
        "score": float(score or 0.0),
        "rank": int(row.get("reservoir_rank") or row.get("teacher_route_rank") or 0),
        "type": str((row.get("route_context_features") or {}).get("reaction_type") or ""),
        "native_replay": True,
        "native_replay_eval_only": bool(row.get("eval_only")),
        "native_replay_state_id": row.get("state_id"),
        "native_replay_route_rank": row.get("teacher_route_rank"),
        "native_replay_teacher_stock_closed": bool(row.get("teacher_stock_closed")),
        "native_replay_teacher_exact_hit": bool(row.get("teacher_exact_hit")),
        "native_replay_teacher_gt_reactant_hit": bool(row.get("teacher_gt_reactant_hit")),
    }


def _dedupe_actions(actions: list[CandidateAction]) -> list[CandidateAction]:
    return _dedupe_actions_with_diagnostics(actions)[0]


def _dedupe_actions_with_diagnostics(actions: list[CandidateAction]) -> tuple[list[CandidateAction], dict[str, dict[str, int]]]:
    out: list[CandidateAction] = []
    seen: set[str] = set()
    dedupe_dropped: dict[str, int] = {}
    invalid_dropped: dict[str, int] = {}
    for action in actions:
        source = action.source or "unknown"
        key = action.canonical_key
        if key in seen:
            dedupe_dropped[source] = dedupe_dropped.get(source, 0) + 1
            continue
        if "no_reactants" in action.validity_flags or "no_main_reactant" in action.validity_flags:
            invalid_dropped[source] = invalid_dropped.get(source, 0) + 1
            continue
        seen.add(key)
        out.append(action)
    for idx, action in enumerate(out, start=1):
        if not action.rank:
            action.rank = idx
    return out, {
        "dedupe_dropped": dedupe_dropped,
        "invalid_dropped": invalid_dropped,
    }


def _apply_source_budget_floor(
    allocation: SourceAllocation,
    *,
    sources: list[str],
    total_budget: int,
    context: ProposalContext | None = None,
) -> SourceAllocation:
    floors = _source_budget_floors()
    floors.update(_contextual_source_budget_floors(context, sources=sources, total_budget=total_budget))
    if not floors:
        return allocation
    budgets = {source: int(allocation.source_budgets.get(source) or 0) for source in sources}
    for source, floor in floors.items():
        if source in budgets:
            budgets[source] = max(budgets[source], floor)
    max_total = max(1, int(total_budget or 1))
    floors = _fit_floor_budget(floors, sources=sources, total_budget=max_total)
    while sum(budgets.values()) > max_total:
        candidates = [
            source
            for source, budget in budgets.items()
            if budget > floors.get(source, 0)
        ]
        if not candidates:
            break
        source = max(candidates, key=lambda item: budgets[item])
        budgets[source] -= 1
    weights_total = sum(max(0, value) for value in budgets.values())
    source_weights = (
        {source: budgets[source] / weights_total for source in budgets}
        if weights_total > 0
        else dict(allocation.source_weights)
    )
    group_probs = _source_group_probs(source_weights) if weights_total > 0 else dict(allocation.source_group_probs)
    metadata_flags = dict(allocation.molecule_flags)
    metadata_flags["source_budget_floor_active"] = True
    return SourceAllocation(
        source_weights=source_weights,
        source_budgets=budgets,
        fallback_budget=allocation.fallback_budget,
        molecule_flags=metadata_flags,
        safety_guard=allocation.safety_guard,
        source_group_probs=group_probs,
        budget_multiplier=float(allocation.budget_multiplier),
        budget_multiplier_label=allocation.budget_multiplier_label,
        decision=allocation.decision,
        policy_confidence=float(allocation.policy_confidence),
        policy_reason=allocation.policy_reason,
        policy_state_id=allocation.policy_state_id,
        selected_source_group=allocation.selected_source_group or (max(group_probs, key=group_probs.get) if group_probs else ""),
        fallback_reason=allocation.fallback_reason,
    )


def _source_budget_floors() -> dict[str, int]:
    raw = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS") or ""
    floors: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            key, value = item.split(":", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            continue
        try:
            floor = int(value)
        except ValueError:
            continue
        if floor > 0:
            floors[key.strip()] = floor
    return floors


def _source_disabled_for_context(source: str, context: ProposalContext | None) -> str:
    """Return a diagnostic reason when a source is disabled by depth policy."""
    raw = os.environ.get("AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH") or ""
    if not raw:
        return ""
    depth = int(getattr(context, "depth", 0) or 0)
    source = str(source or "").strip()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            key, value = item.split(":", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            continue
        if key.strip() != source:
            continue
        try:
            max_depth = int(value)
        except ValueError:
            continue
        if depth > max_depth:
            return f"disabled_after_depth_{max_depth}"
    return ""


def _contextual_source_budget_floors(
    context: ProposalContext | None,
    *,
    sources: list[str],
    total_budget: int,
) -> dict[str, int]:
    if not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_SOURCE_FLOORS", True):
        return {}
    if context is None or int(total_budget or 0) <= 0:
        return {}
    depth = int(getattr(context, "depth", 0) or 0)
    route_metadata = dict(getattr(context, "route_metadata", {}) or {})
    if route_metadata.get("stock_rescue_retry"):
        return _stock_rescue_source_budget_floors(context, sources=sources, total_budget=total_budget)
    if depth != 0:
        return {}
    available = set(sources)
    floors: dict[str, int] = {}
    reaction_type = str(getattr(context, "reaction_type", "") or "").lower()
    ec1 = int(getattr(context, "ec1", 0) or 0)
    if "native_replay" in available and _native_replay_enabled():
        floors["native_replay"] = _native_replay_min_budget()
    if ec1:
        if "v3_retrieval" in available:
            floors["v3_retrieval"] = 3
        if "enzyformer" in available:
            floors["enzyformer"] = 2
        if "enzexpand" in available:
            floors["enzexpand"] = 1
        return floors
    if _is_chemical_reaction_context(reaction_type):
        if "retrochimera" in available:
            floors["retrochimera"] = 3
        if "chem_enzy_onestep" in available:
            floors["chem_enzy_onestep"] = _chem_enzy_onestep_min_budget()
        if "chemtemplates" in available:
            floors["chemtemplates"] = 2
    return floors


def _is_chemical_reaction_context(reaction_type: str) -> bool:
    text = str(reaction_type or "").strip().lower()
    if not text:
        return True
    enzymatic_markers = ("enzyme", "enzymatic", "bio", "rhea", "retrorules")
    return not any(marker in text for marker in enzymatic_markers)


def _stock_rescue_source_budget_floors(
    context: ProposalContext,
    *,
    sources: list[str],
    total_budget: int,
) -> dict[str, int]:
    del total_budget
    available = set(sources)
    route_metadata = dict(getattr(context, "route_metadata", {}) or {})
    ec1 = int(getattr(context, "ec1", 0) or 0)
    enzymatic_route = bool(
        ec1
        or route_metadata.get("enzymatic_only_route")
        or route_metadata.get("carbohydrate_like_route")
    )
    floors: dict[str, int] = {}
    if enzymatic_route:
        for source, floor in (
            ("v3_retrieval", 2),
            ("enzyformer", 2),
            ("retrorules", 2),
            ("enzexpand", 1),
            ("native_replay", _native_replay_min_budget() if _native_replay_enabled() else 0),
        ):
            if source in available:
                floors[source] = floor
        return floors
    for source, floor in (
        ("retrochimera", 2),
        ("chem_enzy_onestep", _chem_enzy_onestep_min_budget()),
        ("chemtemplates", 2),
        ("retrorules", 1),
        ("v3_retrieval", 1),
        ("native_replay", _native_replay_min_budget() if _native_replay_enabled() else 0),
    ):
        if source in available and int(floor or 0) > 0:
            floors[source] = floor
    return floors


def _fit_floor_budget(floors: dict[str, int], *, sources: list[str], total_budget: int) -> dict[str, int]:
    total_budget = max(1, int(total_budget or 1))
    out = {source: int(value) for source, value in floors.items() if source in sources and int(value or 0) > 0}
    while sum(out.values()) > total_budget and out:
        source = max(out, key=lambda key: (out[key], -sources.index(key) if key in sources else 0))
        out[source] -= 1
        if out[source] <= 0:
            out.pop(source, None)
    return out


def _cap_source_request_k(source: str, request_top_k: int) -> int:
    request_top_k = max(1, int(request_top_k or 1))
    if not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_REQUEST_CAPS", True):
        return request_top_k
    caps = {
        "v3_retrieval": 16,
        "enzyformer": 16,
        "enzexpand": 12,
        "retrorules": 12,
        "retrochimera": 16,
        "chem_enzy_onestep": 50,
        "chemtemplates": 12,
    }
    raw = os.environ.get("AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS") or ""
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            key, value = item.split(":", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            continue
        try:
            caps[key.strip()] = int(value)
        except ValueError:
            continue
    cap = caps.get(str(source or ""))
    return min(request_top_k, max(1, int(cap))) if cap else request_top_k


def _native_replay_min_budget() -> int:
    try:
        return max(1, int(os.environ.get("AUTOPLANNER_NATIVE_REPLAY_MIN_BUDGET") or 1))
    except ValueError:
        return 1


def _chem_enzy_onestep_min_budget() -> int:
    try:
        return max(1, int(os.environ.get("AUTOPLANNER_CHEMENZY_ONESTEP_MIN_BUDGET") or 8))
    except ValueError:
        return 8


def _context_diagnostics(context: ProposalContext) -> dict[str, Any]:
    return {
        "depth": int(context.depth or 0),
        "ec1": int(context.ec1 or 0),
        "reaction_type": context.reaction_type or "",
        "T": context.T,
        "pH": context.pH,
        "route_metadata": dict(context.route_metadata or {}),
    }


def canonical_leaf_key(smiles: str) -> str:
    return canonical_smiles(smiles) or smiles


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _retrieval_enabled() -> bool:
    return _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL", False)


def _autoplannrellm_candidate_enabled() -> bool:
    return _env_truthy("AUTOPLANNRELLM_ENABLE") and _env_truthy_default(
        "AUTOPLANNRELLM_ADD_LLM_CANDIDATE",
        True,
    )


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _oxygen_rich_leaf(smiles: str | None) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    heavy = mol.GetNumHeavyAtoms()
    oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    return oxygen >= 5 and oxygen / max(heavy, 1) >= 0.40


def _carbohydrate_like_leaf(smiles: str | None) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None or not _oxygen_rich_leaf(smiles):
        return False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset({"C", "O"})
