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
    "semisynthesis_rescue",
    "chemical_anchor_rescue",
    "chem_enzy_graphfp_fusion",
    "template_relevance",
    "chem_enzy_onestep",
    "chem_enzy_bionav",
    "enzyformer",
    "enzexpand",
    "v3_retrieval",
    "enzyme_precedent",
    "native_replay",
    "retrorules",
    "chemtemplates",
)
ENZYMATIC_SOURCE_ORDER = (
    "chem_enzy_bionav",
    "enzyme_precedent",
    "v3_retrieval",
    "enzyformer",
    "enzexpand",
    "retrorules",
)

_NATIVE_REPLAY_CACHE: dict[tuple[str, bool], dict[str, list[dict[str, Any]]]] = {}
_SEMISYNTHESIS_RESCUE_APPLICABILITY_CACHE: dict[str, bool] = {}
_CHEMICAL_ANCHOR_RESCUE_APPLICABILITY_CACHE: dict[str, bool] = {}
_ROUTE_TREE_CONDITION_PREDICTOR_CACHE: dict[tuple[str, str], Any] = {}
CHEMENZY_ONESTEP_ROUTE_MODE_ENV = "AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE"
CHEMENZY_ONESTEP_EAGER_MODES = {"", "eager", "budgeted", "always"}
CHEMENZY_ONESTEP_FALLBACK_MODES = {"fallback", "gated", "on_empty", "rescue"}
CHEMENZY_ONESTEP_ADAPTIVE_MODES = {"adaptive", "quality", "low_quality"}
CHEMENZY_ONESTEP_DISABLED_MODES = {"0", "false", "off", "disabled", "none"}


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
        sources = _filter_product_applicable_sources(
            product,
            self._ordered_sources(context if int(context.ec1 or 0) else None),
        )
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
        allocation = _apply_source_budget_floor(
            allocation,
            sources=sources,
            total_budget=top_k,
            context=context,
            product=product,
        )
        allocation = _apply_route_source_gates(allocation, sources=sources, total_budget=top_k, context=context)
        self.last_diagnostics["allocation"] = allocation.to_dict()
        self._initialize_source_diagnostics(sources, allocation)
        actions = self._propose_with_allocation(product, context, allocation)
        if actions:
            adaptive = _chem_enzy_onestep_adaptive_fallback(actions, context, allocation, total_budget=top_k)
            force_sources: set[str] = set()
            fallback_budget = int(allocation.fallback_budget or 0)
            if adaptive is not None:
                self.last_diagnostics["chem_enzy_onestep_adaptive_fallback"] = adaptive
                if bool(adaptive.get("triggered")):
                    force_sources.add("chem_enzy_onestep")
                    fallback_budget = int(adaptive.get("budget") or fallback_budget)
            fallback_sources = self._fallback_sources(
                allocation,
                context=context,
                allow_safety_override=False,
                force_sources=force_sources,
                product=product,
            )
            if fallback_budget > 0 and fallback_sources:
                actions.extend(
                    self._propose_from_sources(
                        product,
                        context,
                        fallback_sources,
                        top_k=fallback_budget,
                        allocation=allocation,
                    )
            )
            actions = self._dedupe_and_record(actions)
            actions = self._maybe_append_autoplannrellm_candidate(product, actions, context)
            self._observe_source_gate(product, context, allocation)
            return actions
        if allocation.fallback_budget <= 0:
            return self._maybe_append_autoplannrellm_candidate(product, [], context)
        fallback_sources = self._fallback_sources(
            allocation,
            context=context,
            allow_safety_override=True,
            product=product,
        )
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
        sources = _filter_product_applicable_sources(
            product,
            self._ordered_sources(context if int(context.ec1 or 0) else None),
        )
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
        allocation = _apply_source_budget_floor(
            allocation,
            sources=sources,
            total_budget=top_k,
            context=context,
            product=product,
        )
        allocation = _apply_route_source_gates(allocation, sources=sources, total_budget=top_k, context=context)
        self.last_diagnostics["allocation"] = allocation.to_dict()
        self._initialize_source_diagnostics(sources, allocation)
        actions = self._propose_with_allocation(product, context, allocation)
        if allocation.fallback_budget > 0:
            adaptive = (
                _chem_enzy_onestep_adaptive_fallback(actions, context, allocation, total_budget=top_k)
                if actions
                else None
            )
            force_sources: set[str] = set()
            fallback_budget = int(allocation.fallback_budget or 0)
            if adaptive is not None:
                self.last_diagnostics["chem_enzy_onestep_adaptive_fallback"] = adaptive
                if bool(adaptive.get("triggered")):
                    force_sources.add("chem_enzy_onestep")
                    fallback_budget = int(adaptive.get("budget") or fallback_budget)
            fallback_sources = self._fallback_sources(
                allocation,
                context=context,
                allow_safety_override=not actions,
                force_sources=force_sources,
                product=product,
            )
            if fallback_sources:
                fallback_actions = self._propose_from_sources(
                    product,
                    context,
                    fallback_sources,
                    top_k=fallback_budget,
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

    def _fallback_sources(
        self,
        allocation: SourceAllocation,
        *,
        context: ProposalContext | None = None,
        allow_safety_override: bool,
        force_sources: set[str] | None = None,
        product: str = "",
    ) -> list[str]:
        sources = [source for source in self._ordered_sources(None) if allocation.source_budgets.get(source, 0) <= 0]
        sources = _filter_product_applicable_sources(product, sources)
        sources = _filter_route_fallback_sources(
            sources,
            context=context,
            allow_safety_override=allow_safety_override,
            force_sources=force_sources,
        )
        if _chem_enzy_graphfp_fusion_suppressed(allocation, context):
            sources = [source for source in sources if source != "chem_enzy_graphfp_fusion"]
        if _bridge_gate_suppresses_enzymatic_fallback(allocation):
            sources = [source for source in sources if source_group(source) not in {"enzymatic", "rhea_retrorules"}]
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
            if source == "semisynthesis_rescue" and not _semisynthesis_rescue_applicable(product):
                self._record_source_skip(source, reason="semisynthesis_rescue_not_applicable")
                continue
            if source == "chemical_anchor_rescue" and not _chemical_anchor_rescue_applicable(product):
                self._record_source_skip(source, reason="chemical_anchor_rescue_not_applicable")
                continue
            disabled_reason = _source_disabled_for_context(source, context)
            if disabled_reason:
                self._record_source_skip(source, reason=disabled_reason)
                continue
            engine = self.retro_engine.get(source)
            if engine is None and source not in {"v3_retrieval", "native_replay", "enzyme_precedent"}:
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
                action = _attach_runtime_condition_prediction(action)
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
            if "enzyme_precedent" not in ordered and _enzyme_precedent_enabled():
                ordered.insert(0, "enzyme_precedent")
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
            or (source == "enzyme_precedent" and _enzyme_precedent_enabled())
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
        if source == "enzyme_precedent" and engine is None:
            try:
                from cascade_planner.cascadeboard.enzyme_precedent_retrieval import retrieve_enzyme_precedents

                return list(
                    retrieve_enzyme_precedents(
                        product,
                        ec_class=str(context.ec1) if context.ec1 else "",
                        top_k=top_k,
                    )
                    or []
                )[:top_k]
            except Exception:
                return []
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


def _bridge_gate_suppresses_enzymatic_fallback(allocation: SourceAllocation) -> bool:
    flags = allocation.molecule_flags or {}
    if not bool(flags.get("bridge_gate_checked")):
        return False
    try:
        hits = int(flags.get("bridge_gate_hits") or 0)
    except (TypeError, ValueError):
        hits = 0
    if hits > 0:
        return False
    return str(allocation.policy_reason or "").startswith("bridge_gate_no_hits")


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
    seen: dict[tuple[str, str], int] = {}
    dedupe_dropped: dict[str, int] = {}
    invalid_dropped: dict[str, int] = {}
    for action in actions:
        source = action.source or "unknown"
        key = (action.canonical_key, _dedupe_source_group(source))
        if key in seen:
            existing_idx = seen[key]
            existing = out[existing_idx]
            if _prefer_duplicate_action(action, existing):
                out[existing_idx] = _merge_duplicate_action(action, existing)
                dropped_source = existing.source or "unknown"
                dedupe_dropped[dropped_source] = dedupe_dropped.get(dropped_source, 0) + 1
            else:
                out[existing_idx] = _merge_duplicate_action(existing, action)
                dedupe_dropped[source] = dedupe_dropped.get(source, 0) + 1
            continue
        if "no_reactants" in action.validity_flags or "no_main_reactant" in action.validity_flags:
            invalid_dropped[source] = invalid_dropped.get(source, 0) + 1
            continue
        seen[key] = len(out)
        out.append(action)
    for idx, action in enumerate(out, start=1):
        if not action.rank:
            action.rank = idx
    return out, {
        "dedupe_dropped": dedupe_dropped,
        "invalid_dropped": invalid_dropped,
    }


def _dedupe_source_group(source: str) -> str:
    """Keep chemical and enzymatic duplicate reactions as separate evidence."""
    return source_group(source)


def _prefer_duplicate_action(candidate: CandidateAction, existing: CandidateAction) -> bool:
    return _duplicate_source_priority(candidate.source) < _duplicate_source_priority(existing.source)


def _duplicate_source_priority(source: str) -> int:
    if str(source or "").lower() in {"semisynthesis_rescue", "chemical_anchor_rescue"}:
        return 0
    return 10


def _merge_duplicate_action(primary: CandidateAction, duplicate: CandidateAction) -> CandidateAction:
    metadata = dict(primary.metadata)
    provenance = _action_source_provenance(primary)
    metadata["source_provenance"] = provenance
    duplicate_items = _merged_duplicate_provenance(metadata.get("duplicate_source_provenance"))
    for item in _action_all_source_provenance(duplicate):
        _append_unique_provenance(duplicate_items, item)
    metadata["duplicate_source_provenance"] = duplicate_items
    primary.metadata = metadata
    return primary


def _action_all_source_provenance(action: CandidateAction) -> list[dict[str, Any]]:
    items = [_action_source_provenance(action)]
    for item in _merged_duplicate_provenance(action.metadata.get("duplicate_source_provenance")):
        _append_unique_provenance(items, item)
    return items


def _action_source_provenance(action: CandidateAction) -> dict[str, Any]:
    raw = action.metadata.get("source_provenance") if isinstance(action.metadata, dict) else None
    provenance = dict(raw) if isinstance(raw, dict) else {}
    provenance["source"] = action.source or provenance.get("source") or "unknown"
    provenance.setdefault("rank", int(action.rank or 0))
    provenance.setdefault("raw_score", float(action.raw_score or 0.0))
    provenance.setdefault("evidence_present", bool(action.metadata.get("evidence") if isinstance(action.metadata, dict) else False))
    return provenance


def _merged_duplicate_provenance(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _append_unique_provenance(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    source = str(item.get("source") or "")
    rank = item.get("rank")
    for existing in items:
        if str(existing.get("source") or "") == source and existing.get("rank") == rank:
            return
    items.append(dict(item))


def _apply_source_budget_floor(
    allocation: SourceAllocation,
    *,
    sources: list[str],
    total_budget: int,
    context: ProposalContext | None = None,
    product: str = "",
) -> SourceAllocation:
    floors = _merge_source_budget_floors(
        _source_budget_floors(),
        _contextual_source_budget_floors(context, sources=sources, total_budget=total_budget, product=product),
        _bridge_hit_source_budget_floors(allocation, sources=sources, product=product),
    )
    floors = _filter_floors_allowed_by_allocation(floors, allocation)
    if not floors:
        return allocation
    requested_floors = dict(floors)
    budgets = {source: int(allocation.source_budgets.get(source) or 0) for source in sources}
    for source, floor in floors.items():
        if source in budgets:
            budgets[source] = max(budgets[source], floor)
    max_total = max(1, int(total_budget or 1))
    floors = _fit_floor_budget(floors, sources=sources, total_budget=max_total)
    floors = _protect_applicable_semisynthesis_floor(
        floors,
        requested_floors=requested_floors,
        sources=sources,
        total_budget=max_total,
        product=product,
    )
    floors = _protect_applicable_chemical_anchor_floor(
        floors,
        requested_floors=requested_floors,
        sources=sources,
        total_budget=max_total,
        product=product,
    )
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


def _apply_route_source_gates(
    allocation: SourceAllocation,
    *,
    sources: list[str],
    total_budget: int,
    context: ProposalContext | None = None,
) -> SourceAllocation:
    del total_budget
    allocation = _apply_chem_enzy_graphfp_fusion_route_gate(allocation, sources=sources, context=context)
    return _apply_chem_enzy_onestep_route_mode(allocation, sources=sources, context=context)


def _apply_chem_enzy_graphfp_fusion_route_gate(
    allocation: SourceAllocation,
    *,
    sources: list[str],
    context: ProposalContext | None,
) -> SourceAllocation:
    source = "chem_enzy_graphfp_fusion"
    if source not in sources:
        return allocation
    if not _chem_enzy_graphfp_fusion_suppressed(allocation, context):
        return allocation
    budgets = {item: int(allocation.source_budgets.get(item) or 0) for item in sources}
    if budgets.get(source, 0) <= 0:
        return allocation
    budgets = _zero_source_budget_and_redistribute(
        source=source,
        budgets=budgets,
        sources=sources,
        allocation=allocation,
    )
    metadata_flags = dict(allocation.molecule_flags)
    metadata_flags["chem_enzy_graphfp_fusion_route_gate_active"] = True
    if _allocation_bridge_hits(allocation) > 0:
        metadata_flags["chem_enzy_graphfp_fusion_bridge_gate_active"] = True
    if _explicit_enzyme_context(context):
        metadata_flags["chem_enzy_graphfp_fusion_ec_context_gate_active"] = True
    return _replace_allocation_budgets(
        allocation,
        budgets=budgets,
        molecule_flags=metadata_flags,
        fallback_reason=allocation.fallback_reason or "chem_enzy_graphfp_fusion_bridge_hit",
    )


def _chem_enzy_graphfp_fusion_suppressed(
    allocation: SourceAllocation,
    context: ProposalContext | None,
) -> bool:
    if _allocation_bridge_hits(allocation) > 0 and not _env_truthy(
        "AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_ALLOW_BRIDGE_HITS"
    ):
        return True
    if _explicit_enzyme_context(context) and not _env_truthy(
        "AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_ALLOW_ENZYME_CONTEXTS"
    ):
        return True
    return False


def _explicit_enzyme_context(context: ProposalContext | None) -> bool:
    if context is None:
        return False
    route_metadata = dict(getattr(context, "route_metadata", {}) or {})
    return bool(
        int(getattr(context, "ec1", 0) or 0)
        or route_metadata.get("bridge_ec_context_injected")
        or route_metadata.get("carbohydrate_like_route")
    )


def _apply_chem_enzy_onestep_route_mode(
    allocation: SourceAllocation,
    *,
    sources: list[str],
    context: ProposalContext | None,
) -> SourceAllocation:
    source = "chem_enzy_onestep"
    if source not in sources:
        return allocation
    mode = _chem_enzy_onestep_route_mode()
    if mode in CHEMENZY_ONESTEP_EAGER_MODES:
        return allocation
    if (
        mode in (CHEMENZY_ONESTEP_FALLBACK_MODES | CHEMENZY_ONESTEP_ADAPTIVE_MODES)
        and _chem_enzy_onestep_budgeted_context(context)
    ):
        return allocation
    budgets = {item: int(allocation.source_budgets.get(item) or 0) for item in sources}
    if budgets.get(source, 0) <= 0:
        return allocation
    budgets = _zero_source_budget_and_redistribute(
        source=source,
        budgets=budgets,
        sources=sources,
        allocation=allocation,
    )
    metadata_flags = dict(allocation.molecule_flags)
    metadata_flags["chem_enzy_onestep_route_gate_active"] = True
    metadata_flags[f"chem_enzy_onestep_route_mode_{mode}"] = True
    return _replace_allocation_budgets(
        allocation,
        budgets=budgets,
        molecule_flags=metadata_flags,
        fallback_reason=allocation.fallback_reason or f"chem_enzy_onestep_{mode}",
    )


def _zero_source_budget_and_redistribute(
    *,
    source: str,
    budgets: dict[str, int],
    sources: list[str],
    allocation: SourceAllocation,
) -> dict[str, int]:
    out = dict(budgets)
    freed = max(0, int(out.get(source) or 0))
    out[source] = 0
    if freed <= 0:
        return out
    recipients = [
        item
        for item in sources
        if item != source and int(out.get(item) or 0) > 0
    ]
    if not recipients:
        recipients = [item for item in sources if item != source]
    if not recipients:
        return out
    recipients.sort(key=lambda item: float(allocation.source_weights.get(item) or 0.0), reverse=True)
    for idx in range(freed):
        out[recipients[idx % len(recipients)]] = int(out.get(recipients[idx % len(recipients)]) or 0) + 1
    return out


def _replace_allocation_budgets(
    allocation: SourceAllocation,
    *,
    budgets: dict[str, int],
    molecule_flags: dict[str, Any] | None = None,
    fallback_reason: str = "",
) -> SourceAllocation:
    weights_total = sum(max(0, int(value or 0)) for value in budgets.values())
    source_weights = (
        {source: int(budgets[source]) / weights_total for source in budgets}
        if weights_total > 0
        else dict(allocation.source_weights)
    )
    group_probs = _source_group_probs(source_weights) if weights_total > 0 else dict(allocation.source_group_probs)
    return SourceAllocation(
        source_weights=source_weights,
        source_budgets=dict(budgets),
        fallback_budget=allocation.fallback_budget,
        molecule_flags=dict(molecule_flags if molecule_flags is not None else allocation.molecule_flags),
        safety_guard=allocation.safety_guard,
        source_group_probs=group_probs,
        budget_multiplier=float(allocation.budget_multiplier),
        budget_multiplier_label=allocation.budget_multiplier_label,
        decision=allocation.decision,
        policy_confidence=float(allocation.policy_confidence),
        policy_reason=allocation.policy_reason,
        policy_state_id=allocation.policy_state_id,
        selected_source_group=allocation.selected_source_group or (max(group_probs, key=group_probs.get) if group_probs else ""),
        fallback_reason=fallback_reason or allocation.fallback_reason,
    )


def _merge_source_budget_floors(*items: dict[str, int]) -> dict[str, int]:
    floors: dict[str, int] = {}
    for item in items:
        for source, value in (item or {}).items():
            try:
                floor = int(value)
            except (TypeError, ValueError):
                continue
            if floor > 0:
                floors[source] = max(int(floors.get(source) or 0), floor)
    return floors


def _filter_route_fallback_sources(
    sources: list[str],
    *,
    context: ProposalContext | None,
    allow_safety_override: bool,
    force_sources: set[str] | None = None,
) -> list[str]:
    mode = _chem_enzy_onestep_route_mode()
    force_sources = set(force_sources or set())
    if "chem_enzy_onestep" not in sources:
        return sources
    if mode in CHEMENZY_ONESTEP_DISABLED_MODES:
        return [source for source in sources if source != "chem_enzy_onestep"]
    if mode in (CHEMENZY_ONESTEP_FALLBACK_MODES | CHEMENZY_ONESTEP_ADAPTIVE_MODES):
        if (
            "chem_enzy_onestep" in force_sources
            or allow_safety_override
            or _chem_enzy_onestep_budgeted_context(context)
        ):
            return sources
        return [source for source in sources if source != "chem_enzy_onestep"]
    return sources


def _chem_enzy_onestep_adaptive_fallback(
    actions: list[CandidateAction],
    context: ProposalContext | None,
    allocation: SourceAllocation,
    *,
    total_budget: int,
) -> dict[str, Any] | None:
    mode = _chem_enzy_onestep_route_mode()
    if mode not in CHEMENZY_ONESTEP_ADAPTIVE_MODES:
        return None
    if "chem_enzy_onestep" not in allocation.source_budgets:
        return None
    if int(allocation.source_budgets.get("chem_enzy_onestep") or 0) > 0:
        return None
    if _chem_enzy_onestep_budgeted_context(context):
        return None
    snapshot = _candidate_action_quality_snapshot(actions)
    budget = _chem_enzy_onestep_adaptive_budget(total_budget=total_budget)
    reason = ""
    min_actions = _chem_enzy_onestep_adaptive_min_actions()
    reject_fraction_threshold = _chem_enzy_onestep_adaptive_reject_fraction()
    if min_actions > 0 and int(snapshot.get("strong_count") or 0) < min_actions:
        reason = "low_strong_action_count"
    elif (
        int(snapshot.get("gate_count") or 0) > 0
        and float(snapshot.get("gate_reject_fraction") or 0.0) >= reject_fraction_threshold
    ):
        reason = "high_gate_reject_fraction"
    triggered = bool(reason and budget > 0)
    return {
        "triggered": triggered,
        "mode": mode,
        "reason": reason,
        "budget": int(budget if triggered else 0),
        "min_strong_actions": int(min_actions),
        "reject_fraction_threshold": float(reject_fraction_threshold),
        **snapshot,
    }


def _candidate_action_quality_snapshot(actions: list[CandidateAction]) -> dict[str, Any]:
    seen: set[str] = set()
    valid_count = 0
    strong_count = 0
    weak_count = 0
    gate_count = 0
    gate_reject_count = 0
    invalid_count = 0
    dedupe_dropped = 0
    severe_flags = {"no_reactants", "no_main_reactant", "product_mismatch", "self_loop"}
    for action in actions:
        key = action.canonical_key
        if key in seen:
            dedupe_dropped += 1
            continue
        seen.add(key)
        flags = set(action.validity_flags or ())
        if "no_reactants" in flags or "no_main_reactant" in flags:
            invalid_count += 1
            continue
        valid_count += 1
        gate = action.metadata.get("proposal_gate") if isinstance(action.metadata, dict) else None
        gate_reject = False
        if isinstance(gate, dict):
            gate_count += 1
            gate_reject = bool(gate.get("hard_reject")) or str(gate.get("decision") or "").lower() == "reject"
            if gate_reject:
                gate_reject_count += 1
        if flags & severe_flags or gate_reject:
            weak_count += 1
        else:
            strong_count += 1
    return {
        "raw_count": int(len(actions)),
        "deduped_count": int(len(seen)),
        "valid_count": int(valid_count),
        "strong_count": int(strong_count),
        "weak_count": int(weak_count),
        "invalid_count": int(invalid_count),
        "dedupe_dropped": int(dedupe_dropped),
        "gate_count": int(gate_count),
        "gate_reject_count": int(gate_reject_count),
        "gate_reject_fraction": round(gate_reject_count / max(gate_count, 1), 6),
    }


def _chem_enzy_onestep_adaptive_budget(*, total_budget: int) -> int:
    default_budget = min(max(1, int(total_budget or 1)), 4)
    max_budget = _env_int("AUTOPLANNER_CHEMENZY_ONESTEP_ADAPTIVE_MAX_BUDGET", default_budget)
    return max(0, min(max(1, int(total_budget or 1)), max_budget))


def _chem_enzy_onestep_adaptive_min_actions() -> int:
    return max(0, _env_int("AUTOPLANNER_CHEMENZY_ONESTEP_ADAPTIVE_MIN_ACTIONS", 2))


def _chem_enzy_onestep_adaptive_reject_fraction() -> float:
    return min(1.0, max(0.0, _env_float("AUTOPLANNER_CHEMENZY_ONESTEP_ADAPTIVE_REJECT_FRACTION", 0.75)))


def _chem_enzy_onestep_budgeted_context(context: ProposalContext | None) -> bool:
    if context is None:
        return False
    route_metadata = dict(getattr(context, "route_metadata", {}) or {})
    constraints = dict(getattr(context, "constraints", {}) or {})
    return bool(
        route_metadata.get("stock_rescue_retry")
        or route_metadata.get("force_chem_enzy_onestep")
        or constraints.get("force_chem_enzy_onestep")
    )


def _chem_enzy_onestep_budgeted_by_mode(context: ProposalContext | None) -> bool:
    mode = _chem_enzy_onestep_route_mode()
    if mode in CHEMENZY_ONESTEP_EAGER_MODES:
        return True
    if mode in CHEMENZY_ONESTEP_DISABLED_MODES:
        return False
    if mode in (CHEMENZY_ONESTEP_FALLBACK_MODES | CHEMENZY_ONESTEP_ADAPTIVE_MODES):
        return _chem_enzy_onestep_budgeted_context(context)
    return True


def _chem_enzy_onestep_route_mode() -> str:
    return str(os.environ.get(CHEMENZY_ONESTEP_ROUTE_MODE_ENV) or "eager").strip().lower()


def _filter_floors_allowed_by_allocation(
    floors: dict[str, int],
    allocation: SourceAllocation,
) -> dict[str, int]:
    if not floors:
        return {}
    if not _bridge_gate_suppresses_enzymatic_fallback(allocation):
        return floors
    return {
        source: floor
        for source, floor in floors.items()
        if source_group(source) not in {"enzymatic", "rhea_retrorules"}
    }


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


def _bridge_hit_source_budget_floors(
    allocation: SourceAllocation,
    *,
    sources: list[str],
    product: str = "",
) -> dict[str, int]:
    hits = _allocation_bridge_hits(allocation)
    if hits <= 0:
        return {}
    available = set(sources)
    floors: dict[str, int] = {}
    semisynthesis_applicable = _semisynthesis_rescue_applicable(product)
    chemical_anchor_applicable = _chemical_anchor_rescue_applicable(product)
    for source, floor in (
        ("semisynthesis_rescue", _semisynthesis_rescue_min_budget()),
        ("chemical_anchor_rescue", _chemical_anchor_rescue_min_budget()),
        ("enzyme_precedent", 2),
        ("v3_retrieval", 2),
        ("enzyformer", 1),
        ("enzexpand", 1),
        ("chem_enzy_bionav", 1),
    ):
        if source == "semisynthesis_rescue" and not semisynthesis_applicable:
            continue
        if source == "chemical_anchor_rescue" and not chemical_anchor_applicable:
            continue
        if source in available:
            floors[source] = floor
    return floors


def _allocation_bridge_hits(allocation: SourceAllocation) -> int:
    flags = allocation.molecule_flags or {}
    try:
        return int(flags.get("bridge_gate_hits") or 0)
    except (TypeError, ValueError):
        return 0


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
    product: str = "",
) -> dict[str, int]:
    if not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_SOURCE_FLOORS", True):
        return {}
    if context is None or int(total_budget or 0) <= 0:
        return {}
    depth = int(getattr(context, "depth", 0) or 0)
    route_metadata = dict(getattr(context, "route_metadata", {}) or {})
    if route_metadata.get("stock_rescue_retry"):
        return _stock_rescue_source_budget_floors(
            context,
            sources=sources,
            total_budget=total_budget,
            product=product,
        )
    if depth != 0:
        return {}
    available = set(sources)
    floors: dict[str, int] = {}
    reaction_type = str(getattr(context, "reaction_type", "") or "").lower()
    ec1 = int(getattr(context, "ec1", 0) or 0)
    semisynthesis_applicable = _semisynthesis_rescue_applicable(product)
    chemical_anchor_applicable = _chemical_anchor_rescue_applicable(product)
    if "native_replay" in available and _native_replay_enabled():
        floors["native_replay"] = _native_replay_min_budget()
    if ec1:
        if "semisynthesis_rescue" in available and semisynthesis_applicable:
            floors["semisynthesis_rescue"] = _semisynthesis_rescue_min_budget()
        if "chemical_anchor_rescue" in available and chemical_anchor_applicable:
            floors["chemical_anchor_rescue"] = _chemical_anchor_rescue_min_budget()
        if "chem_enzy_bionav" in available:
            floors["chem_enzy_bionav"] = 2
        if "enzyme_precedent" in available:
            floors["enzyme_precedent"] = 3
        if "v3_retrieval" in available:
            floors["v3_retrieval"] = 3
        if "enzyformer" in available:
            floors["enzyformer"] = 1
        if "enzexpand" in available:
            floors["enzexpand"] = 1
        return floors
    if _is_chemical_reaction_context(reaction_type):
        if "semisynthesis_rescue" in available and semisynthesis_applicable:
            floors["semisynthesis_rescue"] = _semisynthesis_rescue_min_budget()
        if "chemical_anchor_rescue" in available and chemical_anchor_applicable:
            floors["chemical_anchor_rescue"] = _chemical_anchor_rescue_min_budget()
        if "retrochimera" in available:
            floors["retrochimera"] = 3
        if "chem_enzy_graphfp_fusion" in available:
            floors["chem_enzy_graphfp_fusion"] = _chem_enzy_graphfp_fusion_min_budget()
        if "template_relevance" in available:
            floors["template_relevance"] = _template_relevance_min_budget()
        if "chem_enzy_onestep" in available and _chem_enzy_onestep_budgeted_by_mode(context):
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
    product: str = "",
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
    semisynthesis_applicable = _semisynthesis_rescue_applicable(product)
    chemical_anchor_applicable = _chemical_anchor_rescue_applicable(product)
    if enzymatic_route:
        if "semisynthesis_rescue" in available and semisynthesis_applicable:
            floors["semisynthesis_rescue"] = _semisynthesis_rescue_min_budget()
        if "chemical_anchor_rescue" in available and chemical_anchor_applicable:
            floors["chemical_anchor_rescue"] = _chemical_anchor_rescue_min_budget()
        for source, floor in (
            ("chem_enzy_bionav", 2),
            ("enzyme_precedent", 3),
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
        ("semisynthesis_rescue", _semisynthesis_rescue_min_budget() if semisynthesis_applicable else 0),
        ("chemical_anchor_rescue", _chemical_anchor_rescue_min_budget() if chemical_anchor_applicable else 0),
        ("retrochimera", 2),
        ("chem_enzy_graphfp_fusion", _chem_enzy_graphfp_fusion_min_budget()),
        ("template_relevance", _template_relevance_min_budget()),
        (
            "chem_enzy_onestep",
            _chem_enzy_onestep_min_budget() if _chem_enzy_onestep_budgeted_by_mode(context) else 0,
        ),
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


def _protect_applicable_semisynthesis_floor(
    floors: dict[str, int],
    *,
    requested_floors: dict[str, int],
    sources: list[str],
    total_budget: int,
    product: str = "",
) -> dict[str, int]:
    source = "semisynthesis_rescue"
    if source not in sources or source not in requested_floors:
        return floors
    if not _semisynthesis_rescue_applicable(product):
        return floors
    total_budget = max(1, int(total_budget or 1))
    target_floor = min(total_budget, max(1, int(requested_floors.get(source) or 1)))
    out = dict(floors)
    out[source] = max(int(out.get(source) or 0), target_floor)
    while sum(out.values()) > total_budget:
        candidates = [item for item in out if item != source and int(out.get(item) or 0) > 0]
        if not candidates:
            out[source] = min(int(out.get(source) or 0), total_budget)
            break
        victim = max(candidates, key=lambda key: (out[key], -sources.index(key) if key in sources else 0))
        out[victim] -= 1
        if out[victim] <= 0:
            out.pop(victim, None)
    return out


def _protect_applicable_chemical_anchor_floor(
    floors: dict[str, int],
    *,
    requested_floors: dict[str, int],
    sources: list[str],
    total_budget: int,
    product: str = "",
) -> dict[str, int]:
    source = "chemical_anchor_rescue"
    if source not in sources or source not in requested_floors:
        return floors
    if not _chemical_anchor_rescue_applicable(product):
        return floors
    total_budget = max(1, int(total_budget or 1))
    target_floor = min(total_budget, max(1, int(requested_floors.get(source) or 1)))
    out = dict(floors)
    out[source] = max(int(out.get(source) or 0), target_floor)
    while sum(out.values()) > total_budget:
        candidates = [item for item in out if item != source and int(out.get(item) or 0) > 0]
        if not candidates:
            out[source] = min(int(out.get(source) or 0), total_budget)
            break
        victim = max(candidates, key=lambda key: (out[key], -sources.index(key) if key in sources else 0))
        out[victim] -= 1
        if out[victim] <= 0:
            out.pop(victim, None)
    return out


def _filter_product_applicable_sources(product: str, sources: list[str]) -> list[str]:
    out = list(sources)
    if "semisynthesis_rescue" in out and not _semisynthesis_rescue_applicable(product):
        out = [source for source in out if source != "semisynthesis_rescue"]
    if "chemical_anchor_rescue" in out and not _chemical_anchor_rescue_applicable(product):
        out = [source for source in out if source != "chemical_anchor_rescue"]
    return out


def _semisynthesis_rescue_applicable(product: str) -> bool:
    key = canonical_smiles(product) or str(product or "")
    if not key:
        return False
    cached = _SEMISYNTHESIS_RESCUE_APPLICABILITY_CACHE.get(key)
    if cached is not None:
        return bool(cached)
    try:
        from cascade_planner.baselines.semisynthesis_rescue import (
            SemisynthesisRescueConfig,
            semisynthesis_rescue_routes,
        )

        applicable = bool(
            semisynthesis_rescue_routes(
                product,
                config=SemisynthesisRescueConfig(enabled=True, max_routes=1),
            )
        )
    except Exception:
        applicable = False
    _SEMISYNTHESIS_RESCUE_APPLICABILITY_CACHE[key] = bool(applicable)
    return bool(applicable)


def _chemical_anchor_rescue_applicable(product: str) -> bool:
    key = canonical_smiles(product) or str(product or "")
    if not key:
        return False
    cached = _CHEMICAL_ANCHOR_RESCUE_APPLICABILITY_CACHE.get(key)
    if cached is not None:
        return bool(cached)
    try:
        from cascade_planner.baselines.chemical_anchor_rescue import (
            ChemicalAnchorRescueConfig,
            chemical_anchor_rescue_routes,
        )

        applicable = bool(
            chemical_anchor_rescue_routes(
                product,
                config=ChemicalAnchorRescueConfig(enabled=True, max_routes=1),
            )
        )
    except Exception:
        applicable = False
    _CHEMICAL_ANCHOR_RESCUE_APPLICABILITY_CACHE[key] = bool(applicable)
    return bool(applicable)


def _cap_source_request_k(source: str, request_top_k: int) -> int:
    request_top_k = max(1, int(request_top_k or 1))
    if not _env_truthy_default("AUTOPLANNER_ROUTE_TREE_V4_REQUEST_CAPS", True):
        return request_top_k
    caps = {
        "v3_retrieval": 16,
        "enzyme_precedent": 24,
        "enzyformer": 16,
        "enzexpand": 12,
        "chem_enzy_bionav": 20,
        "retrorules": 12,
        "retrochimera": 16,
        "semisynthesis_rescue": 8,
        "chemical_anchor_rescue": 4,
        "chem_enzy_graphfp_fusion": 50,
        "template_relevance": 20,
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


def _attach_runtime_condition_prediction(action: CandidateAction) -> CandidateAction:
    if not _route_tree_condition_prediction_enabled():
        return action
    if _action_has_condition_prediction(action):
        return action
    if _route_tree_condition_prediction_chemical_only() and _route_tree_action_is_enzymatic(action):
        return action
    rxn = _condition_rxn_smiles_from_action(action)
    if ">>" not in rxn:
        return _annotate_condition_prediction_failure(action, "missing_reaction_smiles")
    try:
        predictor = _route_tree_condition_predictor()
        top_k = max(1, _env_int("AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION_TOPK", 1))
        if hasattr(predictor, "predict"):
            raw = predictor.predict(rxn, top_k=top_k)
        elif hasattr(predictor, "get_n_conditions"):
            raw = predictor.get_n_conditions(rxn, n=top_k, return_scores=True)
        else:
            raw = predictor(rxn, top_k=top_k)
        rows = _normalize_runtime_condition_rows(raw)
    except Exception as exc:
        return _annotate_condition_prediction_failure(action, f"condition_prediction_failed:{type(exc).__name__}")
    if not rows:
        return _annotate_condition_prediction_failure(action, "condition_prediction_empty")
    metadata = dict(action.metadata)
    metadata["condition_predictions"] = rows
    model = _route_tree_condition_model()
    metadata["condition_prediction_model"] = model
    metadata["condition_prediction_source"] = "ChemEnzyRCR" if model == "rcr" else model
    metadata["condition_prediction_trust"] = "weak_runtime_prediction"
    metadata["condition_prediction_enabled_by"] = "route_tree_runtime"
    return _replace_action_metadata(action, metadata)


def _action_has_condition_prediction(action: CandidateAction) -> bool:
    metadata = action.metadata or {}
    if metadata.get("condition_predictions"):
        return True
    return any(value not in (None, "") for value in (action.T, action.pH, action.solvent))


def _annotate_condition_prediction_failure(action: CandidateAction, reason: str) -> CandidateAction:
    metadata = dict(action.metadata or {})
    issues = [str(issue) for issue in metadata.get("condition_prediction_issues") or []]
    issues.append(str(reason))
    metadata["condition_prediction_issues"] = sorted(set(issues))
    metadata["condition_prediction_model"] = _route_tree_condition_model()
    metadata["condition_prediction_enabled_by"] = "route_tree_runtime"
    return _replace_action_metadata(action, metadata)


def _replace_action_metadata(action: CandidateAction, metadata: dict[str, Any]) -> CandidateAction:
    return CandidateAction(
        product=action.product,
        reactants=action.reactants,
        main_reactant=action.main_reactant,
        aux_reactants=action.aux_reactants,
        rxn_smiles=action.rxn_smiles,
        source=action.source,
        raw_score=action.raw_score,
        rank=action.rank,
        reaction_type=action.reaction_type,
        ec=action.ec,
        catalyst=action.catalyst,
        T=action.T,
        pH=action.pH,
        solvent=action.solvent,
        metadata=metadata,
        validity_flags=action.validity_flags,
    )


def _route_tree_condition_prediction_enabled() -> bool:
    return _env_truthy_default("AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION", False)


def _route_tree_condition_prediction_chemical_only() -> bool:
    return _env_truthy_default("AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION_CHEMICAL_ONLY", True)


def _route_tree_condition_model() -> str:
    return str(os.environ.get("AUTOPLANNER_ROUTE_TREE_CONDITION_MODEL") or "rcr").strip().lower() or "rcr"


def _route_tree_condition_vendor_root() -> Path:
    return Path(str(os.environ.get("AUTOPLANNER_ROUTE_TREE_CONDITION_VENDOR_ROOT") or "vendor/ChemEnzyRetroPlanner"))


def _route_tree_condition_predictor() -> Any:
    injected = globals().get("_ROUTE_TREE_CONDITION_PREDICTOR_OVERRIDE")
    if injected is not None:
        return injected
    model = _route_tree_condition_model()
    vendor_root = _route_tree_condition_vendor_root()
    key = (str(vendor_root.resolve() if vendor_root.exists() else vendor_root), model)
    if key in _ROUTE_TREE_CONDITION_PREDICTOR_CACHE:
        return _ROUTE_TREE_CONDITION_PREDICTOR_CACHE[key]
    try:
        from cascade_planner.cascade_search.proposals import _cached_condition_predictor
    except Exception as exc:
        raise RuntimeError(f"condition predictor import failed: {exc}") from exc
    predictor = _cached_condition_predictor(vendor_root, model)
    _ROUTE_TREE_CONDITION_PREDICTOR_CACHE[key] = predictor
    return predictor


def _condition_rxn_smiles_from_action(action: CandidateAction) -> str:
    if ">>" in str(action.rxn_smiles or ""):
        return str(action.rxn_smiles)
    reactants = [str(smi) for smi in action.reactants if smi]
    product = str(action.product or "")
    if reactants and product:
        return ".".join(reactants) + f">>{product}"
    return str(action.rxn_smiles or "")


def _route_tree_action_is_enzymatic(action: CandidateAction) -> bool:
    source = str(action.source or "").lower()
    return bool(action.ec) or source in {
        "enzyformer",
        "enzexpand",
        "retrorules",
        "rhea",
        "rhea_template",
        "rhea_retrorules",
        "enzyme_precedent",
        "v3_retrieval",
        "retrieval",
        "enzymatic",
        "chem_enzy_bionav",
    }


def _normalize_runtime_condition_rows(raw: Any) -> list[dict[str, Any]]:
    try:
        from cascade_planner.cascade_search.proposals import _normalize_condition_prediction_rows

        rows = _normalize_condition_prediction_rows(raw)
    except Exception:
        rows = _fallback_normalize_condition_rows(raw)
    for row in rows:
        row.setdefault("source", "route_tree_condition_prediction")
        row.setdefault("condition_model", _route_tree_condition_model())
    return rows


def _fallback_normalize_condition_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, tuple) and len(raw) == 2:
        combos, scores = raw
        rows = []
        for combo, score in zip(combos or [], scores or []):
            if isinstance(combo, dict):
                row = dict(combo)
            elif isinstance(combo, (list, tuple)) and len(combo) >= 4:
                row = {"Temperature": combo[0], "Solvent": combo[1], "Reagent": combo[2], "Catalyst": combo[3]}
            else:
                continue
            row.setdefault("Score", score)
            rows.append(row)
        return rows
    if isinstance(raw, dict):
        return [dict(raw)]
    if isinstance(raw, list):
        rows = []
        for item in raw:
            if isinstance(item, dict):
                rows.append(dict(item))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                rows.append({"Temperature": item[0], "Solvent": item[1], "Reagent": item[2], "Catalyst": item[3]})
        return rows
    return []


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


def _semisynthesis_rescue_min_budget() -> int:
    try:
        return max(1, int(os.environ.get("AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET") or 2))
    except ValueError:
        return 2


def _chemical_anchor_rescue_min_budget() -> int:
    try:
        return max(1, int(os.environ.get("AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET") or 2))
    except ValueError:
        return 2


def _chem_enzy_graphfp_fusion_min_budget() -> int:
    try:
        return max(1, int(os.environ.get("AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET") or 4))
    except ValueError:
        return 4


def _template_relevance_min_budget() -> int:
    try:
        return max(1, int(os.environ.get("AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET") or 2))
    except ValueError:
        return 2


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


def _enzyme_precedent_enabled() -> bool:
    return _env_truthy_default("AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL", False)


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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return float(default)


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
