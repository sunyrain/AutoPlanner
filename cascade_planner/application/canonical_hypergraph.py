"""One immutable, incrementally derived retrosynthesis hypergraph for V4.

All proposal producers enter through the same admission path.  Worker facts are
replayed from immutable outcomes before ingestion.  The graph owns no model or
evidence authority itself; it records host-validated bindings and publishes one
revision through :class:`RunKernel`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import threading
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger

from cascade_planner.application.condition_predictions import (
    normalize_condition_predictions,
)
from cascade_planner.application.deficit_frontier import (
    compile_deficit_frontier,
    frontier_scientific_projection,
)
from cascade_planner.application.fact_lifecycle import (
    fact_subject,
    fact_subject_digest,
    graph_fact_lifecycle_state,
    validate_fact_lifecycle_event,
)
from cascade_planner.application.canonical_identity import (
    hypothesis_identity,
    molecule_identity,
    reaction_edge_identity,
    route_family_identity,
    source_binding_identity,
    stock_observation_identity,
)
from cascade_planner.application.run_kernel import RunKernel
from cascade_planner.application.proof_policy import (
    ProofPolicy,
    stitch_edge_proof,
    stitch_leaf_stock_proof,
)
from cascade_planner.application.route_innovations import (
    merge_route_innovations,
    normalize_route_innovation,
)
from cascade_planner.application.retrosynthesis_workers import (
    materialization_commands_for_proposals,
)
from cascade_planner.application.worker_runtime import WorkerResult, WorkerRuntime
from cascade_planner.runtime.artifact_store import ArtifactReferenceError


RDLogger.DisableLog("rdApp.*")
CANONICAL_HYPERGRAPH_SCHEMA = "canonical_retrosynthesis_hypergraph.v1"
CANONICAL_HYPERGRAPH_DELTA_SCHEMA = "canonical_hypergraph_delta.v1"
CANONICAL_INGESTION_REPORT_SCHEMA = "canonical_hypergraph_ingestion_report.v1"
_ORIGIN_KINDS = {
    "codex",
    "codex_global_director",
    "chemenzy",
    "template",
    "literature",
    "literature_visual_extraction",
    "literature_source_route",
    "literature_replay",
    "manual",
    "host_product_grounded_repair",
    "self_evo_patent_template",
    "biocatalysis_hypothesis",
    "mechanism_hypothesis",
}


class CanonicalHypergraphError(RuntimeError):
    """Canonical graph input, replay, or publication failed closed."""


@dataclass(frozen=True, slots=True)
class CanonicalIngestionBatch:
    worker_results: tuple[WorkerResult | Mapping[str, Any], ...] = ()
    global_plans: tuple[Mapping[str, Any], ...] = ()
    hypotheses: tuple[Mapping[str, Any], ...] = ()
    route_families: tuple[Mapping[str, Any], ...] = ()
    fact_lifecycle_events: tuple[Mapping[str, Any], ...] = ()
    action_signals: tuple[Mapping[str, Any], ...] = ()
    prior_attempts: Mapping[str, int] = field(default_factory=dict)
    recompute_derived: bool = False


class CanonicalHypergraphStore:
    """Publish one graph revision and one delta through the campaign kernel."""

    def __init__(self, kernel: RunKernel) -> None:
        self.kernel = kernel
        self._lock = threading.RLock()
        run_digest = hashlib.sha256(kernel.spec.run_id.encode("utf-8")).hexdigest()
        self.pointer_name = f"g/{run_digest[:24]}/latest"

    def load(self) -> dict[str, Any]:
        pointer_path = self.kernel.artifacts.pointers_root / f"{self.pointer_name}.json"
        try:
            ref, _ = self.kernel.artifacts.load_pointer(self.pointer_name)
        except ArtifactReferenceError as exc:
            if not pointer_path.is_file():
                return _empty_graph(self.kernel)
            raise CanonicalHypergraphError("canonical_graph_pointer_invalid") from exc
        value = self.kernel.artifacts.read_json(ref)
        if not isinstance(value, Mapping):
            raise CanonicalHypergraphError("canonical_graph_artifact_not_object")
        graph = dict(value)
        _validate_graph(graph, expected_run_id=self.kernel.spec.run_id)
        if int(graph.get("revision") or 0) != self.kernel.state.graph_revision:
            raise CanonicalHypergraphError("canonical_graph_kernel_revision_mismatch")
        if ref.sha256 != str(graph.get("artifact_sha256") or ref.sha256):
            raise CanonicalHypergraphError("canonical_graph_artifact_binding_invalid")
        return graph

    def apply(
        self,
        batch: CanonicalIngestionBatch,
        *,
        worker_runtime: WorkerRuntime | None = None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self._lock:
            previous = self.load()
            replayed: list[WorkerResult] = []
            if batch.worker_results and worker_runtime is None:
                raise CanonicalHypergraphError("worker_runtime_required_for_result_replay")
            for value in batch.worker_results:
                assert worker_runtime is not None
                replayed.append(worker_runtime.replay_result(value))
            graph, report = compile_canonical_hypergraph_revision(
                previous,
                batch=batch,
                replayed_worker_results=replayed,
                acceptance_spec=self.kernel.spec.acceptance,
            )
            if report["changed"] is not True:
                return {**report, "graph": previous, "graph_ref": {}}

            ref = self.kernel.artifacts.put_json(
                graph,
                logical_name="canonical_hypergraph.json",
                producer="autoplanner.canonical_hypergraph",
            )
            evidence_revision = self.kernel.state.evidence_revision + (
                1 if report["evidence_changed"] else 0
            )
            self.kernel.publish_graph_revision(
                int(graph["revision"]),
                graph_sha256=ref.sha256,
                evidence_revision=evidence_revision,
                idempotency_key=f"graph:publish:{idempotency_key}",
            )
            self.kernel.artifacts.write_pointer(
                self.pointer_name,
                ref,
                metadata={
                    "run_id": self.kernel.spec.run_id,
                    "revision": graph["revision"],
                    "scientific_sha256": graph["scientific_sha256"],
                },
            )
            self.kernel.index.index_artifact(
                run_id=self.kernel.spec.run_id,
                artifact_id="canonical_hypergraph",
                ref=ref,
                revision=int(graph["revision"]),
                authority_scope="canonical_hypergraph_revision",
            )
            return {**report, "graph": graph, "graph_ref": ref.to_dict()}

    def materialization_commands(
        self,
        proposals: Iterable[Mapping[str, Any]],
    ) -> tuple[Any, ...]:
        """Compile proposals with current graph duplicate/cycle knowledge."""
        graph = self.load()
        rows = []
        for value in proposals:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            edge_id, _ = reaction_edge_identity(
                row.get("product_smiles"),
                row.get("precursor_smiles") or row.get("reactant_smiles") or [],
            )
            if edge_id and edge_id in graph["edges"]:
                continue
            rows.append(row)
        ancestor_map = {
            str(row.get("product_smiles") or ""): _ancestor_smiles_for_product(
                graph,
                row.get("product_smiles"),
            )
            for row in rows
        }
        return materialization_commands_for_proposals(
            rows,
            run_id=self.kernel.spec.run_id,
            input_revision=self.kernel.state.graph_revision,
            dependency_revisions={
                "graph_revision": self.kernel.state.graph_revision,
                "evidence_revision": self.kernel.state.evidence_revision,
            },
            existing_edge_digests=(
                str(edge.get("edge_digest") or "")
                for edge in graph["edges"].values()
            ),
            ancestor_smiles_by_product=ancestor_map,
        )

    def frontier_materialization_commands(
        self,
        hypothesis_ids: Iterable[str] = (),
    ) -> tuple[Any, ...]:
        graph = self.load()
        selected_ids = {
            str(value) for value in hypothesis_ids if str(value).strip()
        }
        proposals: list[dict[str, Any]] = []
        hypotheses = sorted(
            graph["hypotheses"].values(),
            key=lambda row: (
                -float(row.get("frontier_priority") or 0.0),
                str(row.get("product_smiles") or ""),
                str(row.get("hypothesis_id") or ""),
            ),
        )
        for hypothesis in hypotheses:
            if selected_ids and str(hypothesis.get("hypothesis_id") or "") not in selected_ids:
                continue
            if hypothesis.get("status") != "frontier_candidate":
                continue
            origins = list(hypothesis.get("origin_records") or [{}])
            for origin in origins:
                proposals.append(
                    {
                        "product_smiles": hypothesis["product_smiles"],
                        "precursor_smiles": hypothesis["precursor_smiles"],
                        "condition_predictions": list(
                            hypothesis.get("condition_predictions") or []
                        ),
                        "route_innovations": list(
                            hypothesis.get("route_innovations") or []
                        ),
                        **dict(origin),
                    }
                )
        return self.materialization_commands(proposals)

    def full_recompute_oracle(self) -> dict[str, Any]:
        current = self.load()
        return full_recompute_canonical_hypergraph(
            current,
            acceptance_spec=self.kernel.spec.acceptance,
        )


def compile_canonical_hypergraph_revision(
    previous: Mapping[str, Any],
    *,
    batch: CanonicalIngestionBatch,
    replayed_worker_results: Iterable[WorkerResult] = (),
    acceptance_spec: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge one append-only fact batch and recompute only dirty dependencies."""
    graph = _mutable_graph(previous)
    old_scientific = str(previous.get("scientific_sha256") or "")
    next_revision = int(previous.get("revision") or 0) + 1
    dirty: set[str] = set()
    rejected: list[dict[str, Any]] = []
    evidence_changed = False
    operational_changed = False
    force_full_derived_recompute = False

    target_id = str(graph["target_molecule_id"])
    route_aliases = _route_aliases(graph)
    for plan in sorted(batch.global_plans, key=_digest):
        _ingest_global_plan(
            graph,
            plan,
            target_id=target_id,
            route_aliases=route_aliases,
            dirty=dirty,
            rejected=rejected,
        )
    for route in sorted(batch.route_families, key=_digest):
        _ingest_route_family(
            graph,
            route,
            target_id=target_id,
            route_aliases=route_aliases,
            dirty=dirty,
        )
    for hypothesis in sorted(batch.hypotheses, key=_digest):
        _ingest_hypothesis(
            graph,
            hypothesis,
            route_aliases=route_aliases,
            dirty=dirty,
            rejected=rejected,
        )
    worker_order = {
        "materialize_candidate": 0,
        "record_condition_predictions": 1,
        "discover_sources": 2,
        "extract_exact_source": 3,
        "validate_reaction": 4,
        "audit_deep_leaf_stock": 5,
        "detect_source_conflicts": 6,
    }
    ordered_results = sorted(
        replayed_worker_results,
        key=lambda result: (
            worker_order.get(result.worker_type, 99),
            result.worker_type,
            result.command_id,
        ),
    )
    for result in ordered_results:
        changed_evidence = _ingest_worker_result(
            graph,
            result,
            route_aliases=route_aliases,
            dirty=dirty,
            rejected=rejected,
        )
        evidence_changed = evidence_changed or changed_evidence
    for event in sorted(batch.fact_lifecycle_events, key=_digest):
        evidence_changed = (
            _ingest_fact_lifecycle_event(
                graph,
                event,
                dirty=dirty,
                rejected=rejected,
            )
            or evidence_changed
        )
    for signal in sorted(batch.action_signals, key=_digest):
        operational_changed = (
            _ingest_action_signal(
                graph,
                signal,
                dirty=dirty,
                rejected=rejected,
            )
            or operational_changed
        )

    if not dirty and batch.recompute_derived:
        oracle = full_recompute_canonical_hypergraph(
            graph,
            acceptance_spec=acceptance_spec,
        )
        if oracle["scientific_sha256"] != previous.get("scientific_sha256"):
            graph = oracle
            dirty.update(_all_entity_ids(graph) or {str(graph["target_molecule_id"])})
            force_full_derived_recompute = True
    if not dirty:
        return dict(previous), _report(
            previous,
            dirty=(),
            rejected=rejected,
            evidence_changed=False,
            changed=False,
        )

    dirty.update(_dirty_ancestor_closure(graph, dirty))
    dirty.update(_routes_affected_by(graph, dirty))
    route_changed = any(value.startswith("route-family:") for value in dirty)
    if route_changed:
        dirty.update(str(value) for value in graph["route_families"])

    _refresh_molecules(graph, dirty=dirty)
    _refresh_routes(graph, dirty=dirty, acceptance_spec=acceptance_spec)
    _mark_dominated_routes(graph)
    graph["dependency_index"] = _dependency_index(graph)
    topology_sha256 = _topology_digest(graph)
    graph["topology_sha256"] = topology_sha256
    frontier = compile_deficit_frontier(
        {**graph, "scientific_sha256": topology_sha256},
        acceptance_spec=acceptance_spec,
        prior_attempts=batch.prior_attempts,
        previous_frontier=(
            {} if force_full_derived_recompute else dict(previous.get("deficit_frontier") or {})
        ),
        dirty_entity_ids=None if force_full_derived_recompute else dirty,
    )
    graph["deficit_frontier"] = frontier
    graph["portfolio_ranking"] = _portfolio_ranking(graph)
    graph["entity_revisions"] = {
        **dict(graph.get("entity_revisions") or {}),
        **{entity_id: next_revision for entity_id in dirty},
    }
    graph["revision"] = next_revision
    graph["previous_scientific_sha256"] = old_scientific
    graph["delta"] = {
        "schema_version": CANONICAL_HYPERGRAPH_DELTA_SCHEMA,
        "revision": next_revision,
        "dirty_entity_ids": sorted(dirty),
        "dirty_entity_count": len(dirty),
        "total_entity_count": _entity_count(graph),
        "recomputed_fraction": round(len(dirty) / max(1, _entity_count(graph)), 6),
        "rejected": rejected,
    }
    graph["scientific_sha256"] = _scientific_digest(graph)
    if graph["scientific_sha256"] == old_scientific and not operational_changed:
        return dict(previous), _report(
            previous,
            dirty=(),
            rejected=rejected,
            evidence_changed=False,
            changed=False,
        )
    graph["content_sha256"] = _graph_content_digest(graph)
    return graph, _report(
        graph,
        dirty=dirty,
        rejected=rejected,
        evidence_changed=evidence_changed,
        changed=True,
    )


def full_recompute_canonical_hypergraph(
    value: Mapping[str, Any],
    *,
    acceptance_spec: Any,
) -> dict[str, Any]:
    """Rebuild all derived fields as the incremental correctness oracle."""
    graph = _mutable_graph(value)
    all_entities = _all_entity_ids(graph)
    _rebuild_all_adjacency(graph)
    _refresh_molecules(graph, dirty=all_entities)
    _refresh_routes(graph, dirty=all_entities, acceptance_spec=acceptance_spec)
    _mark_dominated_routes(graph)
    graph["dependency_index"] = _dependency_index(graph)
    graph["topology_sha256"] = _topology_digest(graph)
    graph["deficit_frontier"] = compile_deficit_frontier(
        {**graph, "scientific_sha256": graph["topology_sha256"]},
        acceptance_spec=acceptance_spec,
    )
    graph["portfolio_ranking"] = _portfolio_ranking(graph)
    graph["scientific_sha256"] = _scientific_digest(graph)
    return graph


def canonical_scientific_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_HYPERGRAPH_SCHEMA,
        "run_id": str(value.get("run_id") or ""),
        "target_molecule_id": str(value.get("target_molecule_id") or ""),
        "molecules": _sorted_mapping(value.get("molecules")),
        "edges": _sorted_mapping(value.get("edges")),
        "source_bindings": _sorted_mapping(value.get("source_bindings")),
        "exact_records": _sorted_mapping(value.get("exact_records")),
        "procedure_records": _sorted_mapping(value.get("procedure_records")),
        "fact_lifecycle_events": _sorted_mapping(value.get("fact_lifecycle_events")),
        "stock_observations": _sorted_mapping(value.get("stock_observations")),
        "route_families": _sorted_mapping(value.get("route_families")),
        "hypotheses": _sorted_mapping(value.get("hypotheses")),
        "conflicts": _sorted_mapping(value.get("conflicts")),
        "deficit_frontier": frontier_scientific_projection(
            dict(value.get("deficit_frontier") or {})
        ),
        "portfolio_ranking": list(value.get("portfolio_ranking") or []),
    }


def _empty_graph(kernel: RunKernel) -> dict[str, Any]:
    target_id, target = molecule_identity(kernel.spec.target_smiles)
    molecule = _molecule_record(target_id, target)
    graph = {
        "schema_version": CANONICAL_HYPERGRAPH_SCHEMA,
        "run_id": kernel.spec.run_id,
        "target_name": kernel.spec.target_name,
        "target_molecule_id": target_id,
        "revision": 0,
        "previous_scientific_sha256": "",
        "molecules": {target_id: molecule},
        "edges": {},
        "source_bindings": {},
        "source_aliases": {},
        "exact_records": {},
        "procedure_records": {},
        "fact_lifecycle_events": {},
        "stock_observations": {},
        "route_families": {},
        "hypotheses": {},
        "conflicts": {},
        "action_signals": {},
        "dependency_index": {"routes_by_entity": {}},
        "deficit_frontier": {},
        "portfolio_ranking": [],
        "entity_revisions": {target_id: 0},
        "delta": {
            "schema_version": CANONICAL_HYPERGRAPH_DELTA_SCHEMA,
            "revision": 0,
            "dirty_entity_ids": [],
            "dirty_entity_count": 0,
            "total_entity_count": 1,
            "recomputed_fraction": 0.0,
            "rejected": [],
        },
        "semantics": {
            "single_canonical_hypergraph": True,
            "blackboard_is_projection_only": True,
            "all_proposals_use_one_ingestion_path": True,
            "facts_require_host_replayed_workers": True,
            "lifecycle_events_require_host_control_authority": True,
            "revoked_facts_remain_append_only_audit_records": True,
        },
    }
    graph["topology_sha256"] = _topology_digest(graph)
    graph["scientific_sha256"] = _scientific_digest(graph)
    graph = full_recompute_canonical_hypergraph(
        graph,
        acceptance_spec=kernel.spec.acceptance,
    )
    graph["content_sha256"] = _graph_content_digest(graph)
    return graph


def _ingest_global_plan(
    graph: dict[str, Any],
    plan: Mapping[str, Any],
    *,
    target_id: str,
    route_aliases: dict[str, str],
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> None:
    proposal_origin_kind = str(
        plan.get("_proposal_origin_kind") or "codex_global_director"
    ).lower()
    if proposal_origin_kind not in _ORIGIN_KINDS:
        proposal_origin_kind = "manual"
    proposal_origin_ref = str(plan.get("_proposal_origin_ref") or "")
    admitted_marker = plan.get("_host_admitted_proposal_ids")
    admitted_ids = (
        {str(value) for value in admitted_marker if str(value)}
        if isinstance(admitted_marker, list)
        else None
    )
    priorities = {
        str(row.get("proposal_id") or ""): float(row.get("priority") or 0.0)
        for row in plan.get("frontier_priorities") or []
        if isinstance(row, Mapping) and str(row.get("proposal_id") or "")
    }
    family_details = {
        str(row.get("route_family_id") or row.get("family_id") or ""): dict(row)
        for row in plan.get("route_families") or []
        if isinstance(row, Mapping)
    }
    for skeleton in plan.get("multi_step_skeletons") or []:
        if not isinstance(skeleton, Mapping):
            continue
        steps = [
            dict(step)
            for step in skeleton.get("steps") or []
            if isinstance(step, Mapping)
            and (
                admitted_ids is None
                or str(step.get("step_id") or "") in admitted_ids
            )
        ]
        if not steps:
            continue
        alias = str(skeleton.get("route_family_id") or skeleton.get("skeleton_id") or "")
        route = {
            **family_details.get(alias, {}),
            "route_family_id": alias,
            "skeleton_ids": [str(skeleton.get("skeleton_id") or "")],
        }
        route_id = _ingest_route_family(
            graph,
            route,
            target_id=target_id,
            route_aliases=route_aliases,
            dirty=dirty,
        )
        for step in steps:
            _ingest_hypothesis(
                graph,
                {
                    **dict(step),
                    "route_family_id": alias,
                    "canonical_route_family_id": route_id,
                    "skeleton_id": str(skeleton.get("skeleton_id") or ""),
                    "origin_kind": proposal_origin_kind,
                    "origin_ref": proposal_origin_ref,
                    "frontier_priority": priorities.get(
                        str(step.get("step_id") or ""),
                        0.0,
                    ),
                },
                route_aliases=route_aliases,
                dirty=dirty,
                rejected=rejected,
            )
    _ingest_provider_frontier_requests(graph, plan, dirty=dirty)


def _ingest_provider_frontier_requests(
    graph: dict[str, Any],
    plan: Mapping[str, Any],
    *,
    dirty: set[str],
) -> None:
    """Annotate canonical molecules selected by Codex for local expansion.

    The annotation is a scheduling request, not a second frontier and not a
    chemistry fact.  ``compile_deficit_frontier`` remains the only queue.
    """

    target_id = str(graph.get("target_molecule_id") or "")
    for raw in plan.get("frontier_priorities") or []:
        if not isinstance(raw, Mapping):
            continue
        providers = sorted(
            {
                str(value).strip().lower()
                for value in raw.get("provider_preferences") or []
                if str(value).strip().lower() == "chemenzy"
            }
        )
        if not providers:
            continue
        molecule_id, canonical = molecule_identity(raw.get("target_smiles"))
        if (
            not molecule_id
            or molecule_id == target_id
            or molecule_id not in graph["molecules"]
        ):
            continue
        molecule = dict(graph["molecules"][molecule_id])
        molecule["provider_expansion_requested"] = True
        molecule["provider_expansion_priority"] = max(
            float(molecule.get("provider_expansion_priority") or 0.0),
            float(raw.get("priority") or 0.0),
        )
        molecule["provider_preferences"] = sorted(
            {*molecule.get("provider_preferences", []), *providers}
        )
        molecule["provider_retron_hints"] = sorted(
            {
                *molecule.get("provider_retron_hints", []),
                *(
                    str(value).strip()
                    for value in raw.get("retron_hints") or []
                    if str(value).strip()
                ),
            }
        )
        molecule["provider_request_ids"] = sorted(
            {
                *molecule.get("provider_request_ids", []),
                str(raw.get("priority_id") or ""),
            }
            - {""}
        )
        molecule["provider_request_rationale"] = str(
            raw.get("rationale") or molecule.get("provider_request_rationale") or ""
        )[:1000]
        graph["molecules"][molecule_id] = _with_digest(molecule)
        dirty.add(molecule_id)


def _ingest_route_family(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    target_id: str,
    route_aliases: dict[str, str],
    dirty: set[str],
) -> str:
    row = dict(value)
    route_id = route_family_identity(row, target_molecule_id=target_id)
    alias = str(row.get("route_family_id") or row.get("family_id") or row.get("name") or "")
    existing = dict(graph["route_families"].get(route_id) or {})
    aliases = sorted({*existing.get("aliases", []), alias} - {""})
    record = {
        "route_family_id": route_id,
        "aliases": aliases,
        "strategy": str(
            row.get("strategic_disconnection")
            or row.get("strategy")
            or row.get("rationale")
            or existing.get("strategy")
            or ""
        ),
        "skeleton_ids": sorted(
            {
                *existing.get("skeleton_ids", []),
                *(str(value) for value in row.get("skeleton_ids") or [] if str(value)),
            }
        ),
        "edge_ids": sorted(set(existing.get("edge_ids") or [])),
        "hypothesis_ids": sorted(set(existing.get("hypothesis_ids") or [])),
        "leaf_molecule_ids": sorted(set(existing.get("leaf_molecule_ids") or [])),
        "selected": row.get("selected", existing.get("selected", True)) is not False,
        "status": str(existing.get("status") or "active"),
        "closed": existing.get("closed") is True,
        "minimum_proof_level": int(existing.get("minimum_proof_level") or 0),
        "stock_closure_rate": float(existing.get("stock_closure_rate") or 0.0),
        "independent_source_group_count": int(
            existing.get("independent_source_group_count") or 0
        ),
        "blocking_deficit_ids": list(existing.get("blocking_deficit_ids") or []),
    }
    graph["route_families"][route_id] = _with_digest(record)
    if alias:
        route_aliases[alias] = route_id
    dirty.add(route_id)
    return route_id


def _ingest_hypothesis(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    route_aliases: Mapping[str, str],
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> None:
    row = dict(value)
    route_innovations, innovation_reasons = _route_innovation_records(row)
    if innovation_reasons:
        rejected.append(
            {
                "kind": "route_innovation",
                "proposal_id": str(row.get("step_id") or ""),
                "reasons": innovation_reasons,
            }
        )
        return
    precursors = row.get("precursor_smiles") or row.get("reactant_smiles") or []
    hypothesis_id, audit = hypothesis_identity(row.get("product_smiles"), precursors)
    canonical_product = str(audit.get("product_smiles") or "")
    canonical_precursors = [
        str(item)
        for item in audit.get("precursor_smiles_multiset") or []
        if str(item)
    ]
    identity_valid = bool(
        hypothesis_id
        and canonical_product
        and canonical_precursors
        and len(canonical_precursors) == len(list(precursors or []))
    )
    if not identity_valid:
        rejected.append(
            {
                "kind": "hypothesis",
                "proposal_id": str(row.get("step_id") or ""),
                "reasons": list(audit.get("reasons") or ["hypothesis_identity_invalid"]),
            }
        )
        return
    route_id = str(row.get("canonical_route_family_id") or "")
    if not route_id:
        route_id = str(route_aliases.get(str(row.get("route_family_id") or "")) or "")
    route_innovations = _bind_innovations_to_routes(
        route_innovations,
        [route_id] if route_id else [],
    )
    origin = _origin_record(row, default_kind="codex_global_director")
    existing = dict(graph["hypotheses"].get(hypothesis_id) or {})
    edge_id = f"edge:{audit['edge_digest']}"
    advisory_only = row.get("advisory_only") is True
    admission_accepted = audit.get("accepted") is True and not advisory_only
    admission_reasons = sorted(
        {
            *(str(reason) for reason in existing.get("admission_reasons") or []),
            *(str(reason) for reason in audit.get("reasons") or []),
            *(
                ["provider_route_quarantined_advisory_only"]
                if advisory_only
                else []
            ),
        }
        - {""}
    )
    record = {
        "hypothesis_id": hypothesis_id,
        "edge_digest": audit["edge_digest"],
        "product_smiles": audit["product_smiles"],
        "precursor_smiles": audit["precursor_smiles_multiset"],
        "status": (
            "materialized"
            if edge_id in graph["edges"]
            else "frontier_candidate"
            if admission_accepted
            else "admission_rejected"
        ),
        "admission_accepted": admission_accepted,
        "admission_reasons": admission_reasons,
        "origin_records": _merge_by_digest(existing.get("origin_records"), [origin]),
        "route_family_ids": sorted(
            {*(existing.get("route_family_ids") or []), route_id} - {""}
        ),
        "route_diversity_gain": float(row.get("route_diversity_gain") or 0.0),
        "frontier_priority": max(
            float(existing.get("frontier_priority") or 0.0),
            float(row.get("frontier_priority") or 0.0),
        ),
        # Predictions are retained as weak annotations only.  They do not
        # enter reaction_proofs and therefore cannot raise an edge proof tier.
        "condition_predictions": _merge_json_rows(
            existing.get("condition_predictions"),
            row.get("condition_predictions"),
        ),
        "route_innovations": merge_route_innovations(
            existing.get("route_innovations"),
            route_innovations,
        ),
        "admission_audit_sha256": _digest(audit),
    }
    graph["hypotheses"][hypothesis_id] = _with_digest(record)
    _ensure_molecules_for_audit(graph, audit, dirty=dirty)
    if not admission_accepted:
        # Keep a structurally identified but admission-rejected proposal as an
        # explicit L0 planning fact.  It must never be scheduled for
        # materialization or count as an edge proof, but retaining it lets the
        # route workbench show the whole proposed skeleton with a red warning
        # instead of silently truncating the route at the rejected step.
        rejected.append(
            {
                "kind": "hypothesis",
                "proposal_id": str(row.get("step_id") or ""),
                "hypothesis_id": hypothesis_id,
                "retained_as_l0": True,
                "reasons": admission_reasons or ["hypothesis_admission_rejected"],
            }
        )
    if edge_id in graph["edges"]:
        edge = dict(graph["edges"][edge_id])
        edge["origin_records"] = _merge_by_digest(
            edge.get("origin_records"),
            [origin],
        )
        edge["route_family_ids"] = sorted(
            {*edge.get("route_family_ids", []), route_id} - {""}
        )
        edge["route_innovations"] = merge_route_innovations(
            edge.get("route_innovations"),
            route_innovations,
        )
        graph["edges"][edge_id] = _with_digest(edge)
        dirty.add(edge_id)
    if route_id and route_id in graph["route_families"]:
        route = dict(graph["route_families"][route_id])
        route["hypothesis_ids"] = sorted(
            {*route.get("hypothesis_ids", []), hypothesis_id}
        )
        if edge_id in graph["edges"]:
            route["edge_ids"] = sorted({*route.get("edge_ids", []), edge_id})
        graph["route_families"][route_id] = _with_digest(route)
        dirty.add(route_id)
    dirty.add(hypothesis_id)


def _ingest_candidate(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    default_origin_kind: str,
    route_aliases: Mapping[str, str],
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> str:
    row = dict(value)
    route_innovations, innovation_reasons = _route_innovation_records(row)
    if innovation_reasons:
        rejected.append(
            {
                "kind": "route_innovation",
                "proposal_id": str(row.get("candidate_id") or row.get("step_id") or ""),
                "reasons": innovation_reasons,
            }
        )
        return ""
    product = row.get("product_smiles")
    precursors = row.get("precursor_smiles") or row.get("reactant_smiles") or []
    edge_id, audit = reaction_edge_identity(product, precursors)
    if not edge_id or audit.get("accepted") is not True:
        rejected.append(
            {
                "kind": "reaction_edge",
                "proposal_id": str(row.get("candidate_id") or row.get("step_id") or ""),
                "reasons": list(audit.get("reasons") or ["candidate_identity_invalid"]),
            }
        )
        return ""
    if _edge_would_cycle(graph, audit):
        rejected.append(
            {
                "kind": "reaction_edge",
                "proposal_id": str(row.get("candidate_id") or row.get("step_id") or ""),
                "reasons": ["canonical_hypergraph_cycle"],
            }
        )
        return ""
    existing = dict(graph["edges"].get(edge_id) or {})
    raw_origins = [
        dict(value)
        for value in row.get("proposal_refs") or row.get("origins") or []
        if isinstance(value, Mapping)
    ] or [row]
    origins = [
        _origin_record(origin, default_kind=default_origin_kind)
        for origin in raw_origins
    ]
    product_id, _ = molecule_identity(audit["product_smiles"])
    precursor_ids = [molecule_identity(value)[0] for value in audit["precursor_smiles_multiset"]]
    route_ids = set(existing.get("route_family_ids") or [])
    for origin in raw_origins:
        alias = str(origin.get("route_family_id") or "")
        if alias and route_aliases.get(alias):
            route_ids.add(str(route_aliases[alias]))
    route_innovations = _bind_innovations_to_routes(
        route_innovations,
        route_ids,
    )
    record = {
        "edge_id": edge_id,
        "edge_digest": audit["edge_digest"],
        "product_molecule_id": product_id,
        "precursor_molecule_ids": precursor_ids,
        "product_smiles": audit["product_smiles"],
        "precursor_smiles": audit["precursor_smiles_multiset"],
        "origin_records": _merge_by_digest(existing.get("origin_records"), origins),
        "route_family_ids": sorted(route_ids),
        "hypothesis_ids": sorted(
            {
                *existing.get("hypothesis_ids", []),
                f"hypothesis:{audit['edge_digest']}",
            }
        ),
        "source_binding_ids": sorted(set(existing.get("source_binding_ids") or [])),
        "exact_record_ids": sorted(set(existing.get("exact_record_ids") or [])),
        "procedure_record_ids": sorted(
            set(existing.get("procedure_record_ids") or [])
        ),
        "independent_source_groups": sorted(
            set(existing.get("independent_source_groups") or [])
        ),
        "reaction_proofs": list(existing.get("reaction_proofs") or []),
        "condition_predictions": _merge_json_rows(
            existing.get("condition_predictions"),
            row.get("condition_predictions"),
        ),
        "route_innovations": merge_route_innovations(
            existing.get("route_innovations"),
            route_innovations,
        ),
        "status": "materialized",
        "admission_audit_sha256": _digest(audit),
    }
    graph["edges"][edge_id] = _with_digest(record)
    _ensure_molecules_for_audit(graph, audit, dirty=dirty)
    product_record = dict(graph["molecules"][product_id])
    product_record["outgoing_edge_ids"] = sorted(
        {*product_record.get("outgoing_edge_ids", []), edge_id}
    )
    graph["molecules"][product_id] = _with_digest(product_record)
    for precursor_id in precursor_ids:
        precursor = dict(graph["molecules"][precursor_id])
        precursor["incoming_edge_ids"] = sorted(
            {*precursor.get("incoming_edge_ids", []), edge_id}
        )
        graph["molecules"][precursor_id] = _with_digest(precursor)
    hypothesis_id = f"hypothesis:{audit['edge_digest']}"
    if hypothesis_id in graph["hypotheses"]:
        hypothesis = dict(graph["hypotheses"][hypothesis_id])
        hypothesis["status"] = "materialized"
        graph["hypotheses"][hypothesis_id] = _with_digest(hypothesis)
        dirty.add(hypothesis_id)
    for route_id in route_ids:
        if route_id not in graph["route_families"]:
            continue
        route = dict(graph["route_families"][route_id])
        route["edge_ids"] = sorted({*route.get("edge_ids", []), edge_id})
        graph["route_families"][route_id] = _with_digest(route)
        dirty.add(route_id)
    dirty.update({edge_id, product_id, *precursor_ids})
    return edge_id


def _ingest_worker_result(
    graph: dict[str, Any],
    result: WorkerResult,
    *,
    route_aliases: Mapping[str, str],
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> bool:
    payload = dict(result.payload)
    if result.worker_type == "materialize_candidate":
        if result.status == "completed" and payload.get("accepted") is True:
            _ingest_candidate(
                graph,
                payload,
                default_origin_kind="codex_global_director",
                route_aliases=route_aliases,
                dirty=dirty,
                rejected=rejected,
            )
        else:
            rejected.append(
                {
                    "kind": "reaction_edge",
                    "proposal_id": result.command_id,
                    "reasons": list(result.failure_reasons or ["materialization_rejected"]),
                }
            )
        return False
    if result.worker_type == "validate_reaction":
        proof = dict(payload.get("reaction_proof") or {})
        edge_id = _edge_id_from_digest(graph, str(payload.get("edge_digest") or ""))
        if not edge_id or not _valid_proof(proof):
            rejected.append(
                {
                    "kind": "reaction_proof",
                    "proposal_id": result.command_id,
                    "reasons": ["reaction_proof_not_replayable_or_edge_missing"],
                }
            )
            return False
        edge = dict(graph["edges"][edge_id])
        edge["reaction_proofs"] = _merge_by_key(
            edge.get("reaction_proofs"),
            [proof],
            key="proof_digest",
        )
        graph["edges"][edge_id] = _with_digest(edge)
        dirty.add(edge_id)
        return True
    if result.worker_type == "record_condition_predictions":
        edge_id = _edge_id_from_digest(graph, str(payload.get("edge_digest") or ""))
        if not edge_id:
            rejected.append(
                {
                    "kind": "condition_prediction",
                    "proposal_id": result.command_id,
                    "reasons": ["condition_prediction_edge_not_materialized"],
                }
            )
            return False
        edge = dict(graph["edges"][edge_id])
        predictions = normalize_condition_predictions(
            payload.get("condition_predictions"), max_candidates=2
        )
        if predictions:
            edge["condition_predictions"] = normalize_condition_predictions(
                [*(edge.get("condition_predictions") or []), *predictions],
                max_candidates=2,
            )
        diagnostic = {
            **dict(payload.get("diagnostics") or {}),
            "command_id": result.command_id,
            "status": result.status,
        }
        edge["condition_prediction_attempts"] = _merge_json_rows(
            edge.get("condition_prediction_attempts"), [diagnostic]
        )
        graph["edges"][edge_id] = _with_digest(edge)
        dirty.add(edge_id)
        # Advisory predictions are graph annotations, not evidence revisions.
        return False
    if result.worker_type in {"discover_sources", "extract_exact_source"}:
        for binding in _source_bindings_from_payload(payload):
            _ingest_source_binding(graph, binding, dirty=dirty, rejected=rejected)
        if result.worker_type == "extract_exact_source":
            for record in payload.get("exact_records") or []:
                if isinstance(record, Mapping):
                    _ingest_exact_record(
                        graph,
                        record,
                        dirty=dirty,
                        rejected=rejected,
                    )
            for record in payload.get("procedure_records") or []:
                if isinstance(record, Mapping):
                    _ingest_procedure_record(
                        graph,
                        record,
                        dirty=dirty,
                        rejected=rejected,
                    )
            for conflict in payload.get("conflicts") or []:
                _ingest_conflict(graph, conflict, dirty=dirty, rejected=rejected)
        return bool(result.material_events)
    if result.worker_type == "detect_source_conflicts":
        for conflict in payload.get("conflicts") or []:
            _ingest_conflict(graph, conflict, dirty=dirty, rejected=rejected)
        return bool(payload.get("conflicts"))
    if result.worker_type in {"audit_deep_leaf_stock", "audit_benchmark_leaf_stock"}:
        for audit in payload.get("leaf_audits") or []:
            _ingest_stock_observation(graph, audit, dirty=dirty, rejected=rejected)
        return bool(payload.get("leaf_audits"))
    return False


def _ingest_source_binding(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> str:
    row = dict(value)
    if not _valid_content_digest(row):
        rejected.append({"kind": "source_binding", "reasons": ["source_binding_digest_invalid"]})
        return ""
    source_id = source_binding_identity(row)
    external_id = str(row.get("binding_id") or "")
    record = {**row, "source_binding_id": source_id, "external_binding_id": external_id}
    record.pop("binding_id", None)
    record.pop("content_sha256", None)
    graph["source_bindings"][source_id] = _with_digest(record)
    if external_id:
        graph["source_aliases"][external_id] = source_id
    dirty.add(source_id)
    return source_id


def _ingest_exact_record(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> None:
    row = dict(value)
    if not _valid_content_digest(row):
        rejected.append({"kind": "exact_record", "reasons": ["exact_record_digest_invalid"]})
        return
    edge_id = _edge_id_from_digest(graph, str(row.get("edge_digest") or ""))
    if not edge_id:
        rejected.append(
            {
                "kind": "exact_record",
                "proposal_id": str(row.get("record_id") or ""),
                "reasons": ["exact_record_edge_not_materialized"],
            }
        )
        return
    record_id = str(row.get("record_id") or f"exact:{_digest(row)}")
    graph["exact_records"][record_id] = row
    source_id = str(
        graph["source_aliases"].get(str(row.get("source_binding_id") or "")) or ""
    )
    edge = dict(graph["edges"][edge_id])
    edge["exact_record_ids"] = sorted({*edge.get("exact_record_ids", []), record_id})
    if source_id:
        edge["source_binding_ids"] = sorted(
            {*edge.get("source_binding_ids", []), source_id}
        )
    group = str(row.get("independence_group") or "")
    if group:
        edge["independent_source_groups"] = sorted(
            {*edge.get("independent_source_groups", []), group}
        )
    graph["edges"][edge_id] = _with_digest(edge)
    dirty.update({edge_id, record_id})


def _ingest_procedure_record(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> None:
    row = dict(value)
    record_id = str(row.get("procedure_record_id") or "")
    if not record_id or not _valid_content_digest(row):
        rejected.append(
            {"kind": "procedure_record", "reasons": ["procedure_record_digest_invalid"]}
        )
        return
    edge_id = _edge_id_from_digest(graph, str(row.get("edge_digest") or ""))
    exact_record_id = str(row.get("exact_record_id") or "")
    exact_record = dict(graph["exact_records"].get(exact_record_id) or {})
    if (
        not edge_id
        or not exact_record
        or str(exact_record.get("edge_digest") or "")
        != str(row.get("edge_digest") or "")
    ):
        rejected.append(
            {
                "kind": "procedure_record",
                "proposal_id": record_id,
                "reasons": ["procedure_record_exact_edge_binding_invalid"],
            }
        )
        return
    source_id = str(
        graph["source_aliases"].get(str(row.get("source_binding_id") or "")) or ""
    )
    if not source_id or source_id not in graph["source_bindings"]:
        rejected.append(
            {
                "kind": "procedure_record",
                "proposal_id": record_id,
                "reasons": ["procedure_record_source_binding_invalid"],
            }
        )
        return
    graph["procedure_records"][record_id] = row
    edge = dict(graph["edges"][edge_id])
    edge["procedure_record_ids"] = sorted(
        {*edge.get("procedure_record_ids", []), record_id}
    )
    edge["source_binding_ids"] = sorted(
        {*edge.get("source_binding_ids", []), source_id}
    )
    graph["edges"][edge_id] = _with_digest(edge)
    dirty.update({edge_id, record_id})


def _ingest_conflict(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> None:
    row = dict(value)
    if not _valid_content_digest(row):
        rejected.append({"kind": "conflict", "reasons": ["conflict_digest_invalid"]})
        return
    conflict_id = str(row.get("conflict_id") or f"conflict:{_digest(row)}")
    graph["conflicts"][conflict_id] = row
    dirty.add(conflict_id)
    subject = str(row.get("subject_id") or "")
    edge_id = _edge_id_from_digest(graph, subject)
    if edge_id:
        dirty.add(edge_id)


def _ingest_stock_observation(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> None:
    row = dict(value)
    if not _valid_content_digest(row):
        rejected.append({"kind": "stock", "reasons": ["stock_audit_digest_invalid"]})
        return
    molecule_id, canonical = molecule_identity(row.get("canonical_smiles"))
    if not molecule_id:
        rejected.append({"kind": "stock", "reasons": ["stock_molecule_invalid"]})
        return
    if molecule_id not in graph["molecules"]:
        graph["molecules"][molecule_id] = _molecule_record(molecule_id, canonical)
    observation_id = stock_observation_identity(row)
    graph["stock_observations"][observation_id] = _with_digest(
        {
            **row,
            "stock_observation_id": observation_id,
            "molecule_id": molecule_id,
        }
    )
    molecule = dict(graph["molecules"][molecule_id])
    molecule["stock_observation_ids"] = sorted(
        {*molecule.get("stock_observation_ids", []), observation_id}
    )
    graph["molecules"][molecule_id] = _with_digest(molecule)
    dirty.update({molecule_id, observation_id})


def _ingest_fact_lifecycle_event(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> bool:
    event = dict(value)
    reasons = validate_fact_lifecycle_event(event)
    event_id = str(event.get("event_id") or "")
    if reasons:
        rejected.append(
            {
                "kind": "fact_lifecycle_event",
                "proposal_id": event_id,
                "reasons": reasons,
            }
        )
        return False
    if event_id in graph["fact_lifecycle_events"]:
        return False
    subject_kind = str(event.get("subject_kind") or "")
    subject_id = str(event.get("subject_id") or "")
    subject = fact_subject(graph, subject_kind, subject_id)
    if not subject:
        reasons.append("fact_lifecycle_subject_missing")
    elif fact_subject_digest(subject_kind, subject) != str(
        event.get("subject_content_sha256") or ""
    ):
        reasons.append("fact_lifecycle_subject_digest_mismatch")
    current = graph_fact_lifecycle_state(graph, subject_kind, subject_id, subject)
    if event.get("action") == "restore":
        if current.get("active") is True:
            reasons.append("fact_lifecycle_restore_requires_inactive_subject")
        if str(event.get("supersedes_event_id") or "") != str(
            current.get("latest_event_id") or ""
        ):
            reasons.append("fact_lifecycle_restore_predecessor_mismatch")
        if str(event.get("effective_at") or "") <= str(
            current.get("effective_at") or ""
        ):
            reasons.append("fact_lifecycle_restore_not_after_predecessor")
    if reasons:
        rejected.append(
            {
                "kind": "fact_lifecycle_event",
                "proposal_id": event_id,
                "reasons": sorted(set(reasons)),
            }
        )
        return False
    graph["fact_lifecycle_events"][event_id] = event
    dirty.update({event_id, subject_id})
    dirty.update(_lifecycle_affected_entities(graph, subject_kind, subject_id))
    return True


def _ingest_action_signal(
    graph: dict[str, Any],
    value: Mapping[str, Any],
    *,
    dirty: set[str],
    rejected: list[dict[str, Any]],
) -> bool:
    row = _json_value(dict(value))
    row.pop("content_sha256", None)
    signal_id = str(row.get("signal_id") or row.get("deficit_id") or "").strip()
    kind = str(row.get("kind") or "").strip()
    status = str(row.get("status") or "open").strip()
    reasons: list[str] = []
    if not signal_id:
        reasons.append("action_signal_identity_missing")
    if kind not in {
        "architecture",
        "evidence",
        "replan",
        "program_discovery",
        "program_review",
        "program_admission",
        "program_validation",
        "experiment_feedback",
    }:
        reasons.append("action_signal_kind_invalid")
    if status not in {"open", "resolved"}:
        reasons.append("action_signal_status_invalid")
    if not str(row.get("object_id") or "").strip():
        reasons.append("action_signal_object_missing")
    if not str(row.get("reason") or "").strip():
        reasons.append("action_signal_reason_missing")
    existing = dict(graph.get("action_signals") or {}).get(signal_id)
    if isinstance(existing, Mapping):
        existing_row = dict(existing)
        if (
            str(existing_row.get("kind") or "") != kind
            or str(existing_row.get("object_id") or "")
            != str(row.get("object_id") or "")
        ):
            reasons.append("action_signal_identity_conflict")
        if (
            str(existing_row.get("status") or "open") == "resolved"
            and status == "open"
        ):
            reasons.append("resolved_action_signal_cannot_reopen")
    if reasons:
        rejected.append(
            {
                "kind": "action_signal",
                "proposal_id": signal_id,
                "reasons": sorted(set(reasons)),
            }
        )
        return False
    normalized = _with_digest(
        {
            **row,
            "signal_id": signal_id,
            "deficit_id": str(row.get("deficit_id") or signal_id),
            "status": status,
            "metadata": _json_value(dict(row.get("metadata") or {})),
        }
    )
    if isinstance(existing, Mapping) and dict(existing) == normalized:
        return False
    graph["action_signals"][signal_id] = normalized
    dirty.add(signal_id)
    dirty.update(
        str(entity_id)
        for entity_id in row.get("entity_ids") or []
        if str(entity_id)
    )
    return True


def _lifecycle_affected_entities(
    graph: Mapping[str, Any], subject_kind: str, subject_id: str
) -> set[str]:
    affected: set[str] = set()
    aliases = dict(graph.get("source_aliases") or {})
    for edge_id, raw_edge in dict(graph.get("edges") or {}).items():
        edge = dict(raw_edge) if isinstance(raw_edge, Mapping) else {}
        matches = False
        if subject_kind == "source_binding":
            matches = subject_id in edge.get("source_binding_ids", [])
            if not matches:
                for record_id in edge.get("exact_record_ids") or []:
                    record = dict(
                        dict(graph.get("exact_records") or {}).get(record_id) or {}
                    )
                    canonical_source = str(
                        aliases.get(str(record.get("source_binding_id") or "")) or ""
                    )
                    if canonical_source == subject_id:
                        matches = True
                        break
        elif subject_kind == "exact_record":
            matches = subject_id in edge.get("exact_record_ids", [])
        elif subject_kind == "procedure_record":
            matches = subject_id in edge.get("procedure_record_ids", [])
        elif subject_kind == "reaction_proof":
            matches = any(
                isinstance(proof, Mapping)
                and str(proof.get("proof_digest") or "") == subject_id
                for proof in edge.get("reaction_proofs") or []
            )
        if matches:
            affected.add(str(edge_id))
            affected.add(str(edge.get("product_molecule_id") or ""))
    if subject_kind == "stock_observation":
        observation = dict(
            dict(graph.get("stock_observations") or {}).get(subject_id) or {}
        )
        affected.add(str(observation.get("molecule_id") or ""))
    affected.discard("")
    return affected


def _refresh_molecules(graph: dict[str, Any], *, dirty: set[str]) -> None:
    for molecule_id in sorted(set(graph["molecules"]) & dirty):
        molecule = dict(graph["molecules"][molecule_id])
        outgoing = [
            edge_id for edge_id in molecule.get("outgoing_edge_ids") or [] if edge_id in graph["edges"]
        ]
        observations = [
            dict(graph["stock_observations"][observation_id])
            for observation_id in molecule.get("stock_observation_ids") or []
            if observation_id in graph["stock_observations"]
        ]
        observations.sort(
            key=lambda row: (
                str(row.get("audited_as_of") or ""),
                str(row.get("stock_observation_id") or ""),
            )
        )
        active_observations = [
            row
            for row in observations
            if graph_fact_lifecycle_state(
                graph,
                "stock_observation",
                str(row.get("stock_observation_id") or ""),
                row,
            ).get("active")
            is True
        ]
        active = active_observations[-1] if active_observations else {}
        inactive_ids = sorted(
            str(row.get("stock_observation_id") or "")
            for row in observations
            if row not in active_observations
        )
        molecule.update(
            {
                "outgoing_edge_ids": sorted(set(outgoing)),
                "incoming_edge_ids": sorted(set(molecule.get("incoming_edge_ids") or [])),
                "is_leaf": not outgoing,
                "active_stock_observation_id": str(
                    active.get("stock_observation_id") or ""
                ),
                "stock_closed": active.get("accepted") is True,
                "inactive_stock_observation_ids": inactive_ids,
            }
        )
        graph["molecules"][molecule_id] = _with_digest(molecule)


def _refresh_routes(
    graph: dict[str, Any],
    *,
    dirty: set[str],
    acceptance_spec: Any,
) -> None:
    policy = ProofPolicy.from_acceptance(acceptance_spec)
    for route_id in sorted(set(graph["route_families"]) & dirty):
        route = dict(graph["route_families"][route_id])
        edge_ids = [value for value in route.get("edge_ids") or [] if value in graph["edges"]]
        hypotheses = [
            value
            for value in route.get("hypothesis_ids") or []
            if value in graph["hypotheses"]
            and graph["hypotheses"][value].get("status") != "materialized"
        ]
        product_ids = {
            str(graph["edges"][edge_id]["product_molecule_id"]) for edge_id in edge_ids
        }
        precursor_ids = {
            str(value)
            for edge_id in edge_ids
            for value in graph["edges"][edge_id]["precursor_molecule_ids"]
        }
        leaves = sorted(precursor_ids - product_ids)
        edge_stitches = [
            stitch_edge_proof(graph, edge_id, policy=policy) for edge_id in edge_ids
        ]
        proof_levels = [int(value.get("achieved_level") or 0) for value in edge_stitches]
        minimum = min(proof_levels, default=0)
        leaf_stitches = [
            stitch_leaf_stock_proof(graph, molecule_id, policy=policy)
            for molecule_id in leaves
        ]
        closed_stock = sum(value.get("accepted") is True for value in leaf_stitches)
        stock_rate = closed_stock / len(leaves) if leaves else 0.0
        required = int(acceptance_spec.minimum_edge_proof_level)
        all_edges_accepted = bool(edge_stitches) and all(
            value.get("accepted") is True for value in edge_stitches
        )
        closed = (
            bool(edge_ids)
            and not hypotheses
            and minimum >= required
            and all_edges_accepted
            and stock_rate == 1.0
        )
        source_groups = {
            str(group)
            for proof in edge_stitches
            for group in proof.get("independent_source_groups") or []
            if str(group)
        }
        source_requirement_met = (
            len(source_groups)
            >= int(acceptance_spec.minimum_independent_source_groups)
        )
        closed = closed and source_requirement_met
        route.update(
            {
                "edge_ids": sorted(set(edge_ids)),
                "leaf_molecule_ids": leaves,
                "unmaterialized_hypothesis_ids": sorted(hypotheses),
                "minimum_proof_level": minimum,
                "all_edges_accepted": all_edges_accepted,
                "stock_closure_rate": round(stock_rate, 6),
                "independent_source_group_count": len(source_groups),
                "independent_source_requirement_met": source_requirement_met,
                "closed": closed,
                "status": "closed" if closed else "active",
            }
        )
        graph["route_families"][route_id] = _with_digest(route)


def _mark_dominated_routes(graph: dict[str, Any]) -> None:
    routes = graph["route_families"]
    active = {
        route_id: dict(route)
        for route_id, route in routes.items()
        if route.get("selected") is not False
    }
    for route_id, route in sorted(active.items()):
        edges = set(route.get("edge_ids") or [])
        leaves = set(route.get("leaf_molecule_ids") or [])
        dominated_by = ""
        if edges:
            for other_id, other in sorted(active.items()):
                if other_id == route_id:
                    continue
                other_edges = set(other.get("edge_ids") or [])
                if (
                    other_edges
                    and other_edges < edges
                    # A shorter route that stops at a purchasable intermediate is
                    # a different procurement strategy, not proof that the deeper
                    # route is redundant.  Dominance is only meaningful when both
                    # families terminate at the same audited material boundary.
                    and set(other.get("leaf_molecule_ids") or []) == leaves
                    and int(other.get("minimum_proof_level") or 0)
                    >= int(route.get("minimum_proof_level") or 0)
                    and float(other.get("stock_closure_rate") or 0.0)
                    >= float(route.get("stock_closure_rate") or 0.0)
                ):
                    dominated_by = other_id
                    break
        row = dict(route)
        if dominated_by:
            row["status"] = "dominated"
            row["dominated_by_route_family_id"] = dominated_by
        else:
            row.pop("dominated_by_route_family_id", None)
            if row.get("status") == "dominated":
                row["status"] = "closed" if row.get("closed") is True else "active"
        routes[route_id] = _with_digest(row)


def _portfolio_ranking(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route_id, route in dict(graph.get("route_families") or {}).items():
        if not isinstance(route, Mapping) or route.get("status") == "dominated":
            continue
        edge_count = len(route.get("edge_ids") or [])
        score = (
            100.0 * (route.get("closed") is True)
            + 12.0 * int(route.get("minimum_proof_level") or 0)
            + 20.0 * float(route.get("stock_closure_rate") or 0.0)
            + 4.0 * min(3, int(route.get("independent_source_group_count") or 0))
            - 1.5 * edge_count
            - 10.0 * len(route.get("unmaterialized_hypothesis_ids") or [])
        )
        rows.append(
            {
                "route_family_id": route_id,
                "score": round(score, 6),
                "closed": route.get("closed") is True,
                "edge_count": edge_count,
                "strategy": str(route.get("strategy") or ""),
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["route_family_id"]))


def _dependency_index(graph: Mapping[str, Any]) -> dict[str, Any]:
    routes_by_entity: dict[str, set[str]] = {}
    for route_id, route in dict(graph.get("route_families") or {}).items():
        edge_ids = [str(value) for value in route.get("edge_ids") or []]
        leaf_ids = [str(value) for value in route.get("leaf_molecule_ids") or []]
        entities = {
            route_id,
            *edge_ids,
            *(str(value) for value in route.get("hypothesis_ids") or []),
            *leaf_ids,
        }
        for edge_id in edge_ids:
            edge = dict(dict(graph.get("edges") or {}).get(edge_id) or {})
            entities.update(str(value) for value in edge.get("source_binding_ids") or [])
            entities.update(str(value) for value in edge.get("exact_record_ids") or [])
            entities.update(str(value) for value in edge.get("procedure_record_ids") or [])
            entities.update(
                str(proof.get("proof_digest") or "")
                for proof in edge.get("reaction_proofs") or []
                if isinstance(proof, Mapping)
            )
        for molecule_id in leaf_ids:
            molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
            entities.update(
                str(value) for value in molecule.get("stock_observation_ids") or []
            )
        entities.discard("")
        for entity_id in entities:
            routes_by_entity.setdefault(entity_id, set()).add(route_id)
    return {
        "routes_by_entity": {
            key: sorted(values) for key, values in sorted(routes_by_entity.items())
        }
    }


def _dirty_ancestor_closure(graph: Mapping[str, Any], dirty: set[str]) -> set[str]:
    queue = [value for value in dirty if value in graph["molecules"]]
    seen = set(queue)
    out: set[str] = set()
    while queue:
        molecule_id = queue.pop(0)
        molecule = graph["molecules"][molecule_id]
        for edge_id in molecule.get("incoming_edge_ids") or []:
            if edge_id not in graph["edges"]:
                continue
            out.add(edge_id)
            product_id = str(graph["edges"][edge_id]["product_molecule_id"])
            if product_id not in seen:
                seen.add(product_id)
                out.add(product_id)
                queue.append(product_id)
    return out


def _routes_affected_by(graph: Mapping[str, Any], dirty: set[str]) -> set[str]:
    index = dict(dict(graph.get("dependency_index") or {}).get("routes_by_entity") or {})
    routes = {
        str(route_id)
        for entity_id in dirty
        for route_id in index.get(entity_id) or []
    }
    for route_id, route in dict(graph.get("route_families") or {}).items():
        entities = {
            *(route.get("edge_ids") or []),
            *(route.get("hypothesis_ids") or []),
            *(route.get("leaf_molecule_ids") or []),
        }
        if entities & dirty:
            routes.add(str(route_id))
    return routes


def _edge_would_cycle(graph: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    product_id, _ = molecule_identity(audit.get("product_smiles"))
    for precursor_smiles in audit.get("precursor_smiles_multiset") or []:
        precursor_id, _ = molecule_identity(precursor_smiles)
        if precursor_id == product_id or _molecule_reaches(graph, precursor_id, product_id):
            return True
    return False


def _molecule_reaches(graph: Mapping[str, Any], start: str, target: str) -> bool:
    queue = [start]
    seen: set[str] = set()
    while queue:
        molecule_id = queue.pop(0)
        if molecule_id == target:
            return True
        if molecule_id in seen:
            continue
        seen.add(molecule_id)
        molecule = dict(graph.get("molecules") or {}).get(molecule_id) or {}
        for edge_id in molecule.get("outgoing_edge_ids") or []:
            edge = dict(graph.get("edges") or {}).get(edge_id) or {}
            queue.extend(str(value) for value in edge.get("precursor_molecule_ids") or [])
    return False


def _ancestor_smiles_for_product(
    graph: Mapping[str, Any],
    product_smiles: Any,
) -> tuple[str, ...]:
    product_id, _ = molecule_identity(product_smiles)
    if not product_id or product_id not in graph["molecules"]:
        return ()
    queue = [product_id]
    seen = {product_id}
    ancestors: set[str] = {str(graph["target_molecule_id"])}
    while queue:
        molecule_id = queue.pop(0)
        molecule = graph["molecules"].get(molecule_id) or {}
        for edge_id in molecule.get("incoming_edge_ids") or []:
            edge = graph["edges"].get(edge_id) or {}
            ancestor_id = str(edge.get("product_molecule_id") or "")
            if ancestor_id and ancestor_id not in seen:
                seen.add(ancestor_id)
                ancestors.add(ancestor_id)
                queue.append(ancestor_id)
    return tuple(
        sorted(
            str(graph["molecules"][molecule_id]["canonical_smiles"])
            for molecule_id in ancestors
            if molecule_id in graph["molecules"] and molecule_id != product_id
        )
    )


def _ensure_molecules_for_audit(
    graph: dict[str, Any],
    audit: Mapping[str, Any],
    *,
    dirty: set[str],
) -> None:
    for smiles in [audit.get("product_smiles"), *(audit.get("precursor_smiles_multiset") or [])]:
        molecule_id, canonical = molecule_identity(smiles)
        if molecule_id and molecule_id not in graph["molecules"]:
            graph["molecules"][molecule_id] = _molecule_record(molecule_id, canonical)
        if molecule_id:
            dirty.add(molecule_id)


def _rebuild_all_adjacency(graph: dict[str, Any]) -> None:
    for molecule_id, molecule in graph["molecules"].items():
        row = dict(molecule)
        row["outgoing_edge_ids"] = []
        row["incoming_edge_ids"] = []
        graph["molecules"][molecule_id] = row
    for edge_id, edge in graph["edges"].items():
        product_id = str(edge["product_molecule_id"])
        product = dict(graph["molecules"][product_id])
        product["outgoing_edge_ids"] = sorted({*product["outgoing_edge_ids"], edge_id})
        graph["molecules"][product_id] = product
        for precursor_id in edge["precursor_molecule_ids"]:
            precursor = dict(graph["molecules"][precursor_id])
            precursor["incoming_edge_ids"] = sorted(
                {*precursor["incoming_edge_ids"], edge_id}
            )
            graph["molecules"][precursor_id] = precursor


def _molecule_record(molecule_id: str, canonical_smiles: str) -> dict[str, Any]:
    return _with_digest(
        {
            "molecule_id": molecule_id,
            "canonical_smiles": canonical_smiles,
            "outgoing_edge_ids": [],
            "incoming_edge_ids": [],
            "stock_observation_ids": [],
            "inactive_stock_observation_ids": [],
            "active_stock_observation_id": "",
            "stock_closed": False,
            "is_leaf": True,
        }
    )


def _route_innovation_records(
    value: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    row = dict(value)
    raw_values = [
        dict(item)
        for item in row.get("route_innovations") or []
        if isinstance(item, Mapping)
    ]
    singular = row.get("route_innovation") or row.get("innovation")
    if not raw_values and isinstance(singular, Mapping):
        raw_values = [dict(singular)]
    elif not raw_values and (
        row.get("innovation_kind") or row.get("proposal_basis")
    ):
        raw_values = [row]
    normalized: list[dict[str, Any]] = []
    reasons: list[str] = []
    for raw in raw_values:
        record, rejected = normalize_route_innovation(
            {**row, "route_innovation": raw}
        )
        reasons.extend(rejected)
        if record:
            normalized.append(record)
    return merge_route_innovations((), normalized), sorted(set(reasons))


def _bind_innovations_to_routes(
    values: Iterable[Mapping[str, Any]],
    route_ids: Iterable[str],
) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    canonical_route_ids = {str(value) for value in route_ids if str(value)}
    for raw in values:
        row = dict(raw)
        if canonical_route_ids:
            row["route_family_ids"] = sorted(
                {
                    *(str(value) for value in row.get("route_family_ids") or []),
                    *canonical_route_ids,
                }
                - {""}
            )
        row.pop("innovation_id", None)
        row.pop("content_sha256", None)
        normalized, reasons = normalize_route_innovation(row)
        if normalized and not reasons:
            bound.append(normalized)
    return merge_route_innovations((), bound)


def _origin_record(value: Mapping[str, Any], *, default_kind: str) -> dict[str, Any]:
    row = dict(value)
    kind = str(row.get("origin_kind") or row.get("source") or default_kind).lower()
    if kind not in _ORIGIN_KINDS:
        kind = default_kind if default_kind in _ORIGIN_KINDS else "manual"
    record = {
        "origin_kind": kind,
        "origin_ref": str(
            row.get("origin_ref")
            or row.get("source_ref")
            or row.get("model")
            or ""
        ),
        "proposal_id": str(row.get("proposal_id") or row.get("step_id") or ""),
        "route_family_id": str(row.get("route_family_id") or ""),
        "skeleton_id": str(row.get("skeleton_id") or ""),
        "transformation_hypothesis": str(
            row.get("transformation_hypothesis") or ""
        ),
    }
    provider_metadata = row.get("provider_reaction_metadata")
    if isinstance(provider_metadata, Mapping):
        metadata = dict(provider_metadata)
        supplied_digest = str(metadata.pop("content_sha256", ""))
        computed_digest = _digest(metadata)
        metadata["content_sha256"] = computed_digest
        record["provider_reaction_metadata"] = metadata
        record["provider_reaction_metadata_sha256"] = computed_digest
        record["provider_reaction_metadata_digest_valid"] = (
            not supplied_digest or supplied_digest == computed_digest
        )
    record["origin_sha256"] = _digest(record)
    return record


def _source_bindings_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = list(payload.get("source_bindings") or [])
    if isinstance(payload.get("source_binding"), Mapping):
        values.append(payload["source_binding"])
    return [dict(value) for value in values if isinstance(value, Mapping)]


def _edge_id_from_digest(graph: Mapping[str, Any], digest: str) -> str:
    candidate = digest if digest.startswith("edge:") else f"edge:{digest}"
    if candidate in dict(graph.get("edges") or {}):
        return candidate
    return next(
        (
            str(edge_id)
            for edge_id, edge in dict(graph.get("edges") or {}).items()
            if str(edge.get("edge_digest") or "") == digest
        ),
        "",
    )


def _valid_proof(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("proof_digest", ""))
    return bool(
        supplied
        and supplied == _digest(row)
        and row.get("schema_version") == "reaction_step_proof.v1"
    )


def _valid_content_digest(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return bool(supplied and supplied == _digest(row))


def _merge_by_digest(existing: Any, incoming: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = {
        str(dict(value).get("origin_sha256") or _digest(value)): dict(value)
        for value in existing or []
        if isinstance(value, Mapping)
    }
    for value in incoming:
        row = dict(value)
        rows[str(row.get("origin_sha256") or _digest(row))] = row
    return [rows[key] for key in sorted(rows)]


def _merge_json_rows(existing: Any, incoming: Any) -> list[dict[str, Any]]:
    """Merge annotation rows by content without granting them authority."""

    rows: dict[str, dict[str, Any]] = {}
    for value in [*(existing or []), *(incoming or [])]:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        row.setdefault("authority_scope", "model_predicted_condition")
        row.setdefault("not_reaction_proof", True)
        rows[_digest(row)] = row
    return [rows[key] for key in sorted(rows)]


def _merge_by_key(existing: Any, incoming: Iterable[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    rows = {
        str(dict(value).get(key) or _digest(value)): dict(value)
        for value in existing or []
        if isinstance(value, Mapping)
    }
    for value in incoming:
        row = dict(value)
        rows[str(row.get(key) or _digest(row))] = row
    return [rows[value] for value in sorted(rows)]


def _route_aliases(graph: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(alias): str(route_id)
        for route_id, route in dict(graph.get("route_families") or {}).items()
        for alias in route.get("aliases") or []
        if str(alias)
    }


def _mutable_graph(value: Mapping[str, Any]) -> dict[str, Any]:
    graph = _json_value(dict(value))
    for key in (
        "molecules",
        "edges",
        "source_bindings",
        "source_aliases",
        "exact_records",
        "procedure_records",
        "fact_lifecycle_events",
        "stock_observations",
        "route_families",
        "hypotheses",
        "conflicts",
        "action_signals",
        "entity_revisions",
    ):
        graph[key] = dict(graph.get(key) or {})
    return graph


def _validate_graph(value: Mapping[str, Any], *, expected_run_id: str) -> None:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    if (
        row.get("schema_version") != CANONICAL_HYPERGRAPH_SCHEMA
        or row.get("run_id") != expected_run_id
        or not supplied
        or supplied != _digest(row)
        or row.get("scientific_sha256") != _scientific_digest(row)
    ):
        raise CanonicalHypergraphError("canonical_graph_validation_failed")


def _topology_digest(graph: Mapping[str, Any]) -> str:
    return _digest(
        {
            "molecules": _sorted_mapping(graph.get("molecules")),
            "edges": _sorted_mapping(graph.get("edges")),
            "source_bindings": _sorted_mapping(graph.get("source_bindings")),
            "exact_records": _sorted_mapping(graph.get("exact_records")),
            "procedure_records": _sorted_mapping(graph.get("procedure_records")),
            "fact_lifecycle_events": _sorted_mapping(
                graph.get("fact_lifecycle_events")
            ),
            "stock_observations": _sorted_mapping(graph.get("stock_observations")),
            "route_families": _sorted_mapping(graph.get("route_families")),
            "hypotheses": _sorted_mapping(graph.get("hypotheses")),
            "conflicts": _sorted_mapping(graph.get("conflicts")),
        }
    )


def _scientific_digest(graph: Mapping[str, Any]) -> str:
    return _digest(canonical_scientific_projection(graph))


def _graph_content_digest(graph: Mapping[str, Any]) -> str:
    row = dict(graph)
    row.pop("content_sha256", None)
    row.pop("artifact_sha256", None)
    return _digest(row)


def _report(
    graph: Mapping[str, Any],
    *,
    dirty: Iterable[str],
    rejected: Iterable[Mapping[str, Any]],
    evidence_changed: bool,
    changed: bool,
) -> dict[str, Any]:
    row = {
        "schema_version": CANONICAL_INGESTION_REPORT_SCHEMA,
        "changed": changed,
        "evidence_changed": evidence_changed,
        "revision": int(graph.get("revision") or 0),
        "scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "dirty_entity_ids": sorted({str(value) for value in dirty}),
        "rejected": [dict(value) for value in rejected],
        "semantics": {
            "single_ingestion_path": True,
            "rejected_inputs_did_not_mutate_graph": True,
            "incremental_projection": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def _all_entity_ids(graph: Mapping[str, Any]) -> set[str]:
    return {
        str(entity_id)
        for key in (
            "molecules",
            "edges",
            "source_bindings",
            "exact_records",
            "procedure_records",
            "fact_lifecycle_events",
            "stock_observations",
            "route_families",
            "hypotheses",
            "conflicts",
            "action_signals",
        )
        for entity_id in dict(graph.get(key) or {})
    }


def _entity_count(graph: Mapping[str, Any]) -> int:
    return len(_all_entity_ids(graph))


def _sorted_mapping(value: Any) -> dict[str, Any]:
    return {
        str(key): _json_value(item)
        for key, item in sorted(dict(value or {}).items(), key=lambda pair: str(pair[0]))
    }


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
