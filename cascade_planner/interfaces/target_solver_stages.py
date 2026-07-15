"""Deterministic stages used by the target-only campaign orchestrator."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.application.canonical_hypergraph import (
    CanonicalIngestionBatch,
)
from cascade_planner.application.proof_policy import (
    stock_boundary_matches,
)
from cascade_planner.application.reaction_proof_versions import (
    CURRENT_REACTION_VALIDATOR_VERSION,
    active_reaction_proofs,
)
from cascade_planner.application.precursor_repair import (
    propose_precursor_repair,
)
from cascade_planner.application.reaction_mapping import (
    ReactionMapper,
    ReactionMappingConfig,
    ReactionMappingError,
    map_reactions_locally,
)
from cascade_planner.application.worker_runtime import WorkerBudget, WorkerCommand
from cascade_planner.application.retrosynthesis_workers import (
    materialization_commands_for_proposals,
)
from cascade_planner.interfaces.live_stock import build_pubchem_vendor_catalog
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.source_locators import canonical_traceable_source_ref


StockCatalogBuilder = Callable[..., Mapping[str, Any]]
InventorySnapshotBuilder = Callable[..., Mapping[str, Any]]


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def validate_materialized_edges(
    service: RetrosynthesisCampaignService,
    *,
    atom_mapper: ReactionMapper | None = None,
    max_reactions: int = 48,
) -> dict[str, Any]:
    graph = service.graph_store.load()
    pending = [
        dict(edge)
        for edge in graph["edges"].values()
        if not active_reaction_proofs(edge.get("reaction_proofs") or [])
    ]
    reactions = {
        str(edge["edge_id"]): (
            ".".join(str(value) for value in edge.get("precursor_smiles") or [])
            + ">>"
            + str(edge.get("product_smiles") or "")
        )
        for edge in pending
    }
    try:
        mapping = map_reactions_locally(
            reactions.values(),
            mapper=atom_mapper,
            config=ReactionMappingConfig(max_reactions=max_reactions),
        )
    except ReactionMappingError as exc:
        mapping = {
            "schema_version": "local_reaction_mapping_report.v1",
            "backend": "rxnmapper",
            "requested_count": len(reactions),
            "mapped_count": 0,
            "failure_count": len(reactions),
            "truncated": False,
            "mapped_reactions": {},
            "failures": [
                {"reaction_smiles": value, "reason": str(exc)}
                for value in reactions.values()
            ],
            "elapsed_s": 0.0,
            "semantics": {
                "local_only": True,
                "hosted_model_calls": 0,
                "mapping_is_not_reaction_proof": True,
            },
        }
    mapped = dict(mapping.get("mapped_reactions") or {})
    commands: list[WorkerCommand] = []
    edge_by_id = {str(edge["edge_id"]): edge for edge in pending}
    for edge_id, reaction in reactions.items():
        mapped_reaction = str(mapped.get(reaction) or "")
        if not mapped_reaction:
            continue
        edge = edge_by_id[edge_id]
        exact_records = [
            graph["exact_records"][record_id]
            for record_id in edge.get("exact_record_ids") or []
            if record_id in graph["exact_records"]
        ]
        commands.append(
            _command(
                service,
                "validate_reaction",
                {
                    "candidate": {
                        "accepted": True,
                        "candidate_id": edge_id,
                        "edge_digest": edge["edge_digest"],
                        "product_smiles": edge["product_smiles"],
                        "precursor_smiles": edge["precursor_smiles"],
                    },
                    "mapped_reaction_smiles": mapped_reaction,
                    "exact_source_records": exact_records,
                    "validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
                },
                task_kind="validation",
                suffix=(
                    f"{str(edge['edge_digest'])[:24]}:"
                    f"{CURRENT_REACTION_VALIDATOR_VERSION.rsplit('.', 1)[-1]}"
                ),
            )
        )
    execution = (
        service.execute_commands(
            commands,
            idempotency_key=(
                f"solve-target:validation:{service.kernel.state.graph_revision}"
            ),
        )
        if commands
        else {"executed_command_count": 0, "material_events": []}
    )
    updated = service.graph_store.load()
    accepted_ids = sorted(
        edge_id
        for edge_id in edge_by_id
        if any(
            isinstance(proof, Mapping) and proof.get("accepted") is True
            for proof in active_reaction_proofs(
                dict(updated["edges"].get(edge_id) or {}).get("reaction_proofs")
                or []
            )
        )
    )
    rejected_ids = sorted(set(edge_by_id) - set(accepted_ids))
    mapping_failures = {
        str(row.get("reaction_smiles") or ""): str(row.get("reason") or "")
        for row in mapping.get("failures") or []
        if isinstance(row, Mapping)
    }
    rejection_diagnostics: list[dict[str, Any]] = []
    rejection_reason_counts: Counter[str] = Counter()
    for edge_id in rejected_ids:
        edge = dict(updated["edges"].get(edge_id) or edge_by_id.get(edge_id) or {})
        reaction = (
            ".".join(str(value) for value in edge.get("precursor_smiles") or [])
            + ">>"
            + str(edge.get("product_smiles") or "")
        )
        proofs = active_reaction_proofs(edge.get("reaction_proofs") or [])
        reasons = {
            str(reason)
            for proof in proofs
            if proof.get("accepted") is not True
            for reason in proof.get("reasons") or []
            if str(reason).strip()
        }
        if reaction not in mapped:
            reasons.add(mapping_failures.get(reaction) or "reaction_mapping_missing")
        if not reasons:
            reasons.add("reaction_validation_not_accepted")
        rejection_reason_counts.update(reasons)
        rejection_diagnostics.append(
            {
                "edge_id": edge_id,
                "product_smiles": str(edge.get("product_smiles") or ""),
                "precursor_smiles": list(edge.get("precursor_smiles") or []),
                "route_family_ids": list(edge.get("route_family_ids") or []),
                "reasons": sorted(reasons),
            }
        )
    return {
        "stage": "reaction_validation",
        "status": (
            "completed"
            if mapping.get("requested_count") == mapping.get("mapped_count")
            else "partial"
        ),
        "pending_edge_count": len(pending),
        "validation_command_count": len(commands),
        "accepted_validation_count": len(accepted_ids),
        "rejected_validation_count": len(rejected_ids),
        "accepted_edge_ids": accepted_ids,
        "rejected_edge_ids": rejected_ids,
        "rejection_diagnostics": rejection_diagnostics,
        "rejection_reason_counts": dict(
            sorted(
                rejection_reason_counts.items(),
                key=lambda row: (-row[1], row[0]),
            )
        ),
        "mapping": mapping,
        "execution": execution,
    }


def repair_rejected_precursor_typos(
    service: RetrosynthesisCampaignService,
    validation: Mapping[str, Any],
    *,
    max_repairs: int = 8,
) -> dict[str, Any]:
    """Materialize narrow product-grounded repairs as new L0 alternatives."""

    graph = service.graph_store.load()
    mapped = dict(dict(validation.get("mapping") or {}).get("mapped_reactions") or {})
    repairs: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for edge_id in list(validation.get("rejected_edge_ids") or [])[:max_repairs]:
        edge = dict(dict(graph.get("edges") or {}).get(str(edge_id)) or {})
        if not edge:
            continue
        reaction = (
            ".".join(str(value) for value in edge.get("precursor_smiles") or [])
            + ">>"
            + str(edge.get("product_smiles") or "")
        )
        repair = propose_precursor_repair(
            mapped_reaction_smiles=str(mapped.get(reaction) or ""),
            product_smiles=str(edge.get("product_smiles") or ""),
            precursor_smiles=edge.get("precursor_smiles") or [],
        )
        repairs.append({"edge_id": edge_id, **repair})
        if repair.get("accepted") is not True:
            continue
        origins = [
            dict(value)
            for value in edge.get("origin_records") or []
            if isinstance(value, Mapping)
        ]
        aliases = sorted(
            {
                str(alias)
                for route_id in edge.get("route_family_ids") or []
                for alias in dict(graph["route_families"].get(route_id) or {}).get(
                    "aliases"
                )
                or []
                if str(alias)
            }
        ) or [""]
        for alias in aliases:
            proposals.append(
                {
                    "product_smiles": repair["product_smiles"],
                    "precursor_smiles": repair["repaired_precursor_smiles"],
                    "origin_kind": "host_product_grounded_repair",
                    "origin_ref": str(edge_id),
                    "proposal_id": (
                        f"repair:{str(edge_id).removeprefix('edge:')[:20]}"
                    ),
                    "route_family_id": alias,
                    "skeleton_id": str(
                        next(
                            (
                                row.get("skeleton_id")
                                for row in origins
                                if row.get("skeleton_id")
                            ),
                            "",
                        )
                    ),
                    "transformation_hypothesis": (
                        f"product-grounded {repair['repair_kind']} correction; "
                        "requires normal host reaction validation"
                    ),
                }
            )
    commands = materialization_commands_for_proposals(
        proposals,
        run_id=service.kernel.spec.run_id,
        input_revision=service.kernel.state.graph_revision,
        dependency_revisions={
            "graph_revision": service.kernel.state.graph_revision,
            "evidence_revision": service.kernel.state.evidence_revision,
        },
        existing_edge_digests=(
            str(value.get("edge_digest") or "")
            for value in graph.get("edges", {}).values()
        ),
    )
    execution = (
        service.execute_commands(
            commands,
            idempotency_key=(
                f"solve-target:precursor-repair:{service.kernel.state.graph_revision}"
            ),
            include_scheduled=False,
        )
        if commands
        else {"changed": False, "executed_command_count": 0, "material_events": []}
    )
    accepted_repairs = [value for value in repairs if value.get("accepted") is True]
    return {
        "stage": "product_grounded_precursor_repair",
        "status": "completed" if accepted_repairs else "reused_or_empty",
        "examined_edge_count": len(repairs),
        "accepted_repair_count": len(accepted_repairs),
        "repairs": repairs,
        "execution": execution,
        "semantics": {
            "repair_is_new_hypothesis": True,
            "original_rejection_preserved": True,
            "reaction_revalidation_required": True,
            "hosted_model_calls": 0,
        },
    }


def discover_director_source_hints(
    service: RetrosynthesisCampaignService,
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    plans = [
        dict(value.get("plan") or {})
        for value in outcomes
        if isinstance(value, Mapping) and isinstance(value.get("plan"), Mapping)
    ]
    frontier = [
        dict(row)
        for plan in plans
        for row in plan.get("source_plan") or []
        if isinstance(row, Mapping)
    ]
    raw_hints = [
        str(hint)
        for plan in plans
        for skeleton in plan.get("multi_step_skeletons") or []
        if isinstance(skeleton, Mapping)
        for step in skeleton.get("steps") or []
        if isinstance(step, Mapping)
        for hint in step.get("source_hints") or []
        if str(hint).strip()
    ]
    sources: list[dict[str, Any]] = []
    for hint in raw_hints:
        source_ref = canonical_traceable_source_ref(hint)
        if not source_ref:
            continue
        sources.append(
            {
                "source_ref": source_ref,
                "source_kind": _source_kind(source_ref),
                "title": hint,
                "provenance": "global_director_source_acquisition_hint",
                "discovered_by": "codex_global_director",
            }
        )
    graph = service.graph_store.load()
    execution: Mapping[str, Any] = {"executed_command_count": 0, "material_events": []}
    if sources:
        execution = service.execute_commands(
            (
                _command(
                    service,
                    "discover_sources",
                    {
                        "sources": sources,
                        "existing_edge_digests": [
                            str(edge.get("edge_digest") or "")
                            for edge in graph["edges"].values()
                        ],
                    },
                    task_kind="evidence",
                    suffix="director-hints",
                ),
            ),
            idempotency_key=(
                f"solve-target:source-hints:{service.kernel.state.graph_revision}"
            ),
            include_scheduled=False,
        )
    return {
        "stage": "source_acquisition_frontier",
        "status": "completed" if sources else "unresolved",
        "source_plan": frontier,
        "traceable_source_hint_count": len(sources),
        "sources": sources,
        "execution": execution,
        "semantics": {
            "source_hint_is_not_exact_evidence": True,
            "structured_extraction_required_for_B3": True,
        },
    }


def ingest_source_discovery_observation(
    service: RetrosynthesisCampaignService,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist connector-discovered source lifecycle through normal workers."""

    sources = [
        dict(value)
        for value in discovery.get("sources") or []
        if isinstance(value, Mapping)
    ]
    if not sources:
        return {
            "stage": "source_discovery_ingestion",
            "status": "not_needed",
            "source_count": 0,
            "execution": {"executed_command_count": 0, "material_events": []},
        }
    graph = service.graph_store.load()
    execution = service.execute_commands(
        (
            _command(
                service,
                "discover_sources",
                {
                    "sources": sources,
                    "existing_exact_records": list(
                        dict(graph.get("exact_records") or {}).values()
                    ),
                    "existing_edge_digests": [
                        str(edge.get("edge_digest") or "")
                        for edge in dict(graph.get("edges") or {}).values()
                        if isinstance(edge, Mapping)
                    ],
                },
                task_kind="evidence",
                suffix=f"connector-{str(discovery.get('content_sha256') or '')[:20]}",
            ),
        ),
        idempotency_key=(
            "solve-target:connector-source-discovery:"
            f"{str(discovery.get('content_sha256') or '')}"
        ),
        include_scheduled=False,
    )
    return {
        "stage": "source_discovery_ingestion",
        "status": "completed" if execution.get("changed") else "reused",
        "source_count": len(sources),
        "execution": execution,
        "semantics": {
            "source_lifecycle_entered_canonical_graph": True,
            "discovery_grants_no_exact_evidence": True,
        },
    }


def materialize_discovered_source_routes(
    service: RetrosynthesisCampaignService,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit target-connected source DAGs through the one V4 proposal path."""

    observations: list[dict[str, Any]] = []
    rejected_observations: list[dict[str, Any]] = []
    for source in discovery.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        raw = source.get("source_route_observation")
        if not isinstance(raw, Mapping) or not raw:
            continue
        observation = dict(raw)
        supplied = str(observation.pop("content_sha256", "") or "")
        reasons: list[str] = []
        if observation.get("schema_version") != (
            "deterministic_source_route_observation.v1"
        ):
            reasons.append("source_route_observation_schema_invalid")
        if not supplied or supplied != _stable_digest(observation):
            reasons.append("source_route_observation_digest_invalid")
        proposals = [
            dict(row)
            for row in observation.get("proposals") or []
            if isinstance(row, Mapping)
        ]
        if not proposals or len(proposals) > 64:
            reasons.append("source_route_proposal_count_invalid")
        if reasons:
            rejected_observations.append(
                {
                    "source_ref": str(source.get("source_ref") or ""),
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        observation["content_sha256"] = supplied
        observations.append(observation)
    if not observations:
        return {
            "stage": "source_route_materialization",
            "status": "not_needed" if not rejected_observations else "rejected",
            "observation_count": 0,
            "proposal_count": 0,
            "rejected_observations": rejected_observations,
            "execution": {"executed_command_count": 0, "material_events": []},
        }

    route_families: dict[str, dict[str, Any]] = {}
    hypotheses: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for observation in observations:
        route_family = dict(observation.get("route_family") or {})
        alias = str(route_family.get("route_family_id") or "")
        if not alias:
            continue
        route_families[alias] = route_family
        for raw in observation.get("proposals") or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            condition = dict(row.get("condition_candidate") or {})
            proposal = {
                **row,
                "origin_kind": "literature_source_route",
                "route_family_id": alias,
                "condition_predictions": (
                    [
                        {
                            **condition,
                            "source_ref": str(row.get("source_ref") or ""),
                            "source_location": dict(row.get("source_location") or {}),
                            "authority_scope": (
                                "source_text_condition_candidate"
                            ),
                            "not_reaction_proof": True,
                        }
                    ]
                    if condition
                    else []
                ),
            }
            proposals.append(proposal)
            hypotheses.append(
                {
                    "step_id": str(
                        row.get("proposal_id") or row.get("step_id") or ""
                    ),
                    "product_smiles": str(row.get("product_smiles") or ""),
                    "precursor_smiles": list(
                        row.get("precursor_smiles")
                        or row.get("reactant_smiles")
                        or []
                    ),
                    "origin_kind": "literature_source_route",
                    "origin_ref": str(row.get("source_ref") or ""),
                    "route_family_id": alias,
                    "transformation_hypothesis": str(
                        row.get("transformation_hypothesis") or ""
                    ),
                    "condition_predictions": proposal["condition_predictions"],
                    "frontier_priority": 0.95,
                }
            )
    if not proposals:
        return {
            "stage": "source_route_materialization",
            "status": "rejected",
            "observation_count": len(observations),
            "proposal_count": 0,
            "reason": "source_route_family_or_proposals_missing",
            "rejected_observations": rejected_observations,
            "execution": {"executed_command_count": 0, "material_events": []},
        }

    observation_identity = _stable_digest(
        sorted(str(row.get("content_sha256") or "") for row in observations)
    )
    family_ingestion = service.apply_batch(
        CanonicalIngestionBatch(
            route_families=tuple(route_families.values()),
            hypotheses=tuple(hypotheses),
        ),
        idempotency_key=f"source-route-families:{observation_identity}",
    )
    graph = service.graph_store.load()
    commands = materialization_commands_for_proposals(
        proposals,
        run_id=service.kernel.spec.run_id,
        input_revision=service.kernel.state.graph_revision,
        dependency_revisions={
            "graph_revision": service.kernel.state.graph_revision,
            "evidence_revision": service.kernel.state.evidence_revision,
        },
        existing_edge_digests=(
            str(edge.get("edge_digest") or "")
            for edge in dict(graph.get("edges") or {}).values()
            if isinstance(edge, Mapping)
        ),
    )
    execution = (
        service.execute_commands(
            commands,
            idempotency_key=f"source-route-materialize:{observation_identity}",
            include_scheduled=False,
        )
        if commands
        else {"changed": False, "executed_command_count": 0, "material_events": []}
    )
    materialized_graph = service.graph_store.load()
    source_family_aliases = set(route_families)
    source_family_ids = {
        str(route_id)
        for route_id, raw_route in dict(
            materialized_graph.get("route_families") or {}
        ).items()
        if isinstance(raw_route, Mapping)
        and source_family_aliases.intersection(
            str(value) for value in raw_route.get("aliases") or []
        )
    }
    materialized_edge_ids = sorted(
        str(edge_id)
        for edge_id, raw_edge in dict(
            materialized_graph.get("edges") or {}
        ).items()
        if isinstance(raw_edge, Mapping)
        and source_family_ids.intersection(
            str(value) for value in raw_edge.get("route_family_ids") or []
        )
    )
    return {
        "stage": "source_route_materialization",
        "status": (
            "completed"
            if execution.get("changed") or family_ingestion.get("changed")
            else "reused_or_empty"
        ),
        "observation_count": len(observations),
        "route_family_count": len(route_families),
        "proposal_count": len(proposals),
        "materialized_edge_ids": materialized_edge_ids,
        "materialization_command_count": len(commands),
        "rejected_observations": rejected_observations,
        "family_ingestion": family_ingestion,
        "execution": execution,
        "material_events": (
            ["target_connected_source_route_materialized"]
            if execution.get("changed") or family_ingestion.get("changed")
            else []
        ),
        "semantics": {
            "canonical_hypergraph_is_only_state_authority": True,
            "source_route_is_proposal_not_proof": True,
            "existing_edges_gain_route_membership_without_duplicate_expansion": True,
            "reaction_validation_required": True,
        },
    }


def audit_live_benchmark_stock(
    service: RetrosynthesisCampaignService,
    *,
    catalog_builder: StockCatalogBuilder | None = None,
    max_molecules: int = 24,
) -> dict[str, Any]:
    graph = service.graph_store.load()
    selection = _selected_stock_audit_molecules(
        graph,
        max_molecules=max_molecules,
    )
    leaf_ids = selection["leaf_molecule_ids"]
    candidate_ids = selection["stock_candidate_molecule_ids"]
    if not leaf_ids:
        return {
            "stage": "benchmark_stock",
            "status": "unresolved",
            "selected_leaf_count": 0,
            "reason": "selected_route_leaves_missing",
            "execution": {"executed_command_count": 0},
        }
    if selection["limit_exceeded"]:
        return {
            "stage": "benchmark_stock",
            "status": "unresolved",
            "selected_leaf_count": len(leaf_ids),
            "selected_stock_candidate_count": 0,
            "internal_stock_candidate_count": 0,
            "reason": "selected_leaf_count_exceeds_stock_audit_limit",
            "max_molecules": max_molecules,
            "execution": {"executed_command_count": 0},
        }
    pending_candidate_ids = [
        molecule_id
        for molecule_id in candidate_ids
        if not _has_recent_boundary_audit(
            graph,
            molecule_id,
            required="benchmark_search",
        )
    ]
    if candidate_ids and not pending_candidate_ids:
        closed_leaf_count = sum(
            _has_boundary_observation(
                graph,
                molecule_id,
                required="benchmark_search",
            )
            for molecule_id in leaf_ids
        )
        closed_candidate_count = sum(
            _has_boundary_observation(
                graph,
                molecule_id,
                required="benchmark_search",
            )
            for molecule_id in candidate_ids
        )
        return {
            "stage": "benchmark_stock",
            "status": "reused",
            "selected_leaf_count": len(leaf_ids),
            "selected_stock_candidate_count": len(candidate_ids),
            "internal_stock_candidate_count": selection[
                "internal_stock_candidate_count"
            ],
            "stock_closed_leaf_count": closed_leaf_count,
            "stock_closed_candidate_count": closed_candidate_count,
            "miss_count": len(candidate_ids) - closed_candidate_count,
            "execution": {"executed_command_count": 0},
        }
    candidate_smiles = [
        graph["molecules"][molecule_id]["canonical_smiles"]
        for molecule_id in pending_candidate_ids
    ]
    builder = catalog_builder or build_pubchem_vendor_catalog
    catalog = dict(builder(candidate_smiles, max_molecules=max_molecules))
    ref = service.kernel.artifacts.put_json(
        catalog,
        logical_name="live_benchmark_stock_catalog.json",
        producer="autoplanner.live_stock.pubchem",
    ).to_dict()
    service.register_artifact_authorities({ref["sha256"]: "benchmark_stock_catalog"})
    timestamp = str(catalog.get("retrieved_at") or _utc_now())
    execution = service.execute_commands(
        (
            _command(
                service,
                "audit_benchmark_leaf_stock",
                {
                    "target_smiles": service.kernel.spec.target_smiles,
                    "selected_deep_leaves": [
                        {
                            "leaf_id": molecule_id,
                            "smiles": graph["molecules"][molecule_id]["canonical_smiles"],
                        }
                        for molecule_id in pending_candidate_ids
                    ],
                    "catalog_artifact_sha256": ref["sha256"],
                    "as_of": timestamp,
                    "max_age_days": 30,
                },
                task_kind="stock",
                suffix="live-benchmark-leaves",
                artifact_refs=(ref,),
            ),
        ),
        idempotency_key=f"solve-target:benchmark-stock:{service.kernel.state.graph_revision}",
    )
    updated = service.graph_store.load()
    closed_leaf_count = sum(
        _has_boundary_observation(
            updated,
            molecule_id,
            required="benchmark_search",
        )
        for molecule_id in leaf_ids
    )
    closed_candidate_count = sum(
        _has_boundary_observation(
            updated,
            molecule_id,
            required="benchmark_search",
        )
        for molecule_id in candidate_ids
    )
    return {
        "stage": "benchmark_stock",
        "status": (
            "completed"
            if closed_candidate_count == len(candidate_ids)
            else "partial"
        ),
        "selected_leaf_count": len(leaf_ids),
        "selected_stock_candidate_count": len(candidate_ids),
        "newly_queried_candidate_count": len(pending_candidate_ids),
        "internal_stock_candidate_count": selection[
            "internal_stock_candidate_count"
        ],
        "catalog_ref": ref,
        "catalog_summary": {
            key: catalog.get(key)
            for key in (
                "adapter_version",
                "catalog_version",
                "requested_molecule_count",
                "queried_molecule_count",
                "truncated",
            )
        },
        "member_count": len(catalog.get("members") or []),
        "stock_closed_leaf_count": closed_leaf_count,
        "stock_closed_candidate_count": closed_candidate_count,
        "miss_count": len(candidate_ids) - closed_candidate_count,
        "execution": execution,
    }


def audit_authoritative_inventory_stock(
    service: RetrosynthesisCampaignService,
    *,
    inventory_builder: InventorySnapshotBuilder,
    required_boundary: str,
    max_molecules: int = 24,
    max_age_days: float = 30.0,
) -> dict[str, Any]:
    """Freeze and audit leaves plus useful internal procurement cut points."""

    graph = service.graph_store.load()
    selection = _selected_stock_audit_molecules(
        graph,
        max_molecules=max_molecules,
    )
    leaf_ids = selection["leaf_molecule_ids"]
    candidate_ids = selection["stock_candidate_molecule_ids"]
    if not leaf_ids:
        return {
            "stage": "authoritative_inventory_stock",
            "status": "unresolved",
            "reason": "selected_route_leaves_missing",
            "selected_leaf_count": 0,
            "execution": {"executed_command_count": 0},
        }
    if selection["limit_exceeded"]:
        return {
            "stage": "authoritative_inventory_stock",
            "status": "unresolved",
            "reason": "selected_leaf_count_exceeds_inventory_audit_limit",
            "selected_leaf_count": len(leaf_ids),
            "selected_stock_candidate_count": 0,
            "internal_stock_candidate_count": 0,
            "max_molecules": max_molecules,
            "execution": {"executed_command_count": 0},
        }
    if all(
        _has_recent_boundary_audit(
            graph,
            molecule_id,
            required=required_boundary,
            max_age_days=max_age_days,
        )
        for molecule_id in candidate_ids
    ):
        closed_leaf_count = sum(
            _has_boundary_observation(
                graph,
                molecule_id,
                required=required_boundary,
            )
            for molecule_id in leaf_ids
        )
        closed_candidate_count = sum(
            _has_boundary_observation(
                graph,
                molecule_id,
                required=required_boundary,
            )
            for molecule_id in candidate_ids
        )
        return {
            "stage": "authoritative_inventory_stock",
            "status": "reused",
            "selected_leaf_count": len(leaf_ids),
            "selected_stock_candidate_count": len(candidate_ids),
            "internal_stock_candidate_count": selection[
                "internal_stock_candidate_count"
            ],
            "stock_closed_leaf_count": closed_leaf_count,
            "stock_closed_candidate_count": closed_candidate_count,
            "miss_count": len(candidate_ids) - closed_candidate_count,
            "execution": {"executed_command_count": 0},
        }
    candidate_smiles = [
        graph["molecules"][molecule_id]["canonical_smiles"]
        for molecule_id in candidate_ids
    ]
    inventory = dict(
        inventory_builder(
            candidate_smiles,
            boundary=required_boundary,
            max_molecules=max_molecules,
        )
    )
    ref = service.kernel.artifacts.put_json(
        inventory,
        logical_name="versioned_inventory_snapshot.json",
        producer="autoplanner.authoritative_inventory",
    ).to_dict()
    service.register_artifact_authorities({ref["sha256"]: "inventory_snapshot_set"})
    timestamp = str(inventory.get("retrieved_at") or _utc_now())
    execution = service.execute_commands(
        (
            _command(
                service,
                "audit_deep_leaf_stock",
                {
                    "target_smiles": service.kernel.spec.target_smiles,
                    "selected_deep_leaves": [
                        {
                            "leaf_id": molecule_id,
                            "smiles": graph["molecules"][molecule_id]["canonical_smiles"],
                        }
                        for molecule_id in candidate_ids
                    ],
                    "inventory_artifact_sha256": ref["sha256"],
                    "as_of": timestamp,
                    "max_age_days": max_age_days,
                },
                task_kind="stock",
                suffix=f"inventory-{ref['sha256'][:20]}",
                artifact_refs=(ref,),
            ),
        ),
        idempotency_key=(
            f"solve-target:inventory-stock:{service.kernel.state.graph_revision}:"
            f"{ref['sha256']}"
        ),
    )
    updated = service.graph_store.load()
    closed_leaf_count = sum(
        _has_boundary_observation(updated, molecule_id, required=required_boundary)
        for molecule_id in leaf_ids
    )
    closed_candidate_count = sum(
        _has_boundary_observation(updated, molecule_id, required=required_boundary)
        for molecule_id in candidate_ids
    )
    return {
        "stage": "authoritative_inventory_stock",
        "status": (
            "completed"
            if closed_leaf_count == len(leaf_ids)
            else "partial"
        ),
        "selected_leaf_count": len(leaf_ids),
        "selected_stock_candidate_count": len(candidate_ids),
        "internal_stock_candidate_count": selection[
            "internal_stock_candidate_count"
        ],
        "stock_closed_leaf_count": closed_leaf_count,
        "stock_closed_candidate_count": closed_candidate_count,
        "inventory_ref": ref,
        "inventory_summary": {
            "adapter_version": inventory.get("adapter_version"),
            "inventory_version": inventory.get("inventory_version"),
            "retrieved_at": inventory.get("retrieved_at"),
            "offer_count": len(inventory.get("offers") or []),
        },
        "execution": execution,
        "semantics": {
            "snapshot_is_host_frozen": True,
            "every_selected_leaf_is_audited": True,
            "internal_procurement_cut_points_are_audited": True,
            "missing_offer_fails_closed": True,
        },
    }


def _selected_stock_audit_molecules(
    graph: Mapping[str, Any],
    *,
    max_molecules: int,
) -> dict[str, Any]:
    """Select bounded stock cut points without sacrificing mandatory leaf audits.

    Deep leaves remain mandatory.  Remaining capacity is spent first on products
    of unvalidated edges, because purchasing such an intermediate can preserve a
    valid downstream route without pretending that the rejected upstream edge is
    sound.  Other internal products are useful alternative procurement cuts.
    """

    selected_routes = [
        dict(route)
        for route in dict(graph.get("route_families") or {}).values()
        if isinstance(route, Mapping) and route.get("selected") is not False
    ]
    leaf_ids = sorted(
        {
            str(molecule_id)
            for route in selected_routes
            for molecule_id in route.get("leaf_molecule_ids") or []
            if str(molecule_id)
        }
    )
    if len(leaf_ids) > max_molecules:
        return {
            "leaf_molecule_ids": leaf_ids,
            "stock_candidate_molecule_ids": [],
            "internal_stock_candidate_count": 0,
            "limit_exceeded": True,
        }
    edge_ids = {
        str(edge_id)
        for route in selected_routes
        for edge_id in route.get("edge_ids") or []
        if str(edge_id)
    }
    edges = dict(graph.get("edges") or {})
    target_id = str(graph.get("target_molecule_id") or "")
    leaf_set = set(leaf_ids)
    rejected_products: set[str] = set()
    other_products: set[str] = set()
    for edge_id in sorted(edge_ids):
        edge = dict(edges.get(edge_id) or {})
        product_id = str(edge.get("product_molecule_id") or "")
        if not product_id or product_id == target_id or product_id in leaf_set:
            continue
        active_proofs = active_reaction_proofs(edge.get("reaction_proofs") or [])
        if not active_proofs or not any(
            proof.get("accepted") is True for proof in active_proofs
        ):
            rejected_products.add(product_id)
        else:
            other_products.add(product_id)
    capacity = max(0, max_molecules - len(leaf_ids))
    internal_ids = (
        sorted(rejected_products)
        + sorted(other_products - rejected_products)
    )[:capacity]
    return {
        "leaf_molecule_ids": leaf_ids,
        "stock_candidate_molecule_ids": leaf_ids + internal_ids,
        "internal_stock_candidate_count": len(internal_ids),
        "limit_exceeded": False,
    }


def _command(
    service: RetrosynthesisCampaignService,
    worker_type: str,
    payload: Mapping[str, Any],
    *,
    task_kind: str,
    suffix: str,
    artifact_refs: tuple[Mapping[str, Any], ...] = (),
) -> WorkerCommand:
    revision = service.kernel.revision
    return WorkerCommand(
        command_id=f"solve-target:{worker_type}:{suffix}",
        run_id=service.kernel.spec.run_id,
        worker_type=worker_type,
        input_revision=revision.graph_revision,
        idempotency_key=(
            f"solve-target:{worker_type}:{suffix}:{revision.graph_revision}"
        ),
        payload=dict(payload),
        budget=WorkerBudget(task_kind=task_kind, timeout_s=180.0),
        dependency_revisions={
            "graph_revision": revision.graph_revision,
            "evidence_revision": revision.evidence_revision,
        },
        artifact_refs=tuple(dict(value) for value in artifact_refs),
    )


def _source_kind(source_ref: str) -> str:
    if source_ref.startswith("doi:"):
        return "paper_si"
    if source_ref.startswith("patent:"):
        return "patent"
    return "paper_si"


def _has_boundary_observation(
    graph: Mapping[str, Any],
    molecule_id: str,
    *,
    required: str,
) -> bool:
    molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
    observation = dict(
        dict(graph.get("stock_observations") or {}).get(
            str(molecule.get("active_stock_observation_id") or "")
        )
        or {}
    )
    return bool(
        observation.get("accepted") is True
        and stock_boundary_matches(observation, required=required)
    )


def _has_recent_boundary_audit(
    graph: Mapping[str, Any],
    molecule_id: str,
    *,
    required: str,
    max_age_days: float = 30.0,
) -> bool:
    molecule = dict(dict(graph.get("molecules") or {}).get(molecule_id) or {})
    observation = dict(
        dict(graph.get("stock_observations") or {}).get(
            str(molecule.get("active_stock_observation_id") or "")
        )
        or {}
    )
    if not observation or not stock_boundary_matches(observation, required=required):
        return False
    if str(observation.get("molecule_id") or "") != molecule_id:
        return False
    if (
        observation.get("accepted") is not True
        and observation.get("authority_valid") is not True
    ):
        return False
    timestamp = str(observation.get("audited_as_of") or "").replace("Z", "+00:00")
    try:
        audited_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    if audited_at.tzinfo is None:
        audited_at = audited_at.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - audited_at).total_seconds() / 86_400.0
    return -5 / 1_440 <= age_days <= max(0.0, float(max_age_days))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "InventorySnapshotBuilder",
    "StockCatalogBuilder",
    "audit_authoritative_inventory_stock",
    "audit_live_benchmark_stock",
    "discover_director_source_hints",
    "ingest_source_discovery_observation",
    "repair_rejected_precursor_typos",
    "validate_materialized_edges",
]
