"""Deterministic stages used by the target-only campaign orchestrator."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.application.proof_policy import stock_boundary_matches
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
        if not any(
            isinstance(proof, Mapping) and proof.get("accepted") is True
            for proof in edge.get("reaction_proofs") or []
        )
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
                },
                task_kind="validation",
                suffix=str(edge["edge_digest"])[:24],
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
            for proof in dict(updated["edges"].get(edge_id) or {}).get(
                "reaction_proofs"
            )
            or []
        )
    )
    rejected_ids = sorted(set(edge_by_id) - set(accepted_ids))
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


def audit_live_benchmark_stock(
    service: RetrosynthesisCampaignService,
    *,
    catalog_builder: StockCatalogBuilder | None = None,
    max_molecules: int = 24,
) -> dict[str, Any]:
    graph = service.graph_store.load()
    leaf_ids = sorted(
        {
            str(molecule_id)
            for route in graph["route_families"].values()
            if route.get("selected") is not False
            for molecule_id in route.get("leaf_molecule_ids") or []
        }
    )
    if not leaf_ids:
        return {
            "stage": "benchmark_stock",
            "status": "unresolved",
            "selected_leaf_count": 0,
            "reason": "selected_route_leaves_missing",
            "execution": {"executed_command_count": 0},
        }
    if leaf_ids and all(
        _has_boundary_observation(
            graph,
            molecule_id,
            required="benchmark_search",
        )
        for molecule_id in leaf_ids
    ):
        return {
            "stage": "benchmark_stock",
            "status": "reused",
            "selected_leaf_count": len(leaf_ids),
            "execution": {"executed_command_count": 0},
        }
    leaves = [graph["molecules"][molecule_id]["canonical_smiles"] for molecule_id in leaf_ids]
    builder = catalog_builder or build_pubchem_vendor_catalog
    catalog = dict(builder(leaves, max_molecules=max_molecules))
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
                        for molecule_id in leaf_ids
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
    return {
        "stage": "benchmark_stock",
        "status": "completed" if not catalog.get("misses") else "partial",
        "selected_leaf_count": len(leaf_ids),
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
        "miss_count": len(catalog.get("misses") or []),
        "execution": execution,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "StockCatalogBuilder",
    "audit_live_benchmark_stock",
    "discover_director_source_hints",
    "repair_rejected_precursor_typos",
    "validate_materialized_edges",
]
