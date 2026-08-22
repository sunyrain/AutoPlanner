"""Deterministic workers that turn global retrosynthesis hypotheses into facts.

The workers in this module do not own a frontier or graph.  They apply cheap
admission gates, invoke existing host validators/providers, normalize evidence,
and return revision-bound records for the single campaign scheduler to ingest.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger

from cascade_planner.application.condition_predictions import (
    CONDITION_PREDICTION_RESULT_SCHEMA,
    normalize_condition_predictions,
)
from cascade_planner.application.chemical_strategy_critic import (
    critique_strategy_candidate,
)
from cascade_planner.application.biocatalytic_step_contract import (
    normalize_biocatalytic_step,
)
from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
    build_source_procedure_record,
    normalize_source_conditions,
)
from cascade_planner.application.worker_runtime import (
    WorkerArtifactReader,
    WorkerBudget,
    WorkerCommand,
    WorkerHandlerSpec,
    WorkerRuntimeError,
)
from cascade_planner.application.route_innovations import (
    merge_route_innovations,
    normalize_route_innovation,
)
from cascade_planner.application.strategy_contract import (
    normalize_reaction_operations,
    normalize_strategy_card,
    reaction_edit_digest,
    strategy_card_has_content,
)
from cascade_planner.harness.reaction_step_verifier import verify_reaction_step
from cascade_planner.application.reaction_proof_versions import (
    CURRENT_REACTION_VALIDATOR_VERSION,
)
from cascade_planner.providers import (
    BenchmarkCatalogStockProvider,
    ProviderContext,
    ProviderResultEnvelope,
    SnapshotStockProvider,
    StockBoundary,
)
from cascade_planner.providers.stock import (
    canonicalize_stock_snapshot,
    stock_snapshot_sha256,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    independent_source_group,
    source_document_identity,
)


RDLogger.DisableLog("rdApp.*")
WORKER_SET_VERSION = "autoplanner.retrosynthesis_workers.v1"
PROOF_STATE_SCHEMA = "retrosynthesis_proof_state.v1"
SOURCE_BINDING_SCHEMA = "normalized_source_binding.v1"
EXACT_SOURCE_RECORD_SCHEMA = "exact_source_reaction_record.v1"
SOURCE_CONFLICT_SCHEMA = "source_evidence_conflict.v1"
INVENTORY_SNAPSHOT_SET_SCHEMA = "inventory_snapshot_set.v1"
VERSIONED_INVENTORY_ARTIFACT_SCHEMA = "versioned_inventory_snapshot.v1"
VERSIONED_BENCHMARK_CATALOG_SCHEMA = "versioned_benchmark_stock_catalog.v1"
DEEP_LEAF_AUDIT_SCHEMA = "deep_leaf_stock_audit.v1"
STRUCTURED_EXTRACTION_SCHEMA = "structured_exact_row_extraction.v1"
_SOURCE_KINDS = {
    "patent",
    "paper_si",
    "curated_registry",
    "image_extraction",
    "codex_claim",
}
_CONDITION_KEYS = (
    "addition_order",
    "atmosphere",
    "base",
    "catalyst",
    "concentration",
    "equivalents",
    "oxidant",
    "pressure",
    "purification",
    "reductant",
    "reagents",
    "scale",
    "solvent",
    "temperature",
    "temperature_c",
    "time",
    "yield",
    "yield_percent",
    "workup",
)
_TRUSTED_EXTRACTION_PRODUCERS = {
    "deterministic_structure_parser",
    "human_curator",
    "typed_connector_structured_extraction",
    "ocr_structured_table_extraction",
    "manual_structured_extraction",
}


def build_retrosynthesis_worker_handlers() -> dict[str, WorkerHandlerSpec]:
    """Return the bounded P4 worker registry used by :class:`WorkerRuntime`."""
    specs = (
        WorkerHandlerSpec(
            "materialize_candidate",
            WORKER_SET_VERSION,
            "proposal",
            materialize_candidate_worker,
        ),
        WorkerHandlerSpec(
            "validate_reaction",
            f"{WORKER_SET_VERSION}+{CURRENT_REACTION_VALIDATOR_VERSION}",
            "validation",
            validate_reaction_worker,
        ),
        WorkerHandlerSpec(
            "record_condition_predictions",
            WORKER_SET_VERSION,
            "other",
            record_condition_predictions_worker,
        ),
        WorkerHandlerSpec(
            "discover_sources",
            WORKER_SET_VERSION,
            "evidence",
            discover_sources_worker,
        ),
        WorkerHandlerSpec(
            "extract_exact_source",
            WORKER_SET_VERSION,
            "evidence",
            extract_exact_source_worker,
        ),
        WorkerHandlerSpec(
            "detect_source_conflicts",
            WORKER_SET_VERSION,
            "validation",
            detect_source_conflicts_worker,
        ),
        WorkerHandlerSpec(
            "audit_deep_leaf_stock",
            WORKER_SET_VERSION,
            "stock",
            audit_deep_leaf_stock_worker,
        ),
        WorkerHandlerSpec(
            "audit_benchmark_leaf_stock",
            WORKER_SET_VERSION,
            "stock",
            audit_benchmark_leaf_stock_worker,
        ),
    )
    return {spec.worker_type: spec for spec in specs}


def materialization_commands_for_global_plan(
    plan: Mapping[str, Any],
    *,
    run_id: str,
    input_revision: int,
    dependency_revisions: Mapping[str, str | int] | None = None,
    existing_edge_digests: Iterable[str] = (),
) -> tuple[WorkerCommand, ...]:
    """Compile every unique Codex skeleton edge into a host-gated command.

    This is only a proposal-to-worker bridge.  It neither admits the edge nor
    writes a frontier.  Identical edges proposed in several route families are
    executed once while retaining every global-plan provenance reference.
    """
    grouped: dict[str, dict[str, Any]] = {}
    family_details = {
        str(value.get("route_family_id") or ""): dict(value)
        for value in plan.get("route_families") or []
        if isinstance(value, Mapping)
    }
    for skeleton in plan.get("multi_step_skeletons") or []:
        if not isinstance(skeleton, Mapping):
            continue
        skeleton_id = str(skeleton.get("skeleton_id") or "")
        route_family_id = str(skeleton.get("route_family_id") or "")
        for step in skeleton.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            product = str(step.get("product_smiles") or "").strip()
            precursors = _string_list(step.get("precursor_smiles"))
            operations = normalize_reaction_operations(
                step.get("reaction_operations") or ()
            )
            family = family_details.get(route_family_id, {})
            # The route-family card is the frozen policy authority.  A step
            # may carry a copied card for provenance, but it must not silently
            # replace the family identity. ReactionJSON edits are edge-level
            # provenance and are carried separately in ``reaction_edit_digest``.
            source_card = family.get("strategy_card") or step.get("strategy_card") or {}
            strategy_card = normalize_strategy_card(
                source_card,
                route_family_id=route_family_id,
            )
            strategy_ref = strategy_card if strategy_card_has_content(strategy_card) else {}
            identity = _digest(
                {
                    "product_smiles": product,
                    "precursor_smiles_multiset": sorted(precursors),
                    "reaction_edit_digest": reaction_edit_digest(operations),
                }
            )
            payload = grouped.setdefault(
                identity,
                {
                    "product_smiles": product,
                    "precursor_smiles": precursors,
                    "reagent_smiles": _string_list(step.get("reagent_smiles")),
                    "condition_predictions": [],
                    "biocatalytic_steps": [],
                    "reaction_operations": [dict(value) for value in operations],
                    "reactionjson_audit": dict(step.get("reactionjson_audit") or {}),
                    "strategy_cards": [],
                    "existing_edge_digests": sorted(
                        {str(value) for value in existing_edge_digests if str(value)}
                    ),
                    "proposal_refs": [],
                },
            )
            payload["condition_predictions"] = _merge_annotation_rows(
                payload.get("condition_predictions"),
                step.get("condition_predictions"),
            )
            if isinstance(step.get("biocatalytic_step"), Mapping) and step.get(
                "biocatalytic_step"
            ):
                payload["biocatalytic_steps"] = _merge_digest_rows(
                    payload.get("biocatalytic_steps"),
                    [step.get("biocatalytic_step")],
                )
            if strategy_card_has_content(strategy_card):
                payload["strategy_cards"] = _merge_strategy_cards(
                    payload.get("strategy_cards"),
                    [strategy_card],
                )
            payload["proposal_refs"].append(
                {
                    "origin_kind": "codex_global_director",
                    "route_family_id": route_family_id,
                    "skeleton_id": skeleton_id,
                    "step_id": str(step.get("step_id") or ""),
                    "transformation_hypothesis": str(
                        step.get("transformation_hypothesis") or ""
                    ),
                    "strategy_id": str(strategy_ref.get("strategy_id") or ""),
                    "strategy_digest": str(
                        strategy_ref.get("strategy_digest") or ""
                    ),
                    "reaction_edit_digest": reaction_edit_digest(operations),
                    "strategy_anchor": step.get("strategy_anchor") is True,
                    "execution_domain": str(
                        step.get("execution_domain") or "chemical"
                    ),
                }
            )
    commands: list[WorkerCommand] = []
    for identity, payload in sorted(grouped.items()):
        payload["proposal_refs"] = sorted(
            payload["proposal_refs"],
            key=lambda row: (
                row["route_family_id"],
                row["skeleton_id"],
                row["step_id"],
            ),
        )
        work_identity = _digest(
            {"edge_identity": identity, "proposal_refs": payload["proposal_refs"]}
        )
        command_identity = _digest(
            {
                "work_identity": work_identity,
                "input_revision": int(input_revision),
                "dependency_revisions": dict(dependency_revisions or {}),
            }
        )
        commands.append(
            WorkerCommand(
                command_id=f"materialize:{command_identity[:24]}",
                run_id=run_id,
                worker_type="materialize_candidate",
                input_revision=int(input_revision),
                idempotency_key=f"materialize:{command_identity}",
                payload=payload,
                budget=WorkerBudget(task_kind="proposal"),
                dependency_revisions=dict(dependency_revisions or {}),
            )
        )
    return tuple(commands)


def condition_prediction_commands_for_edges(
    predictions: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    input_revision: int,
    dependency_revisions: Mapping[str, str | int] | None = None,
    maximum_candidates: int = 2,
) -> tuple[WorkerCommand, ...]:
    """Bind raw predictor output to canonical edge digests for replayable ingestion."""

    commands: list[WorkerCommand] = []
    for value in predictions:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        edge_digest = str(row.get("edge_digest") or "").strip()
        reaction_smiles = str(row.get("reaction_smiles") or "").strip()
        if not edge_digest or ">>" not in reaction_smiles:
            continue
        identity = _digest(
            {
                "edge_digest": edge_digest,
                "reaction_smiles": reaction_smiles,
                "raw_predictions": row.get("raw_predictions") or [],
                "prediction_error": str(row.get("prediction_error") or ""),
                "condition_model": str(row.get("condition_model") or ""),
                "input_revision": int(input_revision),
                "dependency_revisions": {
                    str(key): value
                    for key, value in sorted(
                        dict(dependency_revisions or {}).items()
                    )
                },
            }
        )
        commands.append(
            WorkerCommand(
                command_id=f"conditions:{identity[:24]}",
                run_id=run_id,
                worker_type="record_condition_predictions",
                input_revision=int(input_revision),
                idempotency_key=f"conditions:{identity}",
                payload={
                    "edge_digest": edge_digest,
                    "reaction_smiles": reaction_smiles,
                    "raw_predictions": row.get("raw_predictions") or [],
                    "prediction_error": str(row.get("prediction_error") or ""),
                    "condition_model": str(row.get("condition_model") or ""),
                    "prediction_producer": str(
                        row.get("prediction_producer") or "condition_enrichment"
                    ),
                    "maximum_candidates": max(
                        1, min(2, int(maximum_candidates or 2))
                    ),
                },
                budget=WorkerBudget(task_kind="other", uses_model=False),
                dependency_revisions=dict(dependency_revisions or {}),
            )
        )
    return tuple(commands)


def materialization_commands_for_proposals(
    proposals: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    input_revision: int,
    dependency_revisions: Mapping[str, str | int] | None = None,
    existing_edge_digests: Iterable[str] = (),
    ancestor_smiles_by_product: Mapping[str, Iterable[str]] | None = None,
) -> tuple[WorkerCommand, ...]:
    """Compile ChemEnzy/template/literature/manual proposals through one gate."""
    grouped: dict[str, dict[str, Any]] = {}
    existing_set = {str(value) for value in existing_edge_digests if str(value)}
    existing = sorted(existing_set)
    for raw in proposals:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        product = str(row.get("product_smiles") or "").strip()
        precursors = _string_list(
            row.get("precursor_smiles") or row.get("reactant_smiles")
        )
        operations = normalize_reaction_operations(
            row.get("reaction_operations") or ()
        )
        # Preserve the frozen policy identity from the canonical graph. The
        # per-edge ReactionJSON edit remains in the payload and proposal refs.
        raw_strategy_cards = [
            dict(value)
            for value in row.get("strategy_cards") or []
            if isinstance(value, Mapping)
        ]
        if not raw_strategy_cards and isinstance(row.get("strategy_card"), Mapping):
            raw_strategy_cards = [dict(row.get("strategy_card") or {})]
        strategy_cards = _merge_strategy_cards(
            (),
            [
                normalize_strategy_card(
                    value,
                    route_family_id=str(row.get("route_family_id") or ""),
                )
                for value in raw_strategy_cards
            ],
        )
        scalar_strategy_card = normalize_strategy_card(
            row.get("strategy_card") or {},
            route_family_id=str(row.get("route_family_id") or ""),
        )
        strategy_card = (
            scalar_strategy_card
            if strategy_card_has_content(scalar_strategy_card)
            else strategy_cards[0]
            if strategy_cards
            else {}
        )
        strategy_ref = strategy_card if strategy_card_has_content(strategy_card) else {}
        identity = _digest(
            {
                "product_smiles": product,
                "precursor_smiles_multiset": sorted(precursors),
                "reaction_edit_digest": reaction_edit_digest(operations),
            }
        )
        candidate_audit = audit_retrosynthetic_candidate(product, precursors)
        if str(candidate_audit.get("edge_digest") or "") in existing_set:
            continue
        payload = grouped.setdefault(
            identity,
            {
                "product_smiles": product,
                "precursor_smiles": precursors,
                "reagent_smiles": _string_list(row.get("reagent_smiles")),
                "condition_predictions": [],
                "biocatalytic_steps": [],
                "route_innovations": [],
                "reaction_operations": [dict(value) for value in operations],
                "reactionjson_audit": dict(row.get("reactionjson_audit") or {}),
                "strategy_cards": [],
                "route_innovation_reject_reasons": [],
                "existing_edge_digests": existing,
                "ancestor_smiles": sorted(
                    {
                        str(value)
                        for value in dict(ancestor_smiles_by_product or {}).get(
                            product,
                            (),
                        )
                        if str(value)
                    }
                ),
                "proposal_refs": [],
            },
        )
        payload["proposal_refs"].append(
            {
                "origin_kind": str(row.get("origin_kind") or "manual"),
                "origin_ref": str(
                    row.get("origin_ref")
                    or row.get("source_ref")
                    or row.get("model")
                    or ""
                ),
                "proposal_id": str(
                    row.get("proposal_id") or row.get("step_id") or ""
                ),
                "route_family_id": str(row.get("route_family_id") or ""),
                "canonical_route_family_ids": sorted(
                    {
                        str(value)
                        for value in row.get("canonical_route_family_ids") or []
                        if str(value)
                    }
                    | (
                        {str(row.get("canonical_route_family_id"))}
                        if row.get("canonical_route_family_id")
                        else set()
                    )
                ),
                "skeleton_id": str(row.get("skeleton_id") or ""),
                "transformation_hypothesis": str(
                    row.get("transformation_hypothesis") or ""
                ),
                "strategy_id": str(strategy_ref.get("strategy_id") or ""),
                "strategy_digest": str(strategy_ref.get("strategy_digest") or ""),
                "reaction_edit_digest": reaction_edit_digest(operations),
                "strategy_anchor": row.get("strategy_anchor") is True,
                "execution_domain": str(
                    row.get("execution_domain") or "chemical"
                ),
                "provider_reaction_metadata": (
                    dict(row.get("provider_reaction_metadata") or {})
                    if isinstance(row.get("provider_reaction_metadata"), Mapping)
                    else {}
                ),
            }
        )
        if strategy_cards:
            payload["strategy_cards"] = _merge_strategy_cards(
                payload.get("strategy_cards"),
                strategy_cards,
            )
        raw_innovations = [
            dict(value)
            for value in row.get("route_innovations") or []
            if isinstance(value, Mapping)
        ]
        singular_innovation = row.get("route_innovation") or row.get("innovation")
        if not raw_innovations and isinstance(singular_innovation, Mapping):
            raw_innovations = [dict(singular_innovation)]
        elif not raw_innovations and (
            row.get("innovation_kind") or row.get("proposal_basis")
        ):
            raw_innovations = [row]
        normalized_innovations: list[dict[str, Any]] = []
        for raw_innovation in raw_innovations:
            normalized, innovation_reasons = normalize_route_innovation(
                {
                    **row,
                    "route_innovation": raw_innovation,
                }
            )
            if innovation_reasons:
                payload["route_innovation_reject_reasons"].extend(
                    innovation_reasons
                )
            elif normalized:
                normalized_innovations.append(normalized)
        payload["route_innovations"] = merge_route_innovations(
            payload.get("route_innovations"),
            normalized_innovations,
        )
        payload["condition_predictions"] = _merge_annotation_rows(
            payload.get("condition_predictions"),
            row.get("condition_predictions"),
        )
        raw_biocatalytic_steps = [
            dict(value)
            for value in row.get("biocatalytic_steps") or []
            if isinstance(value, Mapping) and value
        ]
        if (
            not raw_biocatalytic_steps
            and isinstance(row.get("biocatalytic_step"), Mapping)
            and row.get("biocatalytic_step")
        ):
            raw_biocatalytic_steps = [dict(row.get("biocatalytic_step") or {})]
        if raw_biocatalytic_steps:
            payload["biocatalytic_steps"] = _merge_digest_rows(
                payload.get("biocatalytic_steps"),
                raw_biocatalytic_steps,
            )
    commands: list[WorkerCommand] = []
    for identity, payload in sorted(grouped.items()):
        payload["proposal_refs"] = sorted(
            payload["proposal_refs"],
            key=lambda row: (
                row["origin_kind"],
                row["origin_ref"],
                row["proposal_id"],
            ),
        )
        work_identity = _digest(
            {"edge_identity": identity, "proposal_refs": payload["proposal_refs"]}
        )
        command_identity = _digest(
            {
                "work_identity": work_identity,
                "input_revision": int(input_revision),
                "dependency_revisions": dict(dependency_revisions or {}),
            }
        )
        commands.append(
            WorkerCommand(
                command_id=f"materialize:{command_identity[:24]}",
                run_id=run_id,
                worker_type="materialize_candidate",
                input_revision=int(input_revision),
                idempotency_key=f"materialize:{command_identity}",
                payload=payload,
                budget=WorkerBudget(task_kind="proposal"),
                dependency_revisions=dict(dependency_revisions or {}),
            )
        )
    return tuple(commands)


def proof_state(
    *,
    hypothesis: bool = True,
    structural_materialized: bool = False,
    reaction_validated: bool = False,
    exact_source_bound: bool = False,
    independent_source_groups: Iterable[str] = (),
) -> dict[str, Any]:
    """Represent proof axes separately instead of using a misleading scalar."""
    groups = sorted({str(value) for value in independent_source_groups if str(value)})
    if reaction_validated and not structural_materialized:
        raise ValueError("reaction validation requires structural materialization")
    states = ["L0_hypothesis"] if hypothesis else []
    if structural_materialized:
        states.append("L1_structural_materialized")
    if reaction_validated:
        states.append("L2_reaction_validated")
    if exact_source_bound:
        states.append("L3_exact_source")
    independently_supported = exact_source_bound and len(groups) >= 2
    if independently_supported:
        states.append("L3_independently_supported")
    row = {
        "schema_version": PROOF_STATE_SCHEMA,
        "states": states,
        "hypothesis": bool(hypothesis),
        "structural_materialized": bool(structural_materialized),
        "reaction_validated": bool(reaction_validated),
        "exact_source_bound": bool(exact_source_bound),
        "independent_source_groups": groups,
        "independently_supported": independently_supported,
        "semantics": {
            "axes_are_not_interchangeable": True,
            "exact_source_does_not_imply_reaction_validation": True,
            "model_consensus_does_not_count_as_source_independence": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def materialize_candidate_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    """Apply all cheap gates and materialize one exact hyperedge identity."""
    del artifacts
    payload = dict(command.payload)
    product = payload.get("product_smiles")
    precursors = _string_list(
        payload.get("precursor_smiles") or payload.get("reactant_smiles")
    )
    audit = audit_retrosynthetic_candidate(
        product,
        precursors,
        forbidden_return_smiles=_string_list(
            payload.get("ancestor_smiles") or payload.get("forbidden_return_smiles")
        ),
    )
    reasons = list(audit.get("reasons") or [])
    reasons.extend(payload.get("route_innovation_reject_reasons") or [])
    edge_digest = str(audit.get("edge_digest") or "")
    existing = {
        str(value)
        for value in payload.get("existing_edge_digests") or []
        if str(value)
    }
    if edge_digest in existing:
        reasons.append("duplicate_reaction_edge")

    strategy_cards = [
        card
        for value in payload.get("strategy_cards") or []
        if isinstance(value, Mapping)
        for card in (normalize_strategy_card(value),)
        if strategy_card_has_content(card)
    ]
    if not strategy_cards and isinstance(payload.get("strategy_card"), Mapping):
        strategy_cards = [
            normalize_strategy_card(
                payload.get("strategy_card") or {},
            )
        ]
    biocatalytic_steps: list[dict[str, Any]] = []
    for raw_step in payload.get("biocatalytic_steps") or []:
        if not isinstance(raw_step, Mapping):
            continue
        raw_record = dict(raw_step)
        catalyst = dict(raw_record.get("catalyst_hypothesis") or {})
        normalized_step, _design_reasons = normalize_biocatalytic_step(
            raw_record,
            execution_domain=str(
                raw_record.get("execution_domain") or "enzymatic"
            ),
            product_smiles=product,
            precursor_smiles=precursors,
            enzyme_label=str(catalyst.get("enzyme_label") or ""),
            step_id=str(raw_record.get("step_id") or ""),
        )
        if normalized_step:
            biocatalytic_steps.append(normalized_step)
    biocatalytic_steps = _merge_digest_rows((), biocatalytic_steps)
    critic = critique_strategy_candidate(
        product_smiles=product,
        precursor_smiles=precursors,
        strategy_card=strategy_cards[0] if strategy_cards else {},
        reaction_operations=payload.get("reaction_operations") or (),
        reactionjson_audit=dict(payload.get("reactionjson_audit") or {}),
        reaction_family=" | ".join(
            str(value.get("transformation_hypothesis") or "")
            for value in payload.get("proposal_refs") or []
            if isinstance(value, Mapping)
        ),
        conditions=[
            item
            for prediction in payload.get("condition_predictions") or []
            if isinstance(prediction, Mapping)
            for item in prediction.get("reagents") or []
        ],
        catalyst=" | ".join(
            str(value.get("catalyst") or "")
            for value in payload.get("condition_predictions") or []
            if isinstance(value, Mapping)
        ),
        enzyme=" | ".join(
            str(value.get("enzyme") or "")
            for value in payload.get("condition_predictions") or []
            if isinstance(value, Mapping)
        ),
        is_strategy_defining_step=any(
            value.get("strategy_anchor") is True
            for value in payload.get("proposal_refs") or []
            if isinstance(value, Mapping)
        ),
    )
    proposal_refs = [
        dict(value)
        for value in payload.get("proposal_refs") or []
        if isinstance(value, Mapping)
    ]
    provider_template_topology = bool(
        not payload.get("reaction_operations")
        and any(
            str(value.get("origin_kind") or "") == "chemenzy"
            for value in proposal_refs
        )
    )
    critic_is_admission_authority = bool(
        any(strategy_card_has_content(card) for card in strategy_cards)
        or payload.get("reaction_operations")
    ) and not provider_template_topology
    if critic_is_admission_authority:
        reasons.extend(critic.get("blocking_reasons") or [])

    canonical_reagents: list[str] = []
    raw_reagents = _string_list(payload.get("reagent_smiles"))
    for reagent in raw_reagents:
        canonical = _canonical_smiles(reagent)
        if not canonical:
            reasons.append("invalid_reagent_smiles")
        else:
            canonical_reagents.append(canonical)
    reasons = sorted(set(reasons))
    accepted = not reasons and audit.get("accepted") is True
    candidate_id = f"edge:{edge_digest[:24]}" if edge_digest else ""
    result_payload = {
        "schema_version": "materialized_reaction_candidate.v2",
        "candidate_id": candidate_id,
        "edge_digest": edge_digest,
        "edge_identity": dict(audit.get("edge_identity") or {}),
        "product_smiles": str(audit.get("product_smiles") or ""),
        "precursor_smiles": list(audit.get("precursor_smiles_multiset") or []),
        "reagent_smiles": sorted(canonical_reagents),
        "condition_predictions": _merge_annotation_rows(
            (), payload.get("condition_predictions")
        ),
        "biocatalytic_steps": biocatalytic_steps,
        "route_innovations": merge_route_innovations(
            (), payload.get("route_innovations") or []
        ),
        "strategy_cards": strategy_cards,
        "reaction_operations": [
            dict(value)
            for value in payload.get("reaction_operations") or []
            if isinstance(value, Mapping)
        ],
        "reaction_edit_digest": reaction_edit_digest(
            payload.get("reaction_operations") or ()
        ),
        "reactionjson_audit": dict(payload.get("reactionjson_audit") or {}),
        "chemical_strategy_critic": critic,
        "admission_semantics": {
            "provider_template_topology": provider_template_topology,
            "reaction_credibility_reported_separately": provider_template_topology,
            "critic_is_admission_authority": critic_is_admission_authority,
        },
        "reaction_smiles": (
            ".".join(audit.get("precursor_smiles_multiset") or [])
            + ">>"
            + str(audit.get("product_smiles") or "")
            if accepted
            else ""
        ),
        "admission_audit": audit,
        "proof_state": proof_state(structural_materialized=accepted),
        "authority_scope": "search_admission_only",
        "accepted": accepted,
        "reasons": reasons,
        "proposal_refs": proposal_refs,
    }
    return {
        "status": "completed" if accepted else "rejected",
        "payload": result_payload,
        "failure_reasons": reasons,
        # One accepted hyperedge is one accepted expansion regardless of its
        # precursor count.  Rejected work reports no expansion ids.
        "accepted_expansion_ids": [edge_digest] if accepted else [],
    }


def record_condition_predictions_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    """Normalize advisory conditions while stripping any spoofed authority."""

    del artifacts
    payload = dict(command.payload)
    edge_digest = str(payload.get("edge_digest") or "")
    reaction_smiles = str(payload.get("reaction_smiles") or "")
    maximum = max(1, min(2, int(payload.get("maximum_candidates") or 2)))
    predictions = normalize_condition_predictions(
        payload.get("raw_predictions"),
        max_candidates=maximum,
        default_model=str(payload.get("condition_model") or ""),
        producer=str(payload.get("prediction_producer") or "condition_enrichment"),
    )
    error = str(payload.get("prediction_error") or "")
    reasons: list[str] = []
    if error:
        reasons.append("condition_predictor_failed")
    if not predictions:
        reasons.append("condition_prediction_empty")
    result_payload = {
        "schema_version": CONDITION_PREDICTION_RESULT_SCHEMA,
        "edge_digest": edge_digest,
        "reaction_smiles": reaction_smiles,
        "condition_predictions": predictions,
        "diagnostics": {
            "attempted": True,
            "returned_candidate_count": len(predictions),
            "maximum_candidates": maximum,
            "condition_model": str(payload.get("condition_model") or ""),
            "prediction_producer": str(
                payload.get("prediction_producer") or "condition_enrichment"
            ),
            "failure_reasons": sorted(set(reasons)),
            "error": error[:1_000],
        },
        "semantics": {
            "prediction_is_not_reaction_proof": True,
            "prediction_is_not_source_evidence": True,
            "source_procedure_supersedes_prediction": True,
        },
    }
    return {
        "status": "completed" if predictions else "partial",
        "payload": result_payload,
        "failure_reasons": sorted(set(reasons)),
        "material_events": ["condition_predictions_added"] if predictions else [],
    }


def _merge_annotation_rows(existing: Any, incoming: Any) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for value in [*(existing or []), *(incoming or [])]:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        row.setdefault("authority_scope", "model_predicted_condition")
        row.setdefault("not_reaction_proof", True)
        rows[_digest(row)] = row
    return [rows[key] for key in sorted(rows)]


def _merge_digest_rows(existing: Any, incoming: Any) -> list[dict[str, Any]]:
    """Merge canonical structured annotations without rewriting authority."""

    rows: dict[str, dict[str, Any]] = {}
    for value in [*(existing or []), *(incoming or [])]:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        key = str(row.get("content_sha256") or _digest(row))
        rows[key] = row
    return [rows[key] for key in sorted(rows)]


def _merge_strategy_cards(existing: Any, incoming: Any) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for value in [*(existing or []), *(incoming or [])]:
        if not isinstance(value, Mapping):
            continue
        card = normalize_strategy_card(value)
        rows[str(card["strategy_digest"])] = card
    return [rows[key] for key in sorted(rows)]


def validate_reaction_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    """Run the existing deterministic reaction verifier on a materialized edge."""
    del artifacts
    payload = dict(command.payload)
    requested_version = str(payload.get("validator_version") or "")
    if requested_version and requested_version != CURRENT_REACTION_VALIDATOR_VERSION:
        return {
            "status": "stale",
            "payload": {},
            "failure_reasons": ["reaction_validator_version_stale"],
        }
    candidate = dict(payload.get("candidate") or {})
    if (
        candidate.get("accepted") is not True
        or not candidate.get("product_smiles")
        or not candidate.get("precursor_smiles")
    ):
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["reaction_candidate_not_materialized"],
        }
    step = {
        "step_id": str(candidate.get("candidate_id") or "candidate"),
        "product_smiles": str(candidate["product_smiles"]),
        "reactant_smiles": list(candidate["precursor_smiles"]),
        "mapped_reaction_smiles": str(
            payload.get("mapped_reaction_smiles")
            or payload.get("atom_mapped_reaction_smiles")
            or ""
        ),
        "condition_candidate": dict(payload.get("condition_candidate") or {}),
        "evidence_bindings": list(payload.get("evidence_bindings") or []),
        "reaction_operations": [
            dict(value)
            for value in candidate.get("reaction_operations") or []
            if isinstance(value, Mapping)
        ],
        "reactionjson_audit": dict(candidate.get("reactionjson_audit") or {}),
        "biocatalytic_steps": [
            dict(value)
            for value in candidate.get("biocatalytic_steps") or []
            if isinstance(value, Mapping)
        ],
    }
    exact_source_records = [
        dict(row)
        for row in payload.get("exact_source_records") or []
        if isinstance(row, Mapping)
    ]
    source_procedure_records = [
        dict(row)
        for row in payload.get("source_procedure_records") or []
        if isinstance(row, Mapping)
    ]
    source_bindings = [
        dict(row)
        for row in payload.get("source_bindings") or []
        if isinstance(row, Mapping)
    ]
    proof = dict(
        verify_reaction_step(
            step,
            graph_and_stock_closed=payload.get("graph_and_stock_closed") is True,
            trusted_precedent_binding=dict(
                payload.get("trusted_precedent_binding") or {}
            ),
            procurement_binding=dict(payload.get("procurement_binding") or {}),
            trusted_stock_providers=dict(
                payload.get("trusted_stock_providers") or {}
            ),
            source_supported_multicentre=_exact_records_support_edge(
                exact_source_records,
                str(candidate.get("edge_digest") or ""),
            ),
            exact_source_records=exact_source_records,
            source_procedure_records=source_procedure_records,
            source_bindings=source_bindings,
        )
    )
    proof["exact_record_ids"] = sorted(
        str(row.get("record_id") or "")
        for row in exact_source_records
        if str(row.get("record_id") or "")
    )
    proof["procedure_record_ids"] = sorted(
        str(row.get("procedure_record_id") or "")
        for row in source_procedure_records
        if str(row.get("procedure_record_id") or "")
    )
    proof["source_binding_ids"] = sorted(
        str(row.get("binding_id") or "")
        for row in source_bindings
        if str(row.get("binding_id") or "")
    )
    proof.pop("proof_digest", None)
    proof["proof_digest"] = _digest(proof)
    validated = proof.get("accepted") is True
    state = proof_state(
        structural_materialized=True,
        reaction_validated=validated,
        exact_source_bound=bool(exact_source_records),
        independent_source_groups=(
            str(row.get("independence_group") or "")
            for row in exact_source_records
        ),
    )
    return {
        "status": "completed" if validated else "partial",
        "payload": {
            "schema_version": "validated_reaction_candidate.v1",
            "candidate_id": candidate.get("candidate_id"),
            "edge_digest": candidate.get("edge_digest"),
            "reaction_proof": proof,
            "proof_state": state,
        },
        "failure_reasons": [] if validated else list(proof.get("reasons") or []),
    }


def _exact_records_support_edge(
    records: Iterable[Mapping[str, Any]],
    edge_digest: str,
) -> bool:
    """Bind multicentre relaxation to canonical exact-source graph records."""

    for raw in records:
        row = dict(raw)
        supplied = str(row.get("content_sha256") or "")
        body = {key: value for key, value in row.items() if key != "content_sha256"}
        if (
            str(row.get("edge_digest") or "") == edge_digest
            and row.get("relation_type") == "exact"
            and row.get("authority_scope") == "source_exact_structure_observation"
            and row.get("not_reaction_validation") is True
            and supplied == _digest(body)
        ):
            return True
    return False


def normalize_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one discovery record without upgrading it to exact evidence."""
    row = dict(value)
    source_kind = _source_kind(row)
    locator_candidates = (
        row.get("source_ref"),
        row.get("doi"),
        row.get("patent_publication"),
        row.get("patent"),
        row.get("pmid"),
        row.get("pmc"),
        row.get("url"),
        (
            str(row.get("local_pdf"))
            if str(row.get("local_pdf") or "").lower().startswith("local_pdf:")
            else f"local_pdf:{row.get('local_pdf')}"
            if row.get("local_pdf")
            else ""
        ),
    )
    source_ref = next(
        (
            canonical
            for candidate in locator_candidates
            if (canonical := canonical_traceable_source_ref(candidate))
        ),
        "",
    )
    registry_id = _bounded_id(row.get("registry_id") or row.get("record_id"))
    artifact_sha256 = str(row.get("artifact_sha256") or "").lower()
    artifact_bound = bool(re.fullmatch(r"[0-9a-f]{64}", artifact_sha256))
    external_locator = bool(source_ref)
    usable = external_locator or (
        source_kind == "curated_registry" and registry_id and artifact_bound
    )
    provenance = str(
        row.get("provenance")
        or row.get("extraction_method")
        or row.get("source_extraction_method")
        or "discovery_metadata"
    ).strip()
    independence_group = (
        "codex_model"
        if source_kind == "codex_claim"
        else independent_source_group(row)
    )
    if not independence_group and source_kind == "curated_registry" and registry_id:
        independence_group = f"registry:{registry_id}"
    if not independence_group and source_ref:
        independence_group = source_document_identity(row)
    patent_family = (
        re.sub(
            r"[^a-z0-9._:-]+",
            "",
            str(row.get("patent_family") or row.get("family_id") or "")
            .strip()
            .lower(),
        )
        if source_kind == "patent"
        else ""
    )
    identity = {
        "source_kind": source_kind,
        "source_ref": source_ref,
        "registry_id": registry_id,
        "artifact_sha256": artifact_sha256 if artifact_bound else "",
        "patent_family": patent_family,
        "independence_group": independence_group,
        "content_scope": str(row.get("content_scope") or row.get("document_type") or ""),
    }
    binding = {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "binding_id": f"source:{_digest(identity)[:24]}",
        **identity,
        "title": " ".join(str(row.get("title") or row.get("source_title") or "").split()),
        "provenance": provenance,
        "discovered_by": str(row.get("discovered_by") or "deterministic_worker"),
        "acquisition_status": str(
            row.get("acquisition_status") or "discovered"
        ),
        "source_pdf_sha256": str(
            row.get("source_pdf_sha256") or row.get("pdf_sha256") or ""
        ).lower(),
        "source_pdf_path": str(row.get("source_pdf_path") or ""),
        "proxy_request_id": str(row.get("proxy_request_id") or ""),
        "visual_candidate_page_count": len(
            [
                value
                for value in row.get("visual_candidate_pages") or []
                if isinstance(value, Mapping)
            ]
        ),
        "exact_row_count_observed": max(0, int(row.get("exact_row_count") or 0)),
        "usable_for_extraction": usable and source_kind != "codex_claim",
        "authority_scope": (
            "model_advisory_claim"
            if source_kind == "codex_claim"
            else "source_discovery_pointer"
        ),
        "semantics": {
            "discovery_is_not_exact_evidence": True,
            "locator_is_not_support": True,
            "independence_is_host_derived": True,
            "acquisition_lifecycle_is_not_proof": True,
        },
    }
    binding["content_sha256"] = _digest(binding)
    return binding


def discover_sources_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    del artifacts
    payload = dict(command.payload)
    raw_sources = [
        dict(value)
        for value in payload.get("sources") or []
        if isinstance(value, Mapping)
    ]
    bindings: list[dict[str, Any]] = []
    scheduled: list[WorkerCommand] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in raw_sources:
        binding = normalize_source_binding(source)
        binding_id = str(binding["binding_id"])
        if binding_id in seen:
            continue
        seen.add(binding_id)
        bindings.append(binding)
        if binding.get("usable_for_extraction") is not True:
            rejected.append(
                {
                    "binding_id": binding_id,
                    "reasons": ["source_not_eligible_for_exact_extraction"],
                }
            )
            continue
        child_identity = _digest(
            {
                "binding_id": binding_id,
                "extraction_artifact_sha256": str(
                    source.get("extraction_artifact_sha256") or ""
                ),
            }
        )
        scheduled.append(
            WorkerCommand(
                command_id=f"extract:{child_identity[:24]}",
                run_id=command.run_id,
                worker_type="extract_exact_source",
                input_revision=command.input_revision,
                idempotency_key=f"extract:{child_identity}",
                payload={
                    "source_binding": binding,
                    "extraction_artifact_sha256": str(
                        source.get("extraction_artifact_sha256") or ""
                    ),
                    "existing_exact_records": list(
                        payload.get("existing_exact_records") or []
                    ),
                    "existing_edge_digests": sorted(
                        {
                            str(value)
                            for value in payload.get("existing_edge_digests") or []
                            if str(value)
                        }
                    ),
                },
                budget=WorkerBudget(
                    task_kind="evidence",
                    timeout_s=min(command.budget.timeout_s, 120.0),
                    max_output_bytes=command.budget.max_output_bytes,
                ),
                dependency_revisions=dict(command.dependency_revisions),
                artifact_refs=command.artifact_refs,
            )
        )
    return {
        "status": "completed" if bindings and not rejected else "partial",
        "payload": {
            "schema_version": "source_discovery_result.v1",
            "source_bindings": bindings,
            "rejected": rejected,
            "extraction_task_count": len(scheduled),
            "no_exact_evidence_claim": True,
        },
        "failure_reasons": ["no_usable_sources_discovered"] if not scheduled else [],
        "scheduled_commands": scheduled,
        "material_events": ["source_bindings_added"] if bindings else [],
    }


def extract_exact_source_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    payload = dict(command.payload)
    supplied_binding = dict(payload.get("source_binding") or {})
    binding = normalize_source_binding(supplied_binding)
    if (
        supplied_binding.get("content_sha256") != binding.get("content_sha256")
        or supplied_binding.get("binding_id") != binding.get("binding_id")
        or binding.get("usable_for_extraction") is not True
    ):
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["source_binding_not_replayable_or_extractable"],
        }

    extraction_sha256 = str(payload.get("extraction_artifact_sha256") or "").lower()
    try:
        extraction = artifacts.read_json(
            extraction_sha256,
            required_authority_scope="structured_exact_row_extraction",
        )
    except WorkerRuntimeError as exc:
        return {
            "status": "partial",
            "payload": {
                "schema_version": "exact_source_extraction_result.v1",
                "source_binding": binding,
                "exact_records": [],
                "procedure_records": [],
                "rejected_rows": [],
                "conflicts": [],
            },
            "failure_reasons": [str(exc)],
        }
    if not isinstance(extraction, Mapping):
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["structured_extraction_artifact_not_object"],
        }
    extraction_row = dict(extraction)
    extractor = dict(extraction_row.get("extractor") or {})
    extraction_reasons: list[str] = []
    if extraction_row.get("schema_version") != STRUCTURED_EXTRACTION_SCHEMA:
        extraction_reasons.append("structured_extraction_schema_invalid")
    if extraction_row.get("source_binding_id") != binding.get("binding_id"):
        extraction_reasons.append("structured_extraction_source_binding_mismatch")
    if str(extractor.get("producer_kind") or "") not in _TRUSTED_EXTRACTION_PRODUCERS:
        extraction_reasons.append("structured_extraction_producer_untrusted")
    if not str(extractor.get("producer_id") or "") or not str(
        extractor.get("version") or ""
    ):
        extraction_reasons.append("structured_extraction_producer_identity_missing")
    if extraction_reasons:
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": sorted(set(extraction_reasons)),
        }

    accepted: list[dict[str, Any]] = []
    accepted_procedures: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(extraction_row.get("rows") or []):
        if not isinstance(raw, Mapping):
            rejected.append({"row_index": index, "reasons": ["exact_row_not_object"]})
            continue
        row = dict(raw)
        audit = audit_retrosynthetic_candidate(
            row.get("product_smiles"),
            _string_list(row.get("reactant_smiles") or row.get("precursor_smiles")),
        )
        reasons = list(audit.get("reasons") or [])
        location_refs = sorted(
            {
                str(value).strip()
                for value in (
                    row.get("location_ref"),
                    row.get("example"),
                    row.get("page"),
                )
                if str(value or "").strip()
            }
        )
        evidence_refs = sorted(
            {
                str(value).strip()
                for value in row.get("evidence_refs") or []
                if str(value or "").strip()
            }
        )
        if not location_refs:
            reasons.append("exact_source_location_missing")
        relation = str(row.get("relation_type") or "exact").lower()
        if relation != "exact":
            reasons.append("source_relation_not_exact")
        if reasons:
            rejected.append({"row_index": index, "reasons": sorted(set(reasons))})
            continue
        conditions = normalize_source_conditions(
            row.get("condition_candidate") or row.get("conditions") or {}
        )
        identity = {
            "binding_id": binding["binding_id"],
            "edge_digest": audit["edge_digest"],
            "location_refs": location_refs,
            "evidence_refs": evidence_refs,
            "conditions": conditions,
        }
        exact_record = {
            "schema_version": EXACT_SOURCE_RECORD_SCHEMA,
            "record_id": f"exact:{_digest(identity)[:24]}",
            "claim_scope_id": str(
                row.get("claim_scope_id")
                or "observation:"
                + _digest(
                    {
                        "binding_id": binding["binding_id"],
                        "step_id": str(row.get("step_id") or ""),
                        "location_refs": location_refs,
                    }
                )[:24]
            ),
            "edge_digest": audit["edge_digest"],
            "edge_identity": audit["edge_identity"],
            "product_smiles": audit["product_smiles"],
            "reactant_smiles": audit["precursor_smiles_multiset"],
            "source_binding_id": binding["binding_id"],
            "source_ref": binding["source_ref"],
            "route_family_id": str(row.get("route_family_id") or ""),
            "independence_group": binding["independence_group"],
            "location_refs": location_refs,
            "evidence_refs": evidence_refs,
            "conditions": conditions,
            "condition_completeness": audit_condition_completeness(conditions),
            "procedure_authority_scope": "",
            "procedure_record_ids": [],
            "relation_type": "exact",
            "provenance": binding["provenance"],
            "extraction_artifact_sha256": extraction_sha256,
            "extraction_artifact_authority_scope": (
                "structured_exact_row_extraction"
            ),
            "extractor": extractor,
            "proof_state": proof_state(
                structural_materialized=True,
                exact_source_bound=True,
                independent_source_groups=[binding["independence_group"]],
            ),
            "authority_scope": "source_exact_structure_observation",
            "not_reaction_validation": True,
            "semantics": {
                "conditions_are_compatibility_projection_only": True,
                "procedure_authority_requires_separate_hash_bound_record": True,
            },
        }
        procedure_record = build_source_procedure_record(
            exact_record=exact_record,
            extraction_row=row,
            source_binding=binding,
            extraction_artifact_sha256=extraction_sha256,
        )
        if procedure_record:
            exact_record["procedure_record_ids"] = [
                procedure_record["procedure_record_id"]
            ]
            accepted_procedures.append(procedure_record)
        exact_record["content_sha256"] = _digest(exact_record)
        accepted.append(exact_record)

    all_records = [
        dict(value)
        for value in payload.get("existing_exact_records") or []
        if isinstance(value, Mapping)
    ] + accepted
    conflicts = detect_source_conflicts(all_records)
    groups_by_edge: dict[str, set[str]] = defaultdict(set)
    for row in all_records:
        if row.get("edge_digest") and row.get("independence_group"):
            groups_by_edge[str(row["edge_digest"])].add(str(row["independence_group"]))
    for row in accepted:
        row["proof_state"] = proof_state(
            structural_materialized=True,
            exact_source_bound=True,
            independent_source_groups=groups_by_edge.get(str(row["edge_digest"]), set()),
        )
        row["content_sha256"] = _digest(
            {key: value for key, value in row.items() if key != "content_sha256"}
        )

    if accepted and not rejected:
        status = "completed"
    elif accepted or rejected:
        status = "partial"
    else:
        status = "partial"
    material_events = []
    if accepted:
        material_events.extend(["exact_rows_added", "material_evidence_added"])
    if accepted_procedures:
        material_events.append("source_procedure_records_added")
    if conflicts:
        material_events.append("source_conflict_added")
    existing_edge_digests = {
        str(value)
        for value in payload.get("existing_edge_digests") or []
        if str(value)
    }
    scheduled_materializations = materialization_commands_for_proposals(
        (
            {
                "product_smiles": row["product_smiles"],
                "precursor_smiles": row["reactant_smiles"],
                "origin_kind": "literature",
                "origin_ref": row["source_ref"],
                "proposal_id": row["record_id"],
                "route_family_id": str(row.get("route_family_id") or ""),
            }
            for row in accepted
            if str(row.get("edge_digest") or "") not in existing_edge_digests
        ),
        run_id=command.run_id,
        input_revision=command.input_revision,
        dependency_revisions=command.dependency_revisions,
    )
    return {
        "status": status,
        "payload": {
            "schema_version": "exact_source_extraction_result.v1",
            "source_binding": binding,
            "exact_records": accepted,
            "procedure_records": accepted_procedures,
            "rejected_rows": rejected,
            "conflicts": conflicts,
        },
        "failure_reasons": ["exact_source_rows_missing_or_rejected"] if not accepted else [],
        "material_events": material_events,
        "scheduled_commands": scheduled_materializations,
    }


def detect_source_conflicts_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    del artifacts
    records = [
        dict(value)
        for value in command.payload.get("exact_records") or []
        if isinstance(value, Mapping)
    ]
    conflicts = detect_source_conflicts(records)
    return {
        "status": "partial" if conflicts else "completed",
        "payload": {
            "schema_version": "source_conflict_detection_result.v1",
            "conflicts": conflicts,
            "record_count": len(records),
        },
        "failure_reasons": ["unresolved_source_conflicts"] if conflicts else [],
        "material_events": ["source_conflict_added"] if conflicts else [],
    }


def detect_source_conflicts(
    exact_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve incompatible exact claims; never silently select a winner."""
    rows = [dict(value) for value in exact_records if isinstance(value, Mapping)]
    conflicts: list[dict[str, Any]] = []
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_edge_claim: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        claim_scope = str(row.get("claim_scope_id") or "")
        edge_digest = str(row.get("edge_digest") or "")
        by_claim[claim_scope].append(row)
        by_edge_claim[(claim_scope, edge_digest)].append(row)

    for claim_scope, candidates in sorted(by_claim.items()):
        edge_digests = sorted({str(row.get("edge_digest") or "") for row in candidates})
        if claim_scope and len(edge_digests) > 1:
            conflicts.append(
                _conflict(
                    conflict_kind="incompatible_exact_structures",
                    subject_id=claim_scope,
                    records=candidates,
                    variants=edge_digests,
                )
            )
    for (claim_scope, edge_digest), candidates in sorted(by_edge_claim.items()):
        # Different publications frequently report valid alternative
        # conditions for the same transformation.  They conflict only when
        # they are assertions about the same explicitly scoped observation.
        if not claim_scope or not edge_digest or len(candidates) < 2:
            continue
        for key in _CONDITION_KEYS:
            variants = sorted(
                {
                    _condition_value(dict(row.get("conditions") or {}).get(key))
                    for row in candidates
                    if dict(row.get("conditions") or {}).get(key) not in (None, "", [])
                }
            )
            if len(variants) > 1:
                conflicts.append(
                    _conflict(
                        conflict_kind=f"incompatible_condition:{key}",
                        subject_id=f"{claim_scope}:{edge_digest}",
                        records=candidates,
                        variants=variants,
                    )
                )
    unique = {str(row["conflict_id"]): row for row in conflicts}
    return [unique[key] for key in sorted(unique)]


def audit_deep_leaf_stock_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    """Audit every selected leaf against one versioned immutable snapshot set."""
    payload = dict(command.payload)
    inventory_sha256 = str(payload.get("inventory_artifact_sha256") or "").lower()
    try:
        raw_inventory = artifacts.read_json(
            inventory_sha256,
            required_authority_scope="inventory_snapshot_set",
        )
    except WorkerRuntimeError as exc:
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": [str(exc)],
        }
    if not isinstance(raw_inventory, Mapping):
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["inventory_artifact_not_object"],
        }
    inventory = dict(raw_inventory)
    if inventory.get("schema_version") != VERSIONED_INVENTORY_ARTIFACT_SCHEMA:
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["inventory_artifact_schema_invalid"],
        }
    offers_raw = [
        dict(value)
        for value in inventory.get("offers") or []
        if isinstance(value, Mapping)
    ]
    reasons: list[str] = []
    adapter_version = str(inventory.get("adapter_version") or "")
    inventory_version = str(inventory.get("inventory_version") or "")
    if not adapter_version:
        reasons.append("inventory_adapter_version_missing")
    if not inventory_version:
        reasons.append("inventory_version_missing")
    try:
        retrieved_at = _timestamp(inventory.get("retrieved_at"))
        as_of = _timestamp(payload.get("as_of"))
    except ValueError as exc:
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": [str(exc)],
        }
    max_age_days = _finite_nonnegative(payload.get("max_age_days"), default=30.0)
    if as_of < retrieved_at:
        reasons.append("inventory_snapshot_from_future")
    age_days = (as_of - retrieved_at).total_seconds() / 86_400.0
    if age_days > max_age_days:
        reasons.append("inventory_snapshot_stale")

    canonical_offers: list[dict[str, Any]] = []
    for index, raw in enumerate(offers_raw):
        try:
            canonical = canonicalize_stock_snapshot(raw)
        except (TypeError, ValueError) as exc:
            reasons.append(f"inventory_offer_invalid:{index}:{type(exc).__name__}")
            continue
        checked_at = _timestamp(canonical["checked_at"])
        if as_of < checked_at:
            reasons.append(f"inventory_offer_from_future:{index}")
        elif (as_of - checked_at).total_seconds() / 86_400.0 > max_age_days:
            reasons.append(f"inventory_offer_stale:{index}")
        canonical_offers.append(canonical)

    snapshot_identity = {
        "schema_version": INVENTORY_SNAPSHOT_SET_SCHEMA,
        "adapter_version": adapter_version,
        "inventory_version": inventory_version,
        "retrieved_at": _iso(retrieved_at),
        "offer_sha256": sorted(stock_snapshot_sha256(row) for row in canonical_offers),
        "source_artifact_sha256": inventory_sha256,
    }
    snapshot_set = {
        **snapshot_identity,
        "snapshot_set_id": f"inventory:{_digest(snapshot_identity)[:24]}",
        "offer_count": len(canonical_offers),
        "immutable_supplier_snapshots": True,
    }
    snapshot_set["content_sha256"] = _digest(snapshot_set)

    provider = SnapshotStockProvider(trusted_snapshots=canonical_offers)
    stale = any(
        value == "inventory_snapshot_stale"
        or value == "inventory_snapshot_from_future"
        or value.startswith("inventory_offer_stale:")
        or value.startswith("inventory_offer_from_future:")
        for value in reasons
    )
    inventory_authority_invalid = bool(reasons)
    audits: list[dict[str, Any]] = []
    selected = list(payload.get("selected_deep_leaves") or [])
    for index, raw_leaf in enumerate(selected):
        leaf = dict(raw_leaf) if isinstance(raw_leaf, Mapping) else {"smiles": raw_leaf}
        leaf_id = str(leaf.get("leaf_id") or f"leaf:{index}")
        canonical = _canonical_smiles(leaf.get("smiles") or leaf.get("canonical_smiles"))
        matching = [
            {**offer, "snapshot_sha256": stock_snapshot_sha256(offer)}
            for offer in canonical_offers
            if canonical and offer.get("canonical_smiles") == canonical
        ]
        provider_result = provider.invoke(
            {
                "schema_version": "stock_lookup_request.v1",
                "smiles": canonical,
                "offers": matching,
            },
            context=ProviderContext(
                run_id=command.run_id,
                case_id=command.run_id,
                target_smiles=str(payload.get("target_smiles") or ""),
                artifact_revision_id=str(command.input_revision),
            ),
        ).to_dict()
        leaf_reasons = list(provider_result.get("reasons") or [])
        if not canonical:
            leaf_reasons.append("deep_leaf_smiles_invalid")
        if stale:
            leaf_reasons.append("inventory_authority_stale")
        if inventory_authority_invalid and not stale:
            leaf_reasons.append("inventory_authority_invalid")
        accepted = (
            provider_result.get("accepted") is True
            and not inventory_authority_invalid
        )
        audit = {
            "schema_version": DEEP_LEAF_AUDIT_SCHEMA,
            "leaf_id": leaf_id,
            "canonical_smiles": canonical,
            "accepted": accepted,
            "authority_valid": not inventory_authority_invalid,
            "inventory_snapshot_set_id": snapshot_set["snapshot_set_id"],
            "inventory_artifact_authority_scope": "inventory_snapshot_set",
            "audited_as_of": _iso(as_of),
            "inventory_retrieved_at": _iso(retrieved_at),
            "provider_result": provider_result,
            "reasons": sorted(set(leaf_reasons)),
            "semantics": {
                "commercial_observation_required": True,
                "commonness_never_implies_availability": True,
                "every_selected_leaf_has_a_record": True,
            },
        }
        audit["content_sha256"] = _digest(audit)
        audits.append(audit)

    if not selected:
        reasons.append("selected_deep_leaves_missing")
    all_closed = bool(audits) and all(row["accepted"] for row in audits)
    unresolved_reasons = sorted(
        {
            *reasons,
            *(
                f"deep_leaf_not_stock_closed:{row['leaf_id']}"
                for row in audits
                if row["accepted"] is not True
            ),
        }
    )
    return {
        "status": "completed" if all_closed and not unresolved_reasons else "partial",
        "payload": {
            "schema_version": "deep_leaf_stock_audit_result.v1",
            "inventory_snapshot_set": snapshot_set,
            "leaf_audits": audits,
            "selected_leaf_count": len(selected),
            "audited_leaf_count": len(audits),
            "stock_closed_leaf_count": sum(row["accepted"] for row in audits),
            "all_selected_leaves_stock_closed": all_closed,
            "inventory_reasons": sorted(set(reasons)),
        },
        "failure_reasons": unresolved_reasons if not all_closed else [],
        "material_events": (
            ["stock_records_added", "stock_boundary_changed"] if audits else []
        ),
    }


def audit_benchmark_leaf_stock_worker(
    command: WorkerCommand,
    artifacts: WorkerArtifactReader,
) -> dict[str, Any]:
    """Audit leaves against a frozen generic benchmark vendor catalog.

    This boundary establishes benchmark-search membership only.  It must never
    be interpreted as a real-time supplier inventory or procurement claim.
    """

    payload = dict(command.payload)
    catalog_sha256 = str(payload.get("catalog_artifact_sha256") or "").lower()
    try:
        raw_catalog = artifacts.read_json(
            catalog_sha256,
            required_authority_scope="benchmark_stock_catalog",
        )
    except WorkerRuntimeError as exc:
        return {"status": "rejected", "payload": {}, "failure_reasons": [str(exc)]}
    if not isinstance(raw_catalog, Mapping):
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["benchmark_catalog_artifact_not_object"],
        }
    catalog = dict(raw_catalog)
    if catalog.get("schema_version") != VERSIONED_BENCHMARK_CATALOG_SCHEMA:
        return {
            "status": "rejected",
            "payload": {},
            "failure_reasons": ["benchmark_catalog_artifact_schema_invalid"],
        }
    reasons: list[str] = []
    try:
        retrieved_at = _timestamp(catalog.get("retrieved_at"))
        as_of = _timestamp(payload.get("as_of"))
    except ValueError as exc:
        return {"status": "rejected", "payload": {}, "failure_reasons": [str(exc)]}
    max_age_days = _finite_nonnegative(payload.get("max_age_days"), default=30.0)
    immutable_catalog = (
        dict(catalog.get("source") or {}).get("immutable_content_addressed") is True
    )
    if as_of < retrieved_at:
        reasons.append("benchmark_catalog_from_future")
    elif (
        not immutable_catalog
        and (as_of - retrieved_at).total_seconds() / 86_400.0 > max_age_days
    ):
        reasons.append("benchmark_catalog_stale")
    adapter_version = str(catalog.get("adapter_version") or "")
    catalog_version = str(catalog.get("catalog_version") or "")
    if not adapter_version:
        reasons.append("benchmark_catalog_adapter_version_missing")
    if not catalog_version:
        reasons.append("benchmark_catalog_version_missing")
    members: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(catalog.get("members") or []):
        if not isinstance(raw, Mapping):
            reasons.append(f"benchmark_catalog_member_invalid:{index}")
            continue
        row = dict(raw)
        canonical = _canonical_smiles(row.get("canonical_smiles"))
        response_sha256 = str(row.get("response_sha256") or "").lower()
        membership_proof_sha256 = str(
            row.get("membership_proof_sha256") or ""
        ).lower()
        vendor_record_valid = (
            int(row.get("vendor_count") or 0) > 0
            and bool(re.fullmatch(r"[0-9a-f]{64}", response_sha256))
        )
        frozen_membership_valid = (
            row.get("membership_verified") is True
            and bool(re.fullmatch(r"[0-9a-f]{64}", membership_proof_sha256))
        )
        if (
            not canonical
            or not (vendor_record_valid or frozen_membership_valid)
        ):
            reasons.append(f"benchmark_catalog_member_invalid:{index}")
            continue
        members[canonical] = {**row, "canonical_smiles": canonical}

    catalog_identity = {
        "schema_version": "benchmark_catalog_snapshot_set.v1",
        "adapter_version": adapter_version,
        "catalog_version": catalog_version,
        "retrieved_at": _iso(retrieved_at),
        "source_artifact_sha256": catalog_sha256,
        "member_count": len(members),
    }
    catalog_identity["content_sha256"] = _digest(catalog_identity)
    snapshot_set_id = f"benchmark-catalog:{catalog_identity['content_sha256'][:24]}"
    invalid_authority = bool(reasons)
    audits: list[dict[str, Any]] = []
    selected = list(payload.get("selected_deep_leaves") or [])
    for index, raw_leaf in enumerate(selected):
        leaf = dict(raw_leaf) if isinstance(raw_leaf, Mapping) else {"smiles": raw_leaf}
        leaf_id = str(leaf.get("leaf_id") or f"leaf:{index}")
        canonical = _canonical_smiles(leaf.get("smiles") or leaf.get("canonical_smiles"))
        member = members.get(canonical, {})
        accepted = bool(canonical and member and not invalid_authority)
        boundary = StockBoundary(
            canonical_smiles=canonical,
            boundary_type="benchmark_stock" if canonical else "unavailable",
            accepted=accepted,
            catalog_bindings=(
                {
                    "catalog_name": str(catalog.get("catalog_name") or "pubchem-vendors"),
                    "catalog_sha256": catalog_sha256,
                    "catalog_version": catalog_version,
                    "canonical_smiles": canonical,
                    "cid": int(member.get("cid") or 0),
                    "vendor_count": int(member.get("vendor_count") or 0),
                    "response_sha256": str(member.get("response_sha256") or ""),
                    "membership_verified": (
                        member.get("membership_verified") is True
                    ),
                    "membership_proof_sha256": str(
                        member.get("membership_proof_sha256") or ""
                    ),
                    "artifact_hash_verified": True,
                    "commercial_orderability_claimed": False,
                },
            )
            if accepted
            else (),
            reasons=() if accepted else ("molecule_not_in_frozen_benchmark_catalog",),
        )
        descriptor = BenchmarkCatalogStockProvider.descriptor
        provider_result = ProviderResultEnvelope(
            provider_id=descriptor.provider_id,
            provider_version=descriptor.version,
            provider_kind=descriptor.kind,
            correlation_group=descriptor.correlation_group,
            output_schema=StockBoundary.schema_version,
            accepted=accepted,
            payload=boundary.to_dict(),
            reasons=boundary.reasons,
            source_refs=(
                str(member.get("source_url") or member.get("catalog_uri") or ""),
            )
            if member
            else (),
        ).to_dict()
        audit = {
            "schema_version": DEEP_LEAF_AUDIT_SCHEMA,
            "leaf_id": leaf_id,
            "canonical_smiles": canonical,
            "accepted": accepted,
            "authority_valid": not invalid_authority,
            "inventory_snapshot_set_id": snapshot_set_id,
            "inventory_artifact_authority_scope": "benchmark_stock_catalog",
            "audited_as_of": _iso(as_of),
            "inventory_retrieved_at": _iso(retrieved_at),
            "provider_result": provider_result,
            "reasons": [] if accepted else list(boundary.reasons),
            "semantics": {
                "benchmark_membership_only": True,
                "immutable_content_addressed_catalog": immutable_catalog,
                "commercial_orderability_claimed": False,
                "every_selected_leaf_has_a_record": True,
            },
        }
        audit["content_sha256"] = _digest(audit)
        audits.append(audit)

    if not selected:
        reasons.append("selected_deep_leaves_missing")
    all_closed = bool(audits) and all(row["accepted"] for row in audits)
    failures = sorted(
        {
            *reasons,
            *(
                f"deep_leaf_not_benchmark_closed:{row['leaf_id']}"
                for row in audits
                if row["accepted"] is not True
            ),
        }
    )
    return {
        "status": "completed" if all_closed and not failures else "partial",
        "payload": {
            "schema_version": "benchmark_leaf_stock_audit_result.v1",
            "inventory_snapshot_set": {**catalog_identity, "snapshot_set_id": snapshot_set_id},
            "leaf_audits": audits,
            "selected_leaf_count": len(selected),
            "audited_leaf_count": len(audits),
            "stock_closed_leaf_count": sum(row["accepted"] for row in audits),
            "all_selected_leaves_stock_closed": all_closed,
            "inventory_reasons": sorted(set(reasons)),
        },
        "failure_reasons": failures if not all_closed else [],
        "material_events": (
            ["stock_records_added", "stock_boundary_changed"] if audits else []
        ),
    }


def _source_kind(row: Mapping[str, Any]) -> str:
    raw = str(row.get("source_kind") or row.get("source_type") or "").lower()
    normalized = raw.replace("-", "_").replace("/", "_").replace(" ", "_")
    aliases = {
        "paper": "paper_si",
        "si": "paper_si",
        "supporting_information": "paper_si",
        "literature": "paper_si",
        "registry": "curated_registry",
        "curated": "curated_registry",
        "image": "image_extraction",
        "ocr": "image_extraction",
        "codex": "codex_claim",
        "model": "codex_claim",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in _SOURCE_KINDS:
        return normalized
    if row.get("patent") or row.get("patent_publication"):
        return "patent"
    if row.get("doi") or row.get("pmid") or row.get("pmc"):
        return "paper_si"
    if row.get("registry_id"):
        return "curated_registry"
    if row.get("local_pdf"):
        return "image_extraction"
    return "codex_claim"


def _conflict(
    *,
    conflict_kind: str,
    subject_id: str,
    records: Iterable[Mapping[str, Any]],
    variants: Iterable[str],
) -> dict[str, Any]:
    record_ids = sorted({str(row.get("record_id") or "") for row in records})
    identity = {
        "conflict_kind": conflict_kind,
        "subject_id": subject_id,
        "record_ids": record_ids,
        "variants": sorted(set(variants)),
    }
    row = {
        "schema_version": SOURCE_CONFLICT_SCHEMA,
        "conflict_id": f"conflict:{_digest(identity)[:24]}",
        **identity,
        "status": "unresolved",
        "semantics": {
            "no_automatic_winner": True,
            "conflict_is_route_deficit": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def _condition_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        return []


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _bounded_id(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9._:-]+", "-", str(value or "").strip())[:160]


def _timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("inventory_timestamp_missing")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError("inventory_timestamp_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("inventory_timestamp_timezone_missing")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_nonnegative(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0 else default


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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
