"""Model-free, resumable replay of compact retrosynthesis acceptance packs.

A replay pack carries facts, not a saved mutable run. The runner reconstructs
those facts through the same campaign service, workers, canonical hypergraph,
proof stitcher, and RunKernel used by live V4 campaigns.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import normalize_source_binding
from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.application.worker_runtime import WorkerBudget, WorkerCommand
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.runtime.paths import RuntimePaths

from .replay_contract import (
    REPLAY_PACK_SCHEMA,
    REPLAY_RESULT_SCHEMA,
    REPLAY_STAGES,
    ReplayPackError,
    dataclass_value,
    load_replay_pack,
    with_replay_pack_digest,
)
from .replay_reporting import build_replay_report as _report
from .replay_lifecycle import apply_replay_lifecycle_events


def run_replay_pack(
    value: str | Path | Mapping[str, Any],
    *,
    paths: RuntimePaths | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    stop_after: str = "",
) -> dict[str, Any]:
    """Rebuild one pack without a network, LLM, or visual-model call."""
    pack = load_replay_pack(value)
    if stop_after and stop_after not in REPLAY_STAGES:
        raise ReplayPackError("replay_stop_after_invalid")
    runtime_paths = paths or RuntimePaths.discover()
    runtime_paths.ensure_runtime_directories()
    identity = str(run_id or pack["case_id"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", identity):
        raise ReplayPackError("replay_run_id_invalid")
    directory = (
        Path(run_dir).expanduser().resolve()
        if run_dir is not None
        else runtime_paths.runs_root / _run_segment(identity)
    )
    service = _open_or_create_service(
        pack,
        paths=runtime_paths,
        run_id=identity,
        run_dir=directory,
    )
    if service.kernel.state.status == "paused":
        revision = service.kernel.state.revision
        service.kernel.resume(idempotency_key=f"replay:resume:{revision}")
    if service.kernel.state.status not in {"running", "completed"}:
        raise ReplayPackError(
            f"replay_run_not_executable:{service.kernel.state.status}"
        )

    refs, authorities = _publish_pack_artifacts(service, pack)
    service.register_artifact_authorities(authorities)
    stages: list[dict[str, Any]] = []
    if service.kernel.state.status == "running":
        _run_plan_stage(service, pack, stages)
        if _pause_after(service, stop_after, "plan"):
            return _report(service, pack, stages, interrupted=True)
        _run_materialization_stage(service, stages)
        if _pause_after(service, stop_after, "materialization"):
            return _report(service, pack, stages, interrupted=True)
        _run_evidence_stage(service, pack, refs, stages)
        if _pause_after(service, stop_after, "evidence"):
            return _report(service, pack, stages, interrupted=True)
        _run_validation_stage(service, pack, refs, stages)
        if _pause_after(service, stop_after, "validation"):
            return _report(service, pack, stages, interrupted=True)
        _run_stock_stage(service, pack, refs["inventory"], stages)
        if _pause_after(service, stop_after, "stock"):
            return _report(service, pack, stages, interrupted=True)
        _run_lifecycle_stage(service, pack, stages)
        if _pause_after(service, stop_after, "lifecycle"):
            return _report(service, pack, stages, interrupted=True)
        _run_closeout_stage(service, stages)
    return _report(service, pack, stages, interrupted=False)


def _open_or_create_service(
    pack: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    run_id: str,
    run_dir: Path,
) -> RetrosynthesisCampaignService:
    spec_path = run_dir / ".autoplanner" / "kernel" / "run_spec.json"
    if spec_path.is_file():
        service = RetrosynthesisCampaignService.open(
            paths.runtime_root,
            run_dir,
            artifact_store_root=paths.artifact_store_root,
            run_index_path=paths.run_index_path,
        )
        if service.kernel.spec.run_id != run_id:
            raise ReplayPackError("replay_run_identity_conflict")
        expected_producer = f"autoplanner.replay:{pack['content_sha256']}"
        if service.kernel.spec.producer != expected_producer:
            raise ReplayPackError("replay_run_pack_conflict")
        return service
    acceptance = dataclass_value(
        RetrosynthesisAcceptanceSpec,
        pack["acceptance"],
    )
    budget = dataclass_value(RetrosynthesisRunBudget, pack["budget"])
    return RetrosynthesisCampaignService.create(
        paths.runtime_root,
        run_dir,
        spec=RunSpec(
            run_id=run_id,
            target_name=str(pack["target"]["name"]),
            target_smiles=str(pack["target"]["smiles"]),
            acceptance=acceptance,
            limits=RunLimits(
                model=budget,
                max_total_tasks=max(64, budget.max_attempt_runs),
                max_evidence_tasks=max(8, len(pack["sources"]) * 2),
                max_stock_tasks=8,
                max_validation_tasks=max(16, len(pack["reactions"])),
            ),
            producer=f"autoplanner.replay:{pack['content_sha256']}",
            created_at=str(pack.get("created_at") or "2026-07-13T00:00:00Z"),
        ),
        artifact_store_root=paths.artifact_store_root,
        run_index_path=paths.run_index_path,
    )


def _publish_pack_artifacts(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    refs: dict[str, Any] = {"sources": {}}
    authorities: dict[str, str] = {}
    for source in pack["sources"]:
        binding = normalize_source_binding(source["binding"])
        artifact = {
            "schema_version": "structured_exact_row_extraction.v1",
            "source_binding_id": binding["binding_id"],
            "extractor": dict(source["extractor"]),
            "rows": list(source["rows"]),
        }
        ref = service.kernel.artifacts.put_json(
            artifact,
            logical_name=f"replay-exact-{binding['binding_id']}.json",
            producer="autoplanner.replay_pack",
        ).to_dict()
        refs["sources"][binding["binding_id"]] = ref
        authorities[ref["sha256"]] = "structured_exact_row_extraction"
    inventory_ref = service.kernel.artifacts.put_json(
        pack["inventory"]["artifact"],
        logical_name="replay-inventory.json",
        producer="autoplanner.replay_pack",
    ).to_dict()
    refs["inventory"] = inventory_ref
    authorities[inventory_ref["sha256"]] = "inventory_snapshot_set"
    return refs, authorities


def _run_plan_stage(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    graph = service.graph_store.load()
    if graph["route_families"]:
        stages.append(_stage("plan", "reused", len(graph["route_families"])))
        return
    result = service.apply_global_plan(
        pack["global_plan"],
        idempotency_key="replay:global-plan",
        proposal_origin_kind="literature_replay",
        proposal_origin_ref=f"replay_pack:{str(pack['content_sha256'])[:24]}",
    )
    stages.append(_stage("plan", "executed", int(result["changed"])))


def _run_materialization_stage(
    service: RetrosynthesisCampaignService,
    stages: list[dict[str, Any]],
) -> None:
    commands = service.graph_store.frontier_materialization_commands()
    if not commands:
        stages.append(_stage("materialization", "reused", 0))
        return
    result = service.execute_commands(
        commands,
        idempotency_key="replay:materialization",
    )
    stages.append(
        _stage(
            "materialization",
            "executed",
            result["executed_command_count"],
        )
    )


def _run_evidence_stage(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
    refs: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    graph = service.graph_store.load()
    # The canonical graph assigns its own full source_binding_id and retains
    # the discovery worker identity as external_binding_id. Recovery must use
    # that external identity; looking for the pre-ingestion ``binding_id``
    # re-scheduled already materialized extraction tasks after a pause.
    present = {
        str(identity)
        for row in graph["source_bindings"].values()
        for identity in (
            row.get("external_binding_id"),
            row.get("source_binding_id"),
            row.get("binding_id"),
        )
        if str(identity or "")
    }
    commands: list[WorkerCommand] = []
    existing_records = list(graph["exact_records"].values())
    existing_edges = [
        str(row["edge_digest"]) for row in graph["edges"].values()
    ]
    for index, source in enumerate(pack["sources"], start=1):
        binding = normalize_source_binding(source["binding"])
        if binding["binding_id"] in present:
            continue
        ref = refs["sources"][binding["binding_id"]]
        commands.append(
            _command(
                service,
                "discover_sources",
                {
                    "sources": [
                        {
                            **dict(source["binding"]),
                            "extraction_artifact_sha256": ref["sha256"],
                        }
                    ],
                    "existing_exact_records": existing_records,
                    "existing_edge_digests": existing_edges,
                },
                task_kind="evidence",
                suffix=f"source-{index}",
                artifact_refs=(ref,),
            )
        )
    if not commands:
        stages.append(_stage("evidence", "reused", len(graph["exact_records"])))
        return
    result = service.execute_commands(
        commands,
        idempotency_key="replay:evidence",
    )
    stages.append(
        _stage("evidence", "executed", result["executed_command_count"])
    )


def _run_validation_stage(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
    refs: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    graph = service.graph_store.load()
    commands: list[WorkerCommand] = []
    all_source_refs = tuple(refs["sources"].values())
    for index, reaction in enumerate(pack["reactions"], start=1):
        edge_id = f"edge:{reaction['edge_digest']}"
        edge = dict(graph["edges"].get(edge_id) or {})
        if any(
            dict(proof).get("accepted") is True
            for proof in edge.get("reaction_proofs") or []
        ):
            continue
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
                        "edge_digest": reaction["edge_digest"],
                        "product_smiles": reaction["product_smiles"],
                        "precursor_smiles": reaction["reactant_smiles"],
                    },
                    "mapped_reaction_smiles": reaction[
                        "mapped_reaction_smiles"
                    ],
                    "exact_source_records": exact_records,
                },
                task_kind="validation",
                suffix=f"edge-{index}",
                artifact_refs=all_source_refs,
            )
        )
    if not commands:
        stages.append(_stage("validation", "reused", len(graph["edges"])))
        return
    result = service.execute_commands(
        commands,
        idempotency_key="replay:validation",
    )
    stages.append(
        _stage("validation", "executed", result["executed_command_count"])
    )


def _run_stock_stage(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
    inventory_ref: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    graph = service.graph_store.load()
    leaf_ids = sorted(
        {
            str(molecule_id)
            for route in graph["route_families"].values()
            if route.get("selected") is not False
            for molecule_id in route.get("leaf_molecule_ids") or []
        }
    )
    if leaf_ids and all(
        graph["molecules"][value].get("stock_closed") is True
        for value in leaf_ids
    ):
        stages.append(_stage("stock", "reused", len(leaf_ids)))
        return
    command = _command(
        service,
        "audit_deep_leaf_stock",
        {
            "target_smiles": pack["target"]["smiles"],
            "selected_deep_leaves": [
                {
                    "leaf_id": molecule_id,
                    "smiles": graph["molecules"][molecule_id][
                        "canonical_smiles"
                    ],
                }
                for molecule_id in leaf_ids
            ],
            "inventory_artifact_sha256": inventory_ref["sha256"],
            "as_of": pack["inventory"]["as_of"],
            "max_age_days": pack["inventory"].get("max_age_days", 30),
        },
        task_kind="stock",
        suffix="selected-leaves",
        artifact_refs=(inventory_ref,),
    )
    result = service.execute_commands(
        (command,),
        idempotency_key="replay:stock",
    )
    stages.append(_stage("stock", "executed", result["executed_command_count"]))


def _run_closeout_stage(
    service: RetrosynthesisCampaignService,
    stages: list[dict[str, Any]],
) -> None:
    revision = service.kernel.state.graph_revision
    result = service.closeout(idempotency_key=f"replay:closeout:{revision}")
    service.publish_workbench()
    service.kernel.apply_stop_decision(idempotency_key=f"replay:stop:{revision}")
    stages.append(
        _stage(
            "closeout",
            "executed",
            int(result["acceptance_report"]["accepted"]),
        )
    )


def _run_lifecycle_stage(
    service: RetrosynthesisCampaignService,
    pack: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    try:
        result = apply_replay_lifecycle_events(service, pack)
    except ValueError as exc:
        raise ReplayPackError(str(exc)) from exc
    if result:
        stages.append(
            _stage("lifecycle", str(result["status"]), int(result["work_count"]))
        )


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
        command_id=f"replay:{worker_type}:{suffix}",
        run_id=service.kernel.spec.run_id,
        worker_type=worker_type,
        input_revision=revision.graph_revision,
        idempotency_key=(
            f"replay:{worker_type}:{suffix}:{revision.graph_revision}"
        ),
        payload=dict(payload),
        budget=WorkerBudget(task_kind=task_kind),
        dependency_revisions={
            "graph_revision": revision.graph_revision,
            "evidence_revision": revision.evidence_revision,
        },
        artifact_refs=tuple(dict(value) for value in artifact_refs),
    )


def _pause_after(
    service: RetrosynthesisCampaignService,
    requested: str,
    current: str,
) -> bool:
    if requested != current:
        return False
    revision = service.kernel.state.revision
    service.kernel.pause(
        idempotency_key=f"replay:pause:{current}:{revision}"
    )
    return True


def _stage(name: str, status: str, work_count: int) -> dict[str, Any]:
    return {
        "stage": name,
        "status": status,
        "work_count": int(work_count),
    }


def _run_segment(run_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip(".-")[:64] or "run"
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"{label}--{digest}"


__all__ = [
    "REPLAY_PACK_SCHEMA",
    "REPLAY_RESULT_SCHEMA",
    "REPLAY_STAGES",
    "ReplayPackError",
    "load_replay_pack",
    "run_replay_pack",
    "with_replay_pack_digest",
]
