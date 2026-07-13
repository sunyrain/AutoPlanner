"""Import trusted structured exact-source rows into an existing V4 run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from cascade_planner.application.reaction_mapping import ReactionMapper
from cascade_planner.application.retrosynthesis_workers import (
    STRUCTURED_EXTRACTION_SCHEMA,
    normalize_source_binding,
)
from cascade_planner.application.worker_runtime import WorkerBudget, WorkerCommand
from cascade_planner.interfaces.campaign_gateway import CampaignGatewayError
from cascade_planner.interfaces.target_solver_stages import validate_materialized_edges


if TYPE_CHECKING:
    from cascade_planner.interfaces.campaign_gateway import CampaignGateway


STRUCTURED_EVIDENCE_IMPORT_SCHEMA = "structured_evidence_import.v1"


def import_structured_evidence(
    gateway: "CampaignGateway",
    *,
    run_id: str,
    import_path: str | Path,
    run_dir: str | Path | None = None,
    atom_mapper: ReactionMapper | None = None,
) -> dict[str, Any]:
    """Import typed rows, revalidate affected edges, and rerun closeout."""

    path = Path(import_path).expanduser().resolve()
    document = _read_import(path)
    service = gateway._open(run_id, run_dir=run_dir)
    result = ingest_structured_evidence_document(
        service,
        document=document,
        atom_mapper=atom_mapper,
    )
    return {
        **result,
        "input_path": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def ingest_structured_evidence_document(
    service: Any,
    *,
    document: Mapping[str, Any],
    atom_mapper: ReactionMapper | None = None,
) -> dict[str, Any]:
    """Ingest a validated in-memory connector response through normal workers."""

    document = validate_structured_evidence_document(document)
    if service.kernel.state.terminal:
        raise CampaignGatewayError("terminal_run_cannot_import_evidence")
    graph = service.graph_store.load()
    existing_extractions = {
        str(row.get("extraction_artifact_sha256") or "")
        for row in graph.get("exact_records", {}).values()
        if isinstance(row, Mapping)
    }
    commands: list[WorkerCommand] = []
    imported_refs: list[dict[str, Any]] = []
    for index, raw_source in enumerate(document["sources"], start=1):
        binding_input = dict(raw_source["binding"])
        binding = normalize_source_binding(binding_input)
        if binding.get("usable_for_extraction") is not True:
            raise ValueError(f"evidence_source_not_extractable:{index}")
        extraction = dict(raw_source["extraction"])
        supplied_binding_id = str(extraction.get("source_binding_id") or "")
        if supplied_binding_id and supplied_binding_id != binding["binding_id"]:
            raise ValueError(f"evidence_source_binding_mismatch:{index}")
        extraction["source_binding_id"] = binding["binding_id"]
        ref = service.kernel.artifacts.put_json(
            extraction,
            logical_name=f"structured-evidence-{index}.json",
            producer="autoplanner.structured_evidence_import",
        ).to_dict()
        imported_refs.append(ref)
        if ref["sha256"] in existing_extractions:
            continue
        service.register_artifact_authorities(
            {ref["sha256"]: "structured_exact_row_extraction"}
        )
        revision = service.kernel.revision
        identity = _digest(
            {
                "binding_id": binding["binding_id"],
                "extraction_sha256": ref["sha256"],
            }
        )
        commands.append(
            WorkerCommand(
                command_id=f"import-evidence:{identity[:24]}",
                run_id=service.kernel.spec.run_id,
                worker_type="discover_sources",
                input_revision=revision.graph_revision,
                idempotency_key=f"import-evidence:{identity}",
                payload={
                    "sources": [
                        {
                            **binding_input,
                            "extraction_artifact_sha256": ref["sha256"],
                        }
                    ],
                    "existing_exact_records": list(graph["exact_records"].values()),
                    "existing_edge_digests": [
                        str(edge.get("edge_digest") or "")
                        for edge in graph["edges"].values()
                    ],
                },
                budget=WorkerBudget(task_kind="evidence", timeout_s=180.0),
                dependency_revisions={
                    "graph_revision": revision.graph_revision,
                    "evidence_revision": revision.evidence_revision,
                },
                artifact_refs=(ref,),
            )
        )
    execution: Mapping[str, Any] = {
        "executed_command_count": 0,
        "material_events": [],
    }
    if commands:
        execution = service.execute_commands(
            commands,
            idempotency_key=f"import-evidence:{_digest(document)}",
        )
    validation = validate_materialized_edges(service, atom_mapper=atom_mapper)
    closeout = service.closeout(
        idempotency_key=f"import-evidence:closeout:{service.kernel.state.graph_revision}"
    )
    workbench = service.publish_workbench()
    final_graph = service.graph_store.load()
    return {
        "schema_version": "structured_evidence_import_result.v1",
        "run_id": service.kernel.spec.run_id,
        "run_dir": str(service.kernel.run_dir),
        "document_sha256": _digest(document),
        "source_count": len(document["sources"]),
        "new_source_command_count": len(commands),
        "imported_artifact_refs": imported_refs,
        "execution": execution,
        "validation": validation,
        "exact_record_count": len(final_graph["exact_records"]),
        "source_binding_count": len(final_graph["source_bindings"]),
        "portfolio": closeout["portfolio"],
        "workbench_ref": workbench["snapshot_ref"],
        "model_invocations": 0,
        "semantics": {
            "source_pointer_is_not_exact_evidence": True,
            "only_trusted_structured_rows_are_imported": True,
            "import_does_not_generate_chemistry": True,
        },
    }


def _read_import(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("structured_evidence_import_unreadable") from exc
    return validate_structured_evidence_document(value)


def validate_structured_evidence_document(value: Any) -> dict[str, Any]:
    """Validate the bounded interchange schema without granting authority."""

    if not isinstance(value, Mapping):
        raise ValueError("structured_evidence_import_not_object")
    row = dict(value)
    if row.get("schema_version") != STRUCTURED_EVIDENCE_IMPORT_SCHEMA:
        raise ValueError("structured_evidence_import_schema_invalid")
    sources = [dict(item) for item in row.get("sources") or [] if isinstance(item, Mapping)]
    if not sources or len(sources) > 128:
        raise ValueError("structured_evidence_import_source_count_invalid")
    for index, source in enumerate(sources, start=1):
        if not isinstance(source.get("binding"), Mapping):
            raise ValueError(f"structured_evidence_binding_missing:{index}")
        extraction = source.get("extraction")
        if not isinstance(extraction, Mapping):
            raise ValueError(f"structured_evidence_extraction_missing:{index}")
        extraction = dict(extraction)
        if extraction.get("schema_version") != STRUCTURED_EXTRACTION_SCHEMA:
            raise ValueError(f"structured_evidence_extraction_schema_invalid:{index}")
        if not isinstance(extraction.get("extractor"), Mapping):
            raise ValueError(f"structured_evidence_extractor_missing:{index}")
        rows = extraction.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"structured_evidence_rows_missing:{index}")
    row["sources"] = sources
    return row


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


__all__ = [
    "STRUCTURED_EVIDENCE_IMPORT_SCHEMA",
    "ingest_structured_evidence_document",
    "import_structured_evidence",
    "validate_structured_evidence_document",
]
