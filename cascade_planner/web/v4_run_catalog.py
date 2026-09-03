"""Federated, paginated Web projection over explicitly registered run indexes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.run_index import RunIndex
from cascade_planner.runtime.run_registry_catalog import (
    RunRegistryBinding,
    RunRegistryCatalog,
    binding_from_paths,
    registry_catalog_path,
)
from cascade_planner.web.v4_target_runtime import (
    historical_job,
    job_projection,
    live_job_progress,
    registry_job,
    run_may_be_live,
)


JOB_CATALOG_PAGE_SCHEMA = "autoplanner_web_job_catalog_page.v1"
MAIN_REGISTRY_ID = "main"


def list_catalog_jobs(
    primary_gateway: Any,
    *,
    active_rows: Sequence[Mapping[str, Any]] = (),
    limit: int = 30,
    offset: int = 0,
    project_id: str = "",
    registry_id: str = "",
) -> dict[str, Any]:
    """Merge live Web rows and explicit registries before one global page cut."""

    resolved_limit = max(1, min(200, int(limit)))
    resolved_offset = max(0, int(offset))
    catalog, bindings = _catalog_bindings(primary_gateway)
    selected_project = str(project_id or "").strip().casefold()
    selected_registry = str(registry_id or "").strip().casefold()
    records: dict[tuple[str, str], dict[str, Any]] = {}
    registry_summaries: list[dict[str, Any]] = []
    binding_by_id = {binding.registry_id: binding for binding in bindings}

    for binding in bindings:
        if selected_project and binding.project_id != selected_project:
            continue
        if selected_registry and binding.registry_id != selected_registry:
            continue
        rows, error = _registry_rows(primary_gateway, binding)
        registry_summaries.append(
            {
                "registry_id": binding.registry_id,
                "registry_label": binding.registry_label,
                "project_id": binding.project_id,
                "project_label": binding.project_label,
                "case_id": binding.case_id,
                "run_count": len(rows),
                "available": not error,
                "error": error,
                "read_only": binding.read_only,
            }
        )
        for run in rows:
            run_id = str(run.get("run_id") or "")
            if run_id:
                records[(binding.registry_id, run_id)] = {
                    "binding": binding,
                    "run": dict(run),
                    "active": None,
                }

    main_binding = binding_by_id.get(MAIN_REGISTRY_ID)
    if (
        main_binding is not None
        and (not selected_project or main_binding.project_id == selected_project)
        and (not selected_registry or MAIN_REGISTRY_ID == selected_registry)
    ):
        for raw in active_rows:
            row = dict(raw)
            run_id = str(row.get("run_id") or "")
            if run_id:
                records[(MAIN_REGISTRY_ID, run_id)] = {
                    "binding": main_binding,
                    "run": row,
                    "active": row,
                }

    ordered = sorted(
        records.values(),
        key=lambda record: (
            str(
                dict(record.get("active") or {}).get("updated_at")
                or dict(record.get("run") or {}).get("updated_at")
                or dict(record.get("active") or {}).get("created_at")
                or ""
            ),
            str(dict(record.get("run") or {}).get("run_id") or ""),
        ),
        reverse=True,
    )
    total_count = len(ordered)
    page_records = ordered[resolved_offset : resolved_offset + resolved_limit]
    jobs = [_project_record(primary_gateway, record) for record in page_records]
    projects = _project_summaries(registry_summaries)
    next_offset = resolved_offset + len(jobs)
    return {
        "schema_version": JOB_CATALOG_PAGE_SCHEMA,
        "jobs": jobs,
        "total_count": total_count,
        "limit": resolved_limit,
        "offset": resolved_offset,
        "returned_count": len(jobs),
        "has_more": next_offset < total_count,
        "next_offset": next_offset if next_offset < total_count else None,
        "projects": projects,
        "registries": registry_summaries,
        "catalog_path": str(catalog.path) if catalog is not None else "",
        "semantics": {
            "catalog_is_explicit_and_not_a_results_scan": True,
            "identity_is_registry_id_plus_run_id": True,
            "run_state_is_read_from_owning_registry": True,
            "catalog_grants_no_scientific_authority": True,
        },
    }


def resolve_catalog_job(
    primary_gateway: Any,
    job_id: str,
    *,
    active_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any] | None, Any | None]:
    """Resolve a composite job identity without collapsing same-named runs."""

    _catalog, bindings = _catalog_bindings(primary_gateway)
    binding_by_id = {binding.registry_id: binding for binding in bindings}
    registry_id, run_id = parse_job_id(job_id, bindings=bindings)
    binding = binding_by_id.get(registry_id)
    if binding is None or not run_id:
        return None, None
    if registry_id == MAIN_REGISTRY_ID:
        for raw in active_rows:
            if str(raw.get("run_id") or "") == run_id:
                return _decorate_job(dict(raw), binding=binding), primary_gateway
    rows, error = _registry_rows(primary_gateway, binding)
    if error:
        return None, None
    run = next(
        (dict(value) for value in rows if str(value.get("run_id") or "") == run_id),
        None,
    )
    if run is None:
        return None, None
    gateway = _gateway_for_binding(primary_gateway, binding)
    projected = (
        registry_job(gateway, run)
        if registry_id == MAIN_REGISTRY_ID or run_may_be_live(run)
        else historical_job(run)
    )
    projected.setdefault("run_dir", str(run.get("run_dir") or ""))
    decorated = _decorate_job(projected, binding=binding)
    # ``_decorate_job`` deliberately strips private Kernel state from catalog
    # list rows.  A resolved detail request is the one place that must retain
    # it until ``live_job_progress`` has built the public projection.
    if isinstance(projected.get("_status_result"), Mapping):
        decorated["_status_result"] = dict(projected["_status_result"])
    return decorated, gateway


def make_job_id(registry_id: str, run_id: str) -> str:
    return f"solve:@{registry_id}:{run_id}"


def parse_job_id(
    job_id: str,
    *,
    bindings: Sequence[RunRegistryBinding] = (),
) -> tuple[str, str]:
    value = str(job_id or "")
    if not value.startswith("solve:@") or ":" not in value[7:]:
        return "", ""
    registry_id, run_id = value[7:].split(":", 1)
    known = {binding.registry_id for binding in bindings}
    return (registry_id, run_id) if registry_id in known and run_id else ("", "")


def catalog_identity(registry_id: str, run_id: str) -> str:
    return f"{registry_id}:{run_id}"


def _catalog_bindings(
    primary_gateway: Any,
) -> tuple[RunRegistryCatalog | None, list[RunRegistryBinding]]:
    paths = getattr(primary_gateway, "paths", None)
    if (
        paths is None
        or getattr(paths, "runtime_root", None) is None
        or getattr(paths, "run_index_path", None) is None
    ):
        root = Path.cwd().resolve()
        return None, [
            RunRegistryBinding(
                registry_id=MAIN_REGISTRY_ID,
                registry_label="Main local registry",
                project_id="local",
                project_label="Local AutoPlanner runs",
                case_id="",
                repository_root=root,
                runtime_root=root / ".unconfigured-web-runtime",
                runs_root=root / ".unconfigured-web-runtime" / "runs",
                artifact_store_root=root / ".unconfigured-web-runtime" / "artifacts",
                run_index_path=root / ".unconfigured-web-runtime" / "run_index.sqlite3",
                external_data_root=root / ".unconfigured-web-runtime" / "external",
                vendor_root=root / "vendor",
                source="gateway_compatibility",
                read_only=False,
                display_order=-1_000,
            )
        ]
    catalog = RunRegistryCatalog(registry_catalog_path(paths))
    primary_binding = binding_from_paths(
        paths,
        registry_id=MAIN_REGISTRY_ID,
        registry_label="Main local registry",
        project_id="local",
        project_label="Local AutoPlanner runs",
        source="web_gateway",
        read_only=False,
        display_order=-1_000,
    )
    registered_main = catalog.get(MAIN_REGISTRY_ID)
    if (
        registered_main is None
        or registered_main.run_index_path != primary_binding.run_index_path
        or not registered_main.enabled
    ):
        catalog.register(primary_binding)
    return catalog, catalog.list_registries()


def _registry_rows(
    primary_gateway: Any,
    binding: RunRegistryBinding,
) -> tuple[list[dict[str, Any]], str]:
    try:
        if binding.registry_id == MAIN_REGISTRY_ID and not binding.run_index_path.is_file():
            value = primary_gateway.list_runs(limit=10_000)
            return [dict(row) for row in value.get("runs") or [] if isinstance(row, Mapping)], ""
        if not binding.run_index_path.is_file():
            return [], "run_index_missing"
        return RunIndex(binding.run_index_path).list_runs(limit=10_000), ""
    except Exception as exc:
        return [], f"{type(exc).__name__}:{exc}"[:300]


def _gateway_for_binding(primary_gateway: Any, binding: RunRegistryBinding) -> Any:
    if binding.registry_id == MAIN_REGISTRY_ID:
        return primary_gateway
    return CampaignGateway(binding.runtime_paths())


def _project_record(primary_gateway: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    binding = record["binding"]
    active = record.get("active")
    if isinstance(active, Mapping):
        raw = dict(active)
        row = job_projection(raw)
        row["progress"] = live_job_progress(lambda: primary_gateway, raw)
        if str(raw.get("status") or "") in {
            "queued",
            "running",
            "cancelling",
            "paused",
        }:
            row["phase"] = str(row["progress"].get("phase") or row.get("phase") or "")
        return _decorate_job(row, binding=binding)
    run = dict(record.get("run") or {})
    if run_may_be_live(run):
        gateway = _gateway_for_binding(primary_gateway, binding)
        raw = registry_job(gateway, run)
        row = job_projection(raw)
        row["progress"] = live_job_progress(lambda: gateway, raw)
        if str(raw.get("status") or "") in {
            "queued",
            "running",
            "cancelling",
            "paused",
        }:
            row["phase"] = str(row["progress"].get("phase") or row.get("phase") or "")
    else:
        row = historical_job(run)
    return _decorate_job(row, binding=binding)


def _decorate_job(
    row: Mapping[str, Any],
    *,
    binding: RunRegistryBinding,
) -> dict[str, Any]:
    value = dict(row)
    run_id = str(value.get("run_id") or "")
    value.update(
        job_id=make_job_id(binding.registry_id, run_id),
        registry_id=binding.registry_id,
        registry_label=binding.registry_label,
        project_id=binding.project_id,
        project_label=binding.project_label,
        case_id=binding.case_id,
        catalog_identity=catalog_identity(binding.registry_id, run_id),
        registry_read_only=binding.read_only,
    )
    value.pop("_status_result", None)
    return value


def _project_summaries(registries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "project_id": "",
            "project_label": "",
            "run_count": 0,
            "registry_count": 0,
            "available_registry_count": 0,
        }
    )
    for registry in registries:
        project_id = str(registry.get("project_id") or "")
        row = values[project_id]
        row["project_id"] = project_id
        row["project_label"] = str(registry.get("project_label") or project_id)
        row["run_count"] += int(registry.get("run_count") or 0)
        row["registry_count"] += 1
        row["available_registry_count"] += int(registry.get("available") is True)
    return sorted(values.values(), key=lambda row: str(row["project_label"]).casefold())


__all__ = [
    "JOB_CATALOG_PAGE_SCHEMA",
    "MAIN_REGISTRY_ID",
    "catalog_identity",
    "list_catalog_jobs",
    "make_job_id",
    "parse_job_id",
    "resolve_catalog_job",
]
