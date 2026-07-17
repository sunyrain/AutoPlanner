"""Canonical V4 workspace helpers shared by the UI and HTTP adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, jsonify, redirect, request

from cascade_planner.application.program_experience_store import (
    DEFAULT_PROGRAM_EXPERIENCE_LIBRARY_NAME,
    read_program_experience_library,
)
from cascade_planner.application.reaction_template_store import (
    DEFAULT_TEMPLATE_LIBRARY_NAME,
    read_template_library,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.web.workspace_catalog import compile_showcase_catalog


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SHARED_RESULTS_DIR = ROOT / "results" / "shared"
PRESENTATION_MANIFEST = SHARED_RESULTS_DIR / "presentation_showcase_20260715" / "manifest.json"
WORKSPACE_RETURN_MARKUP = """      <a id="dashboardReturn" class="icon-button dashboard-return" href="/v4" target="_top" aria-label="返回统一总控台" style="text-decoration:none">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 6-6 6 6 6M4 12h11a5 5 0 0 0 5-5V5"></path></svg>
        <span class="button-label">总控台</span>
      </a>
"""


def static_html(name: str) -> Response:
    path = STATIC_DIR / name
    return Response(path.read_text(encoding="utf-8"), mimetype="text/html")


def showcase_catalog() -> dict[str, Any]:
    return compile_showcase_catalog(
        root=ROOT,
        shared_root=SHARED_RESULTS_DIR,
        manifest_path=PRESENTATION_MANIFEST,
    )


def compiled_program_benchmark_catalog() -> dict[str, Any]:
    """Expose digest-bound, replayable multi-step Program candidates for review.

    These records are deliberately distinct from self-evolution memory: they show
    a compiler-screened replacement and its chemical fallback, not validated
    reaction evidence or an admitted executable Program.
    """

    benchmark_root = ROOT / "benchmarks"
    observations: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in sorted(benchmark_root.glob("*_candidate_route_observation.v1.json")):
        value, error = _read_digest_bound_json(path)
        if error:
            errors.append(f"{path.name}:{error}")
            continue
        digest = str(value.get("content_sha256") or "")
        if digest:
            observations[digest] = value

    records: list[dict[str, Any]] = []
    for path in sorted(benchmark_root.glob("*_candidate_innovation_screen.v1.json")):
        screen, error = _read_digest_bound_json(path)
        if error:
            errors.append(f"{path.name}:{error}")
            continue
        observation = observations.get(str(screen.get("observation_sha256") or ""), {})
        for route_id, route_screen in sorted(dict(screen.get("route_screens") or {}).items()):
            discovery = dict(dict(route_screen).get("discovery") or {})
            for raw_candidate in discovery.get("candidates") or []:
                candidate = dict(raw_candidate)
                innovation = dict(candidate.get("route_innovation") or {})
                chemical_steps = int(innovation.get("chemical_step_equivalent_count") or 0)
                if innovation.get("kind") != "biocatalytic_superstep" or chemical_steps < 2:
                    continue
                boundary = dict(candidate.get("boundary") or {})
                replaced_edge_ids = [
                    str(value) for value in boundary.get("replaced_edge_ids") or [] if str(value)
                ]
                molecules = dict(observation.get("molecules") or {})
                transformations = dict(observation.get("transformations") or {})
                fallback_steps = [
                    _fallback_step_snapshot(
                        edge_id,
                        transformations=transformations,
                        molecules=molecules,
                    )
                    for edge_id in replaced_edge_ids
                ]
                target = dict(observation.get("target") or {})
                precursor_id = str(boundary.get("precursor_molecule_id") or "")
                product_id = str(boundary.get("product_molecule_id") or "")
                records.append(
                    {
                        "benchmark_id": "program-benchmark:"
                        + str(
                            innovation.get("innovation_id") or candidate.get("candidate_id") or ""
                        ),
                        "source_file": path.name,
                        "target_name": str(target.get("name") or "unnamed target"),
                        "route_id": str(route_id),
                        "candidate_id": str(candidate.get("candidate_id") or ""),
                        "innovation_id": str(innovation.get("innovation_id") or ""),
                        "capability_id": str(candidate.get("capability_id") or ""),
                        "chemical_step_equivalent_count": chemical_steps,
                        "physical_step_count": 1,
                        "net_step_savings": int(
                            innovation.get("step_savings") or chemical_steps - 1
                        ),
                        "review_status": str(candidate.get("review_status") or "screen_required"),
                        "validation_status": str(innovation.get("validation_status") or "proposed"),
                        "authority_scope": str(
                            innovation.get("authority_scope") or "proposal_only"
                        ),
                        "evidence_grade": str(innovation.get("evidence_grade") or "low"),
                        "warning_codes": [
                            str(value)
                            for value in candidate.get("warning_codes") or []
                            if str(value)
                        ],
                        "selectivity_objective": str(innovation.get("selectivity_objective") or ""),
                        "substrate_scope_basis": str(innovation.get("substrate_scope_basis") or ""),
                        "precedent_refs": [
                            str(value)
                            for value in innovation.get("precedent_refs") or []
                            if str(value)
                        ],
                        "enzyme": dict(innovation.get("enzyme") or {}),
                        "cofactor_requirements": dict(
                            innovation.get("cofactor_requirements") or {}
                        ),
                        "cofactor_regenerations": dict(
                            innovation.get("cofactor_regenerations") or {}
                        ),
                        "boundary": {
                            "minimum_boundary_proof_level": int(
                                boundary.get("minimum_boundary_proof_level") or 0
                            ),
                            "precursor": _molecule_snapshot(
                                precursor_id,
                                molecules=molecules,
                                fallback_smiles=str(boundary.get("precursor_smiles") or ""),
                            ),
                            "product": _molecule_snapshot(
                                product_id,
                                molecules=molecules,
                                fallback_smiles=str(boundary.get("product_smiles") or ""),
                            ),
                        },
                        "fallback_steps": fallback_steps,
                        "semantics": {
                            "benchmark_replay_only": True,
                            "must_compile_to_a_program_before_admission": True,
                            "requires_exact_substrate_biocatalysis_validation": True,
                            "chemical_fallback_is_preserved": True,
                            "does_not_grant_route_closure_or_reaction_proof": True,
                        },
                    }
                )
    return {
        "schema_version": "autoplanner.compiled_program_benchmark_catalog.v1",
        "ok": not errors,
        "record_count": len(records),
        "records": sorted(records, key=lambda value: value["benchmark_id"]),
        "errors": errors,
        "semantics": {
            "records_are_digest_bound_benchmark_candidates": True,
            "screening_is_not_program_admission": True,
            "candidate_requires_experiment_before_l2_or_route_acceptance": True,
        },
    }


def _read_digest_bound_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, "unreadable"
    if not isinstance(value, dict):
        return {}, "not_object"
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    if not observed or observed != strict_canonical_json_sha256(material):
        return {}, "content_digest_invalid"
    return value, ""


def _molecule_snapshot(
    molecule_id: str,
    *,
    molecules: dict[str, Any],
    fallback_smiles: str = "",
) -> dict[str, str]:
    molecule = dict(molecules.get(molecule_id) or {})
    smiles = str(molecule.get("canonical_smiles") or fallback_smiles)
    label = str(molecule.get("label") or molecule_id or "unresolved boundary")
    return {"molecule_id": molecule_id, "label": label, "smiles": smiles}


def _fallback_step_snapshot(
    edge_id: str,
    *,
    transformations: dict[str, Any],
    molecules: dict[str, Any],
) -> dict[str, Any]:
    transformation = dict(transformations.get(edge_id) or {})
    precursor_ids = [
        str(value) for value in transformation.get("precursor_molecule_ids") or [] if str(value)
    ]
    product_id = str(transformation.get("product_molecule_id") or "")
    return {
        "edge_id": edge_id,
        "precursors": [_molecule_snapshot(value, molecules=molecules) for value in precursor_ids],
        "product": _molecule_snapshot(product_id, molecules=molecules),
        "source_refs": [
            str(value) for value in transformation.get("source_refs") or [] if str(value)
        ],
    }


def self_evolution_catalog(gateway: Any) -> dict[str, Any]:
    """Project the two digest-bound cross-campaign memory stores for the UI."""

    paths = getattr(gateway, "paths", None)
    external_root = (
        Path(getattr(paths, "external_data_root", ROOT / "data_external")).expanduser().resolve()
    )
    memory_root = external_root / "self-evo"
    template_path = memory_root / DEFAULT_TEMPLATE_LIBRARY_NAME
    experience_path = memory_root / DEFAULT_PROGRAM_EXPERIENCE_LIBRARY_NAME

    template_library, template_error = read_template_library(template_path)
    experience_library, experience_error = read_program_experience_library(experience_path)
    templates = [
        dict(value)
        for value in dict(template_library.get("templates") or {}).values()
        if isinstance(value, dict)
    ]
    experiences = [
        dict(value)
        for value in dict(experience_library.get("experiences") or {}).values()
        if isinstance(value, dict)
    ]

    template_rows = []
    for row in sorted(templates, key=lambda value: str(value.get("template_id") or "")):
        successes = len(row.get("successful_edge_digests") or [])
        failures = len(row.get("failed_edge_digests") or [])
        template_rows.append(
            {
                "template_id": str(row.get("template_id") or ""),
                "status": str(row.get("status") or "active"),
                "maturity": str(row.get("maturity") or "single_source_observed"),
                "example_count": int(row.get("example_count") or 0),
                "independent_source_group_count": len(row.get("independent_source_groups") or []),
                "successful_reuse_count": successes,
                "failed_reuse_count": failures,
                "has_reuse_outcome": bool(successes or failures),
                "replay_validated": row.get("maturity") == "reuse_validated",
            }
        )

    experience_rows = []
    domain_counts: dict[str, int] = {}
    for row in sorted(experiences, key=lambda value: str(value.get("experience_id") or "")):
        domain = str(row.get("domain") or "unknown")
        observation_count = len(dict(row.get("observations") or {}))
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        experience_rows.append(
            {
                "experience_id": str(row.get("experience_id") or ""),
                "domain": domain,
                "disposition": str(row.get("disposition") or "inconclusive"),
                "observation_count": observation_count,
                "counts": dict(row.get("counts") or {}),
                "authority_scope": str(row.get("authority_scope") or "proposal_memory_only"),
            }
        )

    active_templates = [row for row in template_rows if row["status"] != "quarantined"]
    mechanism_rows = [row for row in experience_rows if row["domain"] == "mechanism"]
    summary = {
        "reaction_template_count": len(template_rows),
        "retrievable_reaction_template_count": len(active_templates),
        "attempted_reaction_template_count": sum(row["has_reuse_outcome"] for row in template_rows),
        "replay_validated_reaction_template_count": sum(
            row["replay_validated"] for row in template_rows
        ),
        "successful_reuse_count": sum(row["successful_reuse_count"] for row in template_rows),
        "failed_reuse_count": sum(row["failed_reuse_count"] for row in template_rows),
        "program_experience_count": len(experience_rows),
        "mechanism_experience_count": len(mechanism_rows),
        "mechanism_observation_count": sum(row["observation_count"] for row in mechanism_rows),
    }
    return {
        "schema_version": "autoplanner.self_evolution_catalog.v1",
        "ok": not template_error and not experience_error,
        "summary": summary,
        "reaction_templates": {
            "present": template_path.is_file(),
            "integrity": "valid" if not template_error else "invalid",
            "error": template_error,
            "generation": int(template_library.get("generation") or 0),
            "content_sha256": str(template_library.get("content_sha256") or ""),
            "library_name": template_path.name,
            "records": template_rows,
        },
        "program_experience": {
            "present": experience_path.is_file(),
            "integrity": "valid" if not experience_error else "invalid",
            "error": experience_error,
            "generation": int(experience_library.get("generation") or 0),
            "content_sha256": str(experience_library.get("content_sha256") or ""),
            "library_name": experience_path.name,
            "domain_counts": dict(sorted(domain_counts.items())),
            "records": experience_rows,
        },
        "compiled_program_benchmarks": compiled_program_benchmark_catalog(),
        "semantics": {
            "reaction_templates_are_replay_proposals_not_evidence": True,
            "replay_validated_means_a_later_host_edge_succeeded": True,
            "program_experience_requires_replay_validated_experimental_claims": True,
            "negative_inconclusive_and_conflicting_memory_is_retained": True,
            "memory_cannot_grant_route_or_reaction_acceptance": True,
        },
    }


def workspace_payload(gateway: Any) -> dict[str, Any]:
    try:
        listed = gateway.list_runs(limit=40)
        runs = [dict(row) for row in listed.get("runs") or [] if isinstance(row, dict)]
        backend = {
            "available": True,
            "state": "ready",
            "run_count": int(listed.get("run_count") or len(runs)),
        }
        error = ""
    except Exception as exc:  # UI projection must not hide the static showcase.
        runs = []
        backend = {"available": False, "state": "unavailable", "run_count": 0}
        error = f"{type(exc).__name__}:{exc}"
    for row in runs:
        run_id = str(row.get("run_id") or "")
        row["workbench_url"] = f"/api/v4/runs/{run_id}/workbench.html" if run_id else ""
        row["status_url"] = f"/api/v4/runs/{run_id}/status" if run_id else ""
    catalog = showcase_catalog()
    return {
        "schema_version": "autoplanner.workspace.v2",
        "ok": backend["available"] or catalog["ok"],
        "backend": backend,
        "backend_error": error,
        "entrypoints": {
            "primary_page": "/v4",
            "launch": "/v4#new-task",
            "routes": "/v4#routes",
            "runs": "/v4#runs",
            "audits": "/v4#audits",
            "self_evolution": "/v4#evolution",
            "runs_api": "/api/v4/runs",
            "jobs_api": "/api/v4/jobs",
        },
        "runs": runs,
        "showcase": catalog,
        "self_evolution": self_evolution_catalog(gateway),
        "semantics": {
            "canonical_backend_is_the_only_run_authority": True,
            "one_user_facing_page": True,
            "showcase_artifacts_are_read_only": True,
            "workbench_is_rendered_from_the_same_gateway_read_model": True,
        },
    }


def register_workspace_routes(blueprint: Blueprint, gateway_factory: Any) -> None:
    @blueprint.get("/v4")
    def v4_index() -> Response:
        return static_html("workspace.html")

    @blueprint.get("/v4/console")
    def v4_console() -> Response:
        return redirect("/v4#new-task", code=302)

    @blueprint.get("/v4/showcase")
    def v4_showcase() -> Response:
        return redirect("/v4#routes", code=302)

    @blueprint.get("/agent")
    def legacy_agent_workbench() -> Response:
        return redirect("/v4#routes", code=302)

    @blueprint.get("/statins")
    def legacy_statin_showcase() -> Response:
        return redirect("/v4#audits", code=302)

    @blueprint.get("/showcase")
    def legacy_presentation_showcase() -> Response:
        return redirect("/v4#routes", code=302)

    @blueprint.get("/api/v4/workspace")
    def v4_workspace():
        return jsonify(workspace_payload(gateway_factory()))

    @blueprint.get("/api/v4/showcase")
    def v4_showcase_catalog():
        return jsonify(showcase_catalog())

    @blueprint.route("/api/v4/result-file", methods=["GET", "HEAD"])
    def v4_result_file():
        response = result_file_response(
            str(request.args.get("path") or ""),
            head_only=request.method == "HEAD",
        )
        if response is None:
            abort(404)
        return response


def result_file_response(relative_path: str, *, head_only: bool = False) -> Response | None:
    requested = str(relative_path or "").strip().replace("\\", "/")
    candidate = (ROOT / requested).resolve() if requested else ROOT.resolve()
    shared = SHARED_RESULTS_DIR.resolve()
    if not requested or not candidate.is_relative_to(shared) or not candidate.is_file():
        return None
    mimetype = {
        ".html": "text/html; charset=utf-8",
        ".htm": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".jsonl": "application/x-ndjson; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }.get(candidate.suffix.casefold(), "application/octet-stream")
    if head_only:
        response = Response(mimetype=mimetype)
        response.content_length = candidate.stat().st_size
    elif mimetype.startswith("text/") or "json" in mimetype or mimetype == "image/svg+xml":
        body = candidate.read_text(encoding="utf-8", errors="replace")
        if candidate.suffix.casefold() in {".html", ".htm"}:
            body = inject_workspace_return(body)
        response = Response(body, mimetype=mimetype)
    else:
        response = Response(candidate.read_bytes(), mimetype=mimetype)
    if candidate.suffix.casefold() in {".html", ".htm"}:
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "img-src data:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
    return response


def inject_workspace_return(value: str) -> str:
    """Add shell navigation to historical route-workbench HTML at delivery time."""

    marker = '<div class="header-actions">'
    if 'id="dashboardReturn"' in value or marker not in value or 'class="app-header"' not in value:
        return value
    return value.replace(marker, marker + "\n" + WORKSPACE_RETURN_MARKUP, 1)


__all__ = [
    "compiled_program_benchmark_catalog",
    "inject_workspace_return",
    "register_workspace_routes",
    "result_file_response",
    "self_evolution_catalog",
    "showcase_catalog",
    "static_html",
    "workspace_payload",
]
