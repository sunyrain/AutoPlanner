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
from cascade_planner.application.retrosynthesis_run_contract import RetrosynthesisRunBudget
from cascade_planner.harness.route_forest_delivery import render_route_forest_html
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.web.workspace_catalog import compile_showcase_catalog
from cascade_planner.web.workspace_visibility import (
    WORKSPACE_VISIBILITY_SCHEMA,
    WorkspaceVisibilityError,
    workspace_visibility_store,
)


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
    mechanism_packs, mechanism_errors = _compiled_mechanism_proposal_packs()
    errors.extend(mechanism_errors)
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
                observed_route = dict(
                    dict(observation.get("routes") or {}).get(str(route_id)) or {}
                )
                host_edge_ids = [
                    str(value) for value in observed_route.get("edge_ids") or [] if str(value)
                ]
                host_steps = [
                    _fallback_step_snapshot(
                        edge_id,
                        transformations=transformations,
                        molecules=molecules,
                    )
                    for edge_id in host_edge_ids
                ]
                target = dict(observation.get("target") or {})
                precursor_id = str(boundary.get("precursor_molecule_id") or "")
                product_id = str(boundary.get("product_molecule_id") or "")
                benchmark_id = "program-benchmark:" + str(
                    innovation.get("innovation_id") or candidate.get("candidate_id") or ""
                )
                legacy_benchmark_run_id = _program_benchmark_run_id(
                    benchmark_id,
                    target_name=str(target.get("name") or "target"),
                    chemical_steps=chemical_steps,
                )
                benchmark_run_id = _program_host_run_id(
                    benchmark_id,
                    target_name=str(target.get("name") or "target"),
                    route_id=str(route_id),
                    host_steps=len(host_steps),
                )
                legacy_host_run_id = _program_host_run_id(
                    benchmark_id,
                    target_name=str(target.get("name") or "target"),
                    route_id=str(route_id),
                    host_steps=len(host_steps),
                    materialization_contract=1,
                )
                records.append(
                    {
                        "benchmark_id": benchmark_id,
                        "benchmark_run_id": benchmark_run_id,
                        "materialize_url": (
                            "/api/v4/program-benchmarks/" + benchmark_id + "/materialize"
                        ),
                        "workbench_url": f"/api/v4/runs/{benchmark_run_id}/workbench.html",
                        "legacy_run_ids": [legacy_benchmark_run_id, legacy_host_run_id],
                        "source_file": path.name,
                        "target_name": str(target.get("name") or "unnamed target"),
                        "target_key": str(target.get("name") or "unnamed target").casefold(),
                        "target": _molecule_snapshot(
                            str(target.get("molecule_id") or ""),
                            molecules=molecules,
                            fallback_smiles=str(target.get("canonical_smiles") or ""),
                        ),
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
                        "host_route": {
                            "route_id": str(route_id),
                            "baseline_step_count": len(host_steps),
                            "hypothetical_operation_count": max(
                                1,
                                len(host_steps)
                                - int(innovation.get("step_savings") or chemical_steps - 1),
                            ),
                            "source_complete": observed_route.get("source_complete") is True,
                            "source_closure_profile": str(
                                observed_route.get("source_closure_profile") or "unknown"
                            ),
                            "source_refs": [
                                str(value)
                                for value in observed_route.get("source_refs") or []
                                if str(value)
                            ],
                            "warning_codes": [
                                str(value)
                                for value in observed_route.get("warning_codes") or []
                                if str(value)
                            ],
                            "replaced_edge_ids": replaced_edge_ids,
                            "steps": host_steps,
                        },
                        "mechanism_hypothesis_count": len(
                            _matching_mechanism_proposals(
                                mechanism_packs,
                                target_name=str(target.get("name") or "unnamed target"),
                                host_route_id=str(route_id),
                            )
                        ),
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
            "mechanism_hypothesis_counts_are_digest_bound_host_annotations": True,
        },
    }


def compiled_program_overlay_attachments(run_id: str) -> tuple[dict[str, Any], ...]:
    """Return exact host-route bindings for one materialized benchmark run.

    The attachment is intentionally a read-only route annotation.  It can only
    be projected when all source edges and both molecular boundaries match one
    displayed branch, and it never replaces the retained chemical fallback.
    """

    attachments: list[dict[str, Any]] = []
    for raw_record in compiled_program_benchmark_catalog().get("records") or []:
        record = dict(raw_record)
        if str(record.get("benchmark_run_id") or "") != str(run_id):
            continue
        host_route = dict(record.get("host_route") or {})
        attachments.append(
            {
                "schema_version": "route_program_attachment.v1",
                "program_id": str(record.get("innovation_id") or record.get("candidate_id") or ""),
                "program_kind": "biocatalytic_superstep",
                "host_route_id": str(host_route.get("route_id") or ""),
                "host_step_evidence": [
                    {
                        "edge_id": str(value.get("edge_id") or ""),
                        "proof_level": int(value.get("proof_level") or 0),
                        "source_refs": [
                            str(ref) for ref in value.get("source_refs") or [] if str(ref)
                        ],
                        "warning_codes": [
                            str(code) for code in value.get("warning_codes") or [] if str(code)
                        ],
                    }
                    for value in host_route.get("steps") or []
                    if isinstance(value, dict) and str(value.get("edge_id") or "")
                ],
                "replaced_edge_ids": [
                    str(value) for value in host_route.get("replaced_edge_ids") or [] if str(value)
                ],
                "boundary": dict(record.get("boundary") or {}),
                "chemical_step_equivalent_count": int(
                    record.get("chemical_step_equivalent_count") or 0
                ),
                "physical_step_count": int(record.get("physical_step_count") or 1),
                "net_step_savings": int(record.get("net_step_savings") or 0),
                "capability_id": str(record.get("capability_id") or ""),
                "authority_scope": str(record.get("authority_scope") or "proposal_only"),
                "validation_status": str(record.get("validation_status") or "experiment_required"),
                "warning_codes": [
                    str(value) for value in record.get("warning_codes") or [] if str(value)
                ],
                "enzyme": dict(record.get("enzyme") or {}),
                "cofactor_requirements": dict(record.get("cofactor_requirements") or {}),
                "cofactor_regenerations": dict(record.get("cofactor_regenerations") or {}),
                "selectivity_objective": str(record.get("selectivity_objective") or ""),
                "precedent_refs": [
                    str(value) for value in record.get("precedent_refs") or [] if str(value)
                ],
                "required_assays": [
                    {
                        "assay_id": "exact-host-substrate-conversion",
                        "purpose": "verify exact boundary conversion and product identity",
                    },
                    {
                        "assay_id": "stereo-and-side-product-panel",
                        "purpose": "measure selectivity, competing products, and mass balance",
                    },
                    {
                        "assay_id": "fallback-comparison",
                        "purpose": "compare isolated yield and operation count with the six-step fallback",
                    },
                ],
                "semantics": {
                    "route_attachment_not_standalone_route": True,
                    "exact_host_binding_required": True,
                    "chemical_fallback_retained": True,
                    "cannot_grant_route_completion": True,
                },
            }
        )
    return tuple(attachments)


def compiled_mechanism_hypothesis_attachments(
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    """Return digest-bound one-hop proposals attached to a materialized host route."""

    packs, _errors = _compiled_mechanism_proposal_packs()
    attachments: list[dict[str, Any]] = []
    for raw_record in compiled_program_benchmark_catalog().get("records") or []:
        record = dict(raw_record)
        if str(record.get("benchmark_run_id") or "") != str(run_id):
            continue
        host_route = dict(record.get("host_route") or {})
        matches = _matching_mechanism_proposals(
            packs,
            target_name=str(record.get("target_name") or ""),
            host_route_id=str(host_route.get("route_id") or ""),
        )
        for pack, proposal in matches:
            anchor_edge_ids = [
                str(value) for value in proposal.get("anchor_edge_ids") or [] if str(value)
            ]
            anchor_source_refs = [
                str(value) for value in proposal.get("anchor_source_refs") or [] if str(value)
            ]
            checks = [
                str(value) for value in proposal.get("falsifiable_checks") or [] if str(value)
            ]
            proposal_id = str(proposal.get("proposal_id") or "")
            precursor_smiles = str(proposal.get("precursor_smiles") or "")
            product_smiles = str(proposal.get("product_smiles") or "")
            rationale = str(proposal.get("mechanistic_rationale") or "")
            if (
                not proposal_id
                or int(proposal.get("proposal_depth") or 0) != 1
                or not anchor_edge_ids
                or not anchor_source_refs
                or not precursor_smiles
                or not product_smiles
                or product_smiles == precursor_smiles
                or not rationale
                or not checks
            ):
                continue
            attachments.append(
                {
                    "schema_version": "route_mechanism_hypothesis_attachment.v1",
                    "hypothesis_id": proposal_id,
                    "host_route_id": str(host_route.get("route_id") or ""),
                    "anchor_edge_ids": anchor_edge_ids,
                    "precursor_smiles": precursor_smiles,
                    "proposed_product": {
                        "label": str(proposal.get("product_label") or "proposed one-hop product"),
                        "smiles": product_smiles,
                    },
                    "proposal_depth": 1,
                    "mechanistic_rationale": rationale,
                    "elementary_steps": [
                        str(value) for value in proposal.get("elementary_steps") or [] if str(value)
                    ],
                    "falsifiable_checks": checks,
                    "anchor_source_refs": anchor_source_refs,
                    "priority_score": float(proposal.get("priority_score") or 0.0),
                    "authority_scope": "proposal_only",
                    "validation_status": "host_materialization_required",
                    "warning_codes": [
                        "MECHANISM_HYPOTHESIS_UNVALIDATED",
                        "PRODUCT_NOT_ROUTE_REJOINED",
                    ],
                    "source": {
                        "file": str(pack.get("_source_file") or ""),
                        "content_sha256": str(pack.get("content_sha256") or ""),
                    },
                    "semantics": {
                        "display_only_shadow_layer": True,
                        "anchor_evidence_not_promoted": True,
                        "not_a_canonical_reaction_edge": True,
                        "cannot_grant_route_completion": True,
                        "one_hop_only": True,
                    },
                }
            )
    return tuple(sorted(attachments, key=lambda value: str(value.get("hypothesis_id") or "")))


def materialize_compiled_program_benchmark(
    gateway: Any,
    benchmark_id: str,
) -> dict[str, Any]:
    """Create one canonical campaign from a digest-bound Program benchmark."""

    catalog = compiled_program_benchmark_catalog()
    record = next(
        (
            dict(value)
            for value in catalog.get("records") or []
            if str(value.get("benchmark_id") or "") == str(benchmark_id)
        ),
        None,
    )
    if record is None:
        raise ValueError("compiled_program_benchmark_not_found")
    fallback = [dict(value) for value in record.get("fallback_steps") or []]
    if len(fallback) < 2:
        raise ValueError("compiled_program_benchmark_fallback_invalid")
    host_route = dict(record.get("host_route") or {})
    host_steps = [dict(value) for value in host_route.get("steps") or []]
    if len(host_steps) < len(fallback):
        raise ValueError("compiled_program_benchmark_host_route_invalid")
    replaced_edge_ids = {
        str(value) for value in host_route.get("replaced_edge_ids") or [] if str(value)
    }
    steps = []
    for row in host_steps:
        product = dict(row.get("product") or {})
        precursors = [
            str(value.get("smiles") or "")
            for value in row.get("precursors") or []
            if isinstance(value, dict) and str(value.get("smiles") or "")
        ]
        product_smiles = str(product.get("smiles") or "")
        if not product_smiles or not precursors:
            raise ValueError("compiled_program_benchmark_boundary_smiles_missing")
        steps.append(
            {
                "step_id": str(row.get("edge_id") or ""),
                "product_smiles": product_smiles,
                "precursor_smiles": precursors,
                "transformation_hypothesis": (
                    "digest-bound reported chemical fallback within Program span"
                    if str(row.get("edge_id") or "") in replaced_edge_ids
                    else "digest-bound host-route step; evidence level remains independent"
                ),
            }
        )
    run_id = str(record["benchmark_run_id"])
    target_name = (
        f"{record.get('target_name') or 'target'} reported "
        f"{host_route.get('baseline_step_count')} step host route"
    )
    try:
        created = gateway.status(run_id)
    except Exception:
        created = gateway.create_run(
            run_id=run_id,
            target_name=target_name,
            target_smiles=str(dict(record.get("target") or {}).get("smiles") or ""),
            budget=RetrosynthesisRunBudget(
                max_model_invocations=0,
                max_visual_invocations=0,
                max_accepted_expansions=max(32, len(steps)),
                max_attempt_runs=max(32, len(steps)),
            ),
            global_plan={
                "schema_version": "global_campaign_plan.v1",
                "route_families": [
                    {
                        "route_family_id": f"family:{run_id}",
                        "strategic_disconnection": (
                            "complete reported host route with one enzyme Program shadow"
                        ),
                    }
                ],
                "multi_step_skeletons": [
                    {
                        "skeleton_id": f"skeleton:{run_id}",
                        "route_family_id": f"family:{run_id}",
                        "steps": steps,
                    }
                ],
            },
            materialize=True,
        )
    return {
        "schema_version": "autoplanner.compiled_program_benchmark_materialization.v1",
        "benchmark_id": str(record["benchmark_id"]),
        "run_id": run_id,
        "workbench_url": str(record["workbench_url"]),
        "host_route_id": str(record.get("route_id") or ""),
        "chemical_baseline_step_count": int(host_route.get("baseline_step_count") or 0),
        "hypothetical_operation_count": int(host_route.get("hypothetical_operation_count") or 0),
        "status": dict(created.get("status") or {}),
        "semantics": {
            "complete_canonical_host_route_materialized": True,
            "canonical_chemical_fallback_materialized": True,
            "program_review_remains_read_only": True,
            "materialization_does_not_admit_the_program": True,
        },
    }


def _program_benchmark_run_id(
    benchmark_id: str,
    *,
    target_name: str,
    chemical_steps: int,
) -> str:
    slug = (
        "".join(
            value if value.isascii() and value.isalnum() else "-"
            for value in target_name.casefold()
        ).strip("-")
        or "target"
    )
    digest = strict_canonical_json_sha256({"benchmark_id": benchmark_id})[:10]
    return f"program-benchmark-{slug[:24]}-{chemical_steps}to1-{digest}"


def _program_host_run_id(
    benchmark_id: str,
    *,
    target_name: str,
    route_id: str,
    host_steps: int,
    materialization_contract: int = 2,
) -> str:
    slug = (
        "".join(
            value if value.isascii() and value.isalnum() else "-"
            for value in target_name.casefold()
        ).strip("-")
        or "target"
    )
    digest = strict_canonical_json_sha256(
        {
            "benchmark_id": benchmark_id,
            "route_id": route_id,
            "host_steps": host_steps,
            "materialization_contract": materialization_contract,
        }
    )[:10]
    return f"program-host-{slug[:24]}-{host_steps}step-{digest}"


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


def _compiled_mechanism_proposal_packs() -> tuple[list[dict[str, Any]], list[str]]:
    packs: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((ROOT / "benchmarks").glob("*_route_innovation_proposals.v1.json")):
        value, error = _read_digest_bound_json(path)
        if error:
            errors.append(f"{path.name}:{error}")
            continue
        if value.get("schema_version") != "route_mechanism_proposal_replay.v1":
            errors.append(f"{path.name}:schema_invalid")
            continue
        packs.append({**value, "_source_file": path.name})
    return packs, errors


def _matching_mechanism_proposals(
    packs: list[dict[str, Any]],
    *,
    target_name: str,
    host_route_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    target_key = str(target_name or "").casefold()
    for pack in packs:
        if str(pack.get("target_name") or "").casefold() != target_key or str(
            pack.get("host_route_id") or ""
        ) != str(host_route_id or ""):
            continue
        matches.extend(
            (pack, dict(value)) for value in pack.get("proposals") or [] if isinstance(value, dict)
        )
    return matches


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
        "proof_level": int(transformation.get("proof_level") or 0),
        "warning_codes": [
            str(value) for value in transformation.get("warning_codes") or [] if str(value)
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
        examples = []
        for example_id, raw_example in sorted(dict(row.get("examples") or {}).items()):
            if not isinstance(raw_example, dict):
                continue
            example = dict(raw_example)
            examples.append(
                {
                    "example_id": str(example_id),
                    "record_id": str(example.get("record_id") or ""),
                    "source_ref": str(example.get("source_ref") or ""),
                    "claim_scope_id": str(example.get("claim_scope_id") or ""),
                    "precursor_smiles": [
                        str(value) for value in example.get("precursor_smiles") or [] if str(value)
                    ],
                    "product_smiles": str(example.get("product_smiles") or ""),
                    "conditions": dict(example.get("conditions") or {}),
                    "condition_completeness": dict(example.get("condition_completeness") or {}),
                    "procedure_authority_scope": str(
                        example.get("procedure_authority_scope") or ""
                    ),
                    "location_refs": [
                        str(value) for value in example.get("location_refs") or [] if str(value)
                    ],
                    "edge_digest": str(example.get("edge_digest") or ""),
                    "proof_digest": str(example.get("proof_digest") or ""),
                }
            )
        template_rows.append(
            {
                "template_id": str(row.get("template_id") or ""),
                "status": str(row.get("status") or "active"),
                "maturity": str(row.get("maturity") or "single_source_observed"),
                "authority_scope": str(row.get("authority_scope") or "proposal_memory_only"),
                "reaction_smarts": str(row.get("reaction_smarts") or ""),
                "radius": row.get("radius"),
                "extractor_version": str(row.get("extractor_version") or ""),
                "example_count": int(row.get("example_count") or 0),
                "independent_source_group_count": len(row.get("independent_source_groups") or []),
                "independent_source_groups": [
                    str(value) for value in row.get("independent_source_groups") or [] if str(value)
                ],
                "source_refs": [str(value) for value in row.get("source_refs") or [] if str(value)],
                "successful_reuse_count": successes,
                "failed_reuse_count": failures,
                "has_reuse_outcome": bool(successes or failures),
                "replay_validated": row.get("maturity") == "reuse_validated",
                "examples": examples,
                "semantics": dict(row.get("semantics") or {}),
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
                "observations": [
                    {**dict(observation), "observation_id": str(observation_id)}
                    for observation_id, observation in sorted(
                        dict(row.get("observations") or {}).items()
                    )
                    if isinstance(observation, dict)
                ],
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
        visibility = workspace_visibility_store(gateway).snapshot()
        visibility_error = ""
    except WorkspaceVisibilityError as exc:
        visibility = {
            "schema_version": WORKSPACE_VISIBILITY_SCHEMA,
            "revision": 0,
            "updated_at": "",
            "hidden_routes": {},
            "hidden_queue_runs": {},
        }
        visibility_error = str(exc)
    hidden_route_ids = set(dict(visibility.get("hidden_routes") or {}))
    hidden_queue_run_ids = set(dict(visibility.get("hidden_queue_runs") or {}))
    program_benchmarks = compiled_program_benchmark_catalog()
    system_run_ids = sorted(
        {
            run_id
            for row in program_benchmarks.get("records") or []
            for run_id in (
                [str(row.get("benchmark_run_id") or "")]
                + [str(value) for value in row.get("legacy_run_ids") or []]
            )
            if run_id
        }
    )
    system_run_id_set = set(system_run_ids)
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
    # Retired materialization-contract revisions are route examples as well.
    # Prefix classification keeps them out of the task queue without requiring
    # target-specific run ids in source code.
    system_run_id_set.update(
        str(row.get("run_id") or "")
        for row in runs
        if str(row.get("run_id") or "").startswith(("program-host-", "program-benchmark-"))
    )
    system_run_ids = sorted(value for value in system_run_id_set if value)
    for row in runs:
        run_id = str(row.get("run_id") or "")
        row["workbench_url"] = f"/api/v4/runs/{run_id}/workbench.html" if run_id else ""
        row["workbench_pdf_url"] = (
            f"/api/v4/runs/{run_id}/workbench.pdf" if run_id else ""
        )
        row["status_url"] = f"/api/v4/runs/{run_id}/status" if run_id else ""
        row["history_delete_url"] = (
            f"/api/v4/runs/{run_id}/history" if run_id else ""
        )
        row["surface_role"] = "route_example" if run_id in system_run_id_set else "task"
        row["show_in_route_catalog"] = f"run:{run_id}" not in hidden_route_ids
        row["show_in_task_queue"] = (
            run_id not in system_run_id_set and run_id not in hidden_queue_run_ids
        )
    backend["task_run_count"] = sum(bool(row.get("show_in_task_queue")) for row in runs)
    catalog = showcase_catalog()
    return {
        "schema_version": "autoplanner.workspace.v3",
        "ok": backend["available"] or catalog["ok"],
        "backend": backend,
        "backend_error": error,
        "workspace_visibility": {
            **visibility,
            "error": visibility_error,
            "hidden_route_ids": sorted(hidden_route_ids),
            "hidden_queue_run_ids": sorted(hidden_queue_run_ids),
            "semantics": {
                "deletion_is_recoverable_projection_removal": True,
                "scientific_artifacts_are_preserved": True,
            },
        },
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
        "route_workbench": {
            "schema_version": "autoplanner.route_workbench_catalog.v1",
            "program_benchmarks": program_benchmarks,
            "system_run_ids": system_run_ids,
            "semantics": {
                "owns_all_route_graph_review": True,
                "run_outputs_are_reviewed_here_not_in_task_center": True,
                "program_benchmarks_are_route_level_proposals": True,
                "program_benchmarks_are_not_self_evolution_memory": True,
                "mechanism_hypotheses_are_host_route_callouts": True,
                "mechanism_hypotheses_do_not_create_route_edges": True,
            },
        },
        "showcase": catalog,
        "self_evolution": self_evolution_catalog(gateway),
        "semantics": {
            "canonical_backend_is_the_only_run_authority": True,
            "one_user_facing_page": True,
            "showcase_artifacts_are_read_only": True,
            "workbench_is_rendered_from_the_same_gateway_read_model": True,
        },
    }


def _workspace_route_ids(gateway: Any) -> set[str]:
    identities = {
        f"run:{str(row.get('run_id') or '')}"
        for row in gateway.list_runs(limit=1_000).get("runs") or []
        if isinstance(row, dict) and str(row.get("run_id") or "")
    }
    identities.update(
        f"program-host:{str(row.get('benchmark_id') or '')}"
        for row in compiled_program_benchmark_catalog().get("records") or []
        if isinstance(row, dict) and str(row.get("benchmark_id") or "")
    )
    identities.update(
        str(row.get("case_id") or "")
        for row in showcase_catalog().get("cases") or []
        if isinstance(row, dict) and str(row.get("case_id") or "")
    )
    return identities


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

    def delete_workspace_route(route_id: str):
        route_id = str(route_id or "").strip()
        if not route_id:
            return jsonify(
                {"error": "workspace_route_delete_invalid", "reason": "route_id_missing"}
            ), 400
        gateway = gateway_factory()
        if route_id not in _workspace_route_ids(gateway):
            return jsonify(
                {"error": "workspace_route_not_found", "route_id": route_id}
            ), 404
        try:
            return jsonify(workspace_visibility_store(gateway).hide_route(route_id))
        except WorkspaceVisibilityError as exc:
            return jsonify(
                {"error": "workspace_route_delete_failed", "reason": str(exc)}
            ), 400

    @blueprint.delete("/api/v4/workspace/routes")
    def v4_delete_workspace_route():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            payload = {}
        return delete_workspace_route(str(payload.get("route_id") or ""))

    @blueprint.delete("/api/v4/workspace/routes/<path:route_id>")
    def v4_delete_workspace_route_legacy(route_id: str):
        return delete_workspace_route(route_id)

    @blueprint.post("/api/v4/workspace/visibility/restore")
    def v4_restore_workspace_visibility():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(
                workspace_visibility_store(gateway_factory()).restore(
                    scope=str(payload.get("scope") or "all"),
                    identity=str(payload.get("identity") or ""),
                )
            )
        except WorkspaceVisibilityError as exc:
            return jsonify(
                {"error": "workspace_visibility_restore_failed", "reason": str(exc)}
            ), 400

    @blueprint.get("/api/v4/showcase")
    def v4_showcase_catalog():
        return jsonify(showcase_catalog())

    @blueprint.post("/api/v4/program-benchmarks/<path:benchmark_id>/materialize")
    def v4_materialize_program_benchmark(benchmark_id: str):
        return jsonify(
            materialize_compiled_program_benchmark(
                gateway_factory(),
                benchmark_id,
            )
        ), 201

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
    if head_only and candidate.suffix.casefold() in {".html", ".htm"}:
        body = inject_workspace_return(_delivered_text_body(candidate))
        response = Response(mimetype=mimetype)
        response.content_length = len(body.encode("utf-8"))
    elif head_only:
        response = Response(mimetype=mimetype)
        response.content_length = candidate.stat().st_size
    elif mimetype.startswith("text/") or "json" in mimetype or mimetype == "image/svg+xml":
        body = _delivered_text_body(candidate)
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


def _delivered_text_body(candidate: Path) -> str:
    """Render stored route forests with the current interaction client.

    Showcase artifacts are immutable scientific snapshots, but their generated
    HTML shell is not scientific authority.  Re-rendering a sibling forest JSON
    at delivery time keeps historical routes clickable after UI fixes without
    changing the stored route data or its digest-bound semantics.
    """

    if candidate.name == "route_forest.html":
        forest_path = candidate.with_name("explored_route_forest.json")
        if forest_path.is_file():
            try:
                forest = json.loads(forest_path.read_text(encoding="utf-8"))
                if isinstance(forest, dict):
                    return render_route_forest_html(forest)
            except Exception:  # Preserve the stored historical client if re-rendering fails.
                pass
    return candidate.read_text(encoding="utf-8", errors="replace")


def inject_workspace_return(value: str) -> str:
    """Add shell navigation to historical route-workbench HTML at delivery time."""

    marker = '<div class="header-actions">'
    if 'id="dashboardReturn"' in value or marker not in value or 'class="app-header"' not in value:
        return value
    return value.replace(marker, marker + "\n" + WORKSPACE_RETURN_MARKUP, 1)


__all__ = [
    "compiled_mechanism_hypothesis_attachments",
    "compiled_program_benchmark_catalog",
    "compiled_program_overlay_attachments",
    "inject_workspace_return",
    "materialize_compiled_program_benchmark",
    "register_workspace_routes",
    "result_file_response",
    "self_evolution_catalog",
    "showcase_catalog",
    "static_html",
    "workspace_payload",
]
