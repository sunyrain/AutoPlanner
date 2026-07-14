"""Auditable learning and retrieval for replayed patent reaction templates."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.canonical_identity import molecule_identity
from cascade_planner.application.reaction_proof_versions import active_reaction_proofs
from cascade_planner.application.reaction_template_extraction import (
    apply_retro_template,
    extract_retro_template,
)
from cascade_planner.application.reaction_template_examples import (
    merge_template_example,
)
from cascade_planner.application.reaction_template_store import (
    DEFAULT_TEMPLATE_LIBRARY_NAME,
    TEMPLATE_LIBRARY_SCHEMA,
    TEMPLATE_RECORD_SCHEMA,
    build_template_library,
    read_template_library,
    template_digest,
    template_library_lock,
    valid_digest,
    write_template_library,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def synchronize_patent_template_library(
    path: str | Path,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Learn eligible patent examples and record later reuse outcomes."""

    destination = Path(path).expanduser().resolve()
    with template_library_lock(destination):
        library, error = read_template_library(destination)
        if error:
            return _sync_report("blocked_library_integrity", path=destination, reason=error)
        templates = {
            str(key): dict(value)
            for key, value in dict(library.get("templates") or {}).items()
        }
        learned: set[str] = set()
        updated: set[str] = set()
        rejected: dict[str, int] = {}
        for edge_id, raw_edge in sorted(dict(graph.get("edges") or {}).items()):
            edge = dict(raw_edge)
            proof = _eligible_proof(edge)
            exact_ids = [
                str(value) for value in edge.get("exact_record_ids") or [] if str(value)
            ]
            if not exact_ids:
                continue
            if not proof:
                _count(rejected, "current_accepted_mapped_proof_missing")
                continue
            extraction = extract_retro_template(str(proof["mapped_reaction"]))
            if extraction.get("accepted") is not True:
                _count(rejected, str((extraction.get("reasons") or ["rejected"])[0]))
                continue
            for record_id in exact_ids:
                exact = dict(dict(graph.get("exact_records") or {}).get(record_id) or {})
                reason = _exact_patent_rejection(exact, edge, graph)
                if reason:
                    _count(rejected, reason)
                    continue
                template_id = str(extraction["template_id"])
                existing = dict(templates.get(template_id) or {})
                existing_examples = {
                    str(key): dict(value)
                    for key, value in dict(existing.get("examples") or {}).items()
                }
                examples, examples_changed = merge_template_example(
                    existing_examples,
                    exact=exact,
                    edge_id=str(edge_id),
                    edge=edge,
                    proof=proof,
                )
                if not examples_changed:
                    continue
                record = _template_record(
                    extraction,
                    examples=examples,
                    successful_edge_digests=existing.get("successful_edge_digests") or [],
                    failed_edge_digests=existing.get("failed_edge_digests") or [],
                )
                templates[template_id] = record
                (updated if existing else learned).add(template_id)
        outcome_updates = _record_reuse_outcomes(templates, graph)
        changed = bool(learned or updated or outcome_updates)
        if changed:
            library = build_template_library(
                templates,
                generation=int(library.get("generation") or 0) + 1,
            )
            write_template_library(destination, library)
        return {
            **_sync_report(
                "completed" if changed else "reused_or_empty",
                path=destination,
            ),
            "library_sha256": str(library.get("content_sha256") or ""),
            "generation": int(library.get("generation") or 0),
            "template_count": len(templates),
            "learned_template_ids": sorted(learned),
            "updated_template_ids": sorted(updated),
            "reuse_outcome_update_count": outcome_updates,
            "rejected_example_counts": dict(sorted(rejected.items())),
            "model_invocations": 0,
        }


def retrieve_patent_template_candidates(
    path: str | Path,
    *,
    graph: Mapping[str, Any],
    target_smiles: str,
    max_candidates: int = 12,
) -> dict[str, Any]:
    """Apply ranked templates to the target and current open leaves."""

    destination = Path(path).expanduser().resolve()
    library, error = read_template_library(destination)
    if error:
        return _retrieval_report(
            "blocked_library_integrity",
            destination,
            reason=error,
        )
    templates = [
        dict(value)
        for value in dict(library.get("templates") or {}).values()
        if value.get("status") != "quarantined"
    ]
    products = _frontier_products(graph, target_smiles)
    existing = {
        str(edge.get("edge_digest") or "")
        for edge in dict(graph.get("edges") or {}).values()
    }
    route_aliases = {
        str(route_id): tuple(
            sorted(str(alias) for alias in route.get("aliases") or [] if str(alias))
        )
        for route_id, route in dict(graph.get("route_families") or {}).items()
    }
    existing_routes_by_digest: dict[str, set[str]] = {}
    for edge in dict(graph.get("edges") or {}).values():
        digest = str(edge.get("edge_digest") or "")
        for route_id in edge.get("route_family_ids") or []:
            existing_routes_by_digest.setdefault(digest, set()).update(
                route_aliases.get(str(route_id), ())
            )
    proposals: list[dict[str, Any]] = []
    rejected = 0
    exact_example_exclusions = 0
    for template in sorted(templates, key=_template_rank)[:64]:
        example_products = {
            str(value.get("product_smiles") or "")
            for value in dict(template.get("examples") or {}).values()
        }
        for product_smiles, aliases in products:
            if product_smiles in example_products:
                exact_example_exclusions += 1
                continue
            outcomes = apply_retro_template(
                str(template.get("reaction_smarts") or ""),
                product_smiles,
                max_outcomes=4,
            )
            for precursors in outcomes:
                audit = audit_retrosynthetic_candidate(product_smiles, precursors)
                if audit.get("accepted") is not True:
                    rejected += 1
                    continue
                existing_edge_match = str(audit.get("edge_digest") or "") in existing
                proposal_id = "self-evo:" + template_digest(
                    {
                        "template_id": template.get("template_id"),
                        "product_smiles": audit["product_smiles"],
                        "precursor_smiles": audit["precursor_smiles_multiset"],
                    }
                )[:24]
                routed_aliases = (
                    tuple(
                        sorted(
                            existing_routes_by_digest.get(
                                str(audit["edge_digest"]), set()
                            )
                        )
                    )
                    if existing_edge_match
                    else aliases
                )
                for alias in routed_aliases or ("",):
                    proposals.append(
                        {
                            "proposal_id": proposal_id,
                            "product_smiles": audit["product_smiles"],
                            "precursor_smiles": audit["precursor_smiles_multiset"],
                            "route_family_id": alias,
                            "origin_kind": "self_evo_patent_template",
                            "origin_ref": str(template.get("template_id") or ""),
                            "transformation_hypothesis": (
                                "replayed patent-derived reaction-center template"
                            ),
                            "existing_edge_match": existing_edge_match,
                            "template_support": _template_support(template),
                        }
                    )
                existing.add(str(audit["edge_digest"]))
                if len({row["proposal_id"] for row in proposals}) >= max_candidates:
                    break
            if len({row["proposal_id"] for row in proposals}) >= max_candidates:
                break
        if len({row["proposal_id"] for row in proposals}) >= max_candidates:
            break
    unique_count = len({row["proposal_id"] for row in proposals})
    return {
        **_retrieval_report(
            "completed" if proposals else "reused_or_empty",
            destination,
        ),
        "library_sha256": str(library.get("content_sha256") or ""),
        "generation": int(library.get("generation") or 0),
        "template_count": len(templates),
        "frontier_product_count": len(products),
        "candidate_count": unique_count,
        "proposals": proposals,
        "rejected_application_count": rejected,
        "exact_example_exclusion_count": exact_example_exclusions,
        "model_invocations": 0,
    }


def load_patent_template_library(path: str | Path) -> dict[str, Any]:
    """Read a valid library, returning an empty valid library when absent."""

    library, error = read_template_library(Path(path).expanduser().resolve())
    if error:
        raise ValueError(error)
    return library


def _eligible_proof(edge: Mapping[str, Any]) -> dict[str, Any]:
    for proof in reversed(active_reaction_proofs(edge.get("reaction_proofs") or [])):
        row = dict(proof)
        if (
            row.get("accepted") is True
            and str(row.get("mapped_reaction") or "")
            and valid_digest(row, "proof_digest")
        ):
            return row
    return {}


def _exact_patent_rejection(
    exact: Mapping[str, Any],
    edge: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> str:
    if not exact or not valid_digest(exact, "content_sha256"):
        return "exact_record_digest_invalid"
    if (
        exact.get("relation_type") != "exact"
        or exact.get("authority_scope") != "source_exact_structure_observation"
        or exact.get("not_reaction_validation") is not True
        or exact.get("edge_digest") != edge.get("edge_digest")
    ):
        return "exact_record_authority_invalid"
    if str(exact.get("source_ref") or "").lower().startswith("patent:"):
        return ""
    external = str(exact.get("source_binding_id") or "")
    canonical = str(dict(graph.get("source_aliases") or {}).get(external) or external)
    binding = dict(dict(graph.get("source_bindings") or {}).get(canonical) or {})
    return "" if binding.get("source_kind") == "patent" else "exact_record_not_patent"


def _record_reuse_outcomes(
    templates: dict[str, dict[str, Any]],
    graph: Mapping[str, Any],
) -> int:
    updates = 0
    for edge in dict(graph.get("edges") or {}).values():
        origins = [
            dict(value)
            for value in edge.get("origin_records") or []
            if isinstance(value, Mapping)
            and value.get("origin_kind") == "self_evo_patent_template"
        ]
        proofs = [
            value
            for value in active_reaction_proofs(edge.get("reaction_proofs") or [])
            if value.get("schema_version") == "reaction_step_proof.v1"
            and valid_digest(value, "proof_digest")
        ]
        if not origins or not proofs:
            continue
        edge_digest = str(edge.get("edge_digest") or "")
        accepted = any(value.get("accepted") is True for value in proofs)
        for origin in origins:
            template_id = str(origin.get("origin_ref") or "")
            if template_id not in templates:
                continue
            template = dict(templates[template_id])
            successes = set(template.get("successful_edge_digests") or [])
            failures = set(template.get("failed_edge_digests") or [])
            before = (set(successes), set(failures))
            if accepted:
                successes.add(edge_digest)
                failures.discard(edge_digest)
            else:
                failures.add(edge_digest)
            if before == (successes, failures):
                continue
            templates[template_id] = _template_record(
                template,
                examples=dict(template.get("examples") or {}),
                successful_edge_digests=successes,
                failed_edge_digests=failures,
            )
            updates += 1
    return updates


def _template_record(
    extraction: Mapping[str, Any],
    *,
    examples: Mapping[str, Mapping[str, Any]],
    successful_edge_digests: Any,
    failed_edge_digests: Any,
) -> dict[str, Any]:
    example_rows = {str(key): dict(value) for key, value in sorted(examples.items())}
    successes = sorted(set(successful_edge_digests or []))
    failures = sorted(set(failed_edge_digests or []))
    source_groups = sorted(
        {
            str(value.get("independence_group") or "")
            for value in example_rows.values()
        }
        - {""}
    )
    maturity = (
        "reuse_validated"
        if successes
        else "source_corroborated"
        if len(source_groups) >= 2
        else "single_source_observed"
    )
    row = {
        "schema_version": TEMPLATE_RECORD_SCHEMA,
        "template_id": str(extraction.get("template_id") or ""),
        "reaction_smarts": str(extraction.get("reaction_smarts") or ""),
        "extractor_version": str(extraction.get("extractor_version") or ""),
        "radius": int(extraction.get("radius") or 0),
        "examples": example_rows,
        "example_count": len(example_rows),
        "source_refs": sorted(
            {str(value.get("source_ref") or "") for value in example_rows.values()} - {""}
        ),
        "independent_source_groups": source_groups,
        "successful_edge_digests": successes,
        "failed_edge_digests": failures,
        "maturity": maturity,
        "status": "quarantined" if len(failures) >= 3 and not successes else "active",
        "authority_scope": "proposal_memory_only",
        "semantics": {
            "patent_exact_row_and_current_host_proof_required_to_learn": True,
            "reuse_requires_normal_materialization_mapping_and_validation": True,
            "template_never_grants_reaction_or_stock_authority": True,
        },
    }
    row["content_sha256"] = template_digest(row)
    return row


def _frontier_products(
    graph: Mapping[str, Any],
    target_smiles: str,
) -> list[tuple[str, tuple[str, ...]]]:
    products: dict[str, set[str]] = {}
    _, target = molecule_identity(target_smiles)
    if target:
        # Target-level templates are global options for Codex.  Assigning one
        # automatically to every existing family contaminates unrelated route
        # strategies and spends expansion budget before global selection.
        products[target] = set()
    molecules = dict(graph.get("molecules") or {})
    for route in dict(graph.get("route_families") or {}).values():
        aliases = {str(value) for value in route.get("aliases") or [] if str(value)}
        for molecule_id in route.get("leaf_molecule_ids") or []:
            molecule = dict(molecules.get(molecule_id) or {})
            smiles = str(molecule.get("canonical_smiles") or "")
            if (
                smiles
                and molecule.get("stock_closed") is not True
                and molecule.get("provider_expansion_requested") is True
            ):
                products.setdefault(smiles, set()).update(aliases)
    return [
        (smiles, tuple(sorted(aliases)))
        for smiles, aliases in sorted(products.items())
    ][:32]


def _template_rank(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -len(value.get("successful_edge_digests") or []),
        -len(value.get("independent_source_groups") or []),
        -int(value.get("example_count") or 0),
        len(value.get("failed_edge_digests") or []),
        str(value.get("template_id") or ""),
    )


def _template_support(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "example_count": int(value.get("example_count") or 0),
        "independent_source_group_count": len(
            value.get("independent_source_groups") or []
        ),
        "successful_reuse_count": len(value.get("successful_edge_digests") or []),
        "failed_reuse_count": len(value.get("failed_edge_digests") or []),
        "maturity": str(value.get("maturity") or "single_source_observed"),
        "status": str(value.get("status") or "active"),
        "source_refs": list(value.get("source_refs") or [])[:4],
        "grants_no_scientific_authority": True,
    }


def _sync_report(status: str, *, path: Path, reason: str = "") -> dict[str, Any]:
    return {
        "schema_version": "patent_template_library_sync.v1",
        "stage": "patent_template_learning",
        "status": status,
        "library_path": str(path),
        "reason": reason,
        "semantics": {
            "learning_requires_replay_gated_patent_example": True,
            "no_model_calls": True,
        },
    }


def _retrieval_report(status: str, path: Path, *, reason: str = "") -> dict[str, Any]:
    return {
        "schema_version": "patent_template_retrieval.v1",
        "stage": "patent_template_retrieval",
        "status": status,
        "library_path": str(path),
        "reason": reason,
        "candidate_count": 0,
        "proposals": [],
        "semantics": {
            "suggestions_are_not_evidence": True,
            "normal_candidate_gate_required": True,
            "exact_training_example_products_are_excluded": True,
            "target_candidates_require_codex_route_selection": True,
            "no_model_calls": True,
        },
    }


def _count(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1

__all__ = [
    "DEFAULT_TEMPLATE_LIBRARY_NAME",
    "TEMPLATE_LIBRARY_SCHEMA",
    "load_patent_template_library",
    "retrieve_patent_template_candidates",
    "synchronize_patent_template_library",
]
