"""Project the established route-consensus graph into typed v2 records."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from cascade_planner.routes.domain import (
    ROUTE_HYPERGRAPH_OVERLAY_SCHEMA,
    ROUTE_NEIGHBORHOOD_SCHEMA,
    AlternativeSet,
    EvidenceClaim,
    MoleculeIdentity,
    ReactionCandidateEnvelope,
    ReactionHyperedge,
    RouteVariant,
    stable_content_hash,
    stable_domain_id,
)


def build_route_hypergraph_v2_overlay(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build a content-addressed overlay without changing v1 graph semantics."""
    molecule_by_v1_id: dict[str, MoleculeIdentity] = {}
    for row in graph.get("nodes") or []:
        if not isinstance(row, Mapping):
            continue
        molecule = MoleculeIdentity(str(row.get("canonical_isomeric_smiles") or row.get("smiles") or ""))
        if not molecule.validate():
            molecule_by_v1_id[str(row.get("node_id") or molecule.molecule_id)] = molecule

    claims_by_id: dict[str, EvidenceClaim] = {}
    envelopes_by_id: dict[str, ReactionCandidateEnvelope] = {}
    hyperedges_by_id: dict[str, ReactionHyperedge] = {}
    hyperedge_by_v1_step_id: dict[str, str] = {}
    validation_errors: list[str] = []

    for step in graph.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        product = molecule_by_v1_id.get(str(step.get("product_node_id") or ""))
        if product is None:
            product = MoleculeIdentity(str(step.get("product_smiles") or ""))
        precursors: list[MoleculeIdentity] = []
        for node_id, smiles in zip(
            step.get("precursor_node_ids") or [],
            step.get("precursor_smiles") or [],
        ):
            precursor = molecule_by_v1_id.get(str(node_id)) or MoleculeIdentity(str(smiles or ""))
            precursors.append(precursor)
        if product.validate() or not precursors or any(row.validate() for row in precursors):
            validation_errors.append(f"step:{step.get('step_id')}:invalid_molecule_identity")
            continue

        step_claims = _claims_for_step(step)
        for claim in step_claims:
            claims_by_id[claim.claim_id] = claim
            validation_errors.extend(
                f"claim:{claim.claim_id}:{reason}" for reason in claim.validate()
            )

        step_envelopes: list[ReactionCandidateEnvelope] = []
        if step_claims:
            for claim in step_claims:
                candidate_context = _candidate_condition_context(step, claim.candidate_id)
                envelope = ReactionCandidateEnvelope(
                    product=product,
                    precursors=tuple(precursors),
                    reaction_family=str(step.get("reaction_family") or "unspecified"),
                    source_candidate_ids=(claim.candidate_id,) if claim.candidate_id else (),
                    evidence_claims=(claim,),
                    transformation_rationale=" | ".join(
                        str(value) for value in step.get("rationales") or []
                    ),
                    conditions=tuple(
                        str(value)
                        for value in (
                            (candidate_context.get("conditions") or [])
                            if candidate_context
                            else step.get("conditions") or []
                        )
                    ),
                    catalysts=tuple(
                        str(value)
                        for value in (
                            ([candidate_context.get("catalyst")] if candidate_context.get("catalyst") else [])
                            if candidate_context
                            else (step.get("catalysts") or [])
                        )
                    ),
                    enzymes=tuple(
                        str(value)
                        for value in (
                            ([candidate_context.get("enzyme")] if candidate_context.get("enzyme") else [])
                            if candidate_context
                            else (step.get("enzymes") or [])
                        )
                    ),
                    limitations=tuple(str(value) for value in step.get("limitations") or []),
                    required_validation=tuple(
                        str(value) for value in step.get("required_validation") or []
                    ),
                )
                step_envelopes.append(envelope)
        else:
            step_envelopes.append(
                ReactionCandidateEnvelope(
                    product=product,
                    precursors=tuple(precursors),
                    reaction_family=str(step.get("reaction_family") or "unspecified"),
                    source_candidate_ids=tuple(str(value) for value in step.get("proposal_ids") or []),
                    transformation_rationale=" | ".join(
                        str(value) for value in step.get("rationales") or []
                    ),
                    conditions=tuple(str(value) for value in step.get("conditions") or []),
                    catalysts=tuple(str(value) for value in step.get("catalysts") or []),
                    enzymes=tuple(str(value) for value in step.get("enzymes") or []),
                    limitations=tuple(str(value) for value in step.get("limitations") or []),
                    required_validation=tuple(
                        str(value) for value in step.get("required_validation") or []
                    ),
                )
            )
        for envelope in step_envelopes:
            envelopes_by_id[envelope.envelope_id] = envelope
            validation_errors.extend(
                f"envelope:{envelope.envelope_id}:{reason}" for reason in envelope.validate()
            )

        hyperedge = ReactionHyperedge(
            product=product,
            precursors=tuple(precursors),
            candidate_envelope_ids=tuple(row.envelope_id for row in step_envelopes),
            evidence_claim_ids=tuple(row.claim_id for row in step_claims),
            source_channels=tuple(str(value) for value in step.get("source_channels") or []),
            independent_support_groups=tuple(
                str(value) for value in step.get("independent_support_groups") or []
            ),
            reaction_families=tuple(
                str(value)
                for value in (
                    step.get("reaction_families")
                    or [step.get("reaction_family") or "unspecified"]
                )
            ),
            rank_score=float(step.get("rank_score") or 0.0),
            advisory_only=True,
        )
        hyperedges_by_id[hyperedge.hyperedge_id] = hyperedge
        hyperedge_by_v1_step_id[str(step.get("step_id") or "")] = hyperedge.hyperedge_id
        validation_errors.extend(
            f"hyperedge:{hyperedge.hyperedge_id}:{reason}" for reason in hyperedge.validate()
        )

    alternatives = _alternative_sets(hyperedges_by_id.values())
    alternatives_by_product = {row.product_molecule_id: row for row in alternatives}
    variants = _route_variants(
        graph,
        molecule_by_v1_id=molecule_by_v1_id,
        hyperedge_by_v1_step_id=hyperedge_by_v1_step_id,
    )
    neighborhoods = _route_neighborhoods(
        hyperedges_by_id.values(),
        alternatives_by_product=alternatives_by_product,
        root_molecule_id=_mapped_root_id(graph, molecule_by_v1_id),
    )
    validation_errors.extend(
        f"alternative:{row.alternative_set_id}:{reason}"
        for row in alternatives
        for reason in row.validate()
    )
    validation_errors.extend(
        f"variant:{row.route_variant_id}:{reason}"
        for row in variants
        for reason in row.validate()
    )

    payload: dict[str, Any] = {
        "schema_version": ROUTE_HYPERGRAPH_OVERLAY_SCHEMA,
        "source_graph_schema_version": str(graph.get("schema_version") or ""),
        "case_id": str(graph.get("case_id") or ""),
        "root_molecule_id": _mapped_root_id(graph, molecule_by_v1_id),
        "molecules": [row.to_dict() for row in sorted(molecule_by_v1_id.values(), key=lambda item: item.molecule_id)],
        "evidence_claims": [row.to_dict() for row in sorted(claims_by_id.values(), key=lambda item: item.claim_id)],
        "candidate_envelopes": [
            row.to_dict() for row in sorted(envelopes_by_id.values(), key=lambda item: item.envelope_id)
        ],
        "reaction_hyperedges": [
            row.to_dict() for row in sorted(hyperedges_by_id.values(), key=lambda item: item.hyperedge_id)
        ],
        "alternative_sets": [
            row.to_dict() for row in sorted(alternatives, key=lambda item: item.alternative_set_id)
        ],
        "route_variants": [
            row.to_dict() for row in sorted(variants, key=lambda item: item.route_variant_id)
        ],
        "route_neighborhoods": neighborhoods,
        "v1_id_map": {
            "molecule_node_ids": {
                v1_id: molecule.molecule_id
                for v1_id, molecule in sorted(molecule_by_v1_id.items())
            },
            "step_ids": dict(sorted(hyperedge_by_v1_step_id.items())),
        },
        "validation": {
            "valid": not validation_errors,
            "errors": sorted(set(validation_errors)),
        },
        "semantics": {
            "advisory_only": True,
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "v1_graph_remains_authoritative_for_compatibility": True,
        },
    }
    payload["content_hash"] = stable_content_hash(ROUTE_HYPERGRAPH_OVERLAY_SCHEMA, payload)
    return payload


def _claims_for_step(step: Mapping[str, Any]) -> list[EvidenceClaim]:
    rows = [
        EvidenceClaim.from_source_record(record)
        for record in step.get("source_records") or []
        if isinstance(record, Mapping)
    ]
    if rows:
        return _dedupe_claims(rows)

    channels = sorted({str(value or "").strip() for value in step.get("source_channels") or [] if str(value or "").strip()})
    groups = sorted(
        {
            str(value or "").strip()
            for value in step.get("independent_support_groups") or []
            if str(value or "").strip()
        }
    )
    fallback_group = groups[0] if len(groups) == 1 else ""
    for channel in channels:
        support_group = "codex_model" if channel.startswith("codex_") else fallback_group
        rows.append(
            EvidenceClaim(
                source_channel=channel,
                support_group=support_group,
                evidence_level="model_only",
                confidence=str(step.get("confidence") or "low"),
                source_refs=tuple(str(value) for value in step.get("source_refs") or []),
                evidence_refs=tuple(str(value) for value in step.get("evidence_refs") or []),
            )
        )
    return _dedupe_claims(rows)


def _candidate_condition_context(step: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    if not candidate_id:
        return {}
    return next(
        (
            dict(record)
            for record in step.get("condition_support") or []
            if isinstance(record, Mapping)
            and str(record.get("candidate_id") or "") == candidate_id
        ),
        {},
    )


def _alternative_sets(hyperedges: Any) -> list[AlternativeSet]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for edge in hyperedges:
        grouped[edge.product.molecule_id].append(edge.hyperedge_id)
    return [
        AlternativeSet(product_molecule_id=product_id, hyperedge_ids=tuple(edge_ids))
        for product_id, edge_ids in sorted(grouped.items())
        if len(set(edge_ids)) > 1
    ]


def _route_variants(
    graph: Mapping[str, Any],
    *,
    molecule_by_v1_id: Mapping[str, MoleculeIdentity],
    hyperedge_by_v1_step_id: Mapping[str, str],
) -> list[RouteVariant]:
    root_id = _mapped_root_id(graph, molecule_by_v1_id)
    rows: list[RouteVariant] = []
    for route in graph.get("route_hypotheses") or []:
        if not isinstance(route, Mapping):
            continue
        edge_ids = tuple(
            hyperedge_by_v1_step_id[str(step_id)]
            for step_id in route.get("retrosynthetic_step_ids") or []
            if str(step_id) in hyperedge_by_v1_step_id
        )
        node_ids = tuple(
            molecule_by_v1_id[str(node_id)].molecule_id
            for node_id in route.get("node_ids") or []
            if str(node_id) in molecule_by_v1_id
        )
        frontier_ids = tuple(
            molecule_by_v1_id[str(item.get("node_id") or "")].molecule_id
            for item in route.get("frontier") or []
            if isinstance(item, Mapping) and str(item.get("node_id") or "") in molecule_by_v1_id
        )
        if edge_ids:
            rows.append(
                RouteVariant(
                    root_molecule_id=root_id,
                    retrosynthetic_hyperedge_ids=edge_ids,
                    molecule_ids=node_ids,
                    frontier_molecule_ids=frontier_ids,
                    rank_score=float(route.get("rank_score") or 0.0),
                )
            )
    return rows


def _route_neighborhoods(
    hyperedges: Any,
    *,
    alternatives_by_product: Mapping[str, AlternativeSet],
    root_molecule_id: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ReactionHyperedge]] = defaultdict(list)
    for edge in hyperedges:
        grouped[edge.product.molecule_id].append(edge)
    rows: list[dict[str, Any]] = []
    for product_id, edges in sorted(grouped.items()):
        sorted_edges = sorted(edges, key=lambda row: (-row.rank_score, row.hyperedge_id))
        alternative = alternatives_by_product.get(product_id)
        payload = {
            "neighborhood_id": stable_domain_id("neighborhood", product_id),
            "product_molecule_id": product_id,
            "reaction_hyperedge_ids": [row.hyperedge_id for row in sorted_edges],
            "alternative_set_id": alternative.alternative_set_id if alternative else "",
            "source_channels": sorted({value for row in edges for value in row.source_channels}),
            "independent_support_groups": sorted(
                {value for row in edges for value in row.independent_support_groups}
            ),
            "support_group_scope": "union_across_competing_hyperedges",
            "multi_source_hyperedge_ids": [
                row.hyperedge_id
                for row in sorted_edges
                if len(row.independent_support_groups) > 1
            ],
            "max_independent_support_group_count": max(
                (len(row.independent_support_groups) for row in edges),
                default=0,
            ),
            "evidence_claim_ids": sorted({value for row in edges for value in row.evidence_claim_ids}),
            "is_root_neighborhood": product_id == root_molecule_id,
            "advisory_only": True,
            "no_solved_claim": True,
        }
        payload["schema_version"] = ROUTE_NEIGHBORHOOD_SCHEMA
        payload["content_hash"] = stable_content_hash(ROUTE_NEIGHBORHOOD_SCHEMA, payload)
        rows.append(payload)
    return rows


def _mapped_root_id(
    graph: Mapping[str, Any],
    molecule_by_v1_id: Mapping[str, MoleculeIdentity],
) -> str:
    root_v1_id = str(graph.get("root_node_id") or "")
    molecule = molecule_by_v1_id.get(root_v1_id)
    return molecule.molecule_id if molecule else ""


def _dedupe_claims(values: list[EvidenceClaim]) -> list[EvidenceClaim]:
    return sorted({row.claim_id: row for row in values}.values(), key=lambda row: row.claim_id)
