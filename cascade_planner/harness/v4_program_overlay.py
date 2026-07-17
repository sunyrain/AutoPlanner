"""Project multi-edge Program proposals as a separate route-graph layer."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.harness.v4_route_graph_projection import reaction_graph_id


PROGRAM_OVERLAY_SCHEMA = "route_program_overlay.v1"
PROGRAM_ATTACHMENT_SCHEMA = "route_program_attachment.v1"


def compile_program_overlay_layer(
    reviews: Iterable[Mapping[str, Any]],
    attachments: Iterable[Mapping[str, Any]],
    *,
    step_id_by_branch_edge: Mapping[tuple[str, str], str],
    steps: list[dict[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    branches: list[dict[str, Any]],
    graph_nodes: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile canonical-review and exact-host attachment overlays together."""

    attachment_rows = [dict(value) for value in attachments if isinstance(value, Mapping)]
    review_overlays = project_program_overlays(
        reviews,
        step_id_by_branch_edge=step_id_by_branch_edge,
    )
    attachment_overlays = project_program_overlay_attachments(
        attachment_rows,
        steps=steps,
        nodes=nodes,
    )
    attached_ids = {str(value.get("program_id") or "") for value in attachment_overlays}
    overlays = attachment_overlays + [
        value
        for value in review_overlays
        if str(value.get("program_id") or "") not in attached_ids
    ]
    apply_program_host_evidence_attachments(
        [
            value
            for value in attachment_rows
            if str(value.get("program_id") or "") in attached_ids
        ],
        steps=steps,
        branches=branches,
    )
    for step in steps:
        reaction_node = graph_nodes.get(reaction_graph_id(str(step.get("step_id") or "")))
        if reaction_node is not None:
            reaction_node["proof_tier"] = str(
                step.get("proof_tier")
                or reaction_node.get("proof_tier")
                or "L0_advisory"
            )
    return overlays, attachment_overlays


def project_program_overlays(
    reviews: Iterable[Mapping[str, Any]],
    *,
    step_id_by_branch_edge: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Return display-only superstep spans without changing canonical edges."""

    overlays: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_review in reviews:
        if not isinstance(raw_review, Mapping):
            continue
        review = dict(raw_review)
        bundle = dict(review.get("program_bundle") or {})
        branch_id = str(bundle.get("source_route_id") or review.get("route_id") or "")
        plans = {
            str(value.get("program_id") or ""): dict(value)
            for value in dict(
                dict(review.get("validation_frontier") or {}).get("plans") or {}
            ).values()
            if isinstance(value, Mapping) and str(value.get("program_id") or "")
        }
        proposals = dict(bundle.get("program_proposals") or {})
        for raw_proposal in proposals.values():
            if not isinstance(raw_proposal, Mapping):
                continue
            proposal = dict(raw_proposal)
            program_id = str(proposal.get("program_id") or "")
            replaced_edge_ids = [
                str(value)
                for value in proposal.get("equivalent_reference_span") or []
                if str(value)
            ]
            # Single-edge enzyme options remain in the reaction inspector.  A
            # graph overlay is reserved for a real multi-edge Program boundary.
            if not program_id or program_id in seen or len(replaced_edge_ids) < 2:
                continue
            replaced_step_ids = [
                step_id_by_branch_edge.get((branch_id, edge_id), "")
                for edge_id in replaced_edge_ids
            ]
            if not branch_id or any(not value for value in replaced_step_ids):
                continue
            plan = plans.get(program_id, {})
            boundary = dict(plan.get("exact_boundary") or {})
            input_states = _boundary_states(boundary.get("input_states"))
            output_states = _boundary_states(boundary.get("output_states"))
            if not input_states or not output_states:
                continue
            screen = dict(plan.get("screen_matrix") or {})
            enzymes = dict(screen.get("enzyme_candidates") or {})
            overlay = {
                "schema_version": PROGRAM_OVERLAY_SCHEMA,
                "program_id": program_id,
                "program_kind": str(proposal.get("proposal_kind") or "biocatalytic_superstep"),
                "branch_id": branch_id,
                "source_capability_id": str(proposal.get("source_capability_id") or ""),
                "replaced_edge_ids": replaced_edge_ids,
                "replaced_step_ids": replaced_step_ids,
                "input_molecule_node_ids": [row["molecule_id"] for row in input_states],
                "output_molecule_node_ids": [row["molecule_id"] for row in output_states],
                "input_states": input_states,
                "output_states": output_states,
                "chemical_step_equivalent_count": int(
                    proposal.get("chemical_step_equivalent_count") or len(replaced_edge_ids)
                ),
                "isolated_operation_count": int(proposal.get("isolated_operation_count") or 1),
                "net_step_savings": int(proposal.get("net_step_savings") or 0),
                "status": str(proposal.get("status") or "proposal_only"),
                "validation_status": str(plan.get("status") or "experiment_required"),
                "warning_codes": [
                    str(value) for value in proposal.get("warning_codes") or [] if str(value)
                ],
                "candidate_enzyme_ids": [
                    str(value) for value in enzymes.get("candidate_ids") or [] if str(value)
                ],
                "enzyme_classes": [
                    str(value) for value in enzymes.get("classes") or [] if str(value)
                ],
                "enzyme_ec_numbers": [
                    str(value) for value in enzymes.get("ec_numbers") or [] if str(value)
                ],
                "cofactor_and_carrier_ledger": dict(
                    proposal.get("cofactor_and_carrier_ledger") or {}
                ),
                "selectivity_constraints": [
                    str(value)
                    for value in proposal.get("selectivity_constraints") or []
                    if str(value)
                ],
                "precedent_refs": [
                    str(value)
                    for value in dict(proposal.get("claim_refs") or {}).get("precedent_refs") or []
                    if str(value)
                ],
                "analogy_only": dict(proposal.get("claim_refs") or {}).get("analogy_only")
                is True,
                "required_assays": [
                    dict(value)
                    for value in plan.get("required_assays") or []
                    if isinstance(value, Mapping)
                ],
                "validation_gate": dict(proposal.get("validation_gate") or {}),
                "fallback_retained": True,
                "eligible_for_route_completion": False,
                "semantics": {
                    "display_only_shadow_layer": True,
                    "not_a_canonical_reaction_edge": True,
                    "does_not_inherit_replaced_edge_proof": True,
                    "canonical_chemical_steps_remain_visible_fallback": True,
                    "cannot_grant_route_completion": True,
                },
            }
            overlays.append(overlay)
            seen.add(program_id)
    return sorted(overlays, key=lambda row: (row["branch_id"], row["program_id"]))


def project_program_overlay_attachments(
    attachments: Iterable[Mapping[str, Any]],
    *,
    steps: Iterable[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind digest-backed proposals to one exact displayed host branch.

    Attachments are allowed to target advisory planned routes.  Binding still
    requires every replaced source step and both molecular boundaries to match;
    an absent or ambiguous host produces no overlay.
    """

    by_branch: dict[str, dict[str, dict[str, Any]]] = {}
    ambiguous_bindings: set[tuple[str, str]] = set()
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        step = dict(raw_step)
        branch_id = str(step.get("branch_id") or "")
        if not branch_id:
            continue
        for source_id in step.get("source_step_labels") or []:
            if str(source_id):
                key = (branch_id, str(source_id))
                if str(source_id) in by_branch.setdefault(branch_id, {}):
                    ambiguous_bindings.add(key)
                by_branch[branch_id][str(source_id)] = step

    overlays: list[dict[str, Any]] = []
    for raw_attachment in attachments:
        if not isinstance(raw_attachment, Mapping):
            continue
        attachment = dict(raw_attachment)
        if attachment.get("schema_version") != PROGRAM_ATTACHMENT_SCHEMA:
            continue
        program_id = str(attachment.get("program_id") or "")
        if not program_id:
            continue
        replaced_edge_ids = [
            str(value) for value in attachment.get("replaced_edge_ids") or [] if str(value)
        ]
        if len(replaced_edge_ids) < 2:
            continue
        matches = [
            (branch_id, edge_steps)
            for branch_id, edge_steps in by_branch.items()
            if all(edge_id in edge_steps for edge_id in replaced_edge_ids)
            and all(
                (branch_id, edge_id) not in ambiguous_bindings
                for edge_id in replaced_edge_ids
            )
        ]
        if len(matches) != 1:
            continue
        branch_id, edge_steps = matches[0]
        ordered_steps = [edge_steps[edge_id] for edge_id in replaced_edge_ids]
        if any(
            not (
                {str(value) for value in left.get("to_node_ids") or []}
                & {str(value) for value in right.get("from_node_ids") or []}
            )
            for left, right in zip(ordered_steps, ordered_steps[1:])
        ):
            continue
        first = ordered_steps[0]
        last = ordered_steps[-1]
        input_ids = [str(value) for value in first.get("from_node_ids") or [] if str(value)]
        output_ids = [str(value) for value in last.get("to_node_ids") or [] if str(value)]
        if not input_ids or not output_ids:
            continue
        boundary = dict(attachment.get("boundary") or {})
        if not _boundary_matches(
            input_ids,
            expected=dict(boundary.get("precursor") or {}),
            nodes=nodes,
        ) or not _boundary_matches(
            output_ids,
            expected=dict(boundary.get("product") or {}),
            nodes=nodes,
        ):
            continue
        enzyme = dict(attachment.get("enzyme") or {})
        chemical_steps = int(
            attachment.get("chemical_step_equivalent_count") or len(replaced_edge_ids)
        )
        overlays.append(
            {
                "schema_version": PROGRAM_OVERLAY_SCHEMA,
                "program_id": program_id,
                "program_kind": str(
                    attachment.get("program_kind") or "biocatalytic_superstep"
                ),
                "branch_id": branch_id,
                "source_capability_id": str(attachment.get("capability_id") or ""),
                "replaced_edge_ids": replaced_edge_ids,
                "replaced_step_ids": [
                    str(value.get("step_id") or "") for value in ordered_steps
                ],
                "input_molecule_node_ids": input_ids,
                "output_molecule_node_ids": output_ids,
                "input_states": _node_states(input_ids, nodes=nodes),
                "output_states": _node_states(output_ids, nodes=nodes),
                "chemical_step_equivalent_count": chemical_steps,
                "isolated_operation_count": int(
                    attachment.get("physical_step_count") or 1
                ),
                "net_step_savings": int(
                    attachment.get("net_step_savings") or chemical_steps - 1
                ),
                "status": str(attachment.get("authority_scope") or "proposal_only"),
                "validation_status": str(
                    attachment.get("validation_status") or "experiment_required"
                ),
                "warning_codes": [
                    str(value)
                    for value in attachment.get("warning_codes") or []
                    if str(value)
                ],
                "candidate_enzyme_ids": [
                    str(value) for value in enzyme.get("candidate_ids") or [] if str(value)
                ],
                "enzyme_classes": [
                    str(value) for value in enzyme.get("classes") or [] if str(value)
                ],
                "enzyme_ec_numbers": [
                    str(value) for value in enzyme.get("ec_numbers") or [] if str(value)
                ],
                "cofactor_and_carrier_ledger": {
                    "requirements": dict(attachment.get("cofactor_requirements") or {}),
                    "regenerations": dict(attachment.get("cofactor_regenerations") or {}),
                },
                "selectivity_constraints": [
                    str(attachment.get("selectivity_objective") or "")
                ],
                "precedent_refs": [
                    str(value)
                    for value in attachment.get("precedent_refs") or []
                    if str(value)
                ],
                "analogy_only": True,
                "required_assays": [
                    dict(value)
                    for value in attachment.get("required_assays") or []
                    if isinstance(value, Mapping)
                ],
                "validation_gate": {
                    "status": str(
                        attachment.get("validation_status") or "experiment_required"
                    ),
                    "exact_substrate_required": True,
                },
                "fallback_retained": True,
                "eligible_for_route_completion": False,
                "semantics": {
                    "display_only_shadow_layer": True,
                    "not_a_canonical_reaction_edge": True,
                    "does_not_inherit_replaced_edge_proof": True,
                    "canonical_chemical_steps_remain_visible_fallback": True,
                    "cannot_grant_route_completion": True,
                    "attachment_requires_exact_host_binding": True,
                },
            }
        )
    return sorted(overlays, key=lambda row: (row["branch_id"], row["program_id"]))


def apply_program_host_evidence_attachments(
    attachments: Iterable[Mapping[str, Any]],
    *,
    steps: list[dict[str, Any]],
    branches: list[dict[str, Any]],
) -> None:
    """Restore digest-bound host evidence on an exactly matched planned branch.

    This updates only the delivery projection.  Admission-rejected steps remain
    red L0 edges, and the branch stays advisory regardless of literature refs.
    """

    by_branch: dict[str, dict[str, dict[str, Any]]] = {}
    ambiguous_bindings: set[tuple[str, str]] = set()
    for step in steps:
        branch_id = str(step.get("branch_id") or "")
        for source_id in step.get("source_step_labels") or []:
            if branch_id and str(source_id):
                key = (branch_id, str(source_id))
                if str(source_id) in by_branch.setdefault(branch_id, {}):
                    ambiguous_bindings.add(key)
                by_branch[branch_id][str(source_id)] = step
    branches_by_id = {
        str(value.get("branch_id") or ""): value
        for value in branches
        if str(value.get("branch_id") or "")
    }
    for raw_attachment in attachments:
        if not isinstance(raw_attachment, Mapping):
            continue
        attachment = dict(raw_attachment)
        evidence_rows = [
            dict(value)
            for value in attachment.get("host_step_evidence") or []
            if isinstance(value, Mapping) and str(value.get("edge_id") or "")
        ]
        evidence_by_edge = {
            str(value["edge_id"]): value for value in evidence_rows
        }
        if not evidence_by_edge:
            continue
        matches = [
            (branch_id, edge_steps)
            for branch_id, edge_steps in by_branch.items()
            if all(edge_id in edge_steps for edge_id in evidence_by_edge)
            and all(
                (branch_id, edge_id) not in ambiguous_bindings
                for edge_id in evidence_by_edge
            )
        ]
        if len(matches) != 1:
            continue
        branch_id, edge_steps = matches[0]
        branch_refs: set[str] = set()
        for edge_id, evidence in evidence_by_edge.items():
            step = edge_steps[edge_id]
            refs = sorted(
                {str(value) for value in evidence.get("source_refs") or [] if str(value)}
            )
            branch_refs.update(refs)
            warnings = sorted(
                {
                    *(str(value) for value in step.get("validation_findings") or []),
                    *(str(value) for value in evidence.get("warning_codes") or []),
                }
                - {""}
            )
            step["validation_findings"] = warnings
            source_reported = int(evidence.get("proof_level") or 0) >= 1 and bool(refs)
            if source_reported:
                step["source_refs"] = refs
                step["evidence_refs"] = refs
                step["evidence_kinds"] = ["literature_report"]
            # A host-gate failure remains the strongest visual warning even if
            # a source reports the intended transformation.
            if str(step.get("proof_tier") or "") == "L0_rejected":
                if source_reported:
                    step["evidence_label"] = (
                        "Digest-bound literature report retained; host admission rejected"
                    )
                continue
            if not source_reported:
                step["evidence_label"] = "Planner-only bridge; no literature source bound"
                continue
            step["proof_tier"] = "L1_source_reported"
            step["proof_level"] = "L1_source_reported"
            step["evidence_label"] = (
                "Digest-bound literature-reported structure; host reaction validation missing"
            )
            trust = dict(step.get("trust_vector") or {})
            trust["proof_tier"] = "L1_source_reported"
            step["trust_vector"] = trust
            step["visual_encoding"] = {
                "color": "#4f46e5",
                "width": 1.75,
                "opacity": 0.86,
                "dash_pattern": "6 3",
            }
        branch = branches_by_id.get(branch_id)
        if branch is not None:
            branch["source_refs"] = sorted(
                {
                    *(str(value) for value in branch.get("source_refs") or []),
                    *branch_refs,
                }
                - {""}
            )
            branch["host_evidence_projection"] = {
                "schema_version": "route_host_evidence_projection.v1",
                "literature_reported_step_count": sum(
                    int(value.get("proof_level") or 0) >= 1
                    and bool(value.get("source_refs"))
                    for value in evidence_rows
                ),
                "planner_only_step_count": sum(
                    int(value.get("proof_level") or 0) < 1 for value in evidence_rows
                ),
                "does_not_grant_route_completion": True,
            }


def program_overlay_integrity_reasons(
    overlays: Any,
    *,
    nodes: Any,
    steps: Any,
    branches: Any,
    scope: str,
) -> list[str]:
    """Validate overlay boundaries against the delivered canonical fallback."""

    if overlays is None:
        return []
    if not isinstance(overlays, list):
        return [f"{scope}_program_overlays_not_list"]
    node_ids = {
        str(value.get("node_id") or "")
        for value in nodes or []
        if isinstance(value, Mapping)
    }
    step_rows = {
        str(value.get("step_id") or ""): dict(value)
        for value in steps or []
        if isinstance(value, Mapping)
    }
    branch_ids = {
        str(value.get("branch_id") or "")
        for value in branches or []
        if isinstance(value, Mapping)
    }
    reasons: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(overlays):
        prefix = f"{scope}_program_overlay:{index}"
        if not isinstance(raw, Mapping):
            reasons.append(f"{prefix}_not_object")
            continue
        row = dict(raw)
        program_id = str(row.get("program_id") or "")
        branch_id = str(row.get("branch_id") or "")
        replaced = row.get("replaced_step_ids")
        if row.get("schema_version") != PROGRAM_OVERLAY_SCHEMA:
            reasons.append(f"{prefix}_schema_invalid")
        if not program_id or program_id in seen:
            reasons.append(f"{prefix}_program_id_invalid_or_duplicate")
        seen.add(program_id)
        if branch_id not in branch_ids:
            reasons.append(f"{prefix}_branch_unknown")
        if not isinstance(replaced, list) or len(replaced) < 2:
            reasons.append(f"{prefix}_fallback_span_invalid")
            replaced = []
        for step_id in replaced:
            step = step_rows.get(str(step_id))
            if step is None:
                reasons.append(f"{prefix}_fallback_step_unknown:{step_id}")
            elif str(step.get("branch_id") or "") != branch_id:
                reasons.append(f"{prefix}_fallback_step_branch_mismatch:{step_id}")
        for field in ("input_molecule_node_ids", "output_molecule_node_ids"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                reasons.append(f"{prefix}_{field}_invalid")
                continue
            for node_id in values:
                if str(node_id) not in node_ids:
                    reasons.append(f"{prefix}_{field}_unknown:{node_id}")
        if row.get("fallback_retained") is not True:
            reasons.append(f"{prefix}_fallback_not_retained")
        if row.get("eligible_for_route_completion") is not False:
            reasons.append(f"{prefix}_route_completion_authority_invalid")
    return reasons


def _boundary_states(value: Any) -> list[dict[str, str]]:
    return [
        {
            "state_id": str(row.get("state_id") or ""),
            "molecule_id": str(row.get("molecule_id") or ""),
            "canonical_smiles": str(row.get("canonical_smiles") or ""),
        }
        for row in value or []
        if isinstance(row, Mapping) and str(row.get("molecule_id") or "")
    ]


def _boundary_matches(
    node_ids: list[str],
    *,
    expected: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> bool:
    expected_smiles = str(expected.get("smiles") or "")
    return bool(expected_smiles) and any(
        str(dict(nodes.get(node_id) or {}).get("smiles") or "") == expected_smiles
        for node_id in node_ids
    )


def _node_states(
    node_ids: list[str],
    *,
    nodes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "state_id": f"state:{node_id}",
            "molecule_id": node_id,
            "canonical_smiles": str(dict(nodes.get(node_id) or {}).get("smiles") or ""),
        }
        for node_id in node_ids
        if node_id in nodes
    ]


__all__ = [
    "apply_program_host_evidence_attachments",
    "compile_program_overlay_layer",
    "PROGRAM_ATTACHMENT_SCHEMA",
    "PROGRAM_OVERLAY_SCHEMA",
    "program_overlay_integrity_reasons",
    "project_program_overlay_attachments",
    "project_program_overlays",
]
