"""Project one-hop mechanism hypotheses as route-anchored display callouts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


MECHANISM_HYPOTHESIS_ATTACHMENT_SCHEMA = "route_mechanism_hypothesis_attachment.v1"
MECHANISM_HYPOTHESIS_OVERLAY_SCHEMA = "route_mechanism_hypothesis_overlay.v1"


def project_mechanism_hypothesis_overlays(
    attachments: Iterable[Mapping[str, Any]],
    *,
    steps: Iterable[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind one-hop proposals to a unique source edge and exact product state."""

    by_branch: dict[str, dict[str, dict[str, Any]]] = {}
    ambiguous: set[tuple[str, str]] = set()
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        step = dict(raw_step)
        branch_id = str(step.get("branch_id") or "")
        for raw_source_id in step.get("source_step_labels") or []:
            source_id = str(raw_source_id or "")
            if not branch_id or not source_id:
                continue
            key = (branch_id, source_id)
            if source_id in by_branch.setdefault(branch_id, {}):
                ambiguous.add(key)
            by_branch[branch_id][source_id] = step

    overlays: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_attachment in attachments:
        if not isinstance(raw_attachment, Mapping):
            continue
        attachment = dict(raw_attachment)
        hypothesis_id = str(attachment.get("hypothesis_id") or "")
        anchor_edge_ids = [
            str(value) for value in attachment.get("anchor_edge_ids") or [] if str(value)
        ]
        precursor_smiles = str(attachment.get("precursor_smiles") or "")
        proposed_product = dict(attachment.get("proposed_product") or {})
        product_smiles = str(proposed_product.get("smiles") or "")
        if (
            attachment.get("schema_version") != MECHANISM_HYPOTHESIS_ATTACHMENT_SCHEMA
            or not hypothesis_id
            or hypothesis_id in seen
            or int(attachment.get("proposal_depth") or 0) != 1
            or not anchor_edge_ids
            or not precursor_smiles
            or not product_smiles
            or product_smiles == precursor_smiles
        ):
            continue
        matches = [
            (branch_id, edge_steps)
            for branch_id, edge_steps in by_branch.items()
            if all(edge_id in edge_steps for edge_id in anchor_edge_ids)
            and all((branch_id, edge_id) not in ambiguous for edge_id in anchor_edge_ids)
        ]
        if len(matches) != 1:
            continue
        branch_id, edge_steps = matches[0]
        anchor_steps = [edge_steps[edge_id] for edge_id in anchor_edge_ids]
        anchor_node_ids = {
            str(node_id)
            for step in anchor_steps
            for node_id in step.get("to_node_ids") or []
            if str(node_id)
            and str(dict(nodes.get(str(node_id)) or {}).get("smiles") or "")
            == precursor_smiles
        }
        if len(anchor_node_ids) != 1:
            continue
        anchor_node_id = next(iter(anchor_node_ids))
        rationale = str(attachment.get("mechanistic_rationale") or "")
        checks = [
            str(value) for value in attachment.get("falsifiable_checks") or [] if str(value)
        ]
        source_refs = [
            str(value) for value in attachment.get("anchor_source_refs") or [] if str(value)
        ]
        if not rationale or not checks or not source_refs:
            continue
        overlays.append(
            {
                "schema_version": MECHANISM_HYPOTHESIS_OVERLAY_SCHEMA,
                "hypothesis_id": hypothesis_id,
                "branch_id": branch_id,
                "host_route_id": str(attachment.get("host_route_id") or ""),
                "anchor_edge_ids": anchor_edge_ids,
                "anchor_step_ids": [
                    str(value.get("step_id") or "") for value in anchor_steps
                ],
                "anchor_molecule_node_id": anchor_node_id,
                "precursor_state": {
                    "molecule_id": anchor_node_id,
                    "label": str(dict(nodes.get(anchor_node_id) or {}).get("label") or ""),
                    "canonical_smiles": precursor_smiles,
                },
                "proposed_product": {
                    "label": str(proposed_product.get("label") or "proposed product"),
                    "canonical_smiles": product_smiles,
                },
                "proposal_depth": 1,
                "mechanistic_rationale": rationale,
                "elementary_steps": [
                    str(value)
                    for value in attachment.get("elementary_steps") or []
                    if str(value)
                ],
                "falsifiable_checks": checks,
                "anchor_source_refs": source_refs,
                "priority_score": float(attachment.get("priority_score") or 0.0),
                "authority_scope": str(
                    attachment.get("authority_scope") or "proposal_only"
                ),
                "validation_status": str(
                    attachment.get("validation_status") or "host_materialization_required"
                ),
                "warning_codes": [
                    str(value)
                    for value in attachment.get("warning_codes") or []
                    if str(value)
                ],
                "eligible_for_route_completion": False,
                "semantics": {
                    "display_only_shadow_layer": True,
                    "anchor_evidence_not_promoted": True,
                    "not_a_canonical_reaction_edge": True,
                    "cannot_grant_route_completion": True,
                    "one_hop_only": True,
                },
            }
        )
        seen.add(hypothesis_id)
    return sorted(overlays, key=lambda row: (row["branch_id"], row["hypothesis_id"]))


def mechanism_hypothesis_overlay_integrity_reasons(
    overlays: Any,
    *,
    nodes: Any,
    steps: Any,
    branches: Any,
    scope: str,
) -> list[str]:
    """Validate that callouts remain exact, isolated, one-hop proposals."""

    if overlays is None:
        return []
    if not isinstance(overlays, list):
        return [f"{scope}_mechanism_hypotheses_not_list"]
    node_rows = {
        str(value.get("node_id") or ""): dict(value)
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
        prefix = f"{scope}_mechanism_hypothesis:{index}"
        if not isinstance(raw, Mapping):
            reasons.append(f"{prefix}_not_object")
            continue
        row = dict(raw)
        hypothesis_id = str(row.get("hypothesis_id") or "")
        branch_id = str(row.get("branch_id") or "")
        anchor_node_id = str(row.get("anchor_molecule_node_id") or "")
        anchor_steps = row.get("anchor_step_ids")
        precursor = dict(row.get("precursor_state") or {})
        product = dict(row.get("proposed_product") or {})
        if row.get("schema_version") != MECHANISM_HYPOTHESIS_OVERLAY_SCHEMA:
            reasons.append(f"{prefix}_schema_invalid")
        if not hypothesis_id or hypothesis_id in seen:
            reasons.append(f"{prefix}_hypothesis_id_invalid_or_duplicate")
        seen.add(hypothesis_id)
        if branch_id not in branch_ids:
            reasons.append(f"{prefix}_branch_unknown")
        if anchor_node_id not in node_rows:
            reasons.append(f"{prefix}_anchor_node_unknown")
        if not isinstance(anchor_steps, list) or not anchor_steps:
            reasons.append(f"{prefix}_anchor_steps_invalid")
            anchor_steps = []
        for step_id in anchor_steps:
            step = step_rows.get(str(step_id))
            if step is None:
                reasons.append(f"{prefix}_anchor_step_unknown:{step_id}")
                continue
            if str(step.get("branch_id") or "") != branch_id:
                reasons.append(f"{prefix}_anchor_step_branch_mismatch:{step_id}")
            if anchor_node_id not in {str(value) for value in step.get("to_node_ids") or []}:
                reasons.append(f"{prefix}_anchor_node_not_step_product:{step_id}")
        precursor_smiles = str(precursor.get("canonical_smiles") or "")
        product_smiles = str(product.get("canonical_smiles") or "")
        if (
            str(precursor.get("molecule_id") or "") != anchor_node_id
            or precursor_smiles
            != str(node_rows.get(anchor_node_id, {}).get("smiles") or "")
        ):
            reasons.append(f"{prefix}_precursor_state_mismatch")
        if not product_smiles or product_smiles == precursor_smiles:
            reasons.append(f"{prefix}_proposed_product_invalid")
        if int(row.get("proposal_depth") or 0) != 1:
            reasons.append(f"{prefix}_proposal_depth_invalid")
        if not str(row.get("mechanistic_rationale") or ""):
            reasons.append(f"{prefix}_rationale_missing")
        if not isinstance(row.get("falsifiable_checks"), list) or not row.get(
            "falsifiable_checks"
        ):
            reasons.append(f"{prefix}_falsifiable_checks_missing")
        if not isinstance(row.get("anchor_source_refs"), list) or not row.get(
            "anchor_source_refs"
        ):
            reasons.append(f"{prefix}_anchor_source_refs_missing")
        if row.get("eligible_for_route_completion") is not False:
            reasons.append(f"{prefix}_route_completion_authority_invalid")
        semantics = dict(row.get("semantics") or {})
        if semantics.get("anchor_evidence_not_promoted") is not True:
            reasons.append(f"{prefix}_anchor_evidence_isolation_invalid")
        if semantics.get("not_a_canonical_reaction_edge") is not True:
            reasons.append(f"{prefix}_canonical_edge_authority_invalid")
    return reasons


__all__ = [
    "MECHANISM_HYPOTHESIS_ATTACHMENT_SCHEMA",
    "MECHANISM_HYPOTHESIS_OVERLAY_SCHEMA",
    "mechanism_hypothesis_overlay_integrity_reasons",
    "project_mechanism_hypothesis_overlays",
]
