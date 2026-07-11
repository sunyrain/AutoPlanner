"""Compact, digest-bound delivery renderer for explored route forests."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cascade_planner.harness.route_forest_layout import (
    build_branch_lane_projection,
    build_dependency_layout_projection,
    canonical_sha256,
)


DELIVERY_SCHEMA_VERSION = "route_forest_delivery.v1"
_ASSET_ROOT = Path(__file__).with_name("route_forest_ui")
_DEPENDENCY_GRAPH_SCHEMA_VERSION = "molecule_reaction_dependency_graph.v1"
_TEMPLATE_PLACEHOLDERS = ("__TITLE__", "__STYLES__", "__DATA__", "__SCRIPT__")
_TEMPLATE_PLACEHOLDER_PATTERN = re.compile(
    "|".join(re.escape(value) for value in _TEMPLATE_PLACEHOLDERS)
)
_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
_MAX_STRUCTURE_SVG_CHARS = 2_000_000
_REVALIDATED_REPLACEMENT_FLAGS = (
    "accepted",
    "connectivity_revalidated",
    "stock_closure_revalidated",
    "reaction_proof_revalidated",
)

_SAFE_SVG_ELEMENTS = {
    "circle",
    "clipPath",
    "defs",
    "desc",
    "ellipse",
    "g",
    "line",
    "linearGradient",
    "mask",
    "path",
    "polygon",
    "polyline",
    "radialGradient",
    "rect",
    "stop",
    "svg",
    "text",
    "title",
    "tspan",
}
_SAFE_SVG_ATTRIBUTES = {
    "baseProfile",
    "class",
    "clip-path",
    "clip-rule",
    "cx",
    "cy",
    "d",
    "dx",
    "dy",
    "fill",
    "fill-opacity",
    "fill-rule",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "gradientTransform",
    "gradientUnits",
    "height",
    "mask",
    "offset",
    "opacity",
    "pathLength",
    "points",
    "preserveAspectRatio",
    "r",
    "rx",
    "ry",
    "spreadMethod",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "style",
    "text-anchor",
    "transform",
    "version",
    "viewBox",
    "width",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}
_SAFE_SVG_STYLE_PROPERTIES = {
    "clip-rule",
    "fill",
    "fill-opacity",
    "fill-rule",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "opacity",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "text-anchor",
}
_SAFE_SVG_PRESENTATION_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9#.,%+ _-]{1,512}$")
_SAFE_SVG_PRESENTATION_ATTRIBUTES = _SAFE_SVG_STYLE_PROPERTIES | {
    "clip-path",
    "mask",
}
_UNSAFE_SVG_VALUE_PATTERN = re.compile(
    r"(?:javascript\s*:|data\s*:|vbscript\s*:|url\s*\(|expression\s*\(|@import)",
    re.IGNORECASE,
)


def build_route_forest_delivery_payload(forest: Mapping[str, Any]) -> dict[str, Any]:
    """Create the self-contained UI projection without duplicating audit data.

    The complete forest remains authoritative and is bound by
    ``source_forest_sha256``.  Pairwise interface diagnostics can be quadratic
    and are not used to authorize replacements, so only their summary is sent
    to the browser.  Authoritative replacement records and semantics remain
    exact.
    """

    if not isinstance(forest, Mapping):
        raise ValueError("invalid_route_forest_source:not_a_mapping")
    source = dict(forest)
    source_reasons = _route_forest_source_integrity_reasons(source)
    if source_reasons:
        raise ValueError("invalid_route_forest_source:" + ";".join(source_reasons))

    graph = _compact_dependency_graph(source.get("dependency_graph") or {})
    branches = _copy_list(source.get("branches"))
    steps = _copy_list(source.get("steps"))
    nodes = _sanitized_nodes(source.get("nodes"))
    replacement_validation = _compact_replacement_validation(
        source.get("replacement_validation") or {}
    )
    layout = build_dependency_layout_projection(graph)
    lanes = build_branch_lane_projection(branches, graph, steps)

    payload: dict[str, Any] = {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "source_schema_version": str(source.get("schema_version") or ""),
        "source_forest_sha256": canonical_sha256(source),
        "case_id": str(source.get("case_id") or ""),
        "target": _copy_mapping(source.get("target")),
        "counts": _copy_mapping(source.get("counts")),
        "primary_branch_id": str(source.get("primary_branch_id") or ""),
        "primary_selection": _copy_mapping(source.get("primary_selection")),
        "source_revision_context": _copy_mapping(source.get("artifact_revision")),
        "projection_coverage": _copy_mapping(source.get("projection_coverage")),
        "nodes": nodes,
        "steps": steps,
        "branches": branches,
        "modules": _copy_list(source.get("modules")),
        "relationships": _copy_list(source.get("relationships")),
        "dependency_graph": graph,
        "dependency_layout": layout,
        "branch_lanes": lanes,
        "replacement_validation": replacement_validation,
        "route_portfolio_projection": _copy_mapping(
            source.get("route_portfolio_projection")
        ),
        "route_consensus": _copy_mapping(source.get("route_consensus")),
        "route_consensus_graph": _copy_mapping(source.get("route_consensus_graph")),
        "evidence_index": _copy_mapping(source.get("evidence_index")),
        "run_trace": _copy_mapping(source.get("run_trace")),
        "design_notes": _copy_list(source.get("design_notes")),
        "delivery_semantics": {
            "authority": "read_only_projection_bound_to_complete_forest",
            "source_digest": "canonical_sorted_json_sha256",
            "replacement_records": "exact_backend_and_or_validation_records",
            "interface_diagnostics": "summary_only_non_authoritative",
            "dependency_edges": "explicit_source_and_target_ids_only",
            "dependency_trust": (
                "reaction_step_id_joins_top_level_steps_for_trust_and_visual_encoding"
            ),
            "array_adjacency": "never_creates_an_edge",
            "source_revision_context": (
                "source_context_only_never_self_authenticates_delivery"
            ),
            "current_closeout_authority": "external_validated_manifest_only",
            "embedded_json_digest": (
                "sha256_utf8_of_safe_sorted_compact_json_without_embedded_json_sha256"
            ),
            "embedded_json_escaping": "less_than_u2028_u2029_as_json_unicode_escape",
        },
    }
    payload["delivery_sha256"] = canonical_sha256(payload)
    payload["embedded_json_sha256"] = _sha256_text(_serialize_delivery_json(payload))
    return payload


def route_forest_delivery_integrity_reasons(
    payload: Mapping[str, Any],
    *,
    source_forest: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate a compact payload and its optional authoritative source."""

    if not isinstance(payload, Mapping):
        return ["invalid_route_forest_delivery_payload"]

    reasons: list[str] = []
    try:
        _serialize_delivery_json(payload)
    except (TypeError, ValueError):
        reasons.append("route_forest_delivery_not_json_serializable")
    reasons.extend(_route_forest_delivery_shape_reasons(payload))
    reasons.extend(
        _route_relational_integrity_reasons(
            nodes=payload.get("nodes"),
            steps=payload.get("steps"),
            branches=payload.get("branches"),
            primary_branch_id=payload.get("primary_branch_id"),
            primary_selection=payload.get("primary_selection"),
            scope="route_forest_delivery",
        )
    )
    reasons.extend(
        _replacement_validation_integrity_reasons(
            payload.get("replacement_validation"),
            steps=payload.get("steps"),
            branches=payload.get("branches"),
            scope="route_forest_delivery",
        )
    )
    if str(payload.get("schema_version") or "") != DELIVERY_SCHEMA_VERSION:
        reasons.append("invalid_route_forest_delivery_schema")

    reasons.extend(
        _dependency_graph_integrity_reasons(
            payload.get("dependency_graph"),
            nodes=payload.get("nodes"),
            steps=payload.get("steps"),
            branches=payload.get("branches"),
            scope="route_forest_delivery",
        )
    )
    reasons.extend(_delivery_node_integrity_reasons(payload.get("nodes")))

    expected_delivery = str(payload.get("delivery_sha256") or "")
    digest_payload = dict(payload)
    digest_payload.pop("delivery_sha256", None)
    digest_payload.pop("embedded_json_sha256", None)
    delivery_mismatch = not _HEX_SHA256_PATTERN.fullmatch(expected_delivery)
    try:
        delivery_mismatch = delivery_mismatch or expected_delivery != canonical_sha256(
            digest_payload
        )
    except (TypeError, ValueError):
        delivery_mismatch = True
    if delivery_mismatch:
        reasons.append("route_forest_delivery_sha256_mismatch")

    expected_embedded = str(payload.get("embedded_json_sha256") or "")
    embedded_payload = dict(payload)
    embedded_payload.pop("embedded_json_sha256", None)
    embedded_mismatch = not _HEX_SHA256_PATTERN.fullmatch(expected_embedded)
    try:
        embedded_mismatch = embedded_mismatch or expected_embedded != _sha256_text(
            _serialize_delivery_json(embedded_payload)
        )
    except (TypeError, ValueError):
        embedded_mismatch = True
    # A semantic delivery mismatch necessarily changes the embedded digest too;
    # report the independent byte-level failure only when it adds information.
    if embedded_mismatch and not delivery_mismatch:
        reasons.append("route_forest_embedded_json_sha256_mismatch")

    if source_forest is not None:
        if not isinstance(source_forest, Mapping):
            reasons.append("invalid_route_forest_delivery_source")
            return _dedupe_reasons(reasons)
        source_reasons = _route_forest_source_integrity_reasons(source_forest)
        reasons.extend(source_reasons)
        expected_source = str(payload.get("source_forest_sha256") or "")
        source_mismatch = not _HEX_SHA256_PATTERN.fullmatch(expected_source)
        try:
            source_mismatch = source_mismatch or expected_source != canonical_sha256(
                source_forest
            )
        except (TypeError, ValueError):
            source_mismatch = True
        if source_mismatch:
            reasons.append("route_forest_delivery_source_sha256_mismatch")
        source_schema_mismatch = str(payload.get("source_schema_version") or "") != str(
            source_forest.get("schema_version") or ""
        )
        if source_schema_mismatch:
            reasons.append("route_forest_delivery_source_schema_mismatch")
        if (
            not source_reasons
            and not source_mismatch
            and not source_schema_mismatch
            and not delivery_mismatch
            and not embedded_mismatch
        ):
            try:
                expected_payload = build_route_forest_delivery_payload(source_forest)
                projection_matches = _serialize_delivery_json(
                    payload
                ) == _serialize_delivery_json(expected_payload)
            except (TypeError, ValueError):
                projection_matches = False
            if not projection_matches:
                reasons.append("route_forest_delivery_projection_mismatch")
    return _dedupe_reasons(reasons)


def render_route_forest_html(
    forest: Mapping[str, Any],
    *,
    template: str | None = None,
    styles: str | None = None,
    script: str | None = None,
) -> str:
    """Render an offline route workbench from repository-native assets."""

    payload = build_route_forest_delivery_payload(forest)
    target = payload.get("target") or {}
    title = html.escape(
        str(target.get("name") or payload.get("case_id") or "Route forest"),
        quote=True,
    )
    template_text = template if template is not None else _read_asset("template.html")
    styles_text = styles if styles is not None else _read_asset("styles.css")
    script_text = script if script is not None else _read_asset("script.js")
    _validate_template_placeholders(template_text)
    replacements = {
        "__TITLE__": title,
        "__STYLES__": styles_text,
        "__DATA__": _serialize_delivery_json(payload),
        "__SCRIPT__": script_text,
    }
    # Match placeholders only in the original template. Replacement text is
    # returned by the callback and is never scanned a second time.
    return _TEMPLATE_PLACEHOLDER_PATTERN.sub(
        lambda match: replacements[match.group(0)],
        template_text,
    )


def _route_forest_source_integrity_reasons(
    source: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    try:
        _serialize_delivery_json(source)
    except (TypeError, ValueError):
        reasons.append("route_forest_source_not_json_serializable")
    reasons.extend(
        _field_shape_reasons(
            source,
            mapping_fields=(
                "artifact_revision",
                "counts",
                "dependency_graph",
                "evidence_index",
                "primary_selection",
                "projection_coverage",
                "replacement_validation",
                "route_consensus",
                "route_consensus_graph",
                "route_portfolio_projection",
                "run_trace",
                "target",
            ),
            list_fields=(
                "branches",
                "design_notes",
                "modules",
                "nodes",
                "relationships",
                "steps",
            ),
            string_fields=("case_id", "primary_branch_id", "schema_version"),
            scope="route_forest_source",
            required=False,
        )
    )

    if str(source.get("schema_version") or "") != "explored_route_forest.v1":
        reasons.append("invalid_route_forest_source_schema")

    reasons.extend(
        _route_relational_integrity_reasons(
            nodes=source.get("nodes"),
            steps=source.get("steps"),
            branches=source.get("branches"),
            primary_branch_id=source.get("primary_branch_id"),
            primary_selection=source.get("primary_selection"),
            scope="route_forest_source",
        )
    )
    reasons.extend(
        _replacement_validation_integrity_reasons(
            source.get("replacement_validation"),
            steps=source.get("steps"),
            branches=source.get("branches"),
            scope="route_forest_source",
        )
    )

    reasons.extend(
        _dependency_graph_integrity_reasons(
            source.get("dependency_graph"),
            nodes=source.get("nodes"),
            steps=source.get("steps"),
            branches=source.get("branches"),
            scope="route_forest_source",
        )
    )
    return _dedupe_reasons(reasons)


def _route_forest_delivery_shape_reasons(
    payload: Mapping[str, Any],
) -> list[str]:
    reasons = _field_shape_reasons(
        payload,
        mapping_fields=(
            "branch_lanes",
            "counts",
            "delivery_semantics",
            "dependency_graph",
            "dependency_layout",
            "evidence_index",
            "primary_selection",
            "projection_coverage",
            "replacement_validation",
            "route_consensus",
            "route_consensus_graph",
            "route_portfolio_projection",
            "run_trace",
            "source_revision_context",
            "target",
        ),
        list_fields=(
            "branches",
            "design_notes",
            "modules",
            "nodes",
            "relationships",
            "steps",
        ),
        string_fields=(
            "case_id",
            "delivery_sha256",
            "embedded_json_sha256",
            "primary_branch_id",
            "schema_version",
            "source_forest_sha256",
            "source_schema_version",
        ),
        scope="route_forest_delivery",
        required=True,
    )
    source_digest = payload.get("source_forest_sha256")
    if not isinstance(source_digest, str) or not _HEX_SHA256_PATTERN.fullmatch(
        source_digest
    ):
        reasons.append("route_forest_delivery_source_sha256_invalid")
    if str(payload.get("source_schema_version") or "") != "explored_route_forest.v1":
        reasons.append("route_forest_delivery_source_schema_invalid")
    return reasons


def _field_shape_reasons(
    value: Mapping[str, Any],
    *,
    mapping_fields: tuple[str, ...],
    list_fields: tuple[str, ...],
    string_fields: tuple[str, ...],
    scope: str,
    required: bool,
) -> list[str]:
    reasons: list[str] = []
    expected_types: tuple[tuple[tuple[str, ...], type, str], ...] = (
        (mapping_fields, Mapping, "object"),
        (list_fields, list, "array"),
        (string_fields, str, "string"),
    )
    for fields, expected_type, label in expected_types:
        for field in fields:
            if field not in value:
                if required:
                    reasons.append(f"{scope}_{field}_missing")
                continue
            if not isinstance(value.get(field), expected_type):
                reasons.append(f"{scope}_{field}_not_{label}")
    return reasons


def _route_relational_integrity_reasons(
    *,
    nodes: Any,
    steps: Any,
    branches: Any,
    primary_branch_id: Any,
    primary_selection: Any,
    scope: str,
) -> list[str]:
    """Validate the authoritative node, step, and branch foreign keys."""

    node_rows, node_reasons = _identified_rows(
        nodes,
        field="node_id",
        scope=f"{scope}_node",
    )
    step_rows, step_reasons = _identified_rows(
        steps,
        field="step_id",
        scope=f"{scope}_step",
    )
    branch_rows, branch_reasons = _identified_rows(
        branches,
        field="branch_id",
        scope=f"{scope}_branch",
    )
    reasons = [*node_reasons, *step_reasons, *branch_reasons]
    node_ids = set(node_rows)
    branch_ids = set(branch_rows)

    for step_id, step in step_rows.items():
        branch_id = str(step.get("branch_id") or "")
        if not branch_id.strip():
            reasons.append(f"{scope}_step_branch_id_missing:{step_id}")
        elif branch_id not in branch_ids:
            reasons.append(f"{scope}_step_branch_id_unknown:{step_id}:{branch_id}")

        for field in ("from_node_ids", "to_node_ids"):
            endpoint_values = step.get(field)
            if not isinstance(endpoint_values, list):
                reasons.append(f"{scope}_step_{field}_not_array:{step_id}")
                continue
            seen_endpoint_ids: set[str] = set()
            for index, endpoint_value in enumerate(endpoint_values):
                node_id = str(endpoint_value or "")
                if not node_id.strip():
                    reasons.append(
                        f"{scope}_step_{field[:-1]}_missing:{step_id}:{index}"
                    )
                    continue
                if node_id != node_id.strip():
                    reasons.append(
                        f"{scope}_step_{field[:-1]}_not_normalized:{step_id}:{index}"
                    )
                    continue
                if node_id in seen_endpoint_ids:
                    reasons.append(
                        f"{scope}_step_{field[:-1]}_duplicate:{step_id}:{node_id}"
                    )
                    continue
                seen_endpoint_ids.add(node_id)
                if node_id not in node_ids:
                    reasons.append(
                        f"{scope}_step_{field[:-1]}_unknown:{step_id}:{node_id}"
                    )

    memberships_by_step: dict[str, set[str]] = {}
    for branch_id, branch in branch_rows.items():
        branch_step_ids = branch.get("step_ids")
        if not isinstance(branch_step_ids, list):
            reasons.append(f"{scope}_branch_step_ids_not_array:{branch_id}")
            continue
        seen_step_ids: set[str] = set()
        for index, step_id_value in enumerate(branch_step_ids):
            step_id = str(step_id_value or "")
            if not step_id.strip():
                reasons.append(f"{scope}_branch_step_id_missing:{branch_id}:{index}")
                continue
            if step_id != step_id.strip():
                reasons.append(
                    f"{scope}_branch_step_id_not_normalized:{branch_id}:{index}"
                )
                continue
            if step_id in seen_step_ids:
                reasons.append(
                    f"{scope}_branch_step_id_duplicate:{branch_id}:{step_id}"
                )
                continue
            seen_step_ids.add(step_id)
            if step_id not in step_rows:
                reasons.append(f"{scope}_branch_step_id_unknown:{branch_id}:{step_id}")
                continue
            memberships_by_step.setdefault(step_id, set()).add(branch_id)
            step_branch_id = str(step_rows[step_id].get("branch_id") or "")
            if step_branch_id != branch_id:
                reasons.append(
                    f"{scope}_branch_step_owner_mismatch:"
                    f"{branch_id}:{step_id}:{step_branch_id or '<empty>'}"
                )

    for step_id, step in step_rows.items():
        step_branch_id = str(step.get("branch_id") or "")
        if step_branch_id not in branch_ids:
            continue
        memberships = memberships_by_step.get(step_id, set())
        if step_branch_id not in memberships:
            reasons.append(
                f"{scope}_step_branch_membership_missing:{step_id}:{step_branch_id}"
            )

    primary_id = str(primary_branch_id or "")
    if primary_id and primary_id not in branch_ids:
        reasons.append(f"{scope}_primary_branch_id_unknown:{primary_id}")
    if primary_selection is not None and not isinstance(primary_selection, Mapping):
        reasons.append(f"{scope}_primary_selection_not_object")
    elif isinstance(primary_selection, Mapping):
        selected_id = str(primary_selection.get("primary_branch_id") or "")
        if selected_id and selected_id != primary_id:
            reasons.append(
                f"{scope}_primary_selection_branch_id_mismatch:"
                f"{selected_id}:{primary_id or '<empty>'}"
            )
        if str(primary_selection.get("status") or "") == "deterministically_verified":
            selected_branch = branch_rows.get(selected_id or primary_id)
            if selected_branch is None:
                reasons.append(f"{scope}_verified_primary_branch_missing")
            else:
                verified_contract = (
                    str(primary_selection.get("proof_level") or "")
                    == "parent_route_proof"
                    and primary_selection.get("advisory_only") is False
                    and str(selected_branch.get("kind") or "")
                    in {"stitched_verified_route", "direct_verified_route"}
                    and selected_branch.get("solved") is True
                    and selected_branch.get("executable") is True
                    and selected_branch.get("advisory_only") is False
                    and selected_branch.get("not_parent_route_proof") is False
                )
                if not verified_contract:
                    reasons.append(f"{scope}_verified_primary_contract_invalid")
    return _dedupe_reasons(reasons)


def _replacement_validation_integrity_reasons(
    value: Any,
    *,
    steps: Any,
    branches: Any,
    scope: str,
) -> list[str]:
    """Bind replacement claims to complete hidden route projections."""

    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{scope}_replacement_validation_not_object"]
    records = value.get("records")
    if records is None:
        return []
    if not isinstance(records, list):
        return [f"{scope}_replacement_validation_records_not_array"]

    step_rows, _ = _identified_rows(
        steps,
        field="step_id",
        scope=f"{scope}_step",
    )
    branch_rows, _ = _identified_rows(
        branches,
        field="branch_id",
        scope=f"{scope}_branch",
    )
    reasons: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            reasons.append(f"{scope}_replacement_record_not_object:{index}")
            continue
        record_id = str(
            record.get("replacement_id") or record.get("candidate_id") or index
        )
        base_step_id = str(record.get("base_step_id") or "")
        base_branch_id = str(record.get("base_branch_id") or "")
        if not base_step_id.strip():
            reasons.append(f"{scope}_replacement_base_step_id_missing:{record_id}")
        elif base_step_id not in step_rows:
            reasons.append(
                f"{scope}_replacement_base_step_id_unknown:{record_id}:{base_step_id}"
            )
        if not base_branch_id.strip():
            reasons.append(f"{scope}_replacement_base_branch_id_missing:{record_id}")
        elif base_branch_id not in branch_rows:
            reasons.append(
                f"{scope}_replacement_base_branch_id_unknown:"
                f"{record_id}:{base_branch_id}"
            )
        if base_step_id in step_rows and base_branch_id in branch_rows:
            base_branch_step_ids = branch_rows[base_branch_id].get("step_ids")
            if (
                str(step_rows[base_step_id].get("branch_id") or "") != base_branch_id
                or not isinstance(base_branch_step_ids, list)
                or base_step_id
                not in {str(value or "") for value in base_branch_step_ids}
            ):
                reasons.append(
                    f"{scope}_replacement_base_ownership_mismatch:"
                    f"{record_id}:{base_step_id}:{base_branch_id}"
                )

        if record.get("validated") is True:
            candidate_step_id = str(record.get("candidate_step_id") or "")
            candidate_branch_id = str(record.get("candidate_branch_id") or "")
            revalidated_branch_id = str(record.get("revalidated_route_branch_id") or "")
            if not candidate_step_id.strip():
                reasons.append(
                    f"{scope}_replacement_candidate_step_id_missing:{record_id}"
                )
            elif candidate_step_id not in step_rows:
                reasons.append(
                    f"{scope}_replacement_candidate_step_id_unknown:"
                    f"{record_id}:{candidate_step_id}"
                )
            if not candidate_branch_id.strip():
                reasons.append(
                    f"{scope}_replacement_candidate_branch_id_missing:{record_id}"
                )
            elif candidate_branch_id not in branch_rows:
                reasons.append(
                    f"{scope}_replacement_candidate_branch_id_unknown:"
                    f"{record_id}:{candidate_branch_id}"
                )
            if not revalidated_branch_id.strip():
                reasons.append(
                    f"{scope}_replacement_revalidated_branch_id_missing:{record_id}"
                )
            elif revalidated_branch_id not in branch_rows:
                reasons.append(
                    f"{scope}_replacement_revalidated_branch_id_unknown:"
                    f"{record_id}:{revalidated_branch_id}"
                )
            if (
                candidate_branch_id
                and revalidated_branch_id
                and candidate_branch_id != revalidated_branch_id
            ):
                reasons.append(
                    f"{scope}_replacement_candidate_branch_mismatch:"
                    f"{record_id}:{candidate_branch_id}:{revalidated_branch_id}"
                )

            candidate_branch = branch_rows.get(candidate_branch_id)
            if candidate_branch is not None:
                if str(candidate_branch.get("kind") or "") != (
                    "validated_replacement_route"
                ):
                    reasons.append(
                        f"{scope}_replacement_candidate_branch_kind_invalid:"
                        f"{record_id}:{candidate_branch_id}"
                    )
                if candidate_branch.get("listed") is not False:
                    reasons.append(
                        f"{scope}_replacement_candidate_branch_not_hidden:"
                        f"{record_id}:{candidate_branch_id}"
                    )
                if candidate_branch.get("complete") is not True:
                    reasons.append(
                        f"{scope}_replacement_candidate_branch_incomplete:"
                        f"{record_id}:{candidate_branch_id}"
                    )
            if candidate_step_id in step_rows and candidate_branch is not None:
                candidate_branch_step_ids = candidate_branch.get("step_ids")
                if (
                    str(step_rows[candidate_step_id].get("branch_id") or "")
                    != candidate_branch_id
                    or not isinstance(candidate_branch_step_ids, list)
                    or candidate_step_id
                    not in {str(value or "") for value in candidate_branch_step_ids}
                ):
                    reasons.append(
                        f"{scope}_replacement_candidate_ownership_mismatch:"
                        f"{record_id}:{candidate_step_id}:{candidate_branch_id}"
                    )

            if str(record.get("status") or "") != "route_revalidated":
                reasons.append(
                    f"{scope}_replacement_validated_status_invalid:{record_id}"
                )
            for flag in _REVALIDATED_REPLACEMENT_FLAGS:
                if record.get(flag) is not True:
                    reasons.append(
                        f"{scope}_replacement_revalidated_flag_not_true:"
                        f"{record_id}:{flag}"
                    )
            continue

        if record.get("validated") is not False:
            reasons.append(
                f"{scope}_replacement_validated_flag_not_boolean:{record_id}"
            )
        if record.get("accepted") is True:
            reasons.append(f"{scope}_replacement_rejected_record_accepted:{record_id}")
        for field in (
            "candidate_step_id",
            "candidate_branch_id",
            "revalidated_route_branch_id",
        ):
            reference = str(record.get(field) or "")
            if reference:
                reasons.append(
                    f"{scope}_replacement_rejected_preview_reference:"
                    f"{record_id}:{field}:{reference}"
                )
        if str(record.get("status") or "") == "route_revalidated":
            reasons.append(
                f"{scope}_replacement_rejected_status_revalidated:{record_id}"
            )

    return _dedupe_reasons(reasons)


def _dependency_graph_integrity_reasons(
    graph_value: Any,
    *,
    nodes: Any,
    steps: Any,
    branches: Any,
    scope: str,
) -> list[str]:
    route_has_records = any(
        isinstance(value, list) and bool(value) for value in (nodes, steps, branches)
    )
    if graph_value is None:
        return [f"{scope}_dependency_graph_missing"] if route_has_records else []
    if not isinstance(graph_value, Mapping):
        return [f"{scope}_dependency_graph_not_object"]
    graph = dict(graph_value)
    if not graph:
        return [f"{scope}_dependency_graph_empty"] if route_has_records else []
    if not str(graph.get("schema_version") or "") and not any(
        graph.get(key)
        for key in (
            "nodes",
            "molecule_nodes",
            "reaction_nodes",
            "edges",
            "hyperedges",
            "branch_views",
        )
    ):
        return [f"{scope}_dependency_graph_empty"] if route_has_records else []

    reasons: list[str] = []
    if str(graph.get("schema_version") or "") != _DEPENDENCY_GRAPH_SCHEMA_VERSION:
        reasons.append(f"{scope}_dependency_graph_schema_invalid")

    top_node_rows, top_node_reasons = _identified_rows(
        nodes,
        field="node_id",
        scope=f"{scope}_node",
    )
    step_rows, step_reasons = _identified_rows(
        steps,
        field="step_id",
        scope=f"{scope}_step",
    )
    branch_rows, branch_reasons = _identified_rows(
        branches,
        field="branch_id",
        scope=f"{scope}_branch",
    )
    reasons.extend(top_node_reasons)
    reasons.extend(step_reasons)
    reasons.extend(branch_reasons)
    top_node_ids = set(top_node_rows)
    step_ids = set(step_rows)
    branch_ids = set(branch_rows)

    graph_nodes_value = graph.get("nodes")
    using_redundant_nodes = graph_nodes_value is None or (
        isinstance(graph_nodes_value, list)
        and not graph_nodes_value
        and bool(graph.get("molecule_nodes") or graph.get("reaction_nodes"))
    )
    if using_redundant_nodes:
        molecule_nodes = graph.get("molecule_nodes") or []
        reaction_nodes = graph.get("reaction_nodes") or []
        if not isinstance(molecule_nodes, list):
            reasons.append(f"{scope}_dependency_graph_molecule_nodes_not_array")
            molecule_nodes = []
        if not isinstance(reaction_nodes, list):
            reasons.append(f"{scope}_dependency_graph_reaction_nodes_not_array")
            reaction_nodes = []
        graph_nodes_value = [*molecule_nodes, *reaction_nodes]
    elif not isinstance(graph_nodes_value, list):
        reasons.append(f"{scope}_dependency_graph_nodes_not_array")
        graph_nodes_value = []

    graph_nodes, graph_node_reasons = _identified_rows(
        graph_nodes_value,
        field="graph_node_id",
        scope=f"{scope}_dependency_graph_node",
    )
    reasons.extend(graph_node_reasons)
    graph_node_ids = set(graph_nodes)

    for redundant_key in ("molecule_nodes", "reaction_nodes"):
        redundant = graph.get(redundant_key)
        if redundant is None:
            continue
        if not isinstance(redundant, list):
            reasons.append(f"{scope}_dependency_graph_{redundant_key}_not_array")
            continue
        redundant_rows, redundant_reasons = _identified_rows(
            redundant,
            field="graph_node_id",
            scope=f"{scope}_dependency_graph_{redundant_key[:-1]}",
        )
        reasons.extend(redundant_reasons)
        if not using_redundant_nodes:
            for graph_node_id in redundant_rows:
                if graph_node_id not in graph_node_ids:
                    reasons.append(
                        f"{scope}_dependency_graph_{redundant_key[:-1]}_unknown:"
                        f"{graph_node_id}"
                    )

    reaction_graph_nodes_by_step: dict[str, list[str]] = {}
    for graph_node_id, node in graph_nodes.items():
        node_type = str(node.get("node_type") or "")
        if node_type not in {"molecule", "reaction"}:
            reasons.append(
                f"{scope}_dependency_graph_node_type_invalid:"
                f"{graph_node_id}:{node_type or '<empty>'}"
            )
            continue

        if node_type == "molecule":
            molecule_node_id = str(node.get("molecule_node_id") or "")
            if not molecule_node_id.strip():
                reasons.append(
                    f"{scope}_dependency_graph_molecule_node_id_missing:{graph_node_id}"
                )
            elif molecule_node_id not in top_node_ids:
                reasons.append(
                    f"{scope}_dependency_graph_molecule_node_id_unknown:"
                    f"{graph_node_id}:{molecule_node_id}"
                )
            continue

        reaction_step_id = str(node.get("reaction_step_id") or "")
        branch_id = str(node.get("branch_id") or "")
        if not reaction_step_id.strip():
            reasons.append(
                f"{scope}_dependency_graph_reaction_step_id_missing:{graph_node_id}"
            )
        elif reaction_step_id not in step_ids:
            reasons.append(
                f"{scope}_dependency_graph_reaction_step_id_unknown:"
                f"{graph_node_id}:{reaction_step_id}"
            )
        else:
            reaction_graph_nodes_by_step.setdefault(reaction_step_id, []).append(
                graph_node_id
            )

        if not branch_id.strip():
            reasons.append(
                f"{scope}_dependency_graph_reaction_branch_id_missing:{graph_node_id}"
            )
        elif branch_id not in branch_ids:
            reasons.append(
                f"{scope}_dependency_graph_reaction_branch_id_unknown:"
                f"{graph_node_id}:{branch_id}"
            )
        elif reaction_step_id in step_rows:
            step_branch_id = str(step_rows[reaction_step_id].get("branch_id") or "")
            if branch_id != step_branch_id:
                reasons.append(
                    f"{scope}_dependency_graph_reaction_branch_mismatch:"
                    f"{graph_node_id}:{branch_id}:{step_branch_id or '<empty>'}"
                )

    for step_id in step_ids:
        reaction_graph_node_ids = reaction_graph_nodes_by_step.get(step_id, [])
        if not reaction_graph_node_ids:
            reasons.append(f"{scope}_dependency_graph_reaction_node_missing:{step_id}")
        elif len(reaction_graph_node_ids) > 1:
            reasons.append(
                f"{scope}_dependency_graph_reaction_step_id_duplicate:"
                f"{step_id}:{','.join(reaction_graph_node_ids)}"
            )

    edges_value = graph.get("edges")
    if edges_value is None:
        edges_value = []
    elif not isinstance(edges_value, list):
        reasons.append(f"{scope}_dependency_graph_edges_not_array")
        edges_value = []
    edges, edge_reasons = _identified_rows(
        edges_value,
        field="edge_id",
        scope=f"{scope}_dependency_graph_edge",
    )
    reasons.extend(edge_reasons)
    topology_checkable = not edge_reasons
    topology_edges_by_step: dict[str, set[tuple[str, str]]] = {}
    for edge_id, edge in edges.items():
        source_id = str(edge.get("source_graph_node_id") or "")
        target_id = str(edge.get("target_graph_node_id") or "")
        if not source_id.strip():
            reasons.append(f"{scope}_dependency_graph_edge_source_missing:{edge_id}")
            topology_checkable = False
        elif source_id not in graph_node_ids:
            reasons.append(
                f"{scope}_dependency_graph_edge_source_unknown:{edge_id}:{source_id}"
            )
            topology_checkable = False
        if not target_id.strip():
            reasons.append(f"{scope}_dependency_graph_edge_target_missing:{edge_id}")
            topology_checkable = False
        elif target_id not in graph_node_ids:
            reasons.append(
                f"{scope}_dependency_graph_edge_target_unknown:{edge_id}:{target_id}"
            )
            topology_checkable = False

        reaction_step_id = str(edge.get("reaction_step_id") or "")
        branch_id = str(edge.get("branch_id") or "")
        if not reaction_step_id.strip():
            reasons.append(
                f"{scope}_dependency_graph_edge_reaction_step_id_missing:{edge_id}"
            )
            topology_checkable = False
        elif reaction_step_id not in step_ids:
            reasons.append(
                f"{scope}_dependency_graph_edge_reaction_step_id_unknown:"
                f"{edge_id}:{reaction_step_id}"
            )
            topology_checkable = False
        if not branch_id.strip():
            reasons.append(f"{scope}_dependency_graph_edge_branch_id_missing:{edge_id}")
        elif branch_id not in branch_ids:
            reasons.append(
                f"{scope}_dependency_graph_edge_branch_id_unknown:{edge_id}:{branch_id}"
            )
        elif reaction_step_id in step_rows:
            step_branch_id = str(step_rows[reaction_step_id].get("branch_id") or "")
            if branch_id != step_branch_id:
                reasons.append(
                    f"{scope}_dependency_graph_edge_branch_mismatch:"
                    f"{edge_id}:{branch_id}:{step_branch_id or '<empty>'}"
                )

        if source_id not in graph_nodes or target_id not in graph_nodes:
            continue
        source_node = graph_nodes[source_id]
        target_node = graph_nodes[target_id]
        source_type = str(source_node.get("node_type") or "")
        target_type = str(target_node.get("node_type") or "")
        node_type_pair = (source_type, target_type)
        if node_type_pair == ("molecule", "reaction"):
            direction = "input"
            expected_edge_type = "molecule_to_reaction"
            molecule_node = source_node
            reaction_node = target_node
            endpoint_field = "from_node_ids"
        elif node_type_pair == ("reaction", "molecule"):
            direction = "output"
            expected_edge_type = "reaction_to_molecule"
            molecule_node = target_node
            reaction_node = source_node
            endpoint_field = "to_node_ids"
        else:
            reasons.append(
                f"{scope}_dependency_graph_edge_not_bipartite:"
                f"{edge_id}:{source_type or '<empty>'}:{target_type or '<empty>'}"
            )
            topology_checkable = False
            continue

        edge_type = str(edge.get("edge_type") or "")
        if edge_type and edge_type != expected_edge_type:
            reasons.append(
                f"{scope}_dependency_graph_edge_type_mismatch:"
                f"{edge_id}:{edge_type}:{expected_edge_type}"
            )
            topology_checkable = False

        node_reaction_step_id = str(reaction_node.get("reaction_step_id") or "")
        if reaction_step_id in step_rows and node_reaction_step_id != reaction_step_id:
            reasons.append(
                f"{scope}_dependency_graph_edge_reaction_topology_mismatch:"
                f"{edge_id}:{reaction_step_id}:{node_reaction_step_id or '<empty>'}"
            )
            topology_checkable = False
            continue

        molecule_node_id = str(molecule_node.get("molecule_node_id") or "")
        edge_molecule_node_id = str(edge.get("molecule_node_id") or "")
        if edge_molecule_node_id and edge_molecule_node_id != molecule_node_id:
            reasons.append(
                f"{scope}_dependency_graph_edge_molecule_binding_mismatch:"
                f"{edge_id}:{edge_molecule_node_id}:{molecule_node_id or '<empty>'}"
            )
            topology_checkable = False
        if reaction_step_id not in step_rows or not molecule_node_id:
            topology_checkable = False
            continue
        endpoint_values = step_rows[reaction_step_id].get(endpoint_field)
        if not isinstance(endpoint_values, list) or molecule_node_id not in {
            str(value or "") for value in endpoint_values
        }:
            reasons.append(
                f"{scope}_dependency_graph_edge_step_endpoint_mismatch:"
                f"{edge_id}:{reaction_step_id}:{direction}:{molecule_node_id}"
            )
            topology_checkable = False
            continue
        signature = (direction, molecule_node_id)
        step_signatures = topology_edges_by_step.setdefault(
            reaction_step_id,
            set(),
        )
        if signature in step_signatures:
            reasons.append(
                f"{scope}_dependency_graph_edge_topology_duplicate:"
                f"{reaction_step_id}:{direction}:{molecule_node_id}"
            )
            topology_checkable = False
        step_signatures.add(signature)

    if topology_checkable:
        for step_id, step in step_rows.items():
            expected_signatures: set[tuple[str, str]] = set()
            for direction, field in (
                ("input", "from_node_ids"),
                ("output", "to_node_ids"),
            ):
                endpoint_values = step.get(field)
                if isinstance(endpoint_values, list):
                    expected_signatures.update(
                        (direction, str(value or "")) for value in endpoint_values
                    )
            actual_signatures = topology_edges_by_step.get(step_id, set())
            for direction, molecule_node_id in sorted(
                expected_signatures - actual_signatures
            ):
                reasons.append(
                    f"{scope}_dependency_graph_edge_topology_missing:"
                    f"{step_id}:{direction}:{molecule_node_id}"
                )

    branch_views = graph.get("branch_views")
    if not isinstance(branch_views, list):
        reasons.append(f"{scope}_dependency_graph_branch_views_not_array")
        branch_views = []
    seen_view_ids: set[str] = set()
    for index, view in enumerate(branch_views):
        if not isinstance(view, Mapping):
            reasons.append(f"{scope}_dependency_graph_branch_view_not_object:{index}")
            continue
        branch_id = str(view.get("branch_id") or "")
        valid_branch_view = True
        if not branch_id.strip():
            reasons.append(f"{scope}_dependency_graph_branch_view_id_missing:{index}")
            valid_branch_view = False
        elif branch_id in seen_view_ids:
            reasons.append(
                f"{scope}_dependency_graph_branch_view_id_duplicate:{branch_id}"
            )
            valid_branch_view = False
        elif branch_id not in branch_ids:
            reasons.append(
                f"{scope}_dependency_graph_branch_view_id_unknown:{branch_id}"
            )
            valid_branch_view = False
        if branch_id:
            seen_view_ids.add(branch_id)

        view_step_ids_by_key: dict[str, set[str]] = {}
        for key in ("step_ids", "topological_step_ids"):
            values = view.get(key)
            if not isinstance(values, list):
                reasons.append(
                    f"{scope}_dependency_graph_branch_view_{key}_not_array:"
                    f"{branch_id or index}"
                )
                continue
            seen_step_ids: set[str] = set()
            for value in values:
                step_id = str(value or "")
                if not step_id.strip() or step_id not in step_ids:
                    reasons.append(
                        f"{scope}_dependency_graph_branch_view_step_id_unknown:"
                        f"{branch_id or index}:{step_id or '<empty>'}"
                    )
                    continue
                if step_id in seen_step_ids:
                    reasons.append(
                        f"{scope}_dependency_graph_branch_view_step_id_duplicate:"
                        f"{branch_id or index}:{key}:{step_id}"
                    )
                    continue
                seen_step_ids.add(step_id)
                if valid_branch_view:
                    step_branch_id = str(step_rows[step_id].get("branch_id") or "")
                    if step_branch_id != branch_id:
                        reasons.append(
                            f"{scope}_dependency_graph_branch_view_step_owner_mismatch:"
                            f"{branch_id}:{step_id}:{step_branch_id or '<empty>'}"
                        )
            view_step_ids_by_key[key] = seen_step_ids

        if valid_branch_view:
            branch_step_values = branch_rows[branch_id].get("step_ids")
            expected_step_ids = (
                {str(value or "") for value in branch_step_values}
                if isinstance(branch_step_values, list)
                else set()
            )
            for key in ("step_ids", "topological_step_ids"):
                if view_step_ids_by_key.get(key, set()) != expected_step_ids:
                    reasons.append(
                        f"{scope}_dependency_graph_branch_view_{key}_mismatch:"
                        f"{branch_id}"
                    )

    for branch_id in sorted(branch_ids - seen_view_ids):
        reasons.append(f"{scope}_dependency_graph_branch_view_missing:{branch_id}")

    return _dedupe_reasons(reasons)


def _identified_rows(
    value: Any,
    *,
    field: str,
    scope: str,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    if value is None:
        return {}, []
    if not isinstance(value, list):
        return {}, [f"{scope}s_not_array"]
    rows: dict[str, Mapping[str, Any]] = {}
    reasons: list[str] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            reasons.append(f"{scope}_not_object:{index}")
            continue
        identifier = str(row.get(field) or "")
        if not identifier.strip():
            reasons.append(f"{scope}_{field}_missing:{index}")
            continue
        if identifier != identifier.strip():
            reasons.append(f"{scope}_{field}_not_normalized:{index}")
            continue
        if identifier in rows:
            reasons.append(f"{scope}_{field}_duplicate:{identifier}")
            continue
        rows[identifier] = row
    return rows, reasons


def _delivery_node_integrity_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return ["route_forest_delivery_nodes_not_array"]
    reasons: list[str] = []
    for index, node in enumerate(value):
        if not isinstance(node, Mapping):
            reasons.append(f"route_forest_delivery_node_not_object:{index}")
            continue
        structure_svg = node.get("structure_svg")
        if structure_svg is None:
            continue
        if not isinstance(structure_svg, str):
            reasons.append(f"route_forest_delivery_structure_svg_not_string:{index}")
        elif _sanitize_structure_svg(structure_svg) != structure_svg:
            reasons.append(f"route_forest_delivery_structure_svg_unsafe:{index}")
    return reasons


def _sanitized_nodes(value: Any) -> list[Any]:
    nodes = _copy_list(value)
    for node in nodes:
        if not isinstance(node, dict) or "structure_svg" not in node:
            continue
        node["structure_svg"] = _sanitize_structure_svg(node.get("structure_svg"))
    return nodes


def _sanitize_structure_svg(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    source = value.strip()
    if not source or len(source) > _MAX_STRUCTURE_SVG_CHARS:
        return ""
    try:
        root = ElementTree.fromstring(source)
    except (ElementTree.ParseError, ValueError):
        return ""
    namespace, local_name = _split_xml_name(root.tag)
    if local_name != "svg" or namespace not in {"", _SVG_NAMESPACE}:
        return ""
    return _serialize_safe_svg_element(root, root=True)


def _serialize_safe_svg_element(
    element: ElementTree.Element,
    *,
    root: bool = False,
) -> str:
    namespace, tag = _split_xml_name(element.tag)
    if namespace not in {"", _SVG_NAMESPACE} or tag not in _SAFE_SVG_ELEMENTS:
        return ""

    attributes: list[tuple[str, str]] = []
    for raw_name, raw_value in element.attrib.items():
        attribute_namespace, name = _split_xml_name(raw_name)
        if attribute_namespace == _XML_NAMESPACE and name == "space":
            if str(raw_value) in {"default", "preserve"}:
                attributes.append(("xml:space", str(raw_value)))
            continue
        if attribute_namespace or name.lower().startswith("on"):
            continue
        if name not in _SAFE_SVG_ATTRIBUTES:
            continue
        safe_value = _sanitize_svg_attribute_value(name, str(raw_value))
        if safe_value is not None:
            attributes.append((name, safe_value))

    attributes.sort(key=lambda item: item[0])
    rendered_attributes = [
        f'{name}="{html.escape(value, quote=True)}"' for name, value in attributes
    ]
    if root:
        rendered_attributes.insert(0, f'xmlns="{_SVG_NAMESPACE}"')
    opening = f"<{tag}"
    if rendered_attributes:
        opening += " " + " ".join(rendered_attributes)
    opening += ">"

    content: list[str] = []
    if element.text:
        content.append(html.escape(element.text, quote=False))
    for child in element:
        rendered_child = _serialize_safe_svg_element(child)
        if rendered_child:
            content.append(rendered_child)
        if child.tail:
            content.append(html.escape(child.tail, quote=False))
    return opening + "".join(content) + f"</{tag}>"


def _sanitize_svg_attribute_value(name: str, value: str) -> str | None:
    if len(value) > _MAX_STRUCTURE_SVG_CHARS:
        return None
    if name == "style":
        declarations: list[str] = []
        for raw_declaration in value.split(";"):
            if not raw_declaration.strip() or ":" not in raw_declaration:
                continue
            raw_property, raw_property_value = raw_declaration.split(":", 1)
            property_name = raw_property.strip().lower()
            property_value = raw_property_value.strip()
            if (
                property_name not in _SAFE_SVG_STYLE_PROPERTIES
                or not property_value
                or _unsafe_svg_value(property_value)
                or not _SAFE_SVG_PRESENTATION_VALUE_PATTERN.fullmatch(property_value)
            ):
                continue
            declarations.append(f"{property_name}:{property_value}")
        return ";".join(declarations) or None
    if _unsafe_svg_value(value):
        return None
    if name == "class" and not re.fullmatch(r"[A-Za-z0-9 _-]{1,512}", value):
        return None
    if (
        name in _SAFE_SVG_PRESENTATION_ATTRIBUTES
        and not _SAFE_SVG_PRESENTATION_VALUE_PATTERN.fullmatch(value)
    ):
        return None
    return value.strip()


def _unsafe_svg_value(value: str) -> bool:
    if _UNSAFE_SVG_VALUE_PATTERN.search(value):
        return True
    return any(ord(character) < 32 and character not in "\t\n\r" for character in value)


def _split_xml_name(value: Any) -> tuple[str, str]:
    name = str(value or "")
    if name.startswith("{") and "}" in name:
        namespace, local_name = name[1:].split("}", 1)
        return namespace, local_name
    return "", name


def _serialize_delivery_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        serialized.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_template_placeholders(template: str) -> None:
    invalid = [
        f"{placeholder}:{template.count(placeholder)}"
        for placeholder in _TEMPLATE_PLACEHOLDERS
        if template.count(placeholder) != 1
    ]
    if invalid:
        raise ValueError(
            "invalid_route_forest_template_placeholders:" + ";".join(invalid)
        )


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    return list(dict.fromkeys(reasons))


def _compact_dependency_graph(value: Mapping[str, Any]) -> dict[str, Any]:
    graph = dict(value) if isinstance(value, Mapping) else {}
    nodes = _copy_list(graph.get("nodes"))
    if not nodes:
        nodes = [
            *_copy_list(graph.get("molecule_nodes")),
            *_copy_list(graph.get("reaction_nodes")),
        ]
    compact_nodes = []
    for raw_node in nodes:
        node = dict(raw_node)
        node.pop("structure_svg", None)
        if str(node.get("node_type") or "") == "reaction":
            node.pop("trust_vector", None)
            node.pop("visual_encoding", None)
        compact_nodes.append(node)
    compact_edges = []
    for raw_edge in _copy_list(graph.get("edges")):
        edge = dict(raw_edge)
        edge.pop("trust_vector", None)
        edge.pop("visual_encoding", None)
        compact_edges.append(edge)
    retained_keys = (
        "schema_version",
        "graph_kind",
        "direction",
        "acyclic",
        "cycle_graph_node_ids",
        "layout_semantics",
        "no_array_adjacency_edges",
        "proof_tier_legend",
    )
    compact = {key: copy.deepcopy(graph[key]) for key in retained_keys if key in graph}
    compact.update(
        {
            "nodes": compact_nodes,
            "edges": compact_edges,
            "branch_views": _copy_list(graph.get("branch_views")),
        }
    )
    return compact


def _compact_replacement_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    validation = copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    diagnostics = validation.get("interface_diagnostics")
    if isinstance(diagnostics, Mapping):
        summary = {
            key: copy.deepcopy(item)
            for key, item in diagnostics.items()
            if key != "records"
        }
        records = diagnostics.get("records") or []
        summary["source_record_count"] = (
            len(records) if isinstance(records, list) else 0
        )
        summary["records_omitted_from_delivery"] = True
        summary["omission_reason"] = "diagnostics_only_not_replacement_authority"
        validation["interface_diagnostics"] = summary
    return validation


def _copy_mapping(value: Any) -> dict[str, Any]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _copy_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return copy.deepcopy(value)


def _read_asset(name: str) -> str:
    return (_ASSET_ROOT / name).read_text(encoding="utf-8")
