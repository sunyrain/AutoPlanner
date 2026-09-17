"""A sourcing-review request at an exact route frontier, never stock closure.

The Strategy worker may choose to resolve a material instead of inventing its
synthesis. Host binds that request to the selected occurrence and retained
route. Discovery observations justify investigation, not procurement proof.
"""
from __future__ import annotations

from typing import Any, Mapping
from rdkit import Chem

from cascade_planner.application.planning_evidence import canonical_structure


MATERIAL_KINDS = ("defined_compound", "polymer", "mixture", "biological_material")
STRATEGY_FIELDS = ("strategy_query", "critical_assumption", "critic_checkpoint")
MATERIAL_BOUNDARY_GUIDANCE = (
    "At this upstream frontier you may either plan synthesis or request material review. "
    "If a retrieved compound record or source makes this leaf a credible starting-material "
    "candidate, use material_boundary to request identity/sourcing verification before "
    "constructing it. Cite query_key values or source_id values actually returned by "
    "query_planning_evidence; a catalog miss or a name from memory alone is insufficient. "
    "For a material-review request leave strategy_query, critical_assumption and "
    "critic_checkpoint empty. For continued synthesis set material_boundary=null and "
    "fill the ordinary Strategy sentences. Do not replace the exact selected leaf with "
    "a named compound, assign missing stereochemistry, invent a polymer SMILES, or "
    "claim that the route is complete. A different source structure requires identity "
    "or interconversion verification, not automatic equality. This choice retains the "
    "current route prefix for review with unresolved leaves and consumes the ordinary "
    "Strategy budget; it grants no stock or reaction evidence."
)


def material_boundary_schema() -> dict[str, Any]:
    fields = {
        "material_name": {"type": "string", "minLength": 1, "maxLength": 180},
        "material_kind": {"type": "string", "enum": list(MATERIAL_KINDS)},
        "reference_ids": {"type": "array", "minItems": 1, "maxItems": 4,
                          "items": {"type": "string", "minLength": 1, "maxLength": 180}},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 420},
        "unresolved_requirements": {"type": "array", "minItems": 1, "maxItems": 4,
                                    "items": {"type": "string", "minLength": 1, "maxLength": 240}},
    }
    return {"anyOf": [{"type": "null"}, {
        "type": "object", "properties": fields,
        "required": list(fields), "additionalProperties": False,
    }]}


def material_boundary_contract_reasons(card: Mapping[str, Any], *, allowed: bool) -> list[str]:
    request = card.get("material_boundary")
    if request is None:
        return []
    if not allowed:
        return ["material_boundary_not_enabled"]
    if not isinstance(request, Mapping):
        return ["material_boundary_not_object"]
    if any(str(card.get(key) or "").strip() for key in STRATEGY_FIELDS):
        return ["material_boundary_mixed_with_strategy"]
    fields = material_boundary_schema()["anyOf"][1]["properties"]
    if set(request) != set(fields):
        return ["material_boundary_fields_invalid"]
    for key, spec in fields.items():
        value = request[key]
        if spec["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                return ["material_boundary_fields_invalid"]
            if len(value) > spec.get("maxLength", 180) or ("enum" in spec and value not in spec["enum"]):
                return ["material_boundary_fields_invalid"]
        elif (not isinstance(value, list) or not spec["minItems"] <= len(value) <= spec["maxItems"]
              or any(not isinstance(item, str) or not item.strip()
                     or len(item) > spec["items"]["maxLength"] for item in value)):
            return ["material_boundary_fields_invalid"]
    return []


def bind_material_boundary(
    request: Mapping[str, Any], *, references: Mapping[str, Mapping[str, Any]],
    selected_smiles: str, selected_mapped_smiles: str,
    steps: list[dict[str, Any]], task_id: str,
) -> dict[str, Any]:
    """Bind a valid request to real observations and one Host-replayed prefix."""
    reasons = material_boundary_contract_reasons({"material_boundary": request}, allowed=True)
    if reasons:
        raise ValueError(reasons[0])
    if not steps or canonical_structure(selected_mapped_smiles) != canonical_structure(selected_smiles):
        raise ValueError("material_boundary_requires_exact_nonroot_frontier")
    observations = []
    for reference in dict.fromkeys(request["reference_ids"]):
        value = references.get(reference)
        if not value:
            raise ValueError("material_boundary_reference_not_observed")
        # Failed lookups and stock misses are not positive sourcing leads.
        if value.get("status") != "ok":
            raise ValueError("material_boundary_reference_has_no_candidate")
        candidates = value.get("candidates") or []
        sources = value.get("sources") or []
        if not candidates and not sources and not value.get("text"):
            raise ValueError("material_boundary_reference_has_no_candidate")
        identities = []
        for candidate in candidates:
            smiles = str(candidate.get("smiles") or "")
            try:
                exact = canonical_structure(smiles) == canonical_structure(selected_smiles)
            except ValueError:
                exact = False
            identities.append({"cid": candidate.get("cid"), "smiles": smiles,
                               "relation_to_selected_leaf": "exact" if exact else "not_exact"})
        if not sources and value.get("source_id"):
            sources = [{"source_id": value["source_id"]}]
        observations.append({
            "reference_id": reference, "operation": value.get("operation", "search"),
            "identity_candidates": identities,
            "sources": [{k: row.get(k) for k in ("source_id", "title", "doi")}
                        for row in sources],
            "evidence_level": "discovery_only",
        })
    exact_identity = any(candidate["relation_to_selected_leaf"] == "exact"
                         for row in observations for candidate in row["identity_candidates"])
    boundary = {
        "schema_version": "material_boundary_review.v1", "status": "pending_verification",
        "task_id": task_id, "selected_smiles": canonical_structure(selected_smiles),
        "selected_mapped_smiles": selected_mapped_smiles,
        "prefix_step_ids": [str(row.get("step_id") or "") for row in steps],
        **dict(request), "observations": observations,
        "identity_status": "exact_database_record" if exact_identity else "unresolved",
        "availability_status": "unverified",
        "required_checks": list(dict.fromkeys([
            *([] if exact_identity else ["identity_or_explicit_interconversion"]),
            "availability", *([] if request["material_kind"] == "defined_compound" else ["material_specification"]),
            *request["unresolved_requirements"],
        ])),
    }
    if not current_material_boundary({"steps": steps, "material_boundary_review": boundary}):
        raise ValueError("material_boundary_leaf_not_on_retained_frontier")
    return boundary


def current_material_boundary(branch: Mapping[str, Any]) -> dict[str, Any]:
    """Suppress a stale request when repair changes the selected prefix/frontier."""
    boundary = dict(branch.get("material_boundary_review") or {})
    steps = list(branch.get("steps") or [])
    if (not boundary or not steps or boundary.get("prefix_step_ids")
            != [str(row.get("step_id") or "") for row in steps]):
        return {}
    def mapped_identity(value: Any) -> str:
        mol = Chem.MolFromSmiles(str(value or ""))
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""

    selected = mapped_identity(boundary.get("selected_mapped_smiles"))
    products = {mapped_identity(row.get("mapped_product_smiles") or row.get("product_mapped_smiles"))
                for row in steps}
    precursors = {mapped_identity(value) for row in steps for value in row.get("mapped_precursor_smiles") or []}
    if selected not in precursors - products:
        return {}
    return boundary
