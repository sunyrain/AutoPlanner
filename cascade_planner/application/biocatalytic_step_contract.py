"""Canonical hypothesis contract for one biocatalytic ReactionJSON step.

The exact molecular boundary belongs to the host-replayed ReactionJSON.  An
LLM may propose an enzyme, whole-cell or hybrid execution hypothesis for that
boundary, but it cannot grant biochemical validation or claim route-level step
savings.  Savings are computed only after a separate TransformationProgram is
bound to an explicit fallback span.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


BIOCATALYTIC_STEP_SCHEMA = "biocatalytic_reaction_step.v1"
BIOCATALYTIC_INTENT_SCHEMA = "biocatalytic_strategy_intent.v1"
BIOLOGICAL_EXECUTION_DOMAINS = frozenset({"enzymatic", "whole_cell", "hybrid"})
BIOCATALYTIC_MODES = frozenset(
    {"enzyme_reaction", "whole_cell_transformation", "chemoenzymatic_cascade"}
)


def normalize_step_execution_domain(
    value: Any,
    *,
    enzyme_label: str = "",
    biocatalytic_step: Mapping[str, Any] | None = None,
) -> str:
    """Return the per-step execution domain, independent of branch policy."""

    explicit = str(value or "").strip().lower().replace("-", "_")
    if explicit in {"chemical", "enzymatic", "whole_cell", "hybrid", "mechanistic"}:
        return explicit
    raw = dict(biocatalytic_step or {})
    mode = str(raw.get("mode") or "").strip().lower()
    if mode == "whole_cell_transformation":
        return "whole_cell"
    if mode == "chemoenzymatic_cascade":
        return "hybrid"
    if mode == "enzyme_reaction" or str(enzyme_label or "").strip():
        return "enzymatic"
    return "chemical"


def normalize_biocatalytic_strategy_intent(
    value: Mapping[str, Any] | None,
    *,
    execution_domain: str,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize the branch-level biological intent without inventing a step."""

    domain = normalize_step_execution_domain(execution_domain)
    raw = dict(value or {})
    if domain not in BIOLOGICAL_EXECUTION_DOMAINS and not raw:
        return {}, []
    mode = _mode(raw.get("mode"), domain)
    intended_equivalent = _bounded_int(
        raw.get("intended_chemical_step_equivalent_count"), 1, 25
    )
    record = {
        "schema_version": BIOCATALYTIC_INTENT_SCHEMA,
        "mode": mode,
        "enzyme_classes": _strings(raw.get("enzyme_classes")),
        "ec_numbers": _strings(raw.get("ec_numbers")),
        "candidate_ids": _strings(raw.get("candidate_ids")),
        "whole_cell_hosts": _strings(raw.get("whole_cell_hosts")),
        "selectivity_objective": str(raw.get("selectivity_objective") or "").strip(),
        "substrate_scope_basis": str(raw.get("substrate_scope_basis") or "").strip(),
        "cofactor_assessment": _cofactor_assessment(raw.get("cofactor_assessment")),
        "intended_chemical_step_equivalent_count": intended_equivalent,
        "fallback_policy": str(raw.get("fallback_policy") or "").strip(),
        "validation_plan": _strings(raw.get("validation_plan"), ordered=True),
        "authority_scope": "strategy_hypothesis_only",
        "semantics": {
            "does_not_create_a_reaction_step": True,
            "intended_step_equivalence_is_not_verified_savings": True,
            "exact_boundary_is_bound_later_by_host_reactionjson": True,
        },
    }
    reasons = _intent_reasons(record, domain=domain)
    record["design_complete"] = not reasons
    record["design_deficits"] = reasons
    record["content_sha256"] = _digest(record)
    return record, reasons


def normalize_biocatalytic_step(
    value: Mapping[str, Any] | None,
    *,
    execution_domain: str,
    product_smiles: str,
    precursor_smiles: Iterable[str],
    enzyme_label: str = "",
    step_id: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Bind a biological execution hypothesis to one exact host-owned edge.

    Design deficits remain explicit annotations and never invalidate structural
    ReactionJSON replay.  The specialized validation gate is deliberately
    closed here because model output cannot validate its own enzyme claim.
    """

    domain = normalize_step_execution_domain(
        execution_domain,
        enzyme_label=enzyme_label,
        biocatalytic_step=value,
    )
    raw = dict(value or {})
    if raw.get("schema_version") == BIOCATALYTIC_STEP_SCHEMA:
        catalyst = dict(raw.get("catalyst_hypothesis") or {})
        ledger = dict(raw.get("cofactor_ledger") or {})
        raw = {
            "mode": raw.get("mode"),
            "enzyme_label": catalyst.get("enzyme_label"),
            "enzyme_classes": catalyst.get("enzyme_classes"),
            "ec_numbers": catalyst.get("ec_numbers"),
            "candidate_ids": catalyst.get("candidate_ids"),
            "sequence_refs": catalyst.get("sequence_refs"),
            "whole_cell_hosts": catalyst.get("whole_cell_hosts"),
            "selectivity_objective": raw.get("selectivity_objective"),
            "substrate_scope_basis": raw.get("substrate_scope_basis"),
            "cofactor_assessment": ledger.get("assessment"),
            "cofactor_requirements": ledger.get("requirements"),
            "cofactor_regenerations": ledger.get("regenerations"),
            "cosubstrates": ledger.get("cosubstrates"),
            "precedent_refs": raw.get("precedent_refs"),
            "validation_plan": raw.get("validation_plan"),
        }
    if domain not in BIOLOGICAL_EXECUTION_DOMAINS and not raw:
        return {}, []
    mode = _mode(raw.get("mode"), domain)
    enzyme_classes = _strings(raw.get("enzyme_classes"))
    label = str(enzyme_label or raw.get("enzyme_label") or "").strip()
    if label and label not in enzyme_classes:
        enzyme_classes.append(label)
    boundary_inputs = list(
        dict.fromkeys(str(value).strip() for value in precursor_smiles if str(value).strip())
    )
    record = {
        "schema_version": BIOCATALYTIC_STEP_SCHEMA,
        "step_id": str(step_id or ""),
        "execution_domain": domain,
        "mode": mode,
        "boundary": {
            "forward_input_smiles": boundary_inputs,
            "forward_output_smiles": str(product_smiles or "").strip(),
            "authority": "host_replayed_reactionjson",
        },
        "catalyst_hypothesis": {
            "enzyme_label": label,
            "enzyme_classes": sorted(set(enzyme_classes)),
            "ec_numbers": _strings(raw.get("ec_numbers")),
            "candidate_ids": _strings(raw.get("candidate_ids")),
            "sequence_refs": _strings(raw.get("sequence_refs")),
            "whole_cell_hosts": _strings(raw.get("whole_cell_hosts")),
        },
        "selectivity_objective": str(raw.get("selectivity_objective") or "").strip(),
        "substrate_scope_basis": str(raw.get("substrate_scope_basis") or "").strip(),
        "cofactor_ledger": {
            "assessment": _cofactor_assessment(raw.get("cofactor_assessment")),
            "requirements": _strings(raw.get("cofactor_requirements"), ordered=True),
            "regenerations": _strings(raw.get("cofactor_regenerations"), ordered=True),
            "cosubstrates": _strings(raw.get("cosubstrates"), ordered=True),
        },
        "precedent_refs": _strings(raw.get("precedent_refs")),
        "validation_plan": _strings(raw.get("validation_plan"), ordered=True),
        "validation_gate": {
            "accepted": False,
            "required_validation_kind": "exact_substrate_biocatalysis",
            "accepted_validation_ids": [],
            "authority": "specialized_biocatalysis_validation_only",
        },
        "step_accounting": {
            "physical_operation_count": 1,
            "chemical_step_equivalent_count": None,
            "net_step_savings": None,
            "requires_explicit_fallback_span_binding": True,
        },
        "authority_scope": "model_proposed_execution_hypothesis",
        "semantics": {
            "not_reaction_proof": True,
            "not_enzyme_availability_proof": True,
            "does_not_inherit_reactionjson_structural_acceptance": True,
            "route_step_savings_are_computed_only_by_program_span_substitution": True,
        },
    }
    reasons = _step_reasons(record)
    record["design_complete"] = not reasons
    record["design_deficits"] = reasons
    record["content_sha256"] = _digest(record)
    return record, reasons


def biocatalytic_step_proof_gate(
    steps: Iterable[Mapping[str, Any]],
    reaction_proofs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require an exact contract-digest validation for every biological edge.

    A generic ``enzyme`` label or a validation for another substrate boundary
    is intentionally insufficient.
    """

    rows = [dict(value) for value in steps if isinstance(value, Mapping) and value]
    required_digests = {
        str(row.get("content_sha256") or "") for row in rows if row.get("content_sha256")
    }
    if not rows:
        return {
            "required": False,
            "accepted": True,
            "reasons": [],
            "required_step_contract_sha256": [],
            "validated_step_contract_sha256": [],
        }
    validated: set[str] = set()
    for proof in reaction_proofs:
        if not isinstance(proof, Mapping) or proof.get("accepted") is not True:
            continue
        validation = proof.get("biocatalysis_validation")
        if not isinstance(validation, Mapping) or validation.get("accepted") is not True:
            continue
        digest = str(validation.get("step_contract_sha256") or "")
        if digest in required_digests:
            validated.add(digest)
    missing = sorted(required_digests - validated)
    return {
        "required": True,
        "accepted": bool(required_digests) and not missing,
        "reasons": (
            []
            if required_digests and not missing
            else ["exact_biocatalytic_step_validation_missing"]
        ),
        "required_step_contract_sha256": sorted(required_digests),
        "validated_step_contract_sha256": sorted(validated),
        "semantics": {
            "generic_enzyme_labels_do_not_validate_a_step": True,
            "validation_is_bound_to_exact_host_reaction_boundary": True,
        },
    }


def _intent_reasons(record: Mapping[str, Any], *, domain: str) -> list[str]:
    reasons: list[str] = []
    if domain not in BIOLOGICAL_EXECUTION_DOMAINS:
        reasons.append("biocatalytic_intent_on_nonbiological_strategy")
    if str(record.get("mode") or "") not in BIOCATALYTIC_MODES:
        reasons.append("biocatalytic_intent_mode_invalid")
    has_enzyme = bool(
        record.get("enzyme_classes")
        or record.get("ec_numbers")
        or record.get("candidate_ids")
    )
    if record.get("mode") == "whole_cell_transformation":
        if not record.get("whole_cell_hosts"):
            reasons.append("biocatalytic_intent_whole_cell_host_missing")
    elif not has_enzyme:
        reasons.append("biocatalytic_intent_enzyme_hypothesis_missing")
    if not str(record.get("selectivity_objective") or ""):
        reasons.append("biocatalytic_intent_selectivity_objective_missing")
    if not str(record.get("substrate_scope_basis") or ""):
        reasons.append("biocatalytic_intent_substrate_scope_basis_missing")
    if record.get("cofactor_assessment") == "unresolved":
        reasons.append("biocatalytic_intent_cofactor_assessment_unresolved")
    if not str(record.get("fallback_policy") or ""):
        reasons.append("biocatalytic_intent_fallback_policy_missing")
    if not record.get("validation_plan"):
        reasons.append("biocatalytic_intent_validation_plan_missing")
    if (
        record.get("mode") == "chemoenzymatic_cascade"
        and int(record.get("intended_chemical_step_equivalent_count") or 0) < 2
    ):
        reasons.append("biocatalytic_intent_cascade_equivalence_too_small")
    return sorted(set(reasons))


def _step_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(record.get("execution_domain") or "") not in BIOLOGICAL_EXECUTION_DOMAINS:
        reasons.append("biocatalytic_step_on_nonbiological_execution_domain")
    boundary = dict(record.get("boundary") or {})
    catalyst = dict(record.get("catalyst_hypothesis") or {})
    if not boundary.get("forward_output_smiles") or not boundary.get(
        "forward_input_smiles"
    ):
        reasons.append("biocatalytic_step_exact_boundary_missing")
    has_enzyme = bool(
        catalyst.get("enzyme_classes")
        or catalyst.get("ec_numbers")
        or catalyst.get("candidate_ids")
    )
    if record.get("mode") == "whole_cell_transformation":
        if not catalyst.get("whole_cell_hosts"):
            reasons.append("biocatalytic_step_whole_cell_host_missing")
    elif not has_enzyme:
        reasons.append("biocatalytic_step_enzyme_hypothesis_missing")
    if not str(record.get("selectivity_objective") or ""):
        reasons.append("biocatalytic_step_selectivity_objective_missing")
    if not str(record.get("substrate_scope_basis") or ""):
        reasons.append("biocatalytic_step_substrate_scope_basis_missing")
    ledger = dict(record.get("cofactor_ledger") or {})
    if ledger.get("assessment") == "unresolved":
        reasons.append("biocatalytic_step_cofactor_assessment_unresolved")
    if not record.get("validation_plan"):
        reasons.append("biocatalytic_step_validation_plan_missing")
    return sorted(set(reasons))


def _mode(value: Any, domain: str) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in BIOCATALYTIC_MODES:
        return mode
    if domain == "whole_cell":
        return "whole_cell_transformation"
    if domain == "hybrid":
        return "chemoenzymatic_cascade"
    return "enzyme_reaction"


def _cofactor_assessment(value: Any) -> str:
    text = str(value or "unresolved").strip().lower().replace("-", "_")
    return text if text in {"required", "not_required", "unresolved"} else "unresolved"


def _strings(value: Any, *, ordered: bool = False) -> list[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or []) if isinstance(value, Iterable) else []
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned if ordered else sorted(cleaned)


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return minimum


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "BIOCATALYTIC_INTENT_SCHEMA",
    "BIOCATALYTIC_MODES",
    "BIOCATALYTIC_STEP_SCHEMA",
    "BIOLOGICAL_EXECUTION_DOMAINS",
    "normalize_biocatalytic_step",
    "normalize_biocatalytic_strategy_intent",
    "normalize_step_execution_domain",
    "biocatalytic_step_proof_gate",
]
