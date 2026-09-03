"""Independent evidence critic versus same-backbone procedure repair.

The evaluator is deliberately provider-neutral.  Model drafts never acquire
source authority; a repair becomes source-exact only after a new, digest-bound
host observation matches the frozen reaction identity.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from cascade_planner.application.campaign_contract_json import bound_row, digest
from cascade_planner.routes.domain import canonicalize_smiles


INDEPENDENT_CRITIC_ABLATION_SCHEMA = "independent_critic_ablation.v1"
BLIND_PROCEDURE_CASE_SCHEMA = "blind_procedure_case.v1"
PROCEDURE_REPAIR_DRAFT_SCHEMA = "procedure_repair_draft.v1"
LEVELS = ("initial_model_draft", "same_backbone_self_critique", "evidence_triggered_repair")
CONDITION_FIELDS = (
    "reagents",
    "catalyst",
    "base",
    "solvent",
    "temperature",
    "time",
    "atmosphere",
    "addition_order",
    "workup",
    "purification",
    "yield_percent",
)


class IndependentCriticAblationError(ValueError):
    """Raised when a frozen input or model draft violates the ablation contract."""


def compile_blind_procedure_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project real procedure cases to source-free, opaque model inputs."""

    _require_digest(config, "procedure_case_config_digest_invalid")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(config.get("cases") or [], start=1):
        case = dict(raw)
        product = canonicalize_smiles(case.get("product_smiles"))
        reactants = sorted(
            canonicalize_smiles(value)
            for value in case.get("reactant_smiles") or []
            if canonicalize_smiles(value)
        )
        if not product or not reactants:
            raise IndependentCriticAblationError("blind_procedure_case_structure_invalid")
        identity = digest({"product_smiles": product, "reactant_smiles": reactants})
        rows.append(
            bound_row(
                {
                    "schema_version": BLIND_PROCEDURE_CASE_SCHEMA,
                    "opaque_case_id": f"procedure-case-{index:03d}-{identity[:10]}",
                    "product_smiles": product,
                    "reactant_smiles": reactants,
                    "initial_conditions": _empty_conditions(),
                    "semantics": {
                        "source_identity_hidden": True,
                        "publication_hidden": True,
                        "target_name_hidden": True,
                        "reference_conditions_hidden": True,
                        "reaction_identity_is_host_canonicalized": True,
                    },
                }
            )
        )
    return rows


def compile_independent_critic_ablation(
    *,
    config: Mapping[str, Any],
    evidence_suite: Mapping[str, Any],
    initial_drafts: Mapping[str, Mapping[str, Any]],
    self_critique_drafts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score model-only repair and evidence-triggered repair on paired real cases."""

    blind_cases = compile_blind_procedure_cases(config)
    _require_digest(evidence_suite, "procedure_evidence_suite_digest_invalid")
    if evidence_suite.get("accepted") is not True:
        raise IndependentCriticAblationError("procedure_evidence_suite_not_accepted")
    config_cases = [dict(value) for value in config.get("cases") or []]
    evidence_cases = {
        str(value.get("case_id") or ""): dict(value)
        for value in evidence_suite.get("cases") or []
        if isinstance(value, Mapping)
    }
    if {str(value.get("case_id") or "") for value in config_cases} != set(evidence_cases):
        raise IndependentCriticAblationError("procedure_evidence_case_set_mismatch")
    expected_opaque = {str(value["opaque_case_id"]) for value in blind_cases}
    if set(initial_drafts) != expected_opaque or set(self_critique_drafts) != expected_opaque:
        raise IndependentCriticAblationError("procedure_model_draft_case_set_mismatch")

    cases: list[dict[str, Any]] = []
    for config_case, blind in zip(config_cases, blind_cases, strict=True):
        case_id = str(config_case.get("case_id") or "")
        opaque_id = str(blind["opaque_case_id"])
        evidence = evidence_cases[case_id]
        truth = _truth(config_case, evidence, blind)
        initial = _score_model_draft(initial_drafts[opaque_id], truth=truth, blind=blind)
        self_critique = _score_model_draft(
            self_critique_drafts[opaque_id], truth=truth, blind=blind
        )
        evidence_repair = _score_evidence_repair(evidence, truth=truth, blind=blind)
        cases.append(
            {
                "case_id": case_id,
                "opaque_case_id": opaque_id,
                "publication": str(config_case.get("publication") or ""),
                "reaction_class": str(config_case.get("reaction_class") or ""),
                "blind_case_sha256": str(blind["content_sha256"]),
                "official_evidence_sha256": str(evidence.get("content_sha256") or ""),
                "initial_model_draft_sha256": bound_row(initial_drafts[opaque_id])[
                    "content_sha256"
                ],
                "same_backbone_self_critique_sha256": bound_row(
                    self_critique_drafts[opaque_id]
                )["content_sha256"],
                "arms": {
                    "initial_model_draft": initial,
                    "same_backbone_self_critique": self_critique,
                    "evidence_triggered_repair": evidence_repair,
                },
                "paired_differences": {
                    "self_critique_minus_initial_oracle_recall": _difference(
                        self_critique, initial, "frozen_oracle_criterion_recall"
                    ),
                    "evidence_minus_self_critique_oracle_recall": _difference(
                        evidence_repair, self_critique, "frozen_oracle_criterion_recall"
                    ),
                    "evidence_minus_self_critique_presence_recall": _difference(
                        evidence_repair, self_critique, "required_field_presence_recall"
                    ),
                },
            }
        )

    summary = {arm: _summarize_arm(cases, arm) for arm in LEVELS}
    evidence_delta = _mean_difference(
        cases,
        right="evidence_triggered_repair",
        left="same_backbone_self_critique",
        metric="frozen_oracle_criterion_recall",
    )
    self_delta = _mean_difference(
        cases,
        right="same_backbone_self_critique",
        left="initial_model_draft",
        metric="frozen_oracle_criterion_recall",
    )
    return bound_row(
        {
            "schema_version": INDEPENDENT_CRITIC_ABLATION_SCHEMA,
            "case_count": len(cases),
            "config_sha256": str(config.get("content_sha256") or ""),
            "evidence_suite_sha256": str(evidence_suite.get("content_sha256") or ""),
            "arms": summary,
            "paired_mean_differences": {
                "same_backbone_minus_initial_oracle_recall": self_delta,
                "evidence_minus_same_backbone_oracle_recall": evidence_delta,
            },
            "independent_evidence_superiority_observed": bool(evidence_delta > 0),
            "cases": cases,
            "semantics": {
                "case_is_the_paired_unit": True,
                "model_drafts_receive_no_reference_conditions_or_source_identity": True,
                "same_backbone_drafts_have_proposal_authority_only": True,
                "new_digest_bound_host_observation_is_required_for_evidence_repair": True,
                "evidence_repair_preserves_canonical_reaction_identity": True,
                "exact_field_match_is_conservative_not_semantic_similarity": True,
                "primary_oracle_criteria_are_frozen_independently_of_extraction_output": True,
                "experiment_success_is_not_assessed": True,
                "results_do_not_establish_route_or_stock_closure": True,
            },
        }
    )


def validate_procedure_repair_draft(value: Mapping[str, Any], *, opaque_case_id: str) -> dict[str, Any]:
    """Fail closed on model drafts that try to claim evidence or experiment authority."""

    row = dict(value)
    artifact = dict(row.get("output_artifact") or row)
    payload = dict(artifact.get("payload") or {})
    reasons: list[str] = []
    if row.get("status") != "accepted_draft":
        reasons.append("worker_draft_not_accepted")
    if artifact.get("case_id") != opaque_case_id:
        reasons.append("worker_draft_case_mismatch")
    if artifact.get("artifact_type") != "ProcedureRepairDraft":
        reasons.append("worker_draft_artifact_type_invalid")
    if payload.get("schema_version") != PROCEDURE_REPAIR_DRAFT_SCHEMA:
        reasons.append("procedure_repair_draft_schema_invalid")
    if payload.get("authority_scope") != "model_predicted_condition":
        reasons.append("model_condition_authority_scope_invalid")
    if payload.get("no_exact_source_authority") is not True:
        reasons.append("model_draft_claims_exact_source_authority")
    if payload.get("no_experimental_validation_claim") is not True:
        reasons.append("model_draft_claims_experimental_validation")
    conditions = payload.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITION_FIELDS):
        reasons.append("model_condition_field_contract_invalid")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "payload": payload if not reasons else {},
    }


def _truth(
    config_case: Mapping[str, Any], evidence: Mapping[str, Any], blind: Mapping[str, Any]
) -> dict[str, Any]:
    _require_digest(evidence, "procedure_evidence_case_digest_invalid")
    if evidence.get("accepted") is not True:
        raise IndependentCriticAblationError("procedure_evidence_case_not_accepted")
    exact = dict(evidence.get("exact_edge") or {})
    product = canonicalize_smiles(exact.get("product_smiles"))
    reactants = sorted(
        canonicalize_smiles(value)
        for value in exact.get("reactant_smiles") or []
        if canonicalize_smiles(value)
    )
    if product != blind.get("product_smiles") or reactants != list(blind.get("reactant_smiles") or []):
        raise IndependentCriticAblationError("procedure_evidence_reaction_identity_mismatch")
    procedure = dict(evidence.get("procedure") or {})
    completeness = dict(procedure.get("condition_completeness") or {})
    if completeness.get("complete") is not True:
        raise IndependentCriticAblationError("procedure_evidence_conditions_incomplete")
    expectations = dict(config_case.get("condition_expectations") or {})
    expected = dict(expectations.get("equals") or {})
    contains = dict(expectations.get("contains") or {})
    conditions = dict(procedure.get("conditions") or {})
    required = tuple(
        str(value)
        for value in dict(config_case.get("condition_expectations") or {}).get("required_fields") or []
    )
    if not required or any(not _present(conditions.get(field)) for field in required):
        raise IndependentCriticAblationError("procedure_truth_required_field_missing")
    return {
        "conditions": conditions,
        "required_fields": required,
        "expected_exact": expected,
        "expected_contains": contains,
        "reaction_class": str(config_case.get("reaction_class") or ""),
        "product_smiles": product,
        "reactant_smiles": reactants,
    }


def _score_model_draft(
    value: Mapping[str, Any], *, truth: Mapping[str, Any], blind: Mapping[str, Any]
) -> dict[str, Any]:
    validation = validate_procedure_repair_draft(
        value, opaque_case_id=str(blind["opaque_case_id"])
    )
    if not validation["accepted"]:
        return _unassessed_score(validation["reasons"])
    payload = dict(validation["payload"])
    result = _score_conditions(
        dict(payload.get("conditions") or {}),
        truth=truth,
        authority_scope="model_predicted_condition",
        source_bound=False,
        repair_triggered=False,
        structure_preserved=True,
        diagnosis_count=len(payload.get("diagnosis") or []),
        risk_flag_count=len(payload.get("risk_flags") or []),
    )
    result["resource_usage"] = _model_resource_usage(value)
    return result


def _score_evidence_repair(
    evidence: Mapping[str, Any], *, truth: Mapping[str, Any], blind: Mapping[str, Any]
) -> dict[str, Any]:
    official_source = dict(evidence.get("official_source") or {})
    exact = dict(evidence.get("exact_edge") or {})
    evidence_sha = str(evidence.get("content_sha256") or "")
    trigger = bool(
        evidence_sha
        and evidence_sha != blind.get("content_sha256")
        and official_source.get("artifact_sha256")
        and exact.get("binding_id")
        and exact.get("procedure_text_sha256")
    )
    if not trigger:
        raise IndependentCriticAblationError("procedure_evidence_material_event_unbound")
    result = _score_conditions(
        dict(dict(evidence.get("procedure") or {}).get("conditions") or {}),
        truth=truth,
        authority_scope="source_exact_reaction_procedure",
        source_bound=True,
        repair_triggered=trigger,
        structure_preserved=True,
        diagnosis_count=1,
        risk_flag_count=0,
    )
    result["resource_usage"] = {
        "model_invocations": int(
            dict(evidence.get("offline_replay") or {}).get("model_invocations") or 0
        ),
        "visual_invocations": int(
            dict(evidence.get("offline_replay") or {}).get("visual_invocations") or 0
        ),
        "input_tokens": 0,
        "output_tokens": 0,
        "elapsed_s": None,
    }
    result["material_event_binding"] = {
        "event": "source_procedure_records_added",
        "official_source_artifact_sha256": str(official_source["artifact_sha256"]),
        "source_binding_id": str(exact["binding_id"]),
        "procedure_text_sha256": str(exact["procedure_text_sha256"]),
    }
    return result


def _score_conditions(
    conditions: Mapping[str, Any],
    *,
    truth: Mapping[str, Any],
    authority_scope: str,
    source_bound: bool,
    repair_triggered: bool,
    structure_preserved: bool,
    diagnosis_count: int,
    risk_flag_count: int,
) -> dict[str, Any]:
    required = tuple(truth["required_fields"])
    true_conditions = dict(truth["conditions"])
    present = [field for field in required if _present(conditions.get(field))]
    exact = [
        field
        for field in required
        if _present(conditions.get(field))
        and _condition_equal(conditions.get(field), true_conditions.get(field))
    ]
    predicted = [field for field in CONDITION_FIELDS if _present(conditions.get(field))]
    unsupported = [
        field
        for field in predicted
        if not _condition_equal(conditions.get(field), true_conditions.get(field))
    ]
    oracle_criteria = [
        *(f"equals:{field}" for field in sorted(dict(truth["expected_exact"]))),
        *(f"contains:{field}" for field in sorted(dict(truth["expected_contains"]))),
    ]
    oracle_matches = [
        criterion
        for criterion in oracle_criteria
        if _oracle_criterion_matches(criterion, conditions=conditions, truth=truth)
    ]
    exact_closed = bool(source_bound and len(exact) == len(required) and structure_preserved)
    return {
        "assessed": True,
        "required_field_count": len(required),
        "present_required_field_count": len(present),
        "exact_required_field_count": len(exact),
        "predicted_nonempty_field_count": len(predicted),
        "unsupported_nonempty_field_count": len(unsupported),
        "required_field_presence_recall": round(len(present) / len(required), 6),
        "exact_field_recall": round(len(exact) / len(required), 6),
        "frozen_oracle_criterion_count": len(oracle_criteria),
        "frozen_oracle_matched_count": len(oracle_matches),
        "frozen_oracle_criterion_recall": round(
            len(oracle_matches) / max(1, len(oracle_criteria)), 6
        ),
        "unsupported_field_rate": round(len(unsupported) / max(1, len(predicted)), 6),
        "condition_complete_by_presence": len(present) == len(required),
        "exact_condition_closed": exact_closed,
        "source_bound": source_bound,
        "repair_triggered_by_new_host_material": repair_triggered,
        "canonical_reaction_identity_preserved": structure_preserved,
        "authority_scope": authority_scope,
        "diagnosis_count": int(diagnosis_count),
        "risk_flag_count": int(risk_flag_count),
        "matched_fields": exact,
        "matched_oracle_criteria": oracle_matches,
        "unsupported_fields": unsupported,
    }


def _summarize_arm(cases: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rows = [dict(dict(case.get("arms") or {}).get(arm) or {}) for case in cases]
    assessed = [row for row in rows if row.get("assessed") is True]
    resources = {
        key: sum(
            float(dict(row.get("resource_usage") or {}).get(key) or 0)
            for row in assessed
        )
        for key in (
            "model_invocations",
            "visual_invocations",
            "input_tokens",
            "output_tokens",
            "elapsed_s",
        )
    }
    return {
        "case_count": len(rows),
        "assessed_count": len(assessed),
        "condition_complete_count": sum(
            row.get("condition_complete_by_presence") is True for row in assessed
        ),
        "exact_condition_closed_count": sum(
            row.get("exact_condition_closed") is True for row in assessed
        ),
        "source_bound_count": sum(row.get("source_bound") is True for row in assessed),
        "mean_required_field_presence_recall": _mean(
            row["required_field_presence_recall"] for row in assessed
        ),
        "mean_exact_field_recall": _mean(row["exact_field_recall"] for row in assessed),
        "mean_frozen_oracle_criterion_recall": _mean(
            row["frozen_oracle_criterion_recall"] for row in assessed
        ),
        "mean_unsupported_field_rate": _mean(
            row["unsupported_field_rate"] for row in assessed
        ),
        "resource_totals": {
            key: int(value) if key != "elapsed_s" else round(value, 3)
            for key, value in resources.items()
        },
    }


def _model_resource_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    usage = dict(value.get("usage") or {})
    status = str(value.get("status") or "")
    return {
        "model_invocations": int(status == "accepted_draft"),
        "visual_invocations": 0,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "elapsed_s": float(value.get("elapsed_s") or 0),
    }


def _mean_difference(
    cases: Sequence[Mapping[str, Any]], *, right: str, left: str, metric: str
) -> float:
    differences: list[float] = []
    for case in cases:
        arms = dict(case.get("arms") or {})
        right_row = dict(arms.get(right) or {})
        left_row = dict(arms.get(left) or {})
        if right_row.get("assessed") is True and left_row.get("assessed") is True:
            differences.append(float(right_row[metric]) - float(left_row[metric]))
    return _mean(differences)


def _difference(right: Mapping[str, Any], left: Mapping[str, Any], metric: str) -> float | None:
    if right.get("assessed") is not True or left.get("assessed") is not True:
        return None
    return round(float(right[metric]) - float(left[metric]), 6)


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return round(sum(rows) / len(rows), 6) if rows else 0.0


def _condition_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(observed) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        observed_values = observed if isinstance(observed, (list, tuple)) else [observed]
        return {_normalize(value) for value in observed_values if _present(value)} == {
            _normalize(value) for value in expected if _present(value)
        }
    return _normalize(observed) == _normalize(expected)


def _oracle_criterion_matches(
    criterion: str, *, conditions: Mapping[str, Any], truth: Mapping[str, Any]
) -> bool:
    kind, field = criterion.split(":", 1)
    observed = conditions.get(field)
    if kind == "equals":
        return _condition_equal(observed, dict(truth["expected_exact"])[field])
    expected_values = dict(truth["expected_contains"])[field]
    observed_values = observed if isinstance(observed, (list, tuple)) else [observed]
    observed_text = " | ".join(_normalize(value) for value in observed_values)
    return all(_normalize(value) in observed_text for value in expected_values)


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("°", "").split())


def _present(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) > 0
    return bool(str(value or "").strip())


def _empty_conditions() -> dict[str, Any]:
    return {
        field: ([] if field in {"reagents", "base", "solvent"} else 0.0 if field == "yield_percent" else "")
        for field in CONDITION_FIELDS
    }


def _unassessed_score(reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "assessed": False,
        "reasons": list(reasons),
        "required_field_presence_recall": None,
        "exact_field_recall": None,
        "frozen_oracle_criterion_recall": None,
        "unsupported_field_rate": None,
        "condition_complete_by_presence": False,
        "exact_condition_closed": False,
        "source_bound": False,
        "repair_triggered_by_new_host_material": False,
        "canonical_reaction_identity_preserved": True,
        "authority_scope": "none",
    }


def _require_digest(value: Mapping[str, Any], reason: str) -> None:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    if not supplied or supplied != digest(row):
        raise IndependentCriticAblationError(reason)


__all__ = [
    "BLIND_PROCEDURE_CASE_SCHEMA",
    "CONDITION_FIELDS",
    "INDEPENDENT_CRITIC_ABLATION_SCHEMA",
    "IndependentCriticAblationError",
    "compile_blind_procedure_cases",
    "compile_independent_critic_ablation",
    "validate_procedure_repair_draft",
]
