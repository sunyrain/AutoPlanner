from __future__ import annotations

from cascade_planner.application.campaign_contract_json import bound_row
from cascade_planner.eval.independent_critic_ablation import (
    CONDITION_FIELDS,
    compile_blind_procedure_cases,
    compile_independent_critic_ablation,
    validate_procedure_repair_draft,
)


def _config() -> dict:
    return bound_row(
        {
            "schema_version": "real_patent_procedure_gate_cases.v1",
            "cases": [
                {
                    "case_id": "real-case-1",
                    "target_name": "hidden target",
                    "publication": "EP123A1",
                    "reaction_class": "amide_coupling",
                    "product_smiles": "CC(=O)NCC",
                    "reactant_smiles": ["CC(=O)Cl", "CCN"],
                    "condition_expectations": {
                        "required_fields": ["reagents", "solvent", "temperature", "time"],
                        "equals": {"time": "2 h"},
                        "contains": {"solvent": ["tetrahydrofuran"]},
                    },
                }
            ],
        }
    )


def _evidence(config: dict) -> dict:
    blind = compile_blind_procedure_cases(config)[0]
    case = bound_row(
        {
            "case_id": "real-case-1",
            "accepted": True,
            "exact_edge": {
                "product_smiles": blind["product_smiles"],
                "reactant_smiles": blind["reactant_smiles"],
                "binding_id": "det-parser:test-binding",
                "procedure_text_sha256": "a" * 64,
            },
            "official_source": {"artifact_sha256": "b" * 64},
            "offline_replay": {"model_invocations": 0, "visual_invocations": 0},
            "procedure": {
                "condition_completeness": {"complete": True},
                "conditions": {
                    "reagents": ["triethylamine"],
                    "solvent": ["tetrahydrofuran"],
                    "temperature": "0 C",
                    "time": "2 h",
                },
            },
        }
    )
    return bound_row({"accepted": True, "cases": [case]})


def _draft(case_id: str, *, self_critique: bool = False) -> dict:
    conditions = {
        field: ([] if field in {"reagents", "base", "solvent"} else 0.0 if field == "yield_percent" else "")
        for field in CONDITION_FIELDS
    }
    conditions.update(
        {
            "reagents": ["triethylamine"],
            "solvent": ["dichloromethane"],
            "temperature": "ambient" if not self_critique else "0 C",
            "time": "overnight" if not self_critique else "2 h",
        }
    )
    return {
        "status": "accepted_draft",
        "output_artifact": {
            "case_id": case_id,
            "artifact_type": "ProcedureRepairDraft",
            "payload": {
                "schema_version": "procedure_repair_draft.v1",
                "step_id": "S1",
                "reaction_class": "amide coupling",
                "diagnosis": ["conditions absent"],
                "conditions": conditions,
                "missing_information": ["exact source"],
                "risk_flags": ["exotherm"],
                "repair_actions": ["screen base"],
                "authority_scope": "model_predicted_condition",
                "no_exact_source_authority": True,
                "no_experimental_validation_claim": True,
            },
        },
    }


def test_blind_projection_removes_source_and_reference_conditions() -> None:
    row = compile_blind_procedure_cases(_config())[0]

    assert row["opaque_case_id"].startswith("procedure-case-001-")
    assert row["initial_conditions"]["solvent"] == []
    assert "publication" not in row
    assert "target_name" not in row
    assert "condition_expectations" not in row


def test_model_draft_cannot_claim_exact_source_authority() -> None:
    case_id = compile_blind_procedure_cases(_config())[0]["opaque_case_id"]
    draft = _draft(case_id)
    draft["output_artifact"]["payload"]["no_exact_source_authority"] = False

    audit = validate_procedure_repair_draft(draft, opaque_case_id=case_id)

    assert audit["accepted"] is False
    assert "model_draft_claims_exact_source_authority" in audit["reasons"]


def test_real_evidence_repair_outperforms_same_backbone_without_claiming_experiment() -> None:
    config = _config()
    case_id = compile_blind_procedure_cases(config)[0]["opaque_case_id"]

    report = compile_independent_critic_ablation(
        config=config,
        evidence_suite=_evidence(config),
        initial_drafts={case_id: _draft(case_id)},
        self_critique_drafts={case_id: _draft(case_id, self_critique=True)},
    )

    assert report["independent_evidence_superiority_observed"] is True
    assert report["arms"]["same_backbone_self_critique"][
        "mean_frozen_oracle_criterion_recall"
    ] == 0.5
    assert report["arms"]["evidence_triggered_repair"][
        "mean_frozen_oracle_criterion_recall"
    ] == 1.0
    assert report["arms"]["same_backbone_self_critique"]["exact_condition_closed_count"] == 0
    assert report["arms"]["evidence_triggered_repair"]["exact_condition_closed_count"] == 1
    assert report["cases"][0]["arms"]["evidence_triggered_repair"][
        "repair_triggered_by_new_host_material"
    ] is True
    assert report["semantics"]["experiment_success_is_not_assessed"] is True
