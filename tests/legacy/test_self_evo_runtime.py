from __future__ import annotations

from cascade_planner.agent.evolution_manager import EvolutionCandidate, LayeredKnowledgeBase
from cascade_planner.legacy.harness_runtime.self_evo_memory import compile_self_evo_memory
from cascade_planner.legacy.harness_runtime.self_evo_replay import run_self_evo_replay_gate


def _staging_report(candidate: EvolutionCandidate) -> dict:
    kb = LayeredKnowledgeBase()
    kb.add_candidate(candidate, target_run=True)
    kb.promote(candidate.candidate_id, from_layer="candidate", to_layer="shadow", target_run=True)
    kb.promote(candidate.candidate_id, from_layer="shadow", to_layer="staging", target_run=True)
    return {"schema_version": "self_evo_staging_compile_report.v1", "kb": kb.to_dict()}


def _template_metrics() -> dict:
    return {
        "true_solved_rate_delta": 0.0,
        "fake_closure_rate_delta": 0.0,
        "condition_quality_delta": 0.0,
        "template_replay_passes": True,
        "structure_validated": True,
        "evidence_source_credible": True,
        "role_assignment_checked": True,
    }


def _template_candidate(*, with_assets: bool = False) -> EvolutionCandidate:
    payload = {
        "schema_version": "template_candidate.v1",
        "not_raw_reaction_injection": True,
    }
    if with_assets:
        payload = {
            "template_id": "statin_side_chain_template",
            "reaction_class": "statin_side_chain_convergence",
            "template_card": {
                "schema_version": "literature_template_card.v1",
                "template_id": "statin_side_chain_template",
                "validation_status": "draft",
                "template_level": "advisory_strategy",
                "reaction_class": "statin_side_chain_convergence",
                "product_retron": {"retron_type": "statin_heptenoate_side_chain"},
                "evidence_refs": ["ev_template"],
                "not_raw_reaction_injection": True,
            },
            "route_expansion_task": {
                "schema_version": "compiled_route_expansion_task.v1",
                "task_id": "statin_expand_1",
                "evidence_refs": ["ev_template"],
                "preferred_reaction_classes": ["statin_side_chain_convergence"],
                "not_raw_reaction_injection": True,
            },
        }
    return EvolutionCandidate(
        candidate_id="template_candidate",
        candidate_type="TemplateCandidate",
        payload=payload,
        evidence_refs=["ev_template"],
        validation_status="validated",
    )


def test_self_evo_replay_blocks_target_run_and_promotes_cross_case_gate() -> None:
    staging = _staging_report(_template_candidate())

    target_report = run_self_evo_replay_gate(
        staging,
        replay_metrics=_template_metrics(),
        target_run=True,
        allow_production=True,
    )
    cross_case_report = run_self_evo_replay_gate(
        staging,
        replay_metrics=_template_metrics(),
        target_run=False,
        allow_production=True,
    )

    assert target_report["accepted"] is True
    assert target_report["production_write_blocked"] is True
    assert target_report["production_promoted_count"] == 0
    assert cross_case_report["accepted"] is True
    assert cross_case_report["production_write_blocked"] is False
    assert cross_case_report["production_promoted_count"] == 1
    assert "template_candidate" in cross_case_report["kb"]["layers"]["production"]


def test_self_evo_memory_compiles_replay_assets_for_future_runs() -> None:
    replay = run_self_evo_replay_gate(
        _staging_report(_template_candidate(with_assets=True)),
        replay_metrics=_template_metrics(),
        target_run=True,
        allow_production=False,
    )

    memory = compile_self_evo_memory(replay, case_id="statin_case")

    assert memory["accepted"] is True, memory["reasons"]
    assert memory["production_write_blocked"] is True
    assert memory["production_promoted_count"] == 0
    assert memory["reusable_template_cards"][0]["template_id"] == "statin_side_chain_template"
    assert memory["reusable_route_expansion_tasks"][0]["task_id"] == "statin_expand_1"
    assert memory["future_use_policy"][
        "not_route_evidence_until_current_target_relation_checked"
    ] is True


def test_self_evo_memory_keeps_executable_template_extraction_tasks() -> None:
    replay = run_self_evo_replay_gate(
        _staging_report(_template_candidate(with_assets=True)),
        replay_metrics=_template_metrics(),
        target_run=True,
        allow_production=False,
    )
    compiled = {
        "schema_version": "compiled_downstream_consumables.v1",
        "literature_template_plugin": {"template_cards": [], "one_step_rows": []},
        "route_expansion": {"tasks": []},
        "executable_template_maturity": {
            "schema_version": "executable_template_maturity.v1",
            "status": "needs_structured_extraction",
            "extraction_tasks": [
                {
                    "schema_version": "compiled_executable_template_extraction_task.v1",
                    "task_id": "extract_statin_side_chain_step",
                    "source_title": "Statin side-chain process",
                    "reaction_class": "statin_side_chain_convergence",
                    "evidence_refs": ["ev_template"],
                    "required_structured_fields": ["product_smiles", "reactant_smiles"],
                    "precursor_roles": ["beta-keto ester side-chain precursor"],
                    "not_raw_reaction_injection": True,
                }
            ],
        },
    }

    memory = compile_self_evo_memory(
        replay,
        compiled_downstream=compiled,
        case_id="statin_case",
    )

    assert memory["accepted"] is True, memory["reasons"]
    assert memory["reusable_executable_template_extraction_tasks"][0]["task_id"] == (
        "extract_statin_side_chain_step"
    )
    assert any(
        row["hint_type"] == "executable_template_extraction_task"
        for row in memory["query_hints"]
    )
    assert "executable_template_extraction_task" in memory["future_use_policy"]["allowed_use"]


def test_self_evo_replay_rejects_bad_replay_metrics() -> None:
    report = run_self_evo_replay_gate(
        _staging_report(_template_candidate()),
        replay_metrics={"fake_closure_rate_delta": 0.1, "template_replay_passes": False},
        target_run=False,
        allow_production=True,
    )

    assert report["accepted"] is False
    assert report["production_write_blocked"] is True
    assert report["production_promoted_count"] == 0
    assert "fake_closure_rate_regressed" in report["reasons"]
    assert "template_replay_failed" in report["reasons"]
