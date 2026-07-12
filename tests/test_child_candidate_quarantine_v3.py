from __future__ import annotations

import json

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.orchestration.codex_retrosynthesis import (
    DEFAULT_CHILD_ROLES,
    RetrosynthesisTeamConfig,
    build_retrosynthesis_coordinator_task,
    run_codex_retrosynthesis_team,
)


def _candidate(
    candidate_id: str,
    *,
    precursor: str,
    no_solved_claim: bool = True,
) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": "CCO",
        "precursor_smiles": [precursor],
        "reaction_family": "carbonyl reduction",
        "transformation_rationale": "one exact precursor set to the current product",
        "source_channel": "chem_enzy",
        "source_refs": [],
        "evidence_refs": [],
        "evidence_level": "validated",
        "confidence": "high",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "required_validation": ["independent_host_reaction_validation"],
        "no_solved_claim": no_solved_claim,
        "not_parent_route_proof": True,
    }


def _report_payload(case_id: str, role: str, candidates: list[dict]) -> dict:
    return {
        "schema_version": "retrosynthesis_proposal_report.v1",
        "case_id": case_id,
        "agent_role": role,
        "target_smiles": "CCO",
        "candidates": candidates,
        "evidence_refs": [],
        "limitations": [],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def _artifact(case_id: str) -> dict:
    coordinator_candidate = _candidate("coordinator", precursor="CC=O")
    coordinator_candidate.update(
        {
            "source_channel": "codex_strategy",
            "evidence_level": "model_only",
            "confidence": "low",
        }
    )
    return {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": f"{case_id}:proposal_report",
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": case_id,
        "source": "codex_cli",
        "input_refs": ["context_snapshot.json"],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "child-agent synthesis",
        "payload": _report_payload(
            case_id,
            "retrosynthesis_coordinator",
            [coordinator_candidate],
        ),
    }


def _runner(
    task,
    *,
    chemo_candidates: list[dict],
) -> WorkerRunRecord:
    children = []
    for index, role in enumerate(task.child_roles):
        if role == "target_structure_strategist":
            candidates = [_candidate("strategy-valid", precursor="CC=O")]
        elif role == "chemoenzymatic_route_specialist":
            candidates = chemo_candidates
        else:
            candidates = []
        children.append(
            {
                "agent_id": f"child-{index}",
                "role": role,
                "role_binding_method": "explicit_spawn_contract",
                "wait_call_id": f"wait-{index}",
                "status": "completed",
                "message": json.dumps(
                    _report_payload(task.case_id, role, candidates),
                    sort_keys=True,
                ),
            }
        )
    return WorkerRunRecord(
        run_id="team:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        backend="codex_cli",
        output_artifact=_artifact(task.case_id),
        output_validation={"accepted": True, "reasons": []},
        metadata={
            "session_id": "thread-v3",
            "event_summary": {"child_agent_spawn_count": len(task.child_roles)},
            "child_agents": children,
        },
    )


def test_mixed_candidate_report_quarantines_only_bad_edge_and_stays_strict(
    tmp_path,
) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(child_acceptance_mode="valid_subset_l0"),
        runner=lambda task: _runner(
            task,
            chemo_candidates=[
                _candidate("chemo-valid", precursor="CC=O"),
                _candidate("chemo-bad-inventory", precursor="C"),
            ],
        ),
    )

    assert report["accepted"], report["reasons"]
    acceptance = report["child_acceptance"]
    assert acceptance["contract_version"].endswith(".v3")
    assert acceptance["acceptance_tier"] == "strict_all"
    assert acceptance["strict_full_child_completion"] is True
    assert acceptance["partial_fallback_used"] is False
    assert acceptance["admitted_candidate_count"] == 2
    assert acceptance["quarantined_candidate_count"] == 1
    assert acceptance["candidate_partition_count"] == 3
    assert acceptance["raw_candidate_partition_reconciled"] is True
    assert acceptance["filtered_child_roles"] == [
        "chemoenzymatic_route_specialist"
    ]

    chemo = next(
        row
        for row in report["child_reports"]
        if row["role"] == "chemoenzymatic_route_specialist"
    )
    assert chemo["accepted"] is True
    assert chemo["report_disposition"] == "accepted_with_candidate_quarantine"
    admission = chemo["candidate_admission"]
    assert admission["raw_candidate_count"] == 2
    assert admission["admitted_candidate_count"] == 1
    assert admission["rejected_candidate_count"] == 1
    rejected_audit = next(row for row in admission["audits"] if not row["accepted"])
    assert rejected_audit["candidate_id"] == "chemo-bad-inventory"
    assert rejected_audit["reasons"] == ["element_inventory_not_conserved"]

    consensus = report["route_consensus"]
    assert consensus["source_summary"]["candidate_count"] == 2
    assert consensus["source_summary"]["rejected_count"] == 1
    assert consensus["rejected_candidates"][0]["candidate_id"] == (
        "chemo-bad-inventory"
    )
    source_records = consensus["proposals"][0]["source_records"]
    assert {row["candidate_id"] for row in source_records} == {
        "strategy-valid",
        "chemo-valid",
    }
    assert all(
        row["authority_evidence_level"] == "model_only"
        and row["authority_confidence"] == "low"
        and row["authority_bound"] is False
        for row in source_records
    )
    assert {row["state"] for row in report["runtime_summary"]["children"]} == {
        "succeeded"
    }
    empty_reports = [
        row
        for row in report["child_reports"]
        if row["role"] in {"literature_route_scout", "route_evidence_critic"}
    ]
    assert all(row["accepted"] is True for row in empty_reports)
    assert all(row["candidate_count"] == 0 for row in empty_reports)


def test_candidate_local_shape_failures_quarantine_without_poisoning_sibling(
    tmp_path,
) -> None:
    missing_product = _candidate("missing-product", precursor="CC=O")
    missing_product.pop("product_smiles")
    extra_field = _candidate("extra-field", precursor="CC=O")
    extra_field["innocuous_note"] = "candidate-local extension"

    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(child_acceptance_mode="strict_all"),
        runner=lambda task: _runner(
            task,
            chemo_candidates=[
                _candidate("valid-sibling", precursor="CC=O"),
                missing_product,
                extra_field,
            ],
        ),
    )

    assert report["accepted"] is True, report["reasons"]
    chemo = next(
        row
        for row in report["child_reports"]
        if row["role"] == "chemoenzymatic_route_specialist"
    )
    assert chemo["accepted"] is True
    assert chemo["report_disposition"] == "accepted_with_candidate_quarantine"
    admission = chemo["candidate_admission"]
    assert admission["raw_candidate_count"] == 3
    assert admission["candidate_pass_count"] == 1
    assert admission["rejected_candidate_count"] == 2
    rejected = {
        row["candidate_id"]: set(row["reasons"])
        for row in admission["audits"]
        if row["accepted"] is not True
    }
    assert {
        "fields_not_exact",
        "product_smiles_not_string",
        "invalid_product_smiles",
    }.issubset(rejected["missing-product"])
    assert rejected["extra-field"] == {"fields_not_exact"}
    acceptance = report["child_acceptance"]
    assert acceptance["raw_candidate_count"] == 4
    assert acceptance["admitted_candidate_count"] == 2
    assert acceptance["quarantined_candidate_count"] == 2
    assert acceptance["discarded_with_rejected_reports_count"] == 0
    assert acceptance["candidate_partition_count"] == 4
    assert acceptance["raw_candidate_partition_reconciled"] is True
    assert report["route_consensus"]["source_summary"]["rejected_count"] == 2


def test_nonempty_all_quarantined_report_fails_but_original_empty_is_valid(
    tmp_path,
) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(child_acceptance_mode="strict_all"),
        runner=lambda task: _runner(
            task,
            chemo_candidates=[_candidate("only-bad", precursor="C")],
        ),
    )

    assert report["accepted"] is False
    chemo = next(
        row
        for row in report["child_reports"]
        if row["role"] == "chemoenzymatic_route_specialist"
    )
    assert chemo["accepted"] is False
    assert "child_report_no_admissible_candidates" in chemo["validation_reasons"]
    assert chemo["candidate_admission"]["candidate_pass_count"] == 0
    assert chemo["candidate_admission"]["rejected_candidate_count"] == 1
    acceptance = report["child_acceptance"]
    assert acceptance["raw_candidate_count"] == 2
    assert acceptance["admitted_candidate_count"] == 1
    assert acceptance["quarantined_candidate_count"] == 0
    assert acceptance["discarded_with_rejected_reports_count"] == 1
    assert acceptance["candidate_partition_count"] == 2
    assert acceptance["raw_candidate_partition_reconciled"] is True
    critic = next(
        row
        for row in report["child_reports"]
        if row["role"] == "route_evidence_critic"
    )
    assert critic["accepted"] is True
    assert critic["report_disposition"] == "accepted_clean"


def test_candidate_safety_claim_violation_poison_rejects_whole_report(
    tmp_path,
) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(child_acceptance_mode="strict_all"),
        runner=lambda task: _runner(
            task,
            chemo_candidates=[
                _candidate("otherwise-valid", precursor="CC=O"),
                _candidate(
                    "unsafe-claim",
                    precursor="CC=O",
                    no_solved_claim=False,
                ),
            ],
        ),
    )

    assert report["accepted"] is False
    chemo = next(
        row
        for row in report["child_reports"]
        if row["role"] == "chemoenzymatic_route_specialist"
    )
    assert chemo["accepted"] is False
    assert chemo["candidate_count"] == 0
    assert "child_candidate:1:missing_no_solved_claim" in chemo[
        "validation_reasons"
    ]
    assert not any(
        source["candidate_id"] == "otherwise-valid"
        for proposal in report["route_consensus"]["proposals"]
        for source in proposal["source_records"]
    )


def test_coordinator_v4_prompt_requires_one_exact_hyperedge(tmp_path) -> None:
    task = build_retrosynthesis_coordinator_task(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        context_ref=str(tmp_path / "context.json"),
        allowed_workdir=tmp_path,
    )

    assert "exactly one retrosynthetic hyperedge" in task.objective
    assert "must jointly form candidate product_smiles" in task.objective
    assert "Do not telescope" in task.objective
    assert list(task.child_roles) == list(DEFAULT_CHILD_ROLES)
