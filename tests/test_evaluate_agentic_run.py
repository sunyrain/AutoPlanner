from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.evaluate_agentic_run import compare_reports, evaluate_run, main


def _write(root: Path, relative: str, value: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _minimal_run(tmp_path: Path) -> Path:
    run = tmp_path / "紫杉醇-run"
    run.mkdir()
    _write(
        run,
        "target_input.json",
        {
            "case_id": "paclitaxel",
            "target_name": "紫杉醇",
            "target_smiles": "CCO",
            "family_hint": "taxane semisynthesis",
        },
    )
    _write(
        run,
        "final_verdict.json",
        {
            "verdict": "hypothesis_route_proposed",
            "route_status": "no_route_objective_selected",
            "solved": False,
            "stock_audit_passed": False,
            "reasons": ["parent proof missing"],
        },
    )
    _write(
        run,
        "agent_blackboard.json",
        {
            "case_id": "paclitaxel",
            # Bare booleans must fail the repository's strict predicate.
            "parent_route_proof": {"accepted": True, "solved": True},
            "planner_history": [
                {
                    "round_index": 1,
                    "action_count": 2,
                    "action_types": ["extract_pdf_literature_structures", "search_literature"],
                    "validation_accepted": True,
                    "codex_action_planner": {
                        "attempted": True,
                        "fallback_used": False,
                    },
                },
                {
                    "round_index": 2,
                    "action_count": 1,
                    "action_types": ["run_guided_chemenzy"],
                    "validation_accepted": False,
                    "validation_reasons": ["bad payload"],
                    "codex_action_planner": {
                        "attempted": True,
                        "fallback_used": True,
                        "fallback_reason": "rejected draft",
                    },
                },
            ],
            "action_history": [
                {
                    "action_type": "extract_pdf_literature_structures",
                    "status": "accepted",
                    "useful_artifact": True,
                    "stale": False,
                    "changed_blackboard_fields": ["literature_evidence"],
                },
                {
                    "action_type": "search_literature",
                    "status": "accepted",
                    "useful_artifact": False,
                    "stale": True,
                    "changed_blackboard_fields": [],
                },
            ],
            "literature_evidence": {
                "source_candidates": [
                    {
                        "source_ref": "doi:10.1000/example",
                        "documents": [{"document_id": "article"}, {"document_id": "si"}],
                    }
                ],
                "source_refs": ["doi:10.1000/example"],
                "source_lifecycle": [{}],
                "scout_attempts": [{}],
                "pdf_structure_evidence": [
                    {"accepted": True, "summary": {"rendered_page_count": 3}}
                ],
                "exact_rows": [],
                "resolved_structures": [],
                "visual_chains": [
                    {
                        "accepted": True,
                        "exact_ready": False,
                        "exploratory_accepted": True,
                        "step_count": 2,
                        "reasons": [],
                    },
                    {
                        "accepted": False,
                        "step_count": 0,
                        "reasons": ["visual_input_images_missing"],
                    },
                ],
                "process_evidence_rows": [
                    {
                        "process_type": "small_molecule_process_route",
                        "not_parent_route_proof": True,
                    },
                    {
                        "process_type": "whole_cell_biotransformation",
                        "not_parent_route_proof": True,
                    },
                ],
            },
        },
    )
    _write(
        run,
        "codex_retrosynthesis_team/team_report.json",
        {
            "accepted": False,
            "reasons": ["coordinator_status:rejected_output"],
            "coordinator": {
                "status": "rejected_output",
                "required_child_roles": ["chemist", "critic"],
                "observed_child_agents": [
                    {"role": "chemist", "status": "completed"},
                    {"role": "critic", "status": "completed"},
                ],
                "event_summary": {
                    "child_agent_spawn_count": 2,
                    "child_agent_completed_count": 2,
                },
            },
        },
    )
    _write(
        run,
        "codex_retrosynthesis_team/coordinator_task.json",
        {"allowed_tools": ["spawn_agent", "wait"], "child_roles": ["chemist", "critic"]},
    )
    _write(
        run,
        "codex_retrosynthesis_team/coordinator_run_record.json",
        {
            "status": "rejected_output",
            "tool_calls": [{"tool": "spawn_agent"}, {"tool": "shell"}],
            "output_validation": {"accepted": False, "reasons": ["tool_not_allowed"]},
        },
    )
    _write(
        run,
        "codex_retrosynthesis_team/runtime_summary.json",
        {
            "consistent": True,
            "children": [
                {"role": "chemist", "state": "succeeded"},
                {"role": "critic", "state": "succeeded"},
            ],
        },
    )
    _write(
        run,
        "guided_route_verifier_report.json",
        {
            "accepted": False,
            "route_status": "fake_closed_rejected",
            "target_match": True,
            "route_count": 2,
            "accepted_route_count": 0,
            "rejected_route_count": 2,
            "route_proof_blocked": True,
            "failure_events": [{"reason": "large_atom_jump"}, {"reason": "large_atom_jump"}],
        },
    )
    _write(
        run,
        "agentic_capability_audit.json",
        {
            "validation_status": "rejected",
            "payload": {
                "accepted": False,
                "audit_authority": "diagnostic_only",
                "failed_requirements": ["codex_first_source_acquisition_audited"],
                "requirement_checks": [
                    {"requirement_id": "typed_actions", "accepted": True},
                    {"requirement_id": "source_audit", "accepted": False},
                ],
            },
        },
    )
    _write(
        run,
        "explored_route_forest.json",
        {
            "target": {"name": "紫杉醇", "smiles": "CCO"},
            "primary_branch_id": "branch:consensus",
            "primary_selection": {
                "primary_branch_id": "branch:consensus",
                "status": "advisory",
            },
            "branches": [
                {
                    "branch_id": "branch:consensus",
                    "kind": "route_consensus",
                    "synthesis_class": "semisynthesis",
                    "solved": True,
                    "executable": False,
                    "advisory_only": True,
                    "not_parent_route_proof": True,
                },
                {"branch_id": "branch:process", "kind": "process_evidence"},
            ],
            "nodes": [{"node_id": "target"}],
            "steps": [],
            "route_consensus": {
                "source_schema_version": "route_consensus.v1",
                "available": True,
                "quarantined": False,
            },
        },
    )
    return run


def test_evaluate_run_reports_fail_closed_semantics_and_team_runtime(tmp_path: Path) -> None:
    report = evaluate_run(_minimal_run(tmp_path))

    assert report["schema_version"] == "agentic_run_evaluation.v1"
    assert report["target"]["name"] == "紫杉醇"
    assert report["final_verdict"]["claimed_solved"] is False
    assert report["parent_route_proof"]["strict_evaluation_status"] == "evaluated"
    assert report["parent_route_proof"]["strict_solved"] is False

    team = report["codex_team"]
    assert team["accepted"] is False
    assert team["child_spawn_count"] == 2
    assert team["child_completion_count"] == 2
    assert team["runtime_child_state_counts"] == {"succeeded": 2}
    assert team["tool_calls"]["unauthorized_tools"] == ["shell"]
    assert team["tool_calls"]["has_violation"] is True

    planner = report["planner"]
    assert planner["round_count"] == 2
    assert planner["fallback_rounds"] == [2]
    assert planner["transitions"]["useful_count"] == 1
    assert planner["transitions"]["stale_count"] == 1

    assert report["evidence_counts"]["source_documents"] == 2
    assert report["evidence_counts"]["pdf_rendered_pages"] == 3
    assert report["visual_evidence"]["accepted"] == 1
    assert report["visual_evidence"]["rejected"] == 1
    assert report["process_evidence"]["types"] == {
        "small_molecule_process_route": 1,
        "whole_cell_biotransformation": 1,
    }
    assert report["guided_verifier"]["failure_reason_counts"] == {"large_atom_jump": 2}
    assert report["capability_audit"]["failed_requirement_count"] == 1

    forest = report["route_forest"]
    assert forest["branch_kinds"] == {"process_evidence": 1, "route_consensus": 1}
    assert forest["primary_exists"] is True
    assert forest["branch_semantics"]["advisory_claimed_solved_count"] == 1
    assert forest["branch_semantics"]["strictly_usable_solved_count"] == 0
    assert forest["branch_semantics"]["missing_field_counts"]["solved"] == 1
    assert forest["rejected_team_consensus_quarantine"]["passed"] is False
    assert "advisory_branch_claimed_solved" in report["warnings"]
    assert "rejected_team_consensus_not_quarantined" in report["warnings"]
    assert "route_forest_primary_is_advisory" in report["warnings"]


def test_evaluator_separates_placeholder_queries_from_real_sources(tmp_path: Path) -> None:
    run = _minimal_run(tmp_path)
    blackboard = json.loads((run / "agent_blackboard.json").read_text(encoding="utf-8"))
    blackboard["literature_evidence"]["source_candidates"].append(
        {
            "source_ref": "query:paclitaxel:total_synthesis",
            "source_type": "placeholder_query",
            "access_status": "placeholder_only",
            "placeholder_only": True,
        }
    )
    blackboard["literature_evidence"]["source_refs"].append(
        "query:paclitaxel:total_synthesis"
    )
    _write(run, "agent_blackboard.json", blackboard)

    report = evaluate_run(run)
    counts = report["evidence_counts"]

    assert counts["source_candidate_records"] == 2
    assert counts["source_candidates"] == 2
    assert counts["real_source_candidates"] == 1
    assert counts["placeholder_candidates"] == 1
    assert counts["source_documents"] == 2
    assert counts["source_ref_records"] == 2
    assert counts["source_refs"] == 2
    assert counts["real_source_refs"] == 1
    assert counts["placeholder_source_refs"] == 1


def test_evaluator_uses_strict_locator_validation_and_preserves_v1_totals(
    tmp_path: Path,
) -> None:
    run = _minimal_run(tmp_path)
    blackboard = json.loads((run / "agent_blackboard.json").read_text(encoding="utf-8"))
    compound = (
        "patent_publication:WO2021250648A1;"
        "url:https://patents.google.com/patent/WO2021250648A1/en;lines:2-8"
    )
    blackboard["literature_evidence"]["source_candidates"] = [
        {"doi": "not-a-doi"},
        {"url": "banana"},
        {"source_ref": "arbitrary free string"},
        {"pii": "S0140673610611059"},
        {"source_ref": compound},
        {"local_pdf": "evidence/article.pdf"},
    ]
    blackboard["literature_evidence"]["source_refs"] = [
        "doi:not-a-doi",
        "banana",
        "arbitrary free string",
        "pii:S0140673610611059",
        compound,
        "local_pdf:evidence/article.pdf",
    ]
    _write(run, "agent_blackboard.json", blackboard)

    counts = evaluate_run(run)["evidence_counts"]

    assert counts["source_candidates"] == 6
    assert counts["real_source_candidates"] == 3
    assert counts["placeholder_candidates"] == 3
    assert counts["source_refs"] == 6
    assert counts["real_source_refs"] == 3
    assert counts["placeholder_source_refs"] == 3


def test_evaluator_imports_without_rdkit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['rdkit'] = None; "
                "import scripts.evaluate_agentic_run as module; "
                "from pathlib import Path; "
                "report = module.evaluate_run(Path('missing-run')); "
                "assert report['parent_route_proof']['strict_evaluation_status'] == "
                "'unavailable'; "
                "print(module.SCHEMA_VERSION)"
            ),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "agentic_run_evaluation.v1"


def test_evaluator_direct_script_cli_loads_shared_stdlib_helper(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "evaluate_agentic_run.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Evaluate one or compare two saved agentic retrosynthesis runs" in result.stdout


def test_compare_reports_emits_numeric_and_boolean_changes(tmp_path: Path) -> None:
    baseline = evaluate_run(_minimal_run(tmp_path))
    candidate = json.loads(json.dumps(baseline))
    candidate["run_dir"] = "candidate"
    candidate["codex_team"]["accepted"] = True
    candidate["visual_evidence"]["accepted"] = 3
    candidate["evidence_counts"]["exact_rows"] = 4

    comparison = compare_reports(baseline, candidate)

    assert comparison["schema_version"] == "agentic_run_comparison.v1"
    assert comparison["delta"]["team_accepted"] == {
        "baseline": False,
        "candidate": True,
        "changed": True,
    }
    assert comparison["delta"]["visual_accepted"]["delta"] == 2
    assert comparison["delta"]["evidence_counts"]["exact_rows"]["delta"] == 4


def test_evaluator_excludes_fuzzy_target_identity_shortcuts(tmp_path: Path) -> None:
    run = _minimal_run(tmp_path)
    target_input = json.loads((run / "target_input.json").read_text(encoding="utf-8"))
    target_input["target_name"] = "paclitaxel"
    _write(run, "target_input.json", target_input)
    blackboard = json.loads((run / "agent_blackboard.json").read_text(encoding="utf-8"))
    blackboard.setdefault("target_profile", {})["target_name"] = "paclitaxel"
    blackboard["literature_evidence"]["resolved_structures"] = [
        {
            "accepted": True,
            "label": "paclitaxel derivative 12",
            "smiles": "CCO",
            "target_identity_shortcut": True,
        }
    ]
    _write(run, "agent_blackboard.json", blackboard)

    report = evaluate_run(run)

    assert report["evidence_counts"]["resolved_structures_raw"] == 1
    assert report["evidence_counts"]["resolved_structures"] == 0
    assert report["evidence_counts"]["resolved_structures_invalid_target_shortcuts"] == 1
    assert "invalid_target_identity_shortcut_excluded" in report["warnings"]


def test_missing_run_and_json_output_are_robust(tmp_path: Path) -> None:
    missing = evaluate_run(tmp_path / "does-not-exist")
    assert missing["run_exists"] is False
    assert missing["parent_route_proof"]["strict_solved"] is False
    assert missing["route_forest"]["branch_count"] == 0

    run = _minimal_run(tmp_path)
    output = tmp_path / "reports" / "evaluation.json"
    assert main([str(run), "--output", str(output)]) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["target"]["name"] == "紫杉醇"
    assert saved["route_forest"]["branch_semantics"]["strictly_usable_solved_count"] == 0
