from __future__ import annotations

import json

from scripts.legacy.validate_example_runs import ExampleExpectation, validate_example


def test_validate_example_accepts_rendered_diagnostic_run(tmp_path) -> None:
    run_dir = tmp_path / "aspirin_example"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "aspirin_example",
                "target_profile": {
                    "target_name": "aspirin",
                    "target_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                },
                "route_failures": [
                    {
                        "schema_version": "agent_route_failure.v1",
                        "failure_class": "chemenzy_runtime_diagnostic",
                        "reason": "chemenzy_missing_output",
                        "artifact_ref": "guided_chemenzy_result.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    example = ExampleExpectation(
        label="diagnostic",
        run_dir=run_dir,
        required_branch_kinds=("diagnostic_failure",),
        required_text=("chemenzy_missing_output", "aspirin"),
    )

    row = validate_example(example)

    assert row["accepted"], row["reasons"]
    assert row["counts"]["branches"] == 1
    assert (run_dir / "route_forest.html").exists()


def test_validate_example_reports_missing_required_text(tmp_path) -> None:
    run_dir = tmp_path / "minimal_example"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "minimal_example",
                "target_profile": {
                    "target_name": "aspirin",
                    "target_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                },
                "route_failures": [
                    {
                        "schema_version": "agent_route_failure.v1",
                        "reason": "chemenzy_missing_output",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    example = ExampleExpectation(
        label="missing_text",
        run_dir=run_dir,
        required_branch_kinds=("diagnostic_failure",),
        required_text=("not present in forest",),
    )

    row = validate_example(example)

    assert not row["accepted"]
    assert "missing_required_text:not present in forest" in row["reasons"]


def test_validate_example_checks_expected_final_verdict(tmp_path) -> None:
    run_dir = tmp_path / "solved_example"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "solved_example",
                "target_profile": {
                    "target_name": "aspirin",
                    "target_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                },
                "parent_route_proof": {
                    "schema_version": "stitched_parent_route_proof.v1",
                    "proof_mode": "direct_parent_route",
                    "accepted": True,
                    "solved": True,
                    "route_status": "solved",
                    "proof_clauses": {
                        "target_equivalence_passed": True,
                        "parent_route_verifier_accepted": True,
                        "direct_parent_route_verifier_accepted": True,
                        "stock_audit_passed": True,
                        "no_unexplained_large_atom_jump": True,
                        "child_target_route_connected_to_parent_bridge": True,
                        "exact_literature_segment_connected_to_parent_route": True,
                        "analogy_used_only_as_rationale": True,
                    },
                    "source_policy": {
                        "final_verdict_authority": "deterministic_parent_route_proof",
                    },
                    "reasons": [],
                    "route": {
                        "steps": [
                            {
                                "product": "aspirin",
                                "reactants": ["salicylic acid", "acetic anhydride"],
                                "reaction_name": "acetylation",
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_verdict.json").write_text(
        json.dumps({"verdict": "solved", "route_status": "solved", "solved": True}),
        encoding="utf-8",
    )
    example = ExampleExpectation(
        label="solved",
        run_dir=run_dir,
        min_branches=0,
        min_steps=0,
        expected_verdict="solved",
        expected_route_status=("solved",),
        expected_solved=True,
    )

    row = validate_example(example)

    assert row["accepted"], row["reasons"]
    assert row["final_verdict"]["verdict"] == "solved"
    assert row["final_verdict"]["solved"] is True


def test_validate_example_rejects_unresolved_when_hypothesis_verdict_expected(tmp_path) -> None:
    run_dir = tmp_path / "hypothesis_example"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "hypothesis_example",
                "target_profile": {
                    "target_name": "atorvastatin",
                    "target_smiles": "CCO",
                },
                "literature_evidence": {
                    "process_evidence_rows": [
                        {
                            "row_id": "process:atorvastatin",
                            "source_ref": "doi:10.1186/s13065-015-0082-7",
                            "endpoint_labels": ["atorvastatin"],
                            "substrate_or_feedstock_labels": ["advanced ketal ester intermediate 4"],
                            "biocatalyst_or_process_labels": ["Paal-Knorr pyrrole construction"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_verdict.json").write_text(
        json.dumps({"verdict": "unresolved", "route_status": "unresolved", "solved": False}),
        encoding="utf-8",
    )
    example = ExampleExpectation(
        label="hypothesis_expected",
        run_dir=run_dir,
        required_branch_kinds=("process_evidence",),
        required_text=("Paal-Knorr",),
        expected_verdict="hypothesis_route_proposed",
        expected_route_status=("plausible_hypothesis_route",),
        expected_solved=False,
    )

    row = validate_example(example)

    assert not row["accepted"]
    assert "final_verdict_mismatch:unresolved!=hypothesis_route_proposed" in row["reasons"]
    assert "final_route_status_mismatch:unresolved!=plausible_hypothesis_route" in row["reasons"]


def test_validate_example_requires_final_verdict_when_expected(tmp_path) -> None:
    run_dir = tmp_path / "missing_verdict_example"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "missing_verdict_example",
                "target_profile": {
                    "target_name": "aspirin",
                    "target_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                },
                "route_failures": [{"reason": "chemenzy_missing_output"}],
            }
        ),
        encoding="utf-8",
    )
    example = ExampleExpectation(
        label="missing_verdict",
        run_dir=run_dir,
        required_branch_kinds=("diagnostic_failure",),
        expected_verdict="unresolved",
        expected_solved=False,
    )

    row = validate_example(example)

    assert not row["accepted"]
    assert any(reason.startswith("final_verdict_missing:") for reason in row["reasons"])


def test_validate_example_reports_missing_route_forest_html_when_not_rerendering(tmp_path) -> None:
    run_dir = tmp_path / "existing_forest_without_html"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps({"case_id": "existing_forest_without_html"}),
        encoding="utf-8",
    )
    (run_dir / "explored_route_forest.json").write_text(
        json.dumps(
            {
                "schema_version": "explored_route_forest.v1",
                "case_id": "existing_forest_without_html",
                "target": {},
                "counts": {"branches": 0, "steps": 0},
                "branches": [],
                "steps": [],
                "nodes": [],
            }
        ),
        encoding="utf-8",
    )
    example = ExampleExpectation(
        label="missing_html",
        run_dir=run_dir,
        min_branches=0,
        min_steps=0,
    )

    row = validate_example(example, render=False)

    assert not row["accepted"]
    assert any(reason.startswith("route_forest_html_missing:") for reason in row["reasons"])
