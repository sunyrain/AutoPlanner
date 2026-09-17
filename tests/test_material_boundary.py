from copy import deepcopy

import pytest

from cascade_planner.agent.codex_worker import (
    WorkerTask, WorkerProcessResult, run_codex_worker, _worker_model_output_json_schema,
)
from cascade_planner.application.planning_evidence import BoundedPlanningEvidence
from cascade_planner.application.material_boundary import bind_material_boundary, current_material_boundary


def request(reference="source"):
    return {"material_name": "candidate", "material_kind": "defined_compound",
            "reference_ids": [reference], "rationale": "Check a retrieved starting material before assembly",
            "unresolved_requirements": ["supplier availability"]}


def steps(mapped="[CH3:2][OH:3]"):
    return [{"step_id": "step:1", "product_smiles": "CCO", "precursor_smiles": ["C", "CO"],
             "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
             "mapped_precursor_smiles": ["[CH4:1]", mapped]}]


def test_compound_reference_survives_cache_and_resume_without_granting_stock(tmp_path):
    path = tmp_path / "queries.jsonl"
    manager = BoundedPlanningEvidence(journal_path=path, providers={
        "compound": lambda _: {"status": "ok", "candidates": [{"cid": 887, "smiles": "CO"}]},
    })
    observation = manager.query("strategy", {"operation": "compound", "query": "candidate"})
    reference = observation["query_key"]
    assert manager.query("other", {"operation": "compound", "query": "candidate"})["query_key"] == reference
    resumed = BoundedPlanningEvidence(journal_path=path)
    before = resumed.summary()
    boundary = bind_material_boundary(
        request(reference), references=resumed.observed_references([reference]),
        selected_smiles="CO", selected_mapped_smiles="[OH:3][CH3:2]", steps=steps(), task_id="strategy",
    )
    assert resumed.summary() == before
    assert boundary["identity_status"] == "exact_database_record"
    assert boundary["availability_status"] == "unverified"
    assert current_material_boundary({"steps": steps(), "material_boundary_review": boundary}) == boundary
    assert not current_material_boundary({"steps": steps("[CH3:4][OH:5]"), "material_boundary_review": boundary})
    assert not current_material_boundary({"steps": [], "material_boundary_review": boundary})
    assert not any(k in boundary for k in ("stock_closed", "solved", "reaction_validated"))


@pytest.mark.parametrize("reference", [{}, {"source": {"status": "no_hit"}},
    {"source": {"status": "not_in_bound_stock", "operation": "stock", "smiles": "CO"}}])
def test_missing_or_negative_lookup_cannot_pause_synthesis(reference):
    with pytest.raises(ValueError, match="material_boundary_reference"):
        bind_material_boundary(request(), references=reference, selected_smiles="CO",
                               selected_mapped_smiles="[CH3:2][OH:3]", steps=steps(), task_id="strategy")


def test_identity_mismatch_remains_explicit_and_does_not_replace_leaf():
    boundary = bind_material_boundary(
        request(), references={"source": {"status": "ok", "candidates": [{"cid": 1, "smiles": "C=O"}]}},
        selected_smiles="CO", selected_mapped_smiles="[CH3:2][OH:3]", steps=steps(), task_id="strategy",
    )
    assert boundary["identity_status"] == "unresolved"
    assert "identity_or_explicit_interconversion" in boundary["required_checks"]
    assert boundary["selected_smiles"] == "CO"
    wrong_steps = steps()
    wrong_steps[0]["mapped_precursor_smiles"] = ["[CH4:1]", "[CH2:2]=[O:3]"]
    with pytest.raises(ValueError, match="leaf_not_on_retained_frontier"):
        bind_material_boundary(request(), references={"source": {"status": "ok", "sources": [{"source_id": "doi"}]}},
                               selected_smiles="CO", selected_mapped_smiles="[CH3:2][OH:3]",
                               steps=wrong_steps, task_id="strategy")


def test_schema_is_opt_in_and_rejects_mixed_strategy_boundary():
    import json
    task = WorkerTask(task_id="s", case_id="case", task_type="paper_matched_strategy_generator",
                      required_artifact_type="StrategyCardReport")
    assert "material_boundary" not in _worker_model_output_json_schema(task)["properties"]
    enabled = deepcopy(task)
    enabled.host_context["allow_material_boundary"] = True
    assert "material_boundary" in _worker_model_output_json_schema(enabled)["properties"]
    wire = {"strategy_query": "", "critical_assumption": "", "critic_checkpoint": "", "material_boundary": request()}
    def executor(_):
        return WorkerProcessResult(stdout=json.dumps(wire), backend="codex_cli")
    assert run_codex_worker(enabled, runner=executor).status == "accepted_draft"
    assert "material_boundary_not_enabled" in run_codex_worker(task, runner=executor).output_validation["reasons"]
    wire["strategy_query"] = "also build another route"
    assert "material_boundary_mixed_with_strategy" in run_codex_worker(enabled, runner=executor).output_validation["reasons"]
