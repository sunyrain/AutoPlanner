from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade_planner.harness.route_forest_delivery import (
    build_route_forest_delivery_payload,
    route_forest_delivery_integrity_reasons,
)
from cascade_planner.harness.v4_route_workbench import (
    compile_v4_route_forest,
    render_v4_route_workbench_html,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway, CampaignGatewayError
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.web.workspace_surface import (
    compiled_mechanism_hypothesis_attachments,
    compiled_program_benchmark_catalog,
    compiled_program_overlay_attachments,
    materialize_compiled_program_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION = ROOT / "benchmarks" / "bufotalin_candidate_route_observation.v1.json"
SCREEN = ROOT / "benchmarks" / "bufotalin_candidate_innovation_screen.v1.json"
CAPABILITIES = ROOT / "config" / "route_innovation_capabilities.v1.json"
HSDH_CAPABILITY = "hsdh:delta4-3-ketosteroid-to-3-hydroxysteroid"


def _load_bound_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    assert observed == strict_canonical_json_sha256(material)
    return value


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index" / "runs.sqlite3"),
        },
    )


def _reported_interval_plan() -> tuple[dict, str]:
    observation = _load_bound_json(OBSERVATION)
    screen = _load_bound_json(SCREEN)
    route_screen = next(iter(screen["route_screens"].values()))
    candidate = next(
        row
        for row in route_screen["discovery"]["candidates"]
        if row["capability_id"] == HSDH_CAPABILITY
    )
    span = candidate["route_innovation"]["replaced_step_ids"]
    steps = []
    for edge_id in span:
        transformation = observation["transformations"][edge_id]
        steps.append(
            {
                "step_id": edge_id,
                "product_smiles": observation["molecules"][
                    transformation["product_molecule_id"]
                ]["canonical_smiles"],
                "precursor_smiles": [
                    observation["molecules"][value]["canonical_smiles"]
                    for value in transformation["precursor_molecule_ids"]
                ],
                "transformation_hypothesis": (
                    "reported Bufotalin chemical interval; visual structures remain L0"
                ),
            }
        )
    plan = {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:bufotalin-enzyme-interval",
                "strategic_disconnection": "reported six-step selective ketone interval",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:bufotalin-enzyme-interval",
                "route_family_id": "family:bufotalin-enzyme-interval",
                "steps": steps,
            }
        ],
    }
    return plan, steps[-1]["product_smiles"]


def test_reported_bufotalin_interval_replays_as_current_canonical_enzyme_positive(
    tmp_path: Path,
) -> None:
    plan, target_smiles = _reported_interval_plan()
    capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))
    gateway = CampaignGateway(_paths(tmp_path))
    created = gateway.create_run(
        run_id="bufotalin-current-enzyme-interval",
        target_name="reported steroid interval",
        target_smiles=target_smiles,
        global_plan=plan,
        materialize=True,
    )
    graph_revision = created["status"]["graph_revision"]
    workbench = gateway.workbench("bufotalin-current-enzyme-interval")["snapshot"]
    route = max(workbench["routes"].values(), key=lambda row: len(row["edge_ids"]))

    assert len(route["edge_ids"]) == 6
    result = gateway.route_program_innovations(
        "bufotalin-current-enzyme-interval",
        route_id=route["route_id"],
        capabilities=capabilities,
    )
    proposal = next(
        row
        for row in result["program_bundle"]["program_proposals"].values()
        if row["source_capability_id"] == HSDH_CAPABILITY
        and row["chemical_step_equivalent_count"] == 6
    )
    plan = next(
        row
        for row in result["validation_frontier"]["plans"].values()
        if row["program_id"] == proposal["program_id"]
    )
    optimizer_candidate_id = next(
        candidate_id
        for candidate_id, row in result["program_route_candidates"]["candidates"].items()
        if proposal["program_id"] in row["substitution_program_ids"]
    )
    baseline_candidate_id = next(
        candidate_id
        for candidate_id, row in result["program_route_candidates"]["candidates"].items()
        if row["source_kind"] == "baseline"
    )

    assert result["oracle"]["accepted"] is True
    assert proposal["status"] == "proposal_only"
    assert proposal["net_step_savings"] == 5
    assert proposal["warning_codes"] == ["EXACT_SUBSTRATE_UNVALIDATED"]
    assert plan["status"] == "experiment_required"
    assert plan["screen_matrix"]["enzyme_candidates"]["candidate_ids"] == [
        "Ct3alpha-HSDH",
        "Ss3beta-HSDH",
    ]
    assert plan["exact_boundary"]["input_states"][0]["canonical_smiles"] == (
        "CC12CCC3C(CCC4=CC(=O)CCC43C)C1CCC2=O"
    )
    assert plan["exact_boundary"]["output_states"][0]["canonical_smiles"] == (
        "CC12CCC3C(CCC4=CC(O)CCC43C)C1CCC2=O"
    )
    assert plan["grants_validation"] is False
    assert result["program_optimizer_oracle"]["accepted"] is True
    assert optimizer_candidate_id in result["program_optimizer"]["profiles"][
        "exploration"
    ]["pareto_front_ids"]
    assert result["program_optimizer"]["profiles"]["shadow_optimizer"][
        "pareto_front_ids"
    ] == [baseline_candidate_id]
    assert result["program_route_candidates"]["candidates"][optimizer_candidate_id][
        "eligibility"
    ]["shadow_optimizer"] is False
    forest = compile_v4_route_forest(
        workbench,
        program_innovation_reviews=[result],
    )
    assert len(forest["program_overlays"]) == 1
    overlay = forest["program_overlays"][0]
    assert overlay["program_id"] == proposal["program_id"]
    assert overlay["chemical_step_equivalent_count"] == 6
    assert overlay["isolated_operation_count"] == 1
    assert overlay["net_step_savings"] == 5
    assert overlay["candidate_enzyme_ids"] == ["Ct3alpha-HSDH", "Ss3beta-HSDH"]
    assert len(overlay["replaced_step_ids"]) == 6
    assert overlay["fallback_retained"] is True
    assert overlay["eligible_for_route_completion"] is False
    payload = build_route_forest_delivery_payload(forest)
    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []
    html = render_v4_route_workbench_html(
        workbench,
        program_innovation_reviews=[result],
    )
    assert "个化学步骤 → 1 个酶操作" in html
    assert "program-overlay-card" in html
    assert "原 ${equivalent} 步化学 fallback" in html
    with pytest.raises(CampaignGatewayError, match="requires_validated_candidate"):
        gateway.admit_route_program_innovations(
            "bufotalin-current-enzyme-interval",
            route_id=route["route_id"],
            capabilities=capabilities,
            enable_biocatalytic_program_admission=True,
        )
    assert gateway.biocatalytic_program_store(
        "bufotalin-current-enzyme-interval"
    )["replay"]["event_count"] == 0
    assert gateway.status("bufotalin-current-enzyme-interval")["status"][
        "graph_revision"
    ] == graph_revision


def test_bufotalin_program_attaches_to_the_full_twenty_step_host_route(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    record = next(
        value
        for value in compiled_program_benchmark_catalog()["records"]
        if value["target_name"] == "bufotalin"
        and value["chemical_step_equivalent_count"] == 6
    )
    materialized = materialize_compiled_program_benchmark(gateway, record["benchmark_id"])
    workbench = gateway.workbench(materialized["run_id"])["snapshot"]
    attachments = compiled_program_overlay_attachments(materialized["run_id"])
    mechanism_attachments = compiled_mechanism_hypothesis_attachments(
        materialized["run_id"]
    )

    forest = compile_v4_route_forest(
        workbench,
        program_overlay_attachments=attachments,
        mechanism_hypothesis_attachments=mechanism_attachments,
    )

    planned = next(iter(workbench["planned_routes"].values()))
    assert planned["declared_step_count"] == 20
    assert planned["materialized_step_count"] == 15
    assert planned["admission_rejected_step_count"] == 5
    assert len(forest["program_overlays"]) == 1
    overlay = forest["program_overlays"][0]
    assert overlay["branch_id"] == forest["primary_branch_id"]
    assert len(overlay["replaced_edge_ids"]) == 6
    assert len(overlay["replaced_step_ids"]) == 6
    assert overlay["fallback_retained"] is True
    assert overlay["eligible_for_route_completion"] is False
    assert len(forest["mechanism_hypotheses"]) == 1
    mechanism = forest["mechanism_hypotheses"][0]
    assert mechanism["branch_id"] == overlay["branch_id"]
    assert mechanism["anchor_edge_ids"] == ["edge:paper:31"]
    assert mechanism["proposal_depth"] == 1
    assert mechanism["eligible_for_route_completion"] is False
    assert mechanism["proposed_product"]["formula"]
    assert mechanism["proposed_product"]["structure_svg"].startswith("<svg")
    anchor_step = next(
        value for value in forest["steps"] if value["step_id"] == mechanism["anchor_step_ids"][0]
    )
    assert mechanism["anchor_molecule_node_id"] in anchor_step["to_node_ids"]
    assert mechanism["precursor_state"]["canonical_smiles"] == next(
        value["smiles"]
        for value in forest["nodes"]
        if value["node_id"] == mechanism["anchor_molecule_node_id"]
    )
    assert mechanism["proposed_product"]["canonical_smiles"] not in {
        value["smiles"] for value in forest["nodes"]
    }
    branch = next(
        value for value in forest["branches"] if value["branch_id"] == overlay["branch_id"]
    )
    assert len(branch["step_ids"]) == 20
    assert branch["advisory_only"] is True
    assert branch["source_refs"] == ["doi:10.1016/j.tet.2025.134610"]
    host_steps = [
        value for value in forest["steps"] if value["branch_id"] == overlay["branch_id"]
    ]
    assert sum(value["proof_tier"] == "L1_source_reported" for value in host_steps) == 12
    assert sum(value["proof_tier"] == "L0_rejected" for value in host_steps) == 5
    assert sum(bool(value["source_refs"]) for value in host_steps) == 15
    assert sum(
        value["proof_tier"] == "L0_rejected" and bool(value["source_refs"])
        for value in host_steps
    ) == 3
    payload = build_route_forest_delivery_payload(forest)
    assert route_forest_delivery_integrity_reasons(payload, source_forest=forest) == []

    invalid_attachment = dict(attachments[0])
    invalid_attachment["boundary"] = {
        **dict(invalid_attachment["boundary"]),
        "precursor": {
            **dict(invalid_attachment["boundary"]["precursor"]),
            "smiles": "C",
        },
    }
    invalid_forest = compile_v4_route_forest(
        workbench,
        program_overlay_attachments=[invalid_attachment],
    )
    assert invalid_forest["program_overlays"] == []

    invalid_mechanism = {
        **dict(mechanism_attachments[0]),
        "precursor_smiles": "C",
    }
    invalid_mechanism_forest = compile_v4_route_forest(
        workbench,
        mechanism_hypothesis_attachments=[invalid_mechanism],
    )
    assert invalid_mechanism_forest["mechanism_hypotheses"] == []
