from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.target_solver import (
    TargetSolveConfig,
    _current_disposition,
)
from cascade_planner.interfaces.patent_evidence import (
    BuiltinPatentEvidenceConfig,
    build_builtin_patent_evidence_connector,
)
from cascade_planner.runtime import AgentResult, AgentSpec, AgentState
from cascade_planner.runtime.paths import RuntimePaths


TARGET = "CCOC(C)=O"


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
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def _scanned_patent_pdf(label: str = "") -> bytes:
    image = Image.new("RGB", (1200, 1600), "white")
    ImageDraw.Draw(image).text(
        (50, 80),
        f"Ethyl acetate procedures T1-T3 {label}",
        fill="black",
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=buffer.getvalue())
    value = document.tobytes()
    document.close()
    return value


def _patent_html(publication: str) -> bytes:
    return f"""
    <html><head><meta name="DC.relation" content="{publication}"></head>
    <body>
      <div id="p0001" class="description-paragraph">Example 1</div>
      <div id="p0002" class="description-paragraph">
        Ethyl acetate (T1). Ethanol and acetyl chloride were added. The
        reaction mixture was stirred to afford T1.
      </div>
      <div id="p0003" class="description-paragraph">Example 2</div>
      <div id="p0004" class="description-paragraph">
        Ethyl acetate (T2). Ethanol and acetic acid were added. The reaction
        mixture was stirred to afford T2.
      </div>
      <div id="p0005" class="description-paragraph">Example 3</div>
      <div id="p0006" class="description-paragraph">
        Ethyl acetate (T3). Ethanol and acetyl bromide were added. The
        reaction mixture was stirred to afford T3.
      </div>
    </body></html>
    """.encode()


def _plan(context: Any, mode: str) -> dict[str, Any]:
    families = [
        ("family:chloride", "CC(=O)Cl", "acyl chloride substitution"),
        ("family:acid", "CC(=O)O", "direct esterification"),
        ("family:bromide", "CC(=O)Br", "acyl bromide substitution"),
    ]
    return {
        "schema_version": "global_campaign_plan.v1",
        "plan_id": f"blind-plan:{mode}",
        "run_id": context.run_id,
        "mode": mode,
        "context_sha256": context.content_sha256,
        "graph_revision": context.revision.graph_revision,
        "route_families": [
            {
                "route_family_id": family_id,
                "title": hypothesis,
                "strategy": hypothesis,
                "target_smiles": TARGET,
                "advantages": ["short"],
                "risks": ["selectivity"],
                "diversity_basis": precursor,
            }
            for family_id, precursor, hypothesis in families
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": f"skeleton:{index}",
                "route_family_id": family_id,
                "summary": hypothesis,
                "steps": [
                    {
                        "step_id": f"step:{index}",
                        "product_smiles": TARGET,
                        "precursor_smiles": ["CCO", precursor],
                        "transformation_hypothesis": hypothesis,
                        "strategic_role": "target convergence",
                        "source_hints": [],
                        "required_validation": ["atom mapping", "stock audit"],
                        "hypothesis_only": True,
                    }
                ],
            }
            for index, (family_id, precursor, hypothesis) in enumerate(families, start=1)
        ],
        "strategic_disconnections": [],
        "shared_intermediates": [],
        "critical_unknowns": [],
        "source_plan": [],
        "fallback_strategies": [],
        "frontier_priorities": [
            {
                "priority_id": f"priority:{index}",
                "proposal_id": f"step:{index}",
                "priority": 10 - index,
                "rationale": "close complete route",
                "expected_portfolio_gain": "one distinct family",
            }
            for index in (1, 2)
        ],
        "pivot_conditions": [],
        "stop_conditions": [],
        "portfolio_rationale": "Two different target-level acyl donors.",
        "limitations": ["requires host validation"],
    }


def _runner(spec: AgentSpec, context: Any, mode: str, _config: Any) -> AgentResult:
    return AgentResult(
        run_id=spec.run_id,
        agent_id=spec.agent_id,
        parent_agent_id=spec.parent_agent_id,
        attempt=spec.attempt,
        idempotency_key=f"{spec.idempotency_key}:result",
        context_hash=spec.context_hash,
        capabilities=spec.capabilities,
        write_scope=spec.write_scope,
        budget=spec.budget,
        state=AgentState.SUCCEEDED,
        output=_plan(context, mode),
        usage={
            "model_invocations": 1,
            "input_tokens": 1000,
            "output_tokens": 700,
            "wall_time_s": 1.0,
        },
    )


def _failed_runner(
    spec: AgentSpec,
    _context: Any,
    _mode: str,
    _config: Any,
) -> AgentResult:
    return AgentResult(
        run_id=spec.run_id,
        agent_id=spec.agent_id,
        parent_agent_id=spec.parent_agent_id,
        attempt=spec.attempt,
        idempotency_key=f"{spec.idempotency_key}:result",
        context_hash=spec.context_hash,
        capabilities=spec.capabilities,
        write_scope=spec.write_scope,
        budget=spec.budget,
        state=AgentState.FAILED,
        error="provider_unavailable",
        usage={"model_invocations": 1, "wall_time_s": 0.1},
    )


def _mapper(reactions: list[str]) -> list[str]:
    values = []
    for reaction in reactions:
        if "Br" in reaction:
            values.append(
                "[CH3:1][C:2](=[O:3])[Br:4].[CH3:5][CH2:6][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
            )
        elif "Cl" in reaction:
            values.append(
                "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][CH2:6][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
            )
        else:
            values.append(
                "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
            )
    return values


def _catalog(smiles: list[str], **_: Any) -> dict[str, Any]:
    return {
        "schema_version": "versioned_benchmark_stock_catalog.v1",
        "adapter_version": "tests.generic-catalog.v1",
        "catalog_name": "test-generic-catalog",
        "catalog_version": "2026-07-14",
        "retrieved_at": "2026-07-14T00:00:00Z",
        "members": [
            {
                "canonical_smiles": value,
                "cid": index,
                "vendor_count": 1,
                "vendors": [],
                "source_url": f"https://catalog.invalid/{index}",
                "response_sha256": f"{index:064x}",
            }
            for index, value in enumerate(sorted(set(smiles)), start=1)
        ],
        "misses": [],
    }


def _partial_catalog(smiles: list[str], **_: Any) -> dict[str, Any]:
    catalog = _catalog(smiles)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    catalog["catalog_version"] = timestamp
    catalog["retrieved_at"] = timestamp
    missing = sorted(set(smiles))[-1]
    catalog["members"] = [
        row
        for row in catalog["members"]
        if row["canonical_smiles"] != missing
    ]
    catalog["misses"] = [
        {
            "canonical_smiles": missing,
            "cid": 0,
            "reason": "test_catalog_miss",
        }
    ]
    return catalog


def _evidence_connector(request: Any) -> dict[str, Any]:
    rows = [
        {
            "product_smiles": edge["product_smiles"],
            "reactant_smiles": edge["precursor_smiles"],
            "step_id": f"connector-step:{index}",
            "location_ref": f"Example 4, step {index}",
            "conditions": {"temperature_c": 25},
        }
        for index, edge in enumerate(request["edges"], start=1)
    ]
    return {
        "schema_version": "structured_evidence_import.v1",
        "sources": [
            {
                "binding": {
                    "source_kind": source_kind,
                    "source_ref": source_ref,
                    "title": f"Independent source {index}",
                    "provenance": "typed_connector",
                },
                "extraction": {
                    "schema_version": "structured_exact_row_extraction.v1",
                    "extractor": {
                        "producer_kind": "typed_connector_structured_extraction",
                        "producer_id": "tests.target-evidence",
                        "version": "1.0.0",
                    },
                    "rows": rows,
                },
            }
            for index, (source_kind, source_ref) in enumerate(
                (
                    ("patent", "patent:US1234567A1"),
                    ("paper_si", "doi:10.1000/example.1"),
                ),
                start=1,
            )
        ],
    }


def _discovery_only_connector(request: Any) -> dict[str, Any]:
    return {
        "discovery": {
            "schema_version": "source_discovery_observation.v1",
            "provider_id": "tests.discovery",
            "request_sha256": request["content_sha256"],
            "sources": [
                {
                    "publication_number": "US7654321A1",
                    "family_id": "family:discovery",
                    "title": "Alternative ester preparation",
                    "procedure_inventory": [
                        {
                            "label": "11",
                            "name": "ethyl acetate",
                            "page_number": 4,
                            "procedure_excerpt": (
                                "Ethanol and acetic acid were combined under "
                                "the source conditions."
                            ),
                        }
                    ],
                    "exact_edge_ids": [],
                    "exact_row_count": 0,
                }
            ],
            "semantics": {
                "source_text_is_untrusted_data": True,
                "discovery_does_not_grant_exact_evidence": True,
            },
        },
        "receipt": {
            "schema_version": "evidence_connector_receipt.v1",
            "provider_id": "tests.discovery",
            "model_invocations": 0,
        },
    }


def _inventory_builder(smiles: list[str], **_: Any) -> dict[str, Any]:
    checked_at = "2026-07-14T00:00:00Z"
    return {
        "schema_version": "versioned_inventory_snapshot.v1",
        "adapter_version": "tests.inventory.v1",
        "inventory_version": "snapshot-2026-07-14",
        "retrieved_at": checked_at,
        "offers": [
            {
                "schema_version": "stock_offer_snapshot.v1",
                "supplier": "Test Supplier",
                "catalog_number": f"SKU-{index}",
                "smiles": value,
                "checked_at": checked_at,
                "available": True,
                "source_url": f"https://supplier.invalid/SKU-{index}",
            }
            for index, value in enumerate(sorted(set(smiles)), start=1)
        ],
    }


def test_target_only_solver_runs_global_plan_validation_stock_and_resume(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=2,
        minimum_edge_proof_level=2,
        stock_boundary="benchmark_search",
        minimum_independent_source_groups=2,
    )
    budget = RetrosynthesisRunBudget(
        max_model_invocations=2,
        max_total_input_tokens=10_000,
        max_total_output_tokens=5_000,
        max_total_wall_time_s=60,
        max_visual_invocations=0,
        max_accepted_expansions=8,
        max_attempt_runs=16,
    )
    result = gateway.solve_target(
        target_name="opaque blind molecule",
        target_smiles=TARGET,
        run_id="blind-target-e2e",
        acceptance=acceptance,
        budget=budget,
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )

    assert result["preflight"]["accepted"] is True
    assert result["model_cost"]["model_invocations"] == 1
    assert result["accepted_expansion_count"] == 3
    assert result["gates"]["gates"] == {
        "B0_blind_input": True,
        "B1_global_multi_route": True,
        "B2_host_validated_routes": True,
        "B3_exact_multi_source": False,
        "B4_stock_boundary": True,
        "B5_configured_portfolio_acceptance": True,
    }
    assert result["claim"]["generated_route_portfolio"] is True
    assert result["claim"]["exact_multi_source_grade"] is False
    assert result["claim"]["procurement_ready"] is False
    assert result["current_disposition"]["state"] == "accepted"
    assert Path(result["report_path"]).is_file()

    resumed = gateway.solve_target(
        target_name="opaque blind molecule",
        target_smiles=TARGET,
        run_id="blind-target-e2e",
        acceptance=acceptance,
        budget=budget,
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
        ),
        resume=True,
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )
    assert resumed["model_cost"]["model_invocations"] == 1
    assert resumed["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True


def test_current_disposition_does_not_treat_stale_terminal_as_scientific_success() -> None:
    disposition = _current_disposition(
        kernel_status="completed",
        stop_decision={"decision": "completed", "terminal": True},
        claim={"accepted_under_configured_policy": False},
        gates={
            "reaction_proof_version_audit": {"requires_revalidation": True}
        },
    )

    assert disposition["state"] == "terminal_snapshot_requires_revalidation"
    assert disposition["scientifically_accepted"] is False
    assert disposition["requires_revalidation"] is True


def test_target_solver_reports_provider_failure_without_replan_or_false_closure(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="unknown failed provider target",
        target_smiles=TARGET,
        run_id="blind-provider-failure",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
            enable_live_benchmark_stock=False,
        ),
        director_runner=_failed_runner,
    )

    assert result["director_outcomes"] == [
        {
            "schema_version": "global_campaign_director_outcome.v1",
            "status": "failed",
            "invoked": True,
            "cache_hit": False,
            "mode": "initial_architecture",
            "context_sha256": "",
            "plan": None,
            "proposal_audits": [],
            "reasons": [
                "GlobalCampaignDirectorError",
                "director_child_failed:provider_unavailable",
            ],
            "artifact_sha256": "",
            "task_id": "",
        }
    ]
    assert result["model_cost"]["model_invocations"] == 1
    assert result["gates"]["gates"]["B0_blind_input"] is True
    assert all(
        value is False
        for key, value in result["gates"]["gates"].items()
        if key != "B0_blind_input"
    )
    assert result["claim"]["accepted_under_configured_policy"] is False
    assert Path(result["report_path"]).is_file()


def test_resume_reuses_fresh_negative_stock_audits_without_spending_attempts(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    config = TargetSolveConfig(
        use_coordinator=False,
        enable_web_search=False,
        enable_replan=False,
    )
    first = gateway.solve_target(
        target_name="partial stock blind molecule",
        target_smiles=TARGET,
        run_id="blind-partial-stock",
        config=config,
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_partial_catalog,
    )

    def unexpected_catalog_call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("fresh negative stock audit should be reused")

    resumed = gateway.solve_target(
        target_name="partial stock blind molecule",
        target_smiles=TARGET,
        run_id="blind-partial-stock",
        config=config,
        resume=True,
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=unexpected_catalog_call,
    )

    assert first["gates"]["gates"]["B4_stock_boundary"] is False
    assert resumed["attempt_count"] == first["attempt_count"]
    assert any(
        stage["stage"] == "stock"
        and stage["detail"].get("status") == "reused"
        and stage["detail"].get("miss_count") == 1
        for stage in resumed["stages"]
    )
def test_target_solver_ingests_connector_rows_before_stock_and_closeout(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind evidence target",
        target_smiles=TARGET,
        run_id="blind-target-evidence-e2e",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
            minimum_independent_source_groups=2,
            stock_boundary="benchmark_search",
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=20,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_evidence_connector,
    )

    assert result["model_cost"]["model_invocations"] == 1
    assert result["gates"]["gates"]["B2_host_validated_routes"] is True
    assert result["gates"]["gates"]["B3_exact_multi_source"] is True
    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    assert result["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    evidence_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "evidence_acquisition"
    )
    assert evidence_stage["status"] == "completed"
    assert evidence_stage["detail"]["exact_record_count"] == 6


def test_target_solver_replans_globally_from_unbound_source_discovery(
    tmp_path: Path,
) -> None:
    observed_modes: list[str] = []

    def runner(
        spec: AgentSpec, context: Any, mode: str, config: Any
    ) -> AgentResult:
        observed_modes.append(mode)
        if mode == "event_replan":
            assert context.evidence["source_discovery"]["sources"][0][
                "publication_number"
            ] == "US7654321A1"
            assert (
                "source_material_discovered" in context.delta.material_events
            )
        return _runner(spec, context, mode, config)

    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind discovery target",
        target_smiles=TARGET,
        run_id="blind-target-discovery-replan",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_discovery_only_connector,
    )

    assert observed_modes == ["initial_architecture", "event_replan"]
    assert result["model_cost"]["model_invocations"] == 2
    assert any(
        stage["stage"] == "global_replan_budget_gate"
        and stage["detail"]["trigger_reasons"]
        == ["evidence_deficit_with_new_source_material"]
        for stage in result["stages"]
    )
    discovery_stage = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "evidence_acquisition"
    )
    assert discovery_stage["status"] == "discovered_unbound"
    assert discovery_stage["detail"]["exact_record_count"] == 0


def test_target_solver_uses_one_budgeted_visual_candidate_in_global_replan(
    tmp_path: Path,
) -> None:
    image = tmp_path / "visual-source-page.png"
    image.write_bytes(b"target-solver-visual-source-page")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    observed_modes: list[str] = []
    visual_calls = 0

    def connector(request: Any) -> dict[str, Any]:
        result = _discovery_only_connector(request)
        source = result["discovery"]["sources"][0]
        source.update(
            {
                "pdf_sha256": "b" * 64,
                "unresolved_edge_count": len(request["edges"]),
                "visual_candidate_pages": [
                    {
                        "page_number": 4,
                        "image_path": str(image),
                        "image_sha256": image_sha256,
                    }
                ],
            }
        )
        return result

    def visual_provider(request: Any) -> dict[str, Any]:
        nonlocal visual_calls
        visual_calls += 1
        edge = request["edges"][0]
        return {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.visual"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": edge["product_smiles"],
                        "reactant_smiles": edge["precursor_smiles"],
                        "source_locator": "page 4",
                    }
                ]
            },
        }

    def runner(
        spec: AgentSpec, context: Any, mode: str, config: Any
    ) -> AgentResult:
        observed_modes.append(mode)
        if mode == "event_replan":
            visual = context.evidence["visual_source_candidates"]
            assert visual["candidate_step_count"] == 1
            assert visual["candidate_steps"][0]["grants_exact_evidence"] is False
            assert "visual_source_candidates_added" in context.delta.material_events
        return _runner(spec, context, mode, config)

    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind visual discovery target",
        target_smiles=TARGET,
        run_id="blind-target-visual-replan",
        budget=RetrosynthesisRunBudget(
            max_model_invocations=3,
            max_total_input_tokens=50_000,
            max_total_output_tokens=14_000,
            max_total_wall_time_s=720,
            max_visual_invocations=1,
            max_accepted_expansions=8,
            max_attempt_runs=20,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
        visual_evidence_provider=visual_provider,
    )

    assert observed_modes == ["initial_architecture", "event_replan"]
    assert visual_calls == 1
    assert result["model_cost"]["model_invocations"] == 3
    assert result["model_cost"]["visual_invocations"] == 1
    first_evidence = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "evidence_acquisition"
    )
    assert first_evidence["detail"]["visual_evidence"]["status"] == "completed"
    assert first_evidence["detail"]["exact_record_count"] == 0


def test_target_solver_can_close_procurement_from_frozen_supplier_snapshot(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind procurement target",
        target_smiles=TARGET,
        run_id="blind-target-procurement-e2e",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=2,
            minimum_independent_source_groups=2,
            stock_boundary="procurement",
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=16,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        inventory_snapshot_builder=_inventory_builder,
    )

    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    assert result["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    assert result["claim"]["procurement_ready"] is True
    stock_stage = next(stage for stage in result["stages"] if stage["stage"] == "stock")
    assert stock_stage["detail"]["stock_closed_leaf_count"] == 4


def test_validation_fork_replays_global_plan_and_uses_zero_model_calls(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind validation lineage target",
        target_smiles=TARGET,
        run_id="blind-validation-source",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-validation-derived",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_evidence_connector,
    )

    assert derived["schema_version"] == "target_validation_fork_report.v1"
    assert derived["model_cost"]["model_invocations"] == 0
    assert derived["lineage"]["source_run_id"] == source["run_id"]
    assert derived["lineage"]["source_report_sha256"] == source["content_sha256"]
    assert derived["semantics"]["B0_refers_to_bound_source_campaign"] is True
    assert derived["gates"]["gates"] == {
        "B0_blind_input": True,
        "B1_global_multi_route": True,
        "B2_host_validated_routes": True,
        "B3_exact_multi_source": True,
        "B4_stock_boundary": True,
        "B5_configured_portfolio_acceptance": True,
    }
    assert derived["current_disposition"]["state"] == "accepted"
    assert Path(derived["report_path"]).is_file()


def test_scanned_patent_ocr_closes_blind_route_and_zero_model_validation_fork(
    tmp_path: Path,
) -> None:
    structures = {
        "ethyl acetate": TARGET,
        "ethanol": "CCO",
        "acetyl chloride": "CC(=O)Cl",
        "acetic acid": "CC(=O)O",
        "acetyl bromide": "CC(=O)Br",
    }
    names = {value: [name] for name, value in structures.items()}
    ocr_text = (
        "Ethyl acetate (T1). Ethanol and acetyl chloride were added. The "
        "reaction mixture was stirred to afford T1.\n\n"
        "Ethyl acetate (T2). Ethanol and acetic acid were added. The "
        "reaction mixture was stirred to afford T2.\n\n"
        "Ethyl acetate (T3). Ethanol and acetyl bromide were added. The "
        "reaction mixture was stirred to afford T3."
    )
    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(
            cache_dir=tmp_path / "scanned-patents",
            max_patents=2,
            max_ocr_pages=1,
        ),
        candidate_provider=lambda _queries: [
            {
                "publication_number": publication,
                "family_id": family,
                "title": "Scanned preparation of ethyl acetate",
                "snippet": "primary process source",
                "pdf_url": f"https://source.invalid/{publication}.pdf",
            }
            for publication, family in (
                ("US1234567A1", "family:one"),
                ("WO7654321A1", "family:two"),
            )
        ],
        bytes_fetcher=lambda url, _timeout, _limit: _scanned_patent_pdf(url),
        structure_resolver=lambda value: structures[str(value).casefold()],
        candidate_name_resolver=lambda value: names.get(str(value), []),
        ocr_runner=lambda *_args: {
            "text": ocr_text,
            "engine_id": "tesseract",
            "engine_version": "fixture",
        },
    )
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind scanned patent target",
        target_smiles=TARGET,
        run_id="blind-scanned-patent-source",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=2,
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=24,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert source["model_cost"]["model_invocations"] == 1
    assert source["model_cost"]["visual_invocations"] == 0
    assert source["gates"]["gates"]["B3_exact_multi_source"] is True
    assert source["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    evidence = next(
        stage
        for stage in source["stages"]
        if stage["stage"] == "evidence_acquisition"
    )
    assert evidence["detail"]["exact_record_count"] == 6
    assert all(
        row["ocr_audit"]["status"] == "completed"
        for row in evidence["detail"]["discovery"]["sources"]
    )

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-scanned-patent-validation",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert derived["model_cost"]["model_invocations"] == 0
    assert derived["model_cost"]["visual_invocations"] == 0
    assert derived["gates"]["gates"]["B3_exact_multi_source"] is True
    assert derived["current_disposition"]["state"] == "accepted"


def test_primary_patent_html_closes_blind_portfolio_without_pdf_or_visual_model(
    tmp_path: Path,
) -> None:
    structures = {
        "ethyl acetate": TARGET,
        "ethanol": "CCO",
        "acetyl chloride": "CC(=O)Cl",
        "acetic acid": "CC(=O)O",
        "acetyl bromide": "CC(=O)Br",
    }
    names = {value: [name] for name, value in structures.items()}
    pdf_fetches = 0

    def reject_pdf(_url: str, _timeout: float, _limit: int) -> bytes:
        nonlocal pdf_fetches
        pdf_fetches += 1
        raise AssertionError("fully closed primary HTML must skip PDF")

    def fetch_html(url: str, _timeout: float, _limit: int) -> bytes:
        publication = url.rstrip("/").split("/")[-2]
        return _patent_html(publication)

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(
            cache_dir=tmp_path / "html-patents",
            max_patents=2,
        ),
        candidate_provider=lambda _queries: [
            {
                "publication_number": publication,
                "family_id": family,
                "title": "HTML preparation of ethyl acetate",
                "snippet": "search metadata only",
                "html_url": (
                    f"https://patents.google.com/patent/{publication}/en"
                ),
                "pdf_url": f"https://source.invalid/{publication}.pdf",
            }
            for publication, family in (
                ("US1234567A1", "family:html-one"),
                ("WO7654321A1", "family:html-two"),
            )
        ],
        bytes_fetcher=reject_pdf,
        html_fetcher=fetch_html,
        structure_resolver=lambda value: structures[str(value).casefold()],
        candidate_name_resolver=lambda value: names.get(str(value), []),
    )
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind primary HTML patent target",
        target_smiles=TARGET,
        run_id="blind-primary-html-source",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=2,
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=24,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert pdf_fetches == 0
    assert source["model_cost"]["model_invocations"] == 1
    assert source["model_cost"]["visual_invocations"] == 0
    assert source["gates"]["gates"]["B3_exact_multi_source"] is True
    assert source["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    evidence = next(
        stage
        for stage in source["stages"]
        if stage["stage"] == "evidence_acquisition"
    )
    assert evidence["detail"]["exact_record_count"] == 6
    assert all(
        row["html_sha256"] and not row["pdf_sha256"]
        for row in evidence["detail"]["discovery"]["sources"]
    )
    assert list(tmp_path.rglob("*.pdf")) == []
    assert list(tmp_path.rglob("*.png")) == []

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-primary-html-validation",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert pdf_fetches == 0
    assert derived["model_cost"]["model_invocations"] == 0
    assert derived["model_cost"]["visual_invocations"] == 0
    assert derived["current_disposition"]["state"] == "accepted"
