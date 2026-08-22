from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.application.biocatalytic_step_contract import (
    biocatalytic_step_proof_gate,
    normalize_biocatalytic_step,
)
from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
    CanonicalIngestionBatch,
)
from cascade_planner.application.proof_policy import ProofPolicy, stitch_edge_proof
from cascade_planner.application.reaction_proof_versions import (
    CURRENT_REACTION_VALIDATOR_VERSION,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.strategy_contract import normalize_strategy_card
from cascade_planner.application.worker_runtime import WorkerRuntime
from cascade_planner.orchestration.sequential_strategy_director import (
    NodeExpansion,
    _host_route_json_from_steps,
    _step_row,
)


def _biological_card() -> dict:
    return normalize_strategy_card(
        {
            "scaffold_motif": "stereogenic secondary alcohol",
            "key_forward_transformation": "ketoreductase carbonyl reduction",
            "key_bond_changes": ["map 1-map 2"],
            "functional_group_conflicts": ["competing ketone"],
            "protection_policy": "avoid unnecessary protection",
            "stereochemical_plan": "enzyme-controlled hydride delivery",
            "convergence_plan": "prepare ketone then reduce selectively",
            "strategic_step_count": 1,
            "skeleton_change_class": "stereoselective redox",
            "expected_complexity_drop": "medium",
            "orthogonality_basis": "biocatalytic stereocontrol",
            "strategy_signature": "kred-selective-reduction",
            "execution_domain": "hybrid",
            "biocatalytic_intent": {
                "mode": "enzyme_reaction",
                "enzyme_classes": ["ketoreductase"],
                "ec_numbers": [],
                "candidate_ids": [],
                "whole_cell_hosts": [],
                "selectivity_objective": "form the assigned alcohol stereocenter",
                "substrate_scope_basis": "aryl alkyl ketone capability hypothesis",
                "cofactor_assessment": "required",
                "intended_chemical_step_equivalent_count": 2,
                "fallback_policy": "retain an asymmetric chemical reduction route",
                "validation_plan": ["exact-substrate analytical screen"],
            },
        }
    )


def _step_hypothesis() -> dict:
    return {
        "mode": "enzyme_reaction",
        "enzyme_label": "KRED panel",
        "enzyme_classes": ["ketoreductase"],
        "ec_numbers": ["EC 1.1.1.-"],
        "candidate_ids": [],
        "sequence_refs": [],
        "whole_cell_hosts": [],
        "selectivity_objective": "reduce only the mapped ketone to the desired alcohol",
        "substrate_scope_basis": "aryl alkyl ketone panel precedent",
        "cofactor_assessment": "required",
        "cofactor_requirements": ["NADPH"],
        "cofactor_regenerations": ["glucose/GDH"],
        "cosubstrates": ["glucose"],
        "precedent_refs": [],
        "validation_plan": ["exact-substrate conversion and ee screen"],
    }


def test_biocatalytic_reactionjson_binds_exact_boundary_without_claiming_savings() -> None:
    row, reasons = normalize_biocatalytic_step(
        _step_hypothesis(),
        execution_domain="enzymatic",
        product_smiles="CC(O)c1ccccc1",
        precursor_smiles=["CC(=O)c1ccccc1"],
        enzyme_label="KRED panel",
        step_id="route:step:4",
    )

    assert reasons == []
    assert row["boundary"] == {
        "forward_input_smiles": ["CC(=O)c1ccccc1"],
        "forward_output_smiles": "CC(O)c1ccccc1",
        "authority": "host_replayed_reactionjson",
    }
    assert row["validation_gate"]["accepted"] is False
    assert row["step_accounting"]["physical_operation_count"] == 1
    assert row["step_accounting"]["chemical_step_equivalent_count"] is None
    assert row["step_accounting"]["net_step_savings"] is None


def test_routejson_classifies_each_step_instead_of_copying_hybrid_branch_domain() -> None:
    card = _biological_card()
    chemical = _step_row(
        NodeExpansion(
            product_smiles="CCO",
            precursor_smiles=("CC", "O"),
            reaction_family="chemical cleavage",
            rationale="ordinary chemical preparation inside a hybrid strategy",
            mapped_product_smiles="[CH3:1][CH2:2][OH:3]",
            mapped_precursor_smiles=("[CH3:1][CH3:2]", "[OH2:3]"),
            strategy_card=card,
            execution_domain="chemical",
        ),
        step_id="chemical:1",
    )
    enzymatic = _step_row(
        NodeExpansion(
            product_smiles="CC(O)c1ccccc1",
            precursor_smiles=("CC(=O)c1ccccc1",),
            reaction_family="ketoreductase reduction",
            rationale="set the alcohol stereocenter",
            mapped_product_smiles="CC(O)c1ccccc1",
            mapped_precursor_smiles=("CC(=O)c1ccccc1",),
            strategy_card=card,
            enzyme="KRED panel",
            execution_domain="enzymatic",
            biocatalytic_step=_step_hypothesis(),
        ),
        step_id="enzyme:1",
    )

    assert chemical["execution_domain"] == "chemical"
    assert chemical["step_kind"] == "chemical_reaction"
    assert chemical["biocatalytic_step"] == {}
    assert "exact_substrate_biocatalysis" not in chemical["required_validation"]
    assert enzymatic["execution_domain"] == "enzymatic"
    assert enzymatic["step_kind"] == "biocatalytic_reaction"
    assert enzymatic["biocatalytic_step"]["boundary"]["forward_output_smiles"] == (
        "CC(O)c1ccccc1"
    )
    assert "exact_substrate_biocatalysis" in enzymatic["required_validation"]

    route = _host_route_json_from_steps([enzymatic])
    assert route[0]["execution_domain"] == "enzymatic"
    assert route[0]["biocatalytic_step"]["validation_gate"]["accepted"] is False


def test_biocatalytic_step_requires_digest_bound_specialized_validation() -> None:
    step, _ = normalize_biocatalytic_step(
        _step_hypothesis(),
        execution_domain="enzymatic",
        product_smiles="CC(O)c1ccccc1",
        precursor_smiles=["CC(=O)c1ccccc1"],
    )

    generic = biocatalytic_step_proof_gate(
        [step],
        [{"accepted": True, "biocatalysis_validation": {"accepted": True}}],
    )
    exact = biocatalytic_step_proof_gate(
        [step],
        [
            {
                "accepted": True,
                "biocatalysis_validation": {
                    "accepted": True,
                    "step_contract_sha256": step["content_sha256"],
                },
            }
        ],
    )

    assert generic["accepted"] is False
    assert generic["reasons"] == ["exact_biocatalytic_step_validation_missing"]
    assert exact["accepted"] is True
    assert exact["validated_step_contract_sha256"] == [step["content_sha256"]]


def test_biocatalytic_step_survives_canonical_materialization_and_gates_l2(
    tmp_path: Path,
) -> None:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="biocatalytic-materialization",
            target_name="ethyl acetate",
            target_smiles="CCOC(C)=O",
            created_at="2026-08-20T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=0,
                    max_accepted_expansions=8,
                    max_attempt_runs=16,
                ),
                max_total_tasks=16,
            ),
        ),
    )
    kernel.start()
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    step_contract, reasons = normalize_biocatalytic_step(
        {
            "mode": "enzyme_reaction",
            "enzyme_label": "lipase panel",
            "enzyme_classes": ["lipase"],
            "ec_numbers": ["EC 3.1.1.3"],
            "candidate_ids": [],
            "sequence_refs": [],
            "whole_cell_hosts": [],
            "selectivity_objective": "form the ester without auxiliary activation",
            "substrate_scope_basis": "small alcohol and acetate-donor hypothesis",
            "cofactor_assessment": "not_required",
            "cofactor_requirements": [],
            "cofactor_regenerations": [],
            "cosubstrates": [],
            "precedent_refs": [],
            "validation_plan": ["exact-substrate conversion and water-activity screen"],
        },
        execution_domain="enzymatic",
        product_smiles="CCOC(C)=O",
        precursor_smiles=["CCO", "CC(=O)O"],
        step_id="enzyme:esterification",
    )
    assert reasons == []
    plan = {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:lipase",
                "strategic_disconnection": "biocatalytic esterification",
                "strategy_card": _biological_card(),
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:lipase",
                "route_family_id": "family:lipase",
                "steps": [
                    {
                        "step_id": "enzyme:esterification",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)O"],
                        "transformation_hypothesis": "lipase esterification",
                        "execution_domain": "enzymatic",
                        "biocatalytic_step": step_contract,
                    }
                ],
            }
        ],
    }
    admitted = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="biocatalytic-plan",
    )["graph"]
    hypothesis = next(iter(admitted["hypotheses"].values()))
    assert hypothesis["status"] == "frontier_candidate"
    hypothesis_boundary = hypothesis["biocatalytic_steps"][0]["boundary"]
    assert sorted(hypothesis_boundary["forward_input_smiles"]) == sorted(
        ["CCO", "CC(=O)O"]
    )
    assert hypothesis_boundary["forward_output_smiles"] == "CCOC(C)=O"
    assert hypothesis_boundary["authority"] == "host_replayed_reactionjson"

    commands = store.frontier_materialization_commands()
    assert len(commands) == 1
    materialized = store.apply(
        CanonicalIngestionBatch(
            worker_results=tuple(runtime.execute(command) for command in commands)
        ),
        worker_runtime=runtime,
        idempotency_key="biocatalytic-materialized",
    )["graph"]
    edge_id, edge = next(iter(materialized["edges"].items()))
    persisted_step = edge["biocatalytic_steps"][0]
    assert persisted_step["execution_domain"] == "enzymatic"
    assert persisted_step["step_accounting"]["net_step_savings"] is None

    generic_proof = _reaction_proof()
    generic_graph = _graph_with_reaction_proof(materialized, edge_id, generic_proof)
    policy = ProofPolicy(
        minimum_edge_proof_level=2,
        minimum_independent_source_groups=0,
        require_stock_for_every_selected_leaf=False,
        stock_boundary="benchmark_search",
    )
    generic_stitch = stitch_edge_proof(generic_graph, edge_id, policy=policy)
    assert generic_stitch["reaction_validated"] is True
    assert generic_stitch["achieved_level"] == 1
    assert generic_stitch["accepted"] is False
    assert generic_stitch["biocatalytic_step_proof_gate"]["accepted"] is False

    exact_proof = _reaction_proof(
        biocatalysis_validation={
            "accepted": True,
            "step_contract_sha256": persisted_step["content_sha256"],
        }
    )
    exact_graph = _graph_with_reaction_proof(materialized, edge_id, exact_proof)
    exact_stitch = stitch_edge_proof(exact_graph, edge_id, policy=policy)
    assert exact_stitch["achieved_level"] == 2
    assert exact_stitch["accepted"] is True
    assert exact_stitch["biocatalytic_step_proof_gate"]["accepted"] is True


def _reaction_proof(**extra: object) -> dict:
    row = {
        "schema_version": "reaction_step_proof.v1",
        "validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
        "proof_level": "L2_reaction_validated",
        "accepted": True,
        **extra,
    }
    row["proof_digest"] = _digest(row)
    return row


def _graph_with_reaction_proof(
    graph: dict, edge_id: str, proof: dict
) -> dict:
    copied = json.loads(json.dumps(graph))
    edge = dict(copied["edges"][edge_id])
    edge["reaction_proofs"] = [proof]
    edge.pop("content_sha256", None)
    edge["content_sha256"] = _digest(edge)
    copied["edges"][edge_id] = edge
    return copied


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
