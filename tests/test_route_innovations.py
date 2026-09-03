from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
    CanonicalIngestionBatch,
)
from cascade_planner.application.proof_policy import ProofPolicy
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
)
from cascade_planner.application.route_innovations import (
    BIOCATALYTIC_STEP,
    BIOCATALYTIC_SUPERSTEP,
    MECHANISM_EXTRAPOLATION,
    innovation_proof_gate,
    merge_route_innovations,
    normalize_route_innovation,
    route_innovation_summary,
)
from cascade_planner.application.route_innovation_chemenzy import (
    route_innovation_from_chemenzy_step,
)
from cascade_planner.application.route_variants import (
    RouteSubroute,
    build_route_candidate,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.worker_runtime import WorkerRuntime


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _enzyme_option(*, family_id: str = "family:enzyme") -> dict:
    record, reasons = normalize_route_innovation(
        {
            "kind": BIOCATALYTIC_SUPERSTEP,
            "route_family_id": family_id,
            "chemical_step_equivalent_count": 5,
            "replaced_step_ids": ["paper:4", "paper:5", "paper:6", "paper:7", "paper:8"],
            "enzyme_class": "ketoreductase",
            "selectivity_objective": "set both side-chain alcohol stereocentres",
            "substrate_scope_basis": "similarity-predicted",
            "cofactor_requirements": {"NADPH": 1},
        }
    )
    assert reasons == []
    return record


def test_biocatalytic_superstep_records_compression_without_granting_proof() -> None:
    record = _enzyme_option()

    assert record["chemical_step_equivalent_count"] == 5
    assert record["step_savings"] == 4
    assert record["enzyme"]["classes"] == ["ketoreductase"]
    assert record["authority_scope"] == "proposal_only"
    assert record["not_reaction_proof"] is True


def test_biocatalytic_replacement_span_preserves_topological_order() -> None:
    record, reasons = normalize_route_innovation(
        {
            "kind": "biocatalytic_superstep",
            "chemical_step_equivalent_count": 3,
            "replaced_step_ids": ["edge:z", "edge:a", "edge:m", "edge:a"],
            "enzyme_class": "ketoreductase",
            "selectivity_objective": "Reduce the specified ketone stereoselectively.",
        }
    )

    assert reasons == []
    assert record["replaced_step_ids"] == ["edge:z", "edge:a", "edge:m"]


def test_single_biocatalytic_step_is_valid_without_claiming_route_compression() -> None:
    record, reasons = normalize_route_innovation(
        {
            "kind": BIOCATALYTIC_STEP,
            "chemical_step_equivalent_count": 1,
            "enzyme_class": "ketoreductase",
            "selectivity_objective": "stereoselective carbonyl reduction",
        }
    )

    assert reasons == []
    assert record["kind"] == BIOCATALYTIC_STEP
    assert record["step_savings"] == 0


def test_null_like_chemenzy_enzyme_label_is_not_an_enzyme_hypothesis() -> None:
    record, reasons = normalize_route_innovation(
        {
            "kind": BIOCATALYTIC_STEP,
            "enzyme_class": "None",
            "selectivity_objective": "match proposed stereochemistry",
        }
    )

    assert record == {}
    assert reasons == ["biocatalytic_superstep_enzyme_hypothesis_missing"]

    assert route_innovation_from_chemenzy_step(
        {"raw_backend_metadata": {"enzyme_class": None}}
    ) == {}
    assert route_innovation_from_chemenzy_step(
        {"raw_backend_metadata": {"enzyme_class": "None"}}
    ) == {}


def test_route_membership_does_not_duplicate_one_innovation() -> None:
    first, first_reasons = normalize_route_innovation(
        {
            "kind": BIOCATALYTIC_STEP,
            "route_family_id": "family:a",
            "enzyme_class": "ketoreductase",
            "selectivity_objective": "set the product alcohol stereocentre",
        }
    )
    second, second_reasons = normalize_route_innovation(
        {
            "kind": BIOCATALYTIC_STEP,
            "route_family_id": "family:b",
            "enzyme_class": "ketoreductase",
            "selectivity_objective": "set the product alcohol stereocentre",
        }
    )

    assert first_reasons == second_reasons == []
    assert first["innovation_id"] == second["innovation_id"]
    merged = merge_route_innovations([first], [second])
    assert len(merged) == 1
    assert merged[0]["route_family_ids"] == ["family:a", "family:b"]


def test_mechanism_extrapolation_is_one_hop_anchor_bound_and_falsifiable() -> None:
    record, reasons = normalize_route_innovation(
        {
            "kind": MECHANISM_EXTRAPOLATION,
            "anchor_source_refs": ["doi:10.1000/reported-intermediate"],
            "hypothesis_depth": 1,
            "mechanistic_rationale": (
                "The reported allylic alcohol should undergo a directed oxidation "
                "without changing the steroid skeleton."
            ),
            "falsifiable_checks": [
                "LCMS mass balance",
                "confirm alkene retention by NMR",
            ],
        }
    )

    assert reasons == []
    assert record["reported_in_anchor_source"] is False
    assert record["unvalidated_proof_ceiling"] == "L1_structural_materialized"

    rejected, reasons = normalize_route_innovation(
        {
            "kind": MECHANISM_EXTRAPOLATION,
            "anchor_source_refs": ["doi:10.1000/reported-intermediate"],
            "hypothesis_depth": 2,
            "mechanistic_rationale": "A sufficiently explicit mechanistic rationale.",
            "falsifiable_checks": ["LCMS"],
        }
    )
    assert rejected == {}
    assert "mechanism_extrapolation_must_be_one_hop" in reasons


def test_generic_reaction_proof_does_not_validate_an_enzyme_label() -> None:
    option = _enzyme_option()
    plain_proof = {"accepted": True, "proof_digest": "plain"}

    blocked = innovation_proof_gate([option], [plain_proof])
    assert blocked["required"] is True
    assert blocked["accepted"] is False
    assert blocked["reasons"] == ["biocatalysis_validation_missing"]

    proof = {
        "accepted": True,
        "biocatalysis_validation": {
            "accepted": True,
            "innovation_id": option["innovation_id"],
        },
    }
    accepted = innovation_proof_gate([option], [proof])
    assert accepted["accepted"] is True
    assert accepted["validated_innovation_ids"] == [option["innovation_id"]]


def test_unvalidated_enzyme_superstep_stays_visible_but_cannot_close_route() -> None:
    option = _enzyme_option()
    graph = {
        "target_molecule_id": "m:target",
        "edges": {
            "edge:enzyme": {
                "product_molecule_id": "m:target",
                "precursor_molecule_ids": ["m:leaf"],
                "route_innovations": [option],
            }
        },
    }
    edge_proofs = {
        "edge:enzyme": {
            "edge_id": "edge:enzyme",
            "achieved_level": 2,
            "accepted": True,
            "independent_source_groups": [],
            "conflict_ids": [],
            "innovation_proof_gate": {
                "required": True,
                "accepted": False,
                "validated_innovation_ids": [],
                "generic_validation": False,
            },
        }
    }
    route = build_route_candidate(
        graph,
        family_id="family:enzyme",
        family={"strategy": "replace five chemical operations with one enzyme"},
        variant=RouteSubroute(
            edge_ids=frozenset({"edge:enzyme"}),
            leaf_ids=frozenset({"m:leaf"}),
            module_selections=(),
        ),
        edge_proofs=edge_proofs,
        leaf_proof_cache={"m:leaf": {"molecule_id": "m:leaf", "accepted": True}},
        policy=ProofPolicy(
            minimum_edge_proof_level=2,
            minimum_independent_source_groups=0,
            require_stock_for_every_selected_leaf=False,
            stock_boundary="benchmark_search",
        ),
    )

    assert route["complete"] is False
    assert route["all_edges_proven"] is False
    assert route["minimum_edge_proof_level"] == 1
    assert route["physical_step_count"] == 1
    assert route["chemical_step_equivalent_count"] == 5
    assert route["net_step_savings"] == 4
    assert route["unvalidated_biocatalytic_edge_ids"] == ["edge:enzyme"]


def test_route_summary_selects_only_the_matching_family_execution_option() -> None:
    enzyme = _enzyme_option(family_id="family:enzyme")
    mechanism, reasons = normalize_route_innovation(
        {
            "kind": MECHANISM_EXTRAPOLATION,
            "route_family_id": "family:mechanism",
            "anchor_edge_ids": ["edge:reported"],
            "mechanistic_rationale": "A source-anchored redox relay provides one testable next step.",
            "falsifiable_checks": ["detect the predicted intermediate"],
        }
    )
    assert reasons == []
    graph = {
        "edges": {
            "edge:shared": {
                "route_innovations": [enzyme, mechanism],
            }
        }
    }

    summary = route_innovation_summary(
        graph,
        ["edge:shared"],
        route_family_id="family:mechanism",
    )
    assert summary["biocatalytic_superstep_count"] == 0
    assert summary["mechanism_extrapolation_count"] == 1
    assert summary["chemical_step_equivalent_count"] == 1


def test_biocatalytic_step_contract_counts_without_legacy_innovation_annotation() -> None:
    graph = {
        "edges": {
            "edge:p450": {
                "route_innovations": [],
                "biocatalytic_steps": [
                    {
                        "step_id": "step:p450",
                        "content_sha256": "a" * 64,
                        "authority_scope": "model_proposed_execution_hypothesis",
                        "step_accounting": {
                            "physical_operation_count": 1,
                            "chemical_step_equivalent_count": None,
                            "net_step_savings": None,
                        },
                    }
                ],
            }
        }
    }

    summary = route_innovation_summary(graph, ["edge:p450"])

    assert summary["biocatalytic_step_count"] == 1
    assert summary["biocatalytic_edge_ids"] == ["edge:p450"]
    assert summary["chemical_step_equivalent_count"] == 1
    assert summary["net_step_savings"] == 0
    assert summary["selected_options"][0][
        "from_biocatalytic_step_contract"
    ] is True
    assert summary["semantics"][
        "biocatalytic_step_contract_counts_physical_execution_only"
    ] is True


def test_canonical_ingestion_preserves_innovation_from_hypothesis_to_edge(
    tmp_path: Path,
) -> None:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="route-innovation-integration",
            target_name="ethyl acetate",
            target_smiles="CCOC(C)=O",
            created_at="2026-07-15T00:00:00Z",
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
    planned = store.apply(
        CanonicalIngestionBatch(
            global_plans=(
                {
                    "route_families": [
                        {
                            "route_family_id": "enzyme-shortcut",
                            "strategy": "biocatalytic route compression",
                        }
                    ],
                    "multi_step_skeletons": [
                        {
                            "skeleton_id": "skeleton:enzyme",
                            "route_family_id": "enzyme-shortcut",
                            "steps": [
                                {
                                    "step_id": "enzyme:1",
                                    "product_smiles": "CCOC(C)=O",
                                    "precursor_smiles": ["CCO", "CC(=O)Cl"],
                                    "route_innovation": {
                                        "kind": BIOCATALYTIC_SUPERSTEP,
                                        "chemical_step_equivalent_count": 3,
                                        "enzyme_class": "acyltransferase",
                                        "selectivity_objective": (
                                            "chemoselective ester formation"
                                        ),
                                    },
                                }
                            ],
                        }
                    ],
                },
            )
        ),
        idempotency_key="plan",
    )["graph"]
    hypothesis = next(iter(planned["hypotheses"].values()))
    assert hypothesis["route_innovations"][0]["step_savings"] == 2

    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    results = tuple(
        runtime.execute(command)
        for command in store.frontier_materialization_commands()
    )
    materialized = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialize",
    )["graph"]
    edge = next(iter(materialized["edges"].values()))

    assert edge["route_innovations"][0]["kind"] == BIOCATALYTIC_SUPERSTEP
    assert edge["route_innovations"][0]["not_reaction_proof"] is True
    assert materialized["route_families"]
