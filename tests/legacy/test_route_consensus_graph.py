from __future__ import annotations

from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.domain import (
    EvidenceClaim,
    MoleculeIdentity,
    ReactionCandidateEnvelope,
    ReactionHyperedge,
    canonicalize_smiles,
)
from cascade_planner.legacy.routes_runtime.graph import (
    assemble_route_consensus_graph,
    make_route_consensus_expansion,
    route_consensus_frontier_records,
    select_route_consensus_frontier,
)
from cascade_planner.routes.overlay import build_route_hypergraph_v2_overlay
from cascade_planner.legacy.harness_runtime.route_forest import compile_explored_route_forest, render_route_forest_html


def _candidate(
    candidate_id: str,
    *,
    product: str,
    precursors: list[str],
    source_ref: str,
    channel: str = "codex_strategy",
) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "reaction_family": "test disconnection",
        "transformation_rationale": "bounded graph test",
        "source_channel": channel,
        "source_refs": [source_ref],
        "evidence_refs": [],
        "evidence_level": "literature_exact" if channel == "codex_literature" else "model_only",
        "confidence": "medium",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": [],
        "required_validation": ["forward_reconstruction"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def _expansion(
    product: str,
    precursor: str,
    *,
    source_ref: str,
    depth: int,
    candidate_id: str,
) -> dict:
    channel = "codex_literature" if source_ref.startswith("doi:") else "codex_strategy"
    consensus = fuse_route_candidates(
        [
            _candidate(
                candidate_id,
                product=product,
                precursors=[precursor],
                source_ref=source_ref,
                channel=channel,
            )
        ],
        target_smiles=product,
    )
    return make_route_consensus_expansion(
        consensus,
        requested_product_smiles=product,
        consensus_ref=f"consensus:{candidate_id}",
        agent_run_ref=f"agent:{candidate_id}",
        depth=depth,
    )


def test_overlay_merges_identical_chemistry_without_losing_source_support() -> None:
    graph = {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "two-source-same-edge",
        "root_node_id": "target",
        "nodes": [
            {"node_id": "target", "smiles": "CCO"},
            {"node_id": "precursor", "smiles": "CC"},
        ],
        "steps": [
            {
                "step_id": "paper-step",
                "product_node_id": "target",
                "product_smiles": "CCO",
                "precursor_node_ids": ["precursor"],
                "precursor_smiles": ["CC"],
                "reaction_family": "hydration",
                "source_channels": ["literature_exact"],
                "independent_support_groups": ["doi:paper"],
                "proposal_ids": ["paper-binding"],
                "rank_score": 0.8,
            },
            {
                "step_id": "patent-step",
                "product_node_id": "target",
                "product_smiles": "CCO",
                "precursor_node_ids": ["precursor"],
                "precursor_smiles": ["CC"],
                "reaction_family": "hydration",
                "source_channels": ["literature_exact"],
                "independent_support_groups": ["patent:example"],
                "proposal_ids": ["patent-binding"],
                "rank_score": 0.9,
            },
        ],
        "route_hypotheses": [],
    }

    overlay = build_route_hypergraph_v2_overlay(graph)

    assert overlay["validation"] == {"valid": True, "errors": []}
    assert len(overlay["reaction_hyperedges"]) == 1
    merged = overlay["reaction_hyperedges"][0]
    assert merged["independent_support_groups"] == [
        "doi:paper",
        "patent:example",
    ]
    assert len(merged["candidate_envelope_ids"]) == 2
    assert merged["rank_score"] == 0.9
    assert overlay["v1_id_map"]["step_ids"]["paper-step"] == merged[
        "hyperedge_id"
    ]
    assert overlay["v1_id_map"]["step_ids"]["patent-step"] == merged[
        "hyperedge_id"
    ]


def test_two_one_step_expansions_assemble_into_forward_multistep_route() -> None:
    root = _expansion("OCC", "CC=O", source_ref="doi:10.1000/root", depth=0, candidate_id="root")
    middle = _expansion(
        "CC=O",
        "C=CO",
        source_ref="doi:10.1000/middle",
        depth=1,
        candidate_id="middle",
    )

    graph = assemble_route_consensus_graph(
        [root, middle],
        case_id="ethanol",
        target_smiles="CCO",
        max_depth=3,
    )

    assert graph["schema_version"] == "route_consensus_graph.v1"
    assert graph["has_hypotheses"] is True
    assert len(graph["nodes"]) == 3
    assert len(graph["steps"]) == 2
    route = graph["route_hypotheses"][0]
    assert len(route["retrosynthetic_step_ids"]) == 2
    assert route["forward_step_ids"] == list(reversed(route["retrosynthetic_step_ids"]))
    assert len(route["forward_dependencies"]) == 1
    assert route["solved"] is False
    assert route["executable"] is False

    steps = {row["step_id"]: row for row in graph["steps"]}
    root_step = next(row for row in steps.values() if row["product_smiles"] == "CCO")
    middle_step = next(row for row in steps.values() if row["product_smiles"] == "CC=O")
    assert root_step["source_refs"] == ["doi:10.1000/root"]
    assert middle_step["source_refs"] == ["doi:10.1000/middle"]
    assert "consensus:root" not in root_step["source_refs"]

    overlay = graph["v2_overlay"]
    assert overlay["schema_version"] == "route_hypergraph_overlay.v2"
    assert overlay["source_graph_schema_version"] == "route_consensus_graph.v1"
    assert overlay["validation"] == {"valid": True, "errors": []}
    assert len(overlay["reaction_hyperedges"]) == 2
    assert len(overlay["route_variants"]) == 1
    assert len(graph["route_neighborhoods"]) == 2
    assert overlay["content_hash"].startswith("sha256:")


def test_v2_domain_records_are_stable_content_addressed_and_correlate_codex_roles() -> None:
    product = MoleculeIdentity("OCC", names=("ethanol",))
    equivalent_product = MoleculeIdentity("CCO", names=("ethyl alcohol",))
    precursor = MoleculeIdentity("CC=O")
    strategy = EvidenceClaim(
        source_channel="codex_strategy",
        support_group="codex_model",
        evidence_level="model_only",
        confidence="medium",
        candidate_id="strategy",
    )
    critic = EvidenceClaim(
        source_channel="codex_critic",
        support_group="codex_model",
        evidence_level="model_only",
        confidence="medium",
        candidate_id="critic",
    )
    envelope = ReactionCandidateEnvelope(
        product=product,
        precursors=(precursor,),
        reaction_family="reduction",
        source_candidate_ids=("strategy", "critic"),
        evidence_claims=(strategy, critic),
    )
    edge = ReactionHyperedge(
        product=product,
        precursors=(precursor,),
        candidate_envelope_ids=(envelope.envelope_id,),
        evidence_claim_ids=(strategy.claim_id, critic.claim_id),
        source_channels=("codex_strategy", "codex_critic"),
        independent_support_groups=("codex_model",),
        reaction_families=("reduction",),
        rank_score=0.5,
    )

    assert product.molecule_id == equivalent_product.molecule_id
    assert product.content_hash != equivalent_product.content_hash
    assert strategy.validate() == ()
    assert critic.validate() == ()
    assert envelope.validate() == ()
    assert edge.validate() == ()
    assert edge.to_dict()["independent_support_groups"] == ["codex_model"]
    assert edge.to_dict()["content_hash"].startswith("sha256:")

    invalid_codex_group = EvidenceClaim(
        source_channel="codex_literature",
        support_group="literature:doi:10.1000/model-claim",
        evidence_level="analogy",
        confidence="low",
    )
    assert invalid_codex_group.validate() == ("codex_claim_has_independent_support_group",)


def test_codex_self_reported_authority_cannot_increase_rank_before_host_binding() -> None:
    promoted = _candidate(
        "producer-promoted",
        product="CCO",
        precursors=["CC=O"],
        source_ref="doi:10.1000/unbound-model-claim",
        channel="codex_literature",
    )
    promoted["evidence_level"] = "validated"
    promoted["confidence"] = "high"
    baseline = _candidate(
        "producer-baseline",
        product="CCO",
        precursors=["CC=O"],
        source_ref="source:model",
        channel="codex_strategy",
    )
    baseline["evidence_level"] = "model_only"
    baseline["confidence"] = "low"

    promoted_consensus = fuse_route_candidates([promoted], target_smiles="CCO")
    baseline_consensus = fuse_route_candidates([baseline], target_smiles="CCO")
    promoted_proposal = promoted_consensus["proposals"][0]
    source = promoted_proposal["source_records"][0]

    assert promoted_proposal["rank_score"] == baseline_consensus["proposals"][0][
        "rank_score"
    ]
    assert promoted_proposal["status"] == "model_hypothesis"
    assert source["producer_evidence_level"] == "validated"
    assert source["producer_confidence"] == "high"
    assert source["authority_evidence_level"] == "model_only"
    assert source["authority_confidence"] == "low"
    assert source["authority_bound"] is False
    assert {row["reason"] for row in source["normalization_records"]} >= {
        "unbound_codex_producer_cannot_set_evidence_authority",
        "unbound_codex_producer_cannot_set_confidence_authority",
    }
    assert source["acquisition_hints"] == [
        {
            "schema_version": "route_candidate_acquisition_hint.v1",
            "hint_type": "host_evidence_binding",
            "reason": "codex_producer_metadata_is_advisory_only",
            "required_binding": (
                "deterministic_reaction_validation_or_trusted_source_detail_step"
            ),
        }
    ]

    graph = assemble_route_consensus_graph(
        [
            make_route_consensus_expansion(
                promoted_consensus,
                requested_product_smiles="CCO",
                consensus_ref="consensus:producer-promoted",
                agent_run_ref="agent:producer-promoted",
                depth=0,
            )
        ],
        case_id="producer-authority",
        target_smiles="CCO",
        max_depth=2,
    )
    step = graph["steps"][0]
    assert step["authority_policy"] == "host_derived"
    assert step["authority_evidence_level"] == "model_only"
    assert step["producer_evidence_levels"] == ["validated"]
    assert step["producer_confidences"] == ["high"]
    assert step["acquisition_hints"][0]["hint_type"] == "host_evidence_binding"


def test_invalid_producer_enums_are_explicit_and_trusted_host_evidence_survives() -> None:
    malformed = _candidate(
        "malformed-metadata",
        product="CCO",
        precursors=["CC=O"],
        source_ref="source:model",
        channel="codex_strategy",
    )
    malformed["evidence_level"] = "oracle_grade"
    malformed["confidence"] = "absolutely_certain"
    malformed_source = fuse_route_candidates(
        [malformed],
        target_smiles="CCO",
    )["proposals"][0]["source_records"][0]

    assert malformed_source["producer_evidence_level_raw"] == "oracle_grade"
    assert malformed_source["producer_confidence_raw"] == "absolutely_certain"
    assert malformed_source["producer_evidence_level"] == "model_only"
    assert malformed_source["producer_confidence"] == "low"
    invalid_fields = {
        row["field"]
        for row in malformed_source["normalization_records"]
        if row["reason"] == "invalid_enum_value"
    }
    assert invalid_fields == {"producer_evidence_level", "producer_confidence"}
    assert {
        row["field"]
        for row in malformed_source["acquisition_hints"]
        if row["hint_type"] == "producer_enum_correction"
    } == invalid_fields

    trusted = _candidate(
        "host-bound-literature",
        product="CCO",
        precursors=["CC=O"],
        source_ref="doi:10.1000/host-bound",
        channel="literature_exact",
    )
    trusted["evidence_level"] = "literature_exact"
    trusted["confidence"] = "high"
    trusted_proposal = fuse_route_candidates(
        [trusted],
        target_smiles="CCO",
        allow_trusted_literature_exact_evidence=True,
    )["proposals"][0]
    trusted_source = trusted_proposal["source_records"][0]

    assert trusted_source["authority_evidence_level"] == "literature_exact"
    assert trusted_source["authority_confidence"] == "high"
    assert trusted_source["authority_bound"] is True
    assert trusted_source["authority_basis"] == "trusted_literature_adapter"
    assert trusted_proposal["status"] == "evidence_backed_draft"


def test_unexpanded_precursor_is_selected_as_next_codex_frontier() -> None:
    graph = assemble_route_consensus_graph(
        [_expansion("CCO", "CC=O", source_ref="source:model", depth=0, candidate_id="root")],
        case_id="ethanol",
        target_smiles="CCO",
        max_depth=3,
    )

    frontier = select_route_consensus_frontier(graph)

    assert len(frontier) == 1
    assert frontier[0]["target_smiles"] == "CC=O"
    assert frontier[0]["depth"] == 1
    assert frontier[0]["reason"] == "unexpanded"


def test_route_graph_preserves_duplicate_precursor_stoichiometry() -> None:
    consensus = fuse_route_candidates(
        [
            _candidate(
                "homocoupling",
                product="CC",
                precursors=["C", "C"],
                source_ref="source:model",
            )
        ],
        target_smiles="CC",
    )
    expansion = make_route_consensus_expansion(
        consensus,
        requested_product_smiles="CC",
        consensus_ref="consensus:homocoupling",
        agent_run_ref="agent:homocoupling",
        depth=0,
    )

    graph = assemble_route_consensus_graph(
        [expansion],
        case_id="homocoupling",
        target_smiles="CC",
        max_depth=2,
    )

    assert graph["steps"][0]["precursor_smiles"] == ["C", "C"]
    assert len(graph["steps"][0]["precursor_node_ids"]) == 2
    assert graph["steps"][0]["signature"].endswith("<-C.C")


def test_frontier_selection_covers_alternatives_beyond_route_hypothesis_limit() -> None:
    # Keep every synthetic alternative chemically admissible under the shared
    # front-door element-inventory filter.  The former bare carbon chains all
    # omitted the product oxygen, so they were (correctly) rejected before the
    # route-hypothesis truncation behaviour under test could be exercised.
    precursors = ["CCO" + "C" * length for length in range(1, 31)]
    consensus = fuse_route_candidates(
        [
            _candidate(
                f"alternative-{index}",
                product="CCO",
                precursors=[precursor],
                source_ref=f"source:alternative-{index}",
            )
            for index, precursor in enumerate(precursors, start=1)
        ],
        target_smiles="CCO",
    )
    graph = assemble_route_consensus_graph(
        [
            make_route_consensus_expansion(
                consensus,
                requested_product_smiles="CCO",
                consensus_ref="consensus:thirty-alternatives",
                agent_run_ref="agent:thirty-alternatives",
                depth=0,
            )
        ],
        case_id="thirty-alternatives",
        target_smiles="CCO",
        max_depth=3,
    )

    assert len(graph["route_hypotheses"]) == 24
    assert graph["truncation"]["route_hypotheses_truncated"] is True
    frontier = select_route_consensus_frontier(graph, limit=100)
    assert len(frontier) == 30
    assert {row["target_smiles"] for row in frontier} == {
        canonicalize_smiles(value) for value in precursors
    }
    assert {
        row["target_smiles"]
        for row in route_consensus_frontier_records(graph)
        if row["reason"] == "unexpanded"
    } == {canonicalize_smiles(value) for value in precursors}


def test_cycle_is_audited_and_cut_from_route_enumeration() -> None:
    graph = assemble_route_consensus_graph(
        [
            _expansion("CCO", "CC=O", source_ref="source:root", depth=0, candidate_id="root"),
            _expansion("CC=O", "OCC", source_ref="source:cycle", depth=1, candidate_id="cycle"),
        ],
        case_id="cycle",
        target_smiles="CCO",
        max_depth=4,
    )

    assert graph["cycles"]
    assert len(graph["route_hypotheses"]) == 1
    assert len(graph["route_hypotheses"][0]["retrosynthetic_step_ids"]) == 1
    cycle_frontiers = route_consensus_frontier_records(graph)
    assert [row["target_smiles"] for row in cycle_frontiers] == ["CC=O"]
    assert cycle_frontiers[0]["reason"] == "cycle_cut"
    assert graph["semantics"]["advisory_only"] is True


def test_stereoisomeric_intermediates_do_not_join() -> None:
    root = _expansion("CCO", "C[C@H](O)F", source_ref="source:root", depth=0, candidate_id="root")
    wrong_stereo = _expansion(
        "C[C@@H](O)F",
        "CC(=O)F",
        source_ref="source:wrong-stereo",
        depth=1,
        candidate_id="wrong",
    )

    graph = assemble_route_consensus_graph(
        [root, wrong_stereo],
        case_id="stereo",
        target_smiles="CCO",
        max_depth=3,
    )

    route = graph["route_hypotheses"][0]
    assert len(route["retrosynthetic_step_ids"]) == 1
    assert route["frontier"][0]["reason"] == "unexpanded"


def test_route_forest_projects_multistep_graph_in_forward_order_with_step_scoped_sources() -> None:
    graph = assemble_route_consensus_graph(
        [
            _expansion("CCO", "CC=O", source_ref="doi:10.1000/root", depth=0, candidate_id="root"),
            _expansion(
                "CC=O",
                "C=CO",
                source_ref="doi:10.1000/middle",
                depth=1,
                candidate_id="middle",
            ),
        ],
        case_id="ethanol",
        target_smiles="CCO",
        max_depth=3,
    )
    forest = compile_explored_route_forest(
        {
            "case_id": "ethanol",
            "target_profile": {"target_name": "ethanol", "target_smiles": "CCO"},
            "route_consensus_graph": graph,
            "artifact_refs": {"route_consensus_graph": "D:/runs/ethanol/route_consensus_graph.json"},
        }
    )

    branch = next(row for row in forest["branches"] if row["kind"] == "route_consensus_graph")
    steps = {row["step_id"]: row for row in forest["steps"]}
    rendered = [steps[step_id] for step_id in branch["step_ids"]]
    assert [row["graph_step_id"] for row in rendered] == graph["route_hypotheses"][0]["forward_step_ids"]
    assert rendered[0]["source_refs"] == ["doi:10.1000/middle"]
    assert rendered[1]["source_refs"] == ["doi:10.1000/root"]
    assert all(row["independent_support_groups"] == ["codex_model"] for row in rendered)
    assert all(row["independent_source_count"] == 1 for row in rendered)
    assert all(row["multi_source"] is False for row in rendered)
    assert "D:/runs/ethanol/route_consensus_graph.json" not in rendered[0]["source_refs"]
    assert branch["route_level_source_refs"] == ["D:/runs/ethanol/route_consensus_graph.json"]
    assert branch["advisory_only"] is True
    assert branch["solved"] is False
    assert branch["executable"] is False
    assert forest["route_consensus_graph"]["route_count"] == 1
    assert "Codex multi-step hypothesis" in render_route_forest_html(forest)
