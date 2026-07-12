from __future__ import annotations

from cascade_planner.harness.route_forest import (
    _RouteForestCompiler,
    _canonical_json_sha256,
)


def _binding_set(*groups: str) -> dict:
    bindings = [
        {
            "schema_version": "edge_evidence_binding.v1",
            "binding_id": f"binding:{index}",
            "trusted": True,
            "independent_source_group": group,
        }
        for index, group in enumerate(groups, start=1)
    ]
    payload = {
        "schema_version": "edge_evidence_binding_set.v1",
        "reaction_digest": "f" * 64,
        "trusted_binding_count": len(bindings),
        "independent_trusted_source_groups": list(groups),
        "corroborated": len(set(groups)) >= 2,
        "bindings": bindings,
    }
    payload["content_sha256"] = _canonical_json_sha256(payload)
    return payload


def _compiler() -> _RouteForestCompiler:
    compiler = _RouteForestCompiler(
        {
            "case_id": "edge-evidence",
            "target_profile": {"target_name": "target", "target_smiles": "CC=O"},
        }
    )
    compiler.nodes = {
        "reactant": {"canonical_isomeric_smiles": "CCO"},
        "product": {"canonical_isomeric_smiles": "CC=O"},
    }
    return compiler


def _step(*, binding_set: dict | None = None, source_refs: list[str] | None = None) -> dict:
    return {
        "step_id": "step:test",
        "branch_id": "branch:test",
        "from_node_ids": ["reactant"],
        "to_node_ids": ["product"],
        "exactness": "model_hypothesis",
        "origin": "route_consensus_graph",
        "source_refs": list(source_refs or []),
        "independent_support_groups": [],
        "edge_evidence_binding_set": dict(binding_set or {}),
    }


def test_bare_external_citation_does_not_create_source_independence() -> None:
    compiler = _compiler()
    vector = compiler._step_trust_vector(
        _step(source_refs=["doi:10.1000/citation-only"]),
        {"kind": "route_consensus_graph"},
    )

    assert vector["support_group_count"] == 0
    assert vector["source_independence"] == 0.0


def test_edge_binding_set_controls_support_count_and_corroboration() -> None:
    compiler = _compiler()
    vector = compiler._step_trust_vector(
        _step(binding_set=_binding_set("doi:10.1000/one", "doi:10.1000/two")),
        {"kind": "route_consensus_graph"},
    )

    assert vector["support_group_count"] == 2
    assert vector["source_independence"] == 1.0
    assert vector["edge_evidence_binding_set"]["corroborated"] is True


def test_branch_multisource_summary_uses_weakest_edge() -> None:
    compiler = _compiler()
    compiler.steps = {
        "step:two": _step(
            binding_set=_binding_set("doi:10.1000/one", "doi:10.1000/two")
        ),
        "step:zero": {
            **_step(),
            "step_id": "step:zero",
        },
    }
    compiler.branches = [
        {
            "branch_id": "branch:test",
            "kind": "route_consensus_graph",
            "step_ids": ["step:two", "step:zero"],
        }
    ]

    compiler._finalize_trust_vectors()
    trust = compiler.branches[0]["trust_vector"]

    assert trust["min_trusted_source_group_count_across_steps"] == 0
    assert trust["corroborated_edge_count"] == 1
    assert trust["all_edges_corroborated"] is False
    assert trust["visual_encoding"]["width"] == 1.5
