from __future__ import annotations

from pathlib import Path

import cascade_planner.legacy.harness_runtime.codex_edge_verification as edge_evidence
from cascade_planner.legacy.harness_runtime.route_forest import (
    _RouteForestCompiler,
    _validated_edge_evidence_projection,
)
from cascade_planner.harness.reaction_step_verifier import canonical_reaction_digest


def _binding_set(*groups: str) -> dict:
    reaction_digest = canonical_reaction_digest("CC=O", ["CCO"])
    bindings = []
    for index, group in enumerate(groups, start=1):
        precedent = {
            "schema_version": "trusted_precedent_binding.v1",
            "accepted": True,
            "authority": "human_curator",
            "authority_id": "fixture",
            "binding_id": f"binding:{index}",
            "reaction_digest": reaction_digest,
            "source_ref": group,
        }
        materialized_evidence = [
            {
                "source_ref": group,
                "document_id": f"document:{index}",
                "source_pdf_sha256": f"{index:x}" * 64,
                "page_number": index,
                "image_sha256": f"{index + 8:x}" * 64,
            }
        ]
        bindings.append({
            "schema_version": "edge_evidence_binding.v1",
            "binding_id": f"binding:{index}",
            "row_id": f"row:{index}",
            "row_sha256": f"{index + 1:x}" * 64,
            "source_ref": group,
            "source_identity": {"source_ref": group},
            "trusted": True,
            "status": "trusted",
            "independent_source_group": group,
            "authority": "human_curator",
            "authority_id": "fixture",
            "reaction_digest": reaction_digest,
            "trusted_precedent_binding": precedent,
            "trusted_precedent_binding_sha256": edge_evidence._digest(precedent),
            "materialized_evidence": materialized_evidence,
            "materialized_evidence_sha256": edge_evidence._digest(materialized_evidence),
            "condition_sha256": edge_evidence._digest([f"condition:{index}"]),
            "proof_level": "L3_precedent_supported",
            "reasons": [],
        })
    payload = {
        "schema_version": "edge_evidence_binding_set.v1",
        "reaction_digest": reaction_digest,
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "binding_count": len(bindings),
        "trusted_binding_count": len(bindings),
        "independent_trusted_source_groups": list(groups),
        "independent_trusted_source_group_count": len(set(groups)),
        "corroborated": len(set(groups)) >= 2,
        "primary_binding_id": str((bindings or [{}])[0].get("binding_id") or ""),
        "primary_row_sha256": str((bindings or [{}])[0].get("row_sha256") or ""),
        "trusted_condition_variant_count": len(bindings),
        "condition_variants_require_review": len(bindings) > 1,
        "bindings": bindings,
    }
    payload["content_sha256"] = edge_evidence._digest(payload)
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
    assert vector["trusted_source_group_count"] == 2
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


def test_computational_groups_never_count_as_trusted_or_corroborated() -> None:
    compiler = _compiler()
    step = _step()
    step["independent_support_groups"] = [
        "computational:chem_enzy",
        "computational:stock",
    ]
    compiler.steps = {"step:test": step}
    compiler.branches = [
        {
            "branch_id": "branch:test",
            "kind": "route_consensus_graph",
            "step_ids": ["step:test"],
        }
    ]

    compiler._finalize_trust_vectors()
    step_trust = compiler.steps["step:test"]["trust_vector"]
    branch_trust = compiler.branches[0]["trust_vector"]

    assert step_trust["support_group_count"] == 2
    assert step_trust["trusted_source_group_count"] == 0
    assert branch_trust["min_trusted_source_group_count_across_steps"] == 0
    assert branch_trust["corroborated_edge_count"] == 0
    assert branch_trust["all_edges_corroborated"] is False
    assert step_trust["visual_encoding"]["width"] == 1.5


def test_ui_branch_ranking_is_lexicographic_weakest_first() -> None:
    script = Path("cascade_planner/harness/route_forest_ui/script.js").read_text(
        encoding="utf-8"
    )

    assert "function branchDisplayRank(lane)" in script
    assert "function compareBranchDisplay(left, right)" in script
    rank_body = script.split("function branchDisplayRank(lane)", 1)[1].split(
        "function compareBranchDisplay", 1
    )[0]
    assert rank_body.index("min_trusted_source_group_count_across_steps") < rank_body.index(
        "corroborated_edge_count"
    )
    assert ".sort(compareBranchDisplay)" in script


def test_route_forest_projection_requires_matching_host_report_digest() -> None:
    binding_set = _binding_set("doi:10.1000/one")
    reaction_digest = binding_set["reaction_digest"]
    candidate = {
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "edge_evidence_binding_set": binding_set,
    }
    step_proof = {
        "reaction_digest": reaction_digest,
        "validator_version": edge_evidence.REACTION_STEP_VERIFIER_VERSION,
    }
    step_proof["proof_digest"] = edge_evidence._digest(step_proof)
    report = {
        "schema_version": edge_evidence.CODEX_EDGE_VERIFICATION_SCHEMA,
        "reaction_step_verifier_version": edge_evidence.REACTION_STEP_VERIFIER_VERSION,
        "edge_count": 1,
        "edge_verifications": [
            {
                "step_id": "step:test",
                "product_smiles": "CC=O",
                "reactant_smiles": ["CCO"],
                "materialized_candidate": candidate,
                "step_proof": step_proof,
                "edge_evidence_binding_set": binding_set,
            }
        ],
    }
    report["content_sha256"] = edge_evidence._digest(report)
    projection = edge_evidence.project_edge_evidence_binding_sets(report)
    graph = {
        "edge_evidence_binding_sets": projection,
        "codex_edge_verification_summary": {
            "content_sha256": report["content_sha256"],
            "edge_count": 1,
        },
    }

    assert list(_validated_edge_evidence_projection(graph)) == [reaction_digest]
    graph["codex_edge_verification_summary"]["content_sha256"] = "0" * 64
    assert _validated_edge_evidence_projection(graph) == {}
