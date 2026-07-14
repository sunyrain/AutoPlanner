from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cascade_planner.application.reaction_template_extraction import (
    apply_retro_template,
    extract_retro_template,
)
from cascade_planner.application.reaction_template_library import (
    load_patent_template_library,
    retrieve_patent_template_candidates,
    synchronize_patent_template_library,
)
from cascade_planner.harness.reaction_step_verifier import verify_reaction_step
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


ESTER_MAPPING = (
    "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][CH2:6][OH:7]>>"
    "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
)


def test_reaction_center_template_replays_source_and_transfers_to_close_analogue() -> None:
    extracted = extract_retro_template(ESTER_MAPPING)

    assert extracted["accepted"] is True
    assert extracted["source_replay"]["expected_precursor_smiles"] == [
        "CC(=O)Cl",
        "CCO",
    ]
    assert ["CC(=O)Cl", "CCCO"] in apply_retro_template(
        extracted["reaction_smarts"],
        "CCCOC(C)=O",
    )
    assert extract_retro_template("[CH3:1][OH:2]>>[CH3:1][OH:2]") == {
        "schema_version": "retrosynthetic_reaction_template.v1",
        "accepted": False,
        "reasons": ["mapped_reaction_has_no_bond_change"],
    }


def test_template_extractor_replays_multiple_reaction_families() -> None:
    examples = (
        ESTER_MAPPING,
        "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        (
            "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][NH2:6]>>"
            "[CH3:1][C:2](=[O:3])[NH:6][CH3:5]"
        ),
        (
            "[CH3:1][Br:2].[CH3:3][NH2:4]>>"
            "[CH3:1][NH:4][CH3:3]"
        ),
    )

    extracted = [extract_retro_template(value) for value in examples]

    assert all(value["accepted"] is True for value in extracted)
    assert len({value["template_id"] for value in extracted}) == len(examples)
    assert all(
        value["source_replay"]["matching_outcome_count"] >= 1
        for value in extracted
    )


def test_template_extractor_replays_reviewed_complex_route_edges() -> None:
    root = Path(__file__).resolve().parents[1] / "config" / "examples"
    expected_counts = {
        "nirmatrelvir_v4_replay_pack.json": 12,
        "artemisinin_v4_replay_pack.json": 2,
    }

    for filename, expected_count in expected_counts.items():
        document = json.loads((root / filename).read_text(encoding="utf-8"))
        mappings: list[str] = []
        _collect_mapped_reactions(document, mappings)
        extracted = [extract_retro_template(value) for value in mappings]

        assert len(mappings) == expected_count
        assert all(value["accepted"] is True for value in extracted)
        assert all(
            value["source_replay"]["matching_outcome_count"] >= 1
            for value in extracted
        )


def test_patent_library_learns_idempotently_and_retrieves_only_as_l0_proposal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "self-evo.json"
    graph = _learning_graph()

    first = synchronize_patent_template_library(path, graph)
    second = synchronize_patent_template_library(path, graph)
    library = load_patent_template_library(path)
    retrieval = retrieve_patent_template_candidates(
        path,
        graph={},
        target_smiles="CCCOC(C)=O",
    )
    exact_target_retrieval = retrieve_patent_template_candidates(
        path,
        graph={},
        target_smiles="CCOC(C)=O",
    )

    assert first["status"] == "completed"
    assert len(first["learned_template_ids"]) == 1
    assert second["status"] == "reused_or_empty"
    assert second["generation"] == first["generation"]
    template = next(iter(library["templates"].values()))
    assert template["example_count"] == 1
    assert template["maturity"] == "single_source_observed"
    assert template["source_refs"] == ["patent:US1234567A1"]
    assert retrieval["candidate_count"] == 1
    proposal = retrieval["proposals"][0]
    assert proposal["origin_kind"] == "self_evo_patent_template"
    assert proposal["precursor_smiles"] == ["CC(=O)Cl", "CCCO"]
    assert proposal["template_support"]["grants_no_scientific_authority"] is True
    assert retrieval["model_invocations"] == 0
    assert exact_target_retrieval["candidate_count"] == 0
    assert exact_target_retrieval["exact_example_exclusion_count"] == 1


def test_non_patent_failed_or_tampered_examples_never_learn(tmp_path: Path) -> None:
    for label, graph, reason in (
        ("paper", _learning_graph(source_ref="doi:10.1000/example"), "not_patent"),
        ("failed", _learning_graph(accepted=False), "proof_missing"),
        ("tampered", _learning_graph(tamper_exact=True), "digest_invalid"),
    ):
        result = synchronize_patent_template_library(tmp_path / f"{label}.json", graph)
        assert result["template_count"] == 0, reason
        assert result["learned_template_ids"] == []


def test_library_corruption_fails_closed_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text('{"schema_version":"patent_reaction_template_library.v1"}')
    before = path.read_bytes()

    sync = synchronize_patent_template_library(path, _learning_graph())
    retrieval = retrieve_patent_template_candidates(
        path,
        graph={},
        target_smiles="CCCOC(C)=O",
    )

    assert sync["status"] == "blocked_library_integrity"
    assert retrieval["status"] == "blocked_library_integrity"
    assert retrieval["candidate_count"] == 0
    assert path.read_bytes() == before


def test_reuse_feedback_is_digest_bound_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    learned = synchronize_patent_template_library(path, _learning_graph())
    template_id = learned["learned_template_ids"][0]
    graph = _reuse_graph(template_id)

    first = synchronize_patent_template_library(path, graph)
    second = synchronize_patent_template_library(path, graph)
    template = load_patent_template_library(path)["templates"][template_id]

    assert first["reuse_outcome_update_count"] == 1
    assert second["reuse_outcome_update_count"] == 0
    assert len(template["successful_edge_digests"]) == 1
    assert template["failed_edge_digests"] == []
    assert template["maturity"] == "reuse_validated"


def _learning_graph(
    *,
    source_ref: str = "patent:US1234567A1",
    accepted: bool = True,
    tamper_exact: bool = False,
) -> dict[str, Any]:
    product = "CCOC(C)=O"
    precursors = ["CCO", "CC(=O)Cl"]
    audit = audit_retrosynthetic_candidate(product, precursors)
    proof = verify_reaction_step(
        {
            "step_id": "edge:ester",
            "product_smiles": product,
            "reactant_smiles": precursors,
            "mapped_reaction_smiles": ESTER_MAPPING,
        }
    )
    if not accepted:
        proof["accepted"] = False
        proof["proof_level"] = "L2_mapping_consistent"
        proof["proof_digest"] = _digest(
            {key: value for key, value in proof.items() if key != "proof_digest"}
        )
    exact = {
        "schema_version": "exact_source_reaction_record.v1",
        "record_id": "exact:ester",
        "edge_digest": audit["edge_digest"],
        "relation_type": "exact",
        "authority_scope": "source_exact_structure_observation",
        "not_reaction_validation": True,
        "source_ref": source_ref,
        "source_binding_id": "source:external",
        "independence_group": "patent-family:one",
        "location_refs": ["Example 1"],
    }
    exact["content_sha256"] = _digest(exact)
    if tamper_exact:
        exact["location_refs"] = ["Example forged"]
    edge_id = f"edge:{audit['edge_digest']}"
    return {
        "edges": {
            edge_id: {
                "edge_id": edge_id,
                "edge_digest": audit["edge_digest"],
                "product_smiles": audit["product_smiles"],
                "precursor_smiles": audit["precursor_smiles_multiset"],
                "exact_record_ids": ["exact:ester"],
                "reaction_proofs": [proof],
                "origin_records": [],
            }
        },
        "exact_records": {"exact:ester": exact},
        "source_bindings": {},
        "source_aliases": {},
    }


def _reuse_graph(template_id: str) -> dict[str, Any]:
    product = "CCCOC(C)=O"
    precursors = ["CCCO", "CC(=O)Cl"]
    audit = audit_retrosynthetic_candidate(product, precursors)
    mapping = (
        "[CH3:1][C:2](=[O:3])[Cl:4]."
        "[CH3:5][CH2:6][CH2:8][OH:7]>>"
        "[CH3:1][C:2](=[O:3])[O:7][CH2:8][CH2:6][CH3:5]"
    )
    proof = verify_reaction_step(
        {
            "step_id": "edge:analogue",
            "product_smiles": product,
            "reactant_smiles": precursors,
            "mapped_reaction_smiles": mapping,
        }
    )
    assert proof["accepted"] is True
    edge_id = f"edge:{audit['edge_digest']}"
    return {
        "edges": {
            edge_id: {
                "edge_id": edge_id,
                "edge_digest": audit["edge_digest"],
                "product_smiles": audit["product_smiles"],
                "precursor_smiles": audit["precursor_smiles_multiset"],
                "exact_record_ids": [],
                "reaction_proofs": [proof],
                "origin_records": [
                    {
                        "origin_kind": "self_evo_patent_template",
                        "origin_ref": template_id,
                    }
                ],
            }
        },
        "exact_records": {},
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _collect_mapped_reactions(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        mapped = str(
            value.get("mapped_reaction_smiles")
            or value.get("atom_mapped_reaction_smiles")
            or ""
        )
        if mapped and mapped not in out:
            out.append(mapped)
        for child in value.values():
            _collect_mapped_reactions(child, out)
    elif isinstance(value, list):
        for child in value:
            _collect_mapped_reactions(child, out)
