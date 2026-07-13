from __future__ import annotations

from cascade_planner.application.blind_acceptance import (
    compile_blind_acceptance_report,
)
from cascade_planner.application.canonical_identity import (
    molecule_identity,
    reaction_edge_identity,
)


TARGET = "CCOC(C)=O"


def _step(step_id: str, product: str, precursors: list[str]) -> dict:
    return {
        "step_id": step_id,
        "product_smiles": product,
        "precursor_smiles": precursors,
    }


def _outcome(steps: list[dict], *, accepted: set[str]) -> dict:
    return {
        "status": "accepted",
        "proposal_audits": [
            {"proposal_id": step["step_id"], "accepted": step["step_id"] in accepted}
            for step in steps
        ],
        "plan": {
            "plan_id": "plan:blind",
            "mode": "initial_architecture",
            "route_families": [
                {"route_family_id": "family:one", "target_smiles": TARGET}
            ],
            "multi_step_skeletons": [
                {
                    "skeleton_id": "skeleton:one",
                    "route_family_id": "family:one",
                    "steps": steps,
                }
            ],
        },
    }


def _preflight() -> dict:
    return {
        "accepted": True,
        "case": {"acceptance": {"minimum_complete_routes": 1}},
    }


def _portfolio() -> dict:
    return {
        "proof_policy": {
            "minimum_independent_source_groups": 2,
            "stock_boundary": "benchmark_search",
        },
        "selected_routes": [],
        "accepted": False,
    }


def _validated_edge(product: str, precursors: list[str], **extra: object) -> tuple[str, dict]:
    edge_id, audit = reaction_edge_identity(product, precursors)
    assert edge_id and audit["accepted"] is True
    return edge_id, {
        "product_molecule_id": molecule_identity(product)[0],
        "precursor_molecule_ids": [molecule_identity(value)[0] for value in precursors],
        "reaction_proofs": [{"accepted": True}],
        **extra,
    }


def test_blind_acceptance_prunes_only_rejected_tail_expansion() -> None:
    root = _step("step:root", TARGET, ["CCO", "CC(=O)Cl"])
    rejected_tail = _step("step:tail", "CC(=O)Cl", ["CC(=O)O", "Cl"])
    edge_id, edge = _validated_edge(TARGET, root["precursor_smiles"])

    report = compile_blind_acceptance_report(
        preflight=_preflight(),
        director_outcomes=[_outcome([root, rejected_tail], accepted={"step:root"})],
        graph={"edges": {edge_id: edge}, "molecules": {}, "stock_observations": {}},
        portfolio=_portfolio(),
    )

    assert report["gates"]["B1_global_multi_route"] is True
    assert report["gates"]["B2_host_validated_routes"] is True
    assert report["routes"][0]["pruned_rejected_tail_step_ids"] == ["step:tail"]


def test_blind_acceptance_rejects_disconnected_route_after_middle_pruning() -> None:
    root = _step("step:root", TARGET, ["CCO", "CC(=O)Cl"])
    rejected_middle = _step("step:middle", "CC(=O)Cl", ["CC(=O)O", "Cl"])
    disconnected_upstream = _step("step:upstream", "CC(=O)O", ["CC=O", "O"])
    root_id, root_edge = _validated_edge(TARGET, root["precursor_smiles"])
    upstream_id, upstream_edge = _validated_edge(
        disconnected_upstream["product_smiles"],
        disconnected_upstream["precursor_smiles"],
    )

    report = compile_blind_acceptance_report(
        preflight=_preflight(),
        director_outcomes=[
            _outcome(
                [root, rejected_middle, disconnected_upstream],
                accepted={"step:root", "step:upstream"},
            )
        ],
        graph={
            "edges": {root_id: root_edge, upstream_id: upstream_edge},
            "molecules": {},
            "stock_observations": {},
        },
        portfolio=_portfolio(),
    )

    assert report["gates"]["B1_global_multi_route"] is False
    assert report["routes"] == []


def test_blind_acceptance_binds_a_validated_host_repair_to_original_step() -> None:
    original = _step("step:root", TARGET, ["CCO", "CC(=O)Cl"])
    original_id, _ = reaction_edge_identity(TARGET, original["precursor_smiles"])
    repair_id, repair = _validated_edge(
        TARGET,
        ["CCO", "CC(=O)Br"],
        origin_records=[
            {
                "origin_kind": "host_product_grounded_repair",
                "origin_ref": original_id,
            }
        ],
    )

    report = compile_blind_acceptance_report(
        preflight=_preflight(),
        director_outcomes=[_outcome([original], accepted={"step:root"})],
        graph={
            "edges": {
                original_id: {"reaction_proofs": [{"accepted": False}]},
                repair_id: repair,
            },
            "molecules": {},
            "stock_observations": {},
        },
        portfolio=_portfolio(),
    )

    assert report["gates"]["B2_host_validated_routes"] is True
    assert report["routes"][0]["edge_ids"] == [repair_id]
    assert report["routes"][0]["edge_replacements"] == [
        {
            "original_edge_id": original_id,
            "replacement_edge_id": repair_id,
            "reason": "host_product_grounded_repair",
        }
    ]


def test_repair_prunes_only_an_upstream_tail_that_no_longer_feeds_it() -> None:
    wrong_fragment = "COC"
    repaired_fragment = "CCO"
    root = _step("step:root", TARGET, ["CC(=O)O", wrong_fragment])
    obsolete_upstream = _step("step:tail", wrong_fragment, ["CO", "CBr"])
    original_id, original = _validated_edge(TARGET, root["precursor_smiles"])
    original["reaction_proofs"] = [{"accepted": False}]
    repair_id, repair = _validated_edge(
        TARGET,
        ["CC(=O)O", repaired_fragment],
        origin_records=[
            {
                "origin_kind": "host_product_grounded_repair",
                "origin_ref": original_id,
            }
        ],
    )
    tail_id, tail = _validated_edge(
        obsolete_upstream["product_smiles"],
        obsolete_upstream["precursor_smiles"],
    )

    report = compile_blind_acceptance_report(
        preflight=_preflight(),
        director_outcomes=[
            _outcome([root, obsolete_upstream], accepted={"step:root", "step:tail"})
        ],
        graph={
            "edges": {original_id: original, repair_id: repair, tail_id: tail},
            "molecules": {},
            "stock_observations": {},
        },
        portfolio=_portfolio(),
    )

    assert report["gates"]["B2_host_validated_routes"] is True
    assert report["routes"][0]["edge_ids"] == [repair_id]
    assert report["routes"][0]["pruned_after_replacement_edge_ids"] == [tail_id]
