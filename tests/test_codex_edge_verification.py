from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
from types import ModuleType

import cascade_planner.harness.codex_edge_verification as edge_verification
from cascade_planner.harness.codex_edge_verification import (
    verify_codex_consensus_graph,
)


def _graph(*, product: str, precursors: list[str]) -> dict:
    return {
        "schema_version": "route_consensus_graph.v1",
        "target_smiles": product,
        "steps": [
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": "step:1",
                "product_smiles": product,
                "precursor_smiles": precursors,
                "reaction_family": "model label is advisory",
                "conditions": [],
                "source_refs": [],
                "evidence_refs": [],
            }
        ],
    }


def _two_edge_graph() -> dict:
    graph = _graph(product="CC=O", precursors=["CCO"])
    graph["steps"].append(
        {
            "schema_version": "route_consensus_step.v1",
            "step_id": "step:2",
            "product_smiles": "CCC",
            "precursor_smiles": ["CC"],
            "reaction_family": "second advisory edge",
            "conditions": [],
            "source_refs": [],
            "evidence_refs": [],
        }
    )
    return graph


def test_unmapped_codex_edge_is_materialized_and_queued_for_proof() -> None:
    report = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        enable_optional_rxnmapper=False,
    )

    assert report["edge_count"] == 1
    assert report["materialized_edge_count"] == 1
    assert report["mapped_edge_count"] == 0
    assert report["reaction_validated_edge_count"] == 0
    edge = report["edge_verifications"][0]
    assert edge["proof_level"] == "L0_materialized"
    assert "atom_map_materialized_reaction" in edge["required_tasks"]
    assert edge["materialized_candidate"]["no_solved_claim"] is True


def test_injected_atom_mapper_can_promote_recognized_transform_but_not_stock() -> None:
    def mapper(reactions: list[str]) -> list[str]:
        assert reactions == ["CCO>>CC=O"]
        return ["[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"]

    report = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        atom_mapper=mapper,
        enable_optional_rxnmapper=False,
    )

    edge = report["edge_verifications"][0]
    assert report["mapped_edge_count"] == 1
    assert report["reaction_validated_edge_count"] == 1
    assert edge["proof_level"] == "L2_reaction_validated"
    assert edge["reaction_validated"] is True
    assert edge["proof_closed"] is False
    assert "audit_all_precursors_against_trusted_stock_provider" in edge["required_tasks"]


def test_atom_mapped_cut_and_glue_remains_advisory() -> None:
    def mapper(_: list[str]) -> list[str]:
        return ["[CH3:1][CH3:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"]

    report = verify_codex_consensus_graph(
        _graph(product="CCO", precursors=["CC", "O"]),
        atom_mapper=mapper,
        enable_optional_rxnmapper=False,
    )

    edge = report["edge_verifications"][0]
    assert edge["proof_level"] == "L2_mapping_consistent"
    assert edge["reaction_validated"] is False
    assert (
        "bind_deterministic_transform_or_exact_precedent"
        in edge["required_tasks"]
    )


def test_optional_rxnmapper_initialization_oserror_degrades_without_aborting_consensus(
    monkeypatch,
) -> None:
    class BrokenRXNMapper:
        def __init__(self) -> None:
            raise OSError("native runtime could not be loaded")

    fake_module = ModuleType("rxnmapper")
    fake_module.RXNMapper = BrokenRXNMapper
    monkeypatch.setitem(sys.modules, "rxnmapper", fake_module)
    edge_verification._cached_mapper_instance.cache_clear()

    report = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        enable_optional_rxnmapper=True,
    )

    assert report["edge_count"] == 1
    assert report["mapped_edge_count"] == 0
    assert report["reaction_validated_edge_count"] == 0
    assert report["atom_mapper"]["backend"] == "rxnmapper"
    assert report["atom_mapper"]["reasons"] == [
        "rxnmapper_initialization_error:OSError:native runtime could not be loaded"
    ]
    assert report["edge_verifications"][0]["proof_level"] == "L0_materialized"


def test_optional_rxnmapper_import_oserror_degrades_without_aborting_consensus(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def import_with_broken_optional_binary(name, *args, **kwargs):
        if name == "rxnmapper":
            raise OSError("optional binary is incompatible")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rxnmapper", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_with_broken_optional_binary)

    report = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        enable_optional_rxnmapper=True,
    )

    assert report["edge_count"] == 1
    assert report["mapped_edge_count"] == 0
    assert report["atom_mapper"]["reasons"] == [
        "rxnmapper_import_error:OSError:optional binary is incompatible"
    ]


def test_optional_rxnmapper_result_cache_is_bounded_lru_and_reuses_model(
    monkeypatch,
) -> None:
    constructed = 0
    inference_batches: list[list[str]] = []

    class FakeRXNMapper:
        def __init__(self) -> None:
            nonlocal constructed
            constructed += 1

        def get_attention_guided_atom_maps(self, reactions: list[str]) -> list[dict]:
            inference_batches.append(list(reactions))
            return [{"mapped_rxn": ""} for _ in reactions]

    fake_module = ModuleType("rxnmapper")
    fake_module.RXNMapper = FakeRXNMapper
    monkeypatch.setitem(sys.modules, "rxnmapper", fake_module)
    monkeypatch.setattr(edge_verification, "_RXNMAPPER_RESULT_CACHE_MAXSIZE", 2)
    edge_verification._cached_mapper_instance.cache_clear()
    with edge_verification._RXNMAPPER_INFERENCE_LOCK:
        edge_verification._RXNMAPPER_RESULT_CACHE.clear()

    cases = [
        ("CC", ["C"]),
        ("CCC", ["CC"]),
        ("CC", ["C"]),  # cache hit also makes this entry most-recently-used
        ("CCCC", ["CCC"]),
    ]
    for product, precursors in cases:
        verify_codex_consensus_graph(
            _graph(product=product, precursors=precursors),
            enable_optional_rxnmapper=True,
        )

    assert constructed == 1
    assert inference_batches == [["C>>CC"], ["CC>>CCC"], ["CCC>>CCCC"]]
    assert len(edge_verification._RXNMAPPER_RESULT_CACHE) == 2
    assert list(edge_verification._RXNMAPPER_RESULT_CACHE) == [
        "C>>CC",
        "CCC>>CCCC",
    ]

    with edge_verification._RXNMAPPER_INFERENCE_LOCK:
        edge_verification._RXNMAPPER_RESULT_CACHE.clear()
    edge_verification._cached_mapper_instance.cache_clear()


def test_durable_edge_cache_second_call_does_not_invoke_default_mapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mapper_calls: list[list[str]] = []
    mapper_initializations = 0

    def mapper(reactions: list[str]) -> list[str]:
        mapper_calls.append(list(reactions))
        return ["[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"]

    def optional_mapper():
        nonlocal mapper_initializations
        mapper_initializations += 1
        return mapper, {
            "attempted": False,
            "backend": "rxnmapper",
            "request_count": 0,
            "mapped_count": 0,
            "reasons": [],
        }

    monkeypatch.setattr(edge_verification, "_optional_rxnmapper", optional_mapper)
    first = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        work_dir=tmp_path,
    )
    second = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        work_dir=tmp_path,
    )

    assert mapper_calls == [["CCO>>CC=O"]]
    assert mapper_initializations == 1
    assert first["work_cache"]["miss_count"] == 1
    assert first["work_cache"]["hit_count"] == 0
    assert second["work_cache"]["hit_count"] == 1
    assert second["work_cache"]["miss_count"] == 0
    assert second["atom_mapper"]["request_count"] == 0
    cache_ref = Path(second["work_cache"]["result_refs"][0]["ref"])
    assert cache_ref.is_file()


def test_durable_edge_cache_recomputes_only_changed_edge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mapper_calls: list[list[str]] = []

    def mapper(reactions: list[str]) -> list[None]:
        mapper_calls.append(list(reactions))
        return [None for _ in reactions]

    monkeypatch.setattr(
        edge_verification,
        "_optional_rxnmapper",
        lambda: (
            mapper,
            {
                "attempted": False,
                "backend": "rxnmapper",
                "request_count": 0,
                "mapped_count": 0,
                "reasons": [],
            },
        ),
    )
    exact_rows = [
        {
            "row_id": "row:1",
            "product_smiles": "CC=O",
            "reactant_smiles": ["CCO"],
            "conditions": ["condition-a"],
        },
        {
            "row_id": "row:2",
            "product_smiles": "CCC",
            "reactant_smiles": ["CC"],
            "conditions": ["condition-b"],
        },
    ]
    first = verify_codex_consensus_graph(
        _two_edge_graph(),
        exact_rows=exact_rows,
        work_dir=tmp_path,
    )
    changed_rows = json.loads(json.dumps(exact_rows))
    changed_rows[1]["conditions"] = ["condition-b-revised"]
    second = verify_codex_consensus_graph(
        _two_edge_graph(),
        exact_rows=changed_rows,
        work_dir=tmp_path,
    )

    assert first["work_cache"]["miss_count"] == 2
    assert second["work_cache"]["hit_count"] == 1
    assert second["work_cache"]["miss_count"] == 1
    assert mapper_calls == [
        ["CCO>>CC=O", "CC>>CCC"],
        ["CC>>CCC"],
    ]
    statuses = {
        row["step_id"]: row["status"]
        for row in second["work_cache"]["result_refs"]
    }
    assert statuses == {"step:1": "hit", "step:2": "miss"}


def test_durable_edge_cache_tamper_forces_host_recompute(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mapper_call_count = 0

    def mapper(reactions: list[str]) -> list[str]:
        nonlocal mapper_call_count
        mapper_call_count += 1
        return ["[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"]

    monkeypatch.setattr(
        edge_verification,
        "_optional_rxnmapper",
        lambda: (
            mapper,
            {
                "attempted": False,
                "backend": "rxnmapper",
                "request_count": 0,
                "mapped_count": 0,
                "reasons": [],
            },
        ),
    )
    first = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        work_dir=tmp_path,
    )
    cache_path = Path(first["work_cache"]["result_refs"][0]["ref"])
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    entry["result"]["edge_verification"]["proof_level"] = "forged"
    entry["result_sha256"] = edge_verification._digest(entry["result"])
    entry.pop("content_sha256", None)
    entry["content_sha256"] = edge_verification._digest(entry)
    cache_path.write_text(json.dumps(entry), encoding="utf-8")

    second = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        work_dir=tmp_path,
    )
    assert mapper_call_count == 2
    assert second["work_cache"]["hit_count"] == 0
    assert second["work_cache"]["miss_count"] == 1
    assert second["work_cache"]["invalid_entry_count"] == 1
    assert "cache_entry_not_equal_to_current_host_replay" in second[
        "work_cache"
    ]["result_refs"][0]["reasons"]


def test_durable_edge_cache_is_fenced_by_verifier_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mapper_call_count = 0

    def mapper(reactions: list[str]) -> list[None]:
        nonlocal mapper_call_count
        mapper_call_count += 1
        return [None for _ in reactions]

    monkeypatch.setattr(
        edge_verification,
        "_optional_rxnmapper",
        lambda: (
            mapper,
            {
                "attempted": False,
                "backend": "rxnmapper",
                "request_count": 0,
                "mapped_count": 0,
                "reasons": [],
            },
        ),
    )
    first = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        work_dir=tmp_path,
    )
    monkeypatch.setattr(
        edge_verification,
        "REACTION_STEP_VERIFIER_VERSION",
        "reaction_step_verifier.future-test",
    )
    second = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        work_dir=tmp_path,
    )

    assert mapper_call_count == 2
    assert first["work_cache"]["result_refs"][0]["input_sha256"] != second[
        "work_cache"
    ]["result_refs"][0]["input_sha256"]
    assert second["work_cache"]["miss_count"] == 1


def test_injected_mapper_bypasses_durable_work_cache(tmp_path: Path) -> None:
    mapper_call_count = 0

    def mapper(reactions: list[str]) -> list[None]:
        nonlocal mapper_call_count
        mapper_call_count += 1
        return [None for _ in reactions]

    first = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        atom_mapper=mapper,
        enable_optional_rxnmapper=False,
        work_dir=tmp_path,
    )
    second = verify_codex_consensus_graph(
        _graph(product="CC=O", precursors=["CCO"]),
        atom_mapper=mapper,
        enable_optional_rxnmapper=False,
        work_dir=tmp_path,
    )

    assert mapper_call_count == 2
    assert first["work_cache"]["eligible"] is False
    assert second["work_cache"]["bypass_count"] == 1
    assert list((tmp_path / "codex_edge_work_cache").rglob("*.json")) == []
