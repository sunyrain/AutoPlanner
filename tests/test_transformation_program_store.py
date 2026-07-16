from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from cascade_planner.application.transformation_program_store import (
    TransformationProgramAdmissionDisabled,
    TransformationProgramStore,
    TransformationProgramStoreCorruption,
)
from cascade_planner.application.transformation_program_validation import (
    validate_program_projection,
)
from cascade_planner.application.transformation_programs import (
    program_id,
    project_canonical_graph_to_programs,
)
from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def _graph() -> dict:
    return {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "program-store",
        "revision": 4,
        "scientific_sha256": "source-scientific-digest",
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:a": {
                "molecule_id": "molecule:a",
                "canonical_smiles": "CCO",
                "stock_observation_ids": ["stock:a"],
            },
            "molecule:b": {
                "molecule_id": "molecule:b",
                "canonical_smiles": "CC(=O)Cl",
            },
            "molecule:target": {
                "molecule_id": "molecule:target",
                "canonical_smiles": "CCOC(C)=O",
            },
        },
        "edges": {
            "edge:ester": {
                "edge_id": "edge:ester",
                "precursor_molecule_ids": [
                    "molecule:a",
                    "molecule:a",
                    "molecule:b",
                ],
                "product_molecule_id": "molecule:target",
                "source_binding_ids": ["source:paper"],
                "exact_record_ids": ["exact:row"],
                "procedure_record_ids": ["procedure:one"],
                "reaction_proofs": [{"proof_digest": "proof:host"}],
            }
        },
        "route_families": {
            "route:ester": {
                "route_family_id": "route:ester",
                "edge_ids": ["edge:ester"],
                "closed": True,
            }
        },
    }


def _store(tmp_path) -> TransformationProgramStore:
    return TransformationProgramStore(
        run_id="program-store",
        run_dir=tmp_path / "run",
        artifacts=ArtifactStore(tmp_path / "cas"),
    )


def _redigest(value: dict) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = strict_canonical_json_sha256(value)


def test_projection_contract_rejects_forged_authority_even_with_fresh_digests() -> None:
    projection = project_canonical_graph_to_programs(_graph())
    accepted = validate_program_projection(projection, expected_run_id="program-store")
    forged = deepcopy(projection)
    program = forged["programs"][program_id("edge:ester")]
    program["validation_vector"]["authoritative"] = True
    _redigest(program)
    _redigest(forged)
    rejected = validate_program_projection(forged, expected_run_id="program-store")

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert "program_contract_invalid:program:edge:ester" in rejected["reasons"]


def test_store_is_read_only_until_explicitly_enabled_and_then_idempotent(
    tmp_path,
) -> None:
    graph = _graph()
    store = _store(tmp_path)

    status = store.status(graph)
    assert status["initialized"] is False
    assert status["event_count"] == 0
    assert status["oracle"]["accepted"] is False
    assert not store.root.exists()
    with pytest.raises(TransformationProgramAdmissionDisabled, match="explicit_enable_required"):
        store.admit(graph)
    assert not store.root.exists()

    first = store.admit(graph, enable_program_admission=True)
    second = store.admit(graph, enable_program_admission=True)
    replay = store.replay()

    assert first["admitted"] is True
    assert first["created"] is True
    assert second["created"] is False
    assert second["store"]["current_projection_admitted"] is True
    assert second["store"]["oracle"]["accepted"] is True
    assert replay["event_count"] == 1
    assert replay["semantics"]["source_graph_and_projection_objects_verified"] is True


def test_store_reports_a_new_graph_revision_as_unadmitted_without_mutation(
    tmp_path,
) -> None:
    graph = _graph()
    store = _store(tmp_path)
    store.admit(graph, enable_program_admission=True)
    event_paths = list(store.event_root.glob("*/*.json"))
    changed = deepcopy(graph)
    changed["revision"] = 5
    changed["scientific_sha256"] = "new-scientific-digest"

    status = store.status(changed)

    assert status["event_count"] == 1
    assert status["current_projection_admitted"] is False
    assert status["oracle"]["accepted"] is False
    assert list(store.event_root.glob("*/*.json")) == event_paths


def test_store_replay_fails_closed_on_event_tampering(tmp_path) -> None:
    store = _store(tmp_path)
    store.admit(_graph(), enable_program_admission=True)
    event_path = next(store.event_root.glob("*/*.json"))
    event_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(TransformationProgramStoreCorruption):
        store.replay()


def test_store_replay_fails_closed_on_projection_artifact_tampering(tmp_path) -> None:
    store = _store(tmp_path)
    admitted = store.admit(_graph(), enable_program_admission=True)
    digest = admitted["event"]["projection_ref"]["sha256"]
    store.artifacts.object_path(digest).write_bytes(b"{}")

    with pytest.raises(TransformationProgramStoreCorruption, match="artifact_replay_failed"):
        store.replay()


def test_concurrent_identical_admission_publishes_one_event(tmp_path) -> None:
    store = _store(tmp_path)
    graph = _graph()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: store.admit(graph, enable_program_admission=True),
                range(16),
            )
        )

    assert sum(result["created"] is True for result in results) == 1
    assert store.replay()["event_count"] == 1
