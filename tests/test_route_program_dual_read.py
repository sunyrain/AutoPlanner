from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cascade_planner.application.route_program_dual_read import (
    RouteProgramDualReadError,
    project_workbench_routes_to_programs,
    route_program_dual_read_oracle,
)
from cascade_planner.application.transformation_program_store import (
    TransformationProgramStore,
)
from cascade_planner.application.transformation_programs import (
    project_canonical_graph_to_programs,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.replay_pack import run_replay_pack
from cascade_planner.runtime.artifact_store import ArtifactStore
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.runtime.immutable_json_events import publish_immutable_json_event
from cascade_planner.runtime.paths import RuntimePaths


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "benchmarks" / "fluvastatin_current_canonical_graph.v1.json"
WORKBENCH_PATH = ROOT / "benchmarks" / "fluvastatin_current_route_workbench.v1.json"
EVENT_PATH = ROOT / "benchmarks" / "fluvastatin_current_program_admission_event.v1.json"
REPLAY_PACKS = (
    ROOT / "config" / "examples" / "nirmatrelvir_v4_replay_pack.json",
    ROOT / "config" / "examples" / "artemisinin_v4_replay_pack.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _redigest(value: dict) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = strict_canonical_json_sha256(value)


def _isolated_paths(root: Path) -> RuntimePaths:
    repository = root / "repository"
    repository.mkdir(parents=True)
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(root / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(root / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(root / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(root / "index" / "runs.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(root / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(root / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(root / "vendor"),
        },
    )


def test_real_fluvastatin_workbench_has_equivalent_program_dual_read() -> None:
    graph = _load(GRAPH_PATH)
    workbench = _load(WORKBENCH_PATH)
    program_projection = project_canonical_graph_to_programs(graph)

    overlay = project_workbench_routes_to_programs(workbench, program_projection)
    oracle = route_program_dual_read_oracle(workbench, program_projection, overlay)

    assert overlay["content_sha256"] == (
        "8c04f2472a42e0f1668b2ac0dd946e458d686eb488335e99639f6a9761adf675"
    )
    assert overlay["counts"] == {
        "displayed_routes": 5,
        "replacement_routes": 2,
        "edge_references": 17,
        "distinct_programs": 11,
        "physical_step_count_mismatches": 0,
    }
    assert overlay["equivalence"]["accepted"] is True
    assert oracle["accepted"] is True
    assert all(oracle["checks"].values())


@pytest.mark.parametrize("pack_path", REPLAY_PACKS, ids=("nirmatrelvir", "artemisinin"))
def test_model_free_current_replays_pass_gateway_program_dual_read(
    tmp_path: Path,
    pack_path: Path,
) -> None:
    paths = _isolated_paths(tmp_path / pack_path.stem)
    run_id = f"dual-read-{pack_path.stem}"
    replay = run_replay_pack(pack_path, paths=paths, run_id=run_id)

    observed = CampaignGateway(paths).route_program_dual_read(
        run_id,
        run_dir=replay["run_dir"],
    )

    assert replay["accepted"] is True
    assert observed["operation"] == "route-program-dual-read"
    assert observed["oracle"]["accepted"] is True
    assert observed["overlay"]["equivalence"]["accepted"] is True
    assert observed["overlay"]["counts"]["displayed_routes"] == 2
    assert observed["overlay"]["counts"]["physical_step_count_mismatches"] == 0
    assert all(observed["oracle"]["checks"].values())


def test_target_name_is_not_a_dual_read_branch_rule() -> None:
    graph = _load(GRAPH_PATH)
    workbench = _load(WORKBENCH_PATH)
    projection = project_canonical_graph_to_programs(graph)
    baseline = project_workbench_routes_to_programs(workbench, projection)
    renamed = deepcopy(workbench)
    renamed["target"]["name"] = "opaque-current-target"
    _redigest(renamed)

    observed = project_workbench_routes_to_programs(renamed, projection)

    assert observed["counts"] == baseline["counts"]
    assert observed["equivalence"] == baseline["equivalence"]
    assert route_program_dual_read_oracle(renamed, projection, observed)["accepted"] is True


def test_dual_read_oracle_rejects_authority_snapshot_tampering() -> None:
    graph = _load(GRAPH_PATH)
    workbench = _load(WORKBENCH_PATH)
    projection = project_canonical_graph_to_programs(graph)
    observed = project_workbench_routes_to_programs(workbench, projection)
    forged = deepcopy(observed)
    route = next(iter(forged["collections"]["routes"].values()))
    route["authority_snapshot"]["proof_level"] = 99
    route["authority_snapshot_sha256"] = strict_canonical_json_sha256(
        route["authority_snapshot"]
    )
    _redigest(route)
    _redigest(forged)

    oracle = route_program_dual_read_oracle(workbench, projection, forged)

    assert oracle["accepted"] is False
    assert oracle["checks"]["observed_content_digest_valid"] is True
    assert oracle["checks"]["projection_equal"] is False


def test_physical_superstep_mismatch_fails_equivalence_closed() -> None:
    graph = _load(GRAPH_PATH)
    workbench = _load(WORKBENCH_PATH)
    projection = project_canonical_graph_to_programs(graph)
    changed = deepcopy(workbench)
    route = next(iter(changed["routes"].values()))
    route["physical_step_count"] = int(route["physical_step_count"]) + 1
    _redigest(changed)

    overlay = project_workbench_routes_to_programs(changed, projection)
    oracle = route_program_dual_read_oracle(changed, projection, overlay)

    assert overlay["counts"]["physical_step_count_mismatches"] == 1
    assert overlay["equivalence"]["accepted"] is False
    assert oracle["accepted"] is False


def test_graph_revision_or_program_contract_mismatch_fails_before_overlay() -> None:
    graph = _load(GRAPH_PATH)
    workbench = _load(WORKBENCH_PATH)
    projection = project_canonical_graph_to_programs(graph)
    changed = deepcopy(workbench)
    changed["revision"]["graph"] += 1
    _redigest(changed)

    with pytest.raises(RouteProgramDualReadError, match="source_graph_revision_mismatch"):
        project_workbench_routes_to_programs(changed, projection)


def test_real_fluvastatin_program_admission_event_replays_from_frozen_assets(
    tmp_path: Path,
) -> None:
    graph = _load(GRAPH_PATH)
    event = _load(EVENT_PATH)
    projection = project_canonical_graph_to_programs(graph)
    artifacts = ArtifactStore(tmp_path / "cas")
    graph_ref = artifacts.put_json(
        graph,
        logical_name="canonical_hypergraph.program_admission.json",
        producer="autoplanner.transformation_program_store",
    )
    projection_ref = artifacts.put_json(
        projection,
        logical_name="transformation_program_projection.json",
        producer="autoplanner.transformation_program_store",
    )
    store = TransformationProgramStore(
        run_id="statin-fluvastatin-v4-current-20260716",
        run_dir=tmp_path / "run",
        artifacts=artifacts,
    )

    assert graph_ref.to_dict() == event["source_graph_ref"]
    assert projection_ref.to_dict() == event["projection_ref"]
    publish_immutable_json_event(
        store.event_root,
        event,
        content_sha256=event["content_sha256"],
    )

    replay = store.replay()
    status = store.status(graph)
    repeated = store.admit(graph, enable_program_admission=True)

    assert replay["event_count"] == 1
    assert replay["semantics"]["source_graph_and_projection_objects_verified"] is True
    assert status["oracle"]["accepted"] is True
    assert status["current_projection_sha256"] == (
        "24fdb289ca3c333b0e482dbd348a842acfcb2300794e7b3f7a61ea4adbc6f1f0"
    )
    assert repeated["created"] is False
    assert repeated["event"]["counts"] == {
        "chemical_states": 44,
        "operation_nodes": 30,
        "programs": 30,
        "routes": 5,
    }
