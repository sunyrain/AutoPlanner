from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.providers.stock import stock_snapshot_sha256
from cascade_planner.routes.domain import MoleculeIdentity
from cascade_planner.runtime.run_metrics import validate_run_metrics
from scripts import run_nirmatrelvir_v3_golden as golden_runner


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "config/examples/nirmatrelvir_v3_golden_acceptance.json"


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nirmatrelvir_golden_contract_is_internally_consistent() -> None:
    golden = _read(GOLDEN)
    manifest = _read(ROOT / golden["candidate_manifest"])
    snapshots = _read(ROOT / golden["stock_snapshots"])["snapshots"]

    assert golden["schema_version"] == "retrosynthesis_golden_acceptance.v1"
    assert manifest["schema_version"] == "source_route_candidate_manifest.v1"
    assert len(manifest["routes"]) == golden["expected"]["complete_route_count"]
    assert sum(len(route["step_ids"]) for route in manifest["routes"]) == (
        golden["expected"]["approved_source_step_count"]
    )
    assert {row["source_ref"].casefold() for row in manifest["sources"]} == set(
        golden["expected"]["independent_support_groups"]
    )
    assert len(snapshots) == golden["expected"]["stock_terminal_count"]
    assert all(row["available"] is True for row in snapshots)
    assert all(row["source_url"].startswith("https://") for row in snapshots)
    assert all(
        row["snapshot_sha256"] == stock_snapshot_sha256(row)
        for row in snapshots
    )

    patent_steps = [
        row
        for row in manifest["steps"]
        if row["source_ref"].casefold() == "patent:wo2021250648a1"
    ]
    assert len(patent_steps) == 7
    target = MoleculeIdentity(manifest["target_smiles"])
    patent_target = MoleculeIdentity(
        next(row["product_smiles"] for row in patent_steps if row["step_id"] == "patent_13")
    )
    assert target.molecule_id == patent_target.molecule_id


def test_local_golden_source_digests_when_artifacts_are_present() -> None:
    golden = _read(GOLDEN)
    observed = 0
    for source in golden["source_documents"]:
        for path_key, digest_key in (
            ("artifact_path", "artifact_sha256"),
            ("text_companion_path", "text_companion_sha256"),
        ):
            relative = source.get(path_key)
            if not relative:
                continue
            path = ROOT / relative
            if not path.is_file():
                continue
            observed += 1
            assert _sha256(path) == source[digest_key]
    # The repository can be cloned without ignored source PDFs; when present,
    # every one is still required to match the committed golden contract.
    if observed == 0:
        return


def test_golden_runner_emits_model_free_stage_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = _read(GOLDEN)["expected"]
    monkeypatch.setattr(golden_runner, "_verify_source_artifacts", lambda _: None)
    monkeypatch.setattr(
        golden_runner,
        "candidate_steps_from_blackboard",
        lambda *args, **kwargs: [{"step_id": "mock"}],
    )
    monkeypatch.setattr(
        golden_runner,
        "candidate_steps_from_manifest",
        lambda *args, **kwargs: [{"step_id": "mock"}],
    )
    monkeypatch.setattr(
        golden_runner,
        "compile_deterministic_literature_step_registry",
        lambda *args, **kwargs: {
            "approved_binding_count": 1,
            "rejected_step_count": 0,
        },
    )
    monkeypatch.setattr(
        golden_runner,
        "compile_source_route_portfolio",
        lambda **kwargs: {
            "accepted": True,
            "reasons": [],
            "approved_source_step_count": expected[
                "approved_source_step_count"
            ],
            "hyperedge_count": expected["unique_reaction_hyperedge_count"],
            "complete_route_count": expected["complete_route_count"],
            "selected_route_count": expected["selected_route_count"],
            "stock_terminal_count": expected["stock_terminal_count"],
            "model_invocations": 0,
            "independent_support_groups": expected[
                "independent_support_groups"
            ],
        },
    )

    result = golden_runner.run_golden_case(
        golden_path=GOLDEN,
        output_dir=tmp_path,
    )

    metrics = result["run_metrics"]
    assert validate_run_metrics(metrics) == []
    assert metrics["case_id"] == "nirmatrelvir-v3-real-source-dual-route"
    assert metrics["gauges"]["model_invocations"] == 0
    assert metrics["gauges"]["complete_route_count"] == 2
    assert {
        "golden.verify_sources",
        "golden.compile_source_registry",
        "golden.compile_portfolio",
        "golden.enforce_acceptance",
    }.issubset({row["name"] for row in metrics["stages"]})
    assert Path(result["artifacts"]["run_metrics"]).is_file()
