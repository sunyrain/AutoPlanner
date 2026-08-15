from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cascade_planner.application.blind_benchmark_contract import (
    BLIND_CASE_SCHEMA,
    BLIND_MANIFEST_SCHEMA,
    BlindBenchmarkError,
    BlindCase,
    audit_blind_preflight,
    load_blind_manifest,
)

TARGET = "CCOC(N)=O"


def _case(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": BLIND_CASE_SCHEMA,
        "case_id": "blind-01",
        "target_name": "opaque benchmark molecule",
        "target_smiles": TARGET,
        "acceptance": {
            "minimum_complete_routes": 2,
            "minimum_edge_proof_level": 2,
            "stock_boundary": "benchmark_search",
        },
        "budget": {"max_model_invocations": 2},
    }
    row.update(updates)
    return row


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("routes", []),
        ("precursor_smiles", ["CCO"]),
        ("sources", ["doi:example"]),
        ("inventory", {"available": True}),
        ("replay", "answer.json"),
        ("fixture", "target-specific"),
    ],
)
def test_blind_case_rejects_route_source_stock_and_replay_material(
    field: str,
    value: object,
) -> None:
    with pytest.raises(BlindBenchmarkError, match="fields_forbidden"):
        BlindCase.from_dict(_case(**{field: value}))


def test_blind_case_requires_canonical_smiles_and_generic_options_only() -> None:
    with pytest.raises(BlindBenchmarkError, match="not_canonical"):
        BlindCase.from_dict(_case(target_smiles="CCOC(=O)N"))
    row = _case()
    row["budget"] = {"source_refs": ["forbidden"]}
    with pytest.raises(BlindBenchmarkError, match="budget_fields_forbidden"):
        BlindCase.from_dict(row)


@pytest.mark.parametrize("value", [-1, 25, True, 20.0, "20"])
def test_blind_case_bounds_generic_minimum_planning_depth(value: object) -> None:
    row = _case()
    row["acceptance"] = {
        **dict(row["acceptance"]),
        "minimum_planning_route_steps": value,
    }

    with pytest.raises(BlindBenchmarkError, match="planning_route_steps_invalid"):
        BlindCase.from_dict(row)

    accepted = _case()
    accepted["acceptance"] = {
        **dict(accepted["acceptance"]),
        "minimum_planning_route_steps": 20,
    }
    assert (
        BlindCase.from_dict(accepted).acceptance["minimum_planning_route_steps"] == 20
    )


def test_manifest_loads_target_only_cases_and_rejects_duplicate_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": BLIND_MANIFEST_SCHEMA, "cases": [_case()]}),
        encoding="utf-8",
    )
    assert load_blind_manifest(path)[0].target_smiles == TARGET

    duplicate = _case(case_id="blind-02")
    path.write_text(
        json.dumps(
            {"schema_version": BLIND_MANIFEST_SCHEMA, "cases": [_case(), duplicate]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(BlindBenchmarkError, match="target_duplicate"):
        load_blind_manifest(path)


def test_preflight_requires_fresh_run_and_target_absence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "unrelated.txt").write_text("nothing target-specific", "utf-8")
    case = BlindCase.from_dict(_case())
    accepted = audit_blind_preflight(
        case,
        repository_root=repository,
        run_dir=tmp_path / "new-run",
    )
    assert accepted["accepted"] is True
    assert accepted["repository_absence_attested"] is True

    (repository / "hidden-dossier.json").write_text(
        json.dumps({"target_smiles": TARGET, "routes": []}),
        "utf-8",
    )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.json").write_text("{}", "utf-8")
    rejected = audit_blind_preflight(
        case,
        repository_root=repository,
        run_dir=occupied,
    )
    assert rejected["accepted"] is False
    assert rejected["reasons"] == [
        "blind_run_directory_not_fresh",
        "target_material_already_present_in_repository",
    ]


def test_preflight_does_not_treat_generic_blind_label_as_leaked_identity(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "readme.md").write_text(
        "Every blind target starts from only a SMILES.", encoding="utf-8"
    )
    case = BlindCase.from_dict(_case(target_name="blind target"))

    report = audit_blind_preflight(
        case,
        repository_root=repository,
        run_dir=tmp_path / "fresh-run",
    )

    assert report["accepted"] is True
    assert report["repository_matches"] == []


def test_preflight_treats_numbered_opaque_labels_as_generic(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "other-manifest.json").write_text(
        '"opaque benchmark target 001"', encoding="utf-8"
    )
    case = BlindCase.from_dict(_case(target_name="opaque benchmark target 001"))

    report = audit_blind_preflight(
        case,
        repository_root=repository,
        run_dir=tmp_path / "fresh-run",
    )

    assert report["accepted"] is True
    assert report["repository_matches"] == []


def test_manifest_is_the_only_allowed_target_occurrence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = repository / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": BLIND_MANIFEST_SCHEMA, "cases": [_case()]}),
        "utf-8",
    )
    case = load_blind_manifest(manifest)[0]
    report = audit_blind_preflight(
        case,
        repository_root=repository,
        run_dir=tmp_path / "run",
        manifest_path=manifest,
    )
    assert report["accepted"] is True
    assert report["repository_matches"] == []


def test_preflight_scans_evaluator_only_synonyms_and_intermediates_without_emitting_values(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "unrelated.txt").write_text("unrelated chemistry", encoding="utf-8")
    case = BlindCase.from_dict(_case())
    report = audit_blind_preflight(
        case,
        repository_root=repository,
        run_dir=tmp_path / "run",
        additional_leakage_needles={
            "target_synonym": ["private target alias"],
            "key_intermediate_smiles": ["c1ccccc1"],
            "key_intermediate_inchikey": ["UHOVQNZJYSORNB-UHFFFAOYSA-N"],
        },
    )

    assert report["accepted"] is True
    assert report["additional_leakage_needle_counts"] == {
        "key_intermediate_inchikey": 1,
        "key_intermediate_smiles": 1,
        "target_synonym": 1,
    }
    assert report["semantics"]["target_name_smiles_and_inchikey_checked"] is True
    assert report["semantics"]["target_synonym_needles_checked"] is True
    assert report["semantics"]["key_intermediate_needles_checked"] is True
    rendered = json.dumps(report)
    assert "private target alias" not in rendered
    assert "c1ccccc1" not in rendered
    assert "UHOVQNZJYSORNB-UHFFFAOYSA-N" not in rendered


def test_preflight_rejects_evaluator_only_intermediate_leakage(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "hidden-answer.txt").write_text("c1ccccc1", encoding="utf-8")
    report = audit_blind_preflight(
        BlindCase.from_dict(_case()),
        repository_root=repository,
        run_dir=tmp_path / "run",
        additional_leakage_needles={"key_intermediate_smiles": ["c1ccccc1"]},
    )

    assert report["accepted"] is False
    assert report["reasons"] == [
        "evaluator_answer_material_already_present_in_repository"
    ]
    assert (
        report["repository_matches"][0]["needle_sha256"]
        == hashlib.sha256(b"c1ccccc1").hexdigest()
    )


def test_stock_membership_is_not_treated_as_route_answer_knowledge(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    stock = repository / "data" / "stock"
    stock.mkdir(parents=True)
    (stock / "catalog.csv").write_text("c1ccccc1\n", encoding="utf-8")
    (repository / "route-notes.txt").write_text("unrelated", encoding="utf-8")

    report = audit_blind_preflight(
        BlindCase.from_dict(_case()),
        repository_root=repository,
        run_dir=tmp_path / "run",
        additional_leakage_needles={"key_intermediate_smiles": ["c1ccccc1"]},
    )

    assert report["accepted"] is True
    assert report["repository_matches"] == []
    assert (
        report["semantics"]["inventory_membership_is_not_route_answer_knowledge"]
        is True
    )


def test_preflight_does_not_allow_synonym_na_for_a_named_target(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    raw = _case()
    raw["target_name"] = "Example Drug"

    report = audit_blind_preflight(
        BlindCase.from_dict(raw),
        repository_root=repository,
        run_dir=tmp_path / "run",
        target_synonym_not_applicable_reason="this target has no aliases",
    )

    assert report["accepted"] is False
    assert "target_synonym_audit_not_applicable_for_named_target" in report["reasons"]


def test_checked_in_benchmark_summary_is_compact_bound_and_keeps_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "benchmarks" / "blind_targets.v1.json"
    summary_path = root / "benchmarks" / "results" / "blind_benchmark_summary.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "blind_retrosynthesis_benchmark_summary.v1"
    assert (
        summary["manifest_file_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert {row["case_id"] for row in summary["results"]} == {
        row["case_id"] for row in manifest["cases"]
    }
    assert summary["aggregate"]["case_count"] == len(manifest["cases"])
    assert any(
        row["qualified_policy_acceptance"] is False for row in summary["results"]
    )
    forbidden = {
        "target_smiles",
        "routes",
        "precursors",
        "precursor_smiles",
        "reaction_smiles",
        "edge_ids",
        "source_refs",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for child in value.values() for key in keys(child)}
        if isinstance(value, list):
            return {key for child in value for key in keys(child)}
        return set()

    assert not (keys(summary) & forbidden)
