from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_recent_total_synthesis_benchmark.py")
packets = load_script("build_recent_total_synthesis_review_packets.py")
validator = load_script("validate_recent_total_synthesis_review_submission.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def review_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo_root = tmp_path
    dataset_dir = repo_root / "benchmarks" / "recent_total_synthesis"
    source = repo_root / "tmp" / "source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("source evidence", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    paper = {
        "paper_id": "paper-example",
        "doi": "10.1000/example",
        "title": "Total Synthesis of Example",
    }
    target = {
        "target_slot_id": "target-example",
        "paper_id": "paper-example",
        "doi": "10.1000/example",
        "slot_class": "primary",
        "target_name": "Example",
    }
    write_jsonl(dataset_dir / "papers.jsonl", [paper])
    write_jsonl(dataset_dir / "target_slots.jsonl", [target])
    ledger = {
        "schema_version": "recent_total_synthesis_review_decisions.v1",
        "paper_reviews": [],
        "structure_reviews": [],
        "route_reviews": [],
        "adjudications": [],
    }
    ledger_path = dataset_dir / "curation_inputs" / "review_decisions.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(validator, "load_builder", lambda unused: builder)
    binding = {
        "source_artifact_path": "tmp/source.txt",
        "source_artifact_sha256": digest,
        "source_locator": "Scheme 1",
    }
    submission = {
        "schema_version": "recent_total_synthesis_review_submission.v1",
        "packet_type": "target_truth",
        "packet_id": "target-truth--target-example--v1",
        "target_slot_id": "target-example",
        "paper_id": "paper-example",
        "doi": "10.1000/example",
        "reviewer": {
            "reviewer_id": "chemist-01",
            "reviewed_at": "2026-09-02T10:00:00+08:00",
            "attestation": True,
        },
        "structure_review": {
            "decision": "accept",
            "record": {
                "isomeric_smiles": "CCO",
                "source_doi": "10.1000/example",
                **binding,
                "identity_confirmed": True,
                "relative_stereochemistry_confirmed": True,
                "absolute_stereochemistry_status": "not_applicable",
            },
            "reviewer_notes": "",
        },
        "route_review": {
            "decision": "accept",
            "record": {
                "source_doi": "10.1000/example",
                "reference_scope": "strategic_key_step",
                "source_artifacts": [binding],
                "steps": [],
                "strategic_events": [
                    {
                        "event_id": "event-1",
                        "description": "Fragment union establishes the target skeleton.",
                        "transformation_class": "fragment coupling",
                        "source_locator": "Scheme 1",
                    }
                ],
            },
            "reviewer_notes": "",
        },
    }
    return {
        "repo_root": repo_root,
        "dataset_dir": dataset_dir,
        "ledger": ledger,
        "ledger_path": ledger_path,
        "submission": submission,
        "binding": binding,
        "paper": paper,
        "target": target,
        "digest": digest,
    }


@pytest.fixture
def structured_route_fixture(review_fixture: dict) -> dict:
    repo_root = review_fixture["repo_root"]
    dataset_dir = review_fixture["dataset_dir"]
    candidate_path = (
        dataset_dir
        / "curation_candidates"
        / "structured_routes"
        / "target-example.json"
    )
    candidate = {
        "schema_version": packets.STRUCTURED_ROUTE_SCHEMA,
        "candidate_id": "structured-route--target-example--v1",
        "admission_authority": False,
        "extraction_status": "complete_ordered_route_candidate",
        "target_slot_id": "target-example",
        "target_compound_id": "compound-target",
        "paper_id": "paper-example",
        "source_doi": "10.1000/example",
        "reference_scope": "ordered_route",
        "source_artifacts": [review_fixture["binding"]],
        "compounds": [
            {
                "compound_id": "compound-start",
                "label": "1",
                "role": "starting_material",
                "smiles": "C",
                "molecular_formula": "CH4",
            },
            {
                "compound_id": "compound-middle",
                "label": "2",
                "role": "intermediate",
                "smiles": "CO",
                "molecular_formula": "CH4O",
            },
            {
                "compound_id": "compound-target",
                "label": "3",
                "role": "target",
                "smiles": "C=O",
                "molecular_formula": "CH2O",
            },
        ],
        "steps": [
            {
                "order": 1,
                "step_id": "step-1",
                "precursor_compound_ids": ["compound-start"],
                "precursor_labels": ["1"],
                "product_compound_id": "compound-middle",
                "product_label": "2",
                "transformation_class": "oxidation",
                "strategic_role": "install alcohol",
                "source_locator": "Scheme 1, step a",
            },
            {
                "order": 2,
                "step_id": "step-2",
                "precursor_compound_ids": ["compound-middle"],
                "precursor_labels": ["2"],
                "product_compound_id": "compound-target",
                "product_label": "3",
                "transformation_class": "oxidation",
                "strategic_role": "reach target oxidation state",
                "source_locator": "Scheme 1, step b",
            },
        ],
        "strategic_events": [
            {
                "event_id": "event-1",
                "description": "The oxidation sequence reaches the target.",
                "transformation_class": "oxidation sequence",
                "source_locator": "Scheme 1",
            }
        ],
    }
    return {
        "repo_root": repo_root,
        "dataset_dir": dataset_dir,
        "candidate_path": candidate_path,
        "candidate": candidate,
        "targets": {"target-example": review_fixture["target"]},
        "visual_by_target": {
            "target-example": {"visual_canonical_isomeric_smiles": "C=O"}
        },
    }


def write_structured_route_fixture(fixture: dict, candidate: dict) -> None:
    path = fixture["candidate_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate), encoding="utf-8")


def validate_structured_route_fixture(fixture: dict, candidate: dict) -> dict:
    return packets.validate_structured_route_candidate(
        candidate,
        candidate_path=fixture["candidate_path"],
        repo_root=fixture["repo_root"],
        targets=fixture["targets"],
        visual_by_target=fixture["visual_by_target"],
    )


def test_source_audit_checks_every_file_and_reports_missing_packages(tmp_path: Path) -> None:
    artifact = tmp_path / "tmp" / "paper.pdf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"pdf")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipts = {
        "paper-a": {
            "paper_id": "paper-a",
            "source_package_acquired": True,
            "source_package_completeness": "article_only",
            "artifacts": [
                {
                    "artifact_kind": "repository_main_pdf",
                    "cache_path": "tmp/paper.pdf",
                    "sha256": digest,
                }
            ],
        },
        "paper-b": {
            "paper_id": "paper-b",
            "doi": "10.1000/missing",
            "source_package_acquired": False,
            "source_package_completeness": "none",
            "status": "not_in_offline_cache",
            "artifacts": [],
        },
    }
    audit = packets.source_audit(
        repo_root=tmp_path,
        primary_targets=[{"paper_id": "paper-a", "slot_class": "primary"}],
        p1_targets=[{"paper_id": "paper-b", "slot_class": "primary_candidate"}],
        receipts=receipts,
    )
    assert audit["candidate_papers"] == 2
    assert audit["source_packages_acquired"] == 1
    assert audit["verified_artifact_files"] == 1
    assert audit["missing_packages"][0]["doi"] == "10.1000/missing"


def test_structured_route_candidate_is_loaded_as_nonadmitting(
    structured_route_fixture: dict,
) -> None:
    write_structured_route_fixture(
        structured_route_fixture,
        structured_route_fixture["candidate"],
    )
    loaded = packets.load_structured_route_candidates(
        dataset_dir=structured_route_fixture["dataset_dir"],
        repo_root=structured_route_fixture["repo_root"],
        targets=structured_route_fixture["targets"],
        visual_by_target=structured_route_fixture["visual_by_target"],
    )
    candidate = loaded["target-example"]
    assert candidate["admission_authority"] is False
    assert [step["order"] for step in candidate["steps"]] == [1, 2]


def test_structured_route_record_satisfies_route_review_contract(
    structured_route_fixture: dict,
) -> None:
    candidate = validate_structured_route_fixture(
        structured_route_fixture,
        structured_route_fixture["candidate"],
    )
    record = packets.structured_route_record(candidate)
    normalized = builder.normalize_human_route_review_record(
        record,
        target=structured_route_fixture["targets"]["target-example"],
        repo_root=structured_route_fixture["repo_root"],
    )
    assert normalized["reference_scope"] == "ordered_route"
    assert [step["step_id"] for step in normalized["steps"]] == ["step-1", "step-2"]


def test_structured_route_step_order_is_rejected(structured_route_fixture: dict) -> None:
    candidate = json.loads(json.dumps(structured_route_fixture["candidate"]))
    candidate["steps"][1]["order"] = 3
    with pytest.raises(RuntimeError, match="structured_route_step_order_invalid"):
        validate_structured_route_fixture(structured_route_fixture, candidate)


def test_structured_route_unproduced_precursor_is_rejected(
    structured_route_fixture: dict,
) -> None:
    candidate = json.loads(json.dumps(structured_route_fixture["candidate"]))
    candidate["steps"][0]["precursor_compound_ids"] = ["compound-middle"]
    candidate["steps"][0]["precursor_labels"] = ["2"]
    with pytest.raises(RuntimeError, match="structured_route_precursor_not_yet_produced"):
        validate_structured_route_fixture(structured_route_fixture, candidate)


def test_structured_route_target_structure_mismatch_is_rejected(
    structured_route_fixture: dict,
) -> None:
    candidate = json.loads(json.dumps(structured_route_fixture["candidate"]))
    target = next(
        row for row in candidate["compounds"] if row["compound_id"] == "compound-target"
    )
    target["smiles"] = "C#N"
    target["molecular_formula"] = "CHN"
    with pytest.raises(RuntimeError, match="structured_route_target_structure_mismatch"):
        validate_structured_route_fixture(structured_route_fixture, candidate)


def test_structured_route_source_hash_mismatch_is_rejected(
    structured_route_fixture: dict,
) -> None:
    candidate = json.loads(json.dumps(structured_route_fixture["candidate"]))
    candidate["source_artifacts"][0]["source_artifact_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="review_packet_source_hash_mismatch"):
        validate_structured_route_fixture(structured_route_fixture, candidate)


def test_structured_route_admission_authority_is_rejected(
    structured_route_fixture: dict,
) -> None:
    candidate = json.loads(json.dumps(structured_route_fixture["candidate"]))
    candidate["admission_authority"] = True
    with pytest.raises(RuntimeError, match="structured_route_must_be_nonadmitting"):
        validate_structured_route_fixture(structured_route_fixture, candidate)


def test_valid_target_submission_normalizes_both_reviews(review_fixture: dict) -> None:
    rows = validator.validate_submission(
        review_fixture["submission"],
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
        ledger=review_fixture["ledger"],
    )
    assert [field for field, _ in rows] == ["structure_reviews", "route_reviews"]
    assert rows[0][1]["record"]["isomeric_smiles"] == "CCO"


def test_missing_source_hash_is_rejected(review_fixture: dict) -> None:
    submission = json.loads(json.dumps(review_fixture["submission"]))
    submission["structure_review"]["record"]["source_artifact_sha256"] = ""
    with pytest.raises(validator.SubmissionError, match="source_binding_incomplete"):
        validator.validate_submission(
            submission,
            repo_root=review_fixture["repo_root"],
            dataset_dir=review_fixture["dataset_dir"],
        )


def test_wrong_target_binding_is_rejected(review_fixture: dict) -> None:
    submission = json.loads(json.dumps(review_fixture["submission"]))
    submission["paper_id"] = "paper-other"
    with pytest.raises(validator.SubmissionError, match="paper_id"):
        validator.validate_submission(
            submission,
            repo_root=review_fixture["repo_root"],
            dataset_dir=review_fixture["dataset_dir"],
        )


def test_different_duplicate_reviewer_is_rejected(review_fixture: dict) -> None:
    rows = validator.validate_submission(
        review_fixture["submission"],
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
    )
    ledger = json.loads(json.dumps(review_fixture["ledger"]))
    ledger[rows[0][0]].append(rows[0][1])
    changed = json.loads(json.dumps(review_fixture["submission"]))
    changed["structure_review"]["record"]["isomeric_smiles"] = "CCN"
    with pytest.raises(validator.SubmissionError, match="already submitted"):
        validator.validate_submission(
            changed,
            repo_root=review_fixture["repo_root"],
            dataset_dir=review_fixture["dataset_dir"],
            ledger=ledger,
        )


def test_independent_stereo_disagreement_is_retained_for_adjudication(review_fixture: dict) -> None:
    first = validator.validate_submission(
        review_fixture["submission"],
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
    )
    ledger = json.loads(json.dumps(review_fixture["ledger"]))
    ledger[first[0][0]].append(first[0][1])
    second = json.loads(json.dumps(review_fixture["submission"]))
    second["reviewer"]["reviewer_id"] = "chemist-02"
    second["structure_review"]["record"]["isomeric_smiles"] = "C[C@H](O)F"
    rows = validator.validate_submission(
        second,
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
        ledger=ledger,
    )
    assert rows[0][1]["reviewer_id"] == "chemist-02"
    assert rows[0][1]["record"]["isomeric_smiles"] != first[0][1]["record"]["isomeric_smiles"]


def test_ordered_route_step_without_locator_is_rejected(review_fixture: dict) -> None:
    submission = json.loads(json.dumps(review_fixture["submission"]))
    route = submission["route_review"]["record"]
    route["reference_scope"] = "ordered_route"
    route["steps"] = [
        {
            "step_id": "step-1",
            "product_label": "2",
            "precursor_labels": ["1"],
            "transformation_class": "oxidation",
            "strategic_role": "functional-group adjustment",
            "source_locator": "",
        }
    ]
    with pytest.raises(validator.SubmissionError, match="route_review_step_incomplete"):
        validator.validate_submission(
            submission,
            repo_root=review_fixture["repo_root"],
            dataset_dir=review_fixture["dataset_dir"],
        )


def test_merge_is_atomic_and_idempotent(review_fixture: dict) -> None:
    rows = validator.validate_submission(
        review_fixture["submission"],
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
        ledger=review_fixture["ledger"],
    )
    first = validator.merge_rows(
        rows,
        ledger_path=review_fixture["ledger_path"],
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
    )
    second = validator.merge_rows(
        rows,
        ledger_path=review_fixture["ledger_path"],
        repo_root=review_fixture["repo_root"],
        dataset_dir=review_fixture["dataset_dir"],
    )
    assert first == {"appended": 2, "already_present": 0}
    assert second == {"appended": 0, "already_present": 2}
    ledger = json.loads(review_fixture["ledger_path"].read_text(encoding="utf-8"))
    assert len(ledger["structure_reviews"]) == 1
    assert len(ledger["route_reviews"]) == 1
