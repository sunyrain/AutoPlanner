from __future__ import annotations

from pathlib import Path

import pytest

from cascade_planner.interfaces.case_dossier import (
    CaseDossierError,
    compile_case_dossier,
    with_case_dossier_digest,
)
from cascade_planner.interfaces.case_runner import run_case_dossier
from cascade_planner.interfaces.replay_pack import run_replay_pack
from cascade_planner.runtime.paths import RuntimePaths


def _dossier() -> dict:
    source_row = {
        "step_id": "oxidation",
        "claim_scope_id": "doi:10.1000/example:oxidation",
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "relation_type": "exact",
        "location_ref": "page 1",
        "condition_candidate": {"reagents": ["oxidant"]},
    }
    return with_case_dossier_digest(
        {
            "schema_version": "retrosynthesis_case_dossier.v1",
            "case_id": "acetaldehyde-dossier",
            "created_at": "2026-07-13T00:00:00Z",
            "target": {"name": "acetaldehyde", "smiles": "CC=O"},
            "acceptance": {
                "minimum_complete_routes": 1,
                "minimum_edge_proof_level": 3,
                "minimum_independent_source_groups": 1,
                "require_all_selected_leaves_stock_closed": True,
                "require_distinct_edge_sets": True,
                "stock_boundary": "procurement",
            },
            "budget": {
                "max_model_invocations": 0,
                "max_visual_invocations": 0,
                "max_accepted_expansions": 1,
                "max_attempt_runs": 5,
            },
            "routes": [
                {
                    "route_family_id": "route:oxidation",
                    "label": "ethanol oxidation",
                    "selected": True,
                    "steps": [
                        {
                            "step_id": "oxidation",
                            "product_smiles": "CC=O",
                            "reactant_smiles": ["CCO"],
                            "mapped_reaction_smiles": (
                                "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
                            ),
                            "source_refs": ["doi:10.1000/example"],
                        }
                    ],
                }
            ],
            "sources": [
                {
                    "binding": {
                        "artifact_sha256": "a" * 64,
                        "content_scope": "page 1",
                        "provenance": "deterministic_structure_parser_replay",
                        "source_kind": "paper_si",
                        "source_ref": "doi:10.1000/example",
                        "title": "Fixture exact oxidation",
                    },
                    "extractor": {
                        "producer_kind": "deterministic_structure_parser",
                        "producer_id": "test.case_dossier",
                        "version": "1.0.0",
                    },
                    "rows": [source_row],
                }
            ],
            "inventory": {
                "artifact": {
                    "schema_version": "versioned_inventory_snapshot.v1",
                    "adapter_version": "test.inventory.v1",
                    "inventory_version": "fixture-2026-07-13",
                    "retrieved_at": "2026-07-13T00:00:00Z",
                    "offers": [
                        {
                            "supplier": "Fixture Supplier",
                            "catalog_number": "ETOH-1",
                            "canonical_smiles": "CCO",
                            "checked_at": "2026-07-13T00:00:00Z",
                            "available": True,
                            "purity": "99%",
                            "pack_size": "1 L",
                            "price": 1.0,
                            "currency": "USD",
                            "region": "test",
                            "lead_time_days": 0,
                            "source_url": "https://example.test/ethanol",
                            "metadata": {"observation_kind": "test_fixture"},
                        }
                    ],
                },
                "as_of": "2026-07-13T01:00:00Z",
                "max_age_days": 1,
            },
        }
    )


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths.discover(
        repository_root=tmp_path,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index.sqlite3"),
        },
    )


def test_case_dossier_compiles_and_replays_through_canonical_pipeline(
    tmp_path: Path,
) -> None:
    pack = compile_case_dossier(_dossier())
    result = run_replay_pack(pack, paths=_paths(tmp_path))

    assert pack["expected"]["complete_route_count"] == 1
    assert pack["expected"]["stock_terminal_count"] == 1
    assert result["accepted"] is True
    assert result["observed"]["hyperedge_count"] == 1
    assert result["observed"]["attempt_count"] == 1
    assert result["observed"]["settled_task_count"] == 5
    assert result["observed"]["model_invocations"] == 0


def test_case_dossier_fails_closed_on_missing_source_stock_and_mapping() -> None:
    missing_source = _dossier()
    missing_source["sources"][0]["rows"] = []
    missing_source = with_case_dossier_digest(missing_source)
    with pytest.raises(CaseDossierError, match="reaction_without_exact_source"):
        compile_case_dossier(missing_source)

    missing_stock = _dossier()
    missing_stock["inventory"]["artifact"]["offers"] = []
    missing_stock = with_case_dossier_digest(missing_stock)
    with pytest.raises(CaseDossierError, match="stock_leaf_missing"):
        compile_case_dossier(missing_stock)

    missing_mapping = _dossier()
    missing_mapping["routes"][0]["steps"][0]["mapped_reaction_smiles"] = ""
    missing_mapping = with_case_dossier_digest(missing_mapping)
    with pytest.raises(CaseDossierError, match="atom_mapping_required"):
        compile_case_dossier(missing_mapping)


def test_case_dossier_requires_distinct_route_edge_sets() -> None:
    duplicated_route = _dossier()
    duplicated_route["routes"].append(
        {
            **duplicated_route["routes"][0],
            "route_family_id": "route:duplicate",
        }
    )
    duplicated_route["acceptance"]["minimum_complete_routes"] = 2
    duplicated_route = with_case_dossier_digest(duplicated_route)

    with pytest.raises(CaseDossierError, match="route_edge_sets_not_distinct"):
        compile_case_dossier(duplicated_route)


def test_case_dossier_one_command_run_exports_offline_workbench(
    tmp_path: Path,
) -> None:
    result = run_case_dossier(
        _dossier(),
        paths=_paths(tmp_path),
        run_id="one-command-dossier",
        output_dir=tmp_path / "showcase",
    )

    assert result["accepted"] is True
    assert result["status"] == "completed"
    assert result["model_invocations"] == 0
    assert result["visual_invocations"] == 0
    assert result["timing_seconds"]["total"] >= 0
    assert Path(result["export"]["files"]["html"]).is_file()


def test_checked_in_artemisinin_case_closes_with_two_procurement_boundaries(
    tmp_path: Path,
) -> None:
    dossier = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "examples"
        / "artemisinin_v4_case_dossier.json"
    )
    pack = compile_case_dossier(dossier)
    result = run_replay_pack(pack, paths=_paths(tmp_path))

    assert pack["expected"] == {
        "accepted": True,
        "accepted_expansion_count": 2,
        "complete_route_count": 2,
        "exact_record_count": 3,
        "hyperedge_count": 2,
        "independent_source_groups": [
            "doi:10.1002/anie.201801424",
            "doi:10.1038/nature12051",
        ],
        "model_invocations": 0,
        "selected_route_count": 2,
        "stock_terminal_count": 4,
        "validated_edge_count": 2,
        "visual_invocations": 0,
    }
    assert result["accepted"] is True
    assert result["observed"]["complete_route_count"] == 2
    assert result["observed"]["attempt_count"] == 2
    assert result["observed"]["settled_task_count"] == 9
    assert result["observed"]["model_invocations"] == 0
