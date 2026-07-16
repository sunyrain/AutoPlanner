from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.web.workspace_catalog import compile_showcase_catalog


def test_fresh_showcase_replaces_legacy_target_and_projects_closure_axes(
    tmp_path: Path,
) -> None:
    root = tmp_path
    shared = root / "results" / "shared"
    fresh = shared / "bufotalin-fresh" / "showcase"
    fresh.mkdir(parents=True)
    (fresh / "bufotalin-workbench.html").write_text("<html>fresh</html>", encoding="utf-8")
    (fresh / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "v4_blind_expert_showcase.v1",
                "generated_at": "2026-07-16T03:42:55Z",
                "targets": [
                    {
                        "target_name": "bufotalin",
                        "run_id": "bufotalin-fresh-v3",
                        "status": "completed",
                        "claim": "reaction_validated",
                        "maturity": {"label": "3 条结构闭合 · 证据/库存开放"},
                        "evidence": {"sources": 4, "bindings": 0},
                        "gates": {"B0": True, "B1": False},
                        "workbench": {
                            "declared_program_count": 4,
                            "graph_closed_program_count": 3,
                            "graph_open_program_count": 1,
                            "longest_graph_closed_step_count": 12,
                            "process_ready_route_count": 0,
                        },
                        "workbench_file": "bufotalin-workbench.html",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    legacy_workbench = shared / "legacy" / "route.html"
    legacy_workbench.parent.mkdir()
    legacy_workbench.write_text("<html>legacy 20</html>", encoding="utf-8")
    manifest = shared / "presentation" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "bufotalin-v4-20-step",
                        "target_name": "bufotalin",
                        "max_step_count": 20,
                        "artifact_path": str(legacy_workbench.relative_to(root)),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    catalog = compile_showcase_catalog(
        root=root,
        shared_root=shared,
        manifest_path=manifest,
    )

    assert catalog["schema_version"] == "autoplanner.presentation_showcase.v2"
    assert catalog["standard_case_id"] == "bufotalin-fresh-v3"
    assert len(catalog["cases"]) == 1
    case = catalog["cases"][0]
    assert case["graph_closed_program_count"] == 3
    assert case["declared_program_count"] == 4
    assert case["max_step_count"] == 12
    assert case["available"] is True
    assert case["artifact_url"].startswith("/api/v4/result-file?path=")
    assert catalog["semantics"]["route_length_is_descriptive_not_an_objective"] is True


def test_classic_multistep_summary_is_published_as_showcase_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path
    shared = root / "results" / "shared"
    benchmark = shared / "paroutes-panel" / "benchmark"
    benchmark.mkdir(parents=True)
    (benchmark / "index.html").write_text("<html>panel</html>", encoding="utf-8")
    (benchmark / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "classic_multistep_benchmark_summary.v1",
                "aggregates": {
                    "overall": {
                        "target_count": 20,
                        "chem_enzy_route_found_rate": 0.95,
                        "cascade_route_retained_rate": 0.9,
                        "cascade_accepted_rate": 0.7,
                        "benchmark_stock_closed_rate": 0.75,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = compile_showcase_catalog(root=root, shared_root=shared)

    assert len(catalog["audits"]) == 1
    audit = catalog["audits"][0]
    assert audit["available"] is True
    assert audit["target_count"] == 20
    assert audit["route_retained_rate"] == 0.9
