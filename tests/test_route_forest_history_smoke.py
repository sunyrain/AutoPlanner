from __future__ import annotations

import json

from scripts.smoke_route_forest_history import (
    HistorySmokeConfig,
    route_forest_html_contract_reasons,
    smoke_route_forest_history,
)


def test_smoke_route_forest_history_accepts_nonempty_saved_run(tmp_path) -> None:
    run_dir = tmp_path / "diagnostic_run"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "diagnostic_run",
                "target_profile": {"target_name": "aspirin", "target_smiles": "CCO"},
                "route_failures": [
                    {
                        "reason": "chemenzy_missing_output",
                        "artifact_ref": "guided_chemenzy_result.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = smoke_route_forest_history(HistorySmokeConfig(root=tmp_path))

    assert summary["accepted"], summary
    assert summary["checked"] == 1
    assert summary["compiled"] == 1
    assert summary["zero_branch"] == 0
    assert summary["zero_step"] == 0
    assert summary["html_bad"] == 0
    assert summary["rows"][0]["branch_kinds"] == ["diagnostic_failure"]


def test_smoke_route_forest_history_rejects_empty_blackboard_without_branch(tmp_path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps({"case_id": "empty_run"}),
        encoding="utf-8",
    )

    summary = smoke_route_forest_history(HistorySmokeConfig(root=tmp_path))

    assert not summary["accepted"]
    assert summary["checked"] == 1
    assert summary["compiled"] == 1
    assert summary["zero_branch"] == 1
    assert "route_forest_has_zero_branches" in summary["rows"][0]["reasons"]
    assert "route_forest_has_zero_steps" in summary["rows"][0]["reasons"]


def test_html_contract_does_not_depend_on_localized_title_or_route_index() -> None:
    forest = {"schema_version": "explored_route_forest.v1", "branches": [], "steps": []}
    html = f'''<!doctype html>
    <html><body data-readonly-note="does not run new planning">
      <main id="mainRoute"></main><aside id="detail"></aside>
      <script id="forest-data" type="application/json">{json.dumps(forest)}</script>
    </body></html>'''

    assert route_forest_html_contract_reasons(html, expected_forest=forest) == []


def test_html_contract_rejects_stale_or_unrelated_embedded_forest() -> None:
    expected = {"schema_version": "explored_route_forest.v1", "case_id": "current"}
    embedded = {"schema_version": "explored_route_forest.v1", "case_id": "stale"}
    html = f'''<!doctype html>
    <html><body data-readonly-note="does not run new planning">
      <main id="mainRoute"></main><aside id="detail"></aside>
      <script id="forest-data" type="application/json">{json.dumps(embedded)}</script>
    </body></html>'''

    assert route_forest_html_contract_reasons(html, expected_forest=expected) == [
        "route_forest_html_embedded_forest_mismatch"
    ]
