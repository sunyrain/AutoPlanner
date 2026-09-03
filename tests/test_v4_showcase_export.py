from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from cascade_planner.web.v4_showcase_export import (
    build_run_export_html,
    export_run_showcase,
    normalize_branch_indices,
    showcase_filename,
)


def _event(**values):
    return {
        "schema_version": "sequential_director_model_io.v1",
        "timestamp": "2026-08-27T08:00:00Z",
        "model": "gpt-5.6-sol",
        **values,
    }


def _saved_run(tmp_path: Path) -> Path:
    target = "CC(=O)Oc1ccccc1C(=O)O"
    run_dir = tmp_path / "runs" / "showcase-target"
    director = run_dir / ".autoplanner" / "director-workspace"
    director.mkdir(parents=True)
    (run_dir / "target-only-solve-report.json").write_text(
        json.dumps(
            {
                "run_id": "showcase-run",
                "target": {"name": "Aspirin", "canonical_smiles": target},
                "accepted_expansion_count": 1,
                "model_cost": {
                    "model_invocations": 2,
                    "input_tokens": 200,
                    "output_tokens": 70,
                    "wall_time_s": 12.5,
                },
                "current_disposition": {"state": "stock_closed_proof_open"},
                "candidate_lifecycle": {
                    "records": [
                        {
                            "candidate_id": "candidate:1",
                            "product_smiles": target,
                            "precursor_smiles": [
                                "O=C(O)c1ccccc1O",
                                "CC(=O)OC(C)=O",
                            ],
                            "route_family_ids": ["route-family:test:1"],
                            "materialization": {"materialized": True},
                            "origin_records": [
                                {"origin_kind": "codex_global_director"}
                            ],
                        },
                        {
                            "candidate_id": "aiz:tail:1",
                            "product_smiles": "O=C(O)c1ccccc1O",
                            "precursor_smiles": ["Oc1ccccc1Br", "O=C=O"],
                            "route_family_ids": ["route-family:test:1"],
                            "materialization": {"materialized": True},
                            "origin_records": [
                                {
                                    "origin_kind": "aizynthfinder",
                                    "proposal_id": "aizynthfinder:test:route:1:step:1",
                                    "transformation_hypothesis": "AIz short-tail disconnection",
                                    "provider_reaction_metadata": {
                                        "mode": "short_tail"
                                    },
                                }
                            ],
                        },
                        {
                            "candidate_id": "aiz:unreachable",
                            "product_smiles": "CCO",
                            "precursor_smiles": ["CC=O"],
                            "route_family_ids": ["route-family:test:1"],
                            "materialization": {"materialized": True},
                            "origin_records": [
                                {
                                    "origin_kind": "aizynthfinder",
                                    "proposal_id": "aizynthfinder:test:route:2:step:1",
                                    "provider_reaction_metadata": {
                                        "mode": "short_tail"
                                    },
                                }
                            ],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    strategies = _event(
        event="model_output",
        artifact_type="StrategyPortfolioReport",
        task_id="director:test:strategy-portfolio:1",
        status="accepted_draft",
        usage={"input_tokens": 120, "output_tokens": 30},
        output_artifact={
            "payload": {
                "target_smiles": target,
                "strategy_cards": [
                    {
                        "strategy_signature": f"Strategy {index}",
                        "strategy_query": f"Query {index}",
                    }
                    for index in range(1, 4)
                ],
            }
        },
    )
    route = _event(
        event="model_output",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:1:node:1",
        status="accepted_draft",
        usage={"input_tokens": 80, "output_tokens": 40},
        output_artifact={
            "payload": {
                "candidates": [
                    {
                        "candidate_id": "candidate:1",
                        "product_smiles": target,
                        "precursor_smiles": ["N"],
                        "reaction_family": "Phenolic O-acetylation",
                    }
                ]
            }
        },
    )
    replay = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:1:node:2",
        prompt=(
            "instructions\nPaperMatchedRouteBuilderContext:\n"
            + json.dumps(
                {
                    "campaign_target": target,
                    "accepted_path": [
                        {
                            "step_id": "candidate:1",
                            "product_smiles": target,
                            "precursor_smiles": [
                                "O=C(O)c1ccccc1O",
                                "CC(=O)OC(C)=O",
                            ],
                            "reaction_family": "Phenolic O-acetylation",
                            "conditions": ["Ac2O"],
                        }
                    ],
                }
            )
        ),
    )
    (director / "model-io.jsonl").write_text(
        "\n".join(json.dumps(value) for value in (strategies, route, replay)) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _saved_three_branch_run(tmp_path: Path) -> Path:
    run_dir = _saved_run(tmp_path)
    stream = run_dir / ".autoplanner" / "director-workspace" / "model-io.jsonl"
    target = "CC(=O)Oc1ccccc1C(=O)O"
    rows = []
    for branch_index, draft_precursor, host_precursor in (
        (2, "Cl", "CCO"),
        (3, "Br", "CCN"),
    ):
        rows.extend(
            (
                _event(
                    event="model_output",
                    artifact_type="RetrosynthesisProposalReport",
                    task_id=f"director:test:branch:{branch_index}:node:1",
                    status="accepted_draft",
                    usage={"input_tokens": 30, "output_tokens": 10},
                    output_artifact={
                        "payload": {
                            "candidates": [
                                {
                                    "candidate_id": f"candidate:{branch_index}",
                                    "product_smiles": target,
                                    "precursor_smiles": [draft_precursor],
                                    "reaction_family": f"Branch {branch_index} draft",
                                }
                            ]
                        }
                    },
                ),
                _event(
                    event="model_input",
                    artifact_type="RetrosynthesisProposalReport",
                    task_id=f"director:test:branch:{branch_index}:node:2",
                    prompt=(
                        "instructions\nPaperMatchedRouteBuilderContext:\n"
                        + json.dumps(
                            {
                                "campaign_target": target,
                                "accepted_path": [
                                    {
                                        "step_id": f"candidate:{branch_index}",
                                        "product_smiles": target,
                                        "precursor_smiles": [host_precursor],
                                        "reaction_family": (
                                            f"Branch {branch_index} host route"
                                        ),
                                    }
                                ],
                            }
                        )
                    ),
                ),
            )
        )
    with stream.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(value) for value in rows) + "\n")
    return run_dir


def _export_payload(body: str) -> dict:
    match = re.search(
        r'<script id="showcaseData" type="application/json">(.*?)</script>',
        body,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_showcase_is_a_self_contained_playable_document(tmp_path: Path) -> None:
    run_dir = _saved_run(tmp_path)
    body = build_run_export_html(run_dir=run_dir, export_kind="interaction")

    assert body.startswith("<!doctype html>")
    assert "__AUTOPLANNER_SHOWCASE_DATA__" not in body
    assert 'id="play"' in body
    assert 'id="timeline"' in body
    assert 'id="speed"' in body
    assert 'id="graphCanvas"' in body
    assert 'id="routeScale"' in body
    assert 'id="routeTree"' in body
    assert "routeTreeMarkup" in body
    assert "data-replay-step" in body
    assert "frameForStep" in body
    assert 'id="reactionDetail"' in body
    assert 'id="reactionDetailBody"' in body
    assert 'id="reactionReplayJump"' in body
    assert "showReactionDetail" in body
    assert 'id="playbackLayout"' in body
    assert 'id="activityRail"' in body
    assert 'id="activityToggle"' in body
    assert 'id="activityList"' in body
    assert "projection.activities" in body
    assert "renderActivityStream" in body
    assert "activitiesForPath" in body
    assert "activityCollapsed" in body
    assert "autoplanner.showcase.activity-rail" in body
    assert "resetPlaybackView" in body
    assert "resetPlaybackView();if(state.index>=state.frames.length-1)" in body
    assert "frame.proposed_step_ids" in body
    assert "草稿涉及" in body
    assert "indexChanged&&state.frames[state.index]?.topology_changed" not in body
    assert "Critic 依据" in body
    assert "结构与 SMILES" in body
    assert "reactionTechnical" in body
    assert "信息来源边界" not in body
    assert "页面不根据隐式思维过程补写原因" not in body
    assert ">Step ID<" not in body
    assert 'id="layoutToggle"' in body
    assert "setRouteOrientation" in body
    assert "horizontalRoute" in body
    assert "availableHeight" in body
    assert 'id="compactToggle"' in body
    assert "setCompactRoute" in body
    assert "compactModeStorageKey" in body
    assert "compactConnector" in body
    assert "if(state.compactRoute)return" in body
    assert "routeAlternativeSet" in body
    assert "routeAlternativeLane" in body
    assert "routeAlternativeMarkup" in body
    assert "routeAlternativeFork" in body
    assert "syncAlternativeForkPaths" in body
    assert "data-route-lane-toggle" in body
    assert "toggleRouteLaneFocus" in body
    assert "orientation:'horizontal'" in body
    assert "routeOrientation='horizontal'" in body
    assert 'id="layoutToggle" class="control layout-control" type="button" aria-pressed="true"' in body
    assert "inspecting:false" in body
    assert "highlightFrame=state.inspecting?frame:null" in body
    assert "state.inspecting=state.index<state.frames.length-1" in body
    assert "left:0;right:0;bottom:0" in body
    assert "overflow-y:auto;overflow-x:hidden" in body
    assert "word-break:break-all" in body
    assert "来源 / 状态" in body
    assert "条件 / 催化剂" in body
    assert "产物 ·" in body
    assert "前体 ·" in body
    assert "步骤来源与状态" not in body
    assert ">产物 SMILES<" not in body
    assert ">前体 SMILES" not in body
    assert "pointerdown" in body
    assert "pointermove" in body
    assert "setPointerCapture" in body
    assert "closest('button,a,input,select,textarea,[data-replay-step],[data-route-lane-toggle]')" in body
    assert "translate3d" in body
    assert "passive:false" in body
    assert "滚轮缩放" in body
    assert "event.ctrlKey" not in body
    assert '<div class="reaction-scheme"' not in body
    assert 'id="scheme"' not in body
    assert "Host route ledger" not in body
    assert "Phenolic O-acetylation" in body
    assert "fetch(" not in body
    assert "EventSource" not in body
    assert "<script src=" not in body
    assert "<link rel=\"stylesheet\"" not in body
    assert "<svg" in body

    match = re.search(
        r'<script id="showcaseData" type="application/json">(.*?)</script>',
        body,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["metadata"]["run_id"] == "showcase-run"
    assert payload["metadata"]["target_name"] == "Aspirin"
    assert payload["metadata"]["frame_count"] >= 4
    assert payload["summary"]["model_invocations"] == 2
    assert len(payload["projection"]["activities"]) == 2
    assert payload["semantics"]["offline_single_file"] is True
    assert set(payload["molecules"]) >= {
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(O)c1ccccc1O",
        "CC(=O)OC(C)=O",
        "N",
    }


def test_showcase_export_writes_one_html_file(tmp_path: Path) -> None:
    run_dir = _saved_run(tmp_path)
    output = tmp_path / "exports" / "run.html"

    receipt = export_run_showcase(run_dir=run_dir, output_path=output)

    assert output.is_file()
    assert receipt["self_contained"] is True
    assert receipt["size_bytes"] == output.stat().st_size
    assert output.read_text(encoding="utf-8").endswith("</html>\n")


def test_static_route_and_complete_graph_are_distinct_exports(tmp_path: Path) -> None:
    run_dir = _saved_run(tmp_path)

    route = build_run_export_html(
        run_dir=run_dir,
        export_kind="route",
        branch_index=1,
    )
    graph = build_run_export_html(
        run_dir=run_dir,
        export_kind="graph",
    )

    assert "静态逆合成路线" in route
    assert "逐步逆合成路线" in route
    assert 'id="play"' not in route
    assert '"export_kind":"route"' in route
    assert '"branch_index":1' in route
    assert '"N":' not in route
    assert "完整路线图" in graph
    assert 'id="routeTree"' in graph
    assert 'id="zoomFit"' in graph
    assert 'id="originSummary"' in graph
    assert "AIz 收尾" in graph
    assert "大模型步" in graph
    assert "pointerdown" in graph
    assert "pointermove" in graph
    assert "pointerup" in graph
    assert "setPointerCapture" in graph
    assert "translate3d" in graph
    assert "panX" in graph
    assert "panY" in graph
    assert "addEventListener('wheel'" in graph
    assert "passive:false" in graph
    assert "event.ctrlKey" not in graph
    assert "滚轮缩放" in graph
    assert "routeAlternativeSet" in graph
    assert "routeAlternativeLane" in graph
    assert "routeAlternativeFork" in graph
    assert "syncAlternativeForkPaths" in graph
    assert "data-route-lane-toggle" in graph
    assert "touch-action:none" in graph
    assert "overflow:hidden" in graph
    assert "height:clamp(540px,calc(100vh - 245px),780px)" in graph
    assert "scrollLeft" not in graph
    assert "scrollTop" not in graph
    assert 'id="play"' not in graph
    assert '"export_kind":"graph"' in graph

    match = re.search(
        r'<script id="showcaseData" type="application/json">(.*?)</script>',
        graph,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    branch = payload["projection"]["branches"][0]
    assert [step["step_origin"] for step in branch["steps"]] == [
        "large_model",
        "aizynthfinder_short_tail",
    ]
    assert payload["metadata"]["step_origin_counts"] == {
        "aizynthfinder_short_tail": 1,
        "large_model": 1,
        "unknown": 0,
    }
    assert all(step["product_smiles"] != "CCO" for step in branch["steps"])


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ((2,), [2]),
        ((1, 3), [1, 3]),
        ((1, 2, 3), [1, 2, 3]),
    ],
)
def test_exports_keep_only_the_selected_route_subset(
    tmp_path: Path,
    selection: tuple[int, ...],
    expected: list[int],
) -> None:
    run_dir = _saved_three_branch_run(tmp_path)

    payload = _export_payload(
        build_run_export_html(
            run_dir=run_dir,
            export_kind="interaction",
            branch_indices=selection,
        )
    )

    assert payload["metadata"]["branch_index"] == expected[0]
    assert payload["metadata"]["branch_indices"] == expected
    assert payload["metadata"]["branch_selection_count"] == len(expected)
    assert [
        branch["branch_index"] for branch in payload["projection"]["branches"]
    ] == expected
    assert [
        strategy["index"] for strategy in payload["projection"]["strategies"]
    ] == expected
    assert {
        activity["branch_index"]
        for activity in payload["projection"]["activities"]
        if activity["branch_index"] is not None
    }.issubset(set(expected))
    assert all(
        frame.get("branch_index") in (None, *expected)
        for frame in payload["replay"]["frames"]
    )
    assert all(
        update["branch_index"] in expected
        for frame in payload["replay"]["frames"]
        for update in frame.get("branch_updates") or []
    )
    assert all(
        strategy["index"] in expected
        for frame in payload["replay"]["frames"]
        for strategy in frame.get("strategies") or []
    )


def test_two_route_export_omits_unselected_steps_molecules_and_events(
    tmp_path: Path,
) -> None:
    run_dir = _saved_three_branch_run(tmp_path)

    route_body = build_run_export_html(
        run_dir=run_dir,
        export_kind="route",
        branch_indices=(1, 3),
    )
    payload = _export_payload(
        build_run_export_html(
            run_dir=run_dir,
            export_kind="interaction",
            branch_indices=(1, 3),
        )
    )

    assert 'id="strategyList"' in route_body
    assert "branchIndices.map" in route_body
    assert set(payload["molecules"]) >= {"CCN", "Br"}
    assert "CCO" not in payload["molecules"]
    assert "Cl" not in payload["molecules"]
    assert all(
        activity.get("branch_index") != 2
        for activity in payload["projection"]["activities"]
    )
    assert all(
        frame.get("branch_index") != 2
        for frame in payload["replay"]["frames"]
    )


def test_branch_selection_validation_and_filenames() -> None:
    assert normalize_branch_indices("3,1,3") == (1, 3)
    assert showcase_filename(
        "run one",
        export_kind="graph",
        branch_indices=(1, 3),
    ) == "run-one-strategies-1-3-complete-route-graph.html"
    assert showcase_filename(
        "run one",
        export_kind="route",
        branch_indices=(2,),
    ) == "run-one-strategy-2-static-route.html"
    with pytest.raises(ValueError, match="showcase_branch_indices_empty"):
        normalize_branch_indices(())
    with pytest.raises(ValueError, match="showcase_branch_index_invalid"):
        normalize_branch_indices("1,4")


def test_unrecorded_step_origin_is_not_guessed(tmp_path: Path) -> None:
    run_dir = _saved_run(tmp_path)
    report_path = run_dir / "target-only-solve-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidate_lifecycle"] = {"records": []}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    body = build_run_export_html(
        run_dir=run_dir,
        export_kind="graph",
    )
    match = re.search(
        r'<script id="showcaseData" type="application/json">(.*?)</script>',
        body,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    step = payload["projection"]["branches"][0]["steps"][0]
    assert step["step_origin"] == "unknown"
    assert step["step_origin_label"] == "来源未记录"
    assert payload["metadata"]["step_origin_counts"]["large_model"] == 0
