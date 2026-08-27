from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

from flask import Flask
from PIL import Image, ImageChops

from cascade_planner.web.v4_api import create_v4_blueprint
from cascade_planner.web.v4_live_synthesis import (
    project_live_synthesis,
    render_molecule_png,
    render_molecule_svg,
)


def _event(**values):
    return {
        "schema_version": "sequential_director_model_io.v1",
        "timestamp": "2026-08-24T08:00:00Z",
        "model": "gpt-5.6-sol",
        **values,
    }


def test_live_projection_replaces_model_candidate_with_host_replayed_path(
    tmp_path: Path,
) -> None:
    target = "CC(=O)Oc1ccccc1C(=O)O"
    strategy_output = _event(
        event="model_output",
        artifact_type="StrategyPortfolioReport",
        task_id="director:test:strategy-portfolio:1",
        status="accepted_draft",
        usage={"input_tokens": 120, "output_tokens": 30},
        output_artifact={
            "summary": "three strategies",
            "payload": {
                "target_smiles": target,
                "strategy_cards": [
                    {"strategy_signature": f"Strategy {index}", "strategy_query": f"Query {index}"}
                    for index in range(1, 4)
                ],
            },
        },
    )
    route_output = _event(
        event="model_output",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:1:node:2",
        status="accepted_draft",
        usage={"input_tokens": 80, "output_tokens": 40},
        output_artifact={
            "summary": "one route step",
            "payload": {
                "candidates": [
                    {
                        "candidate_id": "candidate:1",
                        "product_smiles": target,
                        "precursor_smiles": [],
                        "reaction_family": "phenolic O-acetylation",
                    }
                ]
            },
        },
    )
    context = {
        "campaign_target": target,
        "accepted_path": [
            {
                "product_smiles": target,
                "precursor_smiles": ["O=C(O)c1ccccc1O", "CC(=O)OC(C)=O"],
                "reaction_family": "phenolic O-acetylation",
                "conditions": ["Ac2O"],
            }
        ],
    }
    replay_input = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:1:node:3",
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
        + json.dumps(context),
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(strategy_output),
                json.dumps(route_output),
                "{malformed",
                json.dumps(replay_input),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={
            "job_id": "solve:live",
            "run_id": "live",
            "target_name": "target",
            "status": "running",
        },
    )

    assert len(projection["strategies"]) == 3
    assert projection["target_smiles"] == target
    assert projection["phase"] == "route_building"
    assert projection["model_output_count"] == 2
    assert projection["parse_error_count"] == 1
    assert projection["usage"] == {
        "input_tokens": 200,
        "output_tokens": 70,
        "model_invocations": 2,
    }
    branch = projection["branches"][0]
    assert branch["pending_step"] is None
    assert branch["steps"][0]["precursor_smiles"] == [
        "O=C(O)c1ccccc1O",
        "CC(=O)OC(C)=O",
    ]
    assert branch["steps"][0]["status"] == "host_replayed"


def test_live_projection_exposes_pending_output_before_host_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        json.dumps(
            _event(
                event="model_output",
                artifact_type="RetrosynthesisProposalReport",
                task_id="director:test:branch:2:node:4",
                status="accepted_draft",
                usage={"output_tokens": 22},
                output_artifact={
                    "payload": {
                        "candidates": [
                            {
                                "candidate_id": "pending",
                                "product_smiles": "CCO",
                                "precursor_smiles": [],
                                "reaction_family": "C-C disconnection",
                            }
                        ]
                    }
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:pending", "run_id": "pending", "status": "running"},
    )

    pending = projection["branches"][1]["pending_step"]
    assert pending["reaction_family"] == "C-C disconnection"
    assert pending["status"] == "pending_host_replay"


def test_live_projection_recovers_paper_matched_selected_leaf(
    tmp_path: Path,
) -> None:
    target = "CC(=O)O"
    draft = _event(
        event="model_output",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:2",
        status="accepted_draft",
        output_artifact={
            "payload": {
                "candidates": [
                    {
                        "candidate_id": "paper-step-1",
                        "product_smiles": target,
                        "precursor_smiles": [],
                        "reaction_family": "ester hydrolysis",
                    }
                ]
            }
        },
    )
    context = {
        "target_smiles": target,
        "connected_path_reactions": [
            {
                "step_id": "paper-step-1",
                "reaction_family": "ester hydrolysis",
                "edit_summary": "replace the acyl substituent with hydroxyl",
            }
        ],
        "selected_leaf_mapped": "[CH3:1][CH2:2][OH:3]",
    }
    replay = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:3",
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
        + json.dumps(context),
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (draft, replay)) + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:paper", "run_id": "paper", "status": "running"},
    )

    branch = projection["branches"][1]
    assert branch["pending_step"] is None
    assert len(branch["steps"]) == 1
    assert branch["steps"][0]["step_id"] == "paper-step-1"
    assert branch["steps"][0]["product_smiles"] == target
    assert branch["steps"][0]["precursor_smiles"] == ["CCO"]
    assert branch["steps"][0]["reaction_family"] == "ester hydrolysis"
    assert branch["steps"][0]["topology_status"] == "root"


def test_historical_running_kernel_is_projected_as_interrupted() -> None:
    projection = project_live_synthesis(
        model_io_path=None,
        job={
            "job_id": "solve:orphaned",
            "run_id": "orphaned",
            "status": "historical",
            "campaign_status": "running",
            "campaign_terminal": False,
        },
    )

    assert projection["status"] == "interrupted"
    assert projection["phase"] == "interrupted"
    assert projection["progress"] < 100
    assert {branch["status"] for branch in projection["branches"]} == {
        "interrupted"
    }


def test_historical_terminal_decision_overrides_stale_running_status() -> None:
    projection = project_live_synthesis(
        model_io_path=None,
        job={
            "job_id": "solve:settled",
            "run_id": "settled",
            "status": "historical",
            "campaign_status": "running",
            "campaign_terminal": True,
            "campaign_decision": "unresolved",
        },
    )

    assert projection["status"] == "unresolved"
    assert projection["phase"] == "unresolved"
    assert projection["progress"] == 100


def test_legacy_stop_signal_does_not_settle_route_builder_branch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        json.dumps(
            _event(
                event="model_output",
                artifact_type="RetrosynthesisProposalReport",
                task_id="director:test:branch:1:node:2",
                status="accepted_draft",
                output_artifact={
                    "summary": "no useful strategic edit",
                    "payload": {
                        "candidates": [],
                        "stop_signal": True,
                        "stop_reason": "no_reasonable_disconnection",
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:stopped", "run_id": "stopped", "status": "running"},
    )

    assert projection["branches"][0]["status"] == "reviewing"


def test_live_projection_uses_compact_and_critic_host_replay_contexts(
    tmp_path: Path,
) -> None:
    target = "CCOC(=O)c1ccccc1"
    first_step = {
        "step_id": "host-step-1",
        "product_smiles": target,
        "precursor_smiles": [
            "O=C(O)c1ccccc1",
            "O=C(O)c1ccccc1",
            "CCO",
        ],
        "reaction_family": "esterification",
    }
    compact_input = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:3",
        prompt="instructions\nCompactBranchContext:\n"
        + json.dumps(
            {
                "branch_id": 2,
                "campaign_target": target,
                "accepted_path": [first_step],
            }
        ),
    )
    route_output = _event(
        event="model_output",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:editor:4",
        status="accepted_draft",
        output_artifact={
            "payload": {
                "candidates": [
                    {
                        "candidate_id": "editor-draft",
                        "product_smiles": target,
                        "precursor_smiles": [],
                        "reaction_family": "edited esterification",
                        "route_json": [
                            {
                                "step_id": "model-draft-without-replay",
                                "product_smiles": target,
                                "precursor_smiles": [],
                                "reaction_family": "draft only",
                            }
                        ],
                    }
                ]
            }
        },
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (compact_input, route_output)) + "\n",
        encoding="utf-8",
    )

    pending_projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:compact", "run_id": "compact", "status": "running"},
    )
    branch = pending_projection["branches"][1]
    assert [row["step_id"] for row in branch["steps"]] == ["host-step-1"]
    assert branch["pending_step"]["step_id"] == "editor-draft"

    final_steps = [
        first_step,
        {
            "step_id": "host-step-2",
            "product_smiles": "O=C(O)c1ccccc1",
            "precursor_smiles": ["O=C(Cl)c1ccccc1", "O"],
            "transformation_hypothesis": "acid hydrolysis",
            "condition_predictions": [
                {
                    "catalyst": "sulfuric acid",
                    "reagents": ["water, controlled heating"],
                }
            ],
        },
        {
            "step_id": "host-step-3",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC=O", "O"],
            "reaction_family": "alcohol oxidation",
        },
    ]
    critic_input = _event(
        event="model_input",
        artifact_type="ChemicalStrategyCritique",
        task_id="critic:test",
        prompt="critic instructions\nBlindRouteCriticInput:\n"
        + json.dumps(
            {
                "branch_id": 2,
                "campaign_target": target,
                "steps": final_steps,
            }
        ),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(critic_input) + "\n")

    replayed_projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:compact", "run_id": "compact", "status": "running"},
    )
    branch = replayed_projection["branches"][1]
    assert branch["pending_step"] is None
    assert branch["status"] == "reviewing"
    assert [row["step_id"] for row in branch["steps"]] == [
        "host-step-1",
        "host-step-2",
        "host-step-3",
    ]
    assert all(row["precursor_smiles"] for row in branch["steps"])
    assert branch["steps"][1]["reaction_family"] == "acid hydrolysis"
    assert branch["steps"][1]["conditions"] == ["water, controlled heating"]
    assert branch["steps"][1]["catalyst"] == "sulfuric acid"
    assert branch["steps"][0]["topology_status"] == "root"
    assert branch["steps"][1]["topology_status"] == "linked"
    assert branch["steps"][1]["parent_step_index"] == 0
    assert branch["steps"][0]["continuation_precursor_indices"] == [0, 1]
    assert branch["steps"][0]["display_precursors"][0][
        "continues_to_next_step"
    ] is True
    assert branch["steps"][0]["display_precursors"][0]["count"] == 2
    assert branch["steps"][0]["display_precursors"][0][
        "child_step_indices"
    ] == [1]
    assert branch["steps"][0]["display_precursors"][1][
        "child_step_indices"
    ] == [2]
    assert branch["steps"][2]["parent_step_index"] == 0


def test_paper_matched_review_transition_settles_prior_branches(
    tmp_path: Path,
) -> None:
    target = "CCOC(=O)c1ccccc1"

    def route_step(branch_index: int) -> dict:
        return {
            "step_id": f"host-step-{branch_index}",
            "product_smiles": target,
            "precursor_smiles": [f"CCO{branch_index}"],
            "reaction_family": f"route {branch_index}",
        }

    def critic_input(branch_index: int, call_index: int) -> dict:
        return _event(
            event="model_input",
            artifact_type="ChemicalStrategyCritique",
            task_id=f"critic:{call_index}",
            prompt="critic instructions\nPaperMatchedRouteCriticInput:\n"
            + json.dumps(
                {
                    "branch_id": branch_index,
                    "campaign_target": target,
                    "steps": [route_step(branch_index)],
                }
            ),
        )

    def critic_output(call_index: int) -> dict:
        return _event(
            event="model_output",
            artifact_type="ChemicalStrategyCritique",
            task_id=f"critic:{call_index}",
            status="accepted_draft",
            output_artifact={"summary": "review complete", "payload": {}},
        )

    branch_one_editor = _event(
        event="model_output",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:1:editor:5",
        status="accepted_draft",
        output_artifact={
            "payload": {
                "candidates": [
                    {
                        "candidate_id": "branch-one-editor-draft",
                        "product_smiles": target,
                        "precursor_smiles": [],
                    }
                ]
            }
        },
    )
    events = [
        critic_input(1, 1),
        critic_output(1),
        branch_one_editor,
        critic_input(2, 2),
        critic_output(2),
        critic_input(3, 3),
    ]
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:review", "run_id": "review", "status": "running"},
    )

    assert [branch["status"] for branch in projection["branches"]] == [
        "complete",
        "complete",
        "reviewing",
    ]
    assert projection["branches"][0]["pending_step"] is None
    assert [
        activity["branch_index"]
        for activity in projection["activities"]
        if activity["kind"] == "critic"
    ] == [1, 2]


def test_paper_matched_critic_mapped_structures_remain_renderable(
    tmp_path: Path,
) -> None:
    critic_input = _event(
        event="model_input",
        artifact_type="ChemicalStrategyCritique",
        task_id="critic:mapped-structures",
        prompt="critic instructions\nPaperMatchedRouteCriticInput:\n"
        + json.dumps(
            {
                "branch_id": 1,
                "campaign_target": "CCO",
                "steps": [
                    {
                        "step_id": "mapped-step-1",
                        "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                        "mapped_precursor_smiles": ["[CH3:1][CH:2]=[O:3]"],
                        "reaction_family": "carbonyl reduction",
                    }
                ],
            }
        ),
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(json.dumps(critic_input) + "\n", encoding="utf-8")

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:mapped", "run_id": "mapped", "status": "running"},
    )

    branch = projection["branches"][0]
    assert branch["status"] == "reviewing"
    assert branch["steps"][0]["product_smiles"] == "CCO"
    assert branch["steps"][0]["precursor_smiles"] == ["CC=O"]
    assert branch["steps"][0]["topology_status"] == "root"


def test_terminal_projection_marks_unreplayed_model_output_as_settled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        json.dumps(
            _event(
                event="model_output",
                artifact_type="RetrosynthesisProposalReport",
                task_id="director:test:branch:1:node:2",
                status="accepted_draft",
                output_artifact={
                    "payload": {
                        "candidates": [
                            {
                                "candidate_id": "unreplayed",
                                "product_smiles": "CCO",
                                "precursor_smiles": [],
                                "reaction_family": "draft disconnection",
                            }
                        ]
                    }
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:ended", "run_id": "ended", "status": "historical"},
    )

    assert projection["branches"][0]["pending_step"]["status"] == (
        "replay_record_unavailable"
    )


def test_live_projection_supports_independent_strategy_cards_and_cancellation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-io.jsonl"
    rows = []
    for branch in range(1, 4):
        rows.append(
            _event(
                event="model_output",
                artifact_type="StrategyCardReport",
                task_id=f"director:test:branch:{branch}:strategy:1",
                status="accepted_draft",
                usage={"output_tokens": 10},
                output_artifact={
                    "summary": f"strategy summary {branch}",
                    "payload": {
                        "target_smiles": "CCO",
                        "strategy_card": {
                            "strategy_signature": f"Independent strategy {branch}",
                            "key_forward_transformation": f"Transformation {branch}",
                        },
                    },
                },
            )
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    cancelling = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:cards", "run_id": "cards", "status": "cancelling"},
    )
    cancelled = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:cards", "run_id": "cards", "status": "cancelled"},
    )

    assert [row["signature"] for row in cancelling["strategies"]] == [
        "Independent strategy 1",
        "Independent strategy 2",
        "Independent strategy 3",
    ]
    assert cancelling["phase"] == "cancelling"
    assert cancelling["progress"] < 100
    assert {row["status"] for row in cancelling["branches"]} == {"cancelling"}
    assert cancelled["phase"] == "cancelled"
    assert cancelled["progress"] == 100
    assert {row["status"] for row in cancelled["branches"]} == {"cancelled"}


def test_live_page_and_molecule_renderer_are_available() -> None:
    class UnusedGateway:
        pass

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(UnusedGateway))
    client = app.test_client()
    page = client.get("/")
    legacy = client.get("/synthesis", query_string={"job": "solve:example"})

    assert page.status_code == 200
    assert page.headers["Cache-Control"] == "no-store"
    assert legacy.status_code == 302
    assert legacy.headers["Location"] == "/?job=solve:example"
    page_html = page.get_data(as_text=True)
    assert "Strategy-first synthesis" in page_html
    assert "EventSource" in page_html
    assert "fetch('/api/v4/jobs'" in page_html
    assert "run_scope:'interactive'" in page_html
    assert "value.status==='repository_hit'" in page_html
    assert 'id="repositoryHitPanel"' in page_html
    assert "目标已在仓库中" in page_html
    assert "3 秒自动刷新" in page_html
    assert "DOMParser" not in page_html
    assert "getBBox()" not in page_html
    assert "moleculeRenderVersion='rdkit-png-v6'" in page_html
    assert "/api/v4/molecule.png?smiles=" in page_html
    assert "document.createElement('img')" in page_html
    assert "object-fit:contain" in page_html
    assert "object-position:center" in page_html
    assert "position:absolute;inset:0" in page_html
    assert 'class="moleculeInline"' in page_html
    assert 'id="cancelButton"' in page_html
    assert "cancelCurrentJob" in page_html
    assert "'/cancel'" in page_html
    assert "该步骤没有可投影的前体" not in page_html
    assert "replay_record_unavailable" in page_html
    assert "反应类型未记录" in page_html

    for smiles in (
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(O)c1ccccc1O",
        "CC(=O)OC(C)=O",
    ):
        svg, valid = render_molecule_svg(smiles)
        assert valid is True
        assert svg.lstrip().startswith("<svg")
        assert not svg.lstrip().startswith("<?xml")
        assert "viewBox='-18 -18 356 236'" in svg
        assert "data-autoplanner-frame='safe-v3'" in svg
        label_origins = [
            (float(x), float(y))
            for x, y in re.findall(
                r"class='atom-\d+' d='M ([\d.]+) ([\d.]+)",
                svg,
            )
        ]
        assert label_origins
        assert all(
            24.0 <= x <= 296.0 and 24.0 <= y <= 170.0
            for x, y in label_origins
        )
        png, png_valid = render_molecule_png(smiles)
        assert png_valid is True
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        image = Image.open(BytesIO(png)).convert("RGB")
        ink_bounds = ImageChops.difference(
            image,
            Image.new("RGB", image.size, "white"),
        ).getbbox()
        assert ink_bounds is not None
        left, top, right, bottom = ink_bounds
        assert left >= 32 and top >= 32
        assert right <= image.width - 32 and bottom <= image.height - 32

    molecule = app.test_client().get("/api/v4/molecule.svg?smiles=CCO")
    assert molecule.status_code == 200
    assert molecule.content_type == "image/svg+xml"
    assert molecule.headers["Cache-Control"] == "no-store"
    assert molecule.data.lstrip().startswith(b"<svg")

    molecule_png = app.test_client().get(
        "/api/v4/molecule.png?smiles=CC(=O)Oc1ccccc1C(=O)O"
    )
    assert molecule_png.status_code == 200
    assert molecule_png.content_type == "image/png"
    assert molecule_png.headers["Cache-Control"] == "no-store"
    assert molecule_png.data.startswith(b"\x89PNG\r\n\x1a\n")

    fallback, valid = render_molecule_svg("not a smiles")
    assert valid is False
    assert "SMILES 暂无法解析" in fallback


def test_live_sse_route_emits_the_durable_projection(tmp_path: Path) -> None:
    director = tmp_path / ".autoplanner" / "director-workspace"
    director.mkdir(parents=True)
    (director / "model-io.jsonl").write_text(
        json.dumps(
            _event(
                event="model_output",
                artifact_type="StrategyPortfolioReport",
                task_id="director:test:strategy-portfolio:1",
                status="accepted_draft",
                output_artifact={
                    "payload": {
                        "target_smiles": "CCO",
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
        )
        + "\n",
        encoding="utf-8",
    )

    class HistoricalGateway:
        def list_runs(self, *, limit):
            return {
                "runs": [
                    {
                        "run_id": "historical-live",
                        "target_name": "historical target",
                    }
                ]
            }

        def status(self, run_id):
            assert run_id == "historical-live"
            return {
                "run_dir": str(tmp_path),
                "campaign_spec": {"target": {"canonical_smiles": "CCO"}},
                "status": {
                    "status": "running",
                    "stop_decision": {"terminal": False},
                },
            }

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(HistoricalGateway))
    response = app.test_client().get(
        "/api/v4/live/solve:historical-live/events",
        buffered=False,
    )
    first_chunk = next(iter(response.response)).decode("utf-8")
    response.close()

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert "event: snapshot" in first_chunk
    assert '"target_smiles":"CCO"' in first_chunk
    assert '"status":"interrupted"' in first_chunk
    assert '"strategies":[{' in first_chunk
