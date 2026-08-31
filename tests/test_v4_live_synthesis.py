from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

from flask import Flask
from PIL import Image, ImageChops

from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.runtime.paths import RuntimePaths
from cascade_planner.runtime.run_index import RUN_MANIFEST_SCHEMA, RunIndex
from cascade_planner.runtime.run_registry_catalog import (
    RunRegistryCatalog,
    binding_from_registry_root,
    registry_catalog_path,
)
from cascade_planner.web.v4_api import create_v4_blueprint
from cascade_planner.web.v4_live_synthesis import (
    _annotate_route_topology,
    _stabilize_replay_alternative_groups,
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
                    {
                        "strategy_signature": f"Strategy {index}",
                        "strategy_query": f"Query {index}",
                    }
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
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n" + json.dumps(context),
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
            "job_id": "solve:@main:live",
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
        job={"job_id": "solve:@main:pending", "run_id": "pending", "status": "running"},
    )

    pending = projection["branches"][1]["pending_step"]
    assert pending["reaction_family"] == "C-C disconnection"
    assert pending["status"] == "pending_host_replay"


def test_replay_merges_worker_ledger_and_host_proposal_audits(
    tmp_path: Path,
) -> None:
    director = tmp_path / ".autoplanner" / "director-workspace"
    director.mkdir(parents=True)
    model_io_path = director / "model-io.jsonl"
    model_io_path.write_text("", encoding="utf-8")
    target = "CCO"
    strategy_stdout = {
        "target_smiles": target,
        "strategy_cards": [
            {
                "strategy_query": f"Route direction {index}",
                "critical_assumption": f"Assumption {index}",
            }
            for index in range(1, 4)
        ],
    }
    candidate = {
        "candidate_id": "director:test:branch:1:node:1:candidate:1",
        "product_smiles": target,
        "precursor_smiles": [],
        "reaction_family": "carbonyl reduction",
        "conditions": ["NaBH4"],
    }

    def worker_record(task_id: str, artifact: dict, *, stdout: str = "") -> dict:
        return {
            "task_id": task_id,
            "record": {
                "task_id": task_id,
                "status": "accepted_draft",
                "metadata": {"model": "gpt-5.6-sol"},
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stdout": stdout,
                "output_artifact": artifact,
            },
        }

    worker_rows = [
        worker_record(
            "director:test:strategy-portfolio:1",
            {
                "artifact_type": "StrategyPortfolioReport",
                "summary": "paper_matched_strategy_generator",
                "payload": {
                    "target_smiles": target,
                    "strategy_cards": ["legacy serialized card"] * 3,
                },
            },
            stdout=json.dumps(strategy_stdout),
        ),
        worker_record(
            "director:test:branch:1:node:1",
            {
                "artifact_type": "RetrosynthesisProposalReport",
                "summary": "paper_matched_route_step",
                "payload": {"candidates": [candidate]},
            },
        ),
    ]
    (director / "sequential-director-worker-records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in worker_rows) + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".autoplanner" / "target-solver-checkpoint.json").write_text(
        json.dumps(
            {
                "director_outcomes": [
                    {
                        "proposal_audits": [
                            {
                                "accepted": True,
                                "proposal_id": "codex:branch:1:node:1:candidate:1",
                                "canonical_product_smiles": target,
                                "canonical_precursor_smiles": ["CC=O"],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=model_io_path,
        job={"job_id": "solve:@main:replay", "run_id": "replay", "status": "paused"},
        include_replay=True,
    )

    assert projection["model_output_count"] == 2
    assert [row["query"] for row in projection["strategies"]] == [
        "Route direction 1",
        "Route direction 2",
        "Route direction 3",
    ]
    frames = projection["replay"]["frames"]
    draft_index = next(
        index for index, frame in enumerate(frames) if frame["kind"] == "model_output"
    )
    host_index = next(
        index for index, frame in enumerate(frames) if frame["kind"] == "host_replay"
    )
    assert draft_index < host_index
    assert frames[draft_index]["branch_updates"][0]["pending_step"]["step_id"] == (
        candidate["candidate_id"]
    )
    host_branch = frames[host_index]["branch_updates"][0]
    assert host_branch["pending_step"] is None
    assert host_branch["steps"][0]["precursor_smiles"] == ["CC=O"]
    assert frames[host_index]["new_step_ids"] == [
        "codex:branch:1:node:1:candidate:1"
    ]

    settled_projection = project_live_synthesis(
        model_io_path=model_io_path,
        job={
            "job_id": "solve:@main:settled",
            "run_id": "settled",
            "status": "paused",
        },
    )
    settled_branch = settled_projection["branches"][0]
    assert settled_branch["steps"][0]["step_id"] == (
        "codex:branch:1:node:1:candidate:1"
    )
    assert settled_branch["steps"][0]["precursor_smiles"] == ["CC=O"]


def test_settled_projection_stitches_materialized_editor_step(
    tmp_path: Path,
) -> None:
    director = tmp_path / ".autoplanner" / "director-workspace"
    director.mkdir(parents=True)
    model_io_path = director / "model-io.jsonl"
    model_io_path.write_text("", encoding="utf-8")

    def worker_record(task_id: str, candidate: dict) -> dict:
        return {
            "task_id": task_id,
            "record": {
                "task_id": task_id,
                "status": "accepted_draft",
                "output_artifact": {
                    "artifact_type": "RetrosynthesisProposalReport",
                    "payload": {"candidates": [candidate]},
                },
            },
        }

    root = {
        "candidate_id": "codex:branch:1:node:1:candidate:1",
        "product_smiles": "CCCC",
        "precursor_smiles": ["CCC=O"],
        "reaction_family": "root transformation",
    }
    upstream = {
        "candidate_id": "codex:branch:1:node:3:candidate:1",
        "product_smiles": "CCO",
        "precursor_smiles": ["CC"],
        "reaction_family": "upstream transformation",
    }
    bridge = {
        "candidate_id": "director:test:branch:1:editor:1:candidate:1",
        "replace_span": {
            "remove_step_ids": [],
            "revised_steps": [
                {
                    "step_id": "codex:branch:1:node:2:bridge:1",
                    "product_smiles": "CCC=O",
                    "precursor_smiles": ["CCO"],
                    "reaction_family": "editor bridge",
                    "conditions": ["saved editor conditions"],
                }
            ],
        },
    }
    worker_rows = [
        worker_record("director:test:branch:1:node:1", root),
        worker_record("director:test:branch:1:node:3", upstream),
        worker_record("director:test:branch:1:editor:1", bridge),
    ]
    (director / "sequential-director-worker-records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in worker_rows) + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".autoplanner" / "target-solver-checkpoint.json").write_text(
        json.dumps(
            {
                "director_outcomes": [
                    {
                        "proposal_audits": [
                            {
                                "accepted": True,
                                "proposal_id": root["candidate_id"],
                                "canonical_product_smiles": "CCCC",
                                "canonical_precursor_smiles": ["CCC=O"],
                            },
                            {
                                "accepted": True,
                                "proposal_id": upstream["candidate_id"],
                                "canonical_product_smiles": "CCO",
                                "canonical_precursor_smiles": ["CC"],
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "target-only-solve-report.json").write_text(
        json.dumps(
            {
                "candidate_lifecycle": {
                    "records": [
                        {
                            "edge_id": "edge:bridge",
                            "materialization": {"materialized": True},
                            "portfolio": {"selected_route_ids": ["route:test"]},
                            "product_smiles": "CCC=O",
                            "precursor_smiles": ["CCO"],
                            "origin_records": [
                                {
                                    "origin_kind": "codex_global_director",
                                    "proposal_id": (
                                        "codex:branch:1:node:2:bridge:1"
                                    ),
                                    "transformation_hypothesis": "editor bridge",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=model_io_path,
        job={"job_id": "settled", "run_id": "settled", "status": "paused"},
    )
    branch = projection["branches"][0]
    assert [step["step_id"] for step in branch["steps"]] == [
        root["candidate_id"],
        "codex:branch:1:node:2:bridge:1",
        upstream["candidate_id"],
    ]
    assert [step["topology_status"] for step in branch["steps"]] == [
        "root",
        "linked",
        "linked",
    ]
    assert branch["steps"][1]["conditions"] == ["saved editor conditions"]
    assert branch["pending_step"] is None


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
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n" + json.dumps(context),
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (draft, replay)) + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:@main:paper", "run_id": "paper", "status": "running"},
    )

    branch = projection["branches"][1]
    assert branch["pending_step"] is None
    assert len(branch["steps"]) == 1
    assert branch["steps"][0]["step_id"] == "paper-step-1"
    assert branch["steps"][0]["product_smiles"] == target
    assert branch["steps"][0]["precursor_smiles"] == ["CCO"]
    assert branch["steps"][0]["reaction_family"] == "ester hydrolysis"
    assert branch["steps"][0]["topology_status"] == "root"


def test_compact_projection_recovers_steps_across_alternating_subpaths(
    tmp_path: Path,
) -> None:
    target = "CCCC"
    root_step = {
        "step_id": "codex:branch:2:node:1:candidate:1",
        "product_smiles": target,
        "precursor_smiles": ["CCC"],
        "reaction_family": "root disconnection",
    }

    def compact_input(call_index: int, step_ids: list[str], leaf: str) -> dict:
        return _event(
            event="model_input",
            artifact_type="RetrosynthesisProposalReport",
            task_id=f"director:test:branch:2:node:{call_index}",
            prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
            + json.dumps(
                {
                    "target_smiles": target,
                    "connected_path_reactions": [
                        {
                            "step_id": step_id,
                            "reaction_family": f"reaction {step_id}",
                        }
                        for step_id in step_ids
                    ],
                    "selected_leaf_mapped": leaf,
                }
            ),
        )

    odd_step = "codex:branch:2:node:3:candidate:1"
    even_step = "codex:branch:2:node:4:candidate:1"
    odd_tail = "codex:branch:2:node:5:candidate:1"
    initial = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:1",
        prompt="instructions\nCompactBranchContext:\n"
        + json.dumps(
            {
                "campaign_target": target,
                "accepted_path": [root_step],
            }
        ),
    )
    events = [
        initial,
        compact_input(3, [root_step["step_id"], odd_step], "CC"),
        compact_input(4, [root_step["step_id"], even_step], "CO"),
        compact_input(
            5,
            [root_step["step_id"], odd_step, odd_tail],
            "C",
        ),
    ]
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:@main:alternating", "status": "running"},
        include_replay=True,
    )

    branch = projection["branches"][1]
    assert [step["step_id"] for step in branch["steps"]] == [
        root_step["step_id"],
        odd_step,
        odd_tail,
    ]
    assert [step["precursor_smiles"] for step in branch["steps"]] == [
        ["CCC"],
        ["CC"],
        ["C"],
    ]
    last_branch_frame = [
        frame
        for frame in projection["replay"]["frames"]
        if frame["branch_index"] == 2
    ][-1]["branch_updates"][0]
    assert all(step["precursor_smiles"] for step in last_branch_frame["steps"])


def test_compact_projection_preserves_convergent_parent_and_child_identity(
    tmp_path: Path,
) -> None:
    target = "CCCC"
    coupled_product = "CCC"
    branch_a = "CC"
    branch_b = "CO"
    root = {
        "step_id": "codex:branch:2:node:1:candidate:1",
        "product_smiles": target,
        "precursor_smiles": [coupled_product],
        "reaction_family": "root disconnection",
    }
    coupling = {
        "step_id": "codex:branch:2:node:2:candidate:1",
        "product_smiles": coupled_product,
        "precursor_smiles": [branch_a, branch_b],
        "reaction_family": "convergent coupling",
    }
    child_a_id = "codex:branch:2:node:3:candidate:1"
    child_b_id = "codex:branch:2:node:4:candidate:1"
    initial = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:2",
        prompt="instructions\nCompactBranchContext:\n"
        + json.dumps(
            {
                "campaign_target": target,
                "accepted_path": [root, coupling],
            }
        ),
    )
    select_a = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:3",
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
        + json.dumps(
            {
                "target_smiles": target,
                "connected_path_reactions": [
                    {"step_id": root["step_id"]},
                    {"step_id": coupling["step_id"]},
                ],
                "selected_leaf_mapped": branch_a,
                "current_split_context": {
                    "parent_step_id": coupling["step_id"],
                    "co_precursors": [{"mapped_smiles": branch_b}],
                },
            }
        ),
    )
    extend_a = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:5",
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
        + json.dumps(
            {
                "target_smiles": target,
                "ancestor_smiles": [branch_a, coupled_product, target],
                "connected_path_reactions": [
                    {"step_id": root["step_id"]},
                    {"step_id": coupling["step_id"]},
                    {"step_id": child_a_id},
                ],
                "selected_leaf_mapped": "C",
            }
        ),
    )
    select_b = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:4",
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
        + json.dumps(
            {
                "target_smiles": target,
                "connected_path_reactions": [
                    {"step_id": root["step_id"]},
                    {"step_id": coupling["step_id"]},
                ],
                "selected_leaf_mapped": branch_b,
                "current_split_context": {
                    "parent_step_id": coupling["step_id"],
                    "co_precursors": [{"mapped_smiles": branch_a}],
                },
            }
        ),
    )
    extend_b = _event(
        event="model_input",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:6",
        prompt="instructions\nPaperMatchedRouteBuilderContext:\n"
        + json.dumps(
            {
                "target_smiles": target,
                "ancestor_smiles": [branch_b, coupled_product, target],
                "connected_path_reactions": [
                    {"step_id": root["step_id"]},
                    {"step_id": coupling["step_id"]},
                    {"step_id": child_b_id},
                ],
                "selected_leaf_mapped": "O",
            }
        ),
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (initial, select_a, extend_a, select_b, extend_b)
        )
        + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:@main:convergent", "status": "running"},
    )

    branch = projection["branches"][1]
    assert [step["step_id"] for step in branch["steps"]] == [
        root["step_id"],
        coupling["step_id"],
        child_a_id,
        child_b_id,
    ]
    assert branch["steps"][1]["precursor_smiles"] == [branch_a, branch_b]
    assert branch["steps"][2]["product_smiles"] == branch_a
    assert branch["steps"][3]["product_smiles"] == branch_b
    assert branch["steps"][2]["parent_step_index"] == 1
    assert branch["steps"][3]["parent_step_index"] == 1
    assert branch["steps"][1]["display_precursors"][0][
        "child_step_indices"
    ] == [2]
    assert branch["steps"][1]["display_precursors"][1][
        "child_step_indices"
    ] == [3]


def test_route_topology_projects_shared_precursor_as_stable_alternative_lanes() -> None:
    steps = [
        {
            "step_id": "root",
            "product_smiles": "CCCCC",
            "precursor_smiles": ["CCCC"],
            "route_provenance": "builder",
        },
        {
            "step_id": "builder-lane",
            "product_smiles": "CCCC",
            "precursor_smiles": ["CCC"],
            "route_provenance": "builder",
        },
        {
            "step_id": "builder-rejected-child",
            "product_smiles": "CCC",
            "precursor_smiles": ["CC"],
            "route_provenance": "builder",
            "critic_verdict": "reject",
        },
        {
            "step_id": "editor-lane",
            "product_smiles": "CCCC",
            "precursor_smiles": ["C=C"],
            "route_provenance": "editor_repair",
        },
        {
            "step_id": "editor-child",
            "product_smiles": "C=C",
            "precursor_smiles": ["C"],
            "route_provenance": "editor_repair",
        },
    ]

    _annotate_route_topology(steps)

    group = steps[0]["display_precursors"][0]
    assert group["child_step_indices"] == [1, 3]
    assert group["has_route_alternatives"] is True
    assert group["alternative_lane_order"] == ["builder-lane", "editor-lane"]
    assert group["alternative_lanes"] == [
        {
            "lane_index": 0,
            "child_step_index": 1,
            "child_step_id": "builder-lane",
            "provenance": "builder",
            "critic_state": "reject",
        },
        {
            "lane_index": 1,
            "child_step_index": 3,
            "child_step_id": "editor-lane",
            "provenance": "editor_repair",
            "critic_state": "pending",
        },
    ]
    assert steps[3]["parent_step_index"] == 0
    assert steps[2]["parent_step_index"] == 1
    assert steps[4]["parent_step_index"] == 3


def test_replay_reserves_final_alternative_order_without_copying_future_steps() -> None:
    final_steps = [
        {
            "step_id": "root",
            "product_smiles": "CCCCC",
            "precursor_smiles": ["CCCC"],
        },
        {
            "step_id": "builder-lane",
            "product_smiles": "CCCC",
            "precursor_smiles": ["CCC"],
        },
        {
            "step_id": "editor-lane",
            "product_smiles": "CCCC",
            "precursor_smiles": ["C=C"],
            "route_provenance": "editor_repair",
        },
    ]
    early_steps = json.loads(json.dumps(final_steps[:2]))
    _annotate_route_topology(final_steps)
    _annotate_route_topology(early_steps)
    frames = [
        {
            "branch_updates": [
                {"branch_index": 2, "steps": early_steps}
            ]
        }
    ]

    _stabilize_replay_alternative_groups(
        frames,
        branches={2: {"steps": final_steps}},
    )

    replay_steps = frames[0]["branch_updates"][0]["steps"]
    replay_group = replay_steps[0]["display_precursors"][0]
    assert [step["step_id"] for step in replay_steps] == ["root", "builder-lane"]
    assert replay_group["child_step_indices"] == [1]
    assert replay_group["has_route_alternatives"] is True
    assert replay_group["alternative_lane_count"] == 2
    assert replay_group["alternative_lane_order"] == [
        "builder-lane",
        "editor-lane",
    ]


def test_historical_running_kernel_is_projected_as_interrupted() -> None:
    projection = project_live_synthesis(
        model_io_path=None,
        job={
            "job_id": "solve:@main:orphaned",
            "run_id": "orphaned",
            "status": "historical",
            "campaign_status": "running",
            "campaign_terminal": False,
        },
    )

    assert projection["status"] == "interrupted"
    assert projection["phase"] == "interrupted"
    assert projection["progress"] < 100
    assert {branch["status"] for branch in projection["branches"]} == {"interrupted"}


def test_paused_projection_is_settled_without_claiming_completion() -> None:
    projection = project_live_synthesis(
        model_io_path=None,
        job={
            "job_id": "solve:@main:paused",
            "run_id": "paused",
            "status": "paused",
            "cancellation_available": False,
        },
    )

    assert projection["status"] == "paused"
    assert projection["phase"] == "paused"
    assert projection["progress"] < 100
    assert projection["cancellation_available"] is False
    assert {branch["status"] for branch in projection["branches"]} == {"paused"}


def test_completed_panel_projects_experiment_complete_and_keeps_campaign_pause() -> None:
    projection = project_live_synthesis(
        model_io_path=None,
        job={
            "job_id": "solve:@paper:complete",
            "run_id": "paper-complete",
            "status": "paused",
            "campaign_status": "paused",
            "experiment_status": "complete",
            "paper_equivalent_status": "solved",
            "campaign_resumable": True,
            "scientific_status": "unresolved",
            "cancellation_available": False,
        },
    )

    assert projection["status"] == "complete"
    assert projection["phase"] == "complete"
    assert projection["progress"] == 100
    assert projection["campaign_status"] == "paused"
    assert projection["experiment_status"] == "complete"
    assert projection["paper_equivalent_status"] == "solved"
    assert projection["campaign_resumable"] is True
    assert projection["scientific_status"] == "unresolved"
    assert projection["cancellation_available"] is False
    assert {branch["status"] for branch in projection["branches"]} == {"complete"}
    assert projection["status_axes"]["paper_equivalent"]["state"] == "solved"
    assert projection["status_axes"]["stock_closure"]["state"] == "pending"
    assert projection["status_axes"]["scientific_acceptance"]["state"] == (
        "unresolved"
    )


def test_settled_projection_keeps_five_outcome_axes_independent(
    tmp_path: Path,
) -> None:
    director = tmp_path / ".autoplanner" / "sequential-director"
    director.mkdir(parents=True)
    model_io_path = director / "model-io.jsonl"
    model_io_path.write_text("", encoding="utf-8")
    (tmp_path / "target-only-solve-report.json").write_text(
        json.dumps(
            {
                "director_outcomes": [
                    {
                        "plan": {
                            "multi_step_skeletons": [
                                {
                                    "routejson_replay_complete": True,
                                    "chemical_critic": {"status": "uncertain"},
                                },
                                {
                                    "routejson_replay_complete": True,
                                    "chemical_critic": {"status": "reject"},
                                },
                                {
                                    "routejson_replay_complete": True,
                                    "chemical_critic": {"status": "viable"},
                                },
                            ]
                        }
                    }
                ],
                "gates": {
                    "counts": {"canonical_stock_closed_routes": 1}
                },
                "paper_equivalent_solved": True,
                "claim": {"accepted_under_configured_policy": False},
            }
        ),
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=model_io_path,
        job={"job_id": "axes", "run_id": "axes", "status": "paused"},
    )
    axes = projection["status_axes"]

    assert axes["routejson_replay"] == {
        "state": "complete",
        "complete_routes": 3,
        "route_count": 3,
    }
    assert axes["stock_closure"] == {
        "state": "closed",
        "canonical_stock_closed_routes": 1,
    }
    assert axes["paper_equivalent"]["state"] == "solved"
    assert axes["chemical_critic"]["state"] == "mixed"
    assert axes["chemical_critic"]["counts"] == {
        "uncertain": 1,
        "reject": 1,
        "viable": 1,
    }
    assert axes["scientific_acceptance"]["state"] == "unresolved"
    assert axes["semantics"]["axes_are_independent"] is True


def test_historical_terminal_decision_overrides_stale_running_status() -> None:
    projection = project_live_synthesis(
        model_io_path=None,
        job={
            "job_id": "solve:@main:settled",
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
        job={"job_id": "solve:@main:stopped", "run_id": "stopped", "status": "running"},
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
        job={"job_id": "solve:@main:compact", "run_id": "compact", "status": "running"},
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
        job={"job_id": "solve:@main:compact", "run_id": "compact", "status": "running"},
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
    assert branch["steps"][0]["display_precursors"][0]["continues_to_next_step"] is True
    assert branch["steps"][0]["display_precursors"][0]["count"] == 2
    assert branch["steps"][0]["display_precursors"][0]["child_step_indices"] == [1]
    assert branch["steps"][0]["display_precursors"][1]["child_step_indices"] == [2]
    assert branch["steps"][2]["parent_step_index"] == 0


def test_editor_draft_keeps_host_topology_until_materialized_replay(
    tmp_path: Path,
) -> None:
    target = "CCCC"
    initial_steps = [
        {
            "step_id": "host-1",
            "product_smiles": target,
            "precursor_smiles": ["CCC=O"],
            "reaction_family": "root disconnection",
        },
        {
            "step_id": "host-2",
            "product_smiles": "CCC=O",
            "precursor_smiles": ["CCCO"],
            "reaction_family": "oxidation",
        },
        {
            "step_id": "host-3",
            "product_smiles": "CCCO",
            "precursor_smiles": ["CC=C"],
            "reaction_family": "hydration",
        },
    ]
    materialized_steps = [
        initial_steps[0],
        {
            "step_id": "edit-2",
            "product_smiles": "CCC=O",
            "precursor_smiles": ["CC(=O)Cl"],
            "reaction_family": "edited homologation",
        },
        {
            "step_id": "edit-3",
            "product_smiles": "CC(=O)Cl",
            "precursor_smiles": ["CC(=O)O"],
            "reaction_family": "acid chloride formation",
        },
        {
            "step_id": "edit-4",
            "product_smiles": "CC(=O)O",
            "precursor_smiles": ["CCO"],
            "reaction_family": "oxidation",
        },
    ]
    editor_candidate = {
        "candidate_id": "editor-patch",
        "repair_summary": "replace two weak steps with a connected three-step span",
        "replace_span": {
            "remove_step_ids": ["host-2", "host-3"],
            "revised_steps": [
                {
                    **step,
                    "precursor_smiles": [],
                    "reaction_operations": [{"operation": "host_materializes"}],
                }
                for step in materialized_steps[1:]
            ],
        },
    }
    events = [
        _event(
            event="model_input",
            artifact_type="RetrosynthesisProposalReport",
            task_id="director:test:branch:1:node:4",
            prompt="builder\nPaperMatchedRouteBuilderContext:\n"
            + json.dumps({"campaign_target": target, "route_json": initial_steps}),
        ),
        _event(
            event="model_output",
            artifact_type="RetrosynthesisProposalReport",
            task_id="director:test:branch:1:editor:8",
            status="accepted_draft",
            output_artifact={"payload": {"candidates": [editor_candidate]}},
        ),
        _event(
            event="model_input",
            artifact_type="RetrosynthesisProposalReport",
            task_id="director:test:branch:1:editor:9",
            prompt="editor\nPaperMatchedRouteEditorContext:\n"
            + json.dumps(
                {
                    "campaign_target": target,
                    "route_json": [
                        initial_steps[0],
                        *editor_candidate["replace_span"]["revised_steps"],
                    ],
                }
            ),
        ),
        _event(
            event="model_input",
            artifact_type="ChemicalStrategyCritique",
            task_id="critic:test:branch:1",
            prompt="critic\nPaperMatchedRouteCriticInput:\n"
            + json.dumps(
                {
                    "branch_id": 1,
                    "campaign_target": target,
                    "steps": materialized_steps,
                }
            ),
        ),
    ]
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "editor-authority", "run_id": "editor", "status": "running"},
        include_replay=True,
    )

    frames = projection["replay"]["frames"]
    editor_frames = [
        frame
        for frame in frames
        if frame["kind"] in {"editor_output", "editor_context"}
    ]
    assert len(editor_frames) == 2
    for frame in editor_frames:
        branch = frame["branch_updates"][0]
        assert [step["step_id"] for step in branch["steps"]] == [
            "host-1",
            "host-2",
            "host-3",
        ]
        assert all(
            step["topology_status"] in {"root", "linked"}
            for step in branch["steps"]
        )
        assert not frame["new_step_ids"]
        assert frame["topology_changed"] is False
    assert set(editor_frames[0]["proposed_step_ids"]) == {
        "host-2",
        "host-3",
        "edit-2",
        "edit-3",
        "edit-4",
    }
    assert editor_frames[0]["branch_updates"][0]["pending_step"] is None

    critic_index = next(
        index for index, frame in enumerate(frames) if frame["kind"] == "critic_start"
    )
    host_frame = frames[critic_index - 1]
    assert host_frame["kind"] == "host_replay"
    assert host_frame["topology_changed"] is True
    assert set(host_frame["topology_changed_step_ids"]) >= {
        "host-2",
        "host-3",
        "edit-2",
        "edit-3",
        "edit-4",
    }
    assert [
        step["step_id"] for step in host_frame["branch_updates"][0]["steps"]
    ] == ["host-1", "edit-2", "edit-3", "edit-4"]
    assert all(
        step["topology_status"] in {"root", "linked"}
        for step in frames[critic_index]["branch_updates"][0]["steps"]
    )
    assert projection["semantics"]["editor_drafts_never_replace_host_topology"]
    assert projection["replay"]["semantics"][
        "editor_host_materialization_is_explicit"
    ]


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
        job={"job_id": "solve:@main:review", "run_id": "review", "status": "running"},
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


def test_live_projection_attaches_builder_role_and_explicit_critic_reasons(
    tmp_path: Path,
) -> None:
    step_id = "codex:branch:2:node:2:candidate:1"
    builder_output = _event(
        event="model_output",
        artifact_type="RetrosynthesisProposalReport",
        task_id="director:test:branch:2:node:2",
        status="schema_accepted",
        output_artifact={
            "payload": {
                "candidates": [
                    {
                        "candidate_id": (
                            "director:test:branch:2:node:2:candidate:1"
                        ),
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": [],
                        "reaction_family": (
                            "Alcohol O-acylation forms the target ester."
                        ),
                        "checkpoint_relation": "executes_checkpoint",
                        "limitations": [
                            "Competing hydrolysis is possible."
                        ],
                    }
                ]
            }
        },
    )
    critic_input = _event(
        event="model_input",
        artifact_type="ChemicalStrategyCritique",
        task_id="critic:step-insight",
        prompt="critic instructions\nPaperMatchedRouteCriticInput:\n"
        + json.dumps(
            {
                "branch_id": 2,
                "campaign_target": "CCOC(C)=O",
                "steps": [
                    {
                        "step_id": step_id,
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "reaction_family": (
                            "Alcohol O-acylation forms the target ester."
                        ),
                    }
                ],
            }
        ),
    )
    critic_output = _event(
        event="model_output",
        artifact_type="ChemicalStrategyCritique",
        task_id="critic:step-insight",
        status="schema_accepted",
        output_artifact={
            "summary": "paper_matched_key_event_critic",
            "payload": {
                "step_assessments": [
                    {
                        "step_id": step_id,
                        "verdict": "uncertain",
                        "reasons": [
                            "The edit directly creates the required ester bond.",
                            "Water can consume the acid chloride.",
                        ],
                        "condition_assessment": "Dry conditions are required.",
                        "suggested_revision": "Use anhydrous solvent and base.",
                    }
                ]
            },
        },
    )
    path = tmp_path / "model-io.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (builder_output, critic_input, critic_output)
        )
        + "\n",
        encoding="utf-8",
    )

    projection = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "insight", "run_id": "insight", "status": "running"},
        include_replay=True,
    )

    step = projection["branches"][1]["steps"][0]
    assert step["checkpoint_relation"] == "executes_checkpoint"
    assert step["builder_limitations"] == [
        "Competing hydrolysis is possible."
    ]
    assert step["critic_verdict"] == "uncertain"
    assert step["critic_reasons"] == [
        "The edit directly creates the required ester bond.",
        "Water can consume the acid chloride.",
    ]
    assert step["critic_condition_assessment"] == (
        "Dry conditions are required."
    )
    assert step["critic_suggested_revision"] == (
        "Use anhydrous solvent and base."
    )
    critic_frame = next(
        frame
        for frame in projection["replay"]["frames"]
        if frame["kind"] == "critic_result"
    )
    replayed_step = critic_frame["branch_updates"][0]["steps"][0]
    assert replayed_step["critic_reasons"] == step["critic_reasons"]
    assert step_id in critic_frame["changed_step_ids"]


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
        job={"job_id": "solve:@main:mapped", "run_id": "mapped", "status": "running"},
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
        job={"job_id": "solve:@main:ended", "run_id": "ended", "status": "historical"},
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
        job={"job_id": "solve:@main:cards", "run_id": "cards", "status": "cancelling"},
    )
    cancelled = project_live_synthesis(
        model_io_path=path,
        job={"job_id": "solve:@main:cards", "run_id": "cards", "status": "cancelled"},
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
    retired = client.get("/synthesis")

    assert page.status_code == 200
    assert page.headers["Cache-Control"] == "no-store"
    assert retired.status_code == 404
    page_html = page.get_data(as_text=True)
    assert "LLM-directed retrosynthesis" in page_html
    assert "进入可审查的路线" in page_html
    assert "EventSource" in page_html
    assert "fetch('/api/v4/jobs'" in page_html
    assert "run_scope:'interactive'" in page_html
    assert "value.status==='repository_hit'" in page_html
    assert 'id="repositoryHitPanel"' in page_html
    assert "目标已在仓库中" in page_html
    assert "全部运行档案 →" in page_html
    assert 'href="/v4#runs"' in page_html
    assert "结果与审查 ↗" not in page_html
    assert "活跃任务由实时流更新" in page_html
    assert "localJobsRefreshMs=60000" in page_html
    assert "setInterval(()=>refreshLocalJobs" not in page_html
    assert "jobDisplayStatus" in page_html
    assert "Paper-equivalent solved" in page_html
    assert "严格证明可继续" in page_html
    assert "当前没有执行中的 Builder" in page_html
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
    assert 'id="exportMenu"' in page_html
    assert 'id="exportRoute"' in page_html
    assert 'id="exportInteraction"' in page_html
    assert 'id="exportGraph"' in page_html
    assert "'/exports/'" in page_html
    assert "'route.html?download=1" in page_html
    assert "'interaction.html?download=1" in page_html
    assert "'graph.html?download=1" in page_html
    assert 'id="replayButton"' in page_html
    assert 'id="replayTimeline"' in page_html
    assert 'id="replayPath"' in page_html
    assert 'id="replaySpeed"' in page_html
    assert "async function loadReplay" in page_html
    assert "'/replay'" in page_html
    assert "compileReplaySnapshots" in page_html
    assert "replayFramesForPath" in page_html
    assert "selectReplayPath" in page_html
    assert "function resetReplayView" in page_html
    assert "resetReplayView();if(appState.replay.index>=appState.replay.frames.length-1)" in page_html
    assert "frame.proposed_step_ids" in page_html
    assert "草稿涉及" in page_html
    assert "indexChanged&&frame.topology_changed" not in page_html
    assert "installRouteCanvas" in page_html
    assert "jumpReplayToStep" in page_html
    assert "data-replay-step" in page_html
    assert 'id="reactionDetail"' in page_html
    assert 'id="reactionDetailBody"' in page_html
    assert 'id="reactionReplayJump"' in page_html
    assert "showReactionDetail" in page_html
    assert 'id="activityToggle"' in page_html
    assert 'aria-controls="activityRail"' in page_html
    assert "activityCollapsed" in page_html
    assert "activityStorageKey='autoplanner.activity-rail'" in page_html
    assert "function toggleActivity" in page_html
    assert "Critic 依据" in page_html
    assert "结构与 SMILES" in page_html
    assert "reactionTechnical" in page_html
    assert "信息来源边界" not in page_html
    assert "页面不根据隐式思维过程补写原因" not in page_html
    assert ">Step ID<" not in page_html
    assert 'id="layoutToggle"' in page_html
    assert "setRouteOrientation" in page_html
    assert "horizontalRoute" in page_html
    assert "availableHeight" in page_html
    assert 'id="compactToggle"' in page_html
    assert "setCompactRoute" in page_html
    assert "compactModeStorageKey" in page_html
    assert "compactConnector" in page_html
    assert "if(appState.compactRoute)return" in page_html
    assert "routeAlternativeSet" in page_html
    assert "routeAlternativeLane" in page_html
    assert "routeAlternativeHeader" in page_html
    assert "routeAlternativeMarkup" in page_html
    assert "routeAlternativeFork" in page_html
    assert "syncAlternativeForkPaths" in page_html
    assert "routeAlternativeSet:before,.routeAlternativeSet:after,.routeAlternativeLane:before{display:none}" in page_html
    assert "data-route-lane-toggle" in page_html
    assert "Editor repair" in page_html
    assert "原始 Builder" in page_html
    assert "toggleRouteLaneFocus" in page_html
    assert "strategy.renderable_step_count" in page_html
    assert "routeMissingNotice" in page_html
    assert "个步骤等待结构恢复" in page_html
    assert "orientation:'horizontal'" in page_html
    assert "savedRouteOrientation='horizontal'" in page_html
    assert 'id="layoutToggle" class="toolButton layoutControl" type="button" aria-pressed="true"' in page_html
    assert "left:0;right:0;bottom:0" in page_html
    assert "overflow-y:auto;overflow-x:hidden" in page_html
    assert "word-break:break-all" in page_html
    assert "closest('button,a,input,select,textarea,[data-replay-step],[data-route-lane-toggle]')" in page_html
    assert "来源 / 状态" in page_html
    assert "产物 ·" in page_html
    assert "前体 ·" in page_html
    assert "步骤来源与状态" not in page_html
    assert ">产物 SMILES<" not in page_html
    assert ">前体 SMILES" not in page_html
    assert "pointerdown" in page_html
    assert "pointermove" in page_html
    assert "setPointerCapture" in page_html
    assert "translate3d" in page_html
    assert "passive:false" in page_html
    assert "event.ctrlKey" not in page_html
    assert "路径 1" in page_html
    assert "全部事件" in page_html
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
        assert all(24.0 <= x <= 296.0 and 24.0 <= y <= 170.0 for x, y in label_origins)
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

    class ExternalLiveGateway:
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
    app.register_blueprint(create_v4_blueprint(ExternalLiveGateway))
    client = app.test_client()
    response = client.get(
        "/api/v4/live/solve:@main:historical-live/events",
        buffered=False,
    )
    first_chunk = next(iter(response.response)).decode("utf-8")
    response.close()

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert "event: snapshot" in first_chunk
    assert '"target_smiles":"CCO"' in first_chunk
    assert '"status":"running"' in first_chunk
    assert '"cancellation_available":false' in first_chunk
    assert '"strategies":[{' in first_chunk

    showcase = client.get(
        "/api/v4/live/solve:@main:historical-live/showcase.html?download=1"
    )
    assert showcase.status_code == 200
    assert showcase.mimetype == "text/html"
    assert "attachment;" in showcase.headers["Content-Disposition"]
    assert "script-src 'unsafe-inline'" in showcase.headers["Content-Security-Policy"]
    assert 'id="showcaseData"' in showcase.get_data(as_text=True)
    assert 'id="play"' in showcase.get_data(as_text=True)

    static_route = client.get(
        "/api/v4/live/solve:@main:historical-live/exports/route.html"
        "?download=1&branch=1"
    )
    assert static_route.status_code == 200
    assert "static-route.html" in static_route.headers["Content-Disposition"]
    assert "静态逆合成路线" in static_route.get_data(as_text=True)
    assert 'id="play"' not in static_route.get_data(as_text=True)

    graph = client.get(
        "/api/v4/live/solve:@main:historical-live/exports/graph.html?download=1"
    )
    assert graph.status_code == 200
    assert "complete-route-graph.html" in graph.headers["Content-Disposition"]
    assert 'id="routeTree"' in graph.get_data(as_text=True)


def test_paused_live_sse_emits_one_snapshot_and_closes(tmp_path: Path) -> None:
    class PausedGateway:
        def list_runs(self, *, limit):
            return {
                "runs": [
                    {
                        "run_id": "paused-live",
                        "target_name": "paused target",
                        "status": "paused",
                        "updated_at": "2026-08-27T08:00:00Z",
                        "run_dir": str(tmp_path),
                    }
                ]
            }

        def status(self, run_id):
            assert run_id == "paused-live"
            return {
                "run_dir": str(tmp_path),
                "campaign_spec": {"target": {"canonical_smiles": "CCO"}},
                "status": {
                    "status": "paused",
                    "stop_decision": {"decision": "paused", "terminal": False},
                },
            }

    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(PausedGateway))
    response = app.test_client().get(
        "/api/v4/live/solve:@main:paused-live/events",
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '"status":"paused"' in body
    assert '"phase":"paused"' in body
    assert "event: complete" in body


def test_live_sse_resolves_external_registry_by_composite_job_id(
    tmp_path: Path,
) -> None:
    primary_paths = RuntimePaths.discover(
        repository_root=tmp_path,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "primary" / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "primary" / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "primary" / "artifacts"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(
                tmp_path / "primary" / "runtime" / "run_index.sqlite3"
            ),
        },
    )
    gateway = CampaignGateway(primary_paths)
    external_root = tmp_path / "paper25" / "case1"
    run_dir = external_root / "runs" / "target"
    director = run_dir / ".autoplanner" / "director-workspace"
    director.mkdir(parents=True)
    (director / "model-io.jsonl").write_text(
        json.dumps(
            _event(
                event="model_output",
                artifact_type="StrategyPortfolioReport",
                task_id="director:external:strategy-portfolio:1",
                status="accepted_draft",
                output_artifact={
                    "payload": {
                        "target_smiles": "CCO",
                        "strategy_cards": [
                            {
                                "strategy_signature": f"External strategy {index}",
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
    RunIndex(external_root / "runtime" / "run_index.sqlite3").upsert_run(
        {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": "paper-case1-run",
            "case_id": "case1",
            "target_name": "paper case 1",
            "status": "completed",
            "revision": 1,
            "updated_at": "2026-08-27T09:00:00Z",
            "run_dir": str(run_dir),
            "accepted": False,
            "cost_totals": {},
            "graph": {},
            "deficits": {},
        }
    )
    RunRegistryCatalog(registry_catalog_path(primary_paths)).register(
        binding_from_registry_root(
            external_root,
            registry_id="paper-case1",
            registry_label="Paper case 1",
            project_id="paper25",
            project_label="Paper 25-step panel",
            case_id="case1",
            repository_root=tmp_path,
        )
    )
    app = Flask(__name__)
    app.register_blueprint(create_v4_blueprint(lambda: gateway))

    response = app.test_client().get(
        "/api/v4/live/solve:@paper-case1:paper-case1-run/events",
        buffered=False,
    )
    first_chunk = next(iter(response.response)).decode("utf-8")
    response.close()

    assert response.status_code == 200
    assert '"job_id":"solve:@paper-case1:paper-case1-run"' in first_chunk
    assert '"strategies":[{' in first_chunk
    assert '"workbench_url":""' in first_chunk

    replay_response = app.test_client().get(
        "/api/v4/live/solve:@paper-case1:paper-case1-run/replay"
    )
    replay = replay_response.get_json()
    assert replay_response.status_code == 200
    assert replay_response.headers["Cache-Control"] == "no-store"
    assert replay["schema_version"] == "autoplanner.live_synthesis_replay.v1"
    assert replay["job_id"] == "solve:@paper-case1:paper-case1-run"
    assert replay["frame_count"] == 3
    assert [frame["kind"] for frame in replay["frames"]] == [
        "initial",
        "strategy",
        "final",
    ]
