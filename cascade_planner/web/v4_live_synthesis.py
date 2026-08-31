"""Live Strategy/Route Builder projection for the SyntheX-matched Web surface.

The sequential director already writes one append-only ``model-io.jsonl`` row
for every model input and output.  This adapter deliberately reads that durable
stream instead of inventing a second progress authority: StrategyPortfolio
outputs create the three cards, Route Builder outputs create pending steps, and
the following host-replayed input replaces each pending step with canonical
precursors.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from html import escape
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Callable, Mapping

from flask import Blueprint, Response, jsonify, request, stream_with_context

from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.web.v4_run_catalog import resolve_catalog_job
from cascade_planner.web.workspace_surface import static_html


GatewayFactory = Callable[[], Any]
_BRANCH_RE = re.compile(r":branch:(\d+):(?:strategy|node|editor):(\d+)")
_PROPOSAL_RE = re.compile(r"branch:(\d+):node:(\d+):candidate:(\d+)")
_ROUTE_CONTEXT_MARKERS = (
    "CompactBranchContext:",
    "PaperMatchedRouteBuilderContext:",
    "PaperMatchedRouteEditorContext:",
)
_CRITIC_CONTEXT_MARKERS = (
    "BlindRouteCriticInput:",
    "PaperMatchedRouteCriticInput:",
)
_STREAM_SETTLED_JOB_STATES = frozenset(
    {
        "complete",
        "unresolved",
        "failed",
        "cancelled",
        "historical",
        "interrupted",
        "paused",
    }
)
_FULL_PROGRESS_JOB_STATES = frozenset(
    {"complete", "unresolved", "failed", "cancelled", "historical"}
)


def register_live_synthesis_routes(
    blueprint: Blueprint,
    factory: GatewayFactory,
    *,
    jobs: dict[str, dict[str, Any]],
    jobs_lock: RLock,
) -> None:
    """Register the focused live UI, SSE stream, and molecule renderer."""

    @blueprint.get("/")
    def live_synthesis_homepage() -> Response:
        response = static_html("live_synthesis.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.get("/api/v4/live/<path:job_id>/events")
    def live_synthesis_events(job_id: str) -> Response:
        row = _job_row(factory, jobs, jobs_lock, job_id)
        if row is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404

        @stream_with_context
        def generate():
            previous_digest = ""
            last_keepalive = time.monotonic()
            terminal_observations = 0
            while True:
                current = _job_row(factory, jobs, jobs_lock, job_id) or row
                model_io_path = _model_io_path(factory, current)
                projection = project_live_synthesis(
                    model_io_path=model_io_path,
                    job=current,
                )
                encoded = json.dumps(
                    projection,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                if digest != previous_digest:
                    yield (
                        f"id: {projection['revision']}\n"
                        "event: snapshot\n"
                        f"data: {encoded}\n\n"
                    )
                    previous_digest = digest
                    last_keepalive = time.monotonic()
                elif time.monotonic() - last_keepalive >= 12.0:
                    yield ": keepalive\n\n"
                    last_keepalive = time.monotonic()

                if str(current.get("status") or "") in _STREAM_SETTLED_JOB_STATES:
                    terminal_observations += 1
                    if terminal_observations >= 2:
                        yield "event: complete\ndata: {}\n\n"
                        break
                else:
                    terminal_observations = 0
                time.sleep(0.7)

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @blueprint.get("/api/v4/live/<path:job_id>/replay")
    def live_synthesis_replay(job_id: str) -> Response:
        """Return the saved, event-ordered route history on demand."""

        row = _job_row(factory, jobs, jobs_lock, job_id)
        if row is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404
        projection = project_live_synthesis(
            model_io_path=_model_io_path(factory, row),
            job=row,
            include_replay=True,
        )
        replay = dict(projection.pop("replay", {}))
        replay.update(
            current_revision=projection["revision"],
            status=projection["status"],
            target_name=projection["target_name"],
            target_smiles=projection["target_smiles"],
        )
        response = jsonify(replay)
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.get("/api/v4/live/<path:job_id>/showcase.html")
    def live_synthesis_showcase(job_id: str) -> Response:
        """Export any catalog-visible job, including isolated smoke registries."""

        row = _job_row(factory, jobs, jobs_lock, job_id)
        if row is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404
        return _showcase_response(
            factory,
            row,
            download=str(request.args.get("download") or "") == "1",
            export_kind="interaction",
            branch_index=_showcase_branch_query(),
        )

    @blueprint.get("/api/v4/live/<path:job_id>/exports/<export_kind>.html")
    def live_synthesis_export(job_id: str, export_kind: str) -> Response:
        """Export a static route, full interaction replay, or complete graph."""

        row = _job_row(factory, jobs, jobs_lock, job_id)
        if row is None:
            return jsonify({"error": "job_not_found", "job_id": job_id}), 404
        return _showcase_response(
            factory,
            row,
            download=str(request.args.get("download") or "") == "1",
            export_kind=export_kind,
            branch_index=_showcase_branch_query(),
        )

    @blueprint.get("/api/v4/runs/<run_id>/showcase.html")
    def run_synthesis_showcase(run_id: str) -> Response:
        """Export a run from the authoritative main registry by run id."""

        status = dict(factory().status(run_id))
        status.update(
            run_id=run_id,
            job_id=str(status.get("job_id") or f"run:{run_id}"),
            registry_id=str(status.get("registry_id") or "main"),
        )
        return _showcase_response(
            factory,
            status,
            download=str(request.args.get("download") or "") == "1",
            export_kind="interaction",
            branch_index=_showcase_branch_query(),
        )

    @blueprint.get("/api/v4/runs/<run_id>/exports/<export_kind>.html")
    def run_synthesis_export(run_id: str, export_kind: str) -> Response:
        status = dict(factory().status(run_id))
        status.update(
            run_id=run_id,
            job_id=str(status.get("job_id") or f"run:{run_id}"),
            registry_id=str(status.get("registry_id") or "main"),
        )
        return _showcase_response(
            factory,
            status,
            download=str(request.args.get("download") or "") == "1",
            export_kind=export_kind,
            branch_index=_showcase_branch_query(),
        )

    @blueprint.get("/api/v4/molecule.svg")
    def molecule_svg() -> Response:
        smiles = str(request.args.get("smiles") or "").strip()
        if len(smiles) > 10_000:
            return jsonify(
                {"error": "invalid_smiles", "reason": "smiles_too_long"}
            ), 400
        svg, valid = render_molecule_svg(smiles)
        response = Response(svg.encode("utf-8"), content_type="image/svg+xml")
        # The page keeps a per-session molecule cache.  Do not let an older SVG
        # framing contract survive a renderer deployment in the HTTP cache.
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Autoplanner-Molecule-Valid"] = (
            "true" if valid else "false"
        )
        return response

    @blueprint.get("/api/v4/molecule.png")
    def molecule_png() -> Response:
        smiles = str(request.args.get("smiles") or "").strip()
        if len(smiles) > 10_000:
            return jsonify(
                {"error": "invalid_smiles", "reason": "smiles_too_long"}
            ), 400
        png, valid = render_molecule_png(smiles)
        response = Response(png, content_type="image/png")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Autoplanner-Molecule-Valid"] = (
            "true" if valid else "false"
        )
        return response


def project_live_synthesis(
    *,
    model_io_path: Path | None,
    job: Mapping[str, Any],
    include_replay: bool = False,
) -> dict[str, Any]:
    """Compile the append-only director model stream into a bounded UI view."""

    target_smiles = str(job.get("target_smiles") or "")
    strategies: list[dict[str, Any]] = []
    branches = {index: _empty_branch(index) for index in range(1, 4)}
    activities: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "model_invocations": 0}
    valid_line_count = 0
    parse_error_count = 0
    route_output_count = 0
    critic_output_count = 0
    active_review_branch: int | None = None
    critic_task_branches: dict[str, int] = {}
    step_insights: dict[str, dict[str, Any]] = {}
    remembered_route_steps: dict[int, dict[str, dict[str, Any]]] = {
        index: {} for index in range(1, 4)
    }
    convergent_branch_mode = {index: False for index in range(1, 4)}
    replay_frames: list[dict[str, Any]] = []
    replay_step_states: dict[int, dict[str, str]] = {
        index: {} for index in range(1, 4)
    }
    replay_topology_states: dict[int, dict[str, str]] = {
        index: {} for index in range(1, 4)
    }
    editor_replay_pending = {index: False for index in range(1, 4)}
    editor_proposed_step_ids: dict[int, list[str]] = {
        index: [] for index in range(1, 4)
    }
    # Settled runs must recover the same durable Host-replayed steps in the
    # normal live projection and in replay.  Limiting these indexes to
    # ``include_replay`` leaves branches whose final Builder/Editor input no
    # longer carries the whole path looking empty on the main canvas even
    # though their canonical proposal audits are intact.
    recover_saved_host_steps = (
        include_replay
        or _projection_job_status(job) in _STREAM_SETTLED_JOB_STATES
    )
    proposal_audits = (
        _proposal_audits(model_io_path) if recover_saved_host_steps else {}
    )
    saved_host_steps = (
        _saved_host_steps(model_io_path) if recover_saved_host_steps else {}
    )
    settled_canonical_steps = (
        _settled_canonical_steps(model_io_path)
        if recover_saved_host_steps
        else {}
    )
    editor_step_ids = set(_saved_editor_steps(model_io_path))
    editor_repair_step_keys = _editor_repair_step_keys(model_io_path)

    def remember_route_steps(
        branch_index: int,
        raw_steps: Any,
        *,
        authoritative: bool = True,
    ) -> None:
        """Retain canonical structures across alternating compact subpaths.

        Paper-matched Builder contexts deliberately compact older reactions to
        descriptors.  A split branch can alternate between two step-id sets,
        so the immediately preceding branch snapshot is not a sufficient
        source for restoring product/precursor structures.  Keep every
        structure observed earlier in this event stream and use it only when a
        later compact descriptor names that same step id.
        """

        remembered = remembered_route_steps[branch_index]
        for raw_step in raw_steps or []:
            if not isinstance(raw_step, Mapping):
                continue
            step = deepcopy(dict(raw_step))
            step_id = str(step.get("step_id") or "")
            if not step_id:
                continue
            prior = remembered.get(step_id, {})
            merged = {**prior, **step}
            for key in ("product_smiles", "precursor_smiles"):
                if prior.get(key) and (
                    not step.get(key) or not authoritative
                ):
                    merged[key] = deepcopy(prior[key])
            remembered[step_id] = merged

    def capture_replay(
        *,
        kind: str,
        title: str,
        detail: str = "",
        branch_index: int | None = None,
        timestamp: str = "",
        branch_indices: tuple[int, ...] | None = None,
        proposed_step_ids: tuple[str, ...] = (),
    ) -> None:
        if not include_replay:
            return
        update_indices = branch_indices or (
            (branch_index,) if branch_index in {1, 2, 3} else ()
        )
        branch_updates: list[dict[str, Any]] = []
        new_step_ids: list[str] = []
        changed_step_ids: list[str] = []
        removed_step_ids: list[str] = []
        topology_changed_step_ids: list[str] = []
        for update_index in update_indices:
            branch_snapshot = deepcopy(branches[update_index])
            _attach_step_insights(branch_snapshot, step_insights)
            _attach_route_provenance(
                branch_snapshot["steps"],
                editor_step_ids=editor_step_ids,
                editor_repair_step_keys=editor_repair_step_keys,
            )
            _annotate_route_topology(branch_snapshot["steps"])
            branch_updates.append(branch_snapshot)
            current_state = _replay_step_state(branch_snapshot)
            previous_state = replay_step_states[update_index]
            new_step_ids.extend(
                value for value in current_state if value not in previous_state
            )
            changed_step_ids.extend(
                value
                for value, fingerprint in current_state.items()
                if value in previous_state
                and previous_state[value] != fingerprint
            )
            removed_step_ids.extend(
                value for value in previous_state if value not in current_state
            )
            replay_step_states[update_index] = current_state
            current_topology = _replay_topology_state(branch_snapshot)
            previous_topology = replay_topology_states[update_index]
            topology_changed_step_ids.extend(
                value
                for value in dict.fromkeys(
                    (*previous_topology.keys(), *current_topology.keys())
                )
                if previous_topology.get(value) != current_topology.get(value)
            )
            replay_topology_states[update_index] = current_topology
        proposed_ids = list(dict.fromkeys(
            str(value) for value in proposed_step_ids if str(value).strip()
        ))
        proposed_set = set(proposed_ids)
        # Draft annotations are useful highlights, but they are not canonical
        # additions or adjustments.  Keep those labels mutually exclusive so
        # the UI can say “草稿涉及” without overstating a Host graph change.
        new_step_ids = [
            value for value in new_step_ids if value not in proposed_set
        ]
        changed_step_ids = [
            value for value in changed_step_ids if value not in proposed_set
        ]
        removed_step_ids = [
            value for value in removed_step_ids if value not in proposed_set
        ]
        focus_step_id = next(
            iter(
                new_step_ids
                or changed_step_ids
                or removed_step_ids
                or proposed_ids
            ),
            "",
        )
        strategy_snapshot = deepcopy(strategies[:3])
        for strategy_index, strategy in enumerate(strategy_snapshot, start=1):
            branch_state = branches[strategy_index]
            strategy.update(
                status=branch_state["status"],
                **_route_step_metrics(branch_state),
                model_calls=branch_state["model_calls"],
            )
        replay_frames.append(
            {
                "frame_index": len(replay_frames),
                "kind": kind,
                "branch_index": branch_index,
                "title": _clean_text(title),
                "detail": _clean_text(detail),
                "timestamp": str(timestamp or ""),
                "strategies": strategy_snapshot,
                "branch_updates": branch_updates,
                "new_step_ids": new_step_ids,
                "changed_step_ids": changed_step_ids,
                "removed_step_ids": removed_step_ids,
                "proposed_step_ids": proposed_ids,
                "topology_changed": bool(topology_changed_step_ids),
                "topology_changed_step_ids": list(
                    dict.fromkeys(topology_changed_step_ids)
                ),
                "focus_step_id": focus_step_id,
            }
        )

    capture_replay(
        kind="initial",
        title="开始读取已保存的合成轨迹",
        detail="路线尚未产生；后续帧均来自持久化的模型与 Host 事件。",
        branch_indices=(1, 2, 3),
    )

    for event in _read_director_events(model_io_path):
        if event is None:
            parse_error_count += 1
            continue
        valid_line_count += 1
        event_kind = str(event.get("event") or "")
        artifact_type = str(event.get("artifact_type") or "")
        task_id = str(event.get("task_id") or "")
        branch_index, call_index = _branch_and_call(task_id)

        if event_kind == "model_input":
            if artifact_type == "RetrosynthesisProposalReport" and branch_index:
                context = _route_builder_context(str(event.get("prompt") or ""))
                if context:
                    target_smiles = str(
                        context.get("campaign_target")
                        or context.get("target_smiles")
                        or target_smiles
                    )
                    branch = branches[branch_index]
                    previous_steps = deepcopy(branch["steps"])
                    previous_pending = deepcopy(branch["pending_step"])
                    is_editor = ":editor:" in task_id
                    path: list[dict[str, Any]] = []
                    incoming_path: list[dict[str, Any]] = []
                    editor_incomplete_ids: list[str] = []
                    editor_context_authoritative = True
                    if "route_json" in context:
                        incoming_path = _route_steps(context.get("route_json"))
                        path = incoming_path
                        if is_editor:
                            (
                                path,
                                editor_incomplete_ids,
                                editor_context_authoritative,
                            ) = _stable_editor_route_snapshot(
                                current_steps=branch["steps"],
                                incoming_steps=incoming_path,
                                remembered_steps=remembered_route_steps[
                                    branch_index
                                ].values(),
                            )
                        branch["steps"] = path
                    elif "accepted_path" in context:
                        incoming_path = _route_steps(context.get("accepted_path"))
                        path = incoming_path
                        if is_editor:
                            (
                                path,
                                editor_incomplete_ids,
                                editor_context_authoritative,
                            ) = _stable_editor_route_snapshot(
                                current_steps=branch["steps"],
                                incoming_steps=incoming_path,
                                remembered_steps=remembered_route_steps[
                                    branch_index
                                ].values(),
                            )
                        branch["steps"] = path
                    elif "connected_path_reactions" in context:
                        incoming_path = _paper_matched_route_steps(
                            context,
                            previous_steps=branch["steps"],
                            remembered_steps=remembered_route_steps[branch_index].values(),
                            target_smiles=target_smiles,
                        )
                        path = incoming_path
                        if is_editor:
                            (
                                path,
                                editor_incomplete_ids,
                                editor_context_authoritative,
                            ) = _stable_editor_route_snapshot(
                                current_steps=branch["steps"],
                                incoming_steps=incoming_path,
                                remembered_steps=remembered_route_steps[
                                    branch_index
                                ].values(),
                            )
                        if path:
                            split_context = context.get("current_split_context")
                            if (
                                isinstance(split_context, Mapping)
                                and split_context.get("co_precursors")
                            ) or any(
                                len(step.get("precursor_smiles") or []) > 1
                                for step in branch["steps"]
                            ):
                                convergent_branch_mode[branch_index] = True
                            if convergent_branch_mode[branch_index]:
                                _merge_compact_route_path(branch["steps"], path)
                            else:
                                branch["steps"] = path
                    remember_route_steps(
                        branch_index,
                        incoming_path or branch["steps"],
                        authoritative=(
                            "connected_path_reactions" not in context
                            and not (
                                is_editor
                                and not editor_context_authoritative
                            )
                        ),
                    )
                    if not (
                        is_editor
                        and not editor_context_authoritative
                        and editor_replay_pending[branch_index]
                    ):
                        branch["pending_step"] = None
                    branch["status"] = "building"
                    if (
                        is_editor
                        and editor_context_authoritative
                        and editor_replay_pending[branch_index]
                    ):
                        capture_replay(
                            kind="host_replay",
                            title="Editor Host replay 已物化局部替换",
                            detail=(
                                "Host-authoritative canonical RouteJSON 已恢复"
                                f" {len(editor_proposed_step_ids[branch_index])} "
                                "个草稿步骤的结构与父子连接。"
                            ),
                            branch_index=branch_index,
                            timestamp=str(event.get("timestamp") or ""),
                        )
                        editor_replay_pending[branch_index] = False
                        editor_proposed_step_ids[branch_index] = []
                    state_changed = (
                        previous_steps != branch["steps"]
                        or previous_pending is not None
                    )
                    if (
                        is_editor
                        or state_changed
                        or "connected_path_reactions" in context
                    ):
                        capture_replay(
                            kind=(
                                "editor_context" if is_editor else "host_replay"
                            ),
                            title=(
                                "Editor 读取局部编辑上下文"
                                if is_editor and not editor_context_authoritative
                                else "Editor 读取当前完整路线"
                                if is_editor
                                else "Host replay 接纳并更新路线"
                            ),
                            detail=(
                                "该 Editor 输入仍含未物化步骤；画布继续使用上一份 Host canonical 拓扑，只高亮局部草稿。"
                                if is_editor and not editor_context_authoritative
                                else "Editor 获得整条 canonical RouteJSON；它只需输出局部替换。"
                                if is_editor
                                else "下一次 Builder 输入携带了 Host 已物化的 canonical 路径。"
                            ),
                            branch_index=branch_index,
                            timestamp=str(event.get("timestamp") or ""),
                            proposed_step_ids=(
                                tuple(dict.fromkeys((
                                    *editor_proposed_step_ids[branch_index],
                                    *editor_incomplete_ids,
                                )))
                                if is_editor
                                else ()
                            ),
                        )
            elif artifact_type == "ChemicalStrategyCritique":
                context = _critic_route_context(str(event.get("prompt") or ""))
                critic_branch_index = _valid_branch_index(context.get("branch_id"))
                if context and critic_branch_index:
                    if (
                        active_review_branch is not None
                        and active_review_branch != critic_branch_index
                    ):
                        _complete_branch_review(branches[active_review_branch])
                    active_review_branch = critic_branch_index
                    critic_task_branches[task_id] = critic_branch_index
                    target_smiles = str(
                        context.get("campaign_target") or target_smiles
                    )
                    path = _route_steps(context.get("steps"))
                    branch = branches[critic_branch_index]
                    critic_incomplete_ids: list[str] = []
                    critic_context_authoritative = bool(path) and all(
                        step.get("precursor_smiles") for step in path
                    )
                    if (
                        editor_replay_pending[critic_branch_index]
                        and not critic_context_authoritative
                    ):
                        (
                            path,
                            critic_incomplete_ids,
                            critic_context_authoritative,
                        ) = _stable_editor_route_snapshot(
                            current_steps=branch["steps"],
                            incoming_steps=path,
                            remembered_steps=remembered_route_steps[
                                critic_branch_index
                            ].values(),
                        )
                    branch["steps"] = path
                    remember_route_steps(critic_branch_index, path)
                    branch["pending_step"] = None
                    if (
                        editor_replay_pending[critic_branch_index]
                        and critic_context_authoritative
                    ):
                        branch["status"] = "building"
                        capture_replay(
                            kind="host_replay",
                            title="Editor Host replay 已物化局部替换",
                            detail=(
                                "Host-authoritative canonical RouteJSON 已恢复"
                                f" {len(editor_proposed_step_ids[critic_branch_index])} "
                                "个草稿步骤的结构与父子连接。"
                            ),
                            branch_index=critic_branch_index,
                            timestamp=str(event.get("timestamp") or ""),
                        )
                        editor_replay_pending[critic_branch_index] = False
                        editor_proposed_step_ids[critic_branch_index] = []
                    branch["status"] = "reviewing"
                    capture_replay(
                        kind="critic_start",
                        title="Critic 开始审查 canonical 路线",
                        detail=(
                            "Critic 看到的是 Host replay 后的完整路线，而不是模型草稿。"
                            if critic_context_authoritative
                            else "Critic 输入仍缺少部分 Host 结构；画布保留上一份 canonical 拓扑。"
                        ),
                        branch_index=critic_branch_index,
                        timestamp=str(event.get("timestamp") or ""),
                        proposed_step_ids=tuple(critic_incomplete_ids),
                    )
            continue

        if event_kind != "model_output":
            continue

        artifact = dict(event.get("output_artifact") or {})
        payload = dict(artifact.get("payload") or {})
        status = str(event.get("status") or "")
        schema_accepted = status in {"accepted_draft", "schema_accepted"}
        event_usage = dict(event.get("usage") or {})
        usage["input_tokens"] += int(event_usage.get("input_tokens") or 0)
        usage["output_tokens"] += int(event_usage.get("output_tokens") or 0)
        usage["model_invocations"] += 1

        if artifact_type == "StrategyPortfolioReport":
            target_smiles = str(payload.get("target_smiles") or target_smiles)
            strategies = _strategy_cards(payload.get("strategy_cards"))
            strategy_review = "strategy-critic" in task_id
            activities.append(
                _activity(
                    event,
                    kind="strategy",
                    title=(
                        "Strategy Critic 已校正策略组合"
                        if strategy_review
                        else "三条正交策略已生成"
                    ),
                    detail=str(
                        artifact.get("summary")
                        or "策略组合已通过结构化输出。"
                    ),
                    branch_index=None,
                )
            )
            capture_replay(
                kind="strategy_review" if strategy_review else "strategy",
                title=(
                    "Strategy Critic 已校正策略组合"
                    if strategy_review
                    else "Strategy Generator 提出三条路线方向"
                ),
                detail=str(
                    artifact.get("summary")
                    or "三条策略假设进入独立 Builder 分支。"
                ),
                timestamp=str(event.get("timestamp") or ""),
            )
            continue

        if artifact_type == "StrategyCardReport" and branch_index:
            target_smiles = str(payload.get("target_smiles") or target_smiles)
            card = dict(payload.get("strategy_card") or {})
            while len(strategies) < branch_index:
                strategies.append(_empty_strategy(len(strategies) + 1))
            strategies[branch_index - 1] = {
                "strategy_id": f"strategy-{branch_index}",
                "index": branch_index,
                "signature": _clean_text(
                    card.get("strategy_signature")
                    or card.get("key_forward_transformation")
                    or f"Strategy {branch_index}"
                ),
                "query": _clean_text(
                    card.get("key_forward_transformation")
                    or card.get("strategy_query")
                    or payload.get("selection_rationale")
                ),
                "status": "planning",
                "step_count": 0,
                "renderable_step_count": 0,
                "unresolved_step_count": 0,
                "model_calls": 0,
            }
            activities.append(
                _activity(
                    event,
                    kind="strategy",
                    title=f"Strategy {branch_index} 已生成",
                    detail=str(
                        artifact.get("summary")
                        or card.get("strategy_basis")
                        or "独立策略卡已完成结构化输出。"
                    ),
                    branch_index=branch_index,
                )
            )
            capture_replay(
                kind="strategy",
                title=f"Strategy {branch_index} 已生成",
                detail=str(
                    artifact.get("summary")
                    or card.get("strategy_basis")
                    or "独立策略卡完成。"
                ),
                branch_index=branch_index,
                timestamp=str(event.get("timestamp") or ""),
            )
            continue

        if artifact_type == "RetrosynthesisProposalReport" and branch_index:
            route_output_count += 1
            branch = branches[branch_index]
            branch["model_calls"] += 1
            is_editor = ":editor:" in task_id
            candidates = [
                dict(value)
                for value in payload.get("candidates") or []
                if isinstance(value, Mapping)
            ]
            for candidate in candidates:
                _record_builder_step_insights(step_insights, candidate)
            if candidates:
                candidate = candidates[0]
                editor_patch_ids = (
                    _editor_patch_step_ids(candidate) if is_editor else []
                )
                if is_editor and editor_patch_ids:
                    # ``replace_span`` is a local edit program.  Its revised
                    # rows can deliberately omit precursor structures until
                    # the Host materializes the graph operations, so showing
                    # one as a pending reaction would create a second topology
                    # authority and a false MODULE PRODUCT subtree.
                    branch["pending_step"] = None
                    editor_replay_pending[branch_index] = True
                    editor_proposed_step_ids[branch_index] = list(
                        dict.fromkeys((
                            *editor_proposed_step_ids[branch_index],
                            *editor_patch_ids,
                        ))
                    )
                else:
                    branch["pending_step"] = _pending_step(candidate, call_index)
                    if is_editor:
                        editor_replay_pending[branch_index] = True
                        editor_proposed_step_ids[branch_index] = list(
                            dict.fromkeys((
                                *editor_proposed_step_ids[branch_index],
                                str(branch["pending_step"].get("step_id") or ""),
                            ))
                        )
                branch["status"] = (
                    "replaying" if schema_accepted else "reviewing"
                )
                replace_span = candidate.get("replace_span")
                revised_steps = (
                    [
                        dict(value)
                        for value in replace_span.get("revised_steps") or []
                        if isinstance(value, Mapping)
                    ]
                    if isinstance(replace_span, Mapping)
                    else []
                )
                family_source = revised_steps[0] if revised_steps else candidate
                family = _clean_text(
                    family_source.get("reaction_family")
                    or family_source.get("transformation_hypothesis")
                )
                detail = str(
                    artifact.get("summary")
                    or candidate.get("repair_summary")
                    or family_source.get("transformation_rationale")
                    or candidate.get("transformation_rationale")
                    or family
                    or "Route Builder 输出了新的逆合成步骤。"
                )
                title = family or f"Route Builder 第 {call_index or branch['model_calls']} 次输出"
            else:
                branch["pending_step"] = None
                branch["status"] = "reviewing"
                detail = str(
                    artifact.get("summary")
                    or "当前分支结束扩展。"
                )
                title = "分支暂停"
            activities.append(
                _activity(
                    event,
                    kind="route",
                    title=title,
                    detail=detail,
                    branch_index=branch_index,
                )
            )
            capture_replay(
                kind="editor_output" if is_editor else "model_output",
                title=(
                    "Editor 提出局部路线替换"
                    if is_editor
                    else "Builder 提出新的逆合成节点"
                ),
                detail=detail,
                branch_index=branch_index,
                timestamp=str(event.get("timestamp") or ""),
                proposed_step_ids=(
                    tuple(editor_patch_ids)
                    if is_editor and candidates
                    else ()
                ),
            )
            if candidates and not is_editor:
                proposal_key = _proposal_key(task_id, candidate)
                audit = proposal_audits.get(proposal_key)
                host_step = None
                if audit and audit.get("accepted") is True:
                    host_step = _host_step_from_audit(candidate, audit)
                elif proposal_key in saved_host_steps:
                    host_step = deepcopy(saved_host_steps[proposal_key])
                if host_step is not None:
                    _upsert_route_step(branch["steps"], host_step)
                    remember_route_steps(branch_index, (host_step,))
                    branch["pending_step"] = None
                    branch["status"] = "building"
                    capture_replay(
                        kind="host_replay",
                        title="Host replay 接纳并物化节点",
                        detail=(
                            f"保存记录确认图编辑生成 {len(host_step['precursor_smiles'])} 个 canonical 前体。"
                        ),
                        branch_index=branch_index,
                        timestamp=str(event.get("timestamp") or ""),
                    )
            continue

        if artifact_type == "ChemicalStrategyCritique":
            critic_output_count += 1
            critic_branch_index = (
                critic_task_branches.get(task_id) or branch_index
            )
            _record_critic_step_insights(step_insights, payload)
            if critic_branch_index in {1, 2, 3}:
                branches[critic_branch_index]["chemical_critic_status"] = str(
                    payload.get("overall_assessment")
                    or payload.get("status")
                    or "unavailable"
                )
                _attach_step_insights(
                    branches[critic_branch_index], step_insights
                )
            activities.append(
                _activity(
                    event,
                    kind="critic",
                    title="独立化学 Critic 已返回",
                    detail=str(
                        artifact.get("summary")
                        or payload.get("route_level_risks")
                        or "正在核对机理、选择性与步骤连续性。"
                    ),
                    branch_index=critic_branch_index,
                )
            )
            capture_replay(
                kind="critic_result",
                title="Critic 返回路线审查结论",
                detail=str(
                    artifact.get("summary")
                    or payload.get("route_level_risks")
                    or "机理、选择性与步骤连续性审查完成。"
                ),
                branch_index=critic_branch_index,
                timestamp=str(event.get("timestamp") or ""),
            )

    for branch_index, canonical_steps in settled_canonical_steps.items():
        branch = branches[branch_index]
        canonical_step_ids = {
            str(step.get("step_id") or "") for step in canonical_steps
        }
        for canonical_step in canonical_steps:
            _merge_canonical_route_step(branch["steps"], canonical_step)
        remember_route_steps(branch_index, branch["steps"])
        pending = branch.get("pending_step")
        if (
            isinstance(pending, Mapping)
            and str(pending.get("step_id") or "") in canonical_step_ids
        ):
            branch["pending_step"] = None
        _order_route_steps(branch["steps"])
        materialized_editor_replay = (
            editor_replay_pending[branch_index]
            and bool(canonical_steps)
            and all(
                str(step.get("product_smiles") or "").strip()
                and bool(step.get("precursor_smiles"))
                for step in canonical_steps
            )
            and bool(
                canonical_step_ids
                & set(editor_proposed_step_ids[branch_index])
            )
        )
        if materialized_editor_replay:
            branch["pending_step"] = None
            branch["status"] = "building"
            capture_replay(
                kind="host_replay",
                title="Editor Host replay 已物化最终局部替换",
                detail=(
                    "最终 candidate lifecycle 确认 Editor 草稿已经成为 "
                    "Host-authoritative canonical 路线。"
                ),
                branch_index=branch_index,
            )
            editor_replay_pending[branch_index] = False
            editor_proposed_step_ids[branch_index] = []

    for branch in branches.values():
        _attach_step_insights(branch, step_insights)
        _attach_route_provenance(
            branch["steps"],
            editor_step_ids=editor_step_ids,
            editor_repair_step_keys=editor_repair_step_keys,
        )
        _annotate_route_topology(branch["steps"])

    for index, strategy in enumerate(strategies[:3], start=1):
        branch = branches[index]
        strategy.update(
            status=branch["status"],
            **_route_step_metrics(branch),
            model_calls=branch["model_calls"],
        )

    job_status = _projection_job_status(job)
    if job_status in {"complete", "unresolved", "historical"}:
        for branch in branches.values():
            if branch["pending_step"]:
                branch["pending_step"]["status"] = "replay_record_unavailable"
            if (
                job_status == "complete" or branch["steps"]
            ) and branch["status"] not in {"failed", "rejected"}:
                branch["status"] = "complete"
        for index, strategy in enumerate(strategies[:3], start=1):
            strategy["status"] = branches[index]["status"]
    elif job_status == "interrupted":
        for branch in branches.values():
            if branch["pending_step"]:
                branch["pending_step"]["status"] = "replay_record_unavailable"
            if branch["status"] not in {"built", "failed", "rejected"}:
                branch["status"] = "interrupted"
        for index, strategy in enumerate(strategies[:3], start=1):
            strategy["status"] = branches[index]["status"]
    elif job_status == "paused":
        for branch in branches.values():
            if branch["pending_step"]:
                branch["pending_step"]["status"] = "replay_record_unavailable"
            if branch["status"] not in {"built", "complete", "failed", "rejected"}:
                branch["status"] = "paused"
        for index, strategy in enumerate(strategies[:3], start=1):
            strategy["status"] = branches[index]["status"]
    elif job_status in {"cancelling", "cancelled"}:
        for branch in branches.values():
            branch["pending_step"] = None
            branch["status"] = job_status
        for strategy in strategies[:3]:
            strategy["status"] = job_status

    capture_replay(
        kind="final",
        title=(
            "任务已暂停，显示最后保存状态"
            if job_status == "paused"
            else "重放抵达当前保存状态"
        ),
        detail="该帧只收口展示状态，不会启动模型或重新执行图编辑。",
        branch_indices=(1, 2, 3),
    )
    if include_replay:
        _stabilize_replay_alternative_groups(
            replay_frames,
            branches=branches,
        )

    phase = _phase(
        job_status=job_status,
        strategy_count=len(strategies),
        route_output_count=route_output_count,
        critic_output_count=critic_output_count,
    )
    progress = _progress(
        job_status=job_status,
        strategy_count=len(strategies),
        route_output_count=route_output_count,
        critic_output_count=critic_output_count,
    )
    revision_seed = (
        f"{valid_line_count}:{parse_error_count}:{job_status}:"
        f"{len(activities)}:{sum(len(v['steps']) for v in branches.values())}:"
        + ":".join(
            f"{index}:{branches[index]['status']}:"
            f"{(branches[index]['pending_step'] or {}).get('step_id', '')}"
            for index in range(1, 4)
        )
    )
    revision = hashlib.sha256(revision_seed.encode("utf-8")).hexdigest()[:16]
    projection = {
        "schema_version": "autoplanner.live_synthesis_projection.v1",
        "revision": revision,
        "job_id": str(job.get("job_id") or ""),
        "run_id": str(job.get("run_id") or ""),
        "target_name": str(job.get("target_name") or "blind target"),
        "target_smiles": target_smiles,
        "status": job_status,
        "campaign_status": str(
            job.get("campaign_status") or job.get("status") or ""
        ),
        "experiment_status": str(job.get("experiment_status") or ""),
        "paper_equivalent_status": str(
            job.get("paper_equivalent_status") or ""
        ),
        "campaign_resumable": job.get("campaign_resumable") is True,
        "scientific_status": str(job.get("scientific_status") or ""),
        "status_axes": _status_axes(
            model_io_path,
            job=job,
            branches=branches,
        ),
        "phase": phase,
        "progress": progress,
        "cancellation_available": job.get("cancellation_available") is not False,
        "execution_source": str(job.get("execution_source") or "web"),
        "activity_observed_at": str(job.get("activity_observed_at") or ""),
        "activity_stale": job.get("activity_stale") is True,
        "strategies": strategies[:3],
        "branches": [branches[index] for index in range(1, 4)],
        "activities": activities[-80:],
        "usage": usage,
        "model_output_count": len(activities),
        "parse_error_count": parse_error_count,
        "workbench_url": (
            f"/api/v4/runs/{str(job.get('run_id') or '')}/workbench.html"
            if job.get("run_id")
            and str(job.get("registry_id") or "main") == "main"
            else ""
        ),
        "semantics": {
            "one_render_revision_per_durable_model_output": True,
            "pending_steps_wait_for_host_replay": True,
            "strategy_cards_are_hypotheses_not_reaction_proof": True,
            "editor_drafts_never_replace_host_topology": True,
            "editor_host_materialization_is_explicit": True,
        },
    }
    if include_replay:
        projection["replay"] = {
            "schema_version": "autoplanner.live_synthesis_replay.v1",
            "job_id": str(job.get("job_id") or ""),
            "frame_count": len(replay_frames),
            "frames": replay_frames,
            "semantics": {
                "source_is_saved_events": True,
                "model_output_is_not_host_acceptance": True,
                "host_replay_frames_are_canonical": True,
                "editor_drafts_never_replace_host_topology": True,
                "editor_host_materialization_is_explicit": True,
                "replay_is_read_only": True,
            },
        }
    return projection


def _job_row(
    factory: GatewayFactory,
    jobs: dict[str, dict[str, Any]],
    jobs_lock: RLock,
    job_id: str,
) -> dict[str, Any] | None:
    with jobs_lock:
        row = dict(jobs.get(job_id) or {})
        active_rows = [dict(value) for value in jobs.values()]
    if row:
        return row
    try:
        gateway = factory()
    except Exception:
        return None
    resolved, _owning_gateway = resolve_catalog_job(
        gateway,
        job_id,
        active_rows=active_rows,
    )
    return resolved


def _model_io_path(factory: GatewayFactory, job: Mapping[str, Any]) -> Path | None:
    run_id = str(job.get("run_id") or "")
    if not run_id:
        return None
    raw_run_dir = str(job.get("run_dir") or "").strip()
    if not raw_run_dir:
        try:
            status = factory().status(run_id)
        except Exception:
            return None
        raw_run_dir = str(status.get("run_dir") or "").strip()
    if not raw_run_dir:
        return None
    run_dir = Path(raw_run_dir)
    return run_dir / ".autoplanner" / "director-workspace" / "model-io.jsonl"


def _showcase_response(
    factory: GatewayFactory,
    job: Mapping[str, Any],
    *,
    download: bool,
    export_kind: str,
    branch_index: int,
) -> Response:
    """Build a bounded single-file response from the same live projection."""

    from cascade_planner.web.v4_showcase_export import (
        build_run_export_html,
        showcase_filename,
    )

    model_io_path = _model_io_path(factory, job)
    if model_io_path is None:
        return jsonify(
            {
                "error": "showcase_unavailable",
                "reason": "run_directory_unavailable",
                "run_id": str(job.get("run_id") or ""),
            }
        ), 422
    try:
        body = build_run_export_html(
            run_dir=model_io_path.parents[2],
            job=job,
            model_io_path=model_io_path,
            export_kind=export_kind,
            branch_index=branch_index,
        )
    except ValueError as exc:
        return jsonify(
            {
                "error": "showcase_unavailable",
                "reason": str(exc),
                "run_id": str(job.get("run_id") or ""),
            }
        ), 422
    response = Response(body, mimetype="text/html")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src data:; font-src 'none'; connect-src 'none'; form-action 'none'; "
        "base-uri 'none'"
    )
    if download:
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{showcase_filename(str(job.get("run_id") or ""), export_kind=export_kind, branch_index=branch_index)}"'
        )
    return response


def _showcase_branch_query() -> int:
    try:
        value = int(request.args.get("branch", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("showcase_branch_index_invalid") from exc
    if value not in {1, 2, 3}:
        raise ValueError(f"showcase_branch_index_invalid:{value}")
    return value


def _read_model_io(path: Path | None):
    if path is None or not path.is_file():
        return ()
    rows: list[dict[str, Any] | None] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    rows.append(None)
                    continue
                rows.append(value if isinstance(value, dict) else None)
    except OSError:
        return ()
    return rows


def _read_director_events(path: Path | None):
    """Merge the complete worker ledger with richer saved input/output rows.

    Older runs wrote Builder/Strategy calls only to the append-only worker
    ledger, while Critic/Editor calls also appeared in ``model-io.jsonl``.
    The worker ledger supplies the complete call order; matching model-io rows
    replace duplicates and add the Host contexts needed to explain repairs.
    """

    raw_rows = list(_read_model_io(path))
    if path is None:
        return raw_rows
    worker_path = path.with_name("sequential-director-worker-records.jsonl")
    worker_events = list(_read_worker_events(worker_path))
    if not worker_events:
        return raw_rows

    raw_by_task: dict[str, list[dict[str, Any]]] = {}
    unbound_rows: list[dict[str, Any] | None] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            unbound_rows.append(None)
            continue
        task_id = str(row.get("task_id") or "")
        if not task_id:
            unbound_rows.append(dict(row))
            continue
        raw_by_task.setdefault(task_id, []).append(dict(row))

    merged: list[dict[str, Any] | None] = []
    worker_task_ids: set[str] = set()
    for worker_event in worker_events:
        if worker_event is None:
            merged.append(None)
            continue
        task_id = str(worker_event.get("task_id") or "")
        worker_task_ids.add(task_id)
        matching = raw_by_task.get(task_id, [])
        merged.extend(
            value for value in matching if value.get("event") == "model_input"
        )
        saved_output = next(
            (
                value
                for value in reversed(matching)
                if value.get("event") == "model_output"
            ),
            None,
        )
        merged.append(saved_output or worker_event)

    for task_id, rows in raw_by_task.items():
        if task_id not in worker_task_ids:
            merged.extend(rows)
    merged.extend(unbound_rows)
    return merged


def _read_worker_events(path: Path):
    if not path.is_file():
        return ()
    rows: list[dict[str, Any] | None] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    wrapper = json.loads(line)
                except json.JSONDecodeError:
                    rows.append(None)
                    continue
                if not isinstance(wrapper, Mapping):
                    rows.append(None)
                    continue
                record = wrapper.get("record")
                if not isinstance(record, Mapping):
                    rows.append(None)
                    continue
                artifact = _worker_output_artifact(record)
                rows.append(
                    {
                        "schema_version": "sequential_director_worker_projection.v1",
                        "event": "model_output",
                        "timestamp": "",
                        "model": str(
                            dict(record.get("metadata") or {}).get("model") or ""
                        ),
                        "artifact_type": str(artifact.get("artifact_type") or ""),
                        "task_id": str(
                            wrapper.get("task_id") or record.get("task_id") or ""
                        ),
                        "status": str(record.get("status") or ""),
                        "usage": dict(record.get("usage") or {}),
                        "output_artifact": artifact,
                    }
                )
    except OSError:
        return ()
    return rows


def _worker_output_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
    artifact = deepcopy(dict(record.get("output_artifact") or {}))
    payload = artifact.get("payload")
    cards = payload.get("strategy_cards") if isinstance(payload, Mapping) else None
    if cards and all(not isinstance(value, Mapping) for value in cards):
        try:
            portable = json.loads(str(record.get("stdout") or ""))
        except json.JSONDecodeError:
            portable = {}
        if isinstance(portable, Mapping):
            artifact["payload"] = {**dict(payload), **dict(portable)}
    return artifact


def _proposal_audits(path: Path | None) -> dict[tuple[int, int, int], dict[str, Any]]:
    if path is None:
        return {}
    checkpoint_path = path.parent.parent / "target-solver-checkpoint.json"
    if not checkpoint_path.is_file():
        return {}
    try:
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for outcome in value.get("director_outcomes") or []:
        if not isinstance(outcome, Mapping):
            continue
        for audit in outcome.get("proposal_audits") or []:
            if not isinstance(audit, Mapping):
                continue
            match = _PROPOSAL_RE.search(str(audit.get("proposal_id") or ""))
            if match:
                rows[tuple(int(item) for item in match.groups())] = dict(audit)
    return rows


def _saved_host_steps(path: Path | None) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Index canonical steps later persisted in Builder/Editor/Critic inputs."""

    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for event in _read_model_io(path):
        if not isinstance(event, Mapping) or event.get("event") != "model_input":
            continue
        artifact_type = str(event.get("artifact_type") or "")
        prompt = str(event.get("prompt") or "")
        raw_steps: Any = []
        if artifact_type == "ChemicalStrategyCritique":
            context = _critic_route_context(prompt)
            raw_steps = context.get("steps")
        elif artifact_type == "RetrosynthesisProposalReport":
            context = _route_builder_context(prompt)
            raw_steps = (
                context.get("route_json")
                if "route_json" in context
                else context.get("accepted_path")
            )
        else:
            continue
        for step in _route_steps(raw_steps):
            match = _PROPOSAL_RE.search(str(step.get("step_id") or ""))
            if not match:
                continue
            key = tuple(int(item) for item in match.groups())
            # The first canonical observation is the state immediately after
            # the original Builder call.  Later Editor contexts may revise it.
            rows.setdefault(key, step)
    return rows


def _solve_report(path: Path | None) -> dict[str, Any]:
    """Read the canonical settled report without turning it into UI authority."""

    if path is None or len(path.parents) < 3:
        return {}
    for name in ("target-only-solve-report.json", "target_solve_report.json"):
        report_path = path.parents[2] / name
        if not report_path.is_file():
            continue
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _status_axes(
    path: Path | None,
    *,
    job: Mapping[str, Any],
    branches: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Project five independent outcome axes from their existing owners.

    This is display-only. RouteJSON replay comes from the Director plan,
    stock and scientific acceptance come from the final gate report,
    paper-equivalent status comes from its benchmark metric, and chemistry
    comes from the independent Critic. No axis is allowed to imply another.
    """

    report = _solve_report(path)
    if report:
        director_plan: dict[str, Any] = {}
        for raw in report.get("director_outcomes") or []:
            if not isinstance(raw, Mapping):
                continue
            plan = raw.get("plan")
            if isinstance(plan, Mapping):
                director_plan = dict(plan)
                break
        skeletons = [
            dict(value)
            for value in director_plan.get("multi_step_skeletons") or []
            if isinstance(value, Mapping)
        ]
        replay_total = len(skeletons)
        replay_complete = sum(
            value.get("routejson_replay_complete") is True
            or dict(value.get("routejson_replay_validation") or {}).get(
                "complete"
            )
            is True
            for value in skeletons
        )
        replay_state = (
            "complete"
            if replay_total and replay_complete == replay_total
            else "partial"
            if replay_complete
            else "unavailable"
        )

        gates = dict(report.get("gates") or {})
        counts = dict(gates.get("counts") or {})
        stock_count = int(
            counts.get("canonical_stock_closed_routes")
            or counts.get("stock_closed_skeletons")
            or 0
        )
        stock_state = "closed" if stock_count > 0 else "open"

        paper = dict(report.get("paper_equivalent") or {})
        paper_solved = (
            report.get("paper_equivalent_solved") is True
            or paper.get("paper_equivalent_solved") is True
        )
        paper_reached = (
            report.get("paper_reach") is True
            or paper.get("paper_reach") is True
        )
        paper_state = (
            "solved" if paper_solved else "reached" if paper_reached else "unresolved"
        )

        critic_counts: dict[str, int] = {}
        for skeleton in skeletons:
            critic = dict(skeleton.get("chemical_critic") or {})
            state = str(
                critic.get("status")
                or critic.get("overall_assessment")
                or "unavailable"
            ).casefold()
            critic_counts[state] = critic_counts.get(state, 0) + 1
        observed_critic_states = {
            key for key, count in critic_counts.items() if count > 0
        }
        chemical_state = (
            next(iter(observed_critic_states))
            if len(observed_critic_states) == 1
            else "mixed"
            if observed_critic_states
            else "unavailable"
        )

        claim = dict(report.get("claim") or {})
        scientifically_accepted = (
            claim.get("accepted_under_configured_policy") is True
            or claim.get("scientific_proof_accepted") is True
        )
        scientific_state = "accepted" if scientifically_accepted else "unresolved"
        return {
            "routejson_replay": {
                "state": replay_state,
                "complete_routes": replay_complete,
                "route_count": replay_total,
            },
            "stock_closure": {
                "state": stock_state,
                "canonical_stock_closed_routes": stock_count,
            },
            "paper_equivalent": {"state": paper_state},
            "chemical_critic": {
                "state": chemical_state,
                "counts": critic_counts,
            },
            "scientific_acceptance": {"state": scientific_state},
            "source": "target_solve_report",
            "semantics": {
                "axes_are_independent": True,
                "display_projection_grants_no_authority": True,
            },
        }

    replay_complete = sum(bool(value.get("steps")) for value in branches.values())
    critic_counts: dict[str, int] = {}
    for branch in branches.values():
        state = str(branch.get("chemical_critic_status") or "").casefold()
        if state:
            critic_counts[state] = critic_counts.get(state, 0) + 1
    observed_critic_states = set(critic_counts)
    chemical_state = (
        next(iter(observed_critic_states))
        if len(observed_critic_states) == 1
        else "mixed"
        if observed_critic_states
        else "pending"
    )
    return {
        "routejson_replay": {
            "state": "partial" if replay_complete else "pending",
            "complete_routes": replay_complete,
            "route_count": len(branches),
        },
        "stock_closure": {"state": "pending"},
        "paper_equivalent": {
            "state": str(job.get("paper_equivalent_status") or "pending")
        },
        "chemical_critic": {
            "state": chemical_state,
            "counts": critic_counts,
        },
        "scientific_acceptance": {
            "state": str(job.get("scientific_status") or "pending")
        },
        "source": "live_event_projection",
        "semantics": {
            "axes_are_independent": True,
            "display_projection_grants_no_authority": True,
        },
    }


def _settled_canonical_steps(
    path: Path | None,
) -> dict[int, list[dict[str, Any]]]:
    """Recover final materialized Codex steps omitted from the model stream.

    Editor mutations are applied by the Host after the model output. Older
    director ledgers do not always contain a following model input that
    repeats those accepted steps, while the final candidate lifecycle does.
    Lifecycle decides materialization and canonical structures; the saved
    Editor output supplies display conditions.
    """

    if path is None:
        return {}
    report = _solve_report(path)
    if not report:
        return {}

    editor_steps = _saved_editor_steps(path)
    by_branch: dict[int, dict[str, dict[str, Any]]] = {
        index: {} for index in range(1, 4)
    }
    lifecycle = dict(report.get("candidate_lifecycle") or {})
    for raw_record in lifecycle.get("records") or []:
        if not isinstance(raw_record, Mapping):
            continue
        record = dict(raw_record)
        if dict(record.get("materialization") or {}).get("materialized") is not True:
            continue
        if not dict(record.get("portfolio") or {}).get("selected_route_ids"):
            continue
        for raw_origin in record.get("origin_records") or []:
            if not isinstance(raw_origin, Mapping):
                continue
            origin = dict(raw_origin)
            if str(origin.get("origin_kind") or "") != "codex_global_director":
                continue
            step_id = str(origin.get("proposal_id") or "")
            branch_index, _call_index = _branch_and_call(step_id)
            if branch_index not in {1, 2, 3} or not step_id:
                continue
            step = deepcopy(editor_steps.get(step_id) or {})
            step.update(
                step_id=step_id,
                product_smiles=str(record.get("product_smiles") or ""),
                precursor_smiles=[
                    str(value)
                    for value in record.get("precursor_smiles") or []
                    if str(value).strip()
                ],
                reaction_family=_clean_text(
                    origin.get("transformation_hypothesis")
                    or step.get("reaction_family")
                ),
                status="host_replayed",
                canonical_edge_id=str(record.get("edge_id") or ""),
            )
            step.setdefault("conditions", [])
            step.setdefault("catalyst", "")
            step.setdefault("transformation_rationale", "")
            by_branch[branch_index][step_id] = step
            break
    return {
        branch_index: list(steps.values())
        for branch_index, steps in by_branch.items()
        if steps
    }


def _saved_editor_steps(path: Path | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for event in _read_director_events(path):
        if not isinstance(event, Mapping) or event.get("event") != "model_output":
            continue
        if str(event.get("artifact_type") or "") != "RetrosynthesisProposalReport":
            continue
        artifact = dict(event.get("output_artifact") or {})
        payload = dict(artifact.get("payload") or {})
        for raw_candidate in payload.get("candidates") or []:
            if not isinstance(raw_candidate, Mapping):
                continue
            replace_span = raw_candidate.get("replace_span")
            if not isinstance(replace_span, Mapping):
                continue
            for step in _route_steps(replace_span.get("revised_steps")):
                step_id = str(step.get("step_id") or "")
                if step_id:
                    step["route_provenance"] = "editor_repair"
                    rows[step_id] = step
    return rows


def _editor_repair_step_keys(
    path: Path | None,
) -> set[tuple[int, int, int]]:
    """Identify Builder nodes generated inside an Editor-repair lineage.

    An Editor often changes the search context and then hands control back to
    the normal Builder.  Those later nodes keep ``:node:`` ids, so looking
    only for Editor ``revised_steps`` mislabels the repaired lane as another
    original Builder chain.  Track the durable event order per branch and
    retain the canonical proposal key instead of depending on id prefixes.
    """

    repair_active: set[int] = set()
    keys: set[tuple[int, int, int]] = set()
    for event in _read_director_events(path):
        if (
            not isinstance(event, Mapping)
            or event.get("event") != "model_output"
            or str(event.get("artifact_type") or "")
            != "RetrosynthesisProposalReport"
        ):
            continue
        task_id = str(event.get("task_id") or "")
        branch_index, _call_index = _branch_and_call(task_id)
        if branch_index not in {1, 2, 3}:
            continue
        artifact = dict(event.get("output_artifact") or {})
        payload = dict(artifact.get("payload") or {})
        candidates = [
            dict(value)
            for value in payload.get("candidates") or []
            if isinstance(value, Mapping)
        ]
        if ":editor:" in task_id:
            repair_active.add(branch_index)
            for candidate in candidates:
                replace_span = candidate.get("replace_span")
                if not isinstance(replace_span, Mapping):
                    continue
                for step in _route_steps(replace_span.get("revised_steps")):
                    match = _PROPOSAL_RE.search(str(step.get("step_id") or ""))
                    if match:
                        keys.add(tuple(int(item) for item in match.groups()))
            continue
        if branch_index not in repair_active:
            continue
        for candidate in candidates:
            if key := _proposal_key(task_id, candidate):
                keys.add(key)
    return keys


def _merge_canonical_route_step(
    steps: list[dict[str, Any]],
    canonical_step: Mapping[str, Any],
) -> None:
    """Overlay canonical identity while retaining richer saved display data."""

    step_id = str(canonical_step.get("step_id") or "")
    for index, current in enumerate(steps):
        if str(current.get("step_id") or "") != step_id:
            continue
        merged = {**dict(canonical_step), **current}
        for key in (
            "product_smiles",
            "precursor_smiles",
            "reaction_family",
            "status",
            "canonical_edge_id",
        ):
            merged[key] = deepcopy(canonical_step.get(key))
        steps[index] = merged
        return
    steps.append(deepcopy(dict(canonical_step)))


def _merge_compact_route_path(
    steps: list[dict[str, Any]],
    compact_path: Any,
) -> None:
    """Accumulate sibling root-to-leaf contexts into one convergent tree.

    A paper-matched context contains only the currently selected leaf path.
    Replacing the branch with that path makes convergent precursor subtrees
    alternate on screen.  Host step ids are stable, so upsert the selected
    path and retain already materialized siblings before restoring tree order.
    """

    for raw_step in compact_path or []:
        if isinstance(raw_step, Mapping):
            _merge_canonical_route_step(steps, raw_step)
    _order_route_steps(steps)


def _stable_editor_route_snapshot(
    *,
    current_steps: Any,
    incoming_steps: Any,
    remembered_steps: Any = (),
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Keep draft-only Editor rows from becoming route-topology authority.

    Editor ``revised_steps`` intentionally carry graph operations while their
    precursor arrays remain empty until Host materialization.  Such a partial
    snapshot is useful as a patch proposal, but replacing the last canonical
    route with it turns every affected product into a false root.  Preserve
    the last Host-visible structures by step id, retain all prior components,
    and wait for a complete later Host/critic context before changing topology.
    """

    incoming = [
        deepcopy(dict(value))
        for value in incoming_steps or []
        if isinstance(value, Mapping)
    ]
    incomplete_ids = [
        str(step.get("step_id") or "")
        for step in incoming
        if not step.get("precursor_smiles")
    ]
    if incoming and not incomplete_ids:
        _order_route_steps(incoming)
        return incoming, [], True

    stable = [
        deepcopy(dict(value))
        for value in current_steps or []
        if isinstance(value, Mapping)
    ]
    remembered_by_id = {
        str(value.get("step_id") or ""): deepcopy(dict(value))
        for value in remembered_steps or []
        if isinstance(value, Mapping) and str(value.get("step_id") or "")
    }
    stable_by_id = {
        str(value.get("step_id") or ""): value
        for value in stable
        if str(value.get("step_id") or "")
    }
    for draft in incoming:
        step_id = str(draft.get("step_id") or "")
        if draft.get("precursor_smiles"):
            _merge_canonical_route_step(stable, draft)
            stable_by_id[step_id] = draft
            continue
        prior = stable_by_id.get(step_id) or remembered_by_id.get(step_id)
        if not prior or not prior.get("precursor_smiles"):
            continue
        display = {**deepcopy(dict(prior)), **draft}
        for key in ("product_smiles", "precursor_smiles", "status"):
            display[key] = deepcopy(prior.get(key))
        _merge_canonical_route_step(stable, display)
    _order_route_steps(stable)
    return stable, list(dict.fromkeys(incomplete_ids)), False


def _editor_patch_step_ids(candidate: Mapping[str, Any]) -> list[str]:
    """Return the saved local replacement boundary without granting authority."""

    replace_span = candidate.get("replace_span")
    if not isinstance(replace_span, Mapping):
        return []
    values = [
        str(value)
        for value in replace_span.get("remove_step_ids") or []
        if str(value).strip()
    ]
    values.extend(
        str(step.get("step_id") or "")
        for step in replace_span.get("revised_steps") or []
        if isinstance(step, Mapping) and str(step.get("step_id") or "").strip()
    )
    return list(dict.fromkeys(values))


def _record_builder_step_insights(
    insights: dict[str, dict[str, Any]],
    candidate: Mapping[str, Any],
) -> None:
    """Retain Builder scheduling fields across later Host route snapshots."""

    sources = [candidate]
    replace_span = candidate.get("replace_span")
    if isinstance(replace_span, Mapping):
        revised_steps = [
            value
            for value in replace_span.get("revised_steps") or []
            if isinstance(value, Mapping)
        ]
        if revised_steps:
            sources = revised_steps
    for source in sources:
        step_id = str(
            source.get("step_id") or candidate.get("candidate_id") or ""
        ).strip()
        if not step_id:
            continue
        relation = _clean_text(
            source.get("checkpoint_relation")
            or candidate.get("checkpoint_relation")
        )
        limitations = _text_values(
            source.get("limitations")
            or source.get("builder_limitations")
            or candidate.get("limitations")
            or candidate.get("builder_limitations")
        )
        values = insights.setdefault(step_id, {})
        if relation:
            values["checkpoint_relation"] = relation
        if limitations:
            values["builder_limitations"] = limitations


def _record_critic_step_insights(
    insights: dict[str, dict[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    """Keep the Critic's explicit per-step assessment without inventing why."""

    for raw in payload.get("step_assessments") or []:
        if not isinstance(raw, Mapping):
            continue
        step_id = str(raw.get("step_id") or "").strip()
        if not step_id:
            continue
        reasons = [
            _clean_text(value)
            for value in raw.get("reasons") or []
            if str(value).strip()
        ]
        insights.setdefault(step_id, {}).update({
            "critic_verdict": _clean_text(raw.get("verdict")),
            "critic_reasons": reasons,
            "critic_suggested_revision": _clean_text(
                raw.get("suggested_revision")
            ),
            "critic_condition_assessment": _clean_text(
                raw.get("condition_assessment")
            ),
        })


def _attach_step_insights(
    branch: dict[str, Any],
    insights: Mapping[str, Mapping[str, Any]],
) -> None:
    """Attach saved Critic fields to current, pending, and canonical steps."""

    rows = list(branch.get("steps") or [])
    pending = branch.get("pending_step")
    if isinstance(pending, dict):
        rows.append(pending)
    for step in rows:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id") or "")
        combined: dict[str, Any] = {}
        match = _PROPOSAL_RE.search(step_id)
        if match:
            identity = match.groups()
            for insight_id, assessment in insights.items():
                candidate_match = _PROPOSAL_RE.search(insight_id)
                if candidate_match and candidate_match.groups() == identity:
                    combined.update(deepcopy(dict(assessment)))
        direct = insights.get(step_id)
        if direct is not None:
            combined.update(deepcopy(dict(direct)))
        if combined:
            step.update(combined)


def _order_route_steps(steps: list[dict[str, Any]]) -> None:
    """Put target-rooted dependencies before children without inventing edges."""

    if len(steps) < 2:
        return
    product_indices: dict[str, list[int]] = {}
    for index, step in enumerate(steps):
        product = str(step.get("product_smiles") or "")
        identity = canonical_smiles(product) or product.strip()
        if identity:
            product_indices.setdefault(identity, []).append(index)

    ordered_indices: list[int] = []
    seen: set[int] = set()

    def visit(index: int) -> None:
        if index in seen:
            return
        seen.add(index)
        ordered_indices.append(index)
        for precursor in steps[index].get("precursor_smiles") or []:
            identity = canonical_smiles(str(precursor)) or str(precursor).strip()
            for child_index in product_indices.get(identity, []):
                visit(child_index)

    visit(0)
    for index in range(len(steps)):
        visit(index)
    steps[:] = [steps[index] for index in ordered_indices]
    for index, step in enumerate(steps, start=1):
        step["index"] = index


def _proposal_key(
    task_id: str,
    candidate: Mapping[str, Any],
) -> tuple[int, int, int] | None:
    match = _PROPOSAL_RE.search(str(candidate.get("candidate_id") or ""))
    if match:
        return tuple(int(item) for item in match.groups())
    branch_index, call_index = _branch_and_call(task_id)
    if branch_index is None or call_index is None or ":node:" not in task_id:
        return None
    return branch_index, call_index, 1


def _host_step_from_audit(
    candidate: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "step_id": str(
            audit.get("proposal_id") or candidate.get("candidate_id") or "host-step"
        ),
        "index": 0,
        "product_smiles": str(audit.get("canonical_product_smiles") or ""),
        "precursor_smiles": [
            str(value)
            for value in audit.get("canonical_precursor_smiles") or []
            if str(value).strip()
        ],
        "reaction_family": _clean_text(
            candidate.get("reaction_family")
            or candidate.get("transformation_hypothesis")
        ),
        "transformation_rationale": _clean_text(
            candidate.get("transformation_rationale")
            or candidate.get("strategic_role")
        ),
        "checkpoint_relation": _clean_text(
            candidate.get("checkpoint_relation")
        ),
        "builder_limitations": _text_values(
            candidate.get("limitations")
            or candidate.get("builder_limitations")
        ),
        "conditions": _route_condition_texts(candidate),
        "catalyst": _route_catalyst(candidate),
        "status": "host_replayed",
    }


def _upsert_route_step(steps: list[dict[str, Any]], step: dict[str, Any]) -> None:
    step_id = str(step.get("step_id") or "")
    for index, current in enumerate(steps):
        if str(current.get("step_id") or "") == step_id:
            steps[index] = step
            break
    else:
        steps.append(step)
    for index, current in enumerate(steps, start=1):
        current["index"] = index


def _replay_step_state(branch: Mapping[str, Any]) -> dict[str, str]:
    rows = list(branch.get("steps") or [])
    pending = branch.get("pending_step")
    if isinstance(pending, Mapping):
        rows.append(pending)
    state: dict[str, str] = {}
    for index, step in enumerate(rows, start=1):
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("step_id") or f"step-{index}")
        core = {
            "product_smiles": step.get("product_smiles"),
            "precursor_smiles": step.get("precursor_smiles"),
            "reaction_family": step.get("reaction_family"),
            "conditions": step.get("conditions"),
            "status": step.get("status"),
            "checkpoint_relation": step.get("checkpoint_relation"),
            "builder_limitations": step.get("builder_limitations"),
            "critic_verdict": step.get("critic_verdict"),
            "critic_reasons": step.get("critic_reasons"),
            "critic_suggested_revision": step.get(
                "critic_suggested_revision"
            ),
        }
        state[step_id] = json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return state


def _replay_topology_state(branch: Mapping[str, Any]) -> dict[str, str]:
    """Fingerprint only Host-visible route connectivity for viewport updates."""

    state: dict[str, str] = {}
    for index, step in enumerate(branch.get("steps") or [], start=1):
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("step_id") or f"step-{index}")
        state[step_id] = json.dumps(
            {
                "product_smiles": step.get("product_smiles"),
                "precursor_smiles": step.get("precursor_smiles"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return state


def _embedded_context(prompt: str, markers: tuple[str, ...]) -> dict[str, Any]:
    matches = [
        (prompt.rfind(marker), marker)
        for marker in markers
        if prompt.rfind(marker) >= 0
    ]
    if not matches:
        return {}
    offset, marker = max(matches, key=lambda value: value[0])
    raw = prompt[offset + len(marker) :].lstrip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _route_builder_context(prompt: str) -> dict[str, Any]:
    return _embedded_context(prompt, _ROUTE_CONTEXT_MARKERS)


def _critic_route_context(prompt: str) -> dict[str, Any]:
    return _embedded_context(prompt, _CRITIC_CONTEXT_MARKERS)


def _valid_branch_index(value: Any) -> int | None:
    try:
        branch_index = int(value)
    except (TypeError, ValueError):
        return None
    return branch_index if branch_index in {1, 2, 3} else None


def _branch_and_call(task_id: str) -> tuple[int | None, int | None]:
    match = _BRANCH_RE.search(task_id)
    if not match:
        return None, None
    branch = int(match.group(1))
    return (branch if branch in {1, 2, 3} else None), int(match.group(2))


def _strategy_cards(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value or [], start=1):
        if not isinstance(raw, Mapping) or index > 3:
            continue
        rows.append(
            {
                "strategy_id": f"strategy-{index}",
                "index": index,
                "signature": _clean_text(
                    raw.get("strategy_signature")
                    or raw.get("key_forward_transformation")
                    or f"Strategy {index}"
                ),
                "query": _clean_text(raw.get("strategy_query")),
                "status": "planning",
                "step_count": 0,
                "renderable_step_count": 0,
                "unresolved_step_count": 0,
                "model_calls": 0,
            }
        )
    return rows


def _empty_strategy(index: int) -> dict[str, Any]:
    return {
        "strategy_id": f"strategy-{index}",
        "index": index,
        "signature": "",
        "query": "等待独立策略输出",
        "status": "planning",
        "step_count": 0,
        "renderable_step_count": 0,
        "unresolved_step_count": 0,
        "model_calls": 0,
    }


def _empty_branch(index: int) -> dict[str, Any]:
    return {
        "branch_index": index,
        "status": "waiting",
        "model_calls": 0,
        "steps": [],
        "pending_step": None,
        "chemical_critic_status": "",
    }


def _route_step_metrics(branch: Mapping[str, Any]) -> dict[str, int]:
    steps = [value for value in branch.get("steps") or [] if isinstance(value, Mapping)]
    renderable = sum(bool(value.get("precursor_smiles")) for value in steps)
    return {
        "step_count": len(steps),
        "renderable_step_count": renderable,
        "unresolved_step_count": len(steps) - renderable,
    }


def _complete_branch_review(branch: dict[str, Any]) -> None:
    """Close the prior branch when sequential review advances to another."""

    branch["pending_step"] = None
    if branch["status"] not in {"failed", "rejected", "cancelled"}:
        branch["status"] = "complete"


def _route_steps(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        precursor_smiles = _route_structure_values(
            raw,
            direct_key="precursor_smiles",
            mapped_key="mapped_precursor_smiles",
        )
        rows.append(
            {
                "step_id": str(raw.get("step_id") or f"step-{index}"),
                "index": index,
                "product_smiles": _route_structure_value(
                    raw,
                    direct_key="product_smiles",
                    mapped_key="mapped_product_smiles",
                ),
                "precursor_smiles": precursor_smiles,
                "reaction_family": _clean_text(
                    raw.get("reaction_family")
                    or raw.get("transformation_hypothesis")
                ),
                "transformation_rationale": _clean_text(
                    raw.get("transformation_rationale")
                    or raw.get("strategic_role")
                ),
                "checkpoint_relation": _clean_text(
                    raw.get("checkpoint_relation")
                ),
                "builder_limitations": _text_values(
                    raw.get("limitations")
                    or raw.get("builder_limitations")
                ),
                "conditions": _route_condition_texts(raw),
                "catalyst": _route_catalyst(raw),
                "status": "host_replayed",
            }
        )
    return rows


def _text_values(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    return [
        cleaned
        for item in values
        if not isinstance(item, Mapping)
        and (cleaned := _clean_text(item))
    ]


def _route_structure_value(
    raw: Mapping[str, Any],
    *,
    direct_key: str,
    mapped_key: str,
) -> str:
    direct = str(raw.get(direct_key) or "").strip()
    if direct:
        return direct
    return canonical_smiles(str(raw.get(mapped_key) or ""))


def _route_structure_values(
    raw: Mapping[str, Any],
    *,
    direct_key: str,
    mapped_key: str,
) -> list[str]:
    direct = raw.get(direct_key)
    direct_values = [direct] if isinstance(direct, str) else list(direct or [])
    rows = [str(value) for value in direct_values if str(value).strip()]
    if rows:
        return rows
    mapped = raw.get(mapped_key)
    mapped_values = [mapped] if isinstance(mapped, str) else list(mapped or [])
    return [
        canonical
        for value in mapped_values
        if (canonical := canonical_smiles(str(value)))
    ]


def _attach_route_provenance(
    steps: Any,
    *,
    editor_step_ids: set[str],
    editor_repair_step_keys: set[tuple[int, int, int]],
) -> None:
    """Retain whether a canonical step came from Builder or Editor.

    The Host remains authoritative for structures and graph edges.  This field
    is presentation provenance only: it lets the renderer distinguish a local
    repair lane from the rejected Builder lane without guessing from node
    numbers or from molecule complexity.
    """

    for raw_step in steps or []:
        if not isinstance(raw_step, dict):
            continue
        step_id = str(raw_step.get("step_id") or "")
        match = _PROPOSAL_RE.search(step_id)
        proposal_key = (
            tuple(int(item) for item in match.groups()) if match else None
        )
        raw_step["route_provenance"] = (
            "editor_repair"
            if step_id in editor_step_ids
            or proposal_key in editor_repair_step_keys
            else "builder"
        )


def _route_lane_critic_state(
    rows: list[dict[str, Any]],
    child_index: int,
) -> str:
    """Summarize saved Critic verdicts inside one topology lane."""

    verdicts: list[str] = []
    pending = [child_index]
    visited: set[int] = set()
    while pending:
        step_index = pending.pop()
        if step_index in visited or not 0 <= step_index < len(rows):
            continue
        visited.add(step_index)
        step = rows[step_index]
        verdict = str(step.get("critic_verdict") or "").casefold()
        if verdict:
            verdicts.append(verdict)
        for group in step.get("display_precursors") or []:
            if not isinstance(group, Mapping):
                continue
            pending.extend(
                value
                for value in group.get("child_step_indices") or []
                if isinstance(value, int)
            )
    if "reject" in verdicts:
        return "reject"
    if "uncertain" in verdicts:
        return "uncertain"
    if verdicts and all(value == "pass" for value in verdicts):
        return "pass"
    return "pending"


def _annotate_route_topology(steps: Any) -> None:
    """Link route steps into a molecule-identity tree for the renderer."""

    rows = [value for value in steps or [] if isinstance(value, dict)]
    group_identities: list[list[str]] = []
    for step in rows:
        precursors = [str(value) for value in step.get("precursor_smiles") or []]
        groups: list[dict[str, Any]] = []
        group_by_identity: dict[str, dict[str, Any]] = {}
        identities: list[str] = []
        for precursor_index, smiles in enumerate(precursors):
            identity = canonical_smiles(smiles) or smiles.strip()
            group = group_by_identity.get(identity)
            if group is None:
                group = {
                    "group_index": len(groups),
                    "smiles": smiles,
                    "count": 0,
                    "source_indices": [],
                    "continues_to_next_step": False,
                    "child_step_indices": [],
                }
                groups.append(group)
                group_by_identity[identity] = group
                identities.append(identity)
            group["count"] += 1
            group["source_indices"].append(precursor_index)
        step["parent_step_index"] = None
        step["parent_precursor_indices"] = []
        step["continuation_precursor_indices"] = []
        step["topology_status"] = "root"
        step["display_precursors"] = groups
        group_identities.append(identities)

    for child_index in range(1, len(rows)):
        product_identity = canonical_smiles(
            str(rows[child_index].get("product_smiles") or "")
        )
        parent_match: tuple[int, int] | None = None
        if product_identity:
            for parent_index in range(child_index - 1, -1, -1):
                try:
                    group_index = group_identities[parent_index].index(
                        product_identity
                    )
                except ValueError:
                    continue
                parent_match = parent_index, group_index
                break
        if parent_match is None:
            rows[child_index]["topology_status"] = "orphan"
            continue
        parent_index, group_index = parent_match
        group = rows[parent_index]["display_precursors"][group_index]
        group["child_step_indices"].append(child_index)
        group["continues_to_next_step"] = child_index == parent_index + 1
        parent_precursor_indices = list(group["source_indices"])
        rows[child_index]["parent_step_index"] = parent_index
        rows[child_index]["parent_precursor_indices"] = parent_precursor_indices
        rows[child_index]["topology_status"] = "linked"
        if child_index == parent_index + 1:
            rows[parent_index]["continuation_precursor_indices"] = (
                parent_precursor_indices
            )

    for parent_index, step in enumerate(rows):
        for group in step.get("display_precursors") or []:
            child_indices = list(group.get("child_step_indices") or [])
            group["has_route_alternatives"] = len(child_indices) > 1
            group["alternative_lane_count"] = len(child_indices)
            group["alternative_lane_order"] = [
                str(rows[child_index].get("step_id") or "")
                for child_index in child_indices
                if 0 <= child_index < len(rows)
            ]
            lanes: list[dict[str, Any]] = []
            for lane_index, child_index in enumerate(child_indices):
                if not 0 <= child_index < len(rows):
                    continue
                child = rows[child_index]
                provenance = str(
                    child.get("route_provenance") or "builder"
                )
                critic_state = _route_lane_critic_state(rows, child_index)
                child["alternative_lane_index"] = lane_index
                child["alternative_lane_count"] = len(child_indices)
                child["alternative_parent_step_index"] = parent_index
                child["alternative_parent_group_index"] = int(
                    group.get("group_index") or 0
                )
                lanes.append(
                    {
                        "lane_index": lane_index,
                        "child_step_index": child_index,
                        "child_step_id": str(child.get("step_id") or ""),
                        "provenance": provenance,
                        "critic_state": critic_state,
                    }
                )
            group["alternative_lanes"] = lanes


def _stabilize_replay_alternative_groups(
    replay_frames: list[dict[str, Any]],
    *,
    branches: Mapping[int, Mapping[str, Any]],
) -> None:
    """Reserve final alternative-lane positions in earlier replay frames.

    A later Editor repair must appear as a new lower lane.  Annotating the
    already visible shared precursor in earlier frames lets the browser keep
    the Builder lane in the same slot instead of rebuilding the layout when
    the second child first arrives.  Future reaction content is never copied
    into an earlier frame.
    """

    templates: dict[tuple[int, str, str], dict[str, Any]] = {}
    for branch_index, branch in branches.items():
        for step in branch.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            parent_step_id = str(step.get("step_id") or "")
            for group in step.get("display_precursors") or []:
                if (
                    not isinstance(group, Mapping)
                    or not group.get("has_route_alternatives")
                ):
                    continue
                identity = canonical_smiles(str(group.get("smiles") or ""))
                templates[(branch_index, parent_step_id, identity)] = {
                    "has_route_alternatives": True,
                    "alternative_lane_count": int(
                        group.get("alternative_lane_count") or 0
                    ),
                    "alternative_lane_order": list(
                        group.get("alternative_lane_order") or []
                    ),
                }
    if not templates:
        return

    for frame in replay_frames:
        for branch in frame.get("branch_updates") or []:
            if not isinstance(branch, Mapping):
                continue
            branch_index = int(branch.get("branch_index") or 0)
            for step in branch.get("steps") or []:
                if not isinstance(step, Mapping):
                    continue
                parent_step_id = str(step.get("step_id") or "")
                for group in step.get("display_precursors") or []:
                    if not isinstance(group, dict):
                        continue
                    identity = canonical_smiles(str(group.get("smiles") or ""))
                    template = templates.get(
                        (branch_index, parent_step_id, identity)
                    )
                    if template:
                        group.update(deepcopy(template))


def _route_condition_predictions(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(value)
        for value in raw.get("condition_predictions") or []
        if isinstance(value, Mapping)
    ]


def _route_condition_texts(raw: Mapping[str, Any]) -> list[str]:
    direct = raw.get("conditions")
    if isinstance(direct, str):
        values = [direct]
    else:
        values = list(direct or [])
    if not values:
        for prediction in _route_condition_predictions(raw):
            reagents = prediction.get("reagents")
            values.extend([reagents] if isinstance(reagents, str) else reagents or [])
    return [_clean_text(value) for value in values if str(value).strip()]


def _route_catalyst(raw: Mapping[str, Any]) -> str:
    direct = _clean_text(raw.get("catalyst"))
    if direct:
        return direct
    for prediction in _route_condition_predictions(raw):
        catalyst = _clean_text(prediction.get("catalyst"))
        if catalyst:
            return catalyst
    return ""


def _paper_matched_route_steps(
    context: Mapping[str, Any],
    *,
    previous_steps: Any,
    remembered_steps: Any = (),
    target_smiles: str,
) -> list[dict[str, Any]]:
    """Recover the host-replayed path from the compact paper-matched context.

    Paper-matched Builder prompts intentionally carry only reaction descriptors
    plus the currently selected, atom-mapped leaf.  Earlier structures are
    retained from the preceding projection; the selected leaf is the primary
    precursor produced by the final connected reaction.
    """

    descriptors = [
        dict(value)
        for value in context.get("connected_path_reactions") or []
        if isinstance(value, Mapping)
    ]
    if not descriptors:
        return []
    remembered_by_id = {
        str(value.get("step_id") or ""): dict(value)
        for value in remembered_steps or []
        if isinstance(value, Mapping) and str(value.get("step_id") or "")
    }
    prior_by_id = {
        str(value.get("step_id") or ""): dict(value)
        for value in previous_steps or []
        if isinstance(value, Mapping) and str(value.get("step_id") or "")
    }
    selected_leaf = canonical_smiles(
        str(context.get("selected_leaf_mapped") or "")
    )
    ancestor_smiles = [
        canonical_smiles(str(value)) or str(value).strip()
        for value in context.get("ancestor_smiles") or []
        if str(value).strip()
    ]
    split_context = context.get("current_split_context")
    co_precursors = (
        [
            canonical_smiles(str(value.get("mapped_smiles") or ""))
            for value in split_context.get("co_precursors") or []
            if isinstance(value, Mapping)
        ]
        if isinstance(split_context, Mapping)
        else []
    )
    rows: list[dict[str, Any]] = []
    for index, descriptor in enumerate(descriptors, start=1):
        step_id = str(descriptor.get("step_id") or f"step-{index}")
        remembered = remembered_by_id.get(step_id, {})
        prior = prior_by_id.get(step_id, {})
        previous_precursors = list(rows[-1]["precursor_smiles"]) if rows else []
        saved_product = str(
            remembered.get("product_smiles")
            or prior.get("product_smiles")
            or ""
        )
        product_smiles = (
            str(target_smiles)
            if index == 1
            else str(
                saved_product
                or (
                    ancestor_smiles[0]
                    if index == len(descriptors) and ancestor_smiles
                    else ""
                )
                or (previous_precursors[0] if previous_precursors else "")
                or ""
            )
        )
        precursor_smiles = [
            str(value)
            for value in (
                remembered.get("precursor_smiles")
                or prior.get("precursor_smiles")
                or []
            )
            if str(value).strip()
        ]
        if index == len(descriptors) and selected_leaf:
            selected_path_precursors = list(
                dict.fromkeys(
                    value
                    for value in (selected_leaf, *co_precursors)
                    if value
                )
            )
            if len(selected_path_precursors) > len(precursor_smiles):
                precursor_smiles = selected_path_precursors
        rows.append(
            {
                "step_id": step_id,
                "index": index,
                "product_smiles": product_smiles
                or saved_product,
                "precursor_smiles": precursor_smiles,
                "reaction_family": _clean_text(
                    descriptor.get("reaction_family")
                    or prior.get("reaction_family")
                    or remembered.get("reaction_family")
                ),
                "transformation_rationale": _clean_text(
                    descriptor.get("edit_summary")
                    or prior.get("transformation_rationale")
                    or remembered.get("transformation_rationale")
                ),
                "checkpoint_relation": _clean_text(
                    descriptor.get("checkpoint_relation")
                    or prior.get("checkpoint_relation")
                    or remembered.get("checkpoint_relation")
                ),
                "builder_limitations": _text_values(
                    descriptor.get("limitations")
                    or prior.get("builder_limitations")
                    or remembered.get("builder_limitations")
                ),
                "conditions": [
                    _clean_text(value)
                    for value in (
                        prior.get("conditions")
                        or remembered.get("conditions")
                        or []
                    )
                    if str(value).strip()
                ],
                "catalyst": _clean_text(
                    prior.get("catalyst") or remembered.get("catalyst")
                ),
                "status": "host_replayed",
            }
        )
    return rows


def _pending_step(candidate: Mapping[str, Any], call_index: int | None) -> dict[str, Any]:
    source = candidate
    replace_span = candidate.get("replace_span")
    if isinstance(replace_span, Mapping):
        revised_steps = [
            value
            for value in replace_span.get("revised_steps") or []
            if isinstance(value, Mapping)
        ]
        if revised_steps:
            source = revised_steps[0]
    return {
        "step_id": str(
            source.get("step_id")
            or candidate.get("candidate_id")
            or f"pending-{call_index or 0}"
        ),
        "index": call_index or 0,
        "product_smiles": str(source.get("product_smiles") or ""),
        "precursor_smiles": [
            str(value)
            for value in source.get("precursor_smiles") or []
            if str(value).strip()
        ],
        "reaction_family": _clean_text(
            source.get("reaction_family")
            or source.get("transformation_hypothesis")
        ),
        "transformation_rationale": _clean_text(
            source.get("transformation_rationale")
            or source.get("strategic_role")
            or candidate.get("repair_summary")
        ),
        "checkpoint_relation": _clean_text(
            source.get("checkpoint_relation")
            or candidate.get("checkpoint_relation")
        ),
        "builder_limitations": _text_values(
            source.get("limitations")
            or source.get("builder_limitations")
            or candidate.get("limitations")
            or candidate.get("builder_limitations")
        ),
        "conditions": _route_condition_texts(source),
        "catalyst": _route_catalyst(source),
        "status": "pending_host_replay",
    }


def _activity(
    event: Mapping[str, Any],
    *,
    kind: str,
    title: str,
    detail: str,
    branch_index: int | None,
) -> dict[str, Any]:
    usage = dict(event.get("usage") or {})
    return {
        "activity_id": str(event.get("task_id") or ""),
        "kind": kind,
        "branch_index": branch_index,
        "title": _clean_text(title),
        "detail": _clean_text(detail),
        "timestamp": str(event.get("timestamp") or ""),
        "model": str(event.get("model") or ""),
        "status": str(event.get("status") or ""),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def _phase(
    *,
    job_status: str,
    strategy_count: int,
    route_output_count: int,
    critic_output_count: int,
) -> str:
    if job_status == "failed":
        return "failed"
    if job_status in {"cancelling", "cancelled"}:
        return job_status
    if job_status == "interrupted":
        return "interrupted"
    if job_status == "paused":
        return "paused"
    if job_status in {"complete", "unresolved", "historical"}:
        return "complete" if job_status == "complete" else job_status
    if strategy_count < 3:
        return "strategy_generation"
    if critic_output_count:
        return "critic_review"
    if route_output_count:
        return "route_building"
    return "strategy_ready"


def _progress(
    *,
    job_status: str,
    strategy_count: int,
    route_output_count: int,
    critic_output_count: int,
) -> int:
    if job_status in _FULL_PROGRESS_JOB_STATES:
        return 100
    value = 5 + strategy_count * 7
    value += min(58, route_output_count * 3)
    value += min(14, critic_output_count * 4)
    return max(5, min(96, value))


def _projection_job_status(job: Mapping[str, Any]) -> str:
    """Project archived rows from the canonical kernel state when available."""

    if str(job.get("experiment_status") or "") == "complete":
        return "complete"
    outer = str(job.get("status") or "queued")
    if outer != "historical":
        return outer
    campaign = str(job.get("campaign_status") or "")
    if job.get("campaign_terminal") is True:
        decision = str(job.get("campaign_decision") or "")
        if decision in {"complete", "unresolved", "failed", "cancelled"}:
            return decision
    if campaign in {"cancelled", "failed"}:
        return campaign
    if campaign in {"created", "running", "paused"}:
        return "interrupted"
    return "historical"


def _clean_text(value: Any) -> str:
    text = str(value or "")
    if any(marker in text for marker in ("鈥", "揅", "搂", "鈫", "酶")):
        try:
            repaired = text.encode("gbk").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        if repaired:
            return repaired
    return text


@lru_cache(maxsize=2_048)
def render_molecule_svg(smiles: str) -> tuple[str, bool]:
    """Render a deterministic, transparent molecule SVG with a safe fallback."""

    if not smiles:
        return _molecule_placeholder("等待输入 SMILES"), False
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.error")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return _molecule_placeholder("SMILES 暂无法解析"), False
        drawer = rdMolDraw2D.MolDraw2DSVG(320, 200)
        options = drawer.drawOptions()
        options.clearBackground = False
        # RDKit's default framing can place terminal heteroatom glyphs directly
        # against the SVG edge.  The route cards then expose that as a clipped O,
        # Cl, or stereochemical label.  Reserve a real safe frame for both the
        # bond geometry and the atom-label outlines.
        options.padding = 0.14
        options.additionalAtomLabelPadding = 0.04
        options.bondLineWidth = 1.7
        options.setHighlightColour((0.35, 0.75, 0.62))
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        svg = re.sub(
            r"^\s*<\?xml[^>]*\?>\s*",
            "",
            drawer.GetDrawingText(),
            count=1,
        )
        svg = re.sub(
            r"viewBox='0 0 320 200'",
            "viewBox='-18 -18 356 236' data-autoplanner-frame='safe-v3'",
            svg,
            count=1,
        )
        return svg, True
    except (ImportError, RuntimeError, ValueError):
        return _molecule_placeholder("结构渲染器不可用"), False


@lru_cache(maxsize=2_048)
def render_molecule_png(smiles: str) -> tuple[bytes, bool]:
    """Render a fixed-pixel molecule image that cannot escape its viewport."""

    if not smiles:
        return _molecule_placeholder_png("Waiting for SMILES"), False
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Draw import rdMolDraw2D

        RDLogger.DisableLog("rdApp.error")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return _molecule_placeholder_png("SMILES unavailable"), False
        drawer = rdMolDraw2D.MolDraw2DCairo(640, 400)
        options = drawer.drawOptions()
        options.clearBackground = True
        options.padding = 0.14
        options.additionalAtomLabelPadding = 0.04
        options.bondLineWidth = 2.2
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return bytes(drawer.GetDrawingText()), True
    except (ImportError, RuntimeError, ValueError):
        return _molecule_placeholder_png("Renderer unavailable"), False


def _molecule_placeholder(message: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200" viewBox="0 0 320 200">
<rect width="320" height="200" rx="18" fill="#f7f9f7"/><path d="M112 104l24-42h48l24 42-24 42h-48z" fill="none" stroke="#cbd8d1" stroke-width="3" stroke-dasharray="7 7"/>
<text x="160" y="178" text-anchor="middle" fill="#7d8d84" font-family="system-ui,sans-serif" font-size="12">{escape(message)}</text></svg>"""


def _molecule_placeholder_png(message: str) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 400), (247, 249, 247))
    draw = ImageDraw.Draw(image)
    draw.regular_polygon(
        (320, 190, 76),
        n_sides=6,
        rotation=30,
        outline=(190, 204, 196),
        width=5,
    )
    draw.text((320, 330), message, anchor="mm", fill=(113, 130, 121))
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


__all__ = [
    "project_live_synthesis",
    "render_molecule_png",
    "register_live_synthesis_routes",
    "render_molecule_svg",
]
