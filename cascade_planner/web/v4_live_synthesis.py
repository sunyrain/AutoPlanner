"""Live Strategy/Route Builder projection for the SyntheX-matched Web surface.

The sequential director already writes one append-only ``model-io.jsonl`` row
for every model input and output.  This adapter deliberately reads that durable
stream instead of inventing a second progress authority: StrategyPortfolio
outputs create the three cards, Route Builder outputs create pending steps, and
the following host-replayed input replaces each pending step with canonical
precursors.
"""
from __future__ import annotations

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
_ROUTE_CONTEXT_MARKERS = (
    "CompactBranchContext:",
    "PaperMatchedRouteBuilderContext:",
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

    for event in _read_model_io(model_io_path):
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
                    path = _route_steps(context.get("accepted_path"))
                    if not path and "connected_path_reactions" in context:
                        path = _paper_matched_route_steps(
                            context,
                            previous_steps=branch["steps"],
                            target_smiles=target_smiles,
                        )
                    if len(path) >= len(branch["steps"]):
                        branch["steps"] = path
                    branch["pending_step"] = None
                    branch["status"] = "building"
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
                    if len(path) >= len(branch["steps"]):
                        branch["steps"] = path
                    branch["pending_step"] = None
                    branch["status"] = "reviewing"
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
            activities.append(
                _activity(
                    event,
                    kind="strategy",
                    title="三条正交策略已生成",
                    detail=str(artifact.get("summary") or "策略组合已通过结构化输出。"),
                    branch_index=None,
                )
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
                "signature": _clean_text(card.get("strategy_signature")),
                "query": _clean_text(
                    card.get("key_forward_transformation")
                    or payload.get("selection_rationale")
                ),
                "status": "planning",
                "step_count": 0,
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
            continue

        if artifact_type == "RetrosynthesisProposalReport" and branch_index:
            route_output_count += 1
            branch = branches[branch_index]
            branch["model_calls"] += 1
            candidates = [
                dict(value)
                for value in payload.get("candidates") or []
                if isinstance(value, Mapping)
            ]
            if candidates:
                candidate = candidates[0]
                branch["pending_step"] = _pending_step(candidate, call_index)
                branch["status"] = (
                    "replaying" if schema_accepted else "reviewing"
                )
                family = _clean_text(
                    candidate.get("reaction_family")
                    or candidate.get("transformation_hypothesis")
                )
                detail = str(
                    artifact.get("summary")
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
            continue

        if artifact_type == "ChemicalStrategyCritique":
            critic_output_count += 1
            critic_branch_index = (
                critic_task_branches.get(task_id) or branch_index
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

    for branch in branches.values():
        _annotate_route_topology(branch["steps"])

    for index, strategy in enumerate(strategies[:3], start=1):
        branch = branches[index]
        strategy.update(
            status=branch["status"],
            step_count=len(branch["steps"]),
            model_calls=branch["model_calls"],
        )

    job_status = _projection_job_status(job)
    if job_status in {"complete", "unresolved", "historical"}:
        for branch in branches.values():
            if branch["pending_step"]:
                branch["pending_step"]["status"] = "replay_record_unavailable"
            if branch["steps"] and branch["status"] not in {"failed", "rejected"}:
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
    return {
        "schema_version": "autoplanner.live_synthesis_projection.v1",
        "revision": revision,
        "job_id": str(job.get("job_id") or ""),
        "run_id": str(job.get("run_id") or ""),
        "target_name": str(job.get("target_name") or "blind target"),
        "target_smiles": target_smiles,
        "status": job_status,
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
        },
    }


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
                "signature": _clean_text(raw.get("strategy_signature")),
                "query": _clean_text(raw.get("strategy_query")),
                "status": "planning",
                "step_count": 0,
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
        "model_calls": 0,
    }


def _empty_branch(index: int) -> dict[str, Any]:
    return {
        "branch_index": index,
        "status": "waiting",
        "model_calls": 0,
        "steps": [],
        "pending_step": None,
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
                "conditions": _route_condition_texts(raw),
                "catalyst": _route_catalyst(raw),
                "status": "host_replayed",
            }
        )
    return rows


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
    prior_by_id = {
        str(value.get("step_id") or ""): dict(value)
        for value in previous_steps or []
        if isinstance(value, Mapping) and str(value.get("step_id") or "")
    }
    selected_leaf = canonical_smiles(
        str(context.get("selected_leaf_mapped") or "")
    )
    rows: list[dict[str, Any]] = []
    for index, descriptor in enumerate(descriptors, start=1):
        step_id = str(descriptor.get("step_id") or f"step-{index}")
        prior = prior_by_id.get(step_id, {})
        previous_precursors = list(rows[-1]["precursor_smiles"]) if rows else []
        product_smiles = (
            str(target_smiles)
            if index == 1
            else str(previous_precursors[0] if previous_precursors else "")
        )
        precursor_smiles = [
            str(value)
            for value in prior.get("precursor_smiles") or []
            if str(value).strip()
        ]
        if index == len(descriptors):
            precursor_smiles = [selected_leaf] if selected_leaf else precursor_smiles
        rows.append(
            {
                "step_id": step_id,
                "index": index,
                "product_smiles": product_smiles
                or str(prior.get("product_smiles") or ""),
                "precursor_smiles": precursor_smiles,
                "reaction_family": _clean_text(
                    descriptor.get("reaction_family")
                    or prior.get("reaction_family")
                ),
                "transformation_rationale": _clean_text(
                    descriptor.get("edit_summary")
                    or prior.get("transformation_rationale")
                ),
                "conditions": [
                    _clean_text(value)
                    for value in prior.get("conditions") or []
                    if str(value).strip()
                ],
                "catalyst": _clean_text(prior.get("catalyst")),
                "status": "host_replayed",
            }
        )
    return rows


def _pending_step(candidate: Mapping[str, Any], call_index: int | None) -> dict[str, Any]:
    return {
        "step_id": str(candidate.get("candidate_id") or f"pending-{call_index or 0}"),
        "index": call_index or 0,
        "product_smiles": str(candidate.get("product_smiles") or ""),
        "precursor_smiles": [
            str(value)
            for value in candidate.get("precursor_smiles") or []
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
        "conditions": _route_condition_texts(candidate),
        "catalyst": _route_catalyst(candidate),
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
