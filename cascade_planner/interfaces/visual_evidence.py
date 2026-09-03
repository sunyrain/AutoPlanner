"""Sparse visual source candidates with explicit RunKernel cost accounting."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from cascade_planner.application.run_kernel import RunKernelError
from cascade_planner.harness.visual_literature_chain_agent import (
    run_visual_literature_chain_agent,
)
from cascade_planner.interfaces.visual_evidence_contract import (
    VISUAL_EVIDENCE_REQUEST_SCHEMA,
    VisualEvidenceError,
    normalized_usage as _normalized_usage,
    stage as _stage,
    validate_request_digest as _validate_request_digest,
)
from cascade_planner.interfaces.visual_observation_normalization import (
    VISUAL_EVIDENCE_OBSERVATION_SCHEMA as VISUAL_EVIDENCE_OBSERVATION_SCHEMA,
    normalize_visual_observation as _normalize_visual_observation,
)
from cascade_planner.interfaces.visual_evidence_materialization import (
    materialize_visual_evidence_candidates,
)
from cascade_planner.interfaces.visual_evidence_request import (
    _compile_visual_evidence_request,
    _visual_no_candidate_reason,
    compile_visual_evidence_request,
)

VisualEvidenceProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class CodexVisualEvidenceConfig:
    cache_dir: str | Path
    model: str = "gpt-5.5"
    reasoning_effort: str = "low"
    timeout_s: float = 240.0
    max_pages: int = 6
    max_steps: int = 16

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("visual_evidence_model_missing")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("visual_evidence_reasoning_effort_invalid")
        if self.timeout_s <= 0 or not 1 <= self.max_pages <= 12:
            raise ValueError("visual_evidence_execution_limit_invalid")
        if not 1 <= self.max_steps <= 32:
            raise ValueError("visual_evidence_step_limit_invalid")


def build_codex_visual_evidence_provider(
    config: CodexVisualEvidenceConfig,
    *,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> VisualEvidenceProvider:
    """Build a one-call, no-repair visual candidate provider over Codex CLI."""

    invoke_agent = runner or run_visual_literature_chain_agent
    cache_root = Path(config.cache_dir).expanduser().resolve()

    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        _validate_request_digest(request)
        source = dict(request.get("source") or {})
        images = [
            Path(str(row.get("image_path") or "")).expanduser().resolve()
            for row in source.get("pages") or []
            if isinstance(row, Mapping)
        ][: config.max_pages]
        output_dir = cache_root / str(request.get("content_sha256") or "")[:24]
        output_dir.mkdir(parents=True, exist_ok=True)
        result = dict(
            invoke_agent(
                image_paths=images,
                output_dir=output_dir,
                target_name=str(request.get("target_name") or ""),
                target_smiles=str(request.get("target_smiles") or ""),
                source_ref=str(source.get("source_ref") or ""),
                source_title=str(source.get("title") or ""),
                expected_labels=[
                    str(value)
                    for value in source.get("expected_labels") or []
                    if str(value).strip()
                ],
                route_sequence_hint=str(source.get("route_sequence_hint") or ""),
                text_snippets=[
                    dict(value)
                    for value in source.get("text_snippets") or []
                    if isinstance(value, Mapping)
                ],
                key_path=output_dir / "ambient-auth-does-not-read-key",
                base_url="",
                model=config.model,
                timeout_s=config.timeout_s,
                allow_repair=False,
                ambient_auth=True,
                reasoning_effort=config.reasoning_effort,
            )
        )
        usage = dict(result.get("usage") or {})
        if (
            (
                list(result.get("attempts") or [])
                or list(dict(result.get("candidate_chain") or {}).get("steps") or [])
            )
            and int(usage.get("model_invocations") or 0) != 1
        ):
            raise VisualEvidenceError("codex_visual_usage_receipt_missing")
        return {
            "schema_version": "codex_visual_evidence_provider_result.v1",
            "request_sha256": str(request.get("content_sha256") or ""),
            "candidate_chain": dict(result.get("candidate_chain") or {}),
            "usage": usage,
            "provider_status": str(result.get("status") or "failed"),
            "provider_reasons": [str(value) for value in result.get("reasons") or []],
            "provider_receipt": {
                "provider_id": "autoplanner.codex_visual_source_candidate",
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "allow_repair": False,
                "max_pages": config.max_pages,
                "model_output_is_advisory": True,
            },
        }

    return invoke


def acquire_visual_evidence_candidates(
    service: Any,
    *,
    evidence_request: Mapping[str, Any],
    discovery: Mapping[str, Any],
    provider: VisualEvidenceProvider | None,
    max_pages: int = 6,
    max_steps: int = 16,
    max_source_attempts: int = 3,
) -> dict[str, Any]:
    """Try ranked sources until an exact segment with conditions is bound.

    Each source remains one independently budgeted visual invocation.  A
    generic/analogous paper is retained for audit but no longer consumes the
    entire campaign's chance to inspect a later exact-target paper.
    """

    if provider is None:
        return _stage("disabled", reason="visual_evidence_provider_not_configured")
    if not 1 <= max_source_attempts <= 8:
        raise ValueError("visual_evidence_source_attempt_limit_invalid")
    attempted_refs: set[str] = set()
    attempts: list[dict[str, Any]] = []
    aggregate_usage = _empty_usage()
    best_stage: dict[str, Any] = {}
    best_quality: tuple[int, ...] = ()
    last_diagnostics: list[dict[str, Any]] = []

    for _attempt_index in range(max_source_attempts):
        if not _visual_invocation_budget_available(service):
            break
        request, candidate_diagnostics = _compile_visual_evidence_request(
            evidence_request=evidence_request,
            discovery=discovery,
            max_pages=max_pages,
            excluded_source_refs=attempted_refs,
        )
        last_diagnostics = candidate_diagnostics
        if not request:
            break
        source_ref = str(dict(request.get("source") or {}).get("source_ref") or "")
        attempted_refs.add(source_ref)
        stage = _acquire_one_visual_source(
            service,
            request=request,
            provider=provider,
            max_steps=max_steps,
        )
        aggregate_usage = _sum_usage(
            aggregate_usage,
            dict(stage.get("model_usage") or {}),
        )
        observation = dict(stage.get("observation") or {})
        quality = _visual_observation_quality(observation)
        attempts.append(
            {
                "attempt_index": len(attempts) + 1,
                "source_ref": source_ref,
                "status": str(stage.get("status") or ""),
                "reason": str(stage.get("reason") or ""),
                "request_sha256": str(request.get("content_sha256") or ""),
                "observation_ref": dict(stage.get("observation_ref") or {}),
                "quality": _visual_quality_projection(observation),
                "model_usage": dict(stage.get("model_usage") or {}),
            }
        )
        if observation and (not best_stage or quality > best_quality):
            best_stage = stage
            best_quality = quality
        if _visual_observation_closes_source_loop(observation):
            best_stage = stage
            break
        if stage.get("status") == "budget_blocked":
            break

    if best_stage:
        selected_source_ref = str(
            dict(dict(best_stage.get("request") or {}).get("source") or {}).get(
                "source_ref"
            )
            or ""
        )
        return _stage(
            str(best_stage.get("status") or "completed"),
            reason=str(best_stage.get("reason") or ""),
            request=dict(best_stage.get("request") or {}),
            observation=dict(best_stage.get("observation") or {}),
            observation_ref=dict(best_stage.get("observation_ref") or {}),
            model_usage=aggregate_usage,
            material_events=list(best_stage.get("material_events") or []),
            attempts=attempts,
            attempted_source_count=len(attempts),
            selected_source_ref=selected_source_ref,
            source_loop_closed=_visual_observation_closes_source_loop(
                dict(best_stage.get("observation") or {})
            ),
            candidate_diagnostics=last_diagnostics,
            semantics={
                "non_exact_visual_source_triggers_next_ranked_source": True,
                "each_source_attempt_is_run_kernel_budgeted": True,
                "selection_prefers_exact_segment_edge_and_condition_binding": True,
            },
        )

    if attempts:
        last = attempts[-1]
        return _stage(
            str(last.get("status") or "unresolved"),
            reason=str(last.get("reason") or "visual_source_attempts_unresolved"),
            model_usage=aggregate_usage,
            attempts=attempts,
            attempted_source_count=len(attempts),
            candidate_diagnostics=last_diagnostics,
        )
    request, candidate_diagnostics = _compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=max_pages,
    )
    if request:
        return _stage(
            "budget_blocked",
            reason=_visual_budget_block_reason(service, request),
            request=request,
            candidate_diagnostics=candidate_diagnostics,
            model_usage=aggregate_usage,
        )
    return _stage(
        "not_needed",
        reason=_visual_no_candidate_reason(candidate_diagnostics),
        candidate_diagnostics=candidate_diagnostics,
        model_usage=aggregate_usage,
    )


def _acquire_one_visual_source(
    service: Any,
    *,
    request: Mapping[str, Any],
    provider: VisualEvidenceProvider,
    max_steps: int,
) -> dict[str, Any]:
    """Execute and settle one already-ranked visual source request."""

    task_id = f"visual-evidence:{str(request['content_sha256'])[:24]}"
    prompt_bytes = len(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    usage: dict[str, Any] = {}
    provider_called = False
    try:
        service.kernel.reserve_task(
            task_id=task_id,
            kind="model",
            idempotency_key=f"reserve:{task_id}",
            input_revision=service.kernel.state.graph_revision,
            uses_model=True,
            visual=True,
            prompt_context_bytes=prompt_bytes,
            metadata={
                "visual_evidence": True,
                "request_sha256": request["content_sha256"],
            },
        )
    except RunKernelError as exc:
        return _stage("budget_blocked", reason=str(exc), request=request)
    try:
        provider_called = True
        raw = provider(request)
        if not isinstance(raw, Mapping):
            raise VisualEvidenceError("visual_provider_result_not_object")
        result = dict(raw)
        usage = _normalized_usage(result.get("usage"))
        observation = _normalize_visual_observation(
            request,
            result=result,
            max_steps=max_steps,
        )
        ref = service.kernel.artifacts.put_json(
            observation,
            logical_name="visual_source_candidate_observation.json",
            producer="autoplanner.visual_evidence",
        )
        service.kernel.settle_task(
            task_id=task_id,
            idempotency_key=f"settle:{task_id}",
            status="completed" if observation["candidate_step_count"] else "rejected",
            output_sha256=ref.sha256,
            failure_reasons=(
                []
                if observation["candidate_step_count"]
                else ["visual_candidate_steps_not_host_parseable"]
            ),
            model_usage=usage,
            elapsed_s=float(usage.get("wall_time_s") or 0.0),
        )
    except Exception as exc:  # provider/task failures must never strand a reservation
        if provider_called and not usage:
            usage = {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_time_s": 0.0,
            }
        if task_id in service.kernel.state.in_flight_tasks:
            service.kernel.settle_task(
                task_id=task_id,
                idempotency_key=f"settle:{task_id}",
                status="failed",
                failure_reasons=[f"{type(exc).__name__}:{str(exc)[:500]}"],
                model_usage=usage,
            )
        return _stage(
            "failed",
            reason=f"{type(exc).__name__}:{exc}",
            request=request,
            model_usage=usage,
        )
    return _stage(
        "completed" if observation["candidate_step_count"] else "unresolved",
        request=request,
        observation=observation,
        observation_ref=ref.to_dict(),
        model_usage=usage,
        material_events=(
            ["visual_source_candidates_added"]
            if observation["candidate_step_count"]
            else []
        ),
    )


def _empty_usage() -> dict[str, Any]:
    return {
        "model_invocations": 0,
        "visual_invocations": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_time_s": 0.0,
    }


def _visual_invocation_budget_available(service: Any) -> bool:
    budget = service.kernel.spec.limits.model
    totals = service.kernel.state.model_totals
    return bool(
        int(totals.get("model_invocations") or 0) < budget.max_model_invocations
        and int(totals.get("visual_invocations") or 0)
        < budget.max_visual_invocations
    )


def _visual_budget_block_reason(
    service: Any,
    request: Mapping[str, Any],
) -> str:
    task_id = f"visual-evidence:{str(request['content_sha256'])[:24]}"
    lifecycle = service.kernel.task_lifecycle(task_id)
    if lifecycle["status"] != "absent":
        return "campaign_visual_evidence_call_already_admitted"

    budget = service.kernel.spec.limits.model
    totals = service.kernel.state.model_totals
    reasons: list[str] = []
    if int(totals.get("visual_invocations") or 0) >= budget.max_visual_invocations:
        reasons.append("visual_invocation_budget_exhausted")
    if int(totals.get("model_invocations") or 0) >= budget.max_model_invocations:
        reasons.append("model_invocation_budget_exhausted")
    return "campaign_visual_evidence_budget_exhausted:" + ",".join(
        reasons or ["run_kernel_budget_unavailable"]
    )


def _sum_usage(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model_invocations": int(left.get("model_invocations") or 0)
        + int(right.get("model_invocations") or 0),
        "visual_invocations": int(left.get("visual_invocations") or 0)
        + int(right.get("visual_invocations") or 0),
        "input_tokens": int(left.get("input_tokens") or 0)
        + int(right.get("input_tokens") or 0),
        "output_tokens": int(left.get("output_tokens") or 0)
        + int(right.get("output_tokens") or 0),
        "wall_time_s": float(left.get("wall_time_s") or 0.0)
        + float(right.get("wall_time_s") or 0.0),
    }


def _visual_observation_quality(observation: Mapping[str, Any]) -> tuple[int, ...]:
    projection = _visual_quality_projection(observation)
    return (
        int(projection["source_loop_closed"]),
        int(projection["exact_source_segment_count"]),
        int(projection["matched_current_edge_count"]),
        int(projection["target_anchored_step_count"]),
        int(projection["condition_bound_step_count"]),
        int(projection["admission_eligible_step_count"]),
        int(projection["candidate_step_count"]),
    )


def _visual_quality_projection(observation: Mapping[str, Any]) -> dict[str, Any]:
    steps = [
        dict(row)
        for row in observation.get("candidate_steps") or []
        if isinstance(row, Mapping)
    ]
    exact_segments = [
        row
        for row in steps
        if row.get("admission_eligible") is True
        and row.get("exact_structure_binding_candidate") is True
        and row.get("not_exact_literature_segment") is not True
        and str(row.get("source_locator") or "").strip()
    ]
    condition_bound = [
        row for row in exact_segments if _meaningful_visual_condition(row)
    ]
    closed = any(
        str(row.get("matched_current_edge_id") or "").strip()
        and _meaningful_visual_condition(row)
        for row in exact_segments
    )
    return {
        "source_loop_closed": closed,
        "exact_source_segment_count": len(exact_segments),
        "matched_current_edge_count": int(
            observation.get("matched_current_edge_count") or 0
        ),
        "target_anchored_step_count": int(
            observation.get("target_anchored_step_count") or 0
        ),
        "condition_bound_step_count": len(condition_bound),
        "admission_eligible_step_count": int(
            observation.get("admission_eligible_step_count") or 0
        ),
        "candidate_step_count": int(observation.get("candidate_step_count") or 0),
    }


def _meaningful_visual_condition(step: Mapping[str, Any]) -> bool:
    condition = dict(step.get("condition_candidate") or {})
    ignored = {
        "schema_version",
        "condition_status",
        "source_type",
        "grants_exact_evidence",
        "source_reference_annotation",
    }
    return any(
        key not in ignored and value not in (None, "", [], {})
        for key, value in condition.items()
    )


def _visual_observation_closes_source_loop(
    observation: Mapping[str, Any],
) -> bool:
    return bool(_visual_quality_projection(observation)["source_loop_closed"])


def rebind_visual_evidence_observation(
    service: Any,
    *,
    request: Mapping[str, Any],
    prior_observation: Mapping[str, Any],
    max_steps: int = 16,
) -> dict[str, Any]:
    """Re-normalize one already-paid visual result against the current graph."""

    source = dict(request.get("source") or {})
    prior = dict(prior_observation or {})
    if not request or not prior:
        return _stage("not_needed", reason="prior_visual_observation_missing")
    if str(prior.get("source_ref") or "") != str(source.get("source_ref") or ""):
        return _stage("not_needed", reason="prior_visual_source_mismatch")
    current_artifact = str(source.get("source_artifact_sha256") or "")
    prior_artifact = str(prior.get("source_artifact_sha256") or "")
    if current_artifact and prior_artifact and current_artifact != prior_artifact:
        return _stage("not_needed", reason="prior_visual_artifact_mismatch")

    raw_steps = []
    for value in prior.get("candidate_steps") or []:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        raw_steps.append(
            {
                "product_smiles": str(row.get("product_smiles") or ""),
                "reactant_smiles": [
                    *list(row.get("precursor_smiles") or []),
                    *list(row.get("spectator_reactant_smiles") or []),
                ],
                "product_label": str(row.get("product_label") or ""),
                "reactant_labels": [
                    *list(row.get("reactant_labels") or []),
                    *list(row.get("spectator_reactant_labels") or []),
                ],
                "source_locator": str(row.get("source_locator") or ""),
                "condition_candidate": dict(row.get("condition_candidate") or {}),
                "structure_derivation": dict(row.get("structure_derivation") or {}),
                "stereochemistry_status": str(
                    row.get("stereochemistry_status") or ""
                ),
                "not_exact_literature_segment": bool(
                    row.get("not_exact_literature_segment")
                ),
                "risk_flags": list(row.get("risk_flags") or []),
            }
        )
    if not raw_steps:
        return _stage("not_needed", reason="prior_visual_candidate_steps_missing")
    observation = _normalize_visual_observation(
        request,
        result={
            "request_sha256": str(request.get("content_sha256") or ""),
            "provider_status": str(prior.get("provider_status") or "completed"),
            "provider_receipt": dict(prior.get("provider_receipt") or {}),
            "candidate_chain": {"steps": raw_steps},
        },
        max_steps=max_steps,
    )
    ref = service.kernel.artifacts.put_json(
        observation,
        logical_name="visual_source_candidate_observation_rebound.json",
        producer="autoplanner.visual_evidence.rebind",
    )
    return _stage(
        "reused",
        reason="prior_visual_observation_rebound_without_model_call",
        request=dict(request),
        observation=observation,
        observation_ref=ref.to_dict(),
        model_usage={
            "model_invocations": 0,
            "visual_invocations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_time_s": 0.0,
        },
        material_events=(
            ["visual_source_candidates_reused"]
            if observation["candidate_step_count"]
            else []
        ),
    )


__all__ = [
    "CodexVisualEvidenceConfig",
    "VISUAL_EVIDENCE_OBSERVATION_SCHEMA",
    "VISUAL_EVIDENCE_REQUEST_SCHEMA",
    "VisualEvidenceError",
    "VisualEvidenceProvider",
    "acquire_visual_evidence_candidates",
    "build_codex_visual_evidence_provider",
    "compile_visual_evidence_request",
    "materialize_visual_evidence_candidates",
    "rebind_visual_evidence_observation",
]
