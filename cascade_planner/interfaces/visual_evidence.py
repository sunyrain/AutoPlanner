"""Sparse visual source candidates with explicit RunKernel cost accounting."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from cascade_planner.application.run_kernel import RunKernelError
from cascade_planner.application.retrosynthesis_workers import (
    materialization_commands_for_proposals,
)
from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
    normalize_source_conditions,
)
from cascade_planner.harness.visual_literature_chain_agent import (
    run_visual_literature_chain_agent,
)
from cascade_planner.interfaces.visual_evidence_contract import (
    VISUAL_EVIDENCE_REQUEST_SCHEMA,
    VisualEvidenceError,
    digest as _digest,
    is_sha256 as _is_sha256,
    materialization_stage as _materialization_stage,
    normalized_usage as _normalized_usage,
    sha256 as _sha256,
    source_kind as _source_kind,
    source_ref as _source_ref,
    stage as _stage,
    validate_request_digest as _validate_request_digest,
)
from cascade_planner.interfaces.visual_observation_normalization import (
    VISUAL_EVIDENCE_OBSERVATION_SCHEMA as VISUAL_EVIDENCE_OBSERVATION_SCHEMA,
    normalize_visual_observation as _normalize_visual_observation,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate

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


def compile_visual_evidence_request(
    *,
    evidence_request: Mapping[str, Any],
    discovery: Mapping[str, Any],
    max_pages: int,
) -> dict[str, Any]:
    request, _diagnostics = _compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=max_pages,
    )
    return request


def _compile_visual_evidence_request(
    *,
    evidence_request: Mapping[str, Any],
    discovery: Mapping[str, Any],
    max_pages: int,
    excluded_source_refs: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 1 <= max_pages <= 12:
        raise ValueError("visual_evidence_page_limit_invalid")
    if str(discovery.get("request_sha256") or "") != str(
        evidence_request.get("content_sha256") or ""
    ):
        return {}, [
            {
                "source_ref": "",
                "status": "rejected",
                "reasons": ["evidence_discovery_request_digest_mismatch"],
            }
        ]
    candidates = []
    excluded = {str(value) for value in (excluded_source_refs or set()) if str(value)}
    diagnostics: list[dict[str, Any]] = []
    for source in discovery.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        publication_number = str(source.get("publication_number") or "").strip()
        source_kind = str(source.get("source_kind") or "").strip().lower()
        source_ref = _source_ref(source)
        diagnostic: dict[str, Any] = {
            "source_ref": source_ref,
            "source_kind": source_kind or _source_kind(source_ref),
            "status": "rejected",
            "reasons": [],
            "declared_page_count": len(source.get("visual_candidate_pages") or []),
            "valid_page_count": 0,
        }
        if source_ref in excluded:
            diagnostic["reasons"] = ["excluded_after_visual_source_attempt"]
            diagnostics.append(diagnostic)
            continue
        source_pdf_sha256 = str(
            source.get("source_pdf_sha256") or source.get("pdf_sha256") or ""
        ).strip().lower()
        source_fulltext_sha256 = str(
            source.get("source_fulltext_sha256")
            or source.get("fulltext_xml_sha256")
            or ""
        ).strip().lower()
        source_artifact_sha256 = source_fulltext_sha256 or source_pdf_sha256
        if not source_ref or not _is_sha256(source_artifact_sha256):
            diagnostic["reasons"] = ["hash_bound_source_artifact_missing"]
            diagnostics.append(diagnostic)
            continue
        exact_row_count = int(source.get("exact_row_count") or 0)
        unresolved_edge_count = int(source.get("unresolved_edge_count") or 0)
        if unresolved_edge_count <= 0 and exact_row_count > 0:
            diagnostic["reasons"] = ["source_exact_rows_already_close_frontier"]
            diagnostics.append(diagnostic)
            continue
        target_relevance = _visual_source_target_relevance(
            source,
            evidence_request=evidence_request,
        )
        if target_relevance["accepted"] is not True:
            diagnostic["reasons"] = list(target_relevance.get("reasons") or [])
            diagnostics.append(diagnostic)
            continue
        pages = []
        for page in source.get("visual_candidate_pages") or []:
            if not isinstance(page, Mapping):
                continue
            row = dict(page)
            path = Path(str(row.get("image_path") or "")).expanduser().resolve()
            digest = str(row.get("image_sha256") or "")
            page_number = int(row.get("page_number") or 0)
            if (
                page_number <= 0
                or not path.is_file()
                or not _is_sha256(digest)
                or _sha256(path) != digest
            ):
                continue
            pages.append(
                {
                    "page_number": page_number,
                    "image_path": str(path),
                    "image_sha256": digest,
                }
            )
            if len(pages) >= max_pages:
                break
        diagnostic["valid_page_count"] = len(pages)
        if not pages:
            diagnostic["reasons"] = ["valid_visual_candidate_pages_missing"]
            diagnostics.append(diagnostic)
            continue
        labels = [
            str(row.get("label") or "")
            for row in source.get("procedure_inventory") or []
            if isinstance(row, Mapping)
            and row.get("visual_expected") is not False
            and str(row.get("label") or "").strip()
        ]
        selected_page_numbers = {
            int(row.get("page_number") or 0) for row in pages
        }
        text_snippets = []
        for raw in source.get("procedure_inventory") or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            page_number = int(row.get("page_number") or 0)
            excerpt = " ".join(
                str(
                    row.get("procedure_excerpt")
                    or row.get("procedure")
                    or row.get("text")
                    or ""
                ).split()
            )
            if (
                not excerpt
                or (selected_page_numbers and page_number not in selected_page_numbers)
            ):
                continue
            text_snippets.append(
                {
                    "compound_label": str(row.get("label") or row.get("name") or "")[:200],
                    "page_number": page_number,
                    "snippet": excerpt[:1_200],
                }
            )
            if len(text_snippets) >= 12:
                break
        route_rows = [
            dict(row)
            for row in dict(source.get("source_route_observation") or {}).get(
                "proposals"
            )
            or []
            if isinstance(row, Mapping)
        ]
        route_sequence_hint = " -> ".join(
            str(
                row.get("product_name")
                or dict(row.get("source_location") or {}).get("label")
                or row.get("proposal_id")
                or ""
            )[:160]
            for row in route_rows[:16]
            if str(
                row.get("product_name")
                or dict(row.get("source_location") or {}).get("label")
                or row.get("proposal_id")
                or ""
            ).strip()
        )
        candidates.append(
            {
                "source_ref": source_ref,
                "source_kind": source_kind or _source_kind(source_ref),
                "publication_number": publication_number,
                "doi": str(source.get("doi") or "")[:500],
                "pmid": str(source.get("pmid") or "")[:100],
                "family_id": str(source.get("family_id") or ""),
                "title": str(source.get("title") or "")[:1000],
                "source_pdf_sha256": source_pdf_sha256,
                "source_fulltext_sha256": source_fulltext_sha256,
                "source_artifact_sha256": source_artifact_sha256,
                "source_artifact_kind": (
                    "europe_pmc_fulltext_xml"
                    if source_fulltext_sha256
                    else "pdf"
                ),
                "expected_labels": list(dict.fromkeys(labels))[:24],
                "text_snippets": text_snippets,
                "route_sequence_hint": route_sequence_hint[:2_000],
                "pages": pages,
                "exact_row_count": exact_row_count,
                "unresolved_edge_count": unresolved_edge_count,
                "source_route_proposal_count": int(
                    source.get("source_route_proposal_count") or len(route_rows)
                ),
                "procedure_count": len(source.get("procedure_inventory") or []),
                "target_relevance": target_relevance,
                "target_relevance_priority": int(
                    target_relevance.get("priority") or 0
                ),
            }
        )
        diagnostic["status"] = "eligible"
        diagnostic["reasons"] = list(target_relevance.get("reasons") or [])
        diagnostics.append(diagnostic)
    if not candidates:
        return {}, diagnostics
    candidates.sort(
        key=lambda row: (
            -int(row["target_relevance_priority"]),
            -int(row["source_route_proposal_count"]),
            -int(row["procedure_count"]),
            -int(row["unresolved_edge_count"]),
            int(row["exact_row_count"] > 0),
            str(row["source_ref"]),
        )
    )
    request = {
        "schema_version": VISUAL_EVIDENCE_REQUEST_SCHEMA,
        "evidence_request_sha256": str(evidence_request.get("content_sha256") or ""),
        "run_id": str(evidence_request.get("run_id") or ""),
        "target_name": str(evidence_request.get("target_name") or ""),
        "target_smiles": str(evidence_request.get("target_smiles") or ""),
        "target_identity": dict(evidence_request.get("target_identity") or {}),
        "edges": [dict(row) for row in evidence_request.get("edges") or []],
        "source": candidates[0],
        "selection_diagnostics": diagnostics,
        "limits": {"max_pages": max_pages, "max_model_invocations": 1},
        "semantics": {
            "visual_output_is_hypothesis_only": True,
            "visual_output_cannot_grant_L2_L3_or_stock": True,
            "host_smiles_and_reaction_normalization_required": True,
        },
    }
    request["content_sha256"] = _digest(request)
    return request, diagnostics


def _visual_source_target_relevance(
    source: Mapping[str, Any],
    *,
    evidence_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank strong bridges first while allowing one bounded structure-first pass.

    Search results mentioning a therapeutic class are common patent noise.
    They may remain frozen source observations, but vision is reserved for a
    named/structural bridge, reaction-bearing procedure context, or a
    materialized target-ranked paper whose missing bridge is exactly what
    visual structure extraction is expected to recover.
    """

    target_name = " ".join(
        str(evidence_request.get("target_name") or "").split()
    ).casefold()
    generic_name = (
        not target_name
        or target_name in {"target", "blind target"}
        or "blind" in target_name
        or bool(re.fullmatch(r"target-[0-9a-f]{8,64}", target_name))
    )
    identity = dict(evidence_request.get("target_identity") or {})
    name_terms = {
        target_name,
        " ".join(str(identity.get("preferred_name") or "").split()).casefold(),
        *(
            " ".join(str(value).split()).casefold()
            for value in identity.get("synonyms") or []
        ),
    } - {""}
    searchable = " ".join(
        [
            str(source.get("title") or ""),
            *[
                str(item.get("name") or item.get("label") or "")
                for item in source.get("procedure_inventory") or []
                if isinstance(item, Mapping)
            ],
        ]
    ).casefold()
    named_match = any(term in searchable for term in name_terms)
    family_terms = _target_chemical_family_terms(name_terms)
    matched_family_terms = sorted(
        term for term in family_terms if term in searchable
    )
    family_match = bool(matched_family_terms)
    target_alias_pdf_match = bool(
        int(source.get("target_alias_hit_page_count") or 0) > 0
        or int(dict(source.get("target_focus") or {}).get("target_alias_hit_page_count") or 0)
        > 0
    )
    exact_match = bool(
        int(source.get("exact_row_count") or 0)
        or source.get("exact_edge_ids")
        or int(source.get("source_route_exact_row_count") or 0)
    )
    target_smiles = str(evidence_request.get("target_smiles") or "")
    frontier_products = {
        target_smiles,
        *(
            str(edge.get("product_smiles") or "")
            for edge in evidence_request.get("edges") or []
            if isinstance(edge, Mapping)
        ),
    } - {""}
    proposals = [
        dict(value)
        for value in dict(source.get("source_route_observation") or {}).get(
            "proposals"
        )
        or []
        if isinstance(value, Mapping)
    ]
    connected_route = any(
        str(proposal.get("product_smiles") or "") in frontier_products
        or str(proposal.get("root_anchor") or "").strip()
        for proposal in proposals
    )
    procedure_rows = [
        dict(item)
        for item in source.get("procedure_inventory") or []
        if isinstance(item, Mapping)
    ]
    procedure_context = any(
        len(excerpt) >= 60
        and sum(
            signal in excerpt.casefold()
            for signal in (
                " was added",
                " were added",
                "stirred",
                "reaction mixture",
                "afforded",
                "yield",
                "purified",
                "synthesis",
            )
        )
        >= 2
        for excerpt in (
            " ".join(
                str(
                    item.get("procedure_excerpt")
                    or item.get("procedure")
                    or item.get("text")
                    or ""
                ).split()
            )
            for item in procedure_rows
        )
    )
    materialized_unbound_paper = bool(
        str(source.get("acquisition_status") or "").lower() == "materialized"
        and _source_kind(_source_ref(source)) in {"paper_si", "doi", "pmid", "pmc"}
        and source.get("visual_candidate_pages")
        and int(source.get("unresolved_edge_count") or 0) > 0
    )
    accepted = bool(
        generic_name
        or named_match
        or target_alias_pdf_match
        or family_match
        or exact_match
        or connected_route
        or procedure_context
        or materialized_unbound_paper
    )
    reasons = [
        reason
        for condition, reason in (
            (generic_name, "generic_target_name_cannot_support_text_filter"),
            (named_match, "named_target_mentioned_in_source"),
            (
                target_alias_pdf_match,
                "target_identity_alias_mentioned_in_native_pdf_text",
            ),
            (family_match, "target_chemical_family_mentioned_in_source"),
            (exact_match, "source_matches_current_exact_edge"),
            (connected_route, "source_route_connects_to_target_frontier"),
            (procedure_context, "reaction_procedure_context_requires_visual_binding"),
            (
                materialized_unbound_paper,
                "materialized_target_ranked_paper_requires_visual_structure_binding",
            ),
        )
        if condition
    ]
    if not accepted:
        reasons.append("source_has_no_target_or_frontier_bridge")
    return {
        "schema_version": "visual_source_target_relevance.v1",
        "accepted": accepted,
        "priority": (
            100
            if exact_match or connected_route
            else 95
            if target_alias_pdf_match
            else 90
            if named_match
            else 80
            if family_match
            else 60
            if procedure_context
            else 30
            if materialized_unbound_paper
            else 10
            if generic_name
            else 0
        ),
        "reasons": reasons,
        "matched_family_terms": matched_family_terms,
        "semantics": {
            "search_result_presence_is_not_relevance": True,
            "rejected_source_bytes_remain_frozen_for_audit": True,
            "bounded_structure_first_visual_pass_breaks_binding_deadlock": True,
        },
    }


def _target_chemical_family_terms(name_terms: set[str]) -> set[str]:
    """Derive conservative suffix roots from structure-resolved target names.

    Chemical identifiers frequently prepend substituents to the actual family
    name (for example, ``pentamethylenefulvene``).  Exact phrase matching then
    misses a paper headed simply "Fulvenes".  Long suffix roots recover that
    relationship without treating generic words such as "target" as chemistry.
    """

    ignored = {
        "compound",
        "research",
        "target",
        "synthesis",
        "preparation",
        "product",
        "unknown",
    }
    roots: set[str] = set()
    for phrase in name_terms:
        for token in re.findall(r"[a-z][a-z0-9]{5,}", phrase.casefold()):
            if token in ignored or token.isdigit():
                continue
            for width in range(7, min(14, len(token)) + 1):
                suffix = token[-width:]
                if suffix not in ignored:
                    roots.add(suffix)
            if len(token) <= 18:
                roots.add(token)
    return roots


def _visual_no_candidate_reason(
    diagnostics: list[dict[str, Any]],
) -> str:
    if not diagnostics:
        return "visual_candidate_sources_missing"
    reasons = {
        str(reason)
        for row in diagnostics
        for reason in row.get("reasons") or []
        if str(reason)
    }
    if "evidence_discovery_request_digest_mismatch" in reasons:
        return "visual_discovery_request_mismatch"
    if any(int(row.get("declared_page_count") or 0) > 0 for row in diagnostics):
        if "source_has_no_target_or_frontier_bridge" in reasons:
            return "visual_candidate_sources_rejected"
        return "visual_candidate_pages_invalid_or_filtered"
    if "hash_bound_source_artifact_missing" in reasons:
        return "visual_source_artifacts_missing"
    return "visual_candidate_pages_missing"


def materialize_visual_evidence_candidates(
    service: Any,
    *,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit a visually extracted literature chain as L0/L1 proposals.

    Page-bound visual extraction is useful chemistry generation, but it is not
    deterministic source proof.  Every step therefore goes through the same
    host materialization gates as ChemEnzy/Codex and remains below L3 until an
    independent structured extractor or curator supplies exact rows.
    """

    steps = [
        dict(row)
        for row in observation.get("candidate_steps") or []
        if isinstance(row, Mapping) and row.get("admission_eligible") is True
    ]
    if not steps:
        return _materialization_stage("not_needed", reason="visual_candidate_steps_missing")
    if not (
        int(observation.get("matched_current_edge_count") or 0)
        or int(observation.get("frontier_anchored_step_count") or 0)
        or int(observation.get("target_anchored_step_count") or 0)
    ):
        return _materialization_stage(
            "not_needed",
            reason="visual_chain_not_connected_to_canonical_target_or_frontier",
            proposal_count=len(steps),
            observation_ref=str(observation.get("content_sha256") or ""),
            semantics={
                "disconnected_visual_chain_retained_as_source_observation": True,
                "disconnected_visual_chain_not_added_to_target_graph": True,
                "visual_chain_grants_exact_evidence": False,
            },
        )
    source_ref = str(observation.get("source_ref") or "")
    graph = service.graph_store.load()
    existing_visual_origins = {
        (
            str(edge.get("edge_digest") or ""),
            str(origin.get("origin_ref") or ""),
            str(origin.get("proposal_id") or ""),
        )
        for edge in dict(graph.get("edges") or {}).values()
        if isinstance(edge, Mapping)
        for origin in edge.get("origin_records") or []
        if isinstance(origin, Mapping)
        and str(origin.get("origin_kind") or "")
        == "literature_visual_extraction"
    }
    proposals = []
    for row in steps:
        condition = dict(row.get("condition_candidate") or {})
        normalized_conditions = normalize_source_conditions(condition)
        edge_digest = str(
            audit_retrosynthetic_candidate(
                row.get("product_smiles"),
                row.get("precursor_smiles") or [],
            ).get("edge_digest")
            or ""
        )
        proposal_id = str(row.get("candidate_id") or "")
        if (edge_digest, source_ref, proposal_id) in existing_visual_origins:
            continue
        proposals.append(
            {
                "product_smiles": str(row.get("product_smiles") or ""),
                "precursor_smiles": list(row.get("precursor_smiles") or []),
                "reagent_smiles": list(
                    row.get("spectator_reactant_smiles") or []
                ),
                "origin_kind": "literature_visual_extraction",
                "origin_ref": source_ref,
                "proposal_id": proposal_id,
                "transformation_hypothesis": (
                    "page-bound literature structure-chain extraction"
                ),
                "condition_predictions": (
                    [
                        {
                            **condition,
                            "conditions": normalized_conditions,
                            "condition_completeness": audit_condition_completeness(
                                normalized_conditions
                            ),
                            "source_ref": source_ref,
                            "source_locator": str(row.get("source_locator") or ""),
                            "authority_scope": "model_extracted_source_condition_candidate",
                            "not_reaction_proof": True,
                            "exact_structure_binding_candidate": bool(
                                row.get("exact_structure_binding_candidate")
                            ),
                            "matched_current_edge_id": str(
                                row.get("matched_current_edge_id") or ""
                            ),
                        }
                    ]
                    if condition
                    else []
                ),
            }
        )
    if not proposals:
        return _materialization_stage(
            "reused_or_empty",
            reason="visual_source_binding_already_materialized",
            proposal_count=0,
            observation_step_count=len(steps),
            exact_structure_binding_candidate_count=sum(
                bool(row.get("exact_structure_binding_candidate")) for row in steps
            ),
            matched_current_edge_ids=sorted(
                {
                    str(row.get("matched_current_edge_id") or "")
                    for row in steps
                    if str(row.get("matched_current_edge_id") or "")
                }
            ),
        )
    revision = service.kernel.revision
    commands = materialization_commands_for_proposals(
        proposals,
        run_id=service.kernel.spec.run_id,
        input_revision=revision.graph_revision,
        dependency_revisions={
            "graph_revision": revision.graph_revision,
            "evidence_revision": revision.evidence_revision,
        },
        # Re-run an already-known identity through the canonical worker so
        # its literature origin and page-bound condition are merged onto the
        # existing edge.  Passing the digest as an exclusion would silently
        # discard the newly acquired source binding.
        existing_edge_digests=(),
    )
    if not commands:
        return _materialization_stage(
            "reused_or_empty",
            reason="visual_chain_already_materialized_or_ineligible",
            proposal_count=len(proposals),
        )
    execution = service.execute_commands(
        commands,
        idempotency_key=f"visual-chain:{str(observation.get('content_sha256') or '')}",
    )
    return _materialization_stage(
        "completed" if execution.get("changed") else "partial",
        proposal_count=len(proposals),
        exact_structure_binding_candidate_count=sum(
            bool(row.get("exact_structure_binding_candidate")) for row in steps
        ),
        matched_current_edge_ids=sorted(
            {
                str(row.get("matched_current_edge_id") or "")
                for row in steps
                if str(row.get("matched_current_edge_id") or "")
            }
        ),
        command_count=len(commands),
        execution=execution,
        material_events=(
            ["visual_literature_chain_materialized"]
            if execution.get("changed")
            else []
        ),
        semantics={
            "visual_chain_enters_canonical_hypergraph": True,
            "visual_chain_grants_exact_evidence": False,
            "host_validation_still_required": True,
        },
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
