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
) -> dict[str, Any]:
    """Run at most one visual task and return only host-normalized hypotheses."""

    if provider is None:
        return _stage("disabled", reason="visual_evidence_provider_not_configured")
    request = compile_visual_evidence_request(
        evidence_request=evidence_request,
        discovery=discovery,
        max_pages=max_pages,
    )
    if not request:
        return _stage("not_needed", reason="visual_candidate_pages_missing")
    task_id = f"visual-evidence:{str(request['content_sha256'])[:24]}"
    prompt_bytes = len(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    usage: dict[str, Any] = {}
    provider_called = False
    if service.kernel.count_task_reservations(
        kind="model",
        metadata={"visual_evidence": True},
    ):
        return _stage(
            "budget_blocked",
            reason="campaign_visual_evidence_call_already_admitted",
            request=request,
        )
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


def compile_visual_evidence_request(
    *,
    evidence_request: Mapping[str, Any],
    discovery: Mapping[str, Any],
    max_pages: int,
) -> dict[str, Any]:
    if not 1 <= max_pages <= 12:
        raise ValueError("visual_evidence_page_limit_invalid")
    if str(discovery.get("request_sha256") or "") != str(
        evidence_request.get("content_sha256") or ""
    ):
        return {}
    candidates = []
    for source in discovery.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        publication_number = str(source.get("publication_number") or "").strip()
        source_kind = str(source.get("source_kind") or "").strip().lower()
        source_ref = _source_ref(source)
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
            continue
        exact_row_count = int(source.get("exact_row_count") or 0)
        unresolved_edge_count = int(source.get("unresolved_edge_count") or 0)
        if unresolved_edge_count <= 0 and exact_row_count > 0:
            continue
        target_relevance = _visual_source_target_relevance(
            source,
            evidence_request=evidence_request,
        )
        if target_relevance["accepted"] is not True:
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
        if not pages:
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
            }
        )
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
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
        "edges": [dict(row) for row in evidence_request.get("edges") or []],
        "source": candidates[0],
        "limits": {"max_pages": max_pages, "max_model_invocations": 1},
        "semantics": {
            "visual_output_is_hypothesis_only": True,
            "visual_output_cannot_grant_L2_L3_or_stock": True,
            "host_smiles_and_reaction_normalization_required": True,
        },
    }
    request["content_sha256"] = _digest(request)
    return request


def _visual_source_target_relevance(
    source: Mapping[str, Any],
    *,
    evidence_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a source-to-target bridge before spending a visual call.

    Search results mentioning a therapeutic class are common patent noise.
    They may remain frozen source observations, but vision is reserved for a
    named-target match, an exact current edge, or a source-route proposal that
    is structurally connected to the target/frontier.
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
    accepted = bool(generic_name or named_match or exact_match or connected_route)
    reasons = [
        reason
        for condition, reason in (
            (generic_name, "generic_target_name_cannot_support_text_filter"),
            (named_match, "named_target_mentioned_in_source"),
            (exact_match, "source_matches_current_exact_edge"),
            (connected_route, "source_route_connects_to_target_frontier"),
        )
        if condition
    ]
    if not accepted:
        reasons.append("source_has_no_target_or_frontier_bridge")
    return {
        "schema_version": "visual_source_target_relevance.v1",
        "accepted": accepted,
        "reasons": reasons,
        "semantics": {
            "search_result_presence_is_not_relevance": True,
            "rejected_source_bytes_remain_frozen_for_audit": True,
        },
    }


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
    graph = service.graph_store.load()
    existing = [
        str(row.get("edge_digest") or "")
        for row in dict(graph.get("edges") or {}).values()
        if isinstance(row, Mapping) and str(row.get("edge_digest") or "")
    ]
    source_ref = str(observation.get("source_ref") or "")
    proposals = []
    for row in steps:
        condition = dict(row.get("condition_candidate") or {})
        proposals.append(
            {
                "product_smiles": str(row.get("product_smiles") or ""),
                "precursor_smiles": list(row.get("precursor_smiles") or []),
                "origin_kind": "literature_visual_extraction",
                "origin_ref": source_ref,
                "proposal_id": str(row.get("candidate_id") or ""),
                "transformation_hypothesis": (
                    "page-bound literature structure-chain extraction"
                ),
                "condition_predictions": (
                    [
                        {
                            **condition,
                            "source_ref": source_ref,
                            "source_locator": str(row.get("source_locator") or ""),
                            "authority_scope": "model_extracted_source_condition_candidate",
                            "not_reaction_proof": True,
                        }
                    ]
                    if condition
                    else []
                ),
            }
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
        existing_edge_digests=existing,
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
]
