"""Sparse visual source candidates with explicit RunKernel cost accounting."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from rdkit import Chem

from cascade_planner.application.run_kernel import RunKernelError
from cascade_planner.harness.reaction_step_verifier import canonical_reaction_digest
from cascade_planner.harness.visual_literature_chain_agent import (
    run_visual_literature_chain_agent,
)


VISUAL_EVIDENCE_REQUEST_SCHEMA = "visual_source_candidate_request.v1"
VISUAL_EVIDENCE_OBSERVATION_SCHEMA = "visual_source_candidate_observation.v1"
VisualEvidenceProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class VisualEvidenceError(RuntimeError):
    """A visual provider or its output violated the bounded draft contract."""


@dataclass(frozen=True, slots=True)
class CodexVisualEvidenceConfig:
    cache_dir: str | Path
    model: str = "gpt-5.5"
    reasoning_effort: str = "low"
    timeout_s: float = 240.0
    max_pages: int = 4
    max_steps: int = 16

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("visual_evidence_model_missing")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("visual_evidence_reasoning_effort_invalid")
        if self.timeout_s <= 0 or not 1 <= self.max_pages <= 8:
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
                route_sequence_hint="",
                text_snippets=[],
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
    max_pages: int = 4,
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
    if not 1 <= max_pages <= 8:
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
        source_pdf_sha256 = str(source.get("pdf_sha256") or "").strip().lower()
        if not publication_number or not _is_sha256(source_pdf_sha256):
            continue
        exact_row_count = int(source.get("exact_row_count") or 0)
        unresolved_edge_count = int(source.get("unresolved_edge_count") or 0)
        if unresolved_edge_count <= 0 and exact_row_count > 0:
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
            if isinstance(row, Mapping) and str(row.get("label") or "").strip()
        ]
        candidates.append(
            {
                "source_ref": f"patent:{publication_number}",
                "publication_number": publication_number,
                "family_id": str(source.get("family_id") or ""),
                "title": str(source.get("title") or "")[:1000],
                "source_pdf_sha256": source_pdf_sha256,
                "expected_labels": list(dict.fromkeys(labels))[:24],
                "pages": pages,
                "exact_row_count": exact_row_count,
                "unresolved_edge_count": unresolved_edge_count,
            }
        )
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            -int(row["unresolved_edge_count"]),
            int(row["exact_row_count"] > 0),
            str(row["publication_number"]),
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


def _normalize_visual_observation(
    request: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    if str(result.get("request_sha256") or "") != str(request.get("content_sha256") or ""):
        raise VisualEvidenceError("visual_provider_request_digest_mismatch")
    source = dict(request.get("source") or {})
    current_edges = {
        canonical_reaction_digest(
            _canonical_smiles(row.get("product_smiles")),
            _canonical_reactants(row.get("precursor_smiles")),
        ): str(row.get("edge_id") or "")
        for row in request.get("edges") or []
        if isinstance(row, Mapping)
        and _canonical_smiles(row.get("product_smiles"))
        and _canonical_reactants(row.get("precursor_smiles"))
    }
    chain = dict(result.get("candidate_chain") or {})
    steps = []
    for index, raw in enumerate(chain.get("steps") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        product = _canonical_smiles(row.get("product_smiles"))
        reactants = _canonical_reactants(row.get("reactant_smiles"))
        if not product or not reactants:
            continue
        reaction_digest = canonical_reaction_digest(product, reactants)
        steps.append(
            {
                "candidate_id": f"visual:{_digest({'source': source.get('source_ref'), 'reaction': reaction_digest})[:24]}",
                "product_smiles": product,
                "precursor_smiles": reactants,
                "product_label": str(row.get("product_label") or "")[:300],
                "reactant_labels": [
                    str(value)[:300]
                    for value in row.get("reactant_labels") or []
                    if str(value).strip()
                ][:12],
                "source_locator": str(row.get("source_locator") or "")[:500],
                "reaction_digest": reaction_digest,
                "matched_current_edge_id": current_edges.get(reaction_digest, ""),
                "relation_type": "visual_candidate",
                "allowed_use": "global_replan_hypothesis_only",
                "host_smiles_parse_accepted": True,
                "grants_exact_evidence": False,
            }
        )
        if len(steps) >= max_steps:
            break
    observation = {
        "schema_version": VISUAL_EVIDENCE_OBSERVATION_SCHEMA,
        "request_sha256": str(request.get("content_sha256") or ""),
        "source_ref": str(source.get("source_ref") or ""),
        "source_pdf_sha256": str(source.get("source_pdf_sha256") or ""),
        "page_bindings": [dict(row) for row in source.get("pages") or []],
        "provider_receipt": dict(result.get("provider_receipt") or {}),
        "provider_status": str(result.get("provider_status") or ""),
        "candidate_steps": steps,
        "candidate_step_count": len(steps),
        "matched_current_edge_count": sum(
            bool(row["matched_current_edge_id"]) for row in steps
        ),
        "semantics": {
            "model_output_is_advisory": True,
            "host_canonicalization_is_not_source_verification": True,
            "deterministic_source_parser_must_independently_reconstruct_exact_rows": True,
            "observation_cannot_grant_L2_L3_or_stock": True,
        },
    }
    observation["content_sha256"] = _digest(observation)
    return observation


def _validate_request_digest(request: Mapping[str, Any]) -> None:
    body = {key: value for key, value in request.items() if key != "content_sha256"}
    if (
        request.get("schema_version") != VISUAL_EVIDENCE_REQUEST_SCHEMA
        or str(request.get("content_sha256") or "") != _digest(body)
    ):
        raise VisualEvidenceError("visual_evidence_request_invalid")


def _normalized_usage(value: Any) -> dict[str, Any]:
    row = dict(value) if isinstance(value, Mapping) else {}
    invocations = max(0, int(row.get("model_invocations") or 0))
    visual = max(0, int(row.get("visual_invocations") or 0))
    if invocations > 1 or visual > 1 or visual > invocations:
        raise VisualEvidenceError("visual_provider_usage_invalid")
    return {
        "model_invocations": invocations,
        "visual_invocations": visual,
        "input_tokens": max(0, int(row.get("input_tokens") or 0)),
        "output_tokens": max(0, int(row.get("output_tokens") or 0)),
        "wall_time_s": max(0.0, float(row.get("wall_time_s") or 0.0)),
    }


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else ""


def _canonical_reactants(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    result = sorted(_canonical_smiles(row) for row in values)
    return result if result and all(result) else []


def _stage(status: str, *, reason: str = "", **values: Any) -> dict[str, Any]:
    return {
        "stage": "visual_evidence",
        "status": status,
        "reason": str(reason),
        "model_invocations": int(
            dict(values.get("model_usage") or {}).get("model_invocations") or 0
        ),
        "visual_invocations": int(
            dict(values.get("model_usage") or {}).get("visual_invocations") or 0
        ),
        **values,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


__all__ = [
    "CodexVisualEvidenceConfig",
    "VISUAL_EVIDENCE_OBSERVATION_SCHEMA",
    "VISUAL_EVIDENCE_REQUEST_SCHEMA",
    "VisualEvidenceError",
    "VisualEvidenceProvider",
    "acquire_visual_evidence_candidates",
    "build_codex_visual_evidence_provider",
    "compile_visual_evidence_request",
]
