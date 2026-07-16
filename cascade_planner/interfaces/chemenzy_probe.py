"""Bounded ChemEnzy proposal ingestion for the target-only V4 campaign."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

from cascade_planner.application.canonical_hypergraph import CanonicalIngestionBatch
from cascade_planner.application.route_innovation_chemenzy import (
    route_innovation_from_chemenzy_step,
)
from cascade_planner.baselines.chem_enzy_adapter import (
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.interfaces.chemenzy_guidance import (
    guided_native_search_policy as _guided_native_search_policy,
)
from cascade_planner.interfaces.chemenzy_runtime_selection import (
    select_chemenzy_runtime as _select_runtime,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)
from cascade_planner.providers.builtins import (
    ChemEnzyProposalProvider as RegisteredChemEnzyProposalProvider,
    build_default_provider_registry,
)
from cascade_planner.providers.contracts import ProviderContext
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


ChemenzyProposalProvider = Callable[..., Mapping[str, Any]]
CHEMENZY_PROVIDER_CAPABILITY_SCHEMA = "provider_capability_snapshot.v1"


@dataclass(frozen=True, slots=True)
class ChemEnzyProposalRequest:
    """One seed or frontier-guided request; never a second search queue."""

    target_smiles: str
    target_name: str = ""
    mode: str = "seed"
    frontier_smiles: tuple[str, ...] = ()
    route_family_ids: tuple[str, ...] = ()
    retron_hints: tuple[str, ...] = ()
    forbidden_smiles: tuple[str, ...] = ()
    limits: Mapping[str, Any] = field(default_factory=dict)
    stop_conditions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "chemenzy_proposal_request.v2"

    def __post_init__(self) -> None:
        if self.mode not in {"seed", "guided_frontier"}:
            raise ValueError("invalid_chemenzy_proposal_mode")
        if not self.target_smiles.strip():
            raise ValueError("chemenzy_target_smiles_required")
        if self.mode == "guided_frontier" and not self.frontier_smiles:
            raise ValueError("guided_chemenzy_frontier_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "target_name": self.target_name,
            "target_smiles": self.target_smiles,
            "frontier_smiles": list(self.frontier_smiles),
            "route_family_ids": list(self.route_family_ids),
            "retron_hints": list(self.retron_hints),
            "forbidden_smiles": list(self.forbidden_smiles),
            "limits": dict(self.limits),
            "stop_conditions": dict(self.stop_conditions),
            "semantics": {
                "canonical_frontier_is_authoritative": True,
                "provider_has_no_private_expansion_state": True,
                "result_is_proposal_only": True,
            },
        }


def run_chemenzy_proposal_stage(
    service: RetrosynthesisCampaignService,
    *,
    target_name: str,
    target_smiles: str,
    enabled: bool,
    provider: ChemenzyProposalProvider | None = None,
    env_prefix: str | Path | None = None,
    vendor_root: str | Path | None = None,
    max_routes: int = 2,
    max_steps: int = 6,
    max_iterations: int = 10,
    expansion_topk: int = 20,
    timeout_s: float = 90.0,
    mode: str = "seed",
    scope: str = "seed",
    parent_route_family_ids: tuple[str, ...] = (),
    retron_hints: tuple[str, ...] = (),
    forbidden_smiles: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Acquire a small proposal pool and admit it through the canonical graph."""

    limits = {
        "max_routes": max(1, int(max_routes)),
        "max_steps": max(1, int(max_steps)),
        "max_iterations": max(1, int(max_iterations)),
        "expansion_topk": max(1, int(expansion_topk)),
        "timeout_s": max(1.0, float(timeout_s)),
    }
    if not enabled:
        return _result(
            "disabled", mode=mode, scope=scope, limits=limits,
            reason="chemenzy_disabled"
        )
    request = ChemEnzyProposalRequest(
        target_name=target_name,
        target_smiles=target_smiles,
        mode=mode,
        frontier_smiles=(target_smiles,) if mode == "guided_frontier" else (),
        route_family_ids=tuple(parent_route_family_ids),
        retron_hints=tuple(retron_hints),
        forbidden_smiles=tuple(forbidden_smiles),
        limits=limits,
        stop_conditions={
            "max_routes": limits["max_routes"],
            "max_steps": limits["max_steps"],
            "timeout_s": limits["timeout_s"],
        },
    )
    raw = (
        dict(
            provider(
                target_name=target_name,
                target_smiles=target_smiles,
                limits=limits,
                request=request.to_dict(),
            )
        )
        if provider is not None
        else _run_builtin_probe(
            service.kernel.run_dir,
            target_name=target_name,
            target_smiles=target_smiles,
            proposal_request=request,
            scope=scope,
            env_prefix=env_prefix,
            vendor_root=vendor_root,
            limits=limits,
        )
    )
    routes = _normalized_routes(raw, target_smiles=target_smiles)
    # A backend-level ``solved`` flag is neither required nor trusted here.
    # ChemEnzy's current launcher returns proposal routes without that field;
    # older adapters sometimes emitted it.  Candidate admission is owned by
    # the host and is deliberately cheaper/weaker than reaction proof.
    eligible = [
        route for route in routes if route.get("proposal_eligible") is True
    ]
    accepted = eligible[: limits["max_routes"]]
    provider_envelope, provider_registration = _provider_admission(
        service,
        target_name=target_name,
        target_smiles=target_smiles,
        routes=accepted,
        limits=limits,
        request=request,
    )
    if provider_envelope.get("accepted") is not True:
        accepted = []
    route_families: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    for route_index, route in enumerate(accepted, start=1):
        alias = f"chemenzy:{scope}:route:{route_index}"
        if not parent_route_family_ids:
            route_families.append(
                {
                    "route_family_id": alias,
                    "strategy": "bounded ChemEnzy multi-step proposal",
                }
            )
        for step_index, step in enumerate(route.get("steps") or [], start=1):
            if step_index > limits["max_steps"]:
                break
            hypothesis = {
                    "step_id": f"{alias}:step:{step_index}",
                    "proposal_id": f"{alias}:step:{step_index}",
                    "route_family_id": alias,
                    "canonical_route_family_id": (
                        parent_route_family_ids[0]
                        if parent_route_family_ids
                        else ""
                    ),
                    "product_smiles": str(step.get("product_smiles") or ""),
                    "precursor_smiles": list(
                        step.get("reactant_smiles")
                        or step.get("precursor_smiles")
                        or []
                    ),
                    "origin_kind": "chemenzy",
                    "origin_ref": f"{alias}:{step.get('source_model') or 'native'}",
                    "transformation_hypothesis": "ChemEnzy one-step expansion",
                    "condition_predictions": list(
                        step.get("condition_predictions") or []
                    ),
                }
            route_innovation = route_innovation_from_chemenzy_step(
                step,
                route_family_id=alias,
            )
            if route_innovation:
                hypothesis["route_innovation"] = route_innovation
            hypotheses.append(hypothesis)
    applied: Mapping[str, Any] = {"changed": False}
    if hypotheses:
        applied = service.apply_batch(
            CanonicalIngestionBatch(
                route_families=tuple(route_families),
                hypotheses=tuple(hypotheses),
            ),
            idempotency_key=f"solve-target:chemenzy:{scope}:proposal-ingestion",
        )
    status = "completed" if hypotheses else str(raw.get("status") or "unresolved")
    return _result(
        status,
        mode=mode,
        scope=scope,
        limits=limits,
        route_count=len(routes),
        host_admitted_route_count=len(eligible),
        selected_proposal_route_count=len(accepted),
        accepted_route_count=len(accepted),
        rejected_route_count=len(routes) - len(eligible),
        budget_truncated_route_count=max(0, len(eligible) - len(accepted)),
        route_admission=[
            {
                "route_index": route.get("route_index"),
                "proposal_eligible": route.get("proposal_eligible") is True,
                "reasons": list(route.get("admission_reasons") or []),
            }
            for route in routes
        ],
        proposal_count=len(hypotheses),
        changed=applied.get("changed") is True,
        provider_envelope=provider_envelope,
        provider_registration=provider_registration,
        runtime_preflight=raw.get("runtime_preflight") or raw.get("preflight") or {},
        runtime_discovery=raw.get("runtime_discovery") or {},
        provider_capability=_provider_capability_snapshot(raw),
        reason=str(raw.get("reason") or ""),
    )


def run_chemenzy_guided_frontier_stage(
    service: RetrosynthesisCampaignService,
    *,
    target_name: str,
    root_target_smiles: str,
    enabled: bool,
    provider: ChemenzyProposalProvider | None = None,
    env_prefix: str | Path | None = None,
    vendor_root: str | Path | None = None,
    max_frontiers: int = 1,
    max_routes: int = 1,
    max_steps: int = 4,
    max_iterations: int = 4,
    expansion_topk: int = 10,
    timeout_s: float = 60.0,
    exclude_frontier_smiles: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Expand only canonical Codex-selected or stock-rejected subtargets."""

    graph = service.graph_store.load()
    excluded = {str(value).strip() for value in exclude_frontier_smiles if str(value).strip()}
    items = [
        dict(item)
        for item in dict(graph.get("deficit_frontier") or {}).get("items") or []
        if isinstance(item, Mapping)
        and item.get("kind") == "expansion"
        and str(dict(item.get("metadata") or {}).get("frontier_smiles") or "")
        not in excluded
    ][: max(0, int(max_frontiers))]
    if not enabled:
        return {
            "schema_version": "v4_chemenzy_guided_frontier_stage.v1",
            "stage": "chemenzy_guided_frontier",
            "status": "disabled",
            "reason": "chemenzy_disabled",
            "frontier_count": len(items),
            "proposal_count": 0,
        }
    if not items:
        return {
            "schema_version": "v4_chemenzy_guided_frontier_stage.v1",
            "stage": "chemenzy_guided_frontier",
            "status": "not_needed",
            "frontier_count": 0,
            "proposal_count": 0,
        }
    results: list[dict[str, Any]] = []
    for item in items:
        metadata = dict(item.get("metadata") or {})
        frontier_smiles = str(metadata.get("frontier_smiles") or "").strip()
        if not frontier_smiles:
            continue
        route_ids = tuple(
            str(value) for value in item.get("route_family_ids") or [] if str(value)
        )
        retrons = tuple(
            dict.fromkeys(
                [
                    *(
                        str(value)
                        for value in metadata.get("retron_hints") or []
                        if str(value).strip()
                    ),
                    *(
            str(dict(graph.get("route_families") or {}).get(route_id, {}).get("strategy") or "")
            for route_id in route_ids
            if str(dict(graph.get("route_families") or {}).get(route_id, {}).get("strategy") or "")
                    ),
                ]
            )
        )
        digest = hashlib.sha256(frontier_smiles.encode("utf-8")).hexdigest()[:12]
        results.append(
            run_chemenzy_proposal_stage(
                service,
                target_name=f"{target_name} frontier {digest}",
                target_smiles=frontier_smiles,
                enabled=True,
                provider=provider,
                env_prefix=env_prefix,
                vendor_root=vendor_root,
                max_routes=max_routes,
                max_steps=max_steps,
                max_iterations=max_iterations,
                expansion_topk=expansion_topk,
                timeout_s=timeout_s,
                mode="guided_frontier",
                scope=f"guided-{digest}",
                parent_route_family_ids=route_ids,
                retron_hints=retrons,
                forbidden_smiles=(root_target_smiles,),
            )
        )
    proposal_count = sum(int(result.get("proposal_count") or 0) for result in results)
    codex_delegated_count = sum(
        str(item.get("reason") or "")
        == "codex_selected_frontier_requires_local_generation"
        for item in items
    )
    return {
        "schema_version": "v4_chemenzy_guided_frontier_stage.v1",
        "stage": "chemenzy_guided_frontier",
        "status": "completed" if proposal_count else "unresolved",
        "frontier_count": len(items),
        "executed_frontier_count": len(results),
        "provider_invocation_count": len(results),
        "codex_delegated_frontier_count": codex_delegated_count,
        "frontier_smiles": [
            str(dict(item.get("metadata") or {}).get("frontier_smiles") or "")
            for item in items
        ],
        "proposal_count": proposal_count,
        "results": results,
        "material_events": ["guided_provider_proposals_added"] if proposal_count else [],
        "semantics": {
            "canonical_frontier_queue": True,
            "frontier_batch_is_bounded": True,
            "one_bounded_pass": True,
            "provider_result_requires_host_materialization": True,
            "enabled_does_not_imply_invoked": True,
        },
    }


def _provider_admission(
    service: RetrosynthesisCampaignService,
    *,
    target_name: str,
    target_smiles: str,
    routes: list[Mapping[str, Any]],
    limits: Mapping[str, Any],
    request: ChemEnzyProposalRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pass a host-normalized, solved-claim-free batch through the existing SPI."""

    candidate_batch = {
        "schema_version": "retrosynthesis_candidate_batch.v1",
        "accepted": True,
        "target_name": target_name,
        "target_smiles": target_smiles,
        "routes": [
            {
                "route_index": index,
                "steps": [dict(step) for step in route.get("steps") or []],
            }
            for index, route in enumerate(routes, start=1)
        ],
        "limits": dict(limits),
        "semantics": {
            "raw_backend_solved_removed": True,
            "proposal_only": True,
            "grants_no_route_proof": True,
        },
    }
    provider = RegisteredChemEnzyProposalProvider(
        lambda _request, *, context: candidate_batch
    )
    registry = build_default_provider_registry(include_chemenzy=provider)
    result = registry.invoke(
        provider.descriptor.provider_id,
        request.to_dict(),
        context=ProviderContext(
            run_id=service.kernel.spec.run_id,
            case_id=service.kernel.spec.run_id,
            target_smiles=target_smiles,
            budget_remaining=dict(limits),
        ),
    )
    envelope = result.to_dict()
    summary = {
        "schema_version": "provider_admission_summary.v1",
        "provider_id": envelope["provider_id"],
        "provider_version": envelope["provider_version"],
        "provider_kind": envelope["provider_kind"],
        "accepted": envelope["accepted"],
        "reasons": list(envelope.get("reasons") or []),
        "no_solved_claim": envelope["no_solved_claim"],
        "result_content_hash": envelope["content_hash"],
        "normalized_candidate_count": len(routes),
    }
    return summary, {
        "descriptor": registry.descriptor(provider.descriptor.provider_id).to_dict(),
        "trust": registry.trust_record(provider.descriptor.provider_id),
    }


def _run_builtin_probe(
    run_dir: Path,
    *,
    target_name: str,
    target_smiles: str,
    proposal_request: ChemEnzyProposalRequest,
    scope: str,
    env_prefix: str | Path | None,
    vendor_root: str | Path | None,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    preflight, discovery = _select_runtime(
        env_prefix=env_prefix,
        vendor_root=vendor_root,
        timeout_s=min(30.0, float(limits["timeout_s"])),
    )
    if preflight.get("production_ready") is not True:
        return {
            "status": "runtime_unavailable",
            "reason": "chemenzy_runtime_not_production_ready",
            "runtime_preflight": preflight,
            "runtime_discovery": discovery,
            "routes": [],
        }
    request = {
        "target_name": target_name,
        "target_smiles": target_smiles,
        "planner_backend": "chem_enzy_native",
        "search_preset": "bounded_probe",
        "max_steps": limits["max_steps"],
        "chem_enzy_iterations": limits["max_iterations"],
        "chem_enzy_expansion_topk": limits["expansion_topk"],
        "stock_mode": "building-block",
        "device": "cpu",
        "enable_rule_verifier_gate": True,
        "enable_condition_prediction": False,
    }
    if proposal_request.mode == "guided_frontier":
        request["chem_enzy_search_policy"] = _guided_native_search_policy(
            proposal_request,
            limits=limits,
        )
        request["search_preset"] = "thorough"
    artifact_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(scope or "probe"))[:80]
    request_path = run_dir / f"chemenzy-v4-{artifact_stem}-request.json"
    output_path = run_dir / f"chemenzy-v4-{artifact_stem}-result.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    command = [
        str(preflight["python_executable"]),
        str(preflight["launcher_path"]),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
        "--vendor-root",
        str(preflight["vendor_root"]),
        "--gpu",
        "-1",
    ]
    stdout_path = run_dir / f"chemenzy-v4-{artifact_stem}-stdout.log"
    stderr_path = run_dir / f"chemenzy-v4-{artifact_stem}-stderr.log"
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(preflight["launcher_path"]).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=float(limits["timeout_s"]),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "reason": "chemenzy_bounded_probe_timeout",
            "runtime_preflight": preflight,
            "runtime_discovery": discovery,
            "routes": [],
        }
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0 or not output_path.is_file():
        return {
            "status": "failed",
            "reason": f"chemenzy_exit_{completed.returncode}",
            "runtime_preflight": preflight,
            "runtime_discovery": discovery,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "routes": [],
        }
    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "status": "failed",
            "reason": "chemenzy_result_invalid",
            "runtime_preflight": preflight,
            "runtime_discovery": discovery,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "routes": [],
        }
    return {
        **dict(result),
        "runtime_preflight": preflight,
        "runtime_discovery": discovery,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "search_executed": True,
        "request_path": str(request_path),
        "output_path": str(output_path),
    }


def _provider_capability_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    preflight = dict(value.get("runtime_preflight") or value.get("preflight") or {})
    executed = value.get("search_executed") is True
    succeeded = executed and value.get("ok") is not False and value.get("status") != "failed"
    return {
        "schema_version": CHEMENZY_PROVIDER_CAPABILITY_SCHEMA,
        "provider_id": "chemenzy",
        "selection_source": str(preflight.get("env_prefix_selection_source") or ""),
        "env_prefix": str(preflight.get("env_prefix") or ""),
        "python_executable": str(preflight.get("python_executable") or ""),
        "levels": {
            "discovered": preflight.get("filesystem_accepted") is True,
            "importable": preflight.get("production_ready") is True,
            "model_loadable": succeeded,
            "smoke_tested": succeeded,
            "campaign_ready": succeeded,
        },
        "search_executed": executed,
        "issues": list(preflight.get("issues") or []),
        "semantics": {
            "import_probe_does_not_claim_model_loaded": True,
            "campaign_ready_requires_successful_search": True,
        },
    }


def _normalized_routes(
    value: Mapping[str, Any], *, target_smiles: str
) -> list[dict[str, Any]]:
    routes = value.get("routes")
    if isinstance(routes, list) and all(
        isinstance(route, Mapping) and isinstance(route.get("steps"), list)
        for route in routes
    ):
        raw_routes = [dict(route) for route in routes]
    else:
        raw_routes = [
            route.to_dict()
        for route in route_candidates_from_chem_enzy_result(
            dict(value), target_smiles=target_smiles
        )
        ]
    return [
        _normalize_proposal_route(route, route_index=index)
        for index, route in enumerate(raw_routes, start=1)
    ]


def _normalize_proposal_route(
    route: Mapping[str, Any], *, route_index: int
) -> dict[str, Any]:
    """Translate old and current launcher schemas into proposal-only rows."""

    normalized_steps: list[dict[str, Any]] = []
    admission_reasons: set[str] = set()
    for step_index, raw_step in enumerate(route.get("steps") or [], start=1):
        if not isinstance(raw_step, Mapping):
            admission_reasons.add("invalid_step_payload")
            continue
        step = dict(raw_step)
        product = str(step.get("product_smiles") or step.get("product") or "").strip()
        reactants = _proposal_reactants(step)
        audit = audit_retrosynthetic_candidate(product, reactants)
        if audit.get("accepted") is not True:
            admission_reasons.update(
                str(reason) for reason in audit.get("reasons") or []
            )
        normalized_steps.append(
            {
                "step_index": step_index,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "rxn_smiles": str(
                    step.get("rxn_smiles")
                    or step.get("reaction_smiles")
                    or ""
                ),
                "source_model": str(
                    step.get("source_model")
                    or step.get("model")
                    or "ChemEnzyRetroPlanner"
                ),
                "score": step.get("score", step.get("confidence")),
                "stock_status": dict(step.get("stock_status") or {}),
                "condition_predictions": list(
                    step.get("condition_predictions") or []
                ),
                "enzyme_ec_annotations": [
                    dict(value)
                    for value in step.get("enzyme_ec_annotations") or []
                    if isinstance(value, Mapping)
                ],
                "catalyst_annotations": [
                    dict(value)
                    for value in step.get("catalyst_annotations") or []
                    if isinstance(value, Mapping)
                ],
                "raw_backend_metadata": dict(
                    step.get("raw_backend_metadata") or {}
                ),
                "is_enzymatic": bool(
                    step.get("is_enzymatic")
                    or step.get("enzyme_ec_annotations")
                ),
                "chemical_step_equivalent_count": step.get(
                    "chemical_step_equivalent_count"
                ),
                "replaced_step_ids": list(step.get("replaced_step_ids") or []),
                "selectivity_objective": str(
                    step.get("selectivity_objective") or ""
                ),
                "host_search_admission": {
                    "accepted": audit.get("accepted") is True,
                    "edge_digest": str(audit.get("edge_digest") or ""),
                    "reasons": list(audit.get("reasons") or []),
                    "not_reaction_proof": True,
                },
            }
        )
    if not normalized_steps:
        admission_reasons.add("missing_route_steps")
    return {
        "route_index": route_index,
        "steps": normalized_steps,
        "proposal_eligible": bool(normalized_steps) and not admission_reasons,
        "admission_reasons": sorted(admission_reasons),
        "backend_route_status": {
            "solved": route.get("solved"),
            "status": route.get("status"),
            "diagnostic_only": True,
        },
        "semantics": {
            "proposal_only": True,
            "host_search_admission_is_not_reaction_proof": True,
            "backend_solved_is_not_admission_authority": True,
        },
    }


def _proposal_reactants(step: Mapping[str, Any]) -> list[str]:
    values = step.get("reactant_smiles") or step.get("precursor_smiles")
    if isinstance(values, str):
        values = [part for part in values.split(".") if part]
    if not isinstance(values, (list, tuple)):
        main = str(
            step.get("main_reactant")
            or step.get("main_reactant_smiles")
            or ""
        ).strip()
        auxiliary = step.get("aux_reactants") or step.get("aux_reactant_smiles") or []
        if isinstance(auxiliary, str):
            auxiliary = [part for part in auxiliary.split(".") if part]
        values = ([main] if main else []) + list(auxiliary or [])
    return [str(value).strip() for value in values if str(value).strip()]


def _result(status: str, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": "v4_chemenzy_proposal_stage.v1",
        "stage": "chemenzy_baseline",
        "status": status,
        **values,
        "semantics": {
            "proposal_only": True,
            "canonical_host_admission_required": True,
            "raw_backend_solved_is_not_route_proof": True,
            "codex_receives_proposals_through_shared_hypergraph": True,
        },
    }


__all__ = [
    "ChemenzyProposalProvider",
    "ChemEnzyProposalRequest",
    "run_chemenzy_guided_frontier_stage",
    "run_chemenzy_proposal_stage",
]
