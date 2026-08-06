"""Bounded ChemEnzy proposal ingestion for the target-only V4 campaign."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import subprocess
import threading
import time
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
from cascade_planner.interfaces.chemenzy_advisory import (
    normalized_quarantined_routes,
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
_CHEMENZY_SUBPROCESS_LOCK = threading.Lock()


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
    max_host_routes: int | None = None,
    max_steps: int = 6,
    max_iterations: int = 10,
    expansion_topk: int = 20,
    timeout_s: float = 90.0,
    mode: str = "seed",
    scope: str = "seed",
    parent_route_family_ids: tuple[str, ...] = (),
    retron_hints: tuple[str, ...] = (),
    forbidden_smiles: tuple[str, ...] = (),
    search_preset: str = "standard",
    stock_names: tuple[str, ...] = (),
    stock_paths: Mapping[str, str] | None = None,
    enable_condition_prediction: bool = True,
    enable_enzyme_assignment: bool = True,
    enable_enzyme_coverage_sidecar: bool = True,
    pandarallel_workers: int = 2,
) -> dict[str, Any]:
    """Acquire a small proposal pool and admit it through the canonical graph."""

    limits = {
        "max_routes": max(1, int(max_routes)),
        "max_host_routes": max(
            1,
            min(
                max(1, int(max_routes)),
                int(max_host_routes or max_routes),
            ),
        ),
        "max_steps": max(1, int(max_steps)),
        "max_iterations": max(1, int(max_iterations)),
        "expansion_topk": max(1, int(expansion_topk)),
        "timeout_s": max(1.0, float(timeout_s)),
        "search_preset": str(search_preset or "standard"),
        "stock_names": [str(value) for value in stock_names if str(value).strip()],
        "stock_paths": {
            str(name): str(path)
            for name, path in dict(stock_paths or {}).items()
            if str(name).strip() and str(path).strip()
        },
        "enable_condition_prediction": bool(enable_condition_prediction),
        "enable_enzyme_assignment": bool(enable_enzyme_assignment),
        "enable_enzyme_coverage_sidecar": bool(enable_enzyme_coverage_sidecar),
        "pandarallel_workers": max(1, min(8, int(pandarallel_workers))),
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
    quarantined_routes = normalized_quarantined_routes(
        raw,
        start_index=len(routes) + 1,
        normalizer=_normalize_proposal_route,
    )
    # A backend-level ``solved`` flag is neither required nor trusted here.
    # ChemEnzy's current launcher returns proposal routes without that field;
    # older adapters sometimes emitted it.  Candidate admission is owned by
    # the host and is deliberately cheaper/weaker than reaction proof.
    eligible = [
        route for route in routes if route.get("proposal_eligible") is True
    ]
    accepted = _select_host_route_portfolio(
        eligible,
        limit=limits["max_host_routes"],
    )
    advisory = quarantined_routes[: limits["max_host_routes"]]
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
    route_alias_by_trace_id: dict[str, str] = {}
    selected_routes = [
        *((route, False) for route in accepted),
        *((route, True) for route in advisory),
    ]
    for route_index, (route, advisory_only) in enumerate(selected_routes, start=1):
        alias = f"chemenzy:{scope}:route:{route_index}"
        trace_id = str(route.get("route_trace_id") or "")
        if trace_id:
            route_alias_by_trace_id[trace_id] = alias
        if not parent_route_family_ids:
            route_families.append(
                {
                    "route_family_id": alias,
                    "selected": not advisory_only,
                    "strategy": (
                        "quarantined ChemEnzy route retained for review"
                        if advisory_only
                        else "bounded ChemEnzy multi-step proposal"
                    ),
                }
            )
        for step_index, step in enumerate(route.get("steps") or [], start=1):
            if step_index > limits["max_steps"]:
                break
            provider_metadata = _provider_reaction_metadata(step)
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
                    "transformation_hypothesis": (
                        _chemenzy_transformation_hypothesis(
                            step,
                            provider_metadata=provider_metadata,
                        )
                    ),
                    "provider_reaction_metadata": provider_metadata,
                    "advisory_only": advisory_only,
                    "provider_admission_reasons": list(
                        route.get("admission_reasons") or []
                    ),
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
    graph = service.graph_store.load()
    canonical_route_id_by_alias = {
        str(alias): str(route_id)
        for route_id, route in dict(graph.get("route_families") or {}).items()
        for alias in dict(route).get("aliases") or []
        if str(alias)
    }
    accepted_trace_ids = {
        str(route.get("route_trace_id") or "") for route in accepted
    } - {""}
    advisory_trace_ids = {
        str(route.get("route_trace_id") or "") for route in advisory
    } - {""}
    all_trace_routes = [
        *((route, False) for route in routes),
        *((route, True) for route in quarantined_routes),
    ]
    route_lineage = []
    for route, quarantined in all_trace_routes:
        trace_id = str(route.get("route_trace_id") or "")
        alias = route_alias_by_trace_id.get(trace_id, "")
        if trace_id in accepted_trace_ids:
            disposition = "host_portfolio_selected"
        elif trace_id in advisory_trace_ids:
            disposition = "quarantined_advisory"
        elif quarantined:
            disposition = "quarantined_advisory_budget_truncated"
        elif route.get("proposal_eligible") is True:
            disposition = "host_portfolio_budget_truncated"
        else:
            disposition = "host_search_rejected"
        route_lineage.append(
            {
                "route_trace_id": trace_id,
                "route_index": route.get("route_index"),
                "raw_route_sha256": str(route.get("raw_route_sha256") or ""),
                "normalized_route_sha256": str(
                    route.get("normalized_route_sha256") or ""
                ),
                "proposal_eligible": route.get("proposal_eligible") is True,
                "host_portfolio_selected": trace_id in accepted_trace_ids,
                "preserved_as_advisory": trace_id in advisory_trace_ids,
                "quarantined": quarantined,
                "disposition": disposition,
                "reasons": list(route.get("admission_reasons") or []),
                "canonical_route_family_alias": alias,
                "canonical_route_family_id": canonical_route_id_by_alias.get(alias, ""),
                "step_proposal_ids": [
                    f"{alias}:step:{index}"
                    for index, _step in enumerate(route.get("steps") or [], start=1)
                ]
                if alias
                else [],
            }
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
        preserved_advisory_route_count=len(advisory),
        rejected_route_count=len(routes) - len(eligible) + len(quarantined_routes),
        budget_truncated_route_count=max(0, len(eligible) - len(accepted)),
        provider_route_reserve=limits["max_routes"],
        host_route_portfolio_limit=limits["max_host_routes"],
        route_selection=[
            {
                "route_index": route.get("route_index"),
                "host_portfolio_rank": rank,
                "selection_features": _route_selection_features(route),
            }
            for rank, route in enumerate(accepted, start=1)
        ],
        route_admission=[
            {
                "route_index": route.get("route_index"),
                "proposal_eligible": route.get("proposal_eligible") is True,
                "reasons": list(route.get("admission_reasons") or []),
            }
            for route in routes
        ]
        + [
            {
                "route_index": route.get("route_index"),
                "proposal_eligible": False,
                "preserved_as_advisory": True,
                "reasons": list(route.get("admission_reasons") or []),
            }
            for route in quarantined_routes
        ],
        request_sha256=_content_sha256(request.to_dict()),
        raw_result_sha256=_content_sha256(raw),
        route_lineage=route_lineage,
        proposal_count=len(hypotheses),
        changed=applied.get("changed") is True,
        provider_envelope=provider_envelope,
        provider_registration=provider_registration,
        runtime_preflight=raw.get("runtime_preflight") or raw.get("preflight") or {},
        runtime_discovery=raw.get("runtime_discovery") or {},
        provider_capability=_provider_capability_snapshot(raw),
        reason=str(raw.get("reason") or ""),
        semantics={
            "provider_rejected_routes_are_retained_as_l0_advisory": True,
            "advisory_routes_never_grant_reaction_proof": True,
            "provider_route_reserve_is_distinct_from_host_portfolio": True,
            "provider_stock_status_is_ranking_only_not_stock_authority": True,
            "route_lineage_is_digest_bound_across_ingestion_boundaries": True,
        },
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
    include_frontier_smiles: tuple[str, ...] = (),
    search_preset: str = "thorough",
    stock_names: tuple[str, ...] = (),
    stock_paths: Mapping[str, str] | None = None,
    enable_condition_prediction: bool = True,
    enable_enzyme_assignment: bool = True,
    enable_enzyme_coverage_sidecar: bool = True,
    pandarallel_workers: int = 2,
) -> dict[str, Any]:
    """Expand only canonical Codex-selected or stock-rejected subtargets."""

    graph = service.graph_store.load()
    excluded = {str(value).strip() for value in exclude_frontier_smiles if str(value).strip()}
    included = {str(value).strip() for value in include_frontier_smiles if str(value).strip()}
    items = [
        dict(item)
        for item in dict(graph.get("deficit_frontier") or {}).get("items") or []
        if isinstance(item, Mapping)
        and item.get("kind") == "expansion"
        and dict(item.get("metadata") or {}).get("target_level_native_search")
        is not True
        and (
            not included
            or str(dict(item.get("metadata") or {}).get("frontier_smiles") or "")
            in included
        )
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
                search_preset=search_preset,
                stock_names=stock_names,
                stock_paths=stock_paths,
                enable_condition_prediction=enable_condition_prediction,
                enable_enzyme_assignment=enable_enzyme_assignment,
                enable_enzyme_coverage_sidecar=enable_enzyme_coverage_sidecar,
                pandarallel_workers=pandarallel_workers,
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
        "search_preset": str(limits.get("search_preset") or "standard"),
        # Keep the native search exhaustive within its configured MCTS budget,
        # but bound the expensive route annotation/materialization pass to the
        # number of proposals the host can actually consume.
        "max_routes": limits["max_routes"],
        "max_steps": limits["max_steps"],
        "chem_enzy_iterations": limits["max_iterations"],
        "chem_enzy_expansion_topk": limits["expansion_topk"],
        "stock_mode": "building-block",
        "device": "cpu",
        "enable_rule_verifier_gate": True,
        "enable_condition_prediction": bool(
            limits.get("enable_condition_prediction", True)
        ),
        "enable_enzyme_assignment": bool(limits.get("enable_enzyme_assignment", True)),
        "enable_enzyme_coverage_sidecar": bool(
            limits.get("enable_enzyme_coverage_sidecar", True)
        ),
    }
    if limits.get("stock_names"):
        request["stock_names"] = list(limits["stock_names"])
    if limits.get("stock_paths"):
        request["stock_paths"] = dict(limits["stock_paths"])
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
    environment = os.environ.copy()
    environment["CHEMENZY_PANDARALLEL_WORKERS"] = str(
        max(1, min(8, int(limits.get("pandarallel_workers") or 2)))
    )
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    queued_at = time.monotonic()
    try:
        with _CHEMENZY_SUBPROCESS_LOCK:
            queue_wait_s = max(0.0, time.monotonic() - queued_at)
            completed = subprocess.run(
                command,
                cwd=str(Path(preflight["launcher_path"]).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=float(limits["timeout_s"]),
                check=False,
                env=environment,
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
        "queue_wait_s": round(queue_wait_s, 3),
        "pandarallel_workers": int(environment["CHEMENZY_PANDARALLEL_WORKERS"]),
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


def compile_chemenzy_route_fingerprints(
    value: Mapping[str, Any], *, target_smiles: str
) -> dict[str, Any]:
    """Compile provider-output fingerprints without granting route authority."""

    routes = _normalized_routes(value, target_smiles=target_smiles)
    quarantined = normalized_quarantined_routes(
        value,
        start_index=len(routes) + 1,
        normalizer=_normalize_proposal_route,
    )
    rows = []
    for route, is_quarantined in [
        *((route, False) for route in routes),
        *((route, True) for route in quarantined),
    ]:
        rows.append(
            {
                "route_index": route.get("route_index"),
                "route_trace_id": str(route.get("route_trace_id") or ""),
                "raw_route_sha256": str(route.get("raw_route_sha256") or ""),
                "normalized_route_sha256": str(
                    route.get("normalized_route_sha256") or ""
                ),
                "proposal_eligible": route.get("proposal_eligible") is True,
                "quarantined": is_quarantined,
                "reasons": list(route.get("admission_reasons") or []),
                "step_count": len(route.get("steps") or []),
            }
        )
    return {
        "schema_version": "chemenzy_route_fingerprint_set.v1",
        "target_smiles": str(target_smiles),
        "raw_result_sha256": _content_sha256(value),
        "route_count": len(routes),
        "quarantined_route_count": len(quarantined),
        "routes": rows,
        "semantics": {
            "proposal_only": True,
            "fingerprints_grant_no_route_or_stock_authority": True,
        },
    }


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
                "raw_backend_metadata": _json_safe_copy(
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
    normalized = {
        "route_index": route_index,
        "steps": normalized_steps,
        "score": route.get("score", route.get("confidence")),
        "stock_status": dict(route.get("stock_status") or {}),
        "search_time_s": route.get("search_time_s"),
        "route_rank": route.get("route_rank", route_index - 1),
        "raw_backend_metadata": _json_safe_copy(
            route.get("raw_backend_metadata") or {}
        ),
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
    normalized["raw_route_sha256"] = _content_sha256(route)
    normalized["normalized_route_sha256"] = _content_sha256(normalized)
    normalized["route_trace_id"] = (
        f"chemenzy-route:{normalized['raw_route_sha256'][:24]}"
    )
    return normalized


def _select_host_route_portfolio(
    routes: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select a bounded, diverse host portfolio from the provider reserve."""

    remaining = sorted(
        (dict(route) for route in routes),
        key=_route_quality_key,
    )
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < max(1, int(limit)):
        ranked: list[tuple[Any, ...]] = []
        for route in remaining:
            signature = _route_edge_signature(route)
            diversity = (
                min(
                    _signature_distance(signature, _route_edge_signature(other))
                    for other in selected
                )
                if selected
                else 1.0
            )
            root = next(iter(sorted(signature)), "")
            root_novel = bool(
                not selected
                or all(
                    root != next(iter(sorted(_route_edge_signature(other))), "")
                    for other in selected
                )
            )
            ranked.append(
                (
                    -diversity,
                    -int(root_novel),
                    _route_quality_key(route),
                    int(route.get("route_index") or 0),
                    route,
                )
            )
        chosen = min(ranked)[-1]
        selected.append(chosen)
        remaining = [
            route
            for route in remaining
            if route.get("route_index") != chosen.get("route_index")
        ]
    return selected


def _route_selection_features(route: Mapping[str, Any]) -> dict[str, Any]:
    steps = [
        dict(step)
        for step in route.get("steps") or []
        if isinstance(step, Mapping)
    ]
    products = {str(step.get("product_smiles") or "") for step in steps}
    leaves = {
        str(value)
        for step in steps
        for value in step.get("reactant_smiles") or []
        if str(value) and str(value) not in products
    }
    provider_stock = {
        str(smiles): status
        for step in steps
        for smiles, status in dict(step.get("stock_status") or {}).items()
        if str(smiles)
    }
    stock_hint_count = sum(provider_stock.get(smiles) is True for smiles in leaves)
    template_step_count = sum(
        bool(dict(step.get("raw_backend_metadata") or {}).get("template"))
        for step in steps
    )
    reaction_smiles_count = sum(bool(step.get("rxn_smiles")) for step in steps)
    scored = [
        value
        for step in steps
        if (value := _finite_float(step.get("score"))) is not None
    ]
    route_score = _finite_float(route.get("score"))
    return {
        "step_count": len(steps),
        "leaf_count": len(leaves),
        "provider_stock_closed_leaf_hint_count": stock_hint_count,
        "provider_stock_closed_leaf_hint_rate": (
            round(stock_hint_count / len(leaves), 6) if leaves else 0.0
        ),
        "template_step_count": template_step_count,
        "reaction_smiles_step_count": reaction_smiles_count,
        "mean_step_score": (
            round(sum(scored) / len(scored), 8) if scored else None
        ),
        "route_score": route_score,
        "edge_signature": sorted(_route_edge_signature(route)),
        "semantics": {
            "provider_stock_is_non_authoritative_ranking_hint": True,
            "template_presence_is_replayability_hint_not_reaction_proof": True,
        },
    }


def _route_quality_key(route: Mapping[str, Any]) -> tuple[Any, ...]:
    features = _route_selection_features(route)
    mean_score = features["mean_step_score"]
    route_score = features["route_score"]
    return (
        -float(features["provider_stock_closed_leaf_hint_rate"]),
        -int(features["template_step_count"]),
        -int(features["reaction_smiles_step_count"]),
        -(float(route_score) if route_score is not None else -1.0),
        -(float(mean_score) if mean_score is not None else -1.0),
        int(features["step_count"]),
        int(route.get("route_index") or 0),
    )


def _route_edge_signature(route: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(dict(step.get("host_search_admission") or {}).get("edge_digest") or "")
        for step in route.get("steps") or []
        if isinstance(step, Mapping)
        and str(dict(step.get("host_search_admission") or {}).get("edge_digest") or "")
    )


def _signature_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


def _provider_reaction_metadata(step: Mapping[str, Any]) -> dict[str, Any]:
    raw = _json_safe_copy(step.get("raw_backend_metadata") or {})
    payload = {
        "schema_version": "chemenzy_provider_reaction_metadata.v1",
        "rxn_smiles": str(step.get("rxn_smiles") or ""),
        "source_model": str(step.get("source_model") or "ChemEnzyRetroPlanner"),
        "score": _finite_float(step.get("score")),
        "stock_status": _json_safe_copy(step.get("stock_status") or {}),
        "template": _json_safe_copy(raw.get("template")),
        "raw_backend_metadata": raw,
        "host_search_admission": _json_safe_copy(
            step.get("host_search_admission") or {}
        ),
        "semantics": {
            "provider_metadata_is_advisory": True,
            "host_template_replay_required_for_reaction_proof": True,
            "provider_stock_status_is_not_stock_authority": True,
        },
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _chemenzy_transformation_hypothesis(
    step: Mapping[str, Any],
    *,
    provider_metadata: Mapping[str, Any],
) -> str:
    source = str(step.get("source_model") or "ChemEnzyRetroPlanner")
    template = provider_metadata.get("template")
    if template:
        return (
            f"{source} reaction proposal with provider template metadata "
            f"{str(provider_metadata.get('content_sha256') or '')[:12]}"
        )
    if step.get("rxn_smiles"):
        return f"{source} reaction proposal with retained reaction SMILES"
    return f"{source} one-step retrosynthesis proposal"


def _json_safe_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe_copy(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
    extra_semantics = dict(values.pop("semantics", {}) or {})
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
            **extra_semantics,
        },
    }


__all__ = [
    "ChemenzyProposalProvider",
    "ChemEnzyProposalRequest",
    "compile_chemenzy_route_fingerprints",
    "run_chemenzy_guided_frontier_stage",
    "run_chemenzy_proposal_stage",
]
