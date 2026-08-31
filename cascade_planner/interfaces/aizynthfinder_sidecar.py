"""Host bridge for paper-budget AiZynthFinder searches.

AiZynthFinder 4.4.1 has an older, mutually constrained dependency stack, so it
runs in ``.venv_aizynth`` rather than being imported by the main AutoPlanner
process.  This module owns the subprocess boundary, validates the returned
route topology, and writes accepted template steps through the canonical host
ingestion path with unambiguous AiZynthFinder provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any, Callable, Mapping, TYPE_CHECKING

from cascade_planner.agent.codex_worker import _run_worker_command

from cascade_planner.application.canonical_hypergraph import (
    CanonicalIngestionBatch,
    molecule_identity,
)
from cascade_planner.interfaces.chemenzy_route_topology import (
    compile_route_topology_lineage,
)

if TYPE_CHECKING:
    from cascade_planner.orchestration.retrosynthesis_service import (
        RetrosynthesisCampaignService,
    )


AiZynthFinderProvider = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AiZynthFinderSidecarConfig:
    python_executable: str = ""
    config_path: str = ""
    runtime_root: str = ""
    mode: str = "short_tail"

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolved_python(self) -> Path:
        explicit = self.python_executable or os.environ.get(
            "AUTOPLANNER_AIZYNTH_PYTHON", ""
        )
        if explicit:
            return Path(explicit).expanduser().resolve()
        scripts = "Scripts" if os.name == "nt" else "bin"
        executable = "python.exe" if os.name == "nt" else "python"
        return (self.repo_root / ".venv_aizynth" / scripts / executable).resolve()

    def resolved_runtime_root(self) -> Path:
        explicit = self.runtime_root or os.environ.get(
            "AUTOPLANNER_AIZYNTH_RUNTIME_ROOT", ""
        )
        if explicit:
            return Path(explicit).expanduser().resolve()
        python_executable = self.resolved_python()
        environment = python_executable.parent.parent
        if (
            python_executable.parent.name.lower() in {"scripts", "bin"}
            and environment.name.lower().startswith((".venv", "venv"))
        ):
            # Worktrees junction ``.venv_aizynth`` to one shared installation.
            # Resolve that junction and bind model/config/stock paths to the
            # installation's owning root, not to the source worktree.
            return environment.parent.resolve()
        return self.repo_root

    def resolved_config(self) -> Path:
        explicit = self.config_path or os.environ.get(
            "AUTOPLANNER_AIZYNTH_CONFIG", ""
        )
        if explicit:
            return Path(explicit).expanduser().resolve()
        return (
            self.resolved_runtime_root()
            / "config"
            / "aizynthfinder.paper.yml"
        ).resolve()


def run_aizynthfinder_sidecar(
    *,
    target_smiles: str,
    timeout_s: float,
    sidecar_config: AiZynthFinderSidecarConfig | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run native AiZynthFinder and return a validated JSON envelope."""

    config = sidecar_config or AiZynthFinderSidecarConfig()
    python_executable = config.resolved_python()
    runtime_root = config.resolved_runtime_root()
    aizynth_config = config.resolved_config()
    script = config.repo_root / "scripts" / "run_aizynthfinder_paper_search.py"
    runtime_binding = {
        "python_executable": str(python_executable),
        "config_path": str(aizynth_config),
        "runtime_root": str(runtime_root),
        "source_root": str(config.repo_root),
        "script_path": str(script),
    }
    missing = [
        str(path)
        for path in (python_executable, aizynth_config, script)
        if not path.is_file()
    ]
    if not runtime_root.is_dir():
        missing.append(str(runtime_root))
    if missing:
        return {
            "schema_version": "aizynthfinder_sidecar_result.v1",
            "status": "failed",
            "reason": "aizynthfinder_runtime_missing",
            "missing_paths": missing,
            "runtime_binding": runtime_binding,
            "search_executed": False,
            "provider_invocation_count": 0,
            "proposal_routes": [],
        }

    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            prefix="autoplanner-aizynth-",
            suffix=".json",
            delete=False,
        ) as handle:
            output_path = handle.name
        command = [
            str(python_executable),
            str(script),
            "--smiles",
            target_smiles,
            "--mode",
            config.mode,
            "--config",
            str(aizynth_config),
            "--output",
            output_path,
        ]
        if cancel_event is None:
            completed = subprocess.run(
                command,
                # AiZynthFinder resolves the portable config's model and stock
                # paths relative to this explicitly bound runtime root.
                cwd=str(runtime_root),
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_s)) + 30.0,
                check=False,
            )
            returncode, stderr = completed.returncode, completed.stderr
        else:
            returncode, _stdout, stderr = _run_worker_command(
                command,
                cwd=runtime_root,
                timeout_s=max(1.0, float(timeout_s)) + 30.0,
                cancel_event=cancel_event,
                cancel_backend="aizynthfinder_sidecar",
            )
        if returncode != 0:
            return {
                "schema_version": "aizynthfinder_sidecar_result.v1",
                "status": "failed",
                "reason": "aizynthfinder_process_failed",
                "returncode": returncode,
                "stderr_tail": stderr[-4000:],
                "runtime_binding": runtime_binding,
                "search_executed": True,
                "provider_invocation_count": 1,
                "proposal_routes": [],
            }
        payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        return {
            "schema_version": "aizynthfinder_sidecar_result.v1",
            "status": "failed",
            "reason": "aizynthfinder_process_timeout",
            "runtime_binding": runtime_binding,
            "search_executed": True,
            "provider_invocation_count": 1,
            "proposal_routes": [],
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "aizynthfinder_sidecar_result.v1",
            "status": "failed",
            "reason": "aizynthfinder_result_invalid",
            "error": str(exc),
            "runtime_binding": runtime_binding,
            "search_executed": True,
            "provider_invocation_count": 1,
            "proposal_routes": [],
        }
    finally:
        if output_path:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                pass

    if payload.get("schema_version") != "aizynthfinder_paper_search.v1":
        return {
            "schema_version": "aizynthfinder_sidecar_result.v1",
            "status": "failed",
            "reason": "aizynthfinder_schema_mismatch",
            "runtime_binding": runtime_binding,
            "search_executed": True,
            "provider_invocation_count": 1,
            "proposal_routes": [],
        }
    _target_id, expected = molecule_identity(target_smiles)
    _result_id, actual = molecule_identity(str(payload.get("target_smiles") or ""))
    if expected != actual:
        return {
            "schema_version": "aizynthfinder_sidecar_result.v1",
            "status": "failed",
            "reason": "aizynthfinder_target_mismatch",
            "expected_target_smiles": expected,
            "actual_target_smiles": actual,
            "runtime_binding": runtime_binding,
            "search_executed": True,
            "provider_invocation_count": 1,
            "proposal_routes": [],
        }
    return {
        **dict(payload),
        "status": "completed",
        "runtime_binding": runtime_binding,
        "search_executed": True,
        "provider_invocation_count": 1,
    }


def run_aizynthfinder_guided_frontier_stage(
    service: "RetrosynthesisCampaignService",
    *,
    frontier_smiles: str,
    parent_route_family_ids: tuple[str, ...],
    timeout_s: float = 1_200.0,
    provider: AiZynthFinderProvider | None = None,
    sidecar_config: AiZynthFinderSidecarConfig | None = None,
    accept_partial_routes: bool = True,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Search one target-reachable leaf and ingest valid template routes.

    ``accept_partial_routes`` is intentionally explicit.  Enhanced AutoPlanner
    campaigns may import one coherent partial route for subsequent local
    repair.  The paper-matched SynthEx arm must set it to ``False``: an
    AiZynthFinder short tail is then promotable only when the provider reports
    every terminal leaf in its frozen stock.
    """

    raw = dict(
        provider(
            target_smiles=frontier_smiles,
            timeout_s=timeout_s,
            mode="short_tail",
        )
        if provider is not None
        else run_aizynthfinder_sidecar(
            target_smiles=frontier_smiles,
            timeout_s=timeout_s,
            sidecar_config=sidecar_config,
            cancel_event=cancel_event,
        )
    )
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("aizynthfinder_sidecar_cancelled")
    provider_invocation_count = int(
        raw.get("provider_invocation_count")
        or raw.get("search_executed") is True
        or provider is not None
    )
    accepted_routes: list[dict[str, Any]] = []
    rejected_routes: list[dict[str, Any]] = []
    for route in raw.get("proposal_routes") or []:
        normalized, reasons = _validate_route(
            dict(route), target_smiles=frontier_smiles
        )
        if reasons:
            rejected_routes.append(
                {
                    "route_trace_id": str(dict(route).get("route_trace_id") or ""),
                    "reasons": reasons,
                }
            )
        else:
            accepted_routes.append(normalized)

    digest = hashlib.sha256(frontier_smiles.encode("utf-8")).hexdigest()[:12]
    # The paper endpoint is existential: one coherent stock-closed route is
    # sufficient.  Importing every partial RouteCollection member floods the
    # host frontier with mutually incompatible steps and can spend the entire
    # materialization budget before one complete tail is replayed.  Preserve
    # one complete route when available.  Partial-route ingestion is a useful
    # AutoPlanner enhancement, but is not the paper short-tail contract and is
    # therefore opt-in at this boundary.
    solved_routes = [
        route
        for route in accepted_routes
        if route.get("all_leaves_in_provider_stock") is True
    ]
    selected_routes = (
        solved_routes[:1]
        if solved_routes
        else accepted_routes[:1]
        if accept_partial_routes
        else []
    )
    hypotheses: list[dict[str, Any]] = []
    route_aliases: dict[str, str] = {}
    frontier_molecule_id, _frontier_key = molecule_identity(frontier_smiles)
    for route_index, route in enumerate(selected_routes, start=1):
        alias = f"aizynthfinder:guided-{digest}:route:{route_index}"
        trace_id = str(route.get("route_trace_id") or "")
        if trace_id:
            route_aliases[trace_id] = alias
        short_tail_binding = {
            "schema_version": "provider_short_tail_binding.v1",
            "provider_group_id": alias,
            "frontier_smiles": frontier_smiles,
            "frontier_molecule_id": frontier_molecule_id,
            "parent_route_family_ids": list(parent_route_family_ids),
            "paper_short_tail_eligible": True,
            "target_rooted_open_leaf": True,
        }
        for step_index, step in enumerate(route.get("steps") or [], start=1):
            metadata = {
                "provider": "aizynthfinder",
                "engine": str(raw.get("engine") or "AiZynthFinder"),
                "mode": str(raw.get("mode") or "short_tail"),
                "policy_probability": float(
                    step.get("policy_probability") or 0.0
                ),
                "policy_probability_rank": step.get("policy_probability_rank"),
                "template_hash": str(step.get("template_hash") or ""),
                "template_code": step.get("template_code"),
                "classification": str(step.get("classification") or ""),
                "mapped_reaction_smiles": str(
                    step.get("mapped_reaction_smiles") or ""
                ),
                "provider_stock_status": list(
                    step.get("reactant_stock_status") or []
                ),
                "provider_stock_is_advisory": True,
                "route_trace_id": str(route.get("route_trace_id") or ""),
                "short_tail_binding": short_tail_binding,
            }
            hypotheses.append(
                {
                    "step_id": f"{alias}:step:{step_index}",
                    "proposal_id": f"{alias}:step:{step_index}",
                    "route_family_id": alias,
                    "canonical_route_family_id": (
                        parent_route_family_ids[0]
                        if parent_route_family_ids
                        else ""
                    ),
                    "canonical_route_family_ids": list(parent_route_family_ids),
                    "product_smiles": str(step.get("product_smiles") or ""),
                    "precursor_smiles": list(step.get("reactant_smiles") or []),
                    "origin_kind": "aizynthfinder",
                    "origin_ref": (
                        f"{alias}:{step.get('source_model') or 'template'}"
                    ),
                    "transformation_hypothesis": (
                        "AiZynthFinder template disconnection: "
                        f"{step.get('product_smiles')} -> "
                        f"{'.'.join(step.get('reactant_smiles') or [])}"
                    ),
                    "provider_reaction_metadata": metadata,
                    "condition_predictions": [],
                }
            )

    applied: Mapping[str, Any] = {"changed": False}
    if hypotheses:
        applied = service.apply_batch(
            CanonicalIngestionBatch(hypotheses=tuple(hypotheses)),
            idempotency_key=(
                f"solve-target:aizynthfinder:guided-{digest}:proposal-ingestion"
            ),
        )
    imported_proposal_ids = {
        str(hypothesis.get("proposal_id") or hypothesis.get("step_id") or "")
        for hypothesis in hypotheses
        if str(hypothesis.get("proposal_id") or hypothesis.get("step_id") or "")
    }
    canonical_proposal_ids: set[str] = set()
    graph_available = False
    graph_store = getattr(service, "graph_store", None)
    graph_loader = getattr(graph_store, "load", None)
    if callable(graph_loader):
        graph = graph_loader()
        graph_available = True
        canonical_proposal_ids = {
            str(origin.get("proposal_id") or "")
            for hypothesis in dict(graph.get("hypotheses") or {}).values()
            if isinstance(hypothesis, Mapping)
            for origin in hypothesis.get("origin_records") or []
            if isinstance(origin, Mapping) and str(origin.get("proposal_id") or "")
        }
    route_lineage: list[dict[str, Any]] = []
    for route in selected_routes:
        trace_id = str(route.get("route_trace_id") or "")
        alias = route_aliases.get(trace_id, "")
        normalized_route_sha256 = _content_sha256(
            {
                "target_smiles": str(route.get("target_smiles") or frontier_smiles),
                "steps": list(route.get("steps") or []),
            }
        )
        route_lineage.append(
            {
                "provider_id": "aizynthfinder",
                "route_trace_id": trace_id,
                "route_index": route.get("route_index"),
                "raw_route_sha256": str(route.get("raw_route_sha256") or ""),
                "normalized_route_sha256": normalized_route_sha256,
                "proposal_eligible": True,
                "host_portfolio_selected": True,
                "preserved_as_advisory": False,
                "quarantined": False,
                "disposition": "host_portfolio_selected",
                "reasons": [],
                "canonical_route_family_alias": alias,
                "canonical_route_family_id": (
                    parent_route_family_ids[0]
                    if parent_route_family_ids
                    else ""
                ),
                "canonical_route_family_ids": list(parent_route_family_ids),
                **compile_route_topology_lineage(
                    route,
                    alias=alias,
                    imported_proposal_ids=imported_proposal_ids,
                    canonical_proposal_ids=canonical_proposal_ids,
                    applicable=graph_available,
                ),
            }
        )
    status = (
        "completed"
        if hypotheses
        else "failed"
        if str(raw.get("status") or "") == "failed"
        else "unresolved"
    )
    strict_no_complete_route = bool(
        not accept_partial_routes and accepted_routes and not solved_routes
    )
    reason = str(raw.get("reason") or "")
    if strict_no_complete_route and not reason:
        reason = "paper_short_tail_no_complete_stock_closed_route"
    return {
        "schema_version": "v4_aizynthfinder_guided_frontier_stage.v1",
        "stage": "aizynthfinder_guided_frontier",
        "provider_id": "aizynthfinder",
        "status": status,
        "frontier_count": 1,
        "executed_frontier_count": provider_invocation_count,
        "frontier_smiles": [frontier_smiles],
        "proposal_count": len(hypotheses),
        "accepted_route_count": len(accepted_routes),
        "selected_proposal_route_count": len(selected_routes),
        "budget_truncated_route_count": max(
            0, len(accepted_routes) - len(selected_routes)
        ),
        "rejected_route_count": len(rejected_routes),
        "rejected_routes": rejected_routes,
        "route_lineage": route_lineage,
        "provider_invocation_count": provider_invocation_count,
        "provider_solved": raw.get("solved") is True,
        "provider_mode": str(raw.get("mode") or ""),
        "provider_budget": dict(raw.get("budget") or {}),
        "complete_provider_route_count": len(solved_routes),
        "partial_route_ingestion_allowed": bool(accept_partial_routes),
        "changed": applied.get("changed") is True,
        "statistics": dict(raw.get("statistics") or {}),
        "runtime_binding": dict(raw.get("runtime_binding") or {}),
        "reason": reason,
        "material_events": (
            ["guided_provider_proposals_added"]
            if hypotheses
            else ["provider_search_no_complete_stock_closed_route"]
            if strict_no_complete_route
            else ["provider_search_exhausted_without_proposal"]
            if provider_invocation_count
            else []
        ),
        "semantics": {
            "paper_short_tail_engine": "AiZynthFinder",
            "target_reachable_frontier_only": True,
            "provider_stock_status_is_advisory": True,
            "host_canonical_ingestion_required": True,
            "one_coherent_tail_selected_before_materialization": True,
            "paper_matched_complete_route_required": not bool(
                accept_partial_routes
            ),
            "legacy_scheduler_action_kind_may_wrap_stage": True,
            "graph_provenance_is_aizynthfinder": True,
        },
    }


def _validate_route(
    route: dict[str, Any], *, target_smiles: str
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    steps = [dict(step) for step in route.get("steps") or []]
    if not steps:
        return route, ["aizynthfinder_route_has_no_steps"]
    _target_id, target_key = molecule_identity(target_smiles)
    _root_id, root_key = molecule_identity(str(steps[0].get("product_smiles") or ""))
    if target_key != root_key:
        reasons.append("aizynthfinder_route_root_mismatch")
    available_products = {target_key}
    seen_products: set[str] = set()
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        product = str(step.get("product_smiles") or "").strip()
        reactants = [
            str(value).strip()
            for value in step.get("reactant_smiles")
            or step.get("precursor_smiles")
            or []
            if str(value).strip()
        ]
        _product_id, product_key = molecule_identity(product)
        reactant_keys = [molecule_identity(value)[1] for value in reactants]
        if not product_key or not reactant_keys or any(not value for value in reactant_keys):
            reasons.append("aizynthfinder_route_step_structure_invalid")
            continue
        if product_key not in available_products:
            reasons.append("aizynthfinder_route_step_disconnected")
        if product_key in seen_products:
            reasons.append("aizynthfinder_route_product_expanded_more_than_once")
        seen_products.add(product_key)
        available_products.update(reactant_keys)
        normalized_steps.append(
            {
                **step,
                "product_smiles": product_key,
                "reactant_smiles": reactant_keys,
                "precursor_smiles": reactant_keys,
            }
        )
    return {
        **route,
        "raw_step_count": int(route.get("step_count") or len(steps)),
        "steps": normalized_steps,
    }, list(dict.fromkeys(reasons))


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
