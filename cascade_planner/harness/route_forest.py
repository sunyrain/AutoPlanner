"""Compile a finished blackboard run into a read-only explored route forest."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.harness.parent_route_proof import is_solved_parent_route_proof
from cascade_planner.harness.reaction_step_verifier import canonical_reaction_digest
from cascade_planner.harness.route_forest_delivery import (
    render_route_forest_html as render_route_forest_workbench_html,
)
from cascade_planner.harness.route_verifier import (
    is_accepted_route_verifier_report,
    verify_chemenzy_raw_routes,
)
from cascade_planner.harness.stitched_route import (
    compile_stitched_semisynthesis_route,
    is_validated_source_detail_literature_step,
)
from cascade_planner.source_locators import (
    canonical_traceable_source_ref,
    source_record_support_group,
)

try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception:  # pragma: no cover - route forest still renders without RDKit.
    Chem = None
    rdDepictor = None
    rdMolDescriptors = None
    rdMolDraw2D = None

try:
    from cascade_planner.application.frontier_ledger import validate_frontier_ledger
except Exception:  # pragma: no cover - missing ledger dependencies fail closed.
    validate_frontier_ledger = None


SCHEMA_VERSION = "explored_route_forest.v1"

PROOF_TIER_RANK = {
    "L0_rejected": 0,
    "L0_advisory": 1,
    "L0_materialized": 2,
    "L1_graph_stock_closed": 3,
    "L2_mapping_consistent": 4,
    "L2_reaction_validated": 5,
    "L3_precedent_supported": 6,
    "L4_procurement_ready": 7,
}
PROOF_TIER_STYLE = {
    "L0_rejected": {"color": "#be123c", "dash_pattern": "2 5", "texture": "crosshatched"},
    "L0_advisory": {"color": "#ea580c", "dash_pattern": "4 5", "texture": "dotted"},
    "L0_materialized": {"color": "#7c3aed", "dash_pattern": "3 3", "texture": "dotted"},
    "L1_graph_stock_closed": {"color": "#a16207", "dash_pattern": "9 4", "texture": "dashed"},
    "L2_mapping_consistent": {"color": "#64748b", "dash_pattern": "5 4", "texture": "striped_advisory"},
    "L2_reaction_validated": {"color": "#2563eb", "dash_pattern": "6 2", "texture": "striped"},
    "L3_precedent_supported": {"color": "#0f766e", "dash_pattern": "", "texture": "solid"},
    "L4_procurement_ready": {"color": "#15803d", "dash_pattern": "", "texture": "double"},
}

CONFIDENCE_RANK = {"failed": 0, "low": 1, "medium": 2, "medium_high": 3, "high": 4}
EXACTNESS_RANK = {
    "failed_or_unresolved": 0,
    "name_only": 1,
    "model_hypothesis": 2,
    "visual_inferred": 3,
    "named_literature": 4,
    "exact_literature_row": 5,
}

_STRUCTURE_CACHE: dict[str, dict[str, Any]] = {}

_SYNTHESIS_CLASS_FIELDS = {
    "synthesis_class",
    "route_claim",
    "route_class_hint",
    "route_objective_type",
    "objective_type",
    "process_type",
}
_SYNTHESIS_CLASSES = {"total_synthesis", "semisynthesis", "biosynthesis", "hybrid", "unspecified"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def frontier_ledger_current_authority_evidence(
    value: Mapping[str, Any],
    *,
    route_consensus_graph: Mapping[str, Any] | None,
    frontier_queue: Mapping[str, Any] | None,
    campaign_policy: Mapping[str, Any] | None,
    campaign_policy_sha256: str = "",
    trusted_stock_provider_instances: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify that one serialized ledger still represents current host state.

    Content hashes provide byte integrity, not freshness or authority.  A
    positive ledger is current only when its digest-bound input identities
    match the graph, queue revision, and immutable campaign policy supplied by
    this host, and the host replays every positive stock envelope again.
    """

    ledger = dict(value) if isinstance(value, Mapping) else {}
    input_bindings = (
        dict(ledger.get("input_bindings") or {})
        if isinstance(ledger.get("input_bindings"), Mapping)
        else {}
    )
    binding_reasons: list[str] = []
    if not input_bindings:
        binding_reasons.append("frontier_ledger_input_bindings_missing")
    elif input_bindings.get("schema_version") != "frontier_ledger_input_bindings.v1":
        binding_reasons.append("frontier_ledger_input_bindings_schema_invalid")

    graph_identity, graph_reasons = _current_route_graph_identity(
        route_consensus_graph
    )
    binding_reasons.extend(graph_reasons)
    bound_graph_identity = str(input_bindings.get("graph_identity_sha256") or "")
    if not _is_sha256(bound_graph_identity):
        binding_reasons.append("frontier_ledger_graph_identity_binding_missing")
    elif graph_identity and bound_graph_identity != graph_identity:
        binding_reasons.append("frontier_ledger_graph_identity_binding_mismatch")

    queue_digest, queue_revision, queue_reasons = _current_frontier_queue_identity(
        frontier_queue
    )
    binding_reasons.extend(queue_reasons)
    current_graph = dict(route_consensus_graph or {})
    current_queue = dict(frontier_queue or {})
    if (
        current_graph
        and current_queue
        and str(current_queue.get("run_id") or "")
        != str(current_graph.get("case_id") or "")
    ):
        binding_reasons.append("current_frontier_queue_graph_case_mismatch")
    bound_queue_digest = str(
        input_bindings.get("frontier_queue_content_sha256") or ""
    )
    bound_queue_revision = input_bindings.get("frontier_queue_revision")
    if not _is_sha256(bound_queue_digest):
        binding_reasons.append("frontier_ledger_queue_digest_binding_missing")
    elif queue_digest and bound_queue_digest != queue_digest:
        binding_reasons.append("frontier_ledger_queue_digest_binding_mismatch")
    if (
        isinstance(bound_queue_revision, bool)
        or not isinstance(bound_queue_revision, int)
        or bound_queue_revision < 0
    ):
        binding_reasons.append("frontier_ledger_queue_revision_binding_missing")
    elif queue_revision is not None and bound_queue_revision != queue_revision:
        binding_reasons.append("frontier_ledger_queue_revision_binding_mismatch")

    policy_digest, policy_reasons = _current_campaign_policy_identity(
        campaign_policy,
        declared_sha256=campaign_policy_sha256,
    )
    binding_reasons.extend(policy_reasons)
    current_policy = dict(campaign_policy or {})
    if current_graph and current_policy:
        if str(current_policy.get("case_id") or "") != str(
            current_graph.get("case_id") or ""
        ):
            binding_reasons.append("current_campaign_policy_graph_case_mismatch")
        policy_target = _canonical_molecule_smiles(
            str(current_policy.get("canonical_target_smiles") or "")
        )
        graph_target = _canonical_molecule_smiles(
            str(current_graph.get("target_smiles") or "")
        )
        if not policy_target or policy_target != graph_target:
            binding_reasons.append("current_campaign_policy_graph_target_mismatch")
    bound_policy_digest = str(
        input_bindings.get("campaign_policy_sha256") or ""
    )
    if not _is_sha256(bound_policy_digest):
        binding_reasons.append("frontier_ledger_campaign_policy_binding_missing")
    elif policy_digest and bound_policy_digest != policy_digest:
        binding_reasons.append("frontier_ledger_campaign_policy_binding_mismatch")

    historical_replay_recorded = _ledger_historical_host_replay_recorded(ledger)
    current_replay_reasons: list[str] = []
    if trusted_stock_provider_instances is None:
        current_replay_reasons.append(
            "current_host_stock_provider_replay_not_supplied"
        )
    elif validate_frontier_ledger is None:
        current_replay_reasons.append("frontier_ledger_validator_unavailable")
    else:
        try:
            current_replay_reasons.extend(
                validate_frontier_ledger(
                    ledger,
                    trusted_stock_provider_instances=dict(
                        trusted_stock_provider_instances
                    ),
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            current_replay_reasons.append(
                f"current_host_stock_provider_replay_error:{type(exc).__name__}"
            )

    blockers = sorted(set([*binding_reasons, *current_replay_reasons]))
    return {
        "schema_version": "route_forest_frontier_ledger_authority.v1",
        "historical_host_replay_recorded": historical_replay_recorded,
        "current_host_replay_verified": bool(
            trusted_stock_provider_instances is not None
            and not current_replay_reasons
        ),
        "current_input_bindings_verified": not binding_reasons,
        "authoritative": not blockers,
        "input_bindings": input_bindings,
        "current_identities": {
            "graph_identity_sha256": graph_identity,
            "frontier_queue_content_sha256": queue_digest,
            "frontier_queue_revision": queue_revision,
            "campaign_policy_sha256": policy_digest,
        },
        "blockers": blockers,
    }


def _current_route_graph_identity(
    value: Mapping[str, Any] | None,
) -> tuple[str, list[str]]:
    graph = dict(value) if isinstance(value, Mapping) else {}
    reasons: list[str] = []
    if not graph:
        return "", ["current_route_consensus_graph_missing"]
    if graph.get("schema_version") != "route_consensus_graph.v1":
        reasons.append("current_route_consensus_graph_schema_invalid")
    case_id = str(graph.get("case_id") or "").strip()
    if not case_id:
        reasons.append("current_route_consensus_graph_case_id_missing")
    target = _canonical_molecule_smiles(str(graph.get("target_smiles") or ""))
    if not target:
        reasons.append("current_route_consensus_graph_target_invalid")
    raw_steps = graph.get("steps")
    if not isinstance(raw_steps, list):
        reasons.append("current_route_consensus_graph_steps_invalid")
        raw_steps = []
    identity_steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            reasons.append(f"current_route_consensus_graph_step_invalid:{index}")
            continue
        step = dict(raw)
        product = _canonical_molecule_smiles(str(step.get("product_smiles") or ""))
        precursors = sorted(
            _canonical_molecule_smiles(str(item or ""))
            for item in step.get("precursor_smiles") or []
        )
        if (
            not str(step.get("step_id") or "")
            or not str(step.get("signature") or "")
            or not product
            or not precursors
            or any(not item for item in precursors)
        ):
            reasons.append(f"current_route_consensus_graph_step_identity_invalid:{index}")
            continue
        identity_steps.append(
            {
                "step_id": str(step.get("step_id") or ""),
                "signature": str(step.get("signature") or ""),
                "product_smiles": product,
                "precursor_smiles": precursors,
            }
        )
    if reasons:
        return "", sorted(set(reasons))
    identity = _canonical_json_sha256(
        {
            "schema_version": str(graph.get("schema_version") or ""),
            "case_id": case_id,
            "target_smiles": target,
            "steps": sorted(
                identity_steps,
                key=lambda row: (row["step_id"], row["signature"]),
            ),
        }
    )
    declared = str(graph.get("graph_identity_sha256") or "")
    if declared and (not _is_sha256(declared) or declared != identity):
        return "", ["current_route_consensus_graph_identity_mismatch"]
    return identity, []


def _current_frontier_queue_identity(
    value: Mapping[str, Any] | None,
) -> tuple[str, int | None, list[str]]:
    queue = dict(value) if isinstance(value, Mapping) else {}
    reasons: list[str] = []
    if not queue:
        return "", None, ["current_frontier_queue_missing"]
    if queue.get("schema_version") != "frontier_queue.v1":
        reasons.append("current_frontier_queue_schema_invalid")
    revision = queue.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        reasons.append("current_frontier_queue_revision_invalid")
        normalized_revision: int | None = None
    else:
        normalized_revision = revision
    if not isinstance(queue.get("jobs"), list):
        reasons.append("current_frontier_queue_jobs_invalid")
    supplied_digest = str(queue.get("content_sha256") or "")
    digest_payload = dict(queue)
    digest_payload.pop("content_sha256", None)
    if (
        not _is_sha256(supplied_digest)
        or supplied_digest != _canonical_json_sha256(digest_payload)
    ):
        reasons.append("current_frontier_queue_digest_invalid")
    return (
        supplied_digest if not reasons else "",
        normalized_revision,
        sorted(set(reasons)),
    )


def _current_campaign_policy_identity(
    value: Mapping[str, Any] | None,
    *,
    declared_sha256: str,
) -> tuple[str, list[str]]:
    policy = dict(value) if isinstance(value, Mapping) else {}
    reasons: list[str] = []
    if not policy:
        return "", ["current_campaign_policy_manifest_missing"]
    if policy.get("schema_version") != "codex_retrosynthesis_campaign_policy.v1":
        reasons.append("current_campaign_policy_schema_invalid")
    supplied_digest = str(policy.get("content_sha256") or "")
    digest_payload = dict(policy)
    digest_payload.pop("content_sha256", None)
    if (
        not _is_sha256(supplied_digest)
        or supplied_digest != _canonical_json_sha256(digest_payload)
    ):
        reasons.append("current_campaign_policy_digest_invalid")
    declared = str(declared_sha256 or "")
    if declared and (not _is_sha256(declared) or declared != supplied_digest):
        reasons.append("current_campaign_policy_declared_digest_mismatch")
    return supplied_digest if not reasons else "", sorted(set(reasons))


def _ledger_historical_host_replay_recorded(ledger: Mapping[str, Any]) -> bool:
    input_validation = dict(ledger.get("input_validation") or {})
    stock_authority = dict(input_validation.get("stock_authority") or {})
    try:
        replayed_claims = int(stock_authority.get("host_replayed_claim_count") or 0)
    except (TypeError, ValueError):
        replayed_claims = 0
    if replayed_claims > 0:
        return True
    return any(
        bool(dict(row.get("stock") or {}).get("closure_replay_bindings"))
        for row in (ledger.get("molecules") or {}).values()
        if isinstance(row, Mapping)
    )


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(str(value or "").strip().lower()))


def compile_explored_route_forest(
    blackboard: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    trusted_stock_provider_instances: Mapping[str, Any] | None = None,
    max_visual_branches: int | None = None,
    max_proposal_branches: int | None = None,
    max_template_branches: int | None = None,
) -> dict[str, Any]:
    """Project a complex blackboard into user-facing explored route branches."""
    compiler = _RouteForestCompiler(
        blackboard,
        run_dir=run_dir,
        trusted_stock_provider_instances=trusted_stock_provider_instances,
    )
    compiler.add_direct_verified_route_branch()
    compiler.add_stitched_verified_route_branch()
    compiler.add_subgoal_verified_route_branches()
    compiler.add_route_portfolio_branches()
    compiler.add_visual_branches(limit=max_visual_branches)
    compiler.add_process_evidence_branches()
    # Graph hypotheses are the canonical route projection.  Consensus rows
    # already represented by one of their explicit graph edges are merged
    # into that edge instead of being rendered a second time as a one-step
    # branch.
    compiler.add_route_consensus_graph_branches(limit=max_proposal_branches)
    compiler.add_route_consensus_branches(limit=max_proposal_branches)
    compiler.add_proposal_branches(limit=max_proposal_branches)
    compiler.add_template_branches(limit=max_template_branches)
    compiler.add_exact_row_branch()
    compiler.add_diagnostic_failure_branch_if_empty()
    return compiler.finish()


def write_route_forest_artifacts(
    blackboard: dict[str, Any],
    *,
    run_dir: str | Path,
    forest_output: str | Path | None = None,
    html_output: str | Path | None = None,
    trusted_stock_provider_instances: Mapping[str, Any] | None = None,
    max_visual_branches: int | None = None,
    max_proposal_branches: int | None = None,
    max_template_branches: int | None = None,
) -> dict[str, Any]:
    """Write the read-only route forest JSON and HTML display for a run."""
    run_path = Path(run_dir).resolve()
    forest_path = Path(forest_output).resolve() if forest_output is not None else run_path / "explored_route_forest.json"
    html_path = Path(html_output).resolve() if html_output is not None else run_path / "route_forest.html"
    forest = compile_explored_route_forest(
        blackboard,
        run_dir=run_path,
        trusted_stock_provider_instances=trusted_stock_provider_instances,
        max_visual_branches=max_visual_branches,
        max_proposal_branches=max_proposal_branches,
        max_template_branches=max_template_branches,
    )
    forest_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    forest_path.write_text(json.dumps(forest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_route_forest_html(forest), encoding="utf-8")
    return {
        "schema_version": "route_forest_outputs.v1",
        "forest_path": str(forest_path),
        "html_path": str(html_path),
        "forest": forest,
        "counts": dict(forest.get("counts") or {}),
        "target": dict(forest.get("target") or {}),
    }


def render_route_forest_html(forest: dict[str, Any]) -> str:
    """Render the compact, digest-bound route workbench."""

    return render_route_forest_workbench_html(forest)


class _RouteForestCompiler:
    def __init__(
        self,
        blackboard: dict[str, Any],
        *,
        run_dir: str | Path | None = None,
        trusted_stock_provider_instances: Mapping[str, Any] | None = None,
    ) -> None:
        self.blackboard = dict(blackboard or {})
        self.run_dir = str(run_dir or "")
        # ``None`` is intentionally distinct from an explicitly supplied
        # empty host trust set.  A serialized replay binding records history;
        # it cannot authenticate itself for the current display process.
        self.trusted_stock_provider_instances = (
            dict(trusted_stock_provider_instances)
            if trusted_stock_provider_instances is not None
            else None
        )
        self.evidence = dict(self.blackboard.get("literature_evidence") or {})
        self.nodes: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}
        self.branches: list[dict[str, Any]] = []
        self._branch_ids: set[str] = set()
        self._consensus_branch_ids: dict[str, str] = {}
        self._projection_coverage: dict[str, dict[str, Any]] = {}
        self._portfolio_branch_ids: dict[str, str] = {}
        self._portfolio_replacement_records: list[dict[str, Any]] = []
        self._portfolio_projection: dict[str, Any] = {
            "schema_version": "route_portfolio_projection.v1",
            "available": False,
            "source_route_count": 0,
            "projected_route_count": 0,
            "rejected_route_count": 0,
            "replacement_preview_branch_count": 0,
            "solver_truncated": False,
            "reasons": [],
        }
        self._frontier_ledger_cache: dict[str, Any] | None = None

    def finish(self) -> dict[str, Any]:
        self._finalize_trust_vectors()
        replacement_validation = self._replacement_validation()
        modules = self._modules()
        relationships = self._branch_relationships()
        dependency_graph = self._dependency_graph()
        target = self._target()
        route_consensus = self._route_consensus_view()
        route_consensus_graph = self._route_consensus_graph_view()
        authority_graph, authority_graph_source = (
            self._frontier_authority_route_consensus_graph()
        )
        route_portfolio = dict(route_consensus_graph.get("route_portfolio") or {})
        primary_selection = self._primary_selection()
        frontier_ledger = self._frontier_ledger_view()
        semantic_summary = self._semantic_summary(frontier_ledger=frontier_ledger)
        synthesis_class_counts: dict[str, int] = {}
        for branch in self.branches:
            synthesis_class = str(branch.get("synthesis_class") or "unspecified")
            synthesis_class_counts[synthesis_class] = synthesis_class_counts.get(synthesis_class, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": str(self.blackboard.get("case_id") or ""),
            "run_dir": self.run_dir,
            "target": target,
            "counts": {
                "branches": len(self.branches),
                "explored_branch_views": len(self.branches),
                "agent_tasks_completed": int(
                    (semantic_summary.get("agent_tasks") or {}).get("completed") or 0
                ),
                "agent_tasks_total": int(
                    (semantic_summary.get("agent_tasks") or {}).get("total") or 0
                ),
                "l0_advisory_branches": int(
                    (semantic_summary.get("route_states") or {}).get("l0_advisory") or 0
                ),
                "reaction_validated_branches": int(
                    (semantic_summary.get("route_states") or {}).get("reaction_validated") or 0
                ),
                "stock_closed_branches": int(
                    (semantic_summary.get("route_states") or {}).get("stock_closed") or 0
                ),
                "complete_portfolio_routes": int(
                    (semantic_summary.get("route_states") or {}).get("complete_portfolio") or 0
                ),
                "verified_parent_routes": int(
                    (semantic_summary.get("route_states") or {}).get("verified_parent") or 0
                ),
                "ledger_l0_break_suggestions": int(
                    (frontier_ledger.get("counts") or {}).get("l0_break_suggestion_edges") or 0
                ),
                "ledger_expanded_work_molecules": int(
                    (frontier_ledger.get("counts") or {}).get("expanded_work_molecules") or 0
                ),
                "ledger_l2_reaction_edges": int(
                    (frontier_ledger.get("counts") or {}).get("l2_reaction_edges") or 0
                ),
                "ledger_l3_precedent_edges": int(
                    (frontier_ledger.get("counts") or {}).get("l3_precedent_edges") or 0
                ),
                "ledger_l4_procurement_edges": int(
                    (frontier_ledger.get("counts") or {}).get("l4_procurement_edges") or 0
                ),
                "ledger_stock_closed_molecules": int(
                    (frontier_ledger.get("counts") or {}).get("stock_closed_molecules") or 0
                ),
                "nodes": len(self.nodes),
                "steps": len(self.steps),
                "visual_chains": len(self.evidence.get("visual_chains") or []),
                "process_evidence_rows": len(self.evidence.get("process_evidence_rows") or []),
                "exact_rows": len(self.evidence.get("exact_rows") or []),
                "broad_templates": len(self.blackboard.get("broad_transform_templates") or []),
                "proposals": len(self.blackboard.get("retrosynthetic_proposals") or []),
                "route_consensus_proposals": len(route_consensus.get("proposals") or []),
                "route_consensus_rejected_candidates": int(
                    (route_consensus.get("source_summary") or {}).get("rejected_count") or 0
                ),
                "route_consensus_graph_routes": int(route_consensus_graph.get("route_count") or 0),
                "route_consensus_graph_steps": int(route_consensus_graph.get("step_count") or 0),
                "route_portfolio_routes": len(route_portfolio.get("routes") or []),
                "route_portfolio_branches": int(
                    self._portfolio_projection.get("projected_route_count") or 0
                ),
                "replacement_preview_branches": int(
                    self._portfolio_projection.get("replacement_preview_branch_count") or 0
                ),
                "semisynthesis_anchors": len(self.blackboard.get("semisynthesis_anchors") or []),
                "scout_attempts": len(self.evidence.get("scout_attempts") or []),
                "relationships": len(relationships),
                "reaction_nodes": len(dependency_graph.get("reaction_nodes") or []),
                "dependency_edges": len(dependency_graph.get("edges") or []),
                "replacement_candidates": int(replacement_validation.get("candidate_count") or 0),
                "validated_replacements": int(replacement_validation.get("validated_count") or 0),
                "truncated_projection_rows": sum(
                    int(row.get("omitted_count") or 0) for row in self._projection_coverage.values()
                ),
                "synthesis_classes": synthesis_class_counts,
            },
            "primary_branch_id": str(primary_selection.get("primary_branch_id") or ""),
            "primary_selection": primary_selection,
            "semantic_summary": semantic_summary,
            "frontier_ledger": frontier_ledger,
            "display_policy": self._display_policy(),
            "branches": self.branches,
            "nodes": sorted(self.nodes.values(), key=lambda row: str(row.get("node_id") or "")),
            "steps": sorted(self.steps.values(), key=lambda row: str(row.get("step_id") or "")),
            "modules": modules,
            "relationships": relationships,
            "dependency_graph": dependency_graph,
            "replacement_validation": replacement_validation,
            "artifact_revision": self._artifact_revision_view(),
            "projection_coverage": {
                "schema_version": "route_forest_projection_coverage.v1",
                "complete": not any(bool(row.get("truncated")) for row in self._projection_coverage.values()),
                "categories": self._projection_coverage,
            },
            "route_consensus": route_consensus,
            "route_consensus_graph": route_consensus_graph,
            "canonical_route_consensus_graph": (
                authority_graph
                if authority_graph_source != "caller_advisory_route_consensus_graph"
                else {}
            ),
            "canonical_route_consensus_graph_source": authority_graph_source,
            "route_portfolio_projection": dict(self._portfolio_projection),
            "evidence_index": self._evidence_index(),
            "run_trace": self._run_trace(),
            "design_notes": [
                "This is a read-only projection of explored blackboard branches.",
                "The UI is read-only: a replacement preview switches to a complete backend AND/OR-revalidated branch and never runs planning.",
                "Named or visual-inferred nodes may intentionally omit SMILES when exact structure recovery was not reliable.",
                "Solved stitched branches are rebuilt only from revalidated proof inputs as stock-to-frontier-to-target DAGs.",
                "Visual, process, and consensus branches remain independent advisory alternatives.",
                "Consensus edges already present in a graph hypothesis are merged into that graph edge and are not projected as duplicate one-step branches.",
                "Standalone route consensus branches are advisory disconnections not represented in the graph; they are never solved or executable routes.",
                "Route consensus graph branches assemble frontier expansions but remain advisory and non-executable.",
                "Agent task completion, reaction validation, stock closure, and complete portfolio closure are separate display states.",
                "Codex role channels are displayed separately but share one correlated support group.",
                "The dependency graph is molecule-reaction bipartite; no edge is inferred from adjacent array positions.",
                "Pairwise replacement interfaces are diagnostics only and never authorize a single-step splice.",
                "Replacement previews require backend connectivity, stock, and reaction-proof revalidation of the complete route.",
            ],
        }

    def _semantic_summary(self, *, frontier_ledger: dict[str, Any]) -> dict[str, Any]:
        """Expose orthogonal execution and chemistry states for honest UI counts."""
        team = dict(self.blackboard.get("codex_agent_team") or {})
        runtime = dict(team.get("runtime_summary") or {})
        children = [dict(row) for row in runtime.get("children") or [] if isinstance(row, dict)]
        campaign = dict(team.get("campaign") or {})
        if children:
            agent_total = len(children)
            agent_completed = sum(
                str(row.get("state") or "").lower() in {"succeeded", "completed"}
                for row in children
            )
        else:
            runs = [dict(row) for row in campaign.get("runs") or [] if isinstance(row, dict)]
            agent_total = len(runs)
            agent_completed = sum(
                str(row.get("status") or row.get("team_status") or "").lower()
                in {"accepted", "accepted_draft", "completed", "succeeded"}
                or row.get("team_report_accepted") is True
                for row in runs
            )

        route_states = {
            "l0_rejected": 0,
            "l0_advisory": 0,
            "reaction_validated": 0,
            "stock_closed": 0,
            "complete_portfolio": 0,
            "verified_parent": 0,
        }
        for branch in self.branches:
            vector = dict(branch.get("trust_vector") or {})
            tier = str(vector.get("proof_tier") or "L0_advisory")
            kind = str(branch.get("kind") or "")
            if tier == "L0_rejected":
                route_states["l0_rejected"] += 1
            elif tier in {"L0_advisory", "L0_materialized"} and branch.get(
                "advisory_only"
            ) is not False:
                route_states["l0_advisory"] += 1
            if PROOF_TIER_RANK.get(tier, 0) >= PROOF_TIER_RANK["L2_reaction_validated"]:
                route_states["reaction_validated"] += 1
            dependency = self._branch_dependency_view(branch)
            if dependency.get("all_leaves_stock_bound") is True:
                route_states["stock_closed"] += 1
            if kind == "proof_eligible_portfolio_route" and branch.get("complete") is True:
                route_states["complete_portfolio"] += 1
            if (
                branch.get("solved") is True
                and branch.get("executable") is True
                and branch.get("advisory_only") is False
            ):
                route_states["verified_parent"] += 1
        ledger_counts = dict(frontier_ledger.get("counts") or {})
        ledger_closure = dict(frontier_ledger.get("closure") or {})
        ledger_authoritative = frontier_ledger.get("authoritative") is True
        if not ledger_authoritative:
            # Keep historical counts in the ledger view for diagnostics, but
            # never project them into completion/progress semantics.
            ledger_counts = self._empty_frontier_ledger_counts()
        route_states.update(
            {
                "ledger_authoritative": ledger_authoritative,
                "l0_break_suggestion_edges": int(
                    ledger_counts.get("l0_break_suggestion_edges") or 0
                ),
                "expanded_work_molecules": int(
                    ledger_counts.get("expanded_work_molecules") or 0
                ),
                "l2_reaction_edges": int(ledger_counts.get("l2_reaction_edges") or 0),
                "l3_precedent_edges": int(ledger_counts.get("l3_precedent_edges") or 0),
                "l4_procurement_edges": int(
                    ledger_counts.get("l4_procurement_edges") or 0
                ),
                "stock_closed_molecules": int(
                    ledger_counts.get("stock_closed_molecules") or 0
                ),
                "benchmark_only_stock_molecules": int(
                    ledger_counts.get("benchmark_only_stock_molecules") or 0
                ),
                "procurement_boundary_molecules": int(
                    ledger_counts.get("procurement_boundary_molecules") or 0
                ),
                "any_benchmark_route_closed": ledger_closure.get(
                    "any_benchmark_route_closed"
                )
                is True,
                "all_explored_benchmark_closed": ledger_closure.get(
                    "all_explored_benchmark_closed"
                )
                is True,
                "any_procurement_route_closed": ledger_closure.get(
                    "any_procurement_route_closed"
                )
                is True,
                "all_explored_procurement_closed": ledger_closure.get(
                    "all_explored_procurement_closed"
                )
                is True,
                "any_route_closed": ledger_closure.get("any_route_closed") is True,
                "all_explored_graph_closed": ledger_closure.get(
                    "all_explored_graph_closed"
                )
                is True,
                "l3_parent_solved": ledger_closure.get("l3_parent_solved") is True,
                "l4_procurement_ready": ledger_closure.get("l4_procurement_ready")
                is True,
            }
        )
        return {
            "schema_version": "route_forest_semantic_summary.v1",
            "agent_tasks": {
                "completed": int(agent_completed),
                "total": int(agent_total),
                "team_artifact_accepted": team.get("accepted") is True,
                # Campaign completion is a scheduler observation.  Only the
                # digest-bound full frontier ledger can authorize graph
                # closure in this projection.
                "proof_closed": ledger_closure.get("all_explored_graph_closed")
                is True,
                "campaign_claimed_graph_complete": campaign.get("graph_complete") is True,
                "semantics": "task completion records execution only; it is not route proof",
            },
            "route_states": route_states,
            "frontier_ledger": frontier_ledger,
            "semantics": {
                "branch": "one explored display view, not necessarily a complete route",
                "l0_rejected": "rejected or structurally incomplete exploration record",
                "l0_advisory": "unvalidated proposal or hypothesis",
                "reaction_validated": "all displayed route proof tiers reach L2 reaction validation or above",
                "stock_closed": "every explicit synthesis leaf is bound to accepted stock evidence",
                "complete_portfolio": "complete proof-eligible AND/OR route selected by the portfolio solver",
                "verified_parent": "deterministically verified executable parent route",
                "l0_break_suggestion_edges": "proposed disconnections that have not reached L2 reaction validation",
                "expanded_work_molecules": "molecule frontiers with a succeeded proposal-expansion queue job",
                "l2_reaction_edges": "exact reaction edges accepted by the current host verifier at L2",
                "l3_precedent_edges": "L3 exact-precedent reaction edges; this is not by itself a solved parent route",
                "stock_closed_molecules": "molecule boundaries accepted by a trusted stock provider",
                "benchmark_only_stock_molecules": "benchmark membership closes a search leaf but never proves procurement",
                "procurement_boundary_molecules": "commercial, in-house, or common-commodity boundaries with procurement authority",
                "any_benchmark_route_closed": "existential benchmark-search fixed point; procurement boundaries are a stronger valid search leaf",
                "all_explored_benchmark_closed": "universal benchmark-search fixed point over every reachable edge and leaf",
                "any_procurement_route_closed": "existential fixed point whose leaves all have procurement authority",
                "all_explored_procurement_closed": "universal fixed point whose leaves all have procurement authority",
                "any_route_closed": "compatibility alias for any_benchmark_route_closed; never a procurement claim",
                "all_explored_graph_closed": "compatibility alias for all_explored_benchmark_closed; never a procurement claim",
                "l3_parent_solved": "a complete executable parent route whose weakest reaction edge reaches L3",
                "l4_procurement_ready": "a complete executable L4 parent route plus a positive procurement-only fixed point",
            },
        }

    def _frontier_ledger_view(self) -> dict[str, Any]:
        """Return a compact, fail-closed view of ``frontier_ledger.v1``.

        A graph/team summary is an index only.  Positive closure claims are
        projected exclusively from a complete ledger whose content digest,
        nested shape, summary binding, and target identity all validate.
        """

        if self._frontier_ledger_cache is not None:
            return dict(self._frontier_ledger_cache)

        summary, summary_source = self._frontier_ledger_summary_candidate()
        summary_reasons: list[str] = []
        if not summary:
            summary_reasons.append("frontier_ledger_summary_missing")
        elif summary.get("schema_version") != "frontier_ledger_summary.v1":
            summary_reasons.append("frontier_ledger_summary_schema_invalid")
        if summary and summary.get("input_valid") is not True:
            summary_reasons.append("frontier_ledger_summary_inputs_invalid")
        if summary and summary.get("ledger_validation_accepted") is not True:
            summary_reasons.append("frontier_ledger_summary_validation_not_accepted")
        for field in (
            "any_benchmark_route_closed",
            "all_explored_benchmark_closed",
            "any_procurement_route_closed",
            "all_explored_procurement_closed",
        ):
            if summary and not isinstance(summary.get(field), bool):
                summary_reasons.append(
                    f"frontier_ledger_summary_{field}_missing_or_invalid"
                )

        candidates = self._frontier_ledger_candidates(summary)
        rejection_reasons: list[str] = list(summary_reasons)
        accepted: dict[str, Any] = {}
        accepted_source = ""
        for source, candidate in candidates:
            if not isinstance(candidate, dict) or not candidate:
                continue
            reasons = (
                list(validate_frontier_ledger(candidate))
                if validate_frontier_ledger is not None
                else ["frontier_ledger_validator_unavailable"]
            )
            reasons.extend(summary_reasons)
            reasons.extend(self._frontier_ledger_consistency_reasons(candidate))
            root_smiles = str((candidate.get("root") or {}).get("canonical_smiles") or "")
            target_smiles = str(self._target().get("smiles") or "")
            if target_smiles and (
                not root_smiles or not _same_molecule(root_smiles, target_smiles)
            ):
                reasons.append("frontier_ledger_target_mismatch")
            if summary:
                expected_digest = str(
                    summary.get("frontier_ledger_content_sha256")
                    or summary.get("ledger_content_sha256")
                    or ""
                )
                if not expected_digest:
                    reasons.append("frontier_ledger_summary_digest_missing")
                elif expected_digest != str(candidate.get("content_sha256") or ""):
                    reasons.append("frontier_ledger_summary_digest_mismatch")
                ledger_summary = dict(candidate.get("summary") or {})
                for field in (
                    "any_benchmark_route_closed",
                    "all_explored_benchmark_closed",
                    "any_procurement_route_closed",
                    "all_explored_procurement_closed",
                    "any_route_closed",
                    "all_explored_graph_closed",
                    "reachable_molecule_count",
                    "reachable_edge_count",
                ):
                    if field in summary and summary.get(field) != ledger_summary.get(field):
                        reasons.append(f"frontier_ledger_summary_{field}_mismatch")
            if reasons:
                rejection_reasons.extend(reasons)
                continue
            accepted = dict(candidate)
            accepted_source = source
            break

        parent_state = self._parent_route_proof_state()
        if not accepted:
            if not candidates:
                rejection_reasons.append("frontier_ledger_artifact_missing")
            view = {
                "schema_version": "route_forest_frontier_ledger_view.v1",
                "source_schema_version": "",
                "available": False,
                "authoritative": False,
                "historical_host_replay_recorded": False,
                "current_host_replay_verified": False,
                "current_input_bindings_verified": False,
                "source_ref": "",
                "summary_source": summary_source,
                "content_sha256": "",
                "validation_reasons": sorted(
                    set(rejection_reasons or ["frontier_ledger_artifact_missing"])
                ),
                "authority_blockers": sorted(
                    set(rejection_reasons or ["frontier_ledger_artifact_missing"])
                ),
                "input_bindings": {},
                "current_identities": {},
                "stage_authority": self._frontier_stage_authority_projection(
                    authoritative=False,
                    molecules={},
                    edges={},
                    frontier_queue={},
                ),
                "counts": self._empty_frontier_ledger_counts(),
                "closure": {
                    "any_benchmark_route_closed": False,
                    "all_explored_benchmark_closed": False,
                    "any_procurement_route_closed": False,
                    "all_explored_procurement_closed": False,
                    "any_route_closed": False,
                    "all_explored_graph_closed": False,
                    "l3_parent_solved": parent_state["l3_parent_solved"],
                    "l4_parent_route_proof_ready": parent_state[
                        "l4_procurement_ready"
                    ],
                    "l4_procurement_ready": False,
                },
                "input_validation": {},
                "semantics": {
                    "missing_or_invalid_ledger_fails_closed": True,
                    "branch_count_is_not_completion": True,
                    "parent_route_proof_is_independent_from_hypergraph_closure": True,
                },
            }
            self._frontier_ledger_cache = view
            return dict(view)

        ledger_summary = dict(accepted.get("summary") or {})
        authority_context = self._current_frontier_authority_context()
        authority = frontier_ledger_current_authority_evidence(
            accepted,
            route_consensus_graph=authority_context["route_consensus_graph"],
            frontier_queue=authority_context["frontier_queue"],
            campaign_policy=authority_context["campaign_policy"],
            campaign_policy_sha256=str(
                authority_context.get("campaign_policy_sha256") or ""
            ),
            trusted_stock_provider_instances=self.trusted_stock_provider_instances,
        )
        authoritative = authority.get("authoritative") is True
        molecules = {
            str(key): dict(value)
            for key, value in (accepted.get("molecules") or {}).items()
            if isinstance(value, dict)
        }
        edges = {
            str(key): dict(value)
            for key, value in (accepted.get("edges") or {}).items()
            if isinstance(value, dict)
        }
        achieved_levels = [
            int((row.get("reaction_proof") or {}).get("achieved_proof_level") or 0)
            for row in edges.values()
        ]
        leaf_rows = [
            row
            for row in molecules.values()
            if not (row.get("proposal") or {}).get("outgoing_edge_signatures")
        ]
        counts = {
            "proposal_edges": len(edges),
            "l0_break_suggestion_edges": sum(level < 2 for level in achieved_levels),
            "expanded_work_molecules": sum(
                (row.get("work") or {}).get("proposal_expansion_succeeded") is True
                for row in molecules.values()
            ),
            "open_work_molecules": sum(
                (row.get("work") or {}).get("open") is True
                for row in molecules.values()
            ),
            "l2_reaction_edges": sum(level == 2 for level in achieved_levels),
            "l3_precedent_edges": sum(level == 3 for level in achieved_levels),
            "l4_procurement_edges": sum(level >= 4 for level in achieved_levels),
            "reaction_validated_edges": sum(level >= 2 for level in achieved_levels),
            "stock_closed_molecules": sum(
                (row.get("stock") or {}).get("closed") is True
                for row in molecules.values()
            ),
            "benchmark_only_stock_molecules": sum(
                (row.get("stock") or {}).get("benchmark_only") is True
                for row in molecules.values()
            ),
            "procurement_boundary_molecules": sum(
                (row.get("stock") or {}).get("procurement_boundary_closed") is True
                for row in molecules.values()
            ),
            "stock_closed_leaves": sum(
                (row.get("stock") or {}).get("closed") is True for row in leaf_rows
            ),
            "benchmark_only_stock_leaves": sum(
                (row.get("stock") or {}).get("benchmark_only") is True
                for row in leaf_rows
            ),
            "procurement_boundary_leaves": sum(
                (row.get("stock") or {}).get("procurement_boundary_closed") is True
                for row in leaf_rows
            ),
            "reachable_leaves": len(leaf_rows),
            "reachable_molecules": int(
                ledger_summary.get("reachable_molecule_count") or len(molecules)
            ),
            "reachable_edges": int(ledger_summary.get("reachable_edge_count") or len(edges)),
        }
        observed_closure = {
            "any_benchmark_route_closed": ledger_summary.get(
                "any_benchmark_route_closed"
            )
            is True,
            "all_explored_benchmark_closed": ledger_summary.get(
                "all_explored_benchmark_closed"
            )
            is True,
            "any_procurement_route_closed": ledger_summary.get(
                "any_procurement_route_closed"
            )
            is True,
            "all_explored_procurement_closed": ledger_summary.get(
                "all_explored_procurement_closed"
            )
            is True,
            "any_route_closed": ledger_summary.get("any_route_closed") is True,
            "all_explored_graph_closed": ledger_summary.get(
                "all_explored_graph_closed"
            )
            is True,
        }
        view = {
            "schema_version": "route_forest_frontier_ledger_view.v1",
            "source_schema_version": str(accepted.get("schema_version") or ""),
            "available": True,
            "authoritative": authoritative,
            "historical_host_replay_recorded": authority.get(
                "historical_host_replay_recorded"
            )
            is True,
            "current_host_replay_verified": authority.get(
                "current_host_replay_verified"
            )
            is True,
            "current_input_bindings_verified": authority.get(
                "current_input_bindings_verified"
            )
            is True,
            "source_ref": accepted_source,
            "summary_source": summary_source,
            "authority_graph_source": str(
                authority_context.get("route_consensus_graph_source") or ""
            ),
            "content_sha256": str(accepted.get("content_sha256") or ""),
            "required_reaction_proof_level": int(
                accepted.get("required_reaction_proof_level") or 2
            ),
            "validation_reasons": list(authority.get("blockers") or []),
            "authority_blockers": list(authority.get("blockers") or []),
            "input_bindings": dict(authority.get("input_bindings") or {}),
            "current_identities": dict(authority.get("current_identities") or {}),
            "stage_authority": self._frontier_stage_authority_projection(
                authoritative=authoritative,
                molecules=molecules,
                edges=edges,
                frontier_queue=authority_context["frontier_queue"],
            ),
            "counts": counts,
            "observed_closure": observed_closure,
            "closure": {
                "any_benchmark_route_closed": bool(
                    authoritative
                    and observed_closure["any_benchmark_route_closed"]
                ),
                "all_explored_benchmark_closed": bool(
                    authoritative
                    and observed_closure["all_explored_benchmark_closed"]
                ),
                "any_procurement_route_closed": bool(
                    authoritative
                    and observed_closure["any_procurement_route_closed"]
                ),
                "all_explored_procurement_closed": bool(
                    authoritative
                    and observed_closure["all_explored_procurement_closed"]
                ),
                "any_route_closed": bool(
                    authoritative and observed_closure["any_route_closed"]
                ),
                "all_explored_graph_closed": bool(
                    authoritative
                    and observed_closure["all_explored_graph_closed"]
                ),
                "l3_parent_solved": parent_state["l3_parent_solved"],
                "l4_parent_route_proof_ready": parent_state[
                    "l4_procurement_ready"
                ],
                "l4_procurement_ready": bool(
                    authoritative
                    and
                    parent_state["l4_procurement_ready"]
                    and observed_closure["any_procurement_route_closed"]
                ),
            },
            "input_validation": dict(accepted.get("input_validation") or {}),
            "semantics": {
                "missing_or_invalid_ledger_fails_closed": True,
                "branch_count_is_not_completion": True,
                "l0_is_unvalidated_proposed_edge_count": True,
                "work_is_durable_queue_execution_not_route_proof": True,
                "reaction_counts_are_exact_edge_proof_counts": True,
                "stock_counts_are_molecule_boundaries_not_complete_routes": True,
                "benchmark_stock_closure_is_not_procurement": True,
                "procurement_uses_procurement_boundary_closed_only": True,
                "any_and_all_closure_are_distinct_fixed_points": True,
                "generic_any_all_mean_benchmark_search_closure": True,
                "procurement_closure_has_an_independent_fixed_point": True,
                "l4_procurement_requires_parent_and_procurement_fixed_point": True,
                "parent_route_proof_is_independent_from_hypergraph_closure": True,
                "historical_replay_record_is_not_current_authority": True,
                "current_host_provider_replay_must_be_passed_explicitly": True,
                "graph_queue_policy_identity_bindings_are_required": True,
            },
        }
        self._frontier_ledger_cache = view
        return dict(view)

    @staticmethod
    def _frontier_stage_authority_projection(
        *,
        authoritative: bool,
        molecules: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
        frontier_queue: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project only current-authority facts needed by UI stage filters.

        The browser must not infer durable expansion, reaction replay, or stock
        closure from branch counts or proof colours.  Entries are therefore
        exposed only after the complete ledger and its current graph/queue/
        provider bindings have passed host validation.
        """

        semantics = {
            "expanded_requires_succeeded_proposal_expansion_job": True,
            "reaction_requires_current_host_edge_replay": True,
            "stock_requires_current_host_provider_replay_for_every_leaf": True,
            "benchmark_stock_is_not_procurement": True,
            "empty_or_unbound_projection_fails_closed": True,
        }
        if not authoritative:
            return {
                "schema_version": "route_forest_stage_authority.v1",
                "authoritative": False,
                "molecules": [],
                "edges": [],
                "reasons": ["frontier_ledger_not_current_authority"],
                "semantics": semantics,
            }

        succeeded_expansion_jobs: dict[str, list[str]] = {}
        for raw_job in frontier_queue.get("jobs") or []:
            if not isinstance(raw_job, Mapping):
                continue
            job = dict(raw_job)
            if (
                str(job.get("state") or "") != "succeeded"
                or str(job.get("closure_kind") or "") != "proposal_expansion"
            ):
                continue
            smiles = _canonical_molecule_smiles(str(job.get("frontier_smiles") or ""))
            job_id = str(job.get("job_id") or "")
            if smiles and job_id:
                succeeded_expansion_jobs.setdefault(smiles, []).append(job_id)

        molecule_rows: list[dict[str, Any]] = []
        for canonical_smiles, raw in sorted(molecules.items()):
            row = dict(raw)
            work = dict(row.get("work") or {})
            stock = dict(row.get("stock") or {})
            host_replay_verified = stock.get("host_replay_verified") is True
            canonical = _canonical_molecule_smiles(
                str(row.get("canonical_smiles") or canonical_smiles)
            )
            ledger_work_job_ids = {
                str(value) for value in work.get("job_ids") or [] if value
            }
            verified_expansion_job_ids = sorted(
                ledger_work_job_ids.intersection(
                    succeeded_expansion_jobs.get(canonical, [])
                )
            )
            current_observation_ids = sorted(
                str(value)
                for value in stock.get("current_observation_ids") or []
                if value
            )
            closure_job_ids = sorted(
                str(value) for value in stock.get("closure_job_ids") or [] if value
            )
            stock_evidence_complete = bool(
                host_replay_verified
                and current_observation_ids
                and closure_job_ids
            )
            molecule_rows.append(
                {
                    "canonical_smiles": canonical,
                    "node_ids": sorted(
                        str(value) for value in row.get("node_ids") or [] if value
                    ),
                    "work": {
                        "proposal_expansion_succeeded": work.get(
                            "proposal_expansion_succeeded"
                        )
                        is True
                        and bool(verified_expansion_job_ids),
                        "job_ids": verified_expansion_job_ids,
                    },
                    "stock": {
                        "benchmark_search_boundary_closed": bool(
                            stock_evidence_complete
                            and stock.get("benchmark_search_boundary_closed")
                            is True
                        ),
                        "procurement_boundary_closed": bool(
                            stock_evidence_complete
                            and stock.get("procurement_boundary_closed") is True
                        ),
                        "host_replay_verified": host_replay_verified,
                        "current_observation_ids": current_observation_ids,
                        "closure_job_ids": closure_job_ids,
                    },
                }
            )

        edge_rows: list[dict[str, Any]] = []
        for signature, raw in sorted(edges.items()):
            row = dict(raw)
            proposal = dict(row.get("proposal") or {})
            proof = dict(row.get("reaction_proof") or {})
            authority = str(proof.get("authority") or "")
            achieved_level = int(proof.get("achieved_proof_level") or 0)
            step_ids = sorted(
                {
                    str(value)
                    for value in [
                        *(proposal.get("step_ids") or []),
                        *(proof.get("step_ids") or []),
                    ]
                    if value
                }
            )
            current_host_validated = bool(
                achieved_level >= 2
                and authority == "current_host_verifier_replay"
                and isinstance(proof.get("host_replay_binding"), dict)
                and proof.get("host_replay_binding")
            )
            edge_rows.append(
                {
                    "exact_edge_signature": str(
                        row.get("exact_edge_signature") or signature
                    ),
                    "step_ids": step_ids,
                    "product_smiles": str(row.get("product_smiles") or ""),
                    "precursor_smiles": sorted(
                        str(value)
                        for value in row.get("precursor_smiles") or []
                        if value
                    ),
                    "reaction_proof": {
                        "achieved_proof_level": achieved_level,
                        "authority": authority,
                        "current_host_reaction_validated": current_host_validated,
                        "proof_request_ids": sorted(
                            str(value)
                            for value in proof.get("proof_request_ids") or []
                            if value
                        ),
                    },
                }
            )

        return {
            "schema_version": "route_forest_stage_authority.v1",
            "authoritative": True,
            "molecules": molecule_rows,
            "edges": edge_rows,
            "reasons": [],
            "semantics": semantics,
        }

    @staticmethod
    def _empty_frontier_ledger_counts() -> dict[str, int]:
        return {
            "proposal_edges": 0,
            "l0_break_suggestion_edges": 0,
            "expanded_work_molecules": 0,
            "open_work_molecules": 0,
            "l2_reaction_edges": 0,
            "l3_precedent_edges": 0,
            "l4_procurement_edges": 0,
            "reaction_validated_edges": 0,
            "stock_closed_molecules": 0,
            "benchmark_only_stock_molecules": 0,
            "procurement_boundary_molecules": 0,
            "stock_closed_leaves": 0,
            "benchmark_only_stock_leaves": 0,
            "procurement_boundary_leaves": 0,
            "reachable_leaves": 0,
            "reachable_molecules": 0,
            "reachable_edges": 0,
        }

    @staticmethod
    def _frontier_ledger_consistency_reasons(value: dict[str, Any]) -> list[str]:
        """Recompute ledger fixed points before exposing closure to the UI."""

        summary = value.get("summary")
        root = value.get("root")
        molecules = value.get("molecules")
        edges = value.get("edges")
        if not all(isinstance(row, dict) for row in (summary, root, molecules, edges)):
            return ["frontier_ledger_projection_shape_invalid"]
        reasons: list[str] = []
        try:
            required_level = int(value.get("required_reaction_proof_level") or 2)
        except (TypeError, ValueError):
            return ["frontier_ledger_required_proof_level_invalid"]
        if required_level not in {2, 3, 4}:
            reasons.append("frontier_ledger_required_proof_level_invalid")

        root_smiles = str(root.get("canonical_smiles") or "")
        if not root_smiles or root_smiles not in molecules:
            reasons.append("frontier_ledger_root_molecule_missing")
        outgoing: dict[str, list[str]] = {}
        stock_closed: dict[str, bool] = {}
        procurement_closed: dict[str, bool] = {}
        terminal_closed: dict[str, bool] = {}
        for smiles, raw in molecules.items():
            molecule = dict(raw) if isinstance(raw, dict) else {}
            signatures = (molecule.get("proposal") or {}).get(
                "outgoing_edge_signatures"
            )
            stock = (molecule.get("stock") or {}).get("closed")
            procurement = (molecule.get("stock") or {}).get(
                "procurement_boundary_closed"
            )
            terminal = (
                (molecule.get("reaction_proof") or {}).get("terminal_closure") or {}
            ).get("closed")
            if not isinstance(signatures, list) or any(
                not isinstance(item, str) or item not in edges for item in signatures
            ):
                reasons.append(f"frontier_ledger_molecule_outgoing_invalid:{smiles}")
                signatures = []
            if (
                not isinstance(stock, bool)
                or not isinstance(procurement, bool)
                or not isinstance(terminal, bool)
            ):
                reasons.append(f"frontier_ledger_molecule_boundary_invalid:{smiles}")
            outgoing[str(smiles)] = list(signatures)
            stock_closed[str(smiles)] = stock is True
            procurement_closed[str(smiles)] = procurement is True
            terminal_closed[str(smiles)] = terminal is True

        proof_closed: dict[str, bool] = {}
        precursors_by_edge: dict[str, list[str]] = {}
        for signature, raw in edges.items():
            edge = dict(raw) if isinstance(raw, dict) else {}
            product = str(edge.get("product_smiles") or "")
            precursors = edge.get("precursor_smiles")
            proof = dict(edge.get("reaction_proof") or {})
            closed = proof.get("closed")
            achieved = proof.get("achieved_proof_level")
            if (
                product not in molecules
                or not isinstance(precursors, list)
                or not precursors
                or any(not isinstance(item, str) or item not in molecules for item in precursors)
                or signature not in outgoing.get(product, [])
            ):
                reasons.append(f"frontier_ledger_edge_dependency_invalid:{signature}")
                precursors = []
            if (
                not isinstance(closed, bool)
                or isinstance(achieved, bool)
                or not isinstance(achieved, int)
                or not 0 <= achieved <= 4
                or closed is not (achieved >= required_level)
            ):
                reasons.append(f"frontier_ledger_edge_proof_invalid:{signature}")
            proof_closed[str(signature)] = closed is True
            precursors_by_edge[str(signature)] = [str(item) for item in precursors]

        def fixed_point(boundaries: dict[str, bool]) -> tuple[dict[str, bool], dict[str, bool]]:
            any_values = {
                str(smiles): boundaries.get(str(smiles), False)
                or terminal_closed.get(str(smiles), False)
                for smiles in molecules
            }
            all_values = {
                str(smiles): bool(
                    not outgoing.get(str(smiles))
                    and (
                        boundaries.get(str(smiles), False)
                        or terminal_closed.get(str(smiles), False)
                    )
                )
                for smiles in molecules
            }
            for _ in range(max(1, len(molecules) + 1)):
                next_any = dict(any_values)
                next_all = dict(all_values)
                for smiles in sorted(str(item) for item in molecules):
                    alternatives = outgoing.get(smiles, [])
                    if not alternatives:
                        continue
                    next_any[smiles] = bool(
                        any_values[smiles]
                        or any(
                            proof_closed.get(signature, False)
                            and all(
                                any_values.get(item, False)
                                for item in precursors_by_edge.get(signature, [])
                            )
                            for signature in alternatives
                        )
                    )
                    next_all[smiles] = bool(
                        all(
                            proof_closed.get(signature, False)
                            and all(
                                all_values.get(item, False)
                                for item in precursors_by_edge.get(signature, [])
                            )
                            for signature in alternatives
                        )
                    )
                if next_any == any_values and next_all == all_values:
                    break
                any_values, all_values = next_any, next_all
            return any_values, all_values

        any_closed, all_closed = fixed_point(stock_closed)
        procurement_any, procurement_all = fixed_point(procurement_closed)

        for smiles, raw in molecules.items():
            closure = (
                dict((raw or {}).get("closure") or {})
                if isinstance(raw, dict)
                else {}
            )
            expected = {
                "any_benchmark_route_closed": any_closed.get(str(smiles), False),
                "all_explored_benchmark_closed": all_closed.get(
                    str(smiles), False
                ),
                "any_procurement_route_closed": procurement_any.get(
                    str(smiles), False
                ),
                "all_explored_procurement_closed": procurement_all.get(
                    str(smiles), False
                ),
                "any_route_closed": any_closed.get(str(smiles), False),
                "all_explored_graph_closed": all_closed.get(str(smiles), False),
            }
            for field, expected_value in expected.items():
                if closure.get(field) is not expected_value:
                    reasons.append(
                        f"frontier_ledger_molecule_closure_mismatch:{smiles}:{field}"
                    )
        for signature, raw in edges.items():
            closure = dict((raw or {}).get("closure") or {}) if isinstance(raw, dict) else {}
            edge_any = bool(
                proof_closed.get(str(signature), False)
                and all(
                    any_closed.get(item, False)
                    for item in precursors_by_edge.get(str(signature), [])
                )
            )
            edge_all = bool(
                proof_closed.get(str(signature), False)
                and all(
                    all_closed.get(item, False)
                    for item in precursors_by_edge.get(str(signature), [])
                )
            )
            procurement_edge_any = bool(
                proof_closed.get(str(signature), False)
                and all(
                    procurement_any.get(item, False)
                    for item in precursors_by_edge.get(str(signature), [])
                )
            )
            procurement_edge_all = bool(
                proof_closed.get(str(signature), False)
                and all(
                    procurement_all.get(item, False)
                    for item in precursors_by_edge.get(str(signature), [])
                )
            )
            expected = {
                "any_benchmark_route_closed": edge_any,
                "all_explored_benchmark_closed": edge_all,
                "any_procurement_route_closed": procurement_edge_any,
                "all_explored_procurement_closed": procurement_edge_all,
                "any_route_closed": edge_any,
                "all_explored_graph_closed": edge_all,
            }
            for field, expected_value in expected.items():
                if closure.get(field) is not expected_value:
                    reasons.append(
                        f"frontier_ledger_edge_closure_mismatch:{signature}:{field}"
                    )

        expected_summary_closure = {
            "any_benchmark_route_closed": any_closed.get(root_smiles, False),
            "all_explored_benchmark_closed": all_closed.get(root_smiles, False),
            "any_procurement_route_closed": procurement_any.get(
                root_smiles, False
            ),
            "all_explored_procurement_closed": procurement_all.get(
                root_smiles, False
            ),
            "any_route_closed": any_closed.get(root_smiles, False),
            "all_explored_graph_closed": all_closed.get(root_smiles, False),
        }
        for field, expected in expected_summary_closure.items():
            if summary.get(field) is not expected:
                reasons.append(f"frontier_ledger_summary_closure_mismatch:{field}")
        expected_counts = {
            "reachable_molecule_count": len(molecules),
            "reachable_edge_count": len(edges),
            "reaction_proven_edge_count": sum(proof_closed.values()),
            "stock_closed_molecule_count": sum(stock_closed.values()),
            "procurement_boundary_closed_molecule_count": sum(
                procurement_closed.values()
            ),
        }
        for field, expected in expected_counts.items():
            if summary.get(field) != expected:
                reasons.append(f"frontier_ledger_summary_{field}_invalid")
        return sorted(set(reasons))

    def _frontier_ledger_summary_candidate(self) -> tuple[dict[str, Any], str]:
        graph, graph_source = self._frontier_authority_route_consensus_graph()
        caller_graph = self.blackboard.get("route_consensus_graph")
        team = dict(self.blackboard.get("codex_agent_team") or {})
        campaign = dict(team.get("campaign") or {})
        rows = [
            (f"{graph_source}.frontier_ledger_summary", graph.get("frontier_ledger_summary")),
            ("route_consensus_graph.frontier_ledger_summary", (caller_graph or {}).get("frontier_ledger_summary") if isinstance(caller_graph, dict) else None),
            ("blackboard.frontier_ledger_summary", self.blackboard.get("frontier_ledger_summary")),
            ("codex_agent_team.frontier_ledger_summary", team.get("frontier_ledger_summary")),
            ("codex_agent_team.campaign.frontier_ledger_summary", campaign.get("frontier_ledger_summary")),
        ]
        for source, value in rows:
            if isinstance(value, dict) and value:
                return dict(value), source
        return {}, ""

    def _frontier_ledger_candidates(
        self,
        summary: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        graph, graph_source = self._frontier_authority_route_consensus_graph()
        caller_graph = self.blackboard.get("route_consensus_graph")
        team = dict(self.blackboard.get("codex_agent_team") or {})
        campaign = dict(team.get("campaign") or {})
        candidates: list[tuple[str, dict[str, Any]]] = []
        embedded = [
            (f"{graph_source}.frontier_ledger", graph.get("frontier_ledger")),
            ("route_consensus_graph.frontier_ledger", (caller_graph or {}).get("frontier_ledger") if isinstance(caller_graph, dict) else None),
            ("blackboard.frontier_ledger", self.blackboard.get("frontier_ledger")),
            ("codex_agent_team.frontier_ledger", team.get("frontier_ledger")),
            ("codex_agent_team.campaign.frontier_ledger", campaign.get("frontier_ledger")),
        ]
        refs: list[tuple[str, Any]] = [
            ("frontier_ledger_summary.source_ref", summary.get("source_ref")),
            ("frontier_ledger_summary.frontier_ledger_ref", summary.get("frontier_ledger_ref")),
            ("blackboard.frontier_ledger_ref", self.blackboard.get("frontier_ledger_ref")),
            ("codex_agent_team.frontier_ledger_ref", team.get("frontier_ledger_ref")),
            ("codex_agent_team.campaign.frontier_ledger_ref", campaign.get("frontier_ledger_ref")),
        ]
        refs.extend(
            (f"artifact_refs.{key}", value)
            for key, value in sorted((self.blackboard.get("artifact_refs") or {}).items())
            if "frontier_ledger" in str(key).lower() or "frontier_ledger" in str(value).lower()
        )
        seen_paths: set[str] = set()
        for source, value in refs:
            path = self._resolve_artifact_path(value)
            path_key = str(path or "")
            if not path_key or path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            payload = self._read_json_dict(path)
            if payload:
                candidates.append((path_key, payload))
        # The persisted, summary-bound artifact is the preferred projection
        # source.  Embedded copies are portability fallbacks and still have
        # to match the same summary digest and target identity.
        for source, value in embedded:
            if isinstance(value, dict) and value:
                candidates.append((source, dict(value)))
        return candidates

    def _current_frontier_authority_context(self) -> dict[str, Any]:
        """Resolve only host-supplied current inputs; never infer authority."""

        graph, graph_source = self._frontier_authority_route_consensus_graph()
        team = dict(self.blackboard.get("codex_agent_team") or {})
        campaign = dict(team.get("campaign") or {})

        queue = (
            dict(campaign.get("frontier_queue") or {})
            if isinstance(campaign.get("frontier_queue"), Mapping)
            else {}
        )
        if not queue:
            queue = (
                dict(self.blackboard.get("frontier_queue") or {})
                if isinstance(self.blackboard.get("frontier_queue"), Mapping)
                else {}
            )
        if not queue:
            for ref in (
                campaign.get("frontier_queue_ref"),
                team.get("frontier_queue_ref"),
                self.blackboard.get("frontier_queue_ref"),
            ):
                queue = self._read_json_dict(self._resolve_artifact_path(ref))
                if queue:
                    break

        policy = (
            dict(campaign.get("campaign_policy") or {})
            if isinstance(campaign.get("campaign_policy"), Mapping)
            else {}
        )
        if not policy:
            policy = (
                dict(team.get("campaign_policy") or {})
                if isinstance(team.get("campaign_policy"), Mapping)
                else {}
            )
        if not policy:
            for ref in (
                campaign.get("campaign_policy_ref"),
                team.get("campaign_policy_ref"),
                self.blackboard.get("campaign_policy_ref"),
            ):
                policy = self._read_json_dict(self._resolve_artifact_path(ref))
                if policy:
                    break
        declared_policy_digests = {
            str(value)
            for value in (
                campaign.get("campaign_policy_sha256"),
                team.get("campaign_policy_sha256"),
                self.blackboard.get("campaign_policy_sha256"),
            )
            if str(value or "")
        }
        declared_policy_digest = (
            next(iter(declared_policy_digests))
            if len(declared_policy_digests) == 1
            else "invalid-conflicting-campaign-policy-digests"
            if declared_policy_digests
            else ""
        )
        return {
            "route_consensus_graph": graph,
            "route_consensus_graph_source": graph_source,
            "frontier_queue": queue,
            "campaign_policy": policy,
            "campaign_policy_sha256": declared_policy_digest,
        }

    def _frontier_authority_route_consensus_graph(
        self,
    ) -> tuple[dict[str, Any], str]:
        """Prefer the durable union graph; caller fusion is display fallback."""

        direct = self.blackboard.get("canonical_route_consensus_graph")
        if isinstance(direct, Mapping) and direct.get("schema_version") == (
            "route_consensus_graph.v1"
        ):
            return dict(direct), "canonical_route_consensus_graph"
        reconciliation = self.blackboard.get(
            "codex_campaign_proof_reconciliation"
        )
        if isinstance(reconciliation, Mapping):
            nested = reconciliation.get("canonical_route_consensus_graph")
            if isinstance(nested, Mapping) and nested.get("schema_version") == (
                "route_consensus_graph.v1"
            ):
                return (
                    dict(nested),
                    "codex_campaign_proof_reconciliation.canonical_route_consensus_graph",
                )
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        canonical_ref = artifact_refs.get("canonical_route_consensus_graph")
        canonical = self._read_json_dict(
            self._resolve_artifact_path(canonical_ref)
        )
        if canonical.get("schema_version") == "route_consensus_graph.v1":
            return canonical, "artifact_refs.canonical_route_consensus_graph"
        caller = self.blackboard.get("route_consensus_graph")
        if isinstance(caller, Mapping):
            return dict(caller), "caller_advisory_route_consensus_graph"
        return {}, "missing_route_consensus_graph"

    def _parent_route_proof_state(self) -> dict[str, bool]:
        parent_rows = [
            branch
            for branch in self.branches
            if branch.get("solved") is True
            and branch.get("executable") is True
            and branch.get("advisory_only") is False
            and branch.get("not_parent_route_proof") is False
        ]
        ranks = [
            PROOF_TIER_RANK.get(
                str((branch.get("trust_vector") or {}).get("proof_tier") or ""),
                0,
            )
            for branch in parent_rows
        ]
        return {
            "l3_parent_solved": any(
                rank >= PROOF_TIER_RANK["L3_precedent_supported"] for rank in ranks
            ),
            "l4_procurement_ready": any(
                rank >= PROOF_TIER_RANK["L4_procurement_ready"] for rank in ranks
            ),
        }

    def _display_policy(self) -> dict[str, Any]:
        return {
            "schema_version": "route_forest_display_policy.v1",
            "default_overview_top_k": 12,
            "default_group_visible_count": 5,
            "ranking": [
                "verified_parent_route",
                "complete_portfolio",
                "stock_closure",
                "reaction_proof_tier",
                "confidence",
                "route_depth",
            ],
            "semantics": "remaining branches are collapsed for display only and stay available on demand",
        }

    def _artifact_revision_view(self) -> dict[str, Any]:
        closeout = dict(self.blackboard.get("closeout_revision") or {})
        digest_refs = {
            str(key): dict(value)
            for key, value in (self.blackboard.get("artifact_digest_refs") or {}).items()
            if isinstance(value, dict)
        }
        committed = bool(
            closeout.get("accepted") is True
            and str(closeout.get("status") or "") == "committed"
            and str(closeout.get("revision_id") or "")
            and str(closeout.get("manifest_sha256") or "")
        )
        if closeout:
            status = (
                "source_context_committed"
                if committed
                else "source_context_not_committed"
            )
        else:
            status = "source_context_unavailable"
        return {
            "schema_version": "route_forest_source_revision_context.v1",
            "status": status,
            "scope": "blackboard_input_closeout_context",
            "committed": committed,
            "self_authenticates_current_forest": False,
            "revision_id": str(closeout.get("revision_id") or ""),
            "manifest_path": str(closeout.get("manifest_path") or ""),
            "manifest_sha256": str(closeout.get("manifest_sha256") or ""),
            "authority": str(closeout.get("authority") or ""),
            "artifact_count": int(closeout.get("artifact_count") or len(digest_refs)),
            "digest_ref_count": len(digest_refs),
            "semantics": (
                "source closeout context only; an external manifest must bind "
                "this forest and rendered delivery"
            ),
        }

    def _limited_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        category: str,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        total = len(rows)
        if limit is None:
            selected = list(rows)
            normalized_limit: int | None = None
        else:
            normalized_limit = max(0, int(limit))
            selected = list(rows[:normalized_limit])
        rendered = len(selected)
        self._projection_coverage[category] = {
            "available_count": total,
            "rendered_count": rendered,
            "omitted_count": max(0, total - rendered),
            "limit": normalized_limit,
            "truncated": rendered < total,
        }
        return selected

    def _finalize_trust_vectors(self) -> None:
        branch_by_id = {str(branch.get("branch_id") or ""): branch for branch in self.branches}
        for step in self.steps.values():
            branch = branch_by_id.get(str(step.get("branch_id") or ""), {})
            step["trust_vector"] = self._step_trust_vector(step, branch)
            step["visual_encoding"] = dict(step["trust_vector"]["visual_encoding"])

        for branch in self.branches:
            vectors = [
                dict((self.steps.get(str(step_id)) or {}).get("trust_vector") or {})
                for step_id in branch.get("step_ids") or []
            ]
            vectors = [row for row in vectors if row]
            if not vectors:
                continue
            numeric_fields = (
                "identity",
                "connectivity",
                "source_independence",
                "stock",
                "conditions",
                "forward_feasibility",
            )
            weakest_tier = min(
                (str(row.get("proof_tier") or "L0_advisory") for row in vectors),
                key=lambda value: PROOF_TIER_RANK.get(value, 0),
            )
            branch_vector = {
                "schema_version": "route_trust_vector.v1",
                **{
                    field: round(min(float(row.get(field) or 0.0) for row in vectors), 3)
                    for field in numeric_fields
                },
                "proof_tier": weakest_tier,
                "aggregation": "weakest_link",
            }
            minimum_support_groups = min(
                (int(row.get("support_group_count") or 0) for row in vectors),
                default=0,
            )
            corroborated_edge_count = sum(
                int(row.get("support_group_count") or 0) >= 2 for row in vectors
            )
            branch_vector.update(
                {
                    "min_trusted_source_group_count_across_steps": minimum_support_groups,
                    "corroborated_edge_count": corroborated_edge_count,
                    "all_edges_corroborated": bool(
                        vectors and corroborated_edge_count == len(vectors)
                    ),
                }
            )
            branch_vector["bottleneck_score"] = min(
                float(branch_vector[field]) for field in numeric_fields
            )
            branch_vector["visual_encoding"] = self._trust_visual_encoding(
                branch_vector,
                support_group_count=minimum_support_groups,
            )
            branch["trust_vector"] = branch_vector

    def _step_trust_vector(self, step: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
        interface_ids = _dedupe(
            [
                *[str(node_id) for node_id in step.get("from_node_ids") or []],
                *[str(node_id) for node_id in step.get("to_node_ids") or []],
            ]
        )
        structured_count = sum(
            bool((self.nodes.get(node_id) or {}).get("canonical_isomeric_smiles")) for node_id in interface_ids
        )
        if interface_ids and structured_count == len(interface_ids):
            identity = 1.0
            identity_status = "exact_structured"
        elif structured_count:
            identity = 0.6
            identity_status = "partially_structured"
        elif interface_ids:
            identity = 0.2
            identity_status = "name_only"
        else:
            identity = 0.0
            identity_status = "missing"

        has_inputs = bool(step.get("from_node_ids"))
        has_outputs = bool(step.get("to_node_ids"))
        branch_kind = str(branch.get("kind") or "")
        exactness = str(step.get("exactness") or "")
        origin = str(step.get("origin") or "")
        if exactness == "failed_or_unresolved" or not (has_inputs and has_outputs):
            connectivity = 0.0
            connectivity_status = "rejected_or_incomplete"
        elif branch_kind in {
            "direct_verified_route",
            "stitched_verified_route",
            "subgoal_verified_route",
            "proof_eligible_portfolio_route",
            "validated_replacement_route",
        }:
            connectivity = 1.0
            connectivity_status = "deterministically_checked"
        else:
            connectivity = 0.65
            connectivity_status = "explicit_interface_only"

        edge_binding_set = _validated_edge_evidence_binding_set(
            step.get("edge_evidence_binding_set")
        )
        if edge_binding_set:
            support_groups = _dedupe(
                [
                    str(value)
                    for value in edge_binding_set.get(
                        "independent_trusted_source_groups"
                    )
                    or []
                    if str(value)
                ]
            )
        else:
            support_groups = _dedupe(
                [
                    str(value)
                    for value in step.get("independent_support_groups") or []
                    if _authoritative_support_group(value)
                ]
            )
        support_group_count = len(support_groups)
        source_independence = min(1.0, 0.5 * support_group_count)
        if support_group_count == 0 and origin.startswith("direct_verified"):
            source_independence = 0.25

        if branch_kind in {"direct_verified_route", "stitched_verified_route"}:
            stock = 1.0
            stock_status = "parent_route_stock_closed"
        elif branch_kind == "subgoal_verified_route":
            stock = 0.85
            stock_status = "subgoal_stock_closed_only"
        elif branch_kind in {
            "proof_eligible_portfolio_route",
            "validated_replacement_route",
        }:
            stock = 1.0
            stock_status = "portfolio_stock_leaves_bound"
        else:
            stock = 0.15
            stock_status = "not_verified"

        condition_status = str(step.get("condition_status") or "not_recorded")
        if condition_status == "available":
            conditions = 0.9
        elif condition_status == "conflicting":
            conditions = 0.0
        elif condition_status in {"not_shown", "not_compiled"}:
            conditions = 0.2
        else:
            conditions = 0.1

        reaction_proof = dict(step.get("reaction_step_proof") or {})
        proof_level_map = {
            "L0_materialized": "L0_materialized",
            "L1_graph_and_stock_closed": "L1_graph_stock_closed",
            "L1_graph_stock_closed": "L1_graph_stock_closed",
            "L2_mapping_consistent": "L2_mapping_consistent",
            "L2_reaction_validated": "L2_reaction_validated",
            "L3_precedent_supported": "L3_precedent_supported",
            "L4_procurement_ready": "L4_procurement_ready",
        }
        authoritative_proof_tier = proof_level_map.get(str(reaction_proof.get("proof_level") or ""), "")
        if reaction_proof.get("proof_source") != "deterministic_reverified_route":
            authoritative_proof_tier = ""

        if exactness == "failed_or_unresolved":
            forward_feasibility = 0.0
        elif authoritative_proof_tier == "L4_procurement_ready":
            forward_feasibility = 1.0
            conditions = 1.0
            stock = 1.0
            stock_status = "procurement_bound"
        elif authoritative_proof_tier == "L3_precedent_supported":
            forward_feasibility = 0.95
        elif authoritative_proof_tier == "L2_reaction_validated":
            forward_feasibility = 0.9
        elif authoritative_proof_tier == "L2_mapping_consistent":
            forward_feasibility = 0.75
        elif exactness == "exact_literature_row" or origin == "stitched_verified_literature_chain":
            # Exact precedent without a deterministic atom-mapped reaction
            # proof remains useful evidence, but cannot skip directly to L3.
            forward_feasibility = 0.65
        elif branch_kind in {"direct_verified_route", "stitched_verified_route", "subgoal_verified_route"}:
            # The current deterministic verifier proves graph/stock closure,
            # not a universal reaction-forward simulation.
            forward_feasibility = 0.55
        elif origin == "process_evidence":
            forward_feasibility = 0.45
        elif exactness == "visual_inferred":
            forward_feasibility = 0.3
        else:
            forward_feasibility = 0.2

        if exactness == "failed_or_unresolved":
            proof_tier = "L0_rejected"
        elif authoritative_proof_tier:
            proof_tier = authoritative_proof_tier
        elif branch_kind in {"direct_verified_route", "stitched_verified_route", "subgoal_verified_route"}:
            proof_tier = "L1_graph_stock_closed"
        elif identity == 1.0:
            proof_tier = "L0_materialized"
        else:
            proof_tier = "L0_advisory"

        numeric = {
            "identity": identity,
            "connectivity": connectivity,
            "source_independence": source_independence,
            "stock": stock,
            "conditions": conditions,
            "forward_feasibility": forward_feasibility,
        }
        vector: dict[str, Any] = {
            "schema_version": "route_trust_vector.v1",
            **{key: round(value, 3) for key, value in numeric.items()},
            "proof_tier": proof_tier,
            "bottleneck_score": round(min(numeric.values()), 3),
            "status": {
                "identity": identity_status,
                "connectivity": connectivity_status,
                "stock": stock_status,
                "conditions": condition_status,
                "forward_feasibility": (
                    str(reaction_proof.get("proof_level") or "deterministically_reaction_validated")
                    if authoritative_proof_tier in {
                        "L2_mapping_consistent",
                        "L2_reaction_validated",
                        "L3_precedent_supported",
                        "L4_procurement_ready",
                    }
                    else (
                        "precedent_without_L2_reaction_validation"
                        if exactness == "exact_literature_row"
                        else "not_universally_proven"
                    )
                ),
            },
            "support_group_count": support_group_count,
            "independent_support_groups": support_groups,
            "edge_evidence_binding_set": edge_binding_set,
            "reaction_step_proof": reaction_proof,
        }
        vector["visual_encoding"] = self._trust_visual_encoding(
            vector,
            support_group_count=support_group_count,
        )
        return vector

    def _trust_visual_encoding(
        self,
        trust_vector: dict[str, Any],
        *,
        support_group_count: int,
    ) -> dict[str, Any]:
        tier = str(trust_vector.get("proof_tier") or "L0_advisory")
        style = dict(PROOF_TIER_STYLE.get(tier) or PROOF_TIER_STYLE["L0_advisory"])
        dimensions = [
            float(trust_vector.get(field) or 0.0)
            for field in (
                "identity",
                "connectivity",
                "source_independence",
                "stock",
                "conditions",
                "forward_feasibility",
            )
        ]
        certainty = sum(dimensions) / len(dimensions)
        return {
            "color": style["color"],
            "width": round(1.5 + min(4, max(0, support_group_count)) * 0.75, 2),
            "opacity": round(0.35 + 0.65 * certainty, 2),
            "dash_pattern": style["dash_pattern"],
            "texture": style["texture"],
            "width_semantics": "independent_support_group_count",
            "opacity_semantics": "mean_trust_dimension",
            "texture_semantics": "proof_tier_and_uncertainty",
        }

    def _replacement_validation(self) -> dict[str, Any]:
        diagnostics = self._interface_replacement_diagnostics()
        rows = [dict(row) for row in self._portfolio_replacement_records]
        by_base: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            base_step_id = str(row.get("base_step_id") or "")
            if base_step_id:
                by_base.setdefault(base_step_id, []).append(row)
        for step_id, step in self.steps.items():
            candidates = by_base.get(str(step_id), [])
            step["replacement_candidate_ids"] = [
                str(row.get("candidate_step_id") or row.get("replacement_id") or "")
                for row in candidates
                if str(row.get("candidate_step_id") or row.get("replacement_id") or "")
            ]
            step["validated_replacement_ids"] = [
                str(row.get("candidate_step_id") or "")
                for row in candidates
                if row.get("validated") is True and str(row.get("candidate_step_id") or "")
            ]
            step["replacement_rejection_count"] = sum(
                row.get("validated") is not True for row in candidates
            )
        validated_count = sum(row.get("validated") is True for row in rows)
        return {
            "schema_version": "route_replacement_validation.v1",
            "validation_engine": "and_or.validate_route_replacement",
            "candidate_count": len(rows),
            "validated_count": validated_count,
            "rejected_count": len(rows) - validated_count,
            "records": rows,
            "interface_diagnostics": diagnostics,
            "semantics": {
                "acceptance": "backend AND/OR route re-solve with connectivity, stock, and reaction proof",
                "interface_diagnostics_only": True,
                "single_step_splicing_forbidden": True,
                "preview_only": True,
                "invalid_candidates_are_not_replaceable": True,
            },
        }

    def _interface_replacement_diagnostics(self) -> dict[str, Any]:
        """Preserve pairwise interface comparison as non-authoritative diagnostics."""

        branch_by_step: dict[str, dict[str, Any]] = {}
        for branch in self.branches:
            for step_id in branch.get("step_ids") or []:
                branch_by_step[str(step_id)] = branch

        rows: list[dict[str, Any]] = []
        all_steps = list(self.steps.values())
        step_by_id = {str(step.get("step_id") or ""): step for step in all_steps}
        ids_by_module: dict[str, set[str]] = {}
        ids_by_output: dict[tuple[str, ...], set[str]] = {}
        non_replaceable_generic_modules = {
            "",
            "other",
            "other_route_module",
            "diagnostic_failure",
            "visual_failed_or_empty",
        }
        for step in all_steps:
            step_id = str(step.get("step_id") or "")
            module_key = str(step.get("module_key") or "")
            output_key = tuple(sorted(str(value) for value in step.get("to_node_ids") or []))
            if module_key not in non_replaceable_generic_modules:
                ids_by_module.setdefault(module_key, set()).add(step_id)
            if output_key:
                ids_by_output.setdefault(output_key, set()).add(step_id)
        for base in all_steps:
            base_id = str(base.get("step_id") or "")
            base_branch = branch_by_step.get(base_id, {})
            base_inputs = sorted(str(value) for value in base.get("from_node_ids") or [])
            base_outputs = sorted(str(value) for value in base.get("to_node_ids") or [])
            base_module = str(base.get("module_key") or "")
            potential_ids = set(ids_by_module.get(base_module, set()))
            potential_ids.update(ids_by_output.get(tuple(base_outputs), set()))
            for candidate_id in sorted(potential_ids):
                candidate = step_by_id.get(candidate_id) or {}
                if not candidate_id or candidate_id == base_id:
                    continue
                candidate_branch = branch_by_step.get(candidate_id, {})
                if str(candidate_branch.get("branch_id") or "") == str(base_branch.get("branch_id") or ""):
                    continue
                candidate_inputs = sorted(str(value) for value in candidate.get("from_node_ids") or [])
                candidate_outputs = sorted(str(value) for value in candidate.get("to_node_ids") or [])
                same_module = bool(base_module and base_module == str(candidate.get("module_key") or ""))
                same_output_ids = bool(base_outputs and base_outputs == candidate_outputs)
                if not (same_module or same_output_ids):
                    continue

                product_identity_exact = bool(base_outputs) and all(
                    bool((self.nodes.get(node_id) or {}).get("canonical_isomeric_smiles"))
                    for node_id in [*base_outputs, *candidate_outputs]
                )
                input_identity_exact = bool(base_inputs) and all(
                    bool((self.nodes.get(node_id) or {}).get("canonical_isomeric_smiles"))
                    for node_id in [*base_inputs, *candidate_inputs]
                )
                product_interface_match = product_identity_exact and base_outputs == candidate_outputs
                reactant_interface_match = input_identity_exact and base_inputs == candidate_inputs
                candidate_complete = bool(candidate_inputs and candidate_outputs)
                candidate_not_rejected = str(candidate.get("exactness") or "") != "failed_or_unresolved"
                interface_compatible = all(
                    [
                        product_interface_match,
                        reactant_interface_match,
                        candidate_complete,
                        candidate_not_rejected,
                    ]
                )
                reasons = _dedupe(
                    [
                        "product_interface_not_exact" if not product_identity_exact else "",
                        "product_interface_mismatch" if product_identity_exact and not same_output_ids else "",
                        "reactant_interface_not_exact" if not input_identity_exact else "",
                        "reactant_interface_mismatch" if input_identity_exact and base_inputs != candidate_inputs else "",
                        "candidate_interface_incomplete" if not candidate_complete else "",
                        "candidate_rejected_or_unresolved" if not candidate_not_rejected else "",
                    ]
                )
                row = {
                    "diagnostic_id": f"interface-diagnostic:{_slug(base_id)}:{_slug(candidate_id)}",
                    "base_step_id": base_id,
                    "candidate_step_id": candidate_id,
                    "base_branch_id": str(base_branch.get("branch_id") or ""),
                    "candidate_branch_id": str(candidate_branch.get("branch_id") or ""),
                    "status": "interface_compatible" if interface_compatible else "interface_mismatch",
                    "interface_compatible": interface_compatible,
                    "validated": False,
                    "same_module": same_module,
                    "product_interface_match": product_interface_match,
                    "reactant_interface_match": reactant_interface_match,
                    "downstream_interface_preserved": product_interface_match,
                    "exact_product_node_ids": base_outputs if product_interface_match else [],
                    "exact_reactant_node_ids": base_inputs if reactant_interface_match else [],
                    "reasons": reasons,
                    "validation_scope": "diagnostic_exact_molecule_graph_interface_only",
                    "diagnostics_only": True,
                    "preview_enabled": False,
                    "does_not_establish_parent_route_proof": True,
                }
                rows.append(row)
        return {
            "schema_version": "route_interface_diagnostics.v1",
            "candidate_count": len(rows),
            "interface_compatible_count": sum(
                row.get("interface_compatible") is True for row in rows
            ),
            "records": rows,
            "authority": "diagnostics_only_not_replacement_validation",
        }

    def _dependency_graph(self) -> dict[str, Any]:
        molecule_nodes = [
            {
                "graph_node_id": f"graph:molecule:{node_id}",
                "node_type": "molecule",
                "molecule_node_id": node_id,
                "label": str(node.get("label") or node_id),
                "role": str(node.get("role") or "intermediate"),
                "canonical_isomeric_smiles": str(node.get("canonical_isomeric_smiles") or ""),
                "structure_svg": str(node.get("structure_svg") or ""),
                "formula": str(node.get("formula") or ""),
            }
            for node_id, node in self.nodes.items()
        ]
        reaction_nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        hyperedges: list[dict[str, Any]] = []
        for step in self.steps.values():
            step_id = str(step.get("step_id") or "")
            reaction_graph_id = f"graph:reaction:{step_id}"
            trust_vector = dict(step.get("trust_vector") or {})
            visual_encoding = dict(trust_vector.get("visual_encoding") or {})
            reaction_nodes.append(
                {
                    "graph_node_id": reaction_graph_id,
                    "node_type": "reaction",
                    "reaction_step_id": step_id,
                    "branch_id": str(step.get("branch_id") or ""),
                    "label": str(step.get("label") or step_id),
                    "portfolio_route_id": str(step.get("portfolio_route_id") or ""),
                    "canonical_hyperedge_id": str(
                        step.get("portfolio_hyperedge_id") or ""
                    ),
                    "proof_tier": str(trust_vector.get("proof_tier") or "L0_advisory"),
                    "trust_vector": trust_vector,
                    "visual_encoding": visual_encoding,
                }
            )
            input_edge_ids: list[str] = []
            output_edge_ids: list[str] = []
            for index, node_id in enumerate(step.get("from_node_ids") or [], start=1):
                edge_id = f"edge:{_slug(step_id)}:input:{index}:{_slug(node_id)}"
                input_edge_ids.append(edge_id)
                edges.append(
                    {
                        "edge_id": edge_id,
                        "edge_type": "molecule_to_reaction",
                        "source_graph_node_id": f"graph:molecule:{node_id}",
                        "target_graph_node_id": reaction_graph_id,
                        "molecule_node_id": str(node_id),
                        "reaction_step_id": step_id,
                        "branch_id": str(step.get("branch_id") or ""),
                        "trust_vector": trust_vector,
                        "visual_encoding": visual_encoding,
                    }
                )
            for index, node_id in enumerate(step.get("to_node_ids") or [], start=1):
                edge_id = f"edge:{_slug(step_id)}:output:{index}:{_slug(node_id)}"
                output_edge_ids.append(edge_id)
                edges.append(
                    {
                        "edge_id": edge_id,
                        "edge_type": "reaction_to_molecule",
                        "source_graph_node_id": reaction_graph_id,
                        "target_graph_node_id": f"graph:molecule:{node_id}",
                        "molecule_node_id": str(node_id),
                        "reaction_step_id": step_id,
                        "branch_id": str(step.get("branch_id") or ""),
                        "trust_vector": trust_vector,
                        "visual_encoding": visual_encoding,
                    }
                )
            hyperedges.append(
                {
                    "hyperedge_id": f"hyperedge:{_slug(step_id)}",
                    "reaction_step_id": step_id,
                    "reaction_graph_node_id": reaction_graph_id,
                    "branch_id": str(step.get("branch_id") or ""),
                    "portfolio_route_id": str(step.get("portfolio_route_id") or ""),
                    "canonical_hyperedge_id": str(
                        step.get("portfolio_hyperedge_id") or ""
                    ),
                    "input_molecule_node_ids": [str(value) for value in step.get("from_node_ids") or []],
                    "output_molecule_node_ids": [str(value) for value in step.get("to_node_ids") or []],
                    "input_edge_ids": input_edge_ids,
                    "output_edge_ids": output_edge_ids,
                    "trust_vector": trust_vector,
                    "visual_encoding": visual_encoding,
                }
            )

        graph_nodes = [*molecule_nodes, *reaction_nodes]
        layout = _dependency_layout(graph_nodes, edges)
        for node in graph_nodes:
            node["layer"] = int(layout["layers"].get(str(node.get("graph_node_id") or ""), 0))
        branch_views = [self._branch_dependency_view(branch) for branch in self.branches]
        return {
            "schema_version": "molecule_reaction_dependency_graph.v1",
            "graph_kind": "molecule_reaction_bipartite_hypergraph",
            "direction": "reactants_to_reaction_to_products",
            "molecule_nodes": molecule_nodes,
            "reaction_nodes": reaction_nodes,
            "nodes": graph_nodes,
            "edges": edges,
            "hyperedges": hyperedges,
            "branch_views": branch_views,
            "acyclic": bool(layout["acyclic"]),
            "cycle_graph_node_ids": list(layout["cycle_node_ids"]),
            "layout_semantics": "layers are derived only from explicit molecule-reaction edges",
            "no_array_adjacency_edges": True,
            "proof_tier_legend": [
                {
                    "proof_tier": tier,
                    "rank": rank,
                    **dict(PROOF_TIER_STYLE[tier]),
                }
                for tier, rank in sorted(PROOF_TIER_RANK.items(), key=lambda item: item[1], reverse=True)
            ],
        }

    def _branch_dependency_view(self, branch: dict[str, Any]) -> dict[str, Any]:
        step_ids = [str(value) for value in branch.get("step_ids") or [] if str(value) in self.steps]
        step_set = set(step_ids)
        producers: dict[str, list[str]] = {}
        for step_id in step_ids:
            for node_id in (self.steps.get(step_id) or {}).get("to_node_ids") or []:
                producers.setdefault(str(node_id), []).append(step_id)
        dependencies: list[dict[str, str]] = []
        for consumer_id in step_ids:
            for node_id in (self.steps.get(consumer_id) or {}).get("from_node_ids") or []:
                for producer_id in producers.get(str(node_id), []):
                    if producer_id == consumer_id or producer_id not in step_set:
                        continue
                    dependencies.append(
                        {
                            "producer_step_id": producer_id,
                            "consumer_step_id": consumer_id,
                            "molecule_node_id": str(node_id),
                        }
                    )
        dependency_pairs = _dedupe(
            [
                f"{row['producer_step_id']}\u0000{row['consumer_step_id']}"
                for row in dependencies
            ]
        )
        order, acyclic = _topological_step_order(step_ids, dependency_pairs)
        consumed = {
            str(node_id)
            for step_id in step_ids
            for node_id in (self.steps.get(step_id) or {}).get("from_node_ids") or []
        }
        produced = {
            str(node_id)
            for step_id in step_ids
            for node_id in (self.steps.get(step_id) or {}).get("to_node_ids") or []
        }
        synthesis_leaves = sorted(consumed - produced)
        synthesis_targets = sorted(produced - consumed)
        stock_aliases = sorted(
            {
                str(value)
                for value in branch.get("stock_terminal_node_ids") or []
                if str(value)
            }
        )
        target_alias = str(branch.get("root_molecule_node_id") or "")
        target_aliases = [target_alias] if target_alias else synthesis_targets
        support_groups = sorted(
            {
                str(value)
                for value in branch.get("independent_support_groups") or []
                if str(value)
            }
        )
        return {
            "branch_id": str(branch.get("branch_id") or ""),
            "step_ids": step_ids,
            "topological_step_ids": order,
            "dependencies": dependencies,
            "root_molecule_node_ids": synthesis_leaves,
            "terminal_molecule_node_ids": synthesis_targets,
            "stock_leaf_molecule_node_ids": stock_aliases or synthesis_leaves,
            "target_molecule_node_ids": target_aliases,
            "all_leaves_stock_bound": bool(synthesis_leaves)
            and set(synthesis_leaves) == set(stock_aliases),
            "portfolio_route_id": str(branch.get("portfolio_route_id") or ""),
            "weakest_proof_tier": str(
                branch.get("weakest_proof_tier")
                or (branch.get("trust_vector") or {}).get("proof_tier")
                or ""
            ),
            "independent_support_groups": support_groups,
            "diversity_score": float(branch.get("diversity_score") or 0.0),
            "portfolio_solver_truncated": bool(
                (branch.get("portfolio_enumeration") or {}).get("solver_truncated")
            ),
            "acyclic": acyclic,
            "dependency_semantics": "producer/consumer links require an explicit shared molecule node",
        }

    def add_direct_verified_route_branch(self) -> None:
        route_result = self._best_direct_route_result()
        if not route_result:
            return
        route = dict(route_result.get("route") or {})
        route_steps = [dict(row) for row in route.get("steps") or [] if isinstance(row, dict)]
        if not route_steps:
            return
        branch_id = "branch:direct_verified_chemenzy_route"
        proof_binding = dict(route_result.get("proof_binding") or {})
        proof_bound = proof_binding.get("accepted") is True
        target = self._target()
        target_smiles = str(target.get("smiles") or "")
        source_refs = _dedupe(
            [
                str(route_result.get("source_ref") or "ChemEnzy route verifier"),
                str(route_result.get("artifact_path") or ""),
            ]
        )
        rendered_steps = _forward_synthesis_step_order(
            route_steps,
            target_smiles=target_smiles,
        )
        if not rendered_steps:
            return
        step_ids: list[str] = []
        reaction_step_proofs = [
            dict(value)
            for value in (route_result.get("reaction_validation") or {}).get("step_proofs") or []
            if isinstance(value, dict)
        ]
        for index, row in enumerate(rendered_steps, start=1):
            product = _route_step_product(row)
            reactants = _route_step_reactants(row)
            if not product and not reactants:
                continue
            product_label = self._route_smiles_label(product, role="product", target_smiles=target_smiles)
            from_nodes = [
                self._add_node(
                    self._route_smiles_label(smiles, role=f"precursor {idx}", target_smiles=target_smiles),
                    role="verified_route_precursor",
                    smiles=smiles if _looks_like_smiles(smiles) else "",
                    exactness="model_hypothesis",
                    confidence=_route_step_confidence(row),
                    source_refs=source_refs,
                    missing=[],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                    ),
                )
                for idx, smiles in enumerate(reactants, start=1)
                if str(smiles or "").strip()
            ]
            to_nodes = [
                self._add_node(
                    product_label,
                    role="target" if _same_molecule(product, target_smiles) else "verified_route_intermediate",
                    smiles=product if _looks_like_smiles(product) else "",
                    exactness="model_hypothesis",
                    confidence=_route_step_confidence(row),
                    source_refs=source_refs,
                    missing=[] if product else ["product missing from ChemEnzy step"],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                    ),
                )
            ]
            label = _clean_label(
                row.get("reaction_type")
                or (row.get("reaction_interpretation") or {}).get("reaction_class")
                or f"ChemEnzy verified step {index}"
            )
            if label.lower() in {"template", "reaction", "step"}:
                label = f"ChemEnzy verified step {index}"
            step_id = self._add_step(
                    branch_id=branch_id,
                    label=label,
                    from_nodes=from_nodes,
                    to_nodes=to_nodes,
                    module_key=_module_key_for_text(
                        " ".join(
                            [
                                label,
                                str((row.get("reaction_interpretation") or {}).get("forward_summary") or ""),
                                str(row.get("reaction_smiles") or ""),
                            ]
                        )
                    ),
                    module_label=_module_label_for_key(_module_key_for_text(label)),
                    confidence=_route_step_confidence(row),
                    exactness="model_hypothesis",
                    source_refs=source_refs,
                    origin="direct_verified_chemenzy_route",
                    summary=str((row.get("reaction_interpretation") or {}).get("forward_summary") or row.get("source") or "ChemEnzy route verifier accepted this step in a solved parent route."),
                    conditions=_conditions_from_row(row),
                    missing=_dedupe(
                        [
                            "computational/template route, not an exact literature row",
                            *[str(x) for x in ((row.get("reaction_interpretation") or {}).get("atom_change") or {}).get("notes") or []],
                        ]
                    )[:8],
                )
            reaction_proof = _matching_reaction_step_proof(reaction_step_proofs, row)
            if reaction_proof:
                self.steps[step_id]["reaction_step_proof"] = {
                    **reaction_proof,
                    "proof_source": "deterministic_reverified_route",
                }
            step_ids.append(step_id)
        if not step_ids:
            return
        self._add_branch(
            branch_id=branch_id,
            title=f"Direct verified route: {target.get('name') or 'target'}",
            kind="direct_verified_route",
            recommendation="verified route" if proof_bound else "revalidated advisory route",
            confidence="high",
            summary=(
                "This route is reconstructed from the accepted deterministic parent proof."
                if proof_bound
                else "The guided artifact was independently replayed, but it is not bound to an accepted parent proof."
            ),
            step_ids=step_ids,
            source_refs=source_refs,
            missing=["Not a literature exact-row route", "Conditions may be template-level unless separately predicted"],
            classification_records=[
                *[dict(row) for row in route_result.get("classification_records") or [] if isinstance(row, dict)],
                route,
            ],
            proof_binding=proof_binding,
        )

    def add_stitched_verified_route_branch(self) -> None:
        """Render the revalidated stock-to-terminal-to-target proof DAG.

        The display is rebuilt exclusively from the stitched proof's immutable
        ``proof_inputs``.  Top-level route summaries, display bindings, and
        other blackboard branches are deliberately outside this trust boundary.
        """
        target = self._target()
        target_smiles = str(target.get("smiles") or "").strip()
        projection = _revalidated_stitched_proof_projection(
            self.blackboard.get("parent_route_proof"),
            expected_target_smiles=target_smiles,
        )
        if not projection:
            return

        branch_id = "branch:stitched_verified_parent_route"
        literature_frontiers = [
            str(smiles)
            for smiles in projection.get("literature_frontier_smiles") or []
            if str(smiles or "").strip()
        ]
        literature_frontier_keys = {
            _canonical_molecule_smiles(smiles) for smiles in literature_frontiers
        }
        stock_terminals = {
            _canonical_molecule_smiles(smiles)
            for smiles in projection.get("stock_terminal_smiles") or []
            if _canonical_molecule_smiles(smiles)
        }
        subgoal_source_refs = ["deterministic:chemenzy-route-verifier"]
        literature_source_refs = _dedupe(
            [
                str(projection.get("literature_source_ref") or ""),
                *[
                    str(ref)
                    for row in projection.get("literature_steps") or []
                    for ref in [
                        row.get("source_ref"),
                        *(row.get("evidence_refs") or []),
                    ]
                    if str(ref or "").strip()
                ],
            ]
        )
        step_ids: list[str] = []
        subgoal_step_ids: list[str] = []
        subgoal_segments: list[dict[str, Any]] = []
        literature_step_ids: list[str] = []
        stock_terminal_node_ids: list[str] = []
        literature_terminal_node_ids: list[str] = []

        for segment_index, segment in enumerate(projection.get("subgoal_segments") or [], start=1):
            segment_frontier = str(segment.get("frontier_smiles") or "")
            segment_step_ids: list[str] = []
            segment_reaction_proofs = [
                dict(value)
                for value in (segment.get("reaction_validation") or {}).get("step_proofs") or []
                if isinstance(value, dict)
            ]
            for route_step_index, row in enumerate(segment.get("steps") or [], start=1):
                product = _route_step_product(row)
                reactants = _route_step_reactants(row)
                from_nodes: list[str] = []
                for precursor_index, smiles in enumerate(reactants, start=1):
                    canonical = _canonical_molecule_smiles(smiles)
                    is_stock = canonical in stock_terminals
                    node_id = self._add_node(
                        self._route_smiles_label(
                            smiles,
                            role=f"stock precursor {precursor_index}" if is_stock else f"route precursor {precursor_index}",
                            target_smiles=target_smiles,
                        ),
                        role="stock_terminal" if is_stock else "stitched_route_intermediate",
                        smiles=smiles,
                        exactness="model_hypothesis",
                        confidence=_route_step_confidence(row),
                        source_refs=subgoal_source_refs,
                        missing=[],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=subgoal_source_refs,
                            evidence_row_id=f"subgoal-{segment_index}-step-{route_step_index}",
                        ),
                    )
                    from_nodes.append(node_id)
                    if is_stock:
                        stock_terminal_node_ids.append(node_id)
                product_is_terminal = _same_molecule(product, segment_frontier)
                product_node = self._add_node(
                    f"literature terminal {segment_index}"
                    if product_is_terminal
                    else self._route_smiles_label(product, role="route intermediate", target_smiles=target_smiles),
                    role="literature_terminal" if product_is_terminal else "stitched_route_intermediate",
                    smiles=product,
                    exactness="model_hypothesis",
                    confidence=_route_step_confidence(row),
                    source_refs=subgoal_source_refs,
                    missing=[],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=subgoal_source_refs,
                        evidence_row_id=f"subgoal-{segment_index}-step-{route_step_index}",
                    ),
                )
                if product_is_terminal:
                    literature_terminal_node_ids.append(product_node)
                label = _clean_label(
                    row.get("reaction_type")
                    or (row.get("reaction_interpretation") or {}).get("reaction_class")
                    or f"Verified stock closure {segment_index}.{route_step_index}"
                )
                step_id = self._add_step(
                    branch_id=branch_id,
                    label=label,
                    from_nodes=from_nodes,
                    to_nodes=[product_node],
                    module_key=f"stitched_stock_closure:{segment_index:02d}:{route_step_index:02d}",
                    module_label=f"Verified stock closure {segment_index}.{route_step_index}",
                    confidence=_route_step_confidence(row),
                    exactness="model_hypothesis",
                    source_refs=subgoal_source_refs,
                    origin="stitched_verified_subgoal_route",
                    summary=str(
                        (row.get("reaction_interpretation") or {}).get("forward_summary")
                        or "The deterministic route verifier accepted this stock-closure step."
                    ),
                    conditions=_conditions_from_row(row),
                    missing=["Computational route step; not an exact literature row"],
                )
                step_ids.append(step_id)
                subgoal_step_ids.append(step_id)
                segment_step_ids.append(step_id)
                reaction_proof = _matching_reaction_step_proof(segment_reaction_proofs, row)
                if reaction_proof:
                    self.steps[step_id]["reaction_step_proof"] = {
                        **reaction_proof,
                        "proof_source": "deterministic_reverified_route",
                    }
            subgoal_segments.append(
                {
                    "segment_id": f"verified_stock_closure_{segment_index}",
                    "frontier_smiles": segment_frontier,
                    "step_ids": segment_step_ids,
                    "status": "deterministically_verified",
                }
            )

        for index, row in enumerate(projection.get("literature_steps") or [], start=1):
            product = _route_step_product(row)
            reactants = _route_step_reactants(row)
            row_source_refs = _dedupe(
                [
                    str(row.get("source_ref") or ""),
                    *[str(ref) for ref in row.get("evidence_refs") or []],
                ]
            )
            from_nodes: list[str] = []
            for precursor_index, smiles in enumerate(reactants, start=1):
                is_terminal = _canonical_molecule_smiles(smiles) in literature_frontier_keys
                node_id = self._add_node(
                    "literature terminal"
                    if is_terminal
                    else self._route_smiles_label(
                        smiles,
                        role=f"literature precursor {precursor_index}",
                        target_smiles=target_smiles,
                    ),
                    role="literature_terminal" if is_terminal else "literature_intermediate",
                    smiles=smiles,
                    exactness="exact_literature_row",
                    confidence="high",
                    source_refs=row_source_refs,
                    missing=[],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=row_source_refs,
                        evidence_row_id=str(row.get("step_id") or index),
                    ),
                )
                from_nodes.append(node_id)
                if is_terminal:
                    literature_terminal_node_ids.append(node_id)
            product_is_target = _same_molecule(product, target_smiles)
            product_node = self._add_node(
                str(target.get("name") or "target")
                if product_is_target
                else self._route_smiles_label(product, role="literature intermediate", target_smiles=target_smiles),
                role="target" if product_is_target else "literature_intermediate",
                smiles=product,
                exactness="exact_literature_row",
                confidence="high",
                source_refs=row_source_refs,
                missing=[],
                identity_namespace=_molecule_identity_namespace(
                    branch_id=branch_id,
                    source_refs=row_source_refs,
                    evidence_row_id=str(row.get("step_id") or index),
                ),
            )
            label = _clean_label(
                row.get("reaction_class")
                or row.get("step_label")
                or row.get("step_id")
                or f"Exact literature step {index}"
            )
            module_key = _module_key_for_text(
                " ".join([label, str(row.get("reaction_smiles") or "")])
            )
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=from_nodes,
                to_nodes=[product_node],
                module_key=module_key,
                module_label=_module_label_for_key(module_key),
                confidence="high",
                exactness="exact_literature_row",
                source_refs=row_source_refs,
                origin="stitched_verified_literature_chain",
                summary="This exact literature edge was revalidated against its source-detail evidence binding.",
                conditions=_conditions_from_row(row),
                missing=[],
            )
            step_ids.append(step_id)
            literature_step_ids.append(step_id)

        if (
            not subgoal_step_ids
            or not literature_step_ids
            or len(set(literature_terminal_node_ids)) != len(literature_frontier_keys)
        ):
            return
        proof_route_steps = [
            *[
                dict(step)
                for segment in projection.get("subgoal_segments") or []
                for step in segment.get("steps") or []
                if isinstance(step, dict)
            ],
            *[
                dict(step)
                for step in projection.get("literature_steps") or []
                if isinstance(step, dict)
            ],
        ]
        proof_route_digest = _route_structure_sha256(proof_route_steps)
        self._add_branch(
            branch_id=branch_id,
            title=f"Stitched verified route: {target.get('name') or 'target'}",
            kind="stitched_verified_route",
            recommendation="deterministically verified stitched route",
            confidence="high",
            summary=(
                "A single revalidated synthesis DAG connects every stock terminal through the "
                "verified subgoal closure and strict source-detail literature chain to the target."
            ),
            step_ids=step_ids,
            source_refs=_dedupe([*subgoal_source_refs, *literature_source_refs]),
            missing=[
                "Stock-closure steps are computational unless independently replaced by exact literature rows"
            ],
            classification_records=[{"synthesis_class": "semisynthesis"}],
            proof_binding={
                "schema_version": "route_forest_parent_proof_binding.v1",
                "accepted": bool(proof_route_digest),
                "proof_mode": "stitched_parent_route",
                "route_structure_sha256": proof_route_digest,
                "binding_source": "proof_evidence.stitched_route.proof_inputs_replay",
            },
        )
        branch = self.branches[-1]
        branch["route_direction"] = "stock_to_literature_terminal_to_target"
        branch["stock_terminal_node_ids"] = _dedupe(stock_terminal_node_ids)
        branch["literature_terminal_node_ids"] = _dedupe(literature_terminal_node_ids)
        branch["segments"] = [
            *subgoal_segments,
            {
                "segment_id": "strict_literature_chain",
                "step_ids": literature_step_ids,
                "status": "source_detail_exact",
            },
        ]

    def add_subgoal_verified_route_branches(self) -> None:
        for index, record in enumerate(self._subgoal_route_records(), start=1):
            route = dict(record.get("route") or {})
            route_steps = [dict(row) for row in route.get("steps") or [] if isinstance(row, dict)]
            subgoal_name = _clean_label(record.get("name") or f"subgoal {index}")
            target_smiles = str(record.get("target_smiles") or "").strip()
            step_source_refs = _dedupe(
                [
                    "route_expansion_subgoal_search_result",
                    str(record.get("search_path") or ""),
                    str(record.get("raw_path") or ""),
                ]
            )
            branch_source_refs = _dedupe(
                [
                    *step_source_refs,
                    *[str(x) for x in record.get("evidence_refs") or [] if str(x).strip()],
                ]
            )
            branch_id = f"branch:subgoal_verified_route:{_slug(subgoal_name)}:{index}"

            def route_label(value: str, *, role: str) -> str:
                text = str(value or "").strip()
                if target_smiles and _same_text(text, target_smiles):
                    return f"{subgoal_name} terminal"
                if not text:
                    return role
                if _looks_like_smiles(text):
                    return _compact_smiles_label(text)
                return _clean_label(text)

            step_ids: list[str] = []
            rendered_steps = list(reversed(route_steps))
            reaction_step_proofs = [
                dict(value)
                for value in (record.get("reaction_validation") or {}).get("step_proofs") or []
                if isinstance(value, dict)
            ]
            for step_index, row in enumerate(rendered_steps, start=1):
                product = str(row.get("product") or "").strip()
                reactants = _route_step_reactants(row)
                if not product and not reactants:
                    continue
                from_nodes = [
                    self._add_node(
                        route_label(smiles, role=f"subgoal precursor {idx}"),
                        role="subgoal_route_precursor",
                        smiles=smiles if _looks_like_smiles(smiles) else "",
                        exactness="model_hypothesis",
                        confidence=_route_step_confidence(row),
                        source_refs=step_source_refs,
                        missing=[],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=step_source_refs,
                            evidence_row_id=f"route-step-{step_index}",
                        ),
                    )
                    for idx, smiles in enumerate(reactants, start=1)
                    if str(smiles or "").strip()
                ]
                to_nodes = [
                    self._add_node(
                        route_label(product, role="subgoal product"),
                        role="subgoal_literature_terminal" if target_smiles and _same_text(product, target_smiles) else "subgoal_route_intermediate",
                        smiles=product if _looks_like_smiles(product) else "",
                        exactness="model_hypothesis",
                        confidence=_route_step_confidence(row),
                        source_refs=step_source_refs,
                        missing=[] if product else ["product missing from ChemEnzy subgoal step"],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=step_source_refs,
                            evidence_row_id=f"route-step-{step_index}",
                        ),
                    )
                ]
                label = _clean_label(
                    row.get("reaction_type")
                    or (row.get("reaction_interpretation") or {}).get("reaction_class")
                    or f"ChemEnzy 子目标闭合 step {step_index}"
                )
                if label.lower() in {"template", "reaction", "step", "chemenzyretroplanner"}:
                    label = f"ChemEnzy 子目标闭合 step {step_index}"
                step_id = self._add_step(
                        branch_id=branch_id,
                        label=label,
                        from_nodes=from_nodes,
                        to_nodes=to_nodes,
                        module_key=f"subgoal_stock_closure:{step_index:02d}",
                        module_label=f"ChemEnzy 子目标闭合 {step_index}",
                        confidence=_route_step_confidence(row),
                        exactness="model_hypothesis",
                        source_refs=step_source_refs,
                        origin="subgoal_verified_chemenzy_route",
                        summary=str(
                            (row.get("reaction_interpretation") or {}).get("forward_summary")
                            or row.get("source")
                            or "ChemEnzy route expansion closed the upstream stock-to-literature-terminal subgoal."
                        ),
                        conditions=_conditions_from_row(row),
                        missing=_dedupe(
                            [
                                "子目标闭合路线：支持 stitched final route，但本身不是最终目标路线",
                                "计算/模板路线，不是 exact literature row",
                                *[str(x) for x in ((row.get("reaction_interpretation") or {}).get("atom_change") or {}).get("notes") or []],
                            ]
                        )[:8],
                    )
                reaction_proof = _matching_reaction_step_proof(reaction_step_proofs, row)
                if reaction_proof:
                    self.steps[step_id]["reaction_step_proof"] = {
                        **reaction_proof,
                        "proof_source": "deterministic_reverified_route",
                    }
                step_ids.append(step_id)
            if not step_ids and target_smiles:
                target_node = self._add_node(
                    f"{subgoal_name} terminal",
                    role="subgoal_literature_terminal",
                    smiles=target_smiles,
                    exactness="model_hypothesis",
                    confidence="high" if record.get("accepted") else "medium",
                    source_refs=branch_source_refs,
                    missing=[],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=branch_source_refs,
                        evidence_row_id="subgoal-terminal",
                    ),
                )
                step_ids.append(
                    self._add_step(
                        branch_id=branch_id,
                        label=f"ChemEnzy 子目标闭合：{subgoal_name}",
                        from_nodes=[
                            self._add_node(
                                "ChemEnzy stock closure route pool",
                                role="subgoal_route_pool",
                                exactness="model_hypothesis",
                                confidence="medium",
                                source_refs=step_source_refs,
                                identity_namespace=_molecule_identity_namespace(
                                    branch_id=branch_id,
                                    source_refs=step_source_refs,
                                    evidence_row_id="subgoal-route-pool",
                                ),
                                missing=["route forest 编译器未加载到 raw ChemEnzy steps"],
                            )
                        ],
                        to_nodes=[target_node],
                        module_key="subgoal_stock_closure",
                        module_label=_module_label_for_key("subgoal_stock_closure"),
                        confidence="high" if record.get("accepted") else "medium",
                        exactness="model_hypothesis",
                        source_refs=step_source_refs,
                        origin="subgoal_verified_chemenzy_route",
                        summary="ChemEnzy route expansion 接受了 stock-to-literature-terminal 子目标闭合。",
                        conditions=[],
                        missing=["只有路线级子目标证明；未加载逐步 raw route"],
                    )
                )
            if not step_ids:
                continue
            verifier_reasons = [str(x) for x in record.get("reasons") or [] if str(x).strip()]
            route_rank = record.get("route_rank")
            self._add_branch(
                branch_id=branch_id,
                title=f"ChemEnzy 子目标闭合：{subgoal_name}",
                kind="subgoal_verified_route",
                recommendation="子目标闭合审计",
                confidence="high" if record.get("accepted") else "medium",
                summary=(
                    "ChemEnzy 已闭合 stitched parent proof 使用的上游 stock-to-literature-terminal 片段。"
                    "它作为独立 advisory 分支展示；只有 proof inputs 重验通过时才会另建拼接验证路线。"
                ),
                step_ids=step_ids,
                source_refs=branch_source_refs,
                missing=_dedupe(
                    [
                        "只支持上游 child target / literature terminal 片段",
                        "不是 literature exact-row 路线",
                        f"best route rank: {route_rank}" if route_rank is not None else "",
                        *[f"verifier noted route-pool issue: {reason}" for reason in verifier_reasons[:4]],
                    ]
                ),
                classification_records=[record, route],
            )

    def add_diagnostic_failure_branch_if_empty(self) -> None:
        if self.branches:
            return
        failures = [dict(row) for row in self.blackboard.get("route_failures") or [] if isinstance(row, dict)]
        diagnostics = [row for row in self._guided_result_artifacts() if isinstance(row.get("chemenzy_runtime_diagnostic"), dict)]
        if not failures and not diagnostics:
            self._add_unclosed_exploration_branch_if_empty()
            return
        reasons = _dedupe(
            [
                *[str(row.get("reason") or row.get("failure_class") or "") for row in failures],
                *[
                    str(reason)
                    for row in diagnostics
                    for reason in (row.get("chemenzy_runtime_diagnostic") or {}).get("reasons") or []
                ],
            ]
        )
        branch_id = "branch:diagnostic_unresolved_route"
        step_id = self._add_step(
            branch_id=branch_id,
            label="No accepted route produced",
            from_nodes=[
                self._add_node(
                    "ChemEnzy / planner diagnostic",
                    role="diagnostic_source",
                    exactness="failed_or_unresolved",
                    confidence="failed",
                    source_refs=[str(row.get("artifact_ref") or "") for row in failures if str(row.get("artifact_ref") or "").strip()],
                    missing=reasons or ["route unresolved"],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=[
                            str(row.get("artifact_ref") or "")
                            for row in failures
                            if str(row.get("artifact_ref") or "").strip()
                        ],
                        evidence_row_id="diagnostic",
                    ),
                )
            ],
            to_nodes=[self._target_node(identity_namespace=branch_id)],
            module_key="diagnostic_failure",
            module_label="Diagnostic failure",
            confidence="failed",
            exactness="failed_or_unresolved",
            source_refs=[str(row.get("artifact_ref") or "") for row in failures if str(row.get("artifact_ref") or "").strip()],
            origin="route_failure_diagnostic",
            summary="The run reached a route-search diagnostic instead of a displayable synthesis route.",
            missing=reasons or ["no displayable route branch"],
        )
        self._add_branch(
            branch_id=branch_id,
            title=f"Unresolved diagnostic: {self._target().get('name') or 'target'}",
            kind="diagnostic_failure",
            recommendation="needs rerun or repair",
            confidence="failed",
            summary="No route branch was available; this panel preserves the failure reason instead of showing a blank route.",
            step_ids=[step_id],
            source_refs=[str(row.get("artifact_ref") or "") for row in failures if str(row.get("artifact_ref") or "").strip()],
            missing=reasons or ["no displayable route branch"],
        )

    def _primary_selection(self) -> dict[str, Any]:
        """Select a real compiled branch without manufacturing route chemistry."""
        if not self.branches:
            return {
                "schema_version": "route_forest_primary_selection.v1",
                "primary_branch_id": "",
                "status": "unavailable",
                "proof_level": "none",
                "advisory_only": True,
                "selection_ambiguous": False,
                "display_tiebreak_only": False,
                "tied_candidate_count": 0,
                "tied_candidate_ids": [],
                "reasons": ["no_compiled_branch"],
            }
        priority = {
            "stitched_verified_route": 80,
            "direct_verified_route": 70,
            "proof_eligible_portfolio_route": 65,
            "exact_literature": 60,
            "subgoal_verified_route": 50,
            "route_consensus_graph": 45,
            "process_evidence": 40,
            "visual_chain": 35,
            "route_consensus": 30,
            "literature_candidate": 25,
            "retrosynthetic_proposal": 20,
            "broad_template": 10,
            "diagnostic_failure": 0,
            "validated_replacement_route": -2,
        }
        selectable = [row for row in self.branches if row.get("listed") is not False]
        candidate_pool = selectable or self.branches

        def selection_score(row: dict[str, Any]) -> tuple[int, int, int]:
            return (
                priority.get(str(row.get("kind") or ""), -1),
                CONFIDENCE_RANK.get(str(row.get("confidence") or ""), 0),
                len(row.get("step_ids") or []),
            )

        top_score = max(selection_score(row) for row in candidate_pool)
        tied_candidates = sorted(
            (
                str(row.get("branch_id") or "")
                for row in candidate_pool
                if selection_score(row) == top_score
            )
        )
        selected = max(
            candidate_pool,
            key=lambda row: (
                *selection_score(row),
                str(row.get("branch_id") or ""),
            ),
        )
        selected_id = str(selected.get("branch_id") or "")
        for branch in self.branches:
            branch["is_primary"] = bool(selected_id and branch.get("branch_id") == selected_id)
        kind = str(selected.get("kind") or "")
        if (
            kind in {"stitched_verified_route", "direct_verified_route"}
            and selected.get("solved") is True
            and selected.get("executable") is True
            and selected.get("advisory_only") is False
        ):
            status = "deterministically_verified"
            proof_level = "parent_route_proof"
            advisory_only = False
        elif kind in {"stitched_verified_route", "direct_verified_route"}:
            status = "advisory"
            proof_level = "replayed_candidate_without_parent_proof_authority"
            advisory_only = True
        elif kind == "exact_literature":
            status = "evidence_backed"
            proof_level = "literature_rows"
            advisory_only = True
        elif kind == "proof_eligible_portfolio_route":
            status = "proof_eligible_portfolio"
            proof_level = str(selected.get("weakest_proof_tier") or "L2_reaction_validated")
            advisory_only = True
        elif kind == "diagnostic_failure":
            status = "diagnostic"
            proof_level = "none"
            advisory_only = True
        else:
            status = "advisory"
            proof_level = "route_hint"
            advisory_only = True
        selection_ambiguous = len(tied_candidates) > 1
        display_tiebreak_only = selection_ambiguous and advisory_only
        reasons = [
            f"selected_from_compiled_branch_kind:{kind or 'unknown'}",
            "no_target_name_route_injection",
        ]
        if selection_ambiguous:
            reasons.extend(
                [
                    f"same_top_rank_candidates:{len(tied_candidates)}",
                    "lexical_branch_id_tiebreak_for_display_only",
                ]
            )
        return {
            "schema_version": "route_forest_primary_selection.v1",
            "primary_branch_id": selected_id,
            "status": status,
            "proof_level": proof_level,
            "advisory_only": advisory_only,
            "synthesis_class": str(selected.get("synthesis_class") or "unspecified"),
            "selection_ambiguous": selection_ambiguous,
            "display_tiebreak_only": display_tiebreak_only,
            "tied_candidate_count": len(tied_candidates),
            "tied_candidate_ids": tied_candidates[:24],
            "reasons": reasons,
        }

    def _add_unclosed_exploration_branch_if_empty(self) -> None:
        if self.branches:
            return
        candidate_records = [
            dict(row)
            for row in self.evidence.get("source_candidates") or []
            if isinstance(row, dict)
        ]
        candidates = [row for row in candidate_records if _source_candidate_has_real_source(row)]
        placeholder_candidates = [
            row for row in candidate_records if not _source_candidate_has_real_source(row)
        ]
        pdf_rows = [dict(row) for row in self.evidence.get("pdf_structure_evidence") or [] if isinstance(row, dict)]
        visual_rows = [dict(row) for row in self.evidence.get("visual_chains") or [] if isinstance(row, dict)]
        exact_rows = [dict(row) for row in self.evidence.get("exact_rows") or [] if isinstance(row, dict)]
        actions = [dict(row) for row in self.blackboard.get("action_history") or [] if isinstance(row, dict)]
        if not any(
            (
                candidates,
                placeholder_candidates,
                pdf_rows,
                visual_rows,
                exact_rows,
                actions,
                self.blackboard.get("current_belief"),
            )
        ):
            return
        source_refs = _dedupe(
            [
                str(row.get("source_ref") or row.get("doi") or row.get("title") or row.get("url") or "")
                for row in candidates[:8]
            ]
        )
        status_bits = [
            f"source candidates: {len(candidates)}" if candidates else "",
            f"placeholder source queries: {len(placeholder_candidates)}"
            if placeholder_candidates
            else "",
            f"PDF evidence rows: {len(pdf_rows)}" if pdf_rows else "",
            f"visual chains: {len(visual_rows)}" if visual_rows else "",
            f"exact rows: {len(exact_rows)}" if exact_rows else "",
            f"actions recorded: {len(actions)}" if actions else "",
        ]
        missing = _dedupe(
            [
                "no displayable route branch was compiled",
                "no deterministic parent-route proof",
                *[bit for bit in status_bits if bit],
            ]
        )
        branch_id = "branch:unclosed_exploration_state"
        step_id = self._add_step(
            branch_id=branch_id,
            label="Exploration recorded, no route branch",
            from_nodes=[
                self._add_node(
                    "Blackboard exploration state",
                    role="diagnostic_source",
                    exactness="failed_or_unresolved",
                    confidence="low",
                    source_refs=source_refs,
                    missing=missing,
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                        evidence_row_id="unclosed-exploration",
                    ),
                )
            ],
            to_nodes=[self._target_node(identity_namespace=branch_id)],
            module_key="diagnostic_failure",
            module_label="Diagnostic / incomplete exploration",
            confidence="low",
            exactness="failed_or_unresolved",
            source_refs=source_refs,
            origin="unclosed_blackboard_exploration",
            summary="The run recorded planning or evidence activity, but no exact, visual, process, proposal, template, or verified route branch was available.",
            missing=missing,
        )
        self._add_branch(
            branch_id=branch_id,
            title=f"Exploration incomplete: {self._target().get('name') or 'target'}",
            kind="diagnostic_failure",
            recommendation="needs more evidence or rerun",
            confidence="low",
            summary="The route forest preserves the non-empty blackboard state instead of showing a blank route.",
            step_ids=[step_id],
            source_refs=source_refs,
            missing=missing,
        )

    def add_visual_branches(self, *, limit: int | None) -> None:
        chains = [row for row in self.evidence.get("visual_chains") or [] if isinstance(row, dict)]
        selected = self._limited_rows(chains, category="visual_chains", limit=limit)
        for index, chain in enumerate(selected, start=1):
            source_ref = str(chain.get("source_ref") or chain.get("source_title") or f"visual:{index}")
            title = str(chain.get("source_title") or source_ref or f"Visual chain {index}")
            branch_id = self._unique_branch_id(f"branch:visual:{_slug(source_ref or title)}:{index}")
            chain_steps = chain.get("steps") or chain.get("candidate_steps") or []
            step_ids: list[str] = []
            if chain_steps:
                for row_index, row in enumerate([x for x in chain_steps if isinstance(x, dict)], start=1):
                    step_ids.append(self._visual_step(branch_id, chain, row, row_index))
            else:
                step_ids.append(
                    self._add_step(
                        branch_id=branch_id,
                        label="Visual extraction produced no route step",
                        from_nodes=[
                            self._add_node(
                                title,
                                role="source_placeholder",
                                exactness="failed_or_unresolved",
                                confidence="failed",
                                source_refs=[source_ref],
                                identity_namespace=_molecule_identity_namespace(
                                    branch_id=branch_id,
                                    source_refs=[source_ref],
                                    evidence_row_id="empty-visual-chain",
                                ),
                            )
                        ],
                        to_nodes=[],
                        module_key="visual_failed_or_empty",
                        module_label="视觉链失败或为空",
                        confidence="failed",
                        exactness="failed_or_unresolved",
                        source_refs=[source_ref],
                        origin="visual_chain",
                        summary=str(chain.get("rejection_reason") or "No candidate step was accepted from this source."),
                        missing=["No displayable route step"],
                    )
                )
            self._add_branch(
                branch_id=branch_id,
                title=f"文献图像分支：{title}",
                kind=(
                    "diagnostic_failure"
                    if chain.get("accepted") is False or not chain_steps
                    else "visual_chain"
                ),
                recommendation="支持/备选",
                confidence="low" if chain_steps else "failed",
                summary="Steps inferred from rendered literature figures. These are useful route hints, not exact proof.",
                step_ids=step_ids,
                source_refs=[source_ref],
                missing=["image-derived structures may be incomplete", "stereochemistry may be partial"],
                classification_records=[chain],
            )

    def add_process_evidence_branches(self, *, limit: int | None = None) -> None:
        rows = [row for row in self.evidence.get("process_evidence_rows") or [] if isinstance(row, dict)]
        selected = self._limited_rows(rows, category="process_evidence", limit=limit)
        for index, row in enumerate(selected, start=1):
            endpoints = _labels_from_any(row.get("endpoint_labels")) or ["process endpoint"]
            substrates = _labels_from_any(row.get("substrate_or_feedstock_labels")) or ["process substrate/feedstock"]
            process_labels = _labels_from_any(row.get("biocatalyst_or_process_labels")) or [
                str(row.get("process_type") or "process")
            ]
            source_refs = _dedupe(
                [
                    str(row.get("source_ref") or ""),
                    str(row.get("source_title") or ""),
                    *[str(item) for item in row.get("evidence_refs") or []],
                ]
            )
            endpoint_text = " / ".join(endpoints[:3])
            substrate_text = " / ".join(substrates[:4])
            process_text = " / ".join(process_labels[:4])
            label = f"{substrate_text} via {process_text} to {endpoint_text}"
            branch_id = self._unique_branch_id(f"branch:process:{_slug(str(row.get('row_id') or index))}")
            module_key = _module_key_for_text(" ".join([label, str(row.get("process_type") or "")]))
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=[
                    self._add_node(
                        label=substrate,
                        role="process_substrate_or_feedstock",
                        exactness="name_only",
                        confidence=str(row.get("confidence") or "medium"),
                        source_refs=source_refs,
                        missing=["structure may be class/name-only in process evidence"],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=source_refs,
                            evidence_row_id=str(row.get("row_id") or index),
                        ),
                    )
                    for substrate in substrates[:6]
                ],
                to_nodes=[
                    self._add_node(
                        label=endpoint,
                        role="process_endpoint",
                        exactness="named_literature",
                        confidence=str(row.get("confidence") or "medium"),
                        source_refs=source_refs,
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=source_refs,
                            evidence_row_id=str(row.get("row_id") or index),
                        ),
                    )
                    for endpoint in endpoints[:4]
                ],
                module_key=module_key,
                module_label=_module_label_for_key(module_key),
                confidence=str(row.get("confidence") or "medium"),
                exactness="named_literature",
                source_refs=source_refs,
                origin="process_evidence",
                summary=str(row.get("summary") or label),
                conditions=_conditions_from_row(row),
                missing=_dedupe(
                    [
                        "process evidence is an advisory route anchor, not an exact reaction row",
                        *[str(item) for item in row.get("risk_flags") or []],
                    ]
                ),
            )
            self._add_branch(
                branch_id=branch_id,
                title=f"文献过程锚点：{endpoint_text}",
                kind="process_evidence",
                recommendation="文献锚点",
                confidence=str(row.get("confidence") or "medium"),
                summary=str(row.get("summary") or label),
                step_ids=[step_id],
                source_refs=source_refs,
                missing=[
                    "not a deterministic stock-closed route",
                    "not exact reaction SMILES",
                    "requires objective-specific verification",
                ],
                classification_records=[row],
            )

    def add_route_consensus_branches(self, *, limit: int | None) -> None:
        """Project canonical consensus proposals without promoting them to routes.

        A consensus proposal is one advisory reaction edge.  Its direct
        literature/evidence references stay on that edge; the consensus JSON
        artifact itself is retained separately as route-level provenance.
        """
        consensus = self._route_consensus_payload()
        proposals = [
            row
            for row in consensus.get("proposals") or []
            if isinstance(row, dict)
            and str(row.get("schema_version") or "") == "route_consensus_proposal.v1"
        ]
        proposals.sort(
            key=lambda row: (
                int(row.get("rank") or 1_000_000),
                -float(row.get("rank_score") or 0.0),
                str(row.get("consensus_id") or ""),
            )
        )
        represented_signatures = self._selected_graph_edge_signatures(limit=limit)
        merged_count = sum(
            _semantic_reaction_edge_key(
                str(row.get("product_smiles") or ""),
                _consensus_precursor_smiles(row),
            ) in represented_signatures
            for row in proposals
        )
        proposals = [
            row
            for row in proposals
            if _semantic_reaction_edge_key(
                str(row.get("product_smiles") or ""),
                _consensus_precursor_smiles(row),
            ) not in represented_signatures
        ]
        target = self._target()
        target_smiles = str(target.get("smiles") or "").strip()
        consensus_target_smiles = str(consensus.get("target_smiles") or "").strip()
        route_level_refs = self._route_consensus_route_refs()

        selected = self._limited_rows(proposals, category="route_consensus", limit=limit)
        self._projection_coverage["route_consensus"]["merged_into_graph_count"] = int(merged_count)
        for index, proposal in enumerate(selected, start=1):
            product_smiles = str(proposal.get("product_smiles") or "").strip()
            precursor_smiles = _consensus_precursor_smiles(proposal)
            if not product_smiles or not precursor_smiles:
                continue
            target_matches = (
                not target_smiles
                or _same_molecule(product_smiles, target_smiles)
                or bool(consensus_target_smiles and _same_text(product_smiles, consensus_target_smiles))
            )
            if not target_matches:
                # A canonical route_consensus.v1 producer already quarantines
                # this case.  Do not reconnect a malformed payload to target.
                continue

            consensus_id = str(proposal.get("consensus_id") or f"proposal-{index}")
            branch_id = f"branch:route_consensus:{_slug(consensus_id)}"
            direct_refs = _consensus_direct_source_refs(proposal)
            all_support_records = _consensus_support_records(proposal, limit=None)
            support_records = all_support_records[:32]
            support_groups = _consensus_independent_support_groups(
                proposal,
                all_support_records,
            )
            independent_source_count = len(support_groups)
            is_multi_source = independent_source_count > 1
            consensus_scope = (
                "multi_source" if is_multi_source else "correlated_single_source"
            )
            conflicts = _consensus_conflicts(proposal)
            evidence_level = str(proposal.get("evidence_level") or "model_only")
            confidence = str(proposal.get("confidence") or "low")
            reaction_family = _clean_label(proposal.get("reaction_family") or "unspecified transformation")
            exactness = "named_literature" if evidence_level == "literature_exact" else "model_hypothesis"
            missing = _dedupe(
                [
                    "advisory consensus only; not solved or executable",
                    "deterministic parent-route proof is still required",
                    *[str(item) for item in proposal.get("limitations") or []],
                    *[
                        f"required validation: {item}"
                        for item in proposal.get("required_validation") or []
                        if str(item).strip()
                    ],
                    "condition conflict requires review" if conflicts else "",
                ]
            )
            precursor_nodes = [
                self._add_node(
                    f"consensus precursor {precursor_index}",
                    role="consensus_precursor",
                    smiles=smiles,
                    exactness=exactness,
                    confidence=confidence,
                    source_refs=direct_refs,
                    missing=missing,
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=direct_refs,
                        evidence_row_id=consensus_id,
                    ),
                )
                for precursor_index, smiles in enumerate(precursor_smiles, start=1)
            ]
            product_node = self._add_node(
                str(target.get("name") or "consensus product"),
                role="target" if target_smiles else "consensus_product",
                smiles=product_smiles,
                exactness=exactness,
                confidence=confidence,
                source_refs=direct_refs,
                missing=missing,
                identity_namespace=_molecule_identity_namespace(
                    branch_id=branch_id,
                    source_refs=direct_refs,
                    evidence_row_id=consensus_id,
                ),
            )
            step_id = self._add_step(
                branch_id=branch_id,
                label=reaction_family,
                from_nodes=precursor_nodes,
                to_nodes=[product_node],
                module_key=_module_key_for_text(reaction_family),
                module_label=_module_label_for_key(_module_key_for_text(reaction_family)),
                confidence=confidence,
                exactness=exactness,
                source_refs=direct_refs,
                origin="route_consensus",
                summary=(
                    "Advisory multi-source retrosynthetic disconnection; it is not a parent-route proof."
                    if is_multi_source
                    else "Advisory consensus that does not establish multiple independent support groups; it is not a parent-route proof."
                ),
                conditions=_conditions_from_row(proposal),
                missing=missing,
            )
            consensus_metadata = {
                "consensus_id": consensus_id,
                "consensus_status": str(proposal.get("status") or "model_hypothesis"),
                "evidence_level": evidence_level,
                "source_channels": _dedupe([str(item) for item in proposal.get("source_channels") or []]),
                "support_records": support_records,
                "support_count": len(all_support_records),
                "support_record_count": len(all_support_records),
                "support_records_truncated": len(all_support_records) > len(support_records),
                "independent_support_groups": support_groups,
                "independent_source_count": independent_source_count,
                "consensus_scope": consensus_scope,
                "multi_source": is_multi_source,
                "codex_roles_correlated": any(group == "codex_model" for group in support_groups),
                "condition_support": [
                    dict(row) for row in proposal.get("condition_support") or [] if isinstance(row, dict)
                ],
                "conflicts": conflicts,
                "rank": int(proposal.get("rank") or index),
                "rank_score": float(proposal.get("rank_score") or 0.0),
                "advisory_only": True,
                "solved": False,
                "executable": False,
                "not_parent_route_proof": True,
            }
            self.steps[step_id].update(consensus_metadata)
            if conflicts:
                self.steps[step_id]["condition_status"] = "conflicting"

            self._add_branch(
                branch_id=branch_id,
                title=(
                    f"多信源共识 #{int(proposal.get('rank') or index)}：{reaction_family}"
                    if is_multi_source
                    else f"相关源候选 #{int(proposal.get('rank') or index)}：{reaction_family}"
                ),
                kind="route_consensus",
                recommendation="多信源建议" if is_multi_source else "未形成多信源",
                confidence=confidence,
                summary=(
                    "Multi-source candidate edge. Independent support is counted by correlated support group, not by Codex role."
                    if is_multi_source
                    else "Consensus candidate does not establish multiple independent support groups. Multiple Codex roles do not make it multi-source."
                ),
                step_ids=[step_id],
                source_refs=direct_refs,
                missing=missing,
                classification_records=[proposal],
            )
            actual_branch = self.branches[-1]
            actual_branch.update(consensus_metadata)
            actual_branch["route_level_source_refs"] = route_level_refs
            self._consensus_branch_ids[consensus_id] = str(actual_branch.get("branch_id") or branch_id)

    def _selected_graph_edge_signatures(self, *, limit: int | None) -> set[str]:
        graph = self._route_consensus_graph_payload()
        if not graph:
            return set()
        steps = {
            str(row.get("step_id") or ""): dict(row)
            for row in graph.get("steps") or []
            if isinstance(row, dict)
        }
        routes = [dict(row) for row in graph.get("route_hypotheses") or [] if isinstance(row, dict)]
        routes.sort(key=lambda row: (-float(row.get("rank_score") or 0.0), str(row.get("route_id") or "")))
        selected = routes if limit is None else routes[: max(0, int(limit))]
        signatures: set[str] = set()
        for route in selected:
            for step_id in route.get("forward_step_ids") or []:
                step = steps.get(str(step_id)) or {}
                if not step:
                    continue
                signature = _semantic_reaction_edge_key(
                    str(step.get("product_smiles") or ""),
                    [str(value) for value in step.get("precursor_smiles") or []],
                )
                if signature:
                    signatures.add(signature)
        return signatures

    def add_route_portfolio_branches(self) -> None:
        """Project every proof-eligible Top-K selection as its own closed DAG.

        The application layer remains the validation authority.  This method
        consumes its immutable portfolio, exact proof/stock bindings, and
        backend replacement catalog.  It never invents proof from advisory
        overlay scores and never splices a replacement step into a route.
        """

        graph = self._route_consensus_graph_payload()
        overlay = dict(graph.get("v2_overlay") or {})
        portfolio = dict(graph.get("route_portfolio") or {})
        bindings = dict(
            graph.get("route_portfolio_bindings")
            or self.blackboard.get("route_portfolio_bindings")
            or portfolio.get("bindings")
            or {}
        )
        if (
            portfolio.get("schema_version") != "route_portfolio.v1"
            or overlay.get("schema_version") != "route_hypergraph_overlay.v2"
            or (overlay.get("validation") or {}).get("valid") is not True
        ):
            return
        integrity_reasons: list[str] = []
        if not _portfolio_content_digest_valid(portfolio):
            integrity_reasons.append("route_portfolio_content_sha256_mismatch")
        if bindings.get("schema_version") != "route_portfolio_bindings.v1":
            integrity_reasons.append("invalid_route_portfolio_bindings_schema")
        if not _portfolio_content_digest_valid(bindings):
            integrity_reasons.append("route_portfolio_bindings_content_sha256_mismatch")
        if integrity_reasons:
            source_route_count = len(portfolio.get("routes") or [])
            self._projection_coverage["route_portfolio"] = {
                "available_count": source_route_count,
                "rendered_count": 0,
                "omitted_count": source_route_count,
                "limit": None,
                "truncated": bool(source_route_count),
            }
            self._portfolio_projection = {
                **self._portfolio_projection,
                "available": True,
                "source_route_count": source_route_count,
                "rejected_route_count": source_route_count,
                "reasons": integrity_reasons,
            }
            return

        context = {
            "overlay": overlay,
            "portfolio": portfolio,
            "bindings": bindings,
            "molecules": {
                str(row.get("molecule_id") or ""): dict(row)
                for row in overlay.get("molecules") or []
                if isinstance(row, dict) and str(row.get("molecule_id") or "")
            },
            "hyperedges": {
                str(row.get("hyperedge_id") or ""): dict(row)
                for row in overlay.get("reaction_hyperedges") or []
                if isinstance(row, dict) and str(row.get("hyperedge_id") or "")
            },
            "claims": {
                str(row.get("claim_id") or ""): dict(row)
                for row in overlay.get("evidence_claims") or []
                if isinstance(row, dict) and str(row.get("claim_id") or "")
            },
            "envelopes": {
                str(row.get("envelope_id") or ""): dict(row)
                for row in overlay.get("candidate_envelopes") or []
                if isinstance(row, dict) and str(row.get("envelope_id") or "")
            },
            "edge_evidence_binding_sets": _validated_edge_evidence_projection(
                graph
            ),
        }
        source_routes = [
            dict(row) for row in portfolio.get("routes") or [] if isinstance(row, dict)
        ]
        rejected: list[dict[str, Any]] = []
        for rank, route in enumerate(source_routes, start=1):
            branch_id, reasons = self._materialize_portfolio_route(
                route,
                context=context,
                rank=rank,
                kind="proof_eligible_portfolio_route",
                listed=True,
            )
            if not branch_id:
                rejected.append(
                    {
                        "route_id": str(route.get("route_id") or ""),
                        "reasons": reasons,
                    }
                )

        catalog = dict(
            graph.get("route_replacement_catalog")
            or self.blackboard.get("route_replacement_catalog")
            or portfolio.get("route_replacement_catalog")
            or portfolio.get("replacement_catalog")
            or {}
        )
        catalog_integrity_valid = bool(
            catalog.get("schema_version") == "route_replacement_catalog.v1"
            and _portfolio_content_digest_valid(catalog)
            and str(catalog.get("portfolio_content_sha256") or "")
            == str(portfolio.get("content_sha256") or "")
            and catalog.get("portfolio_integrity_valid") is True
        )
        preview_branch_ids: set[str] = set()
        replacement_records: list[dict[str, Any]] = []
        catalog_rows = (
            catalog.get("candidates") or catalog.get("records") or []
            if catalog_integrity_valid
            else []
        )
        for index, raw_record in enumerate(catalog_rows, start=1):
            if not isinstance(raw_record, dict):
                continue
            record = dict(raw_record)
            base_route_id = str(
                record.get("base_route_id")
                or record.get("base_portfolio_route_id")
                or ""
            )
            product_id = str(record.get("product_molecule_id") or "")
            base_branch_id = self._portfolio_branch_ids.get(base_route_id, "")
            base_step_id = self._portfolio_step_for_product(base_branch_id, product_id)
            accepted = bool(
                (record.get("accepted") is True or record.get("validated") is True)
                and record.get("connectivity_revalidated") is True
                and record.get("stock_closure_revalidated") is True
                and record.get("reaction_proof_revalidated") is True
            )
            result_branch_id = ""
            candidate_step_id = ""
            result_route = record.get("route")
            if accepted and isinstance(result_route, dict):
                result_route_id = str(result_route.get("route_id") or "")
                result_branch_id = self._portfolio_branch_ids.get(result_route_id, "")
                if not result_branch_id:
                    result_branch_id, materialization_reasons = self._materialize_portfolio_route(
                        dict(result_route),
                        context=context,
                        rank=index,
                        kind="validated_replacement_route",
                        listed=False,
                    )
                    if not result_branch_id:
                        accepted = False
                        record["reasons"] = _dedupe(
                            [
                                *[str(value) for value in record.get("reasons") or []],
                                *materialization_reasons,
                            ]
                        )
                    else:
                        preview_branch_ids.add(result_branch_id)
                candidate_step_id = self._portfolio_step_for_product(
                    result_branch_id,
                    product_id,
                )
                if not candidate_step_id:
                    accepted = False
                    record["reasons"] = _dedupe(
                        [
                            *[str(value) for value in record.get("reasons") or []],
                            "revalidated_route_missing_replacement_product_step",
                        ]
                    )
            replacement_records.append(
                {
                    **record,
                    "replacement_id": str(
                        record.get("replacement_id")
                        or record.get("candidate_id")
                        or f"portfolio-replacement:{_slug(base_route_id)}:{index}"
                    ),
                    "validation_engine": str(
                        record.get("validation_engine")
                        or "and_or.validate_route_replacement"
                    ),
                    "base_route_id": base_route_id,
                    "base_branch_id": base_branch_id,
                    "base_step_id": base_step_id,
                    "candidate_step_id": candidate_step_id,
                    "candidate_branch_id": result_branch_id,
                    "revalidated_route_branch_id": result_branch_id,
                    "accepted": accepted,
                    "validated": accepted,
                    "status": "route_revalidated" if accepted else "rejected",
                    "preview_only": True,
                    "does_not_establish_parent_route_proof": True,
                }
            )
        self._portfolio_replacement_records = replacement_records
        projected_ids = {
            route_id: branch_id
            for route_id, branch_id in self._portfolio_branch_ids.items()
            if (self._branch_by_id(branch_id) or {}).get("listed") is not False
        }
        self._projection_coverage["route_portfolio"] = {
            "available_count": len(source_routes),
            "rendered_count": len(projected_ids),
            "omitted_count": max(0, len(source_routes) - len(projected_ids)),
            "limit": None,
            "truncated": len(projected_ids) < len(source_routes),
        }
        self._portfolio_projection = {
            "schema_version": "route_portfolio_projection.v1",
            "available": True,
            "source_schema_version": str(portfolio.get("schema_version") or ""),
            "source_content_sha256": str(portfolio.get("content_sha256") or ""),
            "bindings_schema_version": str(bindings.get("schema_version") or ""),
            "source_route_count": len(source_routes),
            "projected_route_count": len(projected_ids),
            "rejected_route_count": len(rejected),
            "replacement_preview_branch_count": len(preview_branch_ids),
            "replacement_catalog_integrity_valid": catalog_integrity_valid,
            "solver_truncated": bool(portfolio.get("truncated")),
            "complete_candidate_count": int(portfolio.get("complete_candidate_count") or 0),
            "enumerated_candidate_count": int(portfolio.get("enumerated_candidate_count") or 0),
            "route_branch_ids": dict(sorted(projected_ids.items())),
            "rejected_routes": rejected,
            "reasons": [str(value) for value in portfolio.get("reasons") or []],
        }

    def _materialize_portfolio_route(
        self,
        route: dict[str, Any],
        *,
        context: dict[str, Any],
        rank: int,
        kind: str,
        listed: bool,
    ) -> tuple[str, list[str]]:
        route_id = str(route.get("route_id") or "")
        existing = self._portfolio_branch_ids.get(route_id, "")
        if existing:
            return existing, []
        overlay = dict(context.get("overlay") or {})
        portfolio = dict(context.get("portfolio") or {})
        bindings = dict(context.get("bindings") or {})
        molecules = dict(context.get("molecules") or {})
        hyperedges = dict(context.get("hyperedges") or {})
        claims = dict(context.get("claims") or {})
        envelopes = dict(context.get("envelopes") or {})
        edge_evidence_binding_sets = dict(
            context.get("edge_evidence_binding_sets") or {}
        )
        reasons: list[str] = []
        if route.get("schema_version") != "route_portfolio_item.v1":
            reasons.append("invalid_portfolio_route_schema")
        if not _portfolio_content_digest_valid(route):
            reasons.append("portfolio_route_content_sha256_mismatch")
        if route.get("complete") is not True:
            reasons.append("portfolio_route_not_complete")
        if route.get("reaction_validated") is not True:
            reasons.append("portfolio_route_not_reaction_validated")
        if not route_id:
            reasons.append("missing_portfolio_route_id")
        root_id = str(route.get("root_molecule_id") or overlay.get("root_molecule_id") or "")
        stock_ids = {str(value) for value in route.get("stock_terminal_ids") or [] if str(value)}
        selection_rows = [
            dict(row) for row in route.get("selected_hyperedges") or [] if isinstance(row, dict)
        ]
        edge_levels = dict(bindings.get("edge_proof_levels") or {})
        edge_bindings = dict(
            bindings.get("exact_edge_proof_bindings")
            or bindings.get("edge_proof_bindings")
            or {}
        )
        stock_bindings = dict(bindings.get("stock_bindings") or {})
        selected: list[tuple[str, str, dict[str, Any], int]] = []
        seen_products: set[str] = set()
        required_molecule_ids = {root_id, *stock_ids}
        for selection in selection_rows:
            product_id = str(selection.get("product_molecule_id") or "")
            edge_id = str(selection.get("hyperedge_id") or "")
            edge = dict(hyperedges.get(edge_id) or {})
            if not product_id or not edge_id or not edge:
                reasons.append(f"selected_hyperedge_missing:{edge_id or product_id}")
                continue
            if product_id in seen_products:
                reasons.append(f"duplicate_product_selection:{product_id}")
            seen_products.add(product_id)
            if str(edge.get("product_molecule_id") or "") != product_id:
                reasons.append(f"selected_hyperedge_product_mismatch:{edge_id}")
            binding = dict(edge_bindings.get(edge_id) or {})
            binding_reasons = _portfolio_edge_binding_reasons(
                binding,
                edge_id=edge_id,
                product_id=product_id,
                precursor_ids=[str(value) for value in edge.get("precursor_molecule_ids") or []],
            )
            reasons.extend(
                f"selected_hyperedge_binding:{edge_id}:{reason}"
                for reason in binding_reasons
            )
            level = _portfolio_proof_level(binding.get("portfolio_proof_level"))
            if _portfolio_proof_level(edge_levels.get(edge_id)) != level:
                reasons.append(f"selected_hyperedge_level_binding_mismatch:{edge_id}")
            if level < 2:
                reasons.append(f"selected_hyperedge_below_l2:{edge_id}")
            precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or []]
            required_molecule_ids.update([product_id, *precursor_ids])
            selected.append((product_id, edge_id, edge, level))
        if not selection_rows and root_id not in stock_ids:
            reasons.append("portfolio_route_has_no_selected_hyperedges")
        if selected and root_id not in seen_products:
            reasons.append("portfolio_route_root_not_selected")
        missing_molecules = sorted(value for value in required_molecule_ids if value not in molecules)
        reasons.extend(f"portfolio_molecule_missing:{value}" for value in missing_molecules)
        selected_products = {product_id for product_id, _, _, _ in selected}
        selected_precursors = {
            str(value)
            for _, _, edge, _ in selected
            for value in edge.get("precursor_molecule_ids") or []
        }
        materialized_leaves = selected_precursors - selected_products
        if selected and materialized_leaves != stock_ids:
            reasons.append("portfolio_stock_leaves_do_not_match_selection")
        for stock_id in sorted(stock_ids):
            stock_binding = dict(stock_bindings.get(stock_id) or {})
            molecule = dict(molecules.get(stock_id) or {})
            reasons.extend(
                f"portfolio_stock_binding:{stock_id}:{reason}"
                for reason in _portfolio_stock_binding_reasons(
                    stock_binding,
                    molecule_id=stock_id,
                    canonical_smiles=str(molecule.get("canonical_isomeric_smiles") or ""),
                )
            )
        if selected and not _portfolio_selection_acyclic(selected):
            reasons.append("portfolio_route_cycle_detected")
        if reasons:
            return "", sorted(set(reasons))

        branch_id = f"branch:{kind}:{_slug(route_id)}"
        node_id_by_molecule: dict[str, str] = {}
        route_source_refs: list[str] = []
        rendered_step_ids: list[str] = []
        for product_id, edge_id, edge, level in selected:
            edge_claims = [
                dict(claims.get(str(claim_id)) or {})
                for claim_id in edge.get("evidence_claim_ids") or []
                if str(claim_id) in claims
            ]
            source_refs = _dedupe(
                [
                    str(value)
                    for claim in edge_claims
                    for value in [
                        *[str(item) for item in claim.get("source_refs") or []],
                        *[str(item) for item in claim.get("evidence_refs") or []],
                        str(claim.get("report_ref") or ""),
                    ]
                    if str(value).strip()
                ]
            )
            route_source_refs.extend(source_refs)
            precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or []]
            product_smiles = str(
                (molecules.get(product_id) or {}).get("canonical_isomeric_smiles")
                or ""
            )
            precursor_smiles = [
                str(
                    (molecules.get(molecule_id) or {}).get(
                        "canonical_isomeric_smiles"
                    )
                    or ""
                )
                for molecule_id in precursor_ids
            ]
            reaction_digest = canonical_reaction_digest(
                product_smiles,
                precursor_smiles,
            )
            edge_evidence_binding_set = dict(
                edge_evidence_binding_sets.get(reaction_digest) or {}
            )
            interface_ids = [*precursor_ids, product_id]
            for molecule_id in interface_ids:
                molecule = dict(molecules.get(molecule_id) or {})
                smiles = str(molecule.get("canonical_isomeric_smiles") or "")
                if molecule_id == root_id:
                    label = str(self._target().get("name") or _compact_smiles_label(smiles))
                    role = "target"
                elif molecule_id in stock_ids:
                    label = _compact_smiles_label(smiles)
                    role = "stock_terminal"
                else:
                    label = _compact_smiles_label(smiles)
                    role = "portfolio_intermediate"
                node_id_by_molecule[molecule_id] = self._add_node(
                    label,
                    role=role,
                    smiles=smiles,
                    exactness="model_hypothesis",
                    confidence="high" if level >= 3 else "medium_high",
                    source_refs=source_refs,
                    missing=[],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                        evidence_row_id=edge_id,
                    ),
                )
            envelope_rows = [
                dict(envelopes.get(str(envelope_id)) or {})
                for envelope_id in edge.get("candidate_envelope_ids") or []
                if str(envelope_id) in envelopes
            ]
            conditions = _dedupe(
                [
                    str(value)
                    for envelope in envelope_rows
                    for value in [
                        *[str(item) for item in envelope.get("conditions") or []],
                        *[str(item) for item in envelope.get("catalysts") or []],
                        *[str(item) for item in envelope.get("enzymes") or []],
                    ]
                    if str(value).strip()
                ]
            )
            families = [str(value) for value in edge.get("reaction_families") or [] if str(value)]
            label = " / ".join(families[:2]) or "portfolio reaction"
            module_key = _module_key_for_text(label)
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=[node_id_by_molecule[value] for value in precursor_ids],
                to_nodes=[node_id_by_molecule[product_id]],
                module_key=module_key,
                module_label=_module_label_for_key(module_key),
                confidence="high" if level >= 3 else "medium_high",
                exactness="model_hypothesis",
                source_refs=source_refs,
                origin="route_portfolio",
                summary="Proof-eligible AND/OR portfolio hyperedge projected from the canonical overlay.",
                conditions=[{"label": "reported condition", "value": value} for value in conditions],
                missing=[],
            )
            binding = dict(edge_bindings.get(edge_id) or {})
            proof_tier = _portfolio_proof_tier(level)
            self.steps[step_id].update(
                {
                    "portfolio_route_id": route_id,
                    "portfolio_hyperedge_id": edge_id,
                    "portfolio_product_molecule_id": product_id,
                    "portfolio_precursor_molecule_ids": precursor_ids,
                    "source_channels": [str(value) for value in edge.get("source_channels") or []],
                    "independent_support_groups": [
                        str(value)
                        for value in (
                            edge_evidence_binding_set.get(
                                "independent_trusted_source_groups"
                            )
                            or edge.get("independent_support_groups")
                            or []
                        )
                    ],
                    "edge_evidence_binding_set": edge_evidence_binding_set,
                    "reaction_step_proof": {
                        "proof_source": "deterministic_reverified_route",
                        "proof_level": proof_tier,
                        "level_index": level,
                        "binding_sha256": str(binding.get("binding_sha256") or ""),
                        "portfolio_hyperedge_id": edge_id,
                    },
                    "proof_eligible": True,
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "not_parent_route_proof": True,
                }
            )
            rendered_step_ids.append(step_id)

        if not selected:
            molecule = dict(molecules.get(root_id) or {})
            smiles = str(molecule.get("canonical_isomeric_smiles") or "")
            node_id_by_molecule[root_id] = self._add_node(
                str(self._target().get("name") or _compact_smiles_label(smiles)),
                role="stock_terminal",
                smiles=smiles,
                exactness="model_hypothesis",
                confidence="high",
                source_refs=[],
                missing=[],
                identity_namespace=_molecule_identity_namespace(branch_id=branch_id),
            )

        self._add_branch(
            branch_id=branch_id,
            title=(
                f"Portfolio #{rank}: replacement preview"
                if kind == "validated_replacement_route"
                else f"Portfolio #{rank}: proof-eligible closed route"
            ),
            kind=kind,
            recommendation="proof-eligible AND/OR route",
            confidence=_confidence_from_score(float(route.get("portfolio_score") or route.get("base_score") or 0.0)),
            summary="Every selected hyperedge is L2+ and every materialized leaf is explicitly stock-bound; final parent-proof authority remains separate.",
            step_ids=rendered_step_ids,
            source_refs=_dedupe(route_source_refs),
            missing=(
                ["portfolio enumeration reached its configured bound"]
                if portfolio.get("truncated") is True
                else []
            ),
            classification_records=[],
        )
        branch = self.branches[-1]
        actual_branch_id = str(branch.get("branch_id") or branch_id)
        weakest = min((level for _, _, _, level in selected), default=4)
        branch.update(
            {
                "listed": bool(listed),
                "portfolio_route_id": route_id,
                "portfolio_rank": int(rank),
                "selected_hyperedges": selection_rows,
                "root_molecule_id": root_id,
                "root_molecule_node_id": node_id_by_molecule.get(root_id, ""),
                "stock_terminal_molecule_ids": sorted(stock_ids),
                "stock_terminal_node_ids": sorted(
                    node_id_by_molecule[value] for value in stock_ids if value in node_id_by_molecule
                ),
                "weakest_proof_level": weakest,
                "weakest_proof_tier": _portfolio_proof_tier(weakest),
                "source_channels": [str(value) for value in route.get("source_channels") or []],
                "independent_support_groups": [
                    str(value) for value in route.get("independent_support_groups") or []
                ],
                "base_score": float(route.get("base_score") or 0.0),
                "diversity_score": float(route.get("diversity_score") or 0.0),
                "portfolio_score": float(route.get("portfolio_score") or 0.0),
                "complete": True,
                "reaction_validated": True,
                "proof_eligible": True,
                "portfolio_enumeration": {
                    "complete_candidate_count": int(portfolio.get("complete_candidate_count") or 0),
                    "selected_route_count": len(portfolio.get("routes") or []),
                    "enumerated_candidate_count": int(portfolio.get("enumerated_candidate_count") or 0),
                    "solver_truncated": bool(portfolio.get("truncated")),
                    "reasons": [str(value) for value in portfolio.get("reasons") or []],
                },
                "solved": False,
                "executable": False,
                "advisory_only": True,
                "not_parent_route_proof": True,
            }
        )
        if not rendered_step_ids:
            branch["node_ids"] = _dedupe(
                [*branch.get("node_ids", []), *node_id_by_molecule.values()]
            )
        self._portfolio_branch_ids[route_id] = actual_branch_id
        return actual_branch_id, []

    def _portfolio_step_for_product(self, branch_id: str, product_id: str) -> str:
        branch = self._branch_by_id(branch_id)
        return next(
            (
                str(step_id)
                for step_id in branch.get("step_ids") or []
                if str((self.steps.get(str(step_id)) or {}).get("portfolio_product_molecule_id") or "")
                == product_id
            ),
            "",
        )

    def _branch_by_id(self, branch_id: str) -> dict[str, Any]:
        return next(
            (row for row in self.branches if str(row.get("branch_id") or "") == branch_id),
            {},
        )

    def add_route_consensus_graph_branches(self, *, limit: int | None) -> None:
        graph = self._route_consensus_graph_payload()
        if not graph:
            return
        step_by_id = {
            str(row.get("step_id") or ""): dict(row)
            for row in graph.get("steps") or []
            if isinstance(row, dict) and str(row.get("step_id") or "")
        }
        conflict_by_id = {
            str(row.get("conflict_id") or ""): dict(row)
            for row in graph.get("conflicts") or []
            if isinstance(row, dict) and str(row.get("conflict_id") or "")
        }
        target_smiles = str(self._target().get("smiles") or "")
        routes = [dict(row) for row in graph.get("route_hypotheses") or [] if isinstance(row, dict)]
        routes.sort(key=lambda row: (-float(row.get("rank_score") or 0.0), str(row.get("route_id") or "")))
        selected = self._limited_rows(routes, category="route_consensus_graph", limit=limit)
        for index, route in enumerate(selected, start=1):
            graph_route_id = str(route.get("route_id") or f"route-{index}")
            branch_id = f"branch:route_consensus_graph:{_slug(graph_route_id)}"
            rendered_step_ids: list[str] = []
            graph_steps: list[dict[str, Any]] = []
            for graph_step_id in route.get("forward_step_ids") or []:
                graph_step = dict(step_by_id.get(str(graph_step_id)) or {})
                if not graph_step:
                    continue
                product = str(graph_step.get("product_smiles") or "")
                precursors = [str(value) for value in graph_step.get("precursor_smiles") or [] if str(value).strip()]
                if not product or not precursors:
                    continue
                direct_refs = _dedupe(
                    [
                        *[str(value) for value in graph_step.get("source_refs") or []],
                        *[str(value) for value in graph_step.get("evidence_refs") or []],
                    ]
                )
                from_nodes = [
                    self._add_node(
                        _compact_smiles_label(smiles),
                        role="consensus_graph_precursor",
                        smiles=smiles if _looks_like_smiles(smiles) else "",
                        exactness="model_hypothesis",
                        confidence=str(graph_step.get("confidence") or "low"),
                        source_refs=direct_refs,
                        missing=["advisory graph node; deterministic identity audit still required"],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=direct_refs,
                            evidence_row_id=str(graph_step_id),
                        ),
                    )
                    for smiles in precursors
                ]
                to_nodes = [
                    self._add_node(
                        self._route_smiles_label(product, role="consensus graph product", target_smiles=target_smiles),
                        role="target" if target_smiles and _same_molecule(product, target_smiles) else "consensus_graph_intermediate",
                        smiles=product if _looks_like_smiles(product) else "",
                        exactness="model_hypothesis",
                        confidence=str(graph_step.get("confidence") or "low"),
                        source_refs=direct_refs,
                        missing=["advisory graph node; deterministic identity audit still required"],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=direct_refs,
                            evidence_row_id=str(graph_step_id),
                        ),
                    )
                ]
                family = str(graph_step.get("reaction_family") or "consensus disconnection")
                all_graph_support_records = _consensus_support_records(
                    graph_step,
                    limit=None,
                )
                graph_support_records = all_graph_support_records[:32]
                graph_support_groups = _consensus_independent_support_groups(
                    graph_step,
                    all_graph_support_records,
                )
                edge_evidence_binding_set = _edge_evidence_binding_for_graph_step(
                    graph,
                    graph_step,
                )
                if edge_evidence_binding_set:
                    graph_support_groups = [
                        str(value)
                        for value in edge_evidence_binding_set.get(
                            "independent_trusted_source_groups"
                        )
                        or []
                        if str(value)
                    ]
                conditions = [
                    {"label": "candidate condition", "value": str(value)}
                    for value in graph_step.get("conditions") or []
                    if str(value).strip()
                ]
                display_step_id = self._add_step(
                    branch_id=branch_id,
                    label=family,
                    from_nodes=from_nodes,
                    to_nodes=to_nodes,
                    module_key=_module_key_for_text(family),
                    module_label=_module_label_for_key(_module_key_for_text(family)),
                    confidence=str(graph_step.get("confidence") or "low"),
                    exactness="model_hypothesis",
                    source_refs=direct_refs,
                    origin="route_consensus_graph",
                    summary="Advisory multi-step graph edge assembled from a frontier-specific consensus.",
                    conditions=conditions,
                    missing=_dedupe(
                        [
                            "advisory graph edge; not solved or executable",
                            "deterministic parent-route proof is required",
                            *[str(value) for value in graph_step.get("limitations") or []],
                            *[f"required validation: {value}" for value in graph_step.get("required_validation") or []],
                        ]
                    ),
                )
                self.steps[display_step_id].update(
                    {
                        "graph_step_id": str(graph_step_id),
                        "support_records": graph_support_records,
                        "support_count": len(all_graph_support_records),
                        "support_record_count": len(all_graph_support_records),
                        "support_records_truncated": len(all_graph_support_records)
                        > len(graph_support_records),
                        "independent_support_groups": graph_support_groups,
                        "edge_evidence_binding_set": edge_evidence_binding_set,
                        "independent_source_count": len(graph_support_groups),
                        "consensus_scope": (
                            "multi_source"
                            if len(graph_support_groups) > 1
                            else "correlated_single_source"
                        ),
                        "multi_source": len(graph_support_groups) > 1,
                        "conflicts": [
                            conflict_by_id[str(conflict_id)]
                            for conflict_id in graph_step.get("conflict_ids") or []
                            if str(conflict_id) in conflict_by_id
                        ],
                        "advisory_only": True,
                        "solved": False,
                        "executable": False,
                        "not_parent_route_proof": True,
                    }
                )
                rendered_step_ids.append(display_step_id)
                graph_steps.append(graph_step)
            if not rendered_step_ids:
                continue
            self._add_branch(
                branch_id=branch_id,
                title=f"Codex multi-step hypothesis #{index}",
                kind="route_consensus_graph",
                recommendation="advisory multi-step hypothesis",
                confidence=_confidence_from_score(float(route.get("rank_score") or 0.0)),
                summary="Frontier-specific Codex teams assembled these edges into one read-only forward route hypothesis.",
                step_ids=rendered_step_ids,
                source_refs=[],
                missing=[
                    "not a deterministic parent-route proof",
                    "frontier leaves may remain unexpanded",
                    "all reaction edges require forward and stock validation",
                ],
                classification_records=graph_steps,
            )
            branch = self.branches[-1]
            branch.update(
                {
                    "graph_route_id": graph_route_id,
                    "rank_score": float(route.get("rank_score") or 0.0),
                    "forward_dependencies": [
                        dict(row) for row in route.get("forward_dependencies") or [] if isinstance(row, dict)
                    ],
                    "frontier": [dict(row) for row in route.get("frontier") or [] if isinstance(row, dict)],
                    "conflict_ids": [str(value) for value in route.get("conflict_ids") or []],
                    "route_level_source_refs": self._route_consensus_graph_refs(),
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "not_parent_route_proof": True,
                }
            )
            actual_branch_id = str(branch.get("branch_id") or branch_id)
            merged_ids = _dedupe(
                [
                    str(value)
                    for graph_step in graph_steps
                    for value in graph_step.get("proposal_ids") or []
                    if str(value).strip()
                ]
            )
            branch["merged_consensus_ids"] = merged_ids
            branch["semantic_projection"] = "canonical_multistep_graph"
            for consensus_id in merged_ids:
                self._consensus_branch_ids.setdefault(consensus_id, actual_branch_id)
            for step_id in rendered_step_ids:
                self.steps[step_id]["merged_consensus_details"] = True
                self.steps[step_id]["semantic_projection"] = "canonical_graph_edge"

    def _route_consensus_graph_payload(self) -> dict[str, Any]:
        if self._codex_team_projection_reasons():
            return {}
        direct = self.blackboard.get("route_consensus_graph")
        if isinstance(direct, dict) and direct.get("schema_version") == "route_consensus_graph.v1":
            return dict(direct)
        team = dict(self.blackboard.get("codex_agent_team") or {})
        nested = team.get("route_consensus_graph")
        if (
            team.get("accepted") is True
            and isinstance(nested, dict)
            and nested.get("schema_version") == "route_consensus_graph.v1"
        ):
            return dict(nested)
        return {}

    def _route_consensus_graph_refs(self) -> list[str]:
        team = dict(self.blackboard.get("codex_agent_team") or {})
        refs = [str(team.get("route_consensus_graph_ref") or "")]
        refs.extend(
            str(value)
            for key, value in (self.blackboard.get("artifact_refs") or {}).items()
            if "route_consensus_graph" in str(key).lower() and str(value).strip()
        )
        return _dedupe(refs)

    def _route_consensus_graph_view(self) -> dict[str, Any]:
        graph = self._route_consensus_graph_payload()
        if not graph:
            return {
                "schema_version": "route_consensus_graph_view.v1",
                "available": False,
                "route_count": 0,
                "step_count": 0,
                "node_count": 0,
                "semantics": {"advisory_only": True, "solved": False, "executable": False},
            }
        return {
            "schema_version": "route_consensus_graph_view.v1",
            "source_schema_version": str(graph.get("schema_version") or ""),
            "available": True,
            "has_hypotheses": bool(graph.get("has_hypotheses")),
            "route_count": len(graph.get("route_hypotheses") or []),
            "step_count": len(graph.get("steps") or []),
            "node_count": len(graph.get("nodes") or []),
            "conflict_count": len(graph.get("conflicts") or []),
            "cycle_count": len(graph.get("cycles") or []),
            "route_portfolio": dict(graph.get("route_portfolio") or {}),
            "route_portfolio_bindings": dict(
                graph.get("route_portfolio_bindings")
                or (graph.get("route_portfolio") or {}).get("bindings")
                or {}
            ),
            "route_replacement_catalog": dict(
                graph.get("route_replacement_catalog") or {}
            ),
            "truncation": dict(graph.get("truncation") or {}),
            "route_level_source_refs": self._route_consensus_graph_refs(),
            "semantics": {"advisory_only": True, "solved": False, "executable": False},
        }

    def _route_consensus_payload(self) -> dict[str, Any]:
        if self._codex_team_projection_reasons():
            return {}
        direct = self.blackboard.get("route_consensus")
        if isinstance(direct, dict) and direct.get("schema_version") == "route_consensus.v1":
            return dict(direct)
        team = dict(self.blackboard.get("codex_agent_team") or {})
        nested = team.get("route_consensus")
        if (
            team.get("accepted") is True
            and isinstance(nested, dict)
            and nested.get("schema_version") == "route_consensus.v1"
        ):
            return dict(nested)
        return {}

    def _codex_team_projection_reasons(self) -> list[str]:
        team = self.blackboard.get("codex_agent_team")
        if not isinstance(team, dict) or not team:
            return []
        reasons: list[str] = []
        if team.get("accepted") is not True:
            reasons.append("codex_agent_team_not_accepted")
        validation = team.get("artifact_validation")
        if isinstance(validation, dict) and validation.get("accepted") is not True:
            reasons.append("codex_agent_team_artifact_validation_failed")
        coordinator = team.get("coordinator")
        if isinstance(coordinator, dict):
            status = str(coordinator.get("status") or "").strip()
            if status and status != "accepted_draft":
                reasons.append(f"codex_agent_team_coordinator_status:{status}")
        runtime = team.get("runtime_summary")
        if isinstance(runtime, dict) and runtime.get("consistent") is not True:
            reasons.append("codex_agent_team_runtime_inconsistent")
        child_reports = team.get("child_reports")
        if isinstance(child_reports, list) and any(
            not isinstance(row, dict) or row.get("accepted") is not True for row in child_reports
        ):
            reasons.append("codex_agent_team_child_report_rejected")
        return _dedupe(reasons)

    def _route_consensus_route_refs(self) -> list[str]:
        team = dict(self.blackboard.get("codex_agent_team") or {})
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        refs = [str(team.get("route_consensus_ref") or "")]
        refs.extend(
            str(value)
            for key, value in artifact_refs.items()
            if "route_consensus" in str(key).lower() and str(value).strip()
        )
        return _dedupe(refs)

    def _route_consensus_view(self) -> dict[str, Any]:
        consensus = self._route_consensus_payload()
        if not consensus:
            quarantine_reasons = self._codex_team_projection_reasons()
            return {
                "schema_version": "route_consensus_view.v1",
                "available": False,
                "quarantined": bool(quarantine_reasons),
                "reasons": quarantine_reasons,
                "proposals": [],
                "source_summary": {},
                "semantics": {
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "deterministic_parent_proof_required": True,
                },
            }
        proposal_views = []
        for proposal in consensus.get("proposals") or []:
            if not isinstance(proposal, dict):
                continue
            consensus_id = str(proposal.get("consensus_id") or "")
            all_support_records = _consensus_support_records(proposal, limit=None)
            support_records = all_support_records[:32]
            support_groups = _consensus_independent_support_groups(
                proposal,
                all_support_records,
            )
            independent_source_count = len(support_groups)
            proposal_views.append(
                {
                    "consensus_id": consensus_id,
                    "branch_id": self._consensus_branch_ids.get(consensus_id, ""),
                    "rank": int(proposal.get("rank") or 0),
                    "reaction_family": str(proposal.get("reaction_family") or "unspecified"),
                    "status": str(proposal.get("status") or "model_hypothesis"),
                    "evidence_level": str(proposal.get("evidence_level") or "model_only"),
                    "confidence": str(proposal.get("confidence") or "low"),
                    "rank_score": float(proposal.get("rank_score") or 0.0),
                    "source_channels": _dedupe([str(item) for item in proposal.get("source_channels") or []]),
                    "support_records": support_records,
                    "support_count": len(all_support_records),
                    "support_record_count": len(all_support_records),
                    "support_records_truncated": len(all_support_records)
                    > len(support_records),
                    "independent_support_groups": support_groups,
                    "independent_source_count": independent_source_count,
                    "consensus_scope": (
                        "multi_source"
                        if independent_source_count > 1
                        else "correlated_single_source"
                    ),
                    "multi_source": independent_source_count > 1,
                    "codex_roles_correlated": any(group == "codex_model" for group in support_groups),
                    "source_refs": _consensus_direct_source_refs(proposal),
                    "condition_support": [
                        dict(row) for row in proposal.get("condition_support") or [] if isinstance(row, dict)
                    ],
                    "conflicts": _consensus_conflicts(proposal),
                    "limitations": _dedupe([str(item) for item in proposal.get("limitations") or []]),
                    "required_validation": _dedupe(
                        [str(item) for item in proposal.get("required_validation") or []]
                    ),
                    "advisory_only": True,
                    "solved": False,
                    "executable": False,
                    "not_parent_route_proof": True,
                }
            )
        return {
            "schema_version": "route_consensus_view.v1",
            "source_schema_version": "route_consensus.v1",
            "available": True,
            "has_candidates": bool(proposal_views),
            "accepted_as_route": False,
            "route_level_source_refs": self._route_consensus_route_refs(),
            "source_summary": dict(consensus.get("source_summary") or {}),
            "proposals": proposal_views,
            "rejected_candidates": [
                dict(row) for row in consensus.get("rejected_candidates") or [] if isinstance(row, dict)
            ],
            "semantics": {
                "advisory_only": True,
                "solved": False,
                "executable": False,
                "deterministic_parent_proof_required": True,
                "codex_roles_are_correlated": True,
            },
        }

    def add_proposal_branches(self, *, limit: int | None) -> None:
        proposals = [row for row in self.blackboard.get("retrosynthetic_proposals") or [] if isinstance(row, dict)]
        if self._codex_team_projection_reasons():
            proposals = [
                row
                for row in proposals
                if str(row.get("source_type") or "") != "multi_source_consensus"
                and not str(row.get("proposal_id") or "").startswith("consensus:")
            ]
        if self._route_consensus_payload():
            proposals = [
                row
                for row in proposals
                if str(row.get("source_type") or "") != "multi_source_consensus"
                and not str(row.get("proposal_id") or "").startswith("consensus:")
            ]
        proposals.sort(key=lambda row: (not bool(row.get("executable")), -float(row.get("score") or 0.0)))
        seen_labels: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        for proposal in proposals:
            label = _clean_label(proposal.get("proposal_label") or proposal.get("proposal_type") or "proposal")
            dedupe_key = f"{label}:{proposal.get('precursor_smiles') or ''}"[:220]
            if dedupe_key in seen_labels:
                continue
            seen_labels.add(dedupe_key)
            deduplicated.append(proposal)
        selected = self._limited_rows(deduplicated, category="retrosynthetic_proposals", limit=limit)
        for proposal in selected:
            label = _clean_label(proposal.get("proposal_label") or proposal.get("proposal_type") or "proposal")
            branch_id = self._unique_branch_id(f"branch:proposal:{_slug(str(proposal.get('proposal_id') or label))}")
            precursor_nodes = self._proposal_precursor_nodes(proposal, branch_id=branch_id)
            if not precursor_nodes:
                precursor_nodes = [
                    self._add_node(
                        f"Strategic precursor: {label}",
                        role="hypothesis_precursor",
                        exactness="model_hypothesis",
                        confidence=str(proposal.get("confidence") or "medium"),
                        source_refs=[str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()],
                        missing=["no machine-readable precursor structure"],
                        identity_namespace=_molecule_identity_namespace(
                            branch_id=branch_id,
                            source_refs=[
                                str(x)
                                for x in proposal.get("evidence_refs") or []
                                if str(x).strip()
                            ],
                            evidence_row_id=str(proposal.get("proposal_id") or label),
                        ),
                    )
                ]
            source_refs = [str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()][:8]
            product_smiles = str(
                proposal.get("product_smiles") or proposal.get("target_smiles") or ""
            ).strip()
            requested_target_smiles = str(self._target().get("smiles") or "")
            if product_smiles and _same_molecule(product_smiles, requested_target_smiles):
                product_node = self._target_node(identity_namespace=branch_id)
            else:
                product_label = _clean_label(
                    proposal.get("product_label")
                    or proposal.get("target_label")
                    or (_compact_smiles_label(product_smiles) if product_smiles else "unbound proposal product")
                )
                product_node = self._add_node(
                    product_label,
                    role="hypothesis_product",
                    smiles=product_smiles if _looks_like_smiles(product_smiles) else "",
                    exactness="model_hypothesis",
                    confidence=str(proposal.get("confidence") or "medium"),
                    source_refs=source_refs,
                    missing=[
                        "proposal product is not the requested target"
                        if product_smiles
                        else "proposal product identity is not bound to the requested target"
                    ],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                        evidence_row_id=str(proposal.get("proposal_id") or label),
                    ),
                )
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=precursor_nodes,
                to_nodes=[product_node],
                module_key=_module_key_for_text(label),
                module_label=_module_label_for_key(_module_key_for_text(label)),
                confidence=str(proposal.get("confidence") or "medium"),
                exactness="model_hypothesis",
                source_refs=source_refs,
                origin="retrosynthetic_proposal",
                summary=str(proposal.get("proposal_type") or proposal.get("route_objective_type") or "Explored proposal"),
                conditions=_conditions_from_row(proposal),
                missing=[str(x) for x in proposal.get("risk_flags") or [] if str(x).strip()][:8],
            )
            self._add_branch(
                branch_id=branch_id,
                title=f"候选逆合成分支：{label}",
                kind="retrosynthetic_proposal",
                recommendation="探索触碰",
                confidence=str(proposal.get("confidence") or "medium"),
                summary="A model/planner proposal touched during exploration. It is not a parent-route proof.",
                step_ids=[step_id],
                source_refs=[str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()][:8],
                missing=[str(x) for x in proposal.get("risk_flags") or [] if str(x).strip()][:8],
                classification_records=[proposal],
            )

    def add_template_branches(self, *, limit: int | None) -> None:
        templates = [row for row in self.blackboard.get("broad_transform_templates") or [] if isinstance(row, dict)]
        selected = self._limited_rows(templates, category="broad_transform_templates", limit=limit)
        for index, template in enumerate(selected, start=1):
            template_id = str(template.get("template_id") or f"template:{index}")
            branch_id = self._unique_branch_id(f"branch:template:{_slug(template_id)}")
            label = _clean_label(template.get("transform_logic") or template.get("objective_type") or template_id)
            source_refs = [str(x) for x in template.get("source_refs") or template.get("evidence_refs") or [] if str(x).strip()]
            from_label, to_label = _template_endpoint_labels(template)
            reactant_smiles = str(
                template.get("reactant_smiles") or template.get("from_smiles") or ""
            ).strip()
            product_smiles = str(
                template.get("product_smiles") or template.get("to_smiles") or ""
            ).strip()
            requested_target_smiles = str(self._target().get("smiles") or "")
            from_node = self._add_node(
                from_label,
                role="template_precursor",
                smiles=reactant_smiles if _looks_like_smiles(reactant_smiles) else "",
                exactness="model_hypothesis",
                confidence="medium",
                source_refs=source_refs,
                missing=["broad-template endpoint; exact structure is not established"],
                identity_namespace=_molecule_identity_namespace(
                    branch_id=branch_id,
                    source_refs=source_refs,
                    evidence_row_id=template_id,
                ),
            )
            if product_smiles and _same_molecule(product_smiles, requested_target_smiles):
                to_node = self._target_node(identity_namespace=branch_id)
            else:
                to_node = self._add_node(
                    to_label,
                    role="template_product",
                    smiles=product_smiles if _looks_like_smiles(product_smiles) else "",
                    exactness="model_hypothesis",
                    confidence="medium",
                    source_refs=source_refs,
                    missing=["broad-template endpoint is not bound to the requested target"],
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                        evidence_row_id=template_id,
                    ),
                )
            step_id = self._add_step(
                branch_id=branch_id,
                label=label,
                from_nodes=[from_node],
                to_nodes=[to_node],
                module_key=_module_key_for_text(label + " " + str(template.get("reaction_center") or "")),
                module_label=_module_label_for_key(_module_key_for_text(label)),
                confidence="medium",
                exactness="model_hypothesis",
                source_refs=source_refs,
                origin="broad_transform_template",
                summary=str(template.get("reaction_center") or template.get("objective_type") or ""),
                conditions=_conditions_from_row(template),
                missing=_dedupe(
                    [
                        *[str(x) for x in template.get("risk_flags") or [] if str(x).strip()],
                        "broad template is advisory and does not imply a route to the requested target",
                    ]
                )[:8],
            )
            self._add_branch(
                branch_id=branch_id,
                title=f"宽泛模板分支：{_clean_label(template.get('objective_type') or template_id)}",
                kind="broad_template",
                recommendation="模板提示",
                confidence="medium",
                summary="A broad transform touched by the planner after exact literature rows were unavailable.",
                step_ids=[step_id],
                source_refs=source_refs,
                missing=_dedupe(
                    [
                        *[str(x) for x in template.get("risk_flags") or [] if str(x).strip()],
                        "broad template is advisory and does not imply a route to the requested target",
                    ]
                )[:8],
                classification_records=[template],
            )

    def add_exact_row_branch(self) -> None:
        rows = self._source_detail_chain_rows()
        if not rows:
            rows = [row for row in self.evidence.get("exact_rows") or [] if isinstance(row, dict)]
        if not rows:
            return
        branch_id = "branch:exact_literature_rows"
        step_ids: list[str] = []
        verified_flags: list[bool] = []
        for index, row in enumerate(rows, start=1):
            source_ref = str(row.get("source_ref") or row.get("source_title") or "")
            source_refs = _dedupe(
                [
                    source_ref,
                    *[str(x) for x in row.get("evidence_refs") or [] if str(x).strip()],
                ]
            )
            label = _clean_label(
                row.get("reaction_label")
                or row.get("step_label")
                or row.get("step_id")
                or row.get("source_template_id")
                or row.get("row_id")
                or f"exact row {index}"
            )
            reactants = _labels_from_any(
                row.get("reactant_labels")
                or row.get("reactants")
                or row.get("reactant_smiles")
                or row.get("main_reactant_smiles")
            )
            products = _labels_from_any(row.get("product_labels") or row.get("products") or row.get("product_smiles"))
            module_key = self._exact_row_module_key(row=row, label=label, index=index)
            module_label = self._exact_row_module_label(row=row, label=label, fallback=module_key)
            row_verified = _exact_row_is_verified(row)
            verified_flags.append(row_verified)
            row_confidence = "high" if row_verified else "medium"
            row_exactness = "exact_literature_row" if row_verified else "named_literature"
            row_missing = [] if row_verified else ["exact row is not deterministically validated"]

            def exact_node(value: str, role: str) -> str:
                return self._add_node(
                    label=_exact_node_label(value, row=row, role=role),
                    role=role,
                    smiles=value if _looks_like_smiles(value) else "",
                    exactness=row_exactness,
                    confidence=row_confidence,
                    source_refs=source_refs,
                    missing=row_missing,
                    identity_namespace=_molecule_identity_namespace(
                        branch_id=branch_id,
                        source_refs=source_refs,
                        evidence_row_id=str(
                            row.get("row_id")
                            or row.get("step_id")
                            or row.get("source_template_id")
                            or index
                        ),
                    ),
                )

            step_ids.append(
                self._add_step(
                    branch_id=branch_id,
                    label=label,
                    from_nodes=[exact_node(x, "exact_reactant") for x in reactants],
                    to_nodes=[exact_node(x, "exact_product") for x in products],
                    module_key=module_key,
                    module_label=module_label,
                    confidence=row_confidence,
                    exactness=row_exactness,
                    source_refs=source_refs,
                    origin="exact_literature_row",
                    summary="Exact row compiled from source details.",
                    conditions=_conditions_from_row(row),
                    missing=row_missing,
                )
            )
        all_verified = bool(verified_flags and all(verified_flags))
        self._add_branch(
            branch_id=branch_id,
            title="Exact literature rows",
            kind="exact_literature" if all_verified else "literature_candidate",
            recommendation="强证据",
            confidence="high" if all_verified else "medium",
            summary="Machine-readable literature rows; only deterministically validated rows are displayed as exact.",
            step_ids=step_ids,
            source_refs=_dedupe([str(row.get("source_ref") or "") for row in rows if str(row.get("source_ref") or "").strip()]),
            missing=[] if all_verified else ["contains unvalidated literature rows"],
            classification_records=rows,
        )

    def _source_detail_chain_rows(self) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        if self.run_dir:
            run = Path(self.run_dir)
            candidates.extend(
                [
                    run / "source_detail_chain_route_result.json",
                    run / "source_detail_chain_route" / "source_detail_route_chain_audit.json",
                ]
            )
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        for key, value in artifact_refs.items():
            if "source_detail" not in str(key).lower() and "source_detail" not in str(value).lower():
                continue
            candidates.append(Path(str(value)))
        seen: set[str] = set()
        for path in candidates:
            try:
                resolved = path.expanduser().resolve()
            except OSError:
                continue
            path_key = str(resolved).lower()
            if path_key in seen or not resolved.is_file():
                continue
            seen.add(path_key)
            try:
                data = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows = _chain_rows_from_source_detail_payload(data)
            if rows:
                return rows
        return []

    def _exact_row_module_key(self, *, row: dict[str, Any], label: str, index: int) -> str:
        source_id = str(row.get("source_template_id") or row.get("step_id") or row.get("row_id") or "").strip()
        if source_id:
            return f"source_detail_exact_step:{_slug(source_id)}"
        return f"source_detail_exact_step:{index:02d}_{_slug(label)}"

    def _exact_row_module_label(self, *, row: dict[str, Any], label: str, fallback: str) -> str:
        condition = dict(row.get("condition_candidate") or {})
        bits = [
            str(row.get("step_id") or row.get("row_id") or label or "").replace("_", " "),
            str(condition.get("reagent") or condition.get("reagents") or "").strip(),
            str(condition.get("reported_yield") or condition.get("yield") or "").strip(),
        ]
        text = " · ".join(bit for bit in bits if bit)
        return text[:80] or fallback

    def _visual_step(self, branch_id: str, chain: dict[str, Any], row: dict[str, Any], index: int) -> str:
        source_ref = str(chain.get("source_ref") or chain.get("source_title") or "")
        identity_namespace = _molecule_identity_namespace(
            branch_id=branch_id,
            source_refs=[source_ref] if source_ref else [],
            evidence_row_id=str(row.get("row_id") or row.get("step_id") or index),
        )
        reactants = _labels_from_any(row.get("reactant_labels") or row.get("reactants") or row.get("main_reactant_smiles") or row.get("reactant_smiles"))
        products = _labels_from_any(row.get("product_label") or row.get("product_labels") or row.get("product_smiles"))
        if not products:
            products = [f"visual product {index}"]
        from_nodes = []
        reactant_smiles = _labels_from_any(row.get("reactant_smiles") or row.get("main_reactant_smiles"))
        for idx, label in enumerate(reactants or [f"visual precursor {index}"], start=0):
            smiles = reactant_smiles[idx] if idx < len(reactant_smiles) and _looks_like_smiles(reactant_smiles[idx]) else ""
            from_nodes.append(
                self._add_node(
                    label=label,
                    role="visual_precursor",
                    smiles=smiles,
                    exactness="visual_inferred",
                    confidence=str(row.get("confidence") or "low"),
                    source_refs=[source_ref] if source_ref else [],
                    missing=[str(x) for x in row.get("risk_flags") or [] if str(x).strip()][:4],
                    identity_namespace=identity_namespace,
                )
            )
        product_smiles = _labels_from_any(row.get("product_smiles"))
        to_nodes = []
        for idx, label in enumerate(products, start=0):
            smiles = product_smiles[idx] if idx < len(product_smiles) and _looks_like_smiles(product_smiles[idx]) else ""
            to_nodes.append(
                self._add_node(
                    label=label,
                    role="visual_product",
                    smiles=smiles,
                    exactness="visual_inferred",
                    confidence=str(row.get("confidence") or "low"),
                    source_refs=[source_ref] if source_ref else [],
                    missing=[str(x) for x in row.get("risk_flags") or [] if str(x).strip()][:4],
                    identity_namespace=identity_namespace,
                )
            )
        label = _clean_label(row.get("reaction_class") or row.get("step_id") or f"visual step {index}")
        return self._add_step(
            branch_id=branch_id,
            label=label,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            module_key=_module_key_for_text(" ".join([label, source_ref, str(row.get("source_locator") or "")])),
            module_label=_module_label_for_key(_module_key_for_text(label)),
            confidence=str(row.get("confidence") or "low"),
            exactness="visual_inferred",
            source_refs=[source_ref] if source_ref else [],
            origin="visual_chain",
            summary=str(row.get("source_locator") or row.get("allowed_use") or ""),
            conditions=_conditions_from_row(row),
            missing=[str(x) for x in row.get("risk_flags") or [] if str(x).strip()][:8],
        )

    def _proposal_precursor_nodes(
        self,
        proposal: dict[str, Any],
        *,
        branch_id: str,
    ) -> list[str]:
        text = str(proposal.get("precursor_smiles") or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split(".") if part.strip()]
        out = []
        source_refs = [
            str(x) for x in proposal.get("evidence_refs") or [] if str(x).strip()
        ][:8]
        identity_namespace = _molecule_identity_namespace(
            branch_id=branch_id,
            source_refs=source_refs,
            evidence_row_id=str(proposal.get("proposal_id") or "proposal"),
        )
        for idx, smiles in enumerate(parts[:5], start=1):
            label = f"proposal precursor {idx}"
            out.append(
                self._add_node(
                    label=label,
                    role="hypothesis_precursor",
                    smiles=smiles if _looks_like_smiles(smiles) else "",
                    exactness="model_hypothesis",
                    confidence=str(proposal.get("confidence") or "medium"),
                    source_refs=source_refs,
                    missing=[str(x) for x in proposal.get("risk_flags") or [] if str(x).strip()][:4],
                    identity_namespace=identity_namespace,
                )
            )
        return out

    def _target_node(self, *, identity_namespace: str) -> str:
        target = self._target()
        return self._add_node(
            target.get("name") or "target",
            role="target",
            smiles=str(target.get("smiles") or ""),
            exactness="name_only",
            confidence="high",
            source_refs=[],
            identity_namespace=_molecule_identity_namespace(branch_id=identity_namespace),
        )

    def _target(self) -> dict[str, Any]:
        profile = dict(self.blackboard.get("target_profile") or {})
        raw_name = str(profile.get("target_name") or self.blackboard.get("case_id") or "")
        family_hint = str(profile.get("family_hint") or "")
        return {
            "name": _display_target_name(raw_name, family_hint, str(self.blackboard.get("case_id") or "")),
            "smiles": str(profile.get("target_smiles") or profile.get("canonical_smiles") or profile.get("isomeric_smiles") or ""),
            "family_hint": family_hint,
        }

    def _add_branch(
        self,
        *,
        branch_id: str,
        title: str,
        kind: str,
        recommendation: str,
        confidence: str,
        summary: str,
        step_ids: list[str],
        source_refs: list[str],
        missing: list[str],
        classification_records: list[dict[str, Any]] | None = None,
        proof_binding: dict[str, Any] | None = None,
    ) -> None:
        # Some branch builders reserve a unique id before materializing their
        # nodes and steps so branch-scoped identity namespaces cannot collide.
        # Reusing that reservation here is intentional; allocating it a second
        # time used to append ``:2`` to the branch while leaving every step on
        # the now-nonexistent pre-dedup id.
        finalized_branch_ids = {
            str(row.get("branch_id") or "") for row in self.branches
        }
        if branch_id in finalized_branch_ids:
            branch_id = self._unique_branch_id(branch_id)
        elif branch_id not in self._branch_ids:
            self._branch_ids.add(branch_id)

        valid_step_ids = [sid for sid in step_ids if sid in self.steps]
        for step_id in valid_step_ids:
            self.steps[step_id]["branch_id"] = branch_id
        title = _branch_title_for_display(branch_id=branch_id, title=title, kind=kind)
        recommendation = _recommendation_for_display(kind=kind, recommendation=recommendation)
        node_ids: list[str] = []
        for step_id in step_ids:
            step = self.steps.get(step_id) or {}
            node_ids.extend([str(x) for x in step.get("from_node_ids") or []])
            node_ids.extend([str(x) for x in step.get("to_node_ids") or []])
        row = {
            "branch_id": branch_id,
            "title": title,
            "kind": kind,
            "recommendation": recommendation,
            "confidence": _normalize_confidence(confidence),
            "summary": summary,
            "step_ids": valid_step_ids,
            "node_ids": _dedupe(node_ids),
            "source_refs": _dedupe(source_refs),
            "missing": _dedupe(missing),
        }
        binding = dict(proof_binding or {})
        verified_parent_route = bool(
            kind in {"direct_verified_route", "stitched_verified_route"}
            and binding.get("accepted") is True
            and str(binding.get("route_structure_sha256") or "")
            and self._final_verdict_allows_solved_branch()
        )
        row.update(
            {
                "solved": verified_parent_route,
                "executable": verified_parent_route,
                "advisory_only": not verified_parent_route,
                "not_parent_route_proof": not verified_parent_route,
                "proof_binding": binding,
            }
        )
        row.update(_classify_synthesis_records(classification_records or []))
        self.branches.append(row)

    def _final_verdict_allows_solved_branch(self) -> bool:
        """Honor an explicitly materialized unresolved verdict, if present.

        Normal controller execution builds the forest before the verdict and
        therefore has no embedded verdict here.  Saved-run refreshes and
        callers that do provide one must never display an unresolved decision
        as a solved/non-advisory branch.
        """
        verdict = self.blackboard.get("final_verdict")
        if not isinstance(verdict, dict) or not verdict:
            return True
        return bool(
            verdict.get("solved") is True
            and str(verdict.get("verdict") or "").strip().lower() == "solved"
            and str(verdict.get("route_status") or "").strip().lower() == "solved"
        )

    def _add_step(
        self,
        *,
        branch_id: str,
        label: str,
        from_nodes: list[str],
        to_nodes: list[str],
        module_key: str,
        module_label: str,
        confidence: str,
        exactness: str,
        source_refs: list[str],
        origin: str,
        summary: str,
        missing: list[str],
        conditions: list[dict[str, str]] | None = None,
    ) -> str:
        if _display_text_is_corrupt(module_label):
            module_label = _module_label_for_key(module_key)
        step_id = f"step:{_slug(branch_id)}:{_slug(label)}:{len(self.steps) + 1}"
        condition_rows = _normalize_condition_rows(conditions or [])
        self.steps[step_id] = {
            "step_id": step_id,
            "branch_id": branch_id,
            "label": label,
            "from_node_ids": [x for x in from_nodes if x],
            "to_node_ids": [x for x in to_nodes if x],
            "module_key": module_key or "other",
            "module_label": module_label or "Other",
            "confidence": _normalize_confidence(confidence),
            "exactness": _normalize_exactness(exactness),
            "source_refs": _dedupe(source_refs),
            "origin": origin,
            "summary": summary,
            "conditions": condition_rows,
            "condition_summary": _condition_summary(condition_rows),
            "condition_status": _condition_status(condition_rows, missing),
            "missing": _dedupe(missing),
        }
        return step_id

    def _add_node(
        self,
        label: str,
        *,
        role: str,
        smiles: str = "",
        exactness: str,
        confidence: str,
        source_refs: list[str],
        missing: list[str] | None = None,
        identity_namespace: str = "",
    ) -> str:
        label = _clean_label(label) or "unnamed node"
        smiles = str(smiles or "").strip()
        namespace = str(identity_namespace or "").strip()
        if not namespace:
            # Fail closed for unstructured identities.  A caller that wants
            # name-only continuity must provide an explicit branch/source/row
            # namespace instead of relying on a repository-global label.
            namespace = _molecule_identity_namespace(
                branch_id=f"unscoped-assertion-{len(self.nodes) + 1}",
                source_refs=source_refs,
                evidence_row_id=role,
            )
        node_id, canonical_smiles = _molecule_node_identity(
            smiles=smiles,
            label=label,
            namespace=namespace,
        )
        existing = self.nodes.get(node_id)
        row = {
            "node_id": node_id,
            "label": label,
            "role": role,
            "roles": [role],
            "smiles": canonical_smiles or smiles,
            "input_smiles": smiles,
            "canonical_isomeric_smiles": canonical_smiles,
            "identity_namespace": "" if canonical_smiles else namespace,
            "representation_kind": "smiles" if smiles else "name_only",
            "exactness": _normalize_exactness(exactness),
            "confidence": _normalize_confidence(confidence),
            "source_refs": _dedupe(source_refs),
            "missing": _dedupe(missing or []),
            "assertions": [{
                "exactness": _normalize_exactness(exactness),
                "confidence": _normalize_confidence(confidence),
                "source_refs": _dedupe(source_refs),
            }],
        }
        if existing:
            row["label"] = _better_node_label(existing.get("label"), label)
            row["roles"] = _dedupe(
                [
                    *[str(item) for item in existing.get("roles") or [existing.get("role")] if str(item or "")],
                    role,
                ]
            )
            row["role"] = _preferred_node_role(row["roles"])
            row["source_refs"] = _dedupe([*(existing.get("source_refs") or []), *row["source_refs"]])
            row["missing"] = _dedupe([*(existing.get("missing") or []), *row["missing"]])
            row["assertions"] = [*(existing.get("assertions") or []), *row["assertions"]]
            row["exactness"] = _worst_ranked(existing.get("exactness"), row["exactness"], EXACTNESS_RANK)
            row["confidence"] = _worst_ranked(existing.get("confidence"), row["confidence"], CONFIDENCE_RANK)
            if not existing.get("smiles") and smiles:
                row["smiles"] = smiles
                row["representation_kind"] = "smiles"
            elif existing.get("smiles"):
                row["smiles"] = existing.get("smiles")
                row["representation_kind"] = "smiles"
        row.update(_structure_payload_for_smiles(row.get("smiles")))
        self.nodes[node_id] = row
        return node_id

    def _node_id_for_label(self, label: str) -> str:
        namespace = _molecule_identity_namespace(
            branch_id="legacy-node-id-for-label",
            evidence_row_id=label,
        )
        node_id, _ = _molecule_node_identity(smiles="", label=label, namespace=namespace)
        if node_id not in self.nodes:
            return self._add_node(
                label,
                role="intermediate",
                exactness="name_only",
                confidence="medium",
                source_refs=[],
                identity_namespace=namespace,
            )
        return node_id

    def _best_direct_route_result(self) -> dict[str, Any]:
        proof = dict(self.blackboard.get("parent_route_proof") or {})
        expected_target = str(self._target().get("smiles") or "")
        proof_accepted = is_solved_parent_route_proof(
            proof,
            expected_target_smiles=expected_target,
        )
        if proof_accepted and str(proof.get("proof_mode") or "") == "direct_parent_route":
            route = _route_from_parent_route_proof(proof)
            parent_verifier = dict((proof.get("proof_evidence") or {}).get("parent_verifier") or {})
            proof_route = dict(parent_verifier.get("accepted_route") or {})
            route_digest = _route_structure_sha256(route.get("steps") or [])
            evidence_digest = _route_structure_sha256(proof_route.get("steps") or [])
            if route.get("steps") and route_digest and route_digest == evidence_digest:
                return {
                    "route": route,
                    "reaction_validation": dict(parent_verifier.get("reaction_validation") or {}),
                    "artifact_path": "",
                    "source_ref": str(proof.get("source_ref") or "parent_route_proof"),
                    "classification_records": [proof, route],
                    "proof_binding": {
                        "schema_version": "route_forest_parent_proof_binding.v1",
                        "accepted": True,
                        "proof_mode": "direct_parent_route",
                        "route_structure_sha256": route_digest,
                        "reaction_proof_sha256": str(
                            (parent_verifier.get("reaction_validation") or {}).get("proof_digest")
                            or ""
                        ),
                        "binding_source": "proof_evidence.parent_verifier.accepted_route",
                    },
                }

        # A guided result can still be useful display evidence, but even a
        # replayed L1/L2 verifier report is not final parent-proof authority.
        for artifact in self._guided_result_artifacts():
            verifier = dict(artifact.get("raw_route_verifier") or {})
            # Never promote the backend's own ``solved`` claim (or an unrelated
            # parent proof) into verification of this artifact.  The route and
            # best rank below are meaningful only under this artifact's
            # deterministic verifier report.
            if not _deterministic_route_verifier_accepted(
                verifier,
                expected_target_smiles=expected_target,
            ):
                continue
            result = dict(artifact.get("result") or {})
            routes = [dict(row) for row in result.get("routes") or artifact.get("routes") or [] if isinstance(row, dict)]
            if not routes:
                continue
            reverified = verify_chemenzy_raw_routes(
                {"result": {**result, "routes": routes}},
                target_smiles=expected_target,
            )
            if not is_accepted_route_verifier_report(
                reverified,
                expected_target_smiles=expected_target,
            ):
                continue
            if (
                verifier.get("best_route_rank") != reverified.get("best_route_rank")
                or int(verifier.get("best_route_step_count") or 0)
                != int(reverified.get("best_route_step_count") or 0)
            ):
                continue
            best_rank = reverified.get("best_route_rank")
            route = _route_by_verified_rank(routes, best_rank)
            if route.get("steps"):
                return {
                    "route": route,
                    "reaction_validation": dict(reverified.get("reaction_validation") or {}),
                    "artifact_path": str(artifact.get("_artifact_path") or ""),
                    "source_ref": str(artifact.get("_artifact_key") or "guided_chemenzy_result"),
                    "classification_records": [artifact, result, route],
                    "proof_binding": {
                        "schema_version": "route_forest_parent_proof_binding.v1",
                        "accepted": False,
                        "proof_mode": "unbound_guided_artifact",
                        "route_structure_sha256": _route_structure_sha256(route.get("steps") or []),
                        "reaction_proof_sha256": str(
                            (reverified.get("reaction_validation") or {}).get("proof_digest")
                            or ""
                        ),
                        "binding_source": str(artifact.get("_artifact_path") or ""),
                        "reasons": ["accepted_parent_route_proof_binding_missing"],
                    },
                }
        return {}

    def _guided_result_artifacts(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        candidate_paths: list[tuple[str, Path]] = []
        for key, value in artifact_refs.items():
            text = str(value or "").strip()
            if not text:
                continue
            if "chemenzy" not in str(key).lower() and "chemenzy" not in text.lower() and "guided" not in text.lower():
                continue
            candidate_paths.append((str(key), Path(text)))
        if self.run_dir:
            run = Path(self.run_dir)
            for name in ("guided_chemenzy_result.json", "guided_chemenzy_raw_result.json"):
                candidate_paths.append((name, run / name))
            for path in run.glob("*guided_chemenzy*_result*.json"):
                candidate_paths.append((path.name, path))
        for key, path in candidate_paths:
            try:
                resolved = path.expanduser().resolve()
            except OSError:
                continue
            path_key = str(resolved).lower()
            if path_key in seen or not resolved.is_file():
                continue
            seen.add(path_key)
            try:
                data = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            data["_artifact_key"] = key
            data["_artifact_path"] = str(resolved)
            out.append(data)
        return out

    def _subgoal_route_records(self) -> list[dict[str, Any]]:
        candidate_paths: list[tuple[str, Path]] = []
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        for key, value in artifact_refs.items():
            text = f"{key} {value}".lower()
            if "route_expansion" in text or "subgoal" in text:
                candidate_paths.append((str(key), Path(str(value))))
        if self.run_dir:
            run = Path(self.run_dir)
            candidate_paths.append(("route_expansion_subgoal_search_result", run / "route_expansion_subgoal_search_result.json"))
            for path in run.glob("*route_expansion_subgoal_search_result*.json"):
                candidate_paths.append((path.name, path))

        out: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_records: set[str] = set()
        for key, path in candidate_paths:
            resolved = self._resolve_artifact_path(path)
            if resolved is None:
                continue
            path_key = str(resolved).lower()
            if path_key in seen_paths or not resolved.is_file():
                continue
            seen_paths.add(path_key)
            data = self._read_json_dict(resolved)
            if not data:
                continue
            if data.get("accepted") is False or data.get("solved") is False:
                continue
            for index, row in enumerate(data.get("subgoals") or [], start=1):
                if not isinstance(row, dict):
                    continue
                verifier = dict(row.get("verifier") or {})
                parent_relevance = dict(row.get("parent_relevance_gate") or {})
                accepted = bool(
                    row.get("accepted") is True
                    and row.get("solved") is True
                    and str(row.get("route_status") or "").strip().lower() == "solved"
                    and parent_relevance.get("accepted") is True
                    and _deterministic_route_verifier_accepted(verifier)
                )
                if not accepted:
                    continue
                subgoal = dict(row.get("subgoal") or {})
                policy = dict(subgoal.get("policy") or {})
                preferred = dict(policy.get("preferred_subgoal") or {})
                terminal = dict(preferred.get("terminal_candidate") or {})
                name = (
                    str(subgoal.get("name") or "").strip()
                    or str(terminal.get("name") or "").strip()
                    or f"subgoal {index}"
                )
                target_smiles = (
                    str(subgoal.get("smiles") or "").strip()
                    or str(terminal.get("canonical_smiles") or terminal.get("smiles") or "").strip()
                )
                raw_path = self._resolve_artifact_path(row.get("raw_result_path"))
                raw = self._read_json_dict(raw_path) if raw_path is not None else {}
                routes = [
                    dict(candidate)
                    for candidate in (raw.get("routes") or (raw.get("result") or {}).get("routes") or [])
                    if isinstance(candidate, dict)
                ]
                best_rank = verifier.get("best_route_rank")
                route = self._choose_route_by_rank(routes, best_rank)
                if not any(_materialized_route_step(step) for step in route.get("steps") or []):
                    continue
                route_rank = route.get("route_rank") if route else best_rank
                record_key = f"{name}|{target_smiles}|{raw_path or ''}|{route_rank}"
                if record_key in seen_records:
                    continue
                seen_records.add(record_key)
                evidence_refs = _dedupe(
                    [
                        *[str(x) for x in policy.get("evidence_refs") or [] if str(x).strip()],
                        str(terminal.get("source_ref") or ""),
                    ]
                )
                out.append(
                    {
                        "name": name,
                        "target_smiles": target_smiles,
                        "accepted": accepted,
                        "route": route,
                        "reaction_validation": dict(verifier.get("reaction_validation") or {}),
                        "route_objective_type": str(
                            subgoal.get("route_objective_type")
                            or policy.get("route_objective_type")
                            or ""
                        ),
                        "route_rank": route_rank,
                        "search_path": str(resolved),
                        "search_key": key,
                        "raw_path": str(raw_path or ""),
                        "evidence_refs": evidence_refs,
                        "reasons": [str(x) for x in verifier.get("reasons") or row.get("reasons") or []],
                        "accepted_route_count": verifier.get("accepted_route_count"),
                        "route_count": row.get("route_count") or raw.get("n_results"),
                    }
                )
        return out

    def _choose_route_by_rank(self, routes: list[dict[str, Any]], best_rank: Any) -> dict[str, Any]:
        return _route_by_verified_rank(routes, best_rank)

    def _resolve_artifact_path(self, path: Any) -> Path | None:
        text = str(path or "").strip()
        if not text:
            return None
        raw = Path(text)
        candidates = [raw]
        if self.run_dir:
            run = Path(self.run_dir)
            candidates.append(run / raw.name)
            candidates.append(run / "route_expansion_subgoals" / raw.name)
            parts = list(raw.parts)
            if run.name in parts:
                index = parts.index(run.name)
                tail = parts[index + 1 :]
                if tail:
                    candidates.append(run.joinpath(*tail))
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
        try:
            return raw.expanduser().resolve()
        except OSError:
            return raw

    def _read_json_dict(self, path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _route_smiles_label(self, value: str, *, role: str, target_smiles: str) -> str:
        text = str(value or "").strip()
        if target_smiles and _same_text(text, target_smiles):
            return self._target().get("name") or "target"
        if not text:
            return role
        if _looks_like_smiles(text):
            return _compact_smiles_label(text)
        return _clean_label(text)

    def _unique_branch_id(self, branch_id: str) -> str:
        base = branch_id
        idx = 2
        while branch_id in self._branch_ids:
            branch_id = f"{base}:{idx}"
            idx += 1
        self._branch_ids.add(branch_id)
        return branch_id

    def _target_node_ids(self) -> set[str]:
        target = self._target()
        target_name = str(target.get("name") or "").strip().lower()
        target_smiles = str(target.get("smiles") or "").strip()
        out: set[str] = set()
        for node_id, node in self.nodes.items():
            label = str(node.get("label") or "").strip().lower()
            smiles = str(node.get("smiles") or "").strip()
            role = str(node.get("role") or "").strip().lower()
            if role == "target":
                out.add(node_id)
                continue
            if target_smiles and smiles == target_smiles:
                out.add(node_id)
                continue
            if target_name and target_name in label and ("free acid" in label or "target" in role):
                out.add(node_id)
        return out

    def _branch_modules(self, branch: dict[str, Any]) -> set[str]:
        generic = {"", "other", "other_route_module", "diagnostic_failure", "visual_failed_or_empty"}
        out: set[str] = set()
        for step_id in branch.get("step_ids") or []:
            step = self.steps.get(str(step_id)) or {}
            key = str(step.get("module_key") or "")
            if key not in generic:
                out.add(key)
        return out

    def _branch_relationships(self) -> list[dict[str, Any]]:
        target_ids = self._target_node_ids()
        out: list[dict[str, Any]] = []
        for left_index, left in enumerate(self.branches):
            for right in self.branches[left_index + 1 :]:
                rel = self._branch_relationship(left, right, target_ids)
                if rel:
                    out.append(rel)
        return out

    def _branch_relationship(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        target_ids: set[str],
    ) -> dict[str, Any] | None:
        major_kinds = {
            "stitched_verified_route",
            "direct_verified_route",
            "subgoal_verified_route",
            "exact_literature",
            "process_evidence",
            "visual_chain",
            "proof_eligible_portfolio_route",
        }
        left_kind = str(left.get("kind") or "")
        right_kind = str(right.get("kind") or "")
        left_major = left_kind in major_kinds
        right_major = right_kind in major_kinds
        left_nodes = set(str(x) for x in left.get("node_ids") or [])
        right_nodes = set(str(x) for x in right.get("node_ids") or [])
        shared_nodes = sorted(left_nodes & right_nodes)
        shared_target_nodes = [node_id for node_id in shared_nodes if node_id in target_ids]
        left_modules = self._branch_modules(left)
        right_modules = self._branch_modules(right)
        shared_modules = sorted(left_modules & right_modules)
        right_refs = set(str(x) for x in right.get("source_refs") or [])
        shared_refs = _dedupe(
            [
                ref
                for ref in left.get("source_refs") or []
                if ref and ref in right_refs and _external_source_ref(ref)
            ]
        )
        if shared_target_nodes and left_major and right_major:
            kind = "shared_target_endpoint"
        elif shared_modules and (left_major or right_major):
            kind = "same_reaction_module"
        elif shared_refs and (left_major or right_major):
            kind = "shared_literature_source"
        else:
            return None
        source_refs = shared_refs or _dedupe([*(left.get("source_refs") or []), *(right.get("source_refs") or [])])[:8]
        if any(kind.startswith("route_consensus") for kind in (left_kind, right_kind)):
            summary = "共识候选与另一分支共享目标或反应模块；该关系仅供候选对照，不构成 solved 或 executable 证明。"
        elif kind == "shared_target_endpoint":
            summary = "这些路线共享目标或终点分子，应作为同一目标下的路线变体对照查看。"
        elif kind == "same_reaction_module":
            labels = [_module_label_for_key(key) for key in shared_modules[:3]]
            summary = f"这些路线触碰了相同反应模块：{'、'.join(labels)}。"
        else:
            summary = "这些路线引用了相同文献来源，可作为同一证据链下的分支查看。"
        shared_node_labels = [
            str((self.nodes.get(node_id) or {}).get("label") or node_id)
            for node_id in shared_nodes[:8]
        ]
        return {
            "relationship_id": f"rel:{_slug(left.get('branch_id'))}:{_slug(right.get('branch_id'))}:{kind}",
            "kind": kind,
            "from_branch_id": str(left.get("branch_id") or ""),
            "to_branch_id": str(right.get("branch_id") or ""),
            "summary": summary,
            "shared_node_ids": shared_nodes[:12],
            "shared_node_labels": shared_node_labels,
            "shared_module_keys": shared_modules[:12],
            "shared_module_labels": [_module_label_for_key(key) for key in shared_modules[:12]],
            "source_refs": source_refs,
        }

    def _modules(self) -> list[dict[str, Any]]:
        rows: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        for step in self.steps.values():
            key = str(step.get("module_key") or "other")
            rows.setdefault(key, []).append(str(step.get("step_id") or ""))
            labels.setdefault(key, str(step.get("module_label") or key))
        return [
            {
                "module_key": key,
                "module_label": labels.get(key, key),
                "step_ids": ids,
                "alternative_count": len(
                    {
                        candidate_id
                        for step_id in ids
                        for candidate_id in (self.steps.get(step_id) or {}).get("validated_replacement_ids") or []
                    }
                ),
                "candidate_count": sum(
                    len((self.steps.get(step_id) or {}).get("replacement_candidate_ids") or [])
                    for step_id in ids
                ),
                "rejected_replacement_count": sum(
                    int((self.steps.get(step_id) or {}).get("replacement_rejection_count") or 0)
                    for step_id in ids
                ),
                "replacement_semantics": "backend_and_or_route_revalidated_only",
            }
            for key, ids in sorted(rows.items(), key=lambda item: (-len(item[1]), item[0]))
        ]

    def _evidence_index(self) -> dict[str, Any]:
        candidate_records = [
            dict(row)
            for row in self.evidence.get("source_candidates") or []
            if isinstance(row, dict)
        ]
        real_candidates = [
            row for row in candidate_records if _source_candidate_has_real_source(row)
        ]
        placeholder_candidates = [
            row for row in candidate_records if not _source_candidate_has_real_source(row)
        ]
        source_identity_summary = dict(
            self.evidence.get("source_identity_summary") or {}
        )

        def identity_count(field: str) -> int:
            try:
                return max(0, int(source_identity_summary.get(field) or 0))
            except (TypeError, ValueError):
                return 0

        def candidate_view(row: dict[str, Any]) -> dict[str, Any]:
            doi = str(row.get("doi") or "").strip()
            pii = str(row.get("pii") or "").strip()
            return {
                "source_ref": str(
                    row.get("source_ref")
                    or (f"doi:{doi}" if doi else "")
                    or (f"pii:{pii}" if pii else "")
                    or row.get("url")
                    or row.get("local_pdf")
                    or row.get("pdf_path")
                    or ""
                ),
                "title": str(row.get("title") or row.get("source_title") or ""),
                "local_pdf": str(row.get("local_pdf") or row.get("pdf_path") or ""),
                "source_type": str(row.get("source_type") or ""),
                "access_status": str(row.get("access_status") or ""),
                "source_discovery_mode": str(row.get("source_discovery_mode") or ""),
                "placeholder_only": not _source_candidate_has_real_source(row),
            }

        return {
            "source_candidates": [
                candidate_view(row)
                for row in candidate_records
            ][:20],
            "real_source_candidates": [
                candidate_view(row)
                for row in real_candidates
            ][:20],
            "placeholder_candidates": [candidate_view(row) for row in placeholder_candidates][:20],
            "source_candidate_summary": {
                # Candidate rows are representations awaiting/holding source
                # locators.  They are not a count of independent documents or
                # source groups.
                "real_source_count": len(real_candidates),
                "real_source_candidate_record_count": len(real_candidates),
                "placeholder_count": len(placeholder_candidates),
                "record_count": len(candidate_records),
            },
            "source_identity_summary": {
                "schema_version": "route_forest_source_identity_summary.v1",
                "document_count": identity_count("document_count"),
                "independent_source_group_count": identity_count(
                    "independent_source_group_count"
                ),
                "representation_count": identity_count("representation_count"),
                "candidate_record_count": len(candidate_records),
                "real_source_candidate_record_count": len(real_candidates),
                "semantics": {
                    "candidate_records_are_not_independent_sources": True,
                    "document_and_representation_counts_are_distinct": True,
                },
            },
            "exact_chain_audits": [
                {
                    "accepted": bool(row.get("accepted")),
                    "source_ref": str(row.get("source_ref") or ""),
                    "reasons": [str(x) for x in row.get("reasons") or []],
                }
                for row in self.evidence.get("exact_chain_audits") or []
                if isinstance(row, dict)
            ][:20],
            "visual_chains": [
                {
                    "accepted": bool(row.get("accepted") or row.get("exploratory_accepted")),
                    "source_ref": str(row.get("source_ref") or row.get("source_title") or ""),
                    "step_count": int(row.get("step_count") or len(row.get("steps") or row.get("candidate_steps") or [])),
                    "reasons": [str(x) for x in row.get("reasons") or []],
                }
                for row in self.evidence.get("visual_chains") or []
                if isinstance(row, dict)
            ][:20],
            "process_evidence_rows": [
                {
                    "source_ref": str(row.get("source_ref") or row.get("source_title") or ""),
                    "endpoint_labels": [str(x) for x in row.get("endpoint_labels") or []],
                    "local_pdf": str(row.get("local_pdf") or row.get("source_pdf_path") or ""),
                }
                for row in self.evidence.get("process_evidence_rows") or []
                if isinstance(row, dict)
            ][:20],
            "route_expansion_subgoals": [
                {
                    "title": str(row.get("name") or "subgoal closure"),
                    "accepted": bool(row.get("accepted")),
                    "step_count": len((row.get("route") or {}).get("steps") or []),
                    "route_rank": row.get("route_rank"),
                    "source_ref": str(row.get("raw_path") or row.get("search_path") or ""),
                    "reasons": [str(x) for x in row.get("reasons") or []],
                }
                for row in self._subgoal_route_records()
            ][:20],
        }

    def _run_trace(self) -> dict[str, Any]:
        artifact_refs = dict(self.blackboard.get("artifact_refs") or {})
        source_candidates = [
            dict(row)
            for row in self.evidence.get("source_candidates") or []
            if isinstance(row, dict)
        ]
        real_source_count = sum(
            1 for row in source_candidates if _source_candidate_has_real_source(row)
        )
        placeholder_count = len(source_candidates) - real_source_count
        source_refs = [str(value) for value in self.evidence.get("source_refs") or []]
        real_source_refs = [
            value for value in source_refs if not _placeholder_source_ref(value)
        ]
        identity_summary = dict(
            self.evidence.get("source_identity_summary") or {}
        )

        def identity_count(field: str) -> int:
            try:
                return max(0, int(identity_summary.get(field) or 0))
            except (TypeError, ValueError):
                return 0
        actions = [
            {
                "round_index": int(row.get("round_index") or 0),
                "action_type": str(row.get("action_type") or ""),
                "useful_artifact": bool(row.get("useful_artifact")),
                "reasons": [str(item) for item in row.get("reasons") or []],
            }
            for row in self.blackboard.get("action_history") or []
            if isinstance(row, dict)
        ]
        return {
            "schema_version": "route_forest_run_trace.v1",
            "run_dir": self.run_dir,
            "actions": actions[:80],
            "artifact_refs": [
                {"key": str(key), "path": str(value)}
                for key, value in sorted(artifact_refs.items())
                if str(value or "").strip()
            ][:120],
            "literature_counts": {
                # v1 compatibility: the established unsuffixed fields retain
                # their original all-record semantics.  Consumers that need
                # validated locators must use the explicit real_* fields.
                "source_candidates": len(source_candidates),
                "candidate_record_count": len(source_candidates),
                "real_source_candidates": real_source_count,
                "real_source_candidate_records": real_source_count,
                "placeholder_candidates": placeholder_count,
                "source_candidate_records": len(source_candidates),
                "source_refs": len(source_refs),
                "real_source_refs": len(real_source_refs),
                "source_ref_records": len(source_refs),
                "placeholder_source_refs": len(source_refs) - len(real_source_refs),
                "document_count": identity_count("document_count"),
                "independent_source_group_count": identity_count(
                    "independent_source_group_count"
                ),
                "representation_count": identity_count("representation_count"),
                "visual_chains": len(self.evidence.get("visual_chains") or []),
                "process_evidence_rows": len(self.evidence.get("process_evidence_rows") or []),
                "exact_rows": len(self.evidence.get("exact_rows") or []),
                "pdf_structure_evidence": len(self.evidence.get("pdf_structure_evidence") or []),
                "scout_attempts": len(self.evidence.get("scout_attempts") or []),
            },
        }


def _classify_synthesis_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify display branches from structured metadata, never target names."""
    classes: set[str] = set()
    evidence: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for path, raw_value in _structured_synthesis_markers(record, path=f"record[{index}]"):
            synthesis_class = _synthesis_class_for_marker(path.rsplit(".", 1)[-1], raw_value)
            if not synthesis_class or synthesis_class == "unspecified":
                continue
            classes.add(synthesis_class)
            evidence.append(f"{path}={raw_value}->{synthesis_class}")
    if "hybrid" in classes or len(classes - {"hybrid"}) > 1:
        synthesis_class = "hybrid"
    elif classes:
        synthesis_class = next(iter(classes))
    else:
        synthesis_class = "unspecified"
    return {
        "synthesis_class": synthesis_class,
        "classification_evidence": _dedupe(evidence)[:24],
        "classification_policy": "structured_metadata_only",
    }


def _structured_synthesis_markers(
    value: Any,
    *,
    path: str,
    depth: int = 0,
) -> list[tuple[str, str]]:
    if depth > 6:
        return []
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child_path = f"{path}.{key}"
            if str(key) in _SYNTHESIS_CLASS_FIELDS and isinstance(nested, (str, int, float)):
                text = str(nested or "").strip()
                if text:
                    out.append((child_path, text))
            elif isinstance(nested, (dict, list, tuple)):
                out.extend(_structured_synthesis_markers(nested, path=child_path, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list, tuple)):
                out.extend(
                    _structured_synthesis_markers(nested, path=f"{path}[{index}]", depth=depth + 1)
                )
    return out


def _synthesis_class_for_marker(field: str, raw_value: Any) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(raw_value or "").strip().lower()).strip("_")
    if not value:
        return ""
    if field == "synthesis_class" and value in _SYNTHESIS_CLASSES:
        return value
    if "total_synth" in value:
        return "total_synthesis"
    if "semisynth" in value:
        return "semisynthesis"
    if any(token in value for token in ("biosynth", "biotransformation", "fermentation")):
        return "biosynthesis"
    return ""


def _dependency_layout(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    node_ids = [str(row.get("graph_node_id") or "") for row in nodes if row.get("graph_node_id")]
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source_graph_node_id") or "")
        target = str(edge.get("target_graph_node_id") or "")
        if source not in adjacency or target not in adjacency or target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1
    queue = sorted(node_id for node_id, count in indegree.items() if count == 0)
    layers = {node_id: 0 for node_id in node_ids}
    visited: list[str] = []
    while queue:
        node_id = queue.pop(0)
        visited.append(node_id)
        for target in sorted(adjacency[node_id]):
            layers[target] = max(layers[target], layers[node_id] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    cycle_node_ids = sorted(set(node_ids) - set(visited))
    if cycle_node_ids:
        cycle_layer = max((layers[node_id] for node_id in visited), default=-1) + 1
        for index, node_id in enumerate(cycle_node_ids):
            layers[node_id] = cycle_layer + index % 2
    return {
        "layers": layers,
        "acyclic": not cycle_node_ids,
        "cycle_node_ids": cycle_node_ids,
    }


def _matching_reaction_step_proof(
    proofs: list[dict[str, Any]],
    route_step: dict[str, Any],
) -> dict[str, Any]:
    product = _canonical_molecule_smiles(_route_step_product(route_step))
    reactants = sorted(
        value
        for value in (
            _canonical_molecule_smiles(item) for item in _route_step_reactants(route_step)
        )
        if value
    )
    matches = [
        proof
        for proof in proofs
        if _canonical_molecule_smiles(str(proof.get("product_smiles") or "")) == product
        and sorted(
            value
            for value in (
                _canonical_molecule_smiles(str(item)) for item in proof.get("reactant_smiles") or []
            )
            if value
        )
        == reactants
    ]
    return dict(matches[0]) if len(matches) == 1 else {}


def _topological_step_order(
    step_ids: list[str],
    encoded_dependencies: list[str],
) -> tuple[list[str], bool]:
    adjacency: dict[str, set[str]] = {step_id: set() for step_id in step_ids}
    indegree: dict[str, int] = {step_id: 0 for step_id in step_ids}
    for encoded in encoded_dependencies:
        producer, _, consumer = encoded.partition("\u0000")
        if producer not in adjacency or consumer not in adjacency or consumer in adjacency[producer]:
            continue
        adjacency[producer].add(consumer)
        indegree[consumer] += 1
    original_position = {step_id: index for index, step_id in enumerate(step_ids)}
    queue = sorted(
        (step_id for step_id, count in indegree.items() if count == 0),
        key=lambda value: original_position[value],
    )
    out: list[str] = []
    while queue:
        step_id = queue.pop(0)
        out.append(step_id)
        for consumer in sorted(adjacency[step_id], key=lambda value: original_position[value]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)
                queue.sort(key=lambda value: original_position[value])
    acyclic = len(out) == len(step_ids)
    if not acyclic:
        out.extend(step_id for step_id in step_ids if step_id not in set(out))
    return out, acyclic


def _module_key_for_text(text: str) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ("heterocycle formation", "heterocycle synthesis", "pyrrole formation")):
        return "heterocycle_core_construction"
    if "ketal" in lowered:
        return "ketal_deprotection"
    if any(token in lowered for token in ("hydrolysis", "saponification")):
        return "ester_hydrolysis"
    if any(token in lowered for token in ("salt formation", "salt isolation", "salt metathesis")):
        return "salt_formation"
    if any(token in lowered for token in ("free acid", "form adjustment", "free base")):
        return "form_adjustment"
    if any(token in lowered for token in ("sidechain installation", "side-chain installation")):
        return "sidechain_installation"
    if any(token in lowered for token in ("sidechain", "side-chain", "anthranilate", "ester_to", "acid chloride", "amide")):
        return "amide_or_sidechain_assembly"
    if any(token in lowered for token in ("protect", "tes", "silyl", "deprotection", "deprotect")):
        return "protection_state_adjustment"
    if any(token in lowered for token in ("semisynthesis", "same-scaffold", "same_core")):
        return "semisynthesis_anchor"
    if any(token in lowered for token in ("core", "b ring", "cage", "scaffold", "ring system")):
        return "scaffold_core_construction"
    if any(token in lowered for token in ("visual", "scheme", "image")):
        return "visual_literature_hint"
    return "other_route_module"

def _display_target_name(raw_name: str, family_hint: str = "", case_id: str = "") -> str:
    name = str(raw_name or case_id or "").strip()
    if "_advisory" in name:
        name = name.split("_advisory", 1)[0]
    if "_fullflow" in name:
        name = name.split("_fullflow", 1)[0]
    return name or "target"


def _labels_from_any(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_labels_from_any(item))
        return _dedupe(out)
    if isinstance(value, dict):
        for key in ("label", "name", "smiles", "canonical_smiles"):
            if str(value.get(key) or "").strip():
                return [str(value.get(key))]
        return []
    text = str(value or "").strip()
    if not text:
        return []
    if "." in text and _looks_like_smiles(text):
        return [part for part in text.split(".") if part]
    return [text]


def _looks_like_smiles(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or " " in raw:
        return False
    if Chem is None:
        # Without the chemistry parser, failing closed is safer than attaching
        # a molecule identity to a human-readable compound name.
        return False
    try:
        return Chem.MolFromSmiles(raw) is not None
    except Exception:
        return False


def _deterministic_route_verifier_accepted(
    verifier: dict[str, Any],
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Use the same fail-closed verifier contract as final-verdict consumers."""
    return bool(
        is_accepted_route_verifier_report(
            verifier,
            expected_target_smiles=expected_target_smiles,
        )
        and verifier.get("stock_audit_passed") is not False
    )


def _route_by_verified_rank(routes: list[dict[str, Any]], best_rank: Any) -> dict[str, Any]:
    """Select only the route explicitly named by a verifier; never fall back."""
    if best_rank is None:
        return {}
    rank = str(best_rank).strip()
    for candidate in routes:
        if isinstance(candidate, dict) and str(candidate.get("route_rank")).strip() == rank:
            return dict(candidate)
    return {}


def _materialized_route_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    product = str(step.get("product") or step.get("product_smiles") or "").strip()
    return bool(product and _route_step_reactants(step))


def _chain_rows_from_source_detail_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    chain_audit = dict(data.get("chain_audit") or {}) if isinstance(data.get("chain_audit"), dict) else {}
    nested_audit = (
        dict(data.get("source_detail_route_chain_audit") or {})
        if isinstance(data.get("source_detail_route_chain_audit"), dict)
        else {}
    )
    candidates = [
        (data.get("chain"), data.get("accepted") is True),
        (
            chain_audit.get("chain"),
            chain_audit.get("accepted") is True and data.get("accepted") is not False,
        ),
        (
            nested_audit.get("chain"),
            nested_audit.get("accepted") is True and data.get("accepted") is not False,
        ),
    ]
    for value, audit_accepted in candidates:
        if not audit_accepted:
            continue
        rows = [dict(row) for row in value or [] if isinstance(row, dict)]
        rows = [
            row
            for row in rows
            if str(row.get("product_smiles") or "").strip()
            and (row.get("reactant_smiles") or row.get("main_reactant_smiles") or row.get("reactants"))
        ]
        if rows:
            # Source-detail chains are stored in retrosynthetic order
            # (target-proximal first). The route forest displays synthesis
            # direction, ending at the requested target.
            return list(reversed(sorted(rows, key=lambda row: int(row.get("step_index") or 0))))
    return []


def _exact_node_label(value: Any, *, row: dict[str, Any], role: str) -> str:
    text = str(value or "").strip()
    if role == "exact_product":
        step_id = str(row.get("step_id") or row.get("source_template_id") or "").strip()
        if step_id:
            parts = step_id.split("_")
            if parts:
                suffix = parts[-1]
                if suffix and not suffix.isdigit():
                    return suffix.replace("-", " ")
    if _looks_like_smiles(text):
        return _compact_smiles_label(text)
    return _clean_label(text) or role


def _external_source_ref(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text.startswith(("idea:", "route_objective:", "target_side_", "broad_template")):
        return False
    return any(token in text for token in ("doi", "10.", "http", ".pdf", "science", "pubmed", "acs", "elsevier", "springer"))


def _structure_payload_for_smiles(smiles: Any) -> dict[str, Any]:
    text = str(smiles or "").strip()
    if not text:
        return {
            "structure_svg": "",
            "structure_valid": False,
            "structure_status": "no_smiles",
            "formula": "",
            "heavy_atom_count": None,
        }
    cached = _STRUCTURE_CACHE.get(text)
    if cached is not None:
        return dict(cached)
    out: dict[str, Any] = {
        "structure_svg": "",
        "structure_valid": False,
        "structure_status": "rdkit_unavailable" if Chem is None else "invalid_smiles",
        "formula": "",
        "heavy_atom_count": None,
    }
    if Chem is not None:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            try:
                if rdDepictor is not None:
                    rdDepictor.Compute2DCoords(mol)
            except Exception:
                pass
            out.update(
                {
                    "structure_svg": _mol_svg(mol),
                    "structure_valid": True,
                    "structure_status": "rendered",
                    "formula": rdMolDescriptors.CalcMolFormula(mol) if rdMolDescriptors is not None else "",
                    "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                }
            )
    _STRUCTURE_CACHE[text] = dict(out)
    return out


def _mol_svg(mol: Any, *, width: int = 240, height: int = 170) -> str:
    if rdMolDraw2D is None:
        return ""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.clearBackground = False
        options.padding = 0.08
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")
        start = svg.find("<svg")
        return svg[start:] if start >= 0 else svg
    except Exception:
        return ""


def _conditions_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def add(label: str, value: Any) -> None:
        text = _condition_value_text(value)
        if text:
            out.append({"label": label, "value": text})

    candidate = row.get("condition_candidate")
    if isinstance(candidate, dict):
        add("试剂", candidate.get("reagent") or candidate.get("reagents"))
        add("催化剂", candidate.get("catalyst") or candidate.get("catalysts"))
        add("碱", candidate.get("base"))
        add("氧化剂", candidate.get("oxidant"))
        add("溶剂", candidate.get("solvent") or candidate.get("solvents"))
        add("温度", candidate.get("temperature"))
        add("时间", candidate.get("duration") or candidate.get("time"))
        add("收率", candidate.get("reported_yield") or candidate.get("yield"))
        add("条件原文", candidate.get("condition_text_transcribed"))
        add("来源依据", candidate.get("source_grounding") or candidate.get("source_excerpt"))

    if not out:
        # Legacy artifacts used several aliases before the visual-agent contract
        # standardized on condition_candidate.
        legacy_fields = (
            ("条件", row.get("condition_text")),
            ("反应条件", row.get("reaction_conditions")),
            ("可见条件", row.get("visible_conditions")),
            ("文献条件", row.get("source_grounded_conditions")),
            ("条件", row.get("conditions")),
            ("试剂", row.get("reagents") or row.get("reagent")),
            ("催化剂", row.get("catalysts") or row.get("catalyst")),
            ("溶剂", row.get("solvents") or row.get("solvent")),
            ("温度", row.get("temperature")),
            ("时间", row.get("duration") or row.get("time")),
            ("收率", row.get("yield") or row.get("yield_percent")),
        )
        for label, value in legacy_fields:
            add(label, value)
    locator = str(row.get("source_locator") or "").strip()
    if locator and any(token in locator.lower() for token in ("condition", "reagent", "arrow", "scheme")):
        add("来源位置", locator)
    return _normalize_condition_rows(out)


def _condition_value_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _condition_value_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = [_condition_value_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def _normalize_condition_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        label = _clean_label(row.get("label") or "条件")[:36] or "条件"
        value = str(row.get("value") or "").strip()
        if not value:
            continue
        value = re.sub(r"\s+", " ", value)[:500]
        key = (label.lower(), value.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": label, "value": value})
    return out[:8]


def _condition_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    parts = [f"{row.get('label')}: {row.get('value')}" for row in rows[:2]]
    text = "；".join(parts)
    return text[:240]


def _condition_status(rows: list[dict[str, str]], missing: list[str]) -> str:
    if rows:
        return "available"
    missing_text = " | ".join(str(item or "") for item in missing).lower()
    if "not shown" in missing_text or "no reaction condition" in missing_text or "conditions_not_shown" in missing_text:
        return "not_shown"
    if "condition" in missing_text or "conditions" in missing_text:
        return "not_compiled"
    return "not_recorded"


def _route_from_parent_route_proof(proof: dict[str, Any]) -> dict[str, Any]:
    if str(proof.get("proof_mode") or "") != "direct_parent_route":
        return {}
    evidence = dict(proof.get("proof_evidence") or {})
    parent_verifier = dict(evidence.get("parent_verifier") or {})
    embedded = parent_verifier.get("accepted_route")
    if isinstance(embedded, dict) and isinstance(embedded.get("steps"), list):
        return dict(embedded)
    return {}


def _route_structure_sha256(steps: Any) -> str:
    """Digest the exact materialized reaction interfaces of a selected route."""
    if not isinstance(steps, list) or not steps:
        return ""
    normalized: list[dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, dict):
            return ""
        product_raw = _route_step_product(raw)
        reactants_raw = _route_step_reactants(raw)
        product = _canonical_molecule_smiles(product_raw)
        reactants = sorted(
            _canonical_molecule_smiles(value)
            for value in reactants_raw
            if _canonical_molecule_smiles(value)
        )
        if not product or not reactants or len(reactants) != len(reactants_raw):
            return ""
        mapped = str(
            raw.get("atom_mapped_reaction_smiles")
            or raw.get("mapped_reaction_smiles")
            or raw.get("reaction_smiles")
            or ""
        ).strip()
        normalized.append(
            {
                "product_canonical_isomeric_smiles": product,
                "reactant_canonical_isomeric_smiles": reactants,
                "atom_mapped_reaction_smiles": mapped if ">>" in mapped else "",
            }
        )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _revalidated_stitched_proof_projection(
    value: Any,
    *,
    expected_target_smiles: str,
) -> dict[str, Any]:
    """Return only chemistry reconstructed from a valid stitched proof input.

    This function intentionally ignores every route-shaped field outside
    ``proof_evidence.stitched_route.proof_inputs``.  Both the subgoal route and
    the literature chain are revalidated before any display node is created.
    """
    if not isinstance(value, dict) or not str(expected_target_smiles or "").strip():
        return {}
    proof = dict(value)
    if proof.get("proof_mode") != "stitched_parent_route" or not is_solved_parent_route_proof(
        proof,
        expected_target_smiles=expected_target_smiles,
    ):
        return {}
    evidence = proof.get("proof_evidence")
    if not isinstance(evidence, dict):
        return {}
    stitched = evidence.get("stitched_route")
    if not isinstance(stitched, dict):
        return {}
    inputs = stitched.get("proof_inputs")
    if not isinstance(inputs, dict) or inputs.get("schema_version") != "stitched_semisynthesis_proof_inputs.v1":
        return {}
    chain = inputs.get("literature_chain_audit")
    selected_subgoal = inputs.get("selected_subgoal")
    provided_verifier = inputs.get("provided_subgoal_verifier")
    raw = inputs.get("subgoal_raw_result")
    stored_expansion = inputs.get("route_expansion_result")
    if not isinstance(stored_expansion, dict):
        stored_expansion = {"subgoals": [dict(selected_subgoal or {})]}
    if not all(
        isinstance(item, dict)
        for item in (chain, selected_subgoal, provided_verifier, raw, stored_expansion)
    ):
        return {}

    recomputed = compile_stitched_semisynthesis_route(
        literature_chain_audit=dict(chain),
        route_expansion_result=dict(stored_expansion),
        subgoal_verifier=dict(provided_verifier),
        subgoal_raw_result=dict(raw),
        target_smiles=expected_target_smiles,
        target_name=str(inputs.get("target_name") or ""),
        case_id=str(inputs.get("case_id") or ""),
    )
    if not (
        recomputed.get("accepted") is True
        and recomputed.get("solved") is True
        and str(recomputed.get("route_status") or "") == "solved"
    ):
        return {}

    if not _same_molecule(str(chain.get("target_smiles") or ""), expected_target_smiles):
        return {}

    raw_literature_steps = chain.get("chain") or chain.get("steps") or []
    if not isinstance(raw_literature_steps, list):
        return {}
    literature_rows = [dict(row) for row in raw_literature_steps if isinstance(row, dict)]
    if not literature_rows or not all(
        is_validated_source_detail_literature_step(row) for row in literature_rows
    ):
        return {}
    literature_steps = _forward_synthesis_step_order(
        literature_rows,
        target_smiles=expected_target_smiles,
    )
    if not literature_steps:
        return {}
    if len(literature_steps) != int((recomputed.get("literature_chain") or {}).get("step_count") or 0):
        return {}
    if not _same_molecule(_route_step_product(literature_steps[-1]), expected_target_smiles):
        return {}

    frontier_smiles = _dedupe(
        [
            str(smiles)
            for smiles in (recomputed.get("literature_chain") or {}).get("graph_terminal_frontier") or []
            if str(smiles or "").strip()
        ]
    )
    coverage = dict(recomputed.get("frontier_coverage_audit") or {})
    if not (
        frontier_smiles
        and coverage.get("accepted") is True
        and int(coverage.get("frontier_count") or 0) == len(frontier_smiles)
        and int(coverage.get("closed_frontier_count") or 0) == len(frontier_smiles)
    ):
        return {}
    expansion_rows = [
        dict(row)
        for row in stored_expansion.get("subgoals") or []
        if isinstance(row, dict)
    ]
    subgoal_segments: list[dict[str, Any]] = []
    stock_terminal_smiles: list[str] = []
    recomputed_closures = [
        dict(row)
        for row in recomputed.get("subgoal_closures") or []
        if isinstance(row, dict)
    ]
    for frontier in frontier_smiles:
        candidates = [
            row
            for row in expansion_rows
            if _same_molecule(_subgoal_projection_target_smiles(row), frontier)
        ]
        if len(candidates) != 1:
            return {}
        candidate = candidates[0]
        candidate_raw = candidate.get("raw_result") or candidate.get("result")
        candidate_verifier = candidate.get("verifier")
        if not isinstance(candidate_raw, dict) or not isinstance(candidate_verifier, dict):
            return {}
        if not is_accepted_route_verifier_report(
            candidate_verifier,
            expected_target_smiles=frontier,
        ):
            return {}
        reverified = verify_chemenzy_raw_routes(
            dict(candidate_raw),
            target_smiles=frontier,
        )
        if not is_accepted_route_verifier_report(
            reverified,
            expected_target_smiles=frontier,
        ):
            return {}
        if (
            candidate_verifier.get("best_route_rank") != reverified.get("best_route_rank")
            or int(candidate_verifier.get("best_route_step_count") or 0)
            != int(reverified.get("best_route_step_count") or 0)
        ):
            return {}
        accepted_route = reverified.get("accepted_route")
        if not isinstance(accepted_route, dict):
            return {}
        subgoal_steps = _forward_synthesis_step_order(
            [
                dict(row)
                for row in accepted_route.get("steps") or []
                if isinstance(row, dict)
            ],
            target_smiles=frontier,
        )
        closure = next(
            (
                row
                for row in recomputed_closures
                if _same_molecule(
                    str((row.get("frontier") or {}).get("input_smiles") or ""),
                    frontier,
                )
            ),
            {},
        )
        if (
            not subgoal_steps
            or len(subgoal_steps) != int(closure.get("best_route_step_count") or 0)
            or not _same_molecule(_route_step_product(subgoal_steps[-1]), frontier)
        ):
            return {}
        product_keys = {
            _canonical_molecule_smiles(_route_step_product(row))
            for row in subgoal_steps
        }
        segment_stock = _dedupe(
            [
                smiles
                for row in subgoal_steps
                for smiles in _route_step_reactants(row)
                if _canonical_molecule_smiles(smiles) not in product_keys
            ]
        )
        if not segment_stock:
            return {}
        stock_terminal_smiles.extend(segment_stock)
        subgoal_segments.append(
            {
                "frontier_smiles": frontier,
                "steps": subgoal_steps,
                "stock_terminal_smiles": segment_stock,
                "reaction_validation": dict(reverified.get("reaction_validation") or {}),
            }
        )
    if (
        sum(len(segment["steps"]) for segment in subgoal_segments) + len(literature_steps)
        != int((recomputed.get("combined_route") or {}).get("combined_step_count") or 0)
    ):
        return {}
    return {
        "subgoal_segments": subgoal_segments,
        "literature_steps": literature_steps,
        "stock_terminal_smiles": _dedupe(stock_terminal_smiles),
        "literature_frontier_smiles": frontier_smiles,
        "literature_source_ref": str(chain.get("source_ref") or ""),
    }


def _subgoal_projection_target_smiles(row: dict[str, Any]) -> str:
    raw = row.get("raw_result") or row.get("result")
    raw_result = dict((raw or {}).get("result") or raw or {}) if isinstance(raw, dict) else {}
    verifier = dict(row.get("verifier") or {})
    audit = dict(verifier.get("target_equivalence_audit") or {})
    selected = dict(row.get("subgoal") or {})
    return str(
        row.get("frontier_smiles")
        or audit.get("request_canonical_isomeric_smiles")
        or audit.get("request_target_smiles")
        or raw_result.get("target")
        or raw_result.get("target_smiles")
        or selected.get("smiles")
        or ""
    ).strip()


def _forward_synthesis_step_order(
    steps: list[dict[str, Any]],
    *,
    target_smiles: str,
) -> list[dict[str, Any]]:
    """Topologically order retrosynthetic edges in forward synthesis order."""
    if not steps:
        return []
    by_product: dict[str, dict[str, Any]] = {}
    reactants_by_product: dict[str, list[str]] = {}
    for row in steps:
        product = _route_step_product(row)
        reactants = _route_step_reactants(row)
        product_key = _canonical_molecule_smiles(product)
        reactant_keys = [_canonical_molecule_smiles(item) for item in reactants]
        if (
            not product_key
            or not reactants
            or any(not item for item in reactant_keys)
            or product_key in by_product
        ):
            return []
        by_product[product_key] = dict(row)
        reactants_by_product[product_key] = reactant_keys

    target_key = _canonical_molecule_smiles(target_smiles)
    if not target_key or target_key not in by_product:
        return []
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(product_key: str) -> bool:
        if product_key in visited:
            return True
        if product_key in visiting:
            return False
        visiting.add(product_key)
        for reactant_key in reactants_by_product[product_key]:
            if reactant_key in by_product and not visit(reactant_key):
                return False
        visiting.remove(product_key)
        visited.add(product_key)
        ordered.append(dict(by_product[product_key]))
        return True

    if not visit(target_key) or len(visited) != len(by_product):
        return []
    return ordered


def _route_step_product(step: dict[str, Any]) -> str:
    product = str(
        step.get("product")
        or step.get("product_smiles")
        or step.get("final_product_smiles")
        or ""
    ).strip()
    if product:
        return product
    reaction_smiles = str(step.get("reaction_smiles") or "").strip()
    return reaction_smiles.split(">>", 1)[1].strip() if ">>" in reaction_smiles else ""


def _canonical_molecule_smiles(smiles: str) -> str:
    _, canonical = _molecule_node_identity(smiles=str(smiles or ""), label=str(smiles or ""))
    return canonical


def _semantic_reaction_edge_key(product_smiles: str, precursor_smiles: list[str]) -> str:
    """Canonical identity for one retrosynthetic hyperedge across projections."""
    product = _canonical_molecule_smiles(product_smiles) or str(product_smiles or "").strip()
    precursors = sorted(
        _canonical_molecule_smiles(value) or str(value or "").strip()
        for value in precursor_smiles
        if str(value or "").strip()
    )
    return f"{product}>>{'.'.join(precursors)}" if product and precursors else ""


def _route_step_reactants(step: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in (
        "main_reactant",
        "main_reactant_smiles",
        "reactant",
        "reactant_smiles",
        "precursor_smiles",
    ):
        value = step.get(key)
        if value:
            out.extend(_labels_from_any(value))
    for item in step.get("aux_reactants") or step.get("reactants") or []:
        out.extend(_labels_from_any(item))
    reaction_smiles = str(step.get("reaction_smiles") or "").strip()
    if not out and ">>" in reaction_smiles:
        left = reaction_smiles.split(">>", 1)[0]
        out.extend(_labels_from_any(left))
    return _dedupe([item for item in out if item])


def _confidence_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.65:
        return "medium_high"
    if score >= 0.4:
        return "medium"
    return "low"


def _route_step_confidence(step: dict[str, Any]) -> str:
    scores = dict(step.get("scores") or {})
    raw = scores.get("confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = None
    if value is None:
        return "medium"
    if value >= 0.75:
        return "high"
    if value >= 0.35:
        return "medium"
    return "low"


def _consensus_precursor_smiles(proposal: dict[str, Any]) -> list[str]:
    raw = proposal.get("precursor_smiles")
    if isinstance(raw, str):
        values = raw.split(".")
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = []
    return _dedupe([str(value).strip() for value in values if str(value).strip()])[:12]


def _source_candidate_has_real_source(row: dict[str, Any]) -> bool:
    if bool(row.get("placeholder_only")):
        return False
    if str(row.get("access_status") or "").strip().lower() == "placeholder_only":
        return False
    if str(row.get("source_type") or "").strip().lower() == "placeholder_query":
        return False
    if str(row.get("source_discovery_mode") or "").strip().lower() == "placeholder":
        return False
    locators = [
        row.get("doi"),
        row.get("pii"),
        row.get("url"),
        row.get("source_ref"),
    ]
    local_path = str(row.get("local_pdf") or row.get("pdf_path") or "").strip()
    if local_path:
        locators.append(
            local_path if local_path.lower().startswith("local_pdf:") else f"local_pdf:{local_path}"
        )
    return any(canonical_traceable_source_ref(value) for value in locators)


def _placeholder_source_ref(value: Any) -> bool:
    return not bool(canonical_traceable_source_ref(value))


def _consensus_support_group(
    source_channel: str,
    evidence_level: str,
    source_refs: list[str],
    evidence_refs: list[str],
    *,
    authority_bound: bool | None = None,
    authority_basis: str = "",
) -> str:
    """Derive support identity without trusting producer-authored groups."""

    normalized_channel = str(source_channel or "other").strip().lower().replace(
        "-", "_"
    )
    if normalized_channel in {"chem_enzy", "template", "stock"}:
        expected_basis = {
            "chem_enzy": "deterministic_chemenzy_adapter",
            "template": "deterministic_template_adapter",
            "stock": "deterministic_stock_provider",
        }[normalized_channel]
        if authority_bound is not True:
            return (
                "codex_model"
                if authority_bound is False
                else "legacy_unverified_support"
            )
        if str(authority_basis or "") != expected_basis:
            return "legacy_unverified_support"

    return source_record_support_group(
        source_channel,
        evidence_level,
        source_refs,
        evidence_refs,
    )


def _source_ref_values(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []
    return _dedupe([str(item) for item in values if str(item).strip()])


def _consensus_support_records(
    proposal: dict[str, Any],
    *,
    limit: int | None = 32,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in proposal.get("source_records") or []:
        if not isinstance(raw, dict):
            continue
        source_channel = str(raw.get("source_channel") or "other")
        evidence_level = str(raw.get("evidence_level") or "model_only")
        source_refs = _source_ref_values(raw.get("source_refs"))
        evidence_refs = _source_ref_values(raw.get("evidence_refs"))
        authority_bound = (
            raw.get("authority_bound")
            if isinstance(raw.get("authority_bound"), bool)
            else None
        )
        authority_basis = str(raw.get("authority_basis") or "")
        records.append(
            {
                "candidate_id": str(raw.get("candidate_id") or ""),
                "source_channel": source_channel,
                "evidence_level": evidence_level,
                "confidence": str(raw.get("confidence") or "low"),
                "support_group": _consensus_support_group(
                    source_channel,
                    evidence_level,
                    source_refs,
                    evidence_refs,
                    authority_bound=authority_bound,
                    authority_basis=authority_basis,
                ),
                "declared_support_group": str(raw.get("support_group") or ""),
                "source_refs": source_refs,
                "evidence_refs": evidence_refs,
                "authority_bound": authority_bound,
                "authority_basis": authority_basis,
            }
        )
    if limit is None:
        return records
    return records[: max(0, int(limit))]


def _consensus_independent_support_groups(
    proposal: dict[str, Any],
    support_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    # Always recompute from the complete producer record set.  The optional
    # argument exists for compatibility with older callers but a display slice
    # must never determine source diversity.
    del support_records
    all_records = _consensus_support_records(proposal, limit=None)
    if all_records:
        return _dedupe(
            [
                _consensus_support_group(
                    str(row.get("source_channel") or "other"),
                    str(row.get("evidence_level") or "model_only"),
                    _source_ref_values(row.get("source_refs")),
                    _source_ref_values(row.get("evidence_refs")),
                    authority_bound=(
                        row.get("authority_bound")
                        if isinstance(row.get("authority_bound"), bool)
                        else None
                    ),
                    authority_basis=str(row.get("authority_basis") or ""),
                )
                for row in all_records
            ]
        )

    # Legacy payloads without source records cannot establish independent
    # groups, regardless of how many groups/channels they declare.
    channels = _source_ref_values(proposal.get("source_channels"))
    declared_groups = _source_ref_values(proposal.get("independent_support_groups"))
    has_codex = any(
        value.lower().startswith("codex") for value in [*channels, *declared_groups]
    )
    if has_codex:
        return ["codex_model"]
    if (
        channels
        or declared_groups
        or _source_ref_values(proposal.get("source_refs"))
        or _source_ref_values(proposal.get("evidence_refs"))
    ):
        return ["legacy_unverified_support"]
    return []


def _authoritative_support_group(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(
        text.startswith("literature:")
        or text.startswith("validated:")
        or text.startswith("computational:")
        or text.startswith("doi:")
        or text.startswith("patent:")
        or text.startswith("patent_family:")
        or text.startswith("pmid:")
        or text.startswith("pmc:")
        or text.startswith("pii:")
    )


def _validated_edge_evidence_binding_set(value: Any) -> dict[str, Any]:
    row = dict(value) if isinstance(value, Mapping) else {}
    if row.get("schema_version") != "edge_evidence_binding_set.v1":
        return {}
    supplied_digest = str(row.get("content_sha256") or "")
    digest_payload = {
        key: item for key, item in row.items() if key != "content_sha256"
    }
    if not supplied_digest or supplied_digest != _canonical_json_sha256(
        digest_payload
    ):
        return {}
    groups = [
        str(item)
        for item in row.get("independent_trusted_source_groups") or []
        if str(item)
    ]
    trusted_bindings = [
        dict(item)
        for item in row.get("bindings") or []
        if isinstance(item, Mapping) and item.get("trusted") is True
    ]
    observed_groups = sorted(
        {
            str(item.get("independent_source_group") or "")
            for item in trusted_bindings
            if str(item.get("independent_source_group") or "")
        }
    )
    if sorted(set(groups)) != observed_groups:
        return {}
    if int(row.get("trusted_binding_count") or 0) != len(trusted_bindings):
        return {}
    if bool(row.get("corroborated")) != (len(observed_groups) >= 2):
        return {}
    return row


def _validated_edge_evidence_projection(
    graph: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    projection = dict(graph.get("edge_evidence_binding_sets") or {})
    if projection.get("schema_version") != "edge_evidence_binding_projection.v1":
        return {}
    supplied_digest = str(projection.get("content_sha256") or "")
    digest_payload = {
        key: item
        for key, item in projection.items()
        if key != "content_sha256"
    }
    if not supplied_digest or supplied_digest != _canonical_json_sha256(
        digest_payload
    ):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for reaction_digest, raw in dict(
        projection.get("by_reaction_digest") or {}
    ).items():
        row = _validated_edge_evidence_binding_set(raw)
        if not row or str(row.get("reaction_digest") or "") != str(
            reaction_digest
        ):
            continue
        rows[str(reaction_digest)] = row
    return rows


def _edge_evidence_binding_for_graph_step(
    graph: Mapping[str, Any],
    step: Mapping[str, Any],
) -> dict[str, Any]:
    reaction_digest = canonical_reaction_digest(
        str(step.get("product_smiles") or ""),
        [str(item) for item in step.get("precursor_smiles") or []],
    )
    return dict(
        _validated_edge_evidence_projection(graph).get(reaction_digest) or {}
    )


def _consensus_direct_source_refs(proposal: dict[str, Any]) -> list[str]:
    values = [
        *_source_ref_values(proposal.get("source_refs")),
        *_source_ref_values(proposal.get("evidence_refs")),
    ]
    for record in _consensus_support_records(proposal, limit=None):
        values.extend(str(item) for item in record.get("source_refs") or [])
        values.extend(str(item) for item in record.get("evidence_refs") or [])
    return _dedupe(values)[:32]


def _consensus_conflicts(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    conflicts = [
        dict(row) for row in proposal.get("condition_conflicts") or [] if isinstance(row, dict)
    ]
    reaction_families = _dedupe([str(item) for item in proposal.get("reaction_families") or []])
    if len(reaction_families) > 1:
        conflicts.append(
            {
                "field": "reaction_family",
                "values": reaction_families,
                "requires_review": True,
            }
        )
    return conflicts[:16]


def _same_molecule(left: str, right: str) -> bool:
    left_id, left_canonical = _molecule_node_identity(smiles=left, label=left)
    right_id, right_canonical = _molecule_node_identity(smiles=right, label=right)
    if left_canonical and right_canonical:
        return left_id == right_id
    return _same_text(left, right)


def _same_text(left: str, right: str) -> bool:
    return str(left or "").strip() == str(right or "").strip()


def _compact_smiles_label(smiles: str, *, max_len: int = 58) -> str:
    text = str(smiles or "").strip()
    if len(text) <= max_len:
        return text
    keep = max(12, (max_len - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def _template_endpoint_labels(template: dict[str, Any]) -> tuple[str, str]:
    """Return advisory endpoint labels without implying a target connection."""
    logic = _clean_label(template.get("transform_logic") or "")
    for separator in ("->", "→", "=>"):
        if separator not in logic:
            continue
        left, right = (part.strip() for part in logic.split(separator, 1))
        if left and right:
            return left, right
    precursor = _clean_label(
        template.get("reactant_label")
        or template.get("from_label")
        or template.get("preserved_scaffold")
        or "unbound template precursor"
    )
    product = _clean_label(
        template.get("product_label")
        or template.get("to_label")
        or f"unbound template product: {template.get('objective_type') or 'unspecified'}"
    )
    return precursor, product


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:90] or "item"


def _display_text_is_corrupt(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    mojibake_markers = (
        "鎺",
        "璺",
        "妫",
        "鍥",
        "鏂",
        "閰",
        "姣嶆",
        "涓荤",
        "氱偣",
        "囬€",
        "惧儚",
    )
    return "�" in text or "\ufffd" in text or any(marker in text for marker in mojibake_markers)


def _branch_title_for_display(*, branch_id: str, title: str, kind: str) -> str:
    if not _display_text_is_corrupt(title):
        return str(title or "")
    return {
        "recommended_strategy": "推荐路线",
        "visual_chain": "图像证据分支",
        "stitched_verified_route": "拼接验证路线",
        "process_evidence": "文献工艺锚点",
        "route_consensus": "共识候选",
        "retrosynthetic_proposal": "备选逆合成分支",
        "broad_template": "通用模板分支",
        "direct_verified_route": "已验证路线",
        "subgoal_verified_route": "子目标闭合路线",
        "exact_literature": "exact row 路线",
        "diagnostic_failure": "探索诊断",
    }.get(str(kind or ""), str(title or "route branch"))


def _recommendation_for_display(*, kind: str, recommendation: str) -> str:
    if not _display_text_is_corrupt(recommendation):
        return str(recommendation or "")
    return {
        "recommended_strategy": "主推荐",
        "stitched_verified_route": "拼接验证",
        "visual_chain": "支持/备选",
        "process_evidence": "工艺锚点",
        "route_consensus": "共识建议",
        "retrosynthetic_proposal": "探索备选",
        "broad_template": "模板提示",
        "direct_verified_route": "已验证",
        "subgoal_verified_route": "子目标闭合",
        "exact_literature": "强证据",
        "diagnostic_failure": "诊断",
    }.get(str(kind or ""), str(recommendation or ""))


def _module_label_for_key(key: str) -> str:
    return {
        "sidechain_installation": "侧链安装",
        "amide_or_sidechain_assembly": "酰胺 / 侧链连接",
        "protection_state_adjustment": "保护基 / 脱保护调整",
        "semisynthesis_anchor": "半合成锚点",
        "heterocycle_core_construction": "杂环母核构建",
        "scaffold_core_construction": "骨架构建 / 母核调整",
        "visual_literature_hint": "图像文献提示",
        "other_route_module": "其他路线模块",
        "ketal_deprotection": "缩酮脱保护",
        "ester_hydrolysis": "酯水解",
        "salt_formation": "成盐 / 分离",
        "form_adjustment": "盐型 / 游离酸调整",
        "subgoal_stock_closure": "ChemEnzy 子目标闭合",
        "diagnostic_failure": "诊断失败",
        "visual_failed_or_empty": "图像链失败或为空",
    }.get(str(key or ""), str(key or ""))


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text[:180]


def _molecule_identity_namespace(
    *,
    branch_id: str,
    source_refs: list[str] | tuple[str, ...] = (),
    evidence_row_id: str = "",
) -> str:
    """Build a stable scope for an assertion that has no structure identity."""
    payload = {
        "branch_id": str(branch_id or "unscoped").strip(),
        "source_refs": sorted({str(value).strip() for value in source_refs if str(value).strip()}),
        "evidence_row_id": str(evidence_row_id or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _molecule_node_identity(*, smiles: str, label: str, namespace: str = "") -> tuple[str, str]:
    """Return a collision-resistant ID while preserving stereochemistry."""
    text = str(smiles or "").strip()
    canonical = ""
    if text and Chem is not None:
        try:
            mol = Chem.MolFromSmiles(text)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            canonical = ""
    identity = (
        f"smiles:{canonical}"
        if canonical
        else (
            f"name:{_clean_label(label).casefold()}|"
            f"namespace:{str(namespace or 'unscoped').strip()}"
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"mol:{digest}", canonical


def _portfolio_proof_level(value: Any) -> int:
    if isinstance(value, dict):
        raw = value.get("level")
        if raw is None:
            raw = value.get("level_index")
        if raw is None:
            raw = value.get("achieved_proof_level")
        if raw is None:
            raw = value.get("proof_level")
        value = raw
    if isinstance(value, str) and not value.strip().isdigit():
        value = {
            "L0_materialized": 0,
            "L1_graph_and_stock_closed": 1,
            "L1_graph_stock_closed": 1,
            "L2_mapping_consistent": 2,
            "L2_reaction_validated": 2,
            "L3_precedent_supported": 3,
            "L4_procurement_ready": 4,
        }.get(value, 0)
    try:
        return max(0, min(4, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _portfolio_edge_binding_reasons(
    binding: dict[str, Any],
    *,
    edge_id: str,
    product_id: str,
    precursor_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    if not binding:
        return ["missing_exact_edge_proof_binding"]
    if binding.get("schema_version") != "exact_edge_proof_binding.v1":
        reasons.append("invalid_schema")
    if str(binding.get("hyperedge_id") or "") != edge_id:
        reasons.append("hyperedge_id_mismatch")
    if str(binding.get("product_molecule_id") or "") != product_id:
        reasons.append("product_molecule_id_mismatch")
    if sorted(str(value) for value in binding.get("precursor_molecule_ids") or []) != sorted(
        precursor_ids
    ):
        reasons.append("precursor_molecule_ids_mismatch")
    if not _portfolio_binding_digest_valid(binding):
        reasons.append("binding_sha256_mismatch")
    named_level = str(binding.get("proof_level") or "")
    expected_level = {
        "L0_materialized": 0,
        "L1_graph_and_stock_closed": 1,
        "L2_mapping_consistent": 0,
        "L2_reaction_validated": 2,
        "L3_precedent_supported": 3,
        "L4_procurement_ready": 4,
    }.get(named_level)
    bound_level = _portfolio_proof_level(binding.get("portfolio_proof_level"))
    if expected_level is None or bound_level != expected_level:
        reasons.append("proof_level_portfolio_level_mismatch")
    if binding.get("advisory") is not (bound_level < 2):
        reasons.append("advisory_flag_mismatch")
    if bound_level >= 2 and binding.get("proof_accepted") is not True:
        reasons.append("portfolio_level_requires_accepted_proof")
    proof_source = str(binding.get("proof_source") or "")
    if proof_source not in {
        "route_proof_bank.v1",
        "legacy_best_accepted_route",
    }:
        reasons.append("invalid_proof_source")
    for field in ("proof_digest", "route_proof_digest", "reaction_digest"):
        if re.fullmatch(r"[0-9a-f]{64}", str(binding.get(field) or "").lower()) is None:
            reasons.append(f"invalid_{field}")
    if named_level in {"L3_precedent_supported", "L4_procurement_ready"} and re.fullmatch(
        r"[0-9a-f]{64}",
        str(binding.get("trusted_precedent_sha256") or "").lower(),
    ) is None:
        reasons.append("invalid_trusted_precedent_sha256")
    if proof_source == "route_proof_bank.v1" and (
        not str(binding.get("proof_bank_entry_id") or "").strip()
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(binding.get("proof_bank_entry_sha256") or "").lower(),
        )
        is None
    ):
        reasons.append("invalid_proof_bank_authority")
    return reasons


def _portfolio_stock_binding_reasons(
    binding: dict[str, Any],
    *,
    molecule_id: str,
    canonical_smiles: str,
) -> list[str]:
    reasons: list[str] = []
    if not binding:
        return ["missing_exact_stock_binding"]
    if binding.get("schema_version") != "exact_stock_binding.v1":
        reasons.append("invalid_schema")
    if str(binding.get("molecule_id") or "") != molecule_id:
        reasons.append("molecule_id_mismatch")
    if str(binding.get("canonical_isomeric_smiles") or "") != canonical_smiles:
        reasons.append("canonical_isomeric_smiles_mismatch")
    if not str(binding.get("catalog_id") or "").strip():
        reasons.append("missing_catalog_id")
    if re.fullmatch(r"[0-9a-f]{64}", str(binding.get("catalog_sha256") or "").lower()) is None:
        reasons.append("invalid_catalog_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(binding.get("evidence_sha256") or "").lower()) is None:
        reasons.append("invalid_evidence_sha256")
    if not str(binding.get("lookup_basis") or "").strip():
        reasons.append("missing_lookup_basis")
    if str(binding.get("binding_authority") or "") not in {
        "strictly_replayed_route_proof_bank.v1",
        "legacy_best_route_independent_stock_audit",
    }:
        reasons.append("invalid_binding_authority")
    if not _portfolio_binding_digest_valid(binding):
        reasons.append("binding_sha256_mismatch")
    return reasons


def _portfolio_binding_digest_valid(binding: dict[str, Any]) -> bool:
    expected = str(binding.get("binding_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        return False
    payload = dict(binding)
    payload.pop("binding_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest() == expected


def _portfolio_content_digest_valid(value: dict[str, Any]) -> bool:
    expected = str(value.get("content_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        return False
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest() == expected


def _portfolio_proof_tier(level: int) -> str:
    return {
        0: "L0_materialized",
        1: "L1_graph_and_stock_closed",
        2: "L2_reaction_validated",
        3: "L3_precedent_supported",
        4: "L4_procurement_ready",
    }.get(max(0, min(4, int(level))), "L0_materialized")


def _portfolio_selection_acyclic(
    selections: list[tuple[str, str, dict[str, Any], int]],
) -> bool:
    nodes: set[str] = set()
    outgoing: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for product_id, _, edge, _ in selections:
        nodes.add(product_id)
        outgoing.setdefault(product_id, set())
        indegree.setdefault(product_id, 0)
        for raw_precursor in edge.get("precursor_molecule_ids") or []:
            precursor_id = str(raw_precursor or "")
            if not precursor_id:
                continue
            nodes.add(precursor_id)
            outgoing.setdefault(precursor_id, set())
            indegree.setdefault(precursor_id, 0)
            if product_id not in outgoing[precursor_id]:
                outgoing[precursor_id].add(product_id)
                indegree[product_id] = indegree.get(product_id, 0) + 1
    ready = sorted(node for node in nodes if indegree.get(node, 0) == 0)
    visited = 0
    while ready:
        node = ready.pop(0)
        visited += 1
        for target in sorted(outgoing.get(node, set())):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return visited == len(nodes)


def _exact_row_is_verified(row: dict[str, Any]) -> bool:
    if row.get("accepted") is False:
        return False
    return is_validated_source_detail_literature_step(row)


def _normalize_confidence(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    if text in CONFIDENCE_RANK:
        return text
    if "high" in text:
        return "high"
    if "low" in text:
        return "low"
    if "fail" in text or "reject" in text:
        return "failed"
    return "medium"


def _normalize_exactness(value: Any) -> str:
    text = str(value or "name_only").strip().lower()
    if text in EXACTNESS_RANK:
        return text
    if "exact" in text:
        return "exact_literature_row"
    if "visual" in text:
        return "visual_inferred"
    if "hypothesis" in text or "model" in text:
        return "model_hypothesis"
    if "fail" in text or "unresolved" in text:
        return "failed_or_unresolved"
    if "literature" in text or "named" in text:
        return "named_literature"
    return "name_only"


def _worst_ranked(a: Any, b: Any, rank: dict[str, int]) -> str:
    av = str(a or "")
    bv = str(b or "")
    return av if rank.get(av, 0) <= rank.get(bv, 0) else bv


def _better_node_label(existing: Any, new: Any) -> str:
    old = _clean_label(existing)
    fresh = _clean_label(new)
    if not old:
        return fresh or "unnamed node"
    if not fresh:
        return old
    generic_prefixes = ("proposal precursor", "visual precursor", "visual product", "target", "unnamed node")
    old_generic = old.lower().startswith(generic_prefixes)
    fresh_generic = fresh.lower().startswith(generic_prefixes)
    if old_generic and not fresh_generic:
        return fresh
    if fresh_generic and not old_generic:
        return old
    if "/" in old and "/" not in fresh:
        return old
    if "/" in fresh and "/" not in old:
        return fresh
    return old if len(old) >= len(fresh) else fresh


def _preferred_node_role(roles: list[str]) -> str:
    priority = {
        "target": 100,
        "literature_terminal": 90,
        "stock_terminal": 80,
        "literature_intermediate": 60,
        "stitched_route_intermediate": 50,
    }
    cleaned = [str(role) for role in roles if str(role or "").strip()]
    if not cleaned:
        return "intermediate"
    return max(cleaned, key=lambda role: priority.get(role, 0))


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
