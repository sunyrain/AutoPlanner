"""Planning, context, innovation, and explicit claim operations for the V4 service."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from cascade_planner.application.campaign_context import CampaignContext
from cascade_planner.application.canonical_hypergraph import CanonicalIngestionBatch
from cascade_planner.application.proof_portfolio import compile_proof_portfolio
from cascade_planner.orchestration import program_innovation_runtime
from cascade_planner.orchestration import route_innovation_runtime
from cascade_planner.orchestration.experimental_claim_admission_runtime import (
    admit_route_experimental_claims,
)
from cascade_planner.orchestration.global_campaign_director import (
    DirectorOutcome,
    director_plan_provenance_sha256,
)


class _RetrosynthesisServicePlanningMixin:
    def apply_global_plan(
        self,
        plan: Mapping[str, Any],
        *,
        idempotency_key: str,
        proposal_origin_kind: str = "manual",
        proposal_origin_ref: str = "",
    ) -> dict[str, Any]:
        trusted_plan = {
            key: value
            for key, value in dict(plan).items()
            if key not in {"_proposal_origin_kind", "_proposal_origin_ref"}
        }
        trusted_plan["_proposal_origin_kind"] = str(proposal_origin_kind).lower()
        trusted_plan["_proposal_origin_ref"] = str(proposal_origin_ref)
        return self.apply_batch(
            CanonicalIngestionBatch(global_plans=(trusted_plan,)),
            idempotency_key=idempotency_key,
        )

    def run_global_director(
        self,
        *,
        mode: str,
        material_events: Iterable[str] = (),
        evidence_observations: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        force: bool = False,
        idempotency_key: str,
    ) -> DirectorOutcome:
        context = self.compile_global_context(
            material_events=material_events,
            evidence_observations=evidence_observations,
        )
        return self.run_global_director_with_context(
            context,
            mode=mode,
            force=force,
            idempotency_key=idempotency_key,
        )

    def run_global_director_with_context(
        self,
        context: CampaignContext,
        *,
        mode: str,
        force: bool = False,
        before_plan_admission: Callable[[], None] | None = None,
        idempotency_key: str,
    ) -> DirectorOutcome:
        """Run against an already frozen canonical context and merge by union."""

        outcome = self.director.run(context, mode=mode, force=force)
        self._previous_context = context
        if before_plan_admission is not None:
            before_plan_admission()
        if outcome.plan is not None and outcome.status == "accepted":
            admitted_ids = sorted(
                str(row.get("proposal_id") or "")
                for row in outcome.proposal_audits
                if row.get("accepted") is True and str(row.get("proposal_id") or "")
            )
            self.apply_global_plan(
                {**outcome.plan.to_dict(), "_host_admitted_proposal_ids": admitted_ids},
                idempotency_key=f"{idempotency_key}:plan",
                proposal_origin_kind="codex_global_director",
                proposal_origin_ref=(
                    "director_plan:"
                    f"{director_plan_provenance_sha256(outcome.plan)}"
                ),
            )
        if mode == "initial_architecture" and outcome.status in {
            "accepted",
            "budget_exhausted",
        }:
            self._record_settled_initial_architecture(
                outcome,
                idempotency_key=f"{idempotency_key}:settled",
            )
        return outcome

    def _record_settled_initial_architecture(
        self,
        outcome: DirectorOutcome,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Persist a terminal initial-architecture attempt as operational state.

        The record is deliberately separate from accepted chemistry hypotheses:
        an accepted Director response may still have every proposal rejected by
        host admission.  That is a settled attempt, not a reason to schedule the
        same initial pass after every unrelated graph revision.
        """

        graph = self.graph_store.load()
        target_id = str(graph.get("target_molecule_id") or "")
        audits = [dict(row) for row in outcome.proposal_audits]
        accepted_count = sum(row.get("accepted") is True for row in audits)
        signal_id = f"director-attempt:initial_architecture:{target_id}"
        return self.publish_action_signals(
            (
                {
                    "signal_id": signal_id,
                    "deficit_id": signal_id,
                    "kind": "architecture",
                    "status": "resolved",
                    "object_id": target_id,
                    "entity_ids": [target_id],
                    "route_family_ids": [],
                    "dependency_ids": [],
                    "deterministic": False,
                    "model_allowed": True,
                    "reason": "initial_global_architecture_attempt_settled",
                    "metadata": {
                        "director_mode": "initial_architecture",
                        "director_status": outcome.status,
                        "context_sha256": outcome.context_sha256,
                        "task_id": outcome.task_id,
                        "invoked": outcome.invoked,
                        "cache_hit": outcome.cache_hit,
                        "proposal_audit_count": len(audits),
                        "host_admitted_proposal_count": accepted_count,
                        "host_rejected_proposal_count": len(audits) - accepted_count,
                    },
                    "resolution": {
                        "status": outcome.status,
                        "artifact_sha256": outcome.artifact_sha256,
                        "reasons": list(outcome.reasons),
                    },
                },
            ),
            idempotency_key=idempotency_key,
        )

    def compile_global_context(
        self,
        *,
        material_events: Iterable[str] = (),
        evidence_observations: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    ) -> CampaignContext:
        graph = self.graph_store.load()
        portfolio = compile_proof_portfolio(
            graph,
            acceptance_spec=self.kernel.spec.acceptance,
        )
        return self.context_compiler.compile(
            kernel=self.kernel,
            hypergraph=graph,
            route_portfolio=portfolio,
            evidence_ledger=evidence_observations,
            material_events=material_events,
            previous=self._previous_context,
            # The audit view does not spend the director's separately
            # reserved prompt budget.
            enforce_limit=False,
        )

    def review_route_innovations(
        self,
        route_id: str,
        *,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return route_innovation_runtime.review_route_innovations(
            self.graph_store.load(),
            acceptance_spec=self.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
        )

    def review_route_program_innovations(
        self,
        route_id: str,
        *,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return program_innovation_runtime.review_route_program_innovations(
            self.graph_store.load(),
            acceptance_spec=self.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
        )

    def admit_route_experimental_claims(
        self,
        route_id: str,
        *,
        capabilities: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        mechanism_proposals: Iterable[Mapping[str, Any]] = (),
        validations: Iterable[Mapping[str, Any]] = (),
        enable_experimental_claim_admission: bool = False,
    ) -> dict[str, Any]:
        return admit_route_experimental_claims(
            self.kernel,
            self.graph_store,
            acceptance_spec=self.kernel.spec.acceptance,
            route_id=route_id,
            capabilities=capabilities,
            mechanism_proposals=mechanism_proposals,
            validations=validations,
            enable_experimental_claim_admission=(
                enable_experimental_claim_admission
            ),
        )



__all__: list[str] = []
