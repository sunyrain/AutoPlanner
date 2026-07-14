"""Campaign lifecycle adapter for replay-gated patent template memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.reaction_template_library import (
    retrieve_patent_template_candidates,
    synchronize_patent_template_library,
)
from cascade_planner.application.retrosynthesis_workers import (
    materialization_commands_for_proposals,
)
from cascade_planner.application.canonical_hypergraph import CanonicalIngestionBatch
from cascade_planner.application.reaction_template_store import (
    DEFAULT_TEMPLATE_LIBRARY_NAME,
)


@dataclass(slots=True)
class PatentSelfEvolutionSession:
    enabled: bool
    library_path: Path
    target_smiles: str
    max_candidates: int
    initial_retrieval: dict[str, Any] = field(default_factory=dict)
    reuse_stages: list[dict[str, Any]] = field(default_factory=list)
    learning_stages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        configured_path: str,
        external_data_root: Path,
        target_smiles: str,
        max_candidates: int,
    ) -> "PatentSelfEvolutionSession":
        path = (
            Path(configured_path).expanduser().resolve()
            if configured_path
            else external_data_root / "self-evo" / DEFAULT_TEMPLATE_LIBRARY_NAME
        )
        return cls(
            enabled=enabled,
            library_path=path,
            target_smiles=target_smiles,
            max_candidates=max_candidates,
        )

    def start(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        self.initial_retrieval = self._retrieve(graph)
        return self.initial_retrieval

    def materialize(self, service: Any) -> dict[str, Any]:
        result = (
            self._materialize(service)
            if self.enabled
            else self._disabled("patent_template_reuse")
        )
        self.reuse_stages.append(result)
        return result

    def _materialize(self, service: Any) -> dict[str, Any]:
        graph = service.graph_store.load()
        retrieval = self._retrieve(graph)
        proposals = [
            dict(value)
            for value in retrieval.get("proposals") or []
            if isinstance(value, Mapping) and str(value.get("route_family_id") or "")
        ]
        existing_proposals = [
            value for value in proposals if value.get("existing_edge_match") is True
        ]
        new_proposals = [
            value for value in proposals if value.get("existing_edge_match") is not True
        ]
        origin_binding = (
            service.apply_batch(
                CanonicalIngestionBatch(hypotheses=tuple(existing_proposals)),
                idempotency_key=(
                    "solve-target:self-evo-origin-binding:"
                    f"{service.kernel.state.graph_revision}:"
                    f"{retrieval.get('library_sha256', '')}"
                ),
            )
            if existing_proposals
            else {"changed": False, "rejected": []}
        )
        graph = service.graph_store.load()
        commands = materialization_commands_for_proposals(
            new_proposals,
            run_id=service.kernel.spec.run_id,
            input_revision=service.kernel.state.graph_revision,
            dependency_revisions={
                "graph_revision": service.kernel.state.graph_revision,
                "evidence_revision": service.kernel.state.evidence_revision,
            },
            existing_edge_digests=(
                str(edge.get("edge_digest") or "")
                for edge in graph["edges"].values()
            ),
        )
        execution = (
            service.execute_commands(
                commands,
                idempotency_key=(
                    "solve-target:self-evo-materialization:"
                    f"{service.kernel.state.graph_revision}:"
                    f"{retrieval.get('library_sha256', '')}"
                ),
                include_scheduled=False,
            )
            if commands
            else {"changed": False, "executed_command_count": 0, "material_events": []}
        )
        changed = bool(
            execution.get("changed") is True or origin_binding.get("changed") is True
        )
        return {
            "stage": "patent_template_reuse",
            "status": (
                "completed"
                if changed
                else str(retrieval.get("status") or "reused_or_empty")
            ),
            "candidate_count": int(retrieval.get("candidate_count") or 0),
            "routed_candidate_count": len(
                {str(value.get("proposal_id") or "") for value in proposals}
            ),
            "materialization_command_count": len(commands),
            "existing_origin_binding_count": len(existing_proposals),
            "origin_binding": origin_binding,
            "retrieval": retrieval,
            "execution": execution,
            "material_events": (
                ["patent_template_candidates_added"]
                if changed
                else []
            ),
            "model_invocations": 0,
            "semantics": {
                "template_reuse_enters_as_l0_proposal": True,
                "normal_mapping_and_reaction_validation_required": True,
                "template_reuse_does_not_call_codex": True,
            },
        }

    def learn(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        result = (
            synchronize_patent_template_library(self.library_path, graph)
            if self.enabled
            else self._disabled("patent_template_learning")
        )
        self.learning_stages.append(result)
        return result

    def observation(self, retrieval: Mapping[str, Any] | None = None) -> dict[str, Any]:
        source = dict(retrieval or self.initial_retrieval)
        proposals = [
            {
                "proposal_id": str(value.get("proposal_id") or ""),
                "template_id": str(value.get("origin_ref") or ""),
                "product_smiles": str(value.get("product_smiles") or ""),
                "precursor_smiles": list(value.get("precursor_smiles") or []),
                "support": dict(value.get("template_support") or {}),
            }
            for value in source.get("proposals") or []
            if isinstance(value, Mapping)
        ]
        unique = {
            str(value["proposal_id"]): value for value in proposals if value["proposal_id"]
        }
        if not unique:
            return {}
        return {
            "self_evo_patent_template_memory": {
                "schema_version": "campaign_patent_template_memory.v1",
                "library_sha256": str(source.get("library_sha256") or ""),
                "generation": int(source.get("generation") or 0),
                "candidates": [unique[key] for key in sorted(unique)],
                "semantics": {
                    "global_route_option_only": True,
                    "not_evidence_or_reaction_validation": True,
                    "host_will_replay_every_selected_candidate": True,
                },
            }
        }

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "library_path": str(self.library_path),
            "initial_retrieval": self.initial_retrieval,
            "reuse_stages": self.reuse_stages,
            "learning_stages": self.learning_stages,
            "model_invocations": 0,
            "semantics": {
                "codex_receives_global_template_options": True,
                "patent_learning_is_replay_gated": True,
                "reuse_reenters_normal_validation": True,
            },
        }

    def _retrieve(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return self._disabled("patent_template_retrieval")
        return retrieve_patent_template_candidates(
            self.library_path,
            graph=graph,
            target_smiles=self.target_smiles,
            max_candidates=self.max_candidates,
        )

    def _disabled(self, stage: str) -> dict[str, Any]:
        return {
            "schema_version": "patent_self_evolution_disabled.v1",
            "stage": stage,
            "status": "disabled",
            "library_path": str(self.library_path),
            "candidate_count": 0,
            "proposals": [],
            "model_invocations": 0,
        }


__all__ = ["PatentSelfEvolutionSession"]
