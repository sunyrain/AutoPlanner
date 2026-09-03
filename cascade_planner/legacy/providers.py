"""Provider adapters that invoke frozen V3 orchestration."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.providers.builtins import build_default_provider_registry
from cascade_planner.providers.contracts import (
    ProviderContext,
    ProviderDescriptor,
    ProviderKind,
    ProviderResultEnvelope,
    StockProvider,
)
from cascade_planner.providers.registry import ProviderRegistry


class CodexRetrosynthesisProvider:
    """Frozen backend around the recursive V3 Codex campaign."""

    descriptor = ProviderDescriptor(
        provider_id="autoplanner.codex_retrosynthesis",
        kind=ProviderKind.AGENT_BACKEND,
        version="1.0.0",
        input_schemas=("codex_retrosynthesis_campaign_request.v1",),
        output_schemas=("codex_retrosynthesis_team_run.v1",),
        correlation_group="codex_model",
        capabilities=(
            "direct_child_agent_spawn",
            "multi_role_retrosynthesis",
            "recursive_frontier_expansion",
        ),
        network_access=True,
        estimated_cost_units=1.0,
    )

    def __init__(
        self,
        *,
        run_dir: str | Path,
        repository_root: str | Path,
        config: Any = None,
        runner: Any = None,
        stock_provider: StockProvider | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.repository_root = Path(repository_root).resolve()
        self.config = config
        self.runner = runner
        self.stock_provider = stock_provider

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        context: ProviderContext,
    ) -> ProviderResultEnvelope:
        from cascade_planner.legacy.orchestration_runtime.codex_retrosynthesis import (
            RetrosynthesisTeamConfig,
            run_codex_retrosynthesis_campaign,
        )

        effective_config = self.config or RetrosynthesisTeamConfig()
        config_updates: dict[str, Any] = {}
        if isinstance(request.get("reaction_proofs"), Mapping):
            config_updates["reaction_proofs"] = {
                str(key): dict(value)
                for key, value in request["reaction_proofs"].items()
                if isinstance(value, Mapping)
            }
        if isinstance(request.get("reaction_proof_reports"), list):
            config_updates["reaction_proof_reports"] = [
                dict(row)
                for row in request["reaction_proof_reports"]
                if isinstance(row, Mapping)
            ]
        if config_updates:
            effective_config = replace(effective_config, **config_updates)

        report = run_codex_retrosynthesis_campaign(
            case_id=str(request.get("case_id") or context.case_id),
            target_name=str(request.get("target_name") or ""),
            target_smiles=str(request.get("target_smiles") or context.target_smiles),
            run_dir=self.run_dir,
            repository_root=self.repository_root,
            blackboard_context=dict(request.get("blackboard_context") or {}),
            literature_sources=[
                dict(row)
                for row in request.get("literature_sources") or []
                if isinstance(row, Mapping)
            ],
            config=effective_config,
            runner=self.runner,
            stock_provider=self.stock_provider,
        )
        return ProviderResultEnvelope(
            provider_id=self.descriptor.provider_id,
            provider_version=self.descriptor.version,
            provider_kind=self.descriptor.kind,
            correlation_group=self.descriptor.correlation_group,
            output_schema="codex_retrosynthesis_team_run.v1",
            accepted=report.get("accepted") is True,
            payload=report,
            reasons=tuple(str(item) for item in report.get("reasons") or []),
            source_refs=tuple(
                str(item)
                for item in (report.get("route_consensus") or {}).get("source_refs") or []
            ),
            evidence_refs=tuple(
                str(item)
                for item in (report.get("route_consensus") or {}).get("evidence_refs") or []
            ),
        )


def build_legacy_provider_registry(
    backend: CodexRetrosynthesisProvider,
) -> ProviderRegistry:
    registry = build_default_provider_registry()
    registry.register(
        backend,
        trusted_descriptor=backend.descriptor,
        authority="autoplanner_legacy_builtin_allowlist.v1",
    )
    return registry


__all__ = ["CodexRetrosynthesisProvider", "build_legacy_provider_registry"]
