"""Gateway operation for provider-neutral strategic route imports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cascade_planner.application.external_strategy_routes import (
    compile_external_strategy_route_bundle,
)
from cascade_planner.application.proof_portfolio import compile_proof_portfolio
from cascade_planner.application.strategy_experiment_closure import (
    compile_strategy_to_experiment_closure,
)
from cascade_planner.interfaces.campaign_gateway_projection import (
    campaign_gateway_result,
)

if TYPE_CHECKING:
    from cascade_planner.interfaces.campaign_gateway import CampaignGateway


def import_external_strategy_routes(
    gateway: CampaignGateway,
    run_id: str,
    bundle: Mapping[str, Any],
    *,
    run_dir: str | Path | None = None,
    materialize: bool = True,
) -> dict[str, Any]:
    """Admit an external strategy without trusting provider self-assessment."""

    service = gateway._open(run_id, run_dir=run_dir)
    compiled = compile_external_strategy_route_bundle(
        bundle,
        expected_target_smiles=service.kernel.spec.target_smiles,
    )
    receipt = dict(compiled["receipt"])
    operations: dict[str, Any] = {
        "external_strategy_plan": service.apply_global_plan(
            compiled["global_plan"],
            idempotency_key=(
                f"gateway:external-strategy:{receipt['source_payload_sha256'][:24]}"
            ),
            proposal_origin_kind="external_strategy",
            proposal_origin_ref=str(compiled["origin_ref"]),
        )
    }
    if materialize:
        revision = service.kernel.state.graph_revision
        operations["materialization"] = service.execute_frontier_materialization(
            idempotency_key=f"gateway:external-strategy-materialization:{revision}"
        )
    graph = service.graph_store.load()
    portfolio = compile_proof_portfolio(
        graph,
        acceptance_spec=service.kernel.spec.acceptance,
    )
    closure = compile_strategy_to_experiment_closure(
        graph=graph,
        portfolio=portfolio,
        import_receipt=receipt,
    )
    return {
        **campaign_gateway_result(
            service,
            operation="import-external-strategy",
            operations=operations,
        ),
        "external_strategy_import": receipt,
        "strategy_to_experiment_closure": closure,
    }


__all__ = ["import_external_strategy_routes"]
