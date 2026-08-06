"""Thin legacy-shaped entry adapter for the single-kernel V4 service."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def run_v4_controller_adapter(
    *,
    target_name: str,
    target_smiles: str,
    output_dir: str | Path,
    retrosynthesis_acceptance_spec: RetrosynthesisAcceptanceSpec | None = None,
    retrosynthesis_run_budget: RetrosynthesisRunBudget | None = None,
    global_plan: Mapping[str, Any] | None = None,
    auto_materialize: bool = False,
    publish_closeout: bool = False,
    **_legacy_options: Any,
) -> dict[str, Any]:
    """Start/resume V4 while accepting the old entrypoint's common keywords."""
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configured_root = str(os.environ.get("AUTOPLANNER_RUNTIME_ROOT") or "").strip()
    runtime_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else run_dir.parent / ".autoplanner-runtime"
    )
    kernel_spec_path = run_dir / ".autoplanner" / "kernel" / "run_spec.json"
    if kernel_spec_path.is_file():
        service = RetrosynthesisCampaignService.open(runtime_root, run_dir)
    else:
        identity = hashlib.sha256(
            f"{run_dir}\0{target_name}\0{target_smiles}".encode("utf-8")
        ).hexdigest()[:24]
        service = RetrosynthesisCampaignService.create(
            runtime_root,
            run_dir,
            spec=RunSpec(
                run_id=f"v4:{identity}",
                target_name=target_name,
                target_smiles=target_smiles,
                acceptance=(
                    retrosynthesis_acceptance_spec
                    or RetrosynthesisAcceptanceSpec()
                ),
                limits=RunLimits(
                    model=(
                        retrosynthesis_run_budget or RetrosynthesisRunBudget()
                    )
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
    if global_plan:
        service.apply_global_plan(global_plan, idempotency_key="adapter:global-plan")
    if auto_materialize:
        service.execute_frontier_materialization(
            idempotency_key="adapter:frontier-materialization"
        )
    closeout = (
        service.closeout(idempotency_key="adapter:closeout")
        if publish_closeout
        else {}
    )
    status = service.status()
    return {
        "schema_version": "v4_controller_adapter_result.v1",
        "engine": "v4",
        "run_dir": str(run_dir),
        "target_input": {
            "target_name": target_name,
            "target_smiles": target_smiles,
        },
        "status": status,
        "closeout": closeout,
        "semantics": {
            "thin_adapter": True,
            "run_kernel_is_operational_authority": True,
            "legacy_options_do_not_create_private_state": True,
        },
    }


__all__ = ["run_v4_controller_adapter"]
