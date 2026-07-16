"""Auditable inventory and telemetry for frozen pre-V4 compatibility paths."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COMPATIBILITY_INVENTORY_SCHEMA = "retrosynthesis_compatibility_inventory.v1"
COMPATIBILITY_USAGE_SCHEMA = "retrosynthesis_compatibility_usage.v1"


@dataclass(frozen=True, slots=True)
class CompatibilityShim:
    shim_id: str
    module: str
    replacement: str
    removal_milestone: str
    telemetry_source: str
    owner: str = "retrosynthesis-v4"
    status: str = "frozen_compatibility"
    scientific_write_authority: bool = False

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.shim_id,
                self.module,
                self.replacement,
                self.removal_milestone,
                self.telemetry_source,
            )
        ):
            raise ValueError("compatibility_shim_metadata_incomplete")
        if self.status != "frozen_compatibility":
            raise ValueError("compatibility_shim_status_invalid")
        if self.scientific_write_authority:
            raise ValueError("compatibility_shim_cannot_claim_v4_write_authority")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SHIMS = (
    CompatibilityShim(
        shim_id="legacy.agentic_blackboard_controller",
        module="cascade_planner.harness.agentic_blackboard_controller",
        replacement="cascade_planner.orchestration.retrosynthesis_service",
        removal_milestone="P10 after saved-run and golden replay migration",
        telemetry_source=".autoplanner/compatibility_usage.jsonl",
    ),
    CompatibilityShim(
        shim_id="legacy.codex_retrosynthesis_campaign",
        module="cascade_planner.orchestration.codex_retrosynthesis",
        replacement="cascade_planner.orchestration.global_campaign_director",
        removal_milestone="P10 after golden replays migrate to canonical plans",
        telemetry_source=".autoplanner/compatibility_usage.jsonl",
    ),
    CompatibilityShim(
        shim_id="legacy.frontier_queue",
        module="cascade_planner.application.frontier_scheduler",
        replacement="cascade_planner.application.deficit_frontier",
        removal_milestone="P10 after legacy saved-run replay migration",
        telemetry_source="parent legacy campaign compatibility event",
    ),
    CompatibilityShim(
        shim_id="legacy.route_deficit_queue",
        module="cascade_planner.application.route_deficit_queue",
        replacement="cascade_planner.application.deficit_frontier",
        removal_milestone="P10 after legacy report replay migration",
        telemetry_source="parent blackboard compatibility event",
    ),
    CompatibilityShim(
        shim_id="legacy.route_portfolio",
        module="cascade_planner.application.route_portfolio",
        replacement="cascade_planner.application.proof_portfolio",
        removal_milestone="P10 after legacy route replay migration",
        telemetry_source="parent blackboard compatibility event",
    ),
    CompatibilityShim(
        shim_id="legacy.retrosynthesis_acceptance",
        module="cascade_planner.application.retrosynthesis_acceptance",
        replacement="cascade_planner.application.proof_policy",
        removal_milestone="P10 after legacy closeout replay migration",
        telemetry_source="parent blackboard compatibility event",
    ),
    CompatibilityShim(
        shim_id="legacy.route_forest",
        module="cascade_planner.harness.route_forest",
        replacement="cascade_planner.application.route_workbench",
        removal_milestone="P10 after legacy workbench golden migration",
        telemetry_source=".autoplanner/compatibility_usage.jsonl",
    ),
    CompatibilityShim(
        shim_id="legacy.local_tool_harness",
        module="cascade_planner.harness.tools",
        replacement="cascade_planner.application.worker_runtime",
        removal_milestone="P10 after remaining saved-run actions become workers",
        telemetry_source="tool_calls.jsonl plus compatibility_usage.jsonl",
    ),
    CompatibilityShim(
        shim_id="legacy.action_planners",
        module="cascade_planner.harness.agent_action_planner",
        replacement="cascade_planner.orchestration.global_campaign_director",
        removal_milestone="P10 after blackboard entrypoint retirement",
        telemetry_source="parent blackboard compatibility event",
    ),
    CompatibilityShim(
        shim_id="legacy.combined_web_surface",
        module="cascade_planner.web.app",
        replacement="cascade_planner.web.v4_app",
        removal_milestone="P10 after legacy UI and saved-run routes are retired",
        telemetry_source="CLI --surface combined selection",
    ),
)


def compatibility_inventory() -> dict[str, Any]:
    rows = [value.to_dict() for value in sorted(_SHIMS, key=lambda row: row.shim_id)]
    payload = {
        "schema_version": COMPATIBILITY_INVENTORY_SCHEMA,
        "shims": rows,
        "semantics": {
            "compatibility_is_not_v4_authority": True,
            "every_shim_has_telemetry_and_removal_milestone": True,
            "new_features_must_not_enter_frozen_shims": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def record_compatibility_use(
    run_dir: str | Path,
    shim_id: str,
    *,
    callsite: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    known = {value.shim_id for value in _SHIMS}
    if shim_id not in known:
        raise ValueError(f"compatibility_shim_not_registered:{shim_id}")
    root = Path(run_dir).expanduser().resolve() / ".autoplanner"
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": COMPATIBILITY_USAGE_SCHEMA,
        "shim_id": shim_id,
        "callsite": str(callsite or "unknown"),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": _json_value(dict(metadata or {})),
        "scientific_authority": False,
    }
    row["content_sha256"] = _digest(row)
    with (root / "compatibility_usage.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


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


__all__ = [
    "COMPATIBILITY_INVENTORY_SCHEMA",
    "COMPATIBILITY_USAGE_SCHEMA",
    "CompatibilityShim",
    "compatibility_inventory",
    "record_compatibility_use",
]
