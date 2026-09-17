"""One accounting boundary for concurrent calls, missing usage and resumption.

Reservations are admission estimates, not provider-side token limits. Keep
reported usage separate from holds; an ambiguous timeout is not a free call.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import math
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.agent.codex_worker import WorkerRunRecord, worker_provider_failure_reason


TOKEN_FLOORS = {
    "builder": (24_000, 16_000),
    "critic": (24_000, 16_000),
    "editor": (20_000, 20_000),
    "strategy": (24_000, 16_000),
}


def saved_worker_records(
    run_dir: Path, *, model_input_sha256: str = "",
) -> list[WorkerRunRecord]:
    """Read durable calls, optionally bound to one exact portable model input.

    Legacy records without that binding still inform budget estimates but
    cannot supply a reusable judgment for a particular model input.
    """
    path = Path(run_dir) / ".autoplanner/director-workspace/sequential-director-worker-records.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            if model_input_sha256 and row.get("portable_model_input_sha256") != model_input_sha256:
                continue
            record = WorkerRunRecord(**row["record"])
        except (ValueError, TypeError, KeyError):
            continue
        records.append(record)
    return records


def stage_call_reservation(run_dir: Path, role: str, *, model: str) -> dict[str, int]:
    records = saved_worker_records(run_dir)
    return {axis: call_token_reserve(records, role, axis, model=model)
            for axis in ("input_tokens", "output_tokens")}


def call_role(task_type: str) -> str:
    if "critic" in task_type:
        return "critic"
    if "editor" in task_type:
        return "editor"
    if "strategy" in task_type:
        return "strategy"
    return "builder"


def reported_tokens(usage: Mapping[str, Any], axis: str) -> int | None:
    alias = "prompt_tokens" if axis == "input_tokens" else "completion_tokens"
    for key in (axis, alias):
        value = usage.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return int(value)
    return None


def call_token_reserve(
    records: Iterable[WorkerRunRecord],
    role: str,
    axis: str,
    *,
    model: str = "",
    floor: int | None = None,
) -> int:
    """Use a role/model high-water mark; cold roles borrow observed model cost."""
    minimum = TOKEN_FLOORS[role][axis == "output_tokens"] if floor is None else floor
    same_role, same_model = [], []
    for row in records:
        if worker_provider_failure_reason(row):
            continue
        metadata = row.metadata or {}
        if model and metadata.get("model") != model:
            continue
        value = reported_tokens(row.usage or {}, axis)
        if value is None:
            continue
        same_model.append(value)
        if call_role(str(metadata.get("task_type") or row.task_id.split(":")[0])) == role:
            same_role.append(value)
    return max(int(minimum), max(same_role or same_model, default=0))


def budget_exposure(records: Iterable[WorkerRunRecord]) -> dict[str, int]:
    rows = list(records)
    result = {
        "model_invocations": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "unknown_input_tokens_held": 0,
        "unknown_output_tokens_held": 0,
        "usage_incomplete_calls": 0,
    }
    for row in rows:
        failure = worker_provider_failure_reason(row)
        # Preserve the existing semantic invocation count. Reported provider
        # tokens still consume money even when no valid model turn completed.
        result["model_invocations"] += not bool(failure)
        metadata = row.metadata or {}
        reservation = metadata.get("budget_reservation") or {}
        role = call_role(str(metadata.get("task_type") or row.task_id.split(":")[0]))
        incomplete = False
        for axis in ("input_tokens", "output_tokens"):
            actual = reported_tokens(row.usage or {}, axis)
            if actual is not None:
                result[axis] += actual
                continue
            # Authentication/launch failures with no reported use are not
            # ambiguous dispatched requests. Other failures retain a hold.
            if failure in {"provider_auth_unavailable", "provider_cli_unavailable"}:
                continue
            incomplete = True
            hold = reservation.get(axis)
            if hold is None:
                hold = call_token_reserve(rows, role, axis, model=str(metadata.get("model") or ""))
            result[f"unknown_{axis}_held"] += max(0, int(hold))
        result["usage_incomplete_calls"] += incomplete
    result["input_tokens"] += result["unknown_input_tokens_held"]
    result["output_tokens"] += result["unknown_output_tokens_held"]
    return result


@dataclass(frozen=True, slots=True)
class NodeCallBudget:
    model_invocations: int
    input_tokens: int
    output_tokens: int
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class ModelCallReservation:
    input_tokens: int
    output_tokens: int


class SharedModelCallLedger:
    """Atomic admission; only mandatory final Critics have a protected pool."""

    def __init__(
        self,
        quota: NodeCallBudget,
        records: Iterable[WorkerRunRecord],
        *,
        protected_model_invocations: int = 0,
        protected_input_tokens: int = 0,
        protected_output_tokens: int = 0,
    ) -> None:
        self._quota = quota
        self._records = list(records)
        self._protected_calls = max(0, protected_model_invocations)
        self._protected_input = max(0, protected_input_tokens)
        self._protected_output = max(0, protected_output_tokens)
        self._inflight = {"model_invocations": 0, "input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()

    def _protected(self) -> dict[str, int]:
        slots = self._protected_calls
        return {
            "model_invocations": slots,
            "input_tokens": slots
            * call_token_reserve(
                self._records, "critic", "input_tokens", floor=self._protected_input // slots
            )
            if slots
            else self._protected_input,
            "output_tokens": slots
            * call_token_reserve(
                self._records, "critic", "output_tokens", floor=self._protected_output // slots
            )
            if slots
            else self._protected_output,
        }

    def reserve(
        self, *, input_tokens: int = 0, output_tokens: int = 0, task: Any = None
    ) -> tuple[ModelCallReservation | None, str]:
        with self._lock:
            requested = {
                "input_tokens": max(0, int(input_tokens)),
                "output_tokens": max(0, int(output_tokens)),
            }
            if task is not None:
                for axis in requested:
                    requested[axis] = call_token_reserve(
                        self._records,
                        call_role(task.task_type),
                        axis,
                        model=task.model,
                        floor=max(
                            requested[axis],
                            TOKEN_FLOORS[call_role(task.task_type)][axis == "output_tokens"],
                        ),
                    )
            exposure, protected = budget_exposure(self._records), self._protected()
            for axis, reason in (
                ("model_invocations", "model_invocation"),
                ("input_tokens", "input_token"),
                ("output_tokens", "output_token"),
            ):
                wanted = 1 if axis == "model_invocations" else requested[axis]
                if exposure[axis] + self._inflight[axis] + protected[axis] + wanted > getattr(
                    self._quota, axis
                ):
                    return None, f"{reason}_allocation_exhausted"
            self._inflight["model_invocations"] += 1
            for axis in requested:
                self._inflight[axis] += requested[axis]
            return ModelCallReservation(**requested), ""

    def settle(self, reservation: ModelCallReservation, record: WorkerRunRecord | None) -> None:
        with self._lock:
            self._inflight["model_invocations"] -= 1
            for axis in ("input_tokens", "output_tokens"):
                self._inflight[axis] -= getattr(reservation, axis)
            if record is not None:
                record.metadata.setdefault(
                    "budget_reservation",
                    {
                        "input_tokens": reservation.input_tokens,
                        "output_tokens": reservation.output_tokens,
                    },
                )
                self._records.append(record)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            completed = [r for r in self._records if not worker_provider_failure_reason(r)]
            return {
                "quota": {k: getattr(self._quota, k) for k in self._inflight},
                "committed": {
                    "model_invocations": len(completed),
                    **{
                        axis: sum(reported_tokens(r.usage or {}, axis) or 0 for r in completed)
                        for axis in ("input_tokens", "output_tokens")
                    },
                },
                "budget_exposure": budget_exposure(self._records),
                "inflight": dict(self._inflight),
                "protected_final_critics": self._protected(),
                "semantics": {
                    "shared_across_branches": True,
                    "settled_from_actual_usage": True,
                    "unknown_usage_held_separately": True,
                    "hypothetical_editor_rounds_reserved": False,
                },
            }
