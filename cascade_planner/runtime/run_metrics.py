"""Low-overhead, non-authoritative metrics for one planner run.

Metrics explain runtime and resource behaviour; they never grant chemistry,
evidence, stock, or route-completion authority.  The recorder is deliberately
dependency-free so every CLI and recovery path can use it.
"""
from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, ParamSpec, TypeVar


RUN_METRICS_SCHEMA = "autoplanner_run_metrics.v1"
RUN_STAGE_METRIC_SCHEMA = "autoplanner_run_stage_metric.v1"
_DIGEST_KEY = "content_sha256"
_P = ParamSpec("_P")
_R = TypeVar("_R")
_DEFAULT_MAX_STAGE_ROWS = 2_000


class RunMetricsError(RuntimeError):
    """Raised when a metrics artifact cannot satisfy its own contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop(_DIGEST_KEY, None)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep metric labels compact and free of arbitrary model/source payloads."""
    safe: dict[str, Any] = {}
    for raw_key, value in (attributes or {}).items():
        key = str(raw_key)[:80]
        if isinstance(value, float) and not math.isfinite(value):
            safe[key] = None
        elif isinstance(value, bool | int | float) or value is None:
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = value[:240]
        elif isinstance(value, (list, tuple, set)):
            safe[key] = [str(item)[:120] for item in list(value)[:20]]
        else:
            safe[key] = str(value)[:240]
    return safe


@dataclass(frozen=True)
class _StageToken:
    sequence: int
    name: str
    category: str
    started_at: str
    started_wall_ns: int
    started_cpu_ns: int
    attributes: dict[str, Any]


class RunMetricsRecorder:
    """Thread-safe stage/counter recorder with atomic snapshots."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str = "",
        producer: str = "autoplanner",
        run_kind: str = "",
        max_stage_rows: int = _DEFAULT_MAX_STAGE_ROWS,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "run_metrics.json"
        self.run_id = str(run_id or run_kind)
        self.case_id = ""
        self.producer = str(producer)
        self.started_at = _utc_now()
        self._started_wall_ns = time.perf_counter_ns()
        self._started_cpu_ns = time.process_time_ns()
        self._last_checkpoint_wall_ns = self._started_wall_ns
        self._last_checkpoint_cpu_ns = self._started_cpu_ns
        self._sequence = 0
        self._max_stage_rows = max(1, int(max_stage_rows))
        self._stages: deque[dict[str, Any]] = deque(
            maxlen=self._max_stage_rows
        )
        self._dropped_stage_rows = 0
        self._stage_totals_by_category_ms: defaultdict[str, float] = (
            defaultdict(float)
        )
        self._stage_totals_by_name_ms: defaultdict[str, float] = defaultdict(
            float
        )
        self._stage_status_counts: defaultdict[str, int] = defaultdict(int)
        self._counters: defaultdict[str, float] = defaultdict(float)
        self._gauges: dict[str, float | int | str | bool] = {}
        self._status = "running"
        self._failure_type = ""
        self._lock = threading.RLock()

    def bind_case_id(self, case_id: str) -> None:
        with self._lock:
            self.case_id = str(case_id)

    def increment(self, name: str, value: int | float = 1) -> None:
        normalized = _finite_number(value)
        with self._lock:
            self._counters[str(name)] += normalized

    def gauge(self, name: str, value: float | int | str | bool) -> None:
        normalized: float | int | str | bool = value
        if isinstance(value, float) and not math.isfinite(value):
            normalized = "non_finite"
        with self._lock:
            self._gauges[str(name)] = normalized

    def _append_stage(self, row: dict[str, Any]) -> None:
        wall_ms = _nonnegative_finite_number(row.get("wall_ms"))
        row["wall_ms"] = round(wall_ms, 3)
        row["cpu_ms"] = round(
            _nonnegative_finite_number(row.get("cpu_ms")),
            3,
        )
        with self._lock:
            if len(self._stages) == self._max_stage_rows:
                self._dropped_stage_rows += 1
            self._stages.append(row)
            self._stage_totals_by_category_ms[
                str(row.get("category") or "other")
            ] += wall_ms
            self._stage_totals_by_name_ms[
                str(row.get("name") or "unknown")
            ] += wall_ms
            self._stage_status_counts[
                str(row.get("status") or "unknown")
            ] += 1

    def start_stage(
        self,
        name: str,
        *,
        category: str = "controller",
        attributes: Mapping[str, Any] | None = None,
    ) -> _StageToken:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return _StageToken(
            sequence=sequence,
            name=str(name),
            category=str(category),
            started_at=_utc_now(),
            started_wall_ns=time.perf_counter_ns(),
            started_cpu_ns=time.process_time_ns(),
            attributes=_safe_attributes(attributes),
        )

    def finish_stage(
        self,
        token: _StageToken,
        *,
        status: str = "completed",
        failure_type: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        finished_wall_ns = time.perf_counter_ns()
        finished_cpu_ns = time.process_time_ns()
        merged_attributes = dict(token.attributes)
        merged_attributes.update(_safe_attributes(attributes))
        row = {
            "schema_version": RUN_STAGE_METRIC_SCHEMA,
            "sequence": token.sequence,
            "name": token.name,
            "category": token.category,
            "status": str(status),
            "started_at": token.started_at,
            "finished_at": _utc_now(),
            "wall_ms": round(
                max(0, finished_wall_ns - token.started_wall_ns) / 1_000_000,
                3,
            ),
            "cpu_ms": round(
                max(0, finished_cpu_ns - token.started_cpu_ns) / 1_000_000,
                3,
            ),
            "failure_type": str(failure_type),
            "attributes": merged_attributes,
        }
        self._append_stage(row)
        return row

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str = "controller",
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        token = self.start_stage(
            name,
            category=category,
            attributes=attributes,
        )
        try:
            yield
        except BaseException as exc:
            self.finish_stage(
                token,
                status="failed",
                failure_type=type(exc).__name__,
            )
            raise
        else:
            self.finish_stage(token)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        """Compatibility spelling used by controller and tool instrumentation."""
        category = str(name).partition(".")[0] or "controller"
        with self.stage(name, category=category, attributes=metadata):
            yield

    def observe(
        self,
        name: str,
        *,
        elapsed_s: float,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an operation already timed by its owning subsystem."""
        elapsed_ms = round(_nonnegative_finite_number(elapsed_s) * 1000.0, 3)
        with self._lock:
            self._sequence += 1
            row = {
                "schema_version": RUN_STAGE_METRIC_SCHEMA,
                "sequence": self._sequence,
                "name": str(name),
                "category": str(name).partition(".")[0].partition(":")[0]
                or "operation",
                "status": str(status),
                "started_at": "",
                "finished_at": _utc_now(),
                "wall_ms": elapsed_ms,
                "cpu_ms": 0.0,
                "failure_type": "",
                "attributes": _safe_attributes(metadata),
            }
        self._append_stage(row)
        return row

    def checkpoint(
        self,
        name: str,
        *,
        category: str = "controller",
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the interval since the previous sequential checkpoint."""
        finished_wall_ns = time.perf_counter_ns()
        finished_cpu_ns = time.process_time_ns()
        with self._lock:
            self._sequence += 1
            row = {
                "schema_version": RUN_STAGE_METRIC_SCHEMA,
                "sequence": self._sequence,
                "name": str(name),
                "category": str(category),
                "status": "completed",
                "started_at": "",
                "finished_at": _utc_now(),
                "wall_ms": round(
                    max(0, finished_wall_ns - self._last_checkpoint_wall_ns)
                    / 1_000_000,
                    3,
                ),
                "cpu_ms": round(
                    max(0, finished_cpu_ns - self._last_checkpoint_cpu_ns)
                    / 1_000_000,
                    3,
                ),
                "failure_type": "",
                "attributes": _safe_attributes(attributes),
            }
            self._last_checkpoint_wall_ns = finished_wall_ns
            self._last_checkpoint_cpu_ns = finished_cpu_ns
        self._append_stage(row)
        return row

    def observe_result(self, result: Mapping[str, Any]) -> None:
        board = dict(result.get("agent_blackboard") or {})
        self.gauge("action_batch_count", len(result.get("action_batches") or []))
        self.gauge("tool_call_count", len(result.get("tool_calls") or []))
        graph = dict(
            board.get("canonical_route_consensus_graph")
            or board.get("route_consensus_graph")
            or {}
        )
        self.gauge("molecule_node_count", len(graph.get("molecule_nodes") or []))
        self.gauge("reaction_hyperedge_count", len(graph.get("reaction_hyperedges") or []))
        contract = dict(board.get("retrosynthesis_run_contract") or {})
        cost = dict(
            contract.get("cost_ledger")
            or board.get("retrosynthesis_cost_ledger")
            or {}
        )
        totals = dict(cost.get("totals") or {})
        cost_gauge_names = {
            "wall_time_s": "model_wall_time_s",
        }
        for key in (
            "model_invocations",
            "input_tokens",
            "output_tokens",
            "wall_time_s",
            "accepted_expansions",
            "attempt_runs",
        ):
            if key in totals:
                self.gauge(cost_gauge_names.get(key, key), totals[key])
        if "model_invocations" in result:
            self.gauge("model_invocations", result["model_invocations"])

    def finish(
        self,
        *,
        status: str,
        failure_type: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            self._status = str(status)
            self._failure_type = str(failure_type)
        return self.persist()

    def to_dict(self) -> dict[str, Any]:
        finished_wall_ns = time.perf_counter_ns()
        finished_cpu_ns = time.process_time_ns()
        with self._lock:
            stage_rows = sorted(
                (dict(row) for row in self._stages),
                key=lambda row: int(row.get("sequence") or 0),
            )
            payload: dict[str, Any] = {
                "schema_version": RUN_METRICS_SCHEMA,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "producer": self.producer,
                "status": self._status,
                "failure_type": self._failure_type,
                "started_at": self.started_at,
                "observed_at": _utc_now(),
                "total_wall_ms": round(
                    max(0, finished_wall_ns - self._started_wall_ns) / 1_000_000,
                    3,
                ),
                "total_cpu_ms": round(
                    max(0, finished_cpu_ns - self._started_cpu_ns) / 1_000_000,
                    3,
                ),
                "stages": stage_rows,
                "stage_row_count": int(sum(self._stage_status_counts.values())),
                "retained_stage_row_count": len(stage_rows),
                "dropped_stage_row_count": self._dropped_stage_rows,
                "max_retained_stage_rows": self._max_stage_rows,
                "stage_totals_by_category_ms": {
                    key: round(value, 3)
                    for key, value in sorted(
                        self._stage_totals_by_category_ms.items()
                    )
                },
                "stage_totals_by_name_ms": {
                    key: round(value, 3)
                    for key, value in sorted(
                        self._stage_totals_by_name_ms.items()
                    )
                },
                "stage_status_counts": dict(
                    sorted(self._stage_status_counts.items())
                ),
                "counters": {
                    key: value for key, value in sorted(self._counters.items())
                },
                "gauges": dict(sorted(self._gauges.items())),
                "runtime": {
                    "python": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "platform": platform.system(),
                    "process_id": os.getpid(),
                },
                "semantics": {
                    "metrics_are_observability_only": True,
                    "metrics_grant_no_chemistry_authority": True,
                    "wall_stage_totals_may_overlap": True,
                    "stage_rows_may_be_bounded": True,
                    "stage_aggregates_include_dropped_rows": True,
                },
            }
        payload[_DIGEST_KEY] = _digest(payload)
        return payload

    def persist(self) -> dict[str, Any]:
        payload = self.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            delete=False,
        )
        temp_path = Path(handle.name)
        try:
            with handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return payload


_CURRENT_RUN_METRICS: ContextVar[RunMetricsRecorder | None] = ContextVar(
    "autoplanner_current_run_metrics",
    default=None,
)


def current_run_metrics() -> RunMetricsRecorder | None:
    return _CURRENT_RUN_METRICS.get()


@contextmanager
def run_metric_stage(
    name: str,
    *,
    category: str = "controller",
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    recorder = current_run_metrics()
    if recorder is None:
        yield
        return
    with recorder.stage(name, category=category, attributes=attributes):
        yield


def run_metric_checkpoint(
    name: str,
    *,
    category: str = "controller",
    attributes: Mapping[str, Any] | None = None,
) -> None:
    recorder = current_run_metrics()
    if recorder is not None:
        recorder.checkpoint(name, category=category, attributes=attributes)


def measure_current_stage(
    name: str,
    *,
    category: str = "controller",
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with run_metric_stage(name, category=category):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def record_run_metrics(
    function: Callable[_P, dict[str, Any]],
) -> Callable[_P, dict[str, Any]]:
    """Decorate a run entry point and persist metrics on success or failure."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> dict[str, Any]:
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            return function(*args, **kwargs)
        recorder = RunMetricsRecorder(
            output_dir,
            run_id=str(kwargs.get("target_name") or function.__name__),
            producer=f"{function.__module__}.{function.__name__}",
        )
        context_token = _CURRENT_RUN_METRICS.set(recorder)
        try:
            with recorder.stage("run.total", category="run"):
                result = function(*args, **kwargs)
            preflight = dict(result.get("preflight") or {})
            recorder.bind_case_id(
                str(result.get("case_id") or preflight.get("case_id") or "")
            )
            recorder.observe_result(result)
            payload = recorder.finish(status="completed")
            result["run_metrics"] = payload
            artifacts = result.setdefault("artifacts", {})
            if isinstance(artifacts, dict):
                artifacts["run_metrics"] = str(recorder.path)
            return result
        except BaseException as exc:
            recorder.finish(status="failed", failure_type=type(exc).__name__)
            raise
        finally:
            _CURRENT_RUN_METRICS.reset(context_token)

    return wrapped


def validate_run_metrics(payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("schema_version") != RUN_METRICS_SCHEMA:
        reasons.append("run_metrics_schema_invalid")
    if payload.get("status") not in {"running", "completed", "failed"}:
        reasons.append("run_metrics_status_invalid")
    rows = payload.get("stages")
    if not isinstance(rows, list):
        reasons.append("run_metrics_stages_invalid")
    elif any(not _valid_stage_row(row) for row in rows):
        reasons.append("run_metrics_stage_invalid")
    if not _valid_nonnegative_metric(payload.get("total_wall_ms")):
        reasons.append("run_metrics_total_wall_invalid")
    if not _valid_nonnegative_metric(payload.get("total_cpu_ms")):
        reasons.append("run_metrics_total_cpu_invalid")
    counters = payload.get("counters")
    if not isinstance(counters, dict) or any(
        not _valid_finite_metric(value) for value in counters.values()
    ):
        reasons.append("run_metrics_counters_invalid")
    retained_count = payload.get("retained_stage_row_count")
    total_count = payload.get("stage_row_count")
    dropped_count = payload.get("dropped_stage_row_count")
    if not all(
        isinstance(value, int) and value >= 0
        for value in (retained_count, total_count, dropped_count)
    ) or (
        isinstance(retained_count, int)
        and isinstance(total_count, int)
        and isinstance(dropped_count, int)
        and retained_count + dropped_count != total_count
    ):
        reasons.append("run_metrics_stage_counts_invalid")
    try:
        digest_valid = str(payload.get(_DIGEST_KEY) or "") == _digest(payload)
    except (TypeError, ValueError):
        digest_valid = False
    if not digest_valid:
        reasons.append("run_metrics_content_digest_invalid")
    semantics = dict(payload.get("semantics") or {})
    if semantics.get("metrics_grant_no_chemistry_authority") is not True:
        reasons.append("run_metrics_non_authority_semantics_missing")
    return reasons


def _finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _nonnegative_finite_number(value: Any) -> float:
    return max(0.0, _finite_number(value))


def _valid_finite_metric(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))


def _valid_nonnegative_metric(value: Any) -> bool:
    return _valid_finite_metric(value) and float(value) >= 0.0


def _valid_stage_row(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        value.get("schema_version") == RUN_STAGE_METRIC_SCHEMA
        and str(value.get("name") or "").strip()
        and str(value.get("category") or "").strip()
        and str(value.get("status") or "").strip()
        and _valid_nonnegative_metric(value.get("wall_ms"))
        and _valid_nonnegative_metric(value.get("cpu_ms"))
        and isinstance(value.get("attributes"), dict)
    )


__all__ = [
    "RUN_METRICS_SCHEMA",
    "RUN_STAGE_METRIC_SCHEMA",
    "RunMetricsError",
    "RunMetricsRecorder",
    "current_run_metrics",
    "measure_current_stage",
    "record_run_metrics",
    "run_metric_checkpoint",
    "run_metric_stage",
    "validate_run_metrics",
]
