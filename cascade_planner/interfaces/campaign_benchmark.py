"""Model-free timing and memory benchmark for one campaign service."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from typing import Any

from cascade_planner.interfaces.campaign_gateway_contract import (
    CAMPAIGN_GATEWAY_RESULT_SCHEMA,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def benchmark_campaign(
    service: RetrosynthesisCampaignService,
    *,
    iterations: int,
) -> dict[str, Any]:
    count = max(1, min(25, int(iterations)))
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(count):
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            service.status()
            service.graph_store.full_recompute_oracle()
            service.workbench()
            cpu_samples.append(time.process_time() - cpu_start)
            wall_samples.append(time.perf_counter() - wall_start)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "schema_version": CAMPAIGN_GATEWAY_RESULT_SCHEMA,
        "operation": "benchmark",
        "run_id": service.kernel.spec.run_id,
        "iterations": count,
        "wall_time_s": _sample_summary(wall_samples),
        "cpu_time_s": _sample_summary(cpu_samples),
        "python_peak_bytes": peak,
        "model_invocations": 0,
        "semantics": {
            "model_free": True,
            "network_free": True,
            "measures_status_oracle_and_projection": True,
        },
    }


def _sample_summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "maximum": round(max(values), 6),
    }


__all__ = ["benchmark_campaign"]
