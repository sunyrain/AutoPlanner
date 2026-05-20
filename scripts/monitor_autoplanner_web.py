"""Small terminal monitor for the local AutoPlanner Web service."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, UTC
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor AutoPlanner WebUI health and queued jobs.")
    ap.add_argument("--url", default="http://127.0.0.1:7991", help="Base URL of the Web service.")
    ap.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")
    ap.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    ap.add_argument("--jobs", type=int, default=5, help="Number of recent jobs to show.")
    args = ap.parse_args()

    while True:
        print(snapshot(args.url.rstrip("/"), n_jobs=max(0, args.jobs)), flush=True)
        if args.once:
            break
        time.sleep(max(0.5, args.interval))


def snapshot(base_url: str, *, n_jobs: int = 5) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        status = _fetch_json(f"{base_url}/api/status")
        jobs_payload = _fetch_json(f"{base_url}/api/jobs")
    except Exception as exc:
        return f"[{now}] DOWN {base_url} {type(exc).__name__}: {exc}"

    jobs = list(jobs_payload.get("jobs") or [])
    counts: dict[str, int] = {}
    for job in jobs:
        counts[str(job.get("status") or "unknown")] = counts.get(str(job.get("status") or "unknown"), 0) + 1
    cuda = status.get("cuda") or {}
    gpu_bits = []
    for gpu in cuda.get("devices") or []:
        gpu_bits.append(
            f"gpu{gpu.get('index')}={gpu.get('memory_used_mb', '-')}/{gpu.get('memory_total_mb', '-')}MB"
        )
    lines = [
        f"[{now}] OK {base_url}",
        f"  jobs={len(jobs)} counts={json.dumps(counts, sort_keys=True)}",
        f"  cuda_available={cuda.get('available')} {' '.join(gpu_bits)}",
    ]
    for job in jobs[:n_jobs]:
        summary = job.get("summary") or {}
        lines.append(
            "  "
            + " ".join(
                [
                    str(job.get("job_id") or "-"),
                    f"status={job.get('status') or '-'}",
                    f"routes={summary.get('routes', '-')}",
                    f"elapsed={job.get('elapsed_s', '-')}",
                    f"output={job.get('output_json') or '-'}",
                    f"rejected={job.get('rejected_output_json') or '-'}",
                ]
            )
        )
    return "\n".join(lines)


def _fetch_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc


if __name__ == "__main__":
    main()

