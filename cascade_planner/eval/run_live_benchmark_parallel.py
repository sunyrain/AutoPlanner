"""Parallel orchestrator for live_benchmark shards.

This keeps the benchmark runner itself single-target-loop simple while making
full benchmark runs reproducible: shard launch, GPU assignment, logs, cleanup,
and merge are handled in one place.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DONE_RE = re.compile(
    r"\[benchmark\] done (?P<ordinal>\d+)/(?P<total>\d+) "
    r"idx=(?P<index>\d+) domain=(?P<domain>\S+).*?"
    r"plan=(?P<plan>[01]).*?type@1=(?P<type1>[01]).*?"
    r"stock=(?P<stock>\S+).*?elapsed=(?P<elapsed>[0-9.]+)s"
)
START_RE = re.compile(
    r"\[benchmark\] start (?P<ordinal>\d+)/(?P<total>\d+) "
    r"idx=(?P<index>\d+) domain=(?P<domain>\S+)"
)


@dataclass
class LogProgress:
    starts: int = 0
    dones: int = 0
    total: int | None = None
    last: str = "none"
    slowest: tuple[float, int, str] | None = None
    has_traceback: bool = False


@dataclass
class DynamicQueueResult:
    claimed_indices: list[int]
    duplicate_indices: list[int]
    missing_indices: list[int]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "claimed_indices": list(self.claimed_indices),
            "duplicate_indices": list(self.duplicate_indices),
            "missing_indices": list(self.missing_indices),
        }


class DynamicBenchmarkQueue:
    """In-process dynamic target queue used by the parallel runner tests.

    The subprocess runner still writes shard artifacts, but target assignment is
    represented as a claim-once queue so worker implementations can avoid static
    long-tail shards.
    """

    def __init__(self, indices: Sequence[int]):
        self._pending = list(indices)
        self._claimed: list[int] = []

    def claim_next(self) -> int | None:
        if not self._pending:
            return None
        idx = self._pending.pop(0)
        self._claimed.append(idx)
        return idx

    def audit(self, expected_indices: Sequence[int]) -> DynamicQueueResult:
        counts: dict[int, int] = {}
        for idx in self._claimed:
            counts[idx] = counts.get(idx, 0) + 1
        expected = list(expected_indices)
        return DynamicQueueResult(
            claimed_indices=list(self._claimed),
            duplicate_indices=sorted(idx for idx, count in counts.items() if count > 1),
            missing_indices=sorted(idx for idx in expected if counts.get(idx, 0) == 0),
        )


def simulate_dynamic_worker_claims(indices: Sequence[int], workers: int) -> DynamicQueueResult:
    queue = DynamicBenchmarkQueue(indices)
    active = [True for _ in range(max(1, int(workers or 1)))]
    while any(active):
        for worker_idx in range(len(active)):
            if not active[worker_idx]:
                continue
            claimed = queue.claim_next()
            if claimed is None:
                active[worker_idx] = False
    return queue.audit(indices)


def parse_log_progress(path: Path) -> LogProgress:
    if not path.exists():
        return LogProgress()
    text = path.read_text(errors="replace")
    starts = list(START_RE.finditer(text))
    dones = list(DONE_RE.finditer(text))
    progress = LogProgress(
        starts=len(starts),
        dones=len(dones),
        has_traceback="Traceback" in text,
    )
    if starts:
        progress.total = int(starts[-1].group("total"))
        progress.last = f"running idx={starts[-1].group('index')}"
    if dones:
        last = dones[-1]
        progress.total = int(last.group("total"))
        progress.last = (
            f"done idx={last.group('index')} "
            f"{float(last.group('elapsed')):.1f}s "
            f"plan={last.group('plan')} type1={last.group('type1')} "
            f"stock={last.group('stock')}"
        )
        slow = max(
            (
                (float(m.group("elapsed")), int(m.group("index")), m.group("domain"))
                for m in dones
            ),
            default=None,
        )
        progress.slowest = slow
    return progress


def shard_output_path(output: Path, shard_index: int, num_shards: int) -> Path:
    suffix = output.suffix or ".json"
    stem = output.name[: -len(suffix)] if output.name.endswith(suffix) else output.name
    return output.with_name(f"{stem}_shard{shard_index}of{num_shards}{suffix}")


def shard_log_path(log_dir: Path, output: Path, shard_index: int, num_shards: int) -> Path:
    suffix = output.suffix or ".json"
    stem = output.name[: -len(suffix)] if output.name.endswith(suffix) else output.name
    return log_dir / f"{stem}_shard{shard_index}of{num_shards}.log"


def shard_trace_path(trace_output: Path, shard_index: int, num_shards: int) -> Path:
    suffix = "".join(trace_output.suffixes) or ".jsonl"
    name = trace_output.name
    stem = name[: -len(suffix)] if name.endswith(suffix) else trace_output.stem
    return trace_output.with_name(f"{stem}_shard{shard_index}of{num_shards}{suffix}")


def _repo_pythonpath() -> str:
    repo = str(Path.cwd())
    current = os.environ.get("PYTHONPATH")
    return f"{repo}{os.pathsep}{current}" if current else repo


def _base_env(args: argparse.Namespace, gpu: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _repo_pythonpath()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    env["MKL_NUM_THREADS"] = str(args.threads_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_worker)
    env["NUMEXPR_NUM_THREADS"] = str(args.threads_per_worker)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if args.route_tree_policy:
        env["AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER"] = "1"
        env["AUTOPLANNER_ROUTE_TREE_POLICY"] = args.route_tree_policy
    if getattr(args, "source_gate", ""):
        env["AUTOPLANNER_SOURCE_GATE"] = args.source_gate
    if getattr(args, "proposal_ranker_dir", ""):
        env["AUTOPLANNER_ENABLE_PROPOSAL_RANKERS"] = "1"
        env["AUTOPLANNER_PROPOSAL_RANKER_DIR"] = args.proposal_ranker_dir
    if getattr(args, "enable_v3_retrieval_proposals", False):
        env["AUTOPLANNER_ENABLE_V3_RETRIEVAL_PROPOSALS"] = "1"
    for item in args.extra_env or []:
        if "=" not in item:
            raise ValueError(f"--extra-env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def _benchmark_cmd(args: argparse.Namespace, out: Path, shard_index: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cascade_planner.cascadeboard.live_benchmark",
        "--bench",
        args.bench,
        "--output",
        str(out),
        "--model",
        args.model,
        "--search-mode",
        args.search_mode,
        "--num-shards",
        str(args.workers),
        "--shard-index",
        str(shard_index),
        "--n-results",
        str(args.n_results),
        "--n-candidates-per-skeleton",
        str(args.n_candidates_per_skeleton),
        "--device",
        args.device,
        "--target-log",
        args.target_log,
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.skeleton_samples is not None:
        cmd.extend(["--skeleton-samples", str(args.skeleton_samples)])
    if args.search_budget is not None:
        cmd.extend(["--search-budget", str(args.search_budget)])
    if args.check_stock:
        cmd.append("--check-stock")
    if args.prior_provider != "none":
        cmd.extend(["--prior-provider", args.prior_provider])
    if args.prior_weight != 1.0:
        cmd.extend(["--prior-weight", str(args.prior_weight)])
    if args.prior_cache:
        cmd.extend(["--prior-cache", args.prior_cache])
    if args.constraints_json:
        cmd.extend(["--constraints-json", args.constraints_json])
    if args.trace_output:
        cmd.extend(["--trace-output", str(shard_trace_path(Path(args.trace_output), shard_index, args.workers))])
    return cmd


def _merge_cmd(args: argparse.Namespace, shard_paths: Sequence[Path]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cascade_planner.cascadeboard.live_benchmark",
        "--merge",
        *[str(path) for path in shard_paths],
        "--output",
        args.output,
    ]
    if args.check_stock:
        cmd.append("--check-stock")
    return cmd


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _kill_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _print_progress(log_paths: Sequence[Path], *, final: bool = False) -> None:
    progresses = [parse_log_progress(path) for path in log_paths]
    dones = sum(p.dones for p in progresses)
    expected = sum(p.total or 0 for p in progresses) or "?"
    active = sum(1 for p in progresses if p.starts > p.dones)
    slowest = [p.slowest for p in progresses if p.slowest is not None]
    slowest_msg = ""
    if slowest:
        elapsed, index, domain = max(slowest)
        slowest_msg = f" slowest=idx{index}:{elapsed:.1f}s:{domain}"
    prefix = "[parallel-final]" if final else "[parallel]"
    print(f"{prefix} done={dones}/{expected} active={active}{slowest_msg}", flush=True)
    for path, progress in zip(log_paths, progresses):
        flag = " traceback" if progress.has_traceback else ""
        print(
            f"{prefix} {path.name}: starts={progress.starts} "
            f"dones={progress.dones} last={progress.last}{flag}",
            flush=True,
        )


def run_parallel(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else output.parent / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    original_bench = args.bench
    if args.limit is not None and args.workers > 1:
        limited_bench = _write_limited_benchmark(Path(args.bench), output=output, limit=args.limit)
        args.bench = str(limited_bench)
        args.limit = None

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.gpus else []
    if args.device.startswith("cuda") and not gpus:
        gpus = ["0"]

    shard_paths = [shard_output_path(output, i, args.workers) for i in range(args.workers)]
    log_paths = [shard_log_path(log_dir, output, i, args.workers) for i in range(args.workers)]
    trace_paths = (
        [shard_trace_path(Path(args.trace_output), i, args.workers) for i in range(args.workers)]
        if args.trace_output
        else []
    )
    for path in [*shard_paths, *log_paths, *trace_paths]:
        if path.exists():
            path.unlink()

    procs: list[subprocess.Popen] = []
    try:
        for i in range(args.workers):
            gpu = gpus[i % len(gpus)] if gpus else None
            env = _base_env(args, gpu)
            cmd = _benchmark_cmd(args, shard_paths[i], i)
            log_fh = log_paths[i].open("w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            log_fh.close()
            procs.append(proc)
            gpu_msg = f" gpu={gpu}" if gpu is not None else ""
            print(f"[parallel] started shard={i}/{args.workers} pid={proc.pid}{gpu_msg}", flush=True)

        last_report = 0.0
        while True:
            running = [proc for proc in procs if proc.poll() is None]
            now = time.time()
            if now - last_report >= args.poll_seconds:
                _print_progress(log_paths)
                last_report = now
            if not running:
                break
            time.sleep(min(args.poll_seconds, 5.0))
    except KeyboardInterrupt:
        print("[parallel] interrupted; terminating worker process groups", flush=True)
        for proc in procs:
            _terminate_process_group(proc)
        time.sleep(3.0)
        for proc in procs:
            _kill_process_group(proc)
        raise
    finally:
        for proc in procs:
            if proc.poll() is None:
                _terminate_process_group(proc)

    _print_progress(log_paths, final=True)
    failed = [(idx, proc.returncode) for idx, proc in enumerate(procs) if proc.returncode != 0]
    if failed:
        for idx, code in failed:
            print(f"[parallel] shard {idx} failed with exit code {code}; log={log_paths[idx]}", file=sys.stderr)
        return 1

    merge = subprocess.run(
        _merge_cmd(args, shard_paths),
        env={**os.environ, "PYTHONPATH": _repo_pythonpath()},
        text=True,
        capture_output=True,
        check=False,
    )
    if merge.stderr:
        print(merge.stderr, end="", file=sys.stderr)
    if merge.returncode != 0:
        if merge.stdout:
            print(merge.stdout, end="")
        print(f"[parallel] merge failed with exit code {merge.returncode}", file=sys.stderr)
        return merge.returncode
    if args.trace_output:
        trace_output = Path(args.trace_output)
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        with trace_output.open("w", encoding="utf-8") as out_fh:
            for path in trace_paths:
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8")
                if text and not text.endswith("\n"):
                    text += "\n"
                out_fh.write(text)

    try:
        summary = json.loads(output.read_text()).get("summary", {})
        if original_bench != args.bench:
            summary.setdefault("parallel_limited_benchmark_source", original_bench)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        pass
    return 0


def _write_limited_benchmark(path: Path, *, output: Path, limit: int) -> Path:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"parallel --limit expects list benchmark JSON: {path}")
    limit = max(0, int(limit))
    limited_path = output.with_name(f"{output.stem}_limit{limit}_benchmark.json")
    limited_path.write_text(json.dumps(rows[:limit], indent=2, ensure_ascii=False), encoding="utf-8")
    return limited_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run live_benchmark shards in parallel and merge outputs")
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids for CUDA_VISIBLE_DEVICES")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--n-candidates-per-skeleton", type=int, default=2)
    ap.add_argument("--skeleton-samples", type=int, default=None)
    ap.add_argument("--check-stock", action="store_true")
    ap.add_argument("--prior-provider", default="none", choices=["none", "deterministic", "deepseek"])
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--prior-cache", default=None)
    ap.add_argument(
        "--search-mode",
        default="route_tree",
        choices=["rerank", "stock_aware", "critic_control", "cc_aostar", "hybrid", "stock_rescue", "route_tree", "policy_retry"],
    )
    ap.add_argument("--search-budget", type=int, default=None)
    ap.add_argument("--constraints-json", default=None)
    ap.add_argument("--target-log", default="brief", choices=["none", "brief", "json"])
    ap.add_argument("--trace-output", default=None, help="Merged route_tree trace JSONL output")
    ap.add_argument("--route-tree-policy", default=os.environ.get("AUTOPLANNER_ROUTE_TREE_POLICY", ""))
    ap.add_argument("--source-gate", default=os.environ.get("AUTOPLANNER_SOURCE_GATE", ""))
    ap.add_argument("--proposal-ranker-dir", default=os.environ.get("AUTOPLANNER_PROPOSAL_RANKER_DIR", ""))
    ap.add_argument("--enable-v3-retrieval-proposals", action="store_true")
    ap.add_argument("--extra-env", action="append", default=[])
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_parallel(args))


if __name__ == "__main__":
    main()
