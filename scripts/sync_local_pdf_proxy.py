#!/usr/bin/env python3
"""Sync a local PDF proxy work dir with a remote AutoPlanner server.

Run this on the local machine that has authorized school/library access. It
pulls queued PDF requests from the server, runs the local downloader, then pushes
the returned PDFs and manifest back to the server.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="SSH target, for example user@host.")
    parser.add_argument(
        "--remote-output-dir",
        required=True,
        help="Remote run/output dir, for example /root/autodl-tmp/AutoPlanner/results/shared/RUN.",
    )
    parser.add_argument(
        "--local-output-dir",
        default="",
        help="Local run/output dir. Defaults to the same path relative to the local repo when possible.",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--delay-s", type=float, default=1.0)
    parser.add_argument("--proxy", default="", help="Optional HTTP(S) proxy URL for the local fetch command.")
    parser.add_argument("--interval-s", type=float, default=0.0, help="Repeat every N seconds. 0 means run once.")
    parser.add_argument("--rsync", default="rsync", help="rsync executable.")
    parser.add_argument("--ssh-option", action="append", default=[], help="Extra ssh option passed through rsync -e.")
    args = parser.parse_args(argv)

    while True:
        run_once(args)
        if float(args.interval_s) <= 0:
            return 0
        time.sleep(max(1.0, float(args.interval_s)))


def run_once(args: argparse.Namespace) -> None:
    remote_output = str(args.remote_output_dir).rstrip("/")
    remote_work = f"{remote_output}/evidence/local_pdf_proxy/"
    local_output = Path(args.local_output_dir) if args.local_output_dir else _default_local_output(remote_output)
    local_work = local_output / "evidence" / "local_pdf_proxy"
    local_work.mkdir(parents=True, exist_ok=True)

    pull_cmd = [
        args.rsync,
        "-az",
        "--mkpath",
        "-e",
        _ssh_command(args.ssh_option),
        f"{args.server}:{remote_work}",
        f"{local_work}/",
    ]
    _run(pull_cmd)

    fetch_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "local_pdf_proxy.py"),
        "--output-dir",
        str(local_output),
        "fetch",
        "--timeout-s",
        str(float(args.timeout_s)),
        "--delay-s",
        str(float(args.delay_s)),
    ]
    if args.max_items is not None:
        fetch_cmd.extend(["--max-items", str(int(args.max_items))])
    if args.proxy:
        fetch_cmd.extend(["--proxy", str(args.proxy)])
    _run(fetch_cmd, cwd=ROOT)

    push_cmd = [
        args.rsync,
        "-az",
        "--mkpath",
        "-e",
        _ssh_command(args.ssh_option),
        f"{local_work}/",
        f"{args.server}:{remote_work}",
    ]
    _run(push_cmd)


def _default_local_output(remote_output: str) -> Path:
    marker = "/results/shared/"
    if marker in remote_output:
        suffix = remote_output.split(marker, 1)[1].strip("/")
        return ROOT / "results" / "shared" / suffix
    return ROOT / "results" / "shared" / Path(remote_output).name


def _ssh_command(options: list[str]) -> str:
    parts = ["ssh"]
    for option in options:
        parts.extend(["-o", option])
    return " ".join(shlex.quote(part) for part in parts)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
