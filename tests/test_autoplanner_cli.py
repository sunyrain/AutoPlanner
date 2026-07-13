from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.cli import main


def _storage_args(tmp_path: Path) -> list[str]:
    return [
        "--repository-root",
        str(tmp_path),
        "--runtime-root",
        str(tmp_path / "runtime"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--artifact-store-root",
        str(tmp_path / "cas"),
        "--run-index-path",
        str(tmp_path / "index.sqlite3"),
    ]


def test_cli_run_status_validate_replay_benchmark_export_and_gc(
    tmp_path: Path,
    capsys,
) -> None:
    base = _storage_args(tmp_path)
    assert main(
        [
            *base,
            "run",
            "--run-id",
            "cli-example",
            "--target-name",
            "ethanol",
            "--target-smiles",
            "CCO",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["status"]["model_totals"]["model_invocations"] == 0

    for command in ("status", "validate", "replay"):
        assert main([*base, command, "cli-example"]) == 0
        value = json.loads(capsys.readouterr().out)
        assert value["run_id"] == "cli-example"

    assert main([*base, "benchmark", "cli-example", "--iterations", "1"]) == 0
    benchmark = json.loads(capsys.readouterr().out)
    assert benchmark["model_invocations"] == 0

    destination = tmp_path / "offline"
    assert main(
        [*base, "export", "cli-example", "--output-dir", str(destination)]
    ) == 0
    exported = json.loads(capsys.readouterr().out)
    assert Path(exported["files"]["html"]).is_file()

    assert main([*base, "gc", "--dry-run", "--minimum-age-hours", "0"]) == 0
    gc = json.loads(capsys.readouterr().out)
    assert gc["dry_run"] is True


def test_cli_gc_refuses_an_implicit_destructive_mode(
    tmp_path: Path,
    capsys,
) -> None:
    assert main([*_storage_args(tmp_path), "gc"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["reason"] == "gc_requires_explicit_--dry-run"
