from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "batch_fetch_recent_total_synthesis_authorized.py"
)
SPEC = importlib.util.spec_from_file_location("authorized_batch_fetch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(MODULE_PATH.parent))
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_targeted_batch_update_preserves_other_p0_audit_rows() -> None:
    queue = [
        {"paper_id": "paper-a"},
        {"paper_id": "paper-b"},
        {"paper_id": "paper-c"},
    ]
    existing = [
        {"paper_id": "paper-a", "batch_index": 1, "status": "accepted"},
        {"paper_id": "paper-b", "batch_index": 2, "status": "fetch_failed"},
        {"paper_id": "paper-c", "batch_index": 3, "status": "fetch_failed"},
    ]
    updated = [{"paper_id": "paper-b", "batch_index": 1, "status": "accepted"}]

    merged = batch.merge_batch_rows(queue, existing, updated)

    assert [row["paper_id"] for row in merged] == ["paper-a", "paper-b", "paper-c"]
    assert [row["batch_index"] for row in merged] == [1, 2, 3]
    assert [row["status"] for row in merged] == [
        "accepted",
        "accepted",
        "fetch_failed",
    ]


def test_new_targeted_receipt_can_remain_a_single_row() -> None:
    queue = [{"paper_id": "paper-a"}, {"paper_id": "paper-b"}]
    updated = [{"paper_id": "paper-b", "batch_index": 1, "status": "accepted"}]

    assert batch.merge_batch_rows(queue, [], updated) == [
        {"paper_id": "paper-b", "batch_index": 2, "status": "accepted"}
    ]


def test_batch_runs_isolated_fetches_concurrently(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(
                {
                    "paper_id": f"paper-{index}",
                    "doi": f"10.1021/example.{index}",
                    "review_tier": "P0_source_extraction",
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    source_receipts = tmp_path / "source-receipts.jsonl"
    source_receipts.write_text("", encoding="utf-8")
    cache = tmp_path / "cache"
    output = tmp_path / "batch.jsonl"
    lock = threading.Lock()
    both_started = threading.Event()
    active = 0
    maximum_active = 0

    def fake_run(command, **_kwargs):
        nonlocal active, maximum_active
        paper_root = Path(command[command.index("--output-dir") + 1])
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                both_started.set()
        assert both_started.wait(timeout=2)
        paper_root.mkdir(parents=True, exist_ok=True)
        (paper_root / "authorized-literature-fetch.json").write_text(
            json.dumps({"accepted": True, "artifact_count": 1}),
            encoding="utf-8",
        )
        with lock:
            active -= 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--queue",
            str(queue),
            "--source-receipts",
            str(source_receipts),
            "--cache-dir",
            str(cache),
            "--batch-receipt",
            str(output),
            "--workers",
            "2",
        ],
    )

    assert batch.main() == 0
    assert maximum_active == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["accepted"] for row in rows)


def test_failed_force_refetch_preserves_previous_accepted_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "paper_id": "paper-a",
                "doi": "10.1021/example.1",
                "review_tier": "P0_source_extraction",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_receipts = tmp_path / "source-receipts.jsonl"
    source_receipts.write_text("", encoding="utf-8")
    cache = tmp_path / "cache"
    paper_root = cache / "paper-a"
    paper_root.mkdir(parents=True)
    canonical_receipt = paper_root / "authorized-literature-fetch.json"
    canonical_receipt.write_text(
        json.dumps({"accepted": True, "artifact_count": 2, "sentinel": "keep"}),
        encoding="utf-8",
    )
    output = tmp_path / "batch.jsonl"

    def fake_run(command, **_kwargs):
        attempt_root = Path(command[command.index("--output-dir") + 1])
        assert attempt_root != paper_root
        attempt_root.mkdir(parents=True, exist_ok=True)
        (attempt_root / "authorized-literature-fetch.json").write_text(
            json.dumps({"accepted": False, "artifact_count": 0}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=2, stdout="", stderr="challenge")

    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--queue",
            str(queue),
            "--source-receipts",
            str(source_receipts),
            "--cache-dir",
            str(cache),
            "--batch-receipt",
            str(output),
            "--force-refetch",
        ],
    )

    assert batch.main() == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["status"] == "refetch_failed_preserved"
    assert row["accepted"] is True
    assert row["latest_attempt_accepted"] is False
    assert json.loads(canonical_receipt.read_text(encoding="utf-8"))["sentinel"] == "keep"
