import json

from scripts.mark_bufotalin_run_stopped import mark_run_stopped


def test_mark_run_stopped_updates_manifest_and_appends_event_once(tmp_path):
    root = tmp_path
    (root / "manifest.json").write_text(
        json.dumps({"target": "CCO", "running": True, "completed_cycles": 3}),
        encoding="utf-8",
    )
    (root / "runner_events.jsonl").write_text(
        json.dumps({"event": "start", "target": "CCO"}) + "\n",
        encoding="utf-8",
    )

    first = mark_run_stopped(root, reason="user_cancelled")
    second = mark_run_stopped(root, reason="user_cancelled")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (root / "runner_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    user_stop_events = [event for event in events if event.get("event") == "user_stop"]

    assert first["event_written"] is True
    assert second["already_marked"] is True
    assert second["event_written"] is False
    assert manifest["running"] is False
    assert manifest["stopped"] is True
    assert manifest["stop_reason"] == "user_cancelled"
    assert manifest["completed_cycles"] == 3
    assert len(user_stop_events) == 1
