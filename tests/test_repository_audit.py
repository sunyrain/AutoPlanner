from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.runtime.repository_audit import audit_repository


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_repository_audit_is_read_only_and_reports_review_candidates(
    tmp_path: Path,
) -> None:
    values = {
        "cascade_planner/dead.py": "import os\nVALUE = 1\n",
        "cascade_planner/used.py": "import os\nVALUE = os.name\n",
        "cascade_planner/reexported.py": "from os import name\n__all__ = ['name']\n",
        "docs/a.svg": "<svg/>",
        "docs/b.svg": "<svg/>",
        "results/run.log": "generated",
        ".github/workflows/ci.yml": "name: forbidden",
        "key.txt": "not-a-real-secret",
    }
    for relative, content in values.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    report = audit_repository(tmp_path, tracked_paths=values)

    assert report["status"] == "attention_required"
    assert report["checks"] == {
        "no_tracked_credentials": False,
        "no_github_actions": False,
        "no_generated_artifacts": False,
        "no_missing_tracked_files": True,
    }
    assert report["duplicate_assets"][0]["copy_count"] == 2
    assert any(
        row["path"] == "cascade_planner/dead.py" and row["binding"] == "os"
        for row in report["dead_import_candidates"]
    )
    assert not any(
        row["path"] in {
            "cascade_planner/used.py",
            "cascade_planner/reexported.py",
        }
        for row in report["dead_import_candidates"]
    )
    supplied = report.pop("content_sha256")
    assert supplied == _digest(report)
    assert all((tmp_path / relative).is_file() for relative in values)
