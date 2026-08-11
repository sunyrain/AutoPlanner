from __future__ import annotations

import hashlib
from pathlib import Path

from cascade_planner.baselines.chem_enzy_runtime_probe import (
    _readability,
    _selected_model_path_checks,
)


def test_model_file_content_digest_is_full_and_stable(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint-bytes")

    first = _readability(model, content_digest=True)
    expected = hashlib.sha256(b"checkpoint-bytes").hexdigest()

    assert first["content_sha256"] == expected
    assert first["content_digest_scope"] == "full_file_bytes"
    assert first["content_digest_status"] == "complete"

    model.write_bytes(b"changed-checkpoint-bytes")
    second = _readability(model, content_digest=True)
    assert second["content_sha256"] != first["content_sha256"]


def test_model_directory_content_digest_binds_all_files(tmp_path: Path) -> None:
    model_root = tmp_path / "graph-data"
    model_root.mkdir()
    (model_root / "templates.csv").write_text("a,b\n", encoding="utf-8")
    (model_root / "index.pkl").write_bytes(b"index")

    first = _readability(model_root, content_digest=True)
    assert first["content_digest_scope"] == "directory_file_manifest"
    assert first["content_digest_status"] == "complete"
    assert first["content_file_count"] == 2

    (model_root / "templates.csv").write_text("a,c\n", encoding="utf-8")
    second = _readability(model_root, content_digest=True)
    assert second["content_sha256"] != first["content_sha256"]


def test_selected_model_path_checks_include_content_identity(tmp_path: Path) -> None:
    vendor_root = tmp_path / "vendor"
    model = vendor_root / "graph.ckpt"
    dataset = vendor_root / "data"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint")
    dataset.mkdir()
    (dataset / "templates.csv").write_text("template\n", encoding="utf-8")

    rows = _selected_model_path_checks(
        {
            "one_step_model_configs": {
                "graphfp_models": {
                    "fixture": {
                        "graph_model_dumb": str(model),
                        "graph_dataset_root": str(dataset),
                    }
                }
            }
        },
        ["graphfp_models.fixture"],
    )

    assert len(rows) == 2
    assert all(row["content_digest_status"] == "complete" for row in rows)
    assert all(row["content_sha256"] for row in rows)
