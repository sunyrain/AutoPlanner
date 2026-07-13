from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_codex_entry_agentic_blackboard import (
    _literature_sources_from_args,
)


def test_literature_source_json_preserves_structured_focus_metadata() -> None:
    source = {
        "local_pdf": "source.pdf",
        "source_ref": "doi:10.1000/example",
        "expected_scheme_or_compound_labels": ["compound 6", "acid 10"],
        "route_sequence_hint": "compound 6 to acid 10",
    }

    rows = _literature_sources_from_args(
        argparse.Namespace(literature_source=[json.dumps(source)])
    )

    assert rows == [source]


def test_literature_source_path_shorthand_remains_supported() -> None:
    rows = _literature_sources_from_args(
        argparse.Namespace(
            literature_source=["source.pdf::doi:10.1000/example"]
        )
    )

    assert rows == [
        {"local_pdf": "source.pdf", "source_ref": "doi:10.1000/example"}
    ]


def test_literature_sources_file_preserves_lists_and_paths_with_spaces(
    tmp_path: Path,
) -> None:
    sources = [
        {
            "local_pdf": str(tmp_path / "source with spaces.pdf"),
            "source_ref": "doi:10.1000/example",
            "expected_scheme_or_compound_labels": ["C43", "PF-07321332"],
        }
    ]
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps({"literature_sources": sources}),
        encoding="utf-8",
    )

    rows = _literature_sources_from_args(
        argparse.Namespace(
            literature_source=[],
            literature_sources_file=[str(manifest)],
        )
    )

    assert rows == sources
