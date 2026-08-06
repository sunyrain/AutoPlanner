from __future__ import annotations

from cascade_planner.interfaces.literature_procedure_fragments import (
    source_procedure_fragments,
)


def test_entry_heading_splits_inline_journal_procedure() -> None:
    fragments = source_procedure_fragments(
        "Experimental",
        (
            "6,6-HexatnethyIenefulvene7a (Entry 1). To a solution of "
            "cycloheptanone and cyclopentadiene was added pyrrolidine. "
            "The reaction mixture was stirred and purified in 92% yield.\r\n"
            "6,6-Dimethylfulvene7b (Entry 2). To a solution of acetone and "
            "cyclopentadiene was added pyrrolidine. The reaction mixture was "
            "stirred and purified in 88% yield."
        ),
    )

    assert [(label, name) for _title, _body, label, name in fragments] == [
        ("1", "6,6-HexatnethyIenefulvene7a"),
        ("2", "6,6-Dimethylfulvene7b"),
    ]
    assert "cycloheptanone" in fragments[0][1]
    assert "acetone" not in fragments[0][1]


def test_numbered_patent_heading_splits_following_procedure() -> None:
    fragments = source_procedure_fragments(
        "Examples",
        (
            "(1) 6,6-hexamethylenefulvene\n"
            "Cycloheptanone and cyclopentadiene were added to methanol. "
            "The reaction mixture was stirred and isolated in 83% yield.\n"
            "(2) 6,6-dimethylfulvene\n"
            "Acetone and cyclopentadiene were added to methanol. The reaction "
            "mixture was stirred and isolated in 80% yield."
        ),
    )

    assert [(label, name) for _title, _body, label, name in fragments] == [
        ("1", "6,6-hexamethylenefulvene"),
        ("2", "6,6-dimethylfulvene"),
    ]
