from __future__ import annotations

from cascade_planner.harness.source_condition_extraction import (
    extract_source_conditions,
)


def test_exact_procedure_conditions_are_source_derived_and_complete() -> None:
    text = (
        "A flask was charged with chiral acid (2.9 g, 6.11 mmoles), "
        "triethylamine (3.00 mL, 21.5 mmoles), amine (1.23 g, 6.72 mmoles), "
        "and dimethylformamide (10 mL). The solution was treated with TBTU "
        "(2.26 g, 7.03 mmoles). The reaction was allowed to stir at room "
        "temperature overnight. The mixture was concentrated under vacuum. "
        "The crude product was purified by flash chromatography and "
        "crystallized from hot acetone to give product in 77% yield."
    )

    conditions = extract_source_conditions(
        text,
        source_amount_names=(
            "chiral acid",
            "triethylamine",
            "amine",
            "dimethylformamide",
            "TBTU",
        ),
    )

    assert conditions["reagents"] == [
        "chiral acid",
        "triethylamine",
        "amine",
        "TBTU",
    ]
    assert conditions["solvent"] == ["dimethylformamide"]
    assert conditions["temperature"] == "room temperature"
    assert conditions["time"] == "overnight"
    assert conditions["yield_percent"] == 77.0
    assert conditions["scale"] == "2.9 g, 6.11 mmoles"
    assert "concentrated" in conditions["workup"]
    assert "flash chromatography" in conditions["purification"]


def test_empty_source_text_does_not_invent_conditions() -> None:
    assert extract_source_conditions("") == {}
