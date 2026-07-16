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


def test_biocatalytic_conditions_preserve_programs_ranges_and_conversion() -> None:
    text = (
        "The reaction vessel was charged with Monacolin J hydroxy acid "
        "(10 g, 29.58 mmol). Deionized water (112.0 mL) and NH4OH "
        "(4.2 mL) were added. The pH of the mixture was adjusted from 9.2 "
        "to 9.0 before Variant LovD enzyme (0.10 g) was charged. The mixture "
        "was stirred for 5 minutes at 300 rpm at 25 degrees C. DMB-S-MMP "
        "(7.1 mL, 32.54 mmol) was added to start the enzymatic reaction. "
        "The pH was controlled at 9.0. Approximately 97% conversion was "
        "obtained after 48 h. The product was filtered and washed, then "
        "dried for 24 h to afford 11.4 to 11.7 g (85 to 87% isolated yield)."
    )

    conditions = extract_source_conditions(
        text,
        source_amount_names=(
            "Monacolin J hydroxy acid",
            "deionized water",
            "NH4OH",
            "Variant LovD enzyme",
            "DMB-S-MMP",
        ),
    )

    assert conditions["catalyst"] == ["Variant LovD enzyme"]
    assert conditions["base"] == ["ammonium hydroxide"]
    assert conditions["solvent"] == ["water"]
    assert conditions["ph_program"] == ["9.2", "9.0"]
    assert conditions["ph"] == "9.0"
    assert conditions["agitation"] == "300 rpm"
    assert conditions["temperature"] == "25 degrees C"
    assert conditions["time"] == "48 h"
    assert conditions["conversion_percent"] == 97.0
    assert conditions["yield_percent_range"] == {"min": 85.0, "max": 87.0}
    assert "yield_percent" not in conditions


def test_workup_and_nmr_do_not_pollute_reaction_time_or_temperature() -> None:
    text = (
        "To a stirred solution of compound 33 (15 mg, 0.02 mmol) in THF "
        "(1.5 mL) was added HF (0.25 mL) and pyridine (0.25 mL) at room "
        "temperature. The mixture was allowed to stir for 160 h before it "
        "was quenched with sat. NaHCO3. The product was purified to give "
        "bufotalin (8 mg, 93 %) as a white solid. 1H NMR (500 MHz) showed "
        "three signals. 13C NMR (126 MHz) was recorded."
    )

    conditions = extract_source_conditions(
        text,
        source_amount_names=("compound 33", "HF", "pyridine", "NaHCO3"),
    )

    assert conditions["solvent"] == ["tetrahydrofuran", "pyridine"]
    assert conditions["temperature_program"] == ["room temperature"]
    assert conditions["time_program"] == ["160 h"]
    assert conditions["time"] == "160 h"
    assert conditions["yield_percent"] == 93.0
    assert "13C" not in conditions["temperature"]
    assert conditions["reagents"] == ["compound 33", "HF"]
