from __future__ import annotations

from cascade_planner.interfaces.target_solver_compat import (
    compile_saved_run_objective_compatibility,
)


def test_saved_run_objective_compatibility_preserves_but_disempowers_labels() -> None:
    receipt = compile_saved_run_objective_compatibility(
        {
            "objective_mode": "benchmark_search",
            "config": {"objective_mode": "legacy_unknown_value"},
        },
        {
            "config": {"objective_mode": "scientific_proof"},
            "claim": {"objective_mode": "benchmark_search"},
        },
        requested_objective_mode="procurement_delivery",
    )

    assert receipt["schema_version"] == "saved_run_objective_compatibility.v1"
    assert receipt["legacy_objective_present"] is True
    assert receipt["requested_compatibility_view"] == "procurement_delivery"
    assert [row["value"] for row in receipt["legacy_objective_observations"]] == [
        "benchmark_search",
        "legacy_unknown_value",
        "scientific_proof",
        "benchmark_search",
    ]
    assert receipt["semantics"]["legacy_objective_is_not_a_scheduler_input"]
    assert len(receipt["content_sha256"]) == 64


def test_saved_run_without_legacy_objective_is_equally_supported() -> None:
    receipt = compile_saved_run_objective_compatibility(
        {"config": "legacy-non-object"},
        {"config": [], "claim": "legacy-non-object"},
        requested_objective_mode="scientific_proof",
    )

    assert receipt["legacy_objective_present"] is False
    assert receipt["legacy_objective_observations"] == []
