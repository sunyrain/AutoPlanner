from __future__ import annotations

from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
    build_source_procedure_record,
    normalize_source_conditions,
)


def test_condition_aliases_preserve_operational_fields_without_source_excerpt() -> None:
    conditions = normalize_source_conditions(
        {
            "reagent": "HATU",
            "base": "DIPEA",
            "solvent": "DMF",
            "duration": "2 h",
            "reported_yield": "81%",
            "source_excerpt": "source-authored text is not a condition field",
        }
    )

    assert conditions == {
        "base": "DIPEA",
        "reagents": ["HATU"],
        "solvent": "DMF",
        "time": "2 h",
        "yield": "81%",
    }
    audit = audit_condition_completeness(conditions)
    assert audit["complete"] is False
    assert audit["missing_required_groups"] == ["temperature"]


def test_hash_bound_procedure_exists_even_when_conditions_are_unparsed() -> None:
    procedure = build_source_procedure_record(
        exact_record={
            "record_id": "exact:1",
            "edge_digest": "edge-digest",
            "source_binding_id": "source:external",
            "source_ref": "patent:US123A1",
            "independence_group": "patent-family:123",
            "location_refs": ["Example 3, paragraph 42"],
        },
        extraction_row={
            "evidence_refs": ["procedure-text-sha256:" + "a" * 64],
            "conditions": {},
        },
        source_binding={"artifact_sha256": "b" * 64},
        extraction_artifact_sha256="c" * 64,
    )

    assert procedure is not None
    assert procedure["procedure_status"] == "procedure_located_condition_unparsed"
    assert procedure["conditions"] == {}
    assert procedure["condition_completeness"]["complete"] is False
    assert procedure["source_fragment"]["procedure_text_sha256"] == "a" * 64
    assert procedure["semantics"]["missing_condition_fields_are_not_inferred"] is True


def test_unhashed_condition_projection_cannot_create_procedure_authority() -> None:
    procedure = build_source_procedure_record(
        exact_record={
            "record_id": "exact:1",
            "edge_digest": "edge-digest",
            "source_binding_id": "source:external",
            "source_ref": "doi:10.1000/example",
            "location_refs": ["page 4"],
        },
        extraction_row={"conditions": {"solvent": "water"}},
        source_binding={},
        extraction_artifact_sha256="c" * 64,
    )

    assert procedure is None


def test_yield_range_is_preserved_without_selecting_one_endpoint() -> None:
    conditions = normalize_source_conditions(
        {
            "catalyst": ["Variant LovD enzyme"],
            "solvent": ["water"],
            "temperature_program": ["25°C"],
            "time_program": ["48 h"],
            "temperature": "25°C",
            "time": "48 h",
            "yield_percent_range": {"min": 85.0, "max": 87.0},
        }
    )

    assert "yield_percent" not in conditions
    assert conditions["yield_percent_range"] == {"min": 85.0, "max": 87.0}
    audit = audit_condition_completeness(conditions)
    assert audit["complete"] is True
    assert audit["yield_reported"] is True
