from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cascade_planner.application.program_experience_store import (
    PROGRAM_EXPERIENCE_RECORD_SCHEMA,
    build_program_experience_library,
    write_program_experience_library,
)
from cascade_planner.application.reaction_template_store import (
    TEMPLATE_RECORD_SCHEMA,
    build_template_library,
    template_digest,
    write_template_library,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256
from cascade_planner.web.workspace_surface import (
    compiled_program_benchmark_catalog,
    inject_workspace_return,
    self_evolution_catalog,
)


def _template_record(template_id: str, *, successes: int, failures: int) -> dict:
    row = {
        "schema_version": TEMPLATE_RECORD_SCHEMA,
        "template_id": template_id,
        "status": "active",
        "maturity": "reuse_validated" if successes else "single_source_observed",
        "example_count": 1,
        "independent_source_groups": ["source-group:1"],
        "successful_edge_digests": [f"success:{index}" for index in range(successes)],
        "failed_edge_digests": [f"failure:{index}" for index in range(failures)],
    }
    row["content_sha256"] = template_digest(row)
    return row


def _mechanism_experience() -> dict:
    row = {
        "schema_version": PROGRAM_EXPERIENCE_RECORD_SCHEMA,
        "experience_id": "program-experience:mechanism-1",
        "domain": "mechanism",
        "disposition": "supported",
        "authority_scope": "proposal_memory_only",
        "counts": {"positive": 1, "negative": 0, "inconclusive": 0},
        "observations": {"claim:1": {"polarity": "positive"}},
    }
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def test_self_evolution_catalog_distinguishes_retrieval_attempt_and_validation(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "self-evo"
    first = _template_record("template:first", successes=2, failures=0)
    second = _template_record("template:second", successes=0, failures=1)
    write_template_library(
        memory_root / "patent-reaction-template-library.json",
        build_template_library(
            {first["template_id"]: first, second["template_id"]: second},
            generation=4,
        ),
    )
    mechanism = _mechanism_experience()
    write_program_experience_library(
        memory_root / "program-experience-library.json",
        build_program_experience_library({mechanism["experience_id"]: mechanism}, generation=2),
    )

    result = self_evolution_catalog(
        SimpleNamespace(paths=SimpleNamespace(external_data_root=tmp_path))
    )

    assert result["ok"] is True
    assert result["summary"] == {
        "reaction_template_count": 2,
        "retrievable_reaction_template_count": 2,
        "attempted_reaction_template_count": 2,
        "replay_validated_reaction_template_count": 1,
        "successful_reuse_count": 2,
        "failed_reuse_count": 1,
        "program_experience_count": 1,
        "mechanism_experience_count": 1,
        "mechanism_observation_count": 1,
    }
    assert result["reaction_templates"]["integrity"] == "valid"
    assert result["program_experience"]["domain_counts"] == {"mechanism": 1}
    assert "compiled_program_benchmarks" not in result


def test_self_evolution_catalog_keeps_absent_libraries_visible(tmp_path: Path) -> None:
    result = self_evolution_catalog(
        SimpleNamespace(paths=SimpleNamespace(external_data_root=tmp_path))
    )

    assert result["ok"] is True
    assert result["reaction_templates"]["present"] is False
    assert result["program_experience"]["present"] is False
    assert result["summary"]["reaction_template_count"] == 0
    assert result["summary"]["mechanism_experience_count"] == 0


def test_historical_route_workbench_delivery_gets_workspace_return_once() -> None:
    source = '<header class="app-header"><div class="header-actions"></div></header>'

    delivered = inject_workspace_return(source)

    assert 'id="dashboardReturn"' in delivered
    assert 'href="/v4"' in delivered
    assert inject_workspace_return(delivered) == delivered


def test_non_workbench_html_is_not_modified() -> None:
    source = '<html><div class="header-actions">report</div></html>'

    assert inject_workspace_return(source) == source


def test_compiled_program_benchmark_catalog_exposes_bufotalin_six_to_one_fallback() -> None:
    catalog = compiled_program_benchmark_catalog()

    assert catalog["ok"] is True
    assert catalog["record_count"] >= 1
    record = next(
        value
        for value in catalog["records"]
        if value["target_name"] == "bufotalin" and value["chemical_step_equivalent_count"] == 6
    )
    assert record["physical_step_count"] == 1
    assert record["net_step_savings"] == 5
    assert record["authority_scope"] == "proposal_only"
    assert record["validation_status"] == "proposed_screen_required"
    assert record["warning_codes"] == ["EXACT_SUBSTRATE_UNVALIDATED"]
    assert record["benchmark_run_id"].startswith("program-benchmark-bufotalin-6to1-")
    assert record["materialize_url"].endswith("/materialize")
    assert record["workbench_url"].endswith("/workbench.html")
    assert record["boundary"]["precursor"]["label"] == "Compound 11"
    assert record["boundary"]["product"]["label"] == "Compound 28"
    assert [row["product"]["label"] for row in record["fallback_steps"]] == [
        "Compound 24",
        "Compound 25",
        "Compound 23",
        "Compound 26",
        "Compound 27",
        "Compound 28",
    ]
