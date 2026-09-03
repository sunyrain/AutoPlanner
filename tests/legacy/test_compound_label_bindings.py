from __future__ import annotations

import tempfile
from pathlib import Path

from cascade_planner.research.source_detail_chain_builder import (
    build_source_detail_curator_records_from_chain,
    compile_hybrid_route_set,
    compile_source_detail_chain_route,
    probe_literature_plugin_chain,
    resolve_curator_records_to_source_detail_steps,
)
from cascade_planner.legacy.harness_runtime.visual_structure_extraction import (
    validate_visual_structure_chain,
)


def test_source_local_compound_label_cannot_bind_two_structures() -> None:
    report = validate_visual_structure_chain(
        {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source_ref": "patent:WO2021250648A1",
            "document_id": "WO2021250648A1-main",
            "steps": [
                {
                    "step_id": "first",
                    "product_label": "C16",
                    "product_smiles": "CCN",
                    "reactant_labels": ["start-a"],
                    "reactant_smiles": ["CC"],
                    "condition_candidate": "reagent A",
                    "source_locator": "page 10, example 1",
                },
                {
                    "step_id": "second",
                    "product_label": "c16",
                    "product_smiles": "CCO",
                    "reactant_labels": ["start-b"],
                    "reactant_smiles": ["CO"],
                    "condition_candidate": "reagent B",
                    "source_locator": "page 12, example 2",
                },
            ],
        },
        require_contiguous=False,
    )

    assert report["accepted"] is False
    audit = report["compound_binding_audit"]
    assert audit["accepted"] is False
    assert audit["independent_source_group"] == "patent:WO2021250648A1"
    assert audit["conflict_count"] == 1
    assert audit["conflicts"][0]["normalized_label"] == "c16"
    assert {row["canonical_smiles"] for row in audit["conflicts"][0]["structures"]} == {
        "CCN",
        "CCO",
    }
    assert all(step["accepted"] is False for step in report["steps"])


def test_visual_literature_chain_builds_source_detail_rows_and_plugin_probe() -> None:
    candidate_chain = {
        "schema_version": "visual_structure_candidate_chain.v1",
        "case_id": "acetaldehyde_chain",
        "target_name": "acetaldehyde",
        "target_smiles": "CC=O",
        "source_ref": "doi:10.0000/visual-chain",
        "source_title": "Visual chain source",
        "evidence_refs": ["scheme:1"],
        "route_order": "forward_start_to_target",
        "source_excerpt": "Scheme 1 reports conversion of compound 1 to compound 3.",
        "default_condition_candidate": {
            "solvent": "water",
            "temperature": "25 C",
        },
        "chain": [
            {"label": "1", "smiles": "CC", "source_locator": "Scheme 1, compound 1"},
            {"label": "2", "smiles": "CCO", "source_locator": "Scheme 1, compound 2"},
            {"label": "3", "smiles": "CC=O", "source_locator": "Scheme 1, compound 3"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        validation = validate_visual_structure_chain(
            candidate_chain,
            output_dir=root,
            target_smiles="CC=O",
        )
        curator = build_source_detail_curator_records_from_chain(validation, output_dir=root)
        resolution = resolve_curator_records_to_source_detail_steps(
            curator,
            output_dir=root,
            target_name="acetaldehyde",
            target_smiles="CC=O",
            source_ref="doi:10.0000/visual-chain",
        )
        compiled_route = compile_source_detail_chain_route(
            source_detail_steps=resolution["source_detail_route_steps"],
            output_dir=root,
            target_smiles="CC=O",
            case_id="acetaldehyde_chain",
            terminal_smiles="CC",
            terminal_name="compound 1",
        )
        plugin_probe = probe_literature_plugin_chain(
            plugin_payload=compiled_route["compiled_downstream"]["literature_template_plugin"],
            validation=validation,
            output_dir=root,
        )
        hybrid = compile_hybrid_route_set(
            output_dir=root,
            case_id="acetaldehyde_chain",
            target_smiles="CC=O",
            literature_chain_audit=compiled_route["chain_audit"],
            chemenzy_result={"routes": [{"route": ["exploratory"]}]},
            verifier_report={"accepted": False, "reasons": ["fake_closed"]},
        )

    assert validation["accepted"] is True, validation["reasons"]
    assert validation["continuity_audit"]["target_match"] is True
    assert validation["summary"]["step_count"] == 2
    assert len(curator["records"]) == 2
    assert resolution["summary"]["source_detail_route_step_count"] == 2
    assert compiled_route["accepted"] is False
    assert "source_detail_step_not_trusted_curated" in compiled_route["reasons"]
    assert compiled_route["compiled_downstream"]["literature_template_plugin"][
        "one_step_rows"
    ] == []
    assert compiled_route["chain_audit"]["step_count"] == 0
    assert compiled_route["chain_audit"]["terminal_reached"] is False
    assert plugin_probe["accepted"] is False
    assert plugin_probe["matched_count"] == 0
    assert hybrid["accepted"] is True
    assert hybrid["summary"]["literature_route_count"] == 0
    assert hybrid["summary"]["literature_advisory_route_count"] == 1
