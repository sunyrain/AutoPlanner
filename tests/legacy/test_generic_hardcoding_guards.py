from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from cascade_planner.legacy.harness_runtime.codex_action_planner import (
    _structure_resolution_payload_targets_input_label,
)
from cascade_planner.research.source_detail_chain_builder import (
    audit_source_detail_route_chain,
)
from cascade_planner.legacy.harness_runtime.tools import (
    ToolExecutionState,
    _structure_label_is_target_identity,
    _visual_candidate_quality_score,
    execute_local_tool,
)


def test_generic_planner_and_projection_surfaces_have_no_target_name_branches() -> None:
    root = Path(__file__).resolve().parents[2]
    surfaces = (
        root / "cascade_planner/legacy/harness_runtime/codex_action_planner.py",
        root / "cascade_planner/legacy/harness_runtime/tools.py",
        root / "cascade_planner/legacy/harness_runtime/route_forest.py",
        root / "cascade_planner/research/source_detail_chain_builder.py",
    )
    forbidden = ("paclitaxel", "taxol", "baccatin", "bufotalin", "ouabagenin", "ouabain")

    for path in surfaces:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(token in source for token in forbidden), path


def test_visual_tool_does_not_invent_source_ref_when_unbound() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = root / "page.png"
        image.write_bytes(b"rendered-page")
        state = ToolExecutionState(
            run_dir=root,
            target_input={"target_name": "generic target", "target_smiles": "CCO"},
            preflight={"case_id": "generic_target"},
        )
        with patch(
            "cascade_planner.legacy.harness_runtime.tools.run_visual_literature_chain_agent",
            return_value={
                "schema_version": "visual_literature_chain_extraction_result.v1",
                "accepted": False,
                "status": "no_candidate",
                "candidate_chain": {},
                "candidate_step_count": 0,
                "reasons": ["no_candidate"],
            },
        ) as visual_agent:
            execute_local_tool(
                "extract_visual_literature_chain",
                {"image_paths": [str(image)]},
                state,
            )

    assert visual_agent.call_args.kwargs["source_ref"] == ""


def test_target_identity_synonym_requires_explicit_alias() -> None:
    implicit = {
        "target_name": "target_case",
        "target_smiles": "CCO",
        "family_hint": "taxane paclitaxel",
    }
    explicit = {**implicit, "target_aliases": ["Taxol"]}

    assert not _structure_label_is_target_identity("Taxol", implicit)
    assert _structure_label_is_target_identity("Taxol", explicit)
    assert _structure_label_is_target_identity("Taxol (1)", explicit)
    assert not _structure_label_is_target_identity("paclitaxel derivative 12", explicit)

    blackboard = {
        "case_id": "paclitaxel",
        "target_profile": {"target_name": "paclitaxel", "family_hint": "taxane"},
    }
    assert _structure_resolution_payload_targets_input_label(
        {"label": "paclitaxel (1)"},
        blackboard,
    )
    assert not _structure_resolution_payload_targets_input_label(
        {"label": "paclitaxel derivative 12"},
        blackboard,
    )


def test_visual_candidate_ranking_does_not_privilege_legacy_labels() -> None:
    def candidate(product_label: str, reactant_label: str) -> dict:
        return {
            "source_ref": "source:current",
            "steps": [
                {
                    "product_label": product_label,
                    "product_smiles": "CCO",
                    "reactant_labels": [reactant_label],
                    "reactant_smiles": ["CC"],
                    "source_locator": "scheme 1",
                }
            ],
        }

    legacy_labels = candidate("bufotalin", "11")
    generic_labels = candidate("generic target", "starting material")

    assert _visual_candidate_quality_score(legacy_labels) == _visual_candidate_quality_score(
        generic_labels
    )


def test_source_detail_terminal_is_not_repaired_from_name_doi_or_step_id() -> None:
    target = "CCO"
    observed_terminal = "CC12CCC3C(C1CCC2=O)CCC4=CC(=O)CCC34C"
    rows = [
        {
            "literature_template_trace": {
                "source_template_id": "source_detail_exact_step:24_from_11",
                "source_ref": "doi:10.1016/j.tet.2025.134610",
                "product_smiles": target,
                "reactant_smiles": [observed_terminal],
                "evidence_refs": ["source:legacy-marker"],
            }
        }
    ]

    audit = audit_source_detail_route_chain(
        rows,
        target_smiles=target,
        case_id="generic_terminal_identity",
    )

    assert audit["accepted"]
    assert audit["terminal_smiles"] == audit["observed_terminal_smiles"]
    assert "@" not in audit["terminal_smiles"]
    assert audit["terminal_stereo_repair"] == {}
    assert audit["source_policy"]["automatic_terminal_identity_repair_allowed"] is False
