from __future__ import annotations

import copy
from pathlib import Path

from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
)
from cascade_planner.harness.real_patent_procedure_gate import (
    SNAPSHOT_SCHEMA,
    compile_patent_procedure_gate_suite,
    content_digest,
    load_patent_procedure_gate_config,
    replay_patent_procedure_case,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_gate_config_is_digest_bound_and_has_three_distinct_cases() -> None:
    config = load_patent_procedure_gate_config(
        ROOT / "benchmarks" / "real_patent_procedure_gate_cases.v1.json"
    )
    cases = config["cases"]

    assert len(cases) == 3
    assert len({row["publication"] for row in cases}) == 3
    assert len({row["reaction_class"] for row in cases}) == 3
    assert all(row["source_url"].startswith("https://data.epo.org/") for row in cases)


def test_one_official_xml_case_replays_twice_without_models(tmp_path: Path) -> None:
    publication = "EP0000001B1"
    product_name = "ethyl acetate"
    case = {
        "case_id": "ethyl-acetate-esterification",
        "step_id": "ethyl-acetate",
        "target_name": "Ethyl acetate",
        "reaction_class": "esterification",
        "publication": publication,
        "source_url": (
            "https://data.epo.org/publication-server/rest/v1.2/patents/EP0000001NWB1/document.xml"
        ),
        "target_terms": [product_name, "ethanol", "acetic acid", "sulfuric acid"],
        "product_name": product_name,
        "product_smiles": "CCOC(C)=O",
        "reactant_smiles": ["CCO", "CC(=O)O"],
        "source_structure_names": [
            product_name,
            "ethanol",
            "acetic acid",
            "sulfuric acid",
        ],
        "expected_source_location": {
            "start_element_id": "h0001",
            "end_element_id": "p0001",
        },
        "condition_expectations": {
            "required_fields": ["solvent", "temperature", "time", "yield_percent"],
            "equals": {"time": "2 h", "yield_percent": 80.0},
            "contains": {"solvent": ["ethanol", "acetic acid"]},
        },
    }
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <ep-patent-document id="EP0000001B1" lang="en" country="EP"
      doc-number="0000001" kind="B1" date-publ="20260114">
      <description id="desc" lang="en">
        <heading id="h0001">Step 1. Synthesis of ethyl acetate (C1).</heading>
        <p id="p0001" num="0001">Ethanol (4.6 g, 100 mmol) and acetic
          acid (6.0 g, 100 mmol) were combined, then sulfuric acid (0.1 g,
          1 mmol) was added at 25 degrees C. The mixture was stirred for
          2 h, then concentrated to afford ethyl acetate
          (7.0 g, 80% yield).</p>
        <heading id="h0002">Step 2. Synthesis of unrelated material.</heading>
        <p id="p0002" num="0002">Water (2 mL) was stirred for 9 h.</p>
      </description>
    </ep-patent-document>"""
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA,
        "authority_id": PARSER_AUTHORITY_ID,
        "structures": {
            product_name: "CCOC(C)=O",
            "ethanol": "CCO",
            "acetic acid": "CC(=O)O",
            "sulfuric acid": "OS(=O)(=O)O",
        },
        "candidate_names": {
            "CCOC(C)=O": [product_name],
            "CCO": ["ethanol"],
            "CC(=O)O": ["acetic acid"],
            "O=S(=O)(O)O": ["sulfuric acid"],
        },
        "semantics": {"snapshot_replay_uses_no_network": True},
    }
    snapshot["content_sha256"] = content_digest(snapshot)

    acceptance = replay_patent_procedure_case(
        case,
        source_content=xml,
        resolver_snapshot=snapshot,
        output_dir=tmp_path / case["case_id"],
    )

    assert acceptance["accepted"] is True, acceptance
    assert acceptance["exact_edge"]["source_location"] == {
        "kind": "xml_element_range",
        "start_element_id": "h0001",
        "end_element_id": "p0001",
        "text_sha256": acceptance["exact_edge"]["procedure_text_sha256"],
    }
    assert acceptance["offline_replay"]["registry_digests_equal"] is True
    assert acceptance["offline_replay"]["binding_ids_equal"] is True
    assert acceptance["offline_replay"]["model_invocations"] == 0
    assert acceptance["acquisition_cascade"]["pdf_fallback_count"] == 0
    assert acceptance["showcase"]["portfolio_accepted"] is False
    assert (tmp_path / case["case_id"] / "route_workbench.html").is_file()


def test_suite_requires_three_publications_and_reaction_classes(tmp_path: Path) -> None:
    base = {
        "accepted": True,
        "offline_replay": {
            "registry_digests_equal": True,
            "binding_ids_equal": True,
        },
        "acquisition_cascade": {
            "structured_source_closed": True,
            "pdf_fallback_count": 0,
            "ocr_fallback_count": 0,
            "vision_fallback_count": 0,
        },
        "procedure": {"conditions": {"time": "2 h", "yield_percent": 80.0}},
    }
    cases = []
    for index in range(3):
        row = copy.deepcopy(base)
        row.update(
            {
                "case_id": f"case-{index}",
                "target_name": f"Target {index}",
                "publication": f"EP{index}B1",
                "reaction_class": f"class-{index}",
            }
        )
        cases.append(row)
    config = {
        "suite_id": "test-suite",
        "release_gate": {
            "minimum_case_count": 3,
            "minimum_unique_publications": 3,
            "minimum_unique_reaction_classes": 3,
        },
    }

    summary = compile_patent_procedure_gate_suite(config, cases, output_dir=tmp_path / "gate")

    assert summary["three_case_release_gate_passed"] is True
    assert summary["unique_publication_count"] == 3
    assert summary["unique_reaction_class_count"] == 3
    assert summary["acquisition_cascade"]["structured_source_closed_count"] == 3
    assert summary["acquisition_cascade"]["pdf_fallback_count"] == 0
    assert (tmp_path / "gate" / "index.html").is_file()

    duplicate = copy.deepcopy(cases)
    duplicate[2]["publication"] = duplicate[1]["publication"]
    failed = compile_patent_procedure_gate_suite(config, duplicate, output_dir=tmp_path / "failed")
    assert failed["three_case_release_gate_passed"] is False
