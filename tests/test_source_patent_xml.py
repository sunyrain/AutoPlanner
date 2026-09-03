from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.harness.deterministic_literature_registry import (
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.source_patent_xml import (
    materialize_primary_patent_xml,
)
from cascade_planner.harness.source_text_companion import (
    EPO_ST36_XML_FORMAT,
    materialize_source_text_companion_pages,
    validate_source_text_companion_binding,
)


PUBLICATION = "EP3381900A1"
SOURCE_REF = f"patent:{PUBLICATION}"
SOURCE_URL = (
    "https://data.epo.org/publication-server/rest/v1.2/patents/"
    "EP3381900NWA1/document.xml"
)


def _xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE ep-patent-document PUBLIC "-//EPO//EP PATENT DOCUMENT 1.5//EN"
      "ep-patent-document-v1-5.dtd">
    <ep-patent-document id="EP18163540A1" file="EP18163540NWA1.xml"
      lang="en" country="EP" doc-number="3381900" kind="A1"
      date-publ="20181003" status="n" dtd-version="ep-patent-document-v1-5">
      <description id="desc" lang="en">
        <heading id="h0001"><b>BRIEF DESCRIPTION</b></heading>
        <p id="p0001" num="0001">The route uses compound (3) (ethanol)
          and compound (4) (acetic acid).</p>
        <heading id="h0002"><b>Synthesis of Ethyl acetate (5)</b></heading>
        <p id="p0002" num="0002">Compound 3 (1.0 g, 21.7 mmol) was dissolved
          in tetrahydrofuran. Compound 4 (1.3 g, 21.7 mmol) was added and the
          reaction mixture was stirred for 17 h at 4 degrees C to afford the
          product in 85% yield.</p>
        <heading id="h0003"><b>Unrelated Route B</b></heading>
        <p id="p0003" num="0003">Under nitrogen, another material was added
          and stirred at reflux to afford an impurity in 99% yield.</p>
      </description>
    </ep-patent-document>
    """


def _materialize(tmp_path: Path) -> dict:
    return materialize_primary_patent_xml(
        content=_xml(),
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=SOURCE_URL,
        output_dir=tmp_path,
        target_terms=["ethyl acetate", "ethanol", "acetic acid"],
    )


def _compile(tmp_path: Path, companion: dict, name: str) -> dict:
    structures = {
        "ethyl acetate": "CCOC(C)=O",
        "ethanol": "CCO",
        "acetic acid": "CC(=O)O",
    }
    names = {
        "CCOC(C)=O": ["ethyl acetate"],
        "CCO": ["ethanol"],
        "CC(=O)O": ["acetic acid"],
    }
    return compile_deterministic_literature_step_registry(
        [
            {
                "step_id": "epo-xml-ester",
                "product_smiles": "CCOC(C)=O",
                "reactant_smiles": ["CCO", "CC(=O)O"],
                "source_ref": SOURCE_REF,
                "source_text_companions": [companion],
            }
        ],
        registry_path=tmp_path / name,
        structure_resolver=lambda value: structures.get(
            str(value).casefold(), ""
        ),
        candidate_name_resolver=lambda value: names.get(str(value), []),
    )


def test_epo_st36_xml_is_hash_bound_exact_and_offline_replayable(
    tmp_path: Path,
) -> None:
    materialization = _materialize(tmp_path)

    assert materialization["status"] == "completed"
    assert materialization["model_invocations"] == 0
    assert materialization["visual_invocations"] == 0
    assert materialization["companion"]["format"] == EPO_ST36_XML_FORMAT
    assert materialization["artifact_sha256"] == hashlib.sha256(
        _xml()
    ).hexdigest()

    first = _compile(tmp_path, materialization["companion"], "first.json")
    second = _compile(tmp_path, materialization["companion"], "second.json")
    first_registry = json.loads(
        (tmp_path / "first.json").read_text(encoding="utf-8")
    )
    second_registry = json.loads(
        (tmp_path / "second.json").read_text(encoding="utf-8")
    )

    assert first["approved_binding_count"] == 1
    assert second["approved_binding_count"] == 1
    assert first_registry["content_sha256"] == second_registry["content_sha256"]
    binding = first["records"][0]["binding"]
    assert binding["source_artifact_kind"] == "xml"
    assert binding["source_location"]["kind"] == "xml_element_range"
    assert binding["source_location"]["start_element_id"] == "h0002"
    assert binding["source_location"]["end_element_id"] == "p0002"
    assert binding["parser_audit"]["source_text_authority"] == (
        "hash_bound_primary_xml"
    )
    assert binding["source_conditions"]["yield_percent"] == 85.0
    assert binding["source_conditions"]["time"] == "17 h"
    assert "atmosphere" not in binding["source_conditions"]


def test_epo_xml_replay_rejects_tampering_and_wrong_office_binding(
    tmp_path: Path,
) -> None:
    materialization = _materialize(tmp_path)
    companion = materialization["companion"]
    pages, binding, reasons = materialize_source_text_companion_pages(
        companion,
        source_ref=SOURCE_REF,
    )

    assert pages
    assert not reasons
    assert validate_source_text_companion_binding(
        binding,
        expected_source_ref=SOURCE_REF,
    )
    Path(companion["artifact_path"]).write_text("tampered", encoding="utf-8")
    assert not validate_source_text_companion_binding(
        binding,
        expected_source_ref=SOURCE_REF,
    )

    wrong = materialize_primary_patent_xml(
        content=_xml(),
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=(
            "https://patents.google.com/patent/EP3381900A1/en"
        ),
        output_dir=tmp_path / "wrong",
        target_terms=["ethyl acetate"],
    )
    assert wrong["status"] == "failed"
    assert wrong["reasons"] == ["patent_xml_source_binding_invalid"]


def test_epo_example_heading_resolves_source_alias_and_exact_thioester_procedure(
    tmp_path: Path,
) -> None:
    publication = "EP2483292B1"
    source_ref = f"patent:{publication}"
    source_url = (
        "https://data.epo.org/publication-server/rest/v1.2/patents/"
        "EP2483292NWB1/document.xml"
    )
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <ep-patent-document id="EP10821065B1" lang="en" country="EP"
      doc-number="2483292" kind="B1" date-publ="20141112">
      <description id="desc" lang="en">
        <heading id="h0001"><b>Example 7: Preparation of DMB-S-MMP</b></heading>
        <p id="p0001" num="0001">A solution of N,N-diisopropylethylamine
          (19.9 mL, 120 mmol) and methyl 3-mercaptopropanoate
          (7.21 60 mmol) in isopropyl acetate (i-PrOAc, 100 mL) was cooled
          to 2 degrees C. 2,2-dimethylbutanoyl chloride (8.1 g, 60 mmol)
          was added dropwise over 10 min. The suspension was stirred at
          25 degrees C for 2 h. The reaction was monitored by TLC using
          EtOAc/heptane. The reaction was quenched with ammonium chloride,
          extracted, washed, filtered and concentrated. The crude mixture
          was subjected to column chromatography over silica gel to afford
          10.5 g (80%) of methyl 3-(2,2-dimethylbutanoylthio)propionate
          (DMB-S-MMP).</p>
        <heading id="h0002"><b>Example 8: Unrelated conversion</b></heading>
        <p id="p0002" num="0002">An unrelated product was obtained in
          99% yield under nitrogen.</p>
      </description>
    </ep-patent-document>
    """
    materialization = materialize_primary_patent_xml(
        content=xml,
        publication=publication,
        source_ref=source_ref,
        source_url=source_url,
        output_dir=tmp_path / "source",
        target_terms=["DMB-S-MMP", "2,2-dimethylbutanoyl chloride"],
    )
    structures = {
        "methyl 3-(2,2-dimethylbutanoylthio)propanoate": (
            "CCC(C)(C)C(=O)SCCC(=O)OC"
        ),
        "methyl 3-mercaptopropanoate": "COC(=O)CCS",
        "2,2-dimethylbutanoyl chloride": "CCC(C)(C)C(=O)Cl",
    }
    audit = compile_deterministic_literature_step_registry(
        [
            {
                "step_id": "dmb-s-mmp-thioester",
                "product_smiles": "CCC(C)(C)C(=O)SCCC(=O)OC",
                "reactant_smiles": [
                    "COC(=O)CCS",
                    "CCC(C)(C)C(=O)Cl",
                ],
                "source_ref": source_ref,
                "source_text_companions": [materialization["companion"]],
            }
        ],
        registry_path=tmp_path / "registry.json",
        structure_resolver=lambda value: structures.get(
            str(value).casefold(), ""
        ),
        candidate_name_resolver=lambda _value: [],
    )

    assert audit["approved_binding_count"] == 1, audit
    binding = audit["records"][0]["binding"]
    assert binding["source_location"]["start_element_id"] == "h0001"
    assert binding["source_location"]["end_element_id"] == "p0001"
    assert binding["parser_audit"]["reactant_match_modes"] == [
        "source_amount_name_opsin_exact_structure",
        "source_amount_name_opsin_exact_structure",
    ]
    conditions = binding["source_conditions"]
    assert conditions["solvent"] == ["isopropyl acetate"]
    assert conditions["base"] == ["diisopropylethylamine"]
    assert conditions["temperature"] == "2 degrees C → 25 degrees C"
    assert conditions["time"] == "2 h"
    assert conditions["time_program"] == ["10 min", "2 h"]
    assert conditions["yield_percent"] == 80.0
    assert "TLC" not in conditions["purification"]
    assert "99% yield" not in json.dumps(conditions)

def test_epo_step_heading_binds_exact_hydrolysis_procedure(tmp_path: Path) -> None:
    publication = "EP3953330B1"
    source_ref = f"patent:{publication}"
    source_url = (
        "https://data.epo.org/publication-server/rest/v1.2/patents/EP3953330NWB1/document.xml"
    )
    product_name = (
        "(1R,2S,5S)-6,6-dimethyl-3-[N-(trifluoroacetyl)-L-valyl]-"
        "3-azabicyclo[3.1.0]hexane-2-carboxylic acid"
    )
    reactant_name = (
        "methyl (1R,2S,5S)-6,6-dimethyl-3-[N-(trifluoroacetyl)-L-valyl]-"
        "3-azabicyclo[3.1.0]hexane-2-carboxylate"
    )
    product_smiles = "CC1([C@H]2CN([C@@H]([C@@H]12)C(=O)O)C([C@@H](NC(C(F)(F)F)=O)C(C)C)=O)C"
    reactant_smiles = "CC1([C@H]2CN([C@@H]([C@@H]12)C(=O)OC)C([C@@H](NC(C(F)(F)F)=O)C(C)C)=O)C"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <ep-patent-document id="EP3953330B1" lang="en" country="EP"
      doc-number="3953330" kind="B1" date-publ="20260114">
      <description id="desc" lang="en">
        <heading id="h0012"><b>Step 4. Synthesis of {product_name} (C4).</b></heading>
        <p id="p0199" num="0199">Concentrated hydrochloric acid (20 mL)
          was added to a solution of {reactant_name} (10.0 g, 26 mmol) in
          acetic acid (40 mL) and water (10 mL). The mixture was heated at
          55 degrees C for 3 days, extracted with ethyl acetate, washed,
          dried, filtered and concentrated to afford the title compound
          (8.1 g, 83%).</p>
        <heading id="h0013"><b>Step 5. Synthesis of unrelated material</b></heading>
        <p id="p0200" num="0200">Water (2 mL) was added and the mixture
          stirred at room temperature for 1 h.</p>
      </description>
    </ep-patent-document>
    """.encode()
    materialization = materialize_primary_patent_xml(
        content=xml,
        publication=publication,
        source_ref=source_ref,
        source_url=source_url,
        output_dir=tmp_path / "source",
        target_terms=[product_name, reactant_name],
    )
    structures = {
        product_name.casefold(): product_smiles,
        reactant_name.casefold(): reactant_smiles,
    }
    audit = compile_deterministic_literature_step_registry(
        [
            {
                "step_id": "nirmatrelvir-c3-to-c4-hydrolysis",
                "product_smiles": product_smiles,
                "reactant_smiles": [reactant_smiles],
                "source_ref": source_ref,
                "source_text_companions": [materialization["companion"]],
            }
        ],
        registry_path=tmp_path / "registry.json",
        structure_resolver=lambda value: structures.get(str(value).casefold(), ""),
        candidate_name_resolver=lambda _value: [],
    )

    assert audit["approved_binding_count"] == 1, audit
    binding = audit["records"][0]["binding"]
    assert binding["source_location"]["start_element_id"] == "h0012"
    assert binding["source_location"]["end_element_id"] == "p0199"
    assert binding["parser_audit"]["product_label"] == "C4"
    assert binding["parser_audit"]["reactant_match_modes"] == [
        "source_amount_name_opsin_exact_structure"
    ]
    conditions = binding["source_conditions"]
    assert "Concentrated hydrochloric acid" in conditions["reagents"]
    assert conditions["solvent"] == ["acetic acid", "water"]
    assert conditions["temperature"] == "55 degrees C"
    assert conditions["time"] == "3 days"
    assert conditions["yield_percent"] == 83.0
    assert "room temperature" not in json.dumps(conditions)
