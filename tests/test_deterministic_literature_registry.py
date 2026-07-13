from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
    _extract_labeled_procedures,
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.reaction_step_verifier import (
    canonical_reaction_digest,
)
from cascade_planner.harness.source_text_companion import (
    validate_source_text_companion_binding,
)
from cascade_planner.harness.tools import (
    ToolExecutionState,
    _deterministically_validate_source_detail_steps,
    _source_detail_steps_from_visual_candidate,
)


def _evidence(pdf: Path, *, source_ref: str = "doi:10.1000/exact") -> dict:
    return {
        "schema_version": "materialized_source_evidence.v1",
        "document_id": "pdf:exact",
        "manifest_path": str(pdf.with_suffix(".json")),
        "manifest_sha256": "1" * 64,
        "source_pdf_path": str(pdf),
        "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_number": 1,
        "image_path": str(pdf.with_suffix(".png")),
        "image_sha256": "2" * 64,
        "source_ref": source_ref,
    }


def test_extract_labeled_procedures_keeps_heading_and_forward_procedure() -> None:
    rows = _extract_labeled_procedures(
        [
            {
                "page_number": 1,
                "text": (
                    "\n\nEthanol (T1). Acetic acid was added and the reaction "
                    "mixture was stirred to afford T1.\n\n"
                    "Acetaldehyde (T2). T1 was treated with oxidant and afforded T2."
                ),
            }
        ]
    )

    assert [(row["label"], row["name"]) for row in rows] == [
        ("T1", "Ethanol"),
        ("T2", "Acetaldehyde"),
    ]
    assert "Acetic acid was added" in rows[0]["procedure"]
    assert "T1 was treated" in rows[1]["procedure"]


def test_extract_labeled_procedures_accepts_wrapped_patent_example_headings() -> None:
    rows = _extract_labeled_procedures(
        [
            {
                "page_number": 4,
                "text": (
                    "EXAMPLES\nExample\n1 Preparation of 8-bromo-3-methyl-"
                    "xanthine\n0049 Acetic acid and 3-methyl-xanthine were "
                    "charged. The reaction mixture was stirred to isolate the "
                    "title compound.\n"
                    "Example: 2\nPreparation of 3-methyl-7-(2-butyn-1-yl)-"
                    "8-bromo\nxanthine\n0050 The compound from Example 1 and "
                    "1-bromo-2-butyne were added. The reaction mixture was "
                    "stirred to isolate the title compound."
                ),
            }
        ]
    )

    assert [(row["label"], row["name"]) for row in rows] == [
        ("1", "8-bromo-3-methyl-xanthine"),
        ("2", "3-methyl-7-(2-butyn-1-yl)-8-bromo xanthine"),
    ]
    assert "1-bromo-2-butyne" in rows[1]["procedure"]


def test_extract_labeled_procedures_rejects_patent_cover_metadata() -> None:
    rows = _extract_labeled_procedures(
        [
            {
                "page_number": 1,
                "text": (
                    "United States Patent Application Publication (57). "
                    "The abstract says the reaction mixture was processed.\n\n"
                    "Example 1 Preparation of ethyl acetate\n0049 Ethanol was "
                    "added and the reaction mixture was stirred."
                ),
            }
        ]
    )

    assert [(row["label"], row["name"]) for row in rows] == [
        ("1", "ethyl acetate")
    ]


def test_process_patent_heading_uses_numbered_compound_definitions() -> None:
    pages = [
        {
            "page_number": 1,
            "text": (
                "The process reacts compound (3) (ethanol) and compound (4) "
                "(acetic acid) to give compound (5) (ethyl acetate).\n"
                "Intermediate 4 is obtained commercially\n"
                "Synthesis of ethyl acetate (5)\n"
                "[0045] Intermediate 3 was dissolved in THF. A solution of "
                "intermediate 4 was added and the reaction mixture was stirred "
                "to afford ethyl acetate."
            ),
        }
    ]
    rows = _extract_labeled_procedures(pages)

    experimental = [row for row in rows if row.get("declaration_only") is not True]
    definitions = {
        row["label"]: row["name"]
        for row in rows
        if row.get("declaration_only") is True
    }
    assert [(row["label"], row["name"]) for row in experimental] == [
        ("5", "ethyl acetate")
    ]
    assert definitions == {"3": "ethanol", "4": "acetic acid", "5": "ethyl acetate"}


def test_process_patent_numbered_definitions_authorize_only_the_experiment(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "process-patent.pdf"
    pdf.write_bytes(b"%PDF- deterministic process patent")
    step = {
        "step_id": "ethyl-acetate-process",
        "product_smiles": "CCOC(C)=O",
        "reactant_smiles": ["CCO", "CC(=O)O"],
        "source_ref": "patent:EP0000001A1",
        "source_evidence": [_evidence(pdf, source_ref="patent:EP0000001A1")],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "Compound (3) (ethanol) and compound (4) (acetic acid) give "
                "compound (5) (ethyl acetate).\n"
                "Synthesis of ethyl acetate (5)\n"
                "Intermediate 3 was dissolved in THF and intermediate 4 was "
                "added. The reaction mixture was stirred to afford the product."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {
                "ethanol": "CCO",
                "acetic acid": "CC(=O)O",
                "ethyl acetate": "CCOC(C)=O",
            }[name],
            candidate_name_resolver=lambda _smiles: [],
            pdf_text_loader=lambda _path: pages,
        )

    assert audit["approved_binding_count"] == 1
    assert audit["records"][0]["accepted"] is True


def test_extract_strips_formulation_suffix_before_name_to_structure_parse() -> None:
    rows = _extract_labeled_procedures(
        [
            {
                "page_number": 15,
                "text": (
                    "\n\nNirmatrelvir carboxamide (1 eq MTBE solvate) "
                    "(6, MTBE solvate). T18 was treated with Burgess reagent "
                    "and the reaction mixture afforded 6."
                ),
            }
        ]
    )

    assert rows[0]["name"] == "Nirmatrelvir carboxamide"
    assert rows[0]["label"] == "6"


def test_extract_strips_procedure_heading_before_common_name_resolution() -> None:
    rows = _extract_labeled_procedures(
        [
            {
                "page_number": 12,
                "text": (
                    "\n\nProcedure for streamlined multicomponent synthesis "
                    "of nirmatrelvir (1): The imine 3 and acid 2 were added, "
                    "and the reaction mixture afforded 1."
                ),
            }
        ]
    )

    assert rows[0]["name"] == "nirmatrelvir"


def test_parser_approves_only_source_reconstructed_product_and_reactant(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    evidence = _evidence(pdf)
    step = {
        "step_id": "ethanol_from_acetic_acid",
        "product_smiles": "CCO",
        "reactant_smiles": ["CC(=O)O"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [evidence],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nEthanol (T1). Acetic acid was added and the reaction "
                "mixture was stirred to afford T1."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {"Ethanol": "CCO"}[name],
            candidate_name_resolver=lambda smiles: ["acetic acid"],
            pdf_text_loader=lambda path: pages,
        )

    registry = json.loads((tmp_path / "registry.json").read_text())
    assert audit["approved_binding_count"] == 1
    assert registry["bindings"][0]["reaction_digest"] == (
        canonical_reaction_digest("CCO", ["CC(=O)O"])
    )
    assert registry["bindings"][0]["authority"] == {
        "type": "deterministic_structure_parser",
        "id": PARSER_AUTHORITY_ID,
    }


def test_image_only_pdf_uses_hash_bound_source_text_companion_and_replays_it(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"%PDF- image only fixture")
    html = tmp_path / "patent.html"
    html.write_text(
        """
        <html><head><meta name="DC.relation" content="WO2021250648A1"></head>
        <body>
          <div id="p0001" class="description-paragraph">
            Step 7. Synthesis of Ethanol (C7).
          </div>
          <div id="p0002" class="description-paragraph">
            Acetic acid was added and the reaction mixture was stirred to afford C7.
          </div>
        </body></html>
        """,
        encoding="utf-8",
    )
    source_ref = "patent:WO2021250648A1"
    step = {
        "step_id": "companion_ethanol",
        "product_smiles": "CCO",
        "reactant_smiles": ["CC(=O)O"],
        "source_ref": source_ref,
        "source_evidence": [_evidence(pdf, source_ref=source_ref)],
        "source_text_companions": [
            {
                "schema_version": "trusted_source_text_companion.v1",
                "artifact_path": str(html),
                "artifact_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
                "document_identity": "WO2021250648A1",
                "source_url": (
                    "https://patents.google.com/patent/WO2021250648A1/en"
                ),
                "format": "google_patents_html.v1",
                "sections": [
                    {
                        "page_number": 1,
                        "start_element_id": "p0001",
                        "end_element_id": "p0002",
                    }
                ],
            }
        ],
    }

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {"Ethanol": "CCO"}[name],
            candidate_name_resolver=lambda smiles: ["acetic acid"],
            pdf_text_loader=lambda path: [],
        )

    assert audit["approved_binding_count"] == 1
    binding = audit["records"][0]["binding"]
    companion = binding["source_text_companion"]
    assert companion["sections"][0]["page_number"] == 1
    assert validate_source_text_companion_binding(
        companion,
        expected_source_ref=source_ref,
    )
    html.write_text("tampered", encoding="utf-8")
    assert not validate_source_text_companion_binding(
        companion,
        expected_source_ref=source_ref,
    )


def test_source_labels_materialize_exact_structures_instead_of_trusting_visual_smiles(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "label_materialized",
        "product_name": "T2",
        "reactant_names": ["T1"],
        # Deliberately wrong model structures: the source headings, not these
        # values, must become the approved reaction chemistry.
        "product_smiles": "CCC",
        "reactant_smiles": ["C"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nAcetic acid (T1). Reagent was added and the reaction "
                "mixture was stirred to afford T1.\n\n"
                "Ethanol (T2). T1 was treated with reductant and the reaction "
                "mixture afforded T2."
            ),
        }
    ]
    structures = {"Acetic acid": "CC(=O)O", "Ethanol": "CCO"}

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: structures[name],
            candidate_name_resolver=lambda smiles: [],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 1
    binding = audit["records"][0]["binding"]
    assert binding["source_candidate_reaction_digest"] == (
        canonical_reaction_digest("CCC", ["C"])
    )
    assert binding["reaction_digest"] == canonical_reaction_digest(
        "CCO", ["CC(=O)O"]
    )
    assert binding["source_formulation"] == {
        "schema_version": "literature_source_formulation.v1",
        "product_smiles": "CCO",
        "reactant_smiles": ["CC(=O)O"],
        "reaction_digest": canonical_reaction_digest("CCO", ["CC(=O)O"]),
    }
    diagnostic = audit["records"][0]["candidate_diagnostics"][0]
    assert diagnostic["product_match_mode"] == (
        "source_label_authoritative_structure_reconstruction"
    )
    assert diagnostic["source_label_reactants_materialized"] is True


def test_parser_rejects_unmentioned_model_reactant(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "unmentioned_reactant",
        "product_smiles": "CCO",
        "reactant_smiles": ["CCN"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nEthanol (T1). Acetic acid was added and the reaction "
                "mixture was stirred to afford T1."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: "CCO",
            candidate_name_resolver=lambda smiles: ["ethylamine"],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 0
    assert audit["records"][0]["accepted"] is False
    assert (
        "candidate_reactants_not_all_source_resolved"
        in audit["records"][0]["reasons"]
    )


def test_parser_accepts_source_confirmed_hcl_product_parent(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "ethylamine_hcl",
        "product_smiles": "CC[NH3+].[Cl-]",
        "reactant_smiles": ["CC(N)=O"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nEthylamine (T1), HCl salt. Acetamide was added and "
                "the reaction mixture was stirred to afford T1."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {"Ethylamine": "CCN"}[name],
            candidate_name_resolver=lambda smiles: ["acetamide"],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 1
    diagnostic = audit["records"][0]["candidate_diagnostics"][0]
    assert diagnostic["product_match_mode"] == (
        "source_heading_parent_plus_confirmed_formulation"
    )
    binding = audit["records"][0]["binding"]
    assert binding["reaction_digest"] == canonical_reaction_digest(
        "CCN", ["CC(N)=O"]
    )
    assert binding["source_formulation_reaction_digest"] == (
        canonical_reaction_digest("CC[NH3+].[Cl-]", ["CC(N)=O"])
    )
    assert binding["synthesis_projection"] == {
        "schema_version": "literature_synthesis_projection.v1",
        "normalization_policy": (
            "largest_covalent_fragment_and_counterion_neutralization"
        ),
        "normalization_applied": True,
        "product_smiles": "CCN",
        "reactant_smiles": ["CC(N)=O"],
        "reaction_digest": canonical_reaction_digest("CCN", ["CC(N)=O"]),
    }


def test_parser_projects_confirmed_mtbe_solvate_to_covalent_route(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "acetonitrile_mtbe_solvate",
        "product_smiles": "CC#N.COC(C)(C)C",
        "reactant_smiles": ["CC(N)=O"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nAcetonitrile (T1), MTBE solvate. Acetamide was "
                "dehydrated and the reaction mixture afforded T1."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {"Acetonitrile": "CC#N"}[name],
            candidate_name_resolver=lambda smiles: ["acetamide"],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 1
    binding = audit["records"][0]["binding"]
    assert binding["reaction_digest"] == canonical_reaction_digest(
        "CC#N", ["CC(N)=O"]
    )
    assert binding["synthesis_projection"]["normalization_applied"] is True
    assert binding["synthesis_projection"]["product_smiles"] == "CC#N"


def test_parser_adds_source_declared_atom_donor_to_exact_reaction(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    product = "CCNC(=O)C(F)(F)F"
    donor = "CCOC(=O)C(F)(F)F"
    step = {
        "step_id": "trifluoroacetylation",
        "product_smiles": product,
        "reactant_smiles": ["CCN"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
        "condition_candidate": {
            "reagent": "ethyl trifluoroacetate",
        },
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nN-ethyl-2,2,2-trifluoroacetamide (T1). "
                "Ethylamine and ethyl trifluoroacetate were added, and "
                "the reaction mixture afforded T1."
            ),
        }
    ]
    structures = {
        "N-ethyl-2,2,2-trifluoroacetamide": product,
        "ethyl trifluoroacetate": donor,
    }

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: structures[name],
            candidate_name_resolver=lambda smiles: ["ethylamine"],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 1
    record = audit["records"][0]
    binding = record["binding"]
    assert binding["source_candidate_reaction_digest"] == (
        canonical_reaction_digest(product, ["CCN"])
    )
    assert binding["reaction_digest"] == canonical_reaction_digest(
        product,
        ["CCN", donor],
    )
    assert set(binding["synthesis_projection"]["reactant_smiles"]) == {
        "CCN",
        donor,
    }
    diagnostic = record["candidate_diagnostics"][0]
    assert diagnostic["condition_atom_donor_added"] is True
    assert diagnostic["element_inventory_deficit"] == {}
    assert (
        "source_condition_atom_donor_opsin_exact_structure"
        in binding["parser_audit"]["reactant_match_modes"]
    )


def test_parser_resolves_full_reactant_name_transcribed_in_procedure(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "named_reactant",
        "product_smiles": "CCO",
        "reactant_smiles": ["CC(=O)O"],
        "reactant_labels": ["Ethanoic acid"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nEthanol (T1). Ethanoic acid was added and the reaction "
                "mixture was stirred to afford T1."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {
                "Ethanol": "CCO",
                "Ethanoic acid": "CC(=O)O",
            }[name],
            candidate_name_resolver=lambda smiles: [],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 1
    reactant_match = audit["records"][0]["candidate_diagnostics"][0][
        "reactant_matches"
    ][0]
    assert reactant_match["match_mode"] == (
        "source_declared_name_opsin_exact_structure"
    )


def test_parser_accepts_compiled_exact_row_reactant_names_field(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "compiled_named_reactant",
        "product_smiles": "CCO",
        "reactant_smiles": ["CC(=O)O"],
        # Exact-row compilation uses this field name.  It must not lose the
        # independently replayed source-name binding at the promotion boundary.
        "reactant_names": ["Ethanoic acid"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }
    pages = [
        {
            "page_number": 1,
            "text": (
                "\n\nEthanol (T1). Ethanoic acid was added and the reaction "
                "mixture was stirred to afford T1."
            ),
        }
    ]

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: {
                "Ethanol": "CCO",
                "Ethanoic acid": "CC(=O)O",
            }[name],
            candidate_name_resolver=lambda smiles: [],
            pdf_text_loader=lambda path: pages,
        )

    assert audit["approved_binding_count"] == 1
    assert audit["records"][0]["candidate_diagnostics"][0][
        "reactant_matches"
    ][0]["match_mode"] == "source_declared_name_opsin_exact_structure"


def test_parser_rejects_desolvation_as_synthesis_edge(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "formulation_only",
        "product_smiles": "CCO",
        "reactant_smiles": ["CCO.COC(C)(C)C"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: "CCO",
            candidate_name_resolver=lambda smiles: [],
            pdf_text_loader=lambda path: [],
        )

    assert audit["approved_binding_count"] == 0
    assert audit["records"][0]["reasons"] == [
        "noncovalent_formulation_or_salt_state_transition_not_synthesis_edge"
    ]


def test_parser_rejects_when_opsin_heading_does_not_match_product(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    step = {
        "step_id": "wrong_product",
        "product_smiles": "CCO",
        "reactant_smiles": ["CC(=O)O"],
        "source_ref": "doi:10.1000/exact",
        "source_evidence": [_evidence(pdf)],
    }

    with patch(
        "cascade_planner.harness.deterministic_literature_registry."
        "_materialized_source_evidence_valid",
        return_value=True,
    ):
        audit = compile_deterministic_literature_step_registry(
            [step],
            registry_path=tmp_path / "registry.json",
            structure_resolver=lambda name: "CCN",
            candidate_name_resolver=lambda smiles: ["acetic acid"],
            pdf_text_loader=lambda path: [
                {
                    "page_number": 1,
                    "text": (
                        "\n\nEthanol (T1). Acetic acid was added and the "
                        "reaction mixture was stirred to afford T1."
                    ),
                }
            ],
        )

    assert audit["approved_binding_count"] == 0
    assert audit["records"][0]["reasons"] == [
        "product_not_reconstructed_from_source_heading"
    ]


def test_visual_candidate_carries_matching_pdf_manifest_page_into_exact_compile(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF- deterministic source")
    manifest = tmp_path / "literature_pdf_structure_evidence.json"
    manifest_payload = {
        "schema_version": "literature_pdf_structure_evidence.v1",
        "accepted": True,
        "source_ref": "doi:10.1000/exact",
        "source_pdf_path": str(pdf),
        "rendered_pages": [
            {"page_number": 15, "image_path": str(tmp_path / "page15.png")}
        ],
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "example", "target_smiles": "CCO"},
        preflight={"case_id": "example"},
    )
    state.artifacts["literature_pdf_structure_evidence"] = {
        **manifest_payload,
        "artifact_ref": str(manifest),
    }
    candidate = {
        "schema_version": "visual_structure_candidate_chain.v1",
        "source_ref": "doi:10.1000/exact",
        "evidence_refs": ["current_image:page15"],
        "steps": [
            {
                "step_id": "visual:example",
                "product_smiles": "CCO",
                "reactant_smiles": ["CC=O"],
                "source_ref": "doi:10.1000/exact",
                "source_locator": "page 15",
                "allowed_use": "exact_candidate",
                "not_exact_literature_segment": False,
            }
        ],
    }

    rows = _source_detail_steps_from_visual_candidate(
        state,
        {
            "source_ref": "doi:10.1000/exact",
            "pdf_path": str(pdf),
            "candidate_chain": candidate,
        },
    )

    assert len(rows) == 1
    assert f"{manifest.resolve()}#page=15" in rows[0]["evidence_refs"]


def test_exact_compile_uses_bound_visual_artifact_and_skips_exploratory_steps(
    tmp_path: Path,
) -> None:
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={"target_name": "example", "target_smiles": "CCO"},
        preflight={"case_id": "example"},
    )
    state.artifacts["visual_structure_candidate_chain_history"] = [
        {
            "schema_version": "visual_structure_candidate_chain.v1",
            "source_ref": "doi:10.1000/wrong",
            "steps": [
                {
                    "step_id": "wrong_1",
                    "product_smiles": "CCN",
                    "reactant_smiles": ["CC=N"],
                    "source_locator": "page 1",
                },
                {
                    "step_id": "wrong_2",
                    "product_smiles": "CC=N",
                    "reactant_smiles": ["CC#N"],
                    "source_locator": "page 2",
                },
            ],
        }
    ]
    bound_candidate = {
        "schema_version": "visual_structure_candidate_chain.v1",
        "source_ref": "doi:10.1000/exact",
        "steps": [
            {
                "step_id": "bound_exact",
                "product_smiles": "CCO",
                "reactant_smiles": ["CC=O"],
                "source_locator": "page 15",
                "allowed_use": "exact_candidate",
                "not_exact_literature_segment": False,
            },
            {
                "step_id": "bound_approximate",
                "product_smiles": "CC=O",
                "reactant_smiles": ["CC"],
                "source_locator": "page 14",
                "allowed_use": "exploratory_template_and_guided_hint_only",
                "not_exact_literature_segment": True,
            },
        ],
    }
    artifact = tmp_path / "bound_visual_result.json"
    artifact.write_text(
        json.dumps(
            {
                "accepted": True,
                "result": {"candidate_chain": bound_candidate},
            }
        ),
        encoding="utf-8",
    )

    rows = _source_detail_steps_from_visual_candidate(
        state,
        {
            "source_ref": "doi:10.1000/exact",
            "artifact_ref": str(artifact),
        },
    )

    assert [row["step_id"] for row in rows] == ["bound_exact"]
    assert rows[0]["relation_type"] == "exact"


def test_exploratory_visual_smiles_with_exact_labels_reaches_deterministic_materializer(
    tmp_path: Path,
) -> None:
    source_ref = "patent:WO2021250648A1"
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={
            "target_name": "example",
            "target_smiles": "CCO",
            "deterministic_literature_parser_policy": {"enabled": True},
            "literature_sources": [
                {
                    "source_ref": source_ref,
                    "source_text_companion": {
                        "schema_version": "trusted_source_text_companion.v1"
                    },
                }
            ],
        },
        preflight={"case_id": "example"},
    )
    candidate = {
        "schema_version": "visual_structure_candidate_chain.v1",
        "source_ref": source_ref,
        "steps": [
            {
                "step_id": "stereo_incomplete_but_labeled",
                "product_label": "C33",
                "reactant_labels": ["C32", "C7"],
                "product_smiles": "CCC",
                "reactant_smiles": ["CC", "CN"],
                "source_locator": "PDF page 121",
                "allowed_use": "exploratory_template_and_guided_hint_only",
                "stereochemistry_status": "unspecified",
                "not_exact_literature_segment": True,
            }
        ],
    }

    rows = _source_detail_steps_from_visual_candidate(
        state,
        {"source_ref": source_ref, "candidate_chain": candidate},
    )

    assert [row["step_id"] for row in rows] == [
        "stereo_incomplete_but_labeled"
    ]
    assert rows[0]["curation_status"] == (
        "pending_deterministic_source_label_materialization"
    )
    assert rows[0]["allowed_use"] == (
        "deterministic_source_label_materialization_candidate"
    )
    assert rows[0]["structure_derivation"][
        "model_smiles_remain_advisory"
    ] is True


def test_deterministic_parser_promotes_step_before_exact_downstream_compile(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={
            "deterministic_literature_parser_policy": {
                "enabled": True,
                "registry_path": str(registry),
            }
        },
        preflight={"case_id": "example"},
    )
    step = {
        "step_id": "oxidation",
        "source_ref": "doi:10.1000/exact",
        "product_smiles": "CC=O",
        "reactant_smiles": ["CCO"],
        "validation_status": "draft_validated_by_rdkit_chain",
        "curation_status": "visual_candidate_promoted_for_exact_row_compile",
    }
    binding = {
        "binding_id": "det-parser:approved",
        "reaction_digest": canonical_reaction_digest("CC=O", ["CCO"]),
        "source_ref": "doi:10.1000/exact",
        "status": "approved",
        "authority": {
            "type": "deterministic_structure_parser",
            "id": "test-parser",
        },
    }
    parser_audit = {
        "schema_version": "deterministic_literature_registry_audit.v1",
        "accepted": True,
        "approved_binding_count": 1,
        "records": [{"accepted": True, "binding": binding}],
    }

    with patch.dict(
        "os.environ",
        {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(registry)},
    ), patch(
        "cascade_planner.harness.tools."
        "materialize_source_detail_step_evidence",
        return_value=[{"schema_version": "materialized_source_evidence.v1"}],
    ), patch(
        "cascade_planner.harness.tools."
        "compile_deterministic_literature_step_registry",
        return_value=parser_audit,
    ):
        promoted, audit = _deterministically_validate_source_detail_steps(
            state, [step]
        )

    assert audit is parser_audit
    assert promoted[0]["validation_status"] == "deterministically_validated"
    assert promoted[0]["curation_status"] == "deterministically_validated"
    assert promoted[0]["curator_record_id"] == "det-parser:approved"
    assert promoted[0]["source_evidence"] == [
        {"schema_version": "materialized_source_evidence.v1"}
    ]


def test_deterministic_parser_promotes_covalent_projection_and_keeps_source_formulation(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={
            "deterministic_literature_parser_policy": {
                "enabled": True,
                "registry_path": str(registry),
            }
        },
        preflight={"case_id": "example"},
    )
    source_product = "CC[NH3+].[Cl-]"
    source_reactants = ["CC=O"]
    source_digest = canonical_reaction_digest(
        source_product, source_reactants
    )
    projected_digest = canonical_reaction_digest("CCN", source_reactants)
    step = {
        "step_id": "salt_product",
        "source_ref": "doi:10.1000/exact",
        "product_smiles": source_product,
        "reactant_smiles": source_reactants,
        "main_reactant_smiles": "CC=O",
        "applicability": {
            "reconstructed_product_smiles": source_product,
        },
    }
    binding = {
        "binding_id": "det-parser:projected",
        "reaction_digest": projected_digest,
        "source_formulation_reaction_digest": source_digest,
        "source_ref": "doi:10.1000/exact",
        "status": "approved",
        "authority": {
            "type": "deterministic_structure_parser",
            "id": PARSER_AUTHORITY_ID,
        },
        "synthesis_projection": {
            "schema_version": "literature_synthesis_projection.v1",
            "normalization_policy": (
                "largest_covalent_fragment_and_counterion_neutralization"
            ),
            "normalization_applied": True,
            "product_smiles": "CCN",
            "reactant_smiles": source_reactants,
            "reaction_digest": projected_digest,
        },
    }
    parser_audit = {
        "schema_version": "deterministic_literature_registry_audit.v1",
        "accepted": True,
        "approved_binding_count": 1,
        "records": [{"accepted": True, "binding": binding}],
    }

    with patch.dict(
        "os.environ",
        {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(registry)},
    ), patch(
        "cascade_planner.harness.tools."
        "materialize_source_detail_step_evidence",
        return_value=[{"schema_version": "materialized_source_evidence.v1"}],
    ), patch(
        "cascade_planner.harness.tools."
        "compile_deterministic_literature_step_registry",
        return_value=parser_audit,
    ):
        promoted, _ = _deterministically_validate_source_detail_steps(
            state, [step]
        )

    row = promoted[0]
    assert row["product_smiles"] == "CCN"
    assert row["reactant_smiles"] == source_reactants
    assert row["applicability"]["reconstructed_product_smiles"] == "CCN"
    assert row["source_binding_reaction_digest"] == projected_digest
    assert row["source_formulation"]["product_smiles"] == source_product
    assert row["source_formulation"][
        "source_formulation_reaction_digest"
    ] == source_digest


def test_deterministic_parser_injects_verified_condition_atom_donor(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    state = ToolExecutionState(
        run_dir=tmp_path,
        target_input={
            "deterministic_literature_parser_policy": {
                "enabled": True,
                "registry_path": str(registry),
            }
        },
        preflight={"case_id": "example"},
    )
    product = "CCNC(=O)C(F)(F)F"
    donor = "CCOC(=O)C(F)(F)F"
    candidate_digest = canonical_reaction_digest(product, ["CCN"])
    projected_digest = canonical_reaction_digest(product, ["CCN", donor])
    step = {
        "step_id": "trifluoroacetylation",
        "source_ref": "doi:10.1000/exact",
        "product_smiles": product,
        "reactant_smiles": ["CCN"],
    }
    binding = {
        "binding_id": "det-parser:atom-donor",
        "reaction_digest": projected_digest,
        "source_candidate_reaction_digest": candidate_digest,
        "source_formulation_reaction_digest": projected_digest,
        "source_ref": "doi:10.1000/exact",
        "status": "approved",
        "authority": {
            "type": "deterministic_structure_parser",
            "id": PARSER_AUTHORITY_ID,
        },
        "source_formulation": {
            "product_smiles": product,
            "reactant_smiles": ["CCN", donor],
        },
        "synthesis_projection": {
            "normalization_policy": (
                "largest_covalent_fragment_and_counterion_neutralization"
            ),
            "normalization_applied": False,
            "product_smiles": product,
            "reactant_smiles": ["CCN", donor],
            "reaction_digest": projected_digest,
        },
    }
    parser_audit = {
        "accepted": True,
        "approved_binding_count": 1,
        "records": [{"accepted": True, "binding": binding}],
    }

    with patch.dict(
        "os.environ",
        {"AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY": str(registry)},
    ), patch(
        "cascade_planner.harness.tools."
        "materialize_source_detail_step_evidence",
        return_value=[{"schema_version": "materialized_source_evidence.v1"}],
    ), patch(
        "cascade_planner.harness.tools."
        "compile_deterministic_literature_step_registry",
        return_value=parser_audit,
    ):
        promoted, _ = _deterministically_validate_source_detail_steps(
            state, [step]
        )

    row = promoted[0]
    assert set(row["reactant_smiles"]) == {"CCN", donor}
    assert row["source_binding_reaction_digest"] == projected_digest
    assert row["source_formulation"][
        "source_candidate_reaction_digest"
    ] == candidate_digest
