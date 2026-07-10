from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cascade_planner.harness.agent_action_planner import (
    _source_candidate_payload,
    plan_action_batch,
    validate_action_batch,
)
from cascade_planner.harness.agentic_blackboard import (
    initialize_agent_blackboard,
    update_blackboard_from_action,
)
from cascade_planner.harness.agentic_blackboard_controller import (
    _normalize_literature_sources,
)


DOI = "10.1021/ja00083a066"


class LiteratureSourceDocumentTests(unittest.TestCase):
    def _normalize(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return _normalize_literature_sources(
            literature_pdf_path="",
            literature_pdf_source_ref="",
            literature_sources=rows,
            auto_discover_local_pdfs=False,
            local_pdf_search_dirs=None,
            run_dir=None,
        )

    def _board(self, sources: list[dict[str, object]]) -> dict[str, object]:
        board = initialize_agent_blackboard(
            target_input={
                "case_id": "paclitaxel_documents",
                "target_name": "Paclitaxel",
                "target_smiles": "CC",
                "literature_sources": sources,
            },
            preflight={
                "accepted": True,
                "case_id": "paclitaxel_documents",
                "canonical_smiles": "CC",
                "target_profile": {"heavy_atoms": 47, "rings": 8},
            },
            max_rounds=8,
            budget_limits={"max_visual_calls": 10},
        )
        board["target_side_disconnection_hypotheses"] = {
            "hypotheses": [{"hypothesis_id": "already_seeded"}]
        }
        return board

    def test_normalize_preserves_article_and_si_for_one_doi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            article = Path(tmp) / "holton_article.pdf"
            si = Path(tmp) / "holton_si.pdf"

            sources = self._normalize(
                [
                    {
                        "candidate_id": "holton_article",
                        "doi": DOI,
                        "source_ref": f"doi:{DOI}",
                        "local_pdf": str(article),
                    },
                    {
                        "candidate_id": "holton_si",
                        "doi": DOI,
                        "source_ref": f"doi:{DOI}",
                        "local_pdf": str(si),
                    },
                ]
            )

        self.assertEqual(len(sources), 2)
        self.assertEqual(
            {row["local_pdf"] for row in sources},
            {str(article.resolve()), str(si.resolve())},
        )
        self.assertEqual(len({row["document_id"] for row in sources}), 2)
        self.assertEqual(
            {row["content_scope"] for row in sources},
            {"article", "supplementary_information"},
        )

    def test_normalize_still_dedupes_metadata_only_doi_variants(self) -> None:
        sources = self._normalize(
            [
                {"candidate_id": "doi_plain", "doi": DOI, "title": "Article"},
                {
                    "candidate_id": "doi_url",
                    "doi": f"https://doi.org/{DOI.upper()}",
                    "title": "Duplicate metadata",
                },
            ]
        )

        self.assertEqual(len(sources), 1)

    def test_blackboard_and_planner_process_each_document_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            article = Path(tmp) / "holton_article.pdf"
            si = Path(tmp) / "holton_si.pdf"
            article.write_bytes(b"%PDF-1.4\narticle\n%%EOF\n")
            si.write_bytes(b"%PDF-1.4\nsupplementary information\n%%EOF\n")
            sources = self._normalize(
                [
                    {
                        "candidate_id": "holton_article",
                        "doi": DOI,
                        "source_ref": f"doi:{DOI}",
                        "local_pdf": str(article),
                    },
                    {
                        "candidate_id": "holton_si",
                        "doi": DOI,
                        "source_ref": f"doi:{DOI}",
                        "local_pdf": str(si),
                    },
                ]
            )
            board = self._board(sources)

            candidates = board["literature_evidence"]["source_candidates"]
            self.assertEqual(len(candidates), 2)
            self.assertEqual(len({row["local_pdf"] for row in candidates}), 2)

            first_pdf_batch = plan_action_batch(board, round_index=1, max_actions=1)
            first_pdf = first_pdf_batch["actions"][0]
            self.assertEqual(first_pdf["action_type"], "extract_pdf_literature_structures")
            first_path = first_pdf["payload"]["pdf_path"]
            self.assertTrue(validate_action_batch(first_pdf_batch, blackboard=board)["accepted"])

            first_page = Path(tmp) / "first_page.png"
            first_page.write_bytes(b"rendered-first-page")
            board["literature_evidence"]["pdf_structure_evidence"].append(
                {
                    "evidence_id": "rendered:first",
                    "accepted": True,
                    "source_ref": f"doi:{DOI}",
                    "source_pdf_path": first_path,
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(first_page)}
                    ],
                    "summary": {"rendered_page_count": 1},
                }
            )
            second_pdf_batch = plan_action_batch(board, round_index=2, max_actions=1)
            second_pdf = second_pdf_batch["actions"][0]
            self.assertEqual(second_pdf["action_type"], "extract_pdf_literature_structures")
            second_path = second_pdf["payload"]["pdf_path"]
            self.assertNotEqual(first_path, second_path)

            second_page = Path(tmp) / "second_page.png"
            second_page.write_bytes(b"rendered-second-page")
            board["literature_evidence"]["pdf_structure_evidence"].append(
                {
                    "evidence_id": "rendered:second",
                    "accepted": True,
                    "source_ref": f"doi:{DOI}",
                    "source_pdf_path": second_path,
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(second_page)}
                    ],
                    "summary": {"rendered_page_count": 1},
                }
            )
            first_visual_batch = plan_action_batch(board, round_index=3, max_actions=1)
            first_visual = first_visual_batch["actions"][0]
            self.assertEqual(first_visual["action_type"], "extract_visual_literature_chain")
            first_visual_path = first_visual["payload"]["pdf_path"]
            self.assertIn(first_visual_path, {first_path, second_path})
            self.assertTrue(validate_action_batch(first_visual_batch, blackboard=board)["accepted"])

            board["literature_evidence"]["visual_chains"].append(
                {
                    "chain_id": "visual:first",
                    "accepted": False,
                    "source_ref": f"doi:{DOI}",
                    "source_pdf_path": first_visual_path,
                    "candidate_step_count": 0,
                    "reasons": ["no_relevant_steps"],
                }
            )
            second_visual_batch = plan_action_batch(board, round_index=4, max_actions=1)
            second_visual = second_visual_batch["actions"][0]
            self.assertEqual(second_visual["action_type"], "extract_visual_literature_chain")
            self.assertNotEqual(second_visual["payload"]["pdf_path"], first_visual_path)
            self.assertTrue(validate_action_batch(second_visual_batch, blackboard=board)["accepted"])

    def test_literature_search_does_not_collapse_existing_same_doi_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            article = root / "holton_article.pdf"
            si = root / "holton_si.pdf"
            article.write_bytes(b"%PDF-1.4\narticle\n%%EOF\n")
            si.write_bytes(b"%PDF-1.4\nsupplementary information\n%%EOF\n")
            board = self._board(
                self._normalize(
                    [
                        {
                            "candidate_id": "holton_article",
                            "doi": DOI,
                            "source_ref": f"doi:{DOI}",
                            "local_pdf": str(article),
                        },
                        {
                            "candidate_id": "holton_si",
                            "doi": DOI,
                            "source_ref": f"doi:{DOI}",
                            "local_pdf": str(si),
                        },
                    ]
                )
            )
            incoming_article = dict(
                board["literature_evidence"]["source_candidates"][0]
            )
            incoming_article["title"] = "Agent-refreshed article metadata"

            updated = update_blackboard_from_action(
                board,
                action={
                    "action_id": "search:same-doi",
                    "action_type": "search_literature",
                },
                action_result={
                    "accepted": True,
                    "result": {
                        "schema_version": "literature_scout_report.v1",
                        "accepted": True,
                        "source_candidates": [incoming_article],
                        "source_refs": [f"doi:{DOI}"],
                    },
                },
                round_index=1,
                run_dir=root,
            )

        candidates = updated["literature_evidence"]["source_candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {row["local_pdf"] for row in candidates},
            {str(article.resolve()), str(si.resolve())},
        )
        lifecycle = updated["literature_evidence"]["source_lifecycle"]
        self.assertEqual(len(lifecycle), 2)
        self.assertEqual(
            {row["local_pdf"] for row in lifecycle},
            {str(article.resolve()), str(si.resolve())},
        )

    def test_legacy_source_only_evidence_maps_only_when_document_is_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.pdf"
            second = Path(tmp) / "second.pdf"
            sources = self._normalize(
                [
                    {
                        "doi": "10.1000/first",
                        "source_ref": "doi:10.1000/first",
                        "title": "Improved kilogram-scale synthesis route",
                        "local_pdf": str(first),
                    },
                    {
                        "doi": "10.1000/second",
                        "source_ref": "doi:10.1000/second",
                        "local_pdf": str(second),
                    },
                ]
            )
            board = self._board(sources)
            rendered_page = Path(tmp) / "rendered_first.png"
            rendered_page.write_bytes(b"rendered-page")
            board["literature_evidence"]["pdf_structure_evidence"] = [
                {
                    "accepted": True,
                    "source_ref": "doi:10.1000/first",
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(rendered_page)}
                    ],
                    "summary": {"rendered_page_count": 1},
                }
            ]

            batch = plan_action_batch(board, round_index=1, max_actions=1)

        self.assertEqual(batch["actions"][0]["action_type"], "extract_pdf_literature_structures")
        self.assertEqual(batch["actions"][0]["payload"]["source_ref"], "doi:10.1000/second")

    def test_same_doi_source_ref_is_not_a_document_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            article = Path(tmp) / "article.pdf"
            si = Path(tmp) / "article_si.pdf"
            board = self._board(
                self._normalize(
                    [
                        {
                            "doi": DOI,
                            "source_ref": f"doi:{DOI}",
                            "local_pdf": str(article),
                        },
                        {
                            "doi": DOI,
                            "source_ref": f"doi:{DOI}",
                            "local_pdf": str(si),
                        },
                    ]
                )
            )
            rendered_page = Path(tmp) / "ambiguous_rendered.png"
            rendered_page.write_bytes(b"rendered-page")
            board["literature_evidence"]["pdf_structure_evidence"] = [
                {
                    "accepted": True,
                    "source_ref": f"doi:{DOI}",
                    "rendered_pages": [
                        {"page_number": 1, "image_path": str(rendered_page)}
                    ],
                    "summary": {"rendered_page_count": 1},
                }
            ]
            batch = {
                "schema_version": "agent_action_batch.v1",
                "case_id": "ambiguous_same_doi",
                "round_index": 1,
                "actions": [
                    {
                        "schema_version": "agent_action.v1",
                        "action_id": "visual:ambiguous",
                        "action_type": "extract_visual_literature_chain",
                        "rationale": "extract the DOI source",
                        "expected_artifact": "visual_literature_chain.v1",
                        "success_condition": "one document is extracted",
                        "payload": {"source_ref": f"doi:{DOI}"},
                    }
                ],
            }

            validation = validate_action_batch(batch, blackboard=board)

        self.assertFalse(validation["accepted"])
        self.assertIn(
            "source_sensitive_action_missing_source_binding:0:extract_visual_literature_chain",
            validation["reasons"],
        )

    def test_visual_focus_requires_an_explicit_source_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "legacy-named-source.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            legacy_identifier_row = {
                "title": "Total Synthesis of Ouabagenin and Ouabain",
                "doi": "10.1002/asia.200800429",
                "source_ref": "doi:10.1002/asia.200800429",
                "local_pdf": str(pdf),
            }

            inferred = _source_candidate_payload(
                self._board(self._normalize([legacy_identifier_row]))[
                    "literature_evidence"
                ]["source_candidates"][0]
            )
            explicit_candidate = self._board(
                self._normalize(
                    [
                        {
                            **legacy_identifier_row,
                            "visual_extraction_profile": {
                                "page_numbers": [3, 7],
                                "max_images": 2,
                                "route_sequence_hint": "Inspect only the caller-selected route labels.",
                            },
                        }
                    ]
                )
            )["literature_evidence"]["source_candidates"][0]
            explicit = _source_candidate_payload(explicit_candidate)

        self.assertNotIn("route_sequence_hint", inferred)
        self.assertNotIn("page_numbers", inferred)
        self.assertEqual(explicit["page_numbers"], [3, 7])
        self.assertEqual(explicit["max_images"], 2)
        self.assertEqual(
            explicit["route_sequence_hint"],
            "Inspect only the caller-selected route labels.",
        )


if __name__ == "__main__":
    unittest.main()
