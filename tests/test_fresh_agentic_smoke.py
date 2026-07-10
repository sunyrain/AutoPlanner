from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.harness.parent_route_proof import compile_stitched_parent_route_proof
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from scripts.run_fresh_agentic_smoke import run_fresh_aspirin_smoke


_FIXTURES = Path(__file__).parent / "fixtures"
_SOURCE_PDF = _FIXTURES / "source_evidence_stub.pdf"
_SOURCE_IMAGE = _FIXTURES / "source_page.ppm"
_SOURCE_MANIFEST = _FIXTURES / "source_evidence_manifest.json"
_TRUSTED_REGISTRY = _FIXTURES / "trusted_literature_step_registry.json"


def _strict_source_fields(step_id: str) -> dict:
    template_id = f"source_detail_exact_step:{step_id}"
    return {
        "step_id": step_id,
        "source_template_id": template_id,
        "source_detail_exact_step": True,
        "relation_type": "exact",
        "source_ref": "doi:10.1000/revalidatable-stitch",
        "exact_step_validation": {
            "schema_version": "template_validation_report.v1",
            "accepted": True,
            "allowed_for_one_step_source": True,
            "source_template_id": template_id,
            "reasons": [],
        },
        "source_evidence": [
            {
                "schema_version": "materialized_source_evidence.v1",
                "document_id": "fixture:revalidatable-stitch",
                "manifest_path": str(_SOURCE_MANIFEST.resolve()),
                "manifest_sha256": hashlib.sha256(_SOURCE_MANIFEST.read_bytes()).hexdigest(),
                "source_pdf_path": str(_SOURCE_PDF.resolve()),
                "source_pdf_sha256": hashlib.sha256(_SOURCE_PDF.read_bytes()).hexdigest(),
                "page_number": 1,
                "image_path": str(_SOURCE_IMAGE.resolve()),
                "image_sha256": hashlib.sha256(_SOURCE_IMAGE.read_bytes()).hexdigest(),
                "source_ref": "doi:10.1000/revalidatable-stitch",
            }
        ],
    }


def test_fresh_agentic_smoke_validate_only_accepts_solved_aspirin_run(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY", str(_TRUSTED_REGISTRY))
    run_dir = tmp_path / "fresh_aspirin"
    run_dir.mkdir()
    target_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    salicylic_acid = "O=C(O)c1ccccc1O"
    salicylic_reactants = ["c1ccccc1", "O=C=O", "O"]
    aspirin_reactants = [salicylic_acid, "C=C=O"]
    terminals = ["c1ccccc1", "O=C=O", "O", "C=C=O"]
    stock_path = tmp_path / "fresh_aspirin_stock.smi"
    stock_path.write_text("\n".join(terminals) + "\n", encoding="utf-8")
    salicylic_mapping = (
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[O:7]=[C:8]=[O:9]."
        "[OH2:10]>>[c:1]1([C:8](=[O:7])[OH:9])[cH:2][cH:3][cH:4][cH:5]"
        "[c:6]1[OH:10]"
    )
    aspirin_mapping = (
        "[O:1]=[C:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[OH:10]."
        "[CH2:11]=[C:12]=[O:13]>>"
        "[O:1]=[C:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1"
        "[O:10][C:12]([CH3:11])=[O:13]"
    )
    verifier = verify_chemenzy_raw_routes(
            {
                "target": target_smiles,
                "stock_catalog_context": {
                    "effective_stock_names": ["fresh-aspirin-smoke-stock"],
                    "catalog_bindings": [
                        {
                            "name": "fresh-aspirin-smoke-stock",
                            "path": str(stock_path),
                            "sha256": hashlib.sha256(stock_path.read_bytes()).hexdigest(),
                        }
                    ],
                },
                "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": terminals,
                        "terminal_stock_status": {item: True for item in terminals},
                    },
                        "steps": [
                            {
                                **_strict_source_fields("aspirin_salicylic_acid"),
                                "product": salicylic_acid,
                                "reactant_smiles": salicylic_reactants,
                                "stock_status": {item: True for item in salicylic_reactants},
                                "reaction_type": "materialized salicylic acid smoke route",
                                "atom_mapped_reaction_smiles": salicylic_mapping,
                            },
                            {
                                **_strict_source_fields("aspirin_acetylation"),
                                "product": target_smiles,
                                "reactant_smiles": aspirin_reactants,
                                "stock_status": {item: True for item in terminals},
                                "reaction_type": "materialized aspirin smoke route",
                                "atom_mapped_reaction_smiles": aspirin_mapping,
                        }
                    ],
                }
            ],
        },
        target_smiles=target_smiles,
    )
    proof = compile_stitched_parent_route_proof(
        target_smiles=target_smiles,
        target_name="aspirin",
        case_id="aspirin",
        parent_verifier=verifier,
    )
    assert proof["accepted"], proof["reasons"]
    (run_dir / "agent_blackboard.json").write_text(
        json.dumps(
            {
                "case_id": "aspirin",
                    "target_profile": {
                        "target_name": "aspirin",
                        "target_smiles": target_smiles,
                    },
                    "parent_route_proof": proof,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "action_batch_round_1.json").write_text(
        json.dumps({"actions": [{"action_type": "run_guided_chemenzy"}]}),
        encoding="utf-8",
    )
    (run_dir / "action_batch_round_2.json").write_text(
        json.dumps({"actions": [{"action_type": "stitch_parent_route"}]}),
        encoding="utf-8",
    )
    (run_dir / "final_verdict.json").write_text(
        json.dumps(
            {
                "verdict": "solved",
                "route_status": "solved",
                "solved": True,
                "reasons": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_fresh_aspirin_smoke(output_dir=run_dir, validate_only=True)

    assert summary["accepted"], summary["validation"]["reasons"]
    assert summary["validate_only"] is True
    assert summary["action_types"] == ["run_guided_chemenzy", "stitch_parent_route"]
    assert summary["final_verdict"]["verdict"] == "solved"
    assert (run_dir / "fresh_agentic_smoke_summary.json").exists()
    assert (run_dir / "route_forest.html").exists()
