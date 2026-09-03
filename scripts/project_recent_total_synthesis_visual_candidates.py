#!/usr/bin/env python3
"""Project paper-level Codex visual results onto target slots.

The projection keeps visual and PubChem candidates side by side. It never
updates ``structures.json`` or grants benchmark admission authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


PRIMARY_SLOT_CLASSES = {"primary", "primary_candidate"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-slots",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/target_slots.jsonl"),
    )
    parser.add_argument(
        "--structure-candidates",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/structure_resolution_candidates.jsonl"),
    )
    parser.add_argument(
        "--visual-output-dir",
        type=Path,
        default=Path("tmp/recent-total-synthesis-visual-structure-candidates"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/visual_structure_candidates.jsonl"),
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical(smiles: str, *, isomeric: bool) -> str:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return ""
    if not isomeric:
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=isomeric)


def _pubchem_smiles(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for candidate in row.get("candidates") or []:
        validation = candidate.get("rdkit_validation") or {}
        smiles = str(validation.get("canonical_isomeric_smiles") or "")
        if not smiles:
            smiles = _canonical(str(candidate.get("reported_smiles") or ""), isomeric=True)
        if smiles and smiles not in values:
            values.append(smiles)
    return values


def _relation(visual_smiles: str, pubchem_smiles: list[str]) -> str:
    if not visual_smiles:
        return "not_comparable"
    if not pubchem_smiles:
        return "visual_only"
    visual_isomeric = _canonical(visual_smiles, isomeric=True)
    if visual_isomeric in {_canonical(value, isomeric=True) for value in pubchem_smiles}:
        return "exact_isomeric_match"
    visual_connectivity = _canonical(visual_smiles, isomeric=False)
    if visual_connectivity in {_canonical(value, isomeric=False) for value in pubchem_smiles}:
        return "connectivity_match_stereo_difference"
    return "structure_conflict"


def _bound_image(
    target: dict[str, Any], source_images: list[dict[str, Any]]
) -> dict[str, Any]:
    requested = Path(str(target.get("source_image_name") or "")).name
    if requested:
        for row in source_images:
            if Path(str(row.get("image_path") or "")).name == requested:
                return dict(row)
    locator = str(target.get("source_locator") or "")
    for row in source_images:
        prefix = str(row.get("source_locator") or "").split(";", 1)[0]
        if prefix and prefix in locator:
            return dict(row)
    return dict(source_images[0]) if len(source_images) == 1 else {}


def _repo_relative(value: str) -> str:
    path = Path(str(value or ""))
    if not value or not path.is_absolute():
        return str(value or "")
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _portable_source_image(row: dict[str, Any]) -> dict[str, Any]:
    portable = dict(row)
    for key in ("image_path", "source_artifact_path"):
        if key in portable:
            portable[key] = _repo_relative(str(portable.get(key) or ""))
    return portable


def _review_priority(visual_status: str, relation: str, paper_status: str) -> str:
    if paper_status == "no_visual_source":
        return "source_access_gap"
    if relation == "visual_only":
        return "visual_only_high_value"
    if relation == "exact_isomeric_match":
        return "concordant_candidate"
    if relation == "connectivity_match_stereo_difference":
        return "stereochemistry_review"
    if relation == "structure_conflict":
        return "identity_or_transcription_review"
    if visual_status == "invalid_model_smiles":
        return "invalid_visual_smiles"
    if visual_status == "unresolved":
        return "visual_unresolved"
    return "visual_not_attempted"


def _result_index(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("paper-*/visual-structure-result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row["_result_path"] = str(path)
        results[str(row.get("paper_id") or path.parent.name)] = row
    return results


def project_rows(
    slots: list[dict[str, Any]],
    pubchem_rows: list[dict[str, Any]],
    visual_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pubchem_by_slot = {str(row.get("target_slot_id") or ""): row for row in pubchem_rows}
    projected: list[dict[str, Any]] = []
    for slot in slots:
        if slot.get("slot_class") not in PRIMARY_SLOT_CLASSES:
            continue
        paper_id = str(slot.get("paper_id") or "")
        target_name = str(slot.get("target_name") or "")
        paper = visual_results.get(paper_id, {})
        visual_target = next(
            (
                dict(row)
                for row in paper.get("targets") or []
                if str(row.get("target_name") or "") == target_name
            ),
            {},
        )
        validation = dict(visual_target.get("rdkit_validation") or {})
        visual_smiles = str(validation.get("canonical_isomeric_smiles") or "")
        visual_status = str(visual_target.get("status") or "not_attempted")
        source_image = _portable_source_image(
            _bound_image(visual_target, list(paper.get("source_images") or []))
        )
        pubchem = pubchem_by_slot.get(str(slot.get("target_slot_id") or ""), {})
        pubchem_values = _pubchem_smiles(pubchem)
        relation = _relation(visual_smiles, pubchem_values)
        source_locator = str(visual_target.get("source_locator") or "")
        source_locator_complete = bool(
            source_locator
            and source_image.get("source_artifact_path")
            and source_image.get("source_artifact_sha256")
            and source_image.get("image_sha256")
        )
        paper_status = str(paper.get("status") or "not_attempted")
        projected.append(
            {
                "schema_version": "recent_total_synthesis_visual_structure_candidate.v1",
                "target_slot_id": slot.get("target_slot_id", ""),
                "paper_id": paper_id,
                "doi": slot.get("doi", ""),
                "target_name": target_name,
                "slot_class": slot.get("slot_class", ""),
                "paper_attempt_status": paper_status,
                "visual_status": visual_status,
                "visual_reported_isomeric_smiles": visual_target.get(
                    "reported_isomeric_smiles", ""
                ),
                "visual_canonical_isomeric_smiles": visual_smiles,
                "rdkit_validation": validation or {"status": "missing"},
                "transcription_note": visual_target.get("transcription_note", ""),
                "source_image_name": visual_target.get("source_image_name", ""),
                "source_locator": source_locator,
                "source_image": source_image,
                "source_locator_complete": source_locator_complete,
                "visual_result_path": _repo_relative(str(paper.get("_result_path") or "")),
                "visual_input_sha256": paper.get("input_sha256", ""),
                "visual_model": paper.get("model", ""),
                "visual_reasoning_effort": paper.get("reasoning_effort", ""),
                "pubchem_lookup_status": pubchem.get("lookup_status", "not_available"),
                "pubchem_candidate_count": len(pubchem_values),
                "pubchem_canonical_isomeric_smiles": pubchem_values,
                "visual_pubchem_relation": relation,
                "review_priority": _review_priority(visual_status, relation, paper_status),
                "required_next_action": (
                    "independent source-image identity and stereochemistry review by two curators"
                ),
                "source_concordance_checked": False,
                "stereochemistry_checked_against_paper": False,
                "admission_authority": False,
            }
        )
    return projected


def main() -> int:
    args = parse_args()
    slots = _read_jsonl((REPO_ROOT / args.target_slots).resolve())
    pubchem_rows = _read_jsonl((REPO_ROOT / args.structure_candidates).resolve())
    visual_results = _result_index((REPO_ROOT / args.visual_output_dir).resolve())
    rows = project_rows(slots, pubchem_rows, visual_results)
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    relation_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    paper_status_by_id: dict[str, str] = {}
    for row in rows:
        relation = str(row["visual_pubchem_relation"])
        status = str(row["visual_status"])
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        paper_status_by_id[str(row["paper_id"])] = str(row["paper_attempt_status"])
    paper_status_counts: dict[str, int] = {}
    for status in paper_status_by_id.values():
        paper_status_counts[status] = paper_status_counts.get(status, 0) + 1
    summary = {
        "schema_version": "recent_total_synthesis_visual_structure_projection_summary.v1",
        "paper_attempts": len(paper_status_by_id),
        "paper_attempt_status_counts": dict(sorted(paper_status_counts.items())),
        "target_rows": len(rows),
        "visual_status_counts": dict(sorted(status_counts.items())),
        "visual_pubchem_relation_counts": dict(sorted(relation_counts.items())),
        "rdkit_valid_candidates": sum(
            row["rdkit_validation"].get("status") == "roundtrip_valid" for row in rows
        ),
        "source_locator_complete": sum(bool(row["source_locator_complete"]) for row in rows),
        "admission_authority": False,
    }
    summary_path = output.with_name(f"{output.stem}.summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
