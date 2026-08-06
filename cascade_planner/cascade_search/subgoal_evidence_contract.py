"""Stable evidence and feature contract for cascade subgoal scoring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Recap

from cascade_planner.cascade_search.ids import stable_id
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


QUERY_ROLES = (
    "program_target",
    "target_fragment",
    "step_product",
    "step_product_fragment",
)
EVIDENCE_ROLES = ("program_target", "step_product", "step_reactant")
QUALITY_TIERS = ("gold", "silver")
EVIDENCE_STRENGTHS = (
    "strong_process_evidence",
    "process_evidence",
    "unclear",
    "",
)


def load_program_splits(
    program_manifest: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load train/validation/test program rows from one pack manifest."""
    manifest = json.loads(Path(program_manifest).read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    return {
        split: _read_jsonl(Path(outputs[split]))
        for split in ("train", "val", "test")
    }


def evidence_items(
    programs: list[dict[str, Any]],
    *,
    min_heavy_atoms: int,
) -> list[dict[str, Any]]:
    """Build deduplicated product/reactant evidence rows for runtime retrieval."""
    rows: dict[str, dict[str, Any]] = {}
    for program in programs:
        route_transforms = tuple(
            _norm_transform(step.get("transformation_superclass"))
            for step in program.get("steps") or []
            if isinstance(step, dict)
        )
        compatibility = program.get("compatibility") or {}
        common = {
            "program_id": str(program.get("program_id") or ""),
            "doi": str(program.get("doi") or ""),
            "cascade_id": str(program.get("cascade_id") or ""),
            "cascade_type": _norm(program.get("cascade_type")),
            "quality_tier": _norm(program.get("quality_tier")),
            "evidence_strength": _norm(compatibility.get("evidence_strength")),
            "compatibility_label": _norm(
                compatibility.get("compatibility_label")
            ),
            "route_transforms": route_transforms,
        }
        _add_item(
            rows,
            role="program_target",
            smiles=program.get("target_smiles"),
            transform="",
            source_step_id="",
            common=common,
            min_heavy_atoms=min_heavy_atoms,
        )
        for step in program.get("steps") or []:
            if not isinstance(step, dict):
                continue
            transform = _norm_transform(step.get("transformation_superclass"))
            step_id = str(step.get("transition_id") or step.get("step_id") or "")
            _add_item(
                rows,
                role="step_product",
                smiles=step.get("product_smiles"),
                transform=transform,
                source_step_id=step_id,
                common=common,
                min_heavy_atoms=min_heavy_atoms,
            )
            for reactant in step.get("reactants") or []:
                _add_item(
                    rows,
                    role="step_reactant",
                    smiles=reactant,
                    transform=transform,
                    source_step_id=step_id,
                    common=common,
                    min_heavy_atoms=min_heavy_atoms,
                )
    out = list(rows.values())
    for row in out:
        row["fingerprint"] = molecule_fingerprint(row["smiles"])
    return [row for row in out if row.get("fingerprint") is not None]


def candidate_row(
    query: dict[str, Any],
    evidence: dict[str, Any],
    *,
    similarity: float,
    candidate_rank: int,
    schema: dict[str, Any],
    positive_similarity: float,
    strong_positive_similarity: float,
) -> dict[str, Any]:
    """Encode one query/evidence pair using the serialized model schema."""
    query_transform = _norm_transform(query.get("transform"))
    evidence_transform = _norm_transform(evidence.get("transform"))
    query_route_transforms = {
        _norm_transform(value) for value in query.get("route_transforms") or [] if value
    }
    evidence_route_transforms = {
        _norm_transform(value)
        for value in evidence.get("route_transforms") or []
        if value
    }
    transform_match = bool(
        query_transform
        and evidence_transform
        and query_transform == evidence_transform
    )
    route_transform_overlap = bool(
        query_route_transforms & evidence_route_transforms
    )
    same_cascade_type = bool(
        query.get("cascade_type")
        and evidence.get("cascade_type")
        and query.get("cascade_type") == evidence.get("cascade_type")
    )
    evidence_role = str(evidence.get("role") or "")
    is_product_evidence = evidence_role in {"program_target", "step_product"}
    positive = bool(
        is_product_evidence
        and similarity >= positive_similarity
        and (
            transform_match
            or (
                not query_transform
                and route_transform_overlap
                and same_cascade_type
            )
            or (
                query_transform
                and route_transform_overlap
                and similarity >= strong_positive_similarity
            )
        )
    )
    relevance = 0
    if positive:
        relevance = (
            2
            if similarity >= strong_positive_similarity
            and evidence_role == _preferred_evidence_role(query.get("role"))
            else 1
        )
    row = {
        "query_id": str(query.get("item_id") or ""),
        "query_program_id": query.get("program_id"),
        "query_role": query.get("role"),
        "query_smiles": query.get("smiles"),
        "query_transform": query_transform,
        "query_cascade_type": query.get("cascade_type"),
        "query_heavy_atoms": query.get("heavy_atoms"),
        "query_ring_count": query.get("ring_count"),
        "query_hetero_atoms": query.get("hetero_atoms"),
        "evidence_id": evidence.get("item_id"),
        "evidence_program_id": evidence.get("program_id"),
        "evidence_role": evidence_role,
        "evidence_smiles": evidence.get("smiles"),
        "evidence_transform": evidence_transform,
        "evidence_cascade_type": evidence.get("cascade_type"),
        "evidence_quality_tier": evidence.get("quality_tier"),
        "evidence_strength": evidence.get("evidence_strength"),
        "candidate_rank": int(candidate_rank),
        "similarity": round(float(similarity), 6),
        "transform_match": transform_match,
        "route_transform_overlap": route_transform_overlap,
        "same_cascade_type": same_cascade_type,
        "training_relevance": relevance,
    }
    row["features"] = _feature_row(row, schema)
    return row


def fragments(smiles: str) -> set[str]:
    """Return canonical BRICS and RECAP fragments without dummy atoms."""
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return set()
    out: set[str] = set()
    try:
        out.update(_strip_dummy(value) for value in BRICS.BRICSDecompose(molecule))
    except Exception:
        pass
    try:
        recap = Recap.RecapDecompose(molecule)
        out.update(_strip_dummy(value) for value in recap.GetLeaves())
    except Exception:
        pass
    return {
        canonical_smiles(value)
        for value in out
        if canonical_smiles(value)
    }


def molecule_properties(smiles: str) -> dict[str, Any]:
    """Return stable structural scalar features used by the scorer contract."""
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return {
            "valid": False,
            "heavy_atoms": 0,
            "ring_count": 0,
            "hetero_atoms": 0,
        }
    hetero = sum(
        1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() not in (1, 6)
    )
    return {
        "valid": True,
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "ring_count": int(molecule.GetRingInfo().NumRings()),
        "hetero_atoms": int(hetero),
    }


def molecule_fingerprint(smiles: Any) -> Any:
    """Return the serialized scorer's Morgan-2048 fingerprint contract."""
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)


def _add_item(
    rows: dict[str, dict[str, Any]],
    *,
    role: str,
    smiles: Any,
    transform: str,
    source_step_id: str,
    common: dict[str, Any],
    min_heavy_atoms: int,
) -> None:
    canonical = canonical_smiles(str(smiles or ""))
    props = molecule_properties(canonical)
    if (
        not props["valid"]
        or props["heavy_atoms"] < min_heavy_atoms
        or props["heavy_atoms"] > 90
    ):
        return
    key = stable_id(common.get("program_id"), role, source_step_id, canonical)
    rows[key] = {
        "item_id": key,
        "role": role,
        "smiles": canonical,
        "transform": _norm_transform(transform),
        "source_step_id": source_step_id,
        "heavy_atoms": int(props["heavy_atoms"]),
        "ring_count": int(props["ring_count"]),
        "hetero_atoms": int(props["hetero_atoms"]),
        **common,
    }


def _feature_row(row: dict[str, Any], schema: dict[str, Any]) -> list[float]:
    query_heavy = _float(row.get("query_heavy_atoms"))
    evidence_heavy = _float(row.get("evidence_heavy_atoms"))
    if evidence_heavy == 0.0 and row.get("evidence_smiles"):
        evidence_heavy = float(
            molecule_properties(str(row.get("evidence_smiles")))["heavy_atoms"]
        )
    query_rings = _float(row.get("query_ring_count"))
    evidence_rings = _float(row.get("evidence_ring_count"))
    if evidence_rings == 0.0 and row.get("evidence_smiles"):
        evidence_rings = float(
            molecule_properties(str(row.get("evidence_smiles")))["ring_count"]
        )
    query_hetero = _float(row.get("query_hetero_atoms"))
    evidence_hetero = _float(row.get("evidence_hetero_atoms"))
    if evidence_hetero == 0.0 and row.get("evidence_smiles"):
        evidence_hetero = float(
            molecule_properties(str(row.get("evidence_smiles")))["hetero_atoms"]
        )
    similarity = _float(row.get("similarity"))
    rank = max(1.0, _float(row.get("candidate_rank"), 1.0))
    out = [
        similarity,
        1.0 / rank,
        1.0 / np.log2(rank + 1.0),
        query_heavy,
        evidence_heavy,
        abs(query_heavy - evidence_heavy),
        min(query_heavy, evidence_heavy) / max(max(query_heavy, evidence_heavy), 1.0),
        query_rings,
        evidence_rings,
        abs(query_rings - evidence_rings),
        query_hetero,
        evidence_hetero,
        abs(query_hetero - evidence_hetero),
        float(bool(row.get("same_cascade_type"))),
    ]
    out.extend(_one_hot(row.get("query_role"), QUERY_ROLES))
    out.extend(_one_hot(row.get("evidence_role"), EVIDENCE_ROLES))
    out.extend(_one_hot(row.get("evidence_quality_tier"), QUALITY_TIERS))
    out.extend(_one_hot(row.get("evidence_strength"), EVIDENCE_STRENGTHS))
    for transform in schema.get("evidence_transforms", []):
        out.append(
            float(_norm_transform(row.get("evidence_transform")) == transform)
        )
    out.extend(
        [
            float(bool(row.get("transform_match"))),
            float(bool(row.get("route_transform_overlap"))),
        ]
    )
    return [float(value) for value in out]


def _strip_dummy(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return ""
    try:
        molecule = Chem.DeleteSubstructs(
            molecule,
            Chem.MolFromSmarts("[#0]"),
        )
        Chem.SanitizeMol(molecule)
    except Exception:
        pass
    if molecule is None or molecule.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _preferred_evidence_role(query_role: Any) -> str:
    if str(query_role or "") == "program_target":
        return "program_target"
    return "step_product"


def _one_hot(value: Any, choices: tuple[str, ...]) -> list[float]:
    normalized = _norm(value)
    return [float(normalized == _norm(choice)) for choice in choices]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_transform(value: Any) -> str:
    return _norm(value).replace(" ", "_")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


__all__ = [
    "candidate_row",
    "evidence_items",
    "fragments",
    "load_program_splits",
    "molecule_fingerprint",
    "molecule_properties",
]
