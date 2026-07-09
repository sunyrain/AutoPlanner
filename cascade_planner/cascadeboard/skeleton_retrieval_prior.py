"""Literature/database skeleton retrieval priors.

This module adds known type/EC skeletons from the training pack as soft route
priors. It does not create reaction candidates, stock facts, or conditions.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.skeleton_inpainter import SkeletonResult


RDLogger.DisableLog("rdApp.*")

DEFAULT_SKELETON_PRIOR_PATH = Path("results/shared/training_pack/condition_20260507_v2_metadata/skeleton_prior.jsonl")
DISABLE_ENV = "AUTOPLANNER_DISABLE_SKELETON_RETRIEVAL_PRIOR"


def skeleton_retrieval_prior_enabled() -> bool:
    return str(os.environ.get(DISABLE_ENV) or "").lower() not in {"1", "true", "yes"}


def skeleton_retrieval_prior_metadata(
    pack_path: str | Path = DEFAULT_SKELETON_PRIOR_PATH,
) -> dict[str, Any]:
    return {
        "enabled": skeleton_retrieval_prior_enabled(),
        "path": str(pack_path),
        "disable_env": DISABLE_ENV,
        "note": "known-route skeleton retrieval prior; disable for blind SOTA benchmarking",
    }


def augment_skeletons_with_retrieval_prior(
    skeletons: list[SkeletonResult],
    *,
    target_smiles: str,
    depth: int,
    domain: str = "",
    pack_path: str | Path = DEFAULT_SKELETON_PRIOR_PATH,
    max_new: int = 2,
    min_similarity: float = 0.85,
) -> list[SkeletonResult]:
    """Add retrieved literature skeletons to generated skeleton candidates."""
    if not skeleton_retrieval_prior_enabled():
        return skeletons
    rows = retrieve_skeleton_priors(
        target_smiles,
        depth=depth,
        domain=domain,
        pack_path=pack_path,
        limit=max_new,
        min_similarity=min_similarity,
    )
    if not rows:
        return skeletons

    out = list(skeletons)
    seen = {tuple(skel.types) for skel in out}
    condition_source = skeletons[0] if skeletons else None
    for row in rows:
        types = list(row.get("type_sequence") or [])
        if not types or tuple(types) in seen:
            continue
        seen.add(tuple(types))
        skel = SkeletonResult(
            types=types,
            ec1s=_ec1_sequence(row.get("ec1_sequence"), len(types)),
            ec2s=["NONE"] * len(types),
            Ts=_condition_values(condition_source, "Ts", len(types), default=37.0),
            pHs=_condition_values(condition_source, "pHs", len(types), default=7.0),
            compat_pred=row.get("compatibility") or "empirically_compatible",
            opmode_pred=row.get("operation_mode") or "sequential_isolated",
            issues_pred=[],
            log_prob=float(row.get("similarity") or 0.0) + 2.0,
        )
        setattr(skel, "retrieval_prior", {
            "source": "skeleton_prior_pack",
            "similarity": row.get("similarity"),
            "source_path": row.get("source_path"),
            "doi": row.get("doi"),
            "skeleton_id": row.get("skeleton_id"),
        })
        out.append(skel)

    out.sort(key=lambda skel: float(getattr(skel, "log_prob", 0.0) or 0.0), reverse=True)
    return out


def retrieve_skeleton_priors(
    target_smiles: str,
    *,
    depth: int,
    domain: str = "",
    pack_path: str | Path = DEFAULT_SKELETON_PRIOR_PATH,
    limit: int = 5,
    min_similarity: float = 0.85,
    exclude_exact_target: bool = False,
) -> list[dict[str, Any]]:
    fp = _fp(target_smiles)
    if fp is None:
        return []
    target_canonical = _canonical_smiles(target_smiles) if exclude_exact_target else ""
    rows = []
    for row in _load_prior_rows(str(pack_path)):
        if int(row.get("depth") or 0) != int(depth):
            continue
        if domain and row.get("route_domain") and row.get("route_domain") != domain:
            continue
        if target_canonical and _canonical_smiles(row.get("target_smiles")) == target_canonical:
            continue
        row_fp = row.get("_fp")
        if row_fp is None:
            continue
        sim = float(DataStructs.TanimotoSimilarity(fp, row_fp))
        if sim < min_similarity:
            continue
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        out["similarity"] = round(sim, 6)
        rows.append(out)
    rows.sort(key=lambda row: (float(row.get("similarity") or 0.0), float(row.get("label") or 0.0)), reverse=True)
    return rows[:limit]


@lru_cache(maxsize=4)
def _load_prior_rows(pack_path: str) -> tuple[dict[str, Any], ...]:
    path = Path(pack_path)
    if not path.exists():
        return ()
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = _fp(row.get("target_smiles"))
            if fp is None:
                continue
            row["_fp"] = fp
            rows.append(row)
    return tuple(rows)


def _fp(smiles: str | None):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _canonical_smiles(smiles: str | None) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    return Chem.MolToSmiles(mol) if mol is not None else ""


def _ec1_sequence(values: Any, n: int) -> list[int]:
    out = []
    for value in values or []:
        try:
            text = str(value or "")
            out.append(int(text.split(".", 1)[0]) if text else 0)
        except (TypeError, ValueError):
            out.append(0)
    return (out + [0] * n)[:n]


def _condition_values(skeleton: SkeletonResult | None, attr: str, n: int, *, default: float) -> list[float]:
    values = list(getattr(skeleton, attr, []) or []) if skeleton is not None else []
    return [float(x) for x in (values + [default] * n)[:n]]
