"""Build a pre-scored bridge candidate cache for runtime gating.

The route-tree bridge gate should not call the verifier model repeatedly inside
the search loop. This script scores all exact/similarity bridge candidates once
and writes ``bridge_candidates_scored.parquet`` under the bridge pack.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import (
    BridgeCandidate,
    BridgeVerifierV0Scorer,
    ec_sample,
)
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_MODEL_PATH = Path("results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")
DEFAULT_THRESHOLD = 0.8409896871324669


def exact_candidate(row: dict[str, Any]) -> BridgeCandidate | None:
    smiles = canonical_smiles(str(row.get("canonical_smiles") or ""))
    inchikey = str(row.get("inchikey") or "")
    if not smiles or not inchikey:
        return None
    return BridgeCandidate(
        chemical_smiles=smiles,
        enzyme_smiles=smiles,
        chemical_inchikey=inchikey,
        enzyme_inchikey=inchikey,
        bridge_direction=str(row.get("bridge_direction") or ""),
        confidence_tier=str(row.get("confidence_tier") or ""),
        source="exact_bridge_strict",
        tanimoto=1.0,
        enzyme_ec_sample=ec_sample(row.get("enzyme_ec_sample_json")),
        metadata={
            "chemical_occurrences": int(row.get("chemical_occurrences") or 0),
            "enzyme_occurrences": int(row.get("enzyme_occurrences") or 0),
            "enzyme_ec_unique": int(row.get("enzyme_ec_unique") or 0),
            "bridge_flags_json": row.get("bridge_flags_json") or "[]",
        },
    )


def similarity_candidate(row: dict[str, Any]) -> BridgeCandidate | None:
    chemical = canonical_smiles(str(row.get("chemical_smiles") or ""))
    enzyme = canonical_smiles(str(row.get("enzyme_smiles") or ""))
    chemical_key = str(row.get("chemical_inchikey") or "")
    enzyme_key = str(row.get("enzyme_inchikey") or "")
    if not chemical or not enzyme or not chemical_key or not enzyme_key:
        return None
    return BridgeCandidate(
        chemical_smiles=chemical,
        enzyme_smiles=enzyme,
        chemical_inchikey=chemical_key,
        enzyme_inchikey=enzyme_key,
        bridge_direction=str(row.get("bridge_direction") or ""),
        confidence_tier=str(row.get("confidence_tier") or ""),
        source="similarity_bridge_filtered",
        tanimoto=float(row.get("tanimoto") or 0.0),
        enzyme_ec_sample=ec_sample(row.get("enzyme_ec_sample_json")),
        metadata={
            "chemical_occurrences": int(row.get("chemical_occurrences") or 0),
            "enzyme_occurrences": int(row.get("enzyme_occurrences") or 0),
            "enzyme_ec_unique": int(row.get("enzyme_ec_unique") or 0),
        },
    )


def load_candidates(pack_dir: Path) -> list[BridgeCandidate]:
    candidates: list[BridgeCandidate] = []
    for row in pq.read_table(pack_dir / "exact_bridge_strict.parquet").to_pylist():
        candidate = exact_candidate(row)
        if candidate is not None:
            candidates.append(candidate)
    for row in pq.read_table(pack_dir / "similarity_bridge_filtered.parquet").to_pylist():
        candidate = similarity_candidate(row)
        if candidate is not None:
            candidates.append(candidate)
    return dedupe(candidates)


def dedupe(candidates: list[BridgeCandidate]) -> list[BridgeCandidate]:
    out: list[BridgeCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.chemical_inchikey, candidate.enzyme_inchikey, candidate.bridge_direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def candidate_row(candidate: BridgeCandidate) -> dict[str, Any]:
    return {
        "chemical_smiles": candidate.chemical_smiles,
        "enzyme_smiles": candidate.enzyme_smiles,
        "chemical_inchikey": candidate.chemical_inchikey,
        "enzyme_inchikey": candidate.enzyme_inchikey,
        "bridge_direction": candidate.bridge_direction,
        "confidence_tier": candidate.confidence_tier,
        "source": candidate.source,
        "tanimoto": float(candidate.tanimoto),
        "enzyme_ec_sample_json": json.dumps(list(candidate.enzyme_ec_sample)),
        "verifier_score": float(candidate.verifier_score or 0.0),
        "verifier_pass": bool(candidate.verifier_pass),
        "metadata_json": json.dumps(candidate.metadata, ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-scored bridge verifier cache")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    started = time.monotonic()
    output = args.output or (args.pack_dir / "bridge_candidates_scored.parquet")
    candidates = load_candidates(args.pack_dir)
    scorer = BridgeVerifierV0Scorer(args.model_path, threshold=float(args.threshold))
    scored = scorer.score_candidates(candidates)
    rows = [candidate_row(candidate) for candidate in scored]
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    passed = sum(1 for row in rows if row["verifier_pass"])
    report = {
        "schema_version": "bridge_candidates_scored.v0",
        "pack_dir": str(args.pack_dir),
        "model_path": str(args.model_path),
        "threshold": float(args.threshold),
        "output": str(output),
        "rows": len(rows),
        "passed": passed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
