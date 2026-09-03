#!/usr/bin/env python3
"""Resolve admitted target names to non-admitting PubChem structure candidates.

Name resolution is only a review accelerator.  Results from this script never
populate ``structures.json`` or ``planner_targets.jsonl``: a source-concordant
stereochemical comparison and the review ledger remain mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPResponse
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


USER_AGENT = "AutoPlanner-recent-synthesis-structure-resolution/0.1"
PROPERTIES = (
    "Title,IUPACName,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,"
    "MolecularFormula"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-slots",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/target_slots.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/recent_total_synthesis/structure_resolution_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tmp/recent-total-synthesis-structure-cache"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--include-nonprimary", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cache_key(name: str) -> str:
    return hashlib.sha256(name.casefold().strip().encode("utf-8")).hexdigest()[:20]


def query_url(name: str) -> str:
    return (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name, safe='')}/property/{PROPERTIES}/JSON"
    )


def fetch_payload(url: str, cache_path: Path, *, offline: bool) -> tuple[bytes, bool]:
    if cache_path.exists():
        return cache_path.read_bytes(), True
    if offline:
        raise FileNotFoundError(f"offline structure cache missing: {cache_path}")
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        response: HTTPResponse
        with urlopen(request, timeout=60) as response:
            payload = response.read(10_000_001)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        payload = exc.read(10_000_001)
    if len(payload) > 10_000_000:
        raise ValueError("PubChem response exceeds 10 MB")
    json.loads(payload.decode("utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    time.sleep(0.34)
    return payload, False


def pubchem_properties(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list((payload.get("PropertyTable") or {}).get("Properties") or [])


def smiles_value(candidate: dict[str, Any]) -> str:
    # PUG REST currently emits SMILES/ConnectivitySMILES for requests that still
    # use its older CanonicalSMILES/IsomericSMILES property names.
    return str(
        candidate.get("IsomericSMILES")
        or candidate.get("SMILES")
        or candidate.get("CanonicalSMILES")
        or candidate.get("ConnectivitySMILES")
        or ""
    )


def rdkit_roundtrip(smiles: str) -> dict[str, Any]:
    if not smiles:
        return {"status": "missing_smiles", "canonical_isomeric_smiles": ""}
    try:
        from rdkit import Chem
    except ImportError:
        return {"status": "rdkit_unavailable", "canonical_isomeric_smiles": ""}
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"status": "rdkit_invalid", "canonical_isomeric_smiles": ""}
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return {"status": "roundtrip_valid", "canonical_isomeric_smiles": canonical}


def candidate_record(raw: dict[str, Any]) -> dict[str, Any]:
    smiles = smiles_value(raw)
    return {
        "pubchem_cid": int(raw.get("CID") or 0),
        "pubchem_title": str(raw.get("Title") or ""),
        "iupac_name": str(raw.get("IUPACName") or ""),
        "reported_smiles": smiles,
        "connectivity_smiles": str(raw.get("ConnectivitySMILES") or ""),
        "inchi": str(raw.get("InChI") or ""),
        "inchi_key": str(raw.get("InChIKey") or ""),
        "molecular_formula": str(raw.get("MolecularFormula") or ""),
        "rdkit_validation": rdkit_roundtrip(smiles),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    slots_path = (repo_root / args.target_slots).resolve()
    output_path = (repo_root / args.output).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    rows: list[dict[str, Any]] = []
    lookup_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    slots = [
        row
        for row in read_jsonl(slots_path)
        if args.include_nonprimary or row.get("slot_class") == "primary"
    ]
    for slot in slots:
        name = str(slot.get("target_name") or "").strip()
        base = {
            "schema_version": "recent_total_synthesis_structure_resolution.v1",
            "target_slot_id": slot["target_slot_id"],
            "paper_id": slot["paper_id"],
            "doi": slot["doi"],
            "target_name": name,
            "slot_class": slot["slot_class"],
            "target_identity_status": str(slot.get("target_identity_status") or ""),
            "lookup_provider": "PubChem PUG REST",
            "admission_authority": False,
            "source_concordance_checked": False,
            "stereochemistry_checked_against_paper": False,
            "required_next_action": (
                "compare candidate identity and stereochemistry with article/SI structure"
            ),
        }
        if not name:
            rows.append(
                {
                    **base,
                    "lookup_status": "target_name_pending_source_extraction",
                    "query_url": "",
                    "cache_path": "",
                    "cache_sha256": "",
                    "candidates": [],
                }
            )
            continue

        normalized = name.casefold()
        if normalized not in lookup_cache:
            url = query_url(name)
            path = cache_dir / f"{cache_key(name)}.json"
            try:
                payload, reused = fetch_payload(url, path, offline=args.offline)
                properties = pubchem_properties(json.loads(payload.decode("utf-8")))
                lookup_cache[normalized] = (
                    [candidate_record(candidate) for candidate in properties],
                    {
                        "query_url": url,
                        "cache_path": path.relative_to(repo_root).as_posix(),
                        "cache_sha256": hashlib.sha256(payload).hexdigest(),
                        "cache_reused": reused,
                        "lookup_error": "",
                    },
                )
            except Exception as exc:
                lookup_cache[normalized] = (
                    [],
                    {
                        "query_url": url,
                        "cache_path": (
                            path.relative_to(repo_root).as_posix() if path.exists() else ""
                        ),
                        "cache_sha256": "",
                        "cache_reused": False,
                        "lookup_error": f"{type(exc).__name__}:{exc}",
                    },
                )
        candidates, provenance = lookup_cache[normalized]
        status = (
            "lookup_error"
            if provenance["lookup_error"]
            else "candidate_found_unverified"
            if candidates
            else "no_pubchem_name_match"
        )
        review_flags: list[str] = []
        if "pubchem_conflict" in base["target_identity_status"]:
            review_flags.append("known_name_service_identity_conflict")
        if len(candidates) > 1:
            review_flags.append("multiple_name_resolution_candidates")
        if any(
            candidate.get("reported_smiles")
            and "@" not in str(candidate.get("reported_smiles"))
            for candidate in candidates
        ):
            review_flags.append("candidate_contains_no_explicit_tetrahedral_stereo")
        rows.append(
            {
                **base,
                "lookup_status": status,
                **provenance,
                "candidates": candidates,
                "review_flags": review_flags,
            }
        )

    write_jsonl(output_path, rows)
    summary = {
        "primary_target_slots": len(rows),
        "named_target_slots": sum(bool(row["target_name"]) for row in rows),
        "candidate_found_unverified": sum(
            row["lookup_status"] == "candidate_found_unverified" for row in rows
        ),
        "no_pubchem_name_match": sum(
            row["lookup_status"] == "no_pubchem_name_match" for row in rows
        ),
        "target_name_pending_source_extraction": sum(
            row["lookup_status"] == "target_name_pending_source_extraction"
            for row in rows
        ),
        "lookup_errors": sum(row["lookup_status"] == "lookup_error" for row in rows),
        "admitted_structures": 0,
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
