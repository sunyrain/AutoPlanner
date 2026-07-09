"""V3-based enzymatic reaction retrieval for CascadeBoard live mode.

Instead of template-based EnzExpand (150 templates, low coverage),
use v3's 4886 enzymatic reactions as a nearest-neighbor database.

Input: product SMILES + EC class (from skeleton)
Method: Tanimoto similarity on Morgan FP, filtered by EC class
Output: similar reactions from v3 literature
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

_DB_CACHE = {}
_UNIPROT_CACHE = {}


@dataclass
class EnzReaction:
    product_smi: str
    product_fp: np.ndarray
    reactant_smi: str
    rxn_smiles: str
    ec: str
    ec2: str
    T: float | None
    pH: float | None
    transformation: str
    evidence: dict


def _first_nonempty(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _load_uniprot_cache(cache_path: str = "data/uniprot_cache.json") -> dict:
    p = Path(cache_path)
    cache_key = str(p.resolve())
    if cache_key in _UNIPROT_CACHE:
        return _UNIPROT_CACHE[cache_key]
    if not p.exists():
        _UNIPROT_CACHE[cache_key] = {}
        return _UNIPROT_CACHE[cache_key]
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = {}
    _UNIPROT_CACHE[cache_key] = data if isinstance(data, dict) else {}
    return _UNIPROT_CACHE[cache_key]


def _cached_uniprot_evidence(
    catalyst: dict | None,
    ec: str,
    *,
    cache: dict | None = None,
) -> dict:
    catalyst = catalyst or {}
    cache = _load_uniprot_cache() if cache is None else cache
    uid = (catalyst.get("uniprot_id") or catalyst.get("uniprot_accession") or "").strip()
    organism = (
        catalyst.get("organism")
        or catalyst.get("uniprot_lookup_organism")
        or catalyst.get("uniprot_organism")
        or ""
    )

    cached = None
    if uid:
        cached = cache.get(f"accession::{uid}")
    if not cached and ec:
        cached = cache.get(f"{ec}||{organism}") or cache.get(f"{ec}||")
    if not isinstance(cached, dict):
        return {}
    if cached.get("reviewed") is not None:
        uniprot_status = "reviewed" if cached.get("reviewed") else "unreviewed"
    else:
        uniprot_status = cached.get("entry_type")

    out = {
        "uniprot_accession": cached.get("accession") or uid,
        "uniprot_entry_name": cached.get("entry_name"),
        "uniprot_status": uniprot_status,
        "organism": cached.get("organism"),
        "tax_id": cached.get("tax_id"),
        "protein_existence": cached.get("protein_existence"),
        "reviewed": cached.get("reviewed"),
        "sequence": cached.get("sequence"),
        "sequence_length": cached.get("sequence_length"),
        "enzyme_name": cached.get("protein_name"),
        "protein_name": cached.get("protein_name"),
        "cofactor": cached.get("cofactor"),
        "rhea_ids": cached.get("rhea_ids"),
        "ec_numbers": cached.get("ec_numbers"),
        "uniprot_cache_hit": True,
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _build_evidence(record: dict, cascade: dict, step: dict, catalyst: dict | None, similarity: float) -> dict:
    catalyst = catalyst or {}
    external_ids = catalyst.get("enzyme_external_ids") or {}
    ec = catalyst.get("ec_number") or ""
    cached = _cached_uniprot_evidence(catalyst, ec)
    evidence = {
        "source_db": "v3_cascade",
        "doi": record.get("doi"),
        "literature_title": record.get("title"),
        "cascade_id": cascade.get("cascade_id"),
        "step_id": step.get("step_id"),
        "step_uuid": step.get("step_uuid"),
        "transformation_superclass": step.get("transformation_superclass"),
        "transformation_name": step.get("transformation_name"),
        "uniprot_accession": catalyst.get("uniprot_id"),
        "uniprot_status": catalyst.get("uniprot_status"),
        "organism": catalyst.get("organism") or catalyst.get("uniprot_lookup_organism"),
        "tax_id": catalyst.get("tax_id") or catalyst.get("uniprot_tax_id"),
        "protein_existence": catalyst.get("uniprot_protein_existence"),
        "reviewed": catalyst.get("uniprot_status") == "reviewed",
        "sequence": catalyst.get("uniprot_sequence") or catalyst.get("sequence"),
        "sequence_length": catalyst.get("enzyme_seq_length"),
        "enzyme_name": catalyst.get("component_name_canonical") or catalyst.get("component_name"),
        "protein_name": catalyst.get("uniprot_protein_name"),
        "biocatalyst_format": catalyst.get("biocatalyst_format"),
        "engineering_status": catalyst.get("engineering_status"),
        "cofactor": _first_nonempty(
            catalyst.get("cofactor_required"),
            catalyst.get("cofactor_regeneration_mode"),
        ),
        "cofactor_regeneration_mode": catalyst.get("cofactor_regeneration_mode"),
        "substrate_similarity": round(float(similarity), 4),
        "reaction_center_similarity": None,
        "condition_match": {},
        "literature_precedent": True,
        "support_material": catalyst.get("support_material"),
        "rhea_ids": external_ids.get("rhea"),
        "external_ids": external_ids,
    }
    for key, value in cached.items():
        if value in (None, "", [], {}):
            continue
        if key in {"rhea_ids", "ec_numbers"}:
            evidence[key] = value
        elif not evidence.get(key):
            evidence[key] = value
    conds = step.get("step_conditions") or {}
    evidence["condition_match"] = {
        "temperature_c": conds.get("temperature_c"),
        "ph": conds.get("ph"),
        "solvent": conds.get("solvent"),
        "atmosphere": conds.get("atmosphere"),
        "reaction_time_h": conds.get("reaction_time_h"),
    }
    if cached.get("rhea_ids"):
        external_ids = dict(external_ids)
        external_ids.setdefault("rhea", cached["rhea_ids"])
        evidence["external_ids"] = external_ids
    return {k: v for k, v in evidence.items() if v not in (None, "", [], {})}


def _load_db(data_path: str = "cascade_dataset_v3.json") -> list[EnzReaction]:
    p = Path(data_path)
    cache_key = str(p.resolve())
    if cache_key in _DB_CACHE:
        return _DB_CACHE[cache_key]

    if not p.exists():
        _DB_CACHE[cache_key] = []
        return _DB_CACHE[cache_key]

    data = json.loads(p.read_text())
    if isinstance(data, dict):
        records = data.get("records_kept", data.get("records", []))
    else:
        records = data
    if not isinstance(records, list):
        records = []

    db = []
    for rec in records:
        for cas in rec.get("cascades", []):
            for s in cas.get("steps", []):
                cats = s.get("catalyst_components") or []
                ec = next((x.get("ec_number", "") for x in cats if x and x.get("ec_number")), "")
                if not ec:
                    continue
                rxn = s.get("rxn_smiles", "")
                if not rxn or ">>" not in rxn:
                    continue

                parts = rxn.split(">>")
                product_smi = parts[1].strip()
                reactant_smi = parts[0].strip().split(".")[0]

                mol = Chem.MolFromSmiles(product_smi)
                if mol is None:
                    continue

                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_arr = np.zeros(2048, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_arr)

                conds = s.get("step_conditions") or {}
                catalyst = next((x for x in cats if x and x.get("ec_number")), None) or {}
                db.append(EnzReaction(
                    product_smi=product_smi,
                    product_fp=fp_arr,
                    reactant_smi=reactant_smi,
                    rxn_smiles=rxn,
                    ec=ec,
                    ec2=".".join(ec.split(".")[:2]),
                    T=conds.get("temperature_c"),
                    pH=conds.get("ph"),
                    transformation=s.get("transformation_superclass", ""),
                    evidence=_build_evidence(rec, cas, s, catalyst, 1.0),
                ))

    _DB_CACHE[cache_key] = db
    return db


def retrieve_enzymatic_reactions(
    product_smiles: str,
    ec_class: str = "",
    top_k: int = 10,
    min_similarity: float = 0.2,
) -> list[dict]:
    """Retrieve similar enzymatic reactions from v3 database.

    Args:
        product_smiles: target product SMILES
        ec_class: EC class filter (e.g. "1" for oxidoreductases, "1.1" for alcohol dehydrogenases)
        top_k: number of results
        min_similarity: minimum Tanimoto similarity threshold
    """
    db = _load_db()
    if not db:
        return []

    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return []

    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    query_arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(query_fp, query_arr)

    # Filter by EC class if provided
    if ec_class:
        candidates = [r for r in db if r.ec.startswith(ec_class)]
    else:
        candidates = db

    if not candidates:
        candidates = db  # fallback to all if EC filter too strict

    # Compute Tanimoto similarities
    scored = []
    for r in candidates:
        # Fast Tanimoto via numpy
        intersection = np.sum(query_arr * r.product_fp)
        union = np.sum(query_arr) + np.sum(r.product_fp) - intersection
        sim = intersection / max(union, 1e-8)
        if sim >= min_similarity:
            scored.append((sim, r))

    scored.sort(key=lambda x: -x[0])

    results = []
    seen = set()
    for sim, r in scored[:top_k * 2]:
        # Deduplicate by reactant
        if r.reactant_smi in seen:
            continue
        seen.add(r.reactant_smi)
        results.append({
            "main_reactant": r.reactant_smi,
            "rxn_smiles": r.rxn_smiles,
            "ec": r.ec,
            "score": float(sim),
            "source": "v3_retrieval",
            "type": r.transformation,
            "T": r.T,
            "pH": r.pH,
            "enzyme_uid": r.evidence.get("uniprot_accession"),
            "catalyst": r.evidence.get("enzyme_name"),
            "cofactor": r.evidence.get("cofactor"),
            "evidence": dict(r.evidence, substrate_similarity=round(float(sim), 4)),
            "uniprot_accession": r.evidence.get("uniprot_accession"),
            "organism": r.evidence.get("organism"),
            "tax_id": r.evidence.get("tax_id"),
            "sequence": r.evidence.get("sequence"),
            "sequence_length": r.evidence.get("sequence_length"),
            "protein_existence": r.evidence.get("protein_existence"),
            "reviewed": r.evidence.get("reviewed"),
            "rhea_ids": r.evidence.get("rhea_ids"),
            "doi": r.evidence.get("doi"),
            "pmid": r.evidence.get("pmid"),
            "literature_title": r.evidence.get("literature_title"),
            "cascade_id": r.evidence.get("cascade_id"),
            "step_id": r.evidence.get("step_id"),
            "source_db": r.evidence.get("source_db"),
        })
        if len(results) >= top_k:
            break

    return results
