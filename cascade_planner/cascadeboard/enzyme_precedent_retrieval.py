"""Large enzyme precedent retrieval backed by the bridge pack reaction pool."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


DEFAULT_ENZYME_REACTION_POOL = Path("data/bridge_pack_v0/enzyme_reaction_pool.parquet")
DEFAULT_ENZYME_PRECEDENT_INDEX = Path("data/bridge_pack_v0/enzyme_precedent_index_v2.joblib")
_CACHE: dict[tuple[str, int, int], list["EnzymePrecedent"]] = {}
_BLACKLIST_CACHE: dict[tuple[str, int], set[str]] = {}

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class EnzymePrecedent:
    reaction_id: str
    substrate_smiles: str
    product_smiles: str
    reaction_smiles: str
    product_fp: Any
    product_main_smiles: str
    product_main_fp: Any
    substrate_component_selection: dict[str, Any]
    product_component_selection: dict[str, Any]
    ec_numbers: tuple[str, ...]
    occurrences: int
    source_counts: dict[str, Any]
    rhea_ids: tuple[str, ...]
    example_ids: tuple[str, ...]


def retrieve_enzyme_precedents(
    product_smiles: str,
    *,
    ec_class: str = "",
    top_k: int = 10,
    min_similarity: float | None = None,
    pool_path: str | Path | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve enzyme precedents whose product side resembles ``product_smiles``."""

    top_k = max(0, int(top_k or 0))
    if top_k <= 0:
        return []
    query = Chem.MolFromSmiles(str(product_smiles or ""))
    if query is None:
        return []
    min_similarity = _min_similarity() if min_similarity is None else float(min_similarity)
    path = _pool_path(pool_path)
    blacklist = load_component_blacklist(pool_path=path)
    precedents = load_enzyme_precedents(pool_path=path, max_rows=max_rows)
    if not precedents:
        return []
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query, 2, nBits=2048)
    ec_filter = str(ec_class or "").strip()
    scored: list[tuple[float, float, float, EnzymePrecedent]] = []
    filtered = [row for row in precedents if _ec_match(row.ec_numbers, ec_filter)] if ec_filter else precedents
    if not filtered and ec_filter and _env_truthy_default("AUTOPLANNER_ENZYME_PRECEDENT_EC_FALLBACK_ALL", True):
        filtered = precedents
    for row in filtered:
        sim = float(DataStructs.TanimotoSimilarity(query_fp, row.product_main_fp or row.product_fp))
        full_sim = float(DataStructs.TanimotoSimilarity(query_fp, row.product_fp))
        if sim >= min_similarity:
            rank_score = _rank_score(similarity=sim, full_similarity=full_sim, row=row, blacklist=blacklist)
            scored.append((rank_score, sim, full_sim, row))
    scored.sort(key=lambda item: (item[0], item[1], item[3].occurrences), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank_score, sim, full_sim, row in scored:
        key = row.substrate_smiles
        if key in seen:
            continue
        seen.add(key)
        ecs = list(row.ec_numbers)
        ec = _best_ec(ecs, ec_filter)
        substrate_selection = row.substrate_component_selection or select_main_component(row.substrate_smiles, blacklist=blacklist)
        product_selection = row.product_component_selection or select_main_component(row.product_smiles, blacklist=blacklist)
        transition = transition_signature(
            substrate_selection["main_smiles"],
            row.product_main_smiles,
            substrate_aux=substrate_selection.get("aux_smiles") or [],
            product_aux=product_selection.get("aux_smiles") or [],
        )
        proposal_score = _proposal_score(similarity=sim, transition_score=float(transition.get("transition_quality_score") or 0.0))
        out.append(
            {
                "main_reactant": substrate_selection["main_smiles"],
                "aux_reactants": substrate_selection["aux_smiles"],
                "rxn_smiles": f"{row.substrate_smiles}>>{product_smiles}",
                "source": "enzyme_precedent",
                "score": proposal_score,
                "ec": ec,
                "type": "enzyme_precedent_retrieval",
                "catalyst": "",
                "evidence": {
                    "source_db": "bridge_pack_v0.enzyme_reaction_pool",
                    "reaction_id": row.reaction_id,
                    "precedent_product_smiles": row.product_smiles,
                    "precedent_product_main_smiles": row.product_main_smiles,
                    "precedent_reaction_smiles": row.reaction_smiles,
                    "product_similarity": round(sim, 4),
                    "product_full_similarity": round(full_sim, 4),
                    "retrieval_rank_score": round(rank_score, 4),
                    "substrate_component_selection": substrate_selection,
                    "product_component_selection": product_selection,
                    "transition_signature": transition,
                    "ec_numbers": ecs,
                    "occurrences": row.occurrences,
                    "source_counts": row.source_counts,
                    "rhea_ids": list(row.rhea_ids),
                    "example_ids": list(row.example_ids[:8]),
                },
                "enzyme_ec_numbers": ecs,
                "rhea_ids": list(row.rhea_ids),
                "precedent_reaction_id": row.reaction_id,
                "precedent_product_similarity": round(sim, 4),
                "precedent_product_full_similarity": round(full_sim, 4),
                "precedent_rank_score": round(rank_score, 4),
            }
        )
        if len(out) >= top_k:
            break
    return out


def load_enzyme_precedents(
    *,
    pool_path: str | Path | None = None,
    max_rows: int | None = None,
) -> list[EnzymePrecedent]:
    path = _pool_path(pool_path)
    blacklist = load_component_blacklist(pool_path=path)
    limit = int(max_rows or _env_int("AUTOPLANNER_ENZYME_PRECEDENT_MAX_ROWS", 0) or 0)
    cache_key = (str(path.resolve()), limit, int(path.stat().st_mtime) if path.exists() else 0)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        _CACHE[cache_key] = []
        return []
    index_path = _index_path(path)
    if index_path is not None:
        indexed = _load_index(index_path, pool_path=path, max_rows=limit)
        if indexed is not None:
            _CACHE[cache_key] = indexed
            return indexed
    columns = [
        "reaction_id",
        "substrate_smiles",
        "product_smiles",
        "reaction_smiles",
        "occurrences",
        "source_counts_json",
        "ec_numbers_json",
        "rhea_ids_json",
        "example_ids_json",
    ]
    table = pq.read_table(path, columns=columns)
    rows = table.to_pylist()
    if limit > 0:
        rows = rows[:limit]
    precedents: list[EnzymePrecedent] = []
    for row in rows:
        product = str(row.get("product_smiles") or "")
        substrate = str(row.get("substrate_smiles") or "")
        if not product or not substrate:
            continue
        mol = Chem.MolFromSmiles(product)
        if mol is None:
            continue
        product_selection = select_main_component(product, blacklist=blacklist)
        substrate_selection = select_main_component(substrate, blacklist=blacklist)
        product_main = str(product_selection.get("main_smiles") or product)
        product_main_mol = Chem.MolFromSmiles(product_main)
        precedents.append(
            EnzymePrecedent(
                reaction_id=str(row.get("reaction_id") or ""),
                substrate_smiles=substrate,
                product_smiles=product,
                reaction_smiles=str(row.get("reaction_smiles") or f"{substrate}>>{product}"),
                product_fp=AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                product_main_smiles=product_main,
                product_main_fp=AllChem.GetMorganFingerprintAsBitVect(product_main_mol or mol, 2, nBits=2048),
                substrate_component_selection=substrate_selection,
                product_component_selection=product_selection,
                ec_numbers=tuple(_json_list(row.get("ec_numbers_json"))),
                occurrences=int(row.get("occurrences") or 0),
                source_counts=_json_dict(row.get("source_counts_json")),
                rhea_ids=tuple(_json_list(row.get("rhea_ids_json"))),
                example_ids=tuple(_json_list(row.get("example_ids_json"))),
            )
        )
    _CACHE[cache_key] = precedents
    if index_path is not None and _env_truthy_default("AUTOPLANNER_ENZYME_PRECEDENT_WRITE_INDEX", True):
        _write_index(index_path, pool_path=path, max_rows=limit, precedents=precedents)
    return precedents


def load_component_blacklist(*, pool_path: str | Path | None = None) -> set[str]:
    """Load common cofactors/metabolites used only for component role selection."""

    explicit = os.environ.get("AUTOPLANNER_ENZYME_PRECEDENT_COMPONENT_BLACKLIST")
    if explicit:
        path = Path(explicit)
    else:
        pool = _pool_path(pool_path)
        path = pool.parent / "cofactor_common_metabolite_blacklist.parquet"
    try:
        stat = path.stat()
    except OSError:
        return set()
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns))
    cached = _BLACKLIST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    values: set[str] = set()
    try:
        table = pq.read_table(path, columns=["canonical_smiles"])
    except Exception:
        _BLACKLIST_CACHE[cache_key] = values
        return values
    for row in table.to_pylist():
        can = _canonical(str(row.get("canonical_smiles") or ""))
        if can:
            values.add(can)
    _BLACKLIST_CACHE[cache_key] = values
    return values


def select_main_component(smiles: str, *, blacklist: set[str] | None = None) -> dict[str, Any]:
    """Choose the biosynthetic substrate/product component, not merely the largest molecule.

    Enzyme precedents often include ATP, CoA, NAD(P), phosphate, water, and other
    carrier-like participants.  These can be larger than the real organic
    substrate, so proposal generation should keep them as auxiliary evidence
    rather than hand them to downstream chemical retrosynthesis as the main
    precursor.
    """

    blacklist = blacklist or set()
    parts = [part for part in str(smiles or "").split(".") if part]
    if not parts:
        return {
            "main_smiles": "",
            "aux_smiles": [],
            "main_index": -1,
            "component_count": 0,
            "annotations": [],
            "selection_rule": "empty",
        }
    annotations = [_component_annotation(part, blacklist=blacklist, index=idx) for idx, part in enumerate(parts)]
    preferred = [
        item
        for item in annotations
        if not item["blacklisted"] and not item["carrier_like"] and int(item["heavy_atoms"]) > 1
    ]
    pool = preferred or [item for item in annotations if not item["blacklisted"]] or annotations
    selected = max(pool, key=lambda item: (int(item["carbon_atoms"]), int(item["heavy_atoms"]), -int(item["index"])))
    selected_key = selected["canonical_smiles"] or selected["smiles"]
    aux = [
        item["smiles"]
        for item in annotations
        if (item["canonical_smiles"] or item["smiles"]) != selected_key or int(item["index"]) != int(selected["index"])
    ]
    if preferred:
        rule = "prefer_noncarrier_nonblacklisted_component"
    elif pool is not annotations:
        rule = "fallback_nonblacklisted_component"
    else:
        rule = "fallback_largest_component"
    return {
        "main_smiles": selected["smiles"],
        "aux_smiles": aux,
        "main_index": int(selected["index"]),
        "component_count": len(parts),
        "annotations": annotations,
        "selection_rule": rule,
    }


def transition_signature(
    substrate_main: str,
    product_main: str,
    *,
    substrate_aux: list[str] | tuple[str, ...] = (),
    product_aux: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a lightweight, auditable substrate-product transition signature.

    This is deliberately not atom mapping.  It is a conservative evidence layer
    that records whether the main substrate/product pair has a plausible local
    change and whether element gains are explained by auxiliary participants.
    """

    substrate_mol = Chem.MolFromSmiles(str(substrate_main or ""))
    product_mol = Chem.MolFromSmiles(str(product_main or ""))
    if substrate_mol is None or product_mol is None:
        return {
            "schema_version": "enzyme_precedent_transition.v1",
            "valid": False,
            "substrate_main": str(substrate_main or ""),
            "product_main": str(product_main or ""),
            "transition_quality_score": 0.0,
            "transition_flags": ["invalid_transition_molecule"],
        }
    substrate_counts = _element_counts(substrate_mol)
    product_counts = _element_counts(product_mol)
    substrate_aux_counts = _element_counts_many(substrate_aux)
    product_aux_counts = _element_counts_many(product_aux)
    element_delta = {
        element: int(product_counts.get(element, 0) - substrate_counts.get(element, 0))
        for element in sorted(set(substrate_counts) | set(product_counts))
    }
    heavy_delta = int(product_mol.GetNumHeavyAtoms() - substrate_mol.GetNumHeavyAtoms())
    substrate_product_similarity = _mol_similarity(substrate_mol, product_mol)
    motif_delta = _motif_delta(substrate_mol, product_mol)
    flags: list[str] = []
    substrate_can = _canonical(str(substrate_main or ""))
    product_can = _canonical(str(product_main or ""))
    if substrate_can and substrate_can == product_can:
        flags.append("main_transition_self_loop")
    if substrate_product_similarity < 0.25:
        flags.append("weak_main_transition_similarity")
    if abs(heavy_delta) >= max(18, int(product_mol.GetNumHeavyAtoms() * 0.55)):
        flags.append("large_main_transition_delta_review")
    explained_gains: dict[str, int] = {}
    unexplained_gains: dict[str, int] = {}
    for element, delta in element_delta.items():
        if delta <= 0:
            continue
        available = int(substrate_aux_counts.get(element, 0) + product_aux_counts.get(element, 0))
        if available >= delta:
            explained_gains[element] = int(delta)
        else:
            unexplained_gains[element] = int(delta - available)
    if explained_gains:
        flags.append("auxiliary_explains_element_gain")
    review_unexplained = {element: count for element, count in unexplained_gains.items() if element != "H"}
    if review_unexplained:
        flags.append("unexplained_element_gain_review")
    transition_score = _transition_quality_score(
        substrate_product_similarity=substrate_product_similarity,
        heavy_delta=heavy_delta,
        product_heavy_atoms=int(product_mol.GetNumHeavyAtoms()),
        flags=flags,
    )
    return {
        "schema_version": "enzyme_precedent_transition.v1",
        "valid": True,
        "substrate_main": str(substrate_main or ""),
        "product_main": str(product_main or ""),
        "substrate_product_similarity": round(float(substrate_product_similarity), 4),
        "transition_quality_score": round(float(transition_score), 4),
        "transition_flags": flags,
        "heavy_atom_delta": heavy_delta,
        "element_delta": element_delta,
        "explained_element_gains": explained_gains,
        "unexplained_element_gains": unexplained_gains,
        "substrate_aux_element_counts": substrate_aux_counts,
        "product_aux_element_counts": product_aux_counts,
        "motif_delta": motif_delta,
    }


def _ec_match(ec_numbers: tuple[str, ...], ec_filter: str) -> bool:
    if not ec_filter:
        return True
    prefix = ec_filter.strip().rstrip(".")
    return any(str(ec).startswith(prefix) for ec in ec_numbers)


def _best_ec(ec_numbers: list[str], ec_filter: str) -> str:
    if ec_filter:
        for ec in ec_numbers:
            if str(ec).startswith(ec_filter):
                return str(ec)
    return str(ec_numbers[0]) if ec_numbers else ""


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _carbon_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)


def _component_annotation(smiles: str, *, blacklist: set[str], index: int) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        can = ""
        heavy_atoms = 0
        carbon_atoms = 0
    else:
        try:
            can = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            can = ""
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    carrier = _carrier_like_reasons(can or smiles)
    return {
        "index": int(index),
        "smiles": str(smiles or ""),
        "canonical_smiles": can,
        "heavy_atoms": heavy_atoms,
        "carbon_atoms": carbon_atoms,
        "blacklisted": bool(can and can in blacklist),
        "carrier_like": bool(carrier),
        "carrier_like_reasons": carrier,
    }


def _carrier_like_reasons(smiles: str) -> list[str]:
    text = str(smiles or "")
    reasons: list[str] = []
    phosphate_count = text.count("P(=O)")
    if phosphate_count >= 2:
        reasons.append("polyphosphate")
    if "ncnc" in text and phosphate_count >= 1:
        reasons.append("nucleotide_phosphate")
    if "NCCSC(=O)" in text or "NCCC(=O)NCCSC(=O)" in text:
        reasons.append("coa_thioester_motif")
    if "SCCNC(=O)CCNC(=O)" in text:
        reasons.append("coa_fragment")
    return sorted(set(reasons))


def _proposal_score(*, similarity: float, transition_score: float) -> float:
    sim = max(0.0, min(1.0, float(similarity or 0.0)))
    transition = max(0.0, min(1.0, float(transition_score or 0.0)))
    if transition <= 0.0:
        return sim
    return round(max(0.0, min(1.0, 0.85 * sim + 0.15 * transition)), 6)


def _transition_quality_score(
    *,
    substrate_product_similarity: float,
    heavy_delta: int,
    product_heavy_atoms: int,
    flags: list[str],
) -> float:
    score = 0.35 + 0.55 * max(0.0, min(1.0, float(substrate_product_similarity or 0.0)))
    if "auxiliary_explains_element_gain" in flags:
        score += 0.08
    if "main_transition_self_loop" in flags:
        score -= 0.35
    if "weak_main_transition_similarity" in flags:
        score -= 0.20
    if "large_main_transition_delta_review" in flags:
        score -= 0.15
    if "unexplained_element_gain_review" in flags:
        score -= 0.10
    if abs(int(heavy_delta or 0)) <= max(2, int(product_heavy_atoms * 0.15)):
        score += 0.04
    return max(0.0, min(1.0, score))


def _mol_similarity(left: Any, right: Any) -> float:
    try:
        left_fp = AllChem.GetMorganFingerprintAsBitVect(left, 2, nBits=2048)
        right_fp = AllChem.GetMorganFingerprintAsBitVect(right, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))
    except Exception:
        return 0.0


def _element_counts(mol: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        symbol = str(atom.GetSymbol())
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _element_counts_many(smiles_list: list[str] | tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for smiles in smiles_list or ():
        mol = Chem.MolFromSmiles(str(smiles or ""))
        if mol is None:
            continue
        for element, value in _element_counts(mol).items():
            counts[element] = counts.get(element, 0) + int(value)
    return counts


def _motif_delta(substrate_mol: Any, product_mol: Any) -> dict[str, int]:
    substrate = _motif_counts(substrate_mol)
    product = _motif_counts(product_mol)
    return {
        name: int(product.get(name, 0) - substrate.get(name, 0))
        for name in sorted(set(substrate) | set(product))
    }


def _motif_counts(mol: Any) -> dict[str, int]:
    if mol is None:
        return {}
    patterns = _motif_patterns()
    out: dict[str, int] = {}
    for name, pattern in patterns.items():
        try:
            out[name] = len(mol.GetSubstructMatches(pattern))
        except Exception:
            out[name] = 0
    return out


def _motif_patterns() -> dict[str, Any]:
    if not hasattr(_motif_patterns, "_cache"):
        setattr(
            _motif_patterns,
            "_cache",
            {
                "carbonyl": Chem.MolFromSmarts("[CX3]=[OX1]"),
                "carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]"),
                "ester": Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"),
                "hydroxyl": Chem.MolFromSmarts("[OX2H]"),
                "amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
                "phosphate": Chem.MolFromSmarts("[PX4](=O)([O])[O]"),
                "thioester": Chem.MolFromSmarts("[CX3](=O)S[#6]"),
            },
        )
    return getattr(_motif_patterns, "_cache")


def _rank_score(*, similarity: float, full_similarity: float, row: EnzymePrecedent, blacklist: set[str]) -> float:
    product_selection = row.product_component_selection or select_main_component(row.product_smiles, blacklist=blacklist)
    substrate_selection = row.substrate_component_selection or select_main_component(row.substrate_smiles, blacklist=blacklist)
    score = float(similarity)
    if full_similarity < similarity:
        score += min(0.05, (similarity - full_similarity) * 0.05)
    if product_selection["selection_rule"].startswith("prefer_noncarrier"):
        score += 0.03
    if substrate_selection["selection_rule"].startswith("prefer_noncarrier"):
        score += 0.03
    if substrate_selection["annotations"]:
        main_idx = int(substrate_selection["main_index"])
        main = substrate_selection["annotations"][main_idx] if 0 <= main_idx < len(substrate_selection["annotations"]) else {}
        if main.get("blacklisted") or main.get("carrier_like"):
            score -= 0.20
    if row.occurrences > 0:
        score += min(0.04, float(np.log10(float(row.occurrences) + 1.0)) * 0.01)
    return score


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "")]
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item or "")]


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _min_similarity() -> float:
    try:
        return float(os.environ.get("AUTOPLANNER_ENZYME_PRECEDENT_MIN_SIMILARITY") or 0.2)
    except ValueError:
        return 0.2


def _pool_path(pool_path: str | Path | None) -> Path:
    if pool_path is not None:
        return Path(pool_path)
    raw = os.environ.get("AUTOPLANNER_ENZYME_PRECEDENT_POOL_PATH")
    return Path(raw) if raw else DEFAULT_ENZYME_REACTION_POOL


def _index_path(pool_path: Path) -> Path | None:
    raw = os.environ.get("AUTOPLANNER_ENZYME_PRECEDENT_INDEX_PATH")
    if raw:
        return Path(raw)
    if pool_path == DEFAULT_ENZYME_REACTION_POOL:
        return DEFAULT_ENZYME_PRECEDENT_INDEX
    if _env_truthy_default("AUTOPLANNER_ENZYME_PRECEDENT_INDEX_TMP_POOLS", False):
        return pool_path.with_suffix(".enzyme_precedent_index_v2.joblib")
    return None


def _load_index(index_path: Path, *, pool_path: Path, max_rows: int) -> list[EnzymePrecedent] | None:
    if not index_path.exists():
        return None
    try:
        payload = joblib.load(index_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata") or {}
    try:
        stat = pool_path.stat()
    except OSError:
        return None
    if metadata.get("pool_path") != str(pool_path.resolve()):
        return None
    if int(metadata.get("pool_size") or -1) != int(stat.st_size):
        return None
    if int(metadata.get("pool_mtime_ns") or -1) != int(stat.st_mtime_ns):
        return None
    if int(metadata.get("max_rows") or 0) != int(max_rows or 0):
        return None
    rows = payload.get("precedents")
    return rows if isinstance(rows, list) else None


def _write_index(index_path: Path, *, pool_path: Path, max_rows: int, precedents: list[EnzymePrecedent]) -> None:
    try:
        stat = pool_path.stat()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": "enzyme_precedent_index.v2",
                "metadata": {
                    "pool_path": str(pool_path.resolve()),
                    "pool_size": int(stat.st_size),
                    "pool_mtime_ns": int(stat.st_mtime_ns),
                    "max_rows": int(max_rows or 0),
                    "precedents": len(precedents),
                },
                "precedents": precedents,
            },
            index_path,
            compress=3,
        )
    except Exception:
        return


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _env_truthy_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}
