"""Feature builders shared by vNext training and optional runtime inference."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles
from cascade_planner.cascadeboard.value_function import candidate_value_features
from cascade_planner.vnext.schema import (
    BOTTLENECK_LABELS,
    CANDIDATE_METADATA_FEATURES,
    CANDIDATE_NUMERIC_FEATURES,
    OPERATION_MODE_VALUES,
    ROUTE_FEATURE_NAMES,
    SOURCE_BUDGET_GROUPS,
    SOURCE_VALUES,
)


RDLogger.DisableLog("rdApp.*")

StockChecker = Callable[[str], bool]


def read_jsonl(path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(__import__("json").loads(line))
    return rows


def write_jsonl(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def morgan_fp(smiles: str | None, *, n_bits: int = 256) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def candidate_reactants(candidate: dict[str, Any]) -> list[str]:
    reactants: list[str] = []
    main = candidate.get("main_reactant")
    if main:
        reactants.append(str(main))
    reactants.extend(str(smi) for smi in candidate.get("aux_reactants") or [] if smi)
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if rxn and ">>" in rxn:
        lhs = rxn.split(">>", 1)[0]
        reactants.extend(part.strip() for part in lhs.split(".") if part.strip())
    out: list[str] = []
    seen: set[str] = set()
    for smi in reactants:
        key = canonical_smiles(smi) or smi
        if key in seen:
            continue
        seen.add(key)
        out.append(smi)
    return out


def candidate_reactant_smiles(candidate: dict[str, Any]) -> str:
    return ".".join(candidate_reactants(candidate))


def candidate_feature_dim(n_bits: int = 256) -> int:
    return n_bits * 2 + len(CANDIDATE_NUMERIC_FEATURES) + len(CANDIDATE_METADATA_FEATURES) + len(SOURCE_VALUES)


def route_feature_dim() -> int:
    return len(ROUTE_FEATURE_NAMES) + 7


def node_feature_dim(n_bits: int = 256) -> int:
    return n_bits * 2 + 16


def step_token_dim() -> int:
    return 48


def candidate_feature_vector(
    product: str,
    candidate: dict[str, Any],
    *,
    rank: int | float | None = None,
    gt_available: bool = False,
    n_bits: int = 256,
    stock_checker: StockChecker | None = None,
) -> np.ndarray:
    product_fp = morgan_fp(product, n_bits=n_bits)
    reactant_fp = morgan_fp(candidate_reactant_smiles(candidate), n_bits=n_bits)
    exported = candidate_value_features(product, candidate, stock_checker=stock_checker)
    numeric = [float(exported.get(name) or 0.0) for name in CANDIDATE_NUMERIC_FEATURES]
    metadata = candidate_metadata_features(candidate, rank=rank, gt_available=gt_available)
    source = str(candidate.get("source") or candidate.get("enzyme_source") or "unknown").lower()
    source_features = [1.0 if source == value else 0.0 for value in SOURCE_VALUES]
    return np.concatenate([
        product_fp,
        reactant_fp,
        np.asarray(numeric + metadata + source_features, dtype=np.float32),
    ]).astype(np.float32)


def candidate_metadata_features(
    candidate: dict[str, Any],
    *,
    rank: int | float | None = None,
    gt_available: bool = False,
) -> list[float]:
    evidence = candidate.get("evidence") or {}
    t_value = safe_float(candidate.get("T"))
    ph_value = safe_float(candidate.get("pH"))
    rank_value = float(rank if rank not in (None, "") else candidate.get("rank") or 1.0)
    values = {
        "rank_inverse": 1.0 / max(rank_value, 1.0),
        "rank_log": math.log1p(max(rank_value, 0.0)) / 5.0,
        "has_gt": float(bool(gt_available)),
        "has_ec": float(bool(candidate.get("ec"))),
        "has_type": float(bool(candidate.get("type") or candidate.get("reaction_type"))),
        "has_doi": float(bool(candidate.get("doi") or evidence.get("doi"))),
        "has_uniprot": float(bool(candidate.get("uniprot_accession") or evidence.get("uniprot_accession"))),
        "has_T": float(t_value is not None),
        "has_pH": float(ph_value is not None),
        "has_T_and_pH": float(t_value is not None and ph_value is not None),
        "T_scaled": float(t_value or 0.0) / 100.0,
        "pH_scaled": float(ph_value or 0.0) / 14.0,
        "has_solvent": float(bool(candidate.get("solvent"))),
        "has_catalyst": float(bool(candidate.get("catalyst"))),
        "has_enzyme_uid": float(bool(candidate.get("enzyme_uid"))),
        "has_cofactor": float(bool(candidate.get("cofactor") or evidence.get("cofactor"))),
        "has_condition_match": float(bool(candidate.get("condition_match") or evidence.get("condition_match"))),
    }
    return [float(values.get(name, 0.0)) for name in CANDIDATE_METADATA_FEATURES]


def route_feature_vector(row: dict[str, Any]) -> np.ndarray:
    features = row.get("features") or {}
    metrics = row.get("metrics_summary") or {}
    values = [float(features.get(name) or 0.0) for name in ROUTE_FEATURE_NAMES]
    n_steps = float(row.get("n_steps") or row.get("depth") or len(row.get("type_sequence") or []))
    bottlenecks = set(row.get("recovery_bottleneck_labels") or row.get("bottleneck_labels") or [])
    opmode = str(row.get("operation_mode") or "unknown")
    values.extend([
        min(n_steps, 12.0) / 12.0,
        float(row.get("score") or 0.0) / 100.0,
        float(row.get("confidence") or 0.0),
        float(bool(row.get("doi"))),
        float(bool(row.get("cascade_id"))),
        float(bool(bottlenecks)),
        _one_hot_index(opmode, OPERATION_MODE_VALUES) / max(len(OPERATION_MODE_VALUES) - 1, 1),
    ])
    # Keep a stable length even if old route rows omit newer metrics.
    return np.asarray(values[:route_feature_dim()], dtype=np.float32)


def route_step_tokens(row: dict[str, Any], *, max_steps: int = 8) -> tuple[np.ndarray, np.ndarray]:
    type_seq = list(row.get("type_sequence") or [])
    ec_seq = list(row.get("ec1_sequence") or [])
    src_seq = list(row.get("source_sequence") or [])
    tokens = np.zeros((max_steps, step_token_dim()), dtype=np.float32)
    mask = np.zeros(max_steps, dtype=np.float32)
    for idx in range(min(max_steps, max(len(type_seq), len(ec_seq), len(src_seq)))):
        typ = str(type_seq[idx] if idx < len(type_seq) else "")
        ec1 = str(ec_seq[idx] if idx < len(ec_seq) else "")
        src = str(src_seq[idx] if idx < len(src_seq) else "unknown")
        token = tokens[idx]
        token[0] = idx / max(max_steps - 1, 1)
        token[1] = _safe_ec1(ec1) / 7.0
        token[2 + stable_bucket(typ, 14)] = 1.0
        token[16 + stable_bucket(src.lower(), 12)] = 1.0
        token[28 + stable_bucket(ec1, 12)] = 1.0
        token[40] = float(bool(typ))
        token[41] = float(bool(ec1))
        token[42] = float(_safe_ec1(ec1) > 0)
        mask[idx] = 1.0
    return tokens, mask


def route_label_vector(row: dict[str, Any]) -> dict[str, np.ndarray | float]:
    features = row.get("features") or {}
    labels = set(row.get("recovery_bottleneck_labels") or row.get("bottleneck_labels") or [])
    metrics = row.get("metrics_summary") or {}
    solved = float(row.get("label", 0.0) >= 1.0 or row.get("label_type") == "professional_solved")
    progressive = float(bool(features.get("progressive_route") or metrics.get("progressive_route")))
    stock_closed = float((features.get("strict_stock_solve") or 0.0) > 0.0)
    compatibility = float((features.get("compatibility_success") or 0.0) > 0.0)
    value = _route_value_target(row, solved=solved, stock_closed=stock_closed, progressive=progressive, compatibility=compatibility)
    return {
        "value": value,
        "solved": solved,
        "progressive": progressive,
        "stock_closed": stock_closed,
        "compatibility": compatibility,
        "bottlenecks": np.asarray([float(label in labels) for label in BOTTLENECK_LABELS], dtype=np.float32),
    }


def open_leaf_feature_matrix(
    *,
    target: str,
    open_leaves: list[str] | tuple[str, ...],
    depth: int = 0,
    expanded: set[str] | list[str] | tuple[str, ...] | None = None,
    parent_reactants: set[str] | list[str] | tuple[str, ...] | None = None,
    max_open_leaves: int = 8,
    n_bits: int = 256,
    stock_checker: StockChecker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((max_open_leaves, node_feature_dim(n_bits)), dtype=np.float32)
    mask = np.zeros(max_open_leaves, dtype=np.float32)
    target_fp = morgan_fp(target, n_bits=n_bits)
    target_atoms = _heavy_atoms(target)
    expanded_set = {canonical_smiles(smi) or smi for smi in (expanded or []) if smi}
    parent_set = {canonical_smiles(smi) or smi for smi in (parent_reactants or []) if smi}
    total_open = max(len(open_leaves), 1)
    for idx, leaf in enumerate(list(open_leaves)[:max_open_leaves]):
        leaf_can = canonical_smiles(leaf) or leaf
        leaf_fp = morgan_fp(leaf, n_bits=n_bits)
        leaf_atoms = _heavy_atoms(leaf)
        stock_hit = 0.0
        if stock_checker is not None and leaf:
            try:
                stock_hit = float(bool(stock_checker(leaf)))
            except Exception:
                stock_hit = 0.0
        mol = Chem.MolFromSmiles(leaf or "")
        hetero = 0
        ring_count = 0
        aromatic = 0.0
        if mol is not None:
            hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"})
            ring_count = int(mol.GetRingInfo().NumRings())
            aromatic = float(any(atom.GetIsAromatic() for atom in mol.GetAtoms()))
        denom = float(np.maximum(target_fp.sum() + leaf_fp.sum() - np.minimum(target_fp, leaf_fp).sum(), 1.0))
        tanimoto = float(np.minimum(target_fp, leaf_fp).sum() / denom)
        numeric = np.asarray(
            [
                min(float(depth), 12.0) / 12.0,
                idx / max(max_open_leaves - 1, 1),
                leaf_atoms / max(target_atoms, 1),
                max(0.0, (target_atoms - leaf_atoms) / max(target_atoms, 1)),
                stock_hit,
                float(leaf_can in expanded_set),
                total_open / max(max_open_leaves, 1),
                float(_oxygen_rich(leaf)),
                float(leaf_can in parent_set),
                float(depth <= 0),
                float(stock_hit <= 0.0),
                float(leaf_atoms <= 6),
                aromatic,
                min(float(ring_count), 6.0) / 6.0,
                hetero / max(float(leaf_atoms), 1.0),
                tanimoto,
            ],
            dtype=np.float32,
        )
        features[idx] = np.concatenate([target_fp, leaf_fp, numeric]).astype(np.float32)
        mask[idx] = 1.0
    return features, mask


def source_budget_vector(candidates: list[dict[str, Any]]) -> np.ndarray:
    counts = {name: 0.0 for name in SOURCE_BUDGET_GROUPS}
    for item in candidates:
        candidate = (item or {}).get("candidate") if isinstance(item, dict) and "candidate" in item else item
        source = str((candidate or {}).get("source") or (candidate or {}).get("enzyme_source") or "").lower()
        counts[_source_budget_group(source)] += 1.0
    total = sum(counts.values())
    if total <= 0:
        counts["fallback"] = 1.0
        total = 1.0
    return np.asarray([counts[name] / total for name in SOURCE_BUDGET_GROUPS], dtype=np.float32)


def _route_value_target(
    row: dict[str, Any],
    *,
    solved: float,
    stock_closed: float,
    progressive: float,
    compatibility: float,
) -> float:
    explicit = safe_float(row.get("value_target"))
    if explicit is None:
        explicit = safe_float(row.get("label"))
    if explicit is not None:
        return float(np.clip(explicit, 0.0, 1.0))
    features = row.get("features") or {}
    filled = float(bool(features.get("filled_route")))
    utility = 0.45 * solved + 0.2 * stock_closed + 0.2 * progressive + 0.1 * compatibility + 0.05 * filled
    return float(np.clip(utility, 0.0, 1.0))


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def stable_bucket(value: str, n: int) -> int:
    if not value:
        return 0
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(n, 1)


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    if "rxn_smiles" not in out and out.get("reaction_smiles"):
        out["rxn_smiles"] = out["reaction_smiles"]
    if "type" not in out and out.get("reaction_type"):
        out["type"] = out["reaction_type"]
    if not out.get("reactants"):
        out["reactants"] = candidate_reactants(out)
    out["canonical_reaction"] = canonical_reaction(out.get("rxn_smiles") or out.get("reaction_smiles") or "")
    return out


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_ec1(value: Any) -> int:
    try:
        return int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _one_hot_index(value: str, vocab: list[str]) -> int:
    value = value if value in vocab else "unknown"
    try:
        return vocab.index(value)
    except ValueError:
        return len(vocab) - 1


def _source_budget_group(source: str) -> str:
    if source in {
        "retrochimera",
        "chemtemplates",
        "chemical",
        "template",
        "uspto",
        "uspto50k",
        "chem_enzy_onestep",
        "chem_enzy_graphfp",
        "chem_enzy_onmt",
    }:
        return "chemical"
    if source in {"enzyformer", "enzexpand", "v3_retrieval", "enzymatic", "enzyme"}:
        return "enzymatic"
    if source in {"retrorules", "rhea", "rhea_template", "retrorules_template"}:
        return "rhea_retrorules"
    return "fallback"


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _oxygen_rich(smiles: str | None) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    heavy = mol.GetNumHeavyAtoms()
    oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    return bool(oxygen >= 5 and oxygen / max(heavy, 1) >= 0.40)
