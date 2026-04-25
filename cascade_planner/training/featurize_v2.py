"""Featurization v2: DRFP for rxn + numeric/solvent features + step-pair concat."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import AllChem

_RXN_CACHE: dict[tuple[str, int], np.ndarray] = {}
_MOL_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


# ---------------- DRFP ----------------

def drfp_one(rxn: str, n_bits: int = 2048) -> np.ndarray:
    key = (rxn, n_bits)
    if key in _RXN_CACHE:
        return _RXN_CACHE[key]
    arr = np.asarray(DrfpEncoder.encode([rxn], n_folded_length=n_bits)[0], dtype=np.float32)
    _RXN_CACHE[key] = arr
    return arr


def drfp_batch(rxns: Iterable[str], n_bits: int = 2048) -> np.ndarray:
    out = np.zeros((len(rxns), n_bits), dtype=np.float32)
    for i, r in enumerate(rxns):
        if r:
            out[i] = drfp_one(r, n_bits)
    return out


# ---------------- Morgan FP for small molecules ----------------

def mol_fp(smi: str | None, n_bits: int = 256, radius: int = 2) -> np.ndarray:
    if not smi:
        return np.zeros(n_bits, dtype=np.float32)
    key = (smi, n_bits, radius)
    if key in _MOL_CACHE:
        return _MOL_CACHE[key]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        arr = np.zeros(n_bits, dtype=np.float32)
    else:
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.array(bv, dtype=np.float32)
    _MOL_CACHE[key] = arr
    return arr


# ---------------- numeric condition features ----------------

def numeric_condition(T: float | None, pH: float | None) -> np.ndarray:
    """Return [T, pH, T_missing_mask, pH_missing_mask]; values centered to typical range."""
    out = np.zeros(4, dtype=np.float32)
    if T is None:
        out[2] = 1.0
        out[0] = 0.0
    else:
        out[0] = (float(T) - 25.0) / 30.0  # ~ [-1,1] for 0..55C
    if pH is None:
        out[3] = 1.0
        out[1] = 0.0
    else:
        out[1] = (float(pH) - 7.0) / 3.0  # ~ [-1,1] for 4..10
    return out


def step_features(rxn: str, T: float | None, pH: float | None, solv: str | None,
                  rxn_bits: int = 2048, solv_bits: int = 128) -> np.ndarray:
    return np.concatenate([
        drfp_one(rxn, rxn_bits),
        numeric_condition(T, pH),
        mol_fp(solv, n_bits=solv_bits, radius=2),
    ])


def pair_features(rxn_a: str, rxn_b: str,
                  T_a, ph_a, solv_a, T_b, ph_b, solv_b,
                  rxn_bits: int = 2048, solv_bits: int = 128) -> np.ndarray:
    fa = drfp_one(rxn_a, rxn_bits)
    fb = drfp_one(rxn_b, rxn_bits)
    diff = np.abs(fa - fb)
    return np.concatenate([
        fa, fb, diff,
        numeric_condition(T_a, ph_a),
        numeric_condition(T_b, ph_b),
        mol_fp(solv_a, n_bits=solv_bits),
        mol_fp(solv_b, n_bits=solv_bits),
    ])


def cascade_mean_features(rxns: list[str], avg_T, avg_pH, first_solv,
                          rxn_bits: int = 2048, solv_bits: int = 128) -> np.ndarray:
    valid = [r for r in rxns if r]
    if valid:
        rxn_part = np.stack([drfp_one(r, rxn_bits) for r in valid]).mean(axis=0)
    else:
        rxn_part = np.zeros(rxn_bits, dtype=np.float32)
    return np.concatenate([
        rxn_part,
        numeric_condition(avg_T, avg_pH),
        mol_fp(first_solv, n_bits=solv_bits),
    ])
