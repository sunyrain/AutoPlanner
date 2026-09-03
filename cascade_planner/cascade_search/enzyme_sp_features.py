"""Dependency-light feature projection for the enzyme SP verifier.

The runtime scorer must not import the training CLI because that module also
loads optional LightGBM and Parquet dependencies.  Keeping the feature schema
here makes inference usable anywhere the serialized estimator itself can run.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse


RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
SUBSTRATE_OFFSET = 0
PRODUCT_OFFSET = FP_BITS
SHARED_OFFSET = FP_BITS * 2
NUMERIC_OFFSET = FP_BITS * 3

EC1_VALUES = ["1", "2", "3", "4", "5", "6", "7", "unknown"]
NUMERIC_FEATURES = [
    "substrate_total_heavy",
    "product_total_heavy",
    "substrate_largest_heavy",
    "product_largest_heavy",
    "heavy_abs_delta",
    "heavy_signed_delta",
    "heavy_ratio_min_over_max",
    "substrate_component_count",
    "product_component_count",
    "component_abs_delta",
    "substrate_ring_count",
    "product_ring_count",
    "ring_abs_delta",
    "substrate_hetero_count",
    "product_hetero_count",
    "hetero_abs_delta",
    "substrate_product_tanimoto",
    "shared_bit_count",
    "substrate_only_bit_count",
    "product_only_bit_count",
    "ec_count",
    "ec_known",
]
EC_FEATURES = [f"ec1={value}" for value in EC1_VALUES] + ["ec1=other"]
FEATURE_NAMES = (
    [f"substrate_morgan_{index}" for index in range(FP_BITS)]
    + [f"product_morgan_{index}" for index in range(FP_BITS)]
    + [f"shared_morgan_{index}" for index in range(FP_BITS)]
    + NUMERIC_FEATURES
    + EC_FEATURES
)


def parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    return []


class SideFeatureCache:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}
        self.invalid: Counter[str] = Counter()

    def get(self, smiles: str) -> dict[str, Any]:
        smiles = str(smiles or "")
        if smiles in self.cache:
            return self.cache[smiles]
        bits: set[int] = set()
        total_heavy = 0
        largest_heavy = 0
        component_count = 0
        rings = 0
        hetero = 0
        for part in [item.strip() for item in smiles.split(".") if item.strip()]:
            molecule = Chem.MolFromSmiles(part)
            if molecule is None:
                self.invalid["mol_from_smiles_failed"] += 1
                continue
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                2,
                nBits=FP_BITS,
            )
            bits.update(int(bit) for bit in fingerprint.GetOnBits())
            heavy = int(molecule.GetNumHeavyAtoms())
            total_heavy += heavy
            largest_heavy = max(largest_heavy, heavy)
            component_count += 1
            rings += int(molecule.GetRingInfo().NumRings())
            hetero += sum(
                1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() not in (1, 6)
            )
        info = {
            "bits": sorted(bits),
            "bit_set": bits,
            "total_heavy": total_heavy,
            "largest_heavy": largest_heavy,
            "component_count": component_count,
            "rings": rings,
            "hetero": hetero,
        }
        self.cache[smiles] = info
        return info


def bit_tanimoto(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    union = len(left | right)
    return float(len(left & right) / union) if union else 0.0


def numeric_values(
    row: dict[str, Any],
    substrate: dict[str, Any],
    product: dict[str, Any],
) -> list[float]:
    substrate_bits = substrate["bit_set"]
    product_bits = product["bit_set"]
    shared = substrate_bits & product_bits
    substrate_only = substrate_bits - product_bits
    product_only = product_bits - substrate_bits
    substrate_heavy = float(substrate["total_heavy"])
    product_heavy = float(product["total_heavy"])
    max_heavy = max(substrate_heavy, product_heavy, 1.0)
    min_heavy = min(substrate_heavy, product_heavy)
    ec_count = float(
        row.get("ec_count")
        or len(set(parse_json_list(row.get("ec_numbers_json"))))
    )
    ec_known = str(row.get("ec1") or "unknown") != "unknown"
    return [
        substrate_heavy,
        product_heavy,
        float(substrate["largest_heavy"]),
        float(product["largest_heavy"]),
        abs(product_heavy - substrate_heavy),
        product_heavy - substrate_heavy,
        min_heavy / max_heavy,
        float(substrate["component_count"]),
        float(product["component_count"]),
        abs(float(product["component_count"]) - float(substrate["component_count"])),
        float(substrate["rings"]),
        float(product["rings"]),
        abs(float(product["rings"]) - float(substrate["rings"])),
        float(substrate["hetero"]),
        float(product["hetero"]),
        abs(float(product["hetero"]) - float(substrate["hetero"])),
        bit_tanimoto(substrate_bits, product_bits),
        float(len(shared)),
        float(len(substrate_only)),
        float(len(product_only)),
        ec_count,
        1.0 if ec_known else 0.0,
    ]


def build_matrix(
    rows: list[dict[str, Any]],
    cache: SideFeatureCache,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[str], list[str]]:
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    labels = np.zeros(len(rows), dtype=np.int8)
    weights = np.ones(len(rows), dtype=np.float32)
    label_types: list[str] = []
    row_ids: list[str] = []

    for row_index, row in enumerate(rows):
        substrate = cache.get(row.get("substrate_smiles") or "")
        product = cache.get(row.get("product_smiles") or "")
        labels[row_index] = int(row.get("label") or 0)
        weights[row_index] = float(row.get("label_weight") or 1.0)
        label_types.append(str(row.get("label_type") or "unknown"))
        row_ids.append(str(row.get("row_id") or row_index))

        substrate_bits = substrate["bits"]
        product_bits = product["bits"]
        product_set = product["bit_set"]
        for bit in substrate_bits:
            row_indices.append(row_index)
            column_indices.append(SUBSTRATE_OFFSET + bit)
            data.append(1.0)
        for bit in product_bits:
            row_indices.append(row_index)
            column_indices.append(PRODUCT_OFFSET + bit)
            data.append(1.0)
        for bit in substrate_bits:
            if bit in product_set:
                row_indices.append(row_index)
                column_indices.append(SHARED_OFFSET + bit)
                data.append(1.0)
        for feature_index, value in enumerate(
            numeric_values(row, substrate, product)
        ):
            if value != 0.0:
                row_indices.append(row_index)
                column_indices.append(NUMERIC_OFFSET + feature_index)
                data.append(float(value))

        ec1 = str(row.get("ec1") or "unknown")
        try:
            ec_index = EC1_VALUES.index(ec1)
        except ValueError:
            ec_index = len(EC1_VALUES)
        row_indices.append(row_index)
        column_indices.append(NUMERIC_OFFSET + len(NUMERIC_FEATURES) + ec_index)
        data.append(1.0)

    matrix = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(rows), len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    return matrix, labels, weights, label_types, row_ids
