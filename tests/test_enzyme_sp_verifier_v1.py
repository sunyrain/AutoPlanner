import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_enzyme_sp_verifier_v1_pack import (
    SideCache,
    build_negative_rows,
    build_positive_rows,
    split_name,
)
from scripts.train_enzyme_sp_verifier_v1 import SideFeatureCache, build_matrix, select_threshold


def test_build_enzyme_sp_pack_constructs_same_ec_hard_negatives(tmp_path):
    input_dir = tmp_path / "bridge_pack"
    input_dir.mkdir()
    rows = [
        _rxn("r1", "CCO", "CC=O", ["1.1.1.1"]),
        _rxn("r2", "CCCO", "CCC=O", ["1.1.1.2"]),
        _rxn("r3", "c1ccccc1O", "Oc1ccccc1O", ["1.14.13.1"]),
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_dir / "enzyme_reaction_pool.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"canonical_smiles": "O", "inchikey": "water", "heavy_atoms": 1},
                {"canonical_smiles": "O=P(O)(O)O", "inchikey": "phosphate", "heavy_atoms": 5},
            ]
        ),
        input_dir / "cofactor_common_metabolite_blacklist.parquet",
    )

    cache = SideCache()
    positives, pos_counts = build_positive_rows(
        rows,
        cache=cache,
        max_positives=None,
        min_largest_heavy=2,
        max_total_heavy=80,
        max_components=4,
    )
    negatives, neg_counts = build_negative_rows(
        positives,
        root=input_dir,
        cache=cache,
        negatives_per_positive=2,
        max_negatives=8,
        seed=1,
    )

    assert len(positives) == 3
    assert pos_counts["positive_rows"] == 3
    assert negatives
    assert any(row["label_type"].startswith("same_ec_wrong") for row in negatives)
    assert all(row["label"] == 0 for row in negatives)
    assert split_name("stable-key") == split_name("stable-key")


def test_train_feature_matrix_and_threshold_selection_are_stable():
    rows = [
        {
            "row_id": "p1",
            "substrate_smiles": "CCO",
            "product_smiles": "CC=O",
            "ec_numbers_json": json.dumps(["1.1.1.1"]),
            "ec1": "1",
            "ec_count": 1,
            "label": 1,
            "label_type": "enzyme_reaction_positive",
            "label_weight": 1.0,
        },
        {
            "row_id": "n1",
            "substrate_smiles": "CCO",
            "product_smiles": "c1ccccc1",
            "ec_numbers_json": json.dumps(["1.1.1.1"]),
            "ec1": "1",
            "ec_count": 1,
            "label": 0,
            "label_type": "same_ec_wrong_product",
            "label_weight": 1.0,
        },
    ]
    matrix, labels, weights, label_types, row_ids = build_matrix(rows, SideFeatureCache())
    threshold = select_threshold(labels, labels.astype(float), target_precision=0.9)

    assert matrix.shape[0] == 2
    assert labels.tolist() == [1, 0]
    assert weights.tolist() == [1.0, 1.0]
    assert label_types == ["enzyme_reaction_positive", "same_ec_wrong_product"]
    assert row_ids == ["p1", "n1"]
    assert threshold["threshold"] >= 0.0


def _rxn(reaction_id, substrate, product, ecs):
    return {
        "reaction_id": reaction_id,
        "substrate_key": f"{reaction_id}:s",
        "product_key": f"{reaction_id}:p",
        "substrate_smiles": substrate,
        "product_smiles": product,
        "reaction_smiles": f"{substrate}>>{product}",
        "occurrences": 1,
        "source_counts_json": json.dumps({"fixture": 1}),
        "ec_numbers_json": json.dumps(ecs),
        "ec_unique": len(set(ecs)),
        "rhea_ids_json": "[]",
        "rhea_unique": 0,
        "example_ids_json": "[]",
    }
