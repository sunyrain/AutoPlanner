import json

import pyarrow as pa
import pyarrow.parquet as pq

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0, inchikey_from_smiles


def test_bridge_retriever_returns_exact_before_similarity(tmp_path):
    exact_smiles = "CCO"
    exact_key = inchikey_from_smiles(exact_smiles)
    similarity_smiles = "CCCO"
    similarity_key = inchikey_from_smiles(similarity_smiles)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "inchikey": exact_key,
                    "canonical_smiles": exact_smiles,
                    "bridge_direction": "chemical_product_to_enzyme_substrate",
                    "confidence_tier": "tier1_strict_exact_substrate_bridge",
                    "chemical_occurrences": 1,
                    "enzyme_occurrences": 2,
                    "enzyme_ec_sample_json": json.dumps(["1.1.1.1"]),
                    "enzyme_ec_unique": 1,
                    "bridge_flags_json": "[]",
                }
            ]
        ),
        tmp_path / "exact_bridge_strict.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chemical_inchikey": exact_key,
                    "enzyme_inchikey": similarity_key,
                    "chemical_smiles": exact_smiles,
                    "enzyme_smiles": similarity_smiles,
                    "bridge_direction": "chemical_product_to_similar_enzyme_substrate",
                    "confidence_tier": "tier3_high_similarity_nonexact_bridge",
                    "tanimoto": 0.75,
                    "chemical_occurrences": 1,
                    "enzyme_occurrences": 1,
                    "enzyme_ec_sample_json": json.dumps(["1.1.1.2"]),
                    "enzyme_ec_unique": 1,
                }
            ]
        ),
        tmp_path / "similarity_bridge_filtered.parquet",
    )

    retriever = BridgeRetrieverV0(tmp_path)
    rows = retriever.retrieve(exact_smiles, top_k=4)

    assert len(rows) == 2
    assert rows[0].source == "exact_bridge_strict"
    assert rows[0].enzyme_ec_sample == ("1.1.1.1",)
    assert rows[1].source == "similarity_bridge_filtered"


def test_bridge_retriever_uses_scored_cache_without_scorer(tmp_path):
    exact_smiles = "CCO"
    exact_key = inchikey_from_smiles(exact_smiles)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chemical_smiles": exact_smiles,
                    "enzyme_smiles": exact_smiles,
                    "chemical_inchikey": exact_key,
                    "enzyme_inchikey": exact_key,
                    "bridge_direction": "chemical_product_to_enzyme_substrate",
                    "confidence_tier": "tier1_strict_exact_substrate_bridge",
                    "source": "exact_bridge_strict",
                    "tanimoto": 1.0,
                    "enzyme_ec_sample_json": json.dumps(["1.1.1.1"]),
                    "verifier_score": 0.99,
                    "verifier_pass": True,
                    "metadata_json": "{}",
                }
            ]
        ),
        tmp_path / "bridge_candidates_scored.parquet",
    )

    retriever = BridgeRetrieverV0(tmp_path)
    rows = retriever.retrieve(exact_smiles, top_k=4, require_verifier_pass=True)

    assert len(rows) == 1
    assert rows[0].verifier_score == 0.99
    assert rows[0].verifier_pass is True
