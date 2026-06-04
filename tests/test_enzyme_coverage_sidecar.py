import json

import pyarrow as pa
import pyarrow.parquet as pq

from cascade_planner.cascade_search.enzyme_coverage_sidecar import (
    EnzymeCoverageSidecarConfig,
    build_enzyme_coverage_sidecar,
)


def test_enzyme_coverage_sidecar_attaches_precedent_candidates(tmp_path, monkeypatch):
    pack = tmp_path / "bridge_pack"
    pack.mkdir()
    key = "IKHGUXGNUITLKF-UHFFFAOYSA-N"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chemical_smiles": "CC=O",
                    "enzyme_smiles": "CC=O",
                    "chemical_inchikey": key,
                    "enzyme_inchikey": key,
                    "bridge_direction": "chemical_product_to_enzyme_product",
                    "confidence_tier": "tier2_strict_exact_product_bridge",
                    "source": "exact_bridge_strict",
                    "tanimoto": 1.0,
                    "enzyme_ec_sample_json": json.dumps(["1.1.1.1"]),
                    "verifier_score": 0.99,
                    "verifier_pass": True,
                    "metadata_json": "{}",
                }
            ]
        ),
        pack / "bridge_candidates_scored.parquet",
    )
    pool = tmp_path / "enzyme_reaction_pool.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "reaction_id": "r1",
                    "substrate_smiles": "CCO",
                    "product_smiles": "CC=O",
                    "reaction_smiles": "CCO>>CC=O",
                    "occurrences": 3,
                    "source_counts_json": json.dumps({"fixture": 3}),
                    "ec_numbers_json": json.dumps(["1.1.1.1"]),
                    "rhea_ids_json": "[]",
                    "example_ids_json": "[]",
                }
            ]
        ),
        pool,
    )
    monkeypatch.setenv("AUTOPLANNER_ENZYME_PRECEDENT_POOL_PATH", str(pool))

    report = build_enzyme_coverage_sidecar(
        "CC=O",
        config=EnzymeCoverageSidecarConfig(pack_dir=pack, top_k=3, bridge_top_k=3, max_ec_contexts=1, enable_sp_v1=False),
    )

    assert report["candidate_count"] >= 1
    assert report["bridge_hit_count"] == 1
    assert report["contexts"][0]["top_candidates"][0]["source"] == "enzyme_precedent"
