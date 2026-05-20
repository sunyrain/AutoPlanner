import csv
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.build_external_candidate_pools import build_external_candidate_pools
from cascade_planner.eval.build_external_step_pairs import build_external_step_pairs
from cascade_planner.eval.build_vnext_pack import build_vnext_pack, merge_external_step_pairs_into_vnext_pack
from cascade_planner.eval.train_vnext_from_pack import build_candidate_pool_dataset
from cascade_planner.vnext.features import read_jsonl, write_jsonl


class ExternalVNextDataTest(unittest.TestCase):
    def test_external_step_pair_import_from_local_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ecreact = root / "ecreact.csv"
            with ecreact.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["rxn_smiles", "ec", "source"])
                writer.writeheader()
                writer.writerow({"rxn_smiles": "CCO|1.1.1.1>>CC=O", "ec": "1.1.1.1", "source": "unit"})

            enz = root / "enz.json"
            enz.write_text(json.dumps([{"reactants": "CCN", "product": "CCO", "ec": "2.1.1.1"}]), encoding="utf-8")

            uspto = root / "uspto.tab"
            uspto.write_text("reactant\tproduct\tcategory\nCCCC\tCCC\t1\n", encoding="utf-8")

            rhea = root / "rhea.tar.bz2"
            rhea_root = root / "rhea_files" / "140" / "tsv"
            rhea_root.mkdir(parents=True)
            (rhea_root / "rhea2ec.tsv").write_text("RHEA_ID\tID\n10000\t3.1.1.1\n", encoding="utf-8")
            (rhea_root / "rhea-reaction-smiles.tsv").write_text("10000\tCCOC>>CCO.C=O\n", encoding="utf-8")
            with tarfile.open(rhea, "w:bz2") as tar:
                tar.add(rhea_root.parent, arcname="140")

            out = root / "external"
            manifest = build_external_step_pairs(
                output_dir=out,
                ecreact_csv=ecreact,
                enzymatic_json=[enz],
                uspto_tab=uspto,
                rhea_tar=rhea,
                reactzyme_zip=None,
                retrorules_templates=[],
            )
            rows = read_jsonl(out / "external_step_pairs.jsonl")

        self.assertEqual(manifest["counts"]["external_step_pairs"], 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(manifest["quality"]["with_ec"], 3)
        self.assertIn("ecreact", manifest["quality"]["sources"])
        self.assertTrue(all(row["label_type"] == "external_curated_step" for row in rows))

    def test_external_candidate_pools_and_vnext_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            external_rows = []
            for idx in range(12):
                product = f"{'C' * (idx + 2)}O"
                reactant = f"{'C' * (idx + 1)}N"
                rxn = f"{reactant}>>{product}"
                external_rows.append({
                    "step_id": f"ext_{idx}",
                    "group_id": f"group_{idx}",
                    "target_smiles": product,
                    "product": product,
                    "reactants": [reactant],
                    "reaction_smiles": rxn,
                    "reaction_type": "enzyme_reaction" if idx % 2 else "uspto_class_1",
                    "ec": "1.1.1.1" if idx % 2 else "",
                    "source": "ecreact" if idx % 2 else "uspto50k",
                    "label": 1.0,
                    "label_type": "external_curated_step",
                    "weight": 2.0,
                    "gt_available": True,
                    "candidate": {
                        "main_reactant": reactant,
                        "aux_reactants": [],
                        "source": "ecreact" if idx % 2 else "uspto50k",
                        "reaction_type": "enzyme_reaction" if idx % 2 else "uspto_class_1",
                        "rxn_smiles": rxn,
                        "ec": "1.1.1.1" if idx % 2 else "",
                        "score": 1.0,
                    },
                })
            external_path = root / "external_step_pairs.jsonl"
            write_jsonl(external_path, external_rows)

            pools_out = root / "pools"
            pool_manifest = build_external_candidate_pools(
                external_step_pairs=external_path,
                output_dir=pools_out,
                max_pools=6,
                max_candidates=4,
                seed=7,
            )
            pools = read_jsonl(pools_out / "external_candidate_pools.jsonl")

            pack = root / "base_pack"
            pack.mkdir()
            candidate_row = {
                "candidate_id": "base_1",
                "route_id": "route_1",
                "target_smiles": "CCO",
                "product": "CCO",
                "step_index": 0,
                "rank": 1,
                "label": 1.0,
                "label_type": "benchmark_exact",
                "candidate": {"main_reactant": "CC", "aux_reactants": ["O"], "source": "retrochimera", "score": 1.0},
            }
            route_row = {"route_id": "route_1", "target_smiles": "CCO", "label": 1.0, "label_type": "professional_solved"}
            write_jsonl(pack / "candidate_ranking.jsonl", [candidate_row])
            write_jsonl(pack / "route_value.jsonl", [route_row])
            base_vnext = root / "base_vnext"
            build_vnext_pack(pack_dir=pack, output_dir=base_vnext, max_candidates=4)

            merged = root / "merged_vnext"
            manifest = merge_external_step_pairs_into_vnext_pack(
                base_vnext_pack=base_vnext,
                output_dir=merged,
                external_step_pair_paths=[external_path],
                external_candidate_pool_paths=[pools_out / "external_candidate_pools.jsonl"],
            )
            cache_dir = root / "feature_cache"
            first_ds = build_candidate_pool_dataset(merged, n_bits=16, max_candidates=4, feature_cache_dir=cache_dir)
            second_ds = build_candidate_pool_dataset(merged, n_bits=16, max_candidates=4, feature_cache_dir=cache_dir)

        self.assertEqual(pool_manifest["counts"]["candidate_pools"], 6)
        self.assertTrue(all(pool["positive_count"] == 1 for pool in pools))
        self.assertTrue(all(2 <= len(pool["candidates"]) <= 4 for pool in pools))
        self.assertGreaterEqual(manifest["counts"]["step_pairs"], 13)
        self.assertEqual(manifest["external_candidate_pools"], 6)
        self.assertEqual(manifest["counts"]["candidate_pools"], 7)
        self.assertEqual(manifest["counts"]["search_transitions"], 7)
        self.assertEqual(first_ds.candidate_features.shape[0], 7)
        self.assertFalse(first_ds.feature_schema.get("feature_cache_hit"))
        self.assertTrue(second_ds.feature_schema.get("feature_cache_hit"))


if __name__ == "__main__":
    unittest.main()
