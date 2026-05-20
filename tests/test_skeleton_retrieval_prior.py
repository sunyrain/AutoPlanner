import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cascade_planner.cascadeboard.skeleton_inpainter import SkeletonResult
from cascade_planner.cascadeboard.skeleton_retrieval_prior import (
    augment_skeletons_with_retrieval_prior,
    retrieve_skeleton_priors,
)
from cascade_planner.web.app import _skeleton_to_dict


class SkeletonRetrievalPriorTest(unittest.TestCase):
    def _write_prior(self, rows: list[dict]) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "skeleton_prior.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_retrieve_exact_skeleton_prior(self):
        path = self._write_prior([
            {
                "target_smiles": "CCO",
                "depth": 2,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation", "reduction"],
                "ec1_sequence": ["1.1.1.1", "2"],
                "label": 1.0,
                "source_path": "data/example.json",
            },
            {
                "target_smiles": "CCO",
                "depth": 1,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation"],
                "ec1_sequence": ["1"],
            },
        ])

        rows = retrieve_skeleton_priors(
            "CCO",
            depth=2,
            domain="chemoenzymatic",
            pack_path=path,
            min_similarity=0.99,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["type_sequence"], ["oxidation", "reduction"])
        self.assertEqual(rows[0]["similarity"], 1.0)

    def test_retrieve_can_exclude_exact_target(self):
        path = self._write_prior([
            {
                "target_smiles": "CCO",
                "depth": 1,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation"],
            },
        ])

        rows = retrieve_skeleton_priors(
            "OCC",
            depth=1,
            domain="chemoenzymatic",
            pack_path=path,
            min_similarity=0.99,
            exclude_exact_target=True,
        )

        self.assertEqual(rows, [])

    def test_augment_adds_retrieval_prior_before_generated_skeleton(self):
        path = self._write_prior([
            {
                "target_smiles": "CCO",
                "depth": 1,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation"],
                "ec1_sequence": ["1.1.1.1"],
                "label": 1.0,
                "source_path": "data/example.json",
                "doi": "10.0000/example",
            },
        ])
        generated = SkeletonResult(
            types=["hydrolysis"],
            ec1s=[0],
            ec2s=["NONE"],
            Ts=[25.0],
            pHs=[6.0],
            compat_pred="generated",
            opmode_pred="sequential_isolated",
            issues_pred=[],
            log_prob=0.0,
        )

        out = augment_skeletons_with_retrieval_prior(
            [generated],
            target_smiles="CCO",
            depth=1,
            domain="chemoenzymatic",
            pack_path=path,
            max_new=1,
            min_similarity=0.99,
        )

        self.assertEqual([skel.types for skel in out], [["oxidation"], ["hydrolysis"]])
        self.assertEqual(out[0].ec1s, [1])
        self.assertEqual(out[0].Ts, [25.0])
        self.assertEqual(out[0].pHs, [6.0])
        self.assertEqual(out[0].retrieval_prior["doi"], "10.0000/example")

    def test_augment_can_be_disabled_by_environment(self):
        path = self._write_prior([
            {
                "target_smiles": "CCO",
                "depth": 1,
                "route_domain": "chemoenzymatic",
                "type_sequence": ["oxidation"],
            },
        ])
        skeletons = [
            SkeletonResult(
                types=["hydrolysis"],
                ec1s=[0],
                ec2s=["NONE"],
                Ts=[25.0],
                pHs=[6.0],
                compat_pred="generated",
                opmode_pred="sequential_isolated",
                issues_pred=[],
                log_prob=0.0,
            )
        ]

        with patch.dict(os.environ, {"AUTOPLANNER_DISABLE_SKELETON_RETRIEVAL_PRIOR": "1"}):
            out = augment_skeletons_with_retrieval_prior(
                skeletons,
                target_smiles="CCO",
                depth=1,
                domain="chemoenzymatic",
                pack_path=path,
                max_new=1,
                min_similarity=0.99,
            )

        self.assertIs(out, skeletons)

    def test_web_skeleton_serialization_includes_retrieval_prior(self):
        skeleton = SkeletonResult(
            types=["oxidation"],
            ec1s=[1],
            ec2s=["NONE"],
            Ts=[25.0],
            pHs=[6.0],
            compat_pred="empirically_compatible",
            opmode_pred="sequential_isolated",
            issues_pred=[],
            log_prob=3.0,
        )
        skeleton.retrieval_prior = {"source": "skeleton_prior_pack", "similarity": 1.0}

        row = _skeleton_to_dict(skeleton)

        self.assertEqual(row["retrieval_prior"]["source"], "skeleton_prior_pack")
        self.assertEqual(row["retrieval_prior"]["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
