import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.eval.train_route_selector_v0 import train_route_selector_v0


class TrainRouteSelectorV0Test(unittest.TestCase):
    def test_trains_from_selector_pack_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split, target_idx in [("train", 1), ("val", 2), ("test", 3)]:
                rows.extend(_target_rows(split=split, target_id=f"target_{target_idx}"))
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_selector_v0(pack_jsonl=pack, output_dir=root / "model", seed=7)

            self.assertEqual(report["schema_version"], "route_selector_v0_training.v1")
            self.assertEqual(report["split_counts"], {"train": 3, "val": 3, "test": 3})
            self.assertTrue((root / "model" / "route_pool_pairwise_ranker.pkl").exists())
            self.assertTrue((root / "model" / "route_pool_pairwise_ranker_report.json").exists())
            self.assertIn("selected_method", report["selection"])
            ranker_report = json.loads((root / "model" / "route_pool_pairwise_ranker_report.json").read_text(encoding="utf-8"))
            self.assertIn("audit_guard", ranker_report["baselines"])
            self.assertIn("audit_guard_plus_learned", ranker_report["blends"])

    def test_auto_resplits_grouped_train_only_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for target_idx in [1, 2, 3]:
                rows.extend(_target_rows(split="train", target_id=f"target_{target_idx}"))
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_selector_v0(pack_jsonl=pack, output_dir=root / "model", seed=7)

            self.assertEqual(report["split_counts"], {"train": 3, "val": 3, "test": 3})
            for split in ("train", "val", "test"):
                split_path = root / "model" / "splits" / f"route_selector_{split}.jsonl"
                split_rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len({row["selector_group_id"] for row in split_rows}), 1)

    def test_uses_selector_group_id_for_pairwise_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack.jsonl"
            rows = []
            for split, group_idx in [("train", 1), ("val", 2), ("test", 3)]:
                selector_group_id = f"same_target_{group_idx}"
                rows.extend(
                    [
                        _row(
                            split=split,
                            target_id=f"artifact_{group_idx}_kept",
                            selector_group_id=selector_group_id,
                            route_id="good",
                            native_rank=1,
                            label=2,
                            native_score=0.5,
                            audit_risk=0,
                        ),
                        _row(
                            split=split,
                            target_id=f"artifact_{group_idx}_rejected",
                            selector_group_id=selector_group_id,
                            route_id="bad",
                            native_rank=0,
                            label=0,
                            native_score=0.9,
                            audit_risk=40,
                        ),
                    ]
                )
            pack.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = train_route_selector_v0(pack_jsonl=pack, output_dir=root / "model", seed=7)

            self.assertEqual(report["split_counts"], {"train": 2, "val": 2, "test": 2})
            ranker_report = json.loads((root / "model" / "route_pool_pairwise_ranker_report.json").read_text(encoding="utf-8"))
            self.assertEqual(ranker_report["counts"]["train_positive_groups"], 1)


def _target_rows(*, split: str, target_id: str) -> list[dict]:
    return [
        _row(split=split, target_id=target_id, route_id="good", native_rank=1, label=2, native_score=0.5, audit_risk=0),
        _row(split=split, target_id=target_id, route_id="weak", native_rank=0, label=1, native_score=0.9, audit_risk=20),
        _row(split=split, target_id=target_id, route_id="bad", native_rank=2, label=0, native_score=0.8, audit_risk=40),
    ]


def _row(
    *,
    split: str,
    target_id: str,
    route_id: str,
    native_rank: int,
    label: int,
    native_score: float,
    audit_risk: int,
    selector_group_id: str | None = None,
) -> dict:
    return {
        "split": split,
        "target_id": target_id,
        "selector_group_id": selector_group_id or target_id,
        "route_id": f"{target_id}_{route_id}",
        "native_rank": native_rank,
        "route_label": label,
        "feature": {
            "native_score": native_score,
            "native_rank": float(native_rank),
            "native_inv_rank": 1.0 / (native_rank + 1),
            "n_steps": 1.0,
            "stock_closed": float(label > 0),
            "route_solved": float(label > 0),
            "audit_risk_order": float(audit_risk),
            "audit_is_reject": float(label == 0),
            "large_atom_gain_count": float(label == 0),
        },
    }


if __name__ == "__main__":
    unittest.main()
