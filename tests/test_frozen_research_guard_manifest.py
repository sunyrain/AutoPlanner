from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_CLI_COMPONENTS = {
    "cascade_planner/eval/train_route_pool_ranker.py": "train_route_pool_ranker",
    "cascade_planner/eval/train_route_pool_lambdarank.py": "train_route_pool_lambdarank",
    "cascade_planner/eval/train_ccts_v0_transition_ranker.py": "train_ccts_v0_transition_ranker",
    "cascade_planner/eval/run_v4_full_training_pipeline.py": "v4 full action/source value training pipeline",
}


class FrozenResearchGuardManifestTest(unittest.TestCase):
    def test_representative_frozen_cli_entrypoints_require_legacy_guard(self) -> None:
        for rel_path, component in FROZEN_CLI_COMPONENTS.items():
            text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIn("from cascade_planner.legacy_guard import require_legacy_research_enabled", text)
            self.assertIn('if __name__ == "__main__":', text)
            self.assertIn(f'require_legacy_research_enabled("{component}")', text)


if __name__ == "__main__":
    unittest.main()
