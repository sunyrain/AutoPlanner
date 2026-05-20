import unittest

from cascade_planner.cascadeboard.live_retro import _CachingPredictor, retro_cache_max_entries, retro_engine_cache_stats


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, product_smiles, top_k=10, ec_token=""):
        self.calls += 1
        return [{
            "main_reactant": "CCO",
            "rxn_smiles": f"CCO>>{product_smiles}",
            "score": 1.0,
            "ec": ec_token,
        }][:top_k]


class LiveRetroCacheTest(unittest.TestCase):
    def test_predict_cache_returns_deep_copies(self):
        fake = FakePredictor()
        cached = _CachingPredictor(fake, "fake")

        first = cached.predict("CC=O", top_k=1, ec_token="1")
        first[0]["score"] = 99.0
        second = cached.predict("CC=O", top_k=1, ec_token="1")
        third = cached.predict("CC=O", top_k=1, ec_token="2")

        self.assertEqual(fake.calls, 2)
        self.assertEqual(second[0]["score"], 1.0)
        self.assertEqual(third[0]["ec"], "2")
        self.assertEqual(cached.cache_stats()["hits"], 1)
        self.assertEqual(cached.cache_stats()["misses"], 2)
        self.assertGreater(cached.cache_stats()["max_entries"], 0)
        self.assertIsNotNone(cached.cache_stats()["avg_hit_time_ms"])
        self.assertIsNotNone(cached.cache_stats()["avg_miss_time_ms"])

    def test_predict_cache_evicts_lru_entries(self):
        fake = FakePredictor()
        cached = _CachingPredictor(fake, "fake", max_entries=1)

        cached.predict("A")
        cached.predict("B")
        cached.predict("A")

        self.assertEqual(fake.calls, 3)
        self.assertEqual(cached.cache_stats()["entries"], 1)

    def test_retro_engine_cache_stats_skips_none_engines(self):
        cached = _CachingPredictor(FakePredictor(), "fake")
        cached.predict("CC=O")

        stats = retro_engine_cache_stats({"fake": cached, "missing": None})

        self.assertIn("fake", stats)
        self.assertNotIn("missing", stats)
        self.assertEqual(stats["fake"]["misses"], 1)

    def test_retro_cache_max_entries_is_clamped(self):
        self.assertEqual(retro_cache_max_entries(default=-1), 0)


if __name__ == "__main__":
    unittest.main()
