import unittest

from candidates import registry


class CandidateRegistryTest(unittest.TestCase):
    def test_registry_lists_real_runnable_candidate_modules(self) -> None:
        candidates = registry.default_candidates()

        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(all(candidate.runnable for candidate in candidates))
        self.assertTrue(all(candidate.implementation for candidate in candidates))
        self.assertFalse(any("proxy" in candidate.family for candidate in candidates))

    def test_model_factory_registry_rejects_unknown_implementation(self) -> None:
        with self.assertRaises(KeyError):
            registry.get_model_factory("missing_candidate_impl")


if __name__ == "__main__":
    unittest.main()
