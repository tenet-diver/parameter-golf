import unittest

from train_gpt import resolve_local_sgd_sync_steps, should_average_local_sgd


class LocalSgdTest(unittest.TestCase):
    def test_rejects_invalid_sync_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "LOCAL_SGD_SYNC_STEPS"):
            resolve_local_sgd_sync_steps(0)

    def test_average_schedule_skips_default_and_hits_period_boundary(self) -> None:
        self.assertFalse(should_average_local_sgd(1, 1))
        self.assertFalse(should_average_local_sgd(3, 4))
        self.assertTrue(should_average_local_sgd(4, 4))
        self.assertTrue(should_average_local_sgd(8, 4))


if __name__ == "__main__":
    unittest.main()
