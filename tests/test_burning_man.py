import random
import unittest

from burningManSim import generate_activity_groups


class TestBurningManScenario(unittest.TestCase):
    def test_activity_groups_complete_for_tiny_smoke_runs(self):
        random.seed(1)

        groups = generate_activity_groups(3)

        self.assertEqual(sum(group["size"] for group in groups), 3)
        self.assertTrue(all(group["size"] >= 1 for group in groups))


if __name__ == "__main__":
    unittest.main()
