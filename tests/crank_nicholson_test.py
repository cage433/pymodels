import unittest
import numpy as np

from models.crank_nicholson import CNSolver
from tests.black_scholes_test import random_bs
from tests.pimpedrandom import PimpedRandom

class CrankNicholsonTest(unittest.TestCase):
    def test_matches_bs(self):
        rng = PimpedRandom()
        N = 100
        std_devs = 4
        n_times = 100
        for _ in range(100):
            seed = np.random.randint(0, 100 * 1000)
            # seed = 70889
            rng.seed(seed)

            bs = random_bs(rng)
            bs = bs.dup(T = max(0.01, bs.T)) # Very small times don't work well for CN
            print(bs)

            solver = CNSolver(N, std_devs, n_times)
            numeric_value = solver.solve_bs(bs)
            analytic_value1 = bs.undiscounted_value()
            analytic_value2 = bs.dup(sigma=bs.sigma + 0.01).undiscounted_value()
            delta = max(abs(analytic_value2 - analytic_value1), 0.01)
            print(f"BS {analytic_value1:0.3f}, CN {numeric_value:0.3f}, delta {delta:0.3f}")
            self.assertAlmostEqual(numeric_value, analytic_value1, delta= delta, msg=f"Seed was {seed}")

if __name__ == '__main__':
    unittest.main()

