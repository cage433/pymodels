import unittest
import numpy as np

from models.crank_nicholson import CNSolver
from models.option import ExerciseStyle
from tests.black_scholes_test import random_bs
from tests.pimpedrandom import PimpedRandom


class CrankNicholsonTest(unittest.TestCase):

    def cn_value(self, solver, bs, r):
        return solver.solve(bs.F, bs.K, bs.sigma, bs.T, bs.right, r, ExerciseStyle.EUROPEAN)

    def test_matches_bs(self):
        rng = PimpedRandom()
        N = 100
        std_devs = 4
        n_times = 100
        solver = CNSolver(N, std_devs, n_times)

        def tolerance(bs, d_sigma):
            v1 = bs.undiscounted_value()
            v2 = bs.dup(sigma=bs.sigma + d_sigma).undiscounted_value()
            return max(abs(v2 - v1), 0.01, bs.intrinsic() * 0.01)

        for _ in range(20):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            bs = random_bs(rng)
            r = 0.1 * rng.random()
            bs = bs.dup(T=max(0.01, bs.T))  # Very small times don't work well for CN

            numeric_value = self.cn_value(solver, bs, r)
            analytic_value = bs.undiscounted_value() * np.exp(-r * bs.T)
            tol = tolerance(bs, d_sigma=0.01)

            # vol_frac = abs(analytic_value - numeric_value) / tol
            # print(f"BS {analytic_value:0.3f}, CN {numeric_value:0.3f}, intrinsic {bs.intrinsic():0.3f}, vol frac {vol_frac:0.3f}")
            self.assertAlmostEqual(numeric_value, analytic_value, delta=tol, msg=f"Seed was {seed}")


if __name__ == '__main__':
    unittest.main()
