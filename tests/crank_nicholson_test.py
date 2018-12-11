import unittest
import numpy as np

from models.black_scholes import BlackScholes
from models.crank_nicholson import CNSolver
from models.option import ExerciseStyle, OptionRight
from tests.black_scholes_test import random_bs
from tests.pimpedrandom import PimpedRandom


class CrankNicholsonTest(unittest.TestCase):

    def _build_solver(self, N=100, std_devs=4, n_times=100):
        return CNSolver(N, std_devs, n_times)


    def test_european_close_to_black_scholes(self):
        rng = PimpedRandom()
        solver = self._build_solver()

        def tolerance(bs, d_sigma):
            v1 = bs.undiscounted_value()
            v2 = bs.dup(sigma=bs.sigma + d_sigma).undiscounted_value()
            return max(abs(v2 - v1), 0.01, bs.intrinsic() * 0.01)

        for _ in range(10):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            bs = random_bs(rng)
            bs = bs.dup(T=max(0.01, bs.T))  # Very small times don't work well for CN
            r = 0.1 * rng.random()

            numeric_value = solver.solve(bs.F, bs.K, bs.sigma, bs.T, bs.right, r, ExerciseStyle.EUROPEAN)
            analytic_value = bs.undiscounted_value() * np.exp(-r * bs.T)

            self.assertAlmostEqual(
                numeric_value,
                analytic_value,
                delta=tolerance(bs, d_sigma=0.01),
                msg=f"Seed was {seed}"
            )

    def test_american_worth_at_least_intrinsic(self):
        rng = PimpedRandom()
        solver = self._build_solver()

        for _ in range(5):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            bs = random_bs(rng)
            bs = bs.dup(T=max(0.01, bs.T))  # Very small times don't work well for CN
            r = 0.1 * rng.random()

            numeric_value = solver.solve(bs.F, bs.K, bs.sigma, bs.T, bs.right, r, ExerciseStyle.AMERICAN)
            intrinsic_value = bs.intrinsic()

            self.assertGreaterEqual(numeric_value, intrinsic_value, msg=f"Seesd was {seed}")

    def test_deep_itm_american_put_worth_intrinsic(self):
        F = 100.0
        K = 150.0
        sigma = 0.05
        T = 0.5
        r = 0.2
        bs = BlackScholes(F, K, OptionRight.PUT, sigma, T)
        european_value = bs.undiscounted_value() * np.exp(-r * T)
        intrinsic = bs.intrinsic()
        cn_value = self._build_solver().solve(F, K, sigma, T, OptionRight.PUT, r, ExerciseStyle.AMERICAN)
        self.assertAlmostEqual(cn_value, intrinsic, delta=0.01)
        self.assertLess(european_value, intrinsic)


if __name__ == '__main__':
    unittest.main()
