import unittest
import numpy as np
from tests.pimpedrandom import PimpedRandom
from models.black_scholes import BlackScholes, OptionRight


class BlackScholesTest(unittest.TestCase):

    def random_bs(self, rng, F=None, K=None, right=None, sigma=None, T=None):
        F = F or (80.0 + rng.random() * 20.0)
        K = K or (F + rng.random() * 2.0 - 1.0)
        sigma = sigma or (rng.random() * 0.5)
        T = T or rng.random()
        right = right or rng.enum_choice(OptionRight)
        return BlackScholes(F, K, right, sigma, T)

    def test_intrinsic(self):
        rng = PimpedRandom()
        for _ in range(100):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)
            bs = self.random_bs(rng, sigma=1e-6)

            self.assertAlmostEqual(
                bs.intrinsic(),
                bs.undiscounted_value(),
                delta=1e-4,
                msg=f"Seed was {seed}"
            )


if __name__ == '__main__':
    unittest.main()
