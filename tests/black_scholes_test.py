import unittest
import numpy as np
from tests.pimpedrandom import PimpedRandom
from models.black_scholes import BlackScholes, OptionRight


def random_bs(rng, F=None, K=None, right=None, sigma=None, T=None):
    F = F or (80.0 + rng.random() * 20.0)
    sigma = sigma or (rng.random() * 0.5)
    T = T or rng.random()
    std_devs = 3.0 * (rng.random() - 0.5)
    K = K or (F * np.exp(std_devs * sigma * np.sqrt(T)))
    right = right or rng.enum_choice(OptionRight)
    return BlackScholes(F, K, right, sigma, T)

class BlackScholesTest(unittest.TestCase):

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
