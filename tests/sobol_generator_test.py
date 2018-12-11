import unittest
import numpy as np
from scipy.stats.stats import pearsonr

from models.sobol_generator import SobolGenerator
from tests.pimpedrandom import PimpedRandom


class SobolSeqTest(unittest.TestCase):

    def test_distribution_is_uniform_like(self):
        rng = PimpedRandom()

        for _ in range(10):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)
            n_variables = rng.randint(1, 10)
            sd = SobolGenerator(n_variables)
            n_paths = 1 << rng.randint(8, 12)
            sample = sd.generate(n_paths)

            for i_var in range(n_variables):
                v = sample[i_var]

                self.assertAlmostEqual(0.5, np.asscalar(np.mean(v)), delta=0.01, msg=f"Seed = {seed}")
                self.assertAlmostEqual(1.0 / np.sqrt(12), np.asscalar(np.std(v)), delta=0.01, msg=f"Seed = {seed}")

                for j_var in range(n_variables):
                    if i_var != j_var:
                        self.assertAlmostEqual(
                            0.0,
                            pearsonr(sample[i_var], sample[j_var])[0],
                            delta=0.03,
                            msg=f"Seed = {seed}"
                        )
