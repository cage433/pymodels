import numpy as np

from models.sobol_generator import SobolGenerator
from tests.pimpedrandom import PimpedRandom
from tests.stats_test_mixin import StatsTestMixin


class SobolSeqTest(StatsTestMixin):

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

                self.check_mean(v, expected=0.5, tol=0.03, msg=f"Seed = {seed}")
                self.check_std_dev(v, expected=1.0 / np.sqrt(12), tol=0.01, msg=f"Seed = {seed}")

                for j_var in range(i_var + 1, n_variables):
                    self.check_uncorrelated(
                        sample[i_var], sample[j_var],
                        tol=0.03,
                        msg=f"Seed = {seed}"
                    )
