import numpy as np

from models.brownian_generator import BrownianGenerator
from models.sobol_generator import SobolGenerator
from tests.pimpedrandom import PimpedRandom
from tests.stats_test_mixin import StatsTestMixin
from tests.test_utils import random_times


class BrownianGeneratorTest(StatsTestMixin):

    def test_shape(self):

        rng = PimpedRandom()

        for _ in range(5):
            seed = np.random.random()
            rng.seed(seed)

            n_paths = 100
            n_variables = rng.randint(1, 4)
            n_times = rng.randint(50, 100)
            times = random_times(rng, n_times)

            uniforms = SobolGenerator(n_variables * n_times).generate(n_paths)
            brownians = BrownianGenerator().generate(uniforms, n_variables, times)
            self.assertEqual(brownians.shape, (n_paths, n_variables, n_times))

    def test_independence(self):
        rng = PimpedRandom()

        for _ in range(5):
            seed = np.random.random()
            rng.seed(seed)

            n_paths = 1024 << 2
            n_variables = rng.randint(2, 4)
            n_times = rng.randint(1, 6)
            times = random_times(rng, n_times)

            uniforms = SobolGenerator(n_variables * n_times).generate(n_paths)
            brownians = BrownianGenerator().generate(uniforms, n_variables, times)

            for i_time in range(n_times):
                for i_var in range(n_variables):
                    sample_i = brownians[:, i_var, i_time]

                    self.check_std_dev(sample_i, np.sqrt(times[i_time]), tol=0.03, msg=f"Seed = {seed}")
                    self.check_mean(sample_i, 0.0, tol=0.03, msg=f"Seed = {seed}")

                    for j_var in range(i_var + 1, n_variables):
                        sample_j = brownians[:, j_var, i_time]
                        self.check_uncorrelated(
                            sample_i,
                            sample_j,
                            tol=0.03,
                            msg=f"Seed = {seed}"
                        )
