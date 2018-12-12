import unittest
import numpy as np

from models.brownian_generator import BrownianGenerator
from models.sobol_generator import SobolGenerator
from tests.pimpedrandom import PimpedRandom
from tests.test_utils import random_times


class BrownianGeneratorTest(unittest.TestCase):

    def test_independent_paths(self):

        rng = PimpedRandom()

        for _ in range(10):
            seed = np.random.random()
            rng.seed(seed)

            n_paths = 1024
            n_variables = rng.randint(1, 4)
            n_times = rng.randint(50, 100)
            times = random_times(rng, n_times)

            uniforms = SobolGenerator(n_variables * n_times).generate(n_paths)
            brownians = BrownianGenerator().generate(uniforms, n_variables, times)
            self.assertEqual(brownians.shape, (n_paths, n_variables, n_times))

