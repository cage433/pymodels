import unittest

from models.brownian_bridge import BrownianBridge
import numpy as np

from tests.pimpedrandom import PimpedRandom
from tests.test_utils import random_times


class BrownianBridgeTests(unittest.TestCase):


    def test_spike(self):
        rng = PimpedRandom()
        for _ in range(10):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)
            n_times = np.random.randint(1, 10)
            times = random_times(rng, n_times)
            n_paths = 1024 << 2
            uniforms = np.random.rand(n_paths, n_times)
            bldr = BrownianBridge(times)
            paths = [bldr.generate(u) for u in uniforms]

            for i_time in range(n_times):
                sample = [p[i_time] for p in paths]
                expected_std_dev = np.sqrt(times[i_time])
                self.assertAlmostEqual(
                    np.asscalar(np.mean(sample)),
                    0.0,
                    delta=expected_std_dev * 4.0 / np.sqrt(n_paths),
                    msg=f"Seed = {seed}"
                )

                sample_std_dev = np.asscalar(np.std(sample))
                self.assertAlmostEqual(
                    sample_std_dev,
                    expected_std_dev,
                    delta=0.02,
                    msg=f"Seed = {seed}"
                )
