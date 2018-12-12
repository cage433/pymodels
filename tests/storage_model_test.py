import numpy as np

from models.brownian_generator import generate_brownians
from models.storage_model import CombinedPriceProcess, state_ranges, value_storage_unit
from tests.pimpedrandom import PimpedRandom
from tests.stats_test_mixin import StatsTestMixin
from tests.test_utils import random_times

from scipy.stats import lognorm
from numpy import log, exp, sqrt


class StorageModelTest(StatsTestMixin):

    def test_process(self):

        rng = PimpedRandom()

        for _ in range(10):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            sigma = rng.random() * 0.5
            tilt_vol = rng.uniform(0.0, 1.5)
            n_times = rng.randint(1, 5)
            n_paths = 1024
            times = random_times(rng, n_times)
            brownians = generate_brownians(n_paths, 2, times)
            fwd_price = rng.uniform(90.0, 100.0)
            fwd_prices = np.full((n_times,), fwd_price)

            process = CombinedPriceProcess(fwd_prices, times, sigma, tilt_vol=tilt_vol)
            for i_time in range(n_times):
                t = times[i_time]
                s1 = lognorm.std(sigma * sqrt(t), 0, fwd_price * exp(-0. * sigma * sigma * t))
                s2 = tilt_vol * sqrt(t)
                std_dev = sqrt(s1 * s1 + s2 * s2)
                std_err = std_dev / sqrt(n_paths)
                prices = process.generate(brownians, i_time)
                self.check_mean(prices, fwd_price, 4.0 * std_err)
                self.check_std_dev(prices, std_dev, std_dev * 0.1)

    def test_state_ranges(self):
        rng = PimpedRandom()

        for _ in range(100):
            seed = np.random.randint(0, 100 * 1000)
            # seed = 9120
            rng.seed(seed)

            max_volume = rng.randint(2, 6)
            initial_volume = rng.randint(0, max_volume)
            n_times = rng.randint(2, max_volume * 2)

            m1, m2 = state_ranges(initial_volume, max_volume, n_times)

            self.assertLessEqual(max(m2), max_volume, f"seed = {seed}")
            self.assertLessEqual(max(m1), max_volume, f"seed = {seed}")
            self.assertGreaterEqual(min(m2), 0, f"seed = {seed}")
            self.assertGreaterEqual(min(m1), 0, f"seed = {seed}")

            self.assertEqual(m1[0], initial_volume)
            self.assertEqual(m2[0], initial_volume)
            self.assertEqual(m1[n_times], initial_volume)
            self.assertEqual(m2[n_times], initial_volume)

            for i_time in range(0, n_times):
                self.assertLessEqual(abs(m1[i_time] - m1[i_time + 1]), 1, f"Seed = {seed}")
                self.assertLessEqual(m1[i_time], initial_volume, f"Seed = {seed}")
                self.assertGreaterEqual(m2[i_time], initial_volume, f"Seed = {seed}")
                self.assertLessEqual(abs(m2[i_time] - m2[i_time + 1]), 1, f"Seed = {seed}")

    def test_intrinsic_model(self):

        times = np.array([0, 1])
        fwd_prices = np.array([10.0, 15.5])
        n_paths = 10
        process = CombinedPriceProcess(fwd_prices, times, sigma=0.0, tilt_vol=0.0)
        value = value_storage_unit(
            initial_volume=0,
            max_volume=1,
            process=process,
            n_paths=n_paths
        )
        self.assertAlmostEqual(value, 5.0, delta = 0.01)
