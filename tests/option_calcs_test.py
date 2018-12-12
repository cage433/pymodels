import unittest

import numpy as np

from models.day import Day
from models.option import ExerciseStyle
from models.option_instrument import OptionInstrument
from tests.pimpedrandom import PimpedRandom
from models.option_calcs import OptionRight


def random_option(
        rng: PimpedRandom,
        strike: float = None,
        right: OptionRight = None,
        expiry: Day = None,
        ex_style: ExerciseStyle = None):
    return OptionInstrument(
        strike or 80.0 + rng.random() * 20,
        right or rng.enum_choice(OptionRight),
        expiry or Day(2018, 1, 2) + rng.randint(1, 100),
        ex_style or rng.enum_choice(ExerciseStyle)
    )


class OptionCalcsTest(unittest.TestCase):

    def test_intrinsic(self):
        rng = PimpedRandom()
        for _ in range(100):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)
            option = random_option(rng, ex_style=ExerciseStyle.EUROPEAN)
            market_day = Day(2018, 1, 1)
            fwd_price = option.strike + rng.uniform(-2.0, 2.0)

            self.assertAlmostEqual(
                option.european_value(market_day, fwd_price, sigma=1e-5, r=0.0),
                option.intrinsic(fwd_price),
                delta=1e-3,
                msg=f"Seed was {seed}"
            )

    def test_european_close_to_black_scholes(self):
        rng = PimpedRandom()

        for _ in range(10):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            market_day = Day(2018, 1, 1)
            option = random_option(rng, ex_style=ExerciseStyle.EUROPEAN)
            sigma = rng.random() * 0.5
            fwd_price = rng.uniform(option.strike - 1.0, option.strike + 1.0)

            r = 0.1 * rng.random()

            numeric_value = option.cn_value(
                market_day, fwd_price, sigma, r
            )

            def bs_value(s):
                return option.european_value(market_day, fwd_price, s, r)

            analytic_value = bs_value(sigma)
            tol = max(bs_value(sigma + 0.01) - analytic_value, 0.01, analytic_value * 0.01)

            self.assertAlmostEqual(
                numeric_value,
                analytic_value,
                delta=tol,
                msg=f"Seed was {seed}"
            )

    def test_american_worth_at_least_intrinsic(self):
        rng = PimpedRandom()

        for _ in range(5):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            market_day = Day(2018, 1, 1)
            option = random_option(rng, ex_style=ExerciseStyle.AMERICAN)
            fwd_price = rng.uniform(option.strike - 1.0, option.strike + 1.0)
            sigma = rng.random() * 0.5
            r = 0.1 * rng.random()

            numeric_value = option.cn_value(market_day, fwd_price, sigma, r)
            intrinsic_value = option.intrinsic(fwd_price)

            self.assertGreaterEqual(numeric_value, intrinsic_value, msg=f"Seesd was {seed}")

    def test_deep_itm_american_put_worth_intrinsic(self):
        F = 100.0
        K = 150.0
        sigma = 0.05
        market_day = Day(2018, 1, 1)
        r = 0.2
        option = OptionInstrument(K, OptionRight.PUT, Day(2018, 6, 1), ExerciseStyle.AMERICAN)
        european_value = option.european_value(market_day, F, sigma, r)
        intrinsic = option.intrinsic(F)
        cn_value = option.cn_value(market_day, F, sigma, r)
        self.assertAlmostEqual(cn_value, intrinsic, delta=0.01)
        self.assertLess(european_value, intrinsic)

    def test_mc_value_close_to_bs(self):
        rng = PimpedRandom()

        for _ in range(5):
            seed = np.random.randint(0, 100 * 1000)
            rng.seed(seed)

            market_day = Day(2018, 1, 1)
            option = random_option(rng, ex_style=ExerciseStyle.EUROPEAN)
            fwd_price = rng.uniform(option.strike - 1.0, option.strike + 1.0)
            sigma = rng.random() * 0.5
            r = 0.1 * rng.random()

            mc_value = option.mc_european_value(market_day, fwd_price, sigma, r)
            bs_value = option.european_value(market_day, fwd_price, sigma, r)

            self.assertAlmostEqual(mc_value, bs_value, delta=0.1, msg=f"Seesd was {seed}")
