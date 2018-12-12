from models.option_calcs import intrinsic_value, black_scholes, crank_nicholsonn_value
from models.option import OptionRight, ExerciseStyle
from models.day import Day
from numpy import exp


class OptionInstrument:
    def __init__(self, strike: float, right: OptionRight, expiry: Day, ex_style: ExerciseStyle):
        self.strike = strike
        self.right = right
        self.expiry = expiry
        self.ex_style = ex_style

    def intrinsic(self, fwd_price: float) -> float:
        return intrinsic_value(self.right, self.strike, fwd_price)

    def undiscounted_european_value(self, market_day: Day, fwd_price: float, sigma: float):
        t = self.expiry.time_since(market_day)
        return black_scholes(self.right, self.strike, fwd_price, sigma, t)

    def european_value(self, market_day: Day, fwd_price: float, sigma: float, r: float):
        t = self.expiry.time_since(market_day)
        disc = exp(-r * t)
        return self.undiscounted_european_value(market_day, fwd_price, sigma) * disc

    def cn_value(self, market_day: Day, fwd_price: float, sigma: float, r: float) -> float:
        t = self.expiry.time_since(market_day)
        return crank_nicholsonn_value(
            self.right, self.ex_style, self.strike, fwd_price, sigma, r, t,
            100, 100, 4
        )

    def value(self, market_day: Day, fwd_price: float, sigma: float, r: float) -> float:
        if self.ex_style == ExerciseStyle.EUROPEAN:
            return self.european_value(market_day, fwd_price, sigma, r)
        else:
            return self.cn_value(market_day, fwd_price, sigma, r)
