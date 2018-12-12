from scipy.stats import norm

from numpy import sqrt, log, exp
import numpy as np
from scipy.interpolate import CubicSpline

from models.option import ExerciseStyle, OptionRight


def intrinsic_value(right: OptionRight, strike: float, fwd_price: float) -> float:
    if right is OptionRight.CALL:
        return max(0.0, fwd_price - strike)
    if right is OptionRight.PUT:
        return max(0.0, strike - fwd_price)
    if right is OptionRight.STRADDLE:
        return max(strike - fwd_price, fwd_price - strike)
    raise Exception(f"Unexpected option right {right}")


def black_scholes(
        right: OptionRight,
        strike: float,
        fwd_price: float,
        sigma: float,
        t: float) -> float:
    d1 = 1.0 / (sigma * sqrt(t)) * (
            log(fwd_price / strike) + 0.5 * sigma * sigma * t)
    d2 = d1 - sigma * sqrt(t)
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)

    if right is OptionRight.CALL:
        return N1 * fwd_price - N2 * strike
    if right is OptionRight.PUT:
        return (1.0 - N2) * strike - (1. - N1) * fwd_price
    if right is OptionRight.STRADDLE:
        return (1.0 - 2.0 * N2) * strike - (1. - 2.0 * N1) * fwd_price
    raise Exception(f"Unexpected option right {right}")

def crank_nicholsonn_value(
        right: OptionRight,
        ex_style: ExerciseStyle,
        strike: float,
        fwd_price: float,
        sigma: float,
        r: float,
        time_to_expiry: float,
        n: int, n_times: int, std_devs: float) -> float:

    dz = 2.0 * std_devs / (n - 1.0)
    dt = time_to_expiry / (n_times - 1.0)

    def diffusion_matrices():
        a = 1.0 / (dz * dz)

        m1 = np.zeros((n, n), float)
        np.fill_diagonal(m1[1:n - 1], a / 2)
        np.fill_diagonal(m1[1:n - 1, 1:], 2.0 / dt - a)
        np.fill_diagonal(m1[1:, 2:], a / 2)
        m1[0, 0] = 1.0
        m1[n - 1, n - 1] = 1.0

        m2 = np.zeros((n, n), float)
        np.fill_diagonal(m2[1:n - 1], -a / 2)
        np.fill_diagonal(m2[1:n - 1, 1:], 2.0 / dt + a)
        np.fill_diagonal(m2[1:, 2:], -a / 2)
        m2[0, 0] = 1.0
        m2[n - 1, n - 1] = 1.0

        return m1, m2

    zs = np.arange(-std_devs, std_devs + dz / 2.0, dz)
    z0 = zs[0]
    zn = zs[n - 1]

    (m1, m2) = diffusion_matrices()

    def diffuse(vec, next_low_value, next_high_value):
        v1 = np.matmul(m1, vec)
        v2 = np.linalg.solve(m2, v1) * exp(-r * dt)
        v2[0] = next_low_value
        v2[n - 1] = next_high_value
        return v2

    def price(z, t):
        p = strike * exp(z * sigma - 0.5 * sigma * sigma * t)
        return p

    def undiscounted_bs(z, t):
        return black_scholes(right, strike, price(z, t), sigma, time_to_expiry - t)

    def intrinsic(z, t):
        return intrinsic_value(right, strike, price(z, t))

    def lower_bound(t):
        return intrinsic(z0, t)

    def upper_bound(t):
        return intrinsic(zn, t)

    # This is the vector at time n_times - 2 - can use european values here
    def penultimate_value(z):
        return undiscounted_bs(z, time_to_expiry - dt)

    vec = penultimate_value(zs)

    # Now diffuse the remaining n_times - 2 steps
    for i_near_time in range(n_times - 3, -1, -1):
        t_near = i_near_time * dt
        vec = diffuse(vec, lower_bound(t_near), upper_bound(t_near))

        if ex_style == ExerciseStyle.AMERICAN:
            intrinsics = list(map(lambda z: intrinsic(z, t_near), zs))
            vec = np.maximum(vec, intrinsics)

    prices = list(map(lambda z: price(z, 0), zs))
    cs = CubicSpline(prices, vec)
    return np.asscalar(cs(fwd_price))

