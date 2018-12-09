import numpy as np

from models.black_scholes import BlackScholes
from scipy.interpolate import CubicSpline


class CNSolver:
    def __init__(self, n, std_devs, n_times):
        self.n = n
        self.std_devs = std_devs
        self.dz = 2.0 * std_devs / (n - 1.0)
        self.n_times = n_times


    def _diffusion_matrices(self, dt):
        v = np.zeros((self.n - 2, self.n), float)
        np.fill_diagonal(v, 1.0)
        np.fill_diagonal(v[:, 1:], -2.0)
        np.fill_diagonal(v[:, 2:], 1.0)
        v = v / (2.0 * self.dz * self.dz)

        w = np.zeros((self.n - 2, self.n), float)
        np.fill_diagonal(w[:, 1:], 1.0 / dt)

        m1 = w + v
        m2 = w - v
        return (m1, m2)

    def z_vec(self):
        i_mid = (self.n - 1) / 2.0
        return np.fromfunction(lambda i: (i - i_mid) * self.dz, (self.n,))

    def diffuse(self, vec, dt, next_low_value, next_high_value):
        (m1, m2) = self._diffusion_matrices(dt)
        v1 = m1 * vec
        v2 = np.linalg.solve(m2, v1)
        result = np.zeros((self.n,), float)
        result[1:(self.n - 2)] = v2
        result[0] = next_low_value
        result[self.n - 1] = next_high_value
        return result


    def solve(self, F, K, sigma, T, right, ex_style):
        def price(z, t):
            return F * np.exp(z * sigma * np.sqrt(t) - 0.5 * sigma * sigma * t)

        def intrinsic(z, t):
            return BlackScholes(price(z, t), K, right, sigma, T).intrinsic()

        z_mid = np.log(K / F) + sigma * sigma * T / 2.0
        zs = self.z_vec() + z_mid
        z0 = zs[0]
        zn = zs[self.n - 1]

        def lower_bound(t):
            intrinsic(z0, t)
        def upper_bound(t):
            intrinsic(zn, t)

        vec = map(lambda z: intrinsic(z, T), zs )
        dt = T / (self.n_times - 1.0)

        for i_time in range(self.n_times):
            t_fromt = (self.n_times - i_time - 1)
            vec2 = self.diffuse(vec, dt, lower_bound(t_fromt), upper_bound(t_fromt))
            vec = vec2

        prices = map(lambda z: price(z, 0), zs)
        cs = CubicSpline(prices, vec)
        return cs(F)
