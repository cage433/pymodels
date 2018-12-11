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
        v = np.zeros((self.n, self.n), float)
        np.fill_diagonal(v[1:], 1.0)
        np.fill_diagonal(v, -2.0)
        np.fill_diagonal(v[:, 1:], 1.0)
        v = v / (2.0 * self.dz * self.dz)

        w = np.zeros((self.n, self.n), float)
        np.fill_diagonal(w, 1.0 / dt)

        print(np.shape(v))
        print(np.shape(w))
        m1 = w + v
        m2 = w - v
        m1[0, 0] = 1.0
        m1[self.n - 1, self.n - 1] = 1.0
        m2[0, 0] = 1.0
        m2[self.n - 1, self.n - 1] = 1.0
        return (m1, m2)

    def z_vec(self):
        i_mid = (self.n - 1) / 2.0
        return np.fromfunction(lambda i: (i - i_mid) * self.dz, (self.n,))

    def diffuse(self, vec, dt, next_low_value, next_high_value):
        print(f"Next low {next_low_value}, next high {next_high_value}")
        (m1, m2) = self._diffusion_matrices(dt)
        print(f"Vec {vec}")
        v1 = np.matmul(m1, vec)
        print(v1)
        # print(m2)
        print(f"{len(m2)}, {len(m2[0])}, {len(m1[0])}")
        v2 = np.linalg.solve(m2, v1)
        v2[0] = next_low_value
        v2[self.n - 1] = next_high_value
        print(f"V2 {v2}")
        return v2


    def solve(self, F, K, sigma, T, right, ex_style):
        def price(z, t):
            p = F * np.exp(z * sigma - 0.5 * sigma * sigma * t)
            return p

        def intrinsic(z, t):
            nt = BlackScholes(price(z, t), K, right, sigma, T).intrinsic()
            return nt

        z_mid = np.log(K / F) + sigma * sigma * T / 2.0
        zs = self.z_vec() + z_mid
        z0 = zs[0]
        zn = zs[self.n - 1]

        def lower_bound(t):
            return intrinsic(z0, t)
        def upper_bound(t):
            return intrinsic(zn, t)

        vec = list(map(lambda z: intrinsic(z, T), zs))
        dt = T / (self.n_times - 1.0)

        for i_time in range(self.n_times):
            t_front = (self.n_times - i_time - 1)
            vec2 = self.diffuse(vec, dt, lower_bound(t_front), upper_bound(t_front))
            vec = vec2

        prices = list(map(lambda z: price(z, 0), zs))
        cs = CubicSpline(prices, vec)
        return cs(F)
