import numpy as np

from models.black_scholes import BlackScholes, OptionCalcs
from scipy.interpolate import CubicSpline

from models.option import ExerciseStyle, OptionRight


class CNSolver:
    def __init__(self, n, std_devs, n_times):
        self.n = n
        self.std_devs = std_devs
        self.dz = 2.0 * std_devs / (n - 1.0)
        self.n_times = n_times


    def _diffusion_matrices(self, dt):
        a = 1.0 / (self.dz * self.dz)

        m1 = np.zeros((self.n, self.n), float)
        np.fill_diagonal(m1[1:self.n - 1], a / 2)
        np.fill_diagonal(m1[1:self.n - 1, 1:], 2.0 / dt - a)
        np.fill_diagonal(m1[1:, 2:], a / 2)
        m1[0, 0] = 1.0
        m1[self.n - 1, self.n - 1] = 1.0

        m2 = np.zeros((self.n, self.n), float)
        np.fill_diagonal(m2[1:self.n - 1], -a / 2)
        np.fill_diagonal(m2[1:self.n - 1, 1:], 2.0 / dt + a)
        np.fill_diagonal(m2[1:, 2:], -a / 2)
        m2[0, 0] = 1.0
        m2[self.n - 1, self.n - 1] = 1.0

        return (m1, m2)

    def z_vec(self):
        i_mid = (self.n - 1) / 2.0
        return np.fromfunction(lambda i: (i - i_mid) * self.dz, (self.n,))

    def diffuse(self, vec, dt, r, next_low_value, next_high_value):
        (m1, m2) = self._diffusion_matrices(dt)
        v1 = np.matmul(m1, vec)
        v2 = np.linalg.solve(m2, v1) * np.exp(-r * dt)
        v2[0] = next_low_value
        v2[self.n - 1] = next_high_value
        return v2


    def solve(self, F, K, sigma, T, right, r, ex_style):
        def price(z, t):
            p = K * np.exp(z * sigma - 0.5 * sigma * sigma * t)
            return p

        def bs(z, t):
            return BlackScholes(price(z, t), K, right, sigma, T - t)

        def intrinsic(z, t):
            return bs(z, t).intrinsic()

        zs: np.ndarray = self.z_vec()
        z0 = zs[0]
        zn = zs[self.n - 1]

        def lower_bound(t):
            return intrinsic(z0, t)

        def upper_bound(t):
            return intrinsic(zn, t)

        dt = T / (self.n_times - 1.0)

        # This is the vector at time n_times - 2 - can use european values here
        vec = list(map(lambda z: bs(z, T - dt).undiscounted_value() * np.exp(-r * dt), zs))

        # Now diffuse the remaining n_times - 2 steps
        for i_near_time in range(self.n_times - 3, -1, -1):
            t_near = i_near_time * dt
            vec = self.diffuse(vec, dt, r, lower_bound(t_near), upper_bound(t_near))

            if ex_style == ExerciseStyle.AMERICAN:
                opt = OptionCalcs()
                intrinsics = list(map(lambda z: opt.intrinsic(right, price(z, t_near), K), zs))
                vec = np.maximum(vec, intrinsics)

        prices = list(map(lambda z: price(z, 0), zs))
        cs = CubicSpline(prices, vec)
        return np.asscalar(cs(F))

