import numpy as np

from models.brownian_generator import generate_brownians
from numpy import exp, sqrt


class CombinedPriceProcess:
    def __init__(self,
                 fwd_prices: np.ndarray,
                 times: np.ndarray,
                 sigma: float,
                 tilt_vol: float,
                 ):
        self.fwd_prices = fwd_prices
        self.times = times
        self.sigma = sigma
        self.tilt_vol = tilt_vol

    def generate(self, brownians, i_time):
        t = self.times[i_time]
        z1 = brownians[:, 0, i_time]
        z2 = brownians[:, 1, i_time]
        p1 = self.fwd_prices[i_time] * exp(z1 * self.sigma - 0.5 * self.sigma * self.sigma * t)
        tilt = z2 * self.tilt_vol
        return p1 + tilt


def _design_matrix(brownians, i_time):
    z1 = brownians[:, 0, i_time]
    z2 = brownians[:, 1, i_time]
    n_paths = len(z1)
    m = np.zeros((n_paths, 6))
    m[:, 0] = 1.0
    m[:, 1] = z1
    m[:, 2] = z2
    m[:, 3] = z1 * z1
    m[:, 4] = z2 * z2
    m[:, 5] = z1 * z2

    return m


def state_ranges(initial_volume: int, max_volume: int, n_times: int):
    min_levels = np.zeros(n_times + 1, int)
    max_levels = np.zeros(n_times + 1, int)
    min_levels[n_times] = initial_volume
    max_levels[n_times] = initial_volume
    min_levels[0] = initial_volume
    max_levels[0] = initial_volume

    for i_time in range(n_times - 1, 0, -1):
        max_levels[i_time] = min(max_levels[i_time + 1] + 1, max_volume)
        min_levels[i_time] = max(min_levels[i_time + 1] - 1, 0)

    for i_time in range(1, n_times + 1):
        max_levels[i_time] = min(max_levels[i_time], max_levels[i_time - 1] + 1, max_volume)
        min_levels[i_time] = max(min_levels[i_time], min_levels[i_time - 1] - 1, 0)

    return min_levels, max_levels


def svd_solve(design_matrix, option_values):
    u, s, vt = np.linalg.svd(design_matrix, full_matrices=False)
    v = np.transpose(vt)
    sigma_inv = np.diag(1/s)
    print(f"sigma {s}")
    print(f"u {u}")
    print(f"opt values {option_values}")
    pseudo_inverse = np.matmul(np.matmul(v, sigma_inv), np.transpose(u))
    x = np.matmul(pseudo_inverse, option_values)
    return x


def cond_exp(design_matrix, option_values):
    n_paths = len(design_matrix)
    dm1 = design_matrix[0:n_paths // 2]
    ov1 = option_values[0:n_paths // 2]
    dm2 = design_matrix[n_paths // 2:]
    ov2 = option_values[n_paths // 2:]
    sol1 = svd_solve(dm1, ov1)
    sol2 = svd_solve(dm2, ov2)
    solution = np.zeros((n_paths,), float)
    foo = np.matmul(dm1, sol2)
    solution[0:n_paths // 2] = foo
    np.matmul(dm2, sol1, solution[n_paths // 2:])

    return solution


def value_storage_unit(
        initial_volume: int,
        max_volume: int,
        process: CombinedPriceProcess,
        n_paths: int) -> float:
    n_times = len(process.times)
    brownians = generate_brownians(n_paths, n_variables=2, times=process.times)

    min_levels, max_levels = state_ranges(initial_volume, max_volume, n_times)
    n_states = max_volume + 1
    state_values_eod = np.zeros((n_states, n_paths), float)
    state_values_sod = np.zeros((n_states, n_paths), float)

    for i_exercise in range(n_times - 1, -1, -1):
        sod_range = range(min_levels[i_exercise], max_levels[i_exercise] + 1)

        full_eod_range = range(min_levels[i_exercise + 1], max_levels[i_exercise + 1] + 1)

        dm = _design_matrix(brownians, i_exercise)
        cond_exps = np.zeros((len(full_eod_range), n_paths), float)
        min_eod_state = full_eod_range[0]
        for i_eod in full_eod_range:
            post_ex_values = state_values_eod[i_eod]
            cond_exps[i_eod - min_eod_state] = cond_exp(dm, post_ex_values)
        print(f"ex {i_exercise}")
        print(cond_exps)

        prices = process.generate(brownians, i_exercise)

        for i_sod in sod_range:
            i_eod_min = max(i_sod - 1, min_levels[i_exercise + 1])
            i_eod_max = min(i_sod + 1, max_levels[i_exercise + 1])

            def transfer_value(price, i_eod):
                d_volume = i_eod - i_sod
                return price * -d_volume

            def transition_value(price, i_eod, i_path):
                return transfer_value(price, i_eod) + cond_exps[i_eod - min_eod_state, i_path]

            for i_path in range(n_paths):
                price = prices[i_path]
                i_best_eod = i_eod_min
                best_transition_value = transition_value(price, i_eod_min, i_path)

                for i_eod in range(i_eod_min + 1, i_eod_max + 1):
                    next_value = transition_value(price, i_eod, i_path)
                    if next_value > best_transition_value:
                        best_transition_value = next_value
                        i_best_eod = i_eod

                state_values_sod[i_sod, i_path] = transfer_value(price, i_best_eod) + state_values_eod[i_best_eod, i_path]

                if i_path == 17:
                    print(f"sod = {i_sod}, best eod {i_best_eod}, value {state_values_sod[i_sod, i_path]}")

        print("SOD")
        print(state_values_sod)
        tmp = state_values_eod
        state_values_eod = state_values_sod
        state_values_sod = tmp


    return np.mean(state_values_eod[initial_volume])
