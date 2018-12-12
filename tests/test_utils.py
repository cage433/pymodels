import numpy as np


def random_times(rng, n_times):
    times = np.zeros((n_times,), float)
    times[0] = rng.random() * 0.1
    for i in range(1, n_times):
        times[i] = times[i - 1] + rng.random() * 0.1
    return times
