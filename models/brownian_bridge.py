import numpy as np

from scipy.stats import norm


class BrownianBridge:
    def __init__(self, times):
        n_times = len(times)

        step_map = np.zeros((n_times,), int)
        left_index = np.zeros((n_times,), int)
        bridge_index = np.zeros((n_times,), int)
        right_index = np.zeros((n_times,), int)

        left_weight = np.zeros((n_times,), float)
        right_weight = np.zeros((n_times,), float)
        stddev = np.zeros((n_times,), float)

        step_map[n_times - 1] = 1
        bridge_index[0] = n_times - 1
        stddev[0] = np.sqrt(times[n_times - 1])
        left_weight[0] = 0.0
        right_weight[0] = 0.0

        j = 0
        k = 0
        l = 0

        def index_where(pred, l, i_start):
            for i, v in enumerate(l[i_start:]):
                if pred(v):
                    return i + i_start
            return -1

        def is_zero(n):
            return n == 0

        def is_non_zero(n):
            return n != 0

        for i in range(1, n_times):
            j = index_where(is_zero, step_map, j)
            k = j
            k = index_where(is_non_zero, step_map, k)
            l = j + ((k - 1 - j) >> 1)

            step_map[l] = i
            bridge_index[i] = l
            left_index[i] = j
            right_index[i] = k

            if j > 0:

                if times[k] == times[j - 1]:
                    left_weight[i] = 1.0
                else:
                    left_weight[i] = (times[k] - times[l]) / (times[k] - times[j - 1])
                stddev[i] = np.sqrt[[times[l] - times[j - 1]] * left_weight[i]]
            else:
                if times[k] == 0.0:
                    left_weight[i] = 1.0
                else:
                    left_weight[i] = (times[k] - times[l]) / times[k]
                stddev[i] = np.sqrt[times[l] * left_weight[i]]

            right_weight[i] = 1.0 - left_weight[i]

            j = k + 1
            if j >= n_times:
                j = 0


    def generate(self, uniforms):
        n_variables = len(uniforms)
        normals = [map(norm.ppf, x) for x in uniforms]
