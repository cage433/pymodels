import numpy as np

from models.brownian_bridge import BrownianBridge
from models.sobol_generator import SobolGenerator


def brownians_from_uniforms(uniforms, n_variables, times):
    n_paths = len(uniforms[0])
    n_times = len(times)

    if uniforms.shape != (n_variables * n_times, n_paths):
        raise Exception("Shape for brownians doesn't match")

    brownians = np.zeros((n_paths, n_variables, n_times))
    bridge = BrownianBridge(times)

    var_range = range(n_variables)
    for i_path in range(n_paths):
        for i_var in var_range:
            brownians[i_path, i_var, :] = bridge.generate(uniforms[i_var::n_variables, i_path])

    return brownians


def generate_brownians(n_paths: int, n_variables: int, times):
    n_times = len(times)
    uniforms = SobolGenerator(n_variables * n_times).generate(n_paths)
    return brownians_from_uniforms(uniforms, n_variables, times)
