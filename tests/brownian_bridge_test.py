import unittest

from models.brownian_bridge import BrownianBridge
from models.sobol_generator import SobolGenerator

from scipy.stats import norm

class BrownianBridgeTests(unittest.TestCase):

    def test_spike(self):
        n_paths = 1024
        uniforms = SobolGenerator(1).generate(n_paths)

        normal = map(norm.ppf, uniforms[0])
        for n in normal:
            print(n)

        bb = BrownianBridge(uniforms)
