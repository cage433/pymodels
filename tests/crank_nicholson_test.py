import unittest

from models.crank_nicholson import CNSolver
from models.option import ExerciseStyle
from tests.black_scholes_test import random_bs
from tests.pimpedrandom import PimpedRandom

class CrankNicholsonTest(unittest.TestCase):
    def test_matches_bs(self):
        rng = PimpedRandom()
        rng.seed(1234)
        bs = random_bs(rng)
        print(bs)
        solver = CNSolver(n = 20, std_devs=4.0, n_times=6)
        numeric_value = solver.solve(bs.F, bs.K, bs.sigma, bs.T, bs.right, ExerciseStyle.EUROPEAN)
        analytic_value = bs.undiscounted_value()
        self.assertAlmostEqual(numeric_value, analytic_value, delta = 1e-6)

if __name__ == '__main__':
    unittest.main()

