import unittest
import numpy as np

from scipy.stats.stats import pearsonr

class StatsTestMixin(unittest.TestCase):

    def check_uncorrelated(self, arr1, arr2, tol, msg=""):
        self.assertAlmostEqual(
            0.0,
            pearsonr(arr1, arr2)[0],
            delta=tol,
            msg=msg
        )

    def check_mean(self, arr, expected, tol, msg=""):
        self.assertAlmostEqual(
            expected,
            np.asscalar(np.mean(arr)),
            delta=tol,
            msg=msg
        )

    def check_std_dev(self, arr, expected, tol, msg=""):
        self.assertAlmostEqual(
            expected,
            np.asscalar(np.std(arr)),
            delta=tol,
            msg=msg
        )
