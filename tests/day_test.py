import unittest
from datetime import timedelta

from models.day import Day


class RichDateTest(unittest.TestCase):

    def test_increment(self):

        self.assertEqual(
            Day(2000, 1, 1) + 1,
            Day(2000, 1, 2)
        )
        self.assertEqual(
            Day(2000, 2, 28) + 2,
            Day(2000, 3, 1)
        )
        self.assertEqual(
            Day(2000, 2, 28) + timedelta(2),
            Day(2000, 3, 1)
        )
        self.assertEqual(
            Day(2000, 2, 28) - 2,
            Day(2000, 2, 26)
        )
        self.assertEqual(
            Day(2000, 2, 28) - 0,
            Day(2000, 2, 28)
        )

    def test_time_since(self):
        d1 = Day(2001, 1, 1)
        d2 = d1 + 10
        self.assertAlmostEqual(
            d2.time_since(d1),
            10.0 / 365.25,
            1e03
        )
