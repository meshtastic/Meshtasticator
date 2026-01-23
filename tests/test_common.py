import unittest

import lib.common

class TestCommonFunctions(unittest.TestCase):

    def test_calc_dist(self):
        message = "sanity-checking our euclidean distance calculation"
        # test some pythagorean triple triangles https://en.wikipedia.org/wiki/Pythagorean_triple
        # (3, 4, 5)
        # x diff: 3
        # y diff: 4
        p1 = (-1, -1)
        p2 = (2, 3)
        self.assertEqual(lib.common.calc_dist(p1[0], p2[0], p1[1], p2[1]), 5.0, message)

        # (5, 12, 13)
        # x diff: 5
        # y diff: 12
        p1 = (-1, -1)
        p2 = (4, 11)
        self.assertEqual(lib.common.calc_dist(p1[0], p2[0], p1[1], p2[1]), 13.0, message)

        # test some pythagorean quadruple cuboids https://en.wikipedia.org/wiki/Pythagorean_quadruple
        # (1, 2, 2, 3)
        # x diff: 1
        # y diff: 2
        # z diff: 2
        p1 = (-1, -1, -1)
        p2 = (0, 1, 1)
        self.assertEqual(lib.common.calc_dist(p1[0], p2[0], p1[1], p2[1], p1[2], p2[2]), 3.0, message)

        # (2, 3, 6, 7)
        # x diff: 2
        # y diff: 3
        # z diff: 6
        p1 = (-1, -1, -1)
        p2 = (1, 2, 5)
        self.assertEqual(lib.common.calc_dist(p1[0], p2[0], p1[1], p2[1], p1[2], p2[2]), 7.0, message)

if __name__ == '__main__':
    unittest.main()
