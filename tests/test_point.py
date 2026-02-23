import unittest

import lib.point

class TestPointClass(unittest.TestCase):

    def test_update_xy(self):
        p = lib.point.Point(1, 2, 3)

        self.assertEqual(p.x, 1, "sanity check constructor")
        self.assertEqual(p.y, 2, "sanity check constructor")
        self.assertEqual(p.z, 3, "sanity check constructor")

        p.update_xy(4, 5)
        self.assertEqual(p.x, 4, "sanity check update x coordinate")
        self.assertEqual(p.y, 5, "sanity check update y coordinate")
        self.assertEqual(p.z, 3, "sanity check z coordinate unchanged")

    def test_euclidean_distance(self):
        # replicate test_calc_dist from test_common, but using the Point class
        message = "sanity-checking our euclidean distance calculation"
        # test some pythagorean triple triangles https://en.wikipedia.org/wiki/Pythagorean_triple
        # (3, 4, 5)
        # x diff: 3
        # y diff: 4
        p1 = lib.point.Point(-1, -1, 0)
        p2 = lib.point.Point(2, 3, 0)
        self.assertEqual(p1.euclidean_distance(p2), 5.0, message)
        self.assertEqual(p2.euclidean_distance(p1), 5.0, message+", and commutativity")

        # (5, 12, 13)
        # x diff: 5
        # y diff: 12
        p1 = lib.point.Point(-1, -1, 0)
        p2 = lib.point.Point(4, 11, 0)
        self.assertEqual(p1.euclidean_distance(p2), 13.0, message)
        self.assertEqual(p2.euclidean_distance(p1), 13.0, message+", and commutativity")

        # test some pythagorean quadruple cuboids https://en.wikipedia.org/wiki/Pythagorean_quadruple
        # (1, 2, 2, 3)
        # x diff: 1
        # y diff: 2
        # z diff: 2
        p1 = lib.point.Point(-1, -1, -1)
        p2 = lib.point.Point(0, 1, 1)
        self.assertEqual(p1.euclidean_distance(p2), 3.0, message)
        self.assertEqual(p2.euclidean_distance(p1), 3.0, message+", and commutativity")

        # (2, 3, 6, 7)
        # x diff: 2
        # y diff: 3
        # z diff: 6
        p1 = lib.point.Point(-1, -1, -1)
        p2 = lib.point.Point(1, 2, 5)
        self.assertEqual(p1.euclidean_distance(p2), 7.0, message)
        self.assertEqual(p2.euclidean_distance(p1), 7.0, message+", and commutativity")
