import unittest

from lib.config import Config
import lib.phy

class TestPhy(unittest.TestCase):

    def test_path_loss_estimator(self):
        # make sure we reject invalid model selection integers
        from lib.config import CONFIG
        conf = CONFIG

        model = -1 # invalid model
        self.assertRaises(ValueError, lib.phy.estimate_path_loss, conf, 50, 915, 3, 3, model)
        model = 7 # invalid model
        self.assertRaises(ValueError, lib.phy.estimate_path_loss, conf, 50, 915, 3, 3, model)
        model = 10 # invalid model
        self.assertRaises(ValueError, lib.phy.estimate_path_loss, conf, 50, 915, 3, 3, model)
        model = -10 # invalid model
        self.assertRaises(ValueError, lib.phy.estimate_path_loss, conf, 50, 915, 3, 3, model)

        # TODO: hardcode some expected values for the calculations across different
        # models, to detect unintended changes. This also requires verifying
        # the calculations are correct.

    def test_rootFinder(self):
        # double-check we can find the roots of some polynomials
        message = "sanity-check Newton-Raphson root-finding implementation"
        tolerance = 0.0000001

        def poly1(x):
            ''' roots at x=-3, 0, 2.5 '''
            return (x+3)*(x-2.5)*x

        # should find -3
        res = lib.phy.rootFinder(poly1, -3.5, tol=tolerance)
        diff = abs(res - -3)
        self.assertLess(diff, tolerance, message)

        # should find 0
        res = lib.phy.rootFinder(poly1, -1, tol=tolerance)
        diff = abs(res - 0)
        self.assertLess(diff, tolerance, message)

        # should find 2.5
        res = lib.phy.rootFinder(poly1, 3, tol=tolerance)
        diff = abs(res - 2.5)
        self.assertLess(diff, tolerance, message)

    def test_path_loss_distance_floor_keeps_near_field_calibrated(self):
        conf = Config()
        conf.PATH_LOSS_DISTANCE_FLOOR_M = 780.0

        below_floor = lib.phy.estimate_path_loss(conf, 10.0, conf.FREQ)
        at_floor = lib.phy.estimate_path_loss(conf, 780.0, conf.FREQ)

        self.assertAlmostEqual(below_floor, at_floor)

    def test_estimate_path_loss_accepts_explicit_model(self):
        conf = Config()
        dist = 1500
        freq = conf.FREQ

        explicit = lib.phy.estimate_path_loss(conf, dist, freq, model=0)

        self.assertEqual(conf.MODEL, 5, "explicit model must not mutate config")
        conf.MODEL = 0
        implicit = lib.phy.estimate_path_loss(conf, dist, freq)
        self.assertAlmostEqual(explicit, implicit)

    def test_estimate_path_loss_rejects_unsupported_model(self):
        conf = Config()

        with self.assertRaisesRegex(ValueError, "unsupported path loss model"):
            lib.phy.estimate_path_loss(conf, 1500, conf.FREQ, model=99)


if __name__ == '__main__':
    unittest.main()
