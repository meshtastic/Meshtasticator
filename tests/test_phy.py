import unittest

import lib.phy

class TestPhy(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
