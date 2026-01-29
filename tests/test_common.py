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

    def test_find_random_position(self):
        # mock up the needed objects
        # conf: config from lib.config.Config(). Must have
        # - XSIZE, YSIZE, OX, OY, MINDIST, FREQ, PTX, GL, current_preset property
        # - MODEL, LPLD0, GAMMA, D0
        # (just use an actual config object)
        # nodes: empty list OR list of nodes which must have:
        #  - x, y attributes
        from lib.config import Config
        from lib.phy import estimate_path_loss

        # TODO: iterate this test for each of our supported models, since they
        # change the return value of estimate_path_loss. Also, each LoRa preset
        # has its own sensitivity which changes radio range.
        conf = Config()

        class MyNode:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __repr__(self):
                return f"MyNode(x={self.x}, y={self.y})"

        lower_bound_x = conf.OX - conf.XSIZE/2
        upper_bound_x = conf.OX + conf.XSIZE/2
        lower_bound_y = conf.OY - conf.YSIZE/2
        upper_bound_y = conf.OY + conf.YSIZE/2

        nodes = []
        # conditions that must be held:
        # - found position can 'reach' at least one other node.
        # - found position is not within conf.MINDIST of any other node.
        # - found position is within defined scenario area.
        # - a position is always returned

        # first node case
        position = lib.common.find_random_position(conf, nodes)
        self.assertIsNotNone(position, "always return position")
        self.assertGreaterEqual(position[0], lower_bound_x, f"x within bounds {position=}")
        self.assertLessEqual(position[0], upper_bound_x, f"x within bounds {position=}")
        self.assertGreaterEqual(position[1], lower_bound_y, f"y within bounds {position=}")
        self.assertLessEqual(position[1], upper_bound_y, f"y within bounds {position=}")

        # second node case
        n = MyNode(0, 0)
        nodes = [n]
        position = lib.common.find_random_position(conf, nodes)
        self.assertIsNotNone(position, "always return position")
        self.assertGreaterEqual(position[0], lower_bound_x, f"x within bounds {position=}")
        self.assertLessEqual(position[0], upper_bound_x, f"x within bounds {position=}")
        self.assertGreaterEqual(position[1], lower_bound_y, f"y within bounds {position=}")
        self.assertLessEqual(position[1], upper_bound_y, f"y within bounds {position=}")

        distance = lib.common.calc_dist(n.x, position[0], n.y, position[1])
        self.assertGreaterEqual(distance, conf.MINDIST, f"{position=} not within MINDIST of {n=}")

        # this directly replicates the logic from the function which I dislike.
        # Find a better way to test "found node can reach one other node",
        # perhaps by pre-computing a max distance based on the config params
        # we're using. There are lots of those, but they shouldn't change often.
        pathLoss = estimate_path_loss(conf, distance, conf.FREQ)
        rssi = conf.PTX + 2*conf.GL - pathLoss
        self.assertGreaterEqual(rssi, conf.current_preset["sensitivity"], f"found {position=} is within radio range of {n=}")

if __name__ == '__main__':
    unittest.main()
