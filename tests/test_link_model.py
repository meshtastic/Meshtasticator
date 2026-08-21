import unittest

from lib.config import Config
from lib.link_model import calculate_link_budget
from lib.point import Point
from lib.terrain import TerrainGrid


class DummyNode:
    def __init__(self, nodeid, x, y, gain=0.0, z=1.5, antenna_height=1.5):
        self.nodeid = nodeid
        self.position = Point(x, y, z)
        self.antennaGain = gain
        self.antennaHeight = antenna_height


class TestLinkModel(unittest.TestCase):
    def test_endpoint_antenna_gains_affect_packet_budget(self):
        conf = Config()

        without_gains = calculate_link_budget(conf, DummyNode(1, 0, 0), DummyNode(2, 1000, 0))
        with_gains = calculate_link_budget(conf, DummyNode(1, 0, 0, gain=2.0), DummyNode(2, 1000, 0, gain=3.0))

        # MeshPacket delivery uses both TX and RX antenna gains. The shared link
        # model must keep topology-summary counters on that same budget.
        self.assertAlmostEqual(with_gains.raw_rssi_dbm - without_gains.raw_rssi_dbm, 5.0)
        self.assertAlmostEqual(with_gains.rssi_dbm - without_gains.rssi_dbm, 5.0)

    def test_directed_offset_is_path_loss_not_identity_lookup(self):
        conf = Config()
        baseline = calculate_link_budget(conf, DummyNode(1, 0, 0), DummyNode(2, 1000, 0))
        offset = calculate_link_budget(conf, DummyNode(1, 0, 0), DummyNode(2, 1000, 0), offset_db=4.0)

        self.assertAlmostEqual(offset.path_loss_db - baseline.path_loss_db, 4.0)
        self.assertAlmostEqual(baseline.rssi_dbm - offset.rssi_dbm, 4.0)

    def test_absolute_node_altitude_is_not_used_as_antenna_height(self):
        conf = Config()
        conf.MODEL = 1

        ground_height = calculate_link_budget(
            conf,
            DummyNode(1, 0, 0, z=1.5, antenna_height=1.5),
            DummyNode(2, 1000, 0, z=1.5, antenna_height=1.5),
        )
        absolute_altitude = calculate_link_budget(
            conf,
            DummyNode(1, 0, 0, z=101.5, antenna_height=1.5),
            DummyNode(2, 1000, 0, z=101.5, antenna_height=1.5),
        )

        self.assertAlmostEqual(absolute_altitude.base_path_loss_db, ground_height.base_path_loss_db)

    def test_feature_calibration_applies_to_generated_pairs(self):
        conf = Config()
        conf.LINK_CALIBRATION_MODEL_ENABLED = True
        conf.LINK_CALIBRATION_COEFFICIENTS = {"intercept": -12.0}

        first = calculate_link_budget(conf, DummyNode(1, 0, 0), DummyNode(2, 1000, 0))
        second = calculate_link_budget(conf, DummyNode(9, 0, 0), DummyNode(10, 1000, 0))

        # The calibration is a feature transform, not a lookup keyed by node ID:
        # two generated pairs with the same path features get the same SNR.
        self.assertAlmostEqual(first.snr_db, -12.0)
        self.assertAlmostEqual(second.snr_db, -12.0)

    def test_terrain_loss_flows_through_shared_link_budget(self):
        conf = Config()
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_PROFILE_SAMPLES = 10
        conf.TERRAIN_GRID = TerrainGrid.from_rows([
            (0, 0, 0),
            (500, 0, 120),
            (1000, 0, 0),
        ])

        budget = calculate_link_budget(conf, DummyNode(1, 0, 0), DummyNode(2, 1000, 0))

        self.assertGreater(budget.terrain_loss_db, 0)
        self.assertAlmostEqual(budget.path_loss_db, budget.base_path_loss_db + budget.terrain_loss_db)


if __name__ == "__main__":
    unittest.main()
