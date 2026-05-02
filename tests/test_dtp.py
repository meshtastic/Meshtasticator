import unittest

import simpy

from lib.config import Config
from lib.discrete_event_sim_components import SimulationDataTracking, SimulationState
from lib.dtp import choose_dynamic_tx_power
from lib.node import MeshNode, NodeConfig
from lib.packet import MeshPacket, NODENUM_BROADCAST
from lib.point import Point


class FakePacket:
    def __init__(
        self,
        cr=5,
        is_ack=False,
        retransmissions=3,
        tx_node_id=0,
        orig_tx_node_id=0,
        hop_limit=3,
        dest_id=NODENUM_BROADCAST,
        prior_hop_rssi=None,
        prior_hop_snr=None,
        base_power=30,
    ):
        self.cr = cr
        self.isAck = is_ack
        self.retransmissions = retransmissions
        self.txNodeId = tx_node_id
        self.origTxNodeId = orig_tx_node_id
        self.hopLimit = hop_limit
        self.destId = dest_id
        self.priorHopRssi = prior_hop_rssi
        self.priorHopSnr = prior_hop_snr
        self.baseTxPower = base_power


class FakeTransmitter:
    def __init__(self, queue_depth=0):
        self.queue = [object()] * queue_depth


class FakeNode:
    def __init__(self, util=0.0, queue_depth=0):
        self.conf = Config()
        self.conf.DTP_ENABLED = True
        self._util = util
        self.transmitter = FakeTransmitter(queue_depth)
        self.txAirUtilization = 0.0
        self.prevTxAirUtilization = 0.0

    def channel_utilization_percent(self):
        return self._util


class TestDynamicTxPower(unittest.TestCase):
    def test_dtp_disabled_keeps_base_power(self):
        node = FakeNode()
        node.conf.DTP_ENABLED = False

        decision = choose_dynamic_tx_power(node, FakePacket(base_power=27))

        self.assertEqual(decision.tx_power_dbm, 27)
        self.assertEqual(decision.reason, "dtp_off")

    def test_origin_packet_stays_at_max_power(self):
        node = FakeNode(util=30.0)

        decision = choose_dynamic_tx_power(node, FakePacket(tx_node_id=1, orig_tx_node_id=1, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 30)
        self.assertIn("max_power_origin", decision.reason)

    def test_busy_relay_lowers_power(self):
        node = FakeNode(util=12.0)

        decision = choose_dynamic_tx_power(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 24)
        self.assertIn("busy_relay_power_drop", decision.reason)

    def test_congested_relay_lowers_power_more(self):
        node = FakeNode(util=20.0)

        decision = choose_dynamic_tx_power(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 21)

    def test_direct_relay_without_strong_prior_hop_stays_max_power(self):
        node = FakeNode(util=12.0)

        decision = choose_dynamic_tx_power(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, dest_id=7, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 30)
        self.assertIn("max_power_direct_relay_without_strong_link", decision.reason)

    def test_strong_direct_relay_only_gets_small_drop(self):
        node = FakeNode(util=20.0)
        prior_rssi = node.conf.current_preset["sensitivity"] + node.conf.DTP_STRONG_LINK_MARGIN_DB

        decision = choose_dynamic_tx_power(
            node,
            FakePacket(tx_node_id=2, orig_tx_node_id=1, dest_id=7, prior_hop_rssi=prior_rssi, base_power=30),
        )

        self.assertEqual(decision.tx_power_dbm, 27)
        self.assertIn("direct_relay_cap", decision.reason)

    def test_prior_hop_strength_is_not_absolute_snr(self):
        node = FakeNode(util=20.0)

        decision = choose_dynamic_tx_power(
            node,
            FakePacket(tx_node_id=2, orig_tx_node_id=1, dest_id=7, prior_hop_snr=6.0, base_power=30),
        )

        self.assertEqual(decision.tx_power_dbm, 30)
        self.assertIn("max_power_direct_relay_without_strong_link", decision.reason)

    def test_final_retry_uses_max_power(self):
        node = FakeNode(util=0.0)

        decision = choose_dynamic_tx_power(
            node,
            FakePacket(tx_node_id=2, orig_tx_node_id=1, retransmissions=1, prior_hop_snr=8.0, base_power=30),
        )

        self.assertEqual(decision.tx_power_dbm, 30)
        self.assertIn("max_power_retry_rescue", decision.reason)

    def test_cr8_packet_uses_max_power_even_when_not_retry(self):
        node = FakeNode(util=20.0)

        decision = choose_dynamic_tx_power(node, FakePacket(cr=8, tx_node_id=2, orig_tx_node_id=1, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 30)
        self.assertIn("max_power_retry_rescue", decision.reason)

    def test_power_drop_respects_step_and_minimum(self):
        node = FakeNode(util=20.0)
        node.conf.DTP_MAX_POWER_DROP_DB = 8
        node.conf.DTP_POWER_STEP_DB = 3
        node.conf.DTP_MIN_TX_POWER_DBM = 24

        decision = choose_dynamic_tx_power(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 24)

    def test_minimum_power_clamp_cannot_boost_above_base_power(self):
        node = FakeNode(util=20.0)
        node.conf.DTP_MIN_TX_POWER_DBM = 35

        decision = choose_dynamic_tx_power(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, base_power=30))

        self.assertEqual(decision.tx_power_dbm, 30)
        self.assertIn("drop=0dB", decision.reason)


class TestDynamicTxPowerPacketPhysics(unittest.TestCase):
    def make_nodes(self, distance_m):
        conf = Config()
        conf.NR_NODES = 2
        conf.PTX = 30
        conf.MODEL_ASYMMETRIC_LINKS = False
        conf.LINK_OFFSET = {(0, 1): 0, (1, 0): 0}
        env = simpy.Environment()
        sim_state = SimulationState(conf, env)
        tracking = SimulationDataTracking()
        nodes = [
            MeshNode(conf, sim_state, tracking, NodeConfig(0, Point(0, 0, 1.5), conf.PERIOD)),
            MeshNode(conf, sim_state, tracking, NodeConfig(1, Point(distance_m, 0, 1.5), conf.PERIOD)),
        ]
        sim_state.nodes.extend(nodes)
        return conf, nodes

    def test_lower_tx_power_recomputes_receiver_visibility(self):
        conf, nodes = self.make_nodes(2_000)
        connectivity_map = {0: {1}, 1: {0}}
        baseline_pathloss_matrix = [[None, None], [None, None]]
        packet = MeshPacket(
            conf,
            nodes,
            0,
            NODENUM_BROADCAST,
            0,
            40,
            1,
            0,
            True,
            False,
            None,
            0,
            connectivity_map,
            baseline_pathloss_matrix,
        )

        self.assertTrue(packet.sensedByN[1])

        packet.set_tx_power(18)

        self.assertFalse(packet.sensedByN[1])
        self.assertLess(packet.rssiAtN[1], conf.current_preset["sensitivity"])


if __name__ == "__main__":
    unittest.main()
