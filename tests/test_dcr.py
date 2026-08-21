import unittest

from lib.config import Config
from lib.dcr import CR_NORMAL, CR_RESCUE, CR_SLIM, choose_dynamic_coding_rate
from lib.packet import NODENUM_BROADCAST


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
    ):
        self.cr = cr
        self.isAck = is_ack
        self.retransmissions = retransmissions
        self.txNodeId = tx_node_id
        self.origTxNodeId = orig_tx_node_id
        self.hopLimit = hop_limit
        self.destId = dest_id

    def airtime_for_cr(self, cr):
        return {5: 100.0, 6: 120.0, 7: 140.0, 8: 160.0}[cr]


class FakeTransmitter:
    def __init__(self, queue_depth=0):
        self.queue = [object()] * queue_depth


class FakeNode:
    def __init__(self, util=0.0, queue_depth=0):
        self.conf = Config()
        self.conf.DCR_ENABLED = True
        self._util = util
        self.transmitter = FakeTransmitter(queue_depth)
        self.dcrAirtimeByCr = {5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}
        self.txAirUtilization = 0.0
        self.prevTxAirUtilization = 0.0

    def channel_utilization_percent(self):
        return self._util


class TestDynamicCodingRate(unittest.TestCase):
    def test_dcr_disabled_keeps_packet_cr(self):
        node = FakeNode()
        node.conf.DCR_ENABLED = False
        packet = FakePacket(cr=7)

        decision = choose_dynamic_coding_rate(node, packet)

        self.assertEqual(decision.cr, 7)
        self.assertEqual(decision.reason, "dcr_off")

    def test_idle_user_first_attempt_stays_compact_cr(self):
        node = FakeNode(util=0.0)

        decision = choose_dynamic_coding_rate(node, FakePacket())

        self.assertEqual(decision.cr, CR_SLIM)
        self.assertIn("idle_no_first_attempt_bump", decision.reason)

    def test_idle_user_retry_gets_normal_cr(self):
        node = FakeNode(util=0.0)

        decision = choose_dynamic_coding_rate(node, FakePacket(retransmissions=2))

        self.assertEqual(decision.cr, CR_NORMAL)

    def test_busy_user_packet_can_use_compact_cr(self):
        node = FakeNode(util=12.0)

        decision = choose_dynamic_coding_rate(node, FakePacket())

        self.assertEqual(decision.cr, CR_SLIM)

    def test_current_bucket_airtime_contributes_to_busy_pressure(self):
        node = FakeNode(util=0.0)
        node.txAirUtilization = 6000.0

        decision = choose_dynamic_coding_rate(node, FakePacket(retransmissions=2))

        self.assertEqual(decision.cr, CR_SLIM)
        self.assertIn("channel_busy", decision.reason)

    def test_direct_origin_packet_can_stay_compact_cr(self):
        node = FakeNode(util=0.0)

        decision = choose_dynamic_coding_rate(node, FakePacket(dest_id=7))

        self.assertEqual(decision.cr, CR_SLIM)

    def test_nonbusy_direct_relay_minimum_is_normal_cr(self):
        node = FakeNode(util=5.0)

        decision = choose_dynamic_coding_rate(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, dest_id=7))

        self.assertEqual(decision.cr, CR_NORMAL)
        self.assertIn("direct_relay_min_cr", decision.reason)

    def test_busy_direct_relay_can_stay_compact_cr(self):
        node = FakeNode(util=12.0)

        decision = choose_dynamic_coding_rate(node, FakePacket(tx_node_id=2, orig_tx_node_id=1, dest_id=7))

        self.assertEqual(decision.cr, CR_SLIM)
        self.assertNotIn("direct_relay_min_cr", decision.reason)

    def test_retry_does_not_escalate_on_unrestricted_region_magic_limit(self):
        node = FakeNode(util=12.0)
        node.conf.REGION = node.conf.regions["US"]

        decision = choose_dynamic_coding_rate(node, FakePacket(retransmissions=2))

        self.assertEqual(decision.cr, CR_SLIM)
        self.assertIn("channel_busy", decision.reason)

    def test_quiet_final_retry_can_use_rescue_cr(self):
        node = FakeNode(util=0.0)
        node.conf.DCR_CR8_AIRTIME_LIMIT_PERCENT = 100.0

        decision = choose_dynamic_coding_rate(node, FakePacket(retransmissions=1))

        self.assertEqual(decision.cr, CR_RESCUE)

    def test_ack_minimum_is_normal_even_when_busy(self):
        node = FakeNode(util=12.0)

        decision = choose_dynamic_coding_rate(node, FakePacket(is_ack=True))

        self.assertEqual(decision.cr, CR_NORMAL)

    def test_ack_respects_user_minimum_cr(self):
        node = FakeNode(util=12.0)
        node.conf.DCR_USER_MIN_CR = 7

        decision = choose_dynamic_coding_rate(node, FakePacket(is_ack=True))

        self.assertEqual(decision.cr, 7)

    def test_last_hop_relay_uses_normal_cr_without_retry_evidence(self):
        node = FakeNode(util=5.0)
        packet = FakePacket(tx_node_id=2, orig_tx_node_id=1, hop_limit=1)

        decision = choose_dynamic_coding_rate(node, packet)

        self.assertEqual(decision.cr, CR_NORMAL)
        self.assertIn("last_hop", decision.reason)


if __name__ == "__main__":
    unittest.main()
