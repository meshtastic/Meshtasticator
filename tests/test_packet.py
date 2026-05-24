import unittest

from lib.config import Config
from lib.packet import MeshPacket, NODENUM_BROADCAST
from lib.point import Point


class PacketNode:
    def __init__(self, nodeid, position):
        self.nodeid = nodeid
        self.position = position
        self.antennaGain = 0
        self.hopLimit = 3


class TestMeshPacket(unittest.TestCase):

    def test_receiver_id_lists_match_receiver_flags(self):
        conf = Config()
        conf.NR_NODES = 3
        conf.LINK_OFFSET = {
            (tx_id, rx_id): 0
            for tx_id in range(conf.NR_NODES)
            for rx_id in range(conf.NR_NODES)
            if tx_id != rx_id
        }
        nodes = [
            PacketNode(0, Point(0, 0, conf.HM)),
            PacketNode(1, Point(100, 0, conf.HM)),
            PacketNode(2, Point(1000, 0, conf.HM)),
        ]
        connectivity_map = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        baseline_pathloss_matrix = [[None for _ in nodes] for _ in nodes]

        packet = MeshPacket(
            conf,
            nodes,
            origTxNodeId=0,
            destId=NODENUM_BROADCAST,
            txNodeId=0,
            plen=conf.PACKETLENGTH,
            seq=1,
            genTime=0,
            wantAck=False,
            isAck=False,
            requestId=None,
            now=0,
            connectivity_map=connectivity_map,
            baseline_pathloss_matrix=baseline_pathloss_matrix,
        )

        self.assertEqual(
            packet.sensed_node_ids,
            [node_id for node_id, sensed in enumerate(packet.sensedByN) if sensed],
        )
        self.assertEqual(
            packet.detected_node_ids,
            [node_id for node_id, detected in enumerate(packet.detectedByN) if detected],
        )
        self.assertNotIn(packet.txNodeId, packet.sensed_node_ids)
        self.assertNotIn(packet.txNodeId, packet.detected_node_ids)

    def test_connectivity_map_skips_are_not_classified_as_sensed(self):
        conf = Config()
        conf.NR_NODES = 3
        conf.MOVEMENT_ENABLED = False
        conf.LINK_OFFSET = {
            (tx_id, rx_id): 0
            for tx_id in range(conf.NR_NODES)
            for rx_id in range(conf.NR_NODES)
            if tx_id != rx_id
        }
        nodes = [
            PacketNode(0, Point(0, 0, conf.HM)),
            PacketNode(1, Point(100, 0, conf.HM)),
            PacketNode(2, Point(150, 0, conf.HM)),
        ]
        connectivity_map = {0: {1}, 1: {0}, 2: set()}
        baseline_pathloss_matrix = [[None for _ in nodes] for _ in nodes]

        packet = MeshPacket(
            conf,
            nodes,
            origTxNodeId=0,
            destId=NODENUM_BROADCAST,
            txNodeId=0,
            plen=conf.PACKETLENGTH,
            seq=1,
            genTime=0,
            wantAck=False,
            isAck=False,
            requestId=None,
            now=0,
            connectivity_map=connectivity_map,
            baseline_pathloss_matrix=baseline_pathloss_matrix,
        )

        self.assertTrue(packet.sensedByN[1])
        self.assertFalse(packet.sensedByN[2])
        self.assertFalse(packet.detectedByN[2])
        self.assertNotIn(2, packet.sensed_node_ids)
        self.assertNotIn(2, packet.detected_node_ids)


if __name__ == "__main__":
    unittest.main()
