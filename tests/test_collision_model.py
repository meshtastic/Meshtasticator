import unittest

from lib.config import Config
from lib.phy import check_collision, frequency_collision


class FakePacket:
    def __init__(self, seq, start, end, rssi, sf=11, bw=250e3, freq=869.5e6, sensed=True):
        self.seq = seq
        self.txNodeId = seq
        self.startTime = start
        self.endTime = end
        self.timeOnAir = end - start
        self.freq = freq
        self.bw = bw
        self.sf = sf
        self.rssiAtN = [rssi]
        self.collidedAtN = [False]
        self.collisionReasonAtN = [None]
        self.sensedByN = [sensed]


class TestCaptureCollisionModel(unittest.TestCase):
    def config(self):
        conf = Config()
        conf.NR_NODES = 1
        conf.CAPTURE_COLLISION_MODEL_ENABLED = True
        conf.COLLISION_DUE_TO_INTERFERENCE = False
        return conf

    def test_equal_power_preamble_overlap_loses_both_packets(self):
        conf = self.config()
        existing = FakePacket(1, 0, 1000, -90)
        incoming = FakePacket(2, 100, 1100, -91)

        collided = check_collision(conf, None, incoming, 0, [[existing]])

        self.assertEqual(collided, 1)
        self.assertTrue(incoming.collidedAtN[0])
        self.assertTrue(existing.collidedAtN[0])
        self.assertEqual(incoming.collisionReasonAtN[0], "capture_overlap")

    def test_stronger_packet_captures_weaker_overlap(self):
        conf = self.config()
        existing = FakePacket(1, 0, 1000, -92)
        incoming = FakePacket(2, 100, 1100, -80)

        collided = check_collision(conf, None, incoming, 0, [[existing]])

        self.assertEqual(collided, 0)
        self.assertFalse(incoming.collidedAtN[0])
        self.assertTrue(existing.collidedAtN[0])

    def test_small_late_tail_does_not_destroy_locked_packet(self):
        conf = self.config()
        existing = FakePacket(1, 0, 1000, -90)
        incoming = FakePacket(2, 950, 1950, -91)

        collided = check_collision(conf, None, incoming, 0, [[existing]])

        self.assertEqual(collided, 1)
        self.assertFalse(existing.collidedAtN[0])
        self.assertTrue(incoming.collidedAtN[0])

    def test_undecodable_packet_can_jam_without_becoming_collision_casualty(self):
        conf = self.config()
        existing = FakePacket(1, 0, 1000, -90, sensed=True)
        incoming = FakePacket(2, 100, 1100, -91, sensed=False)

        collided = check_collision(conf, None, incoming, 0, [[existing]])

        self.assertEqual(collided, 0)
        self.assertTrue(existing.collidedAtN[0])
        self.assertFalse(incoming.collidedAtN[0])

    def test_frequency_collision_uses_bandwidth_on_either_packet(self):
        narrow = FakePacket(1, 0, 1000, -90, bw=125e3, freq=869.500e6)
        wide = FakePacket(2, 0, 1000, -90, bw=500e3, freq=869.610e6)

        self.assertTrue(frequency_collision(narrow, wide))

    def test_frequency_collision_normalizes_hz_and_khz_style_fields(self):
        hz_left = FakePacket(1, 0, 1000, -90, bw=250e3, freq=869.500e6)
        hz_right = FakePacket(2, 0, 1000, -90, bw=250e3, freq=869.550e6)
        hz_far = FakePacket(5, 0, 1000, -90, bw=250e3, freq=869.570e6)
        mhz_left = FakePacket(3, 0, 1000, -90, bw=250, freq=869.500)
        mhz_right = FakePacket(4, 0, 1000, -90, bw=250, freq=869.550)

        self.assertTrue(frequency_collision(hz_left, hz_right))
        self.assertFalse(frequency_collision(hz_left, hz_far))
        self.assertTrue(frequency_collision(mhz_left, mhz_right))


if __name__ == "__main__":
    unittest.main()
