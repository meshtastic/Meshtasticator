import unittest

from lib.config import Config
from lib.mac import channel_utilization_percent, get_retransmission_msec


class FakeEnv:
    def __init__(self, now):
        self.now = now


class FakeNode:
    def __init__(self, now, air_utilization):
        self.conf = Config()
        self.env = FakeEnv(now)
        self.airUtilization = air_utilization


class FakePacket:
    sf = 11
    cr = 5
    packetLen = 40
    bw = 250e3


class TestMacTiming(unittest.TestCase):
    def test_channel_utilization_is_bounded_for_early_dense_runs(self):
        node = FakeNode(now=1, air_utilization=10_000)

        self.assertEqual(channel_utilization_percent(node), 100.0)

    def test_retransmission_timeout_does_not_overflow_for_early_dense_runs(self):
        node = FakeNode(now=1, air_utilization=10_000)

        timeout_msec = get_retransmission_msec(node, FakePacket())

        self.assertGreater(timeout_msec, 0)


if __name__ == "__main__":
    unittest.main()
