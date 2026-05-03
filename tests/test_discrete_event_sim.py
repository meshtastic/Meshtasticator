import random
import unittest

import lib.discrete_event_sim

class TestDiscreteEventSim(unittest.TestCase):
    '''manually replicate a 10-node default configuration discrete sim test as
    if executing `loraMesh.py 10`. Set up the config to match our previous
    known good test run, run the sim, then check against some hardcoded
    results from a previous known good test run.

    This will make it easier to make big changes and make sure the behavior
    of the sim doesn't change. Or if the prior behavior was mistaken or
    incorrect, we can update this test.
    '''
    # TODO: add many more tests for SimulationResults, especially finalize method

    def test_simulation_results_finalization(self):
        """SimulationResults is a glorified dictionary, but the finalize
        method does a notable number of simple calculations and makes a
        notable number of assumptions on keys that exist. Do some
        rudimentary testing with mock types, since it expects lists of
        MeshNode and MeshPacket objects.
        """
        from lib.config import CONFIG
        conf = CONFIG

        # nodes must have attributes:
        # - nodeid (int)
        # - usefulPackets (int)
        # - txAirUtilization (float?)
        # - droppedByDelay (int)
        # - isMoving (boolean)
        # - gpsEnabled (boolean)

        # packets must have attributes:
        # (lists which are as long as there are nodes)
        # - collidedAtN list (boolean)
        # - sensedByN list (boolean)
        # - receivedAtN list (boolean)

        # first-order results must have keys:
        # - nodes (list of nodes)
        # - packets (list of packets)
        # - delays (list of ...floats?)
        # - messageSeq - total # of messages
        # - totalPairs (int)
        # - asymmetricLinks (int)
        # - symmetricLinks (int)
        # - noLinks (int)

        # Things which are computed (keys in results):
        # *: conditional on a config setting
        # +: gated by division-by-zero check of some value (may be nan)
        # - potentialReceivers *
        # - sent
        # - nrCollisions
        # - nrSensed
        # - nrReceived
        # - nrUseful
        # - meanDelay
        # - txAirUtilizationRate *+
        # - collisionRate +
        # - nodereach *+
        # - usefulness +
        # - delayDropped
        # - symmetricLinkRate *+
        # - asymmetricLinkRate *+
        # - noLinkRate *+
        # - movingNodes *
        # - gpsEnabled *

        class MockNode:
            def __init__(self, nodeid: int):
                self.nodeid = nodeid
                self.usefulPackets = 0
                self.txAirUtilization = 0.0
                self.droppedByDelay = 0
                self.isMoving = False
                self.gpsEnabled = False

        class MockPacket:
            def __init__(self, num_nodes: int):
                self.collidedAtN = [False for _ in range(num_nodes)]
                self.sensedByN = [False for _ in range(num_nodes)]
                self.receivedAtN = [False for _ in range(num_nodes)]

        # mock situation: 3 nodes who can all mutually see each other, no DMs,
        # moving nodes, asymmetric links (default config)
        # (complete graph. Triangle)
        # 10 messages and 10 packets
        # I probably won't make this perfect, but want some basic numbers
        conf.NR_NODES = 3
        mock_nodes = [MockNode(i) for i in range(3)]
        mock_nodes[0].isMoving = True
        mock_nodes[0].gpsEnabled = True
        for n in mock_nodes:
            # just put some non-zero values in there
            n.usefulPackets = 10
            n.txAirUtilization = 1.0

        mock_packets = [MockPacket(3) for i in range(10)]
        # all packets were sensed by all nodes, no collisions (fudging it)
        for p in mock_packets:
            for i in range(3):
                p.sensedByN[i] = True
                p.receivedAtN[i] = True

        r = {}
        r['nodes'] = mock_nodes
        r['packets'] = mock_packets
        r['delays'] = [1.0 for _ in range(10)]
        r['messageSeq'] = 10 # total # of messages (not packets)

        # as set up, totalPairs = symmetricLinks + asymmetricLinks + noLinks
        r['totalPairs'] = 3
        r['asymmetricLinks'] = 0
        r['symmetricLinks'] = 0
        r['noLinks'] = 0

        sim_results = lib.discrete_event_sim.SimulationResults(r)
        sim_results.finalize(conf)

        # test computations done by finalize, sanity checks

        # keys exist AND are specific good values
        self.assertEqual(sim_results['potentialReceivers'], len(mock_packets) * (conf.NR_NODES - 1), "expected calculation of potential receivers (no DMs)")
        self.assertEqual(sim_results['sent'], len(mock_packets), 'expected calculation of sent packets')
        self.assertEqual(sim_results['nrCollisions'], 0, 'expected nr of collisions')
        self.assertEqual(sim_results['nrSensed'], 30, 'expected nr of sensed packets')
        self.assertEqual(sim_results['nrReceived'], 30, 'expected nr of received packets')
        self.assertEqual(sim_results['nrUseful'], 30, 'expected nr of useful packets')
        self.assertEqual(sim_results['meanDelay'], 1.0, 'expected mean delay')
        self.assertEqual(sim_results['collisionRate'], 0, 'expected calculated collisionRate')
        self.assertEqual(sim_results['usefulness'], 1, 'usefulness is created')
        self.assertEqual(sim_results['delayDropped'], 0, 'expected number of delayDropped')

        # keys exist, not currently checking values
        self.assertIsNotNone(sim_results['txAirUtilizationRate'], 'txAirUtilizationRate is created')
        self.assertIsNotNone(sim_results['nodeReach'], 'nodeReach is created')
        #self.assertIsNotNone(sim_results['x'], 'x is created')

        # check rate calculations in [0, 1] (assuming we mocked sane values)
        self.assertLessEqual(0.0, sim_results['asymmetricLinkRate'], 'calculated asymmetricLinkRate is above or equal to 0')
        self.assertLessEqual(sim_results['asymmetricLinkRate'], 1.0, 'calculated asymmetricLinkRate is below or equal to 1')
        self.assertLessEqual(0.0, sim_results['symmetricLinkRate'], 'calculated symmetricLinkRate is above or equal to 0')
        self.assertLessEqual(sim_results['symmetricLinkRate'], 1.0, 'calculated symmetricLinkRate is below or equal to 1')
        self.assertLessEqual(0.0, sim_results['noLinkRate'], 'calculated noLinkRate is above or equal to 0')
        self.assertLessEqual(sim_results['noLinkRate'], 1.0, 'calculated noLinkRate is below or equal to 1')

        # expect only 1 moving node with gps enabled
        self.assertEqual(sim_results['movingNodes'], 1, 'expected number of moving nodes')
        self.assertEqual(sim_results['gpsEnabled'], 1, 'expected number of gps enabled nodes')


    # TODO: add default-skip GUI test?
    def test_discrete_sim_ten_nodes(self):
        import numpy as np

        from lib.node import default_generate_node_list

        from lib.config import CONFIG
        conf = CONFIG

        # crucial!! and perhaps a tad fragile
        random.seed(conf.SEED)

        self.assertEqual(conf.SEED, 44, "expected default seed for rng")

        # imitate parse_params
        conf.NR_NODES = 10
        conf.update_router_dependencies()
        nodeConfig = default_generate_node_list(conf)
        # skipping GUI graphing to speed things up

        # set up sim
        sim = lib.discrete_event_sim.DiscreteEventSim(conf, nodeConfig)
        sim.run_simulation()

        # collect & unpack results for easy copy/paste of asserts
        results = sim.get_results()

        # put "first order" results in local scope for easy access
        packets = results["packets"]
        packetsAtN = results["packetsAtN"]
        messageSeq = results["messageSeq"]
        messages = results["messages"]
        delays = results["delays"]
        totalPairs = results["totalPairs"]
        symmetricLinks = results["symmetricLinks"]
        asymmetricLinks = results["asymmetricLinks"]
        noLinks = results["noLinks"]
        nodes = results["nodes"]

        # Begin actual tests, comparing against a hardcoded 'known
        # good' run. If these fail then a change has impacted the
        # results a simulation produces. This could be unintended and
        # a bug, it could be a known consequence of a default config
        # change, or it could be because of an improvement or
        # correction to the sim. Whether to keep these hardcoded values
        # and modify your changes, or to update the hardcoded "known good"
        # simulation results is up to your judgement for which is
        # appropriate. Be cautious!
        self.assertEqual(messageSeq, 183, "expected number of messages created")
        sent = results['sent']
        potentialReceivers = results['potentialReceivers']
        self.assertEqual(sent, 895, "expected number of packets sent")
        self.assertEqual(potentialReceivers, 8055, "expected number of potential receivers")

        nrCollisions = results['nrCollisions']
        self.assertEqual(nrCollisions, 332, "expected number of collisions")
        nrSensed = results['nrSensed']
        self.assertEqual(nrSensed, 3173, "expected number of packets sensed")

        nrReceived = results['nrReceived']
        self.assertEqual(nrReceived, 2824, "expected number of packets received")
        meanDelay = results['meanDelay']
        self.assertEqual(round(meanDelay, 2), 11174.95, "expected rounded delay average")
        txAirUtilizationRate = results['txAirUtilizationRate']
        self.assertEqual(round(txAirUtilizationRate * 100, 2), 5.15, "expected rounded average tx air utilization")

        nodeReach = results['nodeReach']
        self.assertEqual(round(nodeReach*100, 2), 85.55, "expected rounded percentage of nodes reached")

        usefulness = results['usefulness']
        self.assertEqual(round(usefulness*100, 2), 49.89, "expected rounded 'usefulness' percentage")

        delayDropped = results['delayDropped']
        self.assertEqual(delayDropped, 1280, "expected number of packets dropped")
        # default config has both asymmetric links and movement enabled
        asymmetricLinkRate = results['asymmetricLinkRate']
        self.assertEqual(round(asymmetricLinkRate * 100, 2), 8.89, "expected rounded percentage of asymmetric links")
        symmetricLinkRate = results['symmetricLinkRate']
        self.assertEqual(round(symmetricLinkRate * 100, 2), 42.22, "expected rounded percentage of symmetric links")
        noLinkRate = results['noLinkRate']
        self.assertEqual(round(noLinkRate * 100, 2), 48.89, "expected rounded percentage of 'no' links")

        movingNodes = results['movingNodes']
        self.assertEqual(movingNodes, 4, "expected number of moving nodes")

        gpsEnabled = results['gpsEnabled']
        self.assertEqual(gpsEnabled, 1, "expected number of nodes with GPS")

if __name__ == '__main__':
    unittest.main()
