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
                self.dcrTxByCr = {5: 0, 6: 0, 7: 0, 8: 0}
                self.dcrAirtimeByCr = {5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}
                self.dtpTxByPower = {}
                self.dtpTxByCrPower = {}
                self.dtpDetectedByTx = 0
                self.dtpSensedByTx = 0
                self.dtpTxCount = 0

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

        r['totalPairs'] = 3
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
        self.assertEqual(sim_results['dcrTxByCr'], {5: 0, 6: 0, 7: 0, 8: 0}, 'expected DCR histogram')
        self.assertEqual(sim_results['dcrAirtimeByCr'], {5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}, 'expected DCR airtime histogram')
        self.assertEqual(sim_results['dtpTxByPower'], {}, 'expected DTP power histogram')
        self.assertEqual(sim_results['dtpTxByCrPower'], {}, 'expected DTP CR/power histogram')
        self.assertEqual(sim_results['dtpMeanDetectedByTx'], 0.0, 'expected DTP detected mean')
        self.assertEqual(sim_results['dtpMeanSensedByTx'], 0.0, 'expected DTP sensed mean')

        # keys exist, not currently checking values
        self.assertIsNotNone(sim_results['txAirUtilizationRate'], 'txAirUtilizationRate is created')
        self.assertIsNotNone(sim_results['nodeReach'], 'nodeReach is created')
        #self.assertIsNotNone(sim_results['x'], 'x is created')

        # check rate calculations in [0, 1] (assuming we mocked sane values)
        self.assertLessEqual(0.0, sim_results['noLinkRate'], 'calculated noLinkRate is above or equal to 0')
        self.assertLessEqual(sim_results['noLinkRate'], 1.0, 'calculated noLinkRate is below or equal to 1')

        # expect only 1 moving node with gps enabled
        self.assertEqual(sim_results['movingNodes'], 1, 'expected number of moving nodes')
        self.assertEqual(sim_results['gpsEnabled'], 1, 'expected number of gps enabled nodes')

    def test_connectivity_map_optimization_is_consistent(self):
        from lib.node import default_generate_node_list

        from lib.config import CONFIG
        conf = CONFIG

        all_results = []

        # somewhat lazily test with connectivity map optimization on and off,
        # to make sure the optimization doesn't change any results/the simulation
        # is consistent regardless of this optimization. Further simulation changes
        # that warrant this kind of testing should be very carefully considered,
        # since that leads to exponential growth in configurations to test.
        for enable_optimization in [True, False]:
            # test against optimization being enabled/disabled
            conf.ENABLE_CONNECTIVITY_MAP = enable_optimization

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
            all_results.append(results)

        # look at just specific simulation results for now. May go as deep as
        # comparing MeshPacket objects later if that seems useful and we feel
        # like adding comparison functions to those objects.
        facets = [
            'potentialReceivers',
            'sent',
            'nrCollisions',
            'nrSensed',
            'nrReceived',
            'nrUseful',
            'meanDelay',
            'txAirUtilizationRate',
            'collisionRate',
            'nodeReach',
            'nrReceived',
            'usefulness',
            'delayDropped',
            'noLinkRate',
            'movingNodes',
            'gpsEnabled',
        ]

        for f in facets:
            self.assertEqual(all_results[0][f], all_results[1][f], f'connectivity map optimization is inconsistent for facet {f}')

    def test_phy_loss_counts_only_sensed_non_collided_copies(self):
        from lib.config import Config

        class MockNode:
            def __init__(self, nodeid: int):
                self.nodeid = nodeid
                self.usefulPackets = 0
                self.txAirUtilization = 0.0
                self.droppedByDelay = 0
                self.isMoving = False
                self.gpsEnabled = False

        class MockPacket:
            def __init__(self):
                self.collidedAtN = [False, True, False]
                self.sensedByN = [True, True, False]
                self.receivedAtN = [False, False, False]
                self.phyLostAtN = [True, True, True]
                self.collisionReasonAtN = [None, "capture", None]
                self.terrainLossAtN = [0.0, 0.0, 0.0]
                self.clutterLossAtN = [0.0, 0.0, 0.0]

        conf = Config()
        conf.NR_NODES = 3
        sim_results = lib.discrete_event_sim.SimulationResults({
            "nodes": [MockNode(0), MockNode(1), MockNode(2)],
            "packets": [MockPacket()],
            "delays": [],
            "messageSeq": 1,
            "totalPairs": 0,
            "asymmetricLinks": 0,
            "symmetricLinks": 0,
            "noLinks": 0,
        })

        sim_results.finalize(conf)

        self.assertEqual(sim_results["nrPhyLoss"], 1)
        self.assertEqual(sim_results["nrCollisions"], 1)

    # TODO: add default-skip GUI test?
    def test_discrete_sim_ten_nodes(self):
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

        messageSeq = results["messageSeq"]

        # Begin actual tests, comparing against a hardcoded 'known
        # good' run. If these fail then a change has impacted the
        # results a simulation produces. This could be unintended and
        # a bug, it could be a known consequence of a default config
        # change, or it could be because of an improvement or
        # correction to the sim. Whether to keep these hardcoded values
        # and modify your changes, or to update the hardcoded "known good"
        # simulation results is up to your judgement for which is
        # appropriate. Be cautious!
        self.assertEqual(messageSeq, 185, "expected number of messages created")
        sent = results['sent']
        potentialReceivers = results['potentialReceivers']
        self.assertEqual(sent, 821, "expected number of packets sent")
        self.assertEqual(potentialReceivers, 7389, "expected number of potential receivers")

        nrCollisions = results['nrCollisions']
        self.assertEqual(nrCollisions, 272, "expected number of collisions")
        nrSensed = results['nrSensed']
        self.assertEqual(nrSensed, 2860, "expected number of packets sensed")

        nrReceived = results['nrReceived']
        self.assertEqual(nrReceived, 2588, "expected number of packets received")
        meanDelay = results['meanDelay']
        self.assertEqual(round(meanDelay, 2), 5915.09, "expected rounded delay average")
        txAirUtilizationRate = results['txAirUtilizationRate']
        self.assertEqual(round(txAirUtilizationRate * 100, 2), 4.75, "expected rounded average tx air utilization")

        nodeReach = results['nodeReach']
        self.assertEqual(round(nodeReach*100, 2), 80.9, "expected rounded percentage of nodes reached")

        usefulness = results['usefulness']
        self.assertEqual(round(usefulness*100, 2), 52.05, "expected rounded 'usefulness' percentage")

        delayDropped = results['delayDropped']
        self.assertEqual(delayDropped, 1125, "expected number of packets dropped")
        # default config has both asymmetric links and movement enabled
        noLinkRate = results['noLinkRate']
        self.assertEqual(round(noLinkRate * 100, 2), 55.56, "expected rounded percentage of 'no' links")

        movingNodes = results['movingNodes']
        self.assertEqual(movingNodes, 4, "expected number of moving nodes")

        gpsEnabled = results['gpsEnabled']
        self.assertEqual(gpsEnabled, 1, "expected number of nodes with GPS")

    def test_sim_does_not_change_config(self):
        import copy

        from lib.node import default_generate_node_list

        # get default config, set node number
        from lib.config import CONFIG
        conf = CONFIG

        # copied from the 10-node test just because, but not necessary
        random.seed(conf.SEED)

        conf.NR_NODES = 3 # smaller number for speed.
        conf.update_router_dependencies()
        nodeConfig = default_generate_node_list(conf)
        # skipping GUI graphing to speed things up

        # get copy of the config pre-run
        old_conf = copy.deepcopy(conf)

        # set up and run sim
        sim = lib.discrete_event_sim.DiscreteEventSim(conf, nodeConfig)
        sim.run_simulation()

        # go through the full sim lifecycle, to cover everywhere that may touch config
        results = sim.get_results()

        # set difference trick to compare configs
        conf_diff = conf.__dict__.items() ^ old_conf.__dict__.items()
        self.assertEqual(len(conf_diff), 0, "config has not been changed by running a simulation")

if __name__ == '__main__':
    unittest.main()
