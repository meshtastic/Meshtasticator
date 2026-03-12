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

    # TODO: add default-skip GUI test?
    def test_discrete_sim_ten_nodes(self):
        import numpy as np

        from lib.config import CONFIG
        conf = CONFIG

        # crucial!! and perhaps a tad fragile
        random.seed(conf.SEED)

        self.assertEqual(conf.SEED, 44, "expected default seed for rng")

        # imitate parse_params
        conf.NR_NODES = 10
        nodeConfig = [None for _ in range(conf.NR_NODES)]
        conf.update_router_dependencies()
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
        self.assertEqual(messageSeq["val"], 180, "expected number of messages created")
        sent = results['sent']
        potentialReceivers = results['potentialReceivers']
        self.assertEqual(sent, 875, "expected number of packets sent")
        self.assertEqual(potentialReceivers, 7875, "expected number of potential receivers")

        nrCollisions = results['nrCollisions']
        self.assertEqual(nrCollisions, 320, "expected number of collisions")
        nrSensed = results['nrSensed']
        self.assertEqual(nrSensed, 3071, "expected number of packets sensed")

        nrReceived = results['nrReceived']
        self.assertEqual(nrReceived, 2743, "expected number of packets received")
        meanDelay = results['meanDelay']
        self.assertEqual(round(meanDelay, 2), 9465.81, "expected rounded delay average")
        txAirUtilizationRate = results['txAirUtilizationRate']
        self.assertEqual(round(txAirUtilizationRate * 100, 2), 5.06, "expected rounded average tx air utilization")

        nodeReach = results['nodeReach']
        self.assertEqual(round(nodeReach*100, 2), 85.06, "expected rounded percentage of nodes reached")

        usefulness = results['usefulness']
        self.assertEqual(round(usefulness*100, 2), 50.24, "expected rounded 'usefulness' percentage")

        delayDropped = results['delayDropped']
        self.assertEqual(delayDropped, 1255, "expected number of packets dropped")
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
