import random
import unittest

from lib.config import CONFIG

conf = CONFIG

class TestFullDiscreteSim(unittest.TestCase):
	'''
	manually replicate a 10-node default configuration discrete sim test as
	if executing `loraMesh.py 10`. Set up the config to match our previous
	known good test run, run the sim, then check against some hardcoded
	results from a previous known good test run.

	This will make it easier to make big changes and make sure the behavior
	of the sim doesn't change. Or if the prior behavior was mistaken or
	incorrect, we can update this test.
	'''

	def test_discrete_sim_ten_nodes(self):
		import simpy
		import numpy as np

		from lib.common import Graph, run_graph_updates, setup_asymmetric_links
		from lib.discrete_event import BroadcastPipe
		from lib.node import MeshNode

		# crucial!! and perhaps a tad fragile
		random.seed(conf.SEED)

		self.assertEqual(conf.SEED, 44, "expected default seed for rng")

		# initial version: get the config, then just change what
		# parse_params would change.
		# TODO: have our own replicate of our reference config, so we
		# have to explicitly update the test when we change config defaults
		conf.NR_NODES = 10

		nodeConfig = [None for _ in range(conf.NR_NODES)]
		conf.update_router_dependencies()
		env = simpy.Environment()
		bc_pipe = BroadcastPipe(env)

		# begin loraMesh.py copypasta, so we can replicate running a sim
		# simulation variables
		nodes = []
		messages = []
		packets = []
		delays = []
		packetsAtN = [[] for _ in range(conf.NR_NODES)]
		messageSeq = {"val": 0}
		totalPairs = 0
		symmetricLinks = 0
		asymmetricLinks = 0
		noLinks = 0

		graph = Graph(conf)
		for i in range(conf.NR_NODES):
			node = MeshNode(conf, nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq)
			nodes.append(node)
			graph.add_node(node)

		totalPairs, symmetricLinks, asymmetricLinks, noLinks = setup_asymmetric_links(conf, nodes)

		if conf.MOVEMENT_ENABLED:
			env.process(run_graph_updates(env, graph, nodes, conf.ONE_MIN_INTERVAL))

		conf.update_router_dependencies()

		# TODO: disable GUI for this, since IMO that's unwanted when running tests
		env.run(until=conf.SIMTIME)
		# end loraMesh.py copypasta

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
		sent = len(packets)
		if conf.DMs:
			potentialReceivers = sent
		else:
			potentialReceivers = sent*(conf.NR_NODES-1)
		self.assertEqual(sent, 875, "expected number of packets sent")
		self.assertEqual(potentialReceivers, 7875, "expected number of potential receivers")

		nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] is True])
		self.assertEqual(nrCollisions, 320, "expected number of collisions")
		nrSensed = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] is True])
		self.assertEqual(nrSensed, 3071, "expected number of packets sensed")

		nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] is True])
		self.assertEqual(nrReceived, 2743, "expected number of packets received")
		meanDelay = np.nanmean(delays)
		self.assertEqual(round(meanDelay, 2), 9465.81, "expected rounded delay average")
		txAirUtilization = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES/conf.SIMTIME*100
		self.assertEqual(round(txAirUtilization, 2), 5.06, "expected rounded average tx air utilization")

		nodeReach = sum([n.usefulPackets for n in nodes])/(messageSeq["val"]*(conf.NR_NODES-1))
		self.assertEqual(round(nodeReach*100, 2), 85.06, "expected rounded percentage of nodes reached")

		usefulness = sum([n.usefulPackets for n in nodes])/nrReceived  # nr of packets that delivered to a packet to a new receiver out of all packets sent
		self.assertEqual(round(usefulness*100, 2), 50.24, "expected rounded 'usefulness' percentage")

		delayDropped = sum(n.droppedByDelay for n in nodes)
		self.assertEqual(delayDropped, 1255, "expected number of packets dropped")
		# default config has both asymmetric links and movement enabled
		self.assertEqual(round(asymmetricLinks / totalPairs * 100, 2), 8.89, "expected rounded percentage of asymmetric links")
		self.assertEqual(round(symmetricLinks / totalPairs * 100, 2), 42.22, "expected rounded percentage of symmetric links")
		self.assertEqual(round(noLinks / totalPairs * 100, 2), 48.89, "expected rounded percentage of 'no' links")

		movingNodes = sum([1 for n in nodes if n.isMoving is True])
		self.assertEqual(movingNodes, 4, "expected number of moving nodes")

		gpsEnabled = sum([1 for n in nodes if n.gpsEnabled is True])
		self.assertEqual(gpsEnabled, 1, "expected number of nodes with GPS")

if __name__ == '__main__':
	unittest.main()
