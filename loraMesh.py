#!/usr/bin/env python3
import sys
import random

import simpy
import numpy as np

from lib.common import Graph, getParams, plotSchedule, runGraphUpdates, setupAsymmetricLinks
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode

VERBOSE = True
conf = Config()
random.seed(conf.SEED)

if VERBOSE:
	def verboseprint(*args, **kwargs):
		print(*args, **kwargs)
else:
	def verboseprint(*args, **kwargs):
		pass

nodeConfig = getParams(conf, sys.argv)
conf.updateRouterDependencies()
env = simpy.Environment()
bc_pipe = BroadcastPipe(env)

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
	node = MeshNode(conf, nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq, verboseprint)
	nodes.append(node)
	graph.addNode(node)

totalPairs, symmetricLinks, asymmetricLinks, noLinks = setupAsymmetricLinks(conf, nodes)

if conf.MOVEMENT_ENABLED:
	env.process(runGraphUpdates(env, graph, nodes, conf.ONE_MIN_INTERVAL))

conf.updateRouterDependencies()

# start simulation
print("\n====== START OF SIMULATION ======")
env.run(until=conf.SIMTIME)

# compute statistics
print("\n====== END OF SIMULATION ======")
print("*******************************")
print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
print('Number of messages created:', messageSeq["val"])
sent = len(packets)
if conf.DMs:
	potentialReceivers = sent
else:
	potentialReceivers = sent*(conf.NR_NODES-1)
print('Number of packets sent:', sent, 'to', potentialReceivers, 'potential receivers')
nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] is True])
print("Number of collisions:", nrCollisions)
nrSensed = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] is True])
print("Number of packets sensed:", nrSensed)
nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] is True])
print("Number of packets received:", nrReceived)
meanDelay = np.nanmean(delays)
print('Delay average (ms):', round(meanDelay, 2))
txAirUtilization = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES/conf.SIMTIME*100
print('Average Tx air utilization:', round(txAirUtilization, 2), '%')
if nrSensed != 0:
	collisionRate = float((nrCollisions)/nrSensed)
	print("Percentage of packets that collided:", round(collisionRate*100, 2))
else:
	print("No packets sensed.")
nodeReach = sum([n.usefulPackets for n in nodes])/(messageSeq["val"]*(conf.NR_NODES-1))
print("Average percentage of nodes reached:", round(nodeReach*100, 2))
if nrReceived != 0:
	usefulness = sum([n.usefulPackets for n in nodes])/nrReceived  # nr of packets that delivered to a packet to a new receiver out of all packets sent
	print("Percentage of received packets containing new message:", round(usefulness*100, 2))
else:
	print('No packets received.')
delayDropped = sum(n.droppedByDelay for n in nodes)
print("Number of packets dropped by delay/hop limit:", delayDropped)

if conf.MODEL_ASYMMETRIC_LINKS:
	print("Asymmetric links:", round(asymmetricLinks / totalPairs * 100, 2), '%')
	print("Symmetric links:", round(symmetricLinks / totalPairs * 100, 2), '%')
	print("No links:", round(noLinks / totalPairs * 100, 2), '%')

if conf.MOVEMENT_ENABLED:
	movingNodes = sum([1 for n in nodes if n.isMoving is True])
	print("Number of moving nodes:", movingNodes)
	gpsEnabled = sum([1 for n in nodes if n.gpsEnabled is True])
	print("Number of moving nodes w/ GPS:", gpsEnabled)

graph.save()

if conf.PLOT:
	plotSchedule(conf, packets, messages)
