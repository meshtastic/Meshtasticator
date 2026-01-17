#!/usr/bin/env python3
import argparse
import os
import sys
import random

import yaml
import simpy
import numpy as np

from lib.common import Graph, plot_schedule, gen_scenario, run_graph_updates, setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode

VERBOSE = True
conf = Config()
random.seed(conf.SEED)


def verboseprint(*args, **kwargs):
	if VERBOSE:
		print(*args, **kwargs)


def parse_params(conf, args):

	# previous cli behavior:
	# loraMesh.py [nr_nodes [router_type]] | [--from-file [file_name]]
	# we'll replicate the intent with argparse, but more strictly, so flags like '--never--from-file' will no longer be accepted
	parser = argparse.ArgumentParser(
		description='run a single interactive or discrete Meshtastic network simulation'
		)

	# only allow one of --from-file optional, or nr_nodes positional exclusively
	group = parser.add_mutually_exclusive_group()
	group.add_argument('nr_nodes', nargs='?', type=int, help='Number of nodes to generate. If unspecified, do interactive simulation')
	group.add_argument('--from-file', nargs='?', const='nodeConfig.yaml', type=str, metavar='filename', help='Name of yaml file storing node config under "out/" directory. If unspecified, defaults to "nodeConfig.yaml".')

	# the earlier behavior of specifying `router_type` as an optional positional arg with `nr_nodes` is difficult to exactly
	# replicate with argparse, especially since nesting groups was an unintended feature and deprecated.
	# Just implement as an optional argument, and manually treat it as incompatible with `--from-file`
	parser.add_argument('--router-type', type=conf.ROUTER_TYPE, choices=conf.ROUTER_TYPE, help='Router type to use, taken from ROUTER_TYPE enum. Omit the leading "ROUTER_TYPE". Incompatible with --from-file')

	parsed_arguments = parser.parse_args()

	if parsed_arguments.from_file is not None and parsed_arguments.router_type is not None:
		parser.error("Incompatible argument selection. --from-file and --router-type can not be used together")

	if parsed_arguments.from_file is not None:
		with open(os.path.join("out", parsed_arguments.from_file), 'r') as file:
			config = yaml.load(file, Loader=yaml.FullLoader)
	elif parsed_arguments.nr_nodes is not None:
		conf.NR_NODES = parsed_arguments.nr_nodes
		config = [None for _ in range(conf.NR_NODES)]
		if parsed_arguments.router_type is not None:
			routerType = parsed_arguments.router_type
			conf.SELECTED_ROUTER_TYPE = routerType
			conf.update_router_dependencies()
	else:
		config = gen_scenario(conf)

	if config[0] is not None:
		# yaml file or unspecified nr_nodes
		conf.NR_NODES = len(config.keys())

	if conf.NR_NODES < 2:
		parser.error(f"Need at least two nodes. You specified {conf.NR_NODES}")

	print("Number of nodes:", conf.NR_NODES)
	print("Modem:", conf.MODEM)
	print("Simulation time (s):", conf.SIMTIME/1000)
	print("Period (s):", conf.PERIOD/1000)
	print("Interference level:", conf.INTERFERENCE_LEVEL)
	return config


nodeConfig = parse_params(conf, sys.argv)
conf.update_router_dependencies()
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
	graph.add_node(node)

totalPairs, symmetricLinks, asymmetricLinks, noLinks = setup_asymmetric_links(conf, nodes)

if conf.MOVEMENT_ENABLED:
	env.process(run_graph_updates(env, graph, nodes, conf.ONE_MIN_INTERVAL))

conf.update_router_dependencies()

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
	plot_schedule(conf, packets, messages)
