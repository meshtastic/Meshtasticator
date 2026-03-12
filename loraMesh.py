#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import random

import yaml
import numpy as np

from lib.config import CONFIG
from lib.discrete_event_sim import DiscreteEventSim
from lib.gui import Graph, plot_schedule, gen_scenario

conf = CONFIG
random.seed(conf.SEED)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # default log level

log_level = logging.INFO

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
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose/debug output')

    parsed_arguments = parser.parse_args()

    if parsed_arguments.verbose:
        # set this logger and the loggers for lib.* to DEBUG
        # however, the default of INFO means we shouldn't see debug logs
        # from matplotlib or  PIL. If you want to see those and more, set
        # the default log level for the root logger to DEBUG.
        logger.setLevel(logging.DEBUG)
        lib_logger = logging.getLogger('lib')
        lib_logger.setLevel(logging.DEBUG)
        print("verbose output enabled")

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
    print("Modem:", conf.MODEM_PRESET)
    print("Simulation time (s):", conf.SIMTIME/1000)
    print("Period (s):", conf.PERIOD/1000)
    print("Interference level:", conf.INTERFERENCE_LEVEL)
    return config

nodeConfig = parse_params(conf, sys.argv)
conf.update_router_dependencies()
graph = Graph(conf)

# set up sim
sim = DiscreteEventSim(conf, nodeConfig, graph)

# run sim
print("\n====== START OF SIMULATION ======")
sim.run_simulation()

# collect, process & display results
print("\n====== END OF SIMULATION ======")

results = sim.get_results()

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

# collect second-order results from finalized results
sent = results['sent']
potentialReceivers = results['potentialReceivers']
nrCollisions = results['nrCollisions']
nrSensed = results['nrSensed']
nrReceived = results['nrReceived']
meanDelay = results['meanDelay']
txAirUtilization = results['txAirUtilization']
collisionRate = results['collisionRate']
nodeReach = results['nodeReach']
usefulness = results['usefulness']
delayDropped = results['delayDropped']

print("*******************************")
print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
print('Number of messages created:', messageSeq["val"])
print('Number of packets sent:', sent, 'to', potentialReceivers, 'potential receivers')
print("Number of collisions:", nrCollisions)
print("Number of packets sensed:", nrSensed)
print("Number of packets received:", nrReceived)
print('Delay average (ms):', round(meanDelay, 2))
print('Average Tx air utilization:', round(txAirUtilization * 100, 2), '%')
print("Percentage of packets that collided:", round(collisionRate*100, 2))
print("Average percentage of nodes reached:", round(nodeReach*100, 2))
print("Percentage of received packets containing new message:", round(usefulness*100, 2))
print("Number of packets dropped by delay/hop limit:", delayDropped)

if conf.MODEL_ASYMMETRIC_LINKS:
    asymmetricLinkRate = results['asymmetricLinkRate']
    symmetricLinkRate = results['symmetricLinkRate']
    noLinkRate = results['noLinkRate']
    print("Asymmetric links:", round(asymmetricLinkRate * 100, 2), '%')
    print("Symmetric links:", round(symmetricLinkRate * 100, 2), '%')
    print("No links:", round(noLinkRate * 100, 2), '%')

if conf.MOVEMENT_ENABLED:
    movingNodes = results['movingNodes']
    gpsEnabled = results['gpsEnabled']
    print("Number of moving nodes:", movingNodes)
    print("Number of moving nodes w/ GPS:", gpsEnabled)

graph.save()

if conf.PLOT:
    plot_schedule(conf, packets, messages)
