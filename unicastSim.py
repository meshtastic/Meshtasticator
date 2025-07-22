#!/usr/bin/env python3
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
	# TODO: refactor with argparse
	if len(args) > 3:
		print("Usage: ./loraMesh [nr_nodes] [--from-file [file_name]]")
		print("Do not specify the number of nodes when reading from a file.")
		exit(1)
	else:
		if len(args) > 1:
			if isinstance(args[1], str) and ("--from-file" in args[1]):
				if len(args) > 2:
					string = args[2]
				else:
					string = 'nodeConfig.yaml'
				with open(os.path.join("out", string), 'r') as file:
					config = yaml.load(file, Loader=yaml.FullLoader)
			else:
				conf.NR_NODES = int(args[1])
				config = [None for _ in range(conf.NR_NODES)]
				if len(args) > 2:
					try:
						# Attempt to convert the string args[2] into a valid enum member
						routerType = conf.ROUTER_TYPE(args[2])
						conf.SELECTED_ROUTER_TYPE = routerType
						conf.update_router_dependencies()
					except ValueError:
						# If it fails, print possible values
						valid_types = [member.name for member in conf.ROUTER_TYPE]
						print(f"Invalid router type: {args[2]}")
						print(f"Router type must be one of: {', '.join(valid_types)}")
						exit(1)
				if conf.NR_NODES == -1:
					config = gen_scenario(conf)
		else:
			config = gen_scenario(conf)
		if config[0] is not None:
			conf.NR_NODES = len(config.keys())
		if conf.NR_NODES < 2:
			print("Need at least two nodes.")
			exit(1)

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

