#!/usr/bin/env python3
""" Simulator for letting multiple instances of native programs communicate via TCP as if they did via their LoRa chip.
    Usage: python3 interactiveSim.py [nrNodes] [-p <full-path-to-program>] [-d] [-s]
    Use '-d' for Docker.
    Use '-s' to specify what should be sent using this script.
"""
import os
import time
import argparse
from lib.interactive import InteractiveSim, CommandProcessor

parser = argparse.ArgumentParser(prog='interactiveSim')
parser.add_argument('-s', '--script', action='store_true')
parser.add_argument('-d', '--docker', action='store_true')
parser.add_argument('--from-file', action='store_true')
parser.add_argument('-f', '--forward', action='store_true')
parser.add_argument('-p', '--program', type=str, default=os.getcwd())
parser.add_argument('-c', '--collisions', action='store_true')
parser.add_argument('nrNodes', type=int, nargs='?', choices=range(0, 11), default=0)

sim = InteractiveSim(parser.parse_args())  # Start the simulator

if sim.script:  # Use '-s' as argument if you want to specify what you want to send here
    try:
        time.sleep(45)  # Wait until nodeInfo messages are sent
        sim.show_nodes()  # Show nodeDB as seen by each node

        fromNode = 0  # Node from which a message will be sent
        toNode = 1  # Node to whom a message will be sent (if not a broadcast)

        """ Broadcast Message from node 0. """
        sim.send_broadcast("Hi all", fromNode)

        """ Direct Message from node 0 to node 1. """
        # sim.sendDM("Hi node 0", fromNode, toNode)

        """ Ping node 1 from node 0. """
        # sim.sendPing(fromNode, toNode)

        """ Admin Message (setOwner) from node 0 to node 1.
            First you need to add a shared admin channel. """
        # for n in sim.nodes:
        #     n.addAdminChannel()  # or sim.getNodeById(n.nodeid).setURL(<'YOUR_URL'>)
        # sim.sendFromTo(fromNode, toNode).setOwner(long_name="Test")  # can be any function in Node class

        """ Trace route from node 0 to node 1.
            Result will be in the log of node 0. """
        # sim.traceRoute(fromNode, toNode)

        """ Send a position request from node 0 to node 1. """
        # sim.requestPosition(fromNode, toNode)

        time.sleep(15)  # Wait until messages are sent
        sim.graph.plot_metrics(sim.nodes)  # Plot airtime metrics
        sim.graph.init_routes(sim)  # Visualize the route of messages sent
    except KeyboardInterrupt:
        sim.graph.plot_metrics(sim.nodes)
        sim.graph.init_routes(sim)
else:  # Normal usage with commands
    CommandProcessor().cmdloop(sim)
