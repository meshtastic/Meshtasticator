import random

import numpy as np

from lib import phy
from lib.link_model import calculate_link_budget
from lib.point import Point


def node_antenna_height(node):
    """Return antenna height above ground, falling back to legacy Point.z."""
    return getattr(node, "antennaHeight", getattr(node, "antenna_height", node.position.z))

def find_random_position(conf, node_configs) -> (float, float):
    """Given a simulation config and list of existing node configs/nodes, find a
    randomly chosen position for the next node such that it is within the
    simulation bounds, within radio range of at least one existing node, and
    not within the minimum distance of any existing node.

    Arguments:
    conf -- Config object defining simulation
    node_configs -- list of NodeConfig objects, or node objects, defining pre-existing nodes

    Returns:
    (x, y) - x and y coordinates of location for next node
    """
    foundMin = True
    foundMax = False
    tries = 0
    position = None
    while not (foundMin and foundMax):
        a = random.random()
        b = random.random()
        posx = a*conf.XSIZE+conf.OX-conf.XSIZE/2
        posy = b*conf.YSIZE+conf.OY-conf.YSIZE/2
        pos_candidate = Point(posx, posy, conf.HM)
        if len(node_configs) > 0:
            for n in node_configs:
                dist = n.position.euclidean_distance(pos_candidate)
                if dist < conf.MINDIST:
                    foundMin = False
                    break
                pathLoss = phy.estimate_path_loss(conf, dist, conf.FREQ)
                rssi = conf.PTX + 2*conf.GL - pathLoss
                # At least one node should be able to reach it
                if rssi >= conf.current_preset["sensitivity"]:
                    foundMax = True
            if foundMin and foundMax:
                position = pos_candidate
        else:
            position = pos_candidate
            foundMin = True
            foundMax = True
        tries += 1
        if tries > 1000:
            print('Could not find a location to place the node. Try increasing XSIZE/YSIZE or decreasing MINDIST.')
            break
    return max(-conf.XSIZE/2, position.x), max(-conf.YSIZE/2, position.y)

def calc_dist(x0, x1, y0, y1, z0=0, z1=0):
    return np.sqrt(((abs(x0-x1))**2)+((abs(y0-y1))**2)+((abs(z0-z1)**2)))

def setup_asymmetric_links(conf, nodes):
    """updates conf to populate LINK_OFFSET member to simulate asymmetric links
    """
    asymLinkRng = random.Random(conf.SEED)
    conf.LINK_OFFSET = {}
    totalPairs = 0
    symmetricLinks = 0
    asymmetricLinks = 0
    noLinks = 0
    for i in range(conf.NR_NODES):
        for b in range(conf.NR_NODES):
            if i != b:
                if conf.MODEL_ASYMMETRIC_LINKS:
                    conf.LINK_OFFSET[(i, b)] = asymLinkRng.gauss(conf.MODEL_ASYMMETRIC_LINKS_MEAN, conf.MODEL_ASYMMETRIC_LINKS_STDDEV)
                else:
                    conf.LINK_OFFSET[(i, b)] = 0

    for a in range(conf.NR_NODES):
        for b in range(conf.NR_NODES):
            if a != b:
                # Calculate the same directed budget MeshPacket will use later:
                # per-direction random offset, both endpoint antenna gains, and
                # any enabled terrain/clutter/calibration layers. The summary
                # graph should not be more optimistic than packet delivery.
                nodeA = nodes[a]
                nodeB = nodes[b]
                budgetAB = calculate_link_budget(conf, nodeA, nodeB, conf.LINK_OFFSET[(a, b)])
                budgetBA = calculate_link_budget(conf, nodeB, nodeA, conf.LINK_OFFSET[(b, a)])

                canAhearB = (budgetAB.rssi_dbm >= conf.current_preset["sensitivity"])
                canBhearA = (budgetBA.rssi_dbm >= conf.current_preset["sensitivity"])

                totalPairs += 1
                if canAhearB and canBhearA:
                    symmetricLinks += 1
                elif canAhearB or canBhearA:
                    asymmetricLinks += 1
                else:
                    noLinks += 1

    return totalPairs, symmetricLinks, asymmetricLinks, noLinks
