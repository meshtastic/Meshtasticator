#!/usr/bin/env python3
"""
Burning Man mesh network simulation with:
- 3 routers: 1W output (30dBm), 5dBi antennas, 35ft (10.7m) elevation
- 75% of client nodes in ground clutter near routers (high interference)
- 25% of nodes farther out with less interference
"""
import os
import sys
import random
import yaml
import simpy
import numpy as np
import matplotlib.pyplot as plt

from lib.common import calc_dist, setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode
from lib.packet import MeshPacket
import lib.phy as phy

# Group configuration dictionary - defines all parameters in one place
GROUP_CONFIG = {
    "solo": {
        "probability": 0.20,
        "size_range": (1, 1),
        "cluster_radius": 0,
        "min_group_distance": 50,
        "city_probability": 0.80,
        "city_radius_range": (0, 1.0),  # 0 to 100% of city radius
        "playa_radius_range": (1.0, 1.4),  # 100% to 140% of city radius
    },
    "small_group": {
        "probability": 0.65,
        "size_range": (2, 8),
        "cluster_radius": 20,
        "min_group_distance": 100,
        "city_probability": 0.85,
        "city_radius_range": (0, 1.0),
        "playa_radius_range": (1.0, 1.3),
    },
    "medium_group": {
        "probability": 0.12,
        "size_range": (9, 30),
        "cluster_radius": 35,
        "min_group_distance": 100,
        "city_probability": 0.75,
        "city_radius_range": (0, 1.0),
        "playa_radius_range": (1.0, 1.2),
    },
    "large_camp": {
        "probability": 0.03,
        "size_range": (40, 80),
        "cluster_radius": 60,
        "min_group_distance": 100,
        "city_probability": 1.0,  # Always in city
        "city_radius_range": (0.2, 0.9),  # 20% to 90% of city radius
        "playa_radius_range": (1.0, 1.0),  # Not used since always in city
    },
    "special_event": {
        "probability": 0.0,  # Handled separately
        "size_range": (150, 150),
        "cluster_radius": 150,
        "min_group_distance": 100,
        "city_probability": 0.70,
        "city_radius_range": (0, 1.0),
        "playa_radius_range": (1.0, 1.5),
    }
}

def calculate_burning_man_path_loss(conf, txNode, rxNode, dist, freq):
    """Calculate path loss with realistic ground clutter effects for Burning Man"""
    # Start with base path loss
    base_path_loss = phy.estimate_path_loss(conf, dist, freq, txNode.z, rxNode.z)

    # Apply ground clutter loss ONCE per link based on environment
    additional_loss = 0

    # Get node configs
    tx_in_clutter = hasattr(txNode, 'nodeConfig') and txNode.nodeConfig.get('inGroundClutter', False)
    rx_in_clutter = hasattr(rxNode, 'nodeConfig') and rxNode.nodeConfig.get('inGroundClutter', False)
    tx_in_shadow = hasattr(txNode, 'nodeConfig') and txNode.nodeConfig.get('inRadioShadow', False)
    rx_in_shadow = hasattr(rxNode, 'nodeConfig') and rxNode.nodeConfig.get('inRadioShadow', False)

    # Apply clutter loss based on the actual RF path environment
    if tx_in_clutter and rx_in_clutter:
        # Both in city - RF path goes through clutter area
        clutter_loss = max(5, random.normalvariate(conf.HEAVY_CLUTTER_MEAN, conf.HEAVY_CLUTTER_STD))
        additional_loss += clutter_loss
    elif (tx_in_clutter and not rx_in_clutter) or (not tx_in_clutter and rx_in_clutter):
        # Mixed: one in city, one in open playa
        if txNode.isRouter or rxNode.isRouter:
            # Elevated router can mostly clear city clutter to reach open playa
            clutter_loss = max(1, random.normalvariate(conf.LIGHT_CLUTTER_MEAN / 3, conf.LIGHT_CLUTTER_STD / 2))
        else:
            # Client-to-client mixed case - some clutter effect
            clutter_loss = max(3, random.normalvariate(conf.LIGHT_CLUTTER_MEAN, conf.LIGHT_CLUTTER_STD))
        additional_loss += clutter_loss
    # else: both in open playa - no additional clutter loss, just base path loss

    # Radio shadow REPLACES clutter loss (not additive) if nodes are in shadow
    if tx_in_shadow or rx_in_shadow:
        # Radio shadow loss replaces any clutter loss (worst case scenario)
        shadow_loss = max(15, random.normalvariate(conf.RADIO_SHADOW_MEAN, conf.RADIO_SHADOW_STD))
        additional_loss = shadow_loss  # Replace clutter loss, don't add to it

    return base_path_loss + additional_loss

def assign_user_behavior(conf):
    """Randomly assign a user behavior type based on distribution"""
    import random
    rand_val = random.random()
    cumulative = 0

    for user_type, probability in conf.USER_TYPE_DISTRIBUTION:
        cumulative += probability
        if rand_val <= cumulative:
            return user_type

    # Fallback to camper if something goes wrong
    return "camper"

def check_group_fence_coverage(center_x, center_y, cluster_radius, trash_fence_radius):
    """
    Check if at least 75% of a group would be inside the trash fence.
    Uses circular approximation for simplicity.
    """
    if cluster_radius == 0:
        # Solo node - just check if center is inside fence
        return np.sqrt(center_x**2 + center_y**2) <= trash_fence_radius
    
    # Distance from origin to group center
    center_distance = np.sqrt(center_x**2 + center_y**2)
    
    # If group center + cluster radius is entirely inside fence, 100% coverage
    if center_distance + cluster_radius <= trash_fence_radius:
        return True
    
    # If group center - cluster radius is entirely outside fence, 0% coverage
    if center_distance - cluster_radius >= trash_fence_radius:
        return False
    
    # Approximate coverage using circular intersection
    # For 75% coverage requirement, the group center should be at most
    # cluster_radius * 0.5 away from the fence boundary
    fence_boundary_distance = abs(center_distance - trash_fence_radius)
    return fence_boundary_distance <= cluster_radius * 0.5

def generate_activity_groups(total_clients):
    """Generate activity-based clustering with 20% maximum group size limit"""
    import random

    activity_groups = []
    people_assigned = 0

    # Only include 150-person special event if it's less than 20% of total
    max_group_size = int(total_clients * 0.20)  # 20% limit
    if 150 <= max_group_size:
        activity_groups.append({"type": "special_event", "size": 150})
        people_assigned = 150

    # Activity group distribution for remaining people
    activity_distribution = [
        (group_type, config["probability"], config["size_range"])
        for group_type, config in GROUP_CONFIG.items()
        if config["probability"] > 0  # Skip special_event since handled separately
    ]

    # Generate activity groups for remaining people
    while people_assigned < total_clients:
        # Pick activity type based on distribution
        rand_val = random.random()
        cumulative = 0

        for activity_type, probability, (min_size, max_size) in activity_distribution:
            cumulative += probability
            if rand_val <= cumulative:
                # Generate group size within range
                if activity_type == "solo":
                    size = 1
                elif activity_type == "small_group":
                    # Exponential distribution favoring smaller groups
                    size = min(max_size, max(min_size, int(random.expovariate(1/4) + min_size)))
                elif activity_type == "medium_group":
                    # Normal distribution for medium gatherings
                    size = min(max_size, max(min_size, int(random.normalvariate(18, 6))))
                elif activity_type == "large_camp":
                    # Large theme camps with people hanging around
                    size = min(max_size, max(min_size, int(random.normalvariate(60, 12))))

                # Don't exceed remaining people or 20% rule
                remaining = total_clients - people_assigned
                size = min(size, remaining)
                size = min(size, max_group_size)  # Apply 20% limit

                if size > 0:
                    activity_groups.append({"type": activity_type, "size": size})
                    people_assigned += size
                break

    return activity_groups

def calculate_rebroadcast_priority(rssi, rx_config, distance, conf):
    """
    Calculate the priority score for a node to be a rebroadcaster.
    Higher score = more likely to win the contention window and actually rebroadcast.

    Based on Meshtastic's SNR-weighted delays: better signal strength = shorter delay.
    """
    # Convert RSSI to SNR (Signal-to-Noise Ratio)
    snr = rssi - conf.NOISE_LEVEL

    # Base priority from SNR (higher SNR = higher priority)
    # SNR typically ranges from -20 to +15 dB
    snr_score = max(0, min(35, snr + 20))  # Normalize to 0-35 range

    # Router bonus - routers get prioritized in contention window
    router_bonus = 10 if rx_config['isRouter'] else 0

    # Distance factor - nodes at medium distances are often best rebroadcasters
    # Too close = redundant, too far = poor signal
    distance_score = 0
    if 500 <= distance <= 2000:  # Sweet spot for rebroadcasting
        distance_score = 5
    elif 200 <= distance <= 3000:  # Still useful
        distance_score = 2

    # Ground clutter penalty - nodes in heavy clutter are less likely to be good rebroadcasters
    clutter_penalty = 0
    if rx_config.get('inGroundClutter', False):
        clutter_penalty = -3
    if rx_config.get('inRadioShadow', False):
        clutter_penalty = -10  # Major penalty for radio shadows

    total_score = snr_score + router_bonus + distance_score + clutter_penalty
    return total_score

def precompute_connectable_nodes(conf, node_configs):
    """
    Precompute which nodes could plausibly communicate with each other.
    Uses best-case signal calculations with safety margin to identify potential connections.
    This replaces the O(N²) calculation that happens on every packet transmission.
    """
    print("Precomputing connectable node matrix...")

    # Safety margin to account for variability (asymmetric links, interference, etc.)
    SAFETY_MARGIN_DB = 12  # Conservative margin

    connectivity_matrix = {}
    baseline_path_loss_matrix = {}
    rebroadcast_priority_matrix = {}  # New: SNR-based rebroadcast priorities
    connectivity_stats = {
        'total_possible_links': 0,
        'connectable_links': 0,
        'router_links': 0,
        'client_links': 0,
        'avg_connectable': 0,
        'max_connectable': 0,
        'min_connectable': float('inf'),
        'distance_stats': [],
        'safety_margin_db': SAFETY_MARGIN_DB
    }

    total_pairs = 0
    total_connectable = 0

    for tx_idx, tx_config in enumerate(node_configs):
        connectable_nodes = []
        baseline_path_loss = {}
        rebroadcast_candidates = []  # List of (rx_idx, priority_score) tuples

        for rx_idx, rx_config in enumerate(node_configs):
            if tx_idx == rx_idx:
                continue

            total_pairs += 1

            # Calculate distance
            dist_3d = calc_dist(
                tx_config['x'], rx_config['x'],
                tx_config['y'], rx_config['y'],
                tx_config['z'], rx_config['z']
            )

            # Calculate BEST-CASE path loss (minimum possible loss)
            base_path_loss = phy.estimate_path_loss(conf, dist_3d, conf.FREQ, tx_config['z'], rx_config['z'])

            # Add only the minimum ground clutter loss (best case scenario)
            min_additional_loss = 0
            tx_in_clutter = tx_config.get('inGroundClutter', False)
            rx_in_clutter = rx_config.get('inGroundClutter', False)
            tx_in_shadow = tx_config.get('inRadioShadow', False)
            rx_in_shadow = rx_config.get('inRadioShadow', False)

            # Apply minimum clutter loss based on actual RF path (best case scenario)
            if tx_in_clutter and rx_in_clutter:
                # Both in city - use heavy clutter minimum
                min_additional_loss += max(5, conf.HEAVY_CLUTTER_MEAN - 2 * conf.HEAVY_CLUTTER_STD)
            elif (tx_in_clutter and not rx_in_clutter) or (not tx_in_clutter and rx_in_clutter):
                # Mixed: one in city, one in open playa
                if tx_config['isRouter'] or rx_config['isRouter']:
                    # Elevated router can mostly clear city clutter to reach open playa
                    min_additional_loss += max(1, (conf.LIGHT_CLUTTER_MEAN / 3) - conf.LIGHT_CLUTTER_STD)
                else:
                    # Client-to-client mixed case - light clutter minimum
                    min_additional_loss += max(3, conf.LIGHT_CLUTTER_MEAN - 2 * conf.LIGHT_CLUTTER_STD)
            # else: both in open playa - no additional clutter loss

            # Radio shadow replaces clutter loss in worst case (use minimum shadow loss)
            if tx_in_shadow or rx_in_shadow:
                min_shadow_loss = max(15, conf.RADIO_SHADOW_MEAN - 2 * conf.RADIO_SHADOW_STD)
                min_additional_loss = min_shadow_loss  # Replace clutter, don't add

            best_case_path_loss = base_path_loss + min_additional_loss

            # Calculate best-case RSSI
            best_case_rssi = tx_config['ptx'] + tx_config['antennaGain'] + rx_config['antennaGain'] - best_case_path_loss

            # Check if nodes could plausibly communicate (with safety margin)
            if best_case_rssi >= (conf.current_preset["sensitivity"] + SAFETY_MARGIN_DB):
                connectable_nodes.append(rx_idx)
                baseline_path_loss[rx_idx] = best_case_path_loss
                total_connectable += 1
                connectivity_stats['distance_stats'].append(dist_3d)

                # Calculate rebroadcast priority score
                # Higher score = more likely to be the one that actually rebroadcasts
                priority_score = calculate_rebroadcast_priority(
                    best_case_rssi, rx_config, dist_3d, conf
                )
                rebroadcast_candidates.append((rx_idx, priority_score))

                # Count router vs client links
                if tx_config['isRouter'] or rx_config['isRouter']:
                    connectivity_stats['router_links'] += 1
                else:
                    connectivity_stats['client_links'] += 1

        # Sort rebroadcast candidates by priority (highest first)
        rebroadcast_candidates.sort(key=lambda x: x[1], reverse=True)

        # Store top N candidates most likely to actually rebroadcast
        # This is where the major optimization happens - instead of checking all connectable
        # nodes, we only check the top candidates who will win the contention window
        # AGGRESSIVE: Only keep top 5 to maximize performance
        top_rebroadcasters = rebroadcast_candidates[:min(5, len(rebroadcast_candidates))]

        connectivity_matrix[tx_idx] = connectable_nodes
        baseline_path_loss_matrix[tx_idx] = baseline_path_loss
        rebroadcast_priority_matrix[tx_idx] = [idx for idx, score in top_rebroadcasters]

        # Update stats
        num_connectable = len(connectable_nodes)
        connectivity_stats['max_connectable'] = max(connectivity_stats['max_connectable'], num_connectable)
        connectivity_stats['min_connectable'] = min(connectivity_stats['min_connectable'], num_connectable)

    # Calculate final stats
    connectivity_stats['total_possible_links'] = total_pairs
    connectivity_stats['connectable_links'] = total_connectable
    connectivity_stats['avg_connectable'] = total_connectable / len(node_configs) if node_configs else 0
    connectivity_stats['connectivity_percentage'] = (total_connectable / total_pairs * 100) if total_pairs > 0 else 0

    if connectivity_stats['distance_stats']:
        distances = connectivity_stats['distance_stats']
        connectivity_stats['avg_distance'] = sum(distances) / len(distances)
        connectivity_stats['max_distance'] = max(distances)
        connectivity_stats['min_distance'] = min(distances)

    print(f"✅ Precomputed connectivity for {len(node_configs)} nodes")
    print(f"   Connectable pairs: {total_connectable:,} / {total_pairs:,} possible ({connectivity_stats['connectivity_percentage']:.1f}%)")
    print(f"   Avg connectable per node: {connectivity_stats['avg_connectable']:.1f}")
    print(f"   Range: {connectivity_stats['min_connectable']}-{connectivity_stats['max_connectable']} connectable")
    print(f"   Safety margin: {SAFETY_MARGIN_DB}dB")
    if total_connectable > 0:
        print(f"   Performance improvement: ~{total_pairs/total_connectable:.1f}x reduction in calculations")

        # Calculate rebroadcast optimization stats
        total_priority_nodes = sum(len(rebroadcast_priority_matrix[tx_idx]) for tx_idx in rebroadcast_priority_matrix)
        avg_priority_nodes = total_priority_nodes / len(node_configs) if node_configs else 0
        print(f"   Rebroadcast optimization: avg {avg_priority_nodes:.1f} priority nodes vs {connectivity_stats['avg_connectable']:.1f} total connectable")
        if connectivity_stats['avg_connectable'] > 0:
            rebroadcast_reduction = connectivity_stats['avg_connectable'] / avg_priority_nodes if avg_priority_nodes > 0 else 1
            print(f"   Additional {rebroadcast_reduction:.1f}x reduction from SNR-based rebroadcast prioritization")
    else:
        print(f"   WARNING: No connectable node pairs found! Check path loss parameters.")

    # Analyze router connectivity with real signal calculations
    print(f"\n📡 Router Connectivity Analysis (Real Signal Calculations):")
    sensitivity_threshold = conf.current_preset["sensitivity"]

    for router_idx, router_config in enumerate([n for n in node_configs if n['isRouter']]):
        router_id = router_config['nodeId']

        # Create mock transmitter and receiver node objects for path loss calculation
        class MockNode:
            def __init__(self, config):
                self.nodeConfig = config
                self.isRouter = config['isRouter']
                self.z = config['z']

        router_node = MockNode(router_config)

        # Real-time signal calculation for all nodes
        connectable_nodes = []
        signal_analysis = []

        for rx_id, rx_config in enumerate(node_configs):
            if rx_id == router_id:
                continue  # Skip self

            rx_node = MockNode(rx_config)

            # Calculate 3D distance
            distance = calc_dist(
                router_config['x'], rx_config['x'],
                router_config['y'], rx_config['y'],
                router_config['z'], rx_config['z']
            )

            # Calculate actual path loss using the same function as simulation
            path_loss = calculate_burning_man_path_loss(conf, router_node, rx_node, distance, conf.FREQ)

            # Calculate RSSI: TX power + TX antenna gain + RX antenna gain - path loss
            rssi = (router_config['ptx'] + router_config['antennaGain'] +
                   rx_config['antennaGain'] - path_loss)

            # Check if signal is strong enough to be received
            can_receive = rssi >= sensitivity_threshold

            signal_analysis.append({
                'node_id': rx_id,
                'distance': distance,
                'path_loss': path_loss,
                'rssi': rssi,
                'can_receive': can_receive,
                'config': rx_config
            })

            if can_receive:
                connectable_nodes.append(rx_id)

        # Sort by distance for analysis
        signal_analysis.sort(key=lambda x: x['distance'])

        # Find farthest connectable node
        connectable_signals = [s for s in signal_analysis if s['can_receive']]
        farthest_connectable = max(connectable_signals, key=lambda x: x['distance']) if connectable_signals else None

        # Count by node type and shadow status (only connectable nodes)
        city_clients = 0
        city_shadow_clients = 0
        playa_clients = 0
        playa_shadow_clients = 0
        other_routers = 0

        for signal in connectable_signals:
            config = signal['config']
            if config['isRouter']:
                other_routers += 1
            elif config.get('inGroundClutter', False):
                if config.get('inRadioShadow', False):
                    city_shadow_clients += 1
                else:
                    city_clients += 1
            else:
                if config.get('inRadioShadow', False):
                    playa_shadow_clients += 1
                else:
                    playa_clients += 1

        total_connectable = len(connectable_signals)
        total_tested = len(signal_analysis)

        print(f"   Router {router_id}: {total_connectable}/{total_tested} nodes can receive signal ({total_connectable/total_tested*100:.1f}%)")
        print(f"     - {city_clients} city clients, {city_shadow_clients} city shadow clients")
        print(f"     - {playa_clients} playa clients, {playa_shadow_clients} playa shadow clients, {other_routers} routers")

        if farthest_connectable:
            node_type = "router" if farthest_connectable['config']['isRouter'] else "client"
            print(f"     - Farthest connection: {farthest_connectable['distance']:.0f}m to node {farthest_connectable['node_id']} ({node_type})")
            print(f"       RSSI: {farthest_connectable['rssi']:.1f}dBm (threshold: {sensitivity_threshold}dBm)")
        else:
            print(f"     - No nodes can receive signal from this router!")

        # Show a few example calculations for debugging
        print(f"     - Signal examples (closest 3 connectable):")
        connectable_by_distance = sorted(connectable_signals, key=lambda x: x['distance'])[:3]
        for signal in connectable_by_distance:
            node_type = "router" if signal['config']['isRouter'] else "client"
            print(f"       {signal['distance']:.0f}m to node {signal['node_id']} ({node_type}): "
                  f"RSSI {signal['rssi']:.1f}dBm, path loss {signal['path_loss']:.1f}dB")

        # Show failed connections (nodes that can't receive)
        failed_signals = [s for s in signal_analysis if not s['can_receive']]
        if failed_signals:
            closest_failed = min(failed_signals, key=lambda x: x['distance'])
            print(f"     - Closest node that CAN'T receive: {closest_failed['distance']:.0f}m, "
                  f"RSSI {closest_failed['rssi']:.1f}dBm (too weak by {sensitivity_threshold - closest_failed['rssi']:.1f}dB)")

    return connectivity_matrix, baseline_path_loss_matrix, rebroadcast_priority_matrix, connectivity_stats

class OptimizedMeshPacket(MeshPacket):
    """
    Optimized packet class that uses precomputed connectivity matrix
    to only calculate signals for nodes that could plausibly communicate.
    """
    def __init__(self, conf, nodes, txNodeId, destId, origTxNodeId, packetLength, seq, genTime, wantAck, isAck, origPacket, now, verboseprint):
        # Initialize arrays for all nodes (same as original)
        self.conf = conf
        self.nodes = nodes
        self.txNodeId = txNodeId
        self.destId = destId
        self.origTxNodeId = origTxNodeId
        self.packetLength = packetLength
        self.seq = seq
        self.genTime = genTime
        self.wantAck = wantAck
        self.isAck = isAck
        self.origPacket = origPacket
        self.now = now
        self.verboseprint = verboseprint
        self.txpow = self.conf.PTX

        # Initialize arrays for all nodes
        self.LplAtN = [0 for _ in range(self.conf.NR_NODES)]
        self.rssiAtN = [0 for _ in range(self.conf.NR_NODES)]
        self.sensedByN = [False for _ in range(self.conf.NR_NODES)]
        self.detectedByN = [False for _ in range(self.conf.NR_NODES)]
        self.collidedAtN = [False for _ in range(self.conf.NR_NODES)]
        self.receivedAtN = [False for _ in range(self.conf.NR_NODES)]
        self.onAirToN = [True for _ in range(self.conf.NR_NODES)]

        # Use precomputed connectivity matrix for optimization
        if hasattr(conf, 'CONNECTIVITY_MATRIX') and txNodeId in conf.CONNECTIVITY_MATRIX:
            # For performance, prioritize nodes most likely to actually rebroadcast
            # This is the key optimization: instead of processing ALL connectable nodes,
            # we focus on the ones that will actually matter for routing
            priority_nodes = conf.REBROADCAST_PRIORITY_MATRIX.get(txNodeId, [])

            # AGGRESSIVE OPTIMIZATION: Only process top N most likely rebroadcasters
            # This sacrifices some simulation fidelity for major performance gains
            max_candidates = getattr(conf, 'MAX_REBROADCAST_CANDIDATES', 3)
            nodes_to_process = priority_nodes[:max_candidates]

            # Always include routers if not already included (they're critical for routing)
            all_connectable = conf.CONNECTIVITY_MATRIX[txNodeId]
            for rx_nodeid in all_connectable:
                if (nodes[rx_nodeid].isRouter and
                    rx_nodeid not in nodes_to_process and
                    len(nodes_to_process) < 4):  # Cap at 4 total nodes
                    nodes_to_process.append(rx_nodeid)

            for rx_nodeid in nodes_to_process:
                # Get baseline path loss from precomputation
                baseline_path_loss = conf.BASELINE_PATH_LOSS_MATRIX[txNodeId][rx_nodeid]

                # Apply dynamic factors to get actual path loss
                actual_path_loss = self._calculate_dynamic_path_loss(
                    baseline_path_loss, txNodeId, rx_nodeid, nodes
                )

                # Calculate actual RSSI with all dynamic factors
                tx_node = nodes[txNodeId]
                rx_node = nodes[rx_nodeid]
                actual_rssi = (tx_node.nodeConfig['ptx'] +
                              tx_node.nodeConfig['antennaGain'] +
                              rx_node.nodeConfig['antennaGain'] -
                              actual_path_loss)

                self.LplAtN[rx_nodeid] = actual_path_loss
                self.rssiAtN[rx_nodeid] = actual_rssi

                # Mark as detectable if RSSI is sufficient
                if actual_rssi >= conf.current_preset["sensitivity"]:
                    self.detectedByN[rx_nodeid] = True
                    self.sensedByN[rx_nodeid] = True
        else:
            # Fallback to original behavior if connectivity matrix not available
            super().__init__(conf, nodes, txNodeId, destId, origTxNodeId, packetLength, seq, genTime, wantAck, isAck, origPacket, now, verboseprint)
            return

        # Set packet attributes (matching original MeshPacket)
        self.hopLimit = conf.hopLimit
        self.retransmissions = conf.maxRetransmission
        self.ackReceived = False
        self.packetLen = packetLength
        self.requestId = origPacket

        # Set modem configuration
        self.sf = self.conf.current_preset["sf"]
        self.cr = self.conf.current_preset["cr"]
        self.bw = self.conf.current_preset["bw"]
        self.freq = self.conf.FREQ

        # Calculate timing
        from lib.phy import airtime
        self.timeOnAir = airtime(conf, self.sf, self.cr, self.packetLength, self.bw)
        self.startTime = now
        self.endTime = self.startTime + self.timeOnAir

    def _calculate_dynamic_path_loss(self, baseline_path_loss, tx_nodeid, rx_nodeid, nodes):
        """
        Apply dynamic factors to baseline path loss to get actual path loss.
        OPTIMIZED: Minimize dynamic calculations for performance.
        """
        # OPTIMIZATION: Skip most dynamic factors for performance
        # The baseline already includes ground clutter, so asymmetric links
        # are the main remaining dynamic factor

        if (self.conf.MODEL_ASYMMETRIC_LINKS and
            (tx_nodeid, rx_nodeid) in self.conf.LINK_OFFSET):
            return baseline_path_loss + self.conf.LINK_OFFSET[(tx_nodeid, rx_nodeid)]

        return baseline_path_loss

class OptimizedMeshNode(MeshNode):
    """
    Optimized node class that uses OptimizedMeshPacket for better performance
    """
    def send_packet(self, destId, type=""):
        # increment the shared counter
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]

        from lib.packet import MeshMessage
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))

        # Use optimized packet instead of regular MeshPacket
        p = OptimizedMeshPacket(
            self.conf, self.nodes, self.nodeid, destId, self.nodeid,
            self.conf.PACKETLENGTH, messageSeq, self.env.now, True, False,
            None, self.env.now, self.verboseprint
        )

        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'generated', type, 'message', p.seq, 'to', destId)
        self.packets.append(p)
        self.env.process(self.transmit(p))
        return p

def plot_node_locations(node_configs, conf, connectivity_matrix=None, baseline_path_loss_matrix=None):
    """Plot node locations with signal strength lines to routers - showing both send and receive"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Separate nodes by type
    routers = [n for n in node_configs if n['isRouter']]
    clients = [n for n in node_configs if not n['isRouter'] and not n.get('isClientMute', False)]
    client_mutes = [n for n in node_configs if n.get('isClientMute', False)]

    def plot_on_axis(ax, title, direction):
        """Helper function to plot nodes and signal lines on a specific axis"""
        
        # Draw signal strength lines between clients and routers
        if connectivity_matrix is not None and baseline_path_loss_matrix is not None:
            router_lookup = {r['nodeId']: r for r in routers}
            total_connections_drawn = 0
            max_distance_drawn = 0

            for client in clients:
                client_id = client['nodeId']

                for router_id, router in router_lookup.items():
                    # Create mock nodes for path loss calculation
                    class MockNode:
                        def __init__(self, config):
                            self.nodeConfig = config
                            self.isRouter = config['isRouter']
                            self.z = config['z']

                    client_node = MockNode(client)
                    router_node = MockNode(router)

                    # Calculate 3D distance
                    distance = calc_dist(
                        client['x'], router['x'],
                        client['y'], router['y'],
                        client['z'], router['z']
                    )

                    # Calculate path loss and RSSI based on direction
                    if direction == "send":
                        # Client -> Router (client transmitting to router)
                        path_loss = calculate_burning_man_path_loss(conf, client_node, router_node, distance, conf.FREQ)
                        rssi = (client['ptx'] + client['antennaGain'] + router['antennaGain'] - path_loss)
                    else:
                        # Router -> Client (router transmitting to client)
                        path_loss = calculate_burning_man_path_loss(conf, router_node, client_node, distance, conf.FREQ)
                        rssi = (router['ptx'] + router['antennaGain'] + client['antennaGain'] - path_loss)

                    # Only draw line if signal is strong enough to be received
                    if rssi >= conf.current_preset["sensitivity"]:
                        total_connections_drawn += 1
                        max_distance_drawn = max(max_distance_drawn, distance)

                        # Determine line color and thickness based on signal strength
                        if rssi >= -80:
                            color = 'green'
                            thickness = 2.0
                            alpha = 0.8
                        elif rssi >= -100:
                            color = 'yellow'
                            thickness = 1.5
                            alpha = 0.6
                        elif rssi >= -120:
                            color = 'red'
                            thickness = 1.0
                            alpha = 0.4
                        else:
                            continue

                        # Draw line from client to router
                        ax.plot([client['x'], router['x']],
                               [client['y'], router['y']],
                               color=color, linewidth=thickness, alpha=alpha, zorder=1)

        # Plot routers (red)
        if routers:
            router_x = [n['x'] for n in routers]
            router_y = [n['y'] for n in routers]
            ax.scatter(router_x, router_y, c='red', s=100, marker='s', label=f'Routers ({len(routers)})', zorder=3)

        # Plot regular clients (blue)
        if clients:
            client_x = [n['x'] for n in clients]
            client_y = [n['y'] for n in clients]
            ax.scatter(client_x, client_y, c='blue', s=20, alpha=0.6, label=f'Clients ({len(clients)})', zorder=2)

        # Plot muted clients (black)
        if client_mutes:
            mute_x = [n['x'] for n in client_mutes]
            mute_y = [n['y'] for n in client_mutes]
            ax.scatter(mute_x, mute_y, c='black', s=20, alpha=0.8, label=f'Client Mute ({len(client_mutes)})', zorder=2)

        # Add center plaza circle for reference (open space)
        center_circle = plt.Circle((0, 0), conf.CENTER_PLAZA_RADIUS, fill=False, linestyle=':', color='lightblue', alpha=0.7)
        ax.add_patch(center_circle)
        
        # Add city radius circle for reference
        city_circle = plt.Circle((0, 0), conf.CITY_RADIUS, fill=False, linestyle='--', color='gray', alpha=0.5)
        ax.add_patch(city_circle)

        # Add Burning Man trash fence using actual GPS coordinates
        import numpy as np

        # Actual GPS coordinates of trash fence corners
        gps_coords = [
            (40.78236, -119.23530),  # P1
            (40.80570, -119.21965),  # P2
            (40.80163, -119.18533),  # P3
            (40.77568, -119.17971),  # P4
            (40.76373, -119.21050),  # P5
        ]

        # Convert GPS to relative meters (approximate)
        center_lat = sum(coord[0] for coord in gps_coords) / len(gps_coords)
        center_lon = sum(coord[1] for coord in gps_coords) / len(gps_coords)

        def gps_to_meters(lat, lon, center_lat, center_lon):
            lat_m = (lat - center_lat) * 111000
            lon_m = (lon - center_lon) * 111000 * np.cos(np.radians(center_lat))
            return lon_m, lat_m

        # Convert GPS coordinates to meters
        fence_coords_m = [gps_to_meters(lat, lon, center_lat, center_lon) for lat, lon in gps_coords]

        # Scale to fit our simulation area (trash fence should be ~3km radius)
        current_coords = np.array(fence_coords_m)
        current_radius = np.max(np.sqrt(current_coords[:, 0]**2 + current_coords[:, 1]**2))
        scale_factor = conf.TRASH_FENCE_RADIUS / current_radius

        fence_x = [coord[0] * scale_factor for coord in fence_coords_m]
        fence_y = [coord[1] * scale_factor for coord in fence_coords_m]

        # Rotate fence so longest point (apex) is at 45 degrees from north
        distances = [np.sqrt(fence_x[i]**2 + fence_y[i]**2) for i in range(len(fence_x))]
        apex_idx = np.argmax(distances)
        current_apex_angle = np.arctan2(fence_y[apex_idx], fence_x[apex_idx])
        target_apex_angle = np.radians(45)
        rotation_angle = target_apex_angle - current_apex_angle

        # Apply rotation to all points
        rotated_x = []
        rotated_y = []
        for x, y in zip(fence_x, fence_y):
            rotated_x.append(x * np.cos(rotation_angle) - y * np.sin(rotation_angle))
            rotated_y.append(x * np.sin(rotation_angle) + y * np.cos(rotation_angle))

        fence_x = rotated_x
        fence_y = rotated_y

        # Draw each fence segment as individual vectors
        for i in range(5):  # 5 sides of pentagon
            x_start, y_start = fence_x[i], fence_y[i]
            x_end, y_end = fence_x[(i+1) % 5], fence_y[(i+1) % 5]

            # Draw line segment
            ax.plot([x_start, x_end], [y_start, y_end], color='orange', linewidth=3,
                    label='Trash Fence' if i == 0 else '', zorder=1)

            # Draw arrow to show direction/vector nature
            mid_x, mid_y = (x_start + x_end) / 2, (y_start + y_end) / 2
            dx, dy = x_end - x_start, y_end - y_start
            ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), xytext=(mid_x, mid_y),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2))

        # Add legend entries for signal strength lines
        if connectivity_matrix is not None and baseline_path_loss_matrix is not None:
            # Add dummy lines for legend
            ax.plot([], [], color='green', linewidth=2.0, alpha=0.8, label='Strong Signal (≥-80dBm)')
            ax.plot([], [], color='yellow', linewidth=1.5, alpha=0.6, label='Medium Signal (-80 to -100dBm)')
            ax.plot([], [], color='red', linewidth=1.0, alpha=0.4, label='Weak Signal (-100 to -120dBm)')

        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.set_xlabel('Distance (meters)')
        ax.set_ylabel('Distance (meters)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.text(0, conf.CITY_RADIUS + 200, f'City Radius: {conf.CITY_RADIUS}m',
                ha='center', va='bottom', fontsize=10, color='gray')

        return total_connections_drawn, max_distance_drawn

    # Create both plots
    print("Drawing signal strength visualization lines...")
    
    send_connections, send_max_dist = plot_on_axis(ax1, 'Client Send Signal (Client → Router)', 'send')
    recv_connections, recv_max_dist = plot_on_axis(ax2, 'Client Receive Signal (Router → Client)', 'receive')

    print(f"Send signal lines: {send_connections} connections, max distance: {send_max_dist:.0f}m")
    print(f"Receive signal lines: {recv_connections} connections, max distance: {recv_max_dist:.0f}m")

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs("out/graphics", exist_ok=True)
    plt.savefig("out/graphics/burning_man_nodes.png", dpi=150, bbox_inches='tight')
    plt.show()

# Burning Man specific configuration
class BurningManConfig(Config):
    def __init__(self):
        super().__init__()

        # Black Rock City is roughly 3.4km in diameter (1.7km radius)
        # Deep playa extends much farther
        self.XSIZE = 8000  # 8km x 8km area
        self.YSIZE = 8000

        # Router configuration
        self.ROUTER_COUNT = 3
        self.ROUTER_PTX = 30  # 1W = 30dBm
        self.ROUTER_ANTENNA_GAIN = 5  # 5dBi antenna
        self.ROUTER_HEIGHT = 10.7  # 35 feet in meters

        # Client configuration
        self.CLIENT_PTX = 20  # Standard 100mW = 20dBm
        self.CLIENT_ANTENNA_GAIN = 0  # 0dBi for handheld units
        self.CLIENT_HEIGHT = 1.5  # Person height

        # Ground clutter parameters
        self.CENTER_PLAZA_RADIUS = 850   # 0.85km radius for center open space (Man area)
        self.CITY_RADIUS = 1700  # 1.7km radius for main city area (3.4km diameter)
        self.TRASH_FENCE_RADIUS = 2587  # 2.587km radius trash fence (5.175km diameter event perimeter)

        # Realistic path loss parameters using normal distributions
        # Light clutter: router-to-client or open areas with some obstacles
        self.LIGHT_CLUTTER_MEAN = 6   # Average 6dB loss (field tested: >1.2km in city)
        self.LIGHT_CLUTTER_STD = 2    # ±2dB standard deviation

        # Heavy clutter: client-to-client in dense city areas with art installations
        self.HEAVY_CLUTTER_MEAN = 10  # Average 10dB loss (desert environment with art)
        self.HEAVY_CLUTTER_STD = 3    # ±3dB standard deviation

        # Radio shadow: complete obstruction behind large art/structures
        self.RADIO_SHADOW_MEAN = 20   # Average 20dB loss (replaces clutter, not excessive)
        self.RADIO_SHADOW_STD = 4     # ±4dB standard deviation
        self.RADIO_SHADOW_PROBABILITY = 0.25  # 25% chance of being in complete radio shadow

        # Legacy parameters (kept for compatibility but not used in new model)
        self.GROUND_CLUTTER_LOSS = 25  # Replaced by LIGHT/HEAVY_CLUTTER_MEAN
        self.GROUND_CLUTTER_LOSS_VARIANCE = 10  # Replaced by STD parameters
        self.RADIO_SHADOW_ADDITIONAL_LOSS = 40  # Replaced by RADIO_SHADOW_MEAN

        self.INTERFERENCE_IN_CITY = 0.15  # 15% interference level in city
        self.INTERFERENCE_OUTSIDE = 0.02  # 2% interference in open playa

        # Use SHORT_TURBO for better performance in high-density areas
        self.MODEM_PRESET = "SHORT_TURBO"

        # Disable verbose output for large simulations
        self.VERBOSE_MESH_TRAFFIC = False

        # Use shorter simulation time for performance testing
        self.SIMTIME = 15 * self.ONE_MIN_INTERVAL  # 15 minutes for scaling tests

        # Turbo mode: aggressive optimizations for large-scale testing
        self.TURBO_MODE = True  # Sacrifice some fidelity for speed
        self.MAX_REBROADCAST_CANDIDATES = 3 if self.TURBO_MODE else 5

        # For very large node counts, use even shorter simulation
        # Note: NR_NODES is set later, so this will be applied in run_burning_man_simulation

        # Path loss model selection:
        # MODEL = 0: Log-Distance (harsh, realistic for desert/rural)
        #   - Uses path loss exponent GAMMA (higher = more loss over distance)
        #   - Good for handheld radios in challenging environments
        # MODEL = 1-4: Okumura-Hata variants (urban/suburban)
        # MODEL = 5: 3GPP Suburban Macro (optimistic, for cell towers)
        #   - Designed for elevated base stations with good coverage
        #   - Too optimistic for mesh networks
        # MODEL = 6: 3GPP Urban Macro
        self.MODEL = 5  # Back to 3GPP model that was working

        # Note: Log-distance model with default parameters seems miscalibrated
        # for our 915MHz LoRa scenario. The LPLD0=127.41 might be for different
        # frequency or includes additional losses.

        # Realistic user behavior patterns
        self.USER_BEHAVIORS = {
            "camper": {
                "message_period": 8 * self.ONE_HR_INTERVAL,  # 8 hours average
                "position_period": -1,  # No position broadcasts
                "movement_probability": 0.1,  # 10% chance of being mobile
            },
            "explorer": {
                "message_period": 2 * self.ONE_HR_INTERVAL,  # 2 hours average
                "position_period": 15 * self.ONE_MIN_INTERVAL,  # 15 min when "moving"
                "movement_probability": 0.6,  # 60% chance of being mobile
            },
            "staff": {
                "message_period": 1 * self.ONE_HR_INTERVAL,  # 1 hour average
                "position_period": 15 * self.ONE_MIN_INTERVAL,  # 15 min when "moving"
                "movement_probability": 0.8,  # 80% chance of being mobile
            },
            "heavy_user": {
                "message_period": 30 * self.ONE_MIN_INTERVAL,  # 30 minutes
                "position_period": 10 * self.ONE_MIN_INTERVAL,  # 10 min when "moving"
                "movement_probability": 0.4,  # 40% chance of being mobile
            }
        }

        # Distribution of user types at Burning Man
        self.USER_TYPE_DISTRIBUTION = [
            ("camper", 0.60),      # 60% - mostly stationary campers
            ("explorer", 0.25),    # 25% - people exploring the playa
            ("staff", 0.10),       # 10% - rangers, medics, event staff
            ("heavy_user", 0.05)   # 5% - tech enthusiasts, frequent users
        ]

        # Disable plotting for large simulations
        self.PLOT = False

def place_burning_man_nodes(conf, num_clients):
    """Place routers and clients according to Burning Man layout"""
    nodes_config = []

    # Place 3 routers at Burning Man street addresses
    # Clock positions: 12:00 = north, clockwise
    # Radial streets: A=innermost, distance increases with letters
    # Approximate distances: A=400m, B=600m, C=800m, D=1000m, E=1200m, etc.
    router_configs = [
        {"clock": "7:30", "street": "B"},  # 7:30 & B
        {"clock": "3:00", "street": "E"},  # 3:00 & E
        {"clock": "10:00", "street": "F"},  # 10:00 & F
    ]

    # Convert clock positions to angles (12:00 = 90°, 3:00 = 0°, 6:00 = -90°, 9:00 = 180°)
    clock_to_angle = {
        "3:00": 0,
        "6:00": -90,
        "7:30": -135,  # Between 6:00 and 9:00
        "9:00": 180,
        "10:00": 150,  # Between 9:00 and 12:00
        "12:00": 90
    }

    # Street letter to radius mapping based on city dimensions
    # Esplanade borders the center plaza, then A-K span to city edge
    streets = ['ESPLANADE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    num_streets = len(streets)
    inner_radius = conf.CENTER_PLAZA_RADIUS  # Esplanade at center plaza edge
    outer_radius = conf.CITY_RADIUS * 0.9    # K street near city edge
    
    street_spacing = (outer_radius - inner_radius) / (num_streets - 1)
    street_to_radius = {}
    for i, street in enumerate(streets):
        street_to_radius[street] = inner_radius + (i * street_spacing)

    router_positions = []
    for config in router_configs:
        angle_deg = clock_to_angle[config["clock"]]
        # Apply same offset as fence to keep routers in fence's coordinate system
        angle_deg -= 45  # Opposite direction to match fence coordinate system
        radius = street_to_radius[config["street"]]

        # Convert to x,y coordinates
        angle_rad = np.radians(angle_deg)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        router_positions.append((x, y))

    for i, (x, y) in enumerate(router_positions):
        nodes_config.append({
            'x': x,
            'y': y,
            'z': conf.ROUTER_HEIGHT,
            'isRouter': True,
            'isRepeater': False,
            'isClientMute': False,
            'hopLimit': 7,  # Routers get higher hop limit
            'antennaGain': conf.ROUTER_ANTENNA_GAIN,
            'ptx': conf.ROUTER_PTX,
            'nodeId': i,
            'inGroundClutter': False,
            'userBehavior': 'router',
            'messagePeriod': -1,  # Routers don't generate user messages
            'positionPeriod': -1,  # Routers don't broadcast position (they're fixed)
            'isMobile': False
        })

    # Generate activity-based groups for realistic clustering
    activity_groups = generate_activity_groups(num_clients)

    # Place client nodes based on activity groups with clustering
    for group in activity_groups:
        group_type = group["type"]
        group_size = group["size"]

        # Find a suitable location for the group center
        group_center_placed = False
        attempts = 0
        while not group_center_placed and attempts < 100:
            # Get group configuration
            group_config = GROUP_CONFIG[group_type]
            
            # Randomly place group center anywhere on the square map for even distribution
            center_x = random.uniform(-conf.XSIZE/2 + 200, conf.XSIZE/2 - 200)
            center_y = random.uniform(-conf.YSIZE/2 + 200, conf.YSIZE/2 - 200)

            # Get cluster radius and minimum distance from configuration
            cluster_radius = group_config["cluster_radius"]
            min_group_distance = group_config["min_group_distance"]

            # Check if placement meets group's location preference
            distance_from_center = np.sqrt(center_x**2 + center_y**2)
            is_in_city = distance_from_center <= conf.CITY_RADIUS
            
            # Validate against group's city/playa zone preference
            zone_preference_met = True
            if random.random() < group_config["city_probability"]:
                # This group wants to be in city
                zone_preference_met = is_in_city
            else:
                # This group wants to be in playa
                zone_preference_met = not is_in_city

            # Check if group would be at least 75% inside trash fence
            fence_valid = check_group_fence_coverage(center_x, center_y, cluster_radius, conf.TRASH_FENCE_RADIUS)
            
            # Check distance from existing groups
            too_close_to_existing = False
            for existing in nodes_config:
                if calc_dist(center_x, existing['x'], center_y, existing['y']) < min_group_distance:
                    too_close_to_existing = True
                    break

            # Overall placement validation
            placement_acceptable = zone_preference_met and fence_valid and not too_close_to_existing

            if placement_acceptable:
                group_center_placed = True
            else:
                attempts += 1

        if not group_center_placed:
            # Fallback: place at random location if can't find good spot
            center_x = random.uniform(-conf.XSIZE/2 + 200, conf.XSIZE/2 - 200)
            center_y = random.uniform(-conf.YSIZE/2 + 200, conf.YSIZE/2 - 200)

        # Get group configuration for node placement
        group_config = GROUP_CONFIG[group_type]
        cluster_radius = group_config["cluster_radius"]

        # Place all nodes in this group
        for node_in_group in range(group_size):
            node_placed = False
            node_attempts = 0
            while not node_placed and node_attempts < 50:
                if cluster_radius == 0:
                    # Solo node at exact center
                    x, y = center_x, center_y
                else:
                    # Gaussian clustering around center
                    x = center_x + random.gauss(0, cluster_radius)
                    y = center_y + random.gauss(0, cluster_radius)

                # Check minimum distance from other nodes
                too_close = False
                for other in nodes_config:
                    if calc_dist(x, other['x'], y, other['y']) < conf.MINDIST:
                        too_close = True
                        break

                if not too_close:
                    # Assign user behavior and other properties
                    user_behavior = assign_user_behavior(conf)
                    behavior_config = conf.USER_BEHAVIORS[user_behavior]

                    is_mobile = random.random() < behavior_config["movement_probability"]

                    initial_position_delay = 0
                    if is_mobile and behavior_config["position_period"] > 0:
                        initial_position_delay = random.uniform(0, behavior_config["position_period"])

                    # Determine ground clutter and radio shadows based on location
                    distance_from_center = np.sqrt(x**2 + y**2)
                    in_ground_clutter = distance_from_center <= conf.CITY_RADIUS

                    if in_ground_clutter:
                        in_radio_shadow = random.random() < conf.RADIO_SHADOW_PROBABILITY
                    else:
                        in_radio_shadow = random.random() < (conf.RADIO_SHADOW_PROBABILITY * 0.3)

                    nodes_config.append({
                        'x': x,
                        'y': y,
                        'z': conf.CLIENT_HEIGHT,
                        'isRouter': False,
                        'isRepeater': False,
                        'isClientMute': False,
                        'hopLimit': conf.hopLimit,
                        'antennaGain': conf.CLIENT_ANTENNA_GAIN,
                        'ptx': conf.CLIENT_PTX,
                        'nodeId': len(nodes_config),
                        'inGroundClutter': in_ground_clutter,
                        'inRadioShadow': in_radio_shadow,
                        'userBehavior': user_behavior,
                        'messagePeriod': behavior_config["message_period"],
                        'positionPeriod': behavior_config["position_period"] if is_mobile else -1,
                        'isMobile': is_mobile,
                        'initialPositionDelay': initial_position_delay,
                        'activityGroup': group_type
                    })
                    node_placed = True

                node_attempts += 1

            if not node_placed:
                print(f"Warning: Could not place node {node_in_group+1} in {group_type} group")

    # Enforce fence boundary: move any nodes outside trash fence to just inside
    nodes_config = enforce_fence_boundary(nodes_config, conf)

    return nodes_config

def enforce_fence_boundary(nodes_config, conf):
    """Move any nodes outside the trash fence to just inside the boundary"""
    # Get the fence coordinates (reuse the calculation from plotting)
    import numpy as np

    # Recreate fence coordinates (same as in plot_node_locations)
    gps_coords = [
        (40.78236, -119.23530),  # P1
        (40.80570, -119.21965),  # P2
        (40.80163, -119.18533),  # P3
        (40.77568, -119.17971),  # P4
        (40.76373, -119.21050),  # P5
    ]

    # Convert to meters and scale (same logic as plotting)
    center_lat = sum(coord[0] for coord in gps_coords) / len(gps_coords)
    center_lon = sum(coord[1] for coord in gps_coords) / len(gps_coords)

    def gps_to_meters(lat, lon, center_lat, center_lon):
        lat_m = (lat - center_lat) * 111000
        lon_m = (lon - center_lon) * 111000 * np.cos(np.radians(center_lat))
        return lon_m, lat_m

    fence_coords_m = [gps_to_meters(lat, lon, center_lat, center_lon) for lat, lon in gps_coords]
    current_coords = np.array(fence_coords_m)
    current_radius = np.max(np.sqrt(current_coords[:, 0]**2 + current_coords[:, 1]**2))
    scale_factor = conf.TRASH_FENCE_RADIUS / current_radius

    fence_points = [(coord[0] * scale_factor, coord[1] * scale_factor) for coord in fence_coords_m]

    # Apply fence rotation (same as plotting)
    distances = [np.sqrt(x**2 + y**2) for x, y in fence_points]
    apex_idx = np.argmax(distances)
    current_apex_angle = np.arctan2(fence_points[apex_idx][1], fence_points[apex_idx][0])
    target_apex_angle = np.radians(45)
    rotation_angle = target_apex_angle - current_apex_angle

    # Rotate fence points
    rotated_fence = []
    for x, y in fence_points:
        rotated_x = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
        rotated_y = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
        rotated_fence.append((rotated_x, rotated_y))

    # Check each node and move if outside fence
    nodes_moved = 0
    for node in nodes_config:
        if node['isRouter']:
            continue  # Don't move routers

        node_x, node_y = node['x'], node['y']

        # Simple point-in-polygon test and boundary enforcement
        if not point_in_polygon(node_x, node_y, rotated_fence):
            # Find nearest point on fence boundary
            nearest_x, nearest_y = find_nearest_fence_point(node_x, node_y, rotated_fence)

            # Move slightly inside fence (2% closer to center)
            center_x, center_y = 0, 0  # Fence center
            inward_factor = 0.98

            node['x'] = center_x + (nearest_x - center_x) * inward_factor
            node['y'] = center_y + (nearest_y - center_y) * inward_factor
            nodes_moved += 1

    if nodes_moved > 0:
        print(f"Moved {nodes_moved} nodes inside trash fence boundary")

    return nodes_config

def point_in_polygon(x, y, polygon):
    """Check if point is inside polygon using ray casting"""
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def find_nearest_fence_point(x, y, polygon):
    """Find nearest point on polygon boundary"""
    min_dist = float('inf')
    nearest_x, nearest_y = x, y

    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Find nearest point on line segment
        nx, ny = nearest_point_on_segment(x, y, p1[0], p1[1], p2[0], p2[1])
        dist = np.sqrt((x - nx)**2 + (y - ny)**2)

        if dist < min_dist:
            min_dist = dist
            nearest_x, nearest_y = nx, ny

    return nearest_x, nearest_y

def nearest_point_on_segment(px, py, x1, y1, x2, y2):
    """Find nearest point on line segment"""
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return x1, y1

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp to segment

    return x1 + t * dx, y1 + t * dy

def run_burning_man_simulation(num_clients=100, enable_plotting=False):
    """Run the Burning Man mesh simulation"""
    conf = BurningManConfig()
    conf.NR_NODES = num_clients + conf.ROUTER_COUNT

    print(f"\n=== Burning Man Mesh Simulation ===")
    print(f"Total nodes: {conf.NR_NODES}")
    print(f"Routers: {conf.ROUTER_COUNT}")
    print(f"Clients: {num_clients}")
    print(f"Modem: {conf.MODEM_PRESET}")
    print(f"Simulation time: {conf.SIMTIME/1000}s")
    print(f"Path loss model: Light clutter {conf.LIGHT_CLUTTER_MEAN}±{conf.LIGHT_CLUTTER_STD:.0f}dB, Heavy clutter {conf.HEAVY_CLUTTER_MEAN}±{conf.HEAVY_CLUTTER_STD:.0f}dB (field calibrated)")
    print(f"Radio shadow: {conf.RADIO_SHADOW_PROBABILITY*100:.0f}% probability with {conf.RADIO_SHADOW_MEAN}±{conf.RADIO_SHADOW_STD:.0f}dB loss")

    # Generate node placement
    node_configs = place_burning_man_nodes(conf, num_clients)

    # Save configuration
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out", "burningManConfig.yaml"), 'w') as file:
        yaml.dump(node_configs, file)

    # Precompute connectable nodes matrix (major performance optimization)
    connectivity_matrix, baseline_path_loss_matrix, rebroadcast_priority_matrix, connectivity_stats = precompute_connectable_nodes(conf, node_configs)

    # Generate plot if requested
    if enable_plotting:
        plot_node_locations(node_configs, conf, connectivity_matrix, baseline_path_loss_matrix)

    # Store in config for packet creation
    conf.CONNECTIVITY_MATRIX = connectivity_matrix
    conf.BASELINE_PATH_LOSS_MATRIX = baseline_path_loss_matrix
    conf.REBROADCAST_PRIORITY_MATRIX = rebroadcast_priority_matrix

    # Setup simulation
    random.seed(conf.SEED)
    env = simpy.Environment()
    bc_pipe = BroadcastPipe(env)

    # Create nodes
    nodes = []
    messages = []
    packets = []
    delays = []
    packetsAtN = [[] for _ in range(conf.NR_NODES)]
    messageSeq = {"val": 0}

    for nodeId, nodeConfig in enumerate(node_configs):
        # Override PTX based on node type
        if nodeConfig['isRouter']:
            conf.PTX = nodeConfig['ptx']
        else:
            conf.PTX = nodeConfig['ptx']

        # Set interference level based on location
        if nodeConfig['inGroundClutter']:
            conf.INTERFERENCE_LEVEL = conf.INTERFERENCE_IN_CITY
        else:
            conf.INTERFERENCE_LEVEL = conf.INTERFERENCE_OUTSIDE

        # Use custom message period for realistic user behavior
        message_period = nodeConfig.get('messagePeriod', conf.PERIOD)
        if message_period <= 0:  # Routers and non-messaging nodes
            message_period = conf.SIMTIME * 10  # Very long period (effectively no messages)

        # Use quiet print function for large simulations
        verbose_print = print if conf.VERBOSE_MESH_TRAFFIC else lambda *args, **kwargs: None

        node = OptimizedMeshNode(
            conf, nodes, env, bc_pipe, nodeId, message_period,
            messages, packetsAtN, packets, delays, nodeConfig,
            messageSeq, verbose_print
        )

        # Store the full config for path loss calculations
        node.nodeConfig = nodeConfig

        nodes.append(node)

    # Setup asymmetric links between all nodes
    totalPairs, symmetricLinks, asymmetricLinks, noLinks = setup_asymmetric_links(conf, nodes)

    # Run simulation with progress reporting
    print("\n====== START OF SIMULATION ======")

    # Set up progress reporting every 10%
    progress_intervals = [conf.SIMTIME * i / 10 for i in range(1, 11)]
    current_progress_idx = 0

    # Custom progress tracking function
    def check_progress():
        nonlocal current_progress_idx
        if current_progress_idx < len(progress_intervals) and env.now >= progress_intervals[current_progress_idx]:
            progress_percent = (current_progress_idx + 1) * 10
            elapsed_time = env.now / 1000  # Convert to seconds
            print(f"Progress: {progress_percent}% complete ({elapsed_time:.1f}s simulated)")
            current_progress_idx += 1

    # Run simulation with periodic progress checks
    while env.now < conf.SIMTIME:
        # Run for small time increments to check progress
        next_check = min(env.now + conf.SIMTIME / 100, conf.SIMTIME)  # Check every 1% of sim time
        env.run(until=next_check)
        check_progress()

    # Calculate statistics
    print("\n====== END OF SIMULATION ======")
    print("*" * 40)

    # Separate statistics for routers and clients
    router_nodes = [n for n in nodes if n.isRouter]
    client_nodes = [n for n in nodes if not n.isRouter]
    city_nodes = [n for n in client_nodes if n.nodeConfig['inGroundClutter']]
    playa_nodes = [n for n in client_nodes if not n.nodeConfig['inGroundClutter']]

    print(f"\nRouter nodes: {len(router_nodes)}")
    print(f"Client nodes in city: {len(city_nodes)}")
    print(f"Client nodes in playa: {len(playa_nodes)}")

    # User behavior statistics
    user_behavior_counts = {}
    mobile_counts = {}

    for node in client_nodes:
        behavior = node.nodeConfig['userBehavior']
        is_mobile = node.nodeConfig['isMobile']

        user_behavior_counts[behavior] = user_behavior_counts.get(behavior, 0) + 1
        if is_mobile:
            mobile_counts[behavior] = mobile_counts.get(behavior, 0) + 1

    print(f"\nUser Behavior Distribution:")
    for behavior, count in user_behavior_counts.items():
        mobile_count = mobile_counts.get(behavior, 0)
        percentage = (count / len(client_nodes)) * 100
        mobile_percentage = (mobile_count / count) * 100 if count > 0 else 0
        behavior_config = conf.USER_BEHAVIORS[behavior]
        avg_period_hrs = behavior_config["message_period"] / conf.ONE_HR_INTERVAL
        print(f"  {behavior.title()}: {count} nodes ({percentage:.1f}%) - {mobile_count} mobile ({mobile_percentage:.1f}%) - Avg msg: {avg_period_hrs:.1f}h")

    # Count nodes in radio shadows
    shadow_nodes = [n for n in client_nodes if n.nodeConfig.get('inRadioShadow', False)]
    print(f"\nNodes in radio shadows: {len(shadow_nodes)} ({len(shadow_nodes)/len(client_nodes)*100:.1f}% of clients)")

    # Overall statistics
    print(f'\nMessages created: {messageSeq["val"]}')
    sent = len(packets)
    print(f'Packets sent: {sent}')

    nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid]])
    print(f"Collisions: {nrCollisions}")

    nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid]])
    print(f"Packets received: {nrReceived}")

    # Reachability by node type
    city_reach = sum([n.usefulPackets for n in city_nodes]) / (messageSeq["val"] * len(city_nodes)) * 100 if city_nodes else 0
    playa_reach = sum([n.usefulPackets for n in playa_nodes]) / (messageSeq["val"] * len(playa_nodes)) * 100 if playa_nodes else 0

    print(f"\nReachability:")
    print(f"  City nodes: {round(city_reach, 2)}%")
    print(f"  Playa nodes: {round(playa_reach, 2)}%")

    # Air utilization
    router_air = sum([n.txAirUtilization for n in router_nodes])/len(router_nodes)/conf.SIMTIME*100 if router_nodes else 0
    client_air = sum([n.txAirUtilization for n in client_nodes])/len(client_nodes)/conf.SIMTIME*100 if client_nodes else 0

    print(f"\nAverage Tx air utilization:")
    print(f"  Routers: {round(router_air, 2)}%")
    print(f"  Clients: {round(client_air, 2)}%")

    # Calculate expected vs actual message rates
    expected_msgs_per_hour = 0
    for behavior, count in user_behavior_counts.items():
        behavior_config = conf.USER_BEHAVIORS[behavior]
        msgs_per_hour = count / (behavior_config["message_period"] / conf.ONE_HR_INTERVAL)
        expected_msgs_per_hour += msgs_per_hour
        print(f"  Expected {behavior} traffic: {msgs_per_hour:.2f} msgs/hour from {count} nodes")

    actual_msgs_per_hour = messageSeq["val"] / (conf.SIMTIME / conf.ONE_HR_INTERVAL)
    print(f"\nTraffic Analysis:")
    print(f"  Expected: {expected_msgs_per_hour:.2f} messages/hour")
    print(f"  Actual: {actual_msgs_per_hour:.2f} messages/hour")
    print(f"  Traffic reduction vs 30s period: {round((conf.SIMTIME/1000/30) / actual_msgs_per_hour * len(client_nodes), 1)}x less traffic")

    # Save node placement and report plotting status
    print(f"\nNode placement saved to out/burningManConfig.yaml")
    if enable_plotting:
        print("Simulation complete - plot saved to out/graphics/burning_man_nodes.png")
    else:
        print("Simulation complete - use --plot to generate visualization")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Burning Man Mesh Network Simulation')
    parser.add_argument('num_clients', type=int, nargs='?', default=100,
                       help='Number of client nodes (default: 100)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and display node placement plot')

    args = parser.parse_args()

    run_burning_man_simulation(args.num_clients, enable_plotting=args.plot)
