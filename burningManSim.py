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

def calculate_burning_man_path_loss(conf, txNode, rxNode, dist, freq):
    """Calculate path loss with realistic ground clutter effects for Burning Man"""
    # Start with base path loss
    base_path_loss = phy.estimate_path_loss(conf, dist, freq, txNode.z, rxNode.z)

    # Add ground clutter loss if applicable
    additional_loss = 0

    # Get node configs
    tx_in_clutter = hasattr(txNode, 'nodeConfig') and txNode.nodeConfig.get('inGroundClutter', False)
    rx_in_clutter = hasattr(rxNode, 'nodeConfig') and rxNode.nodeConfig.get('inGroundClutter', False)

    # If transmitter is a router (elevated), check if receiver is in ground clutter
    if txNode.isRouter and rx_in_clutter:
        # Base ground clutter loss with random variation
        clutter_loss = conf.GROUND_CLUTTER_LOSS + random.uniform(-conf.GROUND_CLUTTER_LOSS_VARIANCE, conf.GROUND_CLUTTER_LOSS_VARIANCE)
        additional_loss += clutter_loss

        # Check if receiver is in radio shadow (blocked by large structures)
        if hasattr(rxNode, 'nodeConfig') and rxNode.nodeConfig.get('inRadioShadow', False):
            additional_loss += conf.RADIO_SHADOW_ADDITIONAL_LOSS

    # If receiver is a router (elevated), check if transmitter is in ground clutter
    elif rxNode.isRouter and tx_in_clutter:
        # Base ground clutter loss with random variation
        clutter_loss = conf.GROUND_CLUTTER_LOSS + random.uniform(-conf.GROUND_CLUTTER_LOSS_VARIANCE, conf.GROUND_CLUTTER_LOSS_VARIANCE)
        additional_loss += clutter_loss

        # Check if transmitter is in radio shadow
        if hasattr(txNode, 'nodeConfig') and txNode.nodeConfig.get('inRadioShadow', False):
            additional_loss += conf.RADIO_SHADOW_ADDITIONAL_LOSS

    # If both are clients in ground clutter, apply even more loss
    elif tx_in_clutter and rx_in_clutter:
        # Much higher loss for ground-to-ground communication
        clutter_loss = conf.GROUND_CLUTTER_LOSS * 1.8 + random.uniform(-conf.GROUND_CLUTTER_LOSS_VARIANCE, conf.GROUND_CLUTTER_LOSS_VARIANCE)
        additional_loss += clutter_loss

        # Either node in radio shadow makes communication nearly impossible
        if (hasattr(txNode, 'nodeConfig') and txNode.nodeConfig.get('inRadioShadow', False)) or \
           (hasattr(rxNode, 'nodeConfig') and rxNode.nodeConfig.get('inRadioShadow', False)):
            additional_loss += conf.RADIO_SHADOW_ADDITIONAL_LOSS

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

            if tx_config['isRouter'] and rx_in_clutter:
                # Use minimum clutter loss (best case)
                min_additional_loss += conf.GROUND_CLUTTER_LOSS - conf.GROUND_CLUTTER_LOSS_VARIANCE
                # Radio shadows make connection nearly impossible
                if rx_in_shadow:
                    min_additional_loss += conf.RADIO_SHADOW_ADDITIONAL_LOSS
            elif rx_config['isRouter'] and tx_in_clutter:
                min_additional_loss += conf.GROUND_CLUTTER_LOSS - conf.GROUND_CLUTTER_LOSS_VARIANCE
                if tx_in_shadow:
                    min_additional_loss += conf.RADIO_SHADOW_ADDITIONAL_LOSS
            elif tx_in_clutter and rx_in_clutter:
                # Ground-to-ground with minimum loss
                min_additional_loss += (conf.GROUND_CLUTTER_LOSS * 1.8) - conf.GROUND_CLUTTER_LOSS_VARIANCE
                # Either in shadow makes connection nearly impossible
                if tx_in_shadow or rx_in_shadow:
                    min_additional_loss += conf.RADIO_SHADOW_ADDITIONAL_LOSS

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

    # Analyze router connectivity
    print(f"\n📡 Router Connectivity Analysis:")
    for router_idx, router_config in enumerate([n for n in node_configs if n['isRouter']]):
        router_id = router_config['nodeId']
        connected_nodes = connectivity_matrix.get(router_id, [])

        # Count by node type and shadow status
        city_clients = 0
        city_shadow_clients = 0
        playa_clients = 0
        playa_shadow_clients = 0
        other_routers = 0

        for connected_id in connected_nodes:
            connected_config = node_configs[connected_id]
            if connected_config['isRouter']:
                other_routers += 1
            elif connected_config.get('inGroundClutter', False):
                if connected_config.get('inRadioShadow', False):
                    city_shadow_clients += 1
                else:
                    city_clients += 1
            else:
                if connected_config.get('inRadioShadow', False):
                    playa_shadow_clients += 1
                else:
                    playa_clients += 1

        print(f"   Router {router_id}: {len(connected_nodes)} total connections")
        print(f"     - {city_clients} city clients, {city_shadow_clients} city shadow clients")
        print(f"     - {playa_clients} playa clients, {playa_shadow_clients} playa shadow clients, {other_routers} routers")

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

def plot_node_locations(node_configs, conf):
    """Plot node locations without radio ranges"""
    plt.figure(figsize=(12, 12))

    # Separate nodes by type
    routers = [n for n in node_configs if n['isRouter']]
    clients = [n for n in node_configs if not n['isRouter'] and not n.get('isClientMute', False)]
    client_mutes = [n for n in node_configs if n.get('isClientMute', False)]

    # Plot routers (red)
    if routers:
        router_x = [n['x'] for n in routers]
        router_y = [n['y'] for n in routers]
        plt.scatter(router_x, router_y, c='red', s=100, marker='s', label=f'Routers ({len(routers)})', zorder=3)

    # Plot regular clients (blue)
    if clients:
        client_x = [n['x'] for n in clients]
        client_y = [n['y'] for n in clients]
        plt.scatter(client_x, client_y, c='blue', s=20, alpha=0.6, label=f'Clients ({len(clients)})', zorder=2)

    # Plot muted clients (black)
    if client_mutes:
        mute_x = [n['x'] for n in client_mutes]
        mute_y = [n['y'] for n in client_mutes]
        plt.scatter(mute_x, mute_y, c='black', s=20, alpha=0.8, label=f'Client Mute ({len(client_mutes)})', zorder=2)

    # Add city radius circle for reference
    city_circle = plt.Circle((0, 0), conf.CITY_RADIUS, fill=False, linestyle='--', color='gray', alpha=0.5)
    plt.gca().add_patch(city_circle)
    
    # Add Burning Man trash fence (pentagon perimeter) as vector lines
    import numpy as np
    # Create pentagon points around the trash fence radius
    angles = np.linspace(0, 2*np.pi, 6)  # 6 points to close the pentagon
    fence_x = conf.TRASH_FENCE_RADIUS * np.cos(angles)
    fence_y = conf.TRASH_FENCE_RADIUS * np.sin(angles)
    
    # Draw each fence segment as individual vectors
    for i in range(5):  # 5 sides of pentagon
        x_start, y_start = fence_x[i], fence_y[i]
        x_end, y_end = fence_x[i+1], fence_y[i+1]
        
        # Draw line segment
        plt.plot([x_start, x_end], [y_start, y_end], color='orange', linewidth=3, 
                label='Trash Fence' if i == 0 else '', zorder=1)
        
        # Draw arrow to show direction/vector nature
        mid_x, mid_y = (x_start + x_end) / 2, (y_start + y_end) / 2
        dx, dy = x_end - x_start, y_end - y_start
        plt.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), xytext=(mid_x, mid_y),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    # Set equal aspect ratio and labels
    plt.axis('equal')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Distance (meters)')
    plt.title('Burning Man Mesh Network - Node Placement')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add annotations
    plt.text(0, conf.CITY_RADIUS + 200, f'City Radius: {conf.CITY_RADIUS}m',
             ha='center', va='bottom', fontsize=10, color='gray')

    # Save and show
    os.makedirs("out/graphics", exist_ok=True)
    plt.savefig("out/graphics/burning_man_nodes.png", dpi=150, bbox_inches='tight')
    plt.show()

# Burning Man specific configuration
class BurningManConfig(Config):
    def __init__(self):
        super().__init__()

        # Black Rock City is roughly 2.5 miles (4km) in diameter
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
        self.CITY_RADIUS = 2000  # 2km radius for main city area
        self.TRASH_FENCE_RADIUS = 3000  # 3km radius trash fence (event perimeter)
        self.GROUND_CLUTTER_LOSS = 25  # Additional 25dB loss in city from structures
        self.GROUND_CLUTTER_LOSS_VARIANCE = 10  # Random variation in ground clutter (+/- 10dB)
        self.RADIO_SHADOW_PROBABILITY = 0.15  # 15% chance of being in complete radio shadow
        self.RADIO_SHADOW_ADDITIONAL_LOSS = 40  # Extra 40dB loss for radio shadows
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

    # Place 3 routers in a triangle pattern around center
    # This provides good coverage of the city
    router_positions = [
        (-1500, 0),     # 3 o'clock position
        (750, 1300),    # 11 o'clock position
        (750, -1300)    # 7 o'clock position
    ]

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

    # Place client nodes
    clients_in_city = int(num_clients * 0.75)
    clients_outside = num_clients - clients_in_city

    # Place 75% of clients within city radius with ground clutter
    for i in range(clients_in_city):
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            # Random position within city radius
            angle = random.uniform(0, 2 * np.pi)
            # Bias towards center with sqrt for uniform distribution
            radius = conf.CITY_RADIUS * np.sqrt(random.random())
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Check minimum distance from other nodes
            too_close = False
            for other in nodes_config:
                dist = calc_dist(x, other['x'], y, other['y'])
                if dist < conf.MINDIST:
                    too_close = True
                    break

            if not too_close:
                user_behavior = assign_user_behavior(conf)
                behavior_config = conf.USER_BEHAVIORS[user_behavior]

                # Determine if this user is "mobile" (affects position broadcasting)
                is_mobile = random.random() < behavior_config["movement_probability"]

                # Add random initial delay for position broadcasts (0 to full period)
                initial_position_delay = 0
                if is_mobile and behavior_config["position_period"] > 0:
                    initial_position_delay = random.uniform(0, behavior_config["position_period"])

                # Determine if node is in radio shadow (completely blocked by structures)
                in_radio_shadow = random.random() < conf.RADIO_SHADOW_PROBABILITY

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
                    'inGroundClutter': True,
                    'inRadioShadow': in_radio_shadow,
                    'userBehavior': user_behavior,
                    'messagePeriod': behavior_config["message_period"],
                    'positionPeriod': behavior_config["position_period"] if is_mobile else -1,
                    'isMobile': is_mobile,
                    'initialPositionDelay': initial_position_delay
                })
                placed = True
            attempts += 1

    # Place 25% of clients outside city in open playa
    for i in range(clients_outside):
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            # Place between city edge and simulation boundary
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(conf.CITY_RADIUS, conf.XSIZE/2 - 100)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Check minimum distance
            too_close = False
            for other in nodes_config:
                dist = calc_dist(x, other['x'], y, other['y'])
                if dist < conf.MINDIST:
                    too_close = True
                    break

            if not too_close:
                user_behavior = assign_user_behavior(conf)
                behavior_config = conf.USER_BEHAVIORS[user_behavior]

                # Determine if this user is "mobile" (affects position broadcasting)
                is_mobile = random.random() < behavior_config["movement_probability"]

                # Add random initial delay for position broadcasts (0 to full period)
                initial_position_delay = 0
                if is_mobile and behavior_config["position_period"] > 0:
                    initial_position_delay = random.uniform(0, behavior_config["position_period"])

                # Even playa nodes can have some radio shadows (art installations, RVs, etc.)
                in_radio_shadow = random.random() < (conf.RADIO_SHADOW_PROBABILITY * 0.3)  # 30% of city probability

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
                    'inGroundClutter': False,
                    'inRadioShadow': in_radio_shadow,
                    'userBehavior': user_behavior,
                    'messagePeriod': behavior_config["message_period"],
                    'positionPeriod': behavior_config["position_period"] if is_mobile else -1,
                    'isMobile': is_mobile,
                    'initialPositionDelay': initial_position_delay
                })
                placed = True
            attempts += 1

    return nodes_config

def run_burning_man_simulation(num_clients=100):
    """Run the Burning Man mesh simulation"""
    conf = BurningManConfig()
    conf.NR_NODES = num_clients + conf.ROUTER_COUNT

    print(f"\n=== Burning Man Mesh Simulation ===")
    print(f"Total nodes: {conf.NR_NODES}")
    print(f"Routers: {conf.ROUTER_COUNT}")
    print(f"Clients: {num_clients}")
    print(f"Modem: {conf.MODEM_PRESET}")
    print(f"Simulation time: {conf.SIMTIME/1000}s")
    print(f"Ground clutter loss: {conf.GROUND_CLUTTER_LOSS}±{conf.GROUND_CLUTTER_LOSS_VARIANCE}dB in city")
    print(f"Radio shadow probability: {conf.RADIO_SHADOW_PROBABILITY*100:.0f}% with +{conf.RADIO_SHADOW_ADDITIONAL_LOSS}dB loss")

    # Generate node placement
    node_configs = place_burning_man_nodes(conf, num_clients)

    # Save configuration
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out", "burningManConfig.yaml"), 'w') as file:
        yaml.dump(node_configs, file)

    # Enable plotting to show trash fence
    if conf.NR_NODES <= 100:  # Only plot for small simulations
        plot_node_locations(node_configs, conf)

    # Precompute connectable nodes matrix (major performance optimization)
    connectivity_matrix, baseline_path_loss_matrix, rebroadcast_priority_matrix, connectivity_stats = precompute_connectable_nodes(conf, node_configs)

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

    # Save node placement (but skip plotting for large simulations)
    print(f"\nNode placement saved to out/burningManConfig.yaml")
    print("Simulation complete - no plots generated for large node counts")

if __name__ == "__main__":
    # Default to 100 client nodes, or specify via command line
    num_clients = 100
    if len(sys.argv) > 1:
        num_clients = int(sys.argv[1])

    run_burning_man_simulation(num_clients)
