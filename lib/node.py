#!/usr/bin/env python3
from enum import Enum
import logging
import math
import random

import simpy

from lib.common import find_random_position, node_antenna_height
from lib.config import Config
from lib.discrete_event_sim_components import SimulationState, SimulationDataTracking
from lib.geo import valid_lat_lon
from lib.mac import set_transmit_delay, get_retransmission_msec
from lib.phy import check_collision, is_channel_active, airtime
from lib.packet import NODENUM_BROADCAST, MeshPacket, MeshMessage
from lib.phy import estimate_path_loss
from lib.point import Point
from lib.terrain import NODE_Z_REFERENCE_SEA_LEVEL, apply_terrain_altitude

logger = logging.getLogger(__name__)

# roles taken from the protobuf config meshtastic/config.proto in https://github.com/meshtastic/protobufs
# deprecated roles are included for simulation utility
class MESHTASTIC_ROLE(Enum):
    CLIENT = 'CLIENT'
    CLIENT_MUTE = 'CLIENT_MUTE'
    ROUTER = 'ROUTER'
    ROUTER_CLIENT = 'ROUTER_CLIENT'
    REPEATER = 'REPEATER'
    TRACKER = 'TRACKER'
    SENSOR = 'SENSOR'
    TAK = 'TAK'
    CLIENT_HIDDEN = 'CLIENT_HIDDEN'
    LOST_AND_FOUND = 'LOST_AND_FOUND'
    TAK_TRACKER = 'TAK_TRACKER'
    ROUTER_LATE = 'ROUTER_LATE'
    CLIENT_BASE = 'CLIENT_BASE'

class MeshNodeStats:
    """Statistics, monitoring, and data tracking only relevant to and entirely
    internal to a single particular node
    """

    def __init__(self, nodeid: int):
        self.nodeid = nodeid

        self.packetsHeard = 0
        self.packetsRebroadcast = 0

    def get_stats_dictionary(self) -> dict:
        """Return dictionary holding all internal data
        (may not need this)
        """
        data = {
            "nodeid": self.nodeid,
            "packetsHeard": self.packetsHeard,
            "packetsRebroadcast": self.packetsRebroadcast,
        }
        return data

class NodeConfig:
    """Specific configuration for a node
    """
    def __init__(self, node_id: int, position: Point, period: int, tx_power: int, freq: float, role: MESHTASTIC_ROLE = MESHTASTIC_ROLE.CLIENT, antenna_gain: float = 0, hop_limit: int = 3, neighbor_info: bool = False, antenna_height=None, absolute_altitude=None):
        """Initial configuration of a node

        Arguments:
        node_id -- unique integer id of node (used as list index)
        position -- beginning Point(x, y, z) location of node
        period -- how often to generate messages. Average of an exponential distribution.
        tx_power -- transmit power in dB
        freq -- frequency in Hz
        role -- Meshtastic firmware role. Default: CLIENT
        antenna_gain -- antenna gain in dBi. Default 0
        hop_limit -- hop limit. Default 3
        neighbor_info -- if neighbor info is enabled. Default False
        antenna_height -- antenna height above local ground. Default: position.z
        absolute_altitude -- optional map-reported absolute altitude in meters
        """
        self.node_id = node_id
        self.position = position.copy() # make sure we keep our own point
        self.period = period
        self.tx_power = tx_power
        self.freq = freq
        self.role = role
        self.antenna_gain = antenna_gain
        self.hop_limit = hop_limit
        self.neighbor_info = neighbor_info
        self.antenna_height = position.z if antenna_height is None else antenna_height
        self.absolute_altitude = absolute_altitude

    @classmethod
    def from_gen_scenario_output(cls, node_id: int, node_dict: {}, period: int, tx_power: int, freq: float):
        """create NodeConfig from a node dict as returned from gen_scenario.
        You probably want to iterate over the keys that function gives you
        and pass individual values indexed by them to this method.

        Arguments:
        node_dict -- dictionary defining a single node. From gen_scenario.
        period -- how often to generate messages. Average of an exponential distribution.
        tx_power -- transmit power in dB
        freq -- frequency in Hz
        """
        nd = node_dict
        position = Point(nd['x'], nd['y'], nd['z'])

        # roles
        isRouter = nd['isRouter']
        isRepeater = nd['isRepeater']
        isClientMute = nd['isClientMute']

        # sanity check that only one role is set
        if (isRouter and isRepeater) or \
           (isRepeater and isClientMute) or \
           (isClientMute and isRouter):
           raise Exception(f"invalid combination of roles: {nd}")

        if isRouter:
            role = MESHTASTIC_ROLE.ROUTER
        elif isRepeater:
            role = MESHTASTIC_ROLE.REPEATER
        elif isClientMute:
            role = MESHTASTIC_ROLE.CLIENT_MUTE
        else:
            role = MESHTASTIC_ROLE.CLIENT

        antenna_height = nd.get("antennaHeight", nd["z"])
        absolute_altitude = nd.get("absoluteAltitude")
        return NodeConfig(node_id, position, period, tx_power, freq, role, nd['antennaGain'], nd['hopLimit'], nd['neighborInfo'], antenna_height, absolute_altitude)

    def compute_rssi_and_pathloss_to(self, rx_nodeconf, conf: Config) -> (float, float):
        """Compute RSSI and pathloss from this node config as the transmitting node
        to a receiving node, using a given config for various physical parameters.

        Arguments:
        rx_nodeconf -- NodeConfig of node we are transmitting to
        conf -- Config object specifying various physical parameters

        Returns:
        (rssi, pathloss) -- rssi at rx_nodeconf, and pathloss along the path
        """
        if self.node_id == rx_nodeconf.node_id:
            raise ValueError(f"Calculating rssi/pathloss between identical nodes is invalid. Node ID {self.node_id}")

        # compute path loss
        dist = self.position.euclidean_distance(rx_nodeconf.position)
        pl = estimate_path_loss(conf, dist, self.freq, node_antenna_height(self), node_antenna_height(rx_nodeconf))
        rssi = self.tx_power + self.antenna_gain + rx_nodeconf.antenna_gain - pl

        return (rssi, pl)


def node_configs_from_yaml(raw_config, period: int, tx_power: int = 30, freq: float = 902e6) -> list[NodeConfig]:
    """Convert saved node YAML into NodeConfig objects.

    The GUI writes a plain `{node_id: node_fields}` map. Real-mesh scenario
    files may wrap the same map under `nodes` so they can also store geographic
    origin metadata. Accept both shapes here so saved scenarios can be fed back
    into the normal simulator CLI.
    """
    if isinstance(raw_config, dict) and "nodes" in raw_config:
        node_map = raw_config["nodes"]
    else:
        node_map = raw_config

    if not isinstance(node_map, dict):
        raise ValueError("node YAML must be a node map or an object with a 'nodes' map")

    configs = []
    for sim_node_id, node_dict in enumerate(node_map.values()):
        configs.append(NodeConfig.from_gen_scenario_output(sim_node_id, node_dict, period, tx_power, freq))
    return configs


def origin_from_yaml(raw_config):
    """Return `(lat, lon)` origin metadata from wrapped scenario YAML if present."""
    if not isinstance(raw_config, dict):
        return None

    origin = raw_config.get("origin")
    if not isinstance(origin, dict) or "lat" not in origin or "lon" not in origin:
        return None

    try:
        lat = float(origin["lat"])
        lon = float(origin["lon"])
    except (TypeError, ValueError) as err:
        raise ValueError("origin.lat and origin.lon must be finite numbers") from err

    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("origin.lat and origin.lon must be finite numbers")
    if not valid_lat_lon(lat, lon):
        raise ValueError("origin.lat and origin.lon must be valid latitude/longitude degrees")

    return lat, lon

class MeshNode:
    """Class containing all the particular state of a MeshNode, references to necessary
    external resources like the simpy env, and process functions for simulation
    """
    def __init__(self, conf, sim_state: SimulationState, data_tracking: SimulationDataTracking, nodeConfig: NodeConfig):
        """Create a MeshNode. Houses all node-specific state, sim processes, and
        connections to broader sim environment and data collection.

        Arguments:
        conf -- Config object of various sim parameters
        sim_state -- object holding all mutating state of the simulation
        data_tracking -- object holding data collected from sim, doesn't influence state.
        nodeConfig -- initial configuration of node
        """
        self.conf = conf

        # initially to move repeated rssi/pathloss computation into NodeConfig class.
        # maybe move other state/config (role, period, etc.) into here explicitly
        # rather than binding to a member variable
        self.node_conf = nodeConfig

        self.nodeid = self.node_conf.node_id

        # set up internal RNGs
        self.moveRng = random.Random(self.nodeid)
        self.nodeRng = random.Random(self.nodeid)
        self.rebroadcastRng = random.Random()

        # require the user to specify a node configuration now, including position
        self.position = self.node_conf.position # explicitly use position in node_conf
        self.role = self.node_conf.role
        self.hopLimit = self.node_conf.hop_limit
        self.antennaGain = self.node_conf.antenna_gain
        self.antennaHeight = self.node_conf.antenna_height
        self.absolute_altitude = self.node_conf.absolute_altitude
        self.period = self.node_conf.period

        # using this more like a struct than a proper object.
        self.my_stats = MeshNodeStats(self.nodeid)

        self.messageSeq = sim_state.messageSeq
        self.connectivity_map = sim_state.connectivity_map
        self.baseline_pathloss_matrix = sim_state.baseline_pathloss_matrix
        self.env = sim_state.env
        self.bc_pipe = sim_state.bc_pipe
        self.nodes = sim_state.nodes
        self.packetsAtN = sim_state.packetsAtN
        self.packets = sim_state.packets

        self.delays = data_tracking.delays
        self.messages = data_tracking.messages

        self.nrPacketsSent = 0
        self.timesReceived = {}
        self.isReceiving = []
        self.isTransmitting = False
        self.usefulPackets = 0
        self.txAirUtilization = 0
        self.airUtilization = 0
        self.droppedByDelay = 0
        self.rebroadcastPackets = 0
        self.isMoving = False
        self.gpsEnabled = False

        # Track last broadcast position/time
        self.lastBroadcastPosition = self.position.copy()
        self.lastBroadcastTime = 0

        # track total transmit time for the last 6 buckets (each is 10s in firmware logic)
        self.channelUtilization = [0] * self.conf.CHANNEL_UTILIZATION_PERIODS  # each entry is ms spent on air in that interval
        self.channelUtilizationIndex = 0  # which "bucket" is current
        self.prevTxAirUtilization = 0.0   # how much total tx air-time had been used at last sample

        self.env.process(self.track_channel_utilization())
        if not self.is_repeater:  # repeaters don't generate messages themselves
            self.env.process(self.generate_message())
        self.env.process(self.receive(self.bc_pipe.get_output_conn()))
        self.transmitter = simpy.Resource(self.env, 1)

        # start mobility if enabled
        if self.conf.MOVEMENT_ENABLED and self.moveRng.random() <= self.conf.APPROX_RATIO_NODES_MOVING:
            self.isMoving = True
            if self.moveRng.random() <= self.conf.APPROX_RATIO_OF_NODES_MOVING_W_GPS_ENABLED:
                self.gpsEnabled = True

            # Randomly assign a movement speed
            possibleSpeeds = [
                self.conf.WALKING_METERS_PER_MIN,  # e.g.,  96 m/min
                self.conf.BIKING_METERS_PER_MIN,   # e.g., 390 m/min
                self.conf.DRIVING_METERS_PER_MIN   # e.g., 1500 m/min
            ]
            self.movementStepSize = self.moveRng.choice(possibleSpeeds)

            self.env.process(self.move_node())

    @property
    def is_router(self):
        return self.role == MESHTASTIC_ROLE.ROUTER

    @property
    def is_repeater(self):
        return self.role == MESHTASTIC_ROLE.REPEATER

    @property
    def is_client_mute(self):
        return self.role == MESHTASTIC_ROLE.CLIENT_MUTE

    def track_channel_utilization(self):
        """
        Periodically compute how many seconds of airtime this node consumed
        over the last 10-second block and store it in the ring buffer.
        """
        while True:
            # Wait 10 seconds of simulated time
            yield self.env.timeout(self.conf.TEN_SECONDS_INTERVAL)

            curTotalAirtime = self.txAirUtilization  # total so far, in *milliseconds*
            blockAirtimeMs = curTotalAirtime - self.prevTxAirUtilization

            self.channelUtilization[self.channelUtilizationIndex] = blockAirtimeMs

            self.prevTxAirUtilization = curTotalAirtime
            self.channelUtilizationIndex = (self.channelUtilizationIndex + 1) % self.conf.CHANNEL_UTILIZATION_PERIODS

    def channel_utilization_percent(self) -> float:
        """
        Returns how much of the last 60 seconds (6 x 10s) this node spent transmitting, as a percent.
        """
        sumMs = sum(self.channelUtilization)
        # 6 intervals, each 10 seconds = 60,000 ms total
        # fraction = sum_ms / 60000, then multiply by 100 for percent
        return (sumMs / (self.conf.CHANNEL_UTILIZATION_PERIODS * self.conf.TEN_SECONDS_INTERVAL)) * 100.0

    def move_node(self):
        while True:

            # Pick a random direction and distance
            angle = 2 * math.pi * self.moveRng.random()
            distance = self.movementStepSize * self.moveRng.random()

            # Compute new position
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)

            leftBound = self.conf.OX - self.conf.XSIZE / 2
            rightBound = self.conf.OX + self.conf.XSIZE / 2
            bottomBound = self.conf.OY - self.conf.YSIZE / 2
            topBound = self.conf.OY + self.conf.YSIZE / 2

            # Then in moveNode:
            new_x = min(max(self.position.x + dx, leftBound), rightBound)
            new_y = min(max(self.position.y + dy, bottomBound), topBound)

            # Update node’s position
            self.position.update_xy(new_x, new_y)
            if (
                self.conf.TERRAIN_ENABLED
                and self.conf.TERRAIN_GRID is not None
                and self.conf.NODE_Z_REFERENCE == NODE_Z_REFERENCE_SEA_LEVEL
            ):
                apply_terrain_altitude(self.conf.TERRAIN_GRID, self)

            # update connectivity map:
            # - update for this node: we may have gained and lost reachable nodes
            # - new reachable nodes: add ourselves to their connectivity map entry
            # - lost reachable nodes: remove ourselves from their connectivity map entry
            if self.conf.ENABLE_CONNECTIVITY_MAP:
                # may need to deepcopy if we put more complex things in here
                old_reachable_set = self.connectivity_map[self.nodeid].copy()
                new_reachable_set = set()
                for rx_node in self.nodes:
                    if rx_node.nodeid == self.nodeid:
                        continue # skip self

                    (rssi, pl) = self.node_conf.compute_rssi_and_pathloss_to(rx_node.node_conf, self.conf)

                    # compare with extra margin (set based on 10-node standard test)
                    if rssi + self.conf.CONNECTIVITY_MAP_RSSI_MARGIN > self.conf.current_preset['sensitivity']:
                        new_reachable_set.add(rx_node.nodeid)

                    # cache path loss (it is symmetric, and static until one of the nodes moves)
                    self.baseline_pathloss_matrix[self.nodeid][rx_node.nodeid] = pl
                    self.baseline_pathloss_matrix[rx_node.nodeid][self.nodeid] = pl

                # calculate set differences to detect added and removed nodes
                lost_nodes = old_reachable_set.difference(new_reachable_set)
                gained_nodes = new_reachable_set.difference(old_reachable_set)

                logger.debug(f"{self.env.now:.3f} node {self.nodeid} moved. Connectivity change: -{len(lost_nodes)}, +{len(gained_nodes)}.")
                # TODO: -0, +0 case is very common. Skip what we can in this case.
                # update this node's connectivity map
                self.connectivity_map[self.nodeid] = new_reachable_set
                # add ourself to the connectivity map of every node we gained
                for node_id in gained_nodes:
                    self.connectivity_map[node_id].add(self.nodeid)
                # remove ourself from the connectivity map of every node we lost
                for node_id in lost_nodes:
                    self.connectivity_map[node_id].discard(self.nodeid)

                # connectivity map updated!

            if self.gpsEnabled:
                distanceTraveled = self.position.euclidean_distance(self.lastBroadcastPosition)
                logger.debug(f"{self.env.now:.3f} node {self.nodeid} checks last broadcast position distance: {distanceTraveled} from {self.lastBroadcastPosition} to {self.position}")
                timeElapsed = self.env.now - self.lastBroadcastTime
                if distanceTraveled >= self.conf.SMART_POSITION_DISTANCE_THRESHOLD and timeElapsed >= self.conf.SMART_POSITION_DISTANCE_MIN_TIME:
                    currentUtil = self.channel_utilization_percent()
                    if currentUtil < 25.0:
                        self.send_packet(NODENUM_BROADCAST, "POSITION")
                        self.lastBroadcastPosition.update_xy(self.position.x, self.position.y)
                        self.lastBroadcastTime = self.env.now
                    else:
                        logger.debug(f"{self.env.now:.3f} node {self.nodeid} SKIPS POSITION broadcast (util={currentUtil:.1f}% > 25%)")

            # Wait until next move
            nextMove = self.get_next_time(self.conf.ONE_MIN_INTERVAL)
            if nextMove >= 0:
                yield self.env.timeout(nextMove)
            else:
                break

    def send_packet(self, destId, type=""):
        """We have created a new message and wish to send it to the network
        """
        # increment the shared counter
        messageSeq = self.messageSeq.get()
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        p = MeshPacket(self.conf, self.nodes, self.nodeid, destId, self.nodeid, self.conf.PACKETLENGTH, messageSeq, self.env.now, True, False, None, self.env.now, self.connectivity_map, self.baseline_pathloss_matrix)
        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} generated {type} message {p.seq} to {destId}")
        self.packets.append(p)
        self.env.process(self.transmit(p))
        return p

    def get_next_time(self, period):
        nextGen = self.nodeRng.expovariate(1.0 / float(period))
        # do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
        if self.env.now+nextGen + self.hopLimit * airtime(self.conf, self.conf.current_preset["sf"], self.conf.current_preset["cr"], self.conf.PACKETLENGTH, self.conf.current_preset["bw"]) < self.conf.SIMTIME:
            return nextGen
        return -1
    

    def was_seen_recently(self, packet, ownTransmit=False):
        if packet.seq not in self.timesReceived:
            # First time we know about this packet
            self.timesReceived[packet.seq] = 0 if ownTransmit else 1
            if not ownTransmit:
                self.usefulPackets += 1
        else:
            self.timesReceived[packet.seq] += 0 if ownTransmit else 1


    def perhaps_cancel_dupe(self, packet):
        # Cancel if we've already seen this sequence number
        if packet.seq in self.timesReceived:
            return self.timesReceived[packet.seq] > 2 if self.is_router or self.is_repeater else self.timesReceived[packet.seq] > 1
        return False


    def generate_message(self):
        while True:
            # Returns -1 if we don't make it before the sim ends
            nextGen = self.get_next_time(self.period)
            # do not generate a message near the end of the simulation (otherwise flooding cannot finish in time)
            if nextGen >= 0:
                yield self.env.timeout(nextGen)

                if self.conf.DMs:
                    destId = self.nodeRng.choice([i for i in range(0, len(self.nodes)) if i is not self.nodeid])
                else:
                    destId = NODENUM_BROADCAST

                p = self.send_packet(destId)

                while p.wantAck:  # ReliableRouter: retransmit message if no ACK received after timeout
                    retransmissionMsec = get_retransmission_msec(self, p)
                    yield self.env.timeout(retransmissionMsec)

                    ackReceived = False  # check whether you received an ACK on the transmitted message
                    minRetransmissions = self.conf.maxRetransmission
                    for packetSent in self.packets:
                        if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
                            if packetSent.retransmissions < minRetransmissions:
                                minRetransmissions = packetSent.retransmissions
                            if packetSent.ackReceived:
                                ackReceived = True
                    if ackReceived:
                        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received ACK on generated message with seq. nr. {p.seq}")
                        break
                    else:
                        if minRetransmissions > 0:  # generate new packet with same sequence number
                            pNew = MeshPacket(self.conf, self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.connectivity_map, self.baseline_pathloss_matrix)
                            pNew.retransmissions = minRetransmissions - 1
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} wants to retransmit its generated packet to {destId} with seq.nr. {p.seq} minRetransmissions {minRetransmissions}")
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                        else:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} reliable send of {p.seq} failed.")
                            break
            else:  # do not send this message anymore, since it is close to the end of the simulation
                break

    def transmit(self, packet):
        with self.transmitter.request() as request:
            yield request

            # listen-before-talk from src/mesh/RadioLibInterface.cpp
            txTime = set_transmit_delay(self, packet)
            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} schedules tx. Picked wait time {txTime}")
            yield self.env.timeout(txTime)

            # wait when currently receiving or transmitting, or channel is active
            while any(self.isReceiving) or self.isTransmitting or is_channel_active(self, self.env):
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} delaying tx: busy Tx-ing {self.isTransmitting=} or Rx-ing {any(self.isReceiving)=}, else channel busy!")
                txTime = set_transmit_delay(self, packet)
                yield self.env.timeout(txTime)
            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} ends waiting for scheduled tx")

            # check if you received an ACK for this message in the meantime
            self.was_seen_recently(packet, ownTransmit=True)
            if not self.perhaps_cancel_dupe(packet):  # if you did not receive an ACK for this message in the meantime
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} started low level send {packet.unique_packet_seq} for msg {packet.seq} hopLimit {packet.hopLimit} original Tx {packet.origTxNodeId}")
                self.nrPacketsSent += 1
                for rx_node in self.nodes:
                    if packet.sensedByN[rx_node.nodeid]:
                        if check_collision(self.conf, self.env, packet, rx_node.nodeid, self.packetsAtN) == 0:
                            self.packetsAtN[rx_node.nodeid].append(packet)
                # packet's collidedAtN field is now computed/valid
                packet.startTime = self.env.now
                packet.endTime = self.env.now + packet.timeOnAir
                self.txAirUtilization += packet.timeOnAir
                self.airUtilization += packet.timeOnAir
                self.bc_pipe.put(packet) # queue for nodes to receive packet
                self.isTransmitting = True
                yield self.env.timeout(packet.timeOnAir)
                self.isTransmitting = False
            else:  # received ACK: abort transmit, remove from packets generated
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} in the meantime received ACK, abort packet with seq. nr {packet.unique_packet_seq} for msg {packet.seq}")
                self.packets.remove(packet)

    def receive(self, in_pipe):
        while True:
            p = yield in_pipe.get()

            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} fetches packet {p.unique_packet_seq} for msg {p.seq} from {p.txNodeId} from bc_pipe: sensed: {p.sensedByN[self.nodeid]} collided: {p.collidedAtN[self.nodeid]} on air: {p.onAirToN[self.nodeid]}")
            if p.sensedByN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
                if p.collidedAtN[self.nodeid]:
                    # this packet collided, so we can sense it but not decode it.
                    # Mark it as no-longer on air and leave further processing to
                    # the 'end of transmission' branch
                    p.onAirToN[self.nodeid] = False
                elif not self.isTransmitting:
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} started receiving packet {p.unique_packet_seq} for msg {p.seq} from {p.txNodeId}")
                    p.onAirToN[self.nodeid] = False
                    self.isReceiving.append(True)
                else:  # if you were currently transmitting, you could not have sensed it
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} was transmitting, so could not receive packet {p.unique_packet_seq} for msg {p.seq}")
                    p.sensedByN[self.nodeid] = False
                    p.onAirToN[self.nodeid] = False
            elif p.sensedByN[self.nodeid]:  # end of reception
                try:
                    self.isReceiving[self.isReceiving.index(True)] = False
                except Exception:
                    pass
                self.airUtilization += p.timeOnAir
                # begin receiving packet fine, but a collision begins before we finish receiving.
                if p.collidedAtN[self.nodeid]:
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} could not decode packet {p.unique_packet_seq}.")
                    continue
                p.receivedAtN[self.nodeid] = True
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received packet {p.unique_packet_seq} for msg {p.seq} with delay {round(self.env.now - p.genTime, 2)}") # TODO: better way to calculate delay for log
                self.delays.append(self.env.now - p.genTime)

                # Update history of received packets
                self.was_seen_recently(p)

                # check if implicit ACK for own generated message
                if p.origTxNodeId == self.nodeid:
                    if p.isAck:
                        logger.debug(f"Node {self.nodeid} received real ACK on generated message.")
                    else:
                        logger.debug(f"Node {self.nodeid} received implicit ACK on message sent.")
                    p.ackReceived = True
                    continue

                ackReceived = False
                realAckReceived = False
                for sentPacket in self.packets:
                    # check if ACK for message you currently have in queue
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == p.seq:
                        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received implicit ACK for message in queue.")
                        ackReceived = True
                        sentPacket.ackReceived = True
                    # check if real ACK for message sent
                    if sentPacket.origTxNodeId == self.nodeid and p.isAck and sentPacket.seq == p.requestId:
                        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received real ACK.")
                        realAckReceived = True
                        sentPacket.ackReceived = True

                # send real ACK if you are the destination and you did not yet send the ACK
                if p.wantAck and p.destId == self.nodeid and not any(pA.requestId == p.seq for pA in self.packets):
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} sends a flooding ACK.")
                    messageSeq = self.messageSeq.get()
                    self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, messageSeq))
                    pAck = MeshPacket(self.conf, self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, self.conf.ACKLENGTH, messageSeq, self.env.now, False, True, p.seq, self.env.now, self.connectivity_map, self.baseline_pathloss_matrix)
                    self.packets.append(pAck)
                    self.env.process(self.transmit(pAck))
                # Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
                elif not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
                    self.my_stats.packetsHeard += 1 # packets which could potentially be rebroadcast
                    # FloodingRouter: rebroadcast received packet
                    if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.MANAGED_FLOOD:
                        if not self.is_client_mute:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} schedules rebroadcast for received packet {p.unique_packet_seq} for msg {p.seq}")
                            self.my_stats.packetsRebroadcast += 1
                            pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.connectivity_map, self.baseline_pathloss_matrix)
                            pNew.hopLimit = p.hopLimit - 1
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                else:
                    self.droppedByDelay += 1

    def get_stats(self) -> MeshNodeStats:
        """Get internally-tracked statistics/data. Only valid after the sim ends.
        """
        return self.my_stats

def default_generate_node_list(conf: Config) -> [NodeConfig]:
    """Default function for randomly choosing node configurations for a simulation
    run, based on the provided config and desired number of nodes specified in
    the config.
    """
    # need to identically match RNG usage right now to pass the discrete sim
    # test. If we want to change the reference test, do that in a smaller change.

    node_configs = []

    # replicate default 'no prior config' setup:
    for i in range(conf.NR_NODES):
        # no specified node config, randomly generate one
        # get node's position
        x, y = find_random_position(conf, node_configs)
        z = conf.HM
        position = Point(x, y, z)

        # role
        isRouter = conf.router

        # map misc. booleans into single role
        if isRouter:
            role = MESHTASTIC_ROLE.ROUTER
        else:
            role = MESHTASTIC_ROLE.CLIENT

        # make NodeConfig object to pass to MeshNode constructor
        node_configs.append(NodeConfig(i, position, conf.PERIOD, conf.PTX, conf.FREQ, role))

    return node_configs
