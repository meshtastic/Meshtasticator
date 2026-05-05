import logging
import math
import random

from lib.config import CONFIG

# TODO: if our config deviates from the default, we WILL get incorrect results.
# refactor things to take a config object
conf = CONFIG

logger = logging.getLogger(__name__)

# checked as of tag v2.7.15.567b8ea in meshtastic-firmware repo
NUM_SYM_CAD = 2
NUM_SYM_CAD_24GHZ = 4

#                           CAD duration   +     airPropagationTime+TxRxTurnaround+MACprocessing
def get_current_slot_time(): # from RadioInterface::computeSlotTimeMsec
    # all times in ms
    sum_prop_turnaround_mac_time = 0.2 + 0.4 + 7
    firmware_bw = conf.current_preset["bw"] / 1000 # convert Hz to KHz to match firmware
    symbol_time = (2.0 ** conf.current_preset["sf"]) / firmware_bw

    if conf.REGION['wide_lora']:
        # TODO: currently wide_lora isn't fully implemented
        # currently only 2.4GHz LoRa
        return (NUM_SYM_CAD_24GHZ + (2 * conf.current_preset['sf'] + 3) / 32) * symbol_time + sum_prop_turnaround_mac_time
    else:
        return max(2.25, NUM_SYM_CAD + 0.5) * symbol_time + sum_prop_turnaround_mac_time


def check_collision(conf, env, packet, rx_nodeId, packetsAtN):
    if conf.CAPTURE_COLLISION_MODEL_ENABLED:
        return check_capture_collision(conf, packet, rx_nodeId, packetsAtN)

    # Check for collisions at rx_node
    col = 0
    if conf.COLLISION_DUE_TO_INTERFERENCE:
        if random.randrange(10) <= conf.INTERFERENCE_LEVEL * 10:
            packet.collidedAtN[rx_nodeId] = True

    if packetsAtN[rx_nodeId]:
        for other in packetsAtN[rx_nodeId]:
            if frequency_collision(packet, other) and sf_collision(packet, other):
                if timing_collision(conf, env, packet, other):
                    logger.debug(f'Packet nr. {packet.unique_packet_seq} from {packet.txNodeId} and packet nr. {other.unique_packet_seq} from {other.txNodeId} will collide at node {rx_nodeId}!')
                    c = power_collision(packet, other, rx_nodeId)
                    # mark all the collided packets
                    for p in c:
                        p.collidedAtN[rx_nodeId] = True
                        if p == packet:
                            col = 1
                else:
                    pass  # no timing collision
        return col
    return 0


def check_capture_collision(conf, packet, rx_nodeId, packetsAtN):
    """Check overlap with a capture-aware same-SF collision model.

    The legacy model is intentionally preserved unless explicitly enabled. This
    path models the part that matters for real crowded meshes: a receiver can
    keep a sufficiently stronger packet through a weaker overlap, but equal or
    stronger interference during the preamble/header lock window destroys it.
    Later payload-only overlap is tolerated when it is only a short tail.
    """
    col = 0
    if conf.COLLISION_DUE_TO_INTERFERENCE and random.random() < conf.INTERFERENCE_LEVEL:
        mark_collision(packet, rx_nodeId, "external_interference")
        col = 1

    for other in packetsAtN[rx_nodeId]:
        if not intervals_overlap(packet.startTime, packet.endTime, other.startTime, other.endTime):
            continue
        if not frequency_collision(packet, other) or not sf_collision(packet, other):
            continue

        casualties = capture_collision_casualties(conf, packet, other, rx_nodeId)
        if casualties:
            logger.debug(
                f'Packet nr. {packet.seq} from {packet.txNodeId} and packet nr. '
                f'{other.seq} from {other.txNodeId} overlap at node {rx_nodeId}; '
                f'capture casualties {[p.seq for p, _ in casualties]}'
            )
        for casualty, reason in casualties:
            mark_collision(casualty, rx_nodeId, reason)
            if casualty == packet:
                col = 1
    return col


def frequency_collision(p1, p2):
    delta_khz = _frequency_delta_khz(p1, p2)
    p1_bw_khz = _bandwidth_khz(p1)
    p2_bw_khz = _bandwidth_khz(p2)

    if delta_khz <= 120 and (p1_bw_khz == 500 or p2_bw_khz == 500):
        return True
    elif delta_khz <= 60 and (p1_bw_khz == 250 or p2_bw_khz == 250):
        return True
    elif delta_khz <= 30:
        return True
    return False


def _frequency_delta_khz(p1, p2):
    """Return center-frequency separation in kHz.

    Meshtasticator stores modem frequencies in Hz. Some small tests and older
    LoRaSim-derived snippets use MHz-scale values, so normalize both shapes here
    instead of making the collision predicate depend on caller units.
    """
    delta = abs(p1.freq - p2.freq)
    if max(abs(p1.freq), abs(p2.freq)) > 1e6:
        return delta / 1000.0
    return delta * 1000.0


def _bandwidth_khz(packet):
    """Return LoRa bandwidth in kHz for both Hz and kHz-style packet fields."""
    return packet.bw / 1000.0 if packet.bw > 1000 else packet.bw


def sf_collision(p1, p2):
    return p1.sf == p2.sf


def mark_collision(packet, rx_nodeId, reason):
    packet.collidedAtN[rx_nodeId] = True
    if hasattr(packet, "collisionReasonAtN"):
        packet.collisionReasonAtN[rx_nodeId] = reason


def power_collision(p1, p2, rx_nodeId):
    powerThreshold = 6  # dB
    if abs(p1.rssiAtN[rx_nodeId] - p2.rssiAtN[rx_nodeId]) < powerThreshold:
        # packets are too close to each other, both collide
        # return both packets as casualties
        return (p1, p2)
    elif p1.rssiAtN[rx_nodeId] - p2.rssiAtN[rx_nodeId] < powerThreshold:
        # p2 overpowered p1, return p1 as casualty
        return (p1,)
    # p2 was the weaker packet, return it as a casualty
    return (p2,)


def timing_collision(conf, env, p1, p2):
    """ assuming p1 is the freshly arrived packet, check if the packet collides 
        or not (when only the first n - 5 preamble symbols overlap)
    """
    Tpreamb = 2 ** p1.sf / (1.0 * p1.bw) * (conf.NPREAM - 5)
    p1_cs = env.now + Tpreamb
    if p1_cs < p2.endTime:  # p1 collided with p2 and lost
        return True
    return False


def intervals_overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)


def overlap_ms(p1, p2):
    return max(0.0, min(p1.endTime, p2.endTime) - max(p1.startTime, p2.startTime))


def preamble_lock_window_ms(conf, packet):
    """Approximate the fragile LoRa preamble/header acquisition interval."""
    symbols = max(1, conf.NPREAM - 5)
    return symbols * (2 ** packet.sf) / packet.bw * 1000


def overlaps_preamble_lock(conf, victim, interferer):
    return intervals_overlap(
        victim.startTime,
        min(victim.endTime, victim.startTime + preamble_lock_window_ms(conf, victim)),
        interferer.startTime,
        interferer.endTime,
    )


def packet_survives_overlap(conf, victim, interferer, rx_nodeId):
    """Return whether `victim` survives this one overlapping interferer.

    This is still a compact simulator model, not a chip-level LoRa demodulator.
    It encodes the two big effects the binary model misses: capture by a packet
    that is at least COLLISION_CAPTURE_THRESHOLD_DB stronger at this receiver,
    and small late-tail overlap that does not destroy an already-locked packet.
    """
    desired_margin_db = victim.rssiAtN[rx_nodeId] - interferer.rssiAtN[rx_nodeId]
    if desired_margin_db >= conf.COLLISION_CAPTURE_THRESHOLD_DB:
        return True

    if overlaps_preamble_lock(conf, victim, interferer):
        return False

    fraction = overlap_ms(victim, interferer) / victim.timeOnAir if victim.timeOnAir > 0 else 1.0
    if fraction >= conf.COLLISION_PAYLOAD_OVERLAP_LOSS_FRACTION:
        return False

    return True


def capture_collision_casualties(conf, p1, p2, rx_nodeId):
    casualties = []
    if _packet_was_decodable_at_rx(p1, rx_nodeId) and not packet_survives_overlap(conf, p1, p2, rx_nodeId):
        casualties.append((p1, "capture_overlap"))
    if _packet_was_decodable_at_rx(p2, rx_nodeId) and not packet_survives_overlap(conf, p2, p1, rx_nodeId):
        casualties.append((p2, "capture_overlap"))
    return casualties


def _packet_was_decodable_at_rx(packet, rx_nodeId):
    """Return whether collision loss is meaningful for this packet.

    Capture mode tracks CAD-detectable-but-undecodable packets as interference
    energy. Those packets can jam another packet, but they should not inflate
    collision counters as failed decodes because they were below the receiver's
    demodulation threshold before overlap was considered.
    """
    sensed_by_node = getattr(packet, "sensedByN", None)
    if sensed_by_node is None:
        return True
    return sensed_by_node[rx_nodeId]


def is_channel_active(node, env):
    if random.randrange(10) <= node.conf.INTERFERENCE_LEVEL * 10:
        return True
    for p in node.packets:
        if p.detectedByN[node.nodeid]:
            # You will miss detecting a packet if it has just started before you could do CAD
            if p.startTime + get_current_slot_time() <= env.now <= p.endTime:
                return True
    return False


def airtime(conf, sf, cr, pl, bw):
    pl = pl + conf.HEADERLENGTH  # add Meshtastic header length
    H = 0  # implicit header disabled (H=0) or not (H=1)
    DE = 0  # low data rate optimization enabled (=1) or not (=0)

    if bw == 125e3 and sf in [11, 12]:  # low data rate optimization
        DE = 1
    if sf == 6:  # can only have implicit header with SF6
        H = 1

    Tsym = (2.0 ** sf) / bw
    Tpream = (conf.NPREAM + 4.25) * Tsym
    payloadSymbNB = 8 + max(math.ceil((8.0 * pl - 4.0 * sf + 28 + 16 - 20 * H) / (4.0 * (sf - 2 * DE))) * (cr + 4), 0)
    Tpayload = payloadSymbNB * Tsym

    return (Tpream + Tpayload) * 1000


def estimate_path_loss(conf, dist, freq, txZ=None, rxZ=None, model=None):
    '''Calculate path loss between transmitter and receiver using a specific model

    Arguments:
    conf -- config object
    dist -- distance between nodes in meters
    freq -- frequency in MHz
    txZ -- height of transmitter. Default: conf.HM
    rxZ -- height of receiver. Default: conf.HM
    model -- choice of model (currently integer in [0,6], default: conf.MODEL)

    Returns:
    path loss as float
    '''
    # With randomized movements we may end up on top of another node which is problematic for log(dist)
    #
    # Some real-mesh presets can also set a larger floor as an empirical
    # near-field/clutter calibration. The 3GPP/Hata formulas are not meaningful
    # at apartment-scale separations, and map node positions are coarse enough
    # that "two pins are close" does not mean "two antennas have clear 20 m RF".
    dist = max(dist, conf.PATH_LOSS_DISTANCE_FLOOR_M)
    if txZ is None:
        txZ = conf.HM
    if rxZ is None:
        rxZ = conf.HM
    if model is None:
        model = conf.MODEL

    # Log-Distance model
    if model == 0:
        Lpl = conf.LPLD0 + 10 * conf.GAMMA * math.log10(dist / conf.D0)

    # Okumura-Hata model
    elif 1 <= model <= 4:
        # small and medium-size cities
        if model == 1:
            ahm = (1.1 * (math.log10(freq) - 6.0) - 0.7) * rxZ - (1.56 * (math.log10(freq) - 6.0) - 0.8)
            C = 0
        # metropolitan areas
        elif model == 2:
            if freq <= 200000000:
                ahm = 8.29 * ((math.log10(1.54 * rxZ)) ** 2) - 1.1
            elif freq >= 400000000:
                ahm = 3.2 * ((math.log10(11.75 * rxZ)) ** 2) - 4.97
            C = 0
        # suburban environments
        elif model == 3:
            ahm = (1.1 * (math.log10(freq) - 6.0) - 0.7) * rxZ - (1.56 * (math.log10(freq) - 6.0) - 0.8)
            C = -2 * ((math.log10(freq) - math.log10(28000000)) ** 2) - 5.4
        # rural area
        elif model == 4:
            ahm = (1.1 * (math.log10(freq) - 6.0) - 0.7) * rxZ - (1.56 * (math.log10(freq) - 6.0) - 0.8)
            C = -4.78 * ((math.log10(freq) - 6.0) ** 2) + 18.33 * (math.log10(freq) - 6.0) - 40.98

        A = 69.55 + 26.16 * (math.log10(freq) - 6.0) - 13.82 * math.log10(txZ) - ahm
        B = 44.9 - 6.55 * math.log10(txZ)
        Lpl = A + B * (math.log10(dist) - 3.0) + C

    # 3GPP model
    elif 5 <= model < 7:
        # Suburban Macro
        if model == 5:
            C = 0  # dB
        # Urban Macro
        elif model == 6:
            C = 3  # dB

        Lpl = (44.9 - 6.55 * math.log10(txZ)) * (math.log10(dist) - 3.0) \
            + 45.5 + (35.46 - 1.1 * rxZ) * (math.log10(freq) - 6.0) \
            - 13.82 * math.log10(rxZ) + 0.7 * rxZ + C
    else:
        raise ValueError(f"unsupported path loss model: {model}")

    return Lpl


# TODO: take conf as parameter so we don't use this module's default conf
def zero_link_budget(dist):
    return conf.PTX + 2 * conf.GL - estimate_path_loss(conf, dist, conf.FREQ) - conf.current_preset["sensitivity"]


def rootFinder(func, x0, args=(), tol=1, maxiter=100):
  """Newton-Raphson root finder."""
  x = x0
  for _ in range(maxiter):
      fx = func(x, *args)
      dfx = (func(x + 1e-6, *args) - fx) / 1e-6
      if dfx == 0:
          print("Warning: could not estimate max. range")
          return x
      x_new = x - fx / dfx
      if abs(x_new - x) < tol:
          return x_new
      x = x_new
  print("Warning: could not estimate max. range")
  return x

# TODO: take conf as parameter so we don't use this module's default conf
def zero_link_budget_with_gain(dist, gain):
    return conf.PTX + gain - estimate_path_loss(conf, dist, conf.FREQ) - conf.current_preset["sensitivity"]

def estimate_max_range(gain):
    return rootFinder(zero_link_budget_with_gain, 1500, args=(gain,))

# TODO: take conf as parameter so we don't use this module's default conf
MAXRANGE = rootFinder(zero_link_budget, 1500)
