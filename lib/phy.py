import math
import random

from lib.config import Config

conf = Config()

VERBOSE = False


def verboseprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


#                           CAD duration   +     airPropagationTime+TxRxTurnaround+MACprocessing
SLOT_TIME = 8.5 * (2.0 ** conf.SFMODEM[conf.MODEM]) / conf.BWMODEM[conf.MODEM] * 1000 + 0.2 + 0.4 + 7


def check_collision(conf, env, packet, rx_nodeId, packetsAtN):
    # Check for collisions at rx_node
    col = 0
    if conf.COLLISION_DUE_TO_INTERFERENCE:
        if random.randrange(10) <= conf.INTERFERENCE_LEVEL * 10:
            packet.collidedAtN[rx_nodeId] = True

    if packetsAtN[rx_nodeId]:
        for other in packetsAtN[rx_nodeId]:
            if frequency_collision(packet, other) and sf_collision(packet, other):
                if timing_collision(conf, env, packet, other):
                    verboseprint(f'Packet nr. {packet.seq} from {packet.txNodeId} and packet nr. {other.seq} from {other.txNodeId} will collide!')
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


def frequency_collision(p1, p2):
    if abs(p1.freq - p2.freq) <= 120 and (p1.bw == 500 or p2.freq == 500):
        return True
    elif abs(p1.freq - p2.freq) <= 60 and (p1.bw == 250 or p2.freq == 250):
        return True
    elif abs(p1.freq - p2.freq) <= 30:
        return True
    return False


def sf_collision(p1, p2):
    return p1.sf == p2.sf


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


def is_channel_active(node, env):
    if random.randrange(10) <= node.conf.INTERFERENCE_LEVEL * 10:
        return True
    for p in node.packets:
        if p.detectedByN[node.nodeid]:
            # You will miss detecting a packet if it has just started before you could do CAD
            if p.startTime + SLOT_TIME <= env.now <= p.endTime:
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


def estimate_path_loss(conf, dist, freq, txZ=conf.HM, rxZ=conf.HM):
    # With randomized movements we may end up on top of another node which is problematic for log(dist)
    dist = max(dist, .001)

    # Log-Distance model
    if conf.MODEL == 0:
        Lpl = conf.LPLD0 + 10 * conf.GAMMA * math.log10(dist / conf.D0)

    # Okumura-Hata model
    elif 1 <= conf.MODEL <= 4:
        # small and medium-size cities
        if conf.MODEL == 1:
            ahm = (1.1 * (math.log10(freq) - 6.0) - 0.7) * rxZ - (1.56 * (math.log10(freq) - 6.0) - 0.8)
            C = 0
        # metropolitan areas
        elif conf.MODEL == 2:
            if freq <= 200000000:
                ahm = 8.29 * ((math.log10(1.54 * rxZ)) ** 2) - 1.1
            elif freq >= 400000000:
                ahm = 3.2 * ((math.log10(11.75 * rxZ)) ** 2) - 4.97
            C = 0
        # suburban environments
        elif conf.MODEL == 3:
            ahm = (1.1 * (math.log10(freq) - 6.0) - 0.7) * rxZ - (1.56 * (math.log10(freq) - 6.0) - 0.8)
            C = -2 * ((math.log10(freq) - math.log10(28000000)) ** 2) - 5.4
        # rural area
        elif conf.MODEL == 4:
            ahm = (1.1 * (math.log10(freq) - 6.0) - 0.7) * rxZ - (1.56 * (math.log10(freq) - 6.0) - 0.8)
            C = -4.78 * ((math.log10(freq) - 6.0) ** 2) + 18.33 * (math.log10(freq) - 6.0) - 40.98

        A = 69.55 + 26.16 * (math.log10(freq) - 6.0) - 13.82 * math.log10(txZ) - ahm
        B = 44.9 - 6.55 * math.log10(txZ)
        Lpl = A + B * (math.log10(dist) - 3.0) + C

    # 3GPP model
    elif 5 <= conf.MODEL < 7:
        # Suburban Macro
        if conf.MODEL == 5:
            C = 0  # dB
        # Urban Macro
        elif conf.MODEL == 6:
            C = 3  # dB

        Lpl = (44.9 - 6.55 * math.log10(txZ)) * (math.log10(dist) - 3.0) \
            + 45.5 + (35.46 - 1.1 * rxZ) * (math.log10(freq) - 6.0) \
            - 13.82 * math.log10(rxZ) + 0.7 * rxZ + C

    return Lpl


def zero_link_budget(dist):
    return conf.PTX + 2 * conf.GL - estimate_path_loss(conf, dist, conf.FREQ) - conf.SENSMODEM[conf.MODEM]


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

def zero_link_budget_with_gain(dist, gain):
    return conf.PTX + gain - estimate_path_loss(conf, dist, conf.FREQ) - conf.SENSMODEM[conf.MODEM]

def estimate_max_range(gain):
    return rootFinder(zero_link_budget_with_gain, 1500, args=(gain,))

MAXRANGE = rootFinder(zero_link_budget, 1500)
