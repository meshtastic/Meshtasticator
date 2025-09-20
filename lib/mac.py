import random
from lib.phy import airtime, SLOT_TIME


VERBOSE = False
CWmin = 2
CWmax = 8
PROCESSING_TIME_MSEC = 4500


def verboseprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def set_transmit_delay(node, packet):  # from RadioLibInterface::setTransmitDelay
    for p in reversed(node.packetsAtN[node.nodeid]):
        if p.seq == packet.seq and p.rssiAtN[node.nodeid] != 0 and p.receivedAtN[node.nodeid] is True:
            # verboseprint(round(self.env.now, 3), 'Pick delay with RSSI of node', self.nodeid, 'is', p.rssiAtN[self.nodeid])
            return get_tx_delay_msec_weighted(node, p.rssiAtN[node.nodeid])  # weighted waiting based on RSSI
    return get_tx_delay_msec(node)


def get_tx_delay_msec_weighted(node, rssi):  # from RadioInterface::getTxDelayMsecWeighted
    snr = rssi - node.conf.NOISE_LEVEL
    SNR_MIN = -20
    SNR_MAX = 15
    if snr < SNR_MIN:
        verboseprint(f'Minimum SNR at RSSI of {rssi} dBm')
        snr = SNR_MIN
    if snr > SNR_MAX:
        verboseprint(f'Maximum SNR at RSSI of {rssi} dBm')
        snr = SNR_MAX

    CWsize = int((snr - SNR_MIN) * (CWmax - CWmin) / (SNR_MAX - SNR_MIN) + CWmin)
    if node.isRouter:
        CW = random.randint(0, 2 * CWsize - 1)
    else:
        CW = random.randint(0, 2 ** CWsize - 1)
    verboseprint(f'Node {node.nodeid} has CW size {CWsize} and picked CW {CW}')
    return CW * SLOT_TIME


def get_tx_delay_msec(node):  # from RadioInterface::getTxDelayMsec
    channelUtil = node.airUtilization / node.env.now * 100
    CWsize = int(channelUtil * (CWmax - CWmin) / 100 + CWmin)
    CW = random.randint(0, 2 ** CWsize - 1)
    verboseprint(f'Current channel utilization is {channelUtil}, so picked CW {CW}')
    return CW * SLOT_TIME


def get_retransmission_msec(node, packet):  # from RadioInterface::getRetransmissionMsec
    packetAirtime = int(airtime(node.conf, node.conf.SFMODEM[node.conf.MODEM], node.conf.CRMODEM[node.conf.MODEM], packet.packetLen, node.conf.BWMODEM[node.conf.MODEM]))
    channelUtil = node.airUtilization / node.env.now * 100
    CWsize = int(channelUtil * (CWmax - CWmin) / 100 + CWmin)
    return 2 * packetAirtime + (2 ** CWsize + 2 ** (int((CWmax + CWmin) / 2))) * SLOT_TIME + PROCESSING_TIME_MSEC
