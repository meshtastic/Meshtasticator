import logging
import random

from lib.phy import airtime, get_current_slot_time
from lib.radio_loss import estimate_snr

logger = logging.getLogger(__name__)

CWmin = 2
CWmax = 8
PROCESSING_TIME_MSEC = 4500


def set_transmit_delay(node, packet):  # from RadioLibInterface::setTransmitDelay
    for p in reversed(node.packetsAtN[node.nodeid]):
        if p.seq == packet.seq and p.rssiAtN[node.nodeid] != 0 and p.receivedAtN[node.nodeid] is True:
            return get_tx_delay_msec_weighted(node, p.rssiAtN[node.nodeid])  # weighted waiting based on RSSI
    return get_tx_delay_msec(node)


def get_tx_delay_msec_weighted(node, rssi):  # from RadioInterface::getTxDelayMsecWeighted
    # Use the same reported-SNR estimate as the packet-loss model so calibrated
    # presets do not drive relay delay from an impossible near-field SNR tail.
    snr = estimate_snr(node.conf, rssi)
    SNR_MIN = -20
    SNR_MAX = 15
    if snr < SNR_MIN:
        logger.debug(f'Minimum SNR at RSSI of {rssi} dBm')
        snr = SNR_MIN
    if snr > SNR_MAX:
        logger.debug(f'Maximum SNR at RSSI of {rssi} dBm')
        snr = SNR_MAX

    CWsize = int((snr - SNR_MIN) * (CWmax - CWmin) / (SNR_MAX - SNR_MIN) + CWmin)
    if node.is_router:
        CW = random.randint(0, 2 * CWsize - 1)
    else:
        CW = random.randint(0, 2 ** CWsize - 1)
    logger.debug(f'Node {node.nodeid} has CW size {CWsize} and picked CW {CW}')
    return CW * get_current_slot_time()


def get_tx_delay_msec(node):  # from RadioInterface::getTxDelayMsec
    channelUtil = node.airUtilization / node.env.now * 100
    CWsize = int(channelUtil * (CWmax - CWmin) / 100 + CWmin)
    CW = random.randint(0, 2 ** CWsize - 1)
    logger.debug(f'Current channel utilization is {channelUtil}, so picked CW {CW}')
    return CW * get_current_slot_time()


def get_retransmission_msec(node, packet):  # from RadioInterface::getRetransmissionMsec
    # Retransmission timeout has to follow the physical airtime of the packet
    # that was actually sent. With DCR disabled this is still the preset CR.
    packetAirtime = int(airtime(node.conf, packet.sf, packet.cr, packet.packetLen, packet.bw))
    channelUtil = node.airUtilization / node.env.now * 100
    CWsize = int(channelUtil * (CWmax - CWmin) / 100 + CWmin)
    return 2 * packetAirtime + (2 ** CWsize + 2 ** (int((CWmax + CWmin) / 2))) * get_current_slot_time() + PROCESSING_TIME_MSEC
