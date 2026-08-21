import logging
import random

from lib.phy import airtime, get_current_slot_time
from lib.radio_loss import estimate_snr

logger = logging.getLogger(__name__)

# checked as of tag v2.7.15.567b8ea in meshtastic-firmware repo
CWmin = 3
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
    SNR_MAX = 10
    slot_time_msec = get_current_slot_time()
    if snr < SNR_MIN:
        logger.debug(f'{node.env.now:.3f} Node {node.nodeid} clamping to Minimum SNR at RSSI of {rssi} dBm')
        snr = SNR_MIN
    if snr > SNR_MAX:
        logger.debug(f'{node.env.now:.3f} Node {node.nodeid} clamping to Maximum SNR at RSSI of {rssi} dBm')
        snr = SNR_MAX

    CWsize = int((snr - SNR_MIN) * (CWmax - CWmin) / (SNR_MAX - SNR_MIN) + CWmin)

    if node.is_router:
        delay = random.randint(0, 2 * CWsize) * slot_time_msec
        logger.debug(f'{node.env.now:.3f} Node {node.nodeid} is router, has CW size {CWsize} and picked {delay=}')
    else:
        max_router_delay = 2 * CWmax * slot_time_msec
        delay = max_router_delay + random.randint(0, 2 ** CWsize) * slot_time_msec
        logger.debug(f'{node.env.now:.3f} Node {node.nodeid} is not router, has CW size {CWsize} and picked {delay=}')
    return delay


def get_tx_delay_msec(node):  # from RadioInterface::getTxDelayMsec
    # channelUtilizationPercent is actually computed based on the last CHANNEL_UTILIZATION_PERIODS, summing
    # the utilization of those periods. In v2.7.15.567b8ea this macro is 6, with SECONDS_PER_PERIOD 3600
    channelUtil = node.airUtilization / node.env.now * 100
    CWsize = int(channelUtil * (CWmax - CWmin) / 100 + CWmin)
    CW = random.randint(0, 2 ** CWsize)
    logger.debug(f'{node.env.now:.3f} Current channel utilization is {channelUtil}, so picked {CWsize=} and {CW=}')
    return CW * get_current_slot_time()


def get_retransmission_msec(node, packet):  # from RadioInterface::getRetransmissionMsec
    # Retransmission timeout has to follow the physical airtime of the packet
    # that was actually sent. With DCR disabled this is still the preset CR.
    packetAirtime = int(airtime(node.conf, packet.sf, packet.cr, packet.packetLen, packet.bw))
    channelUtil = node.airUtilization / node.env.now * 100
    CWsize = int(channelUtil * (CWmax - CWmin) / 100 + CWmin)
    return 2 * packetAirtime + (2 ** CWsize + 2 * CWmax + 2 ** (int((CWmax + CWmin) / 2))) * get_current_slot_time() + PROCESSING_TIME_MSEC

# NOTE: there is a getTxDelayMsecWeightedWorst function that we haven't implemented yet.
