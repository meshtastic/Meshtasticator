"""Empirical packet-loss model for the discrete-event simulator.

Meshtasticator's original PHY is binary: if RSSI is above sensitivity and no
collision happens, the packet is decoded. That is good for topology sketches,
but too optimistic for weak links: stronger coding rates should improve payload
decode probability near the edge, while still costing airtime.

The bundled coefficients are intentionally small and documented. They are tuned
from Batumi-area receive observations and neighbor SNR bands, so this remains an
empirical SNR-to-PER curve rather than a full lab-grade demodulator model.

Keep the model opt-in. Baseline simulations must stay exactly comparable to
upstream Meshtasticator until a scenario explicitly enables it.
"""

import math


def estimate_snr(conf, rssi):
    """Estimate packet SNR from simulated RSSI and the configured noise floor."""
    snr = rssi - conf.NOISE_LEVEL
    if conf.REPORTED_SNR_MIN_DB is not None:
        snr = max(conf.REPORTED_SNR_MIN_DB, snr)
    if conf.REPORTED_SNR_MAX_DB is not None:
        snr = min(conf.REPORTED_SNR_MAX_DB, snr)
    return snr


def apply_link_calibration(conf, rssi, features):
    """Map raw path-loss output plus reusable features to calibrated RSSI.

    This deliberately does not accept node IDs. Observed directed links may be
    used to fit the coefficients stored in a preset, but runtime simulation
    applies the same coefficient transform to every generated pair. That keeps
    the model reusable for new points instead of replaying known links.
    """
    if not conf.LINK_CALIBRATION_MODEL_ENABLED or not conf.LINK_CALIBRATION_COEFFICIENTS:
        return rssi

    coefficients = conf.LINK_CALIBRATION_COEFFICIENTS
    calibrated_snr = coefficients.get("intercept", 0.0)
    for name, value in features.items():
        calibrated_snr += coefficients.get(name, 0.0) * value

    if conf.LINK_CALIBRATION_SNR_MIN_DB is not None:
        calibrated_snr = max(conf.LINK_CALIBRATION_SNR_MIN_DB, calibrated_snr)
    if conf.LINK_CALIBRATION_SNR_MAX_DB is not None:
        calibrated_snr = min(conf.LINK_CALIBRATION_SNR_MAX_DB, calibrated_snr)

    return conf.NOISE_LEVEL + calibrated_snr


def payload_success_probability(conf, rssi, cr, packet_len):
    """Return probability that a heard packet's payload decodes.

    RSSI/sensitivity still gates preamble/header hearing elsewhere. This
    function only models payload decode once the receiver was able to hear the
    packet at all. CR therefore improves weak-link payload survival, but it does
    not extend the model below the basic receive threshold.
    """
    snr = estimate_snr(conf, rssi)
    p50_by_cr = conf.PHY_LOSS_SNR_P50_BY_CR
    p50 = p50_by_cr.get(cr, p50_by_cr[5])

    # Longer packets expose more coded symbols to fading/interference. The
    # penalty is deliberately gentle because collisions are modeled separately.
    extra_bytes = max(0, packet_len - conf.PHY_LOSS_REFERENCE_PACKET_BYTES)
    length_penalty = extra_bytes / 100.0 * conf.PHY_LOSS_LONG_PACKET_PENALTY_DB_PER_100B

    x = (snr - p50 - length_penalty) / conf.PHY_LOSS_SNR_TRANSITION_DB
    probability = 1.0 / (1.0 + math.exp(-x))
    return min(conf.PHY_LOSS_MAX_SUCCESS_PROB, max(conf.PHY_LOSS_MIN_SUCCESS_PROB, probability))


def payload_is_lost(conf, rssi, cr, packet_len, random_draw):
    """Decide whether this packet copy is lost to weak-link PHY errors."""
    if not conf.PHY_LOSS_MODEL_ENABLED:
        return False

    return random_draw > payload_success_probability(conf, rssi, cr, packet_len)
