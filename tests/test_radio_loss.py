import unittest

from lib.config import Config
from lib.radio_loss import apply_link_calibration, estimate_snr, payload_success_probability


class TestRadioLoss(unittest.TestCase):
    def test_stronger_coding_rate_improves_weak_link_probability(self):
        conf = Config()
        conf.PHY_LOSS_MODEL_ENABLED = True

        # Around a marginal SNR band, stronger coding rates should buy payload
        # reliability. Airtime cost is accounted for elsewhere.
        rssi = conf.NOISE_LEVEL - 18.0

        cr5 = payload_success_probability(conf, rssi, 5, conf.PACKETLENGTH)
        cr8 = payload_success_probability(conf, rssi, 8, conf.PACKETLENGTH)

        self.assertGreater(cr8, cr5)

    def test_longer_packets_are_penalized_gently(self):
        conf = Config()
        conf.PHY_LOSS_MODEL_ENABLED = True
        rssi = conf.NOISE_LEVEL - 10.0

        short_packet = payload_success_probability(conf, rssi, 6, 40)
        long_packet = payload_success_probability(conf, rssi, 6, 180)

        self.assertGreater(short_packet, long_packet)

    def test_healthy_snr_band_is_high_probability(self):
        conf = Config()
        conf.PHY_LOSS_MODEL_ENABLED = True

        # At a healthy SNR, all CRs should be very likely to decode.
        rssi = conf.NOISE_LEVEL - 7.0

        self.assertGreater(payload_success_probability(conf, rssi, 5, 40), 0.85)

    def test_reported_snr_can_be_clamped_for_real_mesh_presets(self):
        conf = Config()
        conf.REPORTED_SNR_MIN_DB = -21.25
        conf.REPORTED_SNR_MAX_DB = 8.25

        self.assertEqual(estimate_snr(conf, conf.NOISE_LEVEL + 100.0), 8.25)
        self.assertEqual(estimate_snr(conf, conf.NOISE_LEVEL - 100.0), -21.25)

    def test_link_calibration_model_uses_features_not_pair_ids(self):
        conf = Config()
        conf.LINK_CALIBRATION_MODEL_ENABLED = True
        conf.LINK_CALIBRATION_COEFFICIENTS = {
            "intercept": -4.0,
            "raw_snr_clip": 0.5,
            "urban_fraction": -2.0,
        }
        raw_rssi = conf.NOISE_LEVEL - 20.0

        adjusted = apply_link_calibration(conf, raw_rssi, {
            "raw_snr_clip": -20.0,
            "urban_fraction": 0.5,
        })

        self.assertAlmostEqual(estimate_snr(conf, adjusted), -15.0)

    def test_link_calibration_model_can_be_clamped(self):
        conf = Config()
        conf.LINK_CALIBRATION_MODEL_ENABLED = True
        conf.LINK_CALIBRATION_COEFFICIENTS = {"intercept": 40.0}
        conf.LINK_CALIBRATION_SNR_MAX_DB = 8.25

        adjusted = apply_link_calibration(conf, conf.NOISE_LEVEL - 100.0, {})

        self.assertEqual(estimate_snr(conf, adjusted), 8.25)


if __name__ == "__main__":
    unittest.main()
