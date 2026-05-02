import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools import radio_policy_compare


class TestRadioPolicyCompare(unittest.TestCase):
    def test_parse_policy_names_rejects_unknown_policy(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            radio_policy_compare.parse_policy_names("static,nope")

    def test_parse_policy_names_accepts_current_policy_names(self):
        self.assertEqual(radio_policy_compare.parse_policy_names("static,dcr,dtp"), ["static", "dcr", "dtp"])

    def test_parse_args_rejects_thresholds_without_candidate_policy(self):
        with self.assertRaises(SystemExit):
            radio_policy_compare.parse_args(["--max-reach-drop-pp", "1"])

    def test_build_lora_args_adds_shared_physics_flags(self):
        args = radio_policy_compare.parse_args([
            "--preset",
            "batumi",
            "--simtime-seconds",
            "12",
            "--period-seconds",
            "3",
            "--policies",
            "static",
            "--",
            "--no-clutter",
        ])

        lora_args = radio_policy_compare.build_lora_args(args, "static")

        self.assertEqual(lora_args[:2], ["--preset", "batumi"])
        self.assertIn("--no-gui", lora_args)
        self.assertIn("--phy-loss-model", lora_args)
        self.assertIn("--capture-collision-model", lora_args)
        self.assertIn("--no-clutter", lora_args)
        self.assertNotIn("--dcr", lora_args)
        self.assertNotIn("--dtp", lora_args)
        self.assertNotIn("--", lora_args)

    def test_build_lora_args_adds_dcr_policy_flag(self):
        args = radio_policy_compare.parse_args([
            "--policies",
            "dcr",
        ])

        lora_args = radio_policy_compare.build_lora_args(args, "dcr")

        self.assertIn("--dcr", lora_args)

    def test_build_lora_args_adds_dtp_policy_flag(self):
        args = radio_policy_compare.parse_args([
            "--policies",
            "dtp",
        ])

        lora_args = radio_policy_compare.build_lora_args(args, "dtp")

        self.assertIn("--dtp", lora_args)
        self.assertNotIn("--dcr", lora_args)

    def test_summarize_results_formats_table_and_deltas(self):
        static = radio_policy_compare.summarize_results(
            "static",
            "static CR",
            {
                "messageSeq": 10,
                "sent": 100,
                "nrReceived": 40,
                "nrCollisions": 5,
                "nrPhyLoss": 7,
                "nodeReach": 0.25,
                "usefulness": 0.5,
                "txAirUtilizationRate": 0.07,
                "dcrTxByCr": {5: 100, 6: 0, 7: 0, 8: 0},
                "dtpTxByPower": {30: 100},
                "dtpMeanDetectedByTx": 6.0,
                "dtpMeanSensedByTx": 4.0,
            },
            "raw",
        )
        dcr = radio_policy_compare.summarize_results(
            "dcr",
            "Dynamic Coding Rate",
            {
                "messageSeq": 10,
                "sent": 102,
                "nrReceived": 45,
                "nrCollisions": 4,
                "nrPhyLoss": 5,
                "nodeReach": 0.3,
                "usefulness": math.nan,
                "txAirUtilizationRate": 0.08,
                "dcrTxByCr": {5: 80, 6: 15, 7: 5, 8: 0},
                "dtpTxByPower": {30: 102},
                "dtpMeanDetectedByTx": 6.1,
                "dtpMeanSensedByTx": 4.2,
            },
            "raw",
        )

        table = radio_policy_compare.render_table([static, dcr])
        deltas = radio_policy_compare.render_delta_table([static, dcr])

        self.assertIn("policy", table)
        self.assertIn("25.00", table)
        self.assertIn("80/15/5/0", table)
        self.assertIn("n/a", table)
        self.assertIn("reach +5.00 pp", deltas)
        self.assertIn("phy_loss -2", deltas)

    def test_summarize_results_tolerates_missing_future_policy_counters(self):
        summary = radio_policy_compare.summarize_results(
            "static",
            "static CR",
            {
                "messageSeq": 10,
                "sent": 100,
                "nrReceived": 40,
                "nrCollisions": 5,
                "nrPhyLoss": 7,
                "nodeReach": 0.25,
                "usefulness": 0.5,
                "txAirUtilizationRate": 0.07,
            },
            "raw",
        )

        self.assertEqual(summary.cr_mix, "n/a")
        self.assertEqual(summary.dtp_power_mix, "n/a")
        self.assertEqual(summary.dtp_detected, 0.0)
        self.assertEqual(summary.dtp_decodable, 0.0)

    def test_thresholds_flag_reach_and_airtime_regressions(self):
        args = radio_policy_compare.parse_args([])
        args.max_reach_drop_pp = 1
        args.max_tx_air_increase_pp = 0.5
        baseline = radio_policy_compare.PolicySummary(
            name="static",
            description="static",
            messages=1,
            sent=10,
            received=5,
            collisions=1,
            phy_loss=2,
            reach_percent=10.0,
            useful_percent=50.0,
            tx_air_percent=5.0,
            cr_mix="100/0/0/0",
            dtp_power_mix="30:10",
            dtp_detected=4.0,
            dtp_decodable=3.0,
            output="",
        )
        candidate = radio_policy_compare.PolicySummary(
            name="dcr",
            description="dcr",
            messages=1,
            sent=10,
            received=5,
            collisions=1,
            phy_loss=2,
            reach_percent=8.5,
            useful_percent=50.0,
            tx_air_percent=5.75,
            cr_mix="90/10/0/0",
            dtp_power_mix="30:10",
            dtp_detected=4.0,
            dtp_decodable=3.0,
            output="",
        )

        failures = radio_policy_compare.evaluate_thresholds(args, [baseline, candidate])

        self.assertEqual([failure.metric for failure in failures], ["reach", "tx_air"])
        self.assertIn("below allowed -1.00 pp", failures[0].message)
        self.assertIn("above allowed +0.50 pp", failures[1].message)

    def test_report_writers_create_ci_artifacts(self):
        args = radio_policy_compare.parse_args([
            "--simtime-seconds",
            "12",
            "--period-seconds",
            "3",
            "--policies",
            "static",
            "--",
            "--no-clutter",
        ])
        args.max_reach_drop_pp = 2
        baseline = radio_policy_compare.summarize_results(
            "static",
            "static CR",
            {
                "messageSeq": 10,
                "sent": 100,
                "nrReceived": 40,
                "nrCollisions": 5,
                "nrPhyLoss": 7,
                "nodeReach": 0.25,
                "usefulness": 0.5,
                "txAirUtilizationRate": 0.07,
                "dcrTxByCr": {5: 100, 6: 0, 7: 0, 8: 0},
                "dtpTxByPower": {30: 100},
                "dtpMeanDetectedByTx": 6.0,
                "dtpMeanSensedByTx": 4.0,
            },
            "raw",
        )
        dcr = radio_policy_compare.summarize_results(
            "dcr",
            "Dynamic Coding Rate",
            {
                "messageSeq": 10,
                "sent": 102,
                "nrReceived": 45,
                "nrCollisions": 4,
                "nrPhyLoss": 5,
                "nodeReach": 0.3,
                "usefulness": 0.55,
                "txAirUtilizationRate": 0.08,
                "dcrTxByCr": {5: 80, 6: 15, 7: 5, 8: 0},
                "dtpTxByPower": {30: 102},
                "dtpMeanDetectedByTx": 6.1,
                "dtpMeanSensedByTx": 4.2,
            },
            "raw",
        )
        report = radio_policy_compare.build_report(args, [baseline, dcr], [])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "report.json"
            markdown_path = Path(tmpdir) / "report.md"

            radio_policy_compare.write_json_report(json_path, report)
            radio_policy_compare.write_markdown_report(markdown_path, report)

            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(parsed["scenario"]["extra_lora_args"], ["--no-clutter"])
        self.assertEqual(parsed["deltas"][0]["reach_delta_pp"], 5.0)
        self.assertEqual(parsed["thresholds"]["max_reach_drop_pp"], 2.0)
        self.assertNotIn("raw", json.dumps(parsed))
        self.assertIn("# Meshtasticator Radio Policy Comparison", markdown)
        self.assertIn("| dcr | 30.00 | 55.00 | 8.00 |", markdown)


if __name__ == "__main__":
    unittest.main()
