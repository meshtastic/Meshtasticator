import contextlib
import io
import logging
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from array import array
from pathlib import Path
from unittest import mock

from lib.config import Config

import loraMesh


def write_hgt(path, values):
    data = array("h", values)
    if sys.byteorder == "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def generated_positions(node_configs):
    return [
        (round(node.position.x, 6), round(node.position.y, 6), round(node.position.z, 6))
        for node in node_configs
    ]


class TestLoraMeshCli(unittest.TestCase):
    """Regression tests for the top-level CLI wrapper.

    loraMesh.py used to run a simulation while being imported and mutate global
    process state while still rejecting arguments. These tests lock in the more
    tool-friendly behavior: import is quiet, parser failures are side-effect
    free, and accepted headless runs can be used by CI.
    """

    def parse_quietly(self, conf, args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            nodes = loraMesh.parse_params(conf, args)
        return nodes, stdout.getvalue()

    def assert_parser_rejects(self, conf, args):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                loraMesh.parse_params(conf, args)
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_importing_lora_mesh_does_not_run_simulation(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import loraMesh; print('import ok')"],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "import ok")
        self.assertEqual(completed.stderr, "")

    def test_parse_params_uses_supplied_argv(self):
        conf = Config()

        nodes, output = self.parse_quietly(
            conf,
            ["2", "--no-gui", "--simtime-seconds", "1", "--period-seconds", "0.5"],
        )

        self.assertEqual(len(nodes), 2)
        self.assertFalse(conf.GUI_ENABLED)
        self.assertFalse(conf.PLOT)
        self.assertFalse(conf.DCR_ENABLED)
        self.assertEqual(conf.SIMTIME, 1000)
        self.assertEqual(conf.PERIOD, 500)
        self.assertIn("Number of nodes: 2", output)
        self.assertIn("Dynamic Coding Rate: disabled", output)

    def test_parse_params_enables_dcr(self):
        conf = Config()

        _, output = self.parse_quietly(
            conf,
            ["2", "--no-gui", "--simtime-seconds", "1", "--period-seconds", "0.5", "--dcr"],
        )

        self.assertTrue(conf.DCR_ENABLED)
        self.assertIn("Dynamic Coding Rate: enabled", output)

    def test_parse_params_reuses_initial_defaults_after_override_run(self):
        conf = Config()
        default_simtime = conf.SIMTIME
        default_period = conf.PERIOD

        self.parse_quietly(
            conf,
            ["2", "--no-gui", "--simtime-seconds", "1", "--period-seconds", "0.5"],
        )
        nodes, _ = self.parse_quietly(conf, ["2"])

        self.assertTrue(conf.GUI_ENABLED)
        self.assertTrue(conf.PLOT)
        self.assertEqual(conf.SIMTIME, default_simtime)
        self.assertEqual(conf.PERIOD, default_period)
        self.assertEqual([node.period for node in nodes], [default_period, default_period])

    def test_parse_params_preserves_caller_initial_defaults(self):
        conf = Config()
        conf.SIMTIME = 1234
        conf.PERIOD = 2345
        conf.GUI_ENABLED = False
        conf.PLOT = False

        self.parse_quietly(conf, ["2", "--simtime-seconds", "1", "--period-seconds", "0.5"])
        nodes, _ = self.parse_quietly(conf, ["2"])

        self.assertFalse(conf.GUI_ENABLED)
        self.assertFalse(conf.PLOT)
        self.assertEqual(conf.SIMTIME, 1234)
        self.assertEqual(conf.PERIOD, 2345)
        self.assertEqual([node.period for node in nodes], [2345, 2345])

    def test_parse_params_rejects_sub_centisecond_time_overrides(self):
        conf = Config()

        simtime_error = self.assert_parser_rejects(conf, ["2", "--no-gui", "--simtime-seconds", "0.009"])
        period_error = self.assert_parser_rejects(conf, ["2", "--no-gui", "--period-seconds", "0.009"])

        self.assertIn("--simtime-seconds must be at least 0.01 seconds", simtime_error)
        self.assertIn("--period-seconds must be at least 0.01 seconds", period_error)

    def test_no_gui_run_does_not_import_gui_module(self):
        script = textwrap.dedent(
            """\
            import builtins

            real_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "lib.gui":
                    raise AssertionError("headless run imported lib.gui")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import

            import loraMesh

            loraMesh.main(["2", "--no-gui", "--simtime-seconds", "0.01", "--period-seconds", "0.01"])
            print("headless ok")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertIn("headless ok", completed.stdout)

    def test_parse_params_loads_from_file_as_node_configs(self):
        conf = Config()
        scenario = textwrap.dedent(
            """\
            0:
              x: 0
              y: 0
              z: 1
              isRouter: false
              isRepeater: false
              isClientMute: false
              antennaGain: 0
              hopLimit: 3
              neighborInfo: false
            1:
              x: 10
              y: 0
              z: 1
              isRouter: false
              isRepeater: false
              isClientMute: false
              antennaGain: 0
              hopLimit: 3
              neighborInfo: false
            """
        )

        os.makedirs("out", exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir="out", suffix=".yaml", delete=False, encoding="utf-8") as scenario_file:
            scenario_file.write(scenario)
            scenario_filename = os.path.basename(scenario_file.name)

        try:
            nodes, _ = self.parse_quietly(
                conf,
                ["--from-file", scenario_filename, "--no-gui", "--period-seconds", "2"],
            )
        finally:
            os.unlink(os.path.join("out", scenario_filename))

        self.assertEqual([node.node_id for node in nodes], [0, 1])
        self.assertEqual([node.period for node in nodes], [2000, 2000])
        self.assertEqual(conf.NR_NODES, 2)

    def test_parse_params_loads_from_map_payload(self):
        conf = Config()
        payload = [
            {
                "latitude": 416200000,
                "longitude": 415900000,
                "role": 2,
            },
            {
                "latitude": 416300000,
                "longitude": 416000000,
                "role": 0,
            },
        ]

        with mock.patch("loraMesh.fetch_map_payload", return_value=payload):
            nodes, _ = self.parse_quietly(
                conf,
                [
                    "--from-map",
                    "https://example.test/nodes",
                    "--map-bbox",
                    "41.0,41.0,42.0,42.0",
                    "--map-antenna-height",
                    "2.5",
                    "--no-gui",
                ],
            )

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].position.z, 2.5)
        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.625, 41.595))

    def test_parse_params_can_build_srtm_terrain_for_map_payload(self):
        conf = Config()
        payload = [
            {
                "latitude": 416200000,
                "longitude": 415900000,
                "role": 2,
            },
            {
                "latitude": 416300000,
                "longitude": 416000000,
                "role": 0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = tempfile.TemporaryDirectory()
            self.addCleanup(source_dir.cleanup)
            source_path = Path(source_dir.name) / "N41E041.hgt"
            write_hgt(
                source_path,
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
            )

            with mock.patch("loraMesh.fetch_map_payload", return_value=payload):
                nodes, _ = self.parse_quietly(
                    conf,
                    [
                        "--from-map",
                        "https://example.test/nodes",
                        "--map-bbox",
                        "41.5,41.5,41.8,41.8",
                        "--terrain-srtm",
                        "--terrain-srtm-step-meters",
                        "20000",
                        "--terrain-srtm-cache-dir",
                        tmpdir,
                        "--terrain-srtm-url-template",
                        f"{Path(source_dir.name).as_uri()}/{{tile}}.hgt",
                        "--no-gui",
                    ],
                )

        self.assertEqual(len(nodes), 2)
        self.assertTrue(conf.TERRAIN_ENABLED)
        self.assertIsNotNone(conf.TERRAIN_GRID)
        self.assertGreater(len(conf.TERRAIN_GRID.samples), 0)
        self.assertTrue(conf.TERRAIN_NODE_Z_IS_ABSOLUTE_ALTITUDE)
        self.assertNotEqual(nodes[0].position.z, nodes[1].position.z)
        self.assertGreater(nodes[0].position.z, 1.5)
        self.assertGreater(nodes[1].position.z, 1.5)
        self.assertEqual([node.antenna_height for node in nodes], [1.5, 1.5])

    def test_parse_params_clears_geo_origin_for_scenarios_without_origin(self):
        conf = Config()
        conf.GEO_ORIGIN_LAT = 41.625
        conf.GEO_ORIGIN_LON = 41.595
        scenario = textwrap.dedent(
            """\
            nodes:
              3944424993:
                x: 0
                y: 0
                z: 1
                isRouter: false
                isRepeater: false
                isClientMute: false
                antennaGain: 0
                hopLimit: 3
                neighborInfo: false
              3944424994:
                x: 10
                y: 0
                z: 1
                isRouter: false
                isRepeater: false
                isClientMute: false
                antennaGain: 0
                hopLimit: 3
                neighborInfo: false
            """
        )

        os.makedirs("out", exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir="out", suffix=".yaml", delete=False, encoding="utf-8") as scenario_file:
            scenario_file.write(scenario)
            scenario_filename = os.path.basename(scenario_file.name)

        try:
            nodes, _ = self.parse_quietly(conf, ["--from-file", scenario_filename, "--no-gui"])
        finally:
            os.unlink(os.path.join("out", scenario_filename))

        self.assertEqual([node.node_id for node in nodes], [0, 1])
        self.assertIsNone(conf.GEO_ORIGIN_LAT)
        self.assertIsNone(conf.GEO_ORIGIN_LON)

    def test_parse_params_lists_presets_without_scenario_side_effects(self):
        conf = Config()
        random.seed(9123)
        random_state = random.getstate()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                loraMesh.parse_params(conf, ["--list-presets"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIsNone(conf.NR_NODES)
        self.assertEqual(random.getstate(), random_state)
        self.assertIn("Available scenario presets:", stdout.getvalue())
        self.assertIn("batumi: 92 nodes", stdout.getvalue())
        self.assertIn("terrain=yes", stdout.getvalue())
        self.assertIn("clutter=yes", stdout.getvalue())
        self.assertIn("link_calibration=yes", stdout.getvalue())

    def test_parse_params_lists_modem_presets_without_scenario_side_effects(self):
        conf = Config()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                loraMesh.parse_params(conf, ["--list-modem-presets"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIsNone(conf.NR_NODES)
        self.assertIn("Available modem presets:", stdout.getvalue())
        self.assertIn("LONG_FAST (default):", stdout.getvalue())
        self.assertIn("cr=4/5", stdout.getvalue())

    def test_parse_params_help_includes_discovery_and_policy_examples(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                loraMesh.parse_params(Config(), ["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("loraMesh.py --list-presets", stdout.getvalue())
        self.assertIn("--preset batumi --no-gui", stdout.getvalue())
        self.assertIn("--phy-loss-model --capture-collision-model", stdout.getvalue())

    def test_parse_params_loads_batumi_preset_with_bundled_grids(self):
        conf = Config()

        nodes, output = self.parse_quietly(
            conf,
            ["--preset", "batumi", "--no-gui", "--period-seconds", "2"],
        )

        self.assertEqual(len(nodes), 92)
        self.assertEqual(conf.NR_NODES, 92)
        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.6442879, 41.61536))
        self.assertTrue(conf.TERRAIN_ENABLED)
        self.assertTrue(conf.CLUTTER_ENABLED)
        self.assertTrue(conf.LINK_CALIBRATION_MODEL_ENABLED)
        self.assertIn("Terrain model:", output)
        self.assertIn("Clutter model:", output)
        self.assertIn("Link calibration model: enabled", output)

    def test_parse_params_can_disable_bundled_preset_clutter(self):
        conf = Config()

        self.parse_quietly(
            conf,
            ["--preset", "batumi", "--no-gui", "--no-clutter"],
        )

        self.assertTrue(conf.TERRAIN_ENABLED)
        self.assertFalse(conf.CLUTTER_ENABLED)

    def test_parse_params_rejects_before_applying_time_overrides(self):
        conf = Config()
        original_simtime = conf.SIMTIME

        self.assert_parser_rejects(conf, ["1", "--simtime-seconds", "1"])

        self.assertEqual(conf.SIMTIME, original_simtime)

    def test_parse_params_rejects_before_applying_no_gui(self):
        conf = Config()

        self.assert_parser_rejects(conf, ["1", "--no-gui"])

        self.assertTrue(conf.GUI_ENABLED)
        self.assertTrue(conf.PLOT)

    def test_parse_params_rejects_before_enabling_verbose_logging(self):
        conf = Config()
        lora_logger = logging.getLogger("loraMesh")
        lib_logger = logging.getLogger("lib")
        original_lora_level = lora_logger.level
        original_lib_level = lib_logger.level

        try:
            self.assert_parser_rejects(conf, ["1", "--verbose", "--no-gui"])

            self.assertEqual(lora_logger.level, original_lora_level)
            self.assertEqual(lib_logger.level, original_lib_level)
        finally:
            lora_logger.setLevel(original_lora_level)
            lib_logger.setLevel(original_lib_level)

    def test_parse_params_rejects_one_node_before_seeding(self):
        conf = Config()
        random.seed(12345)
        state_before = random.getstate()

        self.assert_parser_rejects(conf, ["1", "--no-gui"])

        self.assertEqual(random.getstate(), state_before)

    def test_parse_params_seeds_generated_scenarios(self):
        conf_a = Config()
        conf_b = Config()

        nodes_a, _ = self.parse_quietly(conf_a, ["3", "--no-gui"])
        random.seed(999)
        random.random()
        nodes_b, _ = self.parse_quietly(conf_b, ["3", "--no-gui"])

        self.assertEqual(generated_positions(nodes_a), generated_positions(nodes_b))


if __name__ == "__main__":
    unittest.main()
