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
from lib.srtm import SRTM_DATA_ATTRIBUTION_URL
from lib.terrain import NODE_Z_REFERENCE_GROUND, NODE_Z_REFERENCE_SEA_LEVEL

import loraMesh


def write_hgt(path, values):
    data = array("h", values)
    if sys.byteorder == "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def generated_positions(node_configs):
    return [
        (
            round(node.position.x, 6),
            round(node.position.y, 6),
            round(node.position.z, 6),
        )
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
        self.assertEqual(conf.SIMTIME, 1000)
        self.assertEqual(conf.PERIOD, 500)
        self.assertIn("Number of nodes: 2", output)

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
        self.assertEqual(
            [node.period for node in nodes], [default_period, default_period]
        )

    def test_parse_params_preserves_caller_initial_defaults(self):
        conf = Config()
        conf.SIMTIME = 1234
        conf.PERIOD = 2345
        conf.GUI_ENABLED = False
        conf.PLOT = False

        self.parse_quietly(
            conf, ["2", "--simtime-seconds", "1", "--period-seconds", "0.5"]
        )
        nodes, _ = self.parse_quietly(conf, ["2"])

        self.assertFalse(conf.GUI_ENABLED)
        self.assertFalse(conf.PLOT)
        self.assertEqual(conf.SIMTIME, 1234)
        self.assertEqual(conf.PERIOD, 2345)
        self.assertEqual([node.period for node in nodes], [2345, 2345])

    def test_parse_params_rejects_sub_centisecond_time_overrides(self):
        conf = Config()

        simtime_error = self.assert_parser_rejects(
            conf, ["2", "--no-gui", "--simtime-seconds", "0.009"]
        )
        period_error = self.assert_parser_rejects(
            conf, ["2", "--no-gui", "--period-seconds", "0.009"]
        )

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
        with tempfile.NamedTemporaryFile(
            "w", dir="out", suffix=".yaml", delete=False, encoding="utf-8"
        ) as scenario_file:
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
        conf.HM = 2.5
        conf.hopLimit = 5
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
                    "--no-gui",
                ],
            )

        self.assertEqual(len(nodes), 2)
        self.assertEqual([node.position.z for node in nodes], [2.5, 2.5])
        self.assertEqual([node.antenna_height for node in nodes], [2.5, 2.5])
        self.assertEqual([node.hop_limit for node in nodes], [5, 5])
        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.625, 41.595))

    def test_parse_params_loads_from_nodedb_payload(self):
        conf = Config()
        conf.HM = 2.5
        conf.hopLimit = 5
        payload = {
            "nodesByNum": {
                1: {
                    "num": 1,
                    "user": {"id": "!00000001", "role": "ROUTER"},
                    "position": {
                        "latitude": 41.62,
                        "longitude": 41.59,
                        "altitude": 120,
                    },
                },
                2: {
                    "num": 2,
                    "user": {"id": "!00000002", "role": "CLIENT"},
                    "position": {"latitudeI": 416300000, "longitudeI": 416000000},
                },
            }
        }

        with mock.patch(
            "loraMesh.fetch_nodedb_payload", return_value=payload
        ) as fetch_nodedb:
            nodes, _ = self.parse_quietly(
                conf,
                [
                    "--from-nodedb",
                    "--nodedb-host",
                    "192.0.2.10",
                    "--map-bbox",
                    "41.0,41.0,42.0,42.0",
                    "--no-gui",
                ],
            )

        fetch_nodedb.assert_called_once_with(
            host="192.0.2.10", port=None, serial_port=None
        )
        self.assertEqual(len(nodes), 2)
        self.assertEqual([node.position.z for node in nodes], [2.5, 2.5])
        self.assertEqual([node.antenna_height for node in nodes], [2.5, 2.5])
        self.assertEqual([node.hop_limit for node in nodes], [5, 5])
        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.625, 41.595))

    def test_parse_params_rejects_nodedb_transport_without_nodedb_source(self):
        conf = Config()

        error = self.assert_parser_rejects(conf, ["2", "--nodedb-host", "192.0.2.10"])

        self.assertIn("--nodedb-* options require --from-nodedb", error)

    def test_parse_params_rejects_nodedb_port_without_host(self):
        conf = Config()

        error = self.assert_parser_rejects(
            conf, ["--from-nodedb", "--nodedb-port", "4404"]
        )

        self.assertIn("--nodedb-port requires --nodedb-host", error)

    def test_parse_params_can_build_srtm_terrain_for_map_payload(self):
        conf = Config()
        payload = [
            {
                "latitude": 416200000,
                "longitude": 415900000,
                "altitude": 500,
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
                nodes, output = self.parse_quietly(
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
        self.assertEqual(conf.NODE_Z_REFERENCE, NODE_Z_REFERENCE_SEA_LEVEL)
        self.assertIn("Terrain data attribution:", output)
        self.assertIn(SRTM_DATA_ATTRIBUTION_URL, output)
        self.assertEqual(nodes[0].position.z, 500)
        self.assertNotEqual(nodes[0].position.z, nodes[1].position.z)
        self.assertGreater(nodes[1].position.z, conf.HM)
        self.assertEqual([node.antenna_height for node in nodes], [conf.HM, conf.HM])

    def test_parse_params_ignores_map_altitude_when_applying_srtm(self):
        conf = Config()
        conf.HM = 2.5
        payload = [
            {
                "latitude": -16400000,
                "longitude": -26400000,
                "altitude": None,
                "role": 0,
            },
            {
                "latitude": -16350000,
                "longitude": -26350000,
                "altitude": -1,
                "role": 0,
            },
            {
                "latitude": -16300000,
                "longitude": -26300000,
                "altitude": 42949649,
                "role": 0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = tempfile.TemporaryDirectory()
            self.addCleanup(source_dir.cleanup)
            source_path = Path(source_dir.name) / "S02W003.hgt"
            write_hgt(
                source_path,
                [100, 110, 120, 130, 140, 150, 160, 170, 180],
            )

            with mock.patch("loraMesh.fetch_map_payload", return_value=payload):
                nodes, _ = self.parse_quietly(
                    conf,
                    [
                        "--from-map",
                        "https://example.test/nodes",
                        "--map-bbox=-1.7,-2.7,-1.2,-2.2",
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

        self.assertEqual(len(nodes), 3)
        self.assertEqual(conf.NODE_Z_REFERENCE, NODE_Z_REFERENCE_SEA_LEVEL)
        self.assertEqual([node.antenna_height for node in nodes], [2.5, 2.5, 2.5])
        self.assertTrue(all(100 < node.position.z < 190 for node in nodes))
        self.assertNotIn(-1, [node.position.z for node in nodes])
        self.assertNotIn(42949649, [node.position.z for node in nodes])

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
        with tempfile.NamedTemporaryFile(
            "w", dir="out", suffix=".yaml", delete=False, encoding="utf-8"
        ) as scenario_file:
            scenario_file.write(scenario)
            scenario_filename = os.path.basename(scenario_file.name)

        try:
            nodes, _ = self.parse_quietly(
                conf, ["--from-file", scenario_filename, "--no-gui"]
            )
        finally:
            os.unlink(os.path.join("out", scenario_filename))

        self.assertEqual([node.node_id for node in nodes], [0, 1])
        self.assertIsNone(conf.GEO_ORIGIN_LAT)
        self.assertIsNone(conf.GEO_ORIGIN_LON)

    def test_parse_params_rejects_one_node_before_changing_geo_origin(self):
        conf = Config()
        conf.GEO_ORIGIN_LAT = 41.625
        conf.GEO_ORIGIN_LON = 41.595
        scenario = textwrap.dedent(
            """\
            origin:
              latitude: 42.0
              longitude: 42.0
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
            """
        )

        os.makedirs("out", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir="out", suffix=".yaml", delete=False, encoding="utf-8"
        ) as scenario_file:
            scenario_file.write(scenario)
            scenario_filename = os.path.basename(scenario_file.name)

        try:
            self.assert_parser_rejects(
                conf, ["--from-file", scenario_filename, "--no-gui"]
            )
        finally:
            os.unlink(os.path.join("out", scenario_filename))

        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.625, 41.595))

    def test_terrain_srtm_generated_scenario_rejects_before_config_mutation(self):
        conf = Config()
        original_simtime = conf.SIMTIME
        original_period = conf.PERIOD
        random.seed(12345)
        state_before = random.getstate()

        error = self.assert_parser_rejects(
            conf,
            [
                "2",
                "--terrain-srtm",
                "--simtime-seconds",
                "1",
                "--period-seconds",
                "2",
                "--no-gui",
            ],
        )

        self.assertIn("--terrain-srtm requires", error)
        self.assertEqual(conf.SIMTIME, original_simtime)
        self.assertEqual(conf.PERIOD, original_period)
        self.assertTrue(conf.GUI_ENABLED)
        self.assertTrue(conf.PLOT)
        self.assertIsNone(conf.NR_NODES)
        self.assertFalse(conf.TERRAIN_ENABLED)
        self.assertEqual(random.getstate(), state_before)

    def test_terrain_srtm_from_file_rejects_uncovered_bbox_before_config_mutation(self):
        conf = Config()
        conf.TERRAIN_ENABLED = True
        terrain_grid = object()
        conf.TERRAIN_GRID = terrain_grid
        conf.GEO_ORIGIN_LAT = 41.625
        conf.GEO_ORIGIN_LON = 41.595
        random.seed(12345)
        state_before = random.getstate()
        scenario = textwrap.dedent(
            """\
            origin:
              lat: 85.0
              lon: 42.0
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
        with tempfile.NamedTemporaryFile(
            "w", dir="out", suffix=".yaml", delete=False, encoding="utf-8"
        ) as scenario_file:
            scenario_file.write(scenario)
            scenario_filename = os.path.basename(scenario_file.name)

        try:
            error = self.assert_parser_rejects(
                conf, ["--from-file", scenario_filename, "--terrain-srtm", "--no-gui"]
            )
        finally:
            os.unlink(os.path.join("out", scenario_filename))

        self.assertIn("could not derive SRTM terrain bbox", error)
        self.assertTrue(conf.TERRAIN_ENABLED)
        self.assertIs(conf.TERRAIN_GRID, terrain_grid)
        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.625, 41.595))
        self.assertEqual(random.getstate(), state_before)

    def test_failed_srtm_load_keeps_previous_terrain_config(self):
        conf = Config()
        terrain_grid = object()
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_GRID = terrain_grid
        conf.TERRAIN_PROFILE_SAMPLES = 7
        conf.NODE_Z_REFERENCE = NODE_Z_REFERENCE_SEA_LEVEL
        conf.GEO_ORIGIN_LAT = 41.625
        conf.GEO_ORIGIN_LON = 41.595
        random.seed(12345)
        state_before = random.getstate()
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
            with mock.patch("loraMesh.fetch_map_payload", return_value=payload):
                error = self.assert_parser_rejects(
                    conf,
                    [
                        "--from-map",
                        "https://example.test/nodes",
                        "--map-bbox",
                        "41.5,41.5,41.8,41.8",
                        "--terrain-srtm",
                        "--terrain-srtm-offline",
                        "--terrain-srtm-cache-dir",
                        tmpdir,
                        "--terrain-profile-samples",
                        "12",
                        "--no-gui",
                    ],
                )

        self.assertIn("could not load SRTM terrain", error)
        self.assertTrue(conf.TERRAIN_ENABLED)
        self.assertIs(conf.TERRAIN_GRID, terrain_grid)
        self.assertEqual(conf.TERRAIN_PROFILE_SAMPLES, 7)
        self.assertEqual(conf.NODE_Z_REFERENCE, NODE_Z_REFERENCE_SEA_LEVEL)
        self.assertEqual((conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON), (41.625, 41.595))
        self.assertEqual(random.getstate(), state_before)

    def test_terrain_profile_samples_resets_between_parse_calls(self):
        conf = Config()
        conf.TERRAIN_PROFILE_SAMPLES = 31
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
            write_hgt(source_path, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            terrain_args = [
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
            ]

            with mock.patch("loraMesh.fetch_map_payload", return_value=payload):
                self.parse_quietly(
                    conf, [*terrain_args, "--terrain-profile-samples", "7"]
                )
                self.assertEqual(conf.TERRAIN_PROFILE_SAMPLES, 7)

                self.parse_quietly(conf, terrain_args)

        self.assertEqual(conf.TERRAIN_PROFILE_SAMPLES, 31)
        self.assertEqual(conf.NODE_Z_REFERENCE, NODE_Z_REFERENCE_SEA_LEVEL)

    def test_successful_plain_parse_clears_previous_terrain_state(self):
        conf = Config()
        loraMesh.get_cli_defaults(conf)
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_GRID = object()
        conf.TERRAIN_PROFILE_SAMPLES = 7
        conf.NODE_Z_REFERENCE = NODE_Z_REFERENCE_SEA_LEVEL

        self.parse_quietly(conf, ["2", "--no-gui"])

        self.assertFalse(conf.TERRAIN_ENABLED)
        self.assertIsNone(conf.TERRAIN_GRID)
        self.assertEqual(conf.TERRAIN_PROFILE_SAMPLES, Config().TERRAIN_PROFILE_SAMPLES)
        self.assertEqual(conf.NODE_Z_REFERENCE, NODE_Z_REFERENCE_GROUND)

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
