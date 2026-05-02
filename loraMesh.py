#!/usr/bin/env python3
import argparse
import logging
import math
import os
import random
from pathlib import Path

import yaml

from lib.config import CONFIG
from lib.map_input import (
    DEFAULT_MAP_NODES_URL,
    fetch_map_payload,
    node_configs_from_map_payload,
    parse_bbox,
)
from lib.nodedb_input import fetch_nodedb_payload, node_configs_from_nodedb_payload
from lib.node import (
    NodeConfig,
    default_generate_node_list,
    node_configs_from_yaml,
    origin_from_yaml,
)
from lib.presets import (
    apply_preset_radio_calibration,
    available_presets,
    load_preset_raw,
    load_preset_terrain_grid,
    load_preset_node_configs,
    preset_clutter_grid,
    preset_origin,
    preset_terrain_grid,
    restore_radio_calibration,
    snapshot_radio_calibration,
)
from lib.srtm import (
    DEFAULT_SRTM_URL_TEMPLATE,
    SRTM_DATA_ATTRIBUTION,
    SRTM_DATA_ATTRIBUTION_URL,
    clamp_bbox_to_srtm_coverage,
    terrain_grid_from_srtm,
    tiles_for_bbox,
)
from lib.terrain import (
    NODE_Z_REFERENCE_SEA_LEVEL,
    apply_terrain_altitudes,
    node_antenna_height,
    xy_to_latlon,
)
from lib.phy import estimate_path_loss

conf = CONFIG
logger = logging.getLogger(__name__)
MIN_TIME_OVERRIDE_SECONDS = 0.01
CLI_DEFAULT_ATTR = "_lora_mesh_cli_defaults"


def configure_logging():
    """Apply CLI logging defaults without changing logging during module import."""
    logging.basicConfig(level=logging.INFO)  # default log level


def get_cli_defaults(conf):
    """Remember the caller's initial CLI defaults across reusable parse calls."""
    if not hasattr(conf, CLI_DEFAULT_ATTR):
        setattr(
            conf,
            CLI_DEFAULT_ATTR,
            {
                "SIMTIME": conf.SIMTIME,
                "PERIOD": conf.PERIOD,
                "GUI_ENABLED": conf.GUI_ENABLED,
                "PLOT": conf.PLOT,
                "TERRAIN_PROFILE_SAMPLES": conf.TERRAIN_PROFILE_SAMPLES,
                "CLUTTER_PROFILE_SAMPLES": conf.CLUTTER_PROFILE_SAMPLES,
                "NODE_Z_REFERENCE": conf.NODE_Z_REFERENCE,
                "RADIO_CALIBRATION": snapshot_radio_calibration(conf),
            },
        )
    return getattr(conf, CLI_DEFAULT_ATTR)


def set_geo_origin(conf, origin):
    """Use scenario geographic origin for lat/lon terrain grids when available."""
    if origin is None:
        conf.GEO_ORIGIN_LAT = None
        conf.GEO_ORIGIN_LON = None
        return
    conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON = origin


def bbox_from_points(points, origin, margin_m=1000.0):
    """Build a geographic bbox around local x/y points when an origin exists."""
    if origin is None:
        return None
    origin_lat, origin_lon = origin
    min_x = min(point.x for point in points) - margin_m
    max_x = max(point.x for point in points) + margin_m
    min_y = min(point.y for point in points) - margin_m
    max_y = max(point.y for point in points) + margin_m
    lat_a, lon_a = xy_to_latlon(min_x, min_y, origin_lat, origin_lon)
    lat_b, lon_b = xy_to_latlon(max_x, max_y, origin_lat, origin_lon)
    return clamp_bbox_to_srtm_coverage(
        (
            min(lat_a, lat_b),
            min(lon_a, lon_b),
            max(lat_a, lat_b),
            max(lon_a, lon_b),
        )
    )


def bbox_from_node_config(node_config, origin, margin_m=1000.0):
    """Build a geographic bbox around local x/y nodes when an origin exists."""
    return bbox_from_points([node.position for node in node_config], origin, margin_m)


def fit_simulation_bounds_to_node_config(conf, node_config, margin_m=1000.0):
    """Expand movement/GUI bounds so loaded coordinates are not clamped."""
    min_x = min(node.position.x for node in node_config) - margin_m
    max_x = max(node.position.x for node in node_config) + margin_m
    min_y = min(node.position.y for node in node_config) - margin_m
    max_y = max(node.position.y for node in node_config) + margin_m

    left = conf.OX - conf.XSIZE / 2
    right = conf.OX + conf.XSIZE / 2
    bottom = conf.OY - conf.YSIZE / 2
    top = conf.OY + conf.YSIZE / 2
    if left <= min_x and max_x <= right and bottom <= min_y and max_y <= top:
        return

    conf.OX = (min_x + max_x) / 2
    conf.OY = (min_y + max_y) / 2
    conf.XSIZE = max_x - min_x
    conf.YSIZE = max_y - min_y


def nodes_have_flat_link_budget(conf, node_a, node_b):
    """Return whether two nodes can hear each other before terrain loss."""
    distance = node_a.position.euclidean_distance(node_b.position)
    path_loss = estimate_path_loss(
        conf,
        distance,
        conf.FREQ,
        node_antenna_height(node_a),
        node_antenna_height(node_b),
    )
    sensitivity = conf.current_preset["sensitivity"]
    antenna_gain_a = getattr(node_a, "antennaGain", getattr(node_a, "antenna_gain", 0))
    antenna_gain_b = getattr(node_b, "antennaGain", getattr(node_b, "antenna_gain", 0))
    tx_power_a = getattr(node_a, "tx_power", conf.PTX)
    tx_power_b = getattr(node_b, "tx_power", conf.PTX)
    rssi_ab = tx_power_a + antenna_gain_a + antenna_gain_b - path_loss
    rssi_ba = tx_power_b + antenna_gain_b + antenna_gain_a - path_loss
    return rssi_ab >= sensitivity or rssi_ba >= sensitivity


def srtm_tiles_for_node_config_links(conf, node_config, origin, margin_m=1000.0):
    """Return SRTM tiles around nodes and flat-link candidate paths."""
    if origin is None:
        return None

    tile_names = set()
    for node in node_config:
        bbox = bbox_from_points([node.position], origin, margin_m)
        tile_names.update(tiles_for_bbox(bbox))

    for index, node_a in enumerate(node_config):
        for node_b in node_config[index + 1 :]:
            if not nodes_have_flat_link_budget(conf, node_a, node_b):
                continue
            bbox = bbox_from_points(
                [node_a.position, node_b.position], origin, margin_m
            )
            tile_names.update(tiles_for_bbox(bbox))

    return sorted(tile_names)


def print_preset_list():
    """Print packaged scenario presets in a copy-pasteable discovery format."""
    print("Available scenario presets:")
    for name in available_presets():
        raw = load_preset_raw(name)
        nodes = raw.get("nodes", {}) if isinstance(raw, dict) else {}
        origin = raw.get("origin", {}) if isinstance(raw, dict) else {}
        calibration = raw.get("radio_calibration", {}) if isinstance(raw, dict) else {}
        observations = raw.get("calibration_observations", []) if isinstance(raw, dict) else []
        terrain = preset_terrain_grid(name) is not None
        clutter = preset_clutter_grid(name) is not None
        calibration_enabled = bool(calibration.get("link_calibration_model"))
        origin_text = "unknown"
        if "lat" in origin and "lon" in origin:
            origin_text = f"{origin['lat']:.5f},{origin['lon']:.5f}"

        print(
            f"  {name}: {len(nodes)} nodes, origin={origin_text}, "
            f"terrain={'yes' if terrain else 'no'}, "
            f"clutter={'yes' if clutter else 'no'}, "
            f"link_calibration={'yes' if calibration_enabled else 'no'}, "
            f"calibration_edges={len(observations)}"
        )


def print_modem_preset_list(conf):
    """Print modem presets with the fields users need for comparable runs."""
    print("Available modem presets:")
    for name, preset in conf.MODEM_PRESETS.items():
        default_marker = " (default)" if name == conf.MODEM_PRESET else ""
        print(
            f"  {name}{default_marker}: "
            f"bw={preset['bw'] / 1000:g} kHz, "
            f"sf={preset['sf']}, "
            f"cr=4/{preset['cr']}, "
            f"sensitivity={preset['sensitivity']:g} dBm"
        )


def parse_params(conf, args=None) -> [NodeConfig]:
    """parses command-line arguments, alters global simulation config, and returns
    a list of node configurations, or a list of None.
    """

    # previous cli behavior:
    # loraMesh.py [nr_nodes [router_type]] | [--from-file [file_name]]
    # we'll replicate the intent with argparse, but more strictly, so flags like '--never--from-file' will no longer be accepted
    parser = argparse.ArgumentParser(
        description="run a single interactive or discrete Meshtastic network simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  loraMesh.py --list-presets
  loraMesh.py --preset batumi --no-gui --simtime-seconds 60 --period-seconds 5
  loraMesh.py --preset batumi --no-gui --simtime-seconds 60 --period-seconds 5 --phy-loss-model --capture-collision-model
  loraMesh.py --from-map 'https://meshtastic.liamcottle.net/api/v1/nodes' --map-bbox 41.50,41.50,41.82,41.86 --map-limit 100 --no-gui
""",
    )

    # only allow one of --from-file optional, or nr_nodes positional exclusively
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "nr_nodes",
        nargs="?",
        type=int,
        help="Number of nodes to generate. If unspecified, do interactive simulation",
    )
    group.add_argument(
        "--from-file",
        nargs="?",
        const="nodeConfig.yaml",
        type=str,
        metavar="filename",
        help='Name of yaml file storing node config under "out/" directory. If unspecified, defaults to "nodeConfig.yaml".',
    )
    group.add_argument(
        "--from-map",
        nargs="?",
        const=DEFAULT_MAP_NODES_URL,
        type=str,
        metavar="url",
        help="Fetch node locations from a Meshtastic map /api/v1/nodes endpoint.",
    )
    group.add_argument(
        "--from-nodedb",
        action="store_true",
        help="Fetch positioned nodes from a local Meshtastic device NodeDB.",
    )
    group.add_argument(
        "--preset",
        choices=available_presets(),
        help="Load a packaged real-mesh scenario preset.",
    )

    # the earlier behavior of specifying `router_type` as an optional positional arg with `nr_nodes` is difficult to exactly
    # replicate with argparse, especially since nesting groups was an unintended feature and deprecated.
    # Just implement as an optional argument, and manually treat it as incompatible with `--from-file`
    parser.add_argument(
        "--router-type",
        type=conf.ROUTER_TYPE,
        choices=conf.ROUTER_TYPE,
        help='Router type to use, taken from ROUTER_TYPE enum. Omit the leading "ROUTER_TYPE". Incompatible with --from-file',
    )
    parser.add_argument(
        "--dcr", action="store_true", help="Enable the Dynamic Coding Rate experiment"
    )
    parser.add_argument(
        "--terrain-srtm",
        action="store_true",
        help="Build terrain directly from cached/downloaded SRTM tiles for the scenario bbox",
    )
    parser.add_argument(
        "--terrain-srtm-step-meters",
        type=float,
        default=1000.0,
        help="SRTM terrain sample spacing in meters",
    )
    parser.add_argument(
        "--terrain-srtm-cache-dir",
        default=str(Path.home() / ".cache" / "meshtasticator" / "srtm"),
        help="where downloaded SRTM .hgt tiles are cached",
    )
    parser.add_argument(
        "--terrain-srtm-url-template",
        default=DEFAULT_SRTM_URL_TEMPLATE,
        help="SRTM download URL template with {lat_band} and {tile}",
    )
    parser.add_argument(
        "--terrain-srtm-offline", action="store_true", help="use cached SRTM tiles only"
    )
    parser.add_argument(
        "--terrain-profile-samples",
        type=int,
        help="number of terrain samples along each TX/RX path",
    )
    parser.add_argument(
        "--clutter-grid",
        type=str,
        help="CSV land-cover clutter grid for optional building/urban excess loss",
    )
    parser.add_argument(
        "--clutter-profile-samples",
        type=int,
        help="number of clutter samples along each TX/RX path",
    )
    parser.add_argument(
        "--no-clutter",
        action="store_true",
        help="disable land-cover clutter even when a grid is available",
    )
    parser.add_argument(
        "--phy-loss-model",
        action="store_true",
        help="enable empirical SNR-to-payload-loss model",
    )
    parser.add_argument(
        "--capture-collision-model",
        action="store_true",
        help="enable capture-aware overlap/collision model",
    )
    parser.add_argument(
        "--map-bbox",
        type=str,
        help="Position import bounding box as min_lat,min_lon,max_lat,max_lon",
    )
    parser.add_argument(
        "--map-limit",
        type=int,
        help="Maximum number of positioned imported nodes after bbox filtering",
    )
    parser.add_argument(
        "--nodedb-host",
        type=str,
        help="Hostname or IP of a Meshtastic TCP device for --from-nodedb",
    )
    parser.add_argument(
        "--nodedb-port",
        type=int,
        help="TCP port of a Meshtastic TCP device for --from-nodedb",
    )
    parser.add_argument(
        "--nodedb-serial-port",
        type=str,
        help="Serial device path for --from-nodedb; defaults to Meshtastic auto-detection",
    )
    parser.add_argument(
        "--simtime-seconds", type=float, help="Override simulation duration in seconds"
    )
    parser.add_argument(
        "--period-seconds",
        type=float,
        help="Override mean message-generation period in seconds",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without Tk/Matplotlib graphing or schedule plotting",
    )
    parser.add_argument(
        "--disable-connectivity-map",
        action="store_true",
        help="disable the connectivity map optimization. May be faster for some scenarios with many moving nodes and/or a densely connected network.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List packaged real-mesh scenario presets and exit",
    )
    parser.add_argument(
        "--list-modem-presets",
        action="store_true",
        help="List Meshtastic modem presets and exit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose/debug output"
    )

    parsed_arguments = parser.parse_args(args)

    if parsed_arguments.list_presets:
        print_preset_list()
    if parsed_arguments.list_modem_presets:
        print_modem_preset_list(conf)
    if parsed_arguments.list_presets or parsed_arguments.list_modem_presets:
        raise SystemExit(0)

    cli_defaults = get_cli_defaults(conf)
    simtime = cli_defaults["SIMTIME"]
    period = cli_defaults["PERIOD"]
    gui_enabled = cli_defaults["GUI_ENABLED"]
    plot_enabled = cli_defaults["PLOT"]

    if parsed_arguments.simtime_seconds is not None:
        if (
            not math.isfinite(parsed_arguments.simtime_seconds)
            or parsed_arguments.simtime_seconds < MIN_TIME_OVERRIDE_SECONDS
        ):
            parser.error(
                f"--simtime-seconds must be at least {MIN_TIME_OVERRIDE_SECONDS} seconds"
            )
        simtime = int(parsed_arguments.simtime_seconds * conf.ONE_SECOND_INTERVAL)

    if parsed_arguments.period_seconds is not None:
        if (
            not math.isfinite(parsed_arguments.period_seconds)
            or parsed_arguments.period_seconds < MIN_TIME_OVERRIDE_SECONDS
        ):
            parser.error(
                f"--period-seconds must be at least {MIN_TIME_OVERRIDE_SECONDS} seconds"
            )
        period = int(parsed_arguments.period_seconds * conf.ONE_SECOND_INTERVAL)

    if parsed_arguments.map_limit is not None and parsed_arguments.map_limit < 1:
        parser.error("--map-limit must be at least 1")
    if not math.isfinite(conf.HM) or conf.HM <= 0:
        parser.error("config HM must be a positive finite antenna height")
    if conf.hopLimit < 0:
        parser.error("config hopLimit must be at least 0")
    if (
        parsed_arguments.terrain_profile_samples is not None
        and parsed_arguments.terrain_profile_samples < 2
    ):
        parser.error("--terrain-profile-samples must be at least 2")
    if (
        not math.isfinite(parsed_arguments.terrain_srtm_step_meters)
        or parsed_arguments.terrain_srtm_step_meters <= 0
    ):
        parser.error("--terrain-srtm-step-meters must be a positive finite number")
    if parsed_arguments.clutter_profile_samples is not None and parsed_arguments.clutter_profile_samples < 1:
        parser.error("--clutter-profile-samples must be at least 1")

    if parsed_arguments.no_gui:
        # Headless CI and smoke runs should not pay Tk startup, per-node
        # plt.pause(), or the final interactive schedule plot. Keep this as an
        # explicit flag so historical visual CLI behavior remains unchanged.
        gui_enabled = False
        plot_enabled = False

    connectivity_map_enabled = not parsed_arguments.disable_connectivity_map

    if (
        parsed_arguments.from_file is not None
        or parsed_arguments.from_map is not None
        or parsed_arguments.from_nodedb
        or parsed_arguments.preset is not None
    ) and parsed_arguments.router_type is not None:
        parser.error(
            "Incompatible argument selection. --from-file/--from-map/--from-nodedb/--preset and --router-type can not be used together"
        )
    if not parsed_arguments.from_nodedb and (
        parsed_arguments.nodedb_host is not None
        or parsed_arguments.nodedb_port is not None
        or parsed_arguments.nodedb_serial_port is not None
    ):
        parser.error("--nodedb-* options require --from-nodedb")
    if (
        parsed_arguments.from_nodedb
        and parsed_arguments.nodedb_port is not None
        and parsed_arguments.nodedb_host is None
    ):
        parser.error("--nodedb-port requires --nodedb-host")
    if parsed_arguments.no_clutter and parsed_arguments.clutter_grid:
        parser.error("--no-clutter can not be combined with --clutter-grid")

    seeded_for_scenario = False
    bounds_follow_node_config = False
    terrain_bbox = None
    terrain_tile_names = None
    scenario_origin = None
    terrain_grid = None
    terrain_enabled = parsed_arguments.terrain_srtm
    terrain_profile_samples = cli_defaults["TERRAIN_PROFILE_SAMPLES"]
    node_z_reference = cli_defaults["NODE_Z_REFERENCE"]
    if parsed_arguments.terrain_profile_samples is not None:
        terrain_profile_samples = parsed_arguments.terrain_profile_samples
    bundled_terrain_grid = None
    bundled_clutter_grid = None
    selected_preset = None
    if parsed_arguments.from_file is not None:
        try:
            if parsed_arguments.map_bbox is not None:
                terrain_bbox = parse_bbox(parsed_arguments.map_bbox)
            with open(
                os.path.join("out", parsed_arguments.from_file), "r", encoding="utf-8"
            ) as file:
                raw_config = yaml.safe_load(file)
            config = node_configs_from_yaml(raw_config, period, conf.PTX, conf.FREQ)
            scenario_origin = origin_from_yaml(raw_config)
        except (OSError, ValueError, yaml.YAMLError) as err:
            parser.error(f"could not load --from-file YAML: {err}")
        nr_nodes = len(config)
        bounds_follow_node_config = True
    elif parsed_arguments.preset is not None:
        selected_preset = parsed_arguments.preset
        config = load_preset_node_configs(parsed_arguments.preset, period)
        scenario_origin = preset_origin(parsed_arguments.preset)
        # Packaged scenarios can carry terrain/clutter grids matched to the
        # node geometry. Use them by default, while still letting explicit CLI
        # files override them for A/B comparison runs.
        bundled_terrain_grid = preset_terrain_grid(parsed_arguments.preset)
        bundled_clutter_grid = preset_clutter_grid(parsed_arguments.preset)
        nr_nodes = len(config)
        bounds_follow_node_config = True
    elif parsed_arguments.from_map is not None:
        if parsed_arguments.map_bbox is None:
            parser.error(
                "--from-map requires --map-bbox min_lat,min_lon,max_lat,max_lon"
            )
        try:
            terrain_bbox = parse_bbox(parsed_arguments.map_bbox)
            raw_map_payload = fetch_map_payload(parsed_arguments.from_map)
            config, map_origin = node_configs_from_map_payload(
                raw_map_payload,
                period,
                bbox=terrain_bbox,
                limit=parsed_arguments.map_limit,
                antenna_height=conf.HM,
                hop_limit=conf.hopLimit,
                tx_power=conf.PTX,
                freq=conf.FREQ,
                return_origin=True,
            )
            scenario_origin = map_origin
        except ValueError as err:
            parser.error(str(err))
        nr_nodes = len(config)
        bounds_follow_node_config = True
    elif parsed_arguments.from_nodedb:
        try:
            if parsed_arguments.map_bbox is not None:
                terrain_bbox = parse_bbox(parsed_arguments.map_bbox)
            raw_nodedb_payload = fetch_nodedb_payload(
                host=parsed_arguments.nodedb_host,
                port=parsed_arguments.nodedb_port,
                serial_port=parsed_arguments.nodedb_serial_port,
            )
            config, nodedb_origin = node_configs_from_nodedb_payload(
                raw_nodedb_payload,
                period,
                bbox=terrain_bbox,
                limit=parsed_arguments.map_limit,
                antenna_height=conf.HM,
                hop_limit=conf.hopLimit,
                tx_power=conf.PTX,
                freq=conf.FREQ,
                return_origin=True,
            )
            scenario_origin = nodedb_origin
        except ValueError as err:
            parser.error(str(err))
        nr_nodes = len(config)
        bounds_follow_node_config = True
    elif parsed_arguments.nr_nodes is not None:
        if parsed_arguments.terrain_srtm:
            parser.error(
                "--terrain-srtm requires --from-map --map-bbox or a scenario file with origin metadata"
            )
        if parsed_arguments.nr_nodes < 2:
            parser.error(
                f"Need at least two nodes. You specified {parsed_arguments.nr_nodes}"
            )
        nr_nodes = parsed_arguments.nr_nodes
        if parsed_arguments.router_type is not None:
            routerType = parsed_arguments.router_type
            conf.SELECTED_ROUTER_TYPE = routerType
            conf.update_router_dependencies()
        # Generated node positions come from the global RNG. Seed immediately
        # before that generation, after every parser-only rejection path above.
        conf.NR_NODES = nr_nodes
        conf.PERIOD = period
        random.seed(conf.SEED)
        seeded_for_scenario = True
        config = default_generate_node_list(conf)
    else:
        if parsed_arguments.terrain_srtm:
            parser.error(
                "--terrain-srtm requires --from-map --map-bbox or a scenario file with origin metadata"
            )
        if not gui_enabled:
            parser.error("--no-gui requires nr_nodes, --from-file, --from-map, or --preset")
        from lib.gui import gen_scenario

        config_dict = gen_scenario(conf)
        config = [NodeConfig.from_gen_scenario_output(node_id, cfg, period, conf.PTX, conf.FREQ) for node_id, cfg in config_dict.items()]
        nr_nodes = len(config)

    if nr_nodes < 2:
        parser.error(f"Need at least two nodes. You specified {nr_nodes}")
    if parsed_arguments.terrain_srtm and terrain_bbox is None:
        try:
            terrain_bbox = bbox_from_node_config(config, scenario_origin)
            terrain_tile_names = srtm_tiles_for_node_config_links(
                conf, config, scenario_origin
            )
        except ValueError as err:
            parser.error(f"could not derive SRTM terrain bbox: {err}")
        if terrain_bbox is None:
            parser.error(
                "--terrain-srtm requires --from-map --map-bbox or a scenario file with origin metadata"
            )
    if parsed_arguments.terrain_srtm and scenario_origin is None:
        parser.error(
            "--terrain-srtm requires --from-map/--from-nodedb or a scenario file with origin metadata"
        )

    if parsed_arguments.terrain_srtm:
        try:
            origin_lat, origin_lon = scenario_origin
            terrain_grid = terrain_grid_from_srtm(
                terrain_bbox,
                parsed_arguments.terrain_srtm_step_meters,
                parsed_arguments.terrain_srtm_cache_dir,
                origin_lat,
                origin_lon,
                parsed_arguments.terrain_srtm_url_template,
                download_missing=not parsed_arguments.terrain_srtm_offline,
                tile_names=terrain_tile_names,
            )
            apply_terrain_altitudes(terrain_grid, config)
            node_z_reference = NODE_Z_REFERENCE_SEA_LEVEL
        except (OSError, ValueError) as err:
            parser.error(f"could not load SRTM terrain: {err}")
    elif bundled_terrain_grid is not None:
        try:
            terrain_grid = load_preset_terrain_grid(parsed_arguments.preset)
            apply_terrain_altitudes(terrain_grid, config)
            terrain_enabled = True
            node_z_reference = NODE_Z_REFERENCE_SEA_LEVEL
        except (OSError, ValueError) as err:
            parser.error(f"could not load preset terrain: {err}")

    if not seeded_for_scenario:
        # File, map, preset, and interactive scenarios do not need random state
        # for node placement, but the later MAC/PHY simulation does. Seed only
        # after successful scenario loading so rejected inputs leave caller RNG
        # state alone.
        random.seed(conf.SEED)

    if bounds_follow_node_config:
        fit_simulation_bounds_to_node_config(conf, config)

    conf.SIMTIME = simtime
    conf.PERIOD = period
    conf.GUI_ENABLED = gui_enabled
    conf.PLOT = plot_enabled
    conf.NR_NODES = nr_nodes
    conf.ENABLE_CONNECTIVITY_MAP = connectivity_map_enabled
    conf.DCR_ENABLED = parsed_arguments.dcr
    set_geo_origin(conf, scenario_origin)
    conf.TERRAIN_ENABLED = terrain_enabled
    conf.TERRAIN_GRID = terrain_grid
    conf.TERRAIN_PROFILE_SAMPLES = terrain_profile_samples
    conf.NODE_Z_REFERENCE = node_z_reference
    if parsed_arguments.clutter_grid:
        conf.CLUTTER_ENABLED = True
        conf.CLUTTER_GRID_FILE = parsed_arguments.clutter_grid
    elif bundled_clutter_grid is not None and not parsed_arguments.no_clutter:
        conf.CLUTTER_ENABLED = True
        conf.CLUTTER_GRID_FILE = str(bundled_clutter_grid)
    else:
        conf.CLUTTER_ENABLED = False
        conf.CLUTTER_GRID_FILE = None
    if parsed_arguments.clutter_profile_samples is not None:
        conf.CLUTTER_PROFILE_SAMPLES = parsed_arguments.clutter_profile_samples
    else:
        conf.CLUTTER_PROFILE_SAMPLES = cli_defaults["CLUTTER_PROFILE_SAMPLES"]
    conf.PHY_LOSS_MODEL_ENABLED = parsed_arguments.phy_loss_model
    conf.CAPTURE_COLLISION_MODEL_ENABLED = parsed_arguments.capture_collision_model
    restore_radio_calibration(conf, cli_defaults["RADIO_CALIBRATION"])
    if parsed_arguments.preset is not None:
        apply_preset_radio_calibration(conf, parsed_arguments.preset)

    if parsed_arguments.verbose:
        # Set this logger and lib.* to DEBUG only after the command line has
        # resolved into a usable scenario. Failed parser inputs should not leave
        # imported callers with noisier logging.
        logger.setLevel(logging.DEBUG)
        lib_logger = logging.getLogger("lib")
        lib_logger.setLevel(logging.DEBUG)
        print("verbose output enabled")

    print("Number of nodes:", conf.NR_NODES)
    print("Modem:", conf.MODEM_PRESET)
    print("Simulation time (s):", conf.SIMTIME / 1000)
    print("Period (s):", conf.PERIOD / 1000)
    print("Interference level:", conf.INTERFERENCE_LEVEL)
    if conf.TERRAIN_ENABLED:
        print(
            "Terrain data attribution:",
            f"{SRTM_DATA_ATTRIBUTION} ({SRTM_DATA_ATTRIBUTION_URL})",
        )
    print("Dynamic Coding Rate:", "enabled" if conf.DCR_ENABLED else "disabled")
    print("PHY loss model:", "enabled" if conf.PHY_LOSS_MODEL_ENABLED else "disabled")
    print("Capture collision model:", "enabled" if conf.CAPTURE_COLLISION_MODEL_ENABLED else "disabled")
    print("Terrain model:", "enabled" if conf.TERRAIN_ENABLED else "disabled")
    print("Clutter model:", conf.CLUTTER_GRID_FILE if conf.CLUTTER_ENABLED else "disabled")
    print("Link calibration model:", "enabled" if conf.LINK_CALIBRATION_MODEL_ENABLED else "disabled")
    return config


def run_simulation(conf, node_config):
    """Run one configured simulation and print the historical CLI summary."""
    # Keep the heavier simulation/GUI import out of module import. That makes
    # CLI parsing unit-testable and lets CI/tools import loraMesh without
    # starting Matplotlib/Tk plumbing as a side effect.
    from lib.discrete_event_sim import DiscreteEventSim

    conf.update_router_dependencies()
    if conf.GUI_ENABLED:
        from lib.gui import Graph

        graph = Graph(conf)
    else:
        graph = None

    # set up sim
    sim = DiscreteEventSim(conf, node_config, graph)

    # run sim
    print("\n====== START OF SIMULATION ======")
    sim.run_simulation()

    # collect, process & display results
    print("\n====== END OF SIMULATION ======")

    results = sim.get_results()

    packets = results["packets"]
    messageSeq = results["messageSeq"]
    messages = results["messages"]

    # collect second-order results from finalized results
    sent = results["sent"]
    potentialReceivers = results["potentialReceivers"]
    nrCollisions = results["nrCollisions"]
    nrSensed = results["nrSensed"]
    nrReceived = results["nrReceived"]
    meanDelay = results["meanDelay"]
    txAirUtilizationRate = results["txAirUtilizationRate"]
    collisionRate = results["collisionRate"]
    nodeReach = results["nodeReach"]
    usefulness = results["usefulness"]
    delayDropped = results["delayDropped"]

    print("*******************************")
    print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
    print("Number of messages created:", messageSeq)
    print(
        "Number of packets sent:", sent, "to", potentialReceivers, "potential receivers"
    )
    print("Number of collisions:", nrCollisions)
    print("Number of packets sensed:", nrSensed)
    print("Number of packets received:", nrReceived)
    print("Delay average (ms):", round(meanDelay, 2))
    print("Average Tx air utilization:", round(txAirUtilizationRate * 100, 2), "%")
    print("Percentage of packets that collided:", round(collisionRate * 100, 2))
    print("Average percentage of nodes reached:", round(nodeReach * 100, 2))
    print(
        "Percentage of received packets containing new message:",
        round(usefulness * 100, 2),
    )
    print("Number of packets dropped by delay/hop limit:", delayDropped)

    if conf.DCR_ENABLED:
        print("DCR TX packets by CR:", results["dcrTxByCr"])
        print("DCR airtime by CR (ms):", {cr: round(ms, 2) for cr, ms in results["dcrAirtimeByCr"].items()})

    if conf.TERRAIN_ENABLED:
        print("Mean terrain obstruction loss (dB):", round(results["meanTerrainLossDb"], 2))
        print("Max terrain obstruction loss (dB):", round(results["maxTerrainLossDb"], 2))
    if conf.CLUTTER_ENABLED:
        print("Mean clutter loss (dB):", round(results["meanClutterLossDb"], 2))
        print("Max clutter loss (dB):", round(results["maxClutterLossDb"], 2))

    if conf.MODEL_ASYMMETRIC_LINKS:
        noLinkRate = results["noLinkRate"]
        print("No links:", round(noLinkRate * 100, 2), "%")

    if conf.MOVEMENT_ENABLED:
        movingNodes = results["movingNodes"]
        gpsEnabled = results["gpsEnabled"]
        print("Number of moving nodes:", movingNodes)
        print("Number of moving nodes w/ GPS:", gpsEnabled)

    if graph is not None:
        graph.save()

    if conf.PLOT:
        from lib.gui import plot_schedule

        plot_schedule(conf, packets, messages)

    return results


def main(args=None):
    configure_logging()
    node_config = parse_params(conf, args)
    return run_simulation(conf, node_config)


if __name__ == "__main__":
    main()
