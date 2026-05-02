#!/usr/bin/env python3
import argparse
import logging
import math
import os
import random
from pathlib import Path

import yaml

from lib.config import CONFIG
from lib.map_input import DEFAULT_MAP_NODES_URL, fetch_map_payload, node_configs_from_map_payload, parse_bbox
from lib.node import NodeConfig, default_generate_node_list, node_configs_from_yaml, origin_from_yaml
from lib.srtm import (
    DEFAULT_SRTM_URL_TEMPLATE,
    clamp_bbox_to_srtm_coverage,
    terrain_grid_from_srtm,
)
from lib.terrain import (
    NODE_Z_REFERENCE_GROUND,
    NODE_Z_REFERENCE_SEA_LEVEL,
    apply_terrain_altitudes,
    xy_to_latlon,
)

conf = CONFIG
logger = logging.getLogger(__name__)
MIN_TIME_OVERRIDE_SECONDS = 0.01
CLI_DEFAULT_ATTR = "_lora_mesh_cli_defaults"


def configure_logging():
    """Apply CLI logging defaults without changing logging during module import."""
    logging.basicConfig(level=logging.INFO) # default log level


def get_cli_defaults(conf):
    """Remember the caller's initial CLI defaults across reusable parse calls."""
    if not hasattr(conf, CLI_DEFAULT_ATTR):
        terrain_defaults = type(conf)()
        setattr(
            conf,
            CLI_DEFAULT_ATTR,
            {
                "SIMTIME": conf.SIMTIME,
                "PERIOD": conf.PERIOD,
                "GUI_ENABLED": conf.GUI_ENABLED,
                "PLOT": conf.PLOT,
                "TERRAIN_PROFILE_SAMPLES": terrain_defaults.TERRAIN_PROFILE_SAMPLES,
                "NODE_Z_REFERENCE": NODE_Z_REFERENCE_GROUND,
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


def bbox_from_node_config(node_config, origin, margin_m=1000.0):
    """Build a geographic bbox around local x/y nodes when an origin exists."""
    if origin is None:
        return None
    origin_lat, origin_lon = origin
    min_x = min(node.position.x for node in node_config) - margin_m
    max_x = max(node.position.x for node in node_config) + margin_m
    min_y = min(node.position.y for node in node_config) - margin_m
    max_y = max(node.position.y for node in node_config) + margin_m
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


def parse_params(conf, args=None) -> [NodeConfig]:
    """parses command-line arguments, alters global simulation config, and returns
    a list of node configurations, or a list of None.
    """

    # previous cli behavior:
    # loraMesh.py [nr_nodes [router_type]] | [--from-file [file_name]]
    # we'll replicate the intent with argparse, but more strictly, so flags like '--never--from-file' will no longer be accepted
    parser = argparse.ArgumentParser(
        description='run a single interactive or discrete Meshtastic network simulation'
        )

    # only allow one of --from-file optional, or nr_nodes positional exclusively
    group = parser.add_mutually_exclusive_group()
    group.add_argument('nr_nodes', nargs='?', type=int, help='Number of nodes to generate. If unspecified, do interactive simulation')
    group.add_argument('--from-file', nargs='?', const='nodeConfig.yaml', type=str, metavar='filename', help='Name of yaml file storing node config under "out/" directory. If unspecified, defaults to "nodeConfig.yaml".')
    group.add_argument('--from-map', nargs='?', const=DEFAULT_MAP_NODES_URL, type=str, metavar='url', help='Fetch node locations from a Meshtastic map /api/v1/nodes endpoint.')

    # the earlier behavior of specifying `router_type` as an optional positional arg with `nr_nodes` is difficult to exactly
    # replicate with argparse, especially since nesting groups was an unintended feature and deprecated.
    # Just implement as an optional argument, and manually treat it as incompatible with `--from-file`
    parser.add_argument('--router-type', type=conf.ROUTER_TYPE, choices=conf.ROUTER_TYPE, help='Router type to use, taken from ROUTER_TYPE enum. Omit the leading "ROUTER_TYPE". Incompatible with --from-file')
    parser.add_argument('--terrain-srtm', action='store_true', help='Build terrain directly from cached/downloaded SRTM tiles for the scenario bbox')
    parser.add_argument('--terrain-srtm-step-meters', type=float, default=1000.0, help='SRTM terrain sample spacing in meters')
    parser.add_argument(
        '--terrain-srtm-cache-dir',
        default=str(Path.home() / ".cache" / "meshtasticator" / "srtm"),
        help='where downloaded SRTM .hgt tiles are cached',
    )
    parser.add_argument('--terrain-srtm-url-template', default=DEFAULT_SRTM_URL_TEMPLATE, help='SRTM download URL template with {lat_band} and {tile}')
    parser.add_argument('--terrain-srtm-offline', action='store_true', help='use cached SRTM tiles only')
    parser.add_argument('--terrain-profile-samples', type=int, help='number of terrain samples along each TX/RX path')
    parser.add_argument('--map-bbox', type=str, help='Map import bounding box as min_lat,min_lon,max_lat,max_lon')
    parser.add_argument('--map-limit', type=int, help='Maximum number of positioned map nodes to import after bbox filtering')
    parser.add_argument('--map-antenna-height', type=float, default=1.5, help='Antenna height in meters for map-imported nodes')
    parser.add_argument('--map-hop-limit', type=int, default=3, help='Hop limit for map-imported nodes')
    parser.add_argument('--simtime-seconds', type=float, help='Override simulation duration in seconds')
    parser.add_argument('--period-seconds', type=float, help='Override mean message-generation period in seconds')
    parser.add_argument('--no-gui', action='store_true', help='Run without Tk/Matplotlib graphing or schedule plotting')
    parser.add_argument('--disable-connectivity-map', action='store_true', help='disable the connectivity map optimization. May be faster for some scenarios with many moving nodes and/or a densely connected network.')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose/debug output')

    parsed_arguments = parser.parse_args(args)

    cli_defaults = get_cli_defaults(conf)
    simtime = cli_defaults["SIMTIME"]
    period = cli_defaults["PERIOD"]
    gui_enabled = cli_defaults["GUI_ENABLED"]
    plot_enabled = cli_defaults["PLOT"]

    if parsed_arguments.simtime_seconds is not None:
        if not math.isfinite(parsed_arguments.simtime_seconds) or parsed_arguments.simtime_seconds < MIN_TIME_OVERRIDE_SECONDS:
            parser.error(f"--simtime-seconds must be at least {MIN_TIME_OVERRIDE_SECONDS} seconds")
        simtime = int(parsed_arguments.simtime_seconds * conf.ONE_SECOND_INTERVAL)

    if parsed_arguments.period_seconds is not None:
        if not math.isfinite(parsed_arguments.period_seconds) or parsed_arguments.period_seconds < MIN_TIME_OVERRIDE_SECONDS:
            parser.error(f"--period-seconds must be at least {MIN_TIME_OVERRIDE_SECONDS} seconds")
        period = int(parsed_arguments.period_seconds * conf.ONE_SECOND_INTERVAL)

    if parsed_arguments.map_limit is not None and parsed_arguments.map_limit < 1:
        parser.error("--map-limit must be at least 1")
    if not math.isfinite(parsed_arguments.map_antenna_height) or parsed_arguments.map_antenna_height <= 0:
        parser.error("--map-antenna-height must be a positive finite number")
    if parsed_arguments.map_hop_limit < 0:
        parser.error("--map-hop-limit must be at least 0")
    if parsed_arguments.terrain_profile_samples is not None and parsed_arguments.terrain_profile_samples < 2:
        parser.error("--terrain-profile-samples must be at least 2")
    if not math.isfinite(parsed_arguments.terrain_srtm_step_meters) or parsed_arguments.terrain_srtm_step_meters <= 0:
        parser.error("--terrain-srtm-step-meters must be a positive finite number")

    if parsed_arguments.no_gui:
        # Headless CI and smoke runs should not pay Tk startup, per-node
        # plt.pause(), or the final interactive schedule plot. Keep this as an
        # explicit flag so historical visual CLI behavior remains unchanged.
        gui_enabled = False
        plot_enabled = False

    # enforce defaulting to True
    if parsed_arguments.disable_connectivity_map:
        conf.ENABLE_CONNECTIVITY_MAP = False
    else:
        conf.ENABLE_CONNECTIVITY_MAP = True

    if (
        parsed_arguments.from_file is not None
        or parsed_arguments.from_map is not None
    ) and parsed_arguments.router_type is not None:
        parser.error("Incompatible argument selection. --from-file/--from-map and --router-type can not be used together")

    seeded_for_scenario = False
    terrain_bbox = None
    scenario_origin = None
    terrain_grid = None
    terrain_enabled = parsed_arguments.terrain_srtm
    terrain_profile_samples = cli_defaults["TERRAIN_PROFILE_SAMPLES"]
    node_z_reference = cli_defaults["NODE_Z_REFERENCE"]
    if parsed_arguments.terrain_profile_samples is not None:
        terrain_profile_samples = parsed_arguments.terrain_profile_samples
    if parsed_arguments.from_file is not None:
        try:
            with open(os.path.join("out", parsed_arguments.from_file), 'r', encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)
            config = node_configs_from_yaml(raw_config, period, conf.PTX, conf.FREQ)
            scenario_origin = origin_from_yaml(raw_config)
        except (OSError, ValueError, yaml.YAMLError) as err:
            parser.error(f"could not load --from-file YAML: {err}")
        nr_nodes = len(config)
    elif parsed_arguments.from_map is not None:
        if parsed_arguments.map_bbox is None:
            parser.error("--from-map requires --map-bbox min_lat,min_lon,max_lat,max_lon")
        try:
            terrain_bbox = parse_bbox(parsed_arguments.map_bbox)
            raw_map_payload = fetch_map_payload(parsed_arguments.from_map)
            config, map_origin = node_configs_from_map_payload(
                raw_map_payload,
                period,
                bbox=terrain_bbox,
                limit=parsed_arguments.map_limit,
                antenna_height=parsed_arguments.map_antenna_height,
                hop_limit=parsed_arguments.map_hop_limit,
                tx_power=conf.PTX,
                freq=conf.FREQ,
                return_origin=True,
            )
            scenario_origin = map_origin
        except ValueError as err:
            parser.error(str(err))
        nr_nodes = len(config)
    elif parsed_arguments.nr_nodes is not None:
        if parsed_arguments.terrain_srtm:
            parser.error("--terrain-srtm requires --from-map --map-bbox or a scenario file with origin metadata")
        if parsed_arguments.nr_nodes < 2:
            parser.error(f"Need at least two nodes. You specified {parsed_arguments.nr_nodes}")
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
            parser.error("--terrain-srtm requires --from-map --map-bbox or a scenario file with origin metadata")
        if not gui_enabled:
            parser.error("--no-gui requires nr_nodes or --from-file")
        from lib.gui import gen_scenario

        config_dict = gen_scenario(conf)
        config = [NodeConfig.from_gen_scenario_output(node_id, cfg, period, conf.PTX, conf.FREQ) for node_id, cfg in config_dict.items()]
        nr_nodes = len(config)

    if nr_nodes < 2:
        parser.error(f"Need at least two nodes. You specified {nr_nodes}")
    if parsed_arguments.terrain_srtm and terrain_bbox is None:
        terrain_bbox = bbox_from_node_config(config, scenario_origin)
        if terrain_bbox is None:
            parser.error("--terrain-srtm requires --from-map --map-bbox or a scenario file with origin metadata")

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
            )
            apply_terrain_altitudes(terrain_grid, config)
            node_z_reference = NODE_Z_REFERENCE_SEA_LEVEL
        except (OSError, ValueError) as err:
            parser.error(f"could not load SRTM terrain: {err}")

    if not seeded_for_scenario:
        # Loaded and interactive scenarios do not need random state for node
        # placement, but the later MAC/PHY simulation does. Seed only after all
        # parser rejections so failed inputs leave caller RNG state alone.
        random.seed(conf.SEED)

    conf.SIMTIME = simtime
    conf.PERIOD = period
    conf.GUI_ENABLED = gui_enabled
    conf.PLOT = plot_enabled
    conf.NR_NODES = nr_nodes
    set_geo_origin(conf, scenario_origin)
    conf.TERRAIN_ENABLED = terrain_enabled
    conf.TERRAIN_GRID = terrain_grid
    conf.TERRAIN_PROFILE_SAMPLES = terrain_profile_samples
    conf.NODE_Z_REFERENCE = node_z_reference

    if parsed_arguments.verbose:
        # Set this logger and lib.* to DEBUG only after the command line has
        # resolved into a usable scenario. Failed parser inputs should not leave
        # imported callers with noisier logging.
        logger.setLevel(logging.DEBUG)
        lib_logger = logging.getLogger('lib')
        lib_logger.setLevel(logging.DEBUG)
        print("verbose output enabled")

    print("Number of nodes:", conf.NR_NODES)
    print("Modem:", conf.MODEM_PRESET)
    print("Simulation time (s):", conf.SIMTIME/1000)
    print("Period (s):", conf.PERIOD/1000)
    print("Interference level:", conf.INTERFERENCE_LEVEL)
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
    sent = results['sent']
    potentialReceivers = results['potentialReceivers']
    nrCollisions = results['nrCollisions']
    nrSensed = results['nrSensed']
    nrReceived = results['nrReceived']
    meanDelay = results['meanDelay']
    txAirUtilizationRate = results['txAirUtilizationRate']
    collisionRate = results['collisionRate']
    nodeReach = results['nodeReach']
    usefulness = results['usefulness']
    delayDropped = results['delayDropped']

    print("*******************************")
    print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
    print('Number of messages created:', messageSeq)
    print('Number of packets sent:', sent, 'to', potentialReceivers, 'potential receivers')
    print("Number of collisions:", nrCollisions)
    print("Number of packets sensed:", nrSensed)
    print("Number of packets received:", nrReceived)
    print('Delay average (ms):', round(meanDelay, 2))
    print('Average Tx air utilization:', round(txAirUtilizationRate * 100, 2), '%')
    print("Percentage of packets that collided:", round(collisionRate*100, 2))
    print("Average percentage of nodes reached:", round(nodeReach*100, 2))
    print("Percentage of received packets containing new message:", round(usefulness*100, 2))
    print("Number of packets dropped by delay/hop limit:", delayDropped)

    if conf.MODEL_ASYMMETRIC_LINKS:
        noLinkRate = results['noLinkRate']
        print("No links:", round(noLinkRate * 100, 2), '%')

    if conf.MOVEMENT_ENABLED:
        movingNodes = results['movingNodes']
        gpsEnabled = results['gpsEnabled']
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
