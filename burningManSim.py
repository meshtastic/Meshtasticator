#!/usr/bin/env python3
"""Compatibility wrapper for the Burning Man preset.

The Burning Man scenario is now ordinary simulator input:

    loraMesh.py --preset burning_man --phy-loss-model --capture-collision-model

Keep this wrapper only so older notes/commands still land on that generic path
instead of reviving a second simulator with separate radio physics.
"""

import argparse
import sys

from lib.config import Config
import loraMesh


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="run the Burning Man preset through loraMesh.py")
    parser.add_argument("num_clients", nargs="?", type=int, help="kept for old commands; preset size is fixed")
    parser.add_argument("--plot", action="store_true", help="enable the normal simulator plot")
    parser.add_argument("--hop-limit", type=int, default=3, help="maximum hop limit")
    parser.add_argument("--simtime-seconds", type=float, help="simulation duration")
    parser.add_argument("--period-seconds", type=float, help="mean message-generation period override")
    parser.add_argument("--use-node-periods", action="store_true", help="respect preset per-node periodMs values")
    parser.add_argument("--dcr", action="store_true", help="enable Dynamic Coding Rate")
    parser.add_argument("--dtp", action="store_true", help="enable Dynamic TX Power")
    parser.add_argument("--phy-loss-model", action="store_true", help="enable empirical SNR-to-payload-loss model")
    parser.add_argument("--capture-collision-model", action="store_true", help="enable capture-aware collision model")
    parser.add_argument("extra_lora_args", nargs=argparse.REMAINDER, help="extra loraMesh.py arguments after --")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    lora_args = ["--preset", "burning_man", "--no-gui", "--hop-limit", str(args.hop_limit)]
    if args.plot:
        lora_args.append("--plot")
    if args.simtime_seconds is not None:
        lora_args.extend(["--simtime-seconds", str(args.simtime_seconds)])
    if args.period_seconds is not None:
        lora_args.extend(["--period-seconds", str(args.period_seconds)])
    if args.use_node_periods:
        lora_args.append("--use-node-periods")
    if args.dcr:
        lora_args.append("--dcr")
    if args.dtp:
        lora_args.append("--dtp")
    if args.phy_loss_model:
        lora_args.append("--phy-loss-model")
    if args.capture_collision_model:
        lora_args.append("--capture-collision-model")
    if args.extra_lora_args:
        lora_args.extend(arg for arg in args.extra_lora_args if arg != "--")

    conf = Config()
    nodes = loraMesh.parse_params(conf, lora_args)
    if nodes:
        loraMesh.run_simulation(conf, nodes)


if __name__ == "__main__":
    main(sys.argv[1:])
