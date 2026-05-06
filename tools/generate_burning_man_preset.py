#!/usr/bin/env python3
"""Generate the packaged Burning Man preset and clutter raster."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.burning_man_preset import (  # noqa: E402
    DEFAULT_CLIENTS,
    DEFAULT_SEED,
    write_burning_man_clutter_csv,
    write_burning_man_preset,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="write deterministic Burning Man preset inputs")
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS, help="number of client nodes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic placement seed")
    parser.add_argument("--clutter-step-meters", type=int, default=250, help="clutter raster spacing")
    parser.add_argument("--nodes-output", type=Path, default=REPO_ROOT / "presets" / "burning_man.yaml")
    parser.add_argument("--clutter-output", type=Path, default=REPO_ROOT / "presets" / "burning_man_clutter.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.clients < 1:
        raise SystemExit("--clients must be at least 1")
    if args.clutter_step_meters < 1:
        raise SystemExit("--clutter-step-meters must be at least 1")

    write_burning_man_preset(args.nodes_output, args.clients, args.seed)
    write_burning_man_clutter_csv(args.clutter_output, args.clutter_step_meters)
    print(f"wrote {args.nodes_output}")
    print(f"wrote {args.clutter_output}")


if __name__ == "__main__":
    main()
