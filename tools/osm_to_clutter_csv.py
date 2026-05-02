#!/usr/bin/env python3
"""CLI wrapper for exporting OSM land-cover clutter into CSV format."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.osm_clutter import main  # noqa: E402


if __name__ == "__main__":
    main()
