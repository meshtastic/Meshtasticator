"""Input adapter for public Meshtastic map and positioned node locations.

The public map is useful as a location source, but it is not a simulator data
model. This adapter only converts map nodes with valid positions into the same
NodeConfig shape the GUI YAML path already uses. Link quality, terrain, and PER
remain simulator concerns configured elsewhere.
"""

import json
import math
import statistics
import urllib.error
import urllib.request

from lib.geo import valid_lat_lon
from lib.node import NodeConfig
from lib.terrain import latlon_to_xy


DEFAULT_MAP_NODES_URL = "https://meshtastic.liamcottle.net/api/v1/nodes"


def decode_map_coordinate(value):
    """Decode Meshtastic map integer coordinates into decimal degrees."""
    if value is None:
        return None
    coordinate = float(value)
    if abs(coordinate) > 180:
        coordinate /= 1e7
    return coordinate


def decode_map_altitude(value):
    """Return a finite positive map altitude in meters, or None for placeholders."""
    if value is None:
        return None
    altitude = float(value)
    if not math.isfinite(altitude) or altitude <= 0:
        return None
    return altitude


def parse_bbox(value):
    """Parse `min_lat,min_lon,max_lat,max_lon` into a numeric tuple."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("map bbox must be min_lat,min_lon,max_lat,max_lon")

    min_lat, min_lon, max_lat, max_lon = [float(part) for part in parts]
    if not all(
        math.isfinite(value)
        for value in (min_lat, min_lon, max_lat, max_lon)
    ):
        raise ValueError("map bbox values must be finite")
    if not valid_lat_lon(min_lat, min_lon) or not valid_lat_lon(max_lat, max_lon):
        raise ValueError("map bbox values must be valid latitude/longitude degrees")
    if min_lat > max_lat or min_lon > max_lon:
        raise ValueError("map bbox minimums must be less than maximums")
    return min_lat, min_lon, max_lat, max_lon


def fetch_map_payload(url=DEFAULT_MAP_NODES_URL):
    request = urllib.request.Request(url, headers={
        "User-Agent": "Meshtasticator map input",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as err:
        raise ValueError(f"could not fetch map payload from {url}: {err}") from err


def role_name_for_node(node):
    role_name = node.get("role_name")
    if role_name:
        return str(role_name).upper()

    # Fallback for map rows where the numeric role is known but the name is not
    # populated. Public map rows may carry this as either an integer or a string.
    # Unrecognized roles stay CLIENT-like unless explicitly mapped.
    try:
        role_value = int(node.get("role"))
    except (TypeError, ValueError):
        role_value = node.get("role")

    return {
        1: "CLIENT_MUTE",
        2: "ROUTER",
        3: "ROUTER_CLIENT",
        4: "REPEATER",
        11: "ROUTER_LATE",
        12: "CLIENT_BASE",
    }.get(role_value, "CLIENT")


def payload_nodes(payload):
    """Return node rows from accepted public-map payload shapes.

    Current map data is normally wrapped as {"nodes": [...]}, but accepting a
    top-level list keeps tests and cached exports from needing a fake envelope.
    """
    if isinstance(payload, dict):
        nodes = payload.get("nodes", [])
    elif isinstance(payload, list):
        nodes = payload
    else:
        raise ValueError("map payload must be a JSON object with nodes or a node list")

    if not isinstance(nodes, list):
        raise ValueError("map payload nodes must be a list")
    return nodes


def filter_positioned_map_nodes(nodes, bbox=None):
    positioned = []
    for node in nodes:
        if not isinstance(node, dict):
            continue

        try:
            lat = decode_map_coordinate(node.get("latitude"))
            lon = decode_map_coordinate(node.get("longitude"))
        except (TypeError, ValueError):
            continue
        if lat is None or lon is None:
            continue
        if not valid_lat_lon(lat, lon):
            continue

        if bbox is not None:
            min_lat, min_lon, max_lat, max_lon = bbox
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue

        positioned.append((node, lat, lon))
    return positioned


def node_configs_from_positioned_rows(
    positioned,
    period,
    antenna_height=1.5,
    hop_limit=3,
    tx_power=30,
    freq=902e6,
    origin=None,
    return_origin=False,
):
    """Build NodeConfig objects from `(node, lat, lon)` positioned rows."""
    if origin is None:
        origin_lat = statistics.median([lat for _, lat, _ in positioned])
        origin_lon = statistics.median([lon for _, _, lon in positioned])
    else:
        try:
            origin_lat, origin_lon = (float(origin[0]), float(origin[1]))
        except (TypeError, ValueError, IndexError) as err:
            raise ValueError("map origin must be valid finite latitude/longitude degrees") from err

        if not valid_lat_lon(origin_lat, origin_lon):
            raise ValueError("map origin must be valid finite latitude/longitude degrees")

    configs = []
    origin_tuple = (origin_lat, origin_lon)
    for sim_node_id, (node, lat, lon) in enumerate(positioned):
        x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)
        role_name = role_name_for_node(node)
        node_dict = {
            "x": round(x, 2),
            "y": round(y, 2),
            # Meshtastic map altitude is absolute altitude, not antenna height.
            # Keep z as antenna height unless SRTM is present to sanity-check
            # and apply the optional absolute altitude per node.
            "z": antenna_height,
            "absoluteAltitude": decode_map_altitude(node.get("altitude")),
            "isRouter": role_name in {"ROUTER", "ROUTER_CLIENT", "ROUTER_LATE"},
            "isRepeater": role_name == "REPEATER",
            "isClientMute": role_name == "CLIENT_MUTE",
            "hopLimit": hop_limit,
            "antennaGain": 0,
            "neighborInfo": False,
        }
        configs.append(NodeConfig.from_gen_scenario_output(sim_node_id, node_dict, period, tx_power, freq))

    if return_origin:
        return configs, origin_tuple
    return configs


def node_configs_from_map_payload(
    payload,
    period,
    bbox=None,
    limit=None,
    antenna_height=1.5,
    hop_limit=3,
    tx_power=30,
    freq=902e6,
    origin=None,
    return_origin=False,
):
    """Build NodeConfig objects from a Meshtastic map `/api/v1/nodes` payload."""
    positioned = filter_positioned_map_nodes(payload_nodes(payload), bbox)
    if limit is not None:
        if limit < 1:
            raise ValueError("map limit must be at least 1")
        positioned = positioned[:limit]
    if not positioned:
        raise ValueError("map payload produced no positioned nodes")

    return node_configs_from_positioned_rows(
        positioned,
        period,
        antenna_height=antenna_height,
        hop_limit=hop_limit,
        tx_power=tx_power,
        freq=freq,
        origin=origin,
        return_origin=return_origin,
    )
