"""Input adapter for positions stored in a local Meshtastic device NodeDB."""

from lib.geo import valid_lat_lon
from lib.map_input import (
    decode_map_altitude,
    decode_map_coordinate,
    node_configs_from_positioned_rows,
    role_name_for_node,
)


def fetch_nodedb_payload(host=None, port=None, serial_port=None):
    """Read positioned nodes from a local Meshtastic device.

    This uses the same Python client state that powers `meshtastic --nodes`.
    The CLI pretty-prints `interface.nodesByNum`; the simulator wants that raw
    structure so it can project node positions without parsing a terminal table.
    """
    if host is not None and serial_port is not None:
        raise ValueError("--nodedb-host and --nodedb-serial-port are mutually exclusive")

    try:
        if host is not None:
            from meshtastic import tcp_interface

            iface = tcp_interface.TCPInterface(hostname=host, portNumber=port or 4403)
        else:
            from meshtastic import serial_interface

            iface = serial_interface.SerialInterface(devPath=serial_port)
    except Exception as err:
        raise ValueError(f"could not connect to Meshtastic device: {err}") from err

    try:
        return list((iface.nodesByNum or {}).values())
    finally:
        close = getattr(iface, "close", None)
        if close is not None:
            close()


def nodedb_payload_nodes(payload):
    """Return node rows from accepted NodeDB payload shapes."""
    nodes = payload
    if hasattr(payload, "nodesByNum"):
        nodes = payload.nodesByNum
    elif isinstance(payload, dict) and "nodesByNum" in payload:
        nodes = payload["nodesByNum"]

    if isinstance(nodes, dict):
        nodes = nodes.values()
    if not isinstance(nodes, (list, tuple)) and not hasattr(nodes, "__iter__"):
        raise ValueError("NodeDB payload must be nodesByNum, a node list, or an iterable of node rows")
    return list(nodes)


def role_name_for_nodedb_node(node):
    user = node.get("user") if isinstance(node, dict) else None
    if isinstance(user, dict) and user.get("role") is not None:
        return normalize_nodedb_role(user["role"])
    if isinstance(node, dict) and node.get("role") is not None:
        return normalize_nodedb_role(node["role"])
    return "CLIENT"


def normalize_nodedb_role(raw_role):
    try:
        return role_name_for_node({"role": int(raw_role)})
    except (TypeError, ValueError):
        return str(raw_role).upper()


def positioned_nodedb_nodes(nodes, bbox=None):
    """Return `(node, lat, lon)` rows for NodeDB entries with valid positions."""
    positioned = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        position = node.get("position")
        if not isinstance(position, dict):
            continue

        try:
            lat = position.get("latitude")
            if lat is None:
                lat = decode_map_coordinate(position.get("latitudeI"))
            lon = position.get("longitude")
            if lon is None:
                lon = decode_map_coordinate(position.get("longitudeI"))
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

        node = {
            "role_name": role_name_for_nodedb_node(node),
            "altitude": decode_map_altitude(position.get("altitude")),
        }
        positioned.append((node, lat, lon))
    return positioned


def node_configs_from_nodedb_payload(
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
    """Build NodeConfig objects from a local Meshtastic NodeDB payload."""
    positioned = positioned_nodedb_nodes(nodedb_payload_nodes(payload), bbox)
    if limit is not None:
        if limit < 1:
            raise ValueError("map limit must be at least 1")
        positioned = positioned[:limit]
    if not positioned:
        raise ValueError("NodeDB payload produced no positioned nodes")

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
