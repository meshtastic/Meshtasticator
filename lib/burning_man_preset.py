"""Deterministic Burning Man scenario input for the normal simulator.

This module keeps the old PR #33 event-shape assumptions as data generation:
three elevated routers, clustered camps, dense city-ring clutter, and clear
playa/Man areas. The simulator itself remains the regular loraMesh path.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import yaml


CENTER_PLAZA_RADIUS_M = 850.0
CITY_RADIUS_M = 1700.0
TRASH_FENCE_RADIUS_M = 2587.0
DEFAULT_CLIENTS = 120
DEFAULT_SEED = 42

ROUTER_CONFIGS = [
    {"clock": "7:30", "street": "B"},
    {"clock": "3:00", "street": "E"},
    {"clock": "10:00", "street": "F"},
]

GROUP_CONFIG = {
    "solo": {
        "probability": 0.20,
        "size_range": (1, 1),
        "cluster_radius": 0,
        "min_group_distance": 50,
        "city_probability": 0.80,
    },
    "small_group": {
        "probability": 0.65,
        "size_range": (2, 8),
        "cluster_radius": 20,
        "min_group_distance": 100,
        "city_probability": 0.85,
    },
    "medium_group": {
        "probability": 0.12,
        "size_range": (9, 30),
        "cluster_radius": 35,
        "min_group_distance": 100,
        "city_probability": 0.75,
    },
    "large_camp": {
        "probability": 0.03,
        "size_range": (40, 80),
        "cluster_radius": 60,
        "min_group_distance": 100,
        "city_probability": 1.0,
    },
}

USER_BEHAVIORS = {
    "camper": {"period_s": 8 * 60 * 60, "mobile_probability": 0.1},
    "explorer": {"period_s": 2 * 60 * 60, "mobile_probability": 0.6},
    "staff": {"period_s": 60 * 60, "mobile_probability": 0.8},
    "heavy_user": {"period_s": 30 * 60, "mobile_probability": 0.4},
}

USER_TYPE_DISTRIBUTION = [
    ("camper", 0.60),
    ("explorer", 0.25),
    ("staff", 0.10),
    ("heavy_user", 0.05),
]

ATTENUATION_DISTRIBUTION = [
    (0, 0.80),
    (8, 0.12),
    (15, 0.06),
    (25, 0.02),
]


def _weighted_choice(rng: random.Random, distribution):
    threshold = rng.random()
    cumulative = 0.0
    for value, probability in distribution:
        cumulative += probability
        if threshold <= cumulative:
            return value
    return distribution[-1][0]


def _activity_group_size(rng: random.Random, group_type: str, max_group_size: int) -> int:
    min_size, max_size = GROUP_CONFIG[group_type]["size_range"]
    if group_type == "solo":
        size = 1
    elif group_type == "small_group":
        size = min(max_size, max(min_size, int(rng.expovariate(1 / 4) + min_size)))
    elif group_type == "medium_group":
        size = min(max_size, max(min_size, int(rng.normalvariate(18, 6))))
    else:
        size = min(max_size, max(min_size, int(rng.normalvariate(60, 12))))
    return min(size, max_group_size)


def generate_activity_groups(total_clients: int, rng: random.Random) -> list[dict]:
    groups = []
    assigned = 0
    max_group_size = max(1, int(total_clients * 0.20))
    distribution = [
        (group_type, config["probability"])
        for group_type, config in GROUP_CONFIG.items()
    ]

    while assigned < total_clients:
        group_type = _weighted_choice(rng, distribution)
        size = min(_activity_group_size(rng, group_type, max_group_size), total_clients - assigned)
        if size > 0:
            groups.append({"type": group_type, "size": size})
            assigned += size

    return groups


def burning_man_fence_points() -> list[tuple[float, float]]:
    gps_coords = [
        (40.78236, -119.23530),
        (40.80570, -119.21965),
        (40.80163, -119.18533),
        (40.77568, -119.17971),
        (40.76373, -119.21050),
    ]
    center_lat = sum(lat for lat, _ in gps_coords) / len(gps_coords)
    center_lon = sum(lon for _, lon in gps_coords) / len(gps_coords)

    coords_m = []
    for lat, lon in gps_coords:
        x = (lon - center_lon) * 111000 * math.cos(math.radians(center_lat))
        y = (lat - center_lat) * 111000
        coords_m.append((x, y))

    current_radius = max(math.hypot(x, y) for x, y in coords_m)
    scale = TRASH_FENCE_RADIUS_M / current_radius
    scaled = [(x * scale, y * scale) for x, y in coords_m]

    apex_x, apex_y = max(scaled, key=lambda point: math.hypot(point[0], point[1]))
    rotation = math.radians(45) - math.atan2(apex_y, apex_x)
    return [
        (
            x * math.cos(rotation) - y * math.sin(rotation),
            x * math.sin(rotation) + y * math.cos(rotation),
        )
        for x, y in scaled
    ]


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    p1x, p1y = polygon[0]
    for index in range(1, len(polygon) + 1):
        p2x, p2y = polygon[index % len(polygon)]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _nearest_point_on_segment(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return x1 + t * dx, y1 + t * dy


def _move_inside_fence(x: float, y: float, fence: list[tuple[float, float]]) -> tuple[float, float]:
    nearest = min(
        (
            _nearest_point_on_segment(x, y, fence[index][0], fence[index][1], fence[(index + 1) % len(fence)][0], fence[(index + 1) % len(fence)][1])
            for index in range(len(fence))
        ),
        key=lambda point: math.hypot(x - point[0], y - point[1]),
    )
    return nearest[0] * 0.98, nearest[1] * 0.98


def _router_positions() -> list[tuple[float, float]]:
    clock_to_angle = {
        "3:00": 0,
        "6:00": -90,
        "7:30": -135,
        "9:00": 180,
        "10:00": 150,
        "12:00": 90,
    }
    streets = ["ESPLANADE", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    street_spacing = (CITY_RADIUS_M * 0.9 - CENTER_PLAZA_RADIUS_M) / (len(streets) - 1)
    street_to_radius = {
        street: CENTER_PLAZA_RADIUS_M + index * street_spacing
        for index, street in enumerate(streets)
    }

    positions = []
    for config in ROUTER_CONFIGS:
        angle = math.radians(clock_to_angle[config["clock"]] - 45)
        radius = street_to_radius[config["street"]]
        positions.append((radius * math.cos(angle), radius * math.sin(angle)))
    return positions


def _node_entry(
    x: float,
    y: float,
    z: float,
    *,
    is_router: bool,
    antenna_gain: float,
    hop_limit: int,
    tx_power_dbm: int,
    period_ms: int,
) -> dict:
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "z": round(z, 3),
        "isRouter": is_router,
        "isRepeater": False,
        "isClientMute": False,
        "hopLimit": hop_limit,
        "antennaGain": antenna_gain,
        "neighborInfo": False,
        "antennaHeight": z,
        "txPowerDbm": tx_power_dbm,
        "periodMs": period_ms,
    }


def generate_burning_man_preset(total_clients: int = DEFAULT_CLIENTS, seed: int = DEFAULT_SEED) -> dict:
    rng = random.Random(seed)
    nodes = {}
    fence = burning_man_fence_points()

    for node_id, (x, y) in enumerate(_router_positions()):
        nodes[str(node_id)] = _node_entry(
            x,
            y,
            10.7,
            is_router=True,
            antenna_gain=5,
            hop_limit=7,
            tx_power_dbm=30,
            period_ms=8 * 60 * 60 * 1000,
        )

    node_id = len(nodes)
    for group in generate_activity_groups(total_clients, rng):
        config = GROUP_CONFIG[group["type"]]
        center_x, center_y = _place_group_center(rng, nodes, config, fence)

        for _ in range(group["size"]):
            x, y = _place_group_node(rng, nodes, center_x, center_y, config["cluster_radius"], fence)
            behavior = _weighted_choice(rng, USER_TYPE_DISTRIBUTION)
            attenuation = _weighted_choice(rng, ATTENUATION_DISTRIBUTION)
            period_ms = USER_BEHAVIORS[behavior]["period_s"] * 1000

            entry = _node_entry(
                x,
                y,
                1.5,
                is_router=False,
                antenna_gain=0,
                hop_limit=3,
                tx_power_dbm=20,
                period_ms=period_ms,
            )
            entry["burningManZone"] = _zone_for_point(x, y)
            entry["burningManGroup"] = group["type"]
            entry["burningManUserBehavior"] = behavior
            entry["enclosureLossDb"] = attenuation
            nodes[str(node_id)] = entry
            node_id += 1

    return {
        "origin": {"lat": 40.7867, "lon": -119.2044},
        "radio_environment": {
            "clutter_suburban_loss_db_per_km": 7.0,
            "clutter_open_loss_db_per_km": 0.2,
            "clutter_urban_endpoint_loss_db": 3.0,
            "clutter_max_loss_db": 25.0,
        },
        "scenario": {
            "name": "burning_man",
            "seed": seed,
            "clients": total_clients,
            "description": "Black Rock City-style event mesh generated as normal loraMesh input.",
        },
        "nodes": nodes,
    }


def _place_group_center(rng, nodes, config, fence):
    for _ in range(100):
        city_preferred = rng.random() < config["city_probability"]
        radius_min, radius_max = (0.0, CITY_RADIUS_M) if city_preferred else (CITY_RADIUS_M, TRASH_FENCE_RADIUS_M)
        radius = math.sqrt(rng.uniform(radius_min * radius_min, radius_max * radius_max))
        angle = rng.uniform(-math.pi, math.pi)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        if not point_in_polygon(x, y, fence):
            continue
        if _too_close_to_existing(x, y, nodes, config["min_group_distance"]):
            continue
        return x, y

    angle = rng.uniform(-math.pi, math.pi)
    radius = rng.uniform(0, TRASH_FENCE_RADIUS_M * 0.9)
    return radius * math.cos(angle), radius * math.sin(angle)


def _place_group_node(rng, nodes, center_x, center_y, cluster_radius, fence):
    for _ in range(50):
        if cluster_radius <= 0:
            x, y = center_x, center_y
        else:
            x = center_x + rng.gauss(0, cluster_radius)
            y = center_y + rng.gauss(0, cluster_radius)
        if not point_in_polygon(x, y, fence):
            x, y = _move_inside_fence(x, y, fence)
        if not _too_close_to_existing(x, y, nodes, 10):
            return x, y
    return _move_inside_fence(center_x, center_y, fence)


def _too_close_to_existing(x, y, nodes, min_distance):
    return any(math.hypot(x - node["x"], y - node["y"]) < min_distance for node in nodes.values())


def _zone_for_point(x: float, y: float) -> str:
    radius = math.hypot(x, y)
    if radius <= CENTER_PLAZA_RADIUS_M:
        return "center_playa"
    if radius <= CITY_RADIUS_M:
        return "city_ring"
    return "open_playa"


def clutter_class_for_point(x: float, y: float) -> str:
    return "suburban" if _zone_for_point(x, y) == "city_ring" else "open"


def write_burning_man_clutter_csv(path: Path, step_m: int = 250) -> None:
    fence = burning_man_fence_points()
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(math.ceil(TRASH_FENCE_RADIUS_M / step_m) * step_m)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["x_m", "y_m", "clutter_class"], lineterminator="\n")
        writer.writeheader()
        for y in range(-limit, limit + step_m, step_m):
            for x in range(-limit, limit + step_m, step_m):
                if point_in_polygon(x, y, fence):
                    writer.writerow({"x_m": x, "y_m": y, "clutter_class": clutter_class_for_point(x, y)})


def write_burning_man_preset(path: Path, total_clients: int = DEFAULT_CLIENTS, seed: int = DEFAULT_SEED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(generate_burning_man_preset(total_clients, seed), fh, sort_keys=False)
