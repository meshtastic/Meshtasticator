"""Optional land-cover clutter loss for radio links.

Terrain handles hills, curvature, and Fresnel obstruction. It does not know
whether a lowland path crosses apartment blocks, a beach/coastal opening, or a
mountain-side vantage point looking down into the city. This module adds that
separate, data-driven clutter term from a small raster CSV.
"""

import bisect
import csv
import math
from pathlib import Path

from lib.csv_validation import finite_float, finite_lat_lon
from lib.terrain import latlon_to_xy, terrain_ground_elevation


class ClutterGrid:
    """Nearest-cell lookup for small land-cover rasters."""

    def __init__(self, samples):
        self.samples = samples
        self.xs = sorted({x for x, _, _ in samples})
        self.ys = sorted({y for _, y, _ in samples})
        self.by_xy = {(x, y): clutter_class for x, y, clutter_class in samples}
        self.is_regular = len(self.xs) * len(self.ys) == len(samples)

    @classmethod
    def from_csv(cls, path, origin_lat=None, origin_lon=None):
        samples = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row_number, row in enumerate(reader, start=2):
                if "x_m" in row and "y_m" in row:
                    x = finite_float(row, "x_m", "clutter", row_number)
                    y = finite_float(row, "y_m", "clutter", row_number)
                elif "lat" in row and "lon" in row:
                    if origin_lat is None or origin_lon is None:
                        raise ValueError("lat/lon clutter CSV requires GEO_ORIGIN_LAT and GEO_ORIGIN_LON")
                    lat, lon = finite_lat_lon(row, "clutter", row_number)
                    x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)
                else:
                    raise ValueError("clutter CSV needs x_m/y_m or lat/lon columns")

                clutter_class = row.get("clutter_class")
                if clutter_class is None or not clutter_class.strip():
                    raise ValueError("clutter CSV needs clutter_class column")
                samples.append((x, y, clutter_class.strip().lower()))

        if not samples:
            raise ValueError(f"clutter CSV has no samples: {path}")
        return cls(samples)

    @staticmethod
    def _nearest_axis_value(values, value):
        index = bisect.bisect_left(values, value)
        if index <= 0:
            return values[0]
        if index >= len(values):
            return values[-1]

        before = values[index - 1]
        after = values[index]
        return before if abs(value - before) <= abs(after - value) else after

    def class_at(self, x, y):
        if self.is_regular:
            nearest_x = self._nearest_axis_value(self.xs, x)
            nearest_y = self._nearest_axis_value(self.ys, y)
            return self.by_xy[(nearest_x, nearest_y)]

        _, _, clutter_class = min(
            self.samples,
            key=lambda sample: math.hypot(x - sample[0], y - sample[1]),
        )
        return clutter_class


def _clutter_grid(conf):
    if not conf.CLUTTER_ENABLED or not conf.CLUTTER_GRID_FILE:
        return None

    # Lat/lon CSVs are projected into scenario-local meters, so the projection
    # origin is part of the loaded grid identity, not just metadata.
    cache_identity = (conf.CLUTTER_GRID_FILE, conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON)
    cached_identity = getattr(conf, "_clutter_grid_identity", None)
    if getattr(conf, "_clutter_grid", None) is not None and cached_identity == cache_identity:
        return conf._clutter_grid

    path = Path(conf.CLUTTER_GRID_FILE)
    conf._clutter_grid = ClutterGrid.from_csv(path, conf.GEO_ORIGIN_LAT, conf.GEO_ORIGIN_LON)
    conf._clutter_grid_identity = cache_identity
    return conf._clutter_grid


def _class_loss_db_per_km(conf, clutter_class):
    if clutter_class in {"urban", "building"}:
        return conf.CLUTTER_URBAN_LOSS_DB_PER_KM
    if clutter_class in {"suburban", "residential"}:
        return conf.CLUTTER_SUBURBAN_LOSS_DB_PER_KM
    if clutter_class in {"forest", "wood"}:
        return conf.CLUTTER_FOREST_LOSS_DB_PER_KM
    if clutter_class in {"water", "coastal_water"}:
        return conf.CLUTTER_WATER_LOSS_DB_PER_KM
    return conf.CLUTTER_OPEN_LOSS_DB_PER_KM


def clutter_path_features(conf, tx_point, rx_point):
    """Return coarse land-cover fractions along a radio path.

    The radio calibration model needs reusable features, not node-pair lookup
    tables. These fractions let a fitted model learn patterns such as "urban
    lowland paths behave differently from open coastal paths" and apply that
    lesson to new generated node pairs.
    """
    grid = _clutter_grid(conf)
    if grid is None:
        return {
            "urban_fraction": 0.0,
            "open_fraction": 0.0,
            "water_fraction": 0.0,
            "forest_fraction": 0.0,
            "endpoint_urban_count": 0.0,
        }

    cache = getattr(conf, "_clutter_feature_cache", None)
    if cache is None:
        cache = {}
        conf._clutter_feature_cache = cache

    cache_key = (
        round(tx_point.x, 2),
        round(tx_point.y, 2),
        round(tx_point.z, 2),
        round(rx_point.x, 2),
        round(rx_point.y, 2),
        round(rx_point.z, 2),
        conf.CLUTTER_GRID_FILE,
        conf.GEO_ORIGIN_LAT,
        conf.GEO_ORIGIN_LON,
        conf.CLUTTER_PROFILE_SAMPLES,
    )
    if cache_key in cache:
        return cache[cache_key]

    samples = max(1, conf.CLUTTER_PROFILE_SAMPLES)
    class_counts = {}
    for index in range(samples):
        fraction = (index + 0.5) / samples
        x = tx_point.x + (rx_point.x - tx_point.x) * fraction
        y = tx_point.y + (rx_point.y - tx_point.y) * fraction
        clutter_class = grid.class_at(x, y)
        class_counts[clutter_class] = class_counts.get(clutter_class, 0) + 1

    endpoint_urban_count = 0
    for point in (tx_point, rx_point):
        endpoint_class = grid.class_at(point.x, point.y)
        if endpoint_class in {"urban", "building", "suburban", "residential"}:
            endpoint_urban_count += 1

    features = {
        "urban_fraction": class_counts.get("urban", 0) / samples,
        "open_fraction": class_counts.get("open", 0) / samples,
        "water_fraction": (class_counts.get("water", 0) + class_counts.get("coastal_water", 0)) / samples,
        "forest_fraction": (class_counts.get("forest", 0) + class_counts.get("wood", 0)) / samples,
        "endpoint_urban_count": float(endpoint_urban_count),
    }
    cache[cache_key] = features
    return features


def _is_high_vantage(conf, point):
    ground = terrain_ground_elevation(conf, point)
    return ground is not None and ground >= conf.CLUTTER_HIGH_VANTAGE_ELEVATION_M


def clutter_obstruction_loss(conf, tx_point, rx_point):
    """Estimate extra land-cover clutter loss in dB for a TX/RX path."""
    grid = _clutter_grid(conf)
    if grid is None:
        return 0.0

    cache = getattr(conf, "_clutter_loss_cache", None)
    if cache is None:
        cache = {}
        conf._clutter_loss_cache = cache

    cache_key = (
        round(tx_point.x, 2),
        round(tx_point.y, 2),
        round(tx_point.z, 2),
        round(rx_point.x, 2),
        round(rx_point.y, 2),
        round(rx_point.z, 2),
        conf.CLUTTER_GRID_FILE,
        conf.GEO_ORIGIN_LAT,
        conf.GEO_ORIGIN_LON,
        conf.CLUTTER_PROFILE_SAMPLES,
        conf.CLUTTER_URBAN_LOSS_DB_PER_KM,
        conf.CLUTTER_SUBURBAN_LOSS_DB_PER_KM,
        conf.CLUTTER_FOREST_LOSS_DB_PER_KM,
        conf.CLUTTER_OPEN_LOSS_DB_PER_KM,
        conf.CLUTTER_WATER_LOSS_DB_PER_KM,
        conf.CLUTTER_URBAN_ENDPOINT_LOSS_DB,
        conf.CLUTTER_HIGH_VANTAGE_ELEVATION_M,
        conf.CLUTTER_HIGH_VANTAGE_LOSS_FACTOR,
        conf.CLUTTER_COASTAL_PATH_LOSS_FACTOR,
        conf.CLUTTER_COASTAL_SAMPLE_FRACTION,
        conf.CLUTTER_MAX_LOSS_DB,
    )
    if cache_key in cache:
        return cache[cache_key]

    horizontal_distance = math.hypot(rx_point.x - tx_point.x, rx_point.y - tx_point.y)
    if horizontal_distance <= 0:
        return 0.0

    samples = max(1, conf.CLUTTER_PROFILE_SAMPLES)
    class_counts = {}
    path_loss_rate = 0.0
    for index in range(samples):
        fraction = (index + 0.5) / samples
        x = tx_point.x + (rx_point.x - tx_point.x) * fraction
        y = tx_point.y + (rx_point.y - tx_point.y) * fraction
        clutter_class = grid.class_at(x, y)
        class_counts[clutter_class] = class_counts.get(clutter_class, 0) + 1
        path_loss_rate += _class_loss_db_per_km(conf, clutter_class)

    path_loss = (path_loss_rate / samples) * (horizontal_distance / 1000.0)

    # Coastal or sea-adjacent paths are often real line-of-sight corridors. Do
    # not let a few nearby urban cells make them look like street-canyon links.
    water_samples = class_counts.get("water", 0) + class_counts.get("coastal_water", 0)
    open_samples = class_counts.get("open", 0) + class_counts.get("beach", 0)
    if (water_samples + open_samples) / samples >= conf.CLUTTER_COASTAL_SAMPLE_FRACTION:
        path_loss *= conf.CLUTTER_COASTAL_PATH_LOSS_FACTOR

    tx_high = _is_high_vantage(conf, tx_point)
    rx_high = _is_high_vantage(conf, rx_point)
    if tx_high or rx_high:
        path_loss *= conf.CLUTTER_HIGH_VANTAGE_LOSS_FACTOR

    endpoint_loss = 0.0
    for point, high_vantage in ((tx_point, tx_high), (rx_point, rx_high)):
        endpoint_class = grid.class_at(point.x, point.y)
        if endpoint_class in {"urban", "building", "suburban", "residential"} and not high_vantage:
            endpoint_loss += conf.CLUTTER_URBAN_ENDPOINT_LOSS_DB

    loss = min(path_loss + endpoint_loss, conf.CLUTTER_MAX_LOSS_DB)
    cache[cache_key] = loss
    return loss
