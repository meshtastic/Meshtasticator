"""Optional terrain obstruction model for radio links.

The simulator's default coordinates are local meters with `Point.z` acting like
antenna height. Terrain-aware input loaders can lift points to absolute antenna
altitude (`ground elevation + antenna height`) so the existing 3D distance path
already reflects terrain before any extra RF obstruction loss is applied.

The loss model is intentionally conservative and dependency-free: sample the
path, find the worst Fresnel/line-of-sight obstruction, then apply the standard
single knife-edge diffraction approximation for that obstruction. It is not a
full ray tracer, but it captures the important Batumi-mesh case where hills and
ridges matter more than flat-earth distance alone.
"""

import math


EARTH_RADIUS_M = 6371000.0
NODE_Z_REFERENCE_GROUND = "ground"
NODE_Z_REFERENCE_SEA_LEVEL = "sea_level"
MAX_REASONABLE_STRUCTURE_HEIGHT_M = 850.0


def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    """Project WGS84 lat/lon to local x/y meters with an equirectangular map."""
    origin_lat_rad = math.radians(origin_lat)
    x = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(origin_lat_rad)
    y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


def xy_to_latlon(x, y, origin_lat, origin_lon):
    """Inverse of latlon_to_xy for small local simulation areas."""
    origin_lat_rad = math.radians(origin_lat)
    origin_cos = math.cos(origin_lat_rad)
    if abs(origin_cos) < 1e-9:
        raise ValueError("origin latitude is too close to a pole for local x/y projection")

    lat = origin_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(x / (EARTH_RADIUS_M * origin_cos))
    return lat, lon


class TerrainGrid:
    """Small scattered terrain sample grid with inverse-distance interpolation."""

    def __init__(self, samples):
        self.samples = samples

    @classmethod
    def from_rows(cls, rows):
        """Build a terrain grid from `(x_m, y_m, elevation_m)` samples."""
        samples = []
        for row_number, row in enumerate(rows, start=1):
            try:
                x, y, elevation = row
            except (TypeError, ValueError) as err:
                raise ValueError(f"terrain sample {row_number} must have x, y, and elevation") from err
            x = float(x)
            y = float(y)
            elevation = float(elevation)
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(elevation):
                raise ValueError(f"terrain sample {row_number} values must be finite")
            samples.append((x, y, elevation))

        if not samples:
            raise ValueError("terrain grid has no samples")
        return cls(samples)

    def elevation_at(self, x, y):
        weighted_sum = 0.0
        weight_total = 0.0

        nearest = sorted(
            ((math.hypot(x - sx, y - sy), elevation) for sx, sy, elevation in self.samples),
            key=lambda item: item[0],
        )[:8]

        for distance, elevation in nearest:
            if distance < 0.01:
                return elevation
            weight = 1.0 / (distance * distance)
            weighted_sum += elevation * weight
            weight_total += weight

        return weighted_sum / weight_total


def _terrain_grid(conf):
    """Return the configured in-memory terrain grid when terrain is enabled."""
    if not conf.TERRAIN_ENABLED:
        return None
    return getattr(conf, "TERRAIN_GRID", None)


def terrain_ground_elevation(conf, point):
    """Return terrain elevation at a point, or None when terrain is unavailable."""
    grid = _terrain_grid(conf)
    if grid is None:
        return None
    return grid.elevation_at(point.x, point.y)


def map_altitude_if_plausible(node, ground):
    """Return per-node map altitude when SRTM says it is physically plausible."""
    altitude = getattr(node, "absolute_altitude", None)
    if altitude is None:
        return None
    altitude = float(altitude)
    if not math.isfinite(altitude):
        return None
    if altitude <= ground:
        return None
    if altitude > ground + MAX_REASONABLE_STRUCTURE_HEIGHT_M:
        return None
    return altitude


def node_antenna_height(node):
    """Return node antenna height above ground for config and live node types."""
    return getattr(
        node,
        "antenna_height",
        getattr(node, "antennaHeight", node.position.z),
    )


def apply_terrain_altitude(terrain_grid, node):
    """Lift one node z coordinate to absolute antenna altitude from terrain.

    `node.antenna_height` remains antenna height above local ground. That keeps
    path-loss models with explicit antenna-height terms from receiving absolute
    MSL altitude by mistake.
    """
    ground = terrain_grid.elevation_at(node.position.x, node.position.y)
    map_altitude = map_altitude_if_plausible(node, ground)
    if map_altitude is None:
        node.position.z = ground + node_antenna_height(node)
    else:
        node.position.z = map_altitude


def apply_terrain_altitudes(terrain_grid, node_config):
    """Lift node z coordinates to absolute antenna altitude from terrain."""
    for node in node_config:
        apply_terrain_altitude(terrain_grid, node)


def terrain_antenna_altitude(conf, grid, point):
    """Return absolute antenna altitude for a terrain-backed point."""
    ground = grid.elevation_at(point.x, point.y)
    min_altitude = ground + conf.TERRAIN_MIN_ANTENNA_HEIGHT_M
    if getattr(conf, "NODE_Z_REFERENCE", NODE_Z_REFERENCE_GROUND) == NODE_Z_REFERENCE_SEA_LEVEL:
        return max(point.z, min_altitude)
    return ground + max(point.z, conf.TERRAIN_MIN_ANTENNA_HEIGHT_M)


def knife_edge_loss_db(v):
    """ITU-R single knife-edge diffraction loss approximation."""
    if v <= -0.78:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


def terrain_obstruction_loss(conf, tx_point, rx_point, freq):
    """Estimate extra terrain obstruction loss in dB for a TX/RX path."""
    grid = _terrain_grid(conf)
    if grid is None:
        return 0.0

    # Packet objects are created often and ask for every receiver. In a static
    # topology the terrain term is a pure function of the two endpoint
    # coordinates, so cache it on Config and keep the packet hot path cheap.
    cache = getattr(conf, "_terrain_loss_cache", None)
    if cache is None:
        cache = {}
        conf._terrain_loss_cache = cache

    cache_key = (
        round(tx_point.x, 2),
        round(tx_point.y, 2),
        round(tx_point.z, 2),
        round(rx_point.x, 2),
        round(rx_point.y, 2),
        round(rx_point.z, 2),
        round(freq, 0),
        id(getattr(conf, "TERRAIN_GRID", None)),
        conf.GEO_ORIGIN_LAT,
        conf.GEO_ORIGIN_LON,
        conf.TERRAIN_PROFILE_SAMPLES,
        conf.TERRAIN_FRESNEL_CLEARANCE,
        conf.TERRAIN_EFFECTIVE_EARTH_RADIUS_MULTIPLIER,
        conf.TERRAIN_MIN_ANTENNA_HEIGHT_M,
        conf.TERRAIN_MAX_LOSS_DB,
    )
    if cache_key in cache:
        return cache[cache_key]

    horizontal_distance = math.hypot(rx_point.x - tx_point.x, rx_point.y - tx_point.y)
    if horizontal_distance <= 0:
        return 0.0

    wavelength = 299792458.0 / freq
    tx_height = terrain_antenna_altitude(conf, grid, tx_point)
    rx_height = terrain_antenna_altitude(conf, grid, rx_point)
    effective_earth_radius = EARTH_RADIUS_M * conf.TERRAIN_EFFECTIVE_EARTH_RADIUS_MULTIPLIER
    curvature_scale = (horizontal_distance * horizontal_distance) / (2.0 * effective_earth_radius)

    worst_loss = 0.0
    for i in range(1, conf.TERRAIN_PROFILE_SAMPLES):
        fraction = i / conf.TERRAIN_PROFILE_SAMPLES
        x = tx_point.x + (rx_point.x - tx_point.x) * fraction
        y = tx_point.y + (rx_point.y - tx_point.y) * fraction
        ground = grid.elevation_at(x, y)
        los_height = tx_height + (rx_height - tx_height) * fraction
        d1 = horizontal_distance * fraction
        d2 = horizontal_distance - d1

        fresnel_radius = math.sqrt(wavelength * d1 * d2 / horizontal_distance)
        # A straight local projection makes long links look too flat. Apply the
        # usual 4/3 effective Earth radius bulge at each path sample before
        # measuring Fresnel clearance against the line between antennas.
        earth_bulge = curvature_scale * fraction * (1.0 - fraction)
        obstruction_height = (
            ground
            + earth_bulge
            + conf.TERRAIN_FRESNEL_CLEARANCE * fresnel_radius
            - los_height
        )
        if obstruction_height <= 0:
            continue

        v = obstruction_height * math.sqrt(2.0 * horizontal_distance / (wavelength * d1 * d2))
        worst_loss = max(worst_loss, knife_edge_loss_db(v))

    loss = min(worst_loss, conf.TERRAIN_MAX_LOSS_DB)
    cache[cache_key] = loss
    return loss
