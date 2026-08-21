"""Small geographic validation helpers shared by map-oriented inputs."""

import math


def valid_lat_lon(lat, lon):
    """Return whether latitude/longitude are finite WGS84-style coordinates."""
    return (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )
