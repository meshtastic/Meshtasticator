"""Export public OpenStreetMap land-use/building data to clutter CSV.

This is a standalone data-prep helper. The simulator runtime reads the exported
CSV and never fetches OSM/Overpass data implicitly.
"""

import argparse
import csv
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from lib.geo import valid_lat_lon
from lib.map_input import parse_bbox
from lib.terrain import latlon_to_xy, xy_to_latlon


DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

URBAN_LANDUSE = {
    "commercial",
    "construction",
    "garages",
    "industrial",
    "military",
    "railway",
    "residential",
    "retail",
}
FOREST_VALUES = {"forest", "wood"}
WATER_VALUES = {"basin", "reservoir", "salt_pond", "water"}
OPEN_NATURAL = {"beach", "grassland", "heath", "sand", "scrub"}


def overpass_query(bbox):
    """Build a bounded Overpass query for clutter-relevant OSM polygons."""
    min_lat, min_lon, max_lat, max_lon = bbox
    box = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    return f"""
[out:json][timeout:90];
(
  way["building"]({box});
  way["landuse"~"^(commercial|construction|garages|industrial|military|railway|residential|retail|forest)$"]({box});
  way["natural"~"^(beach|grassland|heath|sand|scrub|water|wood)$"]({box});
  way["water"]({box});
);
out tags geom;
"""


def fetch_overpass_payload(bbox, url=DEFAULT_OVERPASS_URL):
    data = urllib.parse.urlencode({"data": overpass_query(bbox)}).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Meshtasticator OSM clutter exporter",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as err:
        raise ValueError(f"could not fetch OSM clutter payload from {url}: {err}") from err


def parse_origin(value):
    """Parse `lat,lon` for local raster projection origin."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("origin must be lat,lon")

    lat, lon = [float(part) for part in parts]
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("origin values must be finite")
    if not valid_lat_lon(lat, lon):
        raise ValueError("origin values must be valid latitude/longitude degrees")
    return lat, lon


def classify_osm_element(tags):
    """Map OSM tags to broad clutter classes used by the radio model."""
    if tags.get("building"):
        return "urban"

    landuse = tags.get("landuse")
    natural = tags.get("natural")
    water = tags.get("water")

    if landuse in URBAN_LANDUSE:
        return "urban"
    if landuse in FOREST_VALUES or natural in FOREST_VALUES:
        return "forest"
    if landuse in WATER_VALUES or natural in WATER_VALUES or water:
        return "water"
    if natural in OPEN_NATURAL:
        return "open"
    return None


def payload_elements(payload):
    """Return Overpass elements from the expected JSON object shape."""
    if not isinstance(payload, dict):
        raise ValueError("OSM payload must be a JSON object")

    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        raise ValueError("OSM payload elements must be a list")
    return elements


def point_in_polygon(x, y, polygon):
    """Return True when a point is inside a simple polygon."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def polygon_bounds(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_centroid(polygon):
    if not polygon:
        return 0.0, 0.0
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def osm_polygons(payload, origin):
    """Yield `(clutter_class, polygon_xy, bounds, centroid)` from Overpass JSON."""
    origin_lat, origin_lon = origin
    for element in payload_elements(payload):
        if not isinstance(element, dict):
            continue

        geometry = element.get("geometry") or []
        tags = element.get("tags") or {}
        if not isinstance(geometry, list) or not isinstance(tags, dict):
            continue

        clutter_class = classify_osm_element(tags)
        if not clutter_class or len(geometry) < 3:
            continue

        polygon = []
        for point in geometry:
            if not isinstance(point, dict):
                polygon = []
                break
            try:
                lat = float(point["lat"])
                lon = float(point["lon"])
            except (KeyError, TypeError, ValueError):
                polygon = []
                break
            if not valid_lat_lon(lat, lon):
                polygon = []
                break
            polygon.append(latlon_to_xy(lat, lon, origin_lat, origin_lon))
        if len(polygon) < 3:
            continue

        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        bounds = polygon_bounds(polygon)
        centroid = polygon_centroid(polygon)
        yield clutter_class, polygon, bounds, centroid


def _frange(start, stop, step):
    value = start
    epsilon = step / 1000.0
    while value <= stop + epsilon:
        yield value
        value += step


def classify_cell(x, y, polygons, step_m):
    """Classify one clutter grid cell from intersecting OSM polygons."""
    hits = {"urban": 0, "forest": 0, "water": 0, "open": 0}
    half = step_m / 2.0
    for clutter_class, polygon, bounds, centroid in polygons:
        min_x, min_y, max_x, max_y = bounds
        if x < min_x - half or x > max_x + half or y < min_y - half or y > max_y + half:
            continue

        # Land-use polygons often contain the cell center. Building footprints
        # are much smaller than the exported raster cell, so also count nearby
        # building centroids/bounds as urban evidence.
        if point_in_polygon(x, y, polygon):
            hits[clutter_class] = hits.get(clutter_class, 0) + 2
        elif clutter_class == "urban" and min_x - half <= x <= max_x + half and min_y - half <= y <= max_y + half:
            cx, cy = centroid
            if abs(cx - x) <= half and abs(cy - y) <= half:
                hits["urban"] += 1

    if hits["water"] > 0:
        return "water"
    if hits["urban"] > 0:
        return "urban"
    if hits["forest"] > 0:
        return "forest"
    return "open"


def rasterize_clutter(payload, bbox, origin=None, step_m=500.0):
    """Rasterize OSM polygons to rows suitable for `ClutterGrid.from_csv()`."""
    if not math.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be a positive finite number")

    if origin is None:
        min_lat, min_lon, max_lat, max_lon = bbox
        origin = ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)

    origin_lat, origin_lon = origin
    min_lat, min_lon, max_lat, max_lon = bbox
    min_x, min_y = latlon_to_xy(min_lat, min_lon, origin_lat, origin_lon)
    max_x, max_y = latlon_to_xy(max_lat, max_lon, origin_lat, origin_lon)
    min_x, max_x = sorted((min_x, max_x))
    min_y, max_y = sorted((min_y, max_y))

    polygons = list(osm_polygons(payload, origin))
    rows = []
    for x in _frange(math.floor(min_x / step_m) * step_m, math.ceil(max_x / step_m) * step_m, step_m):
        for y in _frange(math.floor(min_y / step_m) * step_m, math.ceil(max_y / step_m) * step_m, step_m):
            lat, lon = xy_to_latlon(x, y, origin_lat, origin_lon)
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            rows.append({
                "x_m": round(x, 2),
                "y_m": round(y, 2),
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "clutter_class": classify_cell(x, y, polygons, step_m),
            })
    return rows


def write_clutter_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["x_m", "y_m", "lat", "lon", "clutter_class"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="export OSM land-use/building clutter to Meshtasticator CSV")
    parser.add_argument("--bbox", required=True, help="min_lat,min_lon,max_lat,max_lon")
    parser.add_argument("--origin", help="origin lat,lon for local x/y output; defaults to bbox center")
    parser.add_argument("--step-meters", type=float, default=500.0, help="output raster spacing in meters")
    parser.add_argument("--output", required=True, help="output clutter CSV path")
    parser.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL, help="Overpass interpreter endpoint")
    parser.add_argument("--input-json", help="read an existing Overpass JSON response instead of fetching")
    args = parser.parse_args(argv)

    try:
        bbox = parse_bbox(args.bbox)
    except ValueError as err:
        parser.error(str(err))

    origin = None
    if args.origin:
        try:
            origin = parse_origin(args.origin)
        except ValueError as err:
            parser.error(str(err))

    if args.input_json:
        try:
            with open(args.input_json, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as err:
            parser.error(f"could not read OSM clutter JSON: {err}")
    else:
        payload = fetch_overpass_payload(bbox, args.overpass_url)

    try:
        rows = rasterize_clutter(payload, bbox, origin=origin, step_m=args.step_meters)
    except ValueError as err:
        parser.error(str(err))
    write_clutter_csv(rows, args.output)


if __name__ == "__main__":
    main()
