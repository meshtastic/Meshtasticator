"""Mapzen Terrain Tiles / SRTM HGT helpers for Meshtasticator terrain inputs.

The simulator can build an in-memory terrain grid directly from cached or
downloaded HGT tiles.
"""

import gzip
import math
import shutil
import sys
import urllib.error
import zipfile
from array import array
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from lib.terrain import TerrainGrid, latlon_to_xy


DEFAULT_SRTM_URL_TEMPLATE = (
    "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_band}/{tile}.hgt.gz"
)
SRTM_DATA_ATTRIBUTION = (
    "Mapzen Terrain Tiles on AWS, using SRTM/NASA and other open elevation sources"
)
SRTM_DATA_ATTRIBUTION_URL = "https://registry.opendata.aws/terrain-tiles/"
HGT_VOID = -32768
SRTM_MIN_LAT = -56.0
SRTM_MAX_LAT = 60.0
SRTM_MIN_LON = -180.0
SRTM_MAX_LON = 180.0


def clamp_bbox_to_srtm_coverage(bbox):
    """Clamp a derived geographic bbox to available SRTM coverage."""
    min_lat, min_lon, max_lat, max_lon = bbox
    clamped = (
        max(min_lat, SRTM_MIN_LAT),
        max(min_lon, SRTM_MIN_LON),
        min(max_lat, SRTM_MAX_LAT),
        min(max_lon, SRTM_MAX_LON),
    )
    if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
        raise ValueError("bbox does not overlap SRTM coverage")
    return clamped


def srtm_tile_name(lat, lon):
    """Return the HGT tile name containing `lat, lon`, for example N41E041."""
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"
    return f"{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}"


def _edge_clamped_sample_coordinate(value, upper_bound):
    return math.nextafter(value, -math.inf) if value == upper_bound else value


def _sample_tile_name(lat, lon):
    lat_for_tile = _edge_clamped_sample_coordinate(lat, SRTM_MAX_LAT)
    lon_for_tile = _edge_clamped_sample_coordinate(lon, SRTM_MAX_LON)
    return srtm_tile_name(lat_for_tile, lon_for_tile)


def _parse_tile_name(tile_name):
    if len(tile_name) != 7 or tile_name[0] not in "NS" or tile_name[3] not in "EW":
        raise ValueError(f"invalid SRTM tile name: {tile_name}")

    lat = int(tile_name[1:3])
    lon = int(tile_name[4:7])
    return (-lat if tile_name[0] == "S" else lat, -lon if tile_name[3] == "W" else lon)


def tile_bbox(tile_name):
    """Return the geographic bbox covered by one SRTM tile."""
    south, west = _parse_tile_name(tile_name)
    return south, west, south + 1.0, west + 1.0


def tiles_for_bbox(bbox):
    """Return sorted SRTM tile names covering a geographic bounding box."""
    min_lat, min_lon, max_lat, max_lon = bbox
    if min_lat < SRTM_MIN_LAT or max_lat > SRTM_MAX_LAT:
        raise ValueError("SRTM coverage is limited to latitudes between 56°S and 60°N")
    if min_lon < SRTM_MIN_LON or max_lon > SRTM_MAX_LON:
        raise ValueError(
            "SRTM coverage is limited to longitudes between 180°W and 180°E"
        )

    min_lat_for_tile = (
        math.nextafter(min_lat, -math.inf) if min_lat == SRTM_MAX_LAT else min_lat
    )
    min_lon_for_tile = (
        math.nextafter(min_lon, -math.inf) if min_lon == SRTM_MAX_LON else min_lon
    )
    max_lat_for_tile = (
        math.nextafter(max_lat, -math.inf)
        if max_lat > min_lat or max_lat == SRTM_MAX_LAT
        else max_lat
    )
    max_lon_for_tile = (
        math.nextafter(max_lon, -math.inf)
        if max_lon > min_lon or max_lon == SRTM_MAX_LON
        else max_lon
    )
    min_lat_tile = math.floor(min_lat_for_tile)
    min_lon_tile = math.floor(min_lon_for_tile)
    max_lat_tile = math.floor(max_lat_for_tile)
    max_lon_tile = math.floor(max_lon_for_tile)
    names = []
    for lat_floor in range(min_lat_tile, max_lat_tile + 1):
        for lon_floor in range(min_lon_tile, max_lon_tile + 1):
            names.append(srtm_tile_name(lat_floor, lon_floor))
    return sorted(set(names))


class SrtmTile:
    """A single SRTM HGT tile loaded into memory.

    HGT files are square grids of signed big-endian 16-bit elevations. Real
    SRTM tiles are usually 1201x1201 or 3601x3601, but tests use tiny square
    files and the reader deliberately accepts those too.
    """

    def __init__(self, tile_name, side, elevations):
        self.tile_name = tile_name
        self.south, self.west = _parse_tile_name(tile_name)
        self.north = self.south + 1
        self.east = self.west + 1
        self.side = side
        self.elevations = elevations

    @classmethod
    def from_hgt(cls, path, tile_name=None):
        path = Path(path)
        tile_name = tile_name or path.name.split(".")[0]
        data = path.read_bytes()
        if len(data) % 2:
            raise ValueError(f"HGT file has odd byte length: {path}")

        cell_count = len(data) // 2
        side = math.isqrt(cell_count)
        if side * side != cell_count:
            raise ValueError(f"HGT file is not a square grid: {path}")

        elevations = array("h")
        elevations.frombytes(data)
        if sys.byteorder == "little":
            elevations.byteswap()
        return cls(tile_name, side, elevations)

    def elevation_at(self, lat, lon):
        """Return nearest-sample elevation in meters, or None for SRTM voids."""
        row = round((self.north - lat) * (self.side - 1))
        col = round((lon - self.west) * (self.side - 1))
        row = min(max(row, 0), self.side - 1)
        col = min(max(col, 0), self.side - 1)
        return self._nearest_valid_elevation(row, col)

    def _nearest_valid_elevation(self, row, col):
        value = self._value_at(row, col)
        if value is not None:
            return value

        # SRTM voids are rare around populated areas. A tiny local search keeps
        # one bad pixel from punching a hole in an otherwise usable terrain grid.
        for radius in range(1, 4):
            candidates = []
            for rr in range(max(0, row - radius), min(self.side, row + radius + 1)):
                for cc in range(max(0, col - radius), min(self.side, col + radius + 1)):
                    if abs(rr - row) != radius and abs(cc - col) != radius:
                        continue
                    value = self._value_at(rr, cc)
                    if value is not None:
                        candidates.append((abs(rr - row) + abs(cc - col), value))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
        return None

    def _value_at(self, row, col):
        value = self.elevations[row * self.side + col]
        if value == HGT_VOID:
            return None
        return float(value)


def ensure_hgt_tile(
    tile_name, cache_dir, url_template=DEFAULT_SRTM_URL_TEMPLATE, download_missing=True
):
    """Return a cached `.hgt` path, downloading and unpacking it when allowed."""
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hgt_path = cache_dir / f"{tile_name}.hgt"
    if hgt_path.exists():
        return hgt_path

    if not download_missing:
        raise FileNotFoundError(f"missing cached SRTM tile: {hgt_path}")

    lat_band = tile_name[:3]
    try:
        url = url_template.format(tile=tile_name, lat_band=lat_band)
    except KeyError as err:
        raise ValueError(
            "url_template may only use {tile} and {lat_band} placeholders"
        ) from err
    parsed_path = Path(urlparse(url).path)
    download_path = (
        cache_dir / f"{tile_name}{''.join(parsed_path.suffixes) or '.download'}"
    )
    partial_hgt_path = cache_dir / f"{tile_name}.hgt.tmp"

    try:
        with urlopen(url, timeout=60) as response, download_path.open("wb") as out:
            shutil.copyfileobj(response, out)
    except (OSError, urllib.error.URLError) as err:
        raise ValueError(
            f"could not download SRTM tile {tile_name} from {url}: {err}"
        ) from err

    try:
        partial_hgt_path.unlink(missing_ok=True)
        if download_path.suffix == ".gz":
            with (
                gzip.open(download_path, "rb") as src,
                partial_hgt_path.open("wb") as out,
            ):
                shutil.copyfileobj(src, out)
        elif download_path.suffix == ".zip":
            with zipfile.ZipFile(download_path) as archive:
                hgt_members = [
                    name for name in archive.namelist() if name.lower().endswith(".hgt")
                ]
                if not hgt_members:
                    raise ValueError(f"zip archive has no .hgt member: {download_path}")
                expected_name = f"{tile_name}.hgt".lower()
                matching_members = [
                    name
                    for name in hgt_members
                    if Path(name).name.lower() == expected_name
                ]
                if not matching_members:
                    raise ValueError(
                        f"zip archive has no {tile_name}.hgt member: {download_path}"
                    )
                with (
                    archive.open(matching_members[0]) as src,
                    partial_hgt_path.open("wb") as out,
                ):
                    shutil.copyfileobj(src, out)
        else:
            download_path.replace(partial_hgt_path)
        partial_hgt_path.replace(hgt_path)
    except (OSError, gzip.BadGzipFile, zipfile.BadZipFile, ValueError) as err:
        partial_hgt_path.unlink(missing_ok=True)
        raise ValueError(f"could not unpack SRTM tile {tile_name}: {err}") from err

    return hgt_path


def _coordinate_values(start, stop, step):
    values = []
    value = start
    while value <= stop + 1e-12:
        values.append(value)
        value += step

    if values and values[-1] < stop:
        values.append(stop)
    return values


def terrain_rows_from_srtm(
    bbox,
    step_meters,
    cache_dir,
    url_template=DEFAULT_SRTM_URL_TEMPLATE,
    download_missing=True,
    tile_names=None,
):
    """Yield lat/lon/elevation rows sampled from SRTM tiles over `bbox`."""
    if not math.isfinite(step_meters) or step_meters <= 0:
        raise ValueError("step_meters must be a positive finite number")

    min_lat, min_lon, max_lat, max_lon = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    lat_step = step_meters / 111320.0
    lon_step = step_meters / (111320.0 * max(math.cos(math.radians(mid_lat)), 0.01))

    requested_tile_names = sorted(set(tile_names or tiles_for_bbox(bbox)))
    tiles = {}
    for tile_name in requested_tile_names:
        path = ensure_hgt_tile(tile_name, cache_dir, url_template, download_missing)
        tiles[tile_name] = SrtmTile.from_hgt(path, tile_name)

    emitted = set()
    for tile_name, tile in tiles.items():
        tile_min_lat, tile_min_lon, tile_max_lat, tile_max_lon = tile_bbox(tile_name)
        sample_min_lat = max(min_lat, tile_min_lat)
        sample_min_lon = max(min_lon, tile_min_lon)
        sample_max_lat = min(max_lat, tile_max_lat)
        sample_max_lon = min(max_lon, tile_max_lon)
        if sample_min_lat > sample_max_lat or sample_min_lon > sample_max_lon:
            continue

        for lat in _coordinate_values(sample_min_lat, sample_max_lat, lat_step):
            for lon in _coordinate_values(sample_min_lon, sample_max_lon, lon_step):
                sample_key = (round(lat, 7), round(lon, 7))
                if sample_key in emitted:
                    continue
                emitted.add(sample_key)

                sample_tile = tiles.get(_sample_tile_name(lat, lon))
                if sample_tile is None:
                    continue
                elevation = sample_tile.elevation_at(lat, lon)
                if elevation is None:
                    continue
                yield {
                    "lat": f"{lat:.7f}",
                    "lon": f"{lon:.7f}",
                    "elevation_m": f"{elevation:.1f}",
                }

        for lat, lon in (
            (sample_min_lat, sample_min_lon),
            (sample_min_lat, sample_max_lon),
            (sample_max_lat, sample_min_lon),
            (sample_max_lat, sample_max_lon),
        ):
            sample_key = (round(lat, 7), round(lon, 7))
            if sample_key in emitted:
                continue
            emitted.add(sample_key)
            tile = tiles.get(_sample_tile_name(lat, lon))
            if tile is None:
                continue
            elevation = tile.elevation_at(lat, lon)
            if elevation is None:
                continue
            yield {
                "lat": f"{lat:.7f}",
                "lon": f"{lon:.7f}",
                "elevation_m": f"{elevation:.1f}",
            }


def terrain_grid_from_srtm(
    bbox,
    step_meters,
    cache_dir,
    origin_lat,
    origin_lon,
    url_template=DEFAULT_SRTM_URL_TEMPLATE,
    download_missing=True,
    tile_names=None,
):
    """Build an in-memory TerrainGrid from SRTM tiles without writing CSV."""
    samples = []
    for row in terrain_rows_from_srtm(
        bbox, step_meters, cache_dir, url_template, download_missing, tile_names
    ):
        lat = float(row["lat"])
        lon = float(row["lon"])
        elevation = float(row["elevation_m"])
        x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)
        samples.append((x, y, elevation))
    return TerrainGrid.from_rows(samples)
