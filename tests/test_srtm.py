import gzip
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

from lib.srtm import (
    HGT_VOID,
    SrtmTile,
    ensure_hgt_tile,
    terrain_grid_from_srtm,
    terrain_rows_from_srtm,
    srtm_tile_name,
    tiles_for_bbox,
)


def write_hgt(path, values):
    data = array("h", values)
    if sys.byteorder == "little":
        data.byteswap()
    Path(path).write_bytes(data.tobytes())


class TestSrtm(unittest.TestCase):
    def test_tile_name_uses_srtm_flooring(self):
        self.assertEqual(srtm_tile_name(41.64, 41.61), "N41E041")
        self.assertEqual(srtm_tile_name(-0.1, -1.2), "S01W002")

    def test_tiles_for_bbox_covers_crossed_integer_degrees(self):
        self.assertEqual(
            tiles_for_bbox((41.5, 41.5, 42.2, 42.2)),
            ["N41E041", "N41E042", "N42E041", "N42E042"],
        )

    def test_hgt_tile_reads_big_endian_elevation_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "N41E041.hgt"
            write_hgt(path, [10, 20, 30, 40, 50, 60, 70, 80, 90])

            tile = SrtmTile.from_hgt(path)

        self.assertEqual(tile.elevation_at(42.0, 41.0), 10)
        self.assertEqual(tile.elevation_at(41.0, 42.0), 90)

    def test_hgt_void_uses_nearby_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "N41E041.hgt"
            write_hgt(path, [10, 20, 30, 40, HGT_VOID, 60, 70, 80, 90])

            tile = SrtmTile.from_hgt(path)

        self.assertIsNotNone(tile.elevation_at(41.5, 41.5))

    def test_terrain_grid_from_srtm_avoids_csv_intermediate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            write_hgt(cache_dir / "N41E041.hgt", [10, 20, 30, 40, 50, 60, 70, 80, 90])

            grid = terrain_grid_from_srtm(
                (41.0, 41.0, 41.1, 41.1),
                step_meters=20000,
                cache_dir=cache_dir,
                origin_lat=41.0,
                origin_lon=41.0,
                download_missing=False,
            )

        self.assertGreater(len(grid.samples), 0)
        self.assertIsNotNone(grid.elevation_at(0, 0))

    def test_terrain_rows_rejects_non_finite_step(self):
        with self.assertRaises(ValueError):
            list(terrain_rows_from_srtm((41.0, 41.0, 41.1, 41.1), float("nan"), "/tmp"))

    def test_ensure_hgt_tile_downloads_and_unpacks_gzip_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            cache_dir = Path(tmpdir) / "cache"
            source_dir.mkdir()
            raw_hgt = source_dir / "N41E041.hgt"
            write_hgt(raw_hgt, [1, 2, 3, 4])
            with raw_hgt.open("rb") as src, gzip.open(source_dir / "N41E041.hgt.gz", "wb") as dst:
                dst.write(src.read())

            path = ensure_hgt_tile(
                "N41E041",
                cache_dir,
                url_template=f"{source_dir.as_uri()}/{{tile}}.hgt.gz",
            )

            self.assertEqual(path.name, "N41E041.hgt")
            self.assertTrue(path.exists())

    def test_ensure_hgt_tile_rejects_unknown_template_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "tile"):
                ensure_hgt_tile("N41E041", tmpdir, url_template="file:///tmp/{missing}.hgt")

    def test_ensure_hgt_tile_does_not_cache_failed_unpack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            cache_dir = Path(tmpdir) / "cache"
            source_dir.mkdir()
            (source_dir / "N41E041.hgt.gz").write_bytes(b"not gzip")

            with self.assertRaisesRegex(ValueError, "could not unpack"):
                ensure_hgt_tile(
                    "N41E041",
                    cache_dir,
                    url_template=f"{source_dir.as_uri()}/{{tile}}.hgt.gz",
                )

            self.assertFalse((cache_dir / "N41E041.hgt").exists())
            self.assertFalse((cache_dir / "N41E041.hgt.tmp").exists())

if __name__ == "__main__":
    unittest.main()
