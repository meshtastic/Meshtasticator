import gzip
import sys
import tempfile
import unittest
import zipfile
from array import array
from pathlib import Path

from lib.srtm import (
    HGT_VOID,
    SrtmTile,
    clamp_bbox_to_srtm_coverage,
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

    def test_tile_name_covers_all_hemispheres(self):
        cases = [
            ((41.64, 41.61), "N41E041"),
            ((41.64, -41.61), "N41W042"),
            ((-41.64, 41.61), "S42E041"),
            ((-41.64, -41.61), "S42W042"),
            ((0.0, 0.0), "N00E000"),
            ((-0.0001, -0.0001), "S01W001"),
        ]

        for (lat, lon), tile_name in cases:
            with self.subTest(lat=lat, lon=lon):
                self.assertEqual(srtm_tile_name(lat, lon), tile_name)

    def test_tiles_for_bbox_covers_crossed_integer_degrees(self):
        self.assertEqual(
            tiles_for_bbox((41.5, 41.5, 42.2, 42.2)),
            ["N41E041", "N41E042", "N42E041", "N42E042"],
        )

    def test_tiles_for_bbox_covers_equator_and_prime_meridian_crossing(self):
        self.assertEqual(
            tiles_for_bbox((-0.2, -0.2, 0.2, 0.2)),
            ["N00E000", "N00W001", "S01E000", "S01W001"],
        )

    def test_tiles_for_bbox_excludes_global_edge_tiles(self):
        self.assertEqual(tiles_for_bbox((59.5, 179.5, 60.0, 180.0)), ["N59E179"])

    def test_tiles_for_bbox_maps_zero_span_global_edges_to_existing_tiles(self):
        self.assertEqual(tiles_for_bbox((60.0, 41.0, 60.0, 41.1)), ["N59E041"])
        self.assertEqual(tiles_for_bbox((59.9, 180.0, 60.0, 180.0)), ["N59E179"])

    def test_tiles_for_bbox_rejects_outside_srtm_latitude_coverage(self):
        with self.assertRaisesRegex(ValueError, "56°S and 60°N"):
            tiles_for_bbox((60.0, 41.0, 60.1, 41.1))

        with self.assertRaisesRegex(ValueError, "56°S and 60°N"):
            tiles_for_bbox((-56.1, 41.0, -56.0, 41.1))

    def test_clamp_bbox_to_srtm_coverage_preserves_overlap(self):
        self.assertEqual(
            clamp_bbox_to_srtm_coverage((59.9, 179.9, 60.1, 180.1)),
            (59.9, 179.9, 60.0, 180.0),
        )

    def test_clamp_bbox_to_srtm_coverage_rejects_no_overlap(self):
        with self.assertRaisesRegex(ValueError, "does not overlap"):
            clamp_bbox_to_srtm_coverage((60.1, 41.0, 60.2, 41.1))

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

    def test_terrain_rows_samples_global_edges_from_existing_tiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            write_hgt(cache_dir / "N59E179.hgt", [10, 20, 30, 40])

            rows = list(
                terrain_rows_from_srtm(
                    (60.0, 179.9, 60.0, 180.0),
                    step_meters=20000,
                    cache_dir=cache_dir,
                    download_missing=False,
                )
            )

        self.assertGreater(len(rows), 0)
        self.assertIn("180.0000000", {row["lon"] for row in rows})

    def test_terrain_rows_can_limit_requested_tiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            write_hgt(cache_dir / "N41E041.hgt", [10, 20, 30, 40])

            rows = list(
                terrain_rows_from_srtm(
                    (41.0, 41.0, 43.1, 43.1),
                    step_meters=20000,
                    cache_dir=cache_dir,
                    download_missing=False,
                    tile_names=["N41E041"],
                )
            )

        self.assertGreater(len(rows), 0)
        self.assertEqual({row["lat"][:2] for row in rows}, {"41"})
        self.assertEqual({row["lon"][:2] for row in rows}, {"41"})

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
            with (
                raw_hgt.open("rb") as src,
                gzip.open(source_dir / "N41E041.hgt.gz", "wb") as dst,
            ):
                dst.write(src.read())

            path = ensure_hgt_tile(
                "N41E041",
                cache_dir,
                url_template=f"{source_dir.as_uri()}/{{tile}}.hgt.gz",
            )

            self.assertEqual(path.name, "N41E041.hgt")
            self.assertTrue(path.exists())

    def test_ensure_hgt_tile_selects_requested_member_from_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            cache_dir = Path(tmpdir) / "cache"
            source_dir.mkdir()
            wrong_hgt = source_dir / "N40E040.hgt"
            requested_hgt = source_dir / "N41E041.hgt"
            write_hgt(wrong_hgt, [1, 2, 3, 4])
            write_hgt(requested_hgt, [10, 20, 30, 40])

            with zipfile.ZipFile(source_dir / "N41E041.hgt.zip", "w") as archive:
                archive.write(wrong_hgt, "nested/N40E040.hgt")
                archive.write(requested_hgt, "nested/N41E041.hgt")

            path = ensure_hgt_tile(
                "N41E041",
                cache_dir,
                url_template=f"{source_dir.as_uri()}/{{tile}}.hgt.zip",
            )

            tile = SrtmTile.from_hgt(path)
            self.assertEqual(tile.elevation_at(42.0, 41.0), 10)

    def test_ensure_hgt_tile_rejects_zip_without_requested_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            cache_dir = Path(tmpdir) / "cache"
            source_dir.mkdir()
            wrong_hgt = source_dir / "N40E040.hgt"
            write_hgt(wrong_hgt, [1, 2, 3, 4])

            with zipfile.ZipFile(source_dir / "N41E041.hgt.zip", "w") as archive:
                archive.write(wrong_hgt, "N40E040.hgt")

            with self.assertRaisesRegex(ValueError, "N41E041.hgt"):
                ensure_hgt_tile(
                    "N41E041",
                    cache_dir,
                    url_template=f"{source_dir.as_uri()}/{{tile}}.hgt.zip",
                )

    def test_ensure_hgt_tile_rejects_unknown_template_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "tile"):
                ensure_hgt_tile(
                    "N41E041", tmpdir, url_template="file:///tmp/{missing}.hgt"
                )

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
