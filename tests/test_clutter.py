import tempfile
import unittest
from pathlib import Path

from lib.clutter import ClutterGrid, clutter_obstruction_loss, clutter_path_features
from lib.config import Config
from lib.point import Point


class TestClutter(unittest.TestCase):
    def test_csv_grid_returns_nearest_regular_cell_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clutter.csv"
            path.write_text(
                "x_m,y_m,clutter_class\n"
                "0,0,urban\n"
                "500,0,water\n",
                encoding="utf-8",
            )

            grid = ClutterGrid.from_csv(path)

        self.assertEqual(grid.class_at(20, 0), "urban")
        self.assertEqual(grid.class_at(480, 0), "water")

    def test_csv_grid_rejects_non_finite_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clutter.csv"
            path.write_text(
                "x_m,y_m,clutter_class\n"
                "0,nan,urban\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "row 2"):
                ClutterGrid.from_csv(path)

    def test_csv_grid_rejects_blank_clutter_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clutter.csv"
            path.write_text(
                "x_m,y_m,clutter_class\n"
                "0,0,   \n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "clutter_class"):
                ClutterGrid.from_csv(path)

    def test_latlon_csv_grid_rejects_out_of_range_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clutter.csv"
            path.write_text(
                "lat,lon,clutter_class\n"
                "91,41,urban\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "latitude/longitude"):
                ClutterGrid.from_csv(path, origin_lat=41.0, origin_lon=41.0)

    def test_urban_clutter_adds_more_loss_than_coastal_open_path(self):
        conf = Config()
        conf.CLUTTER_ENABLED = True
        conf.CLUTTER_PROFILE_SAMPLES = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            urban_path = Path(tmpdir) / "urban.csv"
            urban_path.write_text(
                "x_m,y_m,clutter_class\n"
                "0,0,urban\n"
                "500,0,urban\n"
                "1000,0,urban\n",
                encoding="utf-8",
            )
            conf.CLUTTER_GRID_FILE = str(urban_path)

            urban_loss = clutter_obstruction_loss(conf, Point(0, 0, 2), Point(1000, 0, 2))

            open_path = Path(tmpdir) / "open.csv"
            open_path.write_text(
                "x_m,y_m,clutter_class\n"
                "0,0,water\n"
                "500,0,water\n"
                "1000,0,water\n",
                encoding="utf-8",
            )
            conf._clutter_grid = None
            conf._clutter_loss_cache = {}
            conf.CLUTTER_GRID_FILE = str(open_path)

            open_loss = clutter_obstruction_loss(conf, Point(0, 0, 2), Point(1000, 0, 2))

        self.assertGreater(urban_loss, open_loss)

    def test_latlon_grid_cache_tracks_projection_origin(self):
        conf = Config()
        conf.CLUTTER_ENABLED = True
        conf.CLUTTER_PROFILE_SAMPLES = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clutter.csv"
            path.write_text(
                "lat,lon,clutter_class\n"
                "10.0,10.0,urban\n"
                "10.0,10.01,water\n",
                encoding="utf-8",
            )
            conf.CLUTTER_GRID_FILE = str(path)
            conf.GEO_ORIGIN_LAT = 10.0
            conf.GEO_ORIGIN_LON = 10.0

            first = clutter_path_features(conf, Point(0, 0, 1), Point(0, 0, 1))
            self.assertEqual(first["urban_fraction"], 1.0)

            # The same CSV can be projected around a different scenario origin.
            # Include origin in the grid cache key so map/preset inputs cannot
            # accidentally reuse stale cell coordinates.
            conf.GEO_ORIGIN_LON = 10.01
            second = clutter_path_features(conf, Point(0, 0, 1), Point(0, 0, 1))
            self.assertEqual(second["water_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
