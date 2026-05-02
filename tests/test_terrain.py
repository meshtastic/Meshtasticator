import unittest

from lib.config import Config
from lib.node import NodeConfig
from lib.point import Point
from lib.terrain import TerrainGrid, apply_terrain_altitudes, latlon_to_xy, terrain_ground_elevation, terrain_obstruction_loss, xy_to_latlon


class TestTerrain(unittest.TestCase):
    def test_latlon_projection_preserves_origin(self):
        x, y = latlon_to_xy(41.6, 41.6, 41.6, 41.6)

        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_latlon_projection_round_trips(self):
        lat, lon = 41.65, 41.62
        x, y = latlon_to_xy(lat, lon, 41.6, 41.6)

        out_lat, out_lon = xy_to_latlon(x, y, 41.6, 41.6)

        self.assertAlmostEqual(out_lat, lat)
        self.assertAlmostEqual(out_lon, lon)

    def test_grid_interpolates_exact_sample(self):
        grid = TerrainGrid.from_rows([
            (0, 0, 5),
            (100, 0, 25),
        ])

        self.assertEqual(grid.elevation_at(100, 0), 25)

    def test_grid_rejects_non_finite_samples(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            TerrainGrid.from_rows([(0, float("nan"), 5)])

    def test_configured_grid_provides_ground_elevation(self):
        conf = Config()
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_GRID = TerrainGrid.from_rows([
            (0, 0, 5),
            (100, 0, 25),
        ])

        self.assertEqual(terrain_ground_elevation(conf, Point(0, 0, 1)), 5)

    def test_terrain_altitudes_keep_antenna_height_separate(self):
        conf = Config()
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_GRID = TerrainGrid.from_rows([
            (0, 0, 100),
            (100, 0, 120),
        ])
        nodes = [
            NodeConfig(0, Point(0, 0, 2.5), 1000),
            NodeConfig(1, Point(100, 0, 3.0), 1000),
        ]

        apply_terrain_altitudes(conf, nodes)

        self.assertTrue(conf.TERRAIN_NODE_Z_IS_ABSOLUTE_ALTITUDE)
        self.assertEqual([node.antenna_height for node in nodes], [2.5, 3.0])
        self.assertEqual([node.position.z for node in nodes], [102.5, 123.0])

    def test_ridge_adds_obstruction_loss(self):
        conf = Config()
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_PROFILE_SAMPLES = 10
        conf.TERRAIN_GRID = TerrainGrid.from_rows([
            (0, 0, 0),
            (500, 0, 120),
            (1000, 0, 0),
        ])

        loss = terrain_obstruction_loss(
            conf,
            Point(0, 0, 2),
            Point(1000, 0, 2),
            conf.FREQ,
        )

        self.assertGreater(loss, 0)

    def test_effective_earth_radius_adds_curvature_loss_on_long_flat_link(self):
        conf = Config()
        conf.TERRAIN_ENABLED = True
        conf.TERRAIN_PROFILE_SAMPLES = 10
        conf.TERRAIN_FRESNEL_CLEARANCE = 0.0
        conf.TERRAIN_GRID = TerrainGrid.from_rows([
            (0, 0, 0),
            (25000, 0, 0),
            (50000, 0, 0),
        ])

        loss = terrain_obstruction_loss(
            conf,
            Point(0, 0, 2),
            Point(50000, 0, 2),
            conf.FREQ,
        )

        self.assertGreater(loss, 0)


if __name__ == "__main__":
    unittest.main()
