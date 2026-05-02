import csv
import unittest

import yaml

from lib.config import Config
from lib.presets import (
    PRESET_ROOT,
    apply_preset_radio_calibration,
    load_preset_node_configs,
    preset_calibration_observations,
    preset_clutter_grid,
    preset_origin,
    preset_radio_calibration,
    preset_terrain_grid,
)
from lib.terrain import xy_to_latlon


# Broad enough to include the Batumi coastal/ridge mesh snapshot, narrow enough
# to catch accidentally bundled non-Georgia/global map data.
BATUMI_GEORGIA_BBOX = (41.50, 41.50, 41.82, 41.86)


def inside_bbox(lat, lon, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


class TestPresets(unittest.TestCase):
    def test_batumi_preset_loads_nodes_and_terrain(self):
        configs = load_preset_node_configs("batumi", 1000)

        self.assertEqual(len(configs), 92)
        self.assertTrue(preset_terrain_grid("batumi").exists())
        self.assertTrue(preset_clutter_grid("batumi").exists())
        self.assertEqual(preset_origin("batumi"), (41.6442879, 41.61536))

    def test_batumi_preset_applies_radio_calibration(self):
        conf = Config()

        apply_preset_radio_calibration(conf, "batumi")

        self.assertEqual(preset_radio_calibration("batumi")["noise_level"], -110.5)
        self.assertEqual(conf.NOISE_LEVEL, -110.5)
        self.assertEqual(conf.PATH_LOSS_DISTANCE_FLOOR_M, 780.0)
        self.assertEqual(conf.REPORTED_SNR_MIN_DB, -21.25)
        self.assertEqual(conf.REPORTED_SNR_MAX_DB, 8.25)
        self.assertTrue(conf.LINK_CALIBRATION_MODEL_ENABLED)
        self.assertEqual(conf.LINK_CALIBRATION_SNR_MIN_DB, -35.0)
        self.assertEqual(conf.LINK_CALIBRATION_SNR_MAX_DB, 8.25)
        self.assertIn("raw_snr_clip", conf.LINK_CALIBRATION_COEFFICIENTS)
        self.assertEqual(len(preset_calibration_observations("batumi")), 296)

    def test_batumi_preset_nodes_are_inside_batumi_georgia_area(self):
        raw = yaml.safe_load((PRESET_ROOT / "batumi.yaml").read_text(encoding="utf-8"))
        origin = raw["origin"]

        coords = []
        for node in raw["nodes"].values():
            lat, lon = xy_to_latlon(float(node["x"]), float(node["y"]), origin["lat"], origin["lon"])
            coords.append((lat, lon))

        self.assertEqual(len(coords), 92)
        self.assertTrue(all(inside_bbox(lat, lon, BATUMI_GEORGIA_BBOX) for lat, lon in coords))

    def test_batumi_preset_does_not_publish_source_metadata(self):
        raw = yaml.safe_load((PRESET_ROOT / "batumi.yaml").read_text(encoding="utf-8"))

        for node in raw["nodes"].values():
            self.assertFalse(any(key.startswith("source_") for key in node))

        node_ids = set(raw["nodes"].keys())
        for link in raw["calibration_observations"]:
            self.assertEqual(set(link.keys()), {"from", "to", "snr"})
            self.assertIn(link["from"], node_ids)
            self.assertIn(link["to"], node_ids)

    def test_batumi_terrain_grid_is_inside_georgia_side_of_region(self):
        with preset_terrain_grid("batumi").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        self.assertGreater(len(rows), 0)
        self.assertTrue(all(
            inside_bbox(float(row["lat"]), float(row["lon"]), BATUMI_GEORGIA_BBOX)
            for row in rows
        ))

    def test_batumi_clutter_grid_is_inside_georgia_side_of_region(self):
        with preset_clutter_grid("batumi").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        classes = {row["clutter_class"] for row in rows}
        self.assertGreater(len(rows), 0)
        self.assertIn("urban", classes)
        self.assertIn("open", classes)
        self.assertTrue(all(
            inside_bbox(float(row["lat"]), float(row["lon"]), BATUMI_GEORGIA_BBOX)
            for row in rows
        ))


if __name__ == "__main__":
    unittest.main()
