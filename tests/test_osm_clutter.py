import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lib.osm_clutter import (
    classify_osm_element,
    main as osm_clutter_main,
    parse_origin,
    payload_elements,
    point_in_polygon,
    rasterize_clutter,
    write_clutter_csv,
)


class TestOsmClutter(unittest.TestCase):
    def test_classifies_osm_tags_to_radio_clutter_classes(self):
        self.assertEqual(classify_osm_element({"building": "yes"}), "urban")
        self.assertEqual(classify_osm_element({"landuse": "residential"}), "urban")
        self.assertEqual(classify_osm_element({"natural": "wood"}), "forest")
        self.assertEqual(classify_osm_element({"natural": "water"}), "water")
        self.assertEqual(classify_osm_element({"natural": "beach"}), "open")

    def test_parse_origin_rejects_non_finite_values(self):
        self.assertEqual(parse_origin("41.6,41.6"), (41.6, 41.6))
        with self.assertRaises(ValueError):
            parse_origin("41.6,nan")

    def test_parse_origin_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            parse_origin("91,41.6")

    def test_point_in_polygon(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]

        self.assertTrue(point_in_polygon(5, 5, polygon))
        self.assertFalse(point_in_polygon(15, 5, polygon))

    def test_payload_elements_rejects_malformed_overpass_shape(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            payload_elements([])
        with self.assertRaisesRegex(ValueError, "elements"):
            payload_elements({"elements": {}})

    def test_rasterize_clutter_marks_building_cells_urban(self):
        payload = {
            "elements": [{
                "type": "way",
                "tags": {"building": "yes"},
                "geometry": [
                    {"lat": 0.0005, "lon": 0.0005},
                    {"lat": 0.0005, "lon": 0.0015},
                    {"lat": 0.0015, "lon": 0.0015},
                    {"lat": 0.0015, "lon": 0.0005},
                    {"lat": 0.0005, "lon": 0.0005},
                ],
            }],
        }

        rows = rasterize_clutter(payload, (0.0, 0.0, 0.002, 0.002), origin=(0.0, 0.0), step_m=100.0)

        self.assertIn("urban", {row["clutter_class"] for row in rows})

    def test_rasterize_clutter_skips_malformed_osm_elements(self):
        payload = {
            "elements": [
                "not an element",
                {
                    "type": "way",
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lat": 0.0005, "lon": 0.0005},
                        {"lat": "bad", "lon": 0.0015},
                        {"lat": 0.0015, "lon": 0.0005},
                    ],
                },
                {
                    "type": "way",
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lat": 0.0005, "lon": 0.0005},
                        {"lat": 0.0005, "lon": 0.0015},
                        {"lat": 0.0015, "lon": 0.0015},
                        {"lat": 0.0015, "lon": 0.0005},
                    ],
                },
            ],
        }

        rows = rasterize_clutter(payload, (0.0, 0.0, 0.002, 0.002), origin=(0.0, 0.0), step_m=100.0)

        self.assertIn("urban", {row["clutter_class"] for row in rows})

    def test_rasterize_clutter_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            rasterize_clutter({"elements": []}, (0.0, 0.0, 0.002, 0.002), step_m=0)

    def test_write_clutter_csv_uses_lf_line_endings(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "clutter.csv"

            write_clutter_csv([{
                "x_m": 0,
                "y_m": 0,
                "lat": 41.0,
                "lon": 41.0,
                "clutter_class": "open",
            }], output)

            raw = output.read_bytes()

        self.assertIn(b"\n", raw)
        self.assertNotIn(b"\r\n", raw)

    def test_write_clutter_csv_creates_parent_directories(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nested" / "clutter.csv"

            write_clutter_csv([{
                "x_m": 0,
                "y_m": 0,
                "lat": 41.0,
                "lon": 41.0,
                "clutter_class": "open",
            }], output)

            self.assertTrue(output.exists())

    def test_cli_rejects_invalid_bbox_without_traceback(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                osm_clutter_main([
                    "--bbox", "0,0,nan,0.002",
                    "--input-json", "/dev/null",
                    "--output", "/tmp/unused-clutter.csv",
                ])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("map bbox values must be finite", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
