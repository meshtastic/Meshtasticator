import unittest

from lib.map_input import (
    decode_map_altitude,
    decode_map_coordinate,
    node_configs_from_map_payload,
    parse_bbox,
    payload_nodes,
    role_name_for_node,
)
from lib.nodedb_input import node_configs_from_nodedb_payload, positioned_nodedb_nodes, role_name_for_nodedb_node
from lib.node import MESHTASTIC_ROLE


class TestMapInput(unittest.TestCase):
    def test_decode_map_coordinate(self):
        self.assertEqual(decode_map_coordinate(416219136), 41.6219136)
        self.assertEqual(decode_map_coordinate(41.6219136), 41.6219136)
        self.assertIsNone(decode_map_coordinate(None))

    def test_decode_map_altitude_keeps_only_positive_finite_values(self):
        self.assertEqual(decode_map_altitude(120), 120)
        self.assertEqual(decode_map_altitude("12.5"), 12.5)
        self.assertIsNone(decode_map_altitude(None))
        self.assertIsNone(decode_map_altitude(0))
        self.assertIsNone(decode_map_altitude(-1))
        self.assertIsNone(decode_map_altitude(float("inf")))

    def test_parse_bbox(self):
        self.assertEqual(parse_bbox("41.4,41.0,41.9,42.3"), (41.4, 41.0, 41.9, 42.3))

    def test_parse_bbox_rejects_wrong_order(self):
        with self.assertRaises(ValueError):
            parse_bbox("41.9,41.0,41.4,42.3")

    def test_parse_bbox_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            parse_bbox("41.4,41.0,nan,42.3")

    def test_parse_bbox_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            parse_bbox("41.4,41.0,91,42.3")

    def test_payload_nodes_accepts_dict_or_list(self):
        rows = [{"latitude": 1, "longitude": 2}]

        self.assertIs(payload_nodes({"nodes": rows}), rows)
        self.assertIs(payload_nodes(rows), rows)

    def test_payload_nodes_rejects_malformed_payload(self):
        with self.assertRaises(ValueError):
            payload_nodes({"nodes": {}})
        with self.assertRaises(ValueError):
            payload_nodes("not json")

    def test_map_payload_skips_malformed_node_rows(self):
        payload = [
            "not a node",
            {
                "latitude": 416200000,
                "longitude": 415900000,
                "role": 0,
            },
        ]

        configs = node_configs_from_map_payload(payload, 1000)

        self.assertEqual(len(configs), 1)

    def test_map_payload_skips_bad_coordinates(self):
        payload = [
            {"latitude": "not a number", "longitude": 415900000, "role": 0},
            {"latitude": 910000000, "longitude": 415900000, "role": 0},
            {"latitude": 416200000, "longitude": 415900000, "role": 0},
        ]

        configs = node_configs_from_map_payload(payload, 1000)

        self.assertEqual(len(configs), 1)

    def test_numeric_role_fallback_accepts_string_values(self):
        self.assertEqual(role_name_for_node({"role": "2"}), "ROUTER")
        self.assertEqual(role_name_for_node({"role": 12}), "CLIENT_BASE")

    def test_map_payload_builds_projected_node_configs(self):
        payload = {
            "nodes": [
                {
                    "node_id": "1",
                    "node_id_hex": "!00000001",
                    "long_name": "router",
                    "short_name": "r",
                    "latitude": 416200000,
                    "longitude": 415900000,
                    "altitude": 120,
                    "role": 2,
                    "role_name": "ROUTER",
                },
                {
                    "node_id": "2",
                    "node_id_hex": "!00000002",
                    "long_name": "outside",
                    "short_name": "o",
                    "latitude": 500000000,
                    "longitude": 500000000,
                    "altitude": 5,
                    "role": 0,
                    "role_name": "CLIENT",
                },
            ],
        }

        configs = node_configs_from_map_payload(
            payload,
            1000,
            bbox=(41.0, 41.0, 42.0, 42.0),
            antenna_height=2.5,
            hop_limit=5,
            origin=(41.62, 41.59),
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].node_id, 0)
        self.assertEqual(configs[0].role, MESHTASTIC_ROLE.ROUTER)
        self.assertEqual(configs[0].position.z, 2.5)
        self.assertEqual(configs[0].antenna_height, 2.5)
        self.assertEqual(configs[0].absolute_altitude, 120)
        self.assertEqual(configs[0].hop_limit, 5)

    def test_map_altitude_placeholders_do_not_override_antenna_height(self):
        payload = {
            "nodes": [
                {
                    "latitude": 416200000,
                    "longitude": 415900000,
                    "altitude": None,
                    "role": 0,
                },
                {
                    "latitude": 416300000,
                    "longitude": 416000000,
                    "altitude": -1,
                    "role": 0,
                },
                {
                    "latitude": 416400000,
                    "longitude": 416100000,
                    "altitude": 0,
                    "role": 0,
                },
                {
                    "latitude": 416500000,
                    "longitude": 416200000,
                    "altitude": 42949649,
                    "role": 0,
                },
            ],
        }

        configs = node_configs_from_map_payload(
            payload,
            1000,
            antenna_height=2.5,
        )

        self.assertEqual([config.position.z for config in configs], [2.5, 2.5, 2.5, 2.5])
        self.assertEqual([config.antenna_height for config in configs], [2.5, 2.5, 2.5, 2.5])
        self.assertEqual([config.absolute_altitude for config in configs], [None, None, None, 42949649])

    def test_map_payload_can_return_projection_origin(self):
        payload = [{
            "latitude": 416200000,
            "longitude": 415900000,
            "role": 0,
        }]

        configs, origin = node_configs_from_map_payload(payload, 1000, return_origin=True)

        self.assertEqual(len(configs), 1)
        self.assertEqual(origin, (41.62, 41.59))

    def test_map_payload_rejects_empty_limit(self):
        payload = {
            "nodes": [{
                "latitude": 416200000,
                "longitude": 415900000,
                "role": 0,
            }],
        }

        with self.assertRaises(ValueError):
            node_configs_from_map_payload(payload, 1000, limit=0)

    def test_map_payload_rejects_invalid_projection_origin(self):
        payload = [{
            "latitude": 416200000,
            "longitude": 415900000,
            "role": 0,
        }]

        with self.assertRaises(ValueError):
            node_configs_from_map_payload(payload, 1000, origin=(91.0, 41.59))
        with self.assertRaises(ValueError):
            node_configs_from_map_payload(payload, 1000, origin=("bad", 41.59))

    def test_nodedb_payload_builds_projected_node_configs(self):
        payload = {
            "nodesByNum": {
                1: {
                    "num": 1,
                    "user": {"id": "!00000001", "role": "ROUTER"},
                    "position": {"latitude": 41.62, "longitude": 41.59, "altitude": 120},
                },
                2: {
                    "num": 2,
                    "user": {"id": "!00000002", "role": "CLIENT"},
                    "position": {"latitudeI": 416300000, "longitudeI": 416000000},
                },
                3: {
                    "num": 3,
                    "user": {"id": "!00000003", "role": "CLIENT"},
                    "position": {"latitude": 50.0, "longitude": 50.0},
                },
            }
        }

        configs = node_configs_from_nodedb_payload(
            payload,
            1000,
            bbox=(41.0, 41.0, 42.0, 42.0),
            antenna_height=2.5,
            hop_limit=5,
            origin=(41.62, 41.59),
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual([config.node_id for config in configs], [0, 1])
        self.assertEqual(configs[0].role, MESHTASTIC_ROLE.ROUTER)
        self.assertEqual(configs[0].absolute_altitude, 120)
        self.assertEqual([config.antenna_height for config in configs], [2.5, 2.5])
        self.assertEqual([config.hop_limit for config in configs], [5, 5])

    def test_nodedb_payload_uses_supplied_radio_defaults(self):
        payload = [
            {
                "num": 1,
                "user": {"role": "ROUTER"},
                "position": {
                    "latitude": 41.62,
                    "longitude": 41.59,
                    "altitude": 120,
                },
            },
            {
                "num": 2,
                "user": {"role": "CLIENT"},
                "position": {
                    "latitude": 41.63,
                    "longitude": 41.60,
                    "altitude": 10,
                },
            },
        ]

        configs = node_configs_from_nodedb_payload(
            payload,
            1000,
            tx_power=14,
            freq=433e6,
        )

        self.assertEqual([config.tx_power for config in configs], [14, 14])
        self.assertEqual([config.freq for config in configs], [433e6, 433e6])

    def test_nodedb_payload_skips_unpositioned_nodes(self):
        payload = [
            {"num": 1, "position": {"time": 1640206266}},
            {"num": 2, "position": {"latitude": 41.62, "longitude": 41.59}},
        ]

        positioned = positioned_nodedb_nodes(payload)

        self.assertEqual(len(positioned), 1)
        self.assertEqual(positioned[0][1:], (41.62, 41.59))

    def test_nodedb_role_defaults_to_client(self):
        self.assertEqual(role_name_for_nodedb_node({}), "CLIENT")
        self.assertEqual(role_name_for_nodedb_node({"user": {"role": "router_client"}}), "ROUTER_CLIENT")
        self.assertEqual(role_name_for_nodedb_node({"user": {"role": 2}}), "ROUTER")
        self.assertEqual(role_name_for_nodedb_node({"user": {"role": "12"}}), "CLIENT_BASE")


if __name__ == "__main__":
    unittest.main()
