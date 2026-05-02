import unittest

from lib.node import MESHTASTIC_ROLE, node_configs_from_yaml, origin_from_yaml, packet_is_rx_candidate


def sample_node(x):
    return {
        "x": x,
        "y": 0,
        "z": 1.5,
        "isRouter": False,
        "isRepeater": False,
        "isClientMute": False,
        "hopLimit": 3,
        "antennaGain": 0,
        "neighborInfo": False,
    }


class TestNodeConfigYaml(unittest.TestCase):
    def test_plain_gui_node_map_is_accepted(self):
        configs = node_configs_from_yaml({0: sample_node(10), 1: sample_node(20)}, 1000)

        self.assertEqual([cfg.node_id for cfg in configs], [0, 1])
        self.assertEqual(configs[0].role, MESHTASTIC_ROLE.CLIENT)
        self.assertEqual(configs[1].position.x, 20)

    def test_wrapped_real_mesh_node_map_is_accepted(self):
        raw = {
            "origin": {"lat": 41.64, "lon": 41.62},
            "nodes": {"0": sample_node(10), "1": sample_node(20)},
        }

        configs = node_configs_from_yaml(raw, 1000)

        self.assertEqual([cfg.node_id for cfg in configs], [0, 1])
        self.assertEqual(configs[0].period, 1000)

    def test_wrapped_real_mesh_node_ids_are_remapped_to_sim_indices(self):
        raw = {
            "origin": {"lat": 41.64, "lon": 41.62},
            "nodes": {
                "3944424993": sample_node(10),
                "3944424994": sample_node(20),
            },
        }

        configs = node_configs_from_yaml(raw, 1000)

        self.assertEqual([cfg.node_id for cfg in configs], [0, 1])
        self.assertEqual([cfg.position.x for cfg in configs], [10, 20])

    def test_wrapped_node_map_origin_is_available_for_terrain(self):
        raw = {
            "origin": {"lat": 41.64, "lon": 41.62},
            "nodes": {"0": sample_node(10)},
        }

        self.assertEqual(origin_from_yaml(raw), (41.64, 41.62))

    def test_node_yaml_must_be_a_node_map(self):
        with self.assertRaisesRegex(ValueError, "node YAML"):
            node_configs_from_yaml(["not", "a", "node", "map"], 1000)

    def test_wrapped_node_map_must_be_a_map(self):
        raw = {
            "origin": {"lat": 41.64, "lon": 41.62},
            "nodes": ["not", "a", "node", "map"],
        }

        with self.assertRaisesRegex(ValueError, "node YAML"):
            node_configs_from_yaml(raw, 1000)

    def test_wrapped_node_map_origin_must_be_finite(self):
        raw = {
            "origin": {"lat": "nan", "lon": 41.62},
            "nodes": {"0": sample_node(10)},
        }

        with self.assertRaisesRegex(ValueError, "origin.lat"):
            origin_from_yaml(raw)

    def test_wrapped_node_map_origin_must_be_in_coordinate_range(self):
        raw = {
            "origin": {"lat": 91, "lon": 41.62},
            "nodes": {"0": sample_node(10)},
        }

        with self.assertRaisesRegex(ValueError, "latitude/longitude"):
            origin_from_yaml(raw)


class TestPacketRxCandidate(unittest.TestCase):
    def test_legacy_collision_model_tracks_only_decodable_packets(self):
        packet = type("Packet", (), {
            "sensedByN": [False, True],
            "detectedByN": [True, True],
        })()

        self.assertFalse(packet_is_rx_candidate(packet, 0, capture_model_enabled=False))
        self.assertTrue(packet_is_rx_candidate(packet, 1, capture_model_enabled=False))

    def test_capture_model_tracks_cad_detected_interference(self):
        packet = type("Packet", (), {
            "sensedByN": [False, True],
            "detectedByN": [True, False],
        })()

        self.assertTrue(packet_is_rx_candidate(packet, 0, capture_model_enabled=True))
        self.assertFalse(packet_is_rx_candidate(packet, 1, capture_model_enabled=True))


if __name__ == "__main__":
    unittest.main()
