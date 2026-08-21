import re
import unittest
from pathlib import Path

from lib.config import Config


class TestDocumentation(unittest.TestCase):
    def test_modem_table_matches_config_base_settings(self):
        docs = Path("DISCRETE_EVENT_SIM.md").read_text(encoding="utf-8")
        rows = [
            row
            for row in docs.splitlines()
            if re.match(r"^\| \d+ \|", row)
        ]

        conf = Config()
        self.assertEqual(len(rows), len(conf.MODEM_PRESETS))

        for row, (preset_name, preset) in zip(rows, conf.MODEM_PRESETS.items()):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            _, display_name, bandwidth_khz, coding_rate, spreading_factor, _ = cells

            self.assertEqual(display_name.upper().replace(" ", "_"), preset_name)
            self.assertEqual(float(bandwidth_khz), preset["bw"] / 1000)
            self.assertEqual(coding_rate, f"4/{preset['cr']}")
            self.assertEqual(int(spreading_factor), preset["sf"])


if __name__ == "__main__":
    unittest.main()
