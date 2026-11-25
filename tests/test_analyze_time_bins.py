"""Data-only bin occupancy regression tests (no TGB or GPU dependency)."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_time_bins", ROOT / "scripts" / "analyze_time_bins.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeTimeBinsTest(unittest.TestCase):
    def test_bin_widths_match_training_scripts(self):
        for name in ("tgn", "tgat", "graphmixer"):
            path = ROOT / "models" / f"{name}_time_scale" / "train.py"
            spec = importlib.util.spec_from_file_location(f"{name}_time_scale_train", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for gran, scale in module.SCALE_MAP.items():
                self.assertEqual(1.0 / scale, MODULE.WIDTHS[gran])

    def test_strict_cutoff_padding_and_two_directions(self):
        # Edges at equal time must not see each other; histories have at most K.
        stream = MODULE.accumulate_ages(
            src=[1, 1, 1, 1], dst=[2, 3, 4, 5],
            times=[0, 3600, 3600, 7200], splits=["train", "train", "val", "test"],
            num_neighbors=2)
        self.assertEqual(stream["event_counts"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(list(stream["neighbor_ages"]["train"]), [3600.0])
        self.assertEqual(list(stream["neighbor_ages"]["val"]), [3600.0])
        # At 7200, source history is the last two 3600 interactions; dst is new.
        self.assertEqual(list(stream["neighbor_ages"]["test"]), [3600.0, 3600.0])
        self.assertEqual(stream["neighbor_slots"]["test"], 4)
        self.assertEqual(len(stream["message_proxy_ages"]["test"]), 2)

    def test_bins_are_discrete_and_count_only_valid_ages(self):
        ages = [0, 1, 3599, 3600, 3601, 7200, 36000]
        report = MODULE.summarize_ages(ages, {"hour": 3600})
        self.assertEqual(report["hour"]["counts"], {
            "bin0": 3, "bin1": 2, "bin2_to_9": 1, "bin10_plus": 1})
        self.assertEqual(report["hour"]["distinct_occupied_bins"], 4)
        self.assertAlmostEqual(report["hour"]["fraction"]["bin0"], 3 / 7)
        with self.assertRaisesRegex(ValueError, "negative elapsed"):
            MODULE.summarize_ages([-1])

    def test_requires_sorted_stream(self):
        with self.assertRaisesRegex(ValueError, "sorted"):
            MODULE.accumulate_ages([1, 1], [2, 3], [10, 9], ["train", "train"])


if __name__ == "__main__":
    unittest.main()
