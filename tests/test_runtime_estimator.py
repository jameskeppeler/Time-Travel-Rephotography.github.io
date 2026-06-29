"""Unit tests for gui/runtime_estimator.py.

Exercises the four-tier fallback chain in estimate_runtime_minutes and the
small GPU/RAM heuristic in compute_runtime_scale, both pure functions, no
Qt or MainWindow dependency.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui import runtime_estimator as re_mod


def rec(preset, secs, enh=False, gpu="RTX 3060", record_type="full_run"):
    return {
        "preset": preset,
        "elapsed_seconds": secs,
        "enhancement": enh,
        "gpu_name": gpu,
        "record_type": record_type,
        "success": True,
        "crop_only": False,
    }


# ---------------------------------------------------------------------------
# estimate_runtime_minutes
# ---------------------------------------------------------------------------

class TestEstimateNoRecords(unittest.TestCase):
    def test_empty_records_returns_none(self):
        minutes, note = re_mod.estimate_runtime_minutes([], "RTX 3060", False, 750)
        self.assertIsNone(minutes)
        self.assertIn("No local timing history", note)

    def test_all_unparseable_records_returns_none(self):
        records = [{"preset": "junk", "elapsed_seconds": 5.0}]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertIsNone(minutes)
        self.assertIn("could not be parsed", note)


class TestEstimateExactMatch(unittest.TestCase):
    def test_exact_same_gpu_same_enh_wins(self):
        records = [
            rec(750, 60.0, enh=False, gpu="RTX 3060"),  # match
            rec(750, 600.0, enh=False, gpu="RTX 4090"),  # different GPU, ignored
            rec(750, 999.0, enh=True, gpu="RTX 3060"),   # different enh, ignored
        ]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 1.0)  # 60s / 60
        self.assertIn("same-GPU", note)

    def test_legacy_test_preset_normalized_to_750(self):
        records = [rec("test", 90.0, enh=False, gpu="RTX 3060")]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 1.5)
        self.assertIn("same-GPU", note)

    def test_falls_back_to_any_gpu_when_no_same_gpu_match(self):
        records = [rec(750, 120.0, enh=False, gpu="RTX 4090")]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 2.0)
        self.assertNotIn("same-GPU", note)
        self.assertIn("at this preset", note)

    def test_uses_median_across_multiple_matches(self):
        records = [
            rec(750, 60.0, enh=False, gpu="RTX 3060"),
            rec(750, 120.0, enh=False, gpu="RTX 3060"),
            rec(750, 180.0, enh=False, gpu="RTX 3060"),
        ]
        minutes, _ = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 2.0)  # median 120 / 60


class TestEstimateOppositeInference(unittest.TestCase):
    def test_infers_enh_on_from_enh_off_plus_overhead(self):
        # No enh-on record for preset 750. But enh-off=60s and paired data
        # at preset 1500 shows overhead=30s. Estimate for enh-on should be
        # 60 + 30 = 90s = 1.5 min.
        records = [
            rec(750, 60.0, enh=False, gpu="RTX 3060"),
            rec(1500, 120.0, enh=False, gpu="RTX 3060"),
            rec(1500, 150.0, enh=True, gpu="RTX 3060"),  # overhead at 1500 = 30s
        ]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", True, 750)
        self.assertIsNotNone(minutes)
        self.assertAlmostEqual(minutes, 1.5, places=2)
        self.assertIn("Inferred", note)

    def test_infers_enh_off_from_enh_on_minus_overhead(self):
        records = [
            rec(750, 90.0, enh=True, gpu="RTX 3060"),    # only enh-on at 750
            rec(1500, 120.0, enh=False, gpu="RTX 3060"),
            rec(1500, 150.0, enh=True, gpu="RTX 3060"),  # overhead at 1500 = 30s
        ]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertIsNotNone(minutes)
        # 90 - 30 = 60s = 1.0 min
        self.assertAlmostEqual(minutes, 1.0, places=2)
        self.assertIn("Inferred", note)


class TestEstimateInterpolation(unittest.TestCase):
    def test_log_linear_interpolation_when_no_exact_preset(self):
        # Two known base points; the function log-linearly interpolates.
        records = [
            rec(375, 30.0, enh=False, gpu="RTX 3060"),
            rec(3000, 240.0, enh=False, gpu="RTX 3060"),
        ]
        minutes, note = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        # Log-linear between (375, 30) and (3000, 240). Pure formula check:
        import math
        x0, y0 = 375, 30
        x1, y1 = 3000, 240
        lxp = math.log(750)
        t = (lxp - math.log(x0)) / (math.log(x1) - math.log(x0))
        expected = math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0))) / 60.0
        self.assertAlmostEqual(minutes, expected, places=5)
        self.assertIn("Interpolated", note)

    def test_extrapolates_below_lowest_preset(self):
        records = [
            rec(750, 60.0, enh=False, gpu="RTX 3060"),
            rec(3000, 240.0, enh=False, gpu="RTX 3060"),
        ]
        minutes, _ = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 375)
        # Should still return a number (log-linear extrapolation in [750, 3000]
        # window).
        self.assertIsNotNone(minutes)
        self.assertGreater(minutes, 0)


class TestEstimateEdgeCases(unittest.TestCase):
    def test_zero_and_negative_secs_are_skipped(self):
        records = [
            rec(750, 0.0, enh=False, gpu="RTX 3060"),
            rec(750, -5.0, enh=False, gpu="RTX 3060"),
            rec(750, 60.0, enh=False, gpu="RTX 3060"),
        ]
        minutes, _ = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 1.0)

    def test_unparseable_secs_skipped(self):
        records = [
            {"preset": "750", "elapsed_seconds": "not-a-float", "enhancement": False,
             "gpu_name": "RTX 3060", "success": True, "crop_only": False},
            rec(750, 60.0, enh=False, gpu="RTX 3060"),
        ]
        minutes, _ = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 1.0)

    def test_returns_minutes_not_seconds(self):
        # A regression-prone unit conversion — keep the unit explicit.
        records = [rec(750, 120.0, enh=False, gpu="RTX 3060")]
        minutes, _ = re_mod.estimate_runtime_minutes(records, "RTX 3060", False, 750)
        self.assertAlmostEqual(minutes, 2.0)
        self.assertNotAlmostEqual(minutes, 120.0)


# ---------------------------------------------------------------------------
# compute_runtime_scale
# ---------------------------------------------------------------------------

class TestComputeRuntimeScale(unittest.TestCase):
    def test_baseline_3060(self):
        scale, _ = re_mod.compute_runtime_scale({"gpu_name": "NVIDIA GeForce RTX 3060 Laptop"})
        self.assertAlmostEqual(scale, 1.0)

    def test_3070_is_faster(self):
        scale, note = re_mod.compute_runtime_scale({"gpu_name": "NVIDIA GeForce RTX 3070"})
        self.assertLess(scale, 1.0)
        self.assertIn("3070", note)

    def test_3050_is_slower(self):
        scale, note = re_mod.compute_runtime_scale({"gpu_name": "NVIDIA GeForce RTX 3050"})
        self.assertGreater(scale, 1.0)
        self.assertIn("3050", note)

    def test_4090_dramatic_speedup(self):
        scale, _ = re_mod.compute_runtime_scale({"gpu_name": "RTX 4090"})
        self.assertLessEqual(scale, 0.5)

    def test_unknown_gpu_does_not_crash(self):
        scale, note = re_mod.compute_runtime_scale({"gpu_name": "Intel Arc A770"})
        self.assertGreater(scale, 0)
        self.assertIn("unknown GPU", note)

    def test_low_ram_penalty(self):
        s_low, _ = re_mod.compute_runtime_scale({"gpu_name": "RTX 3060", "ram_gb": 8})
        s_base, _ = re_mod.compute_runtime_scale({"gpu_name": "RTX 3060", "ram_gb": 16})
        self.assertGreater(s_low, s_base)

    def test_high_ram_modest_speedup(self):
        s_high, _ = re_mod.compute_runtime_scale({"gpu_name": "RTX 3060", "ram_gb": 64})
        s_base, _ = re_mod.compute_runtime_scale({"gpu_name": "RTX 3060", "ram_gb": 16})
        self.assertLess(s_high, s_base)

    def test_missing_gpu_field_defaults_to_unknown(self):
        scale, note = re_mod.compute_runtime_scale({})
        self.assertGreater(scale, 0)
        self.assertIn("unknown", note.lower())


if __name__ == "__main__":
    unittest.main()
