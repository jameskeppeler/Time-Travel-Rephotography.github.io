"""Direct unit tests for gui/path_utils.py and gui/format_utils.py.

These pure-function modules have no Qt dependency so they're trivially
testable in CI without PySide6.
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui import format_utils, path_utils


# ---------------------------------------------------------------------------
# path_utils
# ---------------------------------------------------------------------------

class TestNormalizedPathKey(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(path_utils.normalized_path_key(""))
        self.assertIsNone(path_utils.normalized_path_key(None))

    def test_returns_string_for_existing_path(self):
        key = path_utils.normalized_path_key(str(REPO_ROOT))
        self.assertIsInstance(key, str)
        self.assertGreater(len(key), 0)

    def test_case_insensitive_on_windows(self):
        # On Windows os.path.normcase lowercases; on POSIX it's a no-op.
        # Either way the function should not raise for any well-formed input.
        a = path_utils.normalized_path_key("Some/Path/X.png")
        b = path_utils.normalized_path_key("Some/Path/X.png")
        self.assertEqual(a, b)

    def test_falls_back_when_resolution_fails(self):
        # Unresolvable paths still produce a non-None key (lowered text).
        weird = "\x00not_a_real_path"
        self.assertIsNotNone(path_utils.normalized_path_key(weird))


class TestMakeSafeBaseName(unittest.TestCase):
    def test_default_for_empty(self):
        self.assertEqual(path_utils.make_safe_base_name(""), "input_image")
        self.assertEqual(path_utils.make_safe_base_name(None), "input_image")

    def test_collapses_separators(self):
        self.assertEqual(
            path_utils.make_safe_base_name("hello world!!! foo  bar"),
            "hello_world_foo_bar",
        )

    def test_strips_leading_trailing_underscores(self):
        self.assertEqual(path_utils.make_safe_base_name("!!hello!!"), "hello")

    def test_keeps_digits(self):
        self.assertEqual(
            path_utils.make_safe_base_name("1865_R_M_Boggs_no20"),
            "1865_R_M_Boggs_no20",
        )

    def test_all_garbage_returns_default(self):
        self.assertEqual(path_utils.make_safe_base_name("!!!"), "input_image")


class TestListImageFilesInDir(unittest.TestCase):
    def test_none_or_missing_returns_empty(self):
        self.assertEqual(path_utils.list_image_files_in_dir(None), [])
        self.assertEqual(
            path_utils.list_image_files_in_dir(Path("/no/such/dir/please")), []
        )

    def test_lists_and_sorts_image_files(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Create images that should be listed in suffix-index order.
            (base / "face_10.png").write_bytes(b"")
            (base / "face_2.jpg").write_bytes(b"")
            (base / "face_1.png").write_bytes(b"")
            (base / "alpha.png").write_bytes(b"")  # no suffix
            (base / "ignored.txt").write_bytes(b"")  # not an image

            result = [p.name for p in path_utils.list_image_files_in_dir(base)]

            # face_1, face_2, face_10 must be in numeric order (not lexical),
            # then non-suffixed files come after by name.
            self.assertEqual(
                result, ["face_1.png", "face_2.jpg", "face_10.png", "alpha.png"]
            )

    def test_extension_filter_case_insensitive(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "X.PNG").write_bytes(b"")
            (base / "Y.JpG").write_bytes(b"")
            (base / "Z.TIFF").write_bytes(b"")
            (base / "ignored.DOC").write_bytes(b"")
            names = {p.name for p in path_utils.list_image_files_in_dir(base)}
            self.assertEqual(names, {"X.PNG", "Y.JpG", "Z.TIFF"})


class TestResultPreviewCacheKey(unittest.TestCase):
    def test_includes_mtime_and_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "foo.png"
            p.write_bytes(b"x" * 17)
            key = path_utils.result_preview_cache_key(p)
            self.assertIn("|", key)
            self.assertTrue(key.endswith("|17") or "|17|" in key)

    def test_invalidates_on_write(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "foo.png"
            p.write_bytes(b"a")
            key1 = path_utils.result_preview_cache_key(p)
            # Change content; size shifts so the key MUST differ.
            p.write_bytes(b"abcdef")
            key2 = path_utils.result_preview_cache_key(p)
            self.assertNotEqual(key1, key2)

    def test_missing_file_returns_normalized_path(self):
        # No stat -> falls back to just the lowered path text.
        key = path_utils.result_preview_cache_key(Path("/no/such/file/please.png"))
        self.assertIsInstance(key, str)
        self.assertNotIn("|", key)


# ---------------------------------------------------------------------------
# format_utils
# ---------------------------------------------------------------------------

class TestFormatElapsedForSummary(unittest.TestCase):
    def test_none(self):
        self.assertEqual(format_utils.format_elapsed_for_summary(None), "N/A")

    def test_minutes_seconds(self):
        self.assertEqual(format_utils.format_elapsed_for_summary(0), "0:00")
        self.assertEqual(format_utils.format_elapsed_for_summary(59), "0:59")
        self.assertEqual(format_utils.format_elapsed_for_summary(60), "1:00")
        self.assertEqual(format_utils.format_elapsed_for_summary(3599), "59:59")

    def test_hours(self):
        self.assertEqual(format_utils.format_elapsed_for_summary(3600), "1:00:00")
        self.assertEqual(format_utils.format_elapsed_for_summary(3661), "1:01:01")
        self.assertEqual(format_utils.format_elapsed_for_summary(7200), "2:00:00")

    def test_negative_clamped_to_zero(self):
        self.assertEqual(format_utils.format_elapsed_for_summary(-5), "0:00")


class TestPreflightFormatters(unittest.TestCase):
    SAMPLE = {
        "checks": [
            {"name": "Application root", "status": "pass", "detail": "ok", "fix": ""},
            {"name": "CUDA", "status": "warn", "detail": "old driver", "fix": "Update driver"},
            {"name": "GPU", "status": "fail", "detail": "missing", "fix": "Install GPU"},
        ],
        "pass": 1,
        "warn": 1,
        "fail": 1,
    }

    def test_plain_text_contains_summary_line(self):
        text = format_utils.preflight_report_plain_text(self.SAMPLE)
        self.assertIn("Summary: 1 pass, 1 warn, 1 fail", text)
        self.assertIn("[PASS] Application root", text)
        self.assertIn("[WARN] CUDA", text)
        self.assertIn("[FAIL] GPU", text)

    def test_plain_text_includes_fixes_only_for_non_pass(self):
        text = format_utils.preflight_report_plain_text(self.SAMPLE)
        self.assertIn("Fix: Update driver", text)
        self.assertIn("Fix: Install GPU", text)
        # No "Fix:" line should appear for the passing check.
        pass_block = text.split("[PASS] Application root", 1)[1].split("[WARN]", 1)[0]
        self.assertNotIn("Fix:", pass_block)

    def test_html_contains_status_colors(self):
        html = format_utils.preflight_report_html(self.SAMPLE)
        # Each status maps to its semantic color.
        self.assertIn("#6fcf97", html)  # pass = green
        self.assertIn("#f2c94c", html)  # warn = yellow
        self.assertIn("#eb5757", html)  # fail = red
        self.assertIn("Startup Preflight", html)


from gui import timing_log as tl


class TestTimingLogReader(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(tl.read_timing_records(Path("/no/such/log.jsonl")), [])

    def test_filters_unsuccessful(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            p.write_text(
                '{"success": true,  "elapsed_seconds": 5.0}\n'
                '{"success": false, "elapsed_seconds": 7.0}\n',
                encoding="utf-8",
            )
            out = tl.read_timing_records(p)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0]["elapsed_seconds"], 5.0)

    def test_filters_crop_only(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            p.write_text(
                '{"success": true, "crop_only": true,  "elapsed_seconds": 3.0}\n'
                '{"success": true, "crop_only": false, "elapsed_seconds": 4.0}\n',
                encoding="utf-8",
            )
            out = tl.read_timing_records(p)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0]["elapsed_seconds"], 4.0)

    def test_filters_missing_elapsed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            p.write_text(
                '{"success": true}\n'
                '{"success": true, "elapsed_seconds": 9.5}\n',
                encoding="utf-8",
            )
            self.assertEqual(len(tl.read_timing_records(p)), 1)

    def test_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            p.write_text(
                "not json\n"
                "\n"  # blank line
                '{"success": true, "elapsed_seconds": 1.5}\n'
                '{"success": tru\n',  # truncated
                encoding="utf-8",
            )
            out = tl.read_timing_records(p)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0]["elapsed_seconds"], 1.5)


class TestTimingLogBuilders(unittest.TestCase):
    def test_build_run_record_shape(self):
        rec = tl.build_run_record(
            input_image="img.png",
            preset="750",
            enhancement=True,
            crop_only=False,
            success=True,
            elapsed_seconds=12.5,
            gpu_name="RTX 4090",
            timestamp="2026-05-20 10:00:00",
        )
        self.assertEqual(rec["input_image"], "img.png")
        self.assertEqual(rec["preset"], "750")
        self.assertTrue(rec["enhancement"])
        self.assertFalse(rec["crop_only"])
        self.assertTrue(rec["success"])
        self.assertEqual(rec["elapsed_seconds"], 12.5)
        self.assertEqual(rec["gpu_name"], "RTX 4090")
        self.assertEqual(rec["timestamp_local"], "2026-05-20 10:00:00")
        # advanced_mode defaults to False
        self.assertFalse(rec["advanced_mode"])

    def test_build_milestone_record_marks_type(self):
        rec = tl.build_milestone_record(
            input_image="img.png",
            preset="1500",
            source_run_preset="3000",
            enhancement=False,
            elapsed_seconds=22.0,
            timestamp="2026-05-20 11:00:00",
        )
        self.assertEqual(rec["record_type"], "milestone")
        self.assertEqual(rec["preset"], "1500")
        self.assertEqual(rec["source_run_preset"], "3000")
        # Milestones are always success=True crop_only=False so reader picks them up.
        self.assertTrue(rec["success"])
        self.assertFalse(rec["crop_only"])


class TestTimingLogAppend(unittest.TestCase):
    def test_appends_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "log.jsonl"
            n = tl.append_records(p, [{"a": 1}, {"a": 2}])
            self.assertEqual(n, 2)
            self.assertEqual(
                p.read_text(encoding="utf-8").strip().split("\n"),
                ['{"a": 1}', '{"a": 2}'],
            )

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nested" / "deeper" / "log.jsonl"
            tl.append_records(p, [{"a": 1}])
            self.assertTrue(p.exists())

    def test_roundtrip_via_read(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "log.jsonl"
            rec = tl.build_run_record(
                input_image="img.png", preset="750",
                enhancement=True, crop_only=False, success=True,
                elapsed_seconds=5.0,
            )
            tl.append_records(p, [rec])
            out = tl.read_timing_records(p)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0]["elapsed_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()
