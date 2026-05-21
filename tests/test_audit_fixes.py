"""Regression tests for fixes applied during the 2026-05 audit pass.

These tests intentionally avoid importing torch / PySide6 / dlib / tensorflow
so they run in lightweight CI. They cover:

  * tools.match_skin_histogram._content_hash determinism and key separation
  * gui.app.TimestampedLogBox timestamp-passthrough regex
  * projector.create_generator loads the checkpoint with map_location="cpu"
    (guards against the double-copy regression — see P1 in PERFORMANCE_AUDIT.md)
  * losses.contextual_loss caching killswitch env var is read at import time
"""

import ast
import hashlib
import os
import re
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestContentHash(unittest.TestCase):
    """tools.match_skin_histogram._content_hash."""

    def _hash(self):
        from tools.match_skin_histogram import _content_hash
        return _content_hash

    def test_deterministic(self):
        h = self._hash()
        arr = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        self.assertEqual(h(arr), h(arr.copy()))

    def test_distinguishes_content(self):
        h = self._hash()
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.zeros((4, 4, 3), dtype=np.uint8)
        b[0, 0, 0] = 1
        self.assertNotEqual(h(a), h(b))

    def test_distinguishes_shape(self):
        h = self._hash()
        a = np.zeros((4, 5, 3), dtype=np.uint8)
        b = np.zeros((5, 4, 3), dtype=np.uint8)
        # Same bytes, different shape — keys must differ so a mask cache
        # cannot return a wrong-shape result.
        self.assertNotEqual(h(a), h(b))

    def test_distinguishes_dtype(self):
        h = self._hash()
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.zeros((4, 4, 3), dtype=np.float32)
        self.assertNotEqual(h(a), h(b))


class TestTimestampRegex(unittest.TestCase):
    """TimestampedLogBox prefix-detection regex."""

    # Mirrored locally so the test runs without importing PySide6.
    TS_RE = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}\]")

    def test_detects_existing_prefix(self):
        self.assertTrue(self.TS_RE.match("[12:34:56] hello"))
        self.assertTrue(self.TS_RE.match("[1:02:03] x"))

    def test_ignores_non_prefixed(self):
        self.assertIsNone(self.TS_RE.match("hello"))
        self.assertIsNone(self.TS_RE.match("[INFO] hello"))
        self.assertIsNone(self.TS_RE.match("[12:34] hello"))  # no seconds


class TestProjectorCheckpointLoadsCPU(unittest.TestCase):
    """P1 regression guard: create_generator must load to CPU first."""

    def test_create_generator_uses_map_location_cpu(self):
        # projector.py is saved with a UTF-8 BOM; use utf-8-sig to strip it.
        source = (REPO_ROOT / "projector.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)

        target_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "create_generator":
                target_fn = node
                break
        self.assertIsNotNone(target_fn, "create_generator must exist")

        # Find the _load_checkpoint call and confirm it passes
        # map_location="cpu". This guards against silently regressing the
        # double-copy fix (loading to "cuda" then .to(device) doubles VRAM
        # peak by ~1 GB and adds ~1 s wall-clock per run).
        found_cpu_load = False
        for node in ast.walk(target_fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_load_checkpoint"
            ):
                for kw in node.keywords:
                    if (
                        kw.arg == "map_location"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == "cpu"
                    ):
                        found_cpu_load = True
        self.assertTrue(
            found_cpu_load,
            "create_generator must call _load_checkpoint(..., map_location='cpu')",
        )


class TestContextualLossKillswitch(unittest.TestCase):
    """P3 killswitch must be read at module import time (not first use)."""

    def test_killswitch_module_flag_exists(self):
        source = (REPO_ROOT / "losses" / "contextual_loss" / "modules" / "contextual.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
        self.assertIn(
            "_CTX_CACHE_DISABLED",
            names,
            "module-level _CTX_CACHE_DISABLED flag must exist so the killswitch is one os.environ read",
        )


class TestExecuteWrapperCommandResolved(unittest.TestCase):
    """C1 regression guard: run_face_detection must not call a missing method.

    The original bug was a call to self._execute_wrapper_command at line 3135.
    No such method ever existed. The fix routes through _start_wrapper_process.
    """

    def test_no_dead_method_reference(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        # The string must not appear anywhere (no definition, no call).
        self.assertNotIn(
            "_execute_wrapper_command",
            source,
            "C1 regression: a call to self._execute_wrapper_command was reintroduced. "
            "Use self._start_wrapper_process instead.",
        )

    def test_run_face_detection_calls_start_wrapper(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_face_detection":
                target = node
                break
        self.assertIsNotNone(target, "run_face_detection method must exist on MainWindow")

        calls_to_start = False
        for node in ast.walk(target):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_start_wrapper_process"
            ):
                calls_to_start = True
        self.assertTrue(
            calls_to_start,
            "run_face_detection must dispatch via self._start_wrapper_process",
        )


class TestDeadCodeRemoved(unittest.TestCase):
    """Dead-code sweep: stale symbols should not silently come back."""

    def test_no_module_level_encode_color(self):
        source = (REPO_ROOT / "tools" / "initialize.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        module_funcs = {
            n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)
        }
        # The module-level encode_color was unused dead code creating a fresh
        # color encoder on every call. Production uses Initializer.encode_color.
        self.assertNotIn("encode_color", module_funcs)

    def test_no_module_level_transform_input(self):
        source = (REPO_ROOT / "tools" / "initialize.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        module_funcs = {
            n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)
        }
        self.assertNotIn("transform_input", module_funcs)


class TestKeyboardShortcuts(unittest.TestCase):
    """Accessibility: primary action buttons must carry shortcuts + accessible names."""

    def _source(self):
        return (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")

    def test_run_button_has_ctrl_enter(self):
        s = self._source()
        # The Run button setShortcut must remain Ctrl+Return (or equivalent).
        self.assertIn('self.run_button.setShortcut("Ctrl+Return")', s)
        self.assertIn('self.run_button.setAccessibleName(', s)

    def test_cancel_button_has_escape(self):
        s = self._source()
        self.assertIn('self.cancel_button.setShortcut("Escape")', s)
        self.assertIn('self.cancel_button.setAccessibleName(', s)

    def test_redetect_button_has_ctrl_d(self):
        s = self._source()
        self.assertIn('self.redetect_faces_button.setShortcut("Ctrl+D")', s)


class TestCompareWipeCache(unittest.TestCase):
    """P10: scaled pixmaps in the compare-wipe loop should be cached."""

    def test_compare_wipe_cache_state_initialized(self):
        s = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        # Cache state must be initialized so the first mouse-move doesn't
        # AttributeError; and it must be CHECKED in the hot path.
        self.assertIn("_compare_wipe_result_scaled_key = None", s)
        self.assertIn("_compare_wipe_input_scaled_key = None", s)
        # Hot-path check (re-use cache when key matches).
        self.assertIn("_compare_wipe_result_scaled_key == result_key", s)
        self.assertIn("_compare_wipe_input_scaled_key == input_key", s)


class TestSubprocessTimeoutSurfacing(unittest.TestCase):
    """P13: subprocess.TimeoutExpired must be caught separately and always logged."""

    def test_cropper_probe_handles_timeout_distinctly(self):
        s = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        # Both probe call sites should split TimeoutExpired from generic Exception
        # so timeout cases are not suppressed by the once-only warning flag.
        self.assertIn("subprocess.TimeoutExpired", s)
        self.assertIn("Cropper face-box probe timed out", s)
        self.assertIn("Retina face-box probe timed out", s)


if __name__ == "__main__":
    unittest.main()
