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


class TestLogLevelClassification(unittest.TestCase):
    """Mirror TimestampedLogBox classification regexes locally so the test
    runs without PySide6. If these constants drift in gui/app.py without
    matching changes here, the test_gui_smoke.py colorization tests will
    catch it under offscreen Qt."""

    ERR = re.compile(r"(?i)\b(error|failed|failure|exception|traceback|cannot|crashed)\b|could not")
    WARN = re.compile(r"(?i)\b(warning|warn|timed out|deprecated|fallback)\b")

    def _classify(self, s):
        if self.ERR.search(s):
            return "error"
        if self.WARN.search(s):
            return "warn"
        return "info"

    def test_error_keywords_detected(self):
        self.assertEqual(self._classify("Cropper face-box probe failed"), "error")
        self.assertEqual(self._classify("Traceback (most recent call last):"), "error")
        self.assertEqual(self._classify("Could not load input image."), "error")
        self.assertEqual(self._classify("Exception in worker"), "error")

    def test_warning_keywords_detected(self):
        self.assertEqual(self._classify("WARNING: face parsing failed; skipping"), "error")  # "failed" wins
        self.assertEqual(self._classify("Probe timed out (40s)"), "warn")
        self.assertEqual(self._classify("Warning: 3 collisions"), "warn")
        self.assertEqual(self._classify("Fallback to Haar"), "warn")

    def test_info_default(self):
        self.assertEqual(self._classify("GUI loaded successfully."), "info")
        self.assertEqual(self._classify("Run button clicked."), "info")
        self.assertEqual(self._classify("Detected 5 faces."), "info")

    def test_error_priority_over_warning(self):
        # If both cues appear, error wins so the user sees red, not yellow.
        self.assertEqual(self._classify("Warning: backend failed"), "error")

    def test_html_escaping_intent(self):
        # We don't construct the widget here, but verify the in-source
        # escaping calls cover the dangerous characters. TimestampedLogBox
        # was extracted to gui/widgets.py in the Sprint-4 split.
        source = (REPO_ROOT / "gui" / "widgets.py").read_text(encoding="utf-8")
        # The colorization path must escape &, <, > before wrapping in <span>.
        self.assertIn('.replace("&", "&amp;")', source)
        self.assertIn('.replace("<", "&lt;")', source)
        self.assertIn('.replace(">", "&gt;")', source)


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


class TestModuleSplit(unittest.TestCase):
    """Sprint-4 first slice: widget classes and worker helpers must live in
    gui/widgets.py and gui/app.py must import them, not define them."""

    WIDGET_NAMES = [
        "TimestampedLogBox",
        "InstantToolButton",
        "NoScrollComboBox",
        "NoScrollDoubleSpinBox",
        "NoScrollSpinBox",
        "NoScrollSlider",
        "FilmstripContainerWidget",
        "FaceStripToolButton",
        "InputDropLabel",
        "ResultPreviewLabel",
        "InputDetectOverlay",
        "_PixmapLoaderSignals",
        "_PixmapLoader",
        "_PreflightSignals",
        "_PreflightRunner",
    ]

    def test_widgets_module_exists(self):
        widgets_path = REPO_ROOT / "gui" / "widgets.py"
        self.assertTrue(widgets_path.exists(), "gui/widgets.py must exist after the split")

    def test_widget_classes_defined_in_widgets(self):
        source = (REPO_ROOT / "gui" / "widgets.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defined = {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)}
        for name in self.WIDGET_NAMES:
            self.assertIn(name, defined, f"{name} must be defined in gui/widgets.py")

    def test_widget_classes_not_redefined_in_app(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defined = {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)}
        for name in self.WIDGET_NAMES:
            self.assertNotIn(
                name,
                defined,
                f"{name} must not be redefined in gui/app.py (extracted to gui/widgets.py)",
            )

    def test_app_imports_widgets(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("from gui.widgets import", source)


class TestConstantsModule(unittest.TestCase):
    """gui/constants.py owns the source-of-truth defaults and regexes."""

    EXPECTED_NAMES = [
        "DEFAULT_BASIC_ITER_VALUES",
        "DEFAULT_ADVANCED_ITER_VALUES",
        "DEFAULT_ITERATION",
        "QUALITY_PRESET_LABELS",
        "PHOTO_TYPE_DATE_HINTS",
        "WIDE_LAYOUT_MIN_WIDTH",
        "DEFAULT_FACE_FACTOR",
        "DEFAULT_GFPGAN_BLEND",
        "DEFAULT_DET_THRESHOLD",
        "DEFAULT_GAUSSIAN",
        "DEFAULT_NOISE_REGULARIZE",
        "DEFAULT_LR",
        "DEFAULT_CAMERA_LR",
        "DEFAULT_MIX_LAYER_START",
        "DEFAULT_MIX_LAYER_END",
        "IMAGE_EXTENSIONS",
        "CROPPED_FACE_COUNT_RE",
        "ITER_PROGRESS_RE",
        "FACE_SUFFIX_INDEX_RE",
        "RETINA_FACE_BOX_RE",
    ]

    def test_constants_module_importable_without_qt(self):
        # The module must not depend on PySide6 so it can be used in unit
        # tests + non-GUI contexts.
        import importlib
        mod = importlib.import_module("gui.constants")
        for name in self.EXPECTED_NAMES:
            self.assertTrue(hasattr(mod, name), f"gui.constants missing {name}")

    def test_no_constants_redefined_in_app(self):
        # Constants must not be re-declared in gui/app.py (drift hazard).
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_assignments = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        module_assignments.add(tgt.id)
        for name in self.EXPECTED_NAMES:
            self.assertNotIn(
                name,
                module_assignments,
                f"{name} must not be reassigned in gui/app.py (defined in gui/constants.py)",
            )


class TestFaceStripMixin(unittest.TestCase):
    """Sprint-4 architectural slice: FaceStripMixin holds 31 face-strip
    methods that used to live on MainWindow. MRO must keep them callable
    through the existing self.<method>() call sites."""

    EXPECTED_METHODS = [
        "render_face_preview_strip",
        "reset_face_preview_state",
        "initialize_face_preview_entries",
        "_compute_face_strip_render_signature",
        "_sync_face_preview_crop_paths",
        "_make_face_thumb_icon",
        "set_hover_face_preview_index",
        "clear_hover_face_preview_index",
        "_cursor_face_preview_index",
        "select_face_preview",
        "set_all_faces_selected",
        "get_selected_face_indices",
        "get_selected_face_count_text",
        "get_runtime_face_multiplier",
        "get_selected_face_preview_path",
        "get_face_preview_path",
        "_get_face_crop_path",
        "_find_face_index_for_crop_path",
        "_find_face_index_for_result_path",
        "mark_face_running_from_crop_path",
        "mark_face_done_from_result_path",
        "mark_face_failed_from_crop_name",
        "reconcile_face_preview_results",
        "clear_face_preview_strip_layout",
        "update_run_button_for_quick_face_hint",
        "handle_face_auto_follow_toggled",
        "_rehost_face_preview_panel",
        "_configure_face_preview_panel_for_mode",
        "_get_face_preview_header_min_width",
        "_get_face_strip_card_width",
        "_get_focused_face_preview_index",
    ]

    def test_mixin_module_exists(self):
        self.assertTrue((REPO_ROOT / "gui" / "face_strip.py").exists())

    def test_mixin_class_defined(self):
        source = (REPO_ROOT / "gui" / "face_strip.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)}
        self.assertIn("FaceStripMixin", classes)

    def test_methods_live_on_mixin(self):
        source = (REPO_ROOT / "gui" / "face_strip.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        methods = set()
        for cls in ast.iter_child_nodes(tree):
            if isinstance(cls, ast.ClassDef) and cls.name == "FaceStripMixin":
                for item in cls.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
        missing = [m for m in self.EXPECTED_METHODS if m not in methods]
        self.assertFalse(missing, f"Mixin missing: {missing}")

    def test_methods_NOT_redefined_in_main_window(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef) and cls.name == "MainWindow":
                method_names = {
                    item.name
                    for item in cls.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                duplicates = [m for m in self.EXPECTED_METHODS if m in method_names]
                self.assertFalse(
                    duplicates,
                    f"These mixin methods are also defined on MainWindow "
                    f"(duplicate -- mixin version will be shadowed): {duplicates}",
                )

    def test_main_window_inherits_mixin(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        # The class header must reference FaceStripMixin in its bases.
        self.assertRegex(
            source,
            r"class\s+MainWindow\s*\([^)]*FaceStripMixin[^)]*\)\s*:",
        )


class TestDialogsModule(unittest.TestCase):
    """gui/dialogs.py hosts AdvancedSettingsDialog after the second slice."""

    def test_module_exists(self):
        self.assertTrue((REPO_ROOT / "gui" / "dialogs.py").exists())

    def test_dialog_defined_in_dialogs(self):
        source = (REPO_ROOT / "gui" / "dialogs.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defined = {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)}
        self.assertIn("AdvancedSettingsDialog", defined)

    def test_dialog_not_redefined_in_app(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defined = {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)}
        self.assertNotIn("AdvancedSettingsDialog", defined)

    def test_app_imports_dialog(self):
        source = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("from gui.dialogs import AdvancedSettingsDialog", source)


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


class TestPauseAckMechanism(unittest.TestCase):
    """The pause-ACK warning timer must be armed on request and cleared on
    every termination path so it never nags after the run is over."""

    def _source(self):
        return (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")

    def test_helpers_defined(self):
        s = self._source()
        # The timer-management helpers must exist.
        self.assertIn("def _arm_pause_ack_warning(self):", s)
        self.assertIn("def _cancel_pause_ack_warning(self):", s)
        self.assertIn("def _on_pause_ack_warning_timeout(self):", s)

    def test_armed_on_pause_request(self):
        s = self._source()
        # Pause request site must arm the warning.
        idx = s.find('"Status: Pause requested..."')
        self.assertGreater(idx, 0, "Pause request status line must exist")
        # Within the next ~800 characters of source, _arm_pause_ack_warning
        # must be called.
        window = s[idx:idx + 800]
        self.assertIn("_arm_pause_ack_warning()", window)

    def test_cancelled_on_ack_resume_and_termination(self):
        s = self._source()
        # Cancelled on stdout marker "=== Pause requested ===" arrival.
        ack_idx = s.find('s.startswith("=== Pause requested ===")')
        self.assertGreater(ack_idx, 0)
        self.assertIn("_cancel_pause_ack_warning()", s[ack_idx:ack_idx + 400])
        # Cancelled in process_finished, process_error, cancel_run.
        for fn_name in ("def process_finished", "def process_error", "def cancel_run"):
            fn_idx = s.find(fn_name)
            self.assertGreater(fn_idx, 0, f"{fn_name} must exist")
            self.assertIn(
                "_cancel_pause_ack_warning()",
                s[fn_idx:fn_idx + 1200],
                f"{fn_name} must cancel the pause-ack warning timer",
            )


class TestCompareWipeCacheClearedOnReset(unittest.TestCase):
    """Compare-wipe scaled caches must be cleared when the user loads a new
    input image; otherwise stale scaled pixmaps from the previous result
    could leak through during a transition."""

    def test_reset_clears_compare_wipe_cache(self):
        s = (REPO_ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        idx = s.find("def _reset_main_window_for_new_input")
        self.assertGreater(idx, 0)
        body = s[idx:idx + 1800]
        self.assertIn("_compare_wipe_result_scaled_key = None", body)
        self.assertIn("_compare_wipe_input_scaled_key = None", body)


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
