"""Direct-instantiation tests for the controllers.

The whole point of the mixin -> controller migration is that each
controller can now be constructed with a *stub* window object, without
needing a real QMainWindow or even Qt to be available. That property
was promised by the architecture; this file proves it.

These tests skip without PySide6 (since some controllers import Qt
symbols), but they do NOT need an offscreen QApplication or any of the
real GUI plumbing.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import PySide6  # noqa: F401
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class _StubWindow:
    """Tiny stand-in for MainWindow.

    Provides only the attributes/methods the controllers reach for via
    self.window.* . Tests can extend instances ad hoc with setattr.
    """

    def __init__(self):
        # Common attrs the controllers touch via __getattr__ forwarding.
        self.repo_root = REPO_ROOT
        self.log_box = _RecordingLogBox()
        self.status_label = _RecordingLabel()

    def resolve_resource_path(self, rel):
        return Path(self.repo_root) / rel

    def update_view_menu_actions(self):
        # PreflightController calls this after storing a summary text.
        self._update_view_menu_actions_calls = (
            getattr(self, "_update_view_menu_actions_calls", 0) + 1
        )


class _RecordingLogBox:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _RecordingLabel:
    def __init__(self):
        self._text = ""

    def setText(self, t):
        self._text = t

    def text(self):
        return self._text


# ---------------------------------------------------------------------------
# PreflightController
# ---------------------------------------------------------------------------

@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 not installed; skipping")
class TestPreflightControllerDirect(unittest.TestCase):
    def setUp(self):
        from gui.preflight import PreflightController
        self.stub = _StubWindow()
        # PreflightController also needs results_root_edit.text() for
        # collect_report and run_startup_preflight; stub it.
        class _Edit:
            def __init__(self, t):
                self._t = t

            def text(self):
                return self._t

        self.stub.results_root_edit = _Edit("")
        self.ctrl = PreflightController(self.stub)

    def test_initial_state(self):
        self.assertFalse(self.ctrl._running)
        self.assertIsNone(self.ctrl.last_preflight_report)
        self.assertEqual(self.ctrl.last_run_summary_text, "")
        self.assertIsNone(self.ctrl._hardware_info_cache)

    def test_format_elapsed_no_window_needed(self):
        # Pure delegation to format_utils, but reachable on the controller.
        self.assertEqual(self.ctrl.format_elapsed_for_summary(0), "0:00")
        self.assertEqual(self.ctrl.format_elapsed_for_summary(3661), "1:01:01")
        self.assertEqual(self.ctrl.format_elapsed_for_summary(None), "N/A")

    def test_report_plain_text_no_window_needed(self):
        report = {
            "checks": [
                {"name": "X", "status": "pass", "detail": "ok", "fix": ""},
                {"name": "Y", "status": "fail", "detail": "boom", "fix": "do thing"},
            ],
            "pass": 1, "warn": 0, "fail": 1,
        }
        text = self.ctrl.report_plain_text(report)
        self.assertIn("[PASS] X", text)
        self.assertIn("[FAIL] Y", text)
        self.assertIn("Fix: do thing", text)

    def test_store_run_summary_requires_context(self):
        # No current_run_summary_context on the stub -> returns None.
        self.stub.current_run_summary_context = None
        out = self.ctrl.store_run_summary_text(
            success=True, exit_code=0, elapsed_seconds=10,
        )
        self.assertIsNone(out)

    def test_store_run_summary_with_context(self):
        self.stub.current_run_summary_context = {
            "input_image": "img.png",
            "results_root": "results/",
            "quality": "750",
            "photo_type": "Cabinet card",
            "approx_date": "1890",
            "spectral_sensitivity": "G",
            "selected_faces": "2 of 3",
            "enhancement_enabled": True,
            "crop_only": False,
        }
        out = self.ctrl.store_run_summary_text(
            success=True, exit_code=0, elapsed_seconds=125.0,
        )
        self.assertIn("Run Summary", out)
        self.assertIn("Status: Success", out)
        self.assertIn("Elapsed: 2:05", out)
        self.assertIn("img.png", out)
        self.assertEqual(self.ctrl.last_run_summary_text, out)
        # The controller pinged the stub's menu hook.
        self.assertEqual(self.stub._update_view_menu_actions_calls, 1)


# ---------------------------------------------------------------------------
# FaceStripController
# ---------------------------------------------------------------------------

@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 not installed; skipping")
class TestFaceStripControllerDirect(unittest.TestCase):
    def setUp(self):
        from gui.face_strip import FaceStripController
        self.stub = _StubWindow()
        self.ctrl = FaceStripController(self.stub)

    def test_initial_state(self):
        self.assertEqual(self.ctrl.face_preview_entries, [])
        self.assertIsNone(self.ctrl.active_face_preview_index)
        self.assertIsNone(self.ctrl.selected_face_preview_index)
        self.assertFalse(self.ctrl._no_faces_detected)
        self.assertFalse(self.ctrl._user_inspecting_completed_face)
        self.assertEqual(self.ctrl.face_preview_thumb_icon_cache, {})

    def test_state_writes_land_on_controller_not_window(self):
        # No __setattr__ proxy any more.
        self.ctrl.face_preview_entries = [{"index": 0, "selected": True}]
        self.assertEqual(len(self.ctrl.face_preview_entries), 1)
        # The stub window MUST NOT have received a face_preview_entries
        # attribute via setattr forwarding.
        self.assertFalse(hasattr(self.stub, "face_preview_entries"))

    def test_window_reads_fall_through_via_getattr(self):
        # The stub has a log_box; the controller should reach it.
        self.assertIs(self.ctrl.log_box, self.stub.log_box)

    def test_get_selected_face_indices_empty(self):
        # Method that operates purely on self.face_preview_entries.
        self.assertEqual(self.ctrl.get_selected_face_indices(), [])

    def test_get_selected_face_indices_populated(self):
        self.ctrl.face_preview_entries = [
            {"index": 0, "selected": True},
            {"index": 1, "selected": False},
            {"index": 2, "selected": True},
        ]
        self.assertEqual(self.ctrl.get_selected_face_indices(), [0, 2])


# ---------------------------------------------------------------------------
# PipelineController
# ---------------------------------------------------------------------------

@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 not installed; skipping")
class TestPipelineControllerDirect(unittest.TestCase):
    def setUp(self):
        from gui.pipeline import PipelineController
        self.stub = _StubWindow()
        self.ctrl = PipelineController(self.stub)

    def test_initial_state(self):
        self.assertIsNone(self.ctrl.process)
        self.assertFalse(self.ctrl.run_paused)
        self.assertEqual(self.ctrl.current_run_phase, "idle")
        self.assertEqual(self.ctrl.preprocess_stage, "idle")
        self.assertEqual(self.ctrl.stage_started_at, {})
        self.assertEqual(self.ctrl.stage_elapsed, {})
        self.assertEqual(self.ctrl._process_log_pending_text, [])
        self.assertFalse(self.ctrl._process_log_flush_queued)


# ---------------------------------------------------------------------------
# PreviewController
# ---------------------------------------------------------------------------

@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 not installed; skipping")
class TestPreviewControllerDirect(unittest.TestCase):
    def setUp(self):
        from gui.preview import PreviewController
        self.stub = _StubWindow()
        self.ctrl = PreviewController(self.stub)

    def test_initial_state(self):
        self.assertIsNone(self.ctrl.input_pixmap)
        self.assertIsNone(self.ctrl.result_pixmap)
        self.assertEqual(len(self.ctrl.result_preview_pixmap_cache), 0)
        self.assertEqual(self.ctrl.input_face_boxes, [])
        self.assertFalse(self.ctrl._compare_wipe_active)
        self.assertIsNone(self.ctrl._compare_wipe_last_pos)
        self.assertIsNone(self.ctrl._compare_wipe_result_scaled_key)

    def test_result_preview_cache_key_is_a_method(self):
        # Reachable as a regular method on the controller. (The shim on
        # MainWindow delegates here.)
        key = self.ctrl._result_preview_cache_key(Path("/no/such/file/x.png"))
        self.assertIsInstance(key, str)


if __name__ == "__main__":
    unittest.main()
