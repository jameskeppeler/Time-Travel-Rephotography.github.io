"""Headless GUI smoke test.

Constructs the MainWindow under the offscreen Qt platform and verifies that
the basic widgets, signal wiring, and audit-fix invariants survive a real
instantiation. This catches a class of regression that AST-level inspection
in test_audit_fixes.py cannot: missing attributes referenced during
__init__, signal-connection ordering bugs, etc. (The auto_recompose
ordering bug fixed in commit d612099 is exactly the shape of bug this
catches.)

Skips automatically if PySide6 is not installed.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must be set BEFORE PySide6 is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The GUI imports nvidia-smi-style checks; ensure they fail fast and quiet.
os.environ.setdefault("REPHOTO_PARSE_CACHE_DISABLE", "")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


# Ensure the repo root is importable so we can `from gui import app`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 not installed; skipping GUI smoke test")
class TestGuiSmoke(unittest.TestCase):
    """Construct MainWindow and verify the audit-critical wiring."""

    @classmethod
    def setUpClass(cls):
        # A single QApplication is reused across the class. Some Qt builds
        # refuse a second instance.
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from gui import app as gui_app
        cls.gui_app = gui_app
        cls.window = gui_app.MainWindow()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.window.close()
        finally:
            cls.window = None
            # Don't quit the QApplication; pytest may reuse it.

    # ------------------------------------------------------------------
    # Construction-time invariants
    # ------------------------------------------------------------------

    def test_window_constructed(self):
        self.assertIsNotNone(self.window)
        # Window title is set at construction.
        self.assertTrue(bool(self.window.windowTitle()))

    def test_primary_buttons_exist(self):
        for attr in ("run_button", "cancel_button", "redetect_faces_button", "end_early_button"):
            self.assertTrue(hasattr(self.window, attr), f"missing {attr}")

    def test_run_button_shortcut(self):
        # Ctrl+Return is the documented shortcut; ensure it survives __init__.
        seq = self.window.run_button.shortcut().toString()
        self.assertIn("Ctrl", seq)
        # Accept either "Ctrl+Return" or "Ctrl+Enter" depending on Qt platform.
        self.assertTrue("Return" in seq or "Enter" in seq, f"shortcut was {seq!r}")

    def test_cancel_button_shortcut_and_state(self):
        seq = self.window.cancel_button.shortcut().toString()
        self.assertEqual(seq, "Esc")  # Qt canonicalizes "Escape" -> "Esc"
        self.assertFalse(self.window.cancel_button.isEnabled(),
                         "Cancel must start disabled until a run is active")

    def test_redetect_button_shortcut(self):
        seq = self.window.redetect_faces_button.shortcut().toString()
        self.assertEqual(seq, "Ctrl+D")

    def test_accessible_names_set(self):
        # a11y metadata must reach the actual widgets.
        self.assertTrue(self.window.run_button.accessibleName())
        self.assertTrue(self.window.cancel_button.accessibleName())

    def test_log_box_is_timestamped_subclass(self):
        # Confirms commit-round-2: log_box is the TimestampedLogBox class,
        # not a bare QTextEdit. Subsequent log_box.append() calls auto-prefix.
        self.assertEqual(self.window.log_box.__class__.__name__, "TimestampedLogBox")

    def test_log_box_append_prepends_timestamp(self):
        before = self.window.log_box.toPlainText()
        self.window.log_box.append("smoke test message")
        after = self.window.log_box.toPlainText()
        new_lines = after[len(before):].strip().splitlines()
        self.assertTrue(new_lines, "append must add at least one line")
        last = new_lines[-1]
        # Expect "[HH:MM:SS] smoke test message"
        self.assertRegex(last, r"^\[\d{1,2}:\d{2}:\d{2}\]\s+smoke test message$")

    def test_advanced_dialog_constructed_with_defaults(self):
        # advanced_dialog and its widgets are referenced by ~83 call sites;
        # if __init__ ever regresses the order, this catches it.
        self.assertTrue(hasattr(self.window, "advanced_dialog"))
        adv = self.window.advanced_dialog
        for attr in (
            "crop_only_checkbox",
            "use_gfpgan_checkbox",
            "det_threshold_edit",
            "face_factor_edit",
            "auto_recompose_checkbox_adv",
        ):
            self.assertTrue(hasattr(adv, attr), f"AdvancedSettingsDialog missing {attr}")

    def test_auto_recompose_main_and_adv_paired(self):
        # The previously-fixed signal-ordering bug (commit d612099) required
        # both checkboxes to exist before signal wiring. Verify both end up on
        # the window in their canonical attributes.
        self.assertTrue(hasattr(self.window, "auto_recompose_checkbox"))
        self.assertTrue(
            hasattr(self.window.advanced_dialog, "auto_recompose_checkbox_adv")
        )

    def test_pixmap_loader_signals_wired(self):
        # The async pixmap loader infrastructure must be ready immediately;
        # otherwise the first input-image drop would AttributeError.
        self.assertTrue(hasattr(self.window, "_input_pixmap_loader_signals"))
        self.assertTrue(hasattr(self.window, "_result_pixmap_loader_signals"))
        self.assertTrue(hasattr(self.window, "_pixmap_thread_pool"))

    def test_compare_wipe_cache_state_initialized(self):
        # Cache state added with P10 round 3 must exist on a fresh instance;
        # otherwise the first mouse-move over the result preview AttributeErrors.
        for attr in (
            "_compare_wipe_result_scaled",
            "_compare_wipe_result_scaled_key",
            "_compare_wipe_input_scaled",
            "_compare_wipe_input_scaled_key",
        ):
            self.assertTrue(hasattr(self.window, attr), f"missing {attr}")

    def test_log_box_colorizes_errors(self):
        # When an "error" cue is in the message, the TimestampedLogBox should
        # store the line as colorized HTML (toHtml() contains the red color),
        # while toPlainText() still yields the timestamped text.
        before_html = self.window.log_box.toHtml()
        self.window.log_box.append("backend failed to launch: bad path")
        after_html = self.window.log_box.toHtml()
        # The red error color must appear among the newly added HTML.
        new_html = after_html[len(before_html):]
        self.assertIn("#eb5757", new_html.lower())

    def test_log_box_colorizes_warnings(self):
        before_html = self.window.log_box.toHtml()
        self.window.log_box.append("Cropper face-box probe timed out")
        after_html = self.window.log_box.toHtml()
        new_html = after_html[len(before_html):]
        self.assertIn("#f2c94c", new_html.lower())

    def test_log_box_info_lines_unstyled(self):
        # Plain info lines should not be wrapped in a color span.
        before_html = self.window.log_box.toHtml()
        self.window.log_box.append("Detected 5 faces.")
        after_html = self.window.log_box.toHtml()
        new_html = after_html[len(before_html):]
        # Neither the error nor warn color should appear for an info line.
        self.assertNotIn("#eb5757", new_html.lower())
        self.assertNotIn("#f2c94c", new_html.lower())

    def test_face_strip_mixin_methods_resolve_via_mro(self):
        # After the Sprint-4 mixin extraction, face-strip methods live on
        # FaceStripMixin. Confirm MRO still delivers them to MainWindow.
        for name in (
            "render_face_preview_strip",
            "reset_face_preview_state",
            "initialize_face_preview_entries",
            "get_selected_face_indices",
            "_compute_face_strip_render_signature",
        ):
            self.assertTrue(
                callable(getattr(self.window, name, None)),
                f"MainWindow.{name} is not callable via MRO",
            )

    def test_face_strip_render_callable_no_entries(self):
        # Empty entries must render without raising — this exercises the
        # MRO path end-to-end without needing real face data.
        self.window.face_preview_entries = []
        self.window.render_face_preview_strip()  # must not raise
        # Re-call to exercise the signature short-circuit.
        self.window.render_face_preview_strip()

    def test_pause_ack_helpers_callable(self):
        # Round-4 pause-ack helpers must be safely callable even with no
        # active run (no timer to cancel).
        self.window._cancel_pause_ack_warning()  # must not raise
        # Arming requires the helpers to construct a QTimer cleanly.
        self.window._arm_pause_ack_warning()
        self.assertIsNotNone(getattr(self.window, "_pause_ack_warn_timer", None))
        self.window._cancel_pause_ack_warning()
        self.assertIsNone(getattr(self.window, "_pause_ack_warn_timer", None))


if __name__ == "__main__":
    unittest.main()
