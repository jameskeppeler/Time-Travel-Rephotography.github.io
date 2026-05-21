import json
import os
import re
import sys
import ctypes
import platform
import subprocess
import shutil
import time
import math
import random
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QProcess, QRect, QSize, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QFont, QIcon, QImage, QImageReader, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QBoxLayout,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSplitter,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# Shared defaults, presets, and stdout-parsing regexes live in
# gui/constants.py since the Sprint-4 split; re-exported here so existing
# call sites that read them off the gui.app namespace keep working.
from gui import format_utils, path_utils, runtime_estimator, timing_log as timing_log_io
from gui.face_strip import FaceStripController
from gui.pipeline import PipelineMixin
from gui.preflight import PreflightController
from gui.preview import PreviewMixin
from gui.constants import (
    CROP_ALIGN_BOX_RE,
    CROPPED_FACE_COUNT_RE,
    DEFAULT_ADVANCED_ITER_VALUES,
    DEFAULT_BASIC_ITER_VALUES,
    DEFAULT_CAMERA_LR,
    DEFAULT_DET_THRESHOLD,
    DEFAULT_FACE_FACTOR,
    DEFAULT_GAUSSIAN,
    DEFAULT_GFPGAN_BLEND,
    DEFAULT_ITERATION,
    DEFAULT_LR,
    DEFAULT_MIX_LAYER_END,
    DEFAULT_MIX_LAYER_START,
    DEFAULT_NOISE_REGULARIZE,
    FACE_SUFFIX_INDEX_RE,
    IMAGE_EXTENSIONS,
    ITER_PROGRESS_RE,
    NO_CROPS_CREATED_RE,
    PHOTO_TYPE_DATE_HINTS,
    QUALITY_PRESET_LABELS,
    QUICK_FACE_DECISION_RE,
    REPHOTO_CROP_FAIL_RE,
    RETINA_FACE_BOX_RE,
    SIMPLE_FINAL_COPY_RE,
    WIDE_LAYOUT_MIN_WIDTH,
    YEAR_RE,
)

# Widget classes and worker helpers were extracted to gui/widgets.py as the
# first step of the Sprint-4 module split. See [docs/REPHOTO_PARAMETER_GUIDE.md]
# and CLAUDE.md for the planned full split.
from gui.widgets import (
    FaceStripToolButton,
    FilmstripContainerWidget,
    InputDetectOverlay,
    InputDropLabel,
    InstantToolButton,
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSlider,
    NoScrollSpinBox,
    ResultPreviewLabel,
    TimestampedLogBox,
    _PixmapLoader,
    _PixmapLoaderSignals,
    _PreflightRunner,
    _PreflightSignals,
)

# AdvancedSettingsDialog was extracted to gui/dialogs.py as the second
# slice of the Sprint-4 module split.
from gui.dialogs import AdvancedSettingsDialog

class MainWindow(
    PipelineMixin,
    PreviewMixin,
    QMainWindow,
):
    def make_form_label(self, label_text):
        label = QLabel(label_text)
        label.setFixedWidth(168)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return label

    def make_label_with_info(self, label_text, tooltip_text):
        label_widget = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        label_widget.setLayout(row)
        label_widget.setFixedWidth(168)

        label = QLabel(label_text)

        info_button = InstantToolButton()
        info_button.setText("ⓘ")
        info_button.setAutoRaise(True)
        info_button.setFixedSize(16, 16)
        info_button.setToolTip(tooltip_text)
        info_button.setCursor(Qt.PointingHandCursor)

        row.addWidget(label)
        row.addWidget(info_button)
        row.addStretch()

        return label_widget

    def _main_parameters_help_html(self):
        return """
<h3>Main Window Parameters</h3>
<p>
  This section covers only the primary tuning controls shown on the main form.
  Runtime/log controls and action buttons are intentionally omitted.
</p>
<ul>
  <li>
    <b>Photo type / process</b><br/>
    This is a historical context hint. It helps the app choose a more realistic spectral model for early photographic materials and presentation formats.
    If you know the process class (for example Daguerreotype, Ambrotype, or early gelatin silver), setting it correctly usually improves tonal behavior more than trying random refinement tweaks.
  </li>
  <li>
    <b>Approximate date</b><br/>
    Date is used as a second cue for spectral inference and historical rendering assumptions.
    A rough decade is enough. For many portraits, entering a plausible period (for example 1860s, 1890s, 1910s) helps avoid modern-looking color/contrast biases by steering the degradation model toward era-appropriate sensitivity.
  </li>
  <li>
    <b>Spectral sensitivity</b><br/>
    This selects the channel-response model used by the historical degradation component.
    It strongly affects skin/background balance and perceived age of the render. Blue-sensitive and orthochromatic behavior can produce noticeably different facial tonality, so this control has high visual impact even when other settings stay fixed.
  </li>
  <li>
    <b>Quality</b><br/>
    Quality controls optimization effort (step count/schedule), not a simple post-filter.
    Higher values generally improve fidelity and identity stability, but with diminishing returns and longer runtime.
    For iteration speed, start at moderate values, validate crop/spectral behavior, then increase quality only after baseline appearance looks correct.
  </li>
</ul>
"""

    def _advanced_parameters_help_html(self):
        return """
<h3>Advanced Settings Parameters</h3>
<h4>Core Historical</h4>
<ul>
  <li>
    <b>Multi-face selection flow</b><br/>
    The app now auto-detects faces first, then lets you select which faces to rephotograph in the filmstrip when multiple faces are found.
    Single-face photos continue automatically. For multi-face photos, choose one, several, or all faces before continuing.
  </li>
  <li>
    <b>Crop Only</b><br/>
    Stops after detection/cropping so you can validate framing before expensive optimization.
    This is primarily a diagnostic workflow and is useful when results look wrong due to composition rather than model settings.
  </li>
  <li>
    <b>Enhancement</b><br/>
    Controls the pre-pass restoration stage (GFPGAN). It can recover structure in damaged/soft faces but may also inject modern priors.
    If outputs become stylized or over-smoothed, compare runs with enhancement disabled to isolate whether the issue originates before projector optimization.
  </li>
  <li>
    <b>Enhancement blend</b><br/>
    Sets how strongly restored pixels are blended into the face crop before rephoto.
    Lower blend preserves more authentic source texture; higher blend can improve damaged regions but risks synthetic facial surfaces.
    This parameter is often a key lever when balancing realism versus preservation.
  </li>
  <li>
    <b>Face detection sensitivity</b><br/>
    Higher values are stricter and reduce false detections; lower values are more permissive for difficult photos.
    If the crop misses a face, lower this slightly. If it picks up incorrect regions, raise it.
  </li>
  <li>
    <b>Face crop expansion</b><br/>
    Expands context around the detected face box (hairline, jawline, ears, and border context).
    Too tight can cause identity/style drift; too loose may introduce distracting background influence.
    This is one of the highest-impact controls for natural-looking output and should be stabilized early.
  </li>
  <li>
    <b>Gaussian blur</b><br/>
    Models historical softness in the degradation pipeline.
    Moderate blur can better match period optics and film characteristics; too much can flatten facial detail and make identity weaker.
  </li>
</ul>
<h4>Refinement</h4>
<ul>
  <li>
    <b>Identity preservation</b><br/>
    Increases pressure to keep reconstructed facial identity consistent with the source portrait.
    Higher settings help protect facial geometry, but can also resist useful corrections when the source is heavily degraded.
  </li>
  <li>
    <b>Tonal transfer</b><br/>
    Governs how strongly tonal relationships from the source are preserved.
    This is important for period mood and lighting continuity, but overly strong transfer can lock in undesirable exposure or contrast artifacts.
  </li>
  <li>
    <b>Eye preservation</b><br/>
    Adds extra emphasis on eye-region stability (gaze, lids, and local structure).
    Useful when identity drift appears mostly in eyes. Excessive emphasis can occasionally overconstrain expression.
  </li>
  <li>
    <b>Structure matching</b><br/>
    Controls broader perceptual structure constraints beyond exact pixel agreement.
    Increasing it can preserve composition and facial arrangement, but if too strong it may limit creative recovery from noisy artifacts.
  </li>
  <li>
    <b>VGG appearance matching</b><br/>
    Applies additional perceptual appearance guidance.
    Useful for texture/style coherence, but can sometimes introduce color/style drift if set too aggressively for historical inputs.
  </li>
</ul>
<h4>Experimental</h4>
<ul>
  <li>
    <b>Noise regularization</b><br/>
    Penalizes optimization shortcuts that hide artifacts in noise maps.
    Raising it usually improves stability and suppresses synthetic texture speckle, but extremely high values may reduce fine detail recovery.
  </li>
  <li>
    <b>Learning rate</b><br/>
    Main optimizer step size for latent refinement.
    Higher values converge faster but can overshoot and destabilize; lower values are steadier but slower.
  </li>
  <li>
    <b>Camera learning rate</b><br/>
    Separate step size for degradation/camera parameter optimization.
    This affects how quickly viewpoint and imaging-model terms adapt relative to face latent updates.
  </li>
  <li>
    <b>Mix layer start / end</b><br/>
    Defines the latent layer span used during initialization mixing between encoders.
    Lower layers affect coarse structure; higher layers affect finer style/detail. Narrow or shifted ranges can materially change identity/style balance.
  </li>
</ul>
"""

    def show_parameter_help_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Parameter Help")
        dialog.setMinimumSize(860, 620)

        layout = QVBoxLayout(dialog)
        guide = QTextBrowser()
        guide.setOpenExternalLinks(True)
        guide.setHtml(
            self._main_parameters_help_html()
            + "<hr/>"
            + self._advanced_parameters_help_html()
        )
        layout.addWidget(guide)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def show_about_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About Time-Travel Rephotography")
        dialog.setMinimumSize(720, 520)

        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(
            """
<h2>Time-Travel Rephotography</h2>
<p>Local desktop workflow for historical portrait restoration and rephotography.</p>
<p><b>Original Project Website:</b> <a href="https://time-travel-rephotography.github.io/">time-travel-rephotography.github.io</a></p>
<p><b>Original Upstream Repository:</b> <a href="https://github.com/caojiezhang/Time-Travel-Rephotography">github.com/caojiezhang/Time-Travel-Rephotography</a></p>
<p><b>Paper:</b> <a href="https://arxiv.org/abs/2012.12261">Time-Travel Rephotography (SIGGRAPH Asia 2021)</a></p>
<h3>Major Open-Source Projects Used</h3>
<ul>
  <li><a href="https://github.com/NVlabs/stylegan2">NVIDIA StyleGAN2</a></li>
  <li><a href="https://github.com/omertov/encoder4editing">encoder4editing (e4e)</a></li>
  <li><a href="https://github.com/TencentARC/GFPGAN">GFPGAN</a></li>
  <li><a href="https://github.com/xinntao/facexlib">facexlib</a></li>
  <li><a href="https://github.com/zllrunning/face-parsing.PyTorch">face-parsing.PyTorch</a></li>
  <li><a href="https://doc.qt.io/qtforpython/">PySide6 / Qt for Python</a></li>
</ul>
<h3>Notes</h3>
<ul>
  <li>Outputs are highly sensitive to crop framing, spectral mode, and refinement settings.</li>
  <li>Runtime depends on quality preset, GPU class, and enhancement options.</li>
  <li>Use Help -> Parameter Guide for control-by-control explanations.</li>
</ul>
<p>Licensing files in this repo: <code>LICENSE</code>, <code>LICENSE-NVIDIA</code>, <code>LICENSE-STYLEGAN2</code>.</p>
"""
        )
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def open_results_root_folder(self):
        results_path = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
        try:
            results_path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(results_path.resolve()))
        except Exception as e:
            self.log_box.append(f"Could not open results folder: {e}")

    def open_project_readme(self):
        readme_path = self.resolve_resource_path("README.md")
        if not readme_path.exists():
            self.log_box.append("README.md not found.")
            return
        try:
            os.startfile(str(readme_path.resolve()))
        except Exception as e:
            self.log_box.append(f"Could not open README.md: {e}")

    def setup_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        open_image_action = QAction("Open Image...", self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self.browse_for_image)
        file_menu.addAction(open_image_action)

        open_results_action = QAction("Open Results Folder", self)
        open_results_action.triggered.connect(self.open_results_root_folder)
        file_menu.addAction(open_results_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menu_bar.addMenu("&View")
        self.menu_toggle_log_action = QAction("", self)
        self.menu_toggle_log_action.setShortcut("Ctrl+L")
        self.menu_toggle_log_action.triggered.connect(self.toggle_log_visibility)
        view_menu.addAction(self.menu_toggle_log_action)

        self.menu_expand_log_action = QAction("", self)
        self.menu_expand_log_action.setShortcut("Ctrl+Shift+L")
        self.menu_expand_log_action.triggered.connect(self.toggle_log_size)
        view_menu.addAction(self.menu_expand_log_action)
        view_menu.addSeparator()
        self.menu_show_last_summary_action = QAction("Show Last Run Summary", self)
        self.menu_show_last_summary_action.setShortcut("Ctrl+R")
        self.menu_show_last_summary_action.triggered.connect(self.preflight.show_last_run_summary_dialog)
        view_menu.addAction(self.menu_show_last_summary_action)
        view_menu.addSeparator()
        self.menu_face_box_debug_action = QAction("Show Face Box IDs (Debug)", self)
        self.menu_face_box_debug_action.setCheckable(True)
        self.menu_face_box_debug_action.setShortcut("Ctrl+Shift+D")
        self.menu_face_box_debug_action.toggled.connect(self.set_face_box_debug_overlay_enabled)
        view_menu.addAction(self.menu_face_box_debug_action)

        help_menu = menu_bar.addMenu("&Help")
        parameter_help_action = QAction("Parameter Guide", self)
        parameter_help_action.setShortcut("F1")
        parameter_help_action.triggered.connect(self.show_parameter_help_dialog)
        help_menu.addAction(parameter_help_action)

        readme_action = QAction("Open Project README", self)
        readme_action.triggered.connect(self.open_project_readme)
        help_menu.addAction(readme_action)

        preflight_action = QAction("Run Startup Preflight", self)
        preflight_action.triggered.connect(lambda: self.preflight.run_startup_preflight(show_dialog=True, user_initiated=True))
        help_menu.addAction(preflight_action)

        help_menu.addSeparator()
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        self.update_view_menu_actions()

    def update_view_menu_actions(self):
        if hasattr(self, "menu_toggle_log_action"):
            self.menu_toggle_log_action.setText("Hide Log" if self.log_visible else "Show Log")
        if hasattr(self, "menu_expand_log_action"):
            self.menu_expand_log_action.setText("Compact Log" if self.log_expanded else "Expand Log")
            self.menu_expand_log_action.setEnabled(self.log_visible)
        if hasattr(self, "menu_show_last_summary_action"):
            self.menu_show_last_summary_action.setEnabled(bool(self.preflight.last_run_summary_text))
        if hasattr(self, "menu_face_box_debug_action"):
            self.menu_face_box_debug_action.setChecked(bool(getattr(self, "face_box_debug_overlay_enabled", False)))

    def set_face_box_debug_overlay_enabled(self, enabled):
        self.face_box_debug_overlay_enabled = bool(enabled)
        self.refresh_input_preview_scale()

    def detect_app_root(self):
        """Resolve the application root robustly for source runs and packaged executables."""
        script_root = Path(__file__).resolve().parent.parent
        seeds = [script_root]

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            seeds.extend([exe_dir, exe_dir.parent])
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                mp = Path(meipass).resolve()
                seeds.extend([mp, mp.parent])

        markers = ("run_rephoto_with_facecrop.ps1", "projector.py")
        checked = set()

        for seed in seeds:
            for candidate in [seed, *list(seed.parents)[:3]]:
                candidate = candidate.resolve()
                if candidate in checked:
                    continue
                checked.add(candidate)
                if all((candidate / marker).exists() for marker in markers):
                    return candidate

        return script_root

    def resolve_resource_path(self, relative_path):
        """Resolve a resource path from app root and packaged fallbacks."""
        cache = getattr(self, "_resolve_resource_path_cache", None)
        if cache is None:
            cache = {}
            self._resolve_resource_path_cache = cache
        rel_key = str(relative_path)
        cached = cache.get(rel_key)
        if cached is not None:
            return cached

        rel = Path(relative_path)
        roots = []

        app_root = getattr(self, "app_root", Path(__file__).resolve().parent.parent)
        roots.append(app_root)

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            roots.extend([exe_dir, exe_dir.parent])
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                mp = Path(meipass).resolve()
                roots.extend([mp, mp.parent])

        seen = set()
        for root in roots:
            root = root.resolve()
            if root in seen:
                continue
            seen.add(root)
            candidate = root / rel
            if candidate.exists():
                cache[rel_key] = candidate
                return candidate

        result = (app_root / rel).resolve()
        cache[rel_key] = result
        return result

    def _to_bool(self, value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def _set_combo_if_present(self, combo: QComboBox, value: str):
        if not value:
            return
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_iteration_from_value(self, iteration_value):
        try:
            iv = int(iteration_value)
        except Exception:
            iv = DEFAULT_ITERATION
        closest_index = min(range(len(self.iter_values)), key=lambda i: abs(self.iter_values[i] - iv))
        self.iter_slider.setValue(closest_index)


    def closeEvent(self, event):
        try:
            self.stop_quick_face_probe()
        except Exception:
            pass

        # Stop long-lived timers so they don't fire during teardown.
        for attr in ("_elapsed_timer", "_result_stage_timer"):
            timer = getattr(self, attr, None)
            if timer is not None:
                try:
                    timer.stop()
                except RuntimeError:
                    pass

        # Terminate the main rephoto subprocess if one is running; otherwise it
        # keeps the GPU/disk pinned after the window closes.
        proc = getattr(self, "process", None)
        if proc is not None:
            try:
                if proc.state() != QProcess.NotRunning:
                    proc.terminate()
                    # Give it a short grace period, then force-kill if needed.
                    # We cannot block the close, so wait briefly and force.
                    if not proc.waitForFinished(500):
                        self._kill_process_if_running(proc)
            except RuntimeError:
                pass

        super().closeEvent(event)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self._BASE_WINDOW_TITLE)

        # Sprint-4 polish: __init__ used to be 929 lines. It's now
        # split into three named build steps that each do one job.
        self._init_state()
        self._build_ui()
        self._finalize_init()

    def _init_state(self):
        """Set every MainWindow instance attribute used elsewhere on the
        class. No widgets are created here; this method must be safe to
        call before the central widget exists.
        """
        self.app_root = self.detect_app_root()
        self.repo_root = self.app_root
        self.wrapper_script = self.resolve_resource_path("run_rephoto_with_facecrop.ps1")
        self.process = None
        self.run_paused = False
        self.current_stop_flag_path = None
        self.current_pause_flag_path = None
        self.run_started_at = None
        self.rephoto_started_at = None
        self.current_run_summary_context = None
        # PreflightController owns last_run_summary_text / last_preflight_report /
        # hardware-info cache / running flag, etc. Instantiated here so it's
        # available everywhere downstream that reads self.preflight.* .
        self.preflight = PreflightController(self)
        self.face_strip = FaceStripController(self)
        self._timing_records_cache_path = None
        self._timing_records_cache_mtime = None
        self._timing_records_cache = []
        self._haar_face_detector = None
        self._conda_executable_cache = None
        self._facecrop_env_cache_key = None
        self._facecrop_env_name_cache = None
        self._quick_probe_det_threshold = None
        self._last_backend_error_detail = ""

        # Async pixmap loaders (input + result previews). Off-UI-thread image
        # decode avoids freezes on 50-200 MB historical scans.
        self._pixmap_thread_pool = QThreadPool.globalInstance()
        self._input_pixmap_loader_signals = _PixmapLoaderSignals()
        self._input_pixmap_loader_signals.loaded.connect(self._on_input_pixmap_loaded)
        self._input_pixmap_loader_signals.failed.connect(self._on_input_pixmap_failed)
        self._input_pixmap_loader_path = None
        self._result_pixmap_loader_signals = _PixmapLoaderSignals()
        self._result_pixmap_loader_signals.loaded.connect(self._on_result_pixmap_loaded)
        self._result_pixmap_loader_signals.failed.connect(self._on_result_pixmap_failed)
        self._result_pixmap_loader_path = None

        self.preprocess_stage = "idle"
        self.rephoto_stage = None
        self.rephoto_stage_name = None
        self.rephoto_stage_current = 0
        self.rephoto_stage_total = 0
        self.rephoto_total_done_before_stage = 0
        self.rephoto_total_work = 0
        self.rephoto_step_pair = (250, 750)
        self.rephoto_face_current_index = 0
        self.rephoto_face_total = 1

        # run phase state for gating progress bars
        self.current_run_phase = "idle"  # idle, preprocess, rephoto, done, cancelled, crop_only_done
        self.selection_preprocess_mode = False
        self.awaiting_face_selection = False

        # Path tracking from stdout to avoid recursive folder scans
        self.current_crop_output_dir = None
        self.current_gfpgan_output_dir = None
        self._inprocess_preview_crops = False
        self._crop_source_input_key = None  # normalized path key of the input that produced current crops
        self._crop_source_face_factor = None  # face_factor value used when crops were produced
        self.face_strip._pending_face_reselection = None  # set of face indices to pre-select after a re-crop
        self.current_blended_faces_dir = None
        self.current_results_dir = None
        self.current_manifest_path = None
        self.current_run_result_dirs = set()

        # Stage timing instrumentation
        self.stage_started_at = {}  # {"crop": timestamp, "enhance": timestamp, ...}
        self.stage_elapsed = {}    # {"crop": seconds, "enhance": seconds, ...}
        self._newest_image_query_cache = {}
        self._newest_image_query_cache_max_entries = 64
        self._runtime_label_cache_key = None
        self._runtime_label_cache_text = None
        self._runtime_label_cache_tooltip = None

        self.log_expanded = False
        self.log_visible = False

        # Preview state
        self.input_pixmap = None
        self.result_pixmap = None
        self.last_result_image_path = None
        self.last_result_image_cache_key = None
        # Proper LRU: OrderedDict + move_to_end on access avoids the
        # pop+reinsert dance and is robust against dict-ordering surprises.
        self.result_preview_pixmap_cache = OrderedDict()
        self.result_preview_pixmap_cache_max_entries = 96
        self.result_preview_path_before_hover = None
        self.input_face_boxes = []
        self.input_face_box_source = None
        self.face_box_debug_overlay_enabled = False
        self.face_box_probe_cache = {}
        self.face_box_probe_cache_max_entries = 24
        self._face_overlay_detector_warned = False
        self.input_preview_scaled_cache_key = None
        self.input_preview_scaled_cache_pixmap = None
        self.input_preview_render_cache_key = None
        self.input_preview_render_cache_pixmap = None
        self.input_preview_last_display_key = None
        self.result_preview_scaled_cache_key = None
        self.result_preview_scaled_cache_pixmap = None
        self.result_preview_last_display_key = None
        self.face_strip.face_preview_thumb_icon_cache = {}
        self.face_strip.face_preview_thumb_icon_cache_max_entries = 256
        self.face_strip._face_strip_render_signature = None

        # Multi-face preview state (for strategy=all / multi-face detections)
        self.face_strip.face_preview_entries = []
        self.face_strip.active_face_preview_index = None
        self.face_strip.selected_face_preview_index = None
        self.face_strip._user_inspecting_completed_face = False  # True when user manually clicked a completed face during processing
        self.face_strip.hover_face_preview_index = None
        self.face_strip.hover_face_preview_source = None
        self.face_strip.hover_face_box_override = None
        self.face_strip.hover_face_box_cache = {}
        self.quick_face_count_estimate = None
        self.face_strip._no_faces_detected = False
        self.current_wide_preview_side = 360
        self.quick_face_probe_process = None
        self.quick_face_probe_token = 0
        self.quick_face_probe_target_input = None
        self.quick_face_probe_fallback_count = None
        self.quick_face_probe_warned = False
        self.quick_face_probe_stdout = ""
        self.quick_face_probe_last_error = ""
        self._process_stdout_buffer = ""
        self._process_stderr_buffer = ""
        self._process_log_pending_text = []  # Use list for O(1) append instead of string +=
        self._process_log_pending_text_bytes = 0  # Track size instead of len(string)
        self._process_log_flush_queued = False
        self.retina_face_box_probe_warned = False
        self.cropper_face_box_probe_warned = False
        self.auto_detect_faces_on_import = True
        self.auto_detect_faces_armed_input = None
        self.auto_detect_faces_triggered_input = None
        self.suppress_preprocess_ui_until_rephoto = False
        self._last_iter_progress_signature = None
        self._last_preprocess_progress_state = None
        self._last_rephoto_progress_state = None


    def _build_ui(self):
        """Construct the central widget tree, lay out widgets, and wire
        signals. Reads attributes set by _init_state.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

        central = QWidget()
        scroll_area.setWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        central.setLayout(main_layout)

        # --- Input image ---
        input_row = QHBoxLayout()
        self.input_image_edit = QLineEdit()
        self.input_image_edit.setPlaceholderText("Select an input image...")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_for_image)

        input_row.addWidget(self.input_image_edit)
        input_row.addWidget(self.browse_button)

        # --- Main settings ---
        form_layout = QFormLayout()
        self.form_layout = form_layout
        form_layout.setVerticalSpacing(6)
        form_layout.setHorizontalSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.advanced_dialog = AdvancedSettingsDialog(self)
        self.advanced_dialog.strategy_combo.setCurrentText("all")
        self.advanced_dialog.crop_only_checkbox.setChecked(False)
        self.advanced_dialog.use_gfpgan_checkbox.setChecked(False)
        self.advanced_dialog.det_threshold_edit.setValue(0.90)
        self.advanced_dialog.face_factor_edit.setValue(DEFAULT_FACE_FACTOR)
        self.advanced_dialog.gfpgan_blend_edit.setValue(0.45)
        self.advanced_dialog.gaussian_edit.setValue(0.75)
        self.advanced_dialog.identity_preservation_combo.setCurrentText("Default")
        self.advanced_dialog.tonal_transfer_combo.setCurrentText("Default")
        self.advanced_dialog.eye_preservation_combo.setCurrentText("Default")
        self.advanced_dialog.structure_matching_combo.setCurrentText("Default")
        self.advanced_dialog.vgg_appearance_combo.setCurrentText("Default")
        self.advanced_dialog.noise_regularize_edit.setValue(50000.0)
        self.advanced_dialog.lr_edit.setValue(0.1)
        self.advanced_dialog.camera_lr_edit.setValue(0.01)
        self.advanced_dialog.mix_layer_start_edit.setValue(10)
        self.advanced_dialog.mix_layer_end_edit.setValue(18)

        # --- Auto-recomposition checkbox ---
        self.auto_recompose_checkbox = QCheckBox("Auto-recompose after rephoto")
        self.auto_recompose_checkbox.setChecked(False)
        self.auto_recompose_checkbox.setToolTip(
            "Automatically apply Color blend recomposition once each face finishes rephotography"
        )

        self.advanced_dialog.crop_only_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_runtime_label)
        # Sync auto-recompose checkbox between main form and advanced dialog
        self.advanced_dialog.auto_recompose_checkbox_adv.toggled.connect(self._sync_auto_recompose_from_adv)
        self.auto_recompose_checkbox.toggled.connect(self._sync_auto_recompose_from_main)

        # --- Detection threshold (linked to advanced dialog) ---
        self.det_threshold_label = QLabel("Detection threshold:")
        self.det_threshold_slider = NoScrollSlider(Qt.Horizontal)
        self.det_threshold_slider.setMinimum(0)
        self.det_threshold_slider.setMaximum(100)
        self.det_threshold_slider.setValue(90)
        self.det_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.det_threshold_slider.setTickInterval(10)
        self.det_threshold_value_label = QLabel("0.90")
        self.det_threshold_slider.valueChanged.connect(self._update_detection_threshold)
        self.det_threshold_info = InstantToolButton()
        self.det_threshold_info.setText("ⓘ")
        self.det_threshold_info.setAutoRaise(True)
        self.det_threshold_info.setFixedSize(16, 16)
        self.det_threshold_info.setCursor(Qt.PointingHandCursor)
        self.det_threshold_info.setToolTip(
            "Face detection confidence threshold (0.0–1.0). Lower values detect more faces "
            "(including blurry/partial), higher values detect only clear faces. 0.90 is recommended."
        )

        # --- Iteration slider ---
        self.basic_iter_values = DEFAULT_BASIC_ITER_VALUES
        self.advanced_iter_values = DEFAULT_ADVANCED_ITER_VALUES
        self.iter_values = self.basic_iter_values

        self.iter_slider = NoScrollSlider(Qt.Horizontal)
        self.iter_slider.setMinimum(0)
        self.iter_slider.setMaximum(len(self.iter_values) - 1)
        self.iter_slider.setValue(1)  # default 750 (now at index 1)
        self.iter_slider.setTickPosition(QSlider.TicksBelow)
        self.iter_slider.setTickInterval(1)
        self.iter_slider.valueChanged.connect(self.update_iteration_label)

        # --- Quality row ---
        self.quality_label = QLabel("Quality:")

        self.quality_value_label = QLabel("750")
        self.quality_value_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.quality_default_label = QLabel("(default)")
        self.quality_default_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.quality_info = InstantToolButton()
        self.quality_info.setText("ⓘ")
        self.quality_info.setAutoRaise(True)
        self.quality_info.setCursor(Qt.PointingHandCursor)
        self.quality_info.setFixedSize(18, 18)
        self.quality_info.setToolTip(
            "Controls reconstruction quality and runtime. "
            "Higher values usually improve results but take longer. "
            "750 is the default starting point for a quicker run."
        )

        self.runtime_label = QLabel("")
        self.runtime_info = InstantToolButton()
        self.runtime_info.setText("ⓘ")
        self.runtime_info.setAutoRaise(True)
        self.runtime_info.setCursor(Qt.PointingHandCursor)
        self.runtime_info.setFixedSize(16, 16)
        self.runtime_info.setToolTip("Estimated processing time is approximate.")

        self.runtime_prefix_label = QLabel("Est.")
        self.runtime_prefix_label.setStyleSheet("color: #b7bcc5;")
        self.runtime_label.setStyleSheet("color: #b7bcc5;")
        self.rephoto_status_text = "Waiting..."

        quality_header_row = QHBoxLayout()
        quality_header_row.setContentsMargins(0, 0, 0, 0)
        quality_header_row.setSpacing(8)
        quality_header_row.addWidget(self.quality_label)
        quality_header_row.addWidget(self.quality_value_label)
        quality_header_row.addWidget(self.quality_default_label)
        quality_header_row.addWidget(self.quality_info)
        quality_header_row.addStretch(1)

        quality_runtime_cluster = QWidget()
        quality_runtime_layout = QHBoxLayout()
        quality_runtime_layout.setContentsMargins(0, 0, 0, 0)
        quality_runtime_layout.setSpacing(3)
        quality_runtime_cluster.setLayout(quality_runtime_layout)
        quality_runtime_layout.addWidget(self.runtime_prefix_label)
        quality_runtime_layout.addWidget(self.runtime_label)
        quality_runtime_layout.addWidget(self.runtime_info)
        quality_header_row.addWidget(quality_runtime_cluster, 0, Qt.AlignVCenter)

        quality_slider_row = QHBoxLayout()
        quality_slider_row.setContentsMargins(0, 2, 0, 0)
        quality_slider_row.setSpacing(10)
        quality_slider_row.addSpacing(0)
        quality_slider_row.addWidget(self.iter_slider, 1)

        self.quality_widget = QWidget()
        quality_layout = QVBoxLayout()
        quality_layout.setContentsMargins(0, 3, 0, 2)
        quality_layout.setSpacing(4)
        self.quality_widget.setLayout(quality_layout)

        quality_layout.addLayout(quality_header_row)
        quality_layout.addLayout(quality_slider_row)

        # --- Spectral sensitivity inputs ---
        self.photo_type_combo = NoScrollComboBox()
        self.photo_type_combo.addItems([
            "Unknown",
            "Daguerreotype",
            "Ambrotype",
            "Tintype / Ferrotype",
            "Carte de visite (CDV)",
            "Cabinet card",
            "Late cabinet card / dry plate studio portrait",
            "Early gelatin silver print",
            "Black-and-white snapshot / roll-film print",
        ])

        self.approx_date_edit = QLineEdit()
        self.approx_date_edit.setPlaceholderText("e.g. 1865, 1890s, circa 1910")

        self.spectral_mode_combo = NoScrollComboBox()
        self.spectral_mode_combo.addItems(["Auto", "Manual"])

        self.spectral_sensitivity_combo = NoScrollComboBox()
        self.spectral_sensitivity_combo.addItems(["Blue-sensitive", "Orthochromatic", "Panchromatic"])

        # Connect signals for spectral sensitivity inference
        self.spectral_mode_combo.currentTextChanged.connect(self.update_spectral_sensitivity_ui)
        self.photo_type_combo.currentTextChanged.connect(self.update_spectral_sensitivity_ui)
        self.photo_type_combo.currentTextChanged.connect(self._update_date_placeholder_for_photo_type)
        self.approx_date_edit.textChanged.connect(self.update_spectral_sensitivity_ui)

        # Photo type + Approximate date on one row (equal halves)
        photo_date_row = QHBoxLayout()
        photo_date_row.setSpacing(6)
        photo_date_row.setContentsMargins(0, 0, 0, 0)
        photo_date_row.addWidget(self.photo_type_combo, 1)
        date_label = self.make_form_label("Date")
        date_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        photo_date_row.addWidget(date_label)
        photo_date_row.addWidget(self.approx_date_edit, 1)
        photo_date_widget = QWidget()
        photo_date_widget.setLayout(photo_date_row)
        form_layout.addRow(self.make_form_label("Photo type / process"), photo_date_widget)

        # Spectral sensitivity mode + Spectral sensitivity on one row (equal halves)
        spectral_row = QHBoxLayout()
        spectral_row.setSpacing(6)
        spectral_row.setContentsMargins(0, 0, 0, 0)
        spectral_row.addWidget(self.spectral_mode_combo, 1)
        sensitivity_label = self.make_label_with_info(
            "Sensitivity",
            "The historical spectral sensitivity model used by the pipeline. In Auto mode, this is computed from process type and date."
        )
        sensitivity_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spectral_row.addWidget(sensitivity_label)
        spectral_row.addWidget(self.spectral_sensitivity_combo, 1)
        spectral_widget = QWidget()
        spectral_widget.setLayout(spectral_row)
        form_layout.addRow(
            self.make_label_with_info(
                "Spectral sensitivity",
                "Auto recalculates spectral sensitivity based on photo type and date; Manual preserves explicit user choice."
            ),
            spectral_widget,
        )

        self.advanced_settings_button = QPushButton("Advanced Settings...")
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings_dialog)
        self.advanced_settings_button.setMinimumHeight(30)

        self.reset_button = QPushButton("Reset Defaults")
        self.reset_button.clicked.connect(self.reset_form_defaults)
        self.reset_button.setMinimumHeight(30)
        self.reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.reset_button.setToolTip("Reset all main settings and advanced parameters to defaults.")

        advanced_actions_widget = QWidget()
        advanced_actions_layout = QHBoxLayout()
        advanced_actions_layout.setContentsMargins(0, 0, 0, 0)
        advanced_actions_layout.setSpacing(8)
        advanced_actions_widget.setLayout(advanced_actions_layout)
        advanced_actions_layout.addWidget(self.advanced_settings_button)
        advanced_actions_layout.addWidget(self.reset_button)
        advanced_actions_layout.addStretch(1)
        form_layout.addRow(advanced_actions_widget)

        # Add Quality row as proper form row
        form_layout.addRow(self.quality_widget)

        # Detection threshold row
        det_threshold_row = QHBoxLayout()
        det_threshold_row.setContentsMargins(0, 0, 0, 0)
        det_threshold_row.setSpacing(6)
        det_threshold_row.addWidget(self.det_threshold_slider, 1)
        det_threshold_row.addWidget(self.det_threshold_value_label, 0)
        det_threshold_row.addWidget(self.det_threshold_info, 0)
        det_threshold_widget = QWidget()
        det_threshold_widget.setLayout(det_threshold_row)
        form_layout.addRow(self.det_threshold_label, det_threshold_widget)

        # Add Auto-recompose checkbox (full width)
        auto_recompose_widget = QWidget()
        auto_recompose_layout = QHBoxLayout()
        auto_recompose_layout.setContentsMargins(0, 0, 0, 0)
        auto_recompose_layout.setSpacing(0)
        auto_recompose_layout.addWidget(self.auto_recompose_checkbox)
        auto_recompose_layout.addStretch(1)
        auto_recompose_widget.setLayout(auto_recompose_layout)
        form_layout.addRow(auto_recompose_widget)

        # Initialize spectral sensitivity widgets with defaults
        self.photo_type_combo.setCurrentText("Unknown")
        self.approx_date_edit.setText("")
        self.spectral_mode_combo.setCurrentText("Auto")
        self.update_spectral_sensitivity_ui()

        self.update_iteration_label()

        # --- Progress row ---
        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(4)

        progress_bars_layout = QVBoxLayout()
        self.progress_bars_layout = progress_bars_layout
        progress_bars_layout.setContentsMargins(0, 0, 0, 0)
        progress_bars_layout.setSpacing(1)

        self.preprocess_progress_bar = QProgressBar()
        self.preprocess_progress_bar.setRange(0, 100)
        self.preprocess_progress_bar.setValue(0)
        self.preprocess_progress_bar.setVisible(True)
        self.preprocess_progress_bar.setFormat("Preprocess ready")
        self.preprocess_progress_bar.setFixedHeight(16)
        self.preprocess_progress_bar.setStyleSheet("QProgressBar { min-height: 16px; max-height: 16px; border: 1px solid #6d727b; border-radius: 4px; text-align: center; color: #d5d9df; } QProgressBar::groove { background: #2a2f36; border-radius: 4px; } QProgressBar::chunk { background: #2f7be6; border-radius: 4px; margin: 0px; }")
        self.preprocess_progress_bar.setAutoFillBackground(True)
        progress_bars_layout.addWidget(self.preprocess_progress_bar)

        self.rephoto_progress_label = QLabel("Processing")
        self.rephoto_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rephoto_progress_label.setFixedWidth(94)
        self.rephoto_progress_label.setFixedHeight(41)
        self.rephoto_progress_label.setStyleSheet("color: #c7ccd4;")

        self.rephoto_progress_bar = QProgressBar()
        self.rephoto_progress_bar.setRange(0, 100)
        self.rephoto_progress_bar.setValue(0)
        self.rephoto_progress_bar.setVisible(True)
        self.rephoto_progress_bar.setFormat("Waiting... 0% | 0:00")
        self.rephoto_progress_bar.setFixedHeight(24)
        self.rephoto_progress_bar.setStyleSheet("QProgressBar { min-height: 24px; max-height: 24px; border: 1px solid #6d727b; border-radius: 5px; text-align: center; color: #e3e7ec; } QProgressBar::groove { background: #2a2f36; border-radius: 5px; } QProgressBar::chunk { background: #2f7be6; border-radius: 5px; margin: 0px; }")
        self.rephoto_progress_bar.setAutoFillBackground(True)
        progress_bars_layout.addWidget(self.rephoto_progress_bar)

        progress_row.addWidget(self.rephoto_progress_label, 0, Qt.AlignVCenter)
        progress_row.addLayout(progress_bars_layout, 1)

        # Batch queue status label
        self._batch_queue_label = QLabel("")
        self._batch_queue_label.setStyleSheet("color: #8a919c; font-size: 10px;")
        self._batch_queue_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        progress_widget = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(2)
        progress_layout.addLayout(progress_row)
        progress_layout.addWidget(self._batch_queue_label)
        progress_widget.setLayout(progress_layout)
        progress_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self.update_elapsed_label)

        # --- Outputs (Results only) ---
        outputs_group = QGroupBox("Outputs")
        outputs_layout = QFormLayout()
        self.outputs_layout = outputs_layout
        outputs_layout.setVerticalSpacing(3)
        outputs_layout.setContentsMargins(6, 4, 6, 3)
        outputs_group.setLayout(outputs_layout)
        outputs_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        results_row = QHBoxLayout()
        results_row.setContentsMargins(0, 0, 0, 0)
        results_row.setSpacing(6)
        self.results_root_edit = QLineEdit(str(self.repo_root / "results"))
        self.results_browse_button = QPushButton("Browse...")
        self.results_browse_button.clicked.connect(self.browse_results_root)
        results_row.addWidget(self.results_root_edit)
        results_row.addWidget(self.results_browse_button)

        results_widget = QWidget()
        results_widget.setLayout(results_row)
        outputs_layout.addRow("Results folder", results_widget)

        # --- Buttons ---
        button_row = QHBoxLayout()

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.handle_run_button_clicked)
        self.run_button.setShortcut("Ctrl+Return")
        self.run_button.setToolTip("Start the rephotography run (Ctrl+Enter).")
        self.run_button.setAccessibleName("Run rephotography")
        self.run_button.setMinimumHeight(34)
        self.run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.run_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #1a73e8; color: white; border: none; border-radius: 4px; } "
            "QPushButton:hover { background-color: #1455c0; } "
            "QPushButton:pressed { background-color: #0d47a1; } "
            "QPushButton:disabled { "
            "background-color: #2a2f36; color: #8f96a1; border: 1px solid #4a505a; "
            "}"
        )
        # Right-click on Run offers a quick "Preview Crops Only" option.
        self.run_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.run_button.customContextMenuRequested.connect(self._show_run_context_menu)

        secondary_button_style = (
            "QPushButton { border: 1px solid #4a4f57; border-radius: 4px; background-color: #252a31; color: #e6e8eb; } "
            "QPushButton:hover { background-color: #2d333b; } "
            "QPushButton:pressed { background-color: #20252b; } "
            "QPushButton:disabled { color: #8b929c; background-color: #22262c; border: 1px solid #3b4048; }"
        )
        utility_button_style = (
            "QPushButton { border: 1px solid #3f4550; border-radius: 4px; background-color: #20252c; color: #c4cad3; } "
            "QPushButton:hover { background-color: #262b33; } "
            "QPushButton:pressed { background-color: #1d2128; } "
            "QPushButton:disabled { color: #89909a; background-color: #1f2329; border: 1px solid #363c45; }"
        )

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setShortcut("Escape")
        self.cancel_button.setToolTip("Stop the current rephotography run (Esc).")
        self.cancel_button.setAccessibleName("Cancel rephotography")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumHeight(34)
        self.cancel_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cancel_button.setStyleSheet(secondary_button_style)

        self.end_early_button = QPushButton("End Early")
        self.end_early_button.clicked.connect(self.request_end_run_early)
        self.end_early_button.setShortcut("Ctrl+E")
        self.end_early_button.setMinimumHeight(34)
        self.end_early_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.end_early_button.setStyleSheet(secondary_button_style)
        self.end_early_button.setEnabled(False)
        self.end_early_button.setToolTip("Finish this run early using completed iterations only (Ctrl+E).")
        self.end_early_button.setAccessibleName("End run early")

        self.reset_button.setStyleSheet(secondary_button_style)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setMinimumHeight(34)
        self.quit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.quit_button.setStyleSheet(secondary_button_style)

        self.toggle_log_button = QPushButton("Hide Log" if self.log_visible else "Show Log")
        self.toggle_log_button.setMinimumHeight(34)
        self.toggle_log_button.setMinimumWidth(102)
        self.toggle_log_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_log_button.setStyleSheet(utility_button_style)

        button_row.addWidget(self.run_button)
        button_row.addWidget(self.end_early_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.quit_button)
        button_row.addWidget(self.toggle_log_button)

        # --- Previews (Input on left, Result on right) ---
        previews_group = QGroupBox("Previews")
        previews_layout = QVBoxLayout()
        previews_layout.setContentsMargins(6, 6, 6, 6)
        previews_layout.setSpacing(6)
        previews_group.setLayout(previews_layout)
        previews_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        preview_top_row = QSplitter(Qt.Horizontal)
        preview_top_row.setChildrenCollapsible(False)
        preview_top_row.setHandleWidth(6)
        preview_top_row.setStyleSheet(
            "QSplitter::handle { background-color: #3a3f47; border-radius: 2px; }"
        )
        preview_top_row.splitterMoved.connect(self._on_splitter_moved)

        input_group = QWidget()
        input_layout = QVBoxLayout()
        self.input_preview_layout = input_layout
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)
        input_group.setLayout(input_layout)
        input_header = QHBoxLayout()
        input_header.setContentsMargins(0, 0, 0, 0)
        input_header.setSpacing(6)
        input_title = QLabel("Input Preview")
        input_title.setStyleSheet("color: #c7ccd4;")
        input_divider = QFrame()
        input_divider.setFrameShape(QFrame.HLine)
        input_divider.setFrameShadow(QFrame.Sunken)
        input_divider.setStyleSheet("color: #3a3f47;")
        input_header.addWidget(input_title)
        input_header.addWidget(input_divider, 1)
        input_layout.addLayout(input_header)

        self.input_preview_label = InputDropLabel(self)
        self.input_preview_label.setAlignment(Qt.AlignCenter)
        self.input_preview_label.setFixedHeight(300)
        self.input_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.input_preview_label)
        self.input_detect_overlay = InputDetectOverlay(self.input_preview_label)
        # Re-detect faces button (with adjusted threshold)
        self.redetect_faces_button = QPushButton("Re-detect Faces (with current threshold)")
        self.redetect_faces_button.setFixedHeight(32)
        self.redetect_faces_button.setShortcut("Ctrl+D")
        self.redetect_faces_button.setToolTip(
            "Run face detection again using the current threshold setting (Ctrl+D). "
            "Use a lower threshold to detect more faces, higher to detect only clear faces."
        )
        self.redetect_faces_button.setAccessibleName("Re-detect faces")
        self.redetect_faces_button.clicked.connect(self.run_face_detection)
        self.redetect_faces_button.setEnabled(False)
        self.redetect_faces_button.setVisible(False)
        input_layout.addWidget(self.redetect_faces_button)

        self.input_preview_footer_spacer = QWidget()
        self.input_preview_footer_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_preview_footer_spacer.setFixedHeight(32)
        self.input_preview_footer_spacer.setVisible(True)
        input_layout.addWidget(self.input_preview_footer_spacer)

        result_group = QWidget()
        result_layout = QVBoxLayout()
        self.result_preview_layout = result_layout
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(4)
        result_group.setLayout(result_layout)
        result_header = QHBoxLayout()
        result_header.setContentsMargins(0, 0, 0, 0)
        result_header.setSpacing(6)
        result_title = QLabel("Result Preview")
        result_title.setStyleSheet("color: #c7ccd4;")
        result_divider = QFrame()
        result_divider.setFrameShape(QFrame.HLine)
        result_divider.setFrameShadow(QFrame.Sunken)
        result_divider.setStyleSheet("color: #3a3f47;")
        self.result_view_toggle = QPushButton("Rephoto")
        self.result_view_toggle.setCheckable(True)
        self.result_view_toggle.setChecked(False)
        self.result_view_toggle.setFixedHeight(20)
        self.result_view_toggle.setMinimumWidth(90)
        self.result_view_toggle.setToolTip("Toggle between Rephoto result and Recomposited result in the preview and before/after swiper.")
        self.result_view_toggle.setStyleSheet(
            "QPushButton { background: #2a2f38; color: #9aa0aa; border: 1px solid #565c66; "
            "border-radius: 3px; padding: 1px 8px; font-size: 11px; }"
            "QPushButton:checked { background: #3a5f3a; color: #c7e6c7; border-color: #5a8a5a; }"
            "QPushButton:hover { border-color: #8090a0; }"
        )
        self.result_view_toggle.clicked.connect(self._on_result_view_toggle)
        self.result_view_toggle.setVisible(False)
        self._rephoto_result_path = None
        self._recomposited_result_path = None

        result_header.addWidget(result_title)
        result_header.addWidget(self.result_view_toggle)
        result_header.addWidget(result_divider, 1)
        result_layout.addLayout(result_header)

        self.result_preview_label = ResultPreviewLabel(self, "No result image yet.\nRun to generate a preview.")
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setFixedHeight(300)
        self.result_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.result_preview_label.setStyleSheet("border: 1px solid #868b94; border-radius: 4px; color: #b7bcc5;")
        self.result_preview_label.setMouseTracking(True)
        self.result_preview_label.installEventFilter(self)
        self._compare_wipe_active = False
        self._compare_wipe_last_pos = None  # Throttle: min Manhattan distance before scaling
        # Cache the scaled before/after pixmaps so the 60+Hz mouse-move loop
        # rescales only when the label size or source pixmap actually changes.
        self._compare_wipe_result_scaled_key = None
        self._compare_wipe_result_scaled = None
        self._compare_wipe_input_scaled_key = None
        self._compare_wipe_input_scaled = None
        result_layout.addWidget(self.result_preview_label)

        # Stage overlay label for animated stage indicator
        self.result_stage_overlay = QLabel(self.result_preview_label)
        self.result_stage_overlay.setVisible(False)
        self.result_stage_overlay.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.result_stage_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150); color: white; "
            "padding: 4px 8px; border-radius: 6px; font-size: 12px;"
        )
        # Overlay state variables
        self.result_stage_base_text = ""
        self._paused_result_stage_base_text = ""
        self.result_stage_dot_count = 0
        self._result_stage_timer = QTimer(self)
        self._result_stage_timer.setInterval(450)
        self._result_stage_timer.timeout.connect(self.update_result_stage_overlay_animation)

        self.face_preview_header = QWidget()
        face_header_layout = QHBoxLayout()
        face_header_layout.setContentsMargins(0, 0, 0, 0)
        face_header_layout.setSpacing(5)
        self.face_preview_header.setLayout(face_header_layout)

        self.face_preview_summary_label = QLabel("Faces: none")
        self.face_preview_summary_label.setStyleSheet("color: #aeb4be;")
        self.face_preview_summary_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.face_preview_summary_label.setMinimumWidth(0)
        face_header_layout.addWidget(self.face_preview_summary_label, 1, Qt.AlignVCenter)

        self.face_select_all_button = QPushButton("All")
        self.face_select_all_button.setMinimumHeight(20)
        self.face_select_all_button.setMaximumHeight(20)
        self.face_select_all_button.setMinimumWidth(44)
        self.face_select_all_button.clicked.connect(lambda: self.face_strip.set_all_faces_selected(True))
        face_header_layout.addWidget(self.face_select_all_button, 0, Qt.AlignVCenter)

        self.face_select_none_button = QPushButton("None")
        self.face_select_none_button.setMinimumHeight(20)
        self.face_select_none_button.setMaximumHeight(20)
        self.face_select_none_button.setMinimumWidth(50)
        self.face_select_none_button.clicked.connect(lambda: self.face_strip.set_all_faces_selected(False))
        face_header_layout.addWidget(self.face_select_none_button, 0, Qt.AlignVCenter)

        self.face_preview_auto_follow_checkbox = QCheckBox("Auto-follow latest")
        self.face_preview_auto_follow_checkbox.setChecked(True)
        self.face_preview_auto_follow_checkbox.setStyleSheet("color: #c7ccd4;")
        self.face_preview_auto_follow_checkbox.toggled.connect(self.face_strip.handle_face_auto_follow_toggled)
        face_header_layout.addWidget(self.face_preview_auto_follow_checkbox, 0, Qt.AlignVCenter)
        self.face_preview_header.setFixedHeight(22)

        # --- Filmstrip widget (sprocket-hole chrome wrapping a scroll area) ---
        sprocket_band = FilmstripContainerWidget.SPROCKET_BAND
        filmstrip_h = 142
        self.face_preview_strip_filmstrip = FilmstripContainerWidget()
        self.face_preview_strip_filmstrip.setMinimumHeight(filmstrip_h)
        self.face_preview_strip_filmstrip.setMaximumHeight(filmstrip_h)

        self.filmstrip_inner_layout = QVBoxLayout(self.face_preview_strip_filmstrip)
        self.filmstrip_inner_layout.setContentsMargins(0, sprocket_band, 0, sprocket_band)
        self.filmstrip_inner_layout.setSpacing(0)

        self.face_preview_strip_scroll = QScrollArea()
        self.face_preview_strip_scroll.setWidgetResizable(True)
        self.face_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.face_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.face_preview_strip_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
            "QScrollBar:horizontal { height: 5px; background: #1c2028; }"
            "QScrollBar::handle:horizontal { background: #3d4450; border-radius: 2px; min-width: 30px; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
            "QScrollBar:vertical { width: 5px; background: #1c2028; }"
            "QScrollBar::handle:vertical { background: #3d4450; border-radius: 2px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.filmstrip_inner_layout.addWidget(self.face_preview_strip_scroll)

        self.face_preview_strip_container = QWidget()
        self.face_preview_strip_container.setStyleSheet("background: transparent;")
        self.face_preview_strip_layout = QHBoxLayout()
        self.face_preview_strip_layout.setContentsMargins(4, 2, 4, 2)
        self.face_preview_strip_layout.setSpacing(5)
        self.face_preview_strip_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.face_preview_strip_container.setLayout(self.face_preview_strip_layout)
        self.face_preview_strip_container.setMouseTracking(True)
        self.face_preview_strip_scroll.setWidget(self.face_preview_strip_container)
        self.face_preview_strip_scroll.viewport().setMouseTracking(True)
        self.face_preview_strip_scroll.viewport().installEventFilter(self)

        # Filmstrip is always visible (shows empty state when no image loaded)
        self.face_preview_header.setVisible(True)
        self.face_preview_strip_filmstrip.setVisible(True)

        result_buttons_row = QHBoxLayout()
        result_buttons_row.setContentsMargins(0, 0, 0, 0)
        result_buttons_row.setSpacing(6)

        self.open_image_location_button = QPushButton("Open Image Location")
        self.open_image_location_button.clicked.connect(self.open_result_image_location)
        self.open_image_location_button.setEnabled(False)
        self.open_image_location_button.setMinimumHeight(32)
        self.open_image_location_button.setStyleSheet(secondary_button_style)
        result_buttons_row.addWidget(self.open_image_location_button)

        self.recomposite_button = QPushButton("Recomposite Original")
        self.recomposite_button.setToolTip(
            "Apply Photoshop-style Color blend: hue/saturation from the result, luminance from the original input."
        )
        self.recomposite_button.clicked.connect(self.run_recomposite)
        self.recomposite_button.setEnabled(False)
        self.recomposite_button.setMinimumHeight(32)
        self.recomposite_button.setStyleSheet(secondary_button_style)
        result_buttons_row.addWidget(self.recomposite_button)

        result_layout.addLayout(result_buttons_row)

        self.input_preview_group = input_group
        self.result_preview_group = result_group
        preview_top_row.addWidget(input_group)
        preview_top_row.addWidget(result_group)
        preview_top_row.setStretchFactor(0, 1)
        preview_top_row.setStretchFactor(1, 1)
        previews_layout.addWidget(preview_top_row)

        self.face_preview_panel = QWidget()
        face_panel_layout = QVBoxLayout()
        face_panel_layout.setContentsMargins(0, 0, 0, 0)
        face_panel_layout.setSpacing(3)
        self.face_preview_panel.setLayout(face_panel_layout)
        self.face_selection_notice_label = QLabel("Select one or more faces to continue")
        self.face_selection_notice_label.setWordWrap(True)
        self.face_selection_notice_label.setAlignment(Qt.AlignCenter)
        self.face_selection_notice_label.setStyleSheet(
            "QLabel { "
            "color: #e6f4ff; "
            "background-color: #21476f; "
            "border: 1px solid #4b88bc; "
            "border-radius: 4px; "
            "padding: 6px 8px; "
            "font-weight: 600; "
            "}"
        )
        self.face_selection_notice_label.setVisible(False)
        face_panel_layout.addWidget(self.face_selection_notice_label)
        face_panel_layout.addWidget(self.face_preview_header)
        face_panel_layout.addWidget(self.face_preview_strip_filmstrip)
        # Filmstrip panel is always visible
        self.face_preview_panel.setVisible(True)
        previews_layout.addWidget(self.face_preview_panel)

        # Settings section with input image and form layout
        settings_container = QWidget()
        settings_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        settings_layout = QVBoxLayout()
        self.settings_layout = settings_layout
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_container.setLayout(settings_layout)
        settings_layout.addWidget(QLabel("Input Image"))
        settings_layout.addLayout(input_row)
        settings_layout.addLayout(form_layout)

        self.controls_container = QWidget()
        controls_layout = QVBoxLayout()
        self.controls_layout = controls_layout
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setAlignment(Qt.AlignTop)
        self.controls_container.setLayout(controls_layout)
        controls_layout.addWidget(settings_container)
        controls_layout.addWidget(progress_widget)
        controls_layout.addWidget(outputs_group)

        self.button_bar_widget = QWidget()
        self.button_bar_widget.setLayout(button_row)
        self.button_bar_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        controls_layout.addWidget(self.button_bar_widget)

        # --- Log container ---
        self.log_container = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)
        self.log_container.setLayout(log_layout)

        log_header_row = QHBoxLayout()
        self.log_title_label = QLabel("Log Output")
        self.log_title_label.setStyleSheet("color: #b7bcc5;")
        log_header_row.addWidget(self.log_title_label)
        log_header_row.addStretch()
        self.expand_log_button = QPushButton("Expand Log")
        self.expand_log_button.setStyleSheet(secondary_button_style)
        log_header_row.addWidget(self.expand_log_button)

        log_layout.addLayout(log_header_row)

        self.log_box = TimestampedLogBox()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)
        self.log_box.setMaximumHeight(150)
        self.log_box.document().setMaximumBlockCount(4000)
        self.log_box.append("GUI loaded successfully.")
        log_layout.addWidget(self.log_box)

        self.status_label = QLabel("Status: Ready")
        log_layout.addWidget(self.status_label)

        self.expand_log_button.clicked.connect(self.toggle_log_size)
        self.toggle_log_button.clicked.connect(self.toggle_log_visibility)

        self.expand_log_button.setEnabled(self.log_visible)
        self.log_container.setVisible(self.log_visible)
        controls_layout.addWidget(self.log_container)

        self.setup_menu_bar()


    def _finalize_init(self):
        """Run after _build_ui: install the responsive layout, run
        first-paint initial-state pokes, and schedule the deferred
        startup tasks (preflight, Haar cascade warmup).
        """
        # --- Responsive page layout ---
        self.previews_group = previews_group
        self.previews_layout = previews_layout
        self.previews_splitter = preview_top_row
        self.main_layout = main_layout
        self.content_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        main_layout.addLayout(self.content_layout)
        self._wide_layout_active = None
        self.apply_responsive_layout(force=True)

        # Initial state
        self.update_mode_controls()
        self.update_runtime_label()
        if not self.gfpgan_is_available():
            self.log_box.append("GFPGAN not found (deps\\GFPGAN). Enhancement is disabled.")
        else:
            self.log_box.append("GFPGAN found. Enhancement is available.")
        # Settings are no longer persisted between sessions; always start with defaults.
        # Defer startup checks until after first paint to improve perceived launch responsiveness.
        QTimer.singleShot(0, lambda: self.preflight.run_startup_preflight(show_dialog=True))
        # Pre-warm the Haar cascade detector in the background so the first face
        # probe doesn't pay the ~1.5s cold-start penalty.
        QTimer.singleShot(200, self._warm_up_haar_detector)

    # ------------------------------
    # Qt / window events
    # ------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_responsive_layout()
        self._update_wide_preview_dimensions()
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()
        self.position_input_detect_overlay()
        self.position_result_stage_overlay()

    def _on_splitter_moved(self, pos, index):
        """Rescale preview images when the user drags the splitter handle."""
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()
        self.position_input_detect_overlay()
        self.position_result_stage_overlay()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self.apply_responsive_layout)

    def apply_responsive_layout(self, force=False):
        if not hasattr(self, "content_layout"):
            return

        use_wide_layout = self.isFullScreen() or self.width() >= WIDE_LAYOUT_MIN_WIDTH
        if not force and self._wide_layout_active == use_wide_layout:
            return
        self._wide_layout_active = use_wide_layout

        while self.content_layout.count() > 0:
            self.content_layout.takeAt(0)

        self.face_strip._rehost_face_preview_panel(use_wide_layout)

        if use_wide_layout:
            self.face_strip._configure_face_preview_panel_for_mode(True)

            self.content_layout.setDirection(QBoxLayout.LeftToRight)
            self.previews_splitter.setOrientation(Qt.Vertical)
            self.previews_splitter.setHandleWidth(6)
            # Fully fluid: labels expand in both directions so they
            # continuously resize with the window.  The content_layout
            # stretch factors (controls=2, previews=1) give the preview
            # pane ~1/3 of the window width; the vertical splitter
            # distributes height between input and result.
            self.input_preview_label.setMinimumWidth(120)
            self.input_preview_label.setMaximumWidth(16777215)
            self.input_preview_label.setMinimumHeight(80)
            self.input_preview_label.setMaximumHeight(16777215)
            self.input_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.result_preview_label.setMinimumWidth(120)
            self.result_preview_label.setMaximumWidth(16777215)
            self.result_preview_label.setMinimumHeight(80)
            self.result_preview_label.setMaximumHeight(16777215)
            self.result_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # Containers expand so the splitter can resize them.
            self.input_preview_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.result_preview_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # Labels get stretch=1 to fill container height. Do NOT set
            # alignment — Qt's setAlignment on a layout item caps the widget
            # at its sizeHint, defeating the stretch.
            if hasattr(self, "input_preview_layout"):
                self.input_preview_layout.setStretch(1, 1)
                self.input_preview_layout.setAlignment(self.input_preview_label, Qt.Alignment())
            if hasattr(self, "result_preview_layout"):
                self.result_preview_layout.setStretch(1, 1)
                self.result_preview_layout.setAlignment(self.result_preview_label, Qt.Alignment())
            if hasattr(self, "input_preview_footer_spacer"):
                self.input_preview_footer_spacer.setVisible(False)
            self.controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.previews_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.previews_group.setMinimumWidth(200)
            self.previews_group.setMaximumWidth(16777215)
            self.previews_group.setMinimumHeight(0)
            self.previews_group.setMaximumHeight(16777215)
            self.content_layout.addWidget(self.controls_container, 2)
            self.content_layout.addWidget(self.face_preview_panel, 0)
            self.content_layout.addWidget(self.previews_group, 1)
        else:
            self.face_strip._configure_face_preview_panel_for_mode(False)
            self.content_layout.setDirection(QBoxLayout.TopToBottom)
            self.previews_splitter.setOrientation(Qt.Horizontal)
            self.previews_splitter.setHandleWidth(6)
            # Clear fixedWidth from wide mode (setFixedWidth sets both min+max).
            self.input_preview_label.setMinimumWidth(100)
            self.input_preview_label.setMaximumWidth(16777215)
            self.input_preview_label.setMinimumHeight(0)
            self.input_preview_label.setFixedHeight(300)
            self.input_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.result_preview_label.setMinimumWidth(100)
            self.result_preview_label.setMaximumWidth(16777215)
            self.result_preview_label.setMinimumHeight(0)
            self.result_preview_label.setFixedHeight(300)
            self.result_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.input_preview_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.result_preview_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            if hasattr(self, "input_preview_layout"):
                self.input_preview_layout.setStretch(1, 0)  # reset label stretch
                self.input_preview_layout.setAlignment(self.input_preview_label, Qt.Alignment())
            if hasattr(self, "result_preview_layout"):
                self.result_preview_layout.setStretch(1, 0)  # reset label stretch
                self.result_preview_layout.setAlignment(self.result_preview_label, Qt.Alignment())
            if hasattr(self, "input_preview_footer_spacer"):
                if hasattr(self, "open_image_location_button"):
                    self.input_preview_footer_spacer.setFixedHeight(max(24, self.open_image_location_button.minimumHeight()))
                self.input_preview_footer_spacer.setVisible(True)
            self.controls_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.previews_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.previews_group.setMinimumWidth(0)
            self.previews_group.setMaximumWidth(16777215)
            self.previews_group.setMinimumHeight(0)
            self.previews_group.setMaximumHeight(16777215)
            self.content_layout.addWidget(self.previews_group, 0)
            self.content_layout.addWidget(self.controls_container, 0)

        self._apply_mode_layout_profile(use_wide_layout)
        # Always render filmstrip to show placeholders even when empty
        self.face_strip.render_face_preview_strip()
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()
        self.position_input_detect_overlay()

    def _apply_mode_layout_profile(self, use_wide_layout):
        if use_wide_layout:
            self.controls_layout.setSpacing(4)
            self.settings_layout.setSpacing(3)
            self.form_layout.setVerticalSpacing(4)
            self.progress_bars_layout.setSpacing(1)
            self.outputs_layout.setVerticalSpacing(1)
            self.outputs_layout.setContentsMargins(6, 2, 6, 2)
        else:
            self.controls_layout.setSpacing(8)
            self.settings_layout.setSpacing(6)
            self.form_layout.setVerticalSpacing(6)
            self.progress_bars_layout.setSpacing(1)
            self.outputs_layout.setVerticalSpacing(3)
            self.outputs_layout.setContentsMargins(6, 4, 6, 3)

    # ------------------------------
    # UI state / control updates
    # ------------------------------
    def gfpgan_is_available(self):
        cached = getattr(self, "_gfpgan_available_cache", None)
        if cached is not None:
            ts, val = cached
            if (time.time() - ts) < 60.0:
                return val
        result = self.resolve_resource_path(Path("deps") / "GFPGAN").exists()
        self._gfpgan_available_cache = (time.time(), result)
        return result

    
    def update_mode_controls(self):
        """Update UI state based on GFPGAN availability and crop-only mode."""
        crop_only = self.advanced_dialog.crop_only_checkbox.isChecked()
        gfpgan_available = self.gfpgan_is_available()

        # If GFPGAN is not available: enhancement must be disabled
        if not gfpgan_available:
            self.advanced_dialog.use_gfpgan_checkbox.setChecked(True)  # Checked = disabled
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
            self.advanced_dialog.gfpgan_blend_edit.setEnabled(False)
            if hasattr(self, "run_button"):
                self.face_strip.update_run_button_for_quick_face_hint()
            return

        # If crop-only mode: enhancement must be disabled
        if crop_only:
            self.advanced_dialog.use_gfpgan_checkbox.setChecked(True)  # Checked = disabled
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
            self.advanced_dialog.gfpgan_blend_edit.setEnabled(False)
            if hasattr(self, "run_button"):
                self.face_strip.update_run_button_for_quick_face_hint()
            return

        # Otherwise: enhancement can be toggled
        self.advanced_dialog.use_gfpgan_checkbox.setEnabled(True)
        # Update blend spinner based on checkbox state
        enhancement_enabled = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
        self.advanced_dialog.gfpgan_blend_edit.setEnabled(enhancement_enabled)
        if hasattr(self, "run_button"):
            self.face_strip.update_run_button_for_quick_face_hint()

    def update_iteration_mode(self):
        current = self.iter_values[self.iter_slider.value()]
        self.iter_slider.setMaximum(len(self.iter_values) - 1)

        closest_index = min(range(len(self.iter_values)), key=lambda i: abs(self.iter_values[i] - current))
        self.iter_slider.setValue(closest_index)

    def get_elapsed_display_text(self):
        if self.rephoto_started_at is None:
            return "0:00"

        elapsed = int(time.time() - self.rephoto_started_at)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def update_rephoto_bar_format(self):
        elapsed_text = self.get_elapsed_display_text()
        face_prefix = ""
        if self.current_run_phase == "rephoto" and self.rephoto_face_total > 1:
            shown_index = max(1, min(self.rephoto_face_current_index, self.rephoto_face_total))
            face_prefix = f"{shown_index}/{self.rephoto_face_total} "
        self.rephoto_progress_bar.setFormat(f"{face_prefix}{self.rephoto_status_text} %p% | {elapsed_text}")

    def get_effective_rephoto_steps(self):
        preset = self.get_selected_preset_value()
        if preset == 375:
            return (125, 375)
        elif preset == 750:
            return (250, 750)
        else:
            return (250, preset)

    _BASE_WINDOW_TITLE = "Time-Travel Rephotography"

    def _update_title_bar_progress(self, percent, status_text):
        """Show progress in the window title so it's visible from the taskbar."""
        phase = getattr(self, "current_run_phase", "idle")
        if phase in ("rephoto",) and percent > 0:
            elapsed = self.get_elapsed_display_text()
            self.setWindowTitle(f"{percent}% | {elapsed} \u2014 {self._BASE_WINDOW_TITLE}")
        elif phase == "preprocess":
            self.setWindowTitle(f"Preprocessing\u2026 \u2014 {self._BASE_WINDOW_TITLE}")
        else:
            self.setWindowTitle(self._BASE_WINDOW_TITLE)

    def start_rephoto_progress_tracking(self):
        self.rephoto_step_pair = self.get_effective_rephoto_steps()
        self.rephoto_total_work = self.rephoto_step_pair[0] + self.rephoto_step_pair[1]
        self.rephoto_stage = None
        self.rephoto_stage_name = None
        self.rephoto_stage_current = 0
        self.rephoto_stage_total = 0
        self.rephoto_total_done_before_stage = 0
        selected_count = len(self.face_strip.get_selected_face_indices())
        self.rephoto_face_total = max(1, selected_count)
        self.rephoto_face_current_index = 0
        self._last_iter_progress_signature = None

    def update_rephoto_progress_from_iteration(self, current_iter, total_iter):
        if not self.rephoto_stage_name:
            return
        signature = (
            self.rephoto_stage_name,
            int(current_iter),
            int(total_iter),
            int(self.rephoto_face_current_index),
        )
        if self._last_iter_progress_signature == signature:
            return
        self._last_iter_progress_signature = signature
        if self.rephoto_stage_name == "32x32":
            self.rephoto_total_done_before_stage = 0
            self.rephoto_stage_total = self.rephoto_step_pair[0]
        elif self.rephoto_stage_name == "64x64":
            self.rephoto_total_done_before_stage = self.rephoto_step_pair[0]
            self.rephoto_stage_total = self.rephoto_step_pair[1]
        overall_done = self.rephoto_total_done_before_stage + current_iter
        percent = round(100 * overall_done / self.rephoto_total_work)
        self.set_rephoto_progress(percent, "Processing")

    def update_iteration_label(self):
        v = self.iter_values[self.iter_slider.value()]
        label = QUALITY_PRESET_LABELS.get(v)
        if label:
            self.quality_value_label.setText(f"{v}")
            default_suffix = " (default)" if v == DEFAULT_ITERATION else ""
            self.quality_default_label.setText(f"\u2014 {label}{default_suffix}")
            self.quality_default_label.setVisible(True)
        else:
            self.quality_value_label.setText(str(v))
            self.quality_default_label.setVisible(False)

        if hasattr(self, "runtime_label") and hasattr(self, "update_runtime_label"):
            self.update_runtime_label()

    def _update_detection_threshold(self):
        """Update detection threshold display and sync with advanced dialog."""
        val = self.det_threshold_slider.value() / 100.0
        self.det_threshold_value_label.setText(f"{val:.2f}")
        if hasattr(self, "advanced_dialog"):
            self.advanced_dialog.det_threshold_edit.setValue(val)

    def _sync_auto_recompose_from_adv(self):
        """Sync auto-recompose checkbox from advanced dialog to main form."""
        self.auto_recompose_checkbox.blockSignals(True)
        self.auto_recompose_checkbox.setChecked(self.advanced_dialog.auto_recompose_checkbox_adv.isChecked())
        self.auto_recompose_checkbox.blockSignals(False)

    def _sync_auto_recompose_from_main(self):
        """Sync auto-recompose checkbox from main form to advanced dialog."""
        self.advanced_dialog.auto_recompose_checkbox_adv.blockSignals(True)
        self.advanced_dialog.auto_recompose_checkbox_adv.setChecked(self.auto_recompose_checkbox.isChecked())
        self.advanced_dialog.auto_recompose_checkbox_adv.blockSignals(False)

    def _update_batch_queue_status(self):
        """Update batch processing queue status label."""
        if not hasattr(self, "_batch_queue_label"):
            return
        entries = self.face_strip.face_preview_entries
        if not entries:
            self._batch_queue_label.setText("")
            return
        queued = sum(1 for e in entries if e.get("status") == "queued")
        running = sum(1 for e in entries if e.get("status") == "running")
        done = sum(1 for e in entries if e.get("status") == "done")
        failed = sum(1 for e in entries if e.get("status") == "failed")
        total = len(entries)

        if queued > 0:
            status_text = f"Queue: {done}/{total} done • {queued} pending • {failed} failed" if failed else f"Queue: {done}/{total} done • {queued} pending"
        elif running > 0:
            status_text = f"Queue: {done}/{total} done • 1 processing • {failed} failed" if failed else f"Queue: {done}/{total} done • 1 processing"
        else:
            status_text = f"Queue: {done}/{total} done" + (f" • {failed} failed" if failed else "")

        self._batch_queue_label.setText(status_text)

    def parse_approximate_year(self, text):
        """Convert flexible user date text into an approximate integer year, or None."""
        text = text.strip().lower()
        if not text:
            return None

        # Check for 4-digit year
        m = YEAR_RE.search(text)
        if m:
            year = int(m.group(1))
            # If pattern like "1890s", convert to midpoint
            if "s" in text[m.end():m.end()+1]:
                return year + 5
            return year
        return None

    def infer_spectral_sensitivity(self):
        """Infer spectral sensitivity based on photo type and date. Returns one of:
        'Blue-sensitive', 'Orthochromatic', 'Panchromatic'
        """
        photo_type = self.photo_type_combo.currentText()
        approx_year = self.parse_approximate_year(self.approx_date_edit.text())

        # PRIORITY 1: True photographic processes
        if self.is_true_process_type(photo_type):
            return "Blue-sensitive"

        # PRIORITY 3: Format-based types
        if photo_type == "Carte de visite (CDV)":
            return "Blue-sensitive"

        if photo_type == "Cabinet card":
            if approx_year is None:
                return "Blue-sensitive"
            if approx_year < 1890:
                return "Blue-sensitive"
            else:  # 1890 <= year
                return "Orthochromatic"

        if photo_type == "Late cabinet card / dry plate studio portrait":
            if approx_year is None:
                return "Orthochromatic"
            if approx_year < 1880:
                return "Blue-sensitive"
            elif approx_year <= 1919:
                return "Orthochromatic"
            else:  # >= 1920
                return "Panchromatic"

        if photo_type == "Early gelatin silver print":
            if approx_year is None:
                return "Orthochromatic"
            if approx_year < 1920:
                return "Orthochromatic"
            else:  # >= 1920
                return "Panchromatic"

        if photo_type == "Black-and-white snapshot / roll-film print":
            if approx_year is None:
                return "Panchromatic"
            if approx_year < 1920:
                return "Orthochromatic"
            else:  # >= 1920
                return "Panchromatic"

        # PRIORITY 4: Unknown with date-only logic
        if approx_year is not None:
            if approx_year < 1880:
                return "Blue-sensitive"
            elif approx_year <= 1919:
                return "Orthochromatic"
            else:  # >= 1920
                return "Panchromatic"

        # PRIORITY 5: Fallback
        return "Orthochromatic"

    def _update_date_placeholder_for_photo_type(self, photo_type_text):
        """Show a date-range hint in the date field when a photo type is selected."""
        hint = PHOTO_TYPE_DATE_HINTS.get(photo_type_text, PHOTO_TYPE_DATE_HINTS["Unknown"])
        self.approx_date_edit.setPlaceholderText(hint)

    def update_spectral_sensitivity_ui(self):
        """Update spectral sensitivity combo based on mode and infer if Auto."""
        mode = self.spectral_mode_combo.currentText()

        if mode == "Auto":
            self.spectral_sensitivity_combo.setEnabled(False)
            inferred = self.infer_spectral_sensitivity()
            self.spectral_sensitivity_combo.setCurrentText(inferred)
        else:  # Manual
            self.spectral_sensitivity_combo.setEnabled(True)
            # do not overwrite user choice

    def can_select_new_image(self, show_message=False):
        reason = ""
        if self.process is not None:
            reason = "A run is currently in progress."

        if reason:
            if show_message:
                self.status_label.setText(f"Status: {reason}")
                self.log_box.append(f"Image import blocked: {reason}")
            return False
        return True

    def update_image_import_controls(self, force_running=None):
        if force_running is None:
            can_select = self.can_select_new_image(show_message=False)
        else:
            can_select = (not force_running)
        self.browse_button.setEnabled(can_select)
        self.input_image_edit.setEnabled(can_select)

    # ------------------------------
    # Progress tracking
    # ------------------------------
    def _try_fast_rephoto_iteration_progress(self, text: str):
        if self.current_run_phase != "rephoto":
            return False
        if "/" not in text:
            return False
        m = ITER_PROGRESS_RE.search(text)
        if not m:
            return False
        current = int(m.group(1))
        total = int(m.group(2))
        allowed_totals = {self.rephoto_step_pair[0], self.rephoto_step_pair[1]}
        if total not in allowed_totals:
            return False
        if total == self.rephoto_step_pair[0]:
            self.rephoto_stage = "32x32"
            self.rephoto_stage_name = "32x32"
        elif total == self.rephoto_step_pair[1]:
            self.rephoto_stage = "64x64"
            self.rephoto_stage_name = "64x64"
        self.rephoto_stage_total = total
        self.update_rephoto_progress_from_iteration(current, total)
        return True

    # ------------------------------
    # Input / output selection
    # ------------------------------
    def _reset_main_window_for_new_input(self):
        """Clear selection/runtime preview state when user imports a new input image."""
        self.stop_quick_face_probe()
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()
        self.quick_face_count_estimate = None
        self._quick_probe_det_threshold = None
        self._last_backend_error_detail = ""
        self._template_match_cache = {}
        self.auto_detect_faces_armed_input = None
        self.auto_detect_faces_triggered_input = None
        self.suppress_preprocess_ui_until_rephoto = False
        if hasattr(self, "input_image_edit"):
            self.input_image_edit.setText("")
        self.set_input_detect_overlay(False)
        self.face_strip.reset_face_preview_state(preserve_input_overlays=False)
        self.clear_result_stage_overlay()
        self.reset_progress_bars()
        self._reset_wrapper_runtime_tracking()
        self.current_run_summary_context = None
        self.set_result_preview_image(None)
        self.result_preview_label.setText("No result image yet.\nRun to generate a preview.")
        self._rephoto_result_path = None
        self._recomposited_result_path = None
        self._compare_wipe_last_pos = None  # Reset wipe throttle
        self._compare_wipe_result_scaled_key = None
        self._compare_wipe_result_scaled = None
        self._compare_wipe_input_scaled_key = None
        self._compare_wipe_input_scaled = None
        self.result_view_toggle.setVisible(False)
        self.result_view_toggle.setChecked(False)
        self.result_view_toggle.setText("Rephoto")
        self.face_strip.update_run_button_for_quick_face_hint()

    def reset_form_defaults(self):
        answer = QMessageBox.question(
            self,
            "Reset Defaults",
            "Reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.status_label.setText("Status: Reset canceled")
            return

        self.stop_quick_face_probe()
        self.set_input_detect_overlay(False)
        self.suppress_preprocess_ui_until_rephoto = False
        # Reset all advanced-settings values to their defaults.
        self.advanced_dialog.restore_defaults()

        self.iter_values = self.basic_iter_values
        self.iter_slider.setMaximum(len(self.iter_values) - 1)
        self.iter_slider.setValue(self.basic_iter_values.index(DEFAULT_ITERATION))
        self.update_iteration_label()

        self.photo_type_combo.setCurrentText("Unknown")
        self.approx_date_edit.setText("")
        self.spectral_mode_combo.setCurrentText("Auto")
        self.update_spectral_sensitivity_ui()

        self.results_root_edit.setText(str(self.repo_root / "results"))
        self.update_mode_controls()
        self.update_runtime_label()
        self.face_strip.update_run_button_for_quick_face_hint()

        self.log_box.append("Defaults restored.")
        self.status_label.setText("Status: Defaults restored")

    def set_selected_input_image(self, file_path):
        if not self.can_select_new_image(show_message=True):
            return
        was_awaiting = bool(self.awaiting_face_selection)
        self._reset_main_window_for_new_input()
        self.input_image_edit.setText(file_path)
        current_key = self._normalized_path_key(file_path)
        self.auto_detect_faces_armed_input = current_key
        self.auto_detect_faces_triggered_input = None
        if was_awaiting:
            self.log_box.append("Face selection canceled: new image imported.")
        self.log_box.append(f"Selected image: {file_path}")
        self.status_label.setText("Status: Image selected")
        self.set_input_preview_image(Path(file_path))
        self.refresh_quick_face_hint_from_input()

    def browse_for_image(self):
        if not self.can_select_new_image(show_message=True):
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if file_path:
            self.set_selected_input_image(file_path)

    def browse_results_root(self):
        start_dir = self.results_root_edit.text().strip() or str(self.repo_root)
        dir_path = QFileDialog.getExistingDirectory(self, "Select Results Folder", start_dir)
        if dir_path:
            self.results_root_edit.setText(dir_path)
            self.log_box.append(f"Results folder set: {dir_path}")

    def open_advanced_settings_dialog(self):
        if self.process is not None:
            self.log_box.append("Cannot change advanced settings while a run is active.")
            return

        dlg = self.advanced_dialog

        old_identity_preservation = dlg.identity_preservation_combo.currentText()
        old_tonal = dlg.tonal_transfer_combo.currentText()
        old_noise_regularize = dlg.noise_regularize_edit.value()
        old_eye = dlg.eye_preservation_combo.currentText()
        old_structure = dlg.structure_matching_combo.currentText()
        old_vgg_appearance = dlg.vgg_appearance_combo.currentText()
        old_lr = dlg.lr_edit.value()
        old_camera_lr = dlg.camera_lr_edit.value()
        old_mix_layer_start = dlg.mix_layer_start_edit.value()
        old_mix_layer_end = dlg.mix_layer_end_edit.value()
        old_advanced_iters = dlg.enable_advanced_iterations_checkbox.isChecked()
        old_strategy = dlg.strategy_combo.currentText()
        old_crop_only = dlg.crop_only_checkbox.isChecked()
        old_use_gfpgan = dlg.use_gfpgan_checkbox.isChecked()
        old_det_threshold = dlg.det_threshold_edit.value()
        old_face_factor = dlg.face_factor_edit.value()
        old_gfpgan_blend = dlg.gfpgan_blend_edit.value()
        old_gaussian = dlg.gaussian_edit.value()
        # Save main window spectral widgets
        old_photo_type = self.photo_type_combo.currentText()
        old_approx_date = self.approx_date_edit.text()
        old_spectral_mode = self.spectral_mode_combo.currentText()
        old_spectral = self.spectral_sensitivity_combo.currentText()

        if dlg.exec() == QDialog.Accepted:
            # Update iteration slider based on advanced iterations checkbox
            use_advanced_iters = dlg.enable_advanced_iterations_checkbox.isChecked()
            self.update_iteration_slider_mode(use_advanced_iters)
            self.update_mode_controls()
            self.update_runtime_label()
            # Sync detection threshold slider with dialog value
            new_det = dlg.det_threshold_edit.value()
            self.det_threshold_slider.blockSignals(True)
            self.det_threshold_slider.setValue(int(new_det * 100))
            self.det_threshold_slider.blockSignals(False)
            self.det_threshold_value_label.setText(f"{new_det:.2f}")
            self.log_box.append("Advanced settings updated.")
        else:
            dlg.strategy_combo.setCurrentText(old_strategy)
            dlg.crop_only_checkbox.setChecked(old_crop_only)
            dlg.use_gfpgan_checkbox.setChecked(old_use_gfpgan)
            dlg.det_threshold_edit.setValue(old_det_threshold)
            dlg.face_factor_edit.setValue(old_face_factor)
            dlg.gfpgan_blend_edit.setValue(old_gfpgan_blend)
            dlg.gaussian_edit.setValue(old_gaussian)
            dlg.enable_advanced_iterations_checkbox.setChecked(old_advanced_iters)
            # Restore main window spectral widgets
            self.photo_type_combo.setCurrentText(old_photo_type)
            self.approx_date_edit.setText(old_approx_date)
            self.spectral_mode_combo.setCurrentText(old_spectral_mode)
            self.spectral_sensitivity_combo.setCurrentText(old_spectral)
            dlg.identity_preservation_combo.setCurrentText(old_identity_preservation)
            dlg.tonal_transfer_combo.setCurrentText(old_tonal)
            dlg.eye_preservation_combo.setCurrentText(old_eye)
            dlg.structure_matching_combo.setCurrentText(old_structure)
            dlg.vgg_appearance_combo.setCurrentText(old_vgg_appearance)
            dlg.noise_regularize_edit.setValue(old_noise_regularize)
            dlg.lr_edit.setValue(old_lr)
            dlg.camera_lr_edit.setValue(old_camera_lr)
            dlg.mix_layer_start_edit.setValue(old_mix_layer_start)
            dlg.mix_layer_end_edit.setValue(old_mix_layer_end)
            # Restore detection threshold slider to match restored dialog value
            self.det_threshold_slider.blockSignals(True)
            self.det_threshold_slider.setValue(int(old_det_threshold * 100))
            self.det_threshold_slider.blockSignals(False)
            self.det_threshold_value_label.setText(f"{old_det_threshold:.2f}")
            self.update_spectral_sensitivity_ui()
            self.update_mode_controls()
            self.update_runtime_label()

    # ------------------------------
    # Runtime estimation / hardware
    # ------------------------------
    def update_iteration_slider_mode(self, use_advanced=False):
        """Switch iteration slider between basic (4 values) and advanced (full range)."""
        if use_advanced:
            self.iter_values = self.advanced_iter_values
            self.iter_slider.setMaximum(len(self.advanced_iter_values) - 1)
        else:
            # Get current value, find closest match in basic values, then switch
            current_val = int(self.iter_values[self.iter_slider.value()])
            self.iter_values = self.basic_iter_values
            self.iter_slider.setMaximum(len(self.basic_iter_values) - 1)
            # Set to closest basic value
            closest_idx = min(range(len(self.basic_iter_values)), key=lambda i: abs(self.basic_iter_values[i] - current_val))
            self.iter_slider.setValue(closest_idx)
        self.update_iteration_label()

    def get_selected_preset_value(self):
        """Get the selected iteration value from the current slider."""
        return int(self.iter_values[self.iter_slider.value()])

    def get_identity_preservation_value(self):
        preset = self.advanced_dialog.identity_preservation_combo.currentText()

        if preset == "Off":
            return 0.0
        if preset == "Lower":
            return 0.20
        if preset == "Higher":
            return 0.45
        return 0.30

    def get_tonal_transfer_value(self):
        preset = self.advanced_dialog.tonal_transfer_combo.currentText()

        if preset == "Off":
            return 0.0
        if preset == "Lower":
            return 1e9
        if preset == "Higher":
            return 5e10
        return 1e10

    def get_eye_preservation_value(self):
        preset = self.advanced_dialog.eye_preservation_combo.currentText()

        if preset == "Off":
            return 0.0
        if preset == "Lower":
            return 0.05
        if preset == "Higher":
            return 0.20
        return 0.10

    def get_structure_matching_value(self):
        preset = self.advanced_dialog.structure_matching_combo.currentText()

        if preset == "Off":
            return 0.0
        if preset == "Lower":
            return 0.05
        if preset == "Higher":
            return 0.20
        return 0.10

    def get_vgg_appearance_value(self):
        preset = self.advanced_dialog.vgg_appearance_combo.currentText()

        if preset == "Off":
            return 0.0
        if preset == "Lower":
            return 0.5
        if preset == "Higher":
            return 1.5
        return 1.0
        
    def get_spectral_sensitivity_value(self):
        label = self.spectral_sensitivity_combo.currentText()
        
        # Map UI labels to backend codes
        if label == "Blue-sensitive":
            return "b"
        elif label == "Orthochromatic":
            return "gb"
        elif label == "Panchromatic":
            return "g"
        else:
            # Fallback
            return "gb"

    def estimate_runtime_minutes(self, preset_value: int):
        # Baseline observed on RTX 3060 Laptop GPU (approx.)
        anchors = [
            (375, 5),
            (750, 10),
            (1500, 20),
            (3000, 38),
            (6000, 136),
            (18000, 467),
        ]

        # If exact anchor, return it
        for x, y in anchors:
            if preset_value == x:
                return float(y)

        # Log-log interpolation between nearest anchors (reasonable for scaling curves)
        anchors_sorted = sorted(anchors, key=lambda t: t[0])

        if preset_value < anchors_sorted[0][0]:
            x0, y0 = anchors_sorted[0]
            x1, y1 = anchors_sorted[1]
        elif preset_value > anchors_sorted[-1][0]:
            x0, y0 = anchors_sorted[-2]
            x1, y1 = anchors_sorted[-1]
        else:
            for i in range(len(anchors_sorted) - 1):
                if anchors_sorted[i][0] <= preset_value <= anchors_sorted[i + 1][0]:
                    x0, y0 = anchors_sorted[i]
                    x1, y1 = anchors_sorted[i + 1]
                    break

        lx0, ly0 = math.log(x0), math.log(y0)
        lx1, ly1 = math.log(x1), math.log(y1)
        lxp = math.log(preset_value)

        # linear interpolation in log space
        t = (lxp - lx0) / (lx1 - lx0)
        lyp = ly0 + t * (ly1 - ly0)
        return float(math.exp(lyp))

    def load_local_timing_records(self):
        if hasattr(self, "results_root_edit"):
            results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
        else:
            results_root = self.repo_root / "results"
        log_path = timing_log_io.log_path_for(results_root)
        try:
            cache_path = str(log_path.resolve()).lower()
        except Exception:
            cache_path = str(log_path).lower()
        try:
            log_mtime = log_path.stat().st_mtime_ns if log_path.exists() else None
        except OSError:
            log_mtime = None

        if (
            self._timing_records_cache_path == cache_path
            and self._timing_records_cache_mtime == log_mtime
        ):
            return list(self._timing_records_cache)

        records = timing_log_io.read_timing_records(log_path)
        self._timing_records_cache_path = cache_path
        self._timing_records_cache_mtime = log_mtime
        self._timing_records_cache = list(records)
        return records

    def estimate_runtime_from_local_history(self, preset_value: int):
        """Thin wrapper around runtime_estimator.estimate_runtime_minutes;
        kept here only to snapshot widget reads on the UI thread."""
        records = self.load_local_timing_records()
        current_hw = self.preflight.get_hardware_info()
        current_gpu = (current_hw.get("gpu_name") or "").strip()
        current_enh = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
        return runtime_estimator.estimate_runtime_minutes(
            records, current_gpu, current_enh, preset_value
        )

    def compute_runtime_scale(self, hw):
        return runtime_estimator.compute_runtime_scale(hw)

    def update_runtime_label(self):
        preset_val = self.get_selected_preset_value()
        face_multiplier = self.face_strip.get_runtime_face_multiplier()
        enhancement_enabled = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
        cache_key = (
            int(preset_val),
            int(face_multiplier),
            bool(enhancement_enabled),
            str(self.results_root_edit.text().strip() if hasattr(self, "results_root_edit") else ""),
            str(self._timing_records_cache_path or ""),
            int(self._timing_records_cache_mtime or 0),
            int(time.time() // max(1, int(self._hardware_info_cache_ttl_sec))),
        )
        if (
            self._runtime_label_cache_key == cache_key
            and self._runtime_label_cache_text is not None
        ):
            self.runtime_label.setText(self._runtime_label_cache_text)
            if hasattr(self, "runtime_info") and self._runtime_label_cache_tooltip is not None:
                self.runtime_info.setToolTip(self._runtime_label_cache_tooltip)
            return

        # First try local machine history
        local_mins, local_note = self.estimate_runtime_from_local_history(preset_val)

        hw = self.preflight.get_hardware_info()
        scale, scale_note = self.compute_runtime_scale(hw)

        if local_mins is not None:
            est_mins = int(round(local_mins))
            source_note = local_note
        else:
            base_mins = self.estimate_runtime_minutes(preset_val)
            est_mins = int(round(base_mins * scale))
            source_note = f"Baseline curve with hardware scaling. {local_note}"

        est_mins = int(round(est_mins * face_multiplier))

        # Human-friendly duration
        mins_total = max(0, est_mins)
        days, rem = divmod(mins_total, 24 * 60)
        hours, mins = divmod(rem, 60)

        if days > 0:
            runtime_text = f"{days} day {hours} hr {mins} min"
        elif hours > 0:
            runtime_text = f"{hours} hr {mins} min"
        else:
            runtime_text = f"{mins} min"
        self.runtime_label.setText(runtime_text)

        tooltip_text = None
        if hasattr(self, "runtime_info"):
            ram_txt = f"{hw.get('ram_gb')} GB" if hw.get("ram_gb") is not None else "Unknown"
            tooltip_text = (
                "Approximate estimate (best-effort).\n"
                f"Estimate source: {source_note}\n"
                f"Detected GPU: {hw.get('gpu_name')}\n"
                f"Detected CPU cores: {hw.get('cpu_cores')}\n"
                f"Detected RAM: {ram_txt}\n"
                f"Hardware scale factor: {scale:.2f}\n"
                f"Face multiplier: x{face_multiplier}\n"
                f"{scale_note}"
            )
            self.runtime_info.setToolTip(tooltip_text)
        self._runtime_label_cache_key = cache_key
        self._runtime_label_cache_text = runtime_text
        self._runtime_label_cache_tooltip = tooltip_text
    # ------------------------------
    # Validation / command building
    # ------------------------------
    def validate_numeric_inputs(self):
        det = float(self.advanced_dialog.det_threshold_edit.value())

        if not (0 <= det <= 1):
            self.log_box.append("Face detection sensitivity must be between 0 and 1.")
            self.status_label.setText("Status: Invalid detection range")
            return False

        return True
    # ------------------------------
    # Result discovery / preview
    # ------------------------------
    def _find_newest_image_in_tree(
        self,
        root: Path,
        after_epoch: float | None = None,
        preferred_substring: str | None = None,
        excluded_substrings=None,
    ):
        """Return newest image in a tree, optionally preferring paths containing a substring."""
        if root is None or (not root.exists()):
            return None
        try:
            root_key = str(root.resolve()).lower()
        except Exception:
            root_key = str(root).lower()
        preferred = (preferred_substring or "").lower()
        excluded = tuple(sorted((ex or "").lower() for ex in (excluded_substrings or [])))
        cache_key = (
            root_key,
            int(float(after_epoch) * 10) if after_epoch is not None else 0,
            preferred,
            excluded,
        )

        cache_entry = self._newest_image_query_cache.get(cache_key)
        now = time.time()
        if cache_entry is not None:
            ts, cached_path = cache_entry
            if (now - ts) < 0.8:
                if cached_path is None:
                    return None
                p = Path(cached_path)
                if p.exists():
                    return p

        newest_any = None
        newest_any_mtime = -1.0
        newest_preferred = None
        newest_preferred_mtime = -1.0

        root_str = str(root)
        for dirpath, _dirnames, filenames in os.walk(root_str):
            for name in filenames:
                suffix = os.path.splitext(name)[1].lower()
                if suffix not in IMAGE_EXTENSIONS:
                    continue
                full = Path(dirpath) / name
                path_text = str(full).lower()
                if excluded and any(ex in path_text for ex in excluded):
                    continue
                try:
                    mtime = full.stat().st_mtime
                except OSError:
                    continue
                if after_epoch is not None and mtime < after_epoch:
                    continue

                if mtime > newest_any_mtime:
                    newest_any_mtime = mtime
                    newest_any = full

                if preferred and preferred in path_text and mtime > newest_preferred_mtime:
                    newest_preferred_mtime = mtime
                    newest_preferred = full

        newest = newest_preferred or newest_any
        self._newest_image_query_cache[cache_key] = (now, str(newest) if newest is not None else None)
        while len(self._newest_image_query_cache) > int(self._newest_image_query_cache_max_entries):
            oldest_key = next(iter(self._newest_image_query_cache))
            self._newest_image_query_cache.pop(oldest_key, None)
        return newest

    def _find_newest_image_in_dir(
        self,
        folder: Path,
        after_epoch: float | None = None,
        name_contains: str | None = None,
    ):
        if folder is None or (not folder.exists()) or (not folder.is_dir()):
            return None
        filter_text = (name_contains or "").lower()
        newest = None
        newest_mtime = -1.0
        for p in folder.iterdir():
            if (not p.is_file()) or (p.suffix.lower() not in IMAGE_EXTENSIONS):
                continue
            if filter_text and (filter_text not in p.name.lower()):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if after_epoch is not None and mtime < after_epoch:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest = p
        return newest

    def find_latest_image(self, root: Path, after_epoch: float | None):
        # Prefer tracked directories from the active run to avoid scanning large historical trees.
        if self.current_run_result_dirs:
            newest = None
            newest_mtime = -1.0
            for dir_text in self.current_run_result_dirs:
                candidate = self._find_newest_image_in_dir(Path(dir_text), after_epoch=after_epoch)
                if candidate is None:
                    continue
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest = candidate
            if newest is not None:
                return newest
        return self._find_newest_image_in_tree(root, after_epoch=after_epoch)

    def sanitize_name_for_folder(self, text):
        """
        Convert text into a filesystem-safe folder name.
        Replaces invalid path characters with underscores.
        """
        if not text:
            return "output"
        # Replace invalid path characters with underscore
        invalid_chars = r'<>:"|?*\x00'
        result = "".join(c if c not in invalid_chars else "_" for c in text)
        # Remove excessive leading/trailing underscores
        result = result.strip("_")
        # Collapse underscore runs
        result = re.sub(r"_+", "_", result)
        return result if result else "output"

    def find_latest_crop_output(self, after_epoch=None):
        """
        Find the newest image file in preprocess/face_crops created after the run start time.
        Uses tracked path from stdout if available, otherwise does recursive scan.
        Returns Path to the newest crop image, or None if not found.
        """
        # If we have a tracked crop output dir from stdout, use it directly
        if self.current_crop_output_dir:
            tracked_dir = Path(self.current_crop_output_dir)
            newest = self._find_newest_image_in_dir(tracked_dir, after_epoch=after_epoch)
            if newest is not None:
                return newest
        
        # Fallback: recursive scan of default crop directory
        crop_dir = self.repo_root / "preprocess" / "face_crops"
        return self._find_newest_image_in_tree(crop_dir, after_epoch=after_epoch)

    def copy_crop_outputs_to_results_root(self, crop_image_path):
        """
        Copy crop-only output into the selected Results root folder.
        Creates a crop_only_<input_stem> subfolder and copies the crop image there.
        Returns the destination path, or None if copy failed.
        """
        try:
            crop_path = Path(crop_image_path)
            if not crop_path.exists():
                return None

            # Determine results root
            results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
            if not results_root.exists():
                results_root.mkdir(parents=True, exist_ok=True)

            # Get input image stem for folder naming
            input_file = self.input_image_edit.text().strip()
            input_stem = Path(input_file).stem if input_file else "crop_only_output"
            safe_stem = self.sanitize_name_for_folder(input_stem)

            # Create crop-only output folder
            crop_out_folder = results_root / f"crop_only_{safe_stem}"
            crop_out_folder.mkdir(parents=True, exist_ok=True)

            # Copy the crop image
            dest_path = crop_out_folder / crop_path.name
            shutil.copy2(crop_path, dest_path)
            return dest_path

        except Exception as e:
            self.log_box.append(f"Error copying crop output: {e}")
            return None

    def simplify_run_folder(self, folder: Path):
        """
        Keep only two images in the run folder:
          - original.*        (baseline/input image; prefers *_blend_g.*)
          - rephotographed.*  (final projector output; excludes *-init* and *-rand*)
        Deletes other image files in that folder.
        Returns (original_path, rephoto_path).
        """
        if folder is None or (not folder.exists()):
            return (None, None)

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not imgs:
            return (None, None)

        final_candidates = [
            p for p in imgs
            if ("-init" not in p.stem)
            and ("_init" not in p.stem)
            and ("-rand" not in p.stem)
            and ("original" not in p.stem.lower())
            and ("rephotographed" not in p.stem.lower())
        ]

        final = max(final_candidates, key=lambda p: p.stat().st_mtime) if final_candidates else max(imgs, key=lambda p: p.stat().st_mtime)

        remaining = [p for p in imgs if p != final]
        original = None

        cand = [p for p in remaining if "_blend_g" in p.stem]
        if cand:
            original = max(cand, key=lambda p: p.stat().st_mtime)

        if original is None:
            fallback_originals = [
                p for p in remaining
                if ("-rand" not in p.stem)
            ]
            if fallback_originals:
                original = min(fallback_originals, key=lambda p: p.stat().st_mtime)

        if original is None:
            return (None, final)

        orig_target = folder / f"original{original.suffix.lower()}"
        final_target = folder / f"rephotographed{final.suffix.lower()}"

        for t in (orig_target, final_target):
            if t.exists():
                try:
                    t.unlink()
                except OSError:
                    pass

        tmp_orig = folder / f"__tmp_original{original.suffix.lower()}"
        tmp_final = folder / f"__tmp_rephotographed{final.suffix.lower()}"

        original.replace(tmp_orig)
        final.replace(tmp_final)
        tmp_orig.replace(orig_target)
        tmp_final.replace(final_target)

        kept = {orig_target.name.lower(), final_target.name.lower()}
        removed = 0
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts and p.name.lower() not in kept:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass

        self.log_box.append(f"Simplified output: kept {orig_target.name} and {final_target.name}; removed {removed} other image(s).")
        return (orig_target, final_target)

    # Confirm-dialog threshold for high iteration counts. Above this, a run can
    # take hours and is almost always a misclick on the advanced slider.
    _HIGH_ITERATION_CONFIRM_THRESHOLD = 5000

    # Pause ACK: backend prints "=== Pause requested ===" only when it reaches a
    # safe checkpoint. If that marker doesn't arrive within this window, surface
    # a hint so the user knows the request was received but the backend is
    # still in a long step (e.g. mid-StyleGAN forward pass).
    _PAUSE_ACK_WARN_MS = 10_000

    def _normalized_path_key(self, path_text):
        return path_utils.normalized_path_key(path_text)

    def _make_safe_base_name(self, base_text):
        return path_utils.make_safe_base_name(base_text)

    def maybe_auto_start_face_detection_from_import(self, allow_during_probe=False):
        if not self.auto_detect_faces_on_import:
            return
        if self.process is not None or self.awaiting_face_selection:
            return
        if self.current_run_phase in {"preprocess", "rephoto"}:
            return
        if self.advanced_dialog.crop_only_checkbox.isChecked():
            return
        if (not allow_during_probe) and self.quick_face_probe_process is not None:
            return

        current_input = self.input_image_edit.text().strip()
        if not current_input:
            return
        current_key = self._normalized_path_key(current_input)
        if current_key is None:
            return
        if self.auto_detect_faces_armed_input != current_key:
            return
        if self.auto_detect_faces_triggered_input == current_key:
            return

        quick_count = self.quick_face_count_estimate if isinstance(self.quick_face_count_estimate, int) else None
        # Keep import auto-detect scoped to confident multi-face inputs.
        if quick_count is None or quick_count <= 1:
            return

        self.auto_detect_faces_triggered_input = current_key

        # Always use wrapper preprocessing for selection flow.
        # In-process preview crops are useful for speed, but can desync face
        # indices and crop-path identity versus face-crop-plus outputs.

        # Fall back to full PS1 wrapper if in-process detection completely failed.
        if quick_count >= 2:
            self.log_box.append("Multi-face import detected. Auto-starting face detection...")
            self.status_label.setText("Status: Auto-starting face detection...")
            QTimer.singleShot(0, self.run_wrapper)

    def resolve_conda_executable(self):
        cached = self._conda_executable_cache
        if cached is not None:
            return cached or None

        env_conda = str(os.environ.get("CONDA_EXE", "") or "").strip()
        candidates = []
        if env_conda:
            candidates.append(Path(env_conda))

        home = Path.home()
        candidates.extend(
            [
                home / "anaconda3" / "Scripts" / "conda.exe",
                home / "miniconda3" / "Scripts" / "conda.exe",
                home / "mambaforge" / "Scripts" / "conda.exe",
                home / "miniforge3" / "Scripts" / "conda.exe",
            ]
        )

        for candidate in candidates:
            try:
                if candidate.exists():
                    resolved = str(candidate)
                    self._conda_executable_cache = resolved
                    return resolved
            except OSError:
                continue

        resolved = shutil.which("conda.exe") or shutil.which("conda")
        self._conda_executable_cache = resolved if resolved else ""
        return resolved

    def resolve_facecrop_env_name(self):
        cfg_path = self.resolve_resource_path("rephoto_wrapper.config.json")
        legacy_cfg_path = self.resolve_resource_path("run_rephoto_with_facecrop_config.json")
        if not cfg_path.exists() and legacy_cfg_path.exists():
            cfg_path = legacy_cfg_path

        try:
            if cfg_path.exists():
                st = cfg_path.stat()
                cache_key = (str(cfg_path.resolve()).lower(), int(st.st_mtime_ns), int(st.st_size))
            else:
                cache_key = ("", 0, 0)
        except Exception:
            cache_key = ("", 0, 0)

        if self._facecrop_env_cache_key == cache_key and self._facecrop_env_name_cache:
            return self._facecrop_env_name_cache

        env_name = "facecrop_py310"
        if not cfg_path.exists():
            self._facecrop_env_cache_key = cache_key
            self._facecrop_env_name_cache = env_name
            return env_name
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            configured = str(cfg.get("FaceCropEnvName", "") or "").strip()
            if configured:
                env_name = configured
        except Exception:
            pass
        self._facecrop_env_cache_key = cache_key
        self._facecrop_env_name_cache = env_name
        return env_name

    def stop_quick_face_probe(self):
        process = self.quick_face_probe_process
        if process is not None:
            try:
                process.blockSignals(True)
            except Exception:
                pass
            if process.state() != QProcess.NotRunning:
                process.kill()
                process.waitForFinished(300)
            process.deleteLater()

        self.quick_face_probe_process = None
        self.quick_face_probe_token += 1
        self.quick_face_probe_target_input = None
        self.quick_face_probe_fallback_count = None
        self.quick_face_probe_stdout = ""
        self.quick_face_probe_last_error = ""

    def _drain_quick_face_probe_output(self, token, is_error=False):
        if token != self.quick_face_probe_token:
            return
        process = self.quick_face_probe_process
        if process is None:
            return
        chunk = process.readAllStandardError() if is_error else process.readAllStandardOutput()
        text = bytes(chunk).decode("utf-8", errors="ignore")
        if not text:
            return
        if is_error:
            self.quick_face_probe_last_error += text
            if len(self.quick_face_probe_last_error) > 6000:
                self.quick_face_probe_last_error = self.quick_face_probe_last_error[-6000:]
            return
        self.quick_face_probe_stdout += text
        if len(self.quick_face_probe_stdout) > 12000:
            self.quick_face_probe_stdout = self.quick_face_probe_stdout[-12000:]

    def _start_quick_face_probe(self, image_path: Path, fallback_count=None):
        self.stop_quick_face_probe()

        conda_exe = self.resolve_conda_executable()
        if not conda_exe:
            if not self.quick_face_probe_warned:
                self.log_box.append(
                    "Accurate quick face detection unavailable: conda was not found. Falling back to basic estimate."
                )
                self.quick_face_probe_warned = True
            return False

        token = self.quick_face_probe_token + 1
        self.quick_face_probe_token = token
        self.quick_face_probe_target_input = str(image_path)
        self.quick_face_probe_fallback_count = fallback_count
        self.quick_face_probe_stdout = ""
        self.quick_face_probe_last_error = ""

        probe_script = self.resolve_resource_path(Path("tools") / "quick_face_count_retina.py")
        if not probe_script.exists():
            if not self.quick_face_probe_warned:
                self.log_box.append(
                    f"Accurate quick face detection unavailable: missing probe script ({probe_script})."
                )
                self.quick_face_probe_warned = True
            self.quick_face_probe_target_input = None
            self.quick_face_probe_fallback_count = None
            return False

        facecrop_env = self.resolve_facecrop_env_name()
        process = QProcess(self)
        process.setWorkingDirectory(str(self.repo_root))
        process.readyReadStandardOutput.connect(
            lambda t=token: self._drain_quick_face_probe_output(t, is_error=False)
        )
        process.readyReadStandardError.connect(
            lambda t=token: self._drain_quick_face_probe_output(t, is_error=True)
        )
        process.finished.connect(
            lambda exit_code, exit_status, t=token, image=str(image_path), fb=fallback_count: self._quick_face_probe_finished(
                exit_code, exit_status, t, image, fb
            )
        )

        det_threshold = self.advanced_dialog.det_threshold_edit.value()
        self._quick_probe_det_threshold = float(det_threshold)
        args = [
            "run",
            "-n",
            facecrop_env,
            "python",
            str(probe_script),
            "--image",
            str(image_path),
            "--det-threshold",
            f"{det_threshold:.2f}",
            "--resize-size",
            "1536",
            "--decision-cap",
            "2",
        ]

        process.start(conda_exe, args)
        self.quick_face_probe_process = process
        if not process.waitForStarted(250):
            err = process.errorString()
            self.quick_face_probe_process = None
            self.quick_face_probe_target_input = None
            self.quick_face_probe_fallback_count = None
            self._quick_probe_det_threshold = None
            process.deleteLater()
            if not self.quick_face_probe_warned:
                self.log_box.append(
                    f"Accurate quick face detection unavailable: failed to start detector process ({err})."
                )
                self.quick_face_probe_warned = True
            return False
        return True

    def _quick_face_probe_finished(self, exit_code, exit_status, token, image_path, fallback_count):
        try:
            self._drain_quick_face_probe_output(token, is_error=False)
            self._drain_quick_face_probe_output(token, is_error=True)
        except Exception:
            pass

        if token != self.quick_face_probe_token:
            return

        self.quick_face_probe_process = None
        self.quick_face_probe_target_input = None
        self.quick_face_probe_fallback_count = None

        current_input = self.input_image_edit.text().strip()
        current_key = self._normalized_path_key(current_input)
        probe_key = self._normalized_path_key(image_path)
        if (not current_input) or (current_key is None) or (probe_key is None) or (current_key != probe_key):
            self.quick_face_probe_stdout = ""
            self.quick_face_probe_last_error = ""
            return

        precise_count = None
        if exit_status == QProcess.NormalExit and int(exit_code) == 0:
            match = QUICK_FACE_DECISION_RE.search(self.quick_face_probe_stdout)
            if match:
                try:
                    precise_count = int(match.group(1))
                except Exception:
                    precise_count = None

        if isinstance(precise_count, int):
            self.quick_face_count_estimate = precise_count
        else:
            self.quick_face_count_estimate = fallback_count if isinstance(fallback_count, int) else None
            if (not self.quick_face_probe_warned) and (
                exit_status != QProcess.NormalExit or int(exit_code) != 0
            ):
                detail = self.quick_face_probe_last_error.strip().splitlines()
                suffix = f" ({detail[-1]})" if detail else ""
                self.log_box.append(
                    f"Accurate quick face detection fell back to basic estimate (exit {exit_code}){suffix}"
                )
                self.quick_face_probe_warned = True

        if self.quick_face_count_estimate == 0:
            self.status_label.setText("Status: No faces detected in quick scan")
            self.log_box.append(
                "Quick scan found 0 faces at the current threshold. Backend run will auto-lower detection threshold; if it still fails, crop tighter around the face."
            )

        # Parse piggy-backed RETINA_FACE_BOX lines from the count probe so we
        # can skip a separate retina box probe later (saves ~5-10s conda+model load).
        if exit_status == QProcess.NormalExit and int(exit_code) == 0:
            piggybacked_boxes_by_index = {}
            for line in (self.quick_face_probe_stdout or "").splitlines():
                box_match = RETINA_FACE_BOX_RE.search(line.strip())
                if box_match:
                    bi = int(box_match.group(1))
                    bx = int(box_match.group(2))
                    by = int(box_match.group(3))
                    bw = int(box_match.group(4))
                    bh = int(box_match.group(5))
                    if bw >= 8 and bh >= 8:
                        if bi not in piggybacked_boxes_by_index:
                            piggybacked_boxes_by_index[bi] = (bx, by, bw, bh)
            piggybacked_boxes = [piggybacked_boxes_by_index[i] for i in sorted(piggybacked_boxes_by_index)]
            if piggybacked_boxes:
                self._quick_probe_retina_boxes = piggybacked_boxes
                self._quick_probe_retina_boxes_path = image_path

        self.quick_face_probe_stdout = ""
        self.quick_face_probe_last_error = ""

        # Probe finished - stop the "Detecting Faces" overlay.
        self.set_input_detect_overlay(False)

        # Reset the trigger guard so auto-detect can re-fire with the
        # accurate count from the probe (it may have been set earlier).
        current_input = self.input_image_edit.text().strip()
        current_key = self._normalized_path_key(current_input)
        if current_key is not None and self.auto_detect_faces_triggered_input == current_key:
            self.auto_detect_faces_triggered_input = None

        self.face_strip.update_run_button_for_quick_face_hint()
        self.maybe_auto_start_face_detection_from_import()

    def _warm_up_haar_detector(self):
        """Pre-load the Haar cascade so the first face probe is fast."""
        try:
            import cv2
            if self._haar_face_detector is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._haar_face_detector = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

    def _try_fast_inprocess_face_detect(self, image_path: Path):
        """Create preview crops from RetinaFace boxes obtained by the probe.

        Returns True if successful and the filmstrip is populated.
        The crops saved here are rough previews. When the user clicks Run,
        the PS1 wrapper re-detects and produces proper aligned crops via face-crop-plus.
        """
        # Use RetinaFace boxes cached by _quick_face_probe_finished.
        cached_path = getattr(self, "_quick_probe_retina_boxes_path", None)
        cached_boxes = getattr(self, "_quick_probe_retina_boxes", None)
        if not cached_boxes:
            return False
        if self._normalized_path_key(str(image_path)) != self._normalized_path_key(str(cached_path)):
            return False

        try:
            import cv2
        except ImportError:
            return False

        img = cv2.imread(str(image_path))
        if img is None:
            return False

        h, w = img.shape[:2]
        faces = [(int(bx), int(by), int(bw), int(bh)) for bx, by, bw, bh in cached_boxes]
        if len(faces) < 1:
            return False

        # Sort faces left-to-right for consistent ordering.
        faces = sorted(faces, key=lambda r: r[0])

        # Build safe output directory matching the PS1 wrapper convention.
        safe_base = self._make_safe_base_name(image_path.stem)
        crop_out_dir = self.repo_root / "preprocess" / "face_crops" / safe_base
        crop_out_dir.mkdir(parents=True, exist_ok=True)

        # Clear previous crops from this directory.
        for old in crop_out_dir.iterdir():
            if old.is_file() and old.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    old.unlink()
                except OSError:
                    pass

        # Crop each face and keep source boxes in sync with generated crops.
        face_factor = self.advanced_dialog.face_factor_edit.value()
        expand = max(0.10, (1.0 / max(0.1, face_factor)) - 1.0) * 0.25
        crop_paths = []
        original_boxes = []
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            ox, oy, ow, oh = int(fx), int(fy), int(fw), int(fh)

            # Expand the box.
            pad_x = int(ow * expand)
            pad_y = int(oh * expand)
            x1 = max(0, ox - pad_x)
            y1 = max(0, oy - pad_y)
            x2 = min(w, ox + ow + pad_x)
            y2 = min(h, oy + oh + pad_y)

            # Make it square (centered on the face).
            side = max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_path = crop_out_dir / f"{safe_base}_{idx:03d}.png"
            cv2.imwrite(str(out_path), crop)
            crop_paths.append(out_path)
            original_boxes.append((ox, oy, ow, oh))

        if not crop_paths:
            return False

        # Wire up the filmstrip as if the PS1 wrapper had just finished.
        self.current_crop_output_dir = str(crop_out_dir)
        self._crop_source_input_key = self._normalized_path_key(self.input_image_edit.text().strip())
        self._crop_source_face_factor = self.advanced_dialog.face_factor_edit.value()
        # Flag that these are rough preview crops — the PS1 wrapper must re-crop
        # with face-crop-plus for proper alignment when the user clicks Run.
        self._inprocess_preview_crops = True
        # Set state that _prepare_face_selection_after_preprocess expects
        # (normally set by the PS1 wrapper flow / process_finished).
        self.selection_preprocess_mode = False
        self.suppress_preprocess_ui_until_rephoto = False
        self.current_run_phase = "preprocess"
        self.input_face_boxes = original_boxes
        self.input_face_box_source = "retina_probe"
        self.log_box.append(f"In-process face detection: {len(crop_paths)} faces found instantly.")
        self.status_label.setText(f"Status: {len(crop_paths)} faces detected")
        self.set_preprocess_progress(100, "Preprocessing complete")
        self._prepare_face_selection_after_preprocess()
        return True

    def estimate_faces_for_quick_hint(self, image_path: Path):
        """Fast fallback estimate used only for run-button hinting if precise probe is unavailable."""
        try:
            import cv2
        except Exception:
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        max_side = max(h, w)
        target_max_side = 960
        if max_side > target_max_side:
            scale = float(target_max_side) / float(max_side)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = self._haar_face_detector
        if detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            self._haar_face_detector = detector
        if detector is None or detector.empty():
            return None

        min_dim = min(img.shape[0], img.shape[1])
        min_face = max(28, int(round(min_dim * 0.035)))
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )
        return int(len(faces))

    def refresh_quick_face_hint_from_input(self):
        image_path_text = self.input_image_edit.text().strip()
        if not image_path_text:
            self.stop_quick_face_probe()
            self.quick_face_count_estimate = None
            self.face_strip.update_run_button_for_quick_face_hint()
            self.set_input_detect_overlay(False)
            return

        image_path = Path(image_path_text)
        if not image_path.exists():
            self.stop_quick_face_probe()
            self.quick_face_count_estimate = None
            self.face_strip.update_run_button_for_quick_face_hint()
            self.set_input_detect_overlay(False)
            return

        # Launch RetinaFace probe for face detection.
        # When the probe finishes, _quick_face_probe_finished will set the
        # face count, create preview crops, and populate the filmstrip.
        self.face_strip.update_run_button_for_quick_face_hint()
        if self._start_quick_face_probe(image_path, fallback_count=None):
            self.set_input_detect_overlay(True, "Detecting Faces")
            self.log_box.append("Detecting faces (RetinaFace)...")
            return

        # Probe couldn't be launched. Stop the overlay.
        self.set_input_detect_overlay(False)

    def resolve_input_face_boxes_via_retina_probe(self, image_path: Path, expected_count=None):
        if image_path is None or (not image_path.exists()):
            return []

        # Check if the quick count probe already gave us retina boxes for this image.
        cached_path = getattr(self, "_quick_probe_retina_boxes_path", None)
        cached_boxes = getattr(self, "_quick_probe_retina_boxes", None)
        if cached_boxes and cached_path:
            try:
                if self._normalized_path_key(str(image_path)) == self._normalized_path_key(cached_path):
                    boxes = list(cached_boxes)
                    if expected_count is not None and int(expected_count) > 0 and len(boxes) > int(expected_count):
                        boxes = boxes[: int(expected_count)]
                    return boxes
            except Exception:
                pass

        conda_exe = self.resolve_conda_executable()
        if not conda_exe:
            return []
        probe_script = self.resolve_resource_path(Path("tools") / "quick_face_boxes_retina.py")
        if not probe_script.exists():
            return []

        det_threshold = self.advanced_dialog.det_threshold_edit.value()
        facecrop_env = self.resolve_facecrop_env_name()
        command = [
            conda_exe,
            "run",
            "-n",
            facecrop_env,
            "python",
            str(probe_script),
            "--image",
            str(image_path),
            "--det-threshold",
            f"{det_threshold:.2f}",
            "--resize-size",
            "1536",
        ]
        if expected_count is not None and int(expected_count) > 0:
            command.extend(["--max-faces", str(int(expected_count))])

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=40,
                cwd=str(self.repo_root),
            )
        except subprocess.TimeoutExpired:
            self.log_box.append(
                "Retina face-box probe timed out (40s). Detection may have stalled; "
                "see Help → Show Last Backend Error if the run also fails."
            )
            return []
        except Exception as exc:
            if not self.retina_face_box_probe_warned:
                self.log_box.append(f"Retina face-box probe failed to launch: {exc}")
                self.retina_face_box_probe_warned = True
            return []

        if proc.returncode != 0:
            if not self.retina_face_box_probe_warned:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                suffix = f" ({tail[-1]})" if tail else ""
                self.log_box.append(f"Retina face-box probe failed{suffix}")
                self.retina_face_box_probe_warned = True
            return []

        boxes_by_index = {}
        for line in (proc.stdout or "").splitlines():
            match = RETINA_FACE_BOX_RE.search(line.strip())
            if not match:
                continue
            bi = int(match.group(1))
            x = int(match.group(2))
            y = int(match.group(3))
            w = int(match.group(4))
            h = int(match.group(5))
            if w >= 8 and h >= 8:
                if bi not in boxes_by_index:
                    boxes_by_index[bi] = (x, y, w, h)
        boxes = [boxes_by_index[i] for i in sorted(boxes_by_index)]

        if expected_count is not None and int(expected_count) > 0 and len(boxes) > int(expected_count):
            boxes = boxes[: int(expected_count)]
        return boxes

    def _make_face_box_probe_cache_key(self, probe_name, image_path: Path, expected_count, det_threshold, face_factor):
        try:
            stat = image_path.stat()
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
            size_bytes = int(stat.st_size)
        except OSError:
            mtime_ns = 0
            size_bytes = 0
        return (
            str(probe_name),
            str(image_path.resolve()).lower(),
            size_bytes,
            mtime_ns,
            int(expected_count) if expected_count is not None else 0,
            float(f"{float(det_threshold):.2f}"),
            float(f"{float(face_factor):.2f}"),
        )

    def _set_face_box_probe_cache(self, key, boxes):
        self.face_box_probe_cache[key] = [tuple(map(int, b)) for b in boxes]
        while len(self.face_box_probe_cache) > int(self.face_box_probe_cache_max_entries):
            oldest_key = next(iter(self.face_box_probe_cache))
            self.face_box_probe_cache.pop(oldest_key, None)

    def resolve_input_face_boxes_via_cropper_probe(self, image_path: Path, expected_count=None):
        if image_path is None or (not image_path.exists()):
            return []
        conda_exe = self.resolve_conda_executable()
        if not conda_exe:
            return []
        probe_script = self.resolve_resource_path(Path("tools") / "quick_face_boxes_cropper.py")
        if not probe_script.exists():
            return []

        det_threshold = self.advanced_dialog.det_threshold_edit.value()
        face_factor = self.advanced_dialog.face_factor_edit.value()
        cache_key = self._make_face_box_probe_cache_key(
            "cropper_probe",
            image_path=image_path,
            expected_count=expected_count,
            det_threshold=det_threshold,
            face_factor=face_factor,
        )
        if cache_key in self.face_box_probe_cache:
            return list(self.face_box_probe_cache.get(cache_key, []))

        facecrop_env = self.resolve_facecrop_env_name()
        command = [
            conda_exe,
            "run",
            "-n",
            facecrop_env,
            "python",
            str(probe_script),
            "--image",
            str(image_path),
            "--det-threshold",
            f"{det_threshold:.2f}",
            "--resize-size",
            "3072",
            "--output-size",
            "2048",
            "--face-factor",
            f"{face_factor:.2f}",
        ]
        if expected_count is not None and int(expected_count) > 0:
            command.extend(["--max-faces", str(int(expected_count))])

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=65,
                cwd=str(self.repo_root),
            )
        except subprocess.TimeoutExpired:
            # Timeouts are environment-actionable (slow drive/CPU) and rare;
            # log every one rather than suppressing after the first.
            self.log_box.append(
                "Cropper face-box probe timed out (65s). Falling back to "
                "Haar/Retina detection. Slow disk or stalled conda env can cause this."
            )
            return []
        except Exception as exc:
            if not self.cropper_face_box_probe_warned:
                self.log_box.append(f"Cropper face-box probe failed to launch: {exc}")
                self.cropper_face_box_probe_warned = True
            return []

        if proc.returncode != 0:
            if not self.cropper_face_box_probe_warned:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                suffix = f" ({tail[-1]})" if tail else ""
                self.log_box.append(f"Cropper face-box probe failed{suffix}")
                self.cropper_face_box_probe_warned = True
            return []

        boxes_by_index = {}
        for line in (proc.stdout or "").splitlines():
            match = CROP_ALIGN_BOX_RE.search(line.strip())
            if not match:
                continue
            bi = int(match.group(1))
            x = int(match.group(2))
            y = int(match.group(3))
            w = int(match.group(4))
            h = int(match.group(5))
            if w >= 8 and h >= 8:
                if bi not in boxes_by_index:
                    boxes_by_index[bi] = (x, y, w, h)
        boxes = [boxes_by_index[i] for i in sorted(boxes_by_index)]

        if expected_count is not None and int(expected_count) > 0 and len(boxes) > int(expected_count):
            boxes = boxes[: int(expected_count)]
        self._set_face_box_probe_cache(cache_key, boxes)
        return boxes

    def _list_image_files_in_dir(self, folder: Path):
        return path_utils.list_image_files_in_dir(folder)

    def collect_current_crop_files(self):
        if self.current_crop_output_dir:
            tracked = Path(self.current_crop_output_dir)
            tracked_files = self._list_image_files_in_dir(tracked)
            if tracked_files:
                return tracked_files

        crop_root = self.repo_root / "preprocess" / "face_crops"
        if not crop_root.exists():
            return []

        newest_dir = None
        newest_mtime = -1.0
        for p in crop_root.iterdir():
            if not p.is_dir():
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if self.run_started_at is not None and mtime < self.run_started_at:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_dir = p

        if newest_dir is None:
            return []
        return self._list_image_files_in_dir(newest_dir)

    def _face_thumb_icon_cache_key(self, image_path, fallback_text, muted, thumb_size):
        normalized = ""
        mtime_ns = 0
        file_size = 0
        if image_path is not None:
            p = Path(image_path)
            try:
                normalized = str(p.resolve()).lower()
            except Exception:
                normalized = str(p).lower()
            try:
                if p.exists():
                    st = p.stat()
                    mtime_ns = int(st.st_mtime_ns)
                    file_size = int(st.st_size)
            except OSError:
                pass
        return (normalized, str(fallback_text), bool(muted), int(thumb_size), mtime_ns, file_size)

    def _is_processing_active(self):
        return (self.process is not None) or (self.current_run_phase in {"preprocess", "rephoto"})

    def _is_face_interaction_allowed(self, face_index):
        try:
            idx = int(face_index)
        except Exception:
            return False
        if idx < 0 or idx >= len(self.face_strip.face_preview_entries):
            return False
        if self.awaiting_face_selection:
            return True
        if not self._is_processing_active():
            return True
        return bool(self.face_strip.face_preview_entries[idx].get("selected", False))

    def _resolve_input_interaction_face_boxes(self):
        if not self.face_strip.face_preview_entries:
            return []
        resolved = []
        for entry in self.face_strip.face_preview_entries:
            idx = entry.get("index")
            if not isinstance(idx, int) or idx < 0:
                continue
            if not self._is_face_interaction_allowed(idx):
                continue
            if idx in self.face_strip.hover_face_box_cache:
                box = self.face_strip.hover_face_box_cache.get(idx)
            else:
                box = self.resolve_hover_face_box(idx)
                self.face_strip.hover_face_box_cache[idx] = box
            if box is None:
                continue
            x, y, w, h = box
            if w < 8 or h < 8:
                continue
            resolved.append((idx, (x, y, w, h)))
        return resolved

    def _hit_test_input_face_index(self, pos):
        src_pt = self._source_point_from_input_preview_pos(pos)
        if src_pt is None:
            return None
        sx, sy = src_pt
        candidates = []
        for idx, box in self._resolve_input_interaction_face_boxes():
            x, y, w, h = box
            if sx >= x and sx <= (x + w) and sy >= y and sy <= (y + h):
                candidates.append((w * h, idx))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return int(candidates[0][1])

    def resolve_face_box_from_crop_template(self, face_index):
        if not self.face_strip.face_preview_entries:
            return None
        if face_index < 0 or face_index >= len(self.face_strip.face_preview_entries):
            return None

        entry = self.face_strip.face_preview_entries[face_index]
        crop_path = entry.get("crop_path")
        input_path_text = self.input_image_edit.text().strip()
        if (crop_path is None) or (not input_path_text):
            return None

        # Cache template-match results per (input_path, crop_path) to avoid
        # expensive multi-scale cv2.matchTemplate on every hover event.
        cache = getattr(self, "_template_match_cache", None)
        if cache is None:
            cache = {}
            self._template_match_cache = cache
        cache_key = (input_path_text, crop_path)
        if cache_key in cache:
            return cache[cache_key]

        crop_file = Path(crop_path)
        input_file = Path(input_path_text)
        if (not crop_file.exists()) or (not input_file.exists()):
            return None

        try:
            import cv2
        except Exception:
            return None

        input_gray = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)
        crop_gray = cv2.imread(str(crop_file), cv2.IMREAD_GRAYSCALE)
        if input_gray is None or crop_gray is None:
            return None

        ih, iw = input_gray.shape[:2]
        ch, cw = crop_gray.shape[:2]
        if iw < 8 or ih < 8 or cw < 8 or ch < 8:
            return None

        # Keep template matching responsive for very large scans.
        max_side = max(iw, ih)
        input_scale = 1.0
        target_max = 2400
        if max_side > target_max:
            input_scale = float(target_max) / float(max_side)
            nw = max(1, int(round(iw * input_scale)))
            nh = max(1, int(round(ih * input_scale)))
            input_gray_small = cv2.resize(input_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            input_gray_small = input_gray

        sh, sw = input_gray_small.shape[:2]
        if sh < 16 or sw < 16:
            return None

        # face-crop outputs are resized, so template scale may differ from source face size.
        # Search a compact range of scales and keep the highest-confidence match.
        candidate_scales = [1.0, 0.85, 0.72, 0.62, 0.54, 0.46, 0.40, 0.34, 0.29, 0.24, 0.20, 0.17, 0.14, 0.12]
        best = None  # (score, x, y, w, h)
        for extra_scale in candidate_scales:
            tw = max(12, int(round(cw * input_scale * extra_scale)))
            th = max(12, int(round(ch * input_scale * extra_scale)))
            if tw >= sw or th >= sh:
                continue

            try:
                templ = cv2.resize(crop_gray, (tw, th), interpolation=cv2.INTER_AREA)
                response = cv2.matchTemplate(input_gray_small, templ, cv2.TM_CCOEFF_NORMED)
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response)
            except Exception:
                continue

            if best is None or max_val > best[0]:
                best = (float(max_val), int(max_loc[0]), int(max_loc[1]), int(tw), int(th))

        if best is None:
            cache[cache_key] = None
            return None

        best_score, bx, by, bw, bh = best
        # Low-confidence matches are often false positives on repeated textures.
        if best_score < 0.20:
            cache[cache_key] = None
            return None

        inv = (1.0 / input_scale) if input_scale > 0 else 1.0
        x = int(round(bx * inv))
        y = int(round(by * inv))
        w = int(round(bw * inv))
        h = int(round(bh * inv))

        x = max(0, min(x, max(0, iw - 1)))
        y = max(0, min(y, max(0, ih - 1)))
        w = max(8, min(w, iw - x))
        h = max(8, min(h, ih - y))

        # Crop templates include substantial context; tighten to a face-centered region
        # so the hover highlight corresponds more closely to the visible face card.
        tight_w = max(8, int(round(w * 0.62)))
        tight_h = max(8, int(round(h * 0.78)))
        tight_x = x + int(round((w - tight_w) * 0.50))
        tight_y = y + int(round((h - tight_h) * 0.26))
        tight_x = max(0, min(tight_x, max(0, iw - 1)))
        tight_y = max(0, min(tight_y, max(0, ih - 1)))
        tight_w = max(8, min(tight_w, iw - tight_x))
        tight_h = max(8, min(tight_h, ih - tight_y))
        result = (tight_x, tight_y, tight_w, tight_h)
        cache[cache_key] = result
        return result

    def eventFilter(self, watched, event):
        try:
            # Before/after wipe comparison on the result preview label.
            if watched is self.result_preview_label:
                et = event.type()
                if et == QEvent.MouseMove and self.result_pixmap is not None and self.input_pixmap is not None:
                    self._apply_compare_wipe(event.pos())
                    return False
                elif et in (QEvent.Leave, QEvent.HoverLeave):
                    if self._compare_wipe_active:
                        self._compare_wipe_active = False
                        self._compare_wipe_last_pos = None  # Reset throttle for next wipe
                        self.refresh_result_preview_scale()
                    return False

            if hasattr(self, "face_preview_strip_scroll") and watched is self.face_preview_strip_scroll.viewport():
                et = event.type()
                # Block wheel scrolling on the cross-axis (prevents slight
                # vertical jitter in horizontal mode, horizontal jitter in
                # vertical mode).
                if et == QEvent.Wheel:
                    wide = getattr(self, "_wide_layout_active", False)
                    delta = event.angleDelta()
                    if wide and delta.x() != 0:
                        return True  # wide = vertical filmstrip; block horizontal wheel (cross-axis)
                    if not wide and delta.y() != 0:
                        return True  # stacked = horizontal filmstrip; block vertical wheel (cross-axis)
                if et in (QEvent.MouseMove, QEvent.HoverMove):
                    vp_pos = event.pos()
                    container_pos = self.face_preview_strip_container.mapFrom(watched, vp_pos)
                    child = self.face_preview_strip_container.childAt(container_pos)
                    while child is not None and (not isinstance(child, FaceStripToolButton)):
                        child = child.parentWidget()
                    if isinstance(child, FaceStripToolButton):
                        face_index = child.property("faceIndex")
                        if self._is_face_interaction_allowed(face_index):
                            self.face_strip.set_hover_face_preview_index(face_index)
                        else:
                            self.face_strip.clear_hover_face_preview_index()
                    else:
                        self.face_strip.clear_hover_face_preview_index()
                elif et in (QEvent.Leave, QEvent.HoverLeave):
                    self.face_strip.clear_hover_face_preview_index()
                return super().eventFilter(watched, event)

            if isinstance(watched, FaceStripToolButton):
                et = event.type()
                if et in (QEvent.Enter, QEvent.HoverEnter, QEvent.MouseMove, QEvent.HoverMove):
                    face_index = watched.property("faceIndex")
                    if self._is_face_interaction_allowed(face_index):
                        self.face_strip.set_hover_face_preview_index(face_index)
                    else:
                        self.face_strip.clear_hover_face_preview_index()
                elif et in (QEvent.Leave, QEvent.HoverLeave):
                    self.face_strip.clear_hover_face_preview_index()
        except Exception:
            # Never let hover diagnostics break event dispatch.
            pass
        return super().eventFilter(watched, event)

    def find_latest_enhanced_output(self, after_epoch=None):
        """Find the newest enhanced/blended image from GFPGAN output folders.
        Uses tracked path from stdout if available, otherwise does recursive scan."""

        focused_idx = self.face_strip._get_focused_face_preview_index()
        focused_crop = self.face_strip._get_face_crop_path(focused_idx) if focused_idx is not None else None
        if focused_crop is not None:
            focused_enhanced = self._resolve_enhanced_preview_for_crop(focused_crop)
            if focused_enhanced is not None:
                if after_epoch is None:
                    return focused_enhanced
                try:
                    if focused_enhanced.stat().st_mtime >= float(after_epoch):
                        return focused_enhanced
                except OSError:
                    pass

        # Try tracked blended_faces dir first
        if self.current_blended_faces_dir:
            tracked_dir = Path(self.current_blended_faces_dir)
            newest = self._find_newest_image_in_dir(tracked_dir, after_epoch=after_epoch)
            if newest:
                return newest

        # Try tracked GFPGAN output dir (prefer restored/blended-like assets, avoid cmp collages).
        if self.current_gfpgan_output_dir:
            tracked_dir = Path(self.current_gfpgan_output_dir)
            restored_dir = tracked_dir / "restored_faces"
            newest = self._find_newest_image_in_dir(restored_dir, after_epoch=after_epoch)
            if newest:
                return newest
            newest = self._find_newest_image_in_tree(
                tracked_dir,
                after_epoch=after_epoch,
                preferred_substring="restored_faces",
                excluded_substrings=["/cmp/", "\\cmp\\"],
            )
            if newest:
                return newest

        # Fallback: recursive scan of default gfpgan directory
        gfpgan_dir = self.repo_root / "preprocess" / "gfpgan_runs"
        return self._find_newest_image_in_tree(
            gfpgan_dir,
            after_epoch=after_epoch,
            preferred_substring="blended_faces",
            excluded_substrings=["/cmp/", "\\cmp\\"],
        )

    def _update_recomposite_button_state(self):
        """Enable the Recomposite button when we have a result image and a face crop to blend with."""
        if not hasattr(self, "recomposite_button"):
            return
        has_result = self.last_result_image_path is not None
        crop_path = self._get_recomposite_crop_path()
        self.recomposite_button.setEnabled(has_result and crop_path is not None)

    def _get_recomposite_crop_path(self):
        """Return the face crop path for the currently selected/active face, or None."""
        idx = self.face_strip._get_focused_face_preview_index()
        if idx is not None:
            return self.face_strip._get_face_crop_path(idx)
        return None

    def run_recomposite(self):
        """Apply Photoshop-style Color blend + Auto Color using the face crop as the base."""
        if self.last_result_image_path is None:
            self.log_box.append("Recomposite: No result image available.")
            return

        crop_path = self._get_recomposite_crop_path()
        if crop_path is None:
            self.log_box.append("Recomposite: No face crop found for the current face.")
            return

        result_path = Path(self.last_result_image_path)
        if not result_path.exists():
            self.log_box.append(f"Recomposite: Result file not found: {result_path}")
            return

        try:
            import cv2
            repo_str = str(self.repo_root)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            from utils.color_blend import color_blend, auto_color_balance_after_blend

            crop_img = cv2.imread(str(crop_path))
            result_img = cv2.imread(str(result_path))

            if crop_img is None or result_img is None:
                self.log_box.append("Recomposite: Failed to read images.")
                return

            # Resize result to match face crop if dimensions differ
            crop_h, crop_w = crop_img.shape[:2]
            res_h, res_w = result_img.shape[:2]
            if (res_h, res_w) != (crop_h, crop_w):
                result_img = cv2.resize(result_img, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

            # Stage 1: Color blend — luminance from face crop, hue/sat from result
            blended = color_blend(crop_img, result_img)

            out_path = result_path.parent / f"{result_path.stem}_recomposited.png"
            cv2.imwrite(str(out_path), blended)
            self.log_box.append(f"Recomposited image saved: {out_path.name}")

            # Stage 2: Auto Color correction on the blended result
            auto_colored = auto_color_balance_after_blend(blended)
            ac_path = result_path.parent / f"{result_path.stem}_recomposited_autocolor.png"
            cv2.imwrite(str(ac_path), auto_colored)
            self.log_box.append(f"Auto Color corrected: {ac_path.name}")

            # Store both paths for the toggle and show recomposited by default
            self._rephoto_result_path = result_path
            self._recomposited_result_path = ac_path
            self.result_view_toggle.setVisible(True)
            self.result_view_toggle.setChecked(True)
            self.result_view_toggle.setText("Recomposited")
            self.set_result_preview_image(ac_path)

        except Exception as e:
            self.log_box.append(f"Recomposite failed: {e}")

    def _on_result_view_toggle(self):
        """Switch the result preview between Rephoto and Recomposited outputs."""
        is_recomposited = self.result_view_toggle.isChecked()
        if is_recomposited and self._recomposited_result_path is not None:
            self.result_view_toggle.setText("Recomposited")
            self.set_result_preview_image(self._recomposited_result_path)
        elif self._rephoto_result_path is not None:
            self.result_view_toggle.setText("Rephoto")
            self.set_result_preview_image(self._rephoto_result_path)

    def update_elapsed_label(self):
        self.update_rephoto_bar_format()

    def open_result_image_location(self):
        if self.last_result_image_path is None:
            return

        img_path = self.last_result_image_path.resolve()
        results_dir = img_path.parent

        # Organize results folder: move intermediate files, keep only final outputs
        self._organize_results_folder(results_dir)

        os.startfile(str(results_dir))
        QApplication.clipboard().setText(str(img_path))
        self.log_box.append("Opened containing folder. Image path copied to clipboard.")

    def _organize_results_folder(self, results_dir: Path):
        """Organize results folder to show only final outputs.

        Keeps: original image, face crop, final.png, recomposited*.png
        Moves everything else to an 'intermediates' subfolder.
        """
        try:
            results_dir = Path(results_dir)
            intermediates_dir = results_dir / "intermediates"

            # Files to keep (suffixes/patterns)
            keep_patterns = {
                "final.png",
                "recomposited.png",
                "recomposited_autocolor.png",
                "final_recomposited.png",
                "final_recomposited_autocolor.png",
            }

            # Try to copy original input image if available
            input_text = self.input_image_edit.text().strip() if hasattr(self, "input_image_edit") else ""
            if input_text and os.path.isfile(input_text):
                try:
                    src = Path(input_text)
                    dst = results_dir / f"original_{src.name}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
                except Exception:
                    pass

            # Try to copy face crop if available
            crop_path = self._get_recomposite_crop_path()
            if crop_path is not None:
                try:
                    crop_path = Path(crop_path)
                    dst = results_dir / f"crop_{crop_path.name}"
                    if not dst.exists():
                        shutil.copy2(crop_path, dst)
                except Exception:
                    pass

            # Move intermediate files to subfolder
            if not intermediates_dir.exists():
                intermediates_dir.mkdir(exist_ok=True)

            for item in results_dir.iterdir():
                if item.is_file() and item.name not in keep_patterns and not item.name.startswith("original_") and not item.name.startswith("crop_"):
                    try:
                        shutil.move(str(item), str(intermediates_dir / item.name))
                    except Exception:
                        pass

        except Exception as e:
            # Non-fatal; don't block opening the folder
            pass

    def flush_pending_milestones(self):
        if not hasattr(self, "_pending_milestones") or not self._pending_milestones:
            return

        try:
            if hasattr(self, "results_root_edit"):
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
            else:
                results_root = self.repo_root / "results"

            log_path = timing_log_io.log_path_for(results_root)
            hw = self.preflight.get_hardware_info()
            source_preset = str(self.iter_values[self.iter_slider.value()])
            input_image = self.input_image_edit.text().strip()
            enhancement = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
            gpu_name = hw.get("gpu_name", "Unknown GPU")

            records = [
                timing_log_io.build_milestone_record(
                    input_image=input_image,
                    preset=pm["preset"],
                    source_run_preset=source_preset,
                    enhancement=enhancement,
                    elapsed_seconds=pm["elapsed_seconds"],
                    gpu_name=gpu_name,
                )
                for pm in self._pending_milestones
            ]
            written = timing_log_io.append_records(log_path, records)

            self.log_box.append(f"Saved {written} milestone timing record(s).")
            self._pending_milestones = []
            self._timing_records_cache_mtime = None

        except Exception as e:
            self.log_box.append(f"Milestone timing log write failed: {e}")

    def append_timing_log(self, elapsed_seconds: float, success: bool, crop_only: bool):
        try:
            if hasattr(self, "results_root_edit"):
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
            else:
                results_root = self.repo_root / "results"
            log_path = timing_log_io.log_path_for(results_root)

            hw = self.preflight.get_hardware_info()
            record = timing_log_io.build_run_record(
                input_image=self.input_image_edit.text().strip(),
                preset=str(self.iter_values[self.iter_slider.value()]),
                enhancement=(not self.advanced_dialog.use_gfpgan_checkbox.isChecked()),
                crop_only=crop_only,
                success=success,
                elapsed_seconds=elapsed_seconds,
                gpu_name=hw.get("gpu_name", "Unknown GPU"),
            )
            timing_log_io.append_records(log_path, [record])
            self._timing_records_cache_mtime = None

            self.log_box.append(f"Saved timing log: {log_path}")
        except Exception as e:
            self.log_box.append(f"Timing log write failed: {e}")



    def toggle_log_visibility(self):
        self.log_visible = not self.log_visible
        self.log_container.setVisible(self.log_visible)

        if self.log_visible:
            self.toggle_log_button.setText("Hide Log")
            self.expand_log_button.setEnabled(True)
        else:
            self.toggle_log_button.setText("Show Log")
            self.expand_log_button.setEnabled(False)
        self.update_view_menu_actions()

    def toggle_log_size(self):
        self.log_expanded = not self.log_expanded

        if self.log_expanded:
            self.log_box.setMinimumHeight(360)
            self.log_box.setMaximumHeight(360)
            self.expand_log_button.setText("Compact Log")
        else:
            self.log_box.setMinimumHeight(150)
            self.log_box.setMaximumHeight(150)
            self.expand_log_button.setText("Expand Log")
        self.update_view_menu_actions()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 820)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()








