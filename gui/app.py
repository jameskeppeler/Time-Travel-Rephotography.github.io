import os
import sys
import ctypes
import platform
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSlider,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class InputDropLabel(QLabel):
    def __init__(self, parent_window):
        super().__init__("Drop or click to choose an input image.")
        self.parent_window = parent_window
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(340)
        self.setMouseTracking(True)
        self._set_normal_style()

    def _set_normal_style(self):
        self.setStyleSheet("border: 1px dashed #999; border-radius: 6px; background-color: transparent;")

    def _set_hover_style(self):
        self.setStyleSheet("border: 2px solid #1a73e8; border-radius: 6px; background-color: transparent;")

    def _set_drag_style(self):
        self.setStyleSheet("border: 2px solid #1a73e8; border-radius: 6px; background-color: transparent;")

    def enterEvent(self, event):
        self._set_hover_style()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_normal_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent_window.browse_for_image()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    p = Path(url.toLocalFile())
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                        self._set_drag_style()
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._set_hover_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    p = Path(url.toLocalFile())
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                        self.parent_window.set_selected_input_image(str(p))
                        self._set_hover_style()
                        event.acceptProposedAction()
                        return
        self._set_normal_style()
        event.ignore()

class AdvancedSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout()
        self.setLayout(layout)

        form = QFormLayout()
        layout.addLayout(form)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["all", "largest"])

        self.crop_only_checkbox = QCheckBox("Crop-only (debug)")

        self.use_gfpgan_checkbox = QCheckBox("Disable enhancement (GFPGAN)")
        self.use_gfpgan_checkbox.toggled.connect(self.update_enhancement_controls)

        self.det_threshold_edit = QDoubleSpinBox()
        self.det_threshold_edit.setRange(0.0, 1.0)
        self.det_threshold_edit.setSingleStep(0.01)
        self.det_threshold_edit.setDecimals(2)
        self.det_threshold_edit.setValue(0.90)

        self.face_factor_edit = QDoubleSpinBox()
        self.face_factor_edit.setRange(0.10, 2.00)
        self.face_factor_edit.setSingleStep(0.01)
        self.face_factor_edit.setDecimals(2)
        self.face_factor_edit.setValue(0.65)

        self.gfpgan_blend_edit = QDoubleSpinBox()
        self.gfpgan_blend_edit.setRange(0.0, 1.0)
        self.gfpgan_blend_edit.setSingleStep(0.01)
        self.gfpgan_blend_edit.setDecimals(2)
        self.gfpgan_blend_edit.setValue(0.35)

        self.spectral_sensitivity_combo = QComboBox()
        self.spectral_sensitivity_combo.addItems(["b — blue-sensitive", "gb — orthochromatic", "g — panchromatic"])

        self.gaussian_edit = QDoubleSpinBox()
        self.gaussian_edit.setRange(0.0, 5.0)
        self.gaussian_edit.setSingleStep(0.05)
        self.gaussian_edit.setDecimals(2)
        self.gaussian_edit.setValue(0.75)

        self.identity_preservation_combo = QComboBox()
        self.identity_preservation_combo.addItems(["Lower", "Default", "Higher"])
        self.identity_preservation_combo.setCurrentText("Default")

        form.addRow("Faces to enhance", self.strategy_combo)
        form.addRow("Crop Only", self.crop_only_checkbox)
        form.addRow("Enhancement", self.use_gfpgan_checkbox)
        form.addRow("Enhancement blend", self.gfpgan_blend_edit)
        form.addRow("Face detection sensitivity (0–1)", self.det_threshold_edit)
        form.addRow("Face crop expansion", self.face_factor_edit)
        form.addRow("Spectral sensitivity", self.spectral_sensitivity_combo)
        form.addRow("Gaussian blur", self.gaussian_edit)
        form.addRow("Identity preservation", self.identity_preservation_combo)

        self.update_enhancement_controls()

        button_row = QHBoxLayout()

        self.restore_defaults_button = QPushButton("Restore Defaults")
        self.restore_defaults_button.clicked.connect(self.restore_defaults)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        button_row.addWidget(self.restore_defaults_button)
        button_row.addStretch()
        button_row.addWidget(self.button_box)

        layout.addLayout(button_row)

    def restore_defaults(self):
        self.strategy_combo.setCurrentText("largest")
        self.crop_only_checkbox.setChecked(False)
        self.use_gfpgan_checkbox.setChecked(False)
        self.gfpgan_blend_edit.setValue(0.35)
        self.det_threshold_edit.setValue(0.90)
        self.face_factor_edit.setValue(0.65)
        self.spectral_sensitivity_combo.setCurrentText("b — blue-sensitive")
        self.gaussian_edit.setValue(0.75)
        self.identity_preservation_combo.setCurrentText("Default")

    def update_enhancement_controls(self):
        enhancement_enabled = (not self.use_gfpgan_checkbox.isChecked())
        self.gfpgan_blend_edit.setEnabled(enhancement_enabled)
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-Travel Rephotography")

        self.repo_root = Path(__file__).resolve().parent.parent
        self.wrapper_script = self.repo_root / "run_rephoto_with_facecrop.ps1"
        self.process = None
        self.run_started_at = None

        # Progress animation (used for long rephoto stage)
        self._progress_anim_timer = QTimer(self)
        self._progress_anim_timer.setInterval(100)
        self._progress_anim_timer.timeout.connect(self.on_progress_anim_tick)
        self._progress_anim_active = False
        self._progress_anim_t0 = 0.0
        self._progress_anim_duration = 0.0
        self._progress_anim_start = 0
        self._progress_anim_end = 0
        self._progress_anim_stage = ""

        # Preview state
        self.input_pixmap = None
        self.result_pixmap = None
        self.last_result_image_path = None

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        title_label = QLabel("Time-Travel Rephotography")
        main_layout.addWidget(title_label)

        # --- Input image ---
        input_row = QHBoxLayout()
        self.input_image_edit = QLineEdit()
        self.input_image_edit.setPlaceholderText("Select an input image...")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_for_image)

        input_row.addWidget(self.input_image_edit)
        input_row.addWidget(self.browse_button)

        main_layout.addWidget(QLabel("Input Image"))
        main_layout.addLayout(input_row)

        # --- Main settings ---
        form_layout = QFormLayout()

        self.advanced_dialog = AdvancedSettingsDialog(self)
        self.advanced_dialog.strategy_combo.setCurrentText("largest")
        self.advanced_dialog.crop_only_checkbox.setChecked(False)
        self.advanced_dialog.use_gfpgan_checkbox.setChecked(False)
        self.advanced_dialog.det_threshold_edit.setValue(0.90)
        self.advanced_dialog.face_factor_edit.setValue(0.65)
        self.advanced_dialog.gfpgan_blend_edit.setValue(0.35)
        self.advanced_dialog.spectral_sensitivity_combo.setCurrentText("b — blue-sensitive")
        self.advanced_dialog.gaussian_edit.setValue(0.75)
        self.advanced_dialog.identity_preservation_combo.setCurrentText("Default")

        self.advanced_dialog.crop_only_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_runtime_label)

        # --- Iteration slider ---
        self.basic_iter_values = [750, 1500, 3000, 6000, 18000]
        self.advanced_iter_values = [750] + list(range(1000, 20001, 1000))
        self.iter_values = self.basic_iter_values

        self.iter_slider = QSlider(Qt.Horizontal)
        self.advanced_mode_checkbox = QCheckBox("Advanced")
        self.advanced_mode_checkbox.setChecked(False)
        self.advanced_mode_checkbox.toggled.connect(self.update_iteration_mode)
        self.iter_slider.setMinimum(0)
        self.iter_slider.setMaximum(len(self.iter_values) - 1)
        self.iter_slider.setValue(0)  # default 750
        self.iter_slider.setTickPosition(QSlider.TicksBelow)
        self.iter_slider.setTickInterval(1)
        self.iter_slider.valueChanged.connect(self.update_iteration_label)

        self.iter_label = QLabel("")
        self.iter_row_label = QLabel("Iterations")
        self.iter_row_label = QLabel("Iterations")
        self.runtime_label = QLabel("")
        self.runtime_info = QToolButton()
        self.runtime_info.setText("ⓘ")
        self.runtime_info.setAutoRaise(True)
        self.runtime_info.setCursor(Qt.PointingHandCursor)
        self.runtime_info.setFixedSize(18, 18)
        self.runtime_info.setStyleSheet("QToolButton { color: #1a73e8; border: 1px solid #1a73e8; border-radius: 9px; font-weight: bold; padding: 0px; } QToolButton:hover { background: #e8f0fe; }")
        self.runtime_info.setToolTip("Estimated processing time is approximate.")

        slider_wrap = QVBoxLayout()
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.iter_slider)
        slider_row.addWidget(self.advanced_mode_checkbox)
        slider_row_widget = QWidget()
        slider_row_widget.setLayout(slider_row)
        slider_wrap.addWidget(slider_row_widget)
        self.iter_label.setVisible(False)
        slider_wrap.addWidget(self.iter_label)
        slider_widget = QWidget()
        slider_widget.setLayout(slider_wrap)
        self.iter_row_label = QLabel("Iterations")
        form_layout.addRow(self.iter_row_label, slider_widget)

        self.advanced_settings_button = QPushButton("Advanced Settings...")
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings_dialog)

        advanced_wrap = QVBoxLayout()
        advanced_wrap.setContentsMargins(0, 0, 0, 0)
        advanced_wrap.setSpacing(4)
        advanced_wrap.addWidget(self.advanced_settings_button)

        self.advanced_summary_label = QLabel("")
        advanced_wrap.addWidget(self.advanced_summary_label)

        advanced_widget = QWidget()
        advanced_widget.setLayout(advanced_wrap)

        form_layout.addRow("Advanced", advanced_widget)

        self.update_iteration_label()

        main_layout.addLayout(form_layout)

        # --- Outputs (Results only) ---
        outputs_group = QGroupBox("Outputs")
        outputs_layout = QFormLayout()
        outputs_group.setLayout(outputs_layout)

        results_row = QHBoxLayout()
        self.results_root_edit = QLineEdit(str(self.repo_root / "results"))
        self.results_browse_button = QPushButton("Browse...")
        self.results_browse_button.clicked.connect(self.browse_results_root)
        results_row.addWidget(self.results_root_edit)
        results_row.addWidget(self.results_browse_button)

        results_widget = QWidget()
        results_widget.setLayout(results_row)
        outputs_layout.addRow("Results folder", results_widget)

        main_layout.addWidget(outputs_group)

        # --- Buttons ---
        button_row = QHBoxLayout()

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_wrapper)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_run)
        self.cancel_button.setEnabled(False)

        self.reset_button = QPushButton("Reset Defaults")
        self.reset_button.clicked.connect(self.reset_form_defaults)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.quit_button)

        main_layout.addLayout(button_row)

        # --- Progress row ---
        progress_row = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Ready... %p%")
        self.progress_bar.setFixedHeight(28)
        self.progress_bar.setStyleSheet("QProgressBar { min-height: 28px; max-height: 28px; border: 1px solid #999; border-radius: 6px; text-align: center; } QProgressBar::groove { background: #e6e6e6; border-radius: 6px; } QProgressBar::chunk { background: #1a73e8; border-radius: 6px; margin: 0px; }")
        self.progress_bar.setAutoFillBackground(True)

        runtime_pack = QHBoxLayout()
        runtime_pack.setContentsMargins(0, 0, 0, 0)
        runtime_pack.setSpacing(6)
        runtime_pack.addWidget(self.runtime_label)
        runtime_pack.addWidget(self.runtime_info)
        runtime_widget = QWidget()
        runtime_widget.setLayout(runtime_pack)
        progress_row.addWidget(runtime_widget)

        progress_row.addWidget(self.progress_bar, 1)

        self.elapsed_label = QLabel("Elapsed: 0:00")
        progress_row.addWidget(self.elapsed_label)

        progress_widget = QWidget()
        progress_widget.setLayout(progress_row)
        main_layout.addWidget(progress_widget)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self.update_elapsed_label)

        # --- Previews (Input on left, Result on right) ---
        previews_group = QGroupBox("Previews")
        previews_layout = QHBoxLayout()
        previews_group.setLayout(previews_layout)

        input_group = QGroupBox("Input Preview")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.input_preview_label = InputDropLabel(self)
        self.input_preview_label.setAlignment(Qt.AlignCenter)
        self.input_preview_label.setMinimumHeight(340)
        self.input_preview_label.setStyleSheet("border: 1px solid #999;")
        input_layout.addWidget(self.input_preview_label)

        result_group = QGroupBox("Result Preview")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)

        self.result_preview_label = QLabel("No result image yet.")
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setMinimumHeight(340)
        self.result_preview_label.setStyleSheet("border: 1px solid #999;")
        result_layout.addWidget(self.result_preview_label)

        self.open_image_location_button = QPushButton("Open Image Location")
        self.open_image_location_button.clicked.connect(self.open_result_image_location)
        self.open_image_location_button.setEnabled(False)
        result_layout.addWidget(self.open_image_location_button)

        previews_layout.addWidget(input_group)
        previews_layout.addWidget(result_group)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(previews_group)

        log_container = QWidget()
        log_layout = QVBoxLayout()
        log_container.setLayout(log_layout)

        log_layout.addWidget(QLabel("Log Output"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.append("GUI loaded successfully.")
        log_layout.addWidget(self.log_box)

        self.status_label = QLabel("Status: Ready")
        log_layout.addWidget(self.status_label)

        splitter.addWidget(log_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)

        # Reorder layout: previews top, controls middle, log bottom
        controls_container = QWidget()
        controls_layout = QVBoxLayout()
        controls_container.setLayout(controls_layout)

        while main_layout.count() > 2:
            item = main_layout.takeAt(1)
            if item.widget() is not None:
                controls_layout.addWidget(item.widget())
            elif item.layout() is not None:
                controls_layout.addLayout(item.layout())

        if controls_layout.count() >= 6:
            progress_item = controls_layout.takeAt(5)
            if progress_item.widget() is not None:
                controls_layout.insertWidget(2, progress_item.widget())

        main_layout.takeAt(main_layout.count() - 1)
        splitter.setParent(None)

        previews_group.setParent(None)
        log_container.setParent(None)

        bottom_splitter = QSplitter(Qt.Vertical)
        bottom_splitter.addWidget(controls_container)
        bottom_splitter.addWidget(log_container)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(previews_group)
        main_layout.addWidget(bottom_splitter)

        # Initial state
        self.update_mode_controls()
        self.update_advanced_summary_label()
        if not self.gfpgan_is_available():
            self.log_box.append("GFPGAN not found (deps\\GFPGAN). Enhancement is disabled.")
        else:
            self.log_box.append("GFPGAN found. Enhancement is available.")

    def update_advanced_summary_label(self):
        strategy = self.advanced_dialog.strategy_combo.currentText()
        crop_only = self.advanced_dialog.crop_only_checkbox.isChecked()
        enhancement_on = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
        det = self.advanced_dialog.det_threshold_edit.value()
        face_factor = self.advanced_dialog.face_factor_edit.value()
        gfpgan_blend = self.advanced_dialog.gfpgan_blend_edit.value()
        spectral = self.advanced_dialog.spectral_sensitivity_combo.currentText()
        gaussian = self.advanced_dialog.gaussian_edit.value()
        identity = self.advanced_dialog.identity_preservation_combo.currentText()

        if crop_only:
            mode_text = "crop-only"
        else:
            mode_text = "enhancement on" if enhancement_on else "enhancement off"

        self.advanced_summary_label.setText(
            f"Current: {strategy} | {mode_text} | det {det:.2f} | face {face_factor:.2f} | blend {gfpgan_blend:.2f} | spectral {spectral} | blur {gaussian:.2f} | identity {identity}"
        )
    # ------------------------------
    # Qt / window events
    # ------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()

    # ------------------------------
    # UI state / control updates
    # ------------------------------
    def gfpgan_is_available(self):
        return (self.repo_root / "deps" / "GFPGAN").exists()

    
    def update_mode_controls(self):
        crop_only = self.advanced_dialog.crop_only_checkbox.isChecked()
        gfpgan_available = self.gfpgan_is_available()

        if not gfpgan_available:
            if self.advanced_dialog.use_gfpgan_checkbox.isChecked():
                self.advanced_dialog.use_gfpgan_checkbox.setChecked(False)
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
            return

        self.advanced_dialog.use_gfpgan_checkbox.setEnabled(not crop_only)

        if crop_only:
            self.advanced_dialog.use_gfpgan_checkbox.setChecked(True)

    def update_iteration_mode(self):
        current = self.iter_values[self.iter_slider.value()]
        self.iter_values = self.advanced_iter_values if self.advanced_mode_checkbox.isChecked() else self.basic_iter_values
        self.iter_slider.setMaximum(len(self.iter_values) - 1)

        closest_index = min(range(len(self.iter_values)), key=lambda i: abs(self.iter_values[i] - current))
        self.iter_slider.setValue(closest_index)

        self.update_iteration_label()

    def update_iteration_label(self):
        v = self.iter_values[self.iter_slider.value()]
        self.iter_row_label.setText(f"Iterations: {v}")

        if hasattr(self, "runtime_label") and hasattr(self, "update_runtime_label"):
            self.update_runtime_label()

        self.update_advanced_summary_label()
    def set_controls_for_running(self, is_running):
        self.run_button.setEnabled(not is_running)
        self.cancel_button.setEnabled(is_running)
        self.reset_button.setEnabled(not is_running)
        self.quit_button.setEnabled(not is_running)
        self.progress_bar.setVisible(True)

        self.browse_button.setEnabled(not is_running)
        self.input_image_edit.setEnabled(not is_running)

        self.advanced_dialog.strategy_combo.setEnabled(not is_running)
        self.advanced_dialog.crop_only_checkbox.setEnabled(not is_running)
        self.advanced_dialog.det_threshold_edit.setEnabled(not is_running)

        self.advanced_mode_checkbox.setEnabled(not is_running)
        self.iter_slider.setEnabled(not is_running)

        self.results_root_edit.setEnabled(not is_running)
        self.results_browse_button.setEnabled(not is_running)
        self.advanced_settings_button.setEnabled(not is_running)

        if is_running:
            # Reset progress UI on each run start
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Starting... 0%")
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
        else:
            self.update_mode_controls()

    # ------------------------------
    # Progress tracking / animation
    # ------------------------------
    def start_progress_animation(self, start_pct: int, end_pct: int, duration_s: float, stage: str):
        start_pct = int(max(0, min(100, start_pct)))
        end_pct = int(max(0, min(100, end_pct)))
        duration_s = float(max(0.3, duration_s))

        self._progress_anim_active = True
        self._progress_anim_t0 = time.time()
        self._progress_anim_duration = duration_s
        self._progress_anim_start = start_pct
        self._progress_anim_end = end_pct
        self._progress_anim_stage = stage

        # Set initial stage text immediately (direct)
        self._set_progress_direct(start_pct, stage)

        if not self._progress_anim_timer.isActive():
            self._progress_anim_timer.start()
    def stop_progress_animation(self):
        self._progress_anim_active = False
        if hasattr(self, "_progress_anim_timer") and self._progress_anim_timer.isActive():
            self._progress_anim_timer.stop()

    def on_progress_anim_tick(self):
        if not self._progress_anim_active:
            return

        elapsed = time.time() - self._progress_anim_t0
        t = min(1.0, max(0.0, elapsed / self._progress_anim_duration))
        pct = int(round(self._progress_anim_start + t * (self._progress_anim_end - self._progress_anim_start)))

        # Direct set to avoid triggering nested animations
        self._set_progress_direct(pct, self._progress_anim_stage)

        if t >= 1.0:
            self.stop_progress_animation()
    def reset_progress_state(self):
        self.stop_progress_animation()
        self._progress_total_steps = None
        self._progress_crop_count = None
        self._progress_rephoto_done = 0
        self._progress_stage = "Starting"
        self._pending_milestones = []
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting... 0%")

    def _set_progress_direct(self, percent: int, stage: str):
        percent = max(0, min(100, int(percent)))
        self._progress_stage = stage
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{stage}... {percent}%")

    def _set_progress(self, percent: int, stage: str, smooth: bool = True):
        percent = max(0, min(100, int(percent)))

        # If an animation is already running (e.g., rephoto stage), do direct updates
        if (not smooth) or getattr(self, "_progress_anim_active", False):
            self._set_progress_direct(percent, stage)
            return

        cur = int(self.progress_bar.value())

        # If moving backwards or no change, update directly
        if percent <= cur:
            self._set_progress_direct(percent, stage)
            return

        # Smooth short jumps between stages (e.g., 5% -> 15% -> 25%)
        delta = percent - cur
        duration = max(0.6, min(1.6, (delta / 100.0) * 1.2 + 0.4))
        self.start_progress_animation(start_pct=cur, end_pct=percent, duration_s=duration, stage=stage)
    def update_progress_from_line(self, line: str):
        s = (line or "").strip()

        # Stage markers (from the PowerShell wrapper output)
        if s.startswith("=== Face crop step"):
            self._set_progress(5, "Cropping faces")
            return

        if s.startswith("=== GFPGAN step"):
            # Enhancement happens before GPU pre-check; show progress movement even though wrapper doesn't count it as a step.
            if self._progress_total_steps:
                pct = int(round((1.5 / self._progress_total_steps) * 100))
                self._set_progress(max(self.progress_bar.value(), pct), "Enhancing faces")
            else:
                self._set_progress(max(self.progress_bar.value(), 15), "Enhancing faces")
            return

        if s.startswith("=== GPU pre-check"):
            # Step 2 of TotalSteps
            if self._progress_total_steps:
                pct = int(round((2 / self._progress_total_steps) * 100))
                self._set_progress(max(self.progress_bar.value(), pct), "GPU pre-check")
            else:
                self._set_progress(max(self.progress_bar.value(), 25), "GPU pre-check")
            return
        if s.startswith("=== Rephoto step"):
            # Start a continuous progress animation for the long projector stage.
            # We animate toward 99% based on the current estimate.
            try:
                preset_val = self.get_selected_preset_value()
                base_mins = self.estimate_runtime_minutes(preset_val)
                hw = self.get_hardware_info()
                scale, _note = self.compute_runtime_scale(hw)
                est_seconds = max(30.0, float(base_mins) * 60.0 * float(scale))
            except Exception:
                est_seconds = 600.0  # fallback

            # Allocate most of the estimate to the rephoto stage.
            duration = est_seconds * 0.85
            start_pct = max(self.progress_bar.value(), 30)
            self.start_progress_animation(start_pct=start_pct, end_pct=99, duration_s=duration, stage="Rephotographing")
            return
        # Parse crop count so we can compute an overall % like the wrapper's step model
        m = __import__("re").search(r"^Cropped face count:\s*(\d+)\s*$", s)
        if m:
            n = int(m.group(1))
            self._progress_crop_count = n
            self._progress_total_steps = 2 + n  # matches wrapper logic for non-crop-only runs
            pct = int(round((1 / self._progress_total_steps) * 100))
            self._set_progress(max(self.progress_bar.value(), pct), f"Crops ready ({n})")
            return

        # Rephoto milestone markers (captured during run, written only on full success)
        m = __import__("re").search(r"^=== Rephoto milestone ===\s*(\d+)\s*$", s)
        if m and self.run_started_at is not None:
            milestone_val = int(m.group(1))
            elapsed = max(0.0, time.time() - self.run_started_at)

            if not hasattr(self, "_pending_milestones"):
                self._pending_milestones = []

            if not any(pm.get("preset") == milestone_val for pm in self._pending_milestones):
                self._pending_milestones.append({
                    "preset": milestone_val,
                    "elapsed_seconds": elapsed,
                })
                self.log_box.append(f"Captured milestone timing: {milestone_val} iterations at {elapsed:.1f}s")
            return

        # Completion markers
        if s.startswith("CropOnly requested. Skipping rephoto step."):
            self._set_progress(100, "Done (crop-only)")
            return

        if s == "Done.":
            self._set_progress(100, "Done")
            return
    # ------------------------------
    # Input / output selection
    # ------------------------------
    def reset_form_defaults(self):
        self.advanced_dialog.strategy_combo.setCurrentText("largest")
        self.advanced_dialog.crop_only_checkbox.setChecked(False)
        self.advanced_dialog.use_gfpgan_checkbox.setChecked(False)
        self.advanced_dialog.det_threshold_edit.setValue(0.90)

        self.advanced_mode_checkbox.setChecked(False)
        self.iter_values = self.basic_iter_values
        self.iter_slider.setMaximum(len(self.iter_values) - 1)
        self.iter_slider.setValue(0)
        self.update_iteration_label()

        self.results_root_edit.setText(str(self.repo_root / "results"))
        self.update_mode_controls()
        self.update_advanced_summary_label()

        self.log_box.append("Defaults restored.")
        self.status_label.setText("Status: Defaults restored")

    def set_selected_input_image(self, file_path):
        self.input_image_edit.setText(file_path)
        self.log_box.append(f"Selected image: {file_path}")
        self.status_label.setText("Status: Image selected")
        self.set_input_preview_image(Path(file_path))

    def browse_for_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.input_image_edit.setText(file_path)
            self.log_box.append(f"Selected image: {file_path}")
            self.status_label.setText("Status: Image selected")
            self.set_input_preview_image(Path(file_path))

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

        old_strategy = dlg.strategy_combo.currentText()
        old_crop_only = dlg.crop_only_checkbox.isChecked()
        old_use_gfpgan = dlg.use_gfpgan_checkbox.isChecked()
        old_det_threshold = dlg.det_threshold_edit.value()
        old_face_factor = dlg.face_factor_edit.value()
        old_gfpgan_blend = dlg.gfpgan_blend_edit.value()
        old_spectral = dlg.spectral_sensitivity_combo.currentText()
        old_gaussian = dlg.gaussian_edit.value()

        if dlg.exec() == QDialog.Accepted:
            self.update_mode_controls()
            self.update_runtime_label()
            self.update_advanced_summary_label()
            self.log_box.append("Advanced settings updated.")
        else:
            dlg.strategy_combo.setCurrentText(old_strategy)
            dlg.crop_only_checkbox.setChecked(old_crop_only)
            dlg.use_gfpgan_checkbox.setChecked(old_use_gfpgan)
            dlg.det_threshold_edit.setValue(old_det_threshold)
            dlg.face_factor_edit.setValue(old_face_factor)
            dlg.gfpgan_blend_edit.setValue(old_gfpgan_blend)
            dlg.spectral_sensitivity_combo.setCurrentText(old_spectral)
            dlg.gaussian_edit.setValue(old_gaussian)
            self.update_mode_controls()
            self.update_runtime_label()
            self.update_advanced_summary_label()

    # ------------------------------
    # Runtime estimation / hardware
    # ------------------------------
    def get_selected_preset_value(self):
        return int(self.iter_values[self.iter_slider.value()])

    def get_identity_preservation_value(self):
        preset = self.advanced_dialog.identity_preservation_combo.currentText()

        if preset == "Lower":
            return 0.20
        if preset == "Higher":
            return 0.45
        return 0.30

    def estimate_runtime_minutes(self, preset_value: int):
        # Baseline observed on RTX 3060 Laptop GPU (approx.)
        anchors = [
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
        import math
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

    def get_hardware_info(self):
        # CPU
        cpu_cores = os.cpu_count() or 0
        cpu_name = platform.processor() or "Unknown CPU"

        # RAM (GB) via Windows API
        ram_gb = None
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_gb = round(stat.ullTotalPhys / (1024**3), 1)
        except Exception:
            ram_gb = None

        # GPU (best effort)
        gpu_name = "Unknown GPU"
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            line = (r.stdout or "").strip().splitlines()
            if line:
                gpu_name = line[0].strip()
        except Exception:
            pass

        return {
            "cpu_name": cpu_name,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "gpu_name": gpu_name,
        }

    def load_local_timing_records(self):
        if hasattr(self, "results_root_edit"):
            results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
        else:
            results_root = self.repo_root / "results"
        log_path = results_root / "run_timing_log.jsonl"

        if not log_path.exists():
            return []

        records = []
        try:
            import json
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not rec.get("success", False):
                        continue
                    if rec.get("crop_only", False):
                        continue
                    if "elapsed_seconds" not in rec:
                        continue
                    records.append(rec)
        except Exception:
            return []

        return records

    def estimate_runtime_from_local_history(self, preset_value: int):
        """
        Local timing model with milestone support and enhancement fallback.

        Preference:
        1) Exact local timing for this preset, same GPU, same enhancement
           (full run or milestone)
        2) Exact local timing for this preset, same enhancement
        3) Infer missing enhancement mode from the opposite mode using learned overhead
        4) Additive model:
              total ~= base_rephoto_time + enhancement_overhead
           using local history
        Returns:
          (minutes_or_None, source_note)
        """
        records = self.load_local_timing_records()
        if not records:
            return (None, "No local timing history yet.")

        current_hw = self.get_hardware_info() if hasattr(self, "get_hardware_info") else {}
        current_gpu = (current_hw.get("gpu_name") or "").strip()
        current_enh = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())

        def parse_preset(rec):
            p = rec.get("preset")
            if str(p).lower() == "test":
                return 750
            try:
                return int(p)
            except Exception:
                return None

        parsed = []
        for rec in records:
            p = parse_preset(rec)
            if p is None:
                continue
            enh = bool(rec.get("enhancement", False))
            gpu = (rec.get("gpu_name") or "").strip()
            secs = rec.get("elapsed_seconds")
            rtype = rec.get("record_type", "full_run")
            try:
                secs = float(secs)
            except Exception:
                continue
            if secs <= 0:
                continue
            parsed.append((p, secs, gpu, enh, rtype))

        if not parsed:
            return (None, "Local timing history could not be parsed.")

        def exact_matches(preset, enh_flag, same_gpu_only=False):
            out = []
            for p, s, g, e, rt in parsed:
                if p == preset and e == enh_flag:
                    if same_gpu_only and g != current_gpu:
                        continue
                    out.append((s, rt))
            return out

        def avg_secs(matches):
            return sum(s for s, _rt in matches) / len(matches)

        # -------------------------------------------------
        # 1) Exact local timing match first
        # -------------------------------------------------
        exact_same_gpu_same_enh = exact_matches(preset_value, current_enh, same_gpu_only=True)
        if exact_same_gpu_same_enh:
            types = sorted(set(rt for (_s, rt) in exact_same_gpu_same_enh))
            return (
                avg_secs(exact_same_gpu_same_enh) / 60.0,
                f"Using exact local timing from {len(exact_same_gpu_same_enh)} same-GPU record(s) at this preset ({', '.join(types)})."
            )

        exact_same_enh = exact_matches(preset_value, current_enh, same_gpu_only=False)
        if exact_same_enh:
            types = sorted(set(rt for (_s, rt) in exact_same_enh))
            return (
                avg_secs(exact_same_enh) / 60.0,
                f"Using exact local timing from {len(exact_same_enh)} record(s) at this preset ({', '.join(types)})."
            )

        # -------------------------------------------------
        # 2) Learn enhancement overhead from paired on/off local timings
        # -------------------------------------------------
        from collections import defaultdict

        def build_overhead_points(use_same_gpu):
            subset = []
            for p, s, g, e, rt in parsed:
                if use_same_gpu and g != current_gpu:
                    continue
                subset.append((p, s, e))

            by_preset = defaultdict(lambda: {"on": [], "off": []})
            for p, s, e in subset:
                if e:
                    by_preset[p]["on"].append(s)
                else:
                    by_preset[p]["off"].append(s)

            overheads = []
            for p, vals in by_preset.items():
                if vals["on"] and vals["off"]:
                    on_avg = sum(vals["on"]) / len(vals["on"])
                    off_avg = sum(vals["off"]) / len(vals["off"])
                    overheads.append((p, max(0.0, on_avg - off_avg)))
            return sorted(overheads, key=lambda t: t[0])

        overhead_points = build_overhead_points(use_same_gpu=True)
        if not overhead_points:
            overhead_points = build_overhead_points(use_same_gpu=False)

        overhead_secs = None
        overhead_note = "No paired enhancement on/off local records yet."

        exact_over = [ov for (p, ov) in overhead_points if p == preset_value]
        if exact_over:
            overhead_secs = sum(exact_over) / len(exact_over)
            overhead_note = "Exact local enhancement overhead from paired record(s) at this preset."
        elif overhead_points:
            overhead_secs = sum(ov for (_p, ov) in overhead_points) / len(overhead_points)
            overhead_note = f"Average local enhancement overhead from {len(overhead_points)} paired preset point(s)."

        # -------------------------------------------------
        # 3) Infer missing enhancement mode from opposite-mode exact timings
        # -------------------------------------------------
        opposite_exact_same_gpu = exact_matches(preset_value, (not current_enh), same_gpu_only=True)
        if opposite_exact_same_gpu and overhead_secs is not None:
            opposite_secs = avg_secs(opposite_exact_same_gpu)
            if current_enh:
                inferred_secs = opposite_secs + overhead_secs
                note = "Inferred enhancement-on timing from same-GPU enhancement-off timing plus learned overhead."
            else:
                inferred_secs = max(1.0, opposite_secs - overhead_secs)
                note = "Inferred enhancement-off timing from same-GPU enhancement-on timing minus learned overhead."
            return (inferred_secs / 60.0, note + " " + overhead_note)

        opposite_exact = exact_matches(preset_value, (not current_enh), same_gpu_only=False)
        if opposite_exact and overhead_secs is not None:
            opposite_secs = avg_secs(opposite_exact)
            if current_enh:
                inferred_secs = opposite_secs + overhead_secs
                note = "Inferred enhancement-on timing from enhancement-off timing plus learned overhead."
            else:
                inferred_secs = max(1.0, opposite_secs - overhead_secs)
                note = "Inferred enhancement-off timing from enhancement-on timing minus learned overhead."
            return (inferred_secs / 60.0, note + " " + overhead_note)

        # -------------------------------------------------
        # 4) Additive model
        #    total ~= base_rephoto_time + enhancement_overhead
        # -------------------------------------------------

        # Base runtime: prefer enhancement=False
        base_candidates = [(p, s) for (p, s, g, e, rt) in parsed if g == current_gpu and (not e)]
        if len(base_candidates) < 2:
            base_candidates = [(p, s) for (p, s, g, e, rt) in parsed if (not e)]

        exact_base = [s for (p, s) in base_candidates if p == preset_value]
        if exact_base:
            base_secs = sum(exact_base) / len(exact_base)
            base_note = f"Exact local base timing from {len(exact_base)} enhancement-off record(s)."
        else:
            grouped_base = {}
            for p, s in base_candidates:
                grouped_base.setdefault(p, []).append(s)

            base_points = sorted((p, sum(vals) / len(vals)) for p, vals in grouped_base.items())

            if len(base_points) >= 2:
                import math

                if preset_value < base_points[0][0]:
                    x0, y0 = base_points[0]
                    x1, y1 = base_points[1]
                elif preset_value > base_points[-1][0]:
                    x0, y0 = base_points[-2]
                    x1, y1 = base_points[-1]
                else:
                    for i in range(len(base_points) - 1):
                        if base_points[i][0] <= preset_value <= base_points[i + 1][0]:
                            x0, y0 = base_points[i]
                            x1, y1 = base_points[i + 1]
                            break

                if x1 != x0:
                    lx0, ly0 = math.log(x0), math.log(y0)
                    lx1, ly1 = math.log(x1), math.log(y1)
                    lxp = math.log(preset_value)
                    t = (lxp - lx0) / (lx1 - lx0)
                    lyp = ly0 + t * (ly1 - ly0)
                    base_secs = math.exp(lyp)
                    base_note = f"Interpolated local base timing from {len(base_points)} enhancement-off preset point(s)."
                else:
                    base_secs = None
                    base_note = "Duplicate local base presets only; could not interpolate."
            else:
                base_secs = None
                base_note = "Not enough enhancement-off local records to estimate base timing."

        if not current_enh:
            if base_secs is not None:
                return (base_secs / 60.0, base_note)
            return (None, base_note)

        if base_secs is not None and overhead_secs is not None:
            total_secs = base_secs + overhead_secs
            return (total_secs / 60.0, base_note + " " + overhead_note)

        if base_secs is not None:
            return (base_secs / 60.0, base_note + " No local enhancement overhead yet, so enhancement was not added.")

        return (None, base_note + " " + overhead_note)
    def compute_runtime_scale(self, hw):
        """
        Returns (scale_factor, note). Scale factor multiplies the baseline minutes.
        Baseline is your RTX 3060 Laptop measurements.
        This is intentionally conservative: it adjusts estimates slightly, not wildly.
        """
        gpu = (hw.get("gpu_name") or "").lower()
        ram = hw.get("ram_gb")

        scale = 1.0
        notes = []

        # GPU heuristic (baseline: RTX 3060 Laptop)
        if "rtx 3060" in gpu:
            scale *= 1.0
        elif "rtx 3050" in gpu:
            scale *= 1.25
            notes.append("GPU heuristic: RTX 3050 slower than 3060 baseline.")
        elif "rtx 3070" in gpu:
            scale *= 0.85
            notes.append("GPU heuristic: RTX 3070 faster than 3060 baseline.")
        elif "rtx 3080" in gpu:
            scale *= 0.75
            notes.append("GPU heuristic: RTX 3080 faster than 3060 baseline.")
        elif "rtx 4060" in gpu:
            scale *= 0.80
            notes.append("GPU heuristic: RTX 4060 class faster than 3060 baseline.")
        elif "rtx 4070" in gpu:
            scale *= 0.70
            notes.append("GPU heuristic: RTX 4070 class faster than 3060 baseline.")
        elif "rtx 4090" in gpu:
            scale *= 0.40
            notes.append("GPU heuristic: RTX 4090 far faster than 3060 baseline.")
        else:
            notes.append("GPU heuristic: unknown GPU; using baseline scaling.")

        # RAM heuristic (very modest)
        if isinstance(ram, (int, float)):
            if ram < 16:
                scale *= 1.10
                notes.append("RAM heuristic: <16 GB may slow runs slightly.")
            elif ram >= 32:
                scale *= 0.98
                notes.append("RAM heuristic: >=32 GB may help slightly.")

        return (scale, " ".join(notes))

    def update_runtime_label(self):
        preset_val = self.get_selected_preset_value()

        # First try local machine history
        local_mins, local_note = self.estimate_runtime_from_local_history(preset_val)

        hw = self.get_hardware_info()
        scale, scale_note = self.compute_runtime_scale(hw)

        if local_mins is not None:
            est_mins = int(round(local_mins))
            source_note = local_note
        else:
            base_mins = self.estimate_runtime_minutes(preset_val)
            est_mins = int(round(base_mins * scale))
            source_note = f"Baseline curve with hardware scaling. {local_note}"

        # Human-friendly duration
        mins_total = max(0, est_mins)
        days, rem = divmod(mins_total, 24 * 60)
        hours, mins = divmod(rem, 60)

        if days > 0:
            self.runtime_label.setText(f"Est. Processing Time: {days} day {hours} hr {mins} min")
        elif hours > 0:
            self.runtime_label.setText(f"Est. Processing Time: {hours} hr {mins} min")
        else:
            self.runtime_label.setText(f"Est. Processing Time: {mins} min")

        if hasattr(self, "runtime_info"):
            ram_txt = f"{hw.get('ram_gb')} GB" if hw.get("ram_gb") is not None else "Unknown"
            tip = (
                "Approximate estimate (best-effort).\n"
                f"Estimate source: {source_note}\n"
                f"Detected GPU: {hw.get('gpu_name')}\n"
                f"Detected CPU cores: {hw.get('cpu_cores')}\n"
                f"Detected RAM: {ram_txt}\n"
                f"Hardware scale factor: {scale:.2f}\n"
                f"{scale_note}"
            )
            self.runtime_info.setToolTip(tip)
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
    def append_command_preview(self, command):
        preview = " ".join(f'"{part}"' if " " in part else part for part in command)
        self.log_box.append("Wrapper command:")
        self.log_box.append(preview)

    def build_wrapper_command(self):
        input_image = self.input_image_edit.text().strip()
        results_root = self.results_root_edit.text().strip()

        selected_preset = self.iter_values[self.iter_slider.value()]
        preset_value = "test" if selected_preset == 750 else str(selected_preset)
        identity_value = str(self.get_identity_preservation_value())

        command = [
            "powershell.exe",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.wrapper_script),
            "-InputImage",
            input_image,
            "-Preset",
            preset_value,
            "-Strategy",
            self.advanced_dialog.strategy_combo.currentText(),
            "-FaceFactor",
            self.advanced_dialog.face_factor_edit.text().strip(),
            "-DetThreshold",
            self.advanced_dialog.det_threshold_edit.text().strip(),
            "-VGGFace",
            identity_value,
            "-ResultsRoot",
            results_root,
        ]

        if self.advanced_dialog.crop_only_checkbox.isChecked():
            command.append("-CropOnly")
        elif (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()) and self.gfpgan_is_available():
            command.extend([
                "-UseGFPGAN",
                "-GFPGANBlend",
                self.advanced_dialog.gfpgan_blend_edit.text().strip(),
            ])

        return command

    # ------------------------------
    # Result discovery / preview
    # ------------------------------
    def find_latest_image(self, root: Path, after_epoch: float | None):
        if not root.exists():
            return None

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        newest = None
        newest_mtime = -1.0

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
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

    def set_input_preview_image(self, image_path: Path | None):
        self.input_pixmap = None
        if image_path is None or (not image_path.exists()):
            self.input_preview_label.setText("No input image yet.")
            self.input_preview_label.setPixmap(QPixmap())
            return

        pix = QPixmap(str(image_path))
        if pix.isNull():
            self.input_preview_label.setText("Could not load input image.")
            self.input_preview_label.setPixmap(QPixmap())
            return

        self.input_pixmap = pix
        self.refresh_input_preview_scale()

    def set_result_preview_image(self, image_path: Path | None):
        self.last_result_image_path = None
        self.result_pixmap = None
        self.open_image_location_button.setEnabled(False)

        if image_path is None or (not image_path.exists()):
            self.result_preview_label.setText("No result image found.")
            self.result_preview_label.setPixmap(QPixmap())
            return

        pix = QPixmap(str(image_path))
        if pix.isNull():
            self.result_preview_label.setText("Could not load result image.")
            self.result_preview_label.setPixmap(QPixmap())
            return

        self.last_result_image_path = image_path
        self.result_pixmap = pix
        self.open_image_location_button.setEnabled(True)
        self.refresh_result_preview_scale()

    def refresh_input_preview_scale(self):
        if self.input_pixmap is None:
            return
        w = max(1, self.input_preview_label.width() - 10)
        h = max(1, self.input_preview_label.height() - 10)
        scaled = self.input_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.input_preview_label.setPixmap(scaled)

    def refresh_result_preview_scale(self):
        if self.result_pixmap is None:
            return
        w = max(1, self.result_preview_label.width() - 10)
        h = max(1, self.result_preview_label.height() - 10)
        scaled = self.result_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_preview_label.setPixmap(scaled)

    def update_elapsed_label(self):
        if self.run_started_at is None:
            self.elapsed_label.setText("Elapsed: 0:00")
            return

        elapsed = int(time.time() - self.run_started_at)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        if h > 0:
            self.elapsed_label.setText(f"Elapsed: {h}:{m:02d}:{s:02d}")
        else:
            self.elapsed_label.setText(f"Elapsed: {m}:{s:02d}")

    def open_result_image_location(self):
        if self.last_result_image_path is None:
            return

        img_path = self.last_result_image_path.resolve()
        folder = str(img_path.parent)

        os.startfile(folder)
        QApplication.clipboard().setText(str(img_path))
        self.log_box.append("Opened containing folder. Image path copied to clipboard.")

    def append_stdout_from_process(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            for line in text.splitlines():
                self.log_box.append(line)
                self.update_progress_from_line(line)

    def append_stderr_from_process(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            for line in text.splitlines():
                self.log_box.append(line)
                self.update_progress_from_line(line)

    def flush_pending_milestones(self):
        if not hasattr(self, "_pending_milestones") or not self._pending_milestones:
            return

        try:
            if hasattr(self, "results_root_edit"):
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
            else:
                results_root = self.repo_root / "results"

            results_root.mkdir(parents=True, exist_ok=True)
            log_path = results_root / "run_timing_log.jsonl"

            hw = self.get_hardware_info() if hasattr(self, "get_hardware_info") else {}
            source_preset = str(self.iter_values[self.iter_slider.value()])

            import json
            with open(log_path, "a", encoding="utf-8") as f:
                for pm in self._pending_milestones:
                    record = {
                        "record_type": "milestone",
                        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "input_image": self.input_image_edit.text().strip(),
                        "preset": str(pm["preset"]),
                        "source_run_preset": source_preset,
                        "advanced_mode": bool(self.advanced_mode_checkbox.isChecked()) if hasattr(self, "advanced_mode_checkbox") else False,
                        "enhancement": (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()),
                        "crop_only": False,
                        "success": True,
                        "elapsed_seconds": float(pm["elapsed_seconds"]),
                        "gpu_name": hw.get("gpu_name", "Unknown GPU"),
                    }
                    f.write(json.dumps(record) + "\n")

            self.log_box.append(f"Saved {len(self._pending_milestones)} milestone timing record(s).")
            self._pending_milestones = []

        except Exception as e:
            self.log_box.append(f"Milestone timing log write failed: {e}")

    def append_timing_log(self, elapsed_seconds: float, success: bool, crop_only: bool):
        try:
            if hasattr(self, "results_root_edit"):
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
            else:
                results_root = self.repo_root / "results"
            results_root.mkdir(parents=True, exist_ok=True)
            log_path = results_root / "run_timing_log.jsonl"

            hw = self.get_hardware_info() if hasattr(self, "get_hardware_info") else {}
            preset_val = str(self.iter_values[self.iter_slider.value()])

            record = {
                "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_image": self.input_image_edit.text().strip(),
                "preset": preset_val,
                "advanced_mode": bool(self.advanced_mode_checkbox.isChecked()) if hasattr(self, "advanced_mode_checkbox") else False,
                "enhancement": (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()),
                "crop_only": bool(crop_only),
                "success": bool(success),
                "elapsed_seconds": float(elapsed_seconds),
                "gpu_name": hw.get("gpu_name", "Unknown GPU"),
            }

            with open(log_path, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(record) + "\n")

            self.log_box.append(f"Saved timing log: {log_path}")
        except Exception as e:
            self.log_box.append(f"Timing log write failed: {e}")

    def process_finished(self, exit_code, exit_status):
        self.stop_progress_animation()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.log_box.append(f"Process finished with exit code: {exit_code}")

        if exit_code == 0:
            self._set_progress_direct(100, "Done")
            self.status_label.setText("Status: Backend completed successfully")
            if (not self.advanced_dialog.crop_only_checkbox.isChecked()) and (self.run_started_at is not None):
                self.append_timing_log(elapsed_seconds=(time.time() - self.run_started_at), success=True, crop_only=False)
                self.flush_pending_milestones()

            if self.advanced_dialog.crop_only_checkbox.isChecked():
                self.set_result_preview_image(None)
                self.log_box.append("Crop-only run: no result image produced.")
            else:
                if hasattr(self, "results_root_edit"):
                    results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
                else:
                    results_root = self.repo_root / "results"
                newest = self.find_latest_image(results_root, self.run_started_at)

                if newest is None:
                    self.log_box.append("No new result image was found in the results folder.")
                    self.set_result_preview_image(None)
                else:
                    run_folder = newest.parent
                    _, rephoto_path = self.simplify_run_folder(run_folder)
                    self.set_result_preview_image(rephoto_path or newest)
        else:
            self.status_label.setText("Status: Backend returned an error")

        self.set_controls_for_running(False)
        self.process = None
        self.run_started_at = None

        self.reset_progress_state()

    def process_error(self, process_error):
        self.stop_progress_animation()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.log_box.append(f"Process launch error: {process_error}")
        self.status_label.setText("Status: Process launch error")
        self.set_controls_for_running(False)
        self.process = None
        self.run_started_at = None

        self.reset_progress_state()

    def cancel_run(self):
        if self.process is None:
            self.log_box.append("No backend process is running.")
            self.status_label.setText("Status: No backend process to cancel")
            return

        self.stop_progress_animation()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.log_box.append("Cancel requested. Stopping backend process...")
        self.status_label.setText("Status: Cancelling...")

        self.process.terminate()
        if not self.process.waitForFinished(2000):
            self.process.kill()

    def run_wrapper(self):
        input_image = self.input_image_edit.text().strip()

        if not input_image:
            self.log_box.append("No input image selected.")
            self.status_label.setText("Status: Select an image first")
            return

        input_image_path = Path(input_image)
        if not input_image_path.exists():
            self.log_box.append(f"Input image not found: {input_image}")
            self.status_label.setText("Status: Input image not found")
            return

        self.set_input_preview_image(input_image_path)

        if not self.wrapper_script.exists():
            self.log_box.append(f"Wrapper script not found: {self.wrapper_script}")
            self.status_label.setText("Status: Wrapper script missing")
            return

        if self.process is not None:
            self.log_box.append("A backend process is already running.")
            self.status_label.setText("Status: Backend already running")
            return

        if not self.validate_numeric_inputs():
            return

        self.reset_progress_state()
        command = self.build_wrapper_command()

        self.log_box.append("Run button clicked.")
        self.append_command_preview(command)
        self.status_label.setText("Status: Running backend...")
        self.set_controls_for_running(True)

        self.run_started_at = time.time()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.start()
            self.update_elapsed_label()

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repo_root))
        self.process.readyReadStandardOutput.connect(self.append_stdout_from_process)
        self.process.readyReadStandardError.connect(self.append_stderr_from_process)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)

        self.process.start(command[0], command[1:])

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 820)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()








