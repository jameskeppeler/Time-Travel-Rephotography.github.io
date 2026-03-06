import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-Travel Rephotography")

        self.repo_root = Path(__file__).resolve().parent.parent
        self.wrapper_script = self.repo_root / "run_rephoto_with_facecrop.ps1"
        self.process = None
        self.run_started_at = None

        # Hidden defaults (not exposed in UI)
        self.default_face_factor = 0.65
        self.default_gfpgan_blend = 0.35

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

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["all", "largest"])
        self.strategy_combo.setCurrentText("all")
        form_layout.addRow("Faces to enhance", self.strategy_combo)

        self.crop_only_checkbox = QCheckBox("Crop-only (debug)")
        self.crop_only_checkbox.setChecked(False)
        self.crop_only_checkbox.toggled.connect(self.update_mode_controls)
        form_layout.addRow("Crop Only", self.crop_only_checkbox)

        self.use_gfpgan_checkbox = QCheckBox("Enable enhancement (GFPGAN)")
        self.use_gfpgan_checkbox.setChecked(False)
        self.use_gfpgan_checkbox.toggled.connect(self.update_mode_controls)
        form_layout.addRow("Enhancement", self.use_gfpgan_checkbox)
        self.det_threshold_edit = QDoubleSpinBox()
        self.det_threshold_edit.setRange(0.0, 1.0)
        self.det_threshold_edit.setSingleStep(0.01)
        self.det_threshold_edit.setDecimals(2)
        self.det_threshold_edit.setValue(0.90)
        form_layout.addRow("Face detection sensitivity (0–1) [0.90 recommended]", self.det_threshold_edit)
        # --- Iteration slider + test mode ---
        self.test_preset_checkbox = QCheckBox("Test mode (fast)")
        self.test_preset_checkbox.setChecked(False)
        self.basic_iter_values = [1500, 3000, 6000, 18000]
        self.advanced_iter_values = list(range(1000, 100001, 1000))
        self.iter_values = self.basic_iter_values
        self.test_preset_checkbox.toggled.connect(self.update_iteration_label)
        form_layout.addRow("Preset", self.test_preset_checkbox)

        self.iter_slider = QSlider(Qt.Horizontal)
        self.advanced_mode_checkbox = QCheckBox("Advanced")
        self.advanced_mode_checkbox.setChecked(False)
        self.advanced_mode_checkbox.toggled.connect(self.update_iteration_mode)
        self.iter_slider.setMinimum(0)
        self.iter_slider.setMaximum(len(self.iter_values) - 1)
        self.iter_slider.setValue(1)  # default 3000
        self.iter_slider.setTickPosition(QSlider.TicksBelow)
        self.iter_slider.setTickInterval(1)
        self.iter_slider.valueChanged.connect(self.update_iteration_label)

        self.iter_label = QLabel("")
        self.runtime_label = QLabel("")
        self.update_iteration_label()

        slider_wrap = QVBoxLayout()
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.iter_slider)
        slider_row.addWidget(self.advanced_mode_checkbox)
        slider_row_widget = QWidget()
        slider_row_widget.setLayout(slider_row)
        slider_wrap.addWidget(slider_row_widget)
        slider_wrap.addWidget(self.iter_label)
        slider_wrap.addWidget(self.runtime_label)
        slider_widget = QWidget()
        slider_widget.setLayout(slider_wrap)
        form_layout.addRow("Iterations", slider_widget)

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

        # --- Progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Previews (Input on left, Result on right) ---
        previews_group = QGroupBox("Previews")
        previews_layout = QHBoxLayout()
        previews_group.setLayout(previews_layout)

        # Input preview
        input_group = QGroupBox("Input Preview")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.input_preview_label = QLabel("No input image yet.")
        self.input_preview_label.setAlignment(Qt.AlignCenter)
        self.input_preview_label.setMinimumHeight(340)
        self.input_preview_label.setStyleSheet("border: 1px solid #999;")
        input_layout.addWidget(self.input_preview_label)

        # Result preview
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
# Initial state
        self.update_mode_controls()
        if not self.gfpgan_is_available():
            self.log_box.append("GFPGAN not found (deps\\GFPGAN). Enhancement is disabled.")
        else:
            self.log_box.append("GFPGAN found. Enhancement is available.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()

    def gfpgan_is_available(self):
        return (self.repo_root / "deps" / "GFPGAN").exists()

    def update_mode_controls(self):
        crop_only = self.crop_only_checkbox.isChecked()
        gfpgan_available = self.gfpgan_is_available()

        if not gfpgan_available:
            if self.use_gfpgan_checkbox.isChecked():
                self.use_gfpgan_checkbox.setChecked(False)
            self.use_gfpgan_checkbox.setEnabled(False)
            return

        self.use_gfpgan_checkbox.setEnabled(not crop_only)
        if crop_only and self.use_gfpgan_checkbox.isChecked():
            self.use_gfpgan_checkbox.setChecked(False)

    def update_iteration_mode(self):
        # If test mode is on, slider is disabled anyway.
        if self.test_preset_checkbox.isChecked():
            return

        current = self.iter_values[self.iter_slider.value()]
        self.iter_values = self.advanced_iter_values if self.advanced_mode_checkbox.isChecked() else self.basic_iter_values
        self.iter_slider.setMaximum(len(self.iter_values) - 1)

        # Snap to the closest available value in the new list.
        closest_index = min(range(len(self.iter_values)), key=lambda i: abs(self.iter_values[i] - current))
        self.iter_slider.setValue(closest_index)

        self.update_iteration_label()

    def update_iteration_label(self):
        if self.test_preset_checkbox.isChecked():
            self.iter_label.setText("Using preset: test  (wplus_step 250 750)")
            self.iter_slider.setEnabled(False)
            self.advanced_mode_checkbox.setEnabled(False)
        else:
            self.iter_slider.setEnabled(True)
            self.advanced_mode_checkbox.setEnabled(True)
            v = self.iter_values[self.iter_slider.value()]
            self.iter_label.setText(f"Using preset: {v}  (wplus_step 250 {v})")

        # Always refresh the estimate (if present)
        if hasattr(self, "runtime_label") and hasattr(self, "update_runtime_label"):
            self.update_runtime_label()
    def set_controls_for_running(self, is_running):
        self.run_button.setEnabled(not is_running)
        self.cancel_button.setEnabled(is_running)
        self.reset_button.setEnabled(not is_running)
        self.quit_button.setEnabled(not is_running)
        self.progress_bar.setVisible(is_running)

        self.browse_button.setEnabled(not is_running)
        self.input_image_edit.setEnabled(not is_running)

        self.strategy_combo.setEnabled(not is_running)
        self.crop_only_checkbox.setEnabled(not is_running)
        self.det_threshold_edit.setEnabled(not is_running)

        self.test_preset_checkbox.setEnabled(not is_running)
        self.advanced_mode_checkbox.setEnabled((not is_running) and (not self.test_preset_checkbox.isChecked()))
        self.iter_slider.setEnabled((not is_running) and (not self.test_preset_checkbox.isChecked()))

        self.results_root_edit.setEnabled(not is_running)
        self.results_browse_button.setEnabled(not is_running)

        if is_running:
            self.use_gfpgan_checkbox.setEnabled(False)
        else:
            self.update_mode_controls()

    def reset_form_defaults(self):
        self.strategy_combo.setCurrentText("all")
        self.crop_only_checkbox.setChecked(False)
        self.use_gfpgan_checkbox.setChecked(False)
        self.det_threshold_edit.setValue(0.90)

        self.test_preset_checkbox.setChecked(False)
        self.advanced_mode_checkbox.setChecked(False)
        self.iter_values = self.basic_iter_values
        self.iter_slider.setValue(1)
        self.update_iteration_label()

        self.results_root_edit.setText(str(self.repo_root / "results"))
        self.update_mode_controls()

        self.log_box.append("Defaults restored.")
        self.status_label.setText("Status: Defaults restored")

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

    def get_selected_preset_value(self):
        if self.test_preset_checkbox.isChecked():
            return 750
        return int(self.iter_values[self.iter_slider.value()])

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

    def update_runtime_label(self):
        preset_val = self.get_selected_preset_value()
        mins = self.estimate_runtime_minutes(preset_val)

        hours = int(mins // 60)
        rem = int(round(mins - hours * 60))

        if hours > 0:
            self.runtime_label.setText(f"Estimated runtime (approx.): {hours} hr {rem} min (RTX 3060 baseline)")
        else:
            self.runtime_label.setText(f"Estimated runtime (approx.): {rem} min (RTX 3060 baseline)")

    def validate_numeric_inputs(self):
        det = float(self.det_threshold_edit.value())

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

        preset_value = "test" if self.test_preset_checkbox.isChecked() else str(self.iter_values[self.iter_slider.value()])

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
            self.strategy_combo.currentText(),
            "-FaceFactor",
            str(self.default_face_factor),
            "-DetThreshold",
            self.det_threshold_edit.text().strip(),
            "-ResultsRoot",
            results_root,
        ]

        if self.crop_only_checkbox.isChecked():
            command.append("-CropOnly")
        elif self.use_gfpgan_checkbox.isChecked() and self.gfpgan_is_available():
            command.extend([
                "-UseGFPGAN",
                "-GFPGANBlend",
                str(self.default_gfpgan_blend),
            ])

        return command

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
          - rephotographed.*  (final projector output; prefers *init(*)
        Deletes other image files in that folder.
        Returns (original_path, rephoto_path).
        """
        if folder is None or (not folder.exists()):
            return (None, None)

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not imgs:
            return (None, None)

        finals = [p for p in imgs if ("-init(" in p.name) or ("_init(" in p.name) or ("init(" in p.name)]
        final = max(finals, key=lambda p: p.stat().st_mtime) if finals else max(imgs, key=lambda p: p.stat().st_mtime)

        remaining = [p for p in imgs if p != final]
        original = None

        cand = [p for p in remaining if "_blend_g" in p.stem]
        if cand:
            original = max(cand, key=lambda p: p.stat().st_mtime)

        if original is None and remaining:
            original = min(remaining, key=lambda p: p.stat().st_mtime)

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

    def append_stderr_from_process(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            for line in text.splitlines():
                self.log_box.append(line)

    def process_finished(self, exit_code, exit_status):
        self.log_box.append(f"Process finished with exit code: {exit_code}")

        if exit_code == 0:
            self.status_label.setText("Status: Backend completed successfully")

            if self.crop_only_checkbox.isChecked():
                self.set_result_preview_image(None)
                self.log_box.append("Crop-only run: no result image produced.")
            else:
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
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

    def process_error(self, process_error):
        self.log_box.append(f"Process launch error: {process_error}")
        self.status_label.setText("Status: Process launch error")
        self.set_controls_for_running(False)
        self.process = None
        self.run_started_at = None

    def cancel_run(self):
        if self.process is None:
            self.log_box.append("No backend process is running.")
            self.status_label.setText("Status: No backend process to cancel")
            return

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

        # Ensure input preview is shown even if the path was typed manually
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

        command = self.build_wrapper_command()

        self.log_box.append("Run button clicked.")
        self.append_command_preview(command)
        self.status_label.setText("Status: Running backend...")
        self.set_controls_for_running(True)

        self.run_started_at = time.time()

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repo_root))
        self.process.readyReadStandardOutput.connect(self.append_stdout_from_process)
        self.process.readyReadStandardError.connect(self.append_stderr_from_process)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start(command[0], command[1:])


app = QApplication(sys.argv)
window = MainWindow()
window.resize(1100, 820)
window.show()
sys.exit(app.exec())








