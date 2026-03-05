import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
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

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        title_label = QLabel("Time-Travel Rephotography")
        main_layout.addWidget(title_label)

        input_row = QHBoxLayout()
        self.input_image_edit = QLineEdit()
        self.input_image_edit.setPlaceholderText("Select an input image...")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_for_image)

        input_row.addWidget(self.input_image_edit)
        input_row.addWidget(self.browse_button)

        main_layout.addWidget(QLabel("Input Image"))
        main_layout.addLayout(input_row)

        form_layout = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["test", "1500", "3000", "6000", "18000"])
        self.preset_combo.setCurrentText("3000")
        form_layout.addRow("Preset", self.preset_combo)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["all", "largest"])
        self.strategy_combo.setCurrentText("all")
        form_layout.addRow("Strategy", self.strategy_combo)

        self.use_gfpgan_checkbox = QCheckBox("Enable GFPGAN enhancement")
        self.use_gfpgan_checkbox.setChecked(False)
        self.use_gfpgan_checkbox.toggled.connect(self.update_mode_controls)
        form_layout.addRow("GFPGAN", self.use_gfpgan_checkbox)

        self.crop_only_checkbox = QCheckBox("Run crop-only test first")
        self.crop_only_checkbox.setChecked(True)
        self.crop_only_checkbox.toggled.connect(self.update_mode_controls)
        form_layout.addRow("Crop Only", self.crop_only_checkbox)

        self.blend_edit = QLineEdit("0.35")
        form_layout.addRow("Blend Amount", self.blend_edit)

        self.face_factor_edit = QLineEdit("0.65")
        form_layout.addRow("FaceFactor", self.face_factor_edit)

        self.det_threshold_edit = QLineEdit("0.9")
        form_layout.addRow("DetThreshold", self.det_threshold_edit)

        main_layout.addLayout(form_layout)

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

        main_layout.addWidget(QLabel("Log Output"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.append("GUI loaded successfully.")
        self.log_box.append("Run now launches the PowerShell wrapper.")
        self.log_box.append("Crop-only is enabled by default for the first live test.")
        main_layout.addWidget(self.log_box)

        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

        self.update_mode_controls()

    def update_mode_controls(self):
        crop_only = self.crop_only_checkbox.isChecked()
        self.use_gfpgan_checkbox.setEnabled(not crop_only)
        self.blend_edit.setEnabled((not crop_only) and self.use_gfpgan_checkbox.isChecked())

    def set_controls_for_running(self, is_running):
        self.run_button.setEnabled(not is_running)
        self.cancel_button.setEnabled(is_running)
        self.reset_button.setEnabled(not is_running)
        self.browse_button.setEnabled(not is_running)
        self.input_image_edit.setEnabled(not is_running)
        self.preset_combo.setEnabled(not is_running)
        self.strategy_combo.setEnabled(not is_running)
        self.crop_only_checkbox.setEnabled(not is_running)
        self.face_factor_edit.setEnabled(not is_running)
        self.det_threshold_edit.setEnabled(not is_running)

        if is_running:
            self.use_gfpgan_checkbox.setEnabled(False)
            self.blend_edit.setEnabled(False)
        else:
            self.update_mode_controls()

    def validate_numeric_inputs(self):
        checks = [
            ("FaceFactor", self.face_factor_edit.text().strip()),
            ("DetThreshold", self.det_threshold_edit.text().strip()),
        ]

        if (not self.crop_only_checkbox.isChecked()) and self.use_gfpgan_checkbox.isChecked():
            checks.append(("GFPGANBlend", self.blend_edit.text().strip()))

        parsed_values = {}

        for label, value in checks:
            try:
                parsed_values[label] = float(value)
            except ValueError:
                self.log_box.append(f"Invalid numeric value for {label}: {value}")
                self.status_label.setText(f"Status: Invalid {label} value")
                return False

        if parsed_values["FaceFactor"] <= 0:
            self.log_box.append("FaceFactor must be greater than 0.")
            self.status_label.setText("Status: Invalid FaceFactor range")
            return False

        if not (0 <= parsed_values["DetThreshold"] <= 1):
            self.log_box.append("DetThreshold must be between 0 and 1.")
            self.status_label.setText("Status: Invalid DetThreshold range")
            return False

        if "GFPGANBlend" in parsed_values and not (0 <= parsed_values["GFPGANBlend"] <= 1):
            self.log_box.append("GFPGANBlend must be between 0 and 1.")
            self.status_label.setText("Status: Invalid GFPGANBlend range")
            return False

        return True

    def reset_form_defaults(self):
        self.preset_combo.setCurrentText("3000")
        self.strategy_combo.setCurrentText("all")
        self.use_gfpgan_checkbox.setChecked(False)
        self.crop_only_checkbox.setChecked(True)
        self.blend_edit.setText("0.35")
        self.face_factor_edit.setText("0.65")
        self.det_threshold_edit.setText("0.9")
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

    def append_command_preview(self, command):
        preview = " ".join(f'"{part}"' if " " in part else part for part in command)
        self.log_box.append("Wrapper command:")
        self.log_box.append(preview)

    def build_wrapper_command(self):
        input_image = self.input_image_edit.text().strip()

        command = [
            "powershell.exe",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.wrapper_script),
            "-InputImage",
            input_image,
            "-Preset",
            self.preset_combo.currentText(),
            "-Strategy",
            self.strategy_combo.currentText(),
            "-FaceFactor",
            self.face_factor_edit.text().strip(),
            "-DetThreshold",
            self.det_threshold_edit.text().strip(),
        ]

        if self.crop_only_checkbox.isChecked():
            command.append("-CropOnly")
        elif self.use_gfpgan_checkbox.isChecked():
            command.extend([
                "-UseGFPGAN",
                "-GFPGANBlend",
                self.blend_edit.text().strip(),
            ])

        return command

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
        else:
            self.status_label.setText("Status: Backend returned an error")

        self.set_controls_for_running(False)
        self.process = None

    def process_error(self, process_error):
        self.log_box.append(f"Process launch error: {process_error}")
        self.status_label.setText("Status: Process launch error")
        self.set_controls_for_running(False)
        self.process = None

    def cancel_run(self):
        if self.process is None:
            self.log_box.append("No backend process is running.")
            self.status_label.setText("Status: No backend process to cancel")
            return

        self.log_box.append("Cancel requested. Stopping backend process...")
        self.status_label.setText("Status: Cancelling...")

        # Try a gentle stop first; then force-kill if needed
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

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repo_root))
        self.process.readyReadStandardOutput.connect(self.append_stdout_from_process)
        self.process.readyReadStandardError.connect(self.append_stderr_from_process)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start(command[0], command[1:])

app = QApplication(sys.argv)
window = MainWindow()
window.resize(900, 600)
window.show()
sys.exit(app.exec())














