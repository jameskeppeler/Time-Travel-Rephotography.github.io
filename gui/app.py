import subprocess
import sys
from pathlib import Path

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
        form_layout.addRow("GFPGAN", self.use_gfpgan_checkbox)

        self.crop_only_checkbox = QCheckBox("Run crop-only test first")
        self.crop_only_checkbox.setChecked(True)
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

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        button_row.addWidget(self.run_button)
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

    def run_wrapper(self):
        input_image = self.input_image_edit.text().strip()

        if not input_image:
            self.log_box.append("No input image selected.")
            self.status_label.setText("Status: Select an image first")
            return

        if not self.wrapper_script.exists():
            self.log_box.append(f"Wrapper script not found: {self.wrapper_script}")
            self.status_label.setText("Status: Wrapper script missing")
            return

        command = self.build_wrapper_command()

        self.log_box.append("Run button clicked.")
        self.append_command_preview(command)
        self.status_label.setText("Status: Running backend...")
        self.run_button.setEnabled(False)
        QApplication.processEvents()

        try:
            result = subprocess.run(
                command,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )

            if result.stdout:
                self.log_box.append("----- STDOUT -----")
                for line in result.stdout.splitlines():
                    self.log_box.append(line)

            if result.stderr:
                self.log_box.append("----- STDERR -----")
                for line in result.stderr.splitlines():
                    self.log_box.append(line)

            self.log_box.append(f"Process finished with exit code: {result.returncode}")

            if result.returncode == 0:
                self.status_label.setText("Status: Backend completed successfully")
            else:
                self.status_label.setText("Status: Backend returned an error")

        except Exception as exc:
            self.log_box.append(f"Execution error: {exc}")
            self.status_label.setText("Status: Execution error")

        finally:
            self.run_button.setEnabled(True)


app = QApplication(sys.argv)
window = MainWindow()
window.resize(900, 600)
window.show()
sys.exit(app.exec())


