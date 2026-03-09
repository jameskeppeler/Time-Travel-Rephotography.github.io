import os
import re
import sys
import ctypes
import platform
import subprocess
import shutil
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QPixmap
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
    QTabWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSplitter,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)


# ============================================================================
# SHARED DEFAULTS & CONFIGURATION CONSTANTS
# ============================================================================

# Quality/iteration presets
DEFAULT_BASIC_ITER_VALUES = [375, 750, 1500, 3000, 6000, 18000]
DEFAULT_ADVANCED_ITER_VALUES = [750] + list(range(1000, 20001, 1000))
DEFAULT_ITERATION = 750
WIDE_LAYOUT_MIN_WIDTH = 1500

# Enhancement / face detection
DEFAULT_FACE_FACTOR = 0.65
DEFAULT_GFPGAN_BLEND = 0.45
DEFAULT_DET_THRESHOLD = 0.90

# ML parameters
DEFAULT_GAUSSIAN = 0.75
DEFAULT_NOISE_REGULARIZE = 50000.0
DEFAULT_LR = 0.1
DEFAULT_CAMERA_LR = 0.01
DEFAULT_MIX_LAYER_START = 10
DEFAULT_MIX_LAYER_END = 18
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Hot-path regexes for stdout progress parsing
CROPPED_FACE_COUNT_RE = re.compile(r"^Cropped face count:\s*(\d+)\s*$")
ITER_PROGRESS_RE = re.compile(r"(\d+)/(\d+)")
YEAR_RE = re.compile(r"\b(\d{4})")

# ============================================================================
# INSTANT TOOLTIP BUTTON (for minimal responsiveness)
# ============================================================================

class InstantToolButton(QToolButton):
    """QToolButton that shows tooltip immediately on hover without delay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def _show_tooltip(self):
        if self.toolTip() and self.isEnabled() and self.isVisible():
            pos = self.mapToGlobal(self.rect().bottomLeft())
            QToolTip.showText(pos, self.toolTip(), self, self.rect())

    def enterEvent(self, event):
        super().enterEvent(event)
        self._show_tooltip()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._show_tooltip()

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def event(self, event):
        if event.type() == QEvent.ToolTip:
            self._show_tooltip()
            event.accept()
            return True
        return super().event(event)


class InputDropLabel(QLabel):
    def __init__(self, parent_window):
        super().__init__("Drop or click to choose an input image.\nSupported: PNG, JPG, TIFF, WEBP")
        self.parent_window = parent_window
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(340)
        self.setMouseTracking(True)
        self._set_normal_style()

    def _set_normal_style(self):
        self.setStyleSheet("border: 1px dashed #868b94; border-radius: 6px; color: #b7bcc5; background-color: transparent;")

    def _set_hover_style(self):
        self.setStyleSheet("border: 2px solid #1a73e8; border-radius: 6px; color: #d0d4dc; background-color: transparent;")

    def _set_drag_style(self):
        self.setStyleSheet("border: 2px solid #1a73e8; border-radius: 6px; color: #d0d4dc; background-color: transparent;")

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
    def make_label_with_info(self, label_text, tooltip_text):
        label_widget = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        label_widget.setLayout(row)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.setMinimumWidth(600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        tab_widget = QTabWidget()

        core_tab = QWidget()
        core_form = QFormLayout(core_tab)

        refine_tab = QWidget()
        refine_form = QFormLayout(refine_tab)

        exp_tab = QWidget()
        exp_form = QFormLayout(exp_tab)

        tab_widget.addTab(core_tab, "Core Historical")
        tab_widget.addTab(refine_tab, "Refinement")
        tab_widget.addTab(exp_tab, "Experimental")

        layout.addWidget(tab_widget)

        # build widgets
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
        self.face_factor_edit.setValue(DEFAULT_FACE_FACTOR)

        self.gfpgan_blend_edit = QDoubleSpinBox()
        self.gfpgan_blend_edit.setRange(0.0, 1.0)
        self.gfpgan_blend_edit.setSingleStep(0.01)
        self.gfpgan_blend_edit.setDecimals(2)
        self.gfpgan_blend_edit.setValue(0.45)

        self.gaussian_edit = QDoubleSpinBox()
        self.gaussian_edit.setRange(0.0, 5.0)
        self.gaussian_edit.setSingleStep(0.05)
        self.gaussian_edit.setDecimals(2)
        self.gaussian_edit.setValue(0.75)

        self.identity_preservation_combo = QComboBox()
        self.identity_preservation_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.identity_preservation_combo.setCurrentText("Default")

        self.tonal_transfer_combo = QComboBox()
        self.tonal_transfer_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.tonal_transfer_combo.setCurrentText("Default")

        self.eye_preservation_combo = QComboBox()
        self.eye_preservation_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.eye_preservation_combo.setCurrentText("Default")

        self.structure_matching_combo = QComboBox()
        self.structure_matching_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.structure_matching_combo.setCurrentText("Default")

        self.vgg_appearance_combo = QComboBox()
        self.vgg_appearance_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.vgg_appearance_combo.setCurrentText("Default")

        self.noise_regularize_edit = QDoubleSpinBox()
        self.noise_regularize_edit.setRange(0.0, 1000000.0)
        self.noise_regularize_edit.setSingleStep(1000.0)
        self.noise_regularize_edit.setDecimals(1)
        self.noise_regularize_edit.setValue(50000.0)

        self.lr_edit = QDoubleSpinBox()
        self.lr_edit.setRange(0.0001, 10.0)
        self.lr_edit.setSingleStep(0.01)
        self.lr_edit.setDecimals(4)
        self.lr_edit.setValue(0.1)

        self.camera_lr_edit = QDoubleSpinBox()
        self.camera_lr_edit.setRange(0.0001, 1.0)
        self.camera_lr_edit.setSingleStep(0.001)
        self.camera_lr_edit.setDecimals(4)
        self.camera_lr_edit.setValue(0.01)

        self.mix_layer_start_edit = QSpinBox()
        self.mix_layer_start_edit.setRange(0, 18)
        self.mix_layer_start_edit.setValue(10)

        self.mix_layer_end_edit = QSpinBox()
        self.mix_layer_end_edit.setRange(0, 18)
        self.mix_layer_end_edit.setValue(18)

        # populate tabs with labels containing info icons
        core_form.addRow(
            self.make_label_with_info(
                "Faces to enhance",
                "Chooses which detected faces to process. 'largest' usually focuses on the main subject, while 'all' processes every detected face."
            ),
            self.strategy_combo,
        )
        core_form.addRow(
            self.make_label_with_info(
                "Crop Only",
                "Runs only the face-detection and cropping step, without enhancement or rephotography. Useful for debugging inputs."
            ),
            self.crop_only_checkbox,
        )
        core_form.addRow(
            self.make_label_with_info(
                "Enhancement",
                "Turns face enhancement on or off before rephotography. Enhancement can improve damaged or blurry faces, but may also introduce modern-looking details."
            ),
            self.use_gfpgan_checkbox,
        )
        core_form.addRow(
            self.make_label_with_info(
                "Enhancement blend",
                "Controls how strongly the enhancement result is blended into the face crop. Lower values preserve more of the original image; higher values use more of the enhanced result."
            ),
            self.gfpgan_blend_edit,
        )
        core_form.addRow(
            self.make_label_with_info(
                "Face detection sensitivity (0–1)",
                "Controls how strict face detection is. Higher values are more selective and may reduce false detections; lower values may detect weaker faces but can also pick up mistakes."
            ),
            self.det_threshold_edit,
        )
        core_form.addRow(
            self.make_label_with_info(
                "Face crop expansion",
                "Controls how much area around the detected face is included in the crop. Larger values include more surrounding context such as hair and edges of the head."
            ),
            self.face_factor_edit,
        )
        core_form.addRow(
            self.make_label_with_info(
                "Gaussian blur",
                "Applies blur during reconstruction setup. This can help stabilize optimization and reduce harsh detail, but too much blur can soften important features."
            ),
            self.gaussian_edit,
        )

        refine_form.addRow(
            self.make_label_with_info(
                "Identity preservation",
                "Controls how strongly the reconstruction tries to preserve the person’s facial identity. Higher values usually hold closer to the source face."
            ),
            self.identity_preservation_combo,
        )
        refine_form.addRow(
            self.make_label_with_info(
                "Tonal transfer",
                "Controls how strongly tonal relationships from the source image are transferred. Higher values can better preserve historical light-dark structure, but may constrain reconstruction more strongly."
            ),
            self.tonal_transfer_combo,
        )
        refine_form.addRow(
            self.make_label_with_info(
                "Eye preservation",
                "Controls how strongly the eye region is preserved. Higher values can help maintain gaze and eye shape, but may reduce flexibility in reconstruction."
            ),
            self.eye_preservation_combo,
        )
        refine_form.addRow(
            self.make_label_with_info(
                "Structure matching",
                "Controls how strongly broader image structure and perceptual similarity are matched. Higher values can preserve composition and facial arrangement more strongly."
            ),
            self.structure_matching_combo,
        )
        refine_form.addRow(
            self.make_label_with_info(
                "VGG appearance matching",
                "Controls VGG perceptual appearance loss strength. Lower/off can reduce stylistic color drift from model priors."
            ),
            self.vgg_appearance_combo,
        )

        exp_form.addRow(
            self.make_label_with_info(
                "Noise regularization",
                "Penalizes noisy or unstable latent/image patterns during optimization. Higher values usually suppress artifacts, but may also reduce fine detail."
            ),
            self.noise_regularize_edit,
        )
        exp_form.addRow(
            self.make_label_with_info(
                "Learning rate",
                "Controls the main optimization step size. Higher values move faster but can become unstable; lower values are slower but often safer."
            ),
            self.lr_edit,
        )
        exp_form.addRow(
            self.make_label_with_info(
                "Camera learning rate",
                "Controls the optimization step size for camera-related adjustments. Higher values change viewpoint parameters more quickly."
            ),
            self.camera_lr_edit,
        )
        exp_form.addRow(
            self.make_label_with_info(
                "Mix layer start",
                "Sets the first layer in the latent-mixing range. This affects which levels of facial structure/style are influenced during initialization."
            ),
            self.mix_layer_start_edit,
        )
        exp_form.addRow(
            self.make_label_with_info(
                "Mix layer end",
                "Sets the last layer in the latent-mixing range. Together with Mix layer start, this defines the layer span used for mixing."
            ),
            self.mix_layer_end_edit,
        )

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
        self.gfpgan_blend_edit.setValue(DEFAULT_GFPGAN_BLEND)
        self.det_threshold_edit.setValue(DEFAULT_DET_THRESHOLD)
        self.face_factor_edit.setValue(DEFAULT_FACE_FACTOR)
        self.gaussian_edit.setValue(DEFAULT_GAUSSIAN)
        self.identity_preservation_combo.setCurrentText("Default")
        self.tonal_transfer_combo.setCurrentText("Default")
        self.eye_preservation_combo.setCurrentText("Default")
        self.structure_matching_combo.setCurrentText("Default")
        self.vgg_appearance_combo.setCurrentText("Default")
        self.noise_regularize_edit.setValue(DEFAULT_NOISE_REGULARIZE)
        self.lr_edit.setValue(DEFAULT_LR)
        self.camera_lr_edit.setValue(DEFAULT_CAMERA_LR)
        self.mix_layer_start_edit.setValue(DEFAULT_MIX_LAYER_START)
        self.mix_layer_end_edit.setValue(DEFAULT_MIX_LAYER_END)

    def update_enhancement_controls(self):
        enhancement_enabled = (not self.use_gfpgan_checkbox.isChecked())
        self.gfpgan_blend_edit.setEnabled(enhancement_enabled)

class MainWindow(QMainWindow):
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
    <b>Faces to enhance</b><br/>
    Chooses whether enhancement targets only the largest detected face or every detected face in frame.
    <b>largest</b> is safer for portraits and reduces accidental edits to background faces. <b>all</b> is useful for group photos but can introduce inconsistent quality across subjects.
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

        help_menu = menu_bar.addMenu("&Help")
        parameter_help_action = QAction("Parameter Guide", self)
        parameter_help_action.setShortcut("F1")
        parameter_help_action.triggered.connect(self.show_parameter_help_dialog)
        help_menu.addAction(parameter_help_action)

        readme_action = QAction("Open Project README", self)
        readme_action.triggered.connect(self.open_project_readme)
        help_menu.addAction(readme_action)

        preflight_action = QAction("Run Startup Preflight", self)
        preflight_action.triggered.connect(lambda: self.run_startup_preflight(show_dialog=True, user_initiated=True))
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
                return candidate

        return (app_root / rel).resolve()

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

    def save_persisted_settings(self):
        if not hasattr(self, "settings_store"):
            return
        s = self.settings_store

        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/log_visible", self.log_visible)
        s.setValue("window/log_expanded", self.log_expanded)

        s.setValue("main/input_image", self.input_image_edit.text().strip())
        s.setValue("main/results_root", self.results_root_edit.text().strip())
        s.setValue("main/photo_type", self.photo_type_combo.currentText())
        s.setValue("main/approx_date", self.approx_date_edit.text())
        s.setValue("main/spectral_mode", self.spectral_mode_combo.currentText())
        s.setValue("main/spectral_sensitivity", self.spectral_sensitivity_combo.currentText())
        s.setValue("main/advanced_mode", self.advanced_mode_checkbox.isChecked())
        s.setValue("main/quality_iteration", self.get_selected_preset_value())

        dlg = self.advanced_dialog
        s.setValue("advanced/strategy", dlg.strategy_combo.currentText())
        s.setValue("advanced/crop_only", dlg.crop_only_checkbox.isChecked())
        s.setValue("advanced/use_gfpgan_checkbox", dlg.use_gfpgan_checkbox.isChecked())
        s.setValue("advanced/det_threshold", dlg.det_threshold_edit.value())
        s.setValue("advanced/face_factor", dlg.face_factor_edit.value())
        s.setValue("advanced/gfpgan_blend", dlg.gfpgan_blend_edit.value())
        s.setValue("advanced/gaussian", dlg.gaussian_edit.value())
        s.setValue("advanced/identity", dlg.identity_preservation_combo.currentText())
        s.setValue("advanced/tonal", dlg.tonal_transfer_combo.currentText())
        s.setValue("advanced/eye", dlg.eye_preservation_combo.currentText())
        s.setValue("advanced/structure", dlg.structure_matching_combo.currentText())
        s.setValue("advanced/vgg", dlg.vgg_appearance_combo.currentText())
        s.setValue("advanced/noise_regularize", dlg.noise_regularize_edit.value())
        s.setValue("advanced/lr", dlg.lr_edit.value())
        s.setValue("advanced/camera_lr", dlg.camera_lr_edit.value())
        s.setValue("advanced/mix_start", dlg.mix_layer_start_edit.value())
        s.setValue("advanced/mix_end", dlg.mix_layer_end_edit.value())

    def load_persisted_settings(self):
        if not hasattr(self, "settings_store"):
            return
        s = self.settings_store

        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        self.log_visible = self._to_bool(s.value("window/log_visible"), self.log_visible)
        self.log_expanded = self._to_bool(s.value("window/log_expanded"), self.log_expanded)

        input_image = str(s.value("main/input_image", self.input_image_edit.text().strip()) or "").strip()
        if input_image:
            self.input_image_edit.setText(input_image)
            p = Path(input_image)
            if p.exists():
                self.set_input_preview_image(p)

        results_root = str(s.value("main/results_root", self.results_root_edit.text().strip()) or "").strip()
        if results_root:
            self.results_root_edit.setText(results_root)

        self._set_combo_if_present(self.photo_type_combo, str(s.value("main/photo_type", self.photo_type_combo.currentText()) or ""))
        self.approx_date_edit.setText(str(s.value("main/approx_date", self.approx_date_edit.text()) or ""))
        self._set_combo_if_present(self.spectral_mode_combo, str(s.value("main/spectral_mode", self.spectral_mode_combo.currentText()) or ""))

        advanced_mode = self._to_bool(s.value("main/advanced_mode"), self.advanced_mode_checkbox.isChecked())
        self.advanced_mode_checkbox.setChecked(advanced_mode)
        self._set_iteration_from_value(s.value("main/quality_iteration", DEFAULT_ITERATION))

        dlg = self.advanced_dialog
        self._set_combo_if_present(dlg.strategy_combo, str(s.value("advanced/strategy", dlg.strategy_combo.currentText()) or ""))
        dlg.crop_only_checkbox.setChecked(self._to_bool(s.value("advanced/crop_only"), dlg.crop_only_checkbox.isChecked()))
        dlg.use_gfpgan_checkbox.setChecked(self._to_bool(s.value("advanced/use_gfpgan_checkbox"), dlg.use_gfpgan_checkbox.isChecked()))

        def _set_spin(spin, key, default_value, cast=float):
            raw = s.value(key, default_value)
            try:
                spin.setValue(cast(raw))
            except Exception:
                spin.setValue(default_value)

        _set_spin(dlg.det_threshold_edit, "advanced/det_threshold", dlg.det_threshold_edit.value(), float)
        _set_spin(dlg.face_factor_edit, "advanced/face_factor", dlg.face_factor_edit.value(), float)
        _set_spin(dlg.gfpgan_blend_edit, "advanced/gfpgan_blend", dlg.gfpgan_blend_edit.value(), float)
        _set_spin(dlg.gaussian_edit, "advanced/gaussian", dlg.gaussian_edit.value(), float)
        _set_spin(dlg.noise_regularize_edit, "advanced/noise_regularize", dlg.noise_regularize_edit.value(), float)
        _set_spin(dlg.lr_edit, "advanced/lr", dlg.lr_edit.value(), float)
        _set_spin(dlg.camera_lr_edit, "advanced/camera_lr", dlg.camera_lr_edit.value(), float)
        _set_spin(dlg.mix_layer_start_edit, "advanced/mix_start", dlg.mix_layer_start_edit.value(), int)
        _set_spin(dlg.mix_layer_end_edit, "advanced/mix_end", dlg.mix_layer_end_edit.value(), int)

        self._set_combo_if_present(dlg.identity_preservation_combo, str(s.value("advanced/identity", dlg.identity_preservation_combo.currentText()) or ""))
        self._set_combo_if_present(dlg.tonal_transfer_combo, str(s.value("advanced/tonal", dlg.tonal_transfer_combo.currentText()) or ""))
        self._set_combo_if_present(dlg.eye_preservation_combo, str(s.value("advanced/eye", dlg.eye_preservation_combo.currentText()) or ""))
        self._set_combo_if_present(dlg.structure_matching_combo, str(s.value("advanced/structure", dlg.structure_matching_combo.currentText()) or ""))
        self._set_combo_if_present(dlg.vgg_appearance_combo, str(s.value("advanced/vgg", dlg.vgg_appearance_combo.currentText()) or ""))

        self.update_spectral_sensitivity_ui()
        saved_spectral = str(s.value("main/spectral_sensitivity", self.spectral_sensitivity_combo.currentText()) or "")
        if self.spectral_mode_combo.currentText() == "Manual":
            self._set_combo_if_present(self.spectral_sensitivity_combo, saved_spectral)

        self.log_container.setVisible(self.log_visible)
        self.expand_log_button.setEnabled(self.log_visible)
        self.toggle_log_button.setText("Hide Log" if self.log_visible else "Show Log")

        if self.log_expanded:
            self.log_box.setMinimumHeight(360)
            self.log_box.setMaximumHeight(360)
            self.expand_log_button.setText("Compact Log")
        else:
            self.log_box.setMinimumHeight(150)
            self.log_box.setMaximumHeight(150)
            self.expand_log_button.setText("Expand Log")

        self.update_mode_controls()
        self.update_runtime_label()
        self.update_view_menu_actions()

    def closeEvent(self, event):
        self.save_persisted_settings()
        super().closeEvent(event)

    def _collect_preflight_report(self):
        checks = []

        def add_check(name, status, detail, fix=""):
            checks.append({"name": name, "status": status, "detail": detail, "fix": fix})

        add_check("Application root", "pass", f"Using app root: {self.repo_root}")

        wrapper = self.resolve_resource_path("run_rephoto_with_facecrop.ps1")
        if wrapper.exists():
            add_check("Wrapper script", "pass", f"Found: {wrapper}")
        else:
            add_check(
                "Wrapper script",
                "fail",
                f"Missing: {wrapper}",
                "Ensure run_rephoto_with_facecrop.ps1 is bundled next to the app resources."
            )

        stylegan_ckpt = self.resolve_resource_path(Path("checkpoint") / "stylegan2-ffhq-config-f.pt")
        e4e_ckpt = self.resolve_resource_path(Path("checkpoint") / "e4e_ffhq_encode.pt")
        encoder_dir = self.resolve_resource_path(Path("checkpoint") / "encoder")
        encoder_candidates = list(encoder_dir.glob("checkpoint_*.pt")) if encoder_dir.exists() else []

        if stylegan_ckpt.exists():
            add_check("StyleGAN checkpoint", "pass", f"Found: {stylegan_ckpt.name}")
        else:
            add_check("StyleGAN checkpoint", "fail", "Missing stylegan2-ffhq-config-f.pt", "Run bootstrap_local_assets.ps1 to fetch required assets.")

        if e4e_ckpt.exists():
            add_check("e4e checkpoint", "pass", f"Found: {e4e_ckpt.name}")
        else:
            add_check("e4e checkpoint", "fail", "Missing e4e_ffhq_encode.pt", "Run bootstrap_local_assets.ps1 to fetch required assets.")

        if encoder_candidates:
            add_check("Encoder checkpoint", "pass", f"Found {len(encoder_candidates)} file(s) in checkpoint/encoder")
        else:
            add_check("Encoder checkpoint", "fail", "Missing encoder checkpoint in checkpoint/encoder", "Verify checkpoint/encoder/checkpoint_*.pt exists.")

        gfpgan_root = self.resolve_resource_path(Path("deps") / "GFPGAN")
        if gfpgan_root.exists():
            add_check("GFPGAN dependency", "pass", f"Found: {gfpgan_root}")
        else:
            add_check("GFPGAN dependency", "warn", "deps/GFPGAN not found (enhancement unavailable)", "Install or restore deps/GFPGAN if enhancement is needed.")

        if shutil.which("powershell.exe"):
            add_check("PowerShell runtime", "pass", "powershell.exe is available.")
        else:
            add_check("PowerShell runtime", "fail", "powershell.exe not found in PATH.", "Install/enable PowerShell and ensure it is discoverable.")

        results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
        try:
            results_root.mkdir(parents=True, exist_ok=True)
            probe = results_root / ".preflight_write_test.tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            add_check("Results folder write access", "pass", f"Writable: {results_root}")
        except Exception as e:
            add_check("Results folder write access", "fail", f"Cannot write to {results_root}: {e}", "Pick a writable results folder in the Outputs section.")

        hw = self.get_hardware_info()
        gpu_name = (hw.get("gpu_name") or "Unknown GPU").strip()
        if gpu_name and gpu_name.lower() != "unknown gpu":
            add_check("GPU detection", "pass", f"Detected GPU: {gpu_name}")
        else:
            add_check("GPU detection", "warn", "No GPU detected via nvidia-smi.", "Install NVIDIA driver/CUDA stack, or expect slower CPU-only behavior.")

        fail_count = sum(1 for c in checks if c["status"] == "fail")
        warn_count = sum(1 for c in checks if c["status"] == "warn")
        pass_count = sum(1 for c in checks if c["status"] == "pass")
        return {
            "checks": checks,
            "fail": fail_count,
            "warn": warn_count,
            "pass": pass_count,
        }

    def _preflight_report_plain_text(self, report):
        lines = [
            "Startup Preflight Report",
            f"Summary: {report['pass']} pass, {report['warn']} warn, {report['fail']} fail",
            "",
        ]
        for check in report["checks"]:
            lines.append(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
            if check.get("fix") and check["status"] != "pass":
                lines.append(f"  Fix: {check['fix']}")
        return "\n".join(lines)

    def _preflight_report_html(self, report):
        status_color = {"pass": "#6fcf97", "warn": "#f2c94c", "fail": "#eb5757"}
        rows = []
        for c in report["checks"]:
            color = status_color.get(c["status"], "#b7bcc5")
            fix_html = f"<br/><span style='color:#b7bcc5'><b>Fix:</b> {c['fix']}</span>" if c.get("fix") and c["status"] != "pass" else ""
            rows.append(
                f"<li><span style='color:{color}'><b>{c['status'].upper()}</b></span> "
                f"<b>{c['name']}</b>: {c['detail']}{fix_html}</li>"
            )
        return (
            f"<h3>Startup Preflight</h3>"
            f"<p><b>Summary:</b> {report['pass']} pass, {report['warn']} warn, {report['fail']} fail</p>"
            f"<ul>{''.join(rows)}</ul>"
        )

    def show_preflight_dialog(self, report):
        dialog = QDialog(self)
        dialog.setWindowTitle("Startup Preflight Report")
        dialog.setMinimumSize(820, 520)
        layout = QVBoxLayout(dialog)

        browser = QTextBrowser()
        browser.setHtml(self._preflight_report_html(report))
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        copy_button = QPushButton("Copy Report")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self._preflight_report_plain_text(report)))
        buttons.addButton(copy_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def run_startup_preflight(self, show_dialog=False, user_initiated=False):
        report = self._collect_preflight_report()
        self.last_preflight_report = report

        self.log_box.append("")
        self.log_box.append("=== Startup preflight ===")
        for check in report["checks"]:
            self.log_box.append(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
            if check.get("fix") and check["status"] != "pass":
                self.log_box.append(f"  Fix: {check['fix']}")
        self.log_box.append(f"Preflight summary: {report['pass']} pass, {report['warn']} warn, {report['fail']} fail")

        should_show = show_dialog and (user_initiated or report["fail"] > 0)
        if should_show:
            self.show_preflight_dialog(report)

    def _format_elapsed_for_summary(self, elapsed_seconds):
        if elapsed_seconds is None:
            return "N/A"
        elapsed = int(max(0, elapsed_seconds))
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _capture_run_context(self):
        return {
            "input_image": self.input_image_edit.text().strip(),
            "results_root": self.results_root_edit.text().strip(),
            "quality": self.get_selected_preset_value(),
            "photo_type": self.photo_type_combo.currentText(),
            "approx_date": self.approx_date_edit.text().strip(),
            "spectral_sensitivity": self.get_spectral_sensitivity_value(),
            "crop_only": self.advanced_dialog.crop_only_checkbox.isChecked(),
            "enhancement_enabled": (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()) and self.gfpgan_is_available(),
            "faces_to_enhance": self.advanced_dialog.strategy_combo.currentText(),
        }

    def _build_run_summary_text(self, success, exit_code, elapsed_seconds, output_path=None, launch_error=None):
        ctx = self.current_run_summary_context or {}
        status = "Success" if success else "Failed"
        if launch_error:
            status = "Launch Error"

        lines = [
            "Run Summary",
            "-----------",
            f"Status: {status}",
            f"Exit code: {exit_code}",
            f"Elapsed: {self._format_elapsed_for_summary(elapsed_seconds)}",
            "",
            "Key settings:",
            f"- Quality: {ctx.get('quality', 'N/A')}",
            f"- Photo type: {ctx.get('photo_type', 'N/A')}",
            f"- Approximate date: {ctx.get('approx_date', '') or 'N/A'}",
            f"- Spectral sensitivity: {ctx.get('spectral_sensitivity', 'N/A')}",
            f"- Faces to enhance: {ctx.get('faces_to_enhance', 'N/A')}",
            f"- Enhancement enabled: {'Yes' if ctx.get('enhancement_enabled') else 'No'}",
            f"- Crop-only: {'Yes' if ctx.get('crop_only') else 'No'}",
            "",
            f"Input image: {ctx.get('input_image', 'N/A')}",
            f"Results root: {ctx.get('results_root', 'N/A')}",
            f"Output image: {str(output_path) if output_path else 'N/A'}",
        ]

        if launch_error:
            lines.extend(["", f"Launch error: {launch_error}"])

        return "\n".join(lines)

    def show_run_summary_dialog(self, success, exit_code, elapsed_seconds, output_path=None, launch_error=None):
        if self.current_run_summary_context is None:
            return

        summary_text = self._build_run_summary_text(
            success=success,
            exit_code=exit_code,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            launch_error=launch_error,
        )
        self.last_run_summary_text = summary_text

        dialog = QDialog(self)
        dialog.setWindowTitle("Run Summary")
        dialog.setMinimumSize(760, 500)
        layout = QVBoxLayout(dialog)

        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(summary_text)
        layout.addWidget(body)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        copy_button = QPushButton("Copy Run Details")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(summary_text))
        buttons.addButton(copy_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-Travel Rephotography")

        self.app_root = self.detect_app_root()
        self.repo_root = self.app_root
        self.wrapper_script = self.resolve_resource_path("run_rephoto_with_facecrop.ps1")
        self.settings_store = QSettings("TimeTravelRephotography", "DesktopGUI")
        self.process = None
        self.run_started_at = None
        self.current_run_summary_context = None
        self.last_run_summary_text = ""
        self.last_preflight_report = None

        self.preprocess_stage = "idle"
        self.rephoto_stage = None
        self.rephoto_stage_name = None
        self.rephoto_stage_current = 0
        self.rephoto_stage_total = 0
        self.rephoto_total_done_before_stage = 0
        self.rephoto_total_work = 0
        self.rephoto_step_pair = (250, 750)

        # run phase state for gating progress bars
        self.current_run_phase = "idle"  # idle, preprocess, rephoto, done, cancelled, crop_only_done

        # Path tracking from stdout to avoid recursive folder scans
        self.current_crop_output_dir = None
        self.current_gfpgan_output_dir = None
        self.current_blended_faces_dir = None
        self.current_results_dir = None
        self.current_manifest_path = None

        # Stage timing instrumentation
        self.stage_started_at = {}  # {"crop": timestamp, "enhance": timestamp, ...}
        self.stage_elapsed = {}    # {"crop": seconds, "enhance": seconds, ...}

        self.log_expanded = False
        self.log_visible = False

        # Preview state
        self.input_pixmap = None
        self.result_pixmap = None
        self.last_result_image_path = None

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

        central = QWidget()
        scroll_area.setWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setAlignment(Qt.AlignTop)
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
        self.advanced_dialog.strategy_combo.setCurrentText("largest")
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

        self.advanced_dialog.crop_only_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_runtime_label)

        # --- Iteration slider ---
        self.basic_iter_values = DEFAULT_BASIC_ITER_VALUES
        self.advanced_iter_values = DEFAULT_ADVANCED_ITER_VALUES
        self.iter_values = self.basic_iter_values

        self.iter_slider = QSlider(Qt.Horizontal)
        self.advanced_mode_checkbox = QCheckBox("Advanced")
        self.advanced_mode_checkbox.setChecked(False)
        self.advanced_mode_checkbox.toggled.connect(self.update_iteration_mode)
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
        quality_slider_row.addWidget(self.advanced_mode_checkbox, 0, Qt.AlignVCenter)

        self.quality_widget = QWidget()
        quality_layout = QVBoxLayout()
        quality_layout.setContentsMargins(0, 3, 0, 2)
        quality_layout.setSpacing(4)
        self.quality_widget.setLayout(quality_layout)

        quality_layout.addLayout(quality_header_row)
        quality_layout.addLayout(quality_slider_row)

        # --- Spectral sensitivity inputs ---
        self.photo_type_combo = QComboBox()
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

        self.spectral_mode_combo = QComboBox()
        self.spectral_mode_combo.addItems(["Auto", "Manual"])

        self.spectral_sensitivity_combo = QComboBox()
        self.spectral_sensitivity_combo.addItems(["Blue-sensitive", "Orthochromatic", "Panchromatic"])

        # Connect signals for spectral sensitivity inference
        self.spectral_mode_combo.currentTextChanged.connect(self.update_spectral_sensitivity_ui)
        self.photo_type_combo.currentTextChanged.connect(self.update_spectral_sensitivity_ui)
        self.approx_date_edit.textChanged.connect(self.update_spectral_sensitivity_ui)

        form_layout.addRow(self.make_form_label("Photo type / process"), self.photo_type_combo)
        form_layout.addRow(self.make_form_label("Approximate date"), self.approx_date_edit)
        form_layout.addRow(
            self.make_label_with_info(
                "Spectral sensitivity mode",
                "Auto recalculates spectral sensitivity based on photo type and date; Manual preserves explicit user choice."
            ),
            self.spectral_mode_combo,
        )
        form_layout.addRow(
            self.make_label_with_info(
                "Spectral sensitivity",
                "The historical spectral sensitivity model used by the pipeline. In Auto mode, this is computed from process type and date."
            ),
            self.spectral_sensitivity_combo,
        )

        self.advanced_settings_button = QPushButton("Advanced Settings...")
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings_dialog)
        self.advanced_settings_button.setMinimumHeight(30)
        form_layout.addRow(self.make_form_label("Advanced"), self.advanced_settings_button)

        # Add Quality row as proper form row
        form_layout.addRow(self.quality_widget)

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

        progress_widget = QWidget()
        progress_widget.setLayout(progress_row)
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
        self.run_button.clicked.connect(self.run_wrapper)
        self.run_button.setMinimumHeight(34)
        self.run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.run_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #1a73e8; color: white; border: none; border-radius: 4px; } "
            "QPushButton:hover { background-color: #1455c0; } QPushButton:pressed { background-color: #0d47a1; }"
        )

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
        self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumHeight(34)
        self.cancel_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cancel_button.setStyleSheet(secondary_button_style)

        self.reset_button = QPushButton("Reset Defaults")
        self.reset_button.clicked.connect(self.reset_form_defaults)
        self.reset_button.setMinimumHeight(34)
        self.reset_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.quit_button)
        button_row.addWidget(self.toggle_log_button)

        # --- Previews (Input on left, Result on right) ---
        previews_group = QGroupBox("Previews")
        previews_layout = QHBoxLayout()
        previews_layout.setContentsMargins(6, 6, 6, 6)
        previews_layout.setSpacing(6)
        previews_group.setLayout(previews_layout)
        previews_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        input_group = QWidget()
        input_layout = QVBoxLayout()
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

        result_group = QWidget()
        result_layout = QVBoxLayout()
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
        result_header.addWidget(result_title)
        result_header.addWidget(result_divider, 1)
        result_layout.addLayout(result_header)

        self.result_preview_label = QLabel("No result image yet.\nRun to generate a preview.")
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setFixedHeight(300)
        self.result_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.result_preview_label.setStyleSheet("border: 1px solid #868b94; border-radius: 4px; color: #b7bcc5;")
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
        self.result_stage_dot_count = 0
        self._result_stage_timer = QTimer(self)
        self._result_stage_timer.setInterval(450)
        self._result_stage_timer.timeout.connect(self.update_result_stage_overlay_animation)

        self.open_image_location_button = QPushButton("Open Image Location")
        self.open_image_location_button.clicked.connect(self.open_result_image_location)
        self.open_image_location_button.setEnabled(False)
        self.open_image_location_button.setMinimumHeight(32)
        self.open_image_location_button.setStyleSheet(secondary_button_style)
        result_layout.addWidget(self.open_image_location_button)

        previews_layout.addWidget(input_group)
        previews_layout.addWidget(result_group)

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
        self.controls_container.setLayout(controls_layout)
        controls_layout.addWidget(settings_container)
        controls_layout.addWidget(progress_widget)
        controls_layout.addWidget(outputs_group)
        controls_layout.addLayout(button_row)

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

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)
        self.log_box.setMaximumHeight(150)
        self.log_box.append("GUI loaded successfully.")
        log_layout.addWidget(self.log_box)

        self.status_label = QLabel("Status: Ready")
        log_layout.addWidget(self.status_label)

        self.expand_log_button.clicked.connect(self.toggle_log_size)
        self.toggle_log_button.clicked.connect(self.toggle_log_visibility)

        self.expand_log_button.setEnabled(self.log_visible)
        self.log_container.setVisible(self.log_visible)
        controls_layout.addWidget(self.log_container)
        controls_layout.addStretch(1)

        self.setup_menu_bar()

        # --- Responsive page layout ---
        self.previews_group = previews_group
        self.previews_split_layout = previews_layout
        self.content_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        main_layout.addLayout(self.content_layout)
        self._wide_layout_active = None
        self.apply_responsive_layout(force=True)
        self.load_persisted_settings()
        self.apply_responsive_layout(force=True)

        # Initial state
        self.update_mode_controls()
        self.update_runtime_label()
        if not self.gfpgan_is_available():
            self.log_box.append("GFPGAN not found (deps\\GFPGAN). Enhancement is disabled.")
        else:
            self.log_box.append("GFPGAN found. Enhancement is available.")
        self.run_startup_preflight(show_dialog=True)

    # ------------------------------
    # Qt / window events
    # ------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_responsive_layout()
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()
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

        if use_wide_layout:
            preview_side = self._compute_wide_preview_side()
            preview_pane_width = preview_side + 56
            preview_pane_height = (2 * preview_side) + 112

            self.content_layout.setDirection(QBoxLayout.LeftToRight)
            self.previews_split_layout.setDirection(QBoxLayout.TopToBottom)
            self.input_preview_label.setFixedSize(preview_side, preview_side)
            self.input_preview_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.result_preview_label.setFixedSize(preview_side, preview_side)
            self.result_preview_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            self.previews_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
            self.previews_group.setMinimumWidth(preview_pane_width)
            self.previews_group.setMaximumWidth(preview_pane_width)
            self.previews_group.setMinimumHeight(preview_pane_height)
            self.previews_group.setMaximumHeight(preview_pane_height)
            self.content_layout.addWidget(self.controls_container, 6, Qt.AlignTop)
            self.content_layout.addWidget(self.previews_group, 4, Qt.AlignTop)
        else:
            self.content_layout.setDirection(QBoxLayout.TopToBottom)
            self.previews_split_layout.setDirection(QBoxLayout.LeftToRight)
            self.input_preview_label.setMinimumWidth(0)
            self.input_preview_label.setMaximumWidth(16777215)
            self.input_preview_label.setFixedHeight(300)
            self.input_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.result_preview_label.setMinimumWidth(0)
            self.result_preview_label.setMaximumWidth(16777215)
            self.result_preview_label.setFixedHeight(300)
            self.result_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.controls_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.previews_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.previews_group.setMinimumWidth(0)
            self.previews_group.setMaximumWidth(16777215)
            self.previews_group.setMinimumHeight(0)
            self.previews_group.setMaximumHeight(16777215)
            self.content_layout.addWidget(self.previews_group, 0)
            self.content_layout.addWidget(self.controls_container, 0)

        self._apply_mode_layout_profile(use_wide_layout)
        self.refresh_input_preview_scale()
        self.refresh_result_preview_scale()

    def _compute_wide_preview_side(self):
        # Keep square previews while constraining right pane height and width in wide mode.
        side_by_height = int((self.height() - 280) / 2)
        side_by_width = int((self.width() * 0.38) - 56)
        side = min(side_by_height, side_by_width)
        return max(220, min(380, side))

    def _apply_mode_layout_profile(self, use_wide_layout):
        if use_wide_layout:
            self.controls_layout.setSpacing(6)
            self.settings_layout.setSpacing(4)
            self.form_layout.setVerticalSpacing(5)
            self.progress_bars_layout.setSpacing(1)
            self.outputs_layout.setVerticalSpacing(2)
            self.outputs_layout.setContentsMargins(6, 3, 6, 3)
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
        return self.resolve_resource_path(Path("deps") / "GFPGAN").exists()

    
    def update_mode_controls(self):
        """Update UI state based on GFPGAN availability and crop-only mode."""
        crop_only = self.advanced_dialog.crop_only_checkbox.isChecked()
        gfpgan_available = self.gfpgan_is_available()

        # If GFPGAN is not available: enhancement must be disabled
        if not gfpgan_available:
            self.advanced_dialog.use_gfpgan_checkbox.setChecked(True)  # Checked = disabled
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
            self.advanced_dialog.gfpgan_blend_edit.setEnabled(False)
            return

        # If crop-only mode: enhancement must be disabled
        if crop_only:
            self.advanced_dialog.use_gfpgan_checkbox.setChecked(True)  # Checked = disabled
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
            self.advanced_dialog.gfpgan_blend_edit.setEnabled(False)
            return

        # Otherwise: enhancement can be toggled
        self.advanced_dialog.use_gfpgan_checkbox.setEnabled(True)
        # Update blend spinner based on checkbox state
        enhancement_enabled = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
        self.advanced_dialog.gfpgan_blend_edit.setEnabled(enhancement_enabled)

    def update_iteration_mode(self):
        current = self.iter_values[self.iter_slider.value()]
        self.iter_values = self.advanced_iter_values if self.advanced_mode_checkbox.isChecked() else self.basic_iter_values
        self.iter_slider.setMaximum(len(self.iter_values) - 1)

        closest_index = min(range(len(self.iter_values)), key=lambda i: abs(self.iter_values[i] - current))
        self.iter_slider.setValue(closest_index)

    def reset_progress_bars(self):
        self.preprocess_progress_bar.setValue(0)
        self.preprocess_progress_bar.setFormat("Preprocess ready")
        self.rephoto_progress_bar.setValue(0)
        self.rephoto_status_text = "Waiting..."
        self.update_rephoto_bar_format()
        self.preprocess_stage = "idle"
        self.rephoto_stage = None
        self.rephoto_stage_name = None
        self.rephoto_stage_current = 0
        self.rephoto_stage_total = 0
        self.rephoto_total_done_before_stage = 0
        self.rephoto_total_work = 0
        self.rephoto_step_pair = (250, 750)
        self.current_run_phase = "idle"

    def get_elapsed_display_text(self):
        if self.run_started_at is None:
            return "0:00"

        elapsed = int(time.time() - self.run_started_at)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def update_rephoto_bar_format(self):
        elapsed_text = self.get_elapsed_display_text()
        self.rephoto_progress_bar.setFormat(f"{self.rephoto_status_text} %p% | {elapsed_text}")

    def get_effective_rephoto_steps(self):
        preset = self.get_selected_preset_value()
        if preset == 375:
            return (125, 375)
        elif preset == 750:
            return (250, 750)
        else:
            return (250, preset)

    def set_preprocess_progress(self, value, text=None):
        value = max(0, min(100, value))
        self.preprocess_progress_bar.setValue(value)
        if text:
            self.preprocess_progress_bar.setFormat(text)

    def set_rephoto_progress(self, value, text=None):
        value = max(0, min(100, value))
        self.rephoto_progress_bar.setValue(value)
        if text:
            self.rephoto_status_text = text
        self.update_rephoto_bar_format()

    def start_rephoto_progress_tracking(self):
        self.rephoto_step_pair = self.get_effective_rephoto_steps()
        self.rephoto_total_work = self.rephoto_step_pair[0] + self.rephoto_step_pair[1]
        self.rephoto_stage = None
        self.rephoto_stage_name = None
        self.rephoto_stage_current = 0
        self.rephoto_stage_total = 0
        self.rephoto_total_done_before_stage = 0

    def update_rephoto_progress_from_iteration(self, current_iter, total_iter):
        if not self.rephoto_stage_name:
            return
        if self.rephoto_stage_name == "32x32":
            self.rephoto_total_done_before_stage = 0
            self.rephoto_stage_total = self.rephoto_step_pair[0]
        elif self.rephoto_stage_name == "64x64":
            self.rephoto_total_done_before_stage = self.rephoto_step_pair[0]
            self.rephoto_stage_total = self.rephoto_step_pair[1]
        overall_done = self.rephoto_total_done_before_stage + current_iter
        percent = round(100 * overall_done / self.rephoto_total_work)
        self.set_rephoto_progress(percent, "Processing")

        self.update_iteration_label()

    def update_iteration_label(self):
        v = self.iter_values[self.iter_slider.value()]
        self.quality_value_label.setText(str(v))
        # Show default notation only if value matches DEFAULT_ITERATION
        if v == DEFAULT_ITERATION:
            self.quality_default_label.setText("(default)")
            self.quality_default_label.setVisible(True)
        else:
            self.quality_default_label.setVisible(False)

        if hasattr(self, "runtime_label") and hasattr(self, "update_runtime_label"):
            self.update_runtime_label()

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

    def is_true_process_type(self, photo_type):
        """Return True only for actual photographic processes."""
        return photo_type in ("Daguerreotype", "Ambrotype", "Tintype / Ferrotype")

    def infer_spectral_sensitivity(self):
        """Infer spectral sensitivity based on photo type and date. Returns one of:
        'Blue-sensitive', 'Orthochromatic', 'Panchromatic'
        """
        photo_type = self.photo_type_combo.currentText()
        approx_year = self.parse_approximate_year(self.approx_date_edit.text())

        # PRIORITY 2: True photographic processes
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

    def set_controls_for_running(self, is_running):
        self.run_button.setEnabled(not is_running)
        self.cancel_button.setEnabled(is_running)
        self.reset_button.setEnabled(not is_running)
        self.quit_button.setEnabled(not is_running)

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
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
        else:
            self.update_mode_controls()

    # ------------------------------
    # Progress tracking
    # ------------------------------
    def update_progress_from_line(self, line: str):
        s = (line or "").strip()

        # === Path tracking from stdout ===
        if "Crop output dir:" in s:
            self.current_crop_output_dir = s.split("Crop output dir:")[-1].strip()
        if "GFPGAN output:" in s:
            self.current_gfpgan_output_dir = s.split("GFPGAN output:")[-1].strip()
        if "GFPGAN blended faces:" in s:
            self.current_blended_faces_dir = s.split("GFPGAN blended faces:")[-1].strip()
        if "Results:" in s and "Manifest:" not in s:
            self.current_results_dir = s.split("Results:")[-1].strip()
        if "Manifest:" in s:
            self.current_manifest_path = s.split("Manifest:")[-1].strip()

        # === Stage timing instrumentation ===
        current_time = time.time()
        
        # Record when stages start
        if "=== Face crop step" in s and "crop" not in self.stage_started_at:
            self.stage_started_at["crop"] = current_time
        if "=== GFPGAN step" in s and "enhance" not in self.stage_started_at:
            self.stage_started_at["enhance"] = current_time
        if "=== Rephoto step" in s and "rephoto" not in self.stage_started_at:
            self.stage_started_at["rephoto"] = current_time
        
        # Record when stages end and compute elapsed time
        if s.startswith("Cropped face count:") and "crop" in self.stage_started_at and "crop" not in self.stage_elapsed:
            self.stage_elapsed["crop"] = current_time - self.stage_started_at["crop"]
        if "GFPGAN blended faces:" in s and "enhance" in self.stage_started_at and "enhance" not in self.stage_elapsed:
            self.stage_elapsed["enhance"] = current_time - self.stage_started_at["enhance"]

        # === Rephoto start marker, switch phases ===
        if s.startswith("=== Rephoto step"):
            if self.current_run_phase == "preprocess":
                self.set_preprocess_progress(100, "Preprocessing complete")
                self.preprocess_stage = "complete"
                self.current_run_phase = "rephoto"
                self.start_rephoto_progress_tracking()
                self.set_rephoto_progress(0, "Starting rephoto")
                self.set_result_stage_overlay("Rephotographing")
            return

        # Crop‑only skip notice
        if s.startswith("CropOnly requested. Skipping rephoto step."):
            # finalize preprocessing and leave rephoto untouched
            self.set_preprocess_progress(100, "Preprocessing complete")
            self.set_rephoto_progress(0, "Skipped")
            self.current_run_phase = "crop_only_done"
            return

        # Preprocessing stage updates (only before rephoto begins)
        if self.current_run_phase == "preprocess":
            if s.startswith("=== Face crop step"):
                self.set_preprocess_progress(20, "Cropping faces")
                self.preprocess_stage = "cropping"
                self.set_result_stage_overlay("Cropping")
                return

            if s.startswith("=== GFPGAN step"):
                self.set_preprocess_progress(60, "Enhancing faces")
                self.preprocess_stage = "enhancing"
                self.set_result_stage_overlay("Enhancing")
                return

            if s.startswith("=== GPU pre-check"):
                self.set_preprocess_progress(80, "GPU pre-check")
                self.preprocess_stage = "gpu_check"
                return

            # Parse crop count and preview crop if found
            m = CROPPED_FACE_COUNT_RE.search(s)
            if m:
                n = int(m.group(1))
                self.set_preprocess_progress(40, f"Crops ready ({n})")
                self.preprocess_stage = "crops_ready"
                crop_image = self.find_latest_crop_output(after_epoch=self.run_started_at)
                if crop_image:
                    self.preview_stage_image_if_found(crop_image, "Cropping")
                return

            # Watch for GFPGAN blended output
            if "GFPGAN blended faces:" in s or "GFPGAN output:" in s:
                enhanced_image = self.find_latest_enhanced_output(after_epoch=self.run_started_at)
                if enhanced_image:
                    self.preview_stage_image_if_found(enhanced_image, "Enhancing")
                return

        # Rephoto-specific updates (only during rephoto phase)
        if self.current_run_phase == "rephoto":
            if "Optimizing 32x32" in s:
                self.rephoto_stage = "32x32"
                self.rephoto_stage_name = "32x32"
                self.rephoto_stage_current = 0
                self.rephoto_stage_total = self.rephoto_step_pair[0]
                return

            if "Optimizing 64x64" in s:
                self.rephoto_stage = "64x64"
                self.rephoto_stage_name = "64x64"
                self.rephoto_stage_current = 0
                self.rephoto_stage_total = self.rephoto_step_pair[1]
                return

            m = ITER_PROGRESS_RE.search(s)
            if m:
                current = int(m.group(1))
                total = int(m.group(2))

                allowed_totals = {self.rephoto_step_pair[0], self.rephoto_step_pair[1]}
                if total not in allowed_totals:
                    return

                if total == self.rephoto_step_pair[0]:
                    self.rephoto_stage = "32x32"
                    self.rephoto_stage_name = "32x32"
                elif total == self.rephoto_step_pair[1]:
                    self.rephoto_stage = "64x64"
                    self.rephoto_stage_name = "64x64"

                self.rephoto_stage_total = total
                self.update_rephoto_progress_from_iteration(current, total)
                return

        # Completion marker
        if s == "Done.":
            if self.current_run_phase == "rephoto":
                self.set_rephoto_progress(100, "Done")
            elif self.current_run_phase == "crop_only_done":
                self.set_rephoto_progress(0, "Skipped")
            self.current_run_phase = "done"
            return
    # ------------------------------
    # Input / output selection
    # ------------------------------
    def reset_form_defaults(self):
        # Reset all advanced-settings values to their defaults.
        self.advanced_dialog.restore_defaults()

        self.advanced_mode_checkbox.setChecked(False)
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

        old_tonal = dlg.tonal_transfer_combo.currentText()
        old_noise_regularize = dlg.noise_regularize_edit.value()
        old_eye = dlg.eye_preservation_combo.currentText()
        old_structure = dlg.structure_matching_combo.currentText()
        old_vgg_appearance = dlg.vgg_appearance_combo.currentText()
        old_lr = dlg.lr_edit.value()
        old_camera_lr = dlg.camera_lr_edit.value()
        old_mix_layer_start = dlg.mix_layer_start_edit.value()
        old_mix_layer_end = dlg.mix_layer_end_edit.value()
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
            self.update_mode_controls()
            self.update_runtime_label()
            self.log_box.append("Advanced settings updated.")
        else:
            dlg.strategy_combo.setCurrentText(old_strategy)
            dlg.crop_only_checkbox.setChecked(old_crop_only)
            dlg.use_gfpgan_checkbox.setChecked(old_use_gfpgan)
            dlg.det_threshold_edit.setValue(old_det_threshold)
            dlg.face_factor_edit.setValue(old_face_factor)
            dlg.gfpgan_blend_edit.setValue(old_gfpgan_blend)
            dlg.gaussian_edit.setValue(old_gaussian)
            # Restore main window spectral widgets
            self.photo_type_combo.setCurrentText(old_photo_type)
            self.approx_date_edit.setText(old_approx_date)
            self.spectral_mode_combo.setCurrentText(old_spectral_mode)
            self.spectral_sensitivity_combo.setCurrentText(old_spectral)
            dlg.tonal_transfer_combo.setCurrentText(old_tonal)
            dlg.eye_preservation_combo.setCurrentText(old_eye)
            dlg.structure_matching_combo.setCurrentText(old_structure)
            dlg.vgg_appearance_combo.setCurrentText(old_vgg_appearance)
            dlg.noise_regularize_edit.setValue(old_noise_regularize)
            dlg.lr_edit.setValue(old_lr)
            dlg.camera_lr_edit.setValue(old_camera_lr)
            dlg.mix_layer_start_edit.setValue(old_mix_layer_start)
            dlg.mix_layer_end_edit.setValue(old_mix_layer_end)
            dlg.gaussian_edit.setValue(old_gaussian)
            self.update_spectral_sensitivity_ui()
            self.update_mode_controls()
            self.update_runtime_label()

    # ------------------------------
    # Runtime estimation / hardware
    # ------------------------------
    def get_selected_preset_value(self):
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
            self.runtime_label.setText(f"{days} day {hours} hr {mins} min")
        elif hours > 0:
            self.runtime_label.setText(f"{hours} hr {mins} min")
        else:
            self.runtime_label.setText(f"{mins} min")

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
        tonal_value = str(self.get_tonal_transfer_value())
        eye_value = str(self.get_eye_preservation_value())
        structure_value = str(self.get_structure_matching_value())
        vgg_value = str(self.get_vgg_appearance_value())
        spectral_value = self.get_spectral_sensitivity_value()
        gaussian_value = self.advanced_dialog.gaussian_edit.text().strip()
        noise_regularize_value = self.advanced_dialog.noise_regularize_edit.text().strip()
        lr_value = self.advanced_dialog.lr_edit.text().strip()
        camera_lr_value = self.advanced_dialog.camera_lr_edit.text().strip()
        mix_layer_start_value = str(self.advanced_dialog.mix_layer_start_edit.value())
        mix_layer_end_value = str(self.advanced_dialog.mix_layer_end_edit.value())

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
            "-SpectralSensitivity",
            spectral_value,
            "-Gaussian",
            gaussian_value,
            "-VGGFace",
            identity_value,
            "-ColorTransfer",
            tonal_value,
            "-Eye",
            eye_value,
            "-Contextual",
            structure_value,
            "-VGG",
            vgg_value,
            "-NoiseRegularize",
            noise_regularize_value,
            "-LR",
            lr_value,
            "-CameraLR",
            camera_lr_value,
            "-MixLayerStart",
            mix_layer_start_value,
            "-MixLayerEnd",
            mix_layer_end_value,
            "-ResultsRoot",
            results_root,
        ]

        if self.advanced_dialog.crop_only_checkbox.isChecked():
            command.append("-CropOnly")
        else:
            # Enhancement is enabled only if: checkbox is unchecked (not disabled) AND GFPGAN is available
            enhancement_enabled = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()) and self.gfpgan_is_available()
            if enhancement_enabled:
                command.extend([
                    "-UseGFPGAN",
                    "-GFPGANBlend",
                    self.advanced_dialog.gfpgan_blend_edit.text().strip(),
                ])

        return command

    # ------------------------------
    # Result discovery / preview
    # ------------------------------
    def _find_newest_image_in_tree(self, root: Path, after_epoch: float | None = None, preferred_substring: str | None = None):
        """Return newest image in a tree, optionally preferring paths containing a substring."""
        if root is None or (not root.exists()):
            return None

        newest_any = None
        newest_any_mtime = -1.0
        newest_preferred = None
        newest_preferred_mtime = -1.0

        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if after_epoch is not None and mtime < after_epoch:
                continue

            if mtime > newest_any_mtime:
                newest_any_mtime = mtime
                newest_any = p

            if preferred_substring and preferred_substring in str(p) and mtime > newest_preferred_mtime:
                newest_preferred_mtime = mtime
                newest_preferred = p

        return newest_preferred or newest_any

    def find_latest_image(self, root: Path, after_epoch: float | None):
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
            newest = self._find_newest_image_in_tree(tracked_dir)
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
        self.position_result_stage_overlay()

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

    def position_result_stage_overlay(self):
        """Position the stage overlay at bottom-left of the result preview."""
        self.result_stage_overlay.adjustSize()
        x = 10
        y = max(0, self.result_preview_label.height() - self.result_stage_overlay.height() - 10)
        self.result_stage_overlay.move(x, y)

    def set_result_stage_overlay(self, base_text):
        """Start displaying an animated stage overlay with the given base text."""
        self.result_stage_base_text = base_text
        self.result_stage_dot_count = 0
        self.result_stage_overlay.setVisible(True)
        self._result_stage_timer.start()
        self.update_result_stage_overlay_animation()

    def clear_result_stage_overlay(self):
        """Stop and hide the stage overlay."""
        self._result_stage_timer.stop()
        self.result_stage_base_text = ""
        self.result_stage_overlay.setVisible(False)

    def update_result_stage_overlay_animation(self):
        """Update the animated dots on the stage overlay."""
        if not self.result_stage_base_text:
            return
        self.result_stage_dot_count = (self.result_stage_dot_count % 3) + 1
        text = f"{self.result_stage_base_text}{'.' * self.result_stage_dot_count}"
        self.result_stage_overlay.setText(text)
        self.position_result_stage_overlay()

    def find_latest_enhanced_output(self, after_epoch=None):
        """Find the newest enhanced/blended image from GFPGAN output folders.
        Uses tracked path from stdout if available, otherwise does recursive scan."""

        # Try tracked blended_faces dir first
        if self.current_blended_faces_dir:
            tracked_dir = Path(self.current_blended_faces_dir)
            newest = self._find_newest_image_in_tree(tracked_dir)
            if newest:
                return newest

        # Try tracked GFPGAN output dir
        if self.current_gfpgan_output_dir:
            tracked_dir = Path(self.current_gfpgan_output_dir)
            newest = self._find_newest_image_in_tree(tracked_dir)
            if newest:
                return newest

        # Fallback: recursive scan of default gfpgan directory
        gfpgan_dir = self.repo_root / "preprocess" / "gfpgan_runs"
        return self._find_newest_image_in_tree(
            gfpgan_dir,
            after_epoch=after_epoch,
            preferred_substring="blended_faces",
        )

    def preview_stage_image_if_found(self, image_path, stage_name):
        """Preview an image if it exists and update stage overlay."""
        if image_path is not None and image_path.exists():
            self.set_result_preview_image(image_path)
            self.set_result_stage_overlay(stage_name)

    def update_elapsed_label(self):
        self.update_rephoto_bar_format()

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
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.log_box.append(f"Process finished with exit code: {exit_code}")
        elapsed_seconds = (time.time() - self.run_started_at) if self.run_started_at is not None else None
        final_output_path = None

        if exit_code == 0:
            # finalize bars according to phase
            if self.current_run_phase == "rephoto":
                self.set_rephoto_progress(100, "Complete")
            elif self.current_run_phase == "crop_only_done":
                self.set_rephoto_progress(0, "Skipped")
            self.current_run_phase = "done"
            self.status_label.setText("Status: Backend completed successfully")
            if (not self.advanced_dialog.crop_only_checkbox.isChecked()) and (self.run_started_at is not None):
                self.append_timing_log(elapsed_seconds=(time.time() - self.run_started_at), success=True, crop_only=False)
                self.flush_pending_milestones()
                
                # Log stage timing summary
                if self.stage_elapsed:
                    timing_items = []
                    for stage_name in ["crop", "enhance", "rephoto"]:
                        if stage_name in self.stage_elapsed:
                            elapsed_sec = self.stage_elapsed[stage_name]
                            timing_items.append(f"{stage_name}: {elapsed_sec:.1f}s")
                    if timing_items:
                        self.log_box.append(f"Stage timing: {', '.join(timing_items)}")

            if self.advanced_dialog.crop_only_checkbox.isChecked():
                # Crop-only mode: find and copy crop output
                crop_image = self.find_latest_crop_output(after_epoch=self.run_started_at)
                if crop_image is not None:
                    # Copy crop to results root
                    copied_path = self.copy_crop_outputs_to_results_root(crop_image)
                    if copied_path is not None:
                        self.set_result_preview_image(copied_path)
                        final_output_path = copied_path
                        self.log_box.append(f"Crop-only output copied to: {copied_path.parent}")
                        self.log_box.append(f"Previewing crop-only result: {copied_path.name}")
                        self.status_label.setText("Status: Crop-only output ready")
                    else:
                        self.log_box.append("Crop-only run: crop was produced but copy to results failed.")
                        self.set_result_preview_image(crop_image)
                        final_output_path = crop_image
                else:
                    self.log_box.append("Crop-only run: no crop image was found.")
                    self.set_result_preview_image(None)
            else:
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
                newest = self.find_latest_image(results_root, self.run_started_at)

                if newest is None:
                    self.log_box.append("No new result image was found in the results folder.")
                    self.set_result_preview_image(None)
                else:
                    run_folder = newest.parent
                    _, rephoto_path = self.simplify_run_folder(run_folder)
                    final_preview = rephoto_path or newest
                    self.set_result_preview_image(final_preview)
                    final_output_path = final_preview
                    self.log_box.append(f"Previewing final result: {final_preview.name}")

            # Clear overlay after displaying final preview
            self.clear_result_stage_overlay()

        else:
            self.status_label.setText("Status: Backend returned an error")
            self.clear_result_stage_overlay()

        self.show_run_summary_dialog(
            success=(exit_code == 0),
            exit_code=exit_code,
            elapsed_seconds=elapsed_seconds,
            output_path=final_output_path,
        )
        self.current_run_summary_context = None

        self.set_controls_for_running(False)
        self.process = None
        self.run_started_at = None


    def process_error(self, process_error):
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.log_box.append(f"Process launch error: {process_error}")
        self.status_label.setText("Status: Process launch error")
        self.clear_result_stage_overlay()
        elapsed_seconds = (time.time() - self.run_started_at) if self.run_started_at is not None else None
        self.show_run_summary_dialog(
            success=False,
            exit_code=-1,
            elapsed_seconds=elapsed_seconds,
            output_path=None,
            launch_error=str(process_error),
        )
        self.current_run_summary_context = None
        self.set_controls_for_running(False)
        self.process = None
        self.run_started_at = None


    def cancel_run(self):
        if self.process is None:
            self.log_box.append("No backend process is running.")
            self.status_label.setText("Status: No backend process to cancel")
            return

        # preserve current bar values but note cancel
        self.current_run_phase = "cancelled"
        self.preprocess_progress_bar.setFormat(self.preprocess_progress_bar.format() + " (cancelled)")
        self.rephoto_progress_bar.setFormat(self.rephoto_progress_bar.format() + " (cancelled)")
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self.log_box.append("Cancel requested. Stopping backend process...")
        self.status_label.setText("Status: Cancelling...")
        self.clear_result_stage_overlay()

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

        self.clear_result_stage_overlay()
        self.reset_progress_bars()
        self.current_run_summary_context = self._capture_run_context()
        # start in preprocessing phase with minimal indicator
        self.current_run_phase = "preprocess"
        
        # Reset path tracking and stage timing
        self.current_crop_output_dir = None
        self.current_gfpgan_output_dir = None
        self.current_blended_faces_dir = None
        self.current_results_dir = None
        self.current_manifest_path = None
        self.stage_started_at = {}
        self.stage_elapsed = {}
        
        self.set_preprocess_progress(5, "Preprocessing...")
        self.set_rephoto_progress(0, "Waiting...")
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








