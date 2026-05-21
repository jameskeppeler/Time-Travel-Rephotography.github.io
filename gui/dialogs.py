"""Modal dialogs used by the rephotography GUI.

Extracted from gui/app.py as the second slice of the Sprint-4 module split.
Currently hosts only AdvancedSettingsDialog; preflight / run-summary dialogs
remain inlined on MainWindow because they reference per-instance state.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.constants import (
    DEFAULT_CAMERA_LR,
    DEFAULT_DET_THRESHOLD,
    DEFAULT_FACE_FACTOR,
    DEFAULT_GAUSSIAN,
    DEFAULT_GFPGAN_BLEND,
    DEFAULT_LR,
    DEFAULT_MIX_LAYER_END,
    DEFAULT_MIX_LAYER_START,
    DEFAULT_NOISE_REGULARIZE,
)
from gui.widgets import (
    InstantToolButton,
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSpinBox,
)


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

        # Quality presets row
        presets_layout = QHBoxLayout()
        presets_layout.setContentsMargins(0, 0, 0, 8)
        presets_layout.setSpacing(6)
        presets_layout.addWidget(QLabel("Presets:"))

        self.preset_quick_button = QPushButton("Quick")
        self.preset_quick_button.setMaximumWidth(80)
        self.preset_quick_button.clicked.connect(self._apply_preset_quick)
        presets_layout.addWidget(self.preset_quick_button)

        self.preset_balanced_button = QPushButton("Balanced")
        self.preset_balanced_button.setMaximumWidth(80)
        self.preset_balanced_button.clicked.connect(self._apply_preset_balanced)
        presets_layout.addWidget(self.preset_balanced_button)

        self.preset_high_quality_button = QPushButton("High Quality")
        self.preset_high_quality_button.setMaximumWidth(100)
        self.preset_high_quality_button.clicked.connect(self._apply_preset_high_quality)
        presets_layout.addWidget(self.preset_high_quality_button)

        presets_info_button = InstantToolButton()
        presets_info_button.setText("ⓘ")
        presets_info_button.setAutoRaise(True)
        presets_info_button.setFixedSize(16, 16)
        presets_info_button.setCursor(Qt.PointingHandCursor)
        presets_info_button.setToolTip(
            "<b>Quality Presets</b><br/>"
            "<b>Quick:</b> Fast processing, lower loss weights (identity/tonal/eye/structure set to Lower, VGG appearance off), higher learning rate (0.15), lower noise regularization (10000). "
            "Good for quick previews and iterative testing.<br/><br/>"
            "<b>Balanced:</b> Recommended defaults. Moderate loss weights (all set to Default), standard learning rate (0.1), standard noise regularization (50000). "
            "Best for most historical photos where quality and speed matter equally.<br/><br/>"
            "<b>High Quality:</b> Maximum fidelity. All loss weights set to Higher for maximum preservation of identity, tonality, eyes, and structure. "
            "Slower learning rate (0.08), higher noise regularization (100000) for more careful optimization. Best for final results where quality is critical."
        )
        presets_layout.addWidget(presets_info_button)
        presets_layout.addStretch(1)
        layout.addLayout(presets_layout)

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
        self.strategy_combo = NoScrollComboBox()
        self.strategy_combo.addItems(["all"])

        self.crop_only_checkbox = QCheckBox("Crop-only (debug)")

        self.use_gfpgan_checkbox = QCheckBox("Disable enhancement (GFPGAN)")
        self.use_gfpgan_checkbox.toggled.connect(self.update_enhancement_controls)

        self.auto_recompose_checkbox_adv = QCheckBox("Auto-recompose after rephoto")
        self.auto_recompose_checkbox_adv.setChecked(False)
        self.auto_recompose_checkbox_adv.setToolTip(
            "Automatically apply Color blend recomposition once each face finishes rephotography"
        )

        self.det_threshold_edit = NoScrollDoubleSpinBox()
        self.det_threshold_edit.setRange(0.0, 1.0)
        self.det_threshold_edit.setSingleStep(0.01)
        self.det_threshold_edit.setDecimals(2)
        self.det_threshold_edit.setValue(0.90)

        self.face_factor_edit = NoScrollDoubleSpinBox()
        self.face_factor_edit.setRange(0.10, 2.00)
        self.face_factor_edit.setSingleStep(0.01)
        self.face_factor_edit.setDecimals(2)
        self.face_factor_edit.setValue(DEFAULT_FACE_FACTOR)

        self.gfpgan_blend_edit = NoScrollDoubleSpinBox()
        self.gfpgan_blend_edit.setRange(0.0, 1.0)
        self.gfpgan_blend_edit.setSingleStep(0.01)
        self.gfpgan_blend_edit.setDecimals(2)
        self.gfpgan_blend_edit.setValue(0.45)

        self.gaussian_edit = NoScrollDoubleSpinBox()
        self.gaussian_edit.setRange(0.0, 5.0)
        self.gaussian_edit.setSingleStep(0.05)
        self.gaussian_edit.setDecimals(2)
        self.gaussian_edit.setValue(0.75)

        self.identity_preservation_combo = NoScrollComboBox()
        self.identity_preservation_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.identity_preservation_combo.setCurrentText("Default")

        self.tonal_transfer_combo = NoScrollComboBox()
        self.tonal_transfer_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.tonal_transfer_combo.setCurrentText("Default")

        self.eye_preservation_combo = NoScrollComboBox()
        self.eye_preservation_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.eye_preservation_combo.setCurrentText("Default")

        self.structure_matching_combo = NoScrollComboBox()
        self.structure_matching_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.structure_matching_combo.setCurrentText("Default")

        self.vgg_appearance_combo = NoScrollComboBox()
        self.vgg_appearance_combo.addItems(["Off", "Lower", "Default", "Higher"])
        self.vgg_appearance_combo.setCurrentText("Default")

        self.noise_regularize_edit = NoScrollDoubleSpinBox()
        self.noise_regularize_edit.setRange(0.0, 1000000.0)
        self.noise_regularize_edit.setSingleStep(1000.0)
        self.noise_regularize_edit.setDecimals(1)
        self.noise_regularize_edit.setValue(50000.0)

        self.lr_edit = NoScrollDoubleSpinBox()
        self.lr_edit.setRange(0.0001, 10.0)
        self.lr_edit.setSingleStep(0.01)
        self.lr_edit.setDecimals(4)
        self.lr_edit.setValue(0.1)

        self.camera_lr_edit = NoScrollDoubleSpinBox()
        self.camera_lr_edit.setRange(0.0001, 1.0)
        self.camera_lr_edit.setSingleStep(0.001)
        self.camera_lr_edit.setDecimals(4)
        self.camera_lr_edit.setValue(0.01)

        self.mix_layer_start_edit = NoScrollSpinBox()
        self.mix_layer_start_edit.setRange(0, 18)
        self.mix_layer_start_edit.setValue(10)

        self.mix_layer_end_edit = NoScrollSpinBox()
        self.mix_layer_end_edit.setRange(0, 18)
        self.mix_layer_end_edit.setValue(18)

        self.enable_advanced_iterations_checkbox = QCheckBox("Enable advanced iterations")
        self.enable_advanced_iterations_checkbox.setChecked(False)

        # populate tabs with labels containing info icons
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
                "Auto-recompose",
                "Automatically apply Color blend recomposition once each face finishes rephotography, blending the rephoto result back into the original image with preserved luminance."
            ),
            self.auto_recompose_checkbox_adv,
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
                "Advanced iterations",
                "When enabled, the main window shows a fine-grained iteration slider (375-20000) instead of the basic 4-preset quality selector."
            ),
            self.enable_advanced_iterations_checkbox,
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
        self.strategy_combo.setCurrentText("all")
        self.crop_only_checkbox.setChecked(False)
        self.use_gfpgan_checkbox.setChecked(False)
        self.gfpgan_blend_edit.setValue(DEFAULT_GFPGAN_BLEND)
        self.det_threshold_edit.setValue(DEFAULT_DET_THRESHOLD)
        self.face_factor_edit.setValue(DEFAULT_FACE_FACTOR)
        self.gaussian_edit.setValue(DEFAULT_GAUSSIAN)
        self.enable_advanced_iterations_checkbox.setChecked(False)
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

    def _apply_preset_quick(self):
        """Quick preset: fast processing, lower quality loss weights."""
        self.identity_preservation_combo.setCurrentText("Lower")
        self.tonal_transfer_combo.setCurrentText("Lower")
        self.eye_preservation_combo.setCurrentText("Lower")
        self.structure_matching_combo.setCurrentText("Lower")
        self.vgg_appearance_combo.setCurrentText("Off")
        self.lr_edit.setValue(0.15)
        self.noise_regularize_edit.setValue(10000.0)

    def _apply_preset_balanced(self):
        """Balanced preset: recommended settings for most photos."""
        self.identity_preservation_combo.setCurrentText("Default")
        self.tonal_transfer_combo.setCurrentText("Default")
        self.eye_preservation_combo.setCurrentText("Default")
        self.structure_matching_combo.setCurrentText("Default")
        self.vgg_appearance_combo.setCurrentText("Default")
        self.lr_edit.setValue(0.1)
        self.noise_regularize_edit.setValue(50000.0)

    def _apply_preset_high_quality(self):
        """High Quality preset: maximum fidelity, longer processing."""
        self.identity_preservation_combo.setCurrentText("Higher")
        self.tonal_transfer_combo.setCurrentText("Higher")
        self.eye_preservation_combo.setCurrentText("Higher")
        self.structure_matching_combo.setCurrentText("Higher")
        self.vgg_appearance_combo.setCurrentText("Higher")
        self.lr_edit.setValue(0.08)
        self.noise_regularize_edit.setValue(100000.0)


__all__ = ["AdvancedSettingsDialog"]
