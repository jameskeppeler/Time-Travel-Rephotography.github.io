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
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import QEvent, QProcess, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, QRadialGradient, QTextCursor
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
    QToolTip,
    QVBoxLayout,
    QWidget,
)


# ============================================================================
# SHARED DEFAULTS & CONFIGURATION CONSTANTS
# ============================================================================

# Quality/iteration presets
DEFAULT_BASIC_ITER_VALUES = [375, 750, 1500, 3000]
DEFAULT_ADVANCED_ITER_VALUES = [375, 750] + list(range(1000, 20001, 1000))
DEFAULT_ITERATION = 750

# Human-readable labels for the basic quality presets.
QUALITY_PRESET_LABELS = {
    375: "Quick Preview",
    750: "Standard",
    1500: "High",
    3000: "Very High",
}

# Typical date ranges shown as placeholder hints when a photo type is selected.
PHOTO_TYPE_DATE_HINTS = {
    "Unknown": "e.g. 1865, 1890s, circa 1910",
    "Daguerreotype": "e.g. 1840\u20131860",
    "Ambrotype": "e.g. 1854\u20131870",
    "Tintype / Ferrotype": "e.g. 1856\u20131900",
    "Carte de visite (CDV)": "e.g. 1859\u20131890",
    "Cabinet card": "e.g. 1866\u20131900",
    "Late cabinet card / dry plate studio portrait": "e.g. 1885\u20131910",
    "Early gelatin silver print": "e.g. 1890\u20131920",
    "Black-and-white snapshot / roll-film print": "e.g. 1900\u20131950",
}
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
SIMPLE_FINAL_COPY_RE = re.compile(r"^Simple final copy:\s*(.+?)\s*$")
REPHOTO_CROP_FAIL_RE = re.compile(r"projector\.py failed for crop:\s*(.+)$", re.IGNORECASE)
QUICK_FACE_DECISION_RE = re.compile(r"QUICK_FACE_DECISION_COUNT\s*=\s*(\d+)")
RETINA_FACE_BOX_RE = re.compile(r"RETINA_FACE_BOX_(\d+)\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
CROP_ALIGN_BOX_RE = re.compile(r"CROP_ALIGN_BOX_(\d+)\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
FACE_SUFFIX_INDEX_RE = re.compile(r"_(\d+)$")

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


class NoScrollComboBox(QComboBox):
    """QComboBox variant that ignores wheel motion to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox variant that ignores wheel motion to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    """QSpinBox variant that ignores wheel motion to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoScrollSlider(QSlider):
    """QSlider variant that ignores wheel motion to prevent accidental changes."""

    def wheelEvent(self, event):
        event.ignore()


class FaceStripToolButton(QToolButton):
    """Filmstrip card button that exposes lightweight hover callbacks."""

    # Stylesheet cache to avoid regenerating for every button
    _stylesheet_cache = {}
    # Track currently hovered button globally to avoid expensive widget tree queries
    _currently_hovered = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.hover_enter_callback = None
        self.hover_leave_callback = None

    @staticmethod
    def get_stylesheet(border_color, bg_color, text_color, hover_color):
        """Get stylesheet from cache or generate once."""
        key = (border_color, bg_color, text_color, hover_color)
        if key not in FaceStripToolButton._stylesheet_cache:
            FaceStripToolButton._stylesheet_cache[key] = (
                "QToolButton {"
                f" border: 1px solid {border_color}; border-radius: 5px;"
                f" background-color: {bg_color}; color: {text_color}; padding: 2px; }}"
                f"QToolButton:hover {{ background-color: {hover_color}; }}"
                "QToolButton:checked { background-color: #1f2630; }"
            )
        return FaceStripToolButton._stylesheet_cache[key]

    def enterEvent(self, event):
        super().enterEvent(event)
        FaceStripToolButton._currently_hovered = self
        if callable(self.hover_enter_callback):
            self.hover_enter_callback()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if callable(self.hover_enter_callback):
            self.hover_enter_callback()

    def leaveEvent(self, event):
        FaceStripToolButton._currently_hovered = None
        if callable(self.hover_leave_callback):
            self.hover_leave_callback()
        super().leaveEvent(event)


class InputDropLabel(QLabel):
    def __init__(self, parent_window):
        super().__init__("Drop or click to choose an input image.\nSupported: PNG, JPG, TIFF, WEBP")
        self.parent_window = parent_window
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
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
        if hasattr(self.parent_window, "handle_input_preview_mouse_leave"):
            self.parent_window.handle_input_preview_mouse_leave()
        self._set_normal_style()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self.parent_window, "handle_input_preview_mouse_move"):
            self.parent_window.handle_input_preview_mouse_move(event.pos())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if hasattr(self.parent_window, "handle_input_preview_click"):
                if self.parent_window.handle_input_preview_click(event.pos()):
                    return
            if not self.parent_window.can_select_new_image(show_message=True):
                return
            self.parent_window.browse_for_image()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if not self.parent_window.can_select_new_image(show_message=False):
            event.ignore()
            return
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
        if not self.parent_window.can_select_new_image(show_message=True):
            self._set_normal_style()
            event.ignore()
            return
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pw = self.parent_window
        if hasattr(pw, "refresh_input_preview_scale"):
            pw.refresh_input_preview_scale()
        if hasattr(pw, "position_input_detect_overlay"):
            pw.position_input_detect_overlay()


class ResultPreviewLabel(QLabel):
    """QLabel subclass that rescales the result preview on every resize."""
    def __init__(self, parent_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_window = parent_window

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pw = self.parent_window
        if hasattr(pw, "refresh_result_preview_scale"):
            pw.refresh_result_preview_scale()
        if hasattr(pw, "position_result_stage_overlay"):
            pw.position_result_stage_overlay()


class InputDetectOverlay(QWidget):
    """Animated particle-grain overlay for import-time face detection.

    The overlay renders a blurred backdrop plus luminous sampled particles.
    Particle coordinates are normalized so the effect remains stable while the
    window is resized.
    """

    _POOL_SIZE = 5200
    _SPAWN_PER_TICK = 72
    _PARTICLE_MIN_LIFE = 12
    _PARTICLE_MAX_LIFE = 44
    _PARTICLE_MIN_R = 0.35
    _PARTICLE_MAX_R = 0.95
    _TICK_INTERVAL_MS = 55
    _REBUILD_DEBOUNCE_MS = 90

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.base_text = "Detecting Faces"
        self.dot_count = 0
        self._pulse_phase = 0.0
        self._tick_count = 0
        self._target_rect = QRect()
        self._blurred_backdrop = QPixmap()
        self._source_image = QImage()
        # Particle: [nx, ny, r, age, max_age, cr, cg, cb]
        self._particles = []
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(self._TICK_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._on_tick)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(self._REBUILD_DEBOUNCE_MS)
        self._rebuild_timer.timeout.connect(self._rebuild_source_data)

    def _compute_overlay_target_rect(self):
        parent = self.parentWidget()
        if parent is None:
            return QRect()
        try:
            parent_pix = parent.pixmap()
        except Exception:
            parent_pix = None
        if parent_pix is None or parent_pix.isNull():
            return QRect()
        pw = int(parent_pix.width())
        ph = int(parent_pix.height())
        if pw <= 1 or ph <= 1:
            return QRect()
        x = max(0, int(round((self.width() - pw) * 0.5)))
        y = max(0, int(round((self.height() - ph) * 0.5)))
        w = min(pw, max(1, self.width() - x))
        h = min(ph, max(1, self.height() - y))
        return QRect(x, y, w, h)

    def _schedule_rebuild_source_data(self, immediate=False):
        if immediate:
            self._rebuild_timer.stop()
            self._rebuild_source_data()
            return
        if not self._rebuild_timer.isActive():
            self._rebuild_timer.start()

    def _rebuild_source_data(self):
        parent = self.parentWidget()
        if parent is None:
            self._blurred_backdrop = QPixmap()
            self._source_image = QImage()
            self._target_rect = QRect()
            return

        target_rect = self._compute_overlay_target_rect()
        self._target_rect = target_rect
        if target_rect.width() <= 2 or target_rect.height() <= 2:
            self._blurred_backdrop = QPixmap()
            self._source_image = QImage()
            return

        try:
            parent_pix = parent.pixmap()
        except Exception:
            parent_pix = None
        if parent_pix is None or parent_pix.isNull():
            self._blurred_backdrop = QPixmap()
            self._source_image = QImage()
            return

        cw = int(parent_pix.width())
        ch = int(parent_pix.height())
        if cw <= 2 or ch <= 2:
            self._blurred_backdrop = QPixmap()
            self._source_image = QImage()
            return

        # Heavy blur approximation: strong downsample then upscale.
        tiny_w = max(1, int(round(cw / 20.0)))
        tiny_h = max(1, int(round(ch / 20.0)))
        tiny = parent_pix.scaled(tiny_w, tiny_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        mid_w = max(1, int(round(cw / 6.0)))
        mid_h = max(1, int(round(ch / 6.0)))
        mid = tiny.scaled(mid_w, mid_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self._blurred_backdrop = mid.scaled(cw, ch, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        # Small sample source used for particle color lookup.
        sample_max = 128
        if max(cw, ch) > sample_max:
            s = sample_max / max(cw, ch)
            sw = max(1, int(round(cw * s)))
            sh = max(1, int(round(ch * s)))
        else:
            sw, sh = cw, ch
        self._source_image = parent_pix.scaled(sw, sh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).toImage()
        self.update()

    def _sample_colour(self, nx, ny):
        img = self._source_image
        if img.isNull():
            return (220, 225, 235)
        iw, ih = int(img.width()), int(img.height())
        px = max(0, min(iw - 1, int(round(nx * (iw - 1)))))
        py = max(0, min(ih - 1, int(round(ny * (ih - 1)))))
        c = QColor(img.pixel(px, py))
        # Lift toward white for a luminous grain look.
        r = min(255, int(c.red() * 0.45 + 255 * 0.55))
        g = min(255, int(c.green() * 0.45 + 255 * 0.55))
        b = min(255, int(c.blue() * 0.45 + 255 * 0.55))
        return (r, g, b)

    def _spawn_particles(self, count):
        tr = self._target_rect
        if tr.width() <= 2 or tr.height() <= 2:
            return
        particles = self._particles
        for _ in range(count):
            if len(particles) >= self._POOL_SIZE:
                break
            nx = random.random()
            ny = random.random()
            particles.append([
                nx,
                ny,
                random.uniform(self._PARTICLE_MIN_R, self._PARTICLE_MAX_R),
                0,
                random.randint(self._PARTICLE_MIN_LIFE, self._PARTICLE_MAX_LIFE),
                *self._sample_colour(nx, ny),
            ])

    def _age_particles(self):
        alive = []
        for particle in self._particles:
            particle[3] += 1
            if particle[3] <= particle[4]:
                alive.append(particle)
        self._particles = alive

    def start(self, base_text="Detecting Faces"):
        self.base_text = str(base_text or "Detecting Faces")
        self.dot_count = 0
        self._pulse_phase = 0.0
        self._tick_count = 0
        self._particles = []
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect().adjusted(1, 1, -1, -1))
        self._schedule_rebuild_source_data(immediate=True)
        for _ in range(6):
            self._spawn_particles(self._SPAWN_PER_TICK)
        for particle in self._particles:
            max_age = max(1, int(particle[4]))
            particle[3] = random.randint(0, max_age - 1)
        self.setVisible(True)
        self.raise_()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def stop(self):
        self._rebuild_timer.stop()
        self._anim_timer.stop()
        self.setVisible(False)
        self._blurred_backdrop = QPixmap()
        self._source_image = QImage()
        self._target_rect = QRect()
        self._particles = []

    def sync_geometry(self):
        parent = self.parentWidget()
        if parent is None:
            return
        r = parent.rect().adjusted(1, 1, -1, -1)
        geometry_changed = False
        if self.geometry() != r:
            self.setGeometry(r)
            geometry_changed = True
        target_rect = self._compute_overlay_target_rect()
        target_changed = target_rect != self._target_rect
        self._target_rect = target_rect
        if self.isVisible() and (geometry_changed or target_changed):
            # Debounce expensive backdrop rebuild while user drags window edges.
            self._schedule_rebuild_source_data(immediate=False)
        if self.isVisible():
            self.raise_()

    def notify_parent_pixmap_changed(self):
        if not self.isVisible():
            return
        self._schedule_rebuild_source_data(immediate=True)

    def _on_tick(self):
        self._pulse_phase += 0.06
        if self._pulse_phase > 6.283185307179586:
            self._pulse_phase -= 6.283185307179586
        self._tick_count += 1
        if self._tick_count % 6 == 0:
            self.dot_count = (self.dot_count + 1) % 4
        self._age_particles()
        self._spawn_particles(self._SPAWN_PER_TICK)
        self.update()

    def paintEvent(self, event):
        if not self.isVisible():
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            # Always use latest target rect to keep animation anchored while resizing.
            r = self._compute_overlay_target_rect()
            if r != self._target_rect:
                self._target_rect = r
            has_target = r.width() > 2 and r.height() > 2
            pulse = 0.5 + 0.5 * math.sin(self._pulse_phase)

            if has_target and self._blurred_backdrop is not None and (not self._blurred_backdrop.isNull()):
                painter.setOpacity(0.92)
                painter.drawPixmap(r, self._blurred_backdrop)
                painter.setOpacity(1.0)
                painter.fillRect(r, QColor(4, 6, 12, 50))

                rw = float(r.width())
                rh = float(r.height())
                rx = float(r.x())
                ry = float(r.y())
                painter.setPen(Qt.NoPen)
                for nx, ny, pr, age, max_age, cr, cg, cb in self._particles:
                    t = age / max(1, max_age)
                    if t < 0.2:
                        alpha = t / 0.2
                    elif t < 0.6:
                        alpha = 1.0
                    else:
                        alpha = 1.0 - (t - 0.6) / 0.4
                    alpha = max(0.0, min(1.0, alpha))
                    a = int(round(alpha * 195))
                    if a <= 4:
                        continue
                    px = rx + (nx * rw)
                    py = ry + (ny * rh)
                    painter.setBrush(QColor(int(cr), int(cg), int(cb), a))
                    d = max(1, int(round(pr * 2)))
                    painter.drawEllipse(int(round(px - pr)), int(round(py - pr)), d, d)

                cx = r.x() + (r.width() * 0.5)
                cy = r.y() + (r.height() * 0.5)
                pulse_scale = 0.92 + (pulse * 0.15)
                radius = max(72, int(round(min(r.width(), r.height()) * 0.44 * pulse_scale)))
                glow_alpha = int(round(24 + pulse * 28))
                glow = QRadialGradient(cx, cy, radius)
                glow.setColorAt(0.0, QColor(96, 178, 255, glow_alpha + 16))
                glow.setColorAt(0.34, QColor(80, 156, 230, int(glow_alpha * 0.42)))
                glow.setColorAt(0.70, QColor(62, 128, 201, int(glow_alpha * 0.12)))
                glow.setColorAt(1.0, QColor(24, 43, 69, 0))
                painter.fillRect(r, QBrush(glow))

            text = f"{self.base_text}{'.' * self.dot_count}"
            painter.setFont(QFont("Segoe UI", 10, QFont.Medium))
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(text) + 20
            th = metrics.height() + 8
            bounds = r if has_target else self.rect()
            tx = bounds.x() + max(8, (bounds.width() - tw) // 2)
            ty = bounds.y() + max(8, (bounds.height() - th) // 2)
            bubble = QRect(tx, ty, tw, th)

            painter.setPen(QPen(QColor(90, 160, 228, 125), 1))
            painter.setBrush(QColor(14, 28, 46, 174))
            painter.drawRoundedRect(bubble, 8, 8)
            painter.setPen(QColor("#edf6ff"))
            painter.drawText(bubble, Qt.AlignCenter, text)
        finally:
            painter.end()

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
        self.strategy_combo = NoScrollComboBox()
        self.strategy_combo.addItems(["all"])

        self.crop_only_checkbox = QCheckBox("Crop-only (debug)")

        self.use_gfpgan_checkbox = QCheckBox("Disable enhancement (GFPGAN)")
        self.use_gfpgan_checkbox.toggled.connect(self.update_enhancement_controls)

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
        self.menu_show_last_summary_action.triggered.connect(self.show_last_run_summary_dialog)
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
        if hasattr(self, "menu_show_last_summary_action"):
            self.menu_show_last_summary_action.setEnabled(bool(self.last_run_summary_text))
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
        self.stop_quick_face_probe()
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
            "selected_faces": self.get_selected_face_count_text(),
        }

    def _build_run_summary_text(self, success, exit_code, elapsed_seconds, output_path=None, launch_error=None):
        ctx = self.current_run_summary_context or {}
        status = "Success" if success else "Failed"
        if launch_error:
            status = "Launch Error"
        preset_sent = ctx.get("effective_preset")
        if preset_sent is None:
            quality_line = f"- Quality: {ctx.get('quality', 'N/A')}"
        else:
            normalized_preset = "750" if str(preset_sent).strip().lower() == "test" else str(preset_sent)
            quality_line = f"- Quality: {ctx.get('quality', 'N/A')} (preset sent: {normalized_preset})"

        lines = [
            "Run Summary",
            "-----------",
            f"Status: {status}",
            f"Exit code: {exit_code}",
            f"Elapsed: {self._format_elapsed_for_summary(elapsed_seconds)}",
            "",
            "Key settings:",
            quality_line,
            f"- Photo type: {ctx.get('photo_type', 'N/A')}",
            f"- Approximate date: {ctx.get('approx_date', '') or 'N/A'}",
            f"- Spectral sensitivity: {ctx.get('spectral_sensitivity', 'N/A')}",
            f"- Faces rephotographed: {ctx.get('selected_faces', 'N/A')}",
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

    def _store_run_summary_text(self, success, exit_code, elapsed_seconds, output_path=None, launch_error=None):
        if self.current_run_summary_context is None:
            return None

        summary_text = self._build_run_summary_text(
            success=success,
            exit_code=exit_code,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            launch_error=launch_error,
        )
        self.last_run_summary_text = summary_text
        self.update_view_menu_actions()
        return summary_text

    def _show_run_summary_text_dialog(self, summary_text):
        if not summary_text:
            return

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

    def show_run_summary_dialog(self, success, exit_code, elapsed_seconds, output_path=None, launch_error=None):
        summary_text = self._store_run_summary_text(
            success=success,
            exit_code=exit_code,
            elapsed_seconds=elapsed_seconds,
            output_path=output_path,
            launch_error=launch_error,
        )
        if not summary_text:
            return
        self._show_run_summary_text_dialog(summary_text)

    def show_last_run_summary_dialog(self):
        summary_text = (self.last_run_summary_text or "").strip()
        if not summary_text:
            self.status_label.setText("Status: No run summary available yet")
            self.log_box.append("No run summary is available yet.")
            return
        self._show_run_summary_text_dialog(summary_text)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self._BASE_WINDOW_TITLE)

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
        self.last_run_summary_text = ""
        self.last_preflight_report = None
        self._hardware_info_cache = None
        self._hardware_info_cache_ts = 0.0
        self._hardware_info_cache_ttl_sec = 30.0
        self._timing_records_cache_path = None
        self._timing_records_cache_mtime = None
        self._timing_records_cache = []
        self._haar_face_detector = None
        self._conda_executable_cache = None
        self._facecrop_env_cache_key = None
        self._facecrop_env_name_cache = None

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
        self._pending_face_reselection = None  # set of face indices to pre-select after a re-crop
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
        self.result_preview_pixmap_cache = {}
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
        self.face_preview_thumb_icon_cache = {}
        self.face_preview_thumb_icon_cache_max_entries = 256
        self._face_strip_render_signature = None

        # Multi-face preview state (for strategy=all / multi-face detections)
        self.face_preview_entries = []
        self.active_face_preview_index = None
        self.selected_face_preview_index = None
        self.hover_face_preview_index = None
        self.hover_face_preview_source = None
        self.hover_face_box_override = None
        self.hover_face_box_cache = {}
        self.quick_face_count_estimate = None
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

        self.advanced_dialog.crop_only_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_mode_controls)
        self.advanced_dialog.use_gfpgan_checkbox.toggled.connect(self.update_runtime_label)

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
        self.run_button.clicked.connect(self.handle_run_button_clicked)
        self.run_button.setShortcut("Ctrl+Return")
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
        self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumHeight(34)
        self.cancel_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cancel_button.setStyleSheet(secondary_button_style)

        self.end_early_button = QPushButton("End Early")
        self.end_early_button.clicked.connect(self.request_end_run_early)
        self.end_early_button.setMinimumHeight(34)
        self.end_early_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.end_early_button.setStyleSheet(secondary_button_style)
        self.end_early_button.setEnabled(False)
        self.end_early_button.setToolTip("Finish this run early using completed iterations only.")

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
        face_header_layout.addWidget(self.face_preview_summary_label, 0, Qt.AlignVCenter)
        face_header_layout.addStretch(1)

        self.face_select_all_button = QPushButton("All")
        self.face_select_all_button.setMinimumHeight(20)
        self.face_select_all_button.setMaximumHeight(20)
        self.face_select_all_button.setMinimumWidth(44)
        self.face_select_all_button.clicked.connect(lambda: self.set_all_faces_selected(True))
        face_header_layout.addWidget(self.face_select_all_button, 0, Qt.AlignVCenter)

        self.face_select_none_button = QPushButton("None")
        self.face_select_none_button.setMinimumHeight(20)
        self.face_select_none_button.setMaximumHeight(20)
        self.face_select_none_button.setMinimumWidth(50)
        self.face_select_none_button.clicked.connect(lambda: self.set_all_faces_selected(False))
        face_header_layout.addWidget(self.face_select_none_button, 0, Qt.AlignVCenter)

        self.face_preview_auto_follow_checkbox = QCheckBox("Auto-follow latest")
        self.face_preview_auto_follow_checkbox.setChecked(True)
        self.face_preview_auto_follow_checkbox.setStyleSheet("color: #c7ccd4;")
        self.face_preview_auto_follow_checkbox.toggled.connect(self.handle_face_auto_follow_toggled)
        face_header_layout.addWidget(self.face_preview_auto_follow_checkbox, 0, Qt.AlignVCenter)
        self.face_preview_header.setFixedHeight(22)

        self.face_preview_strip_scroll = QScrollArea()
        self.face_preview_strip_scroll.setWidgetResizable(True)
        self.face_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.face_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.face_preview_strip_scroll.setMinimumHeight(142)
        self.face_preview_strip_scroll.setMaximumHeight(142)
        self.face_preview_strip_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #565c66; border-radius: 4px; background: #1f2329; }"
        )

        self.face_preview_strip_container = QWidget()
        self.face_preview_strip_layout = QHBoxLayout()
        self.face_preview_strip_layout.setContentsMargins(6, 6, 6, 6)
        self.face_preview_strip_layout.setSpacing(6)
        self.face_preview_strip_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.face_preview_strip_container.setLayout(self.face_preview_strip_layout)
        self.face_preview_strip_container.setMouseTracking(True)
        self.face_preview_strip_scroll.setWidget(self.face_preview_strip_container)
        self.face_preview_strip_scroll.viewport().setMouseTracking(True)
        self.face_preview_strip_scroll.viewport().installEventFilter(self)

        self.face_preview_header.setVisible(False)
        self.face_preview_strip_scroll.setVisible(False)

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
        face_panel_layout.addWidget(self.face_preview_strip_scroll)
        self.face_preview_panel.setVisible(False)
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

        self.log_box = QTextEdit()
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
        QTimer.singleShot(0, lambda: self.run_startup_preflight(show_dialog=True))
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

    def _update_wide_preview_dimensions(self):
        """Track the current preview label width for face strip card sizing."""
        if not getattr(self, "_wide_layout_active", False):
            return
        self.current_wide_preview_side = max(200, self.input_preview_label.width())

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

    def _rehost_face_preview_panel(self, use_wide_layout):
        if not hasattr(self, "face_preview_panel"):
            return
        if hasattr(self, "previews_layout"):
            self.previews_layout.removeWidget(self.face_preview_panel)
        if hasattr(self, "main_layout"):
            self.main_layout.removeWidget(self.face_preview_panel)
        if hasattr(self, "content_layout"):
            self.content_layout.removeWidget(self.face_preview_panel)

        if use_wide_layout:
            # In wide mode, placement is handled directly inside content layout.
            return
        else:
            # In stacked mode, keep filmstrip attached under preview panes.
            self.previews_layout.insertWidget(1, self.face_preview_panel)

    def _configure_face_preview_panel_for_mode(self, use_wide_layout):
        if not hasattr(self, "face_preview_panel"):
            return

        if use_wide_layout:
            card_w = self._get_face_strip_card_width(True)
            panel_w = card_w + 34
            self.face_preview_strip_layout.setDirection(QBoxLayout.TopToBottom)
            self.face_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.face_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.face_preview_strip_scroll.setMinimumHeight(0)
            self.face_preview_strip_scroll.setMaximumHeight(16777215)
            self.face_preview_strip_scroll.setMinimumWidth(panel_w - 8)
            self.face_preview_strip_scroll.setMaximumWidth(panel_w)
            self.face_preview_panel.setMinimumWidth(panel_w)
            self.face_preview_panel.setMaximumWidth(panel_w)
            self.face_preview_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.face_preview_auto_follow_checkbox.setText("Auto-follow")
        else:
            self.face_preview_strip_layout.setDirection(QBoxLayout.LeftToRight)
            self.face_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.face_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.face_preview_strip_scroll.setMinimumHeight(142)
            self.face_preview_strip_scroll.setMaximumHeight(142)
            self.face_preview_strip_scroll.setMinimumWidth(0)
            self.face_preview_strip_scroll.setMaximumWidth(16777215)
            self.face_preview_panel.setMinimumWidth(0)
            self.face_preview_panel.setMaximumWidth(16777215)
            self.face_preview_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.face_preview_auto_follow_checkbox.setText("Auto-follow latest")

    def _get_face_strip_card_width(self, wide_mode):
        if not wide_mode:
            return 116
        side = max(240, int(getattr(self, "current_wide_preview_side", 360)))
        # Keep vertical rail compact while scaling mildly with preview size.
        return max(98, min(132, int(round(side * 0.30))))

    def apply_responsive_layout(self, force=False):
        if not hasattr(self, "content_layout"):
            return

        use_wide_layout = self.isFullScreen() or self.width() >= WIDE_LAYOUT_MIN_WIDTH
        if not force and self._wide_layout_active == use_wide_layout:
            return
        self._wide_layout_active = use_wide_layout

        while self.content_layout.count() > 0:
            self.content_layout.takeAt(0)

        self._rehost_face_preview_panel(use_wide_layout)

        if use_wide_layout:
            self._configure_face_preview_panel_for_mode(True)

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
            self._configure_face_preview_panel_for_mode(False)
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
        if self.face_preview_entries:
            self.render_face_preview_strip()
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
                self.update_run_button_for_quick_face_hint()
            return

        # If crop-only mode: enhancement must be disabled
        if crop_only:
            self.advanced_dialog.use_gfpgan_checkbox.setChecked(True)  # Checked = disabled
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
            self.advanced_dialog.gfpgan_blend_edit.setEnabled(False)
            if hasattr(self, "run_button"):
                self.update_run_button_for_quick_face_hint()
            return

        # Otherwise: enhancement can be toggled
        self.advanced_dialog.use_gfpgan_checkbox.setEnabled(True)
        # Update blend spinner based on checkbox state
        enhancement_enabled = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked())
        self.advanced_dialog.gfpgan_blend_edit.setEnabled(enhancement_enabled)
        if hasattr(self, "run_button"):
            self.update_run_button_for_quick_face_hint()

    def update_iteration_mode(self):
        current = self.iter_values[self.iter_slider.value()]
        self.iter_slider.setMaximum(len(self.iter_values) - 1)

        closest_index = min(range(len(self.iter_values)), key=lambda i: abs(self.iter_values[i] - current))
        self.iter_slider.setValue(closest_index)

    def reset_progress_bars(self):
        self.preprocess_progress_bar.setValue(0)
        self.preprocess_progress_bar.setFormat("Preprocess ready")
        self.rephoto_progress_bar.setValue(0)
        self.rephoto_status_text = "Waiting..."
        self.update_rephoto_bar_format()
        self._last_preprocess_progress_state = (0, "Preprocess ready")
        self._last_rephoto_progress_state = (0, self.rephoto_status_text)
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
        self._last_iter_progress_signature = None
        self.current_run_phase = "idle"
        self.rephoto_started_at = None

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

    def set_preprocess_progress(self, value, text=None):
        value = max(0, min(100, value))
        current_text = self.preprocess_progress_bar.format()
        next_text = current_text if text is None else str(text)
        next_state = (value, next_text)
        if self._last_preprocess_progress_state == next_state:
            return
        self.preprocess_progress_bar.setValue(value)
        if text:
            self.preprocess_progress_bar.setFormat(text)
        self._last_preprocess_progress_state = next_state
        # Keep title bar in sync during preprocessing
        if getattr(self, "current_run_phase", "idle") == "preprocess":
            self.setWindowTitle(f"{next_text} \u2014 {self._BASE_WINDOW_TITLE}")

    _BASE_WINDOW_TITLE = "Time-Travel Rephotography"

    def set_rephoto_progress(self, value, text=None):
        value = max(0, min(100, value))
        next_status = self.rephoto_status_text if text is None else str(text)
        next_state = (value, next_status)
        if self._last_rephoto_progress_state == next_state:
            return
        self.rephoto_progress_bar.setValue(value)
        if text:
            self.rephoto_status_text = text
        self.update_rephoto_bar_format()
        self._last_rephoto_progress_state = next_state
        self._update_title_bar_progress(value, next_status)

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
        selected_count = len(self.get_selected_face_indices())
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

    def set_controls_for_running(self, is_running):
        self.update_run_button_for_quick_face_hint()
        if is_running:
            # Clear recomposite toggle on new run — stale paths no longer valid
            self._rephoto_result_path = None
            self._recomposited_result_path = None
            self.result_view_toggle.setVisible(False)
            self.result_view_toggle.setChecked(False)
            self.result_view_toggle.setText("Rephoto")
        if hasattr(self, "end_early_button"):
            self.end_early_button.setEnabled(bool(is_running))
        self.cancel_button.setEnabled(is_running)
        self.reset_button.setEnabled(not is_running)
        self.quit_button.setEnabled(not is_running)

        self.update_image_import_controls(force_running=is_running)

        self.advanced_dialog.strategy_combo.setEnabled(not is_running)
        self.advanced_dialog.crop_only_checkbox.setEnabled(not is_running)
        self.advanced_dialog.det_threshold_edit.setEnabled(not is_running)

        self.iter_slider.setEnabled(not is_running)
        self.photo_type_combo.setEnabled(not is_running)
        self.approx_date_edit.setEnabled(not is_running)
        self.spectral_mode_combo.setEnabled(not is_running)
        self.spectral_sensitivity_combo.setEnabled((not is_running) and (self.spectral_mode_combo.currentText() == "Manual"))

        self.results_root_edit.setEnabled(not is_running)
        self.results_browse_button.setEnabled(not is_running)
        self.advanced_settings_button.setEnabled(not is_running)
        if hasattr(self, "face_select_all_button"):
            self.face_select_all_button.setEnabled((not is_running) and self.awaiting_face_selection)
        if hasattr(self, "face_select_none_button"):
            self.face_select_none_button.setEnabled((not is_running) and self.awaiting_face_selection)
        if hasattr(self, "face_preview_auto_follow_checkbox"):
            self.face_preview_auto_follow_checkbox.setEnabled(not is_running)

        if is_running:
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
        else:
            self.update_mode_controls()
            self.update_spectral_sensitivity_ui()

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

    def update_progress_from_line(self, line: str):
        s = (line or "").strip()
        if not s:
            return

        # Hot path: during rephoto, most lines are tqdm iteration progress.
        # Check this first to avoid all the string-contains checks below.
        if self._try_fast_rephoto_iteration_progress(s):
            return

        # Quick reject: lines without "=" or ":" are unlikely to match any markers.
        # tqdm progress bars, loss dumps, etc. all lack the markers we care about.
        has_colon = ":" in s
        has_equals = "===" in s
        if not has_colon and not has_equals:
            return

        if s.startswith("=== Pause requested ==="):
            self.status_label.setText("Status: Paused")
            self._hide_result_stage_overlay_for_pause()
            return
        if s.startswith("=== Resume requested ==="):
            self._set_status_for_running_state()
            self._restore_result_stage_overlay_after_pause()
            return

        # === Path tracking from stdout ===
        if has_colon:
            if s.startswith("Crop:"):
                crop_path_text = s.split("Crop:", 1)[-1].strip()
                if crop_path_text:
                    self.mark_face_running_from_crop_path(crop_path_text)
            if "Crop output dir:" in s:
                self.current_crop_output_dir = s.split("Crop output dir:")[-1].strip()
                self._crop_source_input_key = self._normalized_path_key(self.input_image_edit.text().strip())
                self._crop_source_face_factor = self.advanced_dialog.face_factor_edit.value()
            elif "GFPGAN output:" in s:
                self.current_gfpgan_output_dir = s.split("GFPGAN output:")[-1].strip()
            elif "GFPGAN blended faces:" in s:
                self.current_blended_faces_dir = s.split("GFPGAN blended faces:")[-1].strip()
            elif "Results:" in s and "Manifest:" not in s:
                result_dir = s.split("Results:")[-1].strip()
                self.current_results_dir = result_dir
                if result_dir:
                    self.current_run_result_dirs.add(result_dir)
            elif "Manifest:" in s:
                self.current_manifest_path = s.split("Manifest:")[-1].strip()

            if "Simple final copy:" in s:
                simple_copy_match = SIMPLE_FINAL_COPY_RE.match(s)
                if simple_copy_match:
                    self.mark_face_done_from_result_path(simple_copy_match.group(1))

            if "projector.py failed for crop:" in s:
                fail_match = REPHOTO_CROP_FAIL_RE.search(s)
                if fail_match:
                    self.mark_face_failed_from_crop_name(fail_match.group(1))

        # === Stage timing instrumentation ===
        current_time = None

        if has_equals:
            # Record when stages start
            if "=== Face crop step" in s and "crop" not in self.stage_started_at:
                current_time = time.time()
                self.stage_started_at["crop"] = current_time
            elif "=== GFPGAN step" in s and "enhance" not in self.stage_started_at:
                current_time = time.time()
                self.stage_started_at["enhance"] = current_time
            elif "=== Rephoto step" in s and "rephoto" not in self.stage_started_at:
                current_time = time.time()
                self.stage_started_at["rephoto"] = current_time

        # Record when stages end and compute elapsed time
        if has_colon:
            if s.startswith("Cropped face count:") and "crop" in self.stage_started_at and "crop" not in self.stage_elapsed:
                if current_time is None:
                    current_time = time.time()
                self.stage_elapsed["crop"] = current_time - self.stage_started_at["crop"]
            elif "GFPGAN blended faces:" in s and "enhance" in self.stage_started_at and "enhance" not in self.stage_elapsed:
                if current_time is None:
                    current_time = time.time()
                self.stage_elapsed["enhance"] = current_time - self.stage_started_at["enhance"]

        # === Rephoto start marker, switch phases ===
        if s.startswith("=== Rephoto step"):
            if current_time is None:
                current_time = time.time()
            if self.current_run_phase == "preprocess":
                if self.suppress_preprocess_ui_until_rephoto:
                    self.set_preprocess_progress(0, "Preprocess ready")
                    self.suppress_preprocess_ui_until_rephoto = False
                else:
                    self.set_preprocess_progress(100, "Preprocessing complete")
                self.preprocess_stage = "complete"
                self.current_run_phase = "rephoto"
                self.rephoto_started_at = current_time
                self.start_rephoto_progress_tracking()
            if not self.face_preview_entries:
                self.initialize_face_preview_entries(expected_count=1)
            if self.current_run_phase == "rephoto":
                if self.rephoto_face_total <= 0:
                    self.rephoto_face_total = max(1, len(self.get_selected_face_indices()))
                self.rephoto_face_current_index = min(
                    self.rephoto_face_total,
                    self.rephoto_face_current_index + 1
                )
                self.set_rephoto_progress(0, "Processing")
                self.set_result_stage_overlay("Rephotographing")
            return

        # Crop‑only skip notice
        if s.startswith("CropOnly requested. Skipping rephoto step."):
            if self.suppress_preprocess_ui_until_rephoto:
                self.set_preprocess_progress(0, "Preprocess ready")
                self.set_rephoto_progress(0, "Waiting...")
            else:
                # finalize preprocessing and leave rephoto untouched
                self.set_preprocess_progress(100, "Preprocessing complete")
                self.set_rephoto_progress(0, "Skipped")
            self.current_run_phase = "crop_only_done"
            return

        # Preprocessing stage updates (only before rephoto begins)
        if self.current_run_phase == "preprocess":
            suppress_all_preprocess_ui = bool(self.suppress_preprocess_ui_until_rephoto)
            if s.startswith("=== Face crop step"):
                if not suppress_all_preprocess_ui:
                    self.set_preprocess_progress(20, "Cropping faces")
                    self.preprocess_stage = "cropping"
                    self.set_result_stage_overlay("Cropping")
                return

            if s.startswith("=== GFPGAN step"):
                if not suppress_all_preprocess_ui:
                    self.set_preprocess_progress(60, "Enhancing faces")
                    self.preprocess_stage = "enhancing"
                    self.set_result_stage_overlay("Enhancing")
                return

            if s.startswith("=== GPU pre-check") or s.startswith("=== Pre-flight check"):
                if not suppress_all_preprocess_ui:
                    self.set_preprocess_progress(80, "Pre-flight checks")
                    self.preprocess_stage = "gpu_check"
                return

            # Parse crop count and preview crop if found
            m = CROPPED_FACE_COUNT_RE.search(s)
            if m:
                n = int(m.group(1))
                if not suppress_all_preprocess_ui:
                    self.set_preprocess_progress(40, f"Crops ready ({n})")
                    self.preprocess_stage = "crops_ready"
                # In two-step multi-face mode, keep the strip hidden until preprocess fully finishes.
                # This avoids a brief layout collapse where preview panes shrink before selection UI appears.
                if (not self.selection_preprocess_mode) and (not self.face_preview_entries):
                    self.initialize_face_preview_entries(expected_count=n)
                if not suppress_all_preprocess_ui:
                    crop_image = self.find_latest_crop_output(after_epoch=self.run_started_at)
                    if crop_image:
                        self.preview_stage_image_if_found(crop_image, "Cropping")
                return

            # Watch for GFPGAN blended output
            if "GFPGAN blended faces:" in s or "GFPGAN output:" in s:
                if not suppress_all_preprocess_ui:
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

            if self._try_fast_rephoto_iteration_progress(s):
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
    def _reset_main_window_for_new_input(self):
        """Clear selection/runtime preview state when user imports a new input image."""
        self.stop_quick_face_probe()
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()
        self.quick_face_count_estimate = None
        self._template_match_cache = {}
        self.auto_detect_faces_armed_input = None
        self.auto_detect_faces_triggered_input = None
        self.suppress_preprocess_ui_until_rephoto = False
        if hasattr(self, "input_image_edit"):
            self.input_image_edit.setText("")
        self.set_input_detect_overlay(False)
        self.reset_face_preview_state(preserve_input_overlays=False)
        self.clear_result_stage_overlay()
        self.reset_progress_bars()
        self._reset_wrapper_runtime_tracking()
        self.current_run_summary_context = None
        self.set_result_preview_image(None)
        self.result_preview_label.setText("No result image yet.\nRun to generate a preview.")
        self._rephoto_result_path = None
        self._recomposited_result_path = None
        self._compare_wipe_last_pos = None  # Reset wipe throttle
        self.result_view_toggle.setVisible(False)
        self.result_view_toggle.setChecked(False)
        self.result_view_toggle.setText("Rephoto")
        self.update_run_button_for_quick_face_hint()

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
        self.update_run_button_for_quick_face_hint()

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

    def get_hardware_info(self):
        now = time.time()
        if (
            isinstance(self._hardware_info_cache, dict)
            and (now - float(self._hardware_info_cache_ts)) < float(self._hardware_info_cache_ttl_sec)
        ):
            return dict(self._hardware_info_cache)

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

        info = {
            "cpu_name": cpu_name,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "gpu_name": gpu_name,
        }
        self._hardware_info_cache = dict(info)
        self._hardware_info_cache_ts = now
        return info

    def load_local_timing_records(self):
        if hasattr(self, "results_root_edit"):
            results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
        else:
            results_root = self.repo_root / "results"
        log_path = results_root / "run_timing_log.jsonl"
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

        if not log_path.exists():
            self._timing_records_cache_path = cache_path
            self._timing_records_cache_mtime = None
            self._timing_records_cache = []
            return []

        records = []
        try:
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
            self._timing_records_cache_path = cache_path
            self._timing_records_cache_mtime = log_mtime
            self._timing_records_cache = []
            return []

        self._timing_records_cache_path = cache_path
        self._timing_records_cache_mtime = log_mtime
        self._timing_records_cache = list(records)
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
            return statistics.median(s for s, _rt in matches)

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
                    on_avg = statistics.median(vals["on"])
                    off_avg = statistics.median(vals["off"])
                    overheads.append((p, max(0.0, on_avg - off_avg)))
            return sorted(overheads, key=lambda t: t[0])

        overhead_points = build_overhead_points(use_same_gpu=True)
        if not overhead_points:
            overhead_points = build_overhead_points(use_same_gpu=False)

        overhead_secs = None
        overhead_note = "No paired enhancement on/off local records yet."

        exact_over = [ov for (p, ov) in overhead_points if p == preset_value]
        if exact_over:
            overhead_secs = statistics.median(exact_over)
            overhead_note = "Exact local enhancement overhead from paired record(s) at this preset."
        elif overhead_points:
            overhead_secs = statistics.median(ov for (_p, ov) in overhead_points)
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
            base_secs = statistics.median(exact_base)
            base_note = f"Exact local base timing from {len(exact_base)} enhancement-off record(s)."
        else:
            grouped_base = {}
            for p, s in base_candidates:
                grouped_base.setdefault(p, []).append(s)

            base_points = sorted((p, statistics.median(vals)) for p, vals in grouped_base.items())

            if len(base_points) >= 2:
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
        face_multiplier = self.get_runtime_face_multiplier()
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

        hw = self.get_hardware_info()
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
    def append_command_preview(self, command):
        preview = " ".join(f'"{part}"' if " " in part else part for part in command)
        self.log_box.append("Wrapper command:")
        self.log_box.append(preview)

    def build_wrapper_command(
        self,
        force_crop_only=None,
        force_use_existing_crops=False,
        crop_indices=None,
        crop_names=None,
        require_selection=False,
    ):
        input_image = self.input_image_edit.text().strip()
        results_root = self.results_root_edit.text().strip() or str(self.repo_root / "results")

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
            "all",
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

        stop_flag_path = str(self.current_stop_flag_path or "").strip()
        if stop_flag_path:
            command.extend([
                "-StopFlagPath",
                stop_flag_path,
            ])
        pause_flag_path = str(self.current_pause_flag_path or "").strip()
        if pause_flag_path:
            command.extend([
                "-PauseFlagPath",
                pause_flag_path,
            ])

        normalized_crop_indices = []
        if crop_indices is not None:
            for raw_idx in crop_indices:
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if idx >= 0:
                    normalized_crop_indices.append(idx)
            normalized_crop_indices = sorted(set(normalized_crop_indices))
        normalized_crop_names = []
        if crop_names is not None:
            seen_names = set()
            for raw_name in crop_names:
                name = str(raw_name or "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                normalized_crop_names.append(name)

        crop_only_mode = self.advanced_dialog.crop_only_checkbox.isChecked() if force_crop_only is None else bool(force_crop_only)
        if crop_only_mode:
            command.append("-CropOnly")
        else:
            if require_selection:
                command.append("-RequireSelection")
                if (not normalized_crop_indices) and (not normalized_crop_names):
                    raise ValueError("No face selection was provided for selected-face continuation.")
            if force_use_existing_crops:
                command.append("-UseExistingCrops")
            if normalized_crop_names:
                command.append("-SelectedCropNames")
                command.append(",".join(normalized_crop_names))
            if normalized_crop_indices:
                command.append("-SelectedCropIndices")
                command.append(",".join(str(i) for i in normalized_crop_indices))
            # Enhancement is enabled only if: checkbox is unchecked (not disabled) AND GFPGAN is available
            enhancement_enabled = (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()) and self.gfpgan_is_available()
            if enhancement_enabled:
                blend_value = float(self.advanced_dialog.gfpgan_blend_edit.value())
                if blend_value >= 0.999:
                    # In basic mode, full replacement tends to over-smooth local features
                    # (especially eyes). Auto-reset to the conservative default.
                    blend_value = float(DEFAULT_GFPGAN_BLEND)
                    self.advanced_dialog.gfpgan_blend_edit.setValue(blend_value)
                    self.log_box.append(
                        f"GFPGAN blend 1.0 detected in Basic mode; reset to {blend_value:.2f} to avoid eye smearing."
                    )
                command.extend([
                    "-UseGFPGAN",
                    "-GFPGANBlend",
                    f"{blend_value:.2f}",
                ])
            # Recomposite is always enabled (checkbox was removed as redundant)
            command.append("-RecompositeOriginalImage")

        return command

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

    def clear_face_preview_strip_layout(self):
        self._face_strip_render_signature = None
        if self.hover_face_preview_index is not None:
            self.hover_face_preview_index = None
            self.hover_face_preview_source = None
            self.refresh_input_preview_scale()
        while self.face_preview_strip_layout.count() > 0:
            item = self.face_preview_strip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def reset_face_preview_state(self, preserve_input_overlays=False):
        self.face_preview_entries = []
        self.active_face_preview_index = None
        self.selected_face_preview_index = None
        self.hover_face_preview_index = None
        self.hover_face_preview_source = None
        self.hover_face_box_override = None
        self.hover_face_box_cache = {}
        self.awaiting_face_selection = False
        self.face_preview_summary_label.setText("Faces: none")
        self.clear_face_preview_strip_layout()
        self.face_preview_header.setVisible(False)
        self.face_preview_strip_scroll.setVisible(False)
        if hasattr(self, "face_preview_panel"):
            self.face_preview_panel.setVisible(False)
        self.set_run_button_continue_mode(False)
        self._face_strip_render_signature = None

        if not preserve_input_overlays:
            self.input_face_boxes = []
            self.input_face_box_source = None
            self.input_preview_render_cache_key = None
            self.input_preview_render_cache_pixmap = None
            self.input_preview_last_display_key = None
            self.refresh_input_preview_scale()
        self.update_runtime_label()
        self.update_image_import_controls()

    def set_run_button_continue_mode(self, is_continue_mode):
        if is_continue_mode:
            can_continue = len(self.get_selected_face_indices()) > 0
            self.run_button.setText("Run")
            self.run_button.setEnabled(can_continue)
            if can_continue:
                self.run_button.setToolTip("Select one or more faces in the filmstrip, then run rephotography.")
            else:
                self.run_button.setToolTip("Select at least one face in the filmstrip to enable Run.")
            return
        self.update_run_button_for_quick_face_hint()

    def update_run_button_for_quick_face_hint(self):
        if getattr(self, "awaiting_face_selection", False):
            can_continue = len(self.get_selected_face_indices()) > 0
            self.run_button.setText("Run")
            self.run_button.setEnabled(can_continue)
            if can_continue:
                self.run_button.setToolTip("Select one or more faces in the filmstrip, then run rephotography.")
            else:
                self.run_button.setToolTip("Select at least one face in the filmstrip to enable Run.")
            return

        if self.process is not None:
            if self.run_paused:
                self.run_button.setText("Resume")
                self.run_button.setToolTip("Resume the paused backend run.")
            else:
                self.run_button.setText("Pause")
                self.run_button.setToolTip("Pause the current backend run.")
            self.run_button.setEnabled(True)
            return

        if self.current_run_phase in {"preprocess", "rephoto"}:
            self.run_button.setText("Run")
            self.run_button.setEnabled(False)
            self.run_button.setToolTip("Run the full rephotography workflow.")
            return

        input_path_text = self.input_image_edit.text().strip() if hasattr(self, "input_image_edit") else ""
        input_ready = False
        if input_path_text:
            try:
                input_ready = Path(input_path_text).exists()
            except Exception:
                input_ready = False

        if not input_ready:
            self.run_button.setText("Run")
            self.run_button.setEnabled(False)
            self.run_button.setToolTip("Choose an input image to enable Run.")
            return

        crop_only_mode = self.advanced_dialog.crop_only_checkbox.isChecked()

        # Keep default behavior in crop-only mode.
        if crop_only_mode:
            self.run_button.setText("Run")
            self.run_button.setEnabled(True)
            self.run_button.setToolTip("Run the full rephotography workflow.")
            return

        # UX rule:
        # - If we are confidently single-face (==1), keep simple "Run".
        # - Otherwise (multi-face, zero-face, or unknown estimate), signal the detect/select step.
        quick_count = self.quick_face_count_estimate if isinstance(self.quick_face_count_estimate, int) else None
        if quick_count == 1:
            self.run_button.setText("Run")
            self.run_button.setEnabled(True)
            self.run_button.setToolTip("Single face detected. Run the full rephotography workflow.")
        else:
            self.run_button.setText("Run")
            self.run_button.setEnabled(True)
            self.run_button.setToolTip(
                "Running starts with face detection, then prompts face selection when multiple faces are found."
            )

    def handle_run_button_clicked(self):
        if self.process is not None:
            self.toggle_pause_resume()
            return
        self.run_wrapper()

    def _set_status_for_running_state(self):
        if self.current_run_phase == "preprocess":
            self.status_label.setText("Status: Detecting faces...")
        elif self.current_run_phase == "rephoto":
            self.status_label.setText("Status: Rephotographing...")
        else:
            self.status_label.setText("Status: Running backend...")

    def _control_process_tree_windows(self, root_pid, action):
        try:
            pid = int(root_pid)
        except Exception:
            return (False, "Invalid process id.")
        if pid <= 0:
            return (False, "Process id is not active.")
        if platform.system().lower() != "windows":
            return (False, "Pause/resume is currently implemented for Windows only.")
        if action not in {"pause", "resume"}:
            return (False, f"Unsupported action: {action}")

        cmdlet = "Suspend-Process" if action == "pause" else "Resume-Process"
        order = "Sort-Object -Descending" if action == "pause" else "Sort-Object"

        script = f"""
$rootPid = {pid}
$all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
$children = @{{}}
foreach ($p in $all) {{
    $ppid = [int]$p.ParentProcessId
    $cpid = [int]$p.ProcessId
    if (-not $children.ContainsKey($ppid)) {{ $children[$ppid] = New-Object System.Collections.Generic.List[int] }}
    $children[$ppid].Add($cpid)
}}
$stack = New-Object System.Collections.Generic.Stack[int]
$stack.Push([int]$rootPid)
$ids = New-Object System.Collections.Generic.List[int]
while ($stack.Count -gt 0) {{
    $cur = [int]$stack.Pop()
    if ($ids.Contains($cur)) {{ continue }}
    $ids.Add($cur)
    if ($children.ContainsKey($cur)) {{
        foreach ($child in $children[$cur]) {{ $stack.Push([int]$child) }}
    }}
}}
$ordered = $ids | {order}
foreach ($id in $ordered) {{
    try {{ {cmdlet} -Id $id -ErrorAction Stop | Out-Null }} catch {{ }}
}}
Write-Output "OK"
"""
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=creation_flags,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "Process tree control command failed."
            return (False, detail)
        return (True, "")

    def set_backend_paused(self, paused, quiet=False):
        if self.process is None:
            return False
        target = bool(paused)
        if self.run_paused == target:
            return True

        pause_path_text = str(self.current_pause_flag_path or "").strip()
        if not pause_path_text:
            if not quiet:
                self.log_box.append("Pause control is not initialized for this run.")
                self.status_label.setText("Status: Pause unavailable")
            return False

        pause_path = Path(pause_path_text)
        try:
            if target:
                pause_path.parent.mkdir(parents=True, exist_ok=True)
                pause_path.write_text(f"{time.time()}\n", encoding="utf-8")
            else:
                if pause_path.exists():
                    pause_path.unlink()
        except Exception as exc:
            if not quiet:
                self.log_box.append(f"{'Pause' if target else 'Resume'} failed: {exc}")
                self.status_label.setText(f"Status: {'Pause' if target else 'Resume'} failed")
            return False

        self.run_paused = target
        if not quiet:
            if target:
                self.log_box.append("Pause requested. Backend will pause at the next safe checkpoint.")
                self.status_label.setText("Status: Pause requested...")
                self._hide_result_stage_overlay_for_pause()
            else:
                self.log_box.append("Run resumed.")
                self._set_status_for_running_state()
                self._restore_result_stage_overlay_after_pause()
        self.update_run_button_for_quick_face_hint()
        return True

    def toggle_pause_resume(self):
        if self.process is None:
            return
        self.set_backend_paused(not self.run_paused, quiet=False)

    def _prepare_stop_flag_for_new_run(self):
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()
        flags_dir = self.repo_root / "preprocess" / "_control_flags"
        flags_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        stop_path = flags_dir / f"stop_{stamp}_{os.getpid()}.flag"
        pause_path = flags_dir / f"pause_{stamp}_{os.getpid()}.flag"
        self.current_stop_flag_path = str(stop_path)
        self.current_pause_flag_path = str(pause_path)
        return stop_path

    def _clear_current_stop_flag(self):
        stop_path_text = self.current_stop_flag_path
        self.current_stop_flag_path = None
        if not stop_path_text:
            return
        try:
            p = Path(stop_path_text)
            if p.exists():
                p.unlink()
        except OSError:
            pass

    def _clear_current_pause_flag(self):
        pause_path_text = self.current_pause_flag_path
        self.current_pause_flag_path = None
        if not pause_path_text:
            return
        try:
            p = Path(pause_path_text)
            if p.exists():
                p.unlink()
        except OSError:
            pass

    def request_end_run_early(self):
        if self.process is None:
            self.log_box.append("No backend process is running.")
            self.status_label.setText("Status: No backend process to end early")
            return

        confirm = QMessageBox.question(
            self,
            "End Run Early",
            "Are you sure you want to end this run early?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            self.status_label.setText("Status: End-early canceled")
            return

        stop_path_text = str(self.current_stop_flag_path or "").strip()
        if not stop_path_text:
            self.log_box.append("Early-stop control is not initialized for this run.")
            self.status_label.setText("Status: End-early unavailable")
            return

        stop_path = Path(stop_path_text)
        try:
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.write_text(f"{time.time()}\n", encoding="utf-8")
        except Exception as exc:
            self.log_box.append(f"Failed to request early end: {exc}")
            self.status_label.setText("Status: End-early failed")
            return

        if self.run_paused:
            # Allow the backend to consume the early-stop flag immediately.
            self.set_backend_paused(False, quiet=True)
            self.run_paused = False
            self.update_run_button_for_quick_face_hint()

        self.end_early_button.setEnabled(False)
        self.log_box.append("End-early requested. Finishing with completed iterations only...")
        self.status_label.setText("Status: Ending run early...")

    def _normalized_path_key(self, path_text):
        if not path_text:
            return None
        try:
            return os.path.normcase(str(Path(path_text).resolve()))
        except Exception:
            return os.path.normcase(str(path_text))

    def _make_safe_base_name(self, base_text):
        raw = str(base_text or "")
        if not raw:
            return "input_image"
        out = []
        prev_sep = False
        for ch in raw:
            if ch.isalnum():
                out.append(ch)
                prev_sep = False
            else:
                if not prev_sep:
                    out.append("_")
                    prev_sep = True
        safe = "".join(out).strip("_")
        return safe or "input_image"

    def _seed_wrapper_crop_dir_from_preview(self, input_image_path: Path):
        if input_image_path is None or (not input_image_path.exists()):
            return

        preview_crops = []
        for entry in self.face_preview_entries:
            crop_path = entry.get("crop_path")
            if crop_path is None:
                continue
            p = Path(crop_path)
            if p.exists() and p.is_file():
                preview_crops.append(p)
        if not preview_crops:
            return

        safe_base = self._make_safe_base_name(input_image_path.stem)
        target_dir = self.repo_root / "preprocess" / "face_crops" / safe_base
        target_dir.mkdir(parents=True, exist_ok=True)
        if self._list_image_files_in_dir(target_dir):
            return

        copied = 0
        for src in preview_crops:
            dst = target_dir / src.name
            try:
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                copied += 1
            except Exception:
                continue

        if copied > 0:
            self.current_crop_output_dir = str(target_dir)
            self._crop_source_input_key = self._normalized_path_key(self.input_image_edit.text().strip())
            self._crop_source_face_factor = self.advanced_dialog.face_factor_edit.value()
            self.log_box.append(f"Prepared reusable crops: {target_dir}")

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

        self.update_run_button_for_quick_face_hint()
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
            self.update_run_button_for_quick_face_hint()
            self.set_input_detect_overlay(False)
            return

        image_path = Path(image_path_text)
        if not image_path.exists():
            self.stop_quick_face_probe()
            self.quick_face_count_estimate = None
            self.update_run_button_for_quick_face_hint()
            self.set_input_detect_overlay(False)
            return

        # Launch RetinaFace probe for face detection.
        # When the probe finishes, _quick_face_probe_finished will set the
        # face count, create preview crops, and populate the filmstrip.
        self.update_run_button_for_quick_face_hint()
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

    def get_selected_face_indices(self):
        if not self.face_preview_entries:
            return []
        selected = []
        for entry in self.face_preview_entries:
            idx = entry.get("index")
            if not isinstance(idx, int):
                continue
            if bool(entry.get("selected", False)):
                selected.append(idx)
        return sorted(set(selected))

    def get_selected_face_count_text(self):
        if not self.face_preview_entries:
            return "N/A"
        selected_count = len(self.get_selected_face_indices())
        return f"{selected_count}/{len(self.face_preview_entries)}"

    def get_runtime_face_multiplier(self):
        if not self.face_preview_entries:
            return 1
        selected_count = len(self.get_selected_face_indices())
        return max(1, selected_count)

    def _list_image_files_in_dir(self, folder: Path):
        if folder is None or (not folder.exists()) or (not folder.is_dir()):
            return []
        def _sort_key(p: Path):
            match = FACE_SUFFIX_INDEX_RE.search(p.stem)
            if match:
                try:
                    return (0, int(match.group(1)), p.name.lower())
                except Exception:
                    pass
            return (1, p.name.lower())
        return sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=_sort_key,
        )

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

    def _sync_face_preview_crop_paths(self):
        if not self.face_preview_entries:
            return
        crop_files = []
        if self.current_crop_output_dir:
            crop_files = self._list_image_files_in_dir(Path(self.current_crop_output_dir))

        if not crop_files:
            input_path_text = self.input_image_edit.text().strip()
            if input_path_text:
                input_path = Path(input_path_text)
                safe_base = self._make_safe_base_name(input_path.stem)
                by_input_dir = self.repo_root / "preprocess" / "face_crops" / safe_base
                crop_files = self._list_image_files_in_dir(by_input_dir)

        if not crop_files:
            crop_files = self.collect_current_crop_files()
        if not crop_files:
            return

        files_by_index = {}
        for pos, p in enumerate(crop_files):
            idx = None
            match = FACE_SUFFIX_INDEX_RE.search(p.stem)
            if match:
                try:
                    idx = int(match.group(1))
                except Exception:
                    idx = None
            if idx is None:
                idx = int(pos)
            if idx not in files_by_index:
                files_by_index[idx] = p

        updated = False
        for entry in self.face_preview_entries:
            idx = entry.get("index")
            if not isinstance(idx, int) or idx < 0:
                continue
            new_path = files_by_index.get(idx)
            if new_path is None and idx < len(crop_files):
                new_path = crop_files[idx]
            if new_path is None:
                continue

            current = entry.get("crop_path")
            if current is None:
                entry["crop_path"] = new_path
                updated = True
                continue
            if self._normalized_path_key(str(current)) != self._normalized_path_key(str(new_path)):
                entry["crop_path"] = new_path
                updated = True

        if updated:
            self.hover_face_box_cache = {}
            self.hover_face_box_override = None

    def initialize_face_preview_entries(self, expected_count=None):
        crop_files = self.collect_current_crop_files()
        if expected_count is None:
            expected_count = len(crop_files)
        expected_count = max(0, int(expected_count))
        if expected_count == 0 and not crop_files:
            return
        if expected_count == 0:
            expected_count = len(crop_files)

        self.face_preview_entries = []
        self.hover_face_box_cache = {}
        self.hover_face_box_override = None
        for idx in range(expected_count):
            crop_path = crop_files[idx] if idx < len(crop_files) else None
            self.face_preview_entries.append(
                {
                    "index": idx,
                    "crop_path": crop_path,
                    "result_path": None,
                    "status": "queued",
                    "selected": True,
                }
            )

        self.active_face_preview_index = None
        self.selected_face_preview_index = 0 if self.face_preview_entries else None
        self._sync_face_preview_crop_paths()
        self.update_input_face_boxes_for_preview(expected_count=expected_count)
        self.update_runtime_label()
        self.render_face_preview_strip()

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

    def _make_face_thumb_icon(self, image_path, fallback_text, muted=False, thumb_size=84):
        thumb_size = max(56, int(thumb_size))
        cache_key = self._face_thumb_icon_cache_key(
            image_path=image_path,
            fallback_text=fallback_text,
            muted=muted,
            thumb_size=thumb_size,
        )
        cached = self.face_preview_thumb_icon_cache.pop(cache_key, None)
        if cached is not None:
            self.face_preview_thumb_icon_cache[cache_key] = cached
            return cached

        inner_size = thumb_size - 2
        thumb = QPixmap(thumb_size, thumb_size)
        thumb.fill(QColor("#242933"))

        source_pix = None
        if image_path is not None:
            p = Path(image_path)
            if p.exists():
                pix = QPixmap(str(p))
                if not pix.isNull():
                    source_pix = pix

        painter = QPainter(thumb)
        painter.setRenderHint(QPainter.Antialiasing)

        if source_pix is not None:
            scaled = source_pix.scaled(inner_size, inner_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (thumb_size - scaled.width()) // 2
            y = (thumb_size - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            if muted:
                # Deliberately darken skipped faces to improve selected/skipped contrast.
                painter.fillRect(thumb.rect(), QColor(18, 21, 26, 150))
        else:
            painter.setPen(QColor("#7f8794"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(thumb.rect(), Qt.AlignCenter, fallback_text)

        painter.setPen(QPen(QColor("#59606b"), 1))
        painter.drawRect(0, 0, thumb_size - 1, thumb_size - 1)
        painter.end()
        icon = QIcon(thumb)
        self.face_preview_thumb_icon_cache[cache_key] = icon
        while len(self.face_preview_thumb_icon_cache) > int(self.face_preview_thumb_icon_cache_max_entries):
            oldest_key = next(iter(self.face_preview_thumb_icon_cache))
            self.face_preview_thumb_icon_cache.pop(oldest_key, None)
        return icon

    def _compute_face_strip_render_signature(self, entries, selection_mode, is_processing, wide_mode):
        card_w = self._get_face_strip_card_width(wide_mode)
        selected_idx = int(self.selected_face_preview_index) if isinstance(self.selected_face_preview_index, int) else -1
        active_idx = int(self.active_face_preview_index) if isinstance(self.active_face_preview_index, int) else -1
        entry_signature = []
        for entry in entries:
            idx = entry.get("index")
            if not isinstance(idx, int):
                continue
            icon_mtime_ns = 0
            icon_size = 0
            icon_candidate = entry.get("result_path") or entry.get("crop_path")
            if icon_candidate is not None:
                p = Path(icon_candidate)
                try:
                    if p.exists():
                        st = p.stat()
                        icon_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
                        icon_size = int(st.st_size)
                except OSError:
                    pass
            entry_signature.append(
                (
                    idx,
                    str(entry.get("status", "queued")),
                    bool(entry.get("selected", False)),
                    str(entry.get("crop_path") or ""),
                    str(entry.get("result_path") or ""),
                    icon_mtime_ns,
                    icon_size,
                )
            )
        return (
            bool(selection_mode),
            bool(is_processing),
            bool(wide_mode),
            int(card_w),
            selected_idx,
            active_idx,
            tuple(entry_signature),
        )

    def render_face_preview_strip(self):
        entries = self.face_preview_entries
        selection_mode = self.awaiting_face_selection
        is_processing = self._is_processing_active()
        wide_mode = bool(getattr(self, "_wide_layout_active", False))
        render_signature = self._compute_face_strip_render_signature(
            entries=entries,
            selection_mode=selection_mode,
            is_processing=is_processing,
            wide_mode=wide_mode,
        )
        if self._face_strip_render_signature == render_signature:
            # Still ensure filmstrip is visible even if signature matches
            if hasattr(self, "face_preview_panel"):
                self.face_preview_panel.setVisible(True)
            self.face_preview_header.setVisible(bool(entries))
            self.face_preview_strip_scroll.setVisible(bool(entries))
            return

        self.clear_face_preview_strip_layout()
        if wide_mode:
            self.face_preview_strip_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        else:
            self.face_preview_strip_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        if not entries:
            self.face_preview_summary_label.setText("Faces: none")
            if hasattr(self, "face_selection_notice_label"):
                self.face_selection_notice_label.setVisible(False)
            self.face_preview_header.setVisible(False)
            self.face_preview_strip_scroll.setVisible(False)
            if hasattr(self, "face_preview_panel"):
                self.face_preview_panel.setVisible(False)
            self._face_strip_render_signature = render_signature
            return

        # Consolidate counts into a single loop instead of N+1 iterations
        selected_count = done_count = fail_count = 0
        for e in entries:
            if bool(e.get("selected", False)):
                selected_count += 1
            if e.get("status") == "done":
                done_count += 1
            if e.get("status") == "failed":
                fail_count += 1

        if selection_mode:
            self.run_button.setEnabled(selected_count > 0)
            summary = f"Selected: {selected_count}/{len(entries)}"
            if hasattr(self, "face_selection_notice_label"):
                self.face_selection_notice_label.setText(
                    f"Select one or more faces to run ({selected_count}/{len(entries)} selected)"
                )
                self.face_selection_notice_label.setVisible(True)
        else:
            if wide_mode:
                summary = f"Faces {len(entries)} | Done {done_count}"
            else:
                summary = f"Faces: {len(entries)} | Done: {done_count}"
            if fail_count:
                summary += f" | Failed {fail_count}" if wide_mode else f" | Failed: {fail_count}"
            if hasattr(self, "face_selection_notice_label"):
                self.face_selection_notice_label.setVisible(False)
        self.face_preview_summary_label.setText(summary)

        # Always show the filmstrip, even with single face
        if hasattr(self, "face_preview_panel"):
            self.face_preview_panel.setVisible(True)
        if hasattr(self, "face_select_all_button"):
            # Hide select all/none buttons when single face or not in selection mode
            self.face_select_all_button.setVisible(selection_mode and len(entries) > 1)
        if hasattr(self, "face_select_none_button"):
            self.face_select_none_button.setVisible(selection_mode and len(entries) > 1)
        if hasattr(self, "face_preview_auto_follow_checkbox"):
            self.face_preview_auto_follow_checkbox.setVisible(not selection_mode)
        self.face_preview_header.setVisible(True)
        self.face_preview_strip_scroll.setVisible(True)

        status_style_map = {
            "queued": "#3d4450",
            "running": "#1f6fd9",
            "done": "#2e8b57",
            "failed": "#b05050",
            "skipped": "#525a66",
        }

        display_entries = entries
        if not selection_mode:
            # Keep selected faces first once continuation starts.
            display_entries = sorted(entries, key=lambda e: (not bool(e.get("selected", False)), e["index"]))

        card_w = self._get_face_strip_card_width(wide_mode)
        thumb_size = max(72, min(104, card_w - 24))
        card_h = thumb_size + (50 if wide_mode else 42)

        if wide_mode:
            self.face_preview_strip_layout.addStretch(1)

        for entry in display_entries:
            idx = entry["index"]
            status = entry.get("status", "queued")
            is_selected = bool(entry.get("selected", False))
            if selection_mode:
                label = "Selected" if is_selected else "Skipped"
            else:
                label = status.capitalize()
            icon_path = entry.get("result_path") or entry.get("crop_path")

            button = FaceStripToolButton()
            button.setCheckable(True)
            button.setChecked(is_selected if selection_mode else (idx == self.selected_face_preview_index))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            is_muted = not is_selected
            button.setIconSize(QSize(thumb_size, thumb_size))
            button.setFixedSize(card_w, card_h)
            button.setCursor(Qt.PointingHandCursor)
            button.setFont(QFont("Segoe UI", 8))
            button.setText(f"Face {idx + 1}\n{label}")
            button.setIcon(self._make_face_thumb_icon(icon_path, str(idx + 1), muted=is_muted, thumb_size=thumb_size))
            if selection_mode:
                accent = "#1f6fd9" if is_selected else "#525a66"
            else:
                accent = status_style_map.get(status, "#3d4450")
            border = "#1a73e8" if button.isChecked() else accent
            if is_muted:
                accent = "#3e4652"
                border = accent if not button.isChecked() else "#5b6778"
                text_color = "#748091"
                bg_color = "#171b20"
                hover_color = "#1b2026"
            else:
                text_color = "#d6dbe3"
                bg_color = "#232830"
                hover_color = "#29303a"
            button.setStyleSheet(FaceStripToolButton.get_stylesheet(border, bg_color, text_color, hover_color))
            if is_muted and (not selection_mode) and is_processing:
                button.setEnabled(False)
            button.setProperty("faceIndex", idx)
            button.installEventFilter(self)
            button.hover_enter_callback = (lambda i=idx: self.set_hover_face_preview_index(i))
            button.hover_leave_callback = self.clear_hover_face_preview_index
            button.clicked.connect(
                lambda checked=False, i=idx: self.select_face_preview(i, user_initiated=True, selection_checked=checked)
            )
            if wide_mode:
                self.face_preview_strip_layout.addWidget(button, 0, Qt.AlignHCenter)
            else:
                self.face_preview_strip_layout.addWidget(button)

        self.face_preview_strip_layout.addStretch(1)
        self._face_strip_render_signature = render_signature

    def set_hover_face_preview_index(self, face_index, source="strip"):
        idx = face_index
        if idx is not None:
            try:
                idx = int(idx)
            except Exception:
                idx = None
        source_name = str(source or "strip")
        if isinstance(idx, int) and (not self._is_face_interaction_allowed(idx)):
            if self.hover_face_preview_source == source_name:
                self.clear_hover_face_preview_index()
            return
        if self.hover_face_preview_index == idx and self.hover_face_preview_source == source_name:
            return
        if self.hover_face_preview_index is None:
            self.result_preview_path_before_hover = self.last_result_image_path
        self.hover_face_preview_index = idx
        self.hover_face_preview_source = source_name if isinstance(idx, int) else None
        self.hover_face_box_override = None
        if isinstance(idx, int):
            cached_box = self.hover_face_box_cache.get(idx)
            if cached_box is None:
                cached_box = self.resolve_hover_face_box(idx)
                self.hover_face_box_cache[idx] = cached_box
            self.hover_face_box_override = cached_box
            hover_preview_path = self.get_face_preview_path(idx)
            if hover_preview_path is not None:
                self.set_result_preview_image(hover_preview_path)
        self.refresh_input_preview_scale()

    def clear_hover_face_preview_index(self):
        had_hover = (self.hover_face_preview_index is not None) or (self.hover_face_box_override is not None)
        self.hover_face_preview_index = None
        self.hover_face_preview_source = None
        self.hover_face_box_override = None
        if had_hover:
            restore_path = self.get_selected_face_preview_path()
            if restore_path is None:
                restore_path = self.result_preview_path_before_hover
            if restore_path is not None:
                self.set_result_preview_image(restore_path)
            self.result_preview_path_before_hover = None
            self.refresh_input_preview_scale()

    def _cursor_face_preview_index(self):
        """Get the face index of the currently hovered button.

        Optimized to use cached hover state instead of expensive widget tree queries.
        """
        widget = FaceStripToolButton._currently_hovered
        if not isinstance(widget, FaceStripToolButton):
            return None
        try:
            return int(widget.property("faceIndex"))
        except Exception:
            return None

    def resolve_hover_face_box(self, face_index):
        if face_index is None:
            return None
        try:
            idx = int(face_index)
        except Exception:
            return None
        if idx < 0:
            return None

        # Prefer detector-aligned boxes first. These are already index-aligned
        # to cropper output and avoid template false matches.
        if idx < len(self.input_face_boxes):
            try:
                x, y, w, h = [int(v) for v in self.input_face_boxes[idx]]
            except Exception:
                return None
            if w >= 8 and h >= 8:
                return (x, y, w, h)

        # Fallback only when detector boxes are unavailable.
        box = self.resolve_face_box_from_crop_template(idx)
        if box is not None:
            return box
        return None

    def _input_preview_display_rect(self):
        label = self.input_preview_label
        pix = label.pixmap()
        if pix is None or pix.isNull():
            return None
        pw = int(pix.width())
        ph = int(pix.height())
        if pw <= 0 or ph <= 0:
            return None
        ox = max(0, (label.width() - pw) // 2)
        oy = max(0, (label.height() - ph) // 2)
        return QRect(ox, oy, pw, ph)

    def _source_point_from_input_preview_pos(self, pos):
        if self.input_pixmap is None:
            return None
        display_rect = self._input_preview_display_rect()
        if display_rect is None or (not display_rect.contains(pos)):
            return None
        if display_rect.width() <= 0 or display_rect.height() <= 0:
            return None
        rel_x = float(pos.x() - display_rect.x())
        rel_y = float(pos.y() - display_rect.y())
        src_x = rel_x * (float(self.input_pixmap.width()) / float(display_rect.width()))
        src_y = rel_y * (float(self.input_pixmap.height()) / float(display_rect.height()))
        return (src_x, src_y)

    def _is_processing_active(self):
        return (self.process is not None) or (self.current_run_phase in {"preprocess", "rephoto"})

    def _is_face_interaction_allowed(self, face_index):
        try:
            idx = int(face_index)
        except Exception:
            return False
        if idx < 0 or idx >= len(self.face_preview_entries):
            return False
        if self.awaiting_face_selection:
            return True
        if not self._is_processing_active():
            return True
        return bool(self.face_preview_entries[idx].get("selected", False))

    def _resolve_input_interaction_face_boxes(self):
        if not self.face_preview_entries:
            return []
        resolved = []
        for entry in self.face_preview_entries:
            idx = entry.get("index")
            if not isinstance(idx, int) or idx < 0:
                continue
            if not self._is_face_interaction_allowed(idx):
                continue
            if idx in self.hover_face_box_cache:
                box = self.hover_face_box_cache.get(idx)
            else:
                box = self.resolve_hover_face_box(idx)
                self.hover_face_box_cache[idx] = box
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

    def handle_input_preview_mouse_move(self, pos):
        if not self.face_preview_entries:
            return
        idx = self._hit_test_input_face_index(pos)
        if isinstance(idx, int) and self._is_face_interaction_allowed(idx):
            self.set_hover_face_preview_index(idx, source="input")
            return
        if self.hover_face_preview_source == "input":
            self.clear_hover_face_preview_index()

    def handle_input_preview_mouse_leave(self):
        if self.hover_face_preview_source == "input":
            self.clear_hover_face_preview_index()

    def handle_input_preview_click(self, pos):
        if not self.face_preview_entries:
            return False
        idx = self._hit_test_input_face_index(pos)
        if (not isinstance(idx, int)) or (not self._is_face_interaction_allowed(idx)):
            return False
        self.set_hover_face_preview_index(idx, source="input")
        self.select_face_preview(idx, user_initiated=True)
        return True

    def resolve_face_box_from_crop_template(self, face_index):
        if not self.face_preview_entries:
            return None
        if face_index < 0 or face_index >= len(self.face_preview_entries):
            return None

        entry = self.face_preview_entries[face_index]
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
                if et in (QEvent.MouseMove, QEvent.HoverMove):
                    vp_pos = event.pos()
                    container_pos = self.face_preview_strip_container.mapFrom(watched, vp_pos)
                    child = self.face_preview_strip_container.childAt(container_pos)
                    while child is not None and (not isinstance(child, FaceStripToolButton)):
                        child = child.parentWidget()
                    if isinstance(child, FaceStripToolButton):
                        face_index = child.property("faceIndex")
                        if self._is_face_interaction_allowed(face_index):
                            self.set_hover_face_preview_index(face_index)
                        else:
                            self.clear_hover_face_preview_index()
                    else:
                        self.clear_hover_face_preview_index()
                elif et in (QEvent.Leave, QEvent.HoverLeave):
                    self.clear_hover_face_preview_index()
                return super().eventFilter(watched, event)

            if isinstance(watched, FaceStripToolButton):
                et = event.type()
                if et in (QEvent.Enter, QEvent.HoverEnter, QEvent.MouseMove, QEvent.HoverMove):
                    face_index = watched.property("faceIndex")
                    if self._is_face_interaction_allowed(face_index):
                        self.set_hover_face_preview_index(face_index)
                    else:
                        self.clear_hover_face_preview_index()
                elif et in (QEvent.Leave, QEvent.HoverLeave):
                    self.clear_hover_face_preview_index()
        except Exception:
            # Never let hover diagnostics break event dispatch.
            pass
        return super().eventFilter(watched, event)

    def select_face_preview(self, face_index, user_initiated=False, selection_checked=None):
        if face_index < 0 or face_index >= len(self.face_preview_entries):
            return
        if user_initiated and (not self.awaiting_face_selection) and (not self._is_face_interaction_allowed(face_index)):
            return

        self.selected_face_preview_index = face_index
        entry = self.face_preview_entries[face_index]
        if self.awaiting_face_selection and user_initiated:
            if selection_checked is None:
                entry["selected"] = not bool(entry.get("selected", False))
            else:
                entry["selected"] = bool(selection_checked)
        chosen_path = self.get_face_preview_path(face_index)
        if chosen_path is not None and Path(chosen_path).exists():
            self.set_result_preview_image(Path(chosen_path))

        if user_initiated and self.face_preview_auto_follow_checkbox.isChecked() and (not self.awaiting_face_selection):
            # Keep auto-follow enabled but still allow manual inspection.
            pass

        if self.awaiting_face_selection:
            self.run_button.setEnabled(len(self.get_selected_face_indices()) > 0)
            self.update_runtime_label()

        self.render_face_preview_strip()
        self.refresh_input_preview_scale()

    def handle_face_auto_follow_toggled(self, checked):
        if not checked or not self.face_preview_entries:
            return

        candidate_idx = None
        for entry in reversed(self.face_preview_entries):
            result_path = entry.get("result_path")
            if result_path is not None and Path(result_path).exists():
                candidate_idx = entry["index"]
                break

        if candidate_idx is None and self.active_face_preview_index is not None:
            candidate_idx = self.active_face_preview_index
        if candidate_idx is None:
            candidate_idx = 0

        self.select_face_preview(candidate_idx, user_initiated=False)

    def set_all_faces_selected(self, selected):
        if not self.awaiting_face_selection or not self.face_preview_entries:
            return
        for entry in self.face_preview_entries:
            entry["selected"] = bool(selected)
        self.run_button.setEnabled(len(self.get_selected_face_indices()) > 0)
        self.update_runtime_label()
        self.render_face_preview_strip()
        self.refresh_input_preview_scale()

    def get_selected_face_preview_path(self):
        if not self.face_preview_entries:
            return None

        idx = self.selected_face_preview_index
        if idx is None or idx < 0 or idx >= len(self.face_preview_entries):
            idx = None
            for entry in self.face_preview_entries:
                if entry.get("status") == "done" and entry.get("result_path") is not None:
                    idx = entry["index"]
                    break
            if idx is None:
                idx = 0
            self.selected_face_preview_index = idx

        return self.get_face_preview_path(idx)

    def get_face_preview_path(self, face_index):
        if not self.face_preview_entries:
            return None
        try:
            idx = int(face_index)
        except Exception:
            return None
        if idx < 0 or idx >= len(self.face_preview_entries):
            return None

        entry = self.face_preview_entries[idx]
        preview_path = entry.get("result_path")
        if preview_path is None:
            crop_path = entry.get("crop_path")
            if crop_path is not None:
                preview_path = self._resolve_enhanced_preview_for_crop(Path(crop_path)) or crop_path
        if preview_path is None:
            return None
        p = Path(preview_path)
        return p if p.exists() else None

    def _get_focused_face_preview_index(self):
        if not self.face_preview_entries:
            return None
        candidates = [
            self.active_face_preview_index,
            self.hover_face_preview_index,
            self.selected_face_preview_index,
        ]
        for candidate in candidates:
            if isinstance(candidate, int) and 0 <= candidate < len(self.face_preview_entries):
                return candidate
        selected_indices = self.get_selected_face_indices()
        if selected_indices:
            idx = int(selected_indices[0])
            if 0 <= idx < len(self.face_preview_entries):
                return idx
        return None

    def _get_face_crop_path(self, face_index):
        if not self.face_preview_entries:
            return None
        try:
            idx = int(face_index)
        except Exception:
            return None
        if idx < 0 or idx >= len(self.face_preview_entries):
            return None
        crop_path = self.face_preview_entries[idx].get("crop_path")
        if crop_path is None:
            return None
        p = Path(crop_path)
        return p if p.exists() else None

    def _get_compare_before_source_pixmap(self):
        focused_idx = self._get_focused_face_preview_index()
        crop_path = self._get_face_crop_path(focused_idx) if focused_idx is not None else None
        if crop_path is None and focused_idx is not None:
            self._sync_face_preview_crop_paths()
            crop_path = self._get_face_crop_path(focused_idx)
        if crop_path is not None:
            pix = self._get_result_pixmap_cached(crop_path)
            if pix is not None and (not pix.isNull()):
                return pix
        return self.input_pixmap

    def _resolve_enhanced_preview_for_crop(self, crop_path: Path):
        if crop_path is None or (not self.current_blended_faces_dir):
            return None
        blended_dir = Path(self.current_blended_faces_dir)
        if not blended_dir.exists():
            return None

        stem = crop_path.stem
        preferred = []
        crop_suffix = crop_path.suffix.lower()
        if crop_suffix:
            preferred.append(blended_dir / f"{stem}_blend{crop_suffix}")
        preferred.append(blended_dir / f"{stem}_blend.png")
        for candidate in preferred:
            if candidate.exists():
                return candidate

        matches = []
        for candidate in blended_dir.glob(f"{stem}_blend.*"):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                matches.append(candidate)
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def _find_face_index_for_crop_path(self, crop_path: Path):
        crop_name = crop_path.name.lower()
        crop_stem = crop_path.stem.lower()
        for entry in self.face_preview_entries:
            entry_crop = entry.get("crop_path")
            if entry_crop is None:
                continue
            ec = Path(entry_crop)
            if ec.name.lower() == crop_name or ec.stem.lower() == crop_stem:
                return entry["index"]
        return None

    def _find_face_index_for_result_path(self, result_path: Path):
        result_stem = result_path.stem.lower()
        for entry in self.face_preview_entries:
            entry_crop = entry.get("crop_path")
            if entry_crop is None:
                continue
            crop_stem = Path(entry_crop).stem.lower()
            if not crop_stem:
                continue
            if result_stem == crop_stem:
                return entry["index"]
            if result_stem.startswith(crop_stem + "-"):
                return entry["index"]
            if result_stem.startswith(f"final_{crop_stem}_p"):
                return entry["index"]
            if result_stem == f"{crop_stem}_blend":
                return entry["index"]
        return None

    def mark_face_running_from_crop_path(self, crop_path_text):
        if not self.face_preview_entries:
            return

        crop_path = Path(crop_path_text.strip())
        idx = self._find_face_index_for_crop_path(crop_path)
        if idx is None:
            for entry in self.face_preview_entries:
                if entry.get("status") in {"queued", "failed"}:
                    idx = entry["index"]
                    break

        if idx is None:
            return

        entry = self.face_preview_entries[idx]
        if not bool(entry.get("selected", False)):
            return
        if entry.get("crop_path") is None:
            entry["crop_path"] = crop_path
        if entry.get("status") != "done":
            entry["status"] = "running"

        self.active_face_preview_index = idx
        if self.face_preview_auto_follow_checkbox.isChecked():
            self.selected_face_preview_index = idx
            preview_path = entry.get("result_path")
            if preview_path is None:
                preview_path = self._resolve_enhanced_preview_for_crop(crop_path) or entry.get("crop_path")
            if preview_path is not None and Path(preview_path).exists():
                self.set_result_preview_image(Path(preview_path))
        self.render_face_preview_strip()

    def mark_face_done_from_result_path(self, result_path_text):
        result_path = Path(result_path_text.strip())
        if not self.face_preview_entries:
            self.initialize_face_preview_entries(expected_count=1)

        idx = self.active_face_preview_index
        if idx is None:
            idx = self._find_face_index_for_result_path(result_path)
        if idx is None:
            for entry in self.face_preview_entries:
                if bool(entry.get("selected", False)) and entry.get("status") != "done":
                    idx = entry["index"]
                    break

        if idx is None:
            idx = len(self.face_preview_entries)
            self.face_preview_entries.append(
                {
                    "index": idx,
                    "crop_path": None,
                    "result_path": None,
                    "status": "queued",
                    "selected": True,
                }
            )

        entry = self.face_preview_entries[idx]
        if not bool(entry.get("selected", False)):
            return
        entry["result_path"] = result_path
        entry["status"] = "done"
        self.active_face_preview_index = None

        if self.face_preview_auto_follow_checkbox.isChecked():
            self.selected_face_preview_index = idx
            if result_path.exists():
                self.set_result_preview_image(result_path)

        self.render_face_preview_strip()

    def mark_face_failed_from_crop_name(self, crop_name_text):
        text = (crop_name_text or "").strip()
        if not text or not self.face_preview_entries:
            return
        crop_name = Path(text).name.lower()
        crop_stem = Path(text).stem.lower()

        idx = None
        for entry in self.face_preview_entries:
            entry_crop = entry.get("crop_path")
            if entry_crop is None:
                continue
            ec = Path(entry_crop)
            if ec.name.lower() == crop_name or ec.stem.lower() == crop_stem:
                idx = entry["index"]
                break

        if idx is None and self.active_face_preview_index is not None:
            idx = self.active_face_preview_index
        if idx is None:
            return

        self.face_preview_entries[idx]["status"] = "failed"
        self.active_face_preview_index = None
        self.render_face_preview_strip()

    def reconcile_face_preview_results(self, after_epoch=None):
        if not self.face_preview_entries:
            return

        selected_pending = [
            e for e in self.face_preview_entries
            if bool(e.get("selected", False)) and e.get("status") != "done"
        ]
        if not selected_pending:
            if self.selected_face_preview_index is None:
                for entry in self.face_preview_entries:
                    if entry.get("status") == "done":
                        self.selected_face_preview_index = entry["index"]
                        break
            self.render_face_preview_strip()
            return

        results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
        if not results_root.exists():
            return

        final_candidates = []
        search_dirs = []
        for dir_text in sorted(self.current_run_result_dirs):
            p = Path(dir_text)
            if p.exists() and p.is_dir():
                search_dirs.append(p)

        if search_dirs:
            for folder in search_dirs:
                for p in folder.iterdir():
                    if (not p.is_file()) or p.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    name = p.name.lower()
                    if "-init" in name or "-rand" in name:
                        continue
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        continue
                    if after_epoch is not None and mtime < after_epoch:
                        continue
                    final_candidates.append((mtime, p))

        if not final_candidates:
            for p in results_root.rglob("final_*_p*.png"):
                if not p.is_file():
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if after_epoch is not None and mtime < after_epoch:
                    continue
                final_candidates.append((mtime, p))

        final_candidates.sort(key=lambda t: t[0])
        for _mtime, p in final_candidates:
            idx = self._find_face_index_for_result_path(p)
            if idx is None:
                for entry in self.face_preview_entries:
                    if bool(entry.get("selected", False)) and entry.get("status") != "done":
                        idx = entry["index"]
                        break
            if idx is None:
                continue
            self.face_preview_entries[idx]["result_path"] = p
            self.face_preview_entries[idx]["status"] = "done"

        if self.selected_face_preview_index is None:
            for entry in self.face_preview_entries:
                if entry.get("status") == "done":
                    self.selected_face_preview_index = entry["index"]
                    break

        self.render_face_preview_strip()

    def update_input_face_boxes_for_preview(self, expected_count=None):
        self.input_face_boxes = []
        self.input_face_box_source = None
        self.hover_face_box_cache = {}
        self.hover_face_box_override = None
        if expected_count is None or expected_count <= 0:
            self.refresh_input_preview_scale()
            return
        image_path_text = self.input_image_edit.text().strip()
        if not image_path_text:
            self.refresh_input_preview_scale()
            return

        image_path = Path(image_path_text)
        if not image_path.exists():
            self.refresh_input_preview_scale()
            return

        # Multi-face mapping should prefer cropper probe, which mirrors the same
        # alignment transform used for face-crop outputs.
        if expected_count and int(expected_count) > 1:
            cropper_boxes = self.resolve_input_face_boxes_via_cropper_probe(image_path, expected_count=expected_count)
            if len(cropper_boxes) >= int(expected_count):
                self.input_face_boxes = cropper_boxes
                self.input_face_box_source = "cropper_probe"
                self.refresh_input_preview_scale()
                return
            retina_boxes = self.resolve_input_face_boxes_via_retina_probe(image_path, expected_count=expected_count)
            if len(retina_boxes) >= int(expected_count):
                self.input_face_boxes = retina_boxes
                self.input_face_box_source = "retina_probe"
                self.refresh_input_preview_scale()
                return

        try:
            import cv2
        except Exception:
            if (not self._face_overlay_detector_warned) and expected_count and expected_count > 1:
                self.log_box.append("OpenCV unavailable in GUI env; attempting Retina face-box probe.")
                self._face_overlay_detector_warned = True
            retina_boxes = self.resolve_input_face_boxes_via_retina_probe(image_path, expected_count=expected_count)
            if retina_boxes:
                self.input_face_boxes = retina_boxes
                self.input_face_box_source = "retina_probe"
            self.refresh_input_preview_scale()
            return

        img = cv2.imread(str(image_path))
        if img is None:
            self.refresh_input_preview_scale()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = self._haar_face_detector
        if detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            self._haar_face_detector = detector
        if detector.empty():
            self.refresh_input_preview_scale()
            return

        detected = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected]
        if not boxes:
            retina_boxes = self.resolve_input_face_boxes_via_retina_probe(image_path, expected_count=expected_count)
            if retina_boxes:
                self.input_face_boxes = retina_boxes
                self.input_face_box_source = "retina_probe"
            self.refresh_input_preview_scale()
            return

        if expected_count is not None and expected_count > 0:
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:expected_count]
        boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        if expected_count is not None and expected_count > 0 and len(boxes) < int(expected_count):
            retina_boxes = self.resolve_input_face_boxes_via_retina_probe(image_path, expected_count=expected_count)
            if len(retina_boxes) >= len(boxes):
                boxes = retina_boxes
                self.input_face_box_source = "retina_probe"
        self.input_face_boxes = boxes
        if self.input_face_box_source is None:
            self.input_face_box_source = "cascade"
        self.refresh_input_preview_scale()

    def set_input_preview_image(self, image_path: Path | None):
        self.input_pixmap = None
        self.hover_face_box_override = None
        self.hover_face_preview_source = None
        self.hover_face_box_cache = {}
        self.input_preview_scaled_cache_key = None
        self.input_preview_scaled_cache_pixmap = None
        self.input_preview_render_cache_key = None
        self.input_preview_render_cache_pixmap = None
        self.input_preview_last_display_key = None
        if image_path is None or (not image_path.exists()):
            self.input_face_boxes = []
            self.input_face_box_source = None
            self.set_input_detect_overlay(False)
            self.input_preview_label.setText("No input image yet.")
            self.input_preview_label.setPixmap(QPixmap())
            return

        pix = QPixmap(str(image_path))
        if pix.isNull():
            self.input_face_boxes = []
            self.input_face_box_source = None
            self.set_input_detect_overlay(False)
            self.input_preview_label.setText("Could not load input image.")
            self.input_preview_label.setPixmap(QPixmap())
            return

        self.input_pixmap = pix
        self.update_input_face_boxes_for_preview(
            expected_count=len(self.face_preview_entries) if self.face_preview_entries else None
        )
        if hasattr(self, "input_detect_overlay") and self.input_detect_overlay.isVisible():
            self.input_detect_overlay.notify_parent_pixmap_changed()

    def _result_preview_cache_key(self, image_path: Path):
        p = Path(image_path)
        try:
            normalized = str(p.resolve()).lower()
        except Exception:
            normalized = str(p).lower()
        try:
            st = p.stat()
            return f"{normalized}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
        except OSError:
            return normalized

    def _get_result_pixmap_cached(self, image_path: Path):
        key = self._result_preview_cache_key(image_path)
        cached = self.result_preview_pixmap_cache.pop(key, None)
        if cached is not None and (not cached.isNull()):
            self.result_preview_pixmap_cache[key] = cached
            return cached

        pix = QPixmap(str(image_path))
        if pix.isNull():
            return None

        self.result_preview_pixmap_cache[key] = pix
        while len(self.result_preview_pixmap_cache) > int(self.result_preview_pixmap_cache_max_entries):
            oldest_key = next(iter(self.result_preview_pixmap_cache))
            self.result_preview_pixmap_cache.pop(oldest_key, None)
        return pix

    def set_result_preview_image(self, image_path: Path | None):
        prev_pix = self.result_pixmap
        self.open_image_location_button.setEnabled(False)
        self.recomposite_button.setEnabled(False)

        if image_path is None or (not image_path.exists()):
            self.last_result_image_path = None
            self.last_result_image_cache_key = None
            self.result_pixmap = None
            self.result_preview_scaled_cache_key = None
            self.result_preview_scaled_cache_pixmap = None
            self.result_preview_last_display_key = None
            self.result_preview_label.setText("No result image found.")
            self.result_preview_label.setPixmap(QPixmap())
            return

        image_path = Path(image_path)
        cache_key = self._result_preview_cache_key(image_path)
        if self.last_result_image_cache_key is not None and prev_pix is not None:
            if self.last_result_image_cache_key == cache_key:
                self.last_result_image_path = image_path
                self.last_result_image_cache_key = cache_key
                self.result_pixmap = prev_pix
                self.open_image_location_button.setEnabled(True)
                self._update_recomposite_button_state()
                self.refresh_result_preview_scale()
                self.position_result_stage_overlay()
                return

        pix = self._get_result_pixmap_cached(image_path)
        if pix is None:
            self.last_result_image_path = None
            self.last_result_image_cache_key = None
            self.result_pixmap = None
            self.result_preview_scaled_cache_key = None
            self.result_preview_scaled_cache_pixmap = None
            self.result_preview_last_display_key = None
            self.result_preview_label.setText("Could not load result image.")
            self.result_preview_label.setPixmap(QPixmap())
            return

        self.last_result_image_path = image_path
        self.last_result_image_cache_key = cache_key
        self.result_pixmap = pix
        self.result_preview_scaled_cache_key = None
        self.result_preview_scaled_cache_pixmap = None
        self.result_preview_last_display_key = None
        self.open_image_location_button.setEnabled(True)
        self._update_recomposite_button_state()
        self.refresh_result_preview_scale()
        self.position_result_stage_overlay()

    def refresh_input_preview_scale(self):
        if self.input_pixmap is None:
            return
        w = max(1, self.input_preview_label.width() - 2)
        h = max(1, self.input_preview_label.height() - 2)
        try:
            source_key = int(self.input_pixmap.cacheKey())
        except Exception:
            source_key = 0
        cache_key = (source_key, w, h)
        if (
            self.input_preview_scaled_cache_key == cache_key
            and self.input_preview_scaled_cache_pixmap is not None
            and (not self.input_preview_scaled_cache_pixmap.isNull())
        ):
            base_scaled = self.input_preview_scaled_cache_pixmap
        else:
            base_scaled = self.input_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.input_preview_scaled_cache_key = cache_key
            self.input_preview_scaled_cache_pixmap = base_scaled

        active_idx = self.hover_face_preview_index
        if isinstance(active_idx, int) and self.hover_face_preview_source == "strip":
            cursor_idx = self._cursor_face_preview_index()
            if isinstance(cursor_idx, int):
                if cursor_idx != active_idx:
                    self.hover_face_preview_index = cursor_idx
                    self.hover_face_preview_source = "strip"
                active_idx = cursor_idx
            else:
                self.hover_face_preview_index = None
                self.hover_face_preview_source = None
                self.hover_face_box_override = None
                active_idx = None

        active_box = self.hover_face_box_override
        if active_box is None and isinstance(active_idx, int):
            cached_box = self.hover_face_box_cache.get(active_idx)
            if cached_box is None:
                cached_box = self.resolve_hover_face_box(active_idx)
                self.hover_face_box_cache[active_idx] = cached_box
            active_box = cached_box

        if self.awaiting_face_selection:
            selected_indices = [e["index"] for e in self.face_preview_entries if e.get("selected", False)]
        else:
            selected_indices = []
            if isinstance(self.selected_face_preview_index, int) and self.selected_face_preview_index >= 0:
                selected_indices = [self.selected_face_preview_index]

        selected_boxes = []
        for idx in selected_indices:
            cached_box = self.hover_face_box_cache.get(idx)
            if cached_box is None:
                cached_box = self.resolve_hover_face_box(idx)
                self.hover_face_box_cache[idx] = cached_box
            if cached_box is not None:
                selected_boxes.append((idx, cached_box))

        draw_debug_boxes = bool(self.face_box_debug_overlay_enabled) and bool(self.input_face_boxes)
        draw_hover_box = isinstance(active_idx, int) and (active_box is not None)
        draw_selected_boxes = bool(selected_boxes)
        debug_boxes_sig = tuple(
            tuple(int(v) for v in box) for box in self.input_face_boxes
        ) if draw_debug_boxes else ()
        selected_boxes_sig = tuple(
            (int(idx), tuple(int(v) for v in box)) for idx, box in selected_boxes
        ) if draw_selected_boxes else ()
        active_box_sig = tuple(int(v) for v in active_box) if draw_hover_box else ()
        render_key = (
            cache_key,
            draw_debug_boxes,
            debug_boxes_sig,
            draw_selected_boxes,
            selected_boxes_sig,
            draw_hover_box,
            int(active_idx) if isinstance(active_idx, int) else -1,
            active_box_sig,
        )

        if (
            self.input_preview_render_cache_key == render_key
            and self.input_preview_render_cache_pixmap is not None
            and (not self.input_preview_render_cache_pixmap.isNull())
        ):
            scaled = self.input_preview_render_cache_pixmap
        else:
            scaled = base_scaled
            if draw_debug_boxes or draw_hover_box or draw_selected_boxes:
                draw = QPixmap(base_scaled)
                painter = QPainter(draw)
                painter.setRenderHint(QPainter.Antialiasing)

                sx = draw.width() / max(1.0, float(self.input_pixmap.width()))
                sy = draw.height() / max(1.0, float(self.input_pixmap.height()))

                if draw_debug_boxes:
                    debug_pen = QPen(QColor(102, 181, 255, 150))
                    debug_pen.setWidth(1)
                    painter.setPen(debug_pen)
                    painter.setFont(QFont("Segoe UI", 8))
                    for idx, box in enumerate(self.input_face_boxes):
                        x, y, bw, bh = box
                        rx = int(round(x * sx))
                        ry = int(round(y * sy))
                        rw = max(8, int(round(bw * sx)))
                        rh = max(8, int(round(bh * sy)))
                        hx = max(0, rx - 1)
                        hy = max(0, ry - 1)
                        hw = min(draw.width() - hx - 1, rw + 2)
                        hh = min(draw.height() - hy - 1, rh + 2)
                        if hw <= 4 or hh <= 4:
                            continue
                        painter.drawRoundedRect(hx, hy, hw, hh, 4, 4)
                        badge_w = 16
                        badge_h = 12
                        badge_y = max(0, hy - badge_h)
                        painter.fillRect(hx, badge_y, badge_w, badge_h, QColor(32, 63, 92, 190))
                        painter.setPen(QColor("#d8ecff"))
                        painter.drawText(hx, badge_y, badge_w, badge_h, Qt.AlignCenter, str(idx + 1))
                        painter.setPen(debug_pen)

                if draw_selected_boxes:
                    for selected_idx, selected_box in selected_boxes:
                        x, y, bw, bh = selected_box
                        rx = int(round(x * sx))
                        ry = int(round(y * sy))
                        rw = max(8, int(round(bw * sx)))
                        rh = max(8, int(round(bh * sy)))

                        hx = max(0, rx - 2)
                        hy = max(0, ry - 2)
                        hw = min(draw.width() - hx - 1, rw + 4)
                        hh = min(draw.height() - hy - 1, rh + 4)
                        if hw <= 4 or hh <= 4:
                            continue

                        selected_pen = QPen(QColor(102, 181, 255, 210))
                        selected_pen.setWidth(2)
                        painter.setPen(selected_pen)
                        painter.drawRoundedRect(hx, hy, hw, hh, 5, 5)

                        badge_w = 20
                        badge_h = 14
                        badge_y = max(0, hy - badge_h)
                        painter.fillRect(hx, badge_y, badge_w, badge_h, QColor(39, 93, 152, 205))
                        painter.setPen(QColor("#eaf5ff"))
                        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                        painter.drawText(hx, badge_y, badge_w, badge_h, Qt.AlignCenter, str(selected_idx + 1))

                if draw_hover_box:
                    x, y, bw, bh = active_box
                    rx = int(round(x * sx))
                    ry = int(round(y * sy))
                    rw = max(8, int(round(bw * sx)))
                    rh = max(8, int(round(bh * sy)))

                    hx = max(0, rx - 2)
                    hy = max(0, ry - 2)
                    hw = min(draw.width() - hx - 1, rw + 4)
                    hh = min(draw.height() - hy - 1, rh + 4)
                    if hw > 4 and hh > 4:
                        hover_pen = QPen(QColor("#66b5ff"))
                        hover_pen.setWidth(3)
                        painter.setPen(hover_pen)
                        painter.drawRoundedRect(hx, hy, hw, hh, 5, 5)

                        badge_w = 22
                        badge_h = 16
                        badge_y = max(0, hy - badge_h)
                        painter.fillRect(hx, badge_y, badge_w, badge_h, QColor(26, 115, 232, 220))
                        painter.setPen(QColor("#ffffff"))
                        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                        painter.drawText(hx, badge_y, badge_w, badge_h, Qt.AlignCenter, str(active_idx + 1))
                painter.end()
                scaled = draw
            self.input_preview_render_cache_key = render_key
            self.input_preview_render_cache_pixmap = scaled

        if self.input_preview_last_display_key == render_key:
            return
        self.input_preview_last_display_key = render_key
        self.input_preview_label.setPixmap(scaled)

    def refresh_result_preview_scale(self):
        if self.result_pixmap is None:
            return
        w = max(1, self.result_preview_label.width() - 2)
        h = max(1, self.result_preview_label.height() - 2)
        try:
            source_key = int(self.result_pixmap.cacheKey())
        except Exception:
            source_key = 0
        # Avoid enlarging low-resolution intermediate crops; upscaling can make previews look artificially blurry.
        target_w = min(w, self.result_pixmap.width())
        target_h = min(h, self.result_pixmap.height())
        cache_key = (source_key, target_w, target_h)
        if (
            self.result_preview_scaled_cache_key == cache_key
            and self.result_preview_scaled_cache_pixmap is not None
            and (not self.result_preview_scaled_cache_pixmap.isNull())
        ):
            scaled = self.result_preview_scaled_cache_pixmap
        else:
            scaled = self.result_pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_preview_scaled_cache_key = cache_key
            self.result_preview_scaled_cache_pixmap = scaled
        if self.result_preview_last_display_key == cache_key:
            return
        self.result_preview_last_display_key = cache_key
        self.result_preview_label.setPixmap(scaled)

    def _apply_compare_wipe(self, mouse_pos):
        """Composite a left=input / right=result wipe on the result preview label.

        Throttled: only rescales if mouse moved > 4px since last scale to avoid
        excessive CPU cost on continuous MouseMove events (60+ Hz).
        """
        label = self.result_preview_label
        lw, lh = label.width() - 10, label.height() - 10
        if lw < 10 or lh < 10:
            return

        # Throttle: skip if mouse moved < 4 pixels (Manhattan distance)
        if self._compare_wipe_last_pos is not None:
            dx = abs(mouse_pos.x() - self._compare_wipe_last_pos.x())
            dy = abs(mouse_pos.y() - self._compare_wipe_last_pos.y())
            if dx + dy < 4:
                return
        self._compare_wipe_last_pos = mouse_pos

        # Scale both to the same display size (cached from previous call if unchanged)
        result_scaled = self.result_pixmap.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        sw, sh = result_scaled.width(), result_scaled.height()
        if sw <= 0 or sh <= 0:
            return
        before_pixmap = self._get_compare_before_source_pixmap()
        if before_pixmap is None:
            return
        input_scaled = before_pixmap.scaled(
            sw, sh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        # Mouse x relative to the pixmap (centered in label).
        offset_x = (label.width() - sw) // 2
        split_x = max(0, min(sw, mouse_pos.x() - offset_x))

        composite = QPixmap(sw, sh)
        composite.fill(QColor(0, 0, 0, 0))
        painter = QPainter(composite)
        # Left side: input (before)
        if split_x > 0:
            painter.drawPixmap(0, 0, input_scaled, 0, 0, split_x, sh)
        # Right side: result (after)
        if split_x < sw:
            painter.drawPixmap(split_x, 0, result_scaled, split_x, 0, sw - split_x, sh)
        # Divider line
        pen = QPen(QColor(255, 255, 255, 200), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, sh)
        # Labels
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 180))
        if split_x > 40:
            painter.drawText(8, sh - 8, "Before")
        if (sw - split_x) > 40:
            painter.drawText(sw - 42, sh - 8, "After")
        painter.end()

        self._compare_wipe_active = True
        label.setPixmap(composite)

    def position_input_detect_overlay(self):
        if not hasattr(self, "input_detect_overlay"):
            return
        self.input_detect_overlay.sync_geometry()

    def set_input_detect_overlay(self, active, base_text="Detecting Faces"):
        if not hasattr(self, "input_detect_overlay"):
            return
        if active:
            self.position_input_detect_overlay()
            if (
                self.input_detect_overlay.isVisible()
                and str(getattr(self.input_detect_overlay, "base_text", "")) == str(base_text)
            ):
                return
            self.input_detect_overlay.start(base_text)
            return
        self.input_detect_overlay.stop()

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

    def _hide_result_stage_overlay_for_pause(self):
        current = str(self.result_stage_base_text or "").strip()
        if current:
            self._paused_result_stage_base_text = current
        self.clear_result_stage_overlay()

    def _restore_result_stage_overlay_after_pause(self):
        base = str(getattr(self, "_paused_result_stage_base_text", "") or "").strip()
        self._paused_result_stage_base_text = ""
        if self.process is None:
            return
        if base:
            self.set_result_stage_overlay(base)
            return
        if self.current_run_phase == "rephoto":
            self.set_result_stage_overlay("Rephotographing")

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

        focused_idx = self._get_focused_face_preview_index()
        focused_crop = self._get_face_crop_path(focused_idx) if focused_idx is not None else None
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
        idx = self._get_focused_face_preview_index()
        if idx is not None:
            return self._get_face_crop_path(idx)
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

    def _append_process_log_text(self, text: str):
        if not text:
            return
        self._process_log_pending_text.append(text)
        self._process_log_pending_text_bytes += len(text)
        # Keep pending text bounded in case backend emits very large bursts.
        if self._process_log_pending_text_bytes > 500000:
            # Join, truncate, and reset list
            joined = "".join(self._process_log_pending_text)[-500000:]
            self._process_log_pending_text = [joined]
            self._process_log_pending_text_bytes = len(joined)
        if self._process_log_flush_queued:
            return
        self._process_log_flush_queued = True
        QTimer.singleShot(50, self._flush_process_log_buffer)

    def _flush_process_log_buffer(self):
        self._process_log_flush_queued = False
        if not self._process_log_pending_text:
            return
        text = "".join(self._process_log_pending_text)
        self._process_log_pending_text = []
        self._process_log_pending_text_bytes = 0
        text = text.replace("\r", "\n")
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_box.setTextCursor(cursor)
        self.log_box.ensureCursorVisible()

    def _consume_process_output_lines(self, chunk_text: str, is_error: bool = False):
        if not chunk_text:
            return
        if is_error:
            combined = self._process_stderr_buffer + chunk_text
        else:
            combined = self._process_stdout_buffer + chunk_text
        # Keep incomplete trailing fragments for the next chunk.
        lines = combined.splitlines()
        if combined and combined[-1] not in ("\n", "\r"):
            if lines:
                remainder = lines.pop()
            else:
                remainder = combined
        else:
            remainder = ""

        if is_error:
            self._process_stderr_buffer = remainder
        else:
            self._process_stdout_buffer = remainder

        for line in lines:
            self.update_progress_from_line(line)

    def append_stdout_from_process(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._append_process_log_text(text)
            self._consume_process_output_lines(text, is_error=False)

    def append_stderr_from_process(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self._append_process_log_text(text)
            self._consume_process_output_lines(text, is_error=True)

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

            with open(log_path, "a", encoding="utf-8") as f:
                for pm in self._pending_milestones:
                    record = {
                        "record_type": "milestone",
                        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "input_image": self.input_image_edit.text().strip(),
                        "preset": str(pm["preset"]),
                        "source_run_preset": source_preset,
                        "advanced_mode": False,
                        "enhancement": (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()),
                        "crop_only": False,
                        "success": True,
                        "elapsed_seconds": float(pm["elapsed_seconds"]),
                        "gpu_name": hw.get("gpu_name", "Unknown GPU"),
                    }
                    f.write(json.dumps(record) + "\n")

            self.log_box.append(f"Saved {len(self._pending_milestones)} milestone timing record(s).")
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
            results_root.mkdir(parents=True, exist_ok=True)
            log_path = results_root / "run_timing_log.jsonl"

            hw = self.get_hardware_info() if hasattr(self, "get_hardware_info") else {}
            preset_val = str(self.iter_values[self.iter_slider.value()])

            record = {
                "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_image": self.input_image_edit.text().strip(),
                "preset": preset_val,
                "advanced_mode": False,
                "enhancement": (not self.advanced_dialog.use_gfpgan_checkbox.isChecked()),
                "crop_only": bool(crop_only),
                "success": bool(success),
                "elapsed_seconds": float(elapsed_seconds),
                "gpu_name": hw.get("gpu_name", "Unknown GPU"),
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self._timing_records_cache_mtime = None

            self.log_box.append(f"Saved timing log: {log_path}")
        except Exception as e:
            self.log_box.append(f"Timing log write failed: {e}")

    def _can_reuse_existing_crops(self, input_path_key):
        """Check if valid face-crop-plus crops exist for the given input image."""
        if not self.current_crop_output_dir:
            return False
        if self._inprocess_preview_crops:
            # Preview crops from quick probe are not production quality
            return False
        if self._crop_source_input_key != input_path_key:
            # Crops belong to a different input image
            return False
        # If face_factor (crop expansion) changed, crops must be regenerated
        if self._crop_source_face_factor is not None:
            current_ff = self.advanced_dialog.face_factor_edit.value()
            if abs(current_ff - self._crop_source_face_factor) > 1e-6:
                return False
        crop_dir = Path(self.current_crop_output_dir)
        if not crop_dir.is_dir():
            return False
        crop_files = self._list_image_files_in_dir(crop_dir)
        return len(crop_files) > 0

    def _reset_wrapper_runtime_tracking(self, preserve_crop_dir=False):
        if not preserve_crop_dir:
            self.current_crop_output_dir = None
            self._crop_source_input_key = None
            self._crop_source_face_factor = None
        self.current_gfpgan_output_dir = None
        self.current_blended_faces_dir = None
        self.current_results_dir = None
        self.current_manifest_path = None
        self.current_run_result_dirs = set()
        self.stage_started_at = {}
        self.stage_elapsed = {}
        self.rephoto_started_at = None
        self._last_iter_progress_signature = None

    def _start_wrapper_process(self, command, status_text):
        self.append_command_preview(command)
        self.status_label.setText(status_text)
        self.run_paused = False
        self._paused_result_stage_base_text = ""
        self.set_controls_for_running(True)
        if self.current_run_summary_context is not None:
            try:
                preset_idx = command.index("-Preset")
                if preset_idx + 1 < len(command):
                    self.current_run_summary_context["effective_preset"] = command[preset_idx + 1]
            except ValueError:
                pass

        self.run_started_at = time.time()
        self._process_stdout_buffer = ""
        self._process_stderr_buffer = ""
        self._process_log_pending_text = []  # Use list for O(1) append
        self._process_log_pending_text_bytes = 0
        self._process_log_flush_queued = False
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
        self.update_run_button_for_quick_face_hint()

    def _prepare_face_selection_after_preprocess(self):
        self.set_input_detect_overlay(False)
        crop_files = self.collect_current_crop_files()
        crop_count = len(crop_files)
        self.set_preprocess_progress(0, "Preprocess ready")
        self.set_rephoto_progress(0, "Waiting...")
        if crop_count <= 0:
            self.log_box.append("No detected faces were found after preprocessing.")
            self.status_label.setText("Status: No faces detected")
            self.clear_result_stage_overlay()
            self.show_run_summary_dialog(
                success=False,
                exit_code=1,
                elapsed_seconds=None,
                output_path=None,
                launch_error="No detected faces were found.",
            )
            self.current_run_summary_context = None
            return

        self.initialize_face_preview_entries(expected_count=crop_count)
        self._sync_face_preview_crop_paths()

        if crop_count == 1:
            for entry in self.face_preview_entries:
                entry["selected"] = True
                entry["status"] = "queued"
            self.render_face_preview_strip()
            self.log_box.append("Single face detected. Continuing automatically...")
            self.status_label.setText("Status: Single face detected, continuing...")
            self.awaiting_face_selection = False
            self.update_image_import_controls()
            self.set_run_button_continue_mode(False)
            QTimer.singleShot(0, self.continue_rephoto_with_selected_faces)
            return

        # Restore previous face selections if this was a re-crop (expansion change)
        reselect = self._pending_face_reselection
        self._pending_face_reselection = None
        for entry in self.face_preview_entries:
            idx = entry.get("index")
            entry["selected"] = (reselect is not None and idx in reselect)
            entry["status"] = "queued"

        self.awaiting_face_selection = True
        self.update_image_import_controls()
        self.set_run_button_continue_mode(True)
        self.run_button.setEnabled(False)
        if hasattr(self, "face_select_all_button"):
            self.face_select_all_button.setEnabled(True)
        if hasattr(self, "face_select_none_button"):
            self.face_select_none_button.setEnabled(True)
        self.status_label.setText("Status: Select one or more faces, then click Run")
        self.log_box.append(f"Detected {crop_count} faces. Select one or more faces in the strip, then click Run.")
        self.clear_result_stage_overlay()
        self.render_face_preview_strip()

    def continue_rephoto_with_selected_faces(self):
        if self.process is not None:
            return

        selected_indices = self.get_selected_face_indices()
        if not selected_indices:
            self.status_label.setText("Status: Select at least one face to run")
            self.log_box.append("Select at least one face to rephotograph before running.")
            self.run_button.setEnabled(False)
            return

        self._sync_face_preview_crop_paths()
        self.awaiting_face_selection = False
        self.suppress_preprocess_ui_until_rephoto = False
        self.set_input_detect_overlay(False)
        self.update_image_import_controls()
        self.selection_preprocess_mode = False
        self.set_run_button_continue_mode(False)

        selected_set = set(selected_indices)
        for entry in self.face_preview_entries:
            if entry["index"] in selected_set:
                entry["selected"] = True
                if entry.get("status") != "done":
                    entry["status"] = "queued"
            else:
                entry["selected"] = False
                entry["status"] = "skipped"
        self.selected_face_preview_index = selected_indices[0]
        self.render_face_preview_strip()
        self.current_run_summary_context = self._capture_run_context()
        self.current_run_summary_context["selected_faces"] = self.get_selected_face_count_text()
        self.update_runtime_label()

        self.clear_result_stage_overlay()
        self.reset_progress_bars()
        self.current_run_phase = "preprocess"
        self.set_preprocess_progress(5, "Preparing selected faces...")
        self.set_rephoto_progress(0, "Waiting...")
        self._reset_wrapper_runtime_tracking()
        self._prepare_stop_flag_for_new_run()

        # Continuation must reuse the exact crop set shown in selection UI.
        # Re-cropping here can remap indices and select the wrong face.
        reuse_crops = True
        self._inprocess_preview_crops = False

        selected_crop_names = []
        for idx in selected_indices:
            crop_path = self._get_face_crop_path(idx)
            if crop_path is not None:
                selected_crop_names.append(crop_path.name)
        use_selected_names = len(selected_crop_names) == len(selected_indices)
        try:
            command = self.build_wrapper_command(
                force_crop_only=False,
                force_use_existing_crops=reuse_crops,
                crop_indices=None if use_selected_names else selected_indices,
                crop_names=selected_crop_names if use_selected_names else None,
                require_selection=True,
            )
        except ValueError as exc:
            self.status_label.setText("Status: Face selection error")
            self.log_box.append(str(exc))
            self.awaiting_face_selection = True
            self.set_run_button_continue_mode(True)
            self.run_button.setEnabled(len(self.get_selected_face_indices()) > 0)
            return

        if use_selected_names:
            self.log_box.append(f"Running selected crop name(s): {', '.join(selected_crop_names)}")
        else:
            self.log_box.append(f"Running selected face index(es): {', '.join(str(i) for i in selected_indices)}")
        self._start_wrapper_process(command, "Status: Running selected face(s)...")

    def process_finished(self, exit_code, exit_status):
        self.run_paused = False
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        # Drain any remaining QProcess output that readyRead may not have delivered yet.
        if self.process is not None:
            remaining_out = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            remaining_err = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            if remaining_out:
                self._append_process_log_text(remaining_out)
                self._consume_process_output_lines(remaining_out, is_error=False)
            if remaining_err:
                self._append_process_log_text(remaining_err)
                self._consume_process_output_lines(remaining_err, is_error=True)
        self._flush_process_log_buffer()
        if self._process_stdout_buffer:
            self.update_progress_from_line(self._process_stdout_buffer)
            self._process_stdout_buffer = ""
        if self._process_stderr_buffer:
            self.update_progress_from_line(self._process_stderr_buffer)
            self._process_stderr_buffer = ""
        self.log_box.append(f"Process finished with exit code: {exit_code}")
        elapsed_seconds = (time.time() - self.run_started_at) if self.run_started_at is not None else None
        final_output_path = None

        if exit_code != 0 and self.active_face_preview_index is not None:
            idx = self.active_face_preview_index
            if 0 <= idx < len(self.face_preview_entries):
                self.face_preview_entries[idx]["status"] = "failed"
            self.active_face_preview_index = None
            self.render_face_preview_strip()

        if self.selection_preprocess_mode:
            # Stage 1 of two-step flow (crop detect/preview only).
            self.selection_preprocess_mode = False
            self.suppress_preprocess_ui_until_rephoto = False
            self.set_input_detect_overlay(False)
            self.process = None
            self.set_controls_for_running(False)
            self.run_started_at = None
            self.rephoto_started_at = None

            if exit_code == 0:
                self._prepare_face_selection_after_preprocess()
                return

            self.status_label.setText("Status: Preprocessing failed")
            self.set_preprocess_progress(0, "Preprocess ready")
            self.set_rephoto_progress(0, "Waiting...")
            self.clear_result_stage_overlay()
            self.show_run_summary_dialog(
                success=False,
                exit_code=exit_code,
                elapsed_seconds=elapsed_seconds,
                output_path=None,
            )
            self.current_run_summary_context = None
            return

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
                self._sync_face_preview_crop_paths()
                self.reconcile_face_preview_results(after_epoch=self.run_started_at)
                # Clear stale active index so compare wipe uses selected_face_preview_index.
                self.active_face_preview_index = None
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
                newest = self.find_latest_image(results_root, self.run_started_at)

                if newest is None:
                    from_strip = self.get_selected_face_preview_path()
                    if from_strip is not None:
                        self.set_result_preview_image(from_strip)
                        final_output_path = from_strip
                        self.log_box.append(f"Previewing latest face result: {from_strip.name}")
                    else:
                        self.log_box.append("No new result image was found in the results folder.")
                        self.set_result_preview_image(None)
                else:
                    run_folder = newest.parent
                    if len(self.face_preview_entries) <= 1:
                        _, rephoto_path = self.simplify_run_folder(run_folder)
                        final_preview = rephoto_path or newest
                    else:
                        final_preview = newest
                    selected_face_preview = self.get_selected_face_preview_path()
                    if selected_face_preview is not None:
                        self.set_result_preview_image(selected_face_preview)
                        final_output_path = selected_face_preview
                        self.log_box.append(f"Previewing final face result: {selected_face_preview.name}")
                    else:
                        self.set_result_preview_image(final_preview)
                        final_output_path = final_preview
                        self.log_box.append(f"Previewing final result: {final_preview.name}")

            # Clear overlay after displaying final preview
            self.clear_result_stage_overlay()

        else:
            self.status_label.setText("Status: Backend returned an error")
            self.clear_result_stage_overlay()

        summary_text = self._store_run_summary_text(
            success=(exit_code == 0),
            exit_code=exit_code,
            elapsed_seconds=elapsed_seconds,
            output_path=final_output_path,
        )
        if exit_code != 0 and summary_text:
            self._show_run_summary_text_dialog(summary_text)
        self.selection_preprocess_mode = False
        self.suppress_preprocess_ui_until_rephoto = False
        self.set_input_detect_overlay(False)
        self.current_run_summary_context = None

        # Re-enable face selection if crops still exist so the user can
        # pick additional or different faces for another run.
        has_face_entries = len(self.face_preview_entries) > 1
        if has_face_entries and self.current_crop_output_dir:
            self.awaiting_face_selection = True
            self.set_run_button_continue_mode(True)
            self.render_face_preview_strip()
        else:
            self.awaiting_face_selection = False
            self.set_run_button_continue_mode(False)

        self.process = None
        self.set_controls_for_running(False)
        self.run_started_at = None
        self.rephoto_started_at = None


    def process_error(self, process_error):
        self.run_paused = False
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self._flush_process_log_buffer()
        self._process_stdout_buffer = ""
        self._process_stderr_buffer = ""
        if self.active_face_preview_index is not None:
            idx = self.active_face_preview_index
            if 0 <= idx < len(self.face_preview_entries):
                self.face_preview_entries[idx]["status"] = "failed"
            self.active_face_preview_index = None
            self.render_face_preview_strip()
        self.log_box.append(f"Process launch error: {process_error}")
        self.status_label.setText("Status: Process launch error")
        self.set_input_detect_overlay(False)
        self.suppress_preprocess_ui_until_rephoto = False
        self.clear_result_stage_overlay()
        elapsed_seconds = (time.time() - self.run_started_at) if self.run_started_at is not None else None
        self.show_run_summary_dialog(
            success=False,
            exit_code=-1,
            elapsed_seconds=elapsed_seconds,
            output_path=None,
            launch_error=str(process_error),
        )
        self.selection_preprocess_mode = False
        self.current_run_summary_context = None

        # Re-enable face selection if crops still exist.
        has_face_entries = len(self.face_preview_entries) > 1
        if has_face_entries and self.current_crop_output_dir:
            self.awaiting_face_selection = True
            self.set_run_button_continue_mode(True)
            self.render_face_preview_strip()
        else:
            self.awaiting_face_selection = False
            self.set_run_button_continue_mode(False)

        self.process = None
        self.set_controls_for_running(False)
        self.run_started_at = None
        self.rephoto_started_at = None


    def cancel_run(self):
        if self.process is None:
            if self.awaiting_face_selection:
                self.log_box.append("Face selection step cancelled.")
                self.status_label.setText("Status: Face selection cancelled")
                self.reset_face_preview_state(preserve_input_overlays=True)
                self.current_run_summary_context = None
                return
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
        self.set_input_detect_overlay(False)
        self.suppress_preprocess_ui_until_rephoto = False
        self.clear_result_stage_overlay()
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()

        if self.run_paused:
            self.run_paused = False

        self.process.terminate()
        if not self.process.waitForFinished(2000):
            self.process.kill()

    def _show_run_context_menu(self, pos):
        """Right-click menu on Run button with a 'Preview Crops Only' shortcut."""
        if self.process is not None or self.awaiting_face_selection:
            return
        menu = QMenu(self)
        crop_action = menu.addAction("Preview Crops Only")
        crop_action.setToolTip("Run face detection and cropping without the full rephotography pass.")
        action = menu.exec(self.run_button.mapToGlobal(pos))
        if action is crop_action:
            self._run_crop_only_preview()

    def _run_crop_only_preview(self):
        """Temporarily enable crop-only mode, run, then restore the previous setting."""
        dlg = self.advanced_dialog
        was_crop_only = dlg.crop_only_checkbox.isChecked()
        dlg.crop_only_checkbox.setChecked(True)
        try:
            self.run_wrapper()
        finally:
            # Restore immediately — build_wrapper_command reads this synchronously
            # before QProcess starts, so no race condition.
            dlg.crop_only_checkbox.setChecked(was_crop_only)

    def run_wrapper(self):
        if self.awaiting_face_selection and self.process is None:
            self.continue_rephoto_with_selected_faces()
            return

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

        current_key = self._normalized_path_key(str(input_image_path))
        self.auto_detect_faces_armed_input = None

        # ----------------------------------------------------------
        # Check if we can reuse existing crops from a previous run
        # ----------------------------------------------------------
        reuse_crops = (
            not self.advanced_dialog.crop_only_checkbox.isChecked()
            and self._can_reuse_existing_crops(current_key)
        )

        if reuse_crops:
            quick_count = self.quick_face_count_estimate if isinstance(self.quick_face_count_estimate, int) else None
            crop_dir = Path(self.current_crop_output_dir)
            crop_files = self._list_image_files_in_dir(crop_dir)
            crop_count = len(crop_files)
            is_multi_face = (crop_count > 1) or (quick_count is not None and quick_count > 1)

            self.log_box.append(f"Reusing {crop_count} existing crop(s) — skipping face detection and cropping.")

            if is_multi_face:
                # Multi-face: show the face selection UI using existing crops,
                # then the user clicks Run again to continue_rephoto_with_selected_faces
                self.stop_quick_face_probe()
                self.set_input_detect_overlay(False)
                self.clear_result_stage_overlay()
                self.reset_progress_bars()
                self.current_run_summary_context = self._capture_run_context()
                self.current_run_phase = "preprocess"
                self.selection_preprocess_mode = False
                self.suppress_preprocess_ui_until_rephoto = False
                self._inprocess_preview_crops = False
                self._prepare_face_selection_after_preprocess()
                return

            # Single-face: skip straight to enhancement with -UseExistingCrops
            self.stop_quick_face_probe()
            self.set_input_detect_overlay(False)
            self.clear_result_stage_overlay()
            self.reset_progress_bars()
            self.reset_face_preview_state(preserve_input_overlays=False)
            self.current_run_summary_context = self._capture_run_context()
            self.current_run_phase = "preprocess"
            self.awaiting_face_selection = False
            self.selection_preprocess_mode = False
            self.suppress_preprocess_ui_until_rephoto = False
            self.set_run_button_continue_mode(False)

            self._reset_wrapper_runtime_tracking(preserve_crop_dir=True)
            self._prepare_stop_flag_for_new_run()
            self.set_preprocess_progress(100, "Crops reused")
            self.set_rephoto_progress(0, "Starting...")
            command = self.build_wrapper_command(
                force_use_existing_crops=True,
            )

            self.log_box.append("Run button clicked — reusing existing crops.")
            self._start_wrapper_process(command, "Status: Running enhancement (crops reused)...")
            return

        # ----------------------------------------------------------
        # Re-crop only: same image, but crop expansion changed (multi-face)
        # Face-crop-plus re-detects internally but it's fast; skip the
        # "Detecting Faces" overlay and preserve previous face selections.
        # ----------------------------------------------------------
        recrop_only = (
            not self.advanced_dialog.crop_only_checkbox.isChecked()
            and self.current_crop_output_dir
            and not self._inprocess_preview_crops
            and self._crop_source_input_key == current_key
            and self._crop_source_face_factor is not None
            and abs(self.advanced_dialog.face_factor_edit.value() - self._crop_source_face_factor) > 1e-6
            and len(self.face_preview_entries) > 1
        )

        if recrop_only:
            # Save previous face selections to restore after re-crop
            prev_selected = set(self.get_selected_face_indices())
            self._pending_face_reselection = prev_selected if prev_selected else None

            self.stop_quick_face_probe()
            self.set_input_detect_overlay(False)
            self.clear_result_stage_overlay()
            self.reset_progress_bars()
            self.current_run_summary_context = self._capture_run_context()
            self.current_run_phase = "preprocess"
            self.awaiting_face_selection = False
            self.selection_preprocess_mode = True   # so process_finished routes to selection UI
            self.suppress_preprocess_ui_until_rephoto = False  # no "Detecting Faces" overlay
            self.set_run_button_continue_mode(False)

            self._reset_wrapper_runtime_tracking()  # clear stale crop dir for fresh crops
            self._prepare_stop_flag_for_new_run()
            self.set_preprocess_progress(5, "Re-cropping faces...")
            self.set_rephoto_progress(0, "Waiting...")

            command = self.build_wrapper_command(force_crop_only=True)

            self.log_box.append("Run button clicked — re-cropping with new expansion factor.")
            self._start_wrapper_process(command, "Status: Re-cropping faces...")
            return

        # ----------------------------------------------------------
        # Normal flow: run face detection + cropping from scratch
        # ----------------------------------------------------------
        self.stop_quick_face_probe()
        self.set_input_detect_overlay(False)
        self.clear_result_stage_overlay()
        self.reset_progress_bars()
        self.reset_face_preview_state(preserve_input_overlays=False)
        self.current_run_summary_context = self._capture_run_context()
        self.current_run_phase = "preprocess"
        self.awaiting_face_selection = False
        quick_count = self.quick_face_count_estimate if isinstance(self.quick_face_count_estimate, int) else None
        self.selection_preprocess_mode = bool(
            (not self.advanced_dialog.crop_only_checkbox.isChecked()) and (quick_count != 1)
        )
        self.suppress_preprocess_ui_until_rephoto = bool(self.selection_preprocess_mode)
        if self.selection_preprocess_mode and current_key is not None:
            self.auto_detect_faces_triggered_input = current_key
        self.set_run_button_continue_mode(False)

        self._reset_wrapper_runtime_tracking()
        self._prepare_stop_flag_for_new_run()
        if self.suppress_preprocess_ui_until_rephoto:
            self.set_input_detect_overlay(True, "Detecting Faces")
            self.set_preprocess_progress(0, "Preprocess ready")
            self.set_rephoto_progress(0, "Waiting...")
        else:
            self.set_preprocess_progress(5, "Preprocessing...")
            self.set_rephoto_progress(0, "Waiting...")
        command = self.build_wrapper_command(
            force_crop_only=True if self.selection_preprocess_mode else None,
        )

        self.log_box.append("Run button clicked.")
        status_text = "Status: Detecting faces..." if self.selection_preprocess_mode else "Status: Running backend..."
        self._start_wrapper_process(command, status_text)

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








