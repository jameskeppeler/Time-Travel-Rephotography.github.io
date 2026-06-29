"""Reusable widget classes and worker helpers for the rephotography GUI.

Extracted from gui/app.py as the first step of the Sprint-4 module split.
Nothing here references MainWindow directly — InputDropLabel and
ResultPreviewLabel duck-type their parent_window via hasattr() checks, and
_PreflightRunner treats its owner as an opaque object with a
_collect_preflight_report(...) method. That keeps this module importable
without pulling the rest of app.py.
"""

import math
import random
import re
import time
from pathlib import Path

from PySide6.QtCore import (
    QEvent,
    QObject,
    QRect,
    QRunnable,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QImageReader,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSlider,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QToolTip,
    QWidget,
)


# ============================================================================
# ASYNC IMAGE LOADER (avoids UI freeze on multi-hundred-MB historical scans)
# ============================================================================

class _PixmapLoaderSignals(QObject):
    loaded = Signal(str, QImage)
    failed = Signal(str)


class _PixmapLoader(QRunnable):
    """QImageReader (thread-safe) load of a file, off the UI thread."""

    def __init__(self, path: str, signals: _PixmapLoaderSignals):
        super().__init__()
        self._path = path
        self._signals = signals
        self.setAutoDelete(True)

    def run(self):
        try:
            reader = QImageReader(self._path)
            reader.setAutoTransform(True)
            img = reader.read()
            if img.isNull():
                self._signals.failed.emit(self._path)
            else:
                self._signals.loaded.emit(self._path, img)
        except Exception:
            self._signals.failed.emit(self._path)


# ============================================================================
# TIMESTAMPED LOG WIDGET
# ============================================================================

class TimestampedLogBox(QTextEdit):
    """QTextEdit that auto-prefixes append() output with HH:MM:SS and
    colorizes lines that look like errors or warnings.

    Empty/whitespace-only lines pass through unchanged so they keep working as
    visual separators. Pre-formatted lines that already start with a bracketed
    timestamp (e.g. "[12:34:56] …") are also passed through to avoid double
    stamping when a caller assembles its own header.
    """

    _TS_PREFIX_RE = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}\]")
    # Case-insensitive cues used to decide color level. Order matters:
    # "error" takes priority over "warning" if both appear in the same line.
    _ERROR_CUES_RE = re.compile(
        r"(?i)\b(error|failed|failure|exception|traceback|cannot|crashed)\b|could not"
    )
    _WARN_CUES_RE = re.compile(
        r"(?i)\b(warning|warn|timed out|deprecated|fallback)\b"
    )
    _ERROR_COLOR = "#eb5757"
    _WARN_COLOR = "#f2c94c"

    def _classify(self, s: str) -> str:
        if self._ERROR_CUES_RE.search(s):
            return "error"
        if self._WARN_CUES_RE.search(s):
            return "warn"
        return "info"

    def append(self, text):
        s = text if isinstance(text, str) else str(text)
        if s.strip() == "":
            super().append(s)
            return
        if self._TS_PREFIX_RE.match(s):
            payload = s
        else:
            ts = time.strftime("%H:%M:%S")
            payload = f"[{ts}] {s}"

        level = self._classify(payload)
        if level == "info":
            super().append(payload)
            return

        color = self._ERROR_COLOR if level == "error" else self._WARN_COLOR
        # Escape minimal HTML special chars so user text doesn't break markup.
        safe = (
            payload.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        super().append(f'<span style="color:{color}">{safe}</span>')


# ============================================================================
# ASYNC PREFLIGHT (nvidia-smi + filesystem checks moved off the UI thread)
# ============================================================================

class _PreflightSignals(QObject):
    finished = Signal(dict)


class _PreflightRunner(QRunnable):
    """Runs MainWindow._collect_preflight_report() off the UI thread.

    The collection function is read-only against the rest of the window
    (paths, hardware probes, write test). The one widget read it performs
    (results_root_edit.text) happens before this runnable is scheduled.
    """

    def __init__(self, owner, signals: _PreflightSignals, results_root_text: str):
        super().__init__()
        self._owner = owner
        self._signals = signals
        self._results_root_text = results_root_text
        self.setAutoDelete(True)

    def run(self):
        try:
            report = self._owner._collect_preflight_report(
                results_root_override=self._results_root_text
            )
        except Exception as e:
            report = {
                "checks": [{
                    "name": "Preflight runner",
                    "status": "fail",
                    "detail": f"Preflight execution failed: {e}",
                    "fix": "Report this error; the GUI will still function."
                }],
                "fail": 1,
                "warn": 0,
                "pass": 0,
            }
        self._signals.finished.emit(report)


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


class FilmstripContainerWidget(QWidget):
    """Widget that paints sprocket holes and decorative borders for the filmstrip.

    Supports both horizontal (default) and vertical orientation.  When
    ``show_empty_frames`` is True, empty square frames are painted across
    the full extent of the strip so it looks like continuous unexposed film.

    NOTE: Empty frames are painted across the full visible area. When populated
    with face buttons, the scroll area handles scrolling naturally — we only
    paint decorative borders, not frame boundaries, so the sprocket edges stay
    visually consistent.
    """

    SPROCKET_BAND = 12
    FILM_BASE     = "#22272e"
    SPROCKET_BG   = "#1c2028"
    HOLE_COLOR    = "#2e343d"
    FRAME_BG      = "#191d23"
    FRAME_BORDER  = "#2a3038"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_empty_frames = True
        self.vertical = False  # True when filmstrip runs top-to-bottom

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        band = self.SPROCKET_BAND

        p.fillRect(0, 0, w, h, QColor(self.FILM_BASE))

        if self.vertical:
            self._paint_vertical(p, w, h, band)
        else:
            self._paint_horizontal(p, w, h, band)

        p.end()

    # -- horizontal (sprockets top/bottom, frames left→right) ---------------

    def _paint_horizontal(self, p, w, h, band):
        p.fillRect(0, 0, w, band, QColor(self.SPROCKET_BG))
        p.fillRect(0, h - band, w, band, QColor(self.SPROCKET_BG))

        hole_w, hole_h, radius, spacing = 8, 5, 1.5, 20
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(self.HOLE_COLOR)))
        x = 6
        while x < w:
            p.drawRoundedRect(x, (band - hole_h) // 2, hole_w, hole_h, radius, radius)
            p.drawRoundedRect(x, h - band + (band - hole_h) // 2, hole_w, hole_h, radius, radius)
            x += spacing

        if self.show_empty_frames:
            frame_size = h - band * 2  # square
            gap = 4
            p.setPen(QPen(QColor(self.FRAME_BORDER), 1))
            p.setBrush(QBrush(QColor(self.FRAME_BG)))
            fx = 0
            while fx < w:
                p.drawRect(fx, band, frame_size, frame_size)
                fx += frame_size + gap

    # -- vertical (sprockets left/right, frames top→bottom) -----------------

    def _paint_vertical(self, p, w, h, band):
        p.fillRect(0, 0, band, h, QColor(self.SPROCKET_BG))
        p.fillRect(w - band, 0, band, h, QColor(self.SPROCKET_BG))

        hole_w, hole_h, radius, spacing = 5, 8, 1.5, 20
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(self.HOLE_COLOR)))
        y = 6
        while y < h:
            p.drawRoundedRect((band - hole_w) // 2, y, hole_w, hole_h, radius, radius)
            p.drawRoundedRect(w - band + (band - hole_w) // 2, y, hole_w, hole_h, radius, radius)
            y += spacing

        if self.show_empty_frames:
            frame_size = w - band * 2  # square
            gap = 4
            p.setPen(QPen(QColor(self.FRAME_BORDER), 1))
            p.setBrush(QBrush(QColor(self.FRAME_BG)))
            fy = 0
            while fy < h:
                p.drawRect(band, fy, frame_size, frame_size)
                fy += frame_size + gap


class FaceStripToolButton(QToolButton):
    """Filmstrip frame button that exposes lightweight hover callbacks."""

    _stylesheet_cache = {}
    _currently_hovered = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.hover_enter_callback = None
        self.hover_leave_callback = None

    @staticmethod
    def get_stylesheet(border_color, bg_color, text_color, hover_color):
        key = (border_color, bg_color, text_color, hover_color)
        if key not in FaceStripToolButton._stylesheet_cache:
            FaceStripToolButton._stylesheet_cache[key] = (
                "QToolButton {"
                f" border: 1px solid {border_color};"
                f" background-color: {bg_color}; color: {text_color}; padding: 2px; }}"
                f"QToolButton:hover {{ background-color: {hover_color};"
                f" border: 1px solid {border_color}; }}"
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


__all__ = [
    "_PixmapLoaderSignals",
    "_PixmapLoader",
    "TimestampedLogBox",
    "_PreflightSignals",
    "_PreflightRunner",
    "InstantToolButton",
    "NoScrollComboBox",
    "NoScrollDoubleSpinBox",
    "NoScrollSpinBox",
    "NoScrollSlider",
    "FilmstripContainerWidget",
    "FaceStripToolButton",
    "InputDropLabel",
    "ResultPreviewLabel",
    "InputDetectOverlay",
]
