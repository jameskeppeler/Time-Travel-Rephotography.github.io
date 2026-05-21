"""Preview controller mixin for the rephotography GUI.

Sprint-4 architectural slice: ~750 lines of input/result preview state
machines, compare-wipe compositing, async pixmap callbacks, face-box
overlay, and stage-overlay animation pulled out of MainWindow into a
mixin class. Methods are unchanged — only their physical location moves.
MainWindow inherits from PreviewMixin (and FaceStripMixin, QMainWindow),
so every existing self.<method>() call site continues to work via MRO.

Relies on the following MainWindow instance attributes (all set up in
MainWindow.__init__):

  * Source pixmaps: input_pixmap, result_pixmap
  * Result-preview cache: result_preview_pixmap_cache,
    result_preview_pixmap_cache_max_entries, last_result_image_path,
    last_result_image_cache_key
  * Scale caches: input_preview_scaled_cache_key/pixmap,
    input_preview_render_cache_key/pixmap, input_preview_last_display_key,
    result_preview_scaled_cache_key/pixmap, result_preview_last_display_key
  * Compare-wipe: _compare_wipe_active, _compare_wipe_last_pos,
    _compare_wipe_result_scaled / _compare_wipe_input_scaled (+ their keys)
  * Face-box overlay: input_face_boxes, input_face_box_source,
    hover_face_box_override, hover_face_box_cache, hover_face_preview_index,
    hover_face_preview_source, face_box_debug_overlay_enabled
  * Widgets: input_preview_label, result_preview_label, input_detect_overlay,
    result_stage_overlay_label, _result_stage_timer
  * Other: face_preview_entries, current_run_phase, _wide_layout_active,
    _user_inspecting_completed_face, _result_stage_*, repo_root,
    current_gfpgan_output_dir
"""

from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QImageReader, QPainter, QPen, QPixmap

from gui.constants import IMAGE_EXTENSIONS
from gui.widgets import _PixmapLoader


class PreviewController:
    """Controller that owns the input/result preview subsystem.

    Promoted from PreviewMixin in the third Sprint-4 polish round.
    MainWindow owns it as ``self.preview = PreviewController(self)``.

    Method bodies are unchanged from the mixin era; reads/writes of
    ``self.input_pixmap``, ``self.result_preview_label``, etc. fall
    through to MainWindow via __getattr__ / __setattr__ until a future
    slice migrates state ownership to the controller.
    """

    def __init__(self, window):
        from collections import OrderedDict
        self._window = window
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
        self._compare_wipe_active = False
        self._compare_wipe_last_pos = None
        self._compare_wipe_result_scaled = None
        self._compare_wipe_result_scaled_key = None
        self._compare_wipe_input_scaled = None
        self._compare_wipe_input_scaled_key = None
        # Async pixmap loader for input previews. Off-UI-thread
        # decode keeps the window responsive on 50-200 MB historical scans.
        from PySide6.QtCore import QThreadPool
        from gui.widgets import _PixmapLoaderSignals
        self._pixmap_thread_pool = QThreadPool.globalInstance()
        self._input_pixmap_loader_signals = _PixmapLoaderSignals()
        self._input_pixmap_loader_signals.loaded.connect(self._on_input_pixmap_loaded)
        self._input_pixmap_loader_signals.failed.connect(self._on_input_pixmap_failed)
        self._input_pixmap_loader_path = None
        self.current_wide_preview_side = 360

    def __getattr__(self, name):
        if name.startswith("_PreviewController__") or name == "_window":
            raise AttributeError(name)
        return getattr(self._window, name)
    """[Legacy docstring from the mixin era — see class header above]"""

    def _update_wide_preview_dimensions(self):
        """Track the current preview label width for face strip card sizing."""
        if not getattr(self, "_wide_layout_active", False):
            return
        self.current_wide_preview_side = max(200, self.input_preview_label.width())

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

    def handle_input_preview_mouse_move(self, pos):
        if not self.face_strip.face_preview_entries:
            return
        idx = self._hit_test_input_face_index(pos)
        if isinstance(idx, int) and self._is_face_interaction_allowed(idx):
            self.face_strip.set_hover_face_preview_index(idx, source="input")
            return
        if self.face_strip.hover_face_preview_source == "input":
            self.face_strip.clear_hover_face_preview_index()

    def handle_input_preview_mouse_leave(self):
        if self.face_strip.hover_face_preview_source == "input":
            self.face_strip.clear_hover_face_preview_index()

    def handle_input_preview_click(self, pos):
        if not self.face_strip.face_preview_entries:
            return False
        idx = self._hit_test_input_face_index(pos)
        if (not isinstance(idx, int)) or (not self._is_face_interaction_allowed(idx)):
            return False
        self.face_strip.set_hover_face_preview_index(idx, source="input")
        self.face_strip.select_face_preview(idx, user_initiated=True)
        return True

    def _get_compare_before_source_pixmap(self):
        focused_idx = self.face_strip._get_focused_face_preview_index()
        crop_path = self.face_strip._get_face_crop_path(focused_idx) if focused_idx is not None else None
        if crop_path is None and focused_idx is not None:
            self.face_strip._sync_face_preview_crop_paths()
            crop_path = self.face_strip._get_face_crop_path(focused_idx)
        if crop_path is not None:
            pix = self._get_result_pixmap_cached(crop_path)
            if pix is not None and (not pix.isNull()):
                return pix
        return self.input_pixmap

    def _resolve_enhanced_preview_for_crop(self, crop_path: Path):
        if crop_path is None:
            return None

        # If the input IS already a blended face, return it directly
        stem = crop_path.stem
        if stem.endswith("_blend") and crop_path.exists():
            return crop_path

        if not self.pipeline.current_blended_faces_dir:
            return None
        blended_dir = Path(self.pipeline.current_blended_faces_dir)
        if not blended_dir.exists():
            return None

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

    def update_input_face_boxes_for_preview(self, expected_count=None):
        self.input_face_boxes = []
        self.input_face_box_source = None
        self.face_strip.hover_face_box_cache = {}
        self.face_strip.hover_face_box_override = None
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
        self.face_strip.hover_face_box_override = None
        self.face_strip.hover_face_preview_source = None
        self.face_strip.hover_face_box_cache = {}
        self.input_preview_scaled_cache_key = None
        self.input_preview_scaled_cache_pixmap = None
        self.input_preview_render_cache_key = None
        self.input_preview_render_cache_pixmap = None
        self.input_preview_last_display_key = None
        if image_path is None or (not image_path.exists()):
            self._input_pixmap_loader_path = None
            self.input_face_boxes = []
            self.input_face_box_source = None
            self.set_input_detect_overlay(False)
            self.input_preview_label.setText("No input image yet.")
            self.input_preview_label.setPixmap(QPixmap())
            return

        # Decode off the UI thread. Track the in-flight path so a late
        # callback for a previous load can be discarded.
        path_str = str(image_path)
        self._input_pixmap_loader_path = path_str
        self.input_preview_label.setText("Loading preview…")
        self.input_preview_label.setPixmap(QPixmap())
        self._pixmap_thread_pool.start(_PixmapLoader(path_str, self._input_pixmap_loader_signals))

    def _on_input_pixmap_loaded(self, path: str, image: QImage):
        if path != self._input_pixmap_loader_path:
            return  # stale — superseded by a newer load
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            self._on_input_pixmap_failed(path)
            return
        self.input_pixmap = pix
        self.input_preview_label.setText("")
        self.update_input_face_boxes_for_preview(
            expected_count=len(self.face_strip.face_preview_entries) if self.face_strip.face_preview_entries else None
        )
        if hasattr(self, "input_detect_overlay") and self.input_detect_overlay.isVisible():
            self.input_detect_overlay.notify_parent_pixmap_changed()

    def _on_input_pixmap_failed(self, path: str):
        if path != self._input_pixmap_loader_path:
            return
        self.input_face_boxes = []
        self.input_face_box_source = None
        self.set_input_detect_overlay(False)
        self.input_preview_label.setText("Could not load input image.")
        self.input_preview_label.setPixmap(QPixmap())

    def _result_preview_cache_key(self, image_path: Path):
        return path_utils.result_preview_cache_key(image_path)

    def _get_result_pixmap_cached(self, image_path: Path):
        key = self._result_preview_cache_key(image_path)
        cache = self.result_preview_pixmap_cache
        cached = cache.get(key)
        if cached is not None and (not cached.isNull()):
            cache.move_to_end(key)
            return cached
        if cached is not None:
            # Cached entry is null/invalid; drop it before reloading.
            cache.pop(key, None)

        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull():
            return None
        pix = QPixmap.fromImage(img)
        if pix.isNull():
            return None

        cache[key] = pix
        while len(cache) > int(self.result_preview_pixmap_cache_max_entries):
            cache.popitem(last=False)
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

        active_idx = self.face_strip.hover_face_preview_index
        if isinstance(active_idx, int) and self.face_strip.hover_face_preview_source == "strip":
            cursor_idx = self.face_strip._cursor_face_preview_index()
            if isinstance(cursor_idx, int):
                if cursor_idx != active_idx:
                    self.face_strip.hover_face_preview_index = cursor_idx
                    self.face_strip.hover_face_preview_source = "strip"
                active_idx = cursor_idx
            else:
                self.face_strip.hover_face_preview_index = None
                self.face_strip.hover_face_preview_source = None
                self.face_strip.hover_face_box_override = None
                active_idx = None

        active_box = self.face_strip.hover_face_box_override
        if active_box is None and isinstance(active_idx, int):
            cached_box = self.face_strip.hover_face_box_cache.get(active_idx)
            if cached_box is None:
                cached_box = self.resolve_hover_face_box(active_idx)
                self.face_strip.hover_face_box_cache[active_idx] = cached_box
            active_box = cached_box

        if self.pipeline.awaiting_face_selection:
            selected_indices = [e["index"] for e in self.face_strip.face_preview_entries if e.get("selected", False)]
        else:
            selected_indices = []
            if isinstance(self.face_strip.selected_face_preview_index, int) and self.face_strip.selected_face_preview_index >= 0:
                selected_indices = [self.face_strip.selected_face_preview_index]

        selected_boxes = []
        for idx in selected_indices:
            cached_box = self.face_strip.hover_face_box_cache.get(idx)
            if cached_box is None:
                cached_box = self.resolve_hover_face_box(idx)
                self.face_strip.hover_face_box_cache[idx] = cached_box
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

        # Scale both to the same display size; reuse cached scaled pixmaps
        # when source + target size are unchanged so the mouse-move loop is
        # mostly composite work, not rescaling.
        try:
            result_key = (int(self.result_pixmap.cacheKey()), lw, lh)
        except Exception:
            result_key = None
        if (
            result_key is not None
            and self._compare_wipe_result_scaled_key == result_key
            and self._compare_wipe_result_scaled is not None
            and not self._compare_wipe_result_scaled.isNull()
        ):
            result_scaled = self._compare_wipe_result_scaled
        else:
            result_scaled = self.result_pixmap.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._compare_wipe_result_scaled_key = result_key
            self._compare_wipe_result_scaled = result_scaled
        sw, sh = result_scaled.width(), result_scaled.height()
        if sw <= 0 or sh <= 0:
            return
        before_pixmap = self._get_compare_before_source_pixmap()
        if before_pixmap is None:
            return
        try:
            input_key = (int(before_pixmap.cacheKey()), sw, sh)
        except Exception:
            input_key = None
        if (
            input_key is not None
            and self._compare_wipe_input_scaled_key == input_key
            and self._compare_wipe_input_scaled is not None
            and not self._compare_wipe_input_scaled.isNull()
        ):
            input_scaled = self._compare_wipe_input_scaled
        else:
            input_scaled = before_pixmap.scaled(
                sw, sh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            self._compare_wipe_input_scaled_key = input_key
            self._compare_wipe_input_scaled = input_scaled
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
        if self.pipeline.process is None:
            return
        if base:
            self.set_result_stage_overlay(base)
            return
        if self.pipeline.current_run_phase == "rephoto":
            self.set_result_stage_overlay("Rephotographing")

    def update_result_stage_overlay_animation(self):
        """Update the animated dots on the stage overlay."""
        if not self.result_stage_base_text:
            return
        self.result_stage_dot_count = (self.result_stage_dot_count % 3) + 1
        text = f"{self.result_stage_base_text}{'.' * self.result_stage_dot_count}"
        self.result_stage_overlay.setText(text)
        self.position_result_stage_overlay()

    def preview_stage_image_if_found(self, image_path, stage_name):
        """Preview an image if it exists and update stage overlay."""
        if image_path is not None and image_path.exists():
            self.set_result_preview_image(image_path)
            self.set_result_stage_overlay(stage_name)

