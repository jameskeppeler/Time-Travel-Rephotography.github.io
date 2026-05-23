"""Face-strip controller mixin for the rephotography GUI.

Sprint-4 architectural slice: ~1,100 lines of face-selection filmstrip
state and rendering pulled out of MainWindow into a mixin class. The
methods are unchanged — only their physical location moves. MainWindow
inherits from both QMainWindow and FaceStripMixin, so every existing
self.<method>() call site continues to work via the normal MRO.

Why a mixin and not a controller object:
  * Zero call-site churn — 80+ `self.render_face_preview_strip()` and
    `self.face_preview_entries[i]` references stay verbatim.
  * The methods are deeply intertwined with MainWindow instance state
    (~25 attributes); threading a controller reference through every call
    would obscure the change.

Future work can promote individual methods to take a window reference
explicitly and migrate to a controller; the mixin is the safe staging
ground.
"""

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui import path_utils
from gui.constants import FACE_SUFFIX_INDEX_RE, IMAGE_EXTENSIONS
from gui.widgets import FaceStripToolButton, FilmstripContainerWidget


class FaceStripController:
    """Controller that owns the face-strip subsystem.

    Promoted from FaceStripMixin in the second Sprint-4 polish round.
    Instead of being mixed into MainWindow via MRO, it is now owned by
    MainWindow as ``self.face_strip = FaceStripController(self)``.

    Methods are unchanged from the mixin era; they still access
    ``self.face_preview_entries``, ``self.log_box``, etc.  Those reads
    fall through to ``self._window`` via __getattr__ until a future
    slice migrates the state to the controller. Assignments
    (``self.face_preview_entries = []``) likewise go to the window via
    __setattr__, so MainWindow's existing _init_state continues to be
    the single source of truth for these attributes.

    The win at this step is structural: MainWindow no longer inherits
    from FaceStripMixin, and the face-strip surface is reachable through
    an explicit attribute rather than the leftmost mixin in the bases
    tuple.
    """

    def __init__(self, window):
        self._window = window
        # State owned by this controller.
        self.face_preview_entries = []
        self.active_face_preview_index = None
        self.selected_face_preview_index = None
        self._user_inspecting_completed_face = False
        self.hover_face_preview_index = None
        self.hover_face_preview_source = None
        self.hover_face_box_override = None
        self.hover_face_box_cache = {}
        self.face_preview_thumb_icon_cache = {}
        self.face_preview_thumb_icon_cache_max_entries = 256
        self._face_strip_render_signature = None
        self._no_faces_detected = False
        self._pending_face_reselection = None

    def __getattr__(self, name):
        # Window-owned widgets and any not-yet-migrated state.
        if name.startswith("_FaceStripController__") or name == "_window":
            raise AttributeError(name)
        return getattr(self._window, name)
    """[Legacy docstring from the mixin era — see class header above]

    Relies on the following instance attributes existing on the host class
    (all of them are set up inside MainWindow.__init__ today):
        face_preview_entries, face_preview_panel, face_preview_strip,
        face_preview_strip_filmstrip, face_preview_strip_layout,
        face_preview_header, face_preview_summary_label,
        face_preview_thumb_icon_cache, face_preview_thumb_icon_cache_max_entries,
        face_preview_scroll, face_selection_notice_label,
        face_select_all_button, face_select_none_button,
        face_preview_auto_follow_checkbox, active_face_preview_index,
        selected_face_preview_index, awaiting_face_selection,
        _face_strip_render_signature, _no_faces_detected,
        _wide_layout_active, _user_inspecting_completed_face,
        result_preview_path_before_hover, last_result_image_path,
        last_result_image_cache_key, current_crop_output_dir,
        quick_face_count_estimate, quick_face_count_armed_input,
        quick_face_probe_input_key, _haar_face_detector,
        repo_root, input_image_edit, run_button, log_box, status_label,
        process, advanced_dialog
    plus access to the methods left on MainWindow (set_result_preview_image,
    set_input_preview_image, set_input_detect_overlay, etc.).
    """

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

        filmstrip_h = 142
        band = FilmstripContainerWidget.SPROCKET_BAND
        # Reset cross-axis container constraints from the previous mode
        self.face_preview_strip_container.setMaximumWidth(16777215)
        self.face_preview_strip_container.setMaximumHeight(16777215)

        if use_wide_layout:
            base_card_w = self._get_face_strip_card_width(True)
            # Filmstrip width = card + sprocket bands on each side.
            # Painted frame_size = filmstrip_w - 2*band = base_card_w.
            # Keep the rail wide enough for the header text row as well.
            min_panel_w = self._get_face_preview_header_min_width()
            filmstrip_w = max(base_card_w + 2 * band, min_panel_w)
            panel_w = filmstrip_w
            self.face_preview_strip_layout.setDirection(QBoxLayout.TopToBottom)
            self.face_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.face_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.face_preview_strip_filmstrip.setMinimumHeight(0)
            self.face_preview_strip_filmstrip.setMaximumHeight(16777215)
            self.face_preview_strip_filmstrip.setFixedWidth(filmstrip_w)
            self.face_preview_strip_filmstrip.vertical = True
            self.filmstrip_inner_layout.setContentsMargins(band, 0, band, 0)
            self.face_preview_panel.setMinimumWidth(panel_w)
            self.face_preview_panel.setMaximumWidth(panel_w)
            self.face_preview_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.face_preview_auto_follow_checkbox.setText("Follow")
        else:
            self.face_preview_strip_layout.setDirection(QBoxLayout.LeftToRight)
            self.face_preview_strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.face_preview_strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.face_preview_strip_filmstrip.setMinimumHeight(filmstrip_h)
            self.face_preview_strip_filmstrip.setMaximumHeight(filmstrip_h)
            self.face_preview_strip_filmstrip.setMinimumWidth(0)
            self.face_preview_strip_filmstrip.setMaximumWidth(16777215)
            self.face_preview_strip_filmstrip.vertical = False
            self.filmstrip_inner_layout.setContentsMargins(0, band, 0, band)
            self.face_preview_panel.setMinimumWidth(0)
            self.face_preview_panel.setMaximumWidth(16777215)
            self.face_preview_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.face_preview_auto_follow_checkbox.setText("Auto-follow latest")
        # Invalidate render signature to force re-render of face buttons with new dimensions
        self._face_strip_render_signature = None

    def _get_face_preview_header_min_width(self):
        if not hasattr(self, "face_preview_header"):
            return 160
        layout = self.face_preview_header.layout()
        spacing = layout.spacing() if layout is not None else 5
        margin_lr = 0
        if layout is not None:
            margins = layout.contentsMargins()
            margin_lr = margins.left() + margins.right()
        fm = self.face_preview_summary_label.fontMetrics()

        # Account for both header states: summary+follow and selected+all+none.
        summary_follow = fm.horizontalAdvance("Faces 999 | Done 999")
        selected_summary = fm.horizontalAdvance("Selected: 99/99")
        follow_w = self.face_preview_auto_follow_checkbox.sizeHint().width()
        all_w = self.face_select_all_button.sizeHint().width()
        none_w = self.face_select_none_button.sizeHint().width()
        follow_state_w = summary_follow + spacing + follow_w
        selection_state_w = selected_summary + spacing + all_w + spacing + none_w
        content_w = max(follow_state_w, selection_state_w)
        return int(content_w + margin_lr + 6)

    def _get_face_strip_card_width(self, wide_mode):
        if not wide_mode:
            return 116
        side = max(240, int(getattr(self, "current_wide_preview_side", 360)))
        # Wide mode panel must be wide enough for the header row ("Faces N | Done" + checkbox).
        # panel_w = card_w + 2*SPROCKET_BAND(12) = card_w + 24, so card_w >= 136 → panel >= 160px.
        return max(136, min(156, int(round(side * 0.35))))

    def clear_face_preview_strip_layout(self):
        self._face_strip_render_signature = None
        if self.hover_face_preview_index is not None:
            self.hover_face_preview_index = None
            self.hover_face_preview_source = None
            self.preview.refresh_input_preview_scale()
        while self.face_preview_strip_layout.count() > 0:
            item = self.face_preview_strip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def reset_face_preview_state(self, preserve_input_overlays=False):
        self._no_faces_detected = False
        self.face_preview_entries = []
        self.active_face_preview_index = None
        self.selected_face_preview_index = None
        self.hover_face_preview_index = None
        self.hover_face_preview_source = None
        self.hover_face_box_override = None
        self.hover_face_box_cache = {}
        self.pipeline.awaiting_face_selection = False
        self.face_preview_summary_label.setText("Faces: none")
        self.clear_face_preview_strip_layout()
        # Keep filmstrip visible even when reset (shows empty state)
        self.face_preview_header.setVisible(True)
        self.face_preview_strip_filmstrip.setVisible(True)
        if hasattr(self, "face_preview_panel"):
            self.face_preview_panel.setVisible(True)
        self.pipeline.set_run_button_continue_mode(False)
        self._face_strip_render_signature = None

        if not preserve_input_overlays:
            self.preview.input_face_boxes = []
            self.preview.input_face_box_source = None
            self.preview.input_preview_render_cache_key = None
            self.preview.input_preview_render_cache_pixmap = None
            self.preview.input_preview_last_display_key = None
            self.preview.refresh_input_preview_scale()
        self.update_runtime_label()
        self.update_image_import_controls()

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

        if self.pipeline.process is not None:
            if self.pipeline.run_paused:
                self.run_button.setText("Resume")
                self.run_button.setToolTip("Resume the paused backend run.")
            else:
                self.run_button.setText("Pause")
                self.run_button.setToolTip("Pause the current backend run.")
            self.run_button.setEnabled(True)
            return

        if self.pipeline.current_run_phase in {"preprocess", "rephoto"}:
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
        # - If quick scan found no faces (==0), keep "Run" enabled but explain likely failure.
        # - Otherwise (multi-face or unknown estimate), signal the detect/select step.
        quick_count = self.pipeline.quick_face_count_estimate if isinstance(self.pipeline.quick_face_count_estimate, int) else None
        if quick_count == 1:
            self.run_button.setText("Run")
            self.run_button.setEnabled(True)
            self.run_button.setToolTip("Single face detected. Run the full rephotography workflow.")
        elif quick_count == 0:
            self.run_button.setText("Run")
            self.run_button.setEnabled(True)
            self.run_button.setToolTip(
                "Quick scan found no faces. Run will auto-lower detection threshold; if needed, use a tighter face crop."
            )
        else:
            self.run_button.setText("Run")
            self.run_button.setEnabled(True)
            self.run_button.setToolTip(
                "Running starts with face detection, then prompts face selection when multiple faces are found."
            )

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

    def _sync_face_preview_crop_paths(self):
        if not self.face_preview_entries:
            return
        crop_files = []
        if self.pipeline.current_crop_output_dir:
            crop_files = self._list_image_files_in_dir(Path(self.pipeline.current_crop_output_dir))

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
        collisions = []
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
            else:
                collisions.append((idx, p, files_by_index[idx]))

        if collisions:
            # Two crops resolved to the same index — likely a filename clash
            # (e.g. mixed runs in the same dir). Surface it; the second file
            # would otherwise be silently dropped and the wrong thumbnail
            # would render for that slot.
            sample = collisions[0]
            self.log_box.append(
                f"Warning: {len(collisions)} face crop(s) collided on index "
                f"during preview refresh — keeping first match. "
                f"Example: index {sample[0]} kept '{sample[2].name}', "
                f"dropped '{sample[1].name}'."
            )

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
        self._no_faces_detected = False
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
                    "selected": True,  # may flip in _annotate_entries_with_box_sizes
                    "relative_size": None,  # 0..1 sqrt(box_area / image_area)
                }
            )

        self.active_face_preview_index = None
        self.selected_face_preview_index = 0 if self.face_preview_entries else None
        self._sync_face_preview_crop_paths()
        self.preview.update_input_face_boxes_for_preview(expected_count=expected_count)
        self._annotate_entries_with_box_sizes()
        self.update_runtime_label()
        self.render_face_preview_strip()

        # Show re-detect button if faces were found
        if hasattr(self, "redetect_faces_button"):
            self.redetect_faces_button.setVisible(bool(self.face_preview_entries))
            self.redetect_faces_button.setEnabled(bool(self.face_preview_entries))

    # Auto-deselect kicks in when the largest box is at least this many
    # times the next-largest. 1.2x catches obvious singletons-with-junk;
    # finer dominance still triggers if absolute sizes diverge enough.
    _SIZE_DOMINANCE_RATIO = 1.2

    def _annotate_entries_with_box_sizes(self):
        """Compute a relative-size proxy per entry from the input boxes
        and, when multiple faces are detected with clear size dominance,
        default-deselect the small ones so the user doesn't burn an
        optimization run on a false-positive crop.

        The size proxy is sqrt(box_area / image_area) -- roughly the
        fraction of the image diagonal that the face span occupies. A
        real portrait subject typically scores 0.3+; frame edges,
        smudges, and mat-corner false positives usually score below 0.15.

        Every call ALSO logs the per-face sizes (or the reason no size
        could be computed) so the heuristic's behaviour is debuggable
        from the run log without having to re-run with extra
        instrumentation.
        """
        boxes = list(self.preview.input_face_boxes or [])
        if not boxes:
            self.log_box.append(
                "Warning: face-size annotation skipped (no input boxes available; "
                "auto-deselect cannot run)."
            )
            return
        pix = self.preview.input_pixmap
        if pix is None or pix.isNull():
            self.log_box.append(
                "Warning: face-size annotation skipped (input pixmap not loaded; "
                "auto-deselect cannot run)."
            )
            return
        img_w = max(1, int(pix.width()))
        img_h = max(1, int(pix.height()))
        img_area = float(img_w) * float(img_h)
        if img_area <= 0:
            return

        # Map boxes to entries by positional index (the boxes list is
        # sorted to mirror the cropper's detection order, same as the
        # entry indices).
        for entry in self.face_preview_entries:
            idx = entry.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(boxes):
                continue
            try:
                _, _, bw, bh = boxes[idx]
            except (TypeError, ValueError):
                continue
            try:
                area = max(0.0, float(bw) * float(bh))
            except (TypeError, ValueError):
                continue
            entry["relative_size"] = min(1.0, (area / img_area) ** 0.5)

        # Diagnostic log line so the heuristic is debuggable from the run
        # output. Surfaces every face's size even when no action is taken.
        size_parts = []
        for e in self.face_preview_entries:
            label = f"#{int(e.get('index', -1)) + 1}"
            rs = e.get("relative_size")
            if isinstance(rs, float):
                size_parts.append(f"{label}={int(round(rs * 100))}%")
            else:
                size_parts.append(f"{label}=?")
        self.log_box.append(f"Face sizes: {', '.join(size_parts)}")

        # Auto-deselect smaller faces ONLY when at least 2 faces have
        # sizes recorded and the largest is meaningfully larger than the
        # rest. The most-likely-real face stays selected; the user can
        # re-enable any others with a click (or via the right-click
        # context menu).
        sized = [
            (e["index"], e["relative_size"])
            for e in self.face_preview_entries
            if isinstance(e.get("relative_size"), float)
        ]
        if len(sized) < 2:
            return
        sized.sort(key=lambda t: -t[1])
        top_size = sized[0][1]
        runner_up = sized[1][1]
        if top_size <= 0:
            return
        if runner_up > 0 and top_size < self._SIZE_DOMINANCE_RATIO * runner_up:
            self.log_box.append(
                f"Face sizes too close ({int(round(top_size*100))}% vs "
                f"{int(round(runner_up*100))}%) for auto-deselect; "
                f"keeping all selected."
            )
            return
        top_idx = sized[0][0]
        deselected = 0
        for entry in self.face_preview_entries:
            if entry.get("index") != top_idx and entry.get("relative_size") is not None:
                if entry.get("selected"):
                    entry["selected"] = False
                    deselected += 1
        if deselected:
            self.log_box.append(
                f"Auto-deselected {deselected} small face crop(s) below the "
                f"largest. Click a card to re-include it."
            )

    def _make_face_thumb_icon(self, image_path, fallback_text, muted=False, thumb_size=84, relative_size=None):
        thumb_size = max(56, int(thumb_size))
        # Round to 2 decimals so the cache key doesn't churn on tiny variations.
        size_key = (round(float(relative_size), 2)
                    if isinstance(relative_size, float) else None)
        cache_key = self._face_thumb_icon_cache_key(
            image_path=image_path,
            fallback_text=fallback_text,
            muted=muted,
            thumb_size=thumb_size,
        ) + (size_key,)
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
            # Draw face number overlay in bottom-right corner
            if fallback_text:
                num_font = QFont("Segoe UI", 8, QFont.Bold)
                painter.setFont(num_font)
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(fallback_text) + 6
                th = fm.height() + 2
                nr_x = thumb_size - tw - 2
                nr_y = thumb_size - th - 2
                painter.fillRect(nr_x, nr_y, tw, th, QColor(0, 0, 0, 160))
                painter.setPen(QColor("#d0d4dc"))
                painter.drawText(nr_x, nr_y, tw, th, Qt.AlignCenter, fallback_text)
            # Draw relative-size badge in top-left corner if available.
            # Green when likely a real subject, yellow in the middle, red
            # when the face span is suspiciously small.
            if isinstance(relative_size, float):
                pct = max(0, min(100, int(round(relative_size * 100))))
                size_text = f"{pct}%"
                size_font = QFont("Segoe UI", 7, QFont.Bold)
                painter.setFont(size_font)
                fm = painter.fontMetrics()
                stw = fm.horizontalAdvance(size_text) + 6
                sth = fm.height() + 2
                if pct >= 30:
                    badge_bg = QColor(70, 130, 90, 200)   # green
                elif pct >= 15:
                    badge_bg = QColor(140, 110, 40, 200)  # yellow
                else:
                    badge_bg = QColor(160, 60, 60, 200)   # red
                painter.fillRect(2, 2, stw, sth, badge_bg)
                painter.setPen(QColor("#f0f3f8"))
                painter.drawText(2, 2, stw, sth, Qt.AlignCenter, size_text)
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
            bool(self._no_faces_detected),
            int(card_w),
            selected_idx,
            active_idx,
            tuple(entry_signature),
        )

    def render_face_preview_strip(self):
        entries = self.face_preview_entries
        selection_mode = self.pipeline.awaiting_face_selection
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
            self.face_preview_header.setVisible(True)
            self.face_preview_strip_filmstrip.setVisible(True)
            return

        self.clear_face_preview_strip_layout()
        if wide_mode:
            self.face_preview_strip_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        else:
            self.face_preview_strip_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        if not entries:
            if self._no_faces_detected:
                self.face_preview_summary_label.setText("No faces detected — try another image")
                self.face_preview_summary_label.setStyleSheet("color: #d4a054;")
            else:
                self.face_preview_summary_label.setText("Faces: none")
            self._update_batch_queue_status()
            if hasattr(self, "face_selection_notice_label"):
                self.face_selection_notice_label.setVisible(False)
            if hasattr(self, "face_select_all_button"):
                self.face_select_all_button.setVisible(False)
            if hasattr(self, "face_select_none_button"):
                self.face_select_none_button.setVisible(False)
            if hasattr(self, "face_preview_auto_follow_checkbox"):
                self.face_preview_auto_follow_checkbox.setVisible(False)
            self.face_preview_header.setVisible(True)
            self.face_preview_strip_filmstrip.setVisible(True)
            if hasattr(self, "face_preview_panel"):
                self.face_preview_panel.setVisible(True)
            self._face_strip_render_signature = render_signature

            if self._no_faces_detected:
                # Show a centered notice inside the filmstrip
                self.face_preview_strip_filmstrip.show_empty_frames = False
                notice = QLabel("No faces detected.\nPlease try another image.")
                notice.setAlignment(Qt.AlignCenter)
                notice.setWordWrap(True)
                notice.setStyleSheet(
                    "color: #d4a054; background: transparent; font-size: 13px; padding: 12px;"
                )
                notice.setMinimumHeight(60)
                self.face_preview_strip_layout.setAlignment(Qt.AlignCenter)
                self.face_preview_strip_layout.addWidget(notice)
            else:
                self.face_preview_strip_filmstrip.show_empty_frames = True

            self.face_preview_strip_filmstrip.update()
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
        self.face_preview_summary_label.setStyleSheet("color: #aeb4be;")
        self._update_batch_queue_status()

        # Repaint filmstrip (frames paint behind face buttons for visual continuity)
        self.face_preview_strip_filmstrip.update()

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
        self.face_preview_strip_filmstrip.setVisible(True)

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

        sprocket = FilmstripContainerWidget.SPROCKET_BAND
        if wide_mode:
            # Vertical filmstrip: inset cards from sprocket bands and add
            # breathing room between adjacent tiles.
            gutter = 3  # px each side so card borders don't touch sprockets
            self.face_preview_strip_layout.setContentsMargins(gutter, 2, gutter, 2)
            self.face_preview_strip_layout.setSpacing(7)  # visible gap between 1px borders
        else:
            self.face_preview_strip_layout.setContentsMargins(4, 2, 4, 2)
            self.face_preview_strip_layout.setSpacing(4)

        # Compute frame size to fill area between sprocket bands
        if wide_mode:
            # Available content lane = filmstrip inner width minus gutters and
            # scrollbar so cards never bleed into sprocket bands even when
            # the vertical scrollbar is visible.
            scrollbar_w = max(5, self.face_preview_strip_scroll.verticalScrollBar().sizeHint().width())
            content_lane = self.face_preview_strip_filmstrip.width() - 2 * sprocket
            if content_lane <= 0:
                content_lane = self._get_face_strip_card_width(wide_mode)
            # Keep a small safety margin to avoid 1px border clipping under
            # high-DPI scaling and scrollbar visibility toggles.
            card_w = max(88, content_lane - 2 * gutter - scrollbar_w - 2)
            card_h = card_w  # Square cards in vertical layout
            thumb_size = max(48, card_w - 8)
        else:
            # In stacked mode (horizontal filmstrip): fill the available height
            # Deduct: sprocket bands (x2), layout margins (top+bottom=4), scrollbar (5),
            # and extra safety (4) to prevent sub-pixel rounding from triggering scroll
            avail_h = self.face_preview_strip_filmstrip.maximumHeight() - sprocket * 2 - 4 - 5 - 4
            card_w = max(88, avail_h)
            card_h = avail_h
            thumb_size = max(48, card_h - 8)

        for entry in display_entries:
            idx = entry["index"]
            status = entry.get("status", "queued")
            is_selected = bool(entry.get("selected", False))
            if selection_mode:
                label = "Selected" if is_selected else ""
            else:
                label = status.capitalize()
            icon_path = entry.get("result_path") or entry.get("crop_path")

            button = FaceStripToolButton()
            button.setCheckable(True)
            button.setChecked(is_selected if selection_mode else (idx == self.selected_face_preview_index))
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            # In selection mode, show all faces at full brightness so they look
            # interactable — use borders/check state for selection feedback.
            is_muted = (not is_selected) and (not selection_mode)
            button.setIconSize(QSize(thumb_size, thumb_size))
            button.setFixedSize(card_w, card_h)
            button.setCursor(Qt.PointingHandCursor)
            relative_size = entry.get("relative_size")
            button.setIcon(self._make_face_thumb_icon(
                icon_path, str(idx + 1),
                muted=is_muted,
                thumb_size=thumb_size,
                relative_size=relative_size,
            ))
            # Tooltip surfaces what the badges mean.
            tip_lines = [f"Face #{idx + 1}", f"Status: {status}"]
            if isinstance(relative_size, float):
                pct = int(round(relative_size * 100))
                if pct < 15:
                    quality = "small box — likely false positive"
                elif pct < 30:
                    quality = "borderline size"
                else:
                    quality = "clear subject size"
                tip_lines.append(f"Size: {pct}% of image ({quality})")
            tip_lines.append("Click to toggle. Right-click for more options.")
            button.setToolTip("\n".join(tip_lines))
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            # Stash inline lambdas on the button itself. PySide6 can drop
            # connections to unreferenced Python callables when the slot's
            # receiver isn't a QObject (controller promotion side-effect);
            # keeping a strong reference next to the QObject that fires the
            # signal makes the connection robust.
            button._context_handler = lambda pos, i=idx, btn=button: self._show_face_card_context_menu(btn, i, pos)
            button.customContextMenuRequested.connect(button._context_handler)

            # Frame styling matching app color scheme
            if button.isChecked():
                border = "#4a9eff"
            elif selection_mode:
                if is_selected:
                    border = "#1f6fd9"
                elif status == "done":
                    border = "#2e8b57"  # Green tint for completed faces
                else:
                    border = "#2a3038"
            else:
                border = status_style_map.get(status, "#2a3038")
            if is_muted:
                text_color = "#606878"
                bg_color = "#191d23"
                hover_color = "#1e2430"
            else:
                text_color = "#c8cdd5"
                bg_color = "#191d23"
                hover_color = "#232a35"
            button.setStyleSheet(FaceStripToolButton.get_stylesheet(border, bg_color, text_color, hover_color))
            if is_muted and (not selection_mode) and is_processing:
                button.setEnabled(False)
            button.setProperty("faceIndex", idx)
            # Event filters must be QObjects; the controller is plain-Python.
            # MainWindow owns the eventFilter() method anyway.
            button.installEventFilter(self._window)
            button.hover_enter_callback = (lambda i=idx: self.set_hover_face_preview_index(i))
            button.hover_leave_callback = self.clear_hover_face_preview_index
            # Stash on the button (see note above for context handler).
            button._click_handler = lambda checked=False, i=idx: self.select_face_preview(
                i, user_initiated=True, selection_checked=checked,
            )
            button.clicked.connect(button._click_handler)
            self.face_preview_strip_layout.addWidget(button)

        self.face_preview_strip_layout.addStretch(1)
        self._face_strip_render_signature = render_signature

        # Clamp the container's cross-axis maximum so its content can never
        # exceed the viewport — eliminates slight cross-axis scroll bleed that
        # occurs when the primary-axis scrollbar eats a few pixels.
        def _clamp_cross_axis():
            if wide_mode:
                # Wide mode = vertical filmstrip (TopToBottom); clamp cross-axis (width)
                vp_w = self.face_preview_strip_scroll.viewport().width()
                if vp_w > 0:
                    self.face_preview_strip_container.setMaximumWidth(vp_w)
            else:
                # Stacked mode = horizontal filmstrip (LeftToRight); clamp cross-axis (height)
                vp_h = self.face_preview_strip_scroll.viewport().height()
                if vp_h > 0:
                    self.face_preview_strip_container.setMaximumHeight(vp_h)
        QTimer.singleShot(0, _clamp_cross_axis)

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
            self.preview.result_preview_path_before_hover = self.preview.last_result_image_path
        self.hover_face_preview_index = idx
        self.hover_face_preview_source = source_name if isinstance(idx, int) else None
        self.hover_face_box_override = None
        if isinstance(idx, int):
            cached_box = self.hover_face_box_cache.get(idx)
            if cached_box is None:
                cached_box = self.preview.resolve_hover_face_box(idx)
                self.hover_face_box_cache[idx] = cached_box
            self.hover_face_box_override = cached_box
            hover_preview_path = self.get_face_preview_path(idx)
            if hover_preview_path is not None:
                self.preview.set_result_preview_image(hover_preview_path)
        self.preview.refresh_input_preview_scale()

    def clear_hover_face_preview_index(self):
        had_hover = (self.hover_face_preview_index is not None) or (self.hover_face_box_override is not None)
        self.hover_face_preview_index = None
        self.hover_face_preview_source = None
        self.hover_face_box_override = None
        if had_hover:
            restore_path = self.get_selected_face_preview_path()
            if restore_path is None:
                restore_path = self.preview.result_preview_path_before_hover
            if restore_path is not None:
                self.preview.set_result_preview_image(restore_path)
            self.preview.result_preview_path_before_hover = None
            self.preview.refresh_input_preview_scale()

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

    def select_face_preview(self, face_index, user_initiated=False, selection_checked=None):
        if face_index < 0 or face_index >= len(self.face_preview_entries):
            return
        if user_initiated and (not self.pipeline.awaiting_face_selection) and (not self._is_face_interaction_allowed(face_index)):
            return

        self.selected_face_preview_index = face_index
        entry = self.face_preview_entries[face_index]
        if self.pipeline.awaiting_face_selection and user_initiated:
            if selection_checked is None:
                entry["selected"] = not bool(entry.get("selected", False))
            else:
                entry["selected"] = bool(selection_checked)
        chosen_path = self.get_face_preview_path(face_index)
        if chosen_path is not None and Path(chosen_path).exists():
            self.preview.set_result_preview_image(Path(chosen_path))

        if user_initiated and (not self.pipeline.awaiting_face_selection) and self._is_processing_active():
            # User manually clicked a face during processing — suppress auto-follow
            # so the before/after slider doesn't jump to a different face.
            self._user_inspecting_completed_face = True

        if self.pipeline.awaiting_face_selection:
            selected_count = len(self.get_selected_face_indices())
            self.run_button.setEnabled(selected_count > 0)
            self.update_runtime_label()
            if user_initiated:
                is_now_selected = bool(entry.get("selected", False))
                face_num = face_index + 1
                if selected_count > 0:
                    action = "selected" if is_now_selected else "deselected"
                    self.status_label.setText(
                        f"Status: Face {face_num} {action} — {selected_count} face(s) ready, click Run"
                    )
                else:
                    self.status_label.setText("Status: Select at least one face, then click Run")

        self.render_face_preview_strip()
        self.preview.refresh_input_preview_scale()

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
        if not self.pipeline.awaiting_face_selection or not self.face_preview_entries:
            return
        for entry in self.face_preview_entries:
            entry["selected"] = bool(selected)
        self.run_button.setEnabled(len(self.get_selected_face_indices()) > 0)
        self.update_runtime_label()
        self.render_face_preview_strip()
        self.preview.refresh_input_preview_scale()

    def _show_face_card_context_menu(self, anchor_widget, face_index, pos):
        """Right-click on a face card: offer to skip / re-include / remove."""
        if face_index is None or face_index < 0 or face_index >= len(self.face_preview_entries):
            return
        entry = self.face_preview_entries[face_index]
        status = entry.get("status", "queued")
        is_selected = bool(entry.get("selected", False))

        menu = QMenu(anchor_widget)
        # Toggle selection only makes sense before the run starts.
        if self.pipeline.awaiting_face_selection or status == "queued":
            if is_selected:
                act_toggle = menu.addAction("Skip this face")
                # Stash the handler on the action so PySide6 doesn't drop
                # the connection to a GC-eligible lambda (the controller
                # is not a QObject, so the bound-method route doesn't
                # protect inline lambdas).
                act_toggle._handler = lambda _checked=False, i=face_index: self.select_face_preview(
                    i, user_initiated=True, selection_checked=False,
                )
                act_toggle.triggered.connect(act_toggle._handler)
            else:
                act_toggle = menu.addAction("Include this face")
                act_toggle._handler = lambda _checked=False, i=face_index: self.select_face_preview(
                    i, user_initiated=True, selection_checked=True,
                )
                act_toggle.triggered.connect(act_toggle._handler)
            menu.addSeparator()
        # Remove is always available unless the face is mid-run.
        if status != "running":
            act_remove = menu.addAction("Remove from queue")
            act_remove._handler = lambda _checked=False, i=face_index: self.remove_face_entry(i)
            act_remove.triggered.connect(act_remove._handler)
        else:
            menu.addAction("Remove from queue (busy)").setEnabled(False)
        if menu.actions():
            menu.exec(anchor_widget.mapToGlobal(pos))

    def remove_face_entry(self, face_index):
        """Drop a face from the queue entirely. Reindexes the survivors so
        the filmstrip stays compact."""
        if face_index is None:
            return
        try:
            idx = int(face_index)
        except Exception:
            return
        if idx < 0 or idx >= len(self.face_preview_entries):
            return
        removed = self.face_preview_entries.pop(idx)
        # Reindex survivors so positional invariants (1-based labels, box
        # alignment) still hold.
        for new_pos, entry in enumerate(self.face_preview_entries):
            entry["index"] = new_pos

        # Reset transient selection indices that may now point past the end.
        n = len(self.face_preview_entries)
        if self.selected_face_preview_index is not None and self.selected_face_preview_index >= n:
            self.selected_face_preview_index = (n - 1) if n else None
        if self.active_face_preview_index is not None and self.active_face_preview_index >= n:
            self.active_face_preview_index = None
        if self.hover_face_preview_index is not None and self.hover_face_preview_index >= n:
            self.hover_face_preview_index = None
        self.hover_face_box_cache = {}
        self.hover_face_box_override = None

        label = ""
        crop_path = removed.get("crop_path")
        if crop_path:
            label = f" ({Path(crop_path).name})"
        self.log_box.append(f"Removed face #{idx + 1}{label} from the queue.")

        if n == 0:
            self._no_faces_detected = True
            self.pipeline.awaiting_face_selection = False
            self.set_run_button_continue_mode(False)

        if hasattr(self, "run_button"):
            self.run_button.setEnabled(len(self.get_selected_face_indices()) > 0)
        self.update_runtime_label()
        # Recompute the preview overlay to drop the removed face's box.
        self.preview.update_input_face_boxes_for_preview(expected_count=n if n else None)
        self.render_face_preview_strip()

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

        # Prefer result_path if it exists (the enhanced version)
        if preview_path is not None:
            p = Path(preview_path)
            if p.exists():
                return p
            # If result_path was set but file doesn't exist, still prefer it over crop
            # (it may be in progress or the file may be temporarily unavailable)
            # Only fall through if file truly doesn't exist

        # Fall back to crop path or enhanced version of crop
        crop_path = entry.get("crop_path")
        if crop_path is not None:
            # Only check for enhanced version if we don't already have a result_path set
            if preview_path is None:
                enhanced = self.preview._resolve_enhanced_preview_for_crop(Path(crop_path))
                if enhanced is not None:
                    preview_path = enhanced
            if preview_path is None:
                preview_path = crop_path

        if preview_path is None:
            return None
        p = Path(preview_path)
        return p if p.exists() else None

    def _get_focused_face_preview_index(self):
        if not self.face_preview_entries:
            return None
        # When the user has manually clicked a completed face during processing,
        # prefer their selection over the actively-processing face index.
        if self._user_inspecting_completed_face:
            candidates = [
                self.selected_face_preview_index,
                self.hover_face_preview_index,
                self.active_face_preview_index,
            ]
        else:
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

    def _find_face_index_for_crop_path(self, crop_path: Path):
        crop_name = crop_path.name.lower()
        crop_stem = crop_path.stem.lower()
        # When GFPGAN is used, projector input is the blended face (e.g. face_0_blend.png).
        # Strip the _blend suffix to match against original crop entries (face_0.png).
        stripped_stem = crop_stem
        if stripped_stem.endswith("_blend"):
            stripped_stem = stripped_stem[:-6]
        for entry in self.face_preview_entries:
            entry_crop = entry.get("crop_path")
            if entry_crop is None:
                continue
            ec = Path(entry_crop)
            ec_name = ec.name.lower()
            ec_stem = ec.stem.lower()
            if ec_name == crop_name or ec_stem == crop_stem:
                return entry["index"]
            if stripped_stem != crop_stem and ec_stem == stripped_stem:
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
            # Don't store blended face paths as crop_path — keep original crop
            if not crop_path.stem.endswith("_blend"):
                entry["crop_path"] = crop_path
        if entry.get("status") != "done":
            entry["status"] = "running"

        self.active_face_preview_index = idx
        if self.face_preview_auto_follow_checkbox.isChecked() and not self._user_inspecting_completed_face:
            self.selected_face_preview_index = idx
            preview_path = entry.get("result_path")
            if preview_path is None:
                # Try to resolve the enhanced/blended version for display
                enhanced = self.preview._resolve_enhanced_preview_for_crop(crop_path)
                if enhanced is None:
                    original_crop = entry.get("crop_path")
                    if original_crop is not None:
                        enhanced = self.preview._resolve_enhanced_preview_for_crop(Path(original_crop))
                preview_path = enhanced or entry.get("crop_path")
            if preview_path is not None and Path(preview_path).exists():
                self.preview.set_result_preview_image(Path(preview_path))
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

        if self.face_preview_auto_follow_checkbox.isChecked() and not self._user_inspecting_completed_face:
            self.selected_face_preview_index = idx
            if result_path.exists():
                self.preview.set_result_preview_image(result_path)

        self.render_face_preview_strip()

        # Auto-recompose if enabled
        if getattr(self, "auto_recompose_checkbox", None) and self.auto_recompose_checkbox.isChecked():
            QTimer.singleShot(500, self.run_recomposite)

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
        for dir_text in sorted(self.pipeline.current_run_result_dirs):
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

