"""Pipeline controller mixin for the rephotography GUI.

Sprint-4 architectural slice: ~1,600 lines of subprocess lifecycle and
stdout-parsing logic pulled out of MainWindow into a mixin. The methods
are unchanged — only their physical location moves. MainWindow inherits
from PipelineMixin (plus PreviewMixin, FaceStripMixin, QMainWindow).

This mixin owns:
  * Run lifecycle: handle_run_button_clicked, run_wrapper,
    run_face_detection, _run_crop_only_preview,
    _prepare_face_selection_after_preprocess,
    continue_rephoto_with_selected_faces, process_finished, process_error,
    cancel_run, request_end_run_early.
  * Process management: _kill_process_if_running,
    _terminate_process_nonblocking, _control_process_tree_windows,
    _start_wrapper_process, _show_run_context_menu.
  * Pause / resume: set_backend_paused, toggle_pause_resume,
    _arm_pause_ack_warning, _cancel_pause_ack_warning,
    _on_pause_ack_warning_timeout, _prepare_stop_flag_for_new_run,
    _clear_current_stop_flag, _clear_current_pause_flag.
  * Stdout / stderr: append_stdout_from_process, append_stderr_from_process,
    _append_process_log_text, _flush_process_log_buffer,
    _consume_process_output_lines, update_progress_from_line (the 219-line
    regex-routing dispatch), append_command_preview.
  * Controls + progress bars: set_controls_for_running,
    set_run_button_continue_mode, _set_status_for_running_state,
    reset_progress_bars, set_preprocess_progress, set_rephoto_progress,
    _confirm_high_iteration_count, is_true_process_type.
  * Wrapper command: build_wrapper_command (143-line argv assembler),
    _can_reuse_existing_crops, _reset_wrapper_runtime_tracking,
    _seed_wrapper_crop_dir_from_preview, _capture_run_context.

Relies on standard MainWindow instance state (self.process, self.run_paused,
self.face_strip.face_preview_entries, self.advanced_dialog, self.log_box,
self.status_label, the progress bars, the timing/elapsed timers, etc.).
"""

import os
import platform
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMenu, QMessageBox

from gui.constants import (
    CROPPED_FACE_COUNT_RE,
    ITER_PROGRESS_RE,
    NO_CROPS_CREATED_RE,
    QUICK_FACE_DECISION_RE,
    REPHOTO_CROP_FAIL_RE,
    SIMPLE_FINAL_COPY_RE,
)


class PipelineMixin:
    """Mix into MainWindow to provide subprocess + stdout-parsing pipeline."""

    def _kill_process_if_running(self, proc):
        if proc is None:
            return
        try:
            if proc.state() != QProcess.NotRunning:
                proc.kill()
        except RuntimeError:
            pass

    def _terminate_process_nonblocking(self, proc, grace_ms=2000):
        if proc is None:
            return
        try:
            if proc.state() == QProcess.NotRunning:
                return
            proc.terminate()
            QTimer.singleShot(grace_ms, lambda p=proc: self._kill_process_if_running(p))
        except RuntimeError:
            pass

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
            "selected_faces": self.face_strip.get_selected_face_count_text(),
        }

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

    def run_face_detection(self):
        """Re-run face detection using the current threshold setting."""
        if not self.input_image_edit.text().strip():
            self.log_box.append("No input image selected. Please load an image first.")
            return

        if self.process is not None:
            self.log_box.append("A backend process is already running.")
            return

        # Clear existing face entries and run detection in crop-only mode
        self.face_strip.face_preview_entries = []
        self.face_strip.selected_face_preview_index = None
        self.face_strip.render_face_preview_strip()

        try:
            command = self.build_wrapper_command(force_crop_only=True)
            if not command:
                return

            self.log_box.append(
                f"Re-detecting faces with threshold "
                f"{self.advanced_dialog.det_threshold_edit.value():.2f}..."
            )

            self.stop_quick_face_probe()
            self.set_input_detect_overlay(False)
            self.clear_result_stage_overlay()
            self.reset_progress_bars()
            self.current_run_summary_context = self._capture_run_context()
            self.current_run_phase = "preprocess"
            self.awaiting_face_selection = False
            self.selection_preprocess_mode = True
            self.suppress_preprocess_ui_until_rephoto = False
            self.set_run_button_continue_mode(False)

            self._reset_wrapper_runtime_tracking()
            self._prepare_stop_flag_for_new_run()
            self.set_preprocess_progress(5, "Re-detecting faces...")
            self.set_rephoto_progress(0, "Waiting...")

            self._start_wrapper_process(command, "Status: Re-detecting faces...")
        except Exception as e:
            self.log_box.append(f"Error re-detecting faces: {e}")
            import traceback
            traceback.print_exc()

    def is_true_process_type(self, photo_type):
        """Return True only for actual photographic processes."""
        return photo_type in ("Daguerreotype", "Ambrotype", "Tintype / Ferrotype")

    def set_controls_for_running(self, is_running):
        self.face_strip.update_run_button_for_quick_face_hint()
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
        if hasattr(self, "redetect_faces_button"):
            self.redetect_faces_button.setEnabled(not is_running)
        if hasattr(self, "det_threshold_slider"):
            self.det_threshold_slider.setEnabled(not is_running)

        if is_running:
            self.advanced_dialog.use_gfpgan_checkbox.setEnabled(False)
        else:
            self.update_mode_controls()
            self.update_spectral_sensitivity_ui()

    def update_progress_from_line(self, line: str):
        s = (line or "").strip()
        if not s:
            return

        # Capture actionable backend failure details so summaries are useful.
        no_crops_match = NO_CROPS_CREATED_RE.match(s)
        if no_crops_match:
            crop_dir = no_crops_match.group(1).strip()
            self._last_backend_error_detail = (
                f"No faces were detected for this image (crop output was empty: {crop_dir}). "
                "Try lowering Detection Sensitivity (for example 0.6) or using a tighter face crop."
            )
        elif "Face crop failed (command:" in s:
            self._last_backend_error_detail = s
        elif s.startswith("Input image not found:"):
            self._last_backend_error_detail = s

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
            # Backend acknowledged — no nag needed.
            self._cancel_pause_ack_warning()
            return
        if s.startswith("=== Resume requested ==="):
            self._set_status_for_running_state()
            self._restore_result_stage_overlay_after_pause()
            self._cancel_pause_ack_warning()
            return

        # === Path tracking from stdout ===
        if has_colon:
            if s.startswith("Crop:"):
                crop_path_text = s.split("Crop:", 1)[-1].strip()
                if crop_path_text:
                    self.face_strip.mark_face_running_from_crop_path(crop_path_text)
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
                    self.face_strip.mark_face_done_from_result_path(simple_copy_match.group(1))

            if "projector.py failed for crop:" in s:
                fail_match = REPHOTO_CROP_FAIL_RE.search(s)
                if fail_match:
                    self.face_strip.mark_face_failed_from_crop_name(fail_match.group(1))

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
            if not self.face_strip.face_preview_entries:
                self.face_strip.initialize_face_preview_entries(expected_count=1)
            if self.current_run_phase == "rephoto":
                if self.rephoto_face_total <= 0:
                    self.rephoto_face_total = max(1, len(self.face_strip.get_selected_face_indices()))
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
                if "skipped" in s.lower():
                    if not suppress_all_preprocess_ui:
                        self.set_preprocess_progress(40, "Crops reused")
                        self.preprocess_stage = "crops_ready"
                else:
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
                if (not self.selection_preprocess_mode) and (not self.face_strip.face_preview_entries):
                    self.face_strip.initialize_face_preview_entries(expected_count=n)
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

    def set_run_button_continue_mode(self, is_continue_mode):
        if is_continue_mode:
            can_continue = len(self.face_strip.get_selected_face_indices()) > 0
            self.run_button.setText("Run")
            self.run_button.setEnabled(can_continue)
            if can_continue:
                self.run_button.setToolTip("Select one or more faces in the filmstrip, then run rephotography.")
            else:
                self.run_button.setToolTip("Select at least one face in the filmstrip to enable Run.")
            return
        self.face_strip.update_run_button_for_quick_face_hint()

    def handle_run_button_clicked(self):
        if self.process is not None:
            self.toggle_pause_resume()
            return
        if not self._confirm_high_iteration_count():
            return
        self.run_wrapper()

    def _confirm_high_iteration_count(self) -> bool:
        try:
            preset = int(self.get_selected_preset_value())
        except Exception:
            return True
        if preset <= self._HIGH_ITERATION_CONFIRM_THRESHOLD:
            return True
        # Suppress repeated prompts within the same window session once the
        # user has acknowledged for the chosen value.
        last_ack = getattr(self, "_high_iter_last_acked_value", None)
        if last_ack == preset:
            return True
        approx_minutes = max(1, preset // 250)  # rough heuristic: ~250 iters/min on a modern GPU
        reply = QMessageBox.question(
            self,
            "Confirm long run",
            f"Selected iteration count: {preset}.\n\n"
            f"This is well above the standard presets (≤ 3000) and can take "
            f"roughly {approx_minutes} minutes per face. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._high_iter_last_acked_value = preset
            return True
        return False

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
                self._arm_pause_ack_warning()
            else:
                self.log_box.append("Run resumed.")
                self._set_status_for_running_state()
                self._restore_result_stage_overlay_after_pause()
                self._cancel_pause_ack_warning()
        self.face_strip.update_run_button_for_quick_face_hint()
        return True

    def _arm_pause_ack_warning(self):
        self._cancel_pause_ack_warning()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._on_pause_ack_warning_timeout)
        timer.start(self._PAUSE_ACK_WARN_MS)
        self._pause_ack_warn_timer = timer

    def _cancel_pause_ack_warning(self):
        timer = getattr(self, "_pause_ack_warn_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass
            self._pause_ack_warn_timer = None

    def _on_pause_ack_warning_timeout(self):
        # Only nag if we're still in the "requested" state — if the backend
        # already acknowledged, the stdout-handler cleared this timer.
        if not self.run_paused or self.process is None:
            return
        self.log_box.append(
            "Backend has not yet acknowledged pause. It will pause at the next "
            "checkpoint (long forward passes can delay this by 5–30s)."
        )

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
            self.face_strip.update_run_button_for_quick_face_hint()

        self.end_early_button.setEnabled(False)
        self.log_box.append("End-early requested. Finishing with completed iterations only...")
        self.status_label.setText("Status: Ending run early...")

    def _seed_wrapper_crop_dir_from_preview(self, input_image_path: Path):
        if input_image_path is None or (not input_image_path.exists()):
            return

        preview_crops = []
        for entry in self.face_strip.face_preview_entries:
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

    def _reset_wrapper_runtime_tracking(self, preserve_crop_dir=False, preserve_enhance_dirs=False):
        if not preserve_crop_dir:
            self.current_crop_output_dir = None
            self._crop_source_input_key = None
            self._crop_source_face_factor = None
        if not preserve_enhance_dirs:
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
        self._last_backend_error_detail = ""
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
        self.face_strip.update_run_button_for_quick_face_hint()

    def _prepare_face_selection_after_preprocess(self):
        self.set_input_detect_overlay(False)
        crop_files = self.collect_current_crop_files()
        crop_count = len(crop_files)
        self.set_preprocess_progress(0, "Preprocess ready")
        self.set_rephoto_progress(0, "Waiting...")
        if crop_count <= 0:
            self.face_strip._no_faces_detected = True
            self.face_strip.render_face_preview_strip()
            self.log_box.append("No detected faces were found after preprocessing.")
            self.status_label.setText("Status: No faces detected")
            self.clear_result_stage_overlay()
            self.preflight.show_run_summary_dialog(
                success=False,
                exit_code=1,
                elapsed_seconds=None,
                output_path=None,
                launch_error="No detected faces were found.",
            )
            self.current_run_summary_context = None
            return

        self.face_strip.initialize_face_preview_entries(expected_count=crop_count)
        self.face_strip._sync_face_preview_crop_paths()

        if crop_count == 1:
            for entry in self.face_strip.face_preview_entries:
                entry["selected"] = True
                entry["status"] = "queued"
            self.face_strip.render_face_preview_strip()
            self.log_box.append("Single face detected. Continuing automatically...")
            self.status_label.setText("Status: Single face detected, continuing...")
            self.awaiting_face_selection = False
            self.update_image_import_controls()
            self.set_run_button_continue_mode(False)
            QTimer.singleShot(0, self.continue_rephoto_with_selected_faces)
            return

        # Restore previous face selections if this was a re-crop (expansion change)
        reselect = self.face_strip._pending_face_reselection
        self.face_strip._pending_face_reselection = None
        for entry in self.face_strip.face_preview_entries:
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
        self.face_strip.render_face_preview_strip()

    def continue_rephoto_with_selected_faces(self):
        if self.process is not None:
            return

        selected_indices = self.face_strip.get_selected_face_indices()
        if not selected_indices:
            self.status_label.setText("Status: Select at least one face to run")
            self.log_box.append("Select at least one face to rephotograph before running.")
            self.run_button.setEnabled(False)
            return

        self.face_strip._sync_face_preview_crop_paths()
        self.awaiting_face_selection = False
        self.face_strip._user_inspecting_completed_face = False
        self.suppress_preprocess_ui_until_rephoto = False
        self.set_input_detect_overlay(False)
        self.update_image_import_controls()
        self.selection_preprocess_mode = False
        self.set_run_button_continue_mode(False)

        selected_set = set(selected_indices)
        for entry in self.face_strip.face_preview_entries:
            if entry["index"] in selected_set:
                entry["selected"] = True
                if entry.get("status") != "done":
                    entry["status"] = "queued"
            else:
                entry["selected"] = False
                entry["status"] = "skipped"
        self.face_strip.selected_face_preview_index = selected_indices[0]
        self.face_strip.render_face_preview_strip()
        self.current_run_summary_context = self._capture_run_context()
        self.current_run_summary_context["selected_faces"] = self.face_strip.get_selected_face_count_text()
        self.update_runtime_label()

        self.clear_result_stage_overlay()
        self.reset_progress_bars()
        self.current_run_phase = "preprocess"
        self.set_preprocess_progress(5, "Preparing selected faces...")
        self.set_rephoto_progress(0, "Waiting...")
        self._reset_wrapper_runtime_tracking(preserve_crop_dir=True, preserve_enhance_dirs=True)
        self._prepare_stop_flag_for_new_run()

        # Continuation must reuse the exact crop set shown in selection UI.
        # Re-cropping here can remap indices and select the wrong face.
        reuse_crops = True
        self._inprocess_preview_crops = False

        selected_crop_names = []
        for idx in selected_indices:
            crop_path = self.face_strip._get_face_crop_path(idx)
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
            self.run_button.setEnabled(len(self.face_strip.get_selected_face_indices()) > 0)
            return

        if use_selected_names:
            self.log_box.append(f"Running selected crop name(s): {', '.join(selected_crop_names)}")
        else:
            self.log_box.append(f"Running selected face index(es): {', '.join(str(i) for i in selected_indices)}")
        self._start_wrapper_process(command, "Status: Running selected face(s)...")

    def process_finished(self, exit_code, exit_status):
        self.run_paused = False
        self.face_strip._user_inspecting_completed_face = False
        self._cancel_pause_ack_warning()
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

        if exit_code != 0 and self.face_strip.active_face_preview_index is not None:
            idx = self.face_strip.active_face_preview_index
            if 0 <= idx < len(self.face_strip.face_preview_entries):
                self.face_strip.face_preview_entries[idx]["status"] = "failed"
            self.face_strip.active_face_preview_index = None
            self.face_strip.render_face_preview_strip()

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
            self.preflight.show_run_summary_dialog(
                success=False,
                exit_code=exit_code,
                elapsed_seconds=elapsed_seconds,
                output_path=None,
                launch_error=self._last_backend_error_detail or None,
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
                self.face_strip._sync_face_preview_crop_paths()
                self.face_strip.reconcile_face_preview_results(after_epoch=self.run_started_at)
                # Clear stale active index so compare wipe uses selected_face_preview_index.
                self.face_strip.active_face_preview_index = None
                results_root = Path(self.results_root_edit.text().strip() or (self.repo_root / "results"))
                newest = self.find_latest_image(results_root, self.run_started_at)

                if newest is None:
                    from_strip = self.face_strip.get_selected_face_preview_path()
                    if from_strip is not None:
                        self.set_result_preview_image(from_strip)
                        final_output_path = from_strip
                        self.log_box.append(f"Previewing latest face result: {from_strip.name}")
                    else:
                        self.log_box.append("No new result image was found in the results folder.")
                        self.set_result_preview_image(None)
                else:
                    run_folder = newest.parent
                    if len(self.face_strip.face_preview_entries) <= 1:
                        _, rephoto_path = self.simplify_run_folder(run_folder)
                        final_preview = rephoto_path or newest
                    else:
                        final_preview = newest
                    selected_face_preview = self.face_strip.get_selected_face_preview_path()
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

        summary_text = self.preflight.store_run_summary_text(
            success=(exit_code == 0),
            exit_code=exit_code,
            elapsed_seconds=elapsed_seconds,
            output_path=final_output_path,
            launch_error=(self._last_backend_error_detail or None) if exit_code != 0 else None,
        )
        if exit_code != 0 and summary_text:
            self.preflight.show_run_summary_text_dialog(summary_text)
        self.selection_preprocess_mode = False
        self.suppress_preprocess_ui_until_rephoto = False
        self.set_input_detect_overlay(False)
        self.current_run_summary_context = None

        # Clear process reference BEFORE re-enabling face interaction so that
        # _is_processing_active() returns False during the strip re-render.
        self.process = None

        # Re-enable face selection if crops still exist so the user can
        # pick additional or different faces for another run.
        has_face_entries = len(self.face_strip.face_preview_entries) > 1
        if has_face_entries and self.current_crop_output_dir:
            # Deselect completed faces so the user starts fresh for the next run.
            # "Done" faces keep their status for visual distinction but are no longer
            # pre-selected, making it clear which faces will run next.
            for entry in self.face_strip.face_preview_entries:
                if entry.get("status") == "done":
                    entry["selected"] = False
            self.awaiting_face_selection = True
            self.set_run_button_continue_mode(True)
            done_count = sum(1 for e in self.face_strip.face_preview_entries if e.get("status") == "done")
            remaining = len(self.face_strip.face_preview_entries) - done_count
            self.status_label.setText(
                f"Status: Run complete — select additional faces to process ({remaining} remaining)"
            )
        else:
            self.awaiting_face_selection = False
            self.set_run_button_continue_mode(False)

        self.face_strip.render_face_preview_strip()
        self.set_controls_for_running(False)
        self.run_started_at = None
        self.rephoto_started_at = None

    def process_error(self, process_error):
        self.run_paused = False
        self.face_strip._user_inspecting_completed_face = False
        self._cancel_pause_ack_warning()
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()
        if hasattr(self, "_elapsed_timer"):
            self._elapsed_timer.stop()
        self._flush_process_log_buffer()
        self._process_stdout_buffer = ""
        self._process_stderr_buffer = ""
        if self.face_strip.active_face_preview_index is not None:
            idx = self.face_strip.active_face_preview_index
            if 0 <= idx < len(self.face_strip.face_preview_entries):
                self.face_strip.face_preview_entries[idx]["status"] = "failed"
            self.face_strip.active_face_preview_index = None
            self.face_strip.render_face_preview_strip()
        self.log_box.append(f"Process launch error: {process_error}")
        self.status_label.setText("Status: Process launch error")
        self.set_input_detect_overlay(False)
        self.suppress_preprocess_ui_until_rephoto = False
        self.clear_result_stage_overlay()
        elapsed_seconds = (time.time() - self.run_started_at) if self.run_started_at is not None else None
        self.preflight.show_run_summary_dialog(
            success=False,
            exit_code=-1,
            elapsed_seconds=elapsed_seconds,
            output_path=None,
            launch_error=str(process_error),
        )
        self.selection_preprocess_mode = False
        self.current_run_summary_context = None

        self.process = None

        # Re-enable face selection if crops still exist.
        has_face_entries = len(self.face_strip.face_preview_entries) > 1
        if has_face_entries and self.current_crop_output_dir:
            self.awaiting_face_selection = True
            self.set_run_button_continue_mode(True)
        else:
            self.awaiting_face_selection = False
            self.set_run_button_continue_mode(False)

        self.face_strip.render_face_preview_strip()
        self.set_controls_for_running(False)
        self.run_started_at = None
        self.rephoto_started_at = None

    def cancel_run(self):
        if self.process is None:
            if self.awaiting_face_selection:
                self.log_box.append("Face selection step cancelled.")
                self.status_label.setText("Status: Face selection cancelled")
                self.face_strip.reset_face_preview_state(preserve_input_overlays=True)
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
        self._cancel_pause_ack_warning()
        self.log_box.append("Cancel requested. Stopping backend process...")
        self.status_label.setText("Status: Cancelling...")
        self.set_input_detect_overlay(False)
        self.suppress_preprocess_ui_until_rephoto = False
        self.clear_result_stage_overlay()
        self._clear_current_stop_flag()
        self._clear_current_pause_flag()

        if self.run_paused:
            self.run_paused = False

        # Non-blocking terminate-then-kill: schedule a kill on the event loop
        # 2s later instead of freezing the UI on waitForFinished().
        self._terminate_process_nonblocking(self.process, grace_ms=2000)

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
            self.face_strip.reset_face_preview_state(preserve_input_overlays=False)
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
            and len(self.face_strip.face_preview_entries) > 1
        )

        if recrop_only:
            # Save previous face selections to restore after re-crop
            prev_selected = set(self.face_strip.get_selected_face_indices())
            self.face_strip._pending_face_reselection = prev_selected if prev_selected else None

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
        self.face_strip.reset_face_preview_state(preserve_input_overlays=False)
        self.current_run_summary_context = self._capture_run_context()
        self.current_run_phase = "preprocess"
        self.awaiting_face_selection = False
        quick_count = self.quick_face_count_estimate if isinstance(self.quick_face_count_estimate, int) else None
        self.selection_preprocess_mode = bool(
            (not self.advanced_dialog.crop_only_checkbox.isChecked()) and (quick_count is None or quick_count > 1)
        )
        self.suppress_preprocess_ui_until_rephoto = bool(self.selection_preprocess_mode)
        if self.selection_preprocess_mode and current_key is not None:
            self.auto_detect_faces_triggered_input = current_key
        if quick_count == 0:
            self.log_box.append(
                "Quick scan found 0 faces at current threshold. Running full backend with auto-lower threshold fallback (may take longer)."
            )
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

