"""Preflight, run-summary, and hardware-info mixin for the rephotography GUI.

Sprint-4 architectural slice: pulls the startup-preflight collection, the
preflight dialog, the run-summary text/dialog, and the GPU/CPU/RAM probe
out of MainWindow into a mixin. Methods are unchanged — only their
physical location moves. MainWindow inherits from PreflightMixin (plus
the previous mixins and QMainWindow).
"""

import ctypes
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from gui import format_utils
from gui.widgets import _PreflightRunner, _PreflightSignals


class PreflightMixin:
    """Mix into MainWindow to provide preflight + run-summary + hardware info."""

    def _collect_preflight_report(self, results_root_override: Optional[str] = None):
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

        if results_root_override is not None:
            results_root_text = results_root_override
        else:
            results_root_text = self.results_root_edit.text().strip()
        results_root = Path(results_root_text or (self.repo_root / "results"))
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
        return format_utils.preflight_report_plain_text(report)

    def _preflight_report_html(self, report):
        return format_utils.preflight_report_html(report)

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
        # Heavy preflight (nvidia-smi, file probes, write test) runs off the
        # UI thread so the window paints immediately. Result arrives via slot.
        if getattr(self, "_preflight_running", False):
            return
        self._preflight_running = True
        self._preflight_show_dialog = bool(show_dialog)
        self._preflight_user_initiated = bool(user_initiated)

        # Snapshot widget reads on the UI thread; Qt widgets are not
        # thread-safe to read from a worker.
        try:
            results_root_snapshot = self.results_root_edit.text().strip()
        except Exception:
            results_root_snapshot = ""

        signals = _PreflightSignals()
        signals.finished.connect(self._on_preflight_finished)
        self._preflight_signals = signals
        self._pixmap_thread_pool.start(
            _PreflightRunner(self, signals, results_root_snapshot)
        )

    def _on_preflight_finished(self, report):
        self._preflight_running = False
        self.last_preflight_report = report

        self.log_box.append("")
        self.log_box.append("=== Startup preflight ===")
        for check in report["checks"]:
            self.log_box.append(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
            if check.get("fix") and check["status"] != "pass":
                self.log_box.append(f"  Fix: {check['fix']}")
        self.log_box.append(f"Preflight summary: {report['pass']} pass, {report['warn']} warn, {report['fail']} fail")

        should_show = self._preflight_show_dialog and (self._preflight_user_initiated or report["fail"] > 0)
        if should_show:
            self.show_preflight_dialog(report)

    def _format_elapsed_for_summary(self, elapsed_seconds):
        return format_utils.format_elapsed_for_summary(elapsed_seconds)

    def _build_run_summary_text(self, success, exit_code, elapsed_seconds, output_path=None, launch_error=None):
        ctx = self.current_run_summary_context or {}
        status = "Success" if success else "Failed"
        if launch_error and int(exit_code) < 0:
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
            detail_label = "Launch error" if int(exit_code) < 0 else "Error detail"
            lines.extend(["", f"{detail_label}: {launch_error}"])

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

        # GPU (best effort). Bound the subprocess so a stuck nvidia-smi
        # cannot freeze the UI; this call previously had no timeout.
        gpu_name = "Unknown GPU"
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=2.0
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

