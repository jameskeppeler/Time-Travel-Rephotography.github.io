"""Read / write helpers for the per-run JSONL timing log.

Extracted from MainWindow as part of the Sprint-4 module split. These
functions take only their arguments — no MainWindow state, no Qt — so
they're trivially unit-testable in CI without PySide6.

The log lives at <results_root>/run_timing_log.jsonl with one JSON object
per line. Each record represents either a completed run or a per-step
milestone. The reader filters to successful, non-crop-only records that
have a recorded elapsed_seconds; that is the set the GUI uses to build
runtime estimates.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOG_FILENAME = "run_timing_log.jsonl"


def log_path_for(results_root: Path) -> Path:
    """Canonical absolute path of the timing log file for a results root."""
    return Path(results_root) / LOG_FILENAME


def read_timing_records(log_path: Path) -> List[Dict[str, Any]]:
    """Read the JSONL log and return successful, non-crop_only records
    that include an elapsed_seconds field. Lines that fail JSON parse,
    or records missing required fields, are silently skipped — the goal
    is "best effort" so the GUI never crashes because of one bad line.

    Returns an empty list if the file is missing or unreadable.
    """
    p = Path(log_path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
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
    except OSError:
        return []
    return records


def build_run_record(
    *,
    input_image: str,
    preset: str,
    enhancement: bool,
    crop_only: bool,
    success: bool,
    elapsed_seconds: float,
    gpu_name: str = "Unknown GPU",
    advanced_mode: bool = False,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the dict that gets appended as one JSONL line for a finished run.

    `timestamp` is injectable for deterministic tests; defaults to "now" in
    local time.
    """
    return {
        "timestamp_local": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_image": str(input_image or ""),
        "preset": str(preset),
        "advanced_mode": bool(advanced_mode),
        "enhancement": bool(enhancement),
        "crop_only": bool(crop_only),
        "success": bool(success),
        "elapsed_seconds": float(elapsed_seconds),
        "gpu_name": gpu_name,
    }


def build_milestone_record(
    *,
    input_image: str,
    preset: str,
    source_run_preset: str,
    enhancement: bool,
    elapsed_seconds: float,
    gpu_name: str = "Unknown GPU",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the dict that gets appended for a per-step milestone."""
    return {
        "record_type": "milestone",
        "timestamp_local": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_image": str(input_image or ""),
        "preset": str(preset),
        "source_run_preset": str(source_run_preset),
        "advanced_mode": False,
        "enhancement": bool(enhancement),
        "crop_only": False,
        "success": True,
        "elapsed_seconds": float(elapsed_seconds),
        "gpu_name": gpu_name,
    }


def append_records(log_path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Append `records` as JSONL to `log_path`, creating parent dirs if
    needed. Returns the number of records written. Raises OSError on I/O
    failure — callers decide whether to surface or swallow.
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


__all__ = [
    "LOG_FILENAME",
    "log_path_for",
    "read_timing_records",
    "build_run_record",
    "build_milestone_record",
    "append_records",
]
