"""Pure runtime-estimation helpers for the rephotography GUI.

Extracted from MainWindow as part of the Sprint-4 module split.

estimate_runtime_minutes() is a fairly complex decision tree:
  1. Exact local timing for this preset, same GPU, same enhancement mode.
  2. Exact local timing for this preset, any GPU, same enhancement mode.
  3. Infer the missing enhancement mode from the opposite-mode timings
     plus a learned enhancement-overhead offset.
  4. Additive model: base_rephoto + enhancement_overhead, with log-linear
     interpolation across presets when no exact match exists.

compute_runtime_scale() returns a multiplicative scale factor used when
there is no local history — it adjusts the baseline (RTX 3060 Laptop)
estimate by recognised GPU model and a small RAM heuristic.

Both functions are pure (input -> output) so they're trivially testable.
"""

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_preset(rec: Dict[str, Any]) -> Optional[int]:
    """Coerce a record's preset field to int, treating the legacy "test"
    label as 750. Returns None if unparseable."""
    p = rec.get("preset")
    if str(p).lower() == "test":
        return 750
    try:
        return int(p)
    except Exception:
        return None


def _filter_parsed(records: Iterable[Dict[str, Any]]) -> List[Tuple[int, float, str, bool, str]]:
    """Convert raw JSONL records into (preset, secs, gpu, enh, rtype) tuples,
    discarding ones with bad shape so downstream code can iterate freely."""
    out = []
    for rec in records:
        p = _parse_preset(rec)
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
        out.append((p, secs, gpu, enh, rtype))
    return out


def estimate_runtime_minutes(
    records: Iterable[Dict[str, Any]],
    current_gpu: str,
    current_enh: bool,
    preset_value: int,
) -> Tuple[Optional[float], str]:
    """Estimate runtime in minutes from a JSONL timing-record list.

    Returns (minutes_or_None, source_note). Returns (None, note) when no
    estimate is possible; the caller decides whether to display the note
    or fall back to the hardware-scaled baseline.

    See module docstring for the fallback chain.
    """
    records = list(records or [])
    if not records:
        return (None, "No local timing history yet.")

    current_gpu = (current_gpu or "").strip()
    current_enh = bool(current_enh)

    parsed = _filter_parsed(records)
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

    # ------------------------------------------------------------------
    # 1) Exact local timing match first
    # ------------------------------------------------------------------
    exact_same_gpu_same_enh = exact_matches(preset_value, current_enh, same_gpu_only=True)
    if exact_same_gpu_same_enh:
        types = sorted(set(rt for (_s, rt) in exact_same_gpu_same_enh))
        return (
            avg_secs(exact_same_gpu_same_enh) / 60.0,
            f"Using exact local timing from {len(exact_same_gpu_same_enh)} same-GPU record(s) at this preset ({', '.join(types)}).",
        )

    exact_same_enh = exact_matches(preset_value, current_enh, same_gpu_only=False)
    if exact_same_enh:
        types = sorted(set(rt for (_s, rt) in exact_same_enh))
        return (
            avg_secs(exact_same_enh) / 60.0,
            f"Using exact local timing from {len(exact_same_enh)} record(s) at this preset ({', '.join(types)}).",
        )

    # ------------------------------------------------------------------
    # 2) Learn enhancement overhead from paired on/off local timings
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3) Infer missing enhancement mode from opposite-mode exact timings
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4) Additive model:
    #    total ~= base_rephoto_time + enhancement_overhead
    # ------------------------------------------------------------------
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
        return (
            base_secs / 60.0,
            base_note + " No local enhancement overhead yet, so enhancement was not added.",
        )

    return (None, base_note + " " + overhead_note)


def compute_runtime_scale(hw_info: Dict[str, Any]) -> Tuple[float, str]:
    """Return (scale, note) for the no-local-history fallback path.

    The scale multiplies the baseline (RTX 3060 Laptop) minutes. Intentionally
    conservative — adjusts estimates slightly, not wildly.
    """
    gpu = (hw_info.get("gpu_name") or "").lower()
    ram = hw_info.get("ram_gb")

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


__all__ = [
    "estimate_runtime_minutes",
    "compute_runtime_scale",
]
