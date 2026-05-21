"""Shared defaults, presets, and stdout-parsing regexes for the rephoto GUI.

Extracted from gui/app.py as part of the Sprint-4 module split so that
gui/dialogs.py, gui/widgets.py, and the eventual controller modules can
import the same source-of-truth constants without circular imports.

This module has no Qt dependency — it's pure data and compiled regexes —
which keeps it import-cheap and unit-testable.
"""

import re


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
    "Daguerreotype": "e.g. 1840–1860",
    "Ambrotype": "e.g. 1854–1870",
    "Tintype / Ferrotype": "e.g. 1856–1900",
    "Carte de visite (CDV)": "e.g. 1859–1890",
    "Cabinet card": "e.g. 1866–1900",
    "Late cabinet card / dry plate studio portrait": "e.g. 1885–1910",
    "Early gelatin silver print": "e.g. 1890–1920",
    "Black-and-white snapshot / roll-film print": "e.g. 1900–1950",
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
NO_CROPS_CREATED_RE = re.compile(r"^No cropped face image files were created in:\s*(.+?)\s*$", re.IGNORECASE)


__all__ = [
    # Quality presets
    "DEFAULT_BASIC_ITER_VALUES",
    "DEFAULT_ADVANCED_ITER_VALUES",
    "DEFAULT_ITERATION",
    "QUALITY_PRESET_LABELS",
    # Layout / UI hints
    "PHOTO_TYPE_DATE_HINTS",
    "WIDE_LAYOUT_MIN_WIDTH",
    "IMAGE_EXTENSIONS",
    # Face detection / enhancement defaults
    "DEFAULT_FACE_FACTOR",
    "DEFAULT_GFPGAN_BLEND",
    "DEFAULT_DET_THRESHOLD",
    # ML defaults
    "DEFAULT_GAUSSIAN",
    "DEFAULT_NOISE_REGULARIZE",
    "DEFAULT_LR",
    "DEFAULT_CAMERA_LR",
    "DEFAULT_MIX_LAYER_START",
    "DEFAULT_MIX_LAYER_END",
    # Stdout-parsing regexes
    "CROPPED_FACE_COUNT_RE",
    "ITER_PROGRESS_RE",
    "YEAR_RE",
    "SIMPLE_FINAL_COPY_RE",
    "REPHOTO_CROP_FAIL_RE",
    "QUICK_FACE_DECISION_RE",
    "RETINA_FACE_BOX_RE",
    "CROP_ALIGN_BOX_RE",
    "FACE_SUFFIX_INDEX_RE",
    "NO_CROPS_CREATED_RE",
]
