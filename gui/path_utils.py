"""Pure path / filesystem helpers used by the rephotography GUI.

Extracted from MainWindow as part of the Sprint-4 module split. These
functions take only their arguments — no MainWindow state — so they are
trivially unit-testable and reusable from the planned preview, pipeline,
and face-strip controllers. MainWindow keeps thin shim methods so the
hundreds of self.<method>(...) call sites continue to work unchanged.
"""

import os
from pathlib import Path
from typing import List, Optional

from gui.constants import FACE_SUFFIX_INDEX_RE, IMAGE_EXTENSIONS


def normalized_path_key(path_text) -> Optional[str]:
    """Canonicalize a path string for use as a dict key.

    Returns None for empty input. Resolves symlinks when possible; falls
    back to lower-cased text when resolution fails (e.g. dangling path).
    """
    if not path_text:
        return None
    try:
        return os.path.normcase(str(Path(path_text).resolve()))
    except Exception:
        return os.path.normcase(str(path_text))


def make_safe_base_name(base_text) -> str:
    """Sanitize a filename stem so it's safe to use across filesystems.

    Replaces every non-alphanumeric run with a single underscore and trims
    leading/trailing separators. Returns "input_image" for empty input so
    downstream code never has to special-case the empty stem.
    """
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


def list_image_files_in_dir(folder: Optional[Path]) -> List[Path]:
    """List image files in `folder`, sorted by the FACE_SUFFIX_INDEX
    integer when present (so face_3 comes before face_10) and otherwise
    by case-insensitive name.
    """
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


def result_preview_cache_key(image_path) -> str:
    """Build a stable cache key for a file: resolved-lowered path plus
    mtime and size so the key invalidates on any write or replacement.
    Falls back to just the lowered path text if stat fails.
    """
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


__all__ = [
    "normalized_path_key",
    "make_safe_base_name",
    "list_image_files_in_dir",
    "result_preview_cache_key",
]
