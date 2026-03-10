#!/usr/bin/env python
"""Fast import-time RetinaFace probe for GUI run-button hinting.

This script intentionally avoids crop generation and only emits a count decision:
0 (no face), 1 (single face), or 2+ (multi-face, capped by --decision-cap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Quick RetinaFace face-count probe")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--det-threshold", type=float, default=0.90, help="RetinaFace confidence threshold")
    parser.add_argument("--resize-size", type=int, default=1536, help="Interim resize size for detector batching")
    parser.add_argument(
        "--decision-cap",
        type=int,
        default=2,
        help="Maximum count value to emit for decision routing (default: 2 => 0/1/2+)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if (not image_path.exists()) or (not image_path.is_file()):
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2

    try:
        import cv2
        import torch
        from face_crop_plus.models import RetinaFace
        from face_crop_plus.utils import as_batch, as_tensor
    except Exception as exc:
        print(f"ERROR: import failed: {exc}", file=sys.stderr)
        return 3

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"ERROR: could not read image: {image_path}", file=sys.stderr)
        return 4

    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch, _, _ = as_batch([image_rgb], size=max(512, int(args.resize_size)), padding_mode="constant")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RetinaFace(strategy="all", vis=float(args.det_threshold))
        model.load(device=device)

        batch_tensor = as_tensor(batch, device=device)
        _, indices = model.predict(batch_tensor)
        raw_count = int(len(indices))
    except Exception as exc:
        print(f"ERROR: probe failed: {exc}", file=sys.stderr)
        return 5

    cap = max(1, int(args.decision_cap))
    decision_count = min(raw_count, cap)
    print(f"QUICK_FACE_DECISION_COUNT={decision_count}")
    print(f"QUICK_FACE_RAW_COUNT={raw_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
