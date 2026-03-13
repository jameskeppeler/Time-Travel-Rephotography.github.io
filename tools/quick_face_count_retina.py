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


def _landmarks_to_box(lms, img_w, img_h):
    """Convert 5-point landmarks to an approximate (x, y, w, h) face box."""
    xs = [float(p[0]) for p in lms]
    ys = [float(p[1]) for p in lms]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(6.0, max_x - min_x)
    span_y = max(6.0, max_y - min_y)

    x0 = min_x - 0.95 * span_x
    x1 = max_x + 0.95 * span_x
    y0 = min_y - 1.25 * span_y
    y1 = max_y + 1.05 * span_y

    x0 = max(0.0, min(float(img_w - 1), x0))
    y0 = max(0.0, min(float(img_h - 1), y0))
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))

    w = max(8, int(round(x1 - x0)))
    h = max(8, int(round(y1 - y0)))
    x = max(0, min(img_w - 1, int(round(x0))))
    y = max(0, min(img_h - 1, int(round(y0))))
    w = min(w, max(8, img_w - x))
    h = min(h, max(8, img_h - y))
    return (x, y, w, h)


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

    img_h, img_w = image_bgr.shape[:2]

    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch, unscales, paddings = as_batch([image_rgb], size=max(512, int(args.resize_size)), padding_mode="constant")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RetinaFace(strategy="all", vis=float(args.det_threshold))
        model.load(device=device)

        batch_tensor = as_tensor(batch, device=device)
        landmarks, indices = model.predict(batch_tensor)
        raw_count = int(len(indices))
    except Exception as exc:
        print(f"ERROR: probe failed: {exc}", file=sys.stderr)
        return 5

    cap = max(1, int(args.decision_cap))
    decision_count = min(raw_count, cap)
    print(f"QUICK_FACE_DECISION_COUNT={decision_count}")
    print(f"QUICK_FACE_RAW_COUNT={raw_count}")

    # Piggyback RETINA_FACE_BOX lines so the GUI can skip a separate box probe.
    if len(landmarks) > 0:
        unscale = float(unscales[0]) if len(unscales) else 1.0
        if unscale <= 0:
            unscale = 1.0
        pad_top = int(paddings[0][0]) if len(paddings) else 0
        pad_left = int(paddings[0][2]) if len(paddings) else 0

        boxes = []
        for i, lms in enumerate(landmarks):
            if i < len(indices) and int(indices[i]) != 0:
                continue
            mapped = []
            for p in lms:
                x = (float(p[0]) - pad_left) / unscale
                y = (float(p[1]) - pad_top) / unscale
                mapped.append((x, y))
            boxes.append(_landmarks_to_box(mapped, img_w, img_h))

        for i, (bx, by, bw, bh) in enumerate(boxes):
            print(f"RETINA_FACE_BOX_{i}={bx},{by},{bw},{bh}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
