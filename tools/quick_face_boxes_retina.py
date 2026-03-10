#!/usr/bin/env python
"""Import-time RetinaFace face-box probe for GUI hover/overlay alignment.

Outputs lines in the form:
  RETINA_FACE_BOX_COUNT=<n>
  RETINA_FACE_BOX_<i>=<x>,<y>,<w>,<h>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Quick RetinaFace face-box probe")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--det-threshold", type=float, default=0.90, help="RetinaFace confidence threshold")
    parser.add_argument("--resize-size", type=int, default=1536, help="Interim detector resize size")
    parser.add_argument("--max-faces", type=int, default=0, help="Optional cap (0 means no cap)")
    return parser.parse_args()


def _landmarks_to_box(lms, img_w, img_h):
    xs = [float(p[0]) for p in lms]
    ys = [float(p[1]) for p in lms]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(6.0, max_x - min_x)
    span_y = max(6.0, max_y - min_y)

    # Expand around landmarks to approximate the crop-style face region.
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
    except Exception as exc:
        print(f"ERROR: probe failed: {exc}", file=sys.stderr)
        return 5

    if len(landmarks) == 0:
        print("RETINA_FACE_BOX_COUNT=0")
        return 0

    # as_batch returns per-image unscale and [top, bottom, left, right] padding.
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

    if args.max_faces and args.max_faces > 0:
        boxes = boxes[: int(args.max_faces)]

    print(f"RETINA_FACE_BOX_COUNT={len(boxes)}")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"RETINA_FACE_BOX_{i}={x},{y},{w},{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

