"""Photoshop-style color blending for recompositing face results back into original images.

Implements the "Color" blend mode (hue + saturation from blend layer, luminance from base)
using the standard SetLum/ClipColor approach per CSS Compositing spec.
"""

import cv2
import numpy as np


def _lum(rgb_triple):
    """Compute relative luminance (CIE luma-like) from RGB triple.

    Using the standard rec. 601 weights, matching Photoshop intent.
    """
    return 0.299 * rgb_triple[0] + 0.587 * rgb_triple[1] + 0.114 * rgb_triple[2]


def _sat(rgb_triple):
    """Compute saturation (max - min of RGB channels)."""
    return np.max(rgb_triple) - np.min(rgb_triple)


def _set_lum(rgb, lum_target):
    """SetLum operation: shift RGB to target luminance while preserving hue/sat.

    Args:
        rgb: 3-element array or scalar broadcast to [R, G, B]
        lum_target: target luminance value

    Returns:
        3-element array with RGB shifted to lum_target
    """
    current_lum = _lum(rgb)
    delta = lum_target - current_lum
    return rgb + delta


def _clip_color(rgb):
    """ClipColor operation: ensure RGB values are in [0, 255] while preserving luminance.

    If any channel is out of range, compress the color toward gray while keeping
    lum constant.
    """
    lum = _lum(rgb)
    n = len(rgb)
    rgb_min = np.min(rgb)
    rgb_max = np.max(rgb)

    if rgb_min < 0:
        rgb = lum + (rgb - lum) * lum / (lum - rgb_min)
    if rgb_max > 255:
        rgb = lum + (rgb - lum) * (255 - lum) / (rgb_max - lum)

    return np.clip(rgb, 0, 255)


def color_blend(base_rgb, blend_rgb, alpha_mask=None):
    """Photoshop Color blend mode: hue + saturation from blend, luminance from base.

    Args:
        base_rgb: uint8 BGR image (H, W, 3) — provides luminance
        blend_rgb: uint8 BGR image (H, W, 3) — provides hue + saturation
        alpha_mask: optional float [0, 1] mask (H, W) — areas outside mask keep base

    Returns:
        uint8 BGR image (H, W, 3) with Color blend applied
    """
    # Ensure uint8 inputs
    base = base_rgb.astype(np.float32)
    blend = blend_rgb.astype(np.float32)

    h, w, _ = base.shape
    result = np.zeros((h, w, 3), dtype=np.float32)

    # Per-pixel color blend
    for i in range(h):
        for j in range(w):
            base_pix = base[i, j]
            blend_pix = blend[i, j]

            # Extract luminance from base
            base_lum = _lum(base_pix)

            # Take hue + saturation from blend
            # Reconstruct RGB with blend hue/sat but base luminance
            sat_blend = _sat(blend_pix)

            # Recolor blend to target luminance
            recolored = _set_lum(blend_pix, base_lum)
            recolored = _clip_color(recolored)

            if alpha_mask is not None and alpha_mask[i, j] < 1.0:
                # Blend toward base where alpha < 1
                alpha = alpha_mask[i, j]
                result[i, j] = base_pix * (1 - alpha) + recolored * alpha
            else:
                result[i, j] = recolored

    return np.clip(result, 0, 255).astype(np.uint8)


def recomposite_face_into_image(original_image, face_crop, bbox, use_color_blend=False):
    """Place a processed face crop back into the original image.

    Args:
        original_image: uint8 BGR (H, W, 3) — full original image
        face_crop: uint8 BGR (h, w, 3) — processed face output
        bbox: tuple (x, y, w, h) — bounding box where to place crop
        use_color_blend: if True, apply Color blend mode; else simple replacement

    Returns:
        uint8 BGR image (H, W, 3) with face blended back in
    """
    result = original_image.copy()
    x, y, w, h = bbox

    # Ensure crop fits within bounds
    crop_h, crop_w = face_crop.shape[:2]
    actual_w = min(w, crop_w, original_image.shape[1] - x)
    actual_h = min(h, crop_h, original_image.shape[0] - y)

    if actual_w <= 0 or actual_h <= 0:
        return result

    # Extract regions
    orig_roi = original_image[y:y+actual_h, x:x+actual_w]
    face_roi = face_crop[:actual_h, :actual_w]

    if use_color_blend:
        blended = color_blend(orig_roi, face_roi)
    else:
        blended = face_roi

    result[y:y+actual_h, x:x+actual_w] = blended
    return result
