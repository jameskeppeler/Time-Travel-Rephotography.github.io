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


###############################################################################
# Post-blend Photoshop-like Auto Color correction
#
# This stage runs AFTER the Color blend / recomposite is complete.  It operates
# on the final recomposited RGB image only — it never touches source layers,
# target layers, or intermediate masks.
#
# Processing order:
#   input images → color blend/composite → recomposited image
#       → auto_color_balance_after_blend() → saved/exported output
#
# The algorithm approximates Photoshop's "Auto Color" by:
#   1. Estimating dark/light reference points via percentiles (not min/max)
#      so that outlier pixels do not dominate.
#   2. Stretching tonal range per-channel to improve contrast.
#   3. Detecting a global midtone color cast from approximately neutral
#      (low-saturation) midtone pixels and applying per-channel correction
#      to neutralize it.
#   4. Blending the corrected result with the original at `strength` to
#      keep the effect moderate and natural-looking.
###############################################################################


def auto_color_balance_after_blend(
    image,
    shadow_percentile=0.5,
    highlight_percentile=99.5,
    strength=0.6,
    neutral_threshold=30.0,
    protect_shadows=True,
    protect_highlights=True,
):
    """Photoshop-like Auto Color correction — applied AFTER the color blend step.

    Args:
        image:               uint8 BGR image (H, W, 3) — the recomposited result.
        shadow_percentile:   percentile for dark reference (default 0.5).
        highlight_percentile: percentile for light reference (default 99.5).
        strength:            blend factor 0..1 between original and corrected
                             (0 = no change, 1 = full correction).
        neutral_threshold:   max per-channel spread (max-min) for a pixel to be
                             considered "neutral" when estimating midtone cast.
        protect_shadows:     if True, reduce correction near blacks to avoid
                             crushing shadow detail.
        protect_highlights:  if True, reduce correction near whites to avoid
                             blowing highlight detail.

    Returns:
        uint8 BGR image (H, W, 3) with Auto Color applied.
    """
    if image is None or image.size == 0:
        return image

    img = image.astype(np.float32)
    h, w, c = img.shape
    corrected = np.empty_like(img)

    # --- Step 1: Per-channel percentile stretch (tonal range) ---------------
    for ch in range(c):
        channel = img[:, :, ch].ravel()
        lo = np.percentile(channel, shadow_percentile)
        hi = np.percentile(channel, highlight_percentile)
        span = hi - lo
        if span < 1.0:
            # Channel is essentially flat — leave it alone.
            corrected[:, :, ch] = img[:, :, ch]
        else:
            corrected[:, :, ch] = (img[:, :, ch] - lo) * (255.0 / span)

    corrected = np.clip(corrected, 0, 255)

    # --- Step 2: Midtone color-cast neutralization --------------------------
    # Identify approximately neutral midtone pixels (low saturation, mid luma).
    luma = 0.299 * corrected[:, :, 2] + 0.587 * corrected[:, :, 1] + 0.114 * corrected[:, :, 0]  # BGR order
    ch_min = corrected.min(axis=2)
    ch_max = corrected.max(axis=2)
    spread = ch_max - ch_min  # per-pixel "saturation"

    midtone_mask = (
        (spread < neutral_threshold) &
        (luma > 40) & (luma < 215)
    )

    if np.count_nonzero(midtone_mask) > 64:
        # Compute mean color of neutral midtone pixels.
        neutral_pixels = corrected[midtone_mask]  # (N, 3)
        neutral_mean = neutral_pixels.mean(axis=0)  # (3,)
        gray_target = neutral_mean.mean()  # target: equal channels

        # Per-channel shift to neutralize the cast.
        cast_shift = gray_target - neutral_mean  # (3,)

        # Apply shift globally, scaled by strength.
        corrected = corrected + cast_shift[np.newaxis, np.newaxis, :]

    corrected = np.clip(corrected, 0, 255)

    # --- Step 3: Shadow / highlight protection ------------------------------
    # Build a soft mask that reduces correction intensity near pure black and
    # pure white so we do not crush shadows or blow highlights.
    if protect_shadows or protect_highlights:
        # Luminance of the *original* image (before correction).
        orig_luma = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        protection = np.ones((h, w), dtype=np.float32)

        if protect_shadows:
            # Ramp from 0 at luma=0 to 1 at luma=shadow_knee.
            shadow_knee = 30.0
            shadow_factor = np.clip(orig_luma / shadow_knee, 0, 1)
            protection *= shadow_factor

        if protect_highlights:
            # Ramp from 1 at luma=highlight_knee down to 0 at luma=255.
            highlight_knee = 225.0
            highlight_factor = np.clip((255.0 - orig_luma) / (255.0 - highlight_knee), 0, 1)
            protection *= highlight_factor

        # Blend: protected regions stay closer to original.
        protection_3ch = protection[:, :, np.newaxis]
        corrected = img + protection_3ch * (corrected - img)
        corrected = np.clip(corrected, 0, 255)

    # --- Step 4: Global strength blend with original ------------------------
    if strength < 1.0:
        corrected = img + strength * (corrected - img)
        corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)


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
