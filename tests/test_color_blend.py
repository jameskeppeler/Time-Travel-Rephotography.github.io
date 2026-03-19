"""Unit tests for Photoshop-style Color blend implementation."""

import numpy as np
import unittest

from utils.color_blend import _lum, _sat, _set_lum, _clip_color, color_blend


class TestColorBlendMath(unittest.TestCase):
    """Test the low-level color blend math primitives."""

    def test_lum_grayscale(self):
        """Luminance of grayscale should equal its gray value."""
        rgb = np.array([128.0, 128.0, 128.0])
        lum = _lum(rgb)
        self.assertAlmostEqual(lum, 128.0, places=1)

    def test_lum_red_heavy(self):
        """Red should contribute most to luminance."""
        rgb_red = np.array([255.0, 0.0, 0.0])
        rgb_blue = np.array([0.0, 0.0, 255.0])
        lum_red = _lum(rgb_red)
        lum_blue = _lum(rgb_blue)
        self.assertGreater(lum_red, lum_blue)

    def test_sat_zero_for_gray(self):
        """Saturation of grayscale is zero."""
        rgb = np.array([100.0, 100.0, 100.0])
        sat = _sat(rgb)
        self.assertAlmostEqual(sat, 0.0, places=1)

    def test_sat_pure_hue(self):
        """Saturation of pure color is max - 0."""
        rgb = np.array([255.0, 0.0, 0.0])
        sat = _sat(rgb)
        self.assertAlmostEqual(sat, 255.0, places=1)

    def test_set_lum_preserves_saturation(self):
        """SetLum should preserve saturation (max - min relative to each other)."""
        rgb = np.array([100.0, 50.0, 150.0])
        original_sat = _sat(rgb)

        shifted = _set_lum(rgb, 200.0)
        new_lum = _lum(shifted)
        new_sat = _sat(shifted)

        self.assertAlmostEqual(new_lum, 200.0, places=0)
        self.assertAlmostEqual(new_sat, original_sat, places=0)

    def test_clip_color_in_bounds(self):
        """ClipColor should not modify in-bounds RGB."""
        rgb = np.array([100.0, 150.0, 200.0])
        clipped = _clip_color(rgb)
        np.testing.assert_array_almost_equal(clipped, rgb, decimal=1)

    def test_clip_color_negative(self):
        """ClipColor should bring negative values into [0, 255]."""
        rgb = np.array([100.0, -50.0, 200.0])
        clipped = _clip_color(rgb)
        self.assertGreaterEqual(np.min(clipped), 0.0)
        self.assertLessEqual(np.max(clipped), 255.0)

    def test_clip_color_overflow(self):
        """ClipColor should bring overflow values into [0, 255]."""
        rgb = np.array([100.0, 300.0, 200.0])
        clipped = _clip_color(rgb)
        self.assertGreaterEqual(np.min(clipped), 0.0)
        self.assertLessEqual(np.max(clipped), 255.0)


class TestColorBlendImage(unittest.TestCase):
    """Test full-image Color blend compositing."""

    def test_color_blend_same_images(self):
        """Blending an image with itself should return itself."""
        img = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]], dtype=np.uint8)
        result = color_blend(img, img)
        np.testing.assert_array_almost_equal(result, img, decimal=1)

    def test_color_blend_preserves_shape(self):
        """Color blend should preserve input shape."""
        base = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        blend = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = color_blend(base, blend)
        self.assertEqual(result.shape, base.shape)

    def test_color_blend_grayscale_base(self):
        """Color blend of grayscale base + colored blend should use base gray with blend hue."""
        base = np.full((10, 10, 3), 128, dtype=np.uint8)  # 50% gray
        blend = np.full((10, 10, 3), 0, dtype=np.uint8)
        blend[:, :] = [255, 0, 0]  # Red

        result = color_blend(base, blend)
        # Result should have luminance close to 128 (from base)
        # and reddish tint (from blend hue)
        result_lum = np.mean([0.299 * result[0, 0, 0] + 0.587 * result[0, 0, 1] + 0.114 * result[0, 0, 2]])
        self.assertAlmostEqual(result_lum, 128.0, delta=30)
        self.assertGreater(result[0, 0, 0], 100)  # Should have red component

    def test_color_blend_output_in_range(self):
        """Color blend output should always be uint8 [0, 255]."""
        base = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        blend = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = color_blend(base, blend)

        self.assertGreaterEqual(np.min(result), 0)
        self.assertLessEqual(np.max(result), 255)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
