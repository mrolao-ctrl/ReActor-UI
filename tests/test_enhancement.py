"""No model downloads required: tests for selective enhancer blending."""
import unittest
import numpy as np

from modules.enhancement import blend_selected_faces, sharpen_frame


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full((100, 200, 3), 20, dtype=np.uint8)
        self.changed = np.full((100, 200, 3), 220, dtype=np.uint8)
        self.box = [(30, 20, 70, 80)]

    def test_strength_zero_is_unchanged(self):
        out = blend_selected_faces(self.base, self.changed, self.box, 0)
        self.assertTrue(np.array_equal(out, self.base))

    def test_unselected_region_is_unchanged(self):
        out = blend_selected_faces(self.base, self.changed, self.box, 100)
        np.testing.assert_array_equal(out[:, 140:], self.base[:, 140:])

    def test_partial_strength_preserves_more_of_original(self):
        low = blend_selected_faces(self.base, self.changed, self.box, 40)
        high = blend_selected_faces(self.base, self.changed, self.box, 100)
        self.assertGreater(int(low[50, 50, 0]), 20)
        self.assertLess(int(low[50, 50, 0]), int(high[50, 50, 0]))

    def test_missing_selection_is_identity(self):
        out = blend_selected_faces(self.base, self.changed, [], 100)
        self.assertTrue(np.array_equal(out, self.base))

    def test_sharpen_does_not_change_dimensions(self):
        self.assertEqual(sharpen_frame(self.base).shape, self.base.shape)


if __name__ == '__main__':
    unittest.main()
