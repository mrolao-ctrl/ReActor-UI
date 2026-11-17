"""Basic regression tests for safe frame I/O (run locally with unittest)."""
import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modules.frame_io import read_image, write_frame


class FrameIoTests(unittest.TestCase):
    def test_missing_png_reports_path(self):
        with self.assertRaisesRegex(RuntimeError, 'Cannot read'):
            read_image('this-file-does-not-exist.png', 'test frame')

    def test_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, '0001.png')
            original = np.zeros((8, 8, 3), dtype=np.uint8)
            changed = np.ones((8, 8, 3), dtype=np.uint8) * 127
            self.assertTrue(cv2.imwrite(path, original))
            write_frame(path, changed)
            self.assertTrue(np.array_equal(read_image(path), changed))
            self.assertEqual(
                [name for name in os.listdir(directory) if 'writing' in name],
                []
            )

    def test_failed_write_does_not_destroy_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, '0001.png')
            original = np.zeros((8, 8, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(path, original))
            with patch('modules.frame_io.cv2.imwrite', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'could not write'):
                    write_frame(path, original)
            self.assertTrue(np.array_equal(read_image(path), original))


if __name__ == '__main__':
    unittest.main()
