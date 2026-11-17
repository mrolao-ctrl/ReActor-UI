"""No models required: tests for missing and corrupt PNG recovery."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import modules.globals
from modules.frame_recovery import expected_frame_paths, repair_frames, png_is_readable


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        modules.globals.frame_recovery_log.clear()
        modules.globals.frame_target_boxes.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save_png(self, number, value):
        path = self.folder / f'{number:04d}.png'
        Image.fromarray(np.full((6, 6, 3), value, dtype=np.uint8)).save(path)
        return path

    def test_repairs_missing_frame_and_inherits_selected_bbox(self):
        source = self.save_png(1, 42)
        self.save_png(3, 90)
        paths = expected_frame_paths(str(self.folder), expected_count=3)
        modules.globals.frame_target_boxes[os.path.abspath(str(source))] = [(1, 2, 3, 4)]
        self.assertEqual(1, repair_frames(paths, 'face_swapper'))
        self.assertTrue(png_is_readable(paths[1]))
        self.assertEqual([(1, 2, 3, 4)], modules.globals.frame_target_boxes[os.path.abspath(paths[1])])
        with Image.open(paths[1]) as image:
            self.assertEqual(42, int(np.asarray(image)[0, 0, 0]))

    def test_repairs_truncated_png(self):
        self.save_png(1, 40)
        corrupted = self.save_png(2, 70)
        corrupted.write_bytes(corrupted.read_bytes()[:20])
        paths = expected_frame_paths(str(self.folder), 2)
        self.assertEqual(1, repair_frames(paths, 'enhancer'))
        self.assertTrue(png_is_readable(str(corrupted)))
        self.assertEqual('0002.png', modules.globals.frame_recovery_log[0]['frame'])

    def test_all_corrupt_is_fatal(self):
        (self.folder / '0001.png').write_bytes(b'broken')
        with self.assertRaisesRegex(RuntimeError, 'No usable frames'):
            repair_frames(expected_frame_paths(str(self.folder)), 'extraction')

    def test_expected_count_includes_missing_tail(self):
        self.save_png(1, 10)
        self.assertEqual(4, len(expected_frame_paths(str(self.folder), 4)))


if __name__ == '__main__':
    unittest.main()
