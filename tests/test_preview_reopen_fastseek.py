"""Regression tests: repeated Preview sessions and fast random frame seeks."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from modules import preview_reader


def wait_for(reader, index, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = reader.latest()
        if result is not None and result[1] == index:
            return result
        time.sleep(.01)
    raise AssertionError(f'No result for frame {index}')


class Capture:
    instances = []

    def __init__(self, path):
        self.path = path
        self.position = 0
        self.read_count = 0
        self.closed = False
        self.seek_indices = []
        self.__class__.instances.append(self)

    def isOpened(self):
        return True

    def get(self, key):
        if key == cv2.CAP_PROP_FPS:
            return 30.0
        if key == cv2.CAP_PROP_FRAME_COUNT:
            return 20000
        return 0

    def set(self, key, value):
        self.position = int(value)
        self.seek_indices.append(self.position)
        return True

    def read(self):
        position = self.position
        self.position += 1
        self.read_count += 1
        return True, np.full((64, 128, 3), position % 250, dtype=np.uint8)

    def release(self):
        self.closed = True


class PreviewReopenTests(unittest.TestCase):
    def setUp(self):
        preview_reader.clear_shared_video_preview_reader()
        Capture.instances.clear()

    def tearDown(self):
        preview_reader.clear_shared_video_preview_reader()

    def test_same_media_reopens_without_creating_new_capture(self):
        with patch.object(preview_reader.cv2, 'VideoCapture', Capture):
            reader = preview_reader.shared_video_preview_reader('same.mp4')
            first_gen = reader.request(4)
            original = wait_for(reader, 4)
            self.assertEqual(first_gen, original[0])
            self.assertIsNone(original[-1])
            self.assertEqual(1, len(Capture.instances))
            self.assertEqual(
                (20000, 30.0),
                preview_reader.shared_video_preview_metadata('same.mp4')
            )

            # The popup closes, but the session keeps the video decoder alive.
            reader.invalidate()
            self.assertFalse(Capture.instances[0].closed)
            reopened = preview_reader.shared_video_preview_reader('same.mp4')
            self.assertIs(reader, reopened)
            next_gen = reopened.request(4)
            cached = wait_for(reopened, 4)
            self.assertEqual(next_gen, cached[0])
            self.assertIs(cached[2], original[2])
            self.assertEqual(1, Capture.instances[0].read_count)
            self.assertEqual(1, len(Capture.instances))

            reopened.request(5)
            later = wait_for(reopened, 5)
            self.assertIsNone(later[-1])
            self.assertEqual(2, Capture.instances[0].read_count)

            preview_reader.clear_shared_video_preview_reader()
            until = time.monotonic() + 2
            while not Capture.instances[0].closed and time.monotonic() < until:
                time.sleep(.01)
            self.assertTrue(Capture.instances[0].closed)

    def test_changing_media_replaces_old_decoder(self):
        with patch.object(preview_reader.cv2, 'VideoCapture', Capture):
            old = preview_reader.shared_video_preview_reader('first.mp4')
            old.request(0)
            wait_for(old, 0)
            new = preview_reader.shared_video_preview_reader('second.mp4')
            self.assertIsNot(new, old)
            new.request(0)
            wait_for(new, 0)
            until = time.monotonic() + 2
            while not Capture.instances[0].closed and time.monotonic() < until:
                time.sleep(.01)
            self.assertTrue(Capture.instances[0].closed)
            self.assertEqual(2, len(Capture.instances))

    def test_large_seek_uses_fast_ffmpeg_while_nearby_frames_use_opencv(self):
        with patch.object(preview_reader.cv2, 'VideoCapture', Capture), \
             patch.object(preview_reader.FramePreviewReader, '_read_seek_ffmpeg',
                          side_effect=lambda index, fps:
                          np.full((64, 128, 3), index % 250, dtype=np.uint8)) as fast:
            reader = preview_reader.FramePreviewReader(video_path='video.mp4')
            try:
                reader.request(0)
                wait_for(reader, 0)
                reader.request(1)
                wait_for(reader, 1)
                self.assertEqual(0, fast.call_count)
                reader.request(3000)
                result = wait_for(reader, 3000)
                self.assertIsNone(result[-1])
                fast.assert_called_once_with(3000, 30.0)
                self.assertEqual([0], Capture.instances[0].seek_indices)
            finally:
                reader.close()

    def test_ffmpeg_seek_decodes_full_resolution_png_without_lossy_jpeg(self):
        original = np.zeros((30, 46, 3), dtype=np.uint8)
        original[3:12, 4:13] = [15, 48, 179]
        success, png = cv2.imencode('.png', original)
        self.assertTrue(success)

        class MockProcess:
            returncode = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def communicate(self, timeout):
                self.timeout = timeout
                return png.tobytes(), b''

        from modules import utilities
        with patch.object(utilities, '_start_ffmpeg', return_value=MockProcess()) as start:
            reader = preview_reader.FramePreviewReader(video_path='video.mp4')
            try:
                got = reader._read_seek_ffmpeg(900, 30.0)
            finally:
                reader.close()
        np.testing.assert_array_equal(original, got)
        args = start.call_args.args[0]
        self.assertEqual('ffmpeg', args[0])
        self.assertIn('-ss', args)
        self.assertEqual('30.000000', args[args.index('-ss') + 1])
        self.assertIn('png', args)
        self.assertIn('pipe:1', args)

    def test_ui_reuses_reader_and_releases_only_on_media_change(self):
        source = (Path(__file__).resolve().parents[1] / 'modules' / 'ui.py').read_text('utf-8')
        self.assertIn('shared_video_preview_reader(video_path, max_size=preview_size)', source)
        self.assertIn('frame_reader.invalidate()', source)
        self.assertIn('clear_shared_video_preview_reader()', source)
        self.assertIn('shared_video_preview_metadata(', source)
        self.assertIn('metadata = shared_video_preview_metadata(video_path)', source)
        self.assertNotIn('capture = cv2.VideoCapture(modules.globals.target_path)', source[
            source.index('def open_preview_frame_picker'):
            source.index('def show_processed_preview')
        ])
        self.assertIn("if video_path:", source)


if __name__ == '__main__':
    unittest.main()
