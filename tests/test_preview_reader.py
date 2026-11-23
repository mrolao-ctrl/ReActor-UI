"""Headless tests for responsive preview seeking on compressed videos."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from modules.preview_reader import FramePreviewReader


def await_result(reader, expected_index, timeout=3.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = reader.latest()
        if result and result[1] == expected_index:
            return result
        time.sleep(0.01)
    raise AssertionError(f'No preview result for index {expected_index}')


class FakeCapture:
    instances = []

    def __init__(self, path):
        self.path = path
        self.position = 0
        self.last = None
        self.released = False
        self.seeks = []
        self.reads = []
        self.__class__.instances.append(self)

    def isOpened(self):
        return True

    def get(self, key):
        if key == cv2.CAP_PROP_FPS:
            return 30.0
        if key == cv2.CAP_PROP_FRAME_COUNT:
            return 10000
        raise AssertionError(f'Unexpected property {key}')

    def set(self, key, value):
        if key != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError('Unexpected property')
        self.position = int(value)
        self.seeks.append(int(value))
        return True

    def read(self):
        index = self.position
        self.reads.append(index)
        self.position += 1
        return True, np.full((100, 200, 3), index % 255, dtype=np.uint8)

    def release(self):
        self.released = True


class AsyncPreviewReaderTests(unittest.TestCase):
    def setUp(self):
        FakeCapture.instances.clear()

    def test_one_capture_is_reused_for_seeks_and_sequential_frames(self):
        with patch('modules.preview_reader.cv2.VideoCapture', FakeCapture):
            reader = FramePreviewReader(video_path='demo.mp4', max_size=(80, 40),
                                        prefer_fast_seek=False)
            try:
                for index in (2, 3, 12):
                    reader.request(index)
                    _, returned, original, thumbnail, error = await_result(reader, index)
                    self.assertEqual(index, returned)
                    self.assertIsNone(error)
                    self.assertEqual((100, 200, 3), original.shape)
                    self.assertEqual((40, 80, 3), thumbnail.shape)
                self.assertEqual(1, len(FakeCapture.instances))
                self.assertEqual([2, 12], FakeCapture.instances[0].seeks)
                self.assertEqual([2, 3, 12], FakeCapture.instances[0].reads)
            finally:
                reader.close()
            limit = time.monotonic() + 2
            while not FakeCapture.instances[0].released and time.monotonic() < limit:
                time.sleep(0.01)
            self.assertTrue(FakeCapture.instances[0].released)

    def test_fast_scrubbing_drops_stale_frames_and_skips_intermediate_requests(self):
        reading = threading.Event()
        release = threading.Event()

        class SlowCapture(FakeCapture):
            def read(self):
                if not reading.is_set():
                    reading.set()
                    release.wait(timeout=3)
                return super().read()

        with patch('modules.preview_reader.cv2.VideoCapture', SlowCapture):
            reader = FramePreviewReader(video_path='demo.mp4', prefer_fast_seek=False)
            try:
                reader.request(5)
                self.assertTrue(reading.wait(timeout=2))
                reader.request(90)
                latest_generation = reader.request(110)
                release.set()
                result = await_result(reader, 110)
                self.assertEqual(latest_generation, result[0])
                self.assertIsNone(result[-1])
                self.assertEqual([5, 110], SlowCapture.instances[0].reads)
            finally:
                release.set()
                reader.close()

    def test_close_is_non_blocking_and_prevents_follow_up_reads(self):
        reading = threading.Event()
        release = threading.Event()

        class BlockedCapture(FakeCapture):
            def read(self):
                reading.set()
                release.wait(timeout=3)
                return super().read()

        with patch('modules.preview_reader.cv2.VideoCapture', BlockedCapture):
            reader = FramePreviewReader(video_path='demo.mp4', prefer_fast_seek=False)
            reader.request(7)
            self.assertTrue(reading.wait(timeout=2))
            try:
                start = time.monotonic()
                reader.close()
                self.assertLess(time.monotonic() - start, 0.25)
                self.assertIsNone(reader.request(8))
                release.set()
            finally:
                release.set()
                reader.close()
            limit = time.monotonic() + 2
            while not BlockedCapture.instances[0].released and time.monotonic() < limit:
                time.sleep(0.01)
            self.assertTrue(BlockedCapture.instances[0].released)

    def test_existing_png_sequence_and_single_image_work_without_video_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            first = str(Path(temp) / '0001.png')
            second = str(Path(temp) / '0002.png')
            cv2.imwrite(first, np.full((40, 50, 3), 10, dtype=np.uint8))
            cv2.imwrite(second, np.full((40, 50, 3), 30, dtype=np.uint8))
            with patch('modules.preview_reader.cv2.VideoCapture') as capture:
                reader = FramePreviewReader(frame_paths=[first, second])
                try:
                    reader.request(1)
                    _, _, original, thumbnail, error = await_result(reader, 1)
                    self.assertIsNone(error)
                    self.assertEqual(30, int(original[0, 0, 0]))
                    self.assertEqual(30, int(thumbnail[0, 0, 0]))
                finally:
                    reader.close()

                photo_reader = FramePreviewReader(image_path=first)
                try:
                    photo_reader.request(0)
                    _, _, original, thumbnail, error = await_result(photo_reader, 0)
                    self.assertIsNone(error)
                    self.assertEqual(10, int(original[0, 0, 0]))
                finally:
                    photo_reader.close()
                capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
