"""Responsive preview timeline reader, reused across Preview dialog openings.

A session owns one decoder per selected video. Closing the picker keeps it
idle (and retains the last ORIGINAL frame), rather than reopening the same
file on every Preview. Large random seeks use FFmpeg input-side timestamp
seeking; OpenCV remains the decoder for small/sequential moves and PNG images.
All blocking I/O runs in this worker, never in a Tk callback.
"""
import os
import queue
import subprocess
import threading

import cv2
import numpy as np


class FramePreviewReader:
    def __init__(self, *, video_path=None, frame_paths=None, image_path=None,
                 max_size=(800, 320), prefer_fast_seek=True):
        self.video_path = video_path
        self.frame_paths = tuple(frame_paths or ())
        self.image_path = image_path
        self.max_size = max_size
        self.prefer_fast_seek = prefer_fast_seek
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._pending_index = None
        self._generation = 0
        self._results = queue.Queue(maxsize=1)
        self._cache = None
        self._active_ffmpeg = None
        self._thread = threading.Thread(
            target=self._run, name='ReActorPreviewFrames', daemon=True
        )
        self._thread.start()

    def _cancel_active_seek(self):
        with self._lock:
            process = self._active_ffmpeg
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
            except (OSError, ValueError):
                pass

    def _queue_result(self, result):
        # Never queue a backlog of large decoded video images.
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
            try:
                self._results.put_nowait(result)
            except queue.Full:
                pass

    def request(self, index):
        """Request the most recent position, cancelling obsolete FFmpeg seeks."""
        if self._closed.is_set():
            return None
        index = max(0, int(index))
        with self._lock:
            self._generation += 1
            generation = self._generation
            cached = self._cache if self._cache and self._cache[0] == index else None
            self._pending_index = None if cached else index
        if cached:
            # Reopening Preview at the same frame requires no disk seek.
            self._queue_result((generation, index, cached[1], cached[2], None))
        else:
            self._cancel_active_seek()
            self._wake.set()
        return generation

    def latest(self):
        """Retrieve newest completed result without blocking the Tk thread."""
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def invalidate(self):
        """Detach a closed dialog without closing the session's video decoder."""
        with self._lock:
            self._generation += 1
            self._pending_index = None
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
        self._cancel_active_seek()

    def close(self):
        """Release the reader when the MEDIA changes or the application closes."""
        self._closed.set()
        self.invalidate()
        self._wake.set()

    def _read_seek_ffmpeg(self, index, fps):
        """Fast keyframe-based seek directly to the requested time.

        The FFmpeg result is decoded at source resolution and used unchanged
        by both the picker and the Face Swap preview; no quality loss.
        """
        if fps <= 0:
            raise RuntimeError('Video frame rate is not available.')
        from modules.utilities import _start_ffmpeg
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-ss', f'{index / fps:.6f}', '-i', self.video_path,
            '-map', '0:v:0', '-frames:v', '1',
            '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1'
        ]
        with _start_ffmpeg(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            with self._lock:
                self._active_ffmpeg = process
            try:
                data, stderr = process.communicate(timeout=15)
                if process.returncode:
                    detail = stderr.decode('utf-8', errors='replace')[-700:]
                    raise RuntimeError(f'FFmpeg seek failed: {detail}')
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError('Video seek exceeded 15 seconds.')
            finally:
                with self._lock:
                    if self._active_ffmpeg is process:
                        self._active_ffmpeg = None
        if not data:
            raise RuntimeError('FFmpeg did not return a frame.')
        frame = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise RuntimeError('Unable to decode FFmpeg preview frame.')
        return frame

    def _run(self):
        capture = None
        cv_position = None
        last_position = None
        fps = 0.0
        try:
            while not self._closed.is_set():
                if not self._wake.wait(timeout=0.2):
                    continue
                self._wake.clear()
                while not self._closed.is_set():
                    with self._lock:
                        index = self._pending_index
                        generation = self._generation
                        self._pending_index = None
                    if index is None:
                        break
                    error = None
                    original = None
                    thumbnail = None
                    try:
                        if self.video_path:
                            if capture is None:
                                capture = cv2.VideoCapture(self.video_path)
                                if not capture.isOpened():
                                    raise RuntimeError('Could not open the selected video.')
                                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                            # OpenCV can decode thousands of intervening frames
                            # on large GOP/VR videos. Use FFmpeg's input-side
                            # keyframe seek instead for large jumps. Retain the
                            # OpenCV path for near-sequential frames.
                            distance = abs(index - last_position) if last_position is not None else index
                            threshold = max(60, int(fps * 2)) if fps > 0 else 60
                            fast_seek = (
                                self.prefer_fast_seek and fps > 0
                                and (distance > threshold or
                                     (cv_position is None and index > threshold))
                            )
                            if fast_seek:
                                try:
                                    original = self._read_seek_ffmpeg(index, fps)
                                    cv_position = None
                                except (OSError, RuntimeError, cv2.error) as exc:
                                    # Fall back to the original OpenCV decoder
                                    # if FFmpeg is not available for this codec.
                                    if self._closed.is_set():
                                        break
                                    with self._lock:
                                        still_current = self._generation == generation
                                    if not still_current:
                                        continue
                                    print(f'[PREVIEW] FFmpeg seek unavailable: {exc}', flush=True)
                            if original is None:
                                if cv_position != index:
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                                ok, original = capture.read()
                                if not ok or original is None:
                                    raise RuntimeError('Could not decode this frame.')
                                cv_position = index + 1
                            last_position = index
                        elif self.frame_paths:
                            if index >= len(self.frame_paths):
                                raise RuntimeError('Selected frame is out of range.')
                            original = cv2.imread(self.frame_paths[index])
                        else:
                            original = cv2.imread(self.image_path) if self.image_path else None
                        if original is None:
                            raise RuntimeError('Could not read the selected frame.')
                        height, width = original.shape[:2]
                        if width < 1 or height < 1:
                            raise RuntimeError('Frame dimensions are invalid.')
                        factor = min(
                            self.max_size[0] / width,
                            self.max_size[1] / height, 1.0
                        )
                        if factor < 1.0:
                            thumbnail = cv2.resize(
                                original, (
                                    max(1, round(width * factor)),
                                    max(1, round(height * factor))
                                ), interpolation=cv2.INTER_AREA
                            )
                        else:
                            thumbnail = original
                        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                        error = str(exc)
                        cv_position = None
                        last_position = None
                        original = None
                        thumbnail = None
                    with self._lock:
                        current = generation == self._generation
                        if current and error is None:
                            self._cache = (index, original, thumbnail)
                    if self._closed.is_set():
                        break
                    if current:
                        self._queue_result((generation, index, original, thumbnail, error))
        finally:
            if capture is not None:
                capture.release()


_shared_lock = threading.Lock()
_shared_reader = None
_shared_key = None


def _video_key(path, size):
    path = os.path.normcase(os.path.abspath(path))
    try:
        info = os.stat(path)
        return path, info.st_size, info.st_mtime_ns, tuple(size)
    except OSError:
        return path, None, None, tuple(size)


def shared_video_preview_reader(video_path, max_size=(800, 320)):
    """One reader per video across multiple open/close Preview cycles."""
    global _shared_reader, _shared_key
    key = _video_key(video_path, max_size)
    with _shared_lock:
        if _shared_reader is not None and _shared_key == key:
            return _shared_reader
        if _shared_reader is not None:
            _shared_reader.close()
        _shared_reader = FramePreviewReader(
            video_path=video_path, max_size=max_size
        )
        _shared_key = key
        return _shared_reader


def clear_shared_video_preview_reader():
    """Called only when the selected media changes or the app exits."""
    global _shared_reader, _shared_key
    with _shared_lock:
        if _shared_reader is not None:
            _shared_reader.close()
        _shared_reader = None
        _shared_key = None
