"""Responsive, local-only frame reader for the preview timeline.

OpenCV video seeking can be expensive, particularly on compressed/VR media.
Only this worker touches the VideoCapture. It opens the target once per popup,
coalesces requests while dragging, and sends completed frames back to the
Tk thread via a polling interface. Never update Tk widgets from this thread.
"""
import queue
import threading

import cv2


class FramePreviewReader:
    def __init__(self, *, video_path=None, frame_paths=None, image_path=None,
                 max_size=(800, 320)):
        self.video_path = video_path
        self.frame_paths = tuple(frame_paths or ())
        self.image_path = image_path
        self.max_size = max_size
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._pending_index = None
        self._generation = 0
        self._results = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run, name='ReActorPreviewFrames', daemon=True
        )
        self._thread.start()

    def request(self, index):
        """Request latest frame; older queued positions will be discarded."""
        if self._closed.is_set():
            return None
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._pending_index = max(0, int(index))
        self._wake.set()
        return generation

    def latest(self):
        """Retrieve newest completed result without blocking the UI thread."""
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def close(self):
        """Stop future requests; worker releases VideoCapture on exit.

        No blocking join: the OS video decoder might currently be seeking.
        """
        self._closed.set()
        self._wake.set()

    def _run(self):
        capture = None
        last_index = None
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
                            if last_index is None or index != last_index + 1:
                                # Seek only when necessary; subsequent frames
                                # are decoded without closing/reopening media.
                                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                            ok, original = capture.read()
                            if not ok or original is None:
                                raise RuntimeError('Could not decode this frame.')
                            last_index = index
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
                        factor = min(self.max_size[0] / width,
                                     self.max_size[1] / height, 1.0)
                        if factor < 1.0:
                            thumbnail = cv2.resize(
                                original, (max(1, round(width * factor)),
                                           max(1, round(height * factor))),
                                interpolation=cv2.INTER_AREA
                            )
                        else:
                            thumbnail = original
                        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                        error = str(exc)
                        last_index = None
                        original = None
                        thumbnail = None
                    with self._lock:
                        current = generation == self._generation
                    if self._closed.is_set():
                        break
                    if current:
                        result = (generation, index, original, thumbnail, error)
                        # Never create a backlog of large image arrays.
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
        finally:
            if capture is not None:
                capture.release()
