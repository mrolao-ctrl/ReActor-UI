"""Cooperative stop control for the current desktop processing job.

Cancellation does not destroy or reset the GUI's selected video, face mappings,
source photos or settings. The GUI must wait for the worker to finish before it
unlocks a new job, because native ONNX/GFPGAN calls cannot be killed safely
from a Python thread.
"""
import subprocess
import threading


class ProcessingAborted(Exception):
    """Raised by worker threads when the user stops their current job."""


_lock = threading.RLock()
_stop_event = threading.Event()
_active = False
_processes = set()


def begin() -> None:
    """Start exactly one job; never reset a cancellation still in progress."""
    global _active
    with _lock:
        if _active:
            raise RuntimeError('A processing job is already active')
        _stop_event.clear()
        _processes.clear()
        _active = True


def active() -> bool:
    with _lock:
        return _active


def requested() -> bool:
    return _stop_event.is_set()


def check_cancelled() -> None:
    if _stop_event.is_set():
        raise ProcessingAborted('Processing aborted by user')


def register_process(process: subprocess.Popen) -> None:
    """Track FFmpeg so Abort can stop it even while Python is reading a pipe."""
    with _lock:
        if _active:
            _processes.add(process)
            should_stop = _stop_event.is_set()
        else:
            should_stop = False
    if should_stop:
        _terminate(process)


def unregister_process(process: subprocess.Popen) -> None:
    with _lock:
        _processes.discard(process)


def _terminate(process: subprocess.Popen) -> None:
    try:
        if process.poll() is None:
            process.terminate()
    except (OSError, ValueError, AttributeError):
        pass


def request_abort() -> bool:
    """Signal every frame worker and terminate active FFmpeg subprocesses.

    The GUI deliberately stays locked until the worker raises ProcessingAborted
    or returns, and only then changes the Abort button back to Close.
    """
    with _lock:
        if not _active:
            return False
        _stop_event.set()
        processes = tuple(_processes)
    for process in processes:
        _terminate(process)
    return True


def finish() -> None:
    """Only called by the worker after all its subprocesses/threads have ended."""
    global _active
    with _lock:
        _active = False
        _processes.clear()
        # The next begin() clears the stop flag. Keeping it until then
        # ensures late callbacks from this job cannot resume processing.
