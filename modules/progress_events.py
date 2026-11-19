"""Thread-safe, optional progress publisher for the Tk GUI.

Console tqdm stays independent. The callback MUST only queue UI events: do not
touch Tk widgets from a processing thread.
"""
import threading
import time

_lock = threading.Lock()
_sink = None
_stage = None
_started = 0.0
_last_published = 0.0
_last_done = 0


def set_sink(callback):
    global _sink
    with _lock:
        _sink = callback


def report(stage: str, completed: int, total: int) -> None:
    """Publish at most ~5 updates/s; always include first and final frames."""
    global _stage, _started, _last_published, _last_done
    now = time.monotonic()
    done = max(0, int(completed))
    maximum = max(0, int(total))
    if maximum:
        done = min(done, maximum)
    with _lock:
        if _sink is None:
            return
        changed = stage != _stage or done < _last_done
        if changed:
            _stage = stage
            _started = now
            _last_published = 0.0
        _last_done = done
        if not changed and done != maximum and now - _last_published < 0.20:
            return
        _last_published = now
        elapsed = max(0.0, now - _started)
        rate = done / elapsed if elapsed > 0.05 and done else 0.0
        remaining = (maximum - done) / rate if rate > 0 else None
        sink = _sink
        payload = {
            'type': 'progress', 'stage': stage, 'done': done, 'total': maximum,
            'rate': rate, 'eta': remaining,
        }
    sink(payload)


def reset() -> None:
    global _stage, _started, _last_published, _last_done
    with _lock:
        _stage = None
        _started = 0.0
        _last_published = 0.0
        _last_done = 0
