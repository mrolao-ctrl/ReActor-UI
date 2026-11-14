"""Adjustable face-only enhancement. No extra models and no network calls."""
from typing import Iterable, Sequence

import cv2
import numpy as np


def blend_selected_faces(
    original: np.ndarray,
    enhanced: np.ndarray,
    face_boxes: Iterable[Sequence[float]],
    strength: int = 40,
) -> np.ndarray:
    """Blend an enhanced frame only inside feathered face-shaped masks.

    A global blend changes the skin and features of *every* person. These masks
    preserve the background and unselected faces even when the underlying
    enhancer restored the entire frame.
    """
    strength = max(0, min(100, int(strength)))
    boxes = list(face_boxes or [])
    if strength == 0 or not boxes:
        return original
    if enhanced is None or enhanced.shape != original.shape:
        raise ValueError('Enhanced frame is missing or has a different size')
    height, width = original.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        rx = max(1, int((x2 - x1) * 0.60))
        ry = max(1, int((y2 - y1) * 0.65))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    feather = max(1.5, min(width, height) * 0.006)
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    mask = np.clip(mask * (strength / 100.0), 0.0, 1.0)[:, :, None]
    return np.clip(
        original.astype(np.float32) * (1.0 - mask) +
        enhanced.astype(np.float32) * mask,
        0, 255,
    ).astype(np.uint8)


def sharpen_frame(frame: np.ndarray) -> np.ndarray:
    """Light unsharp mask, without generative reconstruction of facial features."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.1)
    return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
