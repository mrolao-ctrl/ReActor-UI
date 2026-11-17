"""Optional, local, occlusion-aware blending for *non-explicit* test footage.

Uses FaceFusion's XSeg-1 segmentation model (ONNX, 256x256) to estimate
which parts of a face are visible. A hand / glasses / foreground object can
then remain from the input frame instead of being painted over by the swap.

The default path is OFF. All inference stays on the user's computer.
Failure is conservative: return the pre-swap frame, not a full-face overlay.
This cannot recover fully hidden landmarks or guarantee every occlusion.
"""
import os
import threading
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

import modules.globals
from modules.utilities import conditional_download, resolve_relative_path

MODEL_FILENAME = 'xseg_1.onnx'
MODEL_URL = ('https://github.com/facefusion/facefusion-assets/releases/'
             'download/models-3.1.0/xseg_1.onnx')
MODEL_SIZE = 256

# Standard ArcFace 128 five-point template scaled to 256x256.
# Points: left eye, right eye, nose, left mouth corner, right mouth corner.
LANDMARK_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float32) * MODEL_SIZE

_MODEL_SESSION = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_WARNINGS = 0
_WARNINGS_LOCK = threading.Lock()


def model_path() -> str:
    return resolve_relative_path('../models/' + MODEL_FILENAME)


def ensure_model() -> str:
    """Download only if the user enables the optional occlusion feature."""
    path = model_path()
    if not os.path.isfile(path):
        print('[REACTOR.OCCLUSION] Downloading optional XSeg model...')
        conditional_download(os.path.dirname(path), [MODEL_URL])
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f'Occlusion model missing or empty: {path}')
    return path


def get_occluder_session() -> Any:
    global _MODEL_SESSION
    with _MODEL_LOCK:
        if _MODEL_SESSION is None:
            import onnxruntime
            path = ensure_model()
            providers = modules.globals.execution_providers or ['CPUExecutionProvider']
            _MODEL_SESSION = onnxruntime.InferenceSession(path, providers=providers)
    return _MODEL_SESSION


def get_visibility_mask(frame: np.ndarray, face: Any) -> np.ndarray:
    """Return float32 0..1 mask of visible (replaceable) face pixels.

    Input is the *original* frame, before swapping or enhancing. Visibility
    zero means preserve original pixels. The XSeg mask is predicted in
    an aligned face crop then warped back into the original frame.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected a BGR video frame')
    landmarks = np.asarray(getattr(face, 'kps', None), dtype=np.float32)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise ValueError('Cannot align an occluded face without five valid landmarks')

    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks, LANDMARK_TEMPLATE, method=cv2.LMEDS
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError('Could not estimate face alignment')
    aligned = cv2.warpAffine(
        frame, matrix, (MODEL_SIZE, MODEL_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # XSeg-1 expects NHWC BGR float32 in 0..1, not InsightFace's NCHW.
    prepared = np.expand_dims(aligned.astype(np.float32) / 255.0, axis=0)
    session = get_occluder_session()
    name = session.get_inputs()[0].name
    with _INFERENCE_LOCK:
        prediction = session.run(None, {name: prepared})[0]
    prediction = np.squeeze(np.asarray(prediction))
    if prediction.shape != (MODEL_SIZE, MODEL_SIZE) or not np.isfinite(prediction).all():
        raise ValueError('Unexpected XSeg mask shape or non-finite values')

    # XSeg returns approximately 1 for visible face and 0 for an occluder.
    visible = cv2.GaussianBlur(
        np.clip(prediction, 0.0, 1.0).astype(np.float32),
        (0, 0), sigmaX=3.0
    )
    visible = np.clip((visible - 0.45) / 0.55, 0.0, 1.0)
    inverse = cv2.invertAffineTransform(matrix)
    return cv2.warpAffine(
        visible, inverse, (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
    )


def combine_visibility(original: np.ndarray, edited: np.ndarray,
                       visibility: np.ndarray) -> np.ndarray:
    """Restore foreground pixels for visibility=0; keep edited for 1."""
    if original is None or edited is None or original.shape != edited.shape:
        raise ValueError('Original and edited frames must have the same shape')
    if visibility.shape != original.shape[:2]:
        raise ValueError('Visibility mask size does not match frame')
    alpha = np.clip(visibility, 0, 1).astype(np.float32)[:, :, None]
    result = original.astype(np.float32) * (1.0 - alpha)
    result += edited.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def warn_once(message: str) -> None:
    global _WARNINGS
    with _WARNINGS_LOCK:
        _WARNINGS += 1
        count = _WARNINGS
    if count <= 5 or count % 100 == 0:
        print(f'[REACTOR.OCCLUSION] {message} (occurrence {count})', flush=True)


def select_faces_by_boxes(faces: Iterable[Any],
                          boxes: Iterable[Sequence[float]],
                          minimum_iou: float = 0.15) -> Optional[list]:
    """Pair already-swapped face boxes with current detections (not identity).

    None means at least one face is missing/ambiguous, so the enhancer should
    leave the frame alone instead of painting over an occluded foreground.
    """
    faces = list(faces or [])
    boxes = list(boxes or [])
    used = set()
    chosen = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        scores = []
        for index, face in enumerate(faces):
            if index in used:
                continue
            fx1, fy1, fx2, fy2 = map(float, face.bbox)
            intersection = max(0, min(x2, fx2) - max(x1, fx1)) * max(
                0, min(y2, fy2) - max(y1, fy1)
            )
            denom = area + max(0, fx2 - fx1) * max(0, fy2 - fy1) - intersection
            scores.append((intersection / denom if denom > 0 else 0.0, index))
        scores.sort(reverse=True)
        if not scores or scores[0][0] < minimum_iou:
            return None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.05:
            return None
        used.add(scores[0][1])
        chosen.append(faces[scores[0][1]])
    return chosen
