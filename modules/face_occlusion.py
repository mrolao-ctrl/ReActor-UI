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
LANDMARK_TEMPLATE_3 = LANDMARK_TEMPLATE[:3].copy()
LANDMARK_MODES = ('auto', '2-point-left', '2-point-right', '3-point', '5-point')

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


def _upper_face_is_valid(points: np.ndarray) -> bool:
    """Check three non-collinear eye/nose points before estimating alignment."""
    if points.shape != (3, 2) or not np.isfinite(points).all():
        return False
    eye_vector = points[1] - points[0]
    eye_length_sq = float(np.dot(eye_vector, eye_vector))
    if eye_length_sq < 16.0:  # At least 4 pixels between the eyes.
        return False
    nose_vector = points[2] - points[0]
    cross = float(
        eye_vector[0] * nose_vector[1] -
        eye_vector[1] * nose_vector[0]
    )
    nose_height_ratio = abs(cross) / eye_length_sq
    nose_horizontal_ratio = float(np.dot(nose_vector, eye_vector)) / eye_length_sq
    return (0.10 <= nose_height_ratio <= 1.5 and
            -0.6 <= nose_horizontal_ratio <= 1.6)


def _mouth_geometry_reliable(points: np.ndarray, upper_matrix: np.ndarray) -> bool:
    """A geometry-only sanity check, NOT a mouth visibility classifier.

    A microphone may hide the mouth while a detector still guesses plausible
    coordinates. In that case select 3-point explicitly for the best test.
    """
    if points.shape != (5, 2) or not np.isfinite(points[3:]).all():
        return False
    mouth = cv2.transform(points[3:][None, :, :], upper_matrix)[0]
    reference = LANDMARK_TEMPLATE[3:]
    target_mouth_width = float(np.linalg.norm(reference[1] - reference[0]))
    mouth_width = float(np.linalg.norm(mouth[1] - mouth[0]))
    # Rule out collapsed, wildly exaggerated, inverted or displaced mouths.
    if not 0.3 * target_mouth_width <= mouth_width <= 1.7 * target_mouth_width:
        return False
    if np.any(mouth[:, 1] <= LANDMARK_TEMPLATE[2, 1] + 8.0):
        return False
    return float(np.max(np.linalg.norm(mouth - reference, axis=1))) <= 36.0


def _two_point_alignment(points: np.ndarray, side: str) -> np.ndarray:
    """Align a visible eye and mouth corner using a 2D similarity transform.

    Side refers to the landmark index (image-left or image-right), not to
    anatomical handedness. This is an experimental fallback for profiles.
    It cannot determine 3D head pose or landmark visibility from two points.
    """
    if points.shape != (5, 2):
        raise ValueError('Two-point mode requires five labelled landmark slots')
    indices = (0, 3) if side == 'left' else (1, 4)
    source = points[list(indices)]
    if not np.isfinite(source).all():
        raise ValueError(f'Two-point {side} eye/mouth landmarks are unavailable')
    target = LANDMARK_TEMPLATE[list(indices)]
    src_vector = source[1] - source[0]
    dst_vector = target[1] - target[0]
    baseline_sq = float(np.dot(src_vector, src_vector))
    if baseline_sq < 64.0:
        raise ValueError('Two-point landmarks are too close for safe alignment')
    a = float(np.dot(src_vector, dst_vector)) / baseline_sq
    b = float(np.cross(src_vector, dst_vector)) / baseline_sq
    scale = float(np.hypot(a, b))
    if not 0.25 <= scale <= 10.0:
        raise ValueError('Two-point alignment scale is implausible')
    rotation_scale = np.array([[a, -b], [b, a]], dtype=np.float32)
    offset = target[0] - rotation_scale @ source[0]
    return np.column_stack((rotation_scale, offset)).astype(np.float32)


def _three_point_alignment(points: np.ndarray) -> np.ndarray:
    """Fit upper face, refusing poor correspondence to the reference template."""
    if not _upper_face_is_valid(points[:3]):
        raise ValueError('Eye/nose landmarks are missing or unreliable')
    matrix, _ = cv2.estimateAffinePartial2D(
        points[:3], LANDMARK_TEMPLATE_3, method=cv2.LMEDS
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError('Could not align the visible upper face')
    projected = cv2.transform(points[:3][None, :, :], matrix)[0]
    if float(np.max(np.linalg.norm(projected - LANDMARK_TEMPLATE_3, axis=1))) > 18.0:
        raise ValueError('Upper face landmarks cannot be aligned reliably')
    return matrix


def _fixed_alignment(points: np.ndarray, mode: str, strict_auto: bool = False) -> np.ndarray:
    """Produce a candidate without switching to another mode.

    Manual modes keep their previous semantics. Auto additionally rejects a
    five-point fit with inconsistent mouth geometry or large reprojection error.
    """
    if mode in ('2-point-left', '2-point-right'):
        return _two_point_alignment(points, mode.rsplit('-', 1)[1])
    if mode == '3-point':
        return _three_point_alignment(points)
    if mode != '5-point':
        raise ValueError(f'Invalid fixed alignment mode: {mode}')
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('5-point mode requires all five valid facial landmarks')
    upper_matrix = _three_point_alignment(points)
    if strict_auto and not _mouth_geometry_reliable(points, upper_matrix):
        raise ValueError('Mouth landmarks inconsistent with upper-face alignment')
    matrix, _ = cv2.estimateAffinePartial2D(
        points, LANDMARK_TEMPLATE, method=cv2.LMEDS
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError('Could not align the five facial landmarks')
    if strict_auto:
        projected = cv2.transform(points[None, :, :], matrix)[0]
        error = np.linalg.norm(projected - LANDMARK_TEMPLATE, axis=1)
        if float(np.max(error)) > 30.0 or float(np.mean(error)) > 15.0:
            raise ValueError('Five-point fit has excessive reprojection error')
    return matrix


def alignment_quality_score(points: np.ndarray, matrix: np.ndarray, mode: str) -> float:
    """Rank *plausible* alignments; smaller is preferable.

    The score penalizes using fewer points and compares the *unused* landmarks
    for competing two-point fits. Two points always fit their own pair exactly,
    so their fit residual alone is NOT a useful confidence measure. The score
    is a geometric heuristic, not a face visibility/confidence prediction.
    """
    if mode == '5-point':
        indices = (0, 1, 2, 3, 4)
        tier_penalty = 0.0
    elif mode == '3-point':
        indices = (0, 1, 2)
        tier_penalty = 16.0
    elif mode == '2-point-left':
        indices = (0, 3)
        tier_penalty = 40.0
    elif mode == '2-point-right':
        indices = (1, 4)
        tier_penalty = 40.0
    else:
        raise ValueError(f'Invalid alignment mode for scoring: {mode}')

    projected = cv2.transform(points[list(indices)][None, :, :], matrix)[0]
    fit_error = float(np.mean(
        np.linalg.norm(projected - LANDMARK_TEMPLATE[list(indices)], axis=1)
    ))
    score = tier_penalty + fit_error

    if mode.startswith('2-point'):
        # Other points are potentially occluded or hallucinated, so give them
        # weak, clipped influence. This can help distinguish left vs right
        # without allowing unreliable landmarks to veto a working pair.
        held_out = [i for i in range(points.shape[0])
                    if i not in indices and np.isfinite(points[i]).all()]
        if held_out:
            predicted = cv2.transform(points[held_out][None, :, :], matrix)[0]
            distances = np.linalg.norm(
                predicted - LANDMARK_TEMPLATE[held_out], axis=1
            )
            score += float(np.mean(np.minimum(distances, 96.0))) / 8.0
        # Prefer less distorted crops when held-out landmarks cannot decide.
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        angle = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
        score += min(8.0, 2.5 * abs(float(np.log(scale))))
        score += min(8.0, max(0.0, angle - 20.0) / 8.0)
    return score


_AUTO_LOG_LOCK = threading.Lock()
_AUTO_LOG_COUNTS = {}


def _log_auto_fallback(mode: str, score: float) -> None:
    if mode == '5-point':
        return
    with _AUTO_LOG_LOCK:
        count = _AUTO_LOG_COUNTS.get(mode, 0) + 1
        _AUTO_LOG_COUNTS[mode] = count
    if count <= 5 or count % 100 == 0:
        print(
            f'[REACTOR.OCCLUSION] Auto chose {mode} '
            f'(geometry score {score:.1f}, occurrence {count})',
            flush=True
        )


def estimate_alignment_matrix(landmarks: np.ndarray, mode: str = 'auto') -> tuple:
    """Return (2D similarity transform, selected mode) for XSeg.

    Auto evaluates 5-point, 3-point, 2-point-left, 2-point-right and
    selects the best geometrically plausible candidate by score.
    It may still make a poor choice for extreme profiles: InsightFace's
    coordinates do not expose landmark-visibility confidence.
    """
    if mode not in LANDMARK_MODES:
        raise ValueError(f'Invalid occlusion landmark mode: {mode}')
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape not in ((3, 2), (5, 2)):
        raise ValueError('Expected three or five labelled facial landmark slots')
    if mode != 'auto':
        return _fixed_alignment(points, mode), mode

    candidates = []
    reasons = []
    for candidate_mode in ('5-point', '3-point', '2-point-left', '2-point-right'):
        try:
            matrix = _fixed_alignment(
                points, candidate_mode, strict_auto=True
            )
            score = alignment_quality_score(points, matrix, candidate_mode)
            if not np.isfinite(score):
                raise ValueError('Non-finite alignment score')
            candidates.append((score, candidate_mode, matrix))
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            reasons.append(f'{candidate_mode}: {error}')

    if not candidates:
        raise ValueError('No reliable XSeg alignment: ' + '; '.join(reasons))
    score, selected, matrix = min(candidates, key=lambda result: result[0])
    _log_auto_fallback(selected, score)
    return matrix, selected



def get_visibility_mask(frame: np.ndarray, face: Any, mode: str = None) -> np.ndarray:
    """Return float32 0..1 mask of visible (replaceable) face pixels.

    Input is the *original* frame, before swapping or enhancing. Visibility
    zero means preserve original pixels. The XSeg mask is predicted in
    an aligned face crop then warped back into the original frame.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected a BGR video frame')
    landmarks = np.asarray(getattr(face, 'kps', None), dtype=np.float32)
    if mode is None:
        mode = modules.globals.occlusion_landmark_mode
    matrix, used_mode = estimate_alignment_matrix(landmarks, mode)
    # The aligned crop is what XSeg sees; it never receives landmarks.
    # Forced two- or three-point modes ignore other detector landmarks.
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
