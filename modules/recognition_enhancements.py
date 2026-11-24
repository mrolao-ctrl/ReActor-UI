"""Opt-in adaptive face detection and conservative short-window tracking.

Rotated detections are mapped back to original pixels (including the five
keypoints required by INSwapper). A new rotated match must still pass normal
identity thresholds. Temporal continuity may rescue only a nearby detection
for a recently *confidently* identified person. Nothing is saved to disk.
"""
import copy
import math
import cv2
import numpy as np

from modules.face_selector import MIN_MARGIN, MIN_SIMILARITY
from modules.identity_references import person_scores


ADAPTIVE_ANGLES = (-30, 30, -45, 45, -90, 90)
TRACK_MIN_SCORE = 0.18
TRACK_MIN_IOU = 0.42
TRACK_MAX_GAP = 2


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return iw * ih / max(1e-9, area_a + area_b - iw * ih)


def _rotate_with_transform(frame, angle):
    height, width = frame.shape[:2]
    radians = math.radians(angle)
    rotated_w = math.ceil(abs(width * math.cos(radians))
                          + abs(height * math.sin(radians)))
    rotated_h = math.ceil(abs(width * math.sin(radians))
                          + abs(height * math.cos(radians)))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    matrix[0, 2] += (rotated_w - width) / 2
    matrix[1, 2] += (rotated_h - height) / 2
    rotated = cv2.warpAffine(
        frame, matrix, (rotated_w, rotated_h),
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, cv2.invertAffineTransform(matrix)


def _unrotate_face(face, inverse, original_shape):
    """Translate face bbox + *real* alignment keypoints back to source pixels."""
    points = np.asarray(face.kps, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return None
    box = np.asarray(face.bbox, dtype=np.float32)
    if box.shape[0] < 4 or not np.isfinite(box[:4]).all():
        return None
    x1, y1, x2, y2 = box[:4]
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    linear = inverse[:, :2].astype(np.float32)
    offset = inverse[:, 2].astype(np.float32)
    old_corners = corners @ linear.T + offset
    left, top = np.min(old_corners, axis=0)
    right, bottom = np.max(old_corners, axis=0)
    height, width = original_shape[:2]
    left, right = max(0.0, float(left)), min(float(width), float(right))
    top, bottom = max(0.0, float(top)), min(float(height), float(bottom))
    if right - left < 4 or bottom - top < 4:
        return None
    # The face remains an InsightFace Face with all source attributes (e.g.
    # normed_embedding), but alignment coordinates are in the original frame.
    result = copy.copy(face)
    result.bbox = np.asarray([left, top, right, bottom], dtype=np.float32)
    result.kps = points @ linear.T + offset
    return result


def _best_identity(face, assignments):
    scores = person_scores(face, assignments)
    ranked = sorted(
        ((score, i) for i, score in enumerate(scores) if score is not None),
        reverse=True
    )
    if not ranked:
        return None
    top_score, top_index = ranked[0]
    if top_score < MIN_SIMILARITY:
        return None
    if len(ranked) > 1 and top_score - ranked[1][0] < MIN_MARGIN:
        return None
    return top_index


def find_rotated_matches(frame, get_faces, assignments, accepted,
                         check_cancelled=None):
    """Return fresh (assignment index, original-coordinate face) matches.

    'accepted' is the current identity->face dict from the original
    orientation; it is never displaced by a rotated inference.
    """
    if not assignments or len(accepted) >= len(assignments):
        return {}
    extra = {}
    protected = [face.bbox for face in accepted.values()]
    for angle in ADAPTIVE_ANGLES:
        if check_cancelled:
            check_cancelled()
        rotated, inverse = _rotate_with_transform(frame, angle)
        for found in get_faces(rotated) or []:
            face = _unrotate_face(found, inverse, frame.shape)
            if face is None:
                continue
            index = _best_identity(face, assignments)
            if index is None or index in accepted or index in extra:
                continue
            if any(bbox_iou(face.bbox, bbox) > 0.25 for bbox in protected):
                continue
            if any(bbox_iou(face.bbox, current.bbox) > 0.25
                   for current in extra.values()):
                continue
            extra[index] = face
        if len(accepted) + len(extra) >= len(assignments):
            break
    return extra


class TemporalIdentityTracker:
    """Two-frame, box+identity-constrained reuse of a *confirmed* match.

    Explicit frame indices are mandatory: out-of-order threaded processing
    must not reuse state. The caller processes frames sequentially if enabled.
    """

    def __init__(self):
        self._regions = {}
        self._last_frame = {}

    def reset(self):
        self._regions.clear()
        self._last_frame.clear()

    def augment(self, region, frame_index, detections, identities, assignments):
        """Return rescued unmatched identities, updating confirmed anchors."""
        if frame_index is None:
            return {}
        last = self._last_frame.get(region)
        if last is not None and frame_index != last + 1:
            self._regions.pop(region, None)
        self._last_frame[region] = frame_index
        tracked = self._regions.setdefault(region, {})

        # Only confident/rotated matches refresh an anchor. Weak rescued
        # results never extend the last_confident age indefinitely.
        for index, face in identities.items():
            tracked[index] = {
                'box': np.asarray(face.bbox, dtype=np.float32).copy(),
                'last_confident': frame_index
            }

        existing_boxes = [face.bbox for face in identities.values()]
        proposals = []
        for identity, previous in list(tracked.items()):
            if identity in identities:
                continue
            age = frame_index - previous['last_confident']
            if age <= 0 or age > TRACK_MAX_GAP:
                tracked.pop(identity, None)
                continue
            for face_index, face in enumerate(detections):
                if any(bbox_iou(face.bbox, current) > 0.25
                       for current in existing_boxes):
                    continue
                overlap = bbox_iou(previous['box'], face.bbox)
                if overlap < TRACK_MIN_IOU:
                    continue
                scores = person_scores(face, assignments)
                own = scores[identity]
                if own is None or own < TRACK_MIN_SCORE:
                    continue
                competitors = [
                    value for j, value in enumerate(scores)
                    if j != identity and value is not None
                ]
                if competitors and own - max(competitors) < MIN_MARGIN:
                    continue
                proposals.append((overlap, own, identity, face_index))
        proposals.sort(reverse=True)
        used_faces = set()
        rescued = {}
        for overlap, score, identity, face_index in proposals:
            if identity in rescued or face_index in used_faces:
                continue
            # Avoid guessing if multiple detections have similarly plausible
            # overlap and recognition scores for a tracked identity.
            rivals = [
                p for p in proposals if p[2] == identity and p[3] != face_index
            ]
            if rivals and abs(overlap - rivals[0][0]) < 0.10:
                continue
            face = detections[face_index]
            rescued[identity] = face
            used_faces.add(face_index)
            tracked[identity]['box'] = np.asarray(face.bbox, dtype=np.float32).copy()
        return rescued
