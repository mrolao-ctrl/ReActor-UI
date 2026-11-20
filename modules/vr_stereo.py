"""Opt-in side-by-side stereo face matching.

The left and right eye images are analysed separately, at their native
coordinates. Unlike normal 1:1 identity mapping, one reference may explain
several faces (one in each eye). Weak stereo partners are only propagated when
their geometry and face embeddings agree with an already confident match.

The normal non-VR matcher remains untouched.
"""
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np

from modules.face_selector import MIN_MARGIN, MIN_SIMILARITY, normalized_embedding


def stereo_eye_regions(frame):
    """Return copied eye images and their horizontal offsets."""
    if frame is None or len(frame.shape) < 2 or frame.shape[1] < 2:
        return []
    midpoint = frame.shape[1] // 2
    return [
        (frame[:, :midpoint].copy(), 0),
        (frame[:, midpoint:].copy(), midpoint),
    ]


def _references(assignments):
    vectors = []
    for item in assignments:
        vector = np.asarray(item['embedding'], dtype=np.float32).reshape(-1)
        size = float(np.linalg.norm(vector))
        vectors.append(vector / size if np.isfinite(size) and size > 1e-8 else None)
    return vectors


def _score(face, reference):
    vector = normalized_embedding(face)
    if vector is None or reference is None or vector.shape != reference.shape:
        return None
    return float(np.dot(vector, reference))


def match_reusable_faces(faces, assignments, min_similarity=MIN_SIMILARITY,
                         min_margin=MIN_MARGIN):
    """Assign each face its best unique identity, allowing reference reuse.

    Unlike match_distinct_faces, don't reject a confident identity just
    because another detected face matches the same reference equally well.
    Reject conflicts *between different reference identities* instead.
    Returns a dictionary of face index -> assignment index.
    """
    refs = _references(assignments)
    matches = {}
    for face_index, face in enumerate(faces):
        candidates = sorted(
            ((score, assignment_index)
             for assignment_index, ref in enumerate(refs)
             if (score := _score(face, ref)) is not None),
            reverse=True
        )
        if not candidates:
            continue
        top_score, top_index = candidates[0]
        if top_score < min_similarity:
            continue
        if len(candidates) > 1 and top_score - candidates[1][0] < min_margin:
            continue
        matches[face_index] = top_index
    return matches


def _geometry(face, width, height):
    x1, y1, x2, y2 = (float(value) for value in face.bbox[:4])
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None
    return ((x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height)


def pair_stereo_faces(left_faces, right_faces, left_shape, right_shape):
    """Find geometrically corresponding faces between the two eye images.

    Require comparable relative x/y/size and a minimally similar embedding.
    Pairs are mutually one-to-one, even when a video contains several people.
    """
    candidates = []
    for i, left in enumerate(left_faces):
        a = _geometry(left, left_shape[1], left_shape[0])
        left_embedding = normalized_embedding(left)
        if a is None or left_embedding is None:
            continue
        for j, right in enumerate(right_faces):
            b = _geometry(right, right_shape[1], right_shape[0])
            right_embedding = normalized_embedding(right)
            if b is None or right_embedding is None:
                continue
            if right_embedding.shape != left_embedding.shape:
                continue
            if float(np.dot(left_embedding, right_embedding)) < 0.12:
                continue
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            wratio = a[2] / max(1e-8, b[2])
            hratio = a[3] / max(1e-8, b[3])
            if dx > 0.16 or dy > 0.15 or not 0.55 <= wratio <= 1.8 or not 0.55 <= hratio <= 1.8:
                continue
            cost = dx / 0.16 + dy / 0.15 + abs(np.log(wratio)) + abs(np.log(hratio))
            candidates.append((cost, i, j))
    paired = []
    left_used, right_used = set(), set()
    for _, i, j in sorted(candidates):
        if i in left_used or j in right_used:
            continue
        paired.append((i, j))
        left_used.add(i)
        right_used.add(j)
    return paired


def _can_inherit(face, index, assignments):
    """Relax a *paired* eye's identity threshold, not its identity margin."""
    refs = _references(assignments)
    own = _score(face, refs[index])
    if own is None or own < 0.12:
        return False
    rivals = [_score(face, ref) for i, ref in enumerate(refs) if i != index]
    rivals = [score for score in rivals if score is not None]
    return not rivals or own - max(rivals) >= MIN_MARGIN


def _paired_assignments(left_faces, right_faces, assignments, left_shape, right_shape):
    left = match_reusable_faces(left_faces, assignments)
    right = match_reusable_faces(right_faces, assignments)
    for li, ri in pair_stereo_faces(left_faces, right_faces, left_shape, right_shape):
        left_assignment = left.get(li)
        right_assignment = right.get(ri)
        # Never overwrite two different confident mappings.
        if left_assignment is not None and right_assignment is None:
            if _can_inherit(right_faces[ri], left_assignment, assignments):
                right[ri] = left_assignment
        elif right_assignment is not None and left_assignment is None:
            if _can_inherit(left_faces[li], right_assignment, assignments):
                left[li] = right_assignment
    return left, right


def picker_stereo_faces(frame, detect_faces: Callable[[Any], Iterable[Any]]):
    """Expose full-frame bboxes for the Map Faces picker, but detect each eye."""
    output = []
    for eye, offset in stereo_eye_regions(frame):
        for face in detect_faces(eye) or []:
            box = np.asarray(face.bbox, dtype=np.float32).copy()
            box[[0, 2]] += offset
            output.append(SimpleNamespace(
                bbox=box,
                normed_embedding=getattr(face, 'normed_embedding', None),
            ))
    return output


def swap_stereo_assigned_faces(frame, assignments, get_faces, swap_face,
                               check_cancelled=None):
    """Process the SBS eyes independently and record full-frame face boxes."""
    regions = stereo_eye_regions(frame)
    if len(regions) != 2 or not assignments:
        return frame, []
    (left_eye, left_offset), (right_eye, right_offset) = regions
    if check_cancelled:
        check_cancelled()
    left_faces = list(get_faces(left_eye) or [])
    if check_cancelled:
        check_cancelled()
    right_faces = list(get_faces(right_eye) or [])
    left_matches, right_matches = _paired_assignments(
        left_faces, right_faces, assignments, left_eye.shape, right_eye.shape
    )
    boxes = []
    for eye, offset, faces, matches in (
        (left_eye, left_offset, left_faces, left_matches),
        (right_eye, right_offset, right_faces, right_matches),
    ):
        for index, assignment_index in sorted(matches.items()):
            if check_cancelled:
                check_cancelled()
            face = faces[index]
            eye = swap_face(assignments[assignment_index]['source_face'], face, eye)
            x1, y1, x2, y2 = (float(value) for value in face.bbox[:4])
            boxes.append((x1 + offset, y1, x2 + offset, y2))
        frame[:, offset: offset + eye.shape[1]] = eye
    return frame, boxes
