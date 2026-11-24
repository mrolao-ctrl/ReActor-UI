"""On-demand, in-memory explanations for preview face selection.

The diagnostic reproduces the production matchers (including conservative
margins and VR eye pairing). Rotation trials only *measure* whether +/-45
degrees help the detector; they never modify the frame, assignment, swapping
policy, or thresholds. Run outside the Tk thread.
"""
import math

import cv2
import numpy as np

from modules.face_selector import MIN_MARGIN, MIN_SIMILARITY, normalized_embedding
from modules.multi_face import match_distinct_faces
from modules.identity_references import person_scores
from modules.recognition_enhancements import ADAPTIVE_ANGLES
from modules.vr_stereo import (
    _paired_assignments, stereo_eye_regions, match_reusable_faces
)


def _scores(face, assignments):
    return person_scores(face, assignments)

def _reason(scores, selected, assignment_index, face_index, all_scores,
            is_stereo, accepted, all_matches):
    score = scores[assignment_index]
    if selected:
        if score is not None and score < MIN_SIMILARITY and is_stereo:
            return 'Matched using stereo-pair confirmation'
        return 'Matched — face replacement expected'
    if score is None:
        return 'No compatible face embedding'
    if score < MIN_SIMILARITY:
        return 'Below similarity threshold'
    competitors = [
        val for i, val in enumerate(scores)
        if i != assignment_index and val is not None
    ]
    if competitors:
        best_other = max(competitors)
        if best_other >= score + MIN_MARGIN:
            return 'Another mapped person has a stronger similarity score'
        if score - best_other < MIN_MARGIN:
            return 'Ambiguous identity — competing mapped person'
    if not is_stereo:
        other_faces = [
            row[assignment_index] for i, row in enumerate(all_scores)
            if i != face_index and row[assignment_index] is not None
        ]
        if other_faces and score - max(other_faces) < MIN_MARGIN:
            return 'Ambiguous faces — same person matches another detection'
    if face_index in all_matches:
        return 'Face assigned to another mapped person'
    return 'Not selected by conservative matching'


def _build_rows(faces, assignments, match_by_face, stereo):
    scored = [_scores(face, assignments) for face in faces]
    rows = []
    for i, vector in enumerate(scored):
        for assignment_index, score in enumerate(vector):
            selected = match_by_face.get(i) == assignment_index
            rows.append({
                'face': i + 1,
                'person': assignment_index + 1,
                'score': score,
                'matched': selected,
                'reason': _reason(
                    vector, selected, assignment_index, i,
                    scored, stereo, selected, match_by_face
                ),
            })
    return rows


def _rotate_expand(image, degrees):
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), degrees, 1.0
    )
    radians = math.radians(degrees)
    rotated_width = math.ceil(
        abs(width * math.cos(radians)) + abs(height * math.sin(radians))
    )
    rotated_height = math.ceil(
        abs(width * math.sin(radians)) + abs(height * math.cos(radians))
    )
    transform[0, 2] += (rotated_width - width) / 2.0
    transform[1, 2] += (rotated_height - height) / 2.0
    return cv2.warpAffine(
        image, transform, (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _rotation_trials(image, assignments, detect_faces, stereo, adaptive=False):
    trials = []
    for angle in (ADAPTIVE_ANGLES if adaptive else (-45, 45)):
        rotated = _rotate_expand(image, angle)
        faces = list(detect_faces(rotated) or [])
        if stereo:
            matches = match_reusable_faces(faces, assignments)
        else:
            matched = match_distinct_faces(faces, assignments)
            matches = {
                face_index: assignment_index
                for face_index, face in enumerate(faces)
                for assignment_index, assignment in enumerate(assignments)
                if any(
                    entry is assignment and found is face
                    for entry, found in matched
                )
            }
        best = None
        person = None
        for face in faces:
            for assignment_index, score in enumerate(_scores(face, assignments)):
                if score is not None and (best is None or score > best):
                    best, person = score, assignment_index + 1
        trials.append({
            'degrees': angle,
            'count': len(faces),
            'best_similarity': best,
            'best_person': person,
            'matched': bool(matches),
        })
    return trials


def analyse_preview_faces(frame, assignments, detect_faces, *, vr_stereo=False,
                          adaptive=False, check_cancelled=None):
    """Explain face matching on an original, unaltered preview frame.

    Output contains numeric similarities and the production matcher outcome.
    Rotation trials run only if an eye/normal frame had no accepted match.
    These extra detections never feed the actual face swap.
    """
    if frame is None:
        raise ValueError('Select a valid original frame first.')
    if not assignments:
        return {
            'sections': [], 'min_similarity': MIN_SIMILARITY,
            'min_margin': MIN_MARGIN,
            'message': 'Map a target person in Step 02 before diagnosing faces.'
        }

    regions = (
        [(eye, label) for (eye, _), label in zip(
            stereo_eye_regions(frame), ('Left eye', 'Right eye')
        )]
        if vr_stereo else [(frame, 'Original frame')]
    )
    detected = []
    for image, label in regions:
        if check_cancelled:
            check_cancelled()
        detected.append((image, label, list(detect_faces(image) or [])))

    if vr_stereo and len(detected) == 2:
        left, right = detected
        maps = _paired_assignments(
            left[2], right[2], assignments, left[0].shape, right[0].shape
        )
    else:
        faces = detected[0][2]
        matches = match_distinct_faces(faces, assignments)
        maps = ({
            face_index: assignment_index
            for face_index, face in enumerate(faces)
            for assignment_index, assignment in enumerate(assignments)
            if any(entry is assignment and found is face
                   for entry, found in matches)
        },)

    sections = []
    for (image, label, faces), mapping in zip(detected, maps):
        if check_cancelled:
            check_cancelled()
        rows = _build_rows(faces, assignments, mapping, vr_stereo)
        # A rotation trial is a diagnostic hint only, never an implicit
        # replacement of the actual production matching decision.
        rotation = (
            _rotation_trials(image, assignments, detect_faces, vr_stereo, adaptive=adaptive)
            if not mapping else []
        )
        sections.append({
            'title': label, 'detected': len(faces),
            'matched': len(mapping), 'rows': rows, 'rotations': rotation
        })

    return {
        'min_similarity': MIN_SIMILARITY,
        'min_margin': MIN_MARGIN,
        'sections': sections,
        'message': (
            'Rotations keep the normal identity threshold. '
            'An accepted rotated result may be used by Adaptive detection '
            'when that option is enabled; this diagnosis changes no pixels.'
            if adaptive else
            'Rotation trials are diagnostic only; no video frames are changed.'
        )
    }
