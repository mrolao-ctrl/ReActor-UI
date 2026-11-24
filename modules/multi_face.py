"""Local multi-person face assignment and conflict-safe identity matching.

One face analysis is performed per frame. Reference identities are selected
from original (pre-swap) frames in the GUI; each identity has its own source
image. Ambiguous matches are skipped rather than replacing the wrong person.
"""
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np

from modules.face_selector import MIN_MARGIN, MIN_SIMILARITY, normalized_embedding
from modules.identity_references import face_similarity, reference_vectors
from modules.recognition_enhancements import find_rotated_matches


def prepare_assignments(assignments: Iterable[dict], load_face: Callable[[str], Any]) -> list:
    """Resolve each source image once per run, not for every frame."""
    prepared = []
    sources = {}
    for assignment in assignments:
        source_path = assignment['source_path']
        references = reference_vectors(assignment)
        if not references:
            raise ValueError('Invalid identity embedding')
        if source_path not in sources:
            sources[source_path] = load_face(source_path)
        prepared.append({
            'embedding': references[0],
            'embeddings': references,
            'source_face': sources[source_path],
            'source_path': source_path,
        })
    return prepared


def match_distinct_faces(
    faces: Iterable[Any],
    assignments: list,
    min_similarity: float = MIN_SIMILARITY,
    min_margin: float = MIN_MARGIN
) -> List[Tuple[dict, Any]]:
    """Match people 1:1 with margin across detected faces AND identities.

    Do not assign an input face to two different source identities. Reject
    uncertain selections instead of accidentally overwriting the first swap.
    """
    detections = list(faces or [])
    if not detections or not assignments:
        return []
    scores = {}
    for i, assignment in enumerate(assignments):
        for j, face in enumerate(detections):
            score = face_similarity(face, assignment)
            if score is not None:
                scores[i, j] = score

    proposals = []
    for i, assignment in enumerate(assignments):
        candidates = sorted(
            ((score, j) for (a, j), score in scores.items() if a == i),
            reverse=True
        )
        if not candidates or candidates[0][0] < min_similarity:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < min_margin:
            continue
        score, j = candidates[0]
        # If two reference identities could explain the same face, do not
        # guess which one to apply.
        competitor = max(
            (v for (a, face_index), v in scores.items()
             if face_index == j and a != i),
            default=-1.0
        )
        if score - competitor < min_margin:
            continue
        proposals.append((score, i, j))

    proposals.sort(reverse=True)
    used = set()
    matched = []
    for _, i, j in proposals:
        if j not in used:
            used.add(j)
            matched.append((i, assignments[i], detections[j]))
    # Respect assignment order for predictable compositing.
    return [(assignment, face) for _, assignment, face in sorted(matched, key=lambda item: item[0])]


def swap_assigned_faces(
    frame: Any,
    assignments: list,
    get_faces: Callable[[Any], Iterable[Any]],
    swap_face: Callable[[Any, Any, Any], Any],
    adaptive: bool = False,
    tracker=None,
    frame_index=None,
    check_cancelled=None,
) -> tuple:
    """Match once against original pixels, then optionally rescue hard poses."""
    detections = list(get_faces(frame) or [])
    matches = match_distinct_faces(detections, assignments)
    accepted = {
        i: face for assignment, face in matches
        for i, current in enumerate(assignments) if current is assignment
    }
    if adaptive and len(accepted) < len(assignments):
        accepted.update(find_rotated_matches(
            frame, get_faces, assignments, accepted,
            check_cancelled=check_cancelled
        ))
    if tracker is not None and frame_index is not None:
        accepted.update(tracker.augment(
            'normal', frame_index, detections, accepted, assignments
        ))

    boxes = []
    # Preserve source/identity order; identify original faces before swapping.
    for i in sorted(accepted):
        if check_cancelled:
            check_cancelled()
        target = accepted[i]
        frame = swap_face(assignments[i]['source_face'], target, frame)
        boxes.append(tuple(map(float, target.bbox)))
    return frame, boxes
