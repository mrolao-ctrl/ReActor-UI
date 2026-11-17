"""Local multi-person face assignment and conflict-safe identity matching.

One face analysis is performed per frame. Reference identities are selected
from original (pre-swap) frames in the GUI; each identity has its own source
image. Ambiguous matches are skipped rather than replacing the wrong person.
"""
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np

from modules.face_selector import MIN_MARGIN, MIN_SIMILARITY, normalized_embedding


def prepare_assignments(assignments: Iterable[dict], load_face: Callable[[str], Any]) -> list:
    """Resolve each source image once per run, not for every frame."""
    prepared = []
    sources = {}
    for assignment in assignments:
        source_path = assignment['source_path']
        ref = np.asarray(assignment['embedding'], dtype=np.float32).reshape(-1)
        size = np.linalg.norm(ref)
        if not np.isfinite(size) or size < 1e-8:
            raise ValueError('Invalid identity embedding')
        if source_path not in sources:
            sources[source_path] = load_face(source_path)
        prepared.append({
            'embedding': ref / size,
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
    vectors = [normalized_embedding(face) for face in detections]
    references = [np.asarray(a['embedding'], dtype=np.float32).reshape(-1)
                  for a in assignments]

    scores = {}
    for i, ref in enumerate(references):
        norm = float(np.linalg.norm(ref))
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        ref = ref / norm
        for j, vector in enumerate(vectors):
            if vector is not None and vector.shape == ref.shape:
                scores[i, j] = float(np.dot(ref, vector))

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
) -> tuple:
    """Run face detection on unchanged frame; swap selected matches in turn."""
    matches = match_distinct_faces(get_faces(frame), assignments)
    boxes = []
    for assignment, target in matches:
        frame = swap_face(assignment['source_face'], target, frame)
        boxes.append(tuple(map(float, target.bbox)))
    return frame, boxes
