"""Conservative identity matching for selecting one person in a video.

Only in-memory InsightFace embeddings are used. No face crops or embeddings are
saved or sent to any service. If a face cannot be matched confidently, skip it
instead of accidentally replacing another person.
"""

from typing import Any, Iterable, Optional

import numpy as np

MIN_SIMILARITY = 0.35
MIN_MARGIN = 0.06


def normalized_embedding(face: Any) -> Optional[np.ndarray]:
    """Return a unit-length embedding for an InsightFace detection."""
    embedding = getattr(face, 'normed_embedding', None)
    if embedding is None:
        return None
    values = np.asarray(embedding, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(values))
    if not np.isfinite(magnitude) or magnitude < 1e-8:
        return None
    return values / magnitude


def select_matching_face(
    faces: Iterable[Any],
    reference_embedding: np.ndarray,
    min_similarity: float = MIN_SIMILARITY,
    min_margin: float = MIN_MARGIN
) -> Optional[Any]:
    """Find the best identity match, or None when uncertain.

    A similarity threshold reduces false matches. A margin over the second-best
    detected face avoids switching identities when two matches are close.
    """
    reference = np.asarray(reference_embedding, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(reference))
    if not np.isfinite(magnitude) or magnitude < 1e-8:
        return None
    reference = reference / magnitude

    candidates = []
    for face in faces or []:
        embedding = normalized_embedding(face)
        if embedding is None or embedding.shape != reference.shape:
            continue
        score = float(np.dot(reference, embedding))
        candidates.append((score, face))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_face = candidates[0]
    if best_score < min_similarity:
        return None
    if len(candidates) > 1 and best_score - candidates[1][0] < min_margin:
        return None
    return best_face
