"""In-memory multiple reference embeddings for each mapped person.

Keep original single-embedding assignments compatible. Compare a candidate to
the *best* manually verified reference per person (never average unrelated
poses into a weak single vector). Only the user adds reference images; no
automatic enrollment of uncertain detections.
"""
import numpy as np

from modules.face_selector import normalized_embedding


def reference_vectors(assignment):
    raw = assignment.get('embeddings') or [assignment['embedding']]
    vectors = []
    for item in raw:
        vector = np.asarray(item, dtype=np.float32).reshape(-1)
        magnitude = float(np.linalg.norm(vector))
        if np.isfinite(magnitude) and magnitude > 1e-8:
            vector = vector / magnitude
            if vectors and any(
                vector.shape == prev.shape and np.dot(vector, prev) > 0.99999
                for prev in vectors
            ):
                continue
            vectors.append(vector)
    return vectors


def face_similarity(face, assignment):
    vector = normalized_embedding(face)
    if vector is None:
        return None
    scores = [
        float(np.dot(vector, ref))
        for ref in reference_vectors(assignment)
        if vector.shape == ref.shape
    ]
    return max(scores) if scores else None


def person_scores(face, assignments):
    return [face_similarity(face, item) for item in assignments]
