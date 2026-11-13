"""Unit tests for the identity selector; no video, GPU or model required."""
import unittest
from types import SimpleNamespace

import numpy as np

from modules.face_selector import normalized_embedding, select_matching_face


def face(*embedding):
    return SimpleNamespace(normed_embedding=np.asarray(embedding, dtype=np.float32))


class FaceSelectionTests(unittest.TestCase):
    def test_selects_identity_not_leftmost_face(self):
        target = face(1, 0, 0)
        other = face(0, 1, 0)
        self.assertIs(select_matching_face([other, target], [1, 0, 0]), target)

    def test_skips_low_similarity(self):
        self.assertIsNone(select_matching_face([face(0, 1, 0)], [1, 0, 0]))

    def test_skips_ambiguous_frame(self):
        self.assertIsNone(select_matching_face(
            [face(1, 0, 0), face(0.99, 0.1, 0)], [1, 0, 0]
        ))

    def test_empty_frame(self):
        self.assertIsNone(select_matching_face([], [1, 0, 0]))

    def test_invalid_embedding(self):
        self.assertIsNone(normalized_embedding(SimpleNamespace(normed_embedding=None)))
        self.assertIsNone(select_matching_face([face(1, 0, 0)], [0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
