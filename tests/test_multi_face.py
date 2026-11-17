"""Unit tests for distinct, ambiguity-safe multi-person selection."""
import unittest
from unittest.mock import Mock

import numpy as np

from modules.multi_face import match_distinct_faces, swap_assigned_faces, prepare_assignments


class DummyFace:
    def __init__(self, vector, bbox):
        self.normed_embedding = np.array(vector, dtype=np.float32)
        self.bbox = bbox


class MultiFaceTests(unittest.TestCase):
    def test_two_independent_people_matched(self):
        faces = [
            DummyFace([1, 0, 0], [5, 10, 15, 20]),
            DummyFace([0, 1, 0], [25, 10, 35, 20]),
        ]
        assignments = [
            {'embedding': np.array([0, 1, 0]), 'source_face': 'replacement-2'},
            {'embedding': np.array([1, 0, 0]), 'source_face': 'replacement-1'},
        ]
        matches = match_distinct_faces(faces, assignments)
        self.assertEqual(2, len(matches))
        self.assertEqual([faces[1], faces[0]], [face for _, face in matches])

    def test_ambiguous_targets_skipped(self):
        faces = [DummyFace([1, 0], [1, 1, 2, 2])]
        assignments = [
            {'embedding': np.array([1, 0]), 'source_face': '1'},
            {'embedding': np.array([1, 0]), 'source_face': '2'},
        ]
        self.assertEqual([], match_distinct_faces(faces, assignments))

    def test_unconfident_reference_does_not_target_another_person(self):
        faces = [DummyFace([0, 1], [1, 2, 3, 4])]
        assignments = [{'embedding': np.array([1, 0]), 'source_face': 'x'}]
        self.assertEqual([], match_distinct_faces(faces, assignments))

    def test_faces_detected_only_once_and_masks_recorded(self):
        faces = [DummyFace([1, 0], [1, 2, 3, 4]),
                 DummyFace([0, 1], [5, 6, 7, 8])]
        assignments = [{'embedding': np.array([1, 0]), 'source_face': 'a'},
                       {'embedding': np.array([0, 1]), 'source_face': 'b'}]
        get_faces = Mock(return_value=faces)
        swap = Mock(side_effect=lambda source, target, frame: frame + source)
        frame, boxes = swap_assigned_faces('', assignments, get_faces, swap)
        self.assertEqual(1, get_faces.call_count)
        self.assertEqual(2, swap.call_count)
        self.assertEqual('ab', frame)
        self.assertEqual([(1., 2., 3., 4.), (5., 6., 7., 8.)], boxes)

    def test_source_faces_loaded_once(self):
        load_face = Mock(side_effect=lambda path: path + '-loaded')
        assignments = [{'source_path': 'same.png', 'embedding': [1, 0]},
                       {'source_path': 'same.png', 'embedding': [0, 1]}]
        prepared = prepare_assignments(assignments, load_face)
        self.assertEqual(1, load_face.call_count)
        self.assertEqual(2, len(prepared))


if __name__ == '__main__':
    unittest.main()
