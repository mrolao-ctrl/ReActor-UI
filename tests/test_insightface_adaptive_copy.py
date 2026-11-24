"""Regression: actual InsightFace Face dictionary behaviour during rotated swap.

InsightFace app.common.Face is a dict whose __getattr__ returns None for
unknown attributes, including __setstate__. copy.copy on this type raises
TypeError after a rotated detection. The tests use a lightweight equivalent
so GitHub CI needs no multi-GB model/InsightFace dependency.
"""
import copy
import unittest
from unittest.mock import Mock

import numpy as np

from modules.recognition_enhancements import (
    _rotate_with_transform, _unrotate_face, find_rotated_matches,
)
from modules.multi_face import swap_assigned_faces


class InsightFaceLike(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        d = dict(d)
        d.update(kwargs)
        for key, value in d.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def normed_embedding(self):
        emb = self.embedding
        if emb is None:
            return None
        return emb / np.linalg.norm(emb)


def face():
    return InsightFaceLike(
        bbox=np.array([28., 28., 55., 56.], dtype=np.float32),
        kps=np.array([
            [34., 37.], [48., 37.], [41., 43.], [36., 49.], [47., 49.]
        ], dtype=np.float32),
        embedding=np.array([1., 0., 0.], dtype=np.float32),
        det_score=0.98,
    )


class AdaptiveCopyTests(unittest.TestCase):
    def test_insightface_like_dict_reproduces_original_copy_error(self):
        # Protect the regression's premise; InsightFace has this same
        # unconventional __getattr__/dict contract.
        with self.assertRaisesRegex(TypeError, 'NoneType'):
            copy.copy(face())

    def test_original_rotated_face_is_reconstructed_safely(self):
        target = face()
        before_bbox = target.bbox.copy()
        before_kps = target.kps.copy()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, inverse = _rotate_with_transform(image, -30)
        restored = _unrotate_face(target, inverse, image.shape)
        self.assertIsInstance(restored, InsightFaceLike)
        self.assertIsNot(restored, target)
        self.assertIsNot(restored.bbox, target.bbox)
        self.assertIsNot(restored.kps, target.kps)
        self.assertEqual((5, 2), restored.kps.shape)
        self.assertEqual((4,), restored.bbox.shape)
        np.testing.assert_array_equal(target.bbox, before_bbox)
        np.testing.assert_array_equal(target.kps, before_kps)
        self.assertAlmostEqual(1.0, restored.normed_embedding[0])
        self.assertEqual(0.98, restored.det_score)

    def test_accepted_rotated_face_flows_through_swap_without_typeerror(self):
        target = face()
        detector = Mock(side_effect=[[], [target]])
        swapper = Mock(side_effect=lambda source, matched, image: image)
        assignment = {
            'embedding': np.array([1., 0., 0.], dtype=np.float32),
            'embeddings': [np.array([1., 0., 0.], dtype=np.float32)],
            'source_face': 'source',
        }
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result, boxes = swap_assigned_faces(
            image, [assignment], detector, swapper, adaptive=True
        )
        self.assertIs(result, image)
        self.assertEqual(1, swapper.call_count)
        self.assertEqual(1, len(boxes))
        self.assertEqual('source', swapper.call_args.args[0])
        target_passed = swapper.call_args.args[1]
        self.assertIsInstance(target_passed, InsightFaceLike)
        self.assertIsNot(target_passed, target)
        self.assertEqual((5, 2), target_passed.kps.shape)

    def test_no_rotated_identity_does_not_raise_or_swap_wrong_person(self):
        uncertain = face()
        uncertain.embedding = np.array([0., 1., 0.], dtype=np.float32)
        detector = Mock(side_effect=lambda _: [uncertain])
        assignment = {'embedding': np.array([1., 0., 0.], dtype=np.float32)}
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual({}, find_rotated_matches(
            image, detector, [assignment], {}
        ))


if __name__ == '__main__':
    unittest.main()
