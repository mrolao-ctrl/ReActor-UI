"""Synthetic, non-explicit tests: hands/objects are represented by occlusion masks.

No real people, video, or ONNX model is needed to test compositing.
"""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modules.face_occlusion import (
    LANDMARK_TEMPLATE, LANDMARK_TEMPLATE_3, combine_visibility,
    estimate_alignment_matrix, get_visibility_mask, select_faces_by_boxes
)


class DummyFace:
    def __init__(self, box, kps=None):
        self.bbox = np.asarray(box, dtype=np.float32)
        self.kps = np.asarray(
            LANDMARK_TEMPLATE if kps is None else kps, dtype=np.float32
        )


class DummySession:
    def __init__(self, prediction):
        self.prediction = prediction
        self.inputs = []

    def get_inputs(self):
        return [type('Input', (), {'name': 'input'})()]

    def run(self, output_names, feed):
        self.inputs.append(feed['input'])
        return [self.prediction[None, :, :, None]]


class OcclusionTests(unittest.TestCase):
    def test_compositing_respects_foreground_object_pixels(self):
        original = np.full((4, 6, 3), 10, dtype=np.uint8)
        edited = np.full((4, 6, 3), 240, dtype=np.uint8)
        visible = np.ones((4, 6), dtype=np.float32)
        visible[:, :3] = 0
        result = combine_visibility(original, edited, visible)
        self.assertTrue(np.array_equal(result[:, :3], original[:, :3]))
        self.assertTrue(np.array_equal(result[:, 3:], edited[:, 3:]))

    def test_half_alpha_is_feathered(self):
        original = np.full((8, 8, 3), 50, dtype=np.uint8)
        edited = np.full((8, 8, 3), 150, dtype=np.uint8)
        visible = np.full((8, 8), 0.5, dtype=np.float32)
        self.assertEqual(100, int(combine_visibility(original, edited, visible)[3, 3, 0]))

    def test_no_occlusion_preserves_existing_swap(self):
        original = np.zeros((12, 12, 3), dtype=np.uint8)
        edited = np.full_like(original, 200)
        result = combine_visibility(
            original, edited, np.ones((12, 12), dtype=np.float32)
        )
        np.testing.assert_array_equal(result, edited)

    def test_half_frame_mask_inference_is_local(self):
        frame = np.full((256, 256, 3), (25, 100, 220), dtype=np.uint8)
        prediction = np.ones((256, 256), dtype=np.float32)
        prediction[:, :128] = 0  # A foreground object obscures left half.
        fake = DummySession(prediction)
        with patch('modules.face_occlusion.get_occluder_session', return_value=fake):
            mask = get_visibility_mask(frame, DummyFace([40, 40, 180, 200]))
        self.assertEqual((256, 256), mask.shape)
        self.assertLess(float(mask[128, 20]), 0.05)
        self.assertGreater(float(mask[128, 235]), 0.95)
        self.assertEqual((1, 256, 256, 3), fake.inputs[0].shape)
        self.assertEqual(np.float32, fake.inputs[0].dtype)
        self.assertGreaterEqual(float(fake.inputs[0].min()), 0)
        self.assertLessEqual(float(fake.inputs[0].max()), 1)

    def test_invalid_landmarks_rejected(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        bad = DummyFace([10, 20, 30, 40], kps=[[1, 1], [2, 2]])
        with self.assertRaises(ValueError):
            get_visibility_mask(frame, bad)

    def test_auto_keeps_five_points_for_valid_face(self):
        matrix, used = estimate_alignment_matrix(LANDMARK_TEMPLATE, 'auto')
        self.assertEqual('5-point', used)
        self.assertTrue(np.isfinite(matrix).all())

    def test_three_point_mode_ignores_occluded_mouth_landmarks(self):
        occluded = LANDMARK_TEMPLATE.copy()
        occluded[3:] = np.nan
        matrix, used = estimate_alignment_matrix(occluded, '3-point')
        self.assertEqual('3-point', used)
        np.testing.assert_allclose(matrix, np.array([[1, 0, 0], [0, 1, 0]]),
                                   atol=0.001)

    def test_auto_falls_back_when_mouth_is_missing(self):
        occluded = LANDMARK_TEMPLATE.copy()
        occluded[3:] = np.nan
        _, used = estimate_alignment_matrix(occluded, 'auto')
        self.assertEqual('3-point', used)

    def test_auto_falls_back_when_mouth_is_badly_misdetected(self):
        occluded = LANDMARK_TEMPLATE.copy()
        occluded[3] = [140, 80]
        occluded[4] = [140, 81]
        _, used = estimate_alignment_matrix(occluded, 'auto')
        self.assertEqual('3-point', used)

    def test_three_landmark_only_detections_supported(self):
        matrix, used = estimate_alignment_matrix(LANDMARK_TEMPLATE_3, 'auto')
        self.assertEqual('3-point', used)
        self.assertEqual((2, 3), matrix.shape)

    def test_explicit_five_point_rejects_missing_mouth(self):
        occluded = LANDMARK_TEMPLATE.copy()
        occluded[3:] = np.nan
        with self.assertRaisesRegex(ValueError, 'five valid'):
            estimate_alignment_matrix(occluded, '5-point')
        with self.assertRaisesRegex(ValueError, 'five valid'):
            estimate_alignment_matrix(LANDMARK_TEMPLATE_3, '5-point')

    def test_invalid_eye_or_nose_locations_are_rejected_in_all_modes(self):
        occluded = LANDMARK_TEMPLATE.copy()
        occluded[1] = occluded[0]  # Eye baseline collapses to zero.
        for mode in ('auto', '3-point', '5-point'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'Eye/nose'):
                    estimate_alignment_matrix(occluded, mode)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid occlusion'):
            estimate_alignment_matrix(LANDMARK_TEMPLATE, 'unknown')

    def test_xseg_accepts_three_point_microphone_case(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        prediction = np.ones((256, 256), dtype=np.float32)
        prediction[155:230, 96:165] = 0  # Synthetic object in front of mouth.
        fake = DummySession(prediction)
        with patch('modules.face_occlusion.get_occluder_session', return_value=fake):
            mask = get_visibility_mask(
                frame, DummyFace([20, 20, 230, 240], kps=LANDMARK_TEMPLATE_3),
                mode='3-point'
            )
        self.assertEqual(0.0, float(mask[185, 130]))
        self.assertGreater(float(mask[110, 130]), 0.95)

    def test_box_matching_finds_selected_person(self):
        person_a = DummyFace([10, 10, 30, 30])
        person_b = DummyFace([70, 70, 95, 95])
        found = select_faces_by_boxes(
            [person_a, person_b], [[68, 68, 96, 96]]
        )
        self.assertEqual([person_b], found)

    def test_uncertain_or_missing_face_skips_masking(self):
        self.assertIsNone(select_faces_by_boxes([], [[10, 10, 30, 30]]))


if __name__ == '__main__':
    unittest.main()
