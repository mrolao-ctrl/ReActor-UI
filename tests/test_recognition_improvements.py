"""Regression tests for 3 opt-in identity recognition improvements."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import cv2
import numpy as np

from modules.identity_references import face_similarity, person_scores
from modules.multi_face import match_distinct_faces, prepare_assignments, swap_assigned_faces
from modules.recognition_enhancements import (
    TemporalIdentityTracker, _rotate_with_transform, _unrotate_face,
    find_rotated_matches, bbox_iou
)
from modules import vr_stereo
from modules.app_settings import DEFAULTS, validate, apply_to_globals


class Face:
    def __init__(self, vec, box=(20, 20, 40, 40), kps=None):
        self.normed_embedding = np.array(vec, dtype=np.float32)
        self.bbox = np.asarray(box, dtype=np.float32)
        self.kps = np.asarray(
            kps if kps is not None else [
                [24, 27], [35, 27], [30, 31], [26, 35], [34, 35]
            ], dtype=np.float32
        )


def mapping(vec, label='A', more=None):
    refs = [np.asarray(vec, dtype=np.float32)]
    for item in (more or []):
        refs.append(np.asarray(item, dtype=np.float32))
    return {'embedding': refs[0], 'embeddings': refs, 'source_face': label,
            'source_path': label + '.png'}


class ImprovedRecognitionTests(unittest.TestCase):
    def test_second_reference_matches_pose_primary_misses(self):
        mappings = [mapping([1, 0, 0], more=[[0, 1, 0]])]
        angled = Face([0, 1, 0])
        self.assertEqual(1.0, face_similarity(angled, mappings[0]))
        self.assertEqual([(mappings[0], angled)],
                         match_distinct_faces([angled], mappings))
        self.assertEqual({0: 0},
                         vr_stereo.match_reusable_faces([angled], mappings))

    def test_original_single_embedding_is_backward_compatible(self):
        current = {'embedding': np.array([1, 0, 0]), 'source_path': 'ref.png'}
        prepared = prepare_assignments([current], lambda path: 'loaded')
        self.assertEqual(1, len(prepared[0]['embeddings']))
        self.assertEqual(1.0, face_similarity(Face([1, 0, 0]), prepared[0]))
        self.assertEqual(0.0, face_similarity(Face([0, 1, 0]), current))

    def test_competitors_remain_ambiguous_with_multiple_references(self):
        a = mapping([1, 0, 0], more=[[0, 1, 0]])
        b = mapping([0, 1, 0])
        face = Face([0, 1, 0])
        self.assertEqual([], match_distinct_faces([face], [a, b]))
        self.assertEqual({}, vr_stereo.match_reusable_faces([face], [a, b]))

    def test_rotation_remaps_five_keypoints_and_bbox_to_original(self):
        frame = np.zeros((120, 180, 3), np.uint8)
        rotated, inverse = _rotate_with_transform(frame, 45)
        direct = cv2.invertAffineTransform(inverse)
        original = Face([1, 0, 0], box=(40, 35, 70, 75),
                        kps=[[45, 45], [65, 44], [55, 55], [48, 66], [63, 64]])
        def apply(points):
            return np.asarray(points) @ direct[:, :2].T + direct[:, 2]
        x1, y1, x2, y2 = original.bbox
        box_corners = apply([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        min_x, min_y = box_corners.min(axis=0)
        max_x, max_y = box_corners.max(axis=0)
        face = Face([1, 0, 0], box=(min_x, min_y, max_x, max_y),
                    kps=apply(original.kps))
        restored = _unrotate_face(face, inverse, frame.shape)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(restored.kps, original.kps, atol=0.01)
        self.assertEqual((120, 180, 3), frame.shape)
        self.assertFalse(np.shares_memory(restored.kps, face.kps))
        # A rotated axis-aligned box expands on both transforms; five-point
        # landmarks (which the swap model actually uses) must remain precise.
        self.assertGreater(bbox_iou(restored.bbox, original.bbox), 0.20)

    def test_adaptive_rotation_keeps_threshold_and_avoids_duplicate_identity(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        source = Face([1, 0, 0], box=(20, 20, 40, 40))
        # Simulated detector supports testing only the approved angles.
        # Every rotation returns no match until a confident one at +45°.
        calls = {'count': 0}
        def detector(img):
            calls['count'] += 1
            if calls['count'] == 4:
                return [source]
            return []
        found = find_rotated_matches(
            frame, detector, [mapping([1, 0, 0])], {}
        )
        self.assertEqual([0], list(found))
        self.assertEqual(4, calls['count'])
        self.assertEqual({}, find_rotated_matches(
            frame, detector, [mapping([1, 0, 0])],
            {0: source}
        ))

    def test_adaptive_normal_swap_only_after_missing_confident_original(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        upright = Face([0, 1], box=(20, 20, 40, 40))
        angle = Face([1, 0], box=(18, 19, 38, 39))
        detector = Mock(side_effect=[[upright], [], [angle]])
        swapper = Mock(side_effect=lambda source, face, image: image)
        _, boxes = swap_assigned_faces(
            frame, [mapping([1, 0])], detector, swapper, adaptive=True
        )
        self.assertEqual(1, swapper.call_count)
        self.assertEqual('A', swapper.call_args.args[0])
        self.assertEqual(1, len(boxes))
        self.assertLessEqual(detector.call_count, 3)

    def test_temporal_accepts_only_recent_spatially_near_low_confidence_face(self):
        tracker = TemporalIdentityTracker()
        refs = [mapping([1, 0, 0])]
        first = Face([1, 0, 0], box=(20, 20, 40, 40))
        weak = Face([.23, .973, 0], box=(21, 20, 41, 40))
        self.assertEqual({}, tracker.augment('normal', 0, [first], {0: first}, refs))
        restored = tracker.augment('normal', 1, [weak], {}, refs)
        self.assertIs(restored[0], weak)
        self.assertIs(tracker.augment('normal', 2, [weak], {}, refs)[0], weak)
        self.assertEqual({}, tracker.augment('normal', 3, [weak], {}, refs))

    def test_temporal_rejects_wrong_person_motion_and_ambiguous_competitor(self):
        tracker = TemporalIdentityTracker()
        refs = [mapping([1, 0, 0]), mapping([0, 1, 0])]
        first = Face([1, 0, 0], box=(20, 20, 40, 40))
        tracker.augment('normal', 0, [first], {0: first}, refs)
        wrong = Face([0, 1, 0], box=(20, 20, 40, 40))
        far = Face([.23, .973, 0], box=(65, 60, 80, 75))
        self.assertEqual({}, tracker.augment('normal', 1, [wrong], {}, refs))
        self.assertEqual({}, tracker.augment('normal', 2, [far], {}, refs))

    def test_temporal_resets_after_skipped_frame_or_out_of_order(self):
        tracker = TemporalIdentityTracker()
        ref = [mapping([1, 0])]
        confident = Face([1, 0])
        weak = Face([.23, .973])
        tracker.augment('normal', 5, [confident], {0: confident}, ref)
        self.assertEqual({}, tracker.augment('normal', 8, [weak], {}, ref))
        tracker.reset()
        self.assertEqual({}, tracker.augment('normal', 9, [weak], {}, ref))

    def test_temporal_regions_are_independent_for_stereo_views(self):
        tracker = TemporalIdentityTracker()
        ref = [mapping([1, 0])]
        face = Face([1, 0])
        weak = Face([.23, .973])
        tracker.augment('left', 0, [face], {0: face}, ref)
        self.assertEqual({}, tracker.augment('right', 1, [weak], {}, ref))
        self.assertIn(0, tracker.augment('left', 1, [weak], {}, ref))

    def test_flags_default_off_preserve_existing_workflow(self):
        self.assertFalse(DEFAULTS['adaptive_face_detection'])
        self.assertFalse(DEFAULTS['temporal_face_tracking'])
        values = validate({'adaptive_face_detection': True,
                           'temporal_face_tracking': True})
        target = SimpleNamespace(fp_ui={})
        apply_to_globals(values, target)
        self.assertTrue(target.adaptive_face_detection)
        self.assertTrue(target.temporal_face_tracking)

    def test_ui_and_video_tracking_run_sequentially(self):
        root = Path(__file__).resolve().parents[1]
        ui = (root / 'modules' / 'ui.py').read_text('utf-8')
        swapper = (root / 'modules' / 'processors' / 'frame' / 'face_swapper.py').read_text('utf-8')
        engine = (root / 'modules' / 'recognition_enhancements.py').read_text('utf-8')
        vr = (root / 'modules' / 'vr_stereo.py').read_text('utf-8')
        for text in (ui, swapper, engine, vr):
            ast.parse(text)
        self.assertIn("text='+ Reference'", ui)
        self.assertIn("reference_index=position", ui)
        self.assertIn("text='Temporal identity tracking (slower)'", ui)
        self.assertIn("text='Adaptive detection (angled faces)'", ui)
        self.assertIn("if modules.globals.temporal_face_tracking", swapper)
        self.assertIn('for index, path in enumerate(temp_frame_paths):', swapper)
        self.assertIn('swap_stereo_assigned_faces(', swapper)


if __name__ == '__main__':
    unittest.main()
