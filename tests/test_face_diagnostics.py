"""Headless tests for preview identity diagnostics and rotation-only trials."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from modules.face_diagnostics import analyse_preview_faces


class Face:
    def __init__(self, embedding, bbox=(10, 10, 30, 30)):
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)
        self.bbox = np.asarray(bbox, dtype=np.float32)


class FaceDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((60, 200, 3), dtype=np.uint8)
        self.person = [{'embedding': np.array([1., 0., 0.], dtype=np.float32)}]

    def test_confident_face_explains_accepted_matching(self):
        detector = Mock(return_value=[Face([1, 0, 0])])
        result = analyse_preview_faces(self.frame, self.person, detector)
        self.assertEqual(0.35, result['min_similarity'])
        self.assertEqual(0.06, result['min_margin'])
        section = result['sections'][0]
        self.assertEqual(1, section['detected'])
        self.assertEqual(1, section['matched'])
        self.assertAlmostEqual(1.0, section['rows'][0]['score'])
        self.assertTrue(section['rows'][0]['matched'])
        self.assertIn('Matched', section['rows'][0]['reason'])
        self.assertEqual([], section['rotations'])
        detector.assert_called_once()

    def test_weak_similarity_triggers_read_only_rotation_trials(self):
        detector = Mock(side_effect=[
            [Face([0, 1, 0])], [Face([1, 0, 0])], [Face([0, 1, 0])]
        ])
        frame = self.frame.copy()
        result = analyse_preview_faces(frame, self.person, detector)
        section = result['sections'][0]
        self.assertEqual(0, section['matched'])
        self.assertFalse(section['rows'][0]['matched'])
        self.assertIn('Below similarity threshold', section['rows'][0]['reason'])
        self.assertEqual(2, len(section['rotations']))
        self.assertTrue(section['rotations'][0]['matched'])
        self.assertFalse(section['rotations'][1]['matched'])
        self.assertEqual(-45, section['rotations'][0]['degrees'])
        self.assertEqual(45, section['rotations'][1]['degrees'])
        np.testing.assert_array_equal(self.frame, frame)
        self.assertIn('Rotation trials are diagnostic only', result['message'])

    def test_no_face_detected_reports_absence_and_rotation_checks(self):
        detector = Mock(return_value=[])
        result = analyse_preview_faces(self.frame, self.person, detector)
        section = result['sections'][0]
        self.assertEqual(0, section['detected'])
        self.assertEqual([], section['rows'])
        self.assertEqual(2, len(section['rotations']))
        self.assertEqual(3, detector.call_count)

    def test_normal_matching_reports_ambiguity_among_duplicate_faces(self):
        detector = Mock(return_value=[
            Face([1, 0, 0], [10, 10, 30, 30]),
            Face([1, 0, 0], [40, 10, 60, 30])
        ])
        result = analyse_preview_faces(self.frame, self.person, detector)
        section = result['sections'][0]
        self.assertEqual(0, section['matched'])
        self.assertIn('Ambiguous faces', section['rows'][0]['reason'])

    def test_identical_mappings_report_identity_ambiguity(self):
        detector = Mock(return_value=[Face([1, 0, 0])])
        two = self.person + [{'embedding': np.array([1., 0., 0.])}]
        result = analyse_preview_faces(self.frame, two, detector)
        section = result['sections'][0]
        self.assertEqual(0, section['matched'])
        self.assertEqual(2, len(section['rows']))
        self.assertTrue(all('Ambiguous identity' in row['reason']
                            for row in section['rows']))

    def test_stereo_mode_matches_both_eyes_using_production_matcher(self):
        detector = Mock(side_effect=[
            [Face([1, 0, 0], [10, 10, 30, 30])],
            [Face([1, 0, 0], [11, 10, 31, 30])]
        ])
        result = analyse_preview_faces(
            self.frame, self.person, detector, vr_stereo=True
        )
        self.assertEqual(['Left eye', 'Right eye'],
                         [section['title'] for section in result['sections']])
        self.assertEqual([1, 1], [section['matched'] for section in result['sections']])
        self.assertEqual(2, detector.call_count)
        self.assertTrue(all(section['rows'][0]['matched']
                            for section in result['sections']))

    def test_stereo_pair_can_inherit_lower_score_without_loosening_threshold(self):
        detector = Mock(side_effect=[
            [Face([1, 0, 0], [20, 10, 40, 30])],
            [Face([0.2, 0.98, 0], [21, 10, 41, 30])]
        ])
        result = analyse_preview_faces(
            self.frame, self.person, detector, vr_stereo=True
        )
        sections = result['sections']
        self.assertEqual(1, sections[1]['matched'])
        self.assertLess(sections[1]['rows'][0]['score'], 0.35)
        self.assertIn('stereo-pair', sections[1]['rows'][0]['reason'])

    def test_no_mapping_returns_explanation_without_running_detector(self):
        detector = Mock()
        result = analyse_preview_faces(self.frame, [], detector)
        detector.assert_not_called()
        self.assertEqual([], result['sections'])
        self.assertIn('Map a target person', result['message'])

    def test_preview_ui_integrates_diagnosis_without_changing_swap(self):
        root = Path(__file__).resolve().parents[1]
        ui = (root / 'modules/ui.py').read_text(encoding='utf-8')
        diag = (root / 'modules/face_diagnostics_ui.py').read_text(encoding='utf-8')
        swap = (root / 'modules/processors/frame/face_swapper.py').read_text('utf-8')
        ast.parse(ui)
        ast.parse(diag)
        self.assertIn("text='Diagnose faces'", ui)
        self.assertIn("command=diagnose_selected_frame", ui)
        self.assertIn('FaceDiagnosticDialog(', ui)
        self.assertIn("state['frame']", ui)
        self.assertIn('close_diagnostics()', ui)
        self.assertIn('original_frame.copy()', diag)
        self.assertIn('threading.Thread(', diag)
        self.assertIn('analyse_preview_faces(', diag)
        self.assertIn('swap_assigned_faces(', swap)
        self.assertNotIn('face_diagnostics', swap)


if __name__ == '__main__':
    unittest.main()
