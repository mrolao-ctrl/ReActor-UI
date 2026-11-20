"""Regression tests for opt-in side-by-side VR face mapping."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np

from modules import vr_stereo
from modules.multi_face import match_distinct_faces
from modules.app_settings import DEFAULTS, validate, apply_to_globals, save_settings, load_settings


class DummyFace:
    def __init__(self, embedding, bbox):
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)
        self.bbox = np.asarray(bbox, dtype=np.float32)


def assignment(vector, label):
    return {
        'embedding': np.asarray(vector, dtype=np.float32),
        'source_face': label,
        'source_path': label + '.png'
    }


class StereoModeTests(unittest.TestCase):
    def test_legacy_matcher_is_still_one_to_one_and_stereo_reuses_identity(self):
        faces = [
            DummyFace([1, 0, 0], [10, 10, 30, 30]),
            DummyFace([1, 0, 0], [40, 10, 60, 30]),
        ]
        mappings = [assignment([1, 0, 0], 'A')]
        # Original matcher skips equally good twin candidates.
        self.assertEqual([], match_distinct_faces(faces, mappings))
        self.assertEqual({0: 0, 1: 0},
                         vr_stereo.match_reusable_faces(faces, mappings))

    def test_two_people_map_independently_even_with_repeated_detections(self):
        faces = [
            DummyFace([1, 0, 0], [10, 10, 30, 30]),
            DummyFace([0, 1, 0], [40, 10, 60, 30]),
            DummyFace([1, 0, 0], [70, 10, 90, 30]),
        ]
        mappings = [
            assignment([1, 0, 0], 'A'), assignment([0, 1, 0], 'B')
        ]
        self.assertEqual(
            {0: 0, 1: 1, 2: 0},
            vr_stereo.match_reusable_faces(faces, mappings)
        )

    def test_multi_person_ambiguity_rejected(self):
        face = DummyFace([1, 0], [10, 10, 30, 30])
        mappings = [
            assignment([1, 0], 'A'),
            assignment([1, 0], 'B'),
        ]
        self.assertEqual({}, vr_stereo.match_reusable_faces([face], mappings))

    def test_eye_partner_inherits_when_geometry_and_weak_embedding_agree(self):
        left = [DummyFace([1, 0, 0], [25, 10, 45, 30])]
        right = [DummyFace([0.20, 0.98, 0], [26, 11, 46, 31])]
        mappings = [assignment([1, 0, 0], 'A')]
        normal = vr_stereo.match_reusable_faces(right, mappings)
        self.assertEqual({}, normal)
        left_map, right_map = vr_stereo._paired_assignments(
            left, right, mappings, (60, 100, 3), (60, 100, 3)
        )
        self.assertEqual({0: 0}, left_map)
        self.assertEqual({0: 0}, right_map)

    def test_no_propagation_to_a_different_person_or_wrong_position(self):
        left = [DummyFace([1, 0, 0], [20, 10, 40, 30])]
        wrong_person = [DummyFace([0, 1, 0], [20, 10, 40, 30])]
        distant_face = [DummyFace([0.20, 0.98, 0], [70, 10, 90, 30])]
        mapping = [assignment([1, 0, 0], 'A')]
        for right in (wrong_person, distant_face):
            _, matches = vr_stereo._paired_assignments(
                left, right, mapping, (60, 100, 3), (60, 100, 3)
            )
            self.assertEqual({}, matches)

    def test_unmatched_face_not_inherited_if_another_identity_is_ambiguous(self):
        left = [DummyFace([1, 0, 0], [20, 10, 40, 30])]
        right = [DummyFace([0.48, 0.48, 0.73], [20, 10, 40, 30])]
        refs = [assignment([1, 0, 0], 'A'),
                assignment([0, 1, 0], 'B')]
        _, inherited = vr_stereo._paired_assignments(
            left, right, refs, (60, 100, 3), (60, 100, 3)
        )
        self.assertEqual({}, inherited)

    def test_stereo_swap_operates_on_each_eye_and_records_full_frame_boxes(self):
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        image[:, 100:, :] = 20
        left_face = DummyFace([1, 0, 0], [20, 10, 40, 30])
        right_face = DummyFace([1, 0, 0], [22, 10, 42, 30])
        detector = Mock(side_effect=[[left_face], [right_face]])
        seen = []

        def replacement(source, target, eye):
            seen.append((source, tuple(target.bbox), eye.shape))
            x1, y1, x2, y2 = map(int, target.bbox)
            eye[y1:y2, x1:x2] = 255
            return eye

        processed, boxes = vr_stereo.swap_stereo_assigned_faces(
            image, [assignment([1, 0, 0], 'A')], detector, replacement
        )
        self.assertEqual(2, detector.call_count)
        self.assertEqual(2, len(seen))
        self.assertEqual('A', seen[0][0])
        self.assertEqual('A', seen[1][0])
        self.assertEqual((60, 100, 3), seen[0][2])
        self.assertEqual([(20., 10., 40., 30.), (122., 10., 142., 30.)], boxes)
        self.assertTrue(np.all(processed[10:30, 20:40] == 255))
        self.assertTrue(np.all(processed[10:30, 122:142] == 255))
        self.assertTrue(np.all(processed[0:9, 0:10] == 0))
        self.assertTrue(np.all(processed[0:9, 160:180] == 20))

    def test_stereo_picker_bboxes_right_face_using_full_frame_coordinates(self):
        frame = np.zeros((60, 200, 3), dtype=np.uint8)
        left = DummyFace([1, 0], [20, 10, 40, 30])
        right = DummyFace([1, 0], [22, 10, 42, 30])
        get_faces = Mock(side_effect=[[left], [right]])
        faces = vr_stereo.picker_stereo_faces(frame, get_faces)
        self.assertEqual(2, get_faces.call_count)
        self.assertEqual([20, 10, 40, 30], list(faces[0].bbox))
        self.assertEqual([122, 10, 142, 30], list(faces[1].bbox))
        self.assertEqual([22, 10, 42, 30], list(right.bbox))

    def test_enhancer_uses_eye_local_boxes_and_keeps_sbs_layout(self):
        enhancer_path = (Path(__file__).resolve().parents[1] /
                         'modules' / 'processors' / 'frame' / 'face_enhancer.py')
        program = ast.parse(enhancer_path.read_text(encoding='utf-8'))
        node = next(n for n in program.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'process_stereo_frame')
        seen_boxes = []

        def blend(eye, candidate, local_boxes, strength):
            seen_boxes.append((list(local_boxes), strength, eye.shape))
            copy = eye.copy()
            for x1, y1, x2, y2 in local_boxes:
                copy[int(y1):int(y2), int(x1):int(x2)] = 90
            return copy

        context = {
            'Frame': np.ndarray, 'target_boxes': lambda frame, path: [
                (10, 5, 20, 15), (110, 5, 120, 15)
            ],
            'stereo_eye_regions': vr_stereo.stereo_eye_regions,
            'job_control': SimpleNamespace(check_cancelled=lambda: None),
            'modules': SimpleNamespace(
                globals=SimpleNamespace(occlusion_protection=False)
            ),
            'enhanced_candidate': lambda eye, mode: eye + 1,
            'blend_selected_faces': blend,
        }
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(enhancer_path), 'exec'), context)
        input_frame = np.zeros((40, 200, 3), dtype=np.uint8)
        result = context['process_stereo_frame'](input_frame, strength=35)
        self.assertEqual(2, len(seen_boxes))
        self.assertEqual(
            [([(10, 5, 20, 15)], 35, (40, 100, 3)),
             ([(10, 5, 20, 15)], 35, (40, 100, 3))],
            seen_boxes
        )
        self.assertTrue(np.all(result[5:15, 10:20] == 90))
        self.assertTrue(np.all(result[5:15, 110:120] == 90))
        self.assertEqual(0, int(result[25, 40, 0]))

    def test_vr_mode_changes_png_resume_fingerprint_only_when_enabled(self):
        import tempfile
        import modules.globals
        from modules.frames_workflow import job_fingerprint
        original = modules.globals.vr_stereo_mode
        try:
            with tempfile.TemporaryDirectory() as root:
                folder = Path(root)
                photo = folder / 'source.png'
                photo.write_bytes(b'just a filename for this hashing test')
                references = [{
                    'embedding': [1., 0.], 'source_path': str(photo)
                }]
                modules.globals.vr_stereo_mode = False
                base = job_fingerprint(
                    root, str(folder / 'output.mp4'), references, 30, False, 1
                )
                modules.globals.vr_stereo_mode = True
                stereo = job_fingerprint(
                    root, str(folder / 'output.mp4'), references, 30, False, 1
                )
                modules.globals.vr_stereo_mode = False
                again = job_fingerprint(
                    root, str(folder / 'output.mp4'), references, 30, False, 1
                )
                self.assertNotEqual(base, stereo)
                self.assertEqual(base, again)
        finally:
            modules.globals.vr_stereo_mode = original

    def test_settings_default_off_and_survive_round_trip(self):
        import tempfile
        self.assertFalse(DEFAULTS['vr_stereo_mode'])
        self.assertFalse(validate({'vr_stereo_mode': 'yes'})['vr_stereo_mode'])
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'settings.json'
            save_settings({'vr_stereo_mode': True}, target)
            self.assertTrue(load_settings(target)['vr_stereo_mode'])
        dest = SimpleNamespace(fp_ui={})
        apply_to_globals({'vr_stereo_mode': True}, dest)
        self.assertTrue(dest.vr_stereo_mode)

    def test_real_processing_and_preview_use_same_flag(self):
        root = Path(__file__).resolve().parents[1]
        ui = (root / 'modules' / 'ui.py').read_text(encoding='utf-8')
        swapper = (root / 'modules' / 'processors' / 'frame' / 'face_swapper.py').read_text('utf-8')
        enhancer = (root / 'modules' / 'processors' / 'frame' / 'face_enhancer.py').read_text('utf-8')
        png_workflow = (root / 'modules' / 'frames_workflow.py').read_text('utf-8')
        self.assertIn("text='VR / Stereo mode (side-by-side)'", ui)
        self.assertIn("picker_stereo_faces(frame, get_many_faces)", ui)
        self.assertIn('if modules.globals.vr_stereo_mode:', swapper)
        self.assertIn('swap_stereo_assigned_faces(', swapper)
        self.assertIn('swap_assigned_faces(', swapper)
        self.assertIn('if modules.globals.vr_stereo_mode:', enhancer)
        self.assertIn('return process_stereo_frame(', enhancer)
        self.assertIn("if modules.globals.vr_stereo_mode:", png_workflow)
        self.assertIn("record['vr_stereo_mode'] = True", png_workflow)
        self.assertIn('face_swapper.prepare_multi_assignments()', ui)
        for text in (ui, swapper, enhancer, png_workflow):
            ast.parse(text)


if __name__ == '__main__':
    unittest.main()
