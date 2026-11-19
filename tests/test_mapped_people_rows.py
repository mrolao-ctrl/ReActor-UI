"""Mapped-person rows: target crops, source photos, and one-person removal.

Tests only the small pure UI helpers using AST so no desktop or model
installation is needed in CI.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image, ImageOps

UI_PATH = Path(__file__).resolve().parents[1] / 'modules' / 'ui.py'
SOURCE = UI_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def extract(name, context):
    node = next(
        item for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(UI_PATH), 'exec'
    )
    exec(code, context)
    return context[name]


class MappedPeopleListTests(unittest.TestCase):
    def test_face_crop_uses_original_frame_and_correct_rgb_channels(self):
        crop = extract(
            '_crop_selected_target_face',
            {'Image': Image, 'cv2': cv2}
        )
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[30:50, 35:55] = [17, 43, 201]  # OpenCV BGR
        before = frame.copy()
        cropped = crop(frame, (35, 30, 55, 50))

        self.assertIsInstance(cropped, Image.Image)
        self.assertEqual((24, 24), cropped.size)  # 12% per side
        self.assertEqual((201, 43, 17), cropped.getpixel((12, 12)))
        np.testing.assert_array_equal(before, frame)
        # The crop must not alias the source frame buffer.
        frame[:] = 0
        self.assertEqual((201, 43, 17), cropped.getpixel((12, 12)))

    def test_crop_clips_to_frame_boundaries_and_rejects_empty_boxes(self):
        crop = extract('_crop_selected_target_face', {'Image': Image, 'cv2': cv2})
        frame = np.full((32, 40, 3), 127, dtype=np.uint8)
        clipped = crop(frame, (-3, -4, 20, 20))
        self.assertGreater(clipped.width, 0)
        self.assertGreater(clipped.height, 0)
        self.assertLessEqual(clipped.width, 40)
        self.assertLessEqual(clipped.height, 32)
        self.assertIsNone(crop(frame, (10, 10, 10, 19)))
        self.assertIsNone(crop(frame, (99, 99, 120, 120)))
        self.assertIsNone(crop(None, (0, 0, 1, 1)))

    def test_thumbnail_preserves_original_source_aspect_ratio(self):
        ctx = {
            'Image': Image, 'ImageOps': ImageOps,
            'ctk': SimpleNamespace(CTkImage=lambda **kwargs: kwargs),
        }
        thumbnail = extract('_mapping_thumbnail', ctx)
        image = Image.new('RGB', (400, 200), 'white')
        result = thumbnail(image, (112, 112))
        self.assertEqual((112, 56), result['size'])

    def test_removing_middle_identity_preserves_other_mappings(self):
        remove = extract('_remove_face_assignment', {})
        first = {'source_path': 'one.png', 'embedding': object(), 'target_crop': 'a'}
        middle = {'source_path': 'two.png', 'embedding': object(), 'target_crop': 'b'}
        last = {'source_path': 'three.png', 'embedding': object(), 'target_crop': 'c'}
        mappings = [first, middle, last]
        self.assertIs(remove(mappings, 1), middle)
        self.assertEqual([first, last], mappings)
        self.assertIsNone(remove(mappings, 7))
        self.assertIsNone(remove(mappings, -1))
        self.assertEqual([first, last], mappings)

    def test_list_widgets_and_removal_actions_are_wired(self):
        main = extract_source('create_root')
        update = extract_source('update_assignments_label')
        picker = extract_source('open_face_picker')
        remove = extract_source('remove_assignment')

        self.assertIn('assignments_list = ctk.CTkFrame(source_card', main)
        self.assertIn('MAPPED PEOPLE', main)
        self.assertIn('assignments_list.winfo_children()', update)
        self.assertIn('child.destroy()', update)
        self.assertIn("assignment.get('target_crop')", update)
        self.assertIn('ImageOps.exif_transpose(opened)', update)
        self.assertIn('command=lambda position=index: remove_assignment(position)', update)
        self.assertIn('row.grid(row=index', update)
        self.assertIn('update_assignments_label()', remove)
        self.assertIn("state['frame'] = frame", picker)
        self.assertIn("state['frame_number'] = frame_index + 1", picker)
        self.assertIn("_crop_selected_target_face(state['frame'], face.bbox)", picker)
        self.assertIn("'target_crop': crop", picker)
        self.assertIn("'target_frame': state['frame_number']", picker)
        self.assertIn("modules.globals.face_assignments.append({", picker)


def extract_source(name):
    node = next(
        item for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node)


if __name__ == '__main__':
    unittest.main()
