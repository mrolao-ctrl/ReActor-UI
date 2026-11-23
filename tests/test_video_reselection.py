"""Regression: selecting a second target after rendering must not lock videos.

Headless tests execute only the pure selection/preview functions via AST.
No customtkinter installation, FFmpeg, ONNX or private media is required.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import cv2
import numpy as np
from PIL import Image, ImageOps


UI = Path(__file__).resolve().parents[1] / 'modules' / 'ui.py'


def extract_function(name, context):
    tree = ast.parse(UI.read_text(encoding='utf-8'))
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    code = compile(ast.fix_missing_locations(ast.Module(
        body=[node], type_ignores=[])), str(UI), 'exec')
    exec(code, context)
    return context[name]


class VideoSelectionRegressionTests(unittest.TestCase):
    def test_video_preview_releases_capture_on_every_successful_selection(self):
        instances = []

        class FakeCapture:
            def __init__(self, _path):
                self.released = False
                instances.append(self)

            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((200, 300, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        context = {
            'cv2': SimpleNamespace(
                VideoCapture=FakeCapture, COLOR_BGR2RGB=cv2.COLOR_BGR2RGB,
                cvtColor=lambda frame, conversion: frame,
            ),
            'ctk': SimpleNamespace(CTkImage=lambda **kwargs: kwargs),
            'Image': Image, 'ImageOps': ImageOps, 'Path': Path,
            'Tuple': tuple
        }
        render = extract_function('render_video_preview', context)
        first = render('one.mp4', (200, 200))
        second = render('two.mp4', (200, 200))
        self.assertTrue(first and second)
        self.assertEqual(2, len(instances))
        self.assertTrue(all(item.released for item in instances))

    def test_capture_is_released_when_frame_cannot_be_read(self):
        class FakeCapture:
            released = False

            def __init__(self, _):
                pass

            def isOpened(self):
                return True

            def read(self):
                return False, None

            def release(self):
                FakeCapture.released = True

        context = {
            'cv2': SimpleNamespace(
                VideoCapture=FakeCapture, COLOR_BGR2RGB=cv2.COLOR_BGR2RGB
            ),
            'ctk': SimpleNamespace(CTkImage=Mock()),
            'Image': Image, 'ImageOps': ImageOps, 'Path': Path,
            'Tuple': tuple
        }
        render = extract_function('render_video_preview', context)
        with self.assertRaisesRegex(RuntimeError, 'Unable to read preview frame'):
            render('unreadable.mp4', (200, 200))
        self.assertTrue(FakeCapture.released)

    def test_bad_new_video_does_not_destroy_previous_selection(self):
        old_assignments = [{'source_path': 'first-person.png'}]
        state = SimpleNamespace(
            target_path='old.mp4', output_path='old-output.mp4',
            frames_directory=None, target_face_embedding='original',
            frame_target_boxes={'original': []},
            preview_face_boxes=['old'], face_assignments=old_assignments
        )
        status = Mock()
        preview = SimpleNamespace(withdraw=Mock())
        label = SimpleNamespace(configure=Mock())
        context = {
            'ctk': SimpleNamespace(filedialog=SimpleNamespace(
                askopenfilename=lambda **kwargs: 'broken.mp4'
            )),
            'modules': SimpleNamespace(globals=state),
            'cv2': cv2, 'os': __import__('os'), 'Path': Path,
            'RECENT_DIRECTORY_TARGET': None, 'FRAMES_PREVIEW_PATHS': ['old.png'],
            'img_ft': ('Image', '*.png'), 'vid_ft': ('Video', '*.mp4'),
            'ROOT': object(), 'PREVIEW': preview, 'target_label': label,
            'is_image': lambda path: False,
            'is_video': lambda path: True,
            'MEDIA_THUMBNAIL_SIZE': (160, 160),
            'render_video_preview': Mock(side_effect=RuntimeError('cannot read')),
            'render_image_preview': Mock(),
            'update_assignments_label': Mock(),
            'update_status': status,
        }
        select = extract_function('select_target_path', context)
        select()
        self.assertEqual('old.mp4', state.target_path)
        self.assertEqual('old-output.mp4', state.output_path)
        self.assertEqual('original', state.target_face_embedding)
        self.assertIs(old_assignments, state.face_assignments)
        preview.withdraw.assert_not_called()
        label.configure.assert_not_called()
        self.assertIn('Cannot open selected target:', status.call_args.args[0])

    def test_new_valid_video_resets_stale_processing_state(self):
        state = SimpleNamespace(
            target_path='old.mp4', output_path='old-output.mp4',
            frames_directory='old-frames', target_face_embedding='original',
            frame_target_boxes={'original': []}, preview_face_boxes=['old'],
            face_assignments=[{'source_path': 'old-person.png'}]
        )
        thumbnail = object()
        label = SimpleNamespace(configure=Mock())
        preview = SimpleNamespace(withdraw=Mock())
        context = {
            'ctk': SimpleNamespace(filedialog=SimpleNamespace(
                askopenfilename=lambda **kwargs: 'next.mp4'
            )),
            'modules': SimpleNamespace(globals=state),
            'cv2': cv2, 'os': __import__('os'), 'Path': Path,
            'RECENT_DIRECTORY_TARGET': None, 'FRAMES_PREVIEW_PATHS': ['old.png'],
            'img_ft': ('Image', '*.png'), 'vid_ft': ('Video', '*.mp4'),
            'ROOT': object(), 'PREVIEW': preview, 'target_label': label,
            'is_image': lambda path: False, 'is_video': lambda path: True,
            'MEDIA_THUMBNAIL_SIZE': (160, 160),
            'render_video_preview': Mock(return_value=thumbnail),
            'render_image_preview': Mock(),
            'update_assignments_label': Mock(), 'update_status': Mock()
        }
        select = extract_function('select_target_path', context)
        select()
        self.assertEqual('next.mp4', state.target_path)
        self.assertIsNone(state.output_path)
        self.assertIsNone(state.frames_directory)
        self.assertIsNone(state.target_face_embedding)
        self.assertEqual([], state.face_assignments)
        self.assertEqual({}, state.frame_target_boxes)
        self.assertEqual([], state.preview_face_boxes)
        self.assertEqual([], context['FRAMES_PREVIEW_PATHS'])
        preview.withdraw.assert_called_once()
        label.configure.assert_called_once_with(image=thumbnail, text='')
        self.assertIs(label.image, thumbnail)


if __name__ == '__main__':
    unittest.main()
