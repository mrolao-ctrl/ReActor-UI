"""GUI contract checks without importing Tkinter or machine-learning models.

The CI runner is headless: tests assert the layout wiring and dialog mechanics
via Python's AST, while a local Windows launch is needed for pixel-level QA.
"""
import ast
from pathlib import Path
import unittest


UI_PATH = Path(__file__).resolve().parents[1] / 'modules' / 'ui.py'
SOURCE = UI_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def function_body(name):
    node = next(x for x in TREE.body if isinstance(x, ast.FunctionDef) and x.name == name)
    return node, ast.get_source_segment(SOURCE, node)


def widget_text_labels(node, widget):
    labels = []
    for call in ast.walk(node):
        if (not isinstance(call, ast.Call) or
                not isinstance(call.func, ast.Attribute) or
                call.func.attr != widget):
            continue
        for keyword in call.keywords:
            if keyword.arg == 'text' and isinstance(keyword.value, ast.Constant):
                labels.append(keyword.value.value)
    return labels


class ModernInterfaceTests(unittest.TestCase):
    def test_main_screen_uses_mapping_not_legacy_face_buttons(self):
        node, source = function_body('create_root')
        buttons = widget_text_labels(node, 'CTkButton')
        self.assertIn('+  Add person + source photo', buttons)
        self.assertIn('Clear person assignments', buttons)
        self.assertIn('Select target video / image', buttons)
        self.assertIn('Use existing PNG frames', buttons)
        self.assertNotIn('Select a face', buttons)
        self.assertNotIn('Choose person (single)', buttons)
        self.assertNotIn('Many faces', widget_text_labels(node, 'CTkSwitch'))
        self.assertNotIn('many_faces_switch', SOURCE)
        self.assertIn('CTkScrollableFrame', source)
        self.assertIn('Start processing', buttons)
        self.assertIn('Preview', buttons)
        self.assertIn('Close', buttons)

    def test_existing_switches_and_options_are_still_available(self):
        node, _ = function_body('create_root')
        switches = widget_text_labels(node, 'CTkSwitch')
        for name in ('Keep fps', 'Keep audio', 'Keep frames', 'Face Enhancer',
                     'NSFW', 'Occlusion protection (XSeg)'):
            with self.subTest(name=name):
                self.assertIn(name, switches)
        for function in ('select_frames_directory', 'preview_target_frame',
                         'select_output_path', 'update_preview', 'set_many_faces',
                         'select_source_path'):
            function_body(function)  # Legacy logic remains callable if required.

    def test_person_picker_is_raised_and_file_dialog_owned(self):
        node, source = function_body('open_face_picker')
        self.assertIn('_bring_dialog_to_front(picker)', source)
        self.assertIn('parent=picker', source)
        self.assertIn('face_assignments.append', source)
        helper, helper_source = function_body('_bring_dialog_to_front')
        self.assertIn('dialog.transient(ROOT)', helper_source)
        self.assertIn('dialog.lift()', helper_source)
        self.assertIn("dialog.attributes('-topmost', False)", helper_source)

    def test_enhancer_images_full_native_resolution_and_vertical(self):
        node, source = function_body('compare_enhancer')
        self.assertIn('for amount in (0, 40, 70):', source)
        self.assertIn('size=image.size', source)
        self.assertIn("card.pack(anchor='nw'", source)
        self.assertIn("orientation='horizontal'", source)
        self.assertIn("orientation='vertical'", source)
        # The image preview must never be downsized by PIL in this function.
        resized = [
            call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in ('contain', 'fit', 'resize', 'thumbnail')
        ]
        self.assertEqual([], resized)
        self.assertIn('preview.image = shown', source)
        self.assertIn('blend_selected_faces', source)


if __name__ == '__main__':
    unittest.main()
