"""Headless regression for a compact Media Library page.

At normal Windows window size the two media actions should fit alongside the
thumbnail instead of pushing them below a vertically scrollable card.
"""
import ast
from pathlib import Path
import unittest


UI_PATH = Path(__file__).resolve().parents[1] / 'modules' / 'ui.py'
SOURCE = UI_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def function_source(name):
    node = next(
        x for x in TREE.body
        if isinstance(x, ast.FunctionDef) and x.name == name
    )
    return ast.get_source_segment(SOURCE, node)


class CompactMediaLayoutTests(unittest.TestCase):
    def test_small_thumbnail_and_both_buttons_share_one_grid_row(self):
        root = function_source('create_root')
        self.assertIn('MEDIA_THUMBNAIL_SIZE = (160, 160)', SOURCE)
        self.assertIn("target_card.grid_columnconfigure(0, weight=2)", root)
        self.assertIn("target_card.grid_columnconfigure(1, weight=3)", root)
        self.assertIn("target_label = ctk.CTkLabel(", root)
        self.assertIn("text='No target selected', height=164, wraplength=260", root)
        self.assertIn("target_label.grid(row=1, column=0", root)
        self.assertIn("target_actions.grid(row=1, column=1", root)
        self.assertIn("target_actions, text='Select target video / image'", root)
        self.assertIn("target_actions, text='Use existing PNG frames'", root)
        self.assertNotIn("text='No target selected', height=210", root)

    def test_target_image_video_and_png_thumbnails_use_same_compact_size(self):
        video = function_source('select_target_path')
        pngs = function_source('select_frames_directory')
        self.assertIn('render_image_preview(target_path, MEDIA_THUMBNAIL_SIZE)', video)
        self.assertIn('render_video_preview(target_path, MEDIA_THUMBNAIL_SIZE)', video)
        self.assertIn('render_image_preview(first, MEDIA_THUMBNAIL_SIZE)', pngs)
        # Source photos in Map Faces remain at their original dimensions.
        source = function_source('select_source_path')
        self.assertIn('render_image_preview(modules.globals.source_path, (200, 200))', source)

    def test_all_other_workflow_cards_and_navigation_are_unchanged(self):
        root = function_source('create_root')
        self.assertIn('workflow_cards = (media, people, processing, occlusion, enhancer)', root)
        self.assertIn('show_step(0)', root)
        self.assertIn('command=open_preview_frame_picker', root)
        self.assertIn('command=select_frames_directory', root)


if __name__ == '__main__':
    unittest.main()
