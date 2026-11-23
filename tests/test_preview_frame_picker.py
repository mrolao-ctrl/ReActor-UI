"""Headless regressions for the Preview frame selection workflow.

The frame picker displays unmodified source frames while scrubbing, then
runs the existing single-frame face-swap preview only on confirmation.
"""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


UI_PATH = Path(__file__).resolve().parents[1] / 'modules' / 'ui.py'
SOURCE = UI_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def source_for(name):
    node = next(n for n in TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(SOURCE, node)


def load_helper(name):
    node = next(n for n in TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    program = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(program, str(UI_PATH), 'exec'), namespace)
    return namespace[name]


class PreviewFrameSelectionTests(unittest.TestCase):
    def test_clamps_fractional_frame_and_outside_range(self):
        index = load_helper('_preview_frame_index')
        self.assertEqual(0, index(-200, 1319))
        self.assertEqual(0, index(0.2, 1319))
        self.assertEqual(1, index(0.8, 1319))
        self.assertEqual(1318, index(1319, 1319))
        self.assertEqual(0, index(200, 0))
        self.assertEqual(0, index(400, 1))

    def test_frame_time_is_based_on_fps_and_has_no_zero_division(self):
        format_time = load_helper('_preview_frame_time')
        self.assertEqual('00:00:00', format_time(0, 25))
        self.assertEqual('00:01:05', format_time(1625, 25))
        self.assertEqual('01:00:00', format_time(108000, 30))
        self.assertEqual('', format_time(200, 0))

    def test_main_preview_button_opens_frame_picker(self):
        root = source_for('create_root')
        self.assertIn("text='Preview', height=44, command=open_preview_frame_picker", root)

    def test_original_frames_are_not_processed_while_scrubbing(self):
        picker = source_for('open_preview_frame_picker')
        # A frame is decoded, scaled for display and selected before inference.
        self.assertIn('ctk.CTkToplevel(ROOT, fg_color=BG)', picker)
        self.assertIn('_bring_dialog_to_front(picker)', picker)
        self.assertIn('FramePreviewReader(', picker)
        self.assertIn('frame_reader.request(index)', picker)
        self.assertIn('frame_reader.latest()', picker)
        self.assertIn('image = Image.fromarray(thumbnail)', picker)
        self.assertIn('min(320, dialog_height - 365)', picker)
        self.assertIn('ctk.CTkSlider(', picker)
        self.assertIn('number_of_steps=max(1, total - 1)', picker)
        self.assertIn('picker.after(100, load_original_frame)', picker)
        self.assertIn('picker.after(60, show_original_result)', picker)
        self.assertIn('picker.after_cancel(state[\'pending\'])', picker)
        self.assertIn("text='Generate preview'", picker)
        self.assertIn("text='Cancel'", picker)
        self.assertIn('PREVIEW_FRAME_INDEX = index', picker)
        self.assertIn('update_preview(', picker)
        self.assertIn('destination_label=frame_label', picker)
        self.assertIn("input_frame=state['frame']", picker)
        self.assertIn('state[\'processed\'] = True', picker)
        self.assertNotIn('show_processed_preview(index)', picker)
        self.assertNotIn('close_picker()\n        ROOT.after(10', picker)
        self.assertNotIn('face_swapper.process_frame(', picker)

    def test_popup_reserves_visible_footer_before_expandable_media(self):
        picker = source_for('open_preview_frame_picker')
        footer_position = picker.index("actions.pack(side='bottom', fill='x'")
        media_position = picker.index("image_area.pack(fill='both', expand=True")
        self.assertLess(footer_position, media_position)
        self.assertIn("ROOT.winfo_screenheight() - 110", picker)
        self.assertIn("picker.minsize(650, 540)", picker)
        self.assertIn("width=760, height=max(200, min(320, dialog_height - 365))", picker)
        # Equal left and right insets for each row, to align Frame +/-
        # with Cancel / Generate preview inside the bordered footer.
        self.assertIn(
            ".grid(row=0, column=0, padx=(14, 0), pady=(12, 4), sticky='w')",
            picker
        )
        self.assertIn(
            ".grid(row=0, column=2, padx=(0, 14), pady=(12, 4), sticky='e')",
            picker
        )
        self.assertIn(
            ".grid(row=1, column=0, padx=(14, 0), pady=(6, 12), sticky='w')",
            picker
        )
        self.assertIn(
            "row=1, column=2, padx=(0, 14), pady=(6, 12), sticky='e'",
            picker
        )
        self.assertIn("text='Generate preview'", picker)
        self.assertIn('fg_color=PURPLE, hover_color=PURPLE_HOVER', picker)

    def test_processed_result_reuses_existing_swap_and_enhancement_pipeline(self):
        render = source_for('update_preview')
        picker = source_for('open_preview_frame_picker')
        self.assertIn('face_swapper.prepare_multi_assignments()', render)
        self.assertIn('processor.process_frame(source, temp_frame)', render)
        self.assertIn('label = destination_label if destination_label is not None else preview_label', render)
        self.assertIn('label.configure(image=displayed, text=\'\')', render)
        self.assertIn('display_bounds or (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)', render)
        self.assertIn('ImageOps.contain(', render)
        self.assertIn('input_frame.copy() if input_frame is not None', render)
        self.assertIn('label.image = displayed', render)
        self.assertIn('generate_button.configure(state=\'disabled\')', picker)
        self.assertIn('picker.update_idletasks()', picker)
        self.assertIn('state[\'processed\'] = True', picker)
        self.assertNotIn('PREVIEW.deiconify()', picker)
        self.assertNotIn('picker.destroy()\n        # Let the picker', picker)

    def test_navigating_away_discards_processed_image_and_loads_original(self):
        picker = source_for('open_preview_frame_picker')
        schedule = picker[picker.index('    def schedule_frame('):picker.index('    picker_slider.configure(')]
        self.assertIn("state['processed'] = False", schedule)
        self.assertIn("if state['processed']:", schedule)
        self.assertIn("frame_label.configure(image=None, text='Loading original frame…')", schedule)
        self.assertIn('frame_label.image = None', schedule)
        self.assertIn("state['frame'] = None", schedule)
        self.assertIn('picker.after(100, load_original_frame)', schedule)
        load = picker[picker.index('    def show_original_result('):picker.index('    def schedule_frame(')]
        self.assertIn("frame_reader.latest()", load)
        self.assertIn('generation == state[\'requested\']', load)
        self.assertIn("state['processed'] = False", load)
        self.assertIn("frame_label.configure(image=shown, text='')", load)
        self.assertIn("generate_button.configure(state='normal', text='Generate preview')", load)

    def test_video_seek_and_cleanup_are_not_performed_in_tk_thread(self):
        picker = source_for('open_preview_frame_picker')
        self.assertNotIn('preview_target_frame(index)', picker)
        self.assertNotIn('cv2.VideoCapture(', picker[picker.index('    def load_original_frame'):])
        self.assertIn('frame_reader.close()', picker)
        self.assertIn("picker.after_cancel(state['poll'])", picker)
        self.assertIn("state['requested'] = None", picker)
        self.assertIn("frame_reader.request(index)", picker)

    def test_enhancer_compare_and_new_media_use_selected_frame(self):
        comparison = source_for('compare_enhancer')
        self.assertIn('_preview_frame_index(PREVIEW_FRAME_INDEX', comparison)
        self.assertIn('PREVIEW_FRAME_INDEX = 0', source_for('select_target_path'))
        self.assertIn('PREVIEW_FRAME_INDEX = 0', source_for('select_frames_directory'))

    def test_processed_window_only_shows_after_successful_preview(self):
        fn = load_helper('show_processed_preview')
        class FakeWindow:
            def __init__(self):
                self.title = Mock()
                self.deiconify = Mock()
                self.lift = Mock()
                self.focus_force = Mock()
        window = FakeWindow()
        fn.__globals__.update(PREVIEW=window, update_preview=Mock(return_value=False))
        fn(35)
        window.deiconify.assert_not_called()
        fn.__globals__['update_preview'] = Mock(return_value=True)
        fn(35)
        window.deiconify.assert_called_once()
        window.title.assert_called_once_with('Processed preview · Frame 36')


if __name__ == '__main__':
    unittest.main()
