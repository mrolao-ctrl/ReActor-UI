"""Headless regression tests for ReActor's five-stage navigation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


UI_PATH = Path(__file__).resolve().parents[1] / 'modules' / 'ui.py'
SOURCE = UI_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


class FakeWidget:
    def __init__(self):
        self.visible = True
        self.state = {}
        self.user_value = 'kept across pages'

    def grid(self):
        self.visible = True

    def grid_remove(self):
        self.visible = False

    def configure(self, **kwargs):
        self.state.update(kwargs)


class FiveStageNavigationTests(unittest.TestCase):
    def setUp(self):
        node = next(
            item for item in TREE.body
            if isinstance(item, ast.FunctionDef)
            and item.name == '_display_workflow_step'
        )
        # Import the UI helper without loading GUI or AI runtime dependencies.
        ctx = {
            'SECONDARY_HOVER': '#222',
            'PURPLE': '#666',
            'BORDER': '#333',
            'TEXT': '#fff',
            'MUTED': '#aaa',
            'WORKFLOW_TITLES': (
                '01   SELECT MEDIA', '02   MAP FACES', '03   PROCESSING',
                '04   OCCLUSION', '05   FACE ENHANCER',
            ),
        }
        module = ast.fix_missing_locations(
            ast.Module(body=[node], type_ignores=[])
        )
        exec(compile(module, str(UI_PATH), 'exec'), ctx)
        self.show = ctx['_display_workflow_step']
        self.cards = [FakeWidget() for _ in range(5)]
        self.tabs = [FakeWidget() for _ in range(5)]
        self.prev = FakeWidget()
        self.next = FakeWidget()
        self.stage = FakeWidget()
        self.canvas = SimpleNamespace(yview_moveto=lambda value: self.scroll.append(value))
        self.scroll = []
        self.scrollable = SimpleNamespace(_parent_canvas=self.canvas)

    def display(self, step):
        return self.show(
            step, self.cards, self.tabs, self.prev,
            self.next, self.stage, self.scrollable
        )

    def test_each_step_displays_only_its_card_and_updates_controls(self):
        for step in range(5):
            self.assertTrue(self.display(step))
            self.assertEqual(
                [index == step for index in range(5)],
                [card.visible for card in self.cards]
            )
            self.assertEqual(
                'disabled' if step == 0 else 'normal',
                self.prev.state['state']
            )
            self.assertEqual(
                'disabled' if step == 4 else 'normal',
                self.next.state['state']
            )
            self.assertIn(f'STEP {step + 1} OF 5', self.stage.state['text'])
        self.assertEqual([0.0] * 5, self.scroll)

    def test_backtracking_preserves_card_widgets_and_values(self):
        self.display(1)
        self.cards[1].user_value = 'person selected'
        self.display(4)
        self.display(1)
        self.assertEqual('person selected', self.cards[1].user_value)
        self.assertTrue(self.cards[1].visible)
        self.assertEqual('#666', self.tabs[1].state['border_color'])

    def test_out_of_range_step_does_not_change_page(self):
        self.display(2)
        self.assertFalse(self.display(-1))
        self.assertFalse(self.display(5))
        self.assertTrue(self.cards[2].visible)
        self.assertEqual(1, len(self.scroll))

    def test_cards_and_header_buttons_follow_requested_order(self):
        root = next(
            item for item in TREE.body
            if isinstance(item, ast.FunctionDef) and item.name == 'create_root'
        )
        segment = ast.get_source_segment(SOURCE, root)
        self.assertIn(
            'workflow_cards = (media, people, processing, occlusion, enhancer)',
            segment
        )
        self.assertIn('command=lambda index=column: show_step(index)', segment)
        self.assertIn('command=lambda: show_step(active_step - 1)', segment)
        self.assertIn('command=lambda: show_step(active_step + 1)', segment)
        self.assertIn('show_step(0)', segment)
        self.assertIn('workflow_tabs.append(chip)', segment)
        self.assertIn('source_card = ctk.CTkFrame(\n        people,', segment)
        self.assertNotIn('source_card = ctk.CTkFrame(\n        media,', segment)
        self.assertIn('columnspan=2, padx=20', segment)

    def test_media_dialog_defaults_to_video_and_retains_images(self):
        function = next(
            item for item in TREE.body
            if isinstance(item, ast.FunctionDef)
            and item.name == 'select_target_path'
        )
        segment = ast.get_source_segment(SOURCE, function)
        self.assertIn('filetypes=[vid_ft, img_ft]', segment)
        self.assertIn("title='Select target video or image'", segment)


if __name__ == '__main__':
    unittest.main()
