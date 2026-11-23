"""UI regression tests for live FFmpeg frame-extraction progress."""
import ast
from pathlib import Path
import queue
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FakeLabel:
    def __init__(self):
        self.text = ''

    def configure(self, **kwargs):
        if 'text' in kwargs:
            self.text = kwargs['text']


class FakeBar:
    def __init__(self):
        self.value = None

    def cget(self, key):
        if key == 'mode':
            return 'determinate'
        raise AssertionError(key)

    def set(self, value):
        self.value = value


class FakeRoot:
    def winfo_exists(self):
        return True

    def after(self, *args):
        pass


def ui_progress_context():
    source = (ROOT / 'modules' / 'ui.py').read_text('utf-8')
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == '_drain_events'
    )
    namespace = {
        'ROOT': FakeRoot(),
        'queue': queue,
        '_UI_EVENTS': queue.Queue(),
        '_processing_active': True,
        '_abort_requested': False,
        'status_label': FakeLabel(),
        'progress_label': FakeLabel(),
        'progress_bar': FakeBar(),
        '_format_duration': lambda eta: '30s',
    }
    block = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(block, str(ROOT / 'modules' / 'ui.py'), 'exec'), namespace)
    return namespace


class ExtractProgressUITests(unittest.TestCase):
    def test_extraction_displays_current_and_total_with_live_percentage(self):
        context = ui_progress_context()
        context['_UI_EVENTS'].put({
            'type': 'progress', 'stage': 'Extracting frames',
            'done': 125, 'total': 1319, 'rate': 50.0, 'eta': 23.0,
        })
        context['_drain_events']()
        self.assertEqual('Extracting frame 125 of 1,319', context['status_label'].text)
        self.assertIn('Extracting frame 125 of 1,319', context['progress_label'].text)
        self.assertIn('9.5%', context['progress_label'].text)
        self.assertIn('50.00 frames/s', context['progress_label'].text)
        self.assertAlmostEqual(125 / 1319, context['progress_bar'].value)

    def test_extraction_finishes_and_bar_reaches_100_percent(self):
        context = ui_progress_context()
        context['_UI_EVENTS'].put({
            'type': 'progress', 'stage': 'Extracting frames',
            'done': 1319, 'total': 1319, 'rate': 30.0, 'eta': 0,
        })
        context['_drain_events']()
        self.assertEqual('Extracting frame 1,319 of 1,319', context['status_label'].text)
        self.assertIn('100.0%', context['progress_label'].text)
        self.assertEqual(1.0, context['progress_bar'].value)

    def test_encoding_progress_retains_existing_text_and_does_not_overwrite_status(self):
        context = ui_progress_context()
        context['status_label'].text = 'Creating video'
        context['_UI_EVENTS'].put({
            'type': 'progress', 'stage': 'Encoding MP4',
            'done': 100, 'total': 1000, 'rate': 20.0, 'eta': 45,
        })
        context['_drain_events']()
        self.assertEqual('Creating video', context['status_label'].text)
        self.assertIn('Encoding MP4  ·  100/1000 frames', context['progress_label'].text)
        self.assertAlmostEqual(0.1, context['progress_bar'].value)

    def test_frame_total_is_obtained_once_and_reused_for_recovery(self):
        source = (ROOT / 'modules' / 'core.py').read_text('utf-8')
        self.assertIn('total_frames = get_video_frame_total(modules.globals.target_path)', source)
        self.assertIn('extract_frames(modules.globals.target_path, total_frames=total_frames)', source)
        self.assertIn('expected_frame_paths(temp_dir, total_frames)', source)


if __name__ == '__main__':
    unittest.main()
