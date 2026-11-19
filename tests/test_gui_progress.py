"""Progress is emitted without importing Tk or any deep-learning libraries."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from modules import progress_events


ROOT = Path(__file__).resolve().parents[1]


class ProgressTests(unittest.TestCase):
    def tearDown(self):
        progress_events.set_sink(None)
        progress_events.reset()

    def test_throttle_keeps_first_last_and_stage_change(self):
        events = []
        progress_events.set_sink(events.append)
        progress_events.reset()
        with patch('modules.progress_events.time.monotonic',
                   side_effect=[10.0, 10.02, 10.04, 10.30, 11.0]):
            progress_events.report('Face Swap', 0, 207)
            progress_events.report('Face Swap', 1, 207)  # throttled
            progress_events.report('Face Swap', 2, 207)  # throttled
            progress_events.report('Face Swap', 207, 207)  # always delivered
            progress_events.report('Encoding MP4', 0, 207)  # new phase
        self.assertEqual(
            [(e['stage'], e['done']) for e in events],
            [('Face Swap', 0), ('Face Swap', 207), ('Encoding MP4', 0)]
        )
        self.assertEqual(207, events[1]['total'])

    def test_progress_clamps_out_of_range(self):
        events = []
        progress_events.set_sink(events.append)
        progress_events.reset()
        progress_events.report('Frames', 10, 3)
        self.assertEqual(3, events[0]['done'])

    def test_ui_worker_and_events_stay_in_separate_threads(self):
        s = (ROOT / 'modules' / 'ui.py').read_text(encoding='utf-8')
        tree = ast.parse(s)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for required in ('launch_background', '_drain_events',
                         '_set_controls_enabled', 'open_settings', 'show_warning'):
            self.assertIn(required, names)
        self.assertIn("threading.Thread(target=worker", s)
        self.assertIn("ROOT.after(100, _drain_events)", s)
        self.assertIn("_UI_EVENTS.put({'type': 'status'", s)
        self.assertNotIn('ROOT.update()', s)
        self.assertIn("mode='indeterminate'", s)

    def test_existing_png_and_video_publish_frame_progress(self):
        processor = (ROOT / 'modules/processors/frame/core.py').read_text(
            encoding='utf-8'
        )
        frames = (ROOT / 'modules/frames_workflow.py').read_text(
            encoding='utf-8'
        )
        self.assertIn("progress_events.report(self.stage, self.n", processor)
        self.assertIn("'Processing existing PNGs'", frames)
        self.assertIn("'Encoding MP4'", frames)

    def test_safety_dialog_is_enqueued_on_tk_main_thread(self):
        core = (ROOT / 'modules/core.py').read_text(encoding='utf-8')
        self.assertIn("ui.show_warning('ReActor - processing stopped', message)", core)
        self.assertNotIn("messagebox.showwarning('ReActor - processing stopped'", core)


if __name__ == '__main__':
    unittest.main()
