"""Abort must stop work without discarding the current video's mappings/settings."""
import ast
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

from modules import job_control
import modules.globals


ROOT = Path(__file__).resolve().parents[1]


class JobCancellationTests(unittest.TestCase):
    def tearDown(self):
        job_control.request_abort()
        job_control.finish()

    def test_cancel_and_restart_preserves_all_user_inputs(self):
        original = (modules.globals.target_path, modules.globals.face_assignments,
                    modules.globals.enhancer_strength)
        sample_mapping = [{'embedding': [0.5, 0.5], 'source_path': 'source.png',
                           'target_crop': 'kept'}]
        try:
            modules.globals.target_path = 'selected.mp4'
            modules.globals.face_assignments = sample_mapping
            modules.globals.enhancer_strength = 70
            job_control.begin()
            self.assertTrue(job_control.active())
            self.assertTrue(job_control.request_abort())
            with self.assertRaises(job_control.ProcessingAborted):
                job_control.check_cancelled()
            self.assertTrue(job_control.active())  # GUI remains locked.
            job_control.finish()
            self.assertFalse(job_control.active())
            self.assertEqual('selected.mp4', modules.globals.target_path)
            self.assertIs(sample_mapping, modules.globals.face_assignments)
            self.assertEqual(70, modules.globals.enhancer_strength)

            job_control.begin()
            job_control.check_cancelled()
            self.assertFalse(job_control.requested())
        finally:
            modules.globals.target_path, modules.globals.face_assignments, \
                modules.globals.enhancer_strength = original

    def test_abort_terminates_registered_subprocess(self):
        child = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(20)'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            job_control.begin()
            job_control.register_process(child)
            self.assertTrue(job_control.request_abort())
            child.wait(timeout=5)
            self.assertIsNotNone(child.returncode)
            job_control.unregister_process(child)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_new_job_cannot_start_until_previous_worker_has_finished(self):
        job_control.begin()
        job_control.request_abort()
        with self.assertRaisesRegex(RuntimeError, 'already active'):
            job_control.begin()
        job_control.finish()
        job_control.begin()
        job_control.check_cancelled()

    def test_pool_stops_scheduling_and_joins_in_flight_workers(self):
        from modules.processors.frame import core
        previous_threads = modules.globals.execution_threads
        modules.globals.execution_threads = 2
        started = threading.Event()
        release = threading.Event()
        failures = []

        def frame_task(source, paths, progress):
            started.set()
            release.wait(timeout=5)
            job_control.check_cancelled()

        def run_pool():
            try:
                core.multi_process_frame(None, list(range(120)), frame_task)
            except job_control.ProcessingAborted:
                failures.append('cancelled')

        try:
            job_control.begin()
            thread = threading.Thread(target=run_pool)
            thread.start()
            self.assertTrue(started.wait(timeout=5))
            job_control.request_abort()
            release.set()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), 'in-flight frame workers should finish')
            self.assertEqual(['cancelled'], failures)
        finally:
            release.set()
            modules.globals.execution_threads = previous_threads

    def test_abort_ui_wiring_and_unchanged_processing_state(self):
        source = (ROOT / 'modules' / 'ui.py').read_text('utf-8')
        tree = ast.parse(source)
        methods = {
            item.name: ast.get_source_segment(source, item)
            for item in tree.body if isinstance(item, ast.FunctionDef)
        }
        self.assertIn('abort_or_close', methods)
        self.assertIn('request_abort', methods)
        self.assertIn("text='Abort'", methods['launch_background'])
        self.assertIn("text='Close'", methods['_drain_events'])
        self.assertIn('job_control.request_abort()', methods['request_abort'])
        self.assertIn('job_control.finish()', methods['launch_background'])
        self.assertIn("'aborted': aborted", methods['launch_background'])
        self.assertIn('Processing aborted.', methods['_drain_events'])
        self.assertNotIn('face_assignments.clear', methods['request_abort'])
        self.assertNotIn('target_path =', methods['request_abort'])
        self.assertNotIn('clean_temp(', methods['request_abort'])


if __name__ == '__main__':
    unittest.main()
