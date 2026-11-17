"""Make NSFW rejection explicit instead of shutting the GUI without a reason.

These tests inspect the pure status-reporting function without importing the
large GUI/AI runtime or using any media files.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


CORE_PATH = Path(__file__).resolve().parents[1] / 'modules' / 'core.py'


def compile_status_function():
    tree = ast.parse(CORE_PATH.read_text(encoding='utf-8'))
    status = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'report_content_rejection'
    )
    only_function = ast.Module(body=[status], type_ignores=[])
    return compile(ast.fix_missing_locations(only_function), str(CORE_PATH), 'exec')


class SafetyStatusTests(unittest.TestCase):
    def test_rejected_video_reports_reason_in_headless_mode(self):
        status = Mock()
        env = {
            'modules': SimpleNamespace(globals=SimpleNamespace(headless=True)),
            'ui': SimpleNamespace(ROOT=None),
            'update_status': status,
        }
        exec(compile_status_function(), env)
        env['report_content_rejection']('video')
        self.assertEqual(1, status.call_count)
        self.assertIn('video', status.call_args.args[0])
        self.assertIn('No face swap was performed', status.call_args.args[0])

    def test_content_detector_branches_no_longer_call_destroy(self):
        tree = ast.parse(CORE_PATH.read_text(encoding='utf-8'))
        start = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'start'
        )
        checks = [
            node for node in ast.walk(start)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id in ('predict_video', 'predict_image')
        ]
        self.assertEqual(2, len(checks))
        for check in checks:
            function_names = [
                n.func.id for stmt in check.body for n in ast.walk(stmt)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            ]
            self.assertIn('report_content_rejection', function_names)
            self.assertNotIn('destroy', function_names)
            self.assertTrue(any(isinstance(stmt, ast.Return) for stmt in check.body))


if __name__ == '__main__':
    unittest.main()
