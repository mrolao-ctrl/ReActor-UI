"""Windows media subprocesses must not flash console windows during processing."""
import ast
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

from modules import utilities as util


CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)


class WindowsQuietSubprocessTests(unittest.TestCase):
    def test_windows_only_create_no_window(self):
        with patch.object(util.platform, 'system', return_value='Windows'):
            self.assertEqual(
                {'creationflags': CREATE_NO_WINDOW},
                util._windows_subprocess_kwargs()
            )
        with patch.object(util.platform, 'system', return_value='Linux'):
            self.assertEqual({}, util._windows_subprocess_kwargs())

    def test_ffmpeg_popen_is_hidden_and_keeps_stdout_pipe(self):
        fake_process = Mock()
        with patch.object(util.platform, 'system', return_value='Windows'), \
             patch.object(util.subprocess, 'Popen', return_value=fake_process) as popen:
            self.assertIs(
                fake_process,
                util._start_ffmpeg(['ffmpeg', '-version'], stdout=subprocess.PIPE)
            )
        self.assertEqual(CREATE_NO_WINDOW, popen.call_args.kwargs['creationflags'])
        self.assertEqual(subprocess.PIPE, popen.call_args.kwargs['stdout'])

    def test_winerror_448_retry_stays_hidden(self):
        win448 = OSError('The path contains an untrusted mount point')
        win448.winerror = 448
        fake_process = Mock()
        with patch.object(util.platform, 'system', return_value='Windows'), \
             patch.object(util.shutil, 'which', return_value='C:/fake/ffmpeg.exe'), \
             patch.object(util.os.path, 'realpath', return_value='C:/real/ffmpeg.exe'), \
             patch.dict('os.environ', {'LOCALAPPDATA': str(Path.cwd())}), \
             patch.object(util.subprocess, 'Popen',
                          side_effect=[win448, fake_process]) as popen:
            self.assertIs(fake_process, util._start_ffmpeg(['ffmpeg', '-version']))
        self.assertEqual(2, popen.call_count)
        for attempted in popen.call_args_list:
            self.assertEqual(CREATE_NO_WINDOW, attempted.kwargs['creationflags'])

    def test_non_progress_ffmpeg_check_output_uses_quiet_flags(self):
        with patch.object(util.platform, 'system', return_value='Windows'), \
             patch.object(util.subprocess, 'check_output', return_value=b'') as call:
            self.assertTrue(util.run_ffmpeg(['-version']))
        self.assertEqual(CREATE_NO_WINDOW, call.call_args.kwargs['creationflags'])
        self.assertEqual(subprocess.STDOUT, call.call_args.kwargs['stderr'])

    def test_ffprobe_fps_uses_quiet_flags(self):
        with patch.object(util.platform, 'system', return_value='Windows'), \
             patch.object(util.subprocess, 'check_output', return_value=b'30/1') as call:
            self.assertEqual(30.0, util.detect_fps('C:/video.mp4'))
        self.assertEqual(CREATE_NO_WINDOW, call.call_args.kwargs['creationflags'])

    def test_png_export_and_probe_use_same_windows_quiet_options(self):
        source = (Path(__file__).resolve().parents[1] /
                  'modules' / 'frames_workflow.py').read_text('utf-8')
        tree = ast.parse(source)
        func = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == 'assemble_frames')
        code = ast.get_source_segment(source, func)
        self.assertIn('**_windows_subprocess_kwargs()', code)
        self.assertIn('subprocess.Popen(', code)
        self.assertIn('subprocess.run([', code)
        self.assertEqual(2, code.count('**_windows_subprocess_kwargs()'))


if __name__ == '__main__':
    unittest.main()
