"""Regression: WinError 448 after Face Swap must not discard the processed PNGs.

Only metadata detection is exercised; no AI models or real Windows volume.
"""
import ast
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from modules import utilities as util


class ProbeTests(unittest.TestCase):
    def test_successful_ffprobe_uses_existing_media_launcher(self):
        process = MagicMock()
        process.communicate.return_value = (b'30000/1001\n', b'')
        process.returncode = 0
        with patch.object(util, '_start_ffmpeg', return_value=process) as start, \
             patch.object(util.job_control, 'register_process') as register, \
             patch.object(util.job_control, 'unregister_process') as unregister, \
             patch.object(util, '_detect_fps_with_opencv') as fallback:
            fps = util.detect_fps('C:/target/video.mp4')
        self.assertAlmostEqual(29.97002997, fps, places=6)
        self.assertEqual('ffprobe', start.call_args.args[0][0])
        self.assertEqual(subprocess.PIPE, start.call_args.kwargs['stdout'])
        self.assertEqual(subprocess.PIPE, start.call_args.kwargs['stderr'])
        register.assert_called_once_with(process)
        unregister.assert_called_once_with(process)
        fallback.assert_not_called()

    def test_windows_winerror448_falls_back_to_opencv_without_reprocessing(self):
        error = OSError('The path cannot be traversed because it contains an untrusted mount point')
        error.winerror = 448
        with patch.object(util, '_start_ffmpeg', side_effect=error) as start, \
             patch.object(util, '_detect_fps_with_opencv', return_value=25.0) as cv:
            actual = util.detect_fps('C:/video.mp4')
        self.assertEqual(25.0, actual)
        start.assert_called_once()
        cv.assert_called_once_with('C:/video.mp4')

    def test_retry_failure_runtime_error_uses_opencv(self):
        with patch.object(util, '_start_ffmpeg',
                          side_effect=RuntimeError('WinError 448: untrusted mount point')), \
             patch.object(util, '_detect_fps_with_opencv', return_value=23.976):
            self.assertEqual(23.976, util.detect_fps('C:/video.mp4'))

    def test_ffprobe_exit_failure_falls_back_to_opencv(self):
        process = MagicMock()
        process.communicate.return_value = (b'', b'Cannot read stream metadata')
        process.returncode = 1
        with patch.object(util, '_start_ffmpeg', return_value=process), \
             patch.object(util, '_detect_fps_with_opencv', return_value=24.0):
            self.assertEqual(24.0, util.detect_fps('C:/video.mp4'))

    def test_if_both_probes_fail_user_is_told_how_to_preserve_frames(self):
        error = OSError('untrusted mount point')
        error.winerror = 448
        with patch.object(util, '_start_ffmpeg', side_effect=error), \
             patch.object(util, '_detect_fps_with_opencv',
                          side_effect=RuntimeError('OpenCV failed')):
            with self.assertRaisesRegex(RuntimeError, 'All processed PNG frames remain') as context:
                util.detect_fps('C:/movie.mp4')
        self.assertIn('temp', str(context.exception))
        self.assertIn('OpenCV failed', str(context.exception))

    def test_opencv_fallback_releases_capture_after_read(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 29.97
        with patch('cv2.VideoCapture', return_value=capture):
            fps = util._detect_fps_with_opencv('C:/movie.mp4')
        self.assertAlmostEqual(29.97, fps)
        capture.release.assert_called_once()

    def test_reactor_close_cannot_wipe_frames_after_failed_export(self):
        source = (Path(__file__).resolve().parents[1] / 'modules' / 'core.py').read_text('utf-8')
        tree = ast.parse(source)
        destroy = next(n for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == 'destroy')
        block = ast.fix_missing_locations(ast.Module(body=[destroy], type_ignores=[]))
        called = Mock()
        cleanup = Mock()
        namespace = {'quit': called, 'clean_temp': cleanup}
        exec(compile(block, '<core destroy>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / '0001.png'
            saved.write_bytes(b'processed frame remains')
            namespace['destroy']()
            self.assertEqual(b'processed frame remains', saved.read_bytes())
        called.assert_called_once()
        cleanup.assert_not_called()

    def test_30fps_fallback_if_probe_returns_invalid_number(self):
        process = MagicMock()
        process.communicate.return_value = (b'0/0', b'')
        process.returncode = 0
        with patch.object(util, '_start_ffmpeg', return_value=process), \
             patch.object(util, '_detect_fps_with_opencv', return_value=30.0):
            self.assertEqual(30.0, util.detect_fps('C:/movie.mp4'))


if __name__ == '__main__':
    unittest.main()
