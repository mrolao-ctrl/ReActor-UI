"""WinError 448 regression: FFmpeg spawn must not depend on untrusted links.

The extraction output is still under the target video, not in the FFmpeg
working directory. All tests are headless, with no Windows requirement.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules import utilities as util


class FFmpegWindowsLaunchTests(unittest.TestCase):
    def _win448(self):
        error = OSError(448, 'untrusted mount point')
        error.winerror = 448
        return error

    def test_successful_launch_does_not_change_existing_behaviour(self):
        process = Mock()
        with patch.object(util.subprocess, 'Popen', return_value=process) as popen:
            self.assertIs(process, util._start_ffmpeg(['ffmpeg', '-version'],
                                                    stdout=-1))
        popen.assert_called_once_with(['ffmpeg', '-version'], stdout=-1)

    def test_retry_resolves_executable_and_uses_safe_windows_cwd(self):
        with tempfile.TemporaryDirectory() as root:
            process = Mock()
            with patch.object(util.platform, 'system', return_value='Windows'), \
                 patch.object(util.shutil, 'which', return_value='C:/links/ffmpeg.exe'), \
                 patch.object(util.os.path, 'realpath', return_value='C:/real/ffmpeg.exe'), \
                 patch.dict(os.environ, {'LOCALAPPDATA': root}), \
                 patch.object(util.subprocess, 'Popen',
                              side_effect=[self._win448(), process]) as popen:
                command = ['ffmpeg', '-hide_banner', '-i', 'C:/System/BaseFS/Pristine.mp4',
                           '-y', 'C:/System/BaseFS/temp/Pristine/%04d.png']
                actual = util._start_ffmpeg(command, stdout=-1)
            self.assertIs(process, actual)
            self.assertEqual(2, popen.call_count)
            initial = popen.call_args_list[0]
            retry = popen.call_args_list[1]
            self.assertEqual(command, initial.args[0])
            self.assertEqual('C:/real/ffmpeg.exe', retry.args[0][0])
            self.assertEqual(command[1:], retry.args[0][1:])
            self.assertEqual(str(Path(root) / 'ReActor-Studio' / 'ffmpeg-work'),
                             retry.kwargs['cwd'])
            self.assertTrue(Path(retry.kwargs['cwd']).is_dir())

    def test_other_errors_do_not_trigger_retry(self):
        with patch.object(util.platform, 'system', return_value='Windows'), \
             patch.object(util.subprocess, 'Popen',
                          side_effect=OSError(2, 'File missing')) as popen:
            with self.assertRaises(OSError):
                util._start_ffmpeg(['ffmpeg', '-version'])
        popen.assert_called_once()

    def test_non_windows_win448_does_not_retry(self):
        with patch.object(util.platform, 'system', return_value='Linux'), \
             patch.object(util.subprocess, 'Popen',
                          side_effect=self._win448()) as popen:
            with self.assertRaises(OSError):
                util._start_ffmpeg(['ffmpeg', '-version'])
        popen.assert_called_once()

    def test_retry_failure_shows_ffmpeg_locations_and_explanation(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(util.platform, 'system', return_value='Windows'), \
                 patch.object(util.shutil, 'which', return_value='C:/link/ffmpeg.exe'), \
                 patch.object(util.os.path, 'realpath', return_value='C:/real/ffmpeg.exe'), \
                 patch.dict(os.environ, {'LOCALAPPDATA': root}), \
                 patch.object(util.subprocess, 'Popen',
                              side_effect=[self._win448(), self._win448()]):
                with self.assertRaises(RuntimeError) as raised:
                    util._start_ffmpeg(['ffmpeg', '-i', 'input.mp4'])
            message = str(raised.exception)
            self.assertIn('untrusted mount point', message)
            self.assertIn('C:/link/ffmpeg.exe', message)
            self.assertIn('C:/real/ffmpeg.exe', message)
            self.assertIn('Retry directory:', message)

    def test_work_frames_remain_by_video_and_are_not_relocated(self):
        with tempfile.TemporaryDirectory() as root:
            target = str(Path(root) / 'Pristine.mp4')
            self.assertEqual(
                str(Path(root) / 'temp' / 'Pristine'),
                util.get_temp_directory_path(target)
            )
            with patch.object(util, 'run_ffmpeg', return_value=True) as run:
                util.extract_frames(target)
            self.assertIn(str(Path(root) / 'temp' / 'Pristine' / '%04d.png'),
                          run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
