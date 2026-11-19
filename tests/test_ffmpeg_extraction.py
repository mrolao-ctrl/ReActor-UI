"""Regression tests for FFmpeg frame extraction and useful Windows errors."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import modules.utilities as util


class FFmpegExtractionTests(unittest.TestCase):
    def test_video_extraction_uses_software_decoder_and_overwrites_old_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            video = str(Path(directory) / 'M_R_1.mp4')
            with patch.object(util, 'run_ffmpeg', return_value=True) as run:
                util.extract_frames(video)
            args, kwargs = run.call_args
            self.assertEqual('-y', args[0][0])
            self.assertEqual(['-i', video, '-pix_fmt', 'rgb24'], args[0][1:5])
            self.assertEqual(
                os.path.join(util.get_temp_directory_path(video), '%04d.png'),
                args[0][-1]
            )
            self.assertFalse(kwargs['use_hwaccel'])
            self.assertTrue(kwargs['raise_on_error'])

    def test_software_decode_disables_forced_hardware_acceleration(self):
        with patch.object(util.subprocess, 'check_output', return_value=b'') as output:
            self.assertTrue(util.run_ffmpeg(
                ['-y', '-i', 'movie.mp4', '%04d.png'],
                use_hwaccel=False, raise_on_error=True
            ))
            command = output.call_args.args[0]
            self.assertNotIn('-hwaccel', command)
            self.assertIn('-y', command)
            self.assertIn('-i', command)

    def test_non_extraction_commands_preserve_previous_default_hwaccel(self):
        with patch.object(util.subprocess, 'check_output', return_value=b'') as output:
            self.assertTrue(util.run_ffmpeg(['-i', 'test.mp4', 'output.mp4']))
            self.assertIn('-hwaccel', output.call_args.args[0])

    def test_error_includes_ffmpeg_stderr_instead_of_guessing_disk_space(self):
        failure = subprocess.CalledProcessError(
            1, ['ffmpeg'], output=b'[png @ 0x001] Permission denied\nError opening output file'
        )
        with patch.object(util.subprocess, 'check_output', side_effect=failure):
            with self.assertRaises(RuntimeError) as context:
                util.run_ffmpeg(['-i', 'test.mp4'], use_hwaccel=False,
                                raise_on_error=True)
        message = str(context.exception)
        self.assertIn('FFmpeg exited with code 1', message)
        self.assertIn('Permission denied', message)
        self.assertIn('Error opening output file', message)
        self.assertNotIn('Check disk space', message)

    def test_extract_frames_passes_diagnostic_to_caller(self):
        with patch.object(util, 'run_ffmpeg', side_effect=RuntimeError(
            'FFmpeg exited with code 1.\nInvalid data found when processing input'
        )):
            with self.assertRaises(RuntimeError) as context:
                util.extract_frames('sample.mp4')
        self.assertIn('Invalid data found', str(context.exception))
        self.assertIn('Could not extract video frames', str(context.exception))

    def test_temp_folder_is_normalized_and_uses_input_video_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, 'movie.mp4')
            folder = util.get_temp_directory_path(target)
            self.assertEqual(
                os.path.normpath(os.path.join(directory, 'temp', 'movie')), folder
            )

    def test_non_diagnostic_ffmpeg_still_returns_false_on_failure(self):
        error = subprocess.CalledProcessError(1, ['ffmpeg'], output=b'error')
        with patch.object(util.subprocess, 'check_output', side_effect=error):
            self.assertFalse(util.run_ffmpeg(['-i', 'bad.mp4', 'out.mp4']))


if __name__ == '__main__':
    unittest.main()
