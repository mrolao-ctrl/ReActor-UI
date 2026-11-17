"""Test input-only PNG recovery and restart-safe frame processing."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import modules.globals
from modules.frames_workflow import input_sequence, create_work_folder, job_fingerprint


class ExistingFramesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        self.photo = self.root / 'source.png'
        self.write_image(self.photo, 100)
        self.old_settings = {
            'face_assignments': modules.globals.face_assignments,
            'source_path': modules.globals.source_path,
            'enhancer_strength': modules.globals.enhancer_strength,
            'enhancer_mode': modules.globals.enhancer_mode,
            'video_quality': modules.globals.video_quality,
            'video_encoder': modules.globals.video_encoder,
            'many_faces': modules.globals.many_faces,
        }
        modules.globals.face_assignments = [
            {'source_path': str(self.photo), 'embedding': np.array([1, 0])}
        ]
        modules.globals.enhancer_strength = 40
        modules.globals.enhancer_mode = 'GFPGAN'
        modules.globals.video_quality = 22
        modules.globals.video_encoder = 'libx264'
        modules.globals.many_faces = False

    def tearDown(self):
        for name, value in self.old_settings.items():
            setattr(modules.globals, name, value)
        self.tmp.cleanup()

    @staticmethod
    def write_image(path, color):
        Image.fromarray(np.full((16, 16, 3), color, dtype=np.uint8)).save(path)

    def test_missing_first_and_corrupt_middle_are_recovered_from_png_only(self):
        self.write_image(self.inputs / '0002.png', 11)
        (self.inputs / '0003.png').write_bytes(b'not-a-valid-PNG')
        self.write_image(self.inputs / '0004.png', 77)
        resolved, repairs = input_sequence(str(self.inputs))
        self.assertEqual(4, len(resolved))
        self.assertEqual(['0002.png', '0002.png', '0002.png', '0004.png'],
                         [Path(path).name for path in resolved])
        self.assertEqual(2, len(repairs))
        self.assertEqual(b'not-a-valid-PNG', (self.inputs / '0003.png').read_bytes())

    def test_work_folder_cannot_be_reused_with_different_settings(self):
        out = str(self.root / 'final.mp4')
        fingerprint = job_fingerprint(str(self.inputs), out,
                                      modules.globals.face_assignments, 30, False, 2)
        work = create_work_folder(out, fingerprint)
        self.assertEqual(work, create_work_folder(out, fingerprint))
        with self.assertRaisesRegex(RuntimeError, 'incompatible'):
            create_work_folder(out, 'changed-' + fingerprint)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                         'FFmpeg and ffprobe are required for the MP4 integration test')
    def test_resume_only_incomplete_frames_and_preserve_inputs(self):
        from modules.frames_workflow import run_existing_frames
        self.write_image(self.inputs / '0001.png', 11)
        self.write_image(self.inputs / '0003.png', 33)
        unchanged = (self.inputs / '0001.png').read_bytes()
        output = str(self.root / 'out.mp4')
        calls = []

        def replace(source, frame, **kwargs):
            calls.append(kwargs['frame_path'])
            return frame + np.uint8(1)

        fake_swapper = types.ModuleType('modules.processors.frame.face_swapper')
        fake_swapper.prepare_multi_assignments = lambda: [{'source_face': 'fake'}]
        fake_swapper.process_frame = replace
        fake_enhancer = types.ModuleType('modules.processors.frame.face_enhancer')
        fake_enhancer.process_frame = lambda source, frame, **kwargs: frame

        with mock.patch.dict(sys.modules, {
            'modules.processors.frame.face_swapper': fake_swapper,
            'modules.processors.frame.face_enhancer': fake_enhancer,
        }):
            run_existing_frames(str(self.inputs), output, 24, False)
            self.assertEqual(3, len(calls))
            self.assertTrue(Path(output).is_file())
            self.assertEqual(unchanged, (self.inputs / '0001.png').read_bytes())

            run_existing_frames(str(self.inputs), output, 24, False)
            self.assertEqual(3, len(calls))  # Everything reused.

            work = Path(output + '.frames')
            (work / '0002.png').unlink()
            run_existing_frames(str(self.inputs), output, 24, False)
            self.assertEqual(4, len(calls))  # Reprocessed only frame 2.
            self.assertEqual(unchanged, (self.inputs / '0001.png').read_bytes())


if __name__ == '__main__':
    unittest.main()
