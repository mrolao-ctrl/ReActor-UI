"""User settings persist across installs without storing private photographs."""
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules import app_settings


class SettingsTests(unittest.TestCase):
    def test_requested_windows_defaults(self):
        d = app_settings.DEFAULTS
        self.assertEqual(('cpu', 2, 6, 10),
                         (d['execution_provider'], d['execution_threads'],
                          d['max_memory'], d['video_quality']))
        self.assertTrue(d['keep_fps'])
        self.assertFalse(d['keep_frames'])
        self.assertTrue(d['occlusion_protection'])

    def test_settings_round_trip_uses_profile_not_working_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'profile' / 'settings.json'
            values = {'execution_threads': 4, 'video_quality': 8,
                      'occlusion_landmark_mode': '2-point-right'}
            app_settings.save_settings(values, path)
            loaded = app_settings.load_settings(path)
            self.assertEqual(4, loaded['execution_threads'])
            self.assertEqual(8, loaded['video_quality'])
            self.assertEqual('2-point-right', loaded['occlusion_landmark_mode'])
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(app_settings.SETTINGS_VERSION, payload['version'])
            self.assertNotIn('source_photo', payload)
            self.assertNotIn('embedding', payload)

    def test_rejects_untrusted_invalid_settings(self):
        values = app_settings.validate({
            'execution_threads': -1,
            'max_memory': 'bad',
            'video_quality': 200,
            'keep_fps': 'false',
            'occlusion_landmark_mode': 'unavailable'
        })
        for key in ('execution_threads', 'max_memory', 'video_quality',
                    'keep_fps', 'occlusion_landmark_mode'):
            self.assertEqual(app_settings.DEFAULTS[key], values[key])

    def test_corrupt_json_does_not_prevent_opening_app(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'config.json'
            path.write_text('{"foo":', encoding='utf-8')
            self.assertEqual(app_settings.DEFAULTS, app_settings.load_settings(path))

    def test_existing_models_win_over_new_install_path(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            folder = home / 'ReActor-Custom' / 'models'
            folder.mkdir(parents=True)
            (folder / 'inswapper_128.onnx').write_bytes(b'existing locally')
            with patch('modules.app_settings.Path.home', return_value=home):
                # Test the real installed script's preferred old checkout fallback.
                chosen = app_settings.recommended_models_directory()
                if (Path(app_settings.__file__).resolve().parents[1] /
                        'models' / 'inswapper_128.onnx').is_file():
                    self.assertTrue(chosen.endswith('models'))
                else:
                    self.assertEqual(str(folder), chosen)

    def test_in_memory_application_of_persisted_settings(self):
        mock = SimpleNamespace(fp_ui={})
        app_settings.apply_to_globals(
            {'execution_threads': 3, 'enhancer_enabled': True,
             'occlusion_landmark_mode': '3-point'}, mock
        )
        self.assertEqual(3, mock.execution_threads)
        self.assertTrue(mock.fp_ui['face_enhancer'])
        self.assertEqual('3-point', mock.occlusion_landmark_mode)


if __name__ == '__main__':
    unittest.main()
