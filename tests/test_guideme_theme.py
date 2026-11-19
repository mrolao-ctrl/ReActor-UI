"""Guard the GuideMe-inspired ReActor visual identity and installer icon."""
import ast
import json
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class GuideMeVisualThemeTests(unittest.TestCase):
    def test_theme_uses_actual_guideme_dark_palette(self):
        theme = json.loads((ROOT / 'modules' / 'ui.json').read_text('utf-8'))
        self.assertEqual('#050816', theme['CTk']['fg_color'][0])
        self.assertEqual('#6D5CFF', theme['CTkButton']['fg_color'][0])
        self.assertEqual('#6D5CFF', theme['CTkProgressBar']['progress_color'][0])
        self.assertGreaterEqual(theme['CTkButton']['corner_radius'], 10)

    def test_icon_is_valid_and_sized_for_desktop_header(self):
        path = ROOT / 'assets' / 'reactor-studio.png'
        with Image.open(path) as icon:
            self.assertEqual('PNG', icon.format)
            self.assertEqual(icon.size[0], icon.size[1])
            self.assertGreaterEqual(icon.size[0], 96)
            self.assertLess(icon.getpixel((0, 0))[0], 80)

    def test_branded_header_keeps_all_existing_controls(self):
        ui = (ROOT / 'modules' / 'ui.py').read_text('utf-8')
        ast.parse(ui)
        for token in (
            "set_appearance_mode('dark')", "'#050816'", "'#6D5CFF'",
            "text='REACTOR'", "LOCAL WORKSPACE",
            "assets' / 'reactor-studio.png'",
            "Select target video / image", "Use existing PNG frames",
            "Add person + source photo", "Clear person assignments",
            "Occlusion protection (XSeg)", "Compare enhancer",
            "Start processing", "Preview", "Close",
            "launch_background(start)", "progress_bar",
            "open_settings"
        ):
            with self.subTest(token=token):
                self.assertIn(token, ui)

    def test_installer_and_windows_build_use_new_icon(self):
        script = (ROOT / 'installer' / 'ReActor-Studio.iss').read_text('utf-8')
        workflow = (ROOT / '.github' / 'workflows' /
                    'windows-installer.yml').read_text('utf-8')
        self.assertIn('SetupIconFile=..\\assets\\reactor-studio.ico', script)
        self.assertIn('IconFilename:', script)
        self.assertIn('reactor-studio.png', script)
        self.assertIn('reactor-studio.ico', script)
        self.assertIn('Generate the Windows ICO', workflow)
        self.assertIn("assets/reactor-studio.png", workflow)


if __name__ == '__main__':
    unittest.main()
