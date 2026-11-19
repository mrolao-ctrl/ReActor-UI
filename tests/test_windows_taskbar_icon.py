"""Regressions for the Windows taskbar icon on the packaged pythonw GUI.

These tests run headless on Linux; visual taskbar behavior still requires
a Windows launch from an installed shortcut.
"""
from pathlib import Path
import ctypes
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from modules import windows_app_identity


ROOT = Path(__file__).resolve().parents[1]
APP_ID = 'MRolao.ReActorStudio'


class WindowsTaskbarIdentityTests(unittest.TestCase):
    def test_explicit_app_identity_is_stable_and_set_before_tk(self):
        launcher = (ROOT / 'run.py').read_text('utf-8')
        self.assertEqual(APP_ID, windows_app_identity.APP_USER_MODEL_ID)
        self.assertLess(
            launcher.index('set_windows_app_id()'),
            launcher.index('from modules import core')
        )

    def test_appid_is_the_same_for_both_installed_shortcuts(self):
        inno = (ROOT / 'installer' / 'ReActor-Studio.iss').read_text('utf-8')
        self.assertIn(f'#define TaskbarAppId "{APP_ID}"', inno)
        self.assertEqual(2, inno.count('AppUserModelID: "{#TaskbarAppId}"'))
        self.assertEqual(2, inno.count('IconFilename: "{app}\\assets\\reactor-studio.ico"'))

    def test_win32_calls_shell_identity_api(self):
        fn = Mock(return_value=0)
        fake = SimpleNamespace(shell32=SimpleNamespace(
            SetCurrentProcessExplicitAppUserModelID=fn
        ))
        with patch.object(windows_app_identity.sys, 'platform', 'win32'), \
             patch.object(ctypes, 'windll', fake, create=True):
            self.assertTrue(windows_app_identity.set_windows_app_id())
        fn.assert_called_once_with(APP_ID)

    def test_non_windows_is_noop(self):
        with patch.object(windows_app_identity.sys, 'platform', 'linux'):
            self.assertFalse(windows_app_identity.set_windows_app_id())

    def test_ui_uses_real_ico_instead_of_tk_photo_on_windows(self):
        ui = (ROOT / 'modules' / 'ui.py').read_text('utf-8')
        start = ui.index('def _set_app_icon(')
        end = ui.index('def _section(', start)
        icon = ui[start:end]
        self.assertIn("if os.name == 'nt':", icon)
        self.assertIn('window.iconbitmap(str(ico))', icon)
        self.assertIn("assets / 'reactor-studio.ico'", icon)
        self.assertIn("assets / 'reactor-studio.png'", icon)
        self.assertIn("format='ICO'", icon)
        self.assertLess(icon.index('window.iconbitmap'), icon.index('window.iconphoto'))
        self.assertIn('_set_app_icon(root)', ui)
        self.assertIn('_set_app_icon(dialog)', ui)
        self.assertIn('_set_app_icon(preview)', ui)


if __name__ == '__main__':
    unittest.main()
