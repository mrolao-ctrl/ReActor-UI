"""Stable Windows taskbar identity for the installed and source ReActor GUI.

The process must share the same AppUserModelID as the Inno Setup shortcuts.
Call before importing CustomTkinter or creating a window.
"""
import sys

APP_USER_MODEL_ID = 'MRolao.ReActorStudio'


def set_windows_app_id() -> bool:
    """Set the AppUserModelID on Windows. Non-Windows platforms are untouched."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes

        call = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        call.argtypes = (ctypes.c_wchar_p,)
        call.restype = ctypes.c_long
        result = call(APP_USER_MODEL_ID)
        return result == 0  # S_OK
    except (AttributeError, OSError, ValueError, TypeError):
        # Branding must never prevent the actual ReActor engine from opening.
        return False
