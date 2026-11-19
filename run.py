#!/usr/bin/env python3
"""Launch ReActor Studio with optional persistent logs for console-free Windows."""
import os
from pathlib import Path
import sys

# pythonw.exe does not provide console streams; tqdm and error reporting need one.
if sys.stdout is None or sys.stderr is None:
    log_root = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'ReActor-Studio' / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    stream = open(log_root / 'reactor.log', 'a', encoding='utf-8', buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream

from modules.windows_app_identity import set_windows_app_id

# Assign our identity before importing the GUI or creating a Tk window.
# The Inno Setup shortcuts use the same stable AppUserModelID.
set_windows_app_id()

from modules import core

if __name__ == '__main__':
    core.run()
