"""Local, versioned ReActor Studio settings (standard library only).

No personal images, embeddings, or user media ever enter this file.
User settings are independent of application updates. CLI options still win.
"""
import json
import os
from pathlib import Path
import sys

SETTINGS_VERSION = 1
DEFAULTS = {
    'execution_provider': 'cpu',
    'execution_threads': 2,
    'max_memory': 6,
    'video_quality': 10,
    'video_encoder': 'libx264',
    'keep_fps': True,
    'keep_audio': True,
    'keep_frames': False,
    'vr_stereo_mode': False,
    'adaptive_face_detection': False,
    'temporal_face_tracking': False,
    'frames_fps': 30.0,
    'occlusion_protection': True,
    'occlusion_landmark_mode': 'auto',
    'enhancer_enabled': False,
    'enhancer_mode': 'GFPGAN',
    'enhancer_strength': 40,
    'models_directory': '',
}
LANDMARK_MODES = ('auto', '2-point-left', '2-point-right', '3-point', '5-point')
ENCODERS = ('libx264', 'libx265', 'libvpx-vp9')
ENHANCERS = ('GFPGAN', 'Sharpen only')


def settings_path() -> Path:
    root = os.getenv('APPDATA') or (str(Path.home() / 'AppData' / 'Roaming')
                                     if sys.platform == 'win32' else str(Path.home() / '.config'))
    return Path(root) / 'ReActor-Studio' / 'settings.json'


def validate(values: dict) -> dict:
    """Coerce only supported keys; reject invalid values instead of trusting JSON."""
    result = dict(DEFAULTS)
    if not isinstance(values, dict):
        return result
    for key, default in DEFAULTS.items():
        if key not in values:
            continue
        v = values[key]
        try:
            if isinstance(default, bool):
                if not isinstance(v, bool):
                    continue
            elif isinstance(default, int):
                if isinstance(v, bool) or not str(v).isdigit():
                    continue
                v = int(v)
                if not 0 <= v <= (100 if key == 'enhancer_strength' else 51 if key == 'video_quality' else 64):
                    continue
                if key in ('max_memory', 'execution_threads') and v == 0:
                    continue
            elif isinstance(default, float):
                v = float(v)
                if not 1 <= v <= 120:
                    continue
            else:
                if not isinstance(v, str):
                    continue
                if key == 'execution_provider' and v not in ('cpu', 'cuda', 'dml', 'coreml', 'rocm'):
                    continue
                if key == 'occlusion_landmark_mode' and v not in LANDMARK_MODES:
                    continue
                if key == 'video_encoder' and v not in ENCODERS:
                    continue
                if key == 'enhancer_mode' and v not in ENHANCERS:
                    continue
                if key == 'models_directory':
                    v = str(Path(os.path.expandvars(os.path.expanduser(v))).resolve()) if v else ''
            result[key] = v
        except (TypeError, ValueError, OSError, OverflowError):
            continue
    return result


def load_settings(path=None) -> dict:
    path = Path(path) if path is not None else settings_path()
    try:
        content = json.loads(path.read_text(encoding='utf-8'))
        return validate(content)
    except (OSError, ValueError, UnicodeError):
        return dict(DEFAULTS)


def save_settings(values: dict, path=None) -> Path:
    path = Path(path) if path is not None else settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = validate(values)
    payload = {'version': SETTINGS_VERSION, **current}
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                             encoding='utf-8')
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def recommended_models_directory() -> str:
    """Find existing models first, in both source and installed-script layouts.

    The lightweight Inno Setup installer runs normal pythonw.exe, NOT a frozen
    executable, so sys.frozen alone cannot identify installed deployments.
    """
    checkout = Path(__file__).resolve().parents[1] / 'models'
    candidates = [
        checkout,
        Path.home() / 'ReActor-Custom' / 'models',
        Path.home() / 'ReActor-UI' / 'models',
    ]
    for folder in candidates:
        if (folder / 'inswapper_128.onnx').is_file():
            return str(folder)
    return str(Path(os.getenv('LOCALAPPDATA', str(Path.home()))) /
               'ReActor-Studio' / 'models')


def apply_to_globals(settings: dict, globals_module, decode_providers=None) -> None:
    """Apply saved settings to the GUI's in-memory state, no model imports."""
    s = validate(settings)
    for key in ('keep_fps', 'keep_audio', 'keep_frames', 'vr_stereo_mode',
                'adaptive_face_detection', 'temporal_face_tracking',
                'frames_fps', 'max_memory', 'execution_threads', 'video_quality',
                'video_encoder', 'occlusion_protection', 'occlusion_landmark_mode',
                'enhancer_mode', 'enhancer_strength'):
        setattr(globals_module, key, s[key])
    globals_module.models_directory = s['models_directory'] or recommended_models_directory()
    globals_module.fp_ui['face_enhancer'] = s['enhancer_enabled']
    if decode_providers is not None:
        globals_module.execution_providers = decode_providers([s['execution_provider']])
