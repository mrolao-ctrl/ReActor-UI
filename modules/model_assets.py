"""Checks for models required by a ReActor installation without network IO.

Model weights are third-party binaries, not part of this source repository.
These checks are optional and enabled by --offline; the runtime should not
download missing models in that mode. Content-safety checks remain active.
"""
import os
from pathlib import Path

from modules.utilities import resolve_relative_path

# Conservative minimum sizes catch empty files / failed partial downloads.
MIN_BYTES = {
    'inswapper_128.onnx': 500_000_000,
    'xseg_1.onnx': 65_000_000,
    'GFPGANv1.4.pth': 300_000_000,
}


def valid_file(path, minimum_bytes=1) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size >= minimum_bytes
    except OSError:
        return False


def local_models_missing(use_xseg=False, use_enhancer=False) -> list:
    """Files required for a fully offline ReActor run (even with safety checks)."""
    root = Path(resolve_relative_path('../models'))
    expected = ['inswapper_128.onnx']
    if use_xseg:
        expected.append('xseg_1.onnx')
    if use_enhancer:
        expected.append('GFPGANv1.4.pth')
    missing = [
        str(root / filename) for filename in expected
        if not valid_file(root / filename, MIN_BYTES[filename])
    ]

    # InsightFace will otherwise attempt to fetch buffalo_l on first use.
    home = Path(os.environ.get('INSIGHTFACE_HOME', Path.home() / '.insightface'))
    buffalo_dir = home / 'models' / 'buffalo_l'
    try:
        valid_buffalo = buffalo_dir.is_dir() and any(
            valid_file(file, 1024 * 1024)
            for file in buffalo_dir.glob('*.onnx')
        )
    except OSError:
        valid_buffalo = False
    if not valid_buffalo:
        missing.append(str(buffalo_dir / '*.onnx'))

    # OpenNSFW2 also downloads weights implicitly unless present.
    nsfw_home = Path(os.environ.get('OPENNSFW2_HOME', Path.home()))
    nsfw = nsfw_home / '.opennsfw2' / 'weights' / 'open_nsfw_weights.h5'
    if not valid_file(nsfw, 1024 * 1024):
        missing.append(str(nsfw))
    return missing
