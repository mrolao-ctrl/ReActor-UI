"""Safe local image file operations for frame-based video processing.

A corrupt input frame raises an actionable error instead of passing None to
InsightFace. Frame updates use a temporary PNG plus atomic replacement so a
failed write cannot overwrite an existing, valid frame.
"""
import os
import threading

import cv2


def read_image(path: str, description: str = 'image'):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(
            f'Cannot read {description}: {path}. The file may be missing, '
            'incomplete or corrupted (for example after the disk ran out '
            'of space). Verify the file before restarting processing.'
        )
    return image


def write_frame(path: str, image) -> None:
    """Write a processed PNG safely, keeping the existing file on failure."""
    if image is None:
        raise RuntimeError(f'Refusing to write an empty frame: {path}')
    temp_path = os.path.join(
        os.path.dirname(path),
        f'.{os.path.basename(path)}.{threading.get_ident()}.writing.png'
    )
    try:
        if not cv2.imwrite(temp_path, image):
            raise RuntimeError(f'OpenCV could not write frame: {temp_path}')
        os.replace(temp_path, path)
    except (OSError, cv2.error) as error:
        raise RuntimeError(
            f'Could not save frame: {path}. Check free disk space, '
            'folder permissions and disk health.'
        ) from error
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
