"""Repair missing or unreadable PNG frames between processing stages.

Failed frame writes (e.g., full disks) remain fatal. Only missing/corrupt
frames are replaced, using the nearest valid frame from the SAME stage.
Recovery is logged because a duplicated frame cannot guarantee anonymity.
"""
import bisect
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from PIL import Image

import modules.globals
from modules import job_control


def expected_frame_paths(folder: str, expected_count: int = 0) -> List[str]:
    directory = Path(folder)
    existing = [
        int(p.stem) for p in directory.glob('*.png')
        if p.stem.isdigit() and int(p.stem) > 0
    ]
    upper = max([expected_count] + existing)
    if upper <= 0:
        raise RuntimeError(f'No video frames found in {folder}')
    return [str(directory / f'{n:04d}.png') for n in range(1, upper + 1)]


def png_is_readable(path: str) -> bool:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
        with Image.open(path) as image:
            image.load()  # Decode all pixels; opening the header is not enough.
            return image.format == 'PNG'
    except (OSError, ValueError, SyntaxError):
        return False


def repair_frames(paths: Iterable[str], stage: str) -> int:
    job_control.check_cancelled()
    frames = list(paths)
    if not frames:
        raise RuntimeError('There are no frames to repair')

    valid = [i for i, path in enumerate(frames) if png_is_readable(path)]
    if not valid:
        raise RuntimeError(f'No usable frames at stage {stage}. Cannot recover video.')

    good = set(valid)
    repaired = 0
    for index, path in enumerate(frames):
        job_control.check_cancelled()
        if index in good:
            continue
        pos = bisect.bisect_left(valid, index)
        choices = []
        if pos > 0:
            choices.append(valid[pos - 1])
        if pos < len(valid):
            choices.append(valid[pos])
        donor = min(choices, key=lambda i: (abs(i - index), i))
        source = frames[donor]

        temp = f'{path}.recovery.tmp'
        try:
            shutil.copyfile(source, temp)
            os.replace(temp, path)
        except OSError as error:
            raise RuntimeError(
                f'Cannot repair frame {path}. Verify disk space and permissions.'
            ) from error
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

        # Preserve the selected face mask from the donor when available.
        boxes = modules.globals.frame_target_boxes.get(os.path.abspath(source))
        if boxes is not None:
            modules.globals.frame_target_boxes[os.path.abspath(path)] = list(boxes)
        item = {
            'stage': stage,
            'frame': os.path.basename(path),
            'copied_from': os.path.basename(source),
        }
        modules.globals.frame_recovery_log.append(item)
        repaired += 1

    if repaired:
        print(
            f'[REACTOR.RECOVERY] {stage}: replaced {repaired} missing/corrupt '
            'frame(s) with nearest valid frames. Review the final video.'
        )
    return repaired
