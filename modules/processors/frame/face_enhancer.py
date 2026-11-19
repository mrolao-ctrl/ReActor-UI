from typing import Any, List
import cv2
import os
import threading
import gfpgan

from modules import job_control
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.face_occlusion import (
    combine_visibility, get_visibility_mask, select_faces_by_boxes, warn_once
)
import numpy as np
from modules.enhancement import blend_selected_faces, sharpen_frame
from modules.frame_io import read_image, write_frame, UnreadableFrameError
from modules.typing import Frame, Face
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'REACTOR.FACE-ENHANCER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    ])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)
    return FACE_ENHANCER


def enhanced_candidate(temp_frame: Frame, mode: str = None) -> Frame:
    """Compute a full-strength candidate once; blend only chosen faces later."""
    if mode is None:
        mode = modules.globals.enhancer_mode
    if mode == 'Sharpen only':
        return sharpen_frame(temp_frame)
    with THREAD_SEMAPHORE:
        _, _, restored = get_face_enhancer().enhance(temp_frame, paste_back=True)
    if restored is None:
        raise RuntimeError('GFPGAN returned no restored image')
    return restored


def target_boxes(temp_frame: Frame, frame_path=None) -> list:
    if frame_path is None:
        boxes = modules.globals.preview_face_boxes
    else:
        boxes = modules.globals.frame_target_boxes.get(os.path.abspath(frame_path))
    if boxes is not None and (frame_path is not None or boxes):
        return boxes

    # If a person was selected, we must NOT fall back to the leftmost face
    # after the swap: that face may have a different identity by now.
    if modules.globals.target_face_embedding is not None:
        return []

    face = get_one_face(temp_frame)
    return [tuple(map(float, face.bbox))] if face is not None else []


def process_frame(
    source_face: Face,
    temp_frame: Frame,
    frame_path=None,
    strength: int = None,
    mode: str = None
) -> Frame:
    if strength is None:
        strength = modules.globals.enhancer_strength
    if mode is None:
        mode = modules.globals.enhancer_mode
    if int(strength) <= 0:
        return temp_frame
    boxes = target_boxes(temp_frame, frame_path)
    if not boxes:
        return temp_frame

    visibility = None
    if modules.globals.occlusion_protection:
        # GFPGAN may otherwise paint over a hand/object restored by the swap.
        # Match the pre-swap face boxes by location, NOT by changed identity.
        faces = select_faces_by_boxes(get_many_faces(temp_frame), boxes)
        if not faces:
            warn_once('Cannot locate the selected face for safe enhancement; skipping')
            return temp_frame
        try:
            visibility = np.ones(temp_frame.shape[:2], dtype=np.float32)
            for face in faces:
                visibility = np.minimum(
                    visibility, get_visibility_mask(temp_frame, face)
                )
        except (RuntimeError, OSError, ValueError, cv2.error) as error:
            warn_once(f'Enhancer occlusion mask failed; keeping original: {error}')
            return temp_frame

    candidate = enhanced_candidate(temp_frame, mode)
    enhanced = blend_selected_faces(temp_frame, candidate, boxes, strength)
    if visibility is not None:
        return combine_visibility(temp_frame, enhanced, visibility)
    return enhanced


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    for temp_frame_path in temp_frame_paths:
        job_control.check_cancelled()
        try:
            temp_frame = read_image(temp_frame_path, 'frame for face enhancement')
        except UnreadableFrameError as error:
            print(f'[REACTOR.RECOVERY] Will restore unreadable frame after this stage: {error}')
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame, frame_path=temp_frame_path)
        job_control.check_cancelled()
        write_frame(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = read_image(target_path, 'target image for face enhancement')
    result = process_frame(None, target_frame)
    job_control.check_cancelled()
    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f'Could not write image: {output_path}')


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    job_control.check_cancelled()
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames, stage='Face Enhancer')
