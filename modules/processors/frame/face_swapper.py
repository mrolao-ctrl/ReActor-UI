from typing import Any, List
import os
import cv2
import insightface
import threading

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.face_selector import select_matching_face
from modules.multi_face import prepare_assignments, swap_assigned_faces
from modules.face_occlusion import (
    combine_visibility, get_visibility_mask, get_occluder_session, warn_once
)
from modules.frame_io import read_image, write_frame, UnreadableFrameError
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'REACTOR.FACE-SWAPPER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if modules.globals.occlusion_protection:
        try:
            get_occluder_session()
        except (RuntimeError, OSError, ValueError, ImportError) as error:
            update_status(
                f'Occlusion Protection could not load the XSeg model: {error}',
                NAME
            )
            return False
    if modules.globals.face_assignments:
        try:
            prepare_multi_assignments()
        except (RuntimeError, ValueError) as error:
            update_status(str(error), NAME)
            return False
    elif not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    if not modules.globals.face_assignments:
        try:
            if load_source_face(modules.globals.source_path) is None:
                update_status('No face in source path detected.', NAME)
                return False
        except RuntimeError as error:
            update_status(str(error), NAME)
            return False
    if modules.globals.frames_directory:
        return True
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def load_source_face(source_path: str) -> Face:
    image = read_image(source_path, 'reference face photo')
    source_face = get_one_face(image)
    if source_face is None:
        raise RuntimeError(f'No face detected in reference photo: {source_path}')
    return source_face


def prepare_multi_assignments() -> list:
    return prepare_assignments(modules.globals.face_assignments, load_source_face)


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if not modules.globals.occlusion_protection:
        # Preserve the original fast path, byte-for-byte in behaviour.
        return get_face_swapper().get(
            temp_frame, target_face, source_face, paste_back=True
        )

    # Estimate occlusion using the ORIGINAL, unmodified face, then restore
    # the foreground only after the full swap has been generated.
    try:
        visible = get_visibility_mask(temp_frame, target_face)
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        warn_once(f'Could not analyse occlusion; keeping original face: {error}')
        return temp_frame

    swapped = get_face_swapper().get(
        temp_frame, target_face, source_face, paste_back=True
    )
    return combine_visibility(temp_frame, swapped, visible)


def process_frame(source_face: Face, temp_frame: Frame, frame_path=None,
                  prepared_assignments=None) -> Frame:
    # Multiple assignments are always evaluated against the original frame.
    # Later swaps cannot change who the next assignment matches.
    if prepared_assignments is None and isinstance(source_face, list):
        prepared_assignments = source_face
    if prepared_assignments is None and modules.globals.face_assignments:
        # Single frame preview fallback. Video loops prepare this just once.
        prepared_assignments = prepare_multi_assignments()
    if prepared_assignments is not None:
        temp_frame, boxes = swap_assigned_faces(
            temp_frame, prepared_assignments, get_many_faces, swap_face
        )
        if frame_path is None:
            modules.globals.preview_face_boxes = boxes
        else:
            modules.globals.frame_target_boxes[os.path.abspath(frame_path)] = boxes
        return temp_frame

    # Original one-person and many-faces behaviour remains unchanged.
    if modules.globals.target_face_embedding is not None:
        chosen = select_matching_face(
            get_many_faces(temp_frame), modules.globals.target_face_embedding
        )
        targets = [chosen] if chosen is not None else []
    elif modules.globals.many_faces:
        targets = get_many_faces(temp_frame) or []
    else:
        chosen = get_one_face(temp_frame)
        targets = [chosen] if chosen is not None else []

    boxes = [tuple(map(float, face.bbox)) for face in targets]
    for target_face in targets:
        temp_frame = swap_face(source_face, target_face, temp_frame)

    if frame_path is None:
        modules.globals.preview_face_boxes = boxes
    else:
        modules.globals.frame_target_boxes[os.path.abspath(frame_path)] = boxes
    return temp_frame


def process_frames(source_face: Face, temp_frame_paths: List[str], progress: Any = None) -> None:
    for temp_frame_path in temp_frame_paths:
        try:
            temp_frame = read_image(temp_frame_path, 'extracted video frame')
        except UnreadableFrameError as error:
            print(f'[REACTOR.RECOVERY] Will restore unreadable frame after this stage: {error}')
            if progress:
                progress.update(1)
            continue
        result = process_frame(source_face, temp_frame, frame_path=temp_frame_path)
        write_frame(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = prepare_multi_assignments() if modules.globals.face_assignments else load_source_face(source_path)
    target_frame = read_image(target_path, 'target image')
    result = process_frame(source_face, target_frame)
    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f'Could not write image: {output_path}')


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    source_face = prepare_multi_assignments() if modules.globals.face_assignments else load_source_face(source_path)
    modules.processors.frame.core.process_video(source_face, temp_frame_paths, process_frames)
