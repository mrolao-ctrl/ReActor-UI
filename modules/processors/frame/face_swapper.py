from typing import Any, List
import cv2
import insightface
import threading

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.face_selector import select_matching_face
from modules.frame_io import read_image, write_frame
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
    if not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    try:
        if load_source_face(modules.globals.source_path) is None:
            update_status('No face in source path detected.', NAME)
            return False
    except RuntimeError as error:
        update_status(str(error), NAME)
        return False
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


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    # An explicitly selected identity takes priority over the legacy modes.
    # If a frame is ambiguous, do not swap anyone in that frame.
    if modules.globals.target_face_embedding is not None:
        target_face = select_matching_face(
            get_many_faces(temp_frame), modules.globals.target_face_embedding
        )
        if target_face is not None:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    elif modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_face: Face, temp_frame_paths: List[str], progress: Any = None) -> None:
    # The reference face is loaded once for the whole video, not once per frame.
    for temp_frame_path in temp_frame_paths:
        temp_frame = read_image(temp_frame_path, 'extracted video frame')
        result = process_frame(source_face, temp_frame)
        write_frame(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = load_source_face(source_path)
    target_frame = read_image(target_path, 'target image')
    result = process_frame(source_face, target_frame)
    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f'Could not write image: {output_path}')


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    source_face = load_source_face(source_path)
    modules.processors.frame.core.process_video(source_face, temp_frame_paths, process_frames)
