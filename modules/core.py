import os
import sys
import json
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import modules.globals
import modules.metadata
from modules.app_settings import load_settings, recommended_models_directory
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.frame_recovery import expected_frame_paths, repair_frames
from modules.capturer import get_video_frame_total
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, get_temp_directory_path, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    saved = load_settings()
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frames-dir', help='use existing numbered PNG frames instead of a video', dest='frames_dir')
    program.add_argument('--frames-fps', help='FPS for existing frames mode (no audio)', dest='frames_fps', type=float, default=saved['frames_fps'])
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper', 'face_enhancer'] if saved['enhancer_enabled'] else ['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action=argparse.BooleanOptionalAction, default=saved['keep_fps'])
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action=argparse.BooleanOptionalAction, default=saved['keep_audio'])
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action=argparse.BooleanOptionalAction, default=saved['keep_frames'])
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default=saved['video_encoder'], choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=saved['video_quality'], choices=range(52), metavar='[0-51]')
    program.add_argument('--enhancer-strength', help='GFPGAN blend strength 0..100', dest='enhancer_strength', type=int, default=saved['enhancer_strength'], choices=range(101))
    program.add_argument('--enhancer-mode', help='face restoration method', dest='enhancer_mode', choices=['gfpgan', 'sharpen'], default='sharpen' if saved['enhancer_mode'] == 'Sharpen only' else 'gfpgan')
    program.add_argument('--occlusion-protection', help='preserve foreground occluders using optional XSeg model', action=argparse.BooleanOptionalAction, default=saved['occlusion_protection'])
    program.add_argument('--occlusion-landmarks', help='XSeg alignment: auto, visible-side eye+mouth pair, 3 or 5 points', choices=['auto', '2-point-left', '2-point-right', '3-point', '5-point'], default=saved['occlusion_landmark_mode'])
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=saved['max_memory'])
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=[saved['execution_provider']], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=saved['execution_threads'])
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.models_directory = saved['models_directory'] or recommended_models_directory()
    modules.globals.source_path = args.source_path
    modules.globals.frames_directory = args.frames_dir
    modules.globals.frames_fps = args.frames_fps
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = bool(args.source_path or args.target_path or args.output_path or args.frames_dir)
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.enhancer_strength = args.enhancer_strength
    modules.globals.enhancer_mode = 'Sharpen only' if args.enhancer_mode == 'sharpen' else 'GFPGAN'
    modules.globals.occlusion_protection = args.occlusion_protection
    modules.globals.occlusion_landmark_mode = args.occlusion_landmarks
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False
    
    modules.globals.nsfw = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'REACTOR.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)


def report_content_rejection(media_type: str) -> None:
    """Reject flagged input with a clear message instead of silently exiting.

    Nothing is swapped or exported. Keep the GUI running so the user can
    select another (non-explicit) input. Headless runs receive console text.
    """
    message = (
        f'Processing stopped: the {media_type} was flagged by the '
        'content-safety classifier. No face swap was performed. '
        'Choose non-explicit footage.'
    )
    update_status(message, 'REACTOR.SAFETY')
    if not modules.globals.headless and ui.ROOT is not None:
        ui.show_warning('ReActor - processing stopped', message)


def start() -> None:
    # Direct PNG mode never extracts frames or reads an original video.
    if modules.globals.frames_directory:
        modules.globals.frame_target_boxes.clear()
        modules.globals.preview_face_boxes = []
        modules.globals.frame_recovery_log.clear()
        if not modules.globals.output_path:
            raise RuntimeError('Select an output MP4 file for existing frames mode')
        from modules.processors.frame import face_swapper
        from modules.frames_workflow import run_existing_frames
        if not face_swapper.pre_start():
            return
        run_existing_frames(
            modules.globals.frames_directory, modules.globals.output_path,
            float(modules.globals.frames_fps),
            bool(modules.globals.fp_ui.get('face_enhancer', False)),
        )
        update_status('Existing frames exported successfully.')
        return

    modules.globals.frame_target_boxes.clear()
    modules.globals.preview_face_boxes = []
    modules.globals.frame_recovery_log.clear()
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw == False:
            update_status('Checking image content before processing...')
            from modules.predicter import predict_image
            if predict_image(modules.globals.target_path):
                report_content_rejection('image')
                return
        shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if modules.globals.nsfw == False:
        update_status('Checking video content; face swap has not started...')
        from modules.predicter import predict_video
        if predict_video(modules.globals.target_path):
            report_content_rejection('video')
            return
    update_status('Creating temp resources...')
    create_temp(modules.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(modules.globals.target_path)
    temp_dir = get_temp_directory_path(modules.globals.target_path)
    temp_frame_paths = expected_frame_paths(temp_dir, get_video_frame_total(modules.globals.target_path))
    repair_frames(temp_frame_paths, 'extraction')
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        repair_frames(temp_frame_paths, frame_processor.NAME)
        release_resources()
    # handles fps
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path)
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # Validate the actual output before removing any temporary frames.
    if not is_video(modules.globals.output_path) or os.path.getsize(modules.globals.output_path) == 0:
        raise RuntimeError(
            'No valid output video was produced. Temporary frames have been '
            'preserved for diagnosis; check the preceding errors.'
        )
    # The output MP4 is assembled before temporary frames are removed.
    if modules.globals.frame_recovery_log:
        report_path = modules.globals.output_path + '.recovery.json'
        with open(report_path, 'w', encoding='utf-8') as report:
            json.dump({
                'warning': (
                    'Frames were duplicated from nearby frames. Some may not '
                    'contain a successful face swap. Check this video visually '
                    'before using it to hide identities.'
                ),
                'recovered_frame_count': len(modules.globals.frame_recovery_log),
                'frames': modules.globals.frame_recovery_log,
            }, report, indent=2)
        update_status(
            f'Video exported with {len(modules.globals.frame_recovery_log)} '
            f'frame repair(s). Inspect {report_path}'
        )
    clean_temp(modules.globals.target_path)
    update_status('Processing to video succeed!')


def destroy() -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    quit()


def prepare_models() -> bool:
    """Download/check models when actually processing, not before Settings opens."""
    try:
        for processor in get_frame_processors_modules(modules.globals.frame_processors):
            if not processor.pre_check():
                return False
        return True
    except (OSError, RuntimeError, ValueError) as error:
        update_status(f'Model preparation failed: {error}')
        return False


def run() -> None:
    parse_args()
    if not pre_check():
        return
    limit_resources()
    if modules.globals.headless:
        if prepare_models():
            start()
    else:
        # The user can open Settings and select an existing models directory
        # before any model download is attempted.
        def start_from_ui():
            if prepare_models():
                start()
        window = ui.init(start_from_ui, destroy)
        window.mainloop()
