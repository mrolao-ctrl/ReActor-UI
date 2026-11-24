import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

import modules.globals
from modules import progress_events, job_control

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def _windows_subprocess_kwargs() -> dict:
    """Suppress console windows from FFmpeg/FFprobe launched by pythonw.exe.

    This does not suppress stdout/stderr pipes or in-app progress events.
    Windows-only: never pass CREATE_NO_WINDOW to other operating systems.
    """
    if platform.system().lower() == 'windows':
        return {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)}
    return {}


def _start_ffmpeg(commands: List[str], **kwargs) -> subprocess.Popen:
    """Spawn FFmpeg, recovering from Windows untrusted-mount launch errors.

    WinError 448 is reported by CreateProcess *before* FFmpeg can inspect
    the input video or output frames. The initial launch remains unchanged.
    Only if Windows rejects it do we retry the resolved ffmpeg.exe path from
    a regular per-user working directory. The input/output paths are not
    changed, preserving per-video PNG folders and resume behaviour.
    """
    kwargs = {**_windows_subprocess_kwargs(), **kwargs}
    try:
        return subprocess.Popen(commands, **kwargs)
    except OSError as first_error:
        if (platform.system().lower() != 'windows'
                or getattr(first_error, 'winerror', None) != 448):
            raise

        original_cwd = os.getcwd()
        located = shutil.which(commands[0])
        if not located:
            raise RuntimeError(
                f'Windows blocked the FFmpeg launch (WinError 448). '
                f'FFmpeg is not available on PATH. Working directory: '
                f'{original_cwd}.'
            ) from first_error
        try:
            resolved = os.path.realpath(located)
        except OSError:
            resolved = located

        work_root = os.environ.get('LOCALAPPDATA') or os.environ.get('TEMP')
        if not work_root:
            raise RuntimeError(
                f'Windows blocked FFmpeg (WinError 448). '
                f'FFmpeg executable: {located}. Working directory: '
                f'{original_cwd}. Configure a standard LOCALAPPDATA directory.'
            ) from first_error
        safe_cwd = Path(work_root) / 'ReActor-Studio' / 'ffmpeg-work'
        try:
            safe_cwd.mkdir(parents=True, exist_ok=True)
            print(
                f'[REACTOR.FFMPEG] WinError 448 starting {located} from '
                f'{original_cwd}; retrying {resolved} from {safe_cwd}',
                flush=True
            )
            return subprocess.Popen(
                [resolved, *commands[1:]], cwd=str(safe_cwd), **kwargs
            )
        except OSError as retry_error:
            raise RuntimeError(
                f'Windows prevented FFmpeg from starting (WinError 448: '
                f'untrusted mount point). Original executable: {located}. '
                f'Resolved executable: {resolved}. Original working '
                f'directory: {original_cwd}. Retry directory: {safe_cwd}. '
                f'Retry failed: {retry_error}. '
                f'Install ffmpeg.exe in a normal local Windows folder '
                f'(not a symbolic link/junction) and add it to PATH.'
            ) from retry_error


def run_ffmpeg(args: List[str], progress_stage: str = None,
               progress_total: int = 0, *, use_hwaccel: bool = True,
               raise_on_error: bool = False) -> bool:
    """Run FFmpeg normally in CLI; monitor/terminate it during GUI jobs."""
    job_control.check_cancelled()
    commands = ['ffmpeg', '-hide_banner']
    if use_hwaccel:
        commands.extend(['-hwaccel', 'auto'])
    commands.extend(['-loglevel', modules.globals.log_level])
    if progress_stage:
        commands.extend(['-progress', 'pipe:1', '-stats_period', '1'])
    commands.extend(args)

    try:
        if not progress_stage and not job_control.active():
            # Preserve the original direct command behaviour in headless mode.
            subprocess.check_output(
                commands, stderr=subprocess.STDOUT,
                **_windows_subprocess_kwargs()
            )
            return True

        if progress_stage:
            progress_events.report(progress_stage, 0, progress_total)
        with _start_ffmpeg(
            commands, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=bool(progress_stage), bufsize=1 if progress_stage else -1
        ) as process:
            job_control.register_process(process)
            try:
                if progress_stage:
                    for line in process.stdout:
                        job_control.check_cancelled()
                        if line.startswith('frame='):
                            try:
                                frame = int(line.split('=', 1)[1].strip())
                            except ValueError:
                                continue
                            progress_events.report(progress_stage, frame, progress_total)
                    output = None
                    result = process.wait()
                else:
                    output, _ = process.communicate()
                    result = process.returncode
                job_control.check_cancelled()
                if result:
                    raise subprocess.CalledProcessError(result, commands, output=output)
            finally:
                job_control.unregister_process(process)
        if progress_stage:
            progress_events.report(progress_stage, progress_total, progress_total)
        return True
    except (OSError, subprocess.SubprocessError) as error:
        job_control.check_cancelled()
        if raise_on_error:
            output = getattr(error, 'output', None)
            if isinstance(output, bytes):
                details = output.decode('utf-8', errors='replace')
            elif isinstance(output, str):
                details = output
            else:
                details = ''
            details = '\n'.join(details.strip().splitlines()[-12:])[-1800:]
            exit_code = getattr(error, 'returncode', None)
            prefix = (f'FFmpeg exited with code {exit_code}' if exit_code is not None
                      else f'Could not start FFmpeg: {error}')
            raise RuntimeError(
                f'{prefix}.\n{details}' if details else prefix
            ) from error
        return False


def _detect_fps_with_opencv(target_path: str) -> float:
    """Get FPS without spawning a process (Windows mount-point fallback)."""
    import cv2
    import math

    capture = cv2.VideoCapture(target_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f'OpenCV cannot open the selected video: {target_path}')
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.0:
            raise RuntimeError(f'Video does not report a valid FPS: {target_path}')
        return fps
    finally:
        capture.release()


def detect_fps(target_path: str) -> float:
    """Read original FPS; recover safely if Windows refuses to start ffprobe.

    Keep-fps runs *after* face swapping. A failed ffprobe must never force
    another full face-swap pass when OpenCV can read the same video metadata.
    Use the existing WinError 448-aware, windowless subprocess launcher, and
    fall back to OpenCV only when probing fails.
    """
    import math
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1', target_path
    ]
    job_control.check_cancelled()
    try:
        # _start_ffmpeg is shared by the ffmpeg/ffprobe executables. Its
        # Windows retry resolves junctions and uses a safe local working dir.
        # stdout/stderr are pipes; no black terminal window is displayed.
        with _start_ffmpeg(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            job_control.register_process(process)
            try:
                output, errors = process.communicate()
            finally:
                job_control.unregister_process(process)
            job_control.check_cancelled()
            if process.returncode != 0:
                message = errors.decode('utf-8', errors='replace')[-500:]
                raise RuntimeError(
                    f'ffprobe exited with code {process.returncode}: {message}'
                )
        parts = output.decode('utf-8', errors='replace').strip().split('/')
        if len(parts) == 2:
            numerator, denominator = map(int, parts)
            rate = numerator / denominator if denominator else 0.0
        elif len(parts) == 1:
            rate = float(parts[0])
        else:
            rate = 0.0
        if math.isfinite(rate) and rate > 0:
            return rate
        raise RuntimeError('ffprobe returned an invalid frame rate.')
    except (OSError, RuntimeError, ValueError, UnicodeError) as probe_error:
        job_control.check_cancelled()
        print(
            f'[REACTOR.FPS] ffprobe unavailable ({probe_error}); '
            'trying OpenCV without launching another process.',
            flush=True
        )
        try:
            fps = _detect_fps_with_opencv(target_path)
        except (OSError, RuntimeError, ValueError) as cv_error:
            raise RuntimeError(
                f'Could not determine the video FPS. ffprobe: {probe_error}. '
                f'OpenCV: {cv_error}. All processed PNG frames remain in '
                f'{get_temp_directory_path(target_path)}. '
                'No face swap needs to be repeated to export these PNGs.'
            ) from cv_error
        job_control.check_cancelled()
        print(
            f'[REACTOR.FPS] Using OpenCV FPS: {fps:g} '
            '(processed PNGs remain unchanged).',
            flush=True
        )
        return fps


def extract_frames(target_path: str, total_frames: int = 0) -> None:
    job_control.check_cancelled()
    temp_directory_path = get_temp_directory_path(target_path)
    # Decode in software: forced auto-hardware decoding may fail on Windows
    # for formats that otherwise decode correctly. -y also permits a retry
    # after a previous incomplete extraction left numbered PNGs behind.
    # Only enable streaming progress for active GUI jobs; CLI extraction retains
    # its original command and error-reporting behaviour.
    stage = 'Extracting frames' if job_control.active() else None
    try:
        run_ffmpeg(
            ['-y', '-i', target_path, '-pix_fmt', 'rgb24',
             os.path.join(temp_directory_path, '%04d.png')],
            progress_stage=stage, progress_total=max(0, int(total_frames)),
            use_hwaccel=False, raise_on_error=True
        )
    except RuntimeError as error:
        raise RuntimeError(
            f'Could not extract video frames to {temp_directory_path}.\n'
            f'{error}'
        ) from error


def create_video(target_path: str, fps: float = 30.0) -> None:
    job_control.check_cancelled()
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    frame_count = len(glob.glob(os.path.join(glob.escape(temp_directory_path), '*.png')))
    if not run_ffmpeg(['-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.png'), '-c:v', modules.globals.video_encoder, '-crf', str(modules.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path], progress_stage='Encoding MP4', progress_total=frame_count):
        raise RuntimeError(
            f'FFmpeg failed creating the video from {temp_directory_path}. '
            'One or more PNG frames may be missing or corrupt. '
            'Temporary files have been preserved for diagnosis.'
        )


def restore_audio(target_path: str, output_path: str) -> None:
    job_control.check_cancelled()
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.png')))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.normpath(os.path.join(target_directory_path, TEMP_DIRECTORY, target_name))


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def create_temp(target_path: str) -> None:
    job_control.check_cancelled()
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    job_control.check_cancelled()
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    job_control.check_cancelled()
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        job_control.check_cancelled()
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            # A cancelled model download must never masquerade as a valid
            # model file when processing starts again.
            pending = download_file_path + '.part'
            try:
                with urllib.request.urlopen(url) as request:  # type: ignore[attr-defined]
                    total = int(request.headers.get('Content-Length', 0))

                with tqdm(total=total, desc='Downloading', unit='B',
                          unit_scale=True, unit_divisor=1024) as progress:
                    def report_hook(count, block_size, total_size):
                        job_control.check_cancelled()
                        progress.update(block_size)

                    urllib.request.urlretrieve(
                        url, pending, reporthook=report_hook
                    )  # type: ignore[attr-defined]
                job_control.check_cancelled()
                os.replace(pending, download_file_path)
            finally:
                if os.path.exists(pending):
                    os.unlink(pending)


def resolve_relative_path(path: str) -> str:
    # All existing model consumers use ../models[/filename]. Redirect only
    # these paths when a user selected a local models folder in Settings.
    normalized = path.replace('\\', '/')
    if (normalized == '../models' or normalized.startswith('../models/')):
        configured = getattr(modules.globals, 'models_directory', '')
        if configured:
            suffix = normalized[len('../models'):].lstrip('/')
            return os.path.abspath(os.path.join(configured, suffix))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
