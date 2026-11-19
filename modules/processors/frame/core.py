import sys
import importlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm
from modules import progress_events, job_control

import modules
import modules.globals                   

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                sys.exit()
    except ImportError:
        sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES

def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.globals.fp_ui.items():
        if state == True and frame_processor not in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
            modules.globals.frame_processors.append(frame_processor)
        if state == False:
            frame_processor_module = load_frame_processor_module(frame_processor)
            try:
                FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                modules.globals.frame_processors.remove(frame_processor)
            except:
                pass

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    """Stop scheduling frames on Abort; join all in-flight native calls first."""
    job_control.check_cancelled()
    executor = ThreadPoolExecutor(max_workers=modules.globals.execution_threads)
    futures = []
    try:
        for path in temp_frame_paths:
            job_control.check_cancelled()
            futures.append(executor.submit(process_frames, source_path, [path], progress))
        pending = set(futures)
        while pending:
            job_control.check_cancelled()
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
        job_control.check_cancelled()
    finally:
        if job_control.requested():
            for future in futures:
                future.cancel()
        # Wait for already-running model inference to finish. The UI must not
        # unlock a new job until those threads really have stopped.
        executor.shutdown(wait=True, cancel_futures=job_control.requested())


class FrameProgress(tqdm):
    """Preserve the console bar while mirroring completed frames to the GUI."""
    stage = 'Processing frames'

    def update(self, n=1):
        result = super().update(n)
        progress_events.report(self.stage, self.n, self.total or 0)
        return result


def process_video(source_path: str, frame_paths: list[str],
                  process_frames: Callable[[str, List[str], Any], None],
                  stage: str = 'Processing frames') -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    job_control.check_cancelled()
    progress_events.report(stage, 0, total)
    with FrameProgress(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.stage = stage
        progress.set_postfix({'execution_providers': modules.globals.execution_providers, 'execution_threads': modules.globals.execution_threads, 'max_memory': modules.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)
        progress_events.report(stage, progress.n, total)
