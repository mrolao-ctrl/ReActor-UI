"""Resume-safe workflow that uses existing PNG frames, not a source video.

Input PNGs are NEVER modified. Fully processed PNGs are written atomically to
an output-specific .frames folder and are reused on restart. Frame recovery
uses only neighbouring input PNGs, with no attempt to access original media.
"""
import bisect
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
import subprocess

from tqdm import tqdm

import modules.globals
from modules.frame_io import read_image, write_frame
from modules.frame_recovery import png_is_readable


def input_sequence(folder: str) -> tuple:
    directory = Path(folder).resolve()
    if not directory.is_dir():
        raise RuntimeError(f'Frames folder not found: {directory}')
    names = {}
    for path in directory.glob('*.png'):
        if path.stem.isdigit() and int(path.stem) >= 1:
            index = int(path.stem)
            if index in names:
                raise RuntimeError(f'Duplicate frame number in {directory}: {index}')
            names[index] = path
    if not names:
        raise RuntimeError(f'No numbered PNG frames in {directory}')
    # If 0001.png is absent, recover it from the nearest readable PNG.
    # FFmpeg still receives a complete 1..N sequence.
    count = max(names)
    valid = {}
    dimensions = {}
    for number, path in tqdm(sorted(names.items()), desc='Checking input PNGs',
                             unit='frame'):
        if png_is_readable(str(path)):
            valid[number] = str(path)
            with Image.open(path) as image:
                dimensions[number] = image.size
    if not valid:
        raise RuntimeError('All input frames are missing or corrupt')
    standard_size = Counter(dimensions.values()).most_common(1)[0][0]
    for number in list(valid):
        if dimensions[number] != standard_size:
            print(f'[FRAMES] Wrong dimensions in {number:04d}.png; '
                  'recovering from neighboring frames.', flush=True)
            del valid[number]
    good = sorted(valid)
    resolved = []
    repairs = []
    for number in range(1, count + 1):
        if number in valid:
            resolved.append(valid[number])
            continue
        pos = bisect.bisect_left(good, number)
        choices = []
        if pos > 0:
            choices.append(good[pos - 1])
        if pos < len(good):
            choices.append(good[pos])
        donor = min(choices, key=lambda i: (abs(i - number), i))
        resolved.append(valid[donor])
        repairs.append({'frame': f'{number:04d}.png',
                        'copied_from': Path(valid[donor]).name})
    return resolved, repairs


def job_fingerprint(folder: str, output_path: str, sources: list,
                    fps: float, enhancer_enabled: bool, count: int) -> str:
    folder_path = Path(folder).resolve()
    files = sorted((f.name, f.stat().st_size, f.stat().st_mtime_ns)
                   for f in folder_path.glob('*.png') if f.stem.isdigit())
    references = []
    for item in sources:
        path = Path(item['source_path']).resolve()
        stat = path.stat()
        references.append({
            'path': str(path),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'embedding': [float(n) for n in item['embedding']],
        })
    record = {
        'version': 1,
        'folder': str(folder_path),
        'output_path': str(Path(output_path).resolve()),
        'input_frames': files,
        'count': count,
        'references': references,
        'fps': fps,
        'enhancer_enabled': enhancer_enabled,
        'enhancer_strength': modules.globals.enhancer_strength,
        'enhancer_mode': modules.globals.enhancer_mode,
        'video_quality': modules.globals.video_quality,
        'video_encoder': modules.globals.video_encoder,
        'many_faces': modules.globals.many_faces,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest()


def create_work_folder(output_path: str, fingerprint: str) -> Path:
    destination = Path(str(Path(output_path).resolve()) + '.frames')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'job.json'
    if manifest.exists():
        stored = json.loads(manifest.read_text(encoding='utf-8'))
        if stored.get('fingerprint') != fingerprint:
            raise RuntimeError(
                f'Found an incompatible previous job in {destination}. '
                'Choose a different output filename; no files were overwritten.'
            )
    elif any(destination.iterdir()):
        raise RuntimeError(
            f'The work folder is not empty and has no valid job.json: {destination}. '
            'Choose another output filename to preserve existing results.'
        )
    else:
        temp = destination / 'job.json.tmp'
        temp.write_text(json.dumps({'fingerprint': fingerprint}, indent=2),
                        encoding='utf-8')
        os.replace(str(temp), str(manifest))
    return destination


def assemble_frames(folder: Path, output_path: str, fps: float, count: int) -> None:
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + '.partial.mp4')
    if partial.exists():
        partial.unlink()  # Prior incomplete export, not a user source file.
    cmd = [
        'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
        '-framerate', str(fps), '-start_number', '1',
        '-i', str(folder / '%04d.png'),
        '-frames:v', str(count),
        '-an', '-c:v', modules.globals.video_encoder or 'libx264',
        '-crf', str(modules.globals.video_quality if modules.globals.video_quality is not None else 18),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-progress', 'pipe:1', '-stats_period', '1', str(partial)
    ]
    print(f'[FRAMES] Encoding {count} frames at {fps} FPS (no audio)...', flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            line = line.strip()
            if line.startswith('frame='):
                number = int(line.split('=', 1)[1].strip())
                print(f'[FRAMES] Encoded {number}/{count} ({number/count*100:.1f}%)',
                      flush=True)
            elif line and not line.startswith(('fps=', 'stream_', 'bitrate=', 'total_size=',
                                               'out_time', 'dup_frames=', 'drop_frames=',
                                               'speed=', 'progress=', 'quality=')):
                print('[FFmpeg]', line, flush=True)
        exit_code = process.wait()
    if exit_code != 0 or not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(
            f'FFmpeg export failed (code {exit_code}). Completed PNGs are preserved '
            f'in {folder} for resume.'
        )
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(partial)
    ], check=True, capture_output=True, text=True)
    try:
        actual = int(probe.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f'Could not verify video frame count: {probe.stdout}') from error
    if actual != count:
        raise RuntimeError(f'Export has {actual} frames; expected {count}. Keeping PNGs.')
    os.replace(str(partial), str(destination))


def run_existing_frames(
    frames_folder: str,
    output_path: str,
    fps: float,
    enhancer_enabled: bool = False
) -> None:
    """Create a silent MP4 from PNGs; resume completed output frames on retry."""
    if not output_path.lower().endswith('.mp4'):
        raise ValueError('Existing frames mode requires a .mp4 output path')
    if not 1 <= fps <= 120:
        raise ValueError('FPS must be between 1 and 120')

    from modules.processors.frame import face_enhancer, face_swapper

    input_paths, repairs = input_sequence(frames_folder)
    if not modules.globals.face_assignments and not modules.globals.source_path:
        raise RuntimeError('Choose a source face or add one or more person assignments')

    prepared = (
        face_swapper.prepare_multi_assignments()
        if modules.globals.face_assignments else None
    )
    single_source = None if prepared else face_swapper.load_source_face(
        modules.globals.source_path
    )
    sources = modules.globals.face_assignments or [{
        'source_path': modules.globals.source_path,
        'embedding': (
            modules.globals.target_face_embedding.tolist()
            if modules.globals.target_face_embedding is not None else []
        ),
    }]
    fingerprint = job_fingerprint(frames_folder, output_path, sources, fps,
                                  enhancer_enabled, len(input_paths))
    work = create_work_folder(output_path, fingerprint)
    print(f'[FRAMES] Input remains untouched: {frames_folder}')
    print(f'[FRAMES] Resume folder: {work}')
    if repairs:
        print(f'[FRAMES] {len(repairs)} missing/corrupt input frames '
              'will be based on nearby PNGs.')

    # Each output frame is only written ONCE, after both swap and enhancer.
    # An interrupted / corrupted output frame is rebuilt from untouched input.
    skipped = 0
    with tqdm(total=len(input_paths), desc='Swap existing frames',
              unit='frame', dynamic_ncols=True) as bar:
        for index, original in enumerate(input_paths, 1):
            output_frame = work / f'{index:04d}.png'
            if png_is_readable(str(output_frame)):
                skipped += 1
                bar.update(1)
                continue
            frame = read_image(original, 'existing PNG input')
            swapped = face_swapper.process_frame(
                single_source, frame, prepared_assignments=prepared,
                frame_path=str(output_frame)
            )
            if enhancer_enabled:
                swapped = face_enhancer.process_frame(
                    None, swapped, frame_path=str(output_frame)
                )
            write_frame(str(output_frame), swapped)
            bar.update(1)

    print(f'[FRAMES] Reused {skipped}/{len(input_paths)} completed frames.')
    assemble_frames(work, output_path, fps, len(input_paths))
    report = {
        'source_folder': str(Path(frames_folder).resolve()),
        'completed_frames': len(input_paths),
        'already_completed_before_run': skipped,
        'input_recovery': repairs,
        'warning': (
            'Duplicated frames may show unmodified faces. Inspect the result '
            'if the purpose is to hide an identity.'
        )
    }
    if repairs:
        report_path = output_path + '.recovery.json'
        Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'[FRAMES] Recovery report: {report_path}')
    print(f'[FRAMES] Video complete: {output_path}')
