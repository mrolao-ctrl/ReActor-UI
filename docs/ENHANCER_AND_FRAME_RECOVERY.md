# Face-only enhancement and automatic frame recovery

These changes are experimental and developed for the Windows CPU installation
of the ReActor-UI fork. They use the existing GFPGAN v1.4 and INSwapper models.

## Face Enhancer controls

After loading a source image and target video:

- Enable **Face Enhancer**.
- Select **GFPGAN** (default) or **Sharpen only**.
- Set **Strength** from 0 to 100, with 40% as the initial value.
  - 0%: no enhancement.
  - 40%: 40% blend of the restoration within the selected face region.
  - 100%: the full GFPGAN result within the selected face region.
- Click **Compare enhancer (1 frame)** to compare 0%, 40% and 70% on
  a single frame before processing a whole video.
- If **Choose person in video** is used, the enhancer follows the face
  bounding boxes *recorded before the swap* instead of re-identifying
  the person after their face changed. If the selected person was not
  detected on a frame, do not enhance an arbitrary other face.
- With no selected person, the legacy single-face (leftmost) or
  **Many faces** modes remain available.

The selected face is blended with a feathered elliptical mask. Changes are
limited to the face area; nearby hair or jaw edges may require inspection.
Blending cannot restore details the underlying face-swap model never generated.
**Sharpen only** performs local non-generative sharpening and should usually
change facial structure less than GFPGAN.

On CLI, choose `--enhancer-strength 30` and
`--enhancer-mode sharpen` as needed. Set `--frame-processor
face_swapper face_enhancer` to enable both processors in headless mode.

## Automatic frame repair

The program now checks frame integrity:
1. After extracting the original PNG sequence.
2. After Face Swap.
3. After Face Enhancer.

If a PNG is missing or cannot be decoded, the closest good PNG from the
**same processing stage** is copied to its frame number. The source files for
other frames are not overwritten. This can cause a momentary freeze near
replaced frames, especially for multiple consecutive missing frames.

Failed writes (out of disk space or permission errors) are **not**
recoverable, and terminate with a clear error so more files are not damaged.

When recovery is used, the MP4 is accompanied by a
`<output-file>.recovery.json` file containing frame numbers and donor
file names. Carefully inspect the video if face swapping is being used to
hide a person's identity: duplicated or skipped frames may show an
unmodified face. This is a best-effort export, **not a privacy guarantee**.

## Update the existing Windows copy

Close the app, then use PowerShell:

```powershell
cd C:\Users\casa\ReActor-Custom
git pull origin feature/select-target-person
C:\Users\casa\ReActor-UI\.venv\Scripts\python.exe -m unittest discover -s tests -v
C:\Users\casa\ReActor-UI\.venv\Scripts\python.exe run.py --execution-provider cpu --execution-threads 2 --max-memory 6 --video-quality 10 --keep-fps
```

The working folder `C:\Users\casa\ReActor-UI` remains unchanged. No new
model download is required when GFPGAN and INSwapper are already present in
the custom installation's `models` folder.

Before a longer video, test first with a short non-private video and check
the MP4 and optional recovery report. This branch has not been end-to-end
validated on the user's Windows machine.
