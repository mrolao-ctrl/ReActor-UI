# Multiple persons and existing PNG frames (Windows / local CPU)

This experimental feature is on **feature/multi-face-existing-frames**, not main.
It reuses the existing InsightFace / INSwapper / GFPGAN models. No cloud,
upload, or additional AI model is required.

## Workflow A: Multiple people in one new video

1. Choose the **target video** with `Select a target`.
2. Click **Add person + source photo**.
3. Move the frame slider to a clear view of the first person, click the
   marked face, and select a source photo of the replacement identity.
4. Repeat **Add person + source photo** for each other person (e.g. 2).
   The **Mapped people** count updates after each selection.
5. Optionally enable **Face Enhancer** and choose strength (default 40%).
6. Click **Start** and choose the output video path.

The program reads each frame once, detects all faces once, resolves mappings
against the *unmodified* input frame, and then swaps distinct matched
identities. An ambiguous match is skipped instead of overwriting somebody
else. The previous single-face and Many faces workflows remain available.
The original video processing flow still handles frame extraction and audio.

## Workflow B: Reuse already generated PNGs for another person

1. Click **Use existing PNG frames** (e.g.
   `C:\System\BaseFS\temp\vp10`). The input folder remains untouched.
2. Click **Add person + source photo**; choose a clear frame of the
   *second* person, click that face, and select its replacement source photo.
3. Set **Existing frames FPS** (default 30). This mode NEVER reads the
   source video, does not extract frames again, and exports **without audio**.
4. Optionally enable Face Enhancer (this re-enhances only faces swapped
   in THIS run; for already enhanced input, start with it disabled).
5. Click **Start**, choose an output `*.mp4` path.

Output frames are written to `<output.mp4>.frames\0001.png`, etc.
The first person's previously swapped face should remain unchanged if
the second person's identity is matched correctly. Multi-person matching is
conservative: it skips uncertain detections.

### Resume after interruption

Run again with **the same output name, source PNG folder, selected
reference photos/identities, FPS, enhancement settings and encoder quality**.
Frames already completed in the output-specific working folder are reused;
only missing, incomplete or corrupt outputs are regenerated from the
untouched original PNG inputs. No face is re-swapped on a completed frame.

If any settings or sources changed, the manifest intentionally refuses to
reuse the old work folder. Choose a **new output filename**; do not delete
the previous results unless you have verified the MP4.

If the input folder contains missing or corrupt numbered PNGs, the nearest
readable input frame is substituted for processing that frame. The original
input is **not modified**. A recovery report is written as
`<output>.recovery.json`.

PNG output and MP4 files can use substantial disk space. Check available
capacity before running long clips. Interrupted output frame sequences are
preserved for later attempts.

### Security and visual checks

- Face identity features are kept in memory during selection; the
  persisted job manifest stores a SHA-256 fingerprint instead of embeddings.
- A repeated neighbour frame or an uncertain identity match can leave an
  original face visible. Review the output carefully if using face
  swapping for privacy or anonymization.
- This is NOT a guarantee that every face in every frame has been altered.

## Updating the Windows copy (after the old job finishes)

```powershell
cd C:\Users\casa\ReActor-Custom
git status --short
git fetch origin
git switch --track origin/feature/multi-face-existing-frames
C:\Users\casa\ReActor-UI\.venv\Scripts\python.exe -m unittest discover -s tests -v
C:\Users\casa\ReActor-UI\.venv\Scripts\python.exe run.py --execution-provider cpu --execution-threads 2 --max-memory 6 --video-quality 10 --keep-fps
```

If `git status --short` reports local changes, preserve them before
switching branches. If the feature branch already exists locally, use
`git switch feature/multi-face-existing-frames` followed by
`git pull --ff-only` instead of `git switch --track`.

The automated tests on GitHub exercise identity matching, recovering
missing/corrupt PNGs, preserving source frames and resume behavior. The
actual AI models and CustomTkinter UI still require a short local Windows
smoke test before processing long private videos.
