# Select one person in a video (experimental)

This branch adds **Choose person in video** to the existing ReActor-UI. It preserves
the original face-swap and GFPGAN pipeline.

## UI workflow

1. Load a source face with **Select a face**.
2. Load a target video or image with **Select a target**.
3. Click **Choose person in video**.
4. Move the slider to a frame where the intended person is clearly visible.
   Click **Load frame**, then click their highlighted face rectangle.
5. Keep **Many faces** off and process as usual. Selecting a person turns it off.
6. **Face Enhancer**, **Keep fps**, **Keep audio** and quality settings are unchanged.

The selected person's face embedding is held **only in memory** while the app is
running; selecting a new target video clears the selection. No face image,
embedding or video is committed to Git. Detection and matching are local.

Without a selected person, behaviour is exactly as before: **Many faces OFF**
swaps the leftmost detected face, **Many faces ON** swaps all detected faces.

When selected, the model compares facial embeddings in each frame. It leaves
frames untouched if the best match is weak or ambiguous, rather than swap the
wrong person. Difficult angles or occlusions may cause missed frames. This is
not continuous motion tracking and does **not** guarantee correct identity in
every frame. Inspect the resulting video carefully, especially near crossings
and profile changes.

**Known limitation:** the existing GFPGAN face enhancer may enhance more than
the selected face; this branch changes only which face gets swapped. A later
version can add targeted enhancement.

## Test without overwriting the working installation

Your previous installation can be left intact. Example for Windows PowerShell:

```powershell
cd C:\Users\casa
git clone --branch feature/select-target-person https://github.com/mrolao-ctrl/ReActor-UI.git ReActor-Custom
Copy-Item -Path .\ReActor-UI\models -Destination .\ReActor-Custom\models -Recurse
cd .\ReActor-Custom
C:\Users\casa\ReActor-UI\.venv\Scripts\python.exe run.py --execution-provider cpu --execution-threads 2 --max-memory 6 --video-encoder libx264 --video-quality 10 --keep-fps
```

The copied models are approximately 1 GB. This example reuses your working
Python virtual environment from the original folder; it does not modify it.

Run the selector tests:

```powershell
C:\Users\casa\ReActor-UI\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test with a short non-private video containing two consenting adults first.
Do not commit any private videos, face images, output files, or models to GitHub.

## Technical notes

- Identity reference: the InsightFace `normed_embedding` of the selected face.
- Selection: cosine similarity threshold 0.35, with at least 0.06 separation
  from the second-best face in the same frame.
- CPU inference is supported as before.
- The model download URLs in the fork have been updated to currently published
  model releases, avoiding the old 404s.
- The InsightFace/INSwapper model has separate usage and licensing conditions.
