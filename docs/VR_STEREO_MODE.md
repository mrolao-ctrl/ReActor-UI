# VR / Stereo mode (side-by-side)

## When to enable

Use **03 — Output & Processing → VR / Stereo mode (side-by-side)** when a frame contains the *left-eye view on the left* and the *right-eye view on the right*, showing the same scene from slightly different viewpoints. Leave this switch **Off** for ordinary 2D video or stacked/top-bottom stereo. It defaults to Off.

Map the person **once** in **02 — Map Faces**, selecting a clear target face and the desired source photograph. If two mappings were previously created for the left/right copies of the *same person*, remove the duplicate; two nearly identical reference embeddings compete with one another.

## How matching works

- The original frame is split into two images; left and right eye images are analysed independently, rather than detecting faces against a double-wide panorama.
- Matching reuses a target identity reference for multiple face detections. It still requires a minimum similarity, and for several mapped people requires a clear advantage over other reference identities.
- If one eye recognises a person confidently and the other eye's face is weaker, the weaker counterpart can inherit the match **only if** its relative position, bounding box dimensions and facial embedding look consistent with a stereo partner. Ambiguous or unrelated faces are skipped.
- The face-swap model works in local eye coordinates. Results are reassembled in the unchanged side-by-side layout and at the original resolution.
- Face Enhancer and XSeg, if enabled, also work per-eye. Selected face bounding boxes remain in full-frame coordinates for downstream processing.
- The same logic runs for **Preview**, normal video processing, and existing numbered PNGs.

## Performance and constraints

- Face detection runs on each half, so processing may be slower than ordinary 2D mode.
- A detection can still fail if a face is too distorted, occluded, far away or too small. No replacement is made when matching is uncertain.
- The mode assumes side-by-side eye views; it does not automatically detect stereo format or convert equirectangular VR formats.
- Preview a frame with both faces visible before starting the complete video.
- No mappings, embeddings or source images are uploaded. Target reference data is kept in RAM.
- Existing resumable PNG jobs created before this mode remain compatible when VR is **Off**. VR-on jobs are isolated by a different fingerprint to avoid mixing processed frames.

## Recommended quick test

1. Remove the duplicate mapping, keeping one target person and one source photo.
2. Enable **VR / Stereo mode (side-by-side)** in step 03.
3. Press **Preview**, navigate to a frame where both left and right faces are visible, and choose **Generate preview**.
4. Inspect whether both views were replaced consistently. If one eye remains unchanged, choose a frame where the face has a clearer, more frontal view and recreate the mapping once.
5. When the preview works, start the normal export.
