# Improved face recognition — ReActor Studio

These improvements target clips where the target person's head is tilted, facial expression changes, or original InsightFace similarity falls below the conservative default match threshold.

## 1. Multiple reference frames — Step 02

1. Map a person as before: select a face in a video frame and choose a source photo.
2. In the person's mapping row, click **+ Reference**.
3. Choose a different frame from the **same target person** with a different pose or facial expression; click the corresponding face rectangle.
4. Repeat for 3–5 good reference frames (maximum 6 total). **Do not** add the same person as multiple people.

Each person retains a **single source photo**. Additional references are target identities, not alternative source photos. The matcher takes the best similarity among the references, while keeping the existing minimum **0.35** and inter-person margin **0.06**. Very similar references from other mapped people are rejected. References, crops and embeddings exist only in local RAM for this job, not in settings or output files. Existing one-reference mappings remain supported.

## 2. Adaptive detection — Step 03

Enable **Adaptive detection (angled faces)** when the ordinary detector misses the relevant person. On a frame where at least one mapped person is unrecognised, the engine tests ±30°, ±45° and ±90°; it maps both **bounding boxes** and **five facial landmarks** back to the original frame coordinates before swapping.

The target identity must still meet the normal similarity/margin requirements. Existing confident matches are not displaced. A face detected after rotation is not automatically considered the desired person.

The face-selection popup also uses rotated detection as a fallback if there are zero ordinary detections. Preview and normal processing use the same adaptive mode. Expect lower processing speed when many frames require extra detections.

## 3. Temporal tracking — Step 03

Enable **Temporal identity tracking (slower)** for videos with temporary drops in identity similarity. The engine processes **Face Swap frames sequentially** when enabled so tracking context cannot be corrupted by out-of-order processing. It follows a recently confidently identified face across **at most two subsequent frames**, with strict overlap (IoU ≥ 0.42), minimum face similarity (≥ 0.18) and separation from other mapped identities (margin ≥ 0.06). Weak matches are never enrolled as new identity references or allowed to prolong tracking indefinitely.

Tracking is separate for left/right VR side-by-side eyes. Missing frames and discontinuities reset tracking. The random-access single-frame Preview doesn't have preceding frames, so temporal tracking isn't active there. It is active during full video processing and sequential existing-PNG processing.

Because tracked video processing must be ordered, this option can be **significantly slower** than the normal multithreaded pipeline. Turn it off if it isn't needed.

## Recommended validation

1. Add 3–5 reference frames for the problem person.
2. Enable **Adaptive detection (angled faces)** in Step 03.
3. Open Preview on the known difficult frame; use **Diagnose faces** to verify improved similarity and accepted mappings, then **Generate preview** to check the visual result.
4. If the person is recognised on nearby frames but missing briefly during playback, enable **Temporal identity tracking** and test a short video first.
5. If a low-confidence or ambiguous face remains unmatched, the application intentionally leaves it unchanged.

The NSFW/content safety gate, source images, output encoding, XSeg occlusion support, normal VR mapping and Abort functionality are unchanged.
