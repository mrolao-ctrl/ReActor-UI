# Optional occlusion protection (hands, glasses and foreground objects)

**Experimental / non-explicit test footage.** This branch adds an optional
XSeg-1 visibility mask to the existing ReActor-UI INSwapper pipeline.

## How it works

When **Occlusion protection (XSeg)** is enabled, ReActor:

1. Detects and matches the target face exactly as before.
2. Uses the face's five landmarks to align a 256×256 crop.
3. Runs the local XSeg-1 ONNX model on the **pre-swap image**.
4. Generates the normal INSwapper replacement.
5. Composites *visible face pixels* from the replacement and foreground
   occluders from the original frame, with softened transitions.
6. When Face Enhancer is enabled, re-computes the mask for the selected face
   before running GFPGAN, so the enhancer does not cover foreground again.

This mode remains OFF by default, so old workflows are unchanged.

The additional model is downloaded to `models/xseg_1.onnx` on the first
use of the feature (official FaceFusion assets release):
https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx

This is an additional CPU inference per swapped face, and another inference
per enhanced face. On a low-power laptop, it can slow processing considerably.
Test a short, ordinary non-explicit clip first.

## Using the UI

- Choose your video or existing PNGs and source/reference photo(s).
- Enable **Occlusion protection (XSeg)**.
- Keep **Face Enhancer** disabled on the first test, or use 30–40%
  once occlusion protection has been visually checked.
- Use **Preview** or **Compare enhancer (1 frame)**.
- Process a short clip and examine glasses, hands and object edges.

For headless mode, add `--occlusion-protection` to the existing command.

For **existing PNG resume**, enable exactly the same protection settings
when restarting. The resume manifest includes this flag and will refuse
to reuse frames produced under incompatible settings.

## Limitations / safety

- XSeg masks only improve blending *where a face is already detected*.
  They do not make InsightFace recognize a fully obscured face.
- Glasses, hair, fast motion or large objects can be mis-segmented.
  If the segmentation or landmarks fail, that face is left unchanged
  rather than forcing a swap over the foreground. The same conservative
  fallback applies to enhancement when the selected face is not found.
- Editing behind an occluder is not visually verifiable; face masking
  must not be treated as guaranteed identity anonymization.
- The inference is local. No video or images are uploaded.
- The XSeg model is distributed separately by FaceFusion assets. The
  user should check the model's licensing terms before other uses.

## Update

This work is on `feature/occlusion-protection` and is NOT merged to main.
After closing the ReActor process:

```powershell
cd C:\Users\Casa\ReActor-Custom
git status --short
git fetch origin
git switch --track origin/feature/occlusion-protection
C:\Users\Casa\ReActor-UI\.venv\Scripts\python.exe run.py --execution-provider cpu --execution-threads 2 --max-memory 6 --video-quality 10 --keep-fps
```

If the branch exists locally, use `git switch feature/occlusion-protection`
and then `git pull --ff-only`. Preserve any local modifications before
switching branches.

**Validation:** automated unit tests use synthetic masks and do not
require real video or an installed XSeg model. A local Windows end-to-end
test with the downloaded model and ordinary footage is still required.
