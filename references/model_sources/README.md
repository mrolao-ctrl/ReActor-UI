# Reference source code: INSwapper and XSeg

These files are copies of **publicly available upstream Python sources**,
included so the ReActor-UI fork owner can read how the two models are
implemented. They are **reference-only**, not imported by the ReActor runtime.
They do not include trained weights or recreate the full upstream dependency
trees. Do not mistake them for a working replacement of `.onnx` files.

## INSwapper (InsightFace)

- [Reference copy: `insightface_inswapper.py`](insightface_inswapper.py)
- Original: https://github.com/deepinsight/insightface/blob/c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52/python-package/insightface/model_zoo/inswapper.py
- Upstream revision: `c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52`
- Code license: **MIT** according to the [InsightFace Python library README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#license). The pretrained models **do not have the same unrestricted code license**. InsightFace's project README describes noncommercial research restrictions and directs requests for licensing inswapper models to `contact@insightface.ai`: https://github.com/deepinsight/insightface#license

The file shows preprocessing/alignment, source identity embedding preparation,
ONNX Runtime inference, and pasting/blending the output back into the frame.
It is **not the complete original neural-network training implementation**.
The actual trained network graph and learned parameters come from
`inswapper_128.onnx`, which is **not included**.

Our active app currently calls `insightface.model_zoo.get_model(...).get(...)`
from `modules/processors/frame/face_swapper.py`.

## XSeg (DeepFaceLab)

- [Neural-network architecture: `deepfacelab_XSeg.py`](deepfacelab_XSeg.py)
- [Model runner and weight handling: `deepfacelab_XSegNet.py`](deepfacelab_XSegNet.py)
- [Training and ONNX export: `deepfacelab_Model_XSeg.py`](deepfacelab_Model_XSeg.py)
- Original repository: https://github.com/iperov/DeepFaceLab
- Upstream revision: `e4b7543ffa1d73b26fce1e31852727f658ba490c`
- License: **GPL-3.0**, with the upstream license text retained in
  [`GPL-3.0-DeepFaceLab.txt`](GPL-3.0-DeepFaceLab.txt).

These files use the old DeepFaceLab/TensorFlow `core.leras` framework and
**are not runnable as standalone scripts in this fork**. They document the
network architecture, training logic, and mask-inference machinery.
Our active code instead calls the pretrained `xseg_1.onnx` through
ONNX Runtime in `modules/face_occlusion.py`.

## Read vs. run

| What | Included here? | Required for actual face swapping? |
| --- | --- | --- |
| Upstream INSwapper Python wrapper | Yes (reference) | Provided by installed InsightFace |
| Original INSwapper training source | **Not publicly provided as part of these sources** | No, for inference |
| INSwapper weights (`inswapper_128.onnx`) | **No** | **Yes** |
| XSeg network/training/runner source | Yes (reference) | No, for ONNX inference |
| XSeg weights (`xseg_1.onnx`) | **No** | **Yes**, when protection is enabled |

**Offline operation:** Models already installed locally can be used without
network access. Merely adding these Python files to GitHub does not bundle the
model weights, Python packages, InsightFace `buffalo_l` models, FFmpeg or
content-classifier weights. See `feature/offline-models` for separate,
unfinished work on explicit offline checking.

These copies are not modified except for small attribution comments at the
top of each file. Their original licensing, authorship, and redistribution
obligations remain with the upstream copyright holders.
