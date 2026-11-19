# ReActor Studio — Windows installer and live progress

## Scope

The **lightweight Windows installer** copies the current ReActor-UI source and
creates a Start Menu shortcut (optional desktop shortcut). It runs using the
**already installed ReActor Python virtual environment**, and reuses existing
locally installed INSwapper and XSeg model weights instead of copying or
redownloading them.

**This is NOT a standalone/offline installer for a new PC.** It does not include
Python, Python packages, FFmpeg, InsightFace's buffalo_l weights, GFPGAN or
other third-party trained models. Existing computer dependencies are still
required; a completely fresh Windows installation needs those separately.
The installer itself requires no Internet after downloading its EXE.

The EXE is built by GitHub Actions using Inno Setup. On a successful run,
download the **ReActor-Studio-Windows-Installer** artifact and run
**ReActor-Studio-Setup.exe**.

### Where it installs

- App source: %LOCALAPPDATA%\Programs\ReActor Studio
- Persistent settings: %APPDATA%\ReActor-Studio\settings.json
- GUI error log: %LOCALAPPDATA%\ReActor-Studio\logs\reactor.log
- ONNX models: existing folder chosen in **Settings** (typically
  C:\Users\Casa\ReActor-Custom\models or a second ReActor-UI installation)
- Python interpreter: existing user environment, typically
  C:\Users\Casa\ReActor-UI\.venv\Scripts\pythonw.exe

The launcher automatically checks those common locations. When a virtual
environment is elsewhere, set the Windows environment variable
**REACTOR_PYTHON** to the existing Python executable (prefer pythonw.exe).
It also verifies FFmpeg is on PATH before launching.

The installation and future upgrades **do not delete user settings or models**.
The original Git checkout is not modified by installing the app.

## Settings (GUI)

Open **Settings** from the main application header. You can configure:

- Execution provider (only options detected by installed ONNX Runtime)
- Execution threads (default **2**)
- Maximum memory (default **6 GB**, applied on the next launch)
- Video quality CRF (default **10**)
- Video encoder (default **libx264**)
- Model weights folder, which can point at the existing downloaded models
- Save the options currently configured in the main window: preserve FPS,
  audio, extracted frames, PNG FPS, enhancer, XSeg and landmark mode.

Click **Save defaults** to persist the choices across sessions. The settings
file does **not** contain private face photos, frame directories, videos,
embeddings, or NSFW state. User-specific identity assignments remain in RAM.

CLI switches still override saved values. For example,
`python run.py --execution-provider cpu --execution-threads 2 --max-memory 6 --video-quality 10 --keep-fps`
uses the supplied parameters.

## Live processing progress

Main-window footer includes a progress bar, completed and total frames,
percentage, frame rate (or seconds per frame), and estimated remaining time.

- Video processing: progress is shown independently for Face Swap and
  Face Enhancer, while the original terminal `tqdm` progress stays intact.
- Existing numbered PNGs: includes reused completed frames; FFmpeg MP4
  encoding progress is shown separately.
- While models are prepared, NSFW checks run, or FFmpeg extracts the input,
  the bar indicates indeterminate activity and the status line indicates
  the current operation. The ETA applies to the current frame-processing
  stage only; it does not predict all extraction and encoding time.
- The heavy operations execute in a background Python thread. All Tk
  controls and dialogs are updated exclusively through a synchronized
  queue processed on Tk's main thread.
- Start, Preview, Settings and media controls are disabled while processing
  so a job cannot be started twice or altered halfway through.
- The Close action is blocked while processing to avoid losing or corrupting
  work; it is re-enabled after the worker ends.

No new remote API is used. This UI change does not change the content-safety
classifier, the identity matching algorithms or original media.

## Local testing

Run the GitHub unit tests and compile the Windows installer workflow.
The installed GUI must also be manually tested on Windows:

1. Launch via the Start Menu, confirm the GUI opens without PowerShell.
2. Check Settings: 2 threads, 6 GB memory, CRF 10, CPU, existing models folder.
3. Start a short non-explicit video: confirm GUI keeps responding and progress
   updates while console-free.
4. Test PNG reuse/resume and separate MP4 encoding progress.
5. Confirm source images and original PNGs are never overwritten.
6. Change a setting, close/reopen and verify it persists.

**Build checks alone do not confirm runtime compatibility with locally
installed model packages.** Use the saved `reactor.log` to diagnose
a failed console-free launch.
