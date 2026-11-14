import os
from typing import List, Dict

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, 'workflow')

file_types = [
    ('Image', ('*.png','*.jpg','*.jpeg','*.gif','*.bmp')),
    ('Video', ('*.mp4','*.mkv'))
]

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = None
keep_audio = None
keep_frames = None
many_faces = None
# Unit-length reference embedding for the target person selected in the UI.
# None means the original leftmost-face selection remains in effect.
target_face_embedding = None

# Strength 0..100 blends GFPGAN with the unenhanced face-swap output.
enhancer_strength = 40
enhancer_mode = 'GFPGAN'  # or 'Sharpen only'
# Face boxes are recorded before swapping, because swapping changes identity.
# Keys are absolute paths of extracted PNG frames.
frame_target_boxes = {}
preview_face_boxes = []
frame_recovery_log = []
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
fp_ui: Dict[str, bool] = {}
nsfw = None
