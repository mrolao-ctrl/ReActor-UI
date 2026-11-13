import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps, ImageDraw

import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face, get_many_faces
from modules.face_selector import normalized_embedding
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
many_faces_switch = None

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, many_faces_switch

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

    source_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path())
    source_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path())
    target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    person_button = ctk.CTkButton(root, text='Choose person in video', cursor='hand2', command=open_face_picker)
    person_button.place(relx=0.6, rely=0.515, relwidth=0.3, relheight=0.065)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep fps', variable=keep_fps_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_fps', not modules.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.6)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep frames', variable=keep_frames_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.65)

    # for FRAME PROCESSOR ENHANCER tumbler:
    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui['face_enhancer'])
    enhancer_switch = ctk.CTkSwitch(root, text='Face Enhancer', variable=enhancer_value, cursor='hand2', command=lambda: update_tumbler('face_enhancer',enhancer_value.get()))
    enhancer_switch.place(relx=0.1, rely=0.7)

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(root, text='Keep audio', variable=keep_audio_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_audio', keep_audio_value.get()))
    keep_audio_switch.place(relx=0.6, rely=0.6)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, cursor='hand2', command=lambda: set_many_faces(many_faces_value.get()))
    many_faces_switch.place(relx=0.6, rely=0.65)

    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw)
    nsfw_switch = ctk.CTkSwitch(root, text='NSFW', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw', nsfw_value.get()))
    nsfw_switch.place(relx=0.6, rely=0.7)

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.15, rely=0.80, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(root, text='Destroy', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=0.4, rely=0.80, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=0.65, rely=0.80, relwidth=0.2, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Gourieff GitHub', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('URL').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://github.com/Gourieff'))

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value


def set_many_faces(enabled: bool) -> None:
    modules.globals.many_faces = enabled
    if enabled:
        modules.globals.target_face_embedding = None
        update_status('Many faces enabled: individual person selection cleared.')


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE, filetypes=[img_ft])
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    # An identity selected for a previous video must never carry over.
    modules.globals.target_face_embedding = None
    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def open_face_picker() -> None:
    """Select a face on an original video frame and remember its identity.

    All detection happens on the local machine. The selected embedding remains
    in RAM, and frames that cannot be matched confidently are left untouched.
    """
    target_path = modules.globals.target_path
    if not target_path or not (is_image(target_path) or is_video(target_path)):
        update_status('Select a target image or video first.')
        return

    picker = ctk.CTkToplevel(ROOT)
    picker.title('Choose the person to replace')
    picker.geometry('900x720')
    picker.minsize(650, 650)

    info = ctk.CTkLabel(
        picker,
        text='Choose a frame where the person is visible, then click the rectangle around their face.',
        wraplength=830
    )
    info.pack(padx=12, pady=(12, 6))

    preview = ctk.CTkLabel(picker, text='Loading frame...', width=850, height=510)
    preview.pack(padx=12, pady=6, expand=True)

    feedback = ctk.CTkLabel(
        picker,
        text='A confident identity match will be used in every frame; uncertain frames are skipped.',
        wraplength=830
    )
    feedback.pack(padx=12, pady=5)

    state = {'faces': [], 'boxes': [], 'display_size': (0, 0)}
    total_frames = get_video_frame_total(target_path) if is_video(target_path) else 1
    max_frame_index = max(0, total_frames - 1)
    frame_slider = ctk.CTkSlider(
        picker, from_=0, to=max(1, max_frame_index),
        number_of_steps=max(1, max_frame_index)
    )
    frame_slider.set(min(max_frame_index, 30))
    if is_video(target_path):
        frame_slider.pack(fill='x', padx=30, pady=(5, 3))

    def load_frame() -> None:
        frame_index = int(round(frame_slider.get())) if is_video(target_path) else 0
        feedback.configure(text=f'Detecting faces in frame {frame_index + 1}...')
        picker.update_idletasks()
        if is_video(target_path):
            # capturer.get_video_frame expects a one-based frame number.
            frame = get_video_frame(target_path, frame_index + 1)
        else:
            frame = cv2.imread(target_path)
        if frame is None:
            feedback.configure(text='Could not read this frame. Try another one.')
            return

        faces = get_many_faces(frame) or []
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (850, 510), Image.LANCZOS)
        scale_x = image.width / frame.shape[1]
        scale_y = image.height / frame.shape[0]
        draw = ImageDraw.Draw(image)
        boxes = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            box = (int(x1 * scale_x), int(y1 * scale_y),
                   int(x2 * scale_x), int(y2 * scale_y))
            draw.rectangle(box, outline='#00E5FF', width=4)
            draw.text((box[0] + 5, box[1] + 5), str(index + 1), fill='#00E5FF')
            boxes.append(box)

        shown = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        preview.configure(image=shown, text='', width=image.width, height=image.height)
        preview.image = shown  # Keep a strong reference to the Tk image.
        state['faces'] = faces
        state['boxes'] = boxes
        state['display_size'] = image.size
        feedback.configure(
            text=f'Frame {frame_index + 1}: {len(faces)} face(s) found. Click the face to replace.'
            if faces else f'Frame {frame_index + 1}: no faces found. Choose a different frame.'
        )

    def choose_face(event) -> None:
        # CTkLabel centres the image if its widget is larger than the bitmap.
        width, height = state['display_size']
        offset_x = max(0, (preview.winfo_width() - width) / 2)
        offset_y = max(0, (preview.winfo_height() - height) / 2)
        x, y = event.x - offset_x, event.y - offset_y
        for face, box in zip(state['faces'], state['boxes']):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                embedding = normalized_embedding(face)
                if embedding is None:
                    feedback.configure(text='No identity embedding found. Choose another frame.')
                    return
                modules.globals.target_face_embedding = embedding.copy()
                modules.globals.many_faces = False
                many_faces_switch.deselect()
                update_status('Person selected: only that identity will be face-swapped.')
                picker.destroy()
                return
        feedback.configure(text='Click inside one of the highlighted face rectangles.')

    def clear_selection() -> None:
        modules.globals.target_face_embedding = None
        update_status('Person selection cleared: original leftmost-face mode.')
        picker.destroy()

    preview.bind('<Button-1>', choose_face)
    actions = ctk.CTkFrame(picker)
    actions.pack(pady=(5, 12))
    ctk.CTkButton(actions, text='Load frame', command=load_frame).pack(side='left', padx=6, pady=5)
    ctk.CTkButton(actions, text='Clear selection', command=clear_selection).pack(side='left', padx=6, pady=5)
    ctk.CTkButton(actions, text='Cancel', command=picker.destroy).pack(side='left', padx=6, pady=5)

    load_frame()


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', filetypes=[img_ft], defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', filetypes=[vid_ft], defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw == False:
            from modules.predicter import predict_frame
            if predict_frame(temp_frame):
                quit()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)),
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
