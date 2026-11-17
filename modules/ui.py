import os
import webbrowser
from pathlib import Path
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
ROOT_HEIGHT = 1030
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None
FRAMES_PREVIEW_PATHS = []
assignments_label = None
frames_fps_entry = None

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
    global assignments_label, frames_fps_entry

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.geometry(f'{ROOT_WIDTH}x{ROOT_HEIGHT}')
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text='Source face')
    source_label.place(relx=0.1, rely=0.065, relwidth=0.3, relheight=0.23)
    target_label = ctk.CTkLabel(root, text='Target video / frames')
    target_label.place(relx=0.6, rely=0.065, relwidth=0.3, relheight=0.23)

    ctk.CTkButton(root, text='Select a face', cursor='hand2',
                  command=select_source_path).place(
        relx=0.1, rely=0.31, relwidth=0.3, relheight=0.075
    )
    ctk.CTkButton(root, text='Select a target', cursor='hand2',
                  command=select_target_path).place(
        relx=0.6, rely=0.31, relwidth=0.3, relheight=0.075
    )

    ctk.CTkButton(root, text='Use existing PNG frames',
                  command=select_frames_directory).place(
        relx=0.1, rely=0.40, relwidth=0.35, relheight=0.05
    )
    ctk.CTkButton(root, text='Choose person (single)',
                  command=open_face_picker).place(
        relx=0.55, rely=0.40, relwidth=0.35, relheight=0.05
    )
    ctk.CTkButton(root, text='Clear person assignments',
                  command=clear_assignments).place(
        relx=0.1, rely=0.465, relwidth=0.35, relheight=0.05
    )
    ctk.CTkButton(root, text='Add person + source photo',
                  command=lambda: open_face_picker(multiple=True)).place(
        relx=0.55, rely=0.465, relwidth=0.35, relheight=0.05
    )

    assignments_label = ctk.CTkLabel(root, text='Mapped people: 0 (legacy single-face mode)')
    assignments_label.place(relx=0.1, rely=0.525, relwidth=0.8)

    ctk.CTkLabel(root, text='Existing frames FPS:').place(relx=0.1, rely=0.56)
    frames_fps_entry = ctk.CTkEntry(root, width=80)
    frames_fps_entry.insert(0, str(modules.globals.frames_fps))
    frames_fps_entry.place(relx=0.65, rely=0.555)

    keep_fps_value = ctk.BooleanVar(value=bool(modules.globals.keep_fps))
    ctk.CTkSwitch(
        root, text='Keep fps', variable=keep_fps_value, cursor='hand2',
        command=lambda: setattr(modules.globals, 'keep_fps', bool(keep_fps_value.get()))
    ).place(relx=0.1, rely=0.615)

    keep_audio_value = ctk.BooleanVar(value=bool(modules.globals.keep_audio))
    ctk.CTkSwitch(
        root, text='Keep audio', variable=keep_audio_value, cursor='hand2',
        command=lambda: setattr(modules.globals, 'keep_audio', bool(keep_audio_value.get()))
    ).place(relx=0.6, rely=0.615)

    keep_frames_value = ctk.BooleanVar(value=bool(modules.globals.keep_frames))
    ctk.CTkSwitch(
        root, text='Keep frames', variable=keep_frames_value, cursor='hand2',
        command=lambda: setattr(modules.globals, 'keep_frames', bool(keep_frames_value.get()))
    ).place(relx=0.1, rely=0.655)

    many_faces_value = ctk.BooleanVar(value=bool(modules.globals.many_faces))
    many_faces_switch = ctk.CTkSwitch(
        root, text='Many faces', variable=many_faces_value, cursor='hand2',
        command=lambda: set_many_faces(bool(many_faces_value.get()))
    )
    many_faces_switch.place(relx=0.6, rely=0.655)

    enhancer_value = ctk.BooleanVar(value=bool(modules.globals.fp_ui.get('face_enhancer', False)))
    ctk.CTkSwitch(
        root, text='Face Enhancer', variable=enhancer_value, cursor='hand2',
        command=lambda: update_tumbler('face_enhancer', bool(enhancer_value.get()))
    ).place(relx=0.1, rely=0.695)

    nsfw_value = ctk.BooleanVar(value=bool(modules.globals.nsfw))
    ctk.CTkSwitch(
        root, text='NSFW', variable=nsfw_value, cursor='hand2',
        command=lambda: setattr(modules.globals, 'nsfw', bool(nsfw_value.get()))
    ).place(relx=0.6, rely=0.695)

    protection_value = ctk.BooleanVar(
        value=bool(modules.globals.occlusion_protection)
    )
    ctk.CTkSwitch(
        root, text='Occlusion protection (XSeg)',
        variable=protection_value, cursor='hand2',
        command=lambda: setattr(
            modules.globals, 'occlusion_protection', bool(protection_value.get())
        )
    ).place(relx=0.1, rely=0.738)
    ctk.CTkLabel(root, text='Landmarks:').place(
        relx=0.55, rely=0.738, relwidth=0.14
    )
    landmark_mode_menu = ctk.CTkOptionMenu(
        root, values=['Auto', '3-point', '5-point'],
        command=lambda value: setattr(
            modules.globals, 'occlusion_landmark_mode', value.lower()
        )
    )
    landmark_mode_menu.set(
        'Auto' if modules.globals.occlusion_landmark_mode == 'auto'
        else modules.globals.occlusion_landmark_mode
    )
    landmark_mode_menu.place(relx=0.70, rely=0.738, relwidth=0.20)

    ctk.CTkLabel(root, text='Enhancer method:').place(relx=0.1, rely=0.776)
    mode_menu = ctk.CTkOptionMenu(
        root, values=['GFPGAN', 'Sharpen only'],
        command=lambda value: setattr(modules.globals, 'enhancer_mode', value)
    )
    mode_menu.set(modules.globals.enhancer_mode)
    mode_menu.place(relx=0.54, rely=0.776, relwidth=0.36)

    strength_label = ctk.CTkLabel(root, text=f'Strength: {modules.globals.enhancer_strength}%')
    strength_label.place(relx=0.1, rely=0.820)
    strength_slider = ctk.CTkSlider(
        root, from_=0, to=100, number_of_steps=20,
        command=lambda value: set_enhancer_strength(value, strength_label)
    )
    strength_slider.set(modules.globals.enhancer_strength)
    strength_slider.place(relx=0.45, rely=0.820, relwidth=0.45)

    ctk.CTkButton(
        root, text='Compare enhancer (1 frame)', command=compare_enhancer
    ).place(relx=0.28, rely=0.858, relwidth=0.44, relheight=0.039)

    ctk.CTkButton(
        root, text='Start', cursor='hand2', command=lambda: select_output_path(start)
    ).place(relx=0.15, rely=0.912, relwidth=0.2, relheight=0.042)
    ctk.CTkButton(
        root, text='Destroy', cursor='hand2', command=destroy
    ).place(relx=0.4, rely=0.912, relwidth=0.2, relheight=0.042)
    ctk.CTkButton(
        root, text='Preview', cursor='hand2', command=toggle_preview
    ).place(relx=0.65, rely=0.912, relwidth=0.2, relheight=0.042)

    status_label = ctk.CTkLabel(root, text='', justify='center')
    status_label.place(relx=0.1, rely=0.952, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Gourieff GitHub', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.983, relwidth=0.8)
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


def set_enhancer_strength(value: float, label: ctk.CTkLabel) -> None:
    percentage = max(0, min(100, int(round(value))))
    modules.globals.enhancer_strength = percentage
    label.configure(text=f'Strength: {percentage}%')


def compare_enhancer() -> None:
    """Render the face swap and GFPGAN once, then compare blend levels locally."""
    if (not modules.globals.source_path and not modules.globals.face_assignments) or preview_target_count() == 0:
        update_status('Select a source face/mapping and a video or frames first.')
        return

    from modules.processors.frame import face_swapper, face_enhancer
    from modules.enhancement import blend_selected_faces

    frame_index = 0
    if preview_target_count() > 1 and PREVIEW.state() == 'normal':
        frame_index = max(0, int(round(preview_slider.get())))
    frame = preview_target_frame(frame_index)
    frame_number = frame_index + 1
    if frame is None:
        update_status('Cannot read comparison frame.')
        return

    comparison = ctk.CTkToplevel(ROOT)
    comparison.title(f'Enhancer comparison - frame {frame_number}')
    comparison.geometry('1030x650')
    ctk.CTkLabel(
        comparison,
        text='Processing one frame locally. GFPGAN can take a while on CPU.'
    ).pack(pady=6)
    comparison.update_idletasks()

    try:
        face_swapper_source = (
            face_swapper.prepare_multi_assignments()
            if modules.globals.face_assignments else
            face_swapper.load_source_face(modules.globals.source_path)
        )
        swapped = face_swapper.process_frame(face_swapper_source, frame)
        boxes = list(modules.globals.preview_face_boxes)
        if not boxes:
            ctk.CTkLabel(comparison, text='No selected face in this frame. Try another one.').pack()
            return
        candidate = face_enhancer.enhanced_candidate(swapped)
        grid = ctk.CTkFrame(comparison)
        grid.pack(fill='both', expand=True, padx=12, pady=12)
        for column, amount in enumerate((0, 40, 70)):
            sample = blend_selected_faces(swapped, candidate, boxes, amount)
            image = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
            image = ImageOps.contain(image, (315, 525), Image.LANCZOS)
            shown = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            cell = ctk.CTkFrame(grid)
            cell.grid(row=0, column=column, padx=7, sticky='n')
            ctk.CTkLabel(cell, text=f'{amount}% enhancement').pack(pady=4)
            preview = ctk.CTkLabel(cell, image=shown, text='')
            preview.image = shown
            preview.pack(pady=4)
        ctk.CTkLabel(
            comparison, text='Preview only. The video has not been modified.'
        ).pack(pady=5)
    except Exception as error:
        ctk.CTkLabel(comparison, text=f'Comparison failed: {error}', wraplength=850).pack(pady=20)


def update_assignments_label() -> None:
    if assignments_label is not None:
        count = len(modules.globals.face_assignments)
        assignments_label.configure(
            text=f'Mapped people: {count}' if count else
            'Mapped people: 0 (legacy single-face mode)'
        )


def clear_assignments() -> None:
    modules.globals.face_assignments.clear()
    update_assignments_label()
    update_status('All additional person assignments cleared.')


def select_frames_directory() -> None:
    """Use only local PNG files; do not require a corresponding video."""
    global FRAMES_PREVIEW_PATHS
    folder = ctk.filedialog.askdirectory(
        title='Choose an existing numbered PNG frames folder',
        initialdir=RECENT_DIRECTORY_TARGET
    )
    if not folder:
        return
    paths = sorted(
        (p for p in Path(folder).glob('*.png')
         if p.stem.isdigit() and int(p.stem) >= 1),
        key=lambda p: int(p.stem)
    )
    if not paths:
        update_status('This folder contains no numbered PNG frames.')
        return
    first = next((str(path) for path in paths if cv2.imread(str(path)) is not None), None)
    if first is None:
        update_status('No readable PNG found in selected folder.')
        return
    PREVIEW.withdraw()
    FRAMES_PREVIEW_PATHS = [str(p) for p in paths]
    modules.globals.frames_directory = folder
    modules.globals.target_path = None
    modules.globals.source_path = None
    source_label.configure(image=None, text='Source face')
    modules.globals.target_face_embedding = None
    modules.globals.face_assignments.clear()
    update_assignments_label()
    image = render_image_preview(first, (200, 200))
    target_label.configure(
        image=image, text=f'{Path(folder).name} — {len(paths)} PNGs',
        compound='top'
    )
    update_status(
        'Existing frames selected. Choose each person + source image, '
        'then Start. Output is silent MP4; input PNGs remain unchanged.'
    )


def preview_target_count() -> int:
    if modules.globals.frames_directory:
        return len(FRAMES_PREVIEW_PATHS)
    if is_video(modules.globals.target_path):
        return get_video_frame_total(modules.globals.target_path)
    return 1 if is_image(modules.globals.target_path) else 0


def preview_target_frame(index: int):
    if modules.globals.frames_directory:
        if not FRAMES_PREVIEW_PATHS:
            return None
        path = FRAMES_PREVIEW_PATHS[max(0, min(index, len(FRAMES_PREVIEW_PATHS) - 1))]
        return cv2.imread(path)
    if is_video(modules.globals.target_path):
        return get_video_frame(modules.globals.target_path, max(1, index + 1))
    if is_image(modules.globals.target_path):
        return cv2.imread(modules.globals.target_path)
    return None


def set_many_faces(enabled: bool) -> None:
    modules.globals.many_faces = enabled
    if enabled:
        modules.globals.target_face_embedding = None
        modules.globals.face_assignments.clear()
        update_assignments_label()
        update_status('Many faces enabled: individual mappings cleared.')


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
    global RECENT_DIRECTORY_TARGET, FRAMES_PREVIEW_PATHS, img_ft, vid_ft

    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
    if not target_path:
        return
    PREVIEW.withdraw()
    # Selections for an earlier target must never carry over.
    modules.globals.target_face_embedding = None
    modules.globals.frames_directory = None
    FRAMES_PREVIEW_PATHS = []
    modules.globals.face_assignments.clear()
    update_assignments_label()
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


def open_face_picker(multiple: bool = False) -> None:
    """Select a face on an original video frame and remember its identity.

    All detection happens on the local machine. The selected embedding remains
    in RAM, and frames that cannot be matched confidently are left untouched.
    """
    target_path = modules.globals.target_path
    if not modules.globals.frames_directory and not (is_image(target_path) or is_video(target_path)):
        update_status('Select a target video, image, or PNG frames folder first.')
        return

    picker = ctk.CTkToplevel(ROOT)
    picker.title('Add person + reference photo' if multiple else 'Choose the person to replace')
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
    total_frames = preview_target_count()
    max_frame_index = max(0, total_frames - 1)
    frame_slider = ctk.CTkSlider(
        picker, from_=0, to=max(1, max_frame_index),
        number_of_steps=max(1, max_frame_index)
    )
    frame_slider.set(min(max_frame_index, 30))
    if total_frames > 1:
        frame_slider.pack(fill='x', padx=30, pady=(5, 3))

    def load_frame() -> None:
        frame_index = int(round(frame_slider.get())) if total_frames > 1 else 0
        feedback.configure(text=f'Detecting faces in frame {frame_index + 1}...')
        picker.update_idletasks()
        frame = preview_target_frame(frame_index)
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
                if multiple:
                    photo = ctk.filedialog.askopenfilename(
                        title='Select the source photo for this person',
                        initialdir=RECENT_DIRECTORY_SOURCE,
                        filetypes=[img_ft]
                    )
                    if not is_image(photo):
                        feedback.configure(text='Choose a valid source photo to add the person.')
                        return
                    for current in modules.globals.face_assignments:
                        score = float(embedding @ current['embedding'])
                        if score > 0.92:
                            feedback.configure(
                                text='This person is already mapped. Choose another person '
                                     'or clear existing assignments.'
                            )
                            return
                    modules.globals.face_assignments.append({
                        'embedding': embedding.copy(),
                        'source_path': photo
                    })
                    modules.globals.target_face_embedding = None
                    modules.globals.many_faces = False
                    many_faces_switch.deselect()
                    update_assignments_label()
                    update_status(
                        f'{len(modules.globals.face_assignments)} person(s) mapped to source photos.'
                    )
                else:
                    modules.globals.face_assignments.clear()
                    update_assignments_label()
                    modules.globals.target_face_embedding = embedding.copy()
                    modules.globals.many_faces = False
                    many_faces_switch.deselect()
                    update_status('Single person selected: only that identity will be swapped.')
                picker.destroy()
                return
        feedback.configure(text='Click inside one of the highlighted face rectangles.')

    def clear_selection() -> None:
        if not multiple:
            modules.globals.target_face_embedding = None
            update_status('Person selection cleared: original leftmost-face mode.')
        else:
            update_status('Cancelled; existing assignments were preserved.')
        picker.destroy()

    preview.bind('<Button-1>', choose_face)
    actions = ctk.CTkFrame(picker)
    actions.pack(pady=(5, 12))
    ctk.CTkButton(actions, text='Load frame', command=load_frame).pack(side='left', padx=6, pady=5)
    ctk.CTkButton(actions, text='Clear selection' if not multiple else 'Close',
                  command=clear_selection).pack(side='left', padx=6, pady=5)
    ctk.CTkButton(actions, text='Cancel', command=picker.destroy).pack(side='left', padx=6, pady=5)

    load_frame()


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if modules.globals.frames_directory:
        try:
            value = float(frames_fps_entry.get().replace(',', '.'))
            if not 1 <= value <= 120:
                raise ValueError()
            modules.globals.frames_fps = value
        except (TypeError, ValueError):
            update_status('Existing frames FPS must be between 1 and 120.')
            return
        output_path = ctk.filedialog.asksaveasfilename(
            title='Output MP4 (reusable for resume)',
            filetypes=[('MP4 video', '*.mp4')], defaultextension='.mp4',
            initialfile=Path(modules.globals.frames_directory).name + '_edited.mp4',
            initialdir=RECENT_DIRECTORY_OUTPUT
        )
    elif is_image(modules.globals.target_path):
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
    elif (modules.globals.source_path or modules.globals.face_assignments) and preview_target_count():
        init_preview()
        update_preview()
        PREVIEW.deiconify()
    else:
        update_status('Select a target and a source face or person assignment.')


def init_preview() -> None:
    count = preview_target_count()
    if count > 1:
        preview_slider.configure(from_=0, to=count - 1,
                                 number_of_steps=max(1, count - 1))
        preview_slider.pack(fill='x')
        preview_slider.set(0)
    else:
        preview_slider.pack_forget()


def update_preview(frame_number: int = 0) -> None:
    if not ((modules.globals.source_path or modules.globals.face_assignments)
            and preview_target_count()):
        return
    temp_frame = preview_target_frame(max(0, int(round(frame_number))))
    if temp_frame is None:
        update_status('Cannot read selected preview frame.')
        return
    if modules.globals.nsfw is False:
        from modules.predicter import predict_frame
        if predict_frame(temp_frame):
            update_status('Preview is not available for this frame.')
            return
    from modules.processors.frame import face_swapper
    source = (
        face_swapper.prepare_multi_assignments()
        if modules.globals.face_assignments else
        face_swapper.load_source_face(modules.globals.source_path)
    )
    for processor in get_frame_processors_modules(modules.globals.frame_processors):
        temp_frame = processor.process_frame(source, temp_frame)
    image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
    displayed = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
    preview_label.configure(image=displayed)
    preview_label.image = displayed
