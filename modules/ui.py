import os
import tkinter as tk
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
ROOT_HEIGHT = 820
ROOT_WIDTH = 980

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

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def _section(parent, title: str, hint: str = '') -> ctk.CTkFrame:
    """Consistent resizable card for the main application settings."""
    card = ctk.CTkFrame(
        parent, fg_color=('#FFFFFF', '#1F2937'),
        corner_radius=15, border_width=1,
        border_color=('#DFE8E5', '#33443F')
    )
    card.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        card, text=title, anchor='w',
        font=ctk.CTkFont(size=18, weight='bold')
    ).grid(row=0, column=0, padx=20, pady=(17, 0), sticky='ew')
    if hint:
        ctk.CTkLabel(
            card, text=hint, anchor='w',
            text_color=('gray42', 'gray68'), wraplength=780
        ).grid(row=1, column=0, padx=20, pady=(0, 12), sticky='ew')
    return card


def _bring_dialog_to_front(dialog: ctk.CTkToplevel) -> None:
    """Keep child dialogs above their owner, without a permanent topmost flag."""
    dialog.transient(ROOT)
    dialog.lift()
    dialog.attributes('-topmost', True)

    def finish_raise() -> None:
        if dialog.winfo_exists():
            dialog.lift()
            dialog.focus_force()
            dialog.attributes('-topmost', False)

    dialog.after(250, finish_raise)


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label
    global assignments_label, frames_fps_entry

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk(fg_color=('#F1F5F4', '#101A18'))
    root.geometry(f'{ROOT_WIDTH}x{ROOT_HEIGHT}')
    root.minsize(760, 620)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    header = ctk.CTkFrame(
        root, fg_color=('#E4F5EC', '#19362A'), corner_radius=0
    )
    header.grid(row=0, column=0, sticky='ew')
    header.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        header, text='ReActor  /  Studio', anchor='w',
        font=ctk.CTkFont(size=26, weight='bold')
    ).grid(row=0, column=0, padx=26, pady=(16, 0), sticky='ew')
    ctk.CTkLabel(
        header,
        text='Local face replacement  •  Map people, choose your input and refine the output',
        anchor='w', text_color=('gray38', 'gray70')
    ).grid(row=1, column=0, padx=26, pady=(0, 15), sticky='ew')

    content = ctk.CTkScrollableFrame(
        root, fg_color='transparent', corner_radius=0
    )
    content.grid(row=1, column=0, padx=14, pady=(12, 6), sticky='nsew')
    content.grid_columnconfigure(0, weight=1)

    media = _section(
        content, '01  Media',
        'Choose a target image, video, or existing PNG sequence.'
    )
    media.grid(row=0, column=0, padx=8, pady=(0, 12), sticky='ew')
    media.grid_columnconfigure(0, weight=1)
    media.grid_columnconfigure(1, weight=1)

    source_card = ctk.CTkFrame(
        media, fg_color=('#F4F8F6', '#25342F'), corner_radius=12
    )
    source_card.grid(row=2, column=0, padx=(18, 8), pady=(0, 18), sticky='nsew')
    source_card.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        source_card, text='SOURCE PHOTO', anchor='w',
        font=ctk.CTkFont(size=12, weight='bold'),
        text_color=('gray40', 'gray69')
    ).grid(row=0, column=0, padx=15, pady=(12, 3), sticky='ew')
    source_label = ctk.CTkLabel(
        source_card, text='Add a person below to select a source photo',
        height=160, wraplength=280
    )
    source_label.grid(row=1, column=0, padx=15, pady=(0, 12), sticky='nsew')

    target_card = ctk.CTkFrame(
        media, fg_color=('#F4F8F6', '#25342F'), corner_radius=12
    )
    target_card.grid(row=2, column=1, padx=(8, 18), pady=(0, 18), sticky='nsew')
    target_card.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        target_card, text='TARGET MEDIA', anchor='w',
        font=ctk.CTkFont(size=12, weight='bold'),
        text_color=('gray40', 'gray69')
    ).grid(row=0, column=0, padx=15, pady=(12, 3), sticky='ew')
    target_label = ctk.CTkLabel(
        target_card, text='No target selected', height=160, wraplength=280
    )
    target_label.grid(row=1, column=0, padx=15, pady=(0, 12), sticky='nsew')
    ctk.CTkButton(
        target_card, text='Select target video / image',
        command=select_target_path, height=36
    ).grid(row=2, column=0, padx=15, pady=(0, 7), sticky='ew')
    ctk.CTkButton(
        target_card, text='Use existing PNG frames',
        command=select_frames_directory, height=36,
        fg_color=('#D8EDE2', '#365044'),
        text_color=('#174D32', '#F3FFF8'),
        hover_color=('#C8E0D1', '#446254')
    ).grid(row=3, column=0, padx=15, pady=(0, 15), sticky='ew')

    people = _section(
        content, '02  People & source photos',
        'Choose a person in the target frame and associate a source photograph. Repeat for multiple people.'
    )
    people.grid(row=1, column=0, padx=8, pady=(0, 12), sticky='ew')
    people.grid_columnconfigure(0, weight=1)
    people.grid_columnconfigure(1, weight=1)
    ctk.CTkButton(
        people, text='+  Add person + source photo', height=44,
        command=lambda: open_face_picker(multiple=True)
    ).grid(row=2, column=0, padx=(20, 8), pady=(0, 12), sticky='ew')
    ctk.CTkButton(
        people, text='Clear person assignments', height=44,
        fg_color=('#E5ECE9', '#34473E'),
        text_color=('#294E3E', '#E6F6EE'),
        hover_color=('#D7E3DC', '#3C5748'),
        command=clear_assignments
    ).grid(row=2, column=1, padx=(8, 20), pady=(0, 12), sticky='ew')
    assignments_label = ctk.CTkLabel(
        people, text='', anchor='w', text_color=('gray38', 'gray72'),
        wraplength=780
    )
    assignments_label.grid(
        row=3, column=0, columnspan=2, padx=22, pady=(0, 16), sticky='ew'
    )
    update_assignments_label()

    processing = _section(content, '03  Processing', 'Playback and output options.')
    processing.grid(row=2, column=0, padx=8, pady=(0, 12), sticky='ew')
    processing.grid_columnconfigure(0, weight=1)
    processing.grid_columnconfigure(1, weight=1)

    keep_fps_value = ctk.BooleanVar(value=bool(modules.globals.keep_fps))
    ctk.CTkSwitch(
        processing, text='Keep fps', variable=keep_fps_value,
        command=lambda: setattr(modules.globals, 'keep_fps', bool(keep_fps_value.get()))
    ).grid(row=2, column=0, padx=22, pady=(0, 13), sticky='w')

    keep_audio_value = ctk.BooleanVar(value=bool(modules.globals.keep_audio))
    ctk.CTkSwitch(
        processing, text='Keep audio', variable=keep_audio_value,
        command=lambda: setattr(modules.globals, 'keep_audio', bool(keep_audio_value.get()))
    ).grid(row=2, column=1, padx=22, pady=(0, 13), sticky='w')

    keep_frames_value = ctk.BooleanVar(value=bool(modules.globals.keep_frames))
    ctk.CTkSwitch(
        processing, text='Keep frames', variable=keep_frames_value,
        command=lambda: setattr(modules.globals, 'keep_frames', bool(keep_frames_value.get()))
    ).grid(row=3, column=0, padx=22, pady=(0, 16), sticky='w')

    nsfw_value = ctk.BooleanVar(value=bool(modules.globals.nsfw))
    ctk.CTkSwitch(
        processing, text='NSFW', variable=nsfw_value,
        command=lambda: setattr(modules.globals, 'nsfw', bool(nsfw_value.get()))
    ).grid(row=3, column=1, padx=22, pady=(0, 16), sticky='w')

    frames_row = ctk.CTkFrame(processing, fg_color='transparent')
    frames_row.grid(
        row=4, column=0, columnspan=2, padx=22, pady=(0, 14), sticky='w'
    )
    ctk.CTkLabel(
        frames_row, text='Existing PNG frames FPS:'
    ).pack(side='left', padx=(0, 12))
    frames_fps_entry = ctk.CTkEntry(frames_row, width=90)
    frames_fps_entry.insert(0, str(modules.globals.frames_fps))
    frames_fps_entry.pack(side='left')

    occlusion = _section(
        content, '04  Occlusion protection',
        'Keep foreground objects, such as microphones and hands, in front of swapped faces.'
    )
    occlusion.grid(row=3, column=0, padx=8, pady=(0, 12), sticky='ew')
    occlusion.grid_columnconfigure(0, weight=1)
    occlusion.grid_columnconfigure(1, weight=1)

    protection_value = ctk.BooleanVar(value=bool(modules.globals.occlusion_protection))
    ctk.CTkSwitch(
        occlusion, text='Occlusion protection (XSeg)', variable=protection_value,
        command=lambda: setattr(
            modules.globals, 'occlusion_protection', bool(protection_value.get())
        )
    ).grid(row=2, column=0, padx=22, pady=(0, 18), sticky='w')

    align_row = ctk.CTkFrame(occlusion, fg_color='transparent')
    align_row.grid(row=2, column=1, padx=22, pady=(0, 18), sticky='ew')
    ctk.CTkLabel(align_row, text='Landmarks:').pack(side='left', padx=(0, 12))
    landmark_mode_menu = ctk.CTkOptionMenu(
        align_row, values=['Auto', '2-point L', '2-point R', '3-point', '5-point'],
        command=lambda value: setattr(
            modules.globals, 'occlusion_landmark_mode',
            {'2-point L': '2-point-left', '2-point R': '2-point-right'}.get(
                value, value.lower()
            )
        )
    )
    landmark_mode_menu.set({
        'auto': 'Auto', '2-point-left': '2-point L', '2-point-right': '2-point R'
    }.get(modules.globals.occlusion_landmark_mode, modules.globals.occlusion_landmark_mode))
    landmark_mode_menu.pack(side='left', fill='x', expand=True)

    enhancer = _section(
        content, '05  Face Enhancer',
        'Optional detail restoration. Compare three strengths before rendering your video.'
    )
    enhancer.grid(row=4, column=0, padx=8, pady=(0, 14), sticky='ew')
    enhancer.grid_columnconfigure(0, weight=1)
    enhancer.grid_columnconfigure(1, weight=1)

    enhancer_value = ctk.BooleanVar(
        value=bool(modules.globals.fp_ui.get('face_enhancer', False))
    )
    ctk.CTkSwitch(
        enhancer, text='Face Enhancer', variable=enhancer_value,
        command=lambda: update_tumbler('face_enhancer', bool(enhancer_value.get()))
    ).grid(row=2, column=0, padx=22, pady=(0, 14), sticky='w')

    enhancer_mode_row = ctk.CTkFrame(enhancer, fg_color='transparent')
    enhancer_mode_row.grid(row=2, column=1, padx=22, pady=(0, 14), sticky='ew')
    ctk.CTkLabel(
        enhancer_mode_row, text='Method:'
    ).pack(side='left', padx=(0, 12))
    mode_menu = ctk.CTkOptionMenu(
        enhancer_mode_row, values=['GFPGAN', 'Sharpen only'],
        command=lambda value: setattr(modules.globals, 'enhancer_mode', value)
    )
    mode_menu.set(modules.globals.enhancer_mode)
    mode_menu.pack(side='left', fill='x', expand=True)

    strength_row = ctk.CTkFrame(enhancer, fg_color='transparent')
    strength_row.grid(
        row=3, column=0, columnspan=2, padx=22, pady=(0, 14), sticky='ew'
    )
    strength_row.grid_columnconfigure(1, weight=1)
    strength_label = ctk.CTkLabel(
        strength_row, text=f'Strength: {modules.globals.enhancer_strength}%',
        width=132, anchor='w'
    )
    strength_label.grid(row=0, column=0, padx=(0, 12), sticky='w')
    strength_slider = ctk.CTkSlider(
        strength_row, from_=0, to=100, number_of_steps=20,
        command=lambda value: set_enhancer_strength(value, strength_label)
    )
    strength_slider.set(modules.globals.enhancer_strength)
    strength_slider.grid(row=0, column=1, sticky='ew')
    ctk.CTkButton(
        enhancer, text='Compare enhancer  •  Full-resolution frames',
        height=38, command=compare_enhancer
    ).grid(row=4, column=0, columnspan=2, padx=22, pady=(0, 20), sticky='ew')

    footer = ctk.CTkFrame(
        root, fg_color=('#FFFFFF', '#1F2937'), corner_radius=0
    )
    footer.grid(row=2, column=0, sticky='ew')
    for column in range(3):
        footer.grid_columnconfigure(column, weight=1)
    ctk.CTkButton(
        footer, text='Start processing', height=42,
        command=lambda: select_output_path(start)
    ).grid(row=0, column=0, padx=(24, 7), pady=(14, 5), sticky='ew')
    ctk.CTkButton(
        footer, text='Preview', height=42, command=toggle_preview,
        fg_color=('#D8EDE2', '#365044'),
        text_color=('#174D32', '#F3FFF8'),
        hover_color=('#C8E0D1', '#446254')
    ).grid(row=0, column=1, padx=7, pady=(14, 5), sticky='ew')
    ctk.CTkButton(
        footer, text='Close', height=42, command=destroy,
        fg_color=('#E8EBE9', '#34413C'),
        text_color=('#345045', '#EBF6F0'),
        hover_color=('#D6DFDA', '#3D5349')
    ).grid(row=0, column=2, padx=(7, 24), pady=(14, 5), sticky='ew')
    status_label = ctk.CTkLabel(
        footer, text='Ready. Select a target and add a person.',
        anchor='w', justify='left', wraplength=830,
        text_color=('gray38', 'gray70')
    )
    status_label.grid(
        row=1, column=0, columnspan=3, padx=24, pady=(0, 8), sticky='ew'
    )
    github = ctk.CTkLabel(
        footer, text='Gourieff GitHub  ·  Original project',
        cursor='hand2', text_color=('#288A60', '#6ED5A0'),
        font=ctk.CTkFont(size=12)
    )
    github.grid(row=2, column=0, columnspan=3, padx=24, pady=(0, 9))
    github.bind('<Button-1>', lambda event: webbrowser.open('https://github.com/Gourieff'))
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
    """Compare three complete, unscaled frames in a vertically scrollable window.

    The native frame resolution is preserved. Large frames can be inspected
    using both horizontal and vertical scrollbars; original media is unchanged.
    """
    if (not modules.globals.source_path and not modules.globals.face_assignments) or preview_target_count() == 0:
        update_status('Select a person/source mapping and a video or frames first.')
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
    comparison.title(f'Full-resolution enhancer comparison · frame {frame_number}')
    window_width = max(640, min(1300, ROOT.winfo_screenwidth() - 100))
    window_height = max(450, min(850, ROOT.winfo_screenheight() - 100))
    comparison.geometry(f'{window_width}x{window_height}')
    comparison.minsize(620, 420)
    _bring_dialog_to_front(comparison)

    header = ctk.CTkFrame(comparison, fg_color='transparent')
    header.pack(fill='x', padx=20, pady=(14, 8))
    ctk.CTkLabel(
        header, text=f'Enhancer comparison · Frame {frame_number}',
        font=ctk.CTkFont(size=20, weight='bold'), anchor='w'
    ).pack(fill='x')
    instruction = ctk.CTkLabel(
        header, text='Processing one frame locally. This may take time on CPU.',
        text_color=('gray40', 'gray72'), anchor='w'
    )
    instruction.pack(fill='x')

    viewport = ctk.CTkFrame(comparison, fg_color='transparent')
    viewport.pack(fill='both', expand=True, padx=16, pady=6)
    viewport.grid_columnconfigure(0, weight=1)
    viewport.grid_rowconfigure(0, weight=1)

    # CTkScrollableFrame offers one scroll direction; a Canvas allows both
    # directions without shrinking or cropping the source-resolution images.
    canvas = tk.Canvas(
        viewport, bg='#F2F5F3' if ctk.get_appearance_mode() == 'Light' else '#1E2824',
        highlightthickness=0, bd=0
    )
    canvas.grid(row=0, column=0, sticky='nsew')
    vertical = ctk.CTkScrollbar(viewport, orientation='vertical', command=canvas.yview)
    vertical.grid(row=0, column=1, sticky='ns')
    horizontal = ctk.CTkScrollbar(viewport, orientation='horizontal', command=canvas.xview)
    horizontal.grid(row=1, column=0, sticky='ew')
    canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
    contents = ctk.CTkFrame(canvas, fg_color='transparent')
    canvas.create_window((0, 0), window=contents, anchor='nw')
    contents.bind(
        '<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all'))
    )

    ctk.CTkLabel(
        comparison,
        text='Actual frame resolution · Scroll to inspect details · Preview only; no video files changed.',
        text_color=('gray40', 'gray72')
    ).pack(fill='x', padx=20, pady=(0, 12))

    # Give Windows an opportunity to display this long-running operation's UI.
    comparison.update()
    try:
        face_swapper_source = (
            face_swapper.prepare_multi_assignments()
            if modules.globals.face_assignments else
            face_swapper.load_source_face(modules.globals.source_path)
        )
        swapped = face_swapper.process_frame(face_swapper_source, frame)
        boxes = list(modules.globals.preview_face_boxes)
        if not boxes:
            instruction.configure(text='No selected face in this frame. Try another one.')
            return
        candidate = face_enhancer.enhanced_candidate(swapped)
        for amount in (0, 40, 70):
            sample = blend_selected_faces(swapped, candidate, boxes, amount)
            image = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
            # IMPORTANT: no ImageOps.contain / resize here. One sample per row,
            # and the CTkImage retains the source frame's original dimensions.
            shown = ctk.CTkImage(
                light_image=image, dark_image=image, size=image.size
            )
            card = ctk.CTkFrame(
                contents, fg_color=('#FFFFFF', '#25372E'), corner_radius=12
            )
            card.pack(anchor='nw', padx=12, pady=(10, 4))
            ctk.CTkLabel(
                card, text=f'{amount}% enhancement  ·  {image.width} × {image.height}',
                font=ctk.CTkFont(size=15, weight='bold'), anchor='w'
            ).pack(fill='x', padx=14, pady=(12, 6))
            preview = ctk.CTkLabel(card, image=shown, text='')
            preview.image = shown  # Preserve the complete image object in memory.
            preview.pack(padx=14, pady=(0, 14))
        contents.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))
        instruction.configure(text='Ready. Compare the full-size results by scrolling down.')
    except Exception as error:
        instruction.configure(text=f'Comparison failed: {error}')



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
