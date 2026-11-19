import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import messagebox
import webbrowser
from pathlib import Path
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageTk

import modules.globals
import modules.metadata
from modules import progress_events
from modules.app_settings import load_settings, save_settings, validate, recommended_models_directory
from modules.face_analyser import get_one_face, get_many_faces
from modules.face_selector import normalized_embedding
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 820
ROOT_WIDTH = 1040

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
progress_bar = None
progress_label = None
start_button = None
settings_button = None
preview_button = None
_UI_EVENTS = queue.Queue()
_processing_active = False

# GuideMe design tokens, based on the current Android/web dark theme.
# Keep these independent of processing, identity detection, and model settings.
BG = '#050816'
SURFACE = '#0E162D'
CARD = '#131D38'
CARD_RAISED = '#1A2544'
BORDER = '#29375B'
PURPLE = '#6D5CFF'
PURPLE_HOVER = '#8577FF'
BLUE = '#4F8CFF'
TEXT = '#F5F7FF'
MUTED = '#AAB3C5'
SUBTLE = '#7786AB'
SECONDARY_BG = '#222E4F'
SECONDARY_HOVER = '#2E3D62'

img_ft, vid_ft = modules.globals.file_types

WORKFLOW_TITLES = (
    '01   SELECT MEDIA',
    '02   MAP FACES',
    '03   PROCESSING',
    '04   OCCLUSION',
    '05   FACE ENHANCER',
)


def _display_workflow_step(index, cards, tabs, previous_button, next_button,
                           stage_label, scrollable) -> bool:
    """Show one existing card at a time without destroying its state."""
    if not 0 <= index < len(cards):
        return False

    for position, card in enumerate(cards):
        if position == index:
            card.grid()
        else:
            card.grid_remove()
    for position, tab in enumerate(tabs):
        active = position == index
        tab.configure(
            fg_color='#1C2750' if active else '#121B36',
            hover_color='#283963' if active else SECONDARY_HOVER,
            border_color=PURPLE if active else BORDER,
            text_color=TEXT if active else MUTED,
        )
    previous_button.configure(state='disabled' if index == 0 else 'normal')
    next_button.configure(state='disabled' if index == len(cards) - 1 else 'normal')
    stage_label.configure(
        text=f'STEP {index + 1} OF {len(cards)}  ·  '
             f'{WORKFLOW_TITLES[index].split("   ", 1)[1]}'
    )
    # Reset scroll to the top for every page change.
    canvas = getattr(scrollable, '_parent_canvas', None)
    if canvas is not None:
        canvas.yview_moveto(0.0)
    return True


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)
    progress_events.set_sink(_UI_EVENTS.put)
    ROOT.after(100, _drain_events)

    return ROOT


def _set_app_icon(window) -> None:
    """Use the ReActor icon for the actual Windows taskbar/window, not Python.

    CustomTkinter 5.1.3 sets its own default ICO around 200 ms after window
    creation unless `iconbitmap()` was called. A plain Tk `iconphoto()`
    therefore gets overwritten on Windows. Prefer a real ICO here.
    """
    assets = Path(__file__).resolve().parents[1] / 'assets'
    png = assets / 'reactor-studio.png'
    ico = assets / 'reactor-studio.ico'
    if os.name == 'nt':
        try:
            if not ico.is_file() and png.is_file():
                # Source checkouts have the PNG but the Windows installer
                # generates the .ico during its build. Cache a local ICO for
                # source launches; do not write into the installed program.
                cache = (Path(os.getenv('LOCALAPPDATA', str(Path.home()))) /
                         'ReActor-Studio' / 'cache' / 'reactor-studio.ico')
                if not cache.is_file() or cache.stat().st_mtime < png.stat().st_mtime:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(png) as original:
                        original.convert('RGBA').save(
                            cache, format='ICO',
                            sizes=[(16, 16), (24, 24), (32, 32),
                                   (48, 48), (64, 64), (128, 128), (256, 256)]
                        )
                ico = cache
            if ico.is_file():
                window.iconbitmap(str(ico))
                # Explicit iconbitmap prevents CustomTkinter from replacing
                # the window icon with its generic icon on a delayed callback.
                return
        except (OSError, RuntimeError, ValueError, tk.TclError) as error:
            print(f'[REACTOR.UI] Windows ICO could not be applied: {error}')
    if not png.is_file():
        return
    try:
        with Image.open(png) as opened:
            picture = opened.convert('RGBA')
        image = ImageTk.PhotoImage(picture, master=window)
        window.iconphoto(True, image)
        window._reactor_icon_ref = image
    except (OSError, RuntimeError, tk.TclError) as error:
        print(f'[REACTOR.UI] Could not load optional application icon: {error}')



def _section(parent, title: str, hint: str = '') -> ctk.CTkFrame:
    """Consistent resizable card for the main application settings."""
    card = ctk.CTkFrame(
        parent, fg_color=SURFACE,
        corner_radius=20, border_width=1,
        border_color=BORDER
    )
    card.grid_columnconfigure(0, weight=1)
    card.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(
        card, text=title, anchor='w',
        font=ctk.CTkFont(size=18, weight='bold')
    ).grid(row=0, column=0, columnspan=2, padx=20, pady=(17, 0), sticky='ew')
    if hint:
        ctk.CTkLabel(
            card, text=hint, anchor='w',
            text_color=MUTED, wraplength=820
        ).grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 12), sticky='ew')
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
    global assignments_label, frames_fps_entry, start_button, settings_button
    global progress_bar, progress_label, preview_button

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('dark')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk(fg_color=BG)
    _set_app_icon(root)
    root.geometry(f'{ROOT_WIDTH}x{ROOT_HEIGHT}')
    root.minsize(800, 640)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.protocol('WM_DELETE_WINDOW', lambda: request_close(destroy))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    header = ctk.CTkFrame(
        root, fg_color='#0B1430', corner_radius=0, border_width=0
    )
    header.grid(row=0, column=0, sticky='ew')
    header.grid_columnconfigure(1, weight=1)

    logo_path = Path(__file__).resolve().parents[1] / 'assets' / 'reactor-studio.png'
    if logo_path.is_file():
        try:
            with Image.open(logo_path) as opened:
                logo_pil = opened.convert('RGB')
            logo = ctk.CTkImage(light_image=logo_pil, dark_image=logo_pil,
                                size=(56, 56))
            logo_label = ctk.CTkLabel(header, image=logo, text='', width=60)
            logo_label.image = logo
        except OSError:
            logo_label = ctk.CTkLabel(
                header, text='R', width=60,
                font=ctk.CTkFont(size=30, weight='bold'),
                text_color=BLUE
            )
    else:
        logo_label = ctk.CTkLabel(
            header, text='R', width=60,
            font=ctk.CTkFont(size=30, weight='bold'), text_color=BLUE
        )
    logo_label.grid(row=0, column=0, rowspan=3, padx=(26, 16), pady=18)

    ctk.CTkLabel(
        header, text='REACTOR', anchor='w', text_color=TEXT,
        font=ctk.CTkFont(size=25, weight='bold')
    ).grid(row=0, column=1, padx=0, pady=(14, 0), sticky='ew')
    ctk.CTkLabel(
        header, text='FACE STUDIO   /   LOCAL WORKSPACE', anchor='w',
        text_color=BLUE, font=ctk.CTkFont(size=12, weight='bold')
    ).grid(row=1, column=1, padx=0, pady=(0, 3), sticky='ew')
    ctk.CTkLabel(
        header, text='Map identities  ·  Refine details  ·  Export locally',
        anchor='w', text_color=MUTED, font=ctk.CTkFont(size=12)
    ).grid(row=2, column=1, padx=0, pady=(0, 12), sticky='ew')
    settings_button = ctk.CTkButton(
        header, text='⚙  Settings', width=146, height=40,
        fg_color=SECONDARY_BG, hover_color=SECONDARY_HOVER,
        text_color=TEXT, border_width=1, border_color=BORDER,
        command=open_settings
    )
    settings_button.grid(row=0, column=2, rowspan=3,
                         padx=(12, 24), pady=18, sticky='e')


    # Clickable GuideMe-style workflow navigation.
    steps = ctk.CTkFrame(header, fg_color='transparent', corner_radius=0)
    steps.grid(row=3, column=0, columnspan=3, sticky='ew',
               padx=24, pady=(0, 14))
    workflow_tabs = []
    active_step = 0
    for column, step in enumerate(WORKFLOW_TITLES):
        steps.grid_columnconfigure(column, weight=1, uniform='workflow')
        chip = ctk.CTkButton(
            steps, text=step, height=46,
            fg_color='#121B36', hover_color=SECONDARY_HOVER,
            text_color=MUTED, corner_radius=11,
            border_width=1, border_color=BORDER,
            font=ctk.CTkFont(size=11, weight='bold'),
            command=lambda index=column: show_step(index)
        )
        chip.grid(row=0, column=column,
                  padx=(0, 7) if column < len(WORKFLOW_TITLES) - 1 else 0,
                  sticky='ew')
        workflow_tabs.append(chip)

    content = ctk.CTkScrollableFrame(
        root, fg_color='transparent', corner_radius=0
    )
    content.grid(row=1, column=0, padx=18, pady=(16, 8), sticky='nsew')
    content.grid_columnconfigure(0, weight=1)

    media = _section(
        content, '01   MEDIA LIBRARY',
        'Choose a target image, video, or existing PNG sequence.'
    )
    media.grid(row=0, column=0, padx=8, pady=(0, 12), sticky='ew')
    media.grid_columnconfigure(0, weight=1)
    media.grid_columnconfigure(1, weight=1)

    target_card = ctk.CTkFrame(
        media, fg_color=CARD, corner_radius=12
    )
    target_card.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 18), sticky='nsew')
    target_card.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        target_card, text='TARGET MEDIA', anchor='w',
        font=ctk.CTkFont(size=12, weight='bold'),
        text_color=SUBTLE
    ).grid(row=0, column=0, padx=15, pady=(12, 3), sticky='ew')
    target_label = ctk.CTkLabel(
        target_card, text='No target selected', height=210, wraplength=720
    )
    target_label.grid(row=1, column=0, padx=15, pady=(0, 12), sticky='nsew')
    ctk.CTkButton(
        target_card, text='Select target video / image',
        command=select_target_path, height=36
    ).grid(row=2, column=0, padx=15, pady=(0, 7), sticky='ew')
    ctk.CTkButton(
        target_card, text='Use existing PNG frames',
        command=select_frames_directory, height=36,
        fg_color=SECONDARY_BG,
        text_color=TEXT,
        hover_color=SECONDARY_HOVER
    ).grid(row=3, column=0, padx=15, pady=(0, 15), sticky='ew')

    people = _section(
        content, '02   PEOPLE & SOURCE PHOTOS',
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
        fg_color=SECONDARY_BG,
        text_color=TEXT,
        hover_color=SECONDARY_HOVER,
        command=clear_assignments
    ).grid(row=2, column=1, padx=(8, 20), pady=(0, 12), sticky='ew')
    assignments_label = ctk.CTkLabel(
        people, text='', anchor='w', text_color=MUTED,
        wraplength=780
    )
    assignments_label.grid(
        row=3, column=0, columnspan=2, padx=22, pady=(0, 16), sticky='ew'
    )
    source_card = ctk.CTkFrame(
        people, fg_color=CARD, corner_radius=16,
        border_width=1, border_color=BORDER
    )
    source_card.grid(row=4, column=0, columnspan=2,
                     padx=20, pady=(0, 18), sticky='ew')
    source_card.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        source_card, text='SOURCE PHOTO  ·  LATEST MAPPED PERSON', anchor='w',
        font=ctk.CTkFont(size=12, weight='bold'),
        text_color=SUBTLE
    ).grid(row=0, column=0, padx=15, pady=(12, 3), sticky='ew')
    source_label = ctk.CTkLabel(
        source_card, text='Add a person to select a source photo',
        height=150, wraplength=700
    )
    source_label.grid(row=1, column=0, padx=15, pady=(0, 12), sticky='ew')
    update_assignments_label()

    processing = _section(content, '03   OUTPUT & PROCESSING', 'Playback and output options.')
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
        content, '04   OCCLUSION PROTECTION',
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
        content, '05   FACE ENHANCER',
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

    # Shared Previous/Next controls remain visible on every page.
    workflow_cards = (media, people, processing, occlusion, enhancer)
    navigation = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
    navigation.grid(row=2, column=0, padx=24, pady=(3, 9), sticky='ew')
    navigation.grid_columnconfigure(1, weight=1)
    previous_button = ctk.CTkButton(
        navigation, text='←  Previous', width=145, height=40,
        fg_color=SECONDARY_BG, hover_color=SECONDARY_HOVER,
        text_color=TEXT, border_width=1, border_color=BORDER,
        command=lambda: show_step(active_step - 1)
    )
    previous_button.grid(row=0, column=0, sticky='w')
    stage_label = ctk.CTkLabel(
        navigation, text='', text_color=MUTED,
        font=ctk.CTkFont(size=11, weight='bold')
    )
    stage_label.grid(row=0, column=1, padx=12, sticky='ew')
    next_button = ctk.CTkButton(
        navigation, text='Next  →', width=145, height=40,
        command=lambda: show_step(active_step + 1)
    )
    next_button.grid(row=0, column=2, sticky='e')

    def show_step(index: int) -> None:
        nonlocal active_step
        if _display_workflow_step(
            index, workflow_cards, workflow_tabs, previous_button,
            next_button, stage_label, content
        ):
            active_step = index

    show_step(0)

    footer = ctk.CTkFrame(
        root, fg_color='#0B1430', corner_radius=0
    )
    footer.grid(row=3, column=0, sticky='ew')
    for column in range(3):
        footer.grid_columnconfigure(column, weight=1)
    start_button = ctk.CTkButton(
        footer, text='Start processing', height=46,
        command=lambda: select_output_path(lambda: launch_background(start))
    )
    start_button.grid(row=0, column=0, padx=(24, 7), pady=(14, 5), sticky='ew')
    preview_button = ctk.CTkButton(
        footer, text='Preview', height=44, command=toggle_preview,
        fg_color=SECONDARY_BG,
        text_color=TEXT,
        hover_color=SECONDARY_HOVER
    )
    preview_button.grid(row=0, column=1, padx=7, pady=(14, 5), sticky='ew')
    ctk.CTkButton(
        footer, text='Close', height=44, command=lambda: request_close(destroy),
        fg_color=SECONDARY_BG,
        text_color=TEXT,
        hover_color=SECONDARY_HOVER
    ).grid(row=0, column=2, padx=(7, 24), pady=(14, 5), sticky='ew')
    status_label = ctk.CTkLabel(
        footer, text='●  READY  /  Select your target video and map a source face.',
        anchor='w', justify='left', wraplength=830,
        text_color=MUTED
    )
    status_label.grid(
        row=1, column=0, columnspan=3, padx=24, pady=(0, 8), sticky='ew'
    )
    github = ctk.CTkLabel(
        footer, text='Gourieff GitHub  ·  Original project',
        cursor='hand2', text_color=BLUE,
        font=ctk.CTkFont(size=12)
    )
    progress_bar = ctk.CTkProgressBar(footer, mode='determinate', height=12)
    progress_bar.set(0)
    progress_bar.grid(row=2, column=0, columnspan=3, padx=24, pady=(5, 4), sticky='ew')
    progress_label = ctk.CTkLabel(footer, text='No processing in progress', anchor='w', text_color=MUTED)
    progress_label.grid(row=3, column=0, columnspan=3, padx=24, pady=(0, 5), sticky='ew')
    github.grid(row=4, column=0, columnspan=3, padx=24, pady=(0, 9))
    github.bind('<Button-1>', lambda event: webbrowser.open('https://github.com/Gourieff'))
    return root


def open_settings() -> None:
    """Edit persistent defaults; existing media and identity mappings stay in RAM."""
    if _processing_active:
        update_status('Finish the current processing job before changing Settings.')
        return

    dialog = ctk.CTkToplevel(ROOT, fg_color=BG)
    _set_app_icon(dialog)
    dialog.title('ReActor Studio · Settings')
    dialog.geometry('680x700')
    dialog.minsize(580, 540)
    _bring_dialog_to_front(dialog)

    ctk.CTkLabel(
        dialog, text='Application settings',
        font=ctk.CTkFont(size=24, weight='bold'), anchor='w'
    ).pack(fill='x', padx=24, pady=(20, 2))
    ctk.CTkLabel(
        dialog, text='Saved in your Windows profile; retained after app updates.',
        text_color=MUTED, anchor='w'
    ).pack(fill='x', padx=24, pady=(0, 12))

    content = ctk.CTkScrollableFrame(dialog, fg_color='transparent')
    content.pack(fill='both', expand=True, padx=18, pady=(0, 8))
    content.grid_columnconfigure(1, weight=1)

    # Workstation defaults are deliberately CPU / 2 threads / 6 GB / quality 10.
    current = load_settings()
    current.update({
        'keep_fps': bool(modules.globals.keep_fps),
        'keep_audio': bool(modules.globals.keep_audio),
        'keep_frames': bool(modules.globals.keep_frames),
        'occlusion_protection': bool(modules.globals.occlusion_protection),
        'occlusion_landmark_mode': modules.globals.occlusion_landmark_mode,
        'enhancer_enabled': bool(modules.globals.fp_ui.get('face_enhancer', False)),
        'enhancer_mode': modules.globals.enhancer_mode,
        'enhancer_strength': modules.globals.enhancer_strength,
        'models_directory': modules.globals.models_directory or recommended_models_directory(),
    })
    entries = {}

    def add_entry(row: int, title: str, value, options=None):
        ctk.CTkLabel(content, text=title, anchor='w').grid(
            row=row, column=0, padx=(8, 18), pady=11, sticky='w'
        )
        if options is None:
            field = ctk.CTkEntry(content, width=240)
            field.insert(0, str(value))
        else:
            field = ctk.CTkOptionMenu(content, values=options)
            field.set(str(value))
        field.grid(row=row, column=1, padx=(4, 8), pady=11, sticky='ew')
        return field

    # Show only providers available in this installed ONNX Runtime.
    import onnxruntime
    provider_map = {
        'CPUExecutionProvider': 'cpu',
        'CUDAExecutionProvider': 'cuda',
        'DmlExecutionProvider': 'dml',
        'ROCMExecutionProvider': 'rocm',
        'CoreMLExecutionProvider': 'coreml'
    }
    providers = list(dict.fromkeys(
        ['cpu'] + [provider_map[p] for p in onnxruntime.get_available_providers()
                   if p in provider_map]
    ))
    provider = current.get('execution_provider', 'cpu')
    if provider not in providers:
        provider = 'cpu'

    ctk.CTkLabel(
        content, text='PERFORMANCE',
        font=ctk.CTkFont(size=14, weight='bold'), anchor='w'
    ).grid(row=0, column=0, columnspan=2, pady=(8, 3), sticky='ew')

    entries['execution_provider'] = add_entry(1, 'Execution provider', provider, providers)
    entries['execution_threads'] = add_entry(
        2, 'Processing threads', modules.globals.execution_threads or current['execution_threads']
    )
    entries['max_memory'] = add_entry(
        3, 'Maximum memory (GB)', modules.globals.max_memory or current['max_memory']
    )
    entries['video_quality'] = add_entry(
        4, 'Video quality CRF (0–51)', modules.globals.video_quality
        if modules.globals.video_quality is not None else current['video_quality']
    )
    entries['video_encoder'] = add_entry(
        5, 'Video encoder', modules.globals.video_encoder or current['video_encoder'],
        ['libx264', 'libx265', 'libvpx-vp9']
    )

    ctk.CTkLabel(
        content, text='OFFLINE MODELS',
        font=ctk.CTkFont(size=14, weight='bold'), anchor='w'
    ).grid(row=6, column=0, columnspan=2, pady=(12, 3), sticky='ew')
    model_label = ctk.CTkLabel(content, text='Models folder', anchor='w')
    model_label.grid(row=7, column=0, padx=(8, 18), pady=11, sticky='w')
    folder = ctk.CTkFrame(content, fg_color='transparent')
    folder.grid(row=7, column=1, padx=(4, 8), pady=11, sticky='ew')
    folder.grid_columnconfigure(0, weight=1)
    entries['models_directory'] = ctk.CTkEntry(folder)
    entries['models_directory'].insert(0, current['models_directory'])
    entries['models_directory'].grid(row=0, column=0, padx=(0, 6), sticky='ew')

    def browse_models():
        selected = ctk.filedialog.askdirectory(
            parent=dialog, title='Choose folder containing your existing ONNX models'
        )
        if selected:
            entries['models_directory'].delete(0, 'end')
            entries['models_directory'].insert(0, selected)

    ctk.CTkButton(
        folder, text='Browse', width=75, command=browse_models
    ).grid(row=0, column=1, sticky='e')

    ctk.CTkLabel(
        content,
        text='Reuse models/inswapper_128.onnx and models/xseg_1.onnx from '
             'your previous installation. Other InsightFace weights remain in '
             'your user profile. Models are not included in the installer.',
        wraplength=560, justify='left', text_color=MUTED
    ).grid(row=8, column=0, columnspan=2, padx=8, pady=(3, 12), sticky='ew')
    ctk.CTkLabel(
        content, text='All current Keep FPS/Audio/Frames, XSeg and Enhancer '
                      'switches on the main screen are also saved as defaults.',
        wraplength=560, justify='left', text_color=MUTED
    ).grid(row=9, column=0, columnspan=2, padx=8, pady=(3, 12), sticky='ew')

    feedback = ctk.CTkLabel(dialog, text='', anchor='w', wraplength=600)
    feedback.pack(fill='x', padx=24, pady=(0, 8))

    def save():
        values = dict(current)
        values.update({key: widget.get() for key, widget in entries.items()})
        try:
            values['execution_threads'] = int(values['execution_threads'])
            values['max_memory'] = int(values['max_memory'])
            values['video_quality'] = int(values['video_quality'])
            values['frames_fps'] = float(frames_fps_entry.get().replace(',', '.'))
        except (ValueError, TypeError):
            feedback.configure(text='Invalid numeric value (threads, memory, CRF or FPS).')
            return
        checked = validate(values)
        for key in ('execution_threads', 'max_memory', 'video_quality', 'frames_fps'):
            if checked[key] != values[key]:
                feedback.configure(text=f'Invalid value for {key}. Check the allowed range.')
                return
        if values['models_directory'] and Path(values['models_directory']).exists() and not Path(values['models_directory']).is_dir():
            feedback.configure(text='The models path is a file, not a folder.')
            return
        try:
            saved_at = save_settings(checked)
        except OSError as error:
            feedback.configure(text=f'Could not save settings: {error}')
            return
        # Apply ordinary processing values to subsequent jobs in this session.
        modules.globals.execution_threads = checked['execution_threads']
        modules.globals.video_quality = checked['video_quality']
        modules.globals.video_encoder = checked['video_encoder']
        modules.globals.models_directory = checked['models_directory']
        modules.globals.frames_fps = checked['frames_fps']
        frames_fps_entry.delete(0, 'end')
        frames_fps_entry.insert(0, str(checked['frames_fps']))
        # Provider and memory cap require restart (loaded models/resource limits).
        update_status(f'Settings saved to {saved_at}. Restart for provider/memory changes.')
        dialog.destroy()

    bar = ctk.CTkFrame(dialog, fg_color='transparent')
    bar.pack(fill='x', padx=24, pady=(0, 20))
    ctk.CTkButton(bar, text='Save defaults', width=180, height=40,
                  command=save).pack(side='right')
    ctk.CTkButton(
        bar, text='Cancel', width=110, height=40,
        fg_color=SECONDARY_BG,
        text_color=TEXT,
        command=dialog.destroy
    ).pack(side='right', padx=(0, 10))


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent, fg_color=BG)
    _set_app_icon(preview)
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
    """Accept status messages from any thread via a GUI event queue."""
    _UI_EVENTS.put({'type': 'status', 'message': text})


def show_warning(title: str, message: str) -> None:
    """Queue modal dialogs; only Tk's main thread creates them."""
    _UI_EVENTS.put({'type': 'warning', 'title': title, 'message': message})


def _format_duration(seconds) -> str:
    if seconds is None:
        return 'calculating'
    remaining = max(0, int(round(seconds)))
    if remaining >= 3600:
        return f'{remaining // 3600}h {(remaining % 3600) // 60:02d}m'
    if remaining >= 60:
        return f'{remaining // 60}m {remaining % 60:02d}s'
    return f'{remaining}s'


def _set_controls_enabled(enabled: bool) -> None:
    """Lock mutable application options while the worker is processing."""
    state = 'normal' if enabled else 'disabled'

    def descend(widget) -> None:
        for child in widget.winfo_children():
            if isinstance(child, (ctk.CTkButton, ctk.CTkSwitch,
                                  ctk.CTkSlider, ctk.CTkOptionMenu,
                                  ctk.CTkEntry)):
                child.configure(state=state)
            descend(child)

    # The main scrollable section contains all media and processing settings.
    for widget in ROOT.winfo_children():
        if isinstance(widget, ctk.CTkScrollableFrame):
            descend(widget)
    for widget in (start_button, preview_button, settings_button):
        if widget is not None:
            widget.configure(state=state)


def _drain_events() -> None:
    """Apply queued events only on the Tk thread, then schedule next poll."""
    global _processing_active
    if ROOT is None or not ROOT.winfo_exists():
        return
    for _ in range(200):
        try:
            event = _UI_EVENTS.get_nowait()
        except queue.Empty:
            break
        kind = event['type']
        if kind == 'status':
            status_label.configure(text=event['message'])
        elif kind == 'warning':
            messagebox.showwarning(event['title'], event['message'], parent=ROOT)
        elif kind == 'error':
            status_label.configure(text=event['message'])
            messagebox.showerror('ReActor processing error', event['message'], parent=ROOT)
        elif kind == 'progress':
            if progress_bar.cget('mode') != 'determinate':
                progress_bar.stop()
                progress_bar.configure(mode='determinate')
            done, total = event['done'], event['total']
            percent = (100.0 * done / total) if total else 0.0
            progress_bar.set(percent / 100.0)
            rate = event['rate']
            speed = (f'{rate:.2f} frames/s' if rate >= 1 else
                     f'{1 / rate:.2f} s/frame' if rate > 0 else 'calculating speed')
            progress_label.configure(
                text=f"{event['stage']}  ·  {done}/{total} frames  ·  "
                     f"{percent:.1f}%  ·  {speed}  ·  ETA {_format_duration(event['eta'])}"
            )
        elif kind == 'finished':
            _processing_active = False
            _set_controls_enabled(True)
            if progress_bar.cget('mode') == 'indeterminate':
                progress_bar.stop()
                progress_bar.configure(mode='determinate')
                progress_bar.set(0)
    ROOT.after(100, _drain_events)


def request_close(destroy: Callable[[], None]) -> None:
    if _processing_active:
        update_status('Processing is still running. Close is available when it finishes.')
        return
    progress_events.set_sink(None)
    destroy()


def launch_background(process: Callable[[], None]) -> None:
    """Process videos in a worker so Windows continues to repaint the GUI."""
    global _processing_active
    if _processing_active:
        return
    _processing_active = True
    progress_events.reset()
    _set_controls_enabled(False)
    progress_label.configure(text='Preparing models and processing…')
    progress_bar.configure(mode='indeterminate')
    progress_bar.start()

    def worker() -> None:
        try:
            process()
        except Exception as error:
            traceback.print_exc()
            _UI_EVENTS.put({
                'type': 'error',
                'message': f'{type(error).__name__}: {error}',
            })
        finally:
            _UI_EVENTS.put({'type': 'finished'})

    threading.Thread(target=worker, name='ReActorProcessing',
                     daemon=True).start()


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

    comparison = ctk.CTkToplevel(ROOT, fg_color=BG)
    _set_app_icon(comparison)
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
        text_color=MUTED, anchor='w'
    )
    instruction.pack(fill='x')

    viewport = ctk.CTkFrame(comparison, fg_color='transparent')
    viewport.pack(fill='both', expand=True, padx=16, pady=6)
    viewport.grid_columnconfigure(0, weight=1)
    viewport.grid_rowconfigure(0, weight=1)

    # CTkScrollableFrame offers one scroll direction; a Canvas allows both
    # directions without shrinking or cropping the source-resolution images.
    canvas = tk.Canvas(
        viewport, bg=BG,
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

    def wheel_scroll(event) -> None:
        # Shift + wheel scrolls left/right. Ordinary wheel scrolls vertically.
        units = int(-event.delta / 120) if event.delta else 0
        if event.state & 0x1:
            canvas.xview_scroll(units, 'units')
        else:
            canvas.yview_scroll(units, 'units')

    canvas.bind('<MouseWheel>', wheel_scroll)
    contents.bind('<MouseWheel>', wheel_scroll)

    ctk.CTkLabel(
        comparison,
        text='Actual frame resolution · Scroll to inspect details · Preview only; no video files changed.',
        text_color=MUTED
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
                contents, fg_color=CARD, corner_radius=12
            )
            card.pack(anchor='nw', padx=12, pady=(10, 4))
            ctk.CTkLabel(
                card, text=f'{amount}% enhancement  ·  {image.width} × {image.height}',
                font=ctk.CTkFont(size=15, weight='bold'), anchor='w'
            ).pack(fill='x', padx=14, pady=(12, 6))
            preview = ctk.CTkLabel(card, image=shown, text='')
            preview.image = shown  # Preserve the complete image object in memory.
            preview.pack(padx=14, pady=(0, 14))
            preview.bind('<MouseWheel>', wheel_scroll)
            card.bind('<MouseWheel>', wheel_scroll)
        contents.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))
        instruction.configure(text='Ready. Compare the full-size results by scrolling down.')
    except Exception as error:
        instruction.configure(text=f'Comparison failed: {error}')



def update_assignments_label() -> None:
    """Update mapped-person count and the last selected source-photo preview."""
    count = len(modules.globals.face_assignments)
    if assignments_label is not None:
        if count:
            latest = modules.globals.face_assignments[-1]['source_path']
            assignments_label.configure(
                text=f'{count} person(s) mapped  ·  Latest source: {Path(latest).name}'
            )
        else:
            assignments_label.configure(
                text='No people mapped yet. Add a person to start.'
            )
    if source_label is not None:
        if count:
            latest = modules.globals.face_assignments[-1]['source_path']
            image = render_image_preview(latest, (190, 160))
            source_label.configure(image=image, text='')
            source_label.image = image
        elif not modules.globals.source_path:
            source_label.configure(
                image=None, text='Add a person to select a source photo'
            )
            source_label.image = None


def clear_assignments() -> None:
    modules.globals.face_assignments.clear()
    update_assignments_label()
    modules.globals.target_face_embedding = None
    update_status('All person assignments cleared.')


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
    source_label.configure(image=None, text='Add a person to select a source photo')
    source_label.image = None
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
    """Switch media transactionally, keeping the previous job if preview fails.

    In particular, a previously processed video must not leave OpenCV file
    handles or stale frame mappings attached to the next selected target.
    """
    global RECENT_DIRECTORY_TARGET, FRAMES_PREVIEW_PATHS, img_ft, vid_ft

    target_path = ctk.filedialog.askopenfilename(
        title='Select target video or image',
        initialdir=RECENT_DIRECTORY_TARGET, filetypes=[vid_ft, img_ft],
        parent=ROOT
    )
    if not target_path:
        return
    try:
        if is_image(target_path):
            thumbnail = render_image_preview(target_path, (200, 200))
        elif is_video(target_path):
            thumbnail = render_video_preview(target_path, (200, 200))
        else:
            update_status('Not a supported image or video file.')
            return
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        update_status(f'Cannot open selected target: {error}')
        return

    # Only commit the new target once the preview is successfully generated.
    PREVIEW.withdraw()
    modules.globals.target_path = target_path
    modules.globals.output_path = None
    modules.globals.target_face_embedding = None
    modules.globals.frames_directory = None
    modules.globals.frame_target_boxes.clear()
    modules.globals.preview_face_boxes = []
    FRAMES_PREVIEW_PATHS = []
    modules.globals.face_assignments.clear()
    update_assignments_label()
    RECENT_DIRECTORY_TARGET = os.path.dirname(target_path)
    target_label.configure(image=thumbnail, text='')
    target_label.image = thumbnail
    update_status(
        f'Target selected: {Path(target_path).name}. '
        'Add a person and source photo before starting.'
    )


def open_face_picker(multiple: bool = False) -> None:
    """Select a face on an original video frame and remember its identity.

    All detection happens on the local machine. The selected embedding remains
    in RAM, and frames that cannot be matched confidently are left untouched.
    """
    target_path = modules.globals.target_path
    if not modules.globals.frames_directory and not (is_image(target_path) or is_video(target_path)):
        update_status('Select a target video, image, or PNG frames folder first.')
        return

    picker = ctk.CTkToplevel(ROOT, fg_color=BG)
    _set_app_icon(picker)
    picker.title('Add person + reference photo' if multiple else 'Choose the person to replace')
    picker.geometry('900x720')
    picker.minsize(650, 650)
    _bring_dialog_to_front(picker)

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
                        filetypes=[img_ft], parent=picker
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
                    update_assignments_label()
                    update_status(
                        f'{len(modules.globals.face_assignments)} person(s) mapped to source photos.'
                    )
                else:
                    modules.globals.face_assignments.clear()
                    update_assignments_label()
                    modules.globals.target_face_embedding = embedding.copy()
                    modules.globals.many_faces = False
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
    picker.lift()


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
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(light_image=image, dark_image=image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int],
                         frame_number: int = 0) -> ctk.CTkImage:
    """Read one frame and *always* release VideoCapture, even on success."""
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Unable to open video: {Path(video_path).name}')
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame or frame is None:
            raise RuntimeError(
                f'Unable to read preview frame: {Path(video_path).name}'
            )
    finally:
        capture.release()

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(light_image=image, dark_image=image, size=image.size)


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
