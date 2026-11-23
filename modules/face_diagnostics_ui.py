"""Small, read-only diagnostic window for the selected preview frame.

Face analysis runs in a daemon worker; all Tk interactions run on Tk's thread.
Nothing is written to files or sent over the network by this feature.
"""
import queue
import threading
import traceback

import customtkinter as ctk

from modules.face_diagnostics import analyse_preview_faces


class FaceDiagnosticDialog:
    def __init__(self, parent, original_frame, assignments, detector,
                 *, vr_stereo=False, icon_setter=None):
        self._queue = queue.Queue(maxsize=1)
        self._closed = False
        self._after_id = None
        self._window = ctk.CTkToplevel(parent, fg_color='#050816')
        if icon_setter is not None:
            icon_setter(self._window)
        self._window.title('Face detection diagnostics')
        self._window.geometry('660x480')
        self._window.minsize(500, 360)
        self._window.protocol('WM_DELETE_WINDOW', self.close)

        ctk.CTkLabel(
            self._window, text='FACE DETECTION DIAGNOSTICS',
            text_color='#F5F7FF',
            font=ctk.CTkFont(size=18, weight='bold')
        ).pack(fill='x', anchor='w', padx=20, pady=(18, 8))
        self._panel = ctk.CTkScrollableFrame(
            self._window, fg_color='#131D38', corner_radius=12
        )
        self._panel.pack(fill='both', expand=True, padx=20, pady=(0, 16))
        self._line('Analysing the selected original frame…')
        self._after_id = self._window.after(100, self._poll)

        # Keep the session's original frame and mapping embeddings isolated;
        # the face detector does not see a processed Preview image.
        snapshot = [
            {'embedding': item['embedding'].copy()} for item in assignments
        ]
        worker = threading.Thread(
            target=self._analyse,
            args=(original_frame.copy(), snapshot, detector, vr_stereo),
            daemon=True, name='ReActorFaceDiagnostics'
        )
        worker.start()

    def _line(self, message, *, color='#AAB3C5', bold=False):
        ctk.CTkLabel(
            self._panel, text=message, anchor='w', justify='left',
            text_color=color, wraplength=590,
            font=ctk.CTkFont(size=13, weight='bold' if bold else 'normal')
        ).pack(fill='x', padx=12, pady=(5, 4))

    def _analyse(self, frame, assignments, detector, vr_stereo):
        try:
            data = analyse_preview_faces(
                frame, assignments, detector, vr_stereo=vr_stereo
            )
            error = None
        except Exception as exc:
            traceback.print_exc()
            data, error = None, str(exc)
        if not self._closed:
            self._queue.put((data, error))

    def _poll(self):
        self._after_id = None
        if self._closed or not self._window.winfo_exists():
            return
        try:
            data, error = self._queue.get_nowait()
        except queue.Empty:
            self._after_id = self._window.after(100, self._poll)
            return
        for child in self._panel.winfo_children():
            child.destroy()
        if error:
            self._line(f'Unable to diagnose this frame: {error}', color='#F5F7FF')
            return
        self._line(
            f"Similarity threshold: {data['min_similarity']:.2f}  ·  "
            f"Identity ambiguity margin: {data['min_margin']:.2f}",
            color='#4F8CFF'
        )
        if not data['sections']:
            self._line(data['message'])
            return
        for section in data['sections']:
            self._line(
                f"{section['title']} · {section['detected']} detected · "
                f"{section['matched']} accepted",
                color='#F5F7FF', bold=True
            )
            if not section['detected']:
                self._line('No face detected in the original orientation.')
            for row in section['rows']:
                value = (
                    f"{row['score']:.3f}" if row['score'] is not None
                    else 'not available'
                )
                self._line(
                    f"Face {row['face']} / Person {row['person']:02d} · "
                    f"Similarity {value}\n{row['reason']}",
                    color='#4F8CFF' if row['matched'] else '#AAB3C5'
                )
            for trial in section['rotations']:
                best = (
                    f"Person {trial['best_person']:02d}: "
                    f"{trial['best_similarity']:.3f}"
                    if trial['best_person'] is not None
                    else 'No matching embedding'
                )
                outcome = (
                    'Matching possible after rotation'
                    if trial['matched'] else 'Still below matching requirements'
                )
                self._line(
                    f"Rotation test {trial['degrees']:+d}° · "
                    f"{trial['count']} face(s) · {best}\n{outcome}"
                )
        self._line(data['message'])

    def close(self):
        self._closed = True
        if self._after_id is not None:
            try:
                self._window.after_cancel(self._after_id)
            except (RuntimeError, ValueError):
                pass
            self._after_id = None
        if self._window.winfo_exists():
            self._window.destroy()
