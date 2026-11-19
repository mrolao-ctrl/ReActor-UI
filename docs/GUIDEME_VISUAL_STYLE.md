# ReActor Studio · GuideMe-inspired visual identity

This redesign is based on the **current GuideMe dark theme**, not a generic
black/green redesign. The visual reference is the user's separate
`mrolao-ctrl/guideMe` application:

- Main background: `#050816` (midnight navy)
- Primary action and focused controls: `#6D5CFF` (violet)
- Secondary accent and section details: `#4F8CFF` (electric blue)
- Cards: `#0E162D` / `#131D38`
- Muted text: `#AAB3C5`, with clear white labels and rounded corners

The app retains the **same CustomTkinter UI and all existing behavior**,
including multi-person assignments, target selection, PNG resume, XSeg,
enhancement comparison, per-user Settings and live processing progress.

## New icon

`assets/reactor-studio.png` is a custom face-exchange icon: two stylized
profiles and an exchange motif. It intentionally distinguishes ReActor
from GuideMe while sharing its dark navy / violet / blue graphic language.

- Main window, modal settings, person picker, preview and enhancer comparison:
  use a small version of the same icon on the title bar.
- Main header shows the icon at 56x56 next to the REACTOR wordmark.
- Windows GitHub Actions build converts the PNG to a multi-size `.ico`.
- The Inno Setup installer and both Start Menu/desktop shortcuts use that `.ico`.
- The installer includes both `assets/reactor-studio.png` and
  `assets/reactor-studio.ico`; a source-only launch uses the PNG directly.
- No models, identity embeddings, photos, or video content are included.

## Layout

A compact editorial header leads into a four-stage strip:
`SELECT MEDIA`, `MAP FACES`, `ENHANCE`, `EXPORT`. The five
scrollable panels remain unchanged structurally, with a fixed bottom bar
for Start, Preview, Close and the progress indicator.

## Testing

- Headless tests check the dark theme, presence and validity of the icon,
  retention of key controls, and Windows installer icon wiring.
- GitHub Actions builds a Windows installer and checks syntax/unit tests.
- The user should test actual popup/theme rendering and DPI sizes in
  Windows after installing. No changes were made to Face Swap, XSeg,
  content-safety classification or processing performance.
