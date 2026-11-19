# ReActor-UI modern interface (mapping-first)

The screen is arranged in five resizable, scrollable sections:

1. **Media** — inspect the selected target and choose an image, video, or
   existing numbered PNG sequence. A source preview shows the latest mapped
   source photo.
2. **People & source photos** — click **Add person + source photo**, select a
   target person from a frame, then choose the corresponding source image.
   Repeat for additional people. **Clear person assignments** resets mappings.
   The former 'Select a face', 'Choose person (single)', and 'Many faces'
   controls have been removed from the **GUI only**: the internal CLI/legacy
   functionality and one-to-many and multi-person pipelines are still present.
3. **Processing** — Keep fps, Keep audio, Keep frames, NSFW (legacy behavior),
   and Existing PNG frames FPS.
4. **Occlusion protection** — XSeg toggle and Auto / two-point / three-point /
   five-point landmark alignment.
5. **Face Enhancer** — toggle, GFPGAN/Sharpen selection, strength slider, and
   **Compare enhancer • Full-resolution frames**.

The **Start processing**, **Preview**, and **Close** actions remain visible
in a fixed footer while scrolling through the advanced controls.

## Foreground person-picker dialog

The person-picker is owned by the main application window. It is raised and
briefly made topmost to avoid opening behind the main window on Windows; the
topmost flag is cleared shortly thereafter. The nested source-photo file
selector is explicitly parented to the person-picker window.

## Enhancer comparison

The comparison dialog displays **three full-resolution images** (0%, 40%, 70%)
in a **single vertical column**. The original frame dimensions are preserved,
without using a downscaled thumbnail. For frames larger than the display,
horizontal and vertical scrollbars permit inspection of every pixel. Vertical
mouse-wheel scrolling is enabled over the images; Shift+wheel scrolls
horizontally.

Comparing only renders a local preview; it does not overwrite original
footage, input PNGs, or the user's output video.

## Compatibility and testing

- Main workflow: select target -> add one or more people -> choose source
  photograph for each -> configure processing -> preview or start.
- Frame-folder and headless/CLI logic is unaffected.
- Content classifier, model selection, XSeg auto alignment and multi-face
  assignments are unchanged internally.
- CI runs static/structural interface regression tests without a graphical
  display. **Visual QA on Windows** is still required, especially for
  DPI scaling, popup placement, very large frame comparisons and GPU/CPU
  performance.
