# README demo renderer

Regenerate the Python API and Studio UI animations with React, Playwright, and FFmpeg. No Remotion packages, license key, or paid rendering service are needed. See [tooling and licenses](LICENSES.md) for the alternatives considered and upstream license sources.

## Setup and render

Requirements: Node.js 24+ and FFmpeg on `PATH` (`brew install ffmpeg` on macOS or `sudo apt install ffmpeg` on Ubuntu).

From the repository root:

```bash
cd docs/demos/readme
npm ci
npx playwright install chromium --only-shell
npm run render
```

On Linux, install browser system dependencies with `npx playwright install --with-deps chromium --only-shell` instead.

This replaces `docs/assets/readme/api-demo.gif` and `docs/assets/readme/gui-demo.gif`. Render one asset with `npm run render:api` or `npm run render:gui`.

The GUI input is the [committed Studio recording](../../assets/physical_ai_studio_full_overview.mp4). No private download or uncommitted video is required. Its captions remain part of the recording; the stage timings match this 35.4-second version.

## Preview and verify

```bash
npm run preview # http://127.0.0.1:4173; select a demo, play, or scrub
npm run check
npm test
```

Tests cover timeline boundaries, stage highlights, code reveals, backwards seeking, repeatable captures, preview controls, dependency license declarations, GIF metadata, and failure without overwriting published assets. The GIF checks require `ffprobe`, included with FFmpeg.

## Source files

- `src/WorkflowDemo.tsx` and `src/workflow.ts` define the Python API animation.
- `src/GuiDemo.tsx` and `src/gui.ts` map the Studio recording to workflow stages.
- `src/animation.ts` defines the 30 fps scene timeline; `src/config.ts` sets output dimensions and frame rates (API: 1200 × 675 at 15 fps; GUI: 900 × 506 at 8 fps).
- `src/index.tsx` renders an explicit frame, waits for fonts and images, and settles the browser compositor before capture.
- `scripts/render.ts` bundles the scenes, decodes the recording into still frames, captures each scene with Playwright's Chromium headless shell, and encodes looping GIFs with FFmpeg. It replaces each published GIF only after that GIF finishes successfully.

Intermediate images, palettes, and the preview bundle live in `out/` and are ignored by Git. The renderer does not use the previous GIFs as inputs. Fonts come from the locked, OFL-licensed Inter and JetBrains Mono packages rather than machine-specific font installations.
