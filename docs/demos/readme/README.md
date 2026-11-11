# README demo assets

Source files for the Python API and Studio UI demos embedded in the repository README.

```bash
cd docs/demos/readme
npm install
npm run render
```

This writes the two README GIFs to `docs/assets/readme/`. Intermediate MP4 files are written to `out/` and ignored by Git.

Before rendering the Studio UI demo, copy the captionless 720p walkthrough to `public/studio-walkthrough-720p.mp4`. Source videos are ignored by Git; don't commit them.

## Source files

- `src/WorkflowDemo.tsx` and `src/workflow.ts` define the Python API animation.
- `src/GuiDemo.tsx` and `src/gui.ts` place the Studio walkthrough in the same visual theme.
- `scripts/to-gif.sh` converts rendered MP4 files into README GIFs.

Render a single asset with `npm run render:api` or `npm run render:gui`.

Requirements: Node.js and FFmpeg.
