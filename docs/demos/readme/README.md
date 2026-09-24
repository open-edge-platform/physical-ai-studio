# README demo assets

Source files for the animated demos and policy gallery in the repository README.

```bash
cd docs/demos/readme
npm install
npm run render
```

This writes the README GIFs and images to `docs/assets/readme/`. Intermediate MP4 files are written to `out/` and ignored by Git.

Before rendering the Studio UI demo, copy the captionless 720p walkthrough to `public/studio-walkthrough-720p.mp4`. Source videos are ignored by Git; don't commit them.

## Source files

- `src/WorkflowDemo.tsx` defines the shared visual theme and animation for the API and CLI demos.
- `src/workflow.ts` contains the Python API and CLI workflows shown in the demos.
- `src/GuiDemo.tsx` places the Studio walkthrough in the same visual theme.
- `src/gui.ts` maps the captionless walkthrough to Studio stages.
- `src/PolicyGallery.tsx` defines the policy gallery image.
- `src/FeatureGallery.tsx` defines the capabilities and documentation images.
- `src/sections.ts` contains the install commands and README section cards.
- `scripts/to-gif.sh` converts rendered MP4 files into README GIFs.

Render a single asset with `npm run render:api`, `npm run render:cli`, `npm run render:gui`, `npm run render:install`, `npm run render:policies`, `npm run render:build`, or `npm run render:docs`.

Requirements: Node.js and FFmpeg.
