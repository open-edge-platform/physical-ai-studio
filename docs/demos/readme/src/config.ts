import {fps} from './animation.ts';
import {guiDuration, guiImageFps} from './gui.ts';

export const demos = {
  api: {durationInFrames: 500, outputFps: 15, width: 1200, height: 675},
  gui: {durationInFrames: guiDuration, outputFps: guiImageFps, width: 900, height: 506},
} as const;

export type DemoId = keyof typeof demos;

export const outputFrameCount = (id: DemoId): number =>
  Math.ceil((demos[id].durationInFrames * demos[id].outputFps) / fps);
