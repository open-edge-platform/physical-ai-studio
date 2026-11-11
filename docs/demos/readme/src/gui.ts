import {fps} from './animation.ts';

export type GuiClip = {
  title: string;
  detail: string;
  accent: string;
  from: number;
  to: number;
};

// Use the committed recording, not the GIF previously generated from it.
export const guiSource = '../../assets/physical_ai_studio_full_overview.mp4';
export const guiImageFps = 8;

export const guiClips: GuiClip[] = [
  {title: 'Project', detail: 'Start a robotics project', accent: '#7c9cff', from: 0, to: 1},
  {title: 'Robots', detail: 'Configure robot arms', accent: '#c084fc', from: 1, to: 3},
  {title: 'Cameras', detail: 'Connect camera streams', accent: '#21d4a7', from: 3, to: 8},
  {title: 'Environment', detail: 'Pair robots and cameras', accent: '#ffb86b', from: 8, to: 10.5},
  {title: 'Dataset', detail: 'Create a recording dataset', accent: '#f472b6', from: 10.5, to: 11.5},
  {title: 'Record', detail: 'Capture demonstrations', accent: '#38bdf8', from: 11.5, to: 17},
  {title: 'Train', detail: 'Launch policy training', accent: '#a3e635', from: 17, to: 22},
  {title: 'Models', detail: 'Review trained artifacts', accent: '#facc15', from: 22, to: 25},
  {title: 'Run', detail: 'Deploy to physical robots', accent: '#fb7185', from: 25, to: 1061 / fps},
];

export const clipDuration = (clip: GuiClip): number => Math.round((clip.to - clip.from) * fps);

export const guiDuration = guiClips.reduce((total, clip) => total + clipDuration(clip), 0);

export const guiFrameUrl = (frame: number): string => {
  const index = Math.floor((frame * guiImageFps) / fps + 1e-6);
  return `/gui-source/${String(index).padStart(6, '0')}.jpg`;
};
