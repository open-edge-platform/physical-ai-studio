export type GuiClip = {
  title: string;
  detail: string;
  accent: string;
  from: number;
  to: number;
};

export const guiSource = 'studio-walkthrough-720p.mp4';

export const guiClips: GuiClip[] = [
  {title: 'Project', detail: 'Start a robotics project', accent: '#7c9cff', from: 0, to: 1},
  {title: 'Robots', detail: 'Configure robot arms', accent: '#c084fc', from: 1, to: 4},
  {title: 'Cameras', detail: 'Connect camera streams', accent: '#21d4a7', from: 4, to: 9},
  {title: 'Environment', detail: 'Pair robots and cameras', accent: '#ffb86b', from: 9, to: 16},
  {title: 'Dataset', detail: 'Create a recording dataset', accent: '#f472b6', from: 16, to: 22},
  {title: 'Record', detail: 'Capture demonstrations', accent: '#38bdf8', from: 22, to: 28},
  {title: 'Train', detail: 'Launch policy training', accent: '#a3e635', from: 28, to: 33},
  {title: 'Models', detail: 'Review trained artifacts', accent: '#facc15', from: 33, to: 37},
  {title: 'Run', detail: 'Deploy to physical robots', accent: '#fb7185', from: 37, to: 42.666},
];

export const fps = 30;

export const clipDuration = (clip: GuiClip) => Math.round((clip.to - clip.from) * fps);

export const guiDuration = guiClips.reduce((total, clip) => total + clipDuration(clip), 0);
