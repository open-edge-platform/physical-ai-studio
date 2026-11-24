import '@fontsource-variable/inter/wght.css';
import '@fontsource-variable/jetbrains-mono/wght.css';
import {flushSync} from 'react-dom';
import {createRoot} from 'react-dom/client';
import {sceneHeight, sceneWidth} from './animation.ts';
import {demos, type DemoId} from './config.ts';
import {Preview} from './Preview';
import {Demo} from './Root';

// Capture uses an explicit timestamp, never a running CSS animation or wall clock.
declare global {
  interface Window {
    renderFrame: (id: DemoId, frame: number) => Promise<void>;
  }
}

const capture = new URLSearchParams(location.search).has('capture');
const element = document.getElementById('root')!;
const root = createRoot(element);

const resize = () => {
  const scale = Math.min(innerWidth / sceneWidth, (innerHeight - (capture ? 0 : 64)) / sceneHeight);
  element.style.width = `${sceneWidth}px`;
  element.style.height = `${sceneHeight}px`;
  element.style.transform = `scale(${scale})`;
};
addEventListener('resize', resize);
resize();

window.renderFrame = async (id, frame) => {
  if (!Object.hasOwn(demos, id) || !Number.isFinite(frame) || frame < 0 || frame >= demos[id].durationInFrames) {
    throw new RangeError(`Invalid ${id} frame: ${frame}`);
  }
  // Remount on seek so transformed text cannot reuse raster layers from another frame.
  flushSync(() => root.render(<Demo key={`${id}:${frame}`} id={id} frame={frame} />));
  await document.fonts.ready;
  await Promise.all(Array.from(document.images, (image) => image.decode()));
  // Let layout and the compositor settle after a seek before taking a screenshot.
  await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
};

if (capture) {
  await window.renderFrame('api', 0);
} else {
  root.render(<Preview />);
}
