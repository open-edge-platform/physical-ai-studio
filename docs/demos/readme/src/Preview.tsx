import {useEffect, useState} from 'react';
import {createPortal} from 'react-dom';
import {fps} from './animation.ts';
import {demos, type DemoId} from './config.ts';
import {Demo} from './Root';

export const Preview = () => {
  const [id, setId] = useState<DemoId>('api');
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    let request: number;
    let previous = performance.now();
    const tick = (now: number) => {
      const elapsed = ((now - previous) * fps) / 1000;
      previous = now;
      setFrame((value) => (value + elapsed) % demos[id].durationInFrames);
      request = requestAnimationFrame(tick);
    };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  }, [id, playing]);

  return (
    <>
      <Demo id={id} frame={frame} />
      {createPortal(
        <div id="controls">
          <select
            aria-label="Demo"
            value={id}
            onChange={(event) => {
              setId(event.target.value as DemoId);
              setFrame(0);
            }}
          >
            <option value="api">Python API</option>
            <option value="gui">Studio UI</option>
          </select>
          <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
          <input
            aria-label="Frame"
            type="range"
            min={0}
            max={demos[id].durationInFrames - 1}
            value={frame}
            onChange={(event) => {
              setPlaying(false);
              setFrame(Number(event.target.value));
            }}
          />
          <output>{(frame / fps).toFixed(1)} s</output>
        </div>,
        document.body,
      )}
    </>
  );
};
