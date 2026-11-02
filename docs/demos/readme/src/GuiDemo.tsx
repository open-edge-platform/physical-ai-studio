import type {CSSProperties} from 'react';
import {AbsoluteFill, OffthreadVideo, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {clipDuration, fps, guiClips, guiSource} from './gui';

const font = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const mono = '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace';

const card: CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(15, 20, 35, 0.72)',
  boxShadow: '0 30px 90px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06)',
  backdropFilter: 'blur(20px)',
};

const starts = guiClips.reduce<number[]>((result, clip, index) => {
  result.push(index === 0 ? 0 : result[index - 1] + clipDuration(guiClips[index - 1]));
  return result;
}, []);

export const GuiDemo = () => {
  const frame = useCurrentFrame();
  const current = starts.reduce((active, start, index) => (frame >= start ? index : active), 0);
  const clip = guiClips[current];
  const clipFrame = frame - starts[current];
  const labelOpacity = interpolate(clipFrame, [0, 10, clipDuration(clip) - 10, clipDuration(clip)], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background:
          'radial-gradient(circle at 12% 12%, rgba(88,120,255,0.34), transparent 30%), radial-gradient(circle at 88% 78%, rgba(33,212,167,0.20), transparent 32%), linear-gradient(135deg, #070a13 0%, #0c1222 48%, #090d18 100%)',
        color: 'white',
        fontFamily: font,
        padding: 64,
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
        <div>
          <div style={{fontSize: 20, fontWeight: 700, letterSpacing: 3, color: '#8ea8ff'}}>STUDIO UI</div>
          <div style={{fontSize: 48, fontWeight: 800, letterSpacing: -1.4, marginTop: 8}}>From robot setup to deployment</div>
        </div>
        <div style={{...card, borderRadius: 999, padding: '14px 22px', fontFamily: mono, fontSize: 20, color: '#cbd5e1'}}>
          localhost:7860
        </div>
      </div>

      <div style={{display: 'flex', gap: 30, marginTop: 42}}>
        <div style={{...card, borderRadius: 24, width: 1140, height: 666, overflow: 'hidden'}}>
          <div
            style={{
              height: 44,
              display: 'flex',
              alignItems: 'center',
              gap: 9,
              padding: '0 18px',
              borderBottom: '1px solid rgba(255,255,255,0.08)',
              color: '#94a3b8',
              fontSize: 15,
            }}
          >
            <span style={{width: 11, height: 11, borderRadius: 99, background: '#ff5f57'}} />
            <span style={{width: 11, height: 11, borderRadius: 99, background: '#febc2e'}} />
            <span style={{width: 11, height: 11, borderRadius: 99, background: '#28c840'}} />
            <span style={{marginLeft: 14}}>Physical AI Studio</span>
          </div>
          <div style={{position: 'relative', width: 1140, height: 622, background: '#101318'}}>
            <OffthreadVideo
              src={staticFile(guiSource)}
              startFrom={Math.round(guiClips[0].from * fps)}
              endAt={Math.round(guiClips[guiClips.length - 1].to * fps)}
              muted
              style={{width: '100%', height: '100%', objectFit: 'cover'}}
            />
            <div
              style={{
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: 0,
                height: 112,
                background: 'linear-gradient(180deg, rgba(8,11,20,0), rgba(8,11,20,0.86))',
              }}
            />
            <div
              style={{
                position: 'absolute',
                left: 28,
                bottom: 24,
                display: 'flex',
                alignItems: 'center',
                gap: 16,
                opacity: labelOpacity,
              }}
            >
              <span style={{width: 12, height: 12, borderRadius: 99, background: clip.accent, boxShadow: `0 0 20px ${clip.accent}`}} />
              <span style={{fontSize: 28, fontWeight: 800}}>{clip.title}</span>
              <span style={{fontSize: 20, color: '#cbd5e1'}}>{clip.detail}</span>
            </div>
          </div>
        </div>

        <div style={{display: 'flex', flexDirection: 'column', gap: 8, width: 302}}>
          {guiClips.map((item, index) => {
            const active = current === index;
            const done = current > index;
            return (
              <div
                key={item.title}
                style={{
                  ...card,
                  borderRadius: 16,
                  height: 66,
                  padding: '0 18px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 14,
                  borderColor: active ? `${item.accent}aa` : 'rgba(255,255,255,0.10)',
                  transform: `scale(${active ? 1.025 : 1})`,
                  opacity: active || done ? 1 : 0.58,
                }}
              >
                <span
                  style={{
                    width: 26,
                    height: 26,
                    borderRadius: 99,
                    display: 'grid',
                    placeItems: 'center',
                    background: done ? item.accent : 'rgba(255,255,255,0.08)',
                    color: '#07111f',
                    fontSize: 14,
                    fontWeight: 900,
                  }}
                >
                  {done ? '✓' : ''}
                </span>
                <span style={{fontSize: 21, fontWeight: 800}}>{item.title}</span>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
