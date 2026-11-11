import {Highlight, themes} from 'prism-react-renderer';
import type {CSSProperties} from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';

export type Stage = {
  number: string;
  title: string;
  detail: string;
  start: number;
  end: number;
  accent: string;
};

export type CodeLine = {
  text: string;
  start: number;
  stage?: number;
};

export type WorkflowDemoProps = {
  eyebrow: string;
  title: string;
  command: string;
  filename: string;
  language: 'python' | 'bash';
  lines: CodeLine[];
  stages: Stage[];
};

const font = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const mono = '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace';

const card: CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(15, 20, 35, 0.72)',
  boxShadow: '0 30px 90px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06)',
  backdropFilter: 'blur(20px)',
};

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

const bashColor = (part: string, index: number) => {
  if (part.startsWith('#')) return '#7f8ea3';
  if (index === 0 && ['pip', 'git', 'cd', 'cp', 'docker', './setup-devices.sh'].includes(part)) return '#82aaff';
  if (part === 'physicalai') return '#82aaff';
  if (['fit', 'benchmark', 'export', 'run', 'install', 'clone', 'compose', 'up'].includes(part) && index <= 6) return '#c792ea';
  if (part.startsWith('--')) return '#7fdbca';
  if (part === '\\') return '#637777';
  if (part.includes('/') || part.includes('.')) return '#ecc48d';
  return '#d6deeb';
};

const BashLine = ({text}: {text: string}) => {
  if (text.startsWith('#')) return <span style={{color: bashColor(text, 0)}}>{text}</span>;
  return (
    <>
      {text.split(/(\s+)/).map((part, index) => (
        <span key={index} style={{color: part.trim() ? bashColor(part, index) : undefined}}>
          {part}
        </span>
      ))}
    </>
  );
};

export const WorkflowDemo = ({eyebrow, title, command, filename, language, lines, stages}: WorkflowDemoProps) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const intro = spring({frame, fps, config: {damping: 18, stiffness: 90}});
  const current = stages.reduce((active, stage, index) => (frame >= stage.start ? index : active), 0);
  const code = lines.map((line) => line.text).join('\n');

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
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          opacity: intro,
          transform: `translateY(${interpolate(intro, [0, 1], [24, 0])}px)`,
        }}
      >
        <div>
          <div style={{fontSize: 20, fontWeight: 700, letterSpacing: 3, color: '#8ea8ff'}}>{eyebrow}</div>
          <div style={{fontSize: 48, fontWeight: 800, letterSpacing: -1.4, marginTop: 8}}>{title}</div>
        </div>
        <div style={{...card, borderRadius: 999, padding: '14px 22px', fontSize: 20, color: '#cbd5e1'}}>
          {command}
        </div>
      </div>

      <div style={{display: 'flex', gap: 34, marginTop: 42, height: 666}}>
        <div style={{...card, borderRadius: 26, width: 1048, overflow: 'hidden'}}>
          <div
            style={{
              height: 52,
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '0 22px',
              borderBottom: '1px solid rgba(255,255,255,0.08)',
              color: '#94a3b8',
              fontSize: 17,
            }}
          >
            <span style={{width: 12, height: 12, borderRadius: 99, background: '#ff5f57'}} />
            <span style={{width: 12, height: 12, borderRadius: 99, background: '#febc2e'}} />
            <span style={{width: 12, height: 12, borderRadius: 99, background: '#28c840'}} />
            <span style={{marginLeft: 16}}>{filename}</span>
          </div>
          <Highlight theme={themes.nightOwl} code={code} language={language}>
            {({tokens, getLineProps, getTokenProps}) => (
              <pre
                style={{
                  margin: 0,
                  padding: '18px 0',
                  fontFamily: mono,
                  fontSize: 20,
                  lineHeight: 1.43,
                  background: 'transparent',
                }}
              >
                {tokens.map((line, index) => {
                  const meta = lines[index];
                  const opacity = interpolate(frame, [meta.start, meta.start + 12], [0, 1], clamp);
                  const isActive = meta.stage === current;
                  const lineProps = getLineProps({line});
                  return (
                    <div
                      key={index}
                      {...lineProps}
                      style={{
                        ...lineProps.style,
                        display: 'flex',
                        opacity,
                        transform: `translateX(${interpolate(opacity, [0, 1], [-18, 0])}px)`,
                        background: isActive ? 'rgba(124,156,255,0.105)' : 'transparent',
                        borderLeft: isActive ? `4px solid ${stages[current].accent}` : '4px solid transparent',
                        paddingRight: 28,
                      }}
                    >
                      <span style={{width: 68, textAlign: 'right', paddingRight: 24, color: 'rgba(148,163,184,0.45)'}}>
                        {index + 1}
                      </span>
                      <span>
                        {language === 'bash' ? (
                          <BashLine text={meta.text} />
                        ) : (
                          line.map((token, key) => <span key={key} {...getTokenProps({token})} />)
                        )}
                      </span>
                    </div>
                  );
                })}
              </pre>
            )}
          </Highlight>
        </div>

        <div style={{display: 'flex', flexDirection: 'column', gap: 16, width: 390}}>
          {stages.map((stage, index) => {
            const progress = interpolate(frame, [stage.start, stage.end], [0, 1], clamp);
            const isActive = current === index;
            const isDone = frame > stage.end;
            return (
              <div
                key={stage.title}
                style={{
                  ...card,
                  borderRadius: 22,
                  padding: '19px 24px',
                  borderColor: isActive ? `${stage.accent}aa` : 'rgba(255,255,255,0.11)',
                  transform: `scale(${isActive ? 1.025 : 1})`,
                }}
              >
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <div style={{color: stage.accent, fontFamily: mono, fontSize: 17, fontWeight: 700}}>{stage.number}</div>
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 99,
                      display: 'grid',
                      placeItems: 'center',
                      background: isDone ? stage.accent : 'rgba(255,255,255,0.08)',
                      color: '#07111f',
                      fontWeight: 900,
                    }}
                  >
                    {isDone ? '✓' : ''}
                  </div>
                </div>
                <div style={{fontSize: 29, fontWeight: 800, marginTop: 9}}>{stage.title}</div>
                <div style={{fontSize: 18, color: '#aab6ca', marginTop: 5}}>{stage.detail}</div>
                <div style={{height: 6, borderRadius: 99, background: 'rgba(255,255,255,0.08)', marginTop: 16}}>
                  <div
                    style={{
                      width: `${progress * 100}%`,
                      height: '100%',
                      borderRadius: 99,
                      background: stage.accent,
                      boxShadow: `0 0 24px ${stage.accent}`,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
