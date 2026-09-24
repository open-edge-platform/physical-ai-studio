import type {CSSProperties} from 'react';
import {AbsoluteFill} from 'remotion';

type Policy = {
  name: string;
  className: string;
  detail: string;
  accent: string;
};

const policies: Policy[] = [
  {name: 'ACT', className: 'ACT', detail: 'Action chunking transformer', accent: '#7c9cff'},
  {name: 'Pi0.5', className: 'Pi05', detail: 'Open-world VLA', accent: '#c084fc'},
  {name: 'SmolVLA', className: 'SmolVLA', detail: 'Lightweight VLA', accent: '#21d4a7'},
  {name: 'MolmoAct2', className: 'MolmoAct2', detail: 'Action reasoning model', accent: '#ffb86b'},
  {name: 'RLDX-1', className: 'Rldx1', detail: 'Multi-embodiment VLA', accent: '#f472b6'},
  {name: 'XR0', className: 'XR0', detail: 'Xiaomi Robotics-0 VLA', accent: '#38bdf8'},
];

const font = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const mono = '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace';

const card: CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(15, 20, 35, 0.72)',
  boxShadow: '0 24px 70px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06)',
};

export const PolicyGallery = () => {
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
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end'}}>
        <div>
          <div style={{fontSize: 20, fontWeight: 700, letterSpacing: 3, color: '#8ea8ff'}}>NATIVE POLICIES</div>
          <div style={{fontSize: 48, fontWeight: 800, letterSpacing: -1.4, marginTop: 8}}>One API, multiple policies</div>
        </div>
        <div style={{...card, borderRadius: 999, padding: '14px 22px', fontFamily: mono, fontSize: 20, color: '#cbd5e1'}}>
          from physicalai.policies import &lt;Policy&gt;
        </div>
      </div>

      <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginTop: 44}}>
        {policies.map((policy) => (
          <div key={policy.name} style={{...card, borderRadius: 22, padding: '25px 28px', minHeight: 164}}>
            <div style={{fontSize: 32, fontWeight: 800}}>{policy.name}</div>
            <div style={{fontSize: 19, color: '#aab6ca', marginTop: 8}}>{policy.detail}</div>
            <div
              style={{
                marginTop: 22,
                padding: '11px 14px',
                borderRadius: 12,
                background: 'rgba(0,0,0,0.22)',
                fontFamily: mono,
                fontSize: 18,
                color: '#d6deeb',
              }}
            >
              <span style={{color: '#d6deeb'}}>policy</span> = <span style={{color: policy.accent}}>{policy.className}</span>(...)
            </div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
