import type {CSSProperties} from 'react';
import {AbsoluteFill} from 'remotion';

export type FeatureCard = {
  title: string;
  detail: string;
  meta: string;
  accent: string;
};

export type FeatureGalleryProps = {
  eyebrow: string;
  title: string;
  chip: string;
  cards: FeatureCard[];
  columns: number;
};

const font = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const mono = '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace';

const card: CSSProperties = {
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(15, 20, 35, 0.72)',
  boxShadow: '0 24px 70px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06)',
};

export const FeatureGallery = ({eyebrow, title, chip, cards, columns}: FeatureGalleryProps) => {
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
          <div style={{fontSize: 20, fontWeight: 700, letterSpacing: 3, color: '#8ea8ff'}}>{eyebrow}</div>
          <div style={{fontSize: 48, fontWeight: 800, letterSpacing: -1.4, marginTop: 8}}>{title}</div>
        </div>
        <div style={{...card, borderRadius: 999, padding: '14px 22px', fontFamily: mono, fontSize: 20, color: '#cbd5e1'}}>
          {chip}
        </div>
      </div>
      <div style={{display: 'grid', gridTemplateColumns: `repeat(${columns}, 1fr)`, gap: 20, marginTop: 44}}>
        {cards.map((item) => (
          <div key={item.title} style={{...card, borderRadius: 22, padding: '27px 28px', minHeight: 202}}>
            <div style={{height: 5, width: 72, borderRadius: 99, background: item.accent, boxShadow: `0 0 24px ${item.accent}`}} />
            <div style={{fontSize: 32, fontWeight: 800, marginTop: 24}}>{item.title}</div>
            <div style={{fontSize: 20, color: '#aab6ca', marginTop: 10, lineHeight: 1.35}}>{item.detail}</div>
            <div style={{fontFamily: mono, fontSize: 16, color: item.accent, marginTop: 22}}>{item.meta}</div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
