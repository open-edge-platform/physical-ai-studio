import type {FeatureCard} from './FeatureGallery';
import type {CodeLine, Stage} from './WorkflowDemo';

export const installStages: Stage[] = [
  {
    number: '01',
    title: 'Library',
    detail: 'Python API and CLI',
    start: 40,
    end: 100,
    accent: '#7c9cff',
  },
  {
    number: '02',
    title: 'Studio',
    detail: 'Run the app with Docker',
    start: 112,
    end: 235,
    accent: '#21d4a7',
  },
  {
    number: '03',
    title: 'Open',
    detail: 'Start in your browser',
    start: 247,
    end: 305,
    accent: '#ffb86b',
  },
];

export const installLines: CodeLine[] = [
  {text: '# Python API and CLI', start: 40, stage: 0},
  {text: 'pip install physicalai-train', start: 52, stage: 0},
  {text: '', start: 0},
  {text: '# Studio UI with Docker', start: 112, stage: 1},
  {text: 'git clone https://github.com/open-edge-platform/physical-ai-studio.git', start: 124, stage: 1},
  {text: 'cd physical-ai-studio/application/docker', start: 140, stage: 1},
  {text: 'cp .env.example .env', start: 156, stage: 1},
  {text: './setup-devices.sh --cpu  # or --xpu, --cuda', start: 172, stage: 1},
  {text: 'docker compose up -d', start: 188, stage: 1},
  {text: '', start: 0},
  {text: '# Open http://localhost:7860', start: 247, stage: 2},
];

export const buildCards: FeatureCard[] = [
  {
    title: 'Policies',
    detail: 'Train native ACT, Pi0.5, SmolVLA, MolmoAct2, RLDX-1, and XR0 policies.',
    meta: 'physicalai.policies',
    accent: '#7c9cff',
  },
  {
    title: 'Benchmarks',
    detail: 'Evaluate policies in LIBERO, PushT, and RoboCasa simulations.',
    meta: 'physicalai benchmark',
    accent: '#c084fc',
  },
  {
    title: 'Deployment',
    detail: 'Export OpenVINO, ONNX, Torch, and ExecuTorch artifacts.',
    meta: 'physicalai export',
    accent: '#21d4a7',
  },
  {
    title: 'Training at scale',
    detail: 'Use Lightning training with distributed and mixed-precision support.',
    meta: 'physicalai fit',
    accent: '#ffb86b',
  },
];

export const documentationCards: FeatureCard[] = [
  {
    title: 'Library',
    detail: 'Python API, CLI, training, benchmarking, and export guides.',
    meta: 'library/README.md',
    accent: '#7c9cff',
  },
  {
    title: 'Application',
    detail: 'Studio installation, setup, data collection, and workflows.',
    meta: 'application/README.md',
    accent: '#21d4a7',
  },
  {
    title: 'Contributing',
    detail: 'Development setup, standards, and contribution guidelines.',
    meta: 'CONTRIBUTING.md',
    accent: '#ffb86b',
  },
];
