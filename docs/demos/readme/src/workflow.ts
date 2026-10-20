import type {CodeLine, Stage} from './WorkflowDemo';

export const stages: Stage[] = [
  {
    number: '01',
    title: 'Train',
    detail: 'LeRobot dataset → ACT policy',
    start: 48,
    end: 142,
    accent: '#7c9cff',
  },
  {
    number: '02',
    title: 'Benchmark',
    detail: 'Evaluate in PushT simulation',
    start: 154,
    end: 244,
    accent: '#c084fc',
  },
  {
    number: '03',
    title: 'Export',
    detail: 'Optimized OpenVINO artifact',
    start: 256,
    end: 334,
    accent: '#21d4a7',
  },
  {
    number: '04',
    title: 'Deploy',
    detail: 'Run with Physical AI Runtime',
    start: 346,
    end: 430,
    accent: '#ffb86b',
  },
];

export const apiLines: CodeLine[] = [
  {text: 'from physicalai.benchmark.gyms import PushTBenchmark', start: 18},
  {text: 'from physicalai.data import LeRobotDataModule', start: 24},
  {text: 'from physicalai.inference import InferenceModel', start: 30},
  {text: 'from physicalai.policies import ACT', start: 36},
  {text: 'from physicalai.train import Trainer', start: 42},
  {text: '', start: 0},
  {text: 'datamodule = LeRobotDataModule(repo_id="lerobot/pusht", num_workers=0)', start: 58, stage: 0},
  {text: 'policy = ACT()', start: 70, stage: 0},
  {text: 'trainer = Trainer(max_epochs=100)', start: 94, stage: 0},
  {text: 'trainer.fit(model=policy, datamodule=datamodule)', start: 106, stage: 0},
  {text: '', start: 0},
  {text: 'benchmark = PushTBenchmark(num_episodes=50)', start: 164, stage: 1},
  {text: 'results = benchmark.evaluate(policy)', start: 184, stage: 1},
  {text: '', start: 0},
  {text: 'policy.export("./exports/act", backend="openvino")', start: 266, stage: 2},
  {text: '', start: 0},
  {text: 'runtime_policy = InferenceModel("./exports/act")', start: 356, stage: 3},
];
