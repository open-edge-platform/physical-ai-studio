import type {DemoId} from './config.ts';
import {GuiDemo} from './GuiDemo';
import {WorkflowDemo} from './WorkflowDemo';
import {apiLines, stages} from './workflow';

export const Demo = ({id, frame}: {id: DemoId; frame: number}) =>
  id === 'gui' ? (
    <GuiDemo frame={frame} />
  ) : (
    <WorkflowDemo
      frame={frame}
      eyebrow="PYTHON API"
      title="From dataset to deployable policy"
      command="pip install 'physicalai-train[cpu]'"
      filename="train_benchmark_export_deploy.py"
      language="python"
      lines={apiLines}
      stages={stages}
    />
  );
