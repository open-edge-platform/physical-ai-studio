import {Composition} from 'remotion';
import {GuiDemo} from './GuiDemo';
import {guiDuration} from './gui';
import {WorkflowDemo} from './WorkflowDemo';
import {apiLines, stages} from './workflow';

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="ApiDemo"
        component={WorkflowDemo}
        durationInFrames={500}
        fps={30}
        width={1600}
        height={900}
        defaultProps={{
          eyebrow: 'PYTHON API',
          title: 'From dataset to deployable policy',
          command: "pip install 'physicalai-train[cpu]'",
          filename: 'train_benchmark_export_deploy.py',
          language: 'python' as const,
          lines: apiLines,
          stages,
        }}
      />
      <Composition id="GuiDemo" component={GuiDemo} durationInFrames={guiDuration} fps={30} width={1600} height={900} />
    </>
  );
};
