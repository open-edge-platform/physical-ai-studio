import {Composition} from 'remotion';
import {FeatureGallery} from './FeatureGallery';
import {GuiDemo} from './GuiDemo';
import {guiDuration} from './gui';
import {PolicyGallery} from './PolicyGallery';
import {WorkflowDemo} from './WorkflowDemo';
import {apiLines, cliLines, stages} from './workflow';
import {buildCards, documentationCards, installLines, installStages} from './sections';

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
          command: 'pip install physicalai-train',
          filename: 'train_benchmark_export_deploy.py',
          language: 'python' as const,
          lines: apiLines,
          stages,
        }}
      />
      <Composition
        id="CliDemo"
        component={WorkflowDemo}
        durationInFrames={500}
        fps={30}
        width={1600}
        height={900}
        defaultProps={{
          eyebrow: 'CLI',
          title: 'The same workflow from the terminal',
          command: 'physicalai --help',
          filename: 'terminal',
          language: 'bash' as const,
          lines: cliLines,
          stages,
        }}
      />
      <Composition id="GuiDemo" component={GuiDemo} durationInFrames={guiDuration} fps={30} width={1600} height={900} />
      <Composition
        id="InstallDemo"
        component={WorkflowDemo}
        durationInFrames={360}
        fps={30}
        width={1600}
        height={900}
        defaultProps={{
          eyebrow: 'INSTALL',
          title: 'Choose your starting point',
          command: 'Python 3.12+',
          filename: 'terminal',
          language: 'bash' as const,
          lines: installLines,
          stages: installStages,
        }}
      />
      <Composition
        id="BuildGallery"
        component={FeatureGallery}
        durationInFrames={1}
        fps={30}
        width={1600}
        height={520}
        defaultProps={{
          eyebrow: 'WHAT YOU CAN BUILD',
          title: 'From policy training to deployment',
          chip: 'physicalai',
          cards: buildCards,
          columns: 4,
        }}
      />
      <Composition
        id="DocumentationGallery"
        component={FeatureGallery}
        durationInFrames={1}
        fps={30}
        width={1600}
        height={520}
        defaultProps={{
          eyebrow: 'DOCUMENTATION',
          title: 'Go deeper when you need it',
          chip: 'docs',
          cards: documentationCards,
          columns: 3,
        }}
      />
      <Composition id="PolicyGallery" component={PolicyGallery} durationInFrames={1} fps={30} width={1600} height={640} />
    </>
  );
};
