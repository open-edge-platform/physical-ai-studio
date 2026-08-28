import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Flex,
    Heading,
    Item,
    Link,
    ProgressCircle,
    StatusLight,
    Text,
} from '@geti-ui/ui';
import { Back, DownloadIcon, Pause, Play } from '@geti-ui/ui/icons';

import { paths } from '../../../router';
import { useProjectId } from '../../projects/use-project';
import { RobotControlView } from '../../robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { runtimeExportUrl } from '../runtime-export';

interface InferenceViewerProps {
    tasks: string[];
}

export const InferenceViewer = ({ tasks }: InferenceViewerProps) => {
    const { project_id } = useProjectId();

    const [task, setTask] = useState<string>(tasks[0] ?? '');

    const { model, readyForInference, state, startTask, stopTask, environment, observation, inferenceDevice } =
        useRuntimeSession();

    const exportUrl =
        model?.id !== undefined && inferenceDevice !== undefined
            ? runtimeExportUrl({
                  modelId: model.id,
                  environmentId: environment.id,
                  backend: inferenceDevice.backend,
                  device: inferenceDevice.device,
                  task,
              })
            : undefined;

    if (!readyForInference) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.model_loaded ? 'positive' : 'yellow'}>Model</StatusLight>
                    <StatusLight variant={state.connected ? 'positive' : 'yellow'}>Environment</StatusLight>
                </Flex>
                <Button variant={'secondary'} href={paths.project.models.index({ project_id })}>
                    Cancel
                </Button>
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <Flex flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex alignItems={'center'} gap='size-100' height='size-400' margin='size-200'>
                    <Link aria-label='Rewind' href={paths.project.models.index({ project_id })}>
                        <Back fill='white' />
                    </Link>
                    <Heading>Model Run {model?.name}</Heading>
                    <ComboBox flex isRequired allowsCustomValue={false} inputValue={task} onInputChange={setTask}>
                        {tasks.map((taskText, index) => (
                            <Item key={index}>{taskText}</Item>
                        ))}
                    </ComboBox>
                    <ButtonGroup>
                        {exportUrl !== undefined && (
                            <Button
                                href={exportUrl}
                                aria-label='Download runtime export'
                                variant='secondary'
                                target='_blank'
                                rel='noopener noreferrer'
                            >
                                <DownloadIcon />
                                Runtime export
                            </Button>
                        )}
                        {state.follower_source === 'policy' ? (
                            <Button variant='primary' isPending={stopTask.isPending} onPress={() => stopTask.mutate()}>
                                <Pause fill='white' />
                                Stop
                            </Button>
                        ) : (
                            <Button
                                variant='primary'
                                isPending={startTask.isPending}
                                onPress={() => startTask.mutate(task)}
                            >
                                <Play fill='white' />
                                Play
                            </Button>
                        )}
                    </ButtonGroup>
                </Flex>
                <RobotControlView environment={environment} isReady={state.connected} joints={observation} />
            </Flex>
        </RobotModelsProvider>
    );
};
