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
import { Back, Pause, Play } from '@geti-ui/ui/icons';

import { ErrorMessage } from '../../../components/error-page/error-page';
import { useProjectId } from '../../../features/projects/use-project';
import { useRobotControl } from '../../../features/robots/robot-control-provider';
import { RobotControlView } from '../../../features/robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { paths } from '../../../router';

interface InferenceViewerProps {
    tasks: string[];
}

export const InferenceViewer = ({ tasks }: InferenceViewerProps) => {
    const { project_id } = useProjectId();

    const [task, setTask] = useState<string>(tasks[0] ?? '');

    const { model, readyForInference, state, startTask, stopTask } = useRobotControl();

    if (state.error) {
        return <ErrorMessage message={'An error occurred during inference setup'} />;
    }

    if (!readyForInference) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.model_loaded ? 'positive' : 'yellow'}>Model</StatusLight>
                    <StatusLight variant={state.environment_loaded ? 'positive' : 'yellow'}>Environment</StatusLight>
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
                        {state.follower_source === 'model' ? (
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
                <RobotControlView />
            </Flex>
        </RobotModelsProvider>
    );
};
