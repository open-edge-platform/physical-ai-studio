import { useState } from 'react';

import { Button, ButtonGroup, ComboBox, Flex, Heading, Item, Link, ProgressCircle, ToastQueue } from '@geti/ui';
import { Back, Pause, Play } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { SchemaInferenceConfig } from '../../../api/openapi-spec';
import { ErrorMessage } from '../../../components/error-page/error-page';
import { RobotViewer } from '../../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { paths } from '../../../router';
import { CameraView } from '../../datasets/camera-view';
import { Observation, useInference } from './use-inference';
import { useInferenceParams } from './use-inference-params';

interface InferenceViewerProps {
    config: SchemaInferenceConfig;
}

const getVisualisationSourceFromObservation = (observation: Observation | undefined): { [joint: string]: number } => {
    if (observation === undefined) {
        return {};
    }
    if (observation['actions'] !== null) {
        return observation['actions'];
    } else {
        return observation['state'];
    }
};

export const InferenceViewer = ({ config }: InferenceViewerProps) => {
    const { project_id, model_id } = useInferenceParams();

    const { data: model } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { query: { uuid: model_id } },
    });
    const { data: tasks } = $api.useSuspenseQuery('get', '/api/models/{model_id}/tasks', {
        params: { query: { uuid: model_id } },
    });
    const [task, setTask] = useState<string>(tasks[0] ?? '');

    const { startTask, stop, state, observation } = useInference(config, ToastQueue.negative);

    const robots = (config.environment.robots ?? []).map(({ robot }) => robot);

    const visualisation_source = getVisualisationSourceFromObservation(observation.current);

    if (state.error) {
        return <ErrorMessage message={'An error occurred during inference setup'} />;
    }

    if (!state.initialized) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Heading>Initializing</Heading>
                <ProgressCircle isIndeterminate />
            </Flex>
        );
    }

    const onStart = () => {
        startTask(tasks.indexOf(task));
    };

    return (
        <RobotModelsProvider>
            <Flex flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex alignItems={'center'} gap='size-100' height='size-400' margin='size-200'>
                    <Link aria-label='Rewind' href={paths.project.models.index({ project_id })}>
                        <Back fill='white' />
                    </Link>
                    <Heading>Model Run {model.name}</Heading>
                    <ComboBox flex isRequired allowsCustomValue={false} inputValue={task} onInputChange={setTask}>
                        {tasks.map((taskText, index) => (
                            <Item key={index}>{taskText}</Item>
                        ))}
                    </ComboBox>
                    <ButtonGroup>
                        {state.is_running ? (
                            <Button variant='primary' onPress={stop}>
                                <Pause fill='white' />
                                Stop
                            </Button>
                        ) : (
                            <Button variant='primary' onPress={onStart}>
                                <Play fill='white' />
                                Play
                            </Button>
                        )}
                    </ButtonGroup>
                </Flex>
                <Flex direction={'row'} flex gap={'size-100'} margin='size-200'>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {config.environment.cameras!.map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer
                            featureValues={Object.values(visualisation_source)}
                            featureNames={Object.keys(visualisation_source)}
                            robot={robots[0]}
                        />
                    </Flex>
                </Flex>
            </Flex>
        </RobotModelsProvider>
    );
};
