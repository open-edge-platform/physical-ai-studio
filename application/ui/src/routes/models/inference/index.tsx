import { Suspense, useEffect, useRef, useState } from 'react';

import { Button, ButtonGroup, ComboBox, Flex, Heading, Item, Link, ProgressCircle, ToastQueue } from '@geti/ui';
import { Back, Pause, Play } from '@geti/ui/icons';
import { CameraView } from '../../datasets/camera-view';

import { SchemaEnvironmentWithRelations, SchemaInferenceConfig } from '../../../api/openapi-spec';
import { useInferenceParams } from './use-inference-params';
import { $api } from '../../../api/client';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { paths } from '../../../router';
import { Observation, useInference2 } from './use-inference2';
import { RobotViewer } from '../../../features/robots/controller/robot-viewer';



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

export const Index = () => {
    const { project_id, model_id, backend } = useInferenceParams();
    const { observation, readyForInference, state, startTask, loadEnvironment, loadModel, isConnected } = useInference2(ToastQueue.negative);

    const { data: model } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { query: { uuid: model_id } },
    });

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: { path: { dataset_id: model.dataset_id } }
    });

    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } }
    });

    const { data: tasks } = $api.useSuspenseQuery('get', '/api/models/{model_id}/tasks', {
        params: { query: { uuid: model_id } },
    });

    const { data: initialEnvironment } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments/{environment_id}', {
        params: { path: { project_id, environment_id: dataset.environment_id } }
    });


    const [environment, setEnvironment] = useState<SchemaEnvironmentWithRelations>(initialEnvironment);
    const [task, setTask] = useState<string>(tasks[0] ?? ''); // TODO: Add default task on dataset schema

    useEffect(() => {
        if (loadEnvironment.isIdle && isConnected) {
            console.log('environment changed...')
            loadEnvironment.mutate(environment);
        }
    }, [environment, isConnected])

    useEffect(() => {
        if (loadModel.isIdle && isConnected) {
            console.log('model changed...')
            loadModel.mutate({model, backend});
        }
    }, [model, backend, isConnected])

    const visualisation_source = getVisualisationSourceFromObservation(observation.current);
    const robot = environment?.robots?.at(0)?.robot;

    const onStart = () => {
        startTask(0); // TODO: Deal with indices or not? Probably not
    };

    console.log(observation.current)

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
                        {(environment?.cameras ?? []).map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        {robot && <RobotViewer
                            featureValues={Object.values(visualisation_source)}
                            featureNames={Object.keys(visualisation_source)}
                            robot={robot}
                        />}
                    </Flex>
                </Flex>
            </Flex>
        </RobotModelsProvider>
    );
};
